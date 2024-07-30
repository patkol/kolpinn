# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Optional, Callable, Dict, Any
from collections.abc import Sequence
import warnings
import copy
import torch
from torch import nn

from . import mathematics
from . import grids
from .grids import Grid
from . import quantities
from .quantities import QuantityDict, combine_quantity, might_depend_on


class Model:
    """
    Model.apply(q: QuantityDict)
    -> torch.Tensor of type `output_dtype`
    The model is parametrized by `parameters` of type `model_dtype`.
    """

    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        model_dtype,
        output_dtype,
    ):
        self.parameters = parameters
        self.model_dtype = model_dtype
        self.output_dtype = output_dtype
        self.check()

    def apply(self, q: QuantityDict) -> torch.Tensor:
        raise Exception("`Model` is an ABC")

    def set_requires_grad(self, requires_grad: bool):
        _set_requires_grad(requires_grad, self.parameters)

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def load(self, path: str):
        self.replace_parameters(torch.load(path))

    def save(self, path: str):
        torch.save(self.parameters, path)

    def replace_parameters(self, new_parameters: Sequence[torch.Tensor]):
        _replace_parameters(self.parameters, new_parameters)

    def check(self):
        for parameter in self.parameters:
            assert parameter.dtype is self.model_dtype


class ConstModel(Model):
    def __init__(self, value, *, model_dtype, output_dtype=None):
        parameters = [torch.tensor(value, dtype=model_dtype)]
        super().__init__(
            parameters,
            model_dtype=model_dtype,
            output_dtype=output_dtype,
        )

    def apply(self, q: QuantityDict):
        tensor = self.parameters[0].reshape([1] * q.grid.n_dim)
        return tensor.to(self.output_dtype)


class FunctionModel(Model):
    def __init__(self, function, *, output_dtype=None, **kwargs):
        """
        function(q: QuantityDict, **kwargs) -> torch.Tensor
        """
        self.function = function
        self.kwargs = kwargs
        super().__init__([], model_dtype=None, output_dtype=output_dtype)

    def apply(self, q: QuantityDict):
        tensor = self.function(q, **(self.kwargs))
        return tensor.to(self.output_dtype)


def get_model(value, *, model_dtype=None, output_dtype=None, **kwargs):
    """
    Returns a FunctionModel if `value` is callable and a ConstModel otherwise.
    """

    if callable(value):
        return FunctionModel(value, output_dtype=output_dtype, **kwargs)

    assert model_dtype is not None

    return ConstModel(
        value,
        model_dtype=model_dtype,
        output_dtype=output_dtype,
    )


class SimpleNetwork(nn.Module):
    def __init__(
        self,
        activation_function,
        *,
        n_inputs: int,
        n_outputs: int,
        n_neurons_per_hidden_layer: int,
        n_hidden_layers: int,
        dtype,
    ):
        assert n_hidden_layers >= 1, n_hidden_layers

        super().__init__()
        self.first_linear = nn.Linear(
            n_inputs,
            n_neurons_per_hidden_layer,
            dtype=dtype,
        )
        self.hidden_linears = nn.ModuleList(
            [
                nn.Linear(
                    n_neurons_per_hidden_layer, n_neurons_per_hidden_layer, dtype=dtype
                )
                for i in range(n_hidden_layers - 1)
            ],
        )
        self.last_linear = nn.Linear(
            n_neurons_per_hidden_layer,
            n_outputs,
            dtype=dtype,
        )
        self.activation_function = activation_function

        # He initialization
        nn.init.kaiming_normal_(self.first_linear.weight)
        for hidden_linear in self.hidden_linears:
            nn.init.kaiming_normal_(hidden_linear.weight)
        nn.init.kaiming_normal_(self.last_linear.weight)

    def forward(self, inputs):
        outputs = self.first_linear(inputs)
        for hidden_linear in self.hidden_linears:
            outputs = self.activation_function(outputs)
            outputs = hidden_linear(outputs)
        outputs = self.activation_function(outputs)
        outputs = self.last_linear(outputs)

        return outputs


class SimpleNNModel(Model):
    def __init__(
        self,
        inputs_labels: Sequence[str],
        activation_function,
        *,
        n_neurons_per_hidden_layer: int,
        n_hidden_layers: int,
        model_dtype,
        output_dtype,
        device: str,
        complex_polar: Optional[bool] = None,
        r_transformation=torch.nn.Softplus(),
        phi_transformation=lambda x: x,
    ):
        """
        inputs_labels: labels of the quantities that will be input
            to the network
        complex_polar: If output_dtype is complex, whether the two outputs
            of the NN should be interpreted as
            (r,phi) (true) or (Re, Im) (false)
        r/phi_transformation: Applied to r/phi if complex_polar
        """

        if complex_polar is None:
            complex_polar = False

        self.inputs_labels = inputs_labels
        self.n_inputs = len(inputs_labels)
        self.complex_output = output_dtype in (torch.complex64, torch.complex128)
        self.n_outputs = 2 if self.complex_output else 1
        self.network = SimpleNetwork(
            activation_function,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_neurons_per_hidden_layer=n_neurons_per_hidden_layer,
            n_hidden_layers=n_hidden_layers,
            dtype=model_dtype,
        ).to(device)
        self.complex_polar = complex_polar
        self.r_transformation = r_transformation
        self.phi_transformation = phi_transformation
        super().__init__(
            list(self.network.parameters()),
            model_dtype=model_dtype,
            output_dtype=output_dtype,
        )

    def _assemble(self, tensor: torch.Tensor):
        """
        Input: Real ... x 2 tensor
        Output: Complex ... tensor

        If self.complex_output is false, the input is directly returned and
        can have a different shape.
        """

        assert tensor.dtype in (torch.float32, torch.float64)

        if not self.complex_output:
            return tensor

        assert tensor.size(-1) == 2

        if self.complex_polar:
            r = self.r_transformation(tensor[..., 0])
            phi = self.phi_transformation(tensor[..., 1])
            return torch.polar(r, phi)

        # Interpret the two components as the real and imaginary part
        return torch.view_as_complex(tensor)

    def apply(self, q: QuantityDict):
        """
        Returns the evaluated quantity represented by the NN.
        """

        # inputs_tensor[gridpoint, input quantity]
        inputs_tensor = torch.zeros(
            (q.grid.n_points, self.n_inputs),
            dtype=self.model_dtype,
        )
        for i, label in enumerate(self.inputs_labels):
            inputs_tensor[:, i] = quantities.expand_all_dims(
                q[label],
                q.grid,
            ).flatten()

        output = self.network(inputs_tensor)
        output = self._assemble(output)
        output = output.reshape(q.grid.shape)
        output = output.to(self.output_dtype)

        return output

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()


class TransformedModel(Model):
    """
    A parent `Model` where the input quantities `q` can be transformed before
    they are handed to the child.
    The output can then be transformed as well.
    """

    def __init__(
        self,
        child_model: Model,
        *,
        input_transformations: Optional[Dict[str, Callable]] = None,
        output_transformation: Optional[Callable] = None,
        output_dtype=None,
    ):
        """transformation(quantity: Quantity, q: QuantityDict)"""
        if input_transformations is None:
            input_transformations = {}
        if output_transformation is None:

            def output_transformation(quantity, q):
                return quantity

        if output_dtype is None:
            output_dtype = child_model.output_dtype

        self.child_model = child_model
        self.input_transformations = input_transformations
        self.output_transformation = output_transformation
        super().__init__(
            child_model.parameters,
            model_dtype=child_model.model_dtype,
            output_dtype=output_dtype,
        )

    def apply(self, q: QuantityDict):
        transformed_q = copy.copy(q)
        for label, transformation in self.input_transformations.items():
            transformed_q.overwrite(label, transformation(q[label], q))
        child_output = self.child_model.apply(transformed_q)
        transformed_output = self.output_transformation(child_output, q)
        transformed_output = transformed_output.to(self.output_dtype)

        return transformed_output

    def set_train(self):
        self.child_model.set_train()

    def set_eval(self):
        self.child_model.set_eval()


class MultiModel:
    def __init__(
        self,
        qs_trafo: Callable,
        name: str,
        *,
        models: Optional[list[Model]] = None,
        parameters_in: Optional[list[torch.Tensor]] = None,
        networks_in: Optional[list[torch.nn.Module]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        qs_trafo should accept, modify and return qs.
        The MultiModel allows operations that depend on multiple grids.
        The `kwargs` will be provided to `qs_trafo`
        """

        if models is None:
            models = []
        parameters = [] if parameters_in is None else copy.copy(parameters_in)
        networks = [] if networks_in is None else copy.copy(networks_in)
        if kwargs is None:
            kwargs = {}

        for model in models:
            parameters += model.parameters
            networks += _find_networks(model)
        parameters = mathematics.remove_duplicates(parameters)
        networks = mathematics.remove_duplicates(networks)

        self.qs_trafo = qs_trafo
        self.name = name
        self.models = models
        self.parameters = parameters
        self.networks = networks
        self.kwargs = kwargs

    def apply(self, qs: Dict[str, QuantityDict]):
        return self.qs_trafo(qs, **self.kwargs)

    def set_requires_grad(self, requires_grad: bool):
        _set_requires_grad(requires_grad, self.parameters)

    def set_train(self):
        for i in range(len(self.networks)):
            self.networks[i].train()

    def set_eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def load(self, path: str):
        self.replace_parameters(torch.load(path))

    def save(self, path: str):
        torch.save(self.parameters, path)

    def replace_parameters(self, new_parameters: Sequence[torch.Tensor]):
        _replace_parameters(self.parameters, new_parameters)


def get_multi_model(
    model: Model,
    model_name: str,
    grid_name: str,
    *,
    multi_model_name: Optional[str] = None,
):
    """
    Turn a model that does not depend on other grids (q -> q)
    into a MultiModel (qs -> qs).
    If the model has kwargs they can be accessed through `multi_model.kwargs`
    """

    if multi_model_name is None:
        multi_model_name = model_name

    def qs_trafo(qs: Dict[str, QuantityDict], **multi_model_kwargs):
        # Provide q_full if necessary
        if hasattr(model, "kwargs") and "q_full" in model.kwargs:
            qs_full = multi_model_kwargs["qs_full"]
            model.kwargs["q_full"] = qs_full[grid_name]
        else:
            assert len(multi_model_kwargs) == 0, multi_model_kwargs

        # Apply the model
        q = qs[grid_name]
        q[model_name] = model.apply(q)
        return qs

    multi_model_kwargs = None
    if hasattr(model, "kwargs") and "q_full" in model.kwargs:
        multi_model_kwargs = {
            "qs_full": None,
        }

    return MultiModel(
        qs_trafo,
        multi_model_name,
        models=[model],
        kwargs=multi_model_kwargs,
    )


def get_multi_models(
    models_dict: Dict[str, Model],
    grid_name: str,
    *,
    used_models_names: Optional[list[str]],
):
    if used_models_names is not None:
        models_dict = dict((name, models_dict[name]) for name in used_models_names)
    multi_models = [
        get_multi_model(model, model_name, grid_name)
        for model_name, model in models_dict.items()
    ]

    return multi_models


def get_combined_multi_model(
    model: Model,
    model_name: str,
    grid_names: list[str],
    *,
    combined_dimension_name: str,
    required_quantities_labels: list[str],
    multi_model_name: Optional[str] = None,
):
    """
    Like `get_multi_model`, but evaluating on a supergrid of all `grid_names`
    """
    if multi_model_name is None:
        multi_model_name = model_name

    def qs_trafo(qs: Dict[str, QuantityDict]):
        child_grids = dict((grid_name, qs[grid_name].grid) for grid_name in grid_names)
        supergrid = grids.Supergrid(
            child_grids,
            combined_dimension_name,
            copy_all=False,
        )
        q = QuantityDict(supergrid)
        for label in required_quantities_labels:
            q[label] = combine_quantity(
                [qs[child_name][label] for child_name in grid_names],
                list(supergrid.subgrids.values()),
                supergrid,
            )
        quantity = model.apply(q)
        for grid_name in grid_names:
            qs[grid_name][model_name] = quantities.restrict(
                quantity,
                supergrid.subgrids[grid_name],
            )

        return qs

    return MultiModel(
        qs_trafo,
        multi_model_name,
        models=[model],
    )


def _replace_parameters(
    old_parameters: Sequence[torch.Tensor],
    new_parameters: Sequence[torch.Tensor],
):
    assert len(old_parameters) == len(new_parameters)
    for old_parameter, new_parameter in zip(old_parameters, new_parameters):
        requires_grad = old_parameter.requires_grad
        old_parameter.requires_grad_(False)
        old_parameter[...] = new_parameter
        old_parameter.requires_grad_(requires_grad)

    return old_parameters


def _set_requires_grad(
    requires_grad: bool,
    parameters: Sequence[torch.Tensor],
):
    if requires_grad and len(parameters) == 0:
        warnings.warn("Tried to set `requires_grad` on a parameter-less model")
    for i in range(len(parameters)):
        parameters[i].requires_grad_(requires_grad)

    return parameters


def _find_networks(model):
    networks = []
    if hasattr(model, "network"):
        networks.append(model.network)
    if hasattr(model, "child_model"):
        networks += _find_networks(model.child_model)

    return networks


def add_coordinates(qs: Dict[str, QuantityDict]):
    for q in qs.values():
        grid = q.grid
        for label, dimension_values in grid.dimensions.items():
            assert label not in q, label
            q[label] = quantities.unsqueeze_to(
                grid,
                dimension_values,
                [label],
            )

    return qs


coordinates_model = MultiModel(add_coordinates, "coordinates")


def get_qs(
    grids_: Dict[str, Grid],
    models: list[MultiModel],
    quantities_requiring_grad: Dict[str, list[str]],
):
    """Get the non-extended qs that do not depend on trained parameters."""

    qs = dict((grid_name, QuantityDict(grid)) for grid_name, grid in grids_.items())
    coordinates_model.apply(qs)
    set_requires_grad_quantities(
        quantities_requiring_grad,
        qs,
        allow_missing_quantities=True,
    )

    for model in models:
        model.apply(qs)
    set_requires_grad_quantities(quantities_requiring_grad, qs)

    return qs


def set_requires_grad_quantities(
    quantities_dict: Dict[str, list[str]],
    qs: Dict[str, QuantityDict],
    *,
    allow_missing_quantities: bool = False,
):
    """
    Expand and require grad for the given quantities.
    quantities_dict[grid_name] = [quantity_name1, ...]
    """
    for grid_name, quantity_names in quantities_dict.items():
        q = qs[grid_name]
        for quantity_name in quantity_names:
            if quantity_name not in q:
                assert allow_missing_quantities, quantity_name
                continue

            if q[quantity_name].requires_grad:
                for label in q.grid.dimensions_labels:
                    assert might_depend_on(label, q[quantity_name], q.grid)
                continue

            quantity_with_grad = quantities.expand_all_dims(
                q[quantity_name],
                q.grid,
            )
            quantity_with_grad.requires_grad_(True)
            q.overwrite(quantity_name, quantity_with_grad)

    return qs


def set_requires_grad_models(
    requires_grad: bool,
    models: Sequence[MultiModel],
):
    """
    Set `requires_grad` on the parameters of the given models.
    """
    for model in models:
        model.set_requires_grad(requires_grad)
