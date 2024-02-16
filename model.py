from typing import Optional, Callable, Union, Iterable
import warnings
import copy
import torch
from torch import nn

from . import mathematics
from . import grid_quantities
from .grid_quantities import Grid, QuantityDict, combine_quantity, combine_quantities
from .batching import Batcher



class Model:
    """
    Model.apply(q: QuantityDict)
    -> torch.Tensor of type `output_dtype`, dict of intermediate quantities
    The model is parametrized by `parameters` of type `model_dtype`.
    """
    def __init__(
            self,
            parameters: Iterable[torch.Tensor],
            *,
            model_dtype,
            output_dtype,
        ):
        self.parameters = parameters
        self.model_dtype = model_dtype
        self.output_dtype = output_dtype
        self.check()

    def apply(self, q: QuantityDict, **kwargs):
        """
        Should return quantity, dict of intermediate quantities
        """
        raise Exception('`Model` is an ABC')

    def set_requires_grad(self, requires_grad: bool):
        if requires_grad and len(self.parameters) == 0:
            warnings.warn('Tried to set `requires_grad` on a parameter-less model')
        for parameter in self.parameters:
            parameter.requires_grad_(requires_grad)

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def load(self, path: str):
        self.replace_parameters(torch.load(path))

    def save(self, path: str):
        torch.save(self.parameters, path)

    def replace_parameters(self, new_parameters: Iterable[torch.Tensor]):
        assert len(self.parameters) == len(new_parameters)
        for parameter, new_parameter in zip(self.parameters, new_parameters):
            requires_grad = parameter.requires_grad
            parameter.requires_grad_(False)
            parameter[...] = new_parameter
            parameter.requires_grad_(requires_grad)

    def check(self):
        for parameter in self.parameters:
            assert parameter.dtype is self.model_dtype


class ConstModel(Model):
    def __init__(self, value, *, model_dtype, output_dtype):
        parameters = [torch.tensor(value, dtype=model_dtype)]
        super().__init__(
            parameters,
            model_dtype=model_dtype,
            output_dtype=output_dtype,
        )

    def apply(self, q: QuantityDict):
        return self.parameters[0].to(self.output_dtype), {}


class FunctionModel(Model):
    def __init__(self, function, *, output_dtype=None, **kwargs):
        """
        function(q: QuantityDict, **kwargs) -> torch.Tensor
        """
        self.function = function
        self.kwargs = kwargs
        super().__init__([], model_dtype=None, output_dtype=output_dtype)

    def apply(self, q: QuantityDict):
        return self.function(q, **(self.kwargs)).to(self.output_dtype), {}


def get_model(value, *, model_dtype, output_dtype, **kwargs):
    """
    Returns a FunctionModel if `value` is callable and a ConstModel otherwise.
    """

    if callable(value):
        return FunctionModel(value, output_dtype=output_dtype, **kwargs)

    return ConstModel(
               value,
               model_dtype = model_dtype,
               output_dtype = output_dtype,
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
            [nn.Linear(n_neurons_per_hidden_layer,
                       n_neurons_per_hidden_layer,
                       dtype=dtype)
             for i in range(n_hidden_layers-1)],
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
            inputs_labels: list,
            activation_function,
            *,
            n_neurons_per_hidden_layer: int,
            n_hidden_layers: int,
            model_dtype,
            output_dtype,
            device: str,
            complex_polar: Optional[bool] = None,
            r_transformation = torch.nn.Softplus(),
            phi_transformation = lambda x: x,
        ):
        """
        inputs_labels: labels of the quantities that will be input to the network
        complex_polar: If output_dtype is complex, whether the two outputs
            of the NN should be interpreted as (r,phi) (true) or (Re, Im) (false)
        r/phi_transformation: Applied to r/phi if complex_polar
        """

        self.inputs_labels = inputs_labels
        self.n_inputs = len(inputs_labels)
        self.complex_output = output_dtype in (torch.complex64, torch.complex128)
        self.n_outputs = 2 if self.complex_output else 1
        self.network = SimpleNetwork(
            activation_function,
            n_inputs = self.n_inputs,
            n_outputs = self.n_outputs,
            n_neurons_per_hidden_layer = n_neurons_per_hidden_layer,
            n_hidden_layers = n_hidden_layers,
            dtype = model_dtype,
        ).to(device)
        self.complex_polar = complex_polar
        self.r_transformation = r_transformation
        self.phi_transformation = phi_transformation
        super().__init__(
            list(self.network.parameters()),
            model_dtype = model_dtype,
            output_dtype = output_dtype,
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
            r = self.r_transformation(tensor[...,0])
            phi = self.phi_transformation(tensor[...,1])
            return torch.polar(r, phi)

        # Interpret the two components as the real and imaginary part
        return torch.view_as_complex(tensor)

    def apply(
            self,
            q: QuantityDict,
        ):
        """
        Returns output, intermediates
        where
            output is the evaluated quantity represented by the NN and
            intermediates is a dict containing all inputs and
                outputs of the NN after/before applying transformations.
        """

        # inputs_tensor[gridpoint, input quantity]
        inputs_tensor = torch.zeros(
            (q.grid.n_points, self.n_inputs),
            dtype = self.model_dtype,
        )
        for (i, label) in enumerate(self.inputs_labels):
            inputs_tensor[:,i] = grid_quantities.expand_all_dims(q[label], q.grid).flatten()

        nn_outputs_tensor = self.network(inputs_tensor)
        output = self._assemble(nn_outputs_tensor)
        output = output.reshape(q.grid.shape)
        output = output.to(self.output_dtype)

        intermediates = {}
        # OPTIM: Skip if not needed
        for output_index in range(nn_outputs_tensor.size(-1)):
            nn_output_tensor = nn_outputs_tensor[..., output_index]
            nn_output_tensor = nn_output_tensor.reshape(q.grid.shape)
            intermediates[f'nn_output{output_index}'] = nn_output_tensor

        return output, intermediates

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
            input_transformations: Optional[dict[str,Callable]] = None,
            output_transformation: Optional[Callable] = None,
            output_dtype = None,
        ):
        """transformation(quantity: Quantity, q: QuantityDict)"""
        if input_transformations is None:
            input_transformations = {}
        if output_transformation is None:
            output_transformation = lambda quantity, q: quantity
        if output_dtype is None:
            output_dtype = child_model.output_dtype

        self.child_model = child_model
        self.input_transformations = input_transformations
        self.output_transformation = output_transformation
        super().__init__(
            child_model.parameters,
            model_dtype = child_model.model_dtype,
            output_dtype = output_dtype,
        )

    def apply(self, q: QuantityDict, **kwargs):
        transformed_q = copy.copy(q)
        intermediates = {}
        for label, transformation in self.input_transformations.items():
            transformed_q[label] = transformation(q[label], q)
            intermediates[label + '_transformed'] = transformed_q[label]
        child_output, child_intermediates = self.child_model.apply(transformed_q, **kwargs)
        intermediates.update(child_intermediates)
        intermediates['untransformed_output'] = child_output
        transformed_output = self.output_transformation(child_output, q)
        transformed_output = transformed_output.to(self.output_dtype)

        return transformed_output, intermediates

    def set_train(self):
        self.child_model.set_train()

    def set_eval(self):
        self.child_model.set_eval()


def get_extended_q(
        q_in: QuantityDict,
        *,
        models: dict = None,
        models_require_grad: bool,
        quantities_requiring_grad_labels: list[str] = None,
        models_requiring_grad_labels: list[str] = None,
    ):
    """
    Get the quantities including the evaluated models.
    quantities_requiring_grad_labels are expanded and require grad.
    models_requiring_grad_labels require grad iff models_require_grad.
    """

    if models is None:
        models = {}
    if quantities_requiring_grad_labels is None:
        quantities_requiring_grad_labels = []
    if models_requiring_grad_labels is None:
        models_requiring_grad_labels = []

    q = copy.copy(q_in)

    for quantity_label in quantities_requiring_grad_labels:
        q[quantity_label] = grid_quantities.expand_all_dims(q[quantity_label], q.grid)
        q[quantity_label].requires_grad = True

    for model_label in models_requiring_grad_labels:
        models[model_label].set_requires_grad(models_require_grad)

    for model_name, model in models.items():
        output, intermediates = model.apply(q)
        assert not model_name in q, model_name
        q[model_name] = output
        # Overwriting existing intermediates with the same key
        q.update(intermediates)

    return q


def get_extended_q_batchwise(
        batcher: Batcher,
        *,
        models: dict,
        models_require_grad: bool,
        quantities_requiring_grad_labels: list[str] = None,
        models_requiring_grad_labels: list[str] = None,
    ):
    """
    Get the quantities including the evaluated models.
    """

    qs_batch = []
    for q_batch in batcher.get_all():
        qs_batch.append(get_extended_q(
            q_batch,
            models = models,
            models_require_grad = models_require_grad,
            quantities_requiring_grad_labels = quantities_requiring_grad_labels,
            models_requiring_grad_labels = models_requiring_grad_labels,
        ))

    return combine_quantities(qs_batch, batcher.grid_full)


def get_extended_qs(
        batchers: dict[str,Batcher],
        *,
        models_dict: dict[str,dict[str,Model]],
        models_require_grad: bool,
        quantities_requiring_grad_dict: dict[str,list[str]],
        models_requiring_grad_dict: dict[str,list[str]],
        full_grid: bool,
    ):

    qs = {}
    q_function = get_extended_q_batchwise if full_grid else get_extended_q
    for batcher_name, batcher in batchers.items():
        qs[batcher_name] = q_function(
            batcher if full_grid else batcher(),
            models = models_dict[batcher_name],
            models_require_grad = models_require_grad,
            quantities_requiring_grad_labels = quantities_requiring_grad_dict[batcher_name],
            models_requiring_grad_labels = models_requiring_grad_dict[batcher_name],
        )

    return qs
