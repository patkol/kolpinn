"""
The network architecture and functions to interface with it
"""

from typing import Optional, Callable, Union, Iterable
import copy
import torch
from torch import nn

from . import grid_quantities
from .grid_quantities import Grid, Quantity, combine_quantity, QuantityDict
from .batching import Batcher



class Model:
    """
    Model.apply(q: QuantityDict) -> Quantity of type `output_dtype`
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

    def apply(q: QuantityDict, **kwargs):
        raise Exception('`Model` is an ABC')

    def set_requires_grad(self, requires_grad: bool):
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
        return Quantity(self.parameters[0].type(self.output_dtype), q.grid)


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
            input_transformations: dict,
            output_transformation,
            activation_function,
            *,
            n_neurons_per_hidden_layer: int,
            n_hidden_layers: int,
            model_dtype,
            output_dtype,
            device: str,
        ):
        """
        inputs_labels: labels of the quantities that will be input to the network
        The transformations will be applied to the corresponding
            input/output before/after passing it through the network.
            transformation(quantity, q: QuantityDict), only quantity requires grad.
        """

        self.inputs_labels = inputs_labels
        self.n_inputs = len(inputs_labels)
        self.input_transformations = input_transformations
        self.output_transformation = output_transformation
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
        super().__init__(
            list(self.network.parameters()),
            model_dtype = model_dtype,
            output_dtype = output_dtype,
        )

    def apply(
            self,
            q: QuantityDict,
            *,
            get_intermediates: bool = False,
        ):
        # inputs_tensor[gridpoint, input quantity]
        inputs_tensor = torch.zeros(
            (q.grid.n_points, self.n_inputs),
            dtype = self.model_dtype,
        )
        for (i, label) in enumerate(self.inputs_labels):
            input_quantity = Quantity(q[label].get_expanded_values(), q.grid)
            transformed_input = self.input_transformations[label](input_quantity, q)
            inputs_tensor[:,i] = transformed_input.values.flatten()

        outputs_tensor = self.network(inputs_tensor)
        if self.complex_output:
            outputs_tensor = torch.view_as_complex(outputs_tensor)
        outputs_tensor = outputs_tensor.reshape(q.grid.shape)
        outputs_tensor = outputs_tensor.type(self.output_dtype)
        output = Quantity(outputs_tensor, q.grid)
        transformed_output = self.output_transformation(output, q)

        if get_intermediates:
            return inputs_tensor, output, transformed_output

        return transformed_output

    def apply_to_all(
            self,
            batcher: Batcher,
            grid: Grid,
            get_intermediates: bool = False,
        ):
        inputs_tensors = []
        outputs = []
        transformed_outputs = []

        for q, subgrid in batcher.get_all():
            results = self.apply(q, get_intermediates=get_intermediates)

            if not get_intermediates:
                transformed_outputs.append(results[0])
                continue

            inputs_tensors.append(results[0])
            outputs.append(results[1])
            transformed_outputs.append(results[2])

        transformed_output = combine_quantity(transformed_outputs, grid)

        if not get_intermediates:
            return transformed_output

        inputs_tensor = torch.vstack(inputs_tensors)
        output = combine_quantity(outputs, grid)

        return inputs_tensor, output, transformed_output

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()


# TODO: Code duplication in loss.get_losses and batching.get_extended_q
def get_extended_q(
        q_in: QuantityDict,
        *,
        models: dict = None,
        models_require_grad: bool = False,
        diffable_quantities: Optional[dict[str,Callable]] = None,
        quantities_requiring_grad_labels: list[str] = None,
    ):
    """
    Get the quantities including the evaluated models.
    """

    if models is None:
        models = {}
    if diffable_quantities is None:
        diffable_quantities = {}
    if quantities_requiring_grad_labels is None:
        quantities_requiring_grad_labels = []


    q = copy.copy(q_in)

    unexpanded_quantities = {}
    for quantity_requiring_grad_label in quantities_requiring_grad_labels:
        unexpanded_quantity = q[quantity_requiring_grad_label]
        unexpanded_quantities[quantity_requiring_grad_label] = unexpanded_quantity
        q[quantity_requiring_grad_label] = Quantity(
            unexpanded_quantity.get_expanded_values(),
            unexpanded_quantity.grid,
        )
        q[quantity_requiring_grad_label].set_requires_grad(True)

    for model_name, model in models.items():
        assert not model_name in q, model_name
        model.set_requires_grad(models_require_grad)
        q[model_name] = model.apply(q)

    for diffable_quantity_name, diffable_quantity_function in diffable_quantities.items():
        q[diffable_quantity_name] = diffable_quantity_function(q)

    for quantity_requiring_grad_label in quantities_requiring_grad_labels:
        q[quantity_requiring_grad_label] = unexpanded_quantities[quantity_requiring_grad_label]

    return q
