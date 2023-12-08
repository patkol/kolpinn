#!/usr/bin/env python3

import copy
import typing
from typing import Optional
import textwrap
import math
import torch
import itertools
import collections

from . import mathematics


class Grid:
    def __init__(
            self,
            dimensions: dict,
        ):
        """
        dimensions[label] is a 1D tensor
        """

        self.dimensions = dimensions
        self.dimensions_labels = list(dimensions.keys())
        self.dtype = list(self.dimensions.values())[0].dtype
        for label in self.dimensions_labels:
            assert dimensions[label].dtype is self.dtype, \
                   (self.dtype, label, dimensions[label].dtype)
        self.shape = torch.Size([torch.numel(self.dimensions[label])
                                 for label in self.dimensions_labels])
        self.n_dim = len(self.shape)
        self.n_points = math.prod(self.shape)
        self.dim_size = dict((label, torch.numel(self.dimensions[label]))
                             for label in self.dimensions_labels)
        self.index = dict((label, index)
                          for index, label in enumerate(self.dimensions_labels))

    def __repr__(self):
        return f'Grid({self.dimensions.keys()})'

    def __getitem__(self, dimension_label: str) -> torch.Tensor:
        return self.dimensions[dimension_label]

    def get_subgrid(self, conditions_dict: dict, copy_all: bool):
        indices_dict = {}
        for label, condition in conditions_dict.items():
            passing = condition(self.dimensions[label])
            indices_dict[label] = torch.nonzero(passing).flatten()

        return Subgrid(self, indices_dict, copy_all)

    def get_subgrids(self, conditions_dicts: dict, copy_all: bool):
        subgrids = {}
        for grid_label, conditions_dict in conditions_dicts.items():
            subgrids[grid_label] = self.get_subgrid(
                conditions_dict,
                copy_all=copy_all,
            )

        return subgrids


class Subgrid(Grid):
    def __init__(self, parent_grid: Grid, indices_dict: dict, copy_all: bool):
        """
        indices_dict[label]: indices of entries of dimensions[label] to be kept
        If copy_all all dimensions will be deepcopied such that they have
        seperate gradients.
        """

        self.parent_grid = parent_grid
        self.indices_dict = dict((label,
                                  (indices
                                   if torch.is_tensor(indices)
                                   else torch.tensor(indices, dtype=torch.long)))
                                 for (label, indices) in indices_dict.items())

        dimensions = {}
        for label in parent_grid.dimensions_labels:
            dimension = parent_grid[label]
            if label in indices_dict.keys():
                dimension = dimension[self.indices_dict[label]]
            dimensions[label] = dimension

        if copy_all:
            dimensions = copy.deepcopy(dimensions)

        super().__init__(dimensions)

    def __repr__(self):
        return f'Subgrid(parent_grid={self.parent_grid}, indices_dict={self.indices_dict})'

        return ('Subgrid(\n'
                + textwrap.indent(f'parent_grid={self.parent_grid},\n', '    ')
                + textwrap.indent(f'indices_dict={self.indices_dict}),\n', '    ')
                + ')')



class Quantity:
    def __init__(
            self,
            values: torch.Tensor,
            grid: Grid,
        ):
        """
        values[i, j, ...] represents the value at coordinates (i, j, k, ...)
        in the grid (coordinate ordering according to grid.dimensions).
        For dimensions `values` does not depend on it has singleton dimensions.
        If a 0D tensor is provided it will be reshaped.
        """

        if len(values.size()) == 0:
            shape = [1] * grid.n_dim
            values = values.reshape(shape)

        assert len(values.size()) == len(grid.shape), \
               f"{values.size()} {grid.shape}"
        for values_size, dimension_size in zip(values.size(), grid.shape):
            assert values_size in (1, dimension_size), \
                   f"values.size(): {values.size()}, grid.shape: {grid.shape}"

        self.values = values
        self.grid = grid
        self.dtype = values.dtype

    def __repr__(self):
        return ("Quantity(\n"
                + textwrap.indent(f"{self.values},\n", "    ")
                + textwrap.indent(f"{self.grid},\n", "    ")
                + ")")

    def __neg__(self):
        """Return -self"""
        return Quantity(-self.values, self.grid)

    def __add__(self, other):
        """Return self+other"""
        other = self.get_compatible(other)
        return Quantity(self.values + other.values, self.grid)

    def __radd__(self, other):
        """Return other+self"""
        return self + other

    def __sub__(self, other):
        """Return self-other"""
        other = self.get_compatible(other)
        return Quantity(self.values - other.values, self.grid)

    def __rsub__(self, other):
        """Return other-self"""
        other = self.get_compatible(other)
        return Quantity(other.values - self.values, self.grid)

    def __mul__(self, other):
        """Return self*other"""
        other = self.get_compatible(other)
        return Quantity(self.values * other.values, self.grid)

    def __rmul__(self, other):
        """Return other*self"""
        return self * other

    def __truediv__(self, other):
        """Return self/other"""
        other = self.get_compatible(other)
        return Quantity(self.values / other.values, self.grid)

    def __rtruediv__(self, other):
        """Return other/self"""
        other = self.get_compatible(other)
        return Quantity(other.values / self.values, self.grid)

    def __pow__(self, other):
        """Return self**other"""
        other = self.get_compatible(other)
        return Quantity(self.values ** other.values, self.grid)

    def __rpow__(self, other):
        """Return other**self"""
        other = self.get_compatible(other)
        return Quantity(other.values ** self.values, self.grid)

    def __eq__(self, other):
        """Return self==other"""
        other = self.get_compatible(other)
        return Quantity(self.values == other.values, self.grid)

    def __ge__(self, other):
        """Return self>=other"""
        other = self.get_compatible(other)
        return Quantity(self.values >= other.values, self.grid)

    def __le__(self, other):
        """Return self<=other"""
        other = self.get_compatible(other)
        return Quantity(self.values <= other.values, self.grid)

    def __gt__(self, other):
        """Return self>other"""
        other = self.get_compatible(other)
        return Quantity(self.values > other.values, self.grid)

    def __lt__(self, other):
        """Return self<other"""
        other = self.get_compatible(other)
        return Quantity(self.values < other.values, self.grid)

    def compatible(self, other) -> bool:
        return self.grid is other.grid

    def get_compatible(self, other):
        """
        `other` can be a scalar, it is then converted to a quantity.
        """
        if not type(other) == Quantity:
            # `other` must be a scalar
            assert not hasattr(other, '__len__'), (type(other), other)

            dtype = self.values.dtype
            if isinstance(other, complex) and not dtype in (torch.complex64, torch.complex128):
                if dtype == torch.float32:
                    dtype = torch.complex64
                elif dtype == torch.float64:
                    dtype = torch.complex128
                else:
                    raise Exception("Unsupported real datatype")

            device = self.values.device
            other = get_quantity(
                torch.tensor(other, dtype=dtype, device=device),
                [],
                self.grid,
            )

        assert self.compatible(other)

        return other

    def is_singleton_dimension(self, label: str) -> bool:
        return self.values.size(self.grid.index[label]) == 1

    def might_depend_on(self, label: str) -> bool:
        return (not self.is_singleton_dimension(label)) or self.grid.dim_size[label] == 1

    def transform(self, function) -> 'Quantity':
        transformed_values = function(self.values)
        assert transformed_values.size() == self.values.size(), \
               f"{transformed_values}, {self.values}"

        return Quantity(transformed_values, self.grid)

    def sum_dimension(self, label):
        """Sum over the dimension `label`"""

        if self.is_singleton_dimension(label):
            return self * self.grid.dim_size[label]

        sum_index = self.grid.index[label]
        summed_values = torch.sum(self.values, dim = sum_index, keepdim = True)

        return Quantity(summed_values, self.grid)

    def sum_dimensions(self, labels):
        out = self
        for label in labels:
            out = out.sum_dimension(label)
        return out

    def sum_(self):
        summed_values = torch.sum(self.values)
        return Quantity(summed_values, self.grid)

    def mean_dimension(self, label):
        return (self.sum_dimension(label)
                / self.grid.dim_size[label])

    def mean_dimensions(self, labels):
        out = self
        for label in labels:
            out = out.mean_dimension(label)
        return out

    def mean(self):
        mean = torch.mean(self.values)
        return Quantity(mean, self.grid)

    def get_grad(self, input_: 'Quantity', **kwargs) -> 'Quantity':
        """
        kwargs example: retain_graph=True, create_graph=True
        """
        assert self.compatible(input_)

        grad_function = (mathematics.complex_grad
                         if torch.is_complex(self.values)
                         else torch.autograd.grad)

        grad_outputs_dtype = self.dtype
        if self.dtype == torch.complex64:
            grad_outputs_dtype = torch.float32
        elif self.dtype == torch.complex128:
            grad_outputs_dtype = torch.float64

        grad_tensor = grad_function(
            outputs = self.values,
            inputs = input_.values,
            grad_outputs = torch.ones_like(self.values, dtype=grad_outputs_dtype),
            **kwargs,
        )

        if grad_function is torch.autograd.grad:
            grad_tensor = grad_tensor[0]

        return Quantity(grad_tensor, input_.grid)

    def restrict(self, subgrid: Grid):
        if subgrid is self.grid:
            return self

        assert subgrid.parent_grid is self.grid

        restricted_values = self.values
        for values_dim, label in enumerate(self.grid.dimensions_labels):
            if ((not label in subgrid.indices_dict.keys())
                    or self.is_singleton_dimension(label)):
                continue
            indices = subgrid.indices_dict[label]
            restricted_values = restricted_values.index_select(values_dim, indices)

        return Quantity(restricted_values, subgrid)

    def get_expanded_values(self):
        return self.values.expand(self.grid.shape)

    def get_tensor(self, dimensions_labels: list[str]):
        """
        Get the values as a tensor as provided to get_quantity.
        """

        dimensions_to_squeeze = list(range(len(self.grid.dimensions)))
        index = -1
        for label in dimensions_labels:
            assert self.might_depend_on(label), \
                   "Quantity not dependent on " + label
            new_index = self.grid.index[label]
            assert new_index > index, \
                   f"Wrong order: {dimensions_labels} vs. {self.grid.dimensions_labels}"
            index = new_index
            dimensions_to_squeeze.remove(index)

        return self.values.squeeze(dimensions_to_squeeze)

    def set_requires_grad(self, b: bool):
        self.values.requires_grad_(b)

        return self

    def set_dtype(self, new_dtype):
        if new_dtype is None:
            return self

        self.values = self.values.to(new_dtype)
        self.dtype = new_dtype

        return self


class QuantityDict(collections.UserDict):
    def __init__(self, grid, *args, **kwargs):
        self.grid = grid
        super().__init__(*args, **kwargs)

    def __setitem__(self, label: str, quantity: Quantity):
        assert quantity.grid is self.grid, f'\n{quantity.grid}\n{self.grid}'
        super().__setitem__(label, quantity)


class QuantityFactory:
    label: Optional[str] = None

    def __call__(self, quantities: QuantityDict):
        assert not self.label is None
        quantity = self.function(quantities)
        return self.label, quantity

    def function(self, quantities: QuantityDict):
        assert False, "QuantityFactory is an ABC"


class QuantitiesFactory:
    def __init__(self):
        self.factories = []

    def __call__(self, grid: Grid, additional_quantities = None):
        """additional_quantities[label]: Quantity"""

        if additional_quantities is None:
            additional_quantities = {}

        quantities = QuantityDict(grid)
        quantities.update(copy.copy(additional_quantities))

        # Add the dimensions as quantities
        for label, dimension_values in grid.dimensions.items():
            quantities[label] = get_quantity(dimension_values, [label], grid)

        for factory in self.factories:
            label, quantity = factory(quantities)
            quantities[label] = quantity

        return quantities

    def add(self, Factory: typing.Type[QuantityFactory], **kwargs):
        quantity_factory = Factory(**kwargs)
        self.factories.append(quantity_factory)

    def get_quantities_dict(self, grids: dict[str,Grid], additional_quantities = None):
        """
        grids[grid_name] is a grid
        """

        if additional_quantities is None:
            additional_quantities = {}

        quantities_dict = {}
        for grid_name, grid in grids.items():
            restricted_additional_quantities = restrict_quantities(
                additional_quantities,
                grid,
            )
            quantities_dict[grid_name] = self.__call__(
                grid,
                restricted_additional_quantities,
            )

        return quantities_dict


def get_quantity(
        tensor: torch.Tensor,
        dimensions_labels: list[str],
        grid: Grid,
    ) -> Quantity:
    """
    tensor[i, j, ...]: value at
    (grid[dimensions_labels[0]][i], grid[dimensions_labels[1]][j], ...)
    """

    if len(tensor.size()) == 0:
        assert dimensions_labels == []
        shape = [1] * len(grid.dimensions)

    else:
        shape = []
        tensor_index = 0
        for label in grid.dimensions_labels:
            dim_size = 1
            if tensor_index < len(dimensions_labels) and label == dimensions_labels[tensor_index]:
                dim_size = grid.dim_size[label]
                assert tensor.size(tensor_index) == dim_size
                tensor_index += 1

            shape.append(dim_size)

        assert tensor_index == len(dimensions_labels), \
               f"{shape}, {grid.dimensions_labels}, {dimensions_labels}\n Is the dimension order correct?"

    return Quantity(tensor.reshape(shape), grid)



def combine_quantity(quantity_list: list, grid: Grid):
    dtype = quantity_list[0].dtype
    reduced_dimensions_labels = set(quantity_list[0].grid.indices_dict.keys())

    values = torch.zeros(grid.shape, dtype=dtype)
    covered = torch.zeros(grid.shape, dtype=torch.bool)

    for quantity in quantity_list:
        subgrid = quantity.grid

        assert subgrid.parent_grid is grid
        assert quantity.dtype is dtype
        assert set(subgrid.indices_dict.keys()) == reduced_dimensions_labels

        quantity_values = torch.clone(quantity.values)
        quantity_covered = torch.ones_like(quantity_values, dtype=torch.bool)

        # Expand `quantity.values` to the full `grid`, filling unknown
        # entries with zero.
        # Since indexing with tensors in multiple dimensions at once is not
        # possible we perform the expansion step by step.
        for label in reduced_dimensions_labels:
            dim = grid.index[label]
            new_shape = list(quantity_values.size())
            new_shape[dim] = grid.dim_size[label]
            new_values = torch.zeros(new_shape, dtype=dtype)
            new_covered = torch.zeros(new_shape, dtype=torch.bool)
            slices = [slice(None)] * grid.n_dim
            slices[dim] = subgrid.indices_dict[label]
            new_values[slices] = quantity_values
            new_covered[slices] = quantity_covered
            quantity_values = new_values
            quantity_covered = new_covered

        values += quantity_values
        covered += quantity_covered

    assert torch.all(covered)

    return Quantity(values, grid)



def restrict_quantities(q: QuantityDict, subgrid: Subgrid) -> dict:
    return QuantityDict(
        subgrid,
        ((label, quantity.restrict(subgrid)) for (label, quantity) in q.items()),
    )
