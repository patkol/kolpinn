#!/usr/bin/env python3

import copy
import typing
from typing import Optional, Iterable, Union
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
    def __init__(self, values, grid: Grid, **kwargs):
        """
        values[i, j, ...] represents the value at coordinates (i, j, k, ...)
        in the grid (coordinate ordering according to grid.dimensions).
        For dimensions `values` does not depend on it has singleton dimensions.
        """

        self.values = torch.as_tensor(values, **kwargs)

        # Assert that the grid matches the tensor
        assert len(self.values.size()) == len(grid.shape), \
               f"{values.size()} {grid.shape}"
        for values_size, dimension_size in zip(values.size(), grid.shape):
            assert values_size in (1, dimension_size), \
                   f"values.size(): {values.size()}, grid.shape: {grid.shape}"

        self.grid = grid

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def requires_grad(self):
        return self.values.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.values.requires_grad = value

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # https://pytorch.org/docs/stable/notes/extending.html
        if kwargs is None:
            kwargs = {}

        grids = tuple(arg.grid for arg in args if hasattr(arg, 'grid'))
        assert len(grids) > 0
        new_grid = grids[0]
        for grid in grids:
            assert grid is new_grid, f'Grid mismatch: {grid} is not {new_grid}'

        new_args = [arg.values if type(arg) is Quantity else arg for arg in args]
        ret = func(*new_args, **kwargs)
        out = Quantity(ret, grid = new_grid) if torch.is_tensor(ret) else ret

        return out

    def __repr__(self):
        return ('Quantity(\n'
                + textwrap.indent(f'{self.values},\n', '    ')
                + textwrap.indent(f'{self.grid},\n', '    ')
                + ')')

    def __neg__(self):
        """Return -self"""
        return torch.neg(self)

    def __add__(self, other):
        """Return self+other"""
        return torch.add(self, other)

    def __radd__(self, other):
        """Return other+self"""
        return torch.add(other, self)

    def __sub__(self, other):
        """Return self-other"""
        return torch.sub(self, other)

    def __rsub__(self, other):
        """Return other-self"""
        return torch.sub(other, self)

    def __mul__(self, other):
        """Return self*other"""
        return torch.mul(self, other)

    def __rmul__(self, other):
        """Return other*self"""
        return torch.mul(other, self)

    def __truediv__(self, other):
        """Return self/other"""
        return torch.true_divide(self, other)

    def __rtruediv__(self, other):
        """Return other/self"""
        return torch.true_divide(other, self)

    def __pow__(self, other):
        """Return self**other"""
        return torch.pow(self, other)

    def __rpow__(self, other):
        """Return other**self"""
        return torch.pow(other, self)

    def __eq__(self, other):
        """Return self==other"""
        return torch.eq(self, other)

    def __ge__(self, other):
        """Return self>=other"""
        return torch.ge(self, other)

    def __le__(self, other):
        """Return self<=other"""
        return torch.le(self, other)

    def __gt__(self, other):
        """Return self>other"""
        return torch.gt(self, other)

    def __lt__(self, other):
        """Return self<other"""
        return torch.lt(self, other)

    def is_singleton_dimension(self, label: str) -> bool:
        return self.values.size(self.grid.index[label]) == 1

    def might_depend_on(self, label: str) -> bool:
        return (not self.is_singleton_dimension(label)) or self.grid.dim_size[label] == 1

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

    def mean_dimension(self, label):
        return (self.sum_dimension(label)
                / self.grid.dim_size[label])

    def mean_dimensions(self, labels):
        out = self
        for label in labels:
            out = out.mean_dimension(label)
        return out

    def get_grad(self, input_: 'Quantity', **kwargs) -> 'Quantity':
        """
        kwargs example: retain_graph=True, create_graph=True
        """

        assert self.grid == input_.grid
        grad_tensor = mathematics.grad(self.values, input_.values, **kwargs)

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

    def expand_all_dims(self):
        return Quantity(self.values.expand(self.grid.shape), self.grid)

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
            # IDEA: arbitrary order
            assert new_index > index, \
                   f"Wrong order: {dimensions_labels} vs. {self.grid.dimensions_labels}"
            index = new_index
            dimensions_to_squeeze.remove(index)

        return self.values.squeeze(dimensions_to_squeeze)

    def set_dtype(self, *args, **kwargs):
        self.values = self.values.to(*args, **kwargs)
        return self


class QuantityDict(collections.UserDict):
    """ The values can be quantities, tensors and scalars """
    def __init__(self, grid, *args, **kwargs):
        self.grid = grid
        super().__init__(*args, **kwargs)

    def __setitem__(self, label: str, quantity):
        assert type(quantity) is not Quantity or quantity.grid is self.grid, \
                f'\n{quantity.grid}\n{self.grid}'
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

    tensor_indices = [grid.index[label] for label in dimensions_labels]
    values = mathematics.expand(tensor, grid.shape, tensor_indices)

    return Quantity(values, grid)



def combine_quantity(quantity_list, grid: Grid):
    """Combine quantities on subgrids of `grid` to a quantity on `grid`"""

    if type(quantity_list[0]) is not Quantity:
        assert all(quantity == quantity_list[0] for quantity in quantity_list), \
               quantity_list
        return quantity_list[0]

    dtype = quantity_list[0].dtype
    # reduced_dimensions_labels: dims that got sliced in the subgrids
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


def combine_quantities(qs_in: Iterable[QuantityDict], grid: Grid):
    q_combined = QuantityDict(grid)
    for label in qs_in[0].keys():
        q_combined[label] = combine_quantity(
            [q_in[label] for q_in in qs_in],
            grid,
        )

    return q_combined




def restrict_quantities(q: QuantityDict, subgrid: Subgrid) -> dict:
    return QuantityDict(
        subgrid,
        ((label, quantity.restrict(subgrid)) for (label, quantity) in q.items()),
    )
