# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import copy
from collections.abc import Sequence
from typing import Dict
import torch
import collections

from . import mathematics
from .grids import Grid, Subgrid


class QuantityDict(collections.UserDict):
    """
    A dictionary of quantities of type `torch.Tensor` on a common `grid`.
    Overwriting is only possible through the 'overwrite' function in order to
    catch performance bugs.
    """

    def __init__(self, grid: Grid, *args, **kwargs):
        self.grid = grid
        self._allow_overwrite = False
        super().__init__(*args, **kwargs)

    def __setitem__(self, label: str, quantity):
        assert self._allow_overwrite or label not in self, label
        assert torch.is_tensor(quantity), quantity
        assert compatible(quantity, self.grid), quantity
        super().__setitem__(label, quantity)

    def overwrite(self, label: str, quantity):
        assert label in self, label
        self._allow_overwrite = True
        self.__setitem__(label, quantity)
        self._allow_overwrite = False


def compatible(tensor: torch.Tensor, grid: Grid) -> bool:
    if len(tensor.size()) != len(grid.shape):
        return False
    for tensor_size, dimension_size in zip(tensor.size(), grid.shape):
        if tensor_size not in (1, dimension_size):
            return False

    return True


def is_singleton_dimension(
    label: str,
    tensor: torch.Tensor,
    grid: Grid,
    *,
    check_compatible: bool = True,
) -> bool:
    assert not check_compatible or compatible(tensor, grid)
    return tensor.size(grid.index[label]) == 1


def might_depend_on(label: str, tensor: torch.Tensor, grid: Grid) -> bool:
    """Whether tensor might depend on grid[label]"""
    assert compatible(tensor, grid)
    return ((not is_singleton_dimension(label, tensor, grid))
            or grid.dim_size[label] == 1)


def sum_dimension(
    label: str,
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    """Sum over the dimension `label`"""

    assert compatible(tensor, grid)

    if is_singleton_dimension(label, tensor, grid):
        return tensor * grid.dim_size[label]

    sum_index = grid.index[label]
    summed_tensor = torch.sum(tensor, dim=sum_index, keepdim=True)

    return summed_tensor


def sum_dimensions(
    labels: list[str],
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    assert compatible(tensor, grid)

    out = tensor
    for label in labels:
        out = sum_dimension(label, tensor, grid)

    return out


def mean_dimension(
    label: str,
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    assert compatible(tensor, grid)
    return sum_dimension(label, tensor, grid) / grid.dim_size[label]


def mean_dimensions(
    labels: list[str],
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    assert compatible(tensor, grid)

    out = tensor
    for label in labels:
        out = mean_dimension(label, tensor, grid)

    return out


def _get_derivative(
    dim_index,
    tensor: torch.Tensor,
    grid: Grid,
    *,
    slice_=None,
) -> torch.Tensor:
    assert compatible(tensor, grid)

    dimension = grid[grid.dimensions_labels[dim_index]]

    if slice_ is not None:
        slices = [slice(None)] * grid.n_dim
        slices[dim_index] = slice_
        tensor = tensor[slices]
        dimension = dimension[slice_]

    tensor_diff = torch.diff(tensor, dim=dim_index)
    dimension_diff = torch.diff(dimension)
    dimension_diff = mathematics.expand(
        dimension_diff,
        tensor_diff.size(),
        [dim_index],
    )
    derivative = tensor_diff / dimension_diff

    return derivative


def get_fd_derivative(
    dimension: str,
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    """
    Derive along `dimension` using finite differences.
    Central differences are used, and on the boundary an extrapolation
    incorporating three grid points is performed.
    Might be inaccurate on non-equispaced grids.
    """
    assert compatible(tensor, grid)

    dim_index = grid.index[dimension]

    # Calculate the central differences:
    # odd_derivatives starts at index 1 and
    # even at 2
    even_slice = slice(0, None, 2)
    odd_slice = slice(1, None, 2)
    odd_derivative = _get_derivative(
        dim_index,
        tensor,
        grid,
        slice_=even_slice,
    )
    even_derivative = _get_derivative(
        dim_index,
        tensor,
        grid,
        slice_=odd_slice,
    )
    derivative = mathematics.interleave(
        odd_derivative,
        even_derivative,
        dim=dim_index,
    )

    # Calculate the derivatives at the left and right
    left_slice = slice(0, 2)
    right_slice = slice(-2, None)
    left_mid_derivative = _get_derivative(
        dim_index,
        tensor,
        grid,
        slice_=left_slice,
    )
    right_mid_derivative = _get_derivative(
        dim_index,
        tensor,
        grid,
        slice_=right_slice,
    )

    # Extrapolate to the very left and right
    # (interpreting left/right_derivative as defined in the middle between
    # the outermost and the inner point)
    # Equivalent to the left/right-sided stencils at (4) in
    # https://www.colorado.edu/amath/sites/default/files/attached-files/wk10_finitedifferences.pdf
    # for equispaced grids.
    full_slices = [slice(None)] * grid.n_dim
    left_slices = copy.copy(full_slices)
    left_slices[dim_index] = slice(0, 1)
    right_slices = copy.copy(full_slices)
    right_slices[dim_index] = slice(-1, None)
    left_inner_derivative = derivative[left_slices]
    right_inner_derivative = derivative[right_slices]
    left_derivative = 2 * left_mid_derivative - left_inner_derivative
    right_derivative = 2 * right_mid_derivative - right_inner_derivative
    derivative = torch.cat(
        (left_derivative, derivative, right_derivative),
        dim_index,
    )

    return derivative


def get_fd_second_derivative(
    dimension: str,
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    """
    Seperate treatment of the second derivative to make sure
    not only even/odd points are used in each point.
    Will be inaccurate on non-equispaced grids.
    """
    assert compatible(tensor, grid)

    dim_index = grid.index[dimension]

    dx = ((grid[dimension][-1] - grid[dimension][0])
          / (grid.dim_size[dimension] - 1))
    full_slices = [slice(None)] * grid.n_dim
    left_slices = copy.copy(full_slices)
    left_slices[dim_index] = slice(0, -2)
    mid_slices = copy.copy(full_slices)
    mid_slices[dim_index] = slice(1, -1)
    right_slices = copy.copy(full_slices)
    right_slices[dim_index] = slice(2, None)
    second_derivative = (tensor[left_slices]
                         + tensor[right_slices]
                         - 2 * tensor[mid_slices]) / dx**2

    # Extrapolate to the very left and right
    # Equivalent to the left/right-sided stencils at (4) in
    # https://www.colorado.edu/amath/sites/default/files/attached-files/wk10_finitedifferences.pdf
    # for equispaced grids.
    left_slices = copy.copy(full_slices)
    left_slices[dim_index] = slice(0, 1)
    second_left_slices = copy.copy(full_slices)
    second_left_slices[dim_index] = slice(1, 2)
    right_slices = copy.copy(full_slices)
    right_slices[dim_index] = slice(-1, None)
    second_right_slices = copy.copy(full_slices)
    second_right_slices[dim_index] = slice(-2, -1)
    left_derivative = (2 * second_derivative[left_slices]
                       - second_derivative[second_left_slices])
    right_derivative = (2 * second_derivative[right_slices]
                        - second_derivative[second_right_slices])
    second_derivative = torch.cat(
        (left_derivative, second_derivative, right_derivative),
        dim_index,
    )

    return second_derivative


def restrict(tensor: torch.Tensor, subgrid: Subgrid) -> torch.Tensor:
    """
    Restrict `tensor` to `subgrid`.
    If no restriction is performed (because all batched dims are singleton
    in `tensor`) a reference to (not a copy of) `tensor` is returned
    """

    assert compatible(tensor, subgrid.parent_grid)

    restricted_tensor = tensor
    for dim, label in enumerate(subgrid.dimensions_labels):
        if (label not in subgrid.indices_dict.keys()
            or is_singleton_dimension(
                   label,
                   tensor,
                   subgrid,
                   check_compatible=False,
               )):
            continue
        indices = subgrid.indices_dict[label]
        restricted_tensor = restricted_tensor.index_select(dim, indices)

    assert compatible(restricted_tensor, subgrid)

    return restricted_tensor


def expand_all_dims(tensor: torch.Tensor, grid: Grid):
    assert compatible(tensor, grid)
    return tensor.expand(grid.shape)


def squeeze_to(
    dimensions_labels: list[str],
    tensor: torch.Tensor,
    grid: Grid,
) -> torch.Tensor:
    """
    Get a lower-dimensional tensor that only depends on the dimensions in
    `dimensions_labels` and spans over all gridpoints in these dimensions.
    """
    assert compatible(tensor, grid)

    dimensions_to_squeeze = list(range(len(grid.dimensions)))
    index = -1
    for label in dimensions_labels:
        assert might_depend_on(label, tensor, grid), \
               "Quantity not dependent on " + label
        new_index = grid.index[label]
        # IDEA: arbitrary order
        assert new_index > index, \
               f"Wrong order: {dimensions_labels} vs. {grid.dimensions_labels}"
        index = new_index
        dimensions_to_squeeze.remove(index)

    for dimension_to_squeeze in dimensions_to_squeeze:
        assert tensor.size(dim=dimension_to_squeeze) == 1

    return tensor.squeeze(dimensions_to_squeeze)


def unsqueeze_to(
    grid: Grid,
    tensor: torch.Tensor,
    dimensions_labels: Sequence[str],
) -> torch.Tensor:
    """
    tensor[i, j, ...]: value at
    (grid[dimensions_labels[0]][i], grid[dimensions_labels[1]][j], ...)
    Returns the corresponding tensor on the `grid`.
    The input `tensor` must span over all gridpoints in its dimensions.
    """

    tensor_indices = [grid.index[label] for label in dimensions_labels]
    grid_tensor = mathematics.expand(tensor, grid.shape, tensor_indices)
    assert compatible(grid_tensor, grid)

    return grid_tensor


def combine_quantity(quantity_list, subgrid_list, grid: Grid):
    """
    Combine tensors on subgrids of `grid` to a tensor on `grid`.
    All subgrids need to slice the same dimensions.
    """

    assert len(quantity_list) == len(subgrid_list)

    dtype = quantity_list[0].dtype
    # reduced_dimensions_labels: dims that got sliced in the subgrids
    reduced_dimensions_labels = set(subgrid_list[0].indices_dict.keys())

    output_shape = [1] * grid.n_dim
    for i, label in enumerate(grid.dimensions_labels):
        expand_dimension = False

        if label in reduced_dimensions_labels:
            expand_dimension = True

        for quantity, subgrid in zip(quantity_list, subgrid_list):
            if might_depend_on(label, quantity, subgrid):
                expand_dimension = True

        if expand_dimension:
            output_shape[i] = grid.dim_size[label]

    tensor = torch.zeros(output_shape, dtype=dtype)
    covered = torch.zeros(output_shape, dtype=torch.bool)

    for quantity, subgrid in zip(quantity_list, subgrid_list):
        assert compatible(quantity, subgrid)
        assert subgrid.parent_grid is grid
        assert quantity.dtype is dtype
        assert set(subgrid.indices_dict.keys()) == reduced_dimensions_labels, \
               'Different dimensions are sliced in the subgrids'

        sub_tensor = torch.clone(quantity)
        sub_covered = torch.ones_like(sub_tensor, dtype=torch.bool)

        # Expand `quantity` to the full `grid`, filling unknown
        # entries with zero.
        # Since indexing with tensors in multiple dimensions at once is not
        # possible we perform the expansion step by step.
        for label in reduced_dimensions_labels:
            dim = grid.index[label]
            new_shape = list(sub_tensor.size())
            new_shape[dim] = grid.dim_size[label]
            new_tensor = torch.zeros(new_shape, dtype=dtype)
            new_covered = torch.zeros(new_shape, dtype=torch.bool)
            slices = [slice(None)] * grid.n_dim
            slices[dim] = subgrid.indices_dict[label]
            new_tensor[slices] = sub_tensor
            new_covered[slices] = sub_covered
            sub_tensor = new_tensor
            sub_covered = new_covered

        assert not torch.any(torch.logical_and(covered, sub_covered)), \
               'Double coverage'

        tensor += sub_tensor
        covered += sub_covered

    assert torch.all(covered)

    return tensor


def combine_quantities(qs: Sequence[QuantityDict], grid: Grid):
    labels = set(qs[0].keys())
    for q in qs:
        assert set(q.keys()) == labels

    q_combined = QuantityDict(grid)
    for label in labels:
        q_combined[label] = combine_quantity(
            [q[label] for q in qs],
            [q.grid for q in qs],
            grid,
        )

    return q_combined


def restrict_quantities(q: QuantityDict, subgrid: Subgrid) -> QuantityDict:
    return QuantityDict(
        subgrid,
        ((label, restrict(quantity, subgrid))
         for (label, quantity) in q.items()),
    )
