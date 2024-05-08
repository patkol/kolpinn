# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import copy
from collections.abc import Sequence
import textwrap
import math
import torch
import collections

from . import mathematics


class Grid:
    def __init__(
        self,
        dimensions: dict,
    ):
        """
        dimensions[label] is a 1D tensor.

        Tensors on the grid:
        tensor[i, j, ...] represents the value at coordinates (i, j, k, ...)
        in the grid (coordinate ordering according to grid.dimensions).
        For dimensions the tensor does not depend on it has
        singleton dimensions.
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
                          for index, label
                          in enumerate(self.dimensions_labels))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dimensions.keys()})'

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

        for label, indices in indices_dict.items():
            assert min(indices) >= 0 and max(indices) < parent_grid.dim_size[label], \
                   f'{label} {indices} {parent_grid.dim_size}'

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
        return ('Subgrid(\n'
                + textwrap.indent(f'parent_grid={self.parent_grid},\n', '    ')
                + textwrap.indent(f'indices_dict={self.indices_dict}),\n', '    ')
                + ')')


class Supergrid(Grid):
    def __init__(
        self,
        child_grids: dict[str, Grid],
        dimension_name: str,
        copy_all: bool,
    ):
        """
        Concatenate the `child_grids` along `dimension_name`.
        The `indices_dict` then allows one to get the values on the
        child grids:
        `dimensions[dimension_name][self.indices_dict[child_grid_name]]`
        corresponds to
        `child_grid.dimensions[dimension_name]`
        `subgrids` contains the `child_grids` as `Subgrid`s of `self`.
        """

        first_child = next(iter(child_grids.values()))
        dimensions = copy.copy(first_child.dimensions)
        dimensions[dimension_name] = torch.cat(
            [child_grid[dimension_name] for child_grid in child_grids.values()]
        )

        if copy_all:
            dimensions = copy.deepcopy(dimensions)

        super().__init__(dimensions)

        self.indices_dict = {}
        self.subgrids = {}
        first_index = 0
        for child_grid_name, child_grid in child_grids.items():
            for label, dimension in child_grid.dimensions.items():
                assert (label == dimension_name
                        or torch.all(dimension == self.dimensions[label])), \
                    f'Dimension {label} is not the same for all grids'

            dimension_size = child_grid[dimension_name].size(dim=0)
            next_first_index = first_index + dimension_size
            self.indices_dict[child_grid_name] = torch.arange(
                first_index, first_index + dimension_size,
            )
            first_index = next_first_index

            self.subgrids[child_grid_name] = Subgrid(
                self,
                {dimension_name: self.indices_dict[child_grid_name]},
                copy_all=False,
            )

            assert torch.all(self.dimensions[dimension_name][self.indices_dict[child_grid_name]]
                             == child_grid.dimensions[dimension_name]), \
                child_grid_name

        assert first_index == self.dimensions[dimension_name].size(dim=0)


class QuantityDict(collections.UserDict):
    """ The values must be tensors """
    def __init__(self, grid: Grid, *args, **kwargs):
        self.grid = grid
        super().__init__(*args, **kwargs)

    def __setitem__(self, label: str, quantity):
        assert torch.is_tensor(quantity), quantity
        assert compatible(quantity, self.grid), quantity
        super().__setitem__(label, quantity)


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


def restrict(tensor: torch.Tensor, subgrid: Grid) -> torch.Tensor:
    """
    Restrict `tensor` to `subgrid`.
    If `tensor` is already compatible with `subgrid`, no restriction
    is performed.
    """
    if compatible(tensor, subgrid):
        return tensor

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

    return tensor.squeeze(dimensions_to_squeeze)


def unsqueeze_to(
    grid: Grid,
    tensor: torch.Tensor,
    dimensions_labels: list[str],
) -> torch.Tensor:
    """
    tensor[i, j, ...]: value at
    (grid[dimensions_labels[0]][i], grid[dimensions_labels[1]][j], ...)
    Returns the corresponding tensor on the `grid`.
    """

    tensor_indices = [grid.index[label] for label in dimensions_labels]
    grid_tensor = mathematics.expand(tensor, grid.shape, tensor_indices)
    assert compatible(grid_tensor, grid)

    return grid_tensor


def combine_quantity(quantity_list, subgrid_list, grid: Grid):
    """Combine tensors on subgrids of `grid` to a tensor on `grid`"""

    if len(quantity_list) == 1:
        assert compatible(quantity_list[0], subgrid_list[0])
        assert compatible(quantity_list[0], grid)
        return quantity_list[0]

    dtype = quantity_list[0].dtype
    # reduced_dimensions_labels: dims that got sliced in the subgrids
    reduced_dimensions_labels = set(subgrid_list[0].indices_dict.keys())

    tensor = torch.zeros(grid.shape, dtype=dtype)
    covered = torch.zeros(grid.shape, dtype=torch.bool)

    for quantity, subgrid in zip(quantity_list, subgrid_list):
        assert compatible(quantity, subgrid)
        assert subgrid.parent_grid is grid
        assert quantity.dtype is dtype
        assert set(subgrid.indices_dict.keys()) == reduced_dimensions_labels

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

        tensor += sub_tensor
        covered += sub_covered

    assert torch.all(covered)

    return tensor


def combine_quantities(qs: Sequence[QuantityDict], grid: Grid):
    q_combined = QuantityDict(grid)
    for label in qs[0].keys():
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
