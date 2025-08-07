# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Optional
import copy
import torch
import collections

from . import mathematics
from . import grids
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
    return (not is_singleton_dimension(label, tensor, grid)) or grid.dim_size[
        label
    ] == 1


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


def cumsum_dimension(
    label: str,
    tensor: torch.Tensor,
    grid: Grid,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Cumulative sum over the dimension `label`:
    [a, b, c] -> [a, a+b, a+b+c]
                 [a+b+c, b+c, c] if `reverse`
    """

    assert compatible(tensor, grid)

    sum_index = grid.index[label]
    if reverse:
        tensor = torch.flip(tensor, [sum_index])
    summed_tensor = torch.cumsum(tensor, dim=sum_index)
    if reverse:
        summed_tensor = torch.flip(summed_tensor, [sum_index])

    return summed_tensor


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


def get_cumulative_integral(
    label: str,
    start_coordinate: float,
    integrand: torch.Tensor,
    grid: Grid,
    start_value=0,
) -> torch.Tensor:
    """
    Returns a tensor representing the integral in the dimension `label` from
    `start_coordinate` to every coordinate. The value of the integral at
    `start_coordinate` is `start_value`, or zero by default.
    The `grid` must be sorted in the `label`-coordinate.
    """
    coords = grid[label]
    assert torch.equal(torch.sort(coords)[0], coords)

    # Identify the index corresponding to the `start_coordinate`
    start_indices = torch.nonzero(coords == start_coordinate)
    assert start_indices.size() == (1, 1)
    start_index = start_indices[0, 0]

    # Calculate the cumulative integral from index 0
    coords_shape = [1] * grid.n_dim
    coords_shape[grid.index[label]] = grid.dim_size[label]
    coords = coords.reshape(coords_shape)
    left_slice = grids.get_nd_slice(label, slice(None, -1), grid)
    right_slice = grids.get_nd_slice(label, slice(1, None), grid)
    delta_coords = coords[right_slice] - coords[left_slice]
    integrand_mids = (integrand[left_slice] + integrand[right_slice]) / 2
    integral = torch.cumsum(integrand_mids * delta_coords, dim=grid.index[label])
    zeros_shape = list(integral.shape)
    zeros_shape[grid.index[label]] = 1
    integral = torch.cat((torch.zeros(zeros_shape), integral), dim=grid.index[label])

    # Shift the integral s.t. it assumes `start_value` at `start_coordinate`
    start_index_slice = grids.get_nd_slice(
        label, slice(start_index, start_index + 1), grid
    )
    integral += start_value - integral[start_index_slice]

    assert integral.size() == integrand.size()

    return integral


def _get_derivative(
    dim_label: str,
    tensor: torch.Tensor,
    grid: Grid,
    *,
    slice_=None,
) -> torch.Tensor:
    assert compatible(tensor, grid)

    dimension = grid[dim_label]

    if slice_ is not None:
        slices = grids.get_nd_slice(dim_label, slice_, grid)
        tensor = tensor[slices]
        dimension = dimension[slice_]

    dim_index = grid.index[dim_label]
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

    # Calculate the central differences:
    # odd_derivatives starts at index 1 and
    # even at 2
    even_slice = slice(0, None, 2)
    odd_slice = slice(1, None, 2)
    odd_derivative = _get_derivative(
        dimension,
        tensor,
        grid,
        slice_=even_slice,
    )
    even_derivative = _get_derivative(
        dimension,
        tensor,
        grid,
        slice_=odd_slice,
    )
    derivative = mathematics.interleave(
        odd_derivative,
        even_derivative,
        dim=grid.index[dimension],
    )

    # Calculate the derivatives at the left and right
    left_slice = slice(0, 2)
    right_slice = slice(-2, None)
    left_mid_derivative = _get_derivative(
        dimension,
        tensor,
        grid,
        slice_=left_slice,
    )
    right_mid_derivative = _get_derivative(
        dimension,
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
    left_slices = grids.get_nd_slice(dimension, slice(0, 1), grid)
    right_slices = grids.get_nd_slice(dimension, slice(-1, None), grid)
    left_inner_derivative = derivative[left_slices]
    right_inner_derivative = derivative[right_slices]
    left_derivative = 2 * left_mid_derivative - left_inner_derivative
    right_derivative = 2 * right_mid_derivative - right_inner_derivative
    derivative = torch.cat(
        (left_derivative, derivative, right_derivative),
        grid.index[dimension],
    )

    return derivative


def get_fd_second_derivative(
    dimension: str,
    tensor: torch.Tensor,
    grid: Grid,
    *,
    factor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Seperate treatment of the second derivative to make sure
    not only even/odd points are used in each point.
    Will be inaccurate on non-equispaced grids.
    If `factor` is provided, d_x factor d_x tensor is computed.
    """

    assert compatible(tensor, grid)

    dim_index = grid.index[dimension]

    dx = (grid[dimension][-1] - grid[dimension][0]) / (grid.dim_size[dimension] - 1)
    left_slices = grids.get_nd_slice(dimension, slice(0, -2), grid)
    mid_slices = grids.get_nd_slice(dimension, slice(1, -1), grid)
    right_slices = grids.get_nd_slice(dimension, slice(2, None), grid)

    if factor is None:
        second_derivative = (
            tensor[left_slices] + tensor[right_slices] - 2 * tensor[mid_slices]
        ) / dx**2
    else:
        first_derivative_right = (tensor[right_slices] - tensor[mid_slices]) / dx
        first_derivative_left = (tensor[mid_slices] - tensor[left_slices]) / dx
        factor_right = (factor[mid_slices] + factor[right_slices]) / 2
        factor_left = (factor[left_slices] + factor[mid_slices]) / 2
        second_derivative = (
            factor_right * first_derivative_right - factor_left * first_derivative_left
        ) / dx

    # Extrapolate to the very left and right
    # Equivalent to the left/right-sided stencils at (4) in
    # https://www.colorado.edu/amath/sites/default/files/attached-files/wk10_finitedifferences.pdf
    # for equispaced grids.
    left_slices = grids.get_nd_slice(dimension, slice(0, 1), grid)
    second_left_slices = grids.get_nd_slice(dimension, slice(1, 2), grid)
    right_slices = grids.get_nd_slice(dimension, slice(-1, None), grid)
    second_right_slices = grids.get_nd_slice(dimension, slice(-2, -1), grid)
    left_derivative = (
        2 * second_derivative[left_slices] - second_derivative[second_left_slices]
    )
    right_derivative = (
        2 * second_derivative[right_slices] - second_derivative[second_right_slices]
    )
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
        if label not in subgrid.indices_dict.keys() or is_singleton_dimension(
            label,
            tensor,
            subgrid,
            check_compatible=False,
        ):
            continue
        indices = subgrid.indices_dict[label]
        restricted_tensor = restricted_tensor.index_select(dim, indices)

    assert compatible(restricted_tensor, subgrid)

    return restricted_tensor


def restrict_quantities(
    q: QuantityDict,
    subgrid: Subgrid,
    *,
    subgrid_for_restriction: Optional[Subgrid] = None,
) -> QuantityDict:
    if subgrid_for_restriction is None:
        subgrid_for_restriction = subgrid

    return QuantityDict(
        subgrid,
        (
            (label, restrict(quantity, subgrid_for_restriction))
            for (label, quantity) in q.items()
        ),
    )


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
        assert might_depend_on(label, tensor, grid), (
            "Quantity not dependent on " + label
        )
        new_index = grid.index[label]
        # IDEA: arbitrary order
        assert (
            new_index > index
        ), f"Wrong order: {dimensions_labels} vs. {grid.dimensions_labels}"
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
        assert (
            set(subgrid.indices_dict.keys()) == reduced_dimensions_labels
        ), "Different dimensions are sliced in the subgrids"

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
            slices = grids.get_nd_slice(label, subgrid.indices_dict[label], grid)
            new_tensor[slices] = sub_tensor
            new_covered[slices] = sub_covered
            sub_tensor = new_tensor
            sub_covered = new_covered

        assert not torch.any(torch.logical_and(covered, sub_covered)), "Double coverage"

        tensor += sub_tensor
        covered += sub_covered

    assert torch.all(covered)

    return tensor


def combine_quantities(q_sequence: Sequence[QuantityDict], grid: Grid):
    labels = set(q_sequence[0].keys())
    for q in q_sequence:
        assert set(q.keys()) == labels

    q_combined = QuantityDict(grid)
    for label in labels:
        q_combined[label] = combine_quantity(
            [q[label] for q in q_sequence],
            [q.grid for q in q_sequence],
            grid,
        )

    return q_combined


def interpolate(
    quantity_in: torch.Tensor,
    grid_in: Grid,
    grid_out: Grid,
    *,
    dimension_label: str,
    search_method: str,
):
    """
    Interpolate linearly from `quantity_in` on `grid_in` to `grid_out`, where
    the grids differ only in the dimension `dimension_label`.
    For gridpoints outside the range of `grid_in[dimension_label]` we extrapolate
    using the two outermost points.
    Assuming ordered grid_in & grid_out in `dimension_label`.
    If the grids are the same, a clone of quantity_in is returned.
    search_method: "incremental" is good if the output coordinates are dense in the
        input ones. If not, "searchsorted" will be faster.
    """

    assert compatible(quantity_in, grid_in)
    n_dim = grid_in.n_dim

    # Assert grid compatibility
    assert grid_out.n_dim == n_dim
    for (label_in, coordinates_in), (label_out, coordinates_out) in zip(
        grid_in.dimensions.items(), grid_out.dimensions.items()
    ):
        assert label_in == label_out
        if label_in == dimension_label:
            assert torch.equal(torch.sort(coordinates_in)[0], coordinates_in)
            assert torch.equal(torch.sort(coordinates_out)[0], coordinates_out)
            continue
        # No need to check if independent of the current dimension
        if not might_depend_on(label_in, quantity_in, grid_in):
            continue
        assert torch.equal(
            coordinates_in, coordinates_out
        ), "The grids differ in dimensions other than the interpolated one"

    coordinates_in = grid_in[dimension_label]
    coordinates_out = grid_out[dimension_label]

    assert len(coordinates_in) >= 2

    if torch.equal(coordinates_in, coordinates_out):
        return torch.clone(quantity_in)

    dimension_index = grid_in.index[dimension_label]
    shape_out = list(quantity_in.size())
    shape_out[dimension_index] = len(coordinates_out)
    quantity_out = torch.zeros(shape_out, dtype=quantity_in.dtype)

    # index of coordinates_in to the left of the interpolated point (not the rightmost
    # however s.t. we can extrapolate)
    i_in = 0
    for i_out, coordinate_out in enumerate(coordinates_out):
        if search_method == "incremental":
            while (
                i_in + 1 < len(coordinates_in) - 1
                and coordinates_in[i_in + 1] < coordinate_out
            ):
                i_in += 1
        else:
            assert search_method == "searchsorted"
            i_in = torch.searchsorted(coordinates_in, coordinate_out).item() - 1
            i_in = max(0, i_in)
            i_in = min(len(coordinates_in) - 2, i_in)

        i_in_left = i_in
        i_in_right = i_in + 1
        coordinate_left = coordinates_in[i_in_left]
        coordinate_right = coordinates_in[i_in_right]
        slices_left = [slice(None)] * n_dim
        slices_left[dimension_index] = i_in_left
        slices_right = [slice(None)] * n_dim
        slices_right[dimension_index] = i_in_right
        slices_out = [slice(None)] * n_dim
        slices_out[dimension_index] = i_out
        weight_left = (coordinate_right - coordinate_out) / (
            coordinate_right - coordinate_left
        )
        weight_right = 1 - weight_left
        quantity_out[slices_out] = (
            weight_left * quantity_in[slices_left]
            + weight_right * quantity_in[slices_right]
        )

    return quantity_out


def interpolate_multiple(
    quantity_in: torch.Tensor,
    grid_in: Grid,
    grid_out: Grid,
    *,
    dimension_labels: Sequence[str],
    search_method: str,
):
    """
    Interpolate linearly from `quantity_in` on `grid_in` to `grid_out`, where
    the grids differ only in the dimensions `dimension_labels`.
    For gridpoints outside the range of `grid_in[dimension_label]` we extrapolate
    using the two outermost points.
    Assuming ordered grid_in & grid_out in `dimension_labels`.
    If the grids are the same, a clone of quantity_in is returned.
    """

    # Interpolating dimension by dimension
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    current_grid_in_dimensions = copy.copy(grid_in.dimensions)
    current_grid_out_dimensions = copy.copy(grid_in.dimensions)
    current_quantity = quantity_in
    for dimension_label in dimension_labels:
        current_grid_out_dimensions[dimension_label] = grid_out[dimension_label]
        current_quantity = interpolate(
            current_quantity,
            Grid(current_grid_in_dimensions),
            Grid(current_grid_out_dimensions),
            dimension_label=dimension_label,
            search_method=search_method,
        )
        current_grid_in_dimensions = copy.copy(current_grid_out_dimensions)

    assert current_grid_out_dimensions == grid_out.dimensions

    return current_quantity
