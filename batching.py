# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from typing import Sequence, Dict, Callable, Tuple
import itertools
import random

from kolpinn.grids import Grid


def get_randomly_batched_indices_dict(
    grid: Grid,
    batch_sizes: Dict[str, int],
):
    """
    batch_sizes[dimension_name] = batch size of this dimension
    """

    indices_dict: Dict[str, Sequence[int]] = {}
    for batch_dimension, batch_size in batch_sizes.items():
        all_indices = range(grid.dim_size[batch_dimension])
        indices = random.sample(all_indices, batch_size)
        indices.sort()
        indices_dict[batch_dimension] = indices

    return indices_dict


def get_equispaced_batched_indices_dict(
    grid: Grid,
    batch_sizes: Dict[str, int],
    *,
    randomize: bool,
):
    """
    batch_sizes[dimension_name] = approximate batch size of this dimension
    If `randomize` a random starting point is chosen, otherwise it's always the same one
    """

    indices_dict: Dict[str, Sequence[int]] = {}
    for batch_dimension, batch_size in batch_sizes.items():
        dim_size = grid.dim_size[batch_dimension]
        step = dim_size // batch_size
        assert step > 0, f"{batch_dimension}: {dim_size}, {batch_size}"
        start = random.randrange(0, step) if randomize else step // 2
        indices = range(start, dim_size, step)
        indices_dict[batch_dimension] = indices

    return indices_dict


def get_bounds_batched_indices_dict(
    grid: Grid,
    batch_bounds: Dict[str, Tuple[float, float]],
):
    """
    batch_bounds[dimension_name] = (minimum value, maximum value) of this dimension
    The batch will consist of the values satisfying
    minimum_value <= value <= maximum_value
    """

    indices_dict: Dict[str, Sequence[int]] = {}
    for batch_dimension, bounds in batch_bounds.items():
        coords = grid[batch_dimension]
        # start/stop: first/last index within the bounds
        start = 0
        while start < len(coords) and coords[start] < bounds[0]:
            start += 1
        stop = start
        while stop < len(coords) and coords[stop] <= bounds[1]:
            stop += 1

        indices = range(start, stop)
        indices_dict[batch_dimension] = indices

    return indices_dict


def get_combined_batched_indices_dict(
    grid: Grid,
    batched_indices_dict_fns: Sequence[
        Callable[[Grid, bool], Dict[str, Sequence[int]]]
    ],
    *,
    randomize: bool,
):
    """
    Combine multiple batching functions.
    batched_indices_dict_fns should map (grid, randomize) to an indices_dict.
    """

    full_indices_dict: Dict[str, Sequence[int]] = {}
    for batches_indices_dict_fn in batched_indices_dict_fns:
        indices_dict = batches_indices_dict_fn(grid, randomize)
        for label, indices in indices_dict.items():
            if label not in full_indices_dict:
                full_indices_dict[label] = indices
                continue
            full_indices_dict[label] = list(
                filter(lambda x: x in indices, full_indices_dict[label])
            )

    return full_indices_dict


def get_batched_indices_dicts_covering_all(
    grid: Grid,
    batch_sizes: Dict[str, int],
) -> Sequence[Dict[str, Sequence[int]]]:
    """
    Returns batches which together cover the whole grid
    without randomization.
    """
    indices_list_dict: Dict[str, Sequence[Sequence[int]]] = {}
    for batch_dimension, batch_size in batch_sizes.items():
        indices_list = []
        dim_size = grid.dim_size[batch_dimension]
        for start_index in range(0, dim_size, batch_size):
            end_index = min(start_index + batch_size, dim_size)
            indices = range(start_index, end_index)
            indices_list.append(indices)
        indices_list_dict[batch_dimension] = indices_list

    indices_dicts: Sequence[Dict[str, Sequence[int]]] = [
        dict(zip(indices_list_dict.keys(), indices_list))
        for indices_list in itertools.product(*indices_list_dict.values())
    ]
    return indices_dicts
