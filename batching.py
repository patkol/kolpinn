# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Sequence, Dict
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
        start = random.randrange(0, step) if randomize else step // 2
        indices = range(start, dim_size, step)
        indices_dict[batch_dimension] = indices

    return indices_dict


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
