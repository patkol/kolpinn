# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Sequence, Dict
import random

from kolpinn.grids import Grid, Subgrid


def get_random_subgrid(
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

    return Subgrid(grid, indices_dict, copy_all=False)
