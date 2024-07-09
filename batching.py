# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Optional
import copy
import random
import itertools

from .grids import Grid, Subgrid
from .quantities import QuantityDict, restrict_quantities


class Batcher:
    def __init__(
        self,
        q_full: QuantityDict,
        batch_dimensions: list,
        batch_sizes: list,
        additional_indices_dict: Optional[dict] = None,
    ):
        """
        The `additional_indices_dict` can restrict the accessible grid points.
        """

        if additional_indices_dict is None:
            self.additional_indices_dict = {}
        else:
            assert set(additional_indices_dict.keys()).isdisjoint(set(batch_dimensions))
            self.additional_indices_dict = additional_indices_dict

        assert len(batch_dimensions) == len(batch_sizes)

        self.q_full = q_full
        self.grid_full = q_full.grid
        self.batch_dimensions = batch_dimensions
        self.batch_sizes = batch_sizes

    def __call__(self):
        """
        Sample each dimension randomly without replacement and in order
        """

        # TEMP (inconsistent Grid/Subgrid)
        if len(self.batch_dimensions) == 0 and len(self.additional_indices_dict) == 0:
            # IDEA: separate TrivialBatcher class instead
            return copy.copy(self.q_full)

        indices_dict = copy.copy(self.additional_indices_dict)
        for batch_dimension, batch_size in zip(self.batch_dimensions, self.batch_sizes):
            all_indices = range(self.grid_full.dim_size[batch_dimension])
            indices = random.sample(all_indices, batch_size)
            indices.sort()
            indices_dict[batch_dimension] = indices

        return self._get_q(indices_dict)

    def _get_q(self, indices_dict: dict):
        grid = Subgrid(self.grid_full, indices_dict, copy_all=False)
        q = restrict_quantities(self.q_full, grid)

        return q

    def get_all(self):
        """
        Returns batches which together cover the whole grid
        without randomization.
        """

        indices_lists = []
        for batch_dimension, batch_size in zip(self.batch_dimensions,
                                               self.batch_sizes):
            indices_list = []
            dim_size = self.grid_full.dim_size[batch_dimension]
            for start_index in range(0, dim_size, batch_size):
                end_index = min(start_index + batch_size, dim_size)
                indices = list(range(start_index, end_index))
                indices_list.append(list(indices))
            indices_lists.append(indices_list)

        batches = []
        for indices_lists_batch in itertools.product(*indices_lists):
            indices_dict = copy.copy(self.additional_indices_dict)
            for batch_dimension, indices_list in zip(self.batch_dimensions,
                                                     indices_lists_batch):
                indices_dict[batch_dimension] = indices_list
            batches.append(self._get_q(indices_dict))

        return batches


def get_qs(batchers: dict[str, Batcher]):
    return dict((grid_name, batcher())
                for grid_name, batcher in batchers.items())
