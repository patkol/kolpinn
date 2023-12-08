from typing import Optional, Callable
import copy
import random
import itertools
import torch

from . import grid_quantities
from .grid_quantities import Grid, Subgrid, Quantity, QuantityDict, restrict_quantities


class Batcher:
    def __init__(
            self,
            q_full: QuantityDict,
            grid_full: Grid,
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
        self.grid_full = grid_full
        self.batch_dimensions = batch_dimensions
        self.batch_sizes = batch_sizes

    def __call__(self):
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
        batches = [(q1, grid1), ...]
        """

        indices_lists = []
        for batch_dimension, batch_size in zip(self.batch_dimensions, self.batch_sizes):
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
            for batch_dimension, indices_list in zip(self.batch_dimensions, indices_lists_batch):
                indices_dict[batch_dimension] = indices_list
            batches.append(self._get_q(indices_dict))

        return batches

    # TODO: Code duplication in loss.get_losses (There are differences: Batching in model eval)
    def get_extended_q(
            self,
            models: dict,
            quantities_requiring_grad_labels: list[str] = None,
        ):
        """
        Get the quantities including the evaluated models.
        """

        if quantities_requiring_grad_labels is None:
            quantities_requiring_grad_labels = []

        extended_q = copy.copy(self.q_full)

        unexpanded_quantities = {}
        for quantity_requiring_grad_label in quantities_requiring_grad_labels:
            unexpanded_quantity = extended_q[quantity_requiring_grad_label]
            unexpanded_quantities[quantity_requiring_grad_label] = unexpanded_quantity
            extended_q[quantity_requiring_grad_label] = Quantity(
                unexpanded_quantity.get_expanded_values(),
                unexpanded_quantity.grid,
            )
            extended_q[quantity_requiring_grad_label].set_requires_grad(True)

        for model_name, model in models.items():
            assert not model_name in extended_q, model_name
            model_batches = []
            for q in self.get_all():
                model_batch = model.apply(q)
                model_batches.append(model_batch)
            extended_q[model_name] = grid_quantities.combine_quantity(
                model_batches,
                self.grid_full,
            )

        for quantity_requiring_grad_label in quantities_requiring_grad_labels:
            extended_q[quantity_requiring_grad_label] = unexpanded_quantities[quantity_requiring_grad_label]

        return extended_q

