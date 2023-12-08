from typing import Callable, Optional
import torch

from . import grid_quantities
from .grid_quantities import Grid, Quantity, QuantityDict
from .batching import Batcher
from .model import Model, get_extended_q


def get_losses(
        models: dict[str,Model],
        q: QuantityDict,
        loss_functions: dict[str,Callable],
        quantities_requiring_grad_labels: list[str],
        *,
        models_require_grad: bool,
        loss_quantities: Optional[dict[str,Quantity]] = None,
    ):
    """
    models[model_name] = model
    loss_functions[loss_name] = loss_function

    Returns
        loss_quantities[loss_name] = loss_quantity
    """

    if loss_quantities is None:
        loss_quantities = {}

    extended_q = get_extended_q(
        q,
        models=models,
        models_require_grad = models_require_grad,
        quantities_requiring_grad_labels = quantities_requiring_grad_labels,
    )

    for loss_name, loss_function in loss_functions.items():
        loss_quantities[loss_name] = loss_function(
            extended_q,
            with_grad = models_require_grad,
        )

    return loss_quantities


def get_batch_losses(
        models: dict[str,Model],
        batchers: dict[str,Batcher],
        loss_functions: dict[str,dict[str,Callable]],
        quantities_requiring_grad_dict: dict[str,list[str]],
        *,
        models_require_grad: bool,
    ):
    """
    models[model_name] = model
    batchers[batcher_name] = batcher
    loss_functions[batcher_name] = dict[loss_name,loss_fn]
    quantities_requiring_grad_dict[batcher_name] = ['q1_name', 'q2_name', ...]

    Returns
        loss_quantities[loss_name] = loss_quantity
    """

    loss_quantities: dict[str,Quantity] = {}
    for batcher_name, batcher in batchers.items():
        q = batcher()
        get_losses(
            models,
            q,
            loss_functions[batcher_name],
            quantities_requiring_grad_dict[batcher_name],
            models_require_grad = models_require_grad,
            loss_quantities = loss_quantities,
        )

    return loss_quantities


def get_full_losses(
        models: dict[str,Model],
        batchers: dict[str,Batcher],
        loss_functions: dict[str,dict[str,Callable]],
        quantities_requiring_grad_dict: dict[str,list[str]],
        *,
        models_require_grad: bool,
    ):
    """
    Like `get_batch_losses`, but evaluating the loss on the full grids.
    The resulting `loss_quantities` depend on the batched dimensions.
    """

    loss_quantities: dict[str,Quantity] = {}
    for batcher_name, batcher in batchers.items():
        # Get losses per batch
        loss_names = loss_functions[batcher_name].keys()
        loss_quantities_batched = dict((loss_name, [])
                                       for loss_name in loss_names)
        for q in batcher.get_all():
            batch_loss_quantities = get_losses(
                models,
                q,
                loss_functions[batcher_name],
                quantities_requiring_grad_dict[batcher_name],
                models_require_grad = models_require_grad,
            )
            for loss_name, loss_quantity in batch_loss_quantities.items():
                loss_quantities_batched[loss_name].append(loss_quantity)

        # Combine the losses
        for loss_name, loss_sub_quantities in loss_quantities_batched.items():
            loss_quantities[loss_name] = grid_quantities.combine_quantity(
                loss_sub_quantities,
                batcher.grid_full,
            )

    return loss_quantities
