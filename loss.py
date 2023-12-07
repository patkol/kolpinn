from typing import Callable, Optional
import torch

from . import grid_quantities
from .grid_quantities import Grid, Quantity, QuantityDict
from .batching import Batcher
from .model import Model


def get_losses(
        models: dict[str,Model],
        q: QuantityDict,
        loss_functions: dict[str,Callable],
        quantities_requiring_grad_labels: list[str],
        *,
        models_require_grad: bool,
        loss_quantities: Optional[dict[str,Quantity]] = None,
        diffable_quantities: Optional[dict[str,Callable]] = None,
    ):
    """
    models[model_name] = model
    loss_functions[loss_name] = loss_function

    Returns
        loss_quantities[loss_name] = loss_quantity
    """

    if loss_quantities is None:
        loss_quantities = {}
    if diffable_quantities is None:
        diffable_quantities = {}

    unexpanded_quantities = {}
    for quantity_requiring_grad_label in quantities_requiring_grad_labels:
        unexpanded_quantity = q[quantity_requiring_grad_label]
        unexpanded_quantities[quantity_requiring_grad_label] = unexpanded_quantity
        q[quantity_requiring_grad_label] = Quantity(
            unexpanded_quantity.get_expanded_values(),
            unexpanded_quantity.grid,
        )
        q[quantity_requiring_grad_label].set_requires_grad(True)

    for model_name, model in models.items():
        model.set_requires_grad(models_require_grad)
        q[model_name] = model.apply(q)

    for diffable_quantity_name, diffable_quantity_function in diffable_quantities.items():
        q[diffable_quantity_name] = diffable_quantity_function(q)

    for loss_name, loss_function in loss_functions.items():
        loss_quantities[loss_name] = loss_function(
            q,
            with_grad = models_require_grad,
        )

    for quantity_requiring_grad_label in quantities_requiring_grad_labels:
        #q[quantity_requiring_grad_label].set_requires_grad(False)
        q[quantity_requiring_grad_label] = unexpanded_quantities[quantity_requiring_grad_label]

    return loss_quantities


def get_batch_losses(
        models: dict[str,Model],
        batchers: dict[str,Batcher],
        loss_functions: dict[str,dict[str,Callable]],
        quantities_requiring_grad_dict: dict[str,list[str]],
        *,
        models_require_grad: bool,
        diffable_quantities: Optional[dict[str,Callable]] = None,
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
            diffable_quantities = diffable_quantities,
        )

    return loss_quantities


def get_full_losses(
        models: dict[str,Model],
        batchers: dict[str,Batcher],
        loss_functions: dict[str,dict[str,Callable]],
        quantities_requiring_grad_dict: dict[str,list[str]],
        *,
        models_require_grad: bool,
        diffable_quantities: Optional[dict[str,Callable]] = None,
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
                diffable_quantities = diffable_quantities,
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
