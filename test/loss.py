import torch

from kolpinn import mathematics
from kolpinn.grid_quantities import Grid, restrict
from kolpinn.model import FunctionModel

import parameters as params


def get_derivative_loss(q, *, q_full, with_grad):
    """ q_full: unbatched """

    y_dx_full = mathematics.grad(
        q['y'],
        q_full['x'],
        retain_graph=True,
        create_graph=with_grad,
    )
    y_dx = restrict(y_dx_full, q.grid)
    residual = y_dx - q['cos(x)']

    return params.loss_function(residual)
derivative_loss_model = FunctionModel(get_derivative_loss, q_full=None, with_grad=True)

def get_boundary_loss(q, *, with_grad):
    residual = q['y'] - 1

    return params.loss_function(residual)
boundary_loss_model = FunctionModel(get_boundary_loss, with_grad=True)

def get_zero_loss(q, *, q_full, with_grad):
    y_dx_full = mathematics.grad(
        q['y'],
        q_full['x'],
        retain_graph=True,
        create_graph=with_grad,
    )
    y_dx = restrict(y_dx_full, q.grid)
    residual = y_dx - 1

    return params.loss_function(residual)
zero_loss_model = FunctionModel(get_zero_loss, q_full=None, with_grad=True)
