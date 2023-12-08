import torch

import parameters as params

def get_derivative_loss(q, *, with_grad):
    # q['x'] should require grad

    y_dx = q['y'].get_grad(
        q['x'],
        retain_graph=with_grad,
        create_graph=with_grad,
    )
    residual = y_dx - q['c'] * q['cos(x)']

    return params.loss_function(residual)

def get_boundary_loss(q, *, with_grad):
    residual = q['y'] - 1

    return params.loss_function(residual)

def get_zero_loss(q, *, with_grad):
    y_dx = q['y'].get_grad(
        q['x'],
        retain_graph=with_grad,
        create_graph=with_grad,
    )
    residual = y_dx - 1

    return params.loss_function(residual)

loss_functions = {
    'bulk': {'derivative_loss': get_derivative_loss},
    'left': {'left_boundary_loss': get_boundary_loss},
    'right': {'right_boundary_loss': get_boundary_loss},
    'zero': {'zero_loss': get_zero_loss},
}
quantities_requiring_grad_dict = {
    'bulk': ['x'],
    'left': [],
    'right': [],
    'zero': ['x'],
}
