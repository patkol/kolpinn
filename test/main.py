#!/usr/bin/env python3

"""
Solving dy/dx = cos(x), x in (-2pi, +2pi), y = 1 on Boundaries, y'(0) = 1
Exact solution: y(x) = sin(x) + 1
NN approximation: y(x) = f(x) * NN(xs, params), f(x) = exp(0.1x) is the (intentionally bad) guess
"""

import pdb
import random
import time
import numpy as np
import torch

from kolpinn.io import get_next_parameters_index
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid
from kolpinn.batching import Batcher
from kolpinn import model
from kolpinn.model import ConstModel, FunctionModel, SimpleNNModel, \
                          TransformedModel, get_multi_model
from kolpinn.training import Trainer

import parameters as params
import loss
from visualization import visualize


# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_dtype)


# Grids

conditions_dicts = {
    'bulk': {},
    'left': {'x': lambda x: x==params.X_LEFT},
    'right': {'x': lambda x: x==params.X_RIGHT},
}
grid_training = Grid({
    'x': torch.linspace(params.X_LEFT, params.X_RIGHT, params.N_x_training),
})
grids_training = grid_training.get_subgrids(conditions_dicts, copy_all=True)
grids_training['zero'] = Grid({
    'x': torch.tensor([0.]),
})
grid_validation = Grid({
    'x': torch.linspace(params.X_LEFT, params.X_RIGHT, params.N_x_validation),
})
grids_validation = grid_validation.get_subgrids(conditions_dicts, copy_all=True)
grids_validation['zero'] = Grid({
    'x': torch.tensor([0.]),
})

grid_names = ['bulk', 'left', 'right', 'zero']


# Models

## Constant models
cos_model = FunctionModel(lambda q: torch.cos(q['x']))
f_model = FunctionModel(lambda q: q['x'] + 1)#torch.exp(0.1 * q['x']))

const_models = []
const_models.append(get_multi_model(cos_model, 'cos(x)', 'bulk'))
for grid_name in grid_names:
    const_models.append(get_multi_model(f_model, 'f', grid_name))

## Parameter-dependent models
y_output_model = SimpleNNModel(
    ['x'],
    params.activation_function,
    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
    n_hidden_layers = params.n_hidden_layers,
    model_dtype = params.model_dtype,
    output_dtype = params.si_dtype,
    device = params.device,
)
y_model = TransformedModel(
    y_output_model,
    output_transformation = lambda o, q: q['f'] * o,
)

models = []
for grid_name in grid_names:
    models.append(get_multi_model(y_output_model, 'y_output', grid_name))
    models.append(get_multi_model(y_model, 'y', grid_name))

### Add the losses
models.append(get_multi_model(loss.derivative_loss_model, 'derivative_loss', 'bulk'))
for grid_name in ['left', 'right']:
    models.append(get_multi_model(loss.boundary_loss_model, grid_name + '_loss', grid_name))
models.append(get_multi_model(loss.zero_loss_model, 'zero_loss', 'zero'))
used_losses = {
    'bulk': ['derivative_loss'],
    'left': ['left_loss'],
    'right': ['right_loss'],
    'zero': ['zero_loss'],
}

trained_models_labels = ['y_output']
quantities_requiring_grad = {
    'bulk': ['x'],
    'zero': ['x'],
}


# Constant quantities
qs_training = model.get_qs(grids_training, const_models, quantities_requiring_grad)
qs_validation = model.get_qs(grids_validation, const_models, quantities_requiring_grad)


# Batchers
batchers_training = {
    'bulk': Batcher(qs_training['bulk'], grids_training['bulk'], ['x'], [params.batch_size_x]),
    'left': Batcher(qs_training['left'], grids_training['left'], [], []),
    'right': Batcher(qs_training['right'], grids_training['right'], [], []),
    'zero': Batcher(qs_training['zero'], grids_training['zero'], [], []),
}
batchers_validation = {
    'bulk': Batcher(qs_validation['bulk'], grids_validation['bulk'], ['x'], [params.batch_size_x]),
    'left': Batcher(qs_validation['left'], grids_validation['left'], [], []),
    'right': Batcher(qs_validation['right'], grids_validation['right'], [], []),
    'zero': Batcher(qs_validation['zero'], grids_validation['zero'], [], []),
}


# Trainer
trainer = Trainer(
    models = models,
    batchers_training = batchers_training,
    batchers_validation = batchers_validation,
    used_losses = used_losses,
    trained_models_labels = trained_models_labels,
    Optimizer = params.Optimizer,
    optimizer_kwargs = params.optimizer_kwargs,
    Scheduler = params.Scheduler,
    scheduler_kwargs = params.scheduler_kwargs,
    saved_parameters_index = get_next_parameters_index(),
    name = 'trainer',
)
trainer.load(
    params.loaded_parameters_index,
    load_optimizer = params.load_optimizer,
    load_scheduler = params.load_scheduler,
)


if __name__ == "__main__":
    trainer.train(
        report_each = params.report_each,
        max_n_steps = params.max_n_training_steps,
        max_time = params.max_time,
        min_loss = params.min_loss,
    )
    visualize(trainer)
