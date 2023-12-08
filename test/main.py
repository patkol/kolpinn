#!/usr/bin/env python3

"""
Solving dy/dx = c*cos(x), x in (-2pi, +2pi), y = 1 on Boundaries, y'(0) = 1.
Exact solution: y(x) = sin(x) + 1, c = 1
"""

import pdb
import random
import time
import numpy as np
import torch

from kolpinn.io import get_next_parameters_index
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, Quantity
from kolpinn.batching import Batcher
from kolpinn.model import ConstModel, SimpleNNModel
from kolpinn.training import Trainer

import parameters as params
import physics
import loss
from visualization import visualize


# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_dtype)


# Model

y_model = SimpleNNModel(
    ['x'],
    {'x': lambda x, y: x},
    lambda y, q: y,
    params.activation_function,
    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
    n_hidden_layers = params.n_hidden_layers,
    model_dtype = params.model_dtype,
    output_dtype = params.si_dtype,
    device = params.device,
)
c_model = ConstModel(
    3,
    model_dtype = params.model_dtype,
    output_dtype = params.si_dtype,
)
models = {'y': y_model, 'c': c_model}


# Coordinates

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

qs_training = physics.quantities_factory.get_quantities_dict(grids_training)
qs_validation = physics.quantities_factory.get_quantities_dict(grids_validation)

batchers_training = {
    'bulk': Batcher(qs_training['bulk'], grids_training['bulk'], ['x'], [params.batch_size_x]),
    'left': Batcher(qs_training['left'], grids_training['left'], [], []),
    'right': Batcher(qs_training['right'], grids_training['right'], [], []),
    'zero': Batcher(qs_training['zero'], grids_training['zero'], [], []),
}
batchers_validation = {
    'bulk': Batcher(qs_validation['bulk'], grids_validation['bulk'], ['x'], [1]),
    'left': Batcher(qs_validation['left'], grids_validation['left'], [], []),
    'right': Batcher(qs_validation['right'], grids_validation['right'], [], []),
    'zero': Batcher(qs_validation['zero'], grids_validation['zero'], [], []),
}


# Training

trainer = Trainer(
    models,
    batchers_training,
    batchers_validation,
    loss.loss_functions,
    loss.quantities_requiring_grad_dict,
    params.Optimizer,
    params.learn_rate,
    saved_parameters_index = get_next_parameters_index(),
    name = 'trainer',
)
trainer.load(params.loaded_parameters_index)


if __name__ == "__main__":
    trainer.train(params.n_training_steps, params.report_each)
    visualize(batchers_validation, models, trainer)
