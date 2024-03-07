#!/usr/bin/env python3

"""
Solving y = cos(x), x in (-pi, +pi)
Exact solution: y(x) = cos(x)
NN approximation: y(x) = cos(x)
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
from visualization import visualize


# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_dtype)


# Grids

conditions_dicts = {
    'bulk': {},
}
grid_training = Grid({
    'x': torch.linspace(params.X_LEFT, params.X_RIGHT, params.N_x_training),
})
grids_training = grid_training.get_subgrids(conditions_dicts, copy_all=True)
grid_validation = Grid({
    'x': torch.linspace(params.X_LEFT, params.X_RIGHT, params.N_x_validation),
})
grids_validation = grid_validation.get_subgrids(conditions_dicts, copy_all=True)


# Models

## Constant models
cos_model = FunctionModel(lambda q: torch.cos(q['x']))

const_models = []
const_models.append(get_multi_model(cos_model, 'cos(x)', 'bulk'))

## Parameter-dependent models
y_model = SimpleNNModel(
    ['x'],
    params.activation_function,
    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
    n_hidden_layers = params.n_hidden_layers,
    model_dtype = params.model_dtype,
    output_dtype = params.si_dtype,
    device = params.device,
)
loss_model = FunctionModel(lambda q, *, with_grad: params.loss_function(q['y'] - q['cos(x)']))

models = []
models.append(get_multi_model(y_model, 'y', 'bulk'))
models.append(get_multi_model(loss_model, 'loss', 'bulk'))
used_losses = {
    'bulk': ['loss'],
}

trained_models_labels = ['y']
quantities_requiring_grad = {
    'bulk': ['x'],
}


# Constant quantities
qs_training = model.get_qs(grids_training, const_models, quantities_requiring_grad)
qs_validation = model.get_qs(grids_validation, const_models, quantities_requiring_grad)


# Batchers
batchers_training = {
    'bulk': Batcher(qs_training['bulk'], grids_training['bulk'], ['x'], [params.batch_size_x]),
}
batchers_validation = {
    'bulk': Batcher(qs_validation['bulk'], grids_validation['bulk'], ['x'], [params.batch_size_x]),
}


# Trainer
trainer = Trainer(
    models = models,
    batchers_training = batchers_training,
    batchers_validation = batchers_validation,
    used_losses = used_losses,
    quantities_requiring_grad_dict = quantities_requiring_grad,
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
