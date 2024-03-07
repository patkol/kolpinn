import numpy as np
import torch

import kolpinn.mathematics as mathematics

# General
seed = 0
device = 'cuda'
si_dtype = torch.float64

# Training
max_n_training_steps = None
max_time = 3
min_loss = None
report_each = 250
Optimizer = torch.optim.AdamW
optimizer_kwargs = {'lr': 1e-3} # Overwritten by a reloaded optimizer
Scheduler = None
scheduler_kwargs = None
loss_function = mathematics.complex_abs2

# Model
loaded_parameters_index = None
# Whether to use the state of the saved optimizer (possibly overwriting optimizer_kwargs)
load_optimizer = True
load_scheduler = True
n_neurons_per_hidden_layer = 10
n_hidden_layers = 5
activation_function = torch.nn.SiLU()
model_dtype = torch.float64

# Coordinates
X_LEFT = -np.pi
X_RIGHT = np.pi
N_x = 100
N_x_training = N_x
N_x_validation = N_x
batch_size_x = 100
