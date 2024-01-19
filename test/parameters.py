import numpy as np
import torch

import kolpinn.mathematics as mathematics

# General
seed = 0
device = 'cuda'
si_dtype = torch.float64

# Training
max_n_training_steps = 0
max_time = 300
min_loss = 50e-6
report_each = 200
Optimizer = torch.optim.AdamW
optimizer_kwargs = {'lr': 1e-2} # Overwritten by a reloaded optimizer
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = {'factor': 0.2, 'patience': 5}
loss_function = mathematics.complex_abs2

# Model
loaded_parameters_index = 43
# Whether to use the state of the saved optimizer (possibly overwriting optimizer_kwargs)
load_optimizer = True
load_scheduler = True
n_neurons_per_hidden_layer = 10
n_hidden_layers = 5
activation_function = torch.nn.SiLU()
model_dtype = torch.float32

# Coordinates
X_LEFT = -2 * np.pi
X_RIGHT = 2 * np.pi
N_x_training = 1000
N_x_validation = 170
batch_size_x = 100

