import numpy as np
import torch

import kolpinn.mathematics as mathematics

# General
seed = 0
device = 'cuda'
si_dtype = torch.float64

# Training
max_n_training_steps = 2000
max_time = 60
min_loss = 0.00027
report_each = 500
Optimizer = torch.optim.AdamW
learn_rate = 1e-2 # Overwritten by a reloaded optimizer
loss_function = lambda x: x.transform(mathematics.complex_abs2)

# Model
loaded_parameters_index = 11
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

