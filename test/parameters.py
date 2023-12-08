import numpy as np
import torch

# General
seed = 0
device = 'cuda'
si_dtype = torch.float64

# Training
n_training_steps = 2000
report_each = 500
Optimizer = torch.optim.AdamW
learn_rate = 1e-2 # Overwritten by a reloaded optimizer
loss_function = lambda x: (x**2).mean()

# Model
loaded_parameters_index = 1
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

