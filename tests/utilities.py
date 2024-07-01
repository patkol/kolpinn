# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import torch


def get_random_tensor(*, size, seed, **kwargs):
    torch.manual_seed(seed)
    return torch.rand(size, **kwargs)

def get_dependent_tensors(*, multiplier, size, seed, **kwargs):
    a = get_random_tensor(size=size, seed=seed, requires_grad=True, **kwargs)
    b = multiplier * a
    return a, b



