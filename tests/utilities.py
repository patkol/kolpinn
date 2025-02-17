# Copyright (c) 2025 ETH Zurich, Patrice Kolb


import torch

from kolpinn import grids


def get_random_tensor(*, size, seed, **kwargs):
    torch.manual_seed(seed)
    return torch.rand(size, **kwargs)


def get_dependent_tensors(*, multiplier, size, seed, **kwargs):
    a = get_random_tensor(size=size, seed=seed, requires_grad=True, **kwargs)
    b = multiplier * a
    return a, b


def get_random_dimensions(sizes, *, seed: int, **kwargs):
    return dict(
        (key, get_random_tensor(size=(N,), seed=seed + i, **kwargs))
        for i, (key, N) in enumerate(sizes.items())
    )


def get_random_grid(sizes, *, seed: int, **kwargs):
    dimensions = get_random_dimensions(sizes, seed=seed, **kwargs)
    return grids.Grid(dimensions)
