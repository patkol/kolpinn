# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import math
import torch

import kolpinn.mathematics as mathematics

import utilities


def test_transform():
    x = 0.8
    input_range = (0.5, 1)
    output_range = (-1, 4)

    y = mathematics.transform(x, input_range, output_range)

    assert math.isclose(y, 2)

def test_complex_mse_of_equal_tensors_vanishes():
    a = utilities.get_random_tensor(size=(2,3,4), seed=0, dtype=torch.complex64)

    mse = mathematics.complex_mse(a, a)

    assert math.isclose(mse, 0)

def test_complex_mse_of_ones_is_one():
    size = (2,3,4)
    dtype = torch.complex128
    a = torch.ones(size, dtype=dtype)
    b = torch.zeros(size, dtype=dtype)

    mse = mathematics.complex_mse(a, b)

    assert math.isclose(mse, 1)

def test_grad_real_tensor_real_multiplier():
    size = (2,3,4)
    dtype = torch.float64
    multiplier = utilities.get_random_tensor(
        size=size,
        seed=0,
        dtype=dtype,
    )
    a, b = utilities.get_dependent_tensors(
        multiplier=multiplier, size=size, seed=1, dtype=dtype,
    )

    grad = mathematics.grad(b, a)

    assert torch.all(torch.isclose(grad, multiplier))

def test_grad_real_tensor_complex_multiplier():
    size = (2,3,4)
    multiplier = utilities.get_random_tensor(
        size=size,
        seed=0,
        dtype=torch.complex64,
    )
    a, b = utilities.get_dependent_tensors(
        multiplier=multiplier, size=size, seed=1, dtype=torch.float32,
    )

    grad = mathematics.grad(b, a)

    assert torch.all(torch.isclose(grad, multiplier))

def test_generalized_cartesian_prod_no_tensors():
    prod = mathematics.generalized_cartesian_prod()

    assert prod.size() == (0, 0)

def test_generalized_cartesian_prod_one_tensor():
    tensor = utilities.get_random_tensor(
        size=(5,),
        seed=0,
        dtype=torch.complex128,
    )

    prod = mathematics.generalized_cartesian_prod(tensor)

    assert prod.size() == (5,1)
    assert torch.all(prod[:,0] == tensor)

def test_exchange_dims_outer_dims():
    tensor = utilities.get_random_tensor(
        size=(2,3,4),
        seed=0,
        dtype=torch.complex64,
    )

    exchanged_tensor = mathematics.exchange_dims(tensor, 0, 2)
    doubly_exchanged_tensor = mathematics.exchange_dims(exchanged_tensor, 0, 2)
    triply_exchanged_tensor = mathematics.exchange_dims(doubly_exchanged_tensor, 2, 0)

    assert exchanged_tensor.size() == (4,3,2)
    assert torch.all(doubly_exchanged_tensor == tensor)
    assert torch.all(triply_exchanged_tensor == exchanged_tensor)

def test_exchange_dims_with_themselves():
    tensor = utilities.get_random_tensor(
        size=(2,3,4),
        seed=0,
        dtype=torch.complex128,
    )

    exchanged_tensor = mathematics.exchange_dims(tensor, 1, 1)

    assert torch.all(exchanged_tensor == tensor)

def test_remove_duplicates():
    list_ = [1, 'fish', 1]

    cleaned_list = mathematics.remove_duplicates(list_)

    assert len(cleaned_list) == 2
    assert set(cleaned_list) == set([1, 'fish'])

def test_remove_duplicates_without_duplicates():
    list_ = [1, 'fish', 3]

    cleaned_list = mathematics.remove_duplicates(list_)

    assert cleaned_list == list_

def test_remove_duplicates_of_empty_list():
    list_ = []

    cleaned_list = mathematics.remove_duplicates(list_)

    assert cleaned_list == list_

def test_remove_duplicates_of_equal_objects():
    """
    remove_duplicates only removes duplicates in the sense of 'is'
    """

    list_ = [[1,2], 'fish', [1,2]]

    cleaned_list = mathematics.remove_duplicates(list_)

    assert cleaned_list == list_

def test_expand():
    tensor = utilities.get_random_tensor(
        size=(2,3,1),
        seed=0,
        dtype=torch.complex64,
    )
    shape_target = (1,2,3,7,0,1,7)
    indices = [1,2,5]

    expanded_tensor = mathematics.expand(tensor, shape_target, indices)
    squeezed_expanded_tensor = torch.squeeze(expanded_tensor)
    squeezed_tensor = torch.squeeze(tensor)

    assert expanded_tensor.size() == (1,2,3,1,1,1,1)
    assert torch.all(squeezed_expanded_tensor == squeezed_tensor)

def test_expand_to_same():
    size = (2,3,1,5)
    tensor = utilities.get_random_tensor(
        size=size,
        seed=1,
        dtype=torch.complex128,
    )
    shape_target = size
    indices = [i for i in range(len(size))]

    expanded_tensor = mathematics.expand(tensor, shape_target, indices)

    assert torch.all(expanded_tensor == tensor)

def test_expand_empty():
    tensor = utilities.get_random_tensor(
        size=(),
        seed=0,
        dtype=torch.complex64,
    )
    shape_target = (1,2,3,7,1,7)
    indices = []

    expanded_tensor = mathematics.expand(tensor, shape_target, indices)

    assert expanded_tensor.size() == (1,) * len(shape_target)
    assert torch.all(expanded_tensor == tensor.item())

def test_interleave_equal_lengths():
    a = utilities.get_random_tensor(
        size=(2,3,4),
        seed=0,
        dtype=torch.complex64,
    )
    b = utilities.get_random_tensor(
        size=(2,3,4),
        seed=1,
        dtype=torch.float64,
    )

    interleaved = mathematics.interleave(a, b, dim=1)

    assert torch.all(interleaved[:,::2,:] == a)
    assert torch.all(interleaved[:,1::2,:] == b)

def test_interleave_different_lengths():
    a = utilities.get_random_tensor(
        size=(3,3,4),
        seed=0,
        dtype=torch.complex64,
    )
    b = utilities.get_random_tensor(
        size=(2,3,4),
        seed=1,
        dtype=torch.complex64,
    )

    interleaved = mathematics.interleave(a, b, dim=0)

    assert torch.all(interleaved[::2,:,:] == a)
    assert torch.all(interleaved[1::2,:,:] == b)

def test_interleave_with_zero_length():
    a = utilities.get_random_tensor(
        size=(2,3,1),
        seed=0,
        dtype=torch.complex64,
    )
    b = utilities.get_random_tensor(
        size=(2,3,0),
        seed=1,
        dtype=torch.complex64,
    )

    interleaved = mathematics.interleave(a, b, dim=2)

    assert torch.all(interleaved == a)

def test_interleave_nothing():
    a = utilities.get_random_tensor(
        size=(0,3,4),
        seed=0,
        dtype=torch.complex64,
    )
    b = utilities.get_random_tensor(
        size=(0,3,4),
        seed=1,
        dtype=torch.complex64,
    )

    interleaved = mathematics.interleave(a, b, dim=0)

    assert torch.all(interleaved == a)
    assert torch.all(interleaved == b)
