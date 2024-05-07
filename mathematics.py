# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import copy

import torch


identity = lambda x: x

def transform(x, input_range, output_range):
    """Transform ''x'' in ''input_range'' into the ''output_range''"""
    scale_factor = ((output_range[1] - output_range[0]) /
                    (input_range[1]  - input_range[0]))
    return (x - input_range[0]) * scale_factor + output_range[0]

def complex_abs2(a):
    if a.dtype in (torch.float16, torch.float32, torch.float64):
        return a**2

    return torch.real(a)**2 + torch.imag(a)**2

def complex_mse(a, b):
    """nn.MSELoss that works for complex numbers"""
    return complex_abs2(b-a).mean()

def complex_grad(outputs: torch.Tensor, inputs: torch.Tensor, *args, **kwargs):
    """
    Derives the real and imaginary parts of 'outputs' seperately.

    ''torch.autograd.grad'' calculates the Wirtinger derivative instead.
    There is no support for sequences of tensors right now.
    """

    grad_real, = torch.autograd.grad(
        torch.real(outputs),
        inputs,
        *args,
        **kwargs,
    )
    grad_imag, = torch.autograd.grad(
        torch.imag(outputs),
        inputs,
        *args,
        **kwargs,
    )

    return grad_real + 1j * grad_imag

def grad(output: torch.Tensor, input_: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Possibly complex gradient with ones as the `grad_outputs`
    kwargs example: retain_graph=True, create_graph=True
    """

    grad_function = (complex_grad
                     if torch.is_complex(output)
                     else torch.autograd.grad)

    if output.dtype == torch.complex64:
        grad_outputs_dtype = torch.float32
    elif output.dtype == torch.complex128:
        grad_outputs_dtype = torch.float64
    else:
        grad_outputs_dtype = output.dtype

    grad_tensor = grad_function(
        outputs = output,
        inputs = input_,
        grad_outputs = torch.ones_like(output, dtype=grad_outputs_dtype),
        **kwargs,
    )

    if grad_function is torch.autograd.grad:
        grad_tensor = grad_tensor[0]

    return grad_tensor

def generalized_cartesian_prod(*tensors: torch.Tensor):
    """ Generalized to the cases where only one or no tensor is provided. """

    if len(tensors) == 0:
        return torch.zeros((0,0))

    if len(tensors) == 1:
        tensor = tensors[0]
        assert len(tensor.size()) == 1
        return tensor.reshape((-1,1))

    return torch.cartesian_prod(*tensors)

def exchange_dims(tensor: torch.Tensor, dim_1, dim_2):
    permutation = list(range(len(tensor.size())))
    permutation[dim_1], permutation[dim_2] = \
        permutation[dim_2], permutation[dim_1]
    permuted_tensor = tensor.permute(permutation)

    return permuted_tensor

def remove_duplicates(l: list):
    """
    Return a list excluding exact duplicates in the sense of 'is'.
    """

    out = []
    for i in range(len(l)):
        duplicate = False
        for j in range(i):
            if l[j] is l[i]:
                duplicate = True
                break
        if not duplicate:
            out.append(l[i])

    return out

def expand(tensor_in: torch.Tensor, shape_target, indices_in):
    """
    Expand tensor to the `shape_target`.
    The dimension `in_dim` in the input tensor will
    be dimension `indices[in_dim]` in the output tensor.
    The remaining dimensions of the output tensor will be singletons, independent
    of the value in `shape_target`.
    indices_in must be in ascending order.
    """

    if len(tensor_in.size()) == 0:
        assert indices_in == []
        shape_out = [1] * len(shape_target)

    else:
        shape_out = []
        in_dim = 0
        for out_dim in range(len(shape_target)):
            dim_size_out = 1
            if in_dim < len(indices_in) and out_dim == indices_in[in_dim]:
                dim_size_out = shape_target[out_dim]
                assert tensor_in.size(in_dim) == dim_size_out
                in_dim += 1

            shape_out.append(dim_size_out)

        assert in_dim == len(indices_in), \
               f"indices_in={indices_in} might not be ordered"

    return tensor_in.reshape(shape_out)

def interleave(tensor1: torch.Tensor, tensor2: torch.Tensor, *, dim: int):
    """
    Interleave the two tensors along `dim`.
    """

    shape1 = list(tensor1.size())
    shape2 = list(tensor2.size())
    n_dim = len(shape1)

    different_lengths = shape2[dim] == shape1[dim] - 1
    if different_lengths:
        # Treat the last column of tensor1 separately
        slices = [slice(None)] * n_dim
        slices[dim] = slice(0,-1)
        full_tensor1 = tensor1
        tensor1 = full_tensor1[slices]
        shape1 = list(tensor1.size())
        slices[dim] = slice(-1,None)
        last_column = full_tensor1[slices]

    assert shape1 == shape2

    shape_out = copy.copy(shape1)
    shape_out[dim] += shape2[dim]
    out = torch.stack((tensor1, tensor2), dim=dim+1).view(*shape_out)

    if different_lengths:
        out = torch.cat((out, last_column), dim)

    return out
