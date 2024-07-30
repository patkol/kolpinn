# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict
import copy
import itertools
import torch


def transform(x, input_range, output_range):
    """Transform ''x'' in ''input_range'' into the ''output_range''"""

    assert len(input_range) == 2, input_range
    assert len(output_range) == 2, output_range
    assert input_range[0] < input_range[1]
    assert output_range[0] < output_range[1]

    scale_factor = (output_range[1] - output_range[0]) / (
        input_range[1] - input_range[0]
    )
    return (x - input_range[0]) * scale_factor + output_range[0]


def complex_abs2(a: torch.Tensor) -> torch.Tensor:
    if a.dtype in (torch.float16, torch.float32, torch.float64):
        return a**2

    return torch.real(a) ** 2 + torch.imag(a) ** 2


def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """nn.MSELoss that works for complex numbers"""
    return complex_abs2(b - a).mean()


def _complex_grad(outputs: torch.Tensor, inputs: torch.Tensor, *args, **kwargs):
    """
    Derives the complex output by the real input.

    ''torch.autograd.grad'' calculates the Wirtinger derivative instead.
    There is no support for sequences of tensors right now.
    """

    assert torch.is_complex(outputs)
    assert not torch.is_complex(inputs)

    # We need to `retain_graph` for the real gradient such that the graph
    # will be avaliable for the complex one.
    retain_graph = None
    if "retain_graph" in kwargs:
        retain_graph = kwargs["retain_graph"]

    kwargs["retain_graph"] = True
    (grad_real,) = torch.autograd.grad(
        torch.real(outputs),
        inputs,
        *args,
        **kwargs,
    )

    kwargs["retain_graph"] = retain_graph
    (grad_imag,) = torch.autograd.grad(
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

    assert not torch.is_complex(input_)

    grad_outputs = torch.ones_like(output, dtype=torch.int8)

    if torch.is_complex(output):
        grad_tensor = _complex_grad(
            outputs=output,
            inputs=input_,
            grad_outputs=grad_outputs,
            **kwargs,
        )
        return grad_tensor

    grad_tensors = torch.autograd.grad(
        outputs=output,
        inputs=input_,
        grad_outputs=grad_outputs,
        **kwargs,
    )
    return grad_tensors[0]


def generalized_cartesian_prod(*tensors: torch.Tensor):
    """Generalized to the cases where only one or no tensor is provided."""

    if len(tensors) == 0:
        return torch.zeros((0, 0))

    if len(tensors) == 1:
        tensor = tensors[0]
        assert len(tensor.size()) == 1
        return tensor.reshape((-1, 1))

    return torch.cartesian_prod(*tensors)


def exchange_dims(tensor: torch.Tensor, dim_1, dim_2):
    permutation = list(range(len(tensor.size())))
    permutation[dim_1], permutation[dim_2] = permutation[dim_2], permutation[dim_1]
    permuted_tensor = tensor.permute(permutation)

    return permuted_tensor


def remove_duplicates(list_: list):
    """
    Return a list excluding exact duplicates in the sense of 'is'.
    """

    out = []
    for i in range(len(list_)):
        duplicate = False
        for j in range(i):
            if list_[j] is list_[i]:
                duplicate = True
                break
        if not duplicate:
            out.append(list_[i])

    return out


def expand(tensor_in: torch.Tensor, shape_target, indices):
    """
    Expand tensor to the `shape_target`.
    The dimension `in_dim` in the input tensor will
    be dimension `indices[in_dim]` in the output tensor.
    The remaining dimensions of the output tensor will be singletons,
    independent of the value in `shape_target`.
    `indices` must be in ascending order.
    """

    if len(tensor_in.size()) == 0:
        assert indices == []
        shape_out = [1] * len(shape_target)

    else:
        shape_out = []
        in_dim = 0
        for out_dim in range(len(shape_target)):
            dim_size_out = 1
            if in_dim < len(indices) and out_dim == indices[in_dim]:
                dim_size_out = shape_target[out_dim]
                assert tensor_in.size(in_dim) == dim_size_out
                in_dim += 1

            shape_out.append(dim_size_out)

        assert in_dim == len(indices), f"indices={indices} might not be ordered"

    return tensor_in.reshape(shape_out)


def _interleave_equal_lengths(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    *,
    dim: int,
):
    size = list(tensor1.size())
    assert list(tensor2.size()) == size

    size[dim] *= 2

    return torch.stack((tensor1, tensor2), dim=dim + 1).view(*size)


def _interleave_different_lengths(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    *,
    dim: int,
):
    size1 = list(tensor1.size())
    n_dim = len(size1)

    size2 = copy.copy(size1)
    size2[dim] -= 1
    assert tensor2.size() == tuple(size2)

    # Treat the last column of tensor1 separately
    slices = [slice(None)] * n_dim
    slices[dim] = slice(0, -1)
    tensor1_sliced = tensor1[slices]
    slices[dim] = slice(-1, None)
    last_column = tensor1[slices]

    out_sliced = _interleave_equal_lengths(tensor1_sliced, tensor2, dim=dim)

    return torch.cat((out_sliced, last_column), dim)


def interleave(tensor1: torch.Tensor, tensor2: torch.Tensor, *, dim: int):
    """
    Interleave the two tensors along `dim`.
    `tensor1` can be one longer than `tensor2`, the other dimension must match.
    """

    if tensor1.size() == tensor2.size():
        return _interleave_equal_lengths(tensor1, tensor2, dim=dim)

    return _interleave_different_lengths(tensor1, tensor2, dim=dim)


def get_chained_values(d: Dict):
    return itertools.chain.from_iterable(d.values())
