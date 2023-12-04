import torch

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
    return abs2(b-a).mean()

def complex_grad(outputs: torch.Tensor, inputs: torch.Tensor, *args, **kwargs):
    """Derives the real and imaginary parts of 'outputs' seperately.

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
