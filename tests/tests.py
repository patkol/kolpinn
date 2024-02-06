import torch

from kolpinn.mathematics import interleave


def test_interleave(dim):
    tensor1 = torch.arange(36).view(2,3,2,3)
    print(tensor1)
    tensor2 = torch.arange(100,136).view(2,3,2,3)
    print(tensor2)
    print()
    print(interleave(tensor1, tensor2, dim=dim))

def test_interleave2():
    tensor1 = torch.arange(36).view(2,3,2,3)
    print(tensor1)
    tensor2 = torch.arange(100,124).view(2,2,2,3)
    print(tensor2)
    print()
    print(interleave(tensor1, tensor2, dim=1))



if __name__ == "__main__":
    test_interleave(0)
    print()
    print()
    test_interleave(1)
    print()
    print()
    test_interleave(3)
    print()
    print()
    test_interleave2()

