# kolpinn

`kolpinn` is a framework for implementing physics-informed neural networks (PINNs) based on [PyTorch](https://pytorch.org/).

## Implementation

All quantities are represented as PyTorch `Tensor`s on `Grid`s, see `grid_quantities.py`.

`Model`s can generate new quantities, tracking the gradients if necessary (`model.py`).

Batching (`batching.py`) and plotting (`visualization.py`) of the quantities are supported.
