# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from typing import Optional
from collections.abc import Sequence
import os
import matplotlib.pyplot as plt  # type: ignore
import torch

from . import mathematics
from .grids import Grid
from . import training


def get_avg_tensor(
    tensor: torch.Tensor,
    grid: Grid,
    kept_dimensions: Sequence[str],
    avgd_dimensions: Optional[Sequence[str]],
) -> torch.Tensor:
    """
    Return a tensor that only has the `kept_dimensions` in the given order.
    The `avgd_dimensions` are averaged over, and the others must be singleton
    dimensions.
    Extra `avgd_dimensions` that do not appear in the grid are allowed.
    """

    if avgd_dimensions is None:
        avgd_dimensions = ()

    assert set(kept_dimensions).issubset(set(grid.dimensions_labels)), kept_dimensions
    assert set(avgd_dimensions).isdisjoint(set(kept_dimensions)), avgd_dimensions

    tensor = tensor.detach().cpu()
    perm = []
    dims_to_squeeze = []
    for label in grid.dimensions_labels:
        dim = grid.index[label]
        if label in kept_dimensions:
            perm.append(kept_dimensions.index(label))
            continue
        elif label in avgd_dimensions:
            tensor = tensor.mean(dim=dim, keepdim=True)
        else:
            assert (
                tensor.shape[dim] == 1
            ), f"Dimension {label} is not averaged over and not singleton"
        dims_to_squeeze.append(dim)

    tensor = tensor.squeeze(dims_to_squeeze)
    tensor = tensor.permute(perm)
    assert len(tensor.size()) == len(
        kept_dimensions
    ), f"{tensor.size()}, {kept_dimensions}"

    return tensor


def format_unit_name(s):
    return "" if s is None else f" [{s}]"


def add_lineplot(
    ax,
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    lines_dimension=None,
    *,
    avgd_dimensions: Optional[Sequence[str]] = None,
    x_quantity: Optional[torch.Tensor] = None,
    print_raw_data=False,
    quantity_unit=1,
    quantity_unit_name=None,
    x_unit=1,
    x_unit_name=None,  # not used
    lines_unit=1,
    lines_unit_name=None,
    **kwargs,
):
    """
    Plot the average values of `quantity` vs. the x_dimension or vs. the
    averaged x_quantity.
    """

    quantity_unit_name = format_unit_name(quantity_unit_name)
    lines_unit_name = "" if lines_unit_name is None else " " + lines_unit_name

    plot_dimensions = [x_dimension]
    label = quantity_label + quantity_unit_name
    if lines_dimension is not None:
        plot_dimensions.append(lines_dimension)
        label = list(
            f"{lines_dimension} = {v/lines_unit:.2f}" + lines_unit_name
            for v in grid[lines_dimension]
        )

    y_values = get_avg_tensor(quantity, grid, plot_dimensions, avgd_dimensions)
    if x_quantity is None:
        x_values = grid[x_dimension].detach().cpu()
    else:
        x_values = get_avg_tensor(x_quantity, grid, plot_dimensions, avgd_dimensions)

    if print_raw_data:
        print("x:")
        print(x_values)
        print("y:")
        print(y_values)

    ax.plot(
        x_values / x_unit,
        y_values / quantity_unit,
        label=label,
        **kwargs,
    )


def save_lineplot(
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    lines_dimension=None,
    *,
    avgd_dimensions: Optional[Sequence[str]] = None,
    x_quantity: Optional[torch.Tensor] = None,
    x_label: Optional[str] = None,
    path_prefix=None,
    xscale="linear",
    yscale="linear",
    xlim_left=None,
    xlim_right=None,
    ylim_bottom=None,
    ylim_top=None,
    print_raw_data=False,
    quantity_unit=1,
    quantity_unit_name=None,
    x_unit=1,
    x_unit_name=None,  # not used
    lines_unit=1,
    lines_unit_name=None,
    plot_kwargs=None,
    legend=False,
):
    """
    Plot the average values of `quantity`
    """

    if x_label is None:
        assert x_quantity is None
        x_label = x_dimension
    if path_prefix is None:
        path_prefix = "plots/"
    if plot_kwargs is None:
        plot_kwargs = {}

    os.makedirs(path_prefix, exist_ok=True)
    path = (
        path_prefix + f"{quantity_label}_{x_dimension}_{lines_dimension}_lineplot.pdf"
    )

    fig, ax = plt.subplots()
    add_lineplot(
        ax,
        quantity,
        grid,
        quantity_label,
        x_dimension,
        lines_dimension,
        avgd_dimensions=avgd_dimensions,
        x_quantity=x_quantity,
        print_raw_data=print_raw_data,
        quantity_unit=quantity_unit,
        quantity_unit_name=quantity_unit_name,
        x_unit=x_unit,
        x_unit_name=x_unit_name,  # not used
        lines_unit=lines_unit,
        lines_unit_name=lines_unit_name,
        **plot_kwargs,
    )
    ax.set_xlabel(x_label + format_unit_name(x_unit_name))
    ax.set_ylabel(quantity_label + format_unit_name(quantity_unit_name))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.grid(visible=True)
    if legend:
        ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_heatmap(
    fig,
    ax,
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    y_dimension,
    avgd_dimensions: Optional[Sequence[str]] = None,
    print_raw_data=False,  # not used
    quantity_unit=1,
    quantity_unit_name=None,
    x_unit=1,
    x_unit_name=None,
    y_unit=1,
    y_unit_name=None,
    **kwargs,
):
    """
    Plot the average values of `quantity`.
    """

    quantity_unit_name = format_unit_name(quantity_unit_name)
    x_unit_name = format_unit_name(x_unit_name)
    y_unit_name = format_unit_name(y_unit_name)

    plot_dimensions = (y_dimension, x_dimension)

    tensor = get_avg_tensor(quantity, grid, plot_dimensions, avgd_dimensions)
    x_values = grid[x_dimension].detach().cpu()
    y_values = grid[y_dimension].detach().cpu()

    heatmap = ax.pcolormesh(
        x_values / x_unit,
        y_values / y_unit,
        tensor / quantity_unit,
        label=quantity_label + quantity_unit_name,
        linewidth=0,
        rasterized=True,
        **kwargs,
    )
    ax.set_xlabel(x_dimension + x_unit_name)
    ax.set_ylabel(y_dimension + y_unit_name)
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label(quantity_label + quantity_unit_name)


def save_heatmap(
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    y_dimension,
    avgd_dimensions: Optional[Sequence[str]] = None,
    print_raw_data=False,  # not used
    quantity_unit=1,
    quantity_unit_name=None,
    x_unit=1,
    x_unit_name=None,
    y_unit=1,
    y_unit_name=None,
    path_prefix=None,
    **kwargs,
):
    """
    Plot the average values of `quantity`
    """

    if path_prefix is None:
        path_prefix = "plots/"
    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + f"{quantity_label}_{x_dimension}_vs_{y_dimension}_heatmap.pdf"

    fig, ax = plt.subplots()
    add_heatmap(
        fig,
        ax,
        quantity,
        grid,
        quantity_label,
        x_dimension,
        y_dimension,
        avgd_dimensions,
        print_raw_data,
        quantity_unit,
        quantity_unit_name,
        x_unit,
        x_unit_name,
        y_unit,
        y_unit_name,
        **kwargs,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_complex_polar_plot(
    ax,
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    lines_dimension=None,
    avgd_dimensions: Optional[Sequence[str]] = None,
    quantity_unit=1,
    quantity_unit_name=None,
    lines_unit=1,
    lines_unit_name=None,
    **kwargs,
):
    """
    Plot the average values of `quantity`.
    """

    quantity_unit_name = format_unit_name(quantity_unit_name)
    lines_unit_name = "" if lines_unit_name is None else " " + lines_unit_name

    plot_dimensions = [x_dimension]
    label = quantity_label + quantity_unit_name
    if lines_dimension is not None:
        plot_dimensions.append(lines_dimension)
        label = list(
            f"{lines_dimension} = {v/lines_unit:.2f}" + lines_unit_name
            for v in grid[lines_dimension]
        )

    tensor = get_avg_tensor(quantity, grid, plot_dimensions, avgd_dimensions)
    ax.plot(
        torch.angle(tensor),
        torch.abs(tensor) / quantity_unit,
        label=label,
        **kwargs,
    )


def save_complex_polar_plot(
    quantity: torch.Tensor,
    grid: Grid,
    quantity_label,
    x_dimension,
    lines_dimension=None,
    avgd_dimensions: Optional[Sequence[str]] = None,
    path_prefix=None,
    quantity_unit=1,
    quantity_unit_name=None,
    lines_unit=1,
    lines_unit_name=None,
    legend: bool = False,
):
    """
    Plot the average values of `quantity`
    """

    if path_prefix is None:
        path_prefix = "plots/"
    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + f"{quantity_label}_{lines_dimension}_polar_plot.pdf"

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    add_complex_polar_plot(
        ax,
        quantity,
        grid,
        quantity_label,
        x_dimension,
        lines_dimension,
        avgd_dimensions,
        quantity_unit=quantity_unit,
        quantity_unit_name=quantity_unit_name,
        lines_unit=lines_unit,
        lines_unit_name=lines_unit_name,
    )
    ax.grid(visible=True)
    if legend:
        ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_training_history_plot(trainer: training.Trainer, path_prefix=None):
    if len(trainer.training_history.times) == 0:
        return

    if path_prefix is None:
        path_prefix = f"plots/{trainer.config.saved_parameters_index:04d}/"

    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + "training.pdf"

    fig, ax = plt.subplots()
    ax.set_prop_cycle(None)
    ax.plot(
        trainer.training_history.times,
        trainer.training_history.losses,
        linestyle="dashed",
        label=None,
        alpha=0.3,
    )
    ax.set_prop_cycle(None)
    loss_quantities_chain = mathematics.get_chained_values(
        trainer.config.loss_quantities
    )
    loss_names = list(loss_quantities_chain)
    loss_names += ["Total"]
    ax.plot(
        trainer.validation_history.times,
        trainer.validation_history.losses,
        label=loss_names,
    )
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
