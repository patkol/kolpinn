import os
import matplotlib.pyplot as plt
import torch

from .grid_quantities import Grid, Quantity
from . import training



def get_avg_tensor(quantity: Quantity, dimensions):
    grid = quantity.grid
    tensor = quantity.values.detach().cpu()
    perm = []
    dims_to_squeeze = []
    for label in grid.dimensions_labels:
        if label in dimensions:
            perm.append(dimensions.index(label))
            continue

        dim = grid.index[label]
        tensor = tensor.mean(dim = dim, keepdim=True)
        dims_to_squeeze.append(dim)

    tensor = tensor.squeeze(dims_to_squeeze)
    tensor = tensor.permute(perm)
    assert len(tensor.size()) == len(dimensions), \
           f'{tensor.size()}, {dimensions}'

    return tensor

format_unit_name = lambda s: '' if s is None else f' [{s}]'

def add_lineplot(
        ax,
        quantity: Quantity,
        quantity_label,
        x_dimension,
        lines_dimension = None,
        print_raw_data = False,
        quantity_unit = 1,
        quantity_unit_name = None,
        x_unit = 1,
        x_unit_name = None, #  not used
        lines_unit = 1,
        lines_unit_name = None,
        **kwargs,
    ):
    """
    Plot the average values of `quantity`.
    """

    quantity_unit_name = format_unit_name(quantity_unit_name)
    x_unit_name = format_unit_name(x_unit_name)
    lines_unit_name = '' if lines_unit_name is None else ' ' + lines_unit_name

    plot_dimensions = [x_dimension]
    label = quantity_label + quantity_unit_name
    if not lines_dimension is None:
        plot_dimensions.append(lines_dimension)
        label = list(f'{lines_dimension} = {v/lines_unit:.2f}' + lines_unit_name
                     for v in quantity.grid[lines_dimension])

    tensor = get_avg_tensor(quantity, plot_dimensions)
    x_values = quantity.grid[x_dimension].detach().cpu()
    if print_raw_data:
        print("x:")
        print(x_values)
        print("y:")
        print(tensor)

    ax.plot(
        x_values / x_unit,
        tensor / quantity_unit,
        label = label,
        **kwargs,
    )


def save_lineplot(
        quantity: Quantity,
        quantity_label,
        x_dimension,
        lines_dimension = None,
        path_prefix = None,
        xscale = 'linear',
        yscale = 'linear',
        xlim_left = None,
        xlim_right = None,
        ylim_bottom = None,
        ylim_top = None,
        print_raw_data = False,
        quantity_unit = 1,
        quantity_unit_name = None,
        x_unit = 1,
        x_unit_name = None, #  not used
        lines_unit = 1,
        lines_unit_name = None,
    ):
    """
    Plot the average values of `quantity`
    """

    if path_prefix is None:
        path_prefix = 'plots/'
    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + f'{quantity_label}_{x_dimension}_{lines_dimension}_lineplot.pdf'

    fig, ax = plt.subplots()
    add_lineplot(
        ax,
        quantity,
        quantity_label,
        x_dimension,
        lines_dimension,
        print_raw_data = print_raw_data,
        quantity_unit = quantity_unit,
        quantity_unit_name = quantity_unit_name,
        x_unit = x_unit,
        x_unit_name = x_unit_name, #  not used
        lines_unit = lines_unit,
        lines_unit_name = lines_unit_name,
    )
    ax.set_xlabel(x_dimension + format_unit_name(x_unit_name))
    ax.set_ylabel(quantity_label + format_unit_name(quantity_unit_name))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xlim_left, right=xlim_right)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.grid(visible=True)
    if not lines_dimension is None:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)


def add_heatmap(
        fig,
        ax,
        quantity: Quantity,
        quantity_label,
        x_dimension,
        y_dimension,
        print_raw_data = False, # not used
        quantity_unit = 1,
        quantity_unit_name = None,
        x_unit = 1,
        x_unit_name = None,
        y_unit = 1,
        y_unit_name = None,
        **kwargs,
    ):
    """
    Plot the average values of `quantity`.
    """

    quantity_unit_name = format_unit_name(quantity_unit_name)
    x_unit_name = format_unit_name(x_unit_name)
    y_unit_name = format_unit_name(y_unit_name)

    plot_dimensions = (y_dimension, x_dimension)

    tensor = get_avg_tensor(quantity, plot_dimensions)
    x_values = quantity.grid[x_dimension].detach().cpu()
    y_values = quantity.grid[y_dimension].detach().cpu()

    heatmap = ax.pcolormesh(
        x_values / x_unit,
        y_values / y_unit,
        tensor / quantity_unit,
        label = quantity_label + quantity_unit_name,
        linewidth=0,
        rasterized=True,
        **kwargs,
    )
    #ax.invert_yaxis()
    ax.set_xlabel(x_dimension + x_unit_name)
    ax.set_ylabel(y_dimension + y_unit_name)
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label(quantity_label + quantity_unit_name)


def save_heatmap(
        quantity: Quantity,
        quantity_label,
        x_dimension,
        y_dimension,
        print_raw_data = False, # not used
        quantity_unit = 1,
        quantity_unit_name = None,
        x_unit = 1,
        x_unit_name = None,
        y_unit = 1,
        y_unit_name = None,
        path_prefix = None,
        **kwargs,
    ):
    """
    Plot the average values of `quantity`
    """

    if path_prefix is None:
        path_prefix = 'plots/'
    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + f'{quantity_label}_{x_dimension}_vs_{y_dimension}_heatmap.pdf'

    fig, ax = plt.subplots()
    add_heatmap(
        fig,
        ax,
        quantity,
        quantity_label,
        x_dimension,
        y_dimension,
        print_raw_data,
        quantity_unit,
        quantity_unit_name,
        x_unit,
        x_unit_name,
        y_unit,
        y_unit_name,
        **kwargs,
    )
    fig.savefig(path)
    plt.close(fig)


def save_training_history_plot(trainer: training.Trainer, path_prefix = None):
    if path_prefix is None:
        path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    os.makedirs(path_prefix, exist_ok=True)
    path = path_prefix + 'training.pdf'

    fig, ax = plt.subplots()
    ax.set_prop_cycle(None)
    ax.plot(
        trainer.training_loss_times,
        trainer.training_loss_history,
        linestyle = 'dashed',
        label = None,
        alpha = 0.3,
    )
    ax.set_prop_cycle(None)
    ax.plot(
        trainer.validation_loss_times,
        trainer.validation_loss_history,
        label = trainer.loss_names,
    )
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def save_loss_plots(
        trainer: training.Trainer,
        x_dimension,
        lines_dimension = None,
        path_prefix = None,
    ):
    """
    Plot the average loss per batch.
    """

    if path_prefix is None:
        path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    os.makedirs(path_prefix, exist_ok=True)

    losses = trainer.get_validation_losses(save_if_best = False)
    for loss_label, loss in losses.items():
        save_lineplot(
            loss,
            loss_label,
            x_dimension,
            lines_dimension,
            path_prefix,
        )
