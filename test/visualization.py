import matplotlib.pyplot as plt

from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap, add_lineplot


def visualize(trainer):
    path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    visualization.save_training_history_plot(trainer, path_prefix)

    qs = trainer.get_extended_qs(for_training = True) # for_training s.t. the gradient can be taken
    save_lineplot(qs['bulk']['y'], 'y', 'x', path_prefix = path_prefix)
    print(f"c={qs['bulk']['c']}")
    save_lineplot(qs['bulk']['y'].get_fd_derivative('x'), 'dydx', 'x', path_prefix = path_prefix)

    # Plot inputs / outputs
    save_lineplot(qs['bulk']['x_transformed'], 'x_transformed', 'x', path_prefix = path_prefix)
    save_lineplot(qs['bulk']['nn_output0'], 'nn_output0', 'x', path_prefix = path_prefix)
    save_lineplot(qs['bulk']['untransformed_output'], 'untransformed_output', 'x', path_prefix = path_prefix)

    # Second derivatives
    y_dx_exact = qs['bulk']['y'].get_grad(
        qs['bulk']['x'],
        retain_graph=True,
        create_graph=True,
    )
    y_dx_dx_exact = y_dx_exact.get_grad(
        qs['bulk']['x'],
        retain_graph=True,
        create_graph=True,
    )
    y_dx_dx_fd_direct = qs['bulk']['y'].get_fd_second_derivative('x')
    y_dx_dx_fd_indirect = qs['bulk']['y'].get_fd_derivative('x').get_fd_derivative('x')
    fig, ax = plt.subplots()
    add_lineplot(ax, y_dx_dx_exact, 'exact', 'x')
    add_lineplot(ax, y_dx_dx_fd_direct, 'fd_direct', 'x')
    add_lineplot(ax, y_dx_dx_fd_indirect, 'fd_indirect', 'x')
    ax.legend()
    fig.savefig(path_prefix + 'second_derivatives.pdf')
