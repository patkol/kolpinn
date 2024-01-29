from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap


def visualize(trainer):
    path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    visualization.save_training_history_plot(trainer, path_prefix)

    qs = trainer.get_extended_qs()
    qs['bulk']['x'].get_fd_derivative('x')
    save_lineplot(qs['bulk']['y'], 'y', 'x', path_prefix = path_prefix)
    print(f"c={qs['bulk']['c']}")
    save_lineplot(qs['bulk']['y'].get_fd_derivative('x'), 'dydx', 'x', path_prefix = path_prefix)


    # Plot inputs / outputs
    save_lineplot(qs['bulk']['x_transformed'], 'x_transformed', 'x', path_prefix = path_prefix)
    save_lineplot(qs['bulk']['nn_output0'], 'nn_output0', 'x', path_prefix = path_prefix)
    save_lineplot(qs['bulk']['untransformed_output'], 'untransformed_output', 'x', path_prefix = path_prefix)
