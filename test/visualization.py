from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap


def visualize(trainer):
    path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    visualization.save_training_history_plot(trainer, path_prefix)

    qs = trainer.get_extended_qs()
    save_lineplot(qs['bulk']['y'], 'y', 'x', path_prefix = path_prefix)
    print(f"c={qs['bulk']['c']}")
