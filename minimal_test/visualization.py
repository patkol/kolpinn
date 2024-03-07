import matplotlib.pyplot as plt

from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap, add_lineplot


def visualize(trainer):
    path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    visualization.save_training_history_plot(trainer, path_prefix)

    qs = trainer.get_extended_qs(for_training = True) # for_training s.t. the gradient can be taken
    save_lineplot(qs['bulk']['y'], qs['bulk'].grid, 'y', 'x', path_prefix = path_prefix)
