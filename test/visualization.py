from kolpinn.model import get_extended_q_batchwise
from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap


def visualize(batchers: dict, models: dict, trainer):
    path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
    visualization.save_training_history_plot(trainer)
    visualization.save_loss_plots(trainer, 'x')
    visualize_bulk(batchers['bulk'], models, path_prefix)

def visualize_bulk(batcher, models: dict, path_prefix):
    q = get_extended_q_batchwise(batcher, models)
    save_lineplot(q['y'], 'y', 'x', path_prefix = path_prefix)
    print(f"c={q['c']}")



