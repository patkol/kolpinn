from kolpinn import visualization
from kolpinn.visualization import save_lineplot, save_heatmap

def visualize(batchers: dict, models: dict, trainer):
    visualization.save_training_history_plot(trainer)
    visualization.save_loss_plots(trainer, 'x')
    visualize_bulk(batchers['bulk'], models)

def visualize_bulk(batcher, models: dict):
    q = batcher.get_extended_q(models)
    save_lineplot(q['y'], 'y', 'x')



