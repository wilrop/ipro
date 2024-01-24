import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import imageio.v3 as iio


def plot_ipro_2d(box_queues, log_dir, name):
    """Plot the stages of IPRO-2D one by one."""
    pass


def create_gif(log_dir, name):
    """Create a gif from standalone images."""
    image_iter = []
    for file_name in sorted(os.listdir(log_dir)):
        if file_name.startswith('iter'):
            iteration = int(file_name.split('_')[1].split('.')[0])
            file_path = os.path.join(log_dir, file_name)
            image_iter.append((iteration, iio.imread(file_path)))
    sorted_images = sorted(image_iter, key=lambda x: x[0])
    images = [image for _, image in sorted_images]
    iio.imwrite(os.path.join(log_dir, f"{name}.gif"), images, format='GIF', duration=1000)


def read_data(env_id, metric, algs):
    """Read the data for each individual algorithm."""
    datasets = []
    for alg in algs:
        data = pd.read_csv(f'data/{alg}_{env_id}_{metric}.csv')
        datasets.append(data)
    return datasets


def plot_hv(env_id, algs, alg_colors, baselines, baseline_colors, log_scale=True, include_baselines=True):
    """Plot the hypervolume for each algorithm."""
    method = 'IPRO-2D' if env_id == 'deep-sea-treasure-concave-v0' else 'IPRO'
    alg_ids, alg_labels = zip(*algs)
    datasets = read_data(env_id, 'hv', alg_ids)

    fig = plt.figure(figsize=(10, 5))
    last_step_vals = []
    for alg_id, alg_label, data in zip(alg_ids, alg_labels, datasets):
        ax = sns.lineplot(x='Step', y=alg_id, linewidth=2.0, data=data, errorbar='pi', label=f'{method} ({alg_label})',
                          color=alg_colors[alg_label])
        last_step = int(data['Step'].iloc[-1])
        last_val = ax.lines[-1].get_ydata()[-1]
        last_step_vals.append((last_step, last_val))
        ax.scatter(last_step, last_val, marker='*', s=200, color=alg_colors[alg_label])

    if include_baselines:
        baseline_datasets = read_data(env_id, 'hv', baselines)
        for baseline, data in zip(baselines, baseline_datasets):
            ax = sns.lineplot(x='Step', y=baseline, linewidth=2.0, data=data, errorbar='pi', label=baseline,
                              color=baseline_colors[baseline])
            last_step = data['Step'].iloc[-1]
            last_val = ax.lines[-1].get_ydata()[-1]
            last_step_vals.append((last_step, last_val))
            ax.scatter(last_step, last_val, marker='*', s=200, color=baseline_colors[baseline])

    max_step = max([step for step, _ in last_step_vals])
    if env_id == 'deep-sea-treasure-concave-v0':
        max_step += 100000

    if env_id == 'deep-sea-treasure-concave-v0':
        true_pf = np.full(max_step, 4255)
        ax = sns.lineplot(x=range(max_step), y=true_pf, linewidth=2.0, label='True PF', linestyle='--',
                          color=baseline_colors['True PF'])

    for (step, val), label in zip(last_step_vals, list(alg_labels) + baselines):
        if label in alg_colors:
            color = alg_colors[label]
        else:
            color = baseline_colors[label]
        x_data = np.linspace(step, max_step)
        y_data = np.full(len(x_data), val)
        ax = sns.lineplot(x=x_data, y=y_data, linewidth=2.0, linestyle='--', color=color)

    if log_scale:  # Set the y-axis in log scale
        ax.set_xscale('log')

    sns.move_legend(ax, "lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel("Step")
    plt.ylabel('Hypervolume')
    plt.savefig(f"plots/{env_id}_hv.pdf", dpi=fig.dpi)
    plt.clf()


def plot_cov(env_id, algs, alg_colors, log_scale=True):
    """Plot the coverage for each algorithm."""
    method = 'IPRO-2D' if env_id == 'deep-sea-treasure-concave-v0' else 'IPRO'
    alg_ids, alg_labels = zip(*algs)
    datasets = read_data(env_id, 'cov', alg_ids)

    # Plot a lineplot for each algorithm.
    fig = plt.figure(figsize=(10, 5))
    for alg_id, alg_label, data in zip(alg_ids, alg_labels, datasets):
        ax = sns.lineplot(x='Step', y=alg_id, linewidth=2.0, data=data, errorbar='pi', label=f'{method} ({alg_label})',
                          color=alg_colors[alg_label])

    if log_scale:  # Set the y-axis in log scale
        ax.set_xscale('log')

    sns.move_legend(ax, "lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel("Step")
    plt.ylabel('Coverage')
    plt.savefig(f"plots/{env_id}_cov.pdf", dpi=fig.dpi)
    plt.clf()


def plot_hv_cov(env_id,
                algs,
                alg_colors,
                baselines,
                baseline_colors,
                hv_log_scale=True,
                cov_log_scale=True,
                include_baselines=True):
    """Plot the hypervolume and coverage for each algorithm."""
    plot_cov(env_id,
             algs,
             alg_colors,
             log_scale=cov_log_scale)
    plot_hv(env_id,
            algs,
            alg_colors,
            baselines,
            baseline_colors,
            log_scale=hv_log_scale,
            include_baselines=include_baselines)


if __name__ == '__main__':
    algs = [('SN-MO-PPO', 'PPO'), ('SN-MO-DQN', 'DQN'), ('SN-MO-A2C', 'A2C')]
    alg_colors = {
        'PPO': '#1f77b4',
        'DQN': '#ff7f0e',
        'A2C': '#2ca02c',
    }
    baselines = ['GPI-LS', 'PCN']
    baseline_colors = {
        'GPI-LS': '#d62728',
        'PCN': '#9467bd',
        'Envelope': '#8c564b',
        'True PF': '#e377c2'
    }
    env_ids = ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']

    for env_id in env_ids:
        print(f'Plotting {env_id}')
        include_baselines = True
        cov_log_scale = True
        hv_log_scale = True
        plot_hv_cov(env_id,
                    algs,
                    alg_colors,
                    baselines,
                    baseline_colors,
                    include_baselines=include_baselines,
                    cov_log_scale=cov_log_scale,
                    hv_log_scale=hv_log_scale)
