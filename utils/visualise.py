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


def plot_lineplot(env_id, metric, y_label):
    # Read the data for each individual algorithm.
    ppo_data = pd.read_csv(f'results/ppo_{env_id}_{metric}.csv')
    dqn_data = pd.read_csv(f'results/dqn_{env_id}_{metric}.csv')
    a2c_data = pd.read_csv(f'results/a2c_{env_id}_{metric}.csv')

    # Plot a lineplot for each algorithm.
    fig = plt.figure(figsize=(10, 5))
    ax = sns.lineplot(x='Step', y='dqn', linewidth=2.0, data=dqn_data, errorbar='pi', label='DQN')
    ax = sns.lineplot(x='Step', y='a2c', linewidth=2.0, data=a2c_data, errorbar='pi', label='A2C')
    ax = sns.lineplot(x='Step', y='ppo', linewidth=2.0, data=ppo_data, errorbar='pi', label='PPO')

    # Read and plot baseline data.
    if metric == 'hv':
        gpils_data = pd.read_csv(f'results/GPI-LS_{env_id}_{metric}.csv')
        pcn_data = pd.read_csv(f'results/PCN_{env_id}_{metric}.csv')
        envelope_data = pd.read_csv(f'results/Envelope_{env_id}_{metric}.csv')
        ax = sns.lineplot(x='Step', y='GPI-LS', linewidth=2.0, data=gpils_data, errorbar='pi', label='GPI-LS')
        ax = sns.lineplot(x='Step', y='PCN', linewidth=2.0, data=pcn_data, errorbar='pi', label='PCN')
        ax = sns.lineplot(x='Step', y='Envelope', linewidth=2.0, data=envelope_data, errorbar='pi', label='Envelope')

    # Plot the baselines.
    if metric == 'hv' and env_id == 'deep-sea-treasure-concave-v0':
        max_step = int(ax.get_xlim()[1])
        true_pf = np.full(max_step, 4255)
        ax = sns.lineplot(x=range(max_step), y=true_pf, linewidth=2.0, label='True PF', linestyle='--')

    # Set the y-axis in log scale
    ax.set_xscale('log')
    sns.move_legend(ax, "lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.savefig(f"results/{env_id}_{metric}.pdf", dpi=fig.dpi)
    plt.clf()


def plot_hv_cov(env_id):
    """Plot the hypervolume and coverage for each algorithm."""
    plot_lineplot(env_id, 'cov', "Coverage")
    plot_lineplot(env_id, 'hv', "Hypervolume")


if __name__ == '__main__':
    for env_id in ['deep-sea-treasure-concave-v0', 'minecart-v0', 'mo-reacher-v4']:
        print(f'Plotting {env_id}')
        plot_hv_cov(env_id)
