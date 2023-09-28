import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_lineplot(env_id, metric, y_label):
    # Read the data for each individual algorithm.
    ppo_data = pd.read_csv(f'results/ppo_{env_id}_{metric}.csv')
    dqn_data = pd.read_csv(f'results/dqn_{env_id}_{metric}.csv')
    a2c_data = pd.read_csv(f'results/a2c_{env_id}_{metric}.csv')

    # Plot a lineplot for each algorithm.
    fig = plt.figure(figsize=(10, 5))
    ax = sns.lineplot(x='Iteration', y='dqn', linewidth=2.0, data=dqn_data, errorbar='pi', label='DQN')
    ax = sns.lineplot(x='Iteration', y='a2c', linewidth=2.0, data=a2c_data, errorbar='pi', label='A2C')
    ax = sns.lineplot(x='Iteration', y='ppo', linewidth=2.0, data=ppo_data, errorbar='pi', label='PPO')
    max_iter = ppo_data['Iteration'].max() + 1

    # Plot the baselines.
    if metric == 'hv':
        if env_id == 'deep-sea-treasure-concave-v0':
            baselines = {'True PF': np.full(max_iter, 4255)}
        elif env_id == 'minecart-v0':
            baselines = {
                'GPI-LS': [722.835429391406, 722.0529849919758, 720.4498781838754, 716.5630230831209, 715.155832680495],
                'PCN': [562.3794170629624, 562.2471920250745, 560.7465164701142, 499.2242079401762, 454.27223076180957]}
        else:
            baselines = {'GPI-LS': [36408302.37870406, 36273972.49959476, 36257568.87122492, 36255358.6213686,
                                    36241249.26302673],
                         'PCN': [24157119.610899586, 23846669.989973992, 23268621.59945868, 23095473.71940881,
                                 22832642.24314286]}

        baseline_colors = {'True PF': '#d62728', 'GPI-LS': '#9467bd', 'PCN': '#8c564b'}

        for alg, hvs in baselines.items():
            color = baseline_colors[alg]
            mean_hv = np.mean(hvs)
            ax = sns.lineplot(x=range(max_iter), y=np.full(max_iter, mean_hv), linewidth=2.0, label=alg, linestyle='--',
                              color=color)

    sns.move_legend(ax, "lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.savefig(f"results/{env_id}_{metric}.pdf", dpi=fig.dpi)
    plt.clf()


def plot_hv_cov(env_id):
    """Plot the hypervolume and coverage for each algorithm."""
    plot_lineplot(env_id, 'cov', "Coverage")
    plot_lineplot(env_id, 'hv', "Hypervolume")


if __name__ == '__main__':
    env_id = "mo-reacher-v4"
    plot_hv_cov(env_id)
