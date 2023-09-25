import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_lineplot(env_id, metric, y_label):
    # Plot a figure of the dataframe where each column is a line.
    ppo_data = pd.read_csv(f'results/ppo_{env_id}_{metric}.csv')
    dqn_data = pd.read_csv(f'results/dqn_{env_id}_{metric}.csv')
    a2c_data = pd.read_csv(f'results/a2c_{env_id}_{metric}.csv')

    fig = plt.figure(figsize=(10, 5))
    ax = sns.lineplot(x='Iteration', y='ppo', linewidth=2.0, data=ppo_data, errorbar='pi', label='PPO')
    ax = sns.lineplot(x='Iteration', y='dqn', linewidth=2.0, data=dqn_data, errorbar='pi', label='DQN')
    ax = sns.lineplot(x='Iteration', y='a2c', linewidth=2.0, data=a2c_data, errorbar='pi', label='A2C')
    max_iter = ppo_data['Iteration'].max() + 1

    if metric == 'hv':
        if env_id in ['deep-sea-treasure-concave-v0', 'minecart-v0']:
            if env_id == 'deep-sea-treasure-concave-v0':
                baselines = {}
                true_hv = 4255
            else:
                baselines = {'GPI-LS': [891.3790112323194, 890.5252512450548, 889.3775362878422, 885.3060338097989,
                                        884.8193796798131],
                             'PCN': [853.5964514352254, 857.1756180847733, 873.5644291955045, 836.8533498690243,
                                     834.3390908885495]}
                true_hv = 897.627214807107
            ax = sns.lineplot(x=range(max_iter), y=np.full(max_iter, true_hv), linewidth=2.0, label='True PF',
                              linestyle='--')
            # Get the color used for this line.
        else:
            baselines = {'GPI-LS': [36408304, 36273972, 36257568, 36255360, 36241248],
                         'PCN': [24157119.611, 23846669.99, 23268621.599, 23095473.719, 22832642.243]}

        baseline_colors = ('#9467bd', '#8c564b')
        for (alg, hvs), color in zip(baselines.items(), baseline_colors):
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
    plot_lineplot(env_id, 'cov', "Coverage")
    plot_lineplot(env_id, 'hv', "Hypervolume")


if __name__ == '__main__':
    env_id = "deep-sea-treasure-concave-v0"
    plot_hv_cov(env_id)
