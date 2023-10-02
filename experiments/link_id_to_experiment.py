import os
import json


def link_experiments():
    """Creates a json file that links an experiment id to a hyperparameter file and seed"""
    all_files = os.listdir('hyperparams')
    experiments = {}
    idx = 1

    for file in all_files:
        if file[:3] in ['dqn', 'a2c', 'ppo']:
            for seed in range(5):
                experiments[idx] = (file, seed)
                idx += 1

    json.dump(experiments, open('hyperparams/experiments.json', 'w'))


if __name__ == '__main__':
    link_experiments()
