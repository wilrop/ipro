import os
import json


def link_experiments():
    all_files = os.listdir('../experiments/hyperparams')
    experiments = {}
    idx = 1

    for file in all_files:
        if 'minecart' in file:
            for seed in range(5):
                experiments[idx] = (file, seed)
                idx += 1

    json.dump(experiments, open('experiments.json', 'w'))


if __name__ == '__main__':
    link_experiments()
