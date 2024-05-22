import json
import numpy as np
import pygmo as pg
import pandas as pd
import mo_gymnasium as mo_gym
from scipy.spatial import ConvexHull


def extract_front_from_json(file_name):
    """Extract the Pareto front from a JSON file.

    Args:
        file_name (str): The file to load the results from.

    Returns:
        Dict: A dictionary with the point found for each iteration.
    """
    with open(file_name, "r") as f:
        data = json.load(f)

    front = {}
    for key, value in data.items():
        if key.startswith("pareto_point"):
            iter_point = int(key.split('_')[-1])
            front[iter_point] = value
    return front


def extract_front_from_csv(file_name):
    """Extract the Pareto front from a CSV file."""
    front = pd.read_csv(file_name).to_numpy()
    return front


def convex_coverage_set(vecs):
    """Compute a convex coverage set for a set of vectors."""
    origin = np.min(vecs, axis=0)
    extended_policies = np.vstack((vecs, origin))
    return np.array([vecs[idx - 1] for idx in ConvexHull(extended_policies).vertices if idx != 0])


def minecart_front(gamma=0.98):
    """Compute the Pareto front for minecart."""
    env = mo_gym.make("minecart-v0")
    pf = np.array(env.unwrapped.pareto_front(gamma=gamma, symmetric=True))
    ccs = np.array(env.unwrapped.convex_coverage_set(gamma=gamma, symmetric=True))

    ref_point = np.array([-1, -1, -200])
    hv_pf = pg.hypervolume(-pf).compute(-ref_point)
    hv_ccs = pg.hypervolume(-ccs).compute(-ref_point)

    print("Minecart")
    print(f"True PF HV: {hv_pf}")
    print(f"True CCS HV: {hv_ccs}")
    print("----------")


def baseline_fronts(env='minecart', algs=None):
    """Compute the Pareto front in an environment for a baseline algorithm."""
    print(f"Baselines {env}")

    if algs is None:
        algs = ['pcn', 'gpi_ls']

    ref_points = {'reacher': np.array([-50, -50, -50, -50]),
                  'minecart': np.array([-1, -1, -200])}
    ref_point = ref_points[env]

    for alg in algs:
        hypervolumes = []
        for i in range(5):
            found_vecs = extract_front_from_csv(f"{alg}/{env}{i + 1}.csv")
            hv_pf = pg.hypervolume(-found_vecs).compute(-ref_point)
            hypervolumes.append(hv_pf)
        hv_mean = np.mean(hypervolumes)
        hv_std = np.std(hypervolumes)
        print(f'{alg} PF hypervolumes: {hypervolumes}')
        print(f"{alg} PF HV: {hv_mean} +/- {hv_std}")
    print("----------")


def dst_fronts():
    """Compute the Pareto fronts for DST."""
    vecs = np.array([[0.0, 0.0],
                     [1.0, -1.0],
                     [2.0, -3.0],
                     [3.0, -5.0],
                     [5.0, -7.0],
                     [8.0, -8.0],
                     [16.0, -9.0],
                     [24.0, -13.0],
                     [50.0, -14.0],
                     [74.0, -17.0],
                     [124.0, -19.0]])
    ref_point = np.array([0, -50])
    hv = pg.hypervolume(-vecs).compute(-ref_point)
    print("Deep Sea Treasure")
    print(f"HV: {hv}")
    print("----------")


if __name__ == '__main__':
    baseline_fronts(env='minecart', algs=['pcn', 'gpi_ls'])
    minecart_front()
    dst_fronts()
