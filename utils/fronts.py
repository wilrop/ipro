import json
import numpy as np
import pygmo as pg
import pandas as pd
from extract_box import extract_box_from_array


def extract_front_from_json(file_name):
    """Extract the Pareto front from a file name.

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
    """Extract the Pareto front from a file name.

    Args:
        file_name (str): The file to load the results from.

    Returns:
        Dict: A dictionary with the point found for each iteration.
    """
    df = pd.read_csv(file_name)
    df = df.to_dict(orient="index")
    df = {int(key): np.array(list(value.values())) for key, value in df.items()}
    return df


def extract_front_from_wandb_csv(file_name):
    df = pd.read_csv(file_name)
    return df


def reacher_fronts():
    """Compute the Pareto front for mo-reacher."""
    front = extract_front_from_csv("gdpi_reacher.csv")
    front_vecs = np.array(list(front.values()))
    nadir, corner_vecs, ideal = extract_box_from_array(front_vecs)
    box_corners = np.array([[40, -50, -50, -50],
                            [-50, 40, -50, -50],
                            [-50, -50, 40, -50],
                            [-50, -50, -50, 40]])
    ref_point = np.array([-50, -50, -50, -50])  # Used by morl-baselines.

    hv_true = pg.hypervolume(-front_vecs).compute(-ref_point)

    print("Reacher")
    print(f"True HV: {hv_true}")
    print("----------")


def minecart_fronts():
    """Compute the Pareto fronts for minecart."""
    ideals = np.array([[1.5, -1, -200], [-1, 1.5, -200], [-1, -1, 0]])
    correct_vecs = np.array(
        [np.array([0.55612687, 0.11092755, -0.65061296]), np.array([0.85192097, 0.16992796, -0.81650404]),
         np.array([0.60293373, 0.12026385, -0.77743027]), np.array([0.92362357, 0.18423008, -0.94917702]),
         np.array([0.40115733, 0.27441997, -0.67291286]), np.array([0.58274497, 0.39863875, -0.83521908]),
         np.array([0.63179215, 0.43219049, -0.96483263]), np.array([0.65784273, 0.45001093, -1.11527075]),
         np.array([0.47126046, 0.47126046, -0.70880496]), np.array([0.55392683, 0.55392683, -0.79616165]),
         np.array([0.57676679, 0.57676679, -1.11842114]), np.array([0.27441997, 0.40115733, -0.67291286]),
         np.array([0.39863875, 0.58274497, -0.83521908]), np.array([0.43219049, 0.63179215, -0.96483263]),
         np.array([0.45001093, 0.65784273, -1.11527075]), np.array([0.11092755, 0.55612687, -0.65061296]),
         np.array([0.16992796, 0.85192097, -0.81650404]), np.array([0.12026385, 0.60293373, -0.77743027]),
         np.array([0.18423008, 0.92362357, -0.94917702]), np.array([0., 0., -0.24923698])])

    ref_point = np.array([-1, -1, -200])
    pf = np.vstack((correct_vecs, ideals))
    hv_true = pg.hypervolume(-pf).compute(-ref_point)

    print("Minecart")
    print(f"True HV: {hv_true}")
    print("----------")


def minecart_fronts_baseline(alg):
    """Compute the Pareto front in minecart for a baseline algorithm."""
    ref_point = np.array([-1, -1, -200])
    ideals = np.array([[1.5, 0., -0.95999986], [0., 1.5, -0.95999986], [0., 0., -0.24923698]])

    hypervolumes = []
    for i in range(5):
        correct_vecs = extract_front_from_wandb_csv(f"{alg}/minecart{i + 1}.csv")
        pf = np.vstack((correct_vecs, ideals))
        hv = pg.hypervolume(-pf).compute(-ref_point)
        hypervolumes.append(hv)
    hv_mean = np.mean(hypervolumes)
    print(hypervolumes)
    hv_std = np.std(hypervolumes)

    print("Minecart")
    print(f"{alg} HV: {hv_mean} +/- {hv_std}")
    print("----------")


def dst_fronts():
    """Compute the Pareto fronts for DST."""
    vecs = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, -3.0], [3.0, -5.0], [5.0, -7.0], [8.0, -8.0], [16.0, -9.0],
                     [24.0, -13.0], [50.0, -14.0], [74.0, -17.0], [124.0, -19.0]])
    nadir = np.array([0, -19.0]) - 1
    hv = pg.hypervolume(-vecs).compute(-nadir)
    print("Deep Sea Treasure")
    print(f"HV: {hv}")
    print("----------")


if __name__ == '__main__':
    # minecart_fronts_baseline('pcn')
    minecart_fronts()
