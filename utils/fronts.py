import json
import numpy as np
import pygmo as pg


def minecart_fronts():
    ideals = np.array([[1.5, 0., -0.95999986], [0., 1.5, -0.95999986], [0., 0., -0.24923698]])
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

    computed_vecs = []

    nadir = np.array([0., 0., -3.1199985]) - 1

    with open("minecart.json", "r") as f:
        data = json.load(f)

    for key, value in data.items():
        if key.startswith("pareto_point"):
            computed_vecs.append(value)
    computed_vecs = np.array(computed_vecs)

    correct_vecs = np.vstack((correct_vecs, ideals))
    computed_vecs = np.vstack((computed_vecs, ideals))

    hv_true = pg.hypervolume(-correct_vecs).compute(-nadir)
    hv_computed = pg.hypervolume(-computed_vecs).compute(-nadir)

    print("Minecart")
    print(f"True HV: {hv_true}")
    print(f"Computed HV: {hv_computed}")
    print("----------")


def dst_fronts():
    vecs = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, -3.0], [3.0, -5.0], [5.0, -7.0], [8.0, -8.0], [16.0, -9.0],
                     [24.0, -13.0], [50.0, -14.0], [74.0, -17.0], [124.0, -19.0]])
    nadir = np.array([0, -19.0]) - 1
    hv = pg.hypervolume(-vecs).compute(-nadir)
    print("Deep Sea Treasure")
    print(f"HV: {hv}")
    print("----------")


if __name__ == '__main__':
    minecart_fronts()
    dst_fronts()
