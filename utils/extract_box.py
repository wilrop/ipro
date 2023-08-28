import pandas as pd
import numpy as np


def extract_box_from_df(df):
    corner_idcs = df.idxmax()
    corner_vecs = df.iloc[corner_idcs].to_numpy()
    nadir = df.min().to_numpy()
    ideal = df.max().to_numpy()
    return nadir, corner_vecs, ideal


def extract_box_from_array(arr):
    corner_idcs = np.argmax(arr, axis=0)
    corner_vecs = arr[corner_idcs]
    nadir = arr.min(axis=0)
    ideal = arr.max(axis=0)
    return nadir, corner_vecs, ideal


if __name__ == '__main__':
    df = pd.read_csv('udrl.csv')
    extract_box_from_df(df)
    arr = np.array([np.array([0.55612687, 0.11092755, -0.65061296]), np.array([0.85192097, 0.16992796, -0.81650404]),
                    np.array([0.60293373, 0.12026385, -0.77743027]), np.array([0.92362357, 0.18423008, -0.94917702]),
                    np.array([0.40115733, 0.27441997, -0.67291286]), np.array([0.58274497, 0.39863875, -0.83521908]),
                    np.array([0.63179215, 0.43219049, -0.96483263]), np.array([0.65784273, 0.45001093, -1.11527075]),
                    np.array([0.47126046, 0.47126046, -0.70880496]), np.array([0.55392683, 0.55392683, -0.79616165]),
                    np.array([0.57676679, 0.57676679, -1.11842114]), np.array([0.27441997, 0.40115733, -0.67291286]),
                    np.array([0.39863875, 0.58274497, -0.83521908]), np.array([0.43219049, 0.63179215, -0.96483263]),
                    np.array([0.45001093, 0.65784273, -1.11527075]), np.array([0.11092755, 0.55612687, -0.65061296]),
                    np.array([0.16992796, 0.85192097, -0.81650404]), np.array([0.12026385, 0.60293373, -0.77743027]),
                    np.array([0.18423008, 0.92362357, -0.94917702]), np.array([0., 0., -0.24923698])])
    extract_box_from_array(arr)
