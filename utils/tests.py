import numpy as np
from oracles.vector_u import aasf


def check_utility():
    ref = np.array([1, -19])
    ideal = np.array([124, -1])
    vec = np.array([1, -1])
    aug = 0.1
    print(aasf(vec, ref, ref, ideal, aug=aug, backend='numpy'))


def rank_points():
    ref = np.array([0, -19])
    ideal = np.array([124, 0])
    vecs = [np.array([0, 0]), np.array([1, -1]), np.array([2, -3]), np.array([3, -5]), np.array([5, -7]),
            np.array([8, -8]),
            np.array([16, -9]), np.array([24, -13]), np.array([50, -14]), np.array([74, -17]), np.array([124, -19])]
    augs = [0, 0.005, 0.01, 0.1]

    for aug in augs:
        res = [(vec, aasf(vec, ref, ref, ideal, aug=aug, backend='numpy')) for vec in vecs]
        order = sorted(res, key=lambda x: x[1], reverse=True)

        print(f'Augmentation: {aug}')
        for i, (vec, val) in enumerate(order):
            print(f'{i + 1}. {vec} -> {val}')


if __name__ == '__main__':
    rank_points()
    check_utility()
