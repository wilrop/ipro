import numpy as np
from ipro.oracles import aasf


def check_utility():
    ref = np.array([1, -19])
    ideal = np.array([124, -1])
    vec = np.array([1, -1])
    aug = 0.1
    print(aasf(vec, ref, ref, ideal, aug=aug, backend='numpy'))


def rank_points(refs, ideals, vecs, augs, scale):
    for aug in augs:
        for ref, ideal in zip(refs, ideals):
            res = [(vec, aasf(vec, ref, ref, ideal, aug=aug, scale=scale, backend='numpy')) for vec in vecs]
            order = sorted(res, key=lambda x: x[1], reverse=True)

            print(f'Augmentation: {aug}')
            for i, (vec, val) in enumerate(order):
                print(f'{i + 1}. {ref} - {ideal}: {vec} -> {val}')


def rank_dst():
    refs = np.array([[0, -19], [0., -14.]])
    ideals = np.array([[124, 0], [50., 0.]])
    vecs = [np.array([0, 0]), np.array([1, -1]), np.array([2, -3]), np.array([3, -5]), np.array([5, -7]),
            np.array([8, -8]), np.array([16, -9]), np.array([24, -13]), np.array([50, -14]), np.array([74, -17]),
            np.array([124, -19])]
    augs = [0, 0.005, 0.01, 0.1]
    scale = 1000

    rank_points(refs, ideals, vecs, augs, scale)


def rank_minecart_deterministic():
    vecs = np.array([[0., 0., -3.1199985], [0.0, 0.8, -0.87999976], [0.0, 1.5, -1.2199999], [0.28, 1.22, -2.359999],
                     [0.35, 1.15, -1.5799997], [0.4, 0.6, -0.8199998], [0.4, 1.1, -3.1199985], [0.42, 1.08, -1.6199995],
                     [0.6, 0.4, -0.79999983], [0.6, 0.9, -1.0599997], [0.65, 0.85, -1.2799997],
                     [0.75, 0.75, -0.91999984], [0.8, 0.7, -1.3599997], [0.85, 0.65, -1.3399997],
                     [0.9, 0.6, -1.0199997], [1.08, 0.42, -1.3999996], [1.1, 0.4, -1.1799998], [1.15, 0.35, -1.3399997],
                     [1.22, 0.28, -1.2799997], [1.5, 0.0, -0.95999986]])

    refs = np.array([[0., 0., -4.1199985]])
    ideals = np.array([[1.5, 1.5, -0.31999996]])
    augs = [0.005]
    scale = 1000

    rank_points(refs, ideals, vecs, augs, scale)


def rank_minecart():
    vecs = [np.array([0.55612687, 0.11092755, -0.65061296]), np.array([0.85192097, 0.16992796, -0.81650404]),
            np.array([0.60293373, 0.12026385, -0.77743027]), np.array([0.92362357, 0.18423008, -0.94917702]),
            np.array([0.40115733, 0.27441997, -0.67291286]), np.array([0.58274497, 0.39863875, -0.83521908]),
            np.array([0.63179215, 0.43219049, -0.96483263]), np.array([0.65784273, 0.45001093, -1.11527075]),
            np.array([0.47126046, 0.47126046, -0.70880496]), np.array([0.55392683, 0.55392683, -0.79616165]),
            np.array([0.57676679, 0.57676679, -1.11842114]), np.array([0.27441997, 0.40115733, -0.67291286]),
            np.array([0.39863875, 0.58274497, -0.83521908]), np.array([0.43219049, 0.63179215, -0.96483263]),
            np.array([0.45001093, 0.65784273, -1.11527075]), np.array([0.11092755, 0.55612687, -0.65061296]),
            np.array([0.16992796, 0.85192097, -0.81650404]), np.array([0.12026385, 0.60293373, -0.77743027]),
            np.array([0.18423008, 0.92362357, -0.94917702]), np.array([0., 0., -0.24923698])]
    refs = np.array([[0., 0., -4.1199985]])
    ideals = np.array([[1.5, 1.5, -0.31999996]])
    augs = [0.005]
    scale = 1000

    rank_points(refs, ideals, vecs, augs, scale)


if __name__ == '__main__':
    rank_minecart()
