import numpy as np

from utility_function.utility_eval import generalised_expected_utility, generalised_maximum_utility_loss
from utility_function.generate_utility_fns import generate_utility_fns

PF_DST = np.array([
    [1.0, -1.0],
    [2.0, -3.0],
    [3.0, -5.0],
    [5.0, -7.0],
    [8.0, -8.0],
    [16.0, -9.0],
    [24.0, -13.0],
    [50.0, -14.0],
    [74.0, -17.0],
    [124.0, -19.0]]
)


def test_geu(fn_type, seed=0):
    min_vec = np.array([1, -19])
    max_vec = np.array([124, -1])
    num_utility_fns = int(1e4)
    utility_fns = generate_utility_fns(
        min_vec,
        max_vec,
        num_utility_fns,
        num_points=6,
        max_grad=5,
        fn_type=fn_type,
        seed=seed
    )
    ch_geu = generalised_expected_utility(PF_DST[[0, -1]], utility_fns)
    pf_geu = generalised_expected_utility(PF_DST, utility_fns)

    print(f'GEU of CH: {ch_geu}')
    print(f'GEU of PF: {pf_geu}')


def test_mul(fn_type, seed=0):
    min_vec = np.array([1, -19])
    max_vec = np.array([124, -1])
    num_utility_fns = int(1e4)
    utility_fns = generate_utility_fns(
        min_vec,
        max_vec,
        num_utility_fns,
        num_points=6,
        max_grad=5,
        fn_type=fn_type,
        seed=seed
    )
    ch_geu = generalised_maximum_utility_loss(PF_DST[[0, -1]], PF_DST, utility_fns)
    pf_geu = generalised_maximum_utility_loss(PF_DST, PF_DST, utility_fns)

    print(f'GEU of CH: {ch_geu}')
    print(f'GEU of PF: {pf_geu}')


def test_spread(fn_type, seed=0):
    min_vec = np.array([1, -19])
    max_vec = np.array([124, -1])
    num_utility_fns = int(1e4)
    utility_fns = generate_utility_fns(
        min_vec,
        max_vec,
        num_utility_fns,
        num_points=6,
        max_grad=5,
        fn_type=fn_type,
        seed=seed
    )
    counts = np.zeros(len(PF_DST))
    for i, utility_fn in enumerate(utility_fns):
        out = utility_fn(PF_DST)
        counts[np.argmax(out)] += 1

    print(f'Distribution: {counts / num_utility_fns}')


def test_monotonicity(fn_type, seed=0):
    min_vec = np.array([0, 0])
    max_vec = np.array([10, 10])
    num_utility_fns = 20
    utility_fns = generate_utility_fns(
        min_vec,
        max_vec,
        num_utility_fns,
        fn_type=fn_type,
        seed=seed
    )
    test_set = np.array([[3, 3], [2, 3], [3, 2]])
    for u_fn in utility_fns:
        out = u_fn(test_set)
        assert np.argmax(out) == 0, f'out: {out}'
    print(f'{fn_type} appears monotonically increasing.')


if __name__ == '__main__':
    fn_type = 'increasing_cumsum'
    seed = None
    # test_monotonicity(fn_type, seed)
    test_spread(fn_type, seed)
    # test_geu(fn_type, seed)
    # test_mul(fn_type, seed)
