import numpy as np

import maximum_empty_box as meb
import vector_u


def test_func_on_vectors(target, nadir, vectors, func):
    for vector in vectors:
        print(f'Vector: {vector} - Utility: {func(vector, target, nadir)}')


def test_utility_funcs():
    rectangle = (np.array([0, 1]), np.array([1, 1]), np.array([0, 2]), np.array([1, 2]))
    nadir = rectangle[0]
    ideal = rectangle[3]
    vector1 = rectangle[1]
    vector2 = rectangle[2]
    vector3 = np.array([0.5, 1.5])
    vector4 = np.array([0.5, 1])

    print(f'Nadir: {nadir}')
    print(f'Ideal: {ideal}')

    utility_functions = [vector_u.rectangle_u, vector_u.fast_rectangle_u, vector_u.fast_translated_rectangle_u]
    vectors = [vector1, vector2, vector3, vector4]

    for func in utility_functions:
        print(f'Function: {func}')
        test_func_on_vectors(ideal, nadir, vectors, func)
        print(f'---------------------')


def test_maximum_emtpy_box():
    points = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    nadir = np.array([0, 0, 0])
    ideal = np.array([1, 1, 1])
    print(f'Rectangle: {nadir} - {ideal}')
    print(f'Points: {points}')

    new_nadir, new_ideal = meb.dumitrescu_jiang(points, nadir, ideal)
    print(f'New Rectangle: {new_nadir} - {new_ideal}')


def test_concavity():
    tries = 1000000
    num_dims = 2
    low = -5
    high = 5
    func = vector_u.fast_translated_rectangle_u
    for i in range(tries):
        nadir = np.random.uniform(low=low, high=high, size=num_dims)
        ideal = np.random.uniform(low=low, high=high, size=num_dims)
        vec1 = np.random.uniform(low=low, high=high, size=num_dims)
        vec2 = np.random.uniform(low=low, high=high, size=num_dims)
        alpha = np.random.uniform()
        out1 = func((1 - alpha) * vec1 + alpha * vec2, ideal, nadir)
        out2 = (1 - alpha) * func(vec1, ideal, nadir) + alpha * func(vec2, ideal, nadir)
        if not out1 >= out2 and not np.isclose(out1, out2):
            print(out1, out2)
            raise ValueError('Not concave.')

    print('Empirically verified that the utility function is concave.')


if __name__ == '__main__':
    test_utility_funcs()
    test_maximum_emtpy_box()
    test_concavity()
