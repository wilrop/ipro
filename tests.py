import numpy as np

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

    utility_functions = [vector_u.rectangle_u, vector_u.fast_rectangle_u ,vector_u.fast_translated_rectangle_u]
    vectors = [vector1, vector2, vector3, vector4]

    for func in utility_functions:
        print(f'Function: {func}')
        test_func_on_vectors(ideal, nadir, vectors, func)
        print(f'---------------------')


if __name__ == '__main__':
    test_utility_funcs()
