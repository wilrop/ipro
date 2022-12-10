import numpy as np
from pulp import *


def vector_u_strict(vector, target, w=0.5):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is strictly monotonically increasing in the positive orthant when the return vector never
        is undominated by the target vector.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        w (float): A tradeoff hyperparameter that determines how to weight the utility from the original utility
            function with the added term.

    Returns:
        float: The obtained utility.
    """
    c = vector_u(vector, target)
    utility = w * c + (1 - w) * np.linalg.norm(vector)
    return utility


def c_eval(vector, target, c):
    """Evaluate the conditions of the linear program for a given c.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        c (float): A utility.

    Returns:
        ndarray: The evaluation of a c variable into the utility function.
    """
    return vector - c * target / np.linalg.norm(target)


def vector_u(vector, target):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is monotonically increasing in the positive orthant.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.

    Returns:
        float: The obtained utility.
    """
    problem = LpProblem('vectorUtility', LpMaximize)

    c = LpVariable('c', lowBound=0.)
    c_term = target / np.linalg.norm(target)
    for ri, c_termi in zip(vector, c_term):
        problem += ri - c * c_termi >= 0

    problem += c  # Maximise the utility.
    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    c = problem.objective.value()  # Get the objective value.
    return c


vector = np.array([1., 2.])
target = np.array([2., 4.])
utility = vector_u(vector, target)
utilit2 = vector_u_strict(vector, target)
print(utility)
print(utilit2)
print('----------------')

vector = np.array([1., 3.])
target = np.array([2., 4.])
utility = vector_u(vector, target)
utilit2 = vector_u_strict(vector, target)
print(utility)
print(utilit2)
print('----------------')

vector = np.array([2, 2.])
target = np.array([2., 4.])
utility = vector_u(vector, target)
utilit2 = vector_u_strict(vector, target)
print(utility)
print(utilit2)
print('----------------')

vector = np.array([2, 4.])
target = np.array([2., 4.])
utility = vector_u(vector, target)
utilit2 = vector_u_strict(vector, target)
print(utility)
print(utilit2)
print('----------------')
