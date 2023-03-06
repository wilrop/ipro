import numpy as np
import torch
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

    Note:
        For this utility function, you can guarantee that the utility is greater everywhere inside the rectangle than
        on the already discovered edges given an appropriate target vector. Specifically, the target vector needs to
        ensure that the line goes through the bottom left corner of the rectangle.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.

    Returns:
        float: The obtained utility.
    """
    problem = LpProblem('vectorUtility', LpMaximize)

    c = LpVariable('c', lowBound=0.)
    c_term = target / np.linalg.norm(target)
    for vi, c_termi in zip(vector, c_term):
        problem += vi - c * c_termi >= 0

    problem += c  # Maximise the utility.
    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    c = problem.objective.value()  # Get the objective value.
    return c


def rectangle_u(vector, target, nadir):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is monotonically increasing in the positive orthant.

    Note:
        For this utility function, any point outside the rectangle or on the already discovered edges will have a
        lower utility than inside the rectangle for any given target vector.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        float: The obtained utility.
    """
    problem = LpProblem('vectorUtility', LpMaximize)

    c = LpVariable('c', lowBound=0.)
    c_term = target / np.linalg.norm(target - nadir)
    for vi, c_termi in zip(vector, c_term):
        problem += vi - c * c_termi >= 0

    problem += c  # Maximise the utility.
    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    c = problem.objective.value()  # Get the objective value.
    return c


def fast_rectangle_u(vector, target, nadir):
    """Compute the utility as rectangle_u but in a smarter way.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        float: The obtained utility.
    """
    return np.min(vector * np.linalg.norm(target - nadir) / target)


def fast_translated_rectangle_u_(vector, target, nadir):
    """Compute the utility as rectangle_u but in a smarter way.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        float: The obtained utility.
    """
    return np.min((vector - nadir) * np.linalg.norm(target - nadir))


def create_fast_rectangle_u(target, nadir):
    """Create a fast utility function for a specific target and nadir vector.

    Args:
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        function: The fast utility function.
    """
    constant = np.linalg.norm(target - nadir) / target
    return lambda vec: np.min(vec * constant)


def create_fast_translated_rectangle_u(target, nadir):
    """Create a fast utility function for a specific target and nadir vector.

    Args:
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        function: The fast utility function.
    """
    constant = np.linalg.norm(target - nadir)
    return lambda vec: np.min((vec - nadir) * constant)


def create_batched_fast_rectangle_u(target, nadir, backend='numpy'):
    """Create a fast utility function for a specific target and nadir vector.

    Args:
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.
        backend (str): The backend to use for the computation.

    Returns:
        function: The fast utility function.
    """
    constant = np.linalg.norm(target - nadir) / target
    if backend == 'numpy':
        return lambda vec: np.min(vec * constant, axis=-1)
    elif backend == 'torch':
        constant = torch.Tensor(constant)
        return lambda vec: torch.min(vec * constant, dim=-1)[0]
    else:
        raise NotImplementedError


def create_batched_fast_translated_rectangle_u(target, nadir, backend='numpy'):
    """Create a fast utility function for a specific target and nadir vector.

    Args:
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.
        backend (str): The backend to use for the computation.

    Returns:
        function: The fast utility function.
    """
    constant = np.linalg.norm(target - nadir)
    if backend == 'numpy':
        return lambda vec: np.min((vec - nadir) * constant, axis=-1)
    elif backend == 'torch':
        constant = torch.Tensor(constant)  # TODO: Can be made float and probably used instead of numpy now in code.
        return lambda vec: torch.min((vec - nadir) * constant, dim=-1)[0]
    else:
        raise NotImplementedError


def fast_rectangle_u_minimisation(vector, target, nadir):
    """Compute the utility for a minimisation problem.

    Args:
        vector (ndarray): The obtained vector.
        target (ndarray): The target vector.
        nadir (ndarray): The nadir vector.

    Returns:
        float: The obtained utility.
    """
    return np.max(vector * np.linalg.norm(target - nadir) / nadir)
