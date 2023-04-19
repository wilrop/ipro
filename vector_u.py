import numpy as np
import torch


def constrained_sum(vector, referent, nadir, ideal):
    """Compute the constrained utility from a vector.

    Args:
        vector (ndarray): The obtained vector.
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.

    Returns:
        float: The constrained utility.
    """
    min_val = min((vector - referent) / (ideal - nadir))
    if min_val > 0:
        return sum(vector)
    else:
        return min_val


def create_cs(referent, nadir, ideal):
    """Create a constrained sum function.

    Args:
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.

    Returns:
        function: The constrained sum function.
    """
    return lambda vec: constrained_sum(vec, referent, nadir, ideal)


def create_batched_cs(referent, nadir, ideal, backend='numpy'):
    """Create a batched constrained sum function.

    Args:
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal: The ideal vector.
        backend (str, optional): The backend to use. Defaults to 'numpy'.

    Returns:
        function: The batched constrained sum function.
    """
    if backend == 'numpy':
        return lambda vecs: np.apply_along_axis(create_cs(referent, nadir, ideal), -1, vecs)
    elif backend == 'torch':
        def apply_along_axis(function, axis, x):
            return torch.stack([
                function(x_i) for x_i in torch.unbind(x, dim=axis)
            ], dim=axis)

        return lambda vecs: apply_along_axis(create_cs(referent, nadir, ideal), -1, vecs)
    else:
        raise NotImplementedError


def aasf(vector, referent, nadir, ideal, aug=0.):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is monotonically increasing in the positive orthant.

    Args:
        vector (ndarray): The obtained vector.
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.

    Returns:
        float: The obtained utility.
    """
    pos_vec = (ideal - nadir)
    return min((vector - referent) / pos_vec) + aug * sum(vector / pos_vec)


def create_aasf(referent, nadir, ideal, aug=0.):
    """Create a non-batched augmented achievement scalarizing function.

    Args:
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.

    Returns:
        function: The non-batched augmented achievement scalarizing function.
    """
    return lambda vec: aasf(vec, referent, nadir, ideal, aug=aug)


def create_batched_aasf(referent, nadir, ideal, aug=0., backend='numpy'):
    """Create a batched augmented achievement scalarizing function.

    Args:
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.
        backend (str): The backend to use for the computation.

    Returns:
        function: The batched augmented achievement scalarizing function.
    """
    pos_vec = (ideal - nadir)
    if backend == 'numpy':
        return lambda vec: np.min((vec - referent) / pos_vec, axis=-1) + aug * np.sum(vec / pos_vec, axis=-1)
    elif backend == 'torch':
        return lambda vec: torch.min((vec - referent) / pos_vec, dim=-1)[0] + aug * torch.sum(vec / pos_vec, dim=-1)
    else:
        raise NotImplementedError
