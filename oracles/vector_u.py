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
    frac_improvement = (vector - referent) / (ideal - nadir)
    min_val = min(frac_improvement)
    if min_val > 0:
        return sum(frac_improvement)
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


def aasf(vector, referent, nadir, ideal, aug=0., backend='numpy'):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is monotonically increasing in the positive orthant.

    Args:
        vector (ndarray): The obtained vector.
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.
        aug (float, optional): The augmentation factor. Defaults to 0.
        backend (str, optional): The backend to use. Defaults to 'numpy'.

    Returns:
        float: The obtained utility.
    """
    frac_improvement = (vector - referent) / (ideal - nadir)
    if backend == 'numpy':
        return np.min(frac_improvement) + aug * np.mean(frac_improvement)
    elif backend == 'torch':
        return torch.min(frac_improvement) + aug * torch.mean(frac_improvement)


def create_aasf(referent, nadir, ideal, aug=0., backend='numpy'):
    """Create a non-batched augmented achievement scalarizing function.

    Args:
        referent (ndarray): The referent vector.
        nadir (ndarray): The nadir vector.
        ideal (ndarray): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.
        backend (str, optional): The backend to use. Defaults to 'numpy'.

    Returns:
        function: The non-batched augmented achievement scalarizing function.
    """
    return lambda vec: aasf(vec, referent, nadir, ideal, aug=aug, backend=backend)


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
        def numpy_func(vec):
            frac_improvement = (vec - referent) / pos_vec
            return np.min(frac_improvement, axis=-1) + aug * np.mean(frac_improvement, axis=-1)

        return numpy_func
    elif backend == 'torch':
        def torch_func(vec):
            frac_improvement = (vec - referent) / pos_vec
            return torch.min(frac_improvement, dim=-1)[0] + aug * torch.mean(frac_improvement, dim=-1)

        return torch_func
    else:
        raise NotImplementedError
