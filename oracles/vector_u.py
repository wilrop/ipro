import torch
import numpy as np


def aasf(vector, referent, nadir, ideal, aug=0., scale=100, backend='numpy'):
    """Compute the utility from a vector given a specific target vector.

    Note:
        This utility function is monotonically increasing in the positive orthant.

    Args:
        vector (array_like): The obtained vector.
        referent (array_like): The referent vector.
        nadir (array_like): The nadir vector.
        ideal (array_like): The ideal vector.
        aug (float, optional): The augmentation factor. Defaults to 0.
        scale (float, optional): The scaling factor. Defaults to 100.
        backend (str, optional): The backend to use. Defaults to 'numpy'.

    Returns:
        array_like: The obtained utility.
    """
    frac_improvement = scale * (vector - referent) / (ideal - nadir)
    if backend == 'numpy':
        return np.min(frac_improvement, axis=-1) + aug * np.mean(frac_improvement, axis=-1)
    elif backend == 'torch':
        return torch.min(frac_improvement, dim=-1)[0] + aug * torch.mean(frac_improvement, dim=-1)


def create_aasf(referent, nadir, ideal, aug=0., scale=1000, backend='numpy'):
    """Create a non-batched augmented achievement scalarizing function.

    Args:
        referent (array_like): The referent vector.
        nadir (array_like): The nadir vector.
        ideal (array_like): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.
        scale (float): The scaling factor. Defaults to 100.
        backend (str, optional): The backend to use. Defaults to 'numpy'.

    Returns:
        callable: The non-batched augmented achievement scalarizing function.
    """
    return lambda vec: aasf(vec, referent, nadir, ideal, aug=aug, scale=scale, backend=backend)


def create_batched_aasf(referent, nadir, ideal, aug=0., scale=1000, backend='numpy'):
    """Create a batched augmented achievement scalarizing function.

    Args:
        referent (array_like): The referent vector.
        nadir (array_like): The nadir vector.
        ideal (array_like): The ideal vector.
        aug (float): The augmentation factor. Defaults to 0.
        scale (float): The scaling factor. Defaults to 100.
        backend (str): The backend to use for the computation.

    Returns:
        callable: The batched augmented achievement scalarizing function.
    """
    pos_vec = (ideal - nadir)
    if backend == 'numpy':
        def numpy_func(vec):
            frac_improvement = scale * (vec - referent) / pos_vec
            return np.min(frac_improvement, axis=-1) + aug * np.mean(frac_improvement, axis=-1)

        return numpy_func
    elif backend == 'torch':
        def torch_func(vec):
            frac_improvement = scale * (vec - referent) / pos_vec
            return torch.min(frac_improvement, dim=-1)[0] + aug * torch.mean(frac_improvement, dim=-1)

        return torch_func
    else:
        raise NotImplementedError
