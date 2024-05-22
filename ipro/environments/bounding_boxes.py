import numpy as np


def get_bounding_box(env_id):
    """Get the bounding box for the given environment."""
    if env_id == 'deep-sea-treasure-concave-v0':
        minimals = np.array([[0, -50], [0, -50]])
        maximals = np.array([[124, -50], [1, -1]])
        ref_point = np.array([0, -50])
    elif env_id == 'mo-reacher-v4':
        minimals = np.array([[-50, -50, -50, -50], [-50, -50, -50, -50], [-50, -50, -50, -50], [-50, -50, -50, -50]])
        maximals = np.array([[40, -50, -50, -50], [-50, 40, -50, -50], [-50, -50, 40, -50], [-50, -50, -50, 40]])
        ref_point = np.array([-50, -50, -50, -50])
    elif env_id == 'minecart-v0':
        minimals = np.array([[-1, -1, -200], [-1, -1, -200], [-1, -1, -200]])
        maximals = np.array([[1.5, -1, -200], [-1, 1.5, -200], [-1, -1, 0]])
        ref_point = np.array([-1, -1, -200])
    elif env_id == 'mo-walker2d-v4':
        minimals = np.array([[-100, -100], [-100, -100]])
        maximals = np.array([[100, -100], [-100, 100]])
        ref_point = np.array([-100, -100])
    else:
        raise NotImplementedError
    return minimals, maximals, ref_point
