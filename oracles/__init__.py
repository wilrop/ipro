from oracles.finite_oracle import FiniteOracle
from oracles.mo_dqn import MODQN
from oracles.mo_a2c import MOA2C


def init_oracle(alg, *args, **kwargs):
    """Initialize an oracle.

    Args:
        alg (str): The algorithm to use.
        *args: The arguments for the algorithm.
        **kwargs: The keyword arguments for the algorithm.

    Returns:
        Oracle: The oracle.
    """
    if alg == 'finite':
        return FiniteOracle(*args, **kwargs)
    elif alg == 'MO-DQN':
        return MODQN(*args, **kwargs)
    elif alg == 'MO-A2C':
        return MOA2C(*args, **kwargs)
    else:
        raise ValueError(f'Unknown oracle: {alg}')
