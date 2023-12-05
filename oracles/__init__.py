from oracles.debug_oracle import DebugOracle
from oracles.finite_oracle import FiniteOracle
from oracles.mo_dqn import MODQN
from oracles.mo_a2c import MOA2C
from oracles.mo_ppo import MOPPO
from oracles.sn_mo_dqn import SNMODQN
from oracles.sn_mo_a2c import SNMOA2C
from oracles.sn_mo_ppo import SNMOPPO


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
    elif alg == "MO-PPO":
        return MOPPO(*args, **kwargs)
    elif alg == 'SN-MO-DQN':
        return SNMODQN(*args, **kwargs)
    elif alg == 'SN-MO-A2C':
        return SNMOA2C(*args, **kwargs)
    elif alg == 'SN-MO-PPO':
        return SNMOPPO(*args, **kwargs)
    elif alg == 'debug':
        return DebugOracle(*args, **kwargs)
    else:
        raise ValueError(f'Unknown oracle: {alg}')
