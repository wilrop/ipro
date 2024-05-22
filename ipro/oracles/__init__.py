from ipro.oracles.debug_oracle import DebugOracle
from ipro.oracles.finite_oracle import FiniteOracle
from ipro.oracles.mo_dqn import MODQN
from ipro.oracles.mo_a2c import MOA2C
from ipro.oracles.mo_ppo import MOPPO
from ipro.oracles.sn_mo_dqn import SNMODQN
from ipro.oracles.sn_mo_a2c import SNMOA2C
from ipro.oracles.sn_mo_ppo import SNMOPPO
from ipro.oracles.sn_mo_ddpg import SNMODDPG


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
    elif alg == 'SN-MO-DDPG':
        return SNMODDPG(*args, **kwargs)
    elif alg == 'debug':
        return DebugOracle(*args, **kwargs)
    else:
        raise ValueError(f'Unknown oracle: {alg}')
