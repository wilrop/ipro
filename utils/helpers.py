import torch


def load_activation_fn(activation):
    """Load the activation function.

    Args:
        activation (str): The name of the activation function.

    Returns:
        torch.nn.Module: The activation function.
    """
    if activation == 'relu':
        return torch.nn.ReLU
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU
    elif activation == 'tanh':
        return torch.nn.Tanh
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid
    else:
        raise ValueError(f'Invalid activation function: {activation}')


def strtobool(val):
    """Convert a string representation of truth to True or False.

    Args:
        val (str): The string to convert.

    Returns:
        bool: The boolean value.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
