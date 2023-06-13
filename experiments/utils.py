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
