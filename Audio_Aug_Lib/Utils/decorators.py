def check_x_dim(func):
    """
    Check if the input audio signal has the correct shape (num_channels, num_samples), where num_channels is 1 or 2.
    """
    def wrapper(*args, **kwargs):
        arg_names = func.__code__.co_varnames[:len(args)]
        if 'x' not in arg_names:
            raise ValueError('There is no argument x in the function')
        arg_idx = arg_names.index('x')
        x_val = args[arg_idx]
        if len(x_val.shape) != 2 or x_val.shape[0] not in [1, 2]:
            raise ValueError('The input audio signal must have shape (num_channels, num_samples). Num_channels must be 1 or 2')
        return func(*args, **kwargs)
    return wrapper


def check_argument_pos(arg_names: list):
    """
    Check if the arguments specified are > 0.
    :param arg_names: list of strings, the names of the arguments to check.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg_name in arg_names:
                if arg_name in kwargs:
                    if kwargs[arg_name] <= 0:
                        raise ValueError(f'The argument {arg_name} must be positive')
            return func(*args, **kwargs)
        return wrapper
    return decorator

