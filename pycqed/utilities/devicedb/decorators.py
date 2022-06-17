import functools


def only_one_not_none(fields):
    """A decorator that validates that only one field in `fields` is not None

    Args:
        fields (list): list of kwarg parameter names, as str
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            __found_not_none__ = False
            for key, value in kwargs.items():
                if key in fields and value is not None:
                    if __found_not_none__:
                        raise ValueError(
                            f"Only one of the following parameters may be not None: {fields}"
                        )
                    else:
                        __found_not_none__ = True
            return func(*args, **kwargs)

        return wrapper

    return decorate


def all_or_none(fields):
    """A decorator that validates all fields are not None or all are None

    Args:
        fields (list): list of kwarg parameter names, as str
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # __not_none__ tells us if the list should be not None (True) or None (False)
            __not_none__ = None
            for i, (key, value) in enumerate(kwargs.items()):
                if key in fields:
                    if __not_none__ is None:
                        __not_none__ = value is not None
                    elif __not_none__ != (value is not None):
                        raise ValueError(
                            f"These parameters must either be all None or not None: {fields}"
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorate


def at_least_one_not_none(fields):
    """A decorator that validates that at least one field in `fields` is not None

    Args:
        fields (list): list of kwarg parameter names, as str
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            __found_not_none__ = False
            for key, value in kwargs.items():
                if key in fields and value is not None:
                    __found_not_none__ = True
                    break
            if not __found_not_none__:
                raise ValueError(
                    f"At least one of the following parameters must be not None: {fields}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorate
