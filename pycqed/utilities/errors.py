import functools
import logging
import traceback

log = logging.getLogger(__name__)

class NoProgressError(Exception):
    """Exception raised when a measurement does not make progress."""
    pass


def handle_exception(func=None, store_exception=True):
    """
    Decorates a function to handle exceptions, and possibly store them.
    Args:
        func (None, callable): function to be decorated. Note that this should
            not be provided by the user! It is intrinsic to whether or not
            the decorator is called with/without arguments. See
            https://stackoverflow.com/questions/3888158/making-decorators-with-optional-arguments
            for more details.
        store_exception (bool): whether or not the exception should be stored.
            If True, will try to assign the exception under an "exception"
            attribute of the first argument, i.e. the object instance (`self`)
            when the decorated function is a method of a custom.
            This is intended to allow the user to access the exception after
            it occurred. Note that it should never be set to True when
            decorating a static method or a function that is not a method,
            as it would then try to assign the exception to the first argument
            of that function.


    Returns:
        Decorated function / function that decorates depending on whether
        the decorator is called without / with arguments.

    Examples:
        >>> # Handle exceptions for the method "dangerous" and store exception
        >>> class MyExperiment:
        ...     @handle_exception
        ...     def dangerous(self):
        ...         return 1/0
        >>> me = MyExperiment()
        >>> me.dangerous()
        >>> me.exception # exception was saved by decorator

    """
    def _handle_exception(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                output = f(*args, **kwargs)
            except Exception as ex:
                log.error(f'An error occured in {f.__qualname__}')
                output = None
                # store exception in object if no name collision with "exception"
                if store_exception:
                    try:
                        args[0].exception = ex
                    except:
                        log.warning(f'Could not store exception as an attribute'
                                    f' of {args[0]}')
                        # attribute cannot be set
                        pass
                traceback.print_exc()
            return output
        return wrapper
    if func is None:
        # handle_exception was called with arguments; return a function that
        # decorates
        return _handle_exception
    else:
        # handle_exception was called without arguments;
        # return a decorated function
        return _handle_exception(func)
