# FIXME: Please remove the @deprecated implementation from general.py
#.       once we are 100% on py3.13
from functools import wraps
import sys
import warnings

# FIXME: Compensate feature-lag due to older Python version usage
# Only define some decorators while we are
# lagging behind the current Python stable.
#
# See: https://peps.python.org/pep-0702/
#
if sys.version_info < (3, 13):

    def deprecated(reason: str):
        """Marks a deprecated function."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"Call to deprecated function '{func.__name__}': {reason}",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator

else: # Python 3.13+, use a "pass-through" decorator and shout
    from warnings import deprecated # noqa: F401
