"""This file contains utilities to assist with processing property values and types"""
import logging
import re

log = logging.getLogger()

from device_db_client import model


def numbered_py_name_to_type_and_num(py_name_num):
    """Converts a numbered pythonic name `py_name_num` into a py_name and number

    `py_name_num` is equivalent to a python variable name (from PEP8) with a
    number suffix. `py_name_num` is split such that the suffix is returned as
    `number` and the prefix is `py_name`. This means that `py_name` cannot end
    in numbers. Another limitation of this function is that it assumes the
    output `py_name` is at least two characters long.

    Args:
        py_name_num (str): the pythonic name to split

    Returns:
        py_name: the base pythonic name
        number: the number extract from the input py_name
    """
    PY_NAME_REGEX = r'^([A-Za-z_][A-Za-z_0-9]*[A-Za-z_])'
    NUMBER_REGEX = r'(\d+)$'
    py_name = re.search(PY_NAME_REGEX, py_name_num).groups()[0]
    number = int(re.search(NUMBER_REGEX, py_name_num).groups()[0])
    return py_name, number


def add_prefix_to_property_type_name(py_name, sweet_spot, transition):
    """Adds prefixes to a property type py_name

    The format of a py_name with a prefix is
    `<sweet_spot>_<transition>_<py_name>`. If either `sweet_spot` or
    `transition` is `None`, the connecting underscore/s will not be included.
    The values `sweet_spot` and `transition` are assumed to be strings (str) and
    are converted to lower-case before being included in the new property type
    'pythonic' name.

    Example:
        .. code-block:: python
            from itertools import product

            base_py_name = "pi_half_amp"
            transitions = ['ge', 'ef']
            sweet_spots = ['uss','lss',None]

            for t,ss in product(transitions,sweet_spots):
                print(add_prefix_to_property_type_name(base_py_name, t, ss))

    Args:
        py_name (str): the base property type 'pythonic' name
        sweet_spot (Optional[str]): the sweet-spot, or None if no sweet-spot is used
        transition (Optional[str]): the transition (ge, ef), or None if not used

    Returns:
        str: a new 'pythonic' name with prefixes for the `sweet_spot` and `transition`
    """
    new_py_name = py_name
    if transition is not None:
        new_py_name = f"{transition.lower()}_{new_py_name}"
    if sweet_spot is not None:
        new_py_name = f"{sweet_spot.lower()}_{new_py_name}"
    return new_py_name
