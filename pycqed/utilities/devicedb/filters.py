import logging
from enum import Enum

from .validator import PropertyValuesDictValidator

log = logging.getLogger(__name__)


def filtered_out_for_all_filters(node: dict, filters: list = []):
    """A helper function to see if `node` should be filtered out of a property values dictionary based on filter instances in `filters`

    Args:
        node (dict): a node from a property values dictionary
        filters (list, optional): a list of filter instances to filter `node`. Defaults to [].

    Returns:
        bool: whether `node` should be filtered out of a property values dictionary
    """
    for filter in filters:
        if filter.filtered_out(node):
            return True
    return False


class FilterMode(Enum):
    """A helper class to define the filtering mode of a filter class
    """

    INCLUDE = 0
    """Designate that the filter should only include marked/identified nodes
    """
    EXCLUDE = 1
    """Designate that the filter should include all nodes, except those marked/identified.
    """


class BaseFilter:
    """A base class for filtering nodes in a property values dictionary"""
    def __init__(self):
        self.validator = PropertyValuesDictValidator()

    def filtered_out(self, node: dict):
        """Returns whether `node` should be filtered out by this filter instance

        Must be overridden by subclasses. If a subclass only deals with value
        nodes, then it must validate `node` using `self.validator` and return
        False for step nodes. If a subclass only filters out step nodes, and not
        value nodes, it must return False for value nodes.

        Args:
            node (dict): a node in the property values dictionary
        """
        raise NotImplementedError(
            f"`filtered_out` must be overridden by class {type(self).__name__}"
        )


class PropertyTypeFilter(BaseFilter):
    """A filter class for filtering based on property type py_names

    Example:
        .. code-block:: python
            filter = PropertyTypeFilter(include=['ge_freq', 'qscale'], exact_match=False)
    """
    def __init__(self,
                 mode: FilterMode = FilterMode.EXCLUDE,
                 py_name_set=None,
                 exact_match=False):
        """Creates a `PropertyTypeFilter` instance

        Args:
            mode (FilterMode): the mode (include/exclude) for the filter instance. Defaults to EXCLUDE.
            py_name_set (set, optional): the default set of py_names to include or exclude. Defaults to None.
            exact_match (bool, optional): whether to use exact matching for py_names and the include/exclude_set. Defaults to False for suffix matching.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.mode = mode
        if not isinstance(self.mode, FilterMode):
            raise ValueError(
                f"mode must be an instance of FilterMode: mode={mode}")

        self.py_name_set = set(py_name_set)
        self.exact_match = exact_match

    def __py_name_filtered_out_exact(self, py_name: str):
        """Returns whether `py_name` is filtered out, using exact matching

        Args:
            py_name (str): the property type pythonic name

        Returns:
            bool: True if `py_name` is filtered out by the filter instance
        """
        if self.mode == FilterMode.INCLUDE:
            return py_name not in self.py_name_set
        elif self.mode == FilterMode.EXCLUDE:
            return py_name in self.py_name_set
        return False

    def __py_name_filtered_out_suffix(self, py_name: str):
        """Returns whether `py_name` is filtered out, using suffix matching

        Args:
            py_name (str): the property type pythonic name

        Returns:
            bool: True if `py_name` is filtered out by the filter instance
        """
        if self.mode == FilterMode.INCLUDE:
            return all(
                [not py_name.endswith(suffix) for suffix in self.py_name_set])
        elif self.mode == FilterMode.EXCLUDE:
            return any(
                [py_name.endswith(suffix) for suffix in self.py_name_set])
        return False

    def py_name_filtered_out(self, py_name):
        """Returns whether `py_name` is allowed - i.e., not filtered out - using matching scheme from `self.exact_match`.

        Args:
            py_name (str): the property type pythonic name

        Returns:
            bool: True if `py_name` is not filtered out by the filter instance
        """
        if self.exact_match:
            return self.__py_name_filtered_out_exact(py_name=py_name)
        else:
            return self.__py_name_filtered_out_suffix(py_name=py_name)

    def filtered_out(self, node: dict):
        """Returns whether value node `node` is filtered out, based on its 'property_type' value

        Args:
            node (dict): a node of a property values dictionary

        Returns:
            bool: True if `node` should be filtered out of a property values dictionary. False if it isn't filtered out, or is not a value node
        """
        # Only filter value nodes
        if not self.validator.is_value_node(node):
            return False

        return self.py_name_filtered_out(py_name=node['property_type'])


class QubitFilter(BaseFilter):
    """A filter class to filter based on value nodes' `qubit` field"""
    def __init__(self, mode: FilterMode = FilterMode.EXCLUDE, qubit_names=[]):
        """Creates a QubitFilter instance

        All qubit names in a node's `qubits` field must be in/or not in
        `self.qubit_names`; based on `mode`. For example, if
        `mode=FilterMode.INCLUDE`, a value node will be filtered if any qubit
        name in `node['qubits']` is not in `qubit_names`.

        Args:
            mode (Filtermode, optional): the filtering mode for QubitFilter. Defaults to FilterMode.EXCLUDE.
            qubit_names (list, optional): a list of qubit names to use for filtering, alongside mode. Defaults to [].
        """
        super().__init__()
        self.mode = mode
        self.qubit_names = qubit_names

    def __qubit_names_filtered_out(self, qubit_names: list):
        """Returns whether `qubit_names` indicate a node should be filtered out

        Args:
            qubit_names (list[str]): list of qubit names

        Returns:
            bool: whether a node identified by `qubit_names` should be filtered out of a property values dictionary
        """
        if self.mode == FilterMode.INCLUDE:
            # If we are in INCLUDE mode, all qubit names in qubit_names must be
            # in self.qubit_names
            return not all(
                [qb_name in self.qubit_names for qb_name in qubit_names])
        elif self.mode == FilterMode.EXCLUDE:
            # If we are in EXCLUDE moe, no qubit in qubit_names may be in
            # self.qubit_names
            return not any(
                [qb_name in self.qubit_names for qb_name in qubit_names])

    def filtered_out(self, node: dict):
        """Returns whether value node `node` is filtered out, based on its `qubits` value

        Args:
            node (dict): a node of a property values dictionary

        Returns:
            bool: True if `node` should be filtered out of a property values dictionary. False if it isn't filtered out, or is not a value node.
        """
        # Only filter property values nodes
        if not self.validator.is_value_node(node):
            return False

        if isinstance(node['qubits'], str):
            qubit_names = [node['qubits']]
        elif isinstance(node['qubits'], list) and isinstance(
                node['qubits'][0], str):
            qubit_names = node['qubits']
        else:
            log.error(f"Filtering out value node with unknown `qubits` value")
            log.error(node)
            return True

        return self.__qubit_names_filtered_out(qubit_names=qubit_names)
