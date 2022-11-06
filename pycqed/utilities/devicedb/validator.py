import logging
from numbers import Number
from numpy import float64

log = logging.getLogger(__name__)


class PropertyValuesDictValidator:
    """Class to validate property values dictionaries
    """

    STEP_NODE_KEYS = ['step_type', 'property_values']
    """Keys that identify a step node"""
    STEP_NODE_ROOT_EXTRA_KEYS = ['timestamp']
    """Extra keys that identify a root step node"""
    VALUE_NODE_KEYS = [
        'qubits', 'component_type', 'property_type', 'value', 'timestamp',
        'rawdata_folder_path'
    ]
    """Keys that identify a value node"""
    def is_step_node(self, node):
        """Validates whether node is a valid step node

        Args:
            node (dict): a property values dictionary node

        Returns:
            bool: whether the node is a valid step node
        """
        if all(step_key in node.keys() for step_key in self.STEP_NODE_KEYS):
            return True
        return False

    def is_value_node(self, node):
        """Validates whether node is a valid value node

        Args:
            node (dict): a property values dictionary node

        Returns:
            bool: whether the node is a valid value node
        """
        if all(value_keys in node.keys()
               for value_keys in self.VALUE_NODE_KEYS):
            return True
        return False

    def is_root_step_node(self, node):
        """Validates whether node is a valid root step node

        A root step node is a step node with `timestamp` set.

        Args:
            node (dict): a property values dictionary node

        Returns:
            bool: whether the node is a valid root step node
        """
        if self.is_step_node(node):
            if all(root_step_key in node.keys()
                   for root_step_key in self.STEP_NODE_ROOT_EXTRA_KEYS):
                return True
            else:
                log.debug(
                    f"root step node is a step node but not a root step node")
        else:
            log.debug(f"root step node is not a step node")
        return False

    def is_valid_node(
        self,
        node: dict,
    ):
        """Validates a single node (dict) of a property values dictionary

        A node is either a step node or a value node. A value node is identified
        by the presence of a `value` key, and associated metadata such as
        `qubits` and `property_type`. A step node is identified by a `step_type`
        key, and a list of other nodes, called `property_values`. A root step
        node must also contain a `timestamp` entry.

        Args:
            node (dict): a node from a property values dictionary

        Returns:
            bool: whether the node is a valid property values dictionary node
        """
        if self.is_step_node(node):
            # The node is a step node, check types
            if not isinstance(node['step_type'], str):
                log.debug("step_type value must be a str instance")
                return False
            if not isinstance(node['property_values'], list):
                log.debug("`property values` value must be a list")
                return False
        elif self.is_value_node(node):
            # The node is a value node.

            # Check if `qubits` is a string or list of string. This condition is
            # true if `qubits` is neither a string or list OR if it is a list
            # but not of strings.
            if ((not isinstance(node['qubits'], str)
                 and not isinstance(node['qubits'], list)) or
                (isinstance(node['qubits'], list)
                 and not all(isinstance(qb, str) for qb in node['qubits']))):
                log.debug(
                    "`qubits` must either be a qubit name (str) or a list of qubit names"
                )
                return False

            if not isinstance(node['component_type'], str):
                log.debug("component_type must be a py_name (str)")
                return False

            if not isinstance(node['property_type'], str):
                log.debug("property_type must be a py_name (str)")
                return False

            if not isinstance(node['value'], Number) and not isinstance(
                    node['value'], float64):
            # if not isinstance(node['value'], Number):
                log.debug("value must be a number, specifically a float")
                return False

            if not isinstance(node['rawdata_folder_path'], str):
                log.debug("rawdata_folder_path must be a str instance")
                return False

        else:
            log.debug(
                "Node does not contain the right keys to be a property values dictionary node"
            )
            return False

        return True

    def is_valid(self, property_values_dict):
        """Validates whether `property_values_dict` is a valid property values dictionary

        This function walks through `property_values_dict` and validates each node using
        `is_valid_node`.

        Args:
            property_values_dict (dict): the property values dictionary to validate

        Returns:
            bool: whether the dictionary, and all of its nodes, are valid
        """
        return self.__is_valid(property_values_dict=property_values_dict, is_root=True)

    def __is_valid(self, property_values_dict, is_root):
        """FOR INTERNAL CLASS USE ONLY! Allows for root control through `is_root`"""

        # Root node must be a step node
        if is_root and not self.is_root_step_node(property_values_dict):
            log.debug("property_values_dict root node must be a root step node")
            log.debug(f"{property_values_dict}")
            return False

        # Check this node
        if not self.is_valid_node(property_values_dict):
            log.debug("property_values_dict node is not valid")
            log.debug(f"{property_values_dict}")
            return False

        # Current node is valid, validate children if they exist
        if self.is_step_node(property_values_dict):
            for node in property_values_dict['property_values']:
                if not self.__is_valid(node, is_root=False):
                    log.debug(
                        "property_values_dict node is not valid as child node is not valid"
                    )
                    log.debug(f"parent node:")
                    log.debug(f"{property_values_dict}")
                    return False
        return True
