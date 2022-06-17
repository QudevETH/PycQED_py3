import logging

log = logging.getLogger(__name__)
import pycqed.analysis.analysis_toolbox as a_tools
from .property_values import add_prefix_to_property_type_name


def experiment_func_folder_path_from_timestamps(timestamps):
    """Helper function to call `a_tools.get_folder`

    Args:
        timestamps (List[str]): list of timestamps, such as from `analysis.timestamps`

    Returns:
        str: the absolute path to the experiment folder for the first timestamp in `timestamps`
    """
    if len(timestamps) == 0:
        return None
    return a_tools.get_folder(timestamp=timestamps[0])


class ValueNodeCreator:
    """Helper class for creating value nodes with defaults for a given qubit, transition, sweet-spot, and timestamp

    The following example is taken from :func:`RamseyStep.get_device_property_values`.

    Example:
        .. code-block:: python
            property_values_dict = self.get_empty_device_properties_dict()
            sweet_spots = kwargs.get('qubit_sweet_spots', {})

            for qubit_name, qubit_results in analysis_params_dict.items():
                # This transition is not stored in RamseyAnalysis, so we must
                # get it from the settings parameters
                transition = self.get_param_value('transition_name', qubit=qubit_name)
                node_creator = db_utils.ValueNodeCreator(qubit_name,
                                                        self.analysis.timestamps[0],
                                                        sweet_spots.get(qubit_name))

                if 'exp_decay' in qubit_results.keys(
                ) and 'T2_star' in qubit_results['exp_decay'].keys():
                    property_values_dict['property_value'].append(
                        node_creator.create_node(
                            property_type='t2_star',
                            value=qubit_results['exp_decay']['T2_star']))

                if 'exp_decay' in qubit_results.keys(
                ) and f"new_{transition}_freq" in qubit_results['exp_decay'].keys():
                    property_values_dict['property_value'].append(
                        node_creator.create_node(
                            property_type='freq',
                            value=qubit_results['exp_decay']['new_{transition}_freq']))

                if 'T2_echo' in qubit_results.keys():
                    property_values_dict['property_values'].append(
                        node_creator.create_node(property_type='t2_echo',
                                                value=qubit_results['T2_echo']))
                return property_values_dict

    """
    def __init__(self,
                 qubits,
                 timestamp=None,
                 transition=None,
                 sweet_spots=None):
        """Creates a `ValueNodeCreator` instance

        Args:
            qubits (str|list[str]): the qubit name/s for all value nodes to be created
            timestamp (str, optional): the timestamp string for the experiment folder paths. Defaults to None.
            transition (str, optional): the transition for the value nodes. Defaults to None.
            sweet_spot (str, optional): the sweet spot for `qubits`. Defaults to None.

        Raises:
            ValueError: Raised if an input argument is not provided
        """
        if qubits is None:
            raise ValueError('qubits cannot be None')
        self.qubits = qubits
        self.sweet_spot = sweet_spots
        self.transition = transition
        self.timestamp = timestamp
        if self.timestamp is not None:
            self.folder_path = a_tools.get_folder(self.timestamp)

    def create_node(
        self,
        property_type,
        value,
        component_type='qb',
        sweet_spot=None,
        transition=None,
        timestamp=None,
        folder_path=None,
    ):
        """Creates a value node

        `create_node` will look at any defaults passed in the constructor of the
        `ValueNodeCreator` instance, if the argument is None in the call to
        `create_node`.

        `component_type` defaults to 'qb', but can be overwritten if needed. The
        value node's property type py_name is processed by
        `add_prefix_to_property_type_name` using the `sweet_spot` and
        `transition`. Therefore, if they are set to None, the resulting property
        type py_name will be modified accordingly. If `sweet_spot` or
        `transition` is None in `create_node`, the default will be taken from
        the constructor `ValueNodeCreator.__init__`.

        If `timestamp` is provided, in `create_node` or `__init__`, and
        `folder_path` is None; then `folder_path` will be set using
        `a_tools.get_folder(timestamp)`. If `timestamp` is passed to `__init__`,
        and both `folder_path` and `timestamp` are None in `create_node`; then
        the folder_path will be set using `a_tools.get_folder(self.timestamp)`.

        Args:
            property_type (str): the property type py_name for the value node, excluding transition and sweet spot
            value (float): the property value for the value node
            component_type (str, optional): the component type py_name for the value node. Defaults to 'qb'.
            sweet_spot (str, optional): the sweet spot for the value node, if needed. Defaults to None.
            transition (str, optional): the transition for the property type py_name. Defaults to None.
            timestamp (str, optional): the timestamp to be used for the experiment folder path. Defaults to None.
            folder_path (str, optional): the experiment folder path, if not based on `timestamp`. Defaults to None.

        Raises:
            ValueError: Raised if a required input argument is not provided

        Returns:
            dict: the value node
        """
        # Handle object defaults and function kwargs for timestamp, folder_path,
        # sweet_spot, and transition
        if timestamp is not None:
            _func_timestamp = timestamp
        elif self.timestamp is not None:
            _func_timestamp = self.timestamp
        else:
            raise ValueError('Cannot create a value node without a timestamp')

        if folder_path is not None:
            _func_folder_path = folder_path
        elif _func_timestamp is not None:
            _func_folder_path = a_tools.get_folder(_func_timestamp)
        elif self.folder_path is not None:
            _func_folder_path = self.folder_path
        else:
            raise ValueError('Cannot create value without a folder path')

        if sweet_spot is not None:
            _func_sweet_spot = sweet_spot
        elif self.sweet_spot is not None:
            _func_sweet_spot = self.sweet_spot
        else:
            _func_sweet_spot = None

        if transition is not None:
            _func_transition = transition
        elif self.transition is not None:
            _func_transition = self.transition
        else:
            _func_transition = None

        node = {}
        node['qubits'] = self.qubits
        node[
            'component_type'] = component_type if component_type is not None else 'qb'
        node['property_type'] = add_prefix_to_property_type_name(
            property_type, _func_sweet_spot, _func_transition)
        node['value'] = value
        node['timestamp'] = _func_timestamp
        node['rawdata_folder_path'] = _func_folder_path
        return node