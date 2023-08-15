import logging
import sys
from datetime import datetime
import re

import device_db_client

from .client import Client
from .filters import filtered_out_for_all_filters
from .validator import DevicePropertyValuesDictValidator

log = logging.getLogger(__name__)


def WalkValueNodes(property_values_dict, filters=[]):
    """Iterator for all value nodes in the property values dictionary.

    Args:
        property_values_dict (dict): the property values dictionary to walk
        filters (list, optional): list of filters to filter value nodes in `property_values_dict`

    Yields:
        value_node (dict): any value nodes found in the property values dictionary
    """
    validator = DevicePropertyValuesDictValidator()
    if validator.is_step_node(
            property_values_dict) and not filtered_out_for_all_filters(
                property_values_dict, filters):
        for child_node in property_values_dict['property_values']:
            for value_node in WalkValueNodes(child_node, filters=filters):
                yield value_node
    elif validator.is_value_node(
            property_values_dict) and not filtered_out_for_all_filters(
                property_values_dict, filters):
        yield property_values_dict


def WalkLastUniqueValueNodes(property_values_dict, filters=[]):
    """Iterator for the last value node - per unique `qubits`, `component_type`,
    and `property_type` combination - in the property values dictionary

    This iterator will only output the last value nodes for each combination of
    `qubits`, `component_type`, and `property_type`. This way, if multiple steps
    give results for the same property type and component, then only the latest
    one is returned. Some routines may iterate over a combination of steps until
    an optimal result is found. This iterator function makes the process of
    finding that final value node easier.

    **Important information**

    This iterator assumes that the `property_values` list in each step node of
    `property_values_dict` is ordered from oldest to newest. Therefore, the
    _last_ value node is the one further along in all 'property_values' of
    `property_values_dict`.

    Args:
        property_values_dict (dict): the property values dictionary to walk
        filters (list, optional): list of filters to filter value nodes in `property_values_dict`
    """
    # This is a dirty hack to find the last value node. We store the entire list
    # in memory instead of iterating over it all _properly_.
    last_value_nodes = {}
    # Find the last value nodes
    for node in WalkValueNodes(property_values_dict, filters=filters):
        # If a previous node was found,
        # this new node will overwrite it in last_value_nodes
        last_value_nodes[(node['qubits'], node['component_type'],
                          node['property_type'])] = node

    # Iterate over them
    for _, node in last_value_nodes.items():
        yield node


class DevicePropertyValueUploader:
    def __init__(self,
                 client: Client,
                 property_values_dict,
                 experiment_comments=None,
                 db_experiment=None,
                 host="Q",
                 only_upload_latest=True,
                 filters: list = []):
        """Creates a `DevicePropertyValueUploader` to assist with uploading a
        property values dictionary to the device database.

        Args:
            client (Client): the device database client for uploading property values
            property_values_dict (dict): the device property values dictionary to upload
            experiment_comments (str, optional): an optional comment to use when creating a experiment when uploading from this DevicePropertyValueUploader instance. Defaults to None.
            db_experiment (device_db_client.model.experiment.Experiment, optional): optional experiment instance to use instead of creating one. Defaults to None.
            host (str, optional): the 'host' to use for all `FileFolderRawData` instances. If raw data is not stored on the Q-Drive, set this to the hostname of the machine. Defaults to "Q".
            only_upload_latest (bool, optional): whether to filter value nodes in `property_values_dict` to only the last unique nodes. Defaults to True.
            filters (list, optional): a list of filters to apply to `property_values_dict`. Defaults to no filters.

        Raises:
            ValueError: if `client` doesn't have a device in the database
            ValueError: if any input argument is of an incorrect type
            ValueError: if `device_property_values_dict` is not a valid device property values dictionary
        """
        self.client = client
        self.validator = DevicePropertyValuesDictValidator()
        self.device_property_values_dict = property_values_dict
        self.experiment_comments = experiment_comments
        self.not_uploaded_value_nodes = []
        self.uploaded_value_nodes = []
        self.host = host
        self.only_upload_latest = only_upload_latest
        self.filters = filters

        ## VALIDATION OF INPUT PARAMETERS ##
        # Check if the client has a device on the database, which is required
        if not self.client.has_db_device:
            raise ValueError(
                'DevicePropertyValueUploader requires that the database client '
                'is using a device with a valid entry in the database'
            )

        # Set db_experiment, create if None, throw if it's an unknown type
        if db_experiment is None:
            self.db_experiment = self.create_experiment_instance(
                experiment_comments=self.experiment_comments)
        elif not isinstance(db_experiment,
                            device_db_client.model.experiment.Experiment):
            raise ValueError(
                'db_experiment must be of type `Experiment` from the database '
                f'client module. Instead, its type is {type(db_experiment)}'
            )
        else:
            self.db_experiment = db_experiment
        # Validate property values dict
        if not self.validator.is_valid(self.device_property_values_dict):
            raise ValueError(f"Property values dictionary is not valid")

        # Extract value nodes so we can deal with `not_uploaded_value_nodes`
        # and `uploaded_value_nodes` instead of property_values_dict
        self.__extract_value_nodes()

    def __timestamp_to_datetime(self, timestamp):
        TIMESTAMP_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
        return datetime.strptime(timestamp, TIMESTAMP_DATETIME_FORMAT)

    def create_experiment_instance(self, experiment_comments=None):
        """Creates a `Experiment` instance for all property values uploaded by
        `DevicePropertyValueUploader`.

        Args:
            experiment_comments (str, optional): an optional comment for the `Experiment` instance. Defaults to None.

        Raises:
            ValueError: if the property values dictionary is not valid

        Returns:
            Experiment: the `Experiment` instance
        """
        # validate property_values_dict
        if not self.validator.is_valid(self.device_property_values_dict):
            raise ValueError(
                'property_values_dict is not valid, cannot create an '
                'experiment instance'
            )
        experiment = device_db_client.model.experiment.Experiment(
            datetime_taken=self.__timestamp_to_datetime(
                self.device_property_values_dict['timestamp']),
            type=self.device_property_values_dict['step_type'],
            comments=experiment_comments,
        )
        experiment = self.client.get_api_instance().create_experiment(
            experiment=experiment)
        return experiment

    def __extract_value_nodes(self):
        """Extracts value nodes from the property values dictionary.

        If `self.only_upload_latest` is `True`, only the latest unique value
        nodes in `self.property_values_dict` will be extracted. See
        `WalkValueNodes` and `WalkLastUniqueValueNodes` for more information on
        how the nodes are extracted.

        If there are filters in `self.filters`, only nodes which are not
        filtered out will be extracted.
        """
        if self.only_upload_latest:
            iterator = WalkLastUniqueValueNodes
        else:
            iterator = WalkValueNodes
        for value_node in iterator(
            self.device_property_values_dict,
            self.filters
        ):
            if value_node is not None:
                self.not_uploaded_value_nodes.append(value_node)
            else:
                log.debug(f"Encountered None value node: {value_node}")
        log.debug(
            f'Extracted {len(self.not_uploaded_value_nodes)} value nodes from '
            'property values dictionary'
        )

    def upload_value_node(
        self,
        value_node,
        set_accepted=True,
        dry_run=False,
    ):
        """Upload a single value node

        If an error is encountered, `False` is returned and the error is logged
        as an error. If this happens, `value_node` is logged at at the DEBUG
        level.Use `dry_run=True` to test for which values will be uploaded.
        Under dry-run, `upload_value_node()` will execute and print errors
        etc., but will not make the final upload to the database.

        Args:
            value_node (dict): the valid value node to upload
            set_accepted (bool, optional): whether to set the uploaded property value as _accepted_. Defaults to True.
            dry_run (bool, optional): If set to True, will run through the upload process but will not actually upload property values. Defaults to False.

        Returns:
            bool: if the upload was successful
        """
        try:
            qubits = value_node['qubits']
            if isinstance(qubits, list) and len(qubits) != 1:
                raise NotImplementedError(
                    'DevicePropertyValueUploader does not currently support '
                    'value nodes with multiple qubits'
                )
            if (isinstance(qubits, list)
                    and not isinstance(qubits[0], str)) or not isinstance(
                        qubits, str):
                raise ValueError(
                    'qubits must be a list of qubit names or a single qubit '
                    f'name (str): received {qubits}'
                )
            qubit_py_name_num = qubits if isinstance(qubits,
                                                     str) else qubits[0]
            if value_node['component_type'] == 'qb':
                # If the component type is a qubit,
                # the property value is for the qubit itself
                component = self.client.get_component_for(
                    py_name_num=qubit_py_name_num)
            else:
                # If the component type is not a qubit,
                # the property value is for an associated component
                component_type = self.client.get_component_type_for(
                    py_name=value_node['component_type'])

                # Get the qubit
                qubit = self.client.get_component_for(
                    py_name_num=qubit_py_name_num)
                if qubit == None:
                    raise RuntimeError(
                        f"Failed to get the qubit. Failed with error {e}"
                    )

                # Search for component in associated components
                component = None
                for associated_component in [
                    self.client.get_component_for(id=comp_id)
                    for comp_id in qubit.associated_components
                ]:
                    if associated_component.type == component_type.id:
                        component = associated_component
                        break
                # Handle no component found
                if component is None:
                    # Could not find the component as an associated component
                    # for qubit
                    raise ValueError(
                        'Could not find an associated component of type '
                        f'{value_node["component_type"]} for qubits '
                        f'{value_node["qubits"]}'
                    )
            # Get the device property type
            device_property_type = self.client.get_device_property_type_for(
                py_name=value_node['property_type'])
            if device_property_type==None:
                raise ValueError(
                    'Could not find a device property type with py_name '
                    f'{value_node["property_type"]}. Failed with error {e}'
                )

            # Check that the value_node value is within the
            # device property type range
            if device_property_type.min_value is not None and value_node[
                    'value'] < device_property_type.min_value:
                raise ValueError(
                    'Value node\'s value is smaller than the allowed minimum '
                    'value for the device property type:\n\tvalue = '
                    f'{value_node["value"]} < {device_property_type.min_value} '
                    '= min_value'
                )
            if device_property_type.max_value is not None and value_node[
                    'value'] > device_property_type.max_value:
                raise ValueError(
                    'Value node\'s value is larger than the allowed maximum '
                    'value for the device property type:\n\tvalue = '
                    f'{value_node["value"]} > {device_property_type.max_value} '
                    '= max_value'
                )

            if self.client.setup == None:
                raise ValueError(
                    'The config object was created without a valid setup. '
                    'Please specify the setup.'
                )

            # Create objects on the database
            if not dry_run:
                # We now have the component for the raw_data and
                # device property value
                try:
                    # Regex for extracting the routine name out of the rawdata path
                    # E.g. the path could be D:\pydata\20221213\225609_ef_T2_star
                    # while we want only ef_T2_star.
                    # Note that '\\\\' is needed in order to match a literal
                    # backslash (see https://docs.python.org/3/library/re.html)
                    # We want to store only the routine name (e.g. ef_T2_star)
                    # and not the whole path because the live and archive folder
                    # are stored in the `Setup` class in order to prevent
                    # redunant storage of all the folder paths.
                    regex_matches = re.search('(.*)\\\\(\d*)_(.*)', value_node['rawdata_folder_path'])
                    routine_name = regex_matches[3]

                    raw_data = device_db_client.model.timestamp_raw_data.TimestampRawData(
                        setup=int(self.client.setup.id),
                        routine_name=routine_name,
                        timestamp=value_node['timestamp'],
                    )
                    raw_data = self.client.get_or_create_timestamp_raw_data(
                        raw_data)
                except Exception as e:
                    raise RuntimeError(
                        'Failed to create the raw data object. Failed with '
                        f'error {e}'
                    )
                try:
                    device_property_value = device_db_client.model.device_property_value.DevicePropertyValue(
                        value=float(value_node['value']),
                        type=device_property_type.id,
                        experiment=self.db_experiment.id,
                        raw_data=[raw_data.id],
                        component=component.id,
                        device=self.client.db_device.id,
                    )
                    device_property_value = self.client.create_device_property_value(
                        device_property_value=device_property_value)
                except Exception as e:
                    raise RuntimeError(
                        'Failed to create the property value. Failed with '
                        f'error {e}'
                    )
                if set_accepted:
                    try:
                        self.client.get_api_instance(
                        ).set_accepted_device_property_value(
                            id=str(device_property_value.id))
                    except Exception as e:
                        raise RuntimeError(
                            'Failed to set property value as accepted, '
                            f'with error {e}'
                        )
            else:
                log.info(
                    f'Dry-run fake upload of: {value_node["qubits"]}, '
                    f'{value_node["property_type"]}={value_node["value"]}'
                )
            return True
        except Exception as e:
            log.error(f"Failed to upload value node with error {e}")
            log.error(f"value node is '{value_node}'")
            log.error(f'Exception raised in line {sys.exc_info()[2].tb_lineno} '
                'in upload.py')
            return False

    def upload(
        self,
        set_all_accepted=True,
        dry_run=False,
    ):
        """Upload all value nodes in `self.not_uploaded_value_nodes`.

        `self.not_uploaded_value_nodes` should contain all value nodes from
        `self.device_property_values_dict` after instantiation.
        Use `dry_run=True` to test for  which values will be uploaded.
        Under dry-run, `upload()` will print each value node,
        that would have been uploaded, to the console.

        Args:
            set_all_accepted (bool, optional): whether to set all value nodes as accepted once uploaded. Defaults to True.
            dry_run (bool, optional): If set to True, will run through the upload process but will not actually upload property values. Defaults to False.

        Returns:
            bool: whether all remaining value nodes were uploaded
        """
        if len(self.not_uploaded_value_nodes) == 0:
            log.warning(f"There are no value nodes left to upload")
            return True
        else:
            log.info(
                f'Starting upload of {len(self.not_uploaded_value_nodes)} '
                'value nodes'
            )
        new_not_uploaded_value_nodes = []
        successful_upload_count = 0
        failed_upload_count = 0
        for value_node in self.not_uploaded_value_nodes:
            if self.upload_value_node(
                    value_node,
                    set_accepted=set_all_accepted,
                    dry_run=dry_run,
            ):
                # if we weren't in dry-run mode
                if not dry_run:
                    # the upload was a success, add value_node to uploaded list
                    self.uploaded_value_nodes.append(value_node)
                    successful_upload_count += 1
            else:
                if not dry_run:
                    # the upload failed,
                    # add value_node to new_not_uploaded_value_nodes
                    new_not_uploaded_value_nodes.append(value_node)
                    failed_upload_count += 1
        if not dry_run:
            self.not_uploaded_value_nodes = new_not_uploaded_value_nodes
        log.info(
            f'Successfully uploaded {successful_upload_count} value_node '
            'instance(s)'
        )
        log.info(
            f'Failed to upload {failed_upload_count} value node instance(s), '
            'you can try again'
        )
        return failed_upload_count == 0
