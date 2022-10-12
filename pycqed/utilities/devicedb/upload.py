import logging
from datetime import datetime

import device_db_client

from .client import Client
from .filters import filtered_out_for_all_filters
from .validator import PropertyValuesDictValidator

log = logging.getLogger(__name__)


def WalkValueNodes(property_values_dict, filters=[]):
    """Iterator for all value nodes in the property values dictionary

    Args:
        property_values_dict (dict): the property values dictionary to walk
        filters (list, optional): list of filters to filter value nodes in `property_values_dict`

    Yields:
        value_node (dict): any value nodes found in the property values dictionary
    """
    validator = PropertyValuesDictValidator()
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
    """Iterator for the last value node - per unique `qubits`, `component_type`, and `property_type` combination - in the property values dictionary

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
        # If a previous node was found, this new node will overwrite it in last_value_nodes
        last_value_nodes[(node['qubits'], node['component_type'],
                          node['property_type'])] = node

    # Iterate over them
    for _, node in last_value_nodes.items():
        yield node


class PropertyValueUploader:
    def __init__(self,
                 client: Client,
                 property_values_dict,
                 experiment_comments=None,
                 db_experiment=None,
                 host="Q",
                 only_upload_latest=True,
                 filters: list = []):
        """Creates a `PropertyValueUploader` to assist with uploading a property values dictionary to the device database

        Args:
            client (Client): the device database client for uploading property values
            property_values_dict (dict): the property values dictionary to upload
            experiment_comments (str, optional): an optional comment to use when creating a experiment when uploading from this PropertyValueUploader instance. Defaults to None.
            db_experiment (device_db_client.model.experiment.Experiment, optional): optional experiment instance to use instead of creating one. Defaults to None.
            host (str, optional): the 'host' to use for all `FileFolderRawData` instances. If raw data is not stored on the Q-Drive, set this to the hostname of the machine. Defaults to "Q".
            only_upload_latest (bool, optional): whether to filter value nodes in `property_values_dict` to only the last unique nodes. Defaults to True.
            filters (list, optional): a list of filters to apply to `property_values_dict`. Defaults to no filters.

        Raises:
            ValueError: if `client` doesn't have a device in the database
            ValueError: if any input argument is of an incorrect type
            ValueError: if `property_values_dict` is not a valid property values dictionary
        """
        self.client = client
        self.validator = PropertyValuesDictValidator()
        self.property_values_dict = property_values_dict
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
                f"PropertyValueUploader requires that the database client is using a device with a valid entry in the database"
            )

        # Set db_experiment, create if None, throw if it's an unknown type
        if db_experiment is None:
            self.db_experiment = self.create_experiment_instance(
                experiment_comments=self.experiment_comments)
        elif not isinstance(db_experiment,
                            device_db_client.model.experiment.Experiment):
            raise ValueError(
                f"db_experiment must be of type `Experiment` from the database client module. Instead, its type is {type(db_experiment)}"
            )
        else:
            self.db_experiment = db_experiment
        # Validate property values dict
        if not self.validator.is_valid(self.property_values_dict):
            raise ValueError(f"Property values dictionary is not valid")

        # Extract value nodes so we can deal with `not_uploaded_value_nodes` and `uploaded_value_nodes` instead of property_values_dict
        self.__extract_value_nodes()

    def __timestamp_to_datetime(self, timestamp):
        TIMESTAMP_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
        return datetime.strptime(timestamp, TIMESTAMP_DATETIME_FORMAT)

    def create_experiment_instance(self, experiment_comments=None):
        """Creates a `Experiment` instance for all property values uploaded by `PropertyValueUploader`

        Args:
            experiment_comments (str, optional): an optional comment for the `Experiment` instance. Defaults to None.

        Raises:
            ValueError: if the property values dictionary is not valid

        Returns:
            Experiment: the `Experiment` instance
        """
        # validate property_values_dict
        if not self.validator.is_valid(self.property_values_dict):
            raise ValueError(
                f"property_values_dict is not valid, cannot create an experiment instance"
            )
        experiment = device_db_client.model.experiment.Experiment(
            datetime_taken=self.__timestamp_to_datetime(
                self.property_values_dict['timestamp']),
            type=self.property_values_dict['step_type'],
            comments=experiment_comments,
        )
        experiment = self.client.get_api_instance().create_experiment(
            experiment=experiment)
        return experiment

    def __extract_value_nodes(self):
        """Extracts value nodes from the property values dictionary

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
        for value_node in iterator(self.property_values_dict, self.filters):
            if value_node is not None:
                self.not_uploaded_value_nodes.append(value_node)
            else:
                log.debug(f"Encountered None value node: {value_node}")
        log.debug(
            f"Extracted {len(self.not_uploaded_value_nodes)} value nodes from property values dictionary"
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
                    f"PropertyValueUploader does not currently support value nodes with multiple qubits"
                )
            if (isinstance(qubits, list)
                    and not isinstance(qubits[0], str)) or not isinstance(
                        qubits, str):
                raise ValueError(
                    f"qubits must be a list of qubit names or a single qubit name (str): received {qubits}"
                )
            qubit_py_name_num = qubits if isinstance(qubits,
                                                     str) else qubits[0]
            if value_node['component_type'] == 'qb':
                # If the component type is a qubit, the property value is for the qubit itself
                component = self.client.get_component_for(
                    py_name_num=qubit_py_name_num)
            else:
                # If the component type is not a qubit, the property value is for an associated component
                component_type = self.client.get_component_type_for(
                    py_name=value_node['component_type'])
                # get the qubit
                qubit = self.client.get_component_for(
                    py_name_num=qubit_py_name_num)
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
                    # Could not find the component as an associated component for qubit
                    raise ValueError(
                        f"Could not find an associated component of type {value_node['component_type']} for qubits {value_node['qubits']}"
                    )
            # Get the property type
            property_type = self.client.get_property_type_for(
                py_name=value_node['property_type'])
            if property_type is None:
                raise ValueError(
                    f"Could not find a property type with py_name {value_node['property_type']}"
                )

            # Check that the value_node value is within the property type range
            if property_type.min_value is not None and value_node[
                    'value'] < property_type.min_value:
                raise ValueError(
                    f"Value node's value is smaller than the allowed minimum value for the property type:\n\tvalue = {value_node['value']} < {property_type.min_value} = min_value"
                )
            if property_type.max_value is not None and value_node[
                    'value'] > property_type.max_value:
                raise ValueError(
                    f"Value node's value is larger than the allowed maximum value for the property type:\n\tvalue = {value_node['value']} > {property_type.max_value} = max_value"
                )

            # Create objects on the database
            if not dry_run:
                # We now have the component for the raw_data and property value
                raw_data = device_db_client.model.file_folder_raw_data.FileFolderRawData(
                    host=self.host,
                    path=value_node['rawdata_folder_path'],
                    is_file=False,
                    timestamp=value_node['timestamp'],
                )
                raw_data = self.client.get_or_create_filefolder_raw_data(
                    raw_data)
                property_value = device_db_client.model.property_value.PropertyValue(
                    value=float(value_node['value']),
                    type=property_type.id,
                    experiment=self.db_experiment.id,
                    raw_data=[raw_data.id],
                    component=component.id,
                )
                try:
                    property_value = self.client.create_property_value(
                        property_value=property_value)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create the property value. Failed with error {e}"
                    )
                if set_accepted:
                    try:
                        self.client.get_api_instance(
                        ).set_accepted_property_value(
                            id=str(property_value.id))
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to set property value as accepted, with error {e}"
                        )
            else:
                log.info(
                    f"Dry-run fake upload of: {value_node['qubits']}, {value_node['property_type']}={value_node['value']}"
                )
            return True
        except Exception as e:
            log.error(f"Failed to upload value node with error {e}")
            log.error(f"value node is '{value_node}'")
            return False

    def upload(
        self,
        set_all_accepted=True,
        dry_run=False,
    ):
        """Upload all value nodes in `self.not_uploaded_value_nodes`

        `self.not_uploaded_value_nodes` should contain all value nodes from
        `self.property_values_dict` after instantiation. Use `dry_run=True` to test for
        which values will be uploaded. Under dry-run, `upload()` will print
        each value node, that would have been uploaded, to the console.

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
                f"Starting upload of {len(self.not_uploaded_value_nodes)} value nodes"
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
                    # the upload failed, add value_node to new_not_uploaded_value_nodes
                    new_not_uploaded_value_nodes.append(value_node)
                    failed_upload_count += 1
        if not dry_run:
            self.not_uploaded_value_nodes = new_not_uploaded_value_nodes
        log.info(
            f"Successfully uploaded {successful_upload_count} value_node instance/s"
        )
        log.info(
            f"Failed to upload {failed_upload_count} value node instance/s, you can try again"
        )
        return failed_upload_count == 0
