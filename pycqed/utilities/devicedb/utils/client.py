"""This file contains general utilities for database client"""
import logging

log = logging.getLogger()

from device_db_client import model


def find_model_from_list(
    model_list,
    model_name,
    search_kwargs,
    log_empty_list=True,
    log_more_than_one_result=True,
):
    """Helper function to process the length of a database model list

    Example:
        device_name = "ATC75_M136_S17HW02"
        list_of_db_devices = client.get_api_instance().list_devices(name=device_name)
        db_device = devicedb.utils.find_model_from_list(
            model_list=list_of_db_devices,
            model_name="device",
            search_kwargs={"name":device_name,}
        )
        if db_device is None:
            raise ValueError(f"Could not find a device with name {device_name}")

    `search_kwargs` is equivalent to the fields used in the `list_<model_name>`
    call in :class:`~device_db_client.api.api_api.ApiApi`, such as
    :func:`~device_db_client.api.api_api.ApiApi.list_components`: e.g.,
    `list_components(**search_kwargs)`.

    Args:
        model_list (list): list returned from api client `list_<model_name>`
        model_name (str): name of the model, such as `device` or `wafer`
        search_kwargs (dict): dictionary of search terms with which to find the model
        log_empty_list (bool, optional): Whether to log if list is empty. Defaults to True.
        log_more_than_one_result (bool, optional): Whether to log if list length is larger than one. Defaults to True.

    Returns:
        model: the element in `model_list` chosen (first), or None if the length is 0
    """
    if len(model_list) == 0:
        if log_empty_list:
            log.error(
                f"Could not find a {model_name} with the provided search fields"
            )
            log.error(search_kwargs)
        return None
    if len(model_list) > 1:
        if log_more_than_one_result:
            log.warning(
                f"Found more than one {model_name} with the provided search fields, this is unexpected. Choosing the first."
            )
            log.debug(f"Provided search kwargs:")
            log.debug(search_kwargs)
    return model_list[0]


# TODO: Do we really want to check against this list, or is it easier to just use isinstance()?
def throw_if_not_db_model(model_instance):
    DB_MODEL_TYPES = [
        model.wafer.Wafer,
        model.device.Device,
        model.component.Component,
        model.component_type.ComponentType,
        model.coupling.Coupling,
        model.property_type.PropertyType,
        model.property_value.PropertyValue,
        model.unit.Unit,
        model.file_folder_raw_data.FileFolderRawData,
        model.experiment.Experiment,
        model.one_note_raw_data.OneNoteRawData,
    ]
    """List of model classes that can are valid database models.

    `device_db_client.model.verbosedevice.VerboseDevice` is not valid as it
    still represents a `Device`, just for an endpoint which should only be used
    for `GET`
    """
    if model_instance is None:
        raise ValueError(f"model instance cannot be None")
    if type(model_instance) not in DB_MODEL_TYPES:
        raise ValueError(
            f"model instance is not of a valid type: {type(model_instance)}. Must be type in DB_MODEL_TYPES."
        )


def noneless(**kwargs):
    """Returns a dictionary equivalent to kwargs but without keys whose values are None

    Returns:
        dict: the 'noneless' dictionary
    """
    return {k: v for k, v in kwargs.items() if v is not None}
