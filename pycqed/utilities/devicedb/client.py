import logging
from typing import Optional

log = logging.getLogger(__name__)

import device_db_client
from device_db_client import model
from device_db_client.api import api_api
from pycqed.utilities.devicedb import decorators, utils


class Config:
    @decorators.at_least_one_not_none(['username', 'token'])
    @decorators.all_or_none(['username', 'password'])
    def __init__(
        self,
        username=None,
        password=None,
        token=None,
        device_name=None,
        host="https://device-db.qudev.phys.ethz.ch",
    ):
        """Creates a config object for configuring a `Client` instance, mostly for authentication

        If `token` is provided, the client will use Bearer Authentication
        instead of Basic Authentication which uses `username` and `password`. If
        `token` is not provided, `username` and `password` must be provided.

        Args:
            username (str, optional): the username of the user with which to authenticate. Defaults to None.
            password (str, optional): the password for the user identified by username. Defaults to None.
            token (str, optional): the access token for a user on the database. Defaults to None.
            device_name (str, optional): the name of the device to retrieve from the database, if needed. Defaults to None.
            host (str, optional): the url for the server hosting the device database. Defaults to "https://device-db.qudev.phys.ethz.ch".
        """
        self.username = username
        self.password = password
        self.token = token
        self.device_name = device_name
        self.host = host


class Client:
    """A class to manage connections to the device database through PycQED"""

    __db_device = None
    """Cache of the database representation of the device. None indicates the device does not exist on the database."""
    __has_checked_for_db_device = False
    """Marks whether __db_device has been set. Used by `def db_device` and `refresh_db_device()."""

    config: Config = None
    """The client configuration"""
    api_config: device_db_client.Configuration = None
    """The configuration for the internal device db client library (do not edit)"""
    api_client: device_db_client.ApiClient = None
    """The internal api client"""
    def __init__(self, config: Config):
        """Constructs a device database client

        Args:
            config (Config): contains the configuration parameters for the client
        """
        self.config = config
        self.api_config = device_db_client.Configuration(
            host=self.config.host,
            username=self.config.username,
            password=self.config.password,
            access_token=self.config.token,
        )
        self.api_client = device_db_client.ApiClient(self.api_config)

    @property
    def db_device(self):
        """The database representation for the device for this client, None if the device does not exist on the database"""
        if not self.__has_checked_for_db_device:
            self.refresh_db_device()
        return self.__db_device

    @property
    def has_db_device(self):
        """Whether a device exists on the database identified by `self.config.device_name`"""
        return self.db_device is not None

    def refresh_db_device(self):
        """Updates the internal database device instance from the online database, based on `self.config.device_name`"""
        if self.config.device_name is not None:
            self.__db_device = self.get_device_for(
                name=self.config.device_name)

    def get_api_instance(self):
        """Returns the internal API client's api instance, which can make direct calls to the database server

        Returns:
            device_db_client.api_api.ApiApi: the associated api instance
        """
        return api_api.ApiApi(self.api_client)

    def successful_connection(self):
        """Tests that the client can connect to the database

        Returns:
            bool: whether the client can successfully connect to the database
        """
        try:
            api = self.get_api_instance()
            response = api.list_wafers()
            log.info(f"Successfully connected to device database")
            log.debug(f"Connection test (list_wafers) response: {response}")
            return True
        except device_db_client.ApiException as e:
            log.warning(
                f"Failed to connect to device database with API exception: {e}"
            )
            return False
        except Exception as e:
            log.info(f"Failed to connect to device database: {e}")
            return False

    #################################
    # MODEL get_or_create FUNCTIONS #
    #################################

    # TODO: See if there's a simpler way of implementing the get_or_create functions
    # These functions are effectively the same, the only real difference is
    # which variables we use to uniquely identify an instance of the model,
    # other than their id. If there's a way of doing this without implementing
    # each function separately, that would be great. #technical_debt

    def get_or_create_wafer(
        self,
        wafer: device_db_client.model.wafer.Wafer,
    ):
        """Gets a wafer from the DB, creates it if not found.

        This function will get a wafer from the database, if it exists, based on
        the uniquely identifying fields of the class. If such an instance of
        wafer doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            wafer (device_db_client.model.wafer.Wafer): the wafer to get from the db, or create

        Returns:
            wafer: the updated wafer instance from the database
        """
        utils.throw_if_not_db_model(wafer)
        maybe_wafer = self.get_wafer_for(name=wafer.name, log_empty_list=False)
        if maybe_wafer is None:
            log.info(f"Could not find a wafer, creating one instead")
            return self.get_api_instance().create_wafer(wafer=wafer)
        else:
            log.debug(f"Found an already existing wafer")
            return maybe_wafer

    def get_or_create_device(
        self,
        device: device_db_client.model.device.Device,
    ):
        """Gets a device from the DB, creates it if not found.

        This function will get a device from the database, if it exists, based
        on the uniquely identifying fields of the class. If such an instance of
        device doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            device (device_db_client.model.device.Device): the device to get from the db, or create

        Returns:
            device: the updated device instance from the database
        """
        utils.throw_if_not_db_model(device)
        maybe_device = self.get_device_for(name=device.name,
                                           log_empty_list=False)
        if maybe_device is None:
            log.info(f"Could not find a device, creating one instead")
            return self.get_api_instance().create_device(device=device)
        else:
            log.debug(f"Found an already existing device")
            return maybe_device

    def get_or_create_component_type(
        self,
        component_type: device_db_client.model.component_type.ComponentType,
    ):
        """Gets a component type from the DB, creates it if not found.

        This function will get a component type from the database, if it exists,
        based on the uniquely identifying fields of the class. If such an
        instance of component type doesn't exist on the database, it will be
        created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            component_type (device_db_client.model.component_type.ComponentType): the component type to get from the db, or create

        Returns:
            component_type: the updated component type instance from the database
        """
        utils.throw_if_not_db_model(component_type)
        maybe_component_type = self.get_component_type_for(
            py_name=component_type.py_name, log_empty_list=False)
        if maybe_component_type is None:
            log.info(f"Could not find a component_type, creating one instead")
            return self.get_api_instance().create_component_type(
                component_type=component_type)
        else:
            log.debug(f"Found an already existing component_type")
            return maybe_component_type

    def get_or_create_property_type(
        self,
        property_type: device_db_client.model.property_type.PropertyType,
    ):
        """Gets a property type from the DB, creates it if not found.

        This function will get a property type from the database, if it exists,
        based on the uniquely identifying fields of the class. If such an
        instance of property type doesn't exist on the database, it will be
        created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            property_type (device_db_client.model.property_type.PropertyType): the property type to get from the db, or create

        Returns:
            property_type: the updated property type instance from the database
        """
        utils.throw_if_not_db_model(property_type)
        maybe_property_type = self.get_property_type_for(
            py_name=property_type.py_name, log_empty_list=False)
        if maybe_property_type is None:
            log.info(f"Could not find a property_type, creating one instead")
            return self.get_api_instance().create_property_type(
                property_type=property_type)
        else:
            log.debug(f"Found an already existing property_type")
            return maybe_property_type

    def get_or_create_filefolder_raw_data(
        self,
        filefolder_raw_data: device_db_client.model.file_folder_raw_data.
        FileFolderRawData,
    ):
        """Gets a file/folder raw data from the DB, creates it if not found.

        This function will get a file/folder raw data from the database, if it
        exists, based on the uniquely identifying fields of the class. If such
        an instance of file/folder raw data doesn't exist on the database, it
        will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            filefolder_raw_data (device_db_client.model.file_folder_raw_data.FileFolderRawData): the file/folder raw data to get from the db, or create

        Returns:
            filefolder_raw_data: the updated file/folder raw data instance from the database
        """
        utils.throw_if_not_db_model(filefolder_raw_data)
        maybe_filefolder_raw_data = self.get_filefolder_raw_data_for(
            host=filefolder_raw_data.host,
            path=filefolder_raw_data.path,
            log_empty_list=False)
        if maybe_filefolder_raw_data is None:
            log.info(
                f"Could not find a filefolder_raw_data, creating one instead")
            return self.get_api_instance().create_file_folder_raw_data(
                file_folder_raw_data=filefolder_raw_data)
        else:
            log.debug(f"Found an already existing filefolder_raw_data")
            return maybe_filefolder_raw_data

    def get_or_create_onenote_raw_data(self,
                                       onenote_raw_data: device_db_client.
                                       model.one_note_raw_data.OneNoteRawData):
        """Gets a OneNote raw data from the DB, creates it if not found.

        This function will get a OneNote raw data from the database, if it
        exists, based on the uniquely identifying fields of the class. If such
        an instance of OneNote raw data doesn't exist on the database, it will
        be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            onenote_raw_data (device_db_client.model.one_note_raw_data.OneNoteRawData): the OneNote raw data to get from the db, or create

        Returns:
            onenote_raw_data: the updated OneNote raw data instance from the database
        """
        utils.throw_if_not_db_model(onenote_raw_data)
        maybe_onenote_raw_data = self.get_onenote_raw_data_for(
            host=onenote_raw_data.host,
            path=onenote_raw_data.path,
            section_id=onenote_raw_data.section_id,
            page_id=onenote_raw_data.page_id,
            log_empty_list=False)
        if maybe_onenote_raw_data is None:
            log.info(
                f"Could not find a onenote_raw_data, creating one instead")
            return self.get_api_instance().create_one_note_raw_data(
                one_note_raw_data=onenote_raw_data)
        else:
            log.debug(f"Found an already existing onenote_raw_data")
            return maybe_onenote_raw_data

    def get_or_create_unit(self, unit: device_db_client.model.unit.Unit):
        """Gets a unit from the DB, creates it if not found.

        This function will get a unit from the database, if it exists, based on
        the uniquely identifying fields of the class. If such an instance of
        unit doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            unit (device_db_client.model.unit.Unit): the unit to get from the db, or create

        Returns:
            unit: the updated unit instance from the database
        """
        utils.throw_if_not_db_model(unit)
        maybe_unit = self.get_unit_for(name=unit.name, log_empty_list=False)
        if maybe_unit is None:
            log.info(f"Could not find a unit, creating one instead")
            return self.get_api_instance().create_unit(unit=unit)
        else:
            log.debug(f"Found an already existing unit")
            return maybe_unit

    def get_or_create_component(
            self, component: device_db_client.model.component.Component):
        """Gets a component from the DB, creates it if not found.

        This function will get a component from the database, if it exists,
        based on the uniquely identifying fields of the class. If such an
        instance of component doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            component (device_db_client.model.component.Component): the component to get from the db, or create

        Returns:
            component: the updated component instance from the database
        """
        utils.throw_if_not_db_model(component)
        maybe_component = self.get_component_for(type=component.type,
                                                 device=component.device,
                                                 number=component.number,
                                                 log_empty_list=False)
        if maybe_component is None:
            log.info(f"Could not find a component, creating one instead")
            return self.get_api_instance().create_component(
                component=component)
        else:
            log.debug(f"Found an already existing component")
            return maybe_component

    def get_or_create_coupling(
            self, coupling: device_db_client.model.coupling.Coupling):
        """Gets a coupling from the DB, creates it if not found.

        This function will get a coupling from the database, if it exists, based
        on the uniquely identifying fields of the class. If such an instance of
        coupling doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            coupling (device_db_client.model.coupling.Coupling): the coupling to get from the db, or create

        Returns:
            coupling: the updated coupling instance from the database
        """
        utils.throw_if_not_db_model(coupling)
        maybe_coupling = self.get_coupling_for(
            component_ids=coupling.components, log_empty_list=False)
        if maybe_coupling is None:
            log.info(f"Could not find a coupling, creating one instead")
            return self.get_api_instance().create_coupling(coupling=coupling)
        else:
            log.debug(f"Found an already existing coupling")
            return maybe_coupling

    ###################################
    # MODEL get_<model>_for FUNCTIONS #
    ###################################

    @decorators.at_least_one_not_none(['id', 'name'])
    def get_wafer_for(self, id=None, name=None, **kwargs):
        """Get the wafer for the provided search terms

        Args:
            id (int|str): the primary key of the wafer instance on the database
            name (str): the identifying name for the wafer
        """
        search_kwargs = {
            "name": name,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            wafer_list = api.list_wafers(**search_kwargs)
            wafer = utils.find_model_from_list(
                wafer_list,
                'wafer',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                wafer = api.retrieve_wafer(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return wafer

    @decorators.at_least_one_not_none(['id', 'name'])
    def get_device_for(self, id=None, name=None, **kwargs):
        """Get the device for the provided search terms

        Args:
            id (int|str): the primary key of the device instance on the database
            name (str): the identifying name for the device
        """
        search_kwargs = {
            "name": name,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            device_list = api.list_devices(**search_kwargs)
            device = utils.find_model_from_list(
                device_list,
                'device',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                device = api.retrieve_device(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device

    @decorators.only_one_not_none(['id', 'type', 'py_name_num'])
    @decorators.all_or_none(['type', 'device', 'number'])
    def get_component_for(self,
                          id=None,
                          type=None,
                          device=None,
                          number=None,
                          py_name_num=None,
                          **kwargs):
        """Get the component for the provided search terms

        If `py_name_num` is provided, it must be of the following form:
        "<component_type.py_name><number>" and `Client.has_db_device()` must be
        `True. `py_name_num` is then processed by
        :func:`numbered_py_name_to_type_and_num`. The groups of parameters that
        must be 'not None' together are: (`id`), (`type`, `device`, `number`),
        and (`py_name_num`).

        Args:
            id (int|str): the primary key of the component instance on the database
            type (int): the id of the component type for the component
            device (int): the id of the device for the component
            number (int): the identifying number for this component
            py_name_num (str): the pythonic name for this component, with a number suffix
        """
        api = self.get_api_instance()
        component = None
        if py_name_num is not None:
            log.debug("Getting component using py_name_num")
            # Check that we can get database
            if not self.has_db_device:
                log.error(
                    f"Cannot get component using py_name_num if the client does not have a valid database device"
                )
                return None

            # Get the component type from py_name_num
            comp_type_py_name, number = utils.numbered_py_name_to_type_and_num(
                py_name_num)
            component_type = self.get_component_type_for(
                py_name=comp_type_py_name)

            # Check if component type exists from py_name
            if component_type is None:
                log.error(
                    f"Could not find a component type with py_name {comp_type_py_name}"
                )
                return None

            # Get the component
            component = self.get_component_for(
                type=str(component_type.id),
                device=str(self.db_device.id),
                number=str(number),
            )
            return component
        elif type is not None:
            log.debug("Getting component using type, device, and number")
            search_kwargs = {
                "type": str(type),
                "device": str(device),
                "number": str(number),
            }
            search_kwargs = utils.noneless(**search_kwargs)
            component_list = api.list_components(**search_kwargs)
            component = utils.find_model_from_list(
                component_list,
                'component',
                search_kwargs,
                **kwargs,
            )
        elif id is not None:
            log.debug("Getting component from id")
            try:
                component = api.retrieve_component(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return component

    @decorators.only_one_not_none(['id', 'name', 'py_name'])
    def get_component_type_for(self,
                               id=None,
                               name=None,
                               py_name=None,
                               **kwargs):
        """Get the component type for the provided search terms

        Args:
            id (int|str): the primary key of the component_type instance on the database
            name (str): the human readable name of the component type
            py_name (str): the pythonic name of the component type
        """
        search_kwargs = {
            "name": name,
            "py_name": py_name,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            component_type_list = api.list_component_types(**search_kwargs)
            component_type = utils.find_model_from_list(
                component_type_list,
                'component_type',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                component_type = api.retrieve_component_type(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return component_type

    @decorators.at_least_one_not_none(['id', 'component_ids'])
    def get_coupling_for(self, id=None, component_ids: list = None, **kwargs):
        """Get the coupling for the provided search terms

        Args:
            id (int|str): the primary key of the coupling instance on the database
            component_ids (list): list of component ids for which the coupling should be assigned (must be an exact match)
        """
        api = self.get_api_instance()
        if id is None:
            if len(component_ids) == 0:
                raise ValueError(
                    'component_ids must be a list with at least one element')
            coupling_list = api.list_couplings(
                components=str(component_ids[0]))
            # Cannot use find_model_from_list as we need to do more validation
            coupling = None
            for coup in coupling_list:
                if coup.components == component_ids:
                    coupling = coup
                    break
        else:
            try:
                coupling = api.retrieve_coupling(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return coupling

    @decorators.at_least_one_not_none(
        ['id', 'name', 'verbose_name', 'py_name'])
    def get_property_type_for(self,
                              id=None,
                              name=None,
                              verbose_name=None,
                              py_name=None,
                              **kwargs):
        """Get the property type for the provided search terms

        Args:
            id (int|str): the primary key of the property_type instance on the database
            name (str): the human readable name of the property type
            verbose_name (str): the human readable verbose (long) name of the property type
            py_name (str): the pythonic name of the property type
        """
        search_kwargs = {
            "name": name,
            "verbose_name": verbose_name,
            "py_name": py_name,
        }
        api = self.get_api_instance()
        if id is None:
            log.debug(
                "Getting property type using either name, verbose_name, or py_name"
            )
            search_kwargs = utils.noneless(**search_kwargs)
            property_type_list = api.list_property_types(**search_kwargs)
            property_type = utils.find_model_from_list(
                property_type_list,
                'property_type',
                search_kwargs,
                **kwargs,
            )
        else:
            log.debug("Getting property type using id")
            try:
                property_type = api.retrieve_property_type(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return property_type

    @decorators.at_least_one_not_none(['id', 'name'])
    def get_unit_for(self, id=None, name=None, **kwargs):
        """Get the unit for the provided search terms

        Args:
            id (int|str): the primary key of the unit instance on the database
            name (str): the name of the unit
        """
        search_kwargs = {
            "name": name,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            unit_list = api.list_units(**search_kwargs)
            unit = utils.find_model_from_list(
                unit_list,
                'unit',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                unit = api.retrieve_unit(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return unit

    @decorators.at_least_one_not_none(['id', 'host'])
    @decorators.all_or_none(['host', 'path'])
    def get_filefolder_raw_data_for(self,
                                    id=None,
                                    host=None,
                                    path=None,
                                    **kwargs):
        """Get the File/Folder raw data for the provided search terms

        Args:
            id (int): the primary key of the filefolder_raw_data instance on the database
            host (str): the hostname of the computer/network drive with the raw data
            path (str): the absolute directory/file path to the file or folder
        """
        search_kwargs = {
            "host": host,
            "path": path,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            filefolder_raw_data_list = api.list_file_folder_raw_datas(
                **search_kwargs)
            filefolder_raw_data = utils.find_model_from_list(
                filefolder_raw_data_list,
                'filefolder_raw_data',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                filefolder_raw_data = api.retrieve_filefolder_raw_data(
                    id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return filefolder_raw_data

    @decorators.at_least_one_not_none(['id', 'host'])
    @decorators.all_or_none(['host', 'path', 'section_id', 'page_id'])
    def get_onenote_raw_data_for(self,
                                 id=None,
                                 host=None,
                                 path=None,
                                 section_id=None,
                                 page_id=None,
                                 **kwargs):
        """Get the OneNote raw data for the provided search terms

        Args:
            id (int|str): the primary key of the onenote_raw_data instance on the database
            host (str): the hostname of the computer/network drive with the raw data
            path (str): the absolute directory/file path to the onenote file
            section_id (str): the encoded OneNote section id
            page_id (str): the encoded OneNote page id
        """
        search_kwargs = {
            "host": host,
            "path": path,
            "section_id": section_id,
            "page_id": page_id,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            onenote_raw_data_list = api.list_one_note_raw_datas(
                **search_kwargs)
            onenote_raw_data = utils.find_model_from_list(
                onenote_raw_data_list,
                'onenote_raw_data',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                onenote_raw_data = api.retrieve_onenote_raw_data(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return onenote_raw_data

    def get_experiment_for(self, id, **kwargs):
        """Get the experiment for the provided search terms

        Args:
            id (int|str): the primary key of the experiment instance on the database
        """
        if id is None:
            raise ValueError("id cannot be None")
        api = self.get_api_instance()
        try:
            experiment = api.retrieve_experiment(id=str(id))
        except device_db_client.exceptions.NotFoundException:
            return None
        return experiment

    @decorators.only_one_not_none(['component', 'coupling'])
    @decorators.at_least_one_not_none(['property_type'])
    def __validate_get_property_value_arguments(self,
                                                component=None,
                                                coupling=None,
                                                property_type=None):
        """Internal helper function to determine if the input arguments to `Client.get_property_value_for` are valid.

        Only one of `component` and `coupling` can be None, the other must be defined. `property_type` is required.

        Args:
            component (int, optional): the component id for the property value, or None
            coupling (int, optional): the coupling id for the property value, or None
            property_type (int): the property type id for the property value
            
        Raises:
            ValueError: if any of the inputs are invalid
        """
        pass

    @decorators.only_one_not_none(['id', 'property_type'])
    def get_property_value_for(self,
                               id=None,
                               component=None,
                               coupling=None,
                               property_type=None,
                               **kwargs):
        """Get the property_value for the provided search terms
        
        If `id` is not provided, `component` or `coupling`, and `property_type`
        must be provided. If these three parameters are given, it is assumed
        that only _accepted_ property values are requested.
        
        Args:
            id (int|str, optional): the primary key of the property_value instance on the database
            component (int|str, optional): the primary key of the component for this property value
            coupling (int|str, optional): the primary key of the coupling for this property value
            property_type (int|str, optional): the primary key of the property type for this property value
        """
        # Validate input arguments for `component`, `coupling`, and `property_type`.
        if id is None:
            self.__validate_get_property_value_arguments(
                component=component,
                coupling=coupling,
                property_type=property_type)
        api = self.get_api_instance()
        if id is None:
            search_kwargs = {
                "type": str(property_type),
                "is_accepted_value": 'True',
            }
            if component is not None:
                search_kwargs['component'] = str(component)
            if coupling is not None:
                search_kwargs['coupling'] = str(coupling)
            print(search_kwargs)
            property_values = api.list_property_values(**search_kwargs)
            print(property_values)
            property_value = utils.find_model_from_list(
                property_values,
                'PropertyValue',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                property_value = api.retrieve_property_value(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return property_value

    @decorators.only_one_not_none(['component', 'coupling'])
    def get_all_property_values_for(self,
                                    property_type,
                                    component=None,
                                    coupling=None,
                                    **kwargs):
        """Get all property values for the provided search terms

        One of `component` and `coupling` must always be None and the other an
        id.

        Args:
            component (int|str, optional): the primary key of the component for this property value
            coupling (int|str, optional): the primary key of the coupling for this property value
            property_type (int|str, optional): the primary key of the property type for this property value
        """
        # Validate input arguments for `component`, `coupling`, and `property_type`.
        if id is None:
            self.__validate_get_property_value_arguments(
                component=component,
                coupling=coupling,
                property_type=property_type)
        api = self.get_api_instance()
        search_kwargs = {
            "type": str(property_type),
        }
        if component is not None:
            search_kwargs['component'] = str(component)
        if coupling is not None:
            search_kwargs['coupling'] = str(coupling)
        property_values = api.list_property_values(**search_kwargs)
        return property_values

    def create_property_value(
            self, property_value: model.property_value.PropertyValue):
        """Creates a property value, and returns its new instance (with an id)

        Args:
            property_value (model.property_value.PropertyValue): the PropertyValue instance to create

        Returns:
            PropertyValue: the new property value that was created in the database
        """
        return self.get_api_instance().create_property_value(
            property_value=property_value)

    def __get_property_value_from_param_args_for_associated_component(
            self, qubit, property_type, associated_component_type):
        # Nothing to do if there are no associated components
        if len(qubit.associated_components) == 0:
            return None

        assoc_comp = None
        for assoc_comp_id in qubit.associated_components:
            assoc_comp_i = self.get_component_for(id=assoc_comp_id)
            if assoc_comp_i is None:
                log.error(
                    f"Could not find an associated component with id {assoc_comp_id}, but it should exist on the database as it has a foreign key constraint..."
                )
            else:
                if assoc_comp_i.type == associated_component_type.id:
                    log.debug(
                        f"Found an associated component with the hint type:")
                    log.debug(f"\tassoc_comp: {assoc_comp_i}")
                    log.debug(f"\ttype: {associated_component_type}")
                    assoc_comp = assoc_comp_i
                    break

        # Handle no valid associated component found
        if assoc_comp is None:
            log.debug(
                f"Could not find an associated component of type {associated_component_type} for qubit {qubit}"
            )
            return None

        # A valid associated component was found
        return self.get_property_value_for(component=assoc_comp.id,
                                           property_type=property_type.id)

    def get_property_value_from_param_args(
            self,
            qubit_py_name_num,
            property_type_py_name,
            associated_component_type_hint=None):
        """Finds an accepted property value, from a py_name, for a qubit or an associated component
        
        This is a utilities function to easily interface with the settings functionality for automated calibration routines.
        
        Example:
            .. code-block:: python
                # Get the ge_pi_half_amp for qubit 1
                property_value = get_property_value_from_param_args(
                    qubit_py_name_num = 'qb1',
                    property_type_py_name = 'ge_pi_half_amp',
                )
                
                # Get the ro_res_freq for the ro_res associated with qubit 1
                property_value = get_property_value_from_param_args(
                    qubit_py_name_num = 'qb1',
                    property_type_py_name = 'ro_res_freq',
                    associated_component_type_hint = 'ro_res',
                )
                
                # Get the raw floating value for the property value instance
                value = property_value.value

        Returns:
            PropertyValue: the property value instance found, None if not found
        """
        # Get the qubit
        qubit = self.get_component_for(py_name_num=qubit_py_name_num)
        if qubit is None:
            raise ValueError(f"Could not find qubit {qubit_py_name_num}")

        # Get the property type
        property_type = self.get_property_type_for(
            py_name=property_type_py_name)
        if property_type is None:
            log.error(
                f"Could not find a property type with py_name {property_type_py_name}. Make sure it's added to the database."
            )
            return None

        # If associated_component_type_hint is not None, we can check for an associated component from the qubit
        if associated_component_type_hint is not None:
            assoc_comp_type = self.get_component_type_for(
                py_name=associated_component_type_hint)
            if assoc_comp_type is None:
                log.warning(
                    f"Associated component type hint {associated_component_type_hint} does not identify a valid component type on the database"
                )
            else:
                property_value = self.__get_property_value_from_param_args_for_associated_component(
                    qubit=qubit,
                    property_type=property_type,
                    associated_component_type=assoc_comp_type)
                if property_value is not None:
                    log.debug(
                        f"Found a property value for the associated component")
                    return property_value
                else:
                    log.debug(
                        f"Could not find a property value for an associated component of qubit, will try on the qubit itself"
                    )

        # Try find a property value for the type on the qubit
        property_value = self.get_property_value_for(
            component=qubit.id, property_type=property_type.id)
        if property_value is None:
            log.debug(
                f"Could not find a property value of type {property_type_py_name} for qubit {qubit}"
            )
            return None
        else:
            log.debug(
                f"Found a property value of type {property_type_py_name} on qubit {qubit}"
            )
            return property_value

    def __get_all_property_value_from_param_args_for_associated_component(
            self, qubit, property_type, associated_component_type):
        # Nothing to do if there are no associated components
        if len(qubit.associated_components) == 0:
            return None

        assoc_comp = None
        for assoc_comp_id in qubit.associated_components:
            assoc_comp_i = self.get_component_for(id=assoc_comp_id)
            if assoc_comp_i is None:
                log.error(
                    f"Could not find an associated component with id {assoc_comp_id}, but it should exist on the database as it has a foreign key constraint..."
                )
            else:
                if assoc_comp_i.type == associated_component_type.id:
                    log.debug(
                        f"Found an associated component with the hint type:")
                    log.debug(f"\tassoc_comp: {assoc_comp_i}")
                    log.debug(f"\ttype: {associated_component_type}")
                    assoc_comp = assoc_comp_i
                    break

        # Handle no valid associated component found
        if assoc_comp is None:
            log.debug(
                f"Could not find an associated component of type {associated_component_type} for qubit {qubit}"
            )
            return None

        # A valid associated component was found
        return self.get_all_property_values_for(component=assoc_comp.id,
                                                property_type=property_type.id)

    def get_all_property_values_from_param_args(
            self,
            qubit_py_name_num,
            property_type_py_name,
            associated_component_type_hint=None):
        """Finds all accepted property values, from a py_name, for a qubit or an associated component

        This is a utilities function to easily interface with the settings functionality for automated calibration routines.
        
        Example:
            .. code-block:: python
                # Get the ge_pi_half_amp for qubit 1
                property_values = get_all_property_values_from_param_args(
                    qubit_py_name_num = 'qb1',
                    property_type_py_name = 'ge_pi_half_amp',
                )
                
                # Get the ro_res_freq for the ro_res associated with qubit 1
                property_values = get_all_property_values_from_param_args(
                    qubit_py_name_num = 'qb1',
                    property_type_py_name = 'ro_res_freq',
                    associated_component_type_hint = 'ro_res',
                )
                
                # Get the raw floating value for the property value instance
                for i,property_value in enumerate(property_values):
                    print(f"[{i}]: value: {property_value.value}")

        Returns:
            list: list of property values found, empty list if none were found
        """
        # Get the qubit
        qubit = self.get_component_for(py_name_num=qubit_py_name_num)
        if qubit is None:
            raise ValueError(f"Could not find qubit {qubit_py_name_num}")

        # Get the property type
        property_type = self.get_property_type_for(
            py_name=property_type_py_name)
        if property_type is None:
            log.error(
                f"Could not find a property type with py_name {property_type_py_name}. Make sure it's added to the database."
            )
            return None

        # If associated_component_type_hint is not None, we can check for an associated component from the qubit
        if associated_component_type_hint is not None:
            assoc_comp_type = self.get_component_type_for(
                py_name=associated_component_type_hint)
            if assoc_comp_type is None:
                log.warning(
                    f"Associated component type hint {associated_component_type_hint} does not identify a valid component type on the database"
                )
            else:
                property_value = self.__get_property_value_from_param_args_for_associated_component(
                    qubit=qubit,
                    property_type=property_type,
                    associated_component_type=assoc_comp_type)
                if property_value is not None:
                    log.debug(
                        f"Found a property value for the associated component")
                    return property_value
                else:
                    log.debug(
                        f"Could not find a property value for an associated component of qubit, will try on the qubit itself"
                    )

        # Try find property values for the type on the qubit
        property_values = self.get_all_property_values_for(
            component=qubit.id, property_type=property_type.id)
        return property_values
