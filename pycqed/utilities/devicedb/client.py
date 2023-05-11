import logging
import re
from typing import Optional
import datetime
import numpy as np

log = logging.getLogger(__name__)

import device_db_client
from device_db_client import model
from device_db_client.api import api_api
# from device_db_client.apis.tags import api_api # TODO: Needs to be used when upgrading to new version of OpenAPI generator
from pycqed.utilities.devicedb import decorators, utils

class Config:
    @decorators.all_or_none(['username', 'password', 'token'])
    def __init__(
        self,
        username=None,
        password=None,
        token=None,
        device_name=None,
        setup=None,
        host=None,
        use_test_db=False,
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
            setup (str, optional): the name of the setup one which measurements are performed. Defaults to None.
            host (str, optional): the url for the server hosting the device database. Defaults to "https://device-db.qudev.phys.ethz.ch".
            use_test_db (bool, optional): Boolean, specifying if the test database should be used or not. Defaults to False.
        """
        self.username = username
        self.password = password
        self.token = token
        self.device_name = device_name
        self.setup = setup
        self.host = host
        self.use_test_db = use_test_db


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

        # If username, password and token are provided, use those.
        # Otherwise use the configuration loader `get_configuration_obj()`
        # from device_db_client which automatically loads the credentials from
        # the corresponding file.
        if self.config.username != None and self.config.password != None and self.config.token != None:
            self.api_config = device_db_client.Configuration(
                host=self.config.host,
                username=self.config.username,
                password=self.config.password,
                api_key={'sessionAuth': self.config.token}, # Used for tracking
                    # which person/setup uses the API since for all API calls,
                    # the same username from the specific D PHYS user is used
            )
        else:
            if "get_configuration_obj" in dir(device_db_client):
                self.api_config = device_db_client.get_configuration_obj(
                    use_test_db=self.config.use_test_db,
                    host=self.config.host,
                )
            else:
                raise RuntimeError(
                    'The device_db_client library is deprecated! You '
                    'cannot connect to the device database with the old '
                    'library anymore. Please run `git pull` in the '
                    'device_db_client library, restart your Python kernel and '
                    'rerun your code. Then a file will automatically open with '
                    'instructions on how to generate tokens which are required '
                    'from now on. Go to the Device Database documentation if '
                    'any errors should occur: '
                    'https://documentation.qudev.phys.ethz.ch/websites/device_db/dev/index.html'
                )


        self.api_client = device_db_client.ApiClient(self.api_config)
        self.setup_name = self.config.setup
        self.setup = None
        self.check_setup()

        if not self.successful_connection():
            raise RuntimeError(f"Failed to connect to database")
        else:
            print("Succesfully connected to the device database")

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

    def check_setup(self):
        """Checks if a setup exists in the database for the given name and stores it in a property of the client object"""
        if self.setup_name is not None:
            setups = self.get_api_instance().list_setups(name=self.setup_name)
            # TODO
            # In the future, with the new version of the OpenAPI generator,
            # the following has to be used:
            # query_params = {
            #     'name': self.setup_name,
            # }
            # setups = self.get_api_instance().list_setups(query_params=query_params)

            if len(setups) == 0:
                raise ValueError(f"No setup found for the provided name '{self.setup_name}'.")
            elif len(setups) > 1:
                raise ValueError(
                    'More than one setup found for the provided name '
                    f'\'{self.setup_name}\'.')
            else: # Exactly one setup found as it should be the case
                self.setup = setups[0]

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

        # Construct the full device name because that should be unique
        # and we can only search for this full unique name and not the
        # device name without device design and wafer
        wafer = self.get_wafer_for(id=device.wafer)
        if wafer==None:
            raise ValueError(
                f'No wafer found for id "{device.wafer}"'
            )

        device_design = self.get_device_design_for(id=device.devicedesign)
        if device_design==None:
            raise ValueError(
                f'No device design found for id "{device.devicedesign}"'
            )

        full_device_name = f"{wafer.name}_{device_design.name}_{device.name}"

        maybe_device = self.get_device_for(name=full_device_name,
                                           log_empty_list=False)
        if maybe_device is None:
            log.info(f"Could not find a device, creating one instead")
            return self.get_api_instance().create_device(device=device)
        else:
            log.debug(f"Found an already existing device")
            return maybe_device

    def get_or_create_device_design(
        self,
        device_design: device_db_client.model.device_design.DeviceDesign,
    ):
        """Gets a device design from the DB, creates it if not found.

        This function will get a device design from the database, if it exists, based
        on the uniquely identifying fields of the class. If such an instance of
        device doesn't exist on the database, it will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            device_design (device_db_client.model.device_design.DeviceDesign): the device design to get from the db, or create

        Returns:
            device_design: the updated device_design instance from the database
        """
        utils.throw_if_not_db_model(device_design)
        maybe_device_design = self.get_device_design_for(
            name=device_design.name,
            log_empty_list=False)

        if maybe_device_design is None:
            log.info(f"Could not find a device design, creating one instead")
            return self.get_api_instance().create_device_design(
                device_design=device_design)
        else:
            log.debug(f"Found an already existing device design")
            return maybe_device_design

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

    def get_or_create_device_property_type(
        self,
        device_property_type: device_db_client.model.device_property_type.DevicePropertyType,
    ):
        """Gets a device property type from the DB, creates it if not found.

        This function will get a device property type from the database, if it exists,
        based on the uniquely identifying fields of the class. If such an
        instance of property type doesn't exist on the database, it will be
        created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            device_property_type (device_db_client.model.device_property_type.DevicePropertyType): the device property type to get from the db, or create

        Returns:
            device_property_type: the updated device property type instance from the database
        """
        utils.throw_if_not_db_model(device_property_type)
        maybe_device_property_type = self.get_device_property_type_for(
            py_name=device_property_type.py_name,
            log_empty_list=False
        )

        if maybe_device_property_type is None:
            log.info(f"Could not find a device_property_type, creating one instead")
            return self.get_api_instance().create_device_property_type(
                property_type=device_property_type)
        else:
            log.debug(f"Found an already existing device_property_type")
            return maybe_device_property_type

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

    def get_or_create_timestamp_raw_data(
        self,
        timestamp_raw_data: device_db_client.model.timestamp_raw_data.
        TimestampRawData,
    ):
        """Gets a timestamp raw data from the DB, creates it if not found.

        This function will get a timestamp raw data from the database, if it
        exists, based on the uniquely identifying fields of the class. If such
        an instance of timestamp raw data doesn't exist on the database, it
        will be created.

        This is useful when running scripts or code that may fail and thus
        partially create entries on the database. If the model instance was
        created on the database, a duplicate will not be created when the script
        is re-run.

        Args:
            timestamp_raw_data (device_db_client.model.timestamp_raw_data.TimestampRawData): the timestamp raw data to get from the db, or create

        Returns:
            timestamp_raw_data: the updated timestamp raw data instance from the database
        """
        utils.throw_if_not_db_model(timestamp_raw_data)
        maybe_timestamp_raw_data = self.get_api_instance().list_timestamp_raw_datas(
            timestamp=str(timestamp_raw_data.timestamp),
            setup=str(timestamp_raw_data.setup)
        )

        # If no instance in the database was found with that timestamp and setup
        if len(maybe_timestamp_raw_data) == 0:
            log.info(
                f"Could not find a timestamp_raw_data, creating one instead")
            return self.get_api_instance().create_timestamp_raw_data(
                timestamp_raw_data=timestamp_raw_data)
        else:
            # len(maybe_timestamp_raw_data) > 0. Since the combination of
            # timestamp and setup should be unique, we just return the
            # first element
            log.debug(f"Found an already existing timestamp_raw_data")
            return maybe_timestamp_raw_data[0]

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
                                                 devicedesign=component.devicedesign,
                                                 number=component.number,
                                                 log_empty_list=False)
        if maybe_component is None:
            log.info("Could not find a component, creating one instead")
            return self.get_api_instance().create_component(
                component=component)
        else:
            log.debug('Found an already existing component for DeviceDesign '
                f'id {component.devicedesign}, type {component.type} and number '
                '{component.number}.')
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
        """Get the device for the provided search terms.

        Args:
            id (int|str): the primary key of the device instance on the database
            name (str): the full name identifying the device in the format WAFER_DESIGN_DEVICE, e.g. "ATC164M154_S17v2_E5"
        """
        api = self.get_api_instance()

        if id is None:
            regex_matches = re.search('(.*)_(.*)_(.*)', name)
            if name.count('_') > 2:
                raise ValueError(
                    'Device name contains `_` more than two times and cannot '
                    'be uniquely decomposed into `WAFER_DESIGN_DEVICE`. So either '
                    'the wafer, design or device name contain an underscore which '
                    'is not allowed. Please look up the device id in the Device '
                    'Database admin interface and call the function with the id '
                    'instead of the name.'
                )
            elif regex_matches != None:
                wafer_name = regex_matches[1]
                device_design_name = regex_matches[2]
                device_name = regex_matches[3]

                wafer = self.get_wafer_for(name=wafer_name)
                if wafer==None:
                    raise ValueError(
                        f'No wafer found for name "{wafer}"'
                    )
                device_design = self.get_device_design_for(name=device_design_name)
                if device_design==None:
                    raise ValueError(
                        f'No device design found for name "{device_design}"'
                    )

                search_kwargs = {
                    "name": device_name,
                    "wafer": str(wafer.id),
                    "devicedesign": str(device_design.id),
                }
                search_kwargs = utils.noneless(**search_kwargs)
                device_list = api.list_devices(**search_kwargs)
                device = utils.find_model_from_list(
                    device_list,
                    'device',
                    search_kwargs,
                    **kwargs,
                )
            else:
                raise ValueError(
                    'Device name does not follow the pattern '
                    '`WAFER_DESIGN_DEVICE`. Please specify the full device name '
                    'in this format or look up the device id in the Device '
                    'Database admin interface and call the function with the id '
                    'instead of the name.'
                )
        else:
            try:
                device = api.retrieve_device(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device

    @decorators.at_least_one_not_none(['id', 'name'])
    def get_device_design_for(self, id=None, name=None, **kwargs):
        """Get the device design for the provided search terms

        Args:
            id (int|str): the primary key of the device design instance on the database
            name (str): the identifying name for the device design
        """
        search_kwargs = {
            "name": name,
        }
        api = self.get_api_instance()
        if id is None:
            search_kwargs = utils.noneless(**search_kwargs)
            device_designs_list = api.list_device_designs(**search_kwargs)
            device_design = utils.find_model_from_list(
                device_designs_list,
                'device_design',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                device_design = api.retrieve_device_design(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device_design

    @decorators.only_one_not_none(['id', 'type', 'py_name_num'])
    @decorators.all_or_none(['type', 'devicedesign', 'number'])
    def get_component_for(self,
                          id=None,
                          type=None,
                          devicedesign=None,
                          number=None,
                          py_name_num=None,
                          **kwargs):
        """Get the component for the provided search terms

        If `py_name_num` is provided, it must be of the following form:
        "<component_type.py_name><number>" and `Client.has_db_device()` must be
        `True. `py_name_num` is then processed by
        :func:`numbered_py_name_to_type_and_num`. The groups of parameters that
        must be 'not None' together are: (`id`), (`type`, `devicedesign`, `number`),
        and (`py_name_num`).

        Args:
            id (int|str): the primary key of the component instance on the database
            type (int): the id of the component type for the component
            devicedesign (int): the id of the device design for the component
            number (int): the identifying number for this component
            py_name_num (str): the pythonic name for this component, with a number suffix
        """
        api = self.get_api_instance()
        component = None
        if py_name_num is not None:
            log.debug("Getting component using py_name_num")
            # Check that we can get database
            if not self.has_db_device:
                raise SystemError(
                    'Cannot get component using py_name_num if the client '
                    'does not have a valid database device.'
                )

            # Get the component type from py_name_num
            comp_type_py_name, number = utils.numbered_py_name_to_type_and_num(
                py_name_num)
            component_type = self.get_component_type_for(
                py_name=comp_type_py_name)

            # Check if component type exists from py_name
            if component_type is None:
                raise ValueError(
                    'Could not find a component type with py_name '
                    f'{comp_type_py_name}'
                )

            # Get the component
            component = self.get_component_for(
                type=str(component_type.id),
                devicedesign=str(self.db_device.devicedesign),
                number=str(number),
            )
            return component
        elif type is not None:
            log.debug("Getting component using type, devicedesign, and number")
            if number == None:
                raise SyntaxError('When calling get_component_for() without a '
                    'py_name_num argument, a number has to be provided.')
            search_kwargs = {
                "type": str(type),
                "devicedesign": str(devicedesign),
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
        """Get the component type for the provided search terms.

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
        """Get the coupling for the provided search terms.

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
    def get_device_property_type_for(self,
                              id=None,
                              name=None,
                              verbose_name=None,
                              py_name=None,
                              **kwargs):
        """Get the device property type for the provided search terms.

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
                'Getting device property type using either name, verbose_name, '
                'or py_name.'
            )
            search_kwargs = utils.noneless(**search_kwargs)
            device_property_type_list = api.list_device_property_types(
                **search_kwargs)
            device_property_type = utils.find_model_from_list(
                device_property_type_list,
                'device_property_type',
                search_kwargs,
                **kwargs,
            )
        else:
            log.debug("Getting device property type using id")
            try:
                device_property_type = api.retrieve_device_property_type(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device_property_type

    def get_device_property_value_by_name(self, device_name:str=None, qubit_num:int=None, property_name:str=None):
        """Get the asymmetry of the specified qubit on the specified device.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            qubit_num (int): number of the qubit
            property_name (str): pyname of the property type which should be returned

        Returns:
            val: value of the corresponding property type
        """
        property_type_id = self.get_device_property_type_for(py_name=property_name).id
        qubit_type_id = self.get_component_type_for(py_name="qb").id

        device = self.get_device_for(name=device_name)
        component = self.get_component_for(type=qubit_type_id,
                          devicedesign=device.devicedesign,
                          number=qubit_num)

        search_kwargs = {
            "type": str(property_type_id),
            "device": str(device.id),
            "component": str(component.id)
        }

        try:
            search_kwargs = utils.noneless(**search_kwargs)
            device_property_values_list = self.get_api_instance().list_device_property_values(
                **search_kwargs)
        except device_db_client.exceptions.NotFoundException:
            log.debug(f"No value for the property {property_name} for qubit "
                      f"{qubit_num} on device {device_name} found.")
            return None

        if len(device_property_values_list) == 0:
            log.debug(f"No value for the property {property_name} for qubit "
                      f"{qubit_num} on device {device_name} found.")
            return None
        elif len(device_property_values_list) > 1:
            log.debug(f"Found {len(device_property_values_list)} values "
                      f"for the property {property_name} for qubit {qubit_num} "
                      f"on device {device_name}. Just return the first one.")

        return device_property_values_list[0]["value"]

    def get_asymmetry_for(self, device_name:str=None, qubit_num:int=None):
        """Get the asymmetry of the specified qubit on the specified device.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            qubit_num (int): number of the qubit

        Returns:
            d: asymmetry of the SQUID loop of the corresponding qubit
        """

        return self.get_device_property_value_by_name(
                device_name=device_name,
                qubit_num=qubit_num,
                property_name="asymmetry"
                )

    def get_device_design_property_value_by_name(self, device_name:str=None, device_design_name:str=None, qubit_num:int=None, property_name:str=None):
        """Get the value of the qubit parameter specifiec in `property_name`
        of the specified qubit on the specified device design.
        Only one of `device_name` and `device_design_name` should be passed.
        If `device_name` is passed, the associated `device_design_name` is
        automatically determined.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            device_design_name (str): name of the device design
            qubit_num (int): number of the qubit
            property_name (str): pyname of the property type which should be returned

        Returns:
            val: value of the corresponding property type
        """
        property_type_id = self.get_device_design_property_type_for(py_name=property_name).id
        qubit_type_id = self.get_component_type_for(py_name="qb").id

        if device_name != None:
            device = self.get_device_for(name=device_name)
            device_design_id = device.devicedesign
        else:
            device_design = self.get_device_design_for(name=device_design_name)
            device_design_id = device_design.id

        component = self.get_component_for(type=qubit_type_id,
                          devicedesign=device_design_id,
                          number=qubit_num)

        search_kwargs = {
            "type": str(property_type_id),
            "component": str(component.id)
        }

        try:
            search_kwargs = utils.noneless(**search_kwargs)
            device_design_property_values_list = self.get_api_instance().list_device_design_property_values(
                **search_kwargs)
        except device_db_client.exceptions.NotFoundException:
            log.debug(f"No value for the property {property_name} for qubit "
                      f"{qubit_num} on device {device_name} found.")
            return None

        if len(device_design_property_values_list) == 0:
            log.debug(f"No value for the property {property_name} for qubit "
                      f"{qubit_num} on device {device_name} found.")
            return None
        elif len(device_design_property_values_list) > 1:
            log.debug(f"Found {len(device_design_property_values_list)} values "
                      f"for the property {property_name} for qubit {qubit_num} "
                      f"on device {device_name}. Just return the first one.")

        return device_design_property_values_list[0]["value"]

    def get_Ec_for(self, device_name:str=None, device_design_name:str=None, qubit_num:int=None):
        """Get the charging energy E_c in Hz of the specified qubit on the
        specified device. Only one of `device_name` and `device_design_name`
        should be passed.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            device_design_name (str): name of the device design
            qubit_num (int): number of the qubit

        Returns:
            Ec: charging energy in Hz
        """

        return self.get_device_design_property_value_by_name(
                device_name=device_name,
                device_design_name=device_design_name,
                qubit_num=qubit_num,
                property_name="E_c"
                )

    def get_ro_freq_for(self, device_name:str=None, device_design_name:str=None, qubit_num:int=None):
        """Get the bare readout frequency in Hz of the specified qubit on the
        specified device. Only one of `device_name` and `device_design_name`
        should be passed.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            device_design_name (str): name of the device design
            qubit_num (int): number of the qubit

        Returns:
            ro_freq: bare readout frequency in Hz
        """

        return self.get_device_design_property_value_by_name(
                device_name=device_name,
                device_design_name=device_design_name,
                qubit_num=qubit_num,
                property_name="fr"
                )

    def get_Ej_max_for(self, device_name:str=None, qubit_num:int=None):
        """Get the total Josephson energy Ej_max in Hz of the specified qubit on
        the specified device.
        It is calculated as Ej_max = c / R_N with c some conversion factor which
        is stored in the database and R_N the normal state resistance which is
        also stored in the database.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            qubit_num (int): number of the qubit

        Returns:
            Ej_max: total Josephson energy in Hz
        """
        device = self.get_device_for(name=device_name)

        nsr = self.get_device_property_value_by_name(
                device_name=device_name,
                qubit_num=qubit_num,
                property_name="nsr"
                )

        return device.E_J_conversion_factor * 1e6 / nsr # E_J_conversion_factor
            # is in GHz/kOhm and nsr is in Ohm, so multiply by 10^6 to get the
            # result in Hz

    def get_ham_fit_dict_for(self, device_name:str=None, qubit_num:int=None):
        """Get a dictionary with the charging energy, the total Josephson energy
        and the asymmetry of the specified qubit on the specified device. This
        dict can directly be passed to the `qb.fit_ge_freq_from_dc_offset()`
        function. E_c and Ej_max are returned in Hz such that the dict can
        directly be passed to `Qubit_freq_to_dac_res` in
        `pycqed.analysis.fitting_models`.

        Args:
            device_name (str): name of the device in the format WAFER_DESIGN_NUMBER
            qubit_num (int): number of the qubit

        Returns:
            ham_fit_dict(dict): dict of E_c in Hz, Ej_max in Hz and asymmetry
        """
        return dict(E_c=self.get_Ec_for(
                        device_name = device_name,
                        qubit_num = qubit_num
                    ),
                    Ej_max=self.get_Ej_max_for(
                        device_name = device_name,
                        qubit_num = qubit_num
                    ),
                    asymmetry=self.get_asymmetry_for(
                        device_name = device_name,
                        qubit_num = qubit_num
                    )
                )

    @decorators.at_least_one_not_none(
        ['id', 'name', 'verbose_name', 'py_name'])
    def get_device_design_property_type_for(self,
                              id=None,
                              name=None,
                              verbose_name=None,
                              py_name=None,
                              **kwargs):
        """Get the device design property type for the provided search terms.

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
                'Getting device design property type using either name, '
                'verbose_name, or py_name.'
            )
            search_kwargs = utils.noneless(**search_kwargs)
            device_design_property_type_list = api.list_device_design_property_types(**search_kwargs)
            device_design_property_type = utils.find_model_from_list(
                device_design_property_type_list,
                'device_design_property_type',
                search_kwargs,
                **kwargs,
            )
        else:
            log.debug("Getting device design property type using id")
            try:
                device_design_property_type = api.retrieve_device_design_property_type(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device_design_property_type

    @decorators.at_least_one_not_none(['id', 'name'])
    def get_unit_for(self, id=None, name=None, **kwargs):
        """Get the unit for the provided search terms.

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
        """Get the File/Folder raw data for the provided search terms.

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
        """Get the OneNote raw data for the provided search terms.

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
        """Get the experiment for the provided search terms.

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
    def __validate_get_device_property_value_arguments(self,
                                                component=None,
                                                coupling=None,
                                                device_property_type=None):
        """Internal helper function to determine if the input arguments to
        `Client.get_device_property_value_for` are valid.

        Only one of `component` and `coupling` can be None, the other must be
        defined. `device_property_type` is required.

        Args:
            component (int, optional): the component id for the device property value, or None
            coupling (int, optional): the coupling id for the device property value, or None
            device_property_type (int): the device property type id for the device property value

        Raises:
            ValueError: if any of the inputs are invalid
        """
        pass

    @decorators.only_one_not_none(['id', 'device_property_type'])
    def get_device_property_value_for(self,
                               id=None,
                               component=None,
                               coupling=None,
                               device_property_type=None,
                               **kwargs):
        """Get the device_property_value for the provided search terms

        If `id` is not provided, `component` or `coupling`, and `device_property_type`
        must be provided. If these three parameters are given, it is assumed
        that only _accepted_ property values are requested.

        Args:
            id (int|str, optional): the primary key of the device_property_value instance on the database
            component (int|str, optional): the primary key of the component for this device property value
            coupling (int|str, optional): the primary key of the coupling for this device property value
            device_property_type (int|str, optional): the primary key of the device property type for this device property value
        """
        # Validate input arguments for `component`, `coupling`, and `device_property_type`.
        if id is None:
            self.__validate_get_device_property_value_arguments(
                component=component,
                coupling=coupling,
                device_property_type=device_property_type)
        api = self.get_api_instance()
        if id is None:
            search_kwargs = {
                "type": str(device_property_type),
                "is_accepted_value": 'True',
            }
            if component is not None:
                search_kwargs['component'] = str(component)
            if coupling is not None:
                search_kwargs['coupling'] = str(coupling)
            print(search_kwargs)
            device_property_values = api.list_device_property_values(**search_kwargs)
            print(device_property_values)
            device_property_value = utils.find_model_from_list(
                device_property_values,
                'DevicePropertyValue',
                search_kwargs,
                **kwargs,
            )
        else:
            try:
                device_property_value = api.retrieve_device_property_value(id=str(id))
            except device_db_client.exceptions.NotFoundException:
                return None
        return device_property_value

    @decorators.only_one_not_none(['component', 'coupling'])
    def get_all_device_property_values_for(self,
                                    device_property_type,
                                    component=None,
                                    coupling=None,
                                    **kwargs):
        """Get all device property values for the provided search terms.

        One of `component` and `coupling` must always be None and the other an
        id.

        Args:
            component (int|str, optional): the primary key of the component for this device property value
            coupling (int|str, optional): the primary key of the coupling for this device property value
            device_property_type (int|str, optional): the primary key of the property type for this device property value
        """
        # Validate input arguments for `component`, `coupling`, and `device_property_type`.
        if id is None:
            self.__validate_get_device_property_value_arguments(
                component=component,
                coupling=coupling,
                device_property_type=device_property_type)
        api = self.get_api_instance()
        search_kwargs = {
            "type": str(device_property_type),
        }
        if component is not None:
            search_kwargs['component'] = str(component)
        if coupling is not None:
            search_kwargs['coupling'] = str(coupling)
        device_property_values = api.list_device_property_values(**search_kwargs)
        return device_property_values

    def create_device_property_value(
            self, device_property_value: model.device_property_value.DevicePropertyValue):
        """Creates a device property value, and returns its new instance (with an id).

        Args:
            device_property_value (model.device_property_value.DevicePropertyValue): the DevicePropertyValue instance to create

        Returns:
            DevicePropertyValue: the new device property value that was created in the database
        """
        return self.get_api_instance().create_device_property_value(
            device_property_value=device_property_value)

    def __get_device_property_value_from_param_args_for_associated_component(
            self, qubit, device_property_type, associated_component_type):
        # Nothing to do if there are no associated components
        if len(qubit.associated_components) == 0:
            return None

        assoc_comp = None
        for assoc_comp_id in qubit.associated_components:
            assoc_comp_i = self.get_component_for(id=assoc_comp_id)
            if assoc_comp_i is None:
                log.error(
                    'Could not find an associated component with id '
                    f'{assoc_comp_id}, but it should exist on the database as it '
                    'has a foreign key constraint...'
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
                'Could not find an associated component of type '
                f'{associated_component_type} for qubit {qubit}.'
            )
            return None

        # A valid associated component was found
        return self.get_device_property_value_for(component=assoc_comp.id,
                                           device_property_type=device_property_type.id)

    def get_device_property_value_from_param_args(
            self,
            qubit_py_name_num,
            device_property_type_py_name,
            associated_component_type_hint=None):
        """Finds an accepted device property value, from a py_name, for a qubit
        or an associated component

        This is a utilities function to easily interface with the settings
        functionality for automated calibration routines.

        Example:
            .. code-block:: python
                # Get the ge_pi_half_amp for qubit 1
                device_property_value = get_device_property_value_from_param_args(
                    qubit_py_name_num = 'qb1',
                    device_property_type_py_name = 'ge_pi_half_amp',
                )

                # Get the ro_res_freq for the ro_res associated with qubit 1
                device_property_value = get_device_property_value_from_param_args(
                    qubit_py_name_num = 'qb1',
                    device_property_type_py_name = 'ro_res_freq',
                    associated_component_type_hint = 'ro_res',
                )

                # Get the raw floating value for the device property value instance
                value = device_property_value.value

        Returns:
            DevicePropertyValue: the device property value instance found, None if not found
        """
        # Get the qubit
        qubit = self.get_component_for(py_name_num=qubit_py_name_num)
        if qubit is None:
            log.warning(f"Could not find qubit {qubit_py_name_num}")
            return None

        # Get the device property type
        device_property_type = self.get_device_property_type_for(
            py_name=device_property_type_py_name)
        if device_property_type is None:
            log.warning(
                'Could not find a device property type with py_name '
                f'{device_property_type_py_name}. Make sure it\'s added to the '
                'database.'
            )
            return None

        # If associated_component_type_hint is not None, we can check for an
        # associated component from the qubit
        if associated_component_type_hint is not None:
            assoc_comp_type = self.get_component_type_for(
                py_name=associated_component_type_hint)
            if assoc_comp_type is None:
                log.warning(
                    'Associated component type hint '
                    f'{associated_component_type_hint} does not identify a valid '
                    'component type on the database.'
                )
            else:
                device_property_value = self.__get_device_property_value_from_param_args_for_associated_component(
                    qubit=qubit,
                    device_property_type=device_property_type,
                    associated_component_type=assoc_comp_type)
                if device_property_value is not None:
                    log.debug(
                        "Found a device property value for the associated component")
                    return device_property_value
                else:
                    log.debug(
                        'Could not find a device property value for an '
                        'associated component of qubit, will try on the qubit '
                        'itself.'
                    )

        # Try find a device property value for the type on the qubit
        device_property_value = self.get_device_property_value_for(
            component=qubit.id, device_property_type=device_property_type.id)
        if device_property_value is None:
            log.debug(
                'Could not find a device property value of type '
                f'{device_property_type_py_name} for qubit {qubit}.'
            )
            return None
        else:
            log.debug(
                'Found a device property value of type '
                f'{device_property_type_py_name} on qubit {qubit}.'
            )
            return device_property_value

    def __get_all_device_property_value_from_param_args_for_associated_component(
            self, qubit, device_property_type, associated_component_type):
        # Nothing to do if there are no associated components
        if len(qubit.associated_components) == 0:
            return None

        assoc_comp = None
        for assoc_comp_id in qubit.associated_components:
            assoc_comp_i = self.get_component_for(id=assoc_comp_id)
            if assoc_comp_i is None:
                log.error(
                    'Could not find an associated component with id '
                    f'{assoc_comp_id}, but it should exist on the database as it '
                    'has a foreign key constraint...'
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
                'Could not find an associated component of type '
                f'{associated_component_type} for qubit {qubit}'
            )
            return None

        # A valid associated component was found
        return self.get_all_device_property_values_for(
            component=assoc_comp.id,
            device_property_type=device_property_type.id
        )

    def get_all_device_property_values_from_param_args(
            self,
            qubit_py_name_num,
            device_property_type_py_name,
            associated_component_type_hint=None):
        """Finds all accepted device property values, from a py_name, for a
        qubit or an associated component.

        This is a utilities function to easily interface with the settings
        functionality for automated calibration routines.

        Example:
            .. code-block:: python
                # Get the ge_pi_half_amp for qubit 1
                device_property_values = get_all_device_property_values_from_param_args(
                    qubit_py_name_num = 'qb1',
                    device_property_type_py_name = 'ge_pi_half_amp',
                )

                # Get the ro_res_freq for the ro_res associated with qubit 1
                device_property_values = get_all_device_property_values_from_param_args(
                    qubit_py_name_num = 'qb1',
                    device_property_type_py_name = 'ro_res_freq',
                    associated_component_type_hint = 'ro_res',
                )

                # Get the raw floating value for the property value instance
                for i,device_property_value in enumerate(device_property_values):
                    print(f"[{i}]: value: {device_property_value.value}")

        Returns:
            list: list of device property values found, empty list if none were found
        """
        # Get the qubit
        qubit = self.get_component_for(py_name_num=qubit_py_name_num)
        if qubit is None:
            raise ValueError(f"Could not find qubit {qubit_py_name_num}")

        # Get the property type
        device_property_type = self.get_device_property_type_for(
            py_name=device_property_type_py_name)
        if device_property_type is None:
            log.error(
                'Could not find a property type with py_name'
                f'{device_property_type_py_name}. Make sure it\'s added to the '
                'database.'
            )
            return None

        # If associated_component_type_hint is not None, we can check for an associated component from the qubit
        if associated_component_type_hint is not None:
            assoc_comp_type = self.get_component_type_for(
                py_name=associated_component_type_hint)
            if assoc_comp_type is None:
                log.warning(
                    'Associated component type hint '
                    f'{associated_component_type_hint} does not identify a valid '
                    'component type on the database'
                )
            else:
                device_property_value = self.__get_device_property_value_from_param_args_for_associated_component(
                    qubit=qubit,
                    device_property_type=device_property_type,
                    associated_component_type=assoc_comp_type)
                if device_property_value is not None:
                    log.debug(
                        'Found a device property value for the associated '
                        'component')
                    return device_property_value
                else:
                    log.debug(
                        'Could not find a device property value for an '
                        'associated component of qubit, will try on the qubit '
                        'itself.'
                    )

        # Try find property values for the type on the qubit
        device_property_values = self.get_all_device_property_values_for(
            component=qubit.id, device_property_type=device_property_type.id)
        return device_property_values

    @decorators.only_one_not_none(['id', 'name'])
    def get_device_design_connectivity_graph(self, name=None, id=None, return_strings=False, return_tuples=True):
        """Returns the connectivity graph of a device design.

        For example for the S17v2 design, it will return
        [(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (4, 9), (6, 5),
        (6, 11), (8, 3), (8, 7), (8, 9), (8, 13), (10, 5), (10, 9),
        (10, 11), (10, 15), (12, 7), (12, 13), (14, 9), (14, 13),
        (14, 15), (14, 16), (17, 15), (17, 16)]

        Args:
            id (int|str): id of the device design
            name (str): name of the device design
            return_strings (bool):  True: return strings, e.g. 'qb7',
                                    False: return integers, e.g. 7
            return_tuples (bool):   True: return tuples, e.g. [(1, 2), ...]
                                    False: return lists, e.g. [[1, 2], ...]

        Returns:
            connectivity_graph (list): list of tuples or lists, representing the
                connectivity graph of the device design
        """
        api = self.get_api_instance()

        # if id == None, a name has to be provided instead and we can find
        # the id over the name
        if id==None:
            try:
                device_design = self.get_device_design_for(name=name)
                id = device_design.id
            except Exception as e:
                raise SystemError(
                    'Could not find the device design related to the name '
                    f'{name}. Exception: {e}'
                )

        # Get the id of the component type for qubit qubit coupling resonators
        try:
            component_type = self.get_component_type_for(py_name="qb_qb_coupl_res")
        except Exception as e:
            raise SystemError(
                'Could not find the qubit-qubit coupling resonator component '
                f'type in the database. Exception: {e}'
            )
            return False

        # Get the list of all the qubit qubit coupling resonators on that design
        try:
            component_list = api.list_components(
                type=str(component_type.id),
                devicedesign=str(id)
            )
        except Exception as e:
            raise SystemError(
                'Could not get the component list of qubit-qubit coupling '
                f'resonators. Exception: {e}'
            )
            return False

        connectivity_graph = [] # Array which stores qubit connections

        # Iterate over every coupling resonator, find the two qubits connected
        # to the resonator and add them to the connectivity graph
        for component in component_list:
            coupling_list = api.list_couplings(components=str(component.id))
            if len(coupling_list) > 2:
                raise SystemError('Found more than two qubits connected to a '
                    'qubit-qubit coupling resonator. How can that happen?')
            elif len(coupling_list) < 2:
                raise SystemError('Could not find two qubits that are connected '
                    'to a qubit-qubit coupling resonator. How can that happen?')
            else:
                qbs_list = []
                # There are two couplings associated to one coupling resonator:
                # One coupling to the one qubit and one coupling to the other qubit
                for coupling in coupling_list:
                    if len(coupling["components"]) != 2: # Consistency check
                        raise SystemError('Coupling between more or less than '
                            'two qubits is not supported at the moment.')

                    # Check what of the two elements is the coupling resonator
                    # and what is the qubit
                    el1 = coupling["components"][0]
                    el2 = coupling["components"][1]
                    if el1 == component.id:
                        qb = el2
                    elif el2 == component.id:
                        qb = el1

                    # Instead of the component id of the qubit in the database,
                    # find the number of the qubit on the device design
                    qb_id_on_design = api.retrieve_component(id=str(qb))
                    qbs_list.append(qb_id_on_design.number)

                # Append the two qubits as tuples. If tuples are not required,
                # one can also just use connectivity_graph.append(qbs_list)
                if return_strings:
                    qb1 = f"qb{qbs_list[0]}"
                    qb2 = f"qb{qbs_list[1]}"
                else:
                    qb1 = qbs_list[0]
                    qb2 = qbs_list[1]

                if return_tuples:
                    connectivity_graph.append((qb1, qb2))
                else:
                    connectivity_graph.append([qb1, qb2])

        return connectivity_graph

    def upload_normal_state_resistances(self,
        path=None,
        device_name=None,
        device_id=None,
        comments="",
        values_in_kOhm=True,
        set_accepted=True
        ):
        """Uploads normal state resistance measurements that are stored in a
        file to the database.

        Args:
            path (str): path of the file in which all measurements are stored
            device_name (str): name of the device
            device_id (int|str): id of the device
            comments (str): comments regarding the measurements. This will be stored in the `Experiment` object that is created.
            values_in_kOhm (bool): if True, the values in the file are processed in kOhm, if False, in Ohm.
            set_accepted (bool): if True (default), the stored normal state resistances are directly marked as accepted

        Returns:
            device_property_value_array (list): array of created device property value objects
        """

        # Check if correct arguments were provided
        if path == None:
            raise ValueError(
                'You have to provide a path to the text file in which the '
                'normal state resistances are stored.'
            )
        elif device_name == None and device_id == None:
            raise ValueError(
                'You have to provide either the device name or the device id '
                'to which the normal state resistances are related.'
            )
        elif device_name != None and device_id != None:
            raise ValueError(
                'You can only specify the device name or the device id but not '
                'both together (both should be unique by their own already).'
            )

        if device_name != None: # If device name was provided, get id
            device = self.get_device_for(name=device_name)
            device_id = device.id
            device_design = device.devicedesign
        else: # In that the device id was provided. Check if the id is valid and what the related device design is
            try:
                device = self.get_device_for(id=device_id)
                device_name = f"{self.get_wafer_for(id=device.wafer).name}_"\
                    f"{self.get_device_design_for(id=device.devicedesign).name}_"\
                    f"{device.name}" # Used for creating the experiment object
                device_design = device.devicedesign
            except Exception as e:
                raise ValueError('Could not find a device for the provided id '
                    '"{device_id}". Exception: %s\n' % e)

        # Find the right device property type for the normal state resistance
        try:
            nsr_property_type = self.get_device_property_type_for(py_name="nsr")
        except Exception as e:
            raise SystemError('Could not find the right device property type '
            'for normal state resistances. Exception: %s\n' % e)

        # Find the right component type for a qubit
        try:
            qubit_component_type = self.get_component_type_for(py_name="qb")
        except Exception as e:
            raise SystemError('Could not find the right qubit component type: '
                f'%s\n' % e)

        api = self.get_api_instance()

        # Create the experiment object
        experiment = device_db_client.model.experiment.Experiment(
            datetime_taken=datetime.datetime.now(),
            datetime_uploaded=datetime.datetime.now(),
            type="Normal state resistance measurements on device "+device_name,
            comments=comments,
        )
        try:
            experiment_uploaded = api.create_experiment(experiment=experiment)
        except device_db_client.ApiException as e:
            raise SystemError('Exception when calling ApiApi->'
                'create_experiment: %s\n' % e)

        # Read file
        try:
            file = open(path, "r")
        except FileNotFoundError:
            raise ValueError(
                f'The file `{path}` could not be opnened. Please provide a '
                'valid path to the file.'
            )

        device_property_value_array = []
        for line in file:
            if line.strip() != "": # If the line is empty (e.g. if there is an empty line at the end), skip it
                regex_matches = re.search('(.*): (.*)', line)
                qb_num = int(regex_matches[1])
                resistance = float(regex_matches[2])
                if values_in_kOhm:
                    resistance *= 1000

                # Find the right qubit component
                try:
                    qubit_component = self.get_component_for(
                        type=qubit_component_type.id,
                        devicedesign=device_design,
                        number=qb_num,
                    )
                except Exception as e:
                    raise SystemError('Could not find the right qubit '
                        'component for number {qb_num}: %s\n' % e)

                # Create the device property value object
                device_property_value = device_db_client.model.device_property_value.DevicePropertyValue(
                    type=nsr_property_type.id,
                    experiment=experiment_uploaded.id,
                    value=resistance,
                    device=device_id,
                    component=qubit_component.id,
                    raw_data=[],
                )

                try: # Try to upload the measurement value
                    created_device_property_value = api.create_device_property_value(
                        device_property_value=device_property_value
                    )
                    device_property_value_array.append(device_property_value)
                except device_db_client.ApiException as e:
                    raise SystemError('Exception when calling ApiApi->'
                        'create_device_property_value: {e}' % e)

                if set_accepted: # Set the created device property value to accepted
                    try:
                        api.set_accepted_device_property_value(
                            id=str(created_device_property_value.id))
                    except Exception as e:
                        raise RuntimeError(
                            'Failed to set property value as accepted, with '
                            f'error {e}'
                        )

        return device_property_value_array
