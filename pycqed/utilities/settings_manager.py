# Importing annotations is required because Instrument class uses own class
# definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/typed-python-using-the-classes-own-type-inside-class-definition
# for more details.
from __future__ import annotations

from pycqed.analysis_v3 import helper_functions as hlp_mod
import logging
import h5py
import numpy as np
import time
from pycqed.measurement.hdf5_data import DateTimeGenerator \
    as DateTimeGenerator
import os
import pycqed.gui.dict_viewer as dict_viewer
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)

# constant file extensions, ordered in most desired file type
file_extensions = {
    'msgpack': '.msg', 'msgpack_comp': '.msgc', 'pickle': '.pickle',
    'pickle_comp': '.picklec', 'hdf5': '.hdf5'}


class DelegateAttributes:
    """
    Copied from QCodes.utils.helpers (version 0.35.0)
    Mixin class to create attributes of this object by
    delegating them to one or more dictionaries and/or objects.

    Also fixes ``__dir__`` so the delegated attributes will show up
    in ``dir()`` and ``autocomplete``.

    Attribute resolution order:
        1. Real attributes of this object.
        2. Keys of each dictionary in ``delegate_attr_dicts`` (in order).
        3. Attributes of each object in ``delegate_attr_objects`` (in order).
    """
    delegate_attr_dicts = []
    """
    A list of names (strings) of dictionaries
    which are (or will be) attributes of ``self``, whose keys should
    be treated as attributes of ``self``.
    """
    delegate_attr_objects = []
    """
    A list of names (strings) of objects
    which are (or will be) attributes of ``self``, whose attributes
    should be passed through to ``self``.
    """
    omit_delegate_attrs = []
    """
    A list of attribute names (strings)
    to *not* delegate to any other dictionary or object.
    """

    def __getattr__(self, key: str):
        if key in self.omit_delegate_attrs:
            raise AttributeError("'{}' does not delegate attribute {}".format(
                self.__class__.__name__, key))

        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    "dict '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                d = getattr(self, name, None)
                if d is not None:
                    return d[key]
            except KeyError:
                pass

        for name in self.delegate_attr_objects:
            if key == name:
                raise AttributeError(
                    "object '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    return getattr(obj, key)
            except AttributeError:
                pass

        raise AttributeError(
            "'{}' object and its delegates have no attribute '{}'".format(
                self.__class__.__name__, key))


class IndexableDict(OrderedDict):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as keyerror:
            if isinstance(item, int):
                try:
                    return super().__getitem__(list(self.keys())[item])
                except IndexError as e:
                    logger.error(f"Integer seems to be out of range of list"
                                 f" '{list(self.keys())}'")
                    raise e
            else:
                raise keyerror


class Parameter(DelegateAttributes):
    """
    Parameter class which mocks Parameter class of QCodes
    """

    delegate_attr_dicts = ['_vals']

    def __init__(self,
                 name: str,
                 value=None,
                 gettable=True,
                 settable=False) -> None:
        """
        Parameter has a (unique) name and a respective value
        Args:
            name: Name of parameter (str)
            value: Value of parameter (any type)
        """
        self.name = name
        self._value = value
        self.gettable = gettable
        self.settable = settable

    def snapshot(self, reduced=False) -> dict[any, any]:
        """
        Creates a dictionary out of value attribute
        Args:
            reduced (bool): returns a reduced (or light) snapshot where the
                parameter value is (if possible) only the value instead of the
                entire dictionary with metadata. Default is False.
        Returns: dictionary
        """
        if reduced:
            if isinstance(self._value, dict):
                snap = self._value.get('value', self._value)
            else:
                snap = self._value
        else:
            snap: dict[any, any] = self._value
        return snap

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            if self.gettable:
                return self.get()
            else:
                raise NotImplementedError('no get cmd found in' +
                                          f' Parameter {self.name}')
        else:
            if self.settable:
                self.set(*args, **kwargs)
                return None
            else:
                raise NotImplementedError('no set cmd found in' +
                                          f' Parameter {self.name}')

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class Instrument(DelegateAttributes):
    """
    Instrument class which mocks Instrument class of QCodes. An instrument
    object contains given parameters. Submodules are instruments themselves.
    """

    delegate_attr_dicts = ['parameters', 'submodules']

    def __init__(self, name: str):
        """
        Initialization of the Instrument object.
        Args:
            name (str): name of the respective instrument
        """
        self.name = name
        self.parameters = {}
        self.functions = {}
        self.submodules = {}
        # QCodes instruments store the respective Class name of an
        # instrument, e.g. pycqed.instrument_drivers.meta_instrument
        # .qubit_objects.QuDev_transmon.QuDev_transmon
        self.classname: str = None

    def snapshot(self, reduced=False) -> dict[any, any]:
        """
        Creates recursively a snapshot (dictionary) which has the same structure
        as the snapshot from QCodes instruments.
        Args:
            reduced (bool): returns a reduced (or light) snapshot where the
                parameter value is (if possible) only the value instead of the
                entire dictionary with metadata. Default is False.
        Returns (dict): Returns a snapshot as a dictionary with keys
            'functions', 'submodules', 'parameters', '__class__' and 'name'

        """
        param: dict[any, any] = {}
        for key, item in self.parameters.items():
            param[key] = item.snapshot(reduced=reduced)

        submod: dict[any, any] = {}
        for key, item in self.submodules.items():
            submod[key] = item.snapshot(reduced=reduced)

        snap = {
            'functions': self.functions,
            'submodules': submod,
            'parameters': param,
            '__class__': self.classname,
            'name': self.name}

        return snap

    def add_parameter(self, param: Parameter):
        """
        Adds a Parameter object to the dictionary self.parameters
        Args:
            param: Parameter object which is added to the instrument.
        """
        namestr = param.name
        if namestr in self.parameters.keys():
            raise RuntimeError(
                f'Cannot add parameter "{namestr}", because a '
                'parameter of that name is already registered to '
                'the instrument')
        self.parameters[namestr] = param

    def add_classname(self, name: str):
        self.classname = name

    def add_submodule(self, name: str, submod: Instrument):
        """
        Adds a submodule (aka channel) which is an instrument themselves.
        We make no distinction between submodules and channels (unlike QCodes).
        To force submod to be an Instrument, from __future__ import
        annotations has to be added at the beginning.
        Args:
            name (str): name of the respective submodule or channel
            submod (Instrument): Instrument object of the respective submodule.

        """
        self.submodules[name] = submod

    def update(self, instrument: Instrument):
        """
        Updates self with attributes from a given instrument object.
        Args:
            instrument (Instrument): Instrument which attributes should be added
                to self.
        """
        self.parameters.update(instrument.parameters)
        self.functions.update(instrument.functions)
        self.submodules.update(instrument.functions)
        self.classname = instrument.classname


class Station(DelegateAttributes):
    """
    Station class which mocks Station class of QCodes. Station contains all
    given instruments and parameters.
    """

    delegate_attr_dicts = ['components']

    def __init__(self, timestamp: str = None):
        """
        Initialization of the station. Each station is characterized by the
        timestamp.
        Note that inside the station all instruments are added to the components
        attribute.
        When snapshotting the station, the snapshot of the instruments can be
        found in the "instrument" keys and all other items of the components
        attribute are in the "components" key.
        Args:
            timestamp (str): For accepted formats of the timestamp see
            a_tools.verify_timestamp(str)
        """
        self.instruments: dict = {}
        self.parameters: dict = {}
        self.components: dict = {}
        self.config: dict = {}
        if timestamp is None:
            self.timestamp = ''
        else:
            # a_tools.verify_timestamp returns unified version of the timestamps
            from pycqed.analysis import analysis_toolbox as a_tools
            self.timestamp = '_'.join(a_tools.verify_timestamp(timestamp))

    def snapshot(self, reduced=False) -> dict[any, any]:
        """
        Returns a snapshot as a dictionary of the entire station based on the
        structure of QCodes.
        Instrument snapshots are saved in "instrument" key of the dictionary.
        Args:
            reduced (bool): returns a reduced (or light) snapshot where the
                parameter value is (if possible) only the value instead of the
                entire dictionary with metadata. Default is False.
        Returns (dict): snapshot of the station with keys
            'instruments', 'parameters', 'components' and 'config'
        """
        inst: dict[any, any] = {}
        components: dict[any, any] = {}
        for key, item in self.components.items():
            if isinstance(item, Instrument):
                inst[key] = item.snapshot(reduced=reduced)
            else:
                components[key] = item.snapshot(reduced=reduced)

        param: dict[any, any] = {}
        for key, item in self.parameters.items():
            param[key] = item.snapshot(reduced=reduced)

        snap = {
            'instruments': inst,
            'parameters': param,
            'components': components,
            'config': self.config
        }

        return snap

    def add_component(self, inst: Instrument):
        """
        Adds a given instrument to the station under the constriction that
        there exists no instrument with the same name.
        Args:
            inst (Instrument): Instrument object
        """
        namestr = inst.name
        if namestr in self.components.keys():
            raise RuntimeError(
                f'Cannot add component "{namestr}", because a '
                'component of that name is already registered to the station')
        self.components[namestr] = inst

    def get(self, path_to_param):
        """
        Tries to find parameter value for given string. Returns 'not found' if
        an attribute error occurs.
        Args:
            path_to_param (str): path to parameter in form
                %inst_name%.%param_name%

        Returns (str): parameter value, if parameter value is a dictionary it
            tries to get the "value" key.
        """
        param_value = 'not found'
        try:
            param = eval('self.' + path_to_param + '()')
            if isinstance(param, dict):
                param_value = param.get("value", "not found")
            else:
                param_value = param
        except AttributeError:
            param_value = 'not found'
            # logger.warning('Problem at extracting parameter from station.')
        return param_value

    def update(self, station):
        """
        Updates the station with instruments and parameters from another station
        Args:
            station(Station): Station which is included to self
        """
        self.instruments.update(station.instruments)
        self.parameters.update(station.parameters)
        self.config.update(station.config)
        for comp_name, comp_inst in station.components.items():
            if comp_name in self.components.keys():
                self.components[comp_name].update(comp_inst)
            else:
                self.components[comp_name] = comp_inst


class SettingsManager:
    """
    Class which contains different station. Stations can be added in two ways:
    1. Loaded from saved settings in a given file. Supported file types:
        hdf5, pickle (.obj), msgpack (.msg)
    2. Added from a preexisting station. Either settings_manager.Station or
        QCodes.Station (or any other snapshotable type)
    Snapshot of stations can be displayed via dict_viewer module.
    """

    def __init__(self, station=None, timestamp: str = None):
        """
        Initialization of SettingsManager instance. Can be called with a
        preexisting station (settings_manager.Station or QCodes.Station)
        Args:
            station (Station or qcodes.Station): optional, adds given station
                to the settings manager
        """
        self.stations = IndexableDict()
        if station is not None:
            self.add_station(station, timestamp)

    def add_station(self, station, timestamp: str):
        """
        Adds a given station with a timestamp as its unique name.
        Args:
            station (Station): Station object which will be added.
                Either QCodes station or Station class from this module
            timestamp (str): Timestamp to call the station.
                Is converted into unified format.

        """
        if timestamp is not None:
            from pycqed.analysis import analysis_toolbox as a_tools
            timestamp = '_'.join(a_tools.verify_timestamp(timestamp))
        else:
            raise TypeError(f'Cannot add station, because the timestamp is '
                            f'not specified.')
        if hasattr(station, 'snapshot'):
            self.stations[timestamp] = station
        else:
            raise TypeError(f'Cannot add station "{timestamp}", because the '
                            'station is not a QCode or Mock station class '
                            'or is not snapshotable.')

    def load_from_file(self, timestamp: str, folder=None, filepath=None,
                       file_id=None, param_path=None):
        """
        Loads station into the settings manager from saved files.
        Files contain dictionary of the snapshot (pickle, msgpack) or a hdf5
        representation of the snapshot.
        Args:
            timestamp (str): Supports all formats from a_tools.get_folder
            folder (str): Optional, folder of the file if distinct from
                a_tools.datadir
            file_id (str): suffix of the file
            filepath (str): filepath of the file, overwrites timestamp and
                folder
            param_path (list(str)): List of path to parameter which should be
                loaded to the station. Used to load HDF5 files faster. Set to
                None of entire file should be loaded.

        Returns: Station which was created by the saved data.
        """
        station = get_station_from_file(timestamp=timestamp, folder=folder,
                                        filepath=filepath, file_id=file_id,
                                        param_path=param_path)

        self.add_station(station, timestamp)
        return station

    def update_station(self, timestamp, param_path=None, **kwargs):
        """
        Only relevant for HDF-files. Other files will load the entire station
        to the settings manager due to small execution time.
        Updates the station specified by the timestamp with the parameters given
        in param_path. If no param_path are given, the entire station will be
        loaded from the timestamp
        Args:
            timestamp (str, int): timestamp of the station which should be
                updated. If timestamp not in settings manager, the station
                will be loaded in to the settings manager.
                If int it tries to retrieve the timestamp from the loaded
                stations.
            param_path (list): list of parameters which are loaded into a
                station. If None, all parameters are loaded into the station.
                Parameters must be of the form %inst_name%.%param_name%.
            **kwargs: See docstring of self.load_from_file.
        """
        if isinstance(timestamp, int):
            try:
                timestamp = list(self.stations.keys())[timestamp]
            except IndexError as e:
                logger.error(f"Integer seems to be out of range of list"
                             f" '{list(self.stations.keys())}'")
                raise e

        if timestamp not in self.stations.keys():
            self.load_from_file(timestamp, param_path=param_path, **kwargs)
        else:
            if param_path is None:
                self.stations.pop(timestamp)
                self.load_from_file(timestamp=timestamp)
            else:
                tmp_station = get_station_from_file(timestamp=timestamp,
                                                    param_path=param_path,
                                                    **kwargs)
                self.stations[timestamp].update(tmp_station)

    def spawn_snapshot_viewer(self, timestamp, new_process=False):
        """
        Opens a gui window to display the snapshot of the given station.0
        Args:
            timestamp (str): address the station which is already loaded onto
                the settings manager.
            new_process (bool): True if new process should be started, which
                does not block the IPython kernel. False by default because
                it takes some time to start the new process.
        Returns:

        """
        if timestamp not in self.stations.keys():
            print(f"timestamp '{timestamp}' will be loaded onto the station.")
            self.load_from_file(timestamp)
        snapshot_viewer = dict_viewer.SnapshotViewer(
            snapshot=self.stations[timestamp].snapshot(),
            timestamp=timestamp)
        snapshot_viewer.spawn_viewer(new_process=new_process)

    def _compare_dict_instances(self, dict_list, name_list):
        """
        Helper function which compares n dictionaries and returns a combined
        dictionary with the intersection of all keys which values do not
        coincide
        Args:
            dict_list (list of dict): list of dictionaries which are compared
            name_list (list of strings): list of unique names of the
                dictionaries (e.g. timestamp). Names are used as Timestamps
                 objects for unique keys for the combined dictionaries.

        Returns (dict): combined dictionary of with values which do not coincide
            between the different dictionaries

        """
        all_keys = set()
        all_diff = {}
        # set of all keys
        for dic in dict_list:
            all_keys.update(set(dic.keys()))
        for key in all_keys:
            diff = {}
            key_not_in_dicts = False
            dicts_with_key = []

            for i, dic in enumerate(dict_list):
                if key not in dic.keys():
                    # flags that at least one of the dictionaries does not have
                    # the particular key
                    key_not_in_dicts = True
                else:
                    dicts_with_key.append(i)

            # if at least one of the dictionaries does not have the key, the
            # compared dictionaries is build
            if key_not_in_dicts:
                diff[key] = \
                    {Timestamp(name_list[i]): dict_list[i][key]
                     for i in dicts_with_key}
            else:
                # if all the dictionaries have the key, the values are compared
                for i, dic in enumerate(dict_list[1:]):
                    try:
                        np.testing.assert_equal(dict_list[0][key], dic[key])
                    except AssertionError:
                        # items do not coincide
                        if all(isinstance(dic[key], dict) for dic in dict_list):
                            diff[key] = self._compare_dict_instances(
                                [dic[key] for dic in dict_list], name_list)
                            break
                        else:
                            # this occurs when not all items are dictionaries
                            diff[key] = \
                                {Timestamp(name_list[i]): dic[key]
                                 for i, dic in enumerate(dict_list)}
                            break
            all_diff.update(diff)

        return all_diff

    def _compare_station_components(self, timestamps, instruments='all',
                                    reduced_compare=False):
        """
        Helper function to compare multiple stations
        Args:
            timestamps (list of str): timestamps of the station of the settings
                manager which are compared
            instruments (str, list of str): either 'all' (default) or a list of
                instrument names to compare only a subset of instruments
            reduced_compare (bool): if True it compares only the reduced
                snapshot of the stations (i.e. only values and no metadata)

        Returns:
            dict and str of the compared dictionary and the messages of the
            missing instruments/components

        """
        all_components = set()
        all_msg = []
        all_diff = {}

        # create a set of all components of all given stations
        for tsp in timestamps:
            all_components.update(set(self.stations[tsp].components.keys()))

        # check which components are in all stations
        for component in all_components:
            if instruments != 'all' and component not in instruments:
                continue

            # marker if all stations contain component
            component_not_in_all_stations = False
            comp_dict = {}
            for tsp in timestamps:
                if component not in self.stations[tsp].components.keys():
                    all_msg.append(
                        f'\nComponent/Instrument "{component}" missing in dict '
                        f'{tsp}.\n')
                    component_not_in_all_stations = True
                else:
                    comp_dict[Timestamp(tsp)] = 'exists'
            if component_not_in_all_stations:
                # if one of the stations does not have the component, the
                # comparison of this component cannot be done
                # (You have to compare like with like)
                all_diff[component] = comp_dict
            else:
                # snapshots of the components
                if reduced_compare:
                    component_snaps = \
                        [self.stations[tsp].components[component].snapshot(
                            reduced=reduced_compare)
                            for tsp in timestamps]
                else:
                    component_snaps = \
                        [self.stations[tsp].components[component].snapshot()
                         for tsp in timestamps]

                # comparison of the snapshots
                diff = self._compare_dict_instances(component_snaps, timestamps)
                if diff != {}:
                    all_diff[component] = diff

        return all_diff, all_msg

    def compare_stations(self, timestamps, instruments='all',
                         reduced_compare=False, output='viewer',
                         new_process=False):
        """
        Compare instrument settings from n different station in the settings
        manager.
        Args:
            timestamps (list of str): timestamps of the stations which are
                compared. If station is not part of the settings_manager object
                it will be loaded into the settings_manager.
            instruments (str, list of str): either 'all' (default) or a list of
                instrument names to compare only a subset of instruments
            reduced_compare (bool): if True it compares only the reduced
                snapshot of the stations (i.e. only values and no metadata)
            output (str): One of the following output formats:
                'str': return comparison report as str
                'dict': return comparison results as a dict
                'viewer' (default): opens a gui with the compared dictionary
            new_process (bool): True if new process should be started, which
                does not block the IPython kernel. False by default because
                it takes some time to start the new process.
        """
        if timestamps == 'all':
            ts_list = list(self.stations.keys())
        elif isinstance(timestamps, list):
            ts_list = timestamps
        else:
            raise NotImplementedError(f'Timestamp "{timestamps}" is not a '
                                      f'list of timestamps or "all"')

        # checks if stations with respective timestamps are already in the
        # settings manager. otherwise, it loads the station into the sm
        for tsp in set(ts_list).difference(self.stations.keys()):
            print(f"timestamp '{tsp}' will be loaded onto the station.")
            self.load_from_file(tsp)

        # for QCode station reduced comparison is not supported
        if reduced_compare:
            for ts in ts_list:
                if not isinstance(self.stations[ts], Station):
                    logger.warning(
                        f"QCode stations do not support reduced comparison. "
                        f"reduced_compare is set to False.")
                    reduced_compare = False

        diff, msg = self._compare_station_components(
            ts_list, instruments=instruments, reduced_compare=reduced_compare)

        if output == 'str':
            return msg
        elif output == 'dict':
            return diff
        elif output == 'viewer':
            snapshot_viewer = dict_viewer.SnapshotViewer(
                snapshot=diff,
                timestamp=ts_list)
            snapshot_viewer.spawn_viewer(new_process=new_process)


class Timestamp(str):
    """
    Generic timestamp class. Used to distinguish between regular strings and
    timestamp strings. Can be extended with custom timestamp functions (e.g.
    validator of timestamps, output in a particular format, ...)
    """
    pass


class Loader:
    """
    Generic class to load instruments and parameters from a file.
    """

    def __init__(self, timestamp: str = None, filepath: str = None, **kwargs):
        """
        Initialization of generic loader. Should not be initialized but rather
        its heritages.
        Args:
            timestamp (str): timestamp of the file in the format. Supports all
                formats from a_tools.get_folder
            filepath (str): filepath of the file, overwrites timestamp and
                folder (in kwargs)
            **kwargs: folder (str): Optional, folder of the file if distinct from
                a_tools.datadir
                file_id (str): suffix of the file
                extension (str): custom extension of the file
                compression (bool): True if file is compressed with blosc2,
                    usually extracted from file extension
        """
        self.filepath = filepath
        self.timestamp = timestamp
        self.extension = kwargs.get('extension', None)

    def get_filepath(self, **kwargs):
        """
        If no explicit filepath is given, a_tools.get_folder tries to get the
        directory based on the timestamp and the directory defined by
        a_tools.datadir, a_tools.fetch_data_dir, respectively.

        Returns: filepath as a string

        """
        if self.filepath is not None:
            return self.filepath
        from pycqed.analysis import analysis_toolbox as a_tools
        if self.timestamp is not None:
            self.folder = a_tools.get_folder(self.timestamp,
                                         suppress_printing=False,
                                         **kwargs)
        else:
            self.folder = kwargs.get('folder', None)
        self.filepath = a_tools.measurement_filename(self.folder,
                                                     ext=self.extension[1:],
                                                     **kwargs)
        return self.filepath

    @staticmethod
    def get_file_format(timestamp=None, folder=None, filepath=None,
                        file_id=None):
        """
        Returns the file format of a given timestamp.
        Args:
            timestamp (str): timestamp of the file
            folder (str): folder of the file if different from a_tools.datadir
            filepath (str): Optional filepath, overwrites timestamp
            file_id (str): suffix of the file name

        Returns (str): file format as a string (see file_extensions)

        """
        if filepath is None:
            from pycqed.analysis import analysis_toolbox as a_tools
            if timestamp is None:
                # if only folder is given, folder needs to point directly to
                # location of settings files
                folder_dir = folder
            else:
                folder_dir = a_tools.get_folder(timestamp, folder=folder)
            dirname = os.path.split(folder_dir)[1]
            path = Path(folder_dir)

            if file_id is not None:
                dirname = dirname + file_id
            filepath = sorted(path.glob(dirname + ".*"))

            if len(filepath) > 1:
                for format, extension in file_extensions.items():
                    for path in filepath:
                        file_name, file_extension = os.path.splitext(path)
                        if extension == file_extension:
                            logger.warning(
                                f"More than one file found for timestamp "
                                f"'{timestamp}'. File in format '{format}' will"
                                f" be considered.")
                            return format
                raise KeyError(f"More than one file found for "
                               f"timestamp '{timestamp}' and none matches the "
                               f"standard file extensions '{file_extensions}'.")
            elif len(filepath) == 0:
                raise KeyError(f"No file found for timestamp '{timestamp}'")
            else:
                filepath = filepath[0]
        file_name, file_extension = os.path.splitext(filepath)

        for format, extension in file_extensions.items():
            if file_extension == extension:
                return format

        raise KeyError(f"File extension '{file_extension}' not in "
                       f"standard form '{file_extensions}'")

    @staticmethod
    def load_instrument(inst_name, inst_dict):
        """
        Loads instrument object from settings given as a dictionary.
        Args:
            inst_name (str): Instrument name (key of the snapshot/dictionary)
            inst_dict (dict): Instrument values given as a dictionary

        Returns: Instrument object

        """
        inst = Instrument(inst_name)
        # load parameters
        if 'parameters' in inst_dict:
            for param_name, param_values in inst_dict['parameters'].items():
                par = Parameter(param_name, param_values)
                inst.add_parameter(par)
        # load class name
        if '__class__' in inst_dict:
            inst.add_classname(inst_dict['__class__'])
        # load submodules, treats submodules and channels as the same.
        for k in ['submodules', 'channels']:
            if k not in inst_dict:
                continue
            for submod_name, submod_dict in inst_dict[k].items():
                submod_inst = PickleLoader.load_instrument(
                    submod_name, submod_dict)
                inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        """
        Hollow function. Can be used in the future to load components which are
        not instruments.
        Args:
            comp_name (str): Component name (key of the snapshot/dictionary)
            comp (dict): Component values given as a dictionary

        Returns:
        """
        pass

    @staticmethod
    def decompress_file(file):
        import blosc2
        return blosc2.decompress(file)

    def get_station(self, param_path=None):
        """
        Returns a station build from a snapshot. This snapshot is loaded from
        a file using self.get_snapshot() and the timestamp given by the
        initialization of the Loader object.
        Returns (Station): station build from the snapshot.
        """
        if param_path is not None:
            logger.warning("For files which are not HDF5, the entire file will"
                           " be loaded in to the station regardless of "
                           "param_path.")
        station = Station(timestamp=self.timestamp)
        snap = self.get_snapshot()
        for inst_name, inst_dict in snap['instruments'].items():
            inst = self.load_instrument(inst_name, inst_dict)
            station.add_component(inst)

        for comp_name, comp_dict in snap['components'].items():
            # So far, components are not considered.
            # Loader.load_component is a hollow function.
            logger.warning('Components are not considered. '
                           'Loader.load_component is a hollow function.')
            comp = self.load_component(comp_name, comp_dict)
            if comp is not None:
                station.add_component(comp)

        return station

    def get_snapshot(self):
        return dict()


class HDF5Loader(Loader):

    def __init__(self, timestamp=None, filepath=None, extension=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         extension=extension)

        if self.extension is None:
            self.extension = file_extensions['hdf5']
        self.filepath = self.get_filepath(**kwargs)
        self.h5mode = kwargs.get('h5mode', 'r')

    @staticmethod
    def load_instrument(inst_name, inst_group):
        """
        Loads instrument object from settings given as a hdf object.
        Args:
            inst_name (str): Name of the instrument
            inst_group (hdf group): Values of the instrument as a group.

        Returns: Instrument object.

        """
        from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
        inst = Instrument(inst_name)
        # load parameters
        for param_name in list(inst_group.attrs):
            param_value = hlp_mod.decode_parameter_value(
                inst_group.attrs[param_name])
            par = Parameter(param_name, param_value)
            inst.add_parameter(par)
        # load class name
        if '__class__' in inst_group:
            inst.add_classname(
                hlp_mod.decode_parameter_value(inst_group.attrs['__class__']))
        # load submodules
        for submod_name, submod_group in list(inst_group.items()):
            submod_inst = HDF5Loader.load_instrument(submod_name, submod_group)
            inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        pass

    def get_station(self, param_path=None):
        """
        Loads settings from a config file into a station.
        param_path (list): list of parameters which are loaded into a
            station. If None, all parameters are loaded into the station.
            Parameters must be of the form %inst_name%.%param_name%.

        """
        with h5py.File(self.filepath, 'r') as file:
            config_file = file['Instrument settings']
            station = Station(timestamp=self.timestamp)
            if param_path is None:
                for inst_name, inst_group in list(config_file.items()):
                    inst = self.load_instrument(inst_name, inst_group)
                    station.add_component(inst)
            else:
                for path_to_param in param_path:
                    param_value = hlp_mod.extract_from_hdf_file(config_file,
                                                                path_to_param)
                    if param_value == 'not found':
                        # logger.warning(f"Parameter {path_to_param} not found.")
                        continue
                    # if param path is only an instrument (%inst_name%)
                    if len(path_to_param.split('.')) == 1:
                        inst_path = path_to_param
                        inst = Instrument(inst_path[-1])
                    else:
                        param_name = path_to_param.split('.')[-1]
                        param = Parameter(param_name, param_value)

                        inst_path = path_to_param.split('.')[:-1]

                        inst = Instrument(inst_path[-1])
                        inst.add_parameter(param)

                    # if more than one instrument name exists, the instrument
                    # must be a submodule
                    for inst_name in inst_path[-1:0:-1]:
                        submod = inst
                        inst = Instrument(inst_name)
                        inst.add_submodule(submod.name, submod)
                    if inst.name in station.components.keys():
                        station.components[inst.name].update(inst)
                    else:
                        station.add_component(inst)

        return station

    def get_snapshot(self):
        return self.get_station().snapshot()


class PickleLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         **kwargs)

        self.compression = kwargs.get('compression', False)
        if self.extension is None:
            if self.compression:
                self.extension = file_extensions['pickle_comp']
            else:
                self.extension = file_extensions['pickle']
        self.filepath = self.get_filepath(**kwargs)

    def get_snapshot(self):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import pickle

        with open(self.filepath, 'rb') as f:
            if self.compression:
                byte_data = Loader.decompress_file(f.read())
                snap = pickle.loads(byte_data)
            else:
                snap = pickle.load(f)
        return snap


class MsgLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         **kwargs)

        self.compression = kwargs.get('compression', False)
        if self.extension is None:
            if self.compression:
                self.extension = file_extensions['msgpack_comp']
            else:
                self.extension = file_extensions['msgpack']
        self.filepath = self.get_filepath(**kwargs)

    def get_snapshot(self):
        """
        Opens the msg file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import msgpack
        import msgpack_numpy
        # To save and load numpy arrays in msgpack.
        # Note that numpy arrays deserialized by msgpack-numpy are read-only
        # and must be copied if they are to be modified
        msgpack_numpy.patch()

        with open(self.filepath, 'rb') as f:
            if self.compression:
                import blosc2
                byte_data = Loader.decompress_file(f.read())
            else:
                byte_data = f.read()
            # To avoid problems with lists while unpacking use_list has to be
            # set to false. Otherwise, msgpack arrays are converted to Python
            # lists. To allow for tuples as dict keys, strict_map_key has to
            # be set to false. For more information
            # https://stackoverflow.com/questions/66835419/msgpack-dictionary
            # -with-tuple-keys and
            # https://github.com/msgpack/msgpack-python#major-breaking
            # -changes-in-msgpack-10
            snap = msgpack.unpackb(
                byte_data, use_list=False, strict_map_key=False)
        return snap


class Dumper:
    """
    Generic class to serialize dictionaries into a file.
    """

    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        """
        Creates a folder for the file.
        Args:
            name (str): additional label for the file
            data (dict): Dictionary which will be serialized
            datadir (str): root directory
            compression (bool): True if the file should be compressed with
                blosc2
            timestamp (str): timestamp which is used to create the folder and
                filename. It has to be in the format '%Y%m%d_%H%M%S'
        """
        self._name = name
        self.data = data
        self.compression = compression

        if timestamp == None:
            self._localtime = time.localtime()
        else:
            self._localtime = time.strptime(timestamp, '%Y%m%d_%H%M%S')
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        # sets the file path
        self.filepath = DateTimeGenerator().new_filename(
            self, folder=datadir, auto_increase=False)

        self.filepath = self.filepath.replace("%timemark", self._timemark)

        self.folder, self._filename = os.path.split(self.filepath)
        # creates the folder if needed
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def rdg_to_dict(self, raw_dict: dict):
        # helper function to convert Relative_Delay_Graph type to dict
        from pycqed.instrument_drivers.meta_instrument.device \
            import RelativeDelayGraph
        new_snap = {}
        for key, item in raw_dict.items():
            if isinstance(item, dict):
                new_snap[key] = self.rdg_to_dict(item)
            elif isinstance(item, RelativeDelayGraph):
                new_snap[key] = item._reld
            else:
                new_snap[key] = item
        return new_snap

    @staticmethod
    def compress_file(file):
        import blosc2
        return blosc2.compress(file)


class MsgDumper(Dumper):
    """
    Class to dump dictionaries into msg files.
    """
    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        super().__init__(name, data, datadir=datadir, compression=compression,
                         timestamp=timestamp)
        if self.compression:
            self.filepath = self.filepath.replace(
                ".hdf5", file_extensions['msgpack_comp'])
        else:
            self.filepath = self.filepath.replace(".hdf5",
                                                  file_extensions['msgpack'])

    def dump(self, mode='xb'):
        """
        Dumps the data as a binary into a msg file with optional compression
        mode: define which mode you want to open the file in.
            Default 'xb' creates the file and returns error if file exist
        """
        import msgpack
        import msgpack_numpy as msg_np
        msg_np.patch()

        with open(self.filepath, mode=mode) as file:
            packed = msgpack.packb(self.rdg_to_dict(self.data))
            if self.compression:
                packed = Dumper.compress_file(packed)
            file.write(packed)


class PickleDumper(Dumper):

    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        super().__init__(name, data, datadir=datadir, compression=compression,
                         timestamp=timestamp)
        if self.compression:
            self.filepath = self.filepath.replace(
                ".hdf5", file_extensions['pickle_comp'])
        else:
            self.filepath = self.filepath.replace(".hdf5",
                                                  file_extensions['pickle'])

    def dump(self, mode='xb'):
        """
        Dumps the data as a binary into a pickle file with optional compression
        mode: define which mode you want to open the file in.
            Default 'xb' creates the file and returns error if file exist
        """
        import pickle
        with open(self.filepath, mode=mode) as file:
            packed = pickle.dumps(self.rdg_to_dict(self.data))
            if self.compression:
                packed = Dumper.compress_file(packed)
            file.write(packed)


def get_loader_from_format(file_format, **kwargs):
    if 'hdf5' in file_format:
        return HDF5Loader(**kwargs)
    elif 'pickle' in file_format:
        return PickleLoader(**kwargs)
    elif 'msgpack' in file_format:
        return MsgLoader(**kwargs)
    else:
        raise NotImplementedError(f"File format '{file_format}' "
                                  f"not supported!")


def get_loader_from_file(timestamp=None, folder=None, filepath=None,
                         file_id=None, **params):
    file_format = Loader.get_file_format(timestamp=timestamp, folder=folder,
                                         filepath=filepath, file_id=file_id)
    if 'comp' in file_format:
        compression = True
    else:
        compression = False
    kwargs = dict(timestamp=timestamp, filepath=filepath,
                  compression=compression, folder=folder,
                  file_id=file_id)
    return get_loader_from_format(file_format, **kwargs)


def get_station_from_file(timestamp=None, folder=None, filepath=None,
                          file_id=None, param_path=None, **kwargs):
    return get_loader_from_file(timestamp=timestamp, folder=folder,
                                filepath=filepath, file_id=file_id) \
        .get_station(param_path=param_path)
