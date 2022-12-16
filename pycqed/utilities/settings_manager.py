# Importing annotations is required because Instrument class uses own class
# definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/typed-python-using-the-classes-own-type-inside-class-definition
# for more details.
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


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
        self.stations = {}
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

    def load_from_file(self, timestamp: str, file_format=None,
                       compression=False, extension=None,
                       filepath=None):
        """
        Loads station into the settings manager from saved files.
        Files contain dictionary of the snapshot (pickle, msgpack) or a hdf5
        representation of the snapshot.
        Args:
            timestamp (str): Supports all formats from a_tools.get_folder
            file_format (str): 'hdf5', 'pickle' or 'msgpack'
            compression (bool): Set True if files are compressed by blosc2
                (only for pickle and msgpack)
            extension (str): possibility to specify the extension of the file
                if is not matching the default extension (see respective loader
                class for default extensions)
            filepath (str): optionally a filepath can be passed of the file
                which should be loaded.

        Returns: Station which was created by the saved data.

        """
        if file_format is None:
            # if no file format is given, the loader tries to get the file
            # format by the extension of the file
            file_format, compression = Loader.get_file_format(timestamp)
        if file_format == 'hdf5':
            loader = HDF5Loader(timestamp=timestamp, h5mode='r',
                                extension=extension, filepath=filepath)
        elif file_format == 'pickle':
            loader = PickleLoader(timestamp=timestamp, compression=compression,
                                  extension=extension, filepath=filepath)
        elif file_format == 'msgpack':
            loader = MsgLoader(timestamp=timestamp, compression=compression,
                               extension=extension, filepath=filepath)
        else:
            raise NotImplementedError(f"File format '{file_format}' "
                                      f"not supported!")

        station = loader.get_station()
        self.add_station(station, timestamp)
        return station

    def spawn_snapshot_viewer(self, timestamp):
        """
        Opens a gui window to display the snapshot of the given station.0
        Args:
            timestamp (str): address the station which is already loaded onto
                the settings manager.

        Returns:

        """
        # snap = self.stations[timestamp].snapshot()
        # qt_app = QtWidgets.QApplication(sys.argv)
        # snap_viewer = dict_viewer.DictViewerWindow(
        #     snap, 'Snapshot timestamp: %s' % timestamp)
        # qt_app.exec_()
        import pycqed.gui.dict_viewer as dict_viewer
        snapshot_viewer = dict_viewer.SnapshotViewer(
            snapshot=self.stations[timestamp].snapshot(),
            timestamp=timestamp)
        snapshot_viewer.spawn_snapshot_viewer()

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
            for i, tsp in enumerate(timestamps):
                if component not in self.stations[tsp].components.keys():
                    all_msg.append(
                        f'\nComponent/Instrument "{component}" missing in dict '
                        f'{tsp}.\n')
                    component_not_in_all_stations = True
            if component_not_in_all_stations:
                # if one of the stations does not have the component, the
                # comparison of this component cannot be done
                # (You have to compare like with like)
                continue

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
                         reduced_compare=False, output='viewer'):
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
            import pycqed.gui.dict_viewer as dict_viewer
            snapshot_viewer = dict_viewer.SnapshotViewer(
                snapshot=diff,
                timestamp=ts_list)
            snapshot_viewer.spawn_comparison_viewer()


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

    def __init__(self, timestamp: str = None, filepath: str = None,
                 extension=None):
        self.filepath = filepath
        self.timestamp = timestamp
        self.extension = extension

    def get_filepath(self):
        """
        If no explicit filepath is given, a_tools.get_folder tries to get the
        directory based on the timestamp and the directory defined by
        a_tools.datadir, a_tools.fetch_data_dir, respectively.

        Returns: filepath as a string

        """
        if self.filepath is not None:
            return self.filepath
        from pycqed.analysis import analysis_toolbox as a_tools
        folder_dir = a_tools.get_folder(self.timestamp, suppress_printing=False)
        self.filepath = a_tools.measurement_filename(folder_dir,
                                                     ext=self.extension)
        return self.filepath

    @staticmethod
    def get_file_format(timestamp: str):
        """
        Returns the file format and the compression (bool) of a given timestamp.
        Args:
            timestamp (str): timestamp of the file

        Returns (str and bool): file format as a string ('hdf5', 'pickle',
            'msgpack') and if the file is compressed

        """
        from pycqed.analysis import analysis_toolbox as a_tools
        import os
        from pathlib import Path

        folder_dir = a_tools.get_folder(timestamp, suppress_printing=True)
        dirname = os.path.split(folder_dir)[1]
        if dirname[6:9] == '_X_':
            fn = dirname[0:7]+dirname[9:]
        else:
            fn = dirname
        path = Path(folder_dir)

        file_path = sorted(path.glob(fn+".*"))
        if len(file_path) > 1:
            raise KeyError(f"More than one file found for "
                           f"timestamp '{timestamp}'")
        elif len(file_path) == 0:
            raise KeyError(f"No file found for timestamp '{timestamp}'")
        else:
            file_name, file_extension = os.path.splitext(file_path[0])

        if 'pickle' in file_extension:
            if file_extension == '.picklec':
                return "pickle", True
            else:
                return "pickle", False

        elif 'msg' in file_extension:
            if file_extension == '.msgc':
                return "msgpack", True
            else:
                return "msgpack", False

        elif 'hdf5' in file_extension:
            return "hdf5", False

        else:
            raise KeyError(f"File extension '{file_extension}' not in "
                           f"standard form (.hdf5, .pickle(c), .msg(c)")

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

    def get_station(self):
        """
        Returns a station build from a snapshot. This snapshot is loaded from
        a file using self.get_snapshot() and the timestamp given by the
        initialization of the Loader object.
        Returns (Station): station build from the snapshot.
        """
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

    def __init__(self, timestamp=None, filepath=None, h5mode='r',
                 extension=None):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         extension=extension)

        if self.extension is None:
            self.extension = 'hdf5'
        self.filepath = self.get_filepath()
        self.h5mode = h5mode

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
            param_value = BaseDataAnalysis.get_hdf_datafile_param_value(
                inst_group, param_name)
            par = Parameter(param_name, param_value)
            inst.add_parameter(par)
        # load class name
        if '__class__' in inst_group:
            inst.add_classname(
                BaseDataAnalysis.get_hdf_datafile_param_value(
                    inst_group, '__class__'))
        # load submodules
        for submod_name, submod_group in list(inst_group.items()):
            submod_inst = HDF5Loader.load_instrument(submod_name, submod_group)
            inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        pass

    def get_station(self):
        """
        Loads settings from an hdf5 file into a station.
        """
        import h5py

        with h5py.File(self.filepath, self.h5mode) as data_file:
            station = Station(timestamp=self.timestamp)
            instr_settings = data_file['Instrument settings']

            for inst_name, inst_group in list(instr_settings.items()):
                inst = self.load_instrument(inst_name, inst_group)
                station.add_component(inst)
        return station


class PickleLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, compression=False,
                 extension=None):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         extension=extension)

        self.compression = compression
        if self.extension is None:
            if self.compression:
                self.extension = 'picklec'
            else:
                self.extension = 'pickle'
        self.filepath = self.get_filepath()
        # if self.compression:
        #     self.filepath = self.get_filepath(
        #         timestamp, extension='picklec')
        # else:
        #     self.filepath = self.get_filepath(
        #         timestamp, extension='pickle')

    def get_snapshot(self):
        return self._get_snapshot(self.filepath, self.compression)

    @staticmethod
    def _get_snapshot(filepath, compression):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import pickle

        with open(filepath, 'rb') as f:
            if compression:
                import blosc2
                byte_data = blosc2.decompress(f.read())
                snap = pickle.loads(byte_data)
            else:
                snap = pickle.load(f)
        return snap


class MsgLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, compression=False,
                 extension=None):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         extension=extension)

        self.compression = compression
        if self.extension is None:
            if self.compression:
                self.extension = 'msgc'
            else:
                self.extension = 'msg'
        self.filepath = self.get_filepath()

    def get_snapshot(self):
        return self._get_snapshot(self.filepath, self.compression)

    @staticmethod
    def _get_snapshot(filepath, compression):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import msgpack
        import msgpack_numpy
        # To save and load numpy arrays in msgpack.
        # Note that numpy arrays deserialized by msgpack-numpy are read-only
        # and must be copied if they are to be modified
        msgpack_numpy.patch()

        with open(filepath, 'rb') as f:
            if compression:
                import blosc2
                byte_data = blosc2.decompress(f.read())
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
                 compression=False):
        """
        Creates a folder for the file.
        Args:
            name (str): additional label for the file
            data (dict): Dictionary which will be serialized
            datadir (str): root directory
            compression (bool): True if the file should be compressed with
                blosc2
        """
        import time
        from pycqed.measurement.hdf5_data import DateTimeGenerator \
            as DateTimeGenerator
        import os

        self._name = name
        self.data = data
        self.compression = compression

        self._localtime = time.localtime()
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        # sets the file path
        self.filepath = DateTimeGenerator().new_filename(
            self, folder=datadir)

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


class MsgDumper(Dumper):
    """
    Class to dump dictionaries into msg files.
    """
    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False):
        super().__init__(name, data, datadir=datadir, compression=compression)
        self.filepath = self.filepath.replace("hdf5", "msg")
        if self.compression:
            self.filepath = self.filepath + "c"

    def dump(self):
        """
        Dumps the data as a binary into a msg file with optional compression
        """
        import msgpack
        import msgpack_numpy as msg_np
        msg_np.patch()

        with open(self.filepath, 'wb') as file:
            packed = msgpack.packb(self.rdg_to_dict(self.data))
            if self.compression:
                import blosc2
                packed = blosc2.compress(packed)
            file.write(packed)


class PickleDumper(Dumper):

    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False):
        super().__init__(name, data, datadir=datadir, compression=compression)
        self.filepath = self.filepath.replace("hdf5", "pickle")
        if self.compression:
            self.filepath = self.filepath + "c"

    def dump(self):
        """
        Dumps the data as a binary into a pickle file with optional compression
        """
        import pickle
        with open(self.filepath, 'wb') as file:
            packed = pickle.dumps(self.rdg_to_dict(self.data))
            if self.compression:
                import blosc2
                packed = blosc2.compress(packed)
            file.write(packed)
