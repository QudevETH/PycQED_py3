# Importing annotations is required because Instrument class uses own class
# definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/typed-python-using-the-classes-own-type-inside-class-definition
# for more details.
from __future__ import annotations

from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.gui.dict_viewer as dict_viewer
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
import logging

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

    def snapshot(self) -> dict[any, any]:
        """
        Creates a dictionary out of value attribute
        Returns: dictionary
        """
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

    def snapshot(self) -> dict[any, any]:
        """
        Creates recursively a snapshot (dictionary) which has the same structure
        as the snapshot from QCodes instruments.

        Returns (dict): Returns a snapshot as a dictionary with keys
            'functions', 'submodules', 'parameters', '__class__' and 'name'

        """
        param: dict[any, any] = {}
        for key, item in self.parameters.items():
            param[key] = item.snapshot()

        submod: dict[any, any] = {}
        for key, item in self.submodules.items():
            submod[key] = item.snapshot()

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
            self.timestamp = '_'.join(a_tools.verify_timestamp(timestamp))

    def snapshot(self) -> dict[any, any]:
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
                inst[key] = item.snapshot()
            else:
                components[key] = item.snapshot()

        param: dict[any, any] = {}
        for key, item in self.parameters.items():
            param[key] = item.snapshot()

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

    def load_from_file(self, timestamp: str, file_format='msgpack',
                       compression=False, extension=None):
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

        Returns: Station which was created by the saved data.

        """
        if file_format == 'hdf5':
            loader = HDF5Loader(timestamp=timestamp, h5mode='r',
                                extension=extension)
        elif file_format == 'pickle':
            loader = PickleLoader(timestamp=timestamp, compression=compression,
                                  extension=extension)
        elif file_format == 'msgpack':
            loader = MsgLoader(timestamp=timestamp, compression=compression,
                               extension=extension)
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
        snapshot_viewer = dict_viewer.SnapshotViewer(
            snapshot=self.stations[timestamp].snapshot(),
            timestamp=timestamp)
        snapshot_viewer.spawn_snapshot_viewer()


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
        folder_dir = a_tools.get_folder(self.timestamp, suppress_printing=False)
        self.filepath = a_tools.measurement_filename(folder_dir,
                                                     ext=self.extension)
        return self.filepath

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
                self.extension='picklec'
            else:
                self.extension='pickle'
        self.filepath = self.get_filepath()
        # if self.compression:
        #     self.filepath = self.get_filepath(
        #         timestamp, extension='picklec')
        # else:
        #     self.filepath = self.get_filepath(
        #         timestamp, extension='pickle')

    def get_snapshot(self):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import pickle

        with open(self.filepath, 'rb') as f:
            if self.compression:
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
                self.extension='msgc'
            else:
                self.extension='msg'
        self.filepath = self.get_filepath()

    def get_snapshot(self):
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

        with open(self.filepath, 'rb') as f:
            if self.compression:
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
