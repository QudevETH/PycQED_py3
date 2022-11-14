# Importing annotations is required because Instrument class uses own class
# definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/
#   typed-python-using-the-classes-own-type-inside-class-definition
# for more details.
from __future__ import annotations
import sys

import blosc2
import h5py
import pickle
import msgpack
import msgpack_numpy
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.gui.dict_viewer as dict_viewer
import PyQt5.QtWidgets as QtWidgets
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
import qcodes as qc

# To save and load numpy arrays in msgpack.
# Note that numpy arrays deserialized by msgpack-numpy are read-only
# and must be copied if they are to be modified
msgpack_numpy.patch()


class DelegateAttributes:
    """
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
                 vals=None) -> None:
        """
        Parameter has a (unique) name and a respective value
        Args:
            name: Name of parameter (str)
            vals: Value of parameter (any type)
        """
        self._name = str(name)
        self._vals = vals

    def get_name(self):
        return self._name

    def get_value(self):
        return self._vals

    def snapshot(self) -> dict[any, any]:
        """
        Creates a dictionary out of vals attribute
        Returns: dictionary
        """
        snap: dict[any, any] = self._vals
        return snap


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
        namestr = param.get_name()
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
        if timestamp == None:
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
    Class which contains different station which are loaded from saved settings.
    Supported file types: hdf5, pickle (.obj), msgpack (.msg)
    Snapshot of stations can be displayed via dict_viewer module.
    """

    def __init__(self, stat: Station = None):
        """
        Initialization of SettingsManager instance. Can be called with a
        preexisting station.
        Args:
            stat (Station): optional, adds given station to the settings manager
        """
        if stat == None:
            self.stations = {}
        else:
            self.stations = {stat.timestamp: stat}

    def add_station(self, station, timestamp: str):
        """
        Adds a given station with a timestamp as its unique name.
        Args:
            station (Station): Station object which will be added.
                Either QCodes station or Station class from this module
            timestamp (str): Timestamp to call the station.
                Is converted into unified format.

        """
        timestamp = '_'.join(a_tools.verify_timestamp(timestamp))
        if isinstance(station, Station) or isinstance(station, qc.Station):
            self.stations[timestamp] = station
        else:
            raise TypeError(f'Cannot add station "{timestamp}", because the '
                            'station is not a QCode or Mock station class')

    def load_from_file(self, timestamp: str, filetype='msgpack',
                       compression=False):
        """
        Loads station into the settings manager from saved files.
        Files contain dictionary of the snapshot (pickle, msgpack) or a hdf5
        representation of the snapshot.
        Args:
            timestamp (str): Supports all formats from a_tools.get_folder
            filetype (str): 'hdf5', 'pickle' or 'msgpack'
            compression: Set True if files are compressed by blosc2
                (only for pickle and msgpack)

        Returns: Station which was created by the saved data.

        """
        if filetype not in ['hdf5', 'pickle', 'msgpack']:
            raise Exception('File type not supported!')
        if filetype == 'hdf5':
            # hdf5 files are not saved as a dictionary.
            # If follows another loading scheme.
            self.load_from_hdf5(timestamp=timestamp)
            return
        if filetype == 'pickle':
            loader = PickleLoader(timestamp=timestamp, compression=compression)
        if filetype == 'msgpack':
            loader = MsgLoader(timestamp=timestamp, compression=compression)

        snap = loader.get_snapshot()

        stat = Station(timestamp=timestamp)

        for inst_name, inst_dict in snap['instruments'].items():
            inst = Loader.load_instrument(inst_name, inst_dict)
            stat.add_component(inst)

        for comp_name, comp_dict in snap['components'].items():
            # So far, components are not considered.
            # Loader.load_component is a hollow function.
            comp = Loader.load_component(comp_name, comp_dict)
            if comp is not None:
                stat.add_component(comp)

        self.add_station(stat, timestamp)
        return stat

    def load_from_hdf5(self, timestamp: str, h5mode='r'):
        """
        Loads settings from an hdf5 file into a station.
        Args:
            timestamp (str): Uniquelily identifies the file. If file with
                timestamp not in a_toold.datadir, a_tools tries to fetch
                from a_tools.fetch_data_dir.
            h5mode: 'r' for read only mode.

        Returns: station
        """
        loader = HDFLoader(timestamp=timestamp)

        data_file = h5py.File(loader.filepath, h5mode)
        stat = Station(timestamp=timestamp)
        instr_settings = data_file['Instrument settings']

        for inst_name, inst_group in list(instr_settings.items()):
            inst = loader.load_instrument(inst_name, inst_group)
            stat.add_component(inst)
        self.add_station(stat, timestamp)
        return stat

    def spawn_snapshot_viewer(self, timestamp):
        """
        Opens a gui window to display the snapshot of the given station.0
        Args:
            timestamp (str): address the station which is already loaded onto
                the settings manager.

        Returns:

        """
        snap = self.stations[timestamp].snapshot()
        qt_app = QtWidgets.QApplication(sys.argv)
        snap_viewer = dict_viewer.DictViewerWindow(
            snap, 'Snapshot timestamp: %s' % timestamp)
        qt_app.exec_()


class Loader:
    """
    Generic class to load instruments and parameters from a file.
    """
    def __init__(self):
        self.filepath = None

    def get_filepath(self, timestamp=None, filepath=None, extension=None):
        """
        If no explicit filepath is given, a_tools.get_folder tries to get the
        directory based on the timestamp and the directory defined by
        a_tools.datadir, a_tools.fetch_data_dir, respectively.
        Args:
            timestamp (str): timestring in a format which is accepted by a_tools
            filepath (str): explicit filepath. If given, filepath is set to
                this value
            extension (str): extension of the filetype, usually 'hdf5' for hdf5,
                'obj' for pickle and 'msg' for msgpack.
                'objc' and 'msgc' for compressed pickle and msgpack files.

        Returns: filepath as a string

        """
        if filepath is not None:
            return filepath
        folder_dir = a_tools.get_folder(timestamp, suppress_printing=False)
        return a_tools.measurement_filename(folder_dir, ext=extension)

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
            for param_name, param_vals in inst_dict['parameters'].items():
                par = Parameter(param_name, param_vals)
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
            comp_name:
            comp:

        Returns:
        """
        pass


class HDFLoader(Loader):

    def __init__(self, timestamp=None, filepath=None):
        super().__init__()
        self.filepath = self.get_filepath(timestamp, filepath, extension='hdf5')

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
            submod_inst = HDFLoader.load_instrument(submod_name, submod_group)
            inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        pass


class PickleLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, compression=False):
        super().__init__()
        self.compression = compression
        if self.compression:
            self.filepath = self.get_filepath(
                timestamp, filepath, extension='objc')
        else:
            self.filepath = self.get_filepath(
                timestamp, filepath, extension='obj')

    def get_snapshot(self):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        with open(self.filepath, 'rb') as f:
            if self.compression:
                byte_data = blosc2.decompress(f.read())
                snap = pickle.loads(byte_data)
            else:
                snap = pickle.load(f)
            f.close()
        return snap


class MsgLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, compression=False):
        super().__init__()
        self.compression = compression
        if self.compression:
            self.filepath = self.get_filepath(
                timestamp, filepath, extension='msgc')
        else:
            self.filepath = self.get_filepath(
                timestamp, filepath, extension='msg')

    def get_snapshot(self):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        with open(self.filepath, 'rb') as f:
            if self.compression:
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
            f.close()
        return snap
