# General imports
# why this import: Instrument class uses own class definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/typed-python-using-the-classes-own-type-inside-class-definition
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
from pycqed.instrument_drivers.meta_instrument.device import Device, RelativeDelayGraph
from pycqed.measurement import hdf5_data as h5d

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
    Instrument class which mocks Instrument class of QCodes. An instrument object contains given parameters.
    Submodules are instruments themselves.
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
        self.classname: str = None
        # QCodes instruments store the respective Class name of an instrument,
        # e.g. pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon.QuDev_transmon

    def snapshot(self) -> dict[any, any]:
        """
        Creates recursively a snapshot (dictionary) which has the same structure as the snapshot
        from QCodes instruments.

        Returns (dict): Returns a snapshot as a dictionary with keys 'functions', 'submodules', 'parameters',
            '__class__' and 'name'

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
                'parameter of that name is already registered to the instrument')
        self.parameters[namestr] = param

    def add_classname(self, name: str):
        self.classname = name

    def add_submodule(self, name: str, submod: Instrument):
        """
        Adds a submodule (aka channel) which is an instrument themselves. We make no distinction between submodules
        and channels (unlike QCodes).
        To force submod to be an Instrument, from __future__ import annotations has to be added at the beginning.
        Args:
            name (str): name of the respective submodule or channel
            submod (Instrument): Instrument object of the respective submodule.

        """
        self.submodules[name] = submod


class Station(DelegateAttributes):
    """
    Station class which mocks Station class of QCodes. Station contains all given instruments and parameters.
    """

    delegate_attr_dicts = ['components']

    def __init__(self, timestamp: str = None):
        """
        Initialization of the station. Each station is characterized by the timestamp.
        Args:
            timestamp (str): For accepted formats of the timestamp see a_tools.verify_timestamp(str)
        """
        self.instruments: dict = {}
        self.parameters: dict = {}
        self.components: dict = {}
        self.config: dict = {}
        if timestamp == None:
            self.timestamp = ''
        else:
            self.timestamp = '_'.join(a_tools.verify_timestamp(timestamp))
            # a_tools.verify_timestamp returns unified version of the timestamps

    def snapshot(self) -> dict[any, any]:
        """
        Returns a snapshot as a dictionary of the entire station based on the structure of QCodes.
        Returns (dict): snapshot of the station with keys 'instruments', 'parameters', 'components' and 'config'

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
        Adds a given instrument to the station under the constriction that there exists no instrument with the
        same name.
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
    """
    def __init__(self, stat: Station = None):
        """
        Initialization of SettingsManager instance. Can be called with a preexisting station.
        Args:
            stat (Station): optional, adds given station to the settings manager.
        """
        if stat == None:
            self.stations = {}
        else:
            self.stations = {stat.timestamp: stat}

    def add_station(self, station: Station, timestamp: str):
        """
        Adds a given station with a timestamp as its unique name.
        Args:
            station (Station): Station object which will be added.
            timestamp (str): Timestamp to call the station. Is converted into unified format.

        """
        timestamp = '_'.join(a_tools.verify_timestamp(timestamp))
        self.stations[timestamp] = station

    def load_from_file(self, timestamp: str, filetype='msgpack', compression=False):
        if filetype not in ['hdf5', 'pickle', 'msgpack']:
            raise Exception('File type not supported!')
        if filetype == 'hdf5':
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
            comp = Loader.load_component(comp_name, comp_dict)
            if comp is not None:
                stat.add_component(comp)

        self.add_station(stat, timestamp)
        return stat

    def load_from_hdf5(self, timestamp: str, h5mode='r'):
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
        snap = self.stations[timestamp].snapshot()
        qt_app = QtWidgets.QApplication(sys.argv)
        snap_viewer = dict_viewer.DictViewerWindow(snap, 'Snapshot timestamp: %s' % timestamp)
        try:
            qt_app.exec_()
        except Exception as e:
            print(e)


class Loader:
    def __init__(self):
        self.filepath = None

    def get_filepath(self, timestamp=None, filepath=None, extension=None):
        if filepath is not None:
            return filepath
        folder_dir = a_tools.get_folder(timestamp, suppress_printing=False)
        return a_tools.measurement_filename(folder_dir, ext=extension)

    @staticmethod
    def load_instrument(inst_name, inst_dict):
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
                submod_inst = PickleLoader.load_instrument(submod_name, submod_dict)
                inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        pass


class HDFLoader(Loader):

    def __init__(self, timestamp=None, filepath=None):
        super().__init__()
        self.filepath = self.get_filepath(timestamp, filepath, extension='hdf5')

    @staticmethod
    def load_instrument(inst_name, inst_group):
        inst = Instrument(inst_name)
        # load parameters
        for param_name in list(inst_group.attrs):
            param_value = BaseDataAnalysis.get_hdf_datafile_param_value(inst_group, param_name)
            par = Parameter(param_name, param_value)
            inst.add_parameter(par)
        # load class name
        if '__class__' in inst_group:
            inst.add_classname(BaseDataAnalysis.get_hdf_datafile_param_value(inst_group, '__class__'))
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
            self.filepath = self.get_filepath(timestamp, filepath, extension='objc')
        else:
            self.filepath = self.get_filepath(timestamp, filepath, extension='obj')

    def get_snapshot(self):
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
            self.filepath = self.get_filepath(timestamp, filepath, extension='msgc')
        else:
            self.filepath = self.get_filepath(timestamp, filepath, extension='msg')

    def get_snapshot(self):
        with open(self.filepath, 'rb') as f:
            if self.compression:
                byte_data = blosc2.decompress(f.read())
            else:
                byte_data = f.read()
            snap = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
            # https://stackoverflow.com/questions/66835419/msgpack-dictionary-with-tuple-keys
            f.close()
        return snap
