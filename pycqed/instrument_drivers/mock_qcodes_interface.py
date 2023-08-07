# Importing annotations is required because Instrument class uses own class
# definition inside Instrument.add_submodule
# see https://stackoverflow.com/questions/42845972/typed-python-using-the-classes-own-type-inside-class-definition
# for more details.
from __future__ import annotations
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
        # updates the submodules recursively
        for submod_name, submod in instrument.submodules.items():
            if submod_name in self.submodules:
                self.submodules[submod_name].update(submod)
            else:
                self.submodules[submod_name] = submod
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
                param_value = param.get("value", param)
            else:
                param_value = param
        except Exception:
            param_value = 'not found'
            logger.warning(f'Parameter {path_to_param} not found in station.')
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
