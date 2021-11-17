import time
import numpy as np
import qcodes as qc
from qcodes import validators as vals
from qcodes import Instrument
from copy import copy, deepcopy

# driver for NationalInstruments USB6501
try:
    from pycqedscripts.drivers.NI_USB6501 import NationalInstrumentsUSB6501
except Exception:
    NationalInstrumentsUSB6501 = type(None)

import logging
log = logging.getLogger(__name__)


class VirtualNationalInstrumentsUSB6501(Instrument):
    """
    A virtual National Instruments USB-6501 digital I/O card for using
    the SwitchControl class in virtual setups.
    """
    def __init__(self, name, device_name=None):
        super().__init__(name)
        self.port_vals = [0,0,0]
        self.prev_port_vals = [0,0,0]

    def write_port(self, port_number, value):
        self.prev_port_vals[port_number] = self.port_vals[port_number]
        self.port_vals[port_number] = value
        log.info(f'{self.name}: Writing {value} to port {port_number}.')

    def read_port(self, port_number):
        val = self.port_vals[port_number]
        log.info(f'{self.name}: Reading {val} from port {port_number}.')
        return val


class SwitchControl(Instrument):
    """
    A meta-instrument for controlling different switch types via a National
    Instruments USB-6501 digital I/O card.

    Args:
        name:        name of the instrument
        dio:         reference to an NI-USB6501 Intrument
        switches:    a dictionary where the keys are switch names and
            the values are dictionaries with the following keys:
             type: (str or class) switch class name (or class). One of the
                switch type classes defined below or a custom class.
             all further keys: passes kwargs to the switch class init
        switch_time: duration of the pulse to set the switch configuration
    """

    shared_kwargs = ['dio']

    def __init__(self, name, dio, switches, switch_time=50e-3):
        super().__init__(name)

        if not (isinstance(dio, NationalInstrumentsUSB6501) or
                isinstance(dio, VirtualNationalInstrumentsUSB6501)):
            raise Exception('Specified Instrument is not an Instance of '
                            'NationalInstruments_USB6501.')
        self.dio = dio
        self.dio_type = 'NI-USB6501'
        self.n_ports = 3

        # This private variable will be used to prepare multiple switch
        # states and then write them in a single write command.
        self._lock_write = False

        # parameter for switch time
        self.add_parameter('switch_time', unit='s', vals=vals.Numbers(0, 1),
                           label='Duration of the switching pulse',
                           parameter_class=qc.ManualParameter,
                           initial_value=switch_time)

        # parameters for the individual switches
        self.switch_config = deepcopy(switches)
        self.switches = {}
        self.switch_params = {}
        for k, v in self.switch_config.items():
            v = copy(v)
            switch_type = v.pop('type')
            if isinstance(switch_type, str):
                switch_type = eval(switch_type)
            self.switches[k] = switch_type(name=k, instrument=self, **v)
        for switch in self.switches.values():
            param_name = f'{switch.name}_mode'
            self.add_parameter(
                param_name,
                label=switch.label,
                vals=qc.validators.Enum(*switch.modes),
                get_cmd=switch.get,
                set_cmd=switch.set,
                docstring="possible values: " + ', '.join(
                    [f'{m}' for m in switch.modes]),
            )
            self.switch_params[switch.name] = self.parameters[param_name]

        # ensure that the initial switch states are written to the controller
        self.set_switch({})

    def set_switch(self, values):
        """
        Set multiple switches to the given modes with a single hardware
        access. For switches that are not contained in the values dictionary,
        the state currently stored in the switch object will be written to
        the hardware.

        :param values: dict where each key is a switch name (or switch
            object) and value is the mode to set the switch to.
        """
        # Update individual switches without writing to the hardware.
        # (_lock_write is needed since the set parser of each individual
        # switch would otherwise directly write to the hardware.)
        self._lock_write = True
        for switch, val in values.items():
            if not isinstance(switch, str):
                switch = switch.name
            self.parameters[f'{switch}_mode'](val)
        # Now write the current switch states to the hardware.
        self._lock_write = False
        self._set_switch()

    def _set_switch(self):
        """
        Private method as a wrapper for calling _set_switch_NI_USB6501 with
        correct parameters. This method is called from set_switch and from
        the set method of an individual switch. The current switch states
        will only be written to the hardware if self._lock_write is False.
        """
        if self._lock_write:
            return
        state_set = 0
        state_keep = 0
        for switch in self.switches.values():
            pattern = switch.bit_pattern()
            state_set += pattern
            if not switch.latching:
                state_keep += pattern
        self._set_switch_NI_USB6501(state_set, state_keep)

    def _set_switch_NI_USB6501(self, state_set, state_keep):
        """
        Write settings to the dio board: update ports, hold for switch time,
        set back to 0 for latching switches (keep values for non-latching
        switches).
        :param state_set: The bit pattern to set the switches.
        :param state_keep: The bit pattern to keep after setting the switches.
        """
        def write_ports(state):
            state_str = format(state, f'0{8*self.n_ports}b')
            for i in range(self.n_ports):
                self.dio.write_port(i, eval(
                    '0b' + state_str[(self.n_ports-i-1)*8:(self.n_ports-i)*8]))
        write_ports(state_set)
        time.sleep(self.switch_time())
        write_ports(state_keep)


class MultiSwitchControl(Instrument):
    def __init__(self, name, switch_controls, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.switch_controls = switch_controls
        for scA in switch_controls:
            for scB in switch_controls:
                if scA.name.startswith(f'{scB.name}_'):
                    raise NameError(
                        f'The name "{scA.name}" starts with the name of '
                        f'"{scB.name}" plus an underscore. This is not allowed '
                        f'as it might lead to ambiguous parameter names.')
        self.switch_params = {}
        for sc in self.switch_controls:
            for k, param in sc.parameters.items():
                if not k.endswith('_mode'):
                    continue
                sw_name = k[:-5]  # remove '_mode'
                self.switch_params[
                    self._format_switch_name(sc.name, sw_name)] = param
                label = f'{param.label} in switch control {sc.name}'
                self.add_parameter(
                    f'{sc.name}_{sw_name}_mode',
                    label=label, get_cmd=param, set_cmd=param,
                    docstring=f'{label}\npossible values: ' + ', '.join(
                        [f'{v}' for v in param.vals.valid_values]),
                )

    @staticmethod
    def _format_switch_name(switch_control_name, switch_name):
        return f'{switch_control_name}_{switch_name}'

    def _split_switch_name(self, switch_name):
        scs = [sc.name for sc in self.switch_controls
               if switch_name.startswith(f'{sc.name}_')]
        if len(scs) != 1:
            raise KeyError(f'Could not find switch control for switch '
                           f'{switch_name}.')
        return scs[0], switch_name[(len(scs[0]) + 1):]

    @property
    def switches(self):
        return {self._format_switch_name(sc.name, sw_name): sw
                for sc in self.switch_controls if hasattr(sc, 'switches')
                for sw_name, sw in sc.switches.items()}

    def set_switch(self, values):
        values = {self._split_switch_name(k): v for k, v in values.items()}
        for sc in self.switch_controls:
            values_sc = {k[1]: v for k, v in values.items() if k[0] == sc.name}
            if hasattr(sc, 'set_switch'):  # supports setting multiple params
                sc.set_switch(values_sc)
            else:  # set each parameter individually
                for k, v in values_sc.items():
                    self.parameters[f'{sc.name}_{k}_mode'](v)


class SwitchType:
    """
    A base class for switch types to be used with the SwitchControl class.

    Args:
        :param name: (str) name of the switch
        :param bit_mapping: (list of ints) indicate to which physical pin
            number each control bit of the switch is mapped. The length should
            be the same as the the length of the lists in self.modes defined in
            the child class. (default: pin number equals bit number for each
            bit)
        :param default_mode: (str) name of the mode that the switch is set to
            at init (default: None, which will leave the switch in an
            undefined state)
        :param label: (str) a human readable description of the switch
            (default: generated from the name)
        :param instrument: the SwitchControl object to which the switch
            belongs. Will be set by the SwitchControl when instantiating the
            switch and should not be provided by the user.
        :param kw: currently ignored

    Each child class should define:
        self.modes: a dictionary where each key is a mode name (state),
            and the corresponding value is a list of binary vaules (with the
            same length as bit_mapping)
        self.latching: whether the switch is latching (default: True)
    """
    def __init__(self, name, bit_mapping=None, default_mode=None,
                 label=None, instrument=None, **kw):
        self.name = name
        self.label = f'switch configuration of {name}' if label is None else \
            label
        self.modes = {}
        self.default_mode = default_mode
        self.bit_mapping = bit_mapping
        self.latching = True
        self.instrument = instrument
        self._state = self.default_mode

    def set(self, mode):
        self._check_valid_mode(mode)
        self._state = mode
        if self.instrument is not None:
            self.instrument._set_switch()

    def get(self):
        return self._state

    def bit_pattern(self, mode=None):
        if mode is None:
            mode = self._state
        self._check_valid_mode(mode)
        return self._translate_pattern(self.modes[mode])

    def _translate_pattern(self, pattern):
        if isinstance(pattern, list):
            pattern = sum([a << b for a, b in zip(pattern, range(len(pattern)))])
        if self.bit_mapping is None:
            return pattern
        return sum([int(a) << b for a, b in zip(
            f'{pattern:b}'[::-1], self.bit_mapping)])

    def _check_valid_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(f'Trying to set switch {self.name} to '
                             f'invalid mode: {node}')


class MultiSwitch(SwitchType):
    """
    A switch to route one out of multiple inputs to an output (or one
    input to one out of multiple outputs). In addition, the switch can
    block, i.e., route none of the inputs (outputs).

    :param n_states: (int) the number of inputs (or outputs)

    See base class for further parameters.
    """
    def __init__(self, n_states=2, name='switch', default_mode='block', **kw):
        super().__init__(name=name, default_mode=default_mode, **kw)
        self.n_states = n_states
        self.modes = {'block': 0b00000000}
        self.modes.update({(i + 1): (1 << i) for i in range(self.n_states)})


class BinaryChainSwitch(MultiSwitch):
    """
    A chain of two-port switches used as a single multi switch.

    :param n_switches: (int) the number of switches in the chain

    See base class for further parameters.
    """
    def __init__(self, n_switches=2, name='switch', default_mode='block',
                 **kw):
        super().__init__(name=name, n_states=n_switches+1,
                         default_mode=default_mode, **kw)
        b = 0
        for i in range(n_switches):
            self.modes[i + 1] = (1 << 2*i) + b
            b = (b << 2) + 2
        self.modes[n_switches + 1] = b


class UCSwitch(SwitchType):
    """
    The switch in a QuDev upconversion board with the modes 'modulated' and
    'bypass'. See base class for parameters.
    """
    def __init__(self, name='UC', default_mode='modulated', **kw):
        super().__init__(name=name, default_mode=default_mode, **kw)
        self.modes = {'modulated': [0, 1],
                      'bypass': [1, 0]}


class HDIQSwitch(SwitchType):
    """
    The switch in a ZI HDIQ channel with the modes 'modulated', 'spec',
    and 'calib'. See base class for parameters.
    """
    def __init__(self, name='UC', default_mode='modulated', **kw):
        super().__init__(name=name, default_mode=default_mode, **kw)
        self.modes = {'modulated': [0, 0],
                      'spec': [1, 0],
                      'calib': [0, 1]}
        self.latching = False


class WASwitch(SwitchType):
    """
    The switch in a QuDev warm amplifier board with the modes 'measure' and
    'reference'. See base class for parameters.
    """
    def __init__(self, name='WAMP', default_mode='measure', **kw):
        super().__init__(name=name, default_mode=default_mode, **kw)
        # FIXME: bit patterns need to be checked for WA!!
        self.modes = {'reference': [1, 0],
                      'measure': [0, 1]}
