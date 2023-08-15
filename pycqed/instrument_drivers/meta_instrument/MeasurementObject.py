import logging
log = logging.getLogger(__name__)
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from pycqed.instrument_drivers.instrument import Instrument

from pycqed.measurement import sweep_functions as swf


class MeasurementObject(Instrument):
    _acq_weights_type_aliases = {}  # see self.get_acq_weights_type()
    _ro_pulse_type_vals = ['GaussFilteredCosIQPulse',
                           'GaussFilteredCosIQPulseMultiChromatic']
    _allowed_drive_modes = [None]

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        self.msmt_suffix = '_' + name  # used to append to measurement labels

        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available on '
                                     'the measurement object',
                           get_cmd=self._get_operations)

        self.add_parameter('instr_mc',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_acq',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_ro_lo',
            parameter_class=InstrumentRefParameter,
            vals=vals.MultiType(vals.Enum(None), vals.Strings()))
        self.add_parameter('instr_trigger',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_switch',
            parameter_class=InstrumentRefParameter,
            vals=vals.MultiType(vals.Enum(None), vals.Strings()))

        # acquisition parameters
        self.add_parameter('acq_unit', initial_value=0,
                           vals=vals.Enum(0, 1, 2, 3),
                           docstring='Acquisition device unit (only one for '
                                     'UHFQA and up to 4 for SHFQA).',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_I_channel', initial_value=0,
                           vals=vals.Ints(min_value=0),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_Q_channel', initial_value=1,
                           vals=vals.Ints(min_value=0),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_averages', initial_value=1024,
                           vals=vals.Ints(0, 1000000),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_shots', initial_value=4094,
                           docstring='Number of single shot measurements to do'
                                     'in single shot experiments.',
                           vals=vals.Ints(0, 1048576**2),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_length', initial_value=2.2e-6,
                           vals=vals.Numbers(min_value=1e-8,
                                             max_value=100e-6),
                           parameter_class=ManualParameter)
        awt_docstring = 'Determines what type of integration weights to ' +\
                        'use:\n\tSSB: Single sideband demodulation\n\tDSB: ' +\
                        'Double sideband demodulation\n\tcustom: waveforms ' +\
                        'specified in "acq_weights_I" and "acq_weights_Q"' +\
                        '\n\tcustom_2D: waveforms specified in ' +\
                        '"acq_weights_I/I2" and "acq_weights_Q/Q2"\n\t' +\
                        'square_rot: uses a single integration channel with ' +\
                        'boxcar weights\n\tmanual: keeps the weights ' +\
                        'already set in the acquisition device' +\
                        ''.join([f'\n\t{k}: alias for {v}' for k, v
                                 in self._acq_weights_type_aliases.items()])
        self.add_parameter('acq_weights_type', parameter_class=ManualParameter,
                           vals=vals.Enum(
                               'DSB', 'SSB', 'DSB2', 'square_rot', 'manual',
                               'custom', 'custom_2D',
                               *list(self._acq_weights_type_aliases)),
                           initial_value='SSB',
                           docstring=awt_docstring)
        self.add_parameter('acq_weights_I', vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_Q', vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_I2', vals=vals.Arrays(),
                           label='Optimized weights for second integration '
                                 'channel I',
                           docstring=("Used for double weighted integration "
                                      "during qutrit readout"),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_Q2', vals=vals.Arrays(),
                           label='Optimized weights for second integration '
                                 'channel Q',
                           docstring=("Used for double weighted integration "
                                      "during qutrit readout"),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_IQ_angle', initial_value=0,
                           docstring='The phase of the integration weights '
                                     'when using SSB, DSB or square_rot '
                                     'integration weights',
                                     label='Acquisition IQ angle', unit='rad',
                           parameter_class=ManualParameter)

        # readout pulse parameters
        self.add_parameter(
            'ro_fixed_lo_freq', unit='Hz',
            set_cmd=lambda f, s=self: s.configure_mod_freqs(
                'ro', ro_fixed_lo_freq=f),
            docstring='Fix the ro LO to a single frequency or to a set of '
                      'allowed frequencies. For allowed options, see the '
                      'argument fixed_lo in the docstring of '
                      'get_closest_lo_freq.')
        self.add_parameter(
            'ro_freq', unit='Hz',
            set_cmd=lambda f, s=self: s.configure_mod_freqs('ro', ro_freq=f),
            label='Readout frequency')
        self.add_parameter('ro_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout I channel')
        self.add_parameter('ro_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the readout Q channel')
        self.add_parameter('ro_lo_power', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Readout pulse upconversion mixer LO power')
        self.add_operation('RO')
        self.add_pulse_parameter('RO', 'ro_pulse_type', 'pulse_type',
                                 vals=vals.Enum(*self._ro_pulse_type_vals),
                                 initial_value='GaussFilteredCosIQPulse')
        self.add_pulse_parameter('RO', 'ro_I_channel', 'I_channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('RO', 'ro_Q_channel', 'Q_channel',
                                 initial_value=None, vals=vals.MultiType(
                                     vals.Enum(None), vals.Strings()))
        self.add_pulse_parameter('RO', 'ro_amp', 'amplitude',
                                 initial_value=0.001,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_length', 'pulse_length',
                                 initial_value=2e-6, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_delay', 'pulse_delay',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(
            'RO', 'ro_mod_freq', 'mod_frequency', initial_value=100e6,
            set_parser=lambda f, s=self: s.configure_mod_freqs('ro',
                                                               ro_mod_freq=f),
            vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_phase', 'phase',
                                 initial_value=0,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_phi_skew', 'phi_skew',
                                 initial_value=0,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_alpha', 'alpha',
                                 initial_value=1,
                                 vals=vals.MultiType(vals.Numbers(), vals.Lists()))
        self.add_pulse_parameter('RO', 'ro_sigma',
                                 'gaussian_filter_sigma',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_buffer_length_start', 'buffer_length_start',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_buffer_length_end', 'buffer_length_end',
                                 initial_value=10e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_phase_lock', 'phase_lock',
                                 initial_value=False, vals=vals.Bool())
        self.add_pulse_parameter(
            'RO', 'ro_trigger_channels', 'trigger_channels',
            vals=vals.MultiType(vals.Enum(None), vals.Strings(),
                                vals.Lists(vals.Strings())))
        self.add_pulse_parameter(
            'RO', 'ro_trigger_pars', 'trigger_pars',
            vals=vals.MultiType(vals.Enum(None), vals.Dict()))

        # switch parameters
        DEFAULT_SWITCH_MODES = OrderedDict({'default': {}})
        self.add_parameter(
            'switch_modes', parameter_class=ManualParameter,
            initial_value=DEFAULT_SWITCH_MODES, vals=vals.Dict(),
            docstring=
            "A dictionary whose keys are identifiers of switch modes and "
            "whose values are dicts understood by the set_switch method of "
            "the SwitchControls instrument specified in the parameter "
            "instr_switch.")

    def _get_operations(self):
        return self._operations

    def add_operation(self, operation_name):
        self._operations[operation_name] = {}

    def add_pulse_parameter(self,
                            operation_name,
                            parameter_name,
                            argument_name,
                            initial_value=None,
                            vals=vals.Numbers(),
                            **kwargs):
        """
        Add a pulse parameter to the measurement object.

        Args:
            operation_name (str): The operation of which this parameter is an
                argument. e.g. mw_control or CZ
            parameter_name (str): Name of the parameter
            argument_name  (str): Name of the arugment as used in the sequencer
            **kwargs get passed to the add_parameter function
        Raises:
            KeyError: if this instrument already has a parameter with this
                name.
        """
        if parameter_name in self.parameters:
            raise KeyError(
                'Duplicate parameter name {}'.format(parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError(
                'Unknown operation {}, add '.format(operation_name) +
                'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           vals=vals,
                           parameter_class=ManualParameter, **kwargs)

        # for use in RemoteInstruments to add parameters to the server
        # we return the info they need to construct their proxy
        return

    def get_acq_weights_type(self):
        """Returns the value of acq_weights_type after resolving alias names

        Alias names are specified in self._acq_weights_type_aliases.
        """
        wt = self.acq_weights_type()
        return self._acq_weights_type_aliases.get(wt, wt)

    def get_acq_int_channels(self, n_channels=None):
        """Get a list of tuples with the integration channels.

        Args:
            n_channels (int): number of integration channels; if this is None,
                it will be chosen as follows (after resolving aliases according
                to self._acq_weights_type_aliases):
                - 2 for acq_weights_type in ['SSB', 'DSB', 'DSB2',
                    'custom_2D', 'manual']
                - 1 otherwise (in particular for acq_weights_type in
                    ['custom', 'square_rot'])

        Returns
            list with n_channels tuples, where the first entry in each tuple is
            the acq_unit and the second is an integration channel index
        """
        if n_channels is None:
            n_channels = 2 if (self.get_acq_weights_type() in [
                'SSB', 'DSB', 'DSB2', 'custom_2D', 'manual']
                               and self.acq_Q_channel() is not None) else 1
        return [(self.acq_unit(), self.acq_I_channel()),
                (self.acq_unit(), self.acq_Q_channel())][:n_channels]

    def get_acq_inp_channels(self):
        """Get a list of tuples with the acquisition input channels.

        For now, this method assumes that all quadratures available on the
        acquisition unit should be recorded, i.e., two for devices that
        provide I&Q signals, and one otherwise.

        TODO: In the future, a parameter could be added to the measurement
        object to allow recording only one out of two available quadratures.

        Returns
            list of tuples, where the first entry in each tuple is
            the acq_unit and the second is an input channel index
        """
        n_channels = self.instr_acq.get_instr().n_acq_inp_channels
        return [(self.acq_unit(), i) for i in range(n_channels)]

    def get_ro_lo_freq(self):
        """Returns the required local oscillator frequency for readout pulses

        The RO LO freq is calculated from self.ro_mod_freq (intermediate
        frequency) and self.ro_freq.
        """
        # in case of multichromatic readout, take first ro freq, else just
        # wrap the frequency in a list and take the first
        if np.ndim(self.ro_freq()) == 0:
            ro_freq = [self.ro_freq()]
        else:
            ro_freq = self.ro_freq()
        if np.ndim(self.ro_mod_freq()) == 0:
            ro_mod_freq = [self.ro_mod_freq()]
        else:
            ro_mod_freq = self.ro_mod_freq()
        return ro_freq[0] - ro_mod_freq[0]

    def prepare(self, drive=None, switch='modulated'):
        """Prepare instruments for a measurement involving this measurement
        object.

        The preparation includes:
        - configure readout local oscillators
        - call set_readout_weights
        - set switches to the mode required for the measurement

        Args:
            drive (str, None): the kind of drive to be applied (not
                implemented here, to be extended by child classes)
            switch (str): the required switch mode, see the docstring of
            switch_modes
        """
        if drive not in self._allowed_drive_modes:
            raise NotImplementedError(f"Drive mode {drive} not implemented!")

        self.configure_mod_freqs()
        ro_lo = self.instr_ro_lo

        # configure readout local oscillators
        ro_lo_freq = self.get_ro_lo_freq()

        if ro_lo() is not None:  # configure external LO
            if self.ro_Q_channel() is not None:
                # We are on a setup that generates RO pulses by upconverting
                # IQ signals with a continuously running LO, so we switch off
                # gating of the MWG.
                ro_lo.get_instr().pulsemod_state('Off')
            ro_lo.get_instr().power(self.ro_lo_power())
            ro_lo.get_instr().frequency(ro_lo_freq)
            ro_lo.get_instr().on()
        # Provide the ro_lo_freq to the acquisition device to allow
        # configuring an internal LO if needed.
        self.instr_acq.get_instr().set_lo_freq(self.acq_unit(), ro_lo_freq)

        # other preparations
        self.set_readout_weights()
        # set switches to the mode required for the measurement
        self.set_switch(switch)

    def _get_custom_readout_weights(self):
        return dict(
            weights_I=[self.acq_weights_I(), self.acq_weights_I2()],
            weights_Q=[self.acq_weights_Q(), self.acq_weights_Q2()],
        )

    def set_readout_weights(self, weights_type=None, f_mod=None):
        """Set acquisition weights for this measurement object in the
        acquisition device.

        Depending on the weights type, some of the following qcodes
        parameters can have an influence on the programmed weigths (see the
        docstrings of these parameters and of
        AcquisitionDevice._acquisition_generate_weights):
        - instr_acq, acq_unit, acq_I_channel, acq_Q_channel
        - acq_weights_type (if not overridden with the arg weights_type)
        - ro_mod_freq (if not overridden with the arg f_mod)
        - acq_IQ_angle
        - acq_weights_I, acq_weights_I2, acq_weights_Q, acq_weights_Q2

        Args:
            weights_type (str, None): a weights_type understood by
                AcquisitionDevice._acquisition_generate_weights, or the
                default None, in which case the qcodes parameter
                acq_weights_type is used.
            f_mod (float, None): The intermediate frequency of the signal to
                be acquired, or the default None, in which case the qcodes
                parameter ro_mod_freq is used.
        """
        if weights_type is None:
            weights_type = self.get_acq_weights_type()
        if f_mod is None:
            f_mod = self.ro_mod_freq()
        self.instr_acq.get_instr().acquisition_set_weights(
            channels=self.get_acq_int_channels(n_channels=2),
            weights_type=weights_type, mod_freq=f_mod,
            acq_IQ_angle=self.acq_IQ_angle(),
            acq_length=self.acq_length(),
            **self._get_custom_readout_weights()
        )

    def set_switch(self, switch_mode=None):
        """
        Sets the switch control (given in the qcodes parameter instr_switch)
        to the given mode.

        :param switch_mode: (str) the name of a switch mode that is defined in
            the qcodes parameter self.switch_modes. If None, uses the first
            entry of self.switch_modes.
            See the docstring of switch_modes for more details.
        """
        if self.instr_switch() is None:
            return
        if switch_mode is None:
            switch_mode = list(self.switch_modes())[0]
        switch = self.instr_switch.get_instr()
        mode = self.switch_modes().get(switch_mode, None)
        if mode is None:
            log.warning(f'Switch mode {switch_mode} not configured for '
                        f'{self.name}.')
        else:
            switch.set_switch(mode)

    def get_operation_dict(self, operation_dict=None):
        self.configure_mod_freqs()
        if operation_dict is None:
            operation_dict = {}

        for op_name, op in self.operations().items():
            operation_dict[op_name + ' ' + self.name] = {}
            for argument_name, parameter_name in op.items():
                operation_dict[op_name + ' ' + self.name][argument_name] = \
                    self.get(parameter_name)

        operation_dict['RO ' + self.name]['operation_type'] = 'RO'
        operation_dict['Acq ' + self.name] = deepcopy(
            operation_dict['RO ' + self.name])
        operation_dict['Acq ' + self.name]['amplitude'] = 0

        if np.ndim(self.ro_freq()) != 0:
            delta_freqs = np.diff(self.ro_freq(), prepend=self.ro_freq()[0])
            mods = [self.ro_mod_freq() + d for d in delta_freqs]
            operation_dict['RO ' + self.name]['mod_frequency'] = mods

        for code, op in operation_dict.items():
            op['op_code'] = code
        return operation_dict

    def swf_ro_freq_lo(self, bare=False):
        """Create a sweep function for sweeping the readout frequency.

        The sweep is implemented as an LO sweep in case of an acquisition
        device with an external LO. The implementation depends on the
        get_lo_sweep_function method of the acquisition device in case of an
        internal LO (note that it might be an IF sweep or a combined LO and
        IF sweep in that case.)

        Args:
            bare (bool): return the bare LO freq swf without any automatic
                offsets applied. Defaults to False.

        Returns: the Sweep_function object
        """
        if self.instr_ro_lo() is not None:  # external LO
            if bare:
                return swf.mc_parameter_wrapper.wrap_par_to_swf(
                    self.instr_ro_lo.get_instr().frequency)
            else:
                return swf.Offset_Sweep(
                    self.instr_ro_lo.get_instr().frequency,
                    -self.ro_mod_freq(),
                    name='Readout frequency',
                    parameter_name='Readout frequency')
        else:  # no external LO
            return self.instr_acq.get_instr().get_lo_sweep_function(
                self.acq_unit(),
                0 if (self.ro_fixed_lo_freq() or bare) else self.ro_mod_freq(),
                get_closest_lo_freq=(lambda f, s=self:
                                     s.get_closest_lo_freq(f, operation='ro')))

    def swf_ro_mod_freq(self):
        return swf.Offset_Sweep(
            self.ro_mod_freq,
            self.instr_ro_lo.get_instr().frequency(),
            name='Readout frequency',
            parameter_name='Readout frequency')

    def get_closest_lo_freq(self, target_lo_freq, fixed_lo='default',
                            operation=None):
        """Get the closest allowed LO freq for given target LO freq.

        Args:
            target_lo_freq (float): the target Lo freq
            fixed_lo: specification of the allowed LO freq(s), can be:
                - None: no restrictions on the LO freq
                - float: LO fixed to a single freq
                - str: (operation must be provided in this case)
                    - 'default' (default value): use the setting in the
                        measurement object
                    - a qb name to indicated that the LO must be fixed to be
                      the same as for that qb.
                - dict with (a subset of) the following keys:
                    'min' and/or 'max': minimal/maximal allowed LO freq
                    'step': LO fixed to a grid with this step width (grid
                            starting at 'min' if provided and at 0 otherwise)
                - list, np.array: LO fixed to be one of the listed values
            operation (str): the operation for which the LO freq is to be
                determined (e.g., 'ge', 'ro'). Only needed if fixed_lo is a str.

        Returns:
            The allowed LO freq that most closely matches the target
            combination of RF and IF.

        Examples:
            >>> freq, mod_freq = 5898765432, 150e6
            >>> target_lo_freq = freq - mod_freq
            >>> qb.get_closest_lo_freq(target_lo_freq, 'qb1', 'ge')
            >>> qb.get_closest_lo_freq(target_lo_freq, 5.8e9)
            >>> qb.get_closest_lo_freq(
            >>>     target_lo_freq, np.arange(4e9, 6e9 + 1e6, 1e6))
            >>> qb.get_closest_lo_freq(target_lo_freq, {'step': 100e6})
            >>> qb.get_closest_lo_freq(
            >>>     target_lo_freq, {'min': 5.4e9, 'max': 5.6e9})
            >>> qb.get_closest_lo_freq(
            >>>     target_lo_freq, {'min': 6.3e9, 'max': 6.9e9})
            >>> qb.get_closest_lo_freq(
            >>>     target_lo_freq, {'min': 5.4e9, 'max': 6.9e9, 'step': 10e6})
        """
        if fixed_lo == 'default':
            fixed_lo = self.get(f'{operation}_fixed_lo_freq')
        if fixed_lo is None:
            return target_lo_freq
        elif isinstance(fixed_lo, float):
            return fixed_lo
        elif isinstance(fixed_lo, str):
            instr = self.find_instrument(fixed_lo)
            return getattr(instr, f'get_{operation}_lo_freq')()
        elif isinstance(fixed_lo, dict):
            f_min = fixed_lo.get('min', 0)
            f_max = fixed_lo.get('max', np.inf)
            step = fixed_lo.get('step', None)
            lo_freq = max(min(target_lo_freq, f_max) - f_min, 0)
            if step is not None:
                lo_freq = round(lo_freq / step) * step
                if lo_freq > f_max:
                    lo_freq -= step
            lo_freq += f_min
            return lo_freq
        else:
            ind = np.argmin(np.abs(np.array(fixed_lo) - (target_lo_freq)))
            return fixed_lo[ind]

    def configure_mod_freqs(self, operation=None, **kw):
        """Configure modulation freqs (IF) to be compatible with fixed LO freqs

        If {op}_fixed_lo_freq is not None for the operation {op},
        {op}_mod_freq will be updated to {op}_freq' - {op}_fixed_lo_freq.
        The method can be called with kw (see below) as a set_cmd when a
        relevant paramter changes, or without kw as a sanity check, in which
        case it shows a warning when updating an IF.

        Args:
            operation (str, None): configure the IF only for the operation
                indicated by the string or for all operations for which a
                fixed LO freq is configured.
            **kw: If a kew equals the name of a qcodes parameter of the qb,
                the corresponding value supersedes the parameter value.

        Returns:
            - The new IF if called with arguments operation and
              {operation}_mod_freq (can be used as set_parser).
            - None otherwise.
        """
        def get_param(param):
            if param in kw:
                return kw[param]
            else:
                return self.get(param)

        fixed_lo_suffix = '_fixed_lo_freq'
        if operation is None:
            ops = [k[:-len(fixed_lo_suffix)] for k in self.parameters
                   if k.endswith(fixed_lo_suffix)]
        else:
            ops = [operation]

        for op in ops:
            fixed_lo = get_param(f'{op}{fixed_lo_suffix}')
            if fixed_lo is None:
                if operation is not None and f'{op}_mod_freq' in kw:
                    # called for IF change of single op: behave as set_parser
                    return kw[f'{op}_mod_freq']
            else:
                freq = get_param(f'{op}_freq')
                old_mod_freq = get_param(f'{op}_mod_freq')
                if np.ndim(old_mod_freq):
                    raise NotImplementedError(
                        f'{op}: Fixed LO freq in combination with '
                        f'multichromatic mod freq is not implemented.')
                if freq is None:  # freq not yet set
                    mod_freq = old_mod_freq  # no need to update the mod freq
                else:
                    lo_freq = self.get_closest_lo_freq(
                        freq - old_mod_freq, fixed_lo, operation=op)
                    mod_freq = get_param(f'{op}_freq') - lo_freq
                if operation is not None and f'{op}_mod_freq' in kw:
                    # called for IF change of single op: behave as set_parser
                    return mod_freq
                elif old_mod_freq != mod_freq:
                    if not any([k.startswith(f'{op}_') and k != f'{op}_mod_freq'
                            for k in kw]):
                        log.warning(
                            f'{self.name}: {op}_mod_freq {old_mod_freq} is not '
                            f'consistent '
                            f'with the fixed LO freq {fixed_lo} and will be '
                            f'adjusted to {mod_freq}.')
                    self.parameters[f'{op}_mod_freq'].cache._set_from_raw_value(
                        mod_freq)

    def configure_pulsar(self):
        """
        Configure MeasurementObject-specific settings in pulsar:
        - Set AWG channel DC offsets and switch sigouts on,
        see configure_offsets
        """
        # set offsets and turn on AWG outputs
        self.configure_offsets()

    def configure_offsets(self, set_ro_offsets=True, offset_list=None):
        """
        Set AWG channel DC offsets and switch sigouts on.

        :param set_ro_offsets: whether to set offsets for RO channels
        :param offset_list: additional offsets to set
        """
        pulsar = self.instr_pulsar.get_instr()
        if offset_list is None:
            offset_list = []

        if set_ro_offsets:
            offset_list += [('ro_I_channel', 'ro_I_offset'),
                            ('ro_Q_channel', 'ro_Q_offset')]

        for channel_par, offset_par in offset_list:
            ch = self.get(channel_par)
            if ch is not None and ch + '_offset' in pulsar.parameters:
                pulsar.set(ch + '_offset', self.get(offset_par))
                pulsar.sigout_on(ch)

    def link_param_to_operation(self, operation_name, parameter_name,
                                argument_name):
        """
        Links an existing param to an operation for use in the operation dict.

        An example of where to use this would be the flux_channel.
        Only one parameter is specified but it is relevant for multiple flux
        pulses. You don't want a different parameter that specifies the channel
        for the iSWAP and the CZ gate. This can be solved by linking them to
        your operation.

        Args:
            operation_name (str): The operation of which this parameter is an
                argument. e.g. mw_control or CZ
            parameter_name (str): Name of the parameter
            argument_name  (str): Name of the arugment as used in the sequencer
            **kwargs get passed to the add_parameter function
        """
        if parameter_name not in self.parameters:
            raise KeyError('Parameter {} needs to be added first'.format(
                parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')
