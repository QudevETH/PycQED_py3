import logging
log = logging.getLogger(__name__)
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy

from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from qcodes.instrument.base import Instrument

from pycqed.analysis_v2.readout_analysis import Singleshot_Readout_Analysis_Qutrit
from pycqed.measurement import detector_functions as det
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import awg_sweep_functions_multi_qubit as awg_swf2
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.utilities.general import temporary_value
from pycqed.utilities.math import vp_to_dbm, dbm_to_vp
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement import optimization as opti
from pycqed.measurement import mc_parameter_wrapper
import pycqed.analysis_v2.spectroscopy_analysis as sa
from pycqed.utilities import math
import pycqed.analysis.fitting_models as fit_mods
import os
import \
    pycqed.measurement.waveform_control.fluxpulse_predistortion as fl_predist

try:
    import pycqed.simulations.readout_mode_simulations_for_CLEAR_pulse \
        as sim_CLEAR
except ModuleNotFoundError:
    log.warning('"readout_mode_simulations_for_CLEAR_pulse" not imported.')


class MeasurementObject(Instrument):
    # FIXME future remove stuff from Qubit object

    # from Qubit object
    def _get_operations(self):
        return self._operations

    # from Qubit object
    def add_operation(self, operation_name):
        self._operations[operation_name] = {}

    # from Qubit object
    def add_pulse_parameter(self,
                            operation_name,
                            parameter_name,
                            argument_name,
                            initial_value=None,
                            vals=vals.Numbers(),
                            **kwargs):
        """
        Add a pulse parameter to the qubit.

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
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           vals=vals,
                           parameter_class=ManualParameter, **kwargs)

        # for use in RemoteInstruments to add parameters to the server
        # we return the info they need to construct their proxy
        return

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # from Qubit object
        self.msmt_suffix = '_' + name  # used to append to measurement labels

        # from Qubit object
        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available on the qubit',
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
                           vals=vals.Ints(0, 1048576),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_length', initial_value=2.2e-6,
                           vals=vals.Numbers(min_value=1e-8,
                                             max_value=100e-6),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_type', parameter_class=ManualParameter,
                           vals=vals.Enum('DSB', 'SSB'), initial_value='SSB')
        self.add_parameter('acq_IQ_angle', initial_value=0,
                           docstring='The phase of the integration weights '
                                     'when using SSB, DSB or square_rot '
                                     'integration weights',
                                     label='Acquisition IQ angle', unit='rad',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_I', vals=vals.Arrays(),
                           label='Optimized weights for I channel',
                           parameter_class=ManualParameter)
        self.add_parameter('acq_weights_Q', vals=vals.Arrays(),
                           label='Optimized weights for Q channel',
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
                                 vals=vals.Enum('GaussFilteredCosIQPulse',
                                                'GaussFilteredCosIQPulseMultiChromatic',
                                                'GaussFilteredCosIQPulseWithFlux'),
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
        DEFAULT_SWITCH_MODES = {'modulated': {}, 'spec': {}, 'calib': {}}
        self.add_parameter(
            'switch_modes', parameter_class=ManualParameter,
            initial_value=DEFAULT_SWITCH_MODES, vals=vals.Dict(),
            docstring=
            "A dictionary whose keys are identifiers of switch modes and "
            "whose values are dicts understood by the set_switch method of "
            "the SwitchControls instrument specified in the parameter "
            "instr_switch. The keys must include 'modulated' (for routing "
            "the upconverted IF signal to the experiment output of the "
            "upconversion board, used for all experiments that do not "
            "specify a different mode), 'spec' (for routing the LO input to "
            "the experiment output of the upconversion board, used for "
            "qubit spectroscopy), and 'calib' (for routing the upconverted "
            "IF signal to the calibration output of upconversion board, "
            "used for mixer calibration). The keys can include 'no_drive' "
            "(to replace the 'modulated' setting in case of measurements "
            "without drive signal, i.e., when calling qb.prepare with "
            "drive=None) as well as additional custom modes (to be used in "
            "manual calls to set_switch).")

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def get_acq_int_channels(self, n_channels=None):
        """Get a list of tuples with the qubit's integration channels.

        Args:
            n_channels (int): number of integration channels; if this is None,
                it will be chosen as follows:
                2 for ro_weights_type in ['SSB', 'DSB', 'DSB2',
                    'optimal_qutrit', 'manual']
                1 otherwise (in particular for ro_weights_type in
                    ['optimal', 'square_rot'])

        Returns
            list with n_channels tuples, where the first entry in each tuple is
            the acq_unit and the second is an integration channel index
        """
        if n_channels is None:
            n_channels = 2 if (self.acq_weights_type() in [
                'SSB', 'DSB', 'DSB2', 'optimal_qutrit', 'manual']
                               and self.acq_Q_channel() is not None) else 1
        return [(self.acq_unit(), self.acq_I_channel()),
                (self.acq_unit(), self.acq_Q_channel())][:n_channels]

    def get_acq_inp_channels(self):
        """Get a list of tuples with the qubit's acquisition input channels.

        For now, this method assumes that all quadratures available on the
        acquisition unit should be recorded, i.e., two for devices that
        provide I&Q signals, and one otherwise.

        TODO: In the future, a parameter could be added to the qubit object
            to allow recording only one out of two available quadratures.

        Returns
            list of tuples, where the first entry in each tuple is
            the acq_unit and the second is an input channel index
        """
        n_channels = self.instr_acq.get_instr().n_acq_inp_channels
        return [(self.acq_unit(), i) for i in range(n_channels)]

    def get_ro_lo_freq(self):
        """Returns the required local oscillator frequency for readout pulses

        The RO LO freq is calculated from the ro_mod_freq (intermediate
        frequency) and the ro_freq stored in the qubit object.
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

    def prepare(self, drive='timedomain', switch='default'):
        """Prepare instruments for a measurement involving this qubit.

        The preparation includes:
        - call configure_offsets
        - configure readout local oscillators
        - configure qubit drive local oscillator
        - call update_detector_functions
        - call set_readout_weights
        - set switches to the mode required for the measurement

        Args:
            drive (str, None): the kind of drive to be applied, which can be
                None (no drive), 'continuous_spec' (continuous spectroscopy),
                'continuous_spec_modulated' (continuous spectroscopy using
                the modulated configuation of the switch),
                'pulsed_spec' (pulsed spectroscopy), or the default
                'timedomain' (AWG-generated signal upconverted by the mixer)
            switch (str): the required switch mode. Can be a switch mode
                understood by set_switch or the default value 'default', in
                which case the switch mode is determined based on the kind
                of drive ('spec' for continuous/pulsed spectroscopy w/o modulated;
                'no_drive' if drive is None and a switch mode 'no_drive' is
                configured for this qubit; 'modulated' in all other cases).
        """
        self.configure_mod_freqs()
        ro_lo = self.instr_ro_lo

        # configure readout local oscillators
        ro_lo_freq = self.get_ro_lo_freq()

        if ro_lo() is not None:  # configure external LO
            ro_lo.get_instr().pulsemod_state('Off')
            ro_lo.get_instr().power(self.ro_lo_power())
            ro_lo.get_instr().frequency(ro_lo_freq)
            ro_lo.get_instr().on()
        # Provide the ro_lo_freq to the acquisition device to allow
        # configuring an internal LO if needed.
        self.instr_acq.get_instr().set_lo_freq(self.acq_unit(), ro_lo_freq)

        # other preparations
        # set switches to the mode required for the measurement
        # See the docstring of switch_modes for an explanation of the
        # following modes.
        if switch == 'default':
            if drive is None and 'no_drive' in self.switch_modes():
                # use special mode for measurements without drive if that
                # mode is defined
                self.set_switch('no_drive')
            else:
                # use 'spec' for qubit spectroscopy measurements
                # (continuous_spec and pulsed_spec) and 'modulated' otherwise
                self.set_switch(
                    'spec' if drive is not None and drive.endswith('_spec')
                    else 'modulated')
        else:
            # switch mode was explicitly provided by the caller (e.g.,
            # for mixer calib)
            self.set_switch(switch)

    def set_switch(self, switch_mode='modulated'):
        """
        Sets the switch control (given in the qcodes parameter instr_switch)
        to the given mode.

        :param switch_mode: (str) the name of a switch mode that is defined in
            the qcodes parameter switch_modes of this qubit (default:
            'modulated'). See the docstring of switch_modes for more details.
        """
        if self.instr_switch() is None:
            return
        switch = self.instr_switch.get_instr()
        mode = self.switch_modes().get(switch_mode, None)
        if mode is None:
            log.warning(f'Switch mode {switch_mode} not configured for '
                        f'{self.name}.')
        switch.set_switch(mode)

    def swf_ro_freq_lo(self):
        """Create a sweep function for sweeping the readout frequency.

        The sweep is implemented as an LO sweep in case of an acquisition
        device with an external LO. The implementation depends on the
        get_lo_sweep_function method of the acquisition device in case of an
        internal LO (note that it might be an IF sweep or a combined LO and
        IF sweep in that case.)

        Returns: the Sweep_function object
        """
        if self.instr_ro_lo() is not None:  # external LO
            return swf.Offset_Sweep(
                self.instr_ro_lo.get_instr().frequency,
                -self.ro_mod_freq(),
                name='Readout frequency',
                parameter_name='Readout frequency')
        else:  # no external LO
            return self.instr_acq.get_instr().get_lo_sweep_function(
                self.acq_unit(), self.ro_mod_freq())

    def swf_ro_mod_freq(self):
        return swf.Offset_Sweep(
            self.ro_mod_freq,
            self.instr_ro_lo.get_instr().frequency(),
            name='Readout frequency',
            parameter_name='Readout frequency')

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
        Configure qubit-specific settings in pulsar:
        - Reset modulation frequency and amplitude scaling
        - set AWG channel DC offsets and switch sigouts on,
           see configure_offsets
        - set flux distortion, see set_distortion_in_pulsar
        """
        pulsar = self.instr_pulsar.get_instr()
        # make sure that some settings are reset to their default values
        for quad in ['I', 'Q']:
            ch = self.get(f'ge_{quad}_channel')
            if f'{ch}_mod_freq' in pulsar.parameters:
                pulsar.parameters[f'{ch}_mod_freq'](None)
            if f'{ch}_amplitude_scaling' in pulsar.parameters:
                pulsar.parameters[f'{ch}_amplitude_scaling'](1)
        # set offsets and turn on AWG outputs
        self.configure_offsets()
        # set flux distortion
        self.set_distortion_in_pulsar()
