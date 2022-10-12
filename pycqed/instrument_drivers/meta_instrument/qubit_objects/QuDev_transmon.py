import logging
log = logging.getLogger(__name__)
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from copy import deepcopy

from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals

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


class QuDev_transmon(Qubit):
    DEFAULT_FLUX_DISTORTION = dict(
        IIR_filter_list=[],
        FIR_filter_list=[],
        scale_IIR=1,
        distortion='off',
        charge_buildup_compensation=True,
        compensation_pulse_delay=100e-9,
        compensation_pulse_gaussian_filter_sigma=0,
    )

    def __init__(self, name, transition_names=('ge', 'ef'), **kw):
        super().__init__(name, **kw)

        self.transition_names = transition_names

        self.add_parameter('instr_mc',
            parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_ge_lo',
            parameter_class=InstrumentRefParameter,
            vals=vals.MultiType(vals.Enum(None), vals.Strings()))
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

        # device parameters for user only
        # could be cleaned up
        self.add_parameter('f_RO_resonator', label='RO resonator frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('f_RO_purcell', label='RO purcell filter frequency',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_purcell_kappa', label='Purcell filter kappa',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('RO_J_coupling', label='J coupling of RO resonator'
                                                  'and purcell filter',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('Q_RO_resonator', label='RO resonator Q factor',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('ssro_contrast', unit='arb.', initial_value=0,
                           label='integrated g-e trace contrast',
                           parameter_class=ManualParameter)
        self.add_parameter('optimal_acquisition_delay', label='Optimal '
                           'acquisition delay', unit='s', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('anharmonicity', label='Qubit anharmonicity',
                           unit='Hz', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('dynamic_phase', label='CZ dynamic phase',
                           unit='deg', initial_value=0,
                           parameter_class=ManualParameter)
        self.add_parameter('EC_qubit', label='Qubit EC', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('EJ_qubit', label='Qubit EJ', unit='Hz',
                           initial_value=0, parameter_class=ManualParameter)
        self.add_parameter('chi', unit='Hz', parameter_class=ManualParameter,
                           label='Chi')

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
        self.add_pulse_parameter('RO', 'ro_flux_channel', 'flux_channel',
                                 initial_value=None, vals=vals.MultiType(
                                     vals.Enum(None), vals.Strings()))
        self.add_pulse_parameter('RO',
                                 'ro_flux_crosstalk_cancellation_key',
                                 'crosstalk_cancellation_key',
                                 vals=vals.Anything(),
                                 initial_value=False)
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
        self.add_pulse_parameter('RO', 'ro_basis_rotation',
                                 'basis_rotation', initial_value={},
                                 docstring='Dynamic phase acquired by other '
                                           'qubits due to a measurement tone on'
                                           ' this qubit.',
                                 label='RO pulse basis rotation dictionary',
                                 vals=vals.Dict())
        self.add_pulse_parameter(
            'RO', 'ro_disable_repeat_pattern', 'disable_repeat_pattern',
            initial_value=False, vals=vals.Bool(),
            docstring='True means that repeat patterns are not used for '
                      'readout pulses of this qubit even if higher layers '
                      '(like CircuitBuilder) configure a repeat pattern.')
        self.add_pulse_parameter(
            'RO', 'ro_trigger_channels', 'trigger_channels',
            vals=vals.MultiType(vals.Enum(None), vals.Strings(),
                                vals.Lists(vals.Strings())))
        self.add_pulse_parameter(
            'RO', 'ro_trigger_pars', 'trigger_pars',
            vals=vals.MultiType(vals.Enum(None), vals.Dict()))
        self.add_pulse_parameter('RO', 'ro_flux_amplitude', 'flux_amplitude',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_extend_start', 'flux_extend_start',
                                 initial_value=20e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_extend_end', 'flux_extend_end',
                                 initial_value=150e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_gaussian_filter_sigma', 'flux_gaussian_filter_sigma',
                                 initial_value=0.5e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_mirror_pattern',
                                 'mirror_pattern',
                                 initial_value=None, vals=vals.Enum(None,
                                                                    "none",
                                                                    "all",
                                                                    "odd", "even"))


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
        self.add_parameter('acq_weights_type', initial_value='SSB',
                           vals=vals.Enum('SSB', 'DSB', 'DSB2', 'optimal',
                                          'square_rot', 'manual',
                                          'optimal_qutrit'),
                           docstring=(
                               'Determines what type of integration weights to '
                               'use: \n\tSSB: Single sideband demodulation\n\t'
                               'DSB: Double sideband demodulation\n\toptimal: '
                               'waveforms specified in "ro_acq_weight_func_I" '
                               'and "ro_acq_weight_func_Q"\n\tsquare_rot: uses '
                               'a single integration channel with boxcar '
                               'weights'),
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
        self.add_parameter('acq_weights_basis', vals=vals.Lists(),
                           label="weight basis used",
                           docstring=("Used to log the weights basis for "
                                      "integration during qutrit readout. E.g."
                                      " ['ge', 'gf'] or ['ge', 'ortho']."),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_classifier_params', vals=vals.Dict(),
                           initial_value={},
                           label='Parameters for the qutrit classifier.',
                           docstring=("Used in the int_avg_classif_det to "
                                      "classify single shots into g, e, f."),
                           parameter_class=ManualParameter)
        self.add_parameter('acq_state_prob_mtx', vals=vals.Arrays(),
                           label='SSRO correction matrix.',
                           docstring=("Matrix of measured vs prepared qubit "
                                      "states."),
                           parameter_class=ManualParameter)

        # qubit drive pulse parameters
        self.add_parameter('ge_lo_power', unit='dBm',
                           parameter_class=ManualParameter,
                           label='Qubit drive pulse mixer LO power')
        self.add_parameter('ge_I_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line I channel')
        self.add_parameter('ge_Q_offset', unit='V', initial_value=0,
                           parameter_class=ManualParameter,
                           label='DC offset for the drive line Q channel')
        # qubit ge frequency fit parameters
        self.add_parameter('fit_ge_freq_from_flux_pulse_amp',
                           label='Parameters for frequency vs flux pulse '
                                 'amplitude fit',
                           initial_value={}, parameter_class=ManualParameter)
        self.add_parameter('fit_ge_freq_from_dc_offset',
                           label='Parameters for frequency vs flux dc '
                                 'offset fit',
                           initial_value={}, parameter_class=ManualParameter)
        self.add_parameter(
            'fit_ge_amp180_over_ge_freq',
            docstring='String representation of a function to calculate a pi '
                      'pulse amplitude for a given ge transition frequency. '
                      'Alternatively, a list of two arrays to perform an '
                      'interpolation, where the first array contains ge '
                      'frequencies and the second one contains the '
                      'corresponding pi pulse amplitude.',
            initial_value=None, parameter_class=ManualParameter)
        self.add_parameter('fit_ro_freq_over_ge_freq',
                           label='String representation of function to '
                                 'calculate a RO frequency for a given '
                                 'ge transition frequency.',
                           initial_value=None, parameter_class=ManualParameter)
        self.add_parameter('flux_amplitude_bias_ratio',
                           label='Ratio between a flux pulse amplitude '
                                 'and a DC offset change that lead to '
                                 'the same change in flux.',
                           initial_value=None, vals=vals.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('amp_scaling_correction_coeffs',
                           initial_value=[0, 0],
                           parameter_class=ManualParameter,
                           docstring='List/array of two floats representing '
                                     'the coefficients a, b of the 5th order '
                                     'polynomial used to correct for drive '
                                     'electronics nonlinearity when scaling '
                                     'the drive pulse amplitude with respect '
                                     'to the pi-pulse amplitude. Used in '
                                     'calculate_nonlinearity_correction.',
                           vals=vals.MultiType(vals.Lists(), vals.Arrays()))

        # add drive pulse parameters
        for tr_name in self.transition_names:
            if tr_name == 'ge':
                self.add_parameter(
                    f'{tr_name}_fixed_lo_freq', unit='Hz',
                    set_cmd=lambda f, s=self, t=tr_name: s.configure_mod_freqs(
                        t, **{f'{t}_fixed_lo_freq': f}),
                    docstring=f'Fix the {tr_name} LO to a single frequency or '
                              f'to a set of allowed frequencies. For allowed '
                              f'options, see the argument fixed_lo in the '
                              f'docstring of get_closest_lo_freq.')
                freq_kw = dict(set_cmd=lambda f, s=self, t=tr_name:
                               s.configure_mod_freqs(t, **{f'{t}_freq': f}))
            else:
                freq_kw = dict(parameter_class=ManualParameter)
            self.add_parameter(f'{tr_name}_freq',
                               label=f'Qubit {tr_name} drive frequency',
                               unit='Hz', initial_value=0,
                               **freq_kw)
            tn = '' if tr_name == 'ge' else f'_{tr_name}'
            self.add_operation(f'X180{tn}')
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_pulse_type',
                                     'pulse_type',
                                     initial_value='SSB_DRAG_pulse',
                                     vals=vals.Enum(
                                         'SSB_DRAG_pulse',
                                         'SSB_DRAG_pulse_with_cancellation'
                                     ))
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_amp180',
                                     'amplitude',
                                     initial_value=0.001, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_amp90_scale',
                                     'amp90_scale',
                                     initial_value=0.5, vals=vals.Numbers(0, 1))
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_delay',
                                     'pulse_delay',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_sigma',
                                     'sigma',
                                     initial_value=10e-9, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_nr_sigma',
                                     'nr_sigma',
                                     initial_value=5, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_motzoi',
                                     'motzoi',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_X_phase',
                                     'phase',
                                     initial_value=0, vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}',
                                     f'{tr_name}_cancellation_params',
                                     'cancellation_params', initial_value={},
                                     vals=vals.Dict())
            if tr_name == 'ge':
                # The parameters below will be the same for all transitions
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_I_channel',
                                         'I_channel',
                                         initial_value=None,
                                         vals=vals.Strings())
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_Q_channel',
                                         'Q_channel',
                                         initial_value=None,
                                         vals=vals.Strings())
                self.add_pulse_parameter(
                    f'X180{tn}', f'{tr_name}_mod_freq',
                    'mod_frequency', initial_value=-100e6,
                    set_parser=lambda f, s=self, t=tr_name:
                               s.configure_mod_freqs(t, **{f'{t}_mod_freq': f}),
                    vals=vals.Numbers())
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_phi_skew',
                                         'phi_skew',
                                         initial_value=0,
                                         vals=vals.Numbers())
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_alpha',
                                         'alpha',
                                         initial_value=1,
                                         vals=vals.Numbers())
            # coherence times
            self.add_parameter(f'T1{tn}', label=f'{tr_name} relaxation',
                               unit='s', initial_value=0,
                               parameter_class=ManualParameter)
            self.add_parameter(f'T2{tn}', label=f'{tr_name} dephasing Echo',
                               unit='s', initial_value=0,
                               parameter_class=ManualParameter)
            self.add_parameter(f'T2_star{tn}', label=f'{tr_name} dephasing',
                               unit='s', initial_value=0,
                               parameter_class=ManualParameter)


        # add qubit spectroscopy parameters
        self.add_parameter('spec_power', unit='dBm', initial_value=-20,
                           parameter_class=ManualParameter,
                           label='Qubit spectroscopy power')
        self.add_operation('Spec')
        self.add_pulse_parameter('Spec', 'spec_pulse_type', 'pulse_type',
                                 initial_value='SquarePulse',
                                 vals=vals.Enum('SquarePulse'))
        self.add_pulse_parameter('Spec', 'spec_marker_channel', 'channel',
                                 initial_value=None, vals=vals.Strings())
        self.add_pulse_parameter('Spec', 'spec_marker_amp', 'amplitude',
                                 vals=vals.Numbers(), initial_value=1)
        self.add_pulse_parameter('Spec', 'spec_marker_length', 'length',
                                 initial_value=5e-6, vals=vals.Numbers())
        self.add_pulse_parameter('Spec', 'spec_marker_delay', 'pulse_delay',
                                 vals=vals.Numbers(), initial_value=0)

        # Flux pulse parameters
        ps_name = 'flux_pulse'
        op_name = 'FP'
        self.add_operation(op_name)
        self.add_pulse_parameter(op_name, 'flux_pulse_type', 'pulse_type',
                                 initial_value='BufferedCZPulse',
                                 vals=vals.Enum('BufferedSquarePulse',
                                                'BufferedCZPulse',
                                                'NZBufferedCZPulse',
                                                'NZTransitionControlledPulse'))
        self.add_pulse_parameter(op_name, ps_name + '_channel', 'channel',
                                 initial_value='', vals=vals.Strings())
        self.add_pulse_parameter(op_name, ps_name + '_aux_channels_dict',
                                 'aux_channels_dict',
                                 initial_value={}, vals=vals.Dict())
        self.add_pulse_parameter(op_name, ps_name + '_amplitude', 'amplitude',
                                 initial_value=0.5, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, ps_name + '_frequency', 'frequency',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, ps_name + '_phase', 'phase',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, ps_name + '_pulse_length',
                                 'pulse_length',
                                 initial_value=100e-9, vals=vals.Numbers(0))
        self.add_pulse_parameter(op_name, ps_name + '_truncation_length',
                                 'truncation_length',
                                 initial_value=None)
        self.add_pulse_parameter(op_name, ps_name + '_buffer_length_start',
                                 'buffer_length_start', initial_value=20e-9,
                                 vals=vals.Numbers(0))
        self.add_pulse_parameter(op_name, ps_name + '_buffer_length_end',
                                 'buffer_length_end', initial_value=20e-9,
                                 vals=vals.Numbers(0))
        self.add_pulse_parameter(op_name, ps_name + '_extra_buffer_aux_pulse',
                                 'extra_buffer_aux_pulse', initial_value=5e-9,
                                 vals=vals.Numbers(0))
        self.add_pulse_parameter(op_name, ps_name + '_pulse_delay',
                                 'pulse_delay',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, ps_name + '_basis_rotation',
                                 'basis_rotation', initial_value={},
                                 vals=vals.Dict())
        self.add_pulse_parameter(op_name, ps_name + '_gaussian_filter_sigma',
                                 'gaussian_filter_sigma', initial_value=2e-9,
                                 vals=vals.Numbers(0))
        self.add_pulse_parameter(op_name, ps_name + '_trans_amplitude',
                                 '_trans_amplitude', initial_value=0,
                                 vals=vals.Numbers(),
                                 docstring="Used for NZTransitionControlledPulse")
        self.add_pulse_parameter(op_name, ps_name + '_trans_amplitude2',
                                 '_trans_amplitude2', initial_value=0,
                                 vals=vals.Numbers(),
                                 docstring="Used for NZTransitionControlledPulse")
        self.add_pulse_parameter(op_name, ps_name + '_trans_length',
                                 '_trans_length', initial_value=0,
                                 vals=vals.Numbers(0),
                                 docstring="Used for NZTransitionControlledPulse")

        # dc flux parameters
        self.add_parameter('dc_flux_parameter', initial_value=None,
                           label='QCoDeS parameter to sweep the dc flux',
                           parameter_class=ManualParameter)
        self.add_parameter('flux_parking', initial_value=0,
                           label='Flux (in units of phi0) at the parking '
                                 'position.',
                           vals=vals.Numbers(),
                           parameter_class=ManualParameter)

        # ac flux parameters
        self.add_parameter('flux_distortion', parameter_class=ManualParameter,
                           initial_value=deepcopy(
                               self.DEFAULT_FLUX_DISTORTION),
                           vals=vals.Dict())


        # Pulse preparation parameters
        DEFAULT_PREP_PARAMS = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1,
                                   final_reset_pulse=True,
                                   threshold_mapping={
                                       self.name: {0: 'g', 1: 'e'}})

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                            initial_value=DEFAULT_PREP_PARAMS, vals=vals.Dict())

        DEFAULT_GE_LO_CALIBRATION_PARAMS = dict(
            mode='fixed', # or 'freq_dependent'
            freqs=[],
            I_offsets=[],
            Q_offsets=[],
        )
        self.add_parameter('ge_lo_leakage_cal',
                           parameter_class=ManualParameter,
                           initial_value=DEFAULT_GE_LO_CALIBRATION_PARAMS,
                           vals=vals.Dict())

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

        # mixer calibration parameters
        self.add_parameter(
            'drive_mixer_calib_settings', parameter_class=ManualParameter,
            initial_value=dict(), vals=vals.Dict(),
            docstring='A dict whose keys are names of qcodes parameters of '
                      'the qubit object. For mixer calibration, these '
                      'parameters will be temporarily set to the respective '
                      'values provided in the dict.'
        )

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def _drive_mixer_calibration_tmp_vals(self):
        """Convert drive_mixer_calib_settings to temporary values format.

        Returns:
            A list of tuples to be passed to temporary_value (using *).
        """
        return [(self.parameters[k], v)
                for k, v in self.drive_mixer_calib_settings().items()]

    def get_ge_amp180_from_ge_freq(self, ge_freq):
        """
        Calculates the pi pulse amplitude required for a given ge transition
        frequency using the function stored in the parameter
        fit_ge_amp180_over_ge_freq or by performing an interpolation if a data
        array is stored in the parameter. If the parameter is None, the method
        returns None.

        :param ge_freq: ge transition frequency or an array of frequencies
        :return: pi pulse amplitude or an array of amplitudes (or None)
        """
        amp_func = self.fit_ge_amp180_over_ge_freq()
        if amp_func is None:
            log.warning(f'Cannot calculate drive amp for {self.name} since '
                        f'fit_ge_amp180_over_ge_freq is None.')
            return None
        elif isinstance(amp_func, str):
            return eval(amp_func)(ge_freq)
        else:
            i_min, i_max = np.argmin(amp_func[0]), np.argmax(amp_func[0])
            return sp.interpolate.interp1d(
                amp_func[0], amp_func[1], kind='linear',
                fill_value=(amp_func[1][i_min], amp_func[1][i_max]),
                bounds_error=False)(ge_freq)

    def get_ro_freq_from_ge_freq(self, ge_freq):
        """
        Calculates the RO frequency required for a given ge transition
        frequency using the function stored in the parameter
        fit_ro_freq_over_ge_freq. If this parameter is None, the method
        returns None.

        :param ge_freq: ge transition frequency or an array of frequencies
        :return: RO frequency or an array of frequencies (or None)
        """
        freq_func = self.fit_ro_freq_over_ge_freq()
        if freq_func is None:
            log.warning(f'Cannot calculate RO freq for {self.name} since '
                        f'fit_ro_freq_over_ge_freq is None.')
            return None
        return eval(freq_func)(ge_freq)

    def calculate_nonlinearity_correction(self, x):
        """
        Calculates the correction to a linear scaling of the pulse amplitude
        with respect to the pi-pulse amplitude using a 5th order odd polynomial
        and the coefficients from self.amp_scaling_correction_coeffs()

        Args:
             x: drive amplitude scaling factor with respect to the amplitude of
                the pi-pulse (amp180)
        """
        a, b = self.amp_scaling_correction_coeffs()
        return x * (a * (x ** 4 - 1) + b * (x ** 2 - 1) + 1)

    def calculate_frequency(self, bias=None, amplitude=0, transition='ge',
                            model='transmon_res', flux=None, update=False):
        """
        Calculates the transition frequency for a given DC bias and flux
        pulse amplitude using fit parameters stored in the qubit object.
        Note that the qubit parameter flux_amplitude_bias_ratio is used for
        conversion between bias values and amplitudes.

        :param bias: (float) DC bias. If model='approx' is used, the bias is
            optional, and is understood relative to the parking position at
            which the  model was measured. Otherwise, it mandatory and is
            interpreted as voltage of the DC source.
        :param amplitude: (float, default: 0) flux pulse amplitude
        :param transition: (str, default: 'ge') the transition whose
            frequency should be calculated. Currently, only 'ge' is
            implemented.
        :param model: (str, default: 'transmon_res') the model to use.
            'approx': Qubit_dac_to_freq with parameters from
                the qubit parameter fit_ge_freq_from_flux_pulse_amp.
                bias is understood as relative to the parking position.
            'transmon': Qubit_dac_to_freq_precise with parameters from
                the qubit parameter fit_ge_freq_from_dc_offset.
                bias is understood as the voltage of the DC source.
            'transmon_res': Qubit_dac_to_freq_res with parameters from
                the qubit parameter fit_ge_freq_from_dc_offset.
                bias is understood as the voltage of the DC source.
        :param flux: (float, default None) if this is not None, the frequency
            is calculated for the given flux (in units of phi_0) instead of
            for the given bias (for models 'transmon' and 'transmon_res') or
            instead of the given amplitude (for model 'approx'). If both bias
            and flux are None and the model is 'transmon' or 'transmon_res',
            the flux value from self.flux_parking() is used.
        :param update: (bool, default False) whether the result should be
            stored as {transition}_freq parameter of the qubit object.
        :return: calculated ge transition frequency
        """

        if transition not in ['ge', 'ef'] or (transition == 'ef' and
                                              model not in ['transmon_res']):
            raise NotImplementedError(
                f'calculate_frequency: Currently, transition {transition} is '
                f'not implemented for model {model}.')
        flux_amplitude_bias_ratio = self.flux_amplitude_bias_ratio()
        if flux_amplitude_bias_ratio is None:
            if ((model in ['transmon', 'transmon_res'] and amplitude != 0) or
                    (model == ['approx'] and bias is not None and bias != 0)):
                raise ValueError('flux_amplitude_bias_ratio is None, but is '
                                 'required for this calculation.')

        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
            if bias is None and flux is None:
                flux = self.flux_parking()
            if flux is not None:
                bias = self.calculate_voltage_from_flux(flux, model)
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()
            if flux is not None:
                amplitude = self.calculate_voltage_from_flux(flux, model)

        if model == 'approx':
            freq = fit_mods.Qubit_dac_to_freq(
                amplitude + (0 if bias is None or np.all(bias == 0) else
                             bias * flux_amplitude_bias_ratio), **vfc)
        elif model == 'transmon':
            kw = deepcopy(vfc)
            kw.pop('coupling', None)
            # FIXME: 'fr' refers to the bare readout-resonator frequency,
            #  this is not a very descriptive name. Should it be changed to
            #  'bare_ro_res_freq'? This is relevant to the device database.
            kw.pop('fr', None)
            freq = fit_mods.Qubit_dac_to_freq_precise(bias + (
                0 if np.all(amplitude == 0)
                else amplitude / flux_amplitude_bias_ratio), **kw)
        elif model == 'transmon_res':
            freq = fit_mods.Qubit_dac_to_freq_res(
                bias + (0 if np.all(amplitude == 0)
                        else amplitude / flux_amplitude_bias_ratio),
                return_ef=True, **vfc)[0 if transition == 'ge' else 1]
        else:
            raise NotImplementedError(
                "Currently, only the models 'approx', 'transmon', and"
                "'transmon_res' are implemented.")
        if update:
            self.parameters[f'{transition}_freq'](freq)
        return freq

    def calculate_flux_voltage(self, frequency=None, bias=None,
                               amplitude=None, transition='ge',
                               model='transmon_res', flux=None,
                               branch=None):
        """
        Calculates the flux pulse amplitude or DC bias required to reach a
        transition frequency using fit parameters stored in the qubit
        object. Note that the qubit parameter flux_amplitude_bias_ratio is
        used for conversion between bias values and amplitudes.
        :param frequency: (float, default: None = use self.ge_freq())
            transition frequency
        :param bias: (float, default; None) DC bias. If None, the function
            calculates the required DC bias to reach the target frequency
            (potentially taking into account the given flux pulse amplitude).
            Otherwise, it fixes the DC bias and calculates the required pulse
            amplitude. See note below.
        :param amplitude: (float, default: None) flux pulse amplitude. If None,
            the function calculates the required pulse amplitude to reach
            the target frequency (taking into account the given bias).
            Otherwise, it fixes the pulse amplitude and calculates the
            required bias. See note below.
        :param transition: (str, default: 'ge') the transition whose
            frequency should be calculated. Currently, only 'ge' is
            implemented.
        :param model: (str, default: 'transmon_res') the model to use.
            Currently 'transmon_res' and 'approx' are supported. See
            docstring of self.calculate_frequency
        :param flux: (float, default None) if this is not None, the bias
            parameter is overwritten with the bias corresponding to the given
            flux (in units of phi_0) for models 'transmon' and 'transmon_res'.
            This parameter is ignored if the model is 'approx'.
        :param branch: which branch of the flux-to-frequency curve should be
            used. See the meaning of this parameter in Qubit_freq_to_dac
            and Qubit_freq_to_dac_res. If None, this is set to the bias (if
            not None)
        :return: calculated bias or amplitude, depending on which parameters
            are passed in (see above and notes below).

        Notes:
        If model='approx' is used, the bias (parameter or return
            value) is understood relative to the parking position at
            which the model was measured. Otherwise, it is interpreted as
            voltage of the DC source.
        If both bias and amplitude are None, an amplitude is returned if the
            model is 'approx'. For the other models, a bias is returned in
            this case.
        """

        if frequency is None:
            frequency = self.ge_freq()
        if transition not in ['ge']:
            raise NotImplementedError(
                'Currently, only ge transition is implemented.')
        flux_amplitude_bias_ratio = self.flux_amplitude_bias_ratio()

        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
            if flux is not None:
                bias = self.calculate_voltage_from_flux(flux, model)
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()

        if flux_amplitude_bias_ratio is None:
            if bias is not None and amplitude is not None:
                raise ValueError(
                    'flux_amplitude_bias_ratio is None, but is '
                    'required for this calculation.')

        if branch is None:
            if bias is None and flux is None:
                branch = 'negative'
            else:
                # select well-defined branch close to requested flux
                if flux is None:
                    flux = (bias - vfc['dac_sweet_spot']) / vfc['V_per_phi0']
                if flux % 0.5:
                    pass  # do not shift (well-defined branch)
                elif flux != self.flux_parking():
                    # shift slightly in the direction of flux parking
                    flux += np.sign(self.flux_parking()-flux) * 0.25
                elif flux != 0:
                    # shift slightly in the direction of 0
                    flux += -np.sign(flux) * 0.25
                else:
                    # shift slightly to the left to use rising branch as default
                    flux = -0.25
                branch = flux * vfc['V_per_phi0'] + vfc['dac_sweet_spot']

        if model == 'approx':
            val = fit_mods.Qubit_freq_to_dac(frequency, **vfc, branch=branch)
        elif model == 'transmon_res':
            val = fit_mods.Qubit_freq_to_dac_res(
                frequency, **vfc, branch=branch, single_branch=True)
        else:
            raise NotImplementedError(
                "Currently, only the models 'approx' and"
                "'transmon_res' are implemented.")

        if model in ['transmon', 'transmon_res'] and bias is not None:
            # return amplitude
            val = (val - bias) * flux_amplitude_bias_ratio
        elif model in ['approx'] and bias is not None:
            # return amplitude
            val = val - bias * flux_amplitude_bias_ratio
        elif model in ['transmon', 'transmon_res'] and amplitude is not None:
            # return bias, corrected for amplitude
            val = val - amplitude / flux_amplitude_bias_ratio
        elif model in ['approx'] and amplitude is not None:
            # return bias
            val = (val - amplitude) / flux_amplitude_bias_ratio
        # If both bias and amplitude are None, the bare result is returned,
        # see note in the doctring.
        return val

    def calculate_voltage_from_flux(self, flux, model='transmon_res'):
        """
        Calculates the DC bias for a given target flux.

        :param flux: (float) flux in units of phi_0
        :param model: (str, default: 'transmon_res') the model to use,
            see calculate_frequency.
        :return: calculated DC bias if model is transmon or transmon_res,
            calculated flux pulse amplitude otherwise
        """
        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()
        return vfc['dac_sweet_spot'] + vfc['V_per_phi0'] * flux

    def calc_flux_amplitude_bias_ratio(self, amplitude, ge_freq, bias=None,
                                       flux=None, update=False):
        """
        Calculates the conversion factor between flux pulse amplitudes and bias
        voltage changes that lead to the same qubit detuning. The calculation is
        done based on the model Qubit_freq_to_dac_res and the parameters stored
        in the qubit parameter fit_ge_freq_from_dc_offset.

        :param amplitude: (float) flux pulse amplitude
        :param ge_freq: (float) measured ge transition frequency
        :param bias: (float) DC bias, i.e., voltage of the DC source.
        :param flux: (float) if this is not None, the value of the bias
            is overwritten with the voltage corresponding to the given flux
            (in units of phi_0). If both bias and flux are None, the flux
            value from self.flux_parking() is used.
        :param update: (bool, default False) whether the result should be
            stored as flux_amplitude_bias_ratio parameter of the qubit object.
        :return: calculated conversion factor
        """
        if bias is None and flux is None:
            flux = self.flux_parking()
        if flux is not None:
            bias = self.calculate_voltage_from_flux(flux)
        v = fit_mods.Qubit_freq_to_dac_res(
            ge_freq, **self.fit_ge_freq_from_dc_offset(), branch=bias)
        flux_amplitude_bias_ratio = amplitude / (v - bias)
        if flux_amplitude_bias_ratio < 0:
            log.warning('The extracted flux_amplitude_bias_ratio is negative, '
                        'please check your input values.')
        if update:
            self.flux_amplitude_bias_ratio(flux_amplitude_bias_ratio)
        return flux_amplitude_bias_ratio

    def generate_scaled_volt_freq_conv(self, scaling=None, flux=None,
                                       bias=None):
        """
        Generates a scaled and shifted version of the voltage frequency
        conversion dictionary (self.fit_ge_freq_from_dc_offset). This can,
        e.g., be used to calculate flux pulse amplitude to ge frequency
        conversion using fit_mods.Qubit_dac_to_freq_res. This shift is done
        relative to obtain a model that is relative to a flux offset (
        parking position) indicated by either flux or bias.
        :param scaling: the scaling factor. Default: use
            self.flux_amplitude_bias_ratio()
        :param flux: parking position in unit of Phi_0. If both bias and flux
            are None, the flux value from self.flux_parking() is used.
        :param bias: If not None, overwrite flux with the flux resulting from
            the given DC voltage.
        :return: the scaled and shifed voltage frequency conversion dictionary
        """
        vfc = deepcopy(self.fit_ge_freq_from_dc_offset())
        if scaling is None:
            scaling = self.flux_amplitude_bias_ratio()
        if bias is not None:
            flux = (bias - vfc['dac_sweet_spot']) / vfc['V_per_phi0']
        elif flux is None:
            flux = self.flux_parking()
        vfc['V_per_phi0'] *= scaling
        vfc['dac_sweet_spot'] = -flux * vfc['V_per_phi0']
        return vfc

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

    def update_detector_functions(self):
        """
        Instantiates common detector classes and assigns them as attributes.
        See detector_functions.py for all available detector classes and the
        docstrings of the individual detector classes for more details.

        Creates the following attributes:
            - self.int_log_det: IntegratingSingleShotPollDetector with
                data_type='raw'
                Used for single shot acquisition
            - self.dig_log_det: IntegratingSingleShotPollDetector with
                data_type='digitized'
                Used for thresholded single shot acquisition
            - self.int_avg_classif_det: ClassifyingPollDetector
                Used for classified acquisition.
            - self.int_avg_det: IntegratingAveragingPollDetector with
                data_type='raw'
                Used for integrated averaged acquisition
            - self.dig_avg_det: IntegratingAveragingPollDetector with
                data_type='digitized'
                Used for thresholded integrated averaged acquisition
            - int_avg_det_spec: IntegratingAveragingPollDetector with
                single_int_avg=True (soft detector)
                Used for spectroscopy measurements
            - self.inp_avg_det: AveragingPollDetector
                Used for recording timetraces
            - self.scope_fft_det: UHFQC_scope_detector
                Used for acquisition with the scope module of the UHF.
        """
        int_channels = self.get_acq_int_channels()

        self.int_log_det = det.IntegratingSingleShotPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=int_channels, nr_shots=self.acq_shots(),
            integration_length=self.acq_length(),
            data_type='raw')

        self.int_avg_classif_det = det.ClassifyingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=int_channels, nr_shots=self.acq_averages(),
            integration_length=self.acq_length(),
            get_values_function_kwargs={
                'classifier_params': [self.acq_classifier_params()],
                'state_prob_mtx': [self.acq_state_prob_mtx()]
            })

        self.int_avg_det = det.IntegratingAveragingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=int_channels, nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            data_type='raw')

        self.dig_avg_det = det.IntegratingAveragingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=int_channels, nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            data_type='digitized')

        self.inp_avg_det = det.AveragingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=self.get_acq_inp_channels(),
            nr_averages=self.acq_averages(),
            acquisition_length=self.acq_length())

        self.dig_log_det = det.IntegratingSingleShotPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=int_channels, nr_shots=self.acq_shots(),
            integration_length=self.acq_length(),
            data_type='digitized')

        awg_ctrl = self.instr_acq.get_instr().get_awg_control_object()[0]
        self.int_avg_det_spec = det.IntegratingAveragingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=awg_ctrl,
            channels=self.get_acq_int_channels(n_channels=2),
            nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            data_type='raw', real_imag=False, single_int_avg=True)

        if 'UHF' in self.instr_acq.get_instr().__class__.__name__ and hasattr(
                self.instr_acq.get_instr().daq, 'scopeModule'):
            self.scope_fft_det = det.UHFQC_scope_detector(
                UHFQC=self.instr_acq.get_instr(),
                AWG=self.instr_pulsar.get_instr(),
                fft_mode='fft_power',
                nr_averages=self.acq_averages(),
                acquisition_length=self.acq_length()
            )
        else:
            self.scope_fft_det = det.ScopePollDetector(
                acq_dev=self.instr_acq.get_instr(),
                AWG=awg_ctrl,
                channels=self.get_acq_inp_channels(),
                data_type='fft_power',
                nr_averages=self.acq_averages(),
                nr_shots=1,
                acquisition_length=self.acq_length()
            )

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
                'pulsed_spec' (pulsed spectroscopy), or the default
                'timedomain' (AWG-generated signal upconverted by the mixer)
            switch (str): the required switch mode. Can be a switch mode
                understood by set_switch or the default value 'default', in
                which case the switch mode is determined based on the kind
                of drive ('spec' for continuous/pulsed spectroscopy;
                'no_drive' if drive is None and a switch mode 'no_drive' is
                configured for this qubit; 'modulated' in all other cases).
        """
        self.configure_mod_freqs()
        ro_lo = self.instr_ro_lo
        ge_lo = self.instr_ge_lo

        self.configure_offsets(set_ge_offsets=(drive == 'timedomain'))
        self.set_distortion_in_pulsar()
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

        # configure qubit drive local oscillator
        if ge_lo() is not None:
            if drive is None:
                ge_lo.get_instr().off()
            elif drive == 'continuous_spec':
                ge_lo.get_instr().pulsemod_state('Off')
                ge_lo.get_instr().power(self.spec_power())
                ge_lo.get_instr().frequency(self.ge_freq())
                ge_lo.get_instr().on()
            elif drive == 'pulsed_spec':
                ge_lo.get_instr().pulsemod_state('On')
                if 'pulsemod_source' in ge_lo.get_instr().parameters:
                    ge_lo.get_instr().pulsemod_source('EXT')
                ge_lo.get_instr().power(self.spec_power())
                ge_lo.get_instr().frequency(self.ge_freq())
                ge_lo.get_instr().on()
            elif drive == 'timedomain':
                ge_lo.get_instr().pulsemod_state('Off')
                ge_lo.get_instr().power(self.ge_lo_power())
                ge_lo.get_instr().frequency(self.get_ge_lo_freq())
                ge_lo.get_instr().on()
            else:
                raise ValueError("Invalid drive parameter '{}'".format(drive)
                                 + ". Valid options are None, 'continuous_spec"
                                 + "', 'pulsed_spec' and 'timedomain'.")

        param = f'{self.ge_I_channel()}_centerfreq'
        if param in self.instr_pulsar.get_instr().parameters:
            self.instr_pulsar.get_instr().set(param, self.get_ge_lo_freq())

        # other preparations
        self.update_detector_functions()
        self.set_readout_weights()
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

    def get_ge_lo_freq(self):
        """Returns the required local oscillator frequency for drive pulses

        The drive LO freq is calculated from the ge_mod_freq (intermediate
        frequency) and the ge_freq stored in the qubit object.
        """
        return self.ge_freq() - self.ge_mod_freq()

    def get_ge_lo_identifier(self):
        """Returns the ge LO identifier in one of the formats specified below.

        Returns:
            str indicating the instrument name of an external LO
            tuple of drive pulse generating device name (str) and
              synthesizer unit index (int), identifying the internal
              LO in an signal generation unit of an drive pulse
              generating device
          """

        if self.instr_ge_lo() is None:
            pulsar = self.instr_pulsar.get_instr()
            awg = pulsar.get_channel_awg(self.ge_I_channel())
            gen = pulsar.get_centerfreq_generator(self.ge_I_channel())
            return (awg.name, gen)
        else:
            return self.instr_ge_lo()

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

    def get_ro_lo_identifier(self):
        """Returns the ro LO identifier in one of the formats specified below.

        Returns:
            str indicating the instrument name of an external LO
            tuple of acquisition device name (str) and acquisition
              unit index (int), identifying the internal LO in an
              acquisition unit of an acquisition device
        """

        if self.instr_ro_lo() is None:
            return (self.instr_acq(), self.acq_unit())
        else:
            return self.instr_ro_lo()

    def set_readout_weights(self, weights_type=None, f_mod=None):
        """Set acquisition weights for this qubit in the acquisition device.

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
            weights_type = self.acq_weights_type()
        if f_mod is None:
            f_mod = self.ro_mod_freq()
        self.instr_acq.get_instr().acquisition_set_weights(
            channels=self.get_acq_int_channels(n_channels=2),
            weights_type=weights_type, mod_freq=f_mod,
            acq_IQ_angle=self.acq_IQ_angle(),
            weights_I=[self.acq_weights_I(), self.acq_weights_I2()],
            weights_Q=[self.acq_weights_Q(), self.acq_weights_Q2()],
        )

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

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

    def get_ro_pars(self):
        return self.get_operation_dict()['RO ' + self.name]

    def get_acq_pars(self):
        return self.get_operation_dict()['Acq ' + self.name]

    def get_ge_pars(self):
        return self.get_drive_pars('ge')

    def get_ef_pars(self):
        return self.get_drive_pars('ef')

    def get_drive_pars(self, transition_name):
        tn = '' if transition_name == 'ge' else f'_{transition_name}'
        return self.get_operation_dict()[f'X180{tn} ' + self.name]

    def get_operation_dict(self, operation_dict=None):
        self.configure_mod_freqs()
        if operation_dict is None:
            operation_dict = {}
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['Spec ' + self.name]['operation_type'] = 'Other'
        operation_dict['RO ' + self.name]['operation_type'] = 'RO'
        operation_dict['Acq ' + self.name] = deepcopy(
            operation_dict['RO ' + self.name])
        operation_dict['Acq ' + self.name]['amplitude'] = 0
        operation_dict['Acq ' + self.name]['flux_amplitude'] = 0

        for tr_name in self.transition_names:
            tn = '' if tr_name == 'ge' else f'_{tr_name}'
            operation_dict[f'X180{tn} ' + self.name]['basis'] = self.name + tn
            operation_dict[f'X180{tn} ' + self.name]['operation_type'] = 'MW'
            if tr_name != 'ge':
                operation_dict[f'X180{tn} ' + self.name]['I_channel'] = \
                    operation_dict['X180 ' + self.name]['I_channel']
                operation_dict[f'X180{tn} ' + self.name]['Q_channel'] = \
                    operation_dict['X180 ' + self.name]['Q_channel']
                operation_dict[f'X180{tn} ' + self.name]['phi_skew'] = \
                    operation_dict['X180 ' + self.name]['phi_skew']
                operation_dict[f'X180{tn} ' + self.name]['alpha'] = \
                    operation_dict['X180 ' + self.name]['alpha']
                if self.get(f'{tr_name}_freq') == 0:
                    operation_dict[f'X180{tn} ' + self.name][
                        'mod_frequency'] = None
                else:
                    operation_dict['X180_ef ' + self.name][
                        'mod_frequency'] = self.get(f'{tr_name}_freq') - \
                                           self.ge_freq() + self.ge_mod_freq()
            operation_dict.update(add_suffix_to_dict_keys(
                sq.get_pulse_dict_from_pars(
                    operation_dict[f'X180{tn} ' + self.name]),
                f'{tn} ' + self.name))

        if np.ndim(self.ro_freq()) != 0:
            delta_freqs = np.diff(self.ro_freq(), prepend=self.ro_freq()[0])
            mods = [self.ro_mod_freq() + d for d in delta_freqs]
            operation_dict['RO ' + self.name]['mod_frequency'] = mods

        for code, op in operation_dict.items():
            op['op_code'] = code
        return operation_dict

    def swf_drive_lo_freq(self, allow_IF_sweep=True):
        """Create a sweep function for sweeping the drive frequency.

        The sweep is implemented as an LO sweep in case of drive pulse
        generation with an external LO. The implementation depends on the
        get_frequency_sweep_function method of the acquisition device in case
        of an internal LO.

        Args:
            allow_IF_sweep (bool): specifies whether an IF sweep (or a combined
                LO and IF sweep) may be used (default: True). Note that
                setting this to False might lead to a sweep function that is
                only allowed to take specific values supported by the
                internal LO.

        Returns: the Sweep_function object
        """
        if self.instr_ge_lo() is not None:  # external LO
            return mc_parameter_wrapper.wrap_par_to_swf(
                self.instr_ge_lo.get_instr().frequency)
        else:  # no external LO
            pulsar = self.instr_pulsar.get_instr()
            return pulsar.get_frequency_sweep_function(
                self.ge_I_channel(), allow_IF_sweep=allow_IF_sweep)

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

    def measure_resonator_spectroscopy(self, freqs, sweep_points_2D=None,
                                       sweep_function_2D=None,
                                       trigger_separation=3e-6,
                                       upload=True, analyze=True,
                                       close_fig=True, label=None):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if label is None:
            if sweep_function_2D is not None:
                label = 'resonator_scan_2d' + self.msmt_suffix
            else:
                label = 'resonator_scan' + self.msmt_suffix
        self.prepare(drive=None)
        if upload:
            ro_pars = self.get_ro_pars()
            if self.instr_ro_lo() is None:
                ro_pars['mod_frequency'] = 0
            seq = sq.pulse_list_list_seq([[ro_pars]], upload=False)

            for seg in seq.segments.values():
                if hasattr(self.instr_acq.get_instr(),
                           'use_hardware_sweeper') and \
                        self.instr_acq.get_instr().use_hardware_sweeper():
                    self.int_avg_det_spec.AWG = self.instr_pulsar.get_instr()
                    lo_freq, delta_f, _ = self.instr_acq.get_instr()\
                        .get_params_for_spectrum(freqs)
                    self.instr_acq.get_instr().set_lo_freq(self.acq_unit(),
                                                           lo_freq)
                    seg.acquisition_mode = dict(
                        sweeper='hardware',
                        f_start=freqs[0] - lo_freq,
                        f_step=delta_f,
                        n_step=len(freqs),
                        seqtrigger=True,
                    )
                else:
                    seg.acquisition_mode = dict(
                        sweeper='software'
                    )
            self.instr_pulsar.get_instr().program_awgs(seq)

        MC = self.instr_mc.get_instr()
        MC.set_sweep_function(self.swf_ro_freq_lo())
        if sweep_function_2D is not None:
            MC.set_sweep_function_2D(sweep_function_2D)
            mode = '2D'
        else:
            mode = '1D'
        MC.set_sweep_points(freqs)
        if sweep_points_2D is not None:
            MC.set_sweep_points_2D(sweep_points_2D)
        if MC.sweep_functions[0].sweep_control == 'soft':
            MC.set_detector_function(self.int_avg_det_spec)
        else:
            # The following ensures that we use a hard detector if the acq
            # dev provided a sweep function for a hardware IF sweep.
            self.int_avg_det.set_real_imag(False)
            self.int_avg_det.AWG = self.int_avg_det_spec.AWG
            MC.set_detector_function(self.int_avg_det)

        with temporary_value(self.instr_trigger.get_instr().pulse_period,
                             trigger_separation):
            if self.int_avg_det_spec.AWG != self.instr_pulsar.get_instr():
                awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
                self.instr_pulsar.get_instr().start(exclude=[awg_name])
            MC.run(name=label, mode=mode)
            if self.int_avg_det_spec.AWG != self.instr_pulsar.get_instr():
                self.instr_pulsar.get_instr().stop()

        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig, qb_name=self.name,
                                   TwoD=(mode == '2D'))

    def measure_qubit_spectroscopy(self, freqs, sweep_points_2D=None,
            sweep_function_2D=None, pulsed=True, trigger_separation=13e-6,
            upload=True, analyze=True, close_fig=True, label=None):
        """ Varies qubit drive frequency and measures the resonator
        transmittance """
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))
        temp_vals = list()
        pulsar = self.instr_pulsar.get_instr()
        awg_name = pulsar.get(f'{self.ge_I_channel()}_awg')
        hard_sweep = f'{awg_name}_use_hardware_sweeper' in pulsar.parameters and \
                     pulsar.get(f"{awg_name}_use_hardware_sweeper")

        # For pulsed and hard_sweep spectroscopies we add an empty spec pulse to
        # trigger the the drive/marker AWG and afterwards add the actual spec
        # pulse (either empty for a continuous hard sweep or the pulse for the
        # pulsed spectroscopy). This way we are able to implement a delay
        # between the trigger and the spec pulse that is needed in hard_sweeps
        # to set the osc. frequency.
        # FIXME: think about cleaner solution
        empty_trigger = self.get_spec_pars()
        empty_trigger['length'] = 0
        empty_trigger['pulse_delay'] = 0
        if pulsed:
            if label is None:
                if sweep_function_2D is not None:
                    label = 'pulsed_spec_2d' + self.msmt_suffix
                else:
                    label = 'pulsed_spec' + self.msmt_suffix
            self.prepare(drive='pulsed_spec')
            if upload:
                spec_pulse = self.get_spec_pars()
                if hard_sweep or self.instr_ge_lo() is None:
                    # No external LO, use pulse to set the spec power
                    spec_pulse["amplitude"] = dbm_to_vp(self.spec_power())
                seq = sq.pulse_list_list_seq([[empty_trigger,
                                               spec_pulse,
                                               self.get_ro_pars()]],
                                             upload=False)
        else:
            if label is None:
                if sweep_function_2D is not None:
                    label = 'continuous_spec_2d' + self.msmt_suffix
                else:
                    label = 'continuous_spec' + self.msmt_suffix
            self.prepare(drive='continuous_spec')
            if upload:
                if self.instr_ge_lo() is None and not hard_sweep:
                    amp_range = pulsar.get(f'{self.ge_I_channel()}_amp')
                    amp = dbm_to_vp(self.spec_power())
                    gain = amp / amp_range
                    temp_vals += [(pulsar.parameters[
                                       f"{self.ge_I_channel()}_direct_mod_freq"],
                                   self.ge_mod_freq())]
                    temp_vals += [(pulsar.parameters[
                                       f"{self.ge_I_channel()}_direct_output_amp"],
                                   gain)]
                if hard_sweep:
                    # we use the empty pulse to tell pulsar how to configure the
                    # osc sweep and sine output. The empty pulse is also used
                    # to trigger the SeqC code to set the next osc. frequency.
                    # This needs to be done after the RO.
                    seq = sq.pulse_list_list_seq([[self.get_ro_pars(),
                                                   empty_trigger]],
                                                 upload=False)
                else:
                    seq = sq.pulse_list_list_seq([[self.get_ro_pars()]],
                                                 upload=False)
        if upload:
            for seg in seq.segments.values():
                ch = self.ge_I_channel()
                seg.mod_config[ch] = \
                    dict(internal_mod=pulsed)
                if hard_sweep:
                    amp_range = pulsar.get(f'{ch}_amp')
                    amp = dbm_to_vp(self.spec_power())
                    gain = amp / amp_range
                    center_freq, mod_freqs = \
                        pulsar.get_params_for_spectrum(ch, freqs)
                    pulsar.set(f'{ch}_centerfreq', center_freq)
                    seg.sine_config[ch] = dict(continuous=not pulsed,
                                               ignore_waveforms=not pulsed,
                                               gains=tuple(gain * x for x in (
                                               0.0, 1.0, 1.0, 0.0)))
                    seg.sweep_params[f'{ch}_osc_sweep'] = mod_freqs
            pulsar.program_awgs(seq)

        MC = self.instr_mc.get_instr()
        MC.set_sweep_function(self.swf_drive_lo_freq())
        if sweep_function_2D is not None:
            MC.set_sweep_function_2D(sweep_function_2D)
            mode = '2D'
        else:
            mode = '1D'
        MC.set_sweep_points(freqs)
        if sweep_points_2D is not None:
            MC.set_sweep_points_2D(sweep_points_2D)
        if MC.sweep_functions[0].sweep_control == 'soft':
            MC.set_detector_function(self.int_avg_det_spec)
        else:
            # The following ensures that we use a hard detector if the swf
            # provided by swf_drive_lo_freq uses a hardware IF sweep.
            self.int_avg_det.set_real_imag(False)
            MC.set_detector_function(self.int_avg_det)
        temp_vals += [(self.instr_trigger.get_instr().pulse_period,
                       trigger_separation)]
        with temporary_value(*temp_vals):
            awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
            pulsar.start(exclude=[awg_name])
            MC.run(name=label, mode=mode)
            pulsar.stop()

        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig, qb_name=self.name,
                                   TwoD=(mode == '2D'))

    def measure_transients(self, states=('g', 'e'), upload=True,
                           analyze=True, acq_length=4096/1.8e9,
                           prep_params=None, exp_metadata=None, **kw):
        """
        If the resulting transients will be used to caclulate the optimal
        weight functions, then it is important that the UHFQC iavg_delay and
        wint_delay are calibrated such that the weights and traces are
        aligned: iavg_delay = 2*wint_delay.

        """
        MC = self.instr_mc.get_instr()
        name_extra = kw.get('name_extra', None)

        if prep_params is None:
            prep_params = self.preparation_params()
        if exp_metadata is None:
            exp_metadata = dict()
        exp_metadata.update(
            {'sweep_name': 'time',
             'sweep_unit': ['s']})

        with temporary_value(self.acq_length, acq_length):
            self.prepare(drive='timedomain')
            swpts = self.instr_acq.get_instr().get_sweep_points_time_trace(
                acq_length)
            for state in states:
                if state not in ['g', 'e', 'f']:
                    raise ValueError("Unrecognized state: {}. Must be 'g', 'e' "
                                     "or 'f'.".format(state))
                base_name = 'timetrace_{}'.format(state)
                name = base_name + "_" + name_extra if name_extra is not None \
                    else base_name
                seq, _ = sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name, state=state, prep_params=prep_params,
                    upload=False)
                # set sweep function and run measurement
                MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq,
                                                               upload=upload))
                MC.set_sweep_points(swpts)
                MC.set_detector_function(self.inp_avg_det)
                exp_metadata.update(dict(sweep_points_dict=swpts))
                MC.run(name=name + self.msmt_suffix, exp_metadata=exp_metadata)

    def measure_readout_pulse_scope(self, delays, freqs, RO_separation=None,
                                    prep_pulses=None, comm_freq=225e6,
                                    analyze=True, label=None,
                                    close_fig=True, upload=True, verbose=False,
                                    cal_points=((-4, -3), (-2, -1)), MC=None):
        """
        From the documentation of the used sequence function:

        Prepares the AWGs for a readout pulse shape and timing measurement.

        The sequence consists of two readout pulses where the drive pulse start
        time is swept through the first readout pulse. Because the photons in
        the readout resonator induce an ac-Stark shift of the qubit frequency,
        we can determine the readout pulse shape by sweeping the drive frequency
        in an outer loop to determine the qubit frequency.

        Important: This sequence includes two readouts per segment. For this
        reason the calibration points are also duplicated.

        Args:
            delays: A list of delays between the start of the first readout pulse
                    and the center of the drive pulse.
            RO_separation: Separation between the starts of the two readout pulses.
                           If the comm_freq parameter is not None, the used value
                           is increased to satisfy the commensurability constraint.
            cal_points: True for default calibration points, False for no
                        calibration points or a list of two lists, containing
                        the indices of the calibration segments for the ground
                        and excited state.
            comm_freq: The readout pulse separation will be a multiple of
                       1/comm_freq
        """
        if self.instr_ge_lo() is None:
            raise NotImplementedError("qb.measure_readout_pulse_scope is not "
                                      "implemented for setups without ge LO. "
                                      "Use quantum experiment "
                                      "ReadoutPulseScope instead.")

        if delays is None:
            raise ValueError("Unspecified delays for "
                             "measure_readout_pulse_scope")
        if label is None:
            label = 'Readout_pulse_scope' + self.msmt_suffix
        if MC is None:
            MC = self.instr_mc.get_instr()
        if freqs is None:
            freqs = self.f_qubit() + np.linspace(-50e6, 50e6, 201)
        if RO_separation is None:
            RO_separation = 2 * self.ro_length()
            RO_separation += np.max(delays)
            RO_separation += 200e-9  # for slack

        self.prepare(drive='timedomain')
        MC.set_sweep_function(awg_swf.Readout_pulse_scope_swf(
            delays=delays,
            pulse_pars=self.get_ge_pars(),
            RO_pars=self.get_ro_pars(),
            RO_separation=RO_separation,
            cal_points=cal_points,
            prep_pulses=prep_pulses,
            comm_freq=comm_freq,
            verbose=verbose,
            upload=upload))
        MC.set_sweep_points(delays)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            mc_parameter_wrapper.wrap_par_to_swf(
                self.instr_ge_lo.get_instr().frequency),
            -self.ge_mod_freq(),
            parameter_name=self.name + ' drive frequency'))
        MC.set_sweep_points_2D(freqs)

        d = det.IntegratingAveragingPollDetector(
            acq_dev=self.instr_acq.get_instr(),
            AWG=self.instr_pulsar.get_instr(),
            channels=self.int_avg_det.channels,
            nr_averages=self.acq_averages(),
            integration_length=self.acq_length(),
            data_type='raw',
            values_per_point=2,
            values_per_point_suffix=['_probe', '_measure'])
        MC.set_detector_function(d)
        MC.run_2D(label)

        # Create a MeasurementAnalysis object for this measurement
        if analyze:
            ma.MeasurementAnalysis(TwoD=True, auto=True, close_fig=close_fig,
                                   qb_name=self.name)

    def measure_drive_mixer_spectrum(self, if_freqs, amplitude=0.5,
                                     trigger_sep=5e-6, align_frequencies=True):
        MC = self.instr_mc.get_instr()
        if align_frequencies:
            if_freqs = (if_freqs*trigger_sep).astype(np.int)/trigger_sep
        s = swf.Offset_Sweep(
            self.instr_ro_lo.get_instr().frequency,
            self.ge_freq() - self.ro_mod_freq() - self.ge_mod_freq(),
            name='Drive intermediate frequency',
            parameter_name='Drive intermediate frequency')
        MC.set_sweep_function(s)
        MC.set_sweep_points(if_freqs)
        MC.set_detector_function(self.int_avg_det_spec)
        with temporary_value(
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
            *self._drive_mixer_calibration_tmp_vals()
        ):
            drive_pulse = dict(
                    pulse_type='GaussFilteredCosIQPulse',
                    pulse_length=self.acq_length(),
                    ref_point='start',
                    amplitude=amplitude,
                    I_channel=self.ge_I_channel(),
                    Q_channel=self.ge_Q_channel(),
                    mod_frequency=self.ge_mod_freq(),
                    phase_lock=False,
                )
            sq.pulse_list_list_seq([[self.get_acq_pars(), drive_pulse]])

            self.prepare(drive='timedomain', switch='calib')
            self.instr_pulsar.get_instr().start()
            MC.run('ge_uc_spectrum' + self.msmt_suffix)

        a = ma.MeasurementAnalysis(plot_args=dict(log=True, marker=''))
        return a

    def measure_drive_mixer_spectrum_fft(self, ro_lo_freq, amplitude=0.5,
                                         trigger_sep=5e-6):
        MC = self.instr_mc.get_instr()
        s = swf.None_Sweep(
            name='UHF intermediate frequency',
            parameter_name='UHF intermediate frequency',
            unit='Hz')
        with temporary_value(
                (self.ro_freq, ro_lo_freq + self.ro_mod_freq()),
                (self.instr_trigger.get_instr().pulse_period, trigger_sep),
                *self._drive_mixer_calibration_tmp_vals()
        ):
            drive_pulse = dict(
                pulse_type='GaussFilteredCosIQPulse',
                pulse_length=self.acq_length(),
                ref_point='start',
                amplitude=amplitude,
                I_channel=self.ge_I_channel(),
                Q_channel=self.ge_Q_channel(),
                mod_frequency=self.ge_mod_freq(),
                phase_lock=False,
            )
            sq.pulse_list_list_seq([[self.get_acq_pars(), drive_pulse]])

            self.prepare(drive='timedomain', switch='calib')
            MC.set_sweep_function(s)
            MC.set_sweep_points(self.scope_fft_det.get_sweep_vals())
            MC.set_detector_function(self.scope_fft_det)
            self.instr_pulsar.get_instr().start()
            MC.run('ge_uc_spectrum' + self.msmt_suffix)

        a = ma.MeasurementAnalysis(plot_args=dict(log=True, marker=''))
        return a

    def _calibrate_drive_mixer_carrier_common(
            self, detector_generator, update=True, x0=(0., 0.),
            initial_stepsize=0.01, trigger_sep=5e-6, no_improv_break=50,
            upload=True, plot=True):

        MC = self.instr_mc.get_instr()
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': x0,
                        'initial_step': [initial_stepsize, initial_stepsize],
                        'no_improv_break': no_improv_break,
                        'minimize': True,
                        'maxiter': 500}
        chI_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_I_channel())]
        chQ_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_Q_channel())]
        MC.set_sweep_functions([chI_par, chQ_par])
        MC.set_adaptive_function_parameters(ad_func_pars)
        with temporary_value(
                (self.ro_freq, self.ge_freq() - self.ge_mod_freq()),
                (self.instr_trigger.get_instr().pulse_period, trigger_sep),
                *self._drive_mixer_calibration_tmp_vals()
        ):
            if upload:
                sq.pulse_list_list_seq([[self.get_acq_pars(), dict(
                                    pulse_type='GaussFilteredCosIQPulse',
                                    pulse_length=self.acq_length(),
                                    ref_point='start',
                                    amplitude=0,
                                    I_channel=self.ge_I_channel(),
                                    Q_channel=self.ge_Q_channel(),
                                )]])

            self.prepare(drive='timedomain', switch='calib')
            MC.set_detector_function(detector_generator())
            awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
            self.instr_pulsar.get_instr().start(exclude=[awg_name])
            MC.run(name='drive_carrier_calibration' + self.msmt_suffix,
                   mode='adaptive')

        a = ma.OptimizationAnalysis(label='drive_carrier_calibration')
        if plot:
            # v2 creates a pretty picture of the optimizations
            ma.OptimizationAnalysis_v2(label='drive_carrier_calibration')

        ch_1_min = a.optimization_result[0][0]
        ch_2_min = a.optimization_result[0][1]
        if update:
            self.ge_I_offset(ch_1_min)
            self.ge_Q_offset(ch_2_min)
        return ch_1_min, ch_2_min

    def calibrate_drive_mixer_carrier_fft(
            self, update=True, x0=(0., 0.), initial_stepsize=0.01,
            trigger_sep=5e-6, no_improv_break=50, upload=True, plot=True):

        def detector_generator(s=self):
            d = s.scope_fft_det
            d.AWG = None
            idx = np.argmin(np.abs(d.get_sweep_vals() -
                                   np.abs(s.ro_mod_freq())))
            return det.IndexDetector(det.SumDetector(d), (0, idx))

        return self._calibrate_drive_mixer_carrier_common(
            detector_generator, update=update, x0=x0,
            initial_stepsize=initial_stepsize, trigger_sep=trigger_sep,
            no_improv_break=no_improv_break, upload=upload, plot=plot)

    def calibrate_drive_mixer_carrier(self, update=True, x0=(0., 0.),
                                      initial_stepsize=0.01, trigger_sep=5e-6,
                                      no_improv_break=50, upload=True,
                                      plot=True):
        def detector_generator(s=self):
            return det.IndexDetector(s.int_avg_det_spec, 0)

        return self._calibrate_drive_mixer_carrier_common(
            detector_generator, update=update, x0=x0,
            initial_stepsize=initial_stepsize, trigger_sep=trigger_sep,
            no_improv_break=no_improv_break, upload=upload, plot=plot)

    def calibrate_readout_mixer_carrier(self, other_qb, update=True,
                                        x0=(0., 0.),
                                        initial_stepsize=0.01, trigger_sep=5e-6,
                                        no_improv_break=50, upload=True,
                                        plot=True):
        """
        Calibrate readout upconversion mixer local oscillator leakage

        Args:
            other_qb:
                a qubit on another acquisition device that is configured to
                see the LO leakage output of the readout UC of self
            other arguments as in calibrate_drive_mixer_carrier

        Example:
            >>> # configure switches to readout mixer calib configuration
            >>> # ...
            >>>
            >>> with temporary_value((qb_other.ro_mod_freq, 100e6),
            >>>                      (qb_other.acq_length, 1e-6)):
            >>>     qb.calibrate_readout_mixer_carrier(
            >>>         qb_other, trigger_sep=40e-6, no_improv_break=20)
            >>>
            >>> # configure switches to nominal configuration
            >>> # ...
            >>>
            >>> for qb_on_feedline in qubits_feedline:
            >>>     qb_on_feedline.ro_I_offset(qb.ro_I_offset())
            >>>     qb_on_feedline.ro_Q_offset(qb.ro_Q_offset())
        """

        MC = self.instr_mc.get_instr()
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': x0,
                        'initial_step': [initial_stepsize, initial_stepsize],
                        'no_improv_break': no_improv_break,
                        'minimize': True,
                        'maxiter': 500}
        chI_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ro_I_channel())]
        chQ_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ro_Q_channel())]
        MC.set_sweep_functions([chI_par, chQ_par])
        MC.set_adaptive_function_parameters(ad_func_pars)
        if upload:
            sq.pulse_list_list_seq([[other_qb.get_acq_pars(), dict(
                pulse_type='GaussFilteredCosIQPulse',
                pulse_length=self.acq_length(),
                ref_point='start',
                amplitude=0,
                I_channel=self.ro_I_channel(),
                Q_channel=self.ro_Q_channel(),
            )]])

        with temporary_value(
                (other_qb.ro_freq, self.ro_freq() - self.ro_mod_freq()),
                (other_qb.acq_weights_type, 'SSB'),
                (other_qb.acq_length, self.acq_length()),
                (other_qb.instr_trigger.get_instr().pulse_period, trigger_sep),
        ):
            self.prepare(drive=None)
            other_qb.prepare(drive=None)
            MC.set_detector_function(det.IndexDetector(
                other_qb.int_avg_det_spec, 0))
            awg_n = other_qb.instr_acq().get_instr().get_awg_control_object()[1]
            other_qb.instr_pulsar.get_instr().start(exclude=[awg_n])
            MC.run(name='readout_carrier_calibration' + self.msmt_suffix,
                   mode='adaptive')

        a = ma.OptimizationAnalysis(label='readout_carrier_calibration')
        if plot:
            # v2 creates a pretty picture of the optimizations
            ma.OptimizationAnalysis_v2(label='readout_carrier_calibration')

        ch_1_min = a.optimization_result[0][0]
        ch_2_min = a.optimization_result[0][1]
        if update:
            self.ro_I_offset(ch_1_min)
            self.ro_Q_offset(ch_2_min)
        return ch_1_min, ch_2_min

    def calibrate_drive_mixer_carrier_model(self, update=True, trigger_sep=5e-6,
                                            limits=(-0.1, 0.1, -0.1, 0.1),
                                            n_meas=(10, 10), meas_grid=None,
                                            upload=True):
        """Method for calibrating the lo leakage of the drive IQ Mixer

        By applying DC biases on the I and Q inputs of an IQ mixer one can 
        change the bias conditions of the diodes inside the mixer. This can be 
        used to reduce LO leakage. This method measures the LO leakage for 
        different values of DC biases. The subsequent analysis fits an 
        analytical model to the measured data and extracts the settings 
        minimizing the LO leakage.

        Args:
            update (bool, optional): Determines whether the DC biases found from 
                the measurements that minimize the LO leakage 
                are written into the qubit parameters or not. 
                Defaults to True.
            meas_grid (:py:class:'np.array', optional): Grid of points to be 
                measured in form of a Numpy array of shape (2, number of points). 
                The first dimension holding 
                the values for I channel DC biases and the second dimension 
                holding the Q channel DC biases. Both in volts. If no meas_grid 
                is provided a uniform grid is generated using n_meas and limits. 
                Defaults to None.
            n_meas (int or tuple, optional): Tuple, list or 1D array of 
                length 2 that determines the number of measurement points in 
                case meas_grid is not provided. If an integer is provided the 
                input will be transformed to a list n_meas = (n_meas, n_meas).
                n_meas[0] = points in V_I.
                n_meas[1] = points in V_Q.
                Defaults to (10, 10).
            trigger_sep (float, optional): Seperation time in s between trigger
                signals. Defaults to 5e-6 s.
            limits (tuple, optional): Tuple, list or 1D array of length 4 
                holding the limits of the measurement grid in case 
                meas_grid is not provided. Ordered as follows
                (min bias I, max bias I, min bias Q, max bias Q)
                Units: Volts
                Defaults to (-0.1, 0.1, -0.1, 0.1).

        Returns:
            V_I (float): DC bias on I channel that minimizes LO leakage.
            V_Q (float): DC bias on Q channel that minimizes LO leakage.
            ma (:py:class:~'pycqed.timedomain_analysis.MixerCarrierAnalysis'): 
                The MixerCarrierAnalysis object.
        """
        MC = self.instr_mc.get_instr()
        if meas_grid is None:
            if not len(limits) == 4:
                log.error('Input variable `limits` in function call '
                          '`calibrate_drive_mixer_carrier_model` needs to be a list '
                          'or 1D array of length 4.\nFound length '
                          '{} object instead!'.format(len(limits)))
            if isinstance(n_meas, int):
                n_meas = (n_meas, n_meas)
            elif not len(n_meas) == 2:
                log.error('Input variable `n_meas` in function call '
                          '`calibrate_drive_mixer_carrier_model` needs to be a list, '
                          'tuple or 1D array of length 2.\nFound length '
                          '{} object instead!'.format(len(n_meas)))
            meas_grid = np.meshgrid(np.linspace(limits[0], limits[1], n_meas[0]), 
                                    np.linspace(limits[2], limits[3], n_meas[1]))
            meas_grid = np.array([meas_grid[0].flatten(), meas_grid[1].flatten()])    
        else:
            limits = []
            limits.append(np.min(meas_grid[0, :]))
            limits.append(np.max(meas_grid[0, :]))
            limits.append(np.min(meas_grid[1, :]))
            limits.append(np.max(meas_grid[1, :]))
        
        # Check that bounds of measurement grid are reasonable and do not exceed
        # 1 V as this might damage the diodes inside the mixers.
        if np.max(np.abs(meas_grid)) > 1.0:
            log.error('Measurement grid contains DC amplitudes above 1 V. '
                      'Too high DC biases can potentially damage the diodes'
                      'inside the mixer. \n'
                      'Maximum amplitude is {:.2f} mV!'.format(1e3*np.max(meas_grid)))

        chI_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_I_channel())]
        chQ_par = self.instr_pulsar.get_instr().parameters['{}_offset'.format(
            self.ge_Q_channel())]
        MC.set_sweep_functions([chI_par, chQ_par])
        MC.set_sweep_points(meas_grid.T)

        exp_metadata = {'qb_names': [self.name], 'rotate': False, 
                        'cal_points': f"CalibrationPoints(['{self.name}'], [])"}
        with temporary_value(
                (self.ro_freq, self.ge_freq() - self.ge_mod_freq()),
                (self.acq_weights_type, 'SSB'),
                (self.instr_trigger.get_instr().pulse_period, trigger_sep),
                (chI_par, chI_par()),  # for automatic reset after the sweep
                (chQ_par, chQ_par()),  # for automatic reset after the sweep
                *self._drive_mixer_calibration_tmp_vals()
        ):
            if upload:
                sq.pulse_list_list_seq([[self.get_acq_pars(), dict(
                    pulse_type='GaussFilteredCosIQPulse',
                    pulse_length=self.acq_length(),
                    ref_point='start',
                    amplitude=0,
                    I_channel=self.ge_I_channel(),
                    Q_channel=self.ge_Q_channel(),
                )]])

            self.prepare(drive='timedomain', switch='calib')
            MC.set_detector_function(self.int_avg_det_spec)
            awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
            self.instr_pulsar.get_instr().start(exclude=[awg_name])
            MC.run(name='drive_carrier_calibration' + self.msmt_suffix,
                   exp_metadata=exp_metadata)

        a = tda.MixerCarrierAnalysis()
        analysis_params_dict = a.proc_data_dict['analysis_params_dict']

        ch_I_min = analysis_params_dict['V_I']
        ch_Q_min = analysis_params_dict['V_Q']

        if(ch_I_min < limits[0] or ch_I_min > limits[1]):
            log.warning('Optimum for DC bias voltage I channel is outside '
                        'the measured range and no settings will be updated. '
                        'Best V_I according to fitting: {:.2f} mV'.format(ch_I_min*1e3))
            update = False
        if(ch_Q_min < limits[2] or ch_Q_min > limits[3]):
            log.warning('Optimum for DC bias voltage Q channel is outside '
                        'the measured range and no settings will be updated. '
                        'Best V_Q according to fitting: {:.2f} mV'.format(ch_Q_min*1e3))
            update = False

        if update:
            self.ge_I_offset(ch_I_min)
            self.ge_Q_offset(ch_Q_min)
            chI_par(ch_I_min)
            chQ_par(ch_Q_min)

        return ch_I_min, ch_Q_min, a

    def calibrate_drive_mixer_skewness(self, update=True, amplitude=0.5,
                                       trigger_sep=5e-6, no_improv_break=50,
                                       initial_stepsize=(0.15, 10)):
        MC = self.instr_mc.get_instr()
        ad_func_pars = {'adaptive_function': opti.nelder_mead,
                        'x0': [self.ge_alpha(), self.ge_phi_skew()],
                        'initial_step': initial_stepsize,
                        'no_improv_break': no_improv_break,
                        'minimize': True,
                        'maxiter': 500}
        MC.set_sweep_functions([self.ge_alpha, self.ge_phi_skew])
        MC.set_adaptive_function_parameters(ad_func_pars)

        with temporary_value(
            (self.ge_alpha, self.ge_alpha()),
            (self.ge_phi_skew, self.ge_phi_skew()),
            (self.ro_freq, self.ge_freq() - 2*self.ge_mod_freq()),
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
            *self._drive_mixer_calibration_tmp_vals()
        ):
            self.prepare(drive='timedomain', switch='calib')
            detector = self.int_avg_det_spec
            detector.always_prepare = True
            detector.AWG = self.instr_pulsar.get_instr()
            detector.prepare_function = lambda \
                alphaparam=self.ge_alpha, skewparam=self.ge_phi_skew: \
                    sq.pulse_list_list_seq([[self.get_acq_pars(), dict(
                            pulse_type='GaussFilteredCosIQPulse',
                            pulse_length=self.acq_length(),
                            ref_point='start',
                            amplitude=amplitude,
                            I_channel=self.ge_I_channel(),
                            Q_channel=self.ge_Q_channel(),
                            mod_frequency=self.ge_mod_freq(),
                            phase_lock=False,
                            alpha=alphaparam(),
                            phi_skew=skewparam(),
                        )]])
            MC.set_detector_function(det.IndexDetector(detector, 0))
            MC.run(name='drive_skewness_calibration' + self.msmt_suffix,
                   mode='adaptive')

        a = ma.OptimizationAnalysis(label='drive_skewness_calibration')
        # v2 creates a pretty picture of the optimizations
        ma.OptimizationAnalysis_v2(label='drive_skewness_calibration')

        # phi and alpha are the coefficients that go in the predistortion matrix
        alpha = a.optimization_result[0][0]
        phi = a.optimization_result[0][1]
        if update:
            self.ge_alpha(alpha)
            self.ge_phi_skew(phi)
        return alpha, phi

    def calibrate_drive_mixer_skewness_model(
            self, update=True, meas_grid=None, n_meas=(10, 10),
            amplitude=0.1, trigger_sep=5e-6, limits=(0.9, 1.1, -20, 20),
            force_ro_mod_freq=False, **kwargs):
        """Method for calibrating the sideband suppression of the drive IQ Mixer

        The two settings that are used to calibrate the suppression of the 
        unwanted sideband are the amplitude ratio and phase between I and Q. 
        This method measures the sideband suppression for different values of 
        these two settings that are either handed over as meas_grid or generated 
        automatically. The subsequent analysis fits an analytical model to the 
        measured data and extracts the settings minimizing the amplitude of the 
        sideband.

        Args:
            update (bool, optional): Determines whether the setting found from 
                the measurements that are supposed to minimize the sideband 
                suppression are written into the qubit parameters or not. 
                Defaults to True.
            meas_grid (:py:class:'np.array', optional): Grid of points to be 
                measured in form
                of a np.array of shape (2, #points). The first dimension holding 
                the values for the amplitude ratio and the second dimension 
                holding the phi_skew values in degrees. If no meas_grid is 
                provided a uniform grid is generated using n_meas and limits. 
                Defaults to None.
            n_meas (tuple, optional): Tuple, list or 1D array of length 2 that
                determines the number of measurement points in case meas_grid is
                not provided. 
                n_meas[0] = points in amplitude ratio.
                n_meas[1] = points in phi_skew.
                Defaults to (10, 10).
            amplitude (float, optional): Amplitude of the IF signal in V applied
                to the mixer during the measurement. Defaults to 0.1 V.
            trigger_sep (float, optional): Seperation time in s between trigger
                signals. Defaults to 5e-6 s.
            limits (tuple, optional): Tuple, list or 1D array of length 4 
                holding the limits of the measurement grid in case 
                meas_grid is not provided. Ordered as follows
                (min ampl. ratio, max ampl. ratio, min phi_skew, max phi_skew)
                Units: (None, None, deg, deg)
                Defaults to (0.9, 1.1, -20, 20).
            force_ro_mod_freq (bool, optional): Whether to force the current
                ro_mod_freq setting even though it results in non
                commensurable LO frequencies for the specified trigger_sep.
                Defaults to false.

        Returns:
            alpha (float): The amplitude ratio that maximizes the suppression of 
                the unwanted sideband.
            phi_skew (float): The phi_skew that maximizes the suppression of 
                the unwanted sideband.
            ma (:py:class:~'pycqed.timedomain_analysis.MixerSkewnessAnalysis'): 
                The MixerSkewnessAnalysis object.
        """
        if meas_grid is None:
            if not len(limits) == 4:
                log.error('Input variable `limits` in function call '
                          '`calibrate_drive_mixer_skewness_model` needs to be a list '
                          'or 1D array of length 4.\nFound length '
                          '{} object instead!'.format(len(limits)))
            if isinstance(n_meas, int):
                n_meas = [n_meas, n_meas]
            elif not len(n_meas) == 2:
                log.error('Input variable `n_meas` in function call '
                          '`calibrate_drive_mixer_skewness_model` needs to be a list, '
                          'tuple or 1D array of length 2.\nFound length '
                          '{} object instead!'.format(len(n_meas)))
            meas_grid = np.meshgrid(np.linspace(limits[0], limits[1], n_meas[0]), 
                                    np.linspace(limits[2], limits[3], n_meas[1]))
            meas_grid = np.array([meas_grid[0].flatten(), meas_grid[1].flatten()])    
        else:
            limits = []
            limits.append(np.min(meas_grid[0, :]))
            limits.append(np.max(meas_grid[0, :]))
            limits.append(np.min(meas_grid[1, :]))
            limits.append(np.max(meas_grid[1, :]))

        MC = self.instr_mc.get_instr()

        exp_metadata = {'qb_names': [self.name], 'rotate': False,
                        'cal_points': f"CalibrationPoints(['{self.name}'], [])"}

        with temporary_value(
            (self.ro_freq, self.ge_freq() - 2*self.ge_mod_freq()),
            (self.ro_mod_freq, self.ro_mod_freq()), # for automatic reset
            (self.acq_weights_type, 'SSB'),
            (self.instr_trigger.get_instr().pulse_period, trigger_sep),
            *self._drive_mixer_calibration_tmp_vals()
        ):
            pulse_list_list = []
            acq_pars = self.get_acq_pars()
            for alpha, phi_skew in meas_grid.T:
                pulse_list_list.append([deepcopy(acq_pars), dict(
                            pulse_type='GaussFilteredCosIQPulse',
                            pulse_length=self.acq_length(),
                            ref_point='start',
                            amplitude=amplitude,
                            I_channel=self.ge_I_channel(),
                            Q_channel=self.ge_Q_channel(),
                            mod_frequency=self.ge_mod_freq(),
                            phase_lock=False,
                            alpha=alpha,
                            phi_skew=phi_skew,
                        )])
            seq = sq.pulse_list_list_seq(pulse_list_list)

            self.prepare(drive='timedomain', switch='calib')

            # Check commensurability of LO frequencies with trigger sep.
            ro_lo_freq = self.get_ro_lo_freq()
            dr_lo_freq = self.ge_freq() - self.ge_mod_freq()
            # Frequency of the LO phases is given by the LOs beat frequency.
            beat_freq = 0.5*(dr_lo_freq - ro_lo_freq)
            #         = 0.5*(ge_mod_freq + ro_mod_freq) in our case
            beats_per_trigger = np.round(beat_freq * trigger_sep,
                                         int(np.floor(np.log10(1/trigger_sep)))+2)
            if not beats_per_trigger.is_integer():
                log.warning('Difference of RO LO and drive LO frequency '
                            'resulting from the chosen modulation frequencies '
                            'is not an integer multiple of the trigger '
                            'seperation.')
                if not force_ro_mod_freq:
                    if self.ro_fixed_lo_freq() is not None:
                        log.warning(
                            'Automatic adjustment of the RO IF might lead to '
                            'wrong results since ro_fixed_lo_freq is set.')
                    beats_per_trigger = int(beats_per_trigger + 0.5)
                    # FIXME: changing the IF here is probably the wrong moment
                    #  because the pulse seq has already been created above.
                    self.ro_mod_freq(2 * beats_per_trigger/trigger_sep \
                                     - self.ge_mod_freq())
                    log.warning('To ensure commensurability the RO ' 
                                'modulation frequency will temporarily be set '
                                'to {} Hz.'.format(self.ro_mod_freq()))
                    self.prepare(drive='timedomain', switch='calib')

            s1 = awg_swf.SegmentHardSweep(sequence=seq,
                                          parameter_name=r'Amplitude ratio, $\alpha$',
                                          unit='')
            s1.name = 'Amplitude ratio hardware sweep'
            s2 = awg_swf.SegmentHardSweep(sequence=seq,
                                          parameter_name=r'Phase skew, $\phi$',
                                          unit='deg')
            s2.name = 'Phase skew hardware sweep'
            MC.set_sweep_functions([s1, s2])
            MC.set_sweep_points(meas_grid.T)
            MC.set_detector_function(self.int_avg_det)
            MC.run(name='drive_skewness_calibration' + self.msmt_suffix,
                   exp_metadata=exp_metadata)

        a = tda.MixerSkewnessAnalysis()
        analysis_params_dict = a.proc_data_dict['analysis_params_dict']

        _alpha = analysis_params_dict['alpha']
        _phi = analysis_params_dict['phase']

        if(_alpha < limits[0] or _alpha > limits[1]):
            log.warning('Optimum for amplitude ratio is outside '
                        'the measured range and no settings will be updated. '
                        'Best alpha according to fitting: {:.2f}'.format(_alpha))
            update = False
        if(_phi < limits[2] or _phi > limits[3]):
            log.warning('Optimum for phase correction is outside '
                        'the measured range and no settings will be updated. '
                        'Best phi according to fitting: {:.2f} deg'.format(_phi))
            update = False

        if update:
            self.ge_alpha(_alpha)
            self.ge_phi_skew(_phi)

        return _alpha, _phi, a

    def find_qubit_frequency(self, freqs, method='cw_spectroscopy',
                             update=False, trigger_separation=3e-6,
                             close_fig=True, analyze_ef=False, analyze=True,
                             upload=True, label=None, **kw):
        """
        WARNING: Does not automatically update the qubit frequency parameter.
        Set update=True if you want this!

        Args:
            method:                   the spectroscopy type; options: 'pulsed',
                                      'spectrsocopy'
            update:                   whether to update the relevant qubit
                                      parameters with the found frequency(ies)
            MC:                       the measurement control object
            close_fig:                whether or not to close the figure
            analyze_ef:               whether or not to also look for the gf/2

        Keyword Args:
            interactive_plot:        (default=False)
                whether to plot with plotly or not
            analyze_ef:              (default=False)
                whether to look for another f_ge/2 peak/dip
            percentile:              (default=20)
                percentile of the data that is considered background noise
            num_sigma_threshold:     (default=5)
                used to define the threshold above(below) which to look for
                peaks(dips); threshold = background_mean +
                num_sigma_threshold * background_std
            window_len              (default=3)
                filtering window length; uses a_tools.smooth
            analysis_window         (default=10)
                how many data points (calibration points) to remove before
                sending data to peak_finder; uses a_tools.cut_edges,
                data = data[(analysis_window//2):-(analysis_window//2)]
            amp_only                (default=False)
                whether only I data exists
            save_name               (default='Source Frequency')
                figure name with which it will be saved
            auto                    (default=True)
                automatically perform the entire analysis upon call
            label                   (default=none?)
                label of the analysis routine
            folder                  (default=working folder)
                working folder
            NoCalPoints             (default=4)
                number of calibration points
            print_fit_results       (default=True)
                print the fit report
            print_frequency         (default=False)
                whether to print the f_ge and f_gf/2
            make_fig          {default=True)
                    whether or not to make a figure
            show                    (default=True)
                show the plots
            show_guess              (default=False)
                plot with initial guess values
            close_file              (default=True)
                close the hdf5 file

        Returns:
            the peak frequency(ies).
        """
        if not update:
            log.warning("Does not automatically update the qubit "
                            "frequency parameter. "
                            "Set update=True if you want this!")
        if np.any(freqs<500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if freqs is None:
            f_span = kw.get('f_span', 100e6)
            f_mean = kw.get('f_mean', self.f_qubit())
            nr_points = kw.get('nr_points', 100)
            if f_mean == 0:
                log.warning("find_frequency does not know where to "
                                "look for the qubit. Please specify the "
                                "f_mean or the freqs function parameter.")
                return 0
            else:
                freqs = np.linspace(f_mean - f_span/2, f_mean + f_span/2,
                                    nr_points)

        if 'pulse' not in method.lower():
            if label is None:
                label = 'spectroscopy' + self.msmt_suffix
            if analyze_ef:
                label = 'high_power_' + label

            self.measure_qubit_spectroscopy(freqs, pulsed=False,
                                      trigger_separation=trigger_separation,
                                      label=label, close_fig=close_fig)
        else:
            if label is None:
                label = 'pulsed_spec' + self.msmt_suffix
            if analyze_ef:
                label = 'high_power_' + label

            self.measure_qubit_spectroscopy(freqs, pulsed=True, label=label,
                                      close_fig=close_fig, upload=upload)


        if analyze:
            SpecA = ma.Qubit_Spectroscopy_Analysis(
                qb_name=self.name,
                analyze_ef=analyze_ef,
                label=label,
                close_fig=close_fig, **kw)

            f0 = SpecA.fitted_freq
            if update:
                if not analyze_ef:
                    self.ge_freq(f0)
                else:
                    f0_ef = 2*SpecA.fitted_freq_gf_over_2 - f0
                    self.ef_freq(f0_ef)
            if analyze_ef:
                return f0, f0_ef
            else:
                return f0
        else:
            return

    def find_readout_frequency(self, freqs=None, update=False, MC=None,
                               qutrit=False, **kw):
        """
        Find readout frequency at which contrast between the states of the
        qubit is the highest.
        You need a working pi-pulse for this to work, as well as a pi_ef
        pulse if you intend to use `for_3_level_ro`. Also, if your
        readout pulse length is much longer than the T1, the results will not
        be nice as the excited state spectrum will be mixed with the ground
        state spectrum.

        Args:
            freqs: frequencies to sweep
            qutrit (bool): find optimal frequency for 3-level readout.
                                    Default is False.
            **kw:

        Returns:

        """
        # FIXME: Make proper analysis class for this (Ants, 04.12.2017)
        if not update:
            log.info("Does not automatically update the RO resonator "
                         "parameters. Set update=True if you want this!")
        if freqs is None:
            if self.f_RO() is not None:
                f_span = kw.pop('f_span', 20e6)
                fmin = self.f_RO() - f_span
                fmax = self.f_RO() + f_span
                n_freq = kw.pop('n_freq', 401)
                freqs = np.linspace(fmin, fmax, n_freq)
            else:
                raise ValueError("Unspecified frequencies for find_resonator_"
                                 "frequency and no previous value exists")
        if np.any(freqs < 500e6):
            log.warning('Some of the values in the freqs array might be '
                            'too small. The units should be Hz.')
        if MC is None:
            MC = self.instr_mc.get_instr()

        levels = ('g', 'e', 'f') if qutrit else ('g', 'e')

        self.measure_dispersive_shift(freqs, states=levels, analyze=False)
        labels = {l: '{}-spec'.format(l) + self.msmt_suffix for l in levels}
        m_a = {l: ma.MeasurementAnalysis(label=labels[l]) for l in levels}
        trace = {l: m_a[l].measured_values[0] *
                    np.exp(1j * np.pi * m_a[l].measured_values[1] / 180.)
                 for l in levels}
        # FIXME: make something that doesn't require a conditional branching
        if qutrit:
            total_dist = np.abs(trace['e'] - trace['g']) + \
                         np.abs(trace['f'] - trace['g']) + \
                         np.abs(trace['f'] - trace['e'])
        else:
            total_dist = np.abs(trace['e'] - trace['g'])
        fmax = freqs[np.argmax(total_dist)]
        # Plotting which works for qubit or qutrit
        fig, ax = plt.subplots(2)
        ax[0].plot(freqs, np.abs(trace['g']), label='g')
        ax[0].plot(freqs, np.abs(trace['e']), label='e')
        if qutrit:
            ax[0].plot(freqs, np.abs(trace['f']), label='f')
        ax[0].set_ylabel('Amplitude')
        ax[0].legend()
        ax[1].plot(freqs, np.abs(trace['e'] - trace['g']), label='eg')
        if qutrit:
            ax[1].plot(freqs, np.abs(trace['f'] - trace['g']), label='fg')
            ax[1].plot(freqs, np.abs(trace['e'] - trace['f']), label='ef')
        ax[1].plot(freqs, total_dist, label='total distance')
        ax[1].set_xlabel("Freq. [Hz]")
        ax[1].set_ylabel('Distance in IQ plane')
        ax[0].set_title(f"Current RO_freq: {self.ro_freq()} Hz" + "\n"
                        + f"Optimal Freq: {fmax} Hz")
        plt.legend()
        # Save figure into 'g' measurement folder
        m_a['g'].save_fig(fig, 'IQplane_distance')

        if kw.get('analyze', True):
            sa.ResonatorSpectroscopy_v2(labels=[l for l in labels.values()])
        else:
            fmax = freqs[np.argmax(np.abs(trace['e'] - trace['g']))]

        log.info("Optimal RO frequency to distinguish states {}: {} Hz"
                     .format(levels, fmax))

        if kw.get('analyze', True):
            SA = sa.ResonatorSpectroscopy(t_start=[m_a['g'].timestamp_string,
                                                   m_a['e'].timestamp_string],
                                          options_dict=dict(simultan=True,
                                                            fit_options=dict(
                                                            model='hanger_with_pf'),
                                                            scan_label=''),
                                          do_fitting=True)
            # FIXME Nathan: remove 3 level dependency; fix this analysis:
            # if qutrit:
            #     SA2 = sa.ResonatorSpectroscopy(t_start=m_a['f'].timestamp_string,
            #                               options_dict=dict(simultan=False,
            #                                                 fit_options = dict(
            #                                                 model='hanger_with_pf'),
            #                                                 scan_label=''),
            #                               do_fitting=True)

            if update:
                # FIXME Nathan: update parameters accordingly
                self.ro_freq(SA.f_RO if not qutrit else fmax)
                self.chi(SA.chi)
                self.f_RO_resonator(SA.f_RO_res)
                self.f_RO_purcell(SA.f_PF)
                self.RO_purcell_kappa(SA.kappa)
                self.RO_J_coupling(SA.J_)
                if kw.pop('get_CLEAR_params', False):
                    if self.ro_CLEAR_segment_length is None:
                        self.ro_CLEAR_segment_length = self.ro_length/10
                    if kw.get('max_amp_difference', False) :
                        '''this gives the ratio of the maximal hight for'''
                        '''the segments to the base amplitude'''
                        max_diff = kw.pop('max_amp_difference')
                    else:
                        max_diff = 3
                    self.ro_CLEAR_delta_amp_segment = \
                        sim_CLEAR.get_CLEAR_amplitudes(
                            self.f_RO_purcell, self.f_RO_resonator,
                            self.ro_freq, self.RO_purcell_kappa,
                            self.RO_J_coupling, self.chi, 1, self.ro_length,
                            length_segments=self.ro_CLEAR_segment_length,
                            sigma=self.ro_sigma,
                            max_amp_diff=max_diff) * self.ro_amp


    def measure_dispersive_shift(self, freqs, analyze=True, close_fig=True,
                                 upload=True, states=('g','e'), prep_params=None):
        """ Varies the frequency of the microwave source to the resonator and
        measures the transmittance """

        if freqs is None:
            raise ValueError("Unspecified frequencies for "
                             "measure_resonator_spectroscopy")
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array '
                         'might be too small. The units should be Hz.'))
        if prep_params is None:
            prep_params = self.preparation_params()

        assert isinstance(states, tuple), \
            "states should be a tuple, not {}".format(type(states))

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()

        for state in states:
            sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name,
                    state=state, prep_params=prep_params, upload=upload)

            MC.set_sweep_function(self.swf_ro_freq_lo())
            MC.set_sweep_points(freqs)
            MC.set_detector_function(self.int_avg_det_spec)

            awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
            self.instr_pulsar.get_instr().start(exclude=[awg_name])
            MC.run(name=f"{state}-spec" + self.msmt_suffix)
            self.instr_pulsar.get_instr().stop()

            if analyze:
                ma.MeasurementAnalysis(auto=True, close_fig=close_fig,
                                    qb_name=self.name)

    def measure_flux_pulse_timing(self, delays, analyze, label=None, **kw):
        if self.instr_ge_lo() is None:
            raise NotImplementedError("qb.measure_flux_pulse_timing is not "
                                      "implemented for setups without ge LO. "
                                      "Use quantum experiment FluxPulseTiming "
                                      "instead.")
        if label is None:
            label = 'Flux_pulse_timing_{}'.format(self.name)
        self.measure_flux_pulse_scope([self.ge_freq()], delays,
                                      label=label, analyze=False, **kw)
        if analyze:
            tda.FluxPulseTimingAnalysis(qb_names=[self.name])

    def measure_flux_pulse_scope(self, freqs, delays, cz_pulse_name=None,
                                 analyze=True, cal_points=True,
                                 upload=True, label=None,
                                 n_cal_points_per_state=2, cal_states='auto',
                                 prep_params=None, exp_metadata=None, **kw):
        '''
        flux pulse scope measurement used to determine the shape of flux pulses
        set up as a 2D measurement (delay and drive pulse frequecy are
        being swept)
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |


        Args:
            freqs (numpy array): array of drive frequencies
            delays (numpy array): array of delays of the drive pulse w.r.t
                the flux pulse
            pulse_length (float): flux pulse length (if not specified, the
                                    self.flux_pulse_length() is taken)
            pulse_amp (float): flux pulse amplitude  (if not specified, the
                                    self.flux_pulse_amp() is taken)
            pulse_delay (float): flux pulse delay
            MC (MeasurementControl): if None, then the self.MC is taken

        Returns: None

        '''
        if self.instr_ge_lo() is None:
            raise NotImplementedError('qb.measure_flux_pulse_scope is '
                                      'not implemented for setups '
                                      'without external drive LO. Use '
                                      'FluxPulseScope class instead!')

        if label is None:
            label = 'Flux_scope_{}'.format(self.name)
        MC = self.instr_mc.get_instr()
        self.prepare(drive='timedomain')

        if cz_pulse_name is None:
            cz_pulse_name = 'FP ' + self.name

        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                self.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None
        if prep_params is None:
            prep_params = self.preparation_params()

        op_dict = kw.pop('operation_dict', self.get_operation_dict())

        seq, sweep_points, sweep_points_2D = \
            fsqs.fluxpulse_scope_sequence(
                delays=delays, freqs=freqs, qb_name=self.name,
                operation_dict=op_dict,
                cz_pulse_name=cz_pulse_name, cal_points=cp,
                prep_params=prep_params, upload=False, **kw)
        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Delay', unit='s'))
        MC.set_sweep_points(sweep_points)
        MC.set_sweep_function_2D(swf.Offset_Sweep(
            self.instr_ge_lo.get_instr().frequency,
            -self.ge_mod_freq(),
            name='Drive frequency',
            parameter_name='Drive frequency', unit='Hz'))
        MC.set_sweep_points_2D(sweep_points_2D)
        det_func = self.int_avg_det
        MC.set_detector_function(det_func)
        sweep_points = SweepPoints('delay', delays, unit='s',
                                   label=r'delay, $\tau$', dimension=0)
        sweep_points.add_sweep_parameter('freq', freqs, unit='Hz',
                                         label=r'drive frequency, $f_d$',
                                         dimension=1)
        mospm = {self.name: ['delay', 'freq']}
        if exp_metadata is None:
            exp_metadata = {}
        exp_metadata.update({'sweep_points_dict': {self.name: delays},
                             'sweep_points_dict_2D': {self.name: freqs},
                             'sweep_points': sweep_points,
                             'meas_obj_sweep_points_map': mospm,
                             'meas_obj_value_names_map':
                                 {self.name: det_func.value_names},
                             'use_cal_points': cal_points,
                             'preparation_params': prep_params,
                             'cal_points': repr(cp),
                             'rotate': cal_points,
                             'data_to_fit': {self.name: 'pe'},
                             "sweep_name": "Delay",
                             "sweep_unit": "s"})
        MC.run_2D(label, exp_metadata=exp_metadata)

        if analyze:
            try:
                tda.FluxPulseScopeAnalysis(
                    qb_names=[self.name],
                    options_dict=dict(TwoD=True, rotation_type='global_PCA'))
            except Exception:
                ma.MeasurementAnalysis(TwoD=True)

    def measure_T2_freq_sweep(self, flux_lengths=None, n_pulses=None,
                              cz_pulse_name=None,
                              freqs=None, amplitudes=None, phases=[0,120,240],
                              analyze=True, cal_states='auto', cal_points=False,
                              upload=True, label=None, n_cal_points_per_state=2,
                              exp_metadata=None):
        """
        Flux pulse amplitude measurement used to determine the qubits energy in
        dependence of flux pulse amplitude.

        2 sorts of sequences can be generated based on the combination of
        (flux_lengths, n_pulses):
        1. (None, array):
         The ith created pulse sequence is:
        |          ---|X90|  ---------------------------------|X90||RO|
        |          --------(| - fp -| ) x n_pulses[i] ---------
       Each flux pulse has a duration equal to the stored value in the
       operations dict. Note that in this case, the flux_lengths stored in
       the metadata (and hence used by the default analysis) is
       fpl * n_pulses, where fpl is the flux pulse length stored in the
       cz_pulse_name operation, i.e. the total time spent away from sweetspot
       (but it does not account for buffer times before and after each pulse,
       which will however be in the sequence).
        2. (array, None):
        The ith created pulse sequence is:
        |          ---|X90|  ---------------------------------|X90||RO|
       |          --------| -- fp --length=flux_lengths[i]----|
       and the duration of the single flux pulse is adapted according to
       the values specified in flux_lengths

        Args:
            flux_lengths (array):  array containing the flux pulse durations.
                Used if n_pulses is None.
            n_pulses (array): array containing the number of flux pulses. Used
                if flux_lengths is None.
            cz_pulse_name: name of the flux pulse
            freqs: array of drive frequencies (from which the flux pulse
            amplitudes are inferred)
            amplitudes: array of amplitudes of the flux pulse
            phases (array, list): array of phases for the second pi-half pulse
                for the Ramsey experiment
            analyze:
            cal_states:
            cal_points:
            upload:
            label:
            n_cal_points_per_state:
            exp_metadata:

        Returns:

        """
        fit_paras = deepcopy(self.fit_ge_freq_from_flux_pulse_amp())
        if freqs is not None:
            amplitudes = fit_mods.Qubit_freq_to_dac(freqs, **fit_paras)

        amplitudes = np.array(amplitudes)

        if cz_pulse_name is None:
            cz_pulse_name = 'FP ' + self.name

        if np.any((amplitudes > abs(fit_paras['dac_sweet_spot']))):
            amplitudes -= fit_paras['V_per_phi0']
        elif np.any((amplitudes < -abs(fit_paras['dac_sweet_spot']))):
            amplitudes += fit_paras['V_per_phi0']

        if np.any((amplitudes > abs(fit_paras['V_per_phi0']) / 2)):
            amplitudes -= fit_paras['V_per_phi0']
        elif np.any((amplitudes < -abs(fit_paras['V_per_phi0']) / 2)):
            amplitudes += fit_paras['V_per_phi0']

        if np.any(np.isnan(amplitudes)):
            raise ValueError('Specified frequencies resulted in nan amplitude. '
                             'Check frequency range!')

        if amplitudes is None:
            raise ValueError('Either freqs or amplitudes need to be specified')

        if label is None:
            label = 'T2_Frequency_Sweep_{}'.format(self.name)
        MC = self.instr_mc.get_instr()
        self.prepare(drive='timedomain')

        amplitudes = np.array(amplitudes)
        if flux_lengths is not None:
            flux_lengths = np.array(flux_lengths)
        phases = np.array(phases)


        if cal_points:
            cal_states = CalibrationPoints.guess_cal_states(cal_states)
            cp = CalibrationPoints.single_qubit(
                self.name, cal_states, n_per_state=n_cal_points_per_state)
        else:
            cp = None

        seq, sweep_points = \
            fsqs.T2_freq_sweep_seq(
                amplitudes=amplitudes, qb_name=self.name,
                n_pulses=n_pulses,
                operation_dict=self.get_operation_dict(),
                flux_lengths=flux_lengths, phases = phases,
                cz_pulse_name=cz_pulse_name, upload=False, cal_points=cp)
        MC.set_sweep_function(awg_swf.SegmentHardSweep(
            sequence=seq, upload=upload, parameter_name='Amplitude', unit='V'))
        MC.set_sweep_points(sweep_points)
        MC.set_detector_function(self.int_avg_det)
        if exp_metadata is None:
            exp_metadata = {}
        # for legacy reason, store flux lengths in metadata even if n_pulses
        # were used to determine the flux lengths, such that the analysis can
        # easily access them
        if flux_lengths is None and n_pulses is not None:
            flux_lengths = np.array(n_pulses) * \
                           self.get_operation_dict()[cz_pulse_name]['pulse_length']
        exp_metadata.update({'amplitudes': amplitudes,
                             'frequencies': freqs,
                             'phases': phases,
                             'flux_lengths': flux_lengths,
                             'n_pulses': n_pulses,
                             'use_cal_points': cal_points,
                             'cal_points': repr(cp),
                             'rotate': cal_points,
                             'data_to_fit': {self.name: 'pe'},
                             "rotation_type": 'global_PCA' if not cal_points \
                                 else 'cal_states'})
        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            try:
                tda.T2FrequencySweepAnalysis(qb_names=[self.name],
                                             options_dict=dict(TwoD=False))
            except Exception:
                ma.MeasurementAnalysis(TwoD=False)

    def get_closest_lo_freq(self, target_lo_freq, fixed_lo='default',
                            operation=None):
        """Get the closest allowed LO freq for given target LO freq.

        Args:
            target_lo_freq (float): the target Lo freq
            fixed_lo: specification of the allowed LO freq(s), can be:
                - None: no restrictions on the LO freq
                - float: LO fixed to a single freq
                - str: (operation must be provided in this case)
                    - 'default' (default value): use the setting in the qubit
                      object.
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

    def configure_offsets(self, set_ro_offsets=True, set_ge_offsets=True):
        """
        Set AWG channel DC offsets and switch sigouts on.

        :param set_ro_offsets: whether to set offsets for RO channels
        :param set_ge_offsets: whether to set offsets for drive channels
        """
        pulsar = self.instr_pulsar.get_instr()
        offset_list = []
        if set_ro_offsets:
            offset_list += [('ro_I_channel', 'ro_I_offset')]
            if self.ro_Q_channel() is not None:
                offset_list += [('ro_Q_channel', 'ro_Q_offset')]
        if set_ge_offsets:
            ge_lo = self.instr_ge_lo
            if self.ge_lo_leakage_cal()['mode'] == 'fixed':
                offset_list += [('ge_I_channel', 'ge_I_offset'),
                                ('ge_Q_channel', 'ge_Q_offset')]
                if ge_lo() is not None and 'lo_cal_data' in ge_lo.get_instr().parameters:
                    ge_lo.get_instr().lo_cal_data().pop(self.name + '_I', None)
                    ge_lo.get_instr().lo_cal_data().pop(self.name + '_Q', None)
            elif ge_lo() is not None:
                # FIXME: configure lo.lo_cal_interp_kind based on a new setting in
                #  the qubit, e.g. self.ge_lo_leakage_cal()['interp_kind']
                lo_cal = ge_lo.get_instr().lo_cal_data()
                qb_lo_cal = self.ge_lo_leakage_cal()
                i_par = pulsar.parameters[self.get('ge_I_channel') + '_offset']
                q_par = pulsar.parameters[self.get('ge_Q_channel') + '_offset']
                lo_cal[self.name + '_I'] = (i_par, qb_lo_cal['freqs'],
                                            qb_lo_cal['I_offsets'])
                lo_cal[self.name + '_Q'] = (q_par, qb_lo_cal['freqs'],
                                            qb_lo_cal['Q_offsets'])

        for channel_par, offset_par in offset_list:
            ch = self.get(channel_par)
            if ch + '_offset' in pulsar.parameters:
                pulsar.set(ch + '_offset', self.get(offset_par))
                pulsar.sigout_on(ch)

    def set_distortion_in_pulsar(self, datadir=None):
        """
        Configures the fluxline distortion in a pulsar object according to the
        settings in the parameter flux_distortion of the qubit object.

        :param pulsar: the pulsar object. If None, self.find_instrument is
            used to find an obejct called 'Pulsar'.
        :param datadir: path to the pydata directory. If None,
            self.find_instrument is used to find an object called 'MC' and
            the datadir of MC is used.
        """
        if not self.flux_pulse_channel():
            return
        pulsar = self.instr_pulsar.get_instr()
        if datadir is None:
            datadir = self.find_instrument('MC').datadir()
        flux_distortion = deepcopy(self.DEFAULT_FLUX_DISTORTION)
        flux_distortion.update(self.flux_distortion())

        filterCoeffs = fl_predist.process_filter_coeffs_dict(
            flux_distortion, datadir=datadir,
            default_dt=1 / pulsar.clock(channel=self.flux_pulse_channel()))

        pulsar.set(f'{self.flux_pulse_channel()}_distortion_dict',
                   filterCoeffs)
        for param in ['distortion', 'charge_buildup_compensation',
                      'compensation_pulse_delay',
                      'compensation_pulse_gaussian_filter_sigma']:
            pulsar.set(f'{self.flux_pulse_channel()}_{param}',
                       flux_distortion[param])

    def get_channels(self, drive=True, ro=True, flux=True):
        """
        Returns (a subset of) channels.
        Args:
            drive (bool): whether or not to include drive channels
            ro (bool): whether or not to include readout channels
            flux (bool): whether or not to include flux channel

        Returns:
            channels (list)
        """
        d = [self.ge_I_channel(), self.ge_Q_channel()] if drive else []
        r = [self.ro_I_channel(), self.ro_Q_channel()] if ro else []
        f = [self.flux_pulse_channel()] if flux else []
        return d + r + f

    def get_channel_map(self, drive=True, ro=True, flux=True):
        """
        Returns a channel map.
        Args:
            drive (bool): whether or not to include drive channels
            ro (bool): whether or not to include readout channels
            flux (bool): whether or not to include flux channel

        Returns:
            channels (dict): key is the qubit name and value is a
            list of channels
        """
        return {self.name: self.get_channels(drive=drive, ro=ro, flux=flux)}
