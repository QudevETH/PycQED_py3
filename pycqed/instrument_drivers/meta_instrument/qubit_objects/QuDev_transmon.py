import logging

log = logging.getLogger(__name__)
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qcodes.instrument.parameter import InstrumentRefParameter, ManualParameter
from qcodes.utils import validators as vals

import pycqed.analysis.fitting_models as fit_mods
import pycqed.analysis_v2.spectroscopy_analysis as sa
import pycqed.measurement.waveform_control.fluxpulse_predistortion as fl_predist
import pycqed.measurement.waveform_control.pulse as bpl
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.instrument_drivers import instrument
from pycqed.instrument_drivers.meta_instrument.MeasurementObject import (
    MeasurementObject,
)
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement import detector_functions as det
from pycqed.measurement import mc_parameter_wrapper
from pycqed.measurement import optimization as opti
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.waveform_control import reset_schemes as reset
from pycqed.utilities.general import add_suffix_to_dict_keys, temporary_value
from pycqed.utilities.math import dbm_to_vp

try:
    import pycqed.simulations.readout_mode_simulations_for_CLEAR_pulse as sim_CLEAR
except ModuleNotFoundError:
    log.warning('"readout_mode_simulations_for_CLEAR_pulse" not imported.')


class QuDev_transmon(MeasurementObject):
    DEFAULT_FLUX_DISTORTION = dict(
        IIR_filter_list=[],
        FIR_filter_list=[],
        scale_IIR=1,
        distortion='off',
        charge_buildup_compensation=True,
        compensation_pulse_delay=100e-9,
        compensation_pulse_gaussian_filter_sigma=0,
    )

    DEFAULT_GE_LO_CALIBRATION_PARAMS = dict(
        mode='fixed',  # or 'freq_dependent'
        freqs=[],
        I_offsets=[],
        Q_offsets=[],
    )

    DEFAULT_TRANSITION_NAMES = ('ge', 'ef')

    _acq_weights_type_aliases = {
        'optimal': 'custom', 'optimal_qutrit': 'custom_2D',
    }
    _ro_pulse_type_vals = ['GaussFilteredCosIQPulse',
                           'GaussFilteredCosIQPulseMultiChromatic',
                           'GaussFilteredCosIQPulseWithFlux']
    _allowed_drive_modes = [None, 'continuous_spec',
                            'continuous_spec_modulated', 'pulsed_spec',
                            'timedomain']

    def __init__(self, name, transition_names=None, **kw):
        super().__init__(name, **kw)

        if transition_names is None:
            transition_names = self.DEFAULT_TRANSITION_NAMES
        self.transition_names = transition_names

        self.add_parameter('instr_ge_lo',
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

        self.add_pulse_parameter('RO', 'ro_flux_channel', 'flux_channel',
                                 initial_value=None, vals=vals.MultiType(
                                     vals.Enum(None), vals.Strings()))
        self.add_pulse_parameter('RO',
                                 'ro_flux_crosstalk_cancellation_key',
                                 'crosstalk_cancellation_key',
                                 vals=vals.Anything(),
                                 initial_value=False)
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
        self.add_pulse_parameter('RO', 'ro_flux_amplitude', 'flux_amplitude',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_extend_start', 'flux_extend_start',
                                 initial_value=20e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_extend_end', 'flux_extend_end',
                                 initial_value=150e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_gaussian_filter_sigma', 'flux_gaussian_filter_sigma',
                                 initial_value=0.5e-9, vals=vals.Numbers())
        self.add_pulse_parameter('RO', 'ro_flux_mirror_pattern',
                                 'flux_mirror_pattern',
                                 initial_value=None, vals=vals.Enum(None,
                                                                    "none",
                                                                    "all",
                                                                    "odd", "even"))

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
                           label='Parameters for frequency vs flux DC '
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
        for tr_name in self.transmon_transition_names:
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
                                         'SSB_DRAG_pulse_cos',
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
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_env_mod_freq',
                                     'env_mod_frequency',
                                     initial_value=0, vals=vals.Numbers(),
                                     docstring='Modulation frequency of the '
                                               'pulse envelope, introducing '
                                               'a frequency shift of the pulse '
                                               'spectrum by the value of this '
                                               'parameter.')
            self.add_pulse_parameter(f'X180{tn}',
                                     f'{tr_name}_cancellation_freq_offset',
                                     'cancellation_frequency_offset',
                                     initial_value=None,
                                     vals=vals.MultiType(
                                         vals.Enum(None), vals.Numbers()),
                                     docstring='Frequency offset of the '
                                               'cancellation dip of the DRAG '
                                               'pulse with respect to the '
                                               'center frequency of the pulse.')
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_phi_skew',
                                     'phi_skew',
                                     initial_value=0,
                                     vals=vals.Numbers())
            self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_alpha',
                                     'alpha',
                                     initial_value=1,
                                     vals=vals.Numbers())
            if tr_name == 'ge':
                # The parameters below will be the same for all transitions
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_I_channel',
                                         'I_channel',
                                         initial_value=None,
                                         vals=vals.Strings())
                self.add_pulse_parameter(f'X180{tn}', f'{tr_name}_Q_channel',
                                         'Q_channel',
                                         initial_value=None,
                                         vals=vals.MultiType(
                                             vals.Enum(None), vals.Strings()))
                self.add_pulse_parameter(
                    f'X180{tn}', f'{tr_name}_mod_freq',
                    'mod_frequency', initial_value=-100e6,
                    set_parser=lambda f, s=self, t=tr_name:
                               s.configure_mod_freqs(t, **{f'{t}_mod_freq': f}),
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
        self.add_parameter('spec_mod_amp', unit='V', initial_value=0.1,
                           parameter_class=ManualParameter,
                           label='IF amplitude in qb spec',
                           docstring='This configures the amplitude of the IF '
                            'tone when doing a modulated qb spectroscopy (not '
                            'bypassing the mixer, used for multi qb spec).')
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

        # DC flux parameters
        self.add_parameter('dc_flux_parameter', initial_value=None,
                           label='QCoDeS parameter to sweep the DC flux',
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


        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                            set_parser=self._validate_preparation_params)

        self.add_submodule('reset', instrument.InstrumentModule(self, 'reset'))

        self.reset.add_parameter('steps', parameter_class=ManualParameter,
                                initial_value=[], vals=vals.Lists())

        self.add_parameter('ge_lo_leakage_cal',
                           parameter_class=ManualParameter,
                           initial_value=self.DEFAULT_GE_LO_CALIBRATION_PARAMS,
                           vals=vals.Dict())

        # switch parameters
        DEFAULT_SWITCH_MODES = OrderedDict({'modulated': {}, 'spec': {},
                                            'calib': {}})
        self.switch_modes.initial_value=DEFAULT_SWITCH_MODES
        self.switch_modes.docstring=(
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

        if "f0g1" in self.transition_names:
            self.add_f0g1_parameters()

    def add_f0g1_parameters(self):
        # f0g1 pulse parameters
        op_name = "f0g1"
        self.add_operation(op_name)
        self.add_pulse_parameter(
            op_name,
            "f0g1_pulse_type",
            "pulse_type",
            initial_value="f0g1Pulse",
            vals=vals.Enum("f0g1Pulse"),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_I_channel",
            "I_channel",
            initial_value=None,
            vals=vals.Strings(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_Q_channel",
            "Q_channel",
            initial_value=None,
            vals=vals.MultiType(vals.Enum(None), vals.Strings()),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_AcStark_IFCoefs",
            "AcStark_IFCoefs",
            initial_value=np.array([0, 0, 0]),
            vals=vals.Arrays(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_AcStark_IFCoefs_error",
            "AcStark_IFCoefs_error",
            initial_value=np.array([0, 0, 0]),
            vals=vals.Arrays(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_RabiRate_Coefs",
            "RabiRate_Coefs",
            initial_value=np.array([0, 0, 0]),
            vals=vals.Arrays(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_RabiRate_Coefs_error",
            "RabiRate_Coefs_error",
            initial_value=np.array([0, 0, 0]),
            vals=vals.Arrays(),
        )
        self.add_pulse_parameter(
            op_name, "f0g1_kappa", "kappa", initial_value=0.4e8, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_kappa_error",
            "kappa_error",
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name, "f0g1_gamma1", "gamma1", initial_value=0.5e7, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_gamma2", "gamma2", initial_value=0.5e7, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_delta", "delta", initial_value=0, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_a", "a", initial_value=1, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_photonTrunc",
            "photonTrunc",
            initial_value=1.8,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_pulseTrunc",
            "pulseTrunc",
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_junctionTrunc",
            "junctionTrunc",
            initial_value=2,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_junctionSigma",
            "junctionSigma",
            initial_value=1.5e-9,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name, "f0g1_frequency", "frequency", initial_value=0, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_phase", "phase", initial_value=0, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_alpha", "alpha", initial_value=1, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name, "f0g1_phi_skew", "phi_skew", initial_value=0, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_timeReverse",
            "timeReverse",
            initial_value=False,
            vals=vals.Bool(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_lowerFreqPhoton",
            "lowerFreqPhoton",
            initial_value=False,
            vals=vals.Bool(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_driveDetScale",
            "driveDetScale",
            initial_value=0,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_junctionType",
            "junctionType",
            initial_value="ramp",
            vals=vals.Strings(),
        )
        self.add_pulse_parameter(
            op_name, "f0g1_delay", "delay", initial_value=0, vals=vals.Numbers()
        )
        self.add_pulse_parameter(
            op_name,
            "f0g1_delay_error",
            "delay_error",
            initial_value=0,
            vals=vals.Numbers(),
        )

        # flattop_f0g1 pulse for Ac Stark and Rabi Rate calibrations
        op_name = "flattop_f0g1"
        self.add_operation(op_name)
        self.add_pulse_parameter(
            op_name,
            op_name + "_pulse_type",
            "pulse_type",
            initial_value="GaussFilteredCosIQPulse",
            vals=vals.Enum("GaussFilteredCosIQPulse"),
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_amplitude",
            "amplitude",
            vals=vals.Numbers(),
            initial_value=0.5,
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_pulse_length",
            "pulse_length",
            initial_value=100e-9,
            vals=vals.Numbers(),
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_sigma",
            "sigma",
            vals=vals.Numbers(),
            initial_value=2e-9,
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_buffer_length_start",
            "buffer_length_start",
            vals=vals.Numbers(),
            initial_value=20e-9,
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_buffer_length_end",
            "buffer_length_end",
            vals=vals.Numbers(),
            initial_value=20e-9,
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_mod_frequency",
            "mod_frequency",
            vals=vals.Numbers(),
            initial_value=100e6,
        )
        self.add_pulse_parameter(
            op_name, op_name + "_phase", "phase", vals=vals.Numbers(), initial_value=0
        )
        self.add_pulse_parameter(
            op_name, op_name + "_alpha", "alpha", vals=vals.Numbers(), initial_value=1
        )
        self.add_pulse_parameter(
            op_name,
            op_name + "_phi_skew",
            "phi_skew",
            vals=vals.Numbers(),
            initial_value=0,
        )

        # f0g1_reset pulse for unconditional all-microwave reset
        # The operation is spirit is identical to f0g1_flattop with own params
        op_name = 'f0g1_reset_pulse'
        self.add_operation(op_name)
        self.add_pulse_parameter(op_name, op_name + '_pulse_type', 'pulse_type',
                                 initial_value='GaussFilteredCosIQPulse',
                                 vals=vals.Enum('GaussFilteredCosIQPulse'))
        self.add_pulse_parameter(op_name, op_name + '_amplitude', 'amplitude',
                                 vals=vals.Numbers(), initial_value=0.5)
        self.add_pulse_parameter(op_name, op_name + '_pulse_length', 'pulse_length',
                                 initial_value=100e-9, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, op_name + '_gaussian_filter_sigma', 'gaussian_filter_sigma',
                                 vals=vals.Numbers(), initial_value=5e-9)
        self.add_pulse_parameter(op_name, op_name + '_buffer_length_start', 'buffer_length_start',
                                 vals=vals.Numbers(), initial_value=20e-9)
        self.add_pulse_parameter(op_name, op_name + '_buffer_length_end', 'buffer_length_end',
                                 vals=vals.Numbers(), initial_value=20e-9)
        self.add_pulse_parameter(op_name, op_name + '_mod_frequency', 'mod_frequency',
                                 vals=vals.Numbers(), initial_value=100e6)
        self.add_pulse_parameter(op_name, op_name + '_phase', 'phase',
                                 vals=vals.Numbers(), initial_value=0)
        self.add_pulse_parameter(op_name, op_name + '_alpha', 'alpha',
                                 vals=vals.Numbers(), initial_value=1)
        self.add_pulse_parameter(op_name, op_name + '_phi_skew', 'phi_skew',
                                 vals=vals.Numbers(), initial_value=0)

        # ef_for_f0g1_reset pulse for unconditional all-microwave reset
        op_name = 'ef_for_f0g1_reset_pulse'
        self.add_operation(op_name)
        self.add_pulse_parameter(op_name, op_name + '_pulse_type', 'pulse_type',
                                 initial_value='GaussFilteredCosIQPulse',
                                 vals=vals.Enum('GaussFilteredCosIQPulse'))
        self.add_pulse_parameter(op_name, op_name + '_amplitude', 'amplitude',
                                 vals=vals.Numbers(), initial_value=0.5)
        self.add_pulse_parameter(op_name, op_name + '_pulse_length', 'pulse_length',
                                 initial_value=100e-9, vals=vals.Numbers())
        self.add_pulse_parameter(op_name, op_name + '_gaussian_filter_sigma', 'gaussian_filter_sigma',
                                 vals=vals.Numbers(), initial_value=5e-9)
        self.add_pulse_parameter(op_name, op_name + '_buffer_length_start', 'buffer_length_start',
                                 vals=vals.Numbers(), initial_value=20e-9)
        self.add_pulse_parameter(op_name, op_name + '_buffer_length_end', 'buffer_length_end',
                                 vals=vals.Numbers(), initial_value=20e-9)
        self.add_pulse_parameter(op_name, op_name + '_mod_frequency', 'mod_frequency',
                                 vals=vals.Numbers(), initial_value=100e6)
        self.add_pulse_parameter(op_name, op_name + '_phase', 'phase',
                                 vals=vals.Numbers(), initial_value=0)
        self.add_pulse_parameter(op_name, op_name + '_alpha', 'alpha',
                                 vals=vals.Numbers(), initial_value=1)
        self.add_pulse_parameter(op_name, op_name + '_phi_skew', 'phi_skew',
                                 vals=vals.Numbers(), initial_value=0)
        self.add_pulse_parameter(op_name, op_name + '_AcStark_IFCoefs',
                                 'AcStark_IFCoefs',
                                 initial_value=np.array([0, 0, 0]),
                                 vals=vals.Arrays())
        self.add_pulse_parameter(op_name, op_name + '_AcStark_IFCoefs_error',
                                 'AcStark_IFCoefs_error',
                                 initial_value=np.array([0, 0, 0]),
                                 vals=vals.Arrays())

        op_name = 'f0g1_catch'
        self.add_operation(op_name)
        self.add_pulse_parameter(op_name,
                                 op_name + '_kappa', 'kappa',
                                 initial_value=0.4e8, vals=vals.Numbers())
        self.add_pulse_parameter(op_name,
                                 op_name + '_kappa_error', 'kappa_error',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(op_name,
                                 op_name + '_frequency', 'frequency',
                                 initial_value=0, vals=vals.Numbers())
        self.add_pulse_parameter(op_name,
                                 op_name + '_timeReverse', 'timeReverse',
                                 initial_value=True, vals=vals.Bool())

    @property
    def transmon_transition_names(self):
        SPECIAL_TRANSITION_NAMES = ('f0g1',)
        return [tn for tn in self.transition_names
                if tn not in SPECIAL_TRANSITION_NAMES]

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def _validate_preparation_params(self, preparation_params):
        log.error('specifying `preparation_params` in the qubit object is '
                  'deprecated and will have  no effect. Please use `qb.reset.steps()` '
                  'to specify your reset type or directly specify the '
                  '`reset_params` as a keyword argument to the `QuantumExperiment`'
                  'child measurement class.')
        return preparation_params

    def _drive_mixer_calibration_tmp_vals(self):
        """Convert drive_mixer_calib_settings to temporary values format.

        Returns:
            A list of tuples to be passed to temporary_value (using *).
        """
        return [(self.parameters[k], v)
                for k, v in self.drive_mixer_calib_settings().items()]

    def get_ge_amp180_from_ge_freq(self, ge_freq):
        """Calculates the pi pulse amplitude required for a given ge transition
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
        """Calculates the RO frequency required for a given ge transition
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
        """Calculates the correction to a linear scaling of the pulse amplitude
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
        """Calculates the transition frequency for a given DC bias and flux
        pulse amplitude using fit parameters stored in the qubit object.
        Note that the qubit parameter flux_amplitude_bias_ratio is used for
        conversion between bias values and amplitudes.

        :param bias: (float) DC bias. If model='approx' is used, the bias is
            optional, and is understood relative to the parking position at
            which the  model was measured. Otherwise, it mandatory and is
            interpreted as voltage of the DC source.
        :param amplitude: (float, default: 0) flux pulse amplitude
        :param transition: (str or list of str, default: 'ge') the transition
            or transitions whose frequency should be calculated.
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
        :return: calculated transition frequency/frequencies

        TODO: Add feature to automatically make use of
              `InterpolatedHamiltonianModel` to speed up the computation in
              certain use cases.
        """
        if isinstance(transition, (list, tuple)):
            return_list = True
        else:
            transition = [transition]
            return_list = False

        for t in transition:
            if t not in ['ge', 'ef', 'gf']\
                    or (t != 'ge' and model not in ['transmon_res']):
                raise NotImplementedError(
                    f'calculate_frequency: Currently, transition {t} '
                    f'is not implemented for model {model}.')
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
            freqs = [fit_mods.Qubit_dac_to_freq(
                amplitude + (0 if bias is None or np.all(bias == 0) else
                             bias * flux_amplitude_bias_ratio), **vfc)]
        elif model == 'transmon':
            kw = deepcopy(vfc)
            kw.pop('coupling', None)
            # FIXME: 'fr' refers to the bare readout-resonator frequency,
            #  this is not a very descriptive name. Should it be changed to
            #  'bare_ro_res_freq'? This is relevant to the device database.
            kw.pop('fr', None)
            freqs = [fit_mods.Qubit_dac_to_freq_precise(bias + (
                0 if np.all(amplitude == 0)
                else amplitude / flux_amplitude_bias_ratio), **kw)]
        elif model == 'transmon_res':
            freqs = fit_mods.Qubit_dac_to_freq_res(
                bias + (0 if np.all(amplitude == 0)
                        else amplitude / flux_amplitude_bias_ratio),
                return_ef=True, **vfc)
            freqs = [
                {'ge': freqs[0], 'ef': freqs[1], 'gf': freqs[0]+freqs[1]}[t]
                for t in transition]
        else:
            raise NotImplementedError(
                "Currently, only the models 'approx', 'transmon', and"
                "'transmon_res' are implemented.")
        if update:
            for t, f in zip(transition, freqs):
                if f'{t}_freq' in self.parameters:
                    self.parameters[f'{t}_freq'](f)
                else:
                    log.warning(f'Cannot set the frequency of transition {t}!')
        if return_list:
            return freqs
        else:
            return freqs[0]

    def calculate_flux_voltage(self, frequency=None, bias=None,
                               amplitude=None, transition='ge',
                               model='transmon_res', flux=None,
                               branch=None):
        """Calculates the flux pulse amplitude or DC bias required to reach a
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
            implemented for all models. The model 'transmon_res' also allows to
            compute the 'ef' and 'gf' transition.
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
        if model != 'transmon_res' and transition not in ['ge']:
            raise NotImplementedError(
                'Currently, only ge transition is implemented.')
        elif transition not in ['ge', 'ef', 'gf']:
            raise NotImplementedError(
                'Currently, only the ge, ef & gf transitions are implemented.')
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
                frequency, **vfc, branch=branch, single_branch=True,
                transition=transition)
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
        """Calculates the DC bias for a given target flux.

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
        """Calculates the conversion factor between flux pulse amplitudes and bias
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
        """Generates a scaled and shifted version of the voltage frequency
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

    def update_detector_functions(self):
        """Instantiates common detector classes and assigns them as attributes.
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
            data_type='raw', polar=True, single_int_avg=True)

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
        - configure qubit drive local oscillator
        - call update_detector_functions
        - set switches to the mode required for the measurement
        - further preparation, see super().prepare

        Args:
            drive (str, None): the kind of drive to be applied, which can be
                None (no drive), 'continuous_spec' (continuous spectroscopy),
                'continuous_spec_modulated' (continuous spectroscopy using
                the modulated configuration of the switch),
                'pulsed_spec' (pulsed spectroscopy), or the default
                'timedomain' (AWG-generated signal upconverted by the mixer)
            switch (str): the required switch mode. Can be a switch mode
                understood by set_switch or the default value 'default', in
                which case the switch mode is determined based on the kind
                of drive ('spec' for continuous/pulsed spectroscopy w/o
                modulated; 'no_drive' if drive is None and a switch mode
                'no_drive' is configured for this qubit; 'modulated' in all
                other cases).
        """
        if switch == 'default':
            if drive is None and 'no_drive' in self.switch_modes():
                # use special mode for measurements without drive if that
                # mode is defined
                switch = 'no_drive'
            else:
                # use 'spec' for qubit spectroscopy measurements
                # (continuous_spec and pulsed_spec) and 'modulated' otherwise
                switch = 'spec' if drive is not None and drive.endswith(
                    '_spec') else 'modulated'
        else:
            # switch mode was explicitly provided by the caller (e.g.,
            # for mixer calib)
            pass
        super().prepare(drive=drive, switch=switch)
        ge_lo = self.instr_ge_lo

        self.configure_offsets(set_ge_offsets=(drive == 'timedomain'))

        # configure qubit drive local oscillator
        if ge_lo() is not None:
            if drive is None:
                ge_lo.get_instr().off()
            elif 'continuous_spec' in drive:
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
                                 + "', 'pulsed_spec', "
                                 + "'continuous_spec_modulated' and "
                                 + "'timedomain'.")

        param = f'{self.ge_I_channel()}_centerfreq'
        if param in self.instr_pulsar.get_instr().parameters:
            if np.abs(self.instr_pulsar.get_instr().get(param) - self.get_ge_lo_freq()) > 1:
                self.instr_pulsar.get_instr().set(param, self.get_ge_lo_freq())

        # other preparations
        self.update_detector_functions()
        # provide classifier params to acqusition device
        self.instr_acq.get_instr().set_classifier_params(
            self.get_acq_int_channels(), self.acq_classifier_params())

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

    def get_spec_pars(self):
        return self.get_operation_dict()['Spec ' + self.name]

# ---- f0g1 #
    def get_f0g1_pars(self):
        return self.get_operation_dict()['f0g1 ' + self.name]
    def get_flattop_f0g1_pars(self):
        return self.get_operation_dict()['flattop_f0g1 ' + self.name]

    def get_f0g1_catch_pars(self):
        return self.get_operation_dict()['f0g1_catch ' + self.name]
# ---- #

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

    def _add_f0g1_to_operation_dict(self, operation_dict):
        operation_dict['f0g1 ' + self.name]['operation_type'] = 'Other'
        operation_dict['flattop_f0g1 ' + self.name]['operation_type'] = 'Other'
        operation_dict['f0g1_catch ' + self.name]['operation_type'] = 'Other'

        params_to_copy = ['I_channel', 'Q_channel']
        for p in params_to_copy:
            operation_dict['flattop_f0g1 ' + self.name][p] = operation_dict[
                'f0g1 ' + self.name][p]
            operation_dict['f0g1_reset_pulse ' + self.name][p] = operation_dict[
                'f0g1 ' + self.name][p]
            operation_dict['ef_for_f0g1_reset_pulse ' + self.name][p] = \
                operation_dict['X180 ' + self.name][p]

        params_f0g1 = [param for param in list(operation_dict['f0g1 ' +
                                                              self.name].keys())
                       if param not in ['kappa', 'kappa_error', 'frequency',
                                        'timeReverse']]
        for p in params_f0g1:
            operation_dict['f0g1_catch ' + self.name][p] = operation_dict[
                'f0g1 ' + self.name][p]

    def get_operation_dict(self, operation_dict=None):
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['Spec ' + self.name]['operation_type'] = 'Other'
        operation_dict['Acq ' + self.name]['flux_amplitude'] = 0

        if "f0g1" in self.transition_names:
            self._add_f0g1_to_operation_dict(operation_dict)
        for tr_name in self.transmon_transition_names:
            tn = '' if tr_name == 'ge' else f'_{tr_name}'
            operation_dict[f'X180{tn} ' + self.name]['basis'] = self.name + tn
            operation_dict[f'X180{tn} ' + self.name]['operation_type'] = 'MW'
            if tr_name != 'ge':
                operation_dict[f'X180{tn} ' + self.name]['I_channel'] = \
                    operation_dict['X180 ' + self.name]['I_channel']
                operation_dict[f'X180{tn} ' + self.name]['Q_channel'] = \
                    operation_dict['X180 ' + self.name]['Q_channel']
                if self.get(f'{tr_name}_freq') == 0:
                    operation_dict[f'X180{tn} ' + self.name][
                        'mod_frequency'] = None
                else:
                    operation_dict[f'X180{tn} ' + self.name][
                        'mod_frequency'] = self.get(f'{tr_name}_freq') - \
                                           self.ge_freq() + self.ge_mod_freq()

            operation_dict.update(add_suffix_to_dict_keys(
                sq.get_pulse_dict_from_pars(
                    operation_dict[f'X180{tn} ' + self.name]),
                f'{tn} ' + self.name))

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

    def measure_resonator_spectroscopy(self, freqs, sweep_points_2D=None,
                                       sweep_function_2D=None,
                                       trigger_separation=3e-6,
                                       upload=True, analyze=True,
                                       close_fig=True, label=None):
        """Varies the frequency of the microwave source to the resonator and
        measures the transmittance
        """
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array might be '
                             'too small. The units should be Hz.'))

        if label is None:
            if sweep_function_2D is not None:
                label = 'resonator_scan_2d' + self.msmt_suffix
            else:
                label = 'resonator_scan' + self.msmt_suffix
        self.prepare(drive=None)
        sweep_function = self.swf_ro_freq_lo()
        if upload:
            ro_pars = self.get_ro_pars()
            if getattr(sweep_function, 'includes_IF_sweep', False):
                ro_pars['mod_frequency'] = 0
            seq = sq.pulse_list_list_seq([[ro_pars]], upload=False)

            for seg in seq.segments.values():
                if hasattr(self.instr_acq.get_instr(),
                           'use_hardware_sweeper') and \
                        self.instr_acq.get_instr().use_hardware_sweeper():
                    self.int_avg_det_spec.AWG = self.instr_pulsar.get_instr()
                    lo_freq, delta_f, _ = self.instr_acq.get_instr()\
                        .get_params_for_spectrum(
                            freqs, get_closest_lo_freq=(
                                lambda f, s=self: s.get_closest_lo_freq(
                                    f, operation='ro')))
                    self.instr_acq.get_instr().set_lo_freq(self.acq_unit(),
                                                           lo_freq)
                    seg.acquisition_mode = dict(
                        sweeper='hardware',
                        f_start=freqs[0] - lo_freq,
                        f_step=delta_f,
                        n_step=len(freqs),
                    )
                else:
                    seg.acquisition_mode = dict(
                        sweeper='software'
                    )
            self.instr_pulsar.get_instr().program_awgs(seq)

        MC = self.instr_mc.get_instr()
        MC.set_sweep_function(sweep_function)
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
            self.int_avg_det.set_polar(True)
            self.int_avg_det.AWG = self.int_avg_det_spec.AWG
            MC.set_detector_function(self.int_avg_det)

        with temporary_value(self.instr_trigger.get_instr().pulse_period,
                             trigger_separation):
            if self.instr_pulsar.get_instr() != self.int_avg_det_spec.AWG:
                awg_name = self.instr_acq.get_instr().get_awg_control_object()[1]
                self.instr_pulsar.get_instr().start(exclude=[awg_name])
            MC.run(name=label, mode=mode)
            if self.instr_pulsar.get_instr() != self.int_avg_det_spec.AWG:
                self.instr_pulsar.get_instr().stop()

        if analyze:
            ma.MeasurementAnalysis(close_fig=close_fig, qb_name=self.name,
                                   TwoD=(mode == '2D'))

    def measure_qubit_spectroscopy(self, freqs, sweep_points_2D=None,
            sweep_function_2D=None, pulsed=True, trigger_separation=13e-6,
            upload=True, analyze=True, close_fig=True, label=None):
        """Varies qubit drive frequency and measures the resonator
        transmittance
        """
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
                    # The factor of 2 is needed here because the spec pulse is
                    # applied only to the I channel. This means that we get
                    # half the amplitude after upconversion when the
                    # outputamplitude node is set to 0.5, which is the reasonable
                    # setting to use for digital IQ modulation in time-domain
                    # experiments (output pulse has the programmed pulse
                    # amplitude).
                    spec_pulse["amplitude"] = 2 * dbm_to_vp(self.spec_power())
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
            self.int_avg_det.set_polar(True)
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

    def measure_drive_mixer_spectrum(self, if_freqs, amplitude=0.5,
                                     trigger_sep=5e-6, align_frequencies=True):
        """Measure the output spectrum of the drive upconversion mixer.

        A square pulse with the given amplitude is generated at ge_freq.
        The readout local oscillator is swept to resolve different frequencies.
        To average the signal with a fixed phase, the intermediate frequencies
        `if_freq` need to satisfy the commensurability condition that
        `if_freq * trigger_sep` must be an integer. The `if_freqs` can be
        automatically adjusted to satisfy this constraint using the
        `align_frequencies` parameter.

        Args:
            if_freqs:
                A list or array of intermediate frequencies to measure the
                output spectrum for, in hertz.
            amplitude:
                amplitude of the output square pulse in volts. Defaults
                to `0.5`.
            trigger_sep:
                experiment repetition period in seconds. Defaults to `5e-6`.
            align_frequencies:
                Boolean flag, whether to automatically satisfy the
                commensurability constraint. Might lead to in non-uniform
                spacing of sweep points. Defaults to `True`.

        Return:
            MeasurementAnalysis object that has plotted the output spectrum.
        """
        MC = self.instr_mc.get_instr()
        if align_frequencies:
            if_freqs = (if_freqs*trigger_sep).astype(int)/trigger_sep
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
        """Measure the output power spectrum of the drive upconversion mixer.

        A square pulse with the given amplitude is generated at ge_freq.
        The readout local oscillator is set at `ro_lo_freq`, and the power
        spectra of individual timetraces are averaged. Makes use of the
        `self.scope_fft_det` detector.

        Args:
            ro_lo_freq:
                The frequency of the readout local oscillator in hertz.
            amplitude:
                amplitude of the output square pulse in volts. Defaults
                to `0.5`.
            trigger_sep:
                experiment repetition period in seconds. Defaults to `5e-6`.

        Return:
            MeasurementAnalysis object that has plotted the output spectrum.
        """
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
            upload=True, plot=True, **kwargs):

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
                (self.instr_pulsar.get_instr().prepend_zeros,
                 kwargs.get('prepend_zeros', 0)),
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
        """Calibrate drive upconversion mixer local oscillator leakage.

        Measures the average power at the LO frequency at the output of the
        mixer. Uses the Nelder-Mead optimization algorithm and the scope_fft_det
        detector function.

        FIXME: not tested after the changes in MC in !330

        Args:
            update:
                Boolean flag, whether to update the qubit parameters with the
                optimized values. Defaults to `True`.
            x0:
                Initial values for the optimization algorithm.
                Defaults to `(0., 0.)`.
            initial_stepsize:
                Size of the initial step of the optimization algorithm in volts.
                Defaults to `0.01`.
            trigger_sep:
                Experiment repetition period in seconds. Defaults to `5e-6`.
            no_improv_break:
                The optimization will be stopped after this many optimization
                cycles that lead to no improvement in the costfunction.
                Defaults to `50`.
            upload:
                Boolean flag, whether to upload the waveforms for the
                experiment. Defaults to `True`.
            plot:
                Boolean flag, whether to plot the analysis results. Defaults
                to `True`.

        Return:
            Optimal DC offsets for the I and Q output channels.
        """

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
                                      plot=True, **kwargs):
        """Calibrate drive upconversion mixer local oscillator leakage.

        Measures the averaged signal at the LO frequency at the output of the
        upconversion mixer. Uses the Nelder-Mead optimization algorithm and the
        int_avg_det_spec detector function.

        Args:
            update:
                Boolean flag, whether to update the qubit parameters with the
                optimized values. Defaults to `True`.
            x0:
                Initial values for the optimization algorithm.
                Defaults to `(0., 0.)`.
            initial_stepsize:
                Size of the initial step of the optimization algorithm in volts.
                Defaults to `0.01`.
            trigger_sep:
                Experiment repetition period in seconds. Defaults to `5e-6`.
            no_improv_break:
                The optimization will be stopped after this many optimization
                cycles that lead to no improvement in the costfunction.
                Defaults to `50`.
            upload:
                Boolean flag, whether to upload the waveforms for the
                experiment. Defaults to `True`.
            plot:
                Boolean flag, whether to plot the analysis results. Defaults
                to `True`.
            kwargs:
                prepend_zeros: temporary value for pulsar.prepend_zeros.
                    Defaults to 0.

        Return:
            Optimal DC offsets for the I and Q output channels.
        """

        def detector_generator(s=self):
            return det.IndexDetector(s.int_avg_det_spec, 0)

        return self._calibrate_drive_mixer_carrier_common(
            detector_generator, update=update, x0=x0,
            initial_stepsize=initial_stepsize, trigger_sep=trigger_sep,
            no_improv_break=no_improv_break, upload=upload, plot=plot, **kwargs)

    def calibrate_readout_mixer_carrier(self, other_qb, update=True,
                                        x0=(0., 0.),
                                        initial_stepsize=0.01, trigger_sep=5e-6,
                                        no_improv_break=50, upload=True,
                                        plot=True):
        """Calibrate readout upconversion mixer local oscillator leakage

        FIXME: not tested after the changes in MC in !330

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
                                            upload=True, **kwargs):
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
            kwargs:
                prepend_zeros: temporary value for pulsar.prepend_zeros.
                    Defaults to 0.

        Returns:
            V_I (float): DC bias on I channel that minimizes LO leakage.
            V_Q (float): DC bias on Q channel that minimizes LO leakage.
            ma (:py:class:~'pycqed.timedomain_analysis.MixerCarrierAnalysis'): 
                The MixerCarrierAnalysis object.
        """
        MC = self.instr_mc.get_instr()
        if meas_grid is None:
            if len(limits) != 4:
                log.error('Input variable `limits` in function call '
                          '`calibrate_drive_mixer_carrier_model` needs to be a list '
                          'or 1D array of length 4.\nFound length '
                          '{} object instead!'.format(len(limits)))
            if isinstance(n_meas, int):
                n_meas = (n_meas, n_meas)
            elif len(n_meas) != 2:
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
                (self.instr_pulsar.get_instr().prepend_zeros,
                 kwargs.get('prepend_zeros', 0)),
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
        """Calibrate drive upconversion mixer other sideband.

        Measures the averaged signal of a square-pulse at the other sideband
        frequency. Uses the Nelder-Mead optimization algorithm and the
        int_avg_det_spec detector function.

        FIXME: not tested after the changes in MC in !330

        Args:
            update:
                Boolean flag, whether to update the qubit parameters with the
                optimized values. Defaults to `True`.
            amplitude:
                Amplitude of the square pulse used for calibration in volts.
                Defaults to `0.5`.
            trigger_sep:
                Experiment repetition period in seconds. Defaults to `5e-6`.
            no_improv_break:
                The optimization will be stopped after this many optimization
                cycles that lead to no improvement in the costfunction.
                Defaults to `50`.
            initial_stepsize:
                Size of the initial step of the optimization algorithm in volts.
                Defaults to `0.01`.

        Return:
            optimal IQ amplitude ratio `alpha` and phase correction `phi`.
        """
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
            kwargs:
                prepend_zeros: temporary value for pulsar.prepend_zeros.
                    Defaults to 0.

        Returns:
            alpha (float): The amplitude ratio that maximizes the suppression of 
                the unwanted sideband.
            phi_skew (float): The phi_skew that maximizes the suppression of 
                the unwanted sideband.
            ma (:py:class:~'pycqed.timedomain_analysis.MixerSkewnessAnalysis'): 
                The MixerSkewnessAnalysis object.
        """
        if meas_grid is None:
            if len(limits) != 4:
                log.error('Input variable `limits` in function call '
                          '`calibrate_drive_mixer_skewness_model` needs to be a list '
                          'or 1D array of length 4.\nFound length '
                          '{} object instead!'.format(len(limits)))
            if isinstance(n_meas, int):
                n_meas = [n_meas, n_meas]
            elif len(n_meas) != 2:
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
            (self.instr_pulsar.get_instr().prepend_zeros,
             kwargs.get('prepend_zeros', 0)),
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
        """WARNING: Does not automatically update the qubit frequency parameter.
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
        """Find readout frequency at which contrast between the states of the
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
                                 upload=True, states=('g','e'), reset_params=None):
        """Varies the frequency of the microwave source to the resonator and
        measures the transmittance
        """
        if freqs is None:
            raise ValueError("Unspecified frequencies for "
                             "measure_resonator_spectroscopy")
        if np.any(freqs < 500e6):
            log.warning(('Some of the values in the freqs array '
                         'might be too small. The units should be Hz.'))

        assert isinstance(states, tuple), \
            "states should be a tuple, not {}".format(type(states))

        self.prepare(drive='timedomain')
        MC = self.instr_mc.get_instr()

        for state in states:
            sq.single_state_active_reset(
                    operation_dict=self.get_operation_dict(),
                    qb_name=self.name,
                    state=state, reset_params=reset_params, upload=upload)

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

    def measure_T2_freq_sweep(self, flux_lengths=None, n_pulses=None,
                              cz_pulse_name=None,
                              freqs=None, amplitudes=None, phases=[0,120,240],
                              analyze=True, cal_states='auto', cal_points=False,
                              upload=True, label=None, n_cal_points_per_state=2,
                              exp_metadata=None, operation_dict=None,
                              vfc_kwargs=None):
        """Flux pulse amplitude measurement used to determine the qubits energy in
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
            vfc_kwargs: Additional arguments for self.calculate_flux_voltage

        Returns:

        """
        if operation_dict is None:
            operation_dict = self.get_operation_dict()
        if freqs is not None:
            vfc_kwargs = vfc_kwargs or {}
            amplitudes = self.calculate_flux_voltage(
                frequency=freqs,
                **vfc_kwargs,
            )
        amplitudes = np.array(amplitudes)

        if cz_pulse_name is None:
            cz_pulse_name = 'FP ' + self.name

        if np.any(np.isnan(amplitudes)):
            raise ValueError('Specified frequencies resulted in nan amplitude. '
                             'Check frequency range!')

        if amplitudes is None:
            raise ValueError('Either freqs or amplitudes need to be specified')

        if label is None:
            label = 'T2_Frequency_Sweep_{}'.format(self.name)
        MC = self.instr_mc.get_instr()
        self.prepare(drive='timedomain')

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
                # FIXME this is a hack until this measurement is refactored
                #  to a QuantumExperiment, allowing to pass operations not in
                #  the qubit object, e.g. two-qubit gates
                operation_dict=operation_dict,
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
                           operation_dict[cz_pulse_name]['pulse_length']
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

    def configure_pulsar(self):
        """In addition to the super call:
        - Reset modulation frequency and amplitude scaling
        - Set flux distortion, see set_distortion_in_pulsar
        """
        super().configure_pulsar()
        pulsar = self.instr_pulsar.get_instr()
        # make sure that some settings are reset to their default values
        for quad in ['I', 'Q']:
            ch = self.get(f'ge_{quad}_channel')
            if f'{ch}_mod_freq' in pulsar.parameters:
                pulsar.parameters[f'{ch}_mod_freq'](None)
            if f'{ch}_amplitude_scaling' in pulsar.parameters:
                pulsar.parameters[f'{ch}_amplitude_scaling'](1)
        # set flux distortion
        self.set_distortion_in_pulsar()

    def configure_offsets(self, set_ro_offsets=True, set_ge_offsets=True,
                          offset_list=None):
        """Set AWG channel DC offsets and switch sigouts on.

        :param set_ro_offsets: whether to set offsets for RO channels
        :param set_ge_offsets: whether to set offsets for drive channels
        :param offset_list: additional offsets to set
        """
        pulsar = self.instr_pulsar.get_instr()
        if offset_list is None:
            offset_list = []

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

        super().configure_offsets(set_ro_offsets=set_ro_offsets,
                                  offset_list=offset_list)

    def set_distortion_in_pulsar(self, datadir=None):
        """Configures the fluxline distortion in a pulsar object according to the
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
        """Returns (a subset of) channels.

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
        """Returns a channel map.

        Args:
            drive (bool): whether or not to include drive channels
            ro (bool): whether or not to include readout channels
            flux (bool): whether or not to include flux channel

        Returns:
            channels (dict): key is the qubit name and value is a
            list of channels
        """
        return {self.name: self.get_channels(drive=drive, ro=ro, flux=flux)}

    def add_reset_schemes(
        self,
        preselection=True,
        feedback_reset=True,
        parametric_flux_reset=False
    ):
        """Adds reset schemes to the current instance.

        This function adds reset schemes to the current instance of an
        QuDev_transmon. It checks if each scheme is already present before
        adding, so that no duplicates are created. If a scheme is not added
        successfully, a message will be logged stating this. If a submodule
        with the same name as one being attempted to add already exists in
        self.reset.submodules, a ValueError will be raised. The error message
        will specify which submodule and instance names it was called on.

        Args:
            preselection (bool, optional): If True, adds the Preselection
                scheme. Default is True.
            feedback_reset (bool, optional): If True, adds the FeedbackReset
                scheme. Default is True.
            parametric_flux_reset (bool, optional): If True, adds the
                ParametricFluxReset scheme. Default is True. Because of
                some implementation dependencies (LRU) and insufficient
                testing.

        Returns: None

        Raises: ValueError: If a submodule with the same name already exists in
            self.reset.submodules. The error message will specify which
            submodule and instance names it was called on.
        """
        msg = (
            "{} submodule already in {}.reset.submodules. "
            "Submodule won't be created again. "
        )

        # Inform user
        # FIXME: Add to logging framework _and_ print
        print("Added the following reset schemes:")
        print(f"-- preselection: {preselection}")
        print(f"-- feedback_reset: {feedback_reset}")
        print(f"-- parametric_flux_reset: {parametric_flux_reset}")


        if preselection:
            submodule_name = reset.Preselection.DEFAULT_INSTANCE_NAME
            if submodule_name in self.reset.submodules:
                log.error(msg.format(submodule_name, self.name))
            else:
                self.reset.add_submodule("preselection", reset.Preselection(self.reset))

        if feedback_reset:
            submodule_name = reset.FeedbackReset.DEFAULT_INSTANCE_NAME
            if submodule_name in self.reset.submodules:
                log.error(msg.format(submodule_name, self.name))
            else:
                self.reset.add_submodule(
                    submodule_name, reset.FeedbackReset(self.reset)
                )

        if parametric_flux_reset:
            submodule_name = reset.ParametricFluxReset.DEFAULT_INSTANCE_NAME
            if submodule_name in self.reset.submodules:
                log.error(msg.format(submodule_name, self.name))
            else:
                self.reset.add_submodule(
                    submodule_name, reset.ParametricFluxReset(self.reset)
                )

# FIXME: Is this needed with add_reset_schemes covering PFM?
    def add_parametric_flux_modulation(
        self,
        op_name="PFM",
        parameter_prefix="parametric_flux_modulation",
        transition_name="ge",
        pulse_type="BufferedCZPulse",
    ):
        """Adds a parametric flux based reset operation to the qubit object.

        This method allows the user to add a parametric flux modulation
        operation to the qubit object, which can be used to perform a reset
        operation based on the parametric flux. The operation name, parameter
        prefix, transition name, and pulse type can be specified.

        Args:
            op_name (str, optional): The name of the operation to be added.
                Defaults to "PFM".
            parameter_prefix (str, optional): The prefix for the parameters
            associated with the operation. Defaults to 'parametric_flux_modulation'.
            transition_name (str, optional): The name of the transition for
                which the operation is defined. Defaults to 'ge'.
            pulse_type (str, optional): The type of pulse to be used for the
                operation. Defaults to 'BufferedCZPulse'.

        Raises:
            KeyError: If the operation name already exists in the qubit object.
            KeyError: If the pulse type is not recognized.

        Returns:
            None
        """
        tn = '' if transition_name == 'ge' else f'_{transition_name}'
        op_name = f"{op_name}{tn}"
        parameter_prefix = f'{parameter_prefix}{tn}'
        self.add_operation(op_name)

        # Get default pulse params for the pulse type
        pulse_func = bpl.get_pulse_class(pulse_type)
        params = pulse_func.pulse_params()

        for param, init_val in params.items():
            self.add_pulse_parameter(
                op_name, parameter_prefix + '_' + param, param,
                initial_value=init_val, vals=None)

        # needed for unresolved pulses but not attribute of pulse object
        if 'basis_rotation' not in params.keys():
            self.add_pulse_parameter(
                op_name, parameter_prefix + '_basis_rotation',
                'basis_rotation', initial_value={}, vals=None)
