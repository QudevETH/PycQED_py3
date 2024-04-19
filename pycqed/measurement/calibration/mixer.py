import traceback
import numpy as np

from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_swf
from pycqed.measurement.calibration import two_qubit_gates as twoqbcal
from pycqed.measurement.waveform_control.block import ParametricValue
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.utilities.general import temporary_value
from pycqed.measurement.sweep_points import SweepPoints

import logging
log = logging.getLogger(__name__)

default_trigger_sep = 5e-6


class MixerSkewness(twoqbcal.CalibBuilder):
    """
    Mixer skewness calibration measurement for calibrating the sideband
    suppression of the drive IQ Mixer.

    The two settings that are used to calibrate the suppression of the
    unwanted sideband are the amplitude ratio and phase between I and Q.
    This method measures the sideband suppression for different values of
    these two settings that are either handed over as kw 'alpha' and
    'phi_skew' or in the task_list.
    The subsequent analysis fits an analytical model to the
    measured data and extracts the settings minimizing the amplitude of the
    sideband.

    The following kwargs are interpreted by this class:
        - amplitude (float): Amplitude of the calibration drive pulse. Default
        set to 0.1V.
        - trigger_sep (float): Seperation time in s between trigger signals.
            Defaults to 5e-6 s.
        - force_ro_mod_freq (bool, optional): Whether to force the current
            ro_mod_freq setting even though it results in non
            commensurable LO frequencies for the specified trigger_sep.
            Defaults to false.
        - prepend_zeros (int): temporary value for pulsar.prepend_zeros.
            Defaults to 0.
    """
    kw_for_sweep_points = {
        'alpha': dict(param_name='alpha', unit='',
                     label='Amplitude Ratio', dimension=0),
        'phi_skew': dict(param_name='phi_skew', unit='deg',
                      label='Phase Off.', dimension=1)
    }
    default_experiment_name = 'MixerSkewness'

    def __init__(self, task_list, sweep_points=None, amplitude=0.1, **kw):
        try:
            # calibration points not needed for mixer calibration
            kw['cal_states'] = ''
            super().__init__(task_list, sweep_points=sweep_points,
                             **kw)
            self.switch = 'calib'
            trigger_sep = kw.get('trigger_sep', default_trigger_sep)
            force_ro_mod_freq = kw.get('force_ro_mod_freq', False)

            tmp_vals = []
            try:
                for qb_obj in self.get_qubits()[0]:
                    tmp_vals += [
                        # read out at the drive sideband frequency
                        (qb_obj.ro_freq, qb_obj.ge_freq() - 2 *
                                 qb_obj.ge_mod_freq()),
                        # resets ro_mod_freq after it gets changed in
                        # self.commensurability_lo_trigger
                        (qb_obj.ro_mod_freq, qb_obj.ro_mod_freq()),
                        (qb_obj.acq_weights_type, 'SSB'),
                        (qb_obj.instr_trigger.get_instr().pulse_period,
                         trigger_sep),
                        (qb_obj.instr_pulsar.get_instr().prepend_zeros,
                         kw.get('prepend_zeros', 0)),
                        *qb_obj._drive_mixer_calibration_tmp_vals()
                    ]
            except Exception as x:
                log.warning('Qubit objects not found. Temporary values '
                            'for drive mixer calibration specified in the '
                            'qubit objects could not be set.')
            # Preprocess sweep points and tasks before creating the sequences
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            with temporary_value(*tmp_vals):
                try:
                    for qb_obj in self.get_qubits()[0]:
                        # sets ro_mod_freq depending on trigger separation to
                        # avoid beating patterns
                        self.commensurability_lo_trigger(
                            qb_obj, trigger_sep, force_ro_mod_freq)
                except Exception as x:
                    log.warning('No qubit objects found.')
                self.update_operation_dict()
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.sweep_block,
                    block_align=['center', 'end', 'center'],
                    amplitude=amplitude, **kw)

            # updates temporary_value for self.autorun()
            self.temporary_values += tmp_vals
            # run measurement & analysis if requested in kw
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def commensurability_lo_trigger(self, qb_obj, trigger_sep,
                                    force_ro_mod_freq):
        """
        Copied from qb.calibrate_drive_mixer_skewness_model().
        Checks commensurability of LO frequencies with trigger separation
        and sets qb_obj.ro_mod_freq() if needed.
        Args:
            qb_obj (QuDev_transmon): Qubit object which is tested for
                commensurability.
            trigger_sep (float): Seperation time in s between trigger signals.
            force_ro_mod_freq: Whether to force the current ro_mod_freq
                setting even though it results in non commensurable LO
                frequencies for the specified trigger_sep.
        """

        ro_lo_freq = qb_obj.get_ro_lo_freq()
        dr_lo_freq = qb_obj.ge_freq() - qb_obj.ge_mod_freq()
        # Frequency of the LO phases is given by the LOs beat frequency.
        beat_freq = 0.5 * (dr_lo_freq - ro_lo_freq)
        #         = 0.5*(ge_mod_freq + ro_mod_freq) in our case
        beats_per_trigger = np.round(beat_freq * trigger_sep,
                                     int(np.floor(
                                         np.log10(1 / trigger_sep))) + 2)
        if not beats_per_trigger.is_integer():
            log.warning('Difference of RO LO and drive LO frequency '
                        'resulting from the chosen modulation frequencies '
                        'is not an integer multiple of the trigger '
                        'seperation.')
            if not force_ro_mod_freq:
                if qb_obj.ro_fixed_lo_freq() is not None:
                    log.warning(
                        'Automatic adjustment of the RO IF might lead to '
                        'wrong results since ro_fixed_lo_freq is set.')
                beats_per_trigger = int(beats_per_trigger + 0.5)
                qb_obj.ro_mod_freq(2 * beats_per_trigger / trigger_sep \
                                 - qb_obj.ge_mod_freq())
                log.warning('To ensure commensurability the RO '
                            'modulation frequency will temporarily be set '
                            'to {} Hz.'.format(qb_obj.ro_mod_freq()))

    def sweep_block(self, sweep_points, qb, amplitude, **kw):
        # extract AWG channels
        tmp_p = self.block_from_ops('tmp', [f'X180 {qb}'])
        I_channel = tmp_p.pulses[0]['I_channel']
        Q_channel = tmp_p.pulses[0]['Q_channel']
        mod_frequency = tmp_p.pulses[0]['mod_frequency']

        # extract acq_length if qubit object is available
        try:
            qb_obj = self.get_qubits(qb)[0][0]
            acq_length = qb_obj.acq_length()
        except Exception as x:
            acq_length = 250e-9
            log.warning('Qubit object not found. Acquisition length is set '
                        f'to default value {acq_length}.')
        dp_block = self.block_from_pulse_dicts([dict(
                    pulse_type='GaussFilteredCosIQPulse',
                    pulse_length=acq_length,
                    ref_point='start',
                    amplitude=amplitude,
                    I_channel=I_channel,
                    Q_channel=Q_channel,
                    mod_frequency=mod_frequency,
                    phase_lock=False,
                )], 'drive')
        # All sweep points are interpreted as parameters of the drive pulse,
        # except if they are pulse modifier sweep points (see docstring of
        # Block.build).
        for k in list(sweep_points[0].keys()) + list(
                sweep_points.get_sweep_dimension(1, default={}).keys()):
            if '=' not in k:  # '=' indicates a pulse modifier sweep point
                for p in dp_block.pulses:
                    p[k] = ParametricValue(k)
        acq_block = self.block_from_ops('Acq', f'Acq {qb}')
        block = self.simultaneous_blocks('mixer_calib', [dp_block, acq_block])

        # return all generated blocks (parallel_sweep will arrange them)
        return [block]

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        Args:
            analysis_kwargs (dict): keyword arguments for analysis
            **kw: currently ignored
        Returns: the analysis instance
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        if 'TwoD' not in analysis_kwargs['options_dict']:
            if len(self.sweep_points) == 2:
                analysis_kwargs['options_dict']['TwoD'] = True
        self.analysis = tda.MixerSkewnessAnalysis(
            t_start=self.timestamp, **analysis_kwargs)

        return self.analysis

    def run_update(self, **kw):
        """
        Updates qubit parameters 'qb.ge_alpha' and 'qb.ge_phi_skew' with the
        fitted values from the analysis of this measurement.
        If the optimal values lie beyond the sweep point ranges, no values
        are updated.
        Args:
            **kw: currently ignored
        """
        assert self.meas_objs is not None, \
            "Update only works with qubit objects provided."
        assert self.analysis is not None, \
            "Update is only allowed after running the analysis."
        assert len(self.meas_objs) == 1, \
            "Update only works for one qubit measurement."

        qb_obj = self.meas_objs[0]
        alphas = self.sweep_points[0][f'{qb_obj.name}_alpha'][0]
        phi_skews = self.sweep_points[1][f'{qb_obj.name}_phi_skew'][0]

        analysis_params_dict = self.analysis.proc_data_dict[
            'analysis_params_dict']

        _alpha = analysis_params_dict['alpha']
        _phi = analysis_params_dict['phase']

        update = True
        if (_alpha < alphas.min() or _alpha > alphas.max()):
            log.warning('Optimum for amplitude ratio is outside '
                        'the measured range and no settings will be updated. '
                        'Best alpha according to fitting: {:.2f}'.format(
                _alpha))
            update = False
        if (_phi < phi_skews.min() or _phi > phi_skews.max()):
            log.warning('Optimum for phase correction is outside '
                        'the measured range and no settings will be updated. '
                        'Best phi according to fitting: {:.2f} deg'.format(
                _phi))
            update = False

        if update:
            qb_obj.ge_alpha(_alpha)
            qb_obj.ge_phi_skew(_phi)


class MixerCarrier(twoqbcal.CalibBuilder):
    """
    Method for calibrating the lo leakage of the drive IQ Mixer

    By applying DC biases on the I and Q inputs of an IQ mixer one can
    change the bias conditions of the diodes inside the mixer. This can be
    used to reduce LO leakage. This method measures the LO leakage for
    different values of DC biases. The subsequent analysis fits an
    analytical model to the measured data and extracts the settings
    minimizing the LO leakage.
    """

    kw_for_sweep_points = {
        'offset_i': dict(param_name='offset_i', unit='V',
                      label='Offset V_I', dimension=0),
        'offset_q': dict(param_name='offset_q', unit='V',
                         label='Offset V_Q', dimension=1)
    }
    default_experiment_name = 'MixerCarrier'

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            df_name = kw.pop('df_name', 'int_avg_det_spec')
            df_kwargs = kw.pop('df_kwargs', {})
            df_kwargs['live_plot_transform_type'] = \
            df_kwargs.get("live_plot_transform_type", 'mag_phase')
            # calibration points not needed for mixer calibration
            cal_states = kw.pop('cal_states', [])

            self.segment_kwargs = kw.pop('segment_kwargs', dict())
            super().__init__(task_list, sweep_points=sweep_points,
                             df_name=df_name, cal_states=cal_states,
                             df_kwargs=df_kwargs, **kw)
            # resets initialized sweep functions (hard and soft sweep). Will
            # be populated in self.generate_sweep_functions()
            self.switch = 'calib'
            self.sweep_functions = []
            self.sweep_points_pulses = SweepPoints(
                min_length=2)
            trigger_sep = kw.get('trigger_sep', default_trigger_sep)

            tmp_vals = []
            try:
                for qb_obj in self.get_qubits()[0]:
                    tmp_vals += [
                        # read out at the drive leakage frequency
                        (qb_obj.ro_freq, qb_obj.ge_freq() -
                         qb_obj.ge_mod_freq()),
                        (qb_obj.acq_weights_type, 'SSB'),
                        (qb_obj.instr_trigger.get_instr().pulse_period,
                         trigger_sep),
                        (qb_obj.instr_pulsar.get_instr().prepend_zeros,
                         kw.get('prepend_zeros', 0)),
                        *qb_obj._drive_mixer_calibration_tmp_vals()
                    ]
            except Exception as x:
                log.warning('Qubit objects not found. Temporary values '
                            'for drive mixer calibration specified in the '
                            'qubit objects could not be set.')
            # Preprocess sweep points and tasks before creating the sequences
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # Check that bounds of the sweep_points grid do not exceed
            # 1 V as this might damage the diodes inside the mixers.
            for task in self.preprocessed_task_list:
                for sp in task['sweep_points']:
                    for name, p in sp.items():
                        if name in self.kw_for_sweep_points:
                            if np.max(p[0]) > 1.0:
                                log.error('Measurement grid contains DC '
                                          'amplitudes above 1 V. Too high '
                                          'DC biases can potentially damage'
                                          ' the diodes inside the mixer. \n'
                                          'Maximum amplitude is {:.2f} mV!'.format(
                                    1e3 * np.max(p[0])))
            self.resolve_offset_sweep_points(**kw)
            self.generate_sweep_functions()

            with temporary_value(*tmp_vals):
                self.update_operation_dict()
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.sweep_block,
                    block_align=['center', 'end', 'center'], **kw)

            # updates temporary_value for self.autorun()
            self.temporary_values += tmp_vals
            # sets number of measurement points to number sweep points
            self.mc_points = [np.arange(n) for n in self.sweep_points.length()]
            # run measurement & analysis if requested in kw
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def get_sweep_points_for_sweep_n_dim(self):
        return self.sweep_points_pulses

    def resolve_offset_sweep_points(self, **kw):
        """
        This function is called from the init of the class to resolve the
        voltage offset sweep points. The respective sweep functions are
        stored in self.sweep_functions_dict.
        """
        for task in self.preprocessed_task_list:
            if np.any([k in self.kw_for_sweep_points for k in task.keys()]):
                qb_obj = self.get_qubits(task['qb'])[0][0]
                sweep_fcts = self.get_offset_swfs(qb_obj)
                self.sweep_functions_dict.update({
                    task['prefix'] + 'offset_i': sweep_fcts[0],
                    task['prefix'] + 'offset_q': sweep_fcts[1]})

    def sweep_block(self, sweep_points, qb, **kw):
        """This function creates the blocks for a single mixer carrier
        measurement.

        Args:
            sweep_points (SweepPoints): SweepPoints object
            qb (QuDev_transmon): target qubit

        Returns:
            list of :class:`~pycqed.measurement.waveform_control.block.Block`s:
                List of blocks for the operation.
        """
        # extract AWG channels of target qubit
        tmp_p = self.block_from_ops('tmp', [f'X180 {qb}'])
        I_channel = tmp_p.pulses[0]['I_channel']
        Q_channel = tmp_p.pulses[0]['Q_channel']
        mod_frequency = tmp_p.pulses[0]['mod_frequency']

        # extract acq_length if qubit object is available
        try:
            qb_obj = self.get_qubits(qb)[0][0]
            acq_length = qb_obj.acq_length()
        except Exception as x:
            acq_length = 250e-9
            log.warning('Qubit object not found. Acquisition length is set '
                        f'to default value {acq_length}.')
        dp_block = self.block_from_pulse_dicts([dict(
                    pulse_type='GaussFilteredCosIQPulse',
                    pulse_length=acq_length,
                    ref_point='start',
                    amplitude=0,
                    I_channel=I_channel,
                    Q_channel=Q_channel,
                    mod_frequency=mod_frequency,
                    phase_lock=False,
                )], 'drive')

        # create ParametricValues from param_name in sweep_points
        # (e.g. "amplitude", "length", etc.)
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in dp_block.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        acq_block = self.block_from_ops('Acq', f'Acq {qb}')
        block = self.simultaneous_blocks('mixer_calib', [dp_block, acq_block])
        # return all generated blocks (parallel_sweep will arrange them)
        return [block]

    def get_offset_swfs(self, qb_obj):
        """
        Returns sweep functions for the AWG offset voltage parameters of the
        target qubit.
        Args:
            qb_obj (QuDev_transmon): Target qubit

        Returns list(sweep_function.Sweep_function): list of sweep functions

        """
        pulsar = qb_obj.instr_pulsar.get_instr()
        chI_parameter = pulsar.parameters[
            '{}_offset'.format(qb_obj.ge_I_channel())]
        chQ_parameter = pulsar.parameters[
            '{}_offset'.format(qb_obj.ge_Q_channel())]

        sweep_fcts = [wrap_par_to_swf(chI_parameter),
                      wrap_par_to_swf(chQ_parameter)]

        return sweep_fcts

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        Args:
            analysis_kwargs (dict): keyword arguments for analysis
            **kw: currently ignored
        Returns: the analysis instance
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        if 'TwoD' not in analysis_kwargs['options_dict']:
            if len(self.sweep_points) == 2:
                analysis_kwargs['options_dict']['TwoD'] = True
        self.analysis = tda.MixerCarrierAnalysis(
            t_start=self.timestamp, **analysis_kwargs)
        return self.analysis

    def run_update(self, **kw):
        """
        Updates qubit parameters 'qb.ge_I_offset' and 'qb.ge_Q_offset' with
        the fitted values from the analysis of this measurement.
        If the optimal values lie beyond the sweep point ranges, no values
        are updated.
        Args:
            **kw: currently ignored
        """
        assert self.meas_objs is not None, \
            "Update only works with qubit objects provided."
        assert self.analysis is not None, \
            "Update is only allowed after running the analysis."
        assert len(self.meas_objs) == 1, \
            "Update only works for one qubit measurement."

        qb_obj = self.meas_objs[0]
        offsets_i = self.sweep_points[0][f'{qb_obj.name}_offset_i'][0]
        offsets_q = self.sweep_points[1][f'{qb_obj.name}_offset_q'][0]

        analysis_params_dict = self.analysis.proc_data_dict[
            'analysis_params_dict']

        _offset_i = analysis_params_dict['V_I']
        _offset_q = analysis_params_dict['V_Q']

        update = True
        if (_offset_i < offsets_i.min() or _offset_i > offsets_i.max()):
            log.warning('Optimum for DC bias voltage I channel is outside '
                        'the measured range and no settings will be updated. '
                        'Best V_I according to fitting: {:.2f} mV'.format(
                _offset_i))
            update = False
        if (_offset_q < offsets_q.min() or _offset_q > offsets_q.max()):
            log.warning('Optimum for DC bias voltage Q channel is outside '
                        'the measured range and no settings will be updated. '
                        'Best V_Q according to fitting: {:.2f} mV'.format(
                _offset_q))
            update = False

        if update:
            qb_obj.ge_I_offset(_offset_i)
            qb_obj.ge_Q_offset(_offset_q)
