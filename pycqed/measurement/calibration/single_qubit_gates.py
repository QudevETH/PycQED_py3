import numpy as np
from copy import copy, deepcopy
import traceback

from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.utilities.general import temporary_value
from pycqed.measurement import multi_qubit_module as mqm
import logging

from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)


class T1FrequencySweep(CalibBuilder):
    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        """
        Flux pulse amplitude measurement used to determine the qubits energy in
        dependence of flux pulse amplitude.

        Timings of sequence

       |          ---|X180|  ------------------------------|RO|
       |          --------| --------- fluxpulse ---------- |


        :param task_list: list of dicts; see CalibBuilder docstring
        :param sweep_points: SweepPoints class instance with first sweep
            dimension describing the flux pulse lengths and second dimension
            either the flux pulse amplitudes, qubit frequencies, or both.
            !!! If both amplitudes and frequencies are provided, they must be
            be specified in the order amplitudes, frequencies as shown:
            [{'pulse_length': (lengths, 's', 'Flux pulse length')},
             {'flux_pulse_amp': (amps, 'V', 'Flux pulse amplitude'),
              'qubit_freqs': (freqs, 'Hz', 'Qubit frequency')}]
            If this parameter is provided it will be used for all qubits.
        :param qubits: list of QuDev_transmon class instances
        :param kw: keyword arguments
            for_ef (bool, default: False): passed to get_cal_points; see
                docstring there.
            spectator_op_codes (list, default: []): see t1_flux_pulse_block
            all_fits (bool, default: True) passed to run_analysis; see
                docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
         - the entry "qb" in each task should contain one qubit name.

        """
        try:
            self.experiment_name = 'T1_frequency_sweep'
            if task_list is None:
                if sweep_points is None or qubits is None:
                    raise ValueError('Please provide either "sweep_points" '
                                     'and "qubits," or "task_list" containing '
                                     'this information.')
                task_list = [{'qb': qb.name} for qb in qubits]
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.analysis = None
            self.data_to_fit = {qb: 'pe' for qb in self.meas_obj_names}
            self.sweep_points = SweepPoints(
                [{}, {}] if self.sweep_points is None else self.sweep_points)
            self.task_list = self.add_amplitude_sweep_points(
                [copy(t) for t in self.task_list], **kw)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = \
                self.parallel_sweep(self.preprocessed_task_list,
                                    self.t1_flux_pulse_block, **kw)
            self.exp_metadata.update({
                "rotation_type": 'global_PCA' if
                    len(self.cal_points.states) == 0 else 'cal_states'
            })

            if kw.get('compression_seg_lim', None) is None:
                # compress the 2nd sweep dimension completely onto the first
                kw['compression_seg_lim'] = \
                    np.product([len(s) for s in self.mc_points]) \
                    + len(self.cal_points.states)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_amplitude_sweep_points(self, task_list=None, **kw):
        """
        If flux pulse amplitudes are not in the sweep_points in each task, but
        qubit frequencies are, then amplitudes will be calculated based on
        the frequencies and the fit_ge_freq_from_flux_pulse_amp qubit parameter.
        sweep_points entry in each task_list will be updated.
        :param task_list: list of dictionaries describing the the measurement
            for each qubit.
        :return: updated task list
        """
        if task_list is None:
            task_list = self.task_list
        # TODO: check combination of sweep points in task and in sweep_points
        for task in task_list:
            sweep_points = task.get('sweep_points', [{}, {}])
            sweep_points = SweepPoints(sweep_points)
            if len(sweep_points) == 1:
                sweep_points.add_sweep_dimension()
            if 'qubit_freqs' in sweep_points[1]:
                qubit_freqs = sweep_points['qubit_freqs']
            elif len(self.sweep_points) >= 2 and \
                    'qubit_freqs' in self.sweep_points[1]:
                qubit_freqs = self.sweep_points['qubit_freqs']
            else:
                qubit_freqs = None
            if 'amplitude' in sweep_points[1]:
                amplitudes = sweep_points['amplitude']
            elif len(self.sweep_points) >= 2 and \
                    'amplitude' in self.sweep_points[1]:
                amplitudes = self.sweep_points['amplitude']
            else:
                amplitudes = None
            qubits, _ = self.get_qubits(task['qb'])
            if qubit_freqs is None and qubits is not None:
                qb = qubits[0]
                qubit_freqs = qb.calculate_frequency(
                    amplitude=amplitudes,
                    **kw.get('vfc_kwargs', {})
                )
                freq_sweep_points = SweepPoints('qubit_freqs', qubit_freqs,
                                                'Hz', 'Qubit frequency')
                sweep_points.update([{}] + freq_sweep_points)
            if amplitudes is None:
                if qubits is None:
                    raise KeyError('qubit_freqs specified in sweep_points, '
                                   'but no qubit objects available, so that '
                                   'the corresponding amplitudes cannot be '
                                   'computed.')
                qb = qubits[0]
                amplitudes = qb.calculate_flux_voltage(
                    frequency=qubit_freqs,
                    flux=qb.flux_parking(),
                    **kw.get('vfc_kwargs', {})
                )
                if np.any(np.isnan(amplitudes)):
                    raise ValueError('Specified frequencies resulted in nan '
                                     'amplitude. Check frequency range!')
                amp_sweep_points = SweepPoints('amplitude', amplitudes,
                                               'V', 'Flux pulse amplitude')
                sweep_points.update([{}] + amp_sweep_points)
            task['sweep_points'] = sweep_points
        return task_list

    def t1_flux_pulse_block(self, qb, sweep_points,
                            prepend_pulse_dicts=None, **kw):
        """
        Function that constructs the experiment block for one qubit
        :param qb: name or list with the name of the qubit
            to measure. This function expect only one qubit to measure!
        :param sweep_points: SweepPoints class instance
        :param prepend_pulse_dicts: dictionary of pulses to prepend
        :param kw: keyword arguments
            spectator_op_codes: list of op_codes for spectator qubits
        :return: precompiled block
        """

        qubit_name = qb
        if isinstance(qubit_name, list):
            qubit_name = qubit_name[0]
        hard_sweep_dict, soft_sweep_dict = sweep_points
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'pi_pulse'}}
        pp = self.block_from_ops('pipulse',
                                 [f'X180 {qubit_name}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)

        pulse_modifs = {
            'all': {'element_name': 'flux_pulse', 'pulse_delay': 0}}
        fp = self.block_from_ops('flux', [f'FP {qubit_name}'],
                                 pulse_modifs=pulse_modifs)
        for k in hard_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)
        for k in soft_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)

        return self.sequential_blocks(f't1 flux pulse {qubit_name}',
                                      [pb, pp, fp])

    @Timer()
    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw:
            all_fits (bool, default: True): whether to do all fits
        """

        self.all_fits = kw.get('all_fits', True)
        self.do_fitting = kw.get('do_fitting', True)
        self.analysis = tda.T1FrequencySweepAnalysis(
            qb_names=self.meas_obj_names,
            do_fitting=self.do_fitting,
            options_dict=dict(TwoD=True, all_fits=self.all_fits,
                              rotation_type='global_PCA' if not
                                len(self.cal_points.states) else 'cal_states'))


class ParallelLOSweepExperiment(CalibBuilder):
    """
    Base class for (parallel) calibration measurements where the LO is swept
    in sweep dimension 1. The class is based on the concept of a
    multitasking experiment, see docstrings of MultiTaskingExperiment and
    of CalibBuilder for general information.

    The following keys in a task are interpreted by this class:
    - qb: identifies the qubit measured in the task (child classes should be
        implemented in a way that they adopt this convention)
    - fluxline: qcodes parameter to adjust the DC flux offset of the qubit.
        If this is provided, the DC flux offset will be swept together with
        the frequency sweep such that the qubit is tuned to the current
        drive frequency (based on the fit_ge_freq_from_dc_offset parameter
        in the qubit object).
    - fp_assisted_ro_calib_flux (to be used in combination with fluxline):
        Specifies the flux (in units of Phi0 with 0 indicating the upper
        sweep spot) for which the flux-pulse-assisted RO of the qubit has
        been calibrated. If this is provided and fluxline is provided,
        the amplitude of the flux pulse assisted RO will be adjusted
        together with the DC offset during the frequency sweep in order to
        keep the qubit frequency during RO the same as in the calibration.
        This requires an HDAWG as flux AWG.

    Note: parameters with a (*) have not been exhaustively tested for
    parallel measurements.

    :param task_list:  see MultiTaskingExperiment
    :param sweep_points: (SweepPoints object or list of dicts or None)
        sweep points valid for all tasks.
    :param allowed_lo_freqs: (list of float or None) (*) if not None,
        it specifies that the LO should only be set to frequencies in the
        list, and the desired drive frequency is obtained by an IF sweep
        using internal modulation of the HDAWG. This requires an HDAWG as
        drive AWG. Consider using the kwarg optimize_mod_freqs together with
        allowed_lo_freqs (see docstring of resolve_freq_sweep_points).
    :param adapt_drive_amp: (bool) (*) if True, the drive amplitude is adapted
        for each drive frequency based on the parameter
        fit_ge_amp180_over_ge_freq of the qubit using the output amplitude
        scaling of the HDAWG. This requires an HDAWG as drive AWG. Note the
        kwarg adapt_cal_point_drive_amp, which can be used together
        with adapt_drive_amp (see docstring of resolve_freq_sweep_points).
    :param adapt_ro_freq: (bool, default: False) (*) if True, the RO LO
        frequency is adapted for each drive frequency based on the parameter
        fit_ro_freq_over_ge_freq of the qubit

    :param kw: keyword arguments.
        The following kwargs are interpreted by resolve_freq_sweep_points
        (see docstring of that method):
        - optimize_mod_freqs
        - adapt_cal_point_drive_amp

        Moreover, keyword arguments are passed to preprocess_task_list,
        parallel_sweep, and to the parent class.
    """

    def __init__(self, task_list, sweep_points=None, allowed_lo_freqs=None,
                 adapt_drive_amp=False, adapt_ro_freq=False, **kw):
        for task in task_list:
            if not isinstance(task['qb'], str):
                task['qb'] = task['qb'].name
            if not 'prefix' in task:
                task['prefix'] = f"{task['qb']}_"

        # Passing keyword arguments to the super class (even if they are not
        # needed there) makes sure that they are stored in the metadata.
        super().__init__(task_list, sweep_points=sweep_points,
                         allowed_lo_freqs=allowed_lo_freqs,
                         adapt_drive_amp=adapt_drive_amp,
                         adapt_ro_freq=adapt_ro_freq, **kw)
        self.lo_offsets = {}
        self.lo_qubits = {}
        self.qb_offsets = {}
        self.lo_sweep_points = []
        self.allowed_lo_freqs = allowed_lo_freqs
        self.adapt_drive_amp = adapt_drive_amp
        self.adapt_ro_freq = adapt_ro_freq
        self.drive_amp_adaptation = {}
        self.ro_freq_adaptation = {}
        self.ro_flux_amp_adaptation = {}
        self.analysis = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        self.resolve_freq_sweep_points(**kw)
        self.sequences, self.mc_points = self.parallel_sweep(
            self.preprocessed_task_list, self.sweep_block, **kw)

    def resolve_freq_sweep_points(self, freq_sp_suffix='freq', **kw):
        """
        This function is called from the init of the class to resolve the
        frequency sweep points and the settings that need to be swept
        together with the frequency. The results are stored in properties of
        the object, which are then used in run_measurement. Aspects to be
        resolved include (if applicable):
        - (shared) LO freqs and (fixed or swept) IFs
        - drive amplitude adaptation
        - RO freq adaptation
        - flux amplitude adaptation of flux-pulse-assisted RO

        :param freq_sp_suffix: (str, default 'freq') To identify the
            frequency sweep parameter, this string specifies a suffix that is
            contained at the end of the name of the frequency sweep parameter.
        :param kw:
            optimize_mod_freqs: (bool, default: False) If False, the
                ge_mod_freq setting of the first qb on an LO (according to
                the ordering of the task list) determines the LO frequency
                for all qubits on that LO. If True, the ge_mod_freq settings
                will be optimized for the following situations:
                - With allowed_lo_freqs set to None: the LO will be placed in
                  the center of the band of drive frequencies of all qubits on
                  that LO (minimizing the maximum absolute value of
                  ge_mod_freq over all qubits). Do not use in case of a single
                  qubit per LO in this case, as it would result in a
                  ge_mod_freq of 0.
                - With a list allowed_lo_freqs provided: the ge_mod_freq
                  setting of the qubit would act as an unnecessary constant
                  offset in the IF sweep, and optimize_mod_freqs can be used
                  to minimize this offset. In this case optimize_mod_freqs
                  can (and should) even be used in case of a single qubit
                  per LO (where it reduces this offset to 0).
            adapt_cal_point_drive_amp: (bool, default: False) If
                adapt_drive_amp is used, this decides whether the drive
                amplitude should also be adapted for calibration points. To
                implement the case where this is False, it is assumed that
                the calibration point of interest is the one at the drive
                frequency configured in ge_freq of the qubit and that the
                ge_amp180 in the qubit is calibrated for that frequency.
        """
        all_freqs = np.array(
            self.sweep_points.get_sweep_params_property('values', 1, 'all'))
        if np.ndim(all_freqs) == 1:
            all_freqs = [all_freqs]
        all_diffs = [np.diff(freqs) for freqs in all_freqs]
        assert all([len(d) == 0 for d in all_diffs]) or \
            all([np.mean(abs(diff - all_diffs[0]) / all_diffs[0]) < 1e-10
                 for diff in all_diffs]), \
            "The steps between frequency sweep points must be the same for " \
            "all qubits."
        self.lo_sweep_points = all_freqs[0] - all_freqs[0][0]
        self.exp_metadata['lo_sweep_points'] = self.lo_sweep_points

        temp_vals = []
        # Determine which qubits share an LO, and update ge_mod_freq settings
        # - for compatibility in parallel LO sweeps with shared LOs
        # - taking into account the optimize_mod_freqs kwarg
        if self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without checking for ge_mod_freq corrections.')
        else:
            f_start = {}
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f_start[qb] = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)[0]
                self.qb_offsets[qb] = f_start[qb] - self.lo_sweep_points[0]
                lo = qb.instr_ge_lo.get_instr()
                if lo not in self.lo_qubits:
                    self.lo_qubits[lo] = [qb]
                else:
                    self.lo_qubits[lo] += [qb]

            for lo, qbs in self.lo_qubits.items():
                for qb in qbs:
                    if lo not in self.lo_offsets:
                        if kw.get('optimize_mod_freqs', False):
                            fs = [f_start[qb] for qb in self.lo_qubits[lo]]
                            self.lo_offsets[lo] = 1 / 2 * (max(fs) + min(fs))
                        else:
                            self.lo_offsets[lo] = f_start[qb] \
                                                  - qb.ge_mod_freq()
                    temp_vals.append(
                        (qb.ge_mod_freq, f_start[qb] - self.lo_offsets[lo]))
            self.exp_metadata['lo_offsets'] = {
                k.name: v for k, v in self.lo_offsets.items()}

        if self.allowed_lo_freqs is not None:
            # HDAWG internal modulation is needed, switch off modulation in
            # the waveform generation
            for task in self.preprocessed_task_list:
                task['pulse_modifs'] = {'attr=mod_frequency': None}
            self.cal_points.pulse_modifs = {'attr=mod_frequency': None}

        # If applicable, configure drive amplitude adaptation based on the
        # models stored in the qubit objects.
        if self.adapt_drive_amp and self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without adapting drive amp.')
        elif self.adapt_drive_amp:
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)
                amps = qb.get_ge_amp180_from_ge_freq(np.array(f))
                if amps is None:
                    continue
                max_amp = np.max(amps)
                temp_vals.append((qb.ge_amp180, max_amp))
                self.drive_amp_adaptation[qb] = (
                    lambda x, qb=qb, s=max_amp,
                           o=self.qb_offsets[qb] :
                    qb.get_ge_amp180_from_ge_freq(x + o) / s)
                if not kw.get('adapt_cal_point_drive_amp', False):
                    if self.cal_points.pulse_modifs is None:
                        self.cal_points.pulse_modifs = {}
                    self.cal_points.pulse_modifs.update(
                        {f'op_code=X180 {qb.name},attr=amplitude':
                            qb.ge_amp180() / (qb.get_ge_amp180_from_ge_freq(
                                qb.ge_freq()) / max_amp)})
            self.exp_metadata['drive_amp_adaptation'] = {
                qb.name: fnc(self.lo_sweep_points)
                for qb, fnc in self.drive_amp_adaptation.items()}

        # If applicable, configure RO frequency adaptation based on the
        # models stored in the qubit objects.
        if self.adapt_ro_freq and self.qubits is None:
            log.warning('No qubit objects provided. Creating the sequence '
                        'without adapting RO freq.')
        elif self.adapt_ro_freq:
            for task in self.task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                if qb.get_ro_freq_from_ge_freq(qb.ge_freq()) is None:
                    continue
                ro_mwg = qb.instr_ro_lo.get_instr()
                if ro_mwg in self.ro_freq_adaptation:
                    raise NotImplementedError(
                        f'RO adaptation for {qb.name} with LO {ro_mwg.name}: '
                        f'Parallel RO frequency adaptation for qubits '
                        f'sharing an LO is not implemented.')
                self.ro_freq_adaptation[ro_mwg] = (
                    lambda x, mwg=ro_mwg, o=self.qb_offsets[qb],
                           f_mod=qb.ro_mod_freq():
                    qb.get_ro_freq_from_ge_freq(x + o) - f_mod)
            self.exp_metadata['ro_freq_adaptation'] = {
                mwg.name: fnc(self.lo_sweep_points)
                for mwg, fnc in self.ro_freq_adaptation.items()}

        # If applicable, configure flux amplitude adaptation for
        # flux-pulse-assisted RO (based on the models stored in the qubit
        # objects) for cases where the DC offset is swept together with the
        # frequency sweep.
        for task in self.task_list:
            if 'fp_assisted_ro_calib_flux' in task and 'fluxline' in task:
                if self.qubits is None:
                    log.warning('No qubit objects provided. Creating the '
                                'sequence without RO flux amplitude.')
                    break
                qb = self.get_qubits(task['qb'])[0][0]
                if qb.ro_pulse_type() != 'GaussFilteredCosIQPulseWithFlux':
                    continue
                sp = self.exp_metadata['meas_obj_sweep_points_map'][qb.name]
                freq_sp = [s for s in sp if s.endswith(freq_sp_suffix)][0]
                f = self.sweep_points.get_sweep_params_property(
                    'values', 1, freq_sp)
                ro_fp_amp = lambda x, qb=qb, cal_flux=task[
                    'fp_assisted_ro_calib_flux'] : qb.ro_flux_amplitude() - (
                        qb.calculate_flux_voltage(x) -
                        qb.calculate_voltage_from_flux(cal_flux)) \
                        * qb.flux_amplitude_bias_ratio()
                amps = ro_fp_amp(f)
                max_amp = np.max(np.abs(amps))
                temp_vals.append((qb.ro_flux_amplitude, max_amp))
                self.ro_flux_amp_adaptation[qb] = (
                    lambda x, fnc=ro_fp_amp, s=max_amp, o=self.qb_offsets[qb]:
                    fnc(x + o) / s)
                if 'ro_flux_amp_adaptation' not in self.exp_metadata:
                    self.exp_metadata['ro_flux_amp_adaptation'] = {}
                self.exp_metadata['ro_flux_amp_adaptation'][qb.name] = \
                    amps / max_amp

        with temporary_value(*temp_vals):
            self.update_operation_dict()

    def run_measurement(self, **kw):
        """
        Configures additional sweep functions and temporary values for the
        functionality configured in resolve_freq_sweep_points, before calling
        the method of the base class.
        """
        temp_vals = []
        name = 'Drive frequency shift'
        sweep_functions = [swf.Offset_Sweep(
            lo.frequency, offset, name=name, parameter_name=name, unit='Hz')
            for lo, offset in self.lo_offsets.items()]
        if self.allowed_lo_freqs is not None:
            minor_sweep_functions = []
            for lo, qbs in self.lo_qubits.items():
                qb_sweep_functions = []
                for qb in qbs:
                    mod_freq = self.get_pulse(f"X180 {qb.name}")[
                        'mod_frequency']
                    pulsar = qb.instr_pulsar.get_instr()
                    # Pulsar assumes that the first channel in a pair is the
                    # I component. If this is not the case, the following
                    # workaround finds the correct channel to configure
                    # and swaps the sign of the modulation frequency to get
                    # the correct sideband.
                    iq_swapped = (int(qb.ge_I_channel()[-1:])
                                  > int(qb.ge_Q_channel()[-1:]))
                    param = pulsar.parameters[
                        f'{qb.ge_Q_channel()}_mod_freq' if iq_swapped else
                        f'{qb.ge_I_channel()}_mod_freq']
                    # The following temporary value ensures that HDAWG
                    # modulation is set back to its previous state after the end
                    # of the modulation frequency sweep.
                    temp_vals.append((param, None))
                    qb_sweep_functions.append(
                        swf.Transformed_Sweep(param, transformation=(
                            lambda x, o=mod_freq, s=(-1 if iq_swapped else 1)
                            : s * (x + o))))
                minor_sweep_functions.append(swf.multi_sweep_function(
                    qb_sweep_functions))
            sweep_functions = [
                swf.MajorMinorSweep(majsp, minsp,
                                    np.array(self.allowed_lo_freqs) - offset)
                for majsp, minsp, offset in zip(
                    sweep_functions, minor_sweep_functions,
                    self.lo_offsets.values())]
        for qb, adaptation in self.drive_amp_adaptation.items():
            adapt_name = f'Drive amp adaptation freq {qb.name}'
            pulsar = qb.instr_pulsar.get_instr()
            for quad in ['I', 'Q']:
                ch = qb.get(f'ge_{quad}_channel')
                param = pulsar.parameters[f'{ch}_amplitude_scaling']
                sweep_functions += [swf.Transformed_Sweep(
                    param, transformation=adaptation,
                    name=adapt_name, parameter_name=adapt_name, unit='Hz')]
                # The following temporary value ensures that HDAWG
                # amplitude scaling is set back to its previous state after the
                # end of the sweep.
                temp_vals.append((param, 1.0))
        for mwg, adaptation in self.ro_freq_adaptation.items():
            adapt_name = f'RO freq adaptation freq {mwg.name}'
            param = mwg.frequency
            sweep_functions += [swf.Transformed_Sweep(
                param, transformation=adaptation,
                name=adapt_name, parameter_name=adapt_name, unit='Hz')]
            temp_vals.append((param, param()))
        for qb, adaptation in self.ro_flux_amp_adaptation.items():
            adapt_name = f'RO flux amp adaptation freq {qb.name}'
            pulsar = qb.instr_pulsar.get_instr()
            ch = qb.get(f'ro_flux_channel')
            for seg in self.sequences[0].segments.values():
                for p in seg.unresolved_pulses:
                    if (ch in p.pulse_obj.channels and
                        p.pulse_obj.pulse_type
                            != 'GaussFilteredCosIQPulseWithFlux'):
                        raise NotImplementedError(
                            'RO flux amp adaptation cannot be used when the '
                            'sequence contains other flux pulses.')
            param = pulsar.parameters[f'{ch}_amplitude_scaling']
            sweep_functions += [swf.Transformed_Sweep(
                param, transformation=adaptation,
                name=adapt_name, parameter_name=adapt_name, unit='Hz')]
            temp_vals.append((param, param()))
        for task in self.task_list:
            if 'fluxline' not in task:
                continue
            qb = self.get_qubits(task['qb'])[0][0]
            dc_amp = (lambda x, o=self.qb_offsets[qb], qb=qb:
                      qb.calculate_flux_voltage(x + o))
            sweep_functions += [swf.Transformed_Sweep(
                task['fluxline'], transformation=dc_amp,
                name=f'DC Offset {qb.name}',
                parameter_name=f'Parking freq {qb.name}', unit='Hz')]
        self.sweep_functions = [
            self.sweep_functions[0], swf.multi_sweep_function(
                sweep_functions, name=name, parameter_name=name)]
        self.mc_points[1] = self.lo_sweep_points
        with temporary_value(*temp_vals):
            super().run_measurement(**kw)

    def get_meas_objs_from_task(self, task):
        return [task['qb']]

    def sweep_block(self, **kw):
        raise NotImplementedError('Child class has to implement sweep_block.')


class FluxPulseScope(ParallelLOSweepExperiment):
    """
        flux pulse scope measurement used to determine the shape of flux pulses
        set up as a 2D measurement (delay and drive pulse frequecy are
        being swept)
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |

            sweep_points:
            delay (numpy array): array of amplitudes of the flux pulse
            freq (numpy array): array of drive frequencies

        Returns: None

    """
    kw_for_task_keys = ['ro_pulse_delay', 'fp_truncation',
                        'fp_truncation_buffer',
                        'fp_compensation',
                        'fp_compensation_amp',
                        'fp_during_ro', 'tau',
                        'fp_during_ro_length',
                        'fp_during_ro_buffer']
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'delays': dict(param_name='delay', unit='s',
                       label=r'delay, $\tau$',
                       dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'Flux_scope'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, flux_op_code=None,
                    ro_pulse_delay=None,
                    fp_truncation=False, fp_compensation=False,
                    fp_compensation_amp=None, fp_truncation_buffer=None,
                    fp_during_ro=False, tau=None,
                    fp_during_ro_length=None,
                    fp_during_ro_buffer=None, **kw):
        """
        Performs X180 pulse on top of a fluxpulse
        Timings of sequence
        |          ----------           |X180|  ----------------------------  |RO|
        |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->

        :param qb: (str) the name of the qubit
        :param sweep_points: the sweep points containing a parameter delay
            in dimension 0
        :param flux_op_code: (optional str) the flux pulse op_code (default
            FP qb)
        :param ro_pulse_delay: Can be 'auto' to start the readout after
            the end of the flux pulse (or in the middle of the readout-flux-pulse
            if fp_during_ro is True) or a delay in seconds to start a fixed
            amount of time after the drive pulse. If not provided or set to
            None, a default fixed delay of 100e-9 is used. If fp_compensation is
            used, the readout pulse is referenced to the end of the flux
            compensation pulse.
        :param fp_truncation: Truncate the flux pulse after the drive pulse
        :param fp_truncation_buffer: Time buffer after the drive pulse, before
            the truncation happens.
        :param fp_compensation: Truncated custom compensation calculated from a
            first order distortion with dominant time constant tau. Standard
            compensation has to be turned off manually.
        :param fp_compensation_amp: Fixed amplitude for the custom compensation
            pulse.
        :param fp_during_ro: Play a flux pulse during the read-out pulse to
            bring the qubit actively to the parking position in the case where
            the flux-pulse is not filtered yet. This assumes a unipolar flux-pulse.
        :param fp_during_ro_length: Length of the fp_during_ro.
        :param fp_during_ro_buffer: Time buffer between the drive pulse and
            the fp_during_ro
        :param tau: Approximate dominant time constant in the flux line, which
            is used to calculate the amplitude of the fp_during_ro.

        :param kw:
        """
        if flux_op_code is None:
            flux_op_code = f'FP {qb}'
        if ro_pulse_delay is None:
            ro_pulse_delay = 100e-9
        if fp_truncation_buffer is None:
            fp_truncation_buffer = 5e-8
        if fp_compensation_amp is None:
            fp_compensation_amp = -2
        if tau is None:
            tau = 20e-6
        if fp_during_ro_length is None:
            fp_during_ro_length = 2e-6
        if fp_during_ro_buffer is None:
            fp_during_ro_buffer = 0.2e-6

        if ro_pulse_delay is 'auto' and (fp_truncation or \
            hasattr(fp_truncation, '__iter__')):
            raise Exception('fp_truncation does currently not work ' + \
                            'with the auto mode of ro_pulse_delay.')

        assert not (fp_compensation and fp_during_ro)

        pulse_modifs = {'attr=name,op_code=X180': f'FPS_Pi',
                        'attr=element_name,op_code=X180': 'FPS_Pi_el'}
        b = self.block_from_ops(f'ge_flux {qb}',
                                [f'X180 {qb}'] + [flux_op_code] * \
                                (2 if fp_compensation else 1) \
                                + ([f'FP {qb}'] if fp_during_ro else []),
                                pulse_modifs=pulse_modifs)

        fp = b.pulses[1]
        fp['ref_point'] = 'middle'
        bl_start = fp.get('buffer_length_start', 0)
        bl_end = fp.get('buffer_length_end', 0)

        def fp_delay(x, o=bl_start):
            return -(x+o)

        fp['pulse_delay'] = ParametricValue(
            'delay', func=fp_delay)

        fp_length_function = lambda x: fp['pulse_length']

        if (fp_truncation or hasattr(fp_truncation, '__iter__')):
            if not hasattr(fp_truncation, '__iter__'):
                fp_truncation = [-np.inf, np.inf]
            original_fp_length = fp['pulse_length']
            max_fp_sweep_length = np.max(
                sweep_points.get_sweep_params_property(
                    'values', dimension=0, param_names='delay'))
            sweep_diff = max(max_fp_sweep_length - original_fp_length, 0)
            fp_length_function = lambda x, opl=original_fp_length, \
                o=bl_start + fp_truncation_buffer, trunc=fp_truncation: \
                max(min((x + o), opl), 0) if (x>np.min(trunc) and x<np.max(trunc)) else opl

            fp['pulse_length'] = ParametricValue(
                'delay', func=fp_length_function)
            if fp_compensation:
                cp = b.pulses[2]
                cp['name'] = 'FPS_FPC'
                cp['amplitude'] = -np.sign(fp['amplitude']) * np.abs(
                    fp_compensation_amp)
                cp['pulse_delay'] = sweep_diff + bl_start

                def t_trunc(x, fnc=fp_length_function, tau=tau,
                            fp_amp=fp['amplitude'], cp_amp=cp['amplitude']):
                    fp_length = fnc(x)

                    def v_c(tau, fp_length, fp_amp, v_c_start=0):
                        return fp_amp - (fp_amp - v_c_start) * np.exp(
                            -fp_length / tau)

                    v_c_fp = v_c(tau, fp_length, fp_amp, v_c_start=0)
                    return -np.log(cp_amp / (cp_amp - v_c_fp)) * tau

                cp['pulse_length'] = ParametricValue('delay', func=t_trunc)

        # assumes a unipolar flux-pulse for the calculation of the
        # amplitude decay.
        if fp_during_ro:
            rfp = b.pulses[2]

            def rfp_delay(x, fp_delay=fp_delay, fp_length=fp_length_function,\
                fp_bl_start=bl_start, fp_bl_end=bl_end):
                return -(fp_length(x)+fp_bl_end+fp_delay(x))

            def rfp_amp(x, fp_delay=fp_delay, rfp_delay=rfp_delay, tau=tau,
                fp_amp=fp['amplitude'], o=fp_during_ro_buffer-bl_start):
                fp_length=-fp_delay(x)+o
                if fp_length <= 0:
                    return 0
                elif rfp_delay(x) < 0:
                    # in the middle of the fp
                    return -fp_amp * np.exp(-fp_length / tau)
                else:
                    # after the end of the fp
                    return fp_amp * (1 - np.exp(-fp_length / tau))

            rfp['pulse_length'] = fp_during_ro_length
            rfp['pulse_delay'] = ParametricValue('delay', func=rfp_delay)
            rfp['amplitude'] = ParametricValue('delay', func=rfp_amp)
            rfp['buffer_length_start'] = fp_during_ro_buffer

        if ro_pulse_delay == 'auto':
            if fp_during_ro:
                # start the ro pulse in the middle of the fp_during_ro pulse
                delay = fp_during_ro_buffer + fp_during_ro_length/2
                b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'end',
                                    'pulse_delay': delay})
            else:
                delay = \
                    fp['pulse_length'] - np.min(
                        sweep_points.get_sweep_params_property(
                            'values', dimension=0, param_names='delay')) + \
                    fp.get('buffer_length_end', 0) + fp.get('trans_length', 0)
                b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'middle',
                                    'pulse_delay': delay})
        elif fp_compensation:
            b.block_end.update({'ref_pulse': 'FPS_FPC', 'ref_point': 'end',
                                'pulse_delay': ro_pulse_delay})
        else:
            b.block_end.update({'ref_pulse': 'FPS_Pi', 'ref_point': 'end',
                                'pulse_delay': ro_pulse_delay})

        self.data_to_fit.update({qb: 'pe'})
        return b

    @Timer()
    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw:
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        options_dict = {'rotation_type': 'fixed_cal_points' if
                            len(self.cal_points.states) > 0 else 'global_PCA',
                        'TwoD': True}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        analysis_kwargs['options_dict'].update(options_dict)

        self.analysis = tda.FluxPulseScopeAnalysis(
            qb_names=self.meas_obj_names, **analysis_kwargs)


class Cryoscope(CalibBuilder):
    """
    Delft Cryoscope measurement
    (https://aip.scitation.org/doi/pdf/10.1063/1.5133894)
    used to determine the shape of flux pulses set up as a 2D measurement
    (truncation length and phase of second pi-half pulse are being swept)
    Timings of sequence
    |  --- |Y90| ------------------------------------------- |Y90| -  |RO|
    |  ------- | ------ fluxpulse ------ | separation_buffer | -----
    <-  truncation_length  ->

    :param task_list: list of dicts, where each dict contains the parameters of
        a task (= keyword arguments for the block creation function)
    :param sweep_points: SweepPoints class instance. Can also be specified
        separately in each task.
    :param estimation_window: (float or None) delta_tau in the cryoscope paper.
        The extra bit of flux pulse length before truncation in the second
        Ramsey measurement. If None, only one set of Ramsey measurements are
        done. Can also be specified separately in each task.
    :param separation_buffer: (float) extra delay between the (truncated)
        flux pulse and the last pi-half pulse. Can also be specified separately
        in each task.
    :param awg_sample_length: (float) the length of one sample on the flux
        AWG used by the measurement objects in this experiment. Can also be
        specified separately in each task.
    :param sequential: (bool) whether to apply the cryoscope pulses sequentially
        (True) or simultaneously on n-qubits
    :param kw: keyword arguments: passed down to parent class(es)

    The sweep_points for this measurements must contain
        - 0'th sweep dimension: the Ramsey phases, and, optionally, the
        extra_truncation_lengths, which are 0ns for the first Ramsey (first set
        of phases) and the estimation_window for the second Ramsey. This sweep
        dimension is added automatically in add_default_hard_sweep_points;
        user can specify nr_phases and the estimation_window.
        - 1'st sweep dimension: main truncation lengths that specify the length
        of the pulse(s) at each cryoscope point.


    How to use this class:
        - each task must contain the qubit that is measured under the key "qb"
        - specify sweep_points, estimation_window, separation_buffer,
         awg_sample_length either globally as input parameters to the class
         instantiation, or specify them in each task
        - several flux pulses to measure (n pulses between Ramsey pulses):
            - specify the flux_pulse_dicts in each task. This is a list of
            dicts, where each dict can contain the following:
             {'op_code': flux_op_code,
              'truncation_lengths': numpy array
              'spacing': float, # np.arange(0, tot_pulse_length, spacing)
              'nr_points': int, # np.linspace(0, tot_pulse_length, nr_points,
                                              endpoint=True)}
            If truncation_length is given, it will ignore spacing and nr_points.
            If spacing is given, it will ignore nr_points.
            !!! This entry has priority over Option1 below.
        - only one flux pulse to measure (one pulse between Ramsey pulses):
            Option1 :
                - specify the truncation lengths sweep points globally or in
                    each task.
                - optionally, specify flux_op_code in each task
            Option2:
                - specify the flux_pulse_dicts with one dict in the list
        - for any of the cases described above, the user can specify the
            reparking_flux_pulse entry in each task. This entry is a dict that
            specifies the pulse pars for a reparking flux pulse that will be
            applied on top of the flux pulse(s) that are measured by the
            cryoscope (between the 2 Ramsey pulses). The reparking_flux_pulse
            dict must contain at least the 'op_code' entry.
        - for any of the cases described above, the user can specify the
            prepend_pulse_dicts entry in each task.
            See CalibBuilder.block_from_pulse_dicts() for details.

    Example of a task with all possible entry recognized by this class.
    See above for details on how they are used and which ones have priority
        {'qb': qb,
        'flux_op_code': flux_op_code,
        'sweep_points': SweepPoints instance,
        'flux_pulse_dicts': [{'op_code': flux_op_code0,
                              'spacing': 2*hdawg_sample_length,
                              'nr_points': 10,
                              'truncation_lengths': array},
                             {'op_code': flux_op_code1,
                              'spacing': 2*hdawg_sample_length,
                              'nr_points': 10,
                              'truncation_lengths': array}],
        'awg_sample_length': hdawg_sample_length,
        'estimation_window': hdawg_sample_length,
        'separation_buffer': 50e-9,
        'reparking_flux_pulse': {'op_code': f'FP {qb.name}',
                                 'amplitude': -0.5}}
    """

    def __init__(self, task_list, sweep_points=None, estimation_window=None,
                 separation_buffer=50e-9, awg_sample_length=None,
                 sequential=False, **kw):
        try:
            self.experiment_name = 'Cryoscope'
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb']}_"
                if 'awg_sample_length' not in task:
                    task['awg_sample_length'] = awg_sample_length
                if 'estimation_window' not in task:
                    task['estimation_window'] = estimation_window
                if 'separation_buffer' not in task:
                    task['separation_buffer'] = separation_buffer
            # check estimation window
            none_est_windows = [task['estimation_window'] is None for task in
                                task_list]
            if any(none_est_windows) and not all(none_est_windows):
                raise ValueError('Some tasks have estimation_window == None. '
                                 'You can have different values for '
                                 'estimation_window in different tasks, but '
                                 'none these can be None. To use the same '
                                 'estimation window for all tasks, you can set '
                                 'the class input parameter estimation_window.')

            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.sequential = sequential
            self.blocks_to_save = {}
            self.add_default_soft_sweep_points(**kw)
            self.add_default_hard_sweep_points(**kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.sweep_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)
            self.add_blocks_to_metadata()
            self.update_sweep_points(**kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_soft_sweep_points(self, **kw):
        """
        Adds soft sweep points (truncation_lengths) to each task in
        self.task_list if flux_pulse_dicts in task. The truncation_lengths
        array is a concatenation of the truncation lengths created between 0 and
        total length of each pulse in flux_pulse_dicts.
        I also adds continuous_truncation_lengths to each task which contains
        the continuous-time version of the truncation_lengths described above.
        :param kw: keyword_arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        for task in self.task_list:
            awg_sample_length = task['awg_sample_length']
            if 'flux_pulse_dicts' not in task:
                if 'sweep_points' not in task:
                    raise ValueError(f'Please provide either "sweep_points" '
                                     f'or "flux_pulse_dicts" in the task dict '
                                     f'for {task["qb"]}.')
                continue
            else:
                if awg_sample_length is None:
                    raise ValueError(f'Please provide the length of one sample '
                                     f'for the flux AWG of {task["qb"]}')

            flux_pulse_dicts = task['flux_pulse_dicts']
            trunc_lengths = len(flux_pulse_dicts) * ['']
            continuous_trunc_lengths = len(flux_pulse_dicts) * ['']
            for i, fpd in enumerate(flux_pulse_dicts):
                pd_temp = {'element_name': 'dummy'}
                pd_temp.update(self.get_pulse(fpd['op_code']))
                pulse_length = seg_mod.UnresolvedPulse(pd_temp).pulse_obj.length
                if 'truncation_lengths' in fpd:
                    tr_lens = fpd['truncation_lengths']
                elif 'spacing' in fpd:
                    tr_lens = np.arange(0, pulse_length, fpd['spacing'])
                    if not np.isclose(tr_lens[-1], pulse_length):
                        tr_lens = np.append(tr_lens, pulse_length)
                    tr_lens -= tr_lens % awg_sample_length
                    tr_lens += awg_sample_length/2
                elif 'nr_points' in fpd:
                    tr_lens = np.linspace(0, pulse_length, fpd['nr_points'],
                                          endpoint=True)
                    tr_lens -= tr_lens % awg_sample_length
                    tr_lens += awg_sample_length/2
                elif 'truncation_lengths' in fpd:
                    tr_lens = fpd['truncation_lengths']
                else:
                    raise ValueError(f'Please specify either "delta_tau" or '
                                     f'"nr_points" or "truncation_lengths" '
                                     f'for {task["qb"]}')

                trunc_lengths[i] = tr_lens
                task['flux_pulse_dicts'][i]['nr_points'] = len(tr_lens)
                if i:
                    continuous_trunc_lengths[i] = \
                        tr_lens + continuous_trunc_lengths[i-1][-1]
                else:
                    continuous_trunc_lengths[i] = tr_lens

            sp = task.get('sweep_points', SweepPoints())
            sp.update(SweepPoints('truncation_length',
                                  np.concatenate(trunc_lengths),
                                  's', 'Length', dimension=1))
            task['sweep_points'] = sp
            task['continuous_truncation_lengths'] = np.concatenate(
                continuous_trunc_lengths)

    def add_default_hard_sweep_points(self, **kw):
        """
        Adds hard sweep points to self.sweep_points: phases of second pi-half
        pulse and the estimation_window increment to the truncation_length,
        if provided.
        :param kw: keyword_arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        none_est_windows = [task['estimation_window'] is None for task in
                            self.task_list]
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, tile=0 if any(none_est_windows) else 2,
            repeat=0, **kw)

        for task in self.task_list:
            estimation_window = task['estimation_window']
            if estimation_window is None:
                log.warning(f'estimation_window is missing for {task["qb"]}. '
                            f'The global parameter estimation_window is also '
                            f'missing.\nDoing only one Ramsey per truncation '
                            f'length.')
            else:
                nr_phases = self.sweep_points.length(0) // 2
                task_sp = task.pop('sweep_points', SweepPoints())
                task_sp.update(SweepPoints('extra_truncation_length',
                                           [0] * nr_phases +
                                           [estimation_window] * nr_phases,
                                           's', 'Pulse length', dimension=0))
                task['sweep_points'] = task_sp

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Performs a Ramsey phase measurement with a truncated flux pulse between
        the two pi-half pulses.
        Timings of sequence
        |  --- |Y90| ------------------------------------------- |Y90| -  |RO|
        |  ------- | ------ fluxpulse ------ | separation_buffer | -----
                    <-  truncation_length  ->

        :param sp1d_idx: (int) index of sweep point to use from the
            first sweep dimension
        :param sp2d_idx: (int) index of sweep point to use from the
            second sweep dimension
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)

        Assumptions:
            - uses the sweep_points entry in each task. If more than one pulse
            between the two Ramsey pulses, then assumes the sweep_points are a
            concatenation of the truncation_lengths array for each pulse,
            defined between 0 and total length of each pulse.
        """
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task['sweep_points']
            qb = task['qb']

            # pi half pulses blocks
            pihalf_1_bk = self.block_from_ops(f'pihalf_1_{qb}', [f'Y90 {qb}'])
            pihalf_2_bk = self.block_from_ops(f'pihalf_2_{qb}', [f'Y90 {qb}'])
            # set hard sweep phase and delay of second pi-half pulse
            pihalf_2_bk.pulses[0]['phase'] = \
                sweep_points.get_sweep_params_property(
                    'values', 0, 'phase')[sp1d_idx]
            pihalf_2_bk.pulses[0]['pulse_delay'] = task['separation_buffer']

            # pulses to prepend
            prep_bk = self.block_from_pulse_dicts(
                task.get('prepend_pulse_dicts', {}))

            # pulse(s) to measure with cryoscope
            if 'flux_pulse_dicts' in task:
                ops = [fpd['op_code'] for fpd in task['flux_pulse_dicts']]
                main_fpbk = self.block_from_ops(f'fp_main_{qb}', ops)
                n_pts_per_pulse = [fpd['nr_points'] for fpd in
                                   task['flux_pulse_dicts']]
                mask = (np.cumsum(n_pts_per_pulse) <= sp2d_idx)
                meas_pulse_idx = np.count_nonzero(mask)
                # set soft sweep truncation_length
                main_fpbk.pulses[meas_pulse_idx]['truncation_length'] = \
                    sweep_points.get_sweep_params_property(
                        'values', 1, 'truncation_length')[sp2d_idx]
                if task['estimation_window'] is not None:
                    # set hard sweep truncation_length
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] += \
                        sweep_points.get_sweep_params_property(
                            'values', 0, 'extra_truncation_length')[sp1d_idx]
                # for the pulses that come after the pulse that is currently
                # truncated, set all their amplitude parameters to 0
                for pidx in range(meas_pulse_idx+1, len(n_pts_per_pulse)):
                    for k in main_fpbk.pulses[pidx]:
                        if 'amp' in k:
                            main_fpbk.pulses[pidx][k] = 0
            else:
                flux_op_code = task.get('flux_op_code', None)
                if flux_op_code is None:
                    flux_op_code = f'FP {qb}'
                ops = [flux_op_code]
                main_fpbk = self.block_from_ops(f'fp_main_{qb}', ops)
                meas_pulse_idx = 0
                # set soft sweep truncation_length
                for k in sweep_points[1]:
                    main_fpbk.pulses[meas_pulse_idx][k] = \
                        sweep_points.get_sweep_params_property('values', 1, k)[
                            sp2d_idx]
                if task['estimation_window'] is not None:
                    # set hard sweep truncation_length
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] += \
                        sweep_points.get_sweep_params_property(
                            'values', 0, 'extra_truncation_length')[sp1d_idx]

            # reparking flux pulse
            if 'reparking_flux_pulse' in task:
                reparking_fp_params = task['reparking_flux_pulse']
                if 'pulse_length' not in reparking_fp_params:
                    # set pulse length
                    reparking_fp_params['pulse_length'] = self.get_ops_duration(
                        pulses=main_fpbk.pulses)

                repark_fpbk = self.block_from_ops(
                    f'fp_repark_{qb}', reparking_fp_params['op_code'],
                    pulse_modifs={0: reparking_fp_params})

                # truncate the reparking flux pulse
                repark_fpbk.pulses[0]['truncation_length'] = \
                    main_fpbk.pulses[meas_pulse_idx]['truncation_length'] + \
                    repark_fpbk.pulses[0].get('buffer_length_start', 0)
                if meas_pulse_idx:
                    repark_fpbk.pulses[0]['truncation_length'] += \
                        self.get_ops_duration(pulses=main_fpbk.pulses[
                                                     :meas_pulse_idx])

                main_fpbk = self.simultaneous_blocks(
                    'flux_pulses_{qb}', [main_fpbk, repark_fpbk],
                    block_align='center')

            if sp1d_idx == 0 and sp2d_idx == 0:
                self.blocks_to_save[qb] = deepcopy(main_fpbk)


            cryo_blk = self.sequential_blocks(f'cryoscope {qb}',
                [prep_bk, pihalf_1_bk, main_fpbk, pihalf_2_bk])

            parallel_block_list += [cryo_blk]
            self.data_to_fit.update({qb: 'pe'})

        if self.sequential:
            return self.sequential_blocks(
                f'sim_rb_{sp2d_idx}_{sp1d_idx}', parallel_block_list)
        else:
            return self.simultaneous_blocks(
                f'sim_rb_{sp2d_idx}_{sp1d_idx}', parallel_block_list,
                block_align='end')

    def update_sweep_points(self, **kw):
        """
        Updates the soft sweep points in self.sweep_points with the
        continuous_truncation_lengths from each task in preprocessed_task_list,
        if it exists.
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        sp = SweepPoints()
        for task in self.preprocessed_task_list:
            if 'continuous_truncation_lengths' not in task:
                continue
            param_name = f'{task["prefix"]}truncation_length'
            unit = self.sweep_points.get_sweep_params_property(
                'unit', 1, param_names=param_name)
            label = self.sweep_points.get_sweep_params_property(
                'label', 1, param_names=param_name)
            sp.add_sweep_parameter(param_name,
                                   task['continuous_truncation_lengths'],
                                   unit, label, dimension=1)
        self.sweep_points.update(sp)

    @Timer()
    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        """
        qb_names = [task['qb'] for task in self.task_list]
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.CryoscopeAnalysis(
            qb_names=qb_names, **analysis_kwargs)

    def add_blocks_to_metadata(self):
        self.exp_metadata['flux_pulse_blocks'] = {}
        for qb, block in self.blocks_to_save.items():
            self.exp_metadata['flux_pulse_blocks'][qb] = block.build()


class FluxPulseAmplitudeSweep(ParallelLOSweepExperiment):
    """
        Flux pulse amplitude measurement used to determine the qubits energy in
        dependence of flux pulse amplitude.

        pulse sequence:
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |


            sweep_points:
            amplitude (numpy array): array of amplitudes of the flux pulse
            freq (numpy array): array of drive frequencies

        Returns: None

    """
    kw_for_task_keys = ['delay']
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'amps': dict(param_name='amplitude', unit='V',
                       label=r'flux pulse amplitude',
                       dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'Flux_amplitude'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.exp_metadata.update({'rotation_type': 'global_PCA'})
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, flux_op_code=None, delay=None, **kw):
        """
        Performs X180 pulse on top of a fluxpulse
        :param qb: (str) the name of the qubit
        :param flux_op_code: (optional str) the flux pulse op_code (default:
            FP qb)
        :param delay: (optional float): flux pulse delay (default: centered to
            center of drive pulse)
        :param kw:
        """
        if flux_op_code is None:
            flux_op_code = f'FP {qb}'
        pulse_modifs = {'attr=name,op_code=X180': f'FPS_Pi',
                        'attr=element_name,op_code=X180': 'FPS_Pi_el'}
        b = self.block_from_ops(f'ge_flux {qb}',
                                 [f'X180 {qb}', flux_op_code],
                                 pulse_modifs=pulse_modifs)
        fp = b.pulses[1]
        fp['ref_point'] = 'middle'
        if delay is None:
            delay = fp['pulse_length'] / 2
        fp['pulse_delay'] = -fp.get('buffer_length_start', 0) - delay
        fp['amplitude'] = ParametricValue('amplitude')

        b.set_end_after_all_pulses()

        self.data_to_fit.update({qb: 'pe'})

        return b

    @Timer()
    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.FluxAmplitudeSweepAnalysis(
            qb_names=self.meas_obj_names, options_dict=dict(TwoD=True),
            t_start=self.timestamp, **analysis_kwargs)

    def run_update(self, **kw):
        for qb in self.meas_obj_names:
            qb.fit_ge_freq_from_flux_pulse_amp(
                self.analysis.fit_res[f'freq_fit_{qb.name}'].best_values)


class SingleQubitGateCalibExperiment (CalibBuilder):
    """
    Base class for single qubit gate tuneup measurement classes (Rabi, Ramsey,
    T1, QScale, InPhaseAmpCalib). This is a multitasking experiment, see
    docstrings of MultiTaskingExperiment and of CalibBuilder for general
    information. Each task corresponds to a particular qubit to be tuned up
    (specified by the key "qb" in the task), i.e., multiple qubits can be
    tuned up in parallel.
    For convenience, this class accepts the "qubits" argument in which case
    task_list is not needed and will be created from qubits.

    The sequence for each task, further info, and possible parameters of
    the task are described in the docstrings of each child.

    :param task_list:  see MultiTaskingExperiment
    :param sweep_points: (SweepPoints object or list of dicts or None)
        sweep points valid for all tasks.
    :param qubits: list of qubit class instances to be tuned up.
    :param kw: keyword arguments.
        Can be used to provide keyword arguments to set_update_callback,
        sweep_n_dim, autorun, and to the parent class.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified:
        - transition_name: one of the following strings "ge", "ef", "fh"
            specifying which transmon transition to tune up.
            This can be specified in the task list and can be different for
            each task.

            Note: this is different to the transition_name in sweep_block!

    The following keys in a task are interpreted by this class:
        - qb: identifies the qubit measured in the task
        - transition_name

    """
    kw_for_task_keys = ['transition_name']

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        try:
            if task_list is None:
                if qubits is None:
                    raise ValueError('Please provide either "qubits" or '
                                     '"task_list"')
                # Create task_list from qubits
                task_list = [{'qb': qb.name} for qb in qubits]

            for task in task_list:
                if 'qb' in task and not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if 'transition_name' not in task:
                    task['transition_name'] = kw.get('transition_name', 'ge')

            if 'force_2D_sweep' not in kw:
                # Most single qubit time domain analyses do not work with
                # dummy TwoD sweeps
                kw['force_2D_sweep'] = False

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)
            if self.experiment_name is None:
                self.experiment_name = 'SingleQubiGateCalib'
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.update_experiment_name()

            self.state_order = ['g', 'e', 'f', 'h']
            # transition_name_input is added in preprocess_task;
            # see comments there.
            transition_names = [task['transition_name_input'] for task in
                                self.preprocessed_task_list]
            # states in the experiment
            states = ''.join(set([s for s in ''.join(transition_names)]))

            # append sorted states to experiment name
            states_idxs = [self.state_order.index(s) for s in states]
            states_idxs.sort()
            states_sorted = ''.join([self.state_order[i] for i in states_idxs])
            self.experiment_name += f'_{states_sorted}'

            # Used in sweep_block to prepend pulses (see docstring there). The
            # strings in transition_order will be used to specify op codes.
            # Thus "ge" is an empty string (ex: X180 qb4), while for the other
            # recognized transitions there is a prepended underscore
            # (ex: X90_ef qb3).
            self.transition_order = ('', '_ef', '_fh')

            if 'cal_states' not in kw:
                # If the user didn't specify the cal states to be prepared,
                # these will be determined based on the transition names
                # the user wants to tune up (can be different for each task).

                indices = [self.state_order.index(s) for s in states]
                # By default, add a cal state for all the transmon levels from
                # "g" to the highest level involved in the measurement.
                # Ex: Rabi on qb2-ef and qb4-fh --> cal states g,e,f,h
                cal_states = self.state_order[:max(indices)+1]
                # Recreate cal points
                self.create_cal_points(
                    n_cal_points_per_state=kw.get('n_cal_points_per_state', 1),
                    cal_states=''.join(cal_states))

            self.update_sweep_points()
            self.update_data_to_fit()
            self.define_cal_states_rotations()
            # meas_obj_sweep_points_map might be different after running some
            # of the above methods
            self.exp_metadata['meas_obj_sweep_points_map'] = \
                self.sweep_points.get_meas_obj_sweep_points_map(
                    self.meas_obj_names)

            if 'qscale' in self.experiment_name.lower() or \
                    'inphase_amp_calib' in self.experiment_name.lower() or \
                        'drive_amp_calib' in self.experiment_name.lower():
                # For these experiments the pulse sequence is not identical for
                # each all sweep points so the block function must be called
                # at each iteration in sweep_n_dim.
                if len(self.sweep_points[1]) == 0:
                    # This dummy sweep param is added in parallel_sweep of
                    # MultiTaskingExperiment, but since we do not call that
                    # method in this case, we need to add it here to the
                    # sweep points. It is only needed because these experiments
                    # are 1D sweeps but the framework assumes 2D in sweep_n_dim.
                    self.sweep_points.add_sweep_parameter('dummy_sweep_param',
                                                          [0])
                self.sequences, self.mc_points = self.sweep_n_dim(
                    self.sweep_points, body_block=None,
                    body_block_func=self.sweep_block,
                    cal_points=self.cal_points,
                    ro_qubits=self.meas_obj_names, **kw)
            else:
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.sweep_block,
                    block_align='end', **kw)

            # Force 1D sweep: most single qubit time domain analyses do not
            # work with dummy TwoD sweeps
            self.mc_points = [self.mc_points[0], []]

            self.autorun(store_preprocessed_task_list=True, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def preprocess_task(self, task, global_sweep_points, sweep_points=None,
                        **kw):
        """
        Updates the task with

        - transition_name: one of the following: "", "_ef", "_fh".
            These strings are needed for specifying op codes,
            ex: X180 qb4, X90_ef qb3
        - transition_name_input: one of the following: "ge", "ef", "fh".
            These strings are needed when updating qubit parameters,
            ex: qb.ge_amp180, qb.ef_freq
        The input parameters are documented in the docstring of this method in
        the parent class
        """
        task = super().preprocess_task(task, global_sweep_points, sweep_points,
                                       **kw)

        # Check and update transition name.
        transition_name = task.pop('transition_name', None)
        if transition_name is not None:
            # transition_name_input is one of the following: "ge", "ef", "fh".
            # These strings are needed when updating
            # qubit parameters, ex: qb.ge_amp180, qb.ef_freq
            task['transition_name_input'] = transition_name
            if '_' not in transition_name:
                transition_name = f'_{transition_name}'
            if transition_name == '_ge':
                transition_name = ''
            # transition_name will be one of the following: "", "_ef", "_fh".
            # These strings are needed for specifying op codes,
            # ex: X180 qb4, X90_ef qb3
            task['transition_name'] = transition_name
        return task

    def update_data_to_fit(self):
        """
        Update self.data_to_fit based on the transition_name_input of each task.
        """
        for task in self.preprocessed_task_list:
            qb_name = task['qb']
            transition_name = task['transition_name_input']
            self.data_to_fit[qb_name] = f'p{transition_name[-1]}'

    def define_cal_states_rotations(self):
        """
        Creates cal_states_rotations for each qubit based on the information
        in the preprocessed_task_list, and adds it to exp_metadata.
        This is of the form {qb_name: {cal_state: cal_state_order_index}}
        and will be used by the analyses.
        """
        if len(self.cal_states) < 2:
            # Data rotation in analysis needs at least 2 cal states
            return

        cal_states_rotations = {}
        for task in self.preprocessed_task_list:
            qb_name = task['qb']
            if 'cal_states_rotations' in task:
                # specified by user
                cal_states_rotations.update(task['cal_states_rotations'])
            else:
                if len(self.cal_states) > 3:
                    # Analysis can handle at most 3-state rotations
                    # If we've got more than 3 cal states, we choose a subset of
                    # 3: the states of the transition to be tuned up + the state
                    # below the lowest state of the transition.
                    # Ex: transition name = fh --> ['e', 'f', 'h']
                    transition_name = task['transition_name_input']
                    if transition_name == 'ge':
                        # Exception: there no state below g, so here we
                        # include f
                        states = ['g', 'e', 'f']
                    else:
                        indices = [self.state_order.index(s)
                                   for s in transition_name]
                        states = self.state_order[
                                 min(indices)-1:max(indices)+1]
                    rots = [(s, self.cal_states.index(s)) for s in states]
                    # sort by index of cal_state
                    rots.sort(key=lambda t: t[1])
                    cal_states_rotations[qb_name] = {t[0]: i for i, t in
                                                     enumerate(rots)}
                else:
                    # Use all the cal states for rotation
                    cal_states_rotations[qb_name] = \
                        {s: self.state_order.index(s) for s in self.cal_states}

        self.exp_metadata.update({'cal_states_rotations': cal_states_rotations})

    def update_experiment_name(self):
        # Base method for updating the experiment_name.
        # To be overloaded by children.
        pass

    def update_sweep_points(self):
        # Base method for updating the sweep_points.
        # To be overloaded by children.
        pass

    def sweep_block(self, qb, sweep_points, transition_name,
                    prepend_pulse_dicts=None, **kw):
        """
        Base method for the sweep blocks created by the children.
        This function creates a list with the following two blocks
         - prepended pulses specified by the user in prepend_pulse_dicts
         - preparation pulses needed to reach the transition to be tuned up.

        Ex: if transition to be tuned up is ef (fh), this function will prepend
            the pulses specified in prepend_pulse_dicts followed by a ge pulse
            (ge + ef pulses)

        :param qb: name of qubit
        :param sweep_points: SweepPoints instance
        :param transition_name: (string) transition to be tuned up.
            Must be in the form compatible with op codes:
                ge --> ''
                ef --> '_ef'
                fh --> '_fh'
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            block_from_pulse_dicts
        :param kw: keyword arguments
        :return: list with prepended block and prepended transition block
        """

        # create user-specified prepended pulses (pb)
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts,
                                         block_name='prepend')
        # get transition prepended pulses
        tr_prepended_pulses = self.transition_order[
                              :self.transition_order.index(transition_name)]
        return [pb, self.block_from_ops('tr_prepend',
                                        [f'X180{trn} {qb}'
                                         for trn in tr_prepended_pulses])]

    def run_analysis(self, analysis_kwargs=None, **kw):
        # Base method for running analysis.
        # To be overloaded by children.
        pass


class Rabi(SingleQubitGateCalibExperiment):
    """
    Rabi measurement for finding the amplitude of a pi-pulse that excites
    the desired transmon transition. This is a SingleQubitGateCalibExperiment,
    see docstring there for general information.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        |tr_prep_pulses**|  ---  |X180_tr_name**|  ---  |RO*|
                                  sweep amplitude

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
             (i.e. ends of X180_tr_name are aligned with start of RO pulses)

        Note: if the qubits have different drive pulse lengths, then alignment
        at the start of RO pulse means the individual drive pulses may end up
        not being applied in parallel between different qubits.
        Ex: qb2 has longer ef pulse than qb1
            qb1:             |X180| --- |X180_ef| --- |RO|
            qb2:    |X180| --- |     X180_ef    | --- |RO|

        Note: depending on which transition is tuned up in each task, the
        sequence can have different number of pulses for each qubit.

    See docstring of parent class for the first 3 parameters.
    :param amps: (numpy array) amplitude sweep points for X180_tr_name.
        This parameter can be used together with "qubits" for convenience to
        avoid having to specify a task_list.
        If not None, amps will be used to create the first dimension of
        sweep points, which will be identical for all tasks.

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - n

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - n
        - amps
    """

    kw_for_sweep_points = {
        'amps': dict(param_name='amplitude', unit='V',
                     label='Pulse Amplitude', dimension=0)
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 amps=None, **kw):
        try:
            self.kw_for_task_keys += ['n']
            if 'n' not in kw:
                # add default n to kw before passing to init of parent
                kw['n'] = 1
            self.experiment_name = 'Rabi'
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             amps=amps, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_experiment_name(self):
        """
        Updates self.experiment_name with the number of Rabi pulses n.
        """
        n_list = [task['n'] for task in self.preprocessed_task_list]
        if any([n > 1 for n in n_list]):
            # Rabi measurement with more than one pulse
            self.experiment_name += f'-n'
            if len(np.unique(n_list)) == 1:
                # all tasks have the same n; add the value of n to the
                # experiment name
                self.experiment_name += f'{n_list[0]}'

    def sweep_block(self, qb, sweep_points, transition_name, **kw):
        """
        This function creates the blocks for a single Rabi measurement task,
        see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: transmon transition to be tuned up. Can be
            "", "_ef", "_fh". See the docstring of parent method.
        :param kw: keyword arguments
            n: (int, default: 1) number of Rabi pulses (X180_tr_name in the
                pulse sequence). Amplitude of all these pulses will be swept.
        """

        # create prepended pulses
        prepend_blocks = super().sweep_block(qb, sweep_points, transition_name,
                                             **kw)
        n = kw.get('n', 1)
        # add rabi block of n x X180_tr_name pulses
        rabi_block = self.block_from_ops(f'rabi_pulses_{qb}',
                                         n*[f'X180{transition_name} {qb}'])
        # create ParametricValues from param_name in sweep_points
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in rabi_block.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        return self.sequential_blocks(f'rabi_{qb}',
                                      prepend_blocks + [rabi_block])

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis class
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.RabiAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the pi-pulse amplitude (tr_name_amp180) of the qubit in
        each task with the value extracted by the analysis.
        :param kw: keyword arguments
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            amp180 = self.analysis.proc_data_dict['analysis_params_dict'][
                qubit.name]['piPulse']
            qubit.set(f'{task["transition_name_input"]}_amp180', amp180)
            qubit.set(f'{task["transition_name_input"]}_amp90_scale', 0.5)


class Ramsey(SingleQubitGateCalibExperiment):
    """
    Class for running a Ramsey or an Echo measurement.
    This is a SingleQubitGateCalibExperiment, see docstring there
    for general information.

    Ramsey measurement for finding the frequency and associated averaged
    dephasing time (T2*) of a transmon transition.
    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        |tr_prep_pulses**| -- |X90_tr_name**| - ... - |X90_tr_name**| -- |RO*|
                                               sweep      sweep
                                             delay tau    phase

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
             (i.e. ends of 2nd X90_tr_name are aligned with start of RO pulses)

    Echo measurement for finding the dephasing time (T2) associated with a
    transmon transition.
    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        |tpp**| -- |X90_tn**| - ... - |X180_tn**| - ... - |X90_tn**| -- |RO*|
                               sweep               sweep      sweep
                            delay tau/2         delay tau/2   phase

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
             (i.e. ends of 2nd X90_tn are aligned with start of RO pulses)

    Note: if the qubits have different drive pulse lengths, then alignment
    at the start of RO pulse means the individual drive pulses may end up
    not being applied in parallel between different qubits.
    See Rabi class docstring for an example of this situation.

    Note: depending on which transition is tuned up in each task, the
    sequence can have different number of pulses for each qubit.

    See docstring of parent class for the first 3 parameters.
    :param delays: (numpy array) sweep points for the delays of the second
        X90_tr_name pulse.
        This parameter can be used together with qubits for convenience to
        avoid having to specify a task_list.
        If not None, delays will be used to create the first dimension of
        sweep points, which will be identical for all tasks.
    :param echo: (bool, default: False) whether to do an Echo (True) or a
        Ramsey (False) measurement.
    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - artificial_detuning

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - artificial_detuning
        - delays
    """

    kw_for_sweep_points = {
        'delays': dict(param_name='pulse_delay', unit='s',
                       label=r'Second $\pi$-half pulse delay', dimension=0)
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, echo=False, **kw):
        try:
            self.kw_for_task_keys += ['artificial_detuning']
            if 'artificial_detuning' not in kw:
                # add default artificial_detuning to kw before passing to
                # init of parent
                kw['artificial_detuning'] = 0
            self.echo = echo
            if not hasattr(self, 'experiment_name'):
                self.experiment_name = 'Echo' if self.echo else 'Ramsey'
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             delays=delays, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def preprocess_task(self, task, global_sweep_points, sweep_points=None,
                        **kw):
        """
        Updates the task with the value of the first pulse delay sweep point.
        which will be stored under the key "first_delay_point" and will be used
        in sweep_block.
        See also the docstring of this method in the parent classes.
        """
        task = super().preprocess_task(task, global_sweep_points, sweep_points,
                                       **kw)
        sweep_points = task['sweep_points']
        task['first_delay_point'] = sweep_points.get_sweep_params_property(
            'values', 0, 'pulse_delay')[0]

        return task

    def sweep_block(self, qb, sweep_points, transition_name, **kw):
        """
        This function creates the blocks for a single Ramsey/Echo measurement
        task, see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: transmon transition to be tuned up. Can be
            "", "_ef", "_fh". See the docstring of parent method.
        :param kw: keyword arguments
            artificial_detuning: (float, default: 0) detuning of second pi-half
            pulse (X90_tr_name in the pulse sequence). Will be used to calculate
            phase of the 2nd X90_tr_name at each delay.
            first_delay_point: (float) see docstring of update_preproc_tasks
        """

        # create prepended pulses
        prepend_blocks = super().sweep_block(qb, sweep_points, transition_name,
                                             **kw)

        # add ramsey block of 2x X90_tr_name pulses
        pulse_modifs = {1: {'ref_point': 'start'}}
        ramsey_block = self.block_from_ops(f'ramsey_pulses_{qb}',
                                           [f'X90{transition_name} {qb}',
                                            f'X90{transition_name} {qb}'],
                                           pulse_modifs=pulse_modifs)

        first_delay_point = kw['first_delay_point']
        art_det = kw['artificial_detuning']
        # create ParametricValues for the 2nd X90_tr_name pulse from
        # param_name in sweep_points
        for param_name in sweep_points.get_sweep_dimension(0):
            ramsey_block.pulses[-1][param_name] = ParametricValue(param_name)
            if param_name == 'pulse_delay':
                # PrametricValue for the phase to be calculated from each delay
                ramsey_block.pulses[-1]['phase'] = ParametricValue(
                    'pulse_delay', func=lambda x, o=first_delay_point:
                    ((x-o)*art_det*360) % 360)

        if self.echo:
            # add echo block: pi-pulse halfway between the two X90_tr_name
            # pulses
            echo_block = self.block_from_ops(f'echo_pulse_{qb}',
                                             [f'X180{transition_name} {qb}'])
            ramsey_block = self.simultaneous_blocks(f'main_{qb}',
                                                    [ramsey_block, echo_block],
                                                    block_align='center')

        return self.sequential_blocks(f'ramsey_{qb}',
                                      prepend_blocks + [ramsey_block])

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}
        options_dict = analysis_kwargs.pop('options_dict', {})
        options_dict.update(dict(
            fit_gaussian_decay=kw.pop('fit_gaussian_decay', True),
            artificial_detuning=kw.pop('artificial_detuning', None)))
        self.analysis = tda.EchoAnalysis if self.echo else tda.RamseyAnalysis
        self.analysis = self.analysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            options_dict=options_dict, **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the following parameters of the qubit in each task with the
        values extracted by the analysis:

            For Ramsey measurement:
                - transition frequency: tr_name_freq
                - averaged dephasing time: T2_star_tr_name
            For Echo measurement:
                - dephasing time: T2_tr_name

        :param kw: keyword arguments
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            if self.echo:
                T2_echo = self.analysis.proc_data_dict[
                    'analysis_params_dict'][qubit.name]['T2_echo']
                qubit.set(f'T2{task["transition_name"]}', T2_echo)
            else:
                qb_freq = self.analysis.proc_data_dict[
                    'analysis_params_dict'][qubit.name][
                    'exp_decay']['new_qb_freq']
                T2_star = self.analysis.proc_data_dict[
                    'analysis_params_dict'][qubit.name][
                    'exp_decay']['T2_star']
                qubit.set(f'{task["transition_name_input"]}_freq', qb_freq)
                qubit.set(f'T2_star{task["transition_name"]}', T2_star)


class ReparkingRamsey(Ramsey):
    """
    Class for reparking qubits by doing a set of Ramsey measurements at
    different bias voltages. This is a Ramsey experiment, see docstring there
    for pulse sequence and general information.

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

        The following keyword argument, if provided, will be used to create the
        sweep points, which will be identical for all tasks.
        - delays: (numpy array) first dimension sweep points for the delays of
            the second X90_tr_name pulse, see pulse sequence in Ramsey class.
        - dc_voltages: (numpy array) second dimension sweep points for the dc
            voltage values
        - dc_voltage_offsets: (numpy array) second dimension sweep points for
            the voltage offsets from the qubit's current parking voltage value
        Either dc_voltages or dc_voltage_offsets must be specified either as
        kw or in the task_list.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - fluxline: qcodes parameter to adjust the DC flux offset of the qubit.
        - artificial_detuning (see docstring of parent class)

        These kw parameters can be used together with "qubits" (see
        SingleQubitGateCalibExperiment parent class) for convenience to avoid
        having to specify a task_list.
        If a task_list is not specified, all qubits will use the same values for
        the parameters above.

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - fluxline
        - dc_voltages
        - dc_voltage_offsets
    """

    kw_for_sweep_points = {
        'delays': dict(param_name='pulse_delay', unit='s',
                       label=r'Second $\pi$-half pulse delay', dimension=0),
        'dc_voltages': dict(param_name='dc_voltages', unit='V',
                            label=r'DC voltage', dimension=1),
        'dc_voltage_offsets': dict(param_name='dc_voltage_offsets', unit='V',
                                   label=r'DC voltage offset', dimension=1),
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, dc_voltages=None, dc_voltage_offsets=None,  **kw):

        try:
            self.kw_for_task_keys += ['fluxline']
            if 'fluxline' not in kw:
                # add default value for fluxline to kw before passing to
                # init of parent
                kw['fluxline'] = None
            self.experiment_name = 'ReparkingRamsey'
            # the parent class enforces a 1D sweep, so here we must explicitly
            # force it to be a 2D sweep
            force_2D_sweep = True
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             delays=delays,
                             dc_voltages=dc_voltages,
                             dc_voltage_offsets=dc_voltage_offsets,
                             force_2D_sweep=force_2D_sweep, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_sweep_points(self):
        """
        Updates the sweep_points of each task with the dc_voltages sweep values
        calculated from dc_voltage_offsets (if dc_voltages do not already
        exist). They are needed in the analysis.
        """
        for task in self.preprocessed_task_list:
            swpts = task['sweep_points']
            if swpts.find_parameter('dc_voltage_offsets') is not None:
                if swpts.find_parameter('dc_voltages') is not None:
                    # Do not overwrite the values provided by the user
                    log.warning(f'Both "dc_voltages" and "dc_voltage_offsets" '
                                f'were provided for {task["qb"]}. The latter '
                                f'will be ignored.')
                    continue

                fluxline = task['fluxline']
                values_to_set = np.array(swpts.get_sweep_params_property(
                    'values',
                    dimension=swpts.find_parameter('dc_voltage_offsets'),
                    param_names='dc_voltage_offsets')) + fluxline()
                # update sweep points
                par_name = f'{task["prefix"]}dc_voltages'
                self.sweep_points.add_sweep_parameter(par_name, values_to_set,
                                                      'V', 'DC voltage', 1)

    def run_measurement(self, **kw):
        """
        Configures additional sweep functions and temporary values for the
        for the current dc voltage values, before calling the method of the
        base class.
        """
        sweep_functions = []
        temp_vals= []
        sweep_param_name = 'Parking voltage'
        nr_volt_points = self.sweep_points.length(1)
        self.exp_metadata['current_voltages'] = {}
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]

            fluxline = task['fluxline']
            # add current voltage value to temporary values to reset it back to
            # this value at the end of the measurement
            temp_vals.append((fluxline, fluxline()))
            self.exp_metadata['current_voltages'][qb.name] = fluxline()

            # get the dc voltages sweep values
            swpts = task['sweep_points']
            if swpts.find_parameter('dc_voltages') is not None:
                # absolute dc_voltage were given
                values_to_set = swpts.get_sweep_params_property(
                    'values',
                    dimension=swpts.find_parameter('dc_voltages'),
                    param_names='dc_voltages')
            elif swpts.find_parameter('dc_voltage_offsets') is not None:
                # relative dc_voltages were given
                values_to_set = np.array(swpts.get_sweep_params_property(
                    'values',
                    dimension=swpts.find_parameter('dc_voltage_offsets'),
                    param_names='dc_voltage_offsets')) + fluxline()
            else:
                # one or the other must exist
                raise KeyError(f'Please specify either dc_voltages or '
                               f'dc_voltage_offsets for {qb.name}.')

            if len(values_to_set) != nr_volt_points:
                raise ValueError('All tasks must have the same number of '
                                 'voltage sweep points.')

            # create an Indexed_Sweep function for each task
            sweep_functions += [swf.Indexed_Sweep(
                task['fluxline'], values=values_to_set,
                name=f'DC Offset {qb.name}',
                parameter_name=f'{sweep_param_name} {qb.name}', unit='V')]

        self.sweep_functions = [
            self.sweep_functions[0], swf.multi_sweep_function(
                sweep_functions, name=sweep_param_name,
                parameter_name=sweep_param_name)]
        self.mc_points[1] = np.arange(nr_volt_points)

        with temporary_value(*temp_vals):
            super().run_measurement(**kw)

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}
        options_dict = analysis_kwargs.pop('options_dict', {})
        options_dict.update(dict(
            fit_gaussian_decay=kw.pop('fit_gaussian_decay', True),
            artificial_detuning=kw.pop('artificial_detuning', None)))
        self.analysis = tda.ReparkingRamseyAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            options_dict=options_dict, **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the following parameters for the qubit in each task with the
        values extracted by the analysis:
            - transition frequency: tr_name_freq
            - fluxline voltage
        :param kw: keyword arguments
        """

        for task in self.preprocessed_task_list:
            qubit = self.get_qubits(task['qb'])[0][0]
            fluxline = task['fluxline']

            apd = self.analysis.proc_data_dict['analysis_params_dict']
            # set new qubit frequency
            qubit.set(f'{task["transition_name_input"]}_freq',
                      apd['reparking_params'][qubit.name]['new_ss_vals']['ss_freq'])
            # set new voltage
            fluxline(apd['reparking_params'][qubit.name]['new_ss_vals']['ss_volt'])


class T1(SingleQubitGateCalibExperiment):
    """
    T1 measurement for finding the lifetime (T1) associated with a transmon
    transition. This is a SingleQubitGateCalibExperiment,
    see docstring there for general information.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        |tr_prep_pulses**|  --  |X180_tr_name**|  --  ... --  |RO*|
                                                  sweep delay

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
             (i.e. ends of X180_tr_name are aligned)

        Note: if the qubits have different drive pulse lengths, then alignment
        at the end of X180_tr_name means the individual drive pulses may end up
        not being applied in parallel between different qubits.
        See Rabi class docstring for an example of this situation.

        Note: depending on which transition is tuned up in each task, the
        sequence can have different number of pulses for each qubit.

    See docstring of parent class for the first 3 parameters.
    :param delays: (numpy array) sweep points for the delays between
        X180_tr_name and the RO pulse.
        This parameter can be used together with qubits for convenience to
        avoid having to specify a task_list.
        If not None, delays will be used to create the first dimension of
        sweep points, which will be identical for all tasks.
    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - delays
    """

    kw_for_sweep_points = {
        'delays': dict(param_name='pulse_delay', unit='s',
                       label=r'Readout pulse delay', dimension=0),
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, **kw):
        try:
            self.experiment_name = 'T1'
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             delays=delays, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, transition_name, **kw):
        """
        This function creates the blocks for a single T1 measurement task,
        see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: transmon transition to be tuned up. Can be
            "", "_ef", "_fh". See the docstring of parent method.
        :param kw: keyword arguments.
        """

        # create prepended pulses
        prepend_blocks = super().sweep_block(qb, sweep_points, transition_name,
                                             **kw)
        # add t1 block consisting of one X180_tr_name pulse
        t1_block = self.block_from_ops(f'pi_pulse_{qb}',
                                       [f'X180{transition_name} {qb}'])
        # Create ParametricValue for the block end virtual pulse.
        # This effectively introduces a delay between the X180_tr_pulse and
        # the RO pulse which will be added in sweep_n_dim
        t1_block.block_end.update({'ref_point': 'end',
                                   'pulse_delay': ParametricValue('pulse_delay')
                                   })

        return self.sequential_blocks(f't1_{qb}', prepend_blocks + [t1_block])

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.T1Analysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the T1_tr_name of the qubit in each task with the value
        extracted by the analysis.
        :param kw: keyword arguments
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            T1 = self.analysis.proc_data_dict['analysis_params_dict'][
                qubit.name]['T1']
            qubit.set(f'T1{task["transition_name"]}', T1)


class QScale(SingleQubitGateCalibExperiment):
    """
    QScale measurement for finding the motzoi parameter for driving a transmon
    transition without phase errors.
    See the papers:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.83.012308

    This is a SingleQubitGateCalibExperiment, see docstring there
    for general information.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):
        For each qscale in sweep_points, the following 2 sequences are applied:

        |tr_prep_pulses**|  --  |X90_tr_name**| - |X180_tr_name** |  --  |RO*|
        |tr_prep_pulses**|  --  |X90_tr_name**| - |Y180_tr_name** |  --  |RO*|
        |tr_prep_pulses**|  --  |X90_tr_name**| - |mY180_tr_name**|  --  |RO*|
                                 sweep qscale       sweep qscale

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
            (i.e. ends of the last drive pulses aligned with start of RO pulses)

        Note: if the qubits have different drive pulse lengths, then alignment
        at the start of RO pulse means the individual drive pulses may end up
        not being applied in parallel between different qubits.
        See Rabi class docstring for an example of this situation.

        Note: depending on which transition is tuned up in each task, the
        sequence can have different number of pulses for each qubit.

    See docstring of parent class for the first 3 parameters.
    :param qscales: (numpy array) sweep points for the motzoi param of the pairs
        of pulses in the pulse sequence above.
        This parameter can be used together with "qubits" for convenience to
        avoid having to specify a task_list.
        If not None, qscales will be used to create the first dimension of
        sweep points, which will be identical for all tasks.
    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - qscales
    """

    kw_for_sweep_points = {
        'qscales': dict(param_name='motzoi', unit='V',
                        label='Pulse Amplitude', dimension=0,
                        values_func=lambda q: np.repeat(q, 3))
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 qscales=None, **kw):
        try:
            self.experiment_name = 'Qscale'
            # the 3 pairs of pulses to be applied for each sweep point
            self.qscale_base_ops = [['X90', 'X180'], ['X90', 'Y180'],
                                    ['X90', 'mY180']]
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             qscales=qscales, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_sweep_points(self):
        """
        Checks if the qscale sweep points are repeated 3 times (there are 3
        pairs of pulses in qscale_base_ops). Updates the self.sweep_points and
        the sweep points in each task of preprocessed_task_list with the
        repeated qscale sweep points.
        """

        # update self.sweep_points
        swpts = deepcopy(self.sweep_points)
        swp_dim0 = swpts.get_sweep_dimension(0)

        par_names = []
        vals = []
        for par in swp_dim0:
            if 'motzoi' not in par:
                continue

            values = swpts.get_sweep_params_property('values', param_names=par)
            if np.unique(values[:3]).size > 1:
                # the qscale sweep points are not repeated 3 times (for each
                # pair of qscale_base_ops)
                par_names += [par]
                vals += [np.repeat(values, 3)]

        self.sweep_points.update_property(par_names, values=vals)

        # update sweep points in preprocessed_task_list
        for task in self.preprocessed_task_list:
            swpts = task['sweep_points']
            values = swpts.get_sweep_params_property(
                'values', param_names='motzoi')
            if np.unique(values[:3]).size > 1:
                swpts.update_property(['motzoi'], values=[np.repeat(values, 3)])

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        This function creates the blocks for each QScale measurement task,
        see the pulse sequence in the class docstring.
        :param sp1d_idx: sweep point index in the first sweep dimension
        :param sp2d_idx: sweep point index in the second sweep dimension
        :param kw: keyword arguments to be provided in each task
            qb: qubit name
            sweep_points: SweepPoints instance
            transition_name: transmon transition to be tuned up. Can be
                "", "_ef", "_fh". See the docstring of parent method.
        """

        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            transition_name = task['transition_name']
            sweep_points = task['sweep_points']
            qb = task['qb']

            # create prepended pulses
            prepend_blocks = super().sweep_block(**task)
            # add qscale block consisting of the correct set of 2 pulses
            # from qscale_base_ops (depending on sp1d_idx)
            qscale_pulses_block = self.block_from_ops(
                f'qscale_pulses_{qb}', [f'{p}{transition_name} {qb}' for p in
                                        self.qscale_base_ops[sp1d_idx % 3]])
            # set the motzoi parameter of the pulses in the qscale block
            for p in qscale_pulses_block.pulses:
                p['motzoi'] = sweep_points.get_sweep_params_property(
                    'values', 0, 'motzoi')[sp1d_idx]
            # gather the blocks for each task
            parallel_block_list += [self.sequential_blocks(
                f'qscale_{qb}', prepend_blocks + [qscale_pulses_block])]

        return self.simultaneous_blocks(f'qscale_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.QScaleAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the tr_name_motzoi parameter of the qubit in each task with
        the value extracted by the analysis.
        :param kw: keyword arguments
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            qscale = self.analysis.proc_data_dict['analysis_params_dict'][
                qubit.name]['qscale']
            qubit.set(f'{task["transition_name_input"]}_motzoi', qscale)


class InPhaseAmpCalib(SingleQubitGateCalibExperiment):
    """
    In-phase calibration measurement for finding small miscalibrations in the
    pi-pulse amplitude associated with a transmon transition.
    This is a SingleQubitGateCalibExperiment, see docstring there
    for general information.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        |tr_prep_pulses**| -- |X90_tr_name**| -- Nx|X180_tr_name**| -- |RO*|
                                                     sweep N

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
             (i.e. ends of the last X180_tr_name are aligned with start of
             RO pulses)

        Note: if the qubits have different drive pulse lengths, then alignment
        at the start of RO pulse means the individual drive pulses may end up
        not being applied in parallel between different qubits.
        See Rabi class docstring for an example of this situation.

        Note: depending on which transition is tuned up in each task, the
        sequence can have different number of pulses for each qubit.

    See docstring of parent class for the first 3 parameters.
    :param n_pulses: (int) max number of X180_tr_name in the experiment. The
        class will then n_pulses/2 segments with even number of pi-pulses
        (i.e. np.arange(nr_p + 1)[::2]).
        This parameter can be used together with "qubits" for convenience to
        avoid having to specify a task_list.
        If not None, n_pulses will be used to create the first dimension of
        sweep points, which will be identical for all tasks.
    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

        Moreover, the following keyword arguments are understood:
            use_x90_pulses: (bool, default: False) whether apply a train of
                pi-pulses or a train of pairs of pi/2-pulses (allows to
                calibrate the pi/2-pulses)

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - n_pulses
    """

    kw_for_sweep_points = {
        'n_pulses': dict(param_name='n_pulses', unit='',
                         label='Nr. $\\pi$-pulses, $N$', dimension=0,
                         values_func=lambda nr_p:
                         np.arange(nr_p + 1)[::2])
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 n_pulses=None, **kw):
        try:
            self.use_x90_pulses = kw.get('use_x90_pulses', False)
            self.experiment_name = f'Inphase_amp_calib_{n_pulses}'
            if self.use_x90_pulses:
                self.experiment_name += '_x90_pulses'
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             n_pulses=n_pulses, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        This function creates the blocks for each InPhaseCalib measurement task,
        see the pulse sequence in the class docstring.
        :param sp1d_idx: sweep point index in the first sweep dimension
        :param sp2d_idx: sweep point index in the second sweep dimension
        :param kw: keyword arguments to be provided in each task
            qb: qubit name
            sweep_points: SweepPoints instance
            transition_name: transmon transition to be tuned up. Can be
                "", "_ef", "_fh". See the docstring of parent method.
        """
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            transition_name = task['transition_name']
            sweep_points = task['sweep_points']
            qb = task['qb']

            # create prepended pulses
            prepend_blocks = super().sweep_block(**task)
            # add inphase_calib block consisting of the number of X180_tr_name
            # pulses indicated by sp1d_idx
            n_pulses = sweep_points.get_sweep_params_property(
                'values', 0, 'n_pulses')[sp1d_idx]
            pulse_list = [f'X90{transition_name} {qb}'] + \
                         (2*n_pulses * [f'X90{transition_name} {qb}'] if
                          self.use_x90_pulses else
                          n_pulses*[f'X180{transition_name} {qb}'])
            inphase_calib_block = self.block_from_ops(
                f'pulses_{qb}', pulse_list)
            # gather the blocks for each task
            parallel_block_list += [self.sequential_blocks(
                f'inphase_calib_{qb}', prepend_blocks + [inphase_calib_block])]

        return self.simultaneous_blocks(f'inphase_calib_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword argument for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.InPhaseAmpCalibAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the pi-pulse amplitude (tr_name_amp180) of the qubit in
        each task with the value extracted by the analysis.
        :param kw: keyword arguments
        """
        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']]
            qubit.set(f'{task["transition_name_input"]}_amp180',
                      self.analysis.proc_data_dict['analysis_params_dict'][
                          qubit.name]['corrected_amp'])


class DriveAmpCalib(SingleQubitGateCalibExperiment):
    """
    Calibration measurement for the qubit drive amplitude that makes use of
    error amplification from application of N subsequent pulses.

    This is a multitasking experiment, see docstrings of MultiTaskingExperiment
    and of CalibBuilder for general information. Each task corresponds to one
    qubit specified by the key 'qb' (either name of QuDev_transmon instance)
    i.e., multiple qubit can be measured in parallel.

    This experiment can be run in two modes:

    1. The qubit is brought into superposition with a pi-half pulse, after which
    n_pulses pairs of pulses are applied with amplitudes of the first and second
    pulse in each pair chosen such that, ideally, the pair implements a rotation
    of pi. The correct amplitude for the pulse whose amplitude is not fixed by
    fixed_scaling can be found by sweeping around the expected correct
    amplitude (see info about sweep points below).

    This mode is enabled by specifying the input parameter fixed_scaling as
    a fraction of a pi rotation. This number will be used to scale the amplitude
    of the second pulse in the pair, while that of the first pulse will be
    scaled by amp_scalings given in sweep points (see below).

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb:   |X90|  ---  [ |Rx(phi)| --- |Rx(pi - phi)| ] x n_pulses  ---  |RO|

    2. The qubit is brought into superposition with a pi-half pulse, after which
    n_pulses * nr_pulses_pi pulses are applied with amplitudes scaled by
    amp_scaling.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb: |X90| --- [ |Rx(pi/nr_pulses_pi)| ] x n_pulses x n_pulses_pi -- |RO|


    For either mode, expected sweep points, either global or per task
     - n_pulses in dimension 0: number of pairs of pulses after the initial
        pi-half pulse.
     - amp_scalings in dimension 1: dimensionless fractions of a pi rotation
        around the x-axis of the Bloch sphere.


    Idea behind this calibration measurement:
     If the pulses are perfectly calibrated, the qubit will remain in the
     superposition state with 50% excited state probability independent of
     n_pulses and amp_scalings. Miscalibrations will be signaled by an
     oscillation of the excited state probability around 50% with increasing N.
     By minimizing the standard deviation of this oscillation away from the 50%
     line as a function of the amp_scalings we can calibrate any drive amplitude
     with high precision.

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun,
        and to the parent classes.

        The following keyword arguments will be copied as entries in
        sweep_points:
        - n_pulses: int
        - amp_scalings: list or array

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - n_pulses_pi (see sweep_block docstring)

        Moreover, the following keyword arguments are understood:
        fixed_scaling: (float, default: None) toggles between the two ways
            of running this experiment explained above by setting the amplitude
            scaling of one of the pulses in the pair.
    """
    kw_for_sweep_points = {
        'n_pulses': dict(param_name='n_pulses', unit='',
                         label='Nr. $\\pi$-pulses, $N$', dimension=0,
                         values_func=lambda nr_p:
                         np.arange(1, nr_p + 1, 2)),
        'amp_scalings': dict(param_name='amp_scalings', unit='',
                         label='Amplitude Scaling, $r$', dimension=1),
    }

    kw_for_task_keys = ['n_pulses_pi']

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 n_pulses=None, amp_scalings=None, n_pulses_pi=1, **kw):
        try:
            # Define experiment_name and call the parent class __init__
            self.experiment_name = f'Drive_amp_calib'
            if n_pulses is not None:
                self.experiment_name += f'_{n_pulses}pipulses'
            self.fixed_scaling = kw.get('fixed_scaling', None)
            if self.fixed_scaling is None:
                self.experiment_name += f'_{n_pulses_pi}xpi_over_{n_pulses_pi}'
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             n_pulses=n_pulses,
                             n_pulses_pi=n_pulses_pi,
                             amp_scalings=amp_scalings, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, sweep_points, **kw):
        """
        This function creates the block at the current iteration in sweep
        points, specified by sp1d_idx and sp2d_idx.
        :param sp1d_idx: current index in the first sweep dimension
        :param sp2d_idx: current index in the second sweep dimension
        :param sweep_points: SweepPoints object
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)

        Assumes self.preprocessed_task_list has been defined and that it
        contains the entries specified by the following keys:
         - 'qb': qubit name
         - 'sweep_points': SweepPoints instance
         - 'transition_name' (see docstring of parent class)
         - 'n_pulses_pi': int specifying the number of pulses that
         will implement a pi rotation.

        If self.fixed_scaling is None, n_pulses * n_pulses_pi identical
        pulses will be added after the initial pi-half pulse and their
        amplitudes will be scaled by amp_scaling.

        If self.fixed_scaling is specified, n_pulses pairs of X180 pulses will
        be added after the initial pi-half pulse, where the amplitude of the
        first pulse is scaled by amp_scaling, and the amplitude of the second
        scaled by self.fixed_scaling.

        :return: instance of Block created with simultaneous_blocks from a list
            of blocks corresponding to the tasks in preprocessed_task_list.
        """

        # Define list to gather the final blocks for each task
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            transition_name = task['transition_name']
            sweep_points = task['sweep_points']
            qb = task['qb']

            # Get the block to prepend from the parent class
            # (see docstring there)
            prepend_block = super().sweep_block(qb, sweep_points,
                                                transition_name)

            n_pulses = sweep_points.get_sweep_params_property(
                'values', 0, 'n_pulses')[sp1d_idx]
            amp_scaling = sweep_points.get_sweep_params_property(
                'values', 1)[sp2d_idx]

            if self.fixed_scaling is not None:
                # Apply pairs of X180 pulses, where the amplitude of the first
                # pulse is scaled by amp_scaling, and the amplitude of the
                # second scaled by self.fixed_scaling. (specified by sp2d_idx).

                # Create the pulse list for n_pulses specified by sp1d_idx
                pulse_list = [f'X90{transition_name} {qb}'] + \
                             (n_pulses * [f'X180{transition_name} {qb}',
                                          f'X180{transition_name} {qb}'])
                # Create a block from this list of pulses
                drive_calib_block = self.block_from_ops(f'pulses_{qb}',
                                                        pulse_list)
                # Scale amp of all even pulses after the first by amp_scaling
                for pulse_dict in drive_calib_block.pulses[1:][0::2]:
                    pulse_dict['amplitude'] *= amp_scaling
                # Scale amp of all odd pulses after the first by fixed_scaling
                for pulse_dict in drive_calib_block.pulses[1:][1::2]:
                    pulse_dict['amplitude'] *= self.fixed_scaling
            else:
                # Apply n_pulses * n_pulses_pi X180 pulses after the initial
                # pi-half pulse, and scale the amplitude of these X180 pulses
                # by amp_scaling (specified by sp2d_idx).

                n_pulses_pi = task['n_pulses_pi']
                # Create the pulse list for n_pulses specified by sp1d_idx
                pulse_list = [f'X90{transition_name} {qb}'] + \
                             (n_pulses * n_pulses_pi *
                              [f'X180{transition_name} {qb}'])
                # Create a block from this list of pulses
                drive_calib_block = self.block_from_ops(f'pulses_{qb}',
                                                        pulse_list)
                # Divide amp of all pulses after the first by amp_scaling
                for pulse_dict in drive_calib_block.pulses[1:]:
                    pulse_dict['amplitude'] *= amp_scaling

            # Append the final block for this task to parallel_block_list
            parallel_block_list += [self.sequential_blocks(
                f'drive_calib_{qb}', prepend_block + [drive_calib_block])]

        return self.simultaneous_blocks(f'inphase_calib_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')


class RabiFrequencySweep(ParallelLOSweepExperiment):
    """
    Performs a series of ge Rabi experiments for multiple drive frequencies.
    This is a ParallelLOSweepExperiment, see docstrings of
    ParallelLOSweepExperiment, MultiTaskingExperiment and CalibBuilder
    for general information.

           |X180|          ---         |RO|
      sweep amp & freq

    Important notes (see docstring of ParallelLOSweepExperiment for details):
    - Each task corresponds to a qubit (specified by the key qb in the
      task), i.e., multiple qubits can be characterized in parallel.
    - Some options of ParallelLOSweepExperiment have not yet been
      exhaustively tested for parallel measurements.
    - The key fluxline in a task can be set to a qcodes parameter to adjust
      the DC flux offset of the qubit during the sweep.

    :param kw:
        Keyword arguments are passed on to the super class and to autorun.

        The following kwargs are automatically converted to sweep points:
        - amps: drive pulse amplitude (sweep dimension 0)
        - freqs: drive frequency (sweep dimension 1)
    """
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'amps': dict(param_name='amplitude', unit='V',
                     label=r'drive pulse amplitude',
                     dimension=0),
    }

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.experiment_name = 'RabiFrequencySweep'
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, **kw):
        """
        This function creates the block for a single RabiFrequencySweep
        measurement task, see the pulse sequence in the class docstring.
        :param qb: (str) the qubit name
        :param kw: currently ignored
        """
        b = self.block_from_ops(f'ge {qb}', [f'X180 {qb}'])
        b.pulses[0]['amplitude'] = ParametricValue('amplitude')
        self.data_to_fit.update({qb: 'pe'})
        return b


class ActiveReset(CalibBuilder):
    def __init__(self, task_list=None, recalibrate_ro=False,
                 prep_states=('g', 'e'), n_shots=10000,
                 reset_reps=10, set_thresholds=True,
                 **kw):
        """
        Characterize active reset with the following sequence:

        |prep-pulses|--|prep_state_i|--(|RO|--|reset_pulse|) x reset_reps --|RO|

        -Prep-pulses are preselection/active_reset pulses, with parameters defined
        in qb.preparation_params().
        - Prep_state_i is "g", "e", "f" as provided by prep_states.
        - the following readout and reset pulses use the "ro_separation" and
        "post_ro_wait" of the qb.preparation_params() but the number of pulses is set
        by reset_reps, such that we can both apply active reset and characterize the
        reset with different number of pulses.
        Args:
            task_list (list): list of task for the reset. Needs the keys
            recalibrate_ro (bool): whether or not to recalibrate the readout
                before characterizing the active reset.
            prep_states (iterable): list of states on which the reset will be
                characterized
            reset_reps (int): number of readouts used to characterize the reset.
                Note that this parameter does NOT correspond to 'reset_reps' in
                qb.preparation_params() (the latter is used for reset pe
            set_thresholds (bool): whether or not to set the thresholds from
                qb.acq_classifier_params() to the corresponding UHF channel
            n_shots (int): number of single shot measurements
            **kw:
        """

        self.experiment_name = kw.get('experiment_name',
                                      f"active_reset_{prep_states}")

        # build default task in which all qubits are measured
        if task_list is None:
            assert kw.get('qubits', None) is not None, \
                    "qubits must be passed to create default task_list " \
                    "if task_list=None."
            task_list = [{"qubit": [qb.name]} for qb in kw.get('qubits')]

        # configure detector function parameters
        if kw.get("classified", False):
            kw['df_kwargs'] = kw.get('df_kwargs', {})
            if 'det_get_values_kws' not in kw['df_kwargs']:
                kw['df_kwargs'] = {'det_get_values_kws':
                                    {'classified': True,
                                     'correlated': False,
                                     'thresholded': False,
                                     'averaged': False}}
            else:
                # ensure still single shot
                kw['df_kwargs']['det_get_values_kws'].update({'averaged':False})
        else:
            kw['df_name'] = kw.get('df_name', "int_log_det")



        self.set_thresholds = set_thresholds
        self.recalibrate_ro = recalibrate_ro
        # force resetting of thresholds if recalibrating readout
        if self.recalibrate_ro and not self.set_thresholds:
            log.warning(f"recalibrate_ro=True but set_threshold=False,"
                        f" the latest thresholds from the recalibration"
                        f" won't be uploaded to the UHF.")
        self.prep_states = prep_states
        self.reset_reps = reset_reps
        self.n_shots = n_shots

        # init parent
        super().__init__(task_list=task_list, **kw)

        if self.dev is None and self.recalibrate_ro:
            raise NotImplementedError(
                "Device must be past when 'recalibrate_ro' is True"
                " because the mqm.measure_ssro() requires the device "
                "as argument. TODO: transcribe measure_ssro to QExperiment"
                " framework to avoid this constraint.")

        # all tasks must have same init sweep point because this will
        # fix the number of readouts
        # for now sweep points are global. But we could make the second
        # dimension task-dependent when introducing the sweep over
        # thresholds
        default_sp = SweepPoints("initialize", self.prep_states)
        default_sp.add_sweep_dimension()

        # # second dimension to have once only readout and once with feedback
        # default_sp.add_sweep_parameter("pulse_off", [1, 0, 0 ... 0])
        # default_sp.add_sweep_parameter("thresholds_u1", thresholds)

        self.sweep_points = kw.get('sweep_points',
                                   default_sp)

        # get preparation parameters for all qubits. Note: in the future we could
        # possibly modify prep_params to be different for each uhf, as long as
        # the number of readout is the same for all UHFs in the experiment
        self.prep_params = deepcopy(self.get_prep_params())
        # set explicitly some preparation params so that they can be retrieved
        # in the analysis
        for param in ('ro_separation', 'post_ro_wait'):
            if not param in self.prep_params:
                self.prep_params[param] = self.STD_PREP_PARAMS[param]
        # set temporary values
        qb_in_exp = self.find_qubits_in_tasks(self.qubits, self.task_list)
        self.temporary_values.extend([(qb.acq_shots, self.n_shots)
                                      for qb in qb_in_exp])

        # by default empty cal points
        # FIXME: Ideally these 2 lines should be handled properly by lower level class,
        #  that does not assume calibration points, instead of overwriting
        self.cal_points = kw.get('cal_points',
                                 CalibrationPoints([qb.name for qb in qb_in_exp],
                                                   ()))
        self.cal_states = kw.get('cal_states', ())

    def prepare_measurement(self, **kw):

        if self.recalibrate_ro:
            self.analysis = mqm.measure_ssro(self.dev, self.qubits, self.prep_states,
                                             update=True,
                                             n_shots=self.n_shots,
                                             analysis_kwargs=dict(
                                                 options_dict=dict(
                                                 hist_scale="log")))
            # reanalyze to get thresholds
            options_dict = dict(classif_method="threshold", hist_scale="log")
            a = tda.MultiQutrit_Singleshot_Readout_Analysis(qb_names=self.qb_names,
                                                            options_dict=options_dict)
            for qb in self.qubits:
                classifier_params = a.proc_data_dict[
                    'analysis_params']['classifier_params'][qb.name]
                qb.acq_classifier_params().update(classifier_params)
                qb.preparation_params()['threshold_mapping'] = \
                    classifier_params['mapping']
        if self.set_thresholds:
            self._set_thresholds(self.qubits)
        self.exp_metadata.update({"thresholds":
                                      self._get_thresholds(self.qubits)})
        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        self.sequences, self.mc_points = \
            self.parallel_sweep(self.preprocessed_task_list,
                                self.reset_block, block_align="start", **kw)

        # should transform raw voltage to probas in analysis if no cal points
        # and not classified readout already
        predict_proba = len(self.cal_states) == 0 and not self.classified
        self.exp_metadata.update({"n_shots": self.n_shots,
                                  "predict_proba": predict_proba,
                                  "reset_reps": self.reset_reps})

    def reset_block(self, qubit, **kw):
        _ , qubit = self.get_qubits(qubit) # ensure qubit in list format

        prep_params = deepcopy(self.prep_params)

        self.prep_params['ro_separation'] = ro_sep = prep_params.get("ro_separation",
                                 self.STD_PREP_PARAMS['ro_separation'])
        # remove the reset repetition for preparation and use the number
        # of reset reps for characterization (provided in the experiment)
        prep_params.pop('reset_reps', None)
        prep_params.pop('preparation_type', None)

        reset_type = f"active_reset_{'e' if len(self.prep_states) < 3 else 'ef'}"
        reset_block = self.prepare(block_name="reset_ro_and_feedback_pulses",
                                   qb_names=qubit,
                                   preparation_type=reset_type,
                                   reset_reps=self.reset_reps, **prep_params)
        # delay the reset block by appropriate time as self.prepare otherwise adds reset
        # pulses before segment start
        # reset_block.block_start.update({"pulse_delay": ro_sep * self.reset_reps})
        pulse_modifs={"attr=pulse_off, op_code=X180": ParametricValue("pulse_off"),
                      "attr=pulse_off, op_code=X180_ef": ParametricValue("pulse_off")}
        reset_block = Block("Reset_block",
                            reset_block.build(block_delay=ro_sep * self.reset_reps),
                            pulse_modifs=pulse_modifs)

        ro = self.mux_readout(qubit)
        return [reset_block, ro]

    def run_analysis(self, analysis_class=None, **kwargs):

        self.analysis = tda.MultiQutritActiveResetAnalysis(**kwargs)

    def run_update(self, **kw):
        print('Update')

    @staticmethod
    def _set_thresholds(qubits, clf_params=None):
        """
        Sets the thresholds in clf_params to the corresponding UHF channel(s)
        for each qubit in qubits.
        Args:
            qubits (list, QuDevTransmon): (list of) qubit(s)
            clf_params (dict): dictionary containing the thresholds that must
                be set on the corresponding UHF channel(s).
                If several qubits are passed, then it assumes clf_params if of the form:
                {qbi: clf_params_qbi, ...}, where clf_params_qbi contains at least
                the "threshold" key.
                If a single qubit qbi is passed (not in a list), then expects only
                clf_params_qbi.
                If None, then defaults to qb.acq_classifier_params().

        Returns:

        """

        # check if single qubit provided
        if np.ndim(qubits) == 0:
            clf_params = {qubits.name: deepcopy(clf_params)}
            qubits = [qubits]

        if clf_params is None:
            clf_params = {qb.name: qb.acq_classifier_params() for qb in qubits}

        for qb in qubits:
            # perpare correspondance between integration unit (key)
            # and uhf channel
            channels = {0: qb.acq_I_channel(), 1: qb.acq_Q_channel()}
            # set thresholds
            for unit, thresh in clf_params[qb.name]['thresholds'].items():
                qb.instr_uhf.get_instr().set(
                    f'qas_0_thresholds_{channels[unit]}_level', thresh)

    @staticmethod
    def _get_thresholds(qubits, from_clf_params=False, all_qb_channels=False):
        """
        Gets the UHF channel thresholds for each qubit in qubits.
        Args:
            qubits (list, QuDevTransmon): (list of) qubit(s)
            from_clf_params (bool): whether thresholds should be retrieved
                from the classifier parameters (when True) or from the UHF
                channel directly (when False).
            all_qb_channels (bool): whether all thresholds should be retrieved
                or only the ones in use for the current weight type of the qubit.
        Returns:

        """

        # check if single qubit provided
        if np.ndim(qubits) == 0:
            qubits = [qubits]

        thresholds = {}
        for qb in qubits:
            # perpare correspondance between integration unit (key)
            # and uhf channel; check if only one channel is asked for
            # (not asked for all qb channels and weight type uses only 1)
            if not all_qb_channels and qb.acq_weights_type() \
                    in ('square_root', 'optimal'):
                chs = {0: qb.acq_I_channel()}
            else:
                # other weight types have 2 channels
                chs = {0: qb.acq_I_channel(), 1: qb.acq_Q_channel()}

            #get clf thresholds
            if from_clf_params:
                thresh_qb = deepcopy(
                    qb.acq_classifier_params().get("thresholds", {}))
                thresholds[qb.name] = {u: thr for u, thr in thresh_qb.items()
                                       if u in chs}
            # get UHF thresholds
            else:
                thresholds[qb.name] = \
                    {u: qb.instr_uhf.get_instr()
                          .get(f'qas_0_thresholds_{ch}_level')
                     for u, ch in chs.items()}

        return thresholds
