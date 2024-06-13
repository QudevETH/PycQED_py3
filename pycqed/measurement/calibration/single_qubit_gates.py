import numpy as np
from collections import OrderedDict as odict
from copy import copy, deepcopy
import traceback

from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
from pycqed.measurement.waveform_control.block import ParametricValue, Block
from pycqed.measurement.waveform_control import segment as seg_mod
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.utilities.errors import handle_exception
from pycqed.utilities import general as gen
from pycqed.utilities.general import temporary_value
from pycqed.measurement import multi_qubit_module as mqm
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
import logging

from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)


class T1FrequencySweep(CalibBuilder):

    default_experiment_name = 'T1_frequency_sweep'
    kw_for_task_keys = ['transition_name']

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
            transition_name (str, default: 'ge'): Qubit transition to
                measure. Supported values: 'ge', 'ef'.
            spectator_op_codes (list, default: []): see t1_flux_pulse_block
            all_fits (bool, default: True) passed to run_analysis; see
                docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
         - the entry "qb" in each task should contain one qubit name.
         - if force_2D_sweep is False and the first sweep dim is empty or has
           only 1 point, the flux pulse amplitude will be swept as a 1D sweep.

        """
        try:
            if task_list is None:
                if sweep_points is None or qubits is None:
                    raise ValueError('Please provide either "sweep_points" '
                                     'and "qubits," or "task_list" containing '
                                     'this information.')
                task_list = [{'qb': qb.name} for qb in qubits]
            for task in task_list:
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name

            if 'cal_states' not in kw:
                kw['cal_states'] = "gef" if ('ef' in [
                    task.get('transition_name') for task in task_list]
                    + [kw.get('transition_name')]) else "ge"

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.analysis = None
            self.sweep_points = SweepPoints(
                [{}, {}] if self.sweep_points is None else self.sweep_points)
            self.task_list = self.add_amplitude_sweep_points(
                [copy(t) for t in self.task_list], **kw)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            trans_to_pop = {'ge': 'pe', 'ef': 'pf'}
            self.data_to_fit = {
                task['qb']: trans_to_pop[task.get('transition_name', 'ge')]
                for task in self.preprocessed_task_list}
            if not self.force_2D_sweep and self.sweep_points.length(0) <= 1:
                self.sweep_points.reduce_dim(1, inplace=True)
                self._num_sweep_dims = 1
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
        for task in task_list:
            # Combines sweep points in task and in sweep_points
            sweep_points = task.get('sweep_points', [{}, {}])
            sweep_points = SweepPoints(sweep_points)
            if len(sweep_points) == 1:
                sweep_points.add_sweep_dimension()
            if len(self.sweep_points) == 1:
                self.sweep_points.add_sweep_dimension()
            for i in range(len(sweep_points)):
                sweep_points[i].update(self.sweep_points[i])
            if 'qubit_freqs' in sweep_points[1]:
                qubit_freqs = sweep_points['qubit_freqs']
            else:
                qubit_freqs = None
            amplitudes = None
            for key in sweep_points[1]:
                if 'amplitude' in key:  # Detect e.g. amplitude2 from 2qb gates
                    amplitudes = sweep_points[key]
            qubits, _ = self.get_qubits(task['qb'])
            # Computing either qubit_freqs or amplitudes, if not passed.
            # Both can also be passed, e.g. to cache or use a different model.
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
            else:
                raise ValueError("Please specify either qubit_freqs or "
                                 "amplitudes!")
            task['sweep_points'] = sweep_points
        return task_list

    def t1_flux_pulse_block(self, qb, sweep_points, prepend_pulse_dicts=None,
                            op_code=None, **kw):
        """
        Function that constructs the experiment block for one qubit
        :param qb: name or list with the name of the qubit
            to measure. This function expect only one qubit to measure!
        :param sweep_points: SweepPoints class instance
        :param prepend_pulse_dicts: dictionary of pulses to prepend
        :param op_code: optional op_code for the flux pulse
        :param kw: keyword arguments
            spectator_op_codes: list of op_codes for spectator qubits
        :return: precompiled block
        """

        qubit_name = qb
        if isinstance(qubit_name, list):
            qubit_name = qubit_name[0]
        hard_sweep_dict, soft_sweep_dict = sweep_points
        pp = [self.block_from_pulse_dicts(prepend_pulse_dicts)]

        pulse_modifs = {'all': {'element_name': 'pi_pulse'}}
        pp += [self.block_from_ops('pipulse',
                                 [f'X180 {qubit_name}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)]
        if kw.get('transition_name') == 'ef':
            pp += [self.block_from_ops('pipulse_ef',
                                     [f'X180_ef {qubit_name}'] +
                                     kw.get('spectator_op_codes', []),
                                     pulse_modifs=pulse_modifs)]


        pulse_modifs = {
            'all': {'element_name': 'flux_pulse', 'pulse_delay': 0}}
        op_code = f'FP {qubit_name}' if op_code is None else op_code
        fp = self.block_from_ops('flux', [op_code],
                                 pulse_modifs=pulse_modifs)
        for k in hard_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)
        for k in soft_sweep_dict:
            for p in fp.pulses:
                if k in p:
                    p[k] = ParametricValue(k)
        pp += [fp]

        return self.sequential_blocks(f't1 flux pulse {qubit_name}', pp)

    @Timer()
    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param kw:
            all_fits (bool, default: True): whether to do all fits
        """

        if len(self.sweep_points) == 1:
            self.analysis = tda.MultiQubit_TimeDomain_Analysis()
            return
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
                 adapt_drive_amp=False, adapt_ro_freq=False,
                 internal_modulation=None, **kw):
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
        self.internal_modulation = (True if internal_modulation is None
                                            and allowed_lo_freqs is not None
                                    else internal_modulation)
        if allowed_lo_freqs is None and internal_modulation:
            log.warning('ParallelLOSweepExperiment: internal_modulation is '
                        'set to True, but this will be ignored in a pure LO '
                        'sweep, where no allowed_lo_freqs are set.')
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
                lo = qb.get_ge_lo_identifier()
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
                k: v for k, v in self.lo_offsets.items()}

        if self.allowed_lo_freqs is not None:
            if self.internal_modulation:
                # HDAWG internal modulation is needed, switch off modulation
                # in the waveform generation
                for task in self.preprocessed_task_list:
                    task['pulse_modifs'] = {'attr=mod_frequency': None}
                self.cal_points.pulse_modifs = {'attr=mod_frequency': None}
            else:
                def major_minor_func(val, major_values):
                    ind = np.argmin(np.abs(major_values - val))
                    mval = major_values[ind]
                    return (mval, val - mval)

                modifs = {}
                for task in self.preprocessed_task_list:
                    qb = self.get_qubits(task['qb'])[0][0]
                    lo = qb.get_ge_lo_identifier()
                    if len(self.lo_qubits[lo]) > 1:
                        raise NotImplementedError(
                            'ParallelLOSweepExperiment with '
                            'internal_modulation=False is currently only '
                            'implemented for a single qubit per LO.'
                        )
                    if not kw.get('optimize_mod_freqs', False):
                        raise NotImplementedError(
                            'ParallelLOSweepExperiment with '
                            'internal_modulation=False is currently only '
                            'implemented for optimize_mod_freqs=True.'
                        )
                    maj_vals = np.array(self.allowed_lo_freqs)
                    func = lambda x, mv=maj_vals : major_minor_func(x, mv)[1]
                    if 'pulse_modifs' not in task:
                        task['pulse_modifs'] = {}
                    # Below, we replace mod_frequency by a ParametricValue in all X180
                    # pulses in all task-specific blocks and in all calibration point
                    # segments. Since the cal segments are generated globally (and not
                    # per task), we need to manually ensure that a sweep parameter with
                    # task prefix is used if it exists. If no task-specific freq sweep
                    # parameter is found, the global freq sweep parameter is used. For
                    # the task-specific blocks, prefixing is automatically taken into
                    # account by the base class.
                    params = ['freq'] * 2
                    pre_param = task['prefix'] + params[1]
                    if self.sweep_points.find_parameter(pre_param) is not None:
                        params[1] = pre_param
                    for d, sp in zip([task['pulse_modifs'], modifs], params):
                        d.update({
                            f'op_code=X180 {qb.name}, attr=mod_frequency':
                                ParametricValue(sp, func=func)})
                self.cal_points.pulse_modifs = modifs

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
            self.lo_qubits[lo][0].swf_drive_lo_freq(allow_IF_sweep=False),
            offset, name=name, parameter_name=name, unit='Hz')
            for lo, offset in self.lo_offsets.items()]
        if self.allowed_lo_freqs is not None:
            minor_sweep_functions = []
            for lo, qbs in self.lo_qubits.items():
                if not self.internal_modulation:
                    # Minor sweep function not needed. Use dummy sweep.
                    minor_sweep_functions = [
                        swf.Soft_Sweep() for i in range(len(sweep_functions))]
                    break
                qb_sweep_functions = []
                for qb in qbs:
                    mod_freq = self.get_pulses(f"X180 {qb.name}")[0][
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
            temp_vals.append((task['fluxline'], task['fluxline']()))
            qb = self.get_qubits(task['qb'])[0][0]
            dc_amp = (lambda x, o=self.qb_offsets[qb], qb=qb:
                      qb.calculate_flux_voltage(x + o))
            sweep_functions += [swf.Transformed_Sweep(
                task['fluxline'], transformation=dc_amp,
                name=f'DC Offset {qb.name}',
                parameter_name=f'Parking freq {qb.name}', unit='Hz')]
        if self.allowed_lo_freqs is None or self.internal_modulation:
            # The dimension 1 sweep is a parallel sweep of all sweep_functions
            # created here, and they directly understand the sweep points
            # stored in self.lo_sweep_points.
            self.sweep_functions = [
                self.sweep_functions[0], swf.multi_sweep_function(
                    sweep_functions, name=name, parameter_name=name)]
            self.mc_points[1] = self.lo_sweep_points
        else:
            # IF sweep without internal modulation, i.e., we have to
            # reprogram the drive AWG for every sequence. Thus, the sweep
            # in dimension 1 is a parallel sweep of a SegmentSoftSweep and of
            # all sweep_functions created here. Since the SegmentSoftSweep
            # requires indices as sweep points, we use an Indexed_Sweep to
            # translate the indices to the sweep points stored in
            # self.lo_sweep_points, which are required by the sweep
            # functions created here.
            self.sweep_functions = [
                self.sweep_functions[0], swf.multi_sweep_function([
                    awg_swf.SegmentSoftSweep,  # placeholder, see _configure_mc
                    swf.Indexed_Sweep(swf.multi_sweep_function(
                        sweep_functions, name=name, parameter_name=name),
                        self.lo_sweep_points)])]
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
    default_experiment_name = 'Flux_scope'

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
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

        if ro_pulse_delay == 'auto' and (fp_truncation or \
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

    default_experiment_name = 'Cryoscope'

    def __init__(self, task_list, sweep_points=None, estimation_window=None,
                 separation_buffer=50e-9, awg_sample_length=None,
                 sequential=False, **kw):
        try:
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
                pd_temp.update(self.get_pulses(fpd['op_code'])[0])
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

class FluxPulseTiming(FluxPulseScope):
    """
        Flux pulse timing measurement used to determine the determine the
        timing of the flux pulse with respect to the qubit drive.
        It is based on the flux pulse scope measurement and thus
        features the same pulse sequnce but with the drive frequency
        fixed at the qubit ge frequency.
        pulse sequence:
                      <- delay ->
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- fluxpulse ----- |

            sweep_points:
            delay (numpy array): array of delays of the flux pulse

        Returns: None
    """
    default_experiment_name = 'FluxPulseTiming'
    kw_for_sweep_points = dict(
        **FluxPulseScope.kw_for_sweep_points,
        qb=dict(param_name='freq', unit='Hz',
                label=r'drive frequency, $f_d$',
                values_func='get_ge_freq',
                dimension=1),
    )

    def get_ge_freq(self, qb):
        """
        Returns the ge frequency of the provided qubit name. This is used
        to create the sweep points of length 1 of the routine.
        :param qb: (string) name of qubit

        Returns:
            list containing as single entry the ge frequency of the qubit
        """
        qb = self.get_qubits(qb)[0][0]
        return [qb.ge_freq()]

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw:
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.FluxPulseTimingAnalysis(
            qb_names=self.meas_obj_names, **analysis_kwargs)

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
    default_experiment_name = 'Flux_amplitude'

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
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

class ReadoutPulseScope(ParallelLOSweepExperiment):
    """
        Readout pulse scope measurement used to determine the delay of the
        qubit's drive AWG with respect to the qubit readout pulse.

        pulse sequence:
           |    -------------    |X180|  ---------------------  |RO|
           |    ---   | ---- RO ----- |


            sweep_points:
            delays (numpy array): array of delays of the drive pulse w.r.t.
            the readout pulse
            freq (numpy array): array of drive frequencies

        Returns: None

    """

    kw_for_task_keys = ['ro_separation']
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'drive frequency, $f_d$',
                      dimension=1),
        'delays': dict(param_name='delay', unit='s',
                       label=r'readout pulse delay',
                       dimension=0),
    }
    default_experiment_name = 'Readout_pulse_scope'

    def __init__(self, task_list, sweep_points=None, **kw):
        # configure detector function parameters
        kw['df_kwargs'] = kw.get('df_kwargs', {})
        kw['df_kwargs'].update(
            {'values_per_point': 2,
             'values_per_point_suffix': ['_probe', '_measure']})

        try:
            super().__init__(task_list, sweep_points=sweep_points, **kw)
            self.exp_metadata.update({'rotation_type': 'global_PCA'})
            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, ro_separation,
                    prepend_pulse_dicts=None, **kw):
        """
        Performs X180 pulse on top of a readout pulse.
        :param qb: (str) the name of the qubit
        :param sweep_points: (SweepPoints object or list of dicts or None)
        sweep points valid for all tasks.
        :param ro_separation: (float) separation between the two readout
        pulses as specified between the start of both pulses.
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            block_from_pulse_dicts
        :param kw:
        """
        b = self.block_from_ops('ro_ge', [f'RO {qb}', f'X180 {qb}'])

        ro = b.pulses[0]
        # here probe refers to the X180 pulse that "probes" the
        # first readout pulse
        probe = b.pulses[1]
        probe['ref_point'] = 'start'
        probe['ref_point_new'] = 'end'

        # make sure that no pulse starts before 0 point of the block
        probe_pulse_length = probe['sigma'] * probe['nr_sigma']
        ro['pulse_delay'] = -min(sweep_points['delay']) + probe_pulse_length
        probe['pulse_delay'] = ParametricValue('delay')

        b_ro = self.block_from_ops('final_ro', [f'RO {qb}'])

        # Assure that ro separation is comensurate with start granularity
        pulsar_obj = self.get_qubits(qb)[0][0].instr_pulsar.get_instr()
        acq_instr = self.get_qubits(qb)[0][0].instr_acq()
        # Pulsar parameter _element_start_granularity is
        # set to 0 for acq instruments. Access parameter
        # via ELEMENT_START_GRANULARITY
        gran = pulsar_obj.awg_interfaces[acq_instr].ELEMENT_START_GRANULARITY
        ro_separation -= ro_separation % (-gran)
        b_ro.pulses[0]['pulse_delay'] = ro_separation
        b = self.simultaneous_blocks('final', [b, b_ro])
        if prepend_pulse_dicts is not None:
            pb = self.block_from_pulse_dicts(prepend_pulse_dicts,
                                             block_name='prepend')
            b = self.sequential_blocks('final_with_prepend', [pb, b])
        return b

    @Timer()
    def run_analysis(self, analysis_kwargs={}, **kw):
        """
        Runs analysis and stores analysis instances in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        """

        self.analysis = tda.MultiQubit_TimeDomain_Analysis(
            qb_names=self.meas_obj_names, **analysis_kwargs)

    def seg_from_cal_points(self, *args, **kw):
        """
        Configure super.seg_from_cal_points() for sequence with two readouts
        per segment and hence twice the number of calibration points. Check
        super() method for args, kw and returned values.
        """
        # given that this routine makes use of two read out pulses
        # per segment, also two repetitions per cal state have to
        # be added to allow the analysis to run successfully
        number_of_cal_repetitions = 2
        kw['df_values_per_point'] = number_of_cal_repetitions
        return super().seg_from_cal_points(*args, **kw)

    def sweep_n_dim(self, *args, **kw):
        """
        Configure super.sweep_n_dim() for sequence with two readouts
        per segment. Check super() method for args, kw and returned values.
        """
        n_reps = 2
        seqs, vals = super().sweep_n_dim(*args, **kw)
        n_acqs = int(len(vals[0])/n_reps)
        vals[0] = vals[0][:n_acqs]
        return seqs, vals


class SingleQubitGateCalibExperiment(CalibBuilder):
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

        The following keyword arguments and their value will be copied to each
            task which does not already contain them.

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
    default_experiment_name = 'SingleQubitGateCalibExperiment'
    call_parallel_sweep = True  # whether to call parallel_sweep of parent

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

            if not self.call_parallel_sweep:
                # For these experiments the pulse sequence is not identical for
                # each all sweep points so the block function must be called
                # at each iteration in sweep_n_dim.
                if self._num_sweep_dims == 2 and len(self.sweep_points[1]) == 0:
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

            self.autorun(**kw)

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
        in the preprocessed_task_list and self.meas_obj_names,
        and adds it to exp_metadata.
        This is of the form {qb_name: {cal_state: cal_state_order_index}}
        and will be used by the analyses.
        """
        if len(self.cal_states) < 2:
            # Data rotation in analysis needs at least 2 cal states
            return

        cal_states_rotations = {}
        # add cal state rotations which are task-specific
        for task in self.preprocessed_task_list:
            qb_name = task['qb']
            if 'cal_states_rotations' in task:
                # specified by user
                cal_states_rotations.update(task['cal_states_rotations'])
            elif len(self.cal_states) > 3:
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

        # Add task-independent cal states rotations
        for qb_name in self.meas_obj_names:
            if qb_name not in cal_states_rotations:
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
        :param prepend_pulse_dicts: (list of dict) prepended pulses, see
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

    def run_measurement(self, **kw):
        if 'store_preprocessed_task_list' not in kw:
            # parameters in preprocessed_task_list should also be availiable later in analysis
            kw['store_preprocessed_task_list'] = True

        super().run_measurement(**kw)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'].update({
            SingleQubitGateCalibExperiment.__name__: odict({})
            })
        d['task_list_fields'].update({
            SingleQubitGateCalibExperiment.__name__: odict({
                'qb': ((QuDev_transmon, 'single_select'), None),
                'transition_name': (['ge', 'ef', 'fh'], 'ge'),
            })
        })
        return d


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

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys + ['n']
    kw_for_sweep_points = {
        'amps': dict(param_name='amplitude', unit='V',
                     label='Pulse Amplitude', dimension=0)
    }
    default_experiment_name = 'Rabi'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 amps=None, **kw):
        try:
            if 'n' not in kw:
                # add default n to kw before passing to init of parent
                kw['n'] = 1
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

    def sweep_block(self, qb, sweep_points, transition_name,
                    prep_transition=True, **kw):
        """
        This function creates the blocks for a single Rabi measurement task,
        see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: transmon transition to be tuned up. Can be
            "", "_ef", "_fh". See the docstring of parent method.
        :param prep_transition: Whether to prepare the initial state of the
            transition or not. This feature is required, e.g. by the thermal
            population measurement which is sweeping the pulse amplitude of
            the preparation pulse. Defaults to True.
        :param kw: keyword arguments
            n: (int, default: 1) number of Rabi pulses (X180_tr_name in the
                pulse sequence). Amplitude of all these pulses will be swept.
        """

        # create prepended pulses
        prepend_blocks = super().sweep_block(qb, sweep_points,
            transition_name=transition_name if prep_transition else '', **kw)
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

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['task_list_fields'].update({
            Rabi.__name__: odict({
                'n': (int, 1),
            })
        })
        d['sweeping_parameters'].update({
            Rabi.__name__: {
                0: {
                    'amplitude': 'V',
                },
                1: {
                    'sigma': 's',
                },
            }
        })
        return d


class ThermalPopulation(Rabi):
    """
    Experiment to determine the residual thermal population in the e state by
    performing two subsequent Rabi experiments on the ef transition. For one of
    them we prepare the e state before each ef Rabi pulse and for the other,
    the qubit starts out in the thermal equilibrium state. By comparing the
    amplitudes of the two Rabi oscillations, one can infer the thermal e
    state population.

    Args:
        amps (list): Amplitude sweep points, see docstring of the parent class.
        In this QuantumExperiment they are used as amplitude of the X180_ef
        pulse, while the amplitude of the X180_ge pulse is set by qb.ge_amp180.

    TODO: extend to states other than the e state
    """
    default_experiment_name = 'Thermal Population'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 amps=None, **kw):
        try:
            transition_name = kw.pop('transition_name', 'ef')
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             transition_name=transition_name,
                             amps=amps, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_sweep_points(self):
        for qbn in self.qb_names:
            ge_amp = self.dev.get_operation_dict()[f'X180 {qbn}']['amplitude']
            self.sweep_points.add_sweep_parameter(
                param_name=f'{qbn}_amplitude_ge',
                values=np.array([0.0, ge_amp]),
                unit='V',
                label='amplitude ge',
                dimension=1,
            )
        return super().update_sweep_points()

    def sweep_block(self, qb, sweep_points, **kw):
        prepend_pulse_dicts = kw.pop('prepend_pulse_dicts', list())
        prepend_pulse_dicts += [{'op_code': f'X180 {qb}',
                    'amplitude': ParametricValue(f'{qb}_amplitude_ge')}]

        transition_name = kw.pop('transition_name', '_ef')

        return super().sweep_block(qb, sweep_points, transition_name,
            prep_transition=False, prepend_pulse_dicts=prepend_pulse_dicts,
            **kw)

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis class
        :param kw: keyword arguments
            Passed to parent method.
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.ThermalPopulationAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['task_list_fields'].update({
            ThermalPopulation.__name__: odict({
                'transition_name': (['ge', 'ef', 'fh'], 'ef'),
            })
        })
        return d


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
    :param minimum_sampling_ratio: (float, default: 2) minimum sampling
        period of the Ramsey measurement relative to the expected
        artificial detuning. Sampling ratios below this (too low of an
        artificial detuning or too large time steps) trigger a warning.
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

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys + [
        'artificial_detuning']
    kw_for_sweep_points = {
        'delays': dict(param_name='pulse_delay', unit='s',
                       label=r'Second $\pi$-half pulse delay', dimension=0)
    }
    default_experiment_name = 'Ramsey'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, echo=False, minimum_sampling_ratio=2, **kw):
        try:
            if 'artificial_detuning' not in kw:
                # add default artificial_detuning to kw before passing to
                # init of parent
                kw['artificial_detuning'] = 0

            self.echo = echo
            self.minimum_sampling_ratio = minimum_sampling_ratio
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             delays=delays, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_experiment_name(self):
        """
        Updates self.experiment_name to Echo if self.echo is True.
        """
        if self.echo:
            self.experiment_name = self.experiment_name.replace('Ramsey', 'Echo')

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

    def sweep_block(self, qb, sweep_points, transition_name, center_block=None,
                    **kw):
        """
        This function creates the blocks for a single Ramsey/Echo measurement
        task, see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: transmon transition to be tuned up. Can be
            "", "_ef", "_fh". See the docstring of parent method.
        :param center_block: Block instance. This block is executed in the
            center of the Ramsey sequence. Ignored if None. Defaults to None.
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

        delays = sweep_points.get_sweep_params_property('values', 0,
                                                        'pulse_delay')
        delta_t_min = np.min(np.diff(delays))
        artificial_oscillation_period = 1/art_det
        sampling_ratio = artificial_oscillation_period/delta_t_min
        if sampling_ratio < self.minimum_sampling_ratio:
            log.warning(
                f'Chosen artificial detuning {art_det} and minimum delta '
                f'between delays {delta_t_min} results in a sampling ratio of '
                f'{sampling_ratio}, below the minimum of '
                f'{self.minimum_sampling_ratio}. Decrease the spacing between '
                'delays or reduce the artificial detuning.'
            )

        if self.echo:
            # add echo block: pi-pulse halfway between the two X90_tr_name
            # pulses
            echo_block = self.block_from_ops(f'echo_pulse_{qb}',
                                             [f'X180{transition_name} {qb}'])
            ramsey_block = self.simultaneous_blocks(f'main_{qb}',
                                                    [ramsey_block, echo_block],
                                                    block_align='center')
        if center_block is not None:
            ramsey_block = self.simultaneous_blocks(f'main_with_center_{qb}',
                                                [ramsey_block, center_block],
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
        self.analysis = tda.EchoAnalysis if self.echo else tda.RamseyAnalysis
        self.analysis = self.analysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the following parameters of the qubit in each task with the
        values extracted by the analysis:

            For Ramsey measurement:
                - transition frequency: tr_name_freq
                - averaged dephasing time: T2_star_tr_name
            In addition, for Ramsey ef measurement:
                - anharmonicity
            For Echo measurement:
                - dephasing time: T2_tr_name

        :param kw: keyword arguments
        """
        if self._num_sweep_dims > 1:
            log.warning('Updating not supported with 2D '
                        'sweep_points. Skipping update.')
        else:
            for task in self.preprocessed_task_list:
                qubit = [qb for qb in self.meas_objs
                         if qb.name == task['qb']][0]
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
                    qubit.set(f'{task["transition_name_input"]}_freq',
                              qb_freq)
                    qubit.set(f'T2_star{task["transition_name"]}',
                              T2_star)
                    if task["transition_name_input"] == 'ef':
                        qubit.set('anharmonicity',
                                  qb_freq - qubit.get('ge_freq'))

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'].update({
            Ramsey.__name__: odict({
                'echo': (bool, False),
            })
        })
        d['task_list_fields'].update({
            Ramsey.__name__: odict({
                'artificial_detuning': (float, None),
            })
        })
        d['sweeping_parameters'].update({
            Ramsey.__name__: {
                0: {
                    'pulse_delay': 's',
                },
                1: {},
            }
        })
        return d


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

    kw_for_task_keys = Ramsey.kw_for_task_keys + ['fluxline']
    kw_for_sweep_points = {
        'delays': dict(param_name='pulse_delay', unit='s',
                       label=r'Second $\pi$-half pulse delay', dimension=0),
        'dc_voltages': dict(param_name='dc_voltages', unit='V',
                            label=r'DC voltage', dimension=1),
        'dc_voltage_offsets': dict(param_name='dc_voltage_offsets', unit='V',
                                   label=r'DC voltage offset', dimension=1),
    }
    default_experiment_name = 'ReparkingRamsey'


    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, dc_voltages=None, dc_voltage_offsets=None,  **kw):

        try:
            if 'fluxline' not in kw:
                # add default value for fluxline to kw before passing to
                # init of parent
                kw['fluxline'] = None
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
        current dc voltage values, before calling the method of the
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
                      apd['reparking_params'][qubit.name]['new_ss_vals'][
                          'ss_freq'])
            # set new voltage
            fluxline(apd['reparking_params'][qubit.name]['new_ss_vals'][
                         'ss_volt'])


class ResidualZZ(Ramsey):
    """Measurement of the residual ZZ coupling between two qubits.

    This is done by measuring two subsequent Ramsey experiments on the target
    qubit, one of the Ramsey experiments leaves the control qubit in the ground
    state and the other one excites it to the e state. The time at which the
    second qubit is excited depends on whether one chooses to perform an echo
    pulse as part of the Ramsey sequence or not. Without the echo pulse the
    control qb is excited before the Ramsey sequence starts. In the echo case
    the centers of the two X180 gates line up at the middle between the two
    Ramsey X90 pulses. By comparing the detuning between the two Ramsey
    experiments one can infer the residual coupling between the two qubits.

    No echo:
        qbc:    |X180| - ... -
        qbt:     ... - |X90| - ... - |X90| -- |RO|
                              sweep
                            delay tau

    Including echo:
        qbc:          - ... - |X180| - ... -
        qbt:    |X90| - ... - |X180| - ... - |X90| -- |RO|
                       sweep          sweep
                    delay tau/2    delay tau/2

    See parent classes for parameters of the class and the task_list.
    In addition to the target qubit (param name "qb") one needs to specify the
    control qubit with the key "qbc".

    Note: IT IS NOT RECOMMENDED TO RUN RESIDUAL ZZ MEASUREMENTS IN PARALLEL!
    because of the influence between the individual measurments. Parallel
    experiments are only supported for disjoint pairs of qubits,
    e.g.    task1 = {'qb': qb1, 'qbc': qb2, ...} and
            task2 = {'qb': qb7, 'qbc': qb8, ...}
    but not task1 = {'qb': qb1, 'qbc': qb2, ...} and
            task2 = {'qb': qb2, 'qbc': qb4, ...}
    or      task1 = {'qb': qb1, 'qbc': qb2, ...} and
            task2 = {'qb': qb3, 'qbc': qb2, ...}
    """
    task_mobj_keys = ['qb', 'qbc']

    default_experiment_name = 'ResidualZZ'

    def __init__(self, task_list=None, **kw):
        try:
            super().__init__(task_list, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def preprocess_task_list(self, **kw):
        """Calls super method and in addition checks whether the different tasks
        are compatible with each other.

        Raises:
            NotImplementedError: Raised if one qubit is involved in more than
                one task.

        Returns:
            list: preprocessed task list
        """
        preprocessed_task_list =  super().preprocess_task_list(**kw)

        # Warn the user when he is trying to run parallel measurments
        if len(preprocessed_task_list) > 1:
            log.warning('It is not recommended to run residual ZZ measurements '
                        'in parallel! Use at your own risk.')
        # Check that the involved qubits are pairwise dijoint between tasks:
        all_involved_qubits = []
        for task in preprocessed_task_list:
            if task['qb'] in all_involved_qubits \
                    or task['qbc'] in all_involved_qubits:
                raise NotImplementedError(f'Either {task["qb"]} or '
                                          f'{task["qbc"]} is contained in more '
                                          f'than one task. This is not '
                                          f'supported by this experiment.')
            else:
                all_involved_qubits.append(task['qb'])
                all_involved_qubits.append(task['qbc'])
        return preprocessed_task_list

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects (e.g., qubits) of a task. Here we
        overwrite the method to exclude the control qubit from the measure
        objects.
        :param task: a task dictionary
        :return: list of all qubit objects (if available) or names
        """
        qbc = task.pop('qbc')
        qubits = self.find_qubits_in_tasks(self.qb_names, [task])
        task['qbc'] = qbc
        return qubits

    def update_sweep_points(self):
        """Adds sweep point snecessary to turn on and of the pi-pulse of the
        control qubits. Calls super method afterwards.
        """
        for task in self.preprocessed_task_list:
            qbc = task['qbc']
            ge_amp = self.dev.get_operation_dict()[f'X180 {qbc}']['amplitude']
            self.sweep_points.add_sweep_parameter(
                param_name=f'{qbc}_amplitude_ge',
                values=np.array([0.0, ge_amp]),
                unit='V',
                label='amplitude ge',
                dimension=1)
        return super().update_sweep_points()

    def sweep_block(self, qbc, qb, sweep_points, **kw):
        """Adds the pi-pulse used to excite the control qubit by either creating
        a center block (echo) or by passing the correct prepend_pulse_dict to
        the super method.

        Args:
            qbc (str): name of the control qubit.
            qb (str): name of the target qubit.
            sweep_points (SweepPoints): sweep points to be passes to Ramsey
                method.
        """
        center_block = None
        prepend_pulse_dicts = kw.pop('prepend_pulse_dicts', list())
        if self.echo:
            center_block = self.block_from_pulse_dicts([{'op_code': f'X180 {qbc}',
                        'amplitude': ParametricValue(f'{qbc}_amplitude_ge')}],
                                         block_name=f'excitation_{qbc}')
        else:
            prepend_pulse_dicts += [{'op_code': f'X180 {qbc}',
                        'amplitude': ParametricValue(f'{qbc}_amplitude_ge')}]

        return super().sweep_block(qb, sweep_points,
            center_block=center_block,
            prepend_pulse_dicts=prepend_pulse_dicts,
            **kw)

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: keyword arguments
            Passed to parent method.
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.ResidualZZAnalysis(
            qb_names=[task['qb'] for task in self.preprocessed_task_list],
            t_start=self.timestamp, echo=self.echo, **analysis_kwargs)


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
    default_experiment_name = 'T1'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 delays=None, **kw):
        try:
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
        if self._num_sweep_dims > 1:
            log.warning('Updating not supported with 2D '
                        'sweep_points. Skipping update.')
        else:
            for task in self.preprocessed_task_list:
                qubit = [qb for qb in self.meas_objs
                         if qb.name == task['qb']][0]
                T1 = self.analysis.proc_data_dict[
                    'analysis_params_dict'][qubit.name]['T1']
                qubit.set(f'T1{task["transition_name"]}', T1)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['sweeping_parameters'].update({
            T1.__name__: {
                0: {
                    'pulse_delay': 's',
                },
                1: {},
            }
        })
        return d


class PhaseErrorCalib(SingleQubitGateCalibExperiment):
    """
    Phase error calibration measurement for finding the quadrature scaling
    parameter (motzoi) of an SSB_DRAG_pulse or the envelope modulation frequency
    (env_mod_frequency) of an SSB_DRAG_pulse_full/SSB_DRAG_pulse_cos that allows
    to drive a transmon transition without phase errors.
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
                               sweep motzoi or     sweep motzoi or
                               env_mod_frequency   env_mod_frequency

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure and aligned at the end
            (i.e. ends of the last drive pulses aligned with start of RO pulses)

        Note: if the qubits have different drive pulse lengths, then alignment
        at the start of RO pulse means the individual drive pulses may end up
        not being applied in parallel between different qubits.
        See Rabi class docstring for an example of this situation.

        Note: depending on which transition is tuned up in each task, the
        sequence can have different number of pulses for each qubit.

    Args
        See docstring of parent class for the first 3 parameters.

        qscales (numpy array): sweep points for the motzoi param of the pairs
            of pulses in the pulse sequence above.
            This parameter can be used together with "qubits" for convenience to
            avoid having to specify a task_list.
            If not None, qscales will be used to create the first dimension of
            sweep points, which will be identical for all tasks.
        env_mod_freqs (numpy array): sweep points for the env_mod_frequency
            param of the pairs of pulses in the pulse sequence above.
            If not None, env_mod_freqs will be used to create the first
            dimension of sweep points, which will be identical for all tasks.

        Only one of the two sweep parameters should be specified!

    Keyword Args
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - env_mod_freqs
        - qscales
    """

    kw_for_sweep_points = {
        'env_mod_freqs': dict(param_name='env_mod_frequency', unit='Hz',
                              label='Pulse Envelope Frequency', dimension=0,
                              values_func=lambda df: np.repeat(df, 3)),
        'qscales': dict(param_name='motzoi', unit='',
                        label='Quadrature Scaling Factor, $q$', dimension=0,
                        values_func=lambda q: np.repeat(q, 3))
    }
    default_experiment_name = 'PhaseErrorCalib'
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 env_mod_freqs=None, qscales=None, **kw):
        try:
            # the 3 pairs of pulses to be applied for each sweep point
            self.base_ops = [['X90', 'X180'], ['X90', 'Y180'], ['X90', 'mY180']]
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             env_mod_freqs=env_mod_freqs,
                             qscales=qscales, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_sweep_points(self):
        """
        Checks if the sweep points in the first sweep dimension are repeated 3
        times (there are 3 pairs of pulses in base_ops). If they are not,
        this method updates the self.sweep_points and the sweep points in each
        task of preprocessed_task_list with the repeated sweep points.
        """

        # update self.sweep_points
        swpts = deepcopy(self.sweep_points)
        par_names = []
        vals = []
        for par in swpts.get_parameters(0):
            values = swpts[par]
            if np.unique(values[:3]).size > 1:
                # the sweep points are not repeated 3 times (for each
                # pair of base_ops)
                par_names += [par]
                vals += [np.repeat(values, 3)]
        self.sweep_points.update_property(par_names, values=vals)

        # update sweep points in preprocessed_task_list
        for task in self.preprocessed_task_list:
            swpts = task['sweep_points']
            for par in swpts.get_parameters(0):
                values = swpts[par]
                if np.unique(values[:3]).size > 1:
                    swpts.update_property([par], values=[np.repeat(values, 3)])

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        This function creates the blocks for each task, see the pulse sequence
        in the class docstring.

        Args
            sp1d_idx (int): sweep point index in the first sweep dimension
            sp2d_idx (int): sweep point index in the second sweep dimension

        Keyword Args: to be provided in each task
            qb (str): qubit name
            sweep_points (class instance): of the SweepPoints
            transition_name (str): transmon transition to be tuned up. Can be
                "", "_ef", "_fh". See the docstring of parent method.
        """

        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            transition_name = task['transition_name']
            sweep_points = task['sweep_points']
            qb = task['qb']

            # create prepended pulses
            prepend_blocks = super().sweep_block(**task)
            # add task block consisting of the correct set of 2 pulses
            # from base_ops (depending on sp1d_idx)
            task_block = self.block_from_ops(
                f'task_pulses_{qb}', [f'{p}{transition_name} {qb}' for p in
                                      self.base_ops[sp1d_idx % 3]])
            # set the pulse parameters of the pulses in the task block
            swp_pars = sweep_points.get_parameters()
            for p in task_block.pulses:
                for spar in swp_pars:
                    if spar in p:
                        p[spar] = sweep_points[spar][sp1d_idx] if \
                            sweep_points.find_parameter(spar) == 0 else \
                            sweep_points[spar][sp2d_idx]
            # gather the blocks for each task
            parallel_block_list += [self.sequential_blocks(
                f'phase_err_cal_{qb}', prepend_blocks + [task_block])]

        return self.simultaneous_blocks(f'phase_err_cal_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.

        Args
            analysis_kwargs (dict): keyword arguments for analysis

        Keyword Args
            Passed to super call.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.QScaleAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the tr_name_motzoi parameter or the tr_name_env_mod_freq
        parameter of the qubit in each task with the value extracted by the
        analysis.

        Keyword Args
            Not used, but are here to allow pass-through.
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            pulse_par = self.analysis.proc_data_dict['analysis_params_dict'][
                qubit.name]['qscale']
            if self.analysis.pulse_par_name == 'motzoi':
                qubit.set(f'{task["transition_name_input"]}_motzoi', pulse_par)
            else:
                qubit.set(f'{task["transition_name_input"]}_env_mod_freq',
                          pulse_par)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['sweeping_parameters'].update({
            cls.__name__: {
                0: {
                    'env_mod_frequency': 'Hz',
                    'motzoi': '',
                },
                1: {},
            }
        })
        return d


class QScale(PhaseErrorCalib):
    """
    Legacy measurement class.
    PhaseErrorCalib measurement with the name QScale.
    """
    default_experiment_name = 'QScale'


class SingleQubitErrorAmplificationExperiment(SingleQubitGateCalibExperiment):
    """
    Base class for measurements using error amplification.

    This is a multitasking experiment, see docstrings of MultiTaskingExperiment
    and of CalibBuilder for general information. Each task corresponds to one
    qubit specified by the key 'qb' (either name or QuDev_transmon instance),
    i.e. multiple qubits can be measured in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb:  preparation_pulses - [amplification_pulses] x n_repetitions - |RO|
                                  sweep some pulse param

    Expected sweep points, either global or per task:
        - n_repetitions in dimension specified by n_repetitions_sweep_dim:
        list or array specifying the number of times to repeat the
        amplification_pulses
        - (optional) some pulse parameter in the other dimension: list or array
        specifying the values of this pulse parameter

    Idea behind this calibration measurement:
        If the drive pulses are perfectly calibrated, independent of
        n_repetitions, the qubit will remain in the state in which it is
        prepared by the preparation_pulses. Miscalibrations in some pulse
        parameter will be signaled by a deviation from this preparation state.

    Args
        See docstring of parent class for the first 3 parameters.

        n_repetitions (list or array): specifying the number of times to repeat
            the amplification_pulses
        preparation_pulses (list of str): op codes of preparation pulses
        amplification_pulses (list of str): op codes of the group of pulses
            that will be amplified, i.e. repeated n_repetitions times
        pulse_modifs (dict): for the amplification_pulses. Passed to
            block_from_ops, see docstring there for more details.
        analysis_class (class): analysis class, typically from
            timedomain_analysis.py. Defaults to MultiQubit_TimeDomain_Analysis.

    Keyword Args
        Can be used to provide keyword arguments to sweep_n_dim, autorun,
        and to the parent classes.

        The following keyword arguments will be copied as entries in
        sweep_points:
        - n_repetitions
        - some pulse parameter (if provided)

        Moreover, the following keyword arguments are understood:
        n_repetitions_sweep_dim (int, default: 0): dimension in which to
            sweep the n_repetitions
    """

    default_experiment_name = 'SingleQubitErrorAmplificationExperiment'
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 n_repetitions=None, preparation_pulses=(), pulse_modifs=None,
                 amplification_pulses=(), analysis_class=None, **kw):
        try:
            nreps_swpdim = kw.get('n_repetitions_sweep_dim', 0)
            self.kw_for_sweep_points.update({
                'n_repetitions': dict(param_name='n_repetitions', unit='',
                                      label='Nr. repetitions, $N$',
                                      dimension=nreps_swpdim),
            })
            self.preparation_pulses = preparation_pulses
            self.amplification_pulses = amplification_pulses
            self.pulse_modifs = pulse_modifs
            self.analysis_class = analysis_class
            if self.analysis_class is None:
                self.analysis_class = tda.MultiQubit_TimeDomain_Analysis
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             n_repetitions=n_repetitions,
                             preparation_pulses=preparation_pulses,
                             amplification_pulses=amplification_pulses,
                             pulse_modifs=pulse_modifs, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_experiment_name(self):
        """
        Updates self.experiment_name with the number of repetitions.
        """
        n_reps = []
        for task in self.preprocessed_task_list:
            n_reps += [task['sweep_points']['n_repetitions'][-1]]
        if len(np.unique(n_reps)) == 1:
            self.experiment_name += f'_{n_reps[0]}_repetitions'

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        This function creates the blocks for each task, see the pulse sequence
        in the class docstring.

        Args
            sp1d_idx (int): sweep point index in the first sweep dimension
            sp2d_idx (int): sweep point index in the second sweep dimension

        Keyword Args: to be provided in each task
            qb (str): qubit name
            sweep_points (class instance): of the SweepPoints.
                Must contain the sweep parameter 'n_repetitions'.
            transition_name (str): transmon transition to be tuned up. Can be
                "", "_ef", "_fh". See the docstring of parent method.
        """

        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            tr_name = task['transition_name']
            sweep_points = task['sweep_points']
            qb = task['qb']

            # Get the block to prepend from the parent class
            # (see docstring there)
            prepend_block = super().sweep_block(qb, sweep_points, tr_name)

            # Apply self.preparation_pulses
            pulse_list = [f'{p}{tr_name} {qb}' for p in self.preparation_pulses]
            # Find in which sweep dimension n_repetitions are
            nreps_dim = sweep_points.find_parameter('n_repetitions')
            sp_idx_nreps, sp_idx_par = (sp1d_idx, sp2d_idx) if nreps_dim == 0 \
                else (sp2d_idx, sp1d_idx)
            # Apply n_repetitions * self.amplification_pulses
            n_repetitions = sweep_points['n_repetitions'][sp_idx_nreps]
            pulse_list += n_repetitions * [f'{p}{tr_name} {qb}' for
                                           p in self.amplification_pulses]
            # Create a block from this list of pulses
            drive_calib_block = self.block_from_ops(
                f'pulses_{qb}', pulse_list, pulse_modifs=self.pulse_modifs)

            # If sweep_points contains pulse parameter, sweep them for
            # all pulses.
            for dim, sweep_dim in enumerate(sweep_points):
                for param_name in sweep_dim:
                    for pulse_dict in drive_calib_block.pulses:
                        if param_name in pulse_dict:
                            pulse_dict[param_name] = \
                                sweep_points[param_name][sp1d_idx if  dim == 0
                                else sp2d_idx]

            # Append the final block for this task to parallel_block_list
            parallel_block_list += [self.sequential_blocks(
                f'err_ampl_calib_{qb}', prepend_block + [drive_calib_block])]

        return self.simultaneous_blocks(f'err_ampl_calib_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.

        Args
            analysis_kwargs (dict): keyword arguments for analysis class

        Keyword Args
            Passed to super call.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = self.analysis_class(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)


class NPulseAmplitudeCalib(SingleQubitErrorAmplificationExperiment):
    """
    Calibration measurement for the qubit drive amplitude that makes use of
    error amplification from application of N subsequent pulses.

    This is a multitasking experiment, see docstrings of MultiTaskingExperiment
    and of CalibBuilder for general information. Each task corresponds to one
    qubit specified by the key 'qb' (either name or QuDev_transmon instance),
    i.e. multiple qubits can be measured in parallel.

    This experiment can be run in two modes:

    1. The qubit is brought into superposition with a pi-half pulse, after which
    n_repetitions pairs of pulses are applied with amplitudes of the first
    and second pulse in each pair chosen such that, ideally, the pair implements
    a rotation of pi. The correct amplitude for the pulse whose amplitude is not
    fixed by fixed_scaling can be found by sweeping around the expected correct
    amplitude (see info about sweep points below).

    This mode is enabled by specifying the input parameter fixed_scaling as
    a fraction of a pi rotation. This number will be used to scale the amplitude
    of the second pulse in the pair, while that of the first pulse will be
    scaled by amp_scalings given in sweep points (see below).

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb: |X90| --- [ |Rx(phi)| --- |Rx(pi - phi)| ] x n_repetitions --- |RO|

    2. The qubit is brought into superposition with a pi-half pulse, after which
    n_repetitions * nr_pulses_pi pulses are applied with amplitudes scaled by
    amp_scaling.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb: |X90|---[|Rx(pi/nr_pulses_pi)|] x n_repetitions x n_pulses_pi--|RO|

    For either mode, expected sweep points, either global or per task
     - n_repetitions in dimension 0: number of effective pi rotations after the
        initial pi-half pulse.
     - amp_scalings in dimension 1: dimensionless fractions of a pi rotation
        around the x-axis of the Bloch sphere.

    Idea behind this calibration measurement:
     If the pulses are perfectly calibrated, the qubit will remain in the
     superposition state with 50% excited state probability independent of
     n_repetitions and amp_scalings. Miscalibrations will be signaled by an
     oscillation of the excited state probability around 50% with increasing N.

    Note: this measurement is built with 2D sweep points (where the second
     dimension possibly has length 1), and hence needs force_2D_sweep=True
     (defaults to False in SingleQubitGateCalibExperiment) to correctly store
     the data in a 2D format as well.

    Keyword args
        Can be used to provide keyword arguments to sweep_n_dim, autorun,
        and to the parent classes.

        The following keyword arguments will be copied as entries in
        sweep_points:
        - n_repetitions: list or array
        - amp_scalings: list or array

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - n_pulses_pi (int; default: None): the number of pulses that
            will implement a pi rotation.
        - fixed_scaling (float, default: None): toggles between the two ways
            of running this experiment explained above by setting the
            amplitude scaling of the second pulse in the pair (see mode 1
            above).

        Moreover, the following keyword arguments are understood:
            for_leakage (bool, default: False): if True, runs the experiment
                without the first X90 pulse and sets cal_states to 'gef'
    """

    default_experiment_name = 'NPulseAmplitudeCalib'
    kw_for_sweep_points = {
        'amp_scalings': dict(param_name='amp_scalings', unit='',
                             label='Amplitude Scaling, $r$', dimension=1),
    }
    kw_for_task_keys = ['n_pulses_pi', 'fixed_scaling']

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 amp_scalings=None, n_pulses_pi=1, fixed_scaling=None, **kw):
        try:
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             amp_scalings=amp_scalings,
                             n_pulses_pi=n_pulses_pi,
                             fixed_scaling=fixed_scaling,
                             analysis_class=tda.NPulseAmplitudeCalibAnalysis,
                             force_2D_sweep=kw.get('force_2D_sweep', True),
                             **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def update_experiment_name(self):
        """
        Updates self.experiment_name based on the last (largest) entry in
        n_repetitions, and on the values of the parameters n_pulses_pi and
        fixed_scaling.
        """
        n_reps = []
        for task in self.preprocessed_task_list:
            n_reps += [task['sweep_points']['n_repetitions'][-1]]
        if len(np.unique(n_reps)) == 1:
            # all tasks have the same n_repetitions array: append it to the
            # experiment name
            self.experiment_name += f'_{n_reps[0]}_repetitions'

        nr_pi = [task['n_pulses_pi'] for task in self.preprocessed_task_list]
        fixed_sc_exp = any([task['fixed_scaling'] is not None
                            for task in self.preprocessed_task_list])
        if not fixed_sc_exp and len(np.unique(nr_pi)) == 1:
            # fixed_scaling is not used, and all tasks have the same
            # n_pulses_pi: append it to the experiment name
            self.experiment_name += f'_{nr_pi[0]}xpi_over_{nr_pi[0]}'

    def update_sweep_points(self):
        """
        If amp_scalings were not specified in the second sweep dimension,
        this function will add them as np.array([1/n_pulses_pi]) or
        np.array([1 - 1/n_pulses_pi]) if fixed_scaling is not None.
        """
        if len(self.sweep_points) == 1:
            # amp_scalings were not specified
            nr_ppi = []
            prefixes = []
            fixed_scalings = []
            for task in self.preprocessed_task_list:
                swpts = task['sweep_points']
                n_pulses_pi = task['n_pulses_pi']
                fixed_scaling = task['fixed_scaling']
                prefixes += [task['prefix']]
                nr_ppi += [n_pulses_pi]
                fixed_scalings += [fixed_scaling]
                vals = np.array([1/n_pulses_pi]) if fixed_scaling is None \
                    else np.array([1 - 1/n_pulses_pi])
                swpts.add_sweep_parameter('amp_scalings', vals,  unit='',
                                          label='Amplitude Scaling, $r$',
                                          dimension=1)

            # update self.sweep_points
            if len(np.unique(nr_ppi)) == 1:
                # all tasks use the same sweep points
                vals = np.array([1/nr_ppi[0]]) if fixed_scalings[0] is None \
                    else np.array([1 - 1/nr_ppi[0]])
                self.sweep_points.add_sweep_parameter(
                    'amp_scalings', vals, unit='',
                    label='Amplitude Scaling, $r$', dimension=1)
            else:
                # different values for each task
                for i in range(len(nr_ppi)):
                    vals = np.array([1 / nr_ppi[i]]) \
                        if fixed_scalings[i] is None \
                        else np.array([1 - 1 / nr_ppi[i]])
                    self.sweep_points.add_sweep_parameter(
                        f'{prefixes[i]}amp_scalings', vals, unit='',
                        label='Amplitude Scaling, $r$', dimension=1)

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        This function creates the block at the current iteration in sweep_n_dim,
        specified by sp1d_idx and sp2d_idx.

        Args:
            sp1d_idx: current index in the first sweep dimension
            sp2d_idx: current index in the second sweep dimension

        Keyword args
            To allow pass through kw even if it contains entries that are
            not needed

            To be provided in each task
                qb (str): qubit name
                sweep_points (class instance): of the SweepPoints
                transition_name (str): transmon transition to be tuned up.
                Can be "", "_ef", "_fh". See the docstring of parent method.

        Returns:
            instance of Block created with simultaneous_blocks from a list
            of blocks corresponding to the tasks in preprocessed_task_list.

        Assumes self.preprocessed_task_list has been defined and that it
        contains the entries specified by the following keys:
         - 'qb': qubit name
         - 'sweep_points': SweepPoints instance
         - 'transition_name' (see docstring of parent class)
         - 'n_pulses_pi': int specifying the number of pulses that
             will implement a pi rotation.
         - 'fixed_scaling': NOne or float specifying amplitude scaling of the
             second pulse in the pair (see below)

        If fixed_scaling is None, n_repetitions * n_pulses_pi identical
        pulses will be added after the initial pi-half pulse and their
        amplitudes will be scaled by amp_scaling.

        If fixed_scaling is specified, n_repetitions pairs of X180 pulses
        will be added after the initial pi-half pulse, where the amplitude of
        the first pulse is scaled by amp_scaling, and the amplitude of the
        second scaled by fixed_scaling.
        """

        # Define list to gather the final blocks for each task
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            transition_name = task['transition_name']
            sweep_points = task['sweep_points']
            fixed_scaling = task['fixed_scaling']
            qb = task['qb']

            # Get the block to prepend from the parent class
            # (see docstring there)
            prepend_block = SingleQubitGateCalibExperiment.sweep_block(
                    self, qb, sweep_points, transition_name)

            n_reps = sweep_points['n_repetitions'][sp1d_idx]
            sp2d = sweep_points.get_sweep_dimension(1)
            if fixed_scaling is not None:
                if len(sp2d) > 1:
                    raise NotImplementedError('Only one parameter in the second '
                                              'sweep dimension is supported '
                                              'when using fixed_scaling.')
                # After the preparation X90 pulse, apply pairs of X180 pulses,
                # where the amplitude of the first pulse is scaled by
                # amp_scaling, and the amplitude of the second scaled by
                # fixed_scaling. (specified by sp2d_idx).

                # Create the pulse list for n_repetitions specified by sp1d_idx
                pulse_list = [f'X90{transition_name} {qb}'] + \
                             (n_reps * [f'X180{transition_name} {qb}',
                                          f'X180{transition_name} {qb}'])
                # Create a block from this list of pulses
                drive_calib_block = self.block_from_ops(f'pulses_{qb}',
                                                        pulse_list)

                amp_scaling = sweep_points.get_sweep_params_property(
                    'values', 1)[sp2d_idx]
                # Scale by amp_scaling the amp of all even pulses after the
                # preparation pulse ([1:] in the indexing below)
                for pulse_dict in drive_calib_block.pulses[1:][0::2]:
                    pulse_dict['amplitude'] *= amp_scaling
                # Scale by fixed_scaling the amp of all odd pulses after the
                # preparation pulse ([1:] in the indexing below)
                for pulse_dict in drive_calib_block.pulses[1:][1::2]:
                    pulse_dict['amplitude'] *= fixed_scaling
            else:
                # Apply n_repetitions * n_pulses_pi X180 pulses after the
                # initial pi-half pulse, and scale the amplitude of these X180
                # pulses by amp_scaling (specified by sp2d_idx).

                n_pulses_pi = task['n_pulses_pi']
                # Create the pulse list for n_repetitions specified by sp1d_idx
                pulse_list = [f'X90{transition_name} {qb}']
                pulse_list += n_reps * n_pulses_pi * \
                              [f'X180{transition_name} {qb}']

                # Create a block from this list of pulses
                drive_calib_block = self.block_from_ops(f'pulses_{qb}',
                                                        pulse_list)
                # Divide by amp_scaling the amp of all pulses after the
                # preparation pulse ([1:] in the indexing below)
                for pulse_dict in drive_calib_block.pulses[1:]:
                    for param_name in sp2d:
                        values = sp2d[param_name][0]
                        if param_name == 'amp_scalings':
                            pulse_dict['amplitude'] *= values[sp2d_idx]
                        else:
                            pulse_dict[param_name] = values[sp2d_idx]

            # Append the final block for this task to parallel_block_list
            parallel_block_list += [self.sequential_blocks(
                f'drive_calib_{qb}', prepend_block + [drive_calib_block])]

        return self.simultaneous_blocks(f'drive_amp_calib_{sp2d_idx}_{sp1d_idx}',
                                        parallel_block_list, block_align='end')

    def run_update(self, **kw):
        """
        If the experiment was run for a pi-pulse or a pi-half pulse, this
        method updates the pi-pulse amplitude (tr_name_amp180) or the amp90
        scaling (tr_name_amp90_scale) of the qubit in each task with the value
        extracted by the analysis.

        Keyword args:
         to allow pass through kw even though they are not needed
        """

        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            ideal_sc = self.analysis.ideal_scalings[qubit.name]
            if ideal_sc == 1:
                # pi pulse amp calibration
                amp180 = self.analysis.proc_data_dict['analysis_params_dict'][
                    qubit.name]['correct_amplitude']
                qubit.set(f'{task["transition_name_input"]}_amp180', amp180)
            elif ideal_sc == 0.5:
                # pi/2 pulse amp calibration
                amp90_sc = self.analysis.proc_data_dict['analysis_params_dict'][
                    qubit.name]['correct_scalings_mean']
                qubit.set(f'{task["transition_name_input"]}_amp90_scale',
                          amp90_sc)
            else:
                log.info(f'No qubit parameter to update for a {ideal_sc}pi '
                         f'rotation. Update only possible for pi and pi/2.')


class NPulsePhaseErrorCalib(SingleQubitErrorAmplificationExperiment):
    """
    Phase-error calibration measurement using error amplification (see docstring
    of parent class).

    This is an error amplification experiment and a multitasking experiment;
    see docstrings of SingleQubitErrorAmplificationExperiment,
    MultiTaskingExperiment and of CalibBuilder for general information.
    Each task corresponds to one qubit specified by the key 'qb'
    (either name or QuDev_transmon instance), i.e. multiple qubits can be
    measured in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb:   [ |X180| --- |mX180| ] x n_repetitions  ---  |RO|
             sweep env_mod_frequency

    Expected sweep points, either global or per task:
        - env_mod_freqs or qscales in dimension 0
        - n_repetitions in dimension 1

    Idea behind this calibration measurement:
        If the envelope modulation frequency is perfectly calibrated, the qubit
        will remain in the ground state independent of n_repetitions.
        Sweeping the envelope modulation frequency will result in an oscillation
        between the ground and excited states, with the frequency of this
        oscillation dependent on n_repetitions (faster oscillations at larger
        n_repetitions).
        More about this measurement can be read here:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501

    Args (in addition to the ones accepted by the parent classes)
        env_mod_freqs (list or array): specifying the values for the envelope
            modulation frequency of the SSB_DRAG_pulse_full or
            SSB_DRAG_pulse_cos.
        qscales (list or array): specifying the values for the quadrature
            scaling factor of the SSB_DRAG_pulse.

    Keyword Args (in addition to the ones accepted by the parent classes)
        fit_indices (dict): with mobj names as keys and ints as values. Used in
            update and only relevant if len(n_repetitions) > 1, in which case
            there are several traces from which the fitted pulse parameters
            could be taken. The ints in this dict specify the index of the fit
            from which to update the pulse parameter.
    """

    default_experiment_name = 'NPulsePhaseErrorCalib'
    kw_for_sweep_points = {
        'env_mod_freqs': dict(param_name='env_mod_frequency', unit='Hz',
                              label='Envelope Modulation frequency, $f$',
                              dimension=0),
        'qscales': dict(param_name='motzoi', unit='',
                        label='Quadrature Scaling Factor, $q$',
                        dimension=0),
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 env_mod_freqs=None, qscales=None, **kw):
        try:
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             env_mod_freqs=env_mod_freqs,
                             qscales=qscales,
                             n_repetitions_sweep_dim=1,
                             amplification_pulses=['X180', 'mX180'],
                             analysis_class=tda.NPulsePhaseErrorCalibAnalysis,
                             **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis class
        :param kw: keyword arguments
            Passed to parent method.
        """
        data_to_fit = {mobjn: 'pg' for mobjn in self.meas_obj_names}
        analysis_kwargs = {'options_dict': {'data_to_fit': data_to_fit}}
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)

    def run_update(self, **kw):
        """
        Updates the tr_name_motzoi parameter or the tr_name_env_mod_freq
        parameter of the qubit in each task with the value extracted by the
        analysis.

        Keyword Args
            Not used, but are here to allow pass-through.
        """

        fit_indices = kw.get('fit_indices', None)
        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task['qb']][0]
            apd = self.analysis.proc_data_dict['analysis_params_dict']
            key = [k for k in apd if qubit.name in k]
            if len(key) == 1:
                key = key[0]
            else:
                if fit_indices is None:
                    log.warning('More than one fit result was found. Please '
                                'provide "fit_indices." Update was not run.')
                    return
                else:
                    key = key[fit_indices[qubit.name]]

            pulse_par = apd[key]['piPulse']
            if self.analysis.pulse_par_name == 'motzoi':
                qubit.set(f'{task["transition_name_input"]}_motzoi', pulse_par)
            else:
                qubit.set(f'{task["transition_name_input"]}_env_mod_freq',
                          pulse_par)


class NPulseLeakagePulseDelayCalib(SingleQubitErrorAmplificationExperiment):
    """
    Calibration measurement for the separation between a train of X180 pulses
    that results in coherent accumulation of leakage into the f state.

    This is an error amplification experiment and a multitasking experiment;
    see docstrings of SingleQubitErrorAmplificationExperiment,
    MultiTaskingExperiment and of CalibBuilder for general information. Each
    task corresponds to one qubit specified by the key 'qb' (either name or
    QuDev_transmon instance), i.e. multiple qubits can be measured in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb:   |X180| x n_repetitions  ---  |RO|
                sweep pulse_delay

    Expected sweep points, either global or per task:
        - pulse_delays in dimension 0
        - n_repetitions in dimension 1

    Idea behind this calibration measurement:
        Simply applying a train of X180 pulses played back to back will not
        necessarily result in leakage accumulation, because, depending on the
        length of the ge pulse during which the ef transition is driven outside
        the frame of reference set by the LO in which the ge transition was
        calibration, the leakage can accumulate in an incoherent manner,
        result in a destructive-interference effect. In this measurement, we
        find the pulse separation at which the f-state population is maximal.

    Args (in addition to the ones accepted by the parent classes)
        pulse_delays (list or array): delay of an X180 pulses with respect to
            the previous one.

    Keyword Args
        See parent classes.
    """

    default_experiment_name = 'NPulseLeakagePulseDelayCalib'
    kw_for_sweep_points = {
        'pulse_delays': dict(param_name='pulse_delay', unit='s',
                             label='Drive pulse separation, $\\tau$',
                             dimension=0),
    }

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 pulse_delays=None, **kw):
        try:
            if 'cal_states' not in kw:
                kw['cal_states'] = 'gef'

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             pulse_delays=pulse_delays,
                             n_repetitions_sweep_dim=1,
                             amplification_pulses=['X180'], **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()


class NPulseLeakageCalib(SingleQubitErrorAmplificationExperiment):
    """
    Leakage calibration measurement using error amplification (see docstring
    of parent class).

    This is an error amplification experiment and a multitasking experiment;
    see docstrings of SingleQubitErrorAmplificationExperiment,
    MultiTaskingExperiment and of CalibBuilder for general information. Each
    task corresponds to one qubit specified by the key 'qb' (either name or
    QuDev_transmon instance), i.e. multiple qubits can be measured in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qb:   |X180| x n_repetitions  ---  |RO|
        sweep cancellation_frequency_offset

    Expected sweep points, either global or per task:
        - cancellation_freq_offsets in dimension 0
        - n_repetitions in dimension 1

    Idea behind this calibration measurement:
        If the leakage is perfectly cancelled using the
        cancellation_frequency_offset parameter of the DRAG pulse, the f-state
        population will be zero, independent of n_repetitions.
        Sweeping the cancellation frequency offset at a pulse spacing that
        results in coherent leakage accumulation, will produce an oscillation
        in the f-state population with n_repetitions.

    Args (in addition to the ones accepted by the parent classes)
        cancellation_freq_offsets (list or array): frequencies offset from the
            ge transition frequency used to set the cancellation point of the
            SSB_DRAG_pulse_full or SSB_DRAG_pulse_cos.
        pulse_spacing (float): time separation between the X180 pulses in the
            amplification block. Defaults to 0.
            This parameter and its value will be copied to each task which does
            not already contain it.

    Keyword Args
        See parent classes.
    """

    default_experiment_name = 'NPulseLeakageCalib'
    kw_for_sweep_points = {
        'cancellation_freq_offsets': dict(
            param_name='cancellation_frequency_offset', unit='Hz',
            label='Cancellation freq. offset, $f_c$', dimension=0),
    }
    kw_for_task_keys = ['pulse_spacing']

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 cancellation_freq_offsets=None, pulse_spacing=0, **kw):
        try:
            if 'cal_states' not in kw:
                kw['cal_states'] = 'gef'
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points,
                             cancellation_freq_offsets=cancellation_freq_offsets,
                             pulse_spacing=pulse_spacing,
                             n_repetitions_sweep_dim=1,
                             amplification_pulses=['X180'],
                             pulse_modifs={'all': {'pulse_delay': pulse_spacing}},
                             **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()


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
    default_experiment_name = 'RabiFrequencySweep'

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
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

    def run_analysis(self, analysis_kwargs=None, **kw):
        """Run RabiAnalysis

        The RabiAnalysis will create individual Rabi fits for each of the
        qubit frequencies in the sweep, as well as 2D plots of raw and
        corrected data.

        FIXME: We currently do not call the RabiFrequencySweepAnalysis because
         the additional fitting steps in that class do not always work
         reliably yet.

        Args:
            analysis_kwargs: keyword arguments passed to the analysis class
            kw: passed through to super method
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 't_start' not in analysis_kwargs:
            analysis_kwargs['t_start'] = self.timestamp
        super().run_analysis(analysis_class=tda.RabiAnalysis,
                             analysis_kwargs=analysis_kwargs, **kw)


class ActiveReset(CalibBuilder):
    @handle_exception
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
        # second dimension to have once only readout and once with feedback
        default_sp.add_sweep_parameter("pulse_off", [1, 0])
        self.sweep_points = kw.get('sweep_points',
                                   default_sp)

        # get preparation parameters for all qubits. Note: in the future we could
        # possibly modify prep_params to be different for each uhf, as long as
        # the number of readout is the same for all UHFs in the experiment
        self.prep_params = deepcopy(self.get_reset_params())
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
        self.autorun(**kw)

    # FIXME: temporary solution to overwrite base method until the question of
    #  defining whether or not self.prepare_measurement should be used in the
    #  general case.
    def autorun(self, **kw):
        if self.measure:
            self.prepare_measurement(**kw)
        super().autorun(**kw)

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
            [gen.upload_classif_thresholds(qb) for qb in self.qubits]
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
        reset_block = self.reset(block_name="reset_ro_and_feedback_pulses",
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
            chs = {i: ch[1] for i, ch in enumerate(
                qb.get_acq_int_channels(2 if all_qb_channels else None))}

            #get clf thresholds
            if from_clf_params:
                thresh_qb = deepcopy(
                    qb.acq_classifier_params().get("thresholds", {}))
                thresholds[qb.name] = {u: thr for u, thr in thresh_qb.items()
                                       if u in chs}
            # get UHF thresholds
            else:
                thresholds[qb.name] = \
                    {u: qb.instr_acq.get_instr()
                          .get(f'qas_0_thresholds_{ch}_level')
                     for u, ch in chs.items()}

        return thresholds


class f0g1AcStark(SingleQubitGateCalibExperiment):
    """
    Class for the Ac Stark shift calibration measurement for f0g1 transition:
    gets the Ac Stark shift for all drive amplitudes.

    This calibration is based on and explained in the section 5.3
    of Dr. Philipp Kurpiers PhD Thesis, 2019
    (see Q:\PaperArchive\_Theses and Papers\QuDev\PhD\2019)

    Args:
        qubits (list): array of qubits for which the calibration is done
        amp (np.array): array of values for amplitudes of the pulse that are
            going to be swept (dimension 1). Recall that in PycQED the
            dimension of this array is volts (i.e., volts peak, Vp).
        transitionWidthCoefs (np.array): coefficients of the polynomial
            [c0, c1, ...] that determine the width of the range of frequencies
            to sweep. For each amplitude the frequency width is calculated as:
                (c0*amp + c1*amp + c2*amp^2)
        freqPointsPerAmp (float): number of frequency points to sweep.
        length_per_volt (float):
            In this calibration both the pulse amplitude and the pulse length
            are swept such as the product of the two is kept constant.
            The value of the pulse length is therefore inversely proportional
            to the value of the pulse amplitude; It is calculated as follows:
                              length * amplitude = length_per_volt
                        =>                length = length_per_volt / amplitude

    optional args:
        fit_degree (int): degree of the polynomial for the fitting.
            If fit_degree = i, then is going to fit an even polynomial of
            ith degree:  c0 + c2 x^2 +...+ ci x^i. Default value is 4.
        fit_threshold (float): to do the fittings all population values
            above this threshold will be ignored. Default value is 0.
            One can also specify an array of floats, hence fit_threshold
            will be different for each amplitude as per array specified
        update (boolean): if True, the 'f0g1_AcStark_IFCoefs' and
            'f0g1_AcStark_IFCoefs_error' of the qubit object are going to be updated
            after the fitting. If False (default value), nothing will happen.
    """

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys
    kw_for_sweep_points = {
        "freq_i": dict(
            param_name="mod_frequency", unit="Hz", label="Pulse frequency", dimension=0
        ),
        "leng": dict(
            param_name="pulse_length", unit="s", label="Pulse length", dimension=1
        ),
        "amp": dict(param_name="amplitude", unit="V", label="Amplitude", dimension=1),
    }
    default_experiment_name = "f0g1AcStark"
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        kw[
            "transition_name"
        ] = "ef"  # we use 'ef' transition name so PycQED know that has to measure
        # populations for g, e and f state
        # this way PycQED creates a X180_ge pulse automatically too

        # if values are not given use the default ones
        if not "fit_degree" in kw:
            kw["fit_degree"] = 4
        if not "fit_threshold" in kw:
            kw["fit_threshold"] = 0
        if type(kw["fit_threshold"]) == int or type(kw["fit_threshold"]) == float:
            kw["fit_threshold"] = np.zeros_like(kw["amp"]) + kw["fit_threshold"]
        # to be sure that we have a np.array and not a python list we do:
        kw["fit_threshold"] = np.array(kw["fit_threshold"])

        # if values are not given use the default ones, however these values should be given, print a message if
        # some value is not given
        if not "length_per_volt" in kw:
            kw["length_per_volt"] = 50e-9
            print(
                "length_per_volt not specified, using default value: length_per_volt = 50e-9"
            )
        if not "freqPointsPerAmp" in kw:
            kw["freqPointsPerAmp"] = 20
            print(
                "freqPointsPerAmp not specified, using default value: freqPointsPerAmp = 20"
            )
        if not "transitionWidthCoefs" in kw:
            kw["transitionWidthCoefs"] = np.array([50e6, 50e6])
            print(
                "transitionWidthCoefs not specified, using default: transitionWidthCoefs = np.array([25e6, 100e6])"
            )

        # length of the pulse is usually not given, but calculated with 'length_per_volt'
        if not "leng" in kw:
            kw["leng"] = kw["length_per_volt"] / kw["amp"]

        kw["freqPointsPerAmp"] = int(
            kw["freqPointsPerAmp"]
        )  # we say that it has to be an integer
        # we create a list that is not going to be used but needed for PycQED to have a list in the sweeping parm
        kw["freq_i"] = np.arange(kw["freqPointsPerAmp"])

        # now we create two dictionaries
        # 'frequencies': to save the actual frequencies that are going to be swept each amplitude value will have a
        #                different range of frequencies
        # 'IFCoefs': to save the 'f0g1_AcStark_IFCoefs' of each qubit
        kw["frequencies"] = odict()
        kw["IFCoefs"] = odict()
        for qb in qubits:  # we loop for the qubits
            array1D = np.array([])  # will use this array to append all the frequencies
            kw["IFCoefs"][qb.name] = qb.f0g1_AcStark_IFCoefs()
            for amp in kw["amp"]:  # we loop for the amplitudes
                middle_point = np.polyval(
                    np.flip(kw["IFCoefs"][qb.name]), amp
                )  # calculate middle of frequency range for this amplitude
                width = np.polyval(
                    np.flip(kw["transitionWidthCoefs"]), amp
                )  # calculate width of frequency range for this amplitude
                array1D = np.append(
                    array1D,  # append in the array the frequency points for this amplitude
                    np.linspace(
                        middle_point - width / 2,
                        middle_point + width / 2,
                        kw["freqPointsPerAmp"],
                    ),
                )
            kw["frequencies"][qb.name] = array1D.reshape(
                kw["amp"].size, kw["freqPointsPerAmp"]
            )  # we reshape the array
            # so to have a row for each amplitude

        self.frequencies = kw["frequencies"]  # create a variable for frequencies

        try:
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        # in this case we have specified in the 'SingleQubitGateCalibExperiment' class that we want to modify
        # each point manually. Meaning that 'sp1d_idx' is going to count the points in the 0 dimension and
        # 'sp2d_idx' the 1 dimension.
        # We do that because for this experiment the pulse sequence is not identical for each amplitude of the pulse
        # (amplitude is the swept variable in dimension 1): the frequency range changes for each amplitude
        # (frequency is the swept variable in dimension 0)
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task["sweep_points"]
            qb = task["qb"]

            prepend_blocks = super().sweep_block(
                **task
            )  # prepend blocks needed (super function)

            # for the flattop_f0g1 pulse we want our state to be 'f', as we have put 'ef' as transition name then
            # PycQED creates a X180_ge pulse automatically, then we need a X180_ef pulse to populate the f state
            # and once we are in f state we apply the flattop_f0g1 pulse
            # so we create the block of pulses that is going to do that for each point
            AcStark_block = self.block_from_ops(
                f"AcStark_pulses_{qb}", [f"X180_ef {qb}", f"flattop_f0g1 {qb}"]
            )

            # we specify the value for pulse frequency length, and amplitude we are going to use for this point
            AcStark_block.pulses[1]["mod_frequency"] = self.frequencies[qb][sp2d_idx][
                sp1d_idx
            ]  # here the frequency
            AcStark_block.pulses[1][
                "pulse_length"
            ] = sweep_points.get_sweep_params_property("values", 1, "pulse_length")[
                sp2d_idx
            ]  # here the length
            AcStark_block.pulses[1][
                "amplitude"
            ] = sweep_points.get_sweep_params_property("values", 1, "amplitude")[
                sp2d_idx
            ]  # here the amplitude

            parallel_block_list += [
                self.sequential_blocks(
                    f"flattop_f0g1_{qb}", prepend_blocks + [AcStark_block]
                )
            ]

        # return the blocks
        return self.simultaneous_blocks(
            f"flattop_f0g1_{sp2d_idx}_{sp1d_idx}",
            parallel_block_list,
            block_align="end",
        )

    def run_analysis(self, analysis_kwargs=None, **kw):
        # here we run the analysis

        # first we call the super function
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        # then we call the class defined for this analysis: 'f0g1AcStarkAnalysis'
        self.analysis = tda.f0g1AcStarkAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp, **analysis_kwargs
        )

    def run_update(self, **kw):
        # here we update the values found: 'f0g1_AcStark_IFCoefs' and 'f0g1_AcStark_IFCoefs_error'
        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task["qb"]][0]
            IFCoefs = np.array(
                list(self.analysis.proc_data_dict["IFCoefs"][qubit.name].values())
            )
            IFCoefs_error = np.array(
                list(self.analysis.proc_data_dict["IFCoefs_error"][qubit.name].values())
            )

            qubit.set("f0g1_AcStark_IFCoefs", IFCoefs)  # update f0g1_AcStark_IFCoefs
            qubit.set(
                "f0g1_AcStark_IFCoefs_error", IFCoefs_error
            )  # update f0g1_AcStark_IFCoefs_error

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d["sweeping_parameters"].update(
            {
                f0g1AcStark.__name__: {
                    0: {
                        "frequency": "Hz",
                    },
                    1: {
                        "length": "s",
                    },
                    1: {
                        "amplitude": "V",
                    },
                }
            }
        )
        return d


class f0g1RabiRate(SingleQubitGateCalibExperiment):
    """
    Class for the Rabi rate calibration measurement for f0g1 transition:
    gets the f0g1 transition speed (gTilde) for all drive amplitudes.

    This calibration is based on and explained in the section 5.3
    of Dr. Philipp Kurpiers PhD Thesis, 2019
    (see Q:\PaperArchive\_Theses and Papers\QuDev\PhD\2019)

    args:
        qubits (list): array of qubits for which the calibration is done
        amp (np.array): array of values for amplitudes of the pulse that are
            going to be swept (dimension 1, outer sweep dimension).
        max_len_per_volt (float): the inner sweep range (pulse lengths) is
            computed dynamically per pulse amplitude. max_len_per_volt
            determines the maximum pulse length for given amplitude
            via the expression max_len = max_len_per_volt / amplitude
        lengPointsPerAmp (int): gives the number of values for the pulse length
            Values swept are np.linspace(0, max_len, lengPointsPerAmp)

    optional args:
        fit_kappa (float): We are fitting the f state population via the
            damped oscillations model. Kappa could be fixed for that fit by
            specifying this parameter; fit_kappa is the value to be used.
            If fit_kappa is not specified, or given to be 0/False, then
            kappa will also be used as a parameter to be optimised.
        fit_degree (int): degree of the polynomial for the fitting
            (default value is 3, max 5). If fit_degree = i, then we are going to
             fit an odd polynomial of ith degree: c1 x^1 + c3 x^3 + ... + ci x^i.
        update (boolean): if True, the 'f0g1_RabiRate_Coefs' and
            'f0g1_RabiRate_Coefs_error' of the qubit object are going to be updated
            after the fitting. If False (default value), nothing will happen.
    """

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys
    kw_for_sweep_points = {  # we define the parameters that we want to sweep
        "leng_i": dict(
            param_name="pulse_length", unit="s", label="Pulse length", dimension=0
        ),
        "amp": dict(param_name="amplitude", unit="V", label="Amplitude", dimension=1),
        "freq_i": dict(
            param_name="mod_frequency", unit="Hz", label="Frequency", dimension=1
        ),
    }
    default_experiment_name = "f0g1RabiRate"
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        # we create two lists that are not going to be used, but are needed
        # for PycQED to have a list in the sweeping parm
        kw["freq_i"] = np.arange(kw["amp"].size)
        kw["leng_i"] = np.arange(kw["lengPointsPerAmp"])

        kw["transition_name"] = "ef"  # we use 'ef' transition name so
        # PycQED knows that it has to measure populations for g, e and f states
        # This way PycQED creates a X180_ge pulse automatically too

        if not "fit_degree" in kw:
            kw["fit_degree"] = 3
        if not "fit_kappa" in kw:
            kw["fit_kappa"] = 0

        # frequencies of the pulses are calculated with the
        # f0g1_AcStark_IFCoefs of the qubits
        kw["freq"] = {}
        self.frequencies = {}
        for qb in qubits:  # we do it for all qubits
            kw["freq"][qb.name] = np.polyval(
                np.flip(qb.f0g1_AcStark_IFCoefs()), kw["amp"]
            )
            self.frequencies[qb.name] = kw["freq"][qb.name]

        # we add in the metadata all the parameters of the qubit
        parameters = [
            "f0g1_kappa",
            "f0g1_RabiRate_Coefs",
            "T1_ef",
        ]  # list of parameters
        for param in parameters:
            kw[param] = odict()  # for each we create a dict
        for qb in qubits:  # for each qubit
            for param in parameters:
                kw[param][qb.name] = qb.get(f"{param}")  # we put each param in its dict

        # here we calculate the pulse lengths that are going to be used
        kw["lengths"] = odict()  # Variable to store the lengths

        for qb in qubits:
            mlpv = kw["max_len_per_volt"]
            lppa = kw["lengPointsPerAmp"]
            # 2D array of lengths, each line -- lengths from 0 to max_len
            # so that max_len * amplitude product is constant (hence mlpv/amp)
            length_array2D = np.array(
                [np.linspace(0, mlpv / amp, lppa) for amp in kw["amp"]]
            )
            kw["lengths"][qb.name] = length_array2D

        self.lengths = kw["lengths"]  # create a variable for lengths

        try:
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        # in this case we have specified in the 'SingleQubitGateCalibExperiment' class that we want to modify
        # each point manually. Meaning that 'sp1d_idx' is going to count the points in the 0 dimension and
        # 'sp2d_idx' the 1 dimension.
        # We do that because for this experiment the pulse sequence is not identical for each amplitude of the pulse
        # (amplitude and frequency are swept variable in dimension 1)
        # (length is the swept variable in dimension 0)
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task["sweep_points"]
            qb = task["qb"]

            prepend_blocks = super().sweep_block(
                **task
            )  # prepend blocks needed (super function)

            # for the flattop_f0g1 pulse we want our state to be 'f', as we have put 'ef' as transition name then
            # PycQED creates a X180_ge pulse automatically, then we need a X180_ef pulse to populate the f state
            # and once we are in f state we apply the flattop_f0g1 pulse
            # so we create the block of pulses that is going to do that for each point
            block = self.block_from_ops(
                f"rabi_pulses_{qb}", [f"X180_ef {qb}", f"flattop_f0g1 {qb}"]
            )

            # we specify the value for pulse frequency length, and amplitude we are going to use for this point
            block.pulses[1]["pulse_length"] = self.lengths[qb][sp2d_idx][
                sp1d_idx
            ]  # here the length (dim 0)
            block.pulses[1]["amplitude"] = sweep_points.get_sweep_params_property(
                "values", 1, "amplitude"
            )[sp2d_idx]  # here the amplitude (dim 1)
            block.pulses[1]["mod_frequency"] = self.frequencies[qb][
                sp2d_idx
            ]  # here de frequency (dim 1)

            parallel_block_list += [
                self.sequential_blocks(f"flattop_f0g1_{qb}", prepend_blocks + [block])
            ]

        # return the blocks
        return self.simultaneous_blocks(
            f"flattop_f0g1_{sp2d_idx}_{sp1d_idx}",
            parallel_block_list,
            block_align="end",
        )

    def run_analysis(self, analysis_kwargs=None, **kw):
        # here we run the analysis

        # first we call the super function
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        # then we call the class defined for this analysis: 'f0g1RabiRateAnalysis'
        self.analysis = tda.f0g1RabiRateAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp, **analysis_kwargs
        )

    def run_update(self, **kw):
        # here we update the values found: 'f0g1_RabiRate_Coefs', 'f0g1_RabiRate_Coefs_error' and 'f0g1_kappa'
        for task in self.preprocessed_task_list:
            qubit = [qb for qb in self.meas_objs if qb.name == task["qb"]][0]
            Coefs = np.array(self.analysis.proc_data_dict["Coefs"][qubit.name])
            Coefs_error = np.array(
                self.analysis.proc_data_dict["Coefs_error"][qubit.name]
            )
            kappa = self.analysis.proc_data_dict["kappa"][qubit.name]
            kappa_error = self.analysis.proc_data_dict["kappa_error"][qubit.name]

            # Updating the parameters of the qubit object
            qubit.set("f0g1_RabiRate_Coefs", Coefs)
            qubit.set("f0g1_RabiRate_Coefs_error", Coefs_error)
            qubit.set("f0g1_kappa", kappa)
            qubit.set("f0g1_kappa_error", kappa_error)
            qubit.set("f0g1_catch_kappa", kappa)
            qubit.set("f0g1_catch_kappa_error", kappa_error)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d["sweeping_parameters"].update(
            {
                f0g1RabiRate.__name__: {
                    0: {
                        "length": "s",
                    },
                    1: {
                        "amplitude": "V",
                    },
                    1: {
                        "mod_frequency": "Hz",
                    },
                }
            }
        )
        return d


class efWithf0g1AcStark(SingleQubitGateCalibExperiment):
    """
    Class for the Ac Stark shift calibration measurement for ef transition
    driven by a f0g1 pulse:
    gets the Ac Stark shift for all drive amplitudes.

    This calibration is based on and explained in the section 5.3
    of Dr. Philipp Kurpiers PhD Thesis, 2019
    (see Q:\PaperArchive\_Theses and Papers\QuDev\PhD\2019)

    Args:
        :param qubits: (list) array of qubits for which the calibration is done
        :param amp: (np.array) array of values for amplitudes of the f0g1 pulse that
        are going to be swept (dimension 1). Recall that in PycQED the
            dimension of this array is volts (i.e., volts peak, Vp).
        :param width_per_volt: (float) width of the frequency points to sweep for 1V
            of f0g1 drive. Default=80 MHz.
        :param freqPointsPerAmp: (float) number of ef frequency points to sweep.
        :param length_per_volt: (float)
            In this calibration both the f0g1 amplitude and the pulses length
            are swept such as the product of the two is kept constant.
            The value of the pulse length is therefore inversely proportional
            to the value of the pulse amplitude; It is calculated as follows:
                              length * amplitude = length_per_volt
                        =>                length = length_per_volt / amplitude

    :param kw: keyword arguments:
        fit_degree: (int) degree of the polynomial for the fitting.
            If fit_degree = i, then is going to fit an even polynomial of
            ith degree:  c0 + c2 x^2 +...+ ci x^i. Default value is 4.
        fit_threshold: (float) to do the fittings all population values
            above this threshold will be ignored. Default value is 0.
        update: (boolean) if True, the 'ef_for_f0g1_reset_pulse_AcStark_IFCoefs'
            and 'ef_for_f0g1_reset_pulse_AcStark_IFCoefs_error' of the qubit
            object are going to be updated after the fitting. If False (default
            value), nothing will happen.
    """

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys
    kw_for_sweep_points = {
        "freq_i": dict(
            param_name="mod_frequency", unit="Hz", label="Pulse frequency", dimension=0
        ),
        "leng": dict(
            param_name="pulse_length", unit="s", label="Pulse length", dimension=1
        ),
        "amp": dict(param_name="amplitude", unit="V", label="Amplitude", dimension=1),
    }
    default_experiment_name = "efAcStark"
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        kw[
            "transition_name"
        ] = "ef"  # we use 'ef' transition name so PycQED know that has to measure
        # populations for g, e and f state
        # this way PycQED creates a X180_ge pulse automatically too

        # if values are not given use the default ones
        if not "fit_degree" in kw:
            kw["fit_degree"] = 4
        if not "fit_threshold" in kw:
            kw["fit_threshold"] = 0

        # if values are not given use the default ones, however these values should be given, print a message if
        # some value is not given
        if not "length_per_volt" in kw:
            kw["length_per_volt"] = 50e-9
            print(
                "length_per_volt not specified, using default value: length_per_volt = 50e-9"
            )
        if not "freqPointsPerAmp" in kw:
            kw["freqPointsPerAmp"] = 20
            print(
                "freqPointsPerAmp not specified, using default value: freqPointsPerAmp = 20"
            )
        if not "width_per_volt" in kw:
            kw["width_per_volt"] = 80e6
            print(
                "transitionWidthCoefs not specified, using default: transitionWidthCoefs = 80e6"
            )

        # length of the pulse is usually not given, but calculated with 'length_per_volt'
        if not "leng" in kw:
            kw["leng"] = kw["length_per_volt"] / kw["amp"]

        kw["freqPointsPerAmp"] = int(
            kw["freqPointsPerAmp"]
        )  # we say that it has to be an integer
        # we create a list that is not going to be used but needed for PycQED to have a list in the sweeping parm
        kw["freq_i"] = np.arange(kw["freqPointsPerAmp"])

        # now we create two dictionaries
        # 'frequencies': to save the actual frequencies that are going to be swept each amplitude value will have a
        #                different range of frequencies
        # 'IFCoefs': to save the 'ef_AcStark_IFCoefs' of each qubit
        kw["frequencies"] = odict()
        kw["IFCoefs"] = odict()
        self.f0g1_IFCoefs = {}
        self.ef_for_f0g1_reset_pulse_amplitude = {}

        for qb in qubits:  # we loop for the qubits
            array1D = np.array([])  # will use this array to append all the frequencies
            kw["IFCoefs"][qb.name] = qb.ef_for_f0g1_reset_pulse_AcStark_IFCoefs()
            self.f0g1_IFCoefs[qb.name] = qb.f0g1_AcStark_IFCoefs()
            self.ef_for_f0g1_reset_pulse_amplitude[
                qb.name
            ] = qb.ef_for_f0g1_reset_pulse_amplitude()

            for amp in kw["amp"]:  # we loop for the amplitudes
                # calculate middle of frequency range for this amplitude
                middle_point = np.polyval(np.flip(kw["IFCoefs"][qb.name]), amp)
                width = amp * kw["width_per_volt"]
                array1D = np.append(
                    array1D,  # append in the array the frequency points for this amplitude
                    np.linspace(
                        middle_point - width,
                        middle_point + width,
                        kw["freqPointsPerAmp"],
                    ),
                )
            kw["frequencies"][qb.name] = array1D.reshape(
                kw["amp"].size, kw["freqPointsPerAmp"]
            )  # we reshape the array
            # so to have a row for each amplitude

        self.frequencies = kw["frequencies"]  # create a variable for frequencies

        try:
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        # in this case we have specified in the 'SingleQubitGateCalibExperiment' class that we want to modify
        # each point manually. Meaning that 'sp1d_idx' is going to count the points in the 0 dimension and
        # 'sp2d_idx' the 1 dimension.
        # We do that because for this experiment the pulse sequence is not identical for each amplitude of the pulse
        # (amplitude is the swept variable in dimension 1): the frequency range changes for each amplitude
        # (frequency is the swept variable in dimension 0)
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task["sweep_points"]
            qb = task["qb"]

            prepend_blocks = super().sweep_block(
                **task
            )  # prepend blocks needed (super function)

            # for the ef pulse we want our initial state to be 'e'
            # PycQED creates a X180_ge pulse automatically
            # once we are in e state we apply the flattop_f0g1 pulse and the ef
            # pulse simultaneously
            # so we create the block of pulses that is going to do that for
            # each f0g1 amplitude

            # we first create the simultaneous block of ef + f0g1 pulses
            block_ef = self.block_from_ops(
                f"ef180_{qb}", [f"ef_for_f0g1_reset_pulse {qb}"]
            )
            block_f0g1 = self.block_from_ops(
                f"f0g1_reset_pulse {qb}", [f"f0g1_reset_pulse {qb}"]
            )

            # we specify the values of length, frequency, amplitudes
            block_f0g1.pulses[0]["mod_frequency"] = np.polyval(
                np.flip(self.f0g1_IFCoefs[qb]),
                sweep_points.get_sweep_params_property("values", 1, "amplitude")[
                    sp2d_idx
                ],
            )
            block_ef.pulses[0]["mod_frequency"] = self.frequencies[qb][sp2d_idx][
                sp1d_idx
            ]
            block_f0g1.pulses[0][
                "pulse_length"
            ] = sweep_points.get_sweep_params_property("values", 1, "pulse_length")[
                sp2d_idx
            ]
            block_ef.pulses[0]["pulse_length"] = sweep_points.get_sweep_params_property(
                "values", 1, "pulse_length"
            )[sp2d_idx]
            block_f0g1.pulses[0]["amplitude"] = sweep_points.get_sweep_params_property(
                "values", 1, "amplitude"
            )[sp2d_idx]
            block_ef.pulses[0]["amplitude"] = self.ef_for_f0g1_reset_pulse_amplitude[qb]

            simu_blocks = self.simultaneous_blocks(
                f"ef_AcStark_pulses_{qb}", [block_ef, block_f0g1], block_align="end"
            )

            parallel_block_list += [
                self.sequential_blocks(
                    f"ef_AcStark_pulses_{qb}", prepend_blocks + [simu_blocks]
                )
            ]

        # return the blocks
        return self.simultaneous_blocks(
            f"ef_f0g1_AcStark_pulses_{sp2d_idx}_{sp1d_idx}",
            parallel_block_list,
            block_align="end",
        )

    def run_analysis(self, analysis_kwargs=None, **kw):
        # here we run the analysis

        # first we call the super function
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        # then we call the class defined for this analysis: 'efWithf0g1AcStarkAnalysis'
        self.analysis = tda.efWithf0g1AcStarkAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp, **analysis_kwargs
        )

    def run_update(self, **kw):
        try:
            # here we update the values found: 'f0g1_AcStark_IFCoefs' and 'f0g1_AcStark_IFCoefs_error'
            for task in self.preprocessed_task_list:
                qubit = [qb for qb in self.meas_objs if qb.name == task["qb"]][0]
                IFCoefs = np.array(
                    list(self.analysis.proc_data_dict["IFCoefs"][qubit.name].values())
                )
                IFCoefs_error = np.array(
                    list(
                        self.analysis.proc_data_dict["IFCoefs_error"][
                            qubit.name
                        ].values()
                    )
                )

                qubit.set(
                    "ef_for_f0g1_reset_pulse_AcStark_IFCoefs", IFCoefs
                )  # update f0g1_AcStark_IFCoefs
                qubit.set(
                    "ef_for_f0g1_reset_pulse_AcStark_IFCoefs_error", IFCoefs_error
                )  # update f0g1_AcStark_IFCoefs_error
        except:
            print("does not update IF Coefficients since no fitting occured")

    #
    @classmethod
    def gui_kwargs(cls, device):
        try:
            d = super().gui_kwargs(device)
            d["sweeping_parameters"].update(
                {
                    efWithf0g1AcStark.__name__: {
                        0: {
                            "frequency": "Hz",
                        },
                        1: {
                            "length": "s",
                        },
                        1: {
                            "amplitude": "V",
                        },
                    }
                }
            )
            return d
        except:
            print("")


class f0g1ResetRabiCalib(SingleQubitGateCalibExperiment):
    """
    Class for calibrating the Rabi rate of the ef transition while drivng with a f0g1, in order to perform reset

    This calibration is based on and explained in the section 5.3
    of Dr. Paul Magnard PhD Thesis, 2021

    :param qubits: list of qubits for which the calibration is done
    :param amp: array of values for amplitudes of the f0g1 pulse that are
        going to be swept (dimension 1, outer sweep dimension).

    :param kw: keyword arguments:
        max_len_per_volt: (float) if 'leng' is not provided, the inner
            sweep range (pulse lengths) is computed dynamically per pulse amplitude.
            max_len_per_volt determines the maximum pulse length for given amplitude
            via the expression max_len = max_len_per_volt / amplitude
        lengPointsPerAmp: (float) if 'leng' is not provided, the inner
            sweep range (pulse lengths) is computed dynamically per pulse amplitude.
            'lengPointsPerAmp' gives the number of values for the pulse length.
            Values swept are np.linspace(0, max_len, lengPointsPerAmp)
        amp_ef: (float) amplitude of the f0g1 pulse, default=ef_for_f0g1_reset_pulse_amplitude()
        freq_ef: (float) modulation frequency of the ef pulse, default=ef_for_f0g1_reset_pulse_mod_frequency()
        start_from_ef: (boolean) Equals false if the reset is calibrated
            starting from the e state (default value). Otherwise, the reset is
            calibrated from the f state.

    """

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys
    kw_for_sweep_points = {  # we define the parameters that we want to sweep
        "leng_i": dict(
            param_name="pulse_length", unit="s", label="Pulse length", dimension=0
        ),
        "amp": dict(param_name="amplitude", unit="V", label="Amplitude", dimension=1),
        "freq_i": dict(
            param_name="mod_frequency", unit="Hz", label="Frequency", dimension=1
        ),
    }
    default_experiment_name = "f0g1ResetRabiCalib"
    call_parallel_sweep = False  # pulse sequence changes between segments

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        # if leng is not given, kw['lengPointsPerAmp'] has to be an integer
        # if leng is given, then kw['lengPointsPerAmp'] is its number of points
        kw["lengPointsPerAmp"] = (
            int(kw["lengPointsPerAmp"]) if not "leng" in kw else kw["leng"].size
        )
        self.start_from_ef = False if not "start_from_ef" in kw else kw["start_from_ef"]

        # we create two lists that are not going to be used, but are needed
        # for PycQED to have a list in the sweeping parm
        kw["freq_i"] = np.arange(kw["amp"].size)
        kw["leng_i"] = np.arange(kw["lengPointsPerAmp"])

        kw["transition_name"] = "ef"  # we use 'ef' transition name so
        # PycQED knows that it has to measure populations for g, e and f states
        # This way PycQED creates a X180_ge pulse automatically too

        # frequencies of the pulses are calculated with the
        # f0g1_AcStark_IFCoefs of the qubits, and the ef frequency is deduced from previous calibration
        kw["freq_f0g1"] = {}
        kw["freq_ef"] = {}
        kw["amp_ef"] = {}

        self.freq_f0g1 = {}
        self.freq_ef = {}
        self.amp_ef = {}
        for qb in qubits:  # we do it for all qubits
            kw["freq_f0g1"][qb.name] = np.polyval(
                np.flip(qb.f0g1_AcStark_IFCoefs()), kw["amp"]
            )
            self.freq_f0g1[qb.name] = kw["freq_f0g1"][qb.name]
            kw["freq_ef"][qb.name] = np.polyval(
                np.flip(qb.ef_for_f0g1_reset_pulse_AcStark_IFCoefs()), kw["amp"]
            )
            self.freq_ef[qb.name] = kw["freq_ef"][qb.name]
            kw["amp_ef"][qb.name] = qb.ef_for_f0g1_reset_pulse_amplitude()
            self.amp_ef[qb.name] = kw["amp_ef"][qb.name]

        # we add in the metadata all the parameters of the qubit
        parameters = ["kappa", "gamma1", "RabiRate_Coefs"]  # list of parameters
        for param in parameters:
            kw[param] = {}  # for each we create a dict
        for qb in qubits:  # for each qubit
            for param in parameters:
                kw[param][qb.name] = qb.get(
                    f"f0g1_{param}"
                )  # we put each param in its dict

        # here we calculate the pulse lengths that are going to be used
        kw[
            "lengths"
        ] = odict()  # here we will put the pulse lengths that are going to be swept,
        # for each amplitude we can have a different range of pulse lengths
        for qb in qubits:  # we loop for the qubits
            array1D = np.array([])  # will use this array to append all the frequencies
            for amp in kw["amp"]:  # we loop for the amplitudes
                # calculate the max length keeping the
                # length * amplitude product constant
                max_len = kw["max_len_per_volt"] / amp
                array1D = np.append(
                    array1D,  # append in the array the frequency points for this amplitude
                    np.linspace(0, max_len, kw["lengPointsPerAmp"]),
                )

            # we reshape the array so to have a row for each amplitude
            kw["lengths"][qb.name] = array1D.reshape(
                kw["amp"].size, kw["lengPointsPerAmp"]
            )

        self.lengths = kw["lengths"]  # create a variable for lengths

        try:
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sp1d_idx, sp2d_idx, **kw):
        # in this case we have specified in the 'SingleQubitGateCalibExperiment' class that we want to modify
        # each point manually. Meaning that 'sp1d_idx' is going to count the points in the 0 dimension and
        # 'sp2d_idx' the 1 dimension.
        # We do that because for this experiment the pulse sequence is not identical for each amplitude of the pulse
        # (amplitude and frequency are swept variable in dimension 1)
        # (length is the swept variable in dimension 0)
        parallel_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            sweep_points = task["sweep_points"]
            qb = task["qb"]

            prepend_blocks = super().sweep_block(
                **task
            )  # prepend blocks needed (super function)

            # we first create the simultaneous block of ef + f0g1 pulses
            block_ef = self.block_from_ops(
                f"ef180_{qb}", [f"ef_for_f0g1_reset_pulse {qb}"]
            )
            block_f0g1 = self.block_from_ops(
                f"f0g1_reset_pulse {qb}", [f"f0g1_reset_pulse {qb}"]
            )

            # we specify the values of length, frequency, amplitudes
            block_f0g1.pulses[0]["pulse_length"] = self.lengths[qb][sp2d_idx][
                sp1d_idx
            ]  # here the length (dim 0)
            block_ef.pulses[0]["pulse_length"] = self.lengths[qb][sp2d_idx][
                sp1d_idx
            ]  # here the length (dim 0)
            block_f0g1.pulses[0]["amplitude"] = sweep_points.get_sweep_params_property(
                "values", 1, "amplitude"
            )[sp2d_idx]  # here the amplitude (dim 1)
            block_ef.pulses[0]["amplitude"] = self.amp_ef[qb]
            block_f0g1.pulses[0]["mod_frequency"] = self.freq_f0g1[qb][
                sp2d_idx
            ]  # here de frequency (dim 1)
            block_ef.pulses[0]["mod_frequency"] = self.freq_ef[qb][sp2d_idx]

            simu_blocks = self.simultaneous_blocks(
                f"reset_pulses_{qb}", [block_ef, block_f0g1], block_align="end"
            )

            # for the flattop_f0g1 pulse we want our state to be 'f', as we have put 'ef' as transition name then
            # PycQED creates a X180_ge pulse automatically, then we need a X180_ef pulse to populate the f state
            # and once we are in f state we apply the flattop_f0g1 pulse
            # so we create the block of pulses that is going to do that for each point, and add it to the simultaneous
            # block precedently created
            if self.start_from_ef:
                # adding an ef pulse at the start of the reset calibration
                block = self.sequential_blocks(
                    f"reset_calib_pulses_{qb}",
                    [
                        self.block_from_ops(f"ini_pulse_{qb}", [f"X180_ef {qb}"]),
                        simu_blocks,
                    ],
                    destroy=True,
                )
            else:
                block = simu_blocks

            parallel_block_list += [
                self.sequential_blocks(
                    f"f0g1_reset_calib_{qb}", prepend_blocks + [block]
                )
            ]

        # return the blocks
        return self.simultaneous_blocks(
            f"f0g1_reset_cal_{sp2d_idx}_{sp1d_idx}",
            parallel_block_list,
            block_align="end",
        )

    def run_analysis(self, analysis_kwargs=None, **kw):
        # here we run the analysis

        # first we call the super function
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        # then we call the class defined for this analysis: 'f0g1ResetCalibAnalysis'
        self.analysis = tda.f0g1ResetCalibAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp, **analysis_kwargs
        )


class f0g1Pitch(SingleQubitGateCalibExperiment):
    """
    class for the f0g1 pitch calibration: check whether the calibrations of AcStark and RabiRate work correctly

    this calibration is explained in 5.4 section of Dr. Philipp Kurpiers PhD Thesis, 2019

    args:
        qubits (list): array of qubits to which do the calibration
        gamma1 (np.array): array of values for gamma1 that are going to be swept (dimension 1)
            gamma1 is exponential rate of the rising edge of the emitted photon
        pulseTrunc (np.array): array of values for pulseTrunc that ara going to be swept (dimension 0)

    optional args:
        gamma2 (np.array): array of values for gamma2 that are going to be swept (dimension 1)
            gamma2 is exponential rate of the falling edge of the emitted photon
            if 'gamma2' is not given, then 'gamma2' = 'gamma1' will be used
        photonTrunc (float):
            pulseTrunc and photonTrunc dictates how to truncate the pulse. For pulseTrunc=1, the pulse is
            truncated at -photonTrunc*2/gamma1 and photonTrunc*2/gamma2. For pulseTrunc<1, the pulse is truncated
            such that it has the same start time, but a pulse length of pulseTrunc*pulseLength
        junctionTrunc (float):
        junctionSigma (float):
            information about the junction bridging the AWG amplitude
            from the truncated pulse value at the end and zero. These variables denote the junction truncation,
            width and type respectively
    """

    kw_for_task_keys = SingleQubitGateCalibExperiment.kw_for_task_keys
    kw_for_sweep_points = {  # we define the parameters that we want to sweep
        "pulseTrunc": dict(
            param_name="pulseTrunc",
            unit="-",  # ? not sure about the unit
            label="Pulse Truncation",
            dimension=0,
        ),
        "gamma1": dict(param_name="gamma1", unit="Hz", label="gamma1", dimension=1),
        "gamma2": dict(param_name="gamma2", unit="Hz", label="gamma2", dimension=1),
    }
    default_experiment_name = "f0g1Pitch"

    def __init__(self, task_list=None, sweep_points=None, qubits=None, **kw):
        kw[
            "transition_name"
        ] = "ef"  # we use 'ef' transition name so PycQED know that has to measure
        # populations for g, e and f state
        # this way PycQED creates a X180_ge pulse automatically too

        # -- we put the default values of the pulse for each qubit if no values are given when the object is created
        #   if values given then we change the default values
        #   this way all these values are going to be in the metadata of the experiment
        if not "photonTrunc" in kw:
            kw["photonTrunc"] = [qb.f0g1_photonTrunc() for qb in qubits]
        else:
            for qb in qubits:
                qb.f0g1_photonTrunc(kw["photonTrunc"])
            kw["photonTrunc"] = [kw["photonTrunc"] for _ in qubits]

        if not "junctionTrunc" in kw:
            kw["junctionTrunc"] = [qb.f0g1_junctionTrunc() for qb in qubits]
        else:
            for qb in qubits:
                qb.f0g1_junctionTrunc(kw["junctionTrunc"])
            kw["junctionTrunc"] = [kw["junctionTrunc"] for _ in qubits]

        if not "junctionSigma" in kw:
            kw["junctionSigma"] = [qb.f0g1_junctionSigma() for qb in qubits]
        else:
            for qb in qubits:
                qb.f0g1_junctionSigma(kw["junctionSigma"])
            kw["junctionSigma"] = [kw["junctionSigma"] for _ in qubits]
        # --

        # if the user only gives gamma or gamma1 we use the same array for gamma2
        if "gamma1" in kw and not "gamma2" in kw:
            kw["gamma2"] = kw["gamma1"]

        try:  # call the 'SingleQubitGateCalibExperiment' init
            super().__init__(task_list, qubits=qubits, sweep_points=sweep_points, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, transition_name, **kw):
        # here we specify the pulses that we want to apply and for which do we want to sweep its parameters

        prepend_blocks = super().sweep_block(
            qb, sweep_points, transition_name, **kw
        )  # prepend blocks needed (super function)

        # for the f0g1 pulse we want our state to be 'f', as we have put 'ef' as transition name then
        # PycQED creates a X180_ge pulse automatically, then we need a X180_ef pulse to populate the f state
        # and once we are in f state we apply the f0g1 pulse
        # so we create the block of pulses that is going to do that
        block = self.block_from_ops(
            f"f0g1_pulses_{qb}", [f"X180_ef {qb}", f"f0g1 {qb}"]
        )

        # we specify which parameters we want to sweep thanks to the 'sweep_points':
        # 'sweep_points' is an array of dictionaries (the length of the array is defining the dimensions of the sweep,
        # if the array has 2 components we seep in dimension 0 and dimension 1 (2D sweep).
        # each dictionary has as keys the names of the parameter swept  ('param_name') in that dimension
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                pulse_dict = block.pulses[1]  # we seep the f0g1 pulse
                if (
                    param_name in pulse_dict
                ):  # if the parameters are parameters of the f0g1 pulse
                    pulse_dict[param_name] = ParametricValue(
                        param_name
                    )  # we use the 'ParametricValue' function
                    # to be able to sweep that parameter

        # return the blocks
        return self.sequential_blocks(f"f0g1_pitch_{qb}", prepend_blocks + [block])

    def run_analysis(self, analysis_kwargs=None, **kw):
        # here we run the analysis

        # first we call the super function
        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}

        # then we call the class defined for this analysis: 'f0g1PitchAnalysis'
        self.analysis = tda.f0g1PitchAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp, **analysis_kwargs
        )

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d["sweeping_parameters"].update(
            {
                f0g1Pitch.__name__: {
                    0: {
                        "pulseTrunc": "-",
                    },
                    1: {
                        "gamma1": "Hz",
                        "gamma2": "Hz",
                    },
                }
            }
        )
        return d

class LeakageReductionUnit(SingleQubitGateCalibExperiment):
    """
    LRU measurement for finding the amplitude, frequency and pulse length of the LRU.
    This is a SingleQubitGateCalibExperiment, see docstring there for general information.

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun, and
        to the parent class.

    The following keys in a task are interpreted by this class in
    addition to the ones recognized by the parent classes:
        - amplitude
        - pulse_length
        - frequency
        - transition_name
    """

    kw_for_sweep_points = {
        'freqs': dict(param_name='frequency', unit='Hz',
                      label=r'modulation frequency',
                      dimension=1),
        'amps': dict(param_name='amplitude', unit='V',
                       label=r'modulation amplitude',
                       dimension=0),
    }
    kw_for_task_keys = ['num_LRUs']
    default_experiment_name = 'Leakage_reduction_unit'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 amps=None, length= None, **kw):
        try:
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             amps=amps, length=length, **kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, qb, sweep_points, transition_name, num_LRUs=1, **kw):
        """
        This function creates the blocks for the leakage-reduction task,
        see the pulse sequence in the class docstring.
        :param qb: qubit name
        :param sweep_points: SweepPoints instance
        :param transition_name: currently we use 'ef' to prepare the e-level
                                and 'ef' together with the kw argument
                                'prepare_f' to prepare the f-level
        :param kw: keyword arguments:
            - amplitude: (float) amplitude of PFM pulse
            - pulse_length: (float) length of PFM pulse
            - frequency: (float) frequency of PFM pulse
            - prepare_f: (bool) if True, the pulse sequence prepares the f-level
        """

        prepend_blocks = super().sweep_block(qb, sweep_points, transition_name,
                                             **kw)

        # Add ef pulse if specified; transition_name 'ef' only prepares the e-level
        ef_pulse = self.block_from_ops(f'ef_pulse',
                                       [f'X180_ef {qb}'])


        # add modulation pulse
        modulation_block = self.block_from_ops(f'modulation_pulse_{qb}',
                                               [f'PFM{transition_name} {qb}'] * num_LRUs,
                                               # pulse_modifs=pulse_modifs
                                               )
        # create ParametricValues from param_name in sweep_points
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in modulation_block.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        if kw.get('prepare_f', False):
            return self.sequential_blocks(f'leakage_reduction_unit_{qb}',
                                          prepend_blocks + [ef_pulse] + [
                                              modulation_block])
        elif kw.get('prepare_h', False):
            fh_pulse = self.block_from_ops(f'fh_pulse',
                                           [f'X180_fh {qb}'])
            if kw.get('deplete_f', False):
                flux_pulse_amplitude_0 = kw.get('flux_pulse_amplitude_0', 0)
                flux_pulse_length_0 = kw.get('flux_pulse_length_0', 4e-8)
                flux_pulse_frequency_0 = kw.get('flux_pulse_frequency_0', 0)
                pulse_modifs = {'all': {'amplitude': flux_pulse_amplitude_0,
                                    'pulse_length': flux_pulse_length_0,
                                    'frequency': flux_pulse_frequency_0}}
                modulation_block_0 = self.block_from_ops(f'modulation_pulse_0_{qb}',
                                               [f'PFM_ef {qb}'],
                                               pulse_modifs=pulse_modifs
                                               )
                return self.sequential_blocks(f'leakage_reduction_unit_{qb}',
                                              prepend_blocks + [fh_pulse] +
                                              [modulation_block_0] + [
                                                  modulation_block])

            return self.sequential_blocks(f'leakage_reduction_unit_{qb}',
                                          prepend_blocks + [fh_pulse] +
                                          [modulation_block])
        else:
            return self.sequential_blocks(f'leakage_reduction_unit_{qb}',
                                          prepend_blocks + [modulation_block])

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
        self.analysis = tda.LeakageReductionUnitAnalysis(
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)

class DriveAmplitudeNonlinearityCurve(CalibBuilder):
    """
    Calibration measurement for the drive amplitude non-linearity curve.
    This class runs DriveAmpCalib for several values of n_pulses_pi. See
    docstring of DriveAmpCalib for information about input parameters.

    Particular to this class: n_pulses_pi is a list or array of integers. If
    None, it will default to [2,1,3,4,5,6,7].
    The DriveAmpCalib will be run for 1/npp and 1-1/npp with npp in n_pulses_pi,
    except for npp in [1, 2].

    Important remarks:
        - n_pulses_pi will be sorted from lowest to highest, making sure that
         it starts with 2 (ex: [2,1,3,4,5..])
        - if 1 or 2 in n_pulses_pi, the corrected amp180 and amp90_scale will
        be set as temporary values for the remaining measurements.
        The above are done for two reasons:
            - DriveAmpCalib scales amplitudes with respect to amp180, and we
            want to use the calibrated amp180 for n_pulses_pi > 1
            - The first pulse in the DriveAmpCalib sequence is an X90, so we
            start by calibrating the amp90_scale and use it for the remaining
            measurements.

    Keyword args particular to this class:
        - run_complement (bool; default: True): whether to run DriveAmpCalib for
            1-1/npp (True) or only for 1/npp (False).
    """
    default_experiment_name = 'DriveAmpNonlinearityCurve'

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 n_repetitions=None, n_pulses_pi=None,  **kw):

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

            self.n_repetitions = n_repetitions
            self.n_pulses_pi = n_pulses_pi
            self.init_kwargs = {}  # for passing to the DriveAmpCalib msmts
            self.init_kwargs.update(kw)
            self.init_kwargs['update'] = False  # never update in DriveAmpCalib

            # we measure the non-linearity of the control electronics; it
            # doesn't matter which quantum transition we use for it, so we use
            # the lowest
            self.init_kwargs['transition_name'] = 'ge'

            if self.n_pulses_pi is None:
                # the pi pulse amplitude is assumed to be calibrated and its
                # correction is not part of the calibration curve
                # (see DriveAmpNonlinearityCurveAnalysis)
                self.n_pulses_pi = np.arange(2, 8)
            # sort lowest to highest: see docstring for reason
            self.n_pulses_pi = np.sort(self.n_pulses_pi)
            try:
                # If 2 in n_pulses_pi, we want to start by calibrating the
                # amp90_scale and use it for the remaining measurements since
                # the pulse sequence in the DriveAmpCalib experiment starts
                # with X90.
                idx = list(self.n_pulses_pi).index(2)
                if idx == 1:
                    # means first two entries are 1, 2: flip them
                    self.n_pulses_pi = np.concatenate([
                        [2, 1], self.n_pulses_pi[idx+1:]])
            except ValueError:
                # 2 is not in self.n_pulses_pi
                pass
            self.measurements = []  # for collecting instance of DriveAmpCalib
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def run_measurement(self, **kw):
        """
        Overwrites the base method to run a DriveAmpCalib experiment for each
        entry in self.n_pulses_pi. See class docstring for more details.
        """
        run_complement = kw.get('run_complement', True)

        # get qubits for setting temporary values
        qb_in_exp = self.find_qubits_in_tasks(self.qubits, self.task_list)
        temp_vals = []

        # analysis_kwargs to be passed to the DriveAmpCalib measurements
        ana_kw = self.init_kwargs.pop('analysis_kwargs_da_calib', {})
        opt_dict = ana_kw.pop('options_dict', {})
        for i, npp in enumerate(self.n_pulses_pi):
            if npp in [1, 2]:
                with temporary_value(*temp_vals):
                    od = {}
                    if npp == 1 and 'fit_t2_r' not in opt_dict:
                        # do not fit T2 in this case
                        od['fit_t2_r'] = False
                    od.update(opt_dict)
                    analysis_kwargs = {'options_dict': od}
                    analysis_kwargs.update(ana_kw)
                    DACalib = NPulseAmplitudeCalib(
                        task_list=self.task_list,
                        sweep_points=self.sweep_points,
                        qubits=self.qubits,
                        n_repetitions=self.n_repetitions,
                        n_pulses_pi=npp,
                        analysis_kwargs=analysis_kwargs,
                        **self.init_kwargs)
                self.measurements += [DACalib]

                # set the corrections from this measurement as temporary values
                # for the next measurements
                if npp == 1:
                    temp_vals = []
                    for qb in qb_in_exp:
                        amp180 = qb.ge_amp180()  # current amp180
                        # we need to adjust the amp90_scale as well since the
                        # previously calibrated value is with respect to the
                        # current amp180
                        amp90_sc = qb.ge_amp90_scale()  # current amp90_scale
                        amp90 = amp180 * amp90_sc  # calibrated amp90
                        # calibrated amp180
                        corr_amp180 = DACalib.analysis.proc_data_dict[
                                'analysis_params_dict'][qb.name][
                                'correct_scalings_mean'] * amp180
                        # adjust amp90_scale based on the calibrated amp180
                        corr_amp90_sc = amp90 / corr_amp180
                        temp_vals.extend([(qb.ge_amp180, corr_amp180),
                                          (qb.ge_amp90_scale, corr_amp90_sc)])
                else:
                    temp_vals = [
                        (qb.ge_amp90_scale, DACalib.analysis.proc_data_dict[
                            'analysis_params_dict'][qb.name][
                            'correct_scalings_mean'])
                        for qb in qb_in_exp]
            else:
                od = {}
                if 'fit_t2_r' not in opt_dict:
                    # do not fit T2 in this case
                    od['fit_t2_r'] = False
                od.update(opt_dict)
                analysis_kwargs = {'options_dict': od}
                analysis_kwargs.update(ana_kw)
                with temporary_value(*temp_vals):
                    # measure for 1/npp
                    DACalib = NPulseAmplitudeCalib(
                        task_list=self.task_list,
                        sweep_points=self.sweep_points,
                        qubits=self.qubits,
                        n_repetitions=self.n_repetitions,
                        n_pulses_pi=npp,
                        analysis_kwargs=analysis_kwargs,
                        **self.init_kwargs)
                    self.measurements += [DACalib]

                if run_complement:
                    # measure for 1 - 1/npp
                    tl = []
                    for j, task in enumerate(self.task_list):
                        # set the fixed_scaling to the calibrated value of
                        # 1/npp from the previous measurement
                        fixed_scaling = DACalib.analysis.proc_data_dict[
                            'analysis_params_dict'][task['qb']][
                            'correct_scalings_mean']
                        tl_dict = {'fixed_scaling': fixed_scaling}
                        tl_dict.update(task)
                        tl += [tl_dict]
                    with temporary_value(*temp_vals):
                        DACalib = NPulseAmplitudeCalib(
                            task_list=tl,
                            sweep_points=self.sweep_points,
                            qubits=self.qubits,
                            n_repetitions=self.n_repetitions,
                            n_pulses_pi=npp,
                            analysis_kwargs=ana_kw,
                            **self.init_kwargs)
                        self.measurements += [DACalib]

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs DriveAmpNonlinearityCurve and stores analysis instance in
        self.analysis.

        Args:
            analysis_kwargs (dict; default: None): keyword arguments for
                analysis class

        Keyword args:
            Passed to parent method.
        """

        super().run_analysis(analysis_kwargs=analysis_kwargs, **kw)
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = tda.DriveAmplitudeNonlinearityCurveAnalysis(
            qb_names=self.meas_obj_names,
            t_start=self.measurements[0].timestamp,
            t_stop=self.measurements[-1].timestamp,
            **analysis_kwargs)

    def run_update(self, **kw):
        """
        Updates the amp_scaling_correction_coeffs of the qubit in each task
        with the coefficients extracted by the analysis.

        Keyword args:
         to allow pass through kw even though they are not needed
        """

        for mobjn in self.meas_obj_names:
            qubit = [qb for qb in self.meas_objs if qb.name == mobjn][0]
            nl_fit_pars = self.analysis.proc_data_dict['nonlinearity_fit_pars'][
                qubit.name]
            qubit.set('amp_scaling_correction_coeffs',
                      [nl_fit_pars['a'], nl_fit_pars['b']])
