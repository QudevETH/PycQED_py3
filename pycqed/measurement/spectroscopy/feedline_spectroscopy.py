import numpy as np
from copy import copy
from copy import deepcopy
from pycqed.utilities.general import assert_not_none, configure_qubit_mux_readout, configure_qubit_mux_drive
from pycqed.measurement.calibration.two_qubit_gates import MultiTaskingExperiment
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.spectroscopy_analysis as spa
from pycqed.utilities.general import temporary_value
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
import logging
log = logging.getLogger(__name__)


class MultiTaskingSpectroscopyExperiment(MultiTaskingExperiment):
    """Adds functionality to sweep LO and modulation frequencies in
    a spectroscopy experiment. Automatically determines whether the LO, the
    mod. freq. or both are swept.

    Child classes implement the spectroscopy experiment and need to implement
    sweep_block and adjust_sweep_functions.

    Args:
        MultiTaskingExperiment (_type_): _description_
    """
    task_mobj_keys = ['qb']
    @assert_not_none('task_list')
    def __init__(self, task_list, allowed_lo_freqs=None,
                 trigger_separation=10e-6, **kw):
        # Passing keyword arguments to the super class (even if they are not
        # needed there) makes sure that they are stored in the metadata.
        df_name = kw.pop('df_name', 'int_avg_det_spec')
        self.df_kwargs = kw.pop('df_kwargs', dict())
        cal_states = kw.pop('cal_states', [])
        super().__init__(task_list, df_name=df_name, cal_states=cal_states,
                         **kw)
        self.sweep_functions_dict = kw.get('sweep_functions_dict', {})
        self.sweep_functions = []
        self.sweep_points_pulses = SweepPoints(min_length=2, )
        self.allowed_lo_freqs = allowed_lo_freqs
        self.analysis = {}

        self.trigger_separation = trigger_separation
        self.lo_task_dict = {}
        self.lo_frequencies = {}
        self.mod_frequencies = {}

        # Used to set the acquisition mode in the segment, e.g. for fast
        # SHFQA spectroscopy. Default is 'software' sweeper.
        self.segment_kwargs = {'acquisition_mode': dict(sweeper='software')}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)

        self.generate_lo_task_dict()
        # self.check_all_freqs_per_lo()
        self.resolve_freq_sweep_points(**kw)
        self.generate_sweep_functions()
        if len(self.sweep_points_pulses[0]) == 0:
            # Create a single segement if no hard sweep points are provided.
            self.sweep_points_pulses.add_sweep_parameter('dummy_hard_sweep', [0],
                                                  dimension=0)
        if len(self.sweep_points_pulses[1]) == 0:
            # Internally, 1D and 2D sweeps are handled as 2D sweeps.
            # With this dummy soft sweep, exactly one sequence will be created
            # and the data format will be the same as for a true soft sweep.
            self.sweep_points_pulses.add_sweep_parameter('dummy_soft_sweep', [0],
                                                  dimension=1)

        # temp value ensure that mod_freqs etc are set corretcly
        with temporary_value(*self.temporary_values):
            self.update_operation_dict()
            self.sequences, _ = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                segment_kwargs=self.segment_kwargs, **kw)

        self.mc_points = [np.arange(n) for n in self.sweep_points.length()]

        self._fill_temporary_values()

    def get_sweep_points_for_sweep_n_dim(self):
        return self.sweep_points_pulses

    def preprocess_task(self, task, global_sweep_points,
                        sweep_points=None, **kw):
        preprocessed_task = super().preprocess_task(task, global_sweep_points,
                                                    sweep_points, **kw)

        prefix = preprocessed_task['prefix']
        for k, v in preprocessed_task.get('sweep_functions', dict()).items():
            # add task sweep functions to the global sweep_functions dict with
            # the appropriately prefixed key
            self.sweep_functions_dict[prefix + k] = v

        return preprocessed_task

    def generate_sweep_functions(self):
        # loop over all sweep_points dimensions
        for i in range(len(self.sweep_points)):
            if i >= len(self.sweep_functions):
                # add new dimension with empty multi_sweep_function
                # in case i is out of range
                self.sweep_functions.append(
                    swf.multi_sweep_function(
                        [],
                        parameter_name=f"{i}. dim multi sweep"
                    )
                )
            else:
                # refactor current sweep function into multi_sweep_function
                self.sweep_functions[i] = swf.multi_sweep_function(
                    [self.sweep_functions[i]],
                    parameter_name=f"{i}. dim multi sweep"
                )

            for param in self.sweep_points[i].keys():
                if self.sweep_functions_dict.get(param, None) is not None:
                    self.sweep_functions[i].add_sweep_function(
                        swf.Indexed_Sweep(
                            self.sweep_functions_dict[param],
                            values=self.sweep_points[i][param][0],
                            name=self.sweep_points[i][param][2],
                            parameter_name=param
                        )
                    )
                elif 'freq' in param and 'mod' not in param:
                    # Probably qb frequency that is now contained in an lo sweep
                    pass
                else:
                    if len(self.sweep_functions[i].sweep_functions) == 0 \
                        or self.sweep_functions[i].sweep_functions[0] != awg_swf.SegmentSoftSweep:
                        # assuming that this parameter is a pulse paramater and we
                        # therefore need a SegmentSoftSweep as the first sweep
                        # function in our multi_sweep_function
                        self.sweep_functions[i].insert_sweep_function(
                            pos=0,
                            sweep_function=awg_swf.SegmentSoftSweep
                        )
                    self.sweep_points_pulses[i][param] = self.sweep_points[i][param]

    def resolve_freq_sweep_points(self, **kw):
        """
        This function is called from the init of the class to resolve the
        frequency sweep points. The results are stored in properties of
        the object, which are then used in run_measurement. Aspects to be
        resolved include (if applicable):
        - (shared) LO freqs and (fixed or swept) IFs

        :param kw:
            optimize_mod_freqs: (bool, default: True) If False, the
                mod_freq setting of the first qb on an LO (according to
                the ordering of the task list) determines the LO frequency
                for all qubits on that LO. If True, the mod_freq settings
                will be optimized for the following situations:
                - With allowed_lo_freqs set to None: the LO will be placed in
                  the center of the band of drive frequencies of all qubits on
                  that LO (minimizing the maximum absolute value of
                  mod_freq over all qubits). Do not use in case of a single
                  qubit per LO in this case, as it would result in a
                  ge_mod_freq of 0.
                - With a list allowed_lo_freqs provided: the mod_freq
                  setting of the qubit would act as an unnecessary constant
                  offset in the IF sweep, and optimize_mod_freqs can be used
                  to minimize this offset. In this case optimize_mod_freqs
                  can (and should) even be used in case of a single qubit
                  per LO (where it reduces this offset to 0).
        """
        def major_minor_func(val, major_values):
            ind = np.argmin(np.abs(major_values - val))
            mval = major_values[ind]
            return (mval, val - mval)

        for lo, tasks in self.lo_task_dict.items():
            if self.allowed_lo_freqs is not None:
                # allowed_lo_freqs were specified and we have to make sure we
                # only use frequencies from that set.
                # This is also the case where the sweep is solely done using IF
                if kw.get('optimize_mod_freqs', True):
                    mean_freqs = np.mean([task['freqs'] for task in tasks])
                    func = lambda x : major_minor_func(x, self.allowed_lo_freqs)[1]
                    lo_freqs = func(mean_freqs)
                else:
                    lo_freqs
            else:
                # The LO can be set to any value. For LOs only supplying one
                # qubit this will result in a sweep of the LO only. For LO
                # supplying several qubits the LO is set to the value minimizing
                # the maximum absolut modulation frequency.
                freqs_all = np.array([task['freqs'] for task in tasks])
                if kw.get('optimize_mod_freqs', True) and len(tasks) >= 2:
                    # optimize the mod freq to lie in the middle of the overall
                    # frequency range of this LO
                    lo_freqs = 0.5 * (np.max(freqs_all, axis=0)
                                                    + np.min(freqs_all, axis=0))
                else:
                    # if told not to optimize or only one task is using this LO
                    # we use the mod. freq. of the first qb in the task to set
                    # the LO
                    if not getattr(self, 'modulated', True):
                        lo_freqs = tasks[0]['freqs']
                    else:
                        lo_freqs = tasks[0]['freqs'] \
                                    - self.get_mod_from_qb(
                                        self.get_qubits(tasks[0]['qb'])[0][0])()

            qubits = []
            for task in tasks:
                qb = self.get_qubits(task['qb'])[0][0]
                qubits.append(qb)
                mod_freqs = task['freqs'] - lo_freqs
                if all(mod_freqs - mod_freqs[0] == 0):
                    # mod freq is the same for all acquisitions
                    self.temporary_values.append(
                        (self.get_mod_from_qb(qb), mod_freqs[0]))
                    task['mod_freq'] = mod_freqs[0]
                else:
                    mod_freq_key = task['prefix'] + 'mod_freq'
                    self.sweep_points.add_sweep_parameter(mod_freq_key, mod_freqs, unit='Hz', dimension=0)
                if min(abs(mod_freqs)) < 1e3:
                    log.warning(f'Modulation frequency of {qb.name} '
                                f'is {min(abs(mod_freqs))}.')

                self.sweep_functions_dict.pop(qb.name + '_freq', None)

            lo_freq_key = lo + '_freq'
            self.sweep_functions_dict[lo_freq_key] = \
                    self.get_lo_from_qb(self.get_qubits(tasks[0]['qb'])[0][0]).get_instr().frequency

            self.sweep_points.add_sweep_parameter(lo_freq_key, lo_freqs, unit='Hz', dimension=0)

            self.configure_qubit_mux(qubits, {lo: lo_freqs[0]})

        self.update_operation_dict()

    def generate_lo_task_dict(self, **kw):
        """Fills the lo_task_dict with a list of tasks from
        preprocessed_task_list per LO found in the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            lo = self.get_lo_from_qb(qb)()
            if lo not in self.lo_task_dict:
                self.lo_task_dict[lo] = [task]
            else:
                self.lo_task_dict[lo] += [task]

    def check_all_freqs_per_lo(self, **kw):
        """Checks if all frequency sweeps assigned to one LO have the same
        increment in each step.
        """
        for lo, tasks in self.lo_task_dict.items():
            all_freqs = np.array([task['freqs'] for task in tasks])
            if np.ndim(all_freqs) == 1:
                all_freqs = [all_freqs]
            all_diffs = [np.diff(freqs) for freqs in all_freqs]
            assert all([len(d) == 0 for d in all_diffs]) or \
                all([np.mean(abs(diff - all_diffs[0]) / all_diffs[0]) < 1e-10
                    for diff in all_diffs]), \
                "The steps between frequency sweep points " \
                "must be the same for all qubits using the same LO."

    def sweep_block(self, **kw):
        raise NotImplementedError('Child class has to implement sweep_block.')

    def get_lo_from_qb(self, qb, **kw):
        raise NotImplementedError('Child class has to implement'
                                  ' get_lo_from_qb.')

    def get_mod_from_qb(self, qb, **kw):
        raise NotImplementedError('Child class has to implement'
                                  ' get_mod_from_qb.')

    def guess_label(self, **kw):
        """
        Default label with multi-qubit information
        :param kw: keyword arguments
        """
        if self.label is None:
            self.label = self.experiment_name
            for t in self.task_list:
                self.label += f"_{t['qb']}"

    def _fill_temporary_values(self):
        """adds additionally required qcodes parameter to the
        self.temporary_values list
        """
        # make sure that all triggers are set to the correct trigger separation
        triggers = []
        for qb in self.qubits:
            trigger = qb.instr_trigger
            if trigger() not in triggers:
                self.temporary_values.append((trigger.get_instr().pulse_period,
                                              self.trigger_separation))
                triggers.append(trigger())

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        :return: the analysis instance
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        if 'TwoD' not in analysis_kwargs['options_dict']:
            analysis_kwargs['options_dict']['TwoD'] = True
        self.analysis = spa.MultiQubit_Spectroscopy_Analysis(qb_names=self.qb_names,
                                         **analysis_kwargs)
        return self.analysis


class FeedlineSpectroscopy(MultiTaskingSpectroscopyExperiment):
    """

    """
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'RO frequency, $f_{RO}$',
                      dimension=0),
        'volts': dict(param_name='volt', unit='V',
                      label=r'fluxline voltage',
                      dimension=1),
        'ro_amp':  dict(param_name='amplitude', unit='V',
                      label=r'RO pulse amplitude',
                      dimension=1),
        'sweep_points_2D': dict(param_name='sweep_points_2D', unit='',
                      label=r'sweep_points_2D',
                      dimension=1),
    }
    default_experiment_name = 'FeedlineSpectroscopy'

    def __init__(self, task_list,
                 allowed_lo_freqs=None,
                 trigger_separation=10e-6,
                 **kw):
        super().__init__(task_list,
                         allowed_lo_freqs=allowed_lo_freqs,
                         trigger_separation=trigger_separation,
                         **kw)
        self.autorun(**kw)

    def resolve_freq_sweep_points(self, **kw):
        if len(self.preprocessed_task_list) == 1:
            task = self.preprocessed_task_list[0]
            qb = self.get_qubits(task['qb'])[0][0]
            acq_instr = qb.instr_acq.get_instr()
            if (hasattr(acq_instr, 'use_hardware_sweeper') and \
                acq_instr.use_hardware_sweeper()) or False: # TODO: remove or bool
                freqs = task['freqs']
                lo_freq, delta_f, _ = acq_instr.get_params_for_spectrum(freqs)
                acq_instr.set_lo_freq(self.acq_unit(),
                                                        lo_freq)
                self.segment_kwargs['acquisition_mode']= dict(
                        sweeper='hardware',
                        f_start=freqs[0] - lo_freq,
                        f_step=delta_f,
                        n_step=len(freqs),
                        seqtrigger=True,
                    )
                # adopt df kwargs to hard sweep
                self.df_kwargs['single_int_avg'] = False
                return
        return super().resolve_freq_sweep_points(**kw)

    def sweep_block(self, sweep_points, qb, init_state='0',
                    prepend_pulse_dicts=None , **kw):
        """
        This function creates the blocks for a single transmission measurement
        task.

        :param sweep_points: SweepPoints object
        :param qb: target qubit
        :param init_state: initial state qb (default: '0')
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            CircuitBuilder.block_from_pulse_dicts
        :param kw: further keyword arguments
        """
        # create prepended pulses (pb)
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'ro_el'}}
        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"], pulse_modifs=pulse_modifs)

        # create ParametricValues from param_name in sweep_points
        # (e.g. "ro_amp", "ro_length", etc.)
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        # return all generated blocks (parallel_sweep will arrange them)
        return [pb, ro]

    def configure_qubit_mux(self, qubits, lo_freqs_dict):
        idx = {lo: 0 for lo in lo_freqs_dict}
        for qb in qubits:
            # try whether the external LO name is found in the lo_freqs_dict
            qb_ro_mwg = qb.instr_ro_lo()
            if qb_ro_mwg not in lo_freqs_dict:
                # try whether the acquisition device & unit is in the lo_freqs_dict
                qb_ro_mwg2 = (qb.instr_acq(), qb.acq_unit())
                if qb_ro_mwg2 not in lo_freqs_dict:
                    raise ValueError(f'{qb.name}: Neither {qb_ro_mwg} nor '
                                    f'{qb_ro_mwg2} found in lo_freqs_dict.')
                qb_ro_mwg = qb_ro_mwg2
            # qb.ro_mod_freq(qb.ro_freq() - lo_freqs_dict[qb_ro_mwg])
            qb.acq_I_channel(2 * idx[qb_ro_mwg])
            qb.acq_Q_channel(2 * idx[qb_ro_mwg] + 1)
            idx[qb_ro_mwg] += 1
#        return configure_qubit_mux_readout(qubits, lo_freqs_dict)

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ro_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ro_mod_freq

class QubitSpectroscopy(MultiTaskingSpectroscopyExperiment):
    """

    """
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'Drive frequency, $f_{DR}$',
                      dimension=0),
        'volts': dict(param_name='volt', unit='V',
                      label=r'fluxline voltage',
                      dimension=1),
        'spec_power':  dict(param_name='spec_power', unit='',
                      label=r'Power of spec. MWG',
                      sweep_function_2D='spec_power',
                      dimension=1),
    }
    default_experiment_name = 'QubitSpectroscopy'

    def __init__(self, task_list,
                 pulsed=False,
                 allowed_lo_freqs=None,
                 trigger_separation=10e-6,
                 modulated=False,
                 **kw):
        self.modulated = modulated
        self.pulsed = pulsed
        drive = 'pulsed' if self.pulsed else 'continuous'
        drive += '_spec'
        drive += '_modulated' if self.modulated else ''
        super().__init__(task_list,
                         drive=drive,
                         allowed_lo_freqs=allowed_lo_freqs,
                         trigger_separation=trigger_separation,
                         **kw)
        ro_lo_qubits_dict = {}
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            ro_lo = qb.instr_ro_lo()
            if ro_lo not in self.lo_task_dict:
                ro_lo_qubits_dict[ro_lo] = [qb]
            else:
                ro_lo_qubits_dict[ro_lo][0] += [task]
                ro_lo_qubits_dict[ro_lo][1] += [qb]

        for ro_lo, qubits in ro_lo_qubits_dict.items():
            freqs_all = np.array([qb.ro_freq() for qb in qubits])
            if len(freqs_all) >= 2:
                ro_lo_freq = 0.5 * (np.max(freqs_all) + np.min(freqs_all))
            else:
                ro_lo_freq = freqs_all[0] - qubits[0].ro_mod_freq()
            configure_qubit_mux_readout(qubits, {ro_lo: ro_lo_freq})

        self.autorun(**kw)  # run measurement & analysis if requested in kw

    def sweep_block(self, sweep_points, qb, **kw):
        """
        This function creates the blocks for a single transmission measurement
        task.

        :param sweep_points: SweepPoints object
        :param qb: target qubit
        :param kw: further keyword arguments
        """
        pulse_modifs = {'all': {'element_name': 'ro_el'}}
        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"], pulse_modifs=pulse_modifs)

        # create ParametricValues from param_name in sweep_points
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        # return all generated blocks (parallel_sweep will arrange them)
        return [ro]

    def configure_qubit_mux(self, qubits, lo_freqs_dict):
        pass

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ge_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ge_mod_freq

    def _fill_temporary_values(self):
        super()._fill_temporary_values()
        if self.modulated:
            for task in self.preprocessed_task_list:
                if task.get('mod_freq', None) is not None:
                    # FIXME: HDAWG specific code
                    qb = self.get_qubits(task['qb'])[0][0]
                    mod_freq = qb.instr_pulsar.get_instr().parameters[f'{qb.ge_I_channel()}_direct_mod_freq']
                    self.temporary_values.append((mod_freq,
                                                  task['mod_freq']))
                    amp = qb.instr_pulsar.get_instr().parameters[f'{qb.ge_I_channel()}_direct_IQ_output_amp']
                    self.temporary_values.append((amp,
                                                  qb.spec_mod_amp()))

class ReadoutCalibration(FeedlineSpectroscopy):
    """

    """
    kw_for_sweep_points = {'states':  dict(param_name='initialize', unit='',
                           label=r'qubit state',
                           dimension=1),}
    default_experiment_name = 'ReadoutCalibration'

    def preprocess_task(self, task, global_sweep_points, sweep_points=None, **kw):
        """If the task does not provide the states that are to be measured it
        fills the states with the default of `['0', '1']`.
        """
        # FIXME: sweep_n_dim only looks for sweep points 'initialize' but the
        # sweep points we add to a task will have an additional prefix,
        # e.g. 'qb1_initialize'
        if task.get('states', None) is None:
            task['states'] = ['0', '1']
        return super().preprocess_task(task, global_sweep_points,
                                       sweep_points, **kw)

    def get_sweep_points_for_sweep_n_dim(self):
        # FIXME: only temporary solution until the prefix problem (see fixme
        # in preprocess_task) is solved.
        if self.sweep_points_pulses.find_parameter('initilize') is None:
            self.sweep_points_pulses.add_sweep_parameter(param_name='initialize',
                            values=['0', '1'], unit='',
                            label=r'qubit init state',
                            dimension=1)
        return self.sweep_points_pulses

    def run_analysis(self, analysis_kwargs=None, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        # make sure that spectroscopies for all states are ploted in one plot
        analysis_kwargs['options_dict']['plot_all_traces'] = True
        return super().run_analysis(analysis_kwargs, **kw)