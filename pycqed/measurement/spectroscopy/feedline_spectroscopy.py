import numpy as np
from copy import copy
from copy import deepcopy
from pycqed.utilities.general import assert_not_none
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
    def __init__(self, task_list, allowed_lo_freqs=None, **kw):
        # Passing keyword arguments to the super class (even if they are not
        # needed there) makes sure that they are stored in the metadata.
        super().__init__(task_list, **kw)
        self.sweep_functions_dict = kw.get('sweep_functions_dict', {})
        self.sweep_functions = []
        self.allowed_lo_freqs = allowed_lo_freqs
        self.analysis = {}

        self.lo_task_dict = {}
        self.lo_frequencies = {}
        self.mod_frequencies = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)

        # self.generate_lo_task_dict()
        # self.check_all_freqs_per_lo()

        # self.resolve_freq_sweep_points(**kw)
        # self.sequences, self.mc_points = self.parallel_sweep(
        #     self.preprocessed_task_list, self.sweep_block, **kw)

        # self.adjust_sweep_functions()

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
                else:
                    # assuming that this parameter is a pulse paramater and we
                    # therefore nee a SegmentSoftSweep as the first sweep
                    # function in our multi_sweep_function
                    self.sweep_functions[i].insert_sweep_function(
                        pos=0,
                        sweep_function=awg_swf.SegmentSoftSweep
                    )

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
            if not self.pulsed:
                # TODO: Check that all tasks have the same frequencies
                # TODO: Check that all frequencies can be configured on the LO
                self.lo_frequencies[lo] = tasks[0]['freqs']
                continue

            if self.allowed_lo_freqs is not None:
                # allowed_lo_freqs were specified and we have to make sure we
                # only use frequencies from that set.
                # This is also the case where the sweep is solely done using IF
                if kw.get('optimize_mod_freqs', True):
                    mean_freqs = np.mean([task['freqs'] for task in tasks])
                    func = lambda x : major_minor_func(x, self.allowed_lo_freqs)[1]
                    self.lo_frequencies[lo] = func(mean_freqs)
                else:
                    self.lo_frequencies[lo]
            else:
                # The LO can be set to any value. For LOs only supplying one
                # qubit this will result in a sweep of the LO only. For LO
                # supplying several qubits the LO is set to the value minimizing
                # the maximum absolut modulation frequency.
                freqs_all = np.array([task['freqs'] for task in tasks])
                if kw.get('optimize_mod_freqs', True):
                    self.lo_frequencies[lo] = 0.5 * (np.max(freqs_all, axis=0)
                                                    + np.min(freqs_all, axis=0))
                else:
                    self.lo_frequencies[lo] = tasks[0]['freqs'] \
                                              - self.get_qubits(tasks[0]['qb'])[0][0].ro_mod_freq()

            for task in tasks:
                qb = self.get_qubits(task['qb'])[0][0]
                self.mod_frequencies[qb.name] = task['freqs'] \
                                            - self.lo_frequencies[lo]
                if min(abs(self.mod_frequencies[qb.name])) < 1e3:
                    log.warning(f'Modulation frequency of {qb.name}'
                                f'is {min(abs(self.mod_frequencies[qb.name]))}.')

        self.update_operation_dict()

    def generate_lo_task_dict(self, **kw):
        """Fills the lo_task_dict with a list of tasks from
        preprocessed_task_list per LO found in the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            lo = self.get_lo_from_qb(qb).get_instr()
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
        self.analysis = spa.Spectroscopy(qb_names=self.meas_obj_names,
                                    t_start=self.timestamp
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
        'ro_amps':  dict(param_name='ro_amp', unit='V',
                      label=r'RO pulse amplitude',
                      sweep_function_2D='ro_amp',
                      dimension=1),
        'sweep_points_2D': dict(param_name='sweep_points_2D', unit='',
                      label=r'sweep_points_2D',
                      dimension=1),
    }
    default_experiment_name = 'FeedlineSpectroscopy'

    def __init__(self, task_list, **kw):
        self.feedline_task_dict = {}
        super().__init__(task_list, **kw)
        kw.pop('init_state', None)
        # the block alignments are for: prepended pulses, initial
        # rotations, ro pulse.
        self.sequences, self.mc_points = self.parallel_sweep(
            self.preprocessed_task_list, self.sweep_block,
            block_align = ['center', 'end', 'start'], **kw)

        self.autorun(**kw)  # run measurement & analysis if requested in kw

    def preprocess_task_list(self, **kw):
        """Calls super method and afterwards checks that the preprocessed task
        list does not contain more than one qubit per feedline by comparing the
        ro_I_channel and ro_q_channel parameter between the qubits.
        """
        task_list = super().preprocess_task_list()
        ftd = self.feedline_task_dict
        ro_channels = []
        for task in task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            ro_I_channel = qb.ro_I_channel()
            ro_Q_channel = qb.ro_Q_channel()
            if ro_I_channel in ro_channels or ro_Q_channel in ro_channels:
                # Feedline already exists in preprocessed_task_list
                log.warning('Several qubits on the same AWG channel were specified'
                            ' in task_list. The experiment will assume they belong'
                            ' to the same feedline and will measure the frequencies'
                            ' specified for the first task on that feedline and'
                            ' ignore other frequencies of other tasks on that feedline.'
                qb = self.get_qubits(task['qb'])[0][0]
                qb.instr_ro_lo.get_instr()
                ftd[qb.instr_ro_lo.get_instr()]['qbs'].append(task.pop('qb'))
                for k, v in task.items():
                    if k in ftd
            else:
                ro_channels += [ro_I_channel, ro_Q_channel]
                task = deepcopy(task)
                task['qbs'] = [task.pop('qb')]
                ftd[qb.instr_ro_lo.get_instr()] = {task}

        return self.preprocessed_task_list

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

        # create pulses for initial rotations (ir)
        pulse_modifs = {'all': {'element_name': 'initial_rots_el'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'{self.STD_INIT[init_state][0]} {qb}'],
                                 pulse_modifs=pulse_modifs)

        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"])

        # create ParametricValues from param_name in sweep_points
        # (e.g. "ro_amp", "ro_length", etc.)
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        # return all generated blocks (parallel_sweep will arrange them)
        return [pb, ir, ro]

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ro_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ro_mod_freq

class QubitSpectroscopy(MultiTaskingSpectroscopyExperiment):
    """

    """
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'RO frequency, $f_{RO}$',
                      dimension=0),
        'volts': dict(param_name='volt', unit='V',
                      label=r'fluxline voltage',
                      dimension=1),
        'power':  dict(param_name='spec_power', unit='',
                      label=r'Power of spec. MWG',
                      dimension=1),
    }
    default_experiment_name = 'QubitSpectroscopy'

    def __init__(self, task_list, sweep_points=None, **kw):
        super().__init__(task_list, sweep_points, **kw)

    def sweep_block(self, sweep_points, qb, **kw):
        """
        This function creates the blocks for a single transmission measurement
        task.

        :param sweep_points: SweepPoints object
        :param qb: target qubit
        :param kw: further keyword arguments
        """

        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"])

        # create ParametricValues from param_name in sweep_points
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        # return all generated blocks (parallel_sweep will arrange them)
        return [ro]

    def adjust_sweep_functions(self, **kw):
        """adjust the sweep function for the drive LO frequency sweep and
        calls parent to take care of 2nd sweep dimension

        Uses the first qubit in self.preprocessed_task_list.values()['qbs'] to
        sweep the LO frequency.
        """
        # The 2nd sweep functions are automatically generated by parent
        self.sweep_functions = [
            swf.multi_sweep_function(
                        [
                            swf.Indexed_Sweep(drive_lo['qbs'][0].instr_ge_lo.get_instr().frequency),
                                              drive_lo['lo_freqs'])
                            for drive_lo in self.preprocessed_task_list.values()
                        ],
                        name='Drive LO frequency sweep',
                        parameter_name='Drive LO frequency'),
        ]
        super().adjust_sweep_functions()

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ge_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ge_mod_freq