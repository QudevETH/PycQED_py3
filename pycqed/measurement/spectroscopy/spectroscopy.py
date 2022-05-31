import numpy as np
from pycqed.utilities.general import assert_not_none, \
    configure_qubit_mux_readout
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
from pycqed.measurement.waveform_control.block import ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.spectroscopy_analysis as spa
from pycqed.utilities.general import temporary_value
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
import logging
log = logging.getLogger(__name__)


class MultiTaskingSpectroscopyExperiment(CalibBuilder):
    """Adds functionality to sweep LO and modulation frequencies in
    a spectroscopy experiment. Automatically determines whether the LO, the
    mod. freq. or both are swept. Compatible with hard sweeps.

    Child classes implement the spectroscopy experiment and need to implement
    sweep_block.
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
        # Used to set the acquisition mode in the segment, e.g. for fast
        # SHFQA spectroscopy. Default is 'software' sweeper.
        self.segment_kwargs = kw.pop('segment_kwargs', dict())
        super().__init__(task_list, df_name=df_name, cal_states=cal_states,
                         **kw)
        self.sweep_functions_dict = kw.get('sweep_functions_dict', {})
        self.sweep_functions = []
        # sweep points that are passed to sweep_n_dim and used to generate
        # segments. This reduced set is introduce to prevent that a segment
        # is generated for every frequency sweep point.
        self.sweep_points_pulses = SweepPoints(min_length=2, )
        self.allowed_lo_freqs = allowed_lo_freqs
        self.analysis = {}

        self.trigger_separation = trigger_separation
        self.grouped_tasks = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)

        self.group_tasks()
        self.check_all_freqs_per_lo()
        self.resolve_freq_sweep_points(**kw)
        self.generate_sweep_functions()
        if len(self.sweep_points_pulses[0]) == 0:
            # Create a single segement if no hard sweep points are provided.
            self.sweep_points_pulses.add_sweep_parameter('dummy_hard_sweep',
                                                         [0], dimension=0)
        if len(self.sweep_points_pulses[1]) == 0:
            # Internally, 1D and 2D sweeps are handled as 2D sweeps.
            # With this dummy soft sweep, exactly one sequence will be created
            # and the data format will be the same as for a true soft sweep.
            self.sweep_points_pulses.add_sweep_parameter('dummy_soft_sweep',
                                                         [0], dimension=1)

        self._fill_temporary_values()
        # temp value ensure that mod_freqs etc are set corretcly
        with temporary_value(*self.temporary_values):
            # configure RO LOs for potential multiplexed RO
            # This is especially necessary for qubit spectroscopies as they do
            # not take care of the RO LO and mode freqs.
            ro_lo_qubits_dict = {}
            for task in self.preprocessed_task_list:
                qb = self.get_qubit(task)
                ro_lo = qb.instr_ro_lo()
                if ro_lo not in ro_lo_qubits_dict:
                    ro_lo_qubits_dict[ro_lo] = [qb]
                else:
                    ro_lo_qubits_dict[ro_lo] += [qb]
            for ro_lo, qubits in ro_lo_qubits_dict.items():
                freqs_all = np.array([qb.ro_freq() for qb in qubits])
                if len(freqs_all) >= 2:
                    ro_lo_freq = 0.5 * (np.max(freqs_all) + np.min(freqs_all))
                else:
                    ro_lo_freq = freqs_all[0] + qubits[0].ro_mod_freq()
                configure_qubit_mux_readout(qubits, {ro_lo: ro_lo_freq})

            self.update_operation_dict()

            self.sequences, _ = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                segment_kwargs=self.segment_kwargs, **kw)

        self.mc_points = [np.arange(n) for n in self.sweep_points.length()]

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
        """Loops over all sweep points and adds the according sweep function to
        self.sweep_functions. The appropriate sweep function is taken from
        self.sweep_function_dict. For multiple sweep points in one dimension a
        multi_sweep is used.

        Caution: Special behaviour if the sweep point param_name is not found in
        self.sweep_function_dict.keys(): We assume that this sweep point is a
        pulse parameter and insert the class SegmentSoftSweep as placeholder
        that will be replaced by an instance in QE._configure_mc.
        """
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
            elif not isinstance(self.sweep_functions[i],
                                swf.multi_sweep_function):
                # refactor current sweep function into multi_sweep_function
                self.sweep_functions[i] = swf.multi_sweep_function(
                    [self.sweep_functions[i]],
                    parameter_name=f"{i}. dim multi sweep"
                )

            for param in self.sweep_points[i].keys():
                if self.sweep_functions_dict.get(param, None) is not None:
                    sw_ctrl = getattr(self.sweep_functions_dict[param],
                                      'sweep_control', 'soft')
                    if sw_ctrl == 'hard':
                        # hard sweep is not encapsulated by Indexed_Sweep
                        sweep_function = self.sweep_functions_dict[param]
                        self.sweep_functions[i] = \
                            self.sweep_functions_dict[param]
                        # there can only be one hard sweep per dimension
                        break
                    else:
                        sweep_function = swf.Indexed_Sweep(
                            self.sweep_functions_dict[param],
                            values=self.sweep_points[i][param][0],
                            name=self.sweep_points[i][param][2],
                            parameter_name=param
                        )
                        self.sweep_functions[i].add_sweep_function(
                            sweep_function
                        )
                elif 'freq' in param and 'mod' not in param:
                    # Probably qb frequency that is now contained in an lo sweep
                    pass
                else:
                    # assuming that this parameter is a pulse parameter and we
                    # therefore need a SegmentSoftSweep as the first sweep
                    # function in our multi_sweep_function
                    if not self.sweep_functions[i].sweep_functions \
                            or self.sweep_functions[i].sweep_functions[0] != \
                                awg_swf.SegmentSoftSweep:
                        self.sweep_functions[i].insert_sweep_function(
                            pos=0,
                            sweep_function=awg_swf.SegmentSoftSweep
                        )
                    self.sweep_points_pulses[i][param] = \
                        self.sweep_points[i][param]

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
        """
        for lo, tasks in self.grouped_tasks.items():
            if np.any([task.get('hard_sweep', False) for task in tasks]) or \
                    len(tasks) == 1:
                # hard sweeps are taken care of by the child before calling this
                # parent function
                # No need to coordinate the mod freq of different qubits. We
                # can use the swf returned by the qubit itself.
                qb = self.get_qubit(tasks[0])
                freq_key = qb.name + '_freq'
                self.sweep_functions_dict[freq_key] = self.get_swf_from_qb(qb)
                continue

            # We resolve the LO frequency.
            # For LOs supplying several qubits the LO is set to the value
            # minimizing the maximum absolut modulation frequency.
            freqs_all = np.array([task['freqs'] for task in tasks])
            # optimize the mod freq to lie in the middle of the overall
            # frequency range of this LO
            lo_freqs = 0.5 * (np.max(freqs_all, axis=0)
                                            + np.min(freqs_all, axis=0))
            lo_freq_key = lo + '_freq'
            self.sweep_points.add_sweep_parameter(param_name=lo_freq_key,
                                                  values=lo_freqs,
                                                  unit='Hz',
                                                  dimension=0)
            self.sweep_functions_dict[lo_freq_key] = self.get_lo_from_qb(
                                                        self.get_qubit(tasks[0])
                                                     ).get_instr().frequency

            # We resolve the modulation frequency of the different qubits.
            qubits = []
            for task in tasks:
                qb = self.get_qubit(task)
                qubits.append(qb)
                mod_freqs = task['freqs'] - lo_freqs
                if all(mod_freqs - mod_freqs[0] == 0):
                    # mod freq is the same for all acquisitions
                    self.temporary_values.append(
                        (self.get_mod_from_qb(qb), mod_freqs[0]))
                    task['mod_freq'] = mod_freqs[0]
                else:
                    self.sweep_points.add_sweep_parameter(
                        param_name=task['prefix'] + 'mod_frequency',
                        label=task['prefix'] + 'mod_frequency',
                        values=mod_freqs,
                        unit='Hz',
                        dimension=0
                    )

    def group_tasks(self, **kw):
        """Fills the grouped_tasks dict with a list of tasks from
        preprocessed_task_list per LO found in the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            lo_instr = self.get_lo_from_qb(qb)
            if lo_instr is not None:
                lo_name = lo_instr()
                if lo_name not in self.grouped_tasks:
                    self.grouped_tasks[lo_name] = [task]
                else:
                    self.grouped_tasks[lo_name] += [task]

    def check_all_freqs_per_lo(self, **kw):
        """Checks if all frequency sweeps assigned to one LO have the same
        increment in each step.
        """
        for lo, tasks in self.grouped_tasks.items():
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
        self.analysis = spa.MultiQubit_Spectroscopy_Analysis(
            qb_names=self.qb_names, **analysis_kwargs
        )
        return self.analysis

    def get_qubit(self, task):
        return self.get_qubits(task['qb'])[0][0]


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
                         segment_kwargs={'acquisition_mode': \
                                        dict(sweeper='software')},
                         **kw)
        self.autorun(**kw)

    def preprocess_task(self, task, global_sweep_points,
                        sweep_points=None, **kw):
        preprocessed_task = super().preprocess_task(task, global_sweep_points,
                                                    sweep_points, **kw)
        qb = self.get_qubit(preprocessed_task)
        acq_instr = qb.instr_acq.get_instr()
        preprocessed_task['hard_sweep'] = (hasattr(acq_instr, \
                                                   'use_hardware_sweeper') and \
                                           acq_instr.use_hardware_sweeper())
        return preprocessed_task

    def group_tasks(self, **kw):
        """Groups tasks that share an LO ar an acq. device & unit.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            lo_instr = qb.instr_ro_lo
            if lo_instr is not None:
                qb_ro_mwg = lo_instr()
                if qb_ro_mwg not in self.grouped_tasks:
                    self.grouped_tasks[qb_ro_mwg] = [task]
                else:
                    self.grouped_tasks[qb_ro_mwg] += [task]
            else:
                # no external LO, use acq device instead
                qb_ro_mwg = (qb.instr_acq(), qb.acq_unit())
                if qb_ro_mwg not in self.grouped_tasks:
                    self.grouped_tasks[qb_ro_mwg] = [task]
                else:
                    self.grouped_tasks[qb_ro_mwg] += [task]

    def resolve_freq_sweep_points(self, **kw):
        """Configures potential hard_sweeps and afterwards calls super method
        """
        for task in self.preprocessed_task_list:
            if task['hard_sweep']:
                qb = self.get_qubits(task['qb'])[0][0]
                acq_instr = qb.instr_acq.get_instr()
                freqs = task['freqs']
                lo_freq, delta_f, _ = acq_instr.get_params_for_spectrum(freqs)
                # adjust ro_freq in tmp_vals such that qb.prepare will set the
                # correct lo_freq.
                self.temporary_values.append(
                    (qb.ro_freq, lo_freq))
                self.temporary_values.append(
                    (qb.ro_mod_freq, 0))
                self.segment_kwargs['acquisition_mode']= dict(
                        sweeper='hardware',
                        f_start=freqs[0] - lo_freq,
                        f_step=delta_f,
                        n_step=len(freqs),
                        seqtrigger=True,
                    )
                # adopt df kwargs to hard sweep
                self.df_kwargs['single_int_avg'] = False
        return super().resolve_freq_sweep_points(**kw)

    def sweep_block(self, sweep_points, qb, prepend_pulse_dicts=None , **kw):
        """This function creates the blocks for a single transmission
        measurement task.

        Args:
            sweep_points (SweepPoints): SweepPoints object
            qb (QuDev_transmon): target qubit
            prepend_pulse_dicts (dict): prepended pulses, see
                CircuitBuilder.block_from_pulse_dicts. Defaults to None.

        Returns:
            list of :class:`~pycqed.measurement.waveform_control.block.Block`s:
                List of blocks for the operation.
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

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ro_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ro_mod_freq

    def get_swf_from_qb(self, qb: QuDev_transmon):
        return qb.swf_ro_freq_lo()

    def run_update(self, **kw):
        """
        Updates the RO frequency of the qubit in each task with the value
        minimizing the transmission magnitude, found in the analysis.
        :param kw: keyword arguments
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            pdd = self.analysis.proc_data_dict
            ro_freq = pdd['sweep_points_dict'][qb.name]['sweep_points'][
                np.argmin(pdd['projected_data_dict'][qb.name]['Magnitude'])
            ]
            qb.set(f'ro_freq', ro_freq)

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
        self.default_experiment_name += '_pulsed' if self.pulsed \
                                                  else '_continuous'
        super().__init__(task_list,
                         drive=drive,
                         allowed_lo_freqs=allowed_lo_freqs,
                         trigger_separation=trigger_separation,
                         **kw)
        self.autorun(**kw)  # run measurement & analysis if requested in kw

    def sweep_block(self, sweep_points, qb, **kw):
        """This function creates the blocks for a single transmission
        measurement task.

        Args:
            sweep_points (SweepPoints): SweepPoints object
            qb (QuDev_transmon): target qubit

        Returns:
            list of :class:`~pycqed.measurement.waveform_control.block.Block`s:
                List of blocks for the operation.
        """
        # add marker pulse in case we perform pulsed spectroscopy
        if self.pulsed:
            pulse_modifs = {'all': {'element_name': 'spec_el'}}
            spec = self.block_from_ops('spec', [f"Spec {qb}"],
                                       pulse_modifs=pulse_modifs)

        pulse_modifs = {'all': {'element_name': 'ro_el'}}
        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"], pulse_modifs=pulse_modifs)

        # return all generated blocks (parallel_sweep will arrange them)
        if self.pulsed:
            return [spec, ro]
        return [ro]

    def get_lo_from_qb(self, qb, **kw):
        return qb.instr_ge_lo

    def get_mod_from_qb(self, qb, **kw):
        return qb.ge_mod_freq

    def get_swf_from_qb(self, qb: QuDev_transmon):
        if getattr(self, 'modulated', True):
            return swf.Offset_Sweep(
                sweep_function=self.get_lo_from_qb(qb).get_instr().frequency,
                offset=-self.get_mod_from_qb(qb)(),
                name='Drive frequency',
                parameter_name='Drive frequency')
        else:
            return self.get_lo_from_qb(qb).get_instr().frequency

    def _fill_temporary_values(self):
        super()._fill_temporary_values()
        if self.modulated:
            for task in self.preprocessed_task_list:
                if task.get('mod_freq', False):
                    # FIXME: HDAWG specific code
                    qb = self.get_qubit(task)
                    mod_freq = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_mod_freq'
                    ]
                    self.temporary_values.append((mod_freq,
                                                  task['mod_freq']))
                    amp = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_IQ_output_amp'
                    ]
                    self.temporary_values.append((amp,
                                                  qb.spec_mod_amp()))
                else:
                    log.error('Task for modulated spectroscopy does not contain'
                              'mod_freq.')

class ReadoutCalibration(FeedlineSpectroscopy):
    default_experiment_name = 'ReadoutCalibration'

    def __init__(self, task_list, allowed_lo_freqs=None,
                 trigger_separation=10e-6, **kw):
        self.kw_for_sweep_points['states'] = dict(param_name='initialize',
                                                  unit='',
                                                  label=r'qubit state',
                                                  dimension=1)
        super().__init__(task_list, allowed_lo_freqs, trigger_separation, **kw)

    def get_sweep_points_for_sweep_n_dim(self):
        if self.sweep_points_pulses.find_parameter('initialize') is None:
            self.sweep_points_pulses.add_sweep_parameter(
                param_name='initialize',
                values=['g', 'e'], unit='',
                label=r'qubit init state',
                dimension=1
            )
        return self.sweep_points_pulses

    def run_analysis(self, analysis_kwargs=None, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        self.analysis = spa.MultiQubit_AvgRoCalib_Analysis(
            qb_names=self.qb_names,
            **analysis_kwargs
        )
        return self.analysis

    def run_update(self, **kw):
        """
        Updates the RO frequency of the qubit in each task with the value
        maximizing the g-e S21 distance, found in the analysis.
        :param kw: keyword arguments
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            pdd = self.analysis.proc_data_dict
            ro_freq = pdd['sweep_points_dict'][qb.name]['sweep_points'][
                pdd['projected_data_dict'][qb.name]['distance']['g-e'][1]
            ]
            qb.set(f'ro_freq', ro_freq)