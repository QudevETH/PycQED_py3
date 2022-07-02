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
    sweep_block to add the right pulses depending on the configuration and
    purpose of the spectroscopy.

    Keyword Arguments:
        task_list (list[dict[str, object]]): List of tasks that will be
            performed in parallel. Each task correspond to one spectroscopy
            carried out on one qubit (FIXME: should be meas_obj in the future)
        trigger_separation (float, optional): Separation between single
            acquisitions. Defaults to 10e-6 s.
        sweep_functions_dict (dict, optional): Dictionary of sweep functions
            with the key of a value being the name of the sweep parameter for
            which the sweep function will be used. Defaults to `dict()`.
        df_name (str, optional): Specify a specific detector function to be
            used. See :meth:`mqm.get_multiplexed_readout_detector_functions`
            for available options.
        df_kwargs (dict, optional): Kwargs of detector function.
            Defaults to `{"live_plot_transform_type":'mag_phase'}`
        cal_states (list, optional): List of calibration states. Should be left
            empty except for special use cases. Spectroscopies dont need
            calibration points.
        segment_kwargs (dict, optional): Defaults to `dict()`. Passed to
            `sweep_n_dim`.
    """
    task_mobj_keys = ['qb']
    @assert_not_none('task_list')
    def __init__(self, task_list, trigger_separation=10e-6, **kw):
        # Passing keyword arguments to the super class (even if they are not
        # needed there) makes sure that they are stored in the metadata.
        df_name = kw.pop('df_name', 'int_avg_det_spec')
        self.df_kwargs = kw.pop('df_kwargs',
            {"live_plot_transform_type":'mag_phase'})
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
        self.analysis = {}

        self.trigger_separation = trigger_separation
        self.grouped_tasks = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)

        self.check_freqs_length()
        self.group_tasks()
        self.check_all_freqs_per_lo()
        self.check_hard_sweep_compatibility()
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
            # not take care of the RO LO and mod freqs.
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
                    ro_lo_freq = freqs_all[0] - qubits[0].ro_mod_freq()
                configure_qubit_mux_readout(
                    qubits=qubits,
                    lo_freqs_dict={ro_lo: ro_lo_freq},
                    set_mod_freq=(not isinstance(self, FeedlineSpectroscopy))
                )

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

    def check_freqs_length(self):
        """Check that all freq arrays in self.preprocessed_task_list have the
        same length.

        Raises:
            ValueError: In case the check is failed.
        """
        if not self.preprocessed_task_list:
            # task_list is empty
            return
        lengths = np.array([len(task['freqs']) for task
                                               in self.preprocessed_task_list])
        diffs = lengths - lengths[0]
        if not all([d == 0 for d in diffs]):
            raise ValueError("All tasks need to have the same number of "
                             "frequency points.")

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

    def check_hard_sweep_compatibility(self, **kw):
        """Checks that either all tasks are hard sweeps or none. Child classes
        can extend functionality e.g. to only allow one hard sweep per feedline.
        """
        is_hard_sweep = [task['hard_sweep'] for task
                                            in self.preprocessed_task_list]
        if all(is_hard_sweep):
            pass
        elif any(is_hard_sweep):
            raise ValueError("Either all tasks need to be hard sweeps or "
                             "none of them.")

    def sweep_block(self, **kw):
        raise NotImplementedError('Child class has to implement sweep_block.')

    def get_lo_from_qb(self, qb, **kw):
        """Get the InstrumentRefParameter of the qubits LO that is corresponding
        to the frequency that is swept in the first dimension, e.g. drive LO in
        qb spec and RO LO in feedline spec.

        Child classes have to implement this method and should not call this
        super method.

        Args:
            qb (QuDev_transmon): Qubit of which the LO InstrumentRefParameter is
                returned.

        Returns:
            InstrumentRefParameter: The LO of the qubit.

        Raises:
            NotImplementedError: In case the child class did not implement the
                method.
        """
        raise NotImplementedError('Child class has to implement'
                                  ' get_lo_from_qb.')

    def get_mod_from_qb(self, qb, **kw):
        """Returns the QCodes parameter for the modulation frequency of the
        first dimension frequency sweep.

        Child classes have to implement this method and should not call this
        super method.

        Args:
            qb (QuDev_transmon): Qubit of which the mod parameter is returned.

        Raises:
            NotImplementedError: In case the child class did not implement the
                method.
        """
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
        """Shortcut to extract the qubit object from a task.
        """
        return self.get_qubits(task['qb'])[0][0]

    def get_task(self, qb):
        """Find the task that is acting on qb.

        Args:
            qb (str or QuDev_Transmon): Qubit for which the we want to get the
                task that is acting on that qubit.

        Returns:
            dict: first task that is found in `self.preprocessed_task_list` that
                contains qb in task[name]. If no task is found None will be
                returned.
        """
        for task in self.preprocessed_task_list:
            if qb == task['qb'] or \
                    (isinstance(qb, QuDev_transmon) and qb.name == task['qb']):
                return task
        return None


class FeedlineSpectroscopy(MultiTaskingSpectroscopyExperiment):
    """Base class to be able to perform 1d and 2d feedline spectroscopies on one
    or multiple qubits and feedlines.

    For now the feedline for which the spectroscopy is performed is specified by
    passing a qubit on this feedline. FIXME: Part of refactoring to make
    compatible with more generic meas_objs (here: feedline might be a meas_obj)

    Based on the hardware available and the configuration of the hardware the
    class decides to either...
        - ...use a soft sweep to sweep the RO LO MWG frequency. Here, if the
          qubits of several tasks share a feedline, a multiplexed RO pulse is
          applied to parallelize the spectroscopies.
        - ...to use the internal frequency (oscillator) sweep of the acq. AWG
          (e.g. SHFQA) to perform a hard sweep. Here we currently do not support
          parallelization within one feedline (TODO).

    Compatible task dict keys:
        qb: QuDev_transmon which provides references to the relevant instruments
            FIXME: should be refactored in future to be meas_obj
        freqs: List or np.array containing the drive frequencies of the
            spectroscopy measurement. (1. dim. sweep points)
        volts: List or np.array of fluxline voltages to perform a 2D qubit
            spectroscopy. This requires to also provide the proper qcodes
            parameter to set the fluxline voltage as item `'sweep_functions'` in
            the task, e.g.:
            `task['sweep_functions'] = {'volt': fluxlines_dict[qb.name]}`
            FIXME: as soon as the fluxline voltage is accesible through the
            qubit, a convenience wrapper should be implemented.
            (2. dim. sweep points)
        ro_amplitude: List or np.array of amplitudes of the RO pulse.
            (2. dim. sweep points)
        ro_length: List or np.array of pulse lengths of the RO pulse.
            (2. dim. sweep points)

    Important: The number of frequency sweep points in the first dimension needs
        to be the same among all tasks. The same holds for the second sweep
        dimension. The step size in the frequency sweep points needs to be the
        same among tasks with qubits that share an RO LO MWG (if applicable).
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
        'ro_length':  dict(param_name='length', unit='s',
                      label=r'RO pulse length',
                      dimension=1),
        'sweep_points_2D': dict(param_name='sweep_points_2D', unit='',
                      label=r'sweep_points_2D',
                      dimension=1),
    }
    default_experiment_name = 'FeedlineSpectroscopy'

    def __init__(self, task_list,
                 trigger_separation=10e-6,
                 **kw):
        super().__init__(task_list,
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
                qb_acq_dev_unit = (qb.instr_acq(), qb.acq_unit())
                if qb_acq_dev_unit not in self.grouped_tasks:
                    self.grouped_tasks[qb_acq_dev_unit] = [task]
                else:
                    self.grouped_tasks[qb_acq_dev_unit] += [task]

    def check_hard_sweep_compatibility(self, **kw):
        super().check_hard_sweep_compatibility()
        for acq_unit, tasks in self.grouped_tasks.values():
            if not any([task['hard_sweep'] for task in tasks]):
                continue
            if len(tasks) > 1:
                raise ValueError(f"Only one task per feedline for hard sweeps."
                                 f" Qubits {[task['qb'] for task in tasks]} share"
                                 f" acquisition channels {acq_unit}.")

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
    """Base class to be able to perform 1d and 2d qubit spectroscopies on one
    or multiple qubits.

    Based on the hardware available and the configuration of the hardware the
    class decides to either...
        - ...route the LO MWG past the IQ mixer and sweep the MWG frequency
          in a soft sweep, realizing pulsed spectroscopy by gating the MWG, or
        - ...to apply a constant drive tone of constant frequency on the IF of
          the IQ mixer upconversion while still using a soft sweep to sweep the
          MWG frequency and a gate pulse on the MWG for pulsed spectroscopies.
        - ...to use the internal frequency (oscillator) sweep of the drive AWG
          (e.g. SHFSG/QC) to perform a hard sweep. Here the pulsed spectroscopy
          is realized by programming a GaussianFilteredIQSquare pulse to the
          drive AWG.

    Compatible task dict keys:
        qb: QuDev_transmon which provides references to the relevant instruments
            FIXME: should be refactored in future to be meas_obj
        freqs: List or np.array containing the drive frequencies of the
            spectroscopy measurement. (1. dim. sweep points)
        volts: List or np.array of fluxline voltages to perform a 2D qubit
            spectroscopy. This requires to also provide the proper qcodes
            parameter to set the fluxline voltage as item `'sweep_functions'` in
            the task, e.g.:
            `task['sweep_functions'] = {'volt': fluxlines_dict[qb.name]}`
            FIXME: as soon as the fluxline voltage is accesible through the
            qubit, a convenience wrapper should be implemented.
            (2. dim. sweep points)
        spec_power: List or np.array of different LO powers that should be used.
            (2. dim. sweep points)
        spec_pulse_amplitude: List or np.array of amplitudes of the drive pulse
            used in pulsed spectroscopy. (2. dim. sweep points)
        spec_pulse_length: List or np.array of pulse lengths of the drive pulse
            used in pulsed spectroscopy. (2. dim. sweep points)

    Important: The number of frequency sweep points in the first dimension needs
        to be the same among all tasks. The same holds for the second sweep
        dimension. The step size in the frequency sweep points needs to be the
        same among tasks with qubits that share a drive LO MWG (if applicable).
    """
    kw_for_sweep_points = {
        'freqs': dict(param_name='freq', unit='Hz',
                      label=r'Drive frequency, $f_{DR}$',
                      dimension=0),
        'volts': dict(param_name='volt', unit='V',
                      label=r'fluxline voltage',
                      dimension=1),
        'spec_power':  dict(param_name='spec_power', unit='dBm',
                      label=r'Power of spec. MWG',
                      dimension=1),
        'spec_pulse_amplitude':  dict(param_name='amplitude', unit='',
                      label=r'Amplitude of spec. pulse',
                      dimension=1),
        'spec_pulse_length':  dict(param_name='length', unit='s',
                      label=r'Length of spec. pulse',
                      dimension=1),
    }
    default_experiment_name = 'QubitSpectroscopy'

    def __init__(self, task_list,
                 pulsed=False,
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
                         trigger_separation=trigger_separation,
                         segment_kwargs={'mod_config':{},
                                         'sine_config':{},
                                         'sweep_params':{},},
                         **kw)
        self.autorun(**kw)  # run measurement & analysis if requested in kw

    def preprocess_task(self, task, global_sweep_points,
                        sweep_points=None, **kw):
        preprocessed_task = super().preprocess_task(task, global_sweep_points,
                                                    sweep_points, **kw)
        qb = self.get_qubit(preprocessed_task)
        pulsar = qb.instr_pulsar.get_instr()
        awg_name = pulsar.get(f'{qb.ge_I_channel()}_awg')
        preprocessed_task['hard_sweep'] = (
            hasattr(pulsar, f'{awg_name}_use_hardware_sweeper') and \
            pulsar.get(f'{awg_name}_use_hardware_sweeper'))

        # Convenience feature to automatically use the LO power parameter to
        # sweep the spec_power in a soft_sweep spectroscopy
        prefix = preprocessed_task['prefix']
        if 'spec_power' in preprocessed_task \
           and not preprocessed_task['hard_sweep'] \
           and prefix + 'spec_power' not in self.sweep_functions_dict.keys():
            self.sweep_functions_dict[prefix + 'spec_power'] = \
                qb.instr_lo.get_instr().power()
        return preprocessed_task

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
            else:
                # no external LO, use synthesizer
                # FIXME: SHFSG(QC) specifc
                pulsar = qb.instr_pulsar.get_instr()
                id = pulsar.get(f"{qb.ge_I_channel()}_id")
                awg = pulsar.get(f"{qb.ge_I_channel()}_awg")
                sg_channel = pulsar.awg_intefaces[awg].awg.sgchannels[int(id[2]) - 1]
                qb_awg_synth = f"{awg}_synth{sg_channel.synthesizer()}"
                if qb_awg_synth not in self.grouped_tasks:
                    self.grouped_tasks[qb_awg_synth] = [task]
                else:
                    self.grouped_tasks[qb_awg_synth] += [task]

    def check_hard_sweep_compatibility(self, **kw):
        super().check_hard_sweep_compatibility()
        for awg_synth, tasks in self.grouped_tasks.values():
            if not any([task['hard_sweep'] for task in tasks]):
                continue
            elif len(tasks) > 1:
                # FIXME: SHFSG(QC) specific
                raise ValueError(f"Currently only one task per synthesizer is "
                                 f"supported for hard sweeps. Qubits "
                                 f"{[task['qb'] for task in tasks]} share "
                                 f"synthesizer {awg_synth}.")

    def resolve_freq_sweep_points(self, **kw):
        """Configures potential hard_sweeps and afterwards calls super method
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            pulsar = qb.instr_pulsar.get_instr()
            freqs = task['freqs']
            ch = qb.ge_I_channel()
            amp_range = pulsar.get(f'{ch}_amp')
            amp = 10 ** (qb.spec_power() / 20 - 0.5)
            gain = amp / amp_range
            self.segment_kwargs['mod_config'][ch] = \
                dict(internal_mod=self.pulsed)
            self.segment_kwargs['sine_config'][ch] = \
                dict(continous=not self.pulsed,
                     gains=tuple(gain * x for x in (0.0, 1.0, 1.0, 0.0)))
            if task['hard_sweep']:
                center_freq, mod_freqs = pulsar.get_params_for_spectrum(
                    qb.ge_I_channel(), freqs)
                self.segment_kwargs['sweep_params'][f'{ch}_osc_sweep'] = \
                    mod_freqs
                pulsar.set(f'{ch}_centerfreq', center_freq)
                # adopt df kwargs to hard sweep
                self.df_name = 'int_avg_det'
                self.df_kwargs['single_int_avg'] = False
        return super().resolve_freq_sweep_points(**kw)

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
        pulse_modifs = {'all': {'element_name': 'ro_el'}}
        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"], pulse_modifs=pulse_modifs)

        # add marker pulse in case we perform pulsed spectroscopy
        task = self.get_task(qb)
        if self.pulsed:
            qubit = self.get_qubit(task)
            pulse_modifs = {'all': {'element_name': 'spec_el',
                                    'length': self.trigger_separation - 100e-9
                                              - qubit.acq_length()}}
            if task['hard_sweep']:
                pulse_modifs['all']['channel'] = qubit.ge_I_channel()
                pulse_modifs['all']['amplitude'] = \
                    10 ** (qubit.spec_power() / 20 - 0.5)
            spec = self.block_from_ops('spec', [f"Spec {qb}"],
                                       pulse_modifs=pulse_modifs)
            # create ParametricValues from param_name in sweep_points
            for sweep_dict in sweep_points:
                for param_name in sweep_dict:
                    for pulse_dict in spec.pulses:
                        if param_name in pulse_dict:
                            pulse_dict[param_name] = ParametricValue(param_name)
            return [spec, ro]
        else:
            if task is not None and task['hard_sweep']:
                # We need to add a pulse of length 0 to make sure the AWG channel is
                # programmed by pulsar and pulsar gets the hard sweep information
                pulse_modifs = {'all': {'element_name': 'spec_el',
                                        'nr_sigma': 0}}
                empty = self.block_from_ops('spec', [f"X180 {qb}"],
                                        pulse_modifs=pulse_modifs)
                return [empty, ro]
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
            return qb.swf_drive_lo_freq()

    def _fill_temporary_values(self):
        super()._fill_temporary_values()
        if self.modulated:
            for task in self.preprocessed_task_list:
                if task.get('mod_freq', False):
                    # FIXME: HDAWG specific code, should be moved to pulsar
                    qb = self.get_qubit(task)
                    mod_freq = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_mod_freq'
                    ]
                    self.temporary_values.append((mod_freq,
                                                  task['mod_freq']))
                    amp = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_output_amp'
                    ]
                    self.temporary_values.append((amp,
                                                  qb.spec_mod_amp()))
                else:
                    log.error('Task for modulated spectroscopy does not contain'
                              'mod_freq.')


class ReadoutCalibration(FeedlineSpectroscopy):
    """Perform feedline spectroscopies for several initial states to determine
    the RO frequency with the highest contrast.

    While the experiment support parallelization, it only allows to have the
    same initial states for all qubits.

    Arguments:
        states (list[str], optional): List of strings specifying the initial
            states to be measured. Defaults to `["g", "e"]`.

    Compatible task dict keys:
        freqs: See :class:`FeedlineSpectroscopy' for details.

    Updates:
        qb.ro_freq: To the value maximizing the distance in the IQ plane between
            the first two states in task["states"]. If you want to make your RO
            more dicriminating between e & f you should pass states
            ["e", "f", ...] instead of e.g. ["g", "e", "f"].
    """
    default_experiment_name = 'ReadoutCalibration'

    def __init__(self, task_list, trigger_separation=10e-6,
                 states=["g", "e"], **kw):
        self.states = states
        super().__init__(task_list, trigger_separation, **kw)

    def get_sweep_points_for_sweep_n_dim(self):
        if self.sweep_points_pulses.find_parameter('initialize') is None:
            self.sweep_points_pulses.add_sweep_parameter(
                param_name='initialize',
                values=self.states, unit='',
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
                pdd['projected_data_dict'][qb.name]['distance'][
                    f'{self.states[0]}-{self.states[1]}'
                ][1] # (distances, argmax) -> index 1
            ]
            qb.set(f'ro_freq', ro_freq)
