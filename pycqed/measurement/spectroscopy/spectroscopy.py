import numpy as np
import traceback
from pycqed.utilities.general import assert_not_none, \
    configure_qubit_mux_readout
from pycqed.utilities.math import dbm_to_vp
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
from pycqed.measurement.waveform_control.block import ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.spectroscopy_analysis as spa
from pycqed.utilities.general import temporary_value
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
        df_kwargs (dict, optional): Kwargs of the detector function. The entry
            `{"live_plot_transform_type":'mag_phase'}` will be added by default
            if the key "live_plot_transform_type" does not already exist.
        cal_states (list, optional): List of calibration states. Should be left
            empty except for special use cases. Spectroscopies dont need
            calibration points.
        segment_kwargs (dict, optional): Defaults to `dict()`. Passed to
            `sweep_n_dim`.
    """
    task_mobj_keys = ['qb']

    @assert_not_none('task_list')
    def __init__(self, task_list, sweep_points=None, trigger_separation=10e-6,
                 **kw):
        df_name = kw.pop('df_name', 'int_avg_det_spec')
        df_kwargs = kw.pop('df_kwargs', {})
        df_kwargs['live_plot_transform_type'] = \
            df_kwargs.get("live_plot_transform_type", 'mag_phase')
        cal_states = kw.pop('cal_states', [])
        self.segment_kwargs = kw.pop('segment_kwargs', dict())
        """Used to set the acquisition mode, the sine config and modulation
        config.

        These configurations are forwarded to the elements and used by pulsar
        respectively the acq. device to configure the hardware. This is used
        e.g. for hard sweeps on SHF hardware.
        """
        super().__init__(task_list, sweep_points=sweep_points,
                         df_name=df_name, cal_states=cal_states,
                         df_kwargs=df_kwargs, **kw)
        self.sweep_functions_dict = kw.get('sweep_functions_dict', {})
        self.sweep_functions = []
        self.sweep_points_pulses = SweepPoints(min_length=2, )
        """sweep points that are passed to sweep_n_dim and used to generate
        segments. This reduced set is introduce to prevent that a segment
        is generated for every frequency sweep point.
        """

        self.trigger_separation = trigger_separation
        self.grouped_tasks = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)

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
                    set_mod_freq=(not isinstance(self, ResonatorSpectroscopy))
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
        for k, v in preprocessed_task.get('sweep_functions_dict', {}).items():
            # add task sweep functions to the global sweep_functions_dict with
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
        # The following dict of lists will store the mapping between
        # (local/global) sweep parameters and (local/global) sweep functions
        class ListsDict(dict):
            def append(self, key, value):
                if key not in self:
                    self[key] = []
                self[key].append(value)
        sf_for_sp = ListsDict()
        # helper lists needed to determine the mapping
        all_sp = [sp for d in self.sweep_points for sp in d]
        prefixes = [t['prefix'] for t in self.preprocessed_task_list]
        # for each sweep function, find for which sweep params it is needed
        for sf in self.sweep_functions_dict.keys():
            if sf in all_sp:  # exact match
                # local swf for local sp or global swf for global sp
                sf_for_sp.append(sf, sf)
            else:
                # it might be a local sweep function for a global sweep point
                sp_p = list(set([(sp, p) for sp in all_sp for p in prefixes
                                 if sf == p + sp]))
                if len(sp_p):
                    if len(sp_p) > 1:
                        raise ValueError(
                            f'Sweep function matches multiple combinations '
                            f'of prefix and sweep parameter: {sp_p}.')
                    sf_for_sp.append(sp_p[0][0], sf)
        # for each sweep param that does not have a sweep function yet
        for sp in all_sp:
            if sp not in sf_for_sp:
                # It might be a global sweep param that is overridden in all
                # tasks, in which case we can ignore it.
                if all([sf_for_sp.get(p + sp, []) for p in prefixes]):
                    sf_for_sp[sp] = []  # no sweep function needed

        # We can now add the needed sweep functions to self.sweep_functions.
        # loop over all sweep_points dimensions
        for i in range(len(self.sweep_points)):
            sw_ctrl = None
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
                # Add sweep functions according to the mapping sf_for_sp.
                for sf_key in sf_for_sp.get(param, []):
                    sf = self.sweep_functions_dict[sf_key]
                    if sf is None:
                        continue
                    sf_sw_ctrl = getattr(sf, 'sweep_control', 'soft')
                    if sw_ctrl is not None and sf_sw_ctrl != sw_ctrl:
                        raise ValueError(
                            f'Cannot combine soft sweep and hard sweep in '
                            f'dimension {i}.')
                    if sf_sw_ctrl == 'hard':
                        if i != 0:
                            raise ValueError(
                                f'Hard sweeps are only allowed in dimension '
                                f'0. Cannot perform hard sweep for {param} '
                                f'in dim {i}.')
                        # hard sweep is not encapsulated by Indexed_Sweep
                        # and we only need one hard sweep per dimension
                        if sw_ctrl is None:
                            self.sweep_functions[i] = sf
                    else:
                        sweep_function = swf.Indexed_Sweep(
                            sf, values=self.sweep_points[i][param][0],
                            name=self.sweep_points[i][param][2],
                            parameter_name=param
                        )
                        self.sweep_functions[i].add_sweep_function(
                            sweep_function
                        )
                    sw_ctrl = sf_sw_ctrl
                # Params for which no sweep function is present are treated as
                # pulse parameter sweeps below.
                if param in sf_for_sp:
                    continue  # was treated above already
                elif i == 1:  # dimension 1
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
                else:  # dimension 0
                    # assuming that this parameter is a pulse parameter, and we
                    # therefore need a SegmentHardSweep as the first sweep
                    # function in our multi_sweep_function
                    if not self.sweep_functions[i].sweep_functions:
                        # no previous sweep functions defined in dim 0
                        self.sweep_functions[i].insert_sweep_function(
                            pos=0,
                            sweep_function=awg_swf.SegmentHardSweep)
                    elif self.sweep_functions[i].sweep_functions[
                             0].sweep_control != 'hard':
                        # sweep_functinos[0] is not empty but what is there
                        # isn't a hard sweep;
                        raise NotImplementedError(
                            'Combined sweeping of pulse parameters and other '
                            'parameters for which a sweep function is provided '
                            'is not supported in dimension 0.')

                    self.sweep_points_pulses[i][param] = \
                        self.sweep_points[i][param]
                    sw_ctrl = 'hard'

    def resolve_freq_sweep_points(self, **kw):
        """
        This function is called from the init of the class to resolve the
        frequency sweep points. The results are stored in properties of
        the object, which are then used in run_measurement. Aspects to be
        resolved include (if applicable):
        - (shared) LO freqs and (fixed or swept) IFs

        FIXME: add feature to support only a specific range of LO frequencies,
        e.g for SHF center frequencies.
        """
        for lo, tasks in self.grouped_tasks.items():
            if np.any([task.get('hard_sweep', False) for task in tasks]) or \
                    len(tasks) == 1:
                # hard sweeps are taken care of by the child before calling this
                # parent function
                # No need to coordinate the mod freq of different qubits. We
                # can use the swf returned by the qubit itself.
                self.sweep_functions_dict.update({
                    task['prefix'] + 'freq':
                        self.get_lo_swf(self.get_qubit(task))
                    for task in tasks})
                continue

            # We resolve the LO frequency.
            # For LOs supplying several qubits, the LO is set to the value
            # minimizing the maximum absolut modulation frequency.
            sp_all = [task['sweep_points'] for task in tasks]
            dim_all = [sp.find_parameter('freq') for sp in sp_all]
            if any([dim != dim_all[0] for dim in dim_all]):
                raise ValueError(
                    'All frequency sweeps must be in the same dimension.')
            dim = dim_all[0]
            freqs_all = np.array([sp['freq'] for sp in sp_all])

            # optimize the mod freq to lie in the middle of the overall
            # frequency range of this LO
            lo_freqs = 0.5 * (np.max(freqs_all, axis=0)
                              + np.min(freqs_all, axis=0))
            lo_freq_key = self._get_lo_freq_key(lo)
            self.sweep_points.add_sweep_parameter(param_name=lo_freq_key,
                                                  values=lo_freqs,
                                                  unit='Hz',
                                                  dimension=dim)
            # Add the sweep function for the joint LO sweep.
            self.sweep_functions_dict[lo_freq_key] = self.get_lo_swf(
                self.get_qubit(tasks[0]))
            # Indicate that no individual sweep functions are needed.
            self.sweep_functions_dict.update({
                task['prefix'] + 'freq': None for task in tasks})

            # We resolve the modulation frequency of the different qubits.
            qubits = []
            for task in tasks:
                qb = self.get_qubit(task)
                qubits.append(qb)
                freqs = task['sweep_points']['freq']
                mod_freqs = freqs - lo_freqs
                if all(mod_freqs - mod_freqs[0] == 0):
                    # mod freq is the same for all acquisitions
                    # As we require the step size between frequencies to be
                    # equal between all tasks sharing an LO, the modulation
                    # frequency is the same for every sweep point
                    self.temporary_values.append(
                        (self.get_mod_freq_param(qb), mod_freqs[0]))
                    task['mod_freq'] = mod_freqs[0]
                else:
                    # adds compatibility for different modulation frequencies
                    # in each sweep point. This will be used as soon as we add
                    # an allowed_lo_freqs feature to this method and support
                    # e.g. several qa or sg soft sweeps in one experiment.
                    self.sweep_points.add_sweep_parameter(
                        param_name=task['prefix'] + 'mod_frequency',
                        label=task['prefix'] + 'mod_frequency',
                        values=mod_freqs,
                        unit='Hz',
                        dimension=dim
                    )

    def group_tasks(self, **kw):
        """Fills the grouped_tasks dict with a list of tasks from
        preprocessed_task_list per LO found in the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            lo_name = self.get_lo_identifier(qb)
            if lo_name not in self.grouped_tasks:
                self.grouped_tasks[lo_name] = [task]
            else:
                self.grouped_tasks[lo_name] += [task]

    def check_all_freqs_per_lo(self, **kw):
        """Checks if all frequency sweeps assigned to one LO have the same
        increment in each step.
        """
        for lo, tasks in self.grouped_tasks.items():
            all_freqs = np.array([
                task['sweep_points']['freq'] for task in tasks])
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

    def get_lo_identifier(self, qb, **kw):
        """Get the LO identifier from the qubit

        Returns the LO identifier corresponding to the frequency that is
        swept, e.g. drive LO in qb spec and RO LO in resonator spec.

        Child classes have to implement this method and should not call this
        super method.

        Args:
            qb (QuDev_transmon): the qubit object

        Returns:
            str or tuple: the LO identifier

        Raises:
            NotImplementedError: In case the child class did not implement the
                method.
        """
        raise NotImplementedError('Child class has to implement'
                                  ' get_lo_identifier.')

    def get_mod_freq_param(self, qb, **kw):
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
                                  ' get_mod_freq_param.')

    def get_lo_swf(self, qb):
        """Returns a sweep function for a frequency sweep for the qubit.

        Child classes have to implement this method and should not call this
        super method.

        Args:
            qb (QuDev_transmon): Qubit for which the sweep function is
            returned.

        Raises:
            NotImplementedError: In case the child class did not implement the
                method.
        """
        raise NotImplementedError('Child class has to implement get_lo_swf.')

    def _fill_temporary_values(self):
        """adds additionally required qcodes parameter to the
        self.temporary_values list
        """
        # make sure that all triggers are set to the correct trigger separation
        if self.trigger_separation is None:
            return
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
        self.analysis = spa.MultiQubit_Spectroscopy_Analysis(**analysis_kwargs)
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
                    (not isinstance(qb, str) and qb.name == task['qb']):
                return task
        return None

    def _get_lo_freq_key(self, lo):
        return (lo if isinstance(lo, str)
                else '_'.join([f'{s}' for s in lo])) + '_freq'


class ResonatorSpectroscopy(MultiTaskingSpectroscopyExperiment):
    """Base class to be able to perform 1d and 2d feedline spectroscopies on one
    or multiple qubits and feedlines.

    For now the feedline for which the spectroscopy is performed is specified by
    passing a qubit on this feedline. FIXME: Part of refactoring to make
    compatible with more generic meas_objs (here: feedline might be a meas_obj)

    The kwarg drive of the base class gets the default value None for this
    class as this is the default for a resonator spectroscopy.

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
            parameter to set the fluxline voltage as item
            `'sweep_functions_dict'` in the task, e.g.:
            `task['sweep_functions_dict'] = {'volt': fluxlines_dict[qb.name]}`
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
        'ro_length':  dict(param_name='pulse_length', unit='s',
                      label=r'RO pulse length',
                      dimension=1),
    }
    default_experiment_name = 'ResonatorSpectroscopy'

    def __init__(self, task_list, sweep_points=None,
                 trigger_separation=5e-6, drive=None,
                 **kw):
        try:
            super().__init__(task_list, sweep_points=sweep_points,
                             trigger_separation=trigger_separation,
                             drive=drive,
                             segment_kwargs={'acquisition_mode':
                                             dict(sweeper='software')},
                             **kw)
            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

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

    def check_hard_sweep_compatibility(self, **kw):
        super().check_hard_sweep_compatibility()
        for acq_unit, tasks in self.grouped_tasks.items():
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
                qb = self.get_qubit(task)
                acq_instr = qb.instr_acq.get_instr()
                freqs = task['sweep_points']['freq']
                lo_freq, delta_f, _ = acq_instr.get_params_for_spectrum(
                    freqs, get_closest_lo_freq=(
                        lambda f, qb=qb: qb.get_closest_lo_freq(
                            f, operation='ro')))
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
                # adapt df kwargs to hard sweep
                self.df_kwargs['single_int_avg'] = False
        super().resolve_freq_sweep_points(**kw)
        # enforce programming baseband pulses if the freq sweep is
        # implemented as an IF sweep
        for lo, tasks in self.grouped_tasks.items():
            lo_freq_key = self._get_lo_freq_key(lo)
            if lo_freq_key not in self.sweep_functions_dict:
                assert len(tasks) == 1, (
                    f"An entry {lo_freq_key} should exist in "
                    f"self.sweep_functions_dict because tasks were grouped.")
                lo_freq_key = tasks[0]['prefix'] + 'freq'
            if getattr(self.sweep_functions_dict[lo_freq_key],
                       'includes_IF_sweep', False):
                for task in tasks:
                    task['mod_freq'] = 0

    def sweep_block(self, sweep_points, qb, prepend_pulse_dicts=None,
                    mod_freq=None, **kw):
        """This function creates the blocks for a single transmission
        measurement task.

        Args:
            sweep_points (SweepPoints): SweepPoints object
            qb (QuDev_transmon): target qubit
            prepend_pulse_dicts (dict): prepended pulses, see
                CircuitBuilder.block_from_pulse_dicts. Defaults to None.
            mod_freq (float or None): if not None, it overwrites the
                mod_frequency of the RO pulse provided by the operation_dict

        Returns:
            list of :class:`~pycqed.measurement.waveform_control.block.Block`s:
                List of blocks for the operation.
        """
        # create prepended pulses (pb)
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'ro_el'}}
        # create ro pulses (ro)
        ro = self.block_from_ops('ro', [f"RO {qb}"], pulse_modifs=pulse_modifs)
        if mod_freq is not None:
            ro.pulses[0]['mod_frequency'] = mod_freq

        # create ParametricValues from param_name in sweep_points
        # (e.g. "amplitude", "length", etc.)
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(param_name)

        # return all generated blocks (parallel_sweep will arrange them)
        return [pb, ro]

    def get_lo_identifier(self, qb, **kw):
        return qb.get_ro_lo_identifier()

    def get_mod_freq_param(self, qb, **kw):
        return qb.ro_mod_freq

    def get_lo_swf(self, qb):
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
            parameter to set the fluxline voltage as item
            `'sweep_functions_dict'` in the task, e.g.:
            `task['sweep_functions_dict'] = {'volt': fluxlines_dict[qb.name]}`
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

    def __init__(self, task_list, sweep_points=None,
                 pulsed=False,
                 trigger_separation=50e-6,
                 modulated=False,
                 **kw):
        try:
            # FIXME: Automatically detect modulated spectroscopy.
            self.modulated = modulated
            self.pulsed = pulsed
            drive = 'pulsed' if self.pulsed else 'continuous'
            drive += '_spec'
            drive += '_modulated' if self.modulated else ''
            self.default_experiment_name += '_pulsed' if self.pulsed \
                                                    else '_continuous'
            super().__init__(task_list, sweep_points=sweep_points,
                            drive=drive,
                            trigger_separation=trigger_separation,
                            segment_kwargs={'mod_config':{},
                                            'sine_config':{},
                                            'sweep_params':{},},
                            **kw)

            self.autorun(**kw)  # run measurement & analysis if requested in kw
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def preprocess_task(self, task, global_sweep_points,
                        sweep_points=None, **kw):
        preprocessed_task = super().preprocess_task(task, global_sweep_points,
                                                    sweep_points, **kw)
        qb = self.get_qubit(preprocessed_task)
        pulsar = qb.instr_pulsar.get_instr()
        awg_name = pulsar.get(f'{qb.ge_I_channel()}_awg')
        preprocessed_task['hard_sweep'] = pulsar.get(
            f'{awg_name}_use_hardware_sweeper', False)

        # Convenience feature to automatically use the LO power parameter to
        # sweep the spec_power in a soft_sweep spectroscopy
        prefix = preprocessed_task['prefix']
        if (preprocessed_task['sweep_points'].find_parameter('spec_power')
                    is not None
                and not preprocessed_task['hard_sweep']
                and prefix + 'spec_power' not in self.sweep_functions_dict):
            sf = self._get_power_param(qb, get_swf=True)
            if sf is not None:
                self.sweep_functions_dict[prefix + 'spec_power'] = sf

        return preprocessed_task

    def check_hard_sweep_compatibility(self, **kw):
        super().check_hard_sweep_compatibility()
        for awg_synth, tasks in self.grouped_tasks.items():
            if not any([task['hard_sweep'] for task in tasks]):
                continue
            elif len(tasks) > 1:
                raise ValueError(f"Currently only one task per synthesizer is "
                                 f"supported for hard sweeps. Qubits "
                                 f"{[task['qb'] for task in tasks]} share "
                                 f"synthesizer {awg_synth}.")

    def resolve_freq_sweep_points(self, **kw):
        """Configures potential hard_sweeps and afterwards calls super method
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            ch = qb.ge_I_channel()
            _, amp_from_power = self._get_power_param(qb)
            self.segment_kwargs['mod_config'][ch] = \
                dict(internal_mod=self.pulsed)
            self.segment_kwargs['sine_config'][ch] = \
                dict(continuous=not self.pulsed,
                     ignore_waveforms=not self.pulsed,
                     gains=tuple(amp_from_power(qb.spec_power()) * x
                                 for x in (0.0, 1.0, 1.0, 0.0)))
            if task['hard_sweep']:
                pulsar = qb.instr_pulsar.get_instr()
                freqs = task['sweep_points']['freq']
                center_freq, mod_freqs = pulsar.get_params_for_spectrum(
                    qb.ge_I_channel(), freqs)
                self.segment_kwargs['sweep_params'][f'{ch}_osc_sweep'] = \
                    mod_freqs
                pulsar.set(f'{ch}_centerfreq', center_freq)
                # adapt df kwargs to hard sweep
                self.df_name = 'int_avg_det'
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

        task = self.get_task(qb)

        # For pulsed and hard_sweep spectroscopies we add an empty spec pulse to
        # trigger the the drive/marker AWG and afterwards add the actual spec
        # pulse (either empty for a continuous hard sweep or the pulse for the
        # pulsed spectroscopy). This way we are able to implement a delay
        # between the trigger and the spec pulse that is needed in hard_sweeps
        # to set the osc. frequency.
        # FIXME: think about cleaner solution
        pulse_modifs = {'all': {'element_name': 'spec_el',
                                'length': 0, 'pulse_delay': 0}}
        empty_trigger = self.block_from_ops('spec', [f"Spec {qb}"],
                                    pulse_modifs=pulse_modifs)

        # add marker pulse in case we perform pulsed spectroscopy
        if self.pulsed:
            qubit = self.get_qubit(task)
            pulse_modifs = {'op_code=Spec': {'element_name': 'spec_el',}}
            if self._get_power_param(qubit)[0] is None:
                # No external LO, use pulse to set the spec power
                if sweep_points.find_parameter('spec_power') is not None:
                    amp = ParametricValue('spec_power', func=dbm_to_vp)
                else:
                    amp = dbm_to_vp(qubit.spec_power())
                pulse_modifs['op_code=Spec']['amplitude'] = amp
            spec = self.block_from_ops('spec', [f"Z0 {qb}", f"Spec {qb}"],
                                    pulse_modifs=pulse_modifs)
            # create ParametricValues from param_name in sweep_points
            for sweep_dict in sweep_points:
                for param_name in sweep_dict:
                    for pulse_dict in spec.pulses:
                        if param_name in pulse_dict:
                            pulse_dict[param_name] = ParametricValue(param_name)
            return [empty_trigger, spec, ro]
        else:
            if task is not None and task['hard_sweep']:
                # We need to add a pulse of length 0 to make sure the AWG channel is
                # programmed by pulsar and pulsar gets the hard sweep information.
                # The empty pulse is also used to trigger the SeqC code to set
                # the next osc. frequency. This needs to be done after the RO.
                return [ro, empty_trigger]
        return [ro]

    def get_lo_identifier(self, qb, **kw):
        return qb.get_ge_lo_identifier()

    def get_mod_freq_param(self, qb, **kw):
        return qb.ge_mod_freq

    def get_lo_swf(self, qb):
        if getattr(self, 'modulated', True):
            return swf.Offset_Sweep(
                sweep_function=qb.swf_drive_lo_freq(allow_IF_sweep=False),
                offset=-self.get_mod_freq_param(qb)(),
                name='Drive frequency',
                parameter_name='Drive frequency')
        else:
            return qb.swf_drive_lo_freq()

    def _get_power_param(self, qb, get_swf=False):
        if qb.instr_ge_lo() is not None:
            param = qb.instr_ge_lo.get_instr().power
            func = lambda p: p
            sf = param
        elif self.pulsed:
            # For pulsed spec without external LO, we currently implement a
            # pulse parameter sweep instead.
            param, sf = None, None
            func = lambda p: 1.0  # no scaling
        else:
            pulsar = qb.instr_pulsar.get_instr()
            ch = qb.ge_I_channel()
            param = pulsar.parameters[f'{ch}_direct_output_amp']
            amp_range = pulsar.get(f'{ch}_amp')
            func = lambda power, a=amp_range: dbm_to_vp(power) / a
            sf = swf.Transformed_Sweep(
                param, func, 'Spectrocopy power', unit='dBm')
        if get_swf:
            return sf
        else:
            return param, func

    def _fill_temporary_values(self):
        super()._fill_temporary_values()
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            if qb.instr_ge_lo() is None and not task['hard_sweep']:
                # SHFSG Soft Sweep settings:
                param, func = self._get_power_param(qb)
                if param is not None:
                    self.temporary_values.append(
                        (param, func(qb.spec_power())))
                pulsar = qb.instr_pulsar.get_instr()
                mod_freq = pulsar.parameters[
                    f'{qb.ge_I_channel()}_direct_mod_freq']
                self.temporary_values.append(
                    (mod_freq, self.get_mod_freq_param(qb)()))
            elif self.modulated:
                # HDAWG modulated spectroscopy settings:
                if task.get('mod_freq', False):
                    # FIXME: HDAWG specific code, should be moved to pulsar?
                    mod_freq = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_mod_freq'
                    ]
                    self.temporary_values.append((mod_freq,
                                                  task['mod_freq']))
                    amp = qb.instr_pulsar.get_instr().parameters[
                        f'{qb.ge_I_channel()}_direct_output_amp'
                    ]
                    self.temporary_values.append((amp,
                                                  dbm_to_vp(qb.spec_power())))
                else:
                    log.error('Task for modulated spectroscopy does not contain'
                              'mod_freq.')


class MultiStateResonatorSpectroscopy(ResonatorSpectroscopy):
    """Perform feedline spectroscopies for several initial states to determine
    the RO frequency with the highest contrast.

    While the experiment support parallelization, it only allows to have the
    same initial states for all qubits.

    The kwarg drive of the base class ResonatorSpectroscopy gets the default
    value 'timedomain' for this class to enable qubit drive pulses.

    Arguments:
        states (list[str], optional): List of strings specifying the initial
            states to be measured. Defaults to `["g", "e"]`.

    Compatible task dict keys:
        freqs: See :class:`ResonatorSpectroscopy' for details.

    Updates:
        qb.ro_freq: To the value maximizing the distance in the IQ plane between
            the first two states in task["states"]. If you want to make your RO
            more dicriminating between e & f you should pass states
            ["e", "f", ...] instead of e.g. ["g", "e", "f"].
    """
    default_experiment_name = 'MultiStateResonatorSpectroscopy'
    kw_for_sweep_points = dict(
        **ResonatorSpectroscopy.kw_for_sweep_points,
        states=dict(param_name='initialize', unit='',
                    label='qubit init state',
                    dimension=1),
    )

    def __init__(self, task_list, sweep_points=None,
                 trigger_separation=None, drive='timedomain',
                 states=("g", "e"), **kw):
        self.states = list(states)
        super().__init__(task_list, sweep_points=sweep_points,
                         trigger_separation=trigger_separation, drive=drive,
                         states=states,
                         **kw)

    def run_analysis(self, analysis_kwargs=None, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = spa.MultiQubit_AvgRoCalib_Analysis(**analysis_kwargs)
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
