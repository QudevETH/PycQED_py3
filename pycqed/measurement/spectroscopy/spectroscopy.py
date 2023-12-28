import numpy as np
from collections import OrderedDict
from copy import copy
import traceback
from pycqed.utilities.general import assert_not_none, \
    configure_qubit_mux_readout
from pycqed.utilities.math import dbm_to_vp
from pycqed.measurement.calibration import two_qubit_gates as twoqbcal
from pycqed.measurement.waveform_control.block import ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.analysis_v2.spectroscopy_analysis as spa
from pycqed.utilities.general import temporary_value
import logging
log = logging.getLogger(__name__)


class MultiTaskingSpectroscopyExperiment(twoqbcal.CalibBuilder):
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
            Remark: if the Device object passed in the keyword argument dev
            has an attribute fluxlines_dict, the flux voltage parameters
            provided for the qubits in this dict will be added to the
            sweep_functions_dict of the task of the respective qubits as
            sweep functions for the sweep parameter 'volt' (to be potentially
            used by child classes, and to be ignored if no such sweep
            parameter exists).
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
        optimize_mod_freqs: (bool, default: False) Ignored for hard_sweeps or
            LOs supplying only one qb. If False, the mod_freq setting of the
            first qb on an LO (according to the ordering of the task_list)
            determines the LO frequency for all qubits on that LO. If True, the
            LO will be placed in the center of the band of frequencies of all
            qubits on that LO (minimizing the maximum absolute value of mod_freq
            over all qubits). Do not use in case of two qubits per LO, as it
            would lead to overlapping sidebands.
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
        self.sweep_functions = []
        self.sweep_points_pulses = SweepPoints(
            min_length=2 if self.force_2D_sweep else 1)
        """sweep points that are passed to sweep_n_dim and used to generate
        segments. This reduced set is introduce to prevent that a segment
        is generated for every frequency sweep point.
        """

        self.trigger_separation = trigger_separation
        self.grouped_tasks = {}

        self.preprocessed_task_list = self.preprocess_task_list(**kw)
        # new sweep points created from kws might increase the total number of
        # sweep dimensions: extend sweep_points_pulses if needed
        while len(self.sweep_points_pulses) < self._num_sweep_dims:
            self.sweep_points_pulses.add_sweep_dimension()

        self.group_tasks()
        self.check_all_freqs_per_lo()
        self.check_soft_sweep_compatibility()
        self.check_hard_sweep_compatibility()
        self.resolve_freq_sweep_points(**kw)
        self.generate_sweep_functions()

        self._fill_temporary_values()
        # temp value ensure that mod_freqs etc are set corretcly
        # We only use the temp vals of the qubit & dev at this point to
        # prevent issues with setting the trigger period on hardware and reading
        # it back immediately afterwards in the next tmp env.
        with temporary_value(*[t for t in self.temporary_values \
                               if t[0].instrument in self.qubits + [self.dev]]):
            self.update_operation_dict()

            self.sequences, _ = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                segment_kwargs=self.segment_kwargs, **kw)

        self.mc_points = [np.arange(n) for n in self.sweep_points.length()]

    def get_sweep_points_for_sweep_n_dim(self):
        return self.sweep_points_pulses

    def preprocess_task(self, task, global_sweep_points,
                        sweep_points=None, **kw):
        if self.dev and (fld := getattr(self.dev, 'fluxlines_dict', None)):
            if (param := fld.get(self.get_qubit(task).name)):
                task = copy(task)
                task.setdefault('sweep_functions_dict', {})
                task['sweep_functions_dict'].setdefault('volt', param)
        return super().preprocess_task(task, global_sweep_points,
                                       sweep_points, **kw)

    def resolve_freq_sweep_points(self, optimize_mod_freqs=False, **kw):
        """
        This function is called from the init of the class to resolve the
        frequency sweep points. The results are stored in properties of
        the object, which are then used in run_measurement. Aspects to be
        resolved include (if applicable):
        - (shared) LO freqs and (fixed or swept) IFs

        Args:
            optimize_mod_freqs: (bool, default: False) See class docstring.

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
            # For LOs supplying several qubits, the LO is set according to the
            # optimize_mod_freqs parameter, see docstring.
            sp_all = [task['sweep_points'] for task in tasks]
            dim_all = [sp.find_parameter('freq') for sp in sp_all]
            if any([dim != dim_all[0] for dim in dim_all]):
                raise ValueError(
                    'All frequency sweeps must be in the same dimension.')
            dim = dim_all[0]
            freqs_all = np.array([sp['freq'] for sp in sp_all])
            if optimize_mod_freqs:
                # optimize the mod freq to lie in the middle of the overall
                # frequency range of this LO
                lo_freqs = 0.5 * (np.max(freqs_all, axis=0)
                                + np.min(freqs_all, axis=0))
            else:
                # The mod freq of the qb of the first task in the list tasks
                # determines the LO freqs and thereby also the mod freqs of all
                # other qbs using the same LO.
                lo_freqs = sp_all[0]['freq'] - self.get_mod_freq_param(
                    self.get_qubit(tasks[0]))()
            lo_freq_key = self._get_lo_freq_key(lo)
            self.sweep_points.add_sweep_parameter(param_name=lo_freq_key,
                                                  values=lo_freqs,
                                                  unit='Hz',
                                                  dimension=dim)
            # Add the sweep function for the joint LO sweep.
            self.sweep_functions_dict[lo_freq_key] = self.get_lo_swf(
                self.get_qubit(tasks[0]), bare=True)
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

                    # We need to take control over the ro mod freq and therefore
                    # need to diable the `ro_fixed_lo_freq` in the qb.
                    if qb.ro_fixed_lo_freq() is not None:
                        self.temporary_values.append(
                            (qb.ro_fixed_lo_freq, None))
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

    def check_soft_sweep_compatibility(self, **kw):
        """Nothing to check right here. Child classes can extend functionality
        e.g. to only allow one soft sweep per feedline.
        """
        pass

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

    def get_lo_swf(self, qb, bare=False):
        """Returns a sweep function for a frequency sweep for the qubit.

        Child classes have to implement this method and should not call this
        super method.

        Args:
            qb (QuDev_transmon): Qubit for which the sweep function is
            returned.

            bare (bool): return the bare lo freq swf without any automatic
                offsets applied. Defaults to False.

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

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'][twoqbcal.MultiTaskingExperiment.__name__].update({
            'n_cal_points_per_state': (int, 0),
        })
        d['sweeping_parameters'].update({
            MultiTaskingSpectroscopyExperiment.__name__: {
                0: {
                    'freq': 'Hz',
                },
                1: {},
            }
        })
        return d


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
        ro_amp: List or np.array of amplitudes of the RO pulse.
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
        # We turn of repeat_ro (used by sweep_n_dim) by default. This ensures
        # all RO pulses that are different are programmed to the AWG.
        repeat_ro = kw.pop('repeat_ro', False)
        try:
            if task_list is None and (qubits := kw.get('qubits')):
                task_list = [dict(qb=qb) for qb in qubits]
            super().__init__(task_list, sweep_points=sweep_points,
                             trigger_separation=trigger_separation,
                             drive=drive,
                             segment_kwargs={'acquisition_mode':
                                             dict(sweeper='software')},
                             repeat_ro=repeat_ro,
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
                            f, operation='ro', fixed_lo=None)))
                # We need to disable the fixed lo feature in order to set the
                # ro/mod frequency independently in the temporary values.
                if qb.ro_fixed_lo_freq() is not None:
                    self.temporary_values.append(
                        (qb.ro_fixed_lo_freq, None))
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
                    )
                # adapt df kwargs to hard sweep
                self.df_kwargs['single_int_avg'] = False
                # FIXME: Set `enforce_pulsar_restart` to `True` to ensure that
                # RO AWG is restarted after each acquisition. Think
                # about a solution that only restarts the RO AWG.
                self.df_kwargs['enforce_pulsar_restart'] = True
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
                if len(tasks) > 1:
                    raise NotImplementedError('Measureing a soft resonator '
                                              'spectroscopy in parallel on one '
                                              'QA channel not supported at the '
                                              'moment.')
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

    def get_lo_swf(self, qb, bare=False):
        return qb.swf_ro_freq_lo(bare)

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

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'].update({
            ResonatorSpectroscopy.__name__: OrderedDict({
                'trigger_separation': (float, 5e-6)
            })
        })
        d['sweeping_parameters'].update({
            ResonatorSpectroscopy.__name__: {
                0: {},
                1: {
                    'volt': 'V',
                    'amplitude': 'V',
                    'pulse_length': 's',
                },
            }
        })
        return d

class FeedlineSpectroscopy(ResonatorSpectroscopy):
    """
    Performs a 1D resonator spectroscopy for a given feedline.

    The analysis looks for dips in the magnitude result. Two dips are searched
    for each qubit passed in 'feedline'. Each pair of dips is assigned to a
    qubit following the order of the current ro_freq. The two dips at the lowest
    frequency will be assigned to the qubit with the lowest ro_freq and so on.

    The sharpest dip of each pair is chosen to update the readout frequency
    of the corresponding qubit.

    Compatible task_dict keys:
        feedline: list of QuDev_transmons belonging to the same feedline. The
            first qubit of the list will be used to label the task. The qubits
            passed here will be used in the analysis to assign the found dips
            to each qubit.
            FIXME: should be refactored in future to be meas_obj (see parent
            class).
        freqs: List or np.array containing the drive frequencies of the
            spectroscopy measurement. (1. dim. sweep points)
    """

    kw_for_sweep_points = {
        'freqs':
            dict(param_name='freq',
                 unit='Hz',
                 label=r'RO frequency, $f_{RO}$',
                 dimension=0),
    }
    default_experiment_name = 'FeedlineSpectroscopy'

    def __init__(self, task_list, **kw):
        """
        Args:
            task_list: a list of tasks to be sent to the instrument.
        """
        self.qubits = []
        self.feedlines = []

        # Build the task_list for the parent class
        for task in task_list:
            # Choose the first qubit of each feedline as the reference
            task['qb'] = task['feedline'][0]
            self.qubits.append(task['qb'])
            # Remove feedline from task. This is necessary because the qubit
            # objects that it contains are QCoDeS instruments and they can not
            # be pickled when exp_metadata is deepcopied in the base class
            self.feedlines.append(task['feedline'])
            task.pop('feedline')

        # Sort the qubits and the feedlines consistently with what is already
        # done in a MultiTaskingExperiment. Namely, the tasks are sorted using
        # the name of the qubits.
        # Keeping track of this order ensures that each tasks can be analyzed
        # correctly. See the link for the sorting snippet
        # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
        self.qubits, self.feedlines = zip(*sorted(
            zip(self.qubits, self.feedlines), key=lambda pair: pair[0].name))
        # Convert tuples to lists again
        self.qubits = list(self.qubits)
        self.feedlines = list(self.feedlines)
        super().__init__(task_list, qubits=self.qubits, **kw)

    def run_update(self, **kw):
        """
        Updates the readout frequency of the qubits corresponding to the
        measured feedlines.
        """
        for qb_name, feedline in zip(self.qb_names, self.feedlines):
            for qb in feedline:
                try:
                    ro_freq = self.analysis.fit_res[qb_name][
                        f"{qb.name}_RO_frequency"]
                    qb.ro_freq(ro_freq)
                except KeyError:
                    log.warning(f"RO frequency of {qb.name} was not updated")

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Launches the analysis using ResonatorSpectroscopyAnalysis. Search
        for the dips corresponding to the readout resonator and the Purcell
        filter for each qubit of the feedline.
        The number of dips that the algorithm looks for is twice the number
        of qubits belonging to the feedline.

        Args:
            analysis_kwargs (dict): keyword arguments passed to the analysis
                class

        Returns:
            (ResonatorSpectroscopyAnalysis): the analysis object

        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        analysis_kwargs['feedlines_qubits'] = self.feedlines
        self.analysis = spa.FeedlineSpectroscopyAnalysis(**analysis_kwargs)
        return self.analysis


class ResonatorSpectroscopyFluxSweep(ResonatorSpectroscopy):
    """
    Performs a 2D resonator spectroscopy for a given qubit, sweeping the
    frequency and the voltage bias.

    The analysis looks for two dips in the magnitude result at each bias
    voltage. Afterwards, the extrema closest to zero voltage are chosen as the
    USS and LSS.

    If update is True, the qubits are parked at their sweet spot. The readout
    frequency is also updated accordingly. The width of the dips is used to
    decide which of the two dips should be used.

    Compatible task_dict keys:
        qb: see parent class
        freqs: see parent class
        volts: see parent class
    """

    kw_for_sweep_points = {
        'freqs':
            dict(param_name='freq',
                 unit='Hz',
                 label=r'RO frequency, $f_{RO}$',
                 dimension=0),
        'volts':
            dict(param_name='volt',
                 unit='V',
                 label=r'fluxline voltage',
                 dimension=1),
    }

    default_experiment_name = 'ResonatorSpectroscopyFluxSweep'

    def __init__(self, task_list, fluxlines_dict, **kw):
        self.fluxlines_dict = fluxlines_dict
        # Build the task_list for the parent class
        for task in task_list:
            task['sweep_functions_dict'] = {'volt': self.fluxlines_dict[task['qb']]}
        super().__init__(task_list, **kw)

    def run_update(self, **kw):
        """
        Updates each qubit bias voltage to the designated sweet spot.
        The RO frequency is also updated, choosing the dip with the minimum
        average width.

        The 'dac_sweet_spot' and 'V_per_phi0' keys of the
        'fit_ge_freq_from_dc_offset' dictionary are updated, to store the found
        value of the USS and LSS.

        The flux_parking() value is also updated depending on the order of the
        sweet spots,
        """

        for qb in self.qubits:
            if qb.flux_parking() != 0 and np.abs(qb.flux_parking()) != 0.5:
                log.warning((f"{qb.name}.flux_parking() is not 0, 0.5"
                             "nor -0.5. Nothing will be updated."))
            else:
                sweet_spot = self.analysis.fit_res[
                        qb.name][f'{qb.name}_sweet_spot']
                opposite_sweet_spot = self.analysis.fit_res[
                        qb.name][f'{qb.name}_opposite_sweet_spot']
                sweet_spot_RO_freq = self.analysis.fit_res[
                        qb.name][f'{qb.name}_sweet_spot_RO_frequency']

                self.fluxlines_dict[qb.name](sweet_spot)
                qb.ro_freq(sweet_spot_RO_freq)
                # Update V_per_phi0 key in each qubit. From this, it is possible
                # to retrieve the opposite sweet spot
                V_per_phi0 = np.abs(2 * (sweet_spot - opposite_sweet_spot))
                qb.fit_ge_freq_from_dc_offset()['V_per_phi0'] = V_per_phi0

                # Store the USS in 'dac_sweet_spot'
                if qb.flux_parking() == 0:
                    qb.fit_ge_freq_from_dc_offset(
                    )['dac_sweet_spot'] = sweet_spot
                elif qb.flux_parking() == 0.5:
                    qb.fit_ge_freq_from_dc_offset(
                    )['dac_sweet_spot'] = sweet_spot - V_per_phi0/2
                elif qb.flux_parking() == -0.5:
                    qb.fit_ge_freq_from_dc_offset(
                    )['dac_sweet_spot'] = sweet_spot + V_per_phi0/2

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Launches the analysis using ResonatorSpectroscopyFluxSweepAnalysis.
        At each voltage bias, search for two dips corresponding to the readout
        resonator and the Purcell filter. The found dips are used to determine
        the USS and LSS closest to zero. The average width of the dips is
        also calculated.

        Args:
            analysis_kwargs: keyword arguments passed to the analysis class

        Returns: analysis object

        """
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = spa.ResonatorSpectroscopyFluxSweepAnalysis(
            **analysis_kwargs)
        return self.analysis


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
            if task_list is None and (qubits := kw.get('qubits')):
                task_list = [dict(qb=qb) for qb in qubits]
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

    def check_soft_sweep_compatibility(self, **kw):
        super().check_soft_sweep_compatibility()
        for awg_synth, tasks in self.grouped_tasks.items():
            if not all([task['hard_sweep'] for task in tasks]) and \
              len(tasks) > 1 and isinstance(awg_synth, tuple):
                raise NotImplementedError(f"Currently only one task per "
                                          f"synthesizer is supported for soft "
                                          f"sweeps. Qubits "
                                          f"{[task['qb'] for task in tasks]} "
                                          f"share synthesizer {awg_synth}.")

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
                # We need to fix the ge lo frequency temporarily to a specific
                # value to ensure it is not changed by qb.prepare before the
                # measurement.
                self.temporary_values.append(
                    (qb.ge_fixed_lo_freq, center_freq))
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
                # The factor of 2 below is needed here because the spec pulse is
                # applied only to the I channel. This means that we get
                # half the amplitude after upconversion when the
                # outputamplitude node is set to 0.5, which is the reasonable
                # setting to use for digital IQ modulation in time-domain
                # experiments (output pulse has the programmed pulse
                # amplitude).
                if sweep_points.find_parameter('spec_power') is not None:
                    amp = ParametricValue('spec_power',
                                          func=lambda x: 2 * dbm_to_vp(x))
                else:
                    amp = 2 * dbm_to_vp(qubit.spec_power())
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

    def get_lo_swf(self, qb, bare=False):
        if getattr(self, 'modulated', True) and not bare:
            return swf.Offset_Sweep(
                sweep_function=qb.swf_drive_lo_freq(allow_IF_sweep=False),
                offset=-self.get_mod_freq_param(qb)(),
                name='Drive frequency',
                parameter_name='Drive frequency')
        else:
            return qb.swf_drive_lo_freq(allow_IF_sweep=(not bare))

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
                param, func, 'Spectroscopy power', unit='dBm')
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
                pulsar = qb.instr_pulsar.get_instr()
                mod_freq = pulsar.parameters[
                    f'{qb.ge_I_channel()}_direct_mod_freq']
                self.temporary_values.append(
                    (mod_freq, self.get_mod_freq_param(qb)()))
                param, func = self._get_power_param(qb)
                if param is not None:
                    self.temporary_values.append(
                        (param, func(qb.spec_power())))
            elif self.modulated:
                # HDAWG modulated spectroscopy settings:
                # FIXME: HDAWG specific code, should be moved to pulsar?
                mod_freq = qb.instr_pulsar.get_instr().parameters[
                    f'{qb.ge_I_channel()}_direct_mod_freq'
                ]
                self.temporary_values.append((
                    mod_freq,
                    task.get('mod_freq',
                                self.get_mod_freq_param(qb)())))
                amp = qb.instr_pulsar.get_instr().parameters[
                    f'{qb.ge_I_channel()}_direct_output_amp'
                ]
                self.temporary_values.append((amp,
                                              dbm_to_vp(qb.spec_power())))

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['sweeping_parameters'].update({
            QubitSpectroscopy.__name__: {
                0: {},
                1: {
                    'volt': 'V',
                    'spec_power': 'dBm',
                },
            }
        })
        return d


class QubitSpectroscopy1D(QubitSpectroscopy):
    """
    Performs a 1D QubitSpectroscopy and fits the result the result to extract
    the qubit frequency. See QubitSpectroscopy for allowed keywords.

    The analysis class is QubitSpectroscopy1DAnalysis, see it for the allowed
    keyword arguments for the analysis. The keyword arguments for the analysis
    can be passed as a dict via the argument 'analysis_kwargs' of
    QubitSpectroscopy1D.
    """

    kw_for_sweep_points = {
        'freqs':
            dict(param_name='freq',
                 unit='Hz',
                 label=r'Drive frequency, $f_{DR}$',
                 dimension=0),
    }
    default_experiment_name = 'QubitSpectroscopy1D'

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs the analysis to find the qubit frequency. The analysis class is
        QubitSpectroscopy1DAnalysis

        Args:
            analysis_kwargs (dict, optional): keyword arguments for
                QubitSpectroscopy1DAnalysis. See it for the allowed keyword
                arguments. Defaults to None.

        Returns:
            QubitSpectroscopy1DAnalysis: the analysis that was run.
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        self.analysis = spa.QubitSpectroscopy1DAnalysis(**analysis_kwargs)
        return self.analysis

    def run_update(self, **kw):
        """
        Updates the qubit frequency. If the ef transition was analyzed, updates
        both the ge and the ef frequency, otherwise updates only the ge
        frequency.
        """
        for qb in self.qubits:
            f0 = self.analysis.fit_res[qb.name].params['f0'].value
            qb.ge_freq(f0)
            if self.analysis.analyze_ef:
                f0_ef = 2 * self.analysis.fit_res[
                    qb.name].params['f0_gf_over_2'] - f0
                qb.ef_freq(f0_ef)


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

    def run_update(self, distance_metric='sum', **kw):
        """
        Updates the RO frequency of the qubit in each task with the value
        maximizing the g-e S21 distance, found in the analysis.
        : distance_metric (string): "sum" (default) or "{state0}-{state1}", e.g. "g-e".
            "sum" will take the readout frequency that maximizes the sum of
             the distances of all states in the IQ plane while "{state0}-{state1}"
             will take the frequency that maximizes the S21 distance for the listed
             pair of states.
        :param kw: keyword arguments

        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubit(task)
            pdd = self.analysis.proc_data_dict
            ro_freq = pdd['sweep_points_dict'][qb.name]['sweep_points'][
                pdd['projected_data_dict'][qb.name]['distance'][distance_metric][1] # (distances, argmax) -> index 1
            ]
            qb.set(f'ro_freq', ro_freq)
