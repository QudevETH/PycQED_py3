import numpy as np
import traceback

from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import ParametricValue
import pycqed.analysis_v2.timedomain_analysis as tda
import logging

log = logging.getLogger(__name__)


class MeasureSSRO(CalibBuilder):
    """ Performs a single shot readout experiment.

    Prepares the specified states and measures in single shot readout. The
    analysis performs a Gaussian mixture fit to calibrate the state classifier
    and outputs a SSRO probability assignment matrix. This class can also
    perform multiplexed SSRO on multiple qubits, i.e. measuring the state
    preparation fidelities for all combinations of states.

    This is a multitasking experiment, see docstrings of MultiTaskingExperiment
    and of CalibBuilder for general information.

    The segment for each task solely consists of state preparation pulses for
    all the states specified in the `states` keyword followed by a readout.

    Sweeps in dimension 0 and 1 will be interpreted as parameters of the
    readout pulse. 'acq_length' is accepted a special parameter as it specifies
    the acq_length set in the acquisition device. Note that sweep points in
    dimension 0 will be performed alongside the initialisation state sweep.

    For convenience, this class accepts the ``qubits`` argument in which case
    task_list is not needed and will be created from qubits.

    Note: Sweeps of the acq_length do not automatically adjust the holdoff
    between subsequent readouts, i.e. between potential preselection/feedback
    readouts and the final readout, which can lead to the measurement not
    progressing. TODO: Add checks to catch/avoid holdoff errors.

    Args:
        task_list: See docstring of MultiTaskingExperiment.
        qubits: List of qubits on which the SSRO measurement should be performed
        sweep_points: See docstring of QuantumExperiment
        n_shots (int): Number of measurement repetitions (default: 2**15)
        states (str, list): States to perform SSRO on/train the classifier on.
            Can specify custom states for individual qubits using the format
            [[qb1_s1, qb2_s1, ..., qbn_s1], ..., [qb1_sm, ..., qbn_cm]]
            for a measurement of m segments/m sweep points in dimension 0
             (default: 'ge')
        multiplexed_ssro (bool): Prepares all possible state combinations.
            This will perform the respective multiplexed SSRO analysis
            (default: False)
        update_classifier (bool): Whether to update the qubit classifiers.
            Takes effect only if ``update=True`` or if ``run_update`` is called
            manually (default: True)
        update_ro_params (bool): Whether to update the readout pulse
            parameters. Takes effect only if a sweep was performed, and only if
            ``update=True`` or if ``run_update`` is called manually (default:
            True)
        sweep_preselection_ro_pulses (bool): Whether to sweep preselection
            readout pulses the same way as the (final) readout pulse (default:
            True)
        preselection (bool): Whether to perform preselection (default: True)
        **kw: keyword arguments. Can be used to provide keyword arguments to
            parallel_sweep/sweep_n_dim, preprocess_task_list, autorun and to
            the parent class.
            The following keyword arguments will be copied as a key to tasks
            that do not have their own value specified:
            - `amps` as readout pulse amplitude sweep in dimension 1
            - `lengths` as readout pulse length sweep in dimension 1
    """
    default_experiment_name = 'SSRO_measurement'
    kw_for_task_keys = ['sweep_preselection_ro_pulses']
    kw_for_sweep_points = {
        'amps': dict(param_name='amplitude', unit='V',
                     label='RO Pulse Amplitude', dimension=1),
        'ro_lengths': dict(param_name='pulse_length', unit='s',
                           label='RO Pulse Length', dimension=1),
        'acq_lengths': dict(param_name='acq_length', unit='s',
                            label='Acquisition Length', dimension=1)
    }

    def __init__(self, task_list=None, qubits=None, sweep_points=None,
                 n_shots=2**15, states='ge', multiplexed_ssro=False,
                 update_classifier=True, update_ro_params=True,
                 preselection=True, **kw):
        try:
            qubits, task_list = self._parse_qubits_and_tasklist(qubits,
                                                                task_list)

            kw.setdefault('df_name', 'int_log_det')
            kw.update({'cal_states': ()})  # we don't want any cal_states.
            if 'df_kwargs' not in kw:
                kw['df_kwargs'] = {}
            kw['df_kwargs'].update({'nr_shots': n_shots})
            super().__init__(task_list, qubits=qubits, n_shots=n_shots,
                             sweep_points=sweep_points, **kw)

            self.update_classifier = update_classifier
            self.update_ro_params = update_ro_params
            self.multiplexed_ssro = multiplexed_ssro
            self.preselection = preselection

            # create cal_points for analysis, no cal points are measured
            # store in temporary variable because self.parallel_sweep accesses
            # self.cal_points to generate actual cal_points
            ana_cal_points = self._configure_sweep_points(states)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # grouping only has an impact for acq_length sweeps
            self.grouped_tasks = {}
            self.group_tasks(**kw)
            self._resolve_acq_length_sweep_points()
            if self.sweep_functions_dict:
                self.sweep_functions = []
                self.generate_sweep_functions()

            if preselection is not None:
                self.prep_params = self.get_prep_params()
                # force preselection for this measurement if desired by user
                if preselection:
                    self.prep_params['preparation_type'] = "preselection"
                else:
                    self.prep_params['preparation_type'] = "wait"

            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

            self.exp_metadata.update({
                'rotate': False,  # for SSRO data should not be rotated
                'data_to_fit': {},  # no data to fit for SSRO
                'multiplexed_ssro': self.multiplexed_ssro,
            })

            # adding cal points for analysis, no actual cal points are measured
            self.cal_points = ana_cal_points

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def _configure_sweep_points(self, states):
        """Configures self.sweep_points for SSRO

        Modifies self.sweep_points to perform the SSRO measurement, i.e.
        sweeps the init states in dimension 0. Checks whether other
        (incompatible) sweeps were added.

        It uses the methods for the generation of Calibration Points and
        returns the generated cal_points for it to be used in analysis
        (due to backwards-compatibility). No actual Calibration Points are
        measured.
        """
        # use the methods for cal_point generation to generate the states
        cal_points = CalibrationPoints.multi_qubit(
            [t['qb'] for t in self.task_list], states, n_per_state=1,
            all_combinations=self.multiplexed_ssro)
        self.states = cal_points.states

        # parse sweep_points
        try:
            self.sweep_points.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=0)
        except AssertionError:
            if 'initialize' in self.sweep_points.get_parameters():
                log.warning(f'You tried sweeping "initialize". '
                            f'This is already swept in dim 0. Ignoring the '
                            f'manually added sweep of "initialize".')
                self.sweep_points.remove_sweep_parameter('initialize')
            elif len(self.sweep_points.get_parameters(dimension=0)):
                log.warning(f'You tried adding {self.sweep_points.length()[0]}'
                            f' sweep points to dimension 0 which is used to '
                            f'sweep the {len(self.states)} init states. Did '
                            f'you mean to add the sweep points to dimension 1?'
                            f' Ignoring the sweep points of dimension 0.')
                for par in self.sweep_points.get_parameters(dimension=0):
                    self.sweep_points.remove_sweep_parameter(par)
            self.sweep_points.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=0)
        return cal_points

    def group_tasks(self, **kw):
        """Groups the tasks by the acquisition device.

        Fills the grouped_tasks dict with a list of tasks from
        preprocessed_task_list per acquisition device found in
        the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            acq_dev = qb.instr_acq()
            self.grouped_tasks.setdefault(acq_dev, [])
            self.grouped_tasks[acq_dev] += [task]

    def _resolve_acq_length_sweep_points(self):
        """Creates and resolves the acquisition length sweeps
        
        Implements the logic to ensure that the acq_length is swept with the
        ro_length and that the acq_lengths per instr_acq are identical.
        """
        for acq_dev, tasks in self.grouped_tasks.items():
            if not len(tasks):
                continue
            ro_len_swept_acq_len_not = []
            for t in tasks:
                sp = t['sweep_points']
                if sp.find_parameter('pulse_length') is not None and \
                        sp.find_parameter('acq_length') is None:
                    ro_len_swept_acq_len_not.append((sp, t['qb']))
            all_length = np.array([
                t['sweep_points']['acq_length'] for t in tasks
                if t['sweep_points'].find_parameter('acq_length') is not None
            ])
            if len(all_length) != 0 and len(ro_len_swept_acq_len_not) != 0:
                # case some qbs from acq_dev have acq_len swept, some not:
                # choose acq_len sweep points from qb where acq_len is swept
                acq_len_sweep = all_length[0]
            if len(all_length) == 0 and len(ro_len_swept_acq_len_not) != 0:
                # case acq_len not swept of acq_dev: choosing diff = acq_length
                # - ro_length of the first qb of the acq_dev where ro_len is
                # swept to calculate the acq_length sweep
                sp, qbn = ro_len_swept_acq_len_not[0]
                qb = self.get_qubits(qbn)[0][0]
                diff = qb.acq_length() - qb.ro_length()
                acq_len_sweep = sp.get_values('pulse_length') + diff
            if len(all_length) == 0 and len(ro_len_swept_acq_len_not) == 0:
                # case acq_len and ro_len not swept: nothing to do.
                continue
            for sp, qbn in ro_len_swept_acq_len_not:
                sp.add_sweep_parameter('acq_length', acq_len_sweep, 's',
                                       dimension=1,
                                       label='acquisition length (auto)')
                log.warning(f" {qbn}: the readout pulse length is "
                            f"swept while the acquisition length is "
                            f"not. Automatically sweeping acquisition "
                            f"length from {acq_len_sweep[0]:.3g}s to "
                            f"{acq_len_sweep[-1]:.3g}s.")

            # updating all_length
            all_length = np.array([
                t['sweep_points']['acq_length'] for t in tasks
                if t['sweep_points'].find_parameter('acq_length') is not None
            ])
            if (all_length == 0).any():
                raise ValueError(f"Choose non-zero acquisition length. "
                                 f"Encountered 0 in acq_dev {acq_dev} with "
                                 f"acquisition lengths {all_length}")
            # check that all acq_lens are the same for the qbs of the acq_dev
            if not all([np.mean(abs(lengths - all_length[0]) / all_length[0])
                        < 1e-10 for lengths in all_length]):
                raise ValueError(
                    "The acq_length sweep points must be the same for all "
                    "qubits using the same acquisition device, but this is "
                    f"not the case for {acq_dev}.")
            sf = swf.AcquisitionLengthSweep(
                lambda s=self, a=acq_dev: s.get_detector_function(a))
            # only add acq_length sweep fcn to the first qb of every acq_dev
            self.sweep_functions_dict.update({
                tasks[0]['prefix'] + 'acq_length': sf})
            self.sweep_functions_dict.update({
                task['prefix'] + 'acq_length': None
                for task in tasks[1:]})

    def get_detector_function(self, acq_dev):
        """Returns the detector function of the corresponding acq_dev.

        It is a helper function which is called by swf.AcquisitionLengthSweep
        if acq_length is being swept during the measurement. It is used to
        set the new acq_length.
        """
        for d in self.df.detectors:
            if d.acq_dev.name == acq_dev:
                return d
        raise KeyError(f'No detector function found for {acq_dev}.')

    def sweep_block(self, qb, sweep_points,
                    sweep_preselection_ro_pulses=True, **kw):
        """Creates the SSRO sweep block.
        
        Creates the sweep block with one RO pulse and replaces the RO pulse
        parameters with ParametricValues.
        """

        ro_block = self.block_from_ops('ssro_readout', [f'RO {qb}'])
        for sweep_dict in sweep_points:
            for param_name in sweep_dict:
                for pulse_dict in ro_block.pulses:
                    if param_name in pulse_dict:
                        pulse_dict[param_name] = ParametricValue(
                            param_name)
                        if sweep_preselection_ro_pulses:
                            self._prep_sweep_params[qb][
                                param_name] = param_name
        return [ro_block]

    def run_analysis(self, analysis_kwargs=None, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}

        self.analysis = tda.MultiQutrit_Singleshot_Readout_Analysis(
            t_start=self.timestamp, **analysis_kwargs)
        return self.analysis

    def run_update(self, **kw):
        if self.update_classifier:
            self.run_update_classifier()
        if self.update_ro_params:
            self.run_update_ro_params()

    def run_update_classifier(self):
        """Updates qubit classifier.

        Chooses the classifiers that yielded the highest fidelities if a sweep
        was performed.
        """
        pdd = self.analysis.proc_data_dict
        twoD = len(self.sweep_points.length()) == 2

        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            best_indx = pdd['best_fidelity'][qb.name]['sweep_index'] if twoD else 0
            pddap = pdd['analysis_params']

            classifier_params = pddap['classifier_params'][qb.name][best_indx]
            qb.acq_classifier_params().update(classifier_params)
            if 'state_prob_mtx_masked' in pddap:
                log.info(f'Updating classifier of {qb.name}')
                qb.acq_state_prob_mtx(
                    pddap['state_prob_mtx_masked'][qb.name][best_indx])
            else:
                log.warning('Measurement was not run with preselection. '
                            'state_prob_matx updated with non-masked one.')
                qb.acq_state_prob_mtx(
                    pddap['state_prob_mtx'][qb.name][best_indx])

    def run_update_ro_params(self):
        """Updates RO pulse parameters if sweep was performed.

        Sets qubit readout pulse parameters to the values that yielded the
        highest fidelity.
        """
        # only update RO parameters if more than one sp
        pdd = self.analysis.proc_data_dict
        if len(self.sweep_points.length()) != 2 or \
                pdd['n_dim_2_sweep_points'] <= 1:
            return

        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            best_indx = pdd['best_fidelity'][qb.name]['sweep_index']
            qb_sp_dict = pdd['sweep_points_2D_dict'][qb.name]
            for k, v in qb_sp_dict.items():
                param = qb.get_pulse_parameter(operation_name='RO',
                                               argument_name=k)
                if param is not None:
                    param(v[best_indx])
                    log.info(f"Set parameter {param.full_name} "
                             f" to {v[best_indx]}")
                else:
                    log.warning('Could not set RO pulse param of '
                                f'{qb.name} to {v[best_indx]}.')


class OptimalWeights(CalibBuilder):
    """Measures time traces and finds optimal integration weights. 
    
    Applies filters to optimal integration weights if specified.

    For convenience, this class accepts the ``qubits`` argument in which case
    task_list is not needed and will be created from qubits.

    Note: This QE is not implemented for performing custom sweeps. Sweep
    dimension 0 is used for the time samples and sweep dimension 1 is used for
    initialising the qubit states specified in ``states``.

    Args:
        task_list: See docstring of MultiTaskingExperiment.
        sweep_points: See docstring of QuantumExperiment
        qubits: List of qubits on which traces should be measured
        states (tuple, list, str): if str or tuple of single character strings,
            then interprets each letter as a state and does it on all qubits
             simultaneously. e.g. "ge" or ('g', 'e') --> measures all qbs
             in g then all in e.
             If list/tuple of tuples, then interprets the list as custom states:
             each tuple should be of length equal to the number of qubits
             and each state is calibrated individually. e.g. for 2 qubits:
             [('g', 'g'), ('e', 'e'), ('f', 'g')] --> qb1=qb2=g then qb1=qb2=e
             and then qb1 = "f" != qb2 = 'g'
             (default: ('g', 'e'))
        acq_averages (int): Number of measurement repetitions. The total number
            of time traces measured per state is acq_averages * soft_avg.
            (default: 2**15)
        soft_avg (int): Number of times the measurement is repeated
            and averaged over (default: 30)
        acq_length (float): length of timetrace to record (default: None)
        acq_weights_basis (list): shortcut for the corresponding analysis
            parameter, see MultiQutrit_Timetrace_Analysis for details.
        orthonormalize (bool): shortcut for analysis parameter. Whether to
            orthonormalize the optimal weights, see
            MultiQutrit_Timetrace_Analysis (default: True)
        **kw: keyword arguments. Can be used to provide keyword arguments to
            parallel_sweep/sweep_n_dim, preprocess_task_list, autorun, and to
            the parent class.
    """
    default_experiment_name = 'Timetrace'
    kw_for_task_keys = []

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 states=('g', 'e'), acq_averages=2**15, soft_avg=30,
                 acq_length=None, acq_weights_basis=None, orthonormalize=True,
                 **kw):
        try:
            qubits, task_list = self._parse_qubits_and_tasklist(
                qubits, task_list)

            kw.update({'cal_states': (),  # we don't want any cal_states.
                       'df_name': 'inp_avg_det'})

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.soft_avg = soft_avg
            self.orthonormalize = orthonormalize
            self.acq_averages = acq_averages
            self.acq_weights_basis = acq_weights_basis

            self._check_acq_devices()
            self._calculate_sampling_times(acq_length)
            self._configure_sweep_points(states)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # set temporary values for every qubit
            for task in self.preprocessed_task_list:
                qb = self.get_qubit(task)
                self.temporary_values += [(qb.acq_length, self.acq_length), ]
                if acq_averages is not None:
                    self.temporary_values += [(qb.acq_averages, acq_averages)]

            # create sequences and mc_points with an empty sweep block.
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

            self.mc_points[0] = self.time_samples

            self.exp_metadata.update({
                'states': self.states,
                'orthonormalize': self.orthonormalize,
                'acq_weights_basis': self.acq_weights_basis,
                'acq_sampling_rate': self.acq_sampling_rate
            })

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()
        return

    def get_qubit(self, task):
        """Shortcut to extract the qubit object from a task.
        """
        return self.get_qubits(task['qb'])[0][0]

    def _check_acq_devices(self):
        """Checks whether an acquisition device is used more than once.
        """
        qbs_in_tasks = [self.get_qubit(task) for task in self.task_list]
        acq_dev_names = np.array([qb.instr_acq() for qb in qbs_in_tasks])
        unique, counts = np.unique(acq_dev_names, return_counts=True)
        for u, c in zip(unique, counts):
            if c != 1:
                log.warning(
                    f"{np.array(qbs_in_tasks)[acq_dev_names == u]} share the "
                    f"same acquisition device ({u}) and therefore their "
                    f"timetraces should not be measured simultaneously, "
                    f"except if you know what you are doing.")
        return

    def _calculate_sampling_times(self, acq_length):
        """Calculates the sampling times of the acquisition device(s)

        Uses the properties of the acquisition device of the first qubit in the
        task list to calculate the sampling times.
        
        Populates the fields
         - self.acq_sampling_rate
         - self.acq_length
         - self.time_samples
        """
        # get sampling rate from qubits
        acq_sampling_rates = [self.get_qubit(task).instr_acq
                                  .get_instr().acq_sampling_rate
                              for task in self.task_list]
        unique_vals = np.unique(acq_sampling_rates)
        if len(unique_vals) > 1:
            raise NotImplementedError("Currently only supports one sampling "
                                      "rate across all qubits.")
        self.acq_sampling_rate = unique_vals[0]

        # infer acq_length if None from first qubit
        if acq_length is None:
            qb = self.get_qubit(self.task_list[0])
            acq_dev = qb.instr_acq.get_instr()
            if (n := acq_dev.acq_weights_n_samples) is None:
                raise ValueError(
                    'acq_length has to be provided because the acquisition '
                    'device does not have a default acq_weights_n_samples.')
            acq_length = n / self.acq_sampling_rate
        self.acq_length = acq_length

        # get number of samples for all qubits
        samples = [(self.get_qubit(task).instr_acq.get_instr(),
                    self.get_qubit(task).instr_acq.get_instr().
                    convert_time_to_n_samples(acq_length,
                                              align_acq_granularity=True))
                   for task in self.task_list]

        unique_vals = np.unique([s[1] for s in samples])
        if len(unique_vals) > 1:
            raise NotImplementedError("Currently only supports one number of "
                                      "samples across all qubits.")
        # generate sample times from first qb
        self.time_samples = samples[0][0].get_sweep_points_time_trace(
            self.acq_length, align_acq_granularity=True)

    def _configure_sweep_points(self, states):
        """Configures self.sweep_points for the OptimalWeights experiment

        Modifies self.sweep_points to perform the time trace measurement, i.e.
        with sampling times in dimension 0 and the init states in dimension 1.
        """
        cal_points = CalibrationPoints.multi_qubit(
            [t['qb'] for t in self.task_list], states, n_per_state=1)
        self.states = cal_points.states

        # ensure no sweep_points in dim 0 since used for sampling times
        if len(self.sweep_points.length()) > 0 and \
                self.sweep_points.length()[0] > 0:
            log.warning(f'You tried adding sweep points to dimension 0 '
                        f'which is used for the sampling times. '
                        f'Ignoring the sweep points of dimension 0.')
            for par in self.sweep_points.get_parameters(dimension=0):
                self.sweep_points.remove_sweep_parameter(par)

        # add dim 1 sweep_points (timetrace init states)
        try:
            self.sweep_points.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=1)
        except AssertionError:
            if 'initialize' in self.sweep_points.get_parameters():
                log.warning(f'You tried sweeping "initialize". '
                            f'This is already swept in dim 1. Ignoring the '
                            f'manually added sweep of "initialize".')
                self.sweep_points.remove_sweep_parameter('initialize')
            elif len(self.sweep_points.get_parameters(dimension=1)):
                log.warning(f'You tried adding {self.sweep_points.length()[1]} '
                            f'sweep points to dimension 1 which is used to '
                            f'sweep the {len(self.states)} init states. '
                            f'Ignoring the sweep points of dimension 1.')
                for par in self.sweep_points.get_parameters(dimension=1):
                    self.sweep_points.remove_sweep_parameter(par)
            self.sweep_points.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=1)


    def sweep_block(self, qb, **kw):
        """Creates the OptimalWeights sweep block.

        Creates sweep block for a qubit containing only the readout.
        """

        return [self.block_from_ops('timetrace_readout', [f'RO {qb}'])]

    def run_measurement(self, **kw):
        self._set_MC()
        # set temporary values for MC.
        self.temporary_values += [(self.MC.live_plot_enabled, False)]
        if self.soft_avg is not None:
            self.temporary_values += [(self.MC.soft_avg, self.soft_avg)]
            if self.soft_avg > 1:
                # cyclic_soft_avg = False: initialises the states in the order
                # ggg...eee... (instead of gegege...), if soft_avg > 1, has no
                # effect if soft_avg == 1.
                # program_only_on_change = True: speeds up the measurement by
                # not reprogramming the devices after every soft sweep point.
                self.temporary_values += [
                    (self.MC.cyclic_soft_avg, False),
                    (self.MC.program_only_on_change, True)]

        super().run_measurement(**kw)

    def run_analysis(self, **kw):
        analysis_kwargs = kw.pop('analysis_kwargs', {})
        self.analysis = tda.MultiQutrit_Timetrace_Analysis(
            t_start=self.timestamp, **analysis_kwargs)
        return self.analysis

    def run_update(self, **kw):
        for qb in self.qubits:
            log.info(f'Updating qubit weights of {qb.name}.')
            weights = self.analysis.proc_data_dict['analysis_params_dict'][
                'optimal_weights'][qb.name]
            if np.ndim(weights) == 1:
                # single channel
                qb.acq_weights_I(weights.real)
                qb.acq_weights_Q(weights.imag)
            elif np.ndim(weights) == 2 and len(weights) == 1:
                # single channels
                qb.acq_weights_I(weights[0].real)
                qb.acq_weights_Q(weights[0].imag)
            elif np.ndim(weights) == 2 and len(weights) == 2:
                # two channels
                qb.acq_weights_I(weights[0].real)
                qb.acq_weights_Q(weights[0].imag)
                qb.acq_weights_I2(weights[1].real)
                qb.acq_weights_Q2(weights[1].imag)
            else:
                log.warning(f"{qb.name}: Number of weight vectors > 2: "
                            f"{len(weights)}. Cannot update weights "
                            f"automatically.")
            qb.acq_weights_basis(
                self.analysis.proc_data_dict['analysis_params_dict'][
                    'optimal_weights_basis_labels'][qb.name])
            # We intentionally use the key 'centroids' instead of using
            # the same key 'means_' as used by the GMM classifier. This is
            # in order to allow storing both information independently and
            # to avoid unintentionally overwriting one kind of information
            # with the other.
            qb.acq_classifier_params().update({'centroids': np.array(
                self.analysis.proc_data_dict['analysis_params_dict']['means']
                [qb.name]
            )})
