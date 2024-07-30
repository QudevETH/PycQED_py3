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
        preselection (bool, None): Deprecated. Use reset_params instead.
            Whether to perform preselection (default: None).
            If True: Forces preselection.
            If False: Forces no preselection.
        **kw: keyword arguments. Can be used to provide keyword arguments to
            parallel_sweep/sweep_n_dim, preprocess_task_list, autorun and to
            the parent class.
            The following keyword arguments will be copied as a key to tasks
            that do not have their own value specified:
            - `amps` as readout pulse amplitude sweep in dimension 1
            - `lengths` as readout pulse length sweep in dimension 1
    """
    default_experiment_name = 'SSRO_measurement'
    kw_for_task_keys = ['sweep_preselection_ro_pulses',
                        'sweep_feedback_ro_pulses']
    kw_for_sweep_points = {
        'amps': dict(param_name='amplitude', unit='V',
                     label='RO Pulse Amplitude', dimension=1),
        'ro_lengths': dict(param_name='pulse_length', unit='s',
                           label='RO Pulse Length', dimension=1),
        'acq_lengths': dict(param_name='acq_length', unit='s',
                            label='Acquisition Length', dimension=1)
    }

    def __init__(self, task_list=None, qubits=None, sweep_points=None,
                 n_shots=10000, states='ge', multiplexed_ssro=False,
                 update_classifier=True, update_ro_params=True,
                 preselection=None, **kw):
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

            # create cal_points for analysis, no cal points are measured
            # store in temporary variable because self.parallel_sweep accesses
            # self.cal_points to generate actual cal_points
            ana_cal_points = self._configure_sweep_points(states)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # grouping only has an impact for acq_length sweeps
            self.grouped_tasks = {}
            self.group_tasks(**kw)
            self._resolve_acq_length_sweep_points(**kw)
            if self.sweep_functions_dict:
                self.sweep_functions = []
                self.generate_sweep_functions()

            if preselection is not None:
                log.warning(
                    "Using `preselection` keyword argument is deprecated and"
                    " will be removed in a future MR. Please use `reset_params"
                    "='preselection'` or `reset_params={'steps':[]}` instead.")
                self.reset_params = 'preselection' if preselection else False

            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

            self.exp_metadata.update({
                'rotate': False,  # for SSRO data should not be rotated
                'data_to_fit': {},  # no data to fit for SSRO
                'multiplexed_ssro': self.multiplexed_ssro,
            })

            # Adding cal_points for analysis by using that the cal_points are
            # in self._metadata_params. When running the measurement,
            # update_metadata is called, which writes self.cal_points to the
            # metadata, making them available for analysis. No actual cal
            # points are measured.
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

    def get_qubit(self, task):
        """Shortcut to extract the qubit object from a task.
        """
        return self.get_qubits(task['qb'])[0][0]

    def _resolve_acq_length_sweep_points(self,
                                         create_auto_acq_length_sweep=True,
                                         acq_length_sweep_auto_margin=200e-9,
                                         **kw):
        """Creates and resolves the acquisition length sweeps
        
        Implements the logic to ensure that the acq_length is swept with the
        ro_length and that the acq_lengths per instr_acq are identical. It
        also ensures, that the maximal acquisition length is not surpassed.

        Args:
            acq_length_sweep_auto_margin (float): If the acquisition length is
            not specified by the user and the readout pulse length l is longer
            than the acquisition length, the acquisition length is
            automatically set to min(l + auto_margin, max_acq_length).
            (Default: 200ns)
        """
        for acq_dev_name, tasks in self.grouped_tasks.items():
            if not len(tasks):
                continue

            acq_dev = self.dev.find_instrument(acq_dev_name)
            max_acq_length = (acq_dev.acq_weights_n_samples /
                              acq_dev.acq_sampling_rate)
            set_acq_length = max([self.get_qubit(t).acq_length() for t in tasks])

            # if one or more qbs sweeps the acq_length, then the max acq_length
            # of the qbs is used

            acq_len_sweeps = np.array([  # list of all acq_length sweeps
                t['sweep_points']['acq_length'] for t in tasks
                if t['sweep_points'].find_parameter('acq_length') is not None])
            ro_len_sweeps = np.array([  # list of all ro_length sweeps
                t['sweep_points']['pulse_length'] for t in tasks
                if t['sweep_points'].find_parameter('pulse_length') is not None])

            # to be filled if there is a ro_ or acq_length is swept
            acq_len_sweep = None

            if len(acq_len_sweeps != 0):  # at least one acq_length sweep
                if not all([np.all(acq_len_sweeps[0] == s)
                            for s in acq_len_sweeps]):
                    # different acqlen sweeps set, taking max.
                    log.warning(f"The acq_length sweep points are not the "
                                f"same for all qubits using {acq_dev_name}. "
                                f"Taking maximum over qbs.")
                acq_len_sweep = np.max(acq_len_sweeps, axis=0)
                if np.any(acq_len_sweep > max_acq_length):
                    log.warning(f"RO pulse lengths for qbs of {acq_dev_name} "
                                f"are longer than the maximal acquisition "
                                f"length of {max_acq_length:.3g}s.")
            elif len(ro_len_sweeps != 0) and create_auto_acq_length_sweep:
                # at least one ro_length swept with no acq_length sweeps
                max_ro_lengths = np.max(ro_len_sweeps, axis=0)

                acq_len_sweep = np.max(  # lower bound with ro_length
                    [np.ones_like(max_ro_lengths) * set_acq_length,
                     max_ro_lengths + acq_length_sweep_auto_margin], axis=0)

                if np.any(acq_len_sweep > set_acq_length):
                    # changed acq_length from default value
                    t1 = min(acq_len_sweep[0], max_acq_length)
                    t2 = min(acq_len_sweep[-1], max_acq_length)
                    log.warning(f" {acq_dev_name}: Some/all readout RO pulse "
                                f"lengths are swept while the acquisition length "
                                f"is not. Automatically sweeping acquisition "
                                f"length from {t1:.3g}s to {t2:.3g}s.")
                if np.any(max_ro_lengths > max_acq_length):
                    log.warning(f"RO pulse lengths for qbs of {acq_dev_name} "
                                f"are longer than the maximal acquisition "
                                f"length of {max_acq_length:.3g}s. Clipping "
                                f"acquisition length at maximum.")

            if acq_len_sweep is not None:  # last checks and update tasks
                acq_len_sweep = np.min(  # upper bound with max_acq_length
                    [acq_len_sweep,
                     np.ones_like(acq_len_sweep) * max_acq_length], axis=0)

                if np.any(acq_len_sweep/max_acq_length < 1e-10):
                    raise ValueError(f"Choose non-zero acquisition length. "
                                     f"Encountered 0 in {acq_dev_name} with "
                                     f"acquisition lengths {acq_len_sweep}.")

                for t in tasks:
                    sp = t['sweep_points']
                    if sp.find_parameter('acq_length') is None:
                        sp.add_sweep_parameter('acq_length', acq_len_sweep,
                                               's', dimension=1,
                                               label='acq length (auto)')
                    else:
                        sp.update_property(['acq_length', ],
                                           values=[acq_len_sweep, ])

                sf = swf.AcquisitionLengthSweep(
                    lambda s=self, a=acq_dev_name: s.get_detector_function(a))
                # only add acq_length sweep func to first qb of every acq_dev
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
                    sweep_preselection_ro_pulses=True,
                    sweep_feedback_ro_pulses=True, **kw):
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
                                'preselection_'+param_name] = param_name
                        if sweep_feedback_ro_pulses:
                            self._prep_sweep_params[qb][
                                'feedback_'+param_name] = param_name
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
        sps = self.sweep_points  # convenience shortcut

        # ensure no sweep_points in dim 0 since used for sampling times
        if len(sps.length()) > 0 and sps.length()[0] > 0:
            s = '", "'.join([str(k) for k in
                             sps.get_sweep_dimension(0).keys()])
            log.warning(f'You added the sweep points "{s}" to dimension 0 '
                        f'which is used for the sampling times. '
                        f'Ignoring the sweep points of dimension 0.')
            for par in sps.get_parameters(dimension=0):
                sps.remove_sweep_parameter(par)

        # add dim 1 sweep_points (timetrace init states)
        try:
            sps.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=1)
        except AssertionError:
            if 'initialize' in sps.get_parameters():
                pts = sps.get_sweep_params_description('initialize')[0]
                log.warning(f'You tried sweeping "initialize". '
                            f'This is already swept in dim 1 with the sweep '
                            f'points {self.states}. Ignoring the '
                            f'sweep of "initialize" with sweep points {pts}.')
                sps.remove_sweep_parameter('initialize')
            elif len(sps.get_parameters(dimension=1)):
                log.warning(f'You tried adding {sps.length()[1]} '
                            f'sweep points to dimension 1 which is used to '
                            f'sweep the {len(self.states)} init states. '
                            f'Ignoring the sweep points of dimension 1.')
                for par in sps.get_parameters(dimension=1):
                    sps.remove_sweep_parameter(par)
            sps.add_sweep_parameter(
                'initialize', self.states, label='initial state', dimension=1)


    def sweep_block(self, qb, **kw):
        """Creates the OptimalWeights sweep block.

        Creates sweep block for a qubit containing only the readout.
        """

        return [self.block_from_ops('timetrace_readout', [f'RO {qb}'])]

    def run_measurement(self, **kw):
        self._set_MC()
        # set temporary values for MC.
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
            if np.ndim(weights) == 2 and len(weights) == 1:
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
                log.warning(f"{qb.name}: Number of weight vectors != 2: "
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
