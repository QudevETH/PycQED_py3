import numpy as np
import traceback

from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
import pycqed.measurement.sweep_functions as swf
from pycqed.measurement.waveform_control.block import ParametricValue
from pycqed.measurement.sweep_points import SweepPoints
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
        n_shots (int): Number of measurement repetitions, defaults to 2**15.
        states (str, list): States to perform SSRO on/train the classifier on.
            Can specify custom states for individual qubits using the format
            [[qb1_s1, qb2_s1, ..., qbn_s1], ..., [qb1_sm, ..., qbn_cm]]
            for a measurement of m segments/m sweep points in dimension 0.
            Defaults to 'ge'.
        multiplexed_ssro (bool): Prepares all possible state combinations.
            This will perform the respective multiplexed SSRO analysis.
        update_classifier (bool): Whether to update the qubit classifiers.
            Takes effect only if ``update=True`` or if ``run_update`` is called
            manually.
        update_ro_params (bool): Whether to update the readout pulse
            parameters. Takes effect only if a sweep was performed, and only if
            ``update=True`` or if ``run_update`` is called manually.
        sweep_preselection_ro_pulses (bool): Whether to sweep preselection
            readout pulses the same way as the (final) readout pulse.
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
        'ro_length': dict(param_name='pulse_length', unit='s',
                          label='RO Pulse Length', dimension=1),
        'acq_length': dict(param_name='acq_length', unit='s',
                           label='Acquisition Length', dimension=1)
    }

    def __init__(self, task_list=None, qubits=None, sweep_points=None,
                 n_shots=2**15, states='ge', multiplexed_ssro=False,
                 update_classifier=True, update_ro_params=True, **kw):
        try:
            # prepare task_list
            if task_list is None:
                if qubits is None:
                    raise ValueError('Please provide either '
                                     '"qubits" or "task_list"')
                # Create task_list from qubits
                if not isinstance(qubits, list):
                    qubits = [qubits]
                task_list = [{'qb': qb.name} for qb in qubits]
            for task in task_list:
                if 'qb' in task and not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name

            # use the methods for cal_point generation to generate the states
            cal_points = CalibrationPoints.multi_qubit(
                    range(len(task_list)), states, n_per_state=1,
                    all_combinations=multiplexed_ssro)
            states = cal_points.states

            # parse sweep_points
            sweep_points = SweepPoints(sweep_points)
            try:
                sweep_points.add_sweep_parameter(
                    'initialize', states, label='SSRO states', dimension=0)
            except AssertionError:
                log.warning(f'You tried adding {sweep_points.length()[0]} '
                            f'sweep points to dimension 0 which is used to '
                            f'sweep the {len(states)} init states. Did you '
                            f'mean to add the sweep points to dimension 1? '
                            f'Ignoring the sweep points of dimension 0.')
                sweep_points[0] = {'initialize': (states, '', 'SSRO states')}

            self.update_classifier = update_classifier
            self.update_ro_params = update_ro_params

            kw.update({'cal_states': (),  # we don't want any cal_states.
                       'df_kwargs': {'nr_shots': n_shots},
                       'df_name': 'int_log_det',
                       'data_to_fit': {},
                       'multiplexed_ssro': multiplexed_ssro,
                       })
            super().__init__(task_list, qubits=qubits, n_shots=n_shots,
                             sweep_points=sweep_points, **kw)

            # for compatibility with analysis, it has to be a 2D sweep
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            self.grouped_tasks = {}
            self.group_tasks(**kw)
            self._resolve_acq_length_sweep_points()
            if self.sweep_functions_dict:
                self.sweep_functions = []
                self.generate_sweep_functions()

            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

            self.exp_metadata.update({
                'rotate': False,  # for SSRO data should not be rotated
                'states': states,
                # set the main sweep point to be the initialisation states
                'main_sp': {t['qb']: 'initialize'
                            for t in self.preprocessed_task_list},
            })
            # adding cal points for analysis, no actual cal points are measured
            cal_points.qb_names = self.qb_names
            self.cal_points = cal_points

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def group_tasks(self, **kw):
        """Fills the grouped_tasks dict with a list of tasks from
        preprocessed_task_list per acquisition device found in
        the preprocessed_task_list.
        """
        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            acq_dev = qb.instr_acq()
            self.grouped_tasks.setdefault(acq_dev, [])
            self.grouped_tasks[acq_dev] += [task]

    def _resolve_acq_length_sweep_points(self):
        """
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
                for sp, qbn in ro_len_swept_acq_len_not:
                    sp.add_sweep_parameter('acq_length', all_length[0], 's',
                                           dimension=1,
                                           label='acquisition length (auto)')
                    log.warning(f" {qbn}: the readout pulse length is "
                                f"swept while the acquisition length is "
                                f"not. Automatically sweeping acquisition "
                                f"length from {all_length[0][0]:.3g}s to "
                                f"{all_length[0][-1]:.3g}s.")
            if len(all_length) == 0 and len(ro_len_swept_acq_len_not) != 0:
                # case acq_len not swept of acq_dev: choosing diff = acq_length
                # - ro_length of the first qb of the acq_dev where ro_len is
                # swept to calculate the acq_length sweep
                sp, qbn = ro_len_swept_acq_len_not[0]
                qb = self.get_qubits(qbn)[0][0]
                diff = qb.acq_length() - qb.ro_length()
                acq_len_sweep = sp.get_values('pulse_length') + diff
                for sp, qbn in ro_len_swept_acq_len_not:
                    sp.add_sweep_parameter('acq_length', acq_len_sweep, 's',
                                           dimension=1,
                                           label='acquisition length (auto)')
                    log.warning(f" {qbn}: the readout pulse length is "
                                f"swept while the acquisition length is "
                                f"not. Automatically sweeping acquisition "
                                f"length from {acq_len_sweep[0]:.3g}s to "
                                f"{acq_len_sweep[-1]:.3g}s.")
            if len(all_length) == 0 and len(ro_len_swept_acq_len_not) == 0:
                # case acq_len and ro_len not swept: nothing to do.
                continue

            # updating all_length
            all_length = np.array([
                t['sweep_points']['acq_length'] for t in tasks
                if t['sweep_points'].find_parameter('acq_length') is not None
            ])
            if np.ndim(all_length) == 1:
                all_length = [all_length]

            # check that all acq_lens are the same for the qbs of the acq_dev
            if not all([np.mean(abs(lengths - all_length[0]) / all_length[0])
                        < 1e-10 for lengths in all_length]):
                raise ValueError(
                    "The acq_length sweep points must be the same for all "
                    "qubits using the same acquisition device, but this is "
                    f"not the case for {acq_dev}.")
            sf = swf.AcquisitionLengthSweep(
                lambda: self.get_detector_function(acq_dev))
            self.sweep_functions_dict.update({
                tasks[0]['prefix'] + 'acq_length': sf})
            self.sweep_functions_dict.update({
                task['prefix'] + 'acq_length': None
                for task in tasks[1:]})

    def get_detector_function(self, acq_dev):
        """
        Returns the detector function of the corresponding acq_dev.

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
        """
        Adds a RO pulse to the block and replaces pulse parameters with
        ParametricValues.
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
            qb_names=self.meas_obj_names, t_start=self.timestamp,
            **analysis_kwargs)
        return self.analysis

    def run_update(self, **kw):
        if self.update_classifier:
            self.run_update_classifier()
        if self.update_ro_params:
            self.run_update_ro_params()

    def run_update_classifier(self):
        """ Updates qubit classifier.

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
        """ Updates RO pulse parameters if sweep was performed.

        Sets qubit readout pulse parameters to the values that yielded the
        highest fidelity.
        """
        pdd = self.analysis.proc_data_dict
        twoD = len(self.sweep_points.length()) == 2

        for task in self.preprocessed_task_list:
            qb = self.get_qubits(task['qb'])[0][0]
            # only update RO parameters if more than one sp:
            if twoD and pdd['n_dim_2_sweep_points'] > 1:
                best_indx = pdd['best_fidelity'][qb.name]['sweep_index']
                qb_sp_dict = pdd['sweep_points_2D_dict'][qb.name]
                if len(qb_sp_dict) > 0:
                    for k, v in qb_sp_dict.items():
                        param = qb.get_pulse_parameter(operation_name='RO',
                                                       argument_name=k)
                        if param is not None:
                            param(v[best_indx])
                            log.info(f"Set parameter {param.full_name} "
                                     f" to {v[best_indx]}")
                        else:
                            log.warning(' Could not set RO pulse param of '
                                        f'{qb.name} to {v[best_indx]}.')

class OptimalWeights(CalibBuilder):
    """
    Measures time traces for specified states and finds optimal integration
    weights. Applies filters to optimal integration weights if specified.

    For convenience, this class accepts the ``qubits`` argument in which case
    task_list is not needed and will be created from qubits.

    Note: This QE is not implemented for performing custom sweeps. Sweep
    dimension 0 is used for the time samples and sweep dimension 1 is used for
    initialising the qubit states specified in `states`.

    Args:
        task_list: See docstring of MultiTaskingExperiment.
        qubits: List of qubits on which traces should be measured
        sweep_points: See docstring of QuantumExperiment
        states (tuple, list, str): if str or tuple of single character strings,
            then interprets each letter as a state and does it on all qubits
             simultaneously. e.g. "ge" or ('g', 'e') --> measures all qbs
             in g then all in e.
             If list/tuple of tuples, then interprets the list as custom states:
             each tuple should be of length equal to the number of qubits
             and each state is calibrated individually. e.g. for 2 qubits:
             [('g', 'g'), ('e', 'e'), ('f', 'g')] --> qb1=qb2=g then qb1=qb2=e
             and then qb1 = "f" != qb2 = 'g'
        acq_length (float, None): length of timetrace to record
        acq_weights_basis (list): shortcut for analysis parameter.
            list of basis vectors used for computing the weights.
            (see TimetraceAnalysis). e.g. ["ge", "gf"] yields basis vectors e - g
            and f - g. If None, defaults to  ["ge", "ef"] when more than 2
            traces are passed to the analysis and to ['ge'] if 2 traces are
            measured.
        orthonormalize (bool): shortcut for analysis parameter. Whether to
            orthonormalize the optimal weights (see
            MultiQutrit_Timetrace_Analysis)
        **kw: keyword arguments. Can be used to provide keyword arguments to
            parallel_sweep/sweep_n_dim, preprocess_task_list, autorun, to the
            parent class and analysis class (see MultiQutrit_Timetrace_Analysis).
    """
    default_experiment_name = 'Timetrace'
    kw_for_task_keys = []

    def __init__(self, task_list=None, sweep_points=None, qubits=None,
                 states=('g', 'e'), acq_length=None, acq_weights_basis=None,
                 orthonormalize=True, soft_avg=30, n_shots=2**15, **kw):
        try:
            # prepare task_list
            if task_list is None:
                if qubits is None:
                    raise ValueError('Please provide either '
                                     '"qubits" or "task_list"')
                # Create task_list from qubits
                if not isinstance(qubits, list):
                    qubits = [qubits]
                # Create task_list from qubits
                task_list = [{'qb': qb.name} for qb in qubits]
            for task in task_list:
                if 'qb' in task and not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name

            # check, whether an acquisition device is used more than once
            acq_dev_names = np.array([qb.instr_acq() for qb in qubits])
            unique, counts = np.unique(acq_dev_names, return_counts=True)
            for u, c in zip(unique, counts):
                if c != 1:
                    log.warning(
                        f"{np.array(qubits)[acq_dev_names == u]} share the "
                        f"same acquisition device ({u}) and therefore their "
                        f"timetraces should not be measured simultaneously, "
                        f"except if you know what you are doing.")

            cal_points = CalibrationPoints.multi_qubit(
                    range(len(task_list)), states, n_per_state=1)
            states = cal_points.states
            sweep_points = SweepPoints(sweep_points)

            # get sampling rate from first qubit
            self.acq_sampling_rate = qubits[
                0].instr_acq.get_instr().acq_sampling_rate

            # infer acq_length if None
            if acq_length is None:
                acq_length = qubits[0].instr_acq.get_instr().acq_weights_n_samples / \
                             self.acq_sampling_rate
            self.acq_length = acq_length

            # get number of samples for all qubits
            samples = [(qb.instr_acq.get_instr(),
                        qb.instr_acq.get_instr().convert_time_to_n_samples(
                            acq_length, align_acq_granularity=True)) for qb
                       in qubits]
            # sort by number of samples
            samples.sort(key=lambda t: t[1])
            # generate sample times from first qb
            time_samples = samples[0][0].get_sweep_points_time_trace(
                acq_length, align_acq_granularity=True)

            # ensure no sweep_points in first dimension since used for sampling
            # times
            if len(sweep_points.length()) > 0 and sweep_points.length()[0] > 0:
                log.warning(f'You tried adding sweep points to dimension 0 '
                            f'which is used for the sampling times. '
                            f'Ignoring the sweep points of dimension 0.')
                sweep_points[0] = {}

            # add second dimension sweep_points (timetrace init states)
            try:
                sweep_points.add_sweep_parameter(
                    'initialize', states, label='timetrace states', dimension=1)
            except AssertionError:
                log.warning(f'You tried adding {sweep_points.length()[1]} '
                            f'sweep points to dimension 1 which is used to '
                            f'sweep the {len(states)} init states. '
                            f'Ignoring the sweep points of dimension 1.')
                sweep_points[1] = {'initialize': (states, '', 'timetrace states')}

            kw.update({'cal_states': (),  # we don't want any cal_states.
                       'df_name': 'inp_avg_det',
                       })
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # set temporary values for every qubit
            for task in self.preprocessed_task_list:
                qb = self.get_qubits(task['qb'])[0][0]
                self.temporary_values += [(qb.acq_length, acq_length),
                                          (qb.acq_weights_type, 'SSB'), ]
                if isinstance(n_shots, int):
                    self.temporary_values += [(qb.acq_averages, n_shots)]

            self.soft_avg = soft_avg

            # create sequences and mc_points with an empty dummy sweep block.
            self.sequences, _ = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block, **kw)

            self.mc_points[0] = time_samples

            self.exp_metadata.update({
                'states': states,
                'orthonormalize': orthonormalize,
                'acq_weights_basis': acq_weights_basis,
                'acq_sampling_rate': self.acq_sampling_rate
            })
            options_dict = {'orthonormalize': orthonormalize,
                            'acq_weights_basis': acq_weights_basis}
            kw.update(options_dict)

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()
        return

    def sweep_block(self, qb, **kw):
        """
        Creates sweep block for a qubit containing only the readout.
        """

        return [self.block_from_ops('timetrace_readout', [f'RO {qb}'])]

    def run_measurement(self, **kw):
        self._set_MC()
        # set temporary values for MC
        self.temporary_values += [(self.MC.cyclic_soft_avg, False),
                                  (self.MC.live_plot_enabled, False),
                                  (self.MC.program_only_on_change, True)]
        if isinstance(self.soft_avg, int):
            self.temporary_values += [(self.MC.soft_avg, self.soft_avg)]
        super().run_measurement(**kw)

    def run_analysis(self, analysis_kwargs=None, acq_weights_basis=None,
                     orthonormalize=True, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}

        options_dict = dict(orthonormalize=orthonormalize,
                            acq_weights_basis=acq_weights_basis)
        options_dict.update(analysis_kwargs.pop("options_dict", {}))

        self.analysis = tda.MultiQutrit_Timetrace_Analysis(
            t_start=self.timestamp,
            options_dict=options_dict, **analysis_kwargs)
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

    @classmethod
    def gui_kwargs(cls, device):
        # TODO
        d = super().gui_kwargs(device)
        # TODO
        return d

