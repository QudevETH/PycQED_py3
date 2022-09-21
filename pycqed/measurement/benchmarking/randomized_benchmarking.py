import numpy as np
import traceback
from copy import deepcopy
import random
from pycqed.analysis_v3.processing_pipeline import ProcessingPipeline
from pycqed.measurement.calibration.two_qubit_gates import MultiTaskingExperiment
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc
from pycqed.analysis_v3 import *
import logging
log = logging.getLogger(__name__)


class RandomizedBenchmarking(MultiTaskingExperiment):

    kw_for_sweep_points = {
        'cliffords': dict(param_name='cliffords', unit='',
                          label='Nr. Cliffords',
                          dimension=0),
        'nr_seeds,cliffords': dict(param_name='seeds', unit='',
                         label='Seeds', dimension=1,
                         values_func=lambda ns, cliffords: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(len(cliffords))]).T),
    }
    default_experiment_name = 'RB'

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 sweep_type=None, interleaved_gate=None, purity=False,
                 gate_decomposition='HZ', **kw):
        """
        Class to run and analyze the randomized benchmarking experiment on
        one or several qubits in parallel, using the single-qubit Clifford group

        Args:
            task_list (list of dicts): see CalibBuilder docstring
            sweep_points (SweepPoints class instance):
                Ex: [{'cliffords': ([0, 4, 10], '', 'Nr. Cliffords')},
                     {'seeds': (array([0, 1, 2, 3]), '', 'Nr. Seeds')}]
            qubits (list): QuDev_transmon class instances on which to run
                measurement
            sweep_type (dict): of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).
            interleaved_gate (string): the interleaved gate in pycqed notation
                (ex: X90, Y180 etc). Gate must be part of the Clifford group.
            purity (bool): indicates whether to do purity benchmarking
            gate_decomposition (string): what decomposition to use
                to translate the Clifford elements into applicable pulses.
                Possible choices are 'HZ' or 'XY'.
                See HZ_gate_decomposition and XY_gate_decomposition in
                measurement\randomized_benchmarking\clifford_decompositions.py

        Keyword arg:
            passed to parent class; see docstring there

            - tomo_pulses (list, default: ['I', 'X90', 'Y90']) list of op codes
                to use as tomography pulses if purity==True

            The following keyword arguments, if passed, will be used to
            construct sweep points:
            - cliffords: numpy array of sequence lengths (specified as
                total number of Cliffords in a sequence)
            - nr_seeds: int specifying how many random sequences of each length
                to measure
                - the sweep points values that will be constructed from this
                parameter will be an array or random numbers with nr_seeds
                columns and len(cliffords) rows, which represent the seeds for
                the random number generator used to sample a random RB
                sequence. In the block creation function, the random seeds in
                this 2D array are used to generate a random RB sequence (via
                to randomized_benchmarking_sequence(_new)).
                See also the class attribute kw_for_sweep_points.

        The following keys in a task are interpreted by this class in
        addition to the ones recognized by the parent classes:
            - cliffords (see description above)
            - seeds (see description above)

        Assumptions:
         - If nr_seeds and cliffords are specified in kw and they do not exist
          in the task_list, then all tasks will receive the same pulses!
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
         - in rb_block, it assumes only one parameter is being swept in the
         second sweep dimension (cliffords)
         - interleaved_gate and gate_decomposition should be the same for
         all qubits since otherwise the segments will have very different
         lengths for different qubits
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
        """
        try:
            condition1 = kw.get('nr_seeds', None) is None and \
                         kw.get('cliffords', None) is not None
            condition2 = kw.get('nr_seeds', None) is not None and \
                         kw.get('cliffords', None) is None
            if condition1 or condition2:
                # identical pulses on all tasks is enabled when both nr_seeds
                # and cliffords are specified as globals. The class breaks if
                # only one of them is global
                raise ValueError('If one of nr_seeds or cliffords is specified '
                                 'as a global parameter, the other must also '
                                 'be specified. This enables identical pulses '
                                 'on all tasks if you also remove them from the'
                                 'task list.')

            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cliffords': 0, 'seeds': 1}
            self.kw_for_sweep_points = deepcopy(self.kw_for_sweep_points)
            self.kw_for_sweep_points['nr_seeds,cliffords']['dimension'] = \
                self.sweep_type['seeds']
            self.kw_for_sweep_points['cliffords']['dimension'] = \
                self.sweep_type['cliffords']

            # tomo pulses for purity benchmarking
            self.tomo_pulses = kw.get('tomo_pulses', ['I', 'X90', 'Y90'])

            self.purity = purity
            self.interleaved_gate = interleaved_gate

            if self.interleaved_gate is not None:
                # kw_for_sweep_points must be changed to add the random seeds
                # for the IRB sequences. These are added as an extra sweep
                # parameter in dimension 1
                self.kw_for_sweep_points['nr_seeds,cliffords'] = [
                    dict(param_name='seeds', unit='',
                         label='Seeds', dimension=self.sweep_type['seeds'],
                         values_func=lambda ns, cliffords: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(len(cliffords))]).T),
                    dict(param_name='seeds_irb', unit='',
                         label='Seeds', dimension=self.sweep_type['seeds'],
                         values_func=lambda ns, cliffords: np.array(
                             [np.random.randint(0, 1e8, ns)
                              for _ in range(len(cliffords))]).T)]
            elif self.purity:
                # kw_for_sweep_points must be changed to repeat each seed 3
                # times (same seed, i.e. same sequence, for the 3 tomography
                # pulses)
                self.kw_for_sweep_points['nr_seeds,cliffords'] = [
                    dict(param_name='seeds', unit='',
                         label='Seeds', dimension=self.sweep_type['seeds'],
                         values_func=lambda ns, cliffords: np.array(
                             [np.repeat(np.random.randint(0, 1e8, ns), 3)
                              for _ in range(len(cliffords))]).T)]

            kw['cal_states'] = kw.get('cal_states', '')

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.experiment_name += f'_{gate_decomposition}'
            if purity:
                self.experiment_name += '_purity'
            self.identical_pulses = kw.get('nr_seeds', None) is not None and all([
                task.get('nr_seeds', None) is None for task in task_list])
            self.gate_decomposition = gate_decomposition
            self.preprocessed_task_list = self.preprocess_task_list(**kw)

            # Check if we can apply identical pulses on all qubits in task_list
            # Can only do this if they have identical cliffords array
            if self.identical_pulses:
                unique_clf_sets = np.unique([
                    self.sweep_points.get_sweep_params_property(
                        'values', self.sweep_type['cliffords'], k)
                    for k in self.sweep_points.get_sweep_dimension(
                        self.sweep_type['cliffords']) if
                    k.endswith('cliffords')], axis=0)
                if unique_clf_sets.shape[0] > 1:
                    raise ValueError('Cannot apply identical pulses. '
                                     'Not all qubits have the same Cliffords.'
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')

            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.rb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)
            if self.interleaved_gate is not None:
                seqs_irb, _ = self.sweep_n_dim(
                    self.sweep_points, body_block=None,
                    body_block_func_kw={'interleaved_gate':
                                            self.interleaved_gate},
                    body_block_func=self.rb_block, cal_points=self.cal_points,
                    ro_qubits=self.meas_obj_names, **kw)
                # interleave sequences
                self.sequences, self.mc_points = \
                    self.sequences[0].interleave_sequences(
                        [self.sequences, seqs_irb])
                self.exp_metadata['interleaved_gate'] = self.interleaved_gate
            self.exp_metadata['gate_decomposition'] = self.gate_decomposition
            self.exp_metadata['identical_pulses'] = self.identical_pulses

            self.add_processing_pipeline(**kw)
            self.autorun(**kw)
        #
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Base method to create the rb blocks. To be modified by children.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args: see the docstrings of children
        """
        pass

    def add_processing_pipeline(self, **kw):
        """
        Creates and adds the analysis processing pipeline to exp_metadata.
        """
        pass

    def run_analysis(self, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.

        Keyword args:
            keyword_arguments passed to analysis functions; see docstrings there
        """
        pass


class SingleQubitRandomizedBenchmarking(RandomizedBenchmarking):
    """
    Class for running the single qubit randomized benchmarking experiment on
    several qubits in parallel.
    """
    default_experiment_name = 'SingleQubitRB'

    def __init__(self, task_list, sweep_points=None, nr_seeds=None,
                 cliffords=None, **kw):
        """
        Init of the SingleQubitRandomizedBenchmarking class.

        Args:
            nr_seeds (int): the number of times the Clifford group should be
                sampled for each Clifford sequence length.
            cliffords(list/array): integers specifying the number of
                cliffords to apply.

        Keyword args:
            passed to parent class
            interleaved_gate is used here to update the experiment_name.
            dim_hilbert of 2 is hardcoded into the kw

        See docstring of RandomizedBenchmarking for more details.
        """

        if kw.get('interleaved_gate', None) is not None:
            self.experiment_name = 'SingleQubitIRB'

        for task in task_list:
            if 'qb' not in task:
                raise ValueError('Please specify "qb" in each task in '
                                 '"task_list."')
            if not isinstance(task['qb'], str):
                task['qb'] = task['qb'].name
            if 'prefix' not in task:
                task['prefix'] = f"{task['qb']}_"

        kw['dim_hilbert'] = 2
        super().__init__(task_list, sweep_points=sweep_points,
                         nr_seeds=nr_seeds, cliffords=cliffords, **kw)

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Creates simultaneous blocks of RB pulses for each task in
        preprocessed_task_list.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args:
            interleaved_gate (see docstring of parent class)
        """
        interleaved_gate = kw.get('interleaved_gate', None)
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list
        for i, task in enumerate(tl):
            param_name = 'seeds' if interleaved_gate is None else 'seeds_irb'
            seed_idx = sp1d_idx if self.sweep_type['seeds'] == 0 else sp2d_idx
            clf_idx = sp1d_idx if self.sweep_type['cliffords'] == 0 else sp2d_idx
            seed = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seeds'], param_name)[
                seed_idx, clf_idx]
            clifford = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['cliffords'], 'cliffords')[clf_idx]
            cl_seq = rb.randomized_benchmarking_sequence(
                clifford, seed=seed,
                interleaved_gate=interleaved_gate)
            pulse_list = rb.decompose_clifford_seq(
                cl_seq, gate_decomp=self.gate_decomposition)
            if self.purity:
                idx = sp1d_idx if self.sweep_type['seeds'] == 0 else sp2d_idx
                pulse_list += [self.tomo_pulses[idx % 3]]
            pulse_op_codes_list += [pulse_list]

        rb_block_list = [self.block_from_ops(
            f"rb_{task['qb']}", [f"{p} {task['qb']}" for p in
                                 pulse_op_codes_list[0 if self.identical_pulses
                                 else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end',
                                        destroy=self.fast_mode)


class TwoQubitRandomizedBenchmarking(RandomizedBenchmarking):
    """
    Class for running the two-qubit randomized benchmarking experiment on
    several pairs of qubits in parallel.
    """
    default_experiment_name = 'TwoQubitRB'

    def __init__(self, task_list, sweep_points=None, nr_seeds=None,
                 cliffords=None, max_clifford_idx=11520, **kw):
        """
        Each task in task_list corresponds to a qubit pair, which is specified
        with the keys 'qb_1' and 'qb2.'

        Args:
            nr_seeds (int): the number of times the Clifford group should be
                sampled for each Clifford sequence length.
            cliffords (list/array): integers specifying the number of
                cliffords to apply.
            max_clifford_idx (int): allows to restrict the two qubit
                Clifford that is sampled. Set to 24**2 to only sample the tensor
                product of 2 single qubit Clifford groups.

        Keyword args:
            passed to parent class
            interleaved_gate is used here to update the experiment_name.
            gate_decomposition is assigned to tqc.gate_decomposition.
            dim_hilbert of 4 is hardcoded into the kw.

        See docstring of RandomizedBenchmarking for the other parameters.
        """
        if kw.get('purity', False):
            raise NotImplementedError('Purity benchmarking is not implemented '
                                      'for 2QB RB. Set "purity=False."')

        self.max_clifford_idx = max_clifford_idx
        tqc.gate_decomposition = rb.get_clifford_decomposition(
            kw.get('gate_decomposition', 'HZ'))

        for task in task_list:
            for k in ['qb_1', 'qb_2']:
                if not isinstance(task[k], str):
                    task[k] = task[k].name
            if 'prefix' not in task:
                task['prefix'] = f"{task['qb_1']}{task['qb_2']}_"
        kw['for_ef'] = kw.get('for_ef', True)
        if kw.get('interleaved_gate', None) is not None:
            self.experiment_name = 'TwoQubitIRB'
        kw['dim_hilbert'] = 4
        super().__init__(task_list, sweep_points=sweep_points,
                         nr_seeds=nr_seeds, cliffords=cliffords, **kw)

    def guess_label(self, **kw):
        """
        Create default measurement label and assign to self.label.
        """
        suffix = [''.join([task['qb_1'], task['qb_2']])
                  for task in self.task_list]
        suffix = '_'.join(suffix)
        if self.label is None:
            if self.interleaved_gate is None:
                self.label = f'{self.experiment_name}_' \
                             f'{self.gate_decomposition}_{suffix}'
            else:
                self.label = f'{self.experiment_name}_{self.interleaved_gate}_' \
                             f'{self.gate_decomposition}_{suffix}'

    def rb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Creates simultaneous blocks of RB pulses for each task in
        preprocessed_task_list.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args:
            interleaved_gate (see docstring of parent class)
        """
        interleaved_gate = kw.get('interleaved_gate', None)
        rb_block_list = []
        for i, task in enumerate(self.preprocessed_task_list):
            param_name = 'seeds' if interleaved_gate is None else 'seeds_irb'
            seed_idx = sp1d_idx if self.sweep_type['seeds'] == 0 else sp2d_idx
            clf_idx = sp1d_idx if self.sweep_type['cliffords'] == 0 else sp2d_idx
            seed = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seeds'], param_name)[
                seed_idx, clf_idx]
            clifford = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['cliffords'], 'cliffords')[clf_idx]
            cl_seq = rb.randomized_benchmarking_sequence_new(
                clifford, number_of_qubits=2, seed=seed,
                max_clifford_idx=kw.get('max_clifford_idx',
                                        self.max_clifford_idx),
                interleaving_cl=interleaved_gate)

            qb_1 = task['qb_1']
            qb_2 = task['qb_2']
            seq_blocks = []
            single_qb_gates = {qb_1: [], qb_2: []}
            for k, idx in enumerate(cl_seq):
                self.timer.checkpoint("rb_block.seq.iteration.start")
                pulse_tuples_list = tqc.TwoQubitClifford(
                    idx).gate_decomposition
                for j, pulse_tuple in enumerate(pulse_tuples_list):
                    if isinstance(pulse_tuple[1], list):
                        seq_blocks.append(
                            self.simultaneous_blocks(
                                f'blk{k}_{j}', [
                            self.block_from_ops(f'blk{k}_{j}_{qbn}', gates)
                                    for qbn, gates in single_qb_gates.items()],
                            destroy=self.fast_mode))
                        single_qb_gates = {qb_1: [], qb_2: []}
                        seq_blocks.append(self.block_from_ops(
                            f'blk{k}_{j}_cz',
                            f'{kw.get("cz_pulse_name", "CZ")} {qb_1} {qb_2}',
                            ))
                    else:
                        qb_name = qb_1 if '0' in pulse_tuple[1] else qb_2
                        pulse_name = pulse_tuple[0]
                        single_qb_gates[qb_name].append(
                            pulse_name + ' ' + qb_name)
                self.timer.checkpoint("rb_block.seq.iteration.end")

            seq_blocks.append(
                self.simultaneous_blocks(
                    f'blk{i}', [
                        self.block_from_ops(f'blk{i}{qbn}', gates)
                        for qbn, gates in single_qb_gates.items()],
                    destroy=self.fast_mode))
            rb_block_list += [self.sequential_blocks(
                f'rb_block{i}', seq_blocks, destroy=self.fast_mode)]

        return self.simultaneous_blocks(
            f'sim_rb_{sp2d_idx}_{sp1d_idx}', rb_block_list, block_align='end',
            destroy=self.fast_mode)


class SingleQubitXEB(MultiTaskingExperiment):
    """
    Class for running the single qubit cross-entropy benchmarking experiment on
    several qubits in parallel.
    """
    kw_for_sweep_points = {
        'cycles': dict(param_name='cycles', unit='',
                          label='Nr. Cycles',
                          dimension=0),
        'nr_seqs,cycles': dict(param_name='z_rots', unit='',
                              label='$R_z$ angles, $\\phi$', dimension=1,
                              values_func=lambda ns, cycles: [[
                                  list(np.random.uniform(0, 2, nc) * 180)
                                  for nc in cycles] for _ in range(ns)]),
    }
    default_experiment_name = 'SingleQubitXEB'

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, sweep_type=None,
                 init_rotation='X90', **kw):
        """
        Init of the SingleQubitXEB class.
        The experiment consists of applying
        [[Ry - Rz(theta)] * nr_cycles for nr_cycles in cycles] nr_seqs times,
        with random values of theta each time.

        Args:
            task_list (list): see CalibBuilder docstring
            sweep_points (SweepPoints instance):
                Ex: [{'cycles': ([0, 4, 10], '', 'Nr. Cycles')},
                     {'nr_seqs': (array([0, 1, 2, 3]), '', 'Nr. Sequences')}]
            qubits (list): QuDev_transmon class instances on which to run
                measurement
            nr_seqs (int): the number of times to apply a random
                iteration of a sequence consisting of nr_cycles cycles.
                If nr_seqs is specified and it does not exist in the task_list,
                then all qubits will receive the same pulses provided they have
                the same cycles array.
            cycles (list/array): integers specifying the number of
                [Ry - Rz(theta)] cycles to apply.
            sweep_type (dict): of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).
            init_rotation (str): the preparation pulse name

        Keyword args:
            passed to CalibBuilder; see docstring there

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
        """
        try:

            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cycles': 0, 'seqs': 1}
            self.kw_for_sweep_points['nr_seqs,cycles']['dimension'] = \
                self.sweep_type['seqs']
            self.kw_for_sweep_points['cycles']['dimension'] = \
                self.sweep_type['cycles']
            kw['cal_states'] = kw.get('cal_states', '')

            for task in task_list:
                if 'qb' not in task:
                    raise ValueError('Please specify "qb" in each task in '
                                     '"task_list."')
                if not isinstance(task['qb'], str):
                    task['qb'] = task['qb'].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb']}_"

            # if cycles are not added to each task,
            # self.preprocess_task_list(**kw) will fail because
            # kw_for_sweep_points['nr_seqs,cycles'] requires cycles
            if cycles is not None:
                for task in task_list:
                    task['cycles'] = cycles

            if nr_seqs is not None:  # identical pulses on all qubits
                cycles_list = [''] * len(task_list)
                for i, task in enumerate(task_list):
                    if 'cycles' not in task:
                        raise KeyError('Please specify "cycles" either in '
                                       'the task_list or as input '
                                       'parameter to class init.')
                    cycles_list[i] = task['cycles']
                if np.unique(cycles_list, axis=0).shape[0] > 1:
                    # different qubits have different nr cycles; cannot be if
                    # user wants identical pulses on all qubits
                    raise ValueError('Identical pulses on all qubits requires '
                                     'identical cycles arrays. '
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')
                else:
                    cycles = cycles_list[0]

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)
            self.init_rotation = init_rotation
            self.identical_pulses = nr_seqs is not None and all([
                task.get('nr_seqs', None) is None for task in task_list])
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.xeb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)

            self.exp_metadata['identical_pulses'] = self.identical_pulses
            self.exp_metadata['init_rotation'] = self.init_rotation

            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def xeb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Creates simultaneous blocks of XEB pulses for each task in
        preprocessed_task_list.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args:
            to allow pass through kw even if it contains entries that are
            not needed
        """
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list
        for i, task in enumerate(tl):
            seq_idx = sp1d_idx if self.sweep_type['seqs'] == 0 else sp2d_idx
            nrcyc_idx = sp1d_idx if self.sweep_type['cycles'] == 0 else sp2d_idx
            z_angles = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seqs'], 'z_rots')[seq_idx][nrcyc_idx]
            l = [['Y90', f'Z{zang}'] for zang in z_angles]
            # flatten l, prepend init pulse, append to pulse_op_codes_list
            pulse_op_codes_list += [[self.init_rotation] + [e1 for e2 in l for e1 in e2]]

        rb_block_list = [self.block_from_ops(
            f"rb_{task['qb']}", [f"{p} {task['qb']}" for p in
                                 pulse_op_codes_list[0 if
                                 self.identical_pulses else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end',
                                        destroy=self.fast_mode)


class TwoQubitXEB(MultiTaskingExperiment):
    """
    Class for running the two-qubit cross-entropy benchmarking experiment on
    several pairs of qubits in parallel.
    Implementation as in https://www.nature.com/articles/s41567-018-0124-x.
    """
    kw_for_sweep_points = {'cycles': dict(param_name='cycles', unit='',
                                          label='Nr. Cycles', dimension=0),
                           'nr_seqs,cycles,cphase': dict(
                               param_name='gateschoice', unit='',
                               label='cycles gates', dimension=1,
                               values_func='paulis_gen_func'),
                           }
    kw_for_task_keys = ['cphase']
    default_experiment_name = 'TwoQubitXEB'

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 parametric=False, nr_seqs=None, cycles=None, sweep_type=None,
                 **kw):
        """
        Init of the TwoQubitXEB class.
        The experiment consists of applying
        [[Rq - CZ] * nr_cycles for nr_cycles in cycles] nr_seqs times,
        where Rq denotes a pair of parallel single qubit gates sampled randomly
        in each cycles from [X90, Y90, Z45], according to the following rules
        (copied from the reference above):
        "the first one-qubit gate for each qubit after the initial cycle of
        Hadamard gates is always a T gate; and we place a one-qubit gate only
        in the next cycle after a CZ gate in the same qubit."

        Args:
            task_list (list): see CalibBuilder docstring
            sweep_points (SweepPoints instance):
                Ex: [{'cycles': ([0, 4, 10], '', 'Nr. Cycles')},
                     {'nr_seqs': (array([0, 1, 2, 3]), '', 'Nr. Sequences')}]
            qubits (list): QuDev_transmon class instances on which to run
                measurement
            parametric (bool): whether to do parametric C-phase gates, with
                a random angle in each cycle
            nr_seqs (int): the number of times to apply a random
                iteration of a sequence consisting of nr_cycles cycles.
                If nr_seqs is specified and it does not exist in the task_list,
                THEN ALL TASKS WILL RECEIVE THE SAME PULSES provided they have
                the same cycles array.
            cycles (list/array): integers specifying the number of
                random cycles to apply in a sequence.
            sweep_type (dict): of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).

        Keyword args:
            passed to CalibBuilder; see docstring there
            cphase (float; default: 180): value of the C-phase gate angle
                in degrees

        Assumptions:
         - assumes there is one task for each qubit. If task_list is None, it
          will internally create it.
        """
        try:
            self.parametric = parametric
            self.sweep_type = sweep_type
            if self.sweep_type is None:
                self.sweep_type = {'cycles': 0, 'seqs': 1}
            self.kw_for_sweep_points['nr_seqs,cycles,cphase']['dimension'] = \
                self.sweep_type['seqs']
            self.kw_for_sweep_points['cycles']['dimension'] = \
                self.sweep_type['cycles']
            kw['cal_states'] = kw.get('cal_states', '')

            for task in task_list:
                for k in ['qb_1', 'qb_2']:
                    if k not in task:
                        raise ValueError('Please specify "{k}" in each task in '
                                         '"task_list."')
                    if not isinstance(task[k], str):
                        task[k] = task[k].name
                if 'prefix' not in task:
                    task['prefix'] = f"{task['qb_1']}{task['qb_2']}_"

            # if cycles are not added to each task,
            # self.preprocess_task_list(**kw) will fail because
            # kw_for_sweep_points['nr_seqs,cycles'] requires cycles
            if cycles is not None:
                for task in task_list:
                    task['cycles'] = cycles

            if nr_seqs is not None:  # identical pulses on all qubits
                cycles_list = [''] * len(task_list)
                for i, task in enumerate(task_list):
                    if 'cycles' not in task:
                        raise KeyError('Please specify "cycles" either in '
                                       'the task_list or as input '
                                       'parameter to class init.')
                    cycles_list[i] = task['cycles']
                if np.unique(cycles_list, axis=0).shape[0] > 1:
                    # different qubits have different nr cycles; cannot be if
                    # user wants identical pulses on all qubits
                    raise ValueError('Identical pulses on all qubits requires '
                                     'identical cycles arrays. '
                                     'To use non-identical pulses, '
                                     'move nr_seeds from keyword arguments '
                                     'into the tasks.')
                else:
                    cycles = cycles_list[0]

            # add cphase to kw if not already there such that it will be found
            # by generate_kw_sweep_points and will be added to the tasks
            # (the parameters is specified in kw_for_task_keys). Having cphase
            # in kw ensures that it will be passed to paulis_gen_func when
            # generate_kw_sweep_points is called, and hence that the
            # sweep_points are created correctly.
            if 'cphase' not in kw:
                kw['cphase'] = 180

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)
            self.identical_pulses = nr_seqs is not None and all([
                task.get('nr_seqs', None) is None for task in task_list])
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.sequences, self.mc_points = self.sweep_n_dim(
                self.sweep_points, body_block=None,
                body_block_func=self.xeb_block, cal_points=self.cal_points,
                ro_qubits=self.meas_obj_names, **kw)

            self.exp_metadata['identical_pulses'] = self.identical_pulses

            self.autorun(**kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def paulis_gen_func(self, nr_seqs, cycles, cphase=180):
        """
        Creates the list of random gates to be applied in each sequence of the
        experiment.

        Args:
            nr_seqs (int): number of random sequences of gates for each cycle
                length
            cycles (list/array): integers specifying the number of cycles of
                random gates to apply in a sequence
            cphase (float): value of the C-phase gate angle in degrees

        Returns:
             list of strings with op codes
        """
        def gen_random(cycles):
            s_gates = ["X90 ", "Y90 ", "Z45 "]
            lis = []
            for length in cycles:
                cphases = np.random.uniform(0, 1, length) * 180 \
                    if self.parametric else np.repeat([cphase], length)
                gates = []
                gates.append(s_gates[1] + "qb_1")
                sim_str = ' ' if 'Z' in s_gates[1][0:3] else 's '
                gates.append(s_gates[1][0:3] + sim_str + "qb_2")
                gates.append(s_gates[2] + "qb_1")
                sim_str = ' ' if 'Z' in s_gates[2][0:3] else 's '
                gates.append(s_gates[2][0:3] + sim_str + "qb_2")
                gates.append(f"CZ{cphases[0]} " + "qb_1 qb_2")
                if length > 0:
                    for i in range(length - 1):
                        last_1_gate1 = gates[-3][0:4]

                        choice1 = []
                        for gate in s_gates:
                            choice1.append(gate)
                        choice1.remove(last_1_gate1)
                        gate1 = random.choice(choice1)
                        gates.append(gate1 + 'qb_1')

                        last_1_gate2 = gates[-3][0:3] + ' '
                        choice2 = []
                        for gate in s_gates:
                            choice2.append(gate)
                        choice2.remove(last_1_gate2)
                        gate2 = random.choice(choice2)
                        sim_str = ' ' if 'Z' in gate2[:3] else 's '
                        gates.append(gate2[:3] + sim_str + 'qb_2')
                        gates.append(f"CZ{cphases[i+1]} " + 'qb_1 qb_2')
                lis.append(gates)
            return lis
        return [gen_random(cycles) for _ in range(nr_seqs)]

    def xeb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Creates simultaneous blocks of XEB pulses for each task in
        preprocessed_task_list.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args:
            to allow pass through kw even if it contains entries that are
            not needed
        """
        pulse_op_codes_list = []
        tl = [self.preprocessed_task_list[0]] if self.identical_pulses else \
            self.preprocessed_task_list

        for i, task in enumerate(tl):
            seq_idx = sp1d_idx if self.sweep_type['seqs'] == 0 else sp2d_idx
            nrcyc_idx = sp1d_idx if self.sweep_type['cycles'] == 0 else sp2d_idx
            gates_qb_info = task['sweep_points'].get_sweep_params_property(
                'values', self.sweep_type['seqs'], 'gateschoice')[seq_idx][nrcyc_idx]
            l = [gate_qb for gate_qb in gates_qb_info]
            sub_list = []
            for op in l:
                op_split = op.split()
                if len(op_split) == 2:
                    # single qubit gate
                    op = ' '.join([op_split[0], task[op_split[1]]])
                else:
                    # C-phase gate
                    op = ' '.join([op_split[0], task[op_split[1]],
                                   task[op_split[2]]])
                sub_list.append(op)
            pulse_op_codes_list += [sub_list]

        rb_block_list = [
            self.block_from_ops(
                f"rb_{task['qb_1']}{task['qb_2']}",
                [op_list for op_list in pulse_op_codes_list[0 if
                self.identical_pulses else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end',
                                        destroy=self.fast_mode)
