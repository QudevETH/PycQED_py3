import numpy as np
import traceback
from copy import deepcopy
import random
from pycqed.measurement.calibration.two_qubit_gates import MultiTaskingExperiment
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb
from pycqed.measurement.sweep_points import SweepPoints
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc
import logging
log = logging.getLogger(__name__)


class RandomCircuitBenchmarkingMixin:
    """Mixin containing utility functions needed by the RB and XEB classes.

    Classes deriving from this mixin must have the following attributes:
        kw_for_task_keys: see docstring of MultiTaskingExperiment
        kw_for_sweep_points: see docstring of MultiTaskingExperiment

    Creates the following attributes:
        sweep_type: Dict of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).
        identical_pulses: Bool, whether the same XEB experiment should be
            run on all tasks (True), or have unique sequences per task (False)
    """

    seq_lengths_name = 'cliffords'
    """Name of the parameter specifying the sequence lengths 
    
    'cliffords' for RB, 'cycles' for XEB
    """

    randomizations_name = 'seeds'
    """Name of the parameter specifying the number of times to randomize each 
    sequence length. 
    
    'seeds' for RB, 'seqs' for XEB
    """

    @staticmethod
    def update_kw_cal_states(kw):
        """
        Disables cal_points unless user has explicitly set them.

        The cal points are disabled by setting cal_states = '' in the kw.

        Args:
            kw: keyword arguments which will be updated
        """
        kw['cal_states'] = kw.get('cal_states', '')

    def create_sweep_type(self, sweep_type=None):
        """
        Creates the sweep_type attribute.

        Attributes:
            sweep_type (dict): of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).
        """
        self.sweep_type = sweep_type
        if self.sweep_type is None:
            self.sweep_type = {self.seq_lengths_name: 0,
                               self.randomizations_name: 1}

    def update_kw_for_sweep_points_dimension(self):
        """
        Updates the sweep dimensions specified in the class attribute
        kw_for_sweep_points based on the sweep_type attribute.
        """
        self.kw_for_sweep_points = deepcopy(self.kw_for_sweep_points)
        key = f'nr_{self.randomizations_name},{self.seq_lengths_name}'
        if f'{key},cphase,append_gates' in self.kw_for_sweep_points:
            # TwoQubitXEB case
            key = f'{key},cphase,append_gates'
        self.kw_for_sweep_points[key]['dimension'] = \
            self.sweep_type[self.randomizations_name]
        self.kw_for_sweep_points[self.seq_lengths_name]['dimension'] = \
            self.sweep_type[self.seq_lengths_name]

    def check_identical_pulses(self, task_list, **kw):
        """
        Check whether identical pulses should be applied to all tasks.

        The same RB/XEB experiment will be run on all tasks if the input
        parameter self.randomizations_name is provided as a global parameter
        to the init of the RB/XEB measurement class. Therefore, here kw must
        be the kw that were passed to the init of the RB/XEB measurement class.

        Creates the attribute identical_pulses (True if self.randomizations_name
        if provided as a global parameter; False if this parameter is specified
        in each task).

        Args:
            task_list (list of dicts): see CalibBuilder docstring
            **kw: kwargs that were passed to the __init__ of the child class.
                Can contain seq_lengths_name and f'nr_{randomizations_name}'
                if these parameters are passed globally by the user.
        """
        global_seq_lengths = kw.get(self.seq_lengths_name)
        nr_rand = f'nr_{self.randomizations_name}'
        nr_rand_not_in_tasks = all(
            [task.get(nr_rand) is None for task in task_list])
        cphase_not_in_taks = all(
            [task.get('cphase') is None for task in task_list])
        self.identical_pulses = kw.get(nr_rand) is not None and \
                                nr_rand_not_in_tasks and cphase_not_in_taks
        # Check if we can apply identical pulses on all tasks:
        # can only do this if they have identical cliffords array.
        if self.identical_pulses and global_seq_lengths is None:
            raise ValueError(f'Identical pulses on all qubits requires '
                             f'identical {self.seq_lengths_name} arrays: '
                             f'please specify {self.seq_lengths_name} as a '
                             f'global parameter.\n'
                             f'To use non-identical pulses, '
                             f'move nr_{self.randomizations_name} from '
                             f'keyword arguments into the tasks.')

        if not self.identical_pulses:
            # If each task should receive unique pulses, then we must ensure
            # that seq_lengths_name and randomizations_name are in the
            # task_list in order to create the sweep_points when calling
            # generate_kw_sweep_points inside preprocess_task.

            # make sure kw_for_task_keys is a list (instantiated as tuple in the
            # base class)
            self.kw_for_task_keys = list(self.kw_for_task_keys)
            self.kw_for_task_keys += [self.seq_lengths_name,
                                      self.randomizations_name]


class RandomizedBenchmarking(MultiTaskingExperiment,
                             RandomCircuitBenchmarkingMixin):

    """
    Base class for Randomized Benchmarking measurements:
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.062321
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.080505
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.022330

    Attributes in addition to those of the base classes:
        purity: Bool, whether to run purity benchmarking (only implemented for
            single qubit RB):
            https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.260501
        tomo_pulses: List with tomo pulses to use in purity benchmarking
        interleaved_gate: Str (single qb RB) or Int (two-qb RB) specifying the
            name or index inside the Clifford group of the interleaved gate.
        gate_decomposition: Str, name of the decomposition ('XY' or 'HZ') for
            translating the Clifford elements into physical gates
    """

    default_experiment_name = 'RB'
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

        Keyword args:
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
             - If nr_seeds and cliffords are specified in kw and they do not
             exist in the task_list, then all tasks will receive the same pulses
             - in rb_block, it assumes only one parameter is being swept in the
             second sweep dimension (cliffords)
             - interleaved_gate and gate_decomposition should be the same for
             all qubits since otherwise the segments will have very different
             lengths for different qubits
        """
        try:
            self.update_kw_cal_states(kw)
            self.create_sweep_type(sweep_type)
            self.update_kw_for_sweep_points_dimension()
            # tomo pulses for purity benchmarking
            self.tomo_pulses = kw.get('tomo_pulses', ['I', 'X90', 'Y90'])
            self.purity = purity
            self.interleaved_gate = interleaved_gate
            self.gate_decomposition = gate_decomposition

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

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)

            self.default_experiment_name += f'_{self.gate_decomposition}'
            if self.purity:
                self.default_experiment_name += '_purity'

            self.check_identical_pulses(task_list, **kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
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
            self.exp_metadata['purity'] = self.purity

            self.add_processing_pipeline(**kw)
            self.autorun(**kw)

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
    task_mobj_keys = ['qb']

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
            self.default_experiment_name = 'SingleQubitIRB'
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
                                 else i]],
            pulse_modifs=task.get('pulse_modifs', None),
        )
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end',
                                        destroy=self.fast_mode)


class TwoQubitRandomizedBenchmarking(RandomizedBenchmarking):
    """
    Class for running the two-qubit randomized benchmarking experiment on
    several pairs of qubits in parallel.

    Attributes in addition to the ones created by the base class:
        max_clifford_idx: Int, size of the 2QB Clifford group

    """
    default_experiment_name = 'TwoQubitRB'
    task_mobj_keys = ['qb_1', 'qb_2']

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
        if kw.get('interleaved_gate', None) is not None:
            self.default_experiment_name = 'TwoQubitIRB'
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
                            pulse_modifs=task.get('pulse_modifs', None),
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


class CrossEntropyBenchmarking(MultiTaskingExperiment,
                               RandomCircuitBenchmarkingMixin):
    """
    Base class for Cross-Entropy Benchmarking measurements:
    https://www.nature.com/articles/s41567-018-0124-x
    https://www.science.org/doi/10.1126/science.aao4309
    https://www.nature.com/articles/s41586-019-1666-5
    """
    default_experiment_name = 'XEB'
    seq_lengths_name = 'cycles'
    randomizations_name = 'seqs'

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 sweep_type=None, **kw):
        """
        Base class for a cross-entropy benchmarking experiment on
        one or several qubits in parallel (see docstring of children for more
        details).

        Args:
            task_list (list): see CalibBuilder docstring
            sweep_points (SweepPoints instance):
                Ex: [{'cycles': ([0, 4, 10], '', 'Nr. Cycles')},
                     {'nr_seqs': (array([0, 1, 2, 3]), '', 'Nr. Sequences')}]
            qubits (list): QuDev_transmon class instances on which to run
                measurement
            sweep_type (dict): of the form {'cycles': 0/1, 'seqs': 1/0}, where
                the integers specify which parameter should correspond to the
                inner sweep (0), and which to the outer sweep (1).

        Keyword args:
            nr_seqs (int; default: None): the number of times to apply a random
                iteration of a sequence consisting of nr_cycles cycles.
                If nr_seqs is specified and it does not exist in the task_list,
                then all qubits will receive the same pulses provided they have
                the same cycles array.
            cycles (list/array; default: None): integers specifying the number
                of cycles to apply.
            Passed to CalibBuilder; see docstring there.

        Assumptions:
        - if nr_seqs is passed, the same random gates will be applied to all
         tasks (same XEB experiment on all tasks). To have different random
         gates for each task, pass nr_seqs in the task_list.
        """
        try:
            self.kw_for_sweep_points.update({
                'cycles': dict(param_name='cycles', unit='',
                               label='Nr. Cycles', dimension=0),
            })
            self.update_kw_cal_states(kw)
            self.create_sweep_type(sweep_type)
            self.update_kw_for_sweep_points_dimension()

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, **kw)
            self.check_identical_pulses(task_list, **kw)
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

    def xeb_block(self, sp1d_idx, sp2d_idx, **kw):
        """
        Base method to create the xeb blocks. To be modified by children.

        Args:
            sp1d_idx (int): current iteration index in the 1D sweep points array
            sp2d_idx (int): current iteration index in the 2D sweep points array

        Keyword args: see the docstrings of children
        """
        pass


class SingleQubitXEB(CrossEntropyBenchmarking):
    """
    Class for running the single qubit cross-entropy benchmarking experiment on
    several qubits in parallel.
    """
    default_experiment_name = 'SingleQubitXEB'
    kw_for_sweep_points = {
        'nr_seqs,cycles': dict(
            param_name='z_rots', unit='',
            label='$R_z$ angles, $\\phi$',
            dimension=1, values_func=lambda ns, cycles: [[
                list(np.random.uniform(0, 2, nc) * 180)
                for nc in cycles] for _ in range(ns)])}
    task_mobj_keys = ['qb']

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, init_rotation=None, **kw):
        """
        Init of the SingleQubitXEB class.
        The experiment consists of applying
        [[Ry - Rz(theta)] * nr_cycles for nr_cycles in cycles] nr_seqs times,
        with random values of theta each time.

        Args:
            nr_seqs (int): the number of times to apply a random
                iteration of a sequence consisting of nr_cycles cycles.
                If nr_seqs is specified and it does not exist in the task_list,
                THEN ALL TASKS WILL RECEIVE THE SAME PULSES provided they have
                the same cycles array.
            cycles (list/array): integers specifying the number of
                random cycles to apply in a sequence.
            init_rotation (str): the preparation pulse name
            See docstring of base class for the remaining parameters.

        Keyword args:
            See docstring of base class

        Assumptions:
         - assumes there is one task for each qubit.
        """
        try:
            self.init_rotation = init_rotation
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles,
                             init_rotation=init_rotation, **kw)
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
            pulse_l_tmp = [] if self.init_rotation is None \
                else [self.init_rotation]
            pulse_l_tmp += [e1 for e2 in l for e1 in e2]
            pulse_op_codes_list += [pulse_l_tmp]

        rb_block_list = [self.block_from_ops(
            f"rb_{task['qb']}", [f"{p} {task['qb']}" for p in
                                 pulse_op_codes_list[0 if
                                 self.identical_pulses else i]])
            for i, task in enumerate(self.preprocessed_task_list)]

        return self.simultaneous_blocks(f'sim_rb_{sp1d_idx}{sp1d_idx}',
                                        rb_block_list, block_align='end',
                                        destroy=self.fast_mode)


class TwoQubitXEB(CrossEntropyBenchmarking):
    """
    Class for running the two-qubit cross-entropy benchmarking experiment on
    several pairs of qubits in parallel.
    Implementation as in https://www.nature.com/articles/s41567-018-0124-x.

    """
    default_experiment_name = 'TwoQubitXEB'
    kw_for_sweep_points = {
        'nr_seqs,cycles,cphase,append_gates': dict(
            param_name='gateschoice', unit='',
            label='cycles gates', dimension=1,
            values_func='paulis_gen_func')}
    task_mobj_keys = ['qb_1', 'qb_2']
    kw_for_task_keys = ['cphase']

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, **kw):
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
            nr_seqs (int): the number of times to apply a random
                iteration of a sequence consisting of nr_cycles cycles.
            cycles (list/array): integers specifying the number of
                random cycles to apply in a sequence.
            randomize_cphases (bool): whether to do parametric C-phase gates,
                with a random angle in each cycle
            See docstring of base class for remaining parameters.

        Keyword args:
            cphase (float; default: None): value of the C-phase gate angle
                in degrees. If None,
                Allowed values:
                    float: angle of the CZ gates (deg)
                    None: a standard CZ gate (180 deg) is done (see
                    NZTransitionControlledPulse)
                    'randomized': each CZ gate will be run with a different
                    random angle
            See docstring of base class for further parameters.

        Assumptions:
         - assumes there is one task for CZ gate.
         - if cphase is different for each task, the XEB sequences will be
         randomized between tasks even if nr_seqs is specified globally.
        """
        try:
            # add cphase to kw if not already there such that it will be found
            # by generate_kw_sweep_points and will be added to the tasks
            # (the parameter is specified in kw_for_task_keys). Having cphase
            # in kw ensures that it will be passed to paulis_gen_func when
            # generate_kw_sweep_points is called, and hence that the
            # sweep_points are created correctly.
            if kw.get('cphase', None) is None:
                kw['cphase'] = ''

            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)

        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def paulis_gen_func(self, nr_seqs, cycles, cphase='', append_gates=()):
        """
        Creates the list of random gates to be applied in each sequence of the
        experiment.

        Args:
            nr_seqs (int): number of random sequences of gates for each cycle
                length
            cycles (list/array): integers specifying the number of cycles of
                random gates to apply in a sequence
            cphase (float): value of the C-phase gate angle in degrees
            append_gates (list): list of gates to append to the two-qubit gate

        Returns:
             list of strings with op codes
        """

        list_all_seqs = []
        for _ in range(nr_seqs):
            s_gates = ["X90", "Y90", "Z45"]
            lis = []
            for length in cycles:
                cphases = np.random.uniform(0, 1, length) * 360 \
                    if cphase=='randomized' else np.repeat([cphase], length)
                gates = []
                gates.append(s_gates[1] + " qb_1")
                gates.append(s_gates[1] + "s qb_2")
                gates.append(s_gates[2] + " qb_1")
                gates.append(s_gates[2] + "s qb_2")
                gates.append(f"CZ{cphases[0]} qb_1 qb_2")
                gates += append_gates
                if length > 0:
                    for i in range(length - 1):
                        last_1_gate1 = gates[-3-len(append_gates)][:3]

                        choice1 = []
                        for gate in s_gates:
                            choice1.append(gate)
                        choice1.remove(last_1_gate1)
                        gate1 = random.choice(choice1)
                        gates.append(gate1 + " qb_1")

                        last_1_gate2 = gates[-3-len(append_gates)][:3]
                        choice2 = []
                        for gate in s_gates:
                            choice2.append(gate)
                        choice2.remove(last_1_gate2)
                        gate2 = random.choice(choice2)
                        gates.append(gate2 + "s qb_2")
                        gates.append(f"CZ{cphases[i+1]} qb_1 qb_2")
                        gates += append_gates
                lis.append(gates)
            list_all_seqs.append(lis)
        return list_all_seqs

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


class TwoQubitXEBMultiCphase(MultiTaskingExperiment):
    default_experiment_name = 'TwoQubitXEBMultiCphase'
    kw_for_task_keys = ('cphases')
    task_mobj_keys = ['qb_1', 'qb_2']

    def __init__(self, task_list, sweep_points=None, qubits=None,
                 nr_seqs=None, cycles=None, cphases=None, **kw):

        try:
            super().__init__(task_list, qubits=qubits,
                             sweep_points=sweep_points, cphases=cphases,
                             nr_seqs=nr_seqs, cycles=cycles, **kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            self.xeb_measurements = []
            self.measure = kw.pop('measure', True)
            nr_cphases = len(self.preprocessed_task_list[0]['cphases'])
            for i in range(nr_cphases):
                tl = deepcopy(task_list)
                for task in tl:
                    assert len(task['cphases']) == nr_cphases,\
                        "Number of cphases inconsistent between tasks!"
                    task['cphase'] = task['cphases'][i]
                    if 'sweep_points' in task:
                        # Allows to reuse a task list from a previous
                        # measurement, by undoing self.combine_sweep_points
                        task['sweep_points'] = \
                            self.extract_combined_sweep_points(
                                task['sweep_points'], i)
                self.xeb_measurements += [
                    TwoQubitXEB(tl, sweep_points, qubits, nr_seqs=nr_seqs,
                                cycles=cycles, measure=False, **kw)]
            # interleave sequences
            self.sequences, self.mc_points = \
                self.xeb_measurements[0].sequences[0].interleave_sequences(
                    [xeb.sequences for xeb in self.xeb_measurements])
            # combine sweep points
            self.combine_sweep_points()

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def combine_sweep_points(self):
        for i, xebm in enumerate(self.xeb_measurements):
            sweep_points = SweepPoints(xebm.sweep_points)
            sweep_points.append_suffix_to_sweep_params(str(i))
            self.sweep_points.update(sweep_points)

    @staticmethod
    def extract_combined_sweep_points(sp_full, idx, deep=False):
        if deep:
            sp = deepcopy(sp_full)
        else:
            sp = sp_full
        suffix = f'_{idx}'
        # Trim sp
        for d in sp:
            for key in list(d):
                # Remove prefixed value
                val = d.pop(key)
                # Re-add it (without prefix)
                # only if it has the correct prefix
                if key.endswith(suffix):
                    new_key = key[:-len(suffix)]
                    d[new_key] = val
        print(sp[0].keys(), sp[1].keys())
        return sp

