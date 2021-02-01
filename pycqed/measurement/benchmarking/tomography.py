import traceback
from pycqed.measurement.calibration.two_qubit_gates import CalibBuilder
from pycqed.analysis_v2 import tomography_qudev as tomo_analysis
from pycqed.measurement.calibration.calibration_points import CalibrationPoints


class Tomography(CalibBuilder):
    """
    Class to do state or process tomography.

    For state tomography, pass the `pulses` to generate the state from ground
    state. Pass `final_rots_basis` to choose the tomography basis.

    For process tomography, the `pulses` represent the process under
    investigation. Use `final_rots_basis` and `init_rots_basis` to specify the
    tomography bases for the measurement and for the state initialization.

    Args:
        task_list: A list of dictionaries, describing tomography experiments
            to be executed in parallel. Keys for each task:
                qubits: Qubits for the tomography
                pulses: The process or state that is to be measured in
                    tomography. See the docstring of
                    `Tomography.block_from_anything` for allowed formats.
                final_rots_basis, final_all_rots: The basis rotations done
                    before the readout. For the exact format of the arguments,
                    see the docsstring of `CircuitBuilder.tomography_pulses`.
                    Defaults: ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'), True
                init_rots_basis, init_all_rots: The basis rotations done
                    before the pulses for initialization. For the exact format
                    of the arguments, see the docsstring of
                    `CircuitBuilder.tomography_pulses`. Main intended use is
                    for process tomography. Defaults: Defaults: ('I',), True
                prepended_pulses: A sequence of pulses that will be prepended
                    to the initialization rotations, but will come after the
                    preparation pulses (e.g. preselection). See the docstring
                    of `Tomography.block_from_anything` for allowed formats.
                    Default: None
            The default value for any item not specified in the task, except
            for `prepended_pulses` or `qubits`, will be taken from the keyword
            arguments of the init of the class, if provided.
        cal_all_combinations: Generate all combinations of multi-qubit
            calibration states, e.g. 00, 01, 10, 11 for two qubits, instead of
            just the necessary subset assuming no crosstalk, e.g. 00, 11.
            Default: True
        optimize_identity: Implement the identity operation by zero-duration
            virtual pulses if True. Idle for the duration of a pi-pulse if
            False. Default: False
    """

    kw_for_task_keys = ('pulses',
                        'init_rots_basis',
                        'final_rots_basis',
                        'init_all_rots',
                        'final_all_rots')

    def __init__(self, task_list, pulses=None, init_rots_basis=('I',),
                 final_rots_basis=tomo_analysis.DEFAULT_BASIS_ROTS,
                 init_all_rots=True, final_all_rots=True,
                 cal_all_combinations=True, optimize_identity=False, **kw):
        try:
            for task in task_list:
                # convert qubit objects to qubit names
                task['qubits'] = [q if isinstance(q, str) else q.name
                                  for q in task['qubits']]
                # generate an informative task prefix
                if 'prefix' not in task:
                    task['prefix'] = ''.join(task['qubits']) + "_"

            state_tomo = True
            if init_rots_basis != ('I',):
                state_tomo = False
            for task in task_list:
                if task.get('init_rots_basis', ('I',)) != ('I',):
                    state_tomo = False
            if state_tomo:
                self.experiment_name = \
                    f'{kw.get("state_name", "")}StateTomography'
            else:
                self.experiment_name = \
                    f'{kw.get("process_name", "")}ProcessTomography'

            super().__init__(task_list,
                             pulses=pulses,
                             init_rots_basis=init_rots_basis,
                             final_rots_basis=final_rots_basis,
                             init_all_rots=init_all_rots,
                             final_all_rots=final_all_rots,
                             cal_all_combinations=cal_all_combinations,
                             optimize_identity=optimize_identity,
                             **kw)

            self.optimize_identity = optimize_identity
            self.global_prepend_block = None
            self.global_initializations = None
            self.global_finalizations = None

            # Preprocess sweep points and tasks before creating the sequences
            self.preprocessed_task_list = self.preprocess_task_list(
                pulses=pulses,
                init_rots_basis=init_rots_basis,
                final_rots_basis=final_rots_basis,
                init_all_rots=init_all_rots,
                final_all_rots=final_all_rots,
                cal_all_combinations=cal_all_combinations,
                optimize_identity=optimize_identity,
                **kw)

            init_kwargs = kw.get('init_kwargs', {})
            init_kwargs['prepend_block'] = self.global_prepend_block

            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                block_align=['end'], init_kwargs=init_kwargs, **kw)

            self.autorun(**kw)  # run measurement & analysis if requested in kw
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def preprocess_task_list(self, **kw):
        self.preprocessed_task_list = super().preprocess_task_list(**kw)
        return self.preprocess_task_list_tomography(
            preprocessed_task_list=self.preprocessed_task_list)

    def preprocess_task_list_tomography(self, preprocessed_task_list):
        prepend_block_list = []
        initializations_map = {}
        nr_inits = 0
        finalizations_map = {}
        nr_finals = 0

        def process_tomo(qubits, rots_basis, all_rots):
            seg_map = {}
            nr_segs = 0
            if rots_basis is None:
                rots_basis = ('I',)
            tomo = self.tomography_pulses(qubits, rots_basis, all_rots)
            for tomo_seg in tomo:
                for qb in qubits:
                    tomo_seg_qb = tomo_seg[self.qb_names.index(qb)]
                    seg_map[qb] = seg_map.get(qb, [])
                    seg_map[qb].append(tomo_seg_qb)
                    nr_segs = max(nr_segs, len(seg_map[qb]))
            return seg_map, nr_segs, rots_basis

        for task in preprocessed_task_list:
            initializations_map, nr_inits, task['init_rots_basis'] = \
                process_tomo(task['qubits'], task['init_rots_basis'],
                             task['init_all_rots'])
            finalizations_map, nr_finals, task['final_rots_basis'] = \
                process_tomo(task['qubits'], task['final_rots_basis'],
                             task['final_all_rots'])
            if task['prepend_pulses'] is not None:
                prepend_block_list.append(self.block_from_anything(
                    task['prepend_pulses'], 'PrependPulses'))
        for qb in self.qb_names:
            if qb not in initializations_map:
                initializations_map[qb] = []
            if qb not in finalizations_map:
                finalizations_map[qb] = []
        for qb, tomo_init_qb in initializations_map.items():
            while len(tomo_init_qb) < nr_inits:
                tomo_init_qb.append('I')
        for qb, tomo_final_qb in finalizations_map.items():
            while len(tomo_final_qb) < nr_finals:
                tomo_final_qb.append('I')
        self.global_initializations = []
        for i in range(nr_inits):
            self.global_initializations += [[]]
            for qb in self.qb_names:
                self.global_initializations[-1] += [initializations_map[qb][i]]
        self.global_finalizations = []
        for i in range(nr_finals):
            self.global_finalizations += [[]]
            for qb in self.qb_names:
                self.global_finalizations[-1] += [finalizations_map[qb][i]]
        if len(prepend_block_list) > 0:
            self.global_prepend_block = self.simultaneous_blocks(
                'prepend_block', prepend_block_list, block_align='end')
        self.do_optimize_identity()
        self.sweep_points.add_sweep_parameter(
            'finalize', self.global_finalizations, '', 'Final', dimension=0)
        self.sweep_points.add_sweep_parameter(
            'initialize', self.global_initializations, '', 'Init', dimension=1)

        return preprocessed_task_list

    def sweep_block(self, pulses, **kw):
        return self.block_from_anything(pulses, 'Pulses')

    def create_cal_points(self, n_cal_points_per_state=1, cal_states='auto',
                          for_ef=False, cal_all_combinations=False, **kw):
        if not cal_all_combinations:
            return super().create_cal_points(
                n_cal_points_per_state=n_cal_points_per_state,
                cal_states=cal_states,
                for_ef=for_ef, **kw)
        self.cal_states = CalibrationPoints.guess_cal_states(
            cal_states, for_ef=for_ef)
        self.cal_points = CalibrationPoints.multi_qubit(
            self.task_list[0]['qubits'], self.cal_states,
            n_per_state=n_cal_points_per_state, all_combinations=True)
        for task in self.task_list[1:]:
            self.cal_points = CalibrationPoints.combine_parallel(
                self.cal_points,
                CalibrationPoints.multi_qubit(
                    task['qubits'], self.cal_states,
                    n_per_state=n_cal_points_per_state, all_combinations=True)
            )
        self.exp_metadata.update({'cal_points': repr(self.cal_points)})

    def block_from_anything(self, pulses, block_name):
        """
        Convert various input formats into a `Block`.
        Args:
            pulses: A specification of a pulse sequence. Can have the following
                formats:
                    1) Block: A block class is returned unmodified.
                    2) str: A single op code.
                    3) dict: A single pulse dictionary. If the dictionary
                           includes the key `op_code`, then the unspecified
                           pulse parameters are taken from the corresponding
                           operation.
                    4) list of str: A list of op codes.
                    5) list of dict: A list of pulse dictionaries, optionally
                           including the op-codes, see also format 3).
            block_name: Name of the resulting block
        Returns: The input converted to a Block.
        """

        if hasattr(pulses, 'build'):  # Block
            return pulses
        elif isinstance(pulses, str):  # opcode
            return self.block_from_ops(block_name, [pulses])
        elif isinstance(pulses, dict):  # pulse dict
            return self.prepend_pulses_block([pulses], block_name=block_name)
        elif isinstance(pulses[0], str):  # list of opcodes
            return self.block_from_ops(block_name, pulses)
        elif isinstance(pulses[0], dict):  # list of pulse dicts
            return self.prepend_pulses_block(pulses, block_name=block_name)

    def do_optimize_identity(self):
        def recursive_replace_op(in_this, this, with_that):
            if isinstance(in_this, str):
                if in_this == this:
                    return with_that
                elif in_this.startswith(this + ' '):
                    return with_that + ' ' + in_this[len(this) + 1:]
                else:
                    return in_this
            else:
                return [recursive_replace_op(el, this, with_that)
                        for el in in_this]

        replacement_op = 'Z0' if self.optimize_identity else 'X0'
        self.global_finalizations = recursive_replace_op(
            self.global_finalizations, 'I', replacement_op)
        self.global_initializations = recursive_replace_op(
            self.global_initializations, 'I', replacement_op)
