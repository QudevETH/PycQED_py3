import itertools
import pycqed.measurement.quantum_experiment as qe_mod
import pycqed.measurement.waveform_control.block as block_mod
import pycqed.measurement.calibration.calibration_points as cp_mod
from pycqed.measurement import sweep_points as sp_mod
import numpy as np
class SurfaceCodeExperiment(qe_mod.QuantumExperiment):
    block_align = 'center'
    type_to_ops_map = {
        'X': ('Y90', 'mY90'),
        'Y': ('mX90', 'X90'),
        'Z': (None, None),
        'I': ('I', 'I'),
    }

    def __init__(self, data_qubits, ancilla_qubits,
                 readout_rounds, nr_cycles, initializations=None,
                 finalizations=None, ancilla_reset=False,
                 ancilla_dd=True, data_dd=False, data_dd_simple=False, skip_last_ancilla_readout=False,
                 two_qb_gates_off=False,
                 **kw):
        # provide default values
        for k, v in [
            ('qubits', data_qubits + ancilla_qubits),
            ('experiment_name',
             f'S{len(data_qubits + ancilla_qubits)}_experiment'),
        ]:
            kw.update({k: kw.get(k, v)})
        super().__init__(**kw)
        self.data_qubits = data_qubits
        self.ancilla_qubits = ancilla_qubits
        self.readout_rounds = readout_rounds
        self.ancilla_reset = ancilla_reset
        self.nr_cycles = nr_cycles
        self.initializations = initializations
        self.finalizations = finalizations
        self.ancilla_dd = ancilla_dd
        self.data_dd = data_dd
        self.data_dd_simple = data_dd_simple
        self.cycle_length = sum([r['round_length']
                                 for r in self.readout_rounds])
        self.skip_last_ancilla_readout = skip_last_ancilla_readout
        self.final_readout_delay = kw.get('final_readout_delay', 0)
        self.mc_points_override = kw.get('mc_points_override', None)
        self._parse_initializations()
        self._parse_finalizations(
            basis_rots=kw.get('basis_rots', ('I', 'X90', 'Y90')))
        self.sweep_points = kw.get('sweep_points', sp_mod.SweepPoints())
        self.sweep_points.add_sweep_parameter(
            'finalize', self.finalizations, '', 'Final', dimension=0)
        self.sweep_points.add_sweep_parameter(
            'initialize', self.initializations, '', 'Init', dimension=1)
        self.two_qb_gates_off = two_qb_gates_off
        self.sequences, self.mc_points = self.sweep_n_dim(
            self.sweep_points, self.main_block(), repeat_ro=False,
            init_kwargs={'pulse_modifs': {'all': {
                'element_name': 'init_element'}}},
            final_kwargs={'pulse_modifs': {'all': {
                'element_name': 'final_element', 'pulse_delay':5e-9}}},
        )
        if self.mc_points_override is not None:
            self.mc_points[0] = self.mc_points_override

        # TODO (Nathan): in the future, we might want to put the experimental
        #  metadata update
        #  at the beginning of the measurement or in the "prepare measurement", such that
        #  we are sure that the "latest" values of these parameters are used when saving the
        #  metadata.
        self.exp_metadata.update({"nr_cycles": self.nr_cycles,
                                  "ancilla_dd": self.ancilla_dd,
                                  "ancilla_reset": self.ancilla_reset,
                                  "two_qb_gates_off": self.two_qb_gates_off,
                                  'skip_last_ancilla_readout':
                                      self.skip_last_ancilla_readout})


    def _parse_finalizations(self, basis_rots):
        if self.finalizations is None or self.finalizations == 'logical_z':
            self.finalizations = [len(self.data_qubits) * ['I']]
        elif self.finalizations == 'logical_x':
            self.finalizations = [len(self.data_qubits) * ['Y90']]
        elif self.finalizations == 'data_tomo' or self.finalizations == 'tomo':
            self.finalizations = self.tomography_pulses(
                [q.name for q in self.data_qubits], basis_rots)
        elif self.finalizations == 'full_tomo':
            self.finalizations = self.tomography_pulses(
                [q.name for q in self.qubits], basis_rots)
        for i in range(len(self.finalizations)):
            fin = list(self.finalizations[i])
            if len(fin) < len(self.qubits):
                fin += (len(self.qubits) - len(fin)) * ['I']
            self.finalizations[i] = fin[:len(self.qubits)]

    def _parse_initializations(self):
        if self.initializations is None:
            self.initializations = [len(self.qubits) * '0']
        elif self.initializations == 'full_data_z':
            self.initializations = list(
                itertools.product(['0', '1'], repeat=len(self.data_qubits)))
        elif self.initializations == 'full_data_x':
            self.initializations = list(
                itertools.product(['+', '-'], repeat=len(self.data_qubits)))
        for i in range(len(self.initializations)):
            init = list(self.initializations[i])
            if len(init) < len(self.qubits):
                init += (len(self.qubits) - len(init)) * ['0']
            self.initializations[i] = init[:len(self.qubits)]

    def main_block(self):
        """Block structure of the experiment:
        * Full experiment block excluding init. and final (tomo) readout
            o `nr_cycles` copies of the cycle block
                - Gates block for each readout round. The length is explicitly
                  set to the value from readout_round_lengths.
                    + One block for each simultaneous CZ gate step
                    + Interleaved with single-qubit gates for basis rotations
                      and dynamical decouplings
                - The ancilla readout blocks, one for each readout round.
        """
        pulses = []
        for cycle in range(self.nr_cycles):
            c = self._cycle_block(cycle)
            pulses += c.build(ref_pulse='start',
                              block_delay=cycle*self.cycle_length)
        return block_mod.Block(
            'main', pulses, block_end={'pulse_delay': self.final_readout_delay})

    def _parallel_cz_step_block(self, gates, dd_qubits=None, dd_pulse='Y180',
                                anc_comp_qubits=None, anc_comp_pulse='Z180',
                                pulse_modifs=None):
        """
        Creates a block containing parallel CZ gates and dynamical decoupling
        pulses. For now all CZ gates are aligned at the start and the dd pulses
        are aligned to the center of the last CZ gate.

        Args:
            gates: a list of qubit pairs for the parallel two-qubit gates
            dd_qubits: a list of qubits to apply the dynamical decoupling pulses
                to.
        Returns:
            A block containing the relevant pulses.
        """
        dd_qubits = dd_qubits if dd_qubits is not None else []
        anc_comp_qubits = anc_comp_qubits if anc_comp_qubits is not None else []
        block_name = 'CZ_' + '_'.join([''.join(gate) for gate in gates])
        ops = ['CZ {} {}'.format(*gate) for gate in gates]
        ops += [dd_pulse + ' ' + dd_qb for dd_qb in dd_qubits]
        ops += [anc_comp_pulse + ' ' + anc_qb for anc_qb in anc_comp_qubits]
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops]
        return self.simultaneous_blocks(block_name, blocks,
                                        block_align=self.block_align)

    def _readout_round_gates_block(self, readout_round, cycle=0):


        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancilla_steps = {pm['ancilla']: pm['data'] for pm in round_parity_maps}
        total_steps = max([len(v) for v in ancilla_steps.values()])
        ancilla_steps = {a: (total_steps - len(v)) // 2 * [None] +
                            list(v) + (total_steps - len(v) + 1) // 2 * [None]
                         for a, v in ancilla_steps.items()}
        print('anc steps: ', ancilla_steps)
        # cz gates
        element_name = f'parity_map_entangle_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name,
                                    pulse_off=self.two_qb_gates_off)}
        gate_lists = [
            [(a, ds[s]) for a, ds in ancilla_steps.items() if ds[s] is not None]
            for s in range(total_steps)]
        if self.data_dd:
            gate_lists_ancqb = [[qb[0] for qb in gl] for gl in gate_lists]
            gate_lists_dataqb = [[qb[1] for qb in gl] for gl in gate_lists]
            # for each time step the dyn. decoupled qubits are determined among the data qubits to be those
            # which do not participate which do not participate in another gate and are neighbors to an active ancilla
            dd_qubit_lists = [
                [d for qb in gate_lists_ancqb[i] for d in ancilla_steps[qb] if d is not None and d not in gate_lists_dataqb[i]]
                for i in range(total_steps)]
            dd_qubit_lists = [list(np.unique(dd)) for dd in dd_qubit_lists]
            # for each timestep, for each dyn. decoupling pulse on a data qubit determine ancilla qubits which
            # will do a cz gate with that data qubit in a later timestep
            # anc_compensation_lists = [[anc for anc in ancilla_steps for dq in dd_qubit_lists[i] if (anc, dq) in
            #                            [gate for gate_list in gate_lists[i+1:] for gate in gate_list]] for i in
            #                           range(total_steps-1)]
            anc_compensation_lists = []
            for i in range(total_steps-1):
                anc_compensation_lists.append([])
                for dq in dd_qubit_lists[i]:
                    for anc in ancilla_steps:
                        if (anc, dq) in [gate for gate_list in gate_lists[i + 1:] for gate in gate_list]:
                            if anc not in anc_compensation_lists[i]:
                                anc_compensation_lists[i] += [anc]
                            else:
                                anc_compensation_lists[i].remove(anc)
            anc_compensation_lists.append([])
            cz_step_blocks = [
                self._parallel_cz_step_block(gates, dd_qubits=dd_qubits, anc_comp_qubits=anc_comp_qubits,
                                             pulse_modifs=pulse_modifs)
                for gates, dd_qubits, anc_comp_qubits in zip(gate_lists, dd_qubit_lists, anc_compensation_lists)]
        else:
            cz_step_blocks = [self._parallel_cz_step_block(gates, pulse_modifs=pulse_modifs)
                for gates in gate_lists]
        print('gate_lists: ', gate_lists)
        # ancilla dd
        element_name = f'parity_map_ancilla_dd_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name)}
        ops_dd = []
        if self.ancilla_dd:
            if total_steps % 2:
                raise NotImplementedError('Ancilla dynamical decoupling not '
                                          'implemented for odd weight parity maps.')
            for a, ds in ancilla_steps.items():
                ops_dd += [f'Y180 {a}', f'Z180 {a}']
                ops_dd += [f'Z180 {d}' for d in ds[total_steps//2:]
                            if d is not None]
        if self.data_dd_simple:
            for a, ds in ancilla_steps.items():
                for d in ds:
                    if d is not None and f'Y180 {d}' not in ops_dd:
                        # DD on data qubit
                        ops_dd += [f'Y180 {d}']
                        # compensation on data qubit
                        if f'Z180 {d}' not in ops_dd:
                            ops_dd += [f'Z180 {d}']
                        else:
                            ops_dd.remove(f'Z180 {d}')
                # compensation on ancilla qubit
                for d in ds[total_steps // 2:]:
                    if d is not None:
                        if f'Z180 {a}' not in ops_dd:
                            ops_dd += [f'Z180 {a}']
                        else:
                            ops_dd.remove(f'Z180 {a}')
        if self.ancilla_dd or self.data_dd_simple:
            blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                      for op in ops_dd]
            ancilla_dd_block = self.simultaneous_blocks(
                'ancilla_dd_block', blocks, block_align=self.block_align)
            cz_step_blocks = cz_step_blocks[:total_steps//2] + \
                [ancilla_dd_block] + cz_step_blocks[total_steps//2:]

        # data qubit and ancilla basis changes
        qubit_bases = {}
        for pm in round_parity_maps:
            qubit_bases[pm['ancilla']] = pm.get('ancilla_type', "X")
            for qb in pm['data']:
                if qb is None:
                    continue
                assert qubit_bases.get(qb, pm['type']) == pm['type']
                qubit_bases[qb] = pm['type']
        ops_init = []
        ops_final = []
        for qb, basis in qubit_bases.items():
            basis = self.type_to_ops_map.get(basis, basis)
            assert isinstance(basis, tuple)
            if basis[0] is not None:
                ops_init += [f'{basis[0]} {qb}']
            if basis[1] is not None:
                ops_final += [f'{basis[1]} {qb}']
        # add compensation pulses on data qubits to compensate previously applied DD pulses
        if self.data_dd:
            dd_qubit_lists_flattened = [ddqb for dd_list in dd_qubit_lists for ddqb in dd_list]
            for ddqb in np.unique(dd_qubit_lists_flattened):
                nr_dd_pulses = sum(ddqb2 == ddqb for ddqb2 in dd_qubit_lists_flattened)
                comp_pulse = nr_dd_pulses % 2
                # print(ddqb, comp_pulse)
                if comp_pulse:
                    if f'Y180 {ddqb}' in ops_final:
                        ops_final.remove(f'Y180 {ddqb}')
                    elif f'mY90 {ddqb}' in ops_final:
                        ops_final.remove(f'mY90 {ddqb}')
                        ops_final += [f'Y90 {ddqb}']
                    elif f'Y90 {ddqb}' in ops_final:
                        ops_final.remove(f'Y90 {ddqb}')
                        ops_final += [f'mY90 {ddqb}']
                    else:
                        ops_final += [f'Y180 {ddqb}']
        element_name = f'parity_basis_init_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name)}
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops_init]
        init_block = self.simultaneous_blocks('init_block', blocks,
                                              block_align=self.block_align)
        element_name = f'parity_basis_final_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name)}
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops_final]
        final_block = self.simultaneous_blocks('final_block', blocks,
                                               block_align=self.block_align)
        final_block.pulses[-1]['pulse_delay'] = 6e-9
        blocks = [init_block] + cz_step_blocks + [final_block]
        # print('Init block: ', init_block)
        # print('CZ step block: ', cz_step_blocks)
        return self.sequential_blocks(element_name, blocks)

    def _readout_round_readout_block(self, readout_round, cycle=0):
        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancillas = [pm['ancilla'] for pm in round_parity_maps]

        element_name = f'readouts_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name,
                                    ref_point='start')}
        ro_ops = [rpm.get('RO opcode', 'RO') for rpm in round_parity_maps]
        ops = [f'{op} {a}' for a, op in zip(ancillas, ro_ops)]
        ops += [f'Acq {q}' for q in
                self.readout_rounds[readout_round]['dummy_readout_qbs']]
        ops += [f'RO {q}' for q in
                self.readout_rounds[readout_round].get('extra_readout_qbs', [])]
        if cycle == self.nr_cycles - 1:
            ops += [f'Acq {q}' for q in
                    self.readout_rounds[readout_round]\
                        ['dummy_readout_qbs_last_cycle']]
        ro_block = self.block_from_ops(element_name, ops,
                                       pulse_modifs=pulse_modifs)

        element_name = f'resets_{readout_round}_{cycle}'

        thresh_map = self.get_prep_params(ancillas).get('threshold_mapping', {})
        state_ops = dict(g=["I {a:}"], e=["X180 {a:}"],
                             f=["X180_ef {a:}", "X180 {a:}"])

        ancilla_reset_blocks = []
        for a in ancillas:
            if a not in thresh_map:
                raise ValueError(
                    f'Could not find threshold map for ancilla {a} in threshold map obtained '
                    f'from prep_params: {thresh_map}')
            ops_and_codewords = [(state_ops[s], c) for c, s in
                                 thresh_map[a].items()]
            # print(ops_and_codewords)
            cw_blocks = []
            for ops, c in ops_and_codewords:
                cw_blocks.append(self.block_from_ops(f'{element_name}_{a}_codeword_{c}',
                                                     ops, fill_values={'a': a},
                                                     pulse_modifs={i: {
                                                        "codeword": c,
                                                     "element_name": element_name}
                                                                 for i in
                                                                 range(len(ops))}))
            ancilla_reset_blocks.append(
                self.simultaneous_blocks(f"{element_name}_{a}", cw_blocks))
        reset_block = self.simultaneous_blocks(element_name, ancilla_reset_blocks)


        return ro_block, reset_block

    def _cycle_block(self, cycle=0):
        pulses = []
        round_delay = 0
        for readout_round, readout_round_pars in enumerate(self.readout_rounds):
            g = self._readout_round_gates_block(readout_round, cycle)
            r, i = self._readout_round_readout_block(readout_round, cycle)
            pulses += g.build(ref_pulse='start', block_delay=round_delay)
            if self.skip_last_ancilla_readout and cycle == self.nr_cycles - 1 \
                    and readout_round == len(self.readout_rounds) - 1 and \
                    not (self.ancilla_reset == "late"):
                continue
            ro_name = f'readouts_{readout_round}_{cycle}'
            round_delay += readout_round_pars['round_length']
            if self.ancilla_reset:
                if self.ancilla_reset == "late":
                    # put reset pulses right after last pulse on ancilla
                    ref_pulse = pulses[-1]['name'] # gate_block_end_name
                    block_delay = 10e-9

                    pulses += i.build(ref_pulse=ref_pulse, block_delay=block_delay)
                    if self.skip_last_ancilla_readout and cycle == self.nr_cycles - 1 \
                            and readout_round == len(self.readout_rounds) - 1:
                        continue
                    pulses += r.build(name=ro_name)

                else:
                    # referenced to start of previous ro + some delay
                    ref_pulse = ro_name + '-|-start'
                    block_delay = readout_round_pars['reset_delay']
                    pulses += r.build(name=ro_name)
                    pulses += i.build(ref_pulse=ref_pulse, block_delay=block_delay)
            else:
                pulses += r.build(name=ro_name)
        return block_mod.Block(f'cycle_{cycle}', pulses)


class ParityMap(dict):
    def __init__(self, ancilla=None, data=None, type=('Y90', 'Y90'), **kw):
        if ancilla is not None:
            kw.update({'ancilla': ancilla})
        if ancilla is not None:
            kw.update({'data': data})
        if type is not None:
            kw.update({'type': type})
        super().__init__(**kw)

    def _get(self, qubits='ancilla', return_type='str', exclude_none=False):
        qbs = self[qubits]
        return_string = False
        if not np.ndim(qbs) > 0:
            return_string = True
            qbs = [self[qubits]]

        if return_type == "str":
            # print(qbs)
            qubits_to_return = tuple(
                qb.name if qb is not None else None for qb in qbs)
        elif return_type == "obj":
            qubits_to_return = tuple(qb if qb is not None else None for qb in qbs)
        else:
            raise ValueError(f'return_type {return_type} not understood.')
        if exclude_none:
            qubits_to_return = tuple(
                qb for qb in qubits_to_return if qb is not None)
        if return_string:
            qubits_to_return = qubits_to_return[0]
        return qubits_to_return

    def ancilla(self, return_type="str"):
        return self._get(qubits='ancilla', return_type=return_type)

    def data(self, return_type="str", exclude_none=False):
        return self._get(qubits='data', return_type=return_type,
                         exclude_none=exclude_none)

    def string_copy(self):
        from copy import deepcopy
        cp = dict(ancilla=self.ancilla(return_type='str'),
                  data=self.data(return_type='str'))
        cp.update({k:v for k, v in self.items() if k not in ('ancilla', 'data')})
        return deepcopy(cp)