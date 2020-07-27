
import pycqed.measurement.quantum_experiment as qe
import pycqed.measurement.waveform_control.block as block_mod

class SurfaceCodeExperiment(qe.QuantumExperiment):
    block_align = None
    type_to_ops_map = {
        'X': ('Y90', 'mY90'),
        'Y': ('mX90', 'X90'),
        'Z': (None, None),
    }

    def __init__(self, data_qubits, ancilla_qubits,
                 readout_rounds, nr_cycles, initializations,
                 finalizations, ancilla_reset=False,
                 ancilla_dd=True, **kw):
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
        self.cycle_length = sum([r['round_length']
                                 for r in self.readout_rounds])

    def main_block(self):
        """Block tree of the experiment:
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
        return block_mod.Block('main', pulses)

    def _parallel_cz_step_block(self, gates, dd_qubits=None, dd_pulse='X180',
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
        block_name = 'CZ_' + '_'.join([''.join(gate) for gate in gates])
        ops = ['CZ {} {}'.format(*gate) for gate in gates]
        ops += [dd_pulse + ' ' + dd_qb for dd_qb in dd_qubits]
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops]
        return self.simultaneous_blocks(block_name, blocks,
                                        block_align=self.block_align)

    def _readout_round_gates_block(self, readout_round, cycle=0):
        element_name = f'parity_maps_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name)}

        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancilla_steps = {pm['ancilla']: pm['data'] for pm in round_parity_maps}
        total_steps = max([len(v) for v in ancilla_steps.values()])
        if total_steps % 2:
            raise NotImplementedError('Code has not been written for odd weight'
                                      ' parity maps.')
        ancilla_steps = {a: (total_steps - len(v)) // 2 * [None] +
                            list(v) + (total_steps - len(v) + 1) // 2 * [None]
                         for a, v in ancilla_steps.items()}
        # cz gates
        gate_lists = [
            [(a, ds[s]) for a, ds in ancilla_steps.items() if ds[s] is not None]
            for s in range(total_steps)]
        cz_step_blocks = [
            self._parallel_cz_step_block(gates, pulse_modifs=pulse_modifs)
            for gates in gate_lists]

        # ancilla dd
        if self.ancilla_dd:
            ops = []
            for a, ds in ancilla_steps.items():
                ops += [f'Y180 {a}', f'Z180 {a}']
                ops += [f'Z180 {d}' for d in ds[total_steps//2:]]

            blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                      for op in ops]
            ancilla_dd_block = self.simultaneous_blocks(
                'ancilla_dd_block', blocks, block_align=self.block_align)
            cz_step_blocks = cz_step_blocks[:total_steps//2] + \
                [ancilla_dd_block] + cz_step_blocks[total_steps//2:]

        # data qubit and ancilla basis changes
        qubit_bases = {}
        for pm in round_parity_maps:
            assert qubit_bases.get(pm['ancilla'], 'X') == 'X'
            qubit_bases[pm['ancilla']] = 'X'
            for qb in pm['data']:
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
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops_init]
        init_block = self.simultaneous_blocks('init_block', blocks,
                                              block_align=self.block_align)
        blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                  for op in ops_final]
        final_block = self.simultaneous_blocks('final_block', blocks,
                                               block_align=self.block_align)
        blocks = [init_block] + cz_step_blocks + [final_block]

        return self.sequential_blocks(element_name, blocks)

    def _readout_round_readout_block(self, readout_round, cycle=0):
        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancillas = [pm['ancilla'] for pm in round_parity_maps]

        element_name = f'readouts_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name,
                                    ref_point='start')}
        ro_block = self.block_from_ops(element_name,
                                       [f'RO {a}' for a in ancillas],
                                       pulse_modifs=pulse_modifs)

        element_name = f'resets_{readout_round}_{cycle}'
        pulse_modifs = {i: dict(element_name=element_name,
                                codeword=i//len(ancillas),
                                ref_point_new='end')
                        for i in range(2*len(ancillas))}
        ops = [f'I {a}' for a in ancillas] + [f'X180 {a}' for a in ancillas]
        reset_block = self.block_from_ops(element_name, ops,
                                          pulse_modifs=pulse_modifs)

        return ro_block, reset_block

    def _cycle_block(self, cycle=0):
        pulses = []
        round_delay = 0
        for readout_round, readout_round_pars in enumerate(self.readout_rounds):
            g = self._readout_round_gates_block(readout_round, cycle)
            r, i = self._readout_round_readout_block(readout_round, cycle)
            pulses += g.build(ref_pulse='start', block_delay=round_delay)
            ro_name = f'readouts_{readout_round}_{cycle}'
            pulses += r.build(name=ro_name)
            round_delay += readout_round_pars['round_length']
            if self.ancilla_reset:
                pulses += i.build(
                    ref_pulse=ro_name+'-|-start',
                    block_delay=readout_round_pars['reset_delay'])
        return block_mod.Block(f'cycle_{cycle}', pulses)

