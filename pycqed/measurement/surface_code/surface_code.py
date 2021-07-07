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
                 ancilla_dd=True, skip_last_ancilla_readout=False,
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
                'element_name': 'final_element'}}},
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


        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancilla_steps = {pm['ancilla']: pm['data'] for pm in round_parity_maps}
        total_steps = max([len(v) for v in ancilla_steps.values()])
        ancilla_steps = {a: (total_steps - len(v)) // 2 * [None] +
                            list(v) + (total_steps - len(v) + 1) // 2 * [None]
                         for a, v in ancilla_steps.items()}
        # cz gates
        element_name = f'parity_map_entangle_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name,
                                    pulse_off=self.two_qb_gates_off)}
        gate_lists = [
            [(a, ds[s]) for a, ds in ancilla_steps.items() if ds[s] is not None]
            for s in range(total_steps)]
        cz_step_blocks = [
            self._parallel_cz_step_block(gates, pulse_modifs=pulse_modifs)
            for gates in gate_lists]

        # ancilla dd
        element_name = f'parity_map_ancilla_dd_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name)}
        if self.ancilla_dd:
            if total_steps % 2:
                raise NotImplementedError('Ancilla dynamical decoupling not '
                                          'implemented for odd weight parity maps.')
            ops = []
            for a, ds in ancilla_steps.items():
                ops += [f'Y180 {a}', f'Z180 {a}']
                ops += [f'Z180 {d}' for d in ds[total_steps//2:]
                            if d is not None]

            blocks = [self.block_from_ops(op, [op], pulse_modifs=pulse_modifs)
                      for op in ops]
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
        blocks = [init_block] + cz_step_blocks + [final_block]

        return self.sequential_blocks(element_name, blocks)

    def _readout_round_readout_block(self, readout_round, cycle=0):
        round_parity_maps = self.readout_rounds[readout_round]['parity_maps']
        ancillas = [pm['ancilla'] for pm in round_parity_maps]

        element_name = f'readouts_{readout_round}_{cycle}'
        pulse_modifs = {'all': dict(element_name=element_name,
                                    ref_point='start')}
        ops = [f'RO {a}' for a in ancillas]
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
        ops = [f'I {a}' for a in ancillas] + [f'X180 {a}' for a in ancillas]
        pulse_modifs = {i: dict(
            element_name=element_name,
            codeword=i // len(ancillas),
            ref_point='start')
            for i in range(len(ops))}
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
            if self.skip_last_ancilla_readout and cycle == self.nr_cycles - 1 \
                    and readout_round == len(self.readout_rounds) - 1:
                continue
            ro_name = f'readouts_{readout_round}_{cycle}'
            round_delay += readout_round_pars['round_length']
            if self.ancilla_reset:
                pulses += i.build(
                    ref_pulse=ro_name+'-|-start',
                    block_delay=readout_round_pars['reset_delay'])
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