from pprint import pprint

import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq, prepend_pulses
import pycqed.measurement.waveform_control.segment as segment
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control import pulsar as ps



def qaoa_sequence(qb_names, betas, gammas, two_qb_gates_info, operation_dict,
                  init_state='0', cphase_implementation='hardware',
                  prep_params=None, upload=True):

    # create sequence, segment and builder
    seq_name = f'QAOA_{cphase_implementation}_cphase_{qb_names}'
    seg_name = f'segment_0'
    seq = sequence.Sequence(seq_name)
    seg = segment.Segment(seg_name)
    builder = QAOAHelper(qb_names, operation_dict)

    prep_params = {} if prep_params is None else prep_params

    # initialize qubits
    seg.extend(builder.initialize(init_state, prep_params=prep_params).build())
    pprint(builder.initialize(init_state, prep_params=prep_params).build())
    # # QAOA Unitaries
    # for k, (gamma, beta) in enumerate(zip(gammas, betas)):
    #     # Uk
    #     seg.extend(builder.U(f"U_{k}", two_qb_gates_info,
    #                gamma, cphase_implementation).build())
    #     # Dk
    #     seg.extend(builder.D(f"D_{k}", beta).build())
    #
    # # readout qubits
    # seg.extend(builder.mux_readout().build())

    seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())

class HelperBase:

    STD_INIT = {'0': 'I', '1': 'X180', '+': 'Y90', '-': 'mY90'}

    def __init__(self, qb_names, operation_dict):
        self.qb_names = qb_names
        self.operation_dict = operation_dict

    def get_qubits(self, qubits='all'):
        """
        Wrapper to get 'all' qubits, single qubit specified as string
        or list of qubits, checking they are in self.qb_names
        :param qubits: 'all', single qubit name (eg. 'qb1') or list of qb names
        :return: list of qb names
        """
        if qubits == 'all':
            return self.qb_names
        elif qubits in self.qb_names:  # qubits == single qb name eg. 'qb1'
             qubits = [qubits]
        for qb in qubits:
            assert qb in self.qb_names, f"{qb} not found in {self.qb_names}"
        return qubits

    def get_pulse(self, op, parse_z_gate=False):
        """
        Gets a pulse from the operation dictionary, and possibly parses
        arbitrary angle from Z gate operation.
        Examples:
             >>> get_pulse(['Z100 qb1'], parse_z_gate=True)
             will perform a 100 degree Z rotation
        Args:
            op: operation
            parse_z_gate: whether or not to look for Zgates with arbitrary angles.

        Returns: deepcopy of the pulse dictionary

        """
        if parse_z_gate and op.startswith("Z"):
            # assumes operation format of f"Z{angle} qbname"
            # FIXME: This parsing is format dependent and is far from ideal but
            #  until we can get parametrized pulses it is helpful to be able to
            #  parse Z gates
            angle, qbn = op.split(" ")[0][1:], op.split(" ")[1]
            p = self.get_pulse(f"Z180 {qbn}", parse_z_gate=False)
            p['basis_rotation'] = {qbn: float(angle)}
            return p

        return deepcopy(self.operation_dict[op])

    def initialize(self, init_state='0', qubits='all', prep_params=None,
                   simultaneous=True, block_name=None):
        """
        Initializes the specified qubits with the corresponding init_state
        :param init_state (String or list): Can be one of the following
            - one of the standard initializations: '0', '1', '+', '-'.
              In that case the same init_state is done on all qubits
            - list of standard init. Must then be of same length as 'qubits' and
              in the same order.
            - list of arbitrary pulses (which are in the operation_dict). Must be
              of the same lengths as 'qubits' and in the same order. Should not
              include space and qubit name (those are added internally).
        :param qubits (list or 'all'): list of qubits on which init should be
            applied. Defaults to all qubits.
        :param prep_params: preparation parameters
        :return: init segment
        """
        if block_name is None:
            block_name = f"Initialization_{qubits}"
        qubits = self.get_qubits(qubits)
        if prep_params is None:
            prep_params = {}
        if np.ndim(init_state) == 0:
            init_state = [init_state] * len(qubits)
        else:
            assert len(init_state) == len(qubits), \
                "There must be a one to one mapping between initializations and " \
                f"qubits. Got {len(init_state)} init and {len(qubits)} qubits"

        pulses = []
        pulses.extend(self.prepare(qubits, pulse_list=[self.Z_gate(qubits='qb1')],
                                   ref_pulse="start",  **prep_params).build())
        for i, (qbn, init) in enumerate(zip(qubits, init_state)):
            # add qb name and "s" for reference to start of previous pulse
            op = self.STD_INIT.get(init, init) + \
                 f"{'s' if len(pulses) != 0 and simultaneous else ''} " + qbn
            pulse = self.get_pulse(op)
            # if i == 0:
            #     pulse['ref_pulse'] = 'segment_start'
            pulses.append(pulse)

        # # TODO: note, add prep pulses could be in this class also.
        # pulses_with_prep = add_preparation_pulses(pulses, self.operation_dict,
        #                                           qubits, **prep_params)
        return Block(block_name, pulses)

    def prepare(self, qubits='all', pulse_list=None, ref_pulse='start',
                preparation_type='wait', post_ro_wait=1e-6,
                ro_separation=1.5e-6, reset_reps=3, final_reset_pulse=True,
                threshold_mapping=None, block_name=None):
        """
        Prepares specified qb for an experiment by creating preparation pulse for
        preselection or active reset.
        Args:
            qubits: which qubits to prepare. Defaults to all.
            pulse_list (optional): optional pulse list to which the preparation pulses
                will be added
            ref_pulse: reference pulse of the first pulse in the pulse list.
                reset pulse will be added in front of this. If the pulse list is empty,
                reset pulses will simply be before the block_start.
            preparation_type:
                for nothing: 'wait'
                for preselection: 'preselection'
                for active reset on |e>: 'active_reset_e'
                for active reset on |e> and |f>: 'active_reset_ef'
            post_ro_wait: wait time after a readout pulse before applying reset
            ro_separation: spacing between two consecutive readouts
            reset_reps: number of reset repetitions
            final_reset_pulse: whether or not to have a reset pulse at the end
                of the pulse list
            threshold_mapping (dict): thresholds mapping for each qb

        Returns:

        """
        if block_name is None:
            block_name = f"Preparation_{qubits}"
        qb_names = self.get_qubits(qubits)

        if pulse_list is None:
            pulse_list = []
        if threshold_mapping is None:
            threshold_mapping = {qbn: {0: 'g', 1: 'e'} for qbn in qb_names}

        # Calculate the length of a ge pulse, assumed the same for all qubits
        state_ops = dict(g=["I "], e=["X180 "], f=["X180_ef ", "X180 "])

        if len(pulse_list) > 0 and 'ref_pulse' not in pulse_list[0]:
            first_pulse = deepcopy(pulse_list[0])
            first_pulse['ref_pulse'] = ref_pulse
            pulse_list[0] = first_pulse

        if preparation_type == 'wait':
            return Block(block_name, pulse_list)
        elif 'active_reset' in preparation_type:
            reset_ro_pulses = []
            ops_and_codewords = {}
            for i, qbn in enumerate(qb_names):
                reset_ro_pulses.append(self.get_pulse('RO ' + qbn))
                reset_ro_pulses[-1]['ref_point'] = 'start' if i != 0 else 'end'

                if preparation_type == 'active_reset_e':
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1)]
                elif preparation_type == 'active_reset_ef':
                    assert len(threshold_mapping[qbn]) == 4, \
                        "Active reset for the f-level requires a mapping of length 4" \
                            f" but only {len(threshold_mapping)} were given: " \
                            f"{threshold_mapping}"
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1),
                        (state_ops[threshold_mapping[qbn][2]], 2),
                        (state_ops[threshold_mapping[qbn][3]], 3)]
                else:
                    raise ValueError(f'Invalid preparation type {preparation_type}')

            reset_pulses = []
            for i, qbn in enumerate(qb_names):
                for ops, codeword in ops_and_codewords[qbn]:
                    for j, op in enumerate(ops):
                        reset_pulses.append(self.get_pulse(op + qbn))
                        reset_pulses[-1]['codeword'] = codeword
                        if j == 0:
                            reset_pulses[-1]['ref_point'] = 'start'
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait
                        else:
                            reset_pulses[-1]['ref_point'] = 'start'
                            pulse_length = 0
                            for jj in range(1, j + 1):
                                if 'pulse_length' in reset_pulses[-1 - jj]:
                                    pulse_length += reset_pulses[-1 - jj]['pulse_length']
                                else:
                                    pulse_length += reset_pulses[-1 - jj]['sigma'] * \
                                                    reset_pulses[-1 - jj]['nr_sigma']
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait + pulse_length

            prep_pulse_list = []
            for rep in range(reset_reps):
                ro_list = deepcopy(reset_ro_pulses)
                ro_list[0]['name'] = 'refpulse_reset_element_{}'.format(rep)

                for pulse in ro_list:
                    pulse['element_name'] = 'reset_ro_element_{}'.format(rep)
                if rep == 0:
                    ro_list[0]['ref_pulse'] = 'segment_start'
                    ro_list[0]['pulse_delay'] = -reset_reps * ro_separation
                else:
                    ro_list[0]['ref_pulse'] = 'refpulse_reset_element_{}'.format(
                        rep - 1)
                    ro_list[0]['pulse_delay'] = ro_separation
                    ro_list[0]['ref_point'] = 'start'

                rp_list = deepcopy(reset_pulses)
                for j, pulse in enumerate(rp_list):
                    pulse['element_name'] = 'reset_pulse_element_{}'.format(rep)
                    pulse['ref_pulse'] = 'refpulse_reset_element_{}'.format(rep)
                prep_pulse_list += ro_list
                prep_pulse_list += rp_list

            if final_reset_pulse:
                rp_list = deepcopy(reset_pulses)
                for pulse in rp_list:
                    pulse['element_name'] = f'reset_pulse_element_{reset_reps}'
                pulse_list += rp_list
            print(prep_pulse_list, pulse_list)
            return Block(block_name, prep_pulse_list + pulse_list)

        elif preparation_type == 'preselection':
            preparation_pulses = []
            for i, qbn in enumerate(qb_names):
                preparation_pulses.append(self.get_pulse('RO ' + qbn))
                preparation_pulses[-1]['ref_point'] = 'start'
                preparation_pulses[-1]['element_name'] = 'preselection_element'
            preparation_pulses[0]['ref_pulse'] = 'segment_start'
            preparation_pulses[0]['pulse_delay'] = -ro_separation

            return Block(block_name, preparation_pulses + pulse_list)

    def mux_readout(self, qubits='all', element_name='RO',ref_point='end',
                    pulse_delay=0.0):
        block_name = "Readout"
        qubits = self.get_qubits(qubits)
        ro_pulses = []
        for j, qb_name in enumerate(qubits):
            ro_pulse = deepcopy(self.operation_dict['RO ' + qb_name])
            ro_pulse['name'] = '{}_{}'.format(element_name, j)
            ro_pulse['element_name'] = element_name
            if j == 0:
                ro_pulse['pulse_delay'] = pulse_delay
                ro_pulse['ref_point'] = ref_point
            else:
                ro_pulse['ref_point'] = 'start'
            ro_pulses.append(ro_pulse)
        return Block(block_name, ro_pulses)

    def Z_gate(self, theta=0, qubits='all'):

        """
        Software Z-gate of arbitrary rotation.

        :param theta:           rotation angle, in degrees
        :param qubits:      pulse parameters (dict)

        :return: Pulse dict of the Z-gate
        """
        # if qubits is the name of a qb, expects single pulse output
        single_qb_given = False
        if qubits in self.qb_names:
            single_qb_given = True
        qubits = self.get_qubits(qubits)

        pulses = []
        zgate_base_name = 'Z180'

        for qbn in qubits:
            zgate = deepcopy(self.operation_dict[zgate_base_name + f" {qbn}"])
            zgate['basis_rotation'] = {qbn: theta}
            pulses.append(zgate)

        return pulses[0] if single_qb_given else pulses

    def block_from_ops(self, block_name, operations, fill_values=None):
        """
        Returns a block with the given operations.
        Eg.
        >>> ops = ['X180 {qbt:}', 'X90 {qbc:}']
        >>> builder.block_from_ops("MyAwesomeBlock",
        >>>                                ops,
        >>>                                {'qbt': qb1, 'qbc': qb2})
        :param block_name: Name of the block
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values (dict): optional fill values for operations.
        :return:
        """
        return Block(block_name,
                     [self.get_pulse(op.format(**fill_values), True)
                      for op in operations])

class QAOAHelper(HelperBase):

    def U(self, name, gate_sequence_info, gamma, cphase_implementation):
        """
        Returns Unitary propagator pulse sequence (as a Block).
        :param name: name of the block
        :param gate_sequence_info (list) : list of list of information
            dictionaries. Dictionaries contain information about a two QB gate:
            assumes the following keys:
            - qbc: control qubit
            - qbt: target qubit
            - gate_name: name of the 2 qb gate
            - C: coupling btw the two qubits
            - (arb_phase_func): only required when using hardware implementation
               of arbitrary phase gate.
            All dictionaries within the same sub list are executed simultaneously
            Example:
            >>> [
            >>>     # first set of 2qb gates to run together
            >>>     [dict(qbc='qb1', qbt='qb2', gate_name='upCZ qb2 qb1', C=1,
            >>>           arb_phase_func=func_qb1_qb2),
            >>>      dict(qbc='qb4', qbt='qb3', gate_name='upCZ qb4 qb3', C=1,
            >>>           arb_phase_func=func_qb4_qb3)],
            >>>     # second set of 2qb gates
            >>>     [dict(qbc='qb3', qbt='qb2', gate_name='upCZ qb2 qb3', C=1,
            >>>        arb_phase_func=func_qb3_qb2)]
            >>> ]
        :param gamma: rotation angle (in rad)
        :param cphase_implementation: implementation of arbitrary phase gate.
            "software" --> gate is decomposed into single qb gates and 2x CZ gate
            "hardware" --> hardware arbitrary phase gate
        :return: Unitary U (Block)
        """

        assert cphase_implementation in ("software", "hardware")

        if cphase_implementation == "software":
            raise NotImplementedError()

        U = Block(name, [])
        for i, two_qb_gates_same_timing in enumerate(gate_sequence_info):
            simult_bname = f"simultanenous_{i}"
            simultaneous = Block(simult_bname, [])
            for two_qb_gates_info in two_qb_gates_same_timing:
                #gate info
                qbc = two_qb_gates_info["qbc"]
                qbt = two_qb_gates_info["qbt"]
                gate_name = two_qb_gates_info['gate_name']
                C = two_qb_gates_info["C"]

                #virtual gate on qb 0
                z_qbc = self.Z_gate(gamma * C * 180 / np.pi, qbc)

                # virtual gate on qb 1
                z_qbt = self.Z_gate(gamma * C * 180 / np.pi, qbt)

                #arbitrary phase gate
                c_arb_pulse = self.operation_dict[gate_name]
                #get amplitude and dynamic phase from model
                ampl, dyn_phase = two_qb_gates_info['arb_phase_func'](2 * gamma * C)
                c_arb_pulse['amplitude'] = ampl
                c_arb_pulse['element_name'] = "flux_arb_gate"
                c_arb_pulse['basis_rotation'].update(
                    {two_qb_gates_info['qbc']: dyn_phase})

                two_qb_block = Block(f"qbc:{qbc} qbt:{qbt}",
                                     [z_qbc, z_qbt, c_arb_pulse])

                # FIXME: not that nice that we have to add the "runtime"name of the ref
                #  block i.e. prepending all higher level block. I guess a solution
                #  would be to check recursively all reference to the pulse before
                #  changing its name but this might be slow.
                simultaneous.extend(
                    two_qb_block.build(ref_pulse=f"{simult_bname}_start"))
            # add block referenced to start of U_k
            U.extend(simultaneous.build())

        return U

    def D(self, name, beta, qubits='all'):
        if qubits == 'all':
            qubits = self.qb_names

        pulses = []
        ops = ["mY90 {qbn:}", "Z{angle:} {qbn:}", "Y90 {qbn:}"]
        for qbn in qubits:
            D_qbn = self.block_from_ops(f"{qbn}", ops,
                                        dict(qbn=qbn, angle=beta * 180 / np.pi))
            # reference block to beginning of D_k block
            pulses.extend(D_qbn.build(ref_pulse=f"{name}_start"))
        return Block(name, pulses)