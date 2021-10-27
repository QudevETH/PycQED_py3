import itertools
import numpy as np
from copy import copy
from copy import deepcopy
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control.block import ParametricValue
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.segment import Segment
from pycqed.measurement import multi_qubit_module as mqm


class CircuitBuilder:
    """
    A class that helps to build blocks, segments, or sequences, e.g.,
    when implementing quantum algorithms.

    :param dev: the device on which the algorithm will be executed
    :param qubits: a list of qubit objects or names if the builder should
        act only on a subset of qubits (default: all qubits of the device)
    :param kw: keyword arguments
         cz_pulse_name: (str) the prefix of CZ gates (default: upCZ)
         decompose_rotation_gates: (dict of bool) whether arbitrary
            rotation gates should be decomposed into pi rotations
            and virtual Z gates, e.g., {'X': True, 'Y': False}.
            False means direct implementation. For gate types not specified
            here, the default is False.
         prep_params: (dict) custom preparation params (default: from
            instrument settings)
    """

    STD_INIT = {'0': ['I'], '1': ['X180'], '+': ['Y90'], '-': ['mY90'],
                'g': ['I'], 'e': ['X180'], 'f': ['X180', 'X180_ef']}
    STD_PREP_PARAMS = {'preparation_type': 'wait', 'reset_reps': 3,
                       'ro_separation': 1.5e-6, 'post_ro_wait': 1e-6}

    def __init__(self, dev=None, qubits=None, operation_dict=None,
                 filter_qb_names=None, **kw):

        self.dev = dev
        self.qubits, self.qb_names = self.extract_qubits(
            dev, qubits, operation_dict, filter_qb_names)
        self.update_operation_dict(operation_dict)
        self.cz_pulse_name = kw.get('cz_pulse_name', 'upCZ')
        self.decompose_rotation_gates = kw.get('decompose_rotation_gates', {})
        self.prep_params = kw.get('prep_params', None)

    @staticmethod
    def extract_qubits(dev=None, qubits=None, operation_dict=None,
                       filter_qb_names=None):
        assert (dev is not None or qubits is not None or operation_dict is
                not None), \
            "Either dev or qubits or operation_dict has to be provided."
        if dev is None and qubits is None:
            qb_names = list(np.unique([qb for op in operation_dict.keys()
                                       for qb in op.split(' ')[1:]]))
        elif dev is None:
            qb_names = [qb if isinstance(qb, str) else qb.name for qb in
                        qubits]
            if any([isinstance(qb, str) for qb in qubits]):
                qubits = None
        else:
            if qubits is None:
                qubits = dev.get_qubits()
            else:
                # get qubit objects if names have been provided
                qubits = [dev.get_qb(qb) if isinstance(qb, str) else qb for
                          qb in qubits]
            qb_names = [qb.name for qb in qubits]
        if filter_qb_names is not None:
            if qubits is not None:
                qubits = [qb for qb in qubits if qb.name in filter_qb_names]
            qb_names = [qb for qb in qb_names if qb in filter_qb_names]
        return qubits, qb_names

    def update_operation_dict(self, operation_dict=None):
        """
        Updates the stored operation_dict based on the passed operation_dict or
        based on the stored device/qubit objects.
        :param operation_dict: (optional) The operation dict to be used. If
            not provided, an operation dict is generated from  the stored
            device/qubit objects.
        :return:
        """
        if operation_dict is not None:
            self.operation_dict = deepcopy(operation_dict)
        elif self.dev is not None:
            self.operation_dict = deepcopy(self.dev.get_operation_dict())
        else:
            self.operation_dict = deepcopy(mqm.get_operation_dict(self.qubits))

    def get_qubits(self, qb_names=None):
        """
        Wrapper to get 'all' qubits, single qubit specified as string
        or list of qubits, checking they are in self.qubits
        :param qb_names: 'all', single qubit name (eg. 'qb1') or list of
            qb names
        :return: list of qubit object and list of qubit names (first return
            value is None if no qubit objects are stored). The order is
            according to the order stored in self.qb_names (which can be
            modified by self.swap_qubit_indices()).
        """
        if qb_names is None or qb_names == 'all':
            return self.get_qubits(self.qb_names)
        elif not isinstance(qb_names, list):
            qb_names = [qb_names]

        # test if qubit objects have been provided instead of names
        qb_names = [qb if isinstance(qb, str) or isinstance(qb, int)
                    else qb.name for qb in qb_names]
        # test if qubit indices have been provided instead of names
        try:
            ind = [int(i) for i in qb_names]
            qb_names = [self.qb_names[i] for i in ind]
        except ValueError:
            pass

        for qb in qb_names:
            assert qb in self.qb_names, f"{qb} not found in {self.qb_names}"
        if self.qubits is None:
            return None, qb_names
        else:
            qb_map = {qb.name: qb for qb in self.qubits}
            return [qb_map[qb] for qb in qb_names], qb_names

    def get_prep_params(self, qb_names='all'):
        qubits, qb_names = self.get_qubits(qb_names)
        if self.prep_params is not None:
            return self.prep_params
        elif self.dev is not None:
            return self.dev.get_prep_params(qubits)
        elif qubits is not None:
            return mqm.get_multi_qubit_prep_params(qubits)
        else:
            return self.STD_PREP_PARAMS

    def get_cz_operation_name(self, qb1=None, qb2=None, op_code=None, **kw):
        """
        Finds the name of the CZ gate between qb1-qb2 that exists in
        self.operation_dict.
        :param qb1: name of qubit object of one of the gate qubits
        :param qb2: name of qubit object of the other gate qubit
        :param op_code: provide an op_code instead of qb1 and qb2

        :param kw: keyword arguments:
            cz_pulse_name: a custom cz_pulse_name instead of the stored one

        :return: the CZ gate name
        """
        assert (qb1 is None and qb2 is None and op_code is not None) or \
               (qb1 is not None and qb2 is not None and op_code is None), \
            "Provide either qb1&qb2 or op_code!"
        cz_pulse_name = kw.get('cz_pulse_name', self.cz_pulse_name)
        if op_code is not None:
            op_split = op_code.split(' ')
            qb1, qb2 = op_split[1:]
            if op_split[0] != 'CZ':
                cz_pulse_name = op_split[0]

        _, (qb1, qb2) = self.get_qubits([qb1, qb2])
        if f"{cz_pulse_name} {qb1} {qb2}" in self.operation_dict:
            return f"{cz_pulse_name} {qb1} {qb2}"
        elif f"{cz_pulse_name} {qb2} {qb1}" in self.operation_dict:
            return f"{cz_pulse_name} {qb2} {qb1}"
        else:
            raise KeyError(f'CZ gate "{cz_pulse_name} {qb1} {qb2}" not found.')

    def get_pulse(self, op, parse_rotation_gates=False):
        """
        Gets a pulse from the operation dictionary, and possibly parses
        logical indexing as well as arbitrary angle from Z gate operation.
        Examples:
             >>> get_pulse('CZ 0 2', parse_rotation_gates=True)
             will perform a CZ gate (according to cz_pulse_name)
             between the qubits with logical indices 0 and 2
             >>> get_pulse('Z100 qb1', parse_rotation_gates=True)
             will perform a 100 degree Z rotation
             >>> get_pulse('Z:theta qb1', parse_rotation_gates=True)
             will perform a parametric Z rotation with parameter name theta
             >>> get_pulse('Z:2*[theta] qb1', parse_rotation_gates=True)
             will perform a parametric Z rotation with twice the
             value of the parameter named theta. The brackets are used to
             indicated the parameter name. This feature has also been tested
             with some more complicated mathematical expression.
             Note: the mathematical expression should be entered without any
             spaces in between operands. For instance,
             'Z:2*[theta]/180+[theta]**2 qb1' and not
             'Z: 2 * [theta] / 180 + [theta]**2 qb1
        Adding 's' (for simultaneous) in front of an op_code (e.g.,
        'sZ:theta qb1') will reference the pulse to the start of the
        previous pulse.
        Adding 'm' (for minus) in front of an op_code (e.g.,
        'mZ:theta qb1') negates the sign. If 's' is also given,
        it has to be in the order 'sm'.

        Args:
            op: operation (str in the above format, or iterable
            corresponding to the splitted string)
            parse_rotation_gates: whether or not to look for gates with
            arbitrary angles.

        Returns: deepcopy of the pulse dictionary

        """
        op_info = op.split(" ") if isinstance(op, str) else op
        # the call to get_qubits resolves qubits indices if needed
        _, op_info[1:] = self.get_qubits(op_info[1:])
        op_name = op_info[0][1:] if op_info[0][0] == 's' else op_info[0]
        op = op_name + ' ' + ' '.join(op_info[1:])

        if op_info[0].rstrip('0123456789.') == 'CZ' or \
                op_info[0].startswith('CZ:'):
            operation = self.get_cz_operation_name(op_info[1], op_info[2])
            p = deepcopy(self.operation_dict[operation])
        elif parse_rotation_gates and op not in self.operation_dict:
            # assumes operation format of, e.g., f" Z{angle} qbname"
            # FIXME: This parsing is format dependent and is far from ideal but
            #  to generate parametrized pulses it is helpful to be able to
            #  parse Z gates etc.
            factor = -1 if op_name[0] == 'm' else 1
            if factor == -1:
                op_name = op_name[1:]
            if op_name[0] not in ['X', 'Y', 'Z']:
                raise KeyError(f'Gate "{op}" not found.')
            angle, qbn = op_name[1:], op_info[1]
            if angle[-1] == 's' and angle[:-1].isnumeric():
                # For non-parametric gates, an alternative syntax for
                # simultaneous pulses with appended s instead of prepended s
                # is allowed. This code branch treats, e.g., simultaneous
                # virtual Z pulses (like 'Z90s qb1') and simultaneous X/Y
                # rotations with angles different from 90 and 180
                # (like 'X45s qb1'; X90s and X180s are contained in the
                # operations dict anyways, but no other angles).
                op_info[0] = 's' + op_info[0]
                angle = angle[:-1]
            param = None
            if angle[0] == ':':  # angle depends on a parameter
                angle = angle[1:]
                param_start = angle.find('[') + 1
                # If '[' is contained, this indicates that the parameter
                # is part of a mathematical expression. Otherwise, the angle
                # is equal to the parameter.
                if param_start > 0:
                    param_end = angle.find(']', param_start)
                    param = angle[param_start:param_end]
                    angle = angle.replace('[' + param + ']', 'x')
                    f = eval('lambda x : ' + angle)
                else:
                    param = angle

            if self.decompose_rotation_gates.get(op_name[0], False):
                raise NotImplementedError('Decomposed rotations not '
                                          'implemented yet.')
            else:
                p = self.get_pulse(f"{op_name[0]}180 {qbn}")
                if op_name[0] == 'Z':
                    if param is not None:  # angle depends on a parameter
                        if param_start > 0:  # via a mathematical expression
                            func = (lambda x, qb=op_info[1], f=factor,
                                          fnc=eval('lambda x : ' + angle):
                                    {qb: f * fnc(x)})
                        else:  # angle = parameter
                            func = (lambda x, qbn=op_info[1], f=factor:
                                    {qbn: f * x})
                        p['basis_rotation'] = ParametricValue(
                            param, func=func, op_split=(op_name, op_info[1]))
                    else:  # angle is a given value
                        # configure virtual Z gate for this angle
                        p['basis_rotation'] = {qbn: factor * float(angle)}
                else:
                    if param is not None:  # angle depends on a parameter
                        if param_start > 0:  # via a mathematical expression
                            # combine the mathematical expression with a
                            # function that calculates the amplitude
                            func = (
                                lambda x, a=p['amplitude'], f=factor,
                                       fnc=eval('lambda x : ' + angle):
                                a / 180 * ((f * fnc(x) + 180) % (-360) + 180))
                        else:  # angle = parameter
                            func = lambda x, a=p['amplitude'], f=factor: \
                                a / 180 * ((f * x + 180) % (-360) + 180)
                        p['amplitude'] = ParametricValue(
                            param, func=func, op_split=(op_name, op_info[1]))
                    else:  # angle is a given value
                        angle = factor * float(angle)
                        # configure drive pulse amplitude for this angle
                        p['amplitude'] *= ((angle + 180) % (-360) + 180) / 180
        else:
            p = deepcopy(self.operation_dict[op])
        p['op_code'] = op
        if op_info[0][0] == 's':
            p['ref_point'] = 'start'

        return p

    def swap_qubit_indices(self, i, j=None):
        """
        Swaps logical qubit indices by swapping the entries in self.qb_names.
        :param i: (int or iterable): index of the first qubit to be swapped or
            indices of the two qubits to be swapped (as two ints given in the
            first two elements of the iterable)
        :param j: index of the second qubit (if it is not set via param i)
        """
        if j is None:
            i, j = i[0], i[1]
        self.qb_names[i], self.qb_names[j] = self.qb_names[j], self.qb_names[i]

    def initialize(self, init_state='0', qb_names='all', prep_params=None,
                   simultaneous=True, block_name=None, pulse_modifs=None):
        """
        Initializes the specified qubits with the corresponding init_state
        :param init_state (String or list): Can be one of the following
            - one of the standard initializations: '0', '1', '+', '-'.
              In that case the same init_state is done on all qubits
            - list of standard init. Must then be of same length as 'qubits' and
              in the same order.
            - list of arbitrary pulses (which are in the operation_dict). Must
              be of the same lengths as 'qubits' and in the same order. Should
              not include space and qubit name (those are added internally).
        :param qb_names (list or 'all'): list of qubits on which init should be
            applied. Defaults to all qubits.
        :param prep_params: preparation parameters
        :param simultaneous: (bool, default True) whether initialization
            pulses should be applied simultaneously.
        :param block_name: (str, optional) a name to replace the
            automatically generated block name of the initialization block
        :param pulse_modifs: (dict) Modification of pulses parameters.
            See method block_from_ops.
        :return: init block
        """
        if block_name is None:
            block_name = f"Initialization_{qb_names}"
        _, qb_names = self.get_qubits(qb_names)
        if prep_params is None:
            prep_params = self.get_prep_params(qb_names)
        if len(init_state) == 1:
            init_state = [init_state] * len(qb_names)
        else:
            assert len(init_state) == len(qb_names), \
                f"There must be a one to one mapping between initializations " \
                f"and qubits. Got {len(init_state)} init and {len(qb_names)} " \
                f"qubits"

        pulses = []
        for i, (qbn, init) in enumerate(zip(qb_names, init_state)):
            # Allowing for a list of pulses here makes it possible to,
            # e.g., initialize in the f-level.
            if not isinstance(init, list):
                init = self.STD_INIT.get(init, [init])
            if init != ['I']:
                init = [f"{op} {qbn}" for op in init]
                # We just want the pulses, but we can use block_from_ops as
                # a helper to get multiple pulses and to process pulse_modifs
                tmp_block = self.block_from_ops(
                    'tmp_block', init, pulse_modifs=pulse_modifs)
                if simultaneous:
                    tmp_block.pulses[0]['ref_pulse'] = 'start'
                pulses += tmp_block.pulses
        block = Block(block_name, pulses)
        block.set_end_after_all_pulses()
        if len(prep_params) != 0:
            block = self.sequential_blocks(
                block_name, [self.prepare(qb_names, ref_pulse="start",
                                          **prep_params), block])
        return block

    def finalize(self, init_state='0', qb_names='all', simultaneous=True,
                 block_name=None, pulse_modifs=None):
        """
        Applies the specified final rotation to the specified qubits.
        This is basically the same initialize, but without preparation.
        For parameters, see initialize().
        :return: finalization block
        """
        if block_name is None:
            block_name = f"Finalialization_{qb_names}"
        return self.initialize(init_state=init_state, qb_names=qb_names,
                               simultaneous=simultaneous,
                               prep_params={},
                               block_name=block_name,
                               pulse_modifs=pulse_modifs)

    def prepare(self, qb_names='all', ref_pulse='start',
                preparation_type=STD_PREP_PARAMS['preparation_type'],
                post_ro_wait=STD_PREP_PARAMS['post_ro_wait'],
                ro_separation=STD_PREP_PARAMS['ro_separation'],
                reset_reps=STD_PREP_PARAMS['reset_reps'], final_reset_pulse=False,
                pad_end=False, threshold_mapping=None, block_name=None):
        """
        Prepares specified qb for an experiment by creating preparation pulse
        for preselection or active reset.
        Args:
            qb_names: which qubits to prepare. Defaults to all.
            ref_pulse: reference pulse of the first pulse in the pulse list.
                reset pulse will be added in front of this.
                If the pulse list is empty,
                reset pulses will simply be before the block_start.
            preparation_type:
                for nothing: 'wait'
                for preselection: 'preselection'
                for active reset on |e>: 'active_reset_e'
                for active reset on |e> and |f>: 'active_reset_ef'
            post_ro_wait: wait time after a readout pulse before applying reset
            ro_separation: spacing between two consecutive readouts
            reset_reps: number of reset repetitions
            final_reset_pulse: Note: NOT used in this function.
            threshold_mapping (dict): thresholds mapping for each qb
            pad_end (bool): Only used in active reset. Whether or not padding
                should be added after the last reset readout pulse. If False,
                no padding is added and therefore any subsequent pulse will start
                right after the last reset pulse. If True, then the end of the
                prepare block is set such that the subsequent pulse start
                "ro_separation" after the start of the last readout pulse.


        Returns:

        """
        if block_name is None:
            block_name = f"Preparation_{qb_names}"
        _, qb_names = self.get_qubits(qb_names)

        if threshold_mapping is None or len(threshold_mapping) == 0:
            threshold_mapping = {qbn: {0: 'g', 1: 'e'} for qbn in qb_names}

        # Calculate the length of a ge pulse, assumed the same for all qubits
        state_ops = dict(g=["I "], e=["X180 "], f=["X180_ef ", "X180 "])

        # no preparation pulses
        if preparation_type == 'wait':
            return Block(block_name, [])

        # active reset
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
                        f"Active reset for the f-level requires a mapping of " \
                        f"length 4 but only {len(threshold_mapping)} were " \
                        f"given: {threshold_mapping}"
                    ops_and_codewords[qbn] = [
                        (state_ops[threshold_mapping[qbn][0]], 0),
                        (state_ops[threshold_mapping[qbn][1]], 1),
                        (state_ops[threshold_mapping[qbn][2]], 2),
                        (state_ops[threshold_mapping[qbn][3]], 3)]
                else:
                    raise ValueError(f'Invalid preparation type '
                                     f'{preparation_type}')

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
                                    pulse_length += reset_pulses[-1 - jj][
                                        'pulse_length']
                                else:
                                    pulse_length += \
                                        reset_pulses[-1 - jj]['sigma'] * \
                                        reset_pulses[-1 - jj]['nr_sigma']
                            reset_pulses[-1]['pulse_delay'] = post_ro_wait + \
                                                              pulse_length

            prep_pulse_list = []
            for rep in range(reset_reps):
                ro_list = deepcopy(reset_ro_pulses)
                ro_list[0]['name'] = 'refpulse_reset_element_{}'.format(rep)

                for pulse in ro_list:
                    pulse['element_name'] = 'reset_ro_element_{}'.format(rep)
                if rep == 0:
                    ro_list[0]['ref_pulse'] = ref_pulse
                    ro_list[0]['pulse_delay'] = -reset_reps * ro_separation
                else:
                    ro_list[0]['ref_pulse'] = \
                        'refpulse_reset_element_{}'.format(rep - 1)
                    ro_list[0]['pulse_delay'] = ro_separation
                    ro_list[0]['ref_point'] = 'start'

                rp_list = deepcopy(reset_pulses)
                for j, pulse in enumerate(rp_list):
                    pulse['element_name'] = f'reset_pulse_element_{rep}'
                    pulse['ref_pulse'] = f'refpulse_reset_element_{rep}'
                prep_pulse_list += ro_list
                prep_pulse_list += rp_list

            if pad_end:
                block_end = dict(
                    name='end', pulse_type="VirtualPulse",
                    ref_pulse=f'refpulse_reset_element_{reset_reps - 1}',
                    pulse_delay=ro_separation, ref_point="start")
                prep_pulse_list += [block_end]
            return Block(block_name, prep_pulse_list)

        # preselection
        elif preparation_type == 'preselection':
            preparation_pulses = []
            for i, qbn in enumerate(qb_names):
                preparation_pulses.append(self.get_pulse('RO ' + qbn))
                preparation_pulses[-1]['ref_point'] = 'start'
                preparation_pulses[-1]['element_name'] = 'preselection_element'
            preparation_pulses[0]['ref_pulse'] = ref_pulse
            preparation_pulses[0]['name'] = 'preselection_RO'
            preparation_pulses[0]['pulse_delay'] = -ro_separation
            block_end = dict(name='end', pulse_type="VirtualPulse",
                             ref_pulse='preselection_RO',
                             pulse_delay=ro_separation,
                             ref_point='start')
            preparation_pulses += [block_end]
            return Block(block_name, preparation_pulses)

    def mux_readout(self, qb_names='all', element_name='RO', block_name="Readout",
                    **pulse_pars):
        _, qb_names = self.get_qubits(qb_names)
        ro_pulses = []
        for j, qb_name in enumerate(qb_names):
            ro_pulse = deepcopy(self.operation_dict['RO ' + qb_name])
            ro_pulse['name'] = '{}_{}'.format(element_name, j)
            ro_pulse['element_name'] = element_name
            if j == 0:
                ro_pulse.update(pulse_pars)
            else:
                ro_pulse['ref_point'] = 'start'
            ro_pulses.append(ro_pulse)
        block = Block(block_name, ro_pulses)
        block.set_end_after_all_pulses()
        return block

    def Z_gate(self, theta=0, qb_names='all'):

        """
        Software Z-gate of arbitrary rotation.

        :param theta:           rotation angle, in degrees
        :param qb_names:      pulse parameters (dict)

        :return: Pulse dict of the Z-gate
        """

        # if qb_names is the name of a single qb, expects single pulse output
        single_qb_given = not isinstance(qb_names, list)
        _, qb_names = self.get_qubits(qb_names)
        pulses = [self.get_pulse(f'Z{theta} {qbn}', True) for qbn in qb_names]
        return pulses[0] if single_qb_given else pulses

    def get_ops_duration(self, operations=None, pulses=None, fill_values=None,
                         pulse_modifs=None, init_state='0'):
        """
        Calculates the total duration of the operations by resolving a dummy
        segment created from operations.
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values: optional fill values for operations (dict),
            see documentation of block_from_ops().
        :param pulse_modifs: Modification of pulses parameters (dict),
            see documentation of block_from_ops().
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :return: the duration of the operations
        """
        if pulses is None:
            if operations is None:
                raise ValueError('Please provide either "pulses" or '
                                 '"operations."')
            pulses = self.initialize(init_state=init_state).build()
            pulses += self.block_from_ops("Block1", operations,
                                          fill_values=fill_values,
                                          pulse_modifs=pulse_modifs).build()

        seg = Segment('Segment 1', pulses)
        seg.resolve_timing()
        # Using that seg.resolved_pulses was sorted by seg.resolve_timing()
        pulse = seg.resolved_pulses[-1]
        duration = pulse.pulse_obj.algorithm_time() + pulse.pulse_obj.length
        return duration

    def block_from_ops(self, block_name, operations, fill_values=None,
                       pulse_modifs=None):
        """
        Returns a block with the given operations.
        Eg.
        >>> ops = ['X180 {qbt:}', 'X90 {qbc:}']
        >>> builder.block_from_ops("MyAwesomeBlock",
        >>>                                ops,
        >>>                                {'qbt': qb1, 'qbc': qb2})
        :param block_name: Name of the block
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values.
            Instead of a str, each list entry can also be an iterable
            corresponding to the splitted string.
        :param fill_values (dict): optional fill values for operations.
        :param pulse_modifs (dict): Modification of pulses parameters.
            keys:
             -indices of the pulses on  which the pulse modifications should be
             made (backwards compatible)
             -
             values: dictionaries of modifications
            E.g. ops = ["X180 qb1", "Y90 qb2"],
            pulse_modifs = {1: {"ref_point": "start"}}
            This will modify the pulse "Y90 qb2" and reference it to the start
            of the first one.
        :return: The created block
        """
        def op_format(op, **fill_values):
            if isinstance(op, str):
                return op.format(**fill_values)
            else:
                return [s.format(**fill_values) for s in op]

        if fill_values is None:
            fill_values = {}
        if pulse_modifs is None:
            pulse_modifs = {}
        if isinstance(operations, str):
            operations = [operations]
        pulses = [self.get_pulse(op_format(op, **fill_values), True)
                  for op in operations]

        return Block(block_name, pulses, pulse_modifs)

    def seg_from_cal_points(self, cal_points, init_state='0', ro_kwargs=None,
                            block_align='end', segment_prefix='calibration_',
                            **kw):
        """
        Returns a list of segments for each cal state in cal_points.states.
        :param cal_points: CalibrationPoints instance
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :param block_align: passed to simultaneous_blocks; see docstring there
        :param segment_prefix: prefix for segment name (string)
        :param kw: keyword arguments (to allow pass through kw even if it
            contains entries that are not needed)
        :return: list of Segment instances
        """
        if ro_kwargs is None:
            ro_kwargs = {}

        segments = []
        for i, seg_states in enumerate(cal_points.states):
            cal_ops = [[f'{p}{qbn}' for p in cal_points.pulse_label_map[s]]
                       for s, qbn in zip(seg_states, cal_points.qb_names)]
            qb_blocks = [self.block_from_ops(
                f'body_block_{i}_{o}', ops,
                pulse_modifs=cal_points.pulse_modifs)
                for o, ops in enumerate(cal_ops)]
            parallel_qb_block = self.simultaneous_blocks(
                f'parallel_qb_blk_{i}', qb_blocks, block_align=block_align)

            prep = self.initialize(init_state=init_state,
                                   qb_names=cal_points.qb_names)
            ro = self.mux_readout(**ro_kwargs, qb_names=cal_points.qb_names)
            cal_state_block = self.sequential_blocks(
                f'cal_states_{i}', [prep, parallel_qb_block, ro])
            seg = Segment(f'{segment_prefix}_{i}_{"".join(seg_states)}',
                          cal_state_block.build())
            segments.append(seg)

        return segments

    def seg_from_ops(self, operations, fill_values=None, pulse_modifs=None,
                     init_state='0', seg_name='Segment1', ro_kwargs=None):
        """
        Returns a segment with the given operations using the function
        block_from_ops().
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values: optional fill values for operations (dict),
            see documentation of block_from_ops().
        :param pulse_modifs: Modification of pulses parameters (dict),
            see documentation of block_from_ops().
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seg_name: Name (str) of the segment (default: "Segment1")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :return: The created segment
        """
        if ro_kwargs is None:
            ro_kwargs = {}
        pulses = self.initialize(init_state=init_state).build()
        pulses += self.block_from_ops("Block1", operations,
                                      fill_values=fill_values,
                                      pulse_modifs=pulse_modifs).build()
        pulses += self.mux_readout(**ro_kwargs).build()
        return Segment(seg_name, pulses)

    def seq_from_ops(self, operations, fill_values=None, pulse_modifs=None,
                     init_state='0', seq_name='Sequence', ro_kwargs=None):
        """
        Returns a sequence with the given operations using the function
        block_from_ops().
        :param operations: list of operations (str), which can be preformatted
            and later filled with values in the dictionary fill_values
        :param fill_values: optional fill values for operations (dict),
            see documentation of block_from_ops().
        :param pulse_modifs: Modification of pulses parameters (dict),
            see documentation of block_from_ops().
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seq_name: Name (str) of the sequence (default: "Sequence")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :return: The created sequence
        """
        seq = Sequence(seq_name)
        seq.add(self.seg_from_ops(operations=operations,
                                  fill_values=fill_values,
                                  pulse_modifs=pulse_modifs,
                                  init_state=init_state,
                                  ro_kwargs=ro_kwargs))
        return seq

    def simultaneous_blocks(self, block_name, blocks, block_align='start',
                            set_end_after_all_pulses=False,
                            disable_block_counter=False):
        """
        Creates a block with name :block_name: that consists of the parallel
        execution of the given :blocks:. Ensures that any pulse or block
        following the created block will occur after the longest given block.

        CAUTION: For each of the given blocks, the end time of the block is
        determined by the pulse listed last in the block, which is not
        necessarily the one that ends last in terms of timing. To instead
        determine the end time of the block based on the pulse that ends
        last, set set_end_after_all_pulses to True (or adjust the end pulse
        of each block before calling simultaneous_blocks).

        Args:
            block_name (string): name of the block that is created
            blocks (iterable): iterable where each element is a block that has
                to be executed in parallel to the others.
            block_align (str or float): at which point the simultaneous
                blocks should be aligned ('start', 'middle', 'end', or a float
                between 0.0 and 1.0 that determines the alignment point of each
                block relative to the duration the block). Default: 'start'
            set_end_after_all_pulses (bool, default False): in all
                blocks, correct the end pulse to happen after the last pulse.
            disable_block_counter (bool, default False): prevent block.build
                from appending a counter to the block name.
        """

        simultaneous = Block(block_name, [])
        simultaneous_end_pulses = []
        if block_align == 'start':
            # saves computation time in Segment.resolve_timing
            block_align = None
        for block in blocks:
            if set_end_after_all_pulses:
                block.set_end_after_all_pulses()
            simultaneous.extend(block.build(
                ref_pulse=f"start", block_start=dict(block_align=block_align),
                name=block.name if disable_block_counter else None))
            simultaneous_end_pulses.append(simultaneous.pulses[-1]['name'])
        # the name of the simultaneous_end_pulse is used in
        # Segment.resolve_timing and should not be changed
        simultaneous.extend([{"name": f"simultaneous_end_pulse",
                              "pulse_type": "VirtualPulse",
                              "pulse_delay": 0,
                              "ref_pulse": simultaneous_end_pulses,
                              "ref_point": 'end',
                              "ref_function": 'max'
                              }])
        return simultaneous

    def sequential_blocks(self, block_name, blocks,
                          set_end_after_all_pulses=False,
                          disable_block_counter=False):
        """
        Creates a block with name :block_name: that consists of the serial
        execution of the given :blocks:.

        CAUTION: For each of the given blocks, the end time of the block is
        determined by the pulse listed last in the block, which is not
        necessarily the one that ends last in terms of timing. To instead
        determine the end time of the block based on the pulse that ends
        last, set set_end_after_all_pulses to True (or adjust the end pulse
        of each block before calling sequential_blocks).

        Args:
            block_name (string): name of the block that is created
            blocks (iterable): iterable where each element is a block that has
                to be executed one after another.
            set_end_after_all_pulses (bool, default False): in all
                blocks, correct the end pulse to happen after the last pulse.
            disable_block_counter (bool, default False): prevent block.build
                from appending a counter to the block name.
        """

        sequential = Block(block_name, [])
        for block in blocks:
            if set_end_after_all_pulses:
                block.set_end_after_all_pulses()
            sequential.extend(block.build(
                name=block.name if disable_block_counter else None))
        return sequential

    def sweep_n_dim(self, sweep_points, body_block=None, body_block_func=None,
                    cal_points=None, init_state='0', seq_name='Sequence',
                    ro_kwargs=None, return_segments=False, ro_qubits='all',
                    repeat_ro=True, init_kwargs=None, final_kwargs=None, **kw):
        """
        Creates a sequence or a list of segments by doing an N-dim sweep
        over the given operations based on the sweep_points.
        Currently, only 1D and 2D sweeps are implemented.

        :param sweep_points: SweepPoints object
            Note: If it contains sweep points with parameter names of the form
            "Segment.property", the respective property of the created
            Segment objects will be swept.
        :param body_block: block containing the pulses to be swept (excluding
            initialization and readout)
        :param body_block_func: a function that creates the body block at each
            sweep point. Takes as arguments (jth_1d_sweep_point,
            ith_2d_sweep_point, sweep_points, **kw)
        :param cal_points: CalibrationPoints object
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param seq_name: Name (str) of the sequence (default: "Sequence")
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :param return_segments: whether to return segments or the sequence
        :param ro_qubits: is passed as argument qb_names to self.initialize()
            and self.mux_ro() to specify that only subset of qubits should
            be prepared and read out (default: 'all')
        :param repeat_ro: (bool) set repeat pattern for readout pulses
            (default: True)
        :param init_kwargs: Keyword arguments (dict) for the initialization,
            see method initialize().
        :param final_kwargs: Keyword arguments (dict) for the finalization,
            see method finalize().
        :param kw: additional keyword arguments
            body_block_func_kw (dict, default: {}): keyword arguments for the
                body_block_func
            block_align_cal_pts (str, default: 'end'): aligment condition for
                the calpoints segments. Passed to seg_from_cal_points. See
                block_align in docstring for simultaneous_blocks.
        :return:
            - if return_segments==True:
                1D: list of segments, number of 1d sweep points or
                2D: list of list of segments, list of numbers of sweep points
            - else:
                1D: sequence, number acquisition elements
                2D: list of sequences, number of acquisition elements, number
                    of sequences
        """
        sweep_dims = len(sweep_points)
        if sweep_dims > 2:
            raise NotImplementedError('Only 1D and 2D sweeps are implemented.')

        if sum([x is None for x in [body_block, body_block_func]]) != 1:
            raise ValueError('Please specify either "body_block" or '
                             '"body_block_func."')

        if ro_kwargs is None:
            ro_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        if final_kwargs is None:
            final_kwargs = {}

        nr_sp_list = sweep_points.length()
        if sweep_dims == 1:
            sweep_points = copy(sweep_points)
            sweep_points.add_sweep_dimension()
            nr_sp_list.append(1)

        ro = self.mux_readout(**ro_kwargs, qb_names=ro_qubits)
        _, all_ro_qubits = self.get_qubits(ro_qubits)
        all_ro_op_codes = [p['op_code'] for p in ro.pulses]
        if body_block is not None:
            op_codes = [p['op_code'] for p in body_block.pulses if 'op_code'
                        in p]
            all_ro_qubits += [qb for qb in self.qb_names if f'RO {qb}' in
                              op_codes and qb not in all_ro_qubits]
            all_ro_op_codes += [f'RO {qb}' for qb in all_ro_qubits if qb not
                                in ro_qubits]
        sweep_dim_init = sweep_points.find_parameter('initialize')
        sweep_dim_final = sweep_points.find_parameter('finalize')
        if sweep_dim_init is None:
            prep = self.initialize(init_state=init_state,
                                   qb_names=all_ro_qubits, **init_kwargs)
        if sweep_dim_final is None:
            final = Block('Finalization', [])

        seqs = []
        for i in range(nr_sp_list[1]):
            this_seq_name = seq_name + (f'_{i}' if sweep_dims == 2 else '')
            seq = Sequence(this_seq_name)
            for j in range(nr_sp_list[0]):
                dims = j, i
                if sweep_dim_init is not None:
                    prep = self.initialize(
                        init_state=sweep_points.get_sweep_params_property(
                            'values', 'all', 'initialize')[dims[sweep_dim_init]],
                        qb_names=all_ro_qubits, **init_kwargs)
                if body_block is not None:
                    this_body_block =  body_block
                else:
                    this_body_block = body_block_func(
                        j, i, sweep_points=sweep_points,
                        **kw.get('body_block_func_kw', {}))
                if sweep_dim_final is not None:
                    final = self.finalize(
                        init_state=sweep_points.get_sweep_params_property(
                            'values', 'all', 'finalize')[dims[sweep_dim_final]],
                        qb_names=all_ro_qubits, **final_kwargs)

                # As we loop over all sweep points, some of the blocks will be
                # built multiple times (e.g., ro), but they will only be
                # built once per segment. It is thus not necessary to append
                # a counter that is increased each time the block is built.
                # Quite the contrary, it is much more intuitive to have the
                # same block names in each segment, while only the segment
                # name reflects the index of the sweep point. Thus, we call
                # sequential_blocks with disable_block_counter=True.
                segblock = self.sequential_blocks(
                    'segblock', [prep, this_body_block, final, ro],
                    disable_block_counter=True)
                seg = Segment(f'seg{j}', segblock.build(
                    sweep_dicts_list=sweep_points, sweep_index_list=[j, i]))
                # apply Segment sweep points
                for dim in [0, 1]:
                    for param in sweep_points[dim]:
                        if param.startswith('Segment.'):
                            vals = sweep_points.get_sweep_params_property(
                                'values', dim, param)
                            setattr(seg, param[len('Segment.'):],
                                    vals[j if dim == 0 else i])
                # add the new segment to the sequence
                seq.add(seg)
            if cal_points is not None:
                block_align_cal_pts = kw.get('block_align_cal_pts', 'end')
                seq.extend(self.seg_from_cal_points(
                    cal_points, init_state, ro_kwargs,
                    block_align=block_align_cal_pts, **kw))
            seqs.append(seq)

        if return_segments:
            segs = [list(seq.segments.values()) for seq in seqs]
            if sweep_dims == 1:
                return segs[0], nr_sp_list[0]
            else:
                return segs, nr_sp_list

        # repeat UHF seqZ code
        if repeat_ro:
            for s in seqs:
                for ro_op in all_ro_op_codes:
                    s.repeat_ro(ro_op, self.operation_dict)

        if sweep_dims == 1:
            return seqs, [np.arange(seqs[0].n_acq_elements())]
        else:
            return seqs, [np.arange(seqs[0].n_acq_elements()),
                          np.arange(nr_sp_list[1])]

    def tomography_pulses(self, tomo_qubits=None,
                          basis_rots=('I', 'X90', 'Y90'), all_rots=True):

        """
        Generates a complete list of tomography pulse lists for tomo_qubits.
        :param tomo_qubits: None, list of qubit names, or of qubits indices in
            self.get_qubits(). I None, then tomo_qubit = self.get_qubits()[1].
            If list of indices, they will be sorted.
            This parameter is only relevant if basis_rots is not a list of
            lists/tuples.
        :param basis_rots: list of strings or list of lists/tuples of strings,
            where the strings are pycqed pulse names.
        :param all_rots: bool specifying whether to take all possible
            combinations of basis_rots for tomo_qubits, or not.
            This parameter is only relevant if basis_rots is not a list of
            lists/tuples.
        :return:
            If list of lists/tuples, this function will do nothing and will
                just return basis_rots unmodified. Hence, the lists/tuples of
                strings must contain pulse names for each qubit in the
                experiment (i.e. self.get_qubits()).

            If list of strings, this function will return all possible
                combinations of basis_rots for tomo_qubits if all_rots, else it
                will return list with len(basis_rots) lists with
                len(tomo_qubits) repetitions of each pulse in basis_rots
                (i.e. all qubits get the same pulses).
        """

        if not isinstance(basis_rots[0], str):
            return basis_rots

        all_qubit_names = self.get_qubits()[1]
        if tomo_qubits is None:
            tomo_qubits = all_qubit_names
        if isinstance(tomo_qubits[0], str):
            tomo_qubits = [all_qubit_names.index(i) for i in tomo_qubits]
        # sort qubit indices to ensure that basis_rots are always applied on
        # qubits in ascending order as defined by self.get_qubits().
        tomo_qubits.sort()

        if all_rots:
            basis_rots = list(itertools.product(basis_rots,
                                                repeat=len(tomo_qubits)))
        else:
            basis_rots = [len(tomo_qubits) * [br] for br in basis_rots]

        basis_rots_all_qbs = len(basis_rots) * ['']
        for i, br in enumerate(basis_rots):
            temp = len(all_qubit_names)*['I']
            for ti in range(len(tomo_qubits)):
                temp[tomo_qubits[ti]] = br[ti]
            basis_rots_all_qbs[i] = temp

        return basis_rots_all_qbs