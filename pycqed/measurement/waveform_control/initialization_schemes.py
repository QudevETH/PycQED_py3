from copy import deepcopy

from pycqed.instrument_drivers.instrument import InstrumentModule
from pycqed.measurement.waveform_control.block import Block
from qcodes import ManualParameter
from qcodes.utils import validators


class InitializationScheme(InstrumentModule):
    INIT_VALUE = "from parent"

    def __init__(self, parent, name, operations=(),
                 ref_instrument=None, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.add_parameter('repetitions', label='repetitions',
                           unit=None, initial_value=1,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('priority', label='priority',
                           unit=None, initial_value=0,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('buffer_start', label='buffer_start',
                           unit="s", initial_value=0,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)

        self.add_parameter('buffer_end', label='buffer_end',
                           unit="s", initial_value=0,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)

        # FIXME: better name?
        self.add_parameter('repetition_buffer_start', label='repetition_buffer_start',
                           unit="s", initial_value=0,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)
        self.add_parameter('repetition_buffer_end', label='repetition_buffer_end',
                           unit="s", initial_value=0,
                           vals=validators.Numbers(),
                           parameter_class=ManualParameter)


        self.add_parameter('block_start', label='block_start',
                           initial_value={},
                           vals=validators.Dict(),
                           parameter_class=ManualParameter)

        self.add_parameter('block_end', label='block_end',
                            initial_value={},
                           vals=validators.Dict(),
                           parameter_class=ManualParameter)

        self.instr_ref = self.root_instrument if ref_instrument is None else ref_instrument

        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available '
                                     'for this initialization scheme',
                           get_cmd=self._get_operations)

        for operation in operations:
            self.add_operation(operation)


    def add_operation(self, operation_name, init_values=None):
        if operation_name not in self.instr_ref.operations():
            raise ValueError(f'Operation name {operation_name} unknown to '
                             f'reference instrument {self.instr_ref.name}.'
                             'Only operations known to the reference '
                             'instrument can be '
                             'used as part of an initialization scheme.')

        if init_values is None:
            init_values = {}

        # add operation
        self._operations[operation_name] = \
            deepcopy(self.instr_ref._operations[operation_name])

        # add pulse parameters based on reference instrument.
        for pulse_param_name, param_name in self._operations[operation_name].items():
            if self.instr_ref.parameters[param_name].vals is not None:
                vals = validators.MultiType(self.instr_ref.parameters[param_name].vals,
                                            validators.Enum(self.INIT_VALUE))
            else:
                vals = None
            self.add_parameter(param_name,
                initial_value=init_values.get(param_name, self.INIT_VALUE),
                vals=vals,
                parameter_class=ManualParameter)

    def _get_operations(self):
        return self._operations

    def init_block(self, name=None):
        if name is None:
            name = self.short_name
        init = Block(block_name=name, pulse_list=[])
        for i in range(self.repetitions()):
            # build block with buffers for repetition i
            init_i = self._init_block(name + f'_{i}').build(
                block_delay=self.repetition_buffer_start(),
                block_end=dict(pulse_delay=self.repetition_buffer_end()))
            init.extend(init_i)

        # add buffers for the total init block
        be = {"pulse_delay": self.buffer_end()}
        be.update(self.block_end())
        init.block_start.update(self.block_start())
        init.block_end.update(be)
        return init

    def _init_block(self, block_name):
        return Block(block_name, [])

    # FIXME: this is a duplicate of the one in qubit_object; find where
    #   we can put in hierarchy such that we can access it without dupplication
    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        init_specific_params = self.get_init_specific_params()
        ref_instr_op_dict = self.instr_ref.get_operation_dict()

        for op in init_specific_params:
            operation_dict[self.get_opcode(op)] = \
                deepcopy(ref_instr_op_dict[self.get_opcode(op, self.instr_ref)])
            operation_dict[self.get_opcode(op)].update(init_specific_params[op])
            operation_dict[self.get_opcode(op)].update(
                {"op_code": self.get_opcode(op)})
        return operation_dict

    def _get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        for op_name, op in self.operations().items():
            op_code = self.get_opcode(op_name)
            operation_dict[op_code] = {}
            for argument_name, parameter_name in op.items():
                operation_dict[op_code][argument_name] = \
                    self.get(parameter_name)
        return operation_dict

    def get_init_specific_params(self, operations=None):
        if operations is None:
            operations = self.operations().keys()
        params = {}
        for op in operations:
            params[op] = {k: v for k, v in
                          self._get_operation_dict()[op + f" " + self.name].items()
                          if v != self.INIT_VALUE}
        return params

    def get_opcode(self, operation, instr=None):
        if instr is None:
            instr = self
        return operation + f" {instr.name}"


class Preselection(InitializationScheme):

    def __init__(self, parent, **kwargs):

        super().__init__(parent, name="preselection", operations=('RO',), **kwargs)


    def _init_block(self, name):
        op_dict = self.instr_ref.get_operation_dict()
        # FIXME: here, implicitly assumes structure about the operations name which
        #  ideally we would have only where the operations_dict is constructed
        preselection_ro = op_dict[f'RO {self.instr_ref.name}']

        # update pulse parameters with initialization-specific parameters
        preselection_ro.update(self.get_init_specific_params()['RO'])

        # additional changes
        preselection_ro['element_name'] = f'{name}_element'
        return Block(name, [preselection_ro])


class ActiveReset(InitializationScheme):
    # Calculate the length of a ge pulse, assumed the same for all qubits
    state_ops = dict(g=["I"], e=["X180"], f=["X180_ef", "X180"])

    def __init__(self, parent, **kwargs):

        super().__init__(parent, name="active_reset",
                         operations=('RO', 'X180', 'X180_ef'), **kwargs)

        self.add_parameter('codeword_state_map',
                           docstring='dictionary where keys are the codewords'
                                     ' and values are the corresponding '
                                     ' state of the qubit.',
                           initial_value={}, vals=validators.Dict(),
                           parameter_class=ManualParameter,
                           set_parser=self._validate_codeword_state_map)
        self.add_parameter('ro_feedback_delay',
                           docstring='dictionary where keys are the codewords'
                                     ' and values are the corresponding '
                                     ' state of the qubit.',
                           initial_value=0, vals=validators.Numbers(),
                           parameter_class=ManualParameter,
                           get_parser=self._validate_ro_feedback_delay)

    def _init_block(self, name):
        op_dict = self.get_operation_dict()
        # FIXME: here, implicitly assumes structure about the operations name which
        #  ideally we would have only where the operations_dict is constructed
        active_reset_ro = deepcopy(op_dict[self.get_opcode("RO")])

        # additional changes
        active_reset_ro['element_name'] = f'element_{name}'
        active_reset_ro['name'] = f'ro_{name}'
        reset_pulses = [active_reset_ro]
        for j, (codeword, state) in enumerate(self.codeword_state_map().items()):
            for i, opname in enumerate(self.state_ops[state]):
                reset_pulses.append(deepcopy(op_dict[self.get_opcode(opname)]))
                reset_pulses[-1]['phaselock'] = False
                reset_pulses[-1]['codeword'] = codeword
                reset_pulses[-1]['element_name'] = f'reset_pulses_element_{name}'
                if i == 0:
                    reset_pulses[-1]['ref_pulse'] = active_reset_ro['name']
                    print(active_reset_ro['name'])
                    reset_pulses[-1]['pulse_delay'] = self.ro_feedback_delay()
        return Block(name, reset_pulses)

    def _validate_ro_feedback_delay(self, ro_feedback_delay):
        # FIXME: assume reference instrument has acq_length,
        #  and acquisition instrument ?
        minimum_delay = self.instr_ref.acq_length() + \
                        self.instr_ref.instr_acq.get_instr().feedback_latency()
        if ro_feedback_delay < minimum_delay:
            msg = f"ro_feedback_delay ({ro_feedback_delay}s) is shorter " \
                  f"than minimum expected delay computed from acq_length and" \
                  f" acq_instr.feedback_latency ({minimum_delay}s)"
            raise ValueError(msg)

        return ro_feedback_delay

    def _validate_codeword_state_map(self, codeword_state_map):
        if len(codeword_state_map) % 2 != 0:
            msg = f"codeword_state_map must have even number of codewords" \
                  f" but {len(codeword_state_map)} codewords were provided."
            raise ValueError(msg)

        return codeword_state_map

class ParametricFluxReset(InitializationScheme):

    def __init__(self, parent, operations=None, **kwargs):
        if operations is None:
            all_ops = list(parent.root_instrument.operations())
            # find all operations which include PFR (such that, if defined,
            # PRF and PFR_ef are both taken)
            operations = [op for op in all_ops if op.startswith("PFR")]
            if len(operations) == 0:
                raise ValueError("No operation found starting with 'PFR'"
                                 f" in the root instrument "
                                 f"{parent.root_instrument}.")

        super().__init__(parent, name="parametric_flux_reset",
                         operations=operations, **kwargs)

    def _init_block(self, name):
        op_dict = self.get_operation_dict()

        reset_pulses = [deepcopy(op_dict[self.get_opcode(op)])
                        for op in self.operations()]

        return Block(name, reset_pulses)