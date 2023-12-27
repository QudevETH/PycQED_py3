from copy import deepcopy
# FIXME: check at which points to copy_op of CircuitBuilder should be used
#  instead of deepcopy to enable fast mode.

from pycqed.instrument_drivers.instrument import InstrumentModule
import pycqed.measurement.waveform_control.block as block_mod

from qcodes import ManualParameter
from qcodes.utils import validators


class ResetScheme(InstrumentModule):
    DEFAULT_VALUE = "from parent"

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

        self.instr_ref = self.root_instrument if ref_instrument is None \
            else ref_instrument

        self._operations = {}
        self.add_parameter('operations',
                           docstring='a list of all operations available '
                                     'for this initialization scheme',
                           get_cmd=self._get_operations)

        for operation in operations:
            self.add_operation(operation)

        # create a repetition counter so that _init_block() knows about
        # the repetition at run time
        self._rep = 0


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
                                            validators.Enum(self.DEFAULT_VALUE))
            else:
                vals = None
            self.add_parameter(param_name,
                               initial_value=init_values.get(param_name,
                                                             self.DEFAULT_VALUE),
                               vals=vals,
                               parameter_class=ManualParameter)

    def _get_operations(self):
        return self._operations

    def reset_block(self, name=None, **block_kwargs):
        if name is None:
            name = self.short_name
        init = block_mod.Block(block_name=name, pulse_list=[])
        self._rep = 0 # set repetition counter to 0
        for i in range(self.repetitions()):
            # build block with buffers for repetition i
            init_i = self._reset_block(name + f'_{i}', **block_kwargs).build(
                block_delay=self.repetition_buffer_start(),
                block_end=dict(pulse_delay=self.repetition_buffer_end()))
            init.extend(init_i)
            self._rep += 1

        # add buffers for the total init block
        bs = {"pulse_delay": self.buffer_start()}
        be = {"pulse_delay": self.buffer_end()}
        be.update(self.block_end())
        bs.update(self.block_start())
        init.block_start.update(bs)
        init.block_end.update(be)
        return init

    def _reset_block(self, name, **kwargs):
        return block_mod.Block(name, [], **kwargs)

    # FIXME: this is a duplicate of the one in qubit_object; find where
    #   we can put in hierarchy such that we can access it without dupplication
    def get_operation_dict(self, operation_dict=None):
        if operation_dict is None:
            operation_dict = {}
        init_specific_params = self.get_init_specific_params()
        ref_instr_op_dict = self.instr_ref.get_operation_dict()

        for op in init_specific_params:
            operation_dict[self.get_opcode(op)] = \
                deepcopy(ref_instr_op_dict[op + f" {self.instr_ref.name}"])
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
                          self._get_operation_dict()[self.get_opcode(op)].items()
                          if v != self.DEFAULT_VALUE}
        return deepcopy(params)

    def get_opcode(self, operation, instr=None):
        if instr is None:
            instr = self.instr_ref
        return operation + f"_{self.short_name} {instr.name}"

    def get_analysis_instructions(self):
        # instructions such that the analysis knows how to process the data
        # likely to change when analysis is refactored /enhanced to be able
        # to handle more complex reset types (e.g. combinations etc)
        return dict(preparation_type='wait')


class Preselection(ResetScheme):
    DEFAULT_INSTANCE_NAME = "preselection"

    def __init__(self, parent, **kwargs):

        super().__init__(parent, name="preselection", operations=('RO', "FP"),
                         **kwargs)

        self.add_parameter('compensate_ro_flux', label='compensate_ro_flux',
                           initial_value=False,
                           vals=validators.Bool(),
                           parameter_class=ManualParameter)


    def _reset_block(self, name, **kwargs):
        op_dict = self.instr_ref.get_operation_dict()
        # FIXME: here, implicitly assumes structure about the operations name which
        #  ideally we would have only where the operations_dict is constructed
        preselection_ro = deepcopy(op_dict[f'RO {self.instr_ref.name}'])

        # update pulse parameters with initialization-specific parameters
        preselection_ro.update(self.get_init_specific_params()['RO'])
        presel_pulses = [preselection_ro]

        # modify length and amplitude to match the one of ro pulse in
        # preselection_ro if ro_pulse_type == with flux  AND compensate_flux = True
        # FIXME: not great to have this check based on hardcoded pulse name
        #  in the future maybe it should check whether the readout operation
        #  contains a flux pulse, of which type, etc. Also the logic about the
        #  flux pulse buffers and length is dupplicated from the pulse_library:
        #  NOT GREAT.
        if preselection_ro['pulse_type'] == "GaussFilteredCosIQPulseWithFlux" and \
            self.compensate_ro_flux():
            compensation_fp = deepcopy(op_dict[f'FP {self.instr_ref.name}'])
            compensation_fp['pulse_type'] = "BufferedSquarePulse"
            compensation_fp['gaussian_filter_sigma'] = \
                preselection_ro['flux_gaussian_filter_sigma']
            compensation_fp['amplitude'] = -preselection_ro['flux_amplitude']
            compensation_fp['pulse_length'] = \
                preselection_ro['pulse_length'] \
                + preselection_ro['flux_extend_start'] + \
                preselection_ro['flux_extend_end']
            compensation_fp['buffer_length_start'] =\
                preselection_ro['buffer_length_start'] \
                - preselection_ro['flux_extend_start']
            compensation_fp['buffer_length_end'] = \
                preselection_ro['buffer_length_start'] \
                + preselection_ro['pulse_length'] \
                + preselection_ro['buffer_length_end'] \
                - compensation_fp['buffer_length_start'] \
                - compensation_fp['pulse_length']
            compensation_fp.update(self.get_init_specific_params()['FP'])
            presel_pulses = [compensation_fp] + presel_pulses

        # additional changes
        return block_mod.Block(name, presel_pulses,
                               copy_pulses=False,
                               **kwargs)

    def get_analysis_instructions(self):
        # instructions such that the analysis knows how to process the data
        # likely to change when analysis is refactored / enhanced to be able
        # to handle more complex reset types (e.g. combinations etc).
        # for now, the legacy naming conventions are used in the analysis
        return dict(preparation_type='preselection')


class FeedbackReset(ResetScheme):
    # Calculate the length of a ge pulse, assumed the same for all qubits
    state_ops = dict(g=["I"], e=["X180"], f=["X180_ef", "X180"])
    DEFAULT_INSTANCE_NAME = "feedback"

    def __init__(self, parent, **kwargs):

        super().__init__(parent, name=self.DEFAULT_INSTANCE_NAME,
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

    def get_operation_dict(self, operation_dict=None):
        operation_dict = super().get_operation_dict()

        operation_dict[self.get_opcode("I")] = \
            deepcopy(operation_dict[self.get_opcode("X180")])
        operation_dict[self.get_opcode("I")]['amplitude'] = 0
        return operation_dict

    def _reset_block(self, name, **kwargs):
        op_dict = self.get_operation_dict()
        # FIXME: here, implicitly assumes structure about the operations name which
        #  ideally we would have only where the operations_dict is constructed
        active_reset_ro = deepcopy(op_dict[self.get_opcode("RO")])

        # additional changes
        active_reset_ro['name'] = f'ro_{name}'
        reset_pulses = [active_reset_ro]
        for j, (codeword, state) in enumerate(self.codeword_state_map().items()):
            for i, opname in enumerate(self.state_ops[state]):
                reset_pulses.append(deepcopy(op_dict[self.get_opcode(opname)]))
                # Reset pulses cannot include phase information at the moment
                # since we use the exact same waveform(s) (corresponding to
                # a given codeword) for every reset pulse(s) we play (no
                # matter where in the circuit). Therefore, remove phase_lock
                # that references the phase to algorithm time t=0.
                reset_pulses[-1]['phaselock'] = False
                reset_pulses[-1]['codeword'] = codeword
                # all feedback pulses for a given repetition are in the same element
                reset_pulses[-1]['element_name'] = f'reset_pulses_element_{self._rep}'
                if i == 0:
                    reset_pulses[-1]['ref_pulse'] = active_reset_ro['name']
                    reset_pulses[-1]['pulse_delay'] = self.ro_feedback_delay()
                    reset_pulses[-1]['ref_point'] = "start"
        return block_mod.Block(name, reset_pulses, copy_pulses=False, **kwargs)

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

    def get_analysis_instructions(self):
        # instructions such that the analysis knows how to process the data
        # likely to change when analysis is refactored / enhanced to be able
        # to handle more complex reset types (e.g. combinations etc).
        # for now, the legacy naming conventions are used in the analysis
        return dict(preparation_type='active_reset',
                    post_ro_wait=self.ro_feedback_delay(),
                    reset_reps=self.repetitions()
                    )

class ParametricFluxReset(ResetScheme):
    DEFAULT_INSTANCE_NAME = "parametric_flux"

    def __init__(self, parent, operations=None, **kwargs):
        if operations is None:
            all_ops = list(parent.root_instrument.operations())
            # find all operations which include PFM (such that, if defined,
            # PFM and PFM_ef are both taken)
            operations = [op for op in all_ops if op.startswith("PFM")]
            if len(operations) == 0:
                raise ValueError("No operation found starting with 'PFM'"
                                 f" in the root instrument "
                                 f"{parent.root_instrument}.")

        super().__init__(parent, name=self.DEFAULT_INSTANCE_NAME,
                         operations=operations, **kwargs)

    def _reset_block(self, name, **kwargs):
        op_dict = self.get_operation_dict()

        reset_pulses = [deepcopy(op_dict[self.get_opcode(op)])
                        for op in self.operations()]

        return block_mod.Block(name, reset_pulses, copy_pulses=False, **kwargs)
