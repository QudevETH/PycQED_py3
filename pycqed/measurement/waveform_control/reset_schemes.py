"""
Reset Schemes for Quantum Control

This module provides implementations of various reset schemes for quantum control
operations in the PycQED framework. It defines several classes that inherit from
the base ResetScheme class, each implementing a specific reset strategy.

Classes:
    ResetScheme: Base class for reset schemes, providing common functionality.
    Preselection: A preselection-based reset scheme with opt flux compensation.
    FeedbackReset: A feedback-based active reset scheme.
    ParametricFluxReset: A scheme using parametric flux modulation operations.
    F0g1Reset: A scheme for resetting to the f0g1 state.

Each reset scheme class provides methods for constructing reset blocks,
handling operation dictionaries, and generating analysis instructions. These
schemes can be used to initialize quantum systems to desired states before
measurements or other quantum operations.

The module is designed to be flexible and extensible, allowing for easy
addition of new reset schemes as needed.

Typical notebook usage example:

    qb.add_reset_schemes()
    qb.reset.steps(['preselection', 'feedback_reset', 'parametric_flux_reset'])

By default the following reset schemes are added by add_reset_schemes():

- preselection: A preselection-based reset scheme with opt. flux compensation.
- feedback_reset: A feedback-based active reset scheme.

Once the 'steps' are defined, the reset steps are prepended to your pulses.

They can be temporariliy disabled by:

    qb.reset.enabled(False) # disable reset schemes - enabled by default
"""

import logging
log = logging.getLogger(__name__)

from copy import deepcopy
# FIXME: check at which points to copy_op of CircuitBuilder should be used
#  instead of deepcopy to enable fast mode.

from pycqed.instrument_drivers.instrument import InstrumentModule
import pycqed.measurement.waveform_control.block as block_mod

from qcodes import ManualParameter
from qcodes.utils import validators


class ResetScheme(InstrumentModule):
    """
    Provides basic control and execution of the Active Reset scheme.

    This class enables flexible configuration of Active Reset schemes. It allows
    users to define timings, priorities, and buffers for the reset process.
    Parameters can be inherited from parent operations or customized directly.
    """

    # By default the reset scheme uses default values from the parent
    # operation declared in the qubit. This is realized via deepcopy
    # and marking parameters with a DEFAULT_VALUE.
    DEFAULT_VALUE = "from parent"

    def __init__(self, parent, name, operations=(),
                 ref_instrument=None, sweep_params=None, **kwargs):
        """Initializes a ResetScheme instance.

        Args:
          parent: The parent instrument or operation.
          name: The name of the ResetScheme instance.
          operations: A tuple of operations to be included in the reset scheme.
          ref_instrument: An optional reference instrument (if different from parent).
          sweep_params: Parameters for sweeping.
          **kwargs: Additional keyword arguments.
        """
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

        # Create a repetition counter so that _init_block() knows about
        # the repetition at run time
        self._rep = 0


    def add_operation(self, operation_name, init_values=None):
        """
        Adds an operation to the ResetScheme.

        Args:
            operation_name: The name of the operation to add.
            init_values: Optional dictionary of initial values for operation parameters.

        Raises:
            ValueError: If the operation is unknown to the reference instrument.
        """
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
        """Returns a list of available operations.

        Returns:
            A list of operations.
        """
        return self._operations

    def reset_block(self, name=None, sweep_params=None, **block_kwargs):
        """
        Constructs the reset block with repetitions and buffer intervals.

        This function assembles the reset block, incorporating repetitions and associated 
        buffer intervals. For each repetition, the `_reset_block` method is called to 
        generate the core reset instructions. 

        Args:
            name: Optional name for the reset block. If not provided, the short name 
                  of the `ResetScheme` instance is used.
            sweep_params: Optional parameters for sweeping.
            **block_kwargs: Additional keyword arguments to be passed to the 
                            `Block` constructor.

        Returns:
            block_mod.Block: The constructed reset block object.
        """
        if name is None:
            name = self.short_name

        init = block_mod.Block(block_name=name, pulse_list=[])
        self._rep = 0 # Reset internal repetition counter

        # Build blocks with buffers for repetitions
        for i in range(self.repetitions()):
            init_i = self._reset_block(name + f'_{i}', sweep_params,
                                       **block_kwargs).build(
                block_delay=self.repetition_buffer_start(),
                block_end=dict(pulse_delay=self.repetition_buffer_end()))
            init.extend(init_i)
            self._rep += 1

        # Add buffers for the total init block
        block_start = {"pulse_delay": self.buffer_start()}
        block_end = {"pulse_delay": self.buffer_end()}
        block_end.update(self.block_end())
        block_start.update(self.block_start())
        init.block_start.update(block_start)
        init.block_end.update(block_end)
        return init

# FIXME: a private method with the same name as the one above?
# this happens a few more times...
# FIXME: sweep_params is not accessed, why is it here?
    def _reset_block(self, name, sweep_params, **kwargs):
        return block_mod.Block(name, [], **kwargs)

# FIXME: this is a duplicate of the one in qubit_object; find where
# we can put in hierarchy such that we can access it without dupplication
    def get_operation_dict(self, operation_dict=None):
        """
        Generates an operation dictionary for the ResetScheme.

        This function creates a dictionary of operations, combining parameters from the reference 
        instrument with initialization-specific parameters defined in the ResetScheme. This
        dictionary is used to construct the instructions for the reset process.

        Args:
            operation_dict: An optional existing dictionary to update. If None, a new dictionary is created.

        Returns:
            dict: A dictionary containing operations with their associated parameters.
        """
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
        """
        Constructs an operation dictionary for the available operations of the ResetScheme.

        This function generates a dictionary mapping operation codes to their corresponding
        arguments and parameter values. The parameter values are retrieved from the ResetScheme's 
        settings.

        Args:
            operation_dict: An optional existing operation dictionary to update. If None,
                            a new dictionary is created.

        Returns:
            dict: A dictionary containing operation codes as keys and dictionaries of 
                argument-parameter mappings as values.
        """
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
        """
        Retrieves initialization-specific parameters for a set of operations.

        This function extracts parameters that have values deviating from the default settings 
        ('from parent') for a specified set of operations within the ResetScheme. 

        Args:
            operations:  A list or iterable of operation names. If None, all available 
                        operations within the ResetScheme are used.

        Returns:
            dict: A deep copy of a dictionary containing initialization-specific 
                parameters, organized by operation name.
        """
        if operations is None:
            operations = self.operations().keys()
        params = {}
        for op in operations:
            params[op] = {k: v for k, v in
                          self._get_operation_dict()[self.get_opcode(op)].items()
                          if v != self.DEFAULT_VALUE}
        return deepcopy(params)

    def get_opcode(self, operation, instr=None):
        """
        Constructs a unique operation code (opcode).

        This function creates an operation code that combines the operation name, the 
        ResetScheme's short name, and the reference instrument's name.

        Args:
        operation: The name of the operation.
        instr: The reference instrument (optional). If None, defaults to `self.instr_ref`.

        Returns:
            str: The formatted operation code.
        """
        if instr is None:
            instr = self.instr_ref
        return operation + f"_{self.short_name} {instr.name}"

    def get_analysis_instructions(self):
        """
        Provides instructions for analyzing reset data (currently a placeholder).

        This function returns basic instructions for data analysis. In the future, it will 
        likely be expanded to accommodate more complex reset schemes.

        Returns:
            dict: A dictionary containing analysis instructions, currently specifying
                the preparation type as 'wait'.
        """
        return dict(preparation_type='wait')


class Preselection(ResetScheme):
    """
    Implements a preselection-based ResetScheme with optional flux compensation.

    This class provides a ResetScheme tailored for preselection, enabling the configuration 
    of parameters as well as conditional modification of the readout (RO) pulse to 
    include flux compensation.
    """
    DEFAULT_INSTANCE_NAME = "preselection"

    def __init__(self, parent, **kwargs):
        """
        Initializes a Preselection instance.

        Args:
            parent: The parent instrument or operation.
            **kwargs: Additional keyword arguments passed to the parent 'ResetScheme'.
        """

        super().__init__(parent, name="preselection", operations=('RO', "FP"),
                         **kwargs)

        self.add_parameter('compensate_ro_flux', label='compensate_ro_flux',
                           initial_value=False,
                           vals=validators.Bool(),
                           parameter_class=ManualParameter)

    def _reset_block(self, name, sweep_params, **kwargs):
        """
        Constructs a reset block with a preselection readout (RO) pulse.

        Optionally includes a flux compensation pulse (FP) if `compensate_ro_flux` is True. 
        Handles potential parameter conflicts and provides informative comments.

        Args:
            name: Name for the reset block.
            sweep_params: Optional parameters for sweeping.
            **kwargs: Additional keyword arguments for constructing the block.

        Returns:
            block_mod.Block: The constructed reset block object.
        """
        op_dict = self.instr_ref.get_operation_dict()
        # FIXME: here, implicitly assumes structure about the operations name which
        #  ideally we would have only where the operations_dict is constructed
        preselection_ro = deepcopy(op_dict[f'RO {self.instr_ref.name}'])

        # update pulse parameters with initialization-specific parameters
        preselection_ro.update(self.get_init_specific_params()['RO'])

        for k, v in sweep_params.items():
            if k in preselection_ro:
                preselection_ro[k] = block_mod.ParametricValue(v)

        presel_pulses = [preselection_ro]

        # modify length and amplitude to match the one of ro pulse in
        # preselection_ro if ro_pulse_type == with flux  AND compensate_flux = True
        # FIXME: not great to have this check based on hardcoded pulse name
        #  in the future maybe it should check whether the readout operation
        #  contains a flux pulse, of which type, etc. Also the logic about the
        #  flux pulse buffers and length is dupplicated from the pulse_library:
        #  NOT GREAT.
        # FIXME: the following is not compatible with ParametricValue in parameters
        #  of the preselection_ro
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
        """
        Provides instructions for analyzing preselection reset data.

        Returns a dictionary containing the preparation type ('preselection'). 
        Includes a note that enhancements for more complex reset types are possible.
        """
        # instructions such that the analysis knows how to process the data
        # likely to change when analysis is refactored / enhanced to be able
        # to handle more complex reset types (e.g. combinations etc).
        # for now, the legacy naming conventions are used in the analysis
        return dict(preparation_type='preselection')


class FeedbackReset(ResetScheme):
    """
    Provides feedback-based Active Reset scheme implementation.

    This class inherits from `ResetScheme` and extends it to enable Active Reset 
    controlled by feedback from readout measurements. It supports configuration of 
    codeword-state mappings and feedback delays.
    """

    # Calculate the length of a ge pulse, assumed the same for all qubits
    state_ops = dict(g=["I"], e=["X180"], f=["X180_ef", "X180"])
    DEFAULT_INSTANCE_NAME = "feedback"

    def __init__(self, parent, **kwargs):
        """
        Initializes a FeedbackReset instance.

        Args:
            parent: The parent instrument or operation.
            **kwargs: Additional keyword arguments passed to the parent 'ResetScheme'.
        """
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
                           initial_value=4e-6, vals=validators.Numbers(),
                           parameter_class=ManualParameter,
                           get_parser=self._validate_ro_feedback_delay)

    def get_operation_dict(self, operation_dict=None):
        """
        Generates an operation dictionary, including an "I" (identity) operation.
 
        Inherits the operation dictionary from the parent `ResetScheme` class and 
        adds an "I" operation (with zero amplitude) derived from the "X180" operation.
 
        Args:
            operation_dict: Optional existing dictionary to update.
 
        Returns:
            dict: The updated operation dictionary.
        """
        operation_dict = super().get_operation_dict()

        operation_dict[self.get_opcode("I")] = \
            deepcopy(operation_dict[self.get_opcode("X180")])
        operation_dict[self.get_opcode("I")]['amplitude'] = 0
        return operation_dict

    def _reset_block(self, name, sweep_params, **kwargs):
        """
        Generates a block containing a readout (RO) pulse followed by feedback pulses 
        determined by the `codeword_state_map`.

        Args:
            name: Name for the reset block.
            sweep_params: Optional parameters for sweeping.
            **kwargs: Additional keyword arguments for constructing the block.

        Returns:
            block_mod.Block: The constructed reset block object.
        """
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
        """
        Ensures that the `ro_feedback_delay` is sufficient to accommodate acquisition
        length and feedback latency.

        Args:
            ro_feedback_delay: The delay value to validate.

        Raises:
            ValueError: If the delay is too short.
        """
        # FIXME: Assume reference instrument has acq_length,
        # and acquisition instrument ? Further, not every instrument has feedback_latency()
        # For now this ensures backwards compatibility in pycqed and qcodes
        try:
            minimum_delay = self.instr_ref.acq_length() + \
                            self.instr_ref.instr_acq.get_instr().feedback_latency()

            if ro_feedback_delay < minimum_delay:
                msg = f"ro_feedback_delay ({ro_feedback_delay}s) is shorter " \
                    f"than minimum expected delay computed from acq_length and" \
                    f" acq_instr.feedback_latency ({minimum_delay}s)"
                raise ValueError(msg)

        except AttributeError as e:
            msg = f"Error calculating minimum delay: {e}. Check instrument configuration."
            log.warning(msg)  # inform user that we have no clue about the delay

        return ro_feedback_delay

    def _validate_codeword_state_map(self, codeword_state_map):
        """
        Ensures that the `codeword_state_map` contains an even number of codewords.

        Args:
            codeword_state_map: The map to validate.

        Raises:
            ValueError: If the map contains an odd number of codewords.
        """
        if len(codeword_state_map) % 2 != 0:
            msg = f"codeword_state_map must have even number of codewords" \
                  f" but {len(codeword_state_map)} codewords were provided."
            raise ValueError(msg)

        return codeword_state_map

    def get_analysis_instructions(self):
        """
        Provides instructions for analyzing feedback reset data.
 
        Returns a dictionary containing analysis instructions, including preparation type,
        feedback delay, and the number of repetitions.
        """

        # instructions such that the analysis knows how to process the data
        # likely to change when analysis is refactored / enhanced to be able
        # to handle more complex reset types (e.g. combinations etc).
        # for now, the legacy naming conventions are used in the analysis
        return dict(preparation_type='active_reset',
                    post_ro_wait=self.ro_feedback_delay(),
                    reset_reps=self.repetitions()
                    )

class ParametricFluxReset(ResetScheme):
    """
    Implements a ResetScheme using parametric flux modulation (PFM) operations.

    This class dynamically determines available PFM operations from the parent instrument 
    and configures a ResetScheme to utilize them.  
    """
    DEFAULT_INSTANCE_NAME = "parametric_flux"

    def __init__(self, parent, operations=None, **kwargs):
        """
        Initializes a ParametricFluxReset instance.

        Args:
            parent: The parent instrument or operation.
            operations: Optional list of specific PFM operations to use. If None,
                        automatically extracts PFM operations from the parent instrument.
            **kwargs: Additional keyword arguments passed to the parent 'ResetScheme'.

        Raises:
            ValueError: If no PFM operations are found in the parent instrument.
        """
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

    def _reset_block(self, name, sweep_params, **kwargs):
        """
        Creates a block containing copies of all configured PFM operations 
        retrieved from  the operation dictionary.

        Args:
            name: Name for the reset block.
            sweep_params: Optional parameters for sweeping.
            **kwargs: Additional keyword arguments for constructing the block.

        Returns:
            block_mod.Block: The constructed reset block object.
        """
        op_dict = self.get_operation_dict()

        reset_pulses = [deepcopy(op_dict[self.get_opcode(op)])
                        for op in self.operations()]

        return block_mod.Block(name, reset_pulses, copy_pulses=False, **kwargs)
