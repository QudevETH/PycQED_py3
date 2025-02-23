import itertools
import logging
from copy import copy, deepcopy

import numpy as np

from pycqed.measurement import multi_qubit_module as mqm

log = logging.getLogger(__name__)
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control.segment import Segment
from pycqed.measurement.waveform_control.sequence import Sequence


class CircuitBuilder:
    """A class that helps to build blocks, segments, or sequences, e.g.,
    when implementing quantum algorithms.

    :param dev: the device on which the algorithm will be executed
    :param qubits: a list of qubit objects or names if the builder should
        act only on a subset of qubits (default: all qubits of the device)
    :param kw: keyword arguments
         cz_pulse_name: (str) the prefix of CZ gates (default: None,
             in which case we use the default_cz_gate_name of the device
             object if available, or otherwise the first operation from the
             operation_dict whose name contains 'CZ', or we fall back to
             'CZ' if no such operation is found.)
         decompose_rotation_gates: (dict of bool or dict of lists of lists)
            whether arbitrary rotation gates should be decomposed into pi
            rotations and virtual Z gates. Can either be a boolean to
            decompose all gates of a given type, e.g. {'CZ_nztc': True},
            or a list of all qubit pairs for which the gate should be
            decomposed, e.g {'CZ_nztc': [['qb2', 'qb1'], ['qb3', 'qb4]]}. In
            the latter case the single-qubit gates of the decomposition are
            applied to the first qubit of the pair.
         reset_params: (dict) Global reset parameters, see doc string of
            CircuitBuilder.reset()
            If None, they are taken from the individual qubit.reset modules.
         fast_mode: (bool, default: False) activate faster processing by
            - creating segments with fast_mode=True in self.sweep_n_dim
                (see fast_mode parameter in Segment class)
            - deactivating the following features:
                - addressing qubits via logical indices (spin indices)
                - resolution of ParametricValues in self.sweep_n_dim if
                  body_block_func is used
            - copying pulse dicts with copy instead of with deepcopy. This
                means that the user has to ensure that mutable pulse parameters
                (dicts,lists, etc.) do not get modified by their code.
    """

    STD_INIT = {'0': ['I'], '1': ['X180'], '+': ['Y90'], '-': ['mY90'],
                'g': ['I'], 'e': ['X180'], 'f': ['X180', 'X180_ef'],
                'h': ['X180', 'X180_ef', 'X180_fh']}

    def __init__(self, dev=None, qubits=None, operation_dict=None,
                 filter_qb_names=None, **kw):

        self.dev = dev
        self.qubits, self.qb_names = self.extract_qubits(
            dev, qubits, operation_dict, filter_qb_names)
        self._reset_sweep_params = {qb: {} for qb in self.qb_names}
        self.update_operation_dict(operation_dict)
        self.cz_pulse_name = kw.get('cz_pulse_name')
        if self.cz_pulse_name is None:
            if self.dev is not None and self.dev.default_cz_gate_name() is \
                    not None:
                self.cz_pulse_name = self.dev.default_cz_gate_name()
            else:  # try to find a CZ gate in the opreation dict
                op_types = [o.split(' ')[0] for o in self.operation_dict]
                cz_gates = [o for o in op_types if 'CZ' in o] + ['CZ']
                self.cz_pulse_name = cz_gates[0]
        self.decompose_rotation_gates = kw.get('decompose_rotation_gates', {})
        self.fast_mode = kw.get('fast_mode', False)
        self.copy_op = copy if self.fast_mode else deepcopy
        self.reset_params = self._parse_reset(kw.get('reset_params', None))

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

    @staticmethod
    def _parse_reset(reset_params=None):
        """Parse reset parameters such that the output is either "None"
        or a dictionary containing reset parameters.

        Args:
            reset_params (None, str, dict): reset parameters.
            If None, returned reset params will be None.
            If False, returned reset params is a dict with no reset steps
            If string, returned reset params is a dict where the reset step is
            the string.
            If a dict, then reset_params is returned untouched.
            Otherwise, an exception is raised.

        Returns:
            reset_parameters (dict, None): FIXME
        """
        if isinstance(reset_params, str):
            return dict(steps=[reset_params])
        elif reset_params == False:
            return dict(steps=[])
        elif reset_params is None:
            return reset_params
        elif not isinstance(reset_params, dict):
            raise ValueError(f'reset_params must be a string, False,'
                             f' or a dict, but not {reset_params} of type'
                             f' {type(reset_params)}')
        else:
            return reset_params

    def update_operation_dict(self, operation_dict=None):
        """Updates the stored operation_dict based on the passed operation_dict or
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

    def get_qubits(self, qb_names=None, strict=True):
        """Wrapper to get 'all' qubits, single qubit specified as string
        or list of qubits, checking they are in self.qubits
        :param qb_names: 'all', single qubit name (eg. 'qb1') or list of
            qb names
        :param strict: (bool, default: True) if this is True, an error is
            raised if entries of qb_names are not found in the stored qubits.
            If this is False and there are qb_names that are not found,
            their names will be returned as they are, and no qubit objects
            will be returned.
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

        all_found = True
        for qb in qb_names:
            if qb not in self.qb_names:
                if strict:
                    raise AssertionError(f"{qb} not found in {self.qb_names}")
                all_found = False
        if self.qubits is None or not all_found:
            return None, qb_names
        else:
            qb_map = {qb.name: qb for qb in self.qubits}
            return [qb_map[qb] for qb in qb_names], qb_names

    def get_reset_params(self, qb_names="all"):
        """Gets a copy of preparation parameters (used for active reset,
        preselection) for qb_names.

        Args:
            qb_names (list): list of qubit names for which the
                reset params should be retrieved. Default is 'all',
                which retrieves the reset parameters for all qubits.
                Note: this parameter is overwritten by
                self.reset_params["qb_names"] in case it exists.

        Returns:
            reset_params (dict): deep copy of preparation parameters

        """
        reset_params = deepcopy(self.reset_params) or {}

        # collect them from CircuitBuilder
        steps, analysis_instructions = self._get_reset_steps(
            qb_names=reset_params.get('qb_names', qb_names),
            steps=reset_params.get('steps', None))

        reset_params.update(dict(steps=steps, analysis_instructions=
                                 analysis_instructions))
        return reset_params

    def _get_reset_steps(self, qb_names="all", steps=None):
        """Constructs the global reset steps for the given qubits in qb_names
        by retrieving the steps of the individual qubits in qb.reset.steps
        or from the passed `steps`
        Args:
            qb_names (str, list): list of qubit names.
            steps (None, list, str): list of reset steps for all qubits in
                qb_names. If None (default), steps are taken for each qubit
                from qb.reset.steps

        Returns:
            reset_steps (dict): preparation parameters
            analysis_instructions (dict)

        """
        qubits, qb_names = self.get_qubits(qb_names)
        # collect steps and ensure they are lists, and deepcopy them to allow
        # modification within this function without affecting original object
        if qubits is None:
            # no qubits objects passed, so no reset possible
            log.warning("Failed to retrieve reset "
                        "steps from qb.reset.steps() because no qubits "
                        "objects were found (self.qubits is None). "
                        "No reset steps will be added.")
            reset_steps = {qbn: [] for qbn in qb_names}
            analysis_instructions = {qbn: [dict(preparation_type="wait")]
                                     for qbn in qb_names}
        elif steps is None:
            try:
                reset_steps = {qb.name: deepcopy(qb.reset.steps()) for qb in qubits}
                analysis_instructions = \
                    {qb.name: [qb.reset.submodules.get(s).get_analysis_instructions()
                               for s in qb.reset.steps() if s != "padding"]
                     for qb in qubits}
            except AttributeError as e:
                log.error(f"Failed to retrieve reset "
                          f"steps from qb.reset.steps(): {e}. "
                          f"No reset steps will be added.")
                reset_steps = {qbn: [] for qbn in qb_names}
                analysis_instructions = {qbn: [dict(preparation_type="wait")]
                                         for qbn in qb_names}
        else:
            if isinstance(steps, str):
                steps = [steps]
            reset_steps = {qb.name: deepcopy(steps) for qb in qubits}
            analysis_instructions =\
                {qb.name: [qb.reset.submodules.get(s).get_analysis_instructions()
                           for s in steps]
                for qb in qubits}
        return reset_steps, analysis_instructions

    def get_cz_operation_name(self, qb1=None, qb2=None, op_code=None,
                              cz_pulse_name=None, **kw):
        """Finds the name of the CZ gate between qb1-qb2 that exists in
        self.operation_dict.
        :param qb1: name of qubit object of one of the gate qubits
        :param qb2: name of qubit object of the other gate qubit
        :param op_code: provide an op_code instead of qb1 and qb2
        :param cz_pulse_name: specify a CZ pulse name different from
            self.cz_pulse_name (if not specified in the op_code)

        :param kw: keyword arguments:
            cz_pulse_name: a custom cz_pulse_name instead of the stored one

        :return: the CZ gate name
        """
        assert (qb1 is None and qb2 is None and op_code is not None) or \
               (qb1 is not None and qb2 is not None and op_code is None), \
            "Provide either qb1&qb2 or op_code!"
        if cz_pulse_name is None:
            cz_pulse_name = self.cz_pulse_name
        if op_code is not None:
            op_split = op_code.split(' ')
            qb1, qb2 = op_split[1:]
            if op_split[0] != 'CZ':
                cz_pulse_name = op_split[0]

        if not self.fast_mode:
            _, (qb1, qb2) = self.get_qubits([qb1, qb2])
        if f"{cz_pulse_name} {qb1} {qb2}" in self.operation_dict:
            return f"{cz_pulse_name} {qb1} {qb2}"
        elif f"{cz_pulse_name} {qb2} {qb1}" in self.operation_dict:
            return f"{cz_pulse_name} {qb2} {qb1}"
        else:
            raise KeyError(f'CZ gate "{cz_pulse_name} {qb1} {qb2}" not found.')

    def get_pulse(self, *args, **kwargs):
        """Wrapper for get_pulses, for backwards compatibility only"""
        pulses = self.get_pulses(*args, **kwargs)
        if len(pulses) > 1:
            raise ValueError("get_pulse returned several pulses, please use "
                             "the newest version - get_pulses - instead!")
        return pulses[0]

    def get_pulses(self, op):
        """Gets pulse dictionaries, corresponding to the operation op, from the
        operation dictionary, and possibly parses logical indexing as well as
        arbitrary angles.

        Examples:
             >>> get_pulses('CZ 0 2')
             will perform a CZ gate (according to cz_pulse_name)
             between the qubits with logical indices 0 and 2
             >>> get_pulses('Z100 qb1')
             will perform a 100 degree Z rotation
             >>> get_pulses('Z:theta qb1')
             will perform a parametric Z rotation with parameter name theta
             >>> get_pulses('X:2*[theta] qb1')
             will perform a parametric X rotation with twice the
             value of the parameter named theta. The brackets are used to
             indicated the parameter name. This feature has also been tested
             with some more complicated mathematical expression.
             Note: the mathematical expression should be entered without any
             spaces in between operands. For instance,
             'Z:2*[theta]/180+[theta]**2 qb1' and not
             'Z: 2 * [theta] / 180 + [theta]**2 qb1
             >>> get_pulses('CZ_nztc40 qb3 qb2')
             will perform a 40 degrees CZ gate (if allowed by the CZ operation),
             using the CZ_nztc hardware implementation
        Adding 's' (for simultaneous) in front of an op_code (e.g.,
        'sZ:theta qb1') will reference the pulse to the start of the
        previous pulse.
        Adding 'm' (for minus) in front of an op_code (e.g.,
        'mZ:theta qb1') negates the sign. If 's' is also given,
        it has to be in the order 'sm'.
        In addition, if self.decompose_rotation_gates is not None,
        arbitrary-angle CZ gates are decomposed into two standard CZ gates
        and single-qubit gates. Examples:
            decompose_rotation_gates={'CZ_nztc': True} decomposes all CZ_nztc
            decompose_rotation_gates={'CZ_nztc': [['qb3', 'qb2'],]} decomposes
                all CZ_nztc between qb2 and qb3, and applies the single-qubit
                gates to qb3 (first qubit in the pair)
        Note that get_pulses parses one operation 'op', but may return more than
        one pulse, for example for gate decomposition.

        Args:
            op: operation (str in the above format, or iterable
            corresponding to the splitted string)

        Returns: list of pulses, where each pulse is a copy (see self.copy_op)
        of the corresponding pulse dictionary

        """
        # op_split is further parsed and used below
        # op is the op_code which will be stored in the returned pulse
        if isinstance(op, str):
            op_split = op.split(" ")
        else:
            op_split = op
            op = " ".join(op)
        # Extract op_name and qbn (qubit names)
        # op_name: first part of the op_code, e.g. "Z:2*[theta]"
        op_name = op_split[0]
        if self.fast_mode:
            qbn = op_split[1:]
        else:
            # the call to get_qubits resolves qubits indices if needed
            _, qbn = self.get_qubits(op_split[1:], strict=False)
        simultaneous = False
        if op_name[0] == 's':
            simultaneous = True
            op_name = op_name[1:]

        if op in self.operation_dict:
            p = [self.copy_op(self.operation_dict[op])]

        else:
            # assumes operation format of, e.g., f" Z{angle} qbname"
            # FIXME: This parsing is format dependent and is far from ideal but
            #  to generate parametrized pulses it is helpful to be able to
            #  parse Z gates etc.
            # FIXME e- allows to recognise numbers in scientific notation,
            #  but could be made more specific
            op_type = op_name.split(':')[0].rstrip('0123456789.e-')
            angle = op_name[len(op_type):]
            sign = -1 if op_type[0] == 'm' else 1
            if sign == -1:
                op_type = op_type[1:]
            allowed_ops = ['X', 'Y', 'Z', 'CZ']
            # startswith is needed to recognise all CZ gates, e.g. 'CZ_nztc'
            if not any([op_type.startswith(g) for g in allowed_ops]):
                raise KeyError(f'Gate "{op}" not found.')
            param = None  # e.g. 'theta' if angle = '2*[theta]'
            if angle:
                if angle[0] == ':':  # angle depends on a parameter
                    angle = angle[1:]
                    param, angle, param_start = self._parse_param(angle)

            if op_type.startswith('CZ'):  # Two-qubit gate
                if param is not None:
                    # FIXME: this code block is duplicated 3 times, for each
                    #  gate type (CZ, Z, X/Y). This should be cleaned up once
                    #  we improve or generalise further what op codes can be
                    #  parsed by this method.
                    if param_start > 0:
                        func_op_code = eval('lambda x, cb=self : ' + angle)
                    else:
                        func_op_code = None
                    # sign * means that func will be -func_op_code
                    cphase = sign * ParametricValue(
                        param, func=func_op_code, func_op_code=func_op_code,
                        op_split=[op_name, *qbn])
                # op_name = "NameVal" (e.g. "Z100", see docstring)
                elif angle:
                    cphase = float(angle)  # gate angle
                else:  # no cphase specified: standard CZ gate (180 deg)
                    cphase = None
                # Extract the operation which is available in the device
                # If the op_code only contains the generic keyword 'CZ',
                # this will use the default CZ pulse name of the experiment
                device_op = self.get_cz_operation_name(
                    *qbn,
                    cz_pulse_name=None if op_type == 'CZ' else op_type)
                # Get concrete operation implemented on the device, e.g. CZ_nztc
                op_type = device_op.split(" ")[0]
                # Here, we figure out if the gate should be decomposed into
                # CZ and single-qubit gates
                decomp_info = self.decompose_rotation_gates.get(op_type, False)
                # qb_dec is the qubit on which to apply single-qubit
                # gates in case of gate decomposition
                qb_dec = None
                if decomp_info==True:
                    # If True: we decompose the gate
                    # By default: use the qubit order defined in the op code
                    # to apply the single-qubit gates
                    qb_dec = qbn
                elif isinstance(decomp_info, list):
                    # If list: we decompose the gate if it is in the list...
                    for gate_to_decomp in decomp_info:
                        for qbn_reordered in [qbn, qbn[::-1]]:
                            if qbn_reordered == gate_to_decomp:
                                # ... and apply the single-qubit gates to the
                                # qubits in the order passed in
                                # decompose_rotation_gates
                                qb_dec = gate_to_decomp
                # Force resolving to a single CZ if no cphase is required
                # (note that CZ180 will still be decomposed)
                if cphase is None:
                    qb_dec = None
                # If qb_dec is not None, we decompose the gate
                if qb_dec:
                    if isinstance(cphase, ParametricValue):
                        # Update the op_split info in the ParametricValue,
                        # such that it matches the operation decomposition
                        cphase.op_split[0] = 'Z'
                    # CZ_x = diag(1,1,1,e^i*x)  # pycqed sign convention
                    #  = e^(i*x/4)*Z1(x/2)*Z0(x/2)*H1*CZ*H1*Z1(-x/2)*H1*CZ*H1
                    # and replacing each Hadamard H = i*Y*Z(pi) and
                    # discarding the global phase gives the decomposition:
                    decomposed_op = [
                        f'Y90 {qb_dec[0]}',
                        device_op,
                        f'Z180 {qb_dec[0]}',
                        f'Y90 {qb_dec[0]}',
                        f'Z0 {qb_dec[0]}',  # phase set below
                        f'Y90 {qb_dec[0]}',
                        device_op,
                        f'Z180 {qb_dec[0]}',
                        f'Y90 {qb_dec[0]}',
                        f'Z0 {qb_dec[0]}',  # phase set below
                        f'Z0 {qb_dec[1]}',  # phase set below
                    ]
                    p = [
                        self.copy_op(self.operation_dict[do])
                        for do in decomposed_op
                    ]
                    if isinstance(cphase, ParametricValue):
                        raise NotImplementedError
                        # The following will not work with ParametricValue:
                        # this should look like
                        # p[4]['basis_rotation'] = -cphase/2+180
                        # with cphase.func wrapping into a dict, as 'Z' below
                    p[4]['basis_rotation'] = {qb_dec[0]: -cphase/2+180}
                    p[9]['basis_rotation'] = {qb_dec[0]: cphase/2+180}
                    p[10]['basis_rotation'] = {qb_dec[1]: cphase/2}
                else:
                    p = [self.copy_op(self.operation_dict[device_op])]
                    if cphase is not None:
                        p[0]['cphase'] = cphase

            else:  # Single-qubit gate
                if self.decompose_rotation_gates.get(op_type, False):
                    raise NotImplementedError(
                        'Single qb decomposed rotations not implemented yet.')
                if angle == '':
                    angle = 180
                device_op = f"{op_type}180 {qbn[0]}"
                if device_op in self.operation_dict:
                    p = [self.copy_op(self.operation_dict[device_op])]
                else:
                    raise KeyError(f"Operation {op} not found.")
                if op_type == 'Z':
                    if param is not None:  # angle depends on a parameter
                        if param_start > 0:  # via a mathematical expression
                            func_op_code = eval('lambda x, cb=self : ' + angle)
                        else:  # angle = parameter
                            func_op_code = lambda x: x
                        # parameter func
                        func = (lambda x, qb=qbn[0], sign=sign, f=func_op_code:
                                {qb: sign * f(x)})
                        p[0]['basis_rotation'] = ParametricValue(
                            param, func=func, func_op_code=func_op_code,
                            op_split=(op_name, qbn[0]))
                    else:  # angle is a given value
                        # configure virtual Z gate for this angle
                        p[0]['basis_rotation'] = {qbn[0]: sign * float(angle)}
                else:
                    qb, _ = self.get_qubits(qbn[0])
                    corr_func = qb[0].calculate_nonlinearity_correction
                    if param is not None:  # angle depends on a parameter
                        if param_start > 0:  # via a mathematical expression
                            # combine the mathematical expression with a
                            # function that calculates the amplitude
                            func_op_code = eval('lambda x, cb=self : ' + angle)
                        else:  # angle = parameter
                            func_op_code = lambda x: x
                        func = lambda x, a=p[0]['amplitude'], sign=sign,\
                                      f=func_op_code: a * corr_func(
                                ((sign * f(x) + 180) % (-360) + 180) / 180)
                        p[0]['amplitude'] = ParametricValue(
                            param, func=func, func_op_code=func_op_code,
                            op_split=(op_name, qbn[0]))
                    else:  # angle is a given value
                        angle = sign * float(angle)
                        # configure drive pulse amplitude for this angle
                        p[0]['amplitude'] *= corr_func(
                            ((angle + 180) % (-360) + 180) / 180)
        if len(p) == 1:
            # If only one pulse: set its op_code to the initially requested one
            # in order to keep information provided there (e.g. 2qb gate type).
            # If more than one pulse: the operation has been decomposed into
            # several pulses, so these should keep their separate op_code.
            p[0]['op_code'] = op

        if simultaneous:
            p[0]['ref_point'] = 'start'

        return p

    def _parse_param(self, angle):
        param_start = angle.find('[') + 1
        # If '[' is contained, this indicates that the parameter
        # is part of a mathematical expression. Otherwise, the angle
        # is equal to the parameter.
        if param_start > 0:
            param_end = angle.find(']', param_start)
            param = angle[param_start:param_end]
            angle = angle.replace('[' + param + ']', 'x')
        else:
            param = angle
        return param, angle, param_start

    def swap_qubit_indices(self, i, j=None):
        """Swaps logical qubit indices by swapping the entries in self.qb_names.
        :param i: (int or iterable): index of the first qubit to be swapped or
            indices of the two qubits to be swapped (as two ints given in the
            first two elements of the iterable)
        :param j: index of the second qubit (if it is not set via param i)
        """
        if j is None:
            i, j = i[0], i[1]
        self.qb_names[i], self.qb_names[j] = self.qb_names[j], self.qb_names[i]

    def initialize(self, init_state='0', qb_names='all', reset_params=None,
                   simultaneous=True, block_name=None, pulse_modifs=None,
                   prepend_block=None):
        """Initializes the specified qubits with the corresponding init_state
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
        :param reset_params: preparation parameters
        :param simultaneous: (bool, default True) whether initialization
            pulses should be applied simultaneously.
        :param block_name: (str, optional) a name to replace the
            automatically generated block name of the initialization block
        :param pulse_modifs: (dict) Modification of pulses parameters.
            See method block_from_ops.
        :param prepend_block: (Block, optional) An extra block that will be
            executed between the preparation and the initialization.
        :return: init block
        """
        if block_name is None:
            block_name = f"Initialization_{qb_names}"
        _, qb_names = self.get_qubits(qb_names)
        if not len(qb_names):
            return Block(block_name, [])
        if reset_params is None:
            reset_params = self.get_reset_params(qb_names)
        if len(init_state) == 1 and isinstance(init_state, str):
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
        block = Block(block_name, pulses, copy_pulses=False)
        block.set_end_after_all_pulses()
        blocks = []
        # if reset parameters exist and location specifies reset should be done
        # at start of sequence (note: is this really the best way to encode where
        # to do the reset? shall we make this 'per_qubit')?)
        if len(reset_params) != 0 and \
                reset_params.get('location', 'start') == 'start':
            blocks.append(self.reset(**reset_params))
        if prepend_block is not None:
            blocks.append(prepend_block)
        if len(blocks) > 0:
            blocks.append(block)
            block = self.sequential_blocks(block_name, blocks)
        return block

    def finalize(self, init_state='0', qb_names='all', simultaneous=True,
                 block_name=None, pulse_modifs=None):
        """Applies the specified final rotation to the specified qubits.
        This is basically the same initialize, but without preparation.
        For parameters, see initialize().
        :return: finalization block
        """
        if block_name is None:
            block_name = f"Finalialization_{qb_names}"
        return self.initialize(init_state=init_state, qb_names=qb_names,
                               simultaneous=simultaneous,
                               reset_params={},
                               block_name=block_name,
                               pulse_modifs=pulse_modifs)

    def reset(self, qb_names='all', steps=None,
              step_alignment='start', alignment="end",
              block_name=None, **kws):
        """Prepares a reset of a list of qubits using a sequence of steps
        (globally provided as argument to this function or individually
         specified from qubit.init.steps).

        I.e.:

        qb0 --[step 0 ] -- ... - [step i]
        qb1 --[step 0 ] -- ... - [step j]

        Args:
            qb_names: which qubits to prepare. Defaults to all.
            steps (list, str, None, dict): list of steps for the preparation.
                If a list is provided, all steps are the same for all qubits.
                If a string is provided, it is converted to a single-entry list.
                If None (default), takes the steps for each qubit individually
                from qubit.init.steps. The different qubits can
                have different number of steps. The qubits with less steps
                will get padded (waiting time) in the beginning or the end
                of the block, depending on `alignment`.
                If dict, then it assumes that keys are qubit names and values
                are lists of steps.
                To get no reset whatsoever, pass an emtpy list.
                Constraints: if a step includes acquisition(s)
                (e.g. preselection, active reset), then they
                should happen simultaneously for all qubits which are on
                the same acquisition unit (constraint of, e.g., ZI devices) and the
                same number of times across all qubits sharing that acquisition
                device. Note that this function does not name the elements explicitly
                for the acquisition (this is handled by Segment.add()),
                and it DOES ALLOW steps which do not satisfy this
                constraint (in case the constraint is relaxed in the future by the
                instrument drivers).
            step_alignment (str): "start", "center" or "end". Indicates whether
                step i on all qubits should be start, center- or end- aligned.
                Note that if the steps contain readout pulses which
                are generated by the same acquisition unit, only "Start"
                should be used, to ensure the acquisition is started simultaneously
                on all qubits on that acquisition unit. Defaults to "start".
            alignment (str): "start" or "end". In case the number of
                steps is different for the different qubits, specifies whether
                the qubits should be aligned at the start or finish the preparation.

            block_name (str): Name of the preparation block.

        Returns:
            sequantial_blocks: A Reset_{qb_names} block with a sequential list
            of simultaneous prep blocks.
        """
        qubits, qb_names = self.get_qubits(qb_names)
        if block_name is None:
            block_name = f"Reset_{qb_names}"

        # parse steps if not yet done.
        if not isinstance(steps, dict):
            steps, _ = self._get_reset_steps(qb_names, steps)
        max_steps = np.max([len(steps) for steps in steps.values()])

        # pad qubits which have less steps
        for qbn in steps:
            padding = max_steps - len(steps[qbn])
            if alignment == "end":
                # add padding steps at the beginning
                steps[qbn] = ["padding"] * padding + steps[qbn]
            elif alignment == "start":
                # add padding steps at the end
                steps[qbn] = steps[qbn] + ["padding"] * padding
            else:
                raise NotImplementedError("Only start and end alignment of "
                                          "the preparation block is currently"
                                          " supported.")

        # Fill prep list first with parallel reset blocks
        prep = []
        for i in range(max_steps):

            # Collect all parallel Blocks in current step i for each qubit
            simultaneous_blocks = []
            for qb in qubits:
                if steps[qb.name][i] == "padding":
                    simultaneous_blocks.append(
                        Block(f"padding_step_{i}_{qb.name}", [])
                    )
                else: # if reset block
                    try:
                        reset_scheme = qb.reset.instrument_modules[steps[qb.name][i]]
                    except KeyError as e:
                        log.error(
                            f"Reset Step '{steps[qb.name][i]}' not known for {qb.name}."
                            f"Available steps are: {qb.reset.instrument_modules.keys()}")
                        raise e

                    simultaneous_blocks.append(
                        reset_scheme.reset_block(
                            f"step_{i}_{qb.name}",
                            sweep_params=self._reset_sweep_params.get(qb.name, None),
                        )
                    )

            # Finally append all parallel blocks of step i
            prep.append(
                self.simultaneous_blocks(
                    f"Reset_step_{i}",
                    simultaneous_blocks,
                    set_end_after_all_pulses=True,
                    block_align=step_alignment,
                )
            )

        # FIXME: Temporarily dropping support for f0g1_reset.
        #        Will be reintroduced with devel/f0g1_init_schemes.
        #
        # # f0g1 reset
        # elif preparation_type == 'f0g1_reset':
        #     preparation_pulses = []
        #     for i, qbn in enumerate(qb_names):
        #         preparation_pulses.append(
        #             self.get_pulse('f0g1_reset_pulse ' + qbn))
        #         preparation_pulses[-1]['ref_point'] = 'start'
        #         preparation_pulses[-1]['element_name'] = 'f0g1_reset'

        #         preparation_pulses.append(
        #             self.get_pulse('ef_for_f0g1_reset_pulse ' + qbn))
        #         preparation_pulses[-1]['ref_point'] = 'start'
        #         preparation_pulses[-1]['element_name'] = 'f0g1_reset'

        #     preparation_pulses[0]['ref_pulse'] = ref_pulse
        #     preparation_pulses[0]['name'] = 'f0g1_reset_pulse'
        #     preparation_pulses[0]['pulse_delay'] = 0  # -ro_separation

        #     preparation_pulses[1]['ref_pulse'] = ref_pulse
        #     preparation_pulses[1]['name'] = 'ef_for_f0g1_reset_pulse'
        #     preparation_pulses[1]['pulse_delay'] = 0  # -ro_separation

        #     block_end = dict(name='end', pulse_type="VirtualPulse",
        #                      ref_pulse='f0g1_reset_pulse',
        #                      pulse_delay=ro_separation,
        #                      ref_point='end')
        #     preparation_pulses += [block_end]
        #     return Block(block_name, preparation_pulses, copy_pulses=False)


        return self.sequential_blocks(block_name, prep)

    def mux_readout(self, qb_names='all', element_name='RO', block_name="Readout",
                    reset_params=None,
                    **pulse_pars):
        _, qb_names = self.get_qubits(qb_names)
        ro_pulses = []
        for j, qb_name in enumerate(qb_names):
            ro_pulse = self.copy_op(self.operation_dict['RO ' + qb_name])
            ro_pulse['name'] = '{}_{}'.format(element_name, j)
            ro_pulse['element_name'] = element_name
            if j == 0:
                ro_pulse.update(pulse_pars)
            else:
                ro_pulse['ref_point'] = 'start'
            ro_pulses.append(ro_pulse)
        block = Block(block_name, ro_pulses, copy_pulses=False)
        block.set_end_after_all_pulses()
        if reset_params is None:
            reset_params = self.get_reset_params(qb_names)

        if reset_params.get("location", "start") == "end":
            return self.sequential_blocks(
                f"{block_name}_and_reset",
                [block, self.reset(**reset_params)]
            )
        else:
            return block

    def Z_gate(self, theta=0, qb_names='all'):
        """Software Z-gate of arbitrary rotation.

        :param theta:           rotation angle, in degrees
        :param qb_names:      pulse parameters (dict)

        :return: Pulse dict of the Z-gate
        """
        # if qb_names is the name of a single qb, expects single pulse output
        single_qb_given = not isinstance(qb_names, list)
        _, qb_names = self.get_qubits(qb_names)
        pulses = [p for qbn in qb_names for p in self.get_pulses(
            f'Z{theta} {qbn}')]
        return pulses[0] if single_qb_given else pulses

    def get_ops_duration(self, operations=None, pulses=None, fill_values=None,
                         pulse_modifs=None, init_state='0'):
        """Calculates the total duration of the operations by resolving a dummy
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
        """Returns a block with the given operations.
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
            - Format 1:
              keys: indices of the pulses on  which the pulse modifications
                should be made
              values: dictionaries of modifications
              E.g. ops = ["X180 qb1", "Y90 qb2"],
              pulse_modifs = {1: {"ref_point": "start"}}
              This will modify the pulse "Y90 qb2" and reference it to the start
              of the first one.
            - Format 2:
              keys: strings in the format specified for the param
                sweep_dicts_list in the docstring of Block.build to identify
                an attribute in a pulse
              values: the value to be set for the attribute identified by the
                corresponding key
        :return: The created block
        """
        if pulse_modifs is None:
            pulse_modifs = {}
        if isinstance(operations, str):
            operations = [operations]
        if fill_values is not None and len(fill_values):
            def op_format(op, **fill_values):
                if isinstance(op, str):
                    return op.format(**fill_values)
                else:
                    return [s.format(**fill_values) for s in op]
            operations = [op_format(op, **fill_values) for op in operations]
        # the shortcut if op in self.operation_dict is for speed reasons
        p_lists = [[self.copy_op(self.operation_dict[op])]
                   if op in self.operation_dict
                   else self.get_pulses(op)
                   for op in operations]
        pulses = [p for p_list in p_lists for p in p_list]  # flattened

        return Block(block_name, pulses, pulse_modifs, copy_pulses=False)

    def seg_from_cal_points(self, cal_points, init_state='0', ro_kwargs=None,
                            block_align='end', segment_prefix='calibration_',
                            sweep_dicts_list=None, sweep_index_list=None,
                            **kw):
        """Returns a list of segments for each cal state in cal_points.states.
        :param cal_points: CalibrationPoints instance
        :param init_state: initialization state (string or list),
            see documentation of initialize().
        :param ro_kwargs: Keyword arguments (dict) for the function
            mux_readout().
        :param block_align: passed to simultaneous_blocks; see docstring there
        :param segment_prefix: prefix for segment name (string)
        :param sweep_dicts_list: Sweep dicts passed on to Block.build for the
            complete cal_state_block to build a block that corresponds to a
            point of an N-dimensional sweep.
        :param sweep_index_list: Passed on to Block.build for the complete
            cal_state_block. Determines for which sweep points from
            sweep_dicts_list the block should be build.
        :param kw: additional keyword arguments
            df_values_per_point (int, default: 1): number of expected number of readouts
                per sweep point.

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

            prep = self.initialize(init_state='g',
                                   qb_names=cal_points.qb_names)
            ro = self.mux_readout(**ro_kwargs, qb_names=cal_points.qb_names)
            cal_state_block = self.sequential_blocks(
                f'cal_states_{i}', [prep, parallel_qb_block, ro])
            vals_per_point = kw.get('df_values_per_point', 1)
            for j in range(vals_per_point):
                seg = Segment(f'{segment_prefix}_{i*vals_per_point+j}'
                              f'_{"".join(seg_states)}',
                              cal_state_block.build(
                                  sweep_dicts_list=sweep_dicts_list,
                                  sweep_index_list=sweep_index_list))
                segments.append(seg)

        return segments

    def block_from_anything(self, pulses, block_name):
        """Convert various input formats into a `Block`.

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
            return self.block_from_pulse_dicts([pulses], block_name=block_name)
        elif isinstance(pulses[0], str):  # list of opcodes
            return self.block_from_ops(block_name, pulses)
        elif isinstance(pulses[0], dict):  # list of pulse dicts
            return self.block_from_pulse_dicts(pulses, block_name=block_name)

    def block_from_pulse_dicts(self, pulse_dicts,
                               block_name='from_pulse_dicts'):
        """Generates a block from a list of pulse dictionaries.

        Args:
            pulse_dicts: list
                Pulse dictionaries, each containing either 1) an op_code of the
                desired pulse plus optional pulse parameters to overwrite the
                default values of the chosen operation, or 2) a full set of
                pulse parameters.
            block_name: str, optional
                Name of the resulting block
        Returns:
             A block containing the pulses in pulse_dicts
        """
        pulses = []
        if pulse_dicts is not None:
            for i, pp in enumerate(pulse_dicts):
                # op_code determines which pulse to use
                p_list = self.get_pulses(pp['op_code']) if 'op_code' in pp \
                    else [{}]
                # all other entries in the pulse dict are interpreted as
                # pulse parameters that overwrite the default values
                pp_add_entries = {k: v for k, v in pp.items() if k != 'op_code'}
                if len(pp_add_entries):
                    if len(p_list) > 1:
                        raise NotImplementedError(
                            "get_pulses returned more than one pulse for "
                            f"pulse dict {i}, and it is not clear to which "
                            f"one the parameters passed in the pulse dict "
                            f"should be applied.")
                    p_list[0].update(pp_add_entries)
                pulses += p_list
        return Block(block_name, pulses)

    def seg_from_ops(self, operations, fill_values=None, pulse_modifs=None,
                     init_state='0', seg_name='Segment1', ro_kwargs=None):
        """Returns a segment with the given operations using the function
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
        """Returns a sequence with the given operations using the function
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
                            disable_block_counter=False, destroy=False):
        """Creates a block with name :block_name: that consists of the parallel
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
            destroy (bool or list of bools, default False): whether the
                individual blocks can be destroyed (speedup by avoiding
                copying pulses, see Block.build).
        """
        simultaneous = Block(block_name, [])

        if not hasattr(destroy, '__iter__'):
            destroy = [destroy] * len(blocks)

        simultaneous_end_pulses = []

        # saves computation time in Segment.resolve_timing
        if block_align == 'start':
            block_align = None

        for block, d in zip(blocks, destroy):
            if set_end_after_all_pulses:
                block.set_end_after_all_pulses()
            # if there is already custom block start, then just update it
            # with block alignment otherwise create it
            block_start = block.block_start or {'block_align': block_align}
            simultaneous.extend(block.build(
                ref_pulse="start", block_start=block_start,
                name=block.name if disable_block_counter else None,
                destroy=d))
            simultaneous_end_pulses.append(simultaneous.pulses[-1]['name'])

        # the name of the simultaneous_end_pulse is used in
        # Segment.resolve_timing and should not be changed
        simultaneous.extend([{"name": "simultaneous_end_pulse",
                              "pulse_type": "VirtualPulse",
                              "pulse_delay": 0,
                              "ref_pulse": simultaneous_end_pulses,
                              "ref_point": 'end',
                              "ref_function": 'max'
                              }])
        return simultaneous

    def sequential_blocks(self, block_name, blocks,
                          set_end_after_all_pulses=False,
                          disable_block_counter=False, destroy=False):
        """Creates a block with name :block_name: that consists of the serial
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
            destroy (bool or list of bools with same len as blocks, default
                False): whether the individual blocks can be destroyed
                (speedup by avoiding copying pulses, see Block.build).
        """
        sequential = Block(block_name, [])
        if not hasattr(destroy, '__iter__'):
            destroy = [destroy] * len(blocks)
        for block, d in zip(blocks, destroy):
            if set_end_after_all_pulses:
                block.set_end_after_all_pulses()
            sequential.extend(block.build(
                name=block.name if disable_block_counter else None, destroy=d))
        return sequential

    def sweep_n_dim(self, sweep_points, body_block=None, body_block_func=None,
                    cal_points=None, init_state='0', seq_name='Sequence',
                    ro_kwargs=None, return_segments=False, ro_qubits='all',
                    repeat_ro=True, init_kwargs=None, final_kwargs=None,
                    segment_kwargs=None, **kw):
        """Creates a sequence or a list of segments by doing an N-dim sweep
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
        :param segment_kwargs: Keyword arguments (dict) passed to segment.
            (default: None)
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
                1D: sequence, np.arange(number of acquisition elements)
                2D: list of sequences, [np.arange(number of acquisition elements),
                                        np.arange(number of sequences)]
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
        if segment_kwargs is None:
            segment_kwargs = {}

        nr_sp_list = sweep_points.length()
        if sweep_dims == 1:
            sweep_points = copy(sweep_points)
            sweep_points.add_sweep_dimension()
            nr_sp_list.append(1)

        ro = self.mux_readout(**ro_kwargs, qb_names=ro_qubits)
        _, all_ro_qubits = self.get_qubits(ro_qubits)
        # FIXME: checking whether op_code starts with RO or Acq is a good
        #  hack because mux_readout() was modified to also sometimes be RO + reset
        #  but maybe a cleaner solution would be to have a separate block for the
        #  end reset after the readout?
        # FIXME: add an explaining comment why the space in 'RO ' is necessary to
        # avoid matching other op_codes. Which other op_codes starting with 'RO' are there?

        all_ro_op_codes = [p['op_code'] for p in ro.pulses if "op_code" in p
                           and (p['op_code'].startswith('RO ') # space after RO required to avoid matching the wrong OP code
                                or p['op_code'].startswith('Acq'))]
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
                    disable_block_counter=True,
                    destroy=[False, body_block is None, False, False])
                seg = Segment(f'seg{j}', segblock.build(
                    sweep_dicts_list=(
                        None if (body_block is None and self.fast_mode)
                        else sweep_points), sweep_index_list=[j, i],
                    destroy=True), fast_mode=self.fast_mode, **segment_kwargs)
                # apply Segment sweep points
                for dim in [0, 1]:
                    for param in sweep_points[dim]:
                        if param.startswith('Segment.'):
                            vals = sweep_points.get_sweep_params_property(
                                'values', dim, param)
                            setattr(seg, param[len('Segment.'):],
                                    deepcopy(vals[j if dim == 0 else i]))
                # add the new segment to the sequence
                seq.add(seg)
            if cal_points is not None:
                block_align_cal_pts = kw.get('block_align_cal_pts', 'end')
                seq.extend(self.seg_from_cal_points(
                    cal_points, init_state, ro_kwargs,
                    block_align=block_align_cal_pts,
                    sweep_dicts_list=sweep_points[1:], sweep_index_list=[i],
                    **kw))
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
        """Generates a complete list of tomography pulse lists for tomo_qubits.
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
