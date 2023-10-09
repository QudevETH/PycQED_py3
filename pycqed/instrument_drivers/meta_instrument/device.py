"""
The Device class is intended to be used for:
    * store general information about the device (including connectivity graph)
      and references to qubit objects
    * store two-qubit gate parameters

The structure is chosen to resemble the one of the QuDev_transmon class. As such, the two-qubit gate parameters
are stored as instrument parameters of the device, as is the case for single-qubit gates for the QuDev_transmon
class.

* add_2qb_gate *
New two-qubit gates can be added using the add_2qb_gate(gate_name, pulse_type) method. It takes the gate name and the
pulse type intended to be used for the gate as input. It scans the pulse_library.py file for the provided pulse type.
Using the new pulse_params() method of the pulse, the relevant two qubit gate parameters can be added for each connected
qubit.

* get_operation_dict *
As for the QuDev_transmon class the Device class has the ability to return a dictionary of all device operations
(single- and two-qubit) in the form of a dictionary, using the get_operation_dict method.
"""

# General imports
import logging
from copy import copy, deepcopy
import numpy as np
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt

import pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon as qdt
import pycqed.measurement.waveform_control.pulse as bpl
from pycqed.instrument_drivers.instrument import Instrument
from qcodes.instrument.parameter import (ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import plotting as plot_mod
from collections import OrderedDict

log = logging.getLogger(__name__)

class Device(Instrument):
    # params that should not be loaded by pycqed.utilities.general.load_settings
    _params_to_not_load = {'qubits'}


    def __init__(self, name, qubits, connectivity_graph, **kw):
        """
        Instantiates device instrument and adds its parameters.

        Args:
            name (str): name of the device
            qubits (list of QudevTransmon or names of QudevTransmon objects): qubits of the device
            connectivity_graph: list of elements of the form [qb1, qb2] with qb1 and qb2 QudevTransmon objects or names
                         thereof. qb1 and qb2 should be physically connected on the device.
        """
        super().__init__(name, **kw)

        qb_names = [qb if isinstance(qb, str) else qb.name for qb in qubits]
        qubits = [qb if not isinstance(qb, str) else self.find_instrument(qb) for qb in qubits]
        connectivity_graph = [[qb1 if isinstance(qb1, str) else qb1.name,
                               qb2 if isinstance(qb2, str) else qb2.name] for [qb1, qb2] in connectivity_graph]
        self._two_qb_gates = []

        for qb in qubits:
            setattr(self, qb.name, qb)

        self.qubits = qubits
        self.add_parameter('qb_names',
                           vals=vals.Lists(),
                           initial_value=qb_names,
                           parameter_class=ManualParameter)

        self.MWGs = []
        self.TWPAs = []

        self._operations = {}  # dictionary containing dictionaries of operations with parameters

        # Instrument reference parameters
        self.add_parameter('instr_mc',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_dc_source',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_trigger',
                           parameter_class=InstrumentRefParameter)

        self.add_parameter('connectivity_graph',
                           vals=vals.Lists(),
                           label="Qubit Connectivity Graph",
                           docstring="Stores the connections between the qubits "
                                     "in form of a list of lists [qbi_name, qbj_name]",
                           parameter_class=ManualParameter,
                           initial_value=connectivity_graph
                           )
        self.add_parameter('last_calib',
                           vals=vals.Strings(),
                           initial_value='',
                           docstring='stores timestamp of last calibration',
                           parameter_class=ManualParameter)

        self.add_parameter('operations',
                           docstring='a list of operations on the device, without single QB operations.',
                           get_cmd=self._get_operations)
        self.add_parameter('two_qb_gates',
                           docstring='stores all two qubit gate names',
                           get_cmd=lambda s=self: copy(s._two_qb_gates)
                           )
        self.add_parameter(
            'default_cz_gate_name',
            parameter_class=ManualParameter, initial_value=None,
            vals=vals.MultiType(vals.Strings(), vals.Enum(None)),
            set_parser=self._valid_cz_gate,
            docstring='Name of the CZ gate that should be used by default.')

        self.add_parameter('relative_delay_graph',
                           label='Relative Delay Graph',
                           docstring='Stores the relative delays between '
                                     'drive and flux channels of the device.',
                           initial_value=RelativeDelayGraph(),
                           parameter_class=ManualParameter,
                           set_parser=RelativeDelayGraph)

        self.add_parameter('flux_crosstalk_calibs',
                           parameter_class=ManualParameter,
                           )

        # Pulse preparation parameters
        default_prep_params = dict(preparation_type='wait',
                                   post_ro_wait=1e-6, reset_reps=1)

        self.add_parameter('preparation_params', parameter_class=ManualParameter,
                           initial_value=default_prep_params, vals=vals.Dict())

    # General Class Methods

    def add_operation(self, operation_name):
        """
        Adds the name of an operation to the operations dictionary.

        Args:
            operation_name (str): name of the operation
        """

        self._operations[operation_name] = {}

    def add_pulse_parameter(self, operation_name, parameter_name, argument_name,
                            initial_value=None, **kw):
        """
        Adds a pulse parameter to an operation. Makes sure that parameters are not duplicated.
        Adds the pulse parameter to the device instrument.

        Args:
            operation_name (tuple): name of operation in format (gate_name, qb1, qb2)
            parameter_name (str): name of parameter
            argument_name (str): name of the argument that is added to the operations dict
            initial_value: initial value of parameter
        """
        if parameter_name in self.parameters:
            raise KeyError(
                'Duplicate parameter name {}'.format(parameter_name))

        if operation_name in self.operations().keys():
            self._operations[operation_name][argument_name] = parameter_name
        else:
            raise KeyError('Unknown operation {}, add '.format(operation_name) +
                           'first using add operation')

        self.add_parameter(parameter_name,
                           initial_value=initial_value,
                           parameter_class=ManualParameter, **kw)

    def _get_operations(self):
        """
        Private method that is used as getter function for operations parameter
        """
        return self._operations

    def get_operation_dict(self, operation_dict=None, qubits="all"):
        """
        Returns the operations dictionary of the device and qubits, combined with the input
        operation_dict.

        Args:
            operation_dict (dict): input dictionary the operations should be added to
            qubits (list, str): set of qubits to which the operation dictionary should be
                restricted to.

        Returns:
            operation_dict (dict): dictionary containing both qubit and device operations

        """
        qubits = self.get_qubits(qubits, "str")

        if operation_dict is None:
            operation_dict = dict()

        # add 2qb operations
        two_qb_operation_dict = {}
        for op_tag, op in self.operations().items():
            # op_tag is the tuple (gate_name, qb1, qb2) and op the dictionary of the
            # operation
            if op_tag[1] not in qubits or op_tag[2] not in qubits:
                continue
            # Add both qubit combinations to operations dict
            # Still return a string instead of tuple as keys to be consistent
            # with QudevTransmon class
            this_operation = {}
            for argument_name, parameter_name in op.items():
                this_operation[argument_name] = self.get(parameter_name)
            this_operation['op_code'] = op_tag[0] + ' ' + op_tag[1] + ' ' \
                                        + op_tag[2]
            for op_name in [op_tag[0] + ' ' + op_tag[1] + ' ' + op_tag[2],
                            op_tag[0] + ' ' + op_tag[2] + ' ' + op_tag[1]]:
                two_qb_operation_dict[op_name] = this_operation

        operation_dict.update(two_qb_operation_dict)

        # add sqb operations
        for qb in self.get_qubits(qubits):
            operation_dict.update(qb.get_operation_dict())

        # add meas_obj operations
        for mobj in self.TWPAs:
            operation_dict.update(mobj.get_operation_dict())

        return operation_dict

    def get_qb(self, qb_name):
        """
        Wrapper: Returns the qubit instance with name qb_name

        Args:
            qb_name (str): name of the qubit
        Returns:
            qubit instrument with name qubit_name

        """
        return self.find_instrument(qb_name)

    def get_qubits(self, qubits='all', return_type="obj"):
        """
        Wrapper to get qubits as object or str (names), from different
        specification methods. Checks whether qubits are on device.

        or list of qubits objects, checking they are in self.qubits
        :param qubits (str, list): Accepts the following formats:
            - "all" returns all qubits on device, default behavior
            - single qubit string, e.g. "qb1",
            - list of qubit strings, e.g. ['qb1', 'qb2']
            - list of qubit objects, e.g. [qb1, qb2]
            - list of integers specifying the index, e.g. [0, 1] for qb1, qb2
        :param return_type (str): "obj" --> qubit objects are returned.
            "str": --> qubit names are returned.
            "ind": --> returns indices to find the qubits in self.qubits
        :return: list of qb_names or qb objects. Note that a list is
            returned in all cases
        """
        if return_type not in ['obj', 'str', 'ind']:
            raise ValueError(f'Return type: {return_type} not understood')

        qb_names = [qb.name for qb in self.qubits]
        if qubits == 'all':
            if return_type == "obj":
                return copy(self.qubits)
            elif return_type == "str":
                return qb_names
            else:
                return list(range(len(qb_names)))
        elif not isinstance(qubits, (list, tuple)):
            qubits = [qubits]

        # test if qubit indices were provided instead of names
        try:
            ind = [int(i) for i in qubits]
            qubits = [qb_names[i] for i in ind]
        except (ValueError, TypeError):
            pass

        # check whether qubit is on device
        for qb in qubits:
            if not isinstance(qb, (str)): # then should be a qubit object
                qb = qb.name
            assert qb in qb_names, \
                f"{qb} not found on device with qubits: {qb_names}"

        # return subset of qubits
        qubits_to_return = []
        for qb in qubits:
            if not isinstance(qb, str):  # then should be a qubit object
                qb = qb.name
            qubits_to_return.append(qb)

        if return_type == "str":
            return qubits_to_return
        elif return_type == "obj":
            return [self.qubits[qb_names.index(qbn)]
                    for qbn in qubits_to_return]
        else:
            return [qb_names.index(qb) for qb in qubits_to_return]

    def get_pulse_par(self, gate_name, qb1, qb2, param):
        """
        Returns the object of a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
        Returns:
            Parameter object
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            return getattr(self, f'{gate_name}_{qb1_name}_{qb2_name}_{param}')
        except AttributeError:
            try:
                return getattr(self, f'{gate_name}_{qb2_name}_{qb1_name}_{param}')
            except AttributeError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def get_prep_params(self, qb_list):
        """
        Returns the preparation paramters for all qubits in qb_list.

        Args:
            qb_list (list): list of qubit names or objects

        Returns:
            dictionary of preparation parameters
        """

        qb_list = self.get_qubits(qb_list)

        # threshold_map has to be updated for all qubits
        thresh_map = {}
        for i, prep_params in enumerate([qb.preparation_params()
                                         for qb in qb_list]):
            if 'threshold_mapping' in prep_params:

                thresh_map.update({qb_list[i].name:
                                       prep_params['threshold_mapping']})

        prep_params = deepcopy(self.preparation_params())
        prep_params['threshold_mapping'] = thresh_map

        return prep_params

    def get_meas_obj_value_names_map(self, qubits, multi_uhf_det_func):
        # we cannot just use the value_names from the qubit detector functions
        # because the UHF_multi_detector function adds suffixes

        qubits = self.get_qubits(qubits)
        if multi_uhf_det_func.detectors[0].name == 'raw_classifier_det':
            meas_obj_value_names_map = {
                qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                    multi_uhf_det_func.value_names,
                    qb.int_avg_classif_det.value_names)
                for qb in qubits}
        elif multi_uhf_det_func.detectors[0].name == \
                'AveragingPollDetector':
            meas_obj_value_names_map = {
                qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                    multi_uhf_det_func.value_names, qb.inp_avg_det.value_names)
                for qb in qubits}
        else:
            meas_obj_value_names_map = {
                qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                    multi_uhf_det_func.value_names, qb.int_avg_det.value_names)
                for qb in qubits}

        meas_obj_value_names_map.update({
            name + '_object': [name] for name in
            [vn for vn in multi_uhf_det_func.value_names if vn not in
             hlp_mod.flatten_list(list(meas_obj_value_names_map.values()))]})

        return meas_obj_value_names_map

    def get_msmt_suffix(self, qubits='all'):
        """
        Function to get measurement label suffix from the measured qubit names.
        :param qubits: list of QuDev_transmon instances.
        :return: string with the measurement label suffix
        """
        qubits = self.get_qubits(qubits)
        qubit_names = self.get_qubits(qubits, "str")
        if len(qubit_names) == 1:
            msmt_suffix = qubits[0].msmt_suffix
        elif len(qubit_names) > 5:
            msmt_suffix = '_{}qubits'.format(len(qubit_names))
        else:
            msmt_suffix = '_{}'.format(''.join([qbn for qbn in qubit_names]))

        return msmt_suffix

    def get_channel_map(self, qubits="all", drive=True, ro=True, flux=True):
        """
        Gets the channel map for `qubits`
        Args:
            qbs (list): list of qubit objects
            drive (bool): whether or not to include drive pulse channel
            ro (bool): whether or not to include the readout pulse channel
            flux (bool): whether or not to include the flux pulse channel

        Returns:
            channel map (dict): keys are qubit names, values are list of channels
                names.
        """
        qbs = self.get_qubits(qubits, "obj")
        channel_map = {}
        [channel_map.update(qb.get_channel_map(drive=drive, ro=ro, flux=flux))
         for qb in qbs]
        return channel_map

    def get_interaction_frequencies(self, qubit_pairs=None, cz_pulse_name=None,
                                    reference_qb='high'):
        """
        Gets the interaction frequencies for CZ gates between specified qubit pairs.

        Args:
            qubit_pairs (list[tuple], optional): List of qubit pairs. If None,
                retrieves the pairs using "[self.get_qubits(pair) for pair in
                self.connectivity_graph()]". Defaults to None.
            cz_pulse_name (str, optional): Name of the CZ pulse.
                Takes self.default_cz_gate_name if not provided by the user.
            reference_qb (str, optional): Specifies whether to reference the 'high'
                or 'low' qubit. Defaults to 'high'.

        Returns:
            dict: A dictionary where keys are qubit pairs and values are their
                corresponding interaction frequencies.

        Raises:
            ValueError: If reference_qb is neither 'high' nor 'low'.
        """
        if qubit_pairs is None:
            qubit_pairs = [tuple(self.get_qubits(pair)) for pair
                           in self.connectivity_graph()]

        interaction_frequencies = {}
        for qubit_pair in qubit_pairs:
            qbi, qbj = qubit_pair
            interaction_freq = self.get_interaction_frequency(qbi, qbj,
                                                              cz_pulse_name,
                                                              reference_qb)
            interaction_frequencies[qubit_pair] = interaction_freq

        return interaction_frequencies

    def get_interaction_frequency(self, qbi, qbj, cz_pulse_name=None,
                                  reference_qb='high'):
        """
        Gets the interaction frequency of a CZ gate between two qubits.
        The interaction frequency is defined as the ef frequency of the high
        frequency qubit when reference_qb = 'high' and the ge frequency of
        the data qubit when reference_ab = 'low'.

        Args:
            qbi (QuDevTransmon, str): One of the qubits involved in the CZ gate.
            qbj (QuDevTransmon, str): The other qubit involved in the CZ gate.
            cz_pulse_name (str, optional): Name of the CZ pulse.
                Defaults to self.default_cz_gate_name.
            reference_qb (str, optional): Specifies whether to reference the 'high'
                or 'low' qubit. Defaults to 'high'.

        Returns:
            float: The calculated interaction frequency.

        Raises:
            AssertionError: If qbi is the same as qbj.
            ValueError: If reference_qb is neither 'high' nor 'low'.
        """
        assert qbi != qbj

        # ensure to get qubit objects
        qbi, qbj = self.get_qubits([qbi, qbj])

        if cz_pulse_name is None:
            cz_pulse_name = self.default_cz_gate_name()

        # Determine the higher and lower frequency qubits
        if qbi.ge_freq() > qbj.ge_freq():
            qbh, qbl = qbi, qbj
        else:
            qbh, qbl = qbj, qbi

        # Use the specified reference qubit to calculate interaction frequency
        if reference_qb == 'high':
            amp = self.get_pulse_par(cz_pulse_name, qbh.name, qbl.name,
                                     'amplitude')()
            f = qbh.calculate_frequency(amplitude=amp, flux=qbh.flux_parking(),
                                        transition='ef')
            # alternatively, we could use the following way of calculating the interaction
            # frequency, which assumes that the anharmonicity is independent of frequency.
            # it has the advantage to typically rely on data (ge-ef freq) and not the
            # Hamiltonian model.
            # f = qbh.calculate_frequency(amplitude=amp, flux=qbh.flux_parking())
            # f += qbh.anharmonicity()

        elif reference_qb == 'low':
            amp = self.get_pulse_par(cz_pulse_name, qbh.name, qbl.name,
                                     'amplitude2')()
            f = qbl.calculate_frequency(amplitude=amp, flux=qbl.flux_parking())
        else:
            raise ValueError(
                f'reference_qb should be "high" or "low" but was passed as'
                f' "{reference_qb}".')

        return f

    def set_interaction_frequency(self, qbi, qbj, int_freq, cz_pulse_name=None,
                                  update=True):
        """
        Sets the interaction frequency for a CZ gate between two qubits.

        Args:
            qbi (QuDevTransmon or str): One of the qubits involved in the CZ gate.
            qbj (QuDevTransmon or str): The other qubit involved in the CZ gate.
            int_freq (float): The desired interaction frequency for the CZ gate.
            cz_pulse_name (str, optional): Name of the CZ pulse.
                Defaults to self.default_cz_pulse_name.
            update (bool): If True, updates the pulse parameters. Defaults
                to True. The 'amplitude' parameter of the high-frequency qubit and
                the 'amplitude2' parameter of the low-frequency qubit are modified.

        Returns:
            Tuple[float, float]: The amplitudes corresponding to the high and low qubits.

        Raises:
            AssertionError: If qbi is the same as qbj.
        """
        assert qbi != qbj

        # ensure to get qubit objects
        qbi, qbj = self.get_qubits([qbi, qbj])

        # Determine the higher and lower frequency qubits
        if qbi.ge_freq() > qbj.ge_freq():
            qbh, qbl = qbi, qbj
        else:
            qbh, qbl = qbj, qbi

        # Calculate amplitudes based on interaction frequency and flux parking states
        amph = -np.abs(qbh.calculate_flux_voltage(
            int_freq - (qbh.ef_freq() - qbh.ge_freq()), flux=qbh.flux_parking()))
        ampl = np.abs(qbl.calculate_flux_voltage(int_freq, flux=qbl.flux_parking()))

        # Update pulse parameters if requested
        if update:
            if cz_pulse_name is None:
                cz_pulse_name = self.default_cz_gate_name()
            self.get_pulse_par(cz_pulse_name, qbh.name, qbl.name,
                               'amplitude')(amph)
            self.get_pulse_par(cz_pulse_name, qbh.name, qbl.name,
                               'amplitude2')(ampl)

        return amph, ampl

    def set_interaction_frequencies(self, interaction_frequencies,
                                    cz_pulse_name=None,
                                    update=True):
        """
        Sets interaction frequencies for CZ gates based on the provided
        dictionary.

        Args:
            interaction_frequencies (dict): A dictionary where keys are qubit pairs
                and values are their corresponding interaction frequencies.
            cz_pulse_name (str, optional): Name of the CZ pulse. Defaults to None.
            update (bool): If True, updates the pulse parameters.
                See docstring of Device.set_interaction_frequency.
                Defaults to True.
        """
        for qubit_pair, int_freq in interaction_frequencies.items():
            qbi, qbj = qubit_pair
            self.set_interaction_frequency(qbi, qbj, int_freq, cz_pulse_name,
                                           update)

    def set_pulse_par(self, gate_name, qb1, qb2, param, value):
        """
        Sets a value to a two qubit gate parameter.

        Args:
            gate_name (str): Name of the gate
            qb1 (str, QudevTransmon): Name of one qubit
            qb2 (str, QudevTransmon): Name of other qubit
            param (str): name of parameter
            value: value of parameter
        """

        if isinstance(qb1, qdt.QuDev_transmon):
            qb1_name = qb1.name
        else:
            qb1_name = qb1

        if isinstance(qb2, qdt.QuDev_transmon):
            qb2_name = qb2.name
        else:
            qb2_name = qb2

        try:
            self.set(f'{gate_name}_{qb1_name}_{qb2_name}_{param}', value)
        except KeyError:
            try:
                self.set(f'{gate_name}_{qb2_name}_{qb1_name}_{param}', value)
            except KeyError:
                raise ValueError(f'Parameter {param} for the gate '
                                 f'{gate_name} {qb1_name} {qb2_name} '
                                 f'does not exist!')

    def prepare_mwg(self):
        for MWG in self.MWGs:
            MWG.off()
        for TWPA in self.TWPAs:
            TWPA.on()
            pass

    def update_cancellation_params(self):
        for qbc in self.get_qubits():
            if qbc.ge_pulse_type() != 'SSB_DRAG_pulse_with_cancellation':
                continue
            cpars = qbc.ge_cancellation_params()
            for qb in self.get_qubits():
                iq = (qb.ge_I_channel(), qb.ge_Q_channel())
                if iq not in cpars:
                    continue
                cpars[iq]['mod_frequency'] = (qbc.ge_freq() - qb.ge_freq() +
                                            qb.ge_mod_freq())
                cpars[iq]['phi_skew'] = qb.ge_phi_skew()
                cpars[iq]['alpha'] = qb.ge_alpha()

    def set_default_acq_channels(self):
        qbs = self.get_qubits()
        feedlines = {(qb.instr_acq(), qb.acq_unit()) for qb in qbs}
        for fl in feedlines:
            qb_fl = [qb for qb in qbs
                    if qb.instr_acq() == fl[0]
                    and qb.acq_unit() == fl[1]]
            for i, qb in enumerate(qb_fl):
                qb.acq_I_channel(2 * i)
                qb.acq_Q_channel(2 * i + 1)

    def check_connection(self, qubit_a, qubit_b, connectivity_graph=None, raise_exception=True):
        """
        Checks whether two qubits are connected.

        Args:
            qubit_a (str, QudevTransmon): Name of one qubit
            qubit_b (str, QudevTransmon): Name of other qubit
            connectivity_graph: custom connectivity graph. If None device graph will be used.
            raise_exception (Bool): flag whether an error should be raised if qubits are not connected.
        """

        if connectivity_graph is None:
            connectivity_graph = self.connectivity_graph()
        # convert qubit object to name if necessary
        if not isinstance(qubit_a, str):
            qubit_a = qubit_a.name
        if not isinstance(qubit_b, str):
            qubit_b = qubit_b.name
        if [qubit_a, qubit_b] not in connectivity_graph and [qubit_b, qubit_a] not in connectivity_graph:
            if raise_exception:
                raise ValueError(f'Qubits {[qubit_a, qubit_b]}  are not connected!')
            else:
                log.warning('Qubits are not connected!')
                # TODO: implement what we want in case of swap (e.g. determine shortest path of swaps)

    def add_2qb_gate(self, gate_name, pulse_type='BufferedNZFLIPPulse'):
        """
        Method to add a two qubit gate with name gate_name with parameters for
        all connected qubits. The parameters including their default values are taken
        for the Class pulse_type in pulse_library.py.

        Args:
            gate_name (str): Name of gate
            pulse_type (str): Two qubit gate class from pulse_library.py
        """

        # add gate to list of two qubit gates
        self._two_qb_gates.append(gate_name)

        # find pulse module
        pulse_func = bpl.get_pulse_class(pulse_type)

        # for all connected qubits add the operation with name gate_name
        for [qb1, qb2] in self.connectivity_graph():
            op_name = (gate_name, qb1, qb2)
            par_name = f'{gate_name}_{qb1}_{qb2}'
            self.add_operation(op_name)

            # get default pulse params for the pulse type
            params = pulse_func.pulse_params()

            for param, init_val in params.items():
                self.add_pulse_parameter(op_name, par_name + '_' + param, param,
                                         initial_value=init_val)

            # needed for unresolved pulses but not attribute of pulse object
            if 'basis_rotation' not in params.keys():
                self.add_pulse_parameter(op_name, par_name + '_basis_rotation', 'basis_rotation', initial_value={})

            # Update flux pulse channels
            for qb, c in zip([qb1, qb2], ['channel', 'channel2']):
                if c in params:
                    channel = self.get_qb(qb).flux_pulse_channel()
                    if channel == '':
                        raise ValueError(f'No flux pulse channel defined for {qb}!')
                    else:
                        self.set_pulse_par(gate_name, qb1, qb2, c, channel)

        if self.default_cz_gate_name() is None:
            # Make the newly added gate the default
            self.default_cz_gate_name(gate_name)

    def _valid_cz_gate(self, gate_name):
        if gate_name is not None and gate_name not in self._two_qb_gates:
            raise ValueError(
                f'{gate_name} is not a valid two-qubit gate name. Valid '
                f'names are: {self._two_qb_gates}')
        return gate_name

    def get_channel_delays(self, qb_used=None):
        """
        Get AWG channel delays

        Args:
            qb_used (list of string): names of qubits whose delays should be
            set to the AWG channels (useful e.g. for shared AWG channels).
            If None, all qubits are used.

        Returns:
            Dictionary of delay values for the AWG channels of the system to
            correct for relative delays of the channels according to
            `self.relative_delay_graph()`.
        """
        object_delays = self.relative_delay_graph().get_absolute_delays()
        if qb_used is not None:
            object_delays = {
                (qbn, obj_type): delay
                for (qbn, obj_type), delay in object_delays.items()
                if qbn in qb_used
            }
        channel_delays = {}
        for (qbn, obj_type), v in object_delays.items():
            qb = self.get_qb(qbn)
            if obj_type == 'drive':
                ch_to_set = [qb.ge_I_channel(), qb.ge_Q_channel()]
            elif obj_type == 'flux':
                ch_to_set = [qb.flux_pulse_channel()]
            else:
                raise ValueError(f"Unrecognized channel type: {obj_type}!")
            for ch in ch_to_set:
                if ch in channel_delays and channel_delays[ch] != v:
                    log.warning(f"Delay of channel {ch} has conflicting "
                                f"values! This happens if several qubits "
                                f"share the same channel. Please pass qb_used "
                                f"to get_channel_delays in order to set "
                                f"AWG channel delays only for a subset of "
                                f"qubits measured in parallel.")
                if ch:  # If channel exists for this qubit
                    channel_delays[ch] = v
        return channel_delays

    def configure_pulsar(self, qb_used=None):
        """
        Configure pulse generation instrument settings.

        For now, only sets AWG channel delays.

        Args:
            qb_used: see get_channel_delays
        """

        pulsar = self.instr_pulsar.get_instr()

        # configure channel delays
        channel_delays = self.get_channel_delays(qb_used=qb_used)
        for ch, v in channel_delays.items():
            awg = pulsar.get_channel_awg(ch)
            chid = int(pulsar.get(f'{ch}_id')[2:]) - 1
            awg.set(f'sigouts_{chid}_delay', v)

    def configure_flux_crosstalk_cancellation(self, qubits='auto', rounds=-1):
        """
        Configure flux crosstalk cancellation in pulsar based on the
        calibrations stored in the qcodes parameter flux_crosstalk_calibs.
        For each stored calibration, a crosstalk cancellation matrix is
        generated and stored to pulsar's flux_crosstalk_cancellation_mtx dict.

        Note: if a single unnamed calibration is stored in the qcodes
        parameter flux_crosstalk_calibs (old format), the qcodes parameter
        is updated to the new format by storing the single calibration as
        the 'default' calibration.

        :param qubits: (str, list) the qubits for which the cancellation
            should be configure, see get_qubits for possible input formats.
            In addition, the option 'auto' is understood, in which case the
            function detects which qubits have been included in the
            characterization measurements (by checking which diagonal
            entries of the crosstalk matrix are not exactly equal to 1).
            Default: 'auto'
        :param rounds: (int, dict[str, int])
            the number of calibration rounds to be used as a basis for
            calculating the cancellation matrix, or -1 if all
            available round should be used. If the number of rounds should
            be different for the different calibration sets, a dictionary of
            values can be passed, where the keys are calibration keys.
            Default: -1
        """
        # convert values from old format
        if self.flux_crosstalk_calibs() is not None and \
                not isinstance(self.flux_crosstalk_calibs(), dict):
            self.flux_crosstalk_calibs(
                {'default': self.flux_crosstalk_calibs()}
            )
        if self.flux_crosstalk_calibs() is not None and \
                not isinstance(rounds, dict):
            rounds = {k: rounds for k in self.flux_crosstalk_calibs()}

        pulsar = self.instr_pulsar.get_instr()
        calibs = self.flux_crosstalk_calibs()
        if calibs is None:
            calibs = {}
            rounds = {}

        flux_channels = {}
        flux_crosstalk_cancellation_mtx = {}
        for calibration_key, calib in calibs.items():
            calib = deepcopy(calib)
            if calib is None:
                calib = [np.identity(len(self.get_qubits()))]
            rounds_calib = rounds[calibration_key]
            if rounds_calib == -1:
                rounds_calib = len(calib)
            xtalk_qbs = self.get_qubits('all' if qubits == 'auto' else qubits)
            if qubits == 'auto':
                mask = [True] * len(xtalk_qbs)
                for mtx in calib[:rounds_calib]:
                    mask = np.logical_and(mask, np.diag(mtx) != 1)
                xtalk_qbs = [qb for i, qb in enumerate(xtalk_qbs) if mask[i]]
            if len(xtalk_qbs) == 0:
                continue
                # pulsar.flux_crosstalk_cancellation(False)
                # return

            for i in range(rounds_calib):
                calib[i] = np.diag(1 / np.diag(calib[i])) @ calib[i]
            mtx_all = functools.reduce(np.dot, calib[:rounds_calib])
            qb_inds = {qb: ind for qb, ind in
                       zip(xtalk_qbs, self.get_qubits(xtalk_qbs, 'ind'))}
            mtx = np.zeros([len(xtalk_qbs)] * 2)
            for i, qbA in enumerate(xtalk_qbs):
                for j, qbB in enumerate(xtalk_qbs):
                    mtx[i, j] = mtx_all[qb_inds[qbA], qb_inds[qbB]]

            flux_channels[calibration_key] = \
                [qb.flux_pulse_channel() for qb in xtalk_qbs]
            flux_crosstalk_cancellation_mtx[calibration_key] = \
                np.linalg.inv(mtx)

        pulsar.flux_channels(flux_channels)
        pulsar.flux_crosstalk_cancellation_mtx(flux_crosstalk_cancellation_mtx)

    def load_crosstalk_measurements(self, timestamps, round_ind=0,
                                    extract_only=True, options_dict=None,
                                    calibration_key='default'):
        """
        Load results of flux crosstalk calibration measurements and store
        them in the qcodes parameter flux_crosstalk_calibs.

        :param timestamps: (list of str) timestamps of the measurements to
            be loaded.
        :param round_ind: (int) the index of the calibration round to which
            the measurement belong. Default: 0
        :param extract_only: (bool) do not create figures while analyzing the
            the given timestamps (to reduce processing time, useful if
            figures have already been created before). Default: True
        :param options_dict: (dict or None) options_dict to be passed to
            FluxlineCrosstalkAnalysis. Default: None
        :param calibration_key: (str) a name to identify the loaded
            calibration. Default: 'default'
        """
        if options_dict is None:
            options_dict = {}
        if 'TwoD' not in options_dict:
            options_dict['TwoD'] = True
        all_qubits = self.get_qubits()
        calibs = self.flux_crosstalk_calibs()
        if calibs is None:
            self.flux_crosstalk_calibs({})
            calibs = self.flux_crosstalk_calibs()
        calib = calibs.get(calibration_key, None)
        if calib is None:
            calibs[calibration_key] = []
            calib = calibs[calibration_key]
        while len(calib) <= round_ind:
            calib.append(np.identity(len(all_qubits)))

        for ts in timestamps:
            target_qubit_name, crosstalk_qubit_names = tda.a_tools.get_folder(
                ts).split('_')[-2:]
            crosstalk_qubit_names = ['qb' + i for i in
                                     crosstalk_qubit_names.split('qb')[1:]]
            MA = tda.FluxlineCrosstalkAnalysis(t_start=ts,
                                               qb_names=crosstalk_qubit_names,
                                               extract_only=extract_only,
                                               options_dict=options_dict)
            for qbn in crosstalk_qubit_names:
                i = self.get_qubits(qbn, 'ind')[0]
                j = self.get_qubits(target_qubit_name, 'ind')[0]
                dphi_dV = MA.fit_res[f'flux_fit_{qbn}'].best_values['a']
                calib[round_ind][i][j] = dphi_dV
                print(i, j, dphi_dV)

    def plot_flux_crosstalk_matrix(self, qubits='all', round_ind=0, unit='m',
                                   vmax=None, show_and_close=False,
                                   calibration_key='default'):
        """
        Visualize stored flux crosstalk calibration measurements as a matrix.

        :param qubits: (str, list) the qubits to be included, see get_qubits
            for possible input formats. Default: 'all'
        :param round_ind: (int) the index of the calibration round to be
            shown. Default: 0
        :param unit: (str) unit of the flux coupling, where  allowed values
            are '' (Phi_0/V), 'c' (centi Phi_0/V), 'm' (milli Phi_0/V),
            'u' (micro Phi_0/V). Default: 'm'
        :param vmax: (float) maximal value of the colormap of the flux
            coupling. Default: 1 if unit is '', 4 if unit is 'c' or 'm',
            100 if unit is 'u'.
        :param show_and_close: (bool) whether the figure should be shown and
            closed (True) or whether the figure handle should be returned
            (False). Default: False
        :param calibration_key: (str) an identifier of the stored
            calibration that should be plotted. Default: 'default'
        """
        qubits = self.get_qubits(qubits)
        qb_inds = self.get_qubits(qubits, return_type='ind')
        if unit not in ['', 'c', 'm', 'u', None]:
            log.warning(f'unit prefix "{unit}" not understood. Using "m".')
        if unit == 'u':
            phi_factor, phi_unit, diag_prefix = 1e-6, '\\mu\\Phi_0', 'M'
            def_vmax = 100
        elif unit == 'm':
            phi_factor, phi_unit, diag_prefix = 1e-3, 'm\\Phi_0', 'k'
            def_vmax = 4
        elif unit == 'c':
            phi_factor, phi_unit, diag_prefix = 1e-2, 'c\\Phi_0', 'h'
            def_vmax = 4
        else:
            phi_factor, phi_unit, diag_prefix = 1, '\\Phi_0', ''
            def_vmax =  1
        vmax = vmax / phi_factor if vmax is not None else def_vmax

        calibs = self.flux_crosstalk_calibs()
        if not isinstance(calibs, dict):
            calibs = {'default': calibs}
        data_array = calibs[calibration_key][round_ind][
                     qb_inds, :][:, qb_inds]
        for i in range(len(qubits)):
            data_array[i, :] *= np.sign(data_array[i, i])

        fig, ax = plt.subplots()
        fig.set_size_inches(
            2 + (plot_mod.FIGURE_WIDTH_2COL - 2) / 17 * len(qubits),
            1 + 6 / 17 * len(qubits))

        cmap = copy(mpl.cm.RdBu)
        cmap.set_over('k')
        i = ax.imshow(data_array / phi_factor, vmax=vmax, vmin=-vmax,
                      cmap=cmap)
        cbar = fig.colorbar(i)

        ax.set_xticks(np.arange(len(qubits)))
        ax.set_yticks(np.arange(len(qubits)))
        ax.set_xticklabels(['FL' + qb.name[2:] for qb in qubits])
        ax.set_yticklabels(['Q' + qb.name[2:] for qb in qubits])
        ax.set_xlabel('Fluxline')
        ax.set_ylabel('Coupled qubit')
        ax.tick_params(direction='out')
        cbar.set_label(
            f'Flux coupling, $\\mathrm{{d}}\Phi/\\mathrm{{d}}V$ '
            f'($\\mathrm{{{phi_unit}}}$/V)')

        for i in range(len(qubits)):
            for j in range(len(qubits)):
                if i != j:
                    ax.text(i, j, f'{data_array[j, i] / phi_factor:.2f}',
                            color='k', fontsize='small', ha='center',
                            va='center')
                else:
                    ax.text(i, j,
                            f'{data_array[j, i]:.2f}{diag_prefix}',
                            color='w', fontsize='small', ha='center',
                            va='center')

        fig.subplots_adjust(left=0.04, right=1.04)
        if show_and_close:
            plt.show()
            plt.close(fig)
            return
        else:
            return fig

    def __getattr__(self, item):
        """Attribute getter function

        This function is overloaded such that if an attribute is not found in
        the device it is fetched from self.qubits instead.

        Args:
            item: name of the attribute to fetch

        Returns:
            func: function equivalent to a qcodes parameter for all qubits,
            see docstring
        """
        try:
            # Normal behaviour (the requested attribute exists)
            return super().__getattr__(item)
        except AttributeError as e:
            # If the requested attribute does not exist, try to fetch
            # it instead from all the qubits
            # Example:
            # dev.ge_freq() ---> Returns {'qb1': 6.02e9, ...}
            # dev.ge_freq(5.0e9) ---> Sets the ge_freq of all qubits
            qbs_with_attr = [qb for qb in self.qubits if hasattr(qb, item)]
            if qbs_with_attr:
                def func(p=None, common_value_all_qubits=False):
                    """Effective qcodes parameter acting on several qubits

                    Args:
                        p: Value to set to the qubits. If None, the function
                            acts as a getter instead.
                            p can be formatted in two ways:
                            - case 1: a value v to set to the qubits
                            - case 2: a dict of values to set to each qubit,
                                e.g. {'qb1': v1, ...}
                        common_value_all_qubits: If p is a dict, indicates that
                            this whole dict should be set to each qubit. In
                            other words, it should be recognized as case 1
                            and not case 2.
                    """
                    if p is None:
                        # No value passed: getter
                        return {qb.name: qb.__getattr__(item)() for qb in
                                qbs_with_attr}
                    elif isinstance(p, dict) and not common_value_all_qubits:
                        # Parse p to set p[qbn] to each qubit
                        [qb.__getattr__(item)(p[qb.name])
                         for qb in qbs_with_attr if qb.name in p]
                    else:
                        # Directly set p to each qubit
                        # (p can be a value, or a dict if whole_dict)
                        [qb.__getattr__(item)(p)
                            for qb in qbs_with_attr]
                return func
            else:
                # If the attribute does not exist in the qubits either
                raise e


class RelativeDelayGraph:
    """
    Contains the information about relative delays of channels of the device.

    The relative delays are represented by a graph, where each channel is a
    node, and edges represent the relative delays that we can measure.

    Internally this is stored in a dictionary of the form:
        _d = {
            obj1: {
                obj2: delay_obj1_obj2,
                obj3: delay_obj1_obj3,
            }
        }

    The relative delay is defined as the delay of the parent channel minus
    delay of child channel.

    Args:
        reld: dict or RelativeDelayGraph, optional
            Initial value for the delay graph either in the internal
            representation or as another RelativeDelayGraph.
    """

    def __init__(self, reld=None):
        if isinstance(reld, RelativeDelayGraph):
            self._reld = deepcopy(reld._reld)
        else:
            if reld is None:
                self._reld = {}
            else:
                self._reld = deepcopy(reld)

    def increment_relative_delay(self, parent, child, delta_delay):
        """Increment a relative delay between two nodes.

        Use this to update delays based on measurement results with previously
        set delay values.

        Args:
            parent, child:
                node names
            delta_delay: float
                Measured delay difference between the parent and the
                child node that will be added to the graph.
        """
        if child in self._reld[parent]:
            self._reld[parent][child] += delta_delay
        elif parent in self._reld[child]:
            self._reld[child][parent] -= delta_delay
        else:
            raise KeyError(f'No direct connection between {parent} and {child}'
                           ' in relative delay graph')

    def get_relative_delay(self, parent, child):
        """Get the raw relative delay between two directly connected nodes.

        Args:
            parent, child:
                Node names
        Returns:
            relative delay between parent and child
        """
        if child in self._reld[parent]:
            return self._reld[parent][child]
        elif parent in self._reld[child]:
            return -self._reld[child][parent]
        else:
            raise KeyError(f'No direct connection between {parent} and {child}'
                           f' in relative delay graph')

    def set_relative_delay(self, parent, child, delay):
        """Set the raw value of the relative channel delay.

        Use with care, since it resets the previously configured delay between
        the nodes.

        Args:
            parent, child:
                Node names
            delay: float
                Raw delay delay difference between parent and child node.
        """
        if child in self._reld[parent]:
            self._reld[parent][child] = delay
        elif parent in self._reld[child]:
            self._reld[child][parent] = -delay
        else:
            raise KeyError(f'No connection between {parent} and {child} in '
                            'relative delay graph')

    def get_absolute_delays(self):
        """
        Get an abs. delay for each node satisfying configured relative delays.

        Returns: dict
            Keys are node names, values are nonnegative delays for each node
            such that the relative delays are satisfied.
        """
        # determine root of the tree
        abs_delays = OrderedDict()
        min_delay = np.inf
        refs = set()
        children = set()
        for ref in self._reld:
            refs.add(ref)
            for child in self._reld[ref]:
                refs.add(child)
                children.add(child)
        roots = refs - children
        for ref in roots:
            abs_delays[ref] = 0

        while len(roots) > 0:
            ref = list(roots)[0]
            for child in self._reld.get(ref, []):
                abs_delays[child] = abs_delays[ref] - self._reld[ref][child]
                min_delay = min(min_delay, abs_delays[child])
                roots.add(child)
            roots.remove(ref)

        for ref in abs_delays:
            abs_delays[ref] -= min_delay

        return abs_delays

    def __repr__(self):
        return self._reld.__repr__()
