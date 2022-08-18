import time
import serial
import numpy as np
from typing import Union, Tuple, List
from copy import copy, deepcopy
from collections import OrderedDict

import qcodes as qc
from pycqed.instrument_drivers.instrument import Instrument


class ArduinoSwitchControl(Instrument):
    """Class for the RT switch box that uses an Arduino

    This class represents the room-temperature switch box as a Qcodes
    instrument. The representation of the box is constructed from the inputs
    and outputs of the box, the switches, and the connections between the
    switches and to the inputs and outputs. The possible routes between
    the inputs and outputs are automatically generated. Upon initialization,
    the configuration must be given as a dictionary to the 'config' argument
    of the init function. For specifying the switches and connections,
    see Args.

    Within the configuration dictionary, the input and output connectors can
    be initialized in the following ways:
        - As an integer for the number of connectors.
          The inputs are labeled with 'I1', 'I2',..., the outputs are
          with 'O1', 'O2',....
        - A tuple ('X',N), where 'X' is the label and N the number of
          connectors. The connectors are labeled with X1, X2,....
        - As a list to group the connectors. The labeling format
          is 'X.n', where X is the group label and n the connector
          number in the group. The entries of the list can be:
            - A tuple ('X',N) where 'X' is the label of the group and
              N is the number of connectors in the group.
            - An integer N. Groups without a given label will be
              labelled with 'I1', 'I2',... for input groups and
              'O1','O2',... for output groups

    The Qcodes parameters representing switching individual switches follow
    the naming convention 'switch_X_mode' where 'X' is the label of the
    switch (int are converted to str). One can set the switch X with
        self.switch_X_mode(state)
    and read the state with
        self.switch_X_mode()
    Additionally, there exist Qcodes parameters representing the routes from
    the inputs, following the naming convention 'route_I_mode', where 'I'
    is the label of the input. One can set the route from input I to
    output O with
        self.route_I_mode(O)
    and check to which output the input I is connected with
        self.route_I_mode()
    If the input is not connected to any output, None is returned.
    Note: Setting and getting the routes accesses multiple switches and
    therefore takes up to a few seconds.
    For the protocol that is used to communicate with the Arduino to set and
    read the switches, check the documentation of the self._set_switch and
    self._read_switch methods.

    As required for switches in PycQED, there exists a method
        self.set_switch(values)
    where 'values' is a dictionary {switch : state} and switch can be
        - the label of a switch
        - equivalently a string 'switch_X' where X is the label of the switch
        - the label of an input. 'state' is then the label of the output
          to connect to
        - equivalently a string 'route_I' where I is the label of an input.
    Additionally, there exists a method
        self.get_switch(*args)
    where *args contains the labels of switches or inputs (or strings
    'switch_X' and 'route_I' as above) and it returns a dictionary compatible
    with the argument 'values' of the self.set_switch method. This method is
    not standard in PycQED.

    The serial communication is handled globally with class methods (see
    below at the class methods.) For the serial communication of a certain
    instance, the methods self.start_serial, self.end_serial and
    self.assure_serial are used.

    The switches in the switch box are controlled by an Arduino and a printed
    circuit board (PCB). The Arduino is connected to the PC via USB and to the
    PCB via two signal wires for I2C-communication and a ground wire. On the
    PCB, cls.NUM_IO = 5 input/output-expanders, which can be addressed with I2C, can
    control cls.NUM_IO_SWITCHES = 4 switches each, giving a maximum amount of
    cls.MAX_SWITCHES = 20 switches. The Arduino selects the switches by the number
    of the I/O-expander (0,1,2,3,4) (here, usually referred to as group_id) and the
    number of the switch for this I/O-expander (0,1,2,3) (here usually switch_id).
    Using self.switch_ids[(group_id,switch_id)] returns the instance of
    ArduinoSwitchControlSwitch that represents the switch with this id.

    Args:
        name (str): Name (for the initialization as a Qcodes Instrument
        port (str):
            Name of the serial port
            (on Windows usually starts with 'COM')
        config (dict):
            Dictionary for the configuration of the switch box.
            Must include:
                'switches': Specify the switches. The PCB in the switch box
                    allows up to cls.MAX_SWITCHES = 20 switches. Can be:
                    - int <= MAX_SWITCHES: Specify the number of switches.
                                  The switches are then labeled with integers
                                  1,...,num_switches
                    - list(like): Specify labels for the switches in a list
                                  with length <= MAX_SWITCHES
                'inputs': Specify the input connectors.
                          For format of specifying connectors, see above
                'outputs': Specify the output connectors.
                           For format of specifying connectors, see above
                'connections': A list with the connections between the
                               switches and to the inputs and outputs of
                               the box. The elements of the list are tuples
                               (start,end), where 'start' and 'end' must be
                               one of the following:
                               - For an input or output of the switch box,
                                 use the label of the input or output.
                                 e. g. use 'I1' for input I1
                               - For the input of a switch, use the label of
                                 the switch.
                                 e. g. use 1 for switch 1
                               - For the output of a switch, use a tuple
                                 (switch-label, state), where 'switch-label'
                                 is the label of the switch, and 'state' is the
                                 state of the switch, that corresponds to the
                                 output.
                                 e. g. for the output of switch 2, that is
                                 active, when the switch is in state 0, use
                                 the tuple (2,0)
        start_serial (bool):
            Whether to start the serial communication with the Arduino
            upon initialization.

    Attributes:
        port (str): Name of the serial port
        switches (OrderedDict):
            Dictionary {'switch label':switch} of the switch labels as keys
            and instances of ArduinoSwitchControlSwitch as values,
            representing the switches.
        inputs (OrderedDict):
            Dictionary {'input label':input} of the input labels as keys
            and instances of ArduinoSwitchControlConnectors as values,
            representing the inputs.
        input_groups (list): List of the labels for the input groups
        outputs (OrderedDict):
            Dictionary {'output label':output} of the output labels as keys
            and instances of ArduinoSwitchControlConnectors as values,
            representing the outputs.
        output_groups (list): List of the labels for the output groups
        connections (list):
            List of the connections, represented as instances of
            ArduinoSwitchControlConnection.
        routes (OrderedDict):
            Dictionary of dictionaries
            {'input label': {'output label':routes}}
            where 'routes' is a list of possible routes between the input
            and the output, represented as instances of
            ArduinoSwitchControlRoute.
        serial (serial.Serial): serial port to connect to the Arduino

    Class Attributes:
        WRITE_DELAY (float): Waiting period after setting a switch.
                       Default: 0.15 s
        READ_DELAY (float): Waiting period after reading the switch.
                             Default: 0.05 s

    """

    def __init__(self, name, port, config, start_serial=True):
        self._check_if_in_config(config,
                                 'switches', 'inputs', 'outputs',
                                 'connections')

        super().__init__(name)

        self.port = port

        # Switch properties
        # -----------------

        switches = config['switches']
        if isinstance(switches, int):
            # If switches is int, use as number of switches
            # and switch_labels = [1,2,...,switches]
            if not 0 <= switches <= self.MAX_SWITCHES:
                raise ValueError("Number of switches 'switches' must be"
                                 f"between 0 and {self.MAX_SWITCHES}, not {switches}.")
            self._num_switches = switches
            self.switch_labels = [n for n in range(1, self.num_switches + 1)]
        else:
            # If switches is iterable, use entries as switch labels and length
            # as number of switches
            try:
                switch_labels = list(switches)
            except TypeError:
                raise TypeError("Type of 'switches' must be a positive"
                                "integer for the number of switches, or a"
                                "list for the switch labels. The maximum"
                                f"number of switches is {self.MAX_SWITCHES}.")
            num_switches = len(switch_labels)
            if num_switches > self.MAX_SWITCHES:
                num_switches = self.MAX_SWITCHES
                switch_labels = switch_labels[:self.MAX_SWITCHES]
            self._num_switches = num_switches
            self.switch_labels = switch_labels

        # switch ids (group_id, switch_id) 0n PCB
        switch_ids_list = [(n, m) for n in range(self.NUM_IO)
                           for m in range(self.NUM_IO_SWITCHES)]

        # create switches
        self.switch_ids = OrderedDict()
        self.switches = OrderedDict()
        for id, label in zip(switch_ids_list, self.switch_labels):
            switch = self._add_switch(label, id, return_switch=True)
            # create Qcodes parameter
            param_name = f'switch_{label}_mode'
            self.add_parameter(
                param_name,
                label=f'{param_name} of {self.name}',
                vals=qc.validators.Enum(0, 1),
                get_cmd=lambda s=label: self._get_switch(s),
                set_cmd=lambda x, s=label: self._set_switch(s, x),
                docstring="possible values: 0, 1",
            )
            # add parameter as attribute to switch
            switch.mode = self.parameters[param_name]

        # Inputs and outputs
        # ------------------
        inputs = config['inputs']
        self.inputs, self.input_groups = self._create_connectors(
            inputs,
            default_label='I',
            return_groups=True,
            connector_type='input',
            return_labels=False
        )

        outputs = config['outputs']
        self.outputs, self.output_groups = self._create_connectors(
            outputs,
            default_label='O',
            return_groups=True,
            connector_type='output',
            return_labels=False
        )

        self._num_inputs = len(self.inputs)
        self._num_outputs = len(self.outputs)

        # Connections
        # -----------
        connections = config['connections']
        self.connections = []
        self.routes = OrderedDict()
        # find routes from inputs to outputs
        self._process_connections(connections)
        # add routes as Qcodes parameters
        for inp in self.routes:
            param_name = f'route_{inp}_mode'
            self.add_parameter(
                param_name,
                label=f'{param_name} of {self.name}',
                vals=qc.validators.Enum(*self.routes[inp].keys()),
                get_cmd=lambda i=inp: self._get_route(i),
                set_cmd=lambda o, i=inp: self._set_route(i, o),
                docstring=f"possible values: {list(self.routes[inp].keys())}",
            )

        # Start serial communication
        # --------------------------
        self.serial = None
        if start_serial:
            self.start_serial()

    # Magic methods
    # -------------

    def __del__(self):
        # end serial communication before deleting
        self.end_serial()

    # Class constants
    # ---------------

    NUM_IO = 5
    NUM_IO_SWITCHES = 4
    MAX_SWITCHES = NUM_IO*NUM_IO_SWITCHES
    READ_DELAY = 0.05  # after reading a switch
    WRITE_DELAY = 0.15  # after setting a switch

    # Properties
    # ----------
    @property
    def num_switches(self):
        """Number of switches"""
        return self._num_switches

    @property
    def num_inputs(self):
        """Number of inputs"""
        return self._num_inputs

    @property
    def num_outputs(self):
        """Number of outputs"""
        return self._num_outputs

    # Instance methods
    # ----------------

    # - Methods for reading and setting the switches
    #   --------------------------------------------

    def set_switch(self, values):
        """Set multiple switches by values.

        Args:
            values (dict):
                Dictionary of the switch modes to set.
                For switching single switches: {switch label : state}
                For routes starting from an input: {input label : output label}
                Instead of the switch label and input label, the parameter
                can be used (without the _mode),
                e. g. switch 2 can be addressed with 2 or 'switch_2',
                the route starting at input 'I1' can be addressed with 'I1' or
                'route_I1'.
                Read the documentation of this class for more details on
                the Qcodes parameters of the class.
        """
        for label, val in values.items():
            # if label is not a label, get the according label
            if isinstance(label, (ArduinoSwitchControlSwitch,
                                  ArduinoSwitchControlConnector)):
                label = label.label
            # if switch label, get the parameter to switch the switch
            if label in self.switches:
                par = self.parameters[f'switch_{label}_mode']
            # if input label, get the parameter to set a route from this input
            elif label in self.inputs:
                par = self.parameters[f'route_{label}_mode']
            # if parameter name, get the right parameter
            elif label.startswith('switch_'):
                if label[7:] not in [str(lab) for lab in self.switches]:
                    raise SwitchError(f"No switch with label {label[7:]}")
                par = self.parameters[f'{label}_mode']
            elif label.startswith('route_'):
                if label[6:] not in [str(lab) for lab in self.inputs]:
                    raise ConnectorError(f"No input with label {label[6:]}")
                if f'{label}_mode' not in self.parameters:
                    raise RouteError(f"No route starting at input {label[6:]}")
                par = self.parameters[f'{label}_mode']
            else:
                raise ValueError(f"parameter label {label} not recognized.")

            # apply selected parameter for switching
            par(val)

    def get_switch(self, *labels):
        """Get the states of multiple switches

        Args:
            *labels: Can be:
                - The label of a switch to get the state of the switch
                - The label of an input, to get the output it is connected to
                - The according parameters (without _mode)
                  e. g. switch 2 can be addressed with 2 or 'switch_2',
                  the route starting at input 'I1' can be addressed with 'I1' or
                  'route_I1'.
                  Read the documentation of this class for more details on
                  the Qcodes parameters of the class.

        Returns:
            dict:
                The results of reading the switches in a dictionary
                compatible with self.set_switch()
        """
        if len(labels) == 1 and not isinstance(labels[0], str):
            try:
                labels = list(labels[0])
            except TypeError:
                pass
        results = {}
        for label in labels:
            if isinstance(label, (ArduinoSwitchControlSwitch,
                                  ArduinoSwitchControlConnector)):
                label = label.label
            if label in self.switches:
                par = self.parameters[f'switch_{label}_mode']
            elif label in self.inputs:
                par = self.parameters[f'route_{label}_mode']
            elif label.startswith('switch_'):
                if label[7:] not in [str(lab) for lab in self.switches]:
                    raise SwitchError(f"No switch with label {label[7:]}")
                par = self.parameters[f'{label}_mode']
            elif label.startswith('route_'):
                if label[6:] not in [str(lab) for lab in self.inputs]:
                    raise ConnectorError(f"No input with label {label[6:]}")
                if f'{label}_mode' not in self.parameters:
                    raise RouteError(f"No route starting at input {label[6:]}")
                par = self.parameters[f'{label}_mode']
            else:
                raise Exception(f"parameter label {label} not recognized.")

            results[label] = par()
        return results

    # - Methods for serial communication
    #   --------------------------------

    def start_serial(self, override=True):
        """Start serial communication

        Args:
            override (bool): whether an existing serial communication should be
                      closed and replaced. Recommended: True
        """
        ser = self.get_serial(self.port)

        if ser is None or override:
            if ser is not None and ser.is_open:
                ser.close()  # close open serial
            ser = serial.Serial(self.port, timeout=1)  # reate new serial
            self.add_port(self.port, ser)  # add serial to _open_ports
            self.serial = ser  # save serial

            # handle output during setup
            setup_string = self.serial.readline()
            setup_string = setup_string.decode().rstrip()
            if setup_string != 's':  # setup started returns 's'
                print("Setup did not start.")
            else:
                # end of setup returns l
                # failed communication with IO-expander returns group id
                error_string = ser.readline()
                error_string = error_string.decode().rstrip()
                if error_string != 'l':
                    print(f'\nSetup failed for IO-Expander {error_string}')

        # save found serial. Serial might be closed, might have to be opened
        # manually.
        self.serial = ser

    def end_serial(self):
        """Closes serial communication of self.serial

        """
        if self.serial is None:
            ser = self.get_serial(self.port)
            if ser is not None and ser.is_open:
                ser.close()
        elif self.serial.is_open:
            self.serial.close()

    def assure_serial(self):
        """Opens a new serial communication if there is none

        """
        if self.serial is None or not self.serial.is_open:
            self.start_serial()

    # - Method for opening the GUI
    #   --------------------------

    def open_gui(self):
        raise NotImplementedError("The GUI has yet to be implemented. "
                                  "Coming soon!")

    # - Helper functions for initializing and running the box
    #   -----------------------------------------------------

    def switch_by_label(self, label):
        """Return the switch with label 'label'.

        If label is already a switch, it just returns the switch

        Args:
            label: Can be:
                   int or str: label of the switch
                   ArduinoSwitchControlSwitch: the switch itself

        Returns:
            ArduinoSwitchControlSwitch: the switch
        """
        if isinstance(label, ArduinoSwitchControlSwitch):
            return label
        elif label in self.switches:
            return self.switches[label]
        else:
            raise SwitchError(f"No switch with label '{label}' found.")

    def connector_by_label(self, label):
        """Return the connector with label 'label'.

        If label is already a connector, it just returns the connector

        Args:
            label: Can be:
                   - The label of an input or output of the box
                   - The label of a switch for the input of the switch
                   - A tuple (switch, state) for the corresponding output of
                     the switch

        Returns:
            ArduinoSwitchControlConnector: the connector
        """
        # return label, if it is already a connector
        if isinstance(label, ArduinoSwitchControlConnector):
            return label
        # if the label is an input, return the input
        elif label in self.inputs:
            return self.inputs[label]
        # if the label is an output, return the output
        elif label in self.outputs:
            return self.outputs[label]
        # if the label is an switch label, return the input of the switch
        elif label in self.switches:
            return self.switches[label].input
        # handle the tuple (switch,state) for an output of the switch
        else:
            try:
                label = tuple(label)
            except TypeError:
                raise ConnectorError(f"No connector {label} found.")
            if label[0] not in self.switches or (
                    label[1] != 0 and label[1] != 1):
                raise ConnectorError(f"No connector {label} found.")
            return self.switches[label[0]].output[int(label[1])]

    def _add_switch(self, label, id, orientation=0, return_switch=True):
        """Creates a switch and adds it to the relevant dictionaries.

        Args:
            label (int or str): label of the switch
            id (tuple): id of the switch on the PCB,
                        of form (group_id,switch_id)
            orientation: orientation of the switch in the routes between the
                         inputs and outputs. Can be:
                             1: the input of the switch points in the
                                direction of the inputs of the box.
                             -1: the input of the switch points in the
                                 direction of the outputs of the box.
                             0: the orientation of the switch is undetermined.
            return_switch: Whether the switch is returned by the method

        Returns:
            ArduinoSwitchControlSwitch: The created switch.
                If return_switch == False, there are no returns.

        """
        switch = ArduinoSwitchControlSwitch(label, id,
                                            orientation=orientation)
        self.switch_ids[id] = switch
        self.switches[label] = switch

        if return_switch:
            return switch

    def _add_connection(self, con):
        """Creates a connection.

        Connections can be created between two connectors.

        Args:
            con (tuple):
                A tuple (start,end), where start and end can be:
                - An instance of ArduinoSwitchControlConnector
                - The label of an input or output of the box
                - The label of a switch for the input of the switch
                - A tuple (switch, state) for the output of the switch
        """
        # get connectors by the above specified labels
        start = self.connector_by_label(con[0])
        end = self.connector_by_label(con[1])
        if start.parent_type == 'box' and end.parent_type == 'box':
            # make sure, that not two inputs or two outputs are connected
            if start.connector_type == end.connector_type:
                raise ConnectorError(f"Connection {con} connects "
                                     f"input to input or output to output.")
            # make sure, that inputs are always first
            # and outputs are always second
            elif (start.connector_type == 'output'
                  or end.connector_type == 'input'):
                start, end = end, start
        # make sure, that a switch does not connect to itself
        elif start.parent_type == 'switch' and end.parent_type == 'switch':
            if start.switch == end.switch:
                raise ConnectorError(f"Connection {con} connects "
                                     f"a switch to itself.")

        # create connection
        connection = ArduinoSwitchControlConnection(start, end)

        # add connection to attributes
        self.connections.append(connection)

    def _add_route(self, connections):
        """Create a route and add it to the routes dictionary

        Args:
            connections (list):
                A list of ArduinoSwitchControlConnection objects, that
                specify the route from an input to an output of the box.
        """
        route = ArduinoSwitchControlRoute(connections)
        if route.input.label not in self.routes:
            self.routes[route.input.label] = {route.output.label: [route]}
        elif route.output.label not in self.routes[route.input.label]:
            self.routes[route.input.label][route.output.label] = [route]
        else:
            self.routes[route.input.label][route.output.label].append(route)

    def _create_connectors(self, connectors, connector_type, default_label='C',
                           return_groups=False, in_group=False,
                           return_labels=False):
        """Creates connectors from the configuration (see class documentation)

        Args:
            connectors: Configuration of the connectors. Can be:
                - int: Number of connectors. The connectors will be labeled
                       {default_label}{number} (e. g. C1, C2, ...)
                - tuple: Specify the label and number of connectors with
                         (label,number of connectors). The labels will be
                         {label}{number} (e. g. for label=A: A1, A2, ...)
                - list of the tuples above, to create groups. The group label
                  and the connector number are separated with a dot.
                  e. g. [(A,2),(B,3)] results in A.1, A.2, B.1, B.2, B.3
            connector_type (str): Has to be 'input' or 'output'.
            default_label: Label that is used as a default.
            return_groups (bool): Whether the labels of the groups should be
                                  returned as well.
            in_group (bool): Whether this method is called to for an
                             individual group. Needed to handle the recursion
                             of the method.
            return_labels: Whether the labels of the connectors should be
                           returned a swell.

        Returns:
            OrderedDict: Dictionary with the connectors.
                if return_groups, the group labels also get returned
                if return_labels, the connector labels also get returned

        """
        if isinstance(connectors, int):
            if connectors < 0:
                raise ValueError("Number of connectors 'connectors' must"
                                 "be positive.")
            if in_group:
                label_string = default_label + '.'
                group = default_label
            else:
                label_string = default_label
                group = None
            labels = [label_string + str(n) for n in range(1, connectors + 1)]

            connectors_dict = OrderedDict([
                (label, ArduinoSwitchControlConnector(
                    label, 'box', connector_type, group=group
                )) for label in labels])
            if return_groups and return_labels:
                return connectors_dict, [default_label], labels
            elif return_groups:
                return connectors_dict, [default_label]
            elif return_labels:
                return connectors_dict, labels
            else:
                return connectors_dict
        else:
            try:
                connectors = list(connectors)
            except TypeError:
                raise TypeError("'connectors' has wrong format."
                                "Check documentation.")
            if (len(connectors) == 2 and isinstance(connectors[0], str)
                    and isinstance(connectors[1], int)):
                connectors_dict, labels = self._create_connectors(
                    connectors[1],
                    connector_type=connector_type,
                    default_label=connectors[0],
                    return_groups=False,
                    in_group=in_group,
                    return_labels=True
                )
                if return_groups and return_labels:
                    return connectors_dict, [connectors[0]], labels
                elif return_groups:
                    return connectors_dict, [connectors[0]]
                elif return_labels:
                    return connectors_dict, labels
                else:
                    return connectors_dict
            else:
                connectors_dict = OrderedDict()
                groups = []
                labels = []
                for n, group in enumerate(connectors):
                    if isinstance(group, int):
                        group = (f'{default_label}{n + 1}', group)
                    else:
                        group = (group[0], group[1])
                    groups.append(group[0])
                    con_dict, labs = self._create_connectors(
                        group, connector_type=connector_type,
                        return_groups=False, in_group=True,
                        return_labels=True
                    )
                    connectors_dict.update(con_dict)
                    labels += labs
                if return_groups and return_labels:
                    return connectors_dict, groups, labels
                elif return_groups:
                    return connectors_dict, groups
                elif return_labels:
                    return connectors_dict, labels
                else:
                    return connectors_dict

    def _process_connections(self, connections):
        """Creates the connections and finds the routes.

        Args:
            connections: List of connections, as specified for the
                configuration in the documentation of the class.
        """
        # create connection
        for con in connections:
            self._add_connection(con)

        for inp_lab, inp in self.inputs.items():
            # use self._find_routes() to find routes from input inp
            routes_inp = self._find_routes(inp)
            # create routes
            for route in routes_inp:
                self._add_route(route)
        # sort the routes dictionary
        self._sort_routes()

    def _sort_routes(self):
        """Sorts the self.routes dictionary

        Sorting is done to match the order of self.inputs and self.outputs
        """
        sorted_routes = OrderedDict()
        for inp_lab, inp in self.inputs.items():
            if inp_lab not in self.routes:
                continue
            sorted_routes[inp_lab] = OrderedDict()
            for out_lab, out in self.outputs.items():
                if out_lab not in self.routes[inp_lab]:
                    continue
                routes = self.routes[inp_lab][out_lab]
                # If multiple routes between a certain input and output exist,
                # order the routes by length
                if len(routes) > 1:
                    route_lengths = [len(route) for route in routes]
                    sorted_indices = np.argsort(route_lengths)
                    routes = [routes[i] for i in sorted_indices]
                sorted_routes[inp_lab][out_lab] = routes
        self.routes = sorted_routes

    def _find_routes(self, start_node, previous_nodes=None):
        """Find all routes that start at start_node.

        Finds the routes recursively. start_node may be a box input or
        a switch.

        Args:
            start_node: box input or switch
            previous_nodes: List of previous nodes, for when the function
                            calls itself.

        Returns:
            The routes starting at start_node.
        """
        if previous_nodes is None:
            previous_nodes = []

        routes = []
        for con in self.connections:
            if start_node == con.end:
                con.flip()
            if start_node == con.start:
                # if the connection ends in a box output,
                # add the connection (as a route of length 1)
                if con.end.is_box_output():
                    routes.append([con])
                elif con.end.is_box_input():
                    raise Exception("Route in connections detected, "
                                    "that ends at an input.")
                elif con.end.is_switch_output():
                    # check if there is conflict with previous nodes
                    if con.end.switch in previous_nodes:
                        raise Exception("Loop detected in connections at"
                                        f"switch {con.end.switch}.")
                    # check orientation
                    if con.end.switch.orientation == 1:
                        raise Exception("Conflicting switch orientation "
                                        f"for switch {con.end.switch}")
                    # Set orientation of the switch
                    con.end.switch.orientation = -1
                    # Add the node to the previous nodes and call the method
                    # for the next node
                    if con.start.parent_type == 'switch':
                        previous_nodes.append(con.start.switch)
                    else:
                        previous_nodes.append(con.start)
                    next_step = self._find_routes(
                        con.end.switch.input,
                        previous_nodes=previous_nodes
                    )
                    # Merge the current connection with the resulting routes
                    for route in next_step:
                        routes.append([con] + route)
                # proceed the analogously for a switch input
                elif con.end.is_switch_input():
                    if con.end.switch in previous_nodes:
                        raise Exception("Loop detected in connections at"
                                        f"switch {con.end.switch}.")
                    if con.end.switch.orientation == -1:
                        raise Exception("Conflicting switch orientation "
                                        f"for switch {con.end.switch}")
                    con.end.switch.orientation = 1
                    if con.start.parent_type == 'switch':
                        previous_nodes.append(con.start.switch)
                    else:
                        previous_nodes.append(con.start)

                    # continue with both outputs
                    next_step0 = self._find_routes(
                        con.end.switch.output[0],
                        previous_nodes=previous_nodes
                    )

                    next_step1 = self._find_routes(
                        con.end.switch.output[1],
                        previous_nodes=previous_nodes
                    )

                    for route in next_step0:
                        routes.append([con] + route)
                    for route in next_step1:
                        routes.append([con] + route)

                else:
                    raise TypeError(f"Node {con.end} not recognised")

        return routes

    # - Private methods to set and get switches
    #  ----------------------------------------

    def _get_switch(self, switch):
        """Core method for reading the state of a switch.

        NOTE: This method is used to define the QCodes parameters
        of this Instrument. Not intended to be used on its own.
        Use 'self.read_state(switch)' instead!

        Sends a serial command to the Arduino to read the state of
        the switch 'switch'. A read command to the Arduino consists of
        a string of 3 characters 'r{group_id}{switch_id}', where group_id
        and switch_id identify the I/O-expander and switch on the PCB.
        The Arduino returns the state of the indicators,
        '10' for state 0 and '01' for state 1. Indicators '00' or '11'
        mean, that there is a problem somewhere.

        This method gets the correct group_id and switch_id for the
        switch 'switch', sends the read command to the Arduino via the serial
        interface, reads the return from the Arduino and returns the
        according state. The indicator states are saved in the instance
        of the switch: self.switch_by_label(switch).leds .

        The method is used to define the Qcodes parameters of this Instrument.

        Args:
            switch: Switch of which the state should be read.
                    Possible types:
                        int or str: switch label
                        ArduinoSwitchBoxSwitch: the switch itself.

        Returns:
            int: obtained switch state 0 or 1; -1 for indicators 00 or 11.
        """
        switch = self.switch_by_label(switch)
        id = self.switches[switch.label].id
        # make sure that the serial port is open
        self.assure_serial()
        # create command for the arduino and send it
        input_string = 'r' + str(id[0]) + str(id[1])
        self.serial.write(input_string.encode('ascii'))
        time.sleep(self.READ_DELAY)
        # retrieve result
        result = self.serial.readline().decode().rstrip()
        time.sleep(self.READ_DELAY)
        # store the indicators to the switch
        switch.indicators = (int(result[0]), int(result[1]))
        # raise error if the indicators show an error
        if switch.state is None:
            raise SwitchError("Reading the state was unsuccessful: Indicators "
                              f"of the switch show {switch.indicators}.")
        return switch.state

    def _set_switch(self, switch, state):
        """Core method for reading the state of a switch.

        NOTE: This method is used to define the QCodes parameters
        of this Instrument. Not intended to be used on its own.
        Use 'self.set_switch({switch:state})' instead!

        Sends a serial command to the Arduino to set the state of
        the switch 'switch'. A set command to the Arduino consists of
        a string of 3 characters {group_id}{switch_id}{state}',
        where group_id and switch_id identify the I/O-expander and switch
        on the PCB, and state = 0 or 1 is the target state.

        This method gets the correct group_id and switch_id for the
        switch 'switch' and sends the set command to the Arduino via the serial
        interface.

        The method is used to define the Qcodes parameters of this Instrument.

        Args:
            switch: Switch of which the state should be set.
                    Possible types:
                        int or str: switch label
                        ArduinoSwitchBoxSwitch: the switch itself.
        """
        switch = self.switch_by_label(switch)
        id = self.switches[switch.label].id
        # make sure that the serial port is open
        self.assure_serial()
        # create command for the arduino and send it
        input_string = str(id[0]) + str(id[1]) + str(state)
        self.serial.write(input_string.encode('ascii'))
        time.sleep(self.WRITE_DELAY)
        # read switch after setting it, to confirm switching
        try:
            self._get_switch(switch)
        except SwitchError:
            raise SwitchError("Reading switch after switching was "
                              "unsuccessful: Indicators of the switch show "
                              f"{switch.indicators}.")
        # raise error, if the switching was not successful
        if switch.state != state:
            raise SwitchError("Setting the switch was unsuccessful. The "
                              f"switch should be in state {state}, but "
                              f"the indicators show state {switch.state}.")

    def _set_route(self, inp, out, route_number=0):
        """Core method to set a route from an input to an output.

        NOTE: This method is used to define the QCodes parameters
        of this Instrument. Not intended to be used on its own.
        Use 'self.set_switch({inp:out})' instead!

        Args:
            inp: Input connector or its label.
            out: Output connector or its label.
            route_number: Index of route (if multiple routes exist)
        """
        if inp not in self.inputs:
            inp = inp.label
        if out not in self.outputs:
            out = out.label
        if (inp not in self.routes
                or out not in self.routes[inp]):
            raise RouteError(f"No routes found between "
                             f"{inp} and {out}.")
        # get possible routes corresponding to inp and out
        routes = self.routes[inp][out]
        num_routes = len(routes)
        # raise exeptions if needed
        if num_routes == 0:
            raise RouteError(f"No routes found between "
                             f"{inp} and {out}.")
        if route_number >= num_routes:
            raise RouteError("route_number has to be less than the "
                             f"number of routes {num_routes}.")
        # get route from dictionary
        route = self.routes[inp][out][route_number]
        # set switches
        self.set_switch({
            switch.label: state for switch, state in route.get_switch_states()
        })

    def _get_route(self, inp):
        """Core method to get the connected outputs of an input.

        NOTE: This method is used to define the QCodes parameters
        of this Instrument. Not intended to be used on its own.
        Use 'self.get_switch(inp)' instead!

        Args:
            inp: Input connector or its label.

        Returns:
            str: The output label
            If multiple connected outputs exist, a list of the labels
            is returned (Should generally not be the case).
        """
        inp = self.connector_by_label(inp)
        inp_routes = []
        # get the routes starting at the input and the maximum route length
        max_length = 0
        for routes in self.routes[inp.label].values():
            for route in routes:
                inp_routes.append(route)
                if len(route) > max_length:
                    max_length = len(route)

        # to find the outputs, the switches on possible routes are
        # successively measured. Routes that do not fit the measured switch
        # states are eliminated, such that following switches do not have
        # to be read.
        outputs = []  # list for the outputs
        routes = inp_routes
        measured_switch_states = {}  # store already measured switches
        for k in range(max_length):
            routes_left = []  # list to store remaining routes
            if len(routes) == 0:
                break
            for route in routes:
                # action only required if start or end of the
                # connection route[k] is a switch output
                if route[k].start.is_switch_output():
                    # check if the switch is already measured
                    if route[k].start.switch.label in measured_switch_states:
                        state = measured_switch_states[
                            route[k].start.switch.label]
                    else:
                        # measure switch and store result
                        state = route[k].start.switch.mode()
                        measured_switch_states[
                            route[k].start.switch.label] = state
                    # got to next route (such that this one is not added to
                    # the remaining routes)
                    if route[k].start.output_nr != state:
                        continue
                # if a output is reached, add it to outputs
                if route[k].end.is_box_output():
                    outputs.append(route[k].end.label)
                    continue
                # continue analogously to above
                elif route[k].end.is_switch_output():
                    if route[k].end.switch.label in measured_switch_states:
                        state = measured_switch_states[
                            route[k].end.switch.label]
                    else:
                        state = route[k].end.switch.mode()
                        measured_switch_states[
                            route[k].end.switch.label] = state
                    if route[k].end.output_nr != state:
                        continue
                # the route has not been eliminated,
                # add it to the remaining routes
                routes_left.append(route)
            # set routes to routes_left before starting the next iteration
            routes = routes_left

        # returns
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            outputs = [out for out in self.outputs if out in outputs]
            return outputs

    # - Legacy methods, not in use anymore, but might be useful in the future
    #   ---------------------------------------------------------------------

    def _active_routes(self, return_active_connections=False):
        """(Legacy) Returns routes that are currently connected.

        Note: This method does NOT read the switches but relies on the
        internally saved states of the switches. If switches have not been
        read, or there is some error, this will not represent the routes
        that are actually connected on the hardware.

        Args:
            return_active_connections (bool):
                Whether to return the active connections

        Returns:
            list: List of the active routes
                  if return_active_connections: also return a list of
                  the active connections

        """
        act_routes = []
        active_connections = []
        for inp, inp_routes in self.routes.items():
            for out, out_routes in inp_routes.items():
                for route in out_routes:
                    active = True
                    for switch, state in route.get_switch_states():
                        if switch.state != state:
                            active = False
                            break
                    if active:
                        act_routes.append(route)
                        for con in route:
                            if con not in active_connections:
                                active_connections.append(con)
        if return_active_connections:
            return act_routes, active_connections
        else:
            return act_routes

    def _switches_from_input(self, inp):
        """Return all switches, that are reachable from a certain input.

        Args:
            inp: The input (by label or connector object)

        Returns:
            list: A list of the switches.

        """
        inp = self.connector_by_label(inp)
        if not inp.is_box_input():
            raise ConnectorError("Argument has to be a box input.")
        switches = []
        for routes in self.routes[inp.label].values():
            for route in routes:
                for connection in route:
                    if (connection.start.parent_type == 'switch'
                            and connection.start.switch not in switches):
                        switches.append(connection.start.switch)
        return switches

    # Class methods
    # -------------

    # - Methods to handle serial ports
    #   ------------------------------

    _open_ports = {}

    @classmethod
    def get_ports(cls):
        """ Returns dictionary of (possibly) open serial ports

        Returns:
            dict: copy of _open_ports
        """
        return cls._open_ports.copy()

    @classmethod
    def add_port(cls, port, ser):
        """Add port to _open_ports

        Args:
            port (str): port of USB connection to Arduino, like 'COM5'
            ser (serial.Serial): serial connection
        """
        cls._open_ports[port] = ser

    @classmethod
    def get_serial(cls, port):
        """Get serial.Serial object by port

        Args:
            port(str): label of port

        Returns:
            serial.Serial: serial instance of the port
        """
        if port in cls._open_ports:
            return cls._open_ports[port]
        else:
            return None

    @classmethod
    def remove_port(cls, port):
        """Remove a port from cls._open_ports

        Args:
            port (str): label of the port
        """
        if port in cls._open_ports:
            if cls._open_ports[port].is_open:
                cls._open_ports[port].close()
            del cls._open_ports[port]

    # Static methods
    # --------------

    @staticmethod
    def _check_if_in_config(config, *keys):
        """Checks if dictionary 'config' has keys 'keys'

        For info on the dictionary, see class documentation.

        Args:
            config (dict): configuration dictionary
            *keys (str): keys to check
        """
        for key in keys:
            if key not in config:
                raise ValueError(f"Config must contain key '{key}")


# Classes for the components of the switch box
# --------------------------------------------


class ArduinoSwitchControlObject:
    """Base class for switches and connectors

    This class mainly exists for code extendability, if needed.
    """

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"ArduinoSwitchControlObject {self.label}"


class ArduinoSwitchControlConnector(ArduinoSwitchControlObject):
    """Class for connectors of the switch box

    Args:
        label: label of the connector
        parent_type: 'box' or 'switch'
        connector_type: 'input' or 'output'
        group: group, the connector belongs to (for box connectors)
        switch: switch, the connector belongs to (for switch connectors)
        output_nr: number corresponding to the state of the switch
                   (for switch outputs)

    Attributes:
        label: label of the connector
        parent_type (str): 'box' or 'switch'
        connector_type (str): 'input' or 'output'

        group: only for box connectors
        switch: only for switch connectors
        output_nr: only for switch outputs, 0 or 1


    """

    def __init__(self, label, parent_type, connector_type, group=None,
                 switch=None, output_nr=None):
        super().__init__(label)

        # parent_type:
        # if 'box': connector is an input or output of the switch box
        # if 'switch': connector is on a switch
        if parent_type not in ['box', 'switch']:
            raise ValueError("parent_type needs to be 'box' or 'switch', "
                             f"not {parent_type}")
        self.parent_type = parent_type

        # connector_type: 'input' or 'output'
        if connector_type not in ['input', 'output']:
            raise ValueError("connector_type needs to be 'input' or 'output', "
                             f"not {connector_type}")
        self.connector_type = connector_type

        if parent_type == 'box':
            # save group (None allowed)
            self.group = group
        else:
            # save switch
            if switch is None:
                raise TypeError("Keyword 'switch' as to be specified, "
                                "if parent_type == switch.")
            elif not isinstance(switch, ArduinoSwitchControlSwitch):
                raise TypeError("'switch' has to be of type "
                                "ArduinoSwitchControlSwitch, "
                                f"not {type(switch)}.")
            else:
                self.switch = switch

            # if switch output, save output number
            if connector_type == 'output':
                if output_nr is None:
                    raise TypeError("For an output connector of a switch, "
                                    "the number of the output 'output_nr' "
                                    "has to be specified.")
                elif output_nr not in [0, 1]:
                    raise ValueError("output_nr has to be 0 or 1 (as int)")
                else:
                    self.output_nr = output_nr

    def is_box_input(self):
        return (self.parent_type == 'box'
                and self.connector_type == 'input')

    def is_box_output(self):
        return (self.parent_type == 'box'
                and self.connector_type == 'output')

    def is_switch_input(self):
        return (self.parent_type == 'switch'
                and self.connector_type == 'input')

    def is_switch_output(self):
        return (self.parent_type == 'switch'
                and self.connector_type == 'output')


class ArduinoSwitchControlSwitch(ArduinoSwitchControlObject):
    """Class for the switches

    Args:
        label (int or str): label of the switch
        id (tuple): id of the switch on the PCB,
                    of format (group_id,switch_id)
        orientation (int): orientation of the switch in the routes between the
                           inputs and outputs. Can be:
                             1: the input of the switch points in the
                                direction of the inputs of the box.
                             -1: the input of the switch points in the
                                 direction of the outputs of the box.
                             0: the orientation of the switch is undetermined.

    Attributes:
        id (tuple): id of the switch on the PCB,
                    of format (group_id,switch_id)
        input (ArduinoSwitchControlConnector): input connector
        output (list): list of the output connectors
        mode: Qcodes parameter from the parent SwitchControl to set the switch
    """

    def __init__(self, label, id, orientation=0):
        super().__init__(label)
        self.id = id

        # orientation:
        # if 1: the outputs of the switch are in the direction
        #       of the output connectors of the box
        # if -1: the outputs of the switch are in the direction
        #        of the input of the box
        # if 0: orientation of the switch is not specified
        self.orientation = orientation

        # input connector
        input_label = f'{label}_in'
        self.input = ArduinoSwitchControlConnector(
            input_label, 'switch', 'input', switch=self)

        # output connector
        output_label = f'{label}_out_'
        self.output = [
            ArduinoSwitchControlConnector(
                output_label + '0', 'switch', 'output',
                switch=self, output_nr=0),
            ArduinoSwitchControlConnector(
                output_label + '1', 'switch', 'output',
                switch=self, output_nr=1)
        ]

        # state and indicators
        # state should only be set through indicators
        self._state = None
        self.indicators = (None, None)  # indicators set state

        self.mode = None

    def __repr__(self):
        return (f"ArduinoSwitchControlSwitch {self.label} with "
                f"id {self.id}")

    def __str__(self):
        return f"Switch {self.label}"

    @property
    def indicators(self):
        """Indicators of the switches

        Stored indicator states from reading the switch.
        (1,0) corresponds to state 0, (0,1) corresponds to state 1
        (0,0) and (1,1) signal an error with the switches or the PCB
        """
        return self._indicators

    @indicators.setter
    def indicators(self, indicators):
        try:
            indicators = tuple(indicators)
        except TypeError:
            raise TypeError("'indicators' has to be convertible to a tuple.")
        if indicators not in [(None, None), (0, 0), (1, 0), (0, 1), (1, 1)]:
            raise ValueError("'indicators' has to be tuple (0,0), (1,0), "
                             "(0,1) or (1,1). If the state of the indicators "
                             "is not know, it should be (None,None).")
        self._indicators = indicators
        # set state accordingly
        if indicators == (1, 0):
            self._state = 0
        elif indicators == (0, 1):
            self._state = 1
        else:
            self._state = None

    @property
    def state(self):
        """State of the switch corresponding to the indicators"""
        return self._state


class ArduinoSwitchControlConnection:
    """Class for connections between connectors

    Args:
        start (ArduinoSwitchControlConnector): start of the connection
        end (ArduinoSwitchControlConnector): end of the connection
    """

    def __init__(self, start, end):
        if not isinstance(start, ArduinoSwitchControlConnector):
            raise TypeError("'start' has to be of type "
                            "ArduinoSwitchControlConnector, "
                            f" not {type(start)}.")
        if not isinstance(end, ArduinoSwitchControlConnector):
            raise TypeError("'start' has to be of type "
                            "ArduinoSwitchControlConnector, "
                            f" not {type(end)}.")
        if start == end:
            raise ConnectionError("'start' and 'end' must be different.")
        if (start.parent_type == 'box' and end.parent_type == 'box'
                and start.connector_type == end.connector_type):
            raise ConnectionError("The connectors are both inputs or both "
                                  "outputs.")

        self._start = start
        self._end = end

        if start.parent_type == 'box' and start.connector_type == 'output':
            self.flip()
        if end.parent_type == 'box' and end.connector_type == 'input':
            self.flip()

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def __getitem__(self, item):
        if item in [0, 'start']:
            return self.start
        elif item in [1, 'end']:
            return self.end
        else:
            raise IndexError("Index has to be 0 or 'start' for the first "
                             "connector and 1 or 'end' for the second one.")

    def __eq__(self, con):
        try:
            if self.start == con.end and self.end == con.start:
                return True
            elif self.start == con.start and self.end == con.end:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __repr__(self):
        return f"ArduinoSwitchControlConnection " \
               f"({self.start.label},{self.end.label})"

    def flip(self):
        """Flip start and end"""
        self._start, self._end = self._end, self._start


class ArduinoSwitchControlRoute:
    """Class for routes

    Args:
        connections (list): List of instances of ArduinoSwitchControlConnector
                            defining the route

    Attributes:
        connections (list): connections defining the route
        input: input of the route (start of first connection in route)
        output: output of the route (end of last connection in route)
    """

    def __init__(self, connections):
        inp = connections[0].start
        if not (inp.parent_type == 'box' and inp.connector_type == 'input'):
            raise RouteError(f"A route has to begin at an input.")
        out = connections[-1].end
        if not (out.parent_type == 'box' and out.connector_type == 'output'):
            raise RouteError(f"A route has to end at an output.")
        self.input = inp
        self.output = out
        self.connections = connections

    # magic methods to make route iterable

    def __len__(self):
        return len(self.connections)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self.connections):
            raise StopIteration
        self._iter += 1
        return self.connections[self._iter - 1]

    def __getitem__(self, i):
        return self.connections[i]

    def __contains__(self, con):
        return con in self.connections

    def __repr__(self):
        return f"ArduinoSwitchControlRoute {self.input.label}" \
               f" to {self.output.label}"

    def get_switch_states(self):
        """Get states of switches that connect the route

        Returns:
            list: List of tuples (switch,state)
        """
        switches_states = []
        for connection in self.connections:
            if connection.start.is_switch_output():
                switches_states.append((connection.start.switch,
                                        connection.start.output_nr))
            if connection.end.is_switch_output():
                switches_states.append((connection.end.switch,
                                        connection.end.output_nr))
        return switches_states


# Exceptions:
# -----------

class SwitchControlError(Exception):
    pass


class ConnectorError(Exception):
    pass


class SwitchError(Exception):
    pass


class RouteError(Exception):
    pass
