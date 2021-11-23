import serial
import time
import numpy as np

import qcodes as qc
from qcodes import Instrument

from ipywidgets import widgets, HBox, VBox, Layout
from IPython.display import display, set_matplotlib_formats, HTML

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from copy import copy, deepcopy


# switch_box = ArduinoSwitchControl(...)
# switch_box_control = ArduinoSwitchControlWrapper('mybox', switch_box)
#
# class ArduinoSwitchControlWrapper(Instrument):
#     def __init__(self, name, switch_box):
#         super().__init__(name)
#         self.switch_box = switch_box
#         for id in self.switch_box.all_ids:
#             self.add_parameter(...
#                                set_cmd=lambda.... self.switch_box.write...(..)
#                                )
#             self.add_parameter(...
#                                parameter_class=ManualParameter
#                                )


class ArduinoSwitchControl(Instrument):
    def __init__(self, name, port, config, start_serial=True):

        self._check_if_in_config(config,
                                 'switches', 'inputs', 'outputs',
                                 'connections')

        super().__init__(name)

        self.port = port

        # Switch properties
        # -----------------
        # If switches is int, use as number of switches
        # and switch_labels = [1,2,...,switches]

        switches = config['switches']
        if isinstance(switches, int):
            if not 0 <= switches <= 20:
                raise ValueError("Number of switches 'switches' must be"
                                 f"between 0 and 20, not {switches}.")
            self._num_switches = switches
            self.switch_labels = [n for n in range(1, self.num_switches + 1)]
        # If switches is iterable, use entries as switch labels and length
        # as number of switches
        else:
            try:
                switch_labels = list(switches)
            except TypeError:
                raise TypeError("Type of 'switches' must be a positive"
                                "integer for the number of switches, or a"
                                "list for the switch labels. The maximum"
                                "number of switches is 20.")
            num_switches = len(switch_labels)
            if num_switches > 20:
                num_switches = 20
                switch_labels = switch_labels[:20]
                # Issue warning, that switch labels are cut off after 20?
            self._num_switches = num_switches
            self.switch_labels = switch_labels

        # switch states for convenience
        self.switch_states = ([(lab, 0) for lab in self.switch_labels] +
                              [(lab, 1) for lab in self.switch_labels])

        # ids of switches as in the communication to the Arduino
        switch_ids_list = [(n, m) for n in range(5) for m in range(4)]
        self.switch_ids = {}
        self.switches = {}
        self.switch_inputs = []
        self.switch_outputs = []
        self.switch_input_labels = []
        self.switch_output_labels = []
        for id, label in zip(switch_ids_list, self.switch_labels):
            switch = self.add_switch(label, id,
                                     return_switch=True)  # return switch
            # parameter anlegen
            param_name = f'switch_{label}_mode'
            # def get_cmd():
            #     return self._get_switch(switch)
            #
            # def set_cmd(x):
            #     self._set_switch(switch,x)

            self.add_parameter(
                param_name,
                label=f'{param_name} of {self.name}',
                vals=qc.validators.Enum(0, 1),
                get_cmd=lambda s=label: self._get_switch(s),
                set_cmd=lambda x, s=label: self._set_switch(s, x),
                docstring="possible values: 0, 1",
            )
            switch.mode = self.parameters[param_name]

        # Inputs and outputs
        # ------------------
        # The inputs and outputs can be specified in the following ways
        #     - As an integer for the number of connectors.
        #       The inputs are labeled with 'I1', 'I2',..., the outputs are
        #       with 'O1', 'O2',....
        #     - A tuple ('X',N), where 'X' is the label and N the number of
        #       connectors. The connectors are labeled with X1, X2,....
        #     - As a list to group inputs and outputs. The labeling format
        #       is 'X.n', where X is the group label and n the connector
        #       number in the group. The entries of the list can be:
        #         - A tuple ('X',N) where 'X' is the label of the group and
        #           N is the number of connectors in the group.
        #         - An integer N. Groups without a given label will be
        #           labelled with 'I1', 'I2',... for input groups and
        #           'O1','O2',... for output groups

        inputs = config['inputs']

        self.inputs, self.input_groups, self.input_labels = self.create_connectors(
            inputs,
            default_label='I',
            return_groups=True,
            connector_type='input',
            return_labels=True
        )

        outputs = config['outputs']

        self.outputs, self.output_groups, self.output_labels = self.create_connectors(
            outputs,
            default_label='O',
            return_groups=True,
            connector_type='output',
            return_labels=True
        )
        self.num_inputs = len(self.inputs)
        self.num_outputs = len(self.outputs)

        # Connections & Routes
        # --------------------
        # Connections between switches, or between switches and connectors
        # are specified as a list of tuples [(connector1,connector2)],
        # where a connector can be:
        #     - the label of an input or output
        #     - the label of a switch for the input of the switch
        #     - a tuple (switch,state) for the output of a switch
        connections = config['connections']
        self.connections = []
        self.routes = {}
        self.process_connections(connections)

        # Data from GUI
        # -------------

        self.gui_widgets = {}

        self.switch_display = {}
        if 'gui_properties' in config:
            for key, val in config['gui_properties'].items():
                if key in ['switch_positions',
                           'inputs_y',
                           'outputs_y',
                           'connector_spacing',
                           'connector_group_spacing',
                           'connector_radius',
                           'connections_mid_y',
                           'switches_mirrored']:
                    self.switch_display[key] = val

        # Start serial communication
        # --------------------------
        self.serial = None
        if start_serial:
            self.start_serial()

    # Magic methods
    # -------------
    def __del__(self):
        self.end_serial()
        self.remove_port(self.port)
        # close figure

    # class constants
    # ---------------

    SHORT_DELAY = 0.05
    DELAY = 0.15

    LED_ON_COLOR = (0., 1., 0.)
    LED_OFF_COLOR = (0., 0.4, 0.)
    CONNECTOR_COLOR = (0.7, 0.7, 0.7)
    CONNECTOR_COLOR_ACTIVE = (0., 0.9, 0.)
    CONNECTOR_LINE_COLOR = (0., 0., 0.)
    CONNECTOR_LINE_COLOR_ACTIVE = (0., 0.4, 0.)
    CONNECTION_COLOR = (0., 0., 0)
    CONNECTION_COLOR_ACTIVE = CONNECTOR_COLOR_ACTIVE

    # Properties
    # ----------
    @property
    def num_switches(self):
        return self._num_switches

    # Instance methods
    # ----------------

    # - Methods for serial communication
    #   --------------------------------

    def start_serial(self, override: bool = True):
        """Start serial communication

        :param override: whether an existing serial communication
                         should be closed and replaced.
                         Recommended: True
        """
        # get possible open serial
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

    # - Methods for reading and setting the switches
    #   --------------------------------------------

    # id über lambda string
    def read_state_by_switch_id(self, id,
                                refresh_gui=True):
        result = self.switch_ids[id].mode()

        if refresh_gui:
            self.refresh_gui()
        return result

    def set_state_by_switch_id(self, id, state,
                               refresh_gui=True,
                               verify=False):
        switch = self.switch_ids[id]
        switch.mode(state)
        if verify:
            res = switch.mode()

        if refresh_gui:
            self.refresh_gui()

        if verify:
            return state == res

    def read_state(self, switch,
                   refresh_gui=True):
        switch = self.switch_by_label(switch)
        result = switch.mode()

        if refresh_gui:
            self.refresh_gui()
        return result

    def set_state(self, switch, state, refresh_gui=True, verify=False):
        switch = self.switch_by_label(switch)
        switch.mode(state)
        if verify:
            res = switch.mode()

        if refresh_gui:
            self.refresh_gui()

        if verify:
            return state == res

    def set_route(self, in_label, out_label, route_number=0, verify=False):
        if (in_label not in self.routes
                or out_label not in self.routes[in_label]):
            raise ValueError(f"No routes found between "
                             f"{in_label} and {out_label}.")
        routes = self.routes[in_label][out_label]
        num_routes = len(routes)
        if num_routes == 0:
            raise ValueError(f"No routes found between "
                             f"{in_label} and {out_label}.")
        if route_number >= num_routes:
            raise ValueError("route_number has to be less than the "
                             f"number of routes {num_routes}.")
        route = self.routes[in_label][out_label][route_number]
        verified = True
        for switch, state in self.get_switch_states_on_route(route):
            if verify:
                res = self.set_state(switch.label, state,
                                     refresh_gui=False, verify=True)
                verified = verified and res
            else:
                self.set_state(switch.label, state,
                               refresh_gui=False, verify=False)
        self.refresh_gui()
        if verify:
            return verified

    def verify_route(self, in_label, out_label, route_number=0):
        if (in_label not in self.routes
                or out_label not in self.routes[in_label]):
            raise ValueError(f"No routes found between "
                             f"{in_label} and {out_label}.")
        routes = self.routes[in_label][out_label]
        num_routes = len(routes)
        if num_routes == 0:
            raise ValueError(f"No routes found between "
                             f"{in_label} and {out_label}.")
        if route_number >= num_routes:
            raise ValueError("route_number has to be less than the "
                             f"number of routes {num_routes}.")
        route = self.routes[in_label][out_label][route_number]
        for switch, state in self.get_switch_states_on_route(route):
            state0 = self.read_state(switch.label, refresh_gui=False)
            if state != state0:
                self.set_state(switch.label, state, refresh_gui=False)
                state1 = self.read_state(switch.label,
                                         refresh_gui=False)
                if state1 != state:
                    self.refresh_gui()
                    return False
        self.refresh_gui()
        return True

    def active_routes(self, return_active_connections=False):
        act_routes = []
        active_connections = []
        for inp, inp_routes in self.routes.items():
            for out, out_routes in inp_routes.items():
                for route in out_routes:
                    active = True
                    for switch, state in self.get_switch_states_on_route(
                            route):
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

    # - Methods for the GUI
    #   -------------------

    def open_gui(self, show_switch_display=False,
                 **switch_display_options):

        # widgets
        # -------

        # status
        status_title = '<b>Status:</b> <br>'
        if show_switch_display:
            status = widgets.HTML(value=status_title,
                                  layout=Layout(width='180px'))
        else:
            status = widgets.HTML(value=status_title,
                                  layout=Layout(width='600px'))

        def update_status(text):
            status.value = status_title + f'<code>{text}</code>'

        # select switch label and state and buttons for read and set
        switch_select = widgets.Dropdown(
            options=self.switch_labels,
            value=self.switch_labels[0],
            description='Switch',
            disabled=False,
            layout=Layout(width='80%')
        )

        state_select = widgets.Dropdown(
            options=[0, 1],
            value=0,
            description='State',
            disabled=False,
            layout=Layout(width='80%')
        )

        read_button = widgets.Button(description='Read state',
                                     layout=Layout(width='80%'))
        set_button = widgets.Button(description='Set state',
                                    layout=Layout(width='80%'))
        set_verify_checkbox = widgets.Checkbox(
            value=False,
            description='Verify',
            disabled=False,
            indent=True,
            layout=Layout(width='80%')
        )
        title_read_set = widgets.HTML(value='<b>Read and set switch state</b>')

        read_set_box = VBox([title_read_set,
                             switch_select,
                             state_select,
                             set_button,
                             set_verify_checkbox,
                             read_button],
                            layout=Layout(width='180px'))

        # select routes and buttons for set and verify route

        routes_in_options = [x for x in self.input_labels
                             if x in self.routes.keys()]

        routes_in_select = widgets.Dropdown(options=routes_in_options,
                                            values=routes_in_options[0],
                                            description='Input',
                                            layout=Layout(width='80%')
                                            )
        routes_out_init_options = [
            x for x in self.output_labels
            if x in self.routes[routes_in_options[0]].keys()
        ]
        routes_out_select = widgets.Dropdown(
            options=routes_out_init_options,
            description='Output',
            layout=Layout(width='80%')
        )

        set_route_button = widgets.Button(description='Set route',
                                          layout=Layout(width='80%'))

        set_route_verify_checkbox = widgets.Checkbox(
            value=False,
            description='Verify',
            disabled=False,
            indent=True,
            layout=Layout(width='80%')
        )

        verify_route_button = widgets.Button(description='Verify route',
                                             layout=Layout(width='80%'))

        title_route = widgets.HTML(value='<b>Connect input to output</b>')

        routes_box = VBox([title_route,
                           routes_in_select,
                           routes_out_select,
                           set_route_button,
                           set_route_verify_checkbox],
                          layout=Layout(width='180px'))

        # Show active routes

        right_arrow = '&#8594;'

        active_route_displays = {i: widgets.HTML(value=i)
                                 for i in routes_in_options}

        update_act_routes_button = widgets.Button(description='Update',
                                                  layout=Layout(width='80%'))

        title_active_routes = widgets.HTML(value='<b>Active routes</b>')

        act_routes_box = VBox(
            [title_active_routes]
            + [active_route_displays[x] for x in routes_in_options]
            + [update_act_routes_button],
            layout=Layout(width='180px')
        )

        self.gui_widgets.update({
            'status_title': status_title,
            'status': status,
            'switch_select': switch_select,
            'state_select': state_select,
            'read_button': read_button,
            'set_button': set_button,
            'title_read_set': title_read_set,
            'read_set_box': read_set_box,
            'routes_in_select': routes_in_select,
            'routes_out_select': routes_out_select,
            'set_route_button': set_route_button,
            'verify_route_button': verify_route_button,
            'title_route': title_route,
            'active_route_displays': active_route_displays,
            'update_act_routes_button': update_act_routes_button,
            'title_active_routes': title_active_routes,
            'routes_in_options': routes_in_options
        })

        # interaction methods

        def read_button_click(btn):
            result = self.read_state(switch_select.value,
                                     refresh_gui=False)
            if result == 2:
                update_status('State not readable, check connection.')
            elif result in [0, 1]:
                update_status(
                    f'State of switch {switch_select.value} is {result}.')
                state_select.value = int(result)
            else:
                update_status(f'Unexpected return: {result}')
            self.refresh_gui()

        def set_button_click(btn):
            if set_verify_checkbox.value:
                verified = self.set_state(switch_select.value,
                                          state_select.value,
                                          refresh_gui=False,
                                          verify=True)
                if verified:
                    update_status(
                        f'Setting switch {switch_select.value} '
                        f'to state {state_select.value} successful.')
                else:
                    update_status(
                        f'ATTENTION: Setting switch {switch_select.value} '
                        f'to state {state_select.value} NOT successful.')
            else:
                self.set_state(switch_select.value,
                               state_select.value,
                               refresh_gui=False,
                               verify=False)
                update_status(
                    f'Set switch {switch_select.value} to state {state_select.value}.')
            self.refresh_gui()

        def set_route_button_click(btn):
            if set_route_verify_checkbox.value:
                verified = self.set_route(routes_in_select.value,
                                          routes_out_select.value, verify=True)
                if verified:
                    update_status(
                        f'Setting route {routes_in_select.value} to '
                        f'{routes_out_select.value} successful.')
                else:
                    update_status(
                        f'ATTENTION: Setting route {routes_in_select.value}'
                        f' to {routes_out_select.value} NOT successful.')
            else:
                self.set_route(routes_in_select.value, routes_out_select.value,
                               verify=False)
                update_status(
                    f'Set route {routes_in_select.value} to '
                    f'{routes_out_select.value}.')
            self.refresh_gui()

        def verify_route_button_click(btn):
            success = self.verify_route(routes_in_select.value,
                                        routes_out_select.value)
            if success:
                update_status(
                    f'Verified route {routes_in_select.value} '
                    f'to {routes_out_select.value}.')
            else:
                update_status(
                    f'Verifying route {routes_in_select.value} '
                    f'to {routes_out_select.value} not successful.')

        def routes_observe_input(change):
            routes_out_select.options = [
                x for x in self.output_labels
                if x in self.routes[change.new].keys()
            ]

        def click_update_act_routes_button(btn):
            self.refresh_gui()

        read_button.on_click(read_button_click)
        set_button.on_click(set_button_click)
        set_route_button.on_click(set_route_button_click)
        verify_route_button.on_click(verify_route_button_click)
        routes_in_select.observe(routes_observe_input, names='value')
        update_act_routes_button.on_click(click_update_act_routes_button)

        if show_switch_display:
            if 'width' in switch_display_options:
                width = switch_display_options['width']
            else:
                width = 600
                switch_display_options['width'] = width

            if 'height' in switch_display_options:
                height = switch_display_options['height']
            else:
                height = 600
                switch_display_options['height'] = height

            self.switch_display.update(switch_display_options)

            switch_display_output = widgets.Output(
                layout=Layout(width=f'{width + 20}px')
            )

            dpi = 100
            figure_width = width / dpi
            figure_height = height / dpi

            with switch_display_output:
                fig, ax = plt.subplots(
                    figsize=(figure_width, figure_height))
                plt.show()

            fig.canvas.toolbar_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.header_visible = False
            fig.canvas.resizable = False

            self.switch_display.update({
                'output': switch_display_output,
                'fig': fig,
                'width': figure_width,
                'height': figure_height,
                'ax': ax,
            })

            ax.set_position((0, 0, 1, 1))
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

            # ax.set_xticks(list(range(0, 100, 10)))
            # ax.set_yticks(list(range(0, 100, 10)))
            # plt.grid()

            # draw switches
            self._gui_draw_switches()

            # draw connectors
            self._gui_draw_input_connectors()
            self._gui_draw_output_connectors()

            # draw connections
            self._gui_draw_all_connections()

            # Display
            # -------

            left_column = VBox([read_set_box, routes_box,
                                act_routes_box, status],
                               layout=Layout(width='190px'))

            full_box = HBox([left_column, switch_display_output])

        else:
            top_row_box = HBox([read_set_box, routes_box, act_routes_box])
            full_box = VBox([top_row_box, status])

        self.refresh_gui()
        display(full_box)

    def refresh_gui(self):
        self._refresh_gui_active_routes()
        if 'fig' in self.switch_display:
            self._gui_switch_set_all_LEDs()
            self._gui_draw_active_routes()

    # - Methods for getting the states saved by the box
    #   -----------------------------------------------

    def get_saved_output_by_switch_id(self, id):
        return self.switch_ids[id].state

    def get_saved_output_by_switch_label(self, switch_label):
        return self.switches[switch_label].state

    def get_output_checked_by_switch_id(self, id):
        return self.switch_ids[(id)].state_checked

    def get_output_checked_by_switch_label(self, switch_label):
        return self.switches[switch_label].state_checked

    # - Helper functions for initializing and running the box
    #   -----------------------------------------------------

    def switch_by_label(self, label):
        if isinstance(label, ArduinoSwitchControlSwitch):
            return label
        return self.switches[label]

    def return_connector_by_label(self, label):
        if label in self.inputs:
            return self.inputs[label]
        elif label in self.outputs:
            return self.outputs[label]
        elif label in self.switch_input_labels:
            return self.switches[label].input
        elif label in self.switch_output_labels:
            return self.switches[label[0]].output[label[1]]
        elif isinstance(label, (
                ArduinoSwitchControlConnector,
                ArduinoSwitchControlSwitchConnector)):
            return label
        else:
            raise ValueError(f"Label {label} not recognised.")

    def create_connectors(self, connectors, connector_type=None,
                          default_label='C',
                          return_groups=False, in_group=False,
                          return_labels=False):
        if isinstance(connectors, int):
            if connectors < 0:
                raise ValueError("Number of connectors 'connectors' must"
                                 "be positive.")
            if in_group:
                label_string = default_label + '.'
            else:
                label_string = default_label
            labels = [label_string + str(n) for n in range(1, connectors + 1)]
            if connector_type == 'input':
                connectors_dict = {
                    label: ArduinoSwitchControlInput(label,
                                                     group=default_label)
                    for label in labels}
            elif connector_type == 'output':
                connectors_dict = {
                    label: ArduinoSwitchControlOutput(label,
                                                      group=default_label)
                    for label in labels}
            else:
                connectors_dict = {
                    label: ArduinoSwitchControlConnector(label,
                                                         group=default_label)
                    for label in labels}
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
                connectors_dict, labels = self.create_connectors(
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
                connectors_dict = {}
                groups = []
                labels = []
                for n, group in enumerate(connectors):
                    if isinstance(group, int):
                        group = (f'{default_label}{n + 1}', group)
                    else:
                        group = (group[0], group[1])
                    groups.append(group[0])
                    con_dict, labs = self.create_connectors(
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

    def add_switch(self, label, id, return_switch=True):
        switch = ArduinoSwitchControlSwitch(label, id)
        self.switch_ids[id] = switch
        self.switches[label] = switch
        self.switch_inputs.append(switch.input)
        self.switch_outputs.append(switch.output[0])
        self.switch_outputs.append(switch.output[1])
        self.switch_input_labels.append(switch.label)
        self.switch_output_labels.append((switch.label, 0))
        self.switch_output_labels.append((switch.label, 1))
        if return_switch:
            return switch

    def add_connection_to_properties(self, con):
        if isinstance(con.start, ArduinoSwitchControlConnector):
            con.start.connections.append(con)
        elif isinstance(con.start, ArduinoSwitchControlSwitchConnector):
            con.start.connections.append(con)
            con.start.switch.connections.append(con)
        else:
            raise TypeError("con.start not recognized")
        if isinstance(con.end, ArduinoSwitchControlConnector):
            con.end.connections.append(con)
        elif isinstance(con.end, ArduinoSwitchControlSwitchConnector):
            con.end.connections.append(con)
            con.end.switch.connections.append(con)
        else:
            raise TypeError("con.start not recognized")
        self.connections.append(con)

    def add_connection(self, con):
        start = self.return_connector_by_label(con[0])
        end = self.return_connector_by_label(con[1])
        if (isinstance(start, ArduinoSwitchControlInput)
                and isinstance(end, ArduinoSwitchControlInput)):
            raise Exception(f"Connection {con} connects"
                            f" input to input.")
        elif (isinstance(start, ArduinoSwitchControlOutput)
              and isinstance(end, ArduinoSwitchControlOutput)):
            raise Exception(f"Connection {con} connects"
                            f" input to input.")
        elif (isinstance(start, ArduinoSwitchControlOutput) or
              isinstance(end, ArduinoSwitchControlInput)):
            start, end = end, start
        elif (isinstance(start, ArduinoSwitchControlSwitchConnector)
              and isinstance(end, ArduinoSwitchControlSwitchConnector)):
            if start.switch == end.switch:
                raise Exception(f"Switch {start.switch} connects to itself.")

        connection = ArduinoSwitchControlConnection(start, end)

        self.add_connection_to_properties(connection)

    def add_route(self, route):
        inp = route[0].start
        if not isinstance(inp, ArduinoSwitchControlInput):
            raise ValueError(f"First connection in route has to start "
                             f"with an input, not {type(inp)}.")
        out = route[-1].end
        if not isinstance(out, ArduinoSwitchControlOutput):
            raise ValueError(f"Last connection in route has to end "
                             f"with an output, not {type(out)}.")
        route_obj = ArduinoSwitchControlRoute(inp, out, route)

        if inp.label not in self.routes:
            self.routes[inp.label] = {out.label: [route_obj]}
        elif out.label not in self.routes[inp.label]:
            self.routes[inp.label][out.label] = [route_obj]
        else:
            self.routes[inp.label][out.label].append(route_obj)

        # add route to other objects

    def process_connections(self, connections):

        for con in connections:
            self.add_connection(con)

        switch_orientations = [0] * self.num_switches
        for inp_lab, inp in self.inputs.items():
            routes_inp = self._find_routes(inp)
            for route in routes_inp:
                self.add_route(route)
            # out = route_i[-1][1]
            #             if inp in routes:
            #                 if out in routes[inp]:
            #                     routes[inp][out] += [route_i]
            #                 else:
            #                     routes[inp][out] = [route_i]
            #             else:
            #                 routes[inp] = {out: [route_i]}

    def switch_index(self, switch_label):
        return self.switch_labels.index(switch_label)

    def _find_routes(self, start_node, previous_nodes=None):
        if previous_nodes is None:
            previous_nodes = []

        routes = []
        for con in self.connections:
            if start_node == con.end:
                con.flip()
            if start_node == con.start:
                if isinstance(con.end, ArduinoSwitchControlOutput):
                    routes.append([con])
                elif isinstance(con.end, ArduinoSwitchControlInput):
                    raise Exception("Route in connections detected, "
                                    "that ends at an input.")
                elif isinstance(con.end, ArduinoSwitchControlSwitchOutput):
                    # check if there is conflict with previous nodes
                    if con.end.switch in previous_nodes:
                        raise Exception("Loop detected in connections at"
                                        f"switch {con.end.switch}.")
                    # check orientation
                    if con.end.switch.orientation == 1:
                        raise Exception("Conflicting switch orientation "
                                        f"for switch {con.end.switch}")
                    con.end.switch.orientation = -1
                    if isinstance(con.start,
                                  ArduinoSwitchControlSwitchConnector):
                        previous_nodes.append(con.start.switch)
                    else:
                        previous_nodes.append(con.start)
                    next_step = self._find_routes(
                        con.end.switch.input,
                        previous_nodes=previous_nodes
                    )
                    for route in next_step:
                        routes.append([con] + route)

                elif isinstance(con.end, ArduinoSwitchControlSwitchInput):
                    if con.end.switch in previous_nodes:
                        raise Exception("Loop detected in connections at"
                                        f"switch {con.end.switch}.")
                    if con.end.switch.orientation == -1:
                        raise Exception("Conflicting switch orientation "
                                        f"for switch {con.end.switch}")
                    con.end.switch.orientation = 1
                    if isinstance(con.start,
                                  ArduinoSwitchControlSwitchConnector):
                        previous_nodes.append(con.start.switch)
                    else:
                        previous_nodes.append(con.start)
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

    def get_route_start_end_pair_labels(self, sort_outputs=False):
        routes_start_end = []
        for inp, inp_routes in self.routes.items():
            if sort_outputs:
                routes_start_end.append(
                    (inp, list(inp_routes)))
            else:
                routes_start_end += [(inp, out)
                                     for out in inp_routes]
        return routes_start_end

    def _check_if_in_config(self, config, *keys):
        for key in keys:
            if key not in config:
                raise ValueError(f"Config must contain key '{key}")

    # - Private methods to set and get switches
    #   ---------------------------------------

    def _get_switch(self, switch):
        switch = self.switch_by_label(switch)
        id = self.switches[switch.label].id
        self.assure_serial()
        input_string = 'r' + str(id[0]) + str(id[1])
        self.serial.write(input_string.encode('ascii'))
        time.sleep(self.SHORT_DELAY)
        result = self.serial.readline().decode().rstrip()
        time.sleep(self.SHORT_DELAY)
        switch.leds = (int(result[0]), int(result[1]))
        switch.state_checked = True
        if result == '10':
            return 0
        elif result == '01':
            return 1
        else:
            return -1

    def _set_switch(self, switch, state):
        switch = self.switch_by_label(switch)
        id = self.switches[switch.label].id
        self.assure_serial()
        input_string = str(id[0]) + str(id[1]) + str(state)
        self.serial.write(input_string.encode('ascii'))
        time.sleep(self.DELAY)
        if state == 0:
            switch.leds = (1, 0)
        elif state == 1:
            switch.leds = (0, 1)
        switch.state_checked = False

    # - Helper functions for the GUI
    #   ----------------------------

    def _refresh_gui_active_routes(self):
        active_routes = self.active_routes(return_active_connections=False)
        if 'active_route_displays' not in self.gui_widgets:
            return

        for inp in self.input_labels:
            self.gui_widgets['active_route_displays'][inp].value = inp

        for route in active_routes:
            inp = route.start.label
            value = inp + '  &#8594;  ' + route.end.label
            self.gui_widgets['active_route_displays'][inp].value = value

    def _gui_draw_switches(self):
        if 'switch_positions' not in self.switch_display:
            raise Exception("The switch positions must be specified "
                            "in the options for the graph to work.")
        switch_positions = self.switch_display['switch_positions']

        if 'switches_mirrored' not in self.switch_display:
            self.switch_display['switches_mirrored'] = {
                s: False for s in self.switch_labels
            }

        switches_mirrored = self.switch_display['switches_mirrored']

        switch_positions = self.switch_display['switch_positions']

        for (lab, switch), pos, mirror in zip(self.switches.items(),
                                              switch_positions,
                                              switches_mirrored):
            self._gui_draw_switch(switch, pos,
                                  mirror=mirror)

    def _gui_draw_switch(self, switch, position, mirror=False,
                         width=5.3, height=8):
        fig = self.switch_display['fig']
        ax = self.switch_display['ax']
        x, y = position

        x0 = x - width / 2
        x1 = x + width / 2
        y0 = y - height / 2
        y1 = y + height / 2

        switch.gui_properties.update({
            'position': position,
            'x': x,
            'y': y,
            'mirror': mirror,
            'width': width,
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1,
            'fig': fig,
            'ax': ax
        })

        switch.gui_properties['body'] = ax.add_patch(
            Rectangle((x0, y0), width, height,
                      edgecolor=(0, 0, 0),
                      facecolor=(0.9, 0.9, 0.9),
                      fill=True,
                      lw=1.5)
        )

        led_radius = width / 8
        switch.gui_properties['led_radius'] = led_radius

        led_x_disp = width * (1 / 4)
        led_y_disp = height * 6 / 16

        led0_x = x - led_x_disp
        led1_x = x + led_x_disp

        if mirror:
            led0_x, led1_x = led1_x, led0_x

        if switch.orientation == -1:
            led0_y = y + led_y_disp
            led1_y = y + led_y_disp
            out_y = y1
            in_y = y0
        else:
            led0_y = y - led_y_disp
            led1_y = y - led_y_disp
            out_y = y0
            in_y = y1

        led0_pos = (led0_x, led0_y)
        led1_pos = (led1_x, led1_y)

        switch.gui_properties.update({
            'led0_pos': led0_pos,
            'led0_x': led0_x,
            'led0_y': led0_y,
            'led1_pos': led1_pos,
            'led1_x': led1_x,
            'led1_y': led1_y,
            'led_radius': led_radius,
            'out0_pos': (led0_x, out_y),
            'out1_pos': (led1_x, out_y),
            'in_pos': (x, in_y),
        })

        switch.gui_properties['led0'] = ax.add_patch(
            Circle(led0_pos, led_radius,
                   edgecolor=(0., 0., 0.),
                   facecolor=self.LED_ON_COLOR,
                   fill=True,
                   lw=1))
        switch.gui_properties['led1'] = ax.add_patch(
            Circle(led1_pos, led_radius,
                   edgecolor=(0., 0., 0,),
                   facecolor=self.LED_ON_COLOR,
                   fill=True,
                   lw=1))

        switch.gui_properties['led0_label'] = ax.text(
            led0_x, led0_y + 1.2 * led_radius, '0',
            ha='center', va='bottom', size=width * 1.5
        )

        switch.gui_properties['led1_label'] = ax.text(
            led1_x, led1_y + 1.2 * led_radius, '1',
            ha='center', va='bottom', size=width * 1.5
        )

        label_y_disp = -height * 3 / 32
        if switch.orientation == -1:
            label_y_disp *= -1
            va = 'top'
        va = 'bottom'
        switch.gui_properties['label_y_disp'] = label_y_disp

        switch.gui_properties['label'] = ax.text(x, y + label_y_disp,
                                                 switch.label, ha='center',
                                                 va=va, size=3 * width)

        self._gui_switch_set_LEDs(switch)

    def _gui_switch_set_LEDs(self, switch):
        if switch in self.switches:
            switch = self.switches[switch]
        state0, state1 = switch.leds
        gui_properties = switch.gui_properties
        if ('led0' not in switch.gui_properties
                or 'led1' not in switch.gui_properties):
            return
        if state0 is None:
            gui_properties['led0'].set_facecolor((1, 1, 1, 0))
        elif state0 == 1:
            gui_properties['led0'].set_facecolor(self.LED_ON_COLOR)
        elif state0 == 0:
            gui_properties['led0'].set_facecolor(self.LED_OFF_COLOR)
        else:
            gui_properties['led0'].set_facecolor((1, 1, 1, 0))
        if state0 is None:
            gui_properties['led1'].set_facecolor((1, 1, 1, 0))
        elif state1 == 1:
            gui_properties['led1'].set_facecolor(self.LED_ON_COLOR)
        elif state1 == 0:
            gui_properties['led1'].set_facecolor(self.LED_OFF_COLOR)
        else:
            gui_properties['led1'].set_facecolor((1, 1, 1, 0))

    def _gui_switch_set_all_LEDs(self):
        for switch in self.switches:
            self._gui_switch_set_LEDs(switch)

    def _gui_connector_x_positions(self, group_num_connectors):
        if 'connector_spacing' in self.switch_display:
            connector_spacing = self.switch_display['connector_spacing']
        else:
            connector_spacing = 5
        if 'connector_group_spacing' in self.switch_display:
            connector_group_spacing = self.switch_display[
                'connector_group_spacing']
        else:
            connector_group_spacing = 10

        num_groups = len(group_num_connectors)
        num_connectors = sum(group_num_connectors)
        total_length = (num_groups - 1) * (
                connector_group_spacing - connector_spacing) + (
                               num_connectors - 1) * connector_spacing
        start_pos = 50 - total_length / 2
        x_positions = []
        group_x_positions = []
        for k, num_cons in enumerate(group_num_connectors):
            for n in range(num_cons):
                if len(x_positions) == 0:
                    x_positions.append(start_pos)
                    group_x_positions.append(start_pos)
                    continue
                if n == 0:
                    x_positions.append(
                        x_positions[-1] + connector_group_spacing)
                    group_x_positions.append(x_positions[-1])
                else:
                    x_positions.append(
                        x_positions[-1] + connector_spacing)

        return x_positions, group_x_positions

    def _gui_draw_input_connectors(self):
        if 'inputs_y' in self.switch_display:
            inputs_y = self.switch_display['inputs_y']
        else:
            inputs_y = 92

        num_groups = len(self.input_groups)
        num_connectors = len(self.input_labels)
        connectors_by_group = []
        group_num_connectors = []
        for group_label in self.input_groups:
            group_cons = []
            for inp in self.input_labels:
                if inp.startswith(group_label):
                    group_cons.append(self.inputs[inp])
            connectors_by_group.append(group_cons)
            group_num_connectors.append(len(group_cons))
        x_positions, group_x_positions = self._gui_connector_x_positions(
            group_num_connectors)
        for n, (inp, x) in enumerate(zip(self.input_labels, x_positions)):
            con = self.inputs[inp]
            self._gui_draw_connector(con, (x, inputs_y))
        radius = self.inputs[self.input_labels[0]].gui_properties['radius']
        ax = self.switch_display['ax']
        input_group_labels = []
        for lab, x in zip(self.input_groups, group_x_positions):
            input_group_labels.append(ax.text(
                x, inputs_y + 3 * radius, lab,
                ha='center', va='bottom', size=12))

    def _gui_draw_output_connectors(self):
        if 'outputs_y' in self.switch_display:
            outputs_y = self.switch_display['outputs_y']
        else:
            outputs_y = 2

        num_groups = len(self.output_groups)
        num_connectors = len(self.output_labels)
        connectors_by_group = []
        group_num_connectors = []
        for group_label in self.output_groups:
            group_cons = []
            for out in self.output_labels:
                if out.startswith(group_label):
                    group_cons.append(self.outputs[out])
            connectors_by_group.append(group_cons)
            group_num_connectors.append(len(group_cons))
        x_positions, group_x_positions = self._gui_connector_x_positions(
            group_num_connectors)
        for inp, x in zip(self.output_labels, x_positions):
            con = self.outputs[inp]
            self._gui_draw_connector(con, (x, outputs_y))
        radius = self.outputs[self.output_labels[0]].gui_properties['radius']
        ax = self.switch_display['ax']
        output_group_labels = []
        for lab, x in zip(self.output_groups, group_x_positions):
            output_group_labels.append(ax.text(
                x, outputs_y - 3 * radius, lab,
                ha='center', va='top', size=12))

    def _gui_draw_connector(self, connector, position, radius=2):
        if 'connector_radius' in self.switch_display:
            radius = self.switch_display['connector_radius']
        gui_properties = connector.gui_properties
        gui_properties.update({
            'position': position,
            'radius': radius
        })
        ax = self.switch_display['ax']
        gui_properties['body'] = ax.add_patch(
            Circle(position,
                   radius,
                   edgecolor=self.CONNECTOR_LINE_COLOR,
                   facecolor=self.CONNECTOR_COLOR,
                   fill=True,
                   lw=1.5)
        )

        if isinstance(connector, ArduinoSwitchControlInput):
            label_y_pos = position[1] + 1.2 * radius
            va = 'bottom'
        else:
            label_y_pos = position[1] - 1.3 * radius
            va = 'top'

        if connector.group is not None:
            label = connector.label.split('.')[-1]
        else:
            label = connector.label

        gui_properties['label'] = ax.text(position[0],
                                          label_y_pos,
                                          label,
                                          ha='center', va=va,
                                          size=10)
        gui_properties['label_y_pos'] = label_y_pos

    def _gui_draw_connection(self, connection, mid_y=None):
        ax = self.switch_display['ax']
        if isinstance(connection.start, ArduinoSwitchControlConnector):
            gui_properties_start = connection.start.gui_properties
            start_x, start_y = gui_properties_start['position']
        elif isinstance(connection.start, ArduinoSwitchControlSwitchConnector):
            gui_properties_start = connection.start.switch.gui_properties
            if isinstance(connection.start, ArduinoSwitchControlSwitchInput):
                start_x, start_y = gui_properties_start['in_pos']
            elif isinstance(connection.start,
                            ArduinoSwitchControlSwitchOutput):
                state = connection.start.state
                start_x, start_y = gui_properties_start[f'out{state}_pos']
        else:
            raise Exception
        if isinstance(connection.end, ArduinoSwitchControlConnector):
            gui_properties_end = connection.end.gui_properties
            end_x, end_y = gui_properties_end['position']
        elif isinstance(connection.end, ArduinoSwitchControlSwitchConnector):
            gui_properties_end = connection.end.switch.gui_properties
            if isinstance(connection.end, ArduinoSwitchControlSwitchInput):
                end_x, end_y = gui_properties_end['in_pos']
            elif isinstance(connection.end, ArduinoSwitchControlSwitchOutput):
                state = connection.end.state
                end_x, end_y = gui_properties_end[f'out{state}_pos']
        else:
            raise Exception

        if mid_y is None:
            mid_y = (start_y + end_y) / 2

        x_vals = [start_x] * 2 + [end_x] * 2
        y_vals = [start_y] + [mid_y] * 2 + [end_y]

        connection.gui_properties['line'], = ax.plot(x_vals, y_vals,
                                                     ls='solid', lw=1,
                                                     color=(0., 0., 0.))

    def _gui_draw_all_connections(self):
        if 'connections_mid_y' in self.switch_display:
            connections_mid_y = self.switch_display['connections_mid_y']
        else:
            connections_mid_y = {}
            # have format (,), entries connector label,
            # switch label or (switch label,state)
        for con in self.connections:
            con_label = con.get_label()
            if con_label[::-1] in connections_mid_y:
                mid_y = connections_mid_y[con_label[::-1]]
            elif con_label in connections_mid_y:
                mid_y = connections_mid_y[con_label]
            else:
                mid_y = None
            self._gui_draw_connection(con, mid_y)

    def _gui_draw_active_routes(self):
        active_routes, active_connections = self.active_routes(
            return_active_connections=True
        )
        for inp in self.inputs.values():
            inp.gui_properties['body'].set_facecolor(self.CONNECTOR_COLOR)
            inp.gui_properties['body'].set_edgecolor(self.CONNECTOR_LINE_COLOR)
        for out in self.outputs.values():
            out.gui_properties['body'].set_facecolor(self.CONNECTOR_COLOR)
            out.gui_properties['body'].set_edgecolor(self.CONNECTOR_LINE_COLOR)
        for con in self.connections:
            line = con.gui_properties['line']
            if con in active_connections:
                line.set_color(self.CONNECTION_COLOR_ACTIVE)
                line.set_linewidth(1.5)
                if isinstance(con.start, ArduinoSwitchControlConnector):
                    con.start.gui_properties['body'].set_facecolor(
                        self.CONNECTOR_COLOR_ACTIVE)
                    con.start.gui_properties['body'].set_edgecolor(
                        self.CONNECTOR_LINE_COLOR_ACTIVE)
                if isinstance(con.end, ArduinoSwitchControlConnector):
                    con.end.gui_properties['body'].set_facecolor(
                        self.CONNECTOR_COLOR_ACTIVE)
                    con.end.gui_properties['body'].set_edgecolor(
                        self.CONNECTOR_LINE_COLOR_ACTIVE)
            else:
                line.set_color((0., 0., 0.))
                line.set_linewidth(1.)

    # Class methods
    # -------------

    # - Methods to handle serial ports
    #   ----------------------------

    _open_ports = {}

    @classmethod
    def open_ports(cls) -> dict:
        """ Returns dictionary of (possibly) open serial ports

        :return: copy of _open_ports
        :rtype: dict
        """
        return cls._open_ports.copy()

    @classmethod
    def add_port(cls, port: str, ser: serial.Serial):
        """Add port so _open_ports

        :param port: port of USB connection to Arduino, like 'COM5'
        :param ser: serial.Serial of the serial connection
        """
        cls._open_ports[port] = ser

    @classmethod
    def get_serial(cls, port: str) -> serial.Serial:
        """Get serial.Serial object by port

        :param port: label of port

        :return: serial.Serial object of port
        :rtype: serial.Serial
        """
        if port in cls._open_ports:
            return cls._open_ports[port]
        else:
            return None

    @classmethod
    def remove_port(cls, port):
        if port in cls._open_ports:
            if cls._open_ports[port].is_open:
                cls._open_ports[port].close()
            del cls._open_ports[port]

    # Static methods
    # --------------

    @staticmethod
    def get_switch_states_on_route(route):
        switches_states = []
        for connection in route.connections:
            if isinstance(connection.start, ArduinoSwitchControlSwitchOutput):
                switches_states.append((connection.start.switch,
                                        connection.start.state))
            if isinstance(connection.end, ArduinoSwitchControlSwitchOutput):
                switches_states.append((connection.end.switch,
                                        connection.end.state))
        return switches_states


# Classes for components of switch box
# -----------------------------------

class ArduinoSwitchControlObject:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"ArduinoSwitchControlObject {self.label}"

    def __str__(self):
        return self.label


class ArduinoSwitchControlConnector(ArduinoSwitchControlObject):
    def __init__(self, label, group=None):
        super().__init__(label)
        self.group = group
        self.connections = []
        self.gui_properties = {}

    def __repr__(self):
        return f"ArduinoSwitchControlConnector {self.label}"


class ArduinoSwitchControlInput(ArduinoSwitchControlConnector):
    def __init__(self, label, group=None):
        super().__init__(label, group)

    def __repr__(self):
        return f"ArduinoSwitchControlInput {self.label}"


class ArduinoSwitchControlOutput(ArduinoSwitchControlConnector):
    def __init__(self, label, group=None):
        super().__init__(label, group)

    def __repr__(self):
        return f"ArduinoSwitchControlOutput {self.label}"


class ArduinoSwitchControlSwitch(ArduinoSwitchControlObject):
    def __init__(self, label, id, state=None):
        super().__init__(label)
        self.id = id
        self.connections = []
        self.plot_position = None
        self.routes = {}

        self.orientation = 0

        self.input = ArduinoSwitchControlSwitchInput(self)
        self.output = [ArduinoSwitchControlSwitchOutput(self, 0),
                       ArduinoSwitchControlSwitchOutput(self, 1)]
        self.state = state  # check if state None,0,1

        self.state_checked = False
        self.mode = None

        self.gui_properties = {}

    def __repr__(self):
        return (f"ArduinoSwitchControlSwitch {self.label} with "
                f"id {self.id}")

    def __str__(self):
        return str(self.label)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state is None:
            self._leds = (None, None)
        elif state == 0:
            self._leds = (1, 0)
        elif state == 1:
            self._leds = (0, 1)
        elif state == 2:
            self._leds = (1, 1)
        else:
            raise ValueError("State not understood.")
        self._state = state

    @property
    def leds(self):
        return self._leds

    @leds.setter
    def leds(self, leds):
        if (*leds,) == (None, None):
            self._state = None
        elif (*leds,) == (1, 0):
            self._state = 0
        elif (*leds,) == (0, 1):
            self._state = 1
        elif (*leds,) in [(0, 0), (1, 1)]:
            self._state = 2
        else:
            raise ValueError("LED state not understood.")
        self._leds = leds


class ArduinoSwitchControlSwitchConnector:
    def __init__(self, switch):
        self.switch = switch
        self.connections = []

    def __repr__(self):
        return f"ArduinoSwitchControlSwitchConnector at switch {self.switch.label}"


class ArduinoSwitchControlSwitchInput(ArduinoSwitchControlSwitchConnector):
    def __init__(self, switch):
        super().__init__(switch)

    def __repr__(self):
        return f"ArduinoSwitchControlSwitchInput at switch {self.switch.label}."

    def __str__(self):
        return f"{self.switch.label}"


class ArduinoSwitchControlSwitchOutput(ArduinoSwitchControlSwitchConnector):
    def __init__(self, switch, state):
        super().__init__(switch)
        self.state = state

    def __repr__(self):
        return f"ArduinoSwitchControlSwitchOutput at switch {self.switch.label}:{self.state}."

    def __str__(self):
        return f"({self.switch.label},{self.state})"


class ArduinoSwitchControlConnection:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.gui_properties = {}

    def __repr__(self):
        return (f"ArduinoSwitchControlConnection from {self.start} to "
                f"{self.end}")

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

    def flip(self):
        self.start, self.end = self.end, self.start

    def get_label(self):
        if isinstance(self.start, ArduinoSwitchControlConnector):
            start_label = self.start.label
        elif isinstance(self.start, ArduinoSwitchControlSwitchInput):
            start_label = self.start.switch.label
        elif isinstance(self.start, ArduinoSwitchControlSwitchOutput):
            start_label = (self.start.switch.label, self.start.state)
        else:
            raise Exception
        if isinstance(self.end, ArduinoSwitchControlConnector):
            end_label = self.end.label
        elif isinstance(self.end, ArduinoSwitchControlSwitchInput):
            end_label = self.end.switch.label
        elif isinstance(self.end, ArduinoSwitchControlSwitchOutput):
            end_label = (self.end.switch.label, self.end.state)
        else:
            raise Exception
        return start_label, end_label


class ArduinoSwitchControlRoute:
    def __init__(self, start, end, connections):
        self.start = start
        self.end = end
        self.connections = connections

    def __repr__(self):
        return (f"ArduinoSwitchControlRoute from {self.start} to "
                f"{self.end}")

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


# Classes for customized switch boxes
# -----------------------------------

class ArduinoSwitchControlEdelweissV1(ArduinoSwitchControl):
    def __init__(self, name, port, start_serial=True):
        switches = 17
        inputs = ('I', 4)
        outputs = [('A2', 6), ('B2', 7), ('B4', 7)]
        input_connections = [('I1', 11), ('I2', 13), ('I3', 16), ('I4', 17)]
        output_connections = [((1, 1), 'A2.1'),
                              ((3, 1), 'A2.2'),
                              ((1, 0), 'A2.3'),
                              ((3, 0), 'A2.4'),
                              ((2, 0), 'A2.5'),
                              ((5, 0), 'A2.6'),
                              ((4, 1), 'B2.1'),
                              ((6, 1), 'B2.3'),
                              ((6, 0), 'B2.5'),
                              ((8, 1), 'B2.7'),
                              ((8, 0), 'B2.2'),
                              ((12, 0), 'B2.6'),
                              ((12, 1), 'B2.4'),
                              ((15, 0), 'B4.1'),
                              ((15, 1), 'B4.2'),
                              ((16, 1), 'B4.3'),
                              ((14, 1), 'B4.4'),
                              ((17, 0), 'B4.5'),
                              ((17, 1), 'B4.6'),
                              ((16, 0), 'B4.7')]
        switch_connections = [((2, 1), 1),
                              ((4, 0), 6),
                              ((5, 1), 3),
                              ((7, 0), 2),
                              ((7, 1), 4),
                              ((9, 0), 5),
                              ((9, 1), 10),
                              ((10, 0), 8),
                              ((10, 1), 12),
                              ((11, 0), 15),
                              ((11, 1), 7),
                              ((13, 0), 14),
                              ((13, 1), 9)]
        connections = (input_connections + output_connections
                       + switch_connections)

        # gui properties
        rows = [100 / 6 * (n + 1 / 2) for n in range(5)]
        scale = 100 / 6
        switch_positions = [(0.4 * scale, rows[1]),
                            (0.75 * scale, rows[2]),
                            (1. * scale, rows[1]),
                            (2.25 * scale, rows[2]),
                            (1.5 * scale, rows[2]),
                            (2.5 * scale, rows[1]),
                            (1.75 * scale, rows[3]),
                            (3 * scale, rows[1]),
                            (3 * scale, rows[3]),
                            (3.25 * scale, rows[2]),
                            (2.5 * scale, rows[4]),
                            (3.5 * scale, rows[1]),
                            (3.5 * scale, rows[4]),
                            (4.4 * scale, rows[3]),
                            (4.0 * scale, rows[2]),
                            (4.8 * scale, rows[2]),
                            (5.3 * scale, rows[1])]

        switches_mirrored = [True, True, True, True, True,
                             True, False, False, False, False,
                             True, True, True, True, False,
                             True, False]

        connections_mid_y = {('I2', 13): 81,
                             ('I3', 16): 85,
                             ('I4', 17): 89,
                             ((11, 0), 15): 64,
                             ((13, 0), 9): 68,
                             ((13, 1), 14): 68,
                             ((7, 0), 2): 52,
                             ((7, 1), 4): 52,
                             ((9, 0), 5): 48,
                             ((9, 1), 10): 48,
                             ((2, 1), 1): 35,
                             ((2, 0), 'A2.5'): 35,
                             ((5, 1), 3): 31,
                             ((15, 0), 'B4.1'): 13,
                             ((15, 1), 'B4.2'): 17,
                             ((14, 1), 'B4.4'): 28,
                             ((16, 0), 'B4.7'): 35,
                             ((6, 1), 'B2.3'): 16.5,
                             ((6, 0), 'B2.5'): 18,
                             ((8, 1), 'B2.7'): 19,
                             ((8, 0), 'B2.2'): 15,
                             ((12, 0), 'B2.6'): 11,
                             ((12, 1), 'B2.4'): 13,
                             ((1, 1), 'A2.1'): 18,
                             ((1, 0), 'A2.3'): 18,
                             ((3, 1), 'A2.2'): 15,
                             ((3, 0), 'A2.4'): 13
                             }

        outputs_y = 8
        inputs_y = 92
        connector_spacing = 4.3
        connector_group_spacing = 10
        connector_radius = 1.6

        super().__init__(name, port, switches, inputs, outputs, connections,
                         start_serial=start_serial,
                         switch_positions=switch_positions,
                         outputs_y=outputs_y,
                         inputs_y=inputs_y,
                         connector_spacing=connector_spacing,
                         connector_group_spacing=connector_group_spacing,
                         connector_radius=connector_radius,
                         switches_mirrored=switches_mirrored,
                         connections_mid_y=connections_mid_y
                         )
