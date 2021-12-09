import numpy as np


class ArduinoSwitchControlObject:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"ArduinoSwitchControlObject {self.label}"


class ArduinoSwitchControlConnector(ArduinoSwitchControlObject):
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
        return self._state


class ArduinoSwitchControlConnection:
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
        self._start, self._end = self._end, self._start


class ArduinoSwitchControlRoute:
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
