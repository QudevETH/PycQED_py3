from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import telnetlib
import serial
import time

class MicrowaveSwitchSP6TArduinoMega2560(Instrument):
    """
    QCodes driver for a SP6T switch controlled via an Arduino Mega 2560

    The Arduino runs a sketch monitoring serial communication from its
    virtual serial port and sets / reads its 12 GPIOs connected to the
    switch's control and monitor pins accordingly.

    Each message to the Arduino consists of two bytes - a command
    and a parameter.

    Sending the two bytes 'H<n>' sets the control pin of switch channel <n>
    high, sending 'L<n>' sets it low.

    Example: Sending 'L\x01H\x02L\x03L\x04L\x05L\x06'.encode() turns on
    channel 2 of the switch and turns all other channels off.

    Sending the two bytes 'R<n>' makes the Arduino read the state of the
    monitor pin of switch channel <n> and send it back as a single byte.
    NOTE: When the channel is on, the monitor pin is pulled to ground, i.e.
    the returned byte is 0 for channel on and 1 for channel off!
    """

    READY_MESSAGE = "READY"
    _SET_WAIT_TIME = 0.2
    NB_SWITCH_POS = 6

    def __init__(self, name, address, port=57600):
        super().__init__(name)
        self.address = address

        self.port = serial.Serial(address, port)

        if self.port.read(len(self.READY_MESSAGE)) != self.READY_MESSAGE.encode():
            raise ConnectionError("failed to connect to Arduino")

        self.add_parameter(
            "switch_mode",
            vals=vals.Enum(*range(self.NB_SWITCH_POS + 1)),
            set_cmd=self._set_switch,
            get_cmd=self._get_switch,
            docstring="possible values: 0 (no channel connected), 1-6"
            )


    def _set_switch(self, val):
        for i in range(1, self.NB_SWITCH_POS+1):
            if i == val: # set the given channel on
                self.port.write("H".encode() + bytes([i]))
            else:        # and all other channels off
                self.port.write("L".encode() + bytes([i]))

        time.sleep(self._SET_WAIT_TIME)

        # check that the states of the channels were changed correctly
        if val != self._get_switch():
            raise ValueError("Switch did not switch into the specified state.")


    def _get_switch(self):
        lst = []
        for i in range(1, self.NB_SWITCH_POS+1):
            # sequentially read the indicator states of all channels
            self.port.write("R".encode() + bytes([i]))
            lst.append(int(self.port.read()[0]))

        # channels off return indicator value 1, channels on return 0
        if sum(lst) == self.NB_SWITCH_POS-1:
            # if exactly one channel is on, return its number
            return lst.index(0) + 1
        elif sum(lst) == self.NB_SWITCH_POS:
            # if all channels are off, return 0
            # this is consistent with setting behavior (0 -> all channels off)
            return 0
        else:
            # otherwise (more than one channel is on) raise an error
            raise ValueError("Unexpected reading from indicator output!")
