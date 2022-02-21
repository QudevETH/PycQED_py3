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
            if i == val:
                self.port.write("H".encode() + bytes([i]))
            else:
                self.port.write("L".encode() + bytes([i]))

        time.sleep(self._SET_WAIT_TIME)

        if val != self._get_switch():
            raise ValueError("Switch did not switch into the specified state.")


    def _get_switch(self):
        lst = []
        for i in range(1, self.NB_SWITCH_POS+1):
            self.port.write("R".encode() + bytes([i]))
            lst.append(int(self.port.read()[0]))

        if sum(lst) == self.NB_SWITCH_POS-1:
            return lst.index(0) + 1
        elif sum(lst) == self.NB_SWITCH_POS:
            return 0
        else:
            raise ValueError("Unexpected reading from indicator output!")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._set_switch(0)
        self.port.close()

