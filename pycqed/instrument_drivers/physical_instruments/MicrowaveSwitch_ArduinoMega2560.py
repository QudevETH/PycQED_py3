from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import numpy as np
import telnetlib
import serial
import time


class MicrowaveSwitch_SP6T_ArduinoMega2560(Instrument):
    '''
    QCodes driver for a SP6T switch controlled via an Arduino Mega 2560
    '''

    def __init__(self, name, address):
        Instrument.__init__(self, name)
        self.address = address

        self.port = serial.Serial(address, 57600)

        if self.port.read(5) != 'READY'.encode():
            raise ConnectionError('failed to connect to Arduino')

        self.add_parameter(
            'switch_mode',
            label = f'switch_mode of {self.name}',
            vals = vals.Enum(0, 1, 2, 3, 4, 5, 6),
            set_cmd = self._set_switch,
            get_cmd = self._get_switch,
            docstring = 'possible values: 0 (no channel connected), 1-6'
            )


    def _set_switch(self, val):
        for i in range(1, 7):
            if i == val:
                self.port.write('H'.encode() + bytes([i]))
            else:
                self.port.write('L'.encode() + bytes([i]))

        time.sleep(0.2)

        if val != self._get_switch():
            raise ValueError('Switch did not switch into the specified state.')


    def _get_switch(self):
        lst = []
        for i in range(1, 7):
            self.port.write('R'.encode() + bytes([i]))
            lst.append(int(self.port.read()[0]))

        if sum(lst) == 5:
            return lst.index(0) + 1
        elif sum(lst) == 6:
            return 0
        else:
            raise ValueError('Unexpected reading from indicator output!')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._set_switch(0)
        self.port.close()
