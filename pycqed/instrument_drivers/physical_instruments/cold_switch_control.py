import serial
import time
import os
import struct
import matplotlib.pyplot as plt
import scipy.optimize
from datetime import datetime
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

def bits_to_byte(bits):
    res = 0
    for b in bits[::-1]:
        res = (res << 1) | b
    return res

IODIRA = 0x00
IODIRB = 0x01
OLATA = 0x14
OLATB = 0x15

class ColdSwitchController(Instrument):

    SOURCE_SINK_TABLE = {
           5: (1, 2),
           6: (2, 3),
           1: (3, 4),
           2: (4, 5),
           3: (5, 6),
           4: (6, 1)
           }
    NB_SWITCH_POS = 6
    NB_COLD_SWITCHES = 4
    LOG_FILENAME = 'cold_switch_{}_history.log'

    def __init__(self, name, port, power_supply: str, log_dirpath: str):
        super().__init__(name)
        self.port = serial.Serial(port, 115200, timeout=5, writeTimeout=0)
        self.debug = False
        self.power_supply = self.find_instrument(power_supply)
        self.log_dirpath = log_dirpath
        if not self.ping():
            self.port.close()
            raise ConnectionError('Failed to connect to device.')

        print('Connected...')

        # all i/o expander pins set as outputs & low; this initialization
        # should be done in firmware later
        self._write_to_ioexp(0x00, IODIRA, 0x00)
        self._write_to_ioexp(0x00, IODIRB, 0x00)
        self._write_to_ioexp(0x01, IODIRA, 0x00)
        self._write_to_ioexp(0x01, IODIRB, 0x00)
        self._write_to_ioexp(0x02, IODIRA, 0x00)
        self._write_to_ioexp(0x02, IODIRB, 0x00)

        self._write_to_ioexp(0x00, OLATA, 0x00)
        self._write_to_ioexp(0x00, OLATB, 0x00)
        self._write_to_ioexp(0x01, OLATA, 0x00)
        self._write_to_ioexp(0x01, OLATB, 0x00)
        self._write_to_ioexp(0x02, OLATA, 0x00)
        self._write_to_ioexp(0x02, OLATB, 0x00)

        for idx in range(1, self.NB_COLD_SWITCHES + 1):
            self.add_parameter(
                f"switch_{idx}_mode",
                vals=vals.Enum(*range(self.NB_SWITCH_POS + 1)),
                set_cmd=lambda val, idx=idx:
                    self.change_cold_switch_channel(new_channel=val,
                                                    switch_idx=idx),
                get_cmd=lambda idx=idx:
                    self.get_cold_switch_channel(switch_idx=idx),
            )

    def close(self):
        self.port.close()

    def ping(self):
        data = ':p'.encode()
        self.port.write(data)
        resp = self.port.read(1)
        return (resp.decode() == 'P')

    def _turn_on_failsafe(self, n_pulses):
        if self.debug:
            print(f'setting failsafe pulse counter to {n_pulses}')
        data = ':r'.encode() + bytes([n_pulses])
        self.port.write(data)
        resp = self.port.read(1)
        if resp == 'R'.encode():
            if self.debug:
                print('success...')
            return True
        else:
            if self.debug:
                print('fail...')
                print('data read:', resp)
                print('in waiting:', self.port.inWaiting())
                print('remaining data:', self.port.read(self.port.inWaiting()))
            return False

    def _set_dac(self, dac_value):
        if self.debug:
            print(f'setting dac value to {dac_value}')
        data = ':a'.encode() + bytes([dac_value])
        self.port.write(data)
        resp = self.port.read(1)
        if resp == 'A'.encode():
            if self.debug:
                print('success...')
            return True
        else:
            if self.debug:
                print('fail...')
                print('data read:', resp)
                print('in waiting:', self.port.inWaiting())
                print('remaining data:', self.port.read(self.port.inWaiting()))
            return False

    def _read_current_ind(self):
        if self.debug:
            print(f'reading current indicator')
        data = ':c'.encode()
        self.port.write(data)
        resp = self.port.read(2)
        if resp[:1] == 'C'.encode():
            res = bool(resp[1])
            if self.debug:
                print('success...')
                print(f'result = {res}')
            return res
        else:
            if self.debug:
                print('fail...')
                print('data read:', resp)
                print('in waiting:', self.port.inWaiting())
                print('remaining data:', self.port.read(self.port.inWaiting()))
            return None

    def _write_to_ioexp(self, addr, reg, value):
        if self.debug:
            print(f'writing to i/o expander: address = {addr}, register = {reg}, value = {value}')
        data = ':e'.encode() + bytes([addr, reg, value])
        self.port.write(data)
        resp = self.port.read(1)
        if resp == 'E'.encode():
            if self.debug:
                print('success...')
            return True
        else:
            if self.debug:
                print('fail...')
                print('data read:', resp)
                print('in waiting:', self.port.inWaiting())
                print('remaining data:', self.port.read(self.port.inWaiting()))
            return False

    def _calibrate_timing(self, draw_plot = False):
        pulse_lengths = [16 * i for i in range(1, 16)]
        durations = []
        self._set_dac(64)
        print('calibration in progress...')
        for pulse_length in pulse_lengths:
            self._turn_on_failsafe(pulse_length)
            t0 = time.time()
            time.sleep(0.001)
            while self._read_current_ind():
                time.sleep(0.01)
            t1 = time.time()
            durations.append(t1 - t0)
            time.sleep(0.05)
        print('done')

        fitf = lambda x, a, b: a * x + b
        a, b = scipy.optimize.curve_fit(fitf, pulse_lengths, durations)[0]

        if draw_plot:
            plt.plot(pulse_lengths, durations, 'o')
            plt.plot([0, 255], [b, 255 * a + b], 'k--')
            plt.grid()
            plt.xlabel('pulse length [digital units]')
            plt.ylabel('pulse duration [s]')
            plt.show()

        print(f'Pulse length per digital unit = {round(1000 * a, 1)} ms')

    def _set_states(self, states_H, states_L):
        if self.debug:
            print(f'setting states to H = {states_H}, L = {states_L}')

        self._write_to_ioexp(0x00, OLATA, bits_to_byte(states_H[0:8]))
        self._write_to_ioexp(0x00, OLATB, bits_to_byte(states_L[0:8]))
        self._write_to_ioexp(0x01, OLATA, bits_to_byte(states_H[8:16]))
        self._write_to_ioexp(0x01, OLATB, bits_to_byte(states_L[8:16]))
        self._write_to_ioexp(0x02, OLATA, bits_to_byte(states_H[16:24]))
        self._write_to_ioexp(0x02, OLATB, bits_to_byte(states_L[16:24]))

    def _set_route(self, ch_source, ch_sink):
        if ch_source == ch_sink:
            raise ValueError('Source and sink channel have to be different.')
        states_H = [0] * 24
        states_L = [0] * 24
        states_H[ch_source - 1] = 1
        states_L[ch_sink - 1] = 1
        self._set_states(states_H, states_L)

    def _cold_switch_turn_off(self, channel_number, switch_idx, dac_amp=180,
                              failsafe_duration=1):
        self.power_supply.ch1.output(1)
        time.sleep(1.0)

        source, sink = self.SOURCE_SINK_TABLE[channel_number]
        # need to set dac amplitude after turning on the output of the PSU,
        # as the dac amplitude gets reset when turning on the output
        self._set_dac(dac_amp)

        idx_increment = self.NB_SWITCH_POS * (switch_idx - 1)
        self._set_route(source + idx_increment, sink + idx_increment)
        time.sleep(0.05)
        self._turn_on_failsafe(failsafe_duration)

        time.sleep(1.0)
        self.power_supply.ch1.output(0)

    def _cold_switch_turn_on(self, channel_number, switch_idx, dac_amp=180,
                             failsafe_duration=1):
        self.power_supply.ch1.output(1)
        time.sleep(1.0)

        sink, source = self.SOURCE_SINK_TABLE[channel_number]
        # need to set dac amplitude after turning on the output of the PSU,
        # as the dac amplitude gets reset when turning on the output
        self._set_dac(dac_amp)

        idx_increment = self.NB_SWITCH_POS*(switch_idx-1)
        self._set_route(source + idx_increment, sink + idx_increment)
        time.sleep(0.05)
        self._turn_on_failsafe(failsafe_duration)

        time.sleep(1.0)
        self.power_supply.ch1.output(0)

    def get_cold_switch_channel(self, switch_idx):
        with open(os.path.join(self.log_dirpath,
                               self.LOG_FILENAME.format(switch_idx)), 'r') as f:
            current_channel = int(list(f)[-1].split(' to ')[1])
        return current_channel

    def change_cold_switch_channel(self, new_channel, switch_idx):
        old_channel = self.get_cold_switch_channel(switch_idx)
        if old_channel == new_channel:
            return
        with open(os.path.join(self.log_dirpath,
                               self.LOG_FILENAME.format(switch_idx)), 'a') as f:
            f.write(datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
                    + f': {old_channel} to {new_channel}\n')
        self._cold_switch_turn_off(
            channel_number=old_channel, switch_idx=switch_idx)
        time.sleep(0.5)
        self._cold_switch_turn_on(
            channel_number=new_channel, switch_idx=switch_idx)


