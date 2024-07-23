import serial
import time
import os
import matplotlib.pyplot as plt
import scipy.optimize
from datetime import datetime
from pycqed.instrument_drivers.instrument import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


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
    """Driver for QuDev cold switch controller.

    Attributes:
        SOURCE_SINK_TABLE (dict): mapping of the output channels of the
            cold-switch controller to the cold switches.
        NB_SWITCH_POS (int): number of different cold switch positions for
            one cold switch.
        NB_COLD_SWITCHES (int): number of different physical  cold switches
            which the controller is controlling.
        LOG_FILENAME (str): filename of the log file. The number of the cold
            switch is inserted at the parenthesis.
        TIME_FORMAT (str): format of the timestamp inside the log file
    """

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
    TIME_FORMAT = "%Y/%m/%d, %H:%M:%S"

    def __init__(self, name, port, power_supply: str, log_dirpath: str,
                 nb_cold_switches=None,
                 source_sink_table=None,
                 power_supply_channel: str = 'ch1'):
        """
        Initialise the cold switch controller.
        Args:
            name (str): name of the cold switch.
            port (str): serial port of the cold switch (e.g. "COM7")
            power_supply (str): instrument name of the power supply device
            log_dirpath (str): path of the log file folder
            nb_cold_switches (int): see docstring of class
            source_sink_table (int): see docstring of class
            power_supply_channel (int): channel of the power supply which
                supplies the power for switching the cold switches
        """
        super().__init__(name)
        self.port = serial.Serial(port, 115200, timeout=5, writeTimeout=0)
        self.debug = False
        self.power_supply_channel = self.find_instrument(
            power_supply).submodules[power_supply_channel]
        self.nb_cold_switches = nb_cold_switches or self.NB_COLD_SWITCHES
        self.source_sink_table = source_sink_table or self.SOURCE_SINK_TABLE
        if not isinstance(list(self.source_sink_table.values())[0], dict):
            # Only one table provided. Assume that the table is valid for all
            # switches (offset by NB_SWITCH_POS from one switch to the next).
            source_sink_table = {}
            for i in range(1, self.nb_cold_switches + 1):
                idx_increment = self.NB_SWITCH_POS * (i - 1)
                source_sink_table[i] = {
                    k: (v[0] + idx_increment, v[1] + idx_increment)
                    for k, v in self.source_sink_table.items()
                }
            self.source_sink_table = source_sink_table
        self.temperature_param = None
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

        for idx in range(1, self.nb_cold_switches + 1):
            self.add_parameter(
                f"switch_{idx}_mode",
                vals=vals.Enum(*range(self.NB_SWITCH_POS + 1)),
                set_cmd=lambda val, idx=idx:
                    self.change_cold_switch_channel(new_channel=val,
                                                    switch_idx=idx),
                get_cmd=lambda idx=idx:
                    self.get_cold_switch_channel(switch_idx=idx),
            )
        self.add_parameter(
            "dac_amp",
            vals=vals.Ints(min_value=0, max_value=255),
            initial_value=180,
            parameter_class=ManualParameter,
            docstring="Relative output amplitude of the current to the cold "
                      "switch coils. 0 is minimum and 255 is maximum."
        )
        self.add_parameter(
            "min_switching_interval",
            vals=vals.Ints(min_value=0),
            initial_value=900,
            parameter_class=ManualParameter,
            docstring="Minimum waiting time in seconds between two recurring "
                      "switching events."
        )
        self.add_parameter(
            "max_switching_temperature",
            vals=vals.Numbers(min_value=0, max_value=100e-3),
            initial_value=18e-3,
            parameter_class=ManualParameter,
            docstring="Maximum allowed temperature of self.temperature_param "
                      "in Kelvin before a switching event."
        )
        self.add_parameter(
            "failsafe_duration",
            vals=vals.Ints(min_value=1, max_value=100),
            initial_value=1,
            parameter_class=ManualParameter,
            docstring="Number of pulses for a failsafe switching event."
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

    def _cold_switch_turn_off(self, channel_number, switch_idx,
                              failsafe_duration=None):
        failsafe_duration = failsafe_duration or self.failsafe_duration()
        self.power_supply_channel.output(1)
        time.sleep(1.0)

        source, sink = self.source_sink_table[switch_idx][channel_number]
        # need to set dac amplitude after turning on the output of the PSU,
        # as the dac amplitude gets reset when turning on the output
        self._set_dac(self.dac_amp())

        self._set_route(source, sink)
        time.sleep(0.05)
        self._turn_on_failsafe(failsafe_duration)

        time.sleep(1.0)
        self.power_supply_channel.output(0)

    def _cold_switch_turn_on(self, channel_number, switch_idx,
                             failsafe_duration=None):
        failsafe_duration = failsafe_duration or self.failsafe_duration()
        self.power_supply_channel.output(1)
        time.sleep(1.0)

        sink, source = self.source_sink_table[switch_idx][channel_number]
        # need to set dac amplitude after turning on the output of the PSU,
        # as the dac amplitude gets reset when turning on the output
        self._set_dac(self.dac_amp())

        self._set_route(source, sink)
        time.sleep(0.05)
        self._turn_on_failsafe(failsafe_duration)

        time.sleep(1.0)
        self.power_supply_channel.output(0)

    def get_cold_switch_channel(self, switch_idx):
        """
        Returns current cold switch position.
        Returns the current cold switch channel (position) of cold switch
        %switch_idx% based on the latest entry of the log file.

        Args:
            switch_idx: number of the switch (1-indexed)

        Returns: the current cold switch channel (position)
        """
        with open(self._get_logfile_path(switch_idx), 'r') as f:
            current_channel = int(list(f)[-1].split(' to ')[1])
        return current_channel

    def change_cold_switch_channel(self, new_channel, switch_idx):
        """
        Switches cold switch %switch_idx% to the new_channel.
        Changes the cold switch %switch_idx% to the channel %new_channel%
        if the new requested channel differs from the current position.

        Args:
            new_channel: new switch position (1-indexed)
            switch_idx: number of the switch (1-indexed)
        """
        old_channel = self.get_cold_switch_channel(switch_idx)
        if old_channel == new_channel:
            return
        self.check_switching_allowed()
        self.write_to_logfile(switch_idx, old_channel, new_channel)
        self._cold_switch_turn_off(
            channel_number=old_channel, switch_idx=switch_idx)
        time.sleep(0.5)
        self._cold_switch_turn_on(
            channel_number=new_channel, switch_idx=switch_idx)

    def write_to_logfile(self, switch_idx, old_channel, new_channel):
        """
        Writes a switching event from cold switch %switch_idx% to the log file.
        Args:
            switch_idx: number of the switch (1-indexed)
            old_channel: old switch position (1-indexed)
            new_channel: new switch position (1-indexed)
        """
        with open(self._get_logfile_path(switch_idx), 'a') as f:
            f.write(datetime.now().strftime(self.TIME_FORMAT)
                    + f': {old_channel} to {new_channel}\n')

    def _get_logfile_path(self, switch_idx):
        return os.path.join(self.log_dirpath,
                            self.LOG_FILENAME.format(switch_idx))

    def get_last_switching_time(self, switch_idx=None):
        """
        Returns the timestamp of the last switching event.
        Returns the time when the last switching event of cold
        switch %switch_idx% happened. If switch_idx==None the time of the
        most recent switching event of any of the cold switches controlled
        by this controller is returned.
        returned.

        Args:
            switch_idx: number of the switch (1-indexed) or None to take all
            switches into account

        Returns (datetime): Timestamp of the last switching event.
        """
        switch_idx = (range(self.NB_COLD_SWITCHES) if switch_idx is None
                      else [switch_idx])
        times = []
        for i in switch_idx:
            path = self._get_logfile_path(i)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    time_str = list(f)[-1].split(': ')[0]
                    times.append(datetime.strptime(time_str, self.TIME_FORMAT))
        return max(times)

    def check_switching_allowed(self):
        """
        Checks if switching is allowed and raises an exception if not.
        Checks if the time evolved since last switching event is greater
        equal than self.min_switching_interval().
        If self.temperature_param is given, it checks additionally if the
        temperature of this parameter is below or equal
        self.max_switching_temperature.
        If one of the conditions is not satisfied, it raises an exception.
        """
        if not (interv := self.min_switching_interval()):
            return
        timedelta = datetime.now() - self.get_last_switching_time()
        if timedelta.seconds < interv:
            raise Exception(
                f'A cold switch was operated {timedelta.seconds}s ago, but '
                f'the minimum time between two switching events is {interv}s.')
        if self.temperature_param:
            temp = self.temperature_param()
            if temp is None:
                raise Exception(
                    f'Cold switches can currently not be operated because '
                    f'temperature logging data is not available.')
            if temp > (max_temp := self.max_switching_temperature()):
                raise Exception(
                    f'Cold switches can only be operated when the base '
                    f'temperature is below {max_temp/1e-3}mK, but it is '
                    f'currently at {temp/1e-3}mK.')


