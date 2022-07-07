import numpy as np
import time

from pyvisa import VisaIOError
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.QDevil.QDevil_QDAC import QDac, Mode
import logging
log = logging.getLogger(__name__)

class QDacSmooth(QDac):
    """
    Driver for the QDevil QDAC with the ability to set voltages gradually/
    smoothly.

    The voltages can be set in a different number of ways. Below is a
    list of recommended ways in which fluxline/channel 17 (with "index" 16)
    can be set to 0.1V. These examples rely on the channel with index 16 being
    called "fluxline17" in channel_map (e.g. channel_map =
    {..., 16: "fluxline17", ...}).
        QDacSmooth.parameters["volt_fluxline17"](0.1)
        QDacSmooth.volt_fluxline17(0.1)
        QDacSmooth.set_smooth({"volt_fluxline17": 0.1})
        QDacSmooth.set_smooth({16: 0.1})
    In case the voltage should not be set smooth but should be set immediately
    (e.g. for faster voltage setting), the qcodes method QDacSmooth.ch17.v(0.1),
    can be used. However, this will result in an immediate change of the voltage
    (i.e. the voltage will not be set smooth) and is therefore not generally
    recommended.

    Recommended ways to get/read the voltage from fluxline/channel 17 (with
    "index" 16) are:
        QDacSmooth.parameters["volt_fluxline17"]()
        QDacSmooth.volt_fluxline17()
        QDacSmooth.ch17.v()

    Recommended ways to get/read the voltage from all fluxlines/channels (this
    will give you a voltage dict with all fluxline voltages that can also be
    directly passed into QDacSmooth.set_smooth()):
        QDacSmooth.get_fluxline_voltages() (outputs {"fluxline1": 0.1, ...})
        QDacSmooth.get_channel_voltages()  (outputs {0: 0.1, ...})
    """

    def __init__(self, name, port, channel_map):
        super().__init__(name, port, update_currents=False)
        self.channel_map = channel_map

        self.add_parameter('smooth_timestep', unit='s',
                           label="Delay between sending the write commands"
                                 "when changing the voltage smoothly",
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(0.002, 1), initial_value=0.01)

        for ch_number, ch_name in self.channel_map.items():
            stepname = f"volt_{ch_name}_step"
            self.add_parameter(stepname, unit='V',
                               label="Step size when changing the voltage " +
                                     f"smoothly on module {ch_name}",
                               get_cmd=None, set_cmd=None,
                               vals=vals.Numbers(0, 20), initial_value=0.001)

            self.add_parameter(name=f"volt_{ch_name}",
                label=f"DC source voltage on channel {ch_name}", unit='V',
                get_cmd=self.channels[ch_number].v,
                set_cmd=lambda val, ch_number=ch_number: self.set_smooth(
                    {ch_number: val}),
            )
        self.add_parameter('verbose',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=False)

    def set_smooth(self, voltagedict):
        """
        Set the voltages as specified in ``voltagedict` smoothly,
        by changing the output on each module at a rate
        ``volt_#_step/smooth_timestep``.

        Args:
            voltagedict (Dict[float]): A dictionary where keys are names (from
                the channel map) or module slot numbers (starting at 0) where
                values are the desired output voltages.
                Example:
                    {"fluxline1": 0.1, "fluxline2": 0.45, "fluxline15: 2.3, ...}
                    or
                    {0: 0.1, 1: 0.45, 14: 2.3}
        """

        def print_progress(index, total, begintime):
            if self.verbose() and total > 3:  # do not print for tiny changes
                percdone = index / total * 100
                elapsed_time = time.time() - begintime
                # The trailing spaces are to overwrite some characters in case
                # the previous progress message was longer.
                progress_message = (
                    "\r{name}\t{percdone}% completed \telapsed time: "
                    "{t_elapsed}s \ttime left: {t_left}s     ").format(
                    name=self.name,
                    percdone=int(percdone),
                    t_elapsed=round(elapsed_time, 1),
                    t_left=round((100. - percdone) / (percdone) *
                                 elapsed_time, 1) if
                    percdone != 0 else '')

                if percdone != 100:
                    end_char = ''
                else:
                    end_char = '\n'
                print('\r', progress_message, end=end_char)


        v_sweep = {}
        self._update_cache()
        # generate lists of V to apply over time
        for ch_number, voltage in voltagedict.items():
            if not isinstance(ch_number, int):
                ch_number = list(self.channel_map.keys())[
                    list(self.channel_map.values()).index(ch_number)]
            old_voltage = self.channels[ch_number].v()
            if np.abs(voltage - old_voltage) < 1e-10:
                v_sweep[ch_number] = []
            else:
                stepparam = self.parameters[
                    f"volt_{self.channel_map[ch_number]}_step"]()
                v_sweep[ch_number] = np.arange(old_voltage, voltage, np.sign(
                    voltage - old_voltage) * stepparam)
                v_sweep[ch_number] = np.append(v_sweep[ch_number],
                                               voltage)  # end on correct value
        N_steps = max([len(v_sweep[ch_number]) for ch_number in v_sweep])
        begintime = time.time()
        for step in range(N_steps):
            steptime = time.time()
            for ch_number, v_list in v_sweep.items():
                if step < len(v_list):
                    self.channels[ch_number].v(v_list[step])
            time.sleep(max(
                self.parameters['smooth_timestep']() - (time.time() - steptime),
                0))
            print_progress(step + 1, N_steps, begintime)
        self._update_cache()

    def get_fluxline_voltages(self):
        """
        Convenience method to retrieve the fluxline voltages. This will give you
        a voltage dict with all fluxline voltages that can also be directly
        passed into QDacSmooth.set_smooth().

        Returns:
            dict: Keys are the channel names provided by the user in
            channel_map, and the values are the fluxline voltages that are
            currently set.
            E.g. {"fluxline1": 0.1, "fluxline2": -1.4, "fluxline3": 0.3, ...}
        """
        return {ch_name: self.channels[chan].v()
                for ch_name, chan in zip(self.channel_map.values(),
                                         range(self.num_chans))}

    def get_channel_voltages(self):
        """
        Convenience method to retrieve the channel voltages. This will give you
        a voltage dict with all channel voltages that can also be directly
        passed into QDacSmooth.set_smooth().

        Returns:
            dict: Keys are the channel numbers (starting at 0), and the values
            are the fluxline voltages that are currently set.
            E.g. {0: 0.1, 1: -1.4, 2: 0.3, ...}
        """
        return {chan: self.channels[chan].v()
                for chan in range(self.num_chans)}

    def set_mode(self, mode: str="vhigh_ihigh"):
        """
        Method for setting the mode of the QDAC. The mode controls the voltage
        and the current ranges. The voltage range can be either set to
        [-1.1, 1.1]V (vlow) or [-10, 10]V (vhigh). The current range can be
        either set to [0, 1]uA (ilow) or [0, 100]uA (ihigh). Only the
        combinations "vhigh_ihigh", "vhigh_ilow" and "vlow_ilow" are allowed.

        Args:
            mode (str): desired voltage and current mode (either "vhigh_ihigh",
            "vhigh_ilow" or "vlow_ilow").
        """
        def _clipto(ch_number: int, value: float, min_: float, max_: float):
            errmsg = (f"Voltage of channel number {ch_number} is outside the "
                      f"bounds of the new voltage range and is therefore "
                      f"clipped.")
            if value > max_:
                log.warning(errmsg)
                return max_
            elif value < min_:
                log.warning(errmsg)
                return min_
            else:
                return value

        if mode == "vhigh_ihigh":
            qdac_mode = Mode.vhigh_ihigh
        elif mode == "vhigh_ilow":
            qdac_mode = Mode.vhigh_ilow
        elif mode == "vlow_ilow":
            qdac_mode = Mode.vlow_ilow
        else:
            qdac_mode = Mode.vhigh_ihigh
            log.warning(f"{mode} is not a valid mode. Using the default"
                            "vhigh_ihigh mode.")

        # First set the voltages to zero and then ramp up again when
        # changing modes.
        original_volt_dict = self.get_channel_voltages()
        zero_volt_dict = {ch: 0.0 for ch in range(self.num_chans)}
        self.set_smooth(zero_volt_dict)
        for ch_number, ch_volt in original_volt_dict.items():
            self.channels[ch_number].mode(qdac_mode)
            original_volt_dict[ch_number] = _clipto(
                ch_number, ch_volt,
                self.vranges[ch_number+1][qdac_mode.value.v]['Min'],
                self.vranges[ch_number+1][qdac_mode.value.v]['Max'])
        self.set_smooth(original_volt_dict)

    def _update_cache(self, *args, **kwargs):
        # Extends _update_cache() in the qcodes QDevil_QDAC driver that ensures
        # that the cache of the qcodes parameter "volt_{ch_name}" is updated.
        # In turn, this ensures that methods like snapshot() get the correct
        # values, as methods like these get the values from the cache.

        try:
            super()._update_cache(*args, **kwargs)
        except VisaIOError as e:
            # Clear buffers, this is needed to put the QDAC back to a working
            # state e.g. after a keyboard interrupt during last use
            self.device_clear()
            super()._update_cache(*args, **kwargs)
        # the default value {} in the following line prevents a crash in
        # __init__
        for ch_number, ch_name in getattr(self, 'channel_map', {}).items():
            self.parameters[f"volt_{ch_name}"].get()

    def _set_voltage(self, chan: int, v_set: float) -> None:
        # Extends _set_voltage() such that the cache of the qcodes parameter
        # "volt_{ch_name}" is set correctly even if the user would use
        # QDacSmooth.v(...) to set voltages - without this extension the cache
        # of the qcodes parameter alias would not be updated.

        super()._set_voltage(chan, v_set)
        ch_name = self.channel_map.get(chan-1, None)
        if ch_name is not None:
            self.parameters[f"volt_{ch_name}"].cache.set(v_set)