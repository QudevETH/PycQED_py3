import numpy as np
import time
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.QDevil.QDevil_QDAC import QDac, Mode
import logging
log = logging.getLogger(__name__)

class QDacSmooth(QDac):
    def __init__(self, name, port, channel_map):
        super().__init__(name, port, update_currents=False)
        self.channel_map = channel_map

        self.add_parameter('smooth_timestep', unit='s',
                           label="Delay between sending the write commands"
                                 "when changing the voltage smoothly",
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(0.002, 1), initial_value=0.01)
        #         smooth_timestep = self.parameters['smooth_timestep']

        for ch_number, ch_name in self.channel_map.items():
            stepname = f"volt_{ch_name}_step"
            self.add_parameter(stepname, unit='V',
                               label="Step size when changing the voltage " + f"smoothly on module {ch_name}",
                               get_cmd=None, set_cmd=None,
                               vals=vals.Numbers(0, 20), initial_value=0.001)
            #             stepparam = self.parameters[stepname]

            self.add_parameter(name=f"volt_{ch_name}",
                label=f"DC source voltage on channel {ch_name}", unit='V',
                get_cmd=self.channels[ch_number].v,
                set_cmd=lambda val, ch_number=ch_number: self.set_smooth(
                    {ch_number: val}),
                #                 initial_value=self.qdac.getDCVoltage(key)
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
            voltagedict (Dict[float]): A dictionary where keys are module slot
                numbers or names and values are the desired output voltages.
        """

        def print_progress(index, total, begintime):
            if self.verbose() and total > 3:  # do not print for tiny changes
                percdone = index / total * 100
                elapsed_time = time.time() - begintime
                # The trailing spaces are to overwrite some characters in case the
                # previous progress message was longer.
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
                                               voltage)  # end on the correct value
        N_steps = max([len(v_sweep[ch_number]) for ch_number in v_sweep])
        begintime = time.time()
        for step in range(N_steps):
            steptime = time.time()
            for ch_number, v_list in v_sweep.items():
                if step < len(v_list):
                    self.channels[ch_number].v(v_list[step])
            # print(self.parameters['smooth_timestep']() - (time.time() - steptime))
            time.sleep(max(
                self.parameters['smooth_timestep']() - (time.time() - steptime),
                0))
            # time.sleep(self.parameters['smooth_timestep']())
            print_progress(step + 1, N_steps, begintime)
        self._update_cache()

    def set_voltage(self, chan, voltage, immediate=False):
        """Set the output voltage of a channel.

            Args:
                chan (int): Channel number of the channel to set the voltage of.
                voltage (float): The value to set the voltage to.
        """
        vals.Ints(1, self.num_chans).validate(chan)

        if immediate:
            self.channels[chan-1].v(voltage)
        else:
            self.set_smooth({chan-1: voltage})

    def get_voltage(self, chan):
        """Read the output voltage of a channel.

            Args:
                chan (int): Channel number of the channel to get the
                voltage of.

            Returns:
                The current voltage of channel ``chan`` as a ``float``.
        """
        vals.Ints(1, self.num_chans).validate(chan)

        return self.channels[chan-1].v()

    def get_current_fluxline_voltages(self):
        return {chan+1: self.channels[chan].v() for chan in range(self.num_chans)}

    def set_mode(self, mode: str="vhigh_ihigh"):
        if mode == "vhigh_ihigh":
            set_mode = Mode.vhigh_ihigh
        elif mode == "vhigh_ilow":
            set_mode = Mode.vhigh_ilow
        elif mode == "vlow_ilow":
            set_mode = Mode.vlow_ilow
        else:
            set_mode = Mode.vhigh_ihigh
            log.warning(f"{mode} is not a valid mode. Using the default"
                            "vhigh_ihigh mode.")
        for chan in range(self.num_chans):
            self.channels[chan].mode(set_mode)

    def _update_cache(self, *args, **kwargs):
        super()._update_cache(*args, **kwargs)
        # the default value {} in the following line prevents a crash in __init__
        for ch_number, ch_name in getattr(self, 'channel_map', {}).items():
            self.parameters[f"volt_{ch_name}"].get()

    def _set_voltage(self, chan: int, v_set: float) -> None:
        super()._set_voltage(chan, v_set)
        # FIXME: test whether the default values & the if are needed
        ch_name = getattr(self, 'channel_map', {}).get(chan, None)
        if ch_name is not None:
            self.parameters[f"volt_{ch_name}"].cache.set(v_set)

# channel_map_qdac = {i: f'fluxline{i + 1}' for i in range(6)}
# qdac = QDacSmooth('qdac', 'COM3', channel_map_qdac)
# fluxlines_dict = {f'qb{i + 1}': qdac.parameters[f"volt_{channel_map_qdac[i]}"]
#     for i in range(6)}
