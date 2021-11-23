import numpy as np
import time

from qcodes.utils import validators as vals
from qcodes.instrument_drivers.QDevil.QDevil_QDAC import QDac


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

    def set_smooth(self, voltage_dict):
        v_sweep = {}
        for ch_number, voltage in voltage_dict.items():  # generate lists of
            # V to apply over time
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
        for step in range(N_steps):
            for ch_number, v_list in v_sweep.items():
                if step < len(v_list):
                    self.channels[ch_number].v(v_list[step])
            time.sleep(self.parameters['smooth_timestep']())


# channel_map_qdac = {i: f'fluxline{i + 1}' for i in range(6)}
# qdac = QDacSmooth('qdac', 'COM3', channel_map_qdac)
# fluxlines_dict = {f'qb{i + 1}': qdac.parameters[f"volt_{channel_map_qdac[i]}"]
#     for i in range(6)}
