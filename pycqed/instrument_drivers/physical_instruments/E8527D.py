from qcodes.instrument_drivers.agilent import E8527D as E8527D
from typing import Any


class Agilent_E8527D(E8527D.Agilent_E8527D):
    """
    Modified version of the QCoDeS Agilent_E8527D driver with higher maximal
    power and with parameter pulsemod_state for using Pulse Modulation.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Allow powers up to 25 dBm
        self._max_power = 25
        self.power.vals.valid_values = (self._min_power, self._max_power)
        # Add parameter pulsemod_state
        self.add_parameter(
            'pulsemod_state',
            label='Pulse Modulation',
            get_cmd=':OUTP:MOD?',
            set_cmd=':OUTP:MOD {}',
            val_mapping=E8527D.create_on_off_val_mapping(on_val='1',
                                                         off_val='0'))
