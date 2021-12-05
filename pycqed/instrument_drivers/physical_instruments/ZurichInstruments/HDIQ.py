import qcodes as qc
from qcodes import validators as vals
from qcodes import Instrument
from zhinst.hdiq import Hdiq
from collections import OrderedDict

# import logging
# log = logging.getLogger(__name__)


class HDIQ(Instrument):
    """A qcodes instrument for controlling the switches within a ZI HDIQ.

    Args:
        name: Name of the instrument.
        n_channels: Number of upconversion channels of the instrument.
        address: IP address of the instrument.

    Attributes:
        name: Name of the instrument.
        n_channels: Number of upconversion channels of the instrument.
        instrument: Base instrument object instantiated from the ZI driver.
        modes: Possible operation modes of a channel, defining the state of the
            switches routing that channel.
        {channel}_mode: qcodes parameter for the state of a given channel.
    """

    def __init__(self, name, n_channels, address=None):
        super().__init__(name)
        self.n_channels = n_channels
        self.instrument = Hdiq(ip=address)
        self.modes = OrderedDict({'modulated': self.instrument.set_rf_to_exp,
                                  'calib': self.instrument.set_rf_to_calib,
                                  'spec': self.instrument.set_lo_to_exp})

        for ch in range(self.n_channels):
            self.add_parameter(
                f'UC{ch+1}_mode',
                vals=qc.validators.Enum(*self.modes.keys()),
                label=f'active path in upconversion channel {ch+1}',
                get_cmd=lambda ch=ch: list(self.modes)[
                        int(self.instrument.get_channel_status(ch+1))-1],
                set_cmd=lambda mode, ch=ch:
                    self.modes[mode](ch+1),
                docstring = "possible values: " + ', '.join(
                    [f'{m}' for m in self.modes.keys()]),
            )
        self.add_parameter(
            'timeout',
            vals=vals.Numbers(0, 120),
            get_cmd=lambda: self.instrument.timeout,
            set_cmd=None,  # seems read-only, check with ZI if needed
        )
        # self.switches = {} #Maybe for compatibility with other switch types?

    def set_switch(self, values):
        """Sets multiple switches to the given modes.
        This is for now done with multiple hardware accesses
        (seems the only option available in zhinst.hdiq).

        Args:
            values: Dict where each key is a channel name
                and value is the mode to set the switch to.
        """

        for switch, val in values.items():
            self.parameters[f'{switch}_mode'](val)

    def get_idn(self):
        idn_dict = {
            'vendor': 'ZurichInstruments',
            'model': 'HDIQ',
            # there does not seem to be a way to get a device ID or version
        }
        return idn_dict
