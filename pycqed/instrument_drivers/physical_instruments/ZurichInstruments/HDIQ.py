import qcodes as qc
from qcodes import validators as vals
from qcodes import Instrument
import zhinst.hdiq as hdiq
from collections import OrderedDict

class HDIQ(Instrument):
    """A qcodes instrument for controlling the switches within a ZI HDIQ.

    Attributes:
        name: Name of the instrument.
        n_channels: Number of upconversion channels of the instrument.
        instrument: Base instrument object instantiated from the ZI driver.
        modes: Possible operation modes of a channel, defining the state of the
            switches routing that channel.
    """

    def __init__(self, name, address, port=None, n_channels=4):
        """Constructor, initialises the connection with the instrument.

            Args:
                name: Name of the instrument.
                address: IP address of the instrument.
                port: TCP port. If None, will be set to the default of the ZI driver.
                n_channels: Number of upconversion channels of the instrument.
        """
        super().__init__(name)

        kw = {}
        if port is not None:
            kw['port'] = port
        self.instrument = hdiq.Hdiq(ip=address, **kw)
        self.n_channels = n_channels
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
            set_cmd=lambda x: setattr(self.instrument, 'timeout', x),
        )

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
