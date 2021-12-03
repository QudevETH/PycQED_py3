"""Qcodes driver for R&S NGE100B series.

The only driver currently implemented is that of the NGE102B. The NGE103B is
probably very similar except that it has 3 channels instead of 2.
"""

from functools import partial
import io
from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Numbers, Enum

class WrongInstrumentError(Exception):
    """Error raised if the connected VISA instrument is not a R&S NGE100B."""
    pass

class NGE100Channel(InstrumentChannel):
    """Instrument channel for R&S NGE100B series instruments."""

    def __init__(self, parent:VisaInstrument, channel:int):
        """
        Arguments:
            parent: Parent instrument.
            channel: Channel number.
        """
        
        if channel < 1 or channel > parent.NB_CHANNELS:
            raise ValueError(f"Invalid channel '{channel}', channels can "\
                             f"only be between 1 and {parent.NB_CHANNELS}.")

        super().__init__(parent, name=f"ch{channel}")
        self.channel = channel

        self.add_channel_parameter(
            name="voltage",
            get_cmd="VOLTage?",
            set_cmd="VOLTage {}",
            get_parser=float,
            unit="V",
            vals=Numbers(0, 32.0),
            docstring="This command defines or queries the voltage value." \
                      "(adjustable in 10 mV steps)."
        )

        self.add_channel_parameter(
            name="current",
            get_cmd="CURRent?",
            set_cmd="CURRent {}",
            get_parser=float,
            unit="A",
            vals=Numbers(0, 3.0),
            docstring="This command defines or queries the current value."
        )

        self.add_channel_parameter(
            name="output",
            get_cmd="OUTPut?",
            set_cmd="OUTPut {}",
            get_parser=int,
            vals=Enum(0, 1, "OFF", "ON"),
            docstring="This command defines or queries for output state."
        )

        self.add_channel_parameter(
            name="measured_voltage",
            get_cmd="MEASure:VOLTage?",
            get_parser=float,
            docstring="This command queries the measured voltage value."
        )

        self.add_channel_parameter(
            name="measured_current",
            get_cmd="MEASure:CURRent?",
            get_parser=float,
            unit="A",
            docstring="This command queries the measured current value."
        )

        self.add_channel_parameter(
            name="measured_power",
            get_cmd="MEASure:POWer?",
            unit="W",
            get_parser=float,
            docstring="This command queries the measured power value."
        )

    def add_channel_parameter(self, name:str, **kwargs):
        """Builds a command to control a parameter of the channel.
        
        To set any value on a channel of this instrument, one must first select
        the channel, and then read/set the value.
        This function will:
        1. Create a private parameter for the specified command.
        2. Create a "regular" parameter that will under the hood select the
           correct channel before getting/settings the parameter value.
        """

        private_param = f"_{name}"

        private_get_cmd = kwargs.pop("get_cmd", None)
        private_get_parser = kwargs.pop("get_parser", None)
        private_set_cmd = kwargs.pop("set_cmd", False)
        
        self.add_parameter(
            name=private_param,
            get_cmd=private_get_cmd,
            set_cmd=private_set_cmd,
            get_parser=private_get_parser,
        )

        get_cmd = None
        set_cmd = False

        if private_get_cmd:
            get_cmd = partial(self._get_channel_parameter_cmd, private_param)

        if private_set_cmd:
            set_cmd = partial(self._set_channel_parameter_cmd, private_param)

        self.add_parameter(
            name=name,
            get_cmd=get_cmd,
            set_cmd=set_cmd,
            **kwargs
        )

    def _get_channel_parameter_cmd(self, name):
        self.parent._select_channel(self.channel)
        return getattr(self, name).get()

    def _set_channel_parameter_cmd(self, name, value):
        self.parent._select_channel(self.channel)
        return getattr(self, name).set(value)

class NGE102B(VisaInstrument):
    """Qcodes driver for R&S NGE102B - 2-channel programmable DC source.

    Check the manual for the source of implementation details:
    https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_common_library/dl_manuals/gb_sg/nge/NGE100_User_Manual_en_04_Web.pdf
    Relevant sections:
    * 6.2
    * 6.3
    * 6.4.1.1
    * 6.4.1.3
    * 6.6.1

    Example::

        from pycqed.instrument_drivers.physical_instruments.rohde_schwartz_nge102b import NGE102B
        nge = NGE102B("NGE102B", address="TCPIP::172.23.121.65::INST")

        # You can also use 'ch2' instead
        nge.ch1.voltage(3.0)
        nge.ch1.current(1.0)
        nge.ch1.output(1)
    """

    NB_CHANNELS = 2


    def __init__(self, name, address:str=None):
        """
        Arguments:
            name: Name of the instrument.
            address: IP address of the device, e.g. ``TCPIP::192.1.2.3::INST``.
        """
        super().__init__(name, address=address)

        # Check that the accessed device is of the correct kind
        if self.IDN()["model"] != "NGE102B":
            vendor = self.IDN()["vendor"]
            model = self.IDN()["model"]
            raise WrongInstrumentError("The connection to a VISA instrument "\
                f"at {address} could be established, but this instrument is "
                f"not a R&S NGE102B but a '{vendor} {model}'.")

        self.add_parameter(
            name="system_options",
            get_cmd="SYSTem:OPTion?",
            set_cmd=False,
            docstring=("This command returns the list of options installed on "
                       "the instrument.")
        )

        self.add_parameter(
            name="_select_channel",
            get_cmd="INSTrument?",
            set_cmd="INSTrument:NSELect {}",
            docstring="Internally used to select the instrument channel"
        )

        # Adding channels
        self.add_submodule("ch1", NGE100Channel(self, 1))
        self.add_submodule("ch2", NGE100Channel(self, 2))

    def get_screenshot(self) -> io.BytesIO:
        """Returns a screenshot of the instument screen.
        
        Example::

            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            nge = NGE102B(...)

            # Convert bytes to BMP image
            image = mpimg.imread(nge.get_screenshot(), format='bmp')

            # Plot
            plt.imshow(image)
            plt.show()

            # Save to file
            with open("./image.bmp", "wb") as file:
                file.write(image)
        """

        # Need to run low-level VISA query, we get an ASCII encoding error if we
        # try to query this as a qcodes parameter
        screenshot = self.visa_handle.query_binary_values(
            "HCOPy:DATA?", datatype="B",container= bytearray
        )
        return io.BytesIO(screenshot)
