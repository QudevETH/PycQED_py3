"""Qcodes driver for R&S NGE100B series.

The only driver currently tested is that of the NGE102B. The NGE103B probably
also works as it has 3 channels instead of 2.
"""


from abc import ABC, abstractproperty
from functools import partial, wraps
import io
from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers, Enum


def simulation_patch(return_value):
    """Decorator to bypass a method and return another value for simulations.

    This can be use to decorate methods of :class:`NGE100Channel` or
    :class:`NGE100Base` (and derived classes).
    """

    def _decorator(method):
        @wraps(method)
        def _wrapper(self, *args, **kwargs):
            if (hasattr(self, "virtual") and self.virtual) or \
               (self.parent and hasattr(self.parent, "virtual") and self.parent.virtual):
                return return_value
            else:
                return method(self, *args, **kwargs)
        return _wrapper
    return _decorator


class WrongInstrumentError(Exception):
    """Error raised if the connected VISA instrument is not a R&S NGE100B."""
    pass


class NGE100Channel(InstrumentChannel):
    """Instrument channel for R&S NGE100B series instruments."""

    def __init__(self, parent:'NGE100Base', channel:int):
        """
        Arguments:
            parent: Parent instrument.
            channel: Channel number.
        """

        if channel < 1 or channel > parent.nb_channels:
            raise ValueError(f"Invalid channel '{channel}', channels can "
                             f"only be between 1 and {parent.NB_CHANNELS}.")

        super().__init__(parent, name=f"ch{channel}")
        self.channel = channel

        # Store dummy values for simulation
        if self.parent.virtual:
            self._sim_parameters = {}

        self.add_parameter(
            name="voltage",
            get_cmd=partial(self._get_channel_parameter, "VOLTage?"),
            set_cmd=partial(self._set_channel_parameter, "VOLTage {}"),
            get_parser=float,
            unit="V",
            vals=Numbers(0, 32.0),
            docstring="The voltage is adjustable in 10 mV steps. If the set "
                      "value is not on the 10 mV grid, it will be rounded to "
                      "the closest possible value "
                      "(e.g.`` 0.014 -> 0.01``; ``0.015 -> 0.02``)."
        )

        self.add_parameter(
            name="current_limit",
            get_cmd=partial(self._get_channel_parameter, "CURRent?"),
            set_cmd=partial(self._set_channel_parameter, "CURRent {}"),
            get_parser=float,
            unit="A",
            vals=Numbers(0, 3.0),
            docstring="Note that the actual limit may be lower as the maximum "
                      "output power is 33.6 W per channel."
        )

        self.add_parameter(
            name="output",
            get_cmd=partial(self._get_channel_parameter, "OUTPut?"),
            set_cmd=partial(self._set_channel_parameter, "OUTPut {}"),
            get_parser=int,
            vals=Enum(0, 1, "OFF", "ON"),
            docstring="Enable or disable the channel output."
        )

        self.add_parameter(
            name="measured_voltage",
            get_cmd=partial(self._get_channel_parameter, "MEASure:VOLTage?"),
            get_parser=float,
            unit="V",
            docstring="(readonly)"
        )

        self.add_parameter(
            name="measured_current",
            get_cmd=partial(self._get_channel_parameter, "MEASure:CURRent?"),
            get_parser=float,
            unit="A",
            docstring="(readonly)"
        )

        self.add_parameter(
            name="measured_power",
            get_cmd=partial(self._get_channel_parameter, "MEASure:POWer?"),
            get_parser=float,
            unit="W",
            docstring="(readonly)"
        )

        if self.parent.virtual:
            self.add_parameter(
                name="simulated_output_current",
                parameter_class=ManualParameter,
                unit="A",
                initial_value=0.0,
                vals=Numbers(0, 3.0),
                docstring="This value will be forwarded to ``measured_current``. "
                          "Only available for virtual instruments."
            )

    def _param_from_visa_cmd(self, visa_cmd:str) -> str:
        """Util function to extract a parameter name from a visa command."""

        # Remove ?, {} and trailing whitespaces
        return visa_cmd.replace("?", "").replace("{}", "").strip()

    def _get_simulated_value(self, visa_cmd:str):
        visa_cmd = self._param_from_visa_cmd(visa_cmd)

        if visa_cmd == "MEASure:VOLTage":
            # Query the value set by the user instead
            visa_cmd = visa_cmd.replace("MEASure:", "")
        elif visa_cmd == "MEASure:CURRent":
            return self.simulated_output_current()
        elif visa_cmd == "MEASure:POWer":
            return self.measured_voltage() * self.measured_current()

        return self._sim_parameters.get(
                visa_cmd, 0.0
            )

    def _get_channel_parameter(self, visa_cmd:str):
        if self.parent.virtual:
            return self._get_simulated_value(visa_cmd)
        else:
            self.parent.select_channel(self.channel)
            return self.ask_raw(visa_cmd)

    def _set_channel_parameter(self, visa_cmd:str, value):
        if self.parent.virtual:
            self._sim_parameters[self._param_from_visa_cmd(visa_cmd)] = value
        else:
            self.parent.select_channel(self.channel)
            self.write_raw(visa_cmd.format(value))


class NGE100Base(VisaInstrument, ABC):
    """Base Qcodes driver for R&S NGE100 series (abstract class).

    This base class contains the common logic for the NGE100 series, children
    classes merely need to define the number of channels and the model name.

    Check the manual for the source of implementation details:
    https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_common_library/dl_manuals/gb_sg/nge/NGE100_User_Manual_en_04_Web.pdf
    Relevant sections:
    * 6.2
    * 6.3
    * 6.4.1.1
    * 6.4.1.3
    * 6.6.1
    """

    # As recommended in doc of abstractproperty, @property is also applied
    @property
    @abstractproperty
    def nb_channels(self):
        """Number of instrument channels."""

    @property
    @abstractproperty
    def model_name(self):
        """Instrument model name. Used for checking if the connected instrument
        is of the expected model."""

    def __init__(self, name, address:str, virtual=False):
        """
        Arguments:
            name: Name of the instrument.
            address: IP address of the device, e.g. ``TCPIP::192.1.2.3::INST``.
            virtual: Set to True to use a virtual instrument, mocking all
                instruments parameters and methods.
        """

        self.virtual = virtual
        visalib = "@sim" if self.virtual else None

        super().__init__(name, address=address, visalib=visalib)

        # Check that the accessed device is of the correct kind
        if self.IDN()["model"] != self.model_name:
            vendor = self.IDN()["vendor"]
            model = self.IDN()["model"]
            raise WrongInstrumentError("The connection to a VISA instrument "
                f"at {address} could be established, but this instrument is "
                f"not a R&S {self.model_name} but a '{vendor} {model}'.")

        # Adding channels
        for i in range(1, self.nb_channels + 1):
            self.add_submodule(f"ch{i}", NGE100Channel(self, i))

        # Print standard connection message
        self.connect_message()

    def get_idn(self):
        if self.virtual:
            return {
            'vendor': "Rohde&Schwarz",
            'model': self.model_name,
            'serial': "123456",
            'firmware': "1.0"
        }
        else:
            return super().get_idn()

    @simulation_patch(return_value="")
    def get_system_options(self):
        """Get the list of installed options on the instrument."""

        return self.ask_raw("SYSTem:OPTion?")

    @simulation_patch(return_value=io.BytesIO(bytearray()))
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

        # Need to run a binary VISA query, otherwise ASCII encoding error
        screenshot = self.visa_handle.query_binary_values(
            "HCOPy:DATA?", datatype="B", container=bytearray
        )
        return io.BytesIO(screenshot)

    @simulation_patch(return_value=None)
    def select_channel(self, channel:int):
        """Select an instrument channel, necessary prior to reading/setting
        any channel specific parameter.

        No need to call this method directly, it is internally called by
        :class:`NGE100Channel`.
        """

        if not (1 <= channel <= self.nb_channels):
            raise ValueError(f"Invalid channel '{channel}'")

        self.write_raw(f"INSTrument:NSELect {channel}")


class NGE102B(NGE100Base):
    """Qcodes driver for R&S NGE102B - 2-channel programmable DC source.

    See :class:`NGE100`.

    Example::

        from pycqed.instrument_drivers.physical_instruments.rohde_schwarz_nge100 import NGE102B
        nge = NGE102B("NGE102B", address="TCPIP::172.23.121.65::INST")

        # You can also use 'ch2' instead
        nge.ch1.voltage(3.0)
        nge.ch1.current(1.0)
        nge.ch1.output(1)
    """

    @property
    def nb_channels(self):
        return 2

    @property
    @simulation_patch(return_value="Simulated NGE102B")
    def model_name(self):
        return "NGE102B"


class NGE103B(NGE100Base):
    """Qcodes driver for R&S NGE103B - 3-channel programmable DC source.

    See :class:`NGE100`.

    .. warning::

        This instrument driver has not been tested. Only the 2-channel version
        was tested, but it is very likely that the driver works as well for the
        3-channel instrument.
    """

    @property
    def nb_channels(self):
        return 3

    @property
    @simulation_patch(return_value="Simulated NGE103B")
    def model_name(self):
        return "NGE103B"
