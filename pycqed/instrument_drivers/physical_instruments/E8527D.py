from typing import Union, Tuple
from qcodes.utils import validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping
try:
    from qcodes.instrument_drivers.agilent import Agilent_E8257D
except (ImportError, ModuleNotFoundError):
    # older qcodes versions (pre 0.37)
    from qcodes.instrument_drivers.agilent import E8257D as Agilent_E8257D


class Agilent_E8527D(Agilent_E8257D.AgilentE8257D):
    """
    Modified version of the QCoDeS Agilent_E8527D driver with higher maximal
    power and with parameter pulsemod_state for using Pulse Modulation.

    See :class:`qcodes.instrument_drivers.agilent.Agilent_E8527D.AgilentE8257D`
    """

    FREQUENCY_OPTIONS = ['513', '520', '521', '532', '540', '550', '567']
    """Possible frequency options for the specific model variant.

    See the datasheet:
    https://www.keysight.com/ch/de/assets/7018-01233/configuration-guides/5989-1325.pdf
    """


    # Parameters virtual, frequency_option defined as keyword-only arguments to
    # keep backward compatibility with existing code
    def __init__(self, *args, virtual=False, frequency_option="513", **kwargs):
        """Constructor.

        Arguments:
            virtual: Set it to True for virtual (mocked) device.
            frequency_option: Useful only if ``virtual=True``, this
                corresponds to the model variant (possible values: see
                :attr:`FREQUENCY_OPTIONS`).
            args, kwargs: Same arguments as the parent class, see 
                :class:`qcodes.instrument_drivers.agilent.E8527D.Agilent_E8527D`.
        """

        self.virtual = virtual

        # Sanity checks
        if self.virtual and kwargs.get("visalib", None) is not None:
            raise ValueError("The visalib should not be changed for the "
                             "virtual E8527D.")
        if frequency_option not in self.FREQUENCY_OPTIONS:
            raise ValueError(f"Invalid parameter '{frequency_option=}'. "
                             f"Possible values: {self.FREQUENCY_OPTIONS}.")

        if self.virtual:

            kwargs["visalib"] = "@sim"

            # Store dummy (string) values for virtual instrument
            self._virtual_parameters = {
                "DIAG:CPU:INFO:OPT:DET": frequency_option,
                "FREQ:CW": "250000",
                "PHASE": "0.0",
                "POW:AMPL": "-20.0",
                "OUTP": "0",
                "OUTP:MOD": "0",
            }

        super().__init__(*args, **kwargs)
        # Allow powers up to 25 dBm
        self._max_power = 25
        self.power.vals = vals.Numbers(self._min_power, self._max_power)

        if 'status' not in self.parameters:
            # add parameter status to assure compatibility of the driver with
            # the qcodes 0.37
            self.add_parameter(
                'status',
                get_cmd=':OUTP?',
                set_cmd='OUTP {}',
                val_mapping=create_on_off_val_mapping(on_val='1',
                                                      off_val='0'))
        # Add parameter pulsemod_state
        self.add_parameter(
            "pulsemod_state",
            label="Pulse Modulation",
            get_cmd=":OUTP:MOD?",
            set_cmd="OUTP:MOD {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0")
        )
        self.add_parameter(
            "pulsemod_source",
            label="Pulse Modulation Source",
            get_cmd=":PULM:SOUR?",
            set_cmd="PULM:SOUR {}",
            vals=vals.Enum('INT', 'EXT', 'SCAL')
        )
        self.add_parameter(
            "pulsemod_state_status",
            label="Pulse Modulation Enabled",
            get_cmd=":PULM:STAT?",
            set_cmd="PULM:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            docstring=('Command for setting turning on and off the pulse '
                       'modulation. Should always be on since the pulse '
                       'modulation can be turned off by turning off only the '
                       'command pulsemod_state'))

    def ask_raw(self, cmd: str) -> str:
        """Override to bypass visa logic for virtual instruments.

        The default behavior is to store and return values from a dictionnary.
        For more complex behaviors, one can add condition depending on the input
        command.
        """

        if self.virtual:
            cmd = self._param_from_visa_cmd(cmd)
            return self._virtual_parameters.get(cmd, None)
        else:
            return super().ask_raw(cmd)

    def write_raw(self, cmd: str) -> None:
        """Override to bypass visa logic for virtual instruments.

        See :meth:`ask_raw`.
        """

        if self.virtual:
            cmd, value = self._param_from_visa_cmd(cmd, is_set_cmd=True)
            self._virtual_parameters[cmd] = value
        else:
            return super().write_raw(cmd)

    def _param_from_visa_cmd(self, cmd:str, is_set_cmd:bool=False) \
        -> Union[str, Tuple[str, str]]:
        """Util function to extract a parameter name from a visa command.

        Arguments:
            cmd: The raw visa command send by the driver.
            is_set_cmd: Whether this is a set or get command.

        Returns:
            For get commands this returns a sanitized visa command, for set
            commands the value to be set is also returned
        """

        # Remove ?, {} and trailing whitespaces
        cmd = cmd.replace("?", "").replace("{}", "").strip()

        # Seems to be necessary for OUTP
        if cmd[0] == ":":
            cmd = cmd[1:] # remove leading ":"

        if is_set_cmd:
            # Set commands are of the form "SOME:VISA:CMD set_value"
            return cmd.split(" ")
        else:
            return cmd

    def enable_pulse_modulation(self):
        """"Util that enables pulse modulation for time-domain measurements.
        To turn on the modulation, both pulsemod_state and
        pulsemod_state_status should be turned on."""
        self.pulsemod_state(1)
        self.pulsemod_source('EXT')
        self.pulsemod_state_status(1)
