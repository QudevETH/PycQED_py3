from qcodes.instrument_drivers.tektronix import AWG5014
from typing import Any
import numpy as np
from qcodes import validators as vals


class Tektronix_AWG5014(AWG5014.Tektronix_AWG5014):
    """
    This is the PycQED wrapper driver for the Tektronix AWG5014
    Arbitrary Waveform Generator.

    The wrapper adds the possibility to choose variable frequency external
    reference clock and set its multiplier.
    """

    def __init__(
            self,
            name: str,
            address: str,
            timeout: int = 180,
            num_channels: int = 4,
            **kwargs: Any):
        """
        Initializes the AWG5014.

        Args:
            name: name of the instrument
            address: GPIB or ethernet address as used by VISA
            timeout: visa timeout, in secs. long default (180)
                to accommodate large waveforms
            num_channels: number of channels on the device
        """
        super().__init__(name, address, timeout=timeout,
                         num_channels=num_channels, **kwargs)
        
        self.add_parameter('ext_ref_type',
                           label='External reference type',
                           get_cmd='SOURce1:ROSCillator:TYPE?',
                           set_cmd='SOURce1:ROSCillator:TYPE ' + '{}',
                           vals=vals.Enum('FIX', 'VAR'),
                           get_parser=self.newlinestripper)
        
        self.add_parameter('ext_ref_variable_multiplier',
                           label='External reference type',
                           get_cmd='SOURce1:ROSCillator:MULTiplier?',
                           set_cmd='SOURce1:ROSCillator:MULTiplier ' + '{}',
                           get_parser=int,
                           vals=vals.Ints(1, 240))

    def start(self, **kwargs) -> str:
        """
        Starts the AWG.

        Added compatibility with other instruments that accept kwargs
        in start().
        """
        return super().start()

    def generate_sequence_cfg(self, *args, **kwargs):
        """
        This function is used to generate a config file, that is used when
        generating sequence files, from existing settings in the awg.
        Querying the AWG for these settings takes ~0.7 seconds
        """
        AWG_sequence_cfg = super().generate_sequence_cfg(*args, **kwargs)
        AWG_sequence_cfg['EXTERNAL_REFERENCE_TYPE'] = \
            (1 if self.ext_ref_type().startswith('FIX') else 2)
        AWG_sequence_cfg['REFERENCE_MULTIPLIER_RATE'] = \
            self.ext_ref_variable_multiplier()
        return AWG_sequence_cfg

    def pack_waveform(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Converts/packs a waveform and two markers into a 16-bit format
        according to the AWG Integer format specification, see docstring of
        self._pack_waveform.
        """
        return self._pack_waveform(*args, **kwargs)

    def generate_awg_file(self, *args: Any, **kwargs: Any) -> bytes:
        """
        This function generates an .awg-file for uploading to the AWG, see
        docstring of self._generate_awg_file.
        """
        return self._generate_awg_file(*args, **kwargs)