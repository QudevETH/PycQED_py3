from qcodes.instrument_drivers.tektronix import AWG5014
from typing import Any
import numpy as np


class Tektronix_AWG5014(AWG5014.Tektronix_AWG5014):
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