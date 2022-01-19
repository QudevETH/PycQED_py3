import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase

class ZI_base_instrument_qudev(zibase.ZI_base_instrument):
    """
    Class to override functionality of ZI_base_instrument using
    multi-inheritance in ZI_HDAWG_qudev and acquisition_devices.uhfqa.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._awg_source_strings = {}

    def start(self, **kw):
        """
        Start the sequencer
        :param kw: currently ignored, added for compatibilty with other
        instruments that accept kwargs in start().
        """
        super().start()  # ZI_base_instrument.start() does not expect kwargs

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  *args, **kwargs):
        self._awg_source_strings[awg_nr] = program_string
        super().configure_awg_from_string(awg_nr, program_string,
                                          *args, **kwargs)

    def _add_codeword_waveform_parameters(self, num_codewords) -> None:
        """
        Override to remove Delft-specific functionality.
        """
        pass

