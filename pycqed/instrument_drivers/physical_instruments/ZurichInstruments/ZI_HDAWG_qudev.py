"""
    Qudev specific driver for the HDAWG instrument.
"""

import logging

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore

log = logging.getLogger(__name__)

class ZI_HDAWG_qudev(zicore.ZI_HDAWG_core):
    """This is the Qudev specific PycQED driver for the HDAWG instrument
    from Zurich Instruments AG.
    """

    USER_REG_FIRST_SEGMENT = 5
    USER_REG_LAST_SEGMENT = 6

    def start(self, **kw):
        """
        Start the sequencer
        :param kw: currently ignored, added for compatibilty with other
        instruments that accept kwargs in start().
        """
        super().start()  # ZI_HDAWG_core.start() does not expect kwargs
