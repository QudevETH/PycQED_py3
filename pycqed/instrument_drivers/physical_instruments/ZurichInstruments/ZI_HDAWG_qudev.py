"""
    Qudev specific driver for the HDAWG instrument.
"""

import logging

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_base_qudev
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import \
    snapshot_whitelist as snw

log = logging.getLogger(__name__)

class ZI_HDAWG_qudev(zicore.ZI_HDAWG_core,
                     ZI_base_qudev.ZI_base_instrument_qudev):
    """This is the Qudev specific PycQED driver for the HDAWG instrument
    from Zurich Instruments AG.

    Changes compared to the base class include:
    - Add user registers for segment filtering
    - Bypass unneeded check for PC option
    - Add clock_freq method
    - Add _snapshot_whitelist to let MC store only a subset of parameters
      (see MC.store_snapshot_parameters)
    """

    USER_REG_FIRST_SEGMENT = 5
    USER_REG_LAST_SEGMENT = 6

    def __init__(self, *args, interface: str= '1GbE',
                 server: str= 'localhost', **kwargs):
        super().__init__(*args, interface=interface, server=server, **kwargs)
        self.interface = interface
        self.server = server
        self.exclude_from_stop = []
        self._snapshot_whitelist = snw.generate_snapshot_whitelist_hdawg()

    def _check_options(self):
        """
        Override the method in ZI_HDAWG_core, to bypass the unneeded check for
        the PC option.
        """
        pass

    def clock_freq(self):
        return self.system_clocks_sampleclock_freq()

    def stop(self):
        log.info(f"{self.devname}: Stopping '{self.name}'")
        # Stop all AWG's that are not part of exclude_from_stop
        for awg_nr in range(self._num_awgs()):
            if awg_nr not in self.exclude_from_stop:
                self.set('awgs_{}_enable'.format(awg_nr), 0)
        self.check_errors()
