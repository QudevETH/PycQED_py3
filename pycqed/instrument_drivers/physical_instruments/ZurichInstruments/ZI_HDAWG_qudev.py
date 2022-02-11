"""
    Qudev specific driver for the HDAWG instrument.
"""

import logging

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_HDAWG_core as zicore
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_base_qudev

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_whitelist = {
            'IDN',
            'clockbase',
            'system_clocks_referenceclock_source',
            'system_clocks_referenceclock_status',
            'system_clocks_referenceclock_freq'}
        for i in range(4):
            self._snapshot_whitelist.update({
                'awgs_{}_enable'.format(i),
                'awgs_{}_outputs_0_amplitude'.format(i),
                'awgs_{}_outputs_1_amplitude'.format(i)})
        for i in range(8):
            self._snapshot_whitelist.update({
                'sigouts_{}_direct'.format(i),
                'sigouts_{}_offset'.format(i),
                'sigouts_{}_on'.format(i) ,
                'sigouts_{}_range'.format(i),
                'sigouts_{}_delay'.format(i)})

    def _check_options(self):
        """
        Override the method in ZI_HDAWG_core, to bypass the unneeded check for
        the PC option.
        """
        pass

    def clock_freq(self):
        return 2.4e9
