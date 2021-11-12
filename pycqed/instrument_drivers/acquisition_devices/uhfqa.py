from pycqed.instrument_drivers.physical_instruments.ZurichInstruments\
    .UHFQA_core import UHFQA_core
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_base_qudev
import logging
log = logging.getLogger(__name__)


class UHFQA(UHFQA_core, ZI_base_qudev.ZI_base_instrument_qudev):
    """This is the Qudev specific PycQED driver for the 1.8 GSa/s UHFQA instrument
    from Zurich Instruments AG.
    """

    USER_REG_FIRST_SEGMENT = 5
    USER_REG_LAST_SEGMENT = 6

    def acquisition_initialize(self, samples, averages, loop_cnt, channels=(0, 1), mode='rl') -> None:
        # Define the channels to use and subscribe to them
        self._acquisition_nodes = []

        if mode == 'rl':
            for c in channels:
                path = self._get_full_path('qas/0/result/data/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_result_reset(1)
            self.qas_0_result_enable(0)
            self.qas_0_result_length(samples)
            self.qas_0_result_averages(averages)
            ro_mode = 0
        else:
            for c in channels:
                path = self._get_full_path('qas/0/monitor/inputs/{}/wave'.format(c))
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_monitor_reset(1)
            self.qas_0_monitor_enable(1)
            self.qas_0_monitor_length(samples)
            self.qas_0_monitor_averages(averages)
            ro_mode = 1

        self.set('awgs_0_userregs_{}'.format(uhf.UHFQA_core.USER_REG_LOOP_CNT), loop_cnt)
        self.set('awgs_0_userregs_{}'.format(uhf.UHFQA_core.USER_REG_RO_MODE), ro_mode)

    def start(self, **kwargs):
        super().start()  # UHFQA_core.start() does not expect kwargs
