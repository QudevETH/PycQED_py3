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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_whitelist = {
            'IDN',
            'clockbase',}
        for i in range(1):
            self._snapshot_whitelist.update({
                'awgs_{}_enable'.format(i),
                'awgs_{}_outputs_0_amplitude'.format(i),
                'awgs_{}_outputs_1_amplitude'.format(i)})
        for i in range(2):
            self._snapshot_whitelist.update({
                'sigouts_{}_offset'.format(i),
                'sigouts_{}_on'.format(i) ,
                'sigouts_{}_range'.format(i),})

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

        self.set(f'awgs_0_userregs_{self.USER_REG_LOOP_CNT}', loop_cnt)
        self.set(f'awgs_0_userregs_{self.USER_REG_RO_MODE}', ro_mode)

    def poll(self, poll_time=0.1):
        # The timeout of 1ms (second argument) is smaller than in
        # ZI_base_instrument (500ms) to allow fast spectroscopy.
        return self.daq.poll(poll_time, 1, 4, True)
