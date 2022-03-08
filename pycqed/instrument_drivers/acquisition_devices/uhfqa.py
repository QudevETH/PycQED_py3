from pycqed.instrument_drivers.acquisition_devices.base import \
    ZI_AcquisitionDevice
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments\
    .UHFQA_core import UHFQA_core
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import ZI_base_qudev
import logging
log = logging.getLogger(__name__)


class UHFQA(UHFQA_core, ZI_base_qudev.ZI_base_instrument_qudev,
            ZI_AcquisitionDevice):
    """QuDev-specific PycQED driver for the ZI UHFQA

    This is the QuDev-specific PycQED driver for the 1.8 GSa/s UHFQA instrument
    from Zurich Instruments AG.

    Attributes:
        USER_REG_FIRST_SEGMENT (int): index of user register for first segment
            index, see parameter filter_segments in Pulsar
        USER_REG_LAST_SEGMENT (int): index of user register for last segment
            index, see parameter filter_segments in Pulsar
    """

    USER_REG_FIRST_SEGMENT = 5
    USER_REG_LAST_SEGMENT = 6

    acq_length_granularity = 4
    n_acq_int_channels = 10
    acq_sampling_rate = 1.8e9
    acq_weights_n_samples = 4097
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw', 'digitized', 'lin_trans'],
                     'int_avg_corr': ['corr', 'digitized_corr'],
                     'scope': [],
                     'sum': [],
                     'swp_pts': [],
                     'index': [],
                     'none': [],
                     }
    # private lookup dict to translate a data_type to an index understood by
    # the UHF
    _res_logging_indices = {
        'lin_trans': 0,  # applies the linear transformation matrix and
                         # subtracts the offsets defined in the UHFQC.
        'digitized': 1,  # thresholded results (0 or 1)
        'raw': 2,  # raw integrated+averaged results
        # FIXME: 3 is statistics logging. Do we need to implement it here?
        'raw_corr': 4,  # correlations mode before threshold
        'digitized_corr': 5  # correlations mode after threshold
                             # NOTE: thresholds need to be set outside the
                             # detector object.
    }

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
        ZI_AcquisitionDevice.__init__(self, *args, **kwargs)

    def prepare_poll(self):
        super().prepare_poll()
        self.set('qas_0_result_enable', 1)

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        # explicit call to the correct super method since UHFQA_core has a
        # method with the same name.
        ZI_AcquisitionDevice.acquisition_initialize(self,
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        # Overwrite default _acquisition_nodes
        self._acquisition_nodes = []  # will be filled below

        # Do not enable the rerun button; the AWG program uses userregs/0
        # to define the number of iterations in the loop
        self.awgs_0_single(1)

        # Translate mode to one of the two modes of the UHF driver
        self._acq_mode_uhf = 'iavg' if mode == 'avg' else 'rl'

        if data_type is not None:
            self.qas_0_result_source(self._res_logging_indices[data_type])
        if acquisition_length is not None:
            self.qas_0_integration_length(
                self.convert_time_to_n_samples(acquisition_length))

        if self._acq_mode_uhf == 'rl':  # integrated
            for c in channels:
                path = self._get_full_path(f'qas/0/result/data/{c[1]}/wave')
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_result_reset(1)
            self.qas_0_result_enable(0)
            self.qas_0_result_length(n_results)
            self.qas_0_result_averages(averages)
            ro_mode = 0
        else:  # input average = time traces
            # Note: UHFQA in iavg mode returns data in Volts as expected for
            # the avg mode
            for c in channels:
                path = self._get_full_path(f'qas/0/monitor/inputs/{c[1]}/wave')
                self._acquisition_nodes.append(path)
                self.subs(path)
            # Enable automatic readout
            self.qas_0_monitor_reset(1)
            self.qas_0_monitor_enable(1)
            self.qas_0_monitor_length(self.convert_time_to_n_samples(
                acquisition_length))
            self.qas_0_monitor_averages(averages)
            ro_mode = 1

        self.set(f'awgs_0_userregs_{self.USER_REG_LOOP_CNT}', loop_cnt)
        self.set(f'awgs_0_userregs_{self.USER_REG_RO_MODE}', ro_mode)

    def poll(self, poll_time=0.1):
        # The timeout of 1ms (second argument) is smaller than in
        # ZI_base_instrument (500ms) to allow fast spectroscopy.
        dataset = self.daq.poll(poll_time, 1, 4, True)
        for k in dataset:
            if isinstance(dataset[k], list):
                dataset[k] = [a.get('vector', a) for a in dataset[k]]
        return dataset

    def correct_offset(self, channels, data):
        data = super().correct_offset(channels, data)
        if self._acq_data_type == 'lin_trans':
            for i, channel in enumerate(channels):
                data[i] = data[i] - self.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel[1]))
        return data

    def set_classifier_params(self, acquisition_channel, params):
        if params is not None and 'thresholds' in params:
            self.set(
                f'qas_0_thresholds_{acquisition_channel}_level',
                params['thresholds'][0]
            )

    def _reset_n_acquired(self):
        super()._reset_n_acquired()
        self._n_acquisition_progress_last = 0
        self._n_acquisition_progress_add = 0

    def acquisition_progress(self):
        n_acq = (self.qas_0_result_acquired()
                 if self._acq_mode_uhf == 'rl'
                 else self.qas_0_monitor_acquired())
        n_last = self._n_acquisition_progress_last
        if n_last > 0 and n_acq == 0:
            # The UHF reports 0 when it is done. In this case, we keep the
            # last known progress value. This means that the progress
            # indicator will stay at that last known progress value during
            # data transfer, and MC will update the progress to the correct
            # value once it takes over control after the end of the data
            # transfer.
            return n_last + self._n_acquisition_progress_add
        if n_acq < n_last:
            # This workaround is needed because the UHF truncates
            # qas_0_result_acquired at 2**18 (and starts counting from 0
            # again). A decrease in n_acq compared to the last function call
            # indicates that this has happened and that we need to add 2**18
            # to the progess values.
            self._n_acquisition_progress_add += 2 ** 18
        self._n_acquisition_progress_last = n_acq
        return n_acq + self._n_acquisition_progress_add

    def _check_hardware_limitations(self):
        super()._check_hardware_limitations()
        # For the UHF, we currently only check whether the total number of
        # samples is supported by the UHF (2**20 is hardcoded). This limit
        # is usually not reached for averaged readout measurement, but could
        # be exceeded in case of single-shot readout.
        if self._acq_n_results > 2 ** 20:
            raise ValueError(
                f'Acquisition device {self.name} ({self.devname}):'
                f'{self._acq_n_results} > 1048576 not supported by the UHF.'
                f'Please reduce the compression_seg_lim, the number of 1D '
                f'sweep points, or the nr_shots.')

    def _acquisition_set_weight(self, channel, weight):
        self.set(f'qas_0_rotations_{channel[1]}', 1.0 - 1.0j)
        self.set(f'qas_0_integration_weights_{channel[1]}_real',
                 weight[0].copy())
        self.set(f'qas_0_integration_weights_{channel[1]}_imag',
                 weight[1].copy())
