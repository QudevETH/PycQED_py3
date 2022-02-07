import numpy as np
from pycqed.instrument_drivers.acquisition_devices.base import AcquisitionDevice
from qcodes_contrib_drivers.drivers.RohdeSchwarz.RTB2000 import \
    RTB2000 as RTB2000Core
import logging
log = logging.getLogger(__name__)


class RTB2000(RTB2000Core, AcquisitionDevice):
    """Acquisition device wrapper for the R&S RTB2000 oscilloscope series."""

    # FIXME: Do we need to get the time sweep points via
    #  x = np.array(ch.trace.setpoints[0]) ?

    n_acq_units = 4
    n_acq_int_channels = 10  # for emulated weighted integration
    acq_weights_n_samples = 4096  # TODO could be extended
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw',
                                 # 'digitized'
                                 ],
                     # 'scope': [],
                     }
    allowed_weights_types = ['optimal', 'DSB']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AcquisitionDevice.__init__(self, *args, **kwargs)
        self.n_acq_units = len(self.submodules)
        self._acq_integration_weights = {}
        self._last_traces = []

    @property
    def acq_sampling_rate(self):
        return self.sampling_rate()

    @property
    def acq_units(self):
        units = {(ch.channum - 1): ch for ch in self.submodules.values()}
        return [units[i] for i in range(len(units))]

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        # oscilloscope settings
        self.acquisition_type('AVER')
        self.num_acquisitions(averages)
        self.timebase_scale(acquisition_length / 12)  # total length on 12 div
        self.timebase_position(acquisition_length / 2)  # center at half length

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._last_traces = []

    def prepare_poll(self):
        super().prepare_poll()
        for i in self._acq_units_used:
            acq_unit = self.acq_units[i]
            # turn the channels on
            acq_unit.state('ON')
            # prepare the scope for returning data
            acq_unit.trace.prepare_trace()
        self.run_single()

    def poll(self, *args, **kwargs):
        dataset = {}
        if not self.average_complete():
            return dataset
        last_traces = {}
        for i in self._acq_units_used:  # each acq. unit (physical input)
            res = self.acq_units[i].trace.get_raw()
            if self._acq_mode == 'avg':
                dataset.update({(i, 0): [res]})
            elif self._acq_mode == 'int_avg':
                last_traces[i] = res
                # channel is a tuple of physical acquisition unit index (0 or 1)
                # and index of the weighted integration channel
                int_channels = [ch[1] for ch in self._acquisition_nodes if
                                ch[0] == i]
                for ch in int_channels:  # each weighted integration channel
                    weights = self._acq_integration_weights.get((i, ch), [[]])
                    # To have consistent data format for storing in the HDF,
                    # fill with nan values. This allows hybrid operation
                    # together with devices that support more segments.
                    integration_result = np.nan * np.zeros(self._acq_n_results)
                    # Currently treating it as 4 real-valued acq_units,
                    # so there is no imaginary part, i.e., no weights[1].
                    integration_result[0] = np.matrix.dot(
                        res[:len(weights[0])], weights[0][:len(res)])
                    dataset[(i, ch)] = [integration_result]
                self.save_extra_data(f'traces/{i}', np.atleast_2d(last_traces[i]))
        self._last_traces.append(last_traces)
        return dataset

    def acquisition_progress(self):
        n_acq = self.completed_acquisitions()
        n_acq *= self._acq_n_results
        return n_acq

    def _acquisition_set_weight(self, channel, weight):
        self._acq_integration_weights[channel] = weight

    def show_like_on_osci(self, ch, ax):
        '''
        Adjust the plot y-range to show the same
        as on the physical osci display
        '''
        if isinstance(ch, int):
            ch = f'ch{ch+1}'
        if isinstance(ch, str):
            ch = self.submodules[ch]
        ran = ch.range()
        off = ch.offset()
        pos = ch.position()
        sca = ch.scale()
        upper = ran/2 + off - pos*sca
        lower = -ran/2 + off - pos*sca
        ax.set_ylim(lower, upper)
