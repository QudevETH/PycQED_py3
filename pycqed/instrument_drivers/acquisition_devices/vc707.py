import numpy as np
from pycqed.instrument_drivers.acquisition_devices.base import AcquisitionDevice
from vc707_python_interface.qcodes.instrument_drivers import VC707 as \
    VC707_core
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import time
import logging
log = logging.getLogger(__name__)


class VC707(VC707_core, AcquisitionDevice):
    """PycQED acquisition device wrapper for the VC707 FPGA."""

    n_acq_units = 2
    n_acq_int_channels = 2  # TODO
    # TODO: max length seems to be 2**16, but we probably do not want pycqed
    #  to record so long traces by default
    # TODO: In state discrimination mode this is actually 256.
    acq_weights_n_samples = 4096
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw',
                                 # 'digitized'
                                 ],
                     # 'scope': [],
                     }

    def __init__(self, *args, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        AcquisitionDevice.__init__(self, *args, **kwargs)
        self.initialize(verbose=True if verbose else False)
        self._acq_integration_weights = {}
        self._last_traces = []
        self.add_parameter(
            'acq_start_after_awgs', initial_value=False,
            vals=vals.Bool(), parameter_class=ManualParameter,
            docstring="Whether the acquisition should be started after the "
                      "AWG(s), i.e., in prepare_poll_after_AWG_start, "
                      "instead of before, i.e., "
                      "in prepare_poll_before_AWG_start.")

    @property
    def acq_sampling_rate(self):
        return 1.0e9 / self.preprocessing_decimation()

    def prepare_poll_before_AWG_start(self):
        super().prepare_poll_before_AWG_start()
        if not self.acq_start_after_awgs():
            self.averager.run()

    def prepare_poll_after_AWG_start(self):
        super().prepare_poll_after_AWG_start()
        if self.acq_start_after_awgs():
            # The following sleep is a workaround to avoid weird behavior
            # that is potentially caused by an unstable main trigger signal
            # right after starting the main trigger.
            time.sleep(0.1)
            self.averager.run()

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))

        self.averager_nb_samples.set(self.convert_time_to_n_samples(
            acquisition_length))
        if mode == 'avg':
            #  pycqed does timetrace measurements only with a single segment
            self.averager_nb_segments.set(1)
        else:
            self.averager_nb_segments.set(n_results)
        self.averager_nb_averages.set(averages)
        self.averager_loop_in_segment.set(False)  # False = TV mode
        self.averager.configure()
        self._last_traces = []

    def acquisition_progress(self):
        return self.averager.trigger_counter()

    def poll(self, *args, **kwargs):
        dataset = {}
        if self._acq_mode in ['avg', 'int_avg']:
            # int_avg is included because we emulate integration in software
            res = self.averager.read_results()
            if res.shape[-1] == 0:  # no data received
                # the avg block below can handle empty arrays, but the int_avg
                # block cannot, so we just return the empty data already here
                return dataset
        else:
            raise NotImplementedError(
                f'{self.name}: Currently, mode {self._acq_mode} is not '
                f'implemented for {self.__class__.__name__}.')
        last_traces = {}
        for i in self._acq_units_used:  # each acq. unit (physical input)
            # channel is a tuple of physical acquisition unit index (0 or 1)
            # and index of the weighted integration channel
            int_channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            if self._acq_mode == 'avg':
                # i*2 + n takes Re (n=0) or Im (n=1) of the i-th physical
                # input. The index 0 is because pycqed does timetrace
                # measurements only with a single segment.
                # FIXME: check in which cases Im is useful (when using
                #  demodulation?)
                dataset.update({(i, ch): [res[i*2 + n][0]]
                                for n, ch in enumerate(int_channels)})
            elif self._acq_mode == 'int_avg':
                last_traces[i] = res[i * 2:i * 2 + 2]
                for ch in int_channels:  # each weighted integration channel
                    weights = self._acq_integration_weights[(i, ch)]
                    integration_result = []
                    for seg in range(self._acq_n_results):  # each segment
                        # i*2 (i*2+1) takes Re (Im) of the i-th physical
                        # input. Note that Im is useful only with DDC.
                        # FIXME: verify sign & normalization factor
                        integration_result.append(
                            np.matrix.dot(res[i * 2][seg],
                                          weights[0][:res.shape[-1]]) +
                            np.matrix.dot(res[i * 2 + 1][seg],
                                          weights[1][:res.shape[-1]])
                        )
                    dataset[(i, ch)] = [np.array(integration_result)]
                self.save_extra_data(f'traces/{i}', last_traces[i])
        self._last_traces.append(last_traces)
        return dataset

    def _acquisition_set_weight(self, channel, weight):
        self._acq_integration_weights[channel] = weight

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        properties = super().get_value_properties(
            data_type=data_type, acquisition_length=acquisition_length)
        if data_type == 'raw':
            if acquisition_length is None:
                raise ValueError('Please specify acquisition_length.')
            # Units are only valid when using SSB or DSB demodulation.
            # value corresponds to the peak voltage of a cosine with the
            # demodulation frequency.
            properties['value_unit'] = 'Vpeak'
            properties['scaling_factor'] = 1 / (self.acq_sampling_rate
                                                * acquisition_length)
        return properties
