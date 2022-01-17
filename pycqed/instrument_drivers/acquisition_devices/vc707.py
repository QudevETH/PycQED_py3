import numpy as np
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.acquisition_devices.base import \
    AcquisitionDevice
from vc707_python_interface.qcodes.instrument_drivers import VC707 as \
    VC707_core
import logging
log = logging.getLogger(__name__)


class VC707(VC707_core, AcquisitionDevice):
    """
    This is the Qudev specific PycQED driver for the VC707 FPGA instrument.
    """
    n_acq_units = 2
    n_acq_channels = 2  # TODO
    acq_sampling_rate = 1.0e9
    # TODO: max length seems to be 2**16, but we probably do not want pycqed
    #  to record so long traces by default
    acq_weights_n_samples = 4096
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw',
                                 # 'digitized'
                                 ],
                     # 'scope': [],
                     }
    res_logging_indices = {#'raw': 1,  # raw integrated+averaged results
                           #'digitized': 3,  # thresholded results (0 or 1)
                           }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AcquisitionDevice.__init__(self, *args, **kwargs)
        self.initialize()
        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=30,
            parameter_class=ManualParameter,
            vals=validators.Ints())
        self._acq_integration_weights = {}
        self._last_traces = []

    def prepare_poll(self):
        super().prepare_poll()
        self.averager.run()

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        self._acquisition_initialize_base(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acquisition_nodes = deepcopy(channels)

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
        self._last_traces.append(last_traces)
        return dataset

#    def get_value_properties(self, data_type='raw', acquisition_length=None):
#        raise NotImplementedError(
#            'get_value_properties still needs to be implemented for using '
#            'the VC707 in integration mode.')

    def _acquisition_set_weight(self, channel, weight):
        self._acq_integration_weights[channel] = weight



