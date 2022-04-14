import numpy as np
import math
import logging
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from pycqed.instrument_drivers.acquisition_devices.base import \
    AcquisitionDevice
from vc707_python_interface.qcodes.instrument_drivers import VC707 as \
    VC707_core
from vc707_python_interface.modules.base_module import BaseModule
from vc707_python_interface.settings.state_discriminator_settings \
    import DiscriminationUnit


log = logging.getLogger(__name__)


class VC707(VC707_core, AcquisitionDevice):
    """
    This is the Qudev specific PycQED driver for the VC707 FPGA instrument.

    TODO: Currently the state discrimination integration is not optimal in the
    way settings (integration weights, state centroids) are configured. Due to
    the current implementation of the FPGA driver, each time one of these
    settings is changed for one of the channels, all settings are reuploaded.
    """
    n_acq_units = 2
    n_acq_channels = 2  # TODO
    # TODO: Will change with decimation!
    acq_sampling_rate = 1.0e9
    # TODO: max length seems to be 2**16, but we probably do not want pycqed
    # to record so long traces by default.
    # TODO: In state discrimination mode this is actually 256.
    acq_max_trace_samples = 4096
    allowed_modes = {
        "avg": [], # averaged raw input (time trace) in V
        "int_avg": [
            "raw",
            "digitized", # Single-shot readout
        ],
        # "scope": [],
    }
    res_logging_indices = {#"raw": 1,  # raw integrated+averaged results
                           #"digitized": 3,  # thresholded results (0 or 1)
                           }

    def __init__(self, *args, devidx:int, clockmode:int, **kwargs):
        """
        Arguments:
            devidx, clockmode: see :class:`vc707_python_interface.fpga_interface.vc707_interface.VC707InitSettings`.
        """

        super().__init__(*args, **kwargs)
        AcquisitionDevice.__init__(self, *args, **kwargs)

        # Init settings must be set before initialize() is called
        self.init_devidx(devidx)
        self.init_clockmode(clockmode)
        self.initialize()

        self._acq_integration_weights = {}
        self._last_traces = []

    def prepare_poll(self):
        super().prepare_poll()

        if self._get_current_fpga_module_name() == "averager":
            self.averager.configure()
            self.averager.run()
        elif self._get_current_fpga_module_name() == "state_discriminator":
            self.state_discriminator.configure()
            self.state_discriminator.upload_weights()
            self.state_discriminator.run()

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acquisition_nodes = deepcopy(channels)
        self._acq_data_type = data_type

        if self._get_current_fpga_module_name() == "averager":
            self._last_traces = []
            self.averager_nb_samples.set(
                self.convert_time_to_n_samples(acquisition_length)
            )
            self.averager_nb_averages.set(averages)
            self.averager_loop_in_segment.set(False) # False = TV mode

            if mode == "avg":
                # pycqed does timetrace measurements only with a single segment
                self.averager_nb_segments.set(1)
            elif mode == "int_avg":
                self.averager_nb_segments.set(n_results)

        elif self._get_current_fpga_module_name() == "state_discriminator":
            self.state_discriminator_nb_samples.set(
                self.convert_time_to_n_samples(acquisition_length)
            )
            self.state_discriminator_nb_segments.set(n_results)
            # TODO: loop_cnt seems to be `self.nr_shots * self.nr_averages`, what
            # exact value do we need to set. In list_ouput mode the FPGA will
            # return exactly 4**nb_triggers_pow4 values.
            self.state_discriminator_nb_triggers_pow4.set(
                math.log(loop_cnt, 4)
            )
            self.state_discriminator_use_list_output.set(True)

            # TODO: configure fpga acquisition units properly
            print("Don't forget to set discrimination units, and weights.")
            # self.state_discriminator_units.set([])
            # for ch, quadrature in channels:
            #     self.state_discriminator_units.get().append(
            #         DiscriminationUnit()
            #     )

    def poll(self, poll_time=0.1) -> dict:

        # Sanity check
        super()._check_allowed_acquisition()

        # Return empty data if FPGA still running
        if not self._get_current_fpga_module().has_finished():
            return {}

        # Read results and update dataset
        if self._get_current_fpga_module_name() == "averager":
            res = self.averager.read_results()
            if self._acq_mode == "avg":
                return self._adapt_averager_results(res)
            elif self._acq_mode == "int_avg":
                return self._perform_integrated_averager(res)
        elif self._get_current_fpga_module_name() == "state_discriminator":
            res = self.state_discriminator.read_results()
            return self._adapt_state_discriminator_results(res)

    # TODO: Should take in list of tuple (acq_unit, quadrature)
    def set_classifier_params(self, channels, params):
        if self._get_current_fpga_module_name == "state_discriminator":
            if params is not None and 'means_' in params:
                for qubit in self.state_discriminator_qubits.get():
                    if qubit.source_adc in [c[0] for c in channels]:
                        qubit.center_coordinates = params['means_'].ravel()

                self.state_discriminator.upload_weights()

    def _adapt_averager_results(self, raw_results) -> dict:
        """Format the FPGA averager results as expected by PycQED."""

        dataset = {}

        for i in self._acq_units_used:
            int_channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            dataset.update({(i, ch): [raw_results[i*2 + n][0]]
                            for n, ch in enumerate(int_channels)})

        return dataset

    def _adapt_state_discriminator_results(self, raw_results) -> dict:
        """Format the FPGA averager results as expected by PycQED."""

        dataset = {}

        # TODO: What format is expected? The doc says "returns fraction of shots
        # based on the threshold defined in the UHFQC".
        for i in self._acq_units_used:
            dataset[i] = raw_results

        return dataset

    def _perform_integrated_averager(self, averager_results) -> dict:
        """Emulates integrated averager in software."""

        dataset = {}

        for acq_unit in self._acq_units_used:  # each acq. unit (physical input)
            # channel is a tuple of physical acquisition unit index (0 or 1)
            # and index of the weighted integration channel
            int_channels = [
                ch[1] for ch in self._acquisition_nodes if ch[0] == acq_unit
            ]

            for ch in int_channels:
                # Retrieve weights
                weights = self._acq_integration_weights[(acq_unit, ch)]

                # Integrate each segment with weights
                integration_result = \
                    np.dot(integration_result[acq_unit * 2, :, :],
                           weights[0][:averager_results.shape[-1]]) + \
                    np.dot(integration_result[acq_unit * 2 + 1, :, :],
                           weights[1][:averager_results.shape[-1]])

                dataset[(acq_unit, ch)] = [np.array(integration_result)]

        return dataset

#    def get_value_properties(self, data_type="raw", acquisition_length=None):
#        raise NotImplementedError(
#            "get_value_properties still needs to be implemented for using "
#            "the VC707 in integration mode.")

    def _acquisition_set_weight(self, channel, weight):
        super()._acquisition_set_weight()

        # Store a copy for software-emulated integration
        self._acq_integration_weights[channel] = weight

        for qubit in self.state_discriminator_qubits.get():
            if qubit.source_adc == channel:
                qubit.weights = weight

        self.state_discriminator.upload_weights()

    def _get_current_fpga_module_name(self) -> str:
        """Helper function that checks which FPGA module must be used."""

        if self._acq_mode == "avg" or \
           (self._acq_mode == "int_avg" and self._acq_data_type == "raw"):
            return "averager"
        elif self._acq_mode == "int_avg" and self._acq_data_type == "digitized":
            return "state_discriminator"
        else:
            raise ValueError(f"Unknow fpga mode for mode '{self._acq_mode}' "
                             f"and data type '{self._acq_mode}'.")

    def _get_current_fpga_module(self) -> BaseModule:
        """Returns the FPGA module that is used with the acquisition mode."""

        if self._get_current_fpga_module_name() == "averager":
            return self.averager
        elif self._get_current_fpga_module_name() == "state_discriminator":
            return self.state_discriminator
