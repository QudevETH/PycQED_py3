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
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import time


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
    n_acq_int_channels = 2  # TODO
    # TODO: max length seems to be 2**16, but we probably do not want pycqed
    # to record so long traces by default.
    # TODO: In state discrimination mode this is actually 256.
    acq_weights_n_samples = 4096
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

    def __init__(self, *args, devidx:int=None, clockmode:int=None,
                 verbose=False, **kwargs):
        """
        Arguments:
            devidx, clockmode: see :class:`vc707_python_interface.fpga_interface.vc707_interface.VC707InitSettings`.
            verbose: see :meth:`vc707_python_interface.fpga_interface.vc707_interface.VC707Interface.initialize`.
        """

        super().__init__(*args, **kwargs)
        AcquisitionDevice.__init__(self, *args, **kwargs)

        self.initialize(devidx=devidx, clockmode=clockmode, verbose=verbose)
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
            self._prepare_poll()

    def prepare_poll_after_AWG_start(self):
        super().prepare_poll_after_AWG_start()
        if self.acq_start_after_awgs():
            # The following sleep is a workaround to avoid weird behavior
            # that is potentially caused by an unstable main trigger signal
            # right after starting the main trigger.
            time.sleep(0.1)
            self._prepare_poll()

    def _prepare_poll(self):
        if self._get_current_fpga_module_name() == "averager":
            self.averager.run()
        elif self._get_current_fpga_module_name() == "state_discriminator":
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
            self.averager.configure()

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
            self.state_discriminator.configure()

    def acquisition_progress(self):
        if self._get_current_fpga_module_name() == "averager":
            return self.averager.trigger_counter()
        else:
            # FIXME: implement trigger_counter for other modules
            return 0

    def poll(self, poll_time=0.1) -> dict:
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
        if params is not None and 'means_' in params:
            for qubit in self.state_discriminator_qubits.get():
                if qubit.source_adc in [c[0] for c in channels]:
                    qubit.center_coordinates = params['means_'].ravel()

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
                # i*2 (i*2+1) takes Re (Im) of the i-th physical
                # input. Note that Im is useful only with DDC.
                integration_result = \
                    np.dot(averager_results[acq_unit * 2, :, :],
                           weights[0][:averager_results.shape[-1]]) + \
                    np.dot(averager_results[acq_unit * 2 + 1, :, :],
                           weights[1][:averager_results.shape[-1]])

                dataset[(acq_unit, ch)] = [np.array(integration_result)]

        return dataset

    def _acquisition_set_weight(self, channel, weight):
        # Store a copy for software-emulated integration
        self._acq_integration_weights[channel] = weight

        return  # FIXME: following code not compatible with current driver
        for qubit in self.state_discriminator_qubits.get():
            if qubit.source_adc == channel:
                qubit.weights = weight

    def _get_current_fpga_module_name(self) -> str:
        """Helper function that checks which FPGA module must be used."""

        if self._acq_mode == "avg" or \
           (self._acq_mode == "int_avg" and self._acq_data_type == "raw"):
            return "averager"
        elif self._acq_mode == "int_avg" and self._acq_data_type == "digitized":
            return "state_discriminator"
        else:
            raise ValueError(f"Unknown FPGA mode for mode '{self._acq_mode}' "
                             f"and data type '{self._acq_mode}'.")

    def _get_current_fpga_module(self) -> BaseModule:
        """Returns the FPGA module that is used with the acquisition mode."""

        if self._get_current_fpga_module_name() == "averager":
            return self.averager
        elif self._get_current_fpga_module_name() == "state_discriminator":
            return self.state_discriminator

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

