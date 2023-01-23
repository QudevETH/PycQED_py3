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
from vc707_python_interface.settings.histogrammer_settings import (BinSettings,
                                                                   DataInput)
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter
import time
from collections import OrderedDict


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
    n_acq_int_channels = 8  # TODO
    # TODO: max length seems to be 2**16, but we probably do not want pycqed
    # to record so long traces by default.
    # TODO: In state discrimination mode this is actually 256.
    acq_weights_n_samples = 4096
    acq_length_granularity = 8
    allowed_modes = {
        "avg": [], # averaged raw input (time trace) in V
        "int_avg": [
            "raw",
            "digitized", # state discrimination
        ],
        "hist": [
            "raw",
        ]
        # "scope": [],
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
        self._acq_classifier_params = {}
        self._last_traces = []
        self.nb_bins = {}
        self.peak_to_peak = {}
        self._histogrammer_result_lookup = {}
        self.add_parameter(
            'acq_start_after_awgs', initial_value=False,
            vals=vals.Bool(), parameter_class=ManualParameter,
            docstring="Whether the acquisition should be started after the "
                      "AWG(s), i.e., in prepare_poll_after_AWG_start, "
                      "instead of before, i.e., "
                      "in prepare_poll_before_AWG_start.")
        for i in range(self.n_acq_units):
            for j in range(self.n_acq_int_channels // 2):
                self.add_parameter(
                    f'acq_delay_{i}_{j}', initial_value=0,
                    vals=vals.Ints(), parameter_class=ManualParameter,
                    docstring="TODO (in number of samples)")
        self._acq_progress_prev = 0

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
        self._acq_progress_prev = 0
        module = self._get_current_fpga_module_name()
        if module == "averager":
            self.averager.run()
        elif module == "histogrammer":
            self.histogrammer.run()
        elif module == "state_discriminator":
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

        module = self._get_current_fpga_module_name()
        if module == "averager":
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
        elif module in ["histogrammer", "state_discriminator"]:
            module_obj = self._get_current_fpga_module()
            nb_samples = self.convert_time_to_n_samples(self._acq_length, True)
            self.state_discriminator.NB_STATE_ASSIGN_UNITS = 8
            pair_lookup = {(i, j): (i, j // 2) for (i, j) in self._acq_channels}
            pairs = OrderedDict({k: None for k in pair_lookup.values()})
            if module == 'histogrammer':
                self._acquisition_nodes = ['histogram']
            else:
                self._acquisition_nodes = list(pairs)
            for pair_id in pairs:
                if pair_id[0] != 0:
                    raise NotImplementedError(
                        'Histogrammer can only be used on physical channel 0 '
                        'until someone fixes a bug in the FPGA.')
                unit = DiscriminationUnit()
                # Use 2 integrators for hist (will be overwritten below for
                # state disc.):
                unit.nb_states = 3
                unit.weights = [np.zeros(nb_samples) for i in range(
                    4 if unit.nb_states == 3 else 2)]
                unit.delay = self.get(f'acq_delay_{pair_id[0]}_{pair_id[1]}')
                unit.source_adc = pair_id[0]
                pairs[pair_id] = unit
            n_ch_in_pair = {pair_id: 0 for pair_id in pairs}
            for ch in self._acq_channels:
                pair_id = pair_lookup[ch]
                for i in range(2):
                    w = self._acq_integration_weights[ch][i]
                    p = max(nb_samples - len(w), 0)
                    pairs[pair_id].weights[2*n_ch_in_pair[pair_id] + i] =\
                        np.pad(w, p)[:nb_samples]
                n_ch_in_pair[pair_id] += 1
            if module == 'state_discriminator':
                for pair_id in pairs:
                    pairs[pair_id].nb_states = n_ch_in_pair[pair_id] + 1
                    channels = [ch for ch in self._acq_channels
                                if pair_lookup[ch] == pair_id]
                    pairs[pair_id].center_coordinates = tuple(
                        self._acq_classifier_params[tuple(channels)][
                            'centroids'])

            self.state_discriminator.settings.units = list(pairs.values())
            self.state_discriminator.settings.nb_samples = nb_samples
            self.state_discriminator.settings.use_list_output = (
                module == "state_discriminator")
            module_obj.settings.nb_segments = (
                self._acq_n_results // self._acq_loop_cnt)
            nb_triggers = self._acq_n_results
            nb_triggers_pow4_float = np.log(nb_triggers) / np.log(4)
            nb_triggers_pow4 = int(np.ceil(nb_triggers_pow4_float))
            if int(nb_triggers_pow4_float) != nb_triggers_pow4_float:
                log.warning(f'nb_triggers_pow4 got rounded up')  # TODO
            module_obj.settings.nb_triggers_pow4 = nb_triggers_pow4
            if module == 'histogrammer':
                self.histogrammer.settings.nb_states = 1  # no state disc.
                self.histogrammer.settings.enable_time_resolve = False
                integrator_lookup = {}
                for pair_nr, pair_id in enumerate(pairs):
                    for ch in [ch for ch in self._acq_channels
                               if pair_lookup[ch] == pair_id]:
                        integrator_lookup[ch] = pair_nr * 2 + (ch[1] % 2)
                bin_settings = []
                self._histogrammer_result_lookup = {}
                used_hists = 0
                for i in range(self.n_acq_units):
                    if not any([ch[0] == i for ch in self._acq_channels]):
                        continue
                    for j in range(self.n_acq_int_channels):
                        # Here, log2(1) as default means: no histogram
                        nb_bins_pow2 = int(np.log2(self.nb_bins.get(
                            (i, j), 1)))
                        if nb_bins_pow2:
                            self._histogrammer_result_lookup[
                                (i, j)] = used_hists
                            used_hists += 1
                        bin_settings.append(BinSettings(
                            # this exploits that INT_i are consecutive ints
                            DataInput.INT_0 + integrator_lookup.get((i, j), 0),
                            peak_to_peak=self.peak_to_peak.get((i, j), -1),
                            nb_bins_pow2=nb_bins_pow2,
                            delay=0))
                self.histogrammer.settings.bin_settings = bin_settings
            # FIXME: Downscaling set to zero here: hist and state don't work
            # if downscaling is not set to zero and we don't know why exactly
            self.preprocessing_down_scaling(0)
            # TODO does the order of configure and upload matter?
            module_obj.configure()
            # FIXME: It is inefficient to always upload even if nothing
            #  changed. We could introduce a caching mechanism (similarly to
            #  the sequence caching in pulsar) to only upload if needed.
            self.state_discriminator.upload_weights(discriminate=(
                module == "state_discriminator"))

    def acquisition_progress(self):
        return self._fpga.fpga_interface.trigger_counter()

    def poll(self, poll_time=0.1) -> dict:
        # Return empty data if FPGA still running
        module_obj = self._get_current_fpga_module()
        if not module_obj.has_finished():
            return {}
        # Read results and update dataset
        res = module_obj.read_results()
        module = self._get_current_fpga_module_name()
        if module == "averager":
            if self._acq_mode == "avg":
                return self._adapt_averager_results(res)
            elif self._acq_mode == "int_avg":
                return self._perform_integrated_averager(res)
        elif module == "histogrammer":
            return self._adapt_histogrammer_results(res)
        elif module == "state_discriminator":
            return self._adapt_state_discriminator_results(res)

    def set_classifier_params(self, channels, params):
        self._acq_classifier_params[tuple(channels)] = params

    def _adapt_averager_results(self, raw_results) -> dict:
        """Format the FPGA averager results as expected by PycQED."""

        dataset = {}

        for i in self._acq_units_used:
            int_channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            dataset.update({(i, ch): [raw_results[i*2 + n][0]]
                            for n, ch in enumerate(int_channels)})

        return dataset

    def _adapt_histogrammer_results(self, raw_results) -> dict:
        """Format the FPGA histogrammer results as expected by PycQED."""
        # cut by segment (which is the last index)
        res = {'histogram': [[{'data': a.T} for a in raw_results.T]]}
        return res

    def _adapt_state_discriminator_results(self, raw_results) -> dict:
        """Format the FPGA state discriminator results as expected by PycQED."""
        def raw_to_list(data, nb_states, nb_units=0):
            """
            Interpets a decimal int in the basis indicated by nb_states (e.g.,
            binary/ternary for qubit/qutrit readout) and returns the digits
            as a list with the least-significant digit first (so that the
            order of the returned results corresponds to the order in
            self.state_discriminator.settings.units).
            """
            nums = []
            while data:
                data, r = divmod(data, nb_states)
                nums.append(r)
            nums += [0] * max(0, nb_units - len(nums))
            return nums

        units = self.state_discriminator.settings.units
        nb_states = units[0].nb_states
        assert all([u.nb_states == nb_states for u in units])
        data = [raw_to_list(r, nb_states, nb_units=len(units))
                for r in raw_results]
        return {pair: [[d[i] for d in data]]
                for i, pair in enumerate(self._acquisition_nodes)}

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
        # Store a copy for software-emulated integration and to upload later
        self._acq_integration_weights[channel] = weight

    def _get_current_fpga_module_name(self) -> str:
        """Helper function that checks which FPGA module must be used."""

        if self._acq_mode == "avg":
            return "averager"
        elif self._acq_mode == "int_avg":
            if self._acq_data_type == "raw":
                return "averager"
            elif self._acq_data_type.startswith("digitized"):
                return "state_discriminator"
        raise ValueError(f"Unknown FPGA mode for mode '{self._acq_mode}' "
                         f"and data type '{self._acq_data_type}'.")

    def _get_current_fpga_module(self) -> BaseModule:
        """Returns the FPGA module that is used with the acquisition mode."""

        module = self._get_current_fpga_module_name()
        if module == "averager":
            return self.averager
        elif module == 'histogrammer':
            return self.histogrammer
        elif module == "state_discriminator":
            return self.state_discriminator

    def get_int_channels_value_names_map(self, channels, data_type='raw'):
        if data_type == 'digitized':
            pair_lookup = {(i, j): (i, j // 2) for (i, j) in channels}
            n_ch_in_pair = {v: sum([pair_lookup[c] == v for c in channels])
                            for v in pair_lookup.values()}
            ch_vn_map = OrderedDict()
            for ch in channels:
                pair = pair_lookup[ch]
                if n_ch_in_pair[pair] == 2:
                    chs = '_' + '_'.join([f'{c[1]}' for c, p
                                          in pair_lookup.items() if p == pair])
                else:
                    chs = ch[1]
                vn = f'{self.name}_{ch[0]}_{data_type} w{chs}'
                ch_vn_map[ch] = vn
            return ch_vn_map
        return super().get_int_channels_value_names_map(channels, data_type)

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
        elif 'digitized' in data_type:
            properties['value_unit'] = 'state'
        return properties

