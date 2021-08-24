import numpy as np
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import sweep_functions as swf
from pycqed.instrument_drivers.acquisition_devices.base import \
    ZI_AcquisitionDevice
from zhinst.qcodes import SHFQA as SHFQA_core
import logging
log = logging.getLogger(__name__)


class SHFQA(SHFQA_core, ZI_AcquisitionDevice):
    """This is the Qudev specific PycQED driver for the SHFQA instrument
    from Zurich Instruments AG.
    """
    # acq_length_granularity = 4
    acq_sampling_rate = 2.0e9
    acq_max_trace_samples = 4096  #??
    acq_Q_sign = -1
    allowed_modes = {#'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw', 'digitized'],
                     # 'scope': [],
                     }
    res_logging_indices = {'raw': 1,  # raw integrated+averaged results
                           'digitized': 3,  # thresholded results (0 or 1)
                           }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ZI_AcquisitionDevice.__init__(self, *args, **kwargs)
        self.n_acq_units = len(self.qachannels)
        self.lo_freqs = [None] * self.n_acq_units  # re-create with correct length
        self._acq_loop_cnts_last = [None] * self.n_acq_units
        self.n_acq_channels = len(self.qachannels[0].readout.integrations)
        self._reset_acq_poll_inds()
        self._acq_units_modes = {}

        self.awg_active = [False] * self.n_acq_units
        self._awg_programs = {}
        self._waves_to_upload = {}

        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=30,
            parameter_class=ManualParameter,
            vals=validators.Ints())
        self.add_parameter(
            'allowed_lo_freqs',
            initial_value=np.arange(1e9, 8.1e9, 100e6),
            parameter_class=ManualParameter,
            set_parser=lambda x: list(np.atleast_1d(x).flatten()),
            vals=validators.MultiType(validators.Lists(), validators.Arrays(),
                                      validators.Numbers()))

    @property
    def devname(self):
        return self.get_idn()['serial']

    @property
    def daq(self):
        return self._controller._controller._connection.daq

    def _reset_acq_poll_inds(self):
        self._acq_poll_inds = []
        for i in range(self.n_acq_units):
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            self._acq_poll_inds.append([0] * len(channels))

    def set_lo_freq(self, acq_unit, lo_freq):
        super().set_lo_freq(acq_unit, lo_freq)
        self.qachannels[acq_unit].center_freq(lo_freq)
        new_lo_freq = self.qachannels[acq_unit].center_freq()
        if np.abs(new_lo_freq - lo_freq) > 1:
            log.warning(f'{self.name}: center frequency {lo_freq/1e6:.6f} '
                        f'MHz not supported. Setting center frequency to '
                        f'{new_lo_freq/1e6:.6f} MHz.')

    def prepare_poll(self):
        super().prepare_poll()
        for i in self._acq_units_used:
            if self._acq_units_modes[i] == 'readout':
                self.qachannels[i].readout.arm(length=self._acq_n_results,
                                               averages=self._acq_averages)
            else:  # spectroscopy
                self._arm_spectroscopy(i, length=self._acq_n_results,
                                       averages=self._acq_averages)
        self._reset_acq_poll_inds()

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        self._acquisition_initialize_base(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acq_units_modes = {i: self.qachannels[i].mode()
                                 for i in self._acq_units_used}
        self._acquisition_nodes = deepcopy(channels)

        for i in range(self.n_acq_units):
            # Make sure the readout is stopped. It will be started in
            # prepare_poll
            self.qachannels[i].readout.stop()  # readout mode
            if i not in self._acq_units_used:
                self.qachannels[i].sweeper.oscillator_gain(0)  # spectroscopy mode

        log.debug(f'{self.name}: units used: ' + repr(self._acq_units_used))
        for i in self._acq_units_used:
            if self._acq_units_modes[i] == 'readout':
                # Disable rerun; the AWG program defines the number of
                # iterations in the loop
                self.qachannels[i].generator.single(1)
                if data_type is not None:
                    self.qachannels[i].readout.result_source(
                        self.res_logging_indices[data_type])
                if acquisition_length is not None:
                    self.qachannels[i].readout.integration_length(
                        self.convert_time_to_n_samples(acquisition_length))
                if self._acq_loop_cnts_last[i] != loop_cnt:
                    self._program_awg(i)  # reprogramm this acq unit
                    self._acq_loop_cnts_last[i] = loop_cnt
            else:  # spectroscopy
                self.qachannels[i].sweeper.oscillator_gain(1.0)
                self.daq.setInt(
                    self._get_spectroscopy_node(i, "integration_length"),
                    self.convert_time_to_n_samples(acquisition_length))

    def acquisition_finalize(self):
        for ch in self.qachannels:
            ch.sweeper.oscillator_gain(0)

    def set_awg_program(self, acq_unit, awg_program, waves_to_upload):
        self._awg_programs[acq_unit] = awg_program
        self._waves_to_upload[acq_unit] = waves_to_upload
        self._acq_loop_cnts_last[acq_unit] = None  # force programming

    def _program_awg(self, acq_unit):
        awg_program = self._awg_programs.get(acq_unit, None)
        if awg_program is None:
            return
        qachannel = self.qachannels[acq_unit]
        awg_program = awg_program.replace(
            '{loop_count}', f'{self._acq_loop_cnt}')
        qachannel.generator.set_sequence_params(
            sequence_type="Custom", program=awg_program)
        qachannel.generator.compile()
        waves_to_upload = self._waves_to_upload.get(acq_unit, None)
        if waves_to_upload is not None:
            # upload waveforms
            qachannel.generator.reset_queue()
            for wf in waves_to_upload.values():
                qachannel.generator.queue_waveform(wf.copy())
            qachannel.generator.upload_waveforms()
            self._waves_to_upload[acq_unit] = None  # upload only once

    def _get_spectroscopy_node(self, acq_unit, node):
        lookup = {
            'enable': 'result/enable',
            'averages': 'result/averages',
            'length': 'result/length',
            'integration_length': 'length',
            'data': 'result/data/wave',
            'acquired': 'result/acquired',
        }
        return f"/{self.devname}/qachannels/{acq_unit}/spectroscopy/" \
               f"{lookup.get(node, node)}"

    def _arm_spectroscopy(self, acq_unit, length, averages):
        self.daq.setInt(
            self._get_spectroscopy_node(acq_unit, "length"), length)
        self.daq.setInt(
            self._get_spectroscopy_node(acq_unit, "averages"), averages)
        self.daq.setInt(self._get_spectroscopy_node(acq_unit, "enable"), 1)
        self.daq.sync()

    def poll(self, *args, **kwargs):
        dataset = {}
        for i in self._acq_units_used:
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            if self._acq_units_modes[i] == 'readout':
                res = self.qachannels[i].readout.read(integrations=channels,
                                                      blocking=False)
                scaling_factor = np.sqrt(2)\
                                 / (self.acq_sampling_rate * self._acq_length)
                dataset.update(
                    {(i, ch): [np.real(res[n][self._acq_poll_inds[i][n]:])
                               * scaling_factor]
                    for n, ch in enumerate(channels)})
                self._acq_poll_inds[i] = [len(res[n]) for n in range(len(channels))]
            else:  # spectroscopy
                progress = self.daq.getInt(
                    self._get_spectroscopy_node(i, "acquired"))
                if progress == self._acq_averages:
                    node = self._get_spectroscopy_node(i, "data")
                    data = self.daq.get(node, flat=True).get(node, [])
                    data = [a.get('vector', a) for a in data]
                    data = [[np.real(a), np.imag(a)] for a in data]
                    scaling_factor = np.sqrt(2)
                    dataset.update({(i, ch): [a[n % 2] * scaling_factor
                        for a in data] for n, ch in enumerate(channels)})
        return dataset

    def get_lo_sweep_function(self, acq_unit, ro_mod_freq):
        name = 'Readout frequency'
        name_offset = 'Readout frequency with offset'
        return swf.Offset_Sweep(
            swf.MajorMinorSweep(
                self.qachannels[acq_unit].center_freq,
                swf.Offset_Sweep(
                    self.qachannels[acq_unit].sweeper.oscillator_freq,
                    ro_mod_freq),
                    self.allowed_lo_freqs(),
                name=name_offset, parameter_name=name_offset),
            -ro_mod_freq, name=name, parameter_name=name)

    def stop(self):
        for ch in self.qachannels:
            ch.generator.stop()

    def start(self, **kwargs):
        for i, ch in enumerate(self.qachannels):
            if self.awg_active[i]:
                if ch.mode() == 'readout':
                    ch.generator.run()
                else:
                    # No AWG needs to be started in spectroscopy mode.
                    # The pulse generation starts together with the acquisition,
                    # see self.prepare_poll
                    pass
            elif i in self._acq_units_used:
                log.warning(f'{self.name}: acquisition unit {i} is used '
                            f'without an AWG program. This might result in '
                            f'not triggering the acquisition unit.')

    def _acquisition_set_weight(self, channel, weight):
        self.qachannels[channel[0]].readout.integrations[channel[1]].weights(
            weight[0].copy() + 1j * weight[1].copy())

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        properties = super().get_value_properties(
            data_type=data_type, acquisition_length=acquisition_length)
        # if data_type == 'raw':
        if 'raw' in data_type:
            if acquisition_length is None:
                raise ValueError('Please specify acquisition_length.')
            # Units are only valid when using SSB or DSB demodulation.
            # value corresponds to the peak voltage of a cosine with the
            # demodulation frequency.
            if data_type == 'raw_corr':
                # Note that V^2 is in brackets to prevent confusion with unit
                # prefixes
                properties['value_unit'] = '(V^2)'
            else:
                properties['value_unit'] = 'Vpeak'
            properties['scaling_factor'] = 1 # Set separately in poll()
        elif data_type == 'lin_trans':
            properties['value_unit'] = 'a.u.'
            properties['scaling_factor'] = 1
        elif 'digitized' in data_type:
            properties['value_unit'] = 'frac'
            properties['scaling_factor'] = 1
        else:
            raise ValueError(f'Data type {data_type} not understood.')
        return properties