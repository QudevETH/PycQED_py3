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
    """QuDev-specific PycQED driver for the ZI SHFQA

    This is the QuDev-specific PycQED driver for the 2 GSa/s SHFQA instrument
    from Zurich Instruments AG.

    Attributes:
        allowed_lo_freqs (list of floats): List of values that the centre frequency
            (LO) is allowed to take. As of now this is limited to steps
            of 100 MHz
        awg_active (list of bool): Whether the AWG of each acquisition unit has
            been started by Pulsar.
    """
    # acq_length_granularity = 4 #FIXME should this be set to some value?

    # acq_sampling_rate is the effective sampling rate provided by the SHF,
    # even though internally it has an ADC running at 4e9 Sa/s.
    # More details on the chain of downconversions on the SHF input:
    # Signal at center_freq+/-1e9 Hz
    # (or +/-700e6 is the actual analog bandwidth)
    # DC by center_freq+12e9 and filter -> 12e9+/-1e9
    # DC by 9e9 and filter -> 3e9+/-1e9
    # Acq at 4e9 Sa/s (f_Nyq=2e9) -> aliasing to 1e9+/-1e9
    # Digital DC by 1e9 -> 0+/-1e9 (I/Q signal)
    # (this is not a symmetrical signal in f, hence I/Q)
    acq_sampling_rate = 2.0e9
    acq_weights_n_samples = 4096 #TODO: is this the maximum in readout mode?
    acq_Q_sign = -1 # Determined experimentally
    allowed_modes = {#'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw', 'digitized'],
                     # 'scope': [],
                     }
    # private lookup dict to translate a data_type to an index understood by
    # the SHF
    res_logging_indices = {'raw': 1,  # raw integrated+averaged results
                           'digitized': 3,  # thresholded results (0 or 1)
                           }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ZI_AcquisitionDevice.__init__(self, *args, **kwargs)
        self.n_acq_units = len(self.qachannels)
        self.lo_freqs = [None] * self.n_acq_units  # re-create with correct length
        self._acq_loop_cnts_last = [None] * self.n_acq_units
        self.n_acq_int_channels = len(self.qachannels[0].readout.integrations)
        self._reset_acq_poll_inds()
        self._acq_units_modes = {}

        self.awg_active = [False] * self.n_acq_units
        self._awg_programs = {}
        self._waves_to_upload = {}

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
        """Returns the ZI data server (DAQ).
        """
        return self._controller._controller._connection.daq

    def _reset_acq_poll_inds(self):
        """Resets the data indices that have been acquired until now.

        self._acq_poll_inds will be set to a list of lists of zeros,
            with first dimension the number of acquisition units
            and second dimension the number of integration channels
            used per acquisition unit.
        """
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
                        f'{new_lo_freq/1e6:.6f} MHz. This does NOT'
                        f'automatically set the IF!')

    def prepare_poll(self):
        super().prepare_poll()
        for i in self._acq_units_used:
            # Readout mode outputs programmed waveforms, and integrates
            # against different custom weights for each integration channel
            if self._acq_units_modes[i] == 'readout':
                self.qachannels[i].readout.arm(length=self._acq_n_results,
                                               averages=self._acq_averages)
            # Spectroscopy mode outputs a sine wave at a given frequency
            # and integrates against this same signal only
            else:  # spectroscopy
                self._arm_spectroscopy(i, length=self._acq_n_results,
                                       averages=self._acq_averages)
        self._reset_acq_poll_inds()

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acq_units_modes = {i: self.qachannels[i].mode()
                                 for i in self._acq_units_used}

        for i in range(self.n_acq_units):
            # Make sure the readout is stopped. It will be started in
            # prepare_poll
            self.qachannels[i].readout.stop()  # readout mode
            if i not in self._acq_units_used:
                # In spectroscopy mode there seems to be no stop() functionality
                # meaning that the output generator must be always running.
                # Here it is effectively disabled by oscillator_gain(0)
                self.qachannels[i].sweeper.oscillator_gain(0)  # spectroscopy mode

        log.debug(f'{self.name}: units used: ' + repr(self._acq_units_used))
        for i in self._acq_units_used:
            if self._acq_units_modes[i] == 'readout':
                # Disable rerun; the AWG seqc program defines the number of
                # iterations in the loop
                self.qachannels[i].generator.single(1)
                if data_type is not None:
                    self.qachannels[i].readout.result_source(
                        self.res_logging_indices[data_type])
                if acquisition_length is not None:
                    self.qachannels[i].readout.integration_length(
                        self.convert_time_to_n_samples(acquisition_length))
                # This is needed because the SHF currently lacks user
                # registers and thus cannot be programmed by Pulsar,
                # which is instead done in self._program_awg
                if self._acq_loop_cnts_last[i] != loop_cnt:
                    self._program_awg(i)  # reprogramm this acq unit
                    self._acq_loop_cnts_last[i] = loop_cnt
            else:  # spectroscopy
                self.qachannels[i].sweeper.oscillator_gain(1.0)
                self.daq.setInt(
                    self._get_spectroscopy_node(i, "integration_length"),
                    self.convert_time_to_n_samples(acquisition_length))

    def acquisition_finalize(self):
        super().acquisition_finalize()
        for ch in self.qachannels:
            ch.sweeper.oscillator_gain(0)

    def set_awg_program(self, acq_unit, awg_program, waves_to_upload):
        """Receive sequence data from Pulsar.

         This will be uploaded in self.acquisition_initialize.
         The reason is that awg_program is incomplete as Pulsar
         does not know yet the loop count, which is replaced
         in self._program_awg.
        """
        self._awg_programs[acq_unit] = awg_program
        self._waves_to_upload[acq_unit] = waves_to_upload
        # force programming in acquisition_initialize
        self._acq_loop_cnts_last[acq_unit] = None

    def _program_awg(self, acq_unit):
        awg_program = self._awg_programs.get(acq_unit, None)
        if awg_program is None:
            return
        qachannel = self.qachannels[acq_unit]
        # This is now known and can be replaced in the seqc code
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
        # These direct accesses to the zhinst-toolkit were simpler to
        # implement based on the SHF documentation than finding
        # the corresponding higher-level functions in zhinst-qcodes
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
        # 220118 For now, poll reads all data available from the data server
        # at each run, and then returns only the newer data to match the
        # normal behaviour of poll. One could implement an actual poll after
        # ZI has improved the drivers, if that turns out to be a bottleneck.
        # sqrt(2) values are because the SHF seems to return RMS voltages.
        dataset = {}
        for i in self._acq_units_used:
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            if self._acq_units_modes[i] == 'readout':
                res = self.qachannels[i].readout.read(integrations=channels,
                                                      blocking=False)
                # In readout mode the data isn't rescaled yet in the SHF
                # by the number of points
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
        properties['scaling_factor'] = 1 # Set separately in poll()
        return properties