import numpy as np
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import sweep_functions as swf
from pycqed.instrument_drivers.acquisition_devices.base import \
    ZI_AcquisitionDevice
from zhinst.qcodes import SHFQA as SHFQA_core
from pycqed.utilities.timer import Timer, Checkpoint
import logging
import json
import time
log = logging.getLogger(__name__)


class SHFQA(SHFQA_core, ZI_AcquisitionDevice):
    """QuDev-specific PycQED driver for the ZI SHFQA

    This is the QuDev-specific PycQED driver for the 2 GSa/s SHFQA instrument
    from Zurich Instruments AG.

    Attributes:
        awg_active (list of bool): Whether the AWG of each acquisition unit has
            been started by Pulsar.
        _acq_scope_memory: #FIXME is this the correct number?
    """
    acq_length_granularity = 16 #FIXME should this be set to some value?

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
    _acq_scope_memory = 2 ** 18
    acq_weights_n_samples = 4096 #FIXME: is this the maximum in readout mode?
    acq_Q_sign = -1 # Determined experimentally
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw', 'digitized'], #FIXME data types unused
                     # Scope is distinct from avg in the UHF, not here. For
                     # compatibility, we allow this mode here.
                     'scope': ['spectrum', 'timetrace', ],
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
        # Mode of the acquisition units ('readout' or 'spectroscopy')
        # This is different from self._acq_mode (allowed_modes)
        self._acq_units_modes = {}
        self.timer = None

        self.awg_active = [False] * self.n_acq_units
        self._awg_programs = {}
        self._waves_to_upload = {}

        self.add_parameter(
            'allowed_lo_freqs',
            initial_value=np.arange(1e9, 8.1e9, 100e6),
            parameter_class=ManualParameter,
            docstring='List of values that the centre frequency (LO) is '
                      'allowed to take. As of now this is limited to steps '
                      'of 100 MHz.',
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
                        f'{new_lo_freq/1e6:.6f} MHz. This does NOT '
                        f'automatically set the IF!')

    def prepare_poll(self):
        super().prepare_poll()
        for i in self._acq_units_used:
            if self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'readout':
                # Readout mode outputs programmed waveforms, and integrates
                # against different custom weights for each integration channel
                self.qachannels[i].readout.arm(length=self._acq_n_results,
                                               averages=self._acq_averages)
            elif self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'spectroscopy':
                # Spectroscopy mode outputs a sine wave at a given frequency
                # and integrates against this same signal only
                self._arm_spectroscopy(i, length=self._acq_n_results,
                                       averages=self._acq_averages)
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'spectrum':
                pass  # FIXME currently done in poll
            elif (self._acq_mode == 'scope'
                  and self._acq_data_type == 'timetrace')\
                    or self._acq_mode == 'avg':
                # FIXME should this set the length and avg number?
                self._arm_scope()
        self._reset_acq_poll_inds()

    def get_sweep_points_spectrum(self, acquisition_length=None, lo_freq=0):
        """

        For now this only considers acquiring at the normal sampling rate, and
        without e.g. sweeping the (internal) LO
        """
        if acquisition_length is None:
            acquisition_length = self._acq_length
        acq_n_results = self.convert_time_to_n_samples(acquisition_length)
        freqs = np.roll(np.fft.fftfreq(acq_n_results, 1/self.acq_sampling_rate),
                        int(acq_n_results / 2))
        freqs = freqs + lo_freq
        return freqs

    def get_params_from_spectrum(self, requested_freqs):
        """Convenience method for retrieving a center frequency and acquisition
        length needed to measure a spectrum

        """
        # For rounding reasons, we can't measure exactly on these frequencies.
        # Here we extract the frequency spacing and the frequency range
        # (center freq and bandwidth)
        diff_f = np.diff(requested_freqs)
        if not all(diff_f-diff_f[0]<1):  # not equally spaced (arbitrary 1 Hz)
            log.warning(f'Unequal frequency spacing not supported, '
                        f'the measurement will return equally spaced values.')
        # Find closest allowed center frequency
        approx_center_freq = np.mean(requested_freqs)
        id_closest = (np.abs(np.array(self.allowed_lo_freqs()) -
                             approx_center_freq)).argmin()
        center_freq = self.allowed_lo_freqs()[id_closest]
        # Compute the actual needed bandwidth
        min_bandwidth = 2 * max(np.abs(requested_freqs - center_freq))
        if min_bandwidth > self.acq_sampling_rate:
            raise NotImplementedError('Spectrum wider than the bandwidth of '
                                      'the SHF is not yet implemented!')
        # Compute needed acq length
        max_delta_f = min(diff_f)  # Should measure at least to this precision
        acq_length = 1 / max_delta_f
        # Note that this minimum needed acq_length is not strictly correct
        # currently, and should be further rounded up to really give at least
        # this precision in frequency. This is because the hardware cannot
        # exactly acquire for that duration.
        return acq_length

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acq_units_modes = {i: self.qachannels[i].mode()  # Caching
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
            # This is needed because the SHF currently lacks user
            # registers and thus cannot be programmed by Pulsar,
            # which is instead done in self._program_awg
            if self._acq_loop_cnts_last[i] != loop_cnt:
                self._program_awg(i)  # reprogramm this acq unit
                self._acq_loop_cnts_last[i] = loop_cnt
            # TODO: should probably decide actions based on the data type, not
            #  also on the acq unit physical mode
            if self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'readout':
                # Disable rerun; the AWG seqc program defines the number of
                # iterations in the loop
                self.qachannels[i].generator.single(1)
                if data_type is not None:
                    self.qachannels[i].readout.result_source(
                        self.res_logging_indices[data_type])
                if self._acq_length is not None:
                    self.qachannels[i].readout.integration_length(
                        self.convert_time_to_n_samples(self._acq_length))
            elif self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'spectroscopy':
                self.qachannels[i].sweeper.oscillator_gain(1.0)
                self.daq.setInt(
                    self._get_spectroscopy_node(i, "integration_length"),
                    self.convert_time_to_n_samples(self._acq_length))
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'spectrum':
                # Fit as many traces as possible in a single SHF call
                num_points_per_trace = self.convert_time_to_n_samples(
                    self._acq_length)
                num_traces_per_run = int(np.floor(self._acq_scope_memory /
                                                  num_points_per_trace))
                num_points_per_run = num_traces_per_run * num_points_per_trace
                # should avg in software, (hard avg not implemented)
                self._initialize_scope(acq_unit=i, nr_hard_avg=1,
                                       num_points_per_run=num_points_per_run)
            elif (self._acq_mode == 'scope'
                  and self._acq_data_type == 'timetrace')\
                    or self._acq_mode == 'avg':
                # Concatenation of traces in one run not supported for now
                # (would need to think about ergodicity)
                num_points_per_run = self.convert_time_to_n_samples(
                    self._acq_length)
                self._initialize_scope(acq_unit=i,
                                       nr_hard_avg=self._acq_averages,
                                       num_points_per_run=num_points_per_run)
            else:
                raise NotImplementedError

    # Used in acquisition_initialize in modes that use the scope.
    # This might be replaceable by a couple of lines from the qcodes driver
    def _initialize_scope(self, acq_unit, nr_hard_avg, num_points_per_run):
        num_segments = 1  # for segmented averaging (several triggers per time
        # trace) compensation for the delay between generator output and input
        # of the integration unit
        trigger_delay = 200e-9
        self.qachannels[acq_unit].mode('readout')
        self.qachannels[acq_unit].input('on')
        self.qachannels[acq_unit].output('on')
        self.daq.setInt(f"/{self._serial}/scopes/0/segments/count",
                        num_segments)
        if num_segments > 1:
            self.daq.setInt(f"/{self._serial}/scopes/0/segments/enable", 1)
        else:
            self.daq.setInt(f"/{self._serial}/scopes/0/segments/enable", 0)
        if nr_hard_avg > 1:
            self.daq.setInt(f"/{self._serial}/scopes/0/averaging/enable", 1)
        else:
            self.daq.setInt(f"/{self._serial}/scopes/0/averaging/enable", 0)
        self.daq.setInt(f"/{self._serial}/scopes/0/averaging/count",
                        nr_hard_avg)
        self.daq.setInt(f"/{self._serial}/scopes/0/length",
                        num_points_per_run)

        self.daq.setInt(f"/{self._serial}/scopes/0/channels/*/enable", 0)
        input_select = {acq_unit: f"channel{acq_unit}_signal_input"}
        for scope_ch, acq_unit_path in input_select.items():
            self.daq.setString(
                f"/{self._serial}/scopes/0/channels/{scope_ch}/inputselect",
                acq_unit_path)
            self.daq.setInt(
                f"/{self._serial}/scopes/0/channels/{scope_ch}/enable",
                1)
        self.daq.setDouble(f"/{self._serial}/scopes/0/trigger/delay",
                           trigger_delay)

    def acquisition_finalize(self):
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

    def _arm_scope(self):
        path = f"/{self._serial}/scopes/0/enable"
        if self.daq.getInt(path) == 1:
            self.daq.setInt(path, 0)
            self._controller._assert_node_value(path, 0, timeout=self.timeout())
        self.daq.syncSetInt(path, 1)

    @Timer()
    def poll(self, *args, **kwargs):
        # 220118 For now, poll reads all data available from the data server
        # at each run, and then returns only the newer data to match the
        # normal behaviour of poll. One could implement an actual poll after
        # ZI has improved the drivers, if that turns out to be a bottleneck.
        # sqrt(2) are because the SHF seems to return integrated RMS voltages.
        dataset = {}
        for i in self._acq_units_used:
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            if self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'readout':
                res = self.qachannels[i].readout.read(integrations=channels,
                                                      blocking=False)
                # In readout mode the data isn't rescaled yet in the SHF
                # by the number of points
                scaling_factor = np.sqrt(2) \
                                 / (self.acq_sampling_rate * self._acq_length)
                dataset.update(
                    {(i, ch): [np.real(res[n][self._acq_poll_inds[i][n]:])
                               * scaling_factor]
                     for n, ch in enumerate(channels)})
                self._acq_poll_inds[i] = [len(res[n]) for n in range(len(channels))]
            elif self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'spectroscopy':
                progress = self.daq.getInt(
                    self._get_spectroscopy_node(i, "acquired"))
                if progress == self._acq_averages:
                    node = self._get_spectroscopy_node(i, "data")
                    data = self.daq.get(node, flat=True).get(node, [])
                    data = [a.get('vector', a) for a in data]
                    data = [[np.real(a), np.imag(a)] for a in data]
                    scaling_factor = np.sqrt(2)
                    dataset.update({(i, ch): [a[n % 2] * scaling_factor
                                              for a in data]
                                    for n, ch in enumerate(channels)})
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'spectrum':
                if not channels == [0, 1]:  # TODO: one channel in TWPA object
                    raise ValueError()
                # The SHF acquires at full memory, then we get as many traces
                # as possible from that (this could be avoided e.g. if a few
                # points only are needed, in case this slows down measuring)
                num_points_per_run = self.convert_time_to_n_samples(
                    self._acq_length)
                num_traces_per_run = int(np.floor(self._acq_scope_memory /
                                                  num_points_per_run))
                num_runs = int(np.ceil(self._acq_averages/num_traces_per_run))
                if self._acq_n_results != num_points_per_run:
                    raise ValueError("This driver for now makes the simplest" +
                                     "assumption that the number of sweep" +
                                     "points (number of points in the" +
                                     "spectrum) is the same as the length " +
                                     "of the timetraces (to simply do FFT"
                                     "time->freq). To measure a different"
                                     "spectrum e.g. if you need LO sweeping"
                                     "or downsampling, please extend this.")
                timetraces = np.array([])
                for _ in range(num_runs):
                    self._arm_scope()
                    # FIXME this is blocking, to get enough data to average
                    #  in the driver (not the usual behaviour of poll)
                    self._controller._assert_node_value(
                        f"/{self._serial}/scopes/0/enable", 0,
                        timeout=self.timeout())
                    path = f"/{self._serial}/scopes/0/channels/{i}/wave"
                    data = self.daq.get(path.lower(), flat=True)[path][0]["vector"]
                    # This is a 1-D complex time trace
                    timetraces = np.concatenate((timetraces, data))
                timetraces = timetraces[:self._acq_averages*num_points_per_run]
                timetraces = np.reshape(timetraces, (self._acq_averages,
                                                     num_points_per_run))
                # 'norm' by 1/num_samples_per_trace to get the amplitude
                v_peak = np.fft.fft(timetraces,
                                    norm="forward")
                v_peak_rolled = np.roll(v_peak, int(num_points_per_run / 2))
                v_peak_squared = np.mean(np.abs(v_peak_rolled) ** 2, axis=0)
                power_spectrum = 10 * np.log10(v_peak_squared / (2 * 50) / 1e-3)
                dataset.update({(i, 0): [power_spectrum]})
                dataset.update({(i, 1): [0*power_spectrum]})  # I don't care
            elif (self._acq_mode == 'scope' and self._acq_data_type == 'timetrace')\
                    or self._acq_mode == 'avg':
                if self.daq.getInt(f"/{self._serial}/scopes/0/enable") == 0:
                    path = f"/{self._serial}/scopes/0/channels/{i}/wave"
                    timetrace = self.daq.get(path.lower(), flat=True)[path][0]["vector"]
                    dataset.update({(i, 0): [np.real(timetrace)]})
                    dataset.update({(i, 1): [np.imag(timetrace)]})
            else:
                raise NotImplementedError
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

    def snapshot(self, update=False, detailed=False):
        # Extends the base method to request all possible details from the data server if necessary.
        # Would that work similarly for any other ZI instrument?
        if detailed==False:
            return super().snapshot(update)
        else:  # update is not used in this case
            nodes = json.loads(self.daq.listNodesJSON('/' + self.devname))
            shf_settings = {k: self.daq.get(k, settingsonly=False, flat=True) for k in nodes}
            # Remove timestamp data to help diffing
            for key, s in shf_settings.items():
                for same_key in list(s.keys()):  # s is a dict that contains a single key, the same as in shf_settings
                    try:
                        s[same_key].pop('timestamp', None)
                    except:
                        try:
                            for item in s[same_key]:
                                item.pop('timestamp', None)
                        except:  # If impossible to remove, just leave that in the output
                            pass
            return shf_settings
