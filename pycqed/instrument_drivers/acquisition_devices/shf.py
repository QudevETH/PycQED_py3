import numpy as np
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
from pycqed.measurement import sweep_functions as swf
from pycqed.instrument_drivers.acquisition_devices.base import \
    ZI_AcquisitionDevice
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
    zhinst_qcodes_wrappers import ZHInstMixin, ZHInstSGMixin
from zhinst.qcodes import SHFQA as SHFQA_core
from zhinst.qcodes import SHFQC as SHFQC_core
from zhinst.qcodes import AveragingMode
from pycqed.utilities.timer import Timer
import logging
log = logging.getLogger(__name__)


class SHF_AcquisitionDevice(ZI_AcquisitionDevice, ZHInstMixin):
    """QuDev-specific PycQED driver for the ZI SHF instrument series

    This is not meant to be instantiated directly, but should be inherited
    from by the actual instrument classes

    Attributes:
        awg_active (dict of bool): Whether the AWG of each acquisition unit has
            been started by Pulsar (keys are indices of acquisition units).
        _acq_scope_memory (int): Number of points that the scope can acquire
            in one hardware run.
    """
    acq_length_granularity = 16

    # acq_sampling_rate is the effective sampling rate provided by the SHFQA,
    # even though internally it has an ADC running at 4e9 Sa/s.
    # More details on the chain of downconversions on the SHFQA input:
    # Signal at center_freq+/-1e9 Hz
    # (or +/-700e6 is the actual analog bandwidth)
    # DC by center_freq+12e9 and filter -> 12e9+/-1e9
    # DC by 9e9 and filter -> 3e9+/-1e9
    # Acq at 4e9 Sa/s (f_Nyq=2e9) -> aliasing to 1e9+/-1e9
    # Digital DC by 1e9 -> 0+/-1e9 (I/Q signal)
    # (this is not a symmetrical signal in f, hence I/Q)
    acq_sampling_rate = 2.0e9
    _acq_scope_memory = 2 ** 18
    acq_weights_n_samples = 4096
    acq_Q_sign = -1  # Determined experimentally
    allowed_modes = {'avg': [],  # averaged raw input (time trace) in V
                     'int_avg': ['raw', 'digitized'],  # FIXME data types unused
                     # Scope is distinct from avg in the UHF, not here. For
                     # compatibility, we allow this mode here.
                     'scope': ['timedomain', 'fft_power', ],
                     }
    # private lookup dict to translate a data_type to an index understood by
    # the SHF
    res_logging_indices = {'raw': 1,  # raw integrated+averaged results
                           'digitized': 3,  # thresholded results (0 or 1)
                           }
    USER_REG_LOOP_COUNT = 0
    USER_REG_ACQ_LEN = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chs = kwargs.get('valid_qachs', range(len(self.qachannels)))
        self._all_qachs = self.qachannels
        self._valid_qachs = {i: self._all_qachs[i] for i in chs}
        ZI_AcquisitionDevice.__init__(self, *args, **kwargs)
        self.n_acq_units = len(self.qachannels)
        self.lo_freqs = {i: None for i in self._valid_qachs}  # re-create
        self.n_acq_int_channels = self.max_qubits_per_channel
        self._reset_acq_poll_inds()
        # Mode of the acquisition units ('readout' or 'spectroscopy')
        # This is different from self._acq_mode (allowed_modes)
        self._acq_units_modes = {}
        self._awg_program = [None]*self.n_acq_units
        self._awg_source_strings = {}
        self.timer = None

        self.awg_active = {i: False for i in self._valid_qachs}

        self.add_parameter(
            'allowed_lo_freqs',
            initial_value=np.arange(1e9, 8.1e9, 100e6),
            parameter_class=ManualParameter,
            docstring='List of values that the center frequency (LO) is '
                      'allowed to take. As of now this is limited to steps '
                      'of 100 MHz.',
            set_parser=lambda x: list(np.atleast_1d(x).flatten()),
            vals=validators.MultiType(validators.Lists(), validators.Arrays(),
                                      validators.Numbers()))
        # FIXME: This parameter should be removed and existing code should be
        #  refactored to use the parameter "{awg_name}_use_hardware_sweeper" in
        #  pulsar.
        self.add_parameter(
            'use_hardware_sweeper',
            initial_value=False,
            parameter_class=ManualParameter,
            docstring='Bool indicating whether the hardware sweeper should '
                      'be used in spectroscopy mode',
            vals=validators.Bool())

        self.add_parameter(
            'acq_trigger_delay',
            initial_value=0,
            parameter_class=ManualParameter,
            docstring='Delay between the pulse generation and the acquisition.',
            vals=validators.Numbers(min_value=0))

        self.add_parameter(
           'timeout',
           unit='s',
           initial_value=30,
           parameter_class=ManualParameter,
           docstring='Timeout when waiting for scope data.',
           vals=validators.Ints())

        self.add_parameter(
            'allow_scope',
            initial_value=(
                    len(self.qachannels) == len(self._all_qachs)),
            parameter_class=ManualParameter,
            docstring='Whether access to the scope module is allowed. When '
                      'sharing an SHF device between two setup, this should '
                      'only be set to True while the other setup is inactive.',
            vals=validators.Bool())

    def __getattribute__(self, item):
        # returns only valid qa channels instead of all qa-channels
        # if attribute valid_qachs is explicitly specified during the init.
        if item == 'qachannels' and hasattr(self, '_valid_qachs'):
            return self._valid_qachs
        return super().__getattribute__(item)

    def _reset_acq_poll_inds(self):
        """Resets the data indices that have been acquired until now.

        self._acq_poll_inds will be set to a dict of lists of zeros,
            with dict keys being indices of acquisition units and list entries
            corresponding to the integration channels of the acquisition unit.
        """
        self._acq_poll_inds = {}
        for i in self.qachannels:
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            self._acq_poll_inds[i] = [0] * len(channels)

    def set_lo_freq(self, acq_unit, lo_freq):
        super().set_lo_freq(acq_unit, lo_freq)
        # Deep set (synchronous set that returns the value acknowledged by the
        # device)
        self.qachannels[acq_unit].centerfreq(lo_freq, deep=True)
        new_lo_freq = self.qachannels[acq_unit].centerfreq()
        if np.abs(new_lo_freq - lo_freq) > 1:
            log.warning(f'{self.name}: center frequency {lo_freq/1e6:.6f} '
                        f'MHz not supported. Setting center frequency to '
                        f'{new_lo_freq/1e6:.6f} MHz. This does NOT '
                        f'automatically set the IF!')

    def prepare_poll_before_AWG_start(self):
        super().prepare_poll_before_AWG_start()
        for i in self._acq_units_used:
            if not self.awg_active[i]:
                log.warning(f'{self.name}: acquisition unit {i} is used '
                            f'without an AWG program. This might result in '
                            f'not triggering the acquisition unit.')
            if self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'readout':
                self.qachannels[i].readout.run()
            elif self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'spectroscopy':
                self.qachannels[i].spectroscopy.run()
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'fft_power':
                pass  # FIXME currently done in poll
            elif (self._acq_mode == 'scope'
                  and self._acq_data_type == 'timedomain')\
                    or self._acq_mode == 'avg':
                self._arm_scope()
        self._reset_acq_poll_inds()

    def get_sweep_points_spectrum(self, acquisition_length=None, lo_freq=0):
        """Returns the sweep points that will be measured by a software PSD
        measurement.

        For now this only considers acquiring at the normal sampling rate, and
        without e.g. sweeping the (internal) LO.
        This function might not be needed anymore once we use the soft PSD
        from the ZI qcodes.
        """
        if acquisition_length is None:
            acquisition_length = self._acq_length
        acq_n_results = self.convert_time_to_n_samples(acquisition_length)
        freqs = np.roll(np.fft.fftfreq(acq_n_results, 1/self.acq_sampling_rate),
                        int(acq_n_results / 2))
        freqs = freqs + lo_freq
        return freqs

    def get_params_for_spectrum(self, requested_freqs,
                                get_closest_lo_freq=(lambda x: x)):
        """Convenience method for retrieving parameters needed to measure a
        spectrum

        Args:
            requested_freqs (list of double): frequencies to be measured.
                Note that the effectively measured frequencies will be a
                rounded version of these values.
            get_closest_lo_freq (function): a function that takes an LO
                frequency as argument and returns the closest allowed LO
                frequency. This can be used to provide limitations imposed
                by higher-layer settings to the driver.
        """
        # For rounding reasons, we can't measure exactly on these frequencies.
        # Here we extract the frequency spacing and the frequency range
        # (center freq and bandwidth)
        allowed_lo_freqs = np.unique([get_closest_lo_freq(f)
                                      for f in self.allowed_lo_freqs()])
        if len(requested_freqs) == 1:
            id_closest = (np.abs(np.array(allowed_lo_freqs) -
                                requested_freqs[0])).argmin()
            if allowed_lo_freqs[id_closest] - requested_freqs[0] < 10e6:
                # resulting mod_freq would be smaller than 10 MHz
                # TODO: arbitrarily chosen limit of 10 MHz
                id_closest = id_closest + (-1 if id_closest != 0 else +1)
            delta_f = 0
            acq_length = None
        else:
            diff_f = np.diff(requested_freqs)
            if not all(np.abs(diff_f-diff_f[0]) < 1e-3):
                # not equally spaced (arbitrary 1 mHz)
                log.warning(f'Unequal frequency spacing not supported, '
                            f'the measurement will return equally spaced values.')
            # Find closest allowed center frequency
            approx_center_freq = np.mean(requested_freqs)
            id_closest = (np.abs(np.array(allowed_lo_freqs) -
                                approx_center_freq)).argmin()
            # Compute needed acq length
            delta_f = np.mean(diff_f)
            # Note that this might underestimate the necessary acq_length to get
            # the correct freq precision (because of rounding, hardware
            # limitations, etc.)
            acq_length = 1 / delta_f
        center_freq = allowed_lo_freqs[id_closest]
        # Compute the actual needed bandwidth
        min_bandwidth = 2 * max(np.abs(requested_freqs - center_freq))
        if min_bandwidth > self.acq_sampling_rate:
            raise NotImplementedError('Spectrum wider than the bandwidth of '
                                      'the SHF is not yet implemented!')
        # center freq and delta_f are used by the hardware sweeper,
        # acq_length by the software PSD using timetraces
        return center_freq, delta_f, acq_length

    def _acq_unit_exists(self, acq_unit):
        return acq_unit in self.qachannels

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        super().acquisition_initialize(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type)

        self._acq_units_used = list(np.unique([ch[0] for ch in channels]))
        self._acq_units_modes = {i: self.qachannels[i].mode().name  # Caching
                                 for i in self._acq_units_used}

        # Set the scope trigger delay with respect to pulse generation
        self.scopes[0].trigger.delay(self.acq_trigger_delay())

        for i in self.qachannels:
            # Set trigger delay to the same value for all modes. This is
            # necessary e.g. to get consistent acquisition weights.
            self.qachannels[i].readout.integration.delay(
                self.acq_trigger_delay())
            self.qachannels[i].spectroscopy.delay(
                self.acq_trigger_delay())
            # Make sure the readout is stopped. It will be started in
            # prepare_poll
            self.qachannels[i].readout.stop()  # readout mode
            if i not in self._acq_units_used:
                # In spectroscopy mode there seems to be no stop() functionality
                # meaning that the output generator must be always running.
                # Here it is effectively disabled by oscillator_gain(0)
                self.qachannels[i].oscs[0].gain(0)  # spectroscopy mode

        log.debug(f'{self.name}: units used: ' + repr(self._acq_units_used))
        for i in self._acq_units_used:
            self.qachannels[i].generator.userregs[0].value(
                self._acq_loop_cnt)  # Used in seqc code
            # TODO: should probably decide actions based on the data type, not
            #  also on the acq unit physical mode
            if self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'readout':
                if data_type is not None:
                    self.qachannels[i].readout.result.source(
                        self.res_logging_indices[data_type])
                if self._acq_length is not None:
                    self.qachannels[i].readout.integration.length(
                        self.convert_time_to_n_samples(self._acq_length))
                # Readout mode outputs programmed waveforms, and integrates the
                # input with different custom weights for each integration
                # channel
                self.qachannels[i].readout.configure_result_logger(
                    result_length=self._acq_n_results,
                    num_averages=self._acq_averages,
                    result_source="result_of_integration",
                    averaging_mode=AveragingMode.CYCLIC,
                )
            elif self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'spectroscopy':
                self.qachannels[i].oscs[0].gain(1.0)
                self.qachannels[i].spectroscopy.length(
                    self.convert_time_to_n_samples(self._acq_length))
                if self._awg_program[self._get_awg_program_index(i)]:
                    # assume that this sequencer program includes triggering
                    # for the spectroscopy
                    self.qachannels[i].spectroscopy.trigger.channel(
                        f'channel{i}_sequencer_trigger0')
                else:
                    # no sequencer will be running. Use the trigger that
                    # would otherwise trigger the sequencer as a trigger for
                    # the spectroscopy.
                    self.qachannels[i].spectroscopy.trigger.channel(
                        self.qachannels[i].generator.auxtriggers[0].channel())
                # Spectroscopy mode outputs a modulated pulse, whose envelope
                # is programmed by pulsar and whose modulation frequency is
                # given by the sum of the configured center frequency and
                # intermediate frequency, and integrates the input by weighting
                # with the same waveform (without the envelope)
                self.qachannels[i].spectroscopy.configure_result_logger(
                    result_length=self._acq_n_results,
                    num_averages=self._acq_averages,
                    averaging_mode=AveragingMode.CYCLIC,
                )
                self.qachannels[i].generator.userregs[1].value(
                    # Used in seqc code
                    self.convert_time_to_n_samples(self._acq_length)
                )
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'fft_power':
                # Fit as many traces as possible in a single SHF call
                # FIXME this should be disabled when measuring a synchronous
                #  signal instead of noise
                num_points_per_trace = self.convert_time_to_n_samples(
                    self._acq_length)
                num_traces_per_run = int(np.floor(self._acq_scope_memory /
                                                  num_points_per_trace))
                num_points_per_run = num_traces_per_run * num_points_per_trace
                # should avg in software, (hard avg not implemented)
                self._initialize_scope(acq_unit=i, num_hard_avg=1,
                                       num_points_per_run=num_points_per_run)
            elif (self._acq_mode == 'scope'
                  and self._acq_data_type == 'timedomain')\
                    or self._acq_mode == 'avg':
                # Concatenation of traces in one run not supported for now
                # (would need to think about ergodicity)
                num_points_per_run = self.convert_time_to_n_samples(
                    self._acq_length)
                self._initialize_scope(acq_unit=i,
                                       num_hard_avg=self._acq_averages,
                                       num_points_per_run=num_points_per_run)
            else:
                raise NotImplementedError("Mode not recognised!")

    def _initialize_scope(self, acq_unit, num_hard_avg, num_points_per_run):
        num_segments = 1  # for segmented averaging (several triggers per time
        # trace) compensation for the delay between generator output and input
        # of the integration unit
        self.qachannels[acq_unit].mode("readout")
        if len(self._acq_units_used)>1:
            log.warning("Parallel measurements might lead to timing "
                        "discrepancies, since the whole scope is triggered by "
                        "a single acquisition unit.")
        trigger_channel = f'channel{acq_unit}_sequencer_monitor0'
        self.scopes[0].configure(
            input_select={acq_unit: f"channel{acq_unit}_signal_input"},
            num_samples=num_points_per_run,
            trigger_input=trigger_channel,
            num_segments=num_segments,
            num_averages=num_hard_avg,
            # trigger_delay for now defaults to 0 in this function...
            trigger_delay=self.scopes[0].trigger.delay(),
        )

    def acquisition_finalize(self):
        super().acquisition_finalize()
        # Use a transaction since qcodes does not support wildcards
        # Could use the ZI toolkit instead:
        # self._tk_object.qachannels["*"].oscs[0].gain(0)
        with self.set_transaction():
            for ch in self.qachannels.values():
                ch.oscs[0].gain(0)

    def acquisition_progress(self):
        n_acq = {}
        for i in self._acq_units_used:
            if self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'readout':
                n_acq[i] = self.qachannels[i].readout.result.acquired()
            elif self._acq_mode == 'int_avg' \
                    and self._acq_units_modes[i] == 'spectroscopy':
                n_acq[i] = self.qachannels[i].spectroscopy.result.acquired()
            elif self._acq_mode == 'scope' \
                    and self._acq_data_type == 'fft_power':
                return None  # intermediate progress not implemented
            elif (self._acq_mode == 'scope' and self._acq_data_type ==
                  'timedomain') or self._acq_mode == 'avg':
                return None  # intermediate progress not implemented
            else:
                raise NotImplementedError("Mode not recognised!")
        return np.mean(list(n_acq.values()))

    def _get_awg_program_index(self, acq_unit):
        """Return which index of _awg_program corresponds to an acq_unit"""
        return list(self._valid_qachs.keys()).index(acq_unit)

    def set_awg_program(self, acq_unit, awg_program, waves_to_upload=None):
        """
        Program the internal AWGs
        """
        qachannel = self.qachannels[acq_unit]
        self._awg_program[self._get_awg_program_index(acq_unit)] = awg_program
        self.store_awg_source_string(qachannel, awg_program)
        qachannel.generator.load_sequencer_program(awg_program)
        if waves_to_upload is not None:
            # upload waveforms
            qachannel.generator.write_to_waveform_memory(waves_to_upload)

    def has_awg_program(self, acq_unit):
        """Returns whether an acquisition unit has an AWG program

        Args:
            acq_unit (int): index of the acquisition unit
        """
        return bool(self._awg_program[self._get_awg_program_index(acq_unit)])

    def store_awg_source_string(self, channel, awg_str):
        """
        Store AWG source strings to a private property for debugging.

        This function is called automatically when programming a QA
        channel via set_awg_program and currently still needs to be called
        manually after programming an SG channel. The source strings get
        stored in the dict self._awg_source_strings.

        Args:
             channel: the QA or SG channel object for which the AWG was
                programmed
            awg_str: the source string that was programmed to the AWG
        """
        key = channel.short_name[:2] + channel.short_name[-1:]
        self._awg_source_strings[key] = awg_str

    def _arm_scope(self):
        self.scopes[0].stop()
        self.scopes[0].run(single=1)

    @Timer()
    def poll(self, *args, **kwargs):
        # 220118 For now, poll reads all data available from the data server
        # at each run, and then returns only the newer data to match the
        # normal behaviour of poll. One could implement an actual poll after
        # ZI has improved the drivers, if that turns out to be a bottleneck.
        # sqrt(2) are because the SHF seems to return integrated RMS voltages.

        # TODO (from ZI) poll is availabe on the new zhinst-qcodes driver,
        # might be worthwhile considering since it is much faster.
        # The polling is implemented on the session directly and not the
        # device!
        # subscribing  works on every "qcodes-node"
        # e.g. self.qachannels[0].centerfreq.subscribe()
        #
        # To poll the subscribed data use the session
        # result = self.session.poll()
        #
        # The polled data contain ALL nodes subscribed in a session!
        # The result is a dictionary qcodes_node:Data
        # e.g. result[self.qachannels[0].centerfreq] returns the data for the
        # earlier subscribed center frequency
        # https://docs.zhinst.com/zhinst-toolkit/en/latest/first_steps/nodetree.html#Subscribe-/-Unsubscribe
        # This is a toolkit example but it works similar in qcodes.

        dataset = {}
        for i in self._acq_units_used:
            channels = [ch[1] for ch in self._acquisition_nodes if ch[0] == i]
            if self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'readout':

                # Comments from ZI (didn't investigate since we'll likely
                # remove this poll function, see comments above):
                # self.qachannels[i].readout.read() is blocking now but would
                # be the simplest solution.
                # QCoDeS does not support wildcards and therefor needs to get
                # each wave individually.
                # Using toolkit in this case speeds up the progress quite a lot:
                # res = self._tk_object.qachannels[i].readout.result.data["*"].wave()
                # res = [res[i] for i in channels]
                # Alternatively use qcodes in a loop
                res = []
                for channel in channels:
                    res.append(self.qachannels[i].readout.result.data[
                                   channel].wave())

                # In readout mode the data isn't rescaled yet in the SHF
                # by the number of points
                scaling_factor = 1 / (self.acq_sampling_rate * self._acq_length)
                dataset.update(
                    {(i, ch): [np.real(res[n][self._acq_poll_inds[i][n]:])
                               * scaling_factor]
                     for n, ch in enumerate(channels)})
                self._acq_poll_inds[i] = [len(res[n]) for n in range(len(
                    channels))]
            elif self._acq_mode == 'int_avg'\
                    and self._acq_units_modes[i] == 'spectroscopy':
                progress = self.qachannels[i].spectroscopy.result.acquired()
                if progress >= self._acq_loop_cnt * self._acq_n_results:
                    data = self.qachannels[i].spectroscopy.result.data.wave()
                    data = [[np.real(a), np.imag(a)] for a in data]
                    scaling_factor = 1
                    dataset.update({(i, ch): [[a[n % 2] * scaling_factor
                                              for a in data]]
                                    for n, ch in enumerate(channels)})
            elif self._acq_mode == 'scope'\
                    and self._acq_data_type == 'fft_power':
                if not channels == [0, 1]:  # TODO: one channel in TWPA object
                    raise ValueError(
                        "Currently the scope only works with two data "
                        "channels. This will be cleaned up after integrating "
                        "measurements on TWPA objects.")
                # The SHF acquires at full memory, then we get as many traces
                # as possible from that (this could be avoided e.g. if a few
                # points only are needed, in case this slows down measuring)
                num_points_per_trace = self.convert_time_to_n_samples(
                    self._acq_length)
                num_traces_per_run = int(np.floor(self._acq_scope_memory /
                                                  num_points_per_trace))
                num_runs = int(np.ceil(self._acq_averages/num_traces_per_run))
                if self._acq_n_results != num_points_per_trace:
                    raise ValueError(
                        "This driver for now makes the simplest assumption "
                        "that the number of sweep points (number of points "
                        "in the spectrum) is the same as the length of the "
                        "timetraces (to simply do FFT time->freq). To measure "
                        "a different spectrum e.g. if you need LO sweeping or "
                        "downsampling, please extend this."
                    )
                timetraces = np.array([])
                for _ in range(num_runs):
                    self._arm_scope()
                    # FIXME this is blocking, to get enough data to average
                    #  in the driver (not the usual behaviour of poll)
                    self.scopes[0].wait_done(timeout=self.timeout())
                    data = self.scopes[0].channels[i].wave()
                    # This is a 1-D complex time trace
                    timetraces = np.concatenate((timetraces, data))
                timetraces = timetraces[
                             :self._acq_averages*num_points_per_trace]
                timetraces = np.reshape(timetraces, (self._acq_averages,
                                                     num_points_per_trace))
                # 'norm' by 1/num_samples_per_trace to get the amplitude
                v_peak = np.fft.fft(timetraces,
                                    norm="forward")
                v_peak_rolled = np.roll(v_peak, int(num_points_per_trace / 2))
                v_peak_squared = np.mean(np.abs(v_peak_rolled) ** 2, axis=0)
                power_spectrum = 10 * np.log10(v_peak_squared / (2 * 50) / 1e-3)
                dataset.update({(i, 0): [power_spectrum]})
                dataset.update({(i, 1): [0*power_spectrum]})  # I don't care
            elif (self._acq_mode == 'scope' and self._acq_data_type ==
                  'timedomain') or self._acq_mode == 'avg':
                if self.scopes[0].enable() == 0:
                    timetrace = self.scopes[0].channels[i].wave()
                    dataset.update({(i, 0): [np.real(timetrace)]})
                    # use sign convention as is used by UHFQA in avg mode
                    # to ensure compatibility with existing analysis classes
                    # use natural sign in averaged mode
                    sign = {'avg': -1, 'scope': 1}[self._acq_mode]
                    dataset.update({(i, 1): [sign*np.imag(timetrace)]})
            else:
                raise NotImplementedError("Mode not recognised!")
        return dataset

    def get_lo_sweep_function(self, acq_unit, ro_mod_freq,
                              get_closest_lo_freq):
        name = 'Readout frequency'
        if self.use_hardware_sweeper():
            sf = swf.SpectroscopyHardSweep(parameter_name=name)
        else:
            name_offset = 'Readout frequency with offset'
            sf = swf.Offset_Sweep(swf.MajorMinorSweep(
                self.qachannels[acq_unit].centerfreq,
                swf.Offset_Sweep(
                    self.qachannels[acq_unit].oscs[0].freq,
                    ro_mod_freq),
                np.unique([get_closest_lo_freq(f)
                           for f in self.allowed_lo_freqs()]),
                name=name_offset, parameter_name=name_offset),
                -ro_mod_freq, name=name, parameter_name=name)
        sf.includes_IF_sweep = True
        return sf

    def stop(self):
        # Use a transaction since qcodes does not support wildcards
        # Or in toolkit: self._tk_object.qachannels["*"].generator.enable(False)
        with self.set_transaction():
            for ch in self.qachannels.values():
                ch.generator.enable(False)

    def start(self, **kwargs):
        for i, ch in self.qachannels.items():
            if self.awg_active[i]:  # Outputs a waveform
                if self._awg_program[self._get_awg_program_index(i)]:
                    # Using the sequencer
                    # These 2 lines replace ...enable_sequencer(single=True)
                    # which also checks that it started, but sometimes fails
                    # (for short sequences?)
                    ch.generator.single(True)
                    ch.generator.enable(1, deep=True)
                else:
                    # No AWG needs to be started if the acq unit has no program
                    pass

    def _check_allowed_acquisition(self):
        super()._check_allowed_acquisition()
        if self._acq_mode == 'scope' and not self.allow_scope():
            raise ValueError(
                'Trying to access scope module while forbidden by qcodes '
                'parameter allow_scope.')

    def _check_hardware_limitations(self):
        super()._check_hardware_limitations()
        n_samples = self.convert_time_to_n_samples(self._acq_length)
        if self._acq_mode == 'scope' and n_samples > self._acq_scope_memory:
            raise ValueError(
                f'Acquisition device {self.name} ({self.devname}): '
                f'Acquisition length {self._acq_length} corresponds to '
                f'{n_samples} > {self._acq_scope_memory}, which is not '
                f'supported in scope mode.')
        if n_samples != self.convert_time_to_n_samples(self._acq_length, True):
            raise ValueError(
                f'Acquisition device {self.name} ({self.devname}): '
                f'Acquisition length {self._acq_length} corresponds to '
                f'{n_samples} samples, which is not a multiple of the '
                f'granularity {self.acq_length_granularity}.')


    def acquisition_set_weights(self, channels, **kw):
        # Makes super call faster
        with self.set_transaction():
            super().acquisition_set_weights(channels, **kw)

    def _acquisition_set_weight(self, channel, weight):
        self.qachannels[channel[0]].readout.integration.weights[channel[1]]\
            .wave(weight[0].copy() + 1j * weight[1].copy())

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        properties = super().get_value_properties(
            data_type=data_type, acquisition_length=acquisition_length)
        properties['scaling_factor'] = 1  # Set separately in poll()
        return properties


class SHFQA(SHFQA_core, SHF_AcquisitionDevice):
    """QuDev-specific PycQED driver for the ZI SHFQA
    """

    def __init__(self, *args, **kwargs):
        valid_qachs = kwargs.pop('valid_qachs', None)
        self._check_server(kwargs)
        super().__init__(*args, **kwargs)
        if valid_qachs is not None:
            kwargs['valid_qachs'] = valid_qachs
        SHF_AcquisitionDevice.__init__(self, *args, **kwargs)


class SHFQC(SHFQC_core, SHF_AcquisitionDevice, ZHInstSGMixin):
    """QuDev-specific PycQED driver for the ZI SHFQC
    """

    def __init__(self, serial, *args, **kwargs):
        daq = self._check_server(kwargs)
        if daq is not None:
            daq.set_device_type(serial, 'SHFQC')
        super().__init__(serial, *args, **kwargs)
        SHF_AcquisitionDevice.__init__(self, *args, **kwargs)
        self._awg_program += [None] * len(self.sgchannels)
        self._sgchannel_sine_enable = [False] * len(self.sgchannels)

    def start(self, **kwargs):
        SHF_AcquisitionDevice.start(self, **kwargs)
        ZHInstSGMixin.start(self)

    def stop(self):
        SHF_AcquisitionDevice.stop(self)
        ZHInstSGMixin.stop(self)