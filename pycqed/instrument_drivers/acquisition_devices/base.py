import numpy as np
from copy import deepcopy
import logging
log = logging.getLogger(__name__)


class AcquisitionDevice():
    """
    TODO

    note: the acq dev should have a qcode param timeout
    """

    n_acq_units = 1
    n_acq_channels = 1
    acq_length_granularity = 1
    acq_sampling_rate = None
    acq_max_trace_samples = None
    acq_Q_sign = 1
    allowed_modes = []
    allowed_weights_types = ['optimal', 'optimal_qutrit', 'SSB',
                             'DSB', 'DSB2', 'square_rot']

    def __init__(self, *args, **kwargs):
        self._acquisition_nodes = []
        self._acq_mode = None
        self._acq_data_type = None
        self._reset_n_acquired()
        self.lo_freqs = [None] * self.n_acq_units
        self._acq_units_used = []

    def set_lo_freq(self, acq_unit, lo_freq):
        self.lo_freqs[acq_unit] = lo_freq

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        self._acquisition_initialize_base(
            channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type=None)

    def _acquisition_initialize_base(
            self, channels, n_results, averages, loop_cnt,
            mode, acquisition_length, data_type=None):

        self._acquisition_nodes = []
        self._acq_length = acquisition_length
        self._acq_channels = channels
        for ch in channels:
            if ch[0] not in range(self.n_acq_units):
                raise ValueError(f'{self.name}: Acquisition unit {ch[0]} '
                                 f'does not exist.')
            if ch[1] not in range(self.n_acq_channels):
                raise ValueError(f'{self.name}: Acquisition channel {ch[0]} '
                                 f'does not exist.')
        self._acq_n_results = n_results
        self._acq_averages = averages
        self._acq_loop_cnt = loop_cnt
        self._acq_mode = mode
        self._acq_data_type = data_type

        self._reset_n_acquired()
        self._check_allowed_acquisition()
        self._check_hardware_limitations()

    def acquisition_finalize(self):
        pass

    def _reset_n_acquired(self):
        """
        Reset quantities that are needed by aquisition_progress.
        """
        pass

    def _check_allowed_acquisition(self):
        """
        Check whether the specified mode and data_type is supported by the
        acquisition device.
        """

        if self._acq_mode not in self.allowed_modes:
            raise ValueError(f'{self._acq_mode} not supported by {self.name}.')

        if self._acq_data_type is not None and \
                self._acq_data_type not in self.allowed_modes[self._acq_mode]:
            raise ValueError(f'{self._acq_data_type} not supported by '
                             f'{self.name} for an {self._acq_mode} '
                             f'acquisition.')

    def acquisition_nodes(self):
        return deepcopy(self._acquisition_nodes)

    def aquisition_progress(self):
        return 0  # no intermediate progress information available

    def prepare_poll(self):
        """
        This function is called by PollDetector.poll_data right before
        starting the AWGs (i.e., we can rely on that the AWGs have already
        been stopped) for final preparations. This is the place to arm the
        acquisition device (i.e., let it wait for triggers).
        """
        pass

    def poll(self, poll_time=0.1):
        raise NotImplementedError(f'poll not implemented '
                                  f'for {self.__class__.__name__}')

    def start(self, **kw):
        """
        Start the built-in AWG (if present).
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        pass  # default acquisition device does not have an AWG

    def stop(self):
        """
        Stop the built-in AWG (if present).
        """
        pass  # default acquisition device does not have an AWG

    def _check_hardware_limitations(self):
        """
        This method should be implemented in child classes to check whether
        the measurement settings are supported by the hardware.
        """
        pass

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        return {'value_unit': 'a.u.', 'scaling_factor': 1}

    def correct_offset(self, channels, data):
        return data

    def get_lo_sweep_function(self, acq_unit, ro_mod_freq):
        raise NotImplementedError(f'{self.__class__.__name__} does not have '
                                  f'an internal LO.')

    def convert_time_to_n_samples(self, time, align_acq_granularity=False):
        """
        Calculate the number of samples that the specified time corresponds to
        based on the acquisition sampling rate of the device.
        :param time: time in seconds (float)
        :return: number of samples (int)
        """
        n_samples = int(round(time * self.acq_sampling_rate))
        if align_acq_granularity:
            # align to acquisition granularity grid
            n_samples = n_samples - (n_samples % self.acq_length_granularity)
        return n_samples

    def convert_n_samples_to_time(self, n_samples):
        """
        Calculate the duration that the specified number of samples corresponds
        to based on the acquisition sampling rate of the device.
        :param n_samples: number of samples (int)
        :return: time in seconds (float)
        """
        return n_samples / self.acq_sampling_rate

    def get_sweep_points_time_trace(self, acquisition_length=None):
        """
        Get the measurement sweep points expected by the framework for a
        time trace acquisition with the UHFQA.

        The sweep points array will contain time points between 0 and
        acquisition_length. The number of sweep points in the array will be
        equal to the number of samples that acquisition length converts to
        for this device (see convert_time_to_n_samples).

        :param acquisition_length: duration of time trace acquisition in seconds
        :return: array of sweep points
        """
        if acquisition_length is None:
            acquisition_length = self._acq_length
        npoints = self.convert_time_to_n_samples(acquisition_length)
        return np.linspace(0, acquisition_length, npoints, endpoint=False)

    def acquisition_set_weights(self, channels, **kw):
        """
        Set the acquisition weights for a (pair of) weighted integration
        channel(s).

        :param channels: (list of tuple of two int) weighted integration
            channels to be used. Each entry is a tuple of the acquisition
            unit index and the index of the weighted integration channel.
        :param kw: information about the weights to be set, passed to
            self._acquisition_generate_weights
        """
        weights = self._acquisition_generate_weights(**kw)
        if len(channels) > 2:
            log.warning(f'{self.name}: Integration weights can be set only '
                        f'for two channels. Ignoring additional channels.')
            channels = channels[:2]
        if len(weights) > len(channels):
            log.warning(f'{self.name}: Cannot configure {len(weights)} pairs '
                        f'of weights because only {len(channels)} channels '
                        f'were provided. Ignoring additional pairs.')
        for ch, w in zip(channels, weights):
            self._acquisition_set_weight(ch, w)

    def _acquisition_generate_weights(self, weights_type, mod_freq=None,
                                      acq_IQ_angle=0,
                                      weights_I=(), weights_Q=()):
        """
        Generates integration weights based on the provided settings.
        :param weights_type: (str) the type of weights can be:
            - manual: do not set any weights (keep what is configured in the
                acquisition device)
            - optimal: use the optimal weights for single-channel integration
                provided in weights_I[0], weights_Q[0].
            - optimal_qutrit: use the optimal weights for two-channel
                integration provided in weights_I[:2], weights_Q[:2]
            - SSB: single-sideband demodulation using two integration
                channels and both physical input channels.
            - DSB: double-sideband demodulation from first physical input
                channel using two integration channels. Doesn't allow to
                distinguish positive and negative sideband.
            - DSB2: double-sideband demodulation from second physical input
                channel using two integration channels. Doesn't allow to
                distinguish positive and negative sideband.
            - square_rot: same as the first integration channel of a SSB
                demodulation (uses one integration channel, but both physical
                input channels)
        :param mod_freq: (float or None) The modulation frequency (in Hz) when
            using weights type SSB, DSB, DSB, or square_rot. In these cases
            it must not be None. In other cases, it is ignored.
        :param acq_IQ_angle: (float) The phase (in rad) of the integration
            weights when using weights type SSB, DSB, DSB, or square_rot.
            Ignored in other cases.
        :param weights_I: (list/tuple of np.array or None) The i-th entry of
            the list defines the weights for first physical input channel
            used in the i-th integration channel. Must have (at least) one
            entry if weights type is optimal and two entries if weights type
            is optimal_qutrit. Unneeded entries are ignored, and the whole
            list is ignored for other weights types.
        :param weights_Q: (list of np.array or None) Like weights_I,
            but for the second physical input channel.
        """
        if weights_type == 'manual':
            return []
        if weights_type not in self.allowed_weights_types:
            raise ValueError(f'Weights type {weights_type} not supported by '
                             f'{self.name}.')
        aQs = self.acq_Q_sign
        for wt, n_chan in [('optimal', 1), ('optimal_qutrit', 2)]:
            if weights_type == wt:
                if (len(weights_I) < n_chan or len(weights_Q) < n_chan
                        or any([w is None for w in weights_I[:n_chan]])
                        or any([w is None for w in weights_Q[:n_chan]])):
                    log.warning(f'Not enough weights provided for weights '
                                f'type "{wt}". Not setting integration '
                                f'weights.')
                    return []
                else:
                    # FIXME: take acq_Q_sign into account here?
                    return list(zip(weights_I[:n_chan], weights_Q[:n_chan]))

        if mod_freq is None:
            log.warning(f'No modulation frequency provided. Not setting '
                        f'integration weights.')
            return []
        tbase = np.arange(
            0, self.acq_max_trace_samples / self.acq_sampling_rate,
            1 / self.acq_sampling_rate)
        cosI = np.cos(2 * np.pi * mod_freq * tbase + acq_IQ_angle)
        sinI = np.sin(2 * np.pi * mod_freq * tbase + acq_IQ_angle)
        if weights_type == 'SSB':
            return [(cosI, -sinI), (sinI * aQs, cosI * aQs)]
        elif weights_type in 'DSB':
            z = np.zeros_like(cosI)
            return [(cosI, z), (sinI * aQs, z)]
        elif weights_type == 'DSB2':
            z = np.zeros_like(cosI)
            return [(z, -sinI), (z, cosI * aQs)]
        elif weights_type == 'square_rot':
            return [(cosI, -sinI)]
        else:
            raise KeyError('Invalid weights type: {}'.format(weights_type))

    def _acquisition_set_weight(self, channel, weight):
        raise NotImplementedError(f'_acquisition_set_weight not implemented '
                                  f'for {self.__class__.__name__}')


class ZI_AcquisitionDevice(AcquisitionDevice):
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
            properties['scaling_factor'] = 1 / (self.acq_sampling_rate
                                                * acquisition_length)
        elif data_type == 'lin_trans':
            properties['value_unit'] = 'a.u.'
            properties['scaling_factor'] = 1
        elif 'digitized' in data_type:
            properties['value_unit'] = 'frac'
            properties['scaling_factor'] = 1
        else:
            raise ValueError(f'Data type {data_type} not understood.')
        return properties

