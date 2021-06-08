"""
Library containing various pulse shapes.
"""

import sys
import numpy as np
import scipy as sp
from pycqed.measurement.waveform_control import pulse
import logging

log = logging.getLogger(__name__)

pulse.pulse_libraries.add(sys.modules[__name__])


class SSB_DRAG_pulse(pulse.Pulse):
    """In-phase Gaussian pulse with derivative quadrature and SSB modulation.

    Modulation and mixer predistortion added with `apply_modulation` function.

    Args:
        name (str): Name of the pulse, used for referencing to other pulses in a
            sequence. Typically generated automatically by the `Segment` class.
        element_name (str): Name of the element the pulse should be played in.
        I_channel (str): In-phase output channel name.
        Q_channel (str): Quadrature output channel name.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
        amplitude (float): Pulse amplitude in Volts. Defaults to 0.1 V.
        sigma (float): Pulse width standard deviation in seconds. Defaults to
            250 ns.
        nr_sigma (float): Pulse clipping length in units of pulse sigma. Total
            pulse length will be `nr_sigma*sigma`. Defaults to 4.
        motzoi (float): Amplitude of the derivative quadrature in units of
            pulse sigma. Defautls to 0.
        mod_frequency (float): Pulse modulation frequency in Hz. Defaults to
            1 MHz.
        phase (float): Pulse modulation phase in degrees. Defaults to 0.
        phaselock (bool): The phase reference time is the start of the algorithm
            if True and the middle of the pulse otherwise. Defaults to True.
        alpha (float): Ratio of the I_channel and Q_channel output. Defaults to
            1.
        phi_skew (float): Phase offset between I_channel and Q_channel, in
            addition to the nominal 90 degrees. Defaults to 0.
    """

    def __init__(self, element_name, I_channel, Q_channel,
                 name='SSB Drag pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.I_channel = I_channel
        self.Q_channel = Q_channel

        self.phaselock = kw.pop('phaselock', True)

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These
        parameters are set upon calling the super().__init__ method.
        """
        params = {
            'pulse_type': 'SSB_DRAG_pulse',
            'I_channel': None,
            'Q_channel': None,
            'amplitude': 0.1,
            'sigma': 10e-9,
            'nr_sigma': 5,
            'motzoi': 0,
            'mod_frequency': 1e6,
            'phase': 0,
            'alpha': 1,
            'phi_skew': 0,
        }
        return params

    @property
    def channels(self):
        return [c for c in [self.I_channel, self.Q_channel] if c is not None]

    @property
    def length(self):
        return self.sigma * self.nr_sigma

    def chan_wf(self, channel, tvals):
        half = self.nr_sigma * self.sigma / 2
        tc = self.algorithm_time() + half

        gauss_env = np.exp(-0.5 * (tvals - tc) ** 2 / self.sigma ** 2)
        gauss_env -= np.exp(-0.5 * half ** 2 / self.sigma ** 2)
        gauss_env *= self.amplitude * (tvals - tc >= -half) * (
                tvals - tc < half)
        deriv_gauss_env = -self.motzoi * (tvals - tc) * gauss_env / self.sigma

        if self.mod_frequency is not None:
            I_mod, Q_mod = apply_modulation(
                gauss_env, deriv_gauss_env, tvals, self.mod_frequency,
                phase=self.phase, phi_skew=self.phi_skew, alpha=self.alpha,
                tval_phaseref=0 if self.phaselock else tc)
        else:
            # Ignore the Q component and program the I component to both
            # channels. See HDAWG8Pulsar._hdawg_mod_setter
            I_mod, Q_mod = gauss_env, gauss_env

        if channel == self.I_channel:
            return I_mod
        elif channel == self.Q_channel:
            return Q_mod
        else:
            return np.zeros_like(tvals)

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [channel == self.I_channel, self.amplitude, self.sigma]
        hashlist += [self.nr_sigma, self.motzoi, self.mod_frequency]
        phase = self.phase
        if self.mod_frequency is not None:
            phase += 360 * self.phaselock * self.mod_frequency * (
                    self.algorithm_time() + self.nr_sigma * self.sigma / 2)
            hashlist += [self.alpha, self.phi_skew, phase]
        return hashlist


class SSB_DRAG_pulse_with_cancellation(SSB_DRAG_pulse):
    """
    SSB Drag pulse with copies with scaled amp. and offset phase on extra
    channels intended for interferometrically cancelling on-device crosstalk.

    Args:
        name (str): Name of the pulse, used for referencing to other pulses in a
            sequence. Typically generated automatically by the `Segment` class.
        element_name (str): Name of the element the pulse should be played in.
        I_channel (str): In-phase output channel name.
        Q_channel (str): Quadrature output channel name.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
        amplitude (float): Pulse amplitude in Volts. Defaults to 0.1 V.
        sigma (float): Pulse width standard deviation in seconds. Defaults to
            250 ns.
        nr_sigma (float): Pulse clipping length in units of pulse sigma. Total
            pulse length will be `nr_sigma*sigma`. Defaults to 4.
        motzoi (float): Amplitude of the derivative quadrature in units of
            pulse sigma. Defautls to 0.
        mod_frequency (float): Pulse modulation frequency in Hz. Defaults to
            1 MHz.
        phase (float): Pulse modulation phase in degrees. Defaults to 0.
        phaselock (bool): The phase reference time is the start of the algorithm
            if True and the middle of the pulse otherwise. Defaults to True.
        alpha (float): Ratio of the I_channel and Q_channel output. Defaults to
            1.
        phi_skew (float): Phase offset between I_channel and Q_channel, in
            addition to the nominal 90 degrees. Defaults to 0.
        cancellation_params (dict): a parameter dictionary for cancellation
            drives. The keys of the dictionary should be tuples of I- and Q-
            channel names for the cancellation and the values should be
            dictionaries of parameter values. Possible parameters to override
            are 'amplitude', 'phase', 'delay', 'mod_frequency', 'phi_skew',
            'alpha' and 'phaselock'. Cancellation amplitude is a scaling factor
            for the main pulse amplitude and phase is a phase offset. The delay
            is relative to the main pulse.
    """

    @classmethod
    def pulse_params(cls):
        params = super().pulse_params()
        params.update({'cancellation_params': {}})
        return params

    @property
    def channels(self):
        channels = super().channels
        for i, q in self.cancellation_params.keys():
            channels += [i, q]
        return channels

    def chan_wf(self, channel, tvals):
        if channel in [self.I_channel, self.Q_channel]:
            return super().chan_wf(channel, tvals)
        iq_idx = -1
        for (i, q), p in self.cancellation_params.items():
            if channel in [i, q]:
                iq_idx = [i, q].index(channel)
                cpars = p
                break
        if iq_idx == -1:
            return np.zeros_like(tvals)

        half = self.nr_sigma * self.sigma / 2
        tc = self.algorithm_time() + half + cpars.get('delay', 0.0)

        gauss_env = np.exp(-0.5 * (tvals - tc) ** 2 / self.sigma ** 2)
        gauss_env -= np.exp(-0.5 * half ** 2 / self.sigma ** 2)
        gauss_env *= self.amplitude * (tvals - tc >= -half) * (
                tvals - tc < half)
        gauss_env *= cpars.get('amplitude', 1.0)
        deriv_gauss_env = -self.motzoi * (tvals - tc) * gauss_env / self.sigma

        return apply_modulation(
            gauss_env, deriv_gauss_env, tvals,
            cpars.get('mod_frequency', self.mod_frequency),
            phase=self.phase + cpars.get('phase', 0.0),
            phi_skew=cpars.get('phi_skew', self.phi_skew),
            alpha=cpars.get('alpha', self.alpha),
            tval_phaseref=0 if cpars.get('phaselock', self.phaselock)
                else tc)[iq_idx]

    def hashables(self, tstart, channel):
        if channel in [self.I_channel, self.Q_channel]:
            return super().hashables(tstart, channel)
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        for (i, q), cpars in self.cancellation_params.items():
            if channel != i and channel != q:
                continue
            hashlist += [channel == i]
            hashlist += [self.amplitude*cpars.get('amplitude', 1.0)]
            hashlist += [self.sigma]
            hashlist += [self.nr_sigma, self.motzoi]
            hashlist += [cpars.get('mod_frequency', self.mod_frequency)]
            phase = self.phase + cpars.get('phase', 0)
            phase += 360 * cpars.get('phaselock', self.phaselock) * \
                     cpars.get('mod_frequency', self.mod_frequency) * (
                        self.algorithm_time() + self.nr_sigma * self.sigma / 2 +
                        + cpars.get('delay', 0.0))
            hashlist += [cpars.get('alpha', self.alpha)]
            hashlist += [cpars.get('phi_skew', self.phi_skew), phase]
            hashlist += [cpars.get('delay', 0)]
            return hashlist
        return []


class GaussianFilteredPiecewiseConstPulse(pulse.Pulse):
    """
    The base class for different Gaussian-filtered piecewise constant pulses.

    To avoid clipping of the Gaussian-filtered rising and falling edges, the
    pulse should start and end with zero-amplitude buffer segments.

    Args:
        name (str): The name of the pulse, used for referencing to other pulses
            in a sequence. Typically generated automatically by the `Segment`
            class.
        element_name (str): Name of the element the pulse should be played in.
        channels (list of str): Channel names this pulse is played on
        lengths (list of list of float): For each channel, a list of the
            lengths of the pulse segments. Must satisfy
            `len(lengths) == len(channels)`.
        amplitudes (list of list of float): The amplitudes of all pulse
            segments. The shape must match that of `lengths`.
        gaussian_filter_sigma (float): The width of the gaussian filter sigma
            of the pulse.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
    """
    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These
        parameters are set upon calling the super().__init__ method.
        """
        params = {
            'pulse_type': 'GaussianFilteredPiecewiseConstPulse',
            'channels': None,
            'lengths': None,
            'amplitudes': 0,
            'gaussian_filter_sigma': 0,
        }
        return params

    @property
    def length(self):
        max_len = 0
        for channel_lengths in self.lengths:
            max_len = max(max_len, np.sum(channel_lengths))
        return max_len

    def _check_dimensions(self):
        if len(self.lengths) != len(self.channels):
            raise ValueError("Lengths list doesn't match channels list")
        if len(self.amplitudes) != len(self.channels):
            raise ValueError("Amplitudes list doesn't match channels list")
        for chan_lens, chan_amps in zip(self.lengths, self.amplitudes):
            if len(chan_lens) != len(chan_amps):
                raise ValueError("One of the amplitudes lists doesn't match "
                                 "the corresponding lengths list")

    def chan_wf(self, channel, t):
        self._check_dimensions()

        t0 = self.algorithm_time()
        idx = self.channels.index(channel)
        wave = np.zeros_like(t)

        if self.gaussian_filter_sigma > 0:
            timescale = 1 / (np.sqrt(2) * self.gaussian_filter_sigma)
        else:
            timescale = 0

        for seg_len, seg_amp in zip(self.lengths[idx], self.amplitudes[idx]):
            t1 = t0 + seg_len
            if self.gaussian_filter_sigma > 0:
                wave += 0.5 * seg_amp * (sp.special.erf((t - t0) * timescale) -
                                         sp.special.erf((t - t1) * timescale))
            else:
                wave += seg_amp * (t >= t0) * (t < t1)
            t0 = t1
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        idx = self.channels.index(channel)
        chan_lens = self.lengths[idx]
        chan_amps = self.amplitudes[idx]
        hashlist += [len(chan_lens)]
        hashlist += list(chan_lens.copy())
        hashlist += list(chan_amps.copy())
        hashlist += [self.gaussian_filter_sigma]
        return hashlist


class NZTransitionControlledPulse(GaussianFilteredPiecewiseConstPulse):
    """
    A zero-area pulse shape that allows to control the accumulated phase when
    transitioning from the first pulse half to the second pulse half, by having
    an additional, low-amplitude segment between the two main pulse-halves.

    The zero area is achieved by adjusting the lengths for the intermediate
    pulses.
    """
    def __init__(self, element_name, name='NZTC pulse', **kw):
        super().__init__(name, element_name, **kw)
        self._update_lengths_amps_channels()

    @classmethod
    def pulse_params(cls):
        params = {
            'pulse_type': 'NZTransitionControlledPulse',
            'channel': None,
            'channel2': None,
            'amplitude': 0,
            'amplitude2': 0,
            'amplitude_offset': 0,
            'amplitude_offset2': 0,
            'pulse_length': 0,
            'trans_amplitude': 0,
            'trans_amplitude2': 0,
            'trans_length': 0,
            'buffer_length_start': 30e-9,
            'buffer_length_end': 30e-9,
            'channel_relative_delay': 0,
            'gaussian_filter_sigma': 1e-9,
        }
        return params

    def _update_lengths_amps_channels(self):
        self.channels = [c for c in [self.channel, self.channel2]
                         if c is not None]
        self.lengths = []
        self.amplitudes = []

        for ma, ta, ao, d, c in [
            (self.amplitude, self.trans_amplitude, self.amplitude_offset,
             -self.channel_relative_delay/2, self.channel),
            (self.amplitude2, self.trans_amplitude2, self.amplitude_offset2,
             self.channel_relative_delay/2, self.channel2),
        ]:
            if c is None:
                continue
            ml = self.pulse_length
            tl = self.trans_length
            bs = self.buffer_length_start
            be = self.buffer_length_end
            ca0 = ma + ao
            ca1 = -ma + ao
            cl0 = max(-(ml * ao) / ca0, 0) if ca0 else 0
            cl1 = max(-(ml * ao) / ca1, 0) if ca1 else 0

            self.amplitudes.append([0, ma + ao, ta, -ta, -ma + ao, 0])
            self.lengths.append([bs + d - cl0, cl0 + ml / 2, tl / 2,
                                 tl / 2, ml / 2 + cl1, be - d - cl1])

    def chan_wf(self, channel, t):
        self._update_lengths_amps_channels()
        return super().chan_wf(channel, t)


class BufferedSquarePulse(pulse.Pulse):
    def __init__(self,
                 element_name,
                 channel=None,
                 channels=None,
                 name='buffered square pulse',
                 **kw):
        super().__init__(name, element_name, **kw)

        # Set channels
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channels.append(channel)
        else:
            self.channels = channels

        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These
        parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedSquarePulse',
            'channel': None,
            'channels': [],
            'amplitude': 0,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= self.algorithm_time() + self.buffer_length_start)
            wave *= (tvals <
                     self.algorithm_time() + self.buffer_length_start +
                     self.pulse_length)
            return wave
        else:
            tstart = self.algorithm_time() + self.buffer_length_start
            tend = tstart + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * self.amplitude
            return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [self.amplitude, self.pulse_length]
        hashlist += [self.buffer_length_start, self.buffer_length_end]
        hashlist += [self.gaussian_filter_sigma]
        return hashlist


class BufferedCZPulse(pulse.Pulse):
    def __init__(self,
                 channel,
                 element_name,
                 aux_channels_dict=None,
                 name='buffered CZ pulse',
                 **kw):
        super().__init__(name, element_name, **kw)

        # Set channels
        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedCZPulse',
            'channel': None,
            'aux_channels_dict': None,
            'amplitude': 0,
            'frequency': 0,
            'phase': 0,
            'square_wave': False,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'extra_buffer_aux_pulse': 5e-9,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if chan != self.channel:
            amp = self.aux_channels_dict[chan]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * amp
            wave *= (tvals >= self.algorithm_time() + buffer_start)
            wave *= (tvals < self.algorithm_time() + buffer_start + pulse_length)
        else:
            tstart = self.algorithm_time() + buffer_start
            tend = tstart + pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * amp
        t_rel = tvals - tvals[0]
        carrier = np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
        if self.square_wave:
            carrier = np.sign(carrier)
        wave *= carrier
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.frequency, self.phase % 360, self.square_wave]
        return hashlist


class NZBufferedCZPulse(pulse.Pulse):
    def __init__(self, channel, element_name, aux_channels_dict=None,
                 name='NZ buffered CZ pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.length1 = self.alpha * self.pulse_length / (self.alpha + 1)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end


    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'NZBufferedCZPulse',
            'channel': None,
            'aux_channels_dict': None,
            'amplitude': 0,
            'alpha': 1,
            'frequency': 0,
            'phase': 0,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'extra_buffer_aux_pulse': 5e-9,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        amp1 = self.amplitude
        amp2 = -self.amplitude * self.alpha
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        l1 = self.length1
        if chan != self.channel:
            amp1 = self.aux_channels_dict[chan] * amp1
            amp2 = -amp1 * self.alpha
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse
            l1 = self.alpha * pulse_length / (self.alpha + 1)

        if self.gaussian_filter_sigma == 0:
            wave1 = np.ones_like(tvals) * amp1
            wave1 *= (tvals >= tvals[0] + buffer_start)
            wave1 *= (tvals < tvals[0] + buffer_start + l1)

            wave2 = np.ones_like(tvals) * amp2
            wave2 *= (tvals >= tvals[0] + buffer_start + l1)
            wave2 *= (tvals < tvals[0] + buffer_start + pulse_length)

            wave = wave1 + wave2
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + l1
            tend2 = tvals[0] + buffer_start + pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (amp1 * sp.special.erf((tvals - tstart) * scaling) -
                          amp1 * sp.special.erf((tvals - tend) * scaling) +
                          amp2 * sp.special.erf((tvals - tend) * scaling) -
                          amp2 * sp.special.erf((tvals - tend2) * scaling))
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma, self.alpha]
        return hashlist

class BufferedNZFLIPPulse(pulse.Pulse):
    def __init__(self, channel, channel2, element_name, aux_channels_dict=None,
                 name='Buffered FLIP Pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.channel = channel
        self.channel2 = channel2
        self.channels = [self.channel, self.channel2]

        # buffer when fluxing one qubit until the other qubit is fluxed
        self.flux_buffer = {channel: self.flux_buffer_length2,
                            channel2: self.flux_buffer_length}

        self.amps = {channel: self.amplitude, channel2: self.amplitude2}

        alpha1 = self.alpha
        alpha2 = self.alpha
        self.alphas = {channel: alpha1, channel2: alpha2}

        self.length1 = {channel: alpha1*self.pulse_length/(alpha1 + 1)\
                                 + 2*self.flux_buffer[channel2],
                        channel2: alpha2*self.pulse_length/(alpha2 + 1)\
                                  + 2*self.flux_buffer[channel]}

        self.length2 = {channel: self.pulse_length/(alpha1 + 1)\
                                 + 2*self.flux_buffer[channel2],
                        channel2: self.pulse_length/(alpha2 + 1)\
                                  + 2*self.flux_buffer[channel]}

        delay = self.channel_relative_delay  # delay of pulse on channel2 wrt pulse on channel
        bls = self.buffer_length_start  # initial value for buffer length start passed with kw
        ble = self.buffer_length_end  # initial value for buffer length end passed with kw

        # Compute new buffer lengths taking into account channel skewness and additional flux buffers
        # Negative delay means that channel pulse happens after channel2 pulse
        if delay < 0:
            self.buffer_length_start = \
                       {channel: bls - delay + self.flux_buffer[channel],
                        channel2: bls + self.flux_buffer[channel2]}
            self.buffer_length_end = \
                        {channel: ble + self.flux_buffer[channel],
                         channel2: ble - delay + self.flux_buffer[channel2]}
        else:
            self.buffer_length_start = \
                       {channel: bls + self.flux_buffer[channel],
                        channel2: bls + delay + self.flux_buffer[channel2]}
            self.buffer_length_end = \
                        {channel: ble + delay + self.flux_buffer[channel],
                         channel2: ble + self.flux_buffer[channel2]}

        self.length = self.length1[channel] + self.length2[channel] + \
                      self.buffer_length_start[channel] + \
                      self.buffer_length_end[channel] + \
                      2*self.flux_buffer[channel]

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedNZFLIPPulse',
            'channel': None,
            'channel2': None,
            'amplitude': 0,
            'amplitude2': 0,
            'alpha': 1,
            'pulse_length': 0,
            'buffer_length_start': 30e-9,
            'buffer_length_end': 30e-9,
            'flux_buffer_length': 0,
            'flux_buffer_length2': 0,
            'channel_relative_delay': 0,
            'gaussian_filter_sigma': 1e-9,
            'sin_amp': 0,
            'sin_phase': 0,
            'kick_amp': 0,
            'kick_length': 10e-9,
        }
        return params

    def chan_wf(self, chan, tvals):

        amp1 = self.amps[chan]
        amp2 = -amp1*self.alphas[chan]
        buffer_start = self.buffer_length_start[chan]
        flux_buffer = self.flux_buffer[chan]
        l1 = self.length1[chan]
        l2 = self.length2[chan]

        if self.gaussian_filter_sigma == 0:
            # creates first square
            wave1 = np.ones_like(tvals)*amp1
            wave1 *= (tvals >= tvals[0] + buffer_start)
            wave1 *= (tvals < tvals[0] + buffer_start + l1)

            # creates second NZ square
            wave2 = np.ones_like(tvals)*amp2
            wave2 *= (tvals >= tvals[0] + buffer_start + l1 + 2*flux_buffer)
            wave2 *= (tvals < tvals[0] + buffer_start + l1 + l2 \
                      + 2*flux_buffer)

            wave = wave1 + wave2
        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + l1
            tstart2 = tvals[0] + buffer_start + l1 + 2*flux_buffer
            tend2 = tvals[0] + buffer_start + l1 + l2 + 2*flux_buffer
            scaling = 1/np.sqrt(2)/self.gaussian_filter_sigma
            wave = 0.5*(amp1*sp.special.erf((tvals - tstart)*scaling) -
                        amp1*sp.special.erf((tvals - tend)*scaling) +
                        amp2*sp.special.erf((tvals - tstart2)*scaling) -
                        amp2*sp.special.erf((tvals - tend2)*scaling))
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        amp = self.amps[channel]
        buffer_start = self.buffer_length_start[channel]
        buffer_end = self.buffer_length_end[channel]
        pulse_length = self.pulse_length

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma, self.alphas[channel]]
        return hashlist


class BufferedFLIPPulse(pulse.Pulse):
    def __init__(self, channel, channel2, element_name, aux_channels_dict=None,
                 name='Buffered FLIP Pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.channel = channel
        self.channel2 = channel2
        self.channels = [self.channel, self.channel2]

        self._update_amplitudes()

        delay = self.channel_relative_delay  # delay of pulse on channel2 wrt pulse on channel
        bls = self.buffer_length_start  # initial value for buffer length start passed with kw
        ble = self.buffer_length_end  # initial value for buffer length end passed with kw

        self.length1 = {channel: self.pulse_length + 2*self.flux_buffer_length,
                        channel2: self.pulse_length+2*self.flux_buffer_length2}

        # Compute new buffer lengths taking into account channel skewness and additional flux buffers
        # Negative delay means that channel pulse happens after channel2 pulse
        if delay < 0:
            self.buffer_length_start = \
                {channel: bls - delay + self.flux_buffer_length2,
                 channel2: bls + self.flux_buffer_length}
            self.buffer_length_end = \
                {channel: ble + self.flux_buffer_length2,
                 channel2: ble - delay + self.flux_buffer_length}
        else:
            self.buffer_length_start = \
                {channel: bls + self.flux_buffer_length2,
                 channel2: bls + delay + self.flux_buffer_length}
            self.buffer_length_end = \
                {channel: ble + delay + self.flux_buffer_length2,
                 channel2: ble + self.flux_buffer_length}

        self.length = self.length1[channel] + self.buffer_length_start[channel] + \
                      self.buffer_length_end[channel] + 2*self.flux_buffer_length2

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedFLIPPulse',
            'channel': None,
            'channel2': None,
            'amplitude': 0,
            'amplitude2': 0,
            'mirror_pattern': None,
            'mirror_correction': None,
            'pulse_length': 0,
            'buffer_length_start': 30e-9,
            'buffer_length_end': 30e-9,
            'flux_buffer_length': 0,
            'flux_buffer_length2': 0,
            'channel_relative_delay': 0,
            'gaussian_filter_sigma': 1e-9,
            'sin_amp': 0,
            'sin_phase': 0,
            'kick_amp': 0,
            'kick_length': 10e-9,
        }
        return params

    def _update_amplitudes(self):
        self.amps = {self.channel: self.amplitude,
                     self.channel2: self.amplitude2}

    def chan_wf(self, chan, tvals):
        self._update_amplitudes()
        amp = self.amps[chan]
        buffer_start = self.buffer_length_start[chan]
        l1 = self.length1[chan]

        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * amp
            wave *= (tvals >= tvals[0] + buffer_start)
            wave *= (tvals < tvals[0] + buffer_start + l1)

        else:
            tstart = tvals[0] + buffer_start
            tend = tvals[0] + buffer_start + l1
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * amp

            wave_sin = np.sin(2*np.pi*(tvals - tstart)/l1 +
                              self.sin_phase)*self.sin_amp
            wave_sin -= np.sin(self.sin_phase)*self.sin_amp
            wave_sin *= (tvals >= tvals[0] + buffer_start)
            wave_sin *= (tvals < tvals[0] + buffer_start + l1)
            wave += wave_sin

            # kick_start = (tstart + tend) / 2 - self.kick_length / 2
            # kick_end = (tstart + tend) / 2 + self.kick_length / 2
            kick_start = tstart
            kick_end = tstart + self.kick_length

            wave_kick = 0.5 * (sp.special.erf(
                (tvals - kick_start) * scaling) - sp.special.erf(
                (tvals - kick_end) * scaling)) * self.kick_amp *np.sign(amp)
            wave += wave_kick
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        self._update_amplitudes()
        amp = self.amps[channel]
        buffer_start = self.buffer_length_start[channel]
        buffer_end = self.buffer_length_end[channel]
        pulse_length = self.pulse_length

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma]

        return hashlist


class BufferedCryoscopePulse(pulse.Pulse):
    def __init__(self,
                 channel,
                 element_name,
                 aux_channels_dict=None,
                 name='buffered Cryoscope pulse',
                 **kw):
        super().__init__(name, element_name, **kw)

        # Set channels
        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedCryoscopePulse',
            'channel': None,
            'aux_channels_dict': None,
            'amplitude': 0,
            'frequency': 0,
            'phase': 0,
            'square_wave': False,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'extra_buffer_aux_pulse': 5e-9,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if chan != self.channel:
            amp = self.aux_channels_dict[chan]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        
        l = pulse_length
        t0 = self.algorithm_time() + buffer_start
        wave = (np.ones_like(tvals) 
                + 0.3*np.sin(2*np.pi*(tvals-t0)/l)
                + 0.3*np.sin(4*np.pi*(tvals-t0)/l)
                + 0.2*np.sin(6*np.pi*(tvals-t0)/l)
                + 0.2*np.sin(8*np.pi*(tvals-t0)/l)
            ) * amp
        wave *= (tvals >= self.algorithm_time() + buffer_start)
        wave *= (tvals < self.algorithm_time() + buffer_start + pulse_length)
        
        t_rel = tvals - tvals[0]
        carrier = np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
        if self.square_wave:
            carrier = np.sign(carrier)
        wave *= carrier
        return wave

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        amp = self.amplitude
        buffer_start = self.buffer_length_start
        buffer_end = self.buffer_length_end
        pulse_length = self.pulse_length
        if channel != self.channel:
            amp = self.aux_channels_dict[channel]
            buffer_start -= self.extra_buffer_aux_pulse
            buffer_end -= self.extra_buffer_aux_pulse
            pulse_length += 2 * self.extra_buffer_aux_pulse

        hashlist += [amp, pulse_length, buffer_start, buffer_end]
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.frequency, self.phase % 360, self.square_wave]
        return hashlist


class NZMartinisGellarPulse(pulse.Pulse):
    def __init__(self, channel, element_name, wave_generation_func,
                 aux_channels_dict=None,
                 name='NZMartinisGellarPulse', **kw):
        super().__init__(name, element_name, **kw)

        self.channel = channel
        self.aux_channels_dict = aux_channels_dict
        self.channels = [self.channel]
        if self.aux_channels_dict is not None:
            self.channels += list(self.aux_channels_dict)

        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        self.wave_generation_func = wave_generation_func

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'NZMartinisGellarPulse',
            'channel': None,
            'aux_channels_dict': None,
            'theta_f': np.pi / 2,
            'alpha': 1,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'extra_buffer_aux_pulse': 0e-9,
            'wave_generation_func': None,
            'qbc_freq': 0,
            'qbt_freq': 0,
            'anharmonicity': 0,
            'J': 0,
            'loop_asym': 0,
            'dv_dphi': 0,
            'lambda_2': 0,
        }
        return params

    def chan_wf(self, chan, tvals):

        dv_dphi = self.dv_dphi
        if chan != self.channel:
            dv_dphi *= self.aux_channels_dict[chan]

        params_dict = {
            'pulse_length': self.pulse_length,
            'theta_f': self.theta_f,
            'qbc_freq': self.qbc_freq,
            'qbt_freq': self.qbt_freq,
            'anharmonicity': self.anharmonicity,
            'J': self.J,
            'dv_dphi': dv_dphi,
            'loop_asym': self.loop_asym,
            'lambda_2': self.lambda_2,
            'alpha': self.alpha,
            'buffer_length_start': self.buffer_length_start
        }
        return self.wave_generation_func(tvals, params_dict)

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [self.pulse_length, self.theta_f, self.qbc_freq]
        hashlist += [self.qbt_freq, self.anharmonicity, self.J, self.dv_dphi]
        hashlist += [self.loop_asym, self.lambda_2, self.alpha]
        hashlist += [self.buffer_length_start, hash(self.wave_generation_func)]
        return hashlist


class GaussFilteredCosIQPulse(pulse.Pulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse',
                 **kw):
        super().__init__(name, element_name, **kw)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel, self.Q_channel]

        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'GaussFilteredCosIQPulse',
            'I_channel': None,
            'Q_channel': None,
            'amplitude': 0,
            'pulse_length': 0,
            'mod_frequency': 0,
            'phase': 0,
            'buffer_length_start': 10e-9,
            'buffer_length_end': 10e-9,
            'alpha': 1,
            'phi_skew': 0,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals, **kw):
        if self.gaussian_filter_sigma == 0:
            wave = np.ones_like(tvals) * self.amplitude
            wave *= (tvals >= self.algorithm_time() + self.buffer_length_start)
            wave *= (tvals <
                     self.algorithm_time() + self.buffer_length_start +
                     self.pulse_length)
        else:
            tstart = self.algorithm_time() + self.buffer_length_start
            tend = tstart + self.pulse_length
            scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
            wave = 0.5 * (sp.special.erf(
                (tvals - tstart) * scaling) - sp.special.erf(
                (tvals - tend) * scaling)) * self.amplitude
        I_mod, Q_mod = apply_modulation(
            wave,
            np.zeros_like(wave),
            tvals,
            mod_frequency=self.mod_frequency,
            phase=self.phase,
            phi_skew=self.phi_skew,
            alpha=self.alpha,
            tval_phaseref=0 if self.phase_lock else self.algorithm_time())
        if chan == self.I_channel:
            return I_mod
        if chan == self.Q_channel:
            return Q_mod

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [channel == self.I_channel, self.amplitude]
        hashlist += [self.mod_frequency, self.gaussian_filter_sigma]
        hashlist += [self.buffer_length_start, self.buffer_length_end, self.pulse_length]
        phase = self.phase
        phase += 360 * self.phase_lock * self.mod_frequency \
                 * self.algorithm_time()
        hashlist += [self.alpha, self.phi_skew, phase]
        return hashlist


class GaussFilteredCosIQPulseWithFlux(GaussFilteredCosIQPulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 flux_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse with flux pulse',
                 **kw):
        super().__init__(I_channel,
                         Q_channel,
                         element_name,
                         name=name,
                         **kw)
        self.flux_channel = flux_channel
        self.channels.append(flux_channel)
        self.flux_pulse_length = self.pulse_length + self.flux_extend_start + self.flux_extend_end
        self.flux_buffer_length_start = self.buffer_length_start - self.flux_extend_start
        self.flux_buffer_length_end = self.length - self.flux_buffer_length_start - self.flux_pulse_length
        self.fp = BufferedSquarePulse(element_name=self.element_name,
                                      channel=self.flux_channel,
                                      amplitude=self.flux_amplitude,
                                      pulse_length=self.flux_pulse_length,
                                      buffer_length_start=self.flux_buffer_length_start,
                                      buffer_length_end=self.flux_buffer_length_end,
                                      gaussian_filter_sigma=self.flux_gaussian_filter_sigma)

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params_super = super().pulse_params()
        params = {
            **params_super,
            'pulse_type': 'GaussFilteredCosIQPulseWithFlux',
            'flux_channel': None,
            'disable_flux_crosstalk_cancellation': False,
            'flux_amplitude': 0,
            'flux_extend_start': 20e-9,
            'flux_extend_end': 150e-9,
            'flux_gaussian_filter_sigma': 0.5e-9
        }
        return params

    def chan_wf(self, chan, tvals, **kw):
        if chan == self.I_channel or chan == self.Q_channel:
            return super().chan_wf(chan, tvals, **kw)
        elif chan == self.flux_channel:
            self.fp.algorithm_time(self.algorithm_time())
            return self.fp.chan_wf(chan, tvals)
        else:
            return {}

    def hashables(self, tstart, channel):
        if channel == self.I_channel or channel == self.Q_channel:
            return super().hashables(tstart, channel)
        elif channel == self.flux_channel:
            self.fp.algorithm_time(self.algorithm_time())
            return self.fp.hashables(tstart, channel)
        else:
            return []  # empty list if neither of the conditions is satisfied


class GaussFilteredCosIQPulseMultiChromatic(pulse.Pulse):
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='gauss filtered cos IQ pulse multi chromatic',
                 **kw):
        super().__init__(name, element_name, **kw)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel, self.Q_channel]

        if np.ndim(self.mod_frequency) != 1:
            raise ValueError("MultiChromatic Pulse requires a list or 1D array "
                             f"of frequencies. Instead {self.mod_frequency} "
                             f"was given")

        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

        params = dict(amplitude=self.amplitude,
                      phase=self.phase,
                      phi_skew=self.phi_skew,
                      alpha=self.alpha)

        for pname, p in params.items():
            if np.ndim(p) == 0:
                setattr(self, pname, len(self.mod_frequency) * [p])
            elif len(p) != len(self.mod_frequency):
                raise ValueError(f"Received {len(p)} {pname}  but expected "
                                 f"{len(self.mod_frequency)} (number of frequencies)")

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'GaussFilteredCosIQPulseMultiChromatic',
            'I_channel': None,
            'Q_channel': None,
            'amplitude': 0,
            'pulse_length': 0,
            'mod_frequency': [0],
            'phase': 0,
            'buffer_length_start': 10e-9,
            'buffer_length_end': 10e-9,
            'alpha': 1,
            'phi_skew': 0,
            'gaussian_filter_sigma': 0,
        }
        return params

    def chan_wf(self, chan, tvals, **kw):
        I_mods, Q_mods = np.zeros_like(tvals), np.zeros_like(tvals)
        for a, ph, f, phi, alpha in zip(self.amplitude, self.phase,
                                        self.mod_frequency, self.phi_skew,
                                        self.alpha):
            if self.gaussian_filter_sigma == 0:
                wave = np.ones_like(tvals) * a
                wave *= (tvals >= self.algorithm_time() + self.buffer_length_start)
                wave *= (tvals <
                         self.algorithm_time() + self.buffer_length_start +
                         self.pulse_length)
            else:
                tstart = self.algorithm_time() + self.buffer_length_start
                tend = tstart + self.pulse_length
                scaling = 1 / np.sqrt(2) / self.gaussian_filter_sigma
                wave = 0.5 * (sp.special.erf(
                    (tvals - tstart) * scaling) - sp.special.erf(
                    (tvals - tend) * scaling)) * a
            I_mod, Q_mod = apply_modulation(
                wave,
                np.zeros_like(wave),
                tvals,
                mod_frequency=f,
                phase=ph,
                phi_skew=phi,
                alpha=alpha,
                tval_phaseref=0 if self.phase_lock else self.algorithm_time())
            I_mods += I_mod
            Q_mods += Q_mod
        if chan == self.I_channel:
            return I_mods
        if chan == self.Q_channel:
            return Q_mods

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [channel == self.I_channel]
        hashlist += list(self.amplitude)
        hashlist += self.mod_frequency
        hashlist += [self.gaussian_filter_sigma]
        hashlist += [self.buffer_length_start, self.buffer_length_end, self.pulse_length]
        phase = [p + 360 * (not self.phase_lock) * f * self.algorithm_time() \
                 for p, f in zip(self.phase, self.mod_frequency)]
        hashlist += self.alpha
        hashlist += self.phi_skew
        hashlist += phase
        return hashlist


class VirtualPulse(pulse.Pulse):
    def __init__(self, element_name, name='virtual pulse', **kw):
        super().__init__(name, element_name, **kw)
        self.length = self.pulse_length
        self.channels = []

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'VirtualPulse',
            'pulse_length': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        return {}

    def hashables(self, tstart, channel):
        return []


class SquarePulse(pulse.Pulse):
    def __init__(self, element_name, channel=None, channels=None,
                 name='square pulse', **kw):
        super().__init__(name, element_name, **kw)
        if channel is None and channels is None:
            raise ValueError('Must specify either channel or channels')
        elif channels is None:
            self.channel = channel  # this is just for convenience, internally
            # this is the part the sequencer element wants to communicate with
            self.channels.append(channel)
        else:
            self.channels = channels

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'SquarePulse',
            'channel': None,
            'channels': [],
            'amplitude': 0,
            'length': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        return np.ones(len(tvals)) * self.amplitude

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [self.amplitude, self.length]
        return hashlist


class CosPulse(pulse.Pulse):
    def __init__(self, channel, element_name, name='cos pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.channel = channel  # this is just for convenience, internally
        self.channels.append(channel)

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'CosPulse',
            'channel': None,
            'amplitude': 0,
            'length': 0,
            'frequency': 1e6,
            'phase': 0,
        }
        return params

    def chan_wf(self, chan, tvals):
        return self.amplitude * np.cos(2 * np.pi *
                                       (self.frequency * tvals +
                                        self.phase / 360.))

    def hashables(self, tstart, channel):
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [self.amplitude, self.length, self.frequency]
        hashlist += [(self.phase + self.frequency * tstart * 360) % 360.]
        return hashlist


def apply_modulation(ienv, qenv, tvals, mod_frequency,
                     phase=0., phi_skew=0., alpha=1., tval_phaseref=0.):
    """
    Applies single sideband modulation, requires tvals to make sure the
    phases are correct.

    The modulation and predistortion is calculated as
    [I_mod] = [cos(phi_skew)  sin(phi_skew)] [ cos(wt)  sin(wt)] [I_env]
    [Q_mod]   [0              1/alpha      ] [-sin(wt)  cos(wt)] [Q_env],
    where wt = 360 * mod_frequency * (tvals - tval_phaseref) + phase

    The output is normalized such that the determinatnt of the transformation
    matrix is +-1.

    Args:
        ienv (np.ndarray): In-phase envelope waveform.
        qenv (np.ndarray): Quadrature envelope waveform.
        tvals (np.ndarray): Sample start times in seconds.
        mod_frequency (float): Modulation frequency in Hz.
        phase (float): Phase of modulation in degrees. Defaults to 0.
        phi_skew (float): Phase offset between I_channel and Q_channel, in
            addition to the nominal 90 degrees. Defaults to 0.
        alpha (float): Ratio of the I_channel and Q_channel output.
            Defaults to 1.
        tval_phaseref: The reference time in seconds for calculating phase.
            Defaults to 0.

    Returns:
        np.ndarray, np.ndarray: The predistorted and modulated outputs.
    """
    phi = 360 * mod_frequency * (tvals - tval_phaseref) + phase
    phii = phi + phi_skew
    phiq = phi + 90

    # k = 1 / np.cos(np.pi * phi_skew / 180) #  old normalization
    k = np.sqrt(np.abs(alpha / np.cos(np.deg2rad(phi_skew))))

    imod = k * (ienv * np.cos(np.deg2rad(phii)) +
                qenv * np.sin(np.deg2rad(phii)))
    qmod = k * (ienv * np.cos(np.deg2rad(phiq)) +
                qenv * np.sin(np.deg2rad(phiq))) / alpha

    return imod, qmod
