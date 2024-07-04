"""Library containing various pulse shapes.
"""

import logging
import sys

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

from pycqed.measurement.waveform_control import pulse

log = logging.getLogger(__name__)

pulse.pulse_libraries.add(sys.modules[__name__])


class SSB_DRAG_pulse(pulse.Pulse):
    """In-phase Gaussian pulse with derivative quadrature and SSB modulation.

    Attributes:
        I_channel: In-phase output channel name.
        Q_channel: Quadrature output channel name.
        phaselock: The phase reference time is the start of the algorithm if
        True and the middle of the pulse otherwise. Defaults to True.

    """

    SUPPORT_INTERNAL_MOD = True
    SUPPORT_HARMONIZING_AMPLITUDE = True

    def __init__(self, element_name, I_channel, Q_channel,
                 name='SSB Drag pulse', **kw):
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
        super().__init__(name, element_name, **kw)

        self.I_channel = I_channel
        self.Q_channel = Q_channel

        self.phaselock = kw.pop('phaselock', True)

    @classmethod
    def pulse_params(cls):
        """Returns a dictionary of pulse parameters and initial values. These
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


class SSB_DRAG_pulse_cos(SSB_DRAG_pulse):
    """SSB second-order DRAG pulse with quadrature scaling factor and
    frequency detuning.

    When cancellation_frequency_offset is not None, this class applies
    correction factors to the amplitude and env_mod_frequency in order to
    decouple the effects of the three parameters amplitude, env_mod_frequency
    and cancellation_frequency_offset. These correction factors work in the
    limit abs(env_mod_freq) << 1/tg << cancellation_frequency_offset and ensure
    that:

        - the maximum spectral power of the pulse is at the env_mod_frequency
        independent of the value for amplitude or
        cancellation_frequency_offset;

        - the spectral power of the pulse at 0 is not changed by changing
        env_mod_frequency or cancellation_frequency_offset.

    FIXME: A future version of this this class should be adapted to allow
    crosstalk cancellation

    Attributes:
        See parent class for docstring.
        Additional parameter recognised by this class:
            cancellation_frequency_offset (float; default=None):
                frequency offset of the cancellation dip in the pulse spectrum
                with respect to the center frequency. This parameter typically
                takes the value of the transmon anharmonicity (ex: -170e6).
                The quadrature correction is not applied if this parameter
                is None (no cancellation dip in the pulse spectrum).
            env_mod_frequency (float; default=0):
                modulation frequency of the pulse envelope, introducing a
                detuning from mod_frequency

    """

    @classmethod
    def pulse_params(cls):
        params = super().pulse_params()
        params.update({'cancellation_frequency_offset': None,
                       'env_mod_frequency': 0})
        return params

    def chan_wf(self, channel, tvals):
        tg = self.nr_sigma * self.sigma
        half = tg / 2
        tc = self.algorithm_time() + half

        env_mod_freq_corr = self.env_mod_frequency
        amplitude_corr = self.amplitude
        if self.cancellation_frequency_offset is not None:
            # Apply correction factors to decouple the effects of the
            # pulse parameters.
            env_mod_freq_corr += 3 / (self.cancellation_frequency_offset *
                                      tg ** 2 * (np.pi ** 2 - 6))
            amplitude_corr /= 1 - (np.pi ** 2 - 6) * \
                              (tg * env_mod_freq_corr) ** 2 / 6

        # in-phase component
        envi = np.cos(np.pi * (tvals - tc) / tg) ** 2
        # truncate
        envi *= (tvals - tc >= -half) * (tvals - tc < half)
        # apply envelope modulation
        envi = envi * amplitude_corr * \
               np.exp(-2j * np.pi * env_mod_freq_corr * (tvals - tc))

        if self.cancellation_frequency_offset is not None:
            # Apply DRAG correction
            # Calculate quadrature component
            q = -1 / (2 * np.pi * self.cancellation_frequency_offset * tg)
            envq = q * tg * 0.5 * (np.diff(envi, prepend=[0]) +
                                   np.diff(envi, append=[0])) / \
                   (tvals[1]-tvals[0])
        else:
            log.debug('DRAG correction was not applied because '
                      'the cancellation_frequency_offset is 0.')
            envq = np.zeros_like(envi)

        envc = envi + 1j * envq
        # envi is complex if env_mod_frequency != 0, so we re-calculate the
        # real (envi) and imaginary (envq) components from the full complex
        # waveform envc
        envi, envq = np.real(envc), np.imag(envc)

        if self.mod_frequency is not None:
            I_mod, Q_mod = apply_modulation(
                envi, envq, tvals, self.mod_frequency,
                phase=self.phase, phi_skew=self.phi_skew, alpha=self.alpha,
                tval_phaseref=0 if self.phaselock else tc)
        else:
            # Ignore the Q component and program the I component to both
            # channels. See HDAWG8Pulsar._hdawg_mod_setter
            I_mod, Q_mod = envi, envi

        if channel == self.I_channel:
            return I_mod
        elif channel == self.Q_channel:
            return Q_mod
        else:
            return np.zeros_like(tvals)

    def hashables(self, tstart, channel):
        hashlist = super().hashables(tstart, channel)
        hashlist += [self.cancellation_frequency_offset, self.env_mod_frequency]
        return hashlist


class SSB_DRAG_pulse_with_cancellation(SSB_DRAG_pulse):
    """SSB Drag pulse with copies with scaled amp. and offset phase on extra
    channels intended for interferometrically cancelling on-device crosstalk.

    FIXME: This class should be generalized to allow crosstalk cancellation
    for different drive pulse shapes (e.g., SSB_DRAG_pulse_cos).

    Attributes:
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
    """The base class for different Gaussian-filtered piecewise constant pulses.

    To avoid clipping of the Gaussian-filtered rising and falling edges, the
    pulse should start and end with zero-amplitude buffer segments.

    Attributes:
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
        gaussian_filter_sigma (float or list of float): The width of the
            gaussian filter sigma of the pulse. If this is a list, indicates
            a value for each channel in self.channels.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
    """
    @classmethod
    def pulse_params(cls):
        """Returns a dictionary of pulse parameters and initial values. These
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

        if isinstance(self.gaussian_filter_sigma, list):
            gaussian_filter_sigma = self.gaussian_filter_sigma[idx]
        else:
            gaussian_filter_sigma = self.gaussian_filter_sigma

        if gaussian_filter_sigma > 0:
            timescale = 1 / (np.sqrt(2) * gaussian_filter_sigma)
        else:
            timescale = 0

        for seg_len, seg_amp in zip(self.lengths[idx], self.amplitudes[idx]):
            t1 = t0 + seg_len
            if gaussian_filter_sigma > 0:
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
    """A zero-area pulse shape that allows to control the accumulated phase when
    transitioning from the first pulse half to the second pulse half, by having
    an additional, low-amplitude segment between the two main pulse halves.

    Pulse shape:
            1
        ---------
       |         | 2
       |          ---
       |             | 3
       |              ---
       |                 |
    ---                  |                  ---
                         |                 |
                          ---              |
                           3 |             |
                              ---          |
                               2 |         |
                                  ---------
                                      1
    1: Main pulse halves
        - amplitude: amplitude +/- amplitude_offset
        - duration: pulse_length/2 + offset correction to keep a zero area
    2: (Optional) secondary transition step
        - amplitude: trans2_amplitude
        - duration: trans2_length
    3: Mid-pulse step (typically used to set the cphase via its time integral)
        - amplitude: trans_amplitude
        - duration: trans_length
    This pulse is meant to be played on the flux channels of two
    qubits in parallel; attributes ending with '2' refer to the second channel.
    """
    def __init__(self, element_name, name='NZTC pulse', **kw):
        super().__init__(name, element_name, **kw)
        self.is_net_zero = True
        self._update_cphase()
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
            'aux_pulses_list': [],
            'extra_buffer_aux_pulse': 5e-9,
            'pulse_length': 0,
            'trans_amplitude': 0,
            'trans_amplitude2': 0,
            'trans_length': 0,
            'trans2_amplitude': 0,
            'trans2_amplitude2': 0,
            'trans2_length': 0,
            'buffer_length_start': 30e-9,
            'buffer_length_end': 30e-9,
            'channel_relative_delay': 0,
            'gaussian_filter_sigma': 1e-9,
            'cphase': None,
            'cphase_calib_dict': None,
            'cphase_ctrl_params': ['trans_amplitude2', 'basis_rotation'],
            'fixed_pulse_length': None,
        }
        return params

    @staticmethod
    def calc_cphase_params(cphase, cphase_calib_dict, cphase_ctrl_params,
                           target=0, interpolation_type='quadratic'):
        """Calculates pulse parameters to implement a given conditional phase

        During the calibration, cphase_calib_dict is measured:
            {main_control_param: [...], 'cphase': [...], 'other_param': [...]}
        The goal of this function is to interpolate between all parameter
        values, for a given value of cphase.

        Args:
            cphase (float): Value of the conditional phase
            cphase_calib_dict (dict): Calibration dictionary, containing
                measurement values of the cphase as a function of the control
                parameters of the pulse
            cphase_ctrl_params (list): List of pulse parameter names that
                should be interpolated to set the conditional phase
            target (float): Since there might be several sets of parameters
                yielding a given cphase (if the calibrated range is wide
                enough to cover strictly more than 360 degrees), we choose
                one main parameter, cphase_ctrl_params[0], and select the value
                of this parameter which is closest to 'target'

        Returns:
            Dict of control parameters names and values
        """
        # Currently cphase_calib_dict = {'param': [values...] ...} including
        # 'cphase', all with the same number of points.
        # Could be extended to instead hold tuples of (cphase_vals,
        # param_vals) for each param, to allow different granularities
        param_vals = {}

        cp_list = cphase_calib_dict['cphase']
        # The calib dict may be calibrated over a range > 360 degrees.
        # Here, get all possible values of cphase, contained in the range of
        # the calib dict, which match the requested cphase modulo 360 degrees.
        # e.g. if cp_list = [273.2, ..., 1215.1] and cphase = 10.0, then
        # possible_cp = [370.,  730., 1090.] are all the ways one can
        # implement 10 degrees within the range of the calibration dict.
        # In details: min(cp_list)+(cphase-min(cp_list))%360 is the minimum
        # value of cp_list which is equal to cphase modulo 360. We then
        # recover all values equal to cphase modulo 360 by doing an arange.
        # Note that the arange does not include the upper bound, but this
        # would mean missing one possible cphase only if
        # max(cp_list) = cphase (mod. 360).
        possible_cp = np.arange(min(cp_list)+(cphase-min(cp_list))%360,
                                 max(cp_list), 360)
        # Get values of the main control parameter which yield these cphases
        f = interp1d(cp_list, cphase_calib_dict[cphase_ctrl_params[0]],
                     kind=interpolation_type)
        possible_param_vals = f(possible_cp)
        # Choose the value of the main control param closest to target
        id_closest = np.abs(possible_param_vals-target).argmin()
        # This is the cphase (not modulo 360) from the calibration dict which:
        # - equals the requested cphase modulo 360 degrees
        # - requires a value of the main control parameter closest to target
        cphase = possible_cp[id_closest]

        # Now that we have chosen one point of the calib dict, interpolate all
        # control params needed to reach this value of cphase
        for param_name in cphase_ctrl_params:
            cal_data = cphase_calib_dict[param_name]
            if isinstance(cal_data, dict):  # for 'basis_rotation'
                param_vals_dict = {}
                for qbn, qbn_data in cal_data.items():
                    f = interp1d(cp_list, qbn_data, kind=interpolation_type)
                    param_vals_dict.update({qbn: float(f(cphase))})
                param_vals[param_name] = param_vals_dict
            else:
                f = interp1d(cp_list, cal_data, kind=interpolation_type)
                param_vals[param_name] = float(f(cphase))
        return param_vals

    def _update_cphase(self, cphase=None):
        """Update cphase parameter and update all pulse parameters accordingly.

        Args:
            cphase (float, optional): Will be set as pulse parameter. If None
                the currently set cphase parameter is used to compute all pulse
                parameters. Defaults to None.
        """
        if cphase is not None:
            self.cphase = cphase
        if not (self.cphase is None
                or hasattr(self.cphase, '_is_parametric_value')):
            param_dict = \
                self.calc_cphase_params(
                    cphase=self.cphase,
                    cphase_calib_dict=self.cphase_calib_dict,
                    cphase_ctrl_params=self.cphase_ctrl_params,
                    # target the main control param to be close to the
                    # currently calibrated value for the CZ180 gate
                    target=getattr(self, self.cphase_ctrl_params[0])
                )
            for param_name, param_value in param_dict.items():
                setattr(self, param_name, param_value)

    def _update_lengths_amps_channels(self):
        self.channels = [c for c in [self.channel, self.channel2]
                         if c is not None]
        for pulse_pars in self.aux_pulses_list:
            for k in pulse_pars:
                if 'channel' in k:
                    self.channels.append(pulse_pars[k])
        self.lengths = []
        self.amplitudes = []

        # add amplitudes and lengths for gate pulses
        for ma, ta, ao, d, c, ia in [
            (self.amplitude, self.trans_amplitude, self.amplitude_offset,
             -self.channel_relative_delay/2, self.channel,
             self.trans2_amplitude),
            (self.amplitude2, self.trans_amplitude2, self.amplitude_offset2,
             self.channel_relative_delay/2, self.channel2,
             self.trans2_amplitude2),
        ]:
            if c is None:
                continue
            ml = self.pulse_length
            tl = self.trans_length
            il = self.trans2_length
            bs = self.buffer_length_start
            be = self.buffer_length_end
            ca0 = ma + ao
            ca1 = -ma + ao
            cl0 = max(-(ml * ao) / ca0, 0) if ca0 else 0
            cl1 = max(-(ml * ao) / ca1, 0) if ca1 else 0

            self.amplitudes.append([0, ma + ao, ia, ta, -ta, -ia, -ma + ao, 0])
            self.lengths.append([bs + d - cl0, cl0 + ml / 2, il, tl / 2,
                                 tl / 2, il, ml / 2 + cl1, be - d - cl1])
            if self.fixed_pulse_length is not None:
                current_length = np.sum(self.lengths[-1])
                difference = self.fixed_pulse_length - current_length
                # lengths[-1] is the set of lengths created at this iteration
                # of the for loop
                self.lengths[-1][0] += difference / 2
                self.lengths[-1][-1] += difference / 2
        while len(self.lengths) < len(self.channels):
            self.lengths += [[]]
        while len(self.amplitudes) < len(self.channels):
            self.amplitudes += [[]]

    def chan_wf(self, channel, t):
        self._update_lengths_amps_channels()
        wf = super().chan_wf(channel, t)

        # add auxiliary channel pulses
        for aux_dict in self.aux_pulses_list:
            pulse_pars = {k: getattr(self, k) for k in self.pulse_params()}
            pulse_pars['element_name'] = 'aux'
            pulse_pars.update(aux_dict)
            pulse_pars.pop('aux_pulses_list')
            import pycqed.measurement.waveform_control.segment as seg_mod
            pulse_obj = seg_mod.UnresolvedPulse(pulse_pars).pulse_obj
            if channel not in pulse_obj.channels:
                continue
            pulse_obj.algorithm_time(self.algorithm_time() +
                                     pulse_pars.get('pulse_delay', 0))
            wf += pulse_obj.chan_wf(channel, t)

        return wf


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
        """Returns a dictionary of pulse parameters and initial values. These
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
            'mirror_pattern': None
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedCZPulse',
            'channel': None,
            'aux_channels_dict': None,
            'amplitude': 0,
            'frequency': 0,
            'phase': 0,
            'pulse_length': 0,
            'buffer_length_start': 0,
            'buffer_length_end': 0,
            'extra_buffer_aux_pulse': 5e-9,
            'gaussian_filter_sigma': 0,
            'mirror_pattern': None,
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
        wave *= np.cos(
            2 * np.pi * (self.frequency * t_rel + self.phase / 360.))
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
        hashlist += [self.frequency, self.phase % 360]
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params = {
            'pulse_type': 'BufferedFLIPPulse',
            'channel': None,
            'channel2': None,
            'amplitude': 0,
            'amplitude2': 0,
            'mirror_pattern': None,  # see Segment.resolve_mirror
            'mirror_correction': None,  # see Segment.resolve_mirror
            'pulse_length': 0,
            'buffer_length_start': 30e-9,
            'buffer_length_end': 30e-9,
            'flux_buffer_length': 0,
            'flux_buffer_length2': 0,
            'channel_relative_delay': 0,
            'gaussian_filter_sigma': 1e-9,
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        self.channels = [self.I_channel]
        if self.Q_channel is not None:
            self.channels += [self.Q_channel]

        self.phase_lock = kw.pop('phase_lock', False)
        self.length = self.pulse_length + self.buffer_length_start + \
                      self.buffer_length_end

    @classmethod
    def pulse_params(cls):
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
                                      gaussian_filter_sigma=self.flux_gaussian_filter_sigma,
                                      mirror_pattern=kw.get("flux_mirror_pattern",
                                                            None))

    @classmethod
    def pulse_params(cls):
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
        super().__init__ method.
        """
        params_super = super().pulse_params()
        params = {
            **params_super,
            'pulse_type': 'GaussFilteredCosIQPulseWithFlux',
            'flux_channel': None,
            'flux_amplitude': 0,
            'flux_extend_start': 20e-9,
            'flux_extend_end': 150e-9,
            'flux_gaussian_filter_sigma': 0.5e-9,
            # Note that the mirror pattern is included in the pulse parameters
            # to ensure consistency (other flux_* parameters are also stored as
            # attributes of the Pulse object). However, the value of
            # self.fp.mirror_pattern (which should be identical to
            # self.flux_mirror_pattern unless someone messes with it)
            # is the one used by the code to retrieve the pattern to apply.
            'flux_mirror_pattern': None,
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

    def get_mirror_pulse_obj_and_pattern(self):
        # For flux pulse assisted readout, we currently return the mirror pattern
        # of the flux pulse.
        # FIXME: note that this prevents the user to enable mirror pattern on
        #  the readout drive pulse at the moment when using this pulse type.
        #  A better long term solution consists in refactoring the Pulse
        #  abstraction layer in which a clearer separation
        #  is made between pulses and operations (which can contain several pulses).
        return self.fp.get_mirror_pulse_obj_and_pattern()


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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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
        """Returns a dictionary of pulse parameters and initial values. These parameters are set upon calling the
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


class f0g1Pulse(pulse.Pulse):
    """Definition of the symmetric shaped f0g1 Pulse.
    """
    def __init__(self,
                 I_channel,
                 Q_channel,
                 element_name,
                 name='f0g1 pulse', **kw):
        super().__init__(name, element_name, **kw)

        self.I_channel = I_channel
        self.Q_channel = Q_channel
        self.channels = [self.I_channel]
        if self.Q_channel is not None:
            self.channels += [self.Q_channel]

        self.phase_lock = kw.pop('phase_lock', False)

        # with the function relevantTimeValues, calculate the 4 important times
        # (tStart, tStop, tRise, tFall)
        self.tStart, self.tStop, self.tRise, self.tFall = \
            self.relevantTimeValues(
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            photonTrunc=self.photonTrunc,
            pulseTrunc=self.pulseTrunc,
            junctionTrunc=self.junctionTrunc,
            junctionSigma=self.junctionSigma,
            timeReverse=self.timeReverse
        )

        # calculate the length of the pulse
        self.length = self.tStop - self.tStart


    def relevantTimeValues(self, gamma1, gamma2, photonTrunc, pulseTrunc,
                           junctionTrunc, junctionSigma, timeReverse=False):
        """Calculates the important time values for the pulse.

        Args:
            gamma1 (float): exponential rate of the rising edge of the
                emitted photon
            gamma2 (float): exponential rate of the falling edge of the
                emitted photon
            photonTrunc (float), pulseTrunc (float):  Pulse truncation
                properties. For pulseTrunc=1, the pulse is truncated at
                -photonTrunc*2/gamma1 and photonTrunc*2/gamma2.
                For pulseTrunc<1, the pulse is truncated such that it has the
                same start time, but a pulse length of pulseTrunc*pulseLength
            junctionTrunc (float), junctionSigma (float): information about the
                junction bridging the AWG amplitude from the truncated pulse
                value at the end and zero. These variables denote the junction
                truncation, width and type, respectively
            timeReverse (bool): Says if the pulse should be time reversed to
                absorb an incoming photon. One should set a=1 for this to work

        Returns:
            tStart (float): when the pulse starts
            tStop (float): when the pulse ends
            tRise (float): when the pulse starts after the ramp up
            tFall (float): when the pulse ends before the ramp down
        """
        if not timeReverse:
            tRise = -photonTrunc * 2 / gamma1
            tFall = tRise + photonTrunc * pulseTrunc * \
                    (2 / gamma2 + 2 / gamma1)
        else:
            tFall = photonTrunc * 2 / gamma2
            tRise = tFall - photonTrunc * pulseTrunc * \
                    (2 / gamma2 + 2 / gamma1)
        tStart = tRise - junctionTrunc * junctionSigma
        tStop = tFall + junctionTrunc * junctionSigma

        return tStart, tStop, tRise, tFall

    def GTilde(self, kappa, gamma1, gamma2, delta, a, t):
        r"""Generates the drive rate vs time needed to emit a photon with
         the required shape:

        .. math::

            a \cdot \sqrt{\frac{\gamma_1 + \gamma_2}{2}
                \text{ sinc}\left(\pi * \frac{\gamma_1 - \gamma_2}
                {\gamma_1 + \gamma_2}\right)} \cdot \frac{1}{e^{-\gamma_1 t/2}
            + e^{\gamma_2 t/2}}

        Detuned by delta from the resonance frequency of an
            emitter qubit/resonator which leaks at rate kappa.

        The drive rate is expressed as a "population" Rabi rate, i.e. the
        swap matrix element is
        :math:`\tilde{g} * (|0\rangle\langle 1| + \text{h.c.}) / 2`.

        The returned shape of this function is the
            first formula of page 43 - Dr. Paul Magnard PhD Thesis, 2021.
        FIXME: find a better equation in the prior theses (i.e., Philipp's?)

        Parameters
        ----------
        kappa : float
            The fixed effective leakage rate of the
            transfer/emitter resonator-purcell system.
        gamma1 : float
            Exponential rate of the rising edge of the emitted photon.
        gamma2 : float
            Exponential rate of the falling edge of the emitted photon.
        delta : float
            Detuning of the emitted photon with respect to the
            emitter resonator/qubit/cavity resonant frequency.
        a : float
            Fraction of the photon which is emitted. :math:`a^2 = 1`
            corresponds to a full photon emission. :math:`a^2 = 0.5` leads to
            a perfectly entangled emitter-photon system.
        t : array_like or float
            The values of the time for which we want
            the amplitude of :math:`\tilde{g}`.

        Returns:
        -------
        float
            Value of the first formula of page 43 - Dr. Paul Magnard
            PhD Thesis, 2021.
        """
        # we define two variables that will help us in the definitions
        gamma = gamma1 + gamma2

        X = (gamma * np.sinc((gamma1 - gamma2) / gamma) / 2)

        # we define f, the shape of the photon that we are looking for
        f = a * np.sqrt(X) / (np.exp(-gamma1 * t / 2)
                              + np.exp(gamma2 * t / 2))
        assert (f > 0).all(), 'Photon shape is positive'

        # calculate the F2 definition, see page 43 of Paul's 2021 thesis
        # We use hypergeom function to integrate f^2 until time t
        # (for a list of t) to avoid numerical aberrations at high t
        # calculating 0/0. The normalization constant in front
        # of f is chosen such that lim_(t->inf) F(2) = a

        # now let us define F2 for t<0 (_a) and for t>0 (_b) - we do it
        # this way to be able to use numpy lists as an input of t
        F2_a = a * X * np.exp(gamma1 * t) / gamma1 * \
            sp.special.hyp2f1(2, 2 * gamma1 / gamma,
                (3 * gamma1 + gamma2) / gamma, -np.exp(gamma * t / 2))

        F2_b = a * (1 - X * np.exp(-gamma2 * t) / gamma2 *
            sp.special.hyp2f1(2, 2 * gamma2 / gamma,
                (3 * gamma2 + gamma1) / gamma, -np.exp(-gamma * t / 2)))

        # we create the final F2 (real, between 0 and 1)
        F2 = np.int32(t < 0) * F2_a + np.int32(t >= 0) * F2_b

        # derivative of f (real)
        df = f * (np.exp(-gamma1 * t / 2) * gamma1 -
            np.exp(gamma2 * t / 2) * gamma2) / \
            (2 * (np.exp(-gamma1 * t / 2) + np.exp(gamma2 * t / 2)))

        # return (first formula pag. 43 Dr. Paul Magnard PhD Thesis, 2021)
        gTilde_vs_t = 2 * np.sqrt(((df + kappa * f / 2) ** 2 +
            (delta * f) ** 2) / (kappa * (1 - F2) - f ** 2))
        # If the denominator is negative, we want to emit
        # too much photon per time
        assert ((kappa * (1 - F2) - f ** 2) > 0).all(), \
            'Requested coupling should be real'

        # second derivative of f
        ddf = f * (2 * (np.exp(-gamma1 * t / 2) * gamma1 -
                         np.exp(gamma2 * t / 2) * gamma2) ** 2 \
                    - (np.exp(-gamma1 * t / 2) + np.exp(gamma2 * t / 2)) * \
                    (np.exp(-gamma1 * t / 2) * gamma1 ** 2 +
                     np.exp(gamma2 * t / 2) * gamma2 ** 2)) \
              / (4 * (np.exp(-gamma1 * t / 2) + np.exp(gamma2 * t / 2)) ** 3)

        # return (second formula pag. 43 Dr. Paul Magnard PhD Thesis, 2021)
        return_term_1 = f ** 2 / (kappa * (1 - F2) - f ** 2)
        return_term_2 = (ddf * f - df ** 2) / ((delta * f) ** 2 + \
                                               (df + kappa * f / 2) ** 2)

        FrequencyChirp = delta * (return_term_1 + return_term_2)

        # Both returns (as all steps above) should be real-valued
        return gTilde_vs_t, FrequencyChirp


    def joinJunctions(self, gTilde_vs_t, tRise, tFall, junctionTrunc,
                      junctionSigma, junctionType, t):
        """This function creates a 'ramp up' and a 'rump down' for the pulse
        outside of the interval given by 'tStart' and 'tFall'.

        The returned shape will have 3 parts:
            - part_a = ramp up
            - part_b = pulse function
            - part_c = ramp down

        The function is defined as follows:

            - part_a: tRise - junctionTrunc*junctionSigma <= t < tRise
            - part_b: tRise <= t < tFall
            - part_c: tFall <= t < tFall + junctionTrunc*junctionSigma
            - 0: otherwise

        ::

                         ~~~~~~
                      ~~~~    ~~~
                  ~~~~           \\
                /                 \\
               /                   \\
              /                     \\
             /                       \\
            /                         \\
          _/                           \\__
          |-- a --|------- b -----|-- c --|

        The ramps (up and down) can have different shapes: gaussian,
            tanh or ramp (linear).

        Args:
            gTilde_vs_t (array): An array of the shape of the pulse,
                spanning the whole time (a, b and c).
            tRise (float): When part_b starts (when the photon starts).
            tFall (float): When part_b ends (when the photon ends).
            junctionTrunc (float): Information about the junction bridging the
                AWG amplitude from the truncated pulse value at the end
                and zero. Denotes the junction truncation.
            junctionSigma (float): Information about the junction bridging the
                AWG amplitude from the truncated pulse value at the end
                and zero. Denotes the junction width.
            junctionType (str): Defines the shape of the ramps: 'gaussian',
                'tanh' or 'ramp' (linear).

        Returns:
            array: The final pulse shape.
        """
        # raise an error if type_junction is one of the three allowed
        junctionTypes_allowed = ['gaussian', 'tanh', 'ramp']
        if junctionType not in junctionTypes_allowed:
            raise ValueError("joinJunctions: '%s' is an invalid junction "
                             "type. Expected one of: %s" \
                             % (junctionType, junctionTypes_allowed))

        # create 'junction function', its shape depends on junctionType
        if junctionType == "gaussian":
            offset = np.exp(-junctionTrunc ** 2 / 2)
            amp = 1 / (1 - offset)
            junctionFunc = lambda t: amp * \
                                     (np.exp(-t ** 2 /
                                             (2 * junctionSigma ** 2)) -
                                      offset)
        elif junctionType == "tanh":
            offset = np.tanh(-junctionTrunc)
            amp = -1 / (2 * offset)
            junctionFunc = lambda t: amp * (
                        np.tanh((2 * t + junctionTrunc * junctionSigma) /
                                junctionSigma) - offset)
        else:  # ramp
            junctionFunc = lambda t: t / \
                                     (junctionTrunc * junctionSigma) + 1

        # we define the final shape for each part:
        #                    part_a, tRise - junctionTrunc*junctionSigma <= t < tRise
        #                    part_b,                               tRise <= t < tFall
        #                    part_c,                               tFall  <= t < tFall + junctionTrunc*junctionSigma
        #                    0   ,      otherwise

        arg_tRise = np.argmin(abs(t - tRise))
        arg_tFall = np.argmin(abs(t - tFall))
        return_part_a = gTilde_vs_t[arg_tRise] * junctionFunc(t - tRise)
        return_part_b = gTilde_vs_t
        return_part_c = gTilde_vs_t[arg_tFall] * junctionFunc(tFall - t)

        # we put together the final shape
        return_final = np.int32(
            ((tRise - junctionTrunc * junctionSigma) <= t) &
            (t < tRise)) * return_part_a + np.int32((tRise <= t) &
            (t < tFall)) * return_part_b + np.int32((tFall <= t) &
            (t < (tFall + junctionTrunc * junctionSigma))) * return_part_c

        # we return final shape
        return return_final


    def photonShapingPulse(self, t, kappa, gamma1, gamma2, delta, a,
                           acStark_coefs, rabiRate_coefs, timeValues,
                           junctionTrunc, junctionSigma, junctionType="ramp",
                           timeReverse=False, lowerFreqPhoton=False,
                           driveDetScale=0):
        r"""Get the complex amplitude and frequency of the f0g1 pulse.

        This pulse is tailored to emit a photon with shape:

        .. math::

            a \cdot \sqrt{\frac{\gamma_1 + \gamma_2}{2}
                \text{ sinc}\left(\pi * \frac{\gamma_1 - \gamma_2}
                {\gamma_1 + \gamma_2}\right)} \cdot \frac{1}{e^{-\gamma_1 t/2}
            + e^{\gamma_2 t/2}}

        Parameters
        ----------
        t : np.array
            Array of time values. It should start at 'tStart' and end at
            'tStop'; These two values are calculated with the
            'relevantTimeValues' function.
        kappa : float
            The fixed leakage rate of the transfer/emitter
            resonator/qubit/cavity.
        gamma1 : float
            Exponential rate of the rising edge of the emitted photon.
        gamma2 : float
            Exponential rate of the falling edge of the emitted photon.
        delta : float
            Detuning of the emitted photon with respect to the emitter
            resonator/qubit/cavity resonant frequency. ('Smart detuning').
        a : float
            Fraction of the photon which is emitted. :math:`a^2 = 1`
            corresponds to a full photon emission. :math:`a^2 = 0.5` leads to
            a perfectly entangled emitter-photon system.
        acStark_coefs : np.array
            Coefficients of the AcStark function (polynomial): given
            drive amplitude returns frequency shift.
        rabiRate_coefs : np.array
            Coefficients of the RabiRate function (polynomial): given
            drive amplitude returns the coupling strength gTilde.
        timeValues : tuple
            Tuple of 4 values 'tStart, tStop, tRise, tFall': these values
            are given by the 'relevantTimeValues' function.
            junctionTrunc (float), junctionSigma (float): information about the
                junction bridging the AWG amplitude from the truncated pulse
                value at the end and zero. These variables denote the junction
                truncation, width and type, respectively
        junctionType : str
            Defines the shape of the ramps for the truncation: 'gaussian',
            'tanh' or 'ramp' (linear).
        timeReverse : bool
            Says if the pulse should be time reversed to absorb an
            incoming photon. One should set a=1 for this to work.
        lowerFreqPhoton : bool
            Should be set to True if ω_gf > ω_p and to False otherwise.
            Has an impact on the sign of 'delta' in the drive functions.
        driveDetScale : int
            An additional offset for the final frequency of the pulse.

        Returns:
        -------
        amplitude_final : np.array
            Array of amplitudes of the final pulse (it's the envelope of
            the pulse) for each value of time a complex amplitude is returned.
        ifOffset + deltaDet + driveDetScale : float
            Value for the frequency of the pulse.
        """
        # if t, acStark_coefs and gTilde_to_amplitude_coefs are not numpy
        # arrays we create them as so
        t = np.array(t)
        acStark_coefs = np.array(acStark_coefs)
        rabiRate_coefs = np.array(rabiRate_coefs)

        # we get the coefficients of the gTilde to amplitude function:
        rabiRate_coefs = np.append(rabiRate_coefs, np.array(
            [0 for i in range(4 - rabiRate_coefs.size)]))  # if there is less
        # than 4 coefs, add them
        c0, c1, c2, c3 = rabiRate_coefs[:4]  # we put the first 4 coefs to some
        # variables
        gTilde_to_amplitude_coefs = np.array([0, 1 / c1, 0, -c3 / c1 ** 4])
        # we invert the odd polynomial (degree 3)


        # the relevant time values of the pulse
        tStart, tStop, tRise, tFall = timeValues

        # reverse the time is the variable is true
        if timeReverse: t = np.flip(t)


        # get gTilde shape and truncation
        gTilde_vs_t, frequencyChirp = self.GTilde(kappa, gamma1, gamma2, delta, a, t)

        gTildeTruncated_vs_t = self.joinJunctions(gTilde_vs_t, tRise, tFall,
                                                  junctionTrunc, junctionSigma,
                                                  junctionType, t)

        #  ---- get pulse phase

        # get the amplitude of the pulse for all times
        pulse_amplitude = np.polyval(np.flip(gTilde_to_amplitude_coefs),
                                     gTildeTruncated_vs_t)

        # get frequencies thanks to acStark_coefs from the amplitude
        frequencies = np.polyval(np.flip(acStark_coefs),
                                 pulse_amplitude)

        # Add extra frequency due to photon detuning from the emitter
        instantFreqList = frequencyChirp + frequencies

        ifOffset = (np.min(instantFreqList) + np.max(instantFreqList)) / 2
        instantFreqList -= ifOffset

        # Get the pulse phase for each time

        phases = np.zeros(len(t))
        phase_sum = 0
        for i in range(len(t) - 1):
            phases[i] = phase_sum
            phase_sum += (instantFreqList[i + 1] + instantFreqList[i]) * (t[i + 1] - t[i]) / 2
        phases[-1] = phase_sum

        # As frequencies are in Hz, we need a 2Pi factor.
        pulsePhase = 2 * np.pi * phases * (-1) ** timeReverse

        #  ----

        # get the complex amplitude of the pulse drive, which starts at
        # t=tStart and ends at t=tStop.

        # An important minus here is to ensure the consistency with PycQED
        amplitude_final = pulse_amplitude * np.exp(-1j * pulsePhase)

        # subtract delta to the drive frequency if w_p < w_gf, and add delta
        # otherwise
        deltaDet = delta * (-1) ** lowerFreqPhoton

        # return final pulse: 'array of amplitudes' and 'frequency'

        return np.array([amplitude_final,
                         (ifOffset + deltaDet + driveDetScale)], dtype=object)


    @classmethod
    def pulse_params(cls):
        """Returns a dictionary of pulse parameters and initial values. These
        parameters are set upon calling the super().__init__ method.
        """
        params = {
            'pulse_type': 'f0g1Pulse',
            'kappa': 1,
            'gamma1': 1,
            'gamma2': 1,
            'delta': 0,
            'a': 1,
            'photonTrunc': 1.8,
            'pulseTrunc': 0,
            'junctionTrunc': 2,
            'junctionSigma': 1.5e-9,
            'AcStark_IFCoefs' : np.array([0,0,0]),
            'RabiRate_Coefs' : np.array([0,0,0]),
            'timeReverse': False,
            'lowerFreqPhoton': False,
            'driveDetScale': 0,
            'junctionType': 'ramp',

            'I_channel': None,
            'Q_channel': None,
            'frequency': 0,
            'phase': 0,
            'alpha': 1,
            'phi_skew': 0,

            'delay': 0,
        }
        return params


    def chan_wf(self, chan, tvals):
        """Given the channel and an array of times (tvals) it returns the
        amplitudes of the pulse for this channel
        """
        # center the time, let it start and finish at the correct values to get
        # the correct shape
        t = tvals - self.algorithm_time() + self.tStart

        # get the shape and frequency with the 'photonShapingPulse' function
        wave_shape, self.frequency = self.photonShapingPulse(
            t=t,
            kappa=self.kappa,
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            delta=self.delta,
            a=self.a,
            timeValues=(self.tStart, self.tStop, self.tRise, self.tFall),
            junctionTrunc=self.junctionTrunc,
            junctionSigma=self.junctionSigma,
            acStark_coefs=self.AcStark_IFCoefs,
            rabiRate_coefs=self.RabiRate_Coefs,
            junctionType=self.junctionType,
            timeReverse=self.timeReverse,
            lowerFreqPhoton=self.lowerFreqPhoton,
            driveDetScale=self.driveDetScale)

        # we apply sideband modulation to the envelope shape with the frequency
        # computed
        I_mod, Q_mod = apply_modulation(
            np.real(wave_shape),
            np.imag(wave_shape),
            t,
            mod_frequency=self.frequency,
            phase=self.phase,
            phi_skew=self.phi_skew,
            alpha=self.alpha,
            tval_phaseref=0 if self.phase_lock else self.algorithm_time())

        # return I or Q depending on the channel
        if chan == self.I_channel:
            return I_mod
        if chan == self.Q_channel:
            return Q_mod

    def hashables(self, tstart, channel):
        """It returns a hash list to identify the pulse
        """
        hashlist = self.common_hashables(tstart, channel)
        if channel not in self.channels or self.pulse_off:
            return hashlist
        hashlist += [tuple(self.AcStark_IFCoefs.tolist()),
                     tuple(self.RabiRate_Coefs.tolist())]
        hashlist += [self.kappa, self.gamma1, self.gamma2, self.delta, self.a]
        hashlist += [self.timeReverse, self.lowerFreqPhoton,
                     self.driveDetScale, self.junctionType]
        hashlist += [self.photonTrunc, self.pulseTrunc,
                     self.junctionTrunc, self.junctionSigma]
        hashlist += [channel == self.I_channel]
        hashlist += [self.frequency, self.delay]
        phase = self.phase
        phase += 360 * self.phase_lock * self.frequency * self.algorithm_time()
        hashlist += [self.alpha, self.phi_skew, phase]

        return hashlist


def apply_modulation(ienv, qenv, tvals, mod_frequency,
                     phase=0., phi_skew=0., alpha=1., tval_phaseref=0.):
    """Applies single sideband modulation, requires tvals to make sure the
    phases are correct.

    If alpha >= 1.0: The modulation and predistortion is calculated as
    [I_mod] = [cos(phi_skew)  sin(phi_skew)] [ cos(wt)  sin(wt)] [I_env]
    [Q_mod]   [0              1/alpha      ] [-sin(wt)  cos(wt)] [Q_env],
    where wt = 360 * mod_frequency * (tvals - tval_phaseref) - phase

    If alpha < 1.0: I_mod and Q_mod will be multiplied with alpha on top of
    the expression above. This is to make sure that all elements of the
    predistortion matrix is not larger than 1, in order to be compatible with
    mixer_calib modulation mode of ZI HDAWG.

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

    # - sign: so that 'phase' is the phase of the drive on the Bloch sphere.
    # This ensures that calling this method with the phase attribute of a pulse
    # yields a drive right-handed rotated by 'phase' around the vertical axis.
    # Intuitively, this is because, in the laboratory frame, the quantum state
    # precesses clockwise, seen from the top (assuming |0> is the top of the
    # Bloch sphere). This means that the drive also rotates clockwise vs time
    # (one of the two exponentials in cos(\omega t - phase)). The - sign then
    # ensures that the drive rotates counterclockwise as a function of phase.
    phi = 360 * mod_frequency * (tvals - tval_phaseref) - phase
    phii = phi + phi_skew
    phiq = phi + 90

    r = alpha if alpha < 1.0 else 1.0
    imod = r * (ienv * np.cos(np.deg2rad(phii)) +
                qenv * np.sin(np.deg2rad(phii)))
    qmod = r * (ienv * np.cos(np.deg2rad(phiq)) +
                qenv * np.sin(np.deg2rad(phiq))) / alpha

    return imod, qmod
