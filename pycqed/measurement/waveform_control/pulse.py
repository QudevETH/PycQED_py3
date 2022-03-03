"""
The definition of the base pulse object that generates pulse waveforms.

The pulse objects represent an analytical form of the pulses, and can generate
the waveforms for the time-values that are passed in to its waveform generation
function.

The actual pulse implementations are defined in separate modules,
e.g. pulse_library.py.

The module variable `pulse_libraries` is a
"""

import numpy as np
import scipy as sp

pulse_libraries = set()
"""set of module: The set of pulse implementation libraries.

These will be searched when a pulse dictionary is converted to the pulse object.
The pulse class is stored as a string in a pulse dictionary.

Each pulse library module should add itself to this set, e.g.
>>> import sys
>>> from pyceqed.measurement.waveform_control import pulse
>>> pulse.pulse_libraries.add(sys.modules[__name__])
"""


class Pulse:
    """
    The pulse base class.

    Args:
        name (str): The name of the pulse, used for referencing to other pulses
            in a sequence. Typically generated automatically by the `Segment`
            class.
        element_name (str): Name of the element the pulse should be played in.
        codeword (int or 'no_codeword'): The codeword that the pulse belongs in.
            Defaults to 'no_codeword'.
        length (float, optional): The length of the pulse instance in seconds.
            Defaults to 0.
        channels (list of str, optional): A list of channel names that the pulse
            instance generates waveforms form. Defaults to empty list.

    Attrs:
        channel_mask: set[str]
            A set of channel names to be excluded from waveform generation.

    """

    def __init__(self, name, element_name, **kw):

        self.name = name
        self.element_name = element_name
        self.codeword = kw.pop('codeword', 'no_codeword')
        self.pulse_off = kw.pop('pulse_off', False)
        self.truncation_length = kw.pop('truncation_length', None)
        self.truncation_decay_length = kw.pop('truncation_decay_length', None)
        self.truncation_decay_const = kw.pop('truncation_decay_const', None)
        self.crosstalk_cancellation_key = kw.pop('crosstalk_cancellation_key', None)
        self.crosstalk_cancellation_channels = []
        self.crosstalk_cancellation_mtx = None
        self.crosstalk_cancellation_shift_mtx = None
        self.channel_mask = kw.pop('channel_mask', set())

        # Set default pulse_params and overwrite with params in keyword argument
        # list if applicable
        for k, v in self.pulse_params().items():
            setattr(self, k, kw.get(k, v))

        self._t0 = None

    def truncate_wave(self, tvals, wave):
        """
        Truncate a waveform.
        :param tvals: sample start times for the channels to generate
            the waveforms for
        :param wave: waveform sample amplitudes corresponding to tvals
        :return: truncated waveform if truncation_length attribute is not None,
            else unmodified waveform
        """
        trunc_len = getattr(self, 'truncation_length', None)
        if trunc_len is None:
            return wave

        # truncation_length should be (n+0.5) samples to avoid
        # rounding errors
        mask = tvals <= (tvals[0] + trunc_len)
        trunc_dec_len = getattr(self, 'truncation_decay_length', None)
        if trunc_dec_len is not None:
            trunc_dec_const = getattr(self, 'truncation_decay_const', None)
            # add slow decay after truncation
            decay_func = lambda sigma, t, amp, offset: \
                amp*np.exp(-(t-offset)/sigma)
            wave_end = decay_func(trunc_dec_const, tvals[np.logical_not(mask)],
                                  wave[mask][-1], tvals[mask][-1])
            wave = np.concatenate([wave[mask], wave_end])
        else:
            wave *= mask
        return wave

    def waveforms(self, tvals_dict):
        """Generate waveforms for any channels of the pulse.

        Calls `Pulse.chan_wf` internally.

        Args:
            tvals_dict (dict of np.ndarray): a dictionary of the sample
                start times for the channels to generate the waveforms for.

        Returns:
            dict of np.ndarray: a dictionary of the voltage-waveforms for the
            channels that are both in the tvals_dict and in the
            pulse channels list.
        """
        wfs_dict = {}
        for c in self.channels:
            if c in tvals_dict and c not in \
                    self.crosstalk_cancellation_channels:
                wfs_dict[c] = self.chan_wf(c, tvals_dict[c])
                if getattr(self, 'pulse_off', False):
                    wfs_dict[c] = np.zeros_like(wfs_dict[c])
                wfs_dict[c] = self.truncate_wave(tvals_dict[c], wfs_dict[c])
        for c in self.crosstalk_cancellation_channels:
            if c in tvals_dict:
                idx_c = self.crosstalk_cancellation_channels.index(c)
                wfs_dict[c] = np.zeros_like(tvals_dict[c])
                if not getattr(self, 'pulse_off', False):
                    for c2 in self.channels:
                        if c2 not in self.crosstalk_cancellation_channels:
                            continue
                        idx_c2 = self.crosstalk_cancellation_channels.index(c2)
                        factor = self.crosstalk_cancellation_mtx[idx_c, idx_c2]
                        shift = self.crosstalk_cancellation_shift_mtx[
                            idx_c, idx_c2] \
                            if self.crosstalk_cancellation_shift_mtx is not \
                            None else 0
                        wfs_dict[c] += factor * self.chan_wf(
                            c2, tvals_dict[c] - shift)
                    wfs_dict[c] = self.truncate_wave(tvals_dict[c], wfs_dict[c])
        return wfs_dict

    def masked_channels(self):
        masked_channels = set(self.channels) - self.channel_mask
        if len(masked_channels & set(self.crosstalk_cancellation_channels)) > 0:
            return masked_channels | set(self.crosstalk_cancellation_channels)
        else:
            return masked_channels

    def pulse_area(self, channel, tvals):
        """
        Calculates the area of a pulse on the given channel and time-interval.

        Args:
            channel (str): The channel name
            tvals (np.ndarray): the sample start-times

        Returns:
            float: The pulse area.
        """
        if getattr(self, 'pulse_off', False):
            return 0

        if channel in self.crosstalk_cancellation_channels:
            # if channel is a crosstalk cancellation channel, then the area
            # of all flux pulses applied on this channel are
            # retrieved and added together
            wfs = []  # list of waveforms, area computed in return statement
            idx_c = self.crosstalk_cancellation_channels.index(channel)
            if not getattr(self, 'pulse_off', False):
                for c2 in self.channels:
                    if c2 not in self.crosstalk_cancellation_channels:
                        continue
                    idx_c2 = self.crosstalk_cancellation_channels.index(c2)
                    factor = self.crosstalk_cancellation_mtx[idx_c, idx_c2]
                    wfs.append(factor * self.waveforms({c2: tvals})[c2])
        elif channel in self.channels:
            wfs = self.waveforms({channel: tvals})[channel]
        else:
            wfs = np.zeros_like(tvals)
        dt = tvals[1] - tvals[0]

        return np.sum(wfs) * dt

    def algorithm_time(self, val=None):
        """
        Getter and setter for the start time of the pulse.
        """
        if val is None:
            return self._t0
        else:
            self._t0 = val

    def element_time(self, element_start_time):
        """
        Returns the pulse time in the element frame.
        """
        return self.algorithm_time() - element_start_time

    def hashables(self, tstart, channel):
        """Abstract base method for a list of hash-elements for this pulse.

        The hash-elements must uniquely define the returned waveform as it is
        used to determine whether waveforms can be reused.

        Args:
            tstart (float): start time of the element
            channel (str): channel name

        Returns:
            list: A list of hash-elements
        """
        raise NotImplementedError('hashables() not implemented for {}'
                                  .format(str(type(self))[1:-1]))

    def common_hashables(self, tstart, channel):
        if channel not in self.channels:
            return []
        if self.pulse_off:
            return ['Offpulse', self.algorithm_time() - tstart, self.length]
        return [type(self), self.algorithm_time() - tstart,
                self.truncation_length, self.truncation_decay_length,
                self.truncation_decay_const]

    def chan_wf(self, channel, tvals):
        """Abstract base method for generating the pulse waveforms.

        Args:
            channel (str): channel name
            tvals (np.ndarray): the sample start times

        Returns:
            np.ndarray: the waveforms corresponding to `tvals` on
            `channel`
        """
        raise NotImplementedError('chan_wf() not implemented for {}'
                                  .format(str(type(self))[1:-1]))

    @classmethod
    def pulse_params(cls):
        """
        Returns a dictionary of pulse parameters and initial values.
        """
        raise NotImplementedError('pulse_params() not implemented for your pulse')


def get_pulse_class(pulse_type):
    """
    Search in all registered pulse libraries for a given pulse type and
    return the class that implements this pulse type.

    :param pulse_type: (str) the pulse type

    :return: (class) the class that implements the pulse type
    """
    pulse_func = None
    for module in pulse_libraries:
        try:
            pulse_func = getattr(module, pulse_type)
        except AttributeError:
            pass
    if pulse_func is None:
        raise KeyError(f'pulse_type {pulse_type} not recognized.')
    return pulse_func
