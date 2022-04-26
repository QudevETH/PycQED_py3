import numpy as np
from copy import deepcopy
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter
import logging
log = logging.getLogger(__name__)


class AcquisitionDevice():
    """Base class for a standardized acquisition device driver interface.

    This class is not meant to be instantiated, but is only meant to be used
    as a parent class as described in the following:
    Child classes should inherit via multi-inheritance from the underlying
    qcodes driver as first parent and from this class as subsequent parent.
    The init of the child class has to explicitly call the init of this base
    class after calling the super init since the qcodes intrument (first
    parent) will not forward the super call. In the list of attributes,
    (*) indicates constants that are meant to be overwritten by child
    classes if needed.

    Attributes:
        n_acq_units (int): number of physical acquisition units (*)
        n_acq_int_channels (int): number of integration channels per
            acquisition unit (*)
        n_acq_inp_channels (int): number of input channels (quadratures)
            per acquisition unit (*)
        acq_length_granularity (int): indicates that the number of samples
            in an acquired signal must be a multiple of this number (*)
        acq_sampling_rate (float): sampling rate of the acquisition units in
            Hertz (*)
        acq_weights_n_samples (int): number of samples of auto-generated
            integration weights (*)
        acq_Q_sign (1 or -1): sign of auto-generated integration weights for
            the second weighted-integration channel in a pair (*)
        allowed_weights_types (list of str): allowed types of auto-generated
            integration weights (*)
        allowed_modes (dict): each key is a str supported as mode argument
            of acquisition_initialize, and the corresponding value is a list
            of str, which are supported as data_type argument of
            acquisition_initialize if that mode is used (*)
        lo_freqs (list of float/None): LO frequencies of the internal or
            external LOs of all acquisition units, see set_lo_freq
        timer: Timer object (see pycqed.utilities.timer.Timer). This is
            currently set by the detector function, in order to recover timer
            data from the acquisition device through the detector function.
    """

    n_acq_units = 1
    n_acq_int_channels = 1
    n_acq_inp_channels = 2  # I&Q by default, can be overridden by children
    acq_length_granularity = 1
    acq_sampling_rate = None
    acq_weights_n_samples = None
    acq_Q_sign = 1
    allowed_modes = []
    allowed_weights_types = ['optimal', 'optimal_qutrit', 'SSB',
                             'DSB', 'DSB2', 'square_rot']

    def __init__(self, *args, **kwargs):
        """Init of the common part of all acquisition devices.

        Accepts *args and **kwargs for compatibility in super calls,
        but currently ignores them.
        """
        if not hasattr(self, 'IDN'):
            raise AttributeError(
                f'{repr(self)} was not properly initialized as a qcodes '
                f'instrument. See the class docstring of AcquisitionDevice '
                f'for proper usage of this class.')
        self._acquisition_nodes = []
        self._acq_mode = None
        self._acq_data_type = None
        self._reset_n_acquired()
        self.lo_freqs = [None] * self.n_acq_units
        self._acq_units_used = []
        if 'timeout' not in self.parameters:
            # The underlying qcodes driver has not created a parameter
            # timeout. In that case, we add the parameter here.
            self.add_parameter(
                'timeout',
                unit='s',
                initial_value=30,
                parameter_class=ManualParameter,
                vals=validators.Ints())

    def set_lo_freq(self, acq_unit, lo_freq):
        """Set the local oscillator frequency used for an acquisition unit.

        The base AcquisitionDevice only stores the information in the property
        lo_freqs. Child classes, in particular for instruments with an
        internal LO, can implement further functionality by accessing this
        property or by overriding set_lo_freq.

        Remark: This method is called from QuDev_transmon.prepare.

        Args:
             acq_unit (int): the acquisition unit for which the LO frequency
                should be set.
            lo_freq (float): the LO frequency
        """
        self.lo_freqs[acq_unit] = lo_freq

    def acquisition_initialize(self, channels, n_results, averages, loop_cnt,
                               mode, acquisition_length, data_type=None,
                               **kwargs):
        """Initialize the acquisition device.

        Configures the acquisition device for an experiment and performs
        tasks that need to be done once per experiment (i.e., not repeatedly
        in sweeps). The base method stores and validates acquisition
        settings. Further functionality can be implemented in child classes.

        Args:
            channels (list of tuple of int): Channels on which the
                acquisition should be performed. A channel is identified by
                two ints in a tuple. The first int is the index of the
                acquisition unit, the second int is a logical index within
                the physical unit (the exact meaning of the second int
                depends on the acquisition mode, e.g.:
                    - the index of the input channel, i.e., the quadrature
                        (0=I, 1=Q), in 'avg' mode
                    - the weighted-integration channel in 'int_avg' mode
            n_results (int): number of acquisition elements
            averages (int): number of repetitions for averaging
            loop_cnt (int): total number of repetitions (averaging & shots)
            mode (str): acquisition modes. One of the keys of
                self.allowed_modes
            acquisition_length (float): length of the acquired signals in
                seconds
            data_type (str): data type. One of the str listed in
                self.allowed_modes for the chosen mode.
            **kwargs: currently ignored
        """

        self._acquisition_nodes = deepcopy(channels)
        self._acq_length = acquisition_length
        self._acq_channels = channels
        for ch in channels:
            if ch[0] not in range(self.n_acq_units):
                raise ValueError(f'{self.name}: Acquisition unit {ch[0]} '
                                 f'does not exist.')
            if mode == 'int_avg' and ch[1] not in range(
                    self.n_acq_int_channels):
                raise ValueError(f'{self.name}: Integration channel {ch[1]} '
                                 f'does not exist.')
            elif mode == 'avg' and ch[1] not in range(self.n_acq_inp_channels):
                raise ValueError(f'{self.name}: Input channel {ch[1]} does '
                                 f'not exist.')
        self._acq_n_results = n_results
        self._acq_averages = averages
        self._acq_loop_cnt = loop_cnt
        self._acq_mode = mode
        self._acq_data_type = data_type

        self._reset_n_acquired()
        self._check_allowed_acquisition()
        self._check_hardware_limitations()

    def acquisition_finalize(self):
        """Finalize the acquisition device.

        Performs cleanup at the end of an experiment (i.e., not repeatedly in
        sweeps). No actions by default, can be overridden in child classes.
        """
        self.timer = None
        pass

    def _reset_n_acquired(self):
        """Reset quantities that are needed by acquisition_progress.

        No actions by default, can be overridden in child classes.
        """
        pass

    def _check_allowed_acquisition(self):
        """Check whether the specified mode and data_type is supported.
        """

        if self._acq_mode not in self.allowed_modes:
            raise ValueError(f'{self._acq_mode} not supported by {self.name}.')

        if self._acq_data_type is not None and \
                self._acq_data_type not in self.allowed_modes[self._acq_mode]:
            raise ValueError(f'{self._acq_data_type} not supported by '
                             f'{self.name} for an {self._acq_mode} '
                             f'acquisition.')

    def acquisition_nodes(self):
        """Get a list of the currently used acquisition nodes.

        Acquisition nodes are unique identifiers of the acquisition
        channels. By default, they are in the same format as used in the
        channels argument of acquisition_initialize, but a child class could
        replace them by an actual hardware address (such as node names in
        ZI devices).
        """
        return deepcopy(self._acquisition_nodes)

    def acquisition_progress(self):
        """Gets the number of totally acquired data points.

        Here, total refers to counting each acquisition element as many
        times as it has been acquired already (e.g., in a loop of
        repetitions for averaging), i.e., 100% progress corresponds to the
        this method reporting self._acq_n_results * self._acq_loop_cnt.

        The method always returns 0 indicating that no intermediate progress
        information is available. If the child class does not overwrite the
        method with a concrete implementation, progress will be stuck during
        hard sweeps and will only be updated by MC after the hard sweep.
        """
        return 0

    def prepare_poll(self):
        """Final preparations for an acquisition.

        This function is called by PollDetector.poll_data right before
        starting the AWGs (i.e., we can rely on that the AWGs have already
        been stopped in case this is needed) for final preparations. This is
        the place to arm the acquisition device (i.e., let it wait for
        triggers). Unlike acquisition_initialize, this function is called
        multiple times in a soft sweep.
        """
        pass

    def poll(self, poll_time=0.1):
        """Poll data from the acquisition device.

        The method needs to be implemented in a child class.

        Args:
            poll_time (float): time in seconds that the device should wait
            for new data. If no new data is available by the end of this
            time span, an empty data set is returned.

        Returns:
             A dict, where each key is an acquisition node (see
             self.acquisition_nodes), and the corresponding value is a list
             of numpy arrays containing new data.
        """
        raise NotImplementedError(f'poll not implemented '
                                  f'for {self.__class__.__name__}')

    def start(self, **kw):
        """ Start the built-in AWG (if present).

        Args:
            **kw: currently ignored, added for compatibilty with other
                instruments that accept kwargs in start().
        """
        pass  # default acquisition device does not have an AWG

    def stop(self):
        """Stop the built-in AWG (if present).
        """
        pass  # default acquisition device does not have an AWG

    def _check_hardware_limitations(self):
        """Check whether acquisition settings are supported by the hardware

        No checks by default. Should be implemented in child classes to check
        whether the measurement settings are supported by the hardware.
        """
        pass

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        """Returns properties of the returned values of a given data type.

        Args:
            data_type (str): data type. One of the str listed in
                self.allowed_modes for the chosen mode.
            acquisition_length (float): length of the acquired signals in
                seconds

        Returns:
            A dict with the following entries:
            value_unit: the base unit of the returned values (by default a.u.
                unless overridden by the base classes).
            scaling_factor: a scaling factor that needs to be applied to the
                returned values in order to bring them into their base unit
                (by default 1, unless overridden by the base classes).
        """
        return {'value_unit': 'a.u.', 'scaling_factor': 1}

    def correct_offset(self, channels, data):
        """Correct potential offsets in the data.

        By default, an offset of 0 is assumed. Can be overridden by a child
        class.

        Args:
            channels (list of tuple of int): channels for which corrections
                need to be applied (channel format as in the docstring of
                acquisition_initialize)
            data (dict): each key is a channel from the list channels,
                and the corresponding value is a numpy array of data.

        Returns:
            A dict in the same format of data, after correcting the offsets.
        """
        return data

    def get_lo_sweep_function(self, acq_unit, ro_mod_freq):
        """Return a sweep function for sweeping the internal LO

        Needs to be implemented in the child class for acquisition devices
        with an internal LO. No implementation needed if an external LO is
        used.

        Args:
             acq_unit (int): the acquisition unit for which the LO frequency
                should be swept.
            ro_mod_freq (float): the desired intermediate frequency (IF) in
                Hertz. Depending on the implementation in the child class,
                the frequency sweep will either be done by a pure LO sweep
                for this fixed IF, or it will be implemented as a combined
                sweep of LO and IF in a way that the IF is as close as
                possible to the provided value, while accounting for
                hardware limitations of the internal LO.

        Returns:
            A sweep function object for the frequency sweep.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not have '
                                  f'an internal LO.')

    def convert_time_to_n_samples(self, time, align_acq_granularity=False):
        """Convert from time duration to number of samples.

        Calculate the number of samples that the specified time corresponds to
        based on the acquisition sampling rate of the device.

        Args:
            time (float): time in seconds
            align_acq_granularity (bool): Whether the calculated number of
                samples should account for acquisition length granularity by
                shortening the length if needed (default: False).

        Returns:
            number of samples (int)
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

    def get_sweep_points_time_trace(self, acquisition_length=None,
                                    align_acq_granularity=False):
        """
        Get the measurement sweep points expected by the framework for a
        time trace acquisition with the acquisition device.

        The sweep points array will contain time points between 0 and
        acquisition_length. The number of sweep points in the array will be
        equal to the number of samples that acquisition length converts to
        for this device (see convert_time_to_n_samples).

        Args:
            acquisition_length (float): duration of time trace acquisition in
                seconds
            align_acq_granularity (bool): Whether the calculated number of
                samples should account for acquisition length granularity by
                shortening the length if needed (default: False).
        Returns:
            array of sweep points
        """
        if acquisition_length is None:
            acquisition_length = self._acq_length
        npoints = self.convert_time_to_n_samples(acquisition_length,
                                                 align_acq_granularity)
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
            0, self.acq_weights_n_samples / self.acq_sampling_rate,
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
        """Set a vector of integration weights for a channel.

        This method needs to be implemented in each acquisition device child
        class that allows setting integration weights.

        Args:
             channel (tuple of int): Channel for which the weight vector
             should be set (channel format as in acquisition_set_weights).
            weight (tuple of numpy array): vector of complex acquisition
                weights. The first vector in the tuple is the real part, and
                the second vector in the tuple is the imaginary part.
        """
        raise NotImplementedError(f'_acquisition_set_weight not implemented '
                                  f'for {self.__class__.__name__}')


class ZI_AcquisitionDevice(AcquisitionDevice):
    """Base class for ZI acquisition devices.

    This class extends AcquisitionDevice with methods that are common to
    multiple ZI devices.

    FIXME: Check whether the overlap between UHF and SHF is really high
        enough to justify such a class.
    """

    def get_value_properties(self, data_type='raw', acquisition_length=None):
        properties = super().get_value_properties(
            data_type=data_type, acquisition_length=acquisition_length)
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

