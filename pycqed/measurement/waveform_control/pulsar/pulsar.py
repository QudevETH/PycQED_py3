import os
import shutil
import ctypes
import numpy as np
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Set, Tuple

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
import qcodes.utils.validators as vals
import pycqed.utilities.general as gen


log = logging.getLogger(__name__)


class PulsarAWGInterface(ABC):
    """Base interface for AWGs used by the :class:`Pulsar` class.

    To support another type of AWG in the pulsar class, one needs to subclass
    this interface class, and override the methods it defines. In addition, make
    sure to override the supported ``AWG_CLASSES``.

    Attributes:
        pulsar: Pulsar to which the PulsarAWGInterface is added.
        awg: AWG added to the pulsar.
    """

    AWG_CLASSES:List[type] = []
    """List of AWG classes for which the interface is meant.

    Derived classes should override this class attribute.
    """

    GRANULARITY:int = 0
    """Granularity of the AWG."""

    ELEMENT_START_GRANULARITY:float = 0.0
    """TODO: Document + provide unit."""

    MIN_LENGTH:float = 0.0
    """TODO: Document + provide unit."""

    INTER_ELEMENT_DEADTIME:float = 0.0
    """TODO: Document + provide unit."""

    _pulsar_interfaces = []
    """Registered pulsar interfaces. See :meth:`__init_subclass__`."""

    def __init__(self, pulsar:'Pulsar', awg:Instrument, channel_name_map:dict):
        super().__init__()

        self.awg = awg
        self.pulsar = pulsar

        self.create_awg_parameters(channel_name_map)


    def __init_subclass__(cls, **kwargs):
        """Hook to auto-register a new pulsar AWG interface class.

        See https://www.python.org/dev/peps/pep-0487/#subclass-registration.
        """

        super().__init_subclass__(**kwargs)
        if not cls.AWG_CLASSES:
            raise NotImplementedError("Subclasses of PulsarAWGInterface "
                                      "should override 'AWG_CLASSES'.")
        cls._pulsar_interfaces.append(cls)

    @abstractmethod
    def create_awg_parameters(self, channel_name_map:dict):
        """Create parameters in the pulsar specific to the added AWG.

        This function is called in ``__init__()``, and does not need to be
        called directly.

        Arguments:
            channel_name_map: Mapping from channel ids (keys, as string) to
                channels names (values, as string) to be used for this AWG.
                Names for missing ids default to ``{awg.name}_{chid}``.

        TODO: A lot of added parameters in subclasses use magic numbers. They
        should define meaningful class constants instead.
        TODO: Parameters declaration should be ordered better for readability.
        """

        pulsar = self.pulsar
        name = self.awg.name

        pulsar.add_parameter(f"{name}_reuse_waveforms",
                             initial_value=True, vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_minimize_sequencer_memory",
                             initial_value=True, vals=vals.Bool(),
                             parameter_class=ManualParameter,
                             docstring="Minimizes the sequencer memory by "
                                       "repeating specific sequence patterns "
                                       "(eg. readout) passed in "
                                       "'repeat dictionary'.")
        pulsar.add_parameter(f"{name}_enforce_single_element",
                             initial_value=False, vals=vals.Bool(),
                             parameter_class=ManualParameter,
                             docstring="Group all the pulses on this AWG into "
                                       "a single element. Useful for making "
                                       "sure the master AWG has only one "
                                       "waveform per segment.")
        pulsar.add_parameter(f"{name}_granularity",
                             get_cmd=lambda: self.GRANULARITY)
        pulsar.add_parameter(f"{name}_element_start_granularity",
                             initial_value=self.ELEMENT_START_GRANULARITY,
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_min_length",
                             get_cmd=lambda: self.MIN_LENGTH)
        pulsar.add_parameter(f"{name}_inter_element_deadtime",
                             get_cmd=lambda: self.INTER_ELEMENT_DEADTIME)
        pulsar.add_parameter(f"{name}_precompile",
                             initial_value=False, vals=vals.Bool(),
                             label=f"{name} precompile segments",
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_delay",
                             initial_value=0, unit="s",
                             parameter_class=ManualParameter,
                             docstring="Global delay applied to this channel. "
                                       "Positive values move pulses on this "
                                       "channel forward in time.")
        pulsar.add_parameter(f"{name}_trigger_channels",
                             initial_value=[],
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_active",
                             initial_value=True,
                             vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_compensation_pulse_min_length",
                             initial_value=0, unit='s',
                             parameter_class=ManualParameter)

    @abstractmethod
    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all"):
        """Upload the waveforms to the AWG.

        Args:
            awg_sequence: AWG sequence data (not waveforms) as returned from
                ``Sequence.generate_waveforms_sequences``. The key-structure of
                the nested dictionary is like this:
                ``awg_sequence[elname][cw][chid]``, where ``elname`` is the
                element name, cw is the codeword, or the string
                ``"no_codeword"`` and ``chid`` is the channel id. The values are
                hashes of waveforms to be played.
            waveforms: A dictionary of waveforms, keyed by their hash.
            repeat_pattern: Not used for now
            channels_to_upload: list of channel names to upload or ``"all"``.
        """

    @abstractmethod
    def is_awg_running(self) -> bool:
        """Checks whether the sequencer of the AWG is running."""
        pass

    @abstractmethod
    def clock_frequency(self) -> float:
        """Returns the sample clock frequency [Hz] of the AWG."""
        pass

    @abstractmethod
    def sigout_on(self, ch, on:bool=True):
        """Turn channel outputs on or off."""
        pass

    @abstractmethod
    def get_segment_filter_userregs(self) -> List[Tuple[str, str]]:
        """Returns the list of segment filter userregs.

        Returns:
            List of tuples ``(first, last)`` where ``first`` and ``last`` are
            formatted strings. TODO: More accurate description.
        """
        pass


class Pulsar(Instrument):
    """A meta-instrument responsible for all communication with the AWGs.

    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming and changing the parameters of the AWGs
    should be done through Pulsar.

    TODO: All the parameters for added AWGs are named as formatted strings
    containing the names of the AWGs. We could instead use one instrument
    submodule per AWG to make names less verbose.

    Supported AWGs:
    * Tektronix AWG5014
    * ZI HDAWG8
    * ZI UHFQC
    * ZI SHFQA

    Attributes:
        awgs: Names of AWGs in pulsar.
    """

    def __init__(self, name:str='Pulsar', master_awg:str=None):
        """Pulsar constructor.

        Args:
            name: Instrument name.
            master_awg: Name of the AWG that triggers all the other AWG-s and
                should be started last (after other AWG-s are already
                waiting for a trigger.
        """

        super().__init__(name)

        self._sequence_cache = dict()
        self.reset_sequence_cache()

        self.add_parameter('master_awg',
                           parameter_class=InstrumentRefParameter,
                           initial_value=master_awg)
        self.add_parameter('inter_element_spacing',
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum('auto')),
                           set_cmd=self._set_inter_element_spacing,
                           get_cmd=self._get_inter_element_spacing)
        self.add_parameter('reuse_waveforms', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('use_sequence_cache', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool(),
                           set_parser=self._use_sequence_cache_parser)
        self.add_parameter('prepend_zeros', initial_value=0, vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('flux_crosstalk_cancellation', initial_value=False,
                           parameter_class=ManualParameter)
        self.add_parameter('flux_channels', initial_value=[],
                           parameter_class=ManualParameter)
        self.add_parameter('flux_crosstalk_cancellation_mtx',
                           initial_value=None, parameter_class=ManualParameter)
        self.add_parameter('flux_crosstalk_cancellation_shift_mtx',
                           initial_value=None, parameter_class=ManualParameter)
        self.add_parameter('resolve_overlapping_elements', vals=vals.Bool(),
                           initial_value=False, parameter_class=ManualParameter,
                           docstring='Flag determining whether overlapping '
                                     'elements should be resolved by '
                                     'combining them into one element. NB: '
                                     'overlap resolution only applies to'
                                     'non-trigger elements!')
        # This parameter can be used to record only a specified consecutive
        # subset of segments of a programmed hard sweep. This is used by the
        # sweep function FilteredSweep. The parameter expects a tuple of indices
        # indicating the first and the last segment to be measured. (Segments
        # with the property allow_filter set to False are always measured.)
        self.add_parameter('filter_segments',
                           set_cmd=self._set_filter_segments,
                           get_cmd=self._get_filter_segments,
                           initial_value=None)
        self.add_parameter('sigouts_on_after_programming', initial_value=True,
                           parameter_class=ManualParameter, vals=vals.Bool(),
                           docstring='Whether signal outputs should be '
                                     'switched off automatically after '
                                     'programming a AWGs. Can be set to '
                                     'False to save time if it is ensured '
                                     'that the channels are switched on '
                                     'somewhere else.')

        self._inter_element_spacing = 'auto'
        self.channels = set() # channel names
        self.awgs:Set[str] = set() # AWG names
        self.last_sequence = None
        self.last_elements = None
        self._awgs_with_waveforms = set()
        self.channel_groups = {}
        self.num_channel_groups = {}

        self._awgs_prequeried_state = False

        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}
        self._filter_segments = None
        self._filter_segment_functions = {}

        self.num_seg = 0

        Pulsar._instance = self

    # TODO: Should Pulsar be a singleton ? Is it really necessary to have such
    # a method ?
    @classmethod
    def get_instance(cls):
        return cls._instance

    def _use_sequence_cache_parser(self, val):
        if val and not self.use_sequence_cache():
            self.reset_sequence_cache()
        return val

    def reset_sequence_cache(self):
        self._sequence_cache = {}
        # TODO: If this cache contains stuff for each AWG, it may make more
        # sense to store it in PulsarAWGInterface instead ?
        # The following dicts are used in _program_awgs to store information
        # about the last sequence programmed to each AWGs. The keys of the
        # dicts are AWG names and/or channel names. See the code and
        # comments of _program_awgs for details about the structure of the
        # dicts.
        self._sequence_cache['settings'] = {}  # for pulsar settings
        self._sequence_cache['metadata'] = {}  # for segment/element metadata
        self._sequence_cache['hashes'] = {}  # for waveform hashes
        self._sequence_cache['length'] = {}  # for element lengths

    def check_for_other_pulsar(self):
        """
        Checks whether another pulsar has programmed the AWGs and resets the
        sequence cache if this is the case. To make this check possible,
        the pulsar object ID is written to a file in the pycqed app data dir.
        """
        filename = os.path.join(gen.get_pycqed_appdata_dir(), 'pulsar_id')
        current_id = f"{id(self)}"
        try:
            with open(filename, 'r') as f:
                stored_id = f.read()
        except:
            stored_id = None
        if stored_id != current_id:
            log.debug('Another pulsar instance has programmed the AWGs. '
                      'Resetting sequence cache.')
            self.reset_sequence_cache()
        with open(filename, 'w') as f:
            f.write(current_id)

    def define_awg_channels(self, awg, channel_name_map=None):
        """
        The AWG object must be created before creating channels for that AWG

        Args:
            awg: AWG object to add to the pulsar.
            channel_name_map: A dictionary that maps channel ids to channel
                              names. (default {})

        TODO: This is the key method to add more AWGs. Refactor it according to
        new interface class.
        """

        if channel_name_map is None:
            channel_name_map = {}

        # Sanity checks
        for channel_name in channel_name_map.values():
            if channel_name in self.channels:
                raise KeyError("Channel named '{}' already defined".format(
                    channel_name))
        if awg.name in self.awgs:
            raise KeyError("AWG '{}' already added to pulsar".format(awg.name))

        super()._create_awg_parameters(awg, channel_name_map)
        # Reconstruct the set of unique channel groups from the
        # self.channel_groups dictionary, which stores for each channel a list
        # of all channels in the same group.
        self.num_channel_groups[awg.name] = len(set(
            ['---'.join(v) for k, v in self.channel_groups.items()
             if self.get('{}_awg'.format(k)) == awg.name]))

        self.awgs.add(awg.name)
        # Make sure that registers for filter_segments are set in the new AWG.
        self.filter_segments(self.filter_segments())

    def find_awg_channels(self, awg:str) -> List[str]:
        """Return a list of channels associated to an AWG."""

        channel_list = []
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) == awg:
                channel_list.append(channel)

        return channel_list

    # TODO: Rename to conform to naming conventions
    def AWG_obj(self, awg:str, channel:str):
        """
        Return the AWG object corresponding to a channel or an AWG name.

        Args:
            awg: Name of the AWG Instrument.
            channel: Name of the channel

        Returns: An instance of Instrument class corresponding to the AWG
                 requested.
        """

        if awg is not None and channel is not None:
            raise ValueError('Both `awg` and `channel` arguments passed to '
                             'Pulsar.AWG_obj()')
        elif awg is None and channel is not None:
            name = self.get('{}_awg'.format(channel))
        elif awg is not None and channel is None:
            name = awg
        else:
            raise ValueError('Either `awg` or `channel` argument needs to be '
                             'passed to Pulsar.AWG_obj()')
        return Instrument.find_instrument(name)

    # TODO: Ordering of parameters in method is inconsistent
    def clock(self, channel=None, awg=None):
        """Returns the clock rate of channel or AWG.

        Arguments:
            channel: name of the channel.
            awg: AWG.

        Returns: clock rate in samples per second.
        """

        if channel is not None and awg is not None:
            raise ValueError('Both channel and awg arguments passed to '
                             'Pulsar.clock()')
        if channel is None and awg is None:
            raise ValueError('Neither channel nor awg arguments passed to '
                             'Pulsar.clock()')

        if channel is not None:
            awg = self.get('{}_awg'.format(channel))

        if self._awgs_prequeried_state:
            return self._clocks[awg]
        else:
            fail = None
            obj = self.AWG_obj(awg=awg)
            try:
                return super()._clock(obj)
            except AttributeError as e:
                fail = e
            if fail is not None:
                raise TypeError('Unsupported AWG instrument: {} of type {}. '
                                .format(obj.name, type(obj)) + str(fail))

    def active_awgs(self):
        """
        Returns:
            A set of the names of the active AWGs registered

            Inactive AWGs don't get started or stopped. Also the waveforms on
            inactive AWGs don't get updated.
        """
        return {awg for awg in self.awgs if self.get('{}_active'.format(awg))}

    def awgs_with_waveforms(self, awg=None):
        """
        Adds an awg to the set of AWGs with waveforms programmed, or returns
        set of said AWGs.

        TODO: Should either be a private method, or it should also perform
        checks to make sure awg is already part of pulsar.
        """
        if awg == None:
            return self._awgs_with_waveforms
        else:
            self._awgs_with_waveforms.add(awg)
            self._set_filter_segments(self._filter_segments, [awg])

    def start(self, exclude:List[str]=None, stop_first:bool=True):
        """Start the active AWGs.

        If multiple AWGs are used in a setup where the
        slave AWGs are triggered by the master AWG, then the slave AWGs must be
        running and waiting for trigger when the master AWG is started to
        ensure synchronous playback.

        Arguments:
            exclude: Names of AWGs to exclude
            stop_first: Whether all used AWGs should be stopped before starting
                the AWGs.
        """

        if exclude is None:
            exclude = []

        # Start only the AWGs which have at least one channel programmed, i.e.
        # where at least one channel has state = 1.
        awgs_with_waveforms = self.awgs_with_waveforms()
        used_awgs = self.active_awgs() & awgs_with_waveforms

        if stop_first:
            for awg in used_awgs:
                self._stop_awg(awg)

        if self.master_awg() is None:
            for awg in used_awgs:
                if awg not in exclude:
                    self._start_awg(awg)
        else:
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().stop()
            for awg in used_awgs:
                if awg != self.master_awg() and awg not in exclude:
                    self._start_awg(awg)
            tstart = time.time()
            for awg in used_awgs:
                if awg == self.master_awg() or awg in exclude:
                    continue
                good = False
                while not (good or time.time() > tstart + 10):
                    if self._is_awg_running(awg):
                        good = True
                    else:
                        time.sleep(0.1)
                if not good:
                    raise Exception('AWG {} did not start in 10s'
                                    .format(awg))
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().start()

    def stop(self):
        """Stop all active AWGs."""

        awgs_with_waveforms = set(self.awgs_with_waveforms())
        used_awgs = set(self.active_awgs()) & awgs_with_waveforms

        for awg in used_awgs:
            self._stop_awg(awg)

    def program_awgs(self, sequence, awgs='all'):
        try:
            self._program_awgs(sequence, awgs)
        except Exception as e:
            if not self.use_sequence_cache():
                raise
            log.warning(f'Pulsar: Exception {repr(e)} while programming AWGs. '
                        f'Retrying after resetting the sequence cache.')
            self.reset_sequence_cache()
            self._program_awgs(sequence, awgs)

    def _program_awgs(self, sequence, awgs='all'):

        # Stores the last uploaded sequence for easy access and plotting
        self.last_sequence = sequence

        if awgs == 'all':
            awgs = self.active_awgs()

        # initializes the set of AWGs with waveforms
        self._awgs_with_waveforms -= awgs


        # prequery all AWG clock values and AWG amplitudes
        self.AWGs_prequeried(True)

        log.info(f'Starting compilation of sequence {sequence.name}')
        t0 = time.time()
        if self.use_sequence_cache():
            # reset the sequence cache if another pulsar instance has
            # programmed the AWGs
            self.check_for_other_pulsar()
            # get hashes and information about the sequence structure
            channel_hashes, awg_sequences = \
                sequence.generate_waveforms_sequences(get_channel_hashes=True)
            log.debug(f'End of waveform hashing sequence {sequence.name} '
                      f'{time.time() - t0}')
            sequence_cache = self._sequence_cache
            # The following makes sure that the sequence cache is empty if
            # the compilation crashes or gets interrupted.
            self.reset_sequence_cache()
            # Add an empty hash for previously active but now inactive channels
            # in active channel groups. This is to make sure that the change
            # (switching them off) is detected correctly below.
            channel_hashes.update({
                k: {} for k, v in sequence_cache['hashes'].items()
                if k not in channel_hashes and len(v)
                and any([k in self.channel_groups[ch]
                         for ch in channel_hashes.keys()])})
            # first, we check whether programming the whole AWG is mandatory due
            # to changed AWG settings or due to changed metadata
            awgs_to_program = []
            settings_to_check = ['{}_use_placeholder_waves',
                                 '{}_prepend_zeros',
                                 'prepend_zeros']
            settings = {}
            metadata = {}
            for awg, seq in awg_sequences.items():
                settings[awg] = {
                    s.format(awg): (
                        self.get(s.format(awg))
                        if s.format(awg) in self.parameters else None)
                    for s in settings_to_check}
                metadata[awg] = {
                    elname: (
                        el.get('metadata', {}) if el is not None else None)
                    for elname, el in seq.items()}
                if awg not in awgs_to_program:
                    try:
                        np.testing.assert_equal(
                            sequence_cache['settings'].get(awg, {}),
                            settings[awg])
                        np.testing.assert_equal(
                            sequence_cache['metadata'].get(awg, {}),
                            metadata[awg])
                    except AssertionError:  # settings or metadata change
                        awgs_to_program.append(awg)
            for awg in awgs_to_program:
                # update the settings and metadata cache
                sequence_cache['settings'][awg] = settings[awg]
                sequence_cache['metadata'][awg] = metadata[awg]
            # Check for which channels some relevant setting or some hash has
            # changed, in which case the group of channels should be uploaded.
            settings_to_check = ['{}_internal_modulation']
            awgs_with_channels_to_upload = []
            channels_to_upload = []
            channels_to_program = []
            for ch, hashes in channel_hashes.items():
                ch_awg = self.get(f'{ch}_awg')
                settings[ch] = {
                    s.format(ch): (
                        self.get(s.format(ch))
                        if s.format(ch) in self.parameters else None)
                    for s in settings_to_check}
                metadata[ch] = {'repeat_pattern':
                                    sequence.repeat_patterns.get(ch, None)}
                if ch in channels_to_upload or ch_awg in awgs_to_program:
                    continue
                changed_settings = True
                try:
                    np.testing.assert_equal(
                        sequence_cache['settings'].get(ch, {}),
                        settings[ch])
                    changed_settings = False
                    np.testing.assert_equal(
                        sequence_cache['hashes'].get(ch, {}), hashes)
                    np.testing.assert_equal(
                        sequence_cache['metadata'].get(ch, {}), metadata[ch])
                except AssertionError:
                    # changed setting, sequence structure, or hash
                    if ch_awg not in awgs_with_channels_to_upload:
                        awgs_with_channels_to_upload.append(ch_awg)
                    for c in self.channel_groups[ch]:
                        channels_to_upload.append(c)
                        if changed_settings:
                            channels_to_program.append(c)
            # update the settings cache and hashes cache
            for ch in channels_to_upload:
                sequence_cache['settings'][ch] = settings.get(ch, {})
                sequence_cache['hashes'][ch] = channel_hashes.get(ch, {})
                sequence_cache['metadata'][ch] = metadata.get(ch, {})
            # generate the waveforms that we need for uploading
            log.debug(f'Start of waveform generation sequence {sequence.name} '
                     f'{time.time() - t0}')
            waveforms, _ = sequence.generate_waveforms_sequences(
                awgs_to_program + awgs_with_channels_to_upload,
                resolve_segments=False)
            log.debug(f'End of waveform generation sequence {sequence.name} '
                     f'{time.time() - t0}')
            # Check for which channels the sequence structure, or some element
            # length has changed.
            # If placeholder waveforms are used, only those channels (and
            # channels in the same group) will be re-programmed, while other
            # channels can be re-uploaded by replacing the existing waveforms.
            ch_length = {}
            for ch, hashes in channel_hashes.items():
                ch_awg = self.get(f'{ch}_awg')
                if ch_awg in awgs_to_program + awgs_with_channels_to_upload:
                    ch_length[ch] = {
                        elname: {cw: len(waveforms[h]) for cw, h in el.items()}
                        for elname, el in hashes.items()}
                # Checking whether programming is needed is done only for
                # channels that are marked to be uploaded but not yet marked
                # to be programmed.
                if ch not in channels_to_upload or ch in channels_to_program \
                        or ch_awg in awgs_to_program:
                    continue
                try:
                    np.testing.assert_equal(
                        sequence_cache['length'].get(ch, {}),
                        ch_length[ch])
                except AssertionError:  # changed length or sequence structure
                    for c in self.channel_groups[ch]:
                        channels_to_program.append(c)
            # update the length cache
            for ch in channels_to_program:
                sequence_cache['length'][ch] = ch_length.get(ch, {})
            # Update the cache for channels that are on an AWG marked for
            # complete re-programming (these channels might have been skipped
            # above).
            for ch in self.channels:
                if self.get(f'{ch}_awg') in awgs_to_program:
                    sequence_cache['settings'][ch] = settings.get(ch, {})
                    sequence_cache['hashes'][ch] = channel_hashes.get(
                        ch, {})
                    sequence_cache['metadata'][ch] = metadata.get(ch, {})
                    sequence_cache['length'][ch] = ch_length.get(ch, {})
            log.debug(f'awgs_to_program = {repr(awgs_to_program)}\n'
                      f'awgs_with_channels_to_upload = '
                      f'{repr(awgs_with_channels_to_upload)}\n'
                      f'channels_to_upload = {repr(channels_to_upload)}\n'
                      f'channels_to_program = {repr(channels_to_program)}'
                      )
        else:
            waveforms, awg_sequences = sequence.generate_waveforms_sequences()
            awgs_to_program = list(awg_sequences.keys())
            awgs_with_channels_to_upload = []
        log.info(f'Finished compilation of sequence {sequence.name} in '
                 f'{time.time() - t0}')

        channels_used = self._channels_in_awg_sequences(awg_sequences)
        repeat_dict = self._generate_awg_repeat_dict(sequence.repeat_patterns,
                                                     channels_used)
        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}

        for awg in awg_sequences.keys():
            if (awg not in awgs_to_program + awgs_with_channels_to_upload and
                    self.num_channel_groups[awg] == 1):
                # The AWG does not need to be re-programmed, but we have to add
                # it to the set of AWGs with waveforms (which is otherwise
                # done after programming it).
                # Note: If num_channel_groups is not 1, _program_awg will be
                # called with an empty channels_to_upload list to make sure
                # that the correct sub-AWGs get started.
                self.awgs_with_waveforms(awg)
                continue
            log.info(f'Started programming {awg}')
            t0 = time.time()
            if awg in awgs_to_program:
                ch_upl, ch_prg = 'all', 'all'
            else:
                ch_upl = [self.get(f'{ch}_id') for ch in channels_to_upload
                          if self.get(f'{ch}_awg') == awg]
                ch_prg = [self.get(f'{ch}_id') for ch in channels_to_program
                          if self.get(f'{ch}_awg') == awg]
            if awg in repeat_dict.keys():
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms,
                                  repeat_pattern=repeat_dict[awg],
                                  channels_to_upload=ch_upl,
                                  channels_to_program=ch_prg)
            else:
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms,
                                  channels_to_upload=ch_upl,
                                  channels_to_program=ch_prg)
            log.info(f'Finished programming {awg} in {time.time() - t0}')

        if self.use_sequence_cache():
            # Compilation finished sucessfully. Store sequence cache.
            self._sequence_cache = sequence_cache
        self.num_seg = len(sequence.segments)
        self.AWGs_prequeried(False)

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):
        """Program the AWG with a sequence of segments.

        Args:
            obj: the instance of the AWG to program
            sequence: the `Sequence` object that determines the segment order,
                      repetition and trigger wait
            loop: Boolean flag, whether the segments should be looped over.
                  Default is `True`.

        TODO this should not be part of this class but of Pulsar AWG interfaces.
        """

        if repeat_pattern is not None:
            super()._program_awg(obj, awg_sequence, waveforms,
                                 repeat_pattern=repeat_pattern, **kw)
        else:
            super()._program_awg(obj, awg_sequence, waveforms, **kw)

    def _hash_to_wavename(self, h):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        if h not in self._hash_to_wavename_table:
            hash_int = abs(hash(h))
            wname = ''.join(to_base(hash_int, len(alphabet), alphabet))[::-1]
            while wname in self._hash_to_wavename_table.values():
                hash_int += 1
                wname = ''.join(to_base(hash_int, len(alphabet), alphabet)) \
                    [::-1]
            self._hash_to_wavename_table[h] = wname
        return self._hash_to_wavename_table[h]

    def _zi_wave_definition(self, wave, defined_waves=None,
                            placeholder_wave_length=None,
                            placeholder_wave_index=None):
        if defined_waves is None:
            if placeholder_wave_length is None:
                defined_waves = set()
            else:
                defined_waves = set(), dict()
        wave_definition = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if placeholder_wave_length is None:
            # don't use placeholder waves
            for analog, marker, wc in [(wave[0], wave[1], w1),
                                       (wave[2], wave[3], w2)]:
                if analog is not None:
                    wa = self._hash_to_wavename(analog)
                    if wa not in defined_waves:
                        wave_definition.append(f'wave {wa} = "{wa}";')
                        defined_waves.add(wa)
                if marker is not None:
                    wm = self._hash_to_wavename(marker)
                    if wm not in defined_waves:
                        wave_definition.append(f'wave {wm} = "{wm}";')
                        defined_waves.add(wm)
                if analog is not None and marker is not None:
                    if wc not in defined_waves:
                        wave_definition.append(f'wave {wc} = {wa} + {wm};')
                        defined_waves.add(wc)
        else:
            # use placeholder waves
            n = placeholder_wave_length
            if w1 is None and w2 is not None:
                w1 = f'{w2}_but_zero'
            for wc, marker in [(w1, wave[1]), (w2, wave[3])]:
                if wc is not None and wc not in defined_waves[0]:
                    wave_definition.append(
                        f'wave {wc} = placeholder({n}' +
                        ('' if marker is None else ', true') +
                        ');')
                    defined_waves[0].add(wc)
            wave_definition.append(
                f'assignWaveIndex({_zi_wavename_pair_to_argument(w1, w2)},'
                f' {placeholder_wave_index});'
            )
            defined_waves[1][placeholder_wave_index] = wave
        return wave_definition

    def _zi_playback_string(self, name, device, wave, acq=False, codeword=False,
                            prepend_zeros=0, placeholder_wave=False,
                            allow_filter=False):
        playback_string = []
        if allow_filter:
            playback_string.append(
                'if (i_seg >= first_seg && i_seg <= last_seg) {')
        if prepend_zeros:
            playback_string.append(f'playZero({prepend_zeros});')
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True # set this to false once the bugs with HDAWG are fixed
        trig_source = self.get('{}_trigger_source'.format(name))
        if trig_source == 'Dig1':
            playback_string.append(
                'waitDigTrigger(1{});'.format(', 1' if device == 'uhf' else ''))
        elif trig_source == 'Dig2':
            playback_string.append('waitDigTrigger(2,1);')
        else:
            playback_string.append(f'wait{trig_source}Trigger();')

        if codeword and not (w1 is None and w2 is None):
            playback_string.append('playWaveDIO();')
        else:
            if w1 is None and w2 is not None and use_hack and not placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave(marker(1,0)*0*{w2}, {w2});')
            elif w1 is None and w2 is not None and use_hack and placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave({w2}_but_zero, {w2});')
            elif w1 is not None and w2 is None and use_hack and not placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave({w1}, marker(1,0)*0*{w1});')
            elif w1 is not None or w2 is not None:
                playback_string.append('playWave({});'.format(
                    _zi_wavename_pair_to_argument(w1, w2)))
        if acq:
            playback_string.append('setTrigger(RO_TRIG);')
            playback_string.append('setTrigger(WINT_EN);')
        if allow_filter:
            playback_string.append('}')
        return playback_string

    def _zi_interleaved_playback_string(self, name, device, counter,
                                        wave, acq=False, codeword=False):
        playback_string = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if w1 is None or w2 is None:
            raise ValueError('When using HDAWG modulation both I and Q need '
                              'to be defined')

        wname = f'wave{counter}'
        interleaves = [f'wave {wname} = interleave({w1}, {w2});']

        if not codeword:
            if not acq:
                playback_string.append(f'prefetch({wname},{wname});')

        trig_source = self.get('{}_trigger_source'.format(name))
        if trig_source == 'Dig1':
            playback_string.append(
                'waitDigTrigger(1{});'.format(', 1' if device == 'uhf' else ''))
        elif trig_source == 'Dig2':
            playback_string.append('waitDigTrigger(2,1);')
        else:
            playback_string.append(f'wait{trig_source}Trigger();')

        if codeword:
            # playback_string.append('playWaveDIO();')
            raise NotImplementedError('Modulation in combination with codeword'
                                      'pulses has not yet been implemented!')
        else:
            playback_string.append(f'playWave({wname},{wname});')
        if acq:
            playback_string.append('setTrigger(RO_TRIG);')
            playback_string.append('setTrigger(WINT_EN);')
        return playback_string, interleaves

    @staticmethod
    def _zi_playback_string_loop_start(metadata, channels):
        loop_len = metadata.get('loop', False)
        if not loop_len:
            return []
        playback_string = []
        sweep_params = metadata.get('sweep_params', {})
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f'{ch}_'):
                    playback_string.append(
                        f"wave {k} = vect({','.join([f'{a}' for a in v])})")
        playback_string.append(
            f"for (cvar i_sweep = 0; i_sweep < {loop_len}; i_sweep += 1) {{")
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f'{ch}_'):
                    node = k[len(f'{ch}_'):].replace('_', '/')
                    playback_string.append(
                        f'setDouble("{node}", {k}[i_sweep]);')
        return playback_string

    @staticmethod
    def _zi_playback_string_loop_end(metadata):
        return ['}'] if metadata.get('end_loop', False) else []

    def _zi_codeword_table_entry(self, codeword, wave, placeholder_wave=False):
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True
        if w1 is None and w2 is not None and use_hack and not placeholder_wave:
            # This hack is needed due to a bug on the HDAWG.
            # Remove this if case once the bug is fixed.
            return [f'setWaveDIO({codeword}, zeros(1) + marker(1, 0), {w2});']
        elif w1 is None and w2 is not None and use_hack and placeholder_wave:
            return [f'setWaveDIO({codeword}, {w2}_but_zero, {w2});']
        elif not (w1 is None and w2 is None):
            return ['setWaveDIO({}, {});'.format(codeword,
                        _zi_wavename_pair_to_argument(w1, w2))]
        else:
            return []

    def _zi_waves_to_wavenames(self, wave):
        wavenames = []
        for analog, marker in [(wave[0], wave[1]), (wave[2], wave[3])]:
            if analog is None and marker is None:
                wavenames.append(None)
            elif analog is None and marker is not None:
                wavenames.append(self._hash_to_wavename(marker))
            elif analog is not None and marker is None:
                wavenames.append(self._hash_to_wavename(analog))
            else:
                wavenames.append(self._hash_to_wavename((analog, marker)))
        return wavenames

    def _zi_write_waves(self, waveforms):
        wave_dir = _zi_wave_dir()
        for h, wf in waveforms.items():
            filename = os.path.join(wave_dir, self._hash_to_wavename(h)+'.csv')
            if os.path.exists(filename):
                continue
            fmt = '%.18e' if wf.dtype == np.float else '%d'
            np.savetxt(filename, wf, delimiter=",", fmt=fmt)

    def _start_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.start()

    def _stop_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.stop()

    def _is_awg_running(self, awg):
        fail = None
        obj = self.AWG_obj(awg=awg)
        try:
            return super()._is_awg_running(obj)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))

    def _set_inter_element_spacing(self, val):
        self._inter_element_spacing = val

    def _get_inter_element_spacing(self):
        if self._inter_element_spacing != 'auto':
            return self._inter_element_spacing
        else:
            max_spacing = 0
            for awg in self.awgs:
                max_spacing = max(max_spacing, self.get(
                    '{}_inter_element_deadtime'.format(awg)))
            return max_spacing

    def _set_filter_segments(self, val, awgs='with_waveforms'):
        if val is None:
            # TODO: This should be class constants, and could also be used as
            # default value for the filter_segments parameter
            val = (0, 32767)
        self._filter_segments = val
        if awgs == 'with_waveforms':
            awgs = self.awgs_with_waveforms()
        elif awgs == 'all':
            awgs = self.awgs
        for AWG_name in awgs:
            AWG = self.AWG_obj(awg=AWG_name)
            fnc = self._filter_segment_functions.get(AWG_name, None)
            if fnc is None:
                for regs in self._get_segment_filter_userregs(AWG):
                    AWG.set(regs[0], val[0])
                    AWG.set(regs[1], val[1])
            else:
                # used in case of a repeat pattern
                for regs in self._get_segment_filter_userregs(AWG):
                    AWG.set(regs[1], fnc(val[0], val[1]))

    def _get_filter_segments(self):
        return self._filter_segments

    def AWGs_prequeried(self, status=None):
        if status is None:
            return self._awgs_prequeried_state
        elif status:
            self._awgs_prequeried_state = False
            self._clocks = {}
            for awg in self.awgs:
                self._clocks[awg] = self.clock(awg=awg)
            for c in self.channels:
                # prequery also the output amplitude values
                self.get(c + '_amp')
            self._awgs_prequeried_state = True
        else:
            self._awgs_prequeried_state = False

    def _id_channel(self, cid, awg):
        """
        Returns the channel name corresponding to the channel with id `cid` on
        the AWG `awg`.

        Args:
            cid: An id of one of the channels.
            awg: The name of the AWG.

        Returns: The corresponding channel name. If the channel is not found,
                 returns `None`.
        """
        for cname in self.channels:
            if self.get('{}_awg'.format(cname)) == awg and \
               self.get('{}_id'.format(cname)) == cid:
                return cname
        return None

    @staticmethod
    def _channels_in_awg_sequences(awg_sequences):
        """
        identifies all channels used in the given awg keyed sequence
        :param awg_sequences (dict): awg sequences keyed by awg name, i.e. as
        returned by sequence.generate_sequence_waveforms()
        :return: dictionary keyed by awg of with all channel used during the sequence
        """
        channels_used = dict()
        for awg in awg_sequences:
            channels_used[awg] = set()
            for segname in awg_sequences[awg]:
                if awg_sequences[awg][segname] is None:
                    continue
                elements = awg_sequences[awg][segname]
                for cw in elements:
                    if cw != "metadata":
                        channels_used[awg] |= elements[cw].keys()
        return channels_used

    def _generate_awg_repeat_dict(self, repeat_dict_per_ch, channels_used):
        """
        Translates a repeat dictionary keyed by channels to a repeat dictionary
        keyed by awg. Checks whether all channels in channels_used have an entry.
        :param repeat_dict_per_ch: keys: channels_id, values: repeat pattern
        :param channels_used (dict): list of channel used on each awg
        :return:
        """
        awg_ch_repeat_dict = dict()
        repeat_dict_per_awg = dict()
        for cname in repeat_dict_per_ch:
            awg = self.get(f"{cname}_awg")
            chid = self.get(f"{cname}_id")

            if not awg in awg_ch_repeat_dict.keys():
                awg_ch_repeat_dict[awg] = []
            awg_ch_repeat_dict[awg].append(chid)
            if repeat_dict_per_awg.get(awg, repeat_dict_per_ch[cname]) \
                    != repeat_dict_per_ch[cname]:
                raise NotImplementedError(f"Repeat pattern on {cname} is "
                f"different from at least one other channel on {awg}:"
                f"{repeat_dict_per_ch[cname]} vs {repeat_dict_per_awg[awg]}")
            repeat_dict_per_awg[awg] = repeat_dict_per_ch[cname]

        for awg_repeat, chs_repeat in awg_ch_repeat_dict.items():
            for ch in channels_used[awg_repeat]:
                assert ch in chs_repeat, f"Repeat pattern " \
                    f"provided for {awg_repeat} but no pattern was given on " \
                    f"{ch}. All used channels on the same awg must have a " \
                    f"repeat pattern."

        return repeat_dict_per_awg


def to_base(n, b, alphabet=None, prev=None):
    if prev is None: prev = []
    if n == 0:
        if alphabet is None: return prev
        else: return [alphabet[i] for i in prev]
    return to_base(n//b, b, alphabet, prev+[n%b])

def _zi_wave_dir():
    if os.name == 'nt':
        dll = ctypes.windll.shell32
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH + 1)
        if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
            _basedir = buf.value
        else:
            log.warning('Could not extract my documents folder')
    else:
        _basedir = os.path.expanduser('~')
    wave_dir = os.path.join(_basedir, 'Zurich Instruments', 'LabOne',
        'WebServer', 'awg', 'waves')
    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)
    return wave_dir


def _zi_clear_waves():
    wave_dir = _zi_wave_dir()
    for f in os.listdir(wave_dir):
        if f.endswith(".csv"):
            os.remove(os.path.join(wave_dir, f))
        elif f.endswith('.cache'):
            shutil.rmtree(os.path.join(wave_dir, f))


def _zi_wavename_pair_to_argument(w1, w2):
    if w1 is not None and w2 is not None:
        return f'{w1}, {w2}'
    elif w1 is not None and w2 is None:
        return f'1, {w1}'
    elif w1 is None and w2 is not None:
        return f'2, {w2}'
    else:
        return ''