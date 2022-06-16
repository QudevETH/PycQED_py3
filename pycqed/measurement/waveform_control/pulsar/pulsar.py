import os
from copy import deepcopy
from collections import OrderedDict as odict

import numpy as np
import logging
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Set, Tuple, Type, Union

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter, InstrumentRefParameter
import qcodes.utils.validators as vals
import pycqed.utilities.general as gen
from pycqed.utilities.timer import WatchdogTimer, WatchdogException

from .zi_pulsar_mixin import ZIPulsarMixin


log = logging.getLogger(__name__)


_DEFAULT_SEGMENT = (0, 32767)


class PulsarAWGInterface(ABC):
    """Base interface for AWGs used by the :class:`Pulsar` class.

    To support another type of AWG in the pulsar class, one needs to subclass
    this interface class, and override the methods it defines. In addition, make
    sure to override the supported :attr:`AWG_CLASSES`.

    Warning:
        Pulsar interfaces must be imported in ``pulsar/__init__.py`` to make
        sure they are properly registered in
        ``PulsarAWGInterface._pulsar_interfaces``.

    Attributes:
        pulsar: Pulsar to which the PulsarAWGInterface is added.
        awg: AWG added to the pulsar.
    """

    AWG_CLASSES:List[Type[Instrument]] = []
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

    CHANNEL_AMPLITUDE_BOUNDS:Dict[str, Tuple[float, float]] = {}
    """Dictionary containing the amplitude boudaries for each type of channels.

    The keys are the channel types (``"analog"`` or ``"marker"``) and the values
    are tuples of floats ``(min, max)``.
    """

    CHANNEL_OFFSET_BOUNDS:Dict[str, Tuple[float, float]] = {}
    """Dictionary containing the offset boudaries for each type of channels.

    Format is similar to :attr:`CHANNEL_AMPLITUDE_BOUNDS`.
    """

    CHANNEL_RANGE_BOUNDS:Dict[str, Tuple[float, float]] = {}
    """Dictionary containing the range boudaries for each type of channels.

    Format is similar to :attr:`CHANNEL_AMPLITUDE_BOUNDS`.
    """

    CHANNEL_CENTERFREQ_BOUNDS:Dict[str, Tuple[float, float]] = {}
    """Dictionary containing the center freq boudaries for each type of channels.

    Format is similar to :attr:`CHANNEL_AMPLITUDE_BOUNDS`.
    """

    DEFAULT_SEGMENT = _DEFAULT_SEGMENT
    """TODO"""

    IMPLEMENTED_ACCESSORS:List[str] = ["offset", "amp"]
    """List of parameters that can be set or retrieved by :meth:`awg_setter`
    and :meth:`awg_getter`.
    """

    _pulsar_interfaces:List[Type['PulsarAWGInterface']] = []
    """Registered pulsar interfaces. See :meth:`__init_subclass__`."""

    def __init__(self, pulsar:'Pulsar', awg:Instrument):
        super().__init__()

        self.awg = awg
        self.pulsar = pulsar

        self._filter_segment_functions = None
        self._filter_segments_emulation_cache = None

    def __init_subclass__(cls, **kwargs):
        """Hook to auto-register a new pulsar AWG interface class.

        See https://www.python.org/dev/peps/pep-0487/#subclass-registration.
        """

        super().__init_subclass__(**kwargs)
        if not cls.AWG_CLASSES:
            raise NotImplementedError("Subclasses of PulsarAWGInterface "
                                      "should override 'AWG_CLASSES'.")
        cls._pulsar_interfaces.append(cls)

    @classmethod
    def get_interface_class(cls, awg:Union[Instrument, type]):

        for interface in cls._pulsar_interfaces:
            for awg_class in interface.AWG_CLASSES:
                if isinstance(awg, awg_class) or awg == awg_class:
                    return interface

        raise ValueError("Could not find a suitable pulsar AWG interface for "
                        f"{awg=}.")

    @abstractmethod
    def create_awg_parameters(self, channel_name_map:dict):
        """Create parameters in the pulsar specific to the added AWG.

        Subclasses must override this and make sure to call
        :meth:`create_channel_parameters` for each added channel.

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

        pulsar.add_parameter(f"{name}_active",
                             initial_value=True,
                             vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_reuse_waveforms",
                             initial_value=True, vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_minimize_sequencer_memory",
                             initial_value=False, vals=vals.Bool(),
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
        pulsar.add_parameter(f"{name}_compensation_pulse_min_length",
                             initial_value=0, unit='s',
                             parameter_class=ManualParameter)

    @abstractmethod
    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """Create parameters in the pulsar specific to one added channel.

        Arguments:
            id: Channel id. Typically ``ch{index}`` where index is a 1-based
                channel index. For marker channels, usually ``ch{ch_nr + 1}m``.
            ch_name: Name of the channel to address it in rest of the codebase.
            ch_type: Type of channel: ``"analog"`` or ``"marker"``.
        """

        # Sanity check
        if ch_type not in ["analog", "marker"]:
            raise ValueError(f"Invalid {ch_type=}.")

        pulsar = self.pulsar
        awg = self.awg

        pulsar.add_parameter(f"{ch_name}_id", get_cmd=lambda: id)
        pulsar.add_parameter(f"{ch_name}_awg", get_cmd=lambda: awg.name)
        pulsar.add_parameter(f"{ch_name}_type", get_cmd=lambda: ch_type)
        if self._check_if_implemented(id, "amp"):
            pulsar.add_parameter(f"{ch_name}_amp",
                                 label=f"{ch_name} amplitude", unit='V',
                                 set_cmd=partial(self.awg_setter, id, "amp"),
                                 get_cmd=partial(self.awg_getter, id, "amp"),
                                 vals=vals.Numbers(
                                     *self.CHANNEL_AMPLITUDE_BOUNDS[ch_type]))
        if self._check_if_implemented(id, "offset"):
            pulsar.add_parameter(f"{ch_name}_offset", unit='V',
                                 set_cmd=partial(self.awg_setter, id, "offset"),
                                 get_cmd=partial(self.awg_getter, id, "offset"),
                                 vals=vals.Numbers(
                                     *self.CHANNEL_OFFSET_BOUNDS[ch_type]))
        if self._check_if_implemented(id, "range"):
            pulsar.add_parameter(f"{ch_name}_range",
                                 label=f"{ch_name} output range", unit='dBm',
                                 set_cmd=partial(self.awg_setter, id, "range"),
                                 get_cmd=partial(self.awg_getter, id, "range"),
                                 vals=vals.Numbers(
                                     *self.CHANNEL_RANGE_BOUNDS[ch_type]))
        if self._check_if_implemented(id, "centerfreq"):
            pulsar.add_parameter(f"{ch_name}_centerfreq",
                                 label=f"{ch_name} center frequency", unit='Hz',
                                 set_cmd=partial(self.awg_setter, id, "centerfreq"),
                                 get_cmd=partial(self.awg_getter, id, "centerfreq"),
                                 vals=vals.Numbers(
                                     *self.CHANNEL_CENTERFREQ_BOUNDS[ch_type]))

        if ch_type == "analog":
            pulsar.add_parameter(f"{ch_name}_distortion",
                                 label=f"{ch_name} distortion mode",
                                 initial_value="off",
                                 vals=vals.Enum("off", "precalculate"),
                                 parameter_class=ManualParameter)
            pulsar.add_parameter(f"{ch_name}_distortion_dict",
                                 vals=vals.Dict(),
                                 parameter_class=ManualParameter)
            pulsar.add_parameter(f"{ch_name}_charge_buildup_compensation",
                                 parameter_class=ManualParameter,
                                 vals=vals.Bool(), initial_value=False)
            pulsar.add_parameter(f"{ch_name}_compensation_pulse_scale",
                                 parameter_class=ManualParameter,
                                 vals=vals.Numbers(0., 1.), initial_value=0.5)
            pulsar.add_parameter(f"{ch_name}_compensation_pulse_delay",
                                 initial_value=0, unit='s',
                                 parameter_class=ManualParameter)
            pulsar.add_parameter(
                f"{ch_name}_compensation_pulse_gaussian_filter_sigma",
                initial_value=0, unit='s', parameter_class=ManualParameter)

        else: # ch_type == "marker"
            # So far no additional parameters specific to marker channels
            pass

    @abstractmethod
    def awg_getter(self, id:str, param:str):
        """Helper function to get AWG parameters.

        Arguments:
            id: Channel id for which to get the parameter value.
            param: Parameter to get.
        """

        if self._check_if_implemented(id, param):
            raise NotImplementedError(f"Unknown parameter '{param}'.")

    @abstractmethod
    def awg_setter(self, id:str, param:str, value):
        """Helper function to set AWG parameters.

        Arguments:
            id: Channel id for which to set the parameter value.
            param: Parameter to set.
            value: Value to set the parameter.
        """

        if self._check_if_implemented(id, param):
            raise NotImplementedError(f"Unknown parameter '{param}'.")

    @abstractmethod
    def program_awg(self, awg_sequence: dict, waveforms: dict,
                    repeat_pattern=None,
                    channels_to_upload: Union[List[str], str] = "all",
                    channels_to_program: Union[List[str], str] = "all"):
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
            repeat_pattern: TODO: Document. Currently only used in UHFQC.
            channels_to_upload: list of channel names to upload or ``"all"``.
            channels_to_program: List of channel to program. Only relevant for
                the HDAWG.
        """

    def _program_awg(self, awg_sequence:dict, waveforms:dict,
                     repeat_pattern=None,
                     channels_to_upload:Union[List[str], str]="all",
                     channels_to_program:Union[List[str], str]="all",
                     filter_segments=None):
        """Preprocess filter segments before programming actual hardware"""
        awg_sequence = self.get_filtered_awg_sequence(
            awg_sequence, waveforms, filter_segments, repeat_pattern,
            channels_to_upload=channels_to_upload,
            channels_to_program=channels_to_program
        )
        if awg_sequence is not None:
            self.program_awg(
                awg_sequence=awg_sequence,
                waveforms=waveforms,
                repeat_pattern=repeat_pattern,
                channels_to_upload=channels_to_upload,
                channels_to_program=channels_to_program
            )

    def get_filtered_awg_sequence(self, awg_sequence, waveforms,
                                  filter_segments, repeat_pattern, **kwargs):
        # store data for filter segments emulation if needed
        if len(self.get_segment_filter_userregs(include_inactive=True)) == 0:
            # AWG does not support segment filtering in hardware
            use_filter = any([e is not None and
                              e.get('metadata', {}).get('allow_filter', False)
                              for e in awg_sequence.values()])
            if use_filter:
                # filter segments emulation needed
                if repeat_pattern is not None:
                    raise NotImplementedError(
                        f'{self.awg.name} does not support filter_segments and '
                        f'an emulation is needed, but the combination of '
                        f'filter_segments emulation and repeat_pattern is not '
                        f'implemented.')

                if filter_segments is None:
                    # _program_awg was called from self._program_awgs
                    # (otherwise, it would have been called from
                    # self._set_filter_segments)
                    self._filter_segments_emulation_cache = {
                        'awg_sequence': awg_sequence,
                        'waveforms': waveforms,
                        **kwargs,
                    }
                    # We skip the programming, and it will be done in the
                    # next update of filter_segments.
                    # FIXME: This assumes that filter_segments will be set
                    #  after the call to program_awgs (as it is the case in
                    #  FilteredSweep). Find a more general solution.
                    self.pulsar._awgs_with_waveforms.add(self.awg.name)
                    return
                new_awg_sequence = odict()
                i_seg = -1
                for k, v in awg_sequence.items():
                    if v is None:
                        i_seg += 1
                    else:
                        metadata = v.get('metadata', {})
                        if metadata.get('allow_filter', False) and (
                                i_seg < filter_segments[0] or
                                i_seg > filter_segments[1]):
                            continue
                    new_awg_sequence[k] = deepcopy(v)
                self._filter_segments_emulation_cache['filter_segments'] = \
                    filter_segments
                return new_awg_sequence
        # If we are here, it means that the current sequence does not use
        # segment filtering or that no emulation is needed for this AWG.
        # Reset the _filter_segments_emulation_cache for this AWG.
        self._filter_segments_emulation_cache = None
        return awg_sequence

    def start(self):
        """Start the AWG."""

        self.awg.start()

    def stop(self):
        """Stop the AWG."""

        self.awg.stop()

    @abstractmethod
    def is_awg_running(self) -> bool:
        """Checks whether the sequencer of the AWG is running."""

    @abstractmethod
    def clock(self) -> float:
        """Returns the sample clock frequency [Hz] of the AWG."""

    @abstractmethod
    def sigout_on(self, ch, on:bool=True):
        """Turn channel outputs on or off."""

    def get_segment_filter_userregs(self, include_inactive=False) \
            -> List[Tuple[str, str]]:
        """Returns the list of segment filter userregs.

        Returns:
            List of tuples ``(first, last)`` where ``first`` and ``last`` are
            formatted strings. TODO: More accurate description.
        """
        return []

    def set_filter_segments(self, val):
        """TODO: Document"""

        if self._filter_segment_functions is None:
            all_regs = self.get_segment_filter_userregs()
            if len(all_regs) == 0:
                # Emulate filter segments for this AWG
                fsec = self._filter_segments_emulation_cache
                if fsec is not None and fsec.get('filter_segments', None) \
                        != val:
                    self._program_awg(
                        **{k: v for k, v in fsec.items()
                           if k != 'filter_segments'},
                        filter_segments=val,
                    )
            else:  # filter segments supported by the hardware for this AWG
                for regs in all_regs:
                    self.awg.set(regs[0], val[0])
                    self.awg.set(regs[1], val[1])
        else:
            # used in case of a repeat pattern
            for regs in self.get_segment_filter_userregs():
                self.awg.set(regs[1], self._filter_segment_functions(val[0],
                                                                     val[1]))

    def _check_if_implemented(self, id:str, param:str):
        if param in self.IMPLEMENTED_ACCESSORS and (
            not isinstance(self.IMPLEMENTED_ACCESSORS, dict)
            or id in self.IMPLEMENTED_ACCESSORS[param]
        ):
            return True
        return False


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
        awgs: Names of AWGs added to the pulsar.
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
        self.add_parameter('trigger_pulse_parameters', initial_value={},
                           label='trigger pulse parameters',
                           parameter_class=ManualParameter,
                           docstring='A dict whose keys are channel names and '
                                     'whose values are dicts of pulse '
                                     'parameters to overwrite the default '
                                     'trigger pulse parameters whenever a '
                                     'trigger pulse is played on the '
                                     'respective channel. In addition, the '
                                     'dict can contain keys of the form '
                                     '{channel}_first to provide different '
                                     'parameters for the first trigger pulse '
                                     'on that channel in a sequence.')


        self._inter_element_spacing = 'auto'
        self.channels = set() # channel names
        self.awgs:Set[str] = set() # AWG names
        self.awg_interfaces:Dict[str, PulsarAWGInterface] = {}
        self.last_sequence = None
        self.last_elements = None
        self._awgs_with_waveforms = set()
        self.channel_groups = {}
        self.num_channel_groups = {}

        self._awgs_prequeried_state = False

        self._hash_to_wavename_table = {}
        self._filter_segments = None

        Pulsar._instance = self

    # TODO: Should Pulsar be a singleton ? Is it really necessary to have such
    # a method ?
    @classmethod
    def get_instance(cls):
        return cls._instance

    @staticmethod
    def _get_pulsar_id_file():
        return os.path.join(gen.get_pycqed_appdata_dir(), "pulsar_id")

    def _use_sequence_cache_parser(self, val):
        if val and not self.use_sequence_cache():
            self.reset_sequence_cache()
        return val

    def reset_sequence_cache(self):
        """Resets the sequence cache.

        The nested dict ``_sequence_cache`` is used in ``_program_awgs`` to
        store information about the last sequence programmed to each AWGs. The
        keys of the dicts are AWG names and/or channel names. See the code and
        comments of _program_awgs for details about the structure of the dicts.
        """

        self._sequence_cache = {}
        self._sequence_cache['settings'] = {}  # for pulsar settings
        self._sequence_cache['metadata'] = {}  # for segment/element metadata
        self._sequence_cache['hashes'] = {}  # for waveform hashes
        self._sequence_cache['length'] = {}  # for element lengths

    def check_for_other_pulsar(self) -> bool:
        """Returns true if another pulsar has programmed the AWGs.

        To make this check possible, the pulsar object ID is written to a file
        in the pycqed app data dir, see :meth:`_get_pulsar_id_file` and
        :meth:`_write_pulsar_check_file`.
        """

        try:
            with open(self._get_pulsar_id_file(), 'r') as f:
                stored_id = int(f.read())
        except FileNotFoundError:
            return True

        return stored_id != id(self)

    def _write_pulsar_check_file(self):
        """Set this pulsar as the last one to have programmed the AWGs."""

        with open(self._get_pulsar_id_file(), 'w') as f:
            f.write(str(id(self)))

    def define_awg_channels(self, awg:Instrument, channel_name_map:dict=None):
        """Add an AWG with a channel mapping to the pulsar.

        Args:
            awg: AWG object to add to the pulsar.
            channel_name_map: Optional dictionary that maps channel ids to
                channel names.
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

        # Add awg and channels parameters to pulsar
        awg_interface_class = PulsarAWGInterface.get_interface_class(awg)
        awg_interface = awg_interface_class(self, awg)
        awg_interface.create_awg_parameters(channel_name_map)
        self.awg_interfaces[awg.name] = awg_interface

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

    def get_channel_awg(self, channel:str) -> Instrument:
        """Return the AWG corresponding to a channel.

        Args:
            channel: Name of the channel.
        """

        return Instrument.find_instrument(self.get(f"{channel}_awg"))

    def clock(self, channel:str=None, awg:str=None):
        """Returns the clock rate of channel or AWG.

        Arguments:
            channel: name of the channel.
            awg: name of AWG.

        Returns:
            Clock rate in samples per second.
        """

        if channel is not None and awg is not None:
            raise ValueError('Both channel and awg arguments passed to '
                             'Pulsar.clock()')
        if channel is None and awg is None:
            raise ValueError('Neither channel nor awg arguments passed to '
                             'Pulsar.clock()')

        if channel is not None:
            awg = self.get('{}_awg'.format(channel))

        if self.awgs_prequeried:
            return self._clocks[awg]
        else:
            return self.awg_interfaces[awg].clock()

    def active_awgs(self) -> Set[str]:
        """Get the set of names of registered active AWGs.

        Inactive AWGs don't get started or stopped, and their waveforms don't
        get updated.
        """

        return {awg for awg in self.awgs if self.get('{}_active'.format(awg))}

    def add_awg_with_waveforms(self, awg:str):
        """Adds an awg to the set of AWGs with waveforms programmed."""

        if awg not in self.awgs:
            raise ValueError(f"'{awg=}' is not registered in pulsar.")

        self._awgs_with_waveforms.add(awg)
        self._set_filter_segments(self._filter_segments, [awg])

    def start(self, exclude:List[str]=None, stop_first:bool=True):
        """Start the active AWGs.

        If multiple AWGs are used in a setup where the
        slave AWGs are triggered by the master AWG, then the slave AWGs must be
        running and waiting for trigger when the master AWG is started to
        ensure synchronous playback.

        TODO: master_awg does not need to be a pulsar interface. The PQSC
        instrument is currently used on some setups. It would make sense to also
        implement a pulsar interface for the PQSC, such that one can use the
        master AWG similarly to any other AWG in the pulsar.

        Arguments:
            exclude: Names of AWGs to exclude
            stop_first: Whether all used AWGs should be stopped before starting
                the AWGs.
        """

        if exclude is None:
            exclude = []

        # Start only the AWGs which have at least one channel programmed, i.e.
        # where at least one channel has state = 1.
        used_awg_names = self.active_awgs() & self._awgs_with_waveforms
        used_awgs = [self.awg_interfaces[name] for name in used_awg_names]

        if stop_first:
            for awg in used_awgs:
                awg.stop()

        # Exclude AWGs that should not be started
        used_awgs = [awg for awg in used_awgs if awg.awg.name not in exclude]

        # Stop master AWG
        if self.master_awg() and self.master_awg() not in exclude:
            self.master_awg.get_instr().stop()

        # Start slave AWGs
        for awg in used_awgs:
            if awg.awg.name != self.master_awg():
                awg.start()

        # Check that all slave AWGs start within 10s
        awgs_to_check = [awg for awg in used_awgs
                         if awg.awg.name != self.master_awg()]
        try:
            with WatchdogTimer(10) as timer:
                while len(awgs_to_check):
                    timer.check()
                    awgs_to_check = [awg for awg in awgs_to_check
                                     if not awg.is_awg_running()]
                    time.sleep(0.1)
        except WatchdogException:
            raise WatchdogException(f"AWGs {awgs_to_check} did not start in 10s.")

        # Start master AWG
        if self.master_awg() not in exclude:
            self.master_awg.get_instr().start()

    def stop(self):
        """Stop all active AWGs."""

        used_awgs = set(self.active_awgs()) & self._awgs_with_waveforms
        used_awg_interfaces = [self.awg_interfaces[name] for name in used_awgs]

        for awg in used_awg_interfaces:
            awg.stop()

    def sigout_on(self, ch, on:bool=True):
        """Turn channel outputs on or off."""

        awg = self.find_instrument(self.get(ch + '_awg'))
        self.awg_interfaces[awg.name].sigout_on(ch, on)

    def program_awgs(self, sequence, awgs:Union[List[str], str]="all"):
        """Program the AWGs to play a sequence.

        Arguments:
            sequence: See
                :class:`pycqed.measurement.waveform_control.sequence.Sequence`.
            awgs: List of AWGs names, or ``"all"``
        """

        try:
            self._program_awgs(sequence, awgs)
        except Exception as e:
            if not self.use_sequence_cache():
                raise
            log.warning(f'Pulsar: Exception {repr(e)} while programming AWGs. '
                        f'Retrying after resetting the sequence cache.')
            self.reset_sequence_cache()
            self._program_awgs(sequence, awgs)

    def _program_awgs(self, sequence, awgs:Union[List[str], str]='all'):

        # Stores the last uploaded sequence for easy access and plotting
        self.last_sequence = sequence

        if awgs == 'all':
            awgs = self.active_awgs()

        # initializes the set of AWGs with waveforms
        self._awgs_with_waveforms -= awgs


        # Setting the property will prequery all AWG clock and amplitudes
        self.awgs_prequeried = True

        log.info(f'Starting compilation of sequence {sequence.name}')
        t0 = time.time()
        if self.use_sequence_cache():
            self.invalid_cache_if_other_pulsar()
            
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

        # TODO: Check if this could be done somewhere else, such that there is
        # no need to import ZIPulsarMixin in this module.
        ZIPulsarMixin.zi_waves_cleared = False
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
                self.add_awg_with_waveforms(awg)
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

            self.awg_interfaces[awg]._program_awg(
                awg_sequences.get(awg, {}),
                waveforms,
                repeat_dict.get(awg, None),
                channels_to_upload=ch_upl,
                channels_to_program=ch_prg,
            )

            log.info(f'Finished programming {awg} in {time.time() - t0}')

        if self.use_sequence_cache():
            # Compilation finished sucessfully. Store sequence cache.
            self._sequence_cache = sequence_cache

        # Reset prequery state
        self.awgs_prequeried = False

    def invalid_cache_if_other_pulsar(self):
        if self.check_for_other_pulsar():
            log.debug("Another pulsar instance has programmed the AWGs. "
                          "Resetting sequence cache.")
            self.reset_sequence_cache()
            self._write_pulsar_check_file()

    def _hash_to_wavename(self, h):

        def to_base(n, b, alphabet=None, prev=None):
            if prev is None: prev = []
            if n == 0:
                if alphabet is None: return prev
                else: return [alphabet[i] for i in prev]
            return to_base(n//b, b, alphabet, prev+[n%b])

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

    def _set_inter_element_spacing(self, val):
        self._inter_element_spacing = val

    def _get_inter_element_spacing(self):
        if self._inter_element_spacing != 'auto':
            return self._inter_element_spacing
        elif not self.awgs:
            return 0
        else:
            return max([self.get(f"{awg}_inter_element_deadtime")
                        for awg in self.awgs])

    def _set_filter_segments(self, val:Tuple[int, int]=None,
                             awgs='with_waveforms'):
        if val is None:
            val = _DEFAULT_SEGMENT

        self._filter_segments = val
        if awgs == 'with_waveforms':
            awgs = self._awgs_with_waveforms
        elif awgs == 'all':
            awgs = self.awgs
        for awg in awgs:
            self.awg_interfaces[awg].set_filter_segments(val)

    def _get_filter_segments(self):
        return self._filter_segments

    @property
    def awgs_prequeried(self, status=None) -> bool:
        return self._awgs_prequeried_state

    @awgs_prequeried.setter
    def awgs_prequeried(self, status:bool):
        if status:
            self._awgs_prequeried_state = False
            self._clocks = {}
            for awg in self.awgs:
                self._clocks[awg] = self.awg_interfaces[awg].clock()
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
    def _channels_in_awg_sequences(awg_sequences) -> Dict[str, Set[str]]:
        """Identifies all channels used in the given awg keyed sequence.

        Arguments:
            awg_sequences (dict): awg sequences keyed by awg name, i.e. as
                returned by ``sequence.generate_sequence_waveforms()``.

        Returns:
            dictionary keyed by awg of with all channels used during the
            sequence.
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
        """Translates a repeat dictionary keyed by channels to a repeat
        dictionary keyed by awg.

        Checks whether all channels in channels_used have an entry.

        Arguments:
            repeat_dict_per_ch: keys: channels_id, values: repeat pattern.
            channels_used (dict): list of channel used on each awg.
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
