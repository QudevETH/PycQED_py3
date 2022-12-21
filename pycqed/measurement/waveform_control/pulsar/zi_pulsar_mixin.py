import ctypes
import logging
import os
import shutil
import numpy as np
from copy import deepcopy
from typing import Optional, List


log = logging.getLogger(__name__)

try:
    from pycqed.measurement.waveform_control.pulsar.zi_multi_core_compiler. \
        multi_core_compiler import MultiCoreCompiler
except ImportError:
    log.debug('Could not import MultiCoreCompiler, parallel programming of ZI devices will not work.')
    class MultiCoreCompiler():
        def __init__(self):
            self._awgs = {}

class ZIPulsarMixin:
    """Mixin containing utility functions needed by ZI AWG pulsar interfaces.

    Classes deriving from this mixin must have a ``pulsar`` attribute.
    """

    _status_ZIPulsarMixin = dict(zi_waves_clean=False)
    """Dict keeping the status of the class

    key zi_waves_clean: Flag indicating whether the waves dir is clean."""

    @staticmethod
    def _zi_wave_dir():
        if os.name == "nt":
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                _basedir = buf.value
            else:
                log.warning("Could not extract my documents folder")
        else:
            _basedir = os.path.expanduser("~")
        wave_dir = os.path.join(_basedir, "Zurich Instruments", "LabOne",
            "WebServer", "awg", "waves")
        if not os.path.exists(wave_dir):
            os.makedirs(wave_dir)
        return wave_dir

    @classmethod
    def zi_waves_clean(cls, val=None):
        """Get or set whether the waves dir is clean

        Args:
            val (bool or None): if not None, sets the flag zi_waves_clean in
                _status_ZIPulsarMixin to val.

        Returns:
            The current value of the flag zi_waves_clean in
            _status_ZIPulsarMixin if val is None, and None otherwise.
        """
        if val is None:
            return cls._status_ZIPulsarMixin['zi_waves_clean']
        else:
            cls._status_ZIPulsarMixin['zi_waves_clean'] = val

    @classmethod
    def _zi_clear_waves(cls):
        wave_dir = cls._zi_wave_dir()
        for f in os.listdir(wave_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(wave_dir, f))
            elif f.endswith(".cache"):
                shutil.rmtree(os.path.join(wave_dir, f))
        cls.zi_waves_clean(True)

    @staticmethod
    def _zi_wavename_pair_to_argument(w1, w2, internal_mod=False):
        if w1 is not None and w2 is not None:
            if not internal_mod:
                return f"{w1}, {w2}"
            else:
                # This syntax is needed to allow IQ-mixing for
                # imaginary-valued baseband signal.
                return f"1, 2, {w1}, 1, 2, {w2}"
        elif w1 is not None and w2 is None:
            return f"1, {w1}"
        elif w1 is None and w2 is not None:
            return f"2, {w2}"
        else:
            return ""

    def zi_wave_definition(
            self,
            wave,
            defined_waves=None,
            wave_index=None,
            placeholder_wave_length=None,
            internal_mod=False,
    ):
        """ Generate wave definition for the sequence code.

        Args:

             wave: wave to define in the sequence code

             defined_waves: waves that already have definitions. Newly
             defined waves will be added to this dictionary

             wave_index (Optional, int): if specified, assignWaveIndex() will
             be used to assign this wave with index wave_index. This index
             can be used later for binary upload or sequencing with
             command tables.

             placeholder_wave_length (Optional, int): length to reserve for
             the placeholder wave. If not specified, placeholder wave will
             not be used.
        """

        if defined_waves is None:
            defined_waves = set() if wave_index is None else set(), dict()

        if isinstance(defined_waves, set):
            defined_wave_names = defined_waves
            defined_wave_indices = None
        else:
            defined_wave_names = defined_waves[0]
            defined_wave_indices = defined_waves[1]

        wave_definition = []
        w1, w2 = self.zi_waves_to_wavenames(wave)

        # Add wave name definitions.
        if placeholder_wave_length is None:
            # don"t use placeholder waves
            for analog, marker, wc in [(wave[0], wave[1], w1),
                                       (wave[2], wave[3], w2)]:
                if analog is not None:
                    wa = self.pulsar._hash_to_wavename(analog)
                    if wa not in defined_wave_names:
                        wave_definition.append(f'wave {wa} = "{wa}";')
                        defined_wave_names.add(wa)

                if marker is not None:
                    wm = self.pulsar._hash_to_wavename(marker)
                    if wm not in defined_wave_names:
                        wave_definition.append(f'wave {wm} = "{wm}";')
                        defined_wave_names.add(wm)

                if analog is not None and marker is not None:
                    if wc not in defined_wave_names:
                        wave_definition.append(f"wave {wc} = {wa} + {wm};")
                        defined_wave_names.add(wc)
        else:
            if wave_index is None:
                # wave index has to be specified when using placeholder waves
                raise ValueError(f"wave_index must not be None when "
                                 f"specifying placeholder waves")

            n = placeholder_wave_length
            if w1 is None and w2 is not None:
                w1 = f"{w2}_but_zero"
            for wc, marker in [(w1, wave[1]), (w2, wave[3])]:
                if wc is not None and wc not in defined_wave_names:
                    wave_definition.append(
                        f"wave {wc} = placeholder({n}" +
                        ("" if marker is None else ", true") +
                        ");")
                    defined_waves[0].add(wc)

        # Add wave index definitions.
        if wave_index is not None:
            if wave not in defined_wave_indices.values():
                wave_definition.append(
                    f"assignWaveIndex(" + \
                    self._zi_wavename_pair_to_argument(
                        w1, w2, internal_mod=internal_mod) + \
                    f", {wave_index});"
                )
                defined_wave_indices[wave_index] = wave

        return wave_definition

    @staticmethod
    def zi_playback_string_loop_start(metadata, channels):
        """Creates playback string that starts a loop depending on the metadata.

        The method also takes care of oscillator sweeps.
        Args:
            metadata (dict): Dictionary containing the information if a loop
                should be started and which variables to sweep inside the loop.
                Relevant keys are:
                    loop (int): Length of the loop. If not specified, no loop
                        will be started and the returned playback_string will be
                        an empty list.
                    sweep_params (dict): Dictionary of sweeps. Depending on the
                        key, different sweeps will be performed. Only those
                        items will be implemented, whose keys starts with a
                        channel name contained in channels.
                        Special keys (after "{ch_name}_") are:
                            "osc_sweep"
                        The default behaviour will assume a key describing the
                        path of a node of the channel with "/" replaced by "_".
                        The value should be a list of doubles in this case.
            channels (list): list of channel names to be considered

        Returns:
            str: playback_string
        """
        loop_len = metadata.get("loop", False)
        if not loop_len:
            return []
        playback_string = []
        sweep_params = metadata.get("sweep_params", {})
        for k, v in sweep_params.items():
            for ch in channels:
                if not k.startswith(f"{ch}_"):
                    continue
                if k == f"{ch}_osc_sweep":
                    playback_string.append('//set up frequency sweep')
                    start_freq = v[0]
                    freq_inc = 0 if len(v) <= 1 else v[1] - v[0]
                    playback_string.append(
                        f'configFreqSweep(SWEEP_OSC,{start_freq},{freq_inc});')
                else:
                    playback_string.append(
                        f'wave {k} = vect({",".join([f"{a}" for a in v])})')
        playback_string.append(
            f"for (cvar i_sweep = 0; i_sweep < {loop_len}; i_sweep += 1) {{")
        for k, v in sweep_params.items():
            for ch in channels:
                if not k.startswith(f"{ch}_"):
                    continue
                if k == f"{ch}_osc_sweep":
                    playback_string.append('  waitWave();\n')
                    playback_string.append('  setSweepStep(SWEEP_OSC,'
                                           ' i_sweep);\n')
                else:
                    node = k[len(f"{ch}_"):].replace("_", "/")
                    playback_string.append(
                        f'setDouble("{node}", {k}[i_sweep]);')
        return playback_string

    @staticmethod
    def zi_playback_string_loop_end(metadata):
        return ["}"] if metadata.get("end_loop", False) else []

    def zi_codeword_table_entry(self, codeword, wave, placeholder_wave=False,
                                internal_mod=False):
        w1, w2 = self.zi_waves_to_wavenames(wave)
        use_hack = True
        if w1 is None and w2 is not None and use_hack and not placeholder_wave:
            # This hack is needed due to a bug on the HDAWG.
            # Remove this if case once the bug is fixed.
            return [f"setWaveDIO({codeword}, zeros(1) + marker(1, 0), {w2});"]
        elif w1 is None and w2 is not None and use_hack and placeholder_wave:
            return [f"setWaveDIO({codeword}, {w2}_but_zero, {w2});"]
        elif not (w1 is None and w2 is None):
            return ["setWaveDIO({}, {});".format(codeword,
                        self._zi_wavename_pair_to_argument(
                            w1, w2, internal_mod=internal_mod))]
        else:
            return []

    def zi_waves_to_wavenames(self, wave):
        wavenames = []
        for analog, marker in [(wave[0], wave[1]), (wave[2], wave[3])]:
            if analog is None and marker is None:
                wavenames.append(None)
            elif analog is None and marker is not None:
                wavenames.append(self.pulsar._hash_to_wavename(marker))
            elif analog is not None and marker is None:
                wavenames.append(self.pulsar._hash_to_wavename(analog))
            else:
                wavenames.append(self.pulsar._hash_to_wavename((analog, marker)))
        return wavenames

    def zi_write_waves(self, waveforms):
        wave_dir = self._zi_wave_dir()
        for h, wf in waveforms.items():
            filename = os.path.join(wave_dir, self.pulsar._hash_to_wavename(h)+".csv")
            if os.path.exists(filename):
                # Skip writing the CSV file. This happens if reuse_waveforms
                # is True and the same hash appears on multiple AWG modules.
                # Note that it does not happen in cases where
                # use_sequence_cache is True and the same hash had appeared in
                # earlier experiments (because we clear the waves dir before
                # starting programming the AWGs).
                continue
            fmt = "%.18e" if wf.dtype == np.float else "%d"
            np.savetxt(filename, wf, delimiter=",", fmt=fmt)

    def zi_playback_string(self, name, device, wave, acq=False, codeword=False,
                           prepend_zeros=0, placeholder_wave=False,
                           command_table_index=None,
                           internal_mod=False,
                           allow_filter=False):
        playback_string = []
        if allow_filter:
            playback_string.append(
                "if (i_seg >= first_seg && i_seg <= last_seg) {")
        if prepend_zeros:
            playback_string.append(f"playZero({prepend_zeros});")
        w1, w2 = self.zi_waves_to_wavenames(wave)
        use_hack = True # set this to false once the bugs with HDAWG are fixed
        playback_string += self.zi_wait_trigger(name, device)

        if codeword and not (w1 is None and w2 is None):
            playback_string.append("playWaveDIO();")
        elif command_table_index is not None:
            playback_string.append(f"executeTableEntry({command_table_index});")
        else:
            if w1 is None and w2 is not None and use_hack and not placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f"playWave(marker(1,0)*0*{w2}, {w2});")
            elif w1 is None and w2 is not None and use_hack and placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f"playWave({w2}_but_zero, {w2});")
            elif w1 is not None and w2 is None and not placeholder_wave:
                if device == 'shfsg':
                    # Generate real valued output on SG channel
                    playback_string.append(f"playWave(1, 2, {w1});")
                elif use_hack:
                    # This hack is needed due to a bug on the HDAWG.
                    # Remove this if case once the bug is fixed.
                    playback_string.append(f"playWave({w1}, marker(1,0)*0*{w1});")
            elif w1 is not None or w2 is not None:
                playback_string.append("playWave({});".format(
                    self._zi_wavename_pair_to_argument(
                        w1, w2, internal_mod=internal_mod)))
        if acq:
            playback_string.append("setTrigger(RO_TRIG);")
            playback_string.append("setTrigger(WINT_EN);")
        if allow_filter:
            playback_string.append("}")
        return playback_string

    def zi_wait_trigger(self, name, device):
        playback_string = []
        trig_source = self.pulsar.get("{}_trigger_source".format(name))
        if trig_source == "Dig1":
            playback_string.append(
                "waitDigTrigger(1{});".format(", 1" if device == "uhf" else ""))
        elif trig_source == "Dig2":
            playback_string.append("waitDigTrigger(2,1);")
        else:
            playback_string.append(f"wait{trig_source}Trigger();")
        return playback_string

    def _zi_program_generator_awg(
            self,
            awg_sequence,
            waveforms,
            repeat_pattern=None,
            channels_to_upload="all",
            channels_to_program="all"
    ):

        self.wfms_to_upload = {}  # reset waveform upload memory

        use_placeholder_waves = self.pulsar.get(
            f"{self.awg.name}_use_placeholder_waves")
        if not use_placeholder_waves:
            if not self.zi_waves_clean():
                self._zi_clear_waves()

        has_waveforms = False
        for channel_pair in self.awg_modules:
            upload = channels_to_upload == 'all' or \
                any([ch in channels_to_upload
                     for ch in channel_pair.channel_ids])
            program = channels_to_program == 'all' or \
                any([ch in channels_to_program
                     for ch in channel_pair.channel_ids])
            channel_pair.program_awg_channel(
                awg_sequence=awg_sequence,
                waveforms=waveforms,
                program=program,
                upload=upload
            )
            has_waveforms |= any(channel_pair.has_waveforms)

        if self.pulsar.sigouts_on_after_programming():
            for awg_module in self.awg_modules:
                for channel_id in awg_module.analog_channel_ids:
                    channel_name = self.pulsar._id_channel(
                        cid=channel_id,
                        awg=self.awg.name,
                    )
                    self.sigout_on(channel_name)

        if has_waveforms:
            self.pulsar.add_awg_with_waveforms(self.awg.name)


class MultiCoreCompilerQudevZI(MultiCoreCompiler):
    """
    A child of MultiCoreCompiler, which compiles and uploads sequences using
    multiple threads.
    This class enables compilation on a subset of the added awgs, referred to
    as "active awgs" and stored in self._awgs.

    Args:
        awgs (AWG): A list of the AWG Nodes that are target for parallel
            compilation
        use_tempdir (bool, optional): Use a temporary directory to store the
            generated sequences
    """
    _instances = []
    """A list that records all multi-core compilers instantiated from this 
    class (MultiCoreCompilerQudevZI).
    """

    @classmethod
    def get_instance(cls, session=None):
        """Get a multi-core compiler working on the specified session. For
        details regarding the sessions and ZI data server, please refer to
        the docstring of class zhinst.qcodes.ZISession

        Args:
            session (ZISession): a ZI server client session

        Returns:
            mcc (MultiCoreCompilerQudevZI): a multi-core compiler
                corresponding to the specified session.
        """
        for mcc in cls._instances:
            if mcc._session == session or session is None:
                return mcc
        mcc = cls()
        mcc._session = session
        cls._instances.append(mcc)
        return mcc

    def __init__(self, *awgs, **kwargs):
        super().__init__(*awgs, **kwargs)
        self._awgs_all = {}
        # Move existing AWGs to the full AWG dict self._awgs_all. Remove AWGs
        # in the active AWG dict self._awgs to prevent processing all AWGs in
        # the multi-core compiler. Active AWGs will be added when programming
        # individual AWGs.
        self._awgs_all.update(self._awgs)
        self.reset_active_awgs()

    def add_awg(self, awg):
        """
        Add a AWG core to self._awgs_all and reset self._awgs.

        Args:
            awg (AWG): The AWG Nodes that is target for parallel compilation
        """
        super().add_awg(awg)
        self._awgs_all[awg] = self._awgs[awg]
        self.reset_active_awgs()

    def add_active_awg(self, awg):
        """
        Add a AWG core to the active awgs (self._awgs), which are the ones
        that will be programmed.

        Args:
            awg (AWG): The AWG Nodes that is target for parallel compilation
        """
        self._awgs[awg] = self._awgs_all[awg]

    def reset_active_awgs(self):
        """
        Reset the active awgs (self._awgs)
        """
        self._awgs = {}


class ZIMultiCoreCompilerMixin:
    """
    Mixin for creating an instance of MultiCoreCompilerQudevZI for parallel
    compilation and upload of SeqC strings.
    """
    multi_core_compilers = {}
    _disable_mcc = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wfms_to_upload = {}
        """Stores hashes and waveforms to be uploaded after parallel
        compilation. This attribute is only used when self.pulsar.use_mcc() is
        set to True.
        """

    @property
    def multi_core_compiler(self):
        """Getter of the ZI multi-core compiler. If the compiler does not
        exist, a new one will be created and added to the class attribute
        self.multi_core_compilers to keep track of all existing compilers.

        Returns:
            multi_core_compiler (MultiCoreCompilerQudevZI): ZI multi-core
            compiler for the current instance.
        """
        if self not in self.multi_core_compilers:
            awgs = self.awgs_mcc
            session = awgs[0].parent._tk_object._session if len(awgs) else None
            self.multi_core_compilers[self] = \
                MultiCoreCompilerQudevZI.get_instance(session)
        return self.multi_core_compilers[self]

    def _init_mcc(self):
        if not len(self.awgs_mcc):
            return  # no AWG module supports MCC
        try:
            # add AWG modules to multi_core_compiler class variable
            for awg in self.awgs_mcc:
                self.multi_core_compiler.add_awg(awg)
        except Exception as e:
            log.error(f'Failed to initialize MCC for AWG {self.awg.name}: '
                      f'{e}.')
            self._disable_mcc = True
            return
        # register a finalized callback in the main pulsar
        self.pulsar.mcc_finalize_callbacks[self.awg.name] = \
            self.finalize_upload_after_mcc

    def finalize_upload_after_mcc(self):
        """Finalize the upload after the MCC has finished

        The base method uploads waveforms stored in wfms_to_upload. Child
        classes could implement further actions if needed.

        ZI devices currently only support parallel compilation + upload
        of SeqC strings, so we must upload waveforms separately here
        after parallel upload is finished.
        """
        for k, v in self.wfms_to_upload.items():
            awg_nr, wave_idx = k
            waveforms, wave_hashes = v
            # awg_interface that populate wfms_to_upload must implement the
            # method upload_waveforms
            self.upload_waveforms(
                awg_nr=awg_nr, wave_idx=wave_idx,
                waveforms=waveforms, wave_hashes=wave_hashes)

    def upload_waveforms(self, awg_nr, wave_idx, waveforms, wave_hashes):
        """
        Upload waveforms to an awg core (awg_nr).

        Args:
            awg_nr (int): index of awg core (0, 1, 2, or 3)
            wave_idx (int): index of wave upload (0 or 1)
            waveforms (array): waveforms to upload
            wave_hashes: waveforms hashes
        """
        raise NotImplementedError('Method "upload_waveforms" is not '
                                  'implemented for parent class '
                                  '"ZIMultiCoreCompilerMixin". \n'
                                  'Please rewrite this method in children '
                                  'classes with device-specific '
                                  'implementations.')

    @property
    def awgs_mcc(self) -> list:
        """List of AWG cores that support parallel compilation.

        If self._disable_mcc is set to True, returns empty list.
        """
        if self._disable_mcc:
            return []
        else:
            return self._get_awgs_mcc()

    def _get_awgs_mcc(self) -> list:
        """Returns the list of the AWG cores that support parallel compilation.
        """
        raise NotImplementedError('Method "_get_awgs_mcc" is not '
                                  'implemented for parent class '
                                  '"ZIMultiCoreCompilerMixin". \n'
                                  'Please rewrite this method in children '
                                  'classes with device-specific '
                                  'implementations.')


class ZIGeneratorModule:
    """Interface for ZI drive AWG modules. Each instance of this class saves
    configuration of this module and handles communication with the base
    instrument class when programming the module. Implementation of an AWG
    module varies for different ZI devices. Please refer to the docstring of
    child classes for more information.
    """

    _sequence_string_template = (
        "{wave_definitions}\n"
        "\n"
        "{codeword_table_defs}\n"
        "\n"
        "while (1) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def __init__(
            self,
            awg,
            awg_interface,
            awg_nr: int,
    ):
        self._awg = awg
        """Instrument driver of the parent device."""

        self._device_type = "none"
        """Device type of this generator. This parameter should be rewritten 
        in the child classes."""

        self._awg_interface = awg_interface
        """Pulsar interface of the parent device."""

        self.pulsar = awg_interface.pulsar
        """Pulsar instance which saves global configurations for all devices."""

        self._awg_nr = awg_nr
        """AWG module number of the current instance."""

        self.channel_ids = None
        """A list of all programmable IDs of this AWG channel/channel pair."""

        self.analog_channel_ids = None
        """A list of all analog channel IDs of this AWG channel/channel pair."""

        self.marker_channel_ids = None
        """A list of all marker channel IDs of this AWG channel/channel pair."""

        self.i_channel_name = None
        """Full name of the I channel in this AWG module."""

        self._upload_idx = None
        """Node index on the ZI data server."""

        self.waveform_cache = dict()
        """Dict for storing previously-uploaded waveforms."""

        self._defined_waves = None
        """Waves that has been assigned names on this channel."""

        self._wave_definitions = []
        """Wave definition strings to be added to the sequencer code."""

        self._codeword_table = {}
        """Codeword table for DIO wave triggering."""

        self._codeword_table_defs = []
        """Codeword table definitions to be added to the sequencer code."""

        self._command_table = []
        """Command table for pulse sequencing."""

        self._command_table_lookup = {}
        """Dict that records mapping element, codeword (keys) and command 
        table entry index (value)."""

        self._playback_strings = []
        """Playback strings to be added to the sequencer code."""

        self._wave_idx_lookup = {}
        """Look-up dictionary of the defined waveform indices."""

        self._mod_config = {}
        """Internal modulation configuration."""

        self._sine_config = {}
        """Sinusoidal wave generation configuration."""

        self.has_waveforms = {}
        """Dictionary of flags indicating whether this channel has waveform 
        to play in the current programming round."""

        self._use_placeholder_waves = False
        """Whether to use placeholder waves in the sequencer code."""

        self._use_command_table = False
        """Whether to use command table for pulse sequencing."""

        self._use_filter = False
        # TODO: check if this docstring is correct
        """Whether to use filter programmed to the device."""

        self._use_internal_mod = False
        """Whether to use digital modulation when generating waveforms."""

        self._divisor = {}
        # TODO: check if this docstring is correct
        """A dictionary that records down-sampling ratio (divisor) for each 
        channel ID."""

        self._negate_q = False
        """Whether to flip the sign of the Q channel waveform."""

        self._generate_channel_ids(awg_nr=awg_nr)
        self._reset_has_waveform_flags()

    def _generate_channel_ids(
            self,
            awg_nr,
    ):
        """Generates all programmable ids (e.g. I channel, Q channel, marker
        channels) of this channel/channel pair and update the relevant class
        attributes.
        """
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _generate_divisor(self):
        """Generate dummy divisors, which corresponds to no down-samplig for
        the output waveforms."""
        self._divisor = {chid: 1 for chid in self.channel_ids}

    def _reset_sequence_strings(
            self,
            reset_wave_definition: bool = True,
            reset_codeword_table: bool = True,
            reset_playback_strings: bool = True,
            reset_command_table: bool = True,
    ):
        """Resets everything relates to sequence code strings."""
        if reset_wave_definition:
            self._wave_definitions = []

        if reset_codeword_table:
            self._codeword_table = {}
            self._codeword_table_defs = []

        if reset_command_table:
            self._command_table = []
            self._command_table_lookup = {}

        if reset_playback_strings:
            self._playback_strings = []

    def _reset_has_waveform_flags(self):
        """Resets flags saved in self.has_waveforms."""
        for chid in self.channel_ids:
            self.has_waveforms[chid] = False

    def _reset_defined_waves(self):
        self._defined_waves = (set(), dict()) \
            if self._use_placeholder_waves or self._use_command_table\
            else set()

    def program_awg_channel(
            self,
            awg_sequence,
            waveforms,
            upload,
            program,
    ):
        """ Programs this AWG channel/channel pair.

        Args:
            awg_sequence (Dict): AWG sequence data (not waveforms) as returned
                from ``Sequence.generate_waveforms_sequences``.
            waveforms (Dict): A dictionary of waveforms, keyed by their hash.
            upload (Bool): A boolean value that specifies whether the
                waveforms should be uploaded to the device.
            program (Bool): A boolean value that indicates whether the
                sequencer program should be uploaded to the device.
        """
        # Collects settings from pulsar. Uploads sine wave generation
        # and internal modulation settings to the device.
        self._generate_divisor()
        self._reset_has_waveform_flags()
        self._reset_sequence_strings()
        self._update_channel_config(awg_sequence=awg_sequence)
        self._reset_defined_waves()

        # Generates sequencer code according to channel settings and
        # waveforms specified in awg_sequence.
        self._generate_filter_seq_code()
        self._generate_oscillator_seq_code()
        self._generate_wave_seq_code(
            awg_sequence=awg_sequence,
            waveforms=waveforms,
            upload=upload,
        )

        if not any(self.has_waveforms.values()):
            # If no waveform is assigned to this AWG channel/channel pair,
            # AWG instrument driver will be notified that this
            # channel/channel pair doesn't need to be programmed.
            self._awg._awg_program[self._upload_idx] = None
            return
        self._awg._awg_program[self._upload_idx] = True

        # Instruct AWG instrument driver to start this channel/channel pair.
        self._update_awg_instrument_status()

        if not upload:
            # program_awg_channel was called only to decide which AWG modules
            # are active, and the rest of this loop can be skipped.
            return

        if not self._use_placeholder_waves:
            # Waveform upload in the .csv format should be completed before
            # compiling and uploading the sequencer code.
            self._upload_csv_waveforms(
                waveforms=waveforms,
                awg_sequence=awg_sequence,
            )

        # Compiles and upload sequencer code to the device if required.
        self._compile_awg_program(program=program)

        if self._use_placeholder_waves:
            # Waveform upload in the binary format (i.e. with placeholder
            # waves) should be done after compiling and uploading the
            # sequencer code, because waveform memory will be erased when the
            # device is re-programmed with the new sequencer code.
            self._upload_placeholder_waveforms(waveforms=waveforms)

        if self._use_command_table:
            # Command table should be uploaded after compiling and uploading
            # the sequencer code, because the command table memory will be
            # erased when the device is re-programmed with the new sequencer
            # code.
            self._upload_command_table()

        self._set_signal_output_status()
        if any(self.has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self._awg.name)

    def _update_channel_config(
            self,
            awg_sequence,
    ):
        self._update_i_channel_name()
        self._update_use_placeholder_wave_flag()
        self._update_use_filter_flag(awg_sequence=awg_sequence)
        self._update_use_command_table_flag()
        self._update_use_internal_mod_flag()

        # Resolve and upload internal modulation configurations to the device
        # Note that self._use_internal_mod flag will be False when using
        # hardware sweep for spectroscopy measurements. This is to
        # distinguish internal modulation activated for spectroscopy
        # measurements and for time-domain measurements.
        mod_config = self._resolve_channel_config(
            awg_sequence=awg_sequence,
            config_name="mod_config",
            excluded_keys=('phase') if self._use_internal_mod
            else ('mod_freq', 'mod_phase'),
        )
        self._upload_modulation_config(
            mod_config=mod_config.get(self.i_channel_name, dict()))

        # Resolve and upload sine wave generation configurations to the device
        sine_config = self._resolve_channel_config(
            awg_sequence=awg_sequence,
            config_name="sine_config"
        )
        self._upload_sine_generation_config(
            sine_config=sine_config.get(self.i_channel_name, dict()))

    def _update_i_channel_name(self):
        """Get I channel name from self.pulsar.channels ."""
        self.i_channel_name = self.pulsar._id_channel(
            cid=self.analog_channel_ids[0],
            awg=self._awg.name
        )

    def _update_use_filter_flag(
            self,
            awg_sequence,
    ):
        """Updates self._use_filter flag with the setting specified in
        awg_sequence.

        Args:
            awg_sequence (Dict): AWG sequence data (not waveforms) as returned
                from ``Sequence.generate_waveforms_sequences``.
        """

        self._use_filter = any(
            [e is not None and
             e.get('metadata', {}).get('allow_filter', False)
             for e in awg_sequence.values()]
        )

    def _update_use_command_table_flag(self):
        """Updates self._use_command_table flag with the setting specified
        in pulsar."""
        device_param = f"{self._awg.name}_use_command_table"
        device_value = self.pulsar.get(device_param) \
            if hasattr(self.pulsar, device_param) else False

        channel_param = f"{self.i_channel_name}_use_command_table"
        channel_value = self.pulsar.get(channel_param) \
            if hasattr(self.pulsar, channel_param) else False

        self._use_command_table = device_value | channel_value

    def _update_use_placeholder_wave_flag(self):
        """Updates self._use_placeholder_wave flag with the setting specified
        in pulsar."""
        device_param = f"{self._awg.name}_use_placeholder_waves"
        device_value = self.pulsar.get(device_param) \
            if hasattr(self.pulsar, device_param) else False

        channel_param = f"{self.i_channel_name}_use_placeholder_waves"
        channel_value = self.pulsar.get(channel_param) \
            if hasattr(self.pulsar, channel_param) else False

        self._use_placeholder_waves = device_value | channel_value

    def _update_use_internal_mod_flag(self):
        """Updates self._use_internal_mod flag with the setting specified in
        pulsar."""
        device_param = f"{self._awg.name}_internal_modulation"
        device_value = self.pulsar.get(device_param) \
            if hasattr(self.pulsar, device_param) else False

        channel_param = f"{self.i_channel_name}_internal_modulation"
        channel_value = self.pulsar.get(channel_param) \
            if hasattr(self.pulsar, channel_param) else False

        self._use_internal_mod = device_value | channel_value

    def _resolve_channel_config(
            self,
            awg_sequence,
            config_name: str,
            excluded_keys: tuple = tuple(),
    ):
        """Collects and combines a specific configuration setting in 
        awg_sequence. If settings from different elements are coherent with 
        each other, the combined setting will be programmed to the channel.

        Args:
            awg_sequence: A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
            config_name (str): name of the configuration to resolve.
            excluded_keys (tuple): keys to be excluded from the element
                metadata.

        Returns:
            channel_config (dict): Dict that contains channel-specific
                configurations. It is grouped as
                {"channel_name": modulation_configuration}
        """
        # For the moment we only collect settings on the I channel.
        channel_config = dict()

        # Combine internal modulation configurations from all elements in
        # the sequence into one and check if they are compatible with each other
        first_new_dict = True
        for element in awg_sequence:
            awg_sequence_element = awg_sequence[element]
            # Check if there is a valid sequence element and if this element is
            # played on this channel. If no, then this element is irrevalent and
            # we will not consider its modulation parameters.
            if awg_sequence_element is None or \
                    not self.is_element_on_this_module(awg_sequence_element):
                continue
            metadata = awg_sequence_element.get('metadata', {})
            element_config = metadata.get(config_name, {})
            if not self.diff_and_combine_dicts(
                    element_config,
                    channel_config,
                    excluded_keys=excluded_keys,
                    first_new_dict=first_new_dict,
            ):
                raise RuntimeError(
                    f"On {self._awg.name}: Configuration {config_name} in "
                    f"metadata is incompatible between different elements in "
                    f"the same sequence."
                )
            first_new_dict = False

        return channel_config

    def _upload_modulation_config(
            self,
            mod_config: dict,
    ):
        """Upload modulation configurations to the device.

        Args:
            mod_config (dict): Dict that contains internal modulation configuration
                to be uploaded.
        """
        pass

    def _upload_sine_generation_config(
            self,
            sine_config,
    ):
        """Upload sine generation configurations to the device.

        Args:
            sine_config (dict): Dict that contains sine generation configuration
                to be uploaded.
        """
        pass

    def _update_awg_instrument_status(self):
        pass

    def _check_ignore_waveforms(self):
        """Check if all waveforms on this AWG module are ignored. This method
        will be rewritten in child classes that should take such consideration.
        """
        return False

    def _generate_filter_seq_code(self):
        """Generates sequencer code that is relevant to using filters."""
        if self._use_filter:
            self._playback_strings += ['var i_seg = -1;']
            self._wave_definitions += [
                f'var first_seg = getUserReg'
                f'({self._awg.USER_REG_FIRST_SEGMENT});',
                f'var last_seg = getUserReg'
                f'({self._awg.USER_REG_LAST_SEGMENT});',
            ]

    def _generate_oscillator_seq_code(self):
        """Generates sequencer code that is relevant to oscillator
        operations."""
        pass

    def _generate_wave_seq_code(
            self,
            awg_sequence,
            waveforms,
            upload,
    ):
        """Generates sequencer code that is relevant to wave definition and
        sequencing.

        Args:
            awg_sequence (Dict): AWG sequence data (not waveforms) as returned
                from ``Sequence.generate_waveforms_sequences``.
            waveforms (List): A dictionary of waveforms, keyed by their hash.
            upload (Bool): A boolean value that specifies whether the
                waveforms should be uploaded to the device.
        """
        # resets wave index lookup table and wave index counter. They are
        # used when _use_placeholder_wave is enabled.
        self._wave_idx_lookup = {}
        next_wave_idx = 0

        # Keeps track of the current segment. Some devices require taking
        # care of prepending zeros at the first element of a segment.
        current_segment = 'no_segment'
        first_element_of_segment = True

        for element in awg_sequence:
            awg_sequence_element = deepcopy(awg_sequence[element])
            if awg_sequence_element is None:
                current_segment = element
                self._playback_strings.append(f'// Segment {current_segment}')
                if self._use_filter:
                    self._playback_strings.append('i_seg += 1;')
                first_element_of_segment = True
                continue
            self._command_table_lookup[element] = None
            self._wave_idx_lookup[element] = {}

            metadata = awg_sequence_element.pop('metadata', {})

            # Check if analog channels has overlap with the segment trigger
            # group. If no, this segment will not be played.
            trigger_groups = metadata['trigger_groups']
            channels = set(self.pulsar._id_channel(chid, self._awg.name)
                        for chid in self.analog_channel_ids)
            if not self.pulsar.check_channels_in_trigger_groups(
                    set(channels), trigger_groups):
                continue

            self._playback_strings.append(f'// Element {element}')
            # The following line only has an effect if the metadata
            # specifies that the segment should be repeated multiple times.
            self._playback_strings += \
                ZIPulsarMixin.zi_playback_string_loop_start(
                    metadata,
                    self.channel_ids
                )

            nr_cw = len(set(awg_sequence_element.keys()) - {'no_codeword'})
            if nr_cw == 1:
                log.warning(
                    f'Only one codeword has been set for {element}')
                self._playback_strings += \
                    ZIPulsarMixin.zi_playback_string_loop_end(metadata)
                continue

            for cw in awg_sequence_element:
                if cw == 'no_codeword':
                    if nr_cw != 0:
                        continue
                chid_to_hash = awg_sequence_element[cw]

                # With current implementations, 'wave' has to be a tuple of
                # length 4 in order to be unpacked and processed by other
                # methods. Here we use 'None' as placeholders for undefined
                # waveforms, and rewrites the elements if there is a
                # definition. Note that .channel_ids can be shorter than 4
                # for some devices. Please refer to ._generate_channel_ids
                # methods of child classes.
                wave = [None, None, None, None]
                for i, chid in enumerate(self.channel_ids):
                    wave[i] = chid_to_hash.get(chid, None)
                    # Update self.has_waveforms flag of the corresponding
                    # channel.
                    self.has_waveforms[chid] |= wave[i] is not None

                if not upload:
                    # _program_awg was called only to decide which
                    # AWG modules are active, and the rest of this loop
                    # can be skipped
                    continue

                # Flipping the sign of the Q channel waveform if specified
                if wave[2] is not None and self._negate_q:
                    h_pos = wave[2]
                    h_neg = tuple(list(h_pos) + ['negate'])
                    wave[2] = h_neg
                    waveforms[h_neg] = -waveforms[h_pos]

                wave = tuple(wave)
                # Skip this element if it has no waves defined on this
                # channel/channel pair, or sine config instructs pulsar to
                # ignore waveforms.
                if wave == (None, None, None, None) or \
                        self._check_ignore_waveforms():
                    continue

                # Updates the codeword table if there exists codewords.
                if nr_cw != 0:
                    w1, w2 = self._awg_interface.zi_waves_to_wavenames(wave)
                    if cw not in self._codeword_table:
                        self._codeword_table_defs += \
                            self._awg_interface.zi_codeword_table_entry(
                                cw, wave, self._use_placeholder_waves,
                                internal_mod=self._use_internal_mod
                            )
                        self._codeword_table[cw] = (w1, w2)
                    elif self._codeword_table[cw] != (w1, w2) \
                            and self.pulsar.reuse_waveforms():
                        log.warning('Same codeword used for different '
                                    'waveforms. Using first waveform. '
                                    f'Ignoring element {element}.')

                # Update self.has_waveforms flag of the corresponding channel
                # ID if there are waveforms defined.
                for i, chid in enumerate(self.channel_ids):
                    self.has_waveforms[chid] |= wave[i] is not None

                if not upload:
                    # _program_awg was called only to decide which
                    # AWG modules are active, and the rest of this loop
                    # can be skipped
                    continue

                self._wave_idx_lookup[element][cw] = None
                reuse_definition = False
                if self._use_placeholder_waves or self._use_command_table:
                    # If the wave is already assigned an index, we will point
                    # the wave to the existing index and skip the rest of wave
                    # definition.
                    if wave in self._defined_waves[1].values():
                        self._wave_idx_lookup[element][cw] = [
                            i for i, v in self._defined_waves[1].items()
                            if v == wave][0]
                        reuse_definition = True
                    else:
                        self._wave_idx_lookup[element][cw] = next_wave_idx
                        next_wave_idx += 1

                # Update (and thus activate) command table if specified.
                if self._use_command_table:
                    if cw != 'no_codeword':
                        raise RuntimeError(
                            f"On device: {self._awg.name}: Pulse sequencing "
                            f"with DIO and with command table are turned on "
                            f"at the same time. Please do not use them "
                            f"simultaneously, as they conflicts with each "
                            f"other in the sequencer code. "
                        )

                    scaling_factor = metadata.get("scaling_factor", dict())
                    entry_index = len(self._command_table)
                    amplitude = self._extract_command_table_amplitude(
                        scaling_factor=scaling_factor
                    )
                    phase=metadata.get('mod_config', {})\
                        .get(self.i_channel_name, {}).get("phase", 0)

                    entry = self._generate_command_table_entry(
                        entry_index=entry_index,
                        wave_index=self._wave_idx_lookup[element][cw],
                        amplitude=amplitude,
                        phase=phase,
                    )
                    update_entry = True

                    # Check if the same entry already exists in the command
                    # table. If so, the existing entry will be reused and the
                    # new entry will not be uploaded.
                    for existing_entry in self._command_table:
                        if self._compare_command_table_entry(
                            entry,
                            existing_entry
                        ):
                            entry_index = existing_entry["index"]
                            update_entry = False

                    # records mapping between element-codeword and entry index
                    self._command_table_lookup[element] = entry_index
                    if update_entry:
                        self._command_table.append(entry)

                if self._use_placeholder_waves:
                    # No need to add new definitions when reusing old ones
                    if reuse_definition:
                        continue

                    # Check if the longest placeholder wave length equals to
                    # the shortest one. If not, use the longest wave
                    # length to fit all waveforms.
                    placeholder_wave_lengths = \
                        [waveforms[h].size for h in wave if h is not None]
                    if max(placeholder_wave_lengths) != \
                            min(placeholder_wave_lengths):
                        log.warning(f"Waveforms of unequal length on"
                                    f"{self._awg.name}, vawg{self._awg_nr},"
                                    f" {current_segment}, {element}.")

                    # Add new wave definition and save wave index.
                    self._wave_definitions += \
                        self._awg_interface.zi_wave_definition(
                            wave=wave,
                            defined_waves=self._defined_waves,
                            wave_index=self._wave_idx_lookup[element][cw],
                            placeholder_wave_length=max(placeholder_wave_lengths),
                            internal_mod=self._use_internal_mod,
                        )
                else:
                    # No indices will be assigned when not using placeholder
                    # waves.
                    wave = list(wave)
                    for i, h in enumerate(wave):
                        if h is not None:
                            wave[i] = self._with_divisor(h, self.channel_ids[i])
                    wave = tuple(wave)

                    self._wave_definitions += \
                        self._awg_interface.zi_wave_definition(
                            wave=wave,
                            wave_index=self._wave_idx_lookup[element][cw] if
                            self._use_command_table else None,
                            defined_waves=self._defined_waves,
                            internal_mod=self._use_internal_mod,
                        )

            if not upload:
                # _program_awg was called only to decide which AWG modules are
                # active, and the rest of this loop can be skipped.
                continue

            # Add the playback string for the current wave.
            if not self._check_ignore_waveforms():
                self._generate_playback_string(
                    wave=wave,
                    codeword=(nr_cw != 0),
                    use_placeholder_waves=self._use_placeholder_waves,
                    command_table_index=self._command_table_lookup[element],
                    metadata=metadata,
                    first_element_of_segment=first_element_of_segment,
                )
                first_element_of_segment = False
            else:
                self._playback_strings += self._awg_interface.zi_wait_trigger(
                        name=self._awg.name,
                        device=self._device_type,
                    )

            self._playback_strings += \
                ZIPulsarMixin.zi_playback_string_loop_end(metadata)

    def _generate_playback_string(
            self,
            wave,
            codeword,
            use_placeholder_waves,
            command_table_index,
            metadata,
            first_element_of_segment,
    ):
        """Generates a playback string for the given wave and settings.

        Args:
            wave (Tuple): A tuple of four that includes waveform hashes of
                all programmable IDs.
            codeword (Bool): A boolean value that specifies whether to
                play waveforms conditioned on codewords.
            use_placeholder_waves (Bool): A boolean value that specifies
                whether placeholder waves are enabled.
            metadata (Dict): A dict that specifies various device
                configurations of the current element.
            first_element_of_segment (Bool): A boolean value that indicates
                whether this element is the first one in this segment.
        """
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _upload_csv_waveforms(
            self,
            waveforms,
            awg_sequence,
    ):
        """Saves waveforms in .csv files and uploads them to ZI data server.
        This method is used when placeholder waves are disabled.

        Args:
            waveforms (Dict): A dictionary of waveforms, keyed by their hash.
            awg_sequence (Dict): AWG sequence data (not waveforms) as returned
                from ``Sequence.generate_waveforms_sequences``.
        """
        waves_to_upload = {}

        for codewords in awg_sequence.values():
            if codewords is not None:
                for cw, chids in codewords.items():
                    if cw != 'metadata':
                        for chid, h in chids.items():
                            if chid[-1] == 'i' or not self._negate_q:
                                waves_to_upload[h] = waveforms[h]
                            else:
                                h_neg = tuple(list(h) + ['negate'])
                                waves_to_upload[h_neg] = -waveforms[h]

        self._awg_interface.zi_write_waves(waves_to_upload)

    def _generate_command_table_entry(
            self,
            entry_index: int,
            wave_index: int,
            amplitude: float = None,
            phase: float = 0.0,
    ):
        """returns a command table entry in the format specified
        by ZI. Details of the command table can be found in
        https://docs.zhinst.com/shfqc_user_manual/tutorials/tutorial_command_table.html

        Arguments:
            entry_index (int): index of the current command table entry
            wave_index (int): index of the reference wave
            amplitude (Optional, array-like): an array of 4 recording the
                amplitudes specified in the command table. They are grouped in
                the order (amplitude00, amplitude01, amplitude10, amplitude11).
                If not specified, this array is set to (1, -1, 1, 1)
            phase (Optional, float): phase of the waveform. Default is 0.0.

        Returns:
            command_table (Dict):
        """
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    @staticmethod
    def _compare_command_table_entry(
            entry1: dict,
            entry2: dict,
    ):
        """compares if two command table entries equal except for the index

        Arguments:
            entry1 (dict): a dictionary that represents a command table entry.
            entry2 (dict):  a dictionary that represents a command table entry.

        Return:
            equal (bool): if the two entries are equal, except from their
                indices.
        """

        entry1_copy = deepcopy(entry1)
        entry2_copy = deepcopy(entry2)

        entry1_copy["index"] = 0
        entry2_copy["index"] = 0

        return entry1_copy == entry2_copy

    def _extract_command_table_amplitude(
            self,
            scaling_factor,
    ):
        """Get command table 'amplitude' entry from scaling factors
        passed from element metadata.

        Args:
            scaling_factor: metadata passed from awg_sequence that specifies
                the amplitude ratio between the actual wave and the uploaded
                wave.
        """
        if scaling_factor == dict():
            return 1.0

        # check if channels specified in scaling_factor match the
        # analog channels on this AWG module.
        channel_ids = [self.pulsar.get(f"{channel}_id")
                       for channel in scaling_factor.keys()]
        if set(channel_ids) != set(self.analog_channel_ids):
            raise RuntimeError(
                f"Channels specified in 'scaling_factor' metadata does not "
                f"match analog channels on {self.awg.name} generator module "
                f"{self._awg_nr}."
            )

        # check if factors specified in scaling_factor are all equal.
        # TODO: support different scaling factors for different output
        #  channels in this AWG module.
        if len(set(scaling_factor.values())) > 1:
            raise RuntimeError(
                f"Scaling factors on {self.awg.name} generator module"
                f"{self._awg_nr} are defined differently among output channels."
            )

        return list(scaling_factor.values())[0]

    def _compile_awg_program(
            self,
            program,
    ):
        """Compiles the sequencer code and programs the device.

        Args:
            program (Bool): a boolean value that specifies whether we want to
                re-programs this channel/channel pair irrespective of the
                necessity. If set to False, this channel/channel pair will
                only be programmed when the previous waveforms cannot be reused.
        """
        awg_str = self._sequence_string_template.format(
            wave_definitions='\n'.join(self._wave_definitions),
            codeword_table_defs='\n'.join(self._codeword_table_defs),
            playback_string='\n  '.join(self._playback_strings),
        )

        if not self._use_placeholder_waves or program:
            run_compiler = True
        else:
            cached_lookup = self.waveform_cache.get(
                f'wave_idx_lookup', None)
            try:
                np.testing.assert_equal(self._wave_idx_lookup, cached_lookup)
                run_compiler = False
            except AssertionError:
                log.debug(f'{self._awg.name}_{self._awg_nr}: Waveform reuse '
                          f'pattern has changed. Forcing recompilation.')
                run_compiler = True

        if run_compiler:
            self._execute_compiler(awg_str=awg_str)
            if self._use_placeholder_waves:
                self.waveform_cache = dict()
                self.waveform_cache['wave_idx_lookup'] = self._wave_idx_lookup

    def _upload_placeholder_waveforms(
            self,
            waveforms,
    ):
        """Uploads binary waveforms via ZI Python API. This method is used
        when placeholder waves are enabled.

        Args:
            waveforms (Dict): A dictionary of waveforms, keyed by their hash.
        """
        for idx, wave_hashes in self._defined_waves[1].items():
            self._update_waveforms(
                wave_idx=idx,
                wave_hashes=wave_hashes,
                waveforms=waveforms,
            )

    def _update_waveforms(
            self,
            wave_idx,
            wave_hashes,
            waveforms
    ):
        """upload waveforms with Zurich Instrument API to the specified AWG
        module.s

        Arguments:
            wave_idx (int): index assigned to the upload wave.
            wave_hashes (tuple): tuple in groups of four. The elements are
                organized in order (analog_i, marker, analog_q, None).
            waveforms (dict): an complete dictionary of waveforms, specified
                sample-wise and indexed by their hash values.

        Returns:
            None
        """
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _execute_compiler(
            self,
            awg_str,
    ):
        try:
            prev_dio_valid_polarity = self._awg.get(
                'awgs_{}_dio_valid_polarity'.format(self._awg_nr))
        except KeyError:
            prev_dio_valid_polarity = None

        if self.pulsar.use_mcc() and len(self._awg_interface.awgs_mcc) > 0:
            # Parallel seqc string compilation and upload
            self._awg_interface.multi_core_compiler.add_active_awg(
                self._awg_interface.awgs_mcc[self._awg_nr])
            self._awg_interface.multi_core_compiler.load_sequencer_program(
                self._awg_interface.awgs_mcc[self._awg_nr], awg_str)
        else:
            if self.pulsar.use_mcc():
                log.warning(
                    f'Parallel elf compilation not supported for '
                    f'{self._awg.name} ({self._awg.devname}), see debug '
                    f'log when adding the AWG to pulsar.')
            self._save_awg_str(awg_str=awg_str)
            self._configure_awg_str(awg_str=awg_str)

        if prev_dio_valid_polarity is not None:
            self._awg.set('awgs_{}_dio_valid_polarity'.format(self._awg_nr),
                          prev_dio_valid_polarity)

    def _configure_awg_str(
            self,
            awg_str,
    ):
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _save_awg_str(
            self,
            awg_str,
    ):
        # This method will be rewritten for the devices that requires
        # explicitly saving AWG sequencer strings in the attribute
        # self._awg._awg_source_strings.
        pass

    def _upload_command_table(self):
        """ write a list of specified command tables to the device

        Returns:
            upload_status (int): status of the upload.
            - 0: command table data clear
            - 1: command table successfully uploaded
            - 8: uploading of data to the command table failed due to a JSON
                 parsing error.

            See node documentation
            https://docs.zhinst.com/hdawg_user_manual/node_definitions.html or
            https://docs.zhinst.com/shfqc_user_manual/nodedoc.html
            for more details.
        """
        raise NotImplementedError("This method should be rewritten in "
                                  "children classes.")

    def _set_signal_output_status(self):
        """Turns on the output of this channel/channel pair if specified by
        pulsar."""
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _with_divisor(self, h, ch):
        return h if self._divisor[ch] == 1 else (h, self._divisor[ch])

    def get_i_channel(self):
        """Helper function that returns I channel name of this AWG module."""
        return self.pulsar._id_channel(
            cid=self.analog_channel_ids[0],
            awg=self._awg.name
        )

    @property
    def awg(self):
        """Returns AWG instrument driver of this channel."""
        return self._awg

    def diff_and_combine_dicts(
            self,
            new,
            combined,
            excluded_keys=tuple(),
            first_new_dict: bool = False,
    ):
        """Recursively adds entries in dict new to the combined dict and
        checks if the values on the lowest level are the same in combined
        and new.

        Args:
            new (dict): Dict with new values that will be added to the
                combined dict after testing if the values for existing keys
                match with the ones in combined.
            combined (dict): Dict to which the items in new will be added.
            excluded_keys (list[str], optional): List of dict keys (str)
                that will not be added to combined and not be tested to have
                the same values in new and combined. Defaults to tuple().
            first_new_dict (bool): Bool that indicates whether this is the first
                new dict to be combined. If True, combined dict (which is assumed to
                be empty) will copy all entries from the new dict.

        Returns:
            compatible (bool): Whether all values for all keys in new (except
                excluded_keys) that already existed in combined are the
                same for new and combined.
        """
        if not (isinstance(new, dict) and isinstance(combined, dict)):
            if new != combined:
                return False
            else:
                return True

        if first_new_dict:
            combined.update(new)
            return True

        # TODO: see if this is compatible with Kilian's spectroscopy
        #  measurements. To be more specific -- if some elements in spectroscopy
        #  measurements are left with empty modulation configurations. They do
        #  not play a role in configuring the device, but will result in an
        #  error when comparing the dicts.
        if new.keys() != combined.keys():
            return False

        for key in new.keys():
            if key in excluded_keys:
                # we do not care if this is the same in all dicts
                continue
            if not self.diff_and_combine_dicts(
                    new[key],
                    combined[key],
                    excluded_keys,
                    first_new_dict=False,
            ):
                return False
        return True

    def is_element_on_this_module(
            self,
            awg_sequence_element,
    ):
        """Test whether an element is played on this module.

        Args:
            awg_sequence_element: element information stored in
                awg_sequence data.

        Returns:
            on_this_module (bool): whether this element is played on this AWG
                module
        """
        for cw in awg_sequence_element:
            chid_to_hash = awg_sequence_element[cw]
            wave = [None, None, None, None]
            for i, chid in enumerate(self.channel_ids):
                wave[i] = chid_to_hash.get(chid, None)
            if any(wave):
                return True
        return False
