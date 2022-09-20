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
    def _zi_wavename_pair_to_argument(w1, w2):
        if w1 is not None and w2 is not None:
            return f"{w1}, {w2}"
        elif w1 is not None and w2 is None:
            return f"1, {w1}"
        elif w1 is None and w2 is not None:
            return f"2, {w2}"
        else:
            return ""

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
            # don"t use placeholder waves
            for analog, marker, wc in [(wave[0], wave[1], w1),
                                       (wave[2], wave[3], w2)]:
                if analog is not None:
                    wa = self.pulsar._hash_to_wavename(analog)
                    if wa not in defined_waves:
                        wave_definition.append(f'wave {wa} = "{wa}";')
                        defined_waves.add(wa)
                if marker is not None:
                    wm = self.pulsar._hash_to_wavename(marker)
                    if wm not in defined_waves:
                        wave_definition.append(f'wave {wm} = "{wm}";')
                        defined_waves.add(wm)
                if analog is not None and marker is not None:
                    if wc not in defined_waves:
                        wave_definition.append(f"wave {wc} = {wa} + {wm};")
                        defined_waves.add(wc)
        else:
            # use placeholder waves
            n = placeholder_wave_length
            if w1 is None and w2 is not None:
                w1 = f"{w2}_but_zero"
            for wc, marker in [(w1, wave[1]), (w2, wave[3])]:
                if wc is not None and wc not in defined_waves[0]:
                    wave_definition.append(
                        f"wave {wc} = placeholder({n}" +
                        ("" if marker is None else ", true") +
                        ");")
                    defined_waves[0].add(wc)
            wave_definition.append(
                f"assignWaveIndex({self._zi_wavename_pair_to_argument(w1, w2)},"
                f" {placeholder_wave_index});"
            )
            defined_waves[1][placeholder_wave_index] = wave
        return wave_definition

    @staticmethod
    def _zi_playback_string_loop_start(metadata, channels):
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
    def _zi_playback_string_loop_end(metadata):
        return ["}"] if metadata.get("end_loop", False) else []

    def _zi_codeword_table_entry(self, codeword, wave, placeholder_wave=False):
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True
        if w1 is None and w2 is not None and use_hack and not placeholder_wave:
            # This hack is needed due to a bug on the HDAWG.
            # Remove this if case once the bug is fixed.
            return [f"setWaveDIO({codeword}, zeros(1) + marker(1, 0), {w2});"]
        elif w1 is None and w2 is not None and use_hack and placeholder_wave:
            return [f"setWaveDIO({codeword}, {w2}_but_zero, {w2});"]
        elif not (w1 is None and w2 is None):
            return ["setWaveDIO({}, {});".format(codeword,
                        self._zi_wavename_pair_to_argument(w1, w2))]
        else:
            return []

    def _zi_waves_to_wavenames(self, wave):
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

    def _zi_write_waves(self, waveforms):
        wave_dir = self._zi_wave_dir()
        for h, wf in waveforms.items():
            filename = os.path.join(wave_dir, self.pulsar._hash_to_wavename(h)+".csv")
            if os.path.exists(filename):
                # Skip writing the CSV file. This happens if reuse_waveforms
                # is True and the same hash appears on multiple sub-awgs.
                # Note that it does not happen in cases where
                # use_sequence_cache is True and the same hash had appeared in
                # earlier experiments (because we clear the waves dir before
                # starting programming the AWGs).
                continue
            fmt = "%.18e" if wf.dtype == np.float else "%d"
            np.savetxt(filename, wf, delimiter=",", fmt=fmt)

    def _zi_playback_string(self, name, device, wave, acq=False, codeword=False,
                            prepend_zeros=0, placeholder_wave=False,
                            allow_filter=False):
        playback_string = []
        if allow_filter:
            playback_string.append(
                "if (i_seg >= first_seg && i_seg <= last_seg) {")
        if prepend_zeros:
            playback_string.append(f"playZero({prepend_zeros});")
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True # set this to false once the bugs with HDAWG are fixed
        playback_string += self._zi_wait_trigger(name, device)

        if codeword and not (w1 is None and w2 is None):
            playback_string.append("playWaveDIO();")
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
                    self._zi_wavename_pair_to_argument(w1, w2)))
        if acq:
            playback_string.append("setTrigger(RO_TRIG);")
            playback_string.append("setTrigger(WINT_EN);")
        if allow_filter:
            playback_string.append("}")
        return playback_string

    def _zi_interleaved_playback_string(self, name, device, counter,
                                        wave, acq=False, codeword=False):
        playback_string = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if w1 is None or w2 is None:
            raise ValueError("When using HDAWG modulation both I and Q need "
                              "to be defined")

        wname = f"wave{counter}"
        interleaves = [f"wave {wname} = interleave({w1}, {w2});"]

        if not codeword:
            if not acq:
                playback_string.append(f"prefetch({wname},{wname});")

        playback_string += self._zi_wait_trigger(name, device)

        if codeword:
            # playback_string.append("playWaveDIO();")
            raise NotImplementedError("Modulation in combination with codeword"
                                      "pulses has not yet been implemented!")
        else:
            playback_string.append(f"playWave({wname},{wname});")
        if acq:
            playback_string.append("setTrigger(RO_TRIG);")
            playback_string.append("setTrigger(WINT_EN);")
        return playback_string, interleaves

    def _zi_wait_trigger(self, name, device):
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
            return  # no sub-AWG supports MCC
        try:
            # add sub-AWGs to multi_core_compiler class variable
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


class ZIDriveAWGChannel:
    """Interface for ZI drive AWG channels/channel pairs. Each instance of
    this class saves configuration of this channel and handles communication
    with the base instrument class when programming the channel.
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
        """ZI-API Driver of the parent device."""

        self._awg_interface = awg_interface
        """Pulsar interface of the parent device."""

        self.pulsar = awg_interface.pulsar
        """A copy of pulsar instance for the current AWG channel."""

        self._awg_nr = awg_nr
        """AWG channel/channel pair number of the current instance."""

        self.channel_ids = None
        """A list of all programmable IDs of this AWG channel/channel pair."""

        self.analog_channel_ids = None
        """A list of all analog channel IDs of this AWG channel/channel pair."""

        self.marker_channel_ids = None
        """A list of all marker channel IDs of this AWG channel/channel pair."""

        self._upload_idx = None
        """Node index on the ZI data server."""

        self._defined_waves = None
        """Waves that has been assigned names on this channel."""

        self._wave_definitions = []
        """Wave definition strings to be added to the sequencer code."""

        self._codeword_table = {}
        """Codeword table for DIO wave triggering."""

        self._codeword_table_defs = []
        """Codeword table definitions to be added to the sequencer code."""

        self._playback_strings = []
        """Playback strings to be added to the sequencer code."""

        self._counter = 1
        """Counter for interleaved playback string."""

        self._interleaves = []
        """Interleaved playback string to be added to the sequencer code."""

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

        self._use_filter = False
        # TODO: check if this docstring is correct
        """Whether to use filter programmed to the device"""

        self._divisor = {}
        # TODO: check if this docstring is correct
        """A dictionary that records down-sampling ratio (divisor) for each 
        channel ID."""

        self._generate_channel_ids(awg_nr=awg_nr)
        self._generate_divisor()
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
        self._divisor = {chid: 1 for chid in self.channel_ids}

    def _reset_sequence_strings(
            self,
            reset_wave_definition: bool = True,
            reset_codeword_table: bool = True,
            reset_playback_strings: bool = True,
            reset_interleaves: bool = True,
            reset_counter: bool = True,
    ):
        """Resets everything relates to sequence code strings."""
        if reset_wave_definition:
            self._wave_definitions = []

        if reset_codeword_table:
            self._codeword_table = {}
            self._codeword_table_defs = []

        if reset_playback_strings:
            self._playback_strings = []

        if reset_interleaves:
            self._interleaves = []

        if reset_counter:
            self._counter = 1

    def _reset_has_waveform_flags(self):
        """Resets flags saved in self.has_waveforms."""
        for chid in self.channel_ids:
            self.has_waveforms[chid] = False

    def _reset_defined_waves(self):
        """Resets defined waves memory."""
        self._defined_waves = (set(), dict()) \
            if self._use_placeholder_waves \
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
            awg_sequence (List): A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
            waveforms (Dict): A dictionary of waveforms, keyed by their hash.
            upload (Bool): A boolean value that specifies whether the
                waveforms should be uploaded to the device.
            program (Bool): A boolean value that indicates whether the
                sequencer program should be uploaded to the device.
        """
        # Collects settings from pulsar. Uploads sine wave generation
        # and internal modulation settings to the device.
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
            self._awg._awg_program[self._awg_nr] = None
            return

        # Instruct AWG instrument driver to start this channel/channel pair.
        self._update_awg_instrument_status()

        if not upload:
            # program_awg_channel was called only to decide which sub-AWGs
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
        self._set_signal_output_status()
        if any(self.has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self._awg.name)

    def _update_channel_config(
            self,
            awg_sequence,
    ):
        self._update_use_placeholder_wave_flag()
        self._update_use_filter_flag(awg_sequence=awg_sequence)
        self._update_internal_mod_config(awg_sequence=awg_sequence)
        self._update_sine_generation_config(awg_sequence=awg_sequence)

    def _update_use_filter_flag(
            self,
            awg_sequence,
    ):
        """Updates self._use_filter flag with the setting specified in
        awg_sequence."""

        self._use_filter = any(
            [e is not None and
             e.get('metadata', {}).get('allow_filter', False)
             for e in awg_sequence.values()]
        )

    def _update_use_placeholder_wave_flag(self):
        """Updates self._use_placeholder_wave flag with the setting specified
        in pulsar."""
        self._use_placeholder_waves = self.pulsar.get(
            f"{self._awg.name}_use_placeholder_waves")

    def _update_internal_mod_config(
            self,
            awg_sequence,
    ):
        """Updates internal modulation configuration according to the settings
        specified in awg_sequence.

        Args:
            awg_sequence (List): A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
        """
        pass

    def _update_sine_generation_config(
            self,
            awg_sequence,
    ):
        """Updates sine wave generation according to the metadata specified
        in awg_sequence.

        Args:
            awg_sequence (List): A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
        """
        pass

    def _update_awg_instrument_status(self):
        # tell ZI_base_instrument that it should not compile a program on
        # this sub AWG (because we already do it here)
        self._awg._awg_needs_configuration[self._awg_nr] = False
        # tell ZI_base_instrument.start() to start this sub AWG (The base
        # class will start sub AWGs for which _awg_program is not None. Since
        # we set _awg_needs_configuration to False, we do not need to put the
        # actual program here, but anything different from None is sufficient.)
        self._awg._awg_program[self._awg_nr] = True

    def _generate_filter_seq_code(self):
        """Generates sequencer code that is relevant to using filters."""
        if self._use_filter:
            self._playback_strings += ['var i_seg = -1;']
            self._wave_definitions += [
                f'var first_seg = getUserReg({self._awg.USER_REG_FIRST_SEGMENT});',
                f'var last_seg = getUserReg({self._awg.USER_REG_LAST_SEGMENT});',
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
            awg_sequence (List): A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
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
            self._wave_idx_lookup[element] = {}
            self._playback_strings.append(f'// Element {element}')

            metadata = awg_sequence_element.pop('metadata', {})
            # The following line only has an effect if the metadata
            # specifies that the segment should be repeated multiple times.
            self._playback_strings += \
                ZIPulsarMixin._zi_playback_string_loop_start(
                    metadata,
                    self.channel_ids
                )

            nr_cw = len(set(awg_sequence_element.keys()) - \
                        {'no_codeword'})
            if nr_cw == 1:
                log.warning(
                    f'Only one codeword has been set for {element}')
                self._playback_strings += \
                    ZIPulsarMixin._zi_playback_string_loop_end(metadata)
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
                wave = tuple(wave)

                # Skip this element if it has no waves defined on this
                # channel/channel pair.
                if wave == (None, None, None, None):
                    continue

                # Updates the codeword table if there exists codewords.
                if nr_cw != 0:
                    w1, w2 = self._awg_interface._zi_waves_to_wavenames(wave)
                    if cw not in self._codeword_table:
                        self._codeword_table_defs += \
                            self._awg_interface._zi_codeword_table_entry(
                                cw, wave, self._use_placeholder_waves)
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
                    # sub-AWGs are active, and the rest of this loop
                    # can be skipped
                    continue

                if self._use_placeholder_waves:
                    # If the wave is already assigned an index, we will point
                    # the wave to the existing index and skip the rest of wave
                    # definition.
                    if wave in self._defined_waves[1].values():
                        self._wave_idx_lookup[element][cw] = [
                            i for i, v in self._defined_waves[1].items()
                            if v == wave][0]
                        continue
                    self._wave_idx_lookup[element][cw] = next_wave_idx
                    next_wave_idx += 1

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
                        self._awg_interface._zi_wave_definition(
                            wave=wave,
                            defined_waves=self._defined_waves,
                            placeholder_wave_index=self._wave_idx_lookup[element][cw],
                            placeholder_wave_length=max(placeholder_wave_lengths),
                        )
                else:
                    # No indices will be assigned when not using placeholder
                    # waves.
                    wave = tuple(
                        self._with_divisor(h, chid)
                        if h is not None else None
                        for h, chid in zip(wave, self.channel_ids)
                    )
                    self._wave_definitions += \
                        self._awg_interface._zi_wave_definition(
                            wave=wave,
                            defined_waves=self._defined_waves,
                        )

            if not upload:
                # _program_awg was called only to decide which sub-AWGs are
                # active, and the rest of this loop can be skipped.
                continue

            # Add the playback string for the current wave.
            self._generate_playback_string(
                wave=wave,
                codeword=(nr_cw != 0),
                use_placeholder_waves=self._use_placeholder_waves,
                metadata=metadata,
                first_element_of_segment=first_element_of_segment,
            )
            first_element_of_segment = False

            self._playback_strings += \
                ZIPulsarMixin._zi_playback_string_loop_end(metadata)

    def _generate_playback_string(
            self,
            wave,
            codeword,
            use_placeholder_waves,
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
            awg_sequence (List): A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
        """
        waves_to_upload = {
            self._with_divisor(h, chid):
                self._divisor[chid] * waveforms[h][::self._divisor[chid]]
            for codewords in awg_sequence.values()
            if codewords is not None
            for cw, chids in codewords.items()
            if cw != 'metadata'
            for chid, h in chids.items()}
        self._awg_interface._zi_write_waves(waves_to_upload)

    def _compile_awg_program(
            self,
            program
    ):
        """Compiles the sequencer code and programs the device.

        Args:
            program (Bool): a boolean value that specifies whether we want to
                re-programs this channel/channel pair irrespective of the
                necessity. If set to False, this channel/channel pair will
                only be programmed when the previous waveforms cannot be reused.
        """
        awg_str = self._sequence_string_template.format(
            wave_definitions='\n'.join(self._wave_definitions + self._interleaves),
            codeword_table_defs='\n'.join(self._codeword_table_defs),
            playback_string='\n  '.join(self._playback_strings),
        )

        if not self._use_placeholder_waves or program:
            run_compiler = True
        else:
            cached_lookup = self._awg_interface._hdawg_waveform_cache.get(
                f'{self._awg.name}_{self._awg_nr}_wave_idx_lookup', None)
            try:
                np.testing.assert_equal(self._wave_idx_lookup, cached_lookup)
                run_compiler = False
            except AssertionError:
                log.debug(f'{self._awg.name}_{self._awg_nr}: Waveform reuse '
                          f'pattern has changed. Forcing recompilation.')
                run_compiler = True

        if run_compiler:
            # We have to retrieve the following parameter to set it
            # again after programming the AWG.
            prev_dio_valid_polarity = self._awg.get(
                'awgs_{}_dio_valid_polarity'.format(self._awg_nr))

            self._awg.configure_awg_from_string(
                awg_nr=self._awg_nr,
                program_string=awg_str,
                timeout=600
            )

            self._awg.set('awgs_{}_dio_valid_polarity'.format(self._awg_nr),
                         prev_dio_valid_polarity)
            if self._use_placeholder_waves:
                self._awg_interface._hdawg_waveform_cache[
                    f'{self._awg.name}_{self._awg_nr}'] = {}
                self._awg_interface._hdawg_waveform_cache[
                    f'{self._awg.name}_{self._awg_nr}_wave_idx_lookup'] = \
                    self._wave_idx_lookup

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
            self._awg_interface._update_waveforms(
                awg_nr=self._awg_nr,
                wave_idx=idx,
                wave_hashes=wave_hashes,
                waveforms=waveforms,
            )

    def _set_signal_output_status(self):
        """Turns on the output of this channel/channel pair if specified by
        pulsar."""
        raise NotImplementedError("This method should be rewritten in child "
                                  "classes.")

    def _with_divisor(self, h, ch):
        return h if self._divisor[ch] == 1 else (h, self._divisor[ch])


def diff_and_combine_dicts(new, combined, excluded_keys=tuple()):
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

    Returns:
        bool: Whether all values for all keys in new (except
            excluded_keys) that already excisted in combined are the
            same for new and combined.
    """
    if not (isinstance(new, dict) and isinstance(combined, dict)):
        if new != combined:
            return False
        else:
            return True
    for key in new.keys():
        if key in excluded_keys:
            # we do not care if this is the same in all dicts
            continue
        if key in combined.keys():
            if not diff_and_combine_dicts(new[key], combined[key],
                                          excluded_keys):
                return False
        else:
            combined[key] = new[key]
    return True

