import ctypes
import logging
import os
import shutil
import numpy as np


log = logging.getLogger(__name__)


class ZIPulsarMixin:
    """Mixin containing utility functions needed by ZI AWG pulsar interfaces.

    Classes deriving from this mixin must have a ``pulsar`` attribute.
    """

    zi_waves_cleared = False
    """Flag set when waves are cleared."""


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
    def _zi_clear_waves(cls):
        wave_dir = cls._zi_wave_dir()
        for f in os.listdir(wave_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(wave_dir, f))
            elif f.endswith(".cache"):
                shutil.rmtree(os.path.join(wave_dir, f))
        cls.zi_waves_cleared = True

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
