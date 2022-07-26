import ctypes
import logging
import os
import shutil
import numpy as np


log = logging.getLogger(__name__)

try:
    from pycqed.measurement.waveform_control.pulsar.zi_multi_core_compiler. \
        multi_core_compiler import MultiCoreCompiler
except ImportError:
    log.warning('Could not import MultiCoreCompiler, parallel programming of ZI devices will not work.')
    class MultiCoreCompiler():
        def __init__(self):
            self._awgs = {}

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
        loop_len = metadata.get("loop", False)
        if not loop_len:
            return []
        playback_string = []
        sweep_params = metadata.get("sweep_params", {})
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f"{ch}_"):
                    playback_string.append(
                        f'wave {k} = vect({",".join([f"{a}" for a in v])})')
        playback_string.append(
            f"for (cvar i_sweep = 0; i_sweep < {loop_len}; i_sweep += 1) {{")
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f"{ch}_"):
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
                            allow_filter=False, negate_q=False):
        playback_string = []
        if allow_filter:
            playback_string.append(
                "if (i_seg >= first_seg && i_seg <= last_seg) {")
        if prepend_zeros:
            playback_string.append(f"playZero({prepend_zeros});")
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if w2 is not None and negate_q:
            w2 = f"(-({w2}))"
        use_hack = True # set this to false once the bugs with HDAWG are fixed
        trig_source = self.pulsar.get("{}_trigger_source".format(name))
        if trig_source == "Dig1":
            playback_string.append(
                "waitDigTrigger(1{});".format(", 1" if device == "uhf" else ""))
        elif trig_source == "Dig2":
            playback_string.append("waitDigTrigger(2,1);")
        else:
            playback_string.append(f"wait{trig_source}Trigger();")

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
            elif w1 is not None and w2 is None and use_hack and not placeholder_wave:
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

        trig_source = self.pulsar.get("{}_trigger_source".format(name))
        if trig_source == "Dig1":
            playback_string.append(
                "waitDigTrigger(1{});".format(", 1" if device == "uhf" else ""))
        elif trig_source == "Dig2":
            playback_string.append("waitDigTrigger(2,1);")
        else:
            playback_string.append(f"wait{trig_source}Trigger();")

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
    def __init__(self, *awgs, **kwargs):
        super().__init__(*awgs, **kwargs)
        self._awgs_all = {}
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
    multi_core_compiler = MultiCoreCompilerQudevZI()
