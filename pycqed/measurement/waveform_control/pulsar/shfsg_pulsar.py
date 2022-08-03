import logging
from typing import List, Tuple

import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.math import vp_to_dbm, dbm_to_vp
from .zi_pulsar_mixin import ZIPulsarMixin
from .pulsar import PulsarAWGInterface

from pycqed.measurement import sweep_functions as swf
import zhinst

try:
    import zhinst.toolkit
    from zhinst.qcodes import SHFSG as SHFSG_core
except Exception:
    SHFSG_core = type(None)


log = logging.getLogger(__name__)


class SHFGeneratorModulePulsar(PulsarAWGInterface, ZIPulsarMixin):
    """ZI SHFSG and SHFQC signal generator module support for the Pulsar class.

    Supports :class:`pycqed.measurement.waveform_control.segment.Segment`
    objects with the following values for acquisition_mode: 'default'
    """

    AWG_CLASSES = []
    GRANULARITY = 16
    ELEMENT_START_GRANULARITY = 16 / 2.0e9  # TODO: unverified!
    MIN_LENGTH = 32 / 2.0e9
    INTER_ELEMENT_DEADTIME = 0  # TODO: unverified!
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.001, 1),
    }
    CHANNEL_RANGE_BOUNDS = {
        "analog": (-40, 10),
    }
    CHANNEL_RANGE_DIVISOR = 5
    CHANNEL_CENTERFREQ_BOUNDS = {
        "analog": (1e9, 8.0e9),
    }
    IMPLEMENTED_ACCESSORS = ["amp", "centerfreq"]

    _shfsg_sequence_string_template = (
        "{wave_definitions}\n"
        "\n"
        "{codeword_table_defs}\n"
        "\n"
        "while (1) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shfsg_waveform_cache = dict()
        self._sgchannel_sine_enable = [False] * len(self.awg.sgchannels)
        """Determines the sgchannels for which the sine generator is turned on
        (off) in :meth:`start` (:meth:`stop`).
        """

    def _create_all_channel_parameters(self, channel_name_map: dict):
        # real and imaginary part of the wave form channel groups
        for ch_nr in range(len(self.awg.sgchannels)):
            group = []
            for q in ["i", "q"]:
                id = f"sg{ch_nr + 1}{q}"
                ch_name = channel_name_map.get(id, f"{self.awg.name}_{id}")
                self.create_channel_parameters(id, ch_name, "analog")
                self.pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                self.pulsar.channel_groups.update({ch_name: group})
        # FIXME: add support for marker channels

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFSG, valid channel ids are sg#i and sg#q, where # is a number
        from 1 to 8. This defines the harware port used.
        """

        PulsarAWGInterface.create_channel_parameters(self, id, ch_name, ch_type)

        if id[-1] == 'i':
            param_name = f"{ch_name}_direct_mod_freq"
            self.pulsar.add_parameter(
                param_name,
                unit='Hz',
                initial_value=None,
                set_cmd=self._direct_mod_setter(ch_name),
                get_cmd=self._direct_mod_getter(ch_name),
                docstring="Configure and turn on direct output of a sine tone "
                          "of the specified frequency. If set to None the sine "
                          "generators will be turned off."
            )
            # qcodes will not set the initial value if it is None, so we set
            # it manually here to ensure that internal modulation gets
            # switched off in the init.
            self.pulsar.set(param_name, None)

            param_name = f"{ch_name}_direct_output_amp"
            self.pulsar.add_parameter(
                param_name,
                unit='V',
                initial_value=1,
                set_cmd=self._direct_mod_amplitude_setter(ch_name),
                get_cmd=self._direct_mod_amplitude_getter(ch_name),
                docstring="Configure and turn on direct output of a sine tone "
                          "of the specified frequency. If set to None the sine "
                          "generators will be turned off."
            )

        # TODO: Not all AWGs provide an initial value. Should it be the case?
        self.pulsar[f"{ch_name}_amp"].set(1)

    def awg_setter(self, id:str, param:str, value):
        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "amp":
            self.awg.sgchannels[ch].output.range(vp_to_dbm(value))
        if param == "centerfreq":
            new_center_freq = self.awg.synthesizers[
                self.awg.sgchannels[ch].synthesizer()].centerfreq(value,
                                                                  deep=True)
            if np.abs(new_center_freq - value) > 1:
                log.warning(f'{self.name}: center frequency {value/1e6:.6f} '
                            f'MHz not supported. Setting center frequency to '
                            f'{new_center_freq/1e6:.6f} MHz. This does NOT '
                            f'automatically set the IF!')

    def awg_getter(self, id:str, param:str):
        # Sanity checks
        super().awg_getter(id, param)

        ch = int(id[2]) - 1

        if param == "amp":
            if self.pulsar.awgs_prequeried:
                dbm = self.awg.sgchannels[ch].output.range.get_latest()
            else:
                dbm = self.awg.sgchannels[ch].output.range()
            return dbm_to_vp(dbm)
        if param == "centerfreq":
            return self.awg.synthesizers[
                self.awg.sgchannels[ch].synthesizer()].centerfreq()

    # FIXME: clean up and move func. common with HDAWG to zi_pulsar_mixin module
    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        chids = [f'sg{i + 1}{iq}' for i in range(len(self.awg.sgchannels))
                 for iq in ['i', 'q']]

        ch_has_waveforms = {chid: False for chid in chids}
        use_placeholder_waves = self.pulsar\
            .get(f"{self.awg.name}_use_placeholder_waves")
        if not use_placeholder_waves:
            if not self.zi_waves_cleared:
                self._zi_clear_waves()

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

        # Combine the sine and internal modulation config from all elements in
        # the sequence into one and check if they are compatible with each other
        channels = [self.pulsar._id_channel(chid, self.awg.name)
                    for chid in chids]
        combined_mod_config = {ch: {} for ch in channels}
        combined_sine_config = {ch: {} for ch in channels}
        for element in awg_sequence:
            awg_sequence_element = awg_sequence[element]
            if awg_sequence_element is None:
                continue
            metadata = awg_sequence_element.get('metadata', {})
            mod_config = metadata.get('mod_config', {})
            sine_config = metadata.get('sine_config', {})
            if not diff_and_combine_dicts(mod_config, combined_mod_config,
                    excluded_keys=['mod_freq', 'mod_phase']):
                raise Exception('Modulation config in metadata is incompatible'
                                'between different elements in same sequence.')
            if not diff_and_combine_dicts(sine_config, combined_sine_config):
                raise Exception('Sine config in metadata is incompatible'
                                'between different elements in same sequence.')

        # Configure internal modulation for each channel. For the SG modules we
        # take config of the I channel and ignore the Q channel configuration
        for ch, config in combined_mod_config.items():
            if ch.endswith('q'):
                continue
            self.configure_internal_mod(ch,
                enable=config.get('internal_mod', False),
                osc_index=config.get('osc', 0),
                sine_generator_index=config.get('sine', 0),
                gains=config.get('gains', (1.0, - 1.0, 1.0, 1.0)))

        # Configure sine output for each channel. For the SG modules we
        # take config of the I channel and ignore the Q channel configuration
        for ch, config in combined_sine_config.items():
            if ch.endswith('q'):
                continue
            self.configure_sine_generation(ch,
                enable=config.get('continuous', False),
                osc_index=config.get('osc', 0),
                sine_generator_index=config.get('sine', 0),
                gains=config.get('gains', (0.0, 1.0, 1.0, 0.0)))

        first_sg_awg = len(getattr(self.awg, 'qachannels', []))
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            defined_waves = (set(), dict()) if use_placeholder_waves else set()
            codeword_table = {}
            wave_definitions = []
            codeword_table_defs = []
            playback_strings = []
            interleaves = []
            # use the modulation config of the I channel
            mod_config = combined_mod_config.get(
                self.pulsar._id_channel(f'sg{awg_nr + 1}i', self.awg.name), {})
            sine_config = combined_sine_config.get(
                self.pulsar._id_channel(f'sg{awg_nr + 1}i', self.awg.name), {})

            internal_mod = False

            use_filter = any([e is not None and
                              e.get('metadata', {}).get('allow_filter', False)
                              for e in awg_sequence.values()])
            if use_filter:
                playback_strings += ['var i_seg = -1;']
                wave_definitions += [
                    f'var first_seg = getUserReg('
                    f'{self.awg.USER_REG_FIRST_SEGMENT});',
                    f'var last_seg = getUserReg('
                    f'{self.awg.USER_REG_LAST_SEGMENT});',
                ]

            ch1id = f'sg{awg_nr+1}i'
            ch2id = f'sg{awg_nr+1}q'
            chids = [ch1id, ch2id]
            channels = [self.pulsar._id_channel(chid, self.awg.name)
                        for chid in [ch1id, ch2id]]

            

            if mod_config.get('internal_mod', False) \
                    or sine_config.get('continuous', False):
                # Reset the starting phase of all oscillators at the beginning
                # of a sequence using the resetOscPhase instruction. This
                # ensures that the carrier-envelope offset, and thus the final
                # output signal, is identical from one repetition to the next.
                playback_strings.append(f'resetOscPhase();\n')
                osc_id = str(mod_config.get('osc', '0')) \
                    if mod_config.get('internal_mod', False) \
                    else str(sine_config.get('osc', '0'))
                playback_strings.append(f'const SWEEP_OSC = {osc_id};\n')

            counter = 1
            next_wave_idx = 0
            wave_idx_lookup = {}
            current_segment = 'no_segment'
            first_element_of_segment = True
            for element in awg_sequence:
                awg_sequence_element = deepcopy(awg_sequence[element])
                if awg_sequence_element is None:
                    current_segment = element
                    playback_strings.append(f'// Segment {current_segment}')
                    if use_filter:
                        playback_strings.append('i_seg += 1;')
                    first_element_of_segment = True
                    continue
                wave_idx_lookup[element] = {}
                playback_strings.append(f'// Element {element}')

                metadata = awg_sequence_element.pop('metadata', {})

                # The following line only has an effect if the metadata
                # specifies that the segment should be repeated multiple times.
                playback_strings += self._zi_playback_string_loop_start(
                    metadata, [ch1id, ch2id])

                nr_cw = len(set(awg_sequence_element.keys()) - \
                            {'no_codeword'})

                if nr_cw == 1:
                    log.warning(
                        f'Only one codeword has been set for {element}')
                else:
                    for cw in awg_sequence_element:
                        if cw == 'no_codeword':
                            if nr_cw != 0:
                                continue
                        chid_to_hash = awg_sequence_element[cw]
                        wave = tuple(chid_to_hash.get(ch, None) for ch in chids)
                        # include marker chans in the tuple that are currently
                        # not supported
                        wave = (wave[0], None, wave[1], None)
                        ch_has_waveforms[ch1id] |= wave[0] is not None
                        ch_has_waveforms[ch2id] |= wave[2] is not None
                        wave = tuple(None if w is None or not len(waveforms[w])
                                     else w for w in wave)
                        if wave == (None, None, None, None):
                            continue
                        if nr_cw != 0:
                            w1, w2 = self._zi_waves_to_wavenames(wave)
                            if cw not in codeword_table:
                                codeword_table_defs += \
                                    self._zi_codeword_table_entry(
                                        cw, wave, use_placeholder_waves)
                                codeword_table[cw] = (w1, w2)
                            elif codeword_table[cw] != (w1, w2) \
                                    and self.pulsar.reuse_waveforms():
                                log.warning('Same codeword used for different '
                                            'waveforms. Using first waveform. '
                                            f'Ignoring element {element}.')
                        if not len(channels_to_upload):
                            # _program_awg was called only to decide which
                            # sub-AWGs are active, and the rest of this loop
                            # can be skipped
                            continue
                        if use_placeholder_waves:
                            if wave in defined_waves[1].values():
                                wave_idx_lookup[element][cw] = [
                                    i for i, v in defined_waves[1].items()
                                    if v == wave][0]
                                continue
                            wave_idx_lookup[element][cw] = next_wave_idx
                            next_wave_idx += 1
                            placeholder_wave_lengths = [
                                waveforms[h].size for h in wave if h is not None
                            ]
                            if max(placeholder_wave_lengths) != \
                                    min(placeholder_wave_lengths):
                                log.warning(f"Waveforms of unequal length on"
                                            f"{self.awg.name}, vawg{awg_nr}, "
                                            f"{current_segment}, {element}.")
                            wave_definitions += self._zi_wave_definition(
                                wave,
                                defined_waves,
                                max(placeholder_wave_lengths),
                                wave_idx_lookup[element][cw])
                        else:
                            wave_definitions += self._zi_wave_definition(
                                wave, defined_waves)

                    if not len(channels_to_upload):
                        # _program_awg was called only to decide which
                        # sub-AWGs are active, and the rest of this loop
                        # can be skipped
                        continue
                    if not internal_mod:
                        prepend_zeros = False
                        playback_strings += self._zi_playback_string(
                            name=self.awg.name, device='shfsg', wave=wave,
                            codeword=(nr_cw != 0),
                            prepend_zeros=prepend_zeros,
                            placeholder_wave=use_placeholder_waves,
                            allow_filter=metadata.get('allow_filter', False),
                            negate_q=True,
                        )
                    elif not use_placeholder_waves:
                        pb_string, interleave_string = \
                            self._zi_interleaved_playback_string(
                                name=self.awg.name, device='shfsg',
                                counter=counter, wave=wave,
                                codeword=(nr_cw != 0)
                            )
                        counter += 1
                        playback_strings += pb_string
                        interleaves += interleave_string
                    else:
                        raise NotImplementedError("Placeholder waves in "
                                                  "combination with internal "
                                                  "modulation not implemented.")
                    first_element_of_segment = False

                # The following line only has an effect if the metadata
                # specifies that the segment should be repeated multiple times.
                playback_strings += self._zi_playback_string_loop_end(metadata)

            if not any([ch_has_waveforms[ch] for ch in chids]):
                # prevent self.start() from starting this sub AWG
                self.awg._awg_program[awg_nr + first_sg_awg] = None
                continue
            # tell self.start() to start this sub AWG
            # (self.start will start sub AWGs for which _awg_program
            # is not None. We do not need to put the actual program here,
            # but anything different from None is sufficient.)
            self.awg._awg_program[awg_nr + first_sg_awg] = True

            # Having determined whether the sub AWG should be started or
            # not, we can now skip in case no channels need to be uploaded.
            if channels_to_upload != 'all' and not any(
                    [ch in channels_to_upload for ch in chids]):
                continue

            if not use_placeholder_waves:
                waves_to_upload = {h: waveforms[h]
                                   for codewords in awg_sequence.values()
                                   if codewords is not None
                                   for cw, chids in codewords.items()
                                   if cw != 'metadata'
                                   for chid, h in chids.items()}
                self._zi_write_waves(waves_to_upload)

            awg_str = self._shfsg_sequence_string_template.format(
                wave_definitions='\n'.join(wave_definitions + interleaves),
                codeword_table_defs='\n'.join(codeword_table_defs),
                playback_string='\n  '.join(playback_strings),
            )

            if not use_placeholder_waves or channels_to_program == 'all' or \
                    any([ch in channels_to_program for ch in chids]):
                run_compiler = True
            else:
                cached_lookup = self._shfsg_waveform_cache.get(
                    f'{self.awg.name}_{awg_nr}_wave_idx_lookup', None)
                try:
                    np.testing.assert_equal(wave_idx_lookup, cached_lookup)
                    run_compiler = False
                except AssertionError:
                    log.debug(f'{self.awg.name}_{awg_nr}: Waveform reuse '
                              f'pattern has changed. Forcing recompilation.')
                    run_compiler = True

            if run_compiler:
                sgchannel.awg.load_sequencer_program(awg_str, timeout=600)
                if hasattr(self.awg, 'store_awg_source_string'):
                    self.awg.store_awg_source_string(sgchannel, awg_str)

                if use_placeholder_waves:
                    self._shfsg_waveform_cache[f'{self.awg.name}_{awg_nr}'] = {}
                    self._shfsg_waveform_cache[
                        f'{self.awg.name}_{awg_nr}_wave_idx_lookup'] = \
                        wave_idx_lookup

            if use_placeholder_waves:
                for idx, wave_hashes in defined_waves[1].items():
                    self._update_waveforms(awg_nr, idx, wave_hashes, waveforms)

        if self.pulsar.sigouts_on_after_programming():
            for sgchannel in self.awg.sgchannels:
                sgchannel.output.on(True)

        if any(ch_has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self.awg.name)

    def is_awg_running(self):
        is_running = []
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            is_running.append(sgchannel.awg.enable())
        return any(is_running)

    def clock(self):
        return 2.0e9

    def sigout_on(self, ch, on=True):
        chid = self.pulsar.get(ch + '_id')
        self.awg.sgchannels[int(chid[2]) - 1].output.on(on)

    def get_params_for_spectrum(self, ch: str, requested_freqs: list[float]):
        # FIXME: Partially replicated from SHFQA acq dev. Should be refactored
        #   to avoid code replication.
        # FIXME: Extend this method to allow passing several channels and 2D
        # requested_freqs array to allow a shared center frequency between SG
        # channels that share one center frequency. Also think about whether it
        # makes sense to put this functionality in the pulsar and then call
        # the hardware specific method for groups of channels.
        if len(requested_freqs) == 1:
            id_closest = (np.abs(np.array(self.awg.allowed_lo_freqs()) -
                                requested_freqs[0])).argmin()
            if self.awg.allowed_lo_freqs()[id_closest] - requested_freqs[0] < 10e6:
                # resulting mod_freq would be smaller than 10 MHz
                # TODO: arbitrarily chosen limit of 10 MHz
                id_closest = id_closest + (-1 if id_closest != 0 else +1)
        else:
            diff_f = np.diff(requested_freqs)
            if not all(diff_f-diff_f[0] < 1e-3):
                # not equally spaced (arbitrary 1 mHz)
                log.warning(f'Unequal frequency spacing not supported, '
                            f'the measurement will return equally spaced values.')
            # Find closest allowed center frequency
            approx_center_freq = np.mean(requested_freqs)
            id_closest = (np.abs(np.array(self.awg.allowed_lo_freqs()) -
                                approx_center_freq)).argmin()
        center_freq = self.awg.allowed_lo_freqs()[id_closest]
        mod_freqs = requested_freqs - center_freq
        return center_freq, mod_freqs

    def configure_sine_generation(self, ch, enable=True, osc_index=0, freq=None,
                                  phase=0.0, gains=(0.0, 1.0, 1.0, 0.0),
                                  sine_generator_index=0,
                                  force_enable=False):
        """Configure the direct output of the SHF sine generators.

        Unless `force_enable` is set tor True, this method does not enable the
        sine output and instead only sets the corresponding flag in
        'self._sgchannel_sine_enable' which is used in :meth:`self.start`.

        Args:
            ch (str): Name of the SGChannel to configure
            enable (bool, optional): Enable of the sine generator. Also see
                comment above and parameter description of `force_enable`.
                Defaults to True.
            osc_index (int, optional): Index of the digital oscillator to be
                used. Defaults to 0.
            freq (float, optional): Frequency of the digital oscillator. If
                `None` the frequency of the oscillator will
                not be changed. Defaults to `None`.
            phase (float, optional): Phase of the sine generator.
                Defaults to 0.0.
            gains (tuple, optional): Tuple of floats of length 4. Structure:
                (sin I, cos I, sin Q, cos Q). Defaults to `(0.0, 1.0, 1.0, 0.0)`.
            sine_generator_index (int, optional): index of the sine generator to
                be used. Defaults to 0.
            force_enable (bool): In combination with `enable` this parameter
                determines whether the sine output is enabled. Defaults to False
        """
        chid = self.pulsar.get(ch + '_id')
        if freq is None:
            freq = self.awg.sgchannels[int(chid[2]) - 1].oscs[osc_index].freq()
        self.awg.sgchannels[int(chid[2]) - 1].configure_sine_generation(
            enable=(enable and force_enable),
            osc_index=osc_index,
            osc_frequency=freq,
            phase=phase,
            gains=gains,
            sine_generator_index=sine_generator_index,
        )
        self._sgchannel_sine_enable[int(chid[2]) - 1] = enable

    def configure_internal_mod(self, ch, enable=True, osc_index=0, phase=0.0,
                               global_amp=0.5, gains=(1.0, - 1.0, 1.0, 1.0),
                               sine_generator_index=0):
        """Configure the internal modulation of the SG channel.

        Args:
            ch (str): Name of the SGChannel to configure
            enable (bool, optional): Enable of the digital modulation.
                Defaults to True.
            osc_index (int, optional): Index of the digital oscillator to be
                used. Defaults to 0.
            phase (float, optional): Phase of the digital modulation.
                Defaults to 0.0.
            global_amp (float, optional): Defaults to 0.5.
            gains (tuple, optional): Tuple of floats of length 4. Structure:
                (sin I, cos I, sin Q, cos Q). Defaults to (1.0, -1.0, 1.0, 1.0).
            sine_generator_index (int, optional): index of the sine generator to
                be used. Defaults to 0.
        """
        chid = self.pulsar.get(ch + '_id')
        self.awg.sgchannels[int(chid[2]) - 1].configure_pulse_modulation(
            enable=enable,
            osc_index=osc_index,
            osc_frequency=self.awg.sgchannels[int(chid[2]) - 1].oscs[osc_index].freq(),
            phase=phase,
            global_amp=global_amp,
            gains=gains,
            sine_generator_index=sine_generator_index,
        )
        self.configure_sine_generation(ch,
            enable=False, # do not turn on the output of the sine
                          # generator for internal modulation
            osc_index=osc_index,
            sine_generator_index=sine_generator_index)

    def get_frequency_sweep_function(self, ch, mod_freq=0):
        """
        Args:
            ch (str): Name of the SGChannel to configure
            mod_freq(float): Modulation frequency of the pulse uploaded to the
                AWG. In case the continuous output is used, this should be set
                to 0. Defaults to 0.
        """
        chid = self.pulsar.get(ch + '_id')
        name = 'Frequency'
        if self.pulsar.get(f"{self.awg.name}_use_hardware_sweeper"):
            return swf.SpectroscopyHardSweep(parameter_name=name)
        name_offset = 'Frequency with offset'
        return swf.Offset_Sweep(
            swf.MajorMinorSweep(
                self.awg.synthesizers[
                    self.awg.sgchannels[int(chid[2]) - 1].synthesizer()].centerfreq,
                swf.Offset_Sweep(
                    self.awg.sgchannels[int(chid[2]) - 1].oscs[0].freq,
                    # FIXME: osc_id (0) should depend on element metadata['sine_config']['ch']['osc']
                    mod_freq),
                self.awg.allowed_lo_freqs(),
                name=name_offset, parameter_name=name_offset),
            -mod_freq, name=name, parameter_name=name)

    def start(self):
        first_sg_awg = len(getattr(self.awg, 'qachannels', []))
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            if self.awg._awg_program[awg_nr + first_sg_awg] is not None:
                sgchannel.awg.enable(1)
            if self._sgchannel_sine_enable[awg_nr]:
                sgchannel.sines[0].i.enable(1)
                sgchannel.sines[0].q.enable(1)

    def stop(self):
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            sgchannel.awg.enable(0)
            if self._sgchannel_sine_enable[awg_nr]:
                sgchannel.sines[0].i.enable(0)
                sgchannel.sines[0].q.enable(0)

    def _update_waveforms(self, awg_nr, wave_idx, wave_hashes, waveforms):
        if self.pulsar.use_sequence_cache():
            if wave_hashes == self._shfsg_waveform_cache[
                f'{self.awg.name}_{awg_nr}'].get(wave_idx, None):
                log.debug(
                    f'{self.awg.name} awgs{awg_nr}: {wave_idx} same as in cache')
                return
            log.debug(
                f'{self.awg.name} awgs{awg_nr}: {wave_idx} needs to be uploaded')
            self._shfsg_waveform_cache[f'{self.awg.name}_{awg_nr}'][
                wave_idx] = wave_hashes
        a1, m1, a2, m2 = [waveforms.get(h, None) for h in wave_hashes]
        n = max([len(w) for w in [a1, m1, a2, m2] if w is not None])
        if m1 is not None and a1 is None:
            a1 = np.zeros(n)
        if m1 is None and a1 is None and (m2 is not None or a2 is not None):
            # FIXME: test if this hack is needed one marker support is added
            # Hack needed to work around an HDAWG bug where programming only
            # m2 channel does not work. Remove once bug is fixed.
            a1 = np.zeros(n)
        if m2 is not None and a2 is None:
            a2 = np.zeros(n)
        if m1 is not None or m2 is not None:
            m1 = np.zeros(n) if m1 is None else np.pad(m1, n - m1.size)
            m2 = np.zeros(n) if m2 is None else np.pad(m2, n - m2.size)
            if a1 is None:
                mc = m2
            else:
                mc = m1 + 4 * m2
        else:
            mc = None
        a1 = None if a1 is None else np.pad(a1, n - a1.size)
        a2 = None if a2 is None else np.pad(a2, n - a2.size)
        assert mc is None # marker not yet supported on SG

        sgchannel = self.awg.sgchannels[awg_nr]
        waveforms = zhinst.toolkit.waveform.Waveforms()
        # FIXME: Test whether the sign of the second channel needs to be flipped
        waveforms.assign_waveform(wave_idx, a1, a2)
        sgchannel.awg.write_to_waveform_memory(waveforms)

    def _direct_mod_setter(self, ch):
        def s(val):
            if val == None:
                self.configure_sine_generation(ch, enable=False)
            else:
                self.configure_sine_generation(ch, enable=True, freq=val,
                                               force_enable=True)
        return s

    def _direct_mod_getter(self, ch):
        def g():
            chid = self.pulsar.get(ch + '_id')
            sgchannel = self.awg.sgchannels[int(chid[2]) - 1]
            if sgchannel.sines[0].i.enable() or sgchannel.sines[0].q.enable():
                return sgchannel.sines[0].freq()
            else:
                return None
        return g

    def _direct_mod_amplitude_setter(self, ch):
        def s(val):
            chid = self.pulsar.get(ch + '_id')
            sgchannel = self.awg.sgchannels[int(chid[2]) - 1]
            sgchannel.sines[0].i.sin.amplitude(0)
            sgchannel.sines[0].i.cos.amplitude(val)
            sgchannel.sines[0].q.sin.amplitude(val)
            sgchannel.sines[0].q.cos.amplitude(0)
        return s

    def _direct_mod_amplitude_getter(self, ch):
        def g():
            chid = self.pulsar.get(ch + '_id')
            sgchannel = self.awg.sgchannels[int(chid[2]) - 1]
            gains = [sgchannel.sines[0].i.sin.amplitude(),
                     sgchannel.sines[0].i.cos.amplitude(),
                     sgchannel.sines[0].q.sin.amplitude(),
                     sgchannel.sines[0].q.cos.amplitude()]
            val = gains[1]
            if gains == [0, val, val, 0]:
                return val
            log.warning('The current sine gain configuration is not '
                        'supported by pulsar. Cannot retrieve amplitude.')
        return g


class SHFSGPulsar(SHFGeneratorModulePulsar):
    """ZI SHFSG specific Pulsar module"""
    AWG_CLASSES = [SHFSG_core]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.awg._awg_program = [None] * len(self.awg.sgchannels)

    def create_awg_parameters(self, channel_name_map: dict):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        pulsar.add_parameter(f"{name}_use_placeholder_waves",
                             initial_value=False, vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_trigger_source",
                             initial_value="Dig1",
                             vals=vals.Enum("Dig1", "DIO", "ZSync"),
                             parameter_class=ManualParameter,
                             docstring="Defines for which trigger source the "
                                       "AWG should wait, before playing the "
                                       "next waveform. Allowed values are: "
                                       "'Dig1', 'DIO', 'ZSync'.")
        pulsar.add_parameter(f"{name}_use_hardware_sweeper",
                             initial_value=False,
                             parameter_class=ManualParameter,
                             docstring='Bool indicating whether the hardware '
                                       'sweeper should be used in spectroscopy '
                                       'mode',
                             vals=vals.Bool())

        self._create_all_channel_parameters(channel_name_map)
