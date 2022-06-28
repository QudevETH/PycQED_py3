import logging
from typing import List, Tuple

import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.math import vp_to_dbm, dbm_to_vp
from .zi_pulsar_mixin import ZIPulsarMixin
from .pulsar import PulsarAWGInterface

import zhinst
import zhinst.toolkit

try:
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
    IMPLEMENTED_ACCESSORS = ["amp"]

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

    def create_awg_parameters(self, channel_name_map: dict):
        PulsarAWGInterface.create_awg_parameters(self, channel_name_map)

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

        # real and imaginary part of the wave form channel groups
        for ch_nr in range(len(self.awg.sgchannels)):
            group = []
            for q in ["i", "q"]:
                id = f"sg{ch_nr + 1}{q}"
                ch_name = channel_name_map.get(id, f"{name}_{id}")
                self.create_channel_parameters(id, ch_name, "analog")
                pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                pulsar.channel_groups.update({ch_name: group})
        # FIXME: add support for marker channels

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFSG, valid channel ids are sg#i and sg#q, where # is a number
        from 1 to 8. This defines the harware port used.
        """

        super().create_channel_parameters(id, ch_name, ch_type)

        # TODO: Not all AWGs provide an initial value. Should it be the case?
        self.pulsar[f"{ch_name}_amp"].set(1)

    def awg_setter(self, id:str, param:str, value):
        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "amp":
            self.awg.sgchannels[ch].output.range(vp_to_dbm(value))

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

        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            defined_waves = (set(), dict()) if use_placeholder_waves else set()
            codeword_table = {}
            wave_definitions = []
            codeword_table_defs = []
            playback_strings = []
            interleaves = []

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
            # FIXME: internal modulation not yet supported on SHFSG
            # if all([self.pulsar.get(f"{chan}_internal_modulation")
            #         for chan in channels]):
            #     internal_mod = True
            # elif not any([self.pulsar.get(f"{chan}_internal_modulation")
            #               for chan in channels]):
            #     internal_mod = False
            # else:
            #     raise NotImplementedError('Internal modulation can only be'
            #                               'specified per sub AWG!')
            internal_mod = False

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
                        ch_has_waveforms[ch1id] |= wave[0] is not None
                        ch_has_waveforms[ch2id] |= wave[2] is not None
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
                            name=self.awg.name, device='hdawg', wave=wave,
                            codeword=(nr_cw != 0),
                            prepend_zeros=prepend_zeros,
                            placeholder_wave=use_placeholder_waves,
                            allow_filter=metadata.get('allow_filter', False),
                            negate_q=True,
                        )
                    elif not use_placeholder_waves:
                        pb_string, interleave_string = \
                            self._zi_interleaved_playback_string(
                                name=self.awg.name, device='hdawg',
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
                # FIXME: enable this when overhauling sequencer cache for SG
                # prevent ZI_base_instrument.start() from starting this sub AWG
                # self.awg._awg_program[awg_nr] = None
                continue
            # tell ZI_base_instrument that it should not compile a
            # program on this sub AWG (because we already do it here)
            # FIXME: delete this when overhauling sequencer cache for SG
            # self.awg._awg_needs_configuration[awg_nr] = False
            # tell ZI_base_instrument.start() to start this sub AWG
            # (The base class will start sub AWGs for which _awg_program
            # is not None. Since we set _awg_needs_configuration to False,
            # we do not need to put the actual program here, but anything
            # different from None is sufficient.)
            # FIXME: enable this when overhauling sequencer cache for SG
            # self.awg._awg_program[awg_nr] = True

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

    def start(self):
        for sgchannel in self.awg.sgchannels:
            sgchannel.awg.enable(1)

    def stop(self):
        for sgchannel in self.awg.sgchannels:
            sgchannel.awg.enable(0)

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


class SHFSGPulsar(SHFGeneratorModulePulsar):
    """ZI SHFSG specific Pulsar module"""

    AWG_CLASSES = [SHFSG_core]

