import logging
import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter
try:
    from zhinst.qcodes import SHFQA
except Exception:
    SHFQA = type(None)

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class SHFQAPulsar(PulsarAWGInterface):
    """ZI SHFQA specific functionality for the Pulsar class."""

    AWG_CLASSES = [SHFQA]
    GRANULARITY = 4
    ELEMENT_START_GRANULARITY = 4 / 2.0e9 # TODO: unverified!
    MIN_LENGTH = 4 / 2.0e9
    INTER_ELEMENT_DEADTIME = 0 # TODO: unverified!
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.001, 1),
    }
    # TODO: SHFQA had no parameter for offset, should we delete it for this
    # subclass, or just force it to 0 ?
    CHANNEL_OFFSET_BOUNDS = {
        "analog": (0, 0),
    }
    IMPLEMENTED_ACCESSORS = ["amp"]

    _shfqa_sequence_string_template = (
        "// hardcoded value until we figure out user registers\n"
        "var loop_cnt = {loop_count};\n"
        "\n"
        "repeat (loop_cnt) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def create_awg_parameters(self, channel_name_map: dict):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        # Repeat pattern support is not yet implemented for the SHFQA, thus we
        # remove this parameter added in super().create_awg_parameters()
        del pulsar.parameters[f"{name}_minimize_sequencer_memory"]

        pulsar.add_parameter(f"{name}_trigger_source",
                             initial_value="Dig1",
                             vals=vals.Enum("Dig1",),
                             parameter_class=ManualParameter,
                             docstring="Defines for which trigger source the "
                                       "AWG should wait, before playing the "
                                       "next waveform. Only allowed value is "
                                       "'Dig1 for now.")

        # real and imaginary part of the wave form channel groups
        for ch_nr in range(4):
            group = []
            for q in ["i", "q"]:
                id = f"ch{ch_nr + 1}{q}"
                ch_name = channel_name_map.get(id, f"{name}_{id}")
                self.create_channel_parameters(id, ch_name)
                pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFQA, valid channel ids are ch#i and ch#q, where # is a number
        from 1 to 4. This defines the harware port used.
        """

        super().create_channel_parameters(id, ch_name, ch_type)

        # TODO: Not all AWGs provide an initial value. Should it be the case?
        self.pulsar[f"{ch_name}_amp"].set(1)

    @staticmethod
    def awg_setter(self, id:str, param:str, value):

        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "amp":
            self.awg.qachannels[ch].output_range(20 * (np.log10(value) + 0.5))

    def awg_getter(self, id:str, param:str):

        # Sanity checks
        super().awg_getter(id, param)

        ch = int(id[2]) - 1

        if param == "amp":
            if self.pulsar.awgs_prequeried:
                dbm = self.awg.qachannels[ch].output_range.get_latest()
            else:
                dbm = self.awg.qachannels[ch].output_range()
            return 10 ** (dbm /20 - 0.5)

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        # TODO: For now, we only check for channels_to_upload and always
        # re-program when re-uploading (i.e., we ignore channels_to_program).
        # This could be further optimized in the future. Moreover, we currently
        # ignore channels_to_upload in spectroscopy mode, i.e., we always
        # re-upload in spectroscopy mode. This could be optimized in the future.

        grp_has_waveforms = {f'ch{i+1}': False for i in range(4)}
        for i, qachannel in enumerate(self.awg.qachannels):
            grp = f'ch{i+1}'
            chids = [f'ch{i+1}i', f'ch{i+1}q']

            playback_strings = []

            waves_to_upload = {}
            is_spectroscopy = False
            for codewords in awg_sequence.values():
                if codewords is None:
                    continue
                for cw, chid_to_hash in codewords.items():
                    if cw == 'metadata':
                        acq = chid_to_hash.get('acq', False)
                        if acq == 'sweeper':
                            is_spectroscopy = True
                    hi = chid_to_hash.get(chids[0], None)
                    hq = chid_to_hash.get(chids[1], None)
                    if hi is None and hq is None:
                        continue
                    grp_has_waveforms[grp] = True
                    if not len(channels_to_upload):
                        # _program_awg was called only to decide which
                        # sub-AWGs are active, and the rest of this loop
                        # can be skipped
                        continue
                    wi = waveforms.get(hi, np.zeros(1))
                    wq = waveforms.get(hq, np.zeros(1))
                    wlen = max(len(wi), len(wq))
                    w = np.pad(wi, [(0, wlen - len(wi))], mode='constant') - \
                        np.pad(wq, [(0, wlen - len(wq))], mode='constant')*1j
                    waves_to_upload[(hi, hq)] = w
            if not grp_has_waveforms[grp]:
                log.debug(f'{self.awg.name}: no waveforms on group {i}')
                self.awg.awg_active[i] = False
                continue
            self.awg.awg_active[i] = True

            # Having determined whether the group should be started or
            # not, we can now skip in case no channels need to be uploaded.
            if channels_to_upload != 'all' and not any(
                    [ch in channels_to_upload for ch in chids]):
                log.debug(f'{self.awg.name}: skip programming group {i}')
                continue
            log.debug(f'{self.awg.name}: programming group {i}')

            hash_to_index_map = {h: i for i, h in enumerate(waves_to_upload)}

            if is_spectroscopy and len(waves_to_upload) > 1:
                log.error(f"Can not have multiple elements in spectroscopy mode"
                          f"on {self.awg.name}, channel {i+1}")
                continue

            for h, w in waves_to_upload.items():
                max_len = 16*4096 if is_spectroscopy else 4096
                if len(w) > max_len:
                    log.error(f"SHFQA supports max {max_len} sample long "
                              f"waveforms. Clipping the waveform.")
                waves_to_upload[h] = w[:max_len]

            if is_spectroscopy:
                w = list(waves_to_upload.values())
                w = w[0] if len(w) > 0 else None
                qachannel.mode('spectroscopy')
                daq = self.awg._controller._controller.connection._daq
                path = f"/{self.awg.get_idn()['serial']}/qachannels/{i}/" \
                       f"spectroscopy/envelope"
                if w is not None:
                    daq.setVector(path + "/wave", w.astype("complex128"))
                    daq.setInt(path + "/enable", 1)
                    daq.setDouble(path + "/delay", 0)
                else:
                    daq.setInt(path + "/enable", 0)
                daq.sync()
                continue

            def play_element(element, playback_strings):
                awg_sequence_element = deepcopy(awg_sequence[element])
                if awg_sequence_element is None:
                    current_segment = element
                    playback_strings.append(f'// Segment {current_segment}')
                    return playback_strings
                playback_strings.append(f'// Element {element}')

                metadata = awg_sequence_element.pop('metadata', {})
                if list(awg_sequence_element.keys()) != ['no_codeword']:
                    raise NotImplementedError('SHFQA sequencer does currently\
                                                       not support codewords!')
                chid_to_hash = awg_sequence_element['no_codeword']

                acq = metadata.get('acq', False)
                h = tuple([chid_to_hash.get(chid, None) for chid in chids])
                wave_idx = hash_to_index_map.get(h, None)
                wave_mask = f'QA_GEN_{wave_idx}' if wave_idx is not None \
                    else '0x0'
                int_mask = 'QA_INT_ALL' if acq else '0x0'
                monitor = 'true' if acq else 'false'
                playback_strings += [
                    f'waitDigTrigger(1);',
                    f'startQA({wave_mask}, {int_mask}, {monitor}, 0, 0x0);'
                ]

                return playback_strings

            qachannel.mode('readout')
            self._filter_segment_functions[self.awg.name] = None

            if repeat_pattern is not None:
                log.info("Repeat patterns not yet implemented on SHFQA, "
                         "ignoring it")
            for element in awg_sequence:
                playback_strings = play_element(element, playback_strings)

            # provide sequence data to SHFQA object for upload in
            # acquisition_initialize
            self.awg.set_awg_program(
                i,
                self._shfqa_sequence_string_template.format(
                    loop_count='{loop_count}',  # will be replaced by SHFQA driver
                    playback_string='\n  '.join(playback_strings)),
                waves_to_upload)

        if any(grp_has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self.awg.name)


    def is_awg_running(self):

        is_running = []
        for awg_nr in range(4):
            qachannel = self.awg.qachannels[awg_nr]
            if qachannel.mode() == 'readout':
                is_running.append(qachannel.generator.is_running)
            else:  # spectroscopy
                daq = self.awg._controller._controller.connection._daq
                path = f"/{self.awg.get_idn()['serial']}/qachannels/{awg_nr}/" \
                       f"spectroscopy/result/enable"
                is_running.append(daq.getInt(path) != 0)
        return any(is_running)

    def clock(self):
        return 2.0e9

    def sigout_on(self, ch, on=True):
        """Turn channel outputs on or off."""

        awg = self.find_instrument(self.get(ch + '_awg'))
        chid = self.get(ch + '_id')
        awg.qachannels[int(chid[-2]) - 1].output(True)
