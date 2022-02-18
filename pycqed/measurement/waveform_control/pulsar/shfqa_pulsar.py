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

        # Repeat pattern support is not yet implemented for the SHFQA, thus we
        # remove this parameter added in super().create_awg_parameters()
        del pulsar.parameters[f"{self.awg.name}_minimize_sequencer_memory"]





        self.add_parameter('{}_trigger_source'.format(self.awg.name),
                           initial_value='Dig1',
                           vals=vals.Enum('Dig1',),
                           parameter_class=ManualParameter,
                           docstring='Defines for which trigger source \
                                      the AWG should wait, before playing \
                                      the next waveform. Only allowed value \
                                      is "Dig1" for now.')

        # real and imaginary part of the wave form channel groups
        for ch_nr in range(4):
            group = []
            for q in ['i', 'q']:
                id = f'ch{ch_nr + 1}{q}'
                name = channel_name_map.get(id, self.awg.name + '_' + id)
                self._shfqa_create_channel_parameters(id, name, self.awg)
                self.channels.add(name)
                group.append(name)
            for name in group:
                self.channel_groups.update({name: group})

    def _shfqa_create_channel_parameters(self, id, name, awg):
        """Create parameters in the pulsar specific to one added channel

        Args:
            id:
                Channel id. For the SHFQA, valid channel ids are ch#i and ch#q,
                where # is a number from 1 to 4. This defines the harware port
                used.
            name:
                Name of the channel to address it by in rest of the codebase.
            awg:
                Instance of the AWG this channel is on.

        """
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_amp'.format(name),
                           label='{} amplitude'.format(name), unit='V',
                           set_cmd=self._shfqa_setter(awg, id, 'amp'),
                           get_cmd=self._shfqa_getter(awg, id, 'amp'),
                           vals=vals.Numbers(0.001, 1),
                           initial_value=1)
        self.add_parameter('{}_distortion'.format(name),
                           label='{} distortion mode'.format(name),
                           initial_value='off',
                           vals=vals.Enum('off', 'precalculate'),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                           label='{} distortion dictionary'.format(name),
                           vals=vals.Dict(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0., 1.), initial_value=0.5)
        self.add_parameter('{}_compensation_pulse_delay'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter(
            '{}_compensation_pulse_gaussian_filter_sigma'.format(name),
            initial_value=0, unit='s',
            parameter_class=ManualParameter)

    @staticmethod
    def _shfqa_setter(obj, id, par):
        """Generate a function to set the output amplitude of a channel.
                Converts the input in volts to dBm."""
        if par == 'amp':
            def s(val):
                obj.qachannels[int(id[2]) - 1].output_range(
                    20 * (np.log10(val) + 0.5)
                )
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _shfqa_getter(self, obj, id, par):
        """Generate a function to get the output amplitude of a channel.
                Converts the output in dBm to volts."""
        if par == 'amp':
            def g():
                if self._awgs_prequeried_state:
                    dbm = obj.qachannels[int(id[2]) - 1].output_range\
                        .get_latest()
                else:
                    dbm = obj.qachannels[int(id[2]) - 1].output_range()
                return 10**(dbm/20 - 0.5)
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     channels_to_upload='all', **kw):
        """Upload the waveforms to the SHFQA.

        Args:
            obj:
                AWG instance
            awg_sequence:
                AWG sequence data (not waveforms) as returned from
                `Sequence.generate_waveforms_sequences`. The key-structure
                of the nested dictionary is like this:
                `awg_sequence[elname][cw][chid]`,
                where elname is the elment name, cw is the codeword, or the
                string `'no_codeword'` and chid is the channel id. The values
                are hashes of waveforms to be played
            waveforms:
                A dictionary of waveforms, keyed by their hash.
            repeat_pattern:
                Not used for now
            channels_to_upload:
                list of channel names to upload or 'all'

        Keyword args:
            No keyword arguments are used.
        """
        # Note: For now, we only check for channels_to_upload and always
        # re-program when re-uploading (i.e., we ignore channels_to_program).
        # This could be further optimized in the future. Moreover, we currently
        # ignore channels_to_upload in spectroscopy mode, i.e., we always
        # re-upload in spectroscopy mode. This could be optimized in the future.

        grp_has_waveforms = {f'ch{i+1}': False for i in range(4)}
        for i, qachannel in enumerate(obj.qachannels):
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
                log.debug(f'{obj.name}: no waveforms on group {i}')
                obj.awg_active[i] = False
                continue
            obj.awg_active[i] = True

            # Having determined whether the group should be started or
            # not, we can now skip in case no channels need to be uploaded.
            if channels_to_upload != 'all' and not any(
                    [ch in channels_to_upload for ch in chids]):
                log.debug(f'{obj.name}: skip programming group {i}')
                continue
            log.debug(f'{obj.name}: programming group {i}')

            hash_to_index_map = {h: i for i, h in enumerate(waves_to_upload)}

            if is_spectroscopy and len(waves_to_upload) > 1:
                log.error(f"Can not have multiple elements in spectroscopy mode"
                          f"on {obj.name}, channel {i+1}")
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
                daq = obj._controller._controller.connection._daq
                path = f"/{obj.get_idn()['serial']}/qachannels/{i}/" \
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
            self._filter_segment_functions[obj.name] = None

            if repeat_pattern is not None:
                log.info("Repeat patterns not yet implemented on SHFQA, "
                         "ignoring it")
            for element in awg_sequence:
                playback_strings = play_element(element, playback_strings)

            # provide sequence data to SHFQA object for upload in
            # acquisition_initialize
            obj.set_awg_program(
                i,
                self._shfqa_sequence_string_template.format(
                    loop_count='{loop_count}',  # will be replaced by SHFQA driver
                    playback_string='\n  '.join(playback_strings)),
                waves_to_upload)

        if any(grp_has_waveforms.values()):
            self.awgs_with_waveforms(obj.name)


    def _is_awg_running(self, obj):
        """Checks whether the sequencer of AWG `obj` is running"""

        is_running = []
        for awg_nr in range(4):
            qachannel = obj.qachannels[awg_nr]
            if qachannel.mode() == 'readout':
                is_running.append(qachannel.generator.is_running)
            else:  # spectroscopy
                daq = obj._controller._controller.connection._daq
                path = f"/{obj.get_idn()['serial']}/qachannels/{awg_nr}/" \
                       f"spectroscopy/result/enable"
                is_running.append(daq.getInt(path) != 0)
        return any(is_running)

    def _clock(self, obj, cid=None):
        """Returns the sample clock of the SHFQA: 2 GHz."""

        return 2.0e9

    def _get_segment_filter_userregs(self, obj):
        """Segment filter currently not supported on SHFQA"""

        return []

    def sigout_on(self, ch, on=True):
        """Turn channel outputs on or off."""

        awg = self.find_instrument(self.get(ch + '_awg'))
        chid = self.get(ch + '_id')
        awg.qachannels[int(chid[-2]) - 1].output(True)
