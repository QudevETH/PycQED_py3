import logging
from typing import List, Tuple

import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter
from pycqed.utilities.math import vp_to_dbm, dbm_to_vp

try:
    from pycqed.instrument_drivers.acquisition_devices.shf \
        import SHF_AcquisitionDevice
except Exception:
    SHF_AcquisitionDevice = type(None)
try:
    from zhinst.qcodes import SHFQA as SHFQA_core
except Exception:
    SHFQA_core = type(None)

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class SHFAcquisitionModulePulsar(PulsarAWGInterface):
    """ZI SHFQA and SHFQC acquisition module support for the Pulsar class.

    Supports :class:`pycqed.measurement.waveform_control.segment.Segment`
    objects with the following values for acquisition_mode:
        'default' for a measurement in readout mode
        dict in spectroscopy mode, with 'sweeper' key (allowed values:
        'hardware', 'software'), see other allowed keys in
        :class:`pycqed.measurement.waveform_control.segment.Segment`.
    """

    AWG_CLASSES = []
    GRANULARITY = 4
    ELEMENT_START_GRANULARITY = 4 / 2.0e9 # TODO: unverified!
    MIN_LENGTH = 4 / 2.0e9
    INTER_ELEMENT_DEADTIME = 0 # TODO: unverified!
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.001, 1),
    }
    IMPLEMENTED_ACCESSORS = ["amp"]

    def create_awg_parameters(self, channel_name_map: dict):
        PulsarAWGInterface.create_awg_parameters(self, channel_name_map)


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
        for ch_nr in range(len(self.awg.qachannels)):
            group = []
            for q in ["i", "q"]:
                id = f"qa{ch_nr + 1}{q}"
                ch_name = channel_name_map.get(id, f"{name}_{id}")
                self.create_channel_parameters(id, ch_name, "analog")
                pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFQA, valid channel ids are qa#i and qa#q, where # is a number
        from 1 to 4. This defines the harware port used.
        """

        super().create_channel_parameters(id, ch_name, ch_type)

        # TODO: Not all AWGs provide an initial value. Should it be the case?
        self.pulsar[f"{ch_name}_amp"].set(1)

    def awg_setter(self, id:str, param:str, value):

        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "amp":
            self.awg.qachannels[ch].output.range(vp_to_dbm(value))

    def awg_getter(self, id:str, param:str):

        # Sanity checks
        super().awg_getter(id, param)

        ch = int(id[2]) - 1

        if param == "amp":
            if self.pulsar.awgs_prequeried:
                dbm = self.awg.qachannels[ch].output.range.get_latest()
            else:
                dbm = self.awg.qachannels[ch].output.range()
            return dbm_to_vp(dbm)

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        # TODO: For now, we only check for channels_to_upload and always
        # re-program when re-uploading (i.e., we ignore channels_to_program).
        # This could be further optimized in the future. Moreover, we currently
        # ignore channels_to_upload in spectroscopy mode, i.e., we always
        # re-upload in spectroscopy mode. This could be optimized in the future.

        grp_has_waveforms = {}
        for i, qachannel in enumerate(self.awg.qachannels):
            grp = f'qa{i+1}'
            chids = [f'qa{i+1}i', f'qa{i+1}q']
            grp_has_waveforms[grp] = False
            channels = set(self.pulsar._id_channel(chid, self.awg.name)
                        for chid in chids)

            playback_strings = []

            waves_to_upload = {}
            is_spectroscopy = False
            for codewords in awg_sequence.values():
                if codewords is None:
                    continue
                for cw, chid_to_hash in codewords.items():
                    if cw == 'metadata':
                        acq = chid_to_hash.get('acq', False)
                        if acq and 'sweeper' in acq:
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

            # This one needs to be defined here and not as a class constant,
            # otherwise SHFQA.USER_REG_... would crash on setups which do not
            # have an SHFQA object initialised
            shfqa_sequence_string_template = (
                "var loop_cnt = "
                f"getUserReg({SHF_AcquisitionDevice.USER_REG_LOOP_COUNT});\n"
                "var acq_len = "
                f"getUserReg({SHF_AcquisitionDevice.USER_REG_ACQ_LEN});"
                f" // only needed in sweeper mode\n"
                "{prep_string}"
                "\n"
                "repeat (loop_cnt) {{\n"
                "  {playback_string}\n"
                "}}\n"
            )
            shfqa_sweeper_playback_string_template = (
                "for(var i = 0; i < {n_step}; i++)" + " {{\n"
                "    // self-triggering mode\n\n"
                "    // define time from setting the oscillator "
                "frequency to sending the spectroscopy trigger\n"
                "    playZero(400);\n    \n"
                "    // set the oscillator frequency depending "
                "on the loop variable i\n"
                "    setSweepStep(OSC0, i);\n    \n"
                "    waitDigTrigger(1);\n"
                "    resetOscPhase();\n\n"
                "    // define time to the next iteration\n"
                "    playZero(acq_len + 144);\n\n"
                "    // trigger the integration unit and pulsed "
                "playback in pulsed mode\n"
                "    setTrigger(1);\n    setTrigger(0);\n"
                "  }}"
            )
            shfqa_sweeper_prep_string = (
                "const OSC0 = 0;\n"
                "setTrigger(0);\n"
                "configFreqSweep(OSC0, {f_start}, {f_step});\n"
            )

            self.awg.seqtrigger = None

            if is_spectroscopy:
                for element in awg_sequence:
                    # This is a light copy of the readout mode below,
                    # not sure how to make this more general without a
                    # use case.
                    awg_sequence_element = deepcopy(awg_sequence[element])
                    if awg_sequence_element is None:
                        playback_strings.append(f'// Segment {element}')
                        continue
                    playback_strings.append(f'// Element {element}')
                    metadata = awg_sequence_element.pop('metadata', {})
                    # if list(awg_sequence_element.keys()) != ['no_codeword']:
                    #     raise NotImplementedError('SHFQA sequencer does '
                    #         'currently not support codewords!')
                    # chid_to_hash = awg_sequence_element['no_codeword']
                    acq = metadata.get('acq', False)
                    break  # FIXME: assumes there is only one segment
                if acq['sweeper'] == 'hardware':
                    # FIXME: at some point, we need to test whether the freqs
                    #  are supported by the sweeper
                    playback_strings.append(
                        shfqa_sweeper_playback_string_template.format(
                            n_step=acq['n_step']))
                    self.awg.set_awg_program(
                        i,
                        shfqa_sequence_string_template.format(
                            prep_string=shfqa_sweeper_prep_string.format(
                                f_start=acq['f_start'],
                                f_step=acq['f_step'],
                            ),
                            playback_string='\n  '.join(playback_strings)))
                    # The acquisition modules will each be triggered by their
                    # sequencer
                    self.awg.seqtrigger = True
                else:
                    self.awg._awg_program[i] = None  # do not start generator

                # FIXME: check whether some of this code should be moved to
                #  the SHFQA class in the next cleanup
                w = list(waves_to_upload.values())
                w = w[0] if len(w) > 0 else None
                qachannel.mode('spectroscopy')
                daq = self.awg.daq
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

            def play_element(element, playback_strings, acq_unit):
                awg_sequence_element = deepcopy(awg_sequence[element])
                if awg_sequence_element is None:
                    current_segment = element
                    playback_strings.append(f'// Segment {current_segment}')
                    return playback_strings
                playback_strings.append(f'// Element {element}')

                metadata = awg_sequence_element.pop('metadata', {})
                groups = metadata['trigger_group']
                group_channels = []
                for group in groups:
                    group_channels += \
                        self.pulsar.get_trigger_group_channels(group)

                if len(set(group_channels) & channels) == 0:
                    return playback_strings

                # The following line only has an effect if the metadata
                # specifies that the segment should be repeated multiple times.
                playback_strings += self._zi_playback_string_loop_start(
                    metadata, [f'qa{acq_unit+1}i', f'qa{acq_unit+1}q'])

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
                # If 0x1, generates an internal trigger signal from the
                # sequencer module
                trig = '0x1' if (isinstance(acq, dict) and
                                 acq.get('seqtrigger', False)) else '0x0'
                playback_strings += [
                    f'waitDigTrigger(1);',
                    f'startQA({wave_mask}, {int_mask}, {monitor}, 0, {trig});'
                ]
                if trig == '0x1':
                    if self.awg.seqtrigger is None:
                        # The scope will be triggered by this single acq_unit
                        self.awg.seqtrigger = acq_unit
                    playback_strings += [
                        f'wait(3);',  # (3+2)5ns=20ns (wait has 2 cycle offset)
                        f'setTrigger(0x0);'
                    ]
                return playback_strings

            qachannel.mode('readout')
            self._filter_segment_functions = None

            if repeat_pattern is not None:
                log.info("Repeat patterns not yet implemented on SHFQA, "
                         "ignoring it")
            for element in awg_sequence:
                playback_strings = play_element(element, playback_strings, i)
            self.awg.set_awg_program(
                i,
                shfqa_sequence_string_template.format(
                    playback_string='\n  '.join(playback_strings),
                    prep_string=''),
                {hash_to_index_map[k]: v for k, v in waves_to_upload.items()})

        if any(grp_has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self.awg.name)

    def is_awg_running(self):
        is_running = []
        for awg_nr, qachannel in enumerate(self.awg.qachannels):
            if qachannel.mode().name == 'readout':
                is_running.append(qachannel.generator.enable())
            else:  # spectroscopy
                daq = self.awg.daq
                path = f"/{self.awg.get_idn()['serial']}/qachannels/{awg_nr}/" \
                       f"spectroscopy/result/enable"
                is_running.append(daq.getInt(path) != 0)
        return any(is_running)

    def clock(self):
        return 2.0e9

    def sigout_on(self, ch, on=True):
        chid = self.pulsar.get(ch + '_id')
        self.awg.qachannels[int(chid[2]) - 1].output.on(on)


class SHFQAPulsar(SHFAcquisitionModulePulsar):
    """ZI SHFQA specific Pulsar module"""

    AWG_CLASSES = [SHFQA_core]
