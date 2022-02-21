from copy import deepcopy
import logging
import numpy as np

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
        UHFQA_core import UHFQA_core
except Exception:
    UHFQA_core = type(None)

from .pulsar import PulsarAWGInterface
from .zi_pulsar_mixin import ZIPulsarMixin


log = logging.getLogger(__name__)


class UHFQCPulsar(PulsarAWGInterface, ZIPulsarMixin):
    """ZI UHFQC specific functionality for the Pulsar class."""

    AWG_CLASSES = [UHFQA_core]
    GRANULARITY = 16
    ELEMENT_START_GRANULARITY = 8 / 1.8e9
    MIN_LENGTH = 16 / 1.8e9
    # TODO: Check if other values commented out can be removed
    INTER_ELEMENT_DEADTIME = 8 / 1.8e9 # 80 / 2.4e9 # 0 / 2.4e9
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.075, 1.5),
    }
    CHANNEL_OFFSET_BOUNDS = {
        "analog": (-1.5, 1.5),
    }

    _uhf_sequence_string_template = (
        "const WINT_EN   = 0x03ff0000;\n"
        "const WINT_TRIG = 0x00000010;\n"
        "const IAVG_TRIG = 0x00000020;\n"
        "var RO_TRIG;\n"
        "if (getUserReg(1)) {{\n"
        "  RO_TRIG = WINT_EN + IAVG_TRIG;\n"
        "}} else {{\n"
        "  RO_TRIG = WINT_EN + WINT_TRIG;\n"
        "}}\n"
        "setTrigger(WINT_EN);\n"
        "\n"
        "{wave_definitions}\n"
        "\n"
        "var loop_cnt = getUserReg(0);\n"
        "\n"
        "repeat (loop_cnt) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def create_awg_parameters(self, channel_name_map):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        pulsar.add_parameter(f"{name}_trigger_source",
                             initial_value="Dig1",
                             vals=vals.Enum("Dig1", "Dig2", "DIO"),
                             parameter_class=ManualParameter,
                             docstring="Defines for which trigger source the "
                                       "AWG should wait, before playing the "
                                       "next waveform. Allowed values are: "
                                       "'Dig1', 'Dig2', 'DIO'.")

        group = []
        for ch_nr in range(2):
            id = f"ch{ch_nr + 1}"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name)
            pulsar.channels.add(ch_name)
            group.append(ch_name)
        # all channels are considered as a single group
        for ch_name in group:
            pulsar.channel_groups[ch_name] = group

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        super().create_channel_parameters(id, ch_name, ch_type)

        # TODO: Other AWGs do not define an initial value for these. Check if
        # this is a good behavior
        self.pulsar.parameters[f"{ch_name}_amp"].set(0.75)
        self.pulsar.parameters[f"{ch_name}_offset"].set(0)

    def awg_setter(self, id:str, param:str, value):

        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "offset":
            self.awg.set(f"sigouts_{ch}_offset", value)
        elif param == "amp":
            # TODO: Range is divided by 2 in getter, should it be multiplied by
            # 2 here ?
            self.awg.set(f"sigouts_{ch}_range", value)

    def awg_getter(self, id:str, param:str):

        # Sanity checks
        super().awg_getter(id, param)

        ch = int(id[2]) - 1

        if param == "offset":
            return self.awg.get(f"sigouts_{ch}_offset")
        elif param == "amp":
            if self.pulsar.awgs_prequeried:
                return self.awg.parameters[f"sigouts_{ch}_range"].get_latest() / 2
            else:
                return self.awg.get(f"sigouts_{ch}_range") / 2

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):

        if not self.zi_waves_cleared:
            self._zi_clear_waves()

        waves_to_upload = {h: waveforms[h]
                               for codewords in awg_sequence.values()
                                   if codewords is not None
                               for cw, chids in codewords.items()
                                   if cw != 'metadata'
                               for h in chids.values()}
        self._zi_write_waves(waves_to_upload)

        defined_waves = set()
        wave_definitions = []
        playback_strings = []

        use_filter = any([e is not None and
                          e.get('metadata', {}).get('allow_filter', False)
                          for e in awg_sequence.values()])
        if use_filter:
            wave_definitions += [
                f'var first_seg = getUserReg({obj.USER_REG_FIRST_SEGMENT});',
                f'var last_seg = getUserReg({obj.USER_REG_LAST_SEGMENT});',
            ]

        ch_has_waveforms = {'ch1': False, 'ch2': False}

        current_segment = 'no_segment'

        def play_element(element, playback_strings, wave_definitions,
                         allow_filter=True):
            awg_sequence_element = deepcopy(awg_sequence[element])
            if awg_sequence_element is None:
                current_segment = element
                playback_strings.append(f'// Segment {current_segment}')
                if use_filter:
                    playback_strings.append('i_seg += 1;')
                return playback_strings, wave_definitions
            playback_strings.append(f'// Element {element}')

            metadata = awg_sequence_element.pop('metadata', {})
            # The following line only has an effect if the metadata specifies
            # that the segment should be repeated multiple times.
            playback_strings += self._zi_playback_string_loop_start(
                metadata, ['ch1', 'ch2'])
            if list(awg_sequence_element.keys()) != ['no_codeword']:
                raise NotImplementedError('UHFQC sequencer does currently\
                                                       not support codewords!')
            chid_to_hash = awg_sequence_element['no_codeword']

            wave = (chid_to_hash.get('ch1', None), None,
                    chid_to_hash.get('ch2', None), None)
            wave_definitions += self._zi_wave_definition(wave,
                                                         defined_waves)

            acq = metadata.get('acq', False)
            # Remark on allow_filter in the call to _zi_playback_string:
            # the element may be skipped via segment filtering only if
            # play_element was called with allow_filter=True *and* the
            # element metadata allows segment filtering. (Use case for
            # calling play_element with allow_filter=False: repeat patterns,
            # see below.)
            playback_strings += self._zi_playback_string(
                name=obj.name, device='uhf', wave=wave, acq=acq,
                allow_filter=(
                        allow_filter and metadata.get('allow_filter', False)))
            # The following line only has an effect if the metadata specifies
            # that the segment should be repeated multiple times.
            playback_strings += self._zi_playback_string_loop_end(metadata)

            ch_has_waveforms['ch1'] |= wave[0] is not None
            ch_has_waveforms['ch2'] |= wave[2] is not None
            return playback_strings, wave_definitions

        self._filter_segment_functions[obj.name] = None
        if repeat_pattern is None:
            if use_filter:
                playback_strings += ['var i_seg = -1;']
            for element in awg_sequence:
                playback_strings, wave_definitions = play_element(element,
                                                                  playback_strings,
                                                                  wave_definitions)
        else:
            real_indicies = []
            # The allow_filter dict counts for each segment how many elements
            # in the segment have allow_filter set in the metadata.
            # (Note: it will be either True for all elements of the segment
            # or False for all elements of the segment. So the value either
            # equals the number of elements in the segment or 0 in the end.)
            allow_filter = {}
            seg_indices = []
            for index, element in enumerate(awg_sequence):
                if awg_sequence[element] is not None:  # element
                    real_indicies.append(index)
                    metadata = awg_sequence[element].get('metadata', {})
                    if metadata.get('allow_filter', False):
                        allow_filter[seg_indices[-1]] += 1
                else:  # segment
                    seg_indices.append(index)
                    allow_filter[index] = 0
            el_total = len(real_indicies)
            if any(allow_filter.values()):
                if repeat_pattern[1] != 1:
                    raise NotImplementedError(
                        'Element filtering with nested repeat patterns is not'
                        'implemented.')
                n_filter_elements = np.unique(
                    [f for f in allow_filter.values() if f > 0])
                if len(n_filter_elements) > 1:
                    raise NotImplementedError(
                        'Element filtering with repeat patterns requires '
                        'the same number elements in all segments that can '
                        'be filtered.')

                # Tell _set_filter_segments how to calculate the number of
                # acquisition for given indeces of first and last segment.
                def filter_count(first_seg, last_seg, n_tot=repeat_pattern[0],
                                 allow_filter=allow_filter):
                    for i, cnt in enumerate(allow_filter.values()):
                        if cnt == 0:
                            continue
                        if i < first_seg or i > last_seg:
                            n_tot -= cnt
                    return n_tot
                self._filter_segment_functions[obj.name] = filter_count
                # _set_filter_segments will pass the correct number of
                # repetitions via a user register to the SeqC variable last_seg
                repeat_pattern = ('last_seg', 1)

            def repeat_func(n, el_played, index, playback_strings,
                            wave_definitions):
                """
                Helper function to resolve a repeat pattern. It can call
                itself recursively to resolve nested pattern.

                :param n: a repeat pattern or an integer that specifies the
                    number of elements inside a loop, see pattern in the
                    docstring of Sequence.repeat
                :param el_played: helper variable for recursive function
                    calls to keep track of the number of elements that will
                    be played according with the resolved pattern.
                :param index: helper variable for recursive function
                    calls to keep track of the index of the next element
                    to be added.
                :param playback_strings: list of str to which the newly
                    generated SeqC code lines will be appended
                :param wave_definitions: list of wave definitions to which
                    the new wave definitions will be appended
                """
                if isinstance(n, tuple):  # repeat pattern definition
                    el_played_list = []
                    if isinstance(n[0], str):
                        # number of repetitions specified by a SeqC variable
                        playback_strings.append(
                            f'for (var i_rep = 0; i_rep < {n[0]}; '
                            f'i_rep += 1) {{')
                    elif n[0] > 1:
                        # interpret as integer number of repetitions
                        playback_strings.append('repeat ('+str(n[0])+') {')
                    for t in n[1:]:
                        el_cnt, playback_strings, wave_definitions = \
                            repeat_func(t, el_played,
                                        index + np.sum(el_played_list),
                                        playback_strings, wave_definitions)
                        el_played_list.append(el_cnt)
                    if isinstance(n[0], str) or n[0] > 1:
                        # A loop was started above. End it here.
                        playback_strings.append('}')
                    if isinstance(n[0], str):
                        # For variable numbers of repetitions, counting the
                        # number of played elements does not work and we
                        # just return that it is a variable number.
                        return 'variable', playback_strings, wave_definitions
                    return int(n[0] * np.sum(el_played_list)), playback_strings, wave_definitions
                else:  # n is the number of elements inside a loop
                    for k in range(n):
                        # Get the element that is meant to be played repeatedly
                        el_index = real_indicies[int(index)+k]
                        element = list(awg_sequence.keys())[el_index]
                        # Pass allow_filter=False since segment filtering is
                        # already covered by the repeat pattern if needed.
                        playback_strings, wave_definitions = play_element(
                            element, playback_strings, wave_definitions,
                            allow_filter=False)
                        el_played = el_played + 1
                    return el_played, playback_strings, wave_definitions



            el_played, playback_strings, wave_definitions = repeat_func(repeat_pattern, 0, 0,
                                                  playback_strings, wave_definitions)


            if el_played != 'variable' and int(el_played) != int(el_total):
                log.error(el_played, ' is not ', el_total)
                raise ValueError('Check number of sequences in repeat pattern')


        if not (ch_has_waveforms['ch1'] or ch_has_waveforms['ch2']):
            return
        self.pulsar.add_awg_with_waveforms(obj.name)

        awg_str = self._uhf_sequence_string_template.format(
            wave_definitions='\n'.join(wave_definitions),
            playback_string='\n  '.join(playback_strings),
        )

        # Necessary hack to pass the UHFQC drivers sanity check
        # in acquisition_initialize()
        obj._awg_program_features['loop_cnt'] = True
        obj._awg_program_features['avg_cnt']  = False
        # Hack needed to have
        obj._awg_needs_configuration[0] = False
        obj._awg_program[0] = True

        obj.configure_awg_from_string(awg_nr=0, program_string=awg_str, timeout=600)

    def _is_awg_running(self, obj):
        return obj.awgs_0_enable() != 0

    def clock(self):
        return self.awg.clock_freq()

    def get_segment_filter_userregs(self):
        return [(f"awgs_0_userregs_{self.awg.USER_REG_FIRST_SEGMENT}",
                 f"awgs_0_userregs_{self.awg.USER_REG_LAST_SEGMENT}")]

    def sigout_on(self, ch, on=True):
        """Turn channel outputs on or off."""

        awg = self.find_instrument(self.get(ch + '_awg'))
        chid = self.get(ch + '_id')
        awg.set('sigouts_{}_on'.format(int(chid[-1]) - 1), on)
