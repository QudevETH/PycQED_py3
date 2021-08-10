# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015
# Modified by Ants Remm 5/2017
# Modified by Michael Kerschbaum 5/2019
import os
import shutil
import ctypes
import numpy as np
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
import qcodes.utils.validators as vals
import pycqed.utilities.general as gen
import time
from copy import deepcopy

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
from pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 import \
    VirtualAWG8
# exception catching removed because it does not work in python versions before
# 3.6
try:
    from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
except Exception:
    Tektronix_AWG5014 = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
        UHFQuantumController import UHFQC
except Exception:
    UHFQC = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_HDAWG8 import ZI_HDAWG8
except Exception:
    ZI_HDAWG8 = type(None)

try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_base_instrument import merge_waveforms
except Exception:
    pass

log = logging.getLogger(__name__)

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
    dummy_UHFQC import dummy_UHFQC

class UHFQCPulsar:
    """
    Defines the Zurich Instruments UHFQC specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (UHFQC, dummy_UHFQC)
    
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

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, UHFQCPulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        name = awg.name

        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_enforce_single_element'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Group all the pulses on this AWG into "
                                     "a single element. Useful for making sure "
                                     "that the master AWG has only one waveform"
                                     " per segment.")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 16)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=8/(1.8e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           get_cmd=lambda: 16 /(1.8e9))
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           # get_cmd=lambda: 80 / 2.4e9)
                           get_cmd=lambda: 8 / (1.8e9))
                           # get_cmd=lambda: 0 / 2.4e9)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, vals=vals.Bool(),
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_delay'.format(awg.name), 
                           initial_value=0, label='{} delay'.format(name), 
                           unit='s', parameter_class=ManualParameter,
                           docstring='Global delay applied to this '
                                     'channel. Positive values move pulses'
                                     ' on this channel forward in time')
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=[],
                           label='{} trigger channel'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name),
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_trigger_source'.format(awg.name), 
                           initial_value='Dig1',
                           vals=vals.Enum('Dig1', 'Dig2', 'DIO'),
                           parameter_class=ManualParameter, 
                           docstring='Defines for which trigger source \
                                      the AWG should wait, before playing \
                                      the next waveform. Allowed values \
                                      are: "Dig1", "Dig2", "DIO"')

        group = []
        for ch_nr in range(2):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._uhfqc_create_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
        # all channels are considered as a single group
        for name in group:
            self.channel_groups[name] = group

    def _uhfqc_create_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._uhfqc_setter(awg, id, 'amp'),
                            get_cmd=self._uhfqc_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.075, 1.5),
                            initial_value=0.75)
        self.add_parameter('{}_offset'.format(name),
                            label='{} offset'.format(name), unit='V',
                            set_cmd=self._uhfqc_setter(awg, id, 'offset'),
                            get_cmd=self._uhfqc_getter(awg, id, 'offset'),
                            vals=vals.Numbers(-1.5, 1.5),
                            initial_value=0)
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
        self.add_parameter('{}_compensation_pulse_gaussian_filter_sigma'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

    @staticmethod
    def _uhfqc_setter(obj, id, par):
        if par == 'offset':
            def s(val):
                obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
        elif par == 'amp':
            def s(val):
                obj.set('sigouts_{}_range'.format(int(id[2])-1), val)
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _uhfqc_getter(self, obj, id, par):
        if par == 'offset':
            def g():
                return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
        elif par == 'amp':
            def g():
                if self._awgs_prequeried_state:
                    return obj.parameters['sigouts_{}_range' \
                        .format(int(id[2])-1)].get_latest()/2
                else:
                    return obj.get('sigouts_{}_range' \
                        .format(int(id[2])-1))/2
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g 

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms,
                                        repeat_pattern, **kw)

        if not self._zi_waves_cleared:
            _zi_clear_waves()
            self._zi_waves_cleared = True
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
                if isinstance(n, tuple):
                    el_played_list = []
                    if isinstance(n[0], str):
                        playback_strings.append(
                            f'for (var i_rep = 0; i_rep < {n[0]}; '
                            f'i_rep += 1) {{')
                    elif n[0] > 1:
                        playback_strings.append('repeat ('+str(n[0])+') {')
                    for t in n[1:]:
                        el_cnt, playback_strings, wave_definitions = \
                            repeat_func(t, el_played,
                                        index + np.sum(el_played_list),
                                        playback_strings, wave_definitions)
                        el_played_list.append(el_cnt)
                    if isinstance(n[0], str) or n[0] > 1:
                        playback_strings.append('}')
                    if isinstance(n[0], str):
                        return 'variable', playback_strings, wave_definitions
                    return int(n[0] * np.sum(el_played_list)), playback_strings, wave_definitions
                else:
                    for k in range(n):
                        el_index = real_indicies[int(index)+k]
                        element = list(awg_sequence.keys())[el_index]
                        # Pass allow_filter=False since segment filtering is
                        # already covered by the repeat pattern.
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
        self.awgs_with_waveforms(obj.name)
        
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
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)
        return obj.awgs_0_enable() != 0

    def _clock(self, obj, cid=None):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._clock(obj)
        return obj.clock_freq()

    def _get_segment_filter_userregs(self, obj):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._get_segment_filter_userregs(obj)
        return [(f'awgs_0_userregs_{UHFQC.USER_REG_FIRST_SEGMENT}',
                 f'awgs_0_userregs_{UHFQC.USER_REG_LAST_SEGMENT}')]

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        if not isinstance(awg, UHFQCPulsar._supportedAWGtypes):
            return super().sigout_on(ch, on)
        awg.set('sigouts_{}_on'.format(int(ch[-1]) - 1), on)


class HDAWG8Pulsar:
    """
    Defines the Zurich Instruments HDAWG8 specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (ZI_HDAWG8, VirtualAWG8, )

    _hdawg_sequence_string_template = (
        "{wave_definitions}\n"
        "\n"
        "{codeword_table_defs}\n"
        "\n"
        "while (1) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def __init__(self, name):
        super().__init__(name)
        self._hdawg_waveform_cache = dict()

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, HDAWG8Pulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        name = awg.name

        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_use_placeholder_waves'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_enforce_single_element'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Group all the pulses on this AWG into "
                                     "a single element. Useful for making sure "
                                     "that the master AWG has only one waveform"
                                     " per segment.")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 16)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=8/(2.4e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           initial_value=16 /(2.4e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           # get_cmd=lambda: 80 / 2.4e9)
                           get_cmd=lambda: 8 / (2.4e9))
                           # get_cmd=lambda: 0 / 2.4e9)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, vals=vals.Bool(),
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_delay'.format(awg.name), 
                           initial_value=0, label='{} delay'.format(name), 
                           unit='s', parameter_class=ManualParameter,
                           docstring='Global delay applied to this '
                                     'channel. Positive values move pulses'
                                     ' on this channel forward in time')
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=[],
                           label='{} trigger channel'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name),
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_trigger_source'.format(awg.name), 
                           initial_value='Dig1',
                           vals=vals.Enum('Dig1', 'DIO', 'ZSync'),
                           parameter_class=ManualParameter, 
                           docstring='Defines for which trigger source \
                                      the AWG should wait, before playing \
                                      the next waveform. Allowed values \
                                      are: "Dig1", "DIO", "ZSync"')
        self.add_parameter('{}_prepend_zeros'.format(awg.name),
                           initial_value=None,
                           vals=vals.MultiType(vals.Enum(None), vals.Ints(),
                                               vals.Lists(vals.Ints())),
                           parameter_class=ManualParameter)

        group = []
        for ch_nr in range(8):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._hdawg_create_analog_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            id = 'ch{}m'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._hdawg_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            # channel pairs plus the corresponding marker channels are
            # considered as groups
            if (ch_nr + 1) % 2 == 0:
                for name in group:
                    self.channel_groups.update({name: group})
                group = []

    def _hdawg_create_analog_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._hdawg_setter(awg, id, 'offset'),
                           get_cmd=self._hdawg_getter(awg, id, 'offset'),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._hdawg_setter(awg, id, 'amp'),
                            get_cmd=self._hdawg_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 5.0))
        self.add_parameter(
            '{}_amplitude_scaling'.format(name),
            set_cmd=self._hdawg_setter(awg, id, 'amplitude_scaling'),
            get_cmd=self._hdawg_getter(awg, id, 'amplitude_scaling'),
            vals=vals.Numbers(min_value=-1.0, max_value=1.0),
            initial_value=1.0)
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
        self.add_parameter('{}_compensation_pulse_gaussian_filter_sigma'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
        self.add_parameter('{}_internal_modulation'.format(name), 
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        if (int(id[2:]) - 1) % 2  == 0:  # first channel of a pair
            awg_nr = int((int(id[2:]) - 1) / 2)
            param_name = '{}_mod_freq'.format(name)
            self.add_parameter(param_name,
                               unit='Hz',
                               initial_value=None,
                               set_cmd=self._hdawg_mod_setter(awg, awg_nr),
                               get_cmd=self._hdawg_mod_getter(awg, awg_nr),
                               )
            # qcodes will not set the initial value if it is None, so we set
            # it manually here to ensure that internal modulation gets
            # switched off in the init.
            self.set(param_name, None)


    def _hdawg_create_marker_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'marker')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._hdawg_setter(awg, id, 'offset'),
                           get_cmd=self._hdawg_getter(awg, id, 'offset'),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._hdawg_setter(awg, id, 'amp'),
                            get_cmd=self._hdawg_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 5.0))
        
    @staticmethod
    def _hdawg_setter(obj, id, par):
        if par == 'offset':
            if id[-1] != 'm':
                def s(val):
                    obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
            else:
                s = None
        elif par == 'amp':
            if id[-1] != 'm':
                def s(val):
                    obj.set('sigouts_{}_range'.format(int(id[2])-1), 2*val)
            else:
                s = None
        elif par == 'amplitude_scaling' and id[-1] != 'm':
            awg = int((int(id[2:]) - 1) / 2)
            output = (int(id[2:]) - 1) - 2 * awg
            def s(val):
                obj.set(f'awgs_{awg}_outputs_{output}_amplitude', val)
                log.debug(f'awgs_{awg}_outputs_{output}_amplitude: {val}')
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _hdawg_getter(self, obj, id, par):
        if par == 'offset':
            if id[-1] != 'm':
                def g():
                    return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
            else:
                return lambda: 0
        elif par == 'amp':
            if id[-1] != 'm':
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['sigouts_{}_range' \
                            .format(int(id[2])-1)].get_latest()/2
                    else:
                        return obj.get('sigouts_{}_range' \
                            .format(int(id[2])-1))/2
            else:
                return lambda: 1
        elif par == 'amplitude_scaling' and id[-1] != 'm':
            awg = int((int(id[2:]) - 1) / 2)
            output = (int(id[2:]) - 1) - 2 * awg
            def g():
                return obj.get(f'awgs_{awg}_outputs_{output}_amplitude')
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    @staticmethod
    def _hdawg_mod_setter(obj, awg_nr):
        def s(val):
            log.debug(f'{obj.name}_awgs_{awg_nr} modulation freq: {val}')
            if val == None:
                obj.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 0)
                obj.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 0)
            else:
                # FIXME: this currently only works for real-valued baseband
                # signals (zero Q component), and it assumes that the the I
                # component gets programmed to both channels, see the case
                # of mod_frequency=None in
                # pulse_library.SSB_DRAG_pulse.chan_wf.
                # In the future, we should extended this to support general
                # IQ modulation and adapt the pulse library accordingly.
                # Also note that we here assume that the I (Q) channel is the
                # first (second) channel of a pair.
                sideband = np.sign(val)
                freq = np.abs(val)
                obj.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 1)
                obj.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 2)
                obj.set(f'sines_{awg_nr * 2}_oscselect', awg_nr * 4)
                obj.set(f'sines_{awg_nr * 2 + 1}_oscselect', awg_nr * 4)
                obj.set(f'sines_{awg_nr * 2}_phaseshift', 0)
                obj.set(f'sines_{awg_nr * 2 + 1}_phaseshift', sideband * 90)
                obj.set(f'oscs_{awg_nr * 4}_freq', freq)
        return s

    @staticmethod
    def _hdawg_mod_getter(obj, awg_nr):
        def g():
            m0 = obj.get(f'awgs_{awg_nr}_outputs_0_modulation_mode')
            m1 = obj.get(f'awgs_{awg_nr}_outputs_1_modulation_mode')
            if m0 == 0 and m1 == 0:
                return None
            elif m0 == 1 and m1 == 2:
                osc0 = obj.get(f'sines_{awg_nr * 2}_oscselect')
                osc1 = obj.get(f'sines_{awg_nr * 2 + 1}_oscselect')
                if osc0 == osc1:
                    sideband = np.sign(obj.get(
                        f'sines_{awg_nr * 2 + 1}_phaseshift'))
                    return sideband * obj.get(f'oscs_{osc0}_freq')
            log.warning('The current modulation configuration is not '
                        'supported by pulsar. Cannot retrieve modulation '
                        'frequency.')
            return None
        return g

    def get_divisor(self, chid, awg):
        '''
        Divisor is 1 for non modulated channels and 2 for modulated non 
        marker channels.
        '''

        if chid[-1]=='m':
            return 1

        name = self._id_channel(chid, awg)
        if self.get(f"{name}_internal_modulation"):
            return 2
        else: 
            return 1

    
    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     channels_to_upload='all', channels_to_program='all'):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms, repeat_pattern)

        chids = [f'ch{i+1}{m}' for i in range(8) for m in ['','m']]
        divisor = {chid: self.get_divisor(chid, obj.name) for chid in chids}
        def with_divisor(h, ch):
            return (h if divisor[ch] == 1 else (h, divisor[ch]))

        ch_has_waveforms = {chid: False for chid in chids}

        use_placeholder_waves = self.get(f'{obj.name}_use_placeholder_waves')

        if not use_placeholder_waves:
            if not self._zi_waves_cleared:
                _zi_clear_waves()
                self._zi_waves_cleared = True

        for awg_nr in self._hdawg_active_awgs(obj):
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
                    f'var first_seg = getUserReg({obj.USER_REG_FIRST_SEGMENT});',
                    f'var last_seg = getUserReg({obj.USER_REG_LAST_SEGMENT});',
                ]

            ch1id = 'ch{}'.format(awg_nr * 2 + 1)
            ch1mid = 'ch{}m'.format(awg_nr * 2 + 1)
            ch2id = 'ch{}'.format(awg_nr * 2 + 2)
            ch2mid = 'ch{}m'.format(awg_nr * 2 + 2)
            chids = [ch1id, ch1mid, ch2id, ch2mid]

            channels = [
                self._id_channel(chid, obj.name) for chid in [ch1id, ch2id]]
            if all([self.get(
                f'{chan}_internal_modulation') for chan in channels]):
                internal_mod = True
            elif not any([self.get(
                f'{chan}_internal_modulation') for chan in channels]):
                internal_mod = False
            else:
                raise NotImplementedError('Internal modulation can only be' 
                                          'specified per sub AWG!')

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
                    metadata, [ch1id, ch2id, ch1mid, ch2mid])

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
                        if wave == (None, None, None, None):
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
                                            f"{obj.name}, vawg{awg_nr}, "
                                            f"{current_segment}, {element}.")
                            wave_definitions += self._zi_wave_definition(
                                wave,
                                defined_waves,
                                max(placeholder_wave_lengths),
                                wave_idx_lookup[element][cw])
                        else:
                            wave = tuple(
                                with_divisor(h, chid) if h is not None
                                else None for h, chid in zip(wave, chids))
                            wave_definitions += self._zi_wave_definition(
                                wave, defined_waves)
                        
                        if nr_cw != 0:
                            w1, w2 = self._zi_waves_to_wavenames(wave)
                            if cw not in codeword_table:
                                codeword_table_defs += \
                                    self._zi_codeword_table_entry(
                                        cw, wave, use_placeholder_waves)
                                codeword_table[cw] = (w1, w2)
                            elif codeword_table[cw] != (w1, w2) \
                                    and self.reuse_waveforms():
                                log.warning('Same codeword used for different '
                                            'waveforms. Using first waveform. '
                                            f'Ignoring element {element}.')

                        ch_has_waveforms[ch1id] |= wave[0] is not None
                        ch_has_waveforms[ch1mid] |= wave[1] is not None
                        ch_has_waveforms[ch2id] |= wave[2] is not None
                        ch_has_waveforms[ch2mid] |= wave[3] is not None

                    if not internal_mod:
                        if first_element_of_segment:
                            prepend_zeros = self.parameters[
                                f'{obj.name}_prepend_zeros']()
                            if prepend_zeros is None:
                                prepend_zeros = self.prepend_zeros()
                            elif isinstance(prepend_zeros, list):
                                prepend_zeros = prepend_zeros[awg_nr]
                        else:
                            prepend_zeros = 0
                        playback_strings += self._zi_playback_string(
                            name=obj.name, device='hdawg', wave=wave,
                            codeword=(nr_cw != 0),
                            prepend_zeros=prepend_zeros,
                            placeholder_wave=use_placeholder_waves,
                            allow_filter=metadata.get('allow_filter', False))
                    elif not use_placeholder_waves:
                        pb_string, interleave_string = \
                            self._zi_interleaved_playback_string(name=obj.name, 
                            device='hdawg', counter=counter, wave=wave, 
                            codeword=(nr_cw != 0))
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
                # prevent ZI_base_instrument.start() from starting this sub AWG
                obj._awg_program[awg_nr] = None
                continue
            # tell ZI_base_instrument that it should not compile a
            # program on this sub AWG (because we already do it here)
            obj._awg_needs_configuration[awg_nr] = False
            # tell ZI_base_instrument.start() to start this sub AWG
            # (The base class will start sub AWGs for which _awg_program
            # is not None. Since we set _awg_needs_configuration to False,
            # we do not need to put the actual program here, but anything
            # different from None is sufficient.)
            obj._awg_program[awg_nr] = True

            # Having determined whether the sub AWG should be started or
            # not, we can now skip in case no channels need to be uploaded.
            if channels_to_upload != 'all' and not any(
                    [ch in channels_to_upload for ch in chids]):
                continue

            if not use_placeholder_waves:
                waves_to_upload = {with_divisor(h, chid):
                                   divisor[chid]*waveforms[h][::divisor[chid]]
                                   for codewords in awg_sequence.values()
                                       if codewords is not None
                                   for cw, chids in codewords.items()
                                       if cw != 'metadata'
                                   for chid, h in chids.items()}
                self._zi_write_waves(waves_to_upload)

            awg_str = self._hdawg_sequence_string_template.format(
                wave_definitions='\n'.join(wave_definitions+interleaves),
                codeword_table_defs='\n'.join(codeword_table_defs),
                playback_string='\n  '.join(playback_strings),
            )

            if not use_placeholder_waves or channels_to_program == 'all' or \
                    any([ch in channels_to_program for ch in chids]):
                run_compiler = True
            else:
                cached_lookup = self._hdawg_waveform_cache.get(
                    f'{obj.name}_{awg_nr}_wave_idx_lookup', None)
                try:
                    np.testing.assert_equal(wave_idx_lookup, cached_lookup)
                    run_compiler = False
                except AssertionError:
                    log.debug(f'{obj.name}_{awg_nr}: Waveform reuse pattern '
                              f'has changed. Forcing recompilation.')
                    run_compiler = True

            if run_compiler:
                # We have to retrieve the following parameter to set it
                # again after programming the AWG.
                prev_dio_valid_polarity = obj.get(
                    'awgs_{}_dio_valid_polarity'.format(awg_nr))

                obj.configure_awg_from_string(awg_nr, awg_str, timeout=600)

                obj.set('awgs_{}_dio_valid_polarity'.format(awg_nr),
                        prev_dio_valid_polarity)
                if use_placeholder_waves:
                    self._hdawg_waveform_cache[f'{obj.name}_{awg_nr}'] = {}
                    self._hdawg_waveform_cache[
                        f'{obj.name}_{awg_nr}_wave_idx_lookup'] = \
                        wave_idx_lookup

            if use_placeholder_waves:
                for idx, wave_hashes in defined_waves[1].items():
                    self._hdawg_update_waveforms(obj, awg_nr, idx,
                                                 wave_hashes, waveforms)

        if self.sigouts_on_after_programming():
            for ch in range(8):
                obj.set('sigouts_{}_on'.format(ch), True)

        if any(ch_has_waveforms.values()):
            self.awgs_with_waveforms(obj.name)

    def _hdawg_update_waveforms(self, obj, awg_nr, wave_idx, wave_hashes,
                                waveforms):
        if self.use_sequence_cache():
            if wave_hashes == self._hdawg_waveform_cache[
                    f'{obj.name}_{awg_nr}'].get(wave_idx, None):
                log.debug(
                    f'{obj.name} awgs{awg_nr}: {wave_idx} same as in cache')
                return
            log.debug(
                f'{obj.name} awgs{awg_nr}: {wave_idx} needs to be uploaded')
            self._hdawg_waveform_cache[f'{obj.name}_{awg_nr}'][
                wave_idx] = wave_hashes
        a1, m1, a2, m2 = [waveforms.get(h, None) for h in wave_hashes]
        n = max([len(w) for w in [a1, m1, a2, m2] if w is not None])
        if m1 is not None and a1 is None:
            a1 = np.zeros(n)
        if m1 is None and a1 is None and (m2 is not None or a2 is not None):
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
                mc = m1 + 4*m2
        else:
            mc = None
        a1 = None if a1 is None else np.pad(a1, n - a1.size)
        a2 = None if a2 is None else np.pad(a2, n - a2.size)
        wf_raw_combined = merge_waveforms(a1, a2, mc)
        obj.setv(f'awgs/{awg_nr}/waveform/waves/{wave_idx}', wf_raw_combined)

    def _is_awg_running(self, obj):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)

        return any([obj.get('awgs_{}_enable'.format(awg_nr)) for awg_nr in
                    self._hdawg_active_awgs(obj)])

    def _clock(self, obj, cid):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

    def _hdawg_active_awgs(self, obj):
        return [0,1,2,3]

    def _get_segment_filter_userregs(self, obj):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._get_segment_filter_userregs(obj)
        return [(f'awgs_{i}_userregs_{ZI_HDAWG8.USER_REG_FIRST_SEGMENT}',
                 f'awgs_{i}_userregs_{ZI_HDAWG8.USER_REG_LAST_SEGMENT}')
                for i in range(4) if obj._awg_program[i] is not None]

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        if not isinstance(awg, HDAWG8Pulsar._supportedAWGtypes):
            return super().sigout_on(ch, on)
        awg.set('sigouts_{}_on'.format(int(ch[-1]) - 1), on)

class AWG5014Pulsar:
    """
    Defines the Tektronix AWG5014 specific functionality for the Pulsar class
    """
    _supportedAWGtypes = (Tektronix_AWG5014, VirtualAWG5014, )

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, AWG5014Pulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_enforce_single_element'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Group all the pulses on this AWG into "
                                     "a single element. Useful for making sure "
                                     "that the master AWG has only one waveform"
                                     " per segment.")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 4)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=4/(1.2e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           get_cmd=lambda: 256/(1.2e9)) # Can not be triggered 
                                                        # faster than 210 ns.
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           get_cmd=lambda: 0)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, 
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('{}_delay'.format(awg.name), initial_value=0,
                           label='{} delay'.format(awg.name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in  time")
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=[],
                           label='{} trigger channels'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name), 
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(awg.name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

        group = []
        for ch_nr in range(4):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_analog_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            id = 'ch{}m1'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
            id = 'ch{}m2'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            group.append(name)
        # all channels are considered as a single group
        for name in group:
            self.channel_groups.update({name: group})

    def _awg5014_create_analog_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_offset_mode'.format(name), 
                           parameter_class=ManualParameter, 
                           vals=vals.Enum('software', 'hardware'))
        offset_mode_func = self.parameters['{}_offset_mode'.format(name)]
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset', 
                                                        offset_mode_func),
                           get_cmd=self._awg5014_getter(awg, id, 'offset', 
                                                        offset_mode_func),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 2.25))
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
        self.add_parameter('{}_compensation_pulse_gaussian_filter_sigma'.format(name),
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
    
    def _awg5014_create_marker_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'marker')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset'),
                           get_cmd=self._awg5014_getter(awg, id, 'offset'),
                           vals=vals.Numbers(-2.7, 2.7))
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(-5.4, 5.4))

    @staticmethod
    def _awg5014_setter(obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def s(val):
                    if offset_mode_func() == 'software':
                        obj.set('{}_offset'.format(id), val)
                    elif offset_mode_func() == 'hardware':
                        obj.set('{}_DC_out'.format(id), val)
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                        '{}'.format(offset_mode_func()))
            elif par == 'amp':
                def s(val):
                    obj.set('{}_amp'.format(id), 2*val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def s(val):
                    h = obj.get('{}_high'.format(id_raw))
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), val + h - l)
                    obj.set('{}_low'.format(id_raw), val)
            elif par == 'amp':
                def s(val):
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), l + val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _awg5014_getter(self, obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def g():
                    if offset_mode_func() == 'software':
                        return obj.get('{}_offset'.format(id))
                    elif offset_mode_func() == 'hardware':
                        return obj.get('{}_DC_out'.format(id))
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                         '{}'.format(offset_mode_func()))
                                    
            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['{}_amp'.format(id)] \
                                   .get_latest()/2
                    else:
                        return obj.get('{}_amp'.format(id))/2
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def g():
                    return obj.get('{}_low'.format(id_raw))
            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        h = obj.get('{}_high'.format(id_raw))
                        l = obj.get('{}_low'.format(id_raw))
                    else:
                        h = obj.parameters['{}_high'.format(id_raw)]\
                            .get_latest()
                        l = obj.parameters['{}_low'.format(id_raw)]\
                            .get_latest()
                    return h - l
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms,
                                        repeat_pattern, **kw)

        pars = {
            'ch{}_m{}_low'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_m{}_high'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_offset'.format(ch + 1) for ch in range(4)
        }
        old_vals = {}
        for par in pars:
            old_vals[par] = obj.get(par)

        packed_waveforms = {}
        wfname_l = []

        grp_has_waveforms = {f'ch{i+1}': False for i in range(4)}

        for element in awg_sequence:
            if awg_sequence[element] is None:
                continue
            metadata = awg_sequence[element].pop('metadata', {})
            if list(awg_sequence[element].keys()) != ['no_codeword']:
                raise NotImplementedError('AWG5014 sequencer does '
                                          'not support codewords!')
            chid_to_hash = awg_sequence[element]['no_codeword']

            if not any(chid_to_hash):
                continue  # no waveforms
            
            maxlen = max([len(waveforms[h]) for h in chid_to_hash.values()])
            maxlen = max(maxlen, 256)

            wfname_l.append([])
            for grp in [f'ch{i + 1}' for i in range(4)]:
                wave = (chid_to_hash.get(grp, None),
                        chid_to_hash.get(grp + 'm1', None), 
                        chid_to_hash.get(grp + 'm2', None))
                grp_has_waveforms[grp] |= (wave != (None, None, None))
                wfname = self._hash_to_wavename((maxlen, wave))
                grp_wfs = [np.pad(waveforms.get(h, [0]), 
                                  (0, maxlen - len(waveforms.get(h, [0]))), 
                                  'constant', constant_values=0) for h in wave]
                packed_waveforms[wfname] = obj.pack_waveform(*grp_wfs)
                wfname_l[-1].append(wfname)
                if any([wf[0] != 0 for wf in grp_wfs]):
                    log.warning(f'Element {element} starts with non-zero ' 
                                f'entry on {obj.name}.')

        if not any(grp_has_waveforms.values()):
            for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
                obj.set('{}_state'.format(grp), grp_has_waveforms[grp])
            return None

        self.awgs_with_waveforms(obj.name)

        nrep_l = [1] * len(wfname_l)
        goto_l = [0] * len(wfname_l)
        goto_l[-1] = 1
        wait_l = [1] * len(wfname_l)
        logic_jump_l = [0] * len(wfname_l)

        filename = 'pycqed_pulsar.awg'

        awg_file = obj.generate_awg_file(packed_waveforms, np.array(wfname_l).transpose().copy(),
                                         nrep_l, wait_l, goto_l, logic_jump_l,
                                         self._awg5014_chan_cfg(obj.name))
        obj.send_awg_file(filename, awg_file)
        obj.load_awg_file(filename)

        for par in pars:
            obj.set(par, old_vals[par])

        time.sleep(.1)
        # Waits for AWG to be ready
        obj.is_awg_ready()

        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            obj.set('{}_state'.format(grp), 1*grp_has_waveforms[grp])

        hardware_offsets = 0
        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            cname = self._id_channel(grp, obj.name)
            offset_mode = self.get('{}_offset_mode'.format(cname))
            if offset_mode == 'hardware':
                hardware_offsets = 1
            obj.DC_output(hardware_offsets)

        return awg_file

    def _is_awg_running(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)

        return obj.get_state() != 'Idle'

    def _clock(self, obj, cid=None):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

    @staticmethod
    def _awg5014_group_ids(cid):
        """
        Returns all id-s corresponding to a single channel group.
        For example `Pulsar._awg5014_group_ids('ch2')` returns `['ch2',
        'ch2m1', 'ch2m2']`.

        Args:
            cid: An id of one of the AWG5014 channels.

        Returns: A list of id-s corresponding to the same group as `cid`.
        """
        return [cid[:3], cid[:3] + 'm1', cid[:3] + 'm2'] 

    def _awg5014_chan_cfg(self, awg):
        channel_cfg = {}
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            cid = self.get('{}_id'.format(channel))
            amp = self.get('{}_amp'.format(channel))
            off = self.get('{}_offset'.format(channel))
            if self.get('{}_type'.format(channel)) == 'analog':
                offset_mode = self.get('{}_offset_mode'.format(channel))
                channel_cfg['ANALOG_METHOD_' + cid[2]] = 1
                channel_cfg['ANALOG_AMPLITUDE_' + cid[2]] = amp * 2
                if offset_mode == 'software':
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = off
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = 0
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 0
                else:
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = 0
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = off
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 1
            else:
                channel_cfg['MARKER1_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER2_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER{}_LOW_{}'.format(cid[-1], cid[2])] = \
                    off
                channel_cfg['MARKER{}_HIGH_{}'.format(cid[-1], cid[2])] = \
                    off + amp
            channel_cfg['CHANNEL_STATE_' + cid[2]] = 0

        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            if self.get('{}_active'.format(awg)):
                cid = self.get('{}_id'.format(channel))
                channel_cfg['CHANNEL_STATE_' + cid[2]] = 1
        return channel_cfg

    def _get_segment_filter_userregs(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._get_segment_filter_userregs(obj)
        return []

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        if not isinstance(awg, AWG5014Pulsar._supportedAWGtypes):
            return super().sigout_on(ch, on)
        awg.set(f"{self.get(ch + '_id')}_state", on)
        return

class Pulsar(AWG5014Pulsar, HDAWG8Pulsar, UHFQCPulsar, Instrument):
    """
    A meta-instrument responsible for all communication with the AWGs.
    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming and changing the parameters of the AWGs
    should be done through Pulsar. Supports Tektronix AWG5014 and partially
    ZI UHFLI.

    Args:
        master_awg: Name of the AWG that triggers all the other AWG-s and
                    should be started last (after other AWG-s are already
                    waiting for a trigger.
    """
    def __init__(self, name='Pulsar', master_awg=None):
        super().__init__(name)

        self._sequence_cache = dict()
        self.reset_sequence_cache()

        self.add_parameter('master_awg',
                           parameter_class=InstrumentRefParameter,
                           initial_value=master_awg)
        self.add_parameter('inter_element_spacing',
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum('auto')),
                           set_cmd=self._set_inter_element_spacing,
                           get_cmd=self._get_inter_element_spacing)
        self.add_parameter('reuse_waveforms', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('use_sequence_cache', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool(),
                           set_parser=self._use_sequence_cache_parser)
        self.add_parameter('prepend_zeros', initial_value=0, vals=vals.Ints(),
                           parameter_class=ManualParameter)
        self.add_parameter('flux_crosstalk_cancellation', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('flux_channels', initial_value=[],
                           parameter_class=ManualParameter, vals=vals.Lists())
        self.add_parameter('flux_crosstalk_cancellation_mtx',
                           initial_value=None, parameter_class=ManualParameter)
        self.add_parameter('flux_crosstalk_cancellation_shift_mtx',
                           initial_value=None, parameter_class=ManualParameter)
        # This parameter can be used to record only a specified consecutive
        # subset of segments of a programmed hard sweep. This is used by the
        # sweep function FilteredSweep. The parameter expects a tuple of indices
        # indicating the first and the last segment to be measured. (Segments
        # with the property allow_filter set to False are always measured.)
        self.add_parameter('filter_segments',
                           set_cmd=self._set_filter_segments,
                           get_cmd=self._get_filter_segments,
                           initial_value=None)
        self.add_parameter('sigouts_on_after_programming', initial_value=True,
                           parameter_class=ManualParameter, vals=vals.Bool(),
                           docstring='Whether signal outputs should be '
                                     'switched off automatically after '
                                     'programming a AWGs. Can be set to '
                                     'False to save time if it is ensured '
                                     'that the channels are switched on '
                                     'somewhere else.')
        self._inter_element_spacing = 'auto'
        self.channels = set() # channel names
        self.awgs = set() # AWG names
        self.last_sequence = None
        self.last_elements = None
        self._awgs_with_waveforms = set()
        self.channel_groups = {}

        self._awgs_prequeried_state = False

        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}
        self._filter_segments = None
        self._filter_segment_functions = {}

        self.num_seg = 0

        Pulsar._instance = self

    @staticmethod
    def get_instance():
        return Pulsar._instance

    def _use_sequence_cache_parser(self, val):
        if val and not self.use_sequence_cache():
            self.reset_sequence_cache()
        return val

    def reset_sequence_cache(self):
        self._sequence_cache = {}
        # The following dicts are used in _program_awgs to store information
        # about the last sequence programmed to each AWGs. The keys of the
        # dicts are AWG names and/or channel names. See the code and
        # comments of _program_awgs for details about the structure of the
        # dicts.
        self._sequence_cache['settings'] = {}  # for pulsar settings
        self._sequence_cache['metadata'] = {}  # for segment/element metadata
        self._sequence_cache['hashes'] = {}  # for waveform hashes
        self._sequence_cache['length'] = {}  # for element lengths

    def check_for_other_pulsar(self):
        """
        Checks whether another pulsar has programmed the AWGs and resets the
        sequence cache if this is the case. To make this check possible,
        the pulsar object ID is written to a file in the pycqed app data dir.
        """
        filename = os.path.join(gen.get_pycqed_appdata_dir(), 'pulsar_id')
        current_id = f"{id(self)}"
        try:
            with open(filename, 'r') as f:
                stored_id = f.read()
        except:
            stored_id = None
        if stored_id != current_id:
            log.debug('Another pulsar instance has programmed the AWGs. '
                      'Resetting sequence cache.')
            self.reset_sequence_cache()
        with open(filename, 'w') as f:
            f.write(current_id)

    # channel handling
    def define_awg_channels(self, awg, channel_name_map=None):
        """
        The AWG object must be created before creating channels for that AWG

        Args:
            awg: AWG object to add to the pulsar.
            channel_name_map: A dictionary that maps channel ids to channel
                              names. (default {})
        """
        if channel_name_map is None:
            channel_name_map = {}

        for channel_name in channel_name_map.values():
            if channel_name in self.channels:
                raise KeyError("Channel named '{}' already defined".format(
                    channel_name))
        if awg.name in self.awgs:
            raise KeyError("AWG '{}' already added to pulsar".format(awg.name))

        fail = None
        super()._create_awg_parameters(awg, channel_name_map)
        # try:
        #     super()._create_awg_parameters(awg, channel_name_map)
        # except AttributeError as e:
        #     fail = e
        # if fail is not None:
        #     raise TypeError('Unsupported AWG instrument: {}. '
        #                     .format(awg.name) + str(fail))
        
        self.awgs.add(awg.name)
        # Make sure that the registers for filter_segments are set in the
        # new AWG.
        self.filter_segments(self.filter_segments())

    def find_awg_channels(self, awg):
        channel_list = []
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) == awg:
                channel_list.append(channel)

        return channel_list

    def AWG_obj(self, **kw):
        """
        Return the AWG object corresponding to a channel or an AWG name.

        Args:
            awg: Name of the AWG Instrument.
            channel: Name of the channel

        Returns: An instance of Instrument class corresponding to the AWG
                 requested.
        """
        awg = kw.get('awg', None)
        chan = kw.get('channel', None)
        if awg is not None and chan is not None:
            raise ValueError('Both `awg` and `channel` arguments passed to '
                             'Pulsar.AWG_obj()')
        elif awg is None and chan is not None:
            name = self.get('{}_awg'.format(chan))
        elif awg is not None and chan is None:
            name = awg
        else:
            raise ValueError('Either `awg` or `channel` argument needs to be '
                             'passed to Pulsar.AWG_obj()')
        return Instrument.find_instrument(name)

    def clock(self, channel=None, awg=None):
        """
        Returns the clock rate of channel or AWG 'instrument_ref' 
        Args:
            isntrument_ref: name of the channel or AWG
        Returns: clock rate in samples per second
        """
        if channel is not None and awg is not None:
            raise ValueError('Both channel and awg arguments passed to '
                             'Pulsar.clock()')
        if channel is None and awg is None:
            raise ValueError('Neither channel nor awg arguments passed to '
                             'Pulsar.clock()')

        if channel is not None:
            awg = self.get('{}_awg'.format(channel))
     
        if self._awgs_prequeried_state:
            return self._clocks[awg]
        else:
            fail = None
            obj = self.AWG_obj(awg=awg)
            try:
                return super()._clock(obj)
            except AttributeError as e:
                fail = e
            if fail is not None:
                raise TypeError('Unsupported AWG instrument: {} of type {}. '
                                .format(obj.name, type(obj)) + str(fail))

    def active_awgs(self):
        """
        Returns:
            A set of the names of the active AWGs registered

            Inactive AWGs don't get started or stopped. Also the waveforms on
            inactive AWGs don't get updated.
        """
        return {awg for awg in self.awgs if self.get('{}_active'.format(awg))}

    def awgs_with_waveforms(self, awg=None):
        """
        Adds an awg to the set of AWGs with waveforms programmed, or returns 
        set of said AWGs.
        """
        if awg == None:
            return self._awgs_with_waveforms
        else:
            self._awgs_with_waveforms.add(awg)
            self._set_filter_segments(self._filter_segments, [awg])

    def start(self, exclude=None, stop_first=True):
        """
        Start the active AWGs. If multiple AWGs are used in a setup where the
        slave AWGs are triggered by the master AWG, then the slave AWGs must be
        running and waiting for trigger when the master AWG is started to
        ensure synchronous playback.
        :param exclude: (list of str) names of AWGs to exclude
        :param stop_first: (bool, default: True) whether all used AWGs
            should be stopped before starting the AWGs.
        """
        if exclude is None:
            exclude = []

        # Start only the AWGs which have at least one channel programmed, i.e.
        # where at least one channel has state = 1. 
        awgs_with_waveforms = self.awgs_with_waveforms()
        used_awgs = set(self.active_awgs()) & awgs_with_waveforms
        
        if stop_first:
            for awg in used_awgs:
                self._stop_awg(awg)

        if self.master_awg() is None:
            for awg in used_awgs:
                if awg not in exclude:
                    self._start_awg(awg)
        else:
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().stop()
            for awg in used_awgs:
                if awg != self.master_awg() and awg not in exclude:
                    self._start_awg(awg)
            tstart = time.time()
            for awg in used_awgs:
                if awg == self.master_awg() or awg in exclude:
                    continue
                good = False
                while not (good or time.time() > tstart + 10):
                    if self._is_awg_running(awg):
                        good = True
                    else:
                        time.sleep(0.1)
                if not good:
                    raise Exception('AWG {} did not start in 10s'
                                    .format(awg))
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().start()

    def stop(self):
        """
        Stop all active AWGs.
        """

        awgs_with_waveforms = set(self.awgs_with_waveforms())
        used_awgs = set(self.active_awgs()) & awgs_with_waveforms

        for awg in used_awgs:
            self._stop_awg(awg)
    
    def program_awgs(self, sequence, awgs='all'):
        try:
            self._program_awgs(sequence, awgs)
        except Exception as e:
            log.warning(f'Pulsar: Exception {repr(e)} while programming AWGs. '
                        f'Retrying after resetting the sequence cache.')
            self.reset_sequence_cache()
            self._program_awgs(sequence, awgs)

    def _program_awgs(self, sequence, awgs='all'):

        # Stores the last uploaded sequence for easy access and plotting
        self.last_sequence = sequence

        if awgs == 'all':
            awgs = self.active_awgs()

        # initializes the set of AWGs with waveforms
        self._awgs_with_waveforms -= awgs


        # prequery all AWG clock values and AWG amplitudes
        self.AWGs_prequeried(True)

        log.info(f'Starting compilation of sequence {sequence.name}')
        t0 = time.time()
        if self.use_sequence_cache():
            # reset the sequence cache if another pulsar instance has
            # programmed the AWGs
            self.check_for_other_pulsar()
            # get hashes and information about the sequence structure
            channel_hashes, awg_sequences = \
                sequence.generate_waveforms_sequences(get_channel_hashes=True)
            log.debug(f'End of waveform hashing sequence {sequence.name} '
                      f'{time.time() - t0}')
            sequence_cache = self._sequence_cache
            # The following makes sure that the sequence cache is empty if
            # the compilation crashes or gets interrupted.
            self.reset_sequence_cache()
            # Add an empty hash for previously active but now inactive channels
            # on active AWGs. This is to make sure that the change (switching
            # them off) is detected correctly below.
            channel_hashes.update({
                k: {} for k, v in sequence_cache['hashes'].items()
                if k not in channel_hashes and len(v)
                and self.get(f'{k}_awg') in awg_sequences.keys()})
            # first, we check whether programming the whole AWG is mandatory due
            # to changed AWG settings or due to changed metadata
            awgs_to_program = []
            settings_to_check = ['{}_use_placeholder_waves',
                                 '{}_prepend_zeros',
                                 'prepend_zeros']
            settings = {}
            metadata = {}
            for awg, seq in awg_sequences.items():
                settings[awg] = {
                    s.format(awg): (
                        self.get(s.format(awg))
                        if s.format(awg) in self.parameters else None)
                    for s in settings_to_check}
                metadata[awg] = {
                    elname: (
                        el.get('metadata', {}) if el is not None else None)
                    for elname, el in seq.items()}
                if awg not in awgs_to_program:
                    try:
                        np.testing.assert_equal(
                            sequence_cache['settings'].get(awg, {}),
                            settings[awg])
                        np.testing.assert_equal(
                            sequence_cache['metadata'].get(awg, {}),
                            metadata[awg])
                    except AssertionError:  # settings or metadata change
                        awgs_to_program.append(awg)
            for awg in awgs_to_program:
                # update the settings and metadata cache
                sequence_cache['settings'][awg] = settings[awg]
                sequence_cache['metadata'][awg] = metadata[awg]
            # Check for which channels some relevant setting or some hash has
            # changed, in which case the group of channels should be uploaded.
            settings_to_check = ['{}_internal_modulation']
            awgs_with_channels_to_upload = []
            channels_to_upload = []
            channels_to_program = []
            for ch, hashes in channel_hashes.items():
                ch_awg = self.get(f'{ch}_awg')
                settings[ch] = {
                    s.format(ch): (
                        self.get(s.format(ch))
                        if s.format(ch) in self.parameters else None)
                    for s in settings_to_check}
                metadata[ch] = {'repeat_pattern':
                                    sequence.repeat_patterns.get(ch, None)}
                if ch in channels_to_upload or ch_awg in awgs_to_program:
                    continue
                changed_settings = True
                try:
                    np.testing.assert_equal(
                        sequence_cache['settings'].get(ch, {}),
                        settings[ch])
                    changed_settings = False
                    np.testing.assert_equal(
                        sequence_cache['hashes'].get(ch, {}), hashes)
                    np.testing.assert_equal(
                        sequence_cache['metadata'].get(ch, {}), metadata[ch])
                except AssertionError:
                    # changed setting, sequence structure, or hash
                    if ch_awg not in awgs_with_channels_to_upload:
                        awgs_with_channels_to_upload.append(ch_awg)
                    for c in self.channel_groups[ch]:
                        channels_to_upload.append(c)
                        if changed_settings:
                            channels_to_program.append(c)
            # update the settings cache and hashes cache
            for ch in channels_to_upload:
                sequence_cache['settings'][ch] = settings.get(ch, {})
                sequence_cache['hashes'][ch] = channel_hashes.get(ch, {})
                sequence_cache['metadata'][ch] = metadata.get(ch, {})
            # generate the waveforms that we need for uploading
            log.debug(f'Start of waveform generation sequence {sequence.name} '
                     f'{time.time() - t0}')
            waveforms, _ = sequence.generate_waveforms_sequences(
                awgs_to_program + awgs_with_channels_to_upload,
                resolve_segments=False)
            log.debug(f'End of waveform generation sequence {sequence.name} '
                     f'{time.time() - t0}')
            # Check for which channels the sequence structure, or some element
            # length has changed.
            # If placeholder waveforms are used, only those channels (and
            # channels in the same group) will be re-programmed, while other
            # channels can be re-uploaded by replacing the existing waveforms.
            ch_length = {}
            for ch, hashes in channel_hashes.items():
                ch_awg = self.get(f'{ch}_awg')
                if ch_awg in awgs_to_program + awgs_with_channels_to_upload:
                    ch_length[ch] = {
                        elname: {cw: len(waveforms[h]) for cw, h in el.items()}
                        for elname, el in hashes.items()}
                # Checking whether programming is done only for channels that
                # are marked to be uploaded but not yet marked to be programmed
                if ch not in channels_to_upload or ch in channels_to_program \
                        or ch_awg in awgs_to_program:
                    continue
                try:
                    np.testing.assert_equal(
                        sequence_cache['length'].get(ch, {}),
                        ch_length[ch])
                except AssertionError:  # changed length or sequence structure
                    for c in self.channel_groups[ch]:
                        channels_to_program.append(c)
            # update the length cache
            for ch in channels_to_program:
                sequence_cache['length'][ch] = ch_length.get(ch, {})
            # Update the cache for channels that are on an AWG marked for
            # complete re-programming (these channels might have been skipped
            # above).
            for ch in self.channels:
                if self.get(f'{ch}_awg') in awgs_to_program:
                    sequence_cache['settings'][ch] = settings.get(ch, {})
                    sequence_cache['hashes'][ch] = channel_hashes.get(
                        ch, {})
                    sequence_cache['metadata'][ch] = metadata.get(ch, {})
                    sequence_cache['length'][ch] = ch_length.get(ch, {})
            log.debug(f'awgs_to_program = {repr(awgs_to_program)}\n'
                      f'awgs_with_channels_to_upload = '
                      f'{repr(awgs_with_channels_to_upload)}\n'
                      f'channels_to_upload = {repr(channels_to_upload)}\n'
                      f'channels_to_program = {repr(channels_to_program)}'
                      )
        else:
            waveforms, awg_sequences = sequence.generate_waveforms_sequences()
            awgs_to_program = list(awg_sequences.keys())
            awgs_with_channels_to_upload = []
        log.info(f'Finished compilation of sequence {sequence.name} in '
                 f'{time.time() - t0}')

        channels_used = self._channels_in_awg_sequences(awg_sequences)
        repeat_dict = self._generate_awg_repeat_dict(sequence.repeat_patterns,
                                                     channels_used)
        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}

        for awg in awg_sequences.keys():
            if awg not in awgs_to_program + awgs_with_channels_to_upload:
                # The AWG does not need to be re-programmed, but we have to add
                # it to the set of AWGs with waveforms (which is otherwise
                # done after programming it).
                self.awgs_with_waveforms(awg)
                continue
            log.info(f'Started programming {awg}')
            t0 = time.time()
            if awg in awgs_to_program:
                ch_upl, ch_prg = 'all', 'all'
            else:
                ch_upl = [self.get(f'{ch}_id') for ch in channels_to_upload
                          if self.get(f'{ch}_awg') == awg]
                ch_prg = [self.get(f'{ch}_id') for ch in channels_to_program
                          if self.get(f'{ch}_awg') == awg]
            if awg in repeat_dict.keys():
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms,
                                  repeat_pattern=repeat_dict[awg],
                                  channels_to_upload=ch_upl,
                                  channels_to_program=ch_prg)
            else:
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms,
                                  channels_to_upload=ch_upl,
                                  channels_to_program=ch_prg)
            log.info(f'Finished programming {awg} in {time.time() - t0}')

        if self.use_sequence_cache():
            # Compilation finished sucessfully. Store sequence cache.
            self._sequence_cache = sequence_cache
        self.num_seg = len(sequence.segments)
        self.AWGs_prequeried(False)

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):
        """
        Program the AWG with a sequence of segments.

        Args:
            obj: the instance of the AWG to program
            sequence: the `Sequence` object that determines the segment order,
                      repetition and trigger wait
            el_wfs: A dictionary from element name to a dictionary from channel
                    id to the waveform.
            loop: Boolean flag, whether the segments should be looped over.
                  Default is `True`.
        """
        # fail = None
        # try:
        #     super()._program_awg(obj, awg_sequence, waveforms)
        # except AttributeError as e:
        #     fail = e
        # if fail is not None:
        #     raise TypeError('Unsupported AWG instrument: {} of type {}. '
        #                     .format(obj.name, type(obj)) + str(fail))
        if repeat_pattern is not None:
            super()._program_awg(obj, awg_sequence, waveforms,
                                 repeat_pattern=repeat_pattern, **kw)
        else:
            super()._program_awg(obj, awg_sequence, waveforms, **kw)

    def _hash_to_wavename(self, h):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        if h not in self._hash_to_wavename_table:
            hash_int = abs(hash(h))
            wname = ''.join(to_base(hash_int, len(alphabet), alphabet))[::-1]
            while wname in self._hash_to_wavename_table.values():
                hash_int += 1
                wname = ''.join(to_base(hash_int, len(alphabet), alphabet)) \
                    [::-1]
            self._hash_to_wavename_table[h] = wname
        return self._hash_to_wavename_table[h]

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
            # don't use placeholder waves
            for analog, marker, wc in [(wave[0], wave[1], w1),
                                       (wave[2], wave[3], w2)]:
                if analog is not None:
                    wa = self._hash_to_wavename(analog)
                    if wa not in defined_waves:
                        wave_definition.append(f'wave {wa} = "{wa}";')
                        defined_waves.add(wa)
                if marker is not None:
                    wm = self._hash_to_wavename(marker)
                    if wm not in defined_waves:
                        wave_definition.append(f'wave {wm} = "{wm}";')
                        defined_waves.add(wm)
                if analog is not None and marker is not None:
                    if wc not in defined_waves:
                        wave_definition.append(f'wave {wc} = {wa} + {wm};')
                        defined_waves.add(wc)
        else:
            # use placeholder waves
            n = placeholder_wave_length
            if w1 is None and w2 is not None:
                w1 = f'{w2}_but_zero'
            for wc, marker in [(w1, wave[1]), (w2, wave[3])]:
                if wc is not None and wc not in defined_waves[0]:
                    wave_definition.append(
                        f'wave {wc} = placeholder({n}' +
                        ('' if marker is None else ', true') +
                        ');')
                    defined_waves[0].add(wc)
            wave_definition.append(
                f'assignWaveIndex({_zi_wavename_pair_to_argument(w1, w2)},'
                f' {placeholder_wave_index});'
            )
            defined_waves[1][placeholder_wave_index] = wave
        return wave_definition

    def _zi_playback_string(self, name, device, wave, acq=False, codeword=False,
                            prepend_zeros=0, placeholder_wave=False,
                            allow_filter=False):
        playback_string = []
        if allow_filter:
            playback_string.append(
                'if (i_seg >= first_seg && i_seg <= last_seg) {')
        if prepend_zeros:
            playback_string.append(f'playZero({prepend_zeros});')
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True # set this to false once the bugs with HDAWG are fixed
        trig_source = self.get('{}_trigger_source'.format(name))
        if trig_source == 'Dig1':
            playback_string.append(
                'waitDigTrigger(1{});'.format(', 1' if device == 'uhf' else ''))
        elif trig_source == 'Dig2':
            playback_string.append('waitDigTrigger(2,1);')
        else:
            playback_string.append(f'wait{trig_source}Trigger();')

        if codeword and not (w1 is None and w2 is None):
            playback_string.append('playWaveDIO();')
        else:
            if w1 is None and w2 is not None and use_hack and not placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave(marker(1,0)*0*{w2}, {w2});')
            elif w1 is None and w2 is not None and use_hack and placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave({w2}_but_zero, {w2});')
            elif w1 is not None and w2 is None and use_hack and not placeholder_wave:
                # This hack is needed due to a bug on the HDAWG.
                # Remove this if case once the bug is fixed.
                playback_string.append(f'playWave({w1}, marker(1,0)*0*{w1});')
            elif w1 is not None or w2 is not None:
                playback_string.append('playWave({});'.format(
                    _zi_wavename_pair_to_argument(w1, w2)))
        if acq:
            playback_string.append('setTrigger(RO_TRIG);')
            playback_string.append('setTrigger(WINT_EN);')
        if allow_filter:
            playback_string.append('}')
        return playback_string

    def _zi_interleaved_playback_string(self, name, device, counter, 
                                        wave, acq=False, codeword=False):
        playback_string = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if w1 is None or w2 is None:
            raise ValueError('When using HDAWG modulation both I and Q need '  
                              'to be defined')
        
        wname = f'wave{counter}'
        interleaves = [f'wave {wname} = interleave({w1}, {w2});']

        if not codeword:
            if not acq:
                playback_string.append(f'prefetch({wname},{wname});')
        
        trig_source = self.get('{}_trigger_source'.format(name))
        if trig_source == 'Dig1':
            playback_string.append(
                'waitDigTrigger(1{});'.format(', 1' if device == 'uhf' else ''))
        elif trig_source == 'Dig2':
            playback_string.append('waitDigTrigger(2,1);')
        else:
            playback_string.append(f'wait{trig_source}Trigger();')

        if codeword:
            # playback_string.append('playWaveDIO();')
            raise NotImplementedError('Modulation in combination with codeword'
                                      'pulses has not yet been implemented!')
        else:
            playback_string.append(f'playWave({wname},{wname});')
        if acq:
            playback_string.append('setTrigger(RO_TRIG);')
            playback_string.append('setTrigger(WINT_EN);')
        return playback_string, interleaves

    @staticmethod
    def _zi_playback_string_loop_start(metadata, channels):
        loop_len = metadata.get('loop', False)
        if not loop_len:
            return []
        playback_string = []
        sweep_params = metadata.get('sweep_params', {})
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f'{ch}_'):
                    playback_string.append(
                        f"wave {k} = vect({','.join([f'{a}' for a in v])})")
        playback_string.append(
            f"for (cvar i_sweep = 0; i_sweep < {loop_len}; i_sweep += 1) {{")
        for k, v in sweep_params.items():
            for ch in channels:
                if k.startswith(f'{ch}_'):
                    node = k[len(f'{ch}_'):].replace('_', '/')
                    playback_string.append(
                        f'setDouble("{node}", {k}[i_sweep]);')
        return playback_string

    @staticmethod
    def _zi_playback_string_loop_end(metadata):
        return ['}'] if metadata.get('end_loop', False) else []

    def _zi_codeword_table_entry(self, codeword, wave, placeholder_wave=False):
        w1, w2 = self._zi_waves_to_wavenames(wave)
        use_hack = True
        if w1 is None and w2 is not None and use_hack and not placeholder_wave:
            # This hack is needed due to a bug on the HDAWG. 
            # Remove this if case once the bug is fixed.
            return [f'setWaveDIO({codeword}, zeros(1) + marker(1, 0), {w2});']
        elif w1 is None and w2 is not None and use_hack and placeholder_wave:
            return [f'setWaveDIO({codeword}, {w2}_but_zero, {w2});']
        elif not (w1 is None and w2 is None):
            return ['setWaveDIO({}, {});'.format(codeword, 
                        _zi_wavename_pair_to_argument(w1, w2))]
        else:
            return []

    def _zi_waves_to_wavenames(self, wave):
        wavenames = []
        for analog, marker in [(wave[0], wave[1]), (wave[2], wave[3])]:
            if analog is None and marker is None:
                wavenames.append(None)
            elif analog is None and marker is not None:
                wavenames.append(self._hash_to_wavename(marker))
            elif analog is not None and marker is None:
                wavenames.append(self._hash_to_wavename(analog))
            else:
                wavenames.append(self._hash_to_wavename((analog, marker)))
        return wavenames

    def _zi_write_waves(self, waveforms):
        wave_dir = _zi_wave_dir()
        for h, wf in waveforms.items():
            filename = os.path.join(wave_dir, self._hash_to_wavename(h)+'.csv')
            if os.path.exists(filename):
                continue
            fmt = '%.18e' if wf.dtype == np.float else '%d'
            np.savetxt(filename, wf, delimiter=",", fmt=fmt)

    def _start_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.start()

    def _stop_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.stop()

    def _is_awg_running(self, awg):
        fail = None
        obj = self.AWG_obj(awg=awg)
        try:
            return super()._is_awg_running(obj)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))

    def _set_inter_element_spacing(self, val):
        self._inter_element_spacing = val

    def _get_inter_element_spacing(self):
        if self._inter_element_spacing != 'auto':
            return self._inter_element_spacing
        else:
            max_spacing = 0
            for awg in self.awgs:
                max_spacing = max(max_spacing, self.get(
                    '{}_inter_element_deadtime'.format(awg)))
            return max_spacing

    def _set_filter_segments(self, val, awgs='with_waveforms'):
        if val is None:
            val = (0, 32767)
        self._filter_segments = val
        if awgs == 'with_waveforms':
            awgs = self.awgs_with_waveforms()
        elif awgs == 'all':
            awgs = self.awgs
        for AWG_name in awgs:
            AWG = self.AWG_obj(awg=AWG_name)
            fnc = self._filter_segment_functions.get(AWG_name, None)
            if fnc is None:
                for regs in self._get_segment_filter_userregs(AWG):
                    AWG.set(regs[0], val[0])
                    AWG.set(regs[1], val[1])
            else:
                # used in case of a repeat pattern
                for regs in self._get_segment_filter_userregs(AWG):
                    AWG.set(regs[1], fnc(val[0], val[1]))

    def _get_filter_segments(self):
        return self._filter_segments

    def AWGs_prequeried(self, status=None):
        if status is None:
            return self._awgs_prequeried_state
        elif status:
            self._awgs_prequeried_state = False
            self._clocks = {}
            for awg in self.awgs:
                self._clocks[awg] = self.clock(awg=awg)
            for c in self.channels:
                # prequery also the output amplitude values
                self.get(c + '_amp')
            self._awgs_prequeried_state = True
        else:
            self._awgs_prequeried_state = False

    def _id_channel(self, cid, awg):
        """
        Returns the channel name corresponding to the channel with id `cid` on
        the AWG `awg`.

        Args:
            cid: An id of one of the channels.
            awg: The name of the AWG.

        Returns: The corresponding channel name. If the channel is not found,
                 returns `None`.
        """
        for cname in self.channels:
            if self.get('{}_awg'.format(cname)) == awg and \
               self.get('{}_id'.format(cname)) == cid:
                return cname
        return None

    @staticmethod
    def _channels_in_awg_sequences(awg_sequences):
        """
        identifies all channels used in the given awg keyed sequence
        :param awg_sequences (dict): awg sequences keyed by awg name, i.e. as
        returned by sequence.generate_sequence_waveforms()
        :return: dictionary keyed by awg of with all channel used during the sequence
        """
        channels_used = dict()
        for awg in awg_sequences:
            channels_used[awg] = set()
            for segname in awg_sequences[awg]:
                if awg_sequences[awg][segname] is None:
                    continue
                elements = awg_sequences[awg][segname]
                for cw in elements:
                    if cw != "metadata":
                        channels_used[awg] |= elements[cw].keys()
        return channels_used

    def _generate_awg_repeat_dict(self, repeat_dict_per_ch, channels_used):
        """
        Translates a repeat dictionary keyed by channels to a repeat dictionary
        keyed by awg. Checks whether all channels in channels_used have an entry.
        :param repeat_dict_per_ch: keys: channels_id, values: repeat pattern
        :param channels_used (dict): list of channel used on each awg
        :return:
        """
        awg_ch_repeat_dict = dict()
        repeat_dict_per_awg = dict()
        for cname in repeat_dict_per_ch:
            awg = self.get(f"{cname}_awg")
            chid = self.get(f"{cname}_id")

            if not awg in awg_ch_repeat_dict.keys():
                awg_ch_repeat_dict[awg] = []
            awg_ch_repeat_dict[awg].append(chid)
            if repeat_dict_per_awg.get(awg, repeat_dict_per_ch[cname]) \
                    != repeat_dict_per_ch[cname]:
                raise NotImplementedError(f"Repeat pattern on {cname} is "
                f"different from at least one other channel on {awg}:"
                f"{repeat_dict_per_ch[cname]} vs {repeat_dict_per_awg[awg]}")
            repeat_dict_per_awg[awg] = repeat_dict_per_ch[cname]
            
        for awg_repeat, chs_repeat in awg_ch_repeat_dict.items():
            for ch in channels_used[awg_repeat]:
                assert ch in chs_repeat, f"Repeat pattern " \
                    f"provided for {awg_repeat} but no pattern was given on " \
                    f"{ch}. All used channels on the same awg must have a " \
                    f"repeat pattern."

        return repeat_dict_per_awg


def to_base(n, b, alphabet=None, prev=None):
    if prev is None: prev = []
    if n == 0: 
        if alphabet is None: return prev
        else: return [alphabet[i] for i in prev]
    return to_base(n//b, b, alphabet, prev+[n%b])

def _zi_wave_dir():
    if os.name == 'nt':
        dll = ctypes.windll.shell32
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH + 1)
        if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
            _basedir = buf.value
        else:
            log.warning('Could not extract my documents folder')
    else:
        _basedir = os.path.expanduser('~')
    wave_dir = os.path.join(_basedir, 'Zurich Instruments', 'LabOne',
        'WebServer', 'awg', 'waves')
    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)
    return wave_dir


def _zi_clear_waves():
    wave_dir = _zi_wave_dir()
    for f in os.listdir(wave_dir):
        if f.endswith(".csv"):
            os.remove(os.path.join(wave_dir, f))
        elif f.endswith('.cache'):
            shutil.rmtree(os.path.join(wave_dir, f))


def _zi_wavename_pair_to_argument(w1, w2):
    if w1 is not None and w2 is not None:
        return f'{w1}, {w2}'
    elif w1 is not None and w2 is None:
        return f'1, {w1}'
    elif w1 is None and w2 is not None:
        return f'2, {w2}'
    else:
        return ''