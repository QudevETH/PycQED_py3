import logging
import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_HDAWG_core import ZI_HDAWG_core
except Exception:
    ZI_HDAWG_core = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_base_instrument import merge_waveforms
except Exception:
    pass

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class HDAWG8Pulsar(PulsarAWGInterface):
    """ZI HDAWG8 specific functionality for the Pulsar class."""

    awg_classes = [ZI_HDAWG_core]

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
            initial_value=1.0,
            docstring=f"Scales the AWG output of channel {name} with a given "
                      f"factor between -1 and +1."
        )
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
            self.add_parameter(
                param_name,
                unit='Hz',
                initial_value=None,
                set_cmd=self._hdawg_mod_setter(awg, awg_nr),
                get_cmd=self._hdawg_mod_getter(awg, awg_nr),
                docstring=f"Carrier frequency of internal modulation for the "
                          f"channel pair starting with {name}. Positive "
                          f"(negative) sign corresponds to upper (lower) side "
                          f"band. Setting the frequency to None disables "
                          f"internal modulation."
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
            if id[-1] != 'm':  # analog channel
                def s(val):
                    obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
            else:  # marker channel (offset cannot be set)
                s = None
        elif par == 'amp':
            if id[-1] != 'm':  # analog channel
                def s(val):
                    obj.set('sigouts_{}_range'.format(int(id[2])-1), 2*val)
            else:  # marker channel (scaling cannot be set)
                s = None
        elif par == 'amplitude_scaling' and id[-1] != 'm':
            # ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1, etc.
            awg = int((int(id[2:]) - 1) / 2)
            # ch1/ch3/... are output 0, ch2/ch4/... are output 0,
            output = (int(id[2:]) - 1) - 2 * awg
            def s(val):
                obj.set(f'awgs_{awg}_outputs_{output}_amplitude', val)
                log.debug(f'awgs_{awg}_outputs_{output}_amplitude: {val}')
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _hdawg_getter(self, obj, id, par):
        if par == 'offset':
            if id[-1] != 'm':  # analog channel
                def g():
                    return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
            else:  # marker channel (offset always 0)
                return lambda: 0
        elif par == 'amp':
            if id[-1] != 'm':  # analog channel
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['sigouts_{}_range' \
                            .format(int(id[2])-1)].get_latest()/2
                    else:
                        return obj.get('sigouts_{}_range' \
                            .format(int(id[2])-1))/2
            else:  # marker channel
                return lambda: 1
        elif par == 'amplitude_scaling' and id[-1] != 'm':  # analog channel
            # ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1, etc.
            awg = int((int(id[2:]) - 1) / 2)
            # ch1/ch3/... are output 0, ch2/ch4/... are output 0,
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
                # see pycqed\instrument_drivers\physical_instruments\
                #   ZurichInstruments\zi_parameter_files\node_doc_HDAWG8.json
                # for description of the nodes used below.
                # awg_nr: ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1,
                # etc. Mode 1 (2) means that the AWG Output is multiplied with
                # Sine Generator signal 0 (1) of this sub-awg
                obj.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 1)
                obj.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 2)
                # For the oscillator, we can use any index, as long as the
                # respective osc is not needed for anything else. Since we
                # currently use oscs only here, the following index
                # calculated from awg_nr can ensure that a unique osc is
                # used for every channel pair for which we configure
                # internal modulation.
                osc_nr = awg_nr * 4
                # set up the two sines of the channel pair with the same
                # oscillator and with 90 phase shift
                obj.set(f'sines_{awg_nr * 2}_oscselect', osc_nr)
                obj.set(f'sines_{awg_nr * 2 + 1}_oscselect', osc_nr)
                obj.set(f'sines_{awg_nr * 2}_phaseshift', 0)
                # positive (negative) phase shift is needed for upper (
                # lower) sideband
                obj.set(f'sines_{awg_nr * 2 + 1}_phaseshift', sideband * 90)
                # configure the oscillator frequency
                obj.set(f'oscs_{osc_nr}_freq', freq)
        return s

    @staticmethod
    def _hdawg_mod_getter(obj, awg_nr):
        def g():
            m0 = obj.get(f'awgs_{awg_nr}_outputs_0_modulation_mode')
            m1 = obj.get(f'awgs_{awg_nr}_outputs_1_modulation_mode')
            if m0 == 0 and m1 == 0:
                # If modulation mode is 0 for both outputs, internal
                # modulation is switched off (indicated by a modulation
                # frequency set to None).
                return None
            elif m0 == 1 and m1 == 2:
                # these calcuations invert the calculations in
                # _hdawg_mod_setter, see therein for explaining comments
                osc0 = obj.get(f'sines_{awg_nr * 2}_oscselect')
                osc1 = obj.get(f'sines_{awg_nr * 2 + 1}_oscselect')
                if osc0 == osc1:
                    sideband = np.sign(obj.get(
                        f'sines_{awg_nr * 2 + 1}_phaseshift'))
                    return sideband * obj.get(f'oscs_{osc0}_freq')
            # If we have not returned a result at this point, the current
            # AWG settings do not correspond to a configuration made by
            # _hdawg_mod_setter.
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
                        ch_has_waveforms[ch1id] |= wave[0] is not None
                        ch_has_waveforms[ch1mid] |= wave[1] is not None
                        ch_has_waveforms[ch2id] |= wave[2] is not None
                        ch_has_waveforms[ch2mid] |= wave[3] is not None
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

                    if not len(channels_to_upload):
                        # _program_awg was called only to decide which
                        # sub-AWGs are active, and the rest of this loop
                        # can be skipped
                        continue
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
        return any([obj.get('awgs_{}_enable'.format(awg_nr)) for awg_nr in
                    self._hdawg_active_awgs(obj)])

    def _clock(self, obj, cid):
        return obj.clock_freq()

    def _hdawg_active_awgs(self, obj):
        return [0,1,2,3]

    def _get_segment_filter_userregs(self, obj):
        return [(f'awgs_{i}_userregs_{obj.USER_REG_FIRST_SEGMENT}',
                 f'awgs_{i}_userregs_{obj.USER_REG_LAST_SEGMENT}')
                for i in range(4) if obj._awg_program[i] is not None]

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        chid = self.get(ch + '_id')
        awg.set('sigouts_{}_on'.format(int(chid[-1]) - 1), on)
