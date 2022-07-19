import time
import logging
import numpy as np
from copy import deepcopy
from functools import partial

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
from .zi_pulsar_mixin import ZIPulsarMixin, ZIMultiCoreCompilerMixin


log = logging.getLogger(__name__)

class HDAWG8Pulsar(ZIMultiCoreCompilerMixin, PulsarAWGInterface, ZIPulsarMixin):
    """ZI HDAWG8 specific functionality for the Pulsar class."""

    AWG_CLASSES = [ZI_HDAWG_core]
    GRANULARITY = 16
    ELEMENT_START_GRANULARITY = 8 / 2.4e9
    MIN_LENGTH = 16 / 2.4e9
    # TODO: Check if other values commented out should be removed
    INTER_ELEMENT_DEADTIME = 8 / 2.4e9 # 80 / 2.4e9 # 0 / 2.4e9
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.01, 5.0),
        "marker": (0.01, 5.0),
    }
    CHANNEL_OFFSET_BOUNDS = {
        "analog": tuple(), # TODO: Check if there are indeed no bounds for the offset
        "marker": tuple(), # TODO: Check if there are indeed no bounds for the offset
    }
    IMPLEMENTED_ACCESSORS = ["offset", "amp", "amplitude_scaling"]

    _hdawg_sequence_string_template = (
        "{wave_definitions}\n"
        "\n"
        "{codeword_table_defs}\n"
        "\n"
        "while (1) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def __init__(self, pulsar, awg):
        super().__init__(pulsar, awg)
        try:
            # Here we instantiate a zhinst.qcodes-based HDAWG in addition to
            # the one based on the ZI_base_instrument because the parallel
            # uploading of elf files is only supported by the qcodes driver
            from pycqed.instrument_drivers.physical_instruments. \
                ZurichInstruments.zhinst_qcodes_wrappers import HDAWG8
            self._awg_mcc = HDAWG8(awg.devname, name=awg.name + '_mcc',
                                   host='localhost')
        except ImportError as e:
            log.warning(f'Parallel elf compilation not supported for '
                        f'{awg.name} ({awg.devname}):\n{e}')
            self._awg_mcc = None
        # add awgs to multi_core_compiler class variable
        for awg in self.awgs_mcc:
            self.multi_core_compiler.add_awg(awg)

        # dict for storing previously-uploaded waveforms
        self._hdawg_waveform_cache = dict()

    @property
    def awgs_mcc(self) -> list:
        """
        Returns list of the _awg_mcc cores.
        If _awg_mcc was not defined, returns empty list.
        """
        if self._awg_mcc is not None:
            return list(self._awg_mcc.awgs)
        else:
            return []

    def create_awg_parameters(self, channel_name_map):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        # Override _min_length parameter created in base class
        # TODO: Check if this makes sense, it is a constant for the other AWGs
        # Furthermore, it does not really make sense to manually set the minimum
        # length which is a property of the instrument...
        del pulsar.parameters[f"{name}_min_length"]
        pulsar.add_parameter(f"{name}_min_length",
                             initial_value=self.MIN_LENGTH,
                            parameter_class=ManualParameter)

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
        pulsar.add_parameter(f"{name}_prepend_zeros",
                             initial_value=None,
                             vals=vals.MultiType(vals.Enum(None), vals.Ints(),
                                                 vals.Lists(vals.Ints())),
                             parameter_class=ManualParameter)

        group = []
        for ch_nr in range(8):
            id = f"ch{ch_nr + 1}"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name, "analog")
            pulsar.channels.add(ch_name)
            group.append(ch_name)
            id = f"ch{ch_nr + 1}m"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name, "marker")
            pulsar.channels.add(ch_name)
            group.append(ch_name)
            # channel pairs plus the corresponding marker channels are
            # considered as groups
            if (ch_nr + 1) % 2 == 0:
                for ch_name in group:
                    pulsar.channel_groups.update({ch_name: group})
                group = []

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        super().create_channel_parameters(id, ch_name, ch_type)

        pulsar = self.pulsar

        if ch_type == "analog":

            pulsar.add_parameter(
                f"{ch_name}_amplitude_scaling",
                set_cmd=partial(self.awg_setter, id, "amplitude_scaling"),
                get_cmd=partial(self.awg_getter, id, "amplitude_scaling"),
                vals=vals.Numbers(min_value=-1.0, max_value=1.0),
                initial_value=1.0,
                docstring=f"Scales the AWG output of channel by a given factor."
            )
            pulsar.add_parameter(f"{ch_name}_internal_modulation",
                                 initial_value=False, vals=vals.Bool(),
                                 parameter_class=ManualParameter)

            # first channel of a pair
            if (int(id[2:]) - 1) % 2  == 0:
                awg_nr = (int(id[2:]) - 1) // 2
                param_name = f"{ch_name}_mod_freq"
                pulsar.add_parameter(
                    param_name,
                    unit='Hz',
                    initial_value=None,
                    set_cmd=self._hdawg_mod_setter(awg_nr),
                    get_cmd=self._hdawg_mod_getter(awg_nr),
                    docstring="Carrier frequency of internal modulation for "
                              "a channel pair. Positive (negative) sign "
                              "corresponds to upper (lower) side band. Setting "
                              "it to None disables internal modulation."
                )
                # qcodes will not set the initial value if it is None, so we set
                # it manually here to ensure that internal modulation gets
                # switched off in the init.
                pulsar.set(param_name, None)

        else: # ch_type == "marker"
            # So far no additional parameters specific to marker channels
            pass

    def awg_setter(self, id:str, param:str, value):

        # Sanity checks
        super().awg_setter(id, param, value)

        channel_type = "analog" if id[-1] != "m" else "marker"
        ch = int(id[2]) - 1

        if param == "offset":
            if channel_type == "analog":
                self.awg.set(f"sigouts_{ch}_offset", value)
            else:
                pass # raise NotImplementedError("Cannot set offset on marker channels.")
        elif param == "amp":
            if channel_type == "analog":
                self.awg.set(f"sigouts_{ch}_range", 2 * value)
            else:
                pass # raise NotImplementedError("Cannot set amp on marker channels.")
        elif param == "amplitude_scaling" and channel_type == "analog":
            # ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1, etc.
            awg = (int(id[2:]) - 1) // 2
            # ch1/ch3/... are output 0, ch2/ch4/... are output 0,
            output = (int(id[2:]) - 1) - 2 * awg
            self.awg.set(f"awgs_{awg}_outputs_{output}_amplitude", value)

    def awg_getter(self, id:str, param:str):

        # Sanity checks
        super().awg_getter(id, param)

        channel_type = "analog" if id[-1] != "m" else "marker"
        ch = int(id[2]) - 1

        if param == "offset":
            if channel_type == "analog":
                return self.awg.get(f"sigouts_{ch}_offset")
            else:
                return 0
        elif param == "amp":
            if channel_type == "analog":
                if self.pulsar.awgs_prequeried:
                    return self.awg.parameters[f"sigouts_{ch}_range"].get_latest() / 2
                else:
                    return self.awg.get(f"sigouts_{ch}_range") / 2
            else:
                return 1
        elif param == "amplitude_scaling" and channel_type == "analog":
            # ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1, etc.
            awg = (int(id[2:]) - 1) // 2
            # ch1/ch3/... are output 0, ch2/ch4/... are output 0,
            output = (int(id[2:]) - 1) - 2 * awg
            return self.awg.get(f"awgs_{awg}_outputs_{output}_amplitude")

    def _hdawg_mod_setter(self, awg_nr):
        def s(val):
            log.debug(f'{self.awg.name}_awgs_{awg_nr} modulation freq: {val}')
            if val == None:
                self.awg.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 0)
                self.awg.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 0)
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
                self.awg.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 1)
                self.awg.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 2)
                # For the oscillator, we can use any index, as long as the
                # respective osc is not needed for anything else. Since we
                # currently use oscs only here, the following index
                # calculated from awg_nr can ensure that a unique osc is
                # used for every channel pair for which we configure
                # internal modulation.
                osc_nr = awg_nr * 4
                # set up the two sines of the channel pair with the same
                # oscillator and with 90 phase shift
                self.awg.set(f'sines_{awg_nr * 2}_oscselect', osc_nr)
                self.awg.set(f'sines_{awg_nr * 2 + 1}_oscselect', osc_nr)
                self.awg.set(f'sines_{awg_nr * 2}_phaseshift', 0)
                # positive (negative) phase shift is needed for upper (
                # lower) sideband
                self.awg.set(f'sines_{awg_nr * 2 + 1}_phaseshift', sideband * 90)
                # configure the oscillator frequency
                self.awg.set(f'oscs_{osc_nr}_freq', freq)
        return s

    def _hdawg_mod_getter(self, awg_nr):
        def g():
            m0 = self.awg.get(f'awgs_{awg_nr}_outputs_0_modulation_mode')
            m1 = self.awg.get(f'awgs_{awg_nr}_outputs_1_modulation_mode')
            if m0 == 0 and m1 == 0:
                # If modulation mode is 0 for both outputs, internal
                # modulation is switched off (indicated by a modulation
                # frequency set to None).
                return None
            elif m0 == 1 and m1 == 2:
                # these calcuations invert the calculations in
                # _hdawg_mod_setter, see therein for explaining comments
                osc0 = self.awg.get(f'sines_{awg_nr * 2}_oscselect')
                osc1 = self.awg.get(f'sines_{awg_nr * 2 + 1}_oscselect')
                if osc0 == osc1:
                    sideband = np.sign(self.awg.get(
                        f'sines_{awg_nr * 2 + 1}_phaseshift'))
                    return sideband * self.awg.get(f'oscs_{osc0}_freq')
            # If we have not returned a result at this point, the current
            # AWG settings do not correspond to a configuration made by
            # _hdawg_mod_setter.
            log.warning('The current modulation configuration is not '
                        'supported by pulsar. Cannot retrieve modulation '
                        'frequency.')
            return None
        return g

    def get_divisor(self, chid, awg):
        """Divisor is 2 for modulated non-marker channels, 1 for other cases."""

        name = self.pulsar._id_channel(chid, awg)
        if chid[-1]!='m' and self.pulsar.get(f"{name}_internal_modulation"):
            return 2
        else:
            return 1

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):

        self.wfms_to_upload = {}  # store waveforms to upload and hashes
        chids = [f'ch{i+1}{m}' for i in range(8) for m in ['','m']]
        divisor = {chid: self.get_divisor(chid, self.awg.name) for chid in chids}
        def with_divisor(h, ch):
            return (h if divisor[ch] == 1 else (h, divisor[ch]))

        ch_has_waveforms = {chid: False for chid in chids}

        use_placeholder_waves = self.pulsar.get(f"{self.awg.name}_use_placeholder_waves")

        if not use_placeholder_waves:
            if not self.zi_waves_cleared:
                self._zi_clear_waves()

        for awg_nr in self._hdawg_active_awgs():
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
                    f'var first_seg = getUserReg({self.awg.USER_REG_FIRST_SEGMENT});',
                    f'var last_seg = getUserReg({self.awg.USER_REG_LAST_SEGMENT});',
                ]

            ch1id = 'ch{}'.format(awg_nr * 2 + 1)
            ch1mid = 'ch{}m'.format(awg_nr * 2 + 1)
            ch2id = 'ch{}'.format(awg_nr * 2 + 2)
            ch2mid = 'ch{}m'.format(awg_nr * 2 + 2)
            chids = [ch1id, ch1mid, ch2id, ch2mid]

            channels = [self.pulsar._id_channel(chid, self.awg.name)
                        for chid in [ch1id, ch2id]]
            if all([self.pulsar.get(f"{chan}_internal_modulation")
                    for chan in channels]):
                internal_mod = True
            elif not any([self.pulsar.get(f"{chan}_internal_modulation")
                for chan in channels]):
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
                                            f"{self.awg.name}, vawg{awg_nr}, "
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

                    if not len(channels_to_upload):
                        # _program_awg was called only to decide which
                        # sub-AWGs are active, and the rest of this loop
                        # can be skipped
                        continue
                    if not internal_mod:
                        if first_element_of_segment:
                            prepend_zeros = self.pulsar.parameters[
                                f"{self.awg.name}_prepend_zeros"]()
                            if prepend_zeros is None:
                                prepend_zeros = self.pulsar.prepend_zeros()
                            elif isinstance(prepend_zeros, list):
                                prepend_zeros = prepend_zeros[awg_nr]
                        else:
                            prepend_zeros = 0
                        playback_strings += self._zi_playback_string(
                            name=self.awg.name, device='hdawg', wave=wave,
                            codeword=(nr_cw != 0),
                            prepend_zeros=prepend_zeros,
                            placeholder_wave=use_placeholder_waves,
                            allow_filter=metadata.get('allow_filter', False))
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
                # prevent ZI_base_instrument.start() from starting this sub AWG
                self.awg._awg_program[awg_nr] = None
                continue
            # tell ZI_base_instrument that it should not compile a
            # program on this sub AWG (because we already do it here)
            self.awg._awg_needs_configuration[awg_nr] = False
            # tell ZI_base_instrument.start() to start this sub AWG
            # (The base class will start sub AWGs for which _awg_program
            # is not None. Since we set _awg_needs_configuration to False,
            # we do not need to put the actual program here, but anything
            # different from None is sufficient.)
            self.awg._awg_program[awg_nr] = True

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
                    f'{self.awg.name}_{awg_nr}_wave_idx_lookup', None)
                try:
                    np.testing.assert_equal(wave_idx_lookup, cached_lookup)
                    run_compiler = False
                except AssertionError:
                    log.debug(f'{self.awg.name}_{awg_nr}: Waveform reuse pattern '
                              f'has changed. Forcing recompilation.')
                    run_compiler = True

            if run_compiler:
                # We have to retrieve the following parameter to set it
                # again after programming the AWG.
                prev_dio_valid_polarity = self.awg.get(
                    'awgs_{}_dio_valid_polarity'.format(awg_nr))

                if self.pulsar.use_mcc() and len(self.awgs_mcc) > 0:
                    # Parallel seqc string compilation and upload
                    self.multi_core_compiler.add_active_awg(self.awgs_mcc[awg_nr])
                    self.multi_core_compiler.load_sequencer_program(
                        self.awgs_mcc[awg_nr], awg_str)
                else:
                    # Sequential seqc string upload
                    self.awg.configure_awg_from_string(awg_nr, awg_str,
                                                       timeout=600)

                self.awg.set('awgs_{}_dio_valid_polarity'.format(awg_nr),
                        prev_dio_valid_polarity)
                if use_placeholder_waves:
                    self._hdawg_waveform_cache[f'{self.awg.name}_{awg_nr}'] = {}
                    self._hdawg_waveform_cache[
                        f'{self.awg.name}_{awg_nr}_wave_idx_lookup'] = \
                        wave_idx_lookup

            if use_placeholder_waves:
                for idx, wave_hashes in defined_waves[1].items():
                    self._update_waveforms(awg_nr, idx, wave_hashes, waveforms)

        if self.pulsar.sigouts_on_after_programming():
            for ch in range(8):
                self.awg.set('sigouts_{}_on'.format(ch), True)

        if any(ch_has_waveforms.values()):
            self.pulsar.add_awg_with_waveforms(self.awg.name)

    def _update_waveforms(self, awg_nr, wave_idx, wave_hashes, waveforms):
        if self.pulsar.use_sequence_cache():
            if wave_hashes == self._hdawg_waveform_cache[
                    f'{self.awg.name}_{awg_nr}'].get(wave_idx, None):
                log.debug(
                    f'{self.awg.name} awgs{awg_nr}: {wave_idx} same as in cache')
                return
            log.debug(
                f'{self.awg.name} awgs{awg_nr}: {wave_idx} needs to be uploaded')
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
        if self.pulsar.use_mcc() and len(self.awgs_mcc) > 0:
            # Parallel seqc compilation is used, which must take place before
            # waveform upload. Waveforms are added to self.wfms_to_upload and
            # will be uploaded to device in pulsar._program_awgs.
            self.wfms_to_upload[(awg_nr, wave_idx)] = \
                (wf_raw_combined, wave_hashes)
        else:
            self._upload_waveforms(awg_nr, wave_idx, wf_raw_combined, wave_hashes)

    def _upload_waveforms(self, awg_nr, wave_idx, waveforms, wave_hashes):
        """
        Upload waveforms to an awg core (awg_nr).

        Args:
            awg_nr (int): index of awg core (0, 1, 2, or 3)
            wave_idx (int): index of wave upload (0 or 1)
            waveforms (array): waveforms to upload
            wave_hashes: waveforms hashes
        """
        # Upload waveforms to awg
        self.awg.setv(f'awgs/{awg_nr}/waveform/waves/{wave_idx}', waveforms)
        # Save hashes in the cache memory after a successful waveform upload.
        self.save_hashes(awg_nr, wave_idx, wave_hashes)

    def save_hashes(self, awg_nr, wave_idx, wave_hashes):
        """
        Save hashes in the cache memory after a successful waveform upload.

        Args:
            awg_nr (int): index of awg core (0, 1, 2, or 3)
            wave_idx (int): index of wave upload (0 or 1)
            wave_hashes: waveforms hashes
        """
        if self.pulsar.use_sequence_cache():
            self._hdawg_waveform_cache[f'{self.awg.name}_{awg_nr}'][
                wave_idx] = wave_hashes

    def is_awg_running(self):
        return any([self.awg.get('awgs_{}_enable'.format(awg_nr))
                    for awg_nr in self._hdawg_active_awgs()])

    def clock(self):
        return self.awg.clock_freq()

    def _hdawg_active_awgs(self):
        return [0,1,2,3]

    def get_segment_filter_userregs(self, include_inactive=False):
        return [(f'awgs_{i}_userregs_{self.awg.USER_REG_FIRST_SEGMENT}',
                 f'awgs_{i}_userregs_{self.awg.USER_REG_LAST_SEGMENT}')
                for i in range(4) if include_inactive or
                self.awg._awg_program[i] is not None]

    def sigout_on(self, ch, on=True):
        chid = self.pulsar.get(ch + '_id')
        self.awg.set('sigouts_{}_on'.format(int(chid[-1]) - 1), on)
