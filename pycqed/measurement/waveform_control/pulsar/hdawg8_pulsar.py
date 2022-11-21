import time
import logging
import numpy as np
from copy import deepcopy
from functools import partial
import json

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

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments\
    .ZI_base_instrument import MockDAQServer

from .pulsar import PulsarAWGInterface
from .zi_pulsar_mixin import ZIPulsarMixin, ZIMultiCoreCompilerMixin
from .zi_pulsar_mixin import ZIGeneratorModule


log = logging.getLogger(__name__)

class HDAWG8Pulsar(PulsarAWGInterface, ZIPulsarMixin, ZIMultiCoreCompilerMixin):
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
                                   host='localhost', interface=awg.interface,
                                   server=awg.server)
            if getattr(self.awg.daq, 'server', None) == 'emulator':
                # This is a hack for virtual setups to make sure that the
                # ready node is in sync between the two mock DAQ servers.
                for i in range(4):
                    path = f'/{self.awg.devname}/awgs/{i}/ready'
                    self._awg_mcc._session.daq_server.nodes[
                        path] = self.awg.daq.nodes[path]
        except ImportError as e:
            log.debug(f'Error importing zhinst-qcodes: {e}.')
            log.debug(f'Parallel elf compilation will not be available for '
                      f'{awg.name} ({awg.devname}).')
            self._awg_mcc = None
        self._init_mcc()

        # dict for storing previously-uploaded waveforms
        self.waveform_cache = dict()

        # Each AWG module corresponds to an HDAWG channel pair.
        self.awg_modules = []
        for awg_nr in self._hdawg_active_awgs():
            channel_pair = HDAWGGeneratorModule(
                awg=self.awg,
                awg_interface=self,
                awg_nr=awg_nr
            )
            self.awg_modules.append(channel_pair)

    def _get_awgs_mcc(self) -> list:
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
                             parameter_class=ManualParameter,
                             docstring="Configures whether to use placeholder "
                                       "waves in combination with binary "
                                       "waveform uploadon this device. If set "
                                       "to True, placeholder waves "
                                       "will be enabled on all AWG modules on "
                                       "this device. If set to False, pulsar "
                                       "will check channel-specific settings "
                                       "and programs command table on a "
                                       "per-sub-AWG basis."
                             )
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
        pulsar.add_parameter(f"{name}_use_command_table",initial_value=False,
                             vals=vals.Bool(), parameter_class=ManualParameter,
                             docstring="Configures whether to use command table"
                                       "for waveform sequencing on this "
                                       "device. If set to True, command table "
                                       "will be enabled on all AWG modules on "
                                       "this device. If set to False, pulsar "
                                       "will check channel-specific settings "
                                       "and programs command table on a "
                                       "per-sub-AWG basis.")
        pulsar.add_parameter(f"{name}_internal_modulation", initial_value=False,
                             vals=vals.Bool(), parameter_class=ManualParameter,
                             docstring="Configures whether to use digital "
                                       "modulation for waveform generation on "
                                       "this  device. If set to True, internal "
                                       "modulation will be enabled on all AWG "
                                       "modules on this device. If set to "
                                       "False, pulsar will check "
                                       "channel-specific settings and programs "
                                       "command table on a per-sub-AWG basis.")

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

            awg_nr = (int(id[2:]) - 1) // 2
            output_nr = (int(id[2:]) - 1) % 2
            pulsar.add_parameter(
                '{}_modulation_mode'.format(ch_name),
                vals=vals.Enum('Modulation Off', 'Sine 1', 'Sine 2', 'FG 1' ,
                            'FG 2', 'Advanced', 'off', 'direct',
                            0, 1, 2, 3, 4, 5),
                initial_value='Modulation Off',
                set_cmd=self._hdawg_mod_mode_setter(awg_nr, output_nr),
                get_cmd=self._hdawg_mod_mode_getter(awg_nr, output_nr),
                docstring=f"Modulation mode of channel {ch_name}."
            )
            # first channel of a pair
            if (int(id[2:]) - 1) % 2  == 0:
                param_name = f"{ch_name}_mod_freq"
                pulsar.add_parameter(
                    param_name,
                    unit='Hz',
                    initial_value=None,
                    set_cmd=self._hdawg_mod_freq_setter(awg_nr),
                    get_cmd=self._hdawg_mod_freq_getter(awg_nr),
                    docstring="Carrier frequency of internal modulation for "
                              "a channel pair. Positive (negative) sign "
                              "corresponds to upper (lower) side band. Setting "
                              "it to None disables internal modulation."
                )
                # qcodes will not set the initial value if it is None, so we set
                # it manually here to ensure that internal modulation gets
                # switched off in the init.
                pulsar.set(param_name, None)

                param_name = '{}_direct_mod_freq'.format(ch_name)
                pulsar.add_parameter(
                    param_name,
                    unit='Hz',
                    initial_value=None,
                    set_cmd=self._hdawg_mod_freq_setter(awg_nr, direct=True),
                    get_cmd=self._hdawg_mod_freq_getter(awg_nr, direct=True),
                    docstring=f"Directly output I and Q signals for the "
                            f"channel pair starting with {ch_name}. The output is "
                            f"not modulated according to the uploaded waveform. "
                            f"Positive (negative) sign corresponds to upper "
                            f"(lower) side band. Setting the frequency to "
                            f"None disables the output."
                )
                # qcodes will not set the initial value if it is None, so we set
                # it manually here to ensure that internal modulation gets
                # switched off in the init.
                pulsar.set(param_name, None)

                param_name = '{}_direct_output_amp'.format(ch_name)
                pulsar.add_parameter(
                    param_name,
                    unit='V',
                    initial_value=0,
                    set_cmd=self._hdawg_direct_output_amp_setter(awg_nr),
                    get_cmd=self._hdawg_direct_output_amp_getter(awg_nr),
                    docstring=f"Amplitude of the sine generator output used in "
                            f"direct output mode."
                )

                pulsar.add_parameter(
                    f"{ch_name}_use_placeholder_waves",
                    initial_value=False,
                    vals=vals.Bool(),
                    parameter_class=ManualParameter,
                    docstring="Configures whether to use placeholder waves"
                              "on this AWG module. Note that this "
                              "parameter will be ignored if the device-level "
                              "{dev_name}_use_placeholder_waves is set to "
                              "True. In that case, all AWG modules on the "
                              "device will use placeholder waves irrespective "
                              "of the channel-specific setting."
                )

                pulsar.add_parameter(
                    f"{ch_name}_use_command_table",
                    initial_value=False,
                    vals=vals.Bool(),
                    parameter_class=ManualParameter,
                    docstring="Configures whether to use command table for wave"
                              "sequencing on this AWG module. Note that this "
                              "parameter will be ignored if the device-level "
                              "{dev_name}_use_command_table is set to "
                              "True. In that case, all AWG modules on the "
                              "device will use command table irrespective "
                              "of the channel-specific setting."
                )

                pulsar.add_parameter(
                    f"{ch_name}_internal_modulation",
                    initial_value=False,
                    vals=vals.Bool(),
                    parameter_class=ManualParameter,
                    docstring="Configures whether to use digital modulation"
                              "sequencing on this AWG module. Note that this "
                              "parameter will be ignored if the device-level "
                              "{dev_name}_internal_modulation is set to "
                              "True. In that case, all AWG modules on the "
                              "device will use internal modulation irrespective"
                              "of the channel-specific setting."
                )


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

    def _hdawg_direct_output_amp_setter(self, awg_nr):
        def s(val):
            self.awg.set(f'sines_{awg_nr * 2}_amplitudes_0', val)
            self.awg.set(f'sines_{awg_nr * 2 + 1}_amplitudes_1', val)
        return s

    def _hdawg_direct_output_amp_getter(self, awg_nr):
        def g():
            amp0 = self.awg.get(f'sines_{awg_nr * 2}_amplitudes_0')
            amp1 = self.awg.get(f'sines_{awg_nr * 2 + 1}_amplitudes_1')
            if amp0 != amp1:
                log.warning(f"Amplitude of sine generator 0 on awg {awg_nr * 2}"
                            f"is {amp0} V and not equal to the amplitude of "
                            f"sine generator 1 on awg {awg_nr * 2 + 1} which is"
                            f" {amp1} V.")
            return amp0
        return g

    def _hdawg_mod_freq_setter(self, awg_nr, direct=False, amp=0.0):
        def s(val):
            log.debug(f'{self.awg.name}_awgs_{awg_nr} modulation freq: {val}')
            if val == None:
                self.awg.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 0)
                self.awg.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 0)
                self.awg.set(f'sines_{awg_nr * 2}_enables_0', 0)
                self.awg.set(f'sines_{awg_nr * 2}_enables_1', 0)
                self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_0', 0)
                self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_1', 0)
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
                # For the oscillator, we can use any index, as long as the
                # respective osc is not needed for anything else. Since we
                # currently use oscs only here, the following index
                # calculated from awg_nr can ensure that a unique osc is
                # used for every channel pair for which we configure
                # internal modulation.
                osc_nr = awg_nr * 4
                # configure the oscillator frequency
                self.awg.set(f'oscs_{osc_nr}_freq', freq)
                # set up the two sines of the channel pair with the same
                # oscillator and with 90 phase shift
                self.awg.set(f'sines_{awg_nr * 2}_oscselect', osc_nr)
                self.awg.set(f'sines_{awg_nr * 2 + 1}_oscselect', osc_nr)
                self.awg.set(f'sines_{awg_nr * 2}_phaseshift', 0)
                # positive (negative) phase shift is needed for upper (
                # lower) sideband
                self.awg.set(f'sines_{awg_nr * 2 + 1}_phaseshift', sideband * 90)
                # see pycqed\instrument_drivers\physical_instruments\
                #   ZurichInstruments\zi_parameter_files\node_doc_HDAWG8.json
                # for description of the nodes used below.
                # awg_nr: ch1/ch2 are on sub-awg 0, ch3/ch4 are on sub-awg 1,
                # etc. Mode 1 (2) means that the AWG Output is multiplied with
                # Sine Generator signal 0 (1) of this sub-awg
                if direct:
                    self.awg.set(f'sines_{awg_nr * 2}_enables_0', 1)
                    self.awg.set(f'sines_{awg_nr * 2}_enables_1', 0)
                    self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_0', 0)
                    self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_1', 1)
                    self.awg.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 0)
                    self.awg.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 0)
                else:
                    self.awg.set(f'sines_{awg_nr * 2}_enables_0', 0)
                    self.awg.set(f'sines_{awg_nr * 2}_enables_1', 0)
                    self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_0', 0)
                    self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_1', 0)
                    self.awg.set(f'awgs_{awg_nr}_outputs_0_modulation_mode', 1)
                    self.awg.set(f'awgs_{awg_nr}_outputs_1_modulation_mode', 2)
        return s

    def _hdawg_mod_freq_getter(self, awg_nr, direct=False):
        def g():
            modes = [
                self.awg.get(f'awgs_{awg_nr}_outputs_0_modulation_mode'),
                self.awg.get(f'awgs_{awg_nr}_outputs_1_modulation_mode')
            ]
            if direct:
                enables = [
                    self.awg.get(f'sines_{awg_nr * 2}_enables_0'),
                    self.awg.get(f'sines_{awg_nr * 2}_enables_1'),
                    self.awg.get(f'sines_{awg_nr * 2 + 1}_enables_0'),
                    self.awg.get(f'sines_{awg_nr * 2 + 1}_enables_1')
                ]
            if modes == [0, 0] and (not direct or enables != [1, 0, 0, 1]):
                # If modulation mode is 0 for both outputs, internal
                # modulation is switched off (indicated by a modulation
                # frequency set to None).
                return None
            elif (modes == [1, 2] and not direct) or \
                    (modes == [0, 0] and enables == [1, 0, 0, 1]):
                # these calcuations invert the calculations in
                # _hdawg_mod_freq_setter, see therein for explaining comments
                osc0 = self.awg.get(f'sines_{awg_nr * 2}_oscselect')
                osc1 = self.awg.get(f'sines_{awg_nr * 2 + 1}_oscselect')
                if osc0 == osc1:
                    sideband = np.sign(self.awg.get(
                        f'sines_{awg_nr * 2 + 1}_phaseshift'))
                    return sideband * self.awg.get(f'oscs_{osc0}_freq')
            # If we have not returned a result at this point, the current
            # AWG settings do not correspond to a configuration made by
            # _hdawg_mod_freq_setter.
            log.warning('The current modulation configuration is not '
                        'supported by pulsar. Cannot retrieve modulation '
                        'frequency.')
            return None
        return g

    def _hdawg_mod_mode_setter(self, awg_nr, output_nr):
        def s(val):
            # see pycqed\instrument_drivers\physical_instruments\
            #   ZurichInstruments\zi_parameter_files\node_doc_HDAWG8.json
            # for description of the nodes used below.
            mod_mode_dict = {'Modulation Off': 0, 'Sine 1': 1, 'Sine 2': 2,
                             'FG 1': 3, 'FG 2': 4, 'Advanced': 5, 'off':0,
                             'direct':5}
            if isinstance(val, str):
                mode = mod_mode_dict[val]
            else:
                mode = val
            log.debug(f'{self.awg.name}_awgs_{awg_nr} modulation mod: {val} ({mode})')
            self.awg.set(f'awgs_{awg_nr}_outputs_{output_nr}_modulation_mode', mode)
        return s

    def _hdawg_mod_mode_getter(self, awg_nr, output_nr):
        def g():
            return self.awg.get(f'awgs_{awg_nr}_outputs_{output_nr}_modulation_mode')
        return g

    def get_divisor(self, chid, awg):
        """Divisor is 2 for modulated non-marker channels, 1 for other cases."""

        return 1

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        self._zi_program_generator_awg(
            awg_sequence=awg_sequence,
            waveforms=waveforms,
            repeat_pattern=repeat_pattern,
            channels_to_upload=channels_to_upload,
            channels_to_program=channels_to_program,
        )

    def is_awg_running(self):
        return all([self.awg.get('awgs_{}_enable'.format(awg_nr))
                    for awg_nr in self._hdawg_active_awgs()
                    if self.awg._awg_program[awg_nr] is not None])

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
        if chid[-1] != 'm':  # not a marker channel
            self.awg.set('sigouts_{}_on'.format(int(chid[-1]) - 1), on)

    def upload_waveforms(self, awg_nr, wave_idx, waveforms, wave_hashes):
        # This wrapper method is needed because 'finalize_upload_after_mcc'
        # method in 'MultiCoreCompilerQudevZI' class calls 'upload_waveforms'
        # method from device interfaces instead of from channel interfaces.
        self.awg_modules[awg_nr].upload_waveforms(
            wave_idx=wave_idx,
            waveforms=waveforms,
            wave_hashes=wave_hashes
        )

    @staticmethod
    def is_channel_pair(
            ch1: str,
            ch2: str,
            require_ordered: bool,
    ):
        """Returns if the two input channels belongs to the same channel pair.

        Args:
            ch1 (str): channel of the AWG
            ch2 (str): channel of the AWG
            require_ordered (bool): whether ch1 is required to have a smaller
                index than ch2

        Returns:
            is_channel_pair (str): whether these two AWG channels belongs to
                the same channel pair.
        """
        if require_ordered and ch1 > ch2:
            return False

        ch_idx_1 = int(ch1[-1])
        ch_idx_2 = int(ch2[-1])
        ch_idx_smaller = min(ch_idx_1, ch_idx_2)
        ch_idx_larger = max(ch_idx_1, ch_idx_2)

        if ch_idx_smaller % 2 != 1:
            return False
        elif ch_idx_larger == ch_idx_smaller + 1:
            return True
        else:
            return False

    @staticmethod
    def is_i_channel(ch: str):
        """Returns if this channel has the smaller number in its channel
        pair.

        Args:
            ch: channel of an HDAWG.

        Returns:
            is_i_channel (str): whether this channel has the smaller number
            in its channel pair.
        """
        if ch[-1] == 'm':
            return False
        ch_idx = int(ch[-1])
        return ch_idx <= 8 and ch_idx % 2 == 1


class HDAWGGeneratorModule(ZIGeneratorModule):
    """Pulsar interface for ZI HDAWG AWG modules. Each AWG module consists of
    two analog channels and two marker channels. Please refer to ZI user manual
    https://docs.zhinst.com/hdawg_user_manual/overview.html
    for more details."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hdawg_internal_mod = False
        # TODO: this attribute is used only when using the old internal
        #  modulation implementation on HDAWG. Remove this attribute once the
        #  new implementation is deployed.
        """Flag that indicates whether internal modulation is turned on for 
        this device."""

        self._device_type = 'hdawg'
        """Device type of the generator."""

    def _generate_channel_ids(
            self,
            awg_nr
    ):
        ch1id = 'ch{}'.format(awg_nr * 2 + 1)
        ch1mid = 'ch{}m'.format(awg_nr * 2 + 1)
        ch2id = 'ch{}'.format(awg_nr * 2 + 2)
        ch2mid = 'ch{}m'.format(awg_nr * 2 + 2)

        self.channel_ids = [ch1id, ch1mid, ch2id, ch2mid]
        self.analog_channel_ids = [ch1id, ch2id]
        self.marker_channel_ids = [ch1mid, ch2mid]
        self._upload_idx = awg_nr

    def _generate_divisor(self):
        """Generate divisors for all channels. Divisor is 2 for non-modulated
        marker channels, 1 for every other channel."""
        for chid in self.channel_ids:
            self._divisor[chid] = self._awg_interface.get_divisor(
                chid=chid,
                awg=self._awg.name,
            )

    def _generate_oscillator_seq_code(self):
        mod_config = self._mod_config.get(self.i_channel_name, {})
        if mod_config.get('internal_mod', False):
            # Reset the starting phase of all oscillators at the beginning
            # of a sequence using the resetOscPhase instruction. This
            # ensures that the carrier-envelope offset, and thus the final
            # output signal, is identical from one repetition to the next.
            self._playback_strings.append(f'resetOscPhase();\n')

    def _upload_modulation_config(
            self,
            mod_config,
    ):
        if not mod_config:
            # Modulation configuration is empty
            return

        awg_nr = self._awg_nr
        # Set digital modulation to "mixer" mode.
        self.awg.set(f"awgs_{awg_nr}_outputs_0_modulation_mode", 6)
        self.awg.set(f"awgs_{awg_nr}_outputs_1_modulation_mode", 6)

        # Configure gain matrix for mixer calibration.
        alpha = mod_config.get("alpha", 1.0)
        phi_skew = mod_config.get("phi_skew", 0.0)
        self.awg.set(f"awgs_{awg_nr}_outputs_0_gains_0", np.cos(phi_skew))
        self.awg.set(f"awgs_{awg_nr}_outputs_0_gains_1", np.sin(phi_skew))
        self.awg.set(f"awgs_{awg_nr}_outputs_1_gains_0", 0)
        # TODO: check if alpha is always >= 1
        self.awg.set(f"awgs_{awg_nr}_outputs_1_gains_1", 1.0/alpha)

        # Choose oscillators, set phases and modulation frequencies.
        # TODO: check if we can set negative frequencies to the oscillators
        mod_frequency = mod_config.get("mod_frequency", 0.0)
        osc_nr = mod_config.get("osc_nr", awg_nr * 4)
        self.awg.set(f'oscs_{osc_nr}_freq', mod_frequency)
        self.awg.set(f'sines_{awg_nr * 2}_oscselect', osc_nr)
        self.awg.set(f'sines_{awg_nr * 2 + 1}_oscselect', osc_nr)
        self.awg.set(f'sines_{awg_nr * 2}_phaseshift', 0)
        self.awg.set(f'sines_{awg_nr * 2 + 1}_phaseshift', 90)

        # Disable direct output of sine waves.
        self.awg.set(f'sines_{awg_nr * 2}_enables_0', 0)
        self.awg.set(f'sines_{awg_nr * 2}_enables_1', 0)
        self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_0', 0)
        self.awg.set(f'sines_{awg_nr * 2 + 1}_enables_1', 0)

        # Enable Oscillator control from the sequencer code
        self.awg.set(f'awgs_{awg_nr}_enable', 1)

    def _update_waveforms(self, wave_idx, wave_hashes, waveforms):
        awg_nr = self._awg_nr

        if self.pulsar.use_sequence_cache():
            if wave_hashes == self.waveform_cache.get(wave_idx, None):
                log.debug(
                    f'{self._awg.name} awgs{awg_nr}: {wave_idx} same as in '
                    f'cache')
                return
        log.debug(
            f'{self._awg.name} awgs{awg_nr}: {wave_idx} needs to be uploaded')

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
        if self.pulsar.use_mcc() and len(self._awg_interface.awgs_mcc) > 0:
            # Parallel seqc compilation is used, which must take place before
            # waveform upload. Waveforms are added to self.wfms_to_upload and
            # will be uploaded to device in pulsar._program_awgs.
            self._awg_interface.wfms_to_upload[(awg_nr, wave_idx)] = \
                (wf_raw_combined, wave_hashes)
        else:
            self.upload_waveforms(wave_idx, wf_raw_combined, wave_hashes)

    def upload_waveforms(self, wave_idx, waveforms, wave_hashes):
        """
        Upload waveforms to an awg core (awg_nr).

        Args:
            wave_idx (int): index of wave upload (0 or 1)
            waveforms (array): waveforms to upload
            wave_hashes: waveforms hashes
        """
        # Upload waveforms to awg
        self._awg.setv(f'awgs/{self._awg_nr}/waveform/waves/{wave_idx}',
                       waveforms)
        # Save hashes in the cache memory after a successful waveform upload.
        self._save_hashes(wave_idx, wave_hashes)

    def _save_hashes(self, wave_idx, wave_hashes):
        """
        Save hashes in the cache memory after a successful waveform upload.

        Args:
            wave_idx (int): index of wave upload (0 or 1)
            wave_hashes: waveforms hashes
        """
        if self.pulsar.use_sequence_cache():
            self.waveform_cache[wave_idx] = wave_hashes

    def _update_awg_instrument_status(self):
        # tell ZI_base_instrument that it should not compile a program on
        # this sub AWG (because we already do it here)
        self._awg._awg_needs_configuration[self._awg_nr] = False
        # tell ZI_base_instrument.start() to start this sub AWG (The base
        # class will start sub AWGs for which _awg_program is not None. Since
        # we set _awg_needs_configuration to False, we do not need to put the
        # actual program here, but anything different from None is sufficient.)
        self._awg._awg_program[self._awg_nr] = True

    def _generate_playback_string(
            self,
            wave,
            codeword,
            use_placeholder_waves,
            command_table_index,
            metadata,
            first_element_of_segment
    ):
        if first_element_of_segment:
            prepend_zeros = self.pulsar.parameters[
                f"{self._awg.name}_prepend_zeros"]()
            if prepend_zeros is None:
                prepend_zeros = self.pulsar.prepend_zeros()
            elif isinstance(prepend_zeros, list):
                prepend_zeros = prepend_zeros[self._awg_nr]
        else:
            prepend_zeros = 0
        self._playback_strings += self._awg_interface.zi_playback_string(
            name=self._awg.name,
            device='hdawg',
            wave=wave,
            codeword=codeword,
            prepend_zeros=prepend_zeros,
            placeholder_wave=use_placeholder_waves,
            command_table_index=command_table_index,
            allow_filter=metadata.get('allow_filter', False)
        )

    def _configure_awg_str(
            self,
            awg_str
    ):
        self._awg.configure_awg_from_string(
            self._awg_nr,
            program_string=awg_str,
            timeout=600
        )

    def _generate_command_table_entry(
            self,
            entry_index: int,
            wave_index: int,
            amplitude: float = 1.0,
            phase: float = 0.0,
            sideband: str = 'right',
    ):
        """Generates a command table entry in the format specified
        by ZI. Details of the command table can be found in
        https://docs.zhinst.com/shfqc_user_manual/tutorials/tutorial_command_table.html

        Args:
            entry_index (int): index of the command table entry.
            wave_index(int): index of the waveform to play.
            amplitude (float or ndarray): output amplitude with respect to the
                specified waveform. If a scalar is provided, amplitudes of both
                analog  channels are scaled to this value. If an array of
                length 2 is provided, amplitudes of two analog channels are
                specified explicitly. Accepts input range [-1,1].
            phase (float or ndarray): initial phase of the carrier wave. If a
                scalar is provided, phases of both analog channels are
                scaled to this value. If an array of length 2 is provided,
                phases of two analog channels are specified explicitly.
            sideband (str: 'left', 'right'): sideband to generate during
                up-conversion. This parameter is irrelevant to HDAWG channel
                pairs.

        Returns:
            command_table_entry (dict): A command table entry for HDAWG
            channel pairs.
        """

        if isinstance(amplitude, float):
            amplitude = [amplitude] * 2
        elif not ((isinstance(amplitude, np.ndarray) or
                   isinstance(amplitude, list)) and len(amplitude) == 2):
            raise ValueError(f"{self._awg.name} channel pair {self._awg_nr} "
                             f"receives inappropriate command table amplitude "
                             f"value, accepts float or array-like object with "
                             f"length 2.")

        if isinstance(phase, float) or isinstance(phase, int):
            phase = [float(phase), float(phase)]
        elif not ((isinstance(phase, np.ndarray) or
                   isinstance(phase, list)) and len(phase) == 2):
            raise ValueError(f"{self._awg.name} channel pair {self._awg_nr} "
                             f"receives inappropriate command table phase "
                             f"value, accepts float or array-like object with "
                             f"length 2.")

        hdawg_command_table_entry = {
            "index": entry_index,
            "waveform": {"index": wave_index},
            "amplitude0": {"value": amplitude[0]},
            "amplitude1": {"value": amplitude[1]},
            "phase0": {"value": phase[0]},
            "phase1": {"value": phase[1] + 90},
        }

        return hdawg_command_table_entry

    def _upload_command_table(self):
        # ZI data acquisition server
        daq = self._awg.daq
        device_id = self._awg.get_idn()['serial']

        # add a wrapper outside the command table list
        command_table_list_upload = {
            "$schema": "https://json-schema.org/draft-04/schema#",
            "header": {"version": "1.0.0"},
            "table": self._command_table
        }

        data_node = f"/{device_id}/awgs" \
                    f"/{self._awg_nr}/commandtable/data"
        daq.setVector(data_node, json.dumps(command_table_list_upload))

        if not isinstance(daq, MockDAQServer):
            # This is a DAQ server for actual devices. We request upload
            # status from the device and check if it is successful.
            status_node = f"/{device_id}/awgs" \
                          f"/{self._awg_nr}/commandtable/status"
            status = daq.getInt(status_node)

            if status != 1:
                log.warning(f"Failed to upload the command table to "
                            f"{self._awg.name}, error index {status}")
        else:
            # This is a DAQ server for virtual devices. We assume that upload
            # is successful.
            status = 1

        return status

    def _set_signal_output_status(self):
        if self.pulsar.sigouts_on_after_programming():
            for ch in range(8):
                self._awg.set('sigouts_{}_on'.format(ch), True)
