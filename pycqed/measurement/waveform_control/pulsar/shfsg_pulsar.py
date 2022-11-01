import logging
from typing import List, Tuple

import numpy as np

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter

from pycqed.utilities.math import vp_to_dbm, dbm_to_vp
from .zi_pulsar_mixin import ZIPulsarMixin, ZIMultiCoreCompilerMixin
from .zi_pulsar_mixin import ZIGeneratorModule
from .zi_pulsar_mixin import diff_and_combine_dicts
from .pulsar import PulsarAWGInterface

from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import mc_parameter_wrapper
import zhinst

try:
    import zhinst.toolkit
    from zhinst.qcodes import SHFSG as SHFSG_core
except Exception:
    SHFSG_core = type(None)


log = logging.getLogger(__name__)


class SHFGeneratorModulesPulsar(PulsarAWGInterface, ZIPulsarMixin,
                                ZIMultiCoreCompilerMixin):
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
        self._init_mcc()
        self._sgchannel_sine_enable = [False] * len(self.awg.sgchannels)
        """Determines the sgchannels for which the sine generator is turned on
        (off) in :meth:`start` (:meth:`stop`).
        """

        self._awg_modules = []
        for awg_nr in range(len(self.awg.sgchannels)):
            channel = SHFGeneratorModule(
                awg=self.awg,
                awg_interface=self,
                awg_nr=awg_nr
            )
            self._awg_modules.append(channel)

    def _get_awgs_mcc(self) -> list:
        return [sgc.awg for sgc in self.awg.sgchannels]

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
            # set centerfreq
            self.awg.synthesizers[
                self.awg.sgchannels[ch].synthesizer()].centerfreq(value,
                                                                  deep=True)
            # get the new value of centerfreq that was set above
            new_center_freq = self.awg.synthesizers[
                self.awg.sgchannels[ch].synthesizer()].centerfreq()
            if np.abs(new_center_freq - value) > 1:
                log.warning(f'{self.awg.name}: center freq. {value/1e6:.6f} '
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
        is_running = []
        first_sg_awg = len(getattr(self.awg, 'qachannels', []))
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            if self.awg._awg_program[awg_nr + first_sg_awg] is not None:
                is_running.append(sgchannel.awg.enable())
        return all(is_running)

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

    def get_frequency_sweep_function(self, ch, mod_freq=0,
                                     allow_IF_sweep=True):
        """
        Args:
            ch (str): Name of the SGChannel to configure
            mod_freq(float): Modulation frequency of the pulse uploaded to the
                AWG. In case the continuous output is used, this should be set
                to 0. Defaults to 0.
            allow_IF_sweep (bool): specifies whether a combined LO and IF
                sweep may be used (default: True). Note that setting this to
                False leads to a sweep function that is only allowed to take
                values on the 100 MHz grid supported by the synthesizer.
        """
        chid = self.pulsar.get(ch + '_id')
        name = 'Frequency'
        if not allow_IF_sweep:
            return mc_parameter_wrapper.wrap_par_to_swf(
                self.pulsar.parameters[f'{ch}_centerfreq'])
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

    def get_centerfreq_generator(self, ch: str):
        """Return the generator of the center frequency associated with a given channel.

        Args:
            ch: channel of the AWG.
        Returns:
            center_freq_generator module
        """
        chid = self.pulsar.get(ch + '_id')
        return self.awg.sgchannels[int(chid[2]) - 1].synthesizer() - 1

    def _direct_mod_setter(self, ch):
        def s(val):
            if val == None:
                self.awg.configure_sine_generation(
                    self.pulsar.get(ch + '_id'), enable=False)
            else:
                self.awg.configure_sine_generation(
                    self.pulsar.get(ch + '_id'), enable=True, freq=val,
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

    def upload_waveforms(self, awg_nr, wave_idx, waveforms, wave_hashes):
        # This method is needed because 'finalize_upload_after_mcc' method in
        # 'MultiCoreCompilerQudevZI' class calls 'upload_waveforms' method
        # from device interfaces instead of from channel interfaces.
        self._awg_modules[awg_nr].upload_waveforms(
            wave_idx=wave_idx,
            waveforms=waveforms,
            wave_hashes=wave_hashes
        )


class SHFSGPulsar(SHFGeneratorModulesPulsar):
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


class SHFGeneratorModule(ZIGeneratorModule):
    """Pulsar interface for ZI SHF Generator AWG modules. Each AWG module
    consists of one analog channel and one marker channel. There are two AWGs
    in each analog channel, one for generating in-phase (I-) signal and
    the other generating quadrature (Q-) signal. Please refer to ZI user manual
    https://docs.zhinst.com/shfsg_user_manual/overview.html
    for details."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Q channel waveforms on the SG channels needs to be flipped due to
        # its specific realization of modulation and up-conversion.
        self._negate_q = True

    def _generate_channel_ids(
            self,
            awg_nr
    ):
        ch1id = f'sg{awg_nr+1}i'
        ch2id = f'sg{awg_nr+1}q'
        chmid = 'ch{}m'.format(awg_nr * 2 + 1)

        first_sg_awg = len(getattr(self._awg, 'qachannels', []))
        self.channel_ids = [ch1id, chmid, ch2id]
        self.analog_channel_ids = [ch1id, ch2id]
        self.marker_channel_ids = [chmid]
        self._upload_idx = awg_nr + first_sg_awg

    def _update_use_filter_flag(
            self,
            awg_sequence,
    ):
        # FIXME: deactivated until implemented for QA
        self._use_filter = False

    def _update_internal_mod_config(
            self,
            awg_sequence,
    ):
        """Collects and combines internal modulation generation settings
        specified in awg_sequence. If settings from different elements are
        coherent with each other, the combined setting will be programmed to
        the channel.

        Args:
            awg_sequence: A list of elements. Each element consists of a
                waveform-hash for each codeword and each channel.
        """
        channels = [self.pulsar._id_channel(chid, self._awg.name)
                    for chid in self.analog_channel_ids]
        channel_mod_config = {ch: {} for ch in channels}

        # Combine internal modulation configurations from all elements in
        # the sequence into one and check if they are compatible with each other
        for element in awg_sequence:
            awg_sequence_element = awg_sequence[element]
            if awg_sequence_element is None:
                continue
            metadata = awg_sequence_element.get('metadata', {})
            element_mod_config = metadata.get('mod_config', {})
            if not diff_and_combine_dicts(
                    element_mod_config,
                    channel_mod_config,
                    excluded_keys=['mod_freq', 'mod_phase']
            ):
                raise Exception('Modulation config in metadata is incompatible'
                                'between different elements in same sequence.')

        # Configure internal modulation for each channel. For the SG modules we
        # take config of the I channel and ignore the Q channel configuration
        for ch, config in channel_mod_config.items():
            if ch.endswith('q'):
                continue
            self._awg.configure_internal_mod(
                chid=self.pulsar.get(ch + '_id'),
                enable=config.get('internal_mod', False),
                osc_index=config.get('osc', 0),
                sine_generator_index=config.get('sine', 0),
                gains=config.get('gains', (1.0, - 1.0, 1.0, 1.0))
            )

        self._mod_config = channel_mod_config

    def _update_sine_generation_config(
            self,
            awg_sequence,
    ):
        """Collects and combines sine wave generation settings specified in
        awg_sequence. If settings from different elements are coherent with
        each other, the combined setting will be programmed to the channel.

        Args:
            awg_sequence: A list of elements. Each element consists of a
            waveform-hash for each codeword and each channel.
        """
        channels = [self._awg_interface.pulsar._id_channel(chid, self._awg.name)
                    for chid in self.analog_channel_ids]
        channel_sine_config = {ch: {} for ch in channels}

        # Combine sine generation configurations from all elements in
        # the sequence into one and check if they are compatible with each other
        for element in awg_sequence:
            awg_sequence_element = awg_sequence[element]
            if awg_sequence_element is None:
                continue
            metadata = awg_sequence_element.get('metadata', {})
            element_sine_config = metadata.get('sine_config', {})
            if not diff_and_combine_dicts(
                    element_sine_config,
                    channel_sine_config
            ):
                raise Exception('Sine config in metadata is incompatible'
                                'between different elements in same sequence.')

        # Configure sine output for each channel. For the SG modules we
        # take config of the I channel and ignore the Q channel configuration
        for ch, config in channel_sine_config.items():
            if ch.endswith('q'):
                continue
            self._awg.configure_sine_generation(
                chid=self.pulsar.get(ch + '_id'),
                enable=config.get('continuous', False),
                osc_index=config.get('osc', 0),
                sine_generator_index=config.get('sine', 0),
                gains=config.get('gains', (0.0, 1.0, 1.0, 0.0))
            )

        self._sine_config = channel_sine_config

    def _update_waveforms(self, wave_idx, wave_hashes, waveforms):
        awg_nr = self._awg_nr

        # check if the waveform has been uploaded
        if self.pulsar.use_sequence_cache():
            if wave_hashes == self.waveform_cache.get(wave_idx, None):
                log.debug(f'{self._awg.name} awgs{awg_nr}: '
                          f'{wave_idx} same as in cache')
                return
        log.debug(
            f'{self._awg.name} awgs{awg_nr}: {wave_idx} needs to be uploaded')

        # take the waves specified for this channel from the overall wave dict
        a1, m1, a2, m2 = [waveforms.get(h, None) for h in wave_hashes]

        # harmonize the wave lengths to the longest waveform
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

        # Q channel sign needs to be flipped for SHFSG/QC
        if a2 is not None:
            a2 = -a2

        waveforms = zhinst.toolkit.waveform.Waveforms()
        waveforms.assign_waveform(wave_idx, a1, a2)

        if self.pulsar.use_mcc() and len(self._awg_interface.awgs_mcc) > 0:
            # Parallel seqc compilation is used, which must take place before
            # waveform upload. Waveforms are added to self.wfms_to_upload and
            # will be uploaded to device in pulsar._program_awgs.
            self._awg_interface.wfms_to_upload[(awg_nr, wave_idx)] = \
                (waveforms, wave_hashes)
        else:
            self.upload_waveforms(wave_idx, waveforms, wave_hashes)

    def upload_waveforms(self, wave_idx, waveforms, wave_hashes):
        """
        Upload a wavefor to this awg module.

        Args:
            wave_idx (int): index of wave upload (0 or 1)
            waveforms: waveforms to upload
            wave_hashes: waveforms hashes
        """
        # Upload waveforms to awg
        sgchannel = self._awg.sgchannels[self._awg_nr]
        sgchannel.awg.write_to_waveform_memory(waveforms)
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

    def _generate_oscillator_seq_code(self):
        i_channel = self.pulsar._id_channel(
            cid=self.analog_channel_ids[0],
            awg=self._awg.name
        )
        mod_config = self._mod_config[i_channel]
        sine_config = self._sine_config[i_channel]
        if mod_config.get('internal_mod', False) \
                or sine_config.get('continuous', False):
            # Reset the starting phase of all oscillators at the beginning
            # of a sequence using the resetOscPhase instruction. This
            # ensures that the carrier-envelope offset, and thus the final
            # output signal, is identical from one repetition to the next.
            self._playback_strings.append(f'resetOscPhase();\n')
            osc_id = str(self._mod_config.get('osc', '0')) \
                if self._mod_config.get('internal_mod', False) \
                else str(self._sine_config.get('osc', '0'))
            self._playback_strings.append(f'const SWEEP_OSC = {osc_id};\n')

    def _generate_playback_string(
            self,
            wave,
            codeword,
            use_placeholder_waves,
            metadata,
            first_element_of_segment
    ):
        prepend_zeros = 0
        self._playback_strings += self._awg_interface.zi_playback_string(
            name=self._awg.name,
            device='shfsg',
            wave=wave,
            codeword=codeword,
            prepend_zeros=prepend_zeros,
            placeholder_wave=use_placeholder_waves,
            allow_filter=metadata.get('allow_filter', False)
        )

    def _check_ignore_waveforms(self):
        i_channel = self.pulsar._id_channel(
            cid=self.analog_channel_ids[0],
            awg=self._awg.name
        )
        return self._sine_config[i_channel].get("ignore_waveforms", False)

    def _configure_awg_str(
            self,
            awg_str
    ):
        sgchannel = self._awg.sgchannels[self._awg_nr]
        sgchannel.awg.load_sequencer_program(
            sequencer_program=awg_str,
            timeout=600
        )

    def _save_awg_str(
            self,
            awg_str,
    ):
        """Saves awg source string in attribute
        self._awg._awg_source_strings."""
        sgchannel = self._awg.sgchannels[self._awg_nr]
        if hasattr(self._awg, 'store_awg_source_string'):
            self._awg.store_awg_source_string(sgchannel, awg_str)

    def _set_signal_output_status(self):
        if self.pulsar.sigouts_on_after_programming():
            for sgchannel in self._awg.sgchannels:
                sgchannel.output.on(True)
