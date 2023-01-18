from functools import partial

import numpy as np
import logging
from qcodes.instrument.parameter import ManualParameter
import qcodes.utils.validators as vals
import time

try:
    from pycqed.instrument_drivers.physical_instruments.AWG70002A \
        import AWG70002A
except Exception:
    AWG70002A = type(None)

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class AWG70kPulsar(PulsarAWGInterface):
    """Tektronix AWG70k specific functionality for the Pulsar class.
        Currently supports the way SuperQuLAN uses the 70k:
        Slow trigger mode, no marker channels."""

    AWG_CLASSES = [AWG70002A]

    GRANULARITY = 160
    ELEMENT_START_GRANULARITY = 160 / 25e9
    MIN_LENGTH_SAMPLES = 160
    MIN_LENGTH = MIN_LENGTH_SAMPLES / 25e9
    INTER_ELEMENT_DEADTIME = 0.0
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.125, 0.25),
        "marker": (-1.4,1.4),
    }
    CHANNEL_OFFSET_BOUNDS = {
        "analog": tuple(),  # Nonzero offset is not supported currently
        "marker": tuple(),
    }

    def __init__(self, pulsar, awg):
        super().__init__(pulsar, awg)
        self._it_to_ch_name = {}

    def create_awg_parameters(self, channel_name_map):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name
        pulsar.add_parameter(f"{name}_trigger_source",
                             initial_value="TrigA",
                             vals=vals.Enum("TrigA", "TrigB", "Internal"),
                             parameter_class=ManualParameter,
                             docstring="Defines for which trigger source the "
                                       "AWG should wait, before playing the "
                                       "next waveform. Allowed values are: "
                                       "'TrigA', 'TrigB', 'Internal'.")
        group = []
        for ch_nr in range(2):
            id = f"ch{ch_nr + 1}"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name, "analog")
            pulsar.channels.add(ch_name)
            group.append(ch_name)
            # We are using the 70k without markers,
            # otherwise their definitions would go here

        # all channels are considered as a single group
        for ch_name in group:
            pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id: str, ch_name: str, ch_type):
        super().create_channel_parameters(id, ch_name, ch_type)
        self._it_to_ch_name[id] = ch_name

        pulsar = self.pulsar

        if ch_type == "analog":

            pulsar.add_parameter(f"{ch_name}_offset_mode",
                                 parameter_class=ManualParameter,
                                 initial_value="software",
                                 vals=vals.Enum("software"))

            scale_param = f'{ch_name}_amplitude_scaling'

            # The set_cmd of the amplitude scaling makes sure that the AWG amp
            # parameter gets updated when the scaling is changed.
            # The product of the amp and scaling parameters is
            # rounded to the nearest 0.001 when passed to the AWG to prevent
            # a build-up of rounding errors caused by the AWG.
            pulsar.add_parameter(
                scale_param,
                label=f'{ch_name} amplitude scaling',
                set_cmd=(lambda v,
                                g=partial(self.awg_getter, id, "amp",
                                          scale_param=scale_param),
                                s=partial(self.awg_setter, id, "amp"):
                         s(np.round(v * g(), decimals=3))),
                vals=vals.Numbers(0.5, 1.0))

            del pulsar.parameters[f"{ch_name}_amp"]
            pulsar.add_parameter(f"{ch_name}_amp",
                                 label=f"{ch_name} amplitude", unit='V',
                                 set_cmd=partial(self.awg_setter, id, "amp",
                                                 scale_param=scale_param),
                                 get_cmd=partial(self.awg_getter, id, "amp",
                                                 scale_param=scale_param),
                                 vals=vals.Numbers(
                                     *self.CHANNEL_AMPLITUDE_BOUNDS[ch_type]))

            # Due to its set_cmd, amplitude scaling can be set to its initial
            # value only now after the amp param has been created.
            pulsar.parameters[scale_param](1.0)

        else: # ch_type == "marker"
            # So far no additional parameters specific to marker channels
            pass

    def awg_getter(self, id:str, param:str, scale_param=None):

        # Sanity checks
        super().awg_getter(id, param)

        if id in ['ch1', 'ch2']:
            ch_name = self._it_to_ch_name[id]
            offset_mode = self.pulsar.parameters[f"{ch_name}_offset_mode"].get()
            if param == 'offset':
                return 0
                # Nonzero offset is not supported currently
            elif param == 'amp':
                if self.pulsar.awgs_prequeried:
                    amp = self.awg.submodules[id].parameters['awg_amplitude']\
                              .get_latest() / 2
                else:
                    amp = self.awg.submodules[id].awg_amplitude.get() / 2
                if scale_param is not None and self.pulsar.get(scale_param) is \
                        not None:
                    amp /= self.pulsar.get(scale_param)
                return amp

    def awg_setter(self, id:str, param:str, value, scale_param=None):

        # Sanity checks
        super().awg_setter(id, param, value)

        if id in ['ch1', 'ch2']:
            ch_name = self._it_to_ch_name[id]
            offset_mode = self.pulsar.parameters[f"{ch_name}_offset_mode"].get()
            if param == 'offset':
                if value != 0:
                    raise NotImplementedError("Nonzero offset is not supported currently")
            if param == 'amp':
                scale = 1.0 if scale_param is None else self.pulsar.get(scale_param)
                if scale != 1.0:
                    raise ValueError(
                        'Amplitude cannot be changed while amplitude '
                        'scaling is enabled. '
                        f'Current scaling factor {scale_param}: {scale}')
                else:
                    self.awg.submodules[id].awg_amplitude.set(2 * value * scale)


    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):

        packed_waveforms = {}
        wfname_l = []

        grp_has_waveforms = {f'ch{i+1}': False for i in range(2)}

        for element in awg_sequence:
            if awg_sequence[element] is None:
                continue
            metadata = awg_sequence[element].pop('metadata', {})
            if list(awg_sequence[element].keys()) != ['no_codeword']:
                raise NotImplementedError('AWG70k sequencer does '
                                          'not support codewords!')
            chid_to_hash = awg_sequence[element]['no_codeword']

            if not any(chid_to_hash):
                continue  # no waveforms

            maxlen = max([len(waveforms[h]) for h in chid_to_hash.values()])
            maxlen = max(maxlen, self.MIN_LENGTH_SAMPLES)

            wfname_l.append([])
            for grp in [f'ch{i + 1}' for i in range(2)]:
                wave = (chid_to_hash.get(grp, None), )
                grp_has_waveforms[grp] |= (wave != (None))
                wfname = self.pulsar._hash_to_wavename((maxlen, wave))
                grp_wfs = [np.pad(waveforms.get(h, [0]),
                                  (0, maxlen - len(waveforms.get(h, [0]))),
                                  'constant', constant_values=0) for h in wave]
                packed_waveforms[wfname] = np.array(grp_wfs[0])
                wfname_l[-1].append(wfname)
                if any([wf[0] != 0 for wf in grp_wfs]):
                    log.warning(f'Element {element} starts with non-zero '
                                f'entry on {self.awg.name}.')

        if not any(grp_has_waveforms.values()):
            for grp in ['ch1', 'ch2']:
                self.awg.set('{}_state'.format(grp), grp_has_waveforms[grp])
            return None

        self.pulsar.add_awg_with_waveforms(self.awg.name)

        filename = 'pycqed_pulsar_sequence.seqx'
        trigger_dict = {'TrigA': 1, 'TrigB': 2, 'Internal': 3}

        trig_waits = [trigger_dict[self.pulsar.get(
            f'{self.awg.name}_trigger_source')]] * len(wfname_l)
        nreps = [1] * len(wfname_l)
        event_jumps = [0]* len(wfname_l)
        event_jump_to = [0]* len(wfname_l)
        go_to = [0] * len(wfname_l)
        go_to[-1] = 1

        awg_file = self.awg.makeSEQXFile(trig_waits=trig_waits,
                     nreps=nreps,
                     event_jumps=event_jumps,
                     event_jump_to=event_jump_to,
                     go_to=go_to,
                     wfms=packed_waveforms,
                     seqname='Pulsar_Sequence',
                     sequence=wfname_l)

        self.awg.sendSEQXFile(awg_file, filename)
        self.awg.clearSequenceList()
        self.awg.clearWaveformList()
        self.awg.loadSEQXFile(filename)
        self.awg.ch1.setSequenceTrack('Pulsar_Sequence', 1)
        self.awg.ch2.setSequenceTrack('Pulsar_Sequence', 2)

        time.sleep(.1)
        # Waits for AWG to be ready. FIXME Possibly remove it and re-test
        self.awg.wait_for_operation_to_complete()

        for grp in ['ch1', 'ch2']:
            self.awg.submodules[grp].set('state', 1*grp_has_waveforms[grp])

        return awg_file

    def is_awg_running(self):
        return self.awg.run_state() != "Stopped"

    def clock(self):
        return self.awg.sample_rate()

    def sigout_on(self, ch, on=True):
        chid = self.pulsar.get(ch + '_id')
        self.awg.submodules[chid].set("state", on)

