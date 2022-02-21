import numpy as np
import logging
from qcodes.instrument.parameter import ManualParameter
import qcodes.utils.validators as vals
import time

try:
    from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
except Exception:
    Tektronix_AWG5014 = type(None)
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class AWG5014Pulsar(PulsarAWGInterface):
    """Tektronix AWG5014 specific functionality for the Pulsar class."""

    AWG_CLASSES = [Tektronix_AWG5014, VirtualAWG5014]
    GRANULARITY = 4
    ELEMENT_START_GRANULARITY = 4 / 1.2e9
    MIN_LENGTH = 256 / 1.2e9  # Cannot be triggered faster than 210 ns.
    INTER_ELEMENT_DEADTIME = 0.0
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.01, 2.25),
        "marker": (-5.4, 5.4),
    }
    CHANNEL_OFFSET_BOUNDS = {
        "analog": tuple(), # TODO: Check if there are indeed no bounds for the offset
        "marker": (-2.7, 2.7),
    }

    def create_awg_parameters(self, channel_name_map):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        group = []
        for ch_nr in range(4):
            id = f"ch{ch_nr + 1}"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name)
            pulsar.channels.add(ch_name)
            group.append(ch_name)
            id = f"ch{ch_nr + 1}m1"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name)
            pulsar.channels.add(ch_name)
            group.append(ch_name)
            id = f"ch{ch_nr + 1}m2"
            ch_name = channel_name_map.get(id, f"{name}_{id}")
            self.create_channel_parameters(id, ch_name)
            pulsar.channels.add(ch_name)
            group.append(ch_name)
        # all channels are considered as a single group
        for ch_name in group:
            pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id: str, ch_name: str, ch_type):
        super().create_channel_parameters(id, ch_name, ch_type)

        pulsar = self.pulsar

        if ch_type == "analog":

            pulsar.add_parameter(f"{ch_name}_offset_mode",
                                 parameter_class=ManualParameter,
                                 vals=vals.Enum('software', 'hardware'))

        else: # ch_type == "marker"
            # So far no additional parameters specific to marker channels
            pass

    def awg_getter(self, id:str, param:str):

        # Sanity checks
        super().awg_getter(id, param)

        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            offset_mode = self.pulsar.parameters[f"{id}_offset_mode"].get()
            if param == 'offset':
                if offset_mode == 'software':
                    return self.awg.get(f"{id}_offset")
                elif offset_mode == 'hardware':
                    return self.awg.get(f"{id}_DC_out")
                else:
                    raise ValueError(f"Invalid {offset_mode=} mode for AWG5014.")
            elif param == 'amp':
                if self.pulsar.awgs_prequeried:
                    return self.awg.parameters[f"{id}_amp"].get_latest() / 2
                else:
                    return self.awg.get(f"{id}_amp") / 2
        else:
            # Convert ch1m1 to ch1_m1
            id_raw = id[:3] + '_' + id[3:]
            if param == 'offset':
                return self.awg.get(f"{id_raw}_low")
            elif param == 'amp':
                if self.pulsar.awgs_prequeried:
                    h = self.awg.parameters[f"{id_raw}_high"].get_latest()
                    l = self.awg.parameters[f"{id_raw}_low"].get_latest()
                else:
                    h = self.awg.get(f"{id_raw}_high")
                    l = self.awg.get(f"{id_raw}_low")
                return h - l

    def awg_setter(self, id:str, param:str, value):

        # Sanity checks
        super().awg_setter(id, param, value)

        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            offset_mode = self.pulsar.parameters[f"{id}_offset_mode"].get()
            if param == 'offset':
                if offset_mode == 'software':
                    self.awg.set(f"{id}_offset", value)
                elif offset_mode == 'hardware':
                    self.awg.set(f"{id}_DC_out", value)
                else:
                    raise ValueError(f"Invalid {offset_mode=} mode for AWG5014.")
            elif param == 'amp':
                self.awg.set(f"{id}_amp", 2 * value)
        else:
            # Convert ch1m1 to ch1_m1
            id_raw = id[:3] + '_' + id[3:]
            if param == 'offset':
                h = self.awg.get(f"{id_raw}_high")
                l = self.awg.get(f"{id_raw}_low")
                self.awg.set(f"{id_raw}_high", value + h - l)
                self.awg.set(f"{id_raw}_low", value)
            elif param == 'amp':
                l = self.awg.get(f"{id_raw}_low")
                self.awg.set(f"{id_raw}_high", l + value)

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None,
                     **kw):

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
                wfname = self.pulsar._hash_to_wavename((maxlen, wave))
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

        self.pulsar.add_awg_with_waveforms(obj.name)

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
            cname = self.pulsar._id_channel(grp, obj.name)
            offset_mode = self.get('{}_offset_mode'.format(cname))
            if offset_mode == 'hardware':
                hardware_offsets = 1
            obj.DC_output(hardware_offsets)

        return awg_file

    def is_awg_running(self):
        return self.awg.get_state() != "Idle"

    def clock(self):
        return self.awg.clock_freq()

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

    def sigout_on(self, ch, on=True):
        awg = self.find_instrument(self.get(ch + '_awg'))
        chid = self.get(ch + '_id')
        if f"{chid}_state" in awg.parameters:
            awg.set(f"{chid}_state", on)
        else:  # it is a marker channel
            # We cannot switch on the marker channel explicitly. It is on if
            # the corresponding analog channel is on. Raise a warning if
            # the state (of the analog channel) is different from the
            # requested state (of the marker channel).
            if bool(awg.get(f"{chid[:3]}_state")) != bool(on):
                log.warning(f'Pulsar: Cannot set the state of a marker '
                            f'channel. Call sigout_on for the corresponding '
                            f'analog channel {chid[:3]} instead.')
        return
