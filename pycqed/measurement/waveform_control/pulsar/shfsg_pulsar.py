import logging
from typing import List, Tuple

import numpy as np
from copy import deepcopy

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter
try:
    from zhinst.qcodes import SHFSG as SHFSG_core
except Exception:
    SHFSG_core = type(None)

from .pulsar import PulsarAWGInterface


log = logging.getLogger(__name__)


class SHFGeneratorModulePulsar(PulsarAWGInterface):
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
    # TODO: SHFQA had no parameter for offset, should we delete it for this
    # subclass, or just force it to 0 ?
    CHANNEL_OFFSET_BOUNDS = {
        "analog": (0, 0),
    }
    IMPLEMENTED_ACCESSORS = ["amp"]

    def create_awg_parameters(self, channel_name_map: dict):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        # Repeat pattern support is not yet implemented for the SHFSG, thus we
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
        for ch_nr in range(len(self.awg.sgchannels)):
            group = []
            for q in ["i", "q"]:
                id = f"ch{ch_nr + 1}d{q}"
                ch_name = channel_name_map.get(id, f"{name}_{id}")
                self.create_channel_parameters(id, ch_name, "analog")
                pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFSG, valid channel ids are ch#i and ch#q, where # is a number
        from 1 to 8. This defines the harware port used.
        """

        super().create_channel_parameters(id, ch_name, ch_type)

        # TODO: Not all AWGs provide an initial value. Should it be the case?
        self.pulsar[f"{ch_name}_amp"].set(1)

    def awg_setter(self, id:str, param:str, value):
        # Sanity checks
        super().awg_setter(id, param, value)

        ch = int(id[2]) - 1

        if param == "amp":
            self.awg.sgchannels[ch].output.range(20 * (np.log10(value) + 0.5))

    def awg_getter(self, id:str, param:str):
        # Sanity checks
        super().awg_getter(id, param)

        ch = int(id[2]) - 1

        if param == "amp":
            if self.pulsar.awgs_prequeried:
                dbm = self.awg.sgchannels[ch].output.range.get_latest()
            else:
                dbm = self.awg.sgchannels[ch].output.range()
            return 10 ** (dbm / 20 - 0.5)

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        # TODO: Write this code
        pass

    def is_awg_running(self):
        is_running = []
        for awg_nr, sgchannel in enumerate(self.awg.sgchannels):
            is_running.append(sgchannel.awg.enable())
        return any(is_running)

    def clock(self):
        return 2.0e9

    def sigout_on(self, ch, on=True):
        chid = self.pulsar.get(ch + '_id')
        self.awg.qachannels[int(chid[2]) - 1].output.on(on)


class SHFSGPulsar(SHFSG_core):
    """ZI SHFQA specific Pulsar module"""
