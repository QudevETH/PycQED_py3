import logging

from .shfqa_pulsar import SHFAcquisitionModulePulsar
from .shfsg_pulsar import SHFGeneratorModulePulsar

try:
    from zhinst.qcodes import SHFQC as SHFQC_core
except Exception:
    SHF_AcquisitionDevice = type(None)

log = logging.getLogger(__name__)


class SHFQCPulsar(SHFAcquisitionModulePulsar, SHFGeneratorModulePulsar):
    """ZI SHFQC specific Pulsar module"""
    AWG_CLASSES = [SHFQC_core]

    GRANULARITY = 16  # maximum of QA and SG granularities
    ELEMENT_START_GRANULARITY = 16 / 2.0e9  # TODO: unverified!
    MIN_LENGTH = 32 / 2.0e9  # maximum of QA and SG min. lengths
    INTER_ELEMENT_DEADTIME = 0  # TODO: unverified!
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.001, 1),
    }
    # TODO: SHFQC has no parameter for offset, should we delete it for this
    # subclass, or just force it to 0 ?
    CHANNEL_OFFSET_BOUNDS = {
        "analog": (0, 0),
    }
    IMPLEMENTED_ACCESSORS = ["amp"]

    def create_awg_parameters(self, channel_name_map: dict):
        SHFAcquisitionModulePulsar.create_awg_parameters(self)

        # real and imaginary part of the wave form channel groups
        for ch_nr in range(len(self.awg.sgchannels)):
            group = []
            for q in ["i", "q"]:
                id = f"ch{ch_nr + 1}d{q}"
                ch_name = channel_name_map.get(id, f"{self.awg.name}_{id}")
                self.create_channel_parameters(id, ch_name, "analog")
                self.pulsar.channels.add(ch_name)
                group.append(ch_name)
            for ch_name in group:
                self.pulsar.channel_groups.update({ch_name: group})

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFQC, valid channel ids are ch#i and ch#q, where # is a number
        from 1 to 8 except 2. This defines the harware port used.
        """
        fn = SHFAcquisitionModulePulsar.create_channel_parameters \
            if 'r' in id else SHFGeneratorModulePulsar.create_channel_parameters
        return staticmethod(fn)(self, id, ch_name, ch_type)

    def awg_setter(self, id:str, param:str, value):
        fn = SHFAcquisitionModulePulsar.awg_setter if 'r' in id else \
            SHFGeneratorModulePulsar.awg_setter
        return staticmethod(fn)(self, id, param, value)

    def awg_getter(self, id:str, param:str):
        fn = SHFAcquisitionModulePulsar.awg_getter if 'r' in id else \
            SHFGeneratorModulePulsar.awg_getter
        return staticmethod(fn)(self, id, param)

    def program_awg(self, awg_sequence, waveforms, repeat_pattern=None,
                    channels_to_upload="all", channels_to_program="all"):
        args = [self, awg_sequence, waveforms]
        kwargs = dict(repeat_pattern=repeat_pattern,
                      channels_to_upload=channels_to_upload,
                      channels_to_program=channels_to_program)
        SHFAcquisitionModulePulsar.program_awg(*args, **kwargs)
        SHFGeneratorModulePulsar.program_awg(*args, **kwargs)

    def is_awg_running(self):
        return SHFAcquisitionModulePulsar.is_awg_running(self) or \
               SHFGeneratorModulePulsar.is_awg_running(self)

    def sigout_on(self, ch, on=True):
        id = self.pulsar.get(ch + '_id')
        fn = SHFAcquisitionModulePulsar.sigout_on if 'r' in id else \
            SHFGeneratorModulePulsar.sigout_on
        return staticmethod(fn)(self, ch, on=on)

