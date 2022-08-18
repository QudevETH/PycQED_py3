import logging

from qcodes import ManualParameter
import qcodes.utils.validators as vals

from .shfqa_pulsar import SHFAcquisitionModulePulsar
from .shfsg_pulsar import SHFGeneratorModulePulsar

try:
    from zhinst.qcodes import SHFQC as SHFQC_core
except Exception:
    SHFQC_core = type(None)

log = logging.getLogger(__name__)


class SHFQCPulsar(SHFAcquisitionModulePulsar, SHFGeneratorModulePulsar):
    """ZI SHFQC specific Pulsar module"""
    AWG_CLASSES = [SHFQC_core]

    GRANULARITY = 16  # maximum of QA and SG granularities
    ELEMENT_START_GRANULARITY = 16 / 2.0e9  # TODO: unverified!
    MIN_LENGTH = 32 / 2.0e9  # maximum of QA and SG min. lengths
    INTER_ELEMENT_DEADTIME = 0  # TODO: unverified!
    # Lower bound is the one of the SG channels and is lower than the one of
    # the QA channels (-30 dBm ~= 0.01 Vp vs -40 dBm ~= 0.0031 Vp).
    CHANNEL_AMPLITUDE_BOUNDS = {
        "analog": (0.0031, 1),
    }
    IMPLEMENTED_ACCESSORS = ["amp", "centerfreq"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_awg_parameters(self, channel_name_map: dict):
        super().create_awg_parameters(channel_name_map)

        pulsar = self.pulsar
        name = self.awg.name

        pulsar.add_parameter(f"{name}_use_placeholder_waves",
                             initial_value=False, vals=vals.Bool(),
                             parameter_class=ManualParameter)
        pulsar.add_parameter(f"{name}_trigger_source",
                             initial_value="Dig1",
                             vals=vals.Enum("Dig1",),
                             parameter_class=ManualParameter,
                             docstring="Defines for which trigger source the "
                                       "AWG should wait, before playing the "
                                       "next waveform. Only allowed value is "
                                       "'Dig1 for now.")
        pulsar.add_parameter(f"{name}_use_hardware_sweeper",
                             initial_value=False,
                             parameter_class=ManualParameter,
                             docstring='Bool indicating whether the hardware '
                                       'sweeper should be used in for '
                                       'spectroscopies on the SG channels ',
                             vals=vals.Bool())

        SHFAcquisitionModulePulsar._create_all_channel_parameters(
            self, channel_name_map)
        SHFGeneratorModulePulsar._create_all_channel_parameters(
            self, channel_name_map)

    @classmethod
    def _get_superclass(cls, id):
        return SHFAcquisitionModulePulsar if 'qa' in id \
            else SHFGeneratorModulePulsar

    def create_channel_parameters(self, id:str, ch_name:str, ch_type:str):
        """See :meth:`PulsarAWGInterface.create_channel_parameters`.

        For the SHFQC, valid channel ids are sg#i, sg#q, qa1i and qa1q, where #
        is a number from 1 to 6. This defines the harware port used.
        """
        return self._get_superclass(id).create_channel_parameters(
            self, id, ch_name, ch_type)

    def awg_setter(self, id:str, param:str, value):
        return self._get_superclass(id).awg_setter(self, id, param, value)

    def awg_getter(self, id:str, param:str):
        return self._get_superclass(id).awg_getter(self, id, param)

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
        return self._get_superclass(id).sigout_on(self, ch, on=on)

    def start(self):
        SHFAcquisitionModulePulsar.start(self)
        SHFGeneratorModulePulsar.start(self)

    def stop(self):
        SHFAcquisitionModulePulsar.stop(self)
        SHFGeneratorModulePulsar.stop(self)

    def get_params_for_spectrum(self, ch: str, requested_freqs: list[float]):
        id = self.pulsar.get(ch + '_id')
        return self._get_superclass(id) \
            .get_params_for_spectrum(self, ch, requested_freqs)

    def get_frequency_sweep_function(self, ch: str):
        id = self.pulsar.get(ch + '_id')
        return self._get_superclass(id) \
            .get_frequency_sweep_function(self, ch)