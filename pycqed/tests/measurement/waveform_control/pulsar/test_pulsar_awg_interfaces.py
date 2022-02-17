from unittest import TestCase
from typing import List, Type

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.pulsar.pulsar import PulsarAWGInterface
from pycqed.measurement.waveform_control.pulsar.awg5014_pulsar import AWG5014Pulsar
from pycqed.measurement.waveform_control.pulsar.hdwag8_pulsar import HDAWG8Pulsar
from pycqed.measurement.waveform_control.pulsar.shfqa_pulsar import SHFQAPulsar
from pycqed.measurement.waveform_control.pulsar.uhfqc_pulsar import UHFQCPulsar


registered_interfaces:List[Type[PulsarAWGInterface]] = [
    AWG5014Pulsar, HDAWG8Pulsar, SHFQAPulsar, UHFQCPulsar
]


class TestPulsar(TestCase):
    """Generic tests that apply to all pulsar AWG interfaces."""

    def test_interfaces_are_registered(self):
        for interface in registered_interfaces:
            self.assertIn(interface, PulsarAWGInterface._pulsar_interfaces)

    def test_instantiate(self):

        pulsar = Pulsar("pulsar")

        for interface in registered_interfaces:
            instance = interface(pulsar)
