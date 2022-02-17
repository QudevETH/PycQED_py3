from unittest import TestCase

from pycqed.measurement.waveform_control.pulsar import Pulsar


class TestPulsar(TestCase):

    def test_instantiate(self):
        Pulsar("pulsar")

    def test_instantiate_with_master_awg(self):
        Pulsar("pulsar_with_master_awg", "master_awg")
