from unittest import TestCase
import random

from pycqed.measurement.waveform_control.pulsar import Pulsar

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014


class TestPulsar(TestCase):

    def setUp(self):
        """Called before every test in the class."""

        # Random instrument name, otherwise error if tests are run too fast...
        id = random.randint(1, 1000000)
        self.pulsar = Pulsar(f"pulsar_{id}")

        self.awg = VirtualAWG5014(f"awg_{id}")
        self.pulsar.define_awg_channels(self.awg)

    def test_instantiate(self):
        # Will fail if self.SetUp() fails
        pass

    def test_instantiate_with_master_awg(self):
        Pulsar("pulsar_test_instantiate_with_master_awg", "master_awg")

    def test_find_awg_channels(self):
        channels = self.pulsar.find_awg_channels(self.awg.name)

        analog_channels = [f"{self.awg.name}_ch{i}" for i in range(1, 5)]
        marker1_channels = [f"{ch}m1" for ch in analog_channels]
        marker2_channels = [f"{ch}m2" for ch in analog_channels]
        expected = analog_channels + marker1_channels + marker2_channels

        self.assertEqual(set(channels), set(expected))

    def test_get_channel_awg(self):
        awg = self.pulsar.get_channel_awg(f"{self.awg.name}_ch1")

        self.assertEqual(awg, self.awg)

    def  test_define_awg_channels(self):
        awg = VirtualAWG5014("awg_test_get_interface_class")
        self.pulsar.define_awg_channels(awg)

        self.assertIn(awg.name, self.pulsar.awgs)
        self.assertIn(awg.name, self.pulsar.awg_interfaces)
