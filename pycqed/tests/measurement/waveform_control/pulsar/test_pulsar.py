from unittest import TestCase
from unittest.mock import patch, Mock, mock_open
import os
import random

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.segment import Segment

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

    def test_define_awg_channels(self):
        awg = VirtualAWG5014("awg_test_get_interface_class")
        self.pulsar.define_awg_channels(awg)

        self.assertIn(awg.name, self.pulsar.awgs)
        self.assertIn(awg.name, self.pulsar.awg_interfaces)

    def test_program_awgs(self):
        """We run this test with a single AWG, just to check that the Pulsar
        method itself has no errors.

        Each AWG interface are tested in ``test_pulsar_awg_interfaces.py``.
        """

        pulses = [{
            "name": f"pulse",
            "pulse_type": "SquarePulse",
            "pulse_delay": 0,
            "ref_pulse": "previous_pulse",
            "ref_point": "end",
            "length": 5e-8,
            "amplitude": 0.05,
            "channels": [f"{self.awg.name}_ch1"],
            "channel": f"{self.awg.name}_ch1",
        }]

        segment = Segment("segment", pulses)
        sequence = Sequence("sequence", segments=[segment])
        self.pulsar.program_awgs(sequence)

    def test_reset_sequence_cache(self):

        self.pulsar.reset_sequence_cache()

        self.assertTrue(hasattr(self.pulsar, "_sequence_cache"))
        self.assertTrue(isinstance(self.pulsar._sequence_cache, dict))

        for key in ["settings", "metadata", "hashes", "length"]:
            self.assertTrue(self.pulsar._sequence_cache.get(key, None) == {})

    def test_check_for_other_pulsar(self):

        # Create another pulsar, make it override check file
        pulsar2 = Pulsar("pulsar_test_check_for_other_pulsar")
        pulsar2._write_pulsar_check_file()

        # Another pulsar should be detected
        self.assertTrue(self.pulsar.check_for_other_pulsar())

        # Pulsar is now correct instance, so cache should not be reset
        self.pulsar._write_pulsar_check_file()
        self.assertFalse(self.pulsar.check_for_other_pulsar())

        # Test case where check file does not exist yet
        open_raise_file_not_found= mock_open(Mock(side_effect=FileNotFoundError))
        with patch("builtins.open", new_callable=open_raise_file_not_found):
            self.assertTrue(self.pulsar.check_for_other_pulsar())
