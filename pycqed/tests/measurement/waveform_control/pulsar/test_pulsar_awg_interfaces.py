import random
from unittest import TestCase
from typing import List, Type
from unittest.mock import Mock

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.pulsar.pulsar import PulsarAWGInterface
from pycqed.measurement.waveform_control.pulsar.awg5014_pulsar import AWG5014Pulsar
from pycqed.measurement.waveform_control.pulsar.hdwag8_pulsar import HDAWG8Pulsar
from pycqed.measurement.waveform_control.pulsar.shfqa_pulsar import SHFQAPulsar
from pycqed.measurement.waveform_control.pulsar.uhfqc_pulsar import UHFQCPulsar

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014


registered_interfaces:List[Type[PulsarAWGInterface]] = [
    AWG5014Pulsar, HDAWG8Pulsar, SHFQAPulsar, UHFQCPulsar
]


class TestPulsarAWGInterface(TestCase):
    """Generic tests that apply to all pulsar AWG interfaces."""

    def setUp(self):
        """Called before every test in the class."""

        # Random instrument name, otherwise error if tests are run too fast...
        id = random.randint(1, 1000000)
        self.pulsar = Pulsar(f"pulsar_{id}")

        self.awg = VirtualAWG5014(f"awg_{id}")
        self.awg_interface = AWG5014Pulsar(self.pulsar, self.awg)

    def test_interfaces_are_registered(self):
        for interface in registered_interfaces:
            self.assertIn(interface, PulsarAWGInterface._pulsar_interfaces)

    def test_instantiate(self):

        pulsar = Pulsar("pulsar")

        for interface in registered_interfaces:
            with self.subTest(interface.__name__):
                awg = Mock()
                instance = interface(pulsar, awg)

    def test_get_interface_class(self):
        # Assert know class
        interface = PulsarAWGInterface.get_interface_class(VirtualAWG5014)
        self.assertEqual(interface, AWG5014Pulsar)

        # Assert instance of known class
        awg = VirtualAWG5014("awg_test_get_interface_class")
        interface = PulsarAWGInterface.get_interface_class(awg)
        self.assertEqual(interface, AWG5014Pulsar)

        # Assert error for unknown class
        with self.assertRaises(ValueError):
            PulsarAWGInterface.get_interface_class(float)

    def test_create_awg_parameters(self):
        self.awg_interface.create_awg_parameters({})

        # Common params defined in PulsarAWGInterface.create_awg_parameters()
        parameters = [
            "_active",
            "_reuse_waveforms",
            "_minimize_sequencer_memory",
            "_enforce_single_element",
            "_granularity",
            "_element_start_granularity",
            "_min_length",
            "_inter_element_deadtime",
            "_precompile",
            "_delay",
            "_trigger_channels",
            "_compensation_pulse_min_length",
        ]
        parameters = [f"{self.awg.name}{p}" for p in parameters]

        for p in parameters:
            self.assertIn(p, self.pulsar.parameters)

    def test_create_channel_parameters(self):

        for ch_type, suffix in [("analog", ""), ("marker", "m")]:

            id = "ch42"
            ch_name = f"{self.awg.name}_{id}{suffix}"
            self.awg_interface.create_channel_parameters(id, ch_name, ch_type)

            # Common params defined in PulsarAWGInterface.create_channel_parameters()
            parameters = [
                "_id",
                "_awg",
                "_type",
                "_amp",
                "_offset",
            ]
            parameters = [f"{ch_name}{p}" for p in parameters]

            analog_parameters = [
                "_distortion",
                "_distortion_dict",
                "_charge_buildup_compensation",
                "_compensation_pulse_scale",
                "_compensation_pulse_delay",
                "_compensation_pulse_gaussian_filter_sigma",
            ]
            analog_parameters = [f"{ch_name}{p}" for p in analog_parameters]

            marker_parameters = []
            marker_parameters = [f"{ch_name}{p}" for p in marker_parameters]

            if ch_type == "analog":
                all_parameters = parameters + analog_parameters
            else:
                all_parameters = parameters + marker_parameters

            for p in all_parameters:
                self.assertIn(p, self.pulsar.parameters)
