import random
import time
from unittest import TestCase
from typing import List, Tuple, Type
from qcodes.instrument import Instrument

from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control.segment import Segment
from pycqed.measurement.waveform_control.pulsar.pulsar import PulsarAWGInterface
from pycqed.measurement.waveform_control.pulsar.awg5014_pulsar import AWG5014Pulsar
from pycqed.measurement.waveform_control.pulsar.hdwag8_pulsar import HDAWG8Pulsar
from pycqed.measurement.waveform_control.pulsar.shfqa_pulsar import SHFQAPulsar
from pycqed.measurement.waveform_control.pulsar.uhfqc_pulsar import UHFQCPulsar

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments\
    .ZI_HDAWG_qudev import ZI_HDAWG_qudev
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments\
    .UHFQA_core import UHFQA_core


registered_interfaces:List[Type[PulsarAWGInterface]] = [
    AWG5014Pulsar, HDAWG8Pulsar, SHFQAPulsar, UHFQCPulsar
]


def assert_has_parameters(tester:TestCase, instrument:Instrument,
                          parameters:List[str]):

    for p in parameters:
        tester.assertIn(p, instrument.parameters)


class TestPulsarAWGInterface(TestCase):
    """Generic tests that apply to all pulsar AWG interfaces."""

    def setUp(self):
        """Called before every test in the class."""

        # Random instrument name, otherwise error if tests are run too fast...
        id = random.randint(1, 1000000)
        self.pulsar = Pulsar(f"pulsar_{id}")

        self.awg5014 = VirtualAWG5014(f"awg5014_{id}", timeout=20,
                                      address='TCPIP0::192.168.1.4')

        self.hdawg = ZI_HDAWG_qudev(f"hdawg_{id}", device="dev8188",
                                    interface="1GbE", server="emulator")

        # TODO: SHFQ currently has no virtual driver, so we do not unit test it.
        # self.shfqa = SHFQA(f"shfqa_{id}", device="dev8188",
        #                    interface="1GbE", server="emulator")

        self.uhfqc = UHFQA_core(f"uhfqc_{id}", device='dev2268',
                                interface='1GbE', server="emulator")

        self.awgs:List[Tuple[Instrument, Type[PulsarAWGInterface]]] = [
            (self.awg5014, AWG5014Pulsar),
            (self.hdawg, HDAWG8Pulsar),
            # (self.shfqa, SHFQAPulsar),
            (self.uhfqc, UHFQCPulsar),
        ]

    def test_interfaces_are_registered(self):
        for interface in registered_interfaces:
            self.assertIn(interface, PulsarAWGInterface._pulsar_interfaces)

    def test_instantiate(self):
        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):
                awg_interface = interface_class(self.pulsar, awg)

                self.assertEqual(awg, awg_interface.awg)

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

        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):
                awg_interface = interface_class(self.pulsar, awg)
                awg_interface.create_awg_parameters({})

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
                parameters = [f"{awg.name}{p}" for p in parameters]

                assert_has_parameters(self, self.pulsar, parameters)

    def test_create_channel_parameters(self):

        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):
                awg_interface = interface_class(self.pulsar, awg)

                for ch_type, suffix in [("analog", ""), ("marker", "m")]:

                    if awg_interface.__class__ in [UHFQCPulsar, SHFQAPulsar] \
                       and ch_type == "marker":
                        # These classes do not use marker channels
                        continue

                    id = "ch1"
                    ch_name = f"{awg.name}_{id}{suffix}"
                    awg_interface.create_channel_parameters(id, ch_name, ch_type)

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

                    assert_has_parameters(self, self.pulsar, all_parameters)

    def test_program_awg(self):

        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):

                # Register AWG in pulsar
                self.pulsar.define_awg_channels(awg)
                awg_interface = self.pulsar.awg_interfaces[awg.name]

                # Generate sequence
                pulses = [{
                    "name": f"pulse",
                    "pulse_type": "SquarePulse",
                    "pulse_delay": 0,
                    "ref_pulse": "previous_pulse",
                    "ref_point": "end",
                    "length": 5e-8,
                    "amplitude": 0.05,
                    "channels": [f"{awg.name}_ch1"],
                    "channel": f"{awg.name}_ch1",
                }]
                segment = Segment("segment", pulses)
                sequence = Sequence("sequence", segments=[segment])
                waveforms, awg_sequences = sequence.generate_waveforms_sequences()

                # Actual function to test
                awg_interface.program_awg(
                    awg_sequences[awg.name],
                    waveforms,
                )

    def test_awg_getter_setter(self):

        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):

                # Register AWG in pulsar
                self.pulsar.define_awg_channels(awg)
                awg_interface = self.pulsar.awg_interfaces[awg.name]

                for param in awg_interface.IMPLEMENTED_ACCESSORS:
                    awg_interface.awg_setter("ch1", param, 0.0123)
                    v = awg_interface.awg_getter("ch1", param)

                    # TODO: Fix the underlying behavior of the UHFQC
                    if param == "amp" and "uhfqc" in awg.name:
                        continue

                    self.assertEqual(v, 0.0123)

    def test_start_stop_is_running(self):
        """Test starting/stopping/is_running on the AWGs.

        Some AWGs (e.g. HDAWG) will not start if they have no waveforms to play,
        thus we need to upload them to make sure the methods work.
        """

        for awg, interface_class in self.awgs:
            with self.subTest(interface_class):

                # TODO AWG5014 fails because virtual instrument not implemented
                # correctly, missing visa_log and other attributes.
                if "awg5014" in awg.name:
                    continue

                # Register AWG in pulsar
                self.pulsar.define_awg_channels(awg)
                awg_interface = self.pulsar.awg_interfaces[awg.name]

                # Generate sequence
                pulses = [{
                    "name": f"pulse",
                    "pulse_type": "SquarePulse",
                    "pulse_delay": 0,
                    "ref_pulse": "previous_pulse",
                    "ref_point": "end",
                    "length": 5e-8,
                    "amplitude": 0.05,
                    "channels": [f"{awg.name}_ch1"],
                    "channel": f"{awg.name}_ch1",
                }]
                segment = Segment("segment", pulses)
                sequence = Sequence("sequence", segments=[segment])
                waveforms, awg_sequences = sequence.generate_waveforms_sequences()

                awg_interface.program_awg(
                    awg_sequences[awg.name],
                    waveforms,
                )

                # Test start
                awg_interface.start()
                self.assertTrue(awg_interface.is_awg_running())

                # Test stop
                awg_interface.stop()
                self.assertFalse(awg_interface.is_awg_running())
