import unittest

from pycqed.instrument_drivers.physical_instruments.E8527D import Agilent_E8527D


class TestAgilentE8527D(unittest.TestCase):
    """Tests for the custom ``Agilent_E8527D`` driver."""

    PARAMETERS_TO_TEST = [
        "frequency", "phase", "power", "status", "pulsemod_state"
    ]


    @classmethod
    def setUpClass(cls):
        cls.instrument = Agilent_E8527D(
            name="test_agilent_e8527d",
            address="GPIB0::7::INSTR",
            step_attenuator=True,
            virtual=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.instrument.close()

    def test_instantiation(self):
        self.assertEqual(self.instrument.name, "test_agilent_e8527d")

    def test_parameters_exist(self):

        for p in self.PARAMETERS_TO_TEST:
            with self.subTest(p):
                self.assertIn(p, self.instrument.parameters)

    def test_pulsemod_state_validation(self):
        # Test invalid
        with self.assertRaises(ValueError):
            self.instrument.pulsemod_state.set(-1)
        with self.assertRaises(ValueError):
            self.instrument.pulsemod_state.set(33)

        # Test valid
        for valid in [0, 1, "0", "1"]:
            self.instrument.pulsemod_state.set(valid)
            self.assertEqual(self.instrument.pulsemod_state.get(), bool(int(valid)))

    def test_virtual_parameters(self):

        valid_values = {
            "frequency": 250000,
            "phase": 0.78,
            "power": 20,
            "status": True,
            "pulsemod_state": True,
        }

        for parameter in self.PARAMETERS_TO_TEST:
            with self.subTest(parameter):

                # Test getting initial value
                self.assertNotEqual(
                    self.instrument.parameters[parameter].get(), 
                    None
                )

                # Test setting a value
                self.instrument.parameters[parameter].set(
                    valid_values[parameter]
                )
                self.assertAlmostEqual(
                    self.instrument.parameters[parameter].get(),
                    valid_values[parameter],
                    places=4
                )

    def test_snapshot(self):
        snapshot = self.instrument.snapshot(update=True)

        self.assertIn("parameters", snapshot)
        for parameter in self.PARAMETERS_TO_TEST:
            self.assertIn(parameter, snapshot["parameters"])
