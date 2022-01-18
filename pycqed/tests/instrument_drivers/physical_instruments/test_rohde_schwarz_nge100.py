import unittest

from pycqed.instrument_drivers.physical_instruments.rohde_schwarz_nge100 \
    import NGE102B, NGE103B, NGE100Channel


class TestNGE102B(unittest.TestCase):
    """Tests for the ``NGE102B`` driver and ``NGE100Channel``."""

    @classmethod
    def setUpClass(cls):
        cls.instrument = NGE102B(
            "test_nge_102", "TCPIP::192.1.2.3::INS", virtual=True
        )
        # shortcut for instrument.ch1
        cls.ch1:NGE100Channel = cls.instrument.ch1

    @classmethod
    def tearDownClass(cls):
        cls.instrument.close()

    def test_instantiation(self):
        self.assertEqual(self.instrument.name, "test_nge_102")

    def test_nb_channels(self):
        self.assertEqual(self.instrument.nb_channels, 2)
        self.assertTrue(hasattr(self.instrument, "ch1"))
        self.assertTrue(hasattr(self.instrument, "ch2"))

    def test_functions_exist(self):
        functions = ["get_system_options", "get_screenshot", "select_channel"]

        for f in functions:
            with self.subTest(f):
                self.assertTrue(hasattr(self.instrument, f))


    def test_channel_parameters_exist(self):
        parameters = [
            "voltage", "current_limit", "output", "measured_voltage",
            "measured_current", "measured_power",
        ]

        for p in parameters:
            with self.subTest(p):
                self.assertIn(p, self.ch1.parameters)

    def test_voltage_validation(self):
        # Test lower bound (0)
        with self.assertRaises(ValueError):
            self.ch1.voltage.set(-1)
        # Test upper bound (32)
        with self.assertRaises(ValueError):
            self.ch1.voltage.set(33)
    
    def test_current_limit_validation(self):
        # Test lower bound (0)
        with self.assertRaises(ValueError):
            self.ch1.current_limit.set(-0.1)
        # Test upper bound (3)
        with self.assertRaises(ValueError):
            self.ch1.current_limit.set(3.1)

    def test_simulated_output_current(self):
        # Test initial value
        self.assertEqual(self.ch1.simulated_output_current(), 0.0)

        # Test setting a value
        self.ch1.simulated_output_current.set(1.5)
        self.assertEqual(self.ch1.simulated_output_current(), 1.5)

    def test_simulated_measured_values(self):
        # Test initial values are all 0
        self.assertEqual(self.ch1.measured_voltage(), 0.0)
        self.assertEqual(self.ch1.measured_current(), 0.0)
        self.assertEqual(self.ch1.measured_power(), 0.0)

        # Test setting a value
        self.ch1.voltage.set(10)
        self.assertEqual(self.ch1.measured_voltage(), 10)
        self.ch1.simulated_output_current.set(1.5)
        self.assertEqual(self.ch1.measured_current(), 1.5)
        
        # Test power
        self.assertEqual(self.ch1.measured_power(), 10 * 1.5)

    def test_snapshot(self):
        snapshot = self.instrument.snapshot(update=True)

        self.assertIn("submodules", snapshot)
        self.assertIn("ch1", snapshot["submodules"])
        self.assertIn("ch2", snapshot["submodules"])


class TestNGE103B(unittest.TestCase):
    """Very few tests since most of the logic is common to NGE102B"""

    @classmethod
    def setUpClass(cls):
        cls.instrument = NGE103B(
            "test_nge_103", "TCPIP::192.1.2.3::INS", virtual=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.instrument.close()

    def test_instantiation(self):
        self.assertEqual(self.instrument.name, "test_nge_103")

    def test_nb_channels(self):
        self.assertEqual(self.instrument.nb_channels, 3)
