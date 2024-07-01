import pytest

from pycqed.instrument_drivers.physical_instruments.rohde_schwarz_nge100 import (
    NGE102B,
    NGE103B,
    NGE100Channel,
)


@pytest.fixture(scope="class")
def nge102b():
    instrument = NGE102B("test_nge_102", "TCPIP::192.1.2.3::INS", virtual=True)
    # shortcut for instrument.ch1
    ch1 = instrument.ch1
    yield instrument, ch1
    instrument.close()

@pytest.fixture(scope="class")
def nge103b():
    instrument = NGE103B("test_nge_103", "TCPIP::192.1.2.3::INS", virtual=True)
    yield instrument
    instrument.close()

@pytest.mark.hardware
class TestNGE102B:

    def test_instantiation(self, nge102b):
        instrument, ch1 = nge102b
        assert instrument.name == "test_nge_102"

    def test_nb_channels(self, nge102b):
        instrument, ch1 = nge102b
        assert instrument.nb_channels == 2
        assert hasattr(instrument, "ch1")
        assert hasattr(instrument, "ch2")

    def test_functions_exist(self, nge102b):
        instrument, ch1 = nge102b
        functions = ["get_system_options", "get_screenshot", "select_channel"]

        for f in functions:
            assert hasattr(instrument, f)

    def test_channel_parameters_exist(self, nge102b):
        instrument, ch1 = nge102b
        parameters = [
            "voltage",
            "current_limit",
            "output",
            "measured_voltage",
            "measured_current",
            "measured_power",
        ]

        for p in parameters:
            assert p in ch1.parameters

    # FIXME: This overrides the hardware mark
    #@pytest.mark.parametrize("voltage", [-1, 33])
    def test_voltage_validation(voltage):
        with pytest.raises(ValueError):
            self.ch1.voltage.set(voltage)


    # FIXME: This overrides the hardware mark
    #@pytest.mark.parametrize("current_limit", [-0.1, 3.1])
    def test_current_limit_validation(current_limit):
        with pytest.raises(ValueError):
            self.ch1.current_limit.set(current_limit)


    def test_simulated_output_current(nge102b):
        ch1 = nge102b[1]
        ch1.simulated_output_current.set(1.5)
        assert ch1.simulated_output_current() == 1.5


    def test_simulated_measured_values(nge102b):
        ch1 = nge102b[1]

        assert ch1.measured_voltage() == 0.0
        assert ch1.measured_current() == 0.0
        assert ch1.measured_power() == 0.0

        ch1.voltage.set(10)
        assert ch1.measured_voltage() == 10

        ch1.simulated_output_current.set(1.5)
        assert ch1.measured_current() == 1.5
        assert ch1.measured_power() == 10 * 1.5


    def test_snapshot(nge102b):
        instrument = nge102b[0]
        snapshot = instrument.snapshot(update=True)

        assert "submodules" in snapshot
        assert "ch1" in snapshot["submodules"]
        assert "ch2" in snapshot["submodules"]


@pytest.mark.hardware
class TestNGE103B:

    def test_instantiation(self, nge103b):
        instrument = nge103b
        assert instrument.name == "test_nge_103"

    def test_nb_channels(self, nge103b):
        instrument = nge103b
        assert instrument.nb_channels == 3
        assert hasattr(instrument, "ch1")
        assert hasattr(instrument, "ch2")
        assert hasattr(instrument, "ch3")
