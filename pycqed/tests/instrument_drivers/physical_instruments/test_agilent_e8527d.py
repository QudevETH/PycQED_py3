import pytest

from pycqed.instrument_drivers.physical_instruments.E8527D import Agilent_E8527D


@pytest.mark.hardware
class TestAgilentE8527D:
    """Tests for the custom ``Agilent_E8527D`` driver."""

    PARAMETERS_TO_TEST = ["frequency", "phase", "power", "status", "pulsemod_state"]

    @classmethod
    def setup_class(cls):
        cls.instrument = Agilent_E8527D(
            name="test_agilent_e8527d",
            address="GPIB0::7::INSTR",
            step_attenuator=True,
            virtual=True,
        )

    @classmethod
    def teardown_class(cls):
        cls.instrument.close()

    def test_instantiation(self):
        assert self.instrument.name == "test_agilent_e8527d"

    def test_parameters_exist(self):

        for p in self.PARAMETERS_TO_TEST:
            assert p in self.instrument.parameters

    def test_pulsemod_state_validation(self):
        # Test invalid
        with pytest.raises(ValueError):
            self.instrument.pulsemod_state.set(-1)
        with pytest.raises(ValueError):
            self.instrument.pulsemod_state.set(33)

        # Test valid
        for valid in [0, 1, "0", "1"]:
            self.instrument.pulsemod_state.set(valid)
            assert self.instrument.pulsemod_state.get() == bool(int(valid))

    def test_virtual_parameters(self):

        valid_values = {
            "frequency": 250000,
            "phase": 0.78,
            "power": 20,
            "status": True,
            "pulsemod_state": True,
        }

        for parameter in self.PARAMETERS_TO_TEST:
            # Test getting initial value
            assert self.instrument.parameters[parameter].get() is not None

            # Test setting a value
            self.instrument.parameters[parameter].set(valid_values[parameter])
            assert (
                self.instrument.parameters[parameter].get() == valid_values[parameter]
            )

    def test_snapshot(self):
        snapshot = self.instrument.snapshot(update=True)

        assert "parameters" in snapshot
        for parameter in self.PARAMETERS_TO_TEST:
            assert parameter in snapshot["parameters"]
