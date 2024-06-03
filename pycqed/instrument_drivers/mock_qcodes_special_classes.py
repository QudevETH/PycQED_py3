from pycqed.instrument_drivers import mock_qcodes_interface as mqcodes
from pycqed.instrument_drivers.meta_instrument.qubit_objects import \
    qubit_calc_functions as qbcalc


class QuDev_transmon(mqcodes.Instrument, qbcalc.QubitCalcFunctionsMixIn):
    """Mock instrument including calculation function for qubits"""
    # This class is a mock class for the QuDev_transmon instrument. It has to
    # have the same name as the real instrument which it should imitate.
    # Compare _load_custom_mock_class in mock_qcodes_interface.py.
    # This would lead to an issue if multiple qcodes instrument classes have
    # the same name and are supposed to use different mock classes. But this
    # is not the case in the current implementation.

    pass
