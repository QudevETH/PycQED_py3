from pycqed.instrument_drivers import mock_qcodes_interface as mqcodes
from pycqed.instrument_drivers.meta_instrument.qubit_objects import \
    qubit_calc_functions as qbcalc


class QuDev_transmon(mqcodes.Instrument, qbcalc.QubitCalcFunctionsMixIn):
    """Mock instrument including calculation function for qubits"""

    pass
