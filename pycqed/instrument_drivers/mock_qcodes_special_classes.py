from pycqed.instrument_drivers import mock_qcodes_interface as mqcodes
from pycqed.instrument_drivers.meta_instrument.qubit_objects import \
    qubit_calc_functions as qbcalc
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments import \
    snapshot_whitelist as snw


class QuDev_transmon(mqcodes.Instrument, qbcalc.QubitCalcFunctionsMixIn):
    """Mock instrument including calculation function for qubits"""

    pass


class ZI_HDAWG_qudev(mqcodes.Instrument):
    _snapshot_whitelist = \
        snw.generate_snapshot_whitelist_hdawg()


class UHFQA(mqcodes.Instrument):
    _snapshot_whitelist = \
        snw.generate_snapshot_whitelist_uhfqa()
