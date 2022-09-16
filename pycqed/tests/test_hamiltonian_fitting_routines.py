from pycqed.measurement.calibration.automatic_calibration_routines. \
    hamiltonian_fitting_routines import HamiltonianFitting as hf
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon


def test_calculate_qubit_frequency_at_flux():
    qubit = QuDev_transmon('test_qubit')
    parameters_dict = dict(dac_sweet_spot=0,
                           V_per_phi0=1,
                           Ej_max=10e9,
                           E_c=5e9,
                           asymmetry=0.1)
    qubit.fit_ge_freq_from_dc_offset(parameters_dict)
    qubit_approximated_frequency = hf.calculate_qubit_frequency_at_flux(
        qubit=qubit, flux=0.0)
    assert qubit_approximated_frequency/1e9 == 15.0
    additional_parameters = dict(coupling=0e6,
                                 fr=10e9)
    qubit.fit_ge_freq_from_dc_offset().update(additional_parameters)
    qubit_precise_frequency = hf.calculate_qubit_frequency_at_flux(
        qubit=qubit, flux=0.0)
    assert isinstance(qubit_precise_frequency, float)
