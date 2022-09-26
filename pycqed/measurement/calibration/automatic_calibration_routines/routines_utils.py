from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from typing import Literal


def get_transmon_freq_model(qubit: QuDev_transmon) -> Literal[
        'transmon', 'transmon_res']:
    """
    Determines which model will be used to calculate the frequency of a
    qubit, depending on the parameters it has.

    Arguments:
        qubit (QuDev_transmon): Qubit instance.
    The necessary parameters the qubit instance should have in its
     `fit_ge_freq_from_dc_offset()` dict from previous experiments are:
        - 'dac_sweet_spot' (in V)
        - 'V_per_phi0' (in V)
        - 'Ej_max' (in Hz)
        - 'E_c' (in Hz)
        - 'asymmetry' (a.k.a d)

    Optional parameters for a more accurate frequency model:
        - 'coupling'
        - 'fr'

    """
    qubit_hamiltonian_params = qubit.fit_ge_freq_from_dc_offset()
    assert all([k in qubit_hamiltonian_params for k in
                ['dac_sweet_spot', 'V_per_phi0', 'Ej_max', 'E_c',
                 'asymmetry']]), (
        "To calculate the frequency of a transmon, a sufficient model "
        "must be present in the qubit object")

    if all([k in qubit_hamiltonian_params for k in ['coupling', 'fr']]):
        # Use the more accurate model, that takes the resonator into account
        return 'transmon_res'

    else:
        # Use the model that takes only the transmon into account
        return 'transmon'
