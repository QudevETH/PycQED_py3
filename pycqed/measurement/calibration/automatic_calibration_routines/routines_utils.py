import numpy as np

from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from typing import Literal, Dict, Any, Tuple, Union


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


def get_transmon_anharmonicity(qubit: QuDev_transmon) -> float:
    """Get the anharmonicity of a transmon or its estimation as the charging
    energy (E_c) if no anharmonicity is found."""
    if qubit.anharmonicity():  # Not None or 0
        return qubit.anharmonicity()
    else:
        E_c = qubit.fit_ge_freq_from_dc_offset()["E_c"]
        return -E_c


def get_transmon_resonator_coupling(qubit: QuDev_transmon,
                                    uss_transmon_freq: float = None,
                                    uss_readout_freq: float = None,
                                    lss_transmon_freq: float = None,
                                    lss_readout_freq: float = None,
                                    update: bool = False) -> float:
    r"""Get the transmon-readout coupling strength or its estimation if no
    coupling is found in the qubit's attributes.

    Arguments:
        qubit: The qubit instance for which the coupling is returned.
        uss_transmon_freq: transmon frequency at upper sweet spot.
        uss_readout_freq: readout frequency at upper sweet spot.
        lss_transmon_freq: transmon frequency at lower sweet spot.
        lss_readout_freq: readout frequency at lower sweet spot.
        update: whether to update the qubit attribute with the coupling.

    The estimation equation is (see DOI: 10.1103/RevModPhys.93.025005 Eq. 45):
    .. math::
       f_r_{uss} - f_r_{lss} =
        g^2 * (\frac{1}{E_c - Delta_uss} - \frac{1}{E_c - Delta_lss})
    """
    qubit_parameters = qubit.fit_ge_freq_from_dc_offset()
    if "coupling" in qubit_parameters.keys():
        if not qubit_parameters["coupling"] == 0:
            return qubit_parameters["coupling"]
    else:
        assert all([uss_transmon_freq, uss_readout_freq,
                    lss_transmon_freq, lss_readout_freq])
        E_c = qubit_parameters["E_c"]
        readout_frequency_difference = uss_readout_freq - lss_readout_freq
        Delta_uss = uss_transmon_freq - uss_readout_freq
        Delta_lss = lss_transmon_freq - lss_readout_freq
        coefficient = (1 / (E_c - Delta_uss)) - (1 / (E_c - Delta_lss))
        g = np.sqrt(readout_frequency_difference / coefficient)

        if update:
            qubit_parameters["coupling"] = g

        return g


def append_DCsources(routine):
    """Append the DC_sources of a routine as an attribute. This is used when
    the `reload_settings` function is called after the routine ends."""
    routine.DCSources = []
    for qb in routine.qubits:
        dc_source = routine.fluxlines_dict[qb.name].instrument
        if dc_source not in routine.DCSources:
            routine.DCSources.append(dc_source)


def get_qubit_flux_and_voltage(qb: QuDev_transmon,
                               fluxlines_dict: Dict[str, Any] = None,
                               flux: Union[float, Literal['designated',
                                                          'opposite',
                                                          'mid']] = None,
                               voltage: float = None) -> Tuple[
        float, float]:
    """Get the flux and voltage values of a qubit.
    One of the three values [voltage > flux > fluxlines_dict] must be passed,
    and their hierarchy is according to this order."""

    if voltage is None and flux is not None:
        flux = flux_to_float(qb=qb, flux=flux)
        voltage = qb.calculate_voltage_from_flux(flux)
    else:
        voltage = voltage or fluxlines_dict[qb.name]()
        uss = qb.fit_ge_freq_from_dc_offset()['dac_sweet_spot']
        V_per_phi0 = qb.fit_ge_freq_from_dc_offset()['V_per_phi0']
        flux = (voltage - uss) / V_per_phi0

    return flux, voltage


def qb_is_at_designated_sweet_spot(qb: QuDev_transmon,
                                   flux: float = None,
                                   voltage: float = None,
                                   fluxlines_dict: Dict[str, Any] = None,
                                   small_flux: float = 0.1) -> bool:
    """Helps when only the voltage is known, not the flux."""
    if flux is None:
        if voltage is None:
            voltage = fluxlines_dict[qb.name]()
        flux, _ = get_qubit_flux_and_voltage(
            qb=qb,
            voltage=voltage)
    return np.abs(flux - qb.flux_parking()) < small_flux


def flux_to_float(qb: QuDev_transmon,
                  flux: Union[float, Literal['{designated}',
                                             '{opposite}',
                                             '{mid}']]) -> float:
    """
    Return the specified flux as a float, useful with descriptive fluxes.
    Args:
        qb: Qubit element.
        flux: float or literal.

    Returns: flux as float.

    """
    designated_ss_flux = qb.flux_parking()
    if designated_ss_flux == 0:
        # Qubit parked at the USS.
        # LSS will be negative in order to operate the qubit on a rising branch
        # of the freq-over-flux curve
        opposite_ss_flux = -0.5
    elif np.abs(designated_ss_flux) == 0.5:
        # Qubit parked at the LSS
        opposite_ss_flux = 0
    else:
        raise ValueError("Only Sweet Spots are supported!")
    mid_flux = (designated_ss_flux + opposite_ss_flux) / 2

    if isinstance(flux, str):
        flux = eval(
            flux.format(designated=designated_ss_flux,
                        opposite=opposite_ss_flux,
                        mid=mid_flux))

    return float(flux)
