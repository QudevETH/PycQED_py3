"""
This module contains different functions to calculate properties of transmon
qubits and resonators coupled to them. In particular it includes

i) functions that calculate the energy levels for the full transmon model,
including a coupled resonator according to Koch et al., Phys. Rev. A, 76,
042319 (2007) eqs. (2.1) and (3.1), and  the inverse functions, calculating
the Hamiltonian parameters from experimental quantities,

ii) function to calculate, by finding the eigenvalues of pseudo-Hamiltonians,
the effective linewidth of a Purcell-protected readout resonator and the
Purcell-limited lifetime of a qubit coupled to one or two series resonators

iii) functions to calculate the dispersive shifts of energy levels and
transitions of two coupled anharmonic oscillators, as calculated from second
order perturbation theory, and

iv) functions to calculate the process and average fidelities of CZ gates in the
presence of phase and swap errors, decomposed as a Fourier sum.
"""

import numpy as np
import scipy as sp
import scipy.optimize
from typing import Optional, List, Tuple
import logging
log = logging.getLogger(__name__)


def transmon_charge(ng: float = 0., dim_charge: int = 31):
    """Calculate the transmon charge operator.

    Args:
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A (dim_charge x dim_charge) numpy matrix, representing the charge
        operator.
    """
    return np.diag(np.arange(dim_charge) - np.floor(dim_charge / 2) + ng)


def transmon_hamiltonian(ec: float, ej: float, ng: float = 0.,
                         dim_charge: int = 31):
    """Calculate the transmon Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A (dim_charge x dim_charge) numpy matrix, representing the Hamiltonian.
    """
    ham = 4 * ec * transmon_charge(ng, dim_charge)**2
    ham -= 0.5 * ej * (np.diag(np.ones(dim_charge - 1), 1) +
                       np.diag(np.ones(dim_charge - 1), -1))
    return ham


def transmon_levels(ec: float, ej: float, ng: float = 0., dim_charge: int = 31):
    """Calculate the eigenfrequencies of the transmon Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        A list of eigenvalues of the transmon Hamiltonian with the ground-state
        energy subtracted and removed.
    """
    ham = transmon_hamiltonian(ec, ej, ng, dim_charge)
    evals = np.sort(np.linalg.eigvalsh(ham))
    return evals[1:] - evals[0]


def transmon_ec_ej(fge: float, anh: float, ng: float = 0.,
                   dim_charge: int = 31):
    """Calculate the Hamiltonian parameters of a transmon.

    Inverts the function `transmon_levels`.

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        The charging energy and the Josephson energy of the transmon.
    """

    def func(ec_ej_, fge_anh_, ng_, dim_charge_):
        fs = transmon_levels(*ec_ej_, ng_, dim_charge_)
        return - fge_anh_ + [fs[0], fs[1] - 2 * fs[0]]

    ec0 = -anh
    ej0 = -(fge - anh)**2 / 8 / anh
    ec_ej = sp.optimize.fsolve(func, np.array([ec0, ej0]),
                               args=(np.array([fge, anh]), ng, dim_charge))
    return ec_ej[0], ec_ej[1]


def transmon_ej_anh(fge: float, ec: float, ng: float = 0.,
                    dim_charge: int = 31):
    """Calculate the Josephson energy and the anharmonicity of a transmon.

    Inverts the function `transmon_levels`. Useful for finding the Josephson
    energy at a new flux bias point.

    Args:
        fge: The first transition frequency of the transmon.
        ec: Charging energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        The Josephson energy and the anharmonicity of the transmon.
    """

    def func(ej_anh_, fge_ec, ng_, dim_charge_):
        fs = transmon_levels(fge_ec[1], ej_anh_[0], ng_, dim_charge_)
        return [fs[0] - fge_ec[0], fs[1] - 2 * fs[0] - ej_anh_[1]]

    ej0 = (fge + ec)**2 / 8 / ec
    anh0 = -ec
    ej_anh = sp.optimize.fsolve(func, np.array([ej0, anh0]),
                                args=([fge, ec], ng, dim_charge))
    return ej_anh[0], ej_anh[1]


def transmon_ej_fge(fef: float, ec: float, ng: float = 0.,
                    dim_charge: int = 31):
    """Calculate the Josephson energy and the excitation frequency of a transmon

    Inverts the function `transmon_levels`. Useful for finding the Josephson
    energy at a new flux bias point.

    Args:
        fef: The second transition frequency of the transmon.
        ec: Charging energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        The Josephson energy and the anharmonicity of the transmon.
    """

    def func(ej_fge_, fef_ec, ng_, dim_charge_):
        fs = transmon_levels(fef_ec[1], ej_fge_[0], ng_, dim_charge_)
        return [fs[0] - ej_fge_[1], fs[1] - fs[0] - fef_ec[0]]

    ej0 = (fef + 2 * ec)**2 / 8 / ec
    fge0 = fef + ec
    ej_fge = sp.optimize.fsolve(func, np.array([ej0, fge0]),
                                args=([fef, ec], ng, dim_charge))
    return ej_fge[0], ej_fge[1]


def charge_dispersion_ge_ef(fge: Optional[float] = None,
                            anh: Optional[float] = None,
                            ej: Optional[float] = None,
                            ec: Optional[float] = None,
                            ng: float = 0., dim_charge: int = 31):
    """Calculate charge dispersion for first two transitions of the transmon.

    From the set of parameters fge, anh, ej and ec, one of the following pairs
    should be passed in: (fge, anh), (fge, ec) or (ej, ec).

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        ej: Josephson energy of the Hamiltonian.
        ec: Charging energy of the Hamiltonian.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        Charge dispersion of the first and second transition of the transmon.
    """
    if fge is not None and ec is not None:
        if ej is not None and anh is not None:
            raise ValueError('Too many parameters passed to '
                             '`charge_dispersion_ge_ef`')
        ej, anh = transmon_ej_anh(fge, ec, ng, dim_charge)
    elif fge is not None and anh is not None:
        if ec is not None and ej is not None:
            raise ValueError('Too many parameters passed to '
                             '`charge_dispersion_ge_ef`')
        ec, ej = transmon_ec_ej(fge, anh, ng, dim_charge)

    if ec is None or ej is None:
        raise ValueError('Too few parameters passed to '
                         '`charge_dispersion_ge_ef`')

    dfreqs = transmon_levels(ec, ej, ng, dim_charge)
    dfreqs -= transmon_levels(ec, ej, ng + 0.5, dim_charge)
    return dfreqs[0], dfreqs[0] - dfreqs[1]

def transmon_resonator_gqr(ec: float, ej: float, gb: float, i0: int = 0,
                           i1: int = 1, ng: float = 0., dim_charge: int = 31):
    """Calculate the general Jaynes-Cummings Hamiltonian coupling rate.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        gb: Bare transmon-resonator coupling strength.
        i0, i1: Transmon levels coupled by the returned rate.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.

    Returns:
        Jaynes-Cummings transmon-resonator coupling rate g_{i0,i1} according
        to Eq. (3.3) in Koch2007.
    """
    ham = transmon_hamiltonian(ec, ej, ng, dim_charge)
    v, w = np.linalg.eigh(ham)
    w = w.T[np.argsort(v)]
    return np.abs(gb * (w[i0] @ transmon_charge(ng, dim_charge) @ w[i1]))

def resonator_destroy(dim_resonator: int = 10):
    return np.diag(np.sqrt(np.arange(1, dim_resonator)), 1)


def resonator_hamiltonian(frb: float, dim_resonator: int = 10):
    """Calculate the resonator Hamiltonian.

    Args:
        frb: Bare resonator frequency.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A (dim_resonator x dim_resonator) numpy matrix, representing the
        Hamiltonian.
    """
    return frb * np.diag(np.arange(dim_resonator))


def transmon_resonator_levels(ec: float, ej: float, frb: float, gb: float,
                              ng: float = 0., dim_charge: int = 21,
                              dim_resonator: int = 3,
                              states: List[Tuple[int, int]] =
                              ((1, 0), (2, 0), (0, 1), (1, 1))):
    """Calculate eigenfrequencies of the coupled transmon-resonator Hamiltonian.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        frb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.
        states: A list of tuples, corresponding to the transmon and resonator
                level indices for which to calculate the energy levels.

    Returns:
        A list of eigenvalues of the coupled transmon-resonator Hamiltonian
        for the specified states with the ground-state energy subtracted.
    """
    id_mon = np.diag(np.ones(dim_charge))
    id_res = np.diag(np.ones(dim_resonator))
    ham_mon = transmon_hamiltonian(ec, ej, ng, dim_charge)
    ham_res = resonator_hamiltonian(frb, dim_resonator)
    n_mon = transmon_charge(ng, dim_charge)
    a_res = resonator_destroy(dim_resonator)
    ham_int = gb * np.kron(n_mon, a_res + a_res.T)
    ham = np.kron(ham_mon, id_res) + np.kron(id_mon, ham_res) + ham_int

    try:
        levels_full, states_full = np.linalg.eigh(ham)
        levels_transmon, states_transmon = np.linalg.eigh(ham_mon)
        if any(np.isnan(levels_full)) or any(np.isnan(levels_transmon)):
            raise np.linalg.LinAlgError('Some eigenvalues are nan.')
    except np.linalg.LinAlgError as e:
        log.warning(f'Eigenvalue calculation in transmon_resonator_levels '
                    f'failed in first attempt: {e} Trying again.')
        levels_full, states_full = np.linalg.eigh(ham)
        levels_transmon, states_transmon = np.linalg.eigh(ham_mon)
        if any(np.isnan(levels_full)) or any(np.isnan(levels_transmon)):
            raise np.linalg.LinAlgError('Some eigenvalues are nan.')
        log.warning('Second attempt successful.')
    states_transmon = states_transmon[:, np.argsort(levels_transmon)]

    return_idxs = []
    for kt, kr in states:
        bare_state = np.kron(states_transmon[:, kt],
                             np.arange(dim_resonator) == kr)
        return_idxs.append(np.argmax(np.abs(states_full.T @ bare_state)))

    bare_state = np.kron(states_transmon[:, 0],
                         np.arange(dim_resonator) == 0)
    gs_id = np.argmax(np.abs(states_full.T @ bare_state))

    es = levels_full[return_idxs] - levels_full[gs_id]

    return es


def transmon_resonator_fge_anh_frg_chi(ec: float, ej: float, frb: float,
                                       gb: float, ng: float = 0.,
                                       dim_charge: int = 31,
                                       dim_resonator: int = 10):
    """Calculate observable frequencies of a coupled transmon-resonator system.

    Calculates the first transition frequency and anharmonicity of the
    transmon with the resonator in the ground state, the resonator frequency
    for the qubit in the ground state and the dispersive shift of the resonator.

    Args:
        ec: Charging energy of the Hamiltonian.
        ej: Josephson energy of the Hamiltonian.
        frb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) qubit transition frequency, 2) qubit anharmonicity,
        3) the resonator frequency for transmon ground state, and 4) the
        dispersive shift.
    """
    f10, f20, f01, f11 = transmon_resonator_levels(ec, ej, frb, gb, ng,
                                                   dim_charge,
                                                   dim_resonator)
    return f10, f20 - 2 * f10, f01, (f11 - f10 - f01) / 2


def transmon_resonator_ec_ej_frb_gb(fge: float, anh: float, frg: float,
                                    chi: float, ng: float = 0.,
                                    dim_charge: int = 31,
                                    dim_resonator: int = 10):
    """Calculate Hamiltonian parameters of a coupled transmon-resonator system.

    Inverts the function `transmon_resonator_fge_anh_frg_chi`.

    Args:
        fge: The first transition frequency of the transmon.
        anh: Anharmonicity of the transmon.
        frg: Dressed resonator frequency for transmon ground state.
        chi: Dispersive shift of the coupled system.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) the charging energy and 2) Josephson energy of the
        transmon, 3) the bare resonator frequency, and 4) the bare coupling
        strength.
    """

    def func(ec_ej_frb_gb_, fge_anh_frg_chi, ng_, dim_charge_, dim_resonator_):
        return -fge_anh_frg_chi + transmon_resonator_fge_anh_frg_chi(
            *ec_ej_frb_gb_, ng_, dim_charge_, dim_resonator_)

    ec0 = -anh
    ej0 = -(fge - anh)**2 / 8 / anh
    frb0 = frg
    gb0 = np.sqrt(np.abs((fge + anh - frg) * (fge - frg) * chi / anh))
    ec_ej_frb_gb = sp.optimize.fsolve(func, np.array([ec0, ej0, frb0, gb0]),
                                      args=(np.array([fge, anh, frg, chi]),
                                            ng, dim_charge, dim_resonator))
    return tuple(ec_ej_frb_gb)


def transmon_resonator_ej_anh_frg_chi(fge: float, ec: float, frb: float,
                                      gb: float, ng: float = 0.,
                                      dim_charge: int = 31,
                                      dim_resonator: int = 10):
    """Calculate Josephson energy and observable frequencies of a coupled
    transmon-resonator system from the qubit transition frequency and
    Hamiltonian parameters

    Calculates the Josephson energy, the transmon anharmonicity with the
    resonator in the ground state, the resonator frequency for the qubit in
    the ground state and the dispersive shift of the resonator.

    Args:
        ec: Charging energy of the Hamiltonian.
        fge: The first transition frequency of the transmon.
        frb: Bare resonator frequency.
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) transmon Josephson energy, 2) qubit anharmonicity,
        3) the resonator frequency for transmon ground state, and 4) the
        dispersive shift.
    """

    def func(ej_anh_frg_chi_, fge_ec_frb_gb, ng_, dim_charge_, dim_resonator_):
        fge_, ec_, frb_, gb_ = fge_ec_frb_gb
        ej, anh, frg, chi = ej_anh_frg_chi_
        calc_fge_anh_frg_chi = transmon_resonator_fge_anh_frg_chi(
            ec_, ej, frb_, gb_, ng_, dim_charge_, dim_resonator_)
        return calc_fge_anh_frg_chi - np.array([fge_, anh, frg, chi])

    anh0 = -ec
    ej0 = (fge + ec)**2 / 8 / ec
    frg0 = frb
    chi0 = -gb**2 * (fge + ec) / (fge - frb) / (fge - frb - ec) / 16
    ej_anh_frg_chi = sp.optimize.fsolve(func, np.array([ej0, anh0, frg0, chi0]),
                                        args=(np.array([fge, ec, frb, gb]),
                                              ng, dim_charge, dim_resonator))
    return tuple(ej_anh_frg_chi)


def transmon_resonator_ej_anh_frb_chi(fge: float, ec: float, frg: float,
                                      gb: float, ng: float = 0.,
                                      dim_charge: int = 31,
                                      dim_resonator: int = 10):
    """Calculate Josephson energy, resonator bare frequency and observable
    frequencies of a coupled transmon-resonator system from the qubit transition
    frequency, coupled resonator frequency and Hamiltonian parameters.

    Calculates the Josephson energy, the transmon anharmonicity with the
    resonator in the ground state, the bare resonator frequency and the
    dispersive shift of the resonator.

    Args:
        ec: Charging energy of the Hamiltonian.
        fge: The first transition frequency of the transmon.
        frg: Resonator frequency for transmon ground state
        gb: Bare transmon-resonator coupling strength.
        ng: Charge offset of the Hamiltonian.
        dim_charge: Number of charge states to use in calculations.
        dim_resonator: Number of photon number states to use in calculations.

    Returns:
        A tuple of 1) transmon Josephson energy, 2) qubit anharmonicity,
        3) bare resonator frequency, and 4) the dispersive shift.
    """

    def func(ej_anh_frb_chi_, fge_ec_frg_gb, ng_, dim_charge_, dim_resonator_):
        fge_, ec_, frg_, gb_ = fge_ec_frg_gb
        ej, anh, frb, chi = ej_anh_frb_chi_
        calc_fge_anh_frg_chi = transmon_resonator_fge_anh_frg_chi(
            ec_, ej, frb, gb_, ng_, dim_charge_, dim_resonator_)
        return calc_fge_anh_frg_chi - np.array([fge_, anh, frg_, chi])

    anh0 = -ec
    ej0 = (fge + ec)**2 / 8 / ec
    frb0 = frg
    chi0 = -gb**2 * (fge + ec) / (fge - frg) / (fge - frg - ec) / 16
    ej_anh_frb_chi = sp.optimize.fsolve(func, np.array([ej0, anh0, frb0, chi0]),
                                        args=(np.array([fge, ec, frg, gb]),
                                              ng, dim_charge, dim_resonator))
    return tuple(ej_anh_frb_chi)


@np.vectorize
def qubit_resonator_t1_limit(fq, jqr, fr, kr, angular_units: bool = False):
    """Calculate qubit T1 limit due to decay through resonator

    Args:
        fq: Qubit frequency
        jqr: Qubit-resonator coupling rate
        fr: Resonator frequency
        kr: Resonator linewidth
        angular_units: True if the inputs are specified in angular frequency,
            False for regular frequency (default False).

    Returns:
        The qubit T1 limit
    """
    m = np.array([[kr + 2j * fr, 2j * jqr],
                  [2j * jqr, 2j * fq]])
    if not angular_units:
        m *= 2 * np.pi
    return 1 / np.linalg.eigvals(m).real.min()


@np.vectorize
def qubit_resonator_purcell_t1_limit(fq, jqr, fr, jrp, fp, kp,
                                     angular_units=False):
    """Calculate qubit T1 limit due to decay through Purcell-filtered resonator

    Args:
        fq: Qubit frequency
        jqr: Qubit-resonator coupling rate
        fr: Resonator frequency
        jrp: Resonator-Purcell-filter coupling rate
        fp: Purcell filter frequency
        kp: Purcell filter linewidth
        angular_units: True if the inputs are specified in angular frequency,
            False for regular frequency (default False).

    Returns:
        The qubit T1 limit
    """
    m = np.array([[kp + 2j * fp, 2j * jrp, 0],
                  [2j * jrp, 2j * fr, 2j * jqr],
                  [0, 2j * jqr, 2j * fq]])
    if not angular_units:
        m *= 2 * np.pi
    return 1 / np.linalg.eigvals(m).real.min()


@np.vectorize
def resonator_purcell_effective_linewidth(fr, jrp, fp, kp):
    """Calculate effective linewidth of Purcell-filtered resonator

    Args:
        fr: Resonator frequency
        jrp: Resonator-Purcell-filter coupling rate
        fp: Purcell filter frequency
        kp: Purcell filter linewidth

    Returns:
        The resonator effective linewidth
    """
    m = np.array([[kp + 2j * fp, 2j * jrp],
                  [2j * jrp, 2j * fr]])
    return np.linalg.eigvals(m).min().real


def energy_level_dispersive_shift(n1, n2, f1, ah1, f2, ah2, j=1):
    """Analytical model for energy level shift of coupled anharmonic oscillators

    Args:
        n1: Excitation number of the first oscillator
        n2: Excitation number of the second oscillator
        f1: Frequency of the first oscillator
        ah1: Anharmonicity of the first oscillator
        f2: Frequency of the second oscillator
        ah2: Anharmonicity of the second oscillator
        j: j-coupling between the oscillator

    Returns:
        The frequency shift of the state |n1, n2> as calulated according to
        second order perturbation theory
    """
    return j * j * (n1 * (n2 + 1) / (f1 + (n1 - 1) * ah1 - f2 - n2 * ah2) -
                    (n1 + 1) * n2 / (f1 + n1 * ah1 - f2 - (n2 - 1) * ah2))


def transition_dispersive_shift(n_transition, fqb, anh, fqb_spec, anh_spec, j=1,
                                state_spec=1):
    """Analytical model for trans. freq. shift of coupled anharmonic oscillators

    Args:
        n_transition: Index of the transition under consideration
        fqb: Frequency of the oscillator
        anh: Anharmonicity of the oscillator
        fqb_spec: Frequency of the neighboring oscillator
        anh_spec: Anharmonicity of the neighboring oscillator
        j: j-coupling between the oscillators
        state_spec: Excitation number of the neighboring resonator

    Returns:
        The frequency shift of the state n_transition-th transition, calculated
        as the difference of the corresponding energy levels from second order
        perturbation theory

    Examples:
        Calulating the standard residual zz coupling between the first
        transitions of two coupled qubits, detuned by 1000 MHz, with
        anharmonicity of -200 MHz, to find it equal to 83 kHz

        >>> transition_dispersive_shift(1, 5000, -200, 4000, -200, 10)
        -0.08333333333333333

        Calulating the shift of the qubit frequency due to a neighboring qubit
        being in the f-level

        >>> transition_dispersive_shift(1, 5000, -200, 4000, -200, 10, 2)
        -0.11904761904761904

        Calculating the shift of the second transition frequency when a neighbor
        is in the excited state

        >>> transition_dispersive_shift(2, 5000, -200, 4000, -200, 10)
        -0.16666666666666666
    """
    fs = [energy_level_dispersive_shift(n1, n2, fqb, anh, fqb_spec, anh_spec, j)
          for n1, n2 in [(n_transition, state_spec),
                         (n_transition - 1, state_spec),
                         (n_transition, 0),
                         (n_transition - 1, 0)]]
    return (fs[0] - fs[1]) - (fs[2] - fs[3])


def cz_process_fidelity(z1=0, z2=0, zc=0, s=0):
    r"""Analytical formula for two-qubit CZ gate process fidelity

    Calculates the process fidelity of the following unitary to the ideal
    CZ unitary

    .. math:: U = \begin{pmatrix}
        1 & 0                 & 0                 &  0                        \\
        0 & e^{-i z_1} \cos s & \sin s            &  0                        \\
        0 & \sin s            & e^{-i z_2} \cos s &  0                        \\
        0 & 0                 & 0                 & -e^{-i (z_1 + z_2 + z_c)} \\
    \end{pmatrix}.

    The validity of the Fourier-decomposition of the function was verified by
    comparing to outcomes of random inputs to the qutip direct implementation

    Args:
        z1: First qubit Z rotation error in radians
        z2: Second qubit Z rotation error in radians
        zc: Conditional Z rotation error in radians
        s: Swap rotation angle

    Returns:
        The process fidelity, defined as the trace of the product of the
        chi-matrices corresponding to the ideal and actual unitary.

    Examples:
        Calculate the two-qubit gate fidelity in the presence of frequency
        detunings of the first and second transitions of the two qubits.

        >>> j = 7  # MHz
        >>> t_gate = 1/(np.sqrt(8)*j)  # us
        >>> zeta1_ge = -0.10  # MHz
        >>> zeta2_ge = -0.07  # MHz
        >>> zeta2_ef = -0.15  # MHz
        >>> z1 = 2*np.pi*zeta1_ge*t_gate  # rad
        >>> z2 = 2*np.pi*zeta2_ge*t_gate  # rad
        >>> zc = np.pi*(zeta2_ef - zeta1_ge)*t_gate  # rad
        >>> cz_process_fidelity(z1, z2, zc)
        0.9995061478640974

        Calculate the swap error due to finite detuning of the ge<->eg
        transition during the CZ gate, involving the ee<->gf transition.
        >>> j = 7  # MHz
        >>> anh = -200  # MHz
        >>> s = np.arctan(2*j/anh)  # rad
        >>> cz_process_fidelity(s=s)
        0.9975604568019807

    """
    return (+ 6
            + 1 * np.cos(1 * z1 - 1 * z2 + 0 * zc - 2 * s)
            + 2 * np.cos(1 * z1 - 1 * z2 + 0 * zc + 0 * s)
            + 1 * np.cos(1 * z1 - 1 * z2 + 0 * zc + 2 * s)
            + 2 * np.cos(0 * z1 + 0 * z2 + 0 * zc + 2 * s)
            + 2 * np.cos(1 * z1 + 0 * z2 + 0 * zc - 1 * s)
            + 2 * np.cos(1 * z1 + 0 * z2 + 0 * zc + 1 * s)
            + 2 * np.cos(1 * z1 + 0 * z2 + 1 * zc - 1 * s)
            + 2 * np.cos(1 * z1 + 0 * z2 + 1 * zc + 1 * s)
            + 2 * np.cos(0 * z1 + 1 * z2 + 0 * zc - 1 * s)
            + 2 * np.cos(0 * z1 + 1 * z2 + 0 * zc + 1 * s)
            + 2 * np.cos(0 * z1 + 1 * z2 + 1 * zc - 1 * s)
            + 2 * np.cos(0 * z1 + 1 * z2 + 1 * zc + 1 * s)
            + 4 * np.cos(1 * z1 + 1 * z2 + 1 * zc + 0 * s)) / 32


def cz_average_fidelity(z1=0, z2=0, zc=0, s=0):
    r"""Analytical formula for two-qubit CZ gate average fidelity

    State fidelity of the output of the following unitary to the one from  the
    ideal CZ unitary, averaged over all possible input states.

    .. math:: U = \begin{pmatrix}
        1 & 0                 & 0                 &  0                        \\
        0 & e^{-i z_1} \cos s & \sin s            &  0                        \\
        0 & \sin s            & e^{-i z_2} \cos s &  0                        \\
        0 & 0                 & 0                 & -e^{-i (z_1 + z_2 + z_c)} \\
    \end{pmatrix}.

    For unitary operations :math:`F_{a} = (F_{p}*d + 1)/(d + 1)`,
    where :math:`d` is the dimension of the Hilbert space. In our case
    :math:`d=4` and :math:`F_{a} = 0.8*F_{p} + 0.2`.

    Args:
        z1: First qubit Z rotation error in radians
        z2: Second qubit Z rotation error in radians
        zc: Conditional Z rotation error in radians
        s: Swap rotation angle

    Returns:
        The average gate fidelity.
    """
    return 0.8 * cz_process_fidelity(z1, z2, zc, s) + 0.2
