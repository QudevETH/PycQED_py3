import numpy as np
import matplotlib.pyplot as plt
import scipy
import lmfit
import itertools
import logging


#################################
#   Fitting Functions Library   #
#################################

def RandomizedBenchmarkingLeakage(numCliff, pu, pd, p0):
    val = pu/(pd+pu) * (1-np.exp(-(pd+pu)*numCliff)) + p0*np.exp(-(pd+pu)*numCliff)
    return val

def RandomizedBenchmarkingDecay(numCliff, Amplitude, p, offset):
    val = Amplitude * (p ** numCliff) + offset
    return val


def DoubleExpDampOscFunc(t, tau_1, tau_2,
                         freq_1, freq_2,
                         phase_1, phase_2,
                         amp_1, amp_2, osc_offset):
    cos_1 = amp_1 * (np.cos(2 * np.pi * freq_1 * t + phase_1)) * np.exp(-(t / tau_1))
    cos_2 = amp_2 * (np.cos(2 * np.pi * freq_2 * t + phase_2)) * np.exp(-(t / tau_2))
    return cos_1 + cos_2 + osc_offset


def double_RandomizedBenchmarkingDecay(numCliff, p, offset,
                                       invert=1):
    """
    A variety of the RB-curve that allows fitting both the inverting and
    non-inverting exponential.
    The amplitude of the decay curve is constrained to start at 0 or 1.
    The offset is the common point both curves converge to.

    pick invert to be 1 or 0
    """
    # Inverting clifford curve
    val_inv = (1 - offset) * (p ** numCliff) + offset
    # flipping clifford curve
    val_flip = -offset * (p ** numCliff) + offset
    # Using invert as a boolean but not using if statement to allow for
    # arrays to be input in the function
    val = (1 - invert) * val_flip + invert * val_inv
    return val


def LorentzFunc(f, amplitude, center, sigma):
    val = amplitude / np.pi * (sigma / ((f - center) ** 2 + sigma ** 2))
    return val


def Lorentzian(f, A, offset, f0, kappa):
    val = offset + A / np.pi * (kappa / ((f - f0) ** 2 + kappa ** 2))
    return val


def TwinLorentzFunc(f, A_gf_over_2, A, f0_gf_over_2, f0,
                    kappa_gf_over_2, kappa, background=0):
    """
    Twin lorentz with background.

    Args:
    f (float):          frequency sweep points in Hz
    A (float):          amplitude of the tallest/deepest Lorentzian structure
                        in the data
    A_gf_over_2 (float):    amplitude of the other Lorentzian structure in the
                            data; since this function is used for high power
                            qubit spectroscopy, this parameter refers to the
                            Lorentzian structure corresponding to the gf/2
                            transition
    f0 (float):         frequency of the tallest/deepest Lorentzian structure
                        in the data
    f0_gf_over_2 (float):   frequency of the other Lorentzian structure in the
                            data; since this function is used for high power
                            qubit spectroscopy, this parameter refers to the
                            Lorentzian structure corresponding to the gf/2
                            transition
    kappa (float):      kappa (FWHM) of the tallest/deepest Lorentzian structure
                        in the data
    kappa_gf_over_2 (float): kappa (FWHM) of the other Lorentzian structure in
                             the data; since this function is used for high
                             power qubit spectroscopy, this parameter refers to
                             the Lorentzian structure corresponding to the gf/2
                             transition
    background (float):     background offset
    """
    val = (A_gf_over_2 / np.pi * (kappa_gf_over_2 / ((f - f0_gf_over_2) ** 2 + kappa_gf_over_2 ** 2)) +
           A / np.pi * (kappa / ((f - f0) ** 2 + kappa ** 2)) + background)
    return val


def Qubit_dac_to_freq(dac_voltage, f_max,
                      dac_sweet_spot, V_per_phi0=None,
                      dac_flux_coefficient=None,
                      asymmetry=0):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.

    dac_voltage (V)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2)),
    '''

    E_c = 0

    if V_per_phi0 is None and dac_flux_coefficient is None:
        raise ValueError('Please specify "V_per_phi0".')

    if dac_flux_coefficient is not None:
        logging.warning('"dac_flux_coefficient" deprecated. Please use the '
                        'physically meaningful "V_per_phi0" instead.')
        V_per_phi0 = np.pi / dac_flux_coefficient

    qubit_freq = (f_max + E_c) * (
            asymmetry ** 2 + (1 - asymmetry ** 2) *
            np.cos(np.pi / V_per_phi0 *
                   (dac_voltage - dac_sweet_spot)) ** 2) ** 0.25 - E_c
    return qubit_freq


def Resonator_dac_to_freq(dac_voltage, f_max_qubit, f_0_res,
                          E_c, dac_sweet_spot,
                          coupling, V_per_phi0=None,
                          dac_flux_coefficient=None,
                          asymmetry=0):
    qubit_freq = Qubit_dac_to_freq(dac_voltage=dac_voltage, f_max=f_max_qubit, E_c=E_c,
                                   dac_sweet_spot=dac_sweet_spot, V_per_phi0=V_per_phi0,
                                   dac_flux_coefficient=dac_flux_coefficient,
                                   asymmetry=asymmetry)
    delta_qr = (qubit_freq - f_0_res)
    lamb_shift = (coupling ** 2 / delta_qr)
    resonator_freq = f_0_res - lamb_shift

    return resonator_freq


def Qubit_dac_to_detun(dac_voltage, f_max, E_c, dac_sweet_spot, V_per_phi0,
                       asymmetry=0):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.

    dac_voltage (V)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    asymmetry (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2))
    '''
    return f_max - Qubit_dac_to_freq(dac_voltage,
                                     f_max=f_max, E_c=E_c,
                                     dac_sweet_spot=dac_sweet_spot,
                                     V_per_phi0=V_per_phi0,
                                     asymmetry=asymmetry)


def Qubit_freq_to_dac(frequency, f_max,
                      dac_sweet_spot, V_per_phi0=None,
                      dac_flux_coefficient=None, asymmetry=0,
                      branch='positive'):
    '''
    The cosine Arc model for uncalibrated flux for asymmetric qubit.
    This function implements the inverse of "Qubit_dac_to_freq"

    frequency (Hz)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2))
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    branch (enum: 'positive' 'negative')
    '''
    if V_per_phi0 is None and dac_flux_coefficient is None:
        raise ValueError('Please specify "V_per_phi0".')

    E_c = 0
    # asymm_term = (asymmetry**2 + (1-asymmetry**2))
    # dac_term = np.arccos(((frequency+E_c)/((f_max+E_c) * asymm_term))**2)

    dac_term = np.arccos(np.sqrt(
        (((frequency + E_c) / (f_max + E_c)) ** 4 - asymmetry ** 2) /
        (1 - asymmetry ** 2)))

    if dac_flux_coefficient is not None:
        logging.warning('"dac_flux_coefficient" deprecated. Please use the '
                        'physically meaningful "V_per_phi0" instead.')
        V_per_phi0 = np.pi / dac_flux_coefficient

    if branch == 'positive':
        dac_voltage = dac_term * V_per_phi0 / np.pi + dac_sweet_spot
    elif branch == 'negative':
        dac_voltage = -dac_term * V_per_phi0 / np.pi + dac_sweet_spot
    else:
        raise ValueError('branch {} not recognized'.format(branch))

    return dac_voltage


def Qubit_dac_sensitivity(dac_voltage, f_max: float, E_c: float,
                          dac_sweet_spot: float, V_per_phi0: float,
                          asymmetry: float = 0):
    '''
    Derivative of the qubit detuning vs dac at dac_voltage.
    The returned quantity is "dfreq/dPhi (dac_voltage)"
    '''
    cos_term = np.cos(np.pi / V_per_phi0 * (dac_voltage - dac_sweet_spot))
    sin_term = np.sin(np.pi / V_per_phi0 * (dac_voltage - dac_sweet_spot))
    return ((f_max + E_c) * (1 - asymmetry ** 2) * np.pi / (2 * V_per_phi0) *
            cos_term * sin_term * (asymmetry ** 2 + (1 - asymmetry ** 2) *
                                   cos_term ** 2) ** (-0.75))


def QubitFreqDac(dac_voltage, f_max, E_c,
                 dac_sweet_spot, dac_flux_coefficient, asymmetry=0):
    logging.warning('deprecated, replace QubitFreqDac with Qubit_dac_to_freq')
    return Qubit_dac_to_freq(dac_voltage, f_max, E_c,
                             dac_sweet_spot, dac_flux_coefficient, asymmetry)


def QubitFreqFlux(flux, f_max, E_c,
                  flux_zero, dac_offset=0):
    'The cosine Arc model for calibrated flux.'
    calculated_frequency = (f_max + E_c) * np.sqrt(np.abs(
        np.cos(np.pi * (flux - dac_offset) / flux_zero))) - E_c
    return calculated_frequency


def CosFunc(t, amplitude, frequency, phase, offset):
    '''
    parameters:
        t, time in s
        amplitude a.u.
        frequency in Hz (f, not omega!)
        phase in rad
        offset a.u.
    '''
    return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset


def ExpDecayFunc(t, tau, amplitude, offset, n):
    return amplitude * np.exp(-(t / tau) ** n) + offset

def ExpDecayPmod(t, T2echo, delta, n0, chi, kappa, phase, amplitude, offset):
    """
    specific form of exponential decay used for residual resonator photon readout
    """

    return amplitude * (1 - np.imag(np.exp(-t*(1/T2echo + 2j*np.pi*delta)+1j*(
        phase-2*n0*chi*(1-np.exp(-t*(kappa + 2j*chi)))/(kappa + 2j*chi)))))+offset

def CombinedOszExpDecayFunc(t, tau, tau_gauss, phase, n0, chi, delta, amplitude,
                         oscillation_offset, offset):
    """
    Combination of gaussian and exponential decay
    """
    return amplitude * np.exp(-(t / tau)-(t / tau_gauss))*(np.cos(
        2 * np.pi * (n0*2*chi/(2*np.pi)+delta) * t + phase) +
        oscillation_offset) + offset

def idle_error_rate_exp_decay(N, N1, N2, A, offset):
    """
    exponential decay consisting of two components
    """
    return A * np.exp(-N / N1 - (N / N2) ** 2) + offset


def gain_corr_ExpDecayFunc(t, tau, amp, gc):
    """
    Specific form of an exponential decay used for flux corrections.
    Includes a "gain correction" parameter that is ignored when correcting
    the distortions.
    """

    y = gc * (1 + amp * np.exp(-t / tau))
    return y


def gain_corr_double_ExpDecayFunc(t, tau_A, tau_B, amp_A, amp_B, gc):
    """
    Specific form of an exponential decay used for flux corrections.
    Includes a "gain correction" parameter that is ignored when correcting
    the distortions.
    """
    y = gc * (1 + amp_A * np.exp(-t / tau_A) + amp_B * np.exp(-t / tau_B))
    return y


def ExpDampOscFunc(t, tau, n, frequency, phase, amplitude,
                   oscillation_offset, exponential_offset):
    return amplitude * np.exp(-(t / tau) ** n) * (np.cos(
        2 * np.pi * frequency * t + phase) + oscillation_offset) + exponential_offset


def GaussExpDampOscFunc(t, tau, tau_2, frequency, phase, amplitude,
                        oscillation_offset, exponential_offset):
    return amplitude * np.exp(-(t / tau_2) ** 2 - (t / tau)) * (np.cos(
        2 * np.pi * frequency * t + phase) + oscillation_offset) + exponential_offset


def ExpDampDblOscFunc(t, tau, n, freq_1, freq_2, phase_1, phase_2,
                      amp_1, amp_2,
                      osc_offset_1, osc_offset_2, exponential_offset):
    '''
    Exponential decay with double cosine modulation
    '''
    exp_decay = np.exp(-(t / tau) ** n)
    cos_1 = (np.cos(
        2 * np.pi * freq_1 * t + phase_1) + osc_offset_1)
    cos_2 = (np.cos(
        2 * np.pi * freq_2 * t + phase_2) + osc_offset_2)

    return amp_1 * exp_decay * cos_1 + amp_2 * exp_decay * cos_2 + exponential_offset


def HangerFuncAmplitude(f, f0, Q, Qe, A, theta):
    '''
    This is the function for a hanger  which does not take into account
    a possible slope.
    This function may be preferred over SlopedHangerFunc if the area around
    the hanger is small.
    In this case it may misjudge the slope
    Theta is the asymmetry parameter

    Note! units are inconsistent
    f is in Hz
    f0 is in GHz
    '''
    return abs(A * (1. - Q / Qe * np.exp(1.j * theta) / (1. + 2.j * Q * (f / 1.e9 - f0) / f0)))


def HangerFuncComplex(f, pars):
    '''
    This is the complex function for a hanger which DOES NOT
    take into account a possible slope
    Input:
        f = frequency
        pars = parameters dictionary
               f0, Q, Qe, A, theta, phi_v, phi_0

    Author: Stefano Poletto
    '''
    f0 = pars['f0']
    Q = pars['Q']
    Qe = pars['Qe']
    A = pars['A']
    theta = pars['theta']
    phi_v = pars['phi_v']
    phi_0 = pars['phi_0']

    S21 = A * (1 - Q / Qe * np.exp(1j * theta) / (1 + 2.j * Q * (f / 1.e9 - f0) / f0)) * \
          np.exp(1j * (phi_v * f + phi_0))

    return S21


def PolyBgHangerFuncAmplitude(f, f0, Q, Qe, A, theta, poly_coeffs):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible polynomial background
    # NOT DEBUGGED
    return np.abs((1. + np.polyval(poly_coeffs, (f / 1.e9 - f0) / f0)) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))


def SlopedHangerFuncAmplitude(f, f0, Q, Qe, A, theta, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return np.abs((1. + slope * (f / 1.e9 - f0) / f0) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))


def SlopedHangerFuncComplex(f, f0, Q, Qe, A, theta, phi_v, phi_0, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return (1. + slope * (f / 1.e9 - f0) / f0) * np.exp(1.j * (phi_v * f + phi_0 - phi_v * f[0])) * \
           HangerFuncComplex(f, f0, Q, Qe, A, theta)


def linear_with_offset(x, a, b):
    '''
    A linear signal with a fixed offset.
    '''
    return a * x + b


def linear_with_background(x, a, b):
    '''
    A linear signal with a fixed background.
    '''
    return np.sqrt((a * x) ** 2 + b ** 2)


def linear_with_background_and_offset(x, a, b, c):
    '''
    A linear signal with a fixed background.
    '''
    return np.sqrt((a * x) ** 2 + b ** 2) + c


def gaussianCDF(x, amplitude, mu, sigma):
    """
    CDF of gaussian is P(X<=x) = .5 erfc((mu-x)/(sqrt(2)sig))
    """
    return 0.5 * amplitude * scipy.special.erfc((mu - x) / (np.sqrt(2)*sigma))


def double_gaussianCDF(x, A_amplitude, A_mu, A_sigma,
                       B_amplitude, B_mu, B_sigma):
    """
    CDF of two gaussians added on top of each other.

    uses "gaussianCDF"
    """
    CDF_A = gaussianCDF(x, amplitude=A_amplitude, mu=A_mu, sigma=A_sigma)
    CDF_B = gaussianCDF(x, amplitude=B_amplitude, mu=B_mu, sigma=B_sigma)
    return CDF_A + CDF_B

def ro_gauss(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
             B_amplitude, A_spurious, B_spurious):
    '''
    Two double-gaussians with sigma and mu/center of the residuals equal to the
    according state.
    '''
    gauss = lmfit.lineshapes.gaussian
    A_gauss = gauss(x=x[0], center=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = gauss(x=x[1], center=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]


def ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
           B_amplitude, A_spurious, B_spurious):
    cdf = gaussianCDF
    A_gauss = cdf(x=x[0], mu=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = cdf(x=x[1], mu=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]


def ro_CDF_discr(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
                 B_amplitude, A_spurious, B_spurious):
    #A_amplitude /= 1-A_spurious
    #B_amplitude /= 1-B_spurious
    return ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
                  B_amplitude, A_spurious=0, B_spurious=0)


def gaussian_2D(x, y, amplitude=1,
                center_x=0, center_y=0,
                sigma_x=1, sigma_y=1):
    '''
    A 2D gaussian function. if you want to use this for fitting you need to
    flatten your data first.
    '''
    gauss = lmfit.lineshapes.gaussian
    val = (gauss(x, amplitude, center_x, sigma_x) *
           gauss(y, amplitude, center_y, sigma_y))
    return val


def DoubleExpDecayFunc(t, tau1, tau2, amp1, amp2, offset, n):
    return (offset +
            amp1 * np.exp(-(t / tau1) ** n) +
            amp2 * np.exp(-(t / tau2) ** n))


def TripleExpDecayFunc(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, n):
    return (offset +
            amp1 * np.exp(-(t / tau1) ** n) +
            amp2 * np.exp(-(t / tau2) ** n) +
            amp3 * np.exp(-(t / tau3) ** n))


def avoided_crossing_mediated_coupling(flux, f_bus, f_center1, f_center2,
                                       c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_b,  g,  g ]
        [g,   f_1, 0 ]
        [g,   0,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
    f_b = constant

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return

    """
    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result


def avoided_crossing_direct_coupling(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result


def ErfWindow(t, t_start, t_end, t_rise, amplitude, offset):
    '''
    parameters:
        t, time in s
        t_start, start of window in s
        t_end, end of window in s
        amplitude a.u.
        offset a.u.
        t_rise in s (rise time)
    '''
    return offset + amplitude/2*(scipy.special.erf((t - t_start)/(t_rise/2.6))
                                 - scipy.special.erf((t - t_end)/(t_rise/2.6)))

def hanger_with_pf(f, phi, J, kappa_pf, omega_ro, omega_pf, gamma_ro, A):

    S12 = A * np.abs(  np.cos(phi) - np.exp(1j*phi)*kappa_pf*
                       (gamma_ro-2j*(f-omega_ro))/
                       (4*J*J+(kappa_pf-2j*(f-omega_pf))*
                        (gamma_ro-2j*(f-omega_ro)))  )

    return S12


def simultan_hanger_with_pf(f, phi, J, kappa_pf, omega_ro_0, omega_ro_1,
                            omega_pf, gamma_ro, A):
    f0 = f[:int(len(f)/2)]
    f1 = f[int(len(f)/2):]
    S12_0 = A * np.abs(  np.cos(phi) - np.exp(1j*phi)*kappa_pf*
                       (gamma_ro-2j*(f0-omega_ro_0))/
                       (4*J*J+(kappa_pf-2j*(f0-omega_pf))*
                        (gamma_ro-2j*(f0-omega_ro_0)))  )

    S12_1 = A * np.abs(  np.cos(phi) - np.exp(1j*phi)*kappa_pf*
                       (gamma_ro-2j*(f1-omega_ro_1))/
                       (4*J*J+(kappa_pf-2j*(f1-omega_pf))*
                        (gamma_ro-2j*(f1-omega_ro_1)))  )

    return np.hstack((S12_0, S12_1))

def fit_hanger_with_pf(model, data, simultan=False):
    """
    model: a lmfit model of the pf_ro_S12 function
    data: of the shape [[freq,|S12|],[freq,|S12|],...] where freq should be in GHz

    return: a lmfit model.fit object
    """
    def hanger_with_pf_guess(model, data):


        A_guess = 1
        midmax = (data[0, 1]+data[-1, 1])/2

        J_guess = 0.006
        omega_pf_guess = min(data, key = lambda t: t[1])[0]

        pf_filter = [False]*len(data)
        for x in range(0,len(data)):
            if data[x,1] <= midmax:
                pf_filter[x] = True
        kappa_pf_guess = min(2*abs(list(itertools.compress(
                     data[:, 0],pf_filter))[0]- omega_pf_guess),
                     2*abs(list(itertools.compress(
                         data[:, 0],pf_filter))[-1]- omega_pf_guess), 0.05)

        omega_ro_filter = [True]*len(data)
        for x in range(0,len(data)):
            if abs(data[x, 0]-omega_pf_guess) <= kappa_pf_guess/2:
                omega_ro_filter[x] = False
        omega_ro_guess = min(list(itertools.compress(
            data,omega_ro_filter)), key = lambda t: t[1])[0]

        for x in range(0,len(data)):
            if min(omega_pf_guess, omega_ro_guess) <= data[x,0]:
                if data[x,0] <= max(omega_pf_guess, omega_ro_guess):
                    if midmax <= data[x,1]: midmax = data[x,0]


        model.set_param_hint('A', value=A_guess, min = 0, max = 1)
        model.set_param_hint('phi', value=4 , min = 0, max = 2*np.pi)
        model.set_param_hint('J', value=J_guess, min=0, max=0.015)
        model.set_param_hint('kappa_pf', value=kappa_pf_guess, min=0, max=0.05)
        model.set_param_hint('omega_pf', value=omega_pf_guess, min=0)
        model.set_param_hint('gamma_ro', value=0.0001, vary = False)

        model.set_param_hint('omega_ro', value=omega_ro_guess)

        params = model.make_params()

        return params

    if simultan:
        data_loc_0 = np.copy(data[0][:-2])
        data_loc_1 = np.copy(data[1][:-2])

        fit0 = fit_hanger_with_pf(HangerWithPfModel, np.transpose(
            [data_loc_0[:,0], data_loc_0[:,1]]), simultan=False)
        fit1 = fit_hanger_with_pf(HangerWithPfModel, np.transpose(
            [data_loc_1[:,0], data_loc_1[:,1]]), simultan=False)

        if not len(data_loc_1) == len(data_loc_0):
            if len(data_loc_1) < len(data_loc_0):
                data_loc_0 = data_loc_0[:len(data_loc_1)]
            else:
                data_loc_1 = data_loc_1[:len(data_loc_0)]

        data_loc_0[:,0] *= 1e-9
        data_loc_1[:,0] *= 1e-9
        data_loc = np.vstack((data_loc_0, data_loc_1))

        phi_guess = (fit0.params['phi'].value+fit1.params['phi'].value)/2
        J_guess = (fit0.params['J'].value+fit1.params['J'].value)/2*1e-9
        kap_guess = (fit0.params['kappa_pf'].value+
                     fit1.params['kappa_pf'].value)/2*1e-9
        omega_pf_guess = (fit0.params['omega_pf'].value+
                          fit1.params['omega_pf'].value)/2*1e-9

        model.set_param_hint('A', value=1, min = 0, max = 1)
        model.set_param_hint('A', value=1, min = 0, max = 1)
        model.set_param_hint('phi', value=phi_guess , min = 0, max = 2*np.pi)
        model.set_param_hint('J', value=J_guess, min=0, max=0.015)
        model.set_param_hint('kappa_pf', value=kap_guess, min=0, max=0.05)
        model.set_param_hint('omega_pf', value=omega_pf_guess, min=0)
        model.set_param_hint('gamma_ro', value=0.0001, vary = False)

        model.set_param_hint('omega_ro_0', value=
                                 fit0.params['omega_ro'].value*1e-9, min=0)
        model.set_param_hint('omega_ro_1', value=
                                 fit1.params['omega_ro'].value*1e-9, min=0)

        guess = model.make_params()

        fit_out = model.fit( data_loc[:,1]/max(data_loc[:,1]),
                             guess, f=data_loc[:,0])
        tol = 0.4

    else:
        data_loc = np.copy(data)
        data_loc[:,0] *= 1e-9

        guess = hanger_with_pf_guess(model,np.transpose(
                [data_loc[:,0],data_loc[:,1]/max(data_loc[:,1])]))

        fit_out = model.fit( data_loc[:,1]/
                           max(data_loc[:,1]),guess,f=data_loc[:,0],)
        tol = 0.1
    
    if fit_out.chisqr > tol:
        fit_lst = []
        for shift_pf in np.linspace(-0.01, 0.01, 5):
            for shift_ro in np.linspace(-0.01, 0.01, 5):
                for phase in np.linspace(0, 2*np.pi, 6):
                    for kappa_shift in np.linspace(-0.015,0.015,4):
                        for J_shift in  np.linspace(-0.002,0.004,4):
                            if simultan:
                                fit_lst.append(model.fit( data_loc[:,1]/
                                max(data_loc[:,1]),guess,
                                f=data_loc[:,0],
                                phi=phase,omega_pf=float(guess['omega_pf'])+shift_pf,
                                omega_ro_0=float(guess['omega_ro_0'])+shift_ro,
                                omega_ro_1=float(guess['omega_ro_1'])+shift_ro,
                                kappa_pf=float(guess['kappa_pf'])+kappa_shift,
                                J=float(guess['J'])+J_shift))
                            else:
                                fit_lst.append(model.fit( data_loc[:,1]/max(
                                data_loc[:,1]),guess,f=data_loc[:,0],phi=phase,
                                omega_pf=float(guess['omega_pf'])+shift_pf,
                                omega_ro=float(guess['omega_ro'])+shift_ro,
                                kappa_pf=float(guess['kappa_pf'])+kappa_shift,
                                J=float(guess['J'])+J_shift))

            if fit_lst[-1].chisqr <= tol: break

        chisqr_lst = [fit.chisqr for fit in fit_lst]

        fit_out = fit_lst[np.argmin(chisqr_lst)]


    fit_out.params['J'].max = 15e6
    fit_out.params['kappa_pf'].max = 50e6
    fit_out.params['omega_pf'].value *= 1e9
    fit_out.params['kappa_pf'].value *= 1e9
    fit_out.params['J'].value *= 1e9
    fit_out.params['gamma_ro'].value *= 1e9
    fit_out.params['A'].value *= max(data_loc[:,1])
    if simultan:
        fit_out.params['omega_ro_0'].value *= 1e9
        fit_out.params['omega_ro_1'].value *= 1e9
    else:
        fit_out.params['omega_ro'].value *= 1e9

    if fit_out.chisqr > tol:
            logging.warning('The fit did not converge properly: chi^2 = '
                            ''+str(fit_out.chisqr))

    #Unpack simultan fits
    if simultan:
        fit0.params['omega_pf'].value = fit_out.params['omega_pf'].value
        fit0.params['kappa_pf'].value = fit_out.params['kappa_pf'].value
        fit0.params['J'].max = fit_out.params['J'].max
        fit0.params['J'].value = fit_out.params['J'].value
        fit0.params['gamma_ro'].value = fit_out.params['gamma_ro'].value
        fit0.params['omega_ro'].value = fit_out.params['omega_ro_0'].value

        fit1.params['omega_pf'].value = fit_out.params['omega_pf'].value
        fit1.params['kappa_pf'].value = fit_out.params['kappa_pf'].value
        fit1.params['J'].max = fit_out.params['J'].max
        fit1.params['J'].value = fit_out.params['J'].value
        fit1.params['gamma_ro'].value = fit_out.params['gamma_ro'].value
        fit1.params['omega_ro'].value = fit_out.params['omega_ro_1'].value

        fit_out = [fit0, fit1]

    return fit_out

######################
# Residual functions #
######################
def residual_complex_fcn(pars, cmp_fcn, x, y):
    '''
    Residual of a complex function with complex results 'y' and
    real input values 'x'
    For resonators 'x' is the the frequency, 'y' the complex transmission
    Input:
        pars = parameters dictionary (check the corresponding function 'cmp_fcn' for the parameters to pass)
        cmp_fcn = complex function
        x = input real values to 'cmp_fcn'
        y = output complex values from 'cmp_fcn'

    Author = Stefano Poletto
    '''
    cmp_values = cmp_fcn(x, pars)

    res = cmp_values - y
    res = np.append(res.real, res.imag)

    return res


####################
# Guess functions  #
####################
def exp_dec_guess(model, data, t):
    '''
    Assumes exponential decay in estimating the parameters
    '''
    offs_guess = data[np.argmax(t)]
    amp_guess = data[np.argmin(t)] - offs_guess
    # guess tau by looking for value closest to 1/e
    tau_guess = t[np.argmin(abs((amp_guess * (1 / np.e) + offs_guess) - data))]

    model.set_param_hint('amplitude', value=amp_guess)
    model.set_param_hint('tau', value=tau_guess)
    model.set_param_hint('n', value=1, vary=False)
    model.set_param_hint('offset', value=offs_guess)

    params = model.make_params()
    return params


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def arc_guess(freq, dac, dd=0.1):
    '''
    Expects the dac values to be sorted!
    :param freq:
    :param dac:
    :param dd:
    :return:
    '''
    p = round(max(dd * len(dac), 1))
    f_small = np.average(np.sort(freq)[:p]) + np.std(np.sort(freq)[:p])
    f_big = np.average(np.sort(freq)[-p:]) - np.std(np.sort(freq)[-p:])
    #print(f_small * 1e-9, f_big * 1e-9)

    fmax = np.max(freq)
    fmin = np.min(freq)

    dacs_ss = np.where(freq >= f_big)[0]
    dacs_as = np.where(freq <= f_small)[0]

    dacs_ss_groups = group_consecutives(vals=dacs_ss, step=1)
    dacs_as_groups = group_consecutives(vals=dacs_as, step=1)

    dacs_ss_single = []
    for g in dacs_ss_groups:
        ind = g[np.argmax(freq[g])]
        # ind = int(round(np.average(g)))
        dacs_ss_single.append(ind)

    dac_ss_group_index = np.argmin(np.abs(dac[dacs_ss_single]))
    dac_ss_index = dacs_ss_single[dac_ss_group_index]
    min_left = 0
    min_right = len(dac) - 1
    dacs_as_single = []
    for g in dacs_as_groups:
        if 0 in g:
            ind = 0
        elif len(dac) - 1 in g:
            ind = len(dac) - 1
        else:
            ind = int(round(np.average(g)))

        if ind < dac_ss_index:
            min_left = max(ind, min_left)
        elif ind > dac_ss_index:
            min_right = min(ind, min_right)
        dacs_as_single.append(ind)
    # print('maxs', dacs_ss_single)
    # print('mins', dacs_as_single)
    arc_len = (dac[min_right] - dac[min_left])

    # print('%d to %d = %.5f' % (min_left, min_right, arc_len))
    if min_left == 0 or min_right == len(dac) - 1:
        arc_len *= 2
    elif len(dacs_ss_groups) > 1:
        arc_len = np.average(dac[dacs_ss_single[1:]] - dac[dacs_ss_single[:-1]])

    return fmax, fmin, dac[dac_ss_index], arc_len


def Resonator_dac_arch_guess(model, freq, dac_voltage, f_max_qubit: float = None, E_c: float = None):
    fmax, fmin, dac_ss, period = arc_guess(freq=freq, dac=dac_voltage)
    coup_guess = 15e6

    # todo make better f_res guess
    f_res = np.mean(freq)  # - (coup_guess ** 2 / (f_max_qubit - fmax))
    f_max_qubit_vary = f_max_qubit is None
    f_max_qubit = f_max_qubit or f_res - 500e6

    model.set_param_hint('f_0_res', value=f_res, min=f_res / 2, max=2 * f_res)
    model.set_param_hint('f_max_qubit', value=f_max_qubit, min=3e9, max=8.5e9, vary=f_max_qubit_vary)
    model.set_param_hint('dac_sweet_spot', value=dac_ss, min=(dac_ss - 0.005) / 2, max=2 * (dac_ss + 0.005))
    model.set_param_hint('V_per_phi0', value=period, min=(period - 0.005) / 3, max=5 * (period + 0.005))
    model.set_param_hint('asymmetry', value=0, max=1, min=-1)
    model.set_param_hint('coupling', value=coup_guess, min=1e6, max=80e6)
    E_c = E_c or 260e6
    model.set_param_hint('E_c', value=E_c, min=50e6, max=400e6)

    params = model.make_params()
    return params


def Qubit_dac_arch_guess(model, data, dac_voltage):
    f_max, dac_ss = np.max(data), dac_voltage[np.argmax(data)]
    f_min, dac_lss = np.min(data), dac_voltage[np.argmin(data)]
    V_per_phi0 = abs(2*(dac_ss-dac_lss))
    d = (f_min)**2/(f_max)**2
    model.set_param_hint('f_max', value=f_max, min=0)
    model.set_param_hint('dac_sweet_spot', value=dac_ss, min=-3, max=3)
    model.set_param_hint('V_per_phi0', value=V_per_phi0, min=0)
    model.set_param_hint('asymmetry', value=d, min=0, max=1)

    params = model.make_params()
    return params


def idle_err_rate_guess(model, data, N):
    '''
    Assumes exponential decay in estimating the parameters
    '''
    amp_guess = 0.5
    offset = np.mean(data)
    N1 = np.mean(N)
    N2 = np.mean(N)
    params = model.make_params(A=amp_guess,
                               N1=N1,
                               N2=N2,
                               offset=offset)
    return params


def fft_freq_phase_guess(data, t):
    '''
    Guess for a cosine fit using FFT, only works for evenly spaced points
    '''
    # Freq guess ! only valid with uniform sampling
    # Only first half of array is used, because the second half contains the
    # negative frequecy components, and we want a positive frequency.
    w = np.fft.fft(data)[:len(data) // 2]
    f = np.fft.fftfreq(len(data), t[1] - t[0])[:len(w)]
    w[0] = 0  # Removes DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_w = np.abs(w)
    freq_guess = abs(f[abs_w == max(abs_w)][0])
    ph_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess)[0]
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution

    return freq_guess, ph_guess


def Cos_guess(model, data, t, **kwargs):
    """
    Tip: to use this assign this guess function as a method to a model use:
    model.guess = Cos_guess.__get__(
        model, model.__class__)
    """

    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    offs_guess = np.mean(data)

    freq_guess, ph_guess = fft_freq_phase_guess(data, t)

    model.set_param_hint('period', expr='1/frequency')
    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp
    params['frequency'].min = 0

    return params


def exp_damp_osc_guess(model, data, t, n_guess=1):
    """
    Makes a guess for an exponentially damped oscillation.
    Uses the fft_freq_phase guess to guess the oscillation parameters.
    The guess for the exponential is simpler as it sets the exponent (n) at 1
    and the tau at 2/3 of the total range
    """
    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    freq_guess, ph_guess = fft_freq_phase_guess(data, t)
    osc_offs_guess = 0

    tau_guess = 2 / 3 * max(t)
    exp_offs_guess = np.mean(data)

    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               oscillation_offset=osc_offs_guess,
                               exponential_offset=exp_offs_guess,
                               n=n_guess,
                               tau=tau_guess)
    return params


def Cos_amp_phase_guess(model, data, f, t):
    '''
    Guess for a cosine fit with fixed frequency f.
    '''
    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    offs_guess = np.mean(data)

    ph_guess = (-2 * np.pi * t[data == max(data)] * f)[0]
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution

    # model.set_param_hint('period', expr='1')
    params = model.make_params(amplitude=amp_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp

    return params


def gauss_2D_guess(model, data, x, y):
    '''
    takes the mean of every row/column and then uses the regular gauss guess
    function to get a guess for the model parameters.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
        * total surface under the gaussians sums up to 1.

    Potential improvements:
        Make the input also accept a 2D grid of data to prevent reshaping.
        Find a way to extract amplitude guess from data itself, note that
        taking the sum of the data (which should correspond to all data under
                                    the curve) does not do the trick.

    Note: possibly not compatible if the model uses prefixes.
    '''
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    sums = np.sum(((data[:-1,:-1]*dx).transpose()*dy))
    amp = np.sqrt(sums)
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)
    x.sort()
    y.sort()
    xm = lmfit.models.GaussianModel()
    ym = lmfit.models.GaussianModel()
    x_guess = xm.guess(data=x_proj_data, x=np.unique(x))
    x_res = xm.fit(data=x_proj_data, x=np.unique(x), params=x_guess)
    y_guess = ym.guess(data=y_proj_data, x=np.unique(y))
    y_res = ym.fit(data=y_proj_data, x=np.unique(y), params=y_guess)

    x_guess = x_res.params
    y_guess = y_res.params
    model.set_param_hint('amplitude', value=amp, min=0.9*amp, max=1.1*amp,
                         vary=True)
    model.set_param_hint('sigma_x', value=x_guess['sigma'].value, min=0,
                         vary=True)
    model.set_param_hint('sigma_y', value=y_guess['sigma'].value, min=0,
                         vary=True)
    params = model.make_params(center_x=x_guess['center'].value,
                               center_y=y_guess['center'].value,)
    return params


def double_gauss_2D_guess(model, data, x, y):
    '''
    takes the mean of every row/column and then uses the guess
    function of the double gauss.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
    Note: possibly not compatible if the model uses prefixes.
    Note 2: see also gauss_2D_guess() for some notes on how to improve this
            function.
    '''
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)

    # The syntax here is slighly different than when calling a regular guess
    # function because I do not overwrite the class attribute properly.
    x_guess = double_gauss_guess(model=None, data=x_proj_data, x=np.unique(x))
    y_guess = double_gauss_guess(model=None, data=y_proj_data, x=np.unique(y))

    if model is not None:
        pars = model.make_params(A_sigma_x=x_guess['A_sigma'],
                                 A_sigma_y=y_guess['A_sigma'],
                                 A_center_x=x_guess['A_center'],
                                 A_center_y=y_guess['A_center'],
                                 A_amplitude=1,
                                 B_sigma_x=x_guess['B_sigma'],
                                 B_sigma_y=y_guess['B_sigma'],
                                 B_center_y=y_guess['B_center'],
                                 B_center_x=x_guess['B_center'],
                                 B_amplitude=1)
        return pars
    else:
        return x_guess, y_guess


def double_gauss_guess(model, data, x=None, **kwargs):
    '''
    Finds a guess for the intial parametes of the double gauss model.
    Guess is based on taking the cumulative sum of the data and
    finding the points corresponding to 25% and 75%
    it finds sigma by using the property that ~33% of the data is contained
    in the range mu-sigma to mu+sigma.

    Tip: to use this assign this guess function as a method to a model use:
        model.guess = double_gauss_guess.__get__(
            model, model.__class__)
    '''
    if x is None:
        x = np.arange(len(data))
    cdf = np.cumsum(data)
    norm_cdf = cdf / cdf[-1]
    par_dict = {'A_center': x[(np.abs(norm_cdf - 0.25)).argmin()],
                'B_center': x[(np.abs(norm_cdf - 0.75)).argmin()],
                'A_sigma': (x[(np.abs(norm_cdf - 0.25 - .33 / 2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.25 + .33 / 2)).argmin()]),
                'B_sigma': (x[(np.abs(norm_cdf - 0.75 - .33 / 2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.75 + .33 / 2)).argmin()])}

    amp = max(data) * (par_dict['A_sigma'] + par_dict['B_sigma']) / 2.
    if model is not None:
        # Specify explicitly because not all pars are set to those from the par
        # dict
        pars = model.make_params(A_center=par_dict['A_center'],
                                 B_center=par_dict['B_center'],
                                 A_sigma=par_dict['A_sigma'],
                                 B_sigma=par_dict['B_sigma'],
                                 A_amplitude=amp, B_amplitude=amp)
        return pars
    # The else clause is added explicitly to reuse this function for the
    # 2D double gauss model
    else:
        return par_dict

def ro_double_gauss_guess(model, data, x, fixed_p01 = False, fixed_p10 = False):
    # An initial guess is done on the binned data with single gaussians
    # to constrain the fit params and avoid fitting noise if
    # e.g., mmt. ind. rel. is very low
    gmod0 = lmfit.models.GaussianModel()
    guess0 = gmod0.guess(data=data[0], x=x[0])
    gmod1 = lmfit.models.GaussianModel()
    guess1 = gmod1.guess(data=data[1], x=x[1])


    model.set_param_hint(
        'A_center', vary=True, value=guess0['center'].value,
        min=guess0['center'] - 2 * guess0['sigma'],
        max=guess0['center'] + 2 * guess0['sigma'])
    model.set_param_hint(
        'B_center', vary=True, value=guess1['center'].value,
        min=guess1['center'] - 2 * guess1['sigma'],
        max=guess1['center'] + 2 * guess1['sigma'])
    model.set_param_hint('A_sigma', value=guess0['sigma'].value, vary=True)
    model.set_param_hint('B_sigma', value=guess1['sigma'].value, vary=True)

    # Amplitudes
    intarea0 = sum_int(x=x[0], y=data[0])[-1]
    intarea1 = sum_int(x=x[1], y=data[1])[-1]
    model.set_param_hint('A_amplitude', value=intarea0, vary=False)
    model.set_param_hint('B_amplitude', value=intarea1, vary=False)
    model.set_param_hint('SNR', expr='abs(A_center-B_center)*2/(A_sigma+B_sigma)',
                         vary=False)

    # Spurious excitement
    f = np.sqrt(2*np.pi)
    amp0 = 0.99 * np.max(data[0]) * guess0['sigma'] * f
    amp1 = 0.99 * np.max(data[1]) * guess1['sigma'] * f
    spurious0 = max(1-(amp0/intarea0), 1e-3)
    spurious1 = max(1-(amp1/intarea1), 1e-3)
    p01 = fixed_p01 if fixed_p01 is not False else spurious0
    p10 = fixed_p10 if fixed_p10 is not False else spurious1
    model.set_param_hint('A_spurious', value=p01, min=0, max=1,
                         vary=fixed_p01 is False)
    model.set_param_hint('B_spurious', value=p10, min=0, max=1,
                         vary=fixed_p10 is False)

    return model.make_params()


def sum_int(x,y):
    return np.cumsum(y[:-1]*(x[1:]-x[:-1]))


def Gaussian(freq,sigma,mu,ampl,offset):
    '''
    Gaussian function
    '''
    return ampl/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((freq - mu)/sigma)**2) + offset


def half_feed_line_S12_J_func(omega, J, kappaPF, gammaPF, gammaRR, omegaPF, omegaRR, phi, A , B, alpha):
    return abs( A+np.exp(-1j*phi)*2*B*((-1+np.exp(1j*alpha))*(4*J**2+(gammaPF-2*1j*(omegaPF-omega))*(gammaRR-2j*omegaRR+2j*omega)))/(16*J**2+(4*gammaPF+(3+np.exp(1j*alpha))*kappaPF-8j*(omegaPF-omega))*(gammaRR-2j*omegaRR+2j*omega)) )


def half_feed_line_S12_J_guess(model,data):
    '''
    data should have the frequencies in Hz in the first column and the transmission in the second column
    '''

    background_guess = max(data,key=lambda item:item[1])[1]/2
    amp_guess = max(data,key=lambda item:item[1])[0]-min(data,key=lambda item:item[1])[0]

    omegaPF_guess = min(data,key=lambda item:item[1])[0]

    #Remove the PF dip to find the second smallest feature at the RR frequency
    kappaPF_guess_data = np.extract(  (data[:,1]<=(min(data,key=lambda item:item[1])[1]+ background_guess/2)),data[:,0])
    kappaPF_guess = min(abs( kappaPF_guess_data[0]-omegaPF_guess),abs( kappaPF_guess_data[-1]-omegaPF_guess),5e10)

    omegaRR_guess =  min( np.transpose([np.extract(abs(data[:,0]- omegaPF_guess) >= kappaPF_guess/2,data[:,0]),np.extract(abs(data[:,0]- omegaPF_guess) >= kappaPF_guess/2,data[:,1])]),key=lambda item:item[1])[0]
    J_guess = kappaPF_guess/4

    model.set_param_hint('J',value=J_guess,min=0,max=2e7)
    model.set_param_hint('kappaPF',value=kappaPF_guess,min=1e6,max=1e8)
    model.set_param_hint('gammaPF',value=0.001,min=0,max=1e6)
    model.set_param_hint('gammaRR',value=0.001,min=0,max=1e6)
    model.set_param_hint('omegaPF',value=omegaPF_guess,min=omegaPF_guess-2e7,max=omegaPF_guess+2e7)
    model.set_param_hint('omegaRR',value=omegaRR_guess,min=omegaRR_guess-2e7,max=omegaRR_guess+2e7)
    model.set_param_hint('phi',value=(data[0,1]-data[-1,1])/background_guess)
    model.set_param_hint('A',value=background_guess,min=(background_guess)-5,max=(2000*background_guess)+5)
    model.set_param_hint('B',value=background_guess,min=(background_guess)-5,max=(2000*background_guess)+5)
    model.set_param_hint('alpha',value=3,min=0,max=10)

    params=model.make_params()
    return params


#################################
#     User defined Models       #
#################################
# NOTE: it is actually better to instantiate the model within your analysis
# file, this prevents the model params having a memory.
# A valid reason to define it here would be exp_dec_guess if you want to add a guess function
CosModel = lmfit.Model(CosFunc)
CosModel.guess = Cos_guess

half_Feed_lineS12_J_Model = lmfit.Model(half_feed_line_S12_J_func)
half_Feed_lineS12_J_Model.guess = half_feed_line_S12_J_guess
ExpDecayModel = lmfit.Model(ExpDecayFunc)
TripleExpDecayModel = lmfit.Model(TripleExpDecayFunc)
ExpDecayModel.guess = exp_dec_guess  # todo: fix
ExpDampOscModel = lmfit.Model(ExpDampOscFunc)
GaussExpDampOscModel = lmfit.Model(GaussExpDampOscFunc)
ExpDampDblOscModel = lmfit.Model(ExpDampDblOscFunc)
DoubleExpDampOscModel = lmfit.Model(DoubleExpDampOscFunc)
HangerAmplitudeModel = lmfit.Model(HangerFuncAmplitude)
SlopedHangerAmplitudeModel = lmfit.Model(SlopedHangerFuncAmplitude)
PolyBgHangerAmplitudeModel = lmfit.Model(PolyBgHangerFuncAmplitude)
HangerComplexModel = lmfit.Model(HangerFuncComplex)
SlopedHangerComplexModel = lmfit.Model(SlopedHangerFuncComplex)
QubitFreqDacModel = lmfit.Model(Qubit_dac_to_freq)
QubitFreqFluxModel = lmfit.Model(QubitFreqFlux)
TwinLorentzModel = lmfit.Model(TwinLorentzFunc)
LorentzianModel = lmfit.Model(Lorentzian)
RBModel = lmfit.Model(RandomizedBenchmarkingDecay)
LinOModel = lmfit.Model(linear_with_offset)
LinBGModel = lmfit.Model(linear_with_background)
LinBGOModel = lmfit.Model(linear_with_background_and_offset)
ErfWindowModel = lmfit.Model(ErfWindow)
GaussianModel = lmfit.models.GaussianModel
ExponentialModel = lmfit.models.ExponentialModel
HangerWithPfModel = lmfit.Model(hanger_with_pf)
SimHangerWithPfModel = lmfit.Model(simultan_hanger_with_pf,
                                   independent_vars=['f'])

# 2D models
Gaus2D_model = lmfit.Model(gaussian_2D, independent_vars=['x', 'y'])
Gaus2D_model.guess = gauss_2D_guess  # Note: not proper way to add guess func
DoubleGauss2D_model = (lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='A_') +
                       lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='B_'))
DoubleGauss2D_model.guess = double_gauss_2D_guess
###################################
# Models based on lmfit functions #
###################################

LorentzModel = lmfit.Model(lmfit.models.lorentzian)
Lorentz_w_background_Model = lmfit.models.LorentzianModel() + \
                             lmfit.models.LinearModel()
PolyBgHangerAmplitudeModel = (HangerAmplitudeModel *
                              lmfit.models.PolynomialModel(degree=7))

DoubleGaussModel = (lmfit.models.GaussianModel(prefix='A_') +
                    lmfit.models.GaussianModel(prefix='B_'))
DoubleGaussModel.guess = double_gauss_guess  # defines a guess function


def plot_fitres2D_heatmap(fit_res, x, y, axs=None, cmap='viridis'):
    '''
    Convenience function for plotting results of flattened 2D fits.

    It could be argued this does not belong in fitting models (it is not a
    model) but I put it here as it is closely related to all the stuff we do
    with lmfit. If anyone has a better location in mind, let me know (MAR).
    '''
    # fixing the data rotation with [::-1]
    nr_cols = len(np.unique(x))
    data_2D = fit_res.data.reshape(-1, nr_cols, order='C')[::-1]
    fit_2D = fit_res.best_fit.reshape(-1, nr_cols, order='C')[::-1]
    guess_2D = fit_res.init_fit.reshape(-1, nr_cols, order='C')[::-1]
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs[0].imshow(data_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[1].imshow(fit_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[2].imshow(guess_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[0].set_title('data')
    axs[1].set_title('fit-result')
    axs[2].set_title('initial guess')
    return axs

# Before defining a new model, take a look at the built in models in lmfit.

# From http://lmfit.github.io/lmfit-py/builtin_models.html

# Built-in Fitting Models in the models module
# Peak-like models
# GaussianModel
# LorentzianModel
# VoigtModel
# PseudoVoigtModel
# Pearson7Model
# StudentsTModel
# BreitWignerModel
# LognormalModel
# DampedOcsillatorModel
# ExponentialGaussianModel
# SkewedGaussianModel
# DonaichModel
# Linear and Polynomial Models
# ConstantModel
# LinearModel
# QuadraticModel
# ParabolicModel
# PolynomialModel
# Step-like models
# StepModel
# RectangleModel
# Exponential and Power law models
# ExponentialModel
# PowerLawModel
