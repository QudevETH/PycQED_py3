'''
    File:               waveform.py
    Author:             Wouter Vlothuizen and Adriaan Rol
    Purpose:            generate waveforms for all lookuptable based AWGs
    Based on:           pulse.py, pulse_library.py
    Prerequisites:
    Usage:
    Bugs:
'''

import numpy as np
from pycqed.analysis.fitting_models import Qubit_freq_to_dac


def gauss_pulse(amp, sigma_length, nr_sigma=4, sampling_rate=2e8,
                axis='x', phase=0,
                motzoi=0, delay=0):
    '''
    All inputs are in s and Hz.
    phases are in degree.
    '''
    sigma = sigma_length  # old legacy naming, to be replaced
    length = sigma*nr_sigma
    mu = length/2.

    t_step = 1/sampling_rate
    t = np.arange(0, nr_sigma*sigma + .1*t_step, t_step)

    gauss_env = amp*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
    deriv_gauss_env = motzoi * -1 * (t-mu)/(sigma**1) * gauss_env
    # substract offsets
    gauss_env -= (gauss_env[0]+gauss_env[-1])/2.
    deriv_gauss_env -= (deriv_gauss_env[0]+deriv_gauss_env[-1])/2.

    delay_samples = delay*sampling_rate

    # generate pulses
    Zeros = np.zeros(int(delay_samples))
    G = np.array(list(Zeros)+list(gauss_env))
    D = np.array(list(Zeros)+list(deriv_gauss_env))

    if axis == 'y':
        phase += 90

    pulse_I = np.cos(2*np.pi*phase/360)*G - np.sin(2*np.pi*phase/360)*D
    pulse_Q = np.sin(2*np.pi*phase/360)*G + np.cos(2*np.pi*phase/360)*D

    return pulse_I, pulse_Q


def block_pulse(amp, length, sampling_rate=2e8, delay=0, phase=0):
    '''
    Generates the envelope of a block pulse.
        length in s
        amp in V
        sampling_rate in Hz
        empty delay in s
        phase in degrees
    '''
    nr_samples = (length+delay)*sampling_rate
    delay_samples = delay*sampling_rate
    pulse_samples = nr_samples - delay_samples
    amp_I = amp*np.cos(phase*2*np.pi/360)
    amp_Q = amp*np.sin(phase*2*np.pi/360)
    block_I = amp_I * np.ones(int(pulse_samples))
    block_Q = amp_Q * np.ones(int(pulse_samples))
    Zeros = np.zeros(int(delay_samples))
    pulse_I = list(Zeros)+list(block_I)
    pulse_Q = list(Zeros)+list(block_Q)
    return pulse_I, pulse_Q

####################
# Pulse modulation #
####################


def mod_pulse(pulse_I, pulse_Q, f_modulation,
              Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            sin(wt)] [I_env]
    [Q_mod]   [-sin(wt+phi)   cos(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, nr_pulse_samples,
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples) + \
        pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_I*-np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                  Q_phase_delay_rad) + \
        pulse_Q*np.cos(2*np.pi*f_mod_samples*pulse_samples + Q_phase_delay_rad)

    return pulse_I_mod, pulse_Q_mod


def simple_mod_pulse(pulse_I, pulse_Q, f_modulation,
                     Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            0] [I_env]
    [Q_mod]   [0        sin(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, int(nr_pulse_samples),
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                 Q_phase_delay_rad)
    return pulse_I_mod, pulse_Q_mod


def martinis_flux_pulse(length, lambda_coeffs, theta_f,
                        f_01_max,
                        g2,
                        E_c,
                        dac_flux_coefficient,
                        f_interaction=None,
                        f_bus=None,
                        asymmetry=0,
                        sampling_rate=1e9,
                        return_unit='V'):
    """
    Returns the pulse specified by Martinis and Geller
    Phys. Rev. A 90 022307 (2014).

    \theta = \theta _0 + \sum_{n=1}^\infty  (\lambda_n*(1-\cos(n*2*pi*t/t_p))/2

    note that the lambda coefficients are rescaled to ensure that the center
    of the pulse has a value corresponding to theta_f.

    length          (float)
    lambda_coeffs   (list of floats)
    theta_f         (float) final angle of the interaction. This determines the
                    Voltage for the centerpoint of the waveform.

    f_01_max        (float) qubit sweet spot frequency (Hz).
    g2              (float) coupling between 11-02 (Hz),
                            approx sqrt(2) g1 (the 10-01 coupling).
    E_c             (float) Charging energy of the transmon (Hz).
        N.B. specify either f_interaction or f_bus
    f_interaction   (float) interaction frequency (Hz).
    f_bus           (float) frequency of the bus (Hz).
    dac_flux_coefficient  (float) conversion factor for AWG voltage to flux (1/V)
    asymmetry       (float) qubit asymmetry

    sampling_rate   (float)
    return_unit     (enum: ['V', 'eps', 'f01', 'theta']) whether to return the pulse
                    expressed in units of theta: the reference frame of the
                    interaction, units of epsilon: detuning to the bus
                    eps=f12-f_bus
    """
    lambda_coeffs = np.array(lambda_coeffs)
    nr_samples = int(np.round((length)*sampling_rate))  # rounds the nr samples
    length = nr_samples/sampling_rate  # gives back the rounded length
    t_step = 1/sampling_rate
    t = np.arange(0, length, t_step)
    if f_interaction is None:
        f_interaction = f_bus + E_c
    theta_0 = np.arctan(2*g2/(f_01_max-f_interaction))
    # you can not have weaker coupling than the initial coupling
    assert(theta_f > theta_0)
    odd_coeff_lambda_sum = np.sum(lambda_coeffs[::2])
    delta_theta = theta_f - theta_0
    # add a square pulse that reaches theta_f, for this, lambda0 is used
    lambda0 = 1-lambda_coeffs[0]  # only use lambda_coeffs[0] for scaling, this
    # enables fixing the square to 0 in optimizations by setting
    # lambda_coeffs[0]=1
    th_scale_factor = delta_theta/(lambda0+odd_coeff_lambda_sum)
    mart_pulse_theta = np.ones(nr_samples)*theta_0
    mart_pulse_theta += th_scale_factor*np.ones(nr_samples)*lambda0

    for i, lambda_coeff in enumerate(lambda_coeffs):
        n = i+1
        mart_pulse_theta += th_scale_factor * \
            lambda_coeff*(1-np.cos(n*2*np.pi*t/length))/2
    # adding square pulse scaling with lambda0 satisfying the condition
    # lamb0=1-lambda1
    if return_unit == 'theta':
        return mart_pulse_theta

    # Convert theta to detuning to the bus frequency
    mart_pulse_eps = (2*g2)/(np.tan(mart_pulse_theta))
    if return_unit == 'eps':
        return mart_pulse_eps

    # pulse parameterized in the f01 frequency
    mart_pulse_f01 = mart_pulse_eps + f_interaction
    if return_unit == 'f01':
        return mart_pulse_f01
    mart_pulse_V = Qubit_freq_to_dac(
        frequency=mart_pulse_f01,
        f_max=f_01_max, E_c=E_c,
        dac_sweet_spot=0,
        dac_flux_coefficient=dac_flux_coefficient,
        asymmetry=asymmetry, branch='positive')
    return mart_pulse_V


def martinis_flux_pulse_v2(length, lambda_coeffs, theta_f,
                           f_01_max,
                           J2,
                           dac_flux_coefficient,
                           E_c=0,
                           f_bus=None,
                           f_interaction=None,
                           asymmetry=0,
                           sampling_rate=1e9,
                           return_unit='V'):
    """
    Returns the pulse specified by Martinis and Geller
    Phys. Rev. A 90 022307 (2014).

    \theta = \theta _0 + \sum_{n=1}^\infty  (\lambda_n*(1-\cos(n*2*pi*t/t_p))/2

    note that the lambda coefficients are rescaled to ensure that the center
    of the pulse has a value corresponding to theta_f.

    length          (float)
    lambda_coeffs   (list of floats) starting from lambda2 since lambda1 is
                    completely determined by theta_f
    theta_f         (float) final angle of the interaction. This determines the
                    Voltage for the centerpoint of the waveform.

    f_01_max        (float) qubit sweet spot frequency (Hz).
    J2              (float) coupling between 11-02 (Hz),
                    approx sqrt(2) J1 (the 10-01 coupling).
    E_c             (float) Charging energy of the transmon (Hz).
    f_bus           (float) frequency of the bus (Hz).
    f_interaction   (float) interaction frequency (Hz).
    dac_flux_coefficient  (float) conversion factor for AWG voltage to
                    flux (1/V)
    asymmetry       (float) qubit asymmetry

    sampling_rate   (float)
    return_unit     (enum: ['V', 'eps', 'f01', 'theta']) whether to return the
                    pulse expressed in units of theta: the reference frame of
                    the interaction, units of epsilon: detuning to the bus
                    eps=f12-f_bus
    """
    # Define number of samples and time points
    nr_samples = int(np.round((length)*sampling_rate))  # rounds the nr samples
    length = nr_samples/sampling_rate  # gives back the rounded length
    t_step = 1/sampling_rate
    t = np.arange(0, length, t_step)

    # Derived parameters
    lambdas = np.array(lambda_coeffs)
    if f_interaction is None:
        f_interaction = f_bus + E_c
    theta_i = np.arctan(2*J2 / (f_01_max - f_interaction))
    if theta_f < theta_i:
        raise ValueError(
            'theta_f < theta_i: final coupling weaker than initial coupling')

    odd_lambda_sum = np.sum(lambdas[1::2])  # lambdas[0] = lambda2
    lambda1 = (theta_f - theta_i) / 2 - odd_lambda_sum

    # Calculate the wave
    theta_wave = np.ones(nr_samples) * theta_i
    theta_wave += lambda1 * (1 - np.cos(2 * np.pi * t / length))
    for i, lambda_coeff in enumerate(lambdas):
        n = i + 2  # lambdas[0] = lambda2
        theta_wave += lambda_coeff * (1 - np.cos(2 * np.pi * n * t / length))

    # Return in the specified units
    if return_unit == 'theta':
        return theta_wave

    # Convert to detuning from f_interaction
    delta_f_wave = 2 * J2 / np.tan(theta_wave)
    if return_unit == 'eps':
        return delta_f_wave

    # Convert to parametrization of f_01
    f_01_wave = delta_f_wave + f_interaction
    if return_unit == 'f01':
        return f_01_wave

    # Convert to voltage
    voltage_wave = Qubit_freq_to_dac(
        frequency=f_01_wave,
        f_max=f_01_max,
        E_c=E_c,
        dac_sweet_spot=0,
        dac_flux_coefficient=dac_flux_coefficient,
        asymmetry=asymmetry,
        branch='positive')
    return voltage_wave


def mod_gauss(amp, sigma_length, f_modulation, axis='x', phase=0,
              nr_sigma=4,
              motzoi=0, sampling_rate=2e8,
              Q_phase_delay=0, delay=0):
    '''
    Simple gauss pulse maker for CBOX. All inputs are in s and Hz.
    '''
    pulse_I, pulse_Q = gauss_pulse(amp, sigma_length, nr_sigma=nr_sigma,
                                   sampling_rate=sampling_rate, axis=axis,
                                   phase=phase,
                                   motzoi=motzoi, delay=delay)
    pulse_I_mod, pulse_Q_mod = mod_pulse(pulse_I, pulse_Q, f_modulation,
                                         sampling_rate=sampling_rate,
                                         Q_phase_delay=Q_phase_delay)
    return pulse_I_mod, pulse_Q_mod


def mixer_predistortion_matrix(alpha, phi):
    predistortion_matrix = np.array(
        [[1,  np.tan(phi*2*np.pi/360)],
         [0, 1/alpha * 1/np.cos(phi*2*np.pi/360)]])
    return predistortion_matrix
