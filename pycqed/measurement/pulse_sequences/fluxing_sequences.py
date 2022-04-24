import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control.block import Block
from pycqed.measurement.waveform_control import sequence
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    sweep_pulse_params, add_preparation_pulses, pulse_list_list_seq
from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import \
    generate_mux_ro_pulse_list

import logging
log = logging.getLogger(__name__)


def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing the input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude']*pulse_pars['amp90_scale']

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] = 90
    pulses['mY180']['phase'] = 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] = 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] = 90

    return pulses


def Ramsey_with_flux_pulse_meas_seq(thetas, qb, X90_separation, verbose=False,
                                    upload=True, return_seq=False,
                                    cal_points=False):
    '''
    Performs a Ramsey with interleaved Flux pulse

    Timings of sequence
           <----- |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase

    timing of the flux pulse relative to the center of the first X90 pulse

    Args:
        thetas: numpy array of phase shifts for the second pi/2 pulse
        qb: qubit object (must have the methods get_operation_dict(),
        get_drive_pars() etc.
        X90_separation: float (separation of the two pi/2 pulses for Ramsey
        verbose: bool
        upload: bool
        return_seq: bool

    Returns:
        if return_seq:
          seq: qcodes sequence
          el_list: list of pulse elements
        else:
            seq_name: string
    '''
    raise NotImplementedError(
        'Ramsey_with_flux_pulse_meas_seq has not been '
        'converted to the latest waveform generation code and can not be used.')

    qb_name = qb.name
    operation_dict = qb.get_operation_dict()
    pulse_pars = qb.get_drive_pars()
    RO_pars = qb.get_RO_pars()
    seq_name = 'Measurement_Ramsey_sequence_with_Flux_pulse'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    flux_pulse = operation_dict["flux "+qb_name]
    # Used for checking dynamic phase compensation
    # if flux_pulse['amplitude'] != 0:
    #     flux_pulse['basis_rotation'] = {qb_name: -80.41028958782647}

    flux_pulse['ref_point'] = 'end'
    X90_2 = deepcopy(pulses['X90'])
    X90_2['pulse_delay'] = X90_separation - flux_pulse['pulse_delay'] \
                            - X90_2['nr_sigma']*X90_2['sigma']
    X90_2['ref_point'] = 'start'

    for i, theta in enumerate(thetas):
        X90_2['phase'] = theta*180/np.pi
        if cal_points and (i == (len(thetas)-4) or i == (len(thetas)-3)):
            el = multi_pulse_elt(i, station, [RO_pars])
        elif cal_points and (i == (len(thetas)-2) or i == (len(thetas)-1)):
            flux_pulse['amplitude'] = 0
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], flux_pulse, X90_2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def fluxpulse_scope_sequence(
        delays, freqs, qb_name, operation_dict, cz_pulse_name,
        ro_pulse_delay=None, cal_points=None, prep_params=None, upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ----------------------------  |RO|
       |        ---      | --------- fluxpulse ---------- |
                         <-  delay  ->

        :param ro_pulse_delay: Can be 'auto' to start out the readout after
            the end of the flux pulse or a delay in seconds to start a fixed
            amount of time after the drive pulse. If not provided or set to
            None, a default fixed delay of 100e-9 is used.
    '''
    if prep_params is None:
        prep_params = {}
    if ro_pulse_delay is None:
        ro_pulse_delay = 100e-9

    seq_name = 'Fluxpulse_scope_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'FPS_Pi'
    ge_pulse['element_name'] = 'FPS_Pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'FPS_Flux'
    flux_pulse['ref_pulse'] = 'FPS_Pi'
    flux_pulse['ref_point'] = 'middle'
    flux_pulse_delays = -np.asarray(delays) - flux_pulse.get(
        'buffer_length_start', 0)

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPS_Ro'
    ro_pulse['ref_pulse'] = 'FPS_Pi'
    ro_pulse['ref_point'] = 'end'
    ro_pulse['pulse_delay'] = ro_pulse_delay
    if ro_pulse_delay == 'auto':
        ro_pulse['ref_point'] = 'middle'
        ro_pulse['pulse_delay'] = \
            flux_pulse['pulse_length'] - np.min(delays) + \
            flux_pulse.get('buffer_length_end', 0) + \
            flux_pulse.get('trans_length', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(
        pulses, {'FPS_Flux.pulse_delay': flux_pulse_delays})

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    seq.repeat_ro(f"RO {qb_name}", operation_dict)

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements()), freqs


def fluxpulse_amplitude_sequence(amplitudes,
                                 freqs,
                                 qb_name,
                                 operation_dict,
                                 cz_pulse_name,
                                 delay=None,
                                 cal_points=None,
                                 prep_params=None,
                                 upload=True):
    '''
    Performs X180 pulse on top of a fluxpulse

    Timings of sequence

       |          ----------           |X180|  ------------------------ |RO|
       |          ---    | --------- fluxpulse ---------- |
    '''
    if prep_params is None:
        prep_params = {}

    seq_name = 'Fluxpulse_amplitude_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'FPA_Pi'
    ge_pulse['element_name'] = 'FPA_Pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'FPA_Flux'
    flux_pulse['ref_pulse'] = 'FPA_Pi'
    flux_pulse['ref_point'] = 'middle'

    if delay is None:
        delay = flux_pulse['pulse_length'] / 2

    flux_pulse['pulse_delay'] = -flux_pulse.get('buffer_length_start',
                                                0) - delay

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'FPA_Ro'
    ro_pulse['ref_pulse'] = 'FPA_Pi'
    ro_pulse['ref_point'] = 'middle'


    ro_pulse['pulse_delay'] = flux_pulse['pulse_length'] - delay + \
                              flux_pulse.get('buffer_length_end', 0) + \
                              flux_pulse.get('trans_length', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]
    swept_pulses = sweep_pulse_params(pulses,
                                      {'FPA_Flux.amplitude': amplitudes})

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]

    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements()), freqs


def T2_freq_sweep_seq(amplitudes,
                      qb_name,
                      operation_dict,
                      cz_pulse_name,
                      phases,
                      flux_lengths=None,
                      n_pulses=None,
                      cal_points=None,
                      upload=True):
    """
    Performs a Ramsey experiment and interleaving a (train of) flux pulse(s).

    Args:
        amplitudes: amplitudes of the flux pulses
        qb_name:
        operation_dict:
        cz_pulse_name: name of the flux pulse (should be in the operation_dict)
        phases (1D array, list): phases of the second pi-half pulse.
        flux_lengths (1D array, list): list containing the pulse durations.
        See Notes below.
        n_pulses (1D array, list): list containing the number of repetitions
         for the flux pulses. See Notes below.

        cal_points:
        upload:
    Notes:
        2 sorts of sequences can be generated based on the combination of
        (flux_lengths, n_pulses):
        1. (None, array):
         The ith created pulse sequence is:
        |          ---|X90|  ---------------------------------|X90||RO|
        |          --------(| - fp -| ) x n_pulses[i] ---------
       Each flux pulse has a duration equal to the stored value in the operations dict
        2. (array, None):
        The ith created pulse sequence is:
        |          ---|X90|  ---------------------------------|X90||RO|
       |          --------| -- fp --length=flux_lengths[i]----|
       and the duration of the single flux pulse is adapted according to the values
       specified in flux_lengths
       3. other combinations such as (array, array) or (None, None) are currently
       not supported.
    Returns:

    """
    seq_name = 'T2_freq_sweep_seq'
    ge_pulse = deepcopy(operation_dict['X90 ' + qb_name])
    ge_pulse['name'] = 'DF_X90'
    ge_pulse['element_name'] = 'DF_X90_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])

    if (flux_lengths, n_pulses) is (None, None):
        raise ValueError('Expected either flux_lengths or n_pulses but neither'
                         ' got provided.')
    elif flux_lengths is not None and n_pulses is not None:
        raise ValueError('Expected either flux_lengths or n_pulses but both'
                         ' got provided.')

    len_amp = len(amplitudes)
    len_flux = len(flux_lengths) if flux_lengths is not None else len(n_pulses)
    len_phase = len(phases)
    amplitudes = np.repeat(amplitudes, len_flux * len_phase)
    phases = np.tile(phases, len_flux * len_amp)
    #

    if flux_lengths is None:
        # create flux lengths with lengths equal to the current duration of
        # the flux pulse
        flux_lengths = np.ones_like(amplitudes) * flux_pulse["pulse_length"]
        n_pulses = np.tile(np.repeat(n_pulses, len_phase), len_amp)
    elif n_pulses is None:
        # single flux pulse (and length will be swept)
        n_pulses = np.ones_like(amplitudes, dtype=int)
        flux_lengths = np.tile(np.repeat(flux_lengths, len_phase), len_amp)


    ge_pulse2 = deepcopy(operation_dict['X90 ' + qb_name])
    ge_pulse2['name'] = 'DF_X90_2'
    ge_pulse2['element_name'] = 'DF_X90_el'

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'DF_Ro'
    ro_pulse['ref_pulse'] = 'DF_X90_2'
    ro_pulse['ref_point'] = 'end'
    ro_pulse['pulse_delay'] = 0

    swept_pulses = []
    print(amplitudes, phases, n_pulses, flux_lengths)
    for a, ph, n, fl in zip(amplitudes, phases, n_pulses, flux_lengths):
        print(fl)
        f = deepcopy(flux_pulse)
        f['amplitude'] = a
        f['pulse_length'] = fl
        fps = [deepcopy(f) for _ in range(n)]
        ge_pulse2 = deepcopy(ge_pulse2)
        ge_pulse2['phase'] = ph
        pulses = [deepcopy(ge_pulse)] + fps + [deepcopy(ge_pulse2),
                                               deepcopy(ro_pulse)]
        swept_pulses.append(pulses)


    seq = pulse_list_list_seq(swept_pulses, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict))

    seq.repeat_ro('RO ' + qb_name, operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def T1_freq_sweep_seq(amplitudes,
                   qb_name,
                   operation_dict,
                   cz_pulse_name,
                   flux_lengths,
                   cal_points=None,
                   upload=True,
                   prep_params=None):
    '''
    Performs a X180 pulse before changing the qubit frequency with the flux

    Timings of sequence

       |          ---|X180|  ------------------------------|RO|
       |          --------| --------- fluxpulse ---------- |
    '''
    if prep_params is None:
        prep_params = {}

    len_amp = len(amplitudes)
    amplitudes = np.repeat(amplitudes, len(flux_lengths))
    flux_lengths = np.tile(flux_lengths, len_amp)

    seq_name = 'T1_freq_sweep_sequence'
    ge_pulse = deepcopy(operation_dict['X180 ' + qb_name])
    ge_pulse['name'] = 'DF_Pi'
    ge_pulse['element_name'] = 'DF_Pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'DF_Flux'
    flux_pulse['ref_pulse'] = 'DF_Pi'
    flux_pulse['ref_point'] = 'end'
    flux_pulse['pulse_delay'] = 0  #-flux_pulse.get('buffer_length_start', 0)

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'DF_Ro'
    ro_pulse['ref_pulse'] = 'DF_Flux'
    ro_pulse['ref_point'] = 'end'

    ro_pulse['pulse_delay'] = flux_pulse.get('buffer_length_end', 0)

    pulses = [ge_pulse, flux_pulse, ro_pulse]

    swept_pulses = sweep_pulse_params(pulses, {
        'DF_Flux.amplitude': amplitudes,
        'DF_Flux.pulse_length': flux_lengths
    })

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    seq.repeat_ro('RO ' + qb_name, operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def add_suffix(operation_list, suffix):
    return [op + suffix for op in operation_list]