import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    sweep_pulse_params, add_preparation_pulses, pulse_list_list_seq

import logging
log = logging.getLogger(__name__)


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

    if (flux_lengths, n_pulses) == (None, None):
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
    for a, ph, n, fl in zip(amplitudes, phases, n_pulses, flux_lengths):
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
