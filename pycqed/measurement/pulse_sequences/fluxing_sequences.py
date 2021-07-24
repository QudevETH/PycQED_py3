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


def dynamic_phase_seq(qb_names, hard_sweep_dict, operation_dict,
                      cz_pulse_name, cal_points=None,
                      upload=False, prep_params=None,
                      nr_cz_gates=1, prepend_pulse_dicts=None):
    '''
    Performs a Ramsey with interleaved Flux pulse
    Sequence
                   |fluxpulse|
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep phase

    :param qb_names: (list) list of qubit names
    :param hard_sweep_dict: (dict) specifies the sweep information for
        the hard sweep. If None, will default to
            hard_sweep_params['phase'] = {
                'values': np.tile(np.linspace(0, 2 * np.pi, 6) * 180 / np.pi, 2),
                'unit': 'deg'}
    :param operation_dict: (list) list of pulse dictionaries for all qubits
        in qb_names
    :param cz_pulse_name: (str) name of the CZ pulse in the operation dict
    :param cal_points: (CalibrationPoints object)
    :param upload: (bool) whether to upload to AWGs
    :param prep_params: (dict) preparation parameters
    :param nr_cz_gates: (int) number of two-qubit gates to insert between
        Ramsey pulses
    :param prepend_pulse_dicts: (list) list of pulse dictionaries to prepend
        to each segment
    '''

    seq_name = 'Dynamic_phase_seq'

    ge_half_start = [deepcopy(operation_dict['X90 ' + qb_name]) for qb_name in
                     qb_names]
    for i, p in enumerate(ge_half_start):
        p['name'] = f'pi_half_start_{qb_names[i]}'
        p['ref_pulse'] = 'start'
        p['element_name'] = 'pi_half_start'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])

    ge_half_end = [deepcopy(operation_dict['X90 ' + qb_name]) for qb_name in
                   qb_names]
    for i, p in enumerate(ge_half_end):
        p['name'] = f'pi_half_end_{qb_names[i]}'
        p['element_name'] = 'pi_half_end'
        p['ref_pulse'] = 'flux_{}'.format(nr_cz_gates-1)

    ro_pulses = generate_mux_ro_pulse_list(qb_names, operation_dict)

    if prep_params is None:
        prep_params = {}
    if prepend_pulse_dicts is None:
        prepend_pulse_dicts = {}
    pulse_list = []
    for i, pp in enumerate(prepend_pulse_dicts):
        prepend_pulse = deepcopy(operation_dict[pp['op_code']])
        prepend_pulse['name'] = f'prepend_{i}'
        prepend_pulse.update(pp)
        pulse_list += [prepend_pulse]

    pulse_list += Block("ge_half_start pulses", ge_half_start).build()

    flux_pulses = []
    for n in range(nr_cz_gates):
        fp = deepcopy(flux_pulse)
        fp['name'] = 'flux_{}'.format(n)
        flux_pulses.append(fp)
    pulse_list += flux_pulses + ge_half_end + ro_pulses
    hsl = len(list(hard_sweep_dict.values())[0]['values'])

    params_to_set = [param 
        for param in ['amplitude', 'amplitude2', 'dv_dphi', 'trans_amplitude', 
            'trans_amplitude2', 'amplitude_offset', 'amplitude_offset2'] 
        if param in flux_pulse]
    if len(params_to_set) == 0:
        raise ValueError('Unknown flux pulse amplitude control parameter. '
                         'Cannot do measurement without flux pulse.')

    params = {}
    for i, flux_pulse in enumerate(flux_pulses):
        params.update({f'flux_{i}.{param_to_set}':
                  np.concatenate(
                                 [flux_pulse[param_to_set] * np.ones(hsl // 2),
                                  np.zeros(hsl // 2)]) for param_to_set in params_to_set
                  })

        if 'aux_channels_dict' in flux_pulse:
            params.update({f'flux_{i}.aux_channels_dict': np.concatenate([
                [flux_pulse['aux_channels_dict']] * (hsl // 2),
                 [{}] * (hsl // 2)])})
    for qb_name in qb_names:
        params.update({f'pi_half_end_{qb_name}.{k}': v['values']
                       for k, v in hard_sweep_dict.items()})
    swept_pulses = sweep_pulse_params(pulse_list, params)
    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, qb_names, **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def Ramsey_time_with_flux_seq(qb_name, hard_sweep_dict, operation_dict,
                            cz_pulse_name,
                            artificial_detunings=0,
                            cal_points=None,
                            upload=False, prep_params=None):
    '''
    Performs a Ramsey with interleaved Flux pulse
    Sequence
      | ----------  fluxpulse  ---------------  |
        |X90|  -------------------     |X90|  ---  |RO|
                                     sweep time
    '''
    if prep_params is None:
        prep_params = {}

    seq_name = 'Ramsey_flux_seq'

    times = hard_sweep_dict['Delay']['values']

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'flux'
    flux_pulse['element_name'] = 'flux_el'
    flux_pulse['pulse_length'] = 4*np.max(times)
    flux_pulse['pulse_delay'] = -0.5*np.max(times)
    flux_pulse['ref_point'] = 'start'
    flux_pulse['ref_pulse'] = 'Ramsey_x1'

    print(flux_pulse)

    ramsey_ops = ["X90"] * 2
    ramsey_ops += ["RO"]
    ramsey_ops = add_suffix(ramsey_ops, " " + qb_name)


    # pulses
    ramsey_pulses = [deepcopy(operation_dict[op]) for op in ramsey_ops]

    ramsey_pulses += [flux_pulse]

    # name and reference swept pulse
    ramsey_pulses[0]["name"] = f"Ramsey_x1"
    ramsey_pulses[1]["name"] = f"Ramsey_x2"
    ramsey_pulses[1]['ref_point'] = 'start'


    # compute dphase
    a_d = artificial_detunings if np.ndim(artificial_detunings) == 1 \
        else [artificial_detunings]
    dphase = [((t - times[0]) * a_d[i % len(a_d)] * 360) % 360
              for i, t in enumerate(times)]
    # sweep pulses
    params = {f'Ramsey_x2.pulse_delay': times}
    params.update({f'Ramsey_x2.phase': dphase})
    swept_pulses = sweep_pulse_params(ramsey_pulses, params)

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

    return seq, np.arange(seq.n_acq_elements())


def chevron_seqs(qbc_name, qbt_name, qbr_name, hard_sweep_dict, soft_sweep_dict,
                 operation_dict, cz_pulse_name, num_cz_gates=1,
                 prep_params=None,
                 cal_points=None, upload=True):
    '''
    chevron sequence (sweep of the flux pulse length)

    Timings of sequence
                                  <-- length -->
    qb_control:    |X180|  ---   |  fluxpulse   |

    qb_target:     |X180|  --------------------------------------  |RO|

   '''
    if prep_params is None:
        prep_params = {}

    seq_name = 'Chevron_sequence'

    ge_pulse_qbc = deepcopy(operation_dict['X180 ' + qbc_name])
    ge_pulse_qbc['name'] = 'chevron_pi_qbc'
    ge_pulse_qbt = deepcopy(operation_dict['X180s ' + qbt_name])
    ge_pulse_qbt['name'] = 'chevron_pi_qbt'
    ge_pulse_qbt['ref_point'] = 'end'
    ge_pulse_qbt['ref_point_new'] = 'end'
    for ge_pulse in [ge_pulse_qbc, ge_pulse_qbt]:
        ge_pulse['element_name'] = 'chevron_pi_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'chevron_flux'
    flux_pulse['element_name'] = 'chevron_flux_el'

    ro_pulses = generate_mux_ro_pulse_list([qbr_name],
                                           operation_dict)
    if 'pulse_length' in hard_sweep_dict:
        # add buffers to this delay (only used for FLIP Pulse
        nr_flux_buffer = 4 if flux_pulse['pulse_type'] == 'BufferedNZFLIPPulse' \
            else 2  # for NZ pulse we have four buffers
        max_flux_length = max(hard_sweep_dict['pulse_length']['values'])
        ro_pulses[0]['ref_pulse'] = 'chevron_pi_qbc'
        ro_pulses[0]['pulse_delay'] = num_cz_gates * \
            (max_flux_length + flux_pulse.get('buffer_length_start', 0) +
            flux_pulse.get('buffer_length_end', 0) +
            # applies to FLIP gate only. An additional buffer, that can be used to make sure the FPs have rising edges
            # at different times.
            nr_flux_buffer*flux_pulse.get('flux_buffer_length', 0) +
            nr_flux_buffer*flux_pulse.get('flux_buffer_length2', 0) +
            # applies to transition-controlled gate only
            flux_pulse.get('trans_length', 0))
    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        fp_list = []
        flux_p = deepcopy(flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        for j in range(num_cz_gates):
            fp = deepcopy(flux_p)
            fp['name'] = f'chevron_flux_{j}'
            fp_list += [fp]
        pulses = [ge_pulse_qbc, ge_pulse_qbt] + fp_list + ro_pulses
        swept_pulses = sweep_pulse_params(pulses, {
            f'chevron_flux_{j}.{k}': v['values']
            for k, v in hard_sweep_dict.items() for j in range(num_cz_gates)
        })
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    # 1. get all readout pulse names (if they are on different uhf,
    # they will be applied to different channels)
    ro_pulse_names = [f"RO {qbn}" for qbn in [qbr_name]]
    # 2. repeat readout for each ro_pulse.
    for seq in sequences:
        [seq.repeat_ro(pn, operation_dict) for pn in ro_pulse_names]
    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


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
                      flux_lengths,
                      phases,
                      cal_points=None,
                      upload=True):
    '''
    Performs a X180 pulse before changing the qubit frequency with the flux

    Timings of sequence

       |          ---|X180|  ------------------------------|RO|
       |          --------| --------- fluxpulse ---------- |
    '''

    len_amp = len(amplitudes)
    len_flux = len(flux_lengths)
    len_phase = len(phases)
    amplitudes = np.repeat(amplitudes, len_flux * len_phase)
    flux_lengths = np.tile(np.repeat(flux_lengths, len_phase), len_amp)
    phases = np.tile(phases, len_flux * len_amp)

    seq_name = 'T2_freq_sweep_seq'
    ge_pulse = deepcopy(operation_dict['X90 ' + qb_name])
    ge_pulse['name'] = 'DF_X90'
    ge_pulse['element_name'] = 'DF_X90_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'DF_Flux'
    flux_pulse['ref_pulse'] = 'DF_X90'
    flux_pulse['ref_point'] = 'end'
    flux_pulse['pulse_delay'] = 0  #-flux_pulse.get('buffer_length_start', 0)

    ge_pulse2 = deepcopy(operation_dict['X90 ' + qb_name])
    ge_pulse2['name'] = 'DF_X90_2'
    ge_pulse2['ref_pulse'] = 'DF_Flux'
    ge_pulse2['ref_point'] = 'end'
    ge_pulse2['pulse_delay'] = 0
    ge_pulse2['element_name'] = 'DF_X90_el'

    ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
    ro_pulse['name'] = 'DF_Ro'
    ro_pulse['ref_pulse'] = 'DF_X90_2'
    ro_pulse['ref_point'] = 'end'
    ro_pulse['pulse_delay'] = 0

    pulses = [ge_pulse, flux_pulse, ge_pulse2, ro_pulse]

    swept_pulses = sweep_pulse_params(
        pulses, {
            'DF_Flux.amplitude': amplitudes,
            'DF_Flux.pulse_length': flux_lengths,
            'DF_X90_2.phase': phases
        })

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


def cz_bleed_through_phase_seq(phases, qb_name, CZ_pulse_name, CZ_separation,
                               operation_dict, oneCZ_msmt=False, nr_cz_gates=1,
                               verbose=False, upload=True, return_seq=False,
                               upload_all=True, cal_points=True):
    '''
    Performs a Ramsey-like with interleaved Flux pulse

    Timings of sequence
                           CZ_separation
    |CZ|-|CZ|- ... -|CZ| <---------------> |X90|-|CZ|-|X90|-|RO|

    Args:
        end_times: numpy array of delays after second CZ pulse
        qb: qubit object (must have the methods get_operation_dict(),
            get_drive_pars() etc.
        CZ_pulse_name: str of the form
            'CZ ' + qb_target.name + ' ' + qb_control.name
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
        'cz_bleed_through_phase_seq has not been '
        'converted to the latest waveform generation code and can not be used.')

    # if maximum_CZ_separation is None:
    #     maximum_CZ_separation = CZ_separation

    seq_name = 'CZ Bleed Through phase sweep'
    seq = sequence.Sequence(seq_name)
    el_list = []

    X90_1 = deepcopy(operation_dict['X90 ' + qb_name])
    X90_2 = deepcopy(operation_dict['X90 ' + qb_name])
    RO_pars = deepcopy(operation_dict['RO ' + qb_name])
    CZ_pulse1 = deepcopy(operation_dict[CZ_pulse_name])
    CZ_pulse_len = CZ_pulse1['pulse_length']
    drag_pulse_len = deepcopy(X90_1['sigma']* X90_1['nr_sigma'])
    spacerpulse = {'pulse_type': 'SquarePulse',
                   'channel': X90_1['I_channel'],
                   'amplitude': 0.0,
                   'length': CZ_separation - drag_pulse_len,
                   'ref_point': 'end',
                   'pulse_delay': 0}
    if oneCZ_msmt:
        spacerpulse_X90 = {'pulse_type': 'SquarePulse',
                           'channel': X90_1['I_channel'],
                           'amplitude': 0.0,
                           'length': CZ_pulse1['buffer_length_start'] +
                                     CZ_pulse1['buffer_length_end'] +
                                     CZ_pulse_len,
                           'ref_point': 'end',
                           'pulse_delay': 0}
        main_pulse_list = [CZ_pulse1, spacerpulse,
                           X90_1, spacerpulse_X90]
    else:
        CZ_pulse2 = deepcopy(operation_dict[CZ_pulse_name])
        main_pulse_list = int(nr_cz_gates)*[CZ_pulse1]
        main_pulse_list += [spacerpulse, X90_1, CZ_pulse2]
    el_main = multi_pulse_elt(0, station,  main_pulse_list,
                              trigger=True, name='el_main')
    el_list.append(el_main)

    if upload_all:
        upload_AWGs = 'all'
        upload_channels = 'all'
        # else:
        # upload_AWGs = [station.pulsar.get(CZ_pulse1['channel'] + '_AWG')] + \
        #               [station.pulsar.get(ch + '_AWG') for ch in
        #                CZ_pulse1['aux_channels_dict']]
        # upload_channels = [CZ_pulse1['channel']] + \
        #                   list(CZ_pulse1['aux_channels_dict'])

    for i, theta in enumerate(phases):
        if cal_points and (theta == phases[-4] or theta == phases[-3]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        elif cal_points and (theta == phases[-2] or theta == phases[-1]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qb_name], RO_pars],
                                 name='el_{}'.format(i+1), trigger=True)
            el_list.append(el)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=True)
        else:
            X90_2['phase'] = theta*180/np.pi
            el = multi_pulse_elt(i+1, station, [X90_2, RO_pars], trigger=False,
                                 name='el_{}'.format(i+1),
                                 previous_element=el_main)
            el_list.append(el)

            seq.append('m_{}'.format(i), 'el_main',  trigger_wait=True)
            seq.append('e_{}'.format(3*i), 'el_{}'.format(i+1),
                       trigger_wait=False)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def cphase_seqs(qbc_name, qbt_name, hard_sweep_dict, soft_sweep_dict,
                operation_dict, cz_pulse_name, num_cz_gates=1,
                max_flux_length=None, cal_points=None, upload=True,
                prep_params=None, prepend_pulse_dicts=None):

    assert num_cz_gates % 2 != 0

    seq_name = 'Cphase_sequence'

    if prep_params is None:
        prep_params = {}
    if prepend_pulse_dicts is None:
        prepend_pulse_dicts = {}
    prepend_pulses = []
    for i, pp in enumerate(prepend_pulse_dicts):
        prepend_pulse = deepcopy(operation_dict[pp['op_code']])
        prepend_pulse['name'] = f'prepend_{i}'
        prepend_pulse.update(pp)
        prepend_pulses += [prepend_pulse]

    initial_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
                         deepcopy(operation_dict['X90s ' + qbt_name])]
    initial_rotations[0]['name'] = 'cphase_init_pi_qbc'
    initial_rotations[1]['name'] = 'cphase_init_pihalf_qbt'
    initial_rotations[1]['ref_point'] = 'end'
    initial_rotations[1]['ref_point_new'] = 'end'
    for rot_pulses in initial_rotations:
        rot_pulses['element_name'] = 'cphase_initial_rots_el'

    flux_pulse = deepcopy(operation_dict[cz_pulse_name])
    flux_pulse['name'] = 'cphase_flux'

    final_rotations = [deepcopy(operation_dict['X180 ' + qbc_name]),
                       deepcopy(operation_dict['X90s ' + qbt_name])]
    final_rotations[0]['name'] = 'cphase_final_pi_qbc'
    final_rotations[1]['name'] = 'cphase_final_pihalf_qbt'
    # final_rotations = [deepcopy(operation_dict['X90 ' + qbt_name])]
    # final_rotations[0]['name'] = 'cphase_final_pihalf_qbt'

    for rot_pulses in final_rotations:
        rot_pulses['element_name'] = 'cphase_final_rots_el'

    # set pulse delay of final_rotations[0] to max_flux_length
    if max_flux_length is None:
        if 'pulse_length' in soft_sweep_dict:
            max_flux_length = max(soft_sweep_dict['pulse_length']['values'])
            print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
                  f'from sweep points.')
        else:
            max_flux_length = flux_pulse['pulse_length']
            print(f'max_pulse_length = {max_flux_length*1e9:.2f} ns, '
                  f'from pulse dict.')
    # add buffers to this delay (only used for FLIP Pulse
    nr_flux_buffer = 4 if flux_pulse['pulse_type']=='BufferedNZFLIPPulse' \
        else 2  # for NZ pulse we have four buffers
    delay = num_cz_gates*(max_flux_length +
        flux_pulse.get('buffer_length_start', 0) +
        flux_pulse.get('buffer_length_end', 0) +
        # applies to FLIP gate only. An additional buffer, that can be used to make sure the FPs have rising edges at
        # different times.
        nr_flux_buffer*flux_pulse.get('flux_buffer_length', 0) +
        nr_flux_buffer*flux_pulse.get('flux_buffer_length2', 0) +
        # applies to NZTC gate only
        flux_pulse.get('trans_length', 0))
    # # ensure the delay is commensurate with 16/2.4e9
    # comm_const = (16/2.4e9)
    # if delay % comm_const > 1e-15:
    #     delay = comm_const * (delay // comm_const + 1)
    #     print(f'delay adjusted to {delay*1e9:.2f} ns '
    #           f'to fulfill commensurability conditions with 16/2.4e9.')
    final_rotations[0]['ref_pulse'] = 'cphase_init_pi_qbc'
    final_rotations[0]['pulse_delay'] = delay

    ro_pulses = generate_mux_ro_pulse_list([qbc_name, qbt_name],
                                            operation_dict)

    # make sure RO happens after last ge
    ro_pulses[0]['ref_pulse'] = [p['name'] for p in final_rotations]

    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    params = {'cphase_init_pi_qbc.amplitude': np.concatenate(
        [initial_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)]),
              'cphase_final_pi_qbc.amplitude': np.concatenate(
        [final_rotations[0]['amplitude']*np.ones(hsl//2), np.zeros(hsl//2)])}
    params.update({f'cphase_final_pihalf_qbt.{k}': v['values']
                   for k, v in hard_sweep_dict.items()})

    ssl = len(list(soft_sweep_dict.values())[0]['values'])
    sequences = []
    for i in range(ssl):
        fp_list = []
        flux_p = deepcopy(flux_pulse)
        flux_p.update({k: v['values'][i] for k, v in soft_sweep_dict.items()})
        for j in range(num_cz_gates):
            fp = deepcopy(flux_p)
            fp['name'] = f'cphase_flux_{j}'
            fp_list += [fp]
        pulses = prepend_pulses + initial_rotations + fp_list + \
                 final_rotations + ro_pulses
        swept_pulses = sweep_pulse_params(pulses, params)
        swept_pulses_with_prep = \
            [add_preparation_pulses(p, operation_dict, [qbc_name, qbt_name],
                                    **prep_params)
             for p in swept_pulses]
        seq = pulse_list_list_seq(swept_pulses_with_prep,
                                  seq_name+f'_{i}', upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    for s in sequences:
        s.repeat_ro(f"RO {qbc_name}", operation_dict)
        s.repeat_ro(f"RO {qbt_name}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), np.arange(ssl)


def add_suffix(operation_list, suffix):
    return [op + suffix for op in operation_list]