import logging
from pprint import pprint

import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control import sequence as sequence
from pycqed.measurement.waveform_control import segment as segment
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb

import logging
log = logging.getLogger(__name__)





def ramsey_seq_cont_drive(times, pulse_pars, RO_pars,
                          artificial_detuning=None, cal_points=True,
                          upload=True, return_seq=False, **kw):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        cal_points:          whether to use calibration points or not
    '''
    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])

    DRAG_length = pulse_pars['nr_sigma']*pulse_pars['sigma']
    cont_drive_ampl = 0.1 * pulse_pars['amplitude']
    X180_pulse = deepcopy(pulses['X180'])
    cos_pulse = {'pulse_type': 'CosPulse_gauss_rise',
                 'channel': X180_pulse['I_channel'],
                 'frequency': X180_pulse['mod_frequency'],
                 'length': 0,
                 'phase': X180_pulse['phi_skew'],
                 'amplitude': cont_drive_ampl * X180_pulse['alpha'],
                 'pulse_delay': 0,
                 'ref_point': 'end'}
    sin_pulse = {'pulse_type': 'CosPulse_gauss_rise',
                 'channel': X180_pulse['Q_channel'],
                 'frequency': X180_pulse['mod_frequency'],
                 'length': 0,
                 'phase': 90,
                 'amplitude': cont_drive_ampl * X180_pulse['alpha'],
                 'pulse_delay': 0,
                 'ref_point': 'simultaneous'}

    for i, tau in enumerate(times):

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            X90_separation = tau - DRAG_length
            if X90_separation > 0:
                pulse_pars_x2['ref_point'] = 'end'
                cos_pls1 = deepcopy(cos_pulse)
                sin_pls1 = deepcopy(sin_pulse)
                cos_pls1['length'] = X90_separation/2
                sin_pls1['length'] = X90_separation/2
                cos_pls2 = deepcopy(cos_pls1)
                sin_pls2 = deepcopy(sin_pls1)
                cos_pls2['amplitude'] = -cos_pls1['amplitude']
                cos_pls2['pulse_type'] = 'CosPulse_gauss_fall'
                sin_pls2['amplitude'] = -sin_pls1['amplitude']
                sin_pls2['pulse_type'] = 'CosPulse_gauss_fall'

                pulse_dict_list = [pulses['X90'], cos_pls1, sin_pls1,
                                   cos_pls2, sin_pls2, pulse_pars_x2, RO_pars]
            else:
                pulse_pars_x2['ref_point'] = 'start'
                pulse_pars_x2['pulse_delay'] = tau
                pulse_dict_list = [pulses['X90'], pulse_pars_x2, RO_pars]

            seg = segment.Segment('segment_{}'.format(i), pulse_dict_list)

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True, upload=True, return_seq=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        cal_points:          whether to use calibration points or not
    '''
    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['ref_point'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau
        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['X90'], pulse_pars_x2, RO_pars])

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def ramsey_seq_VZ(times, pulse_pars, RO_pars,
                   artificial_detuning=None,
                   cal_points=True, upload=True, return_seq=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        cal_points:          whether to use calibration points or not
    '''
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['ref_point'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
        else:
            Dphase = ((tau-times[0]) * 1e6 * 360) % 360
        Z_gate = Z(Dphase, pulse_pars)

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses['X90'], Z_gate, pulse_pars_x2, RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def ramsey_seq_multiple_detunings(times, pulse_pars, RO_pars,
               artificial_detunings=None, cal_points=True,
               upload=True, return_seq=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    !!! Each value in the times array must be repeated len(artificial_detunings)
    times!!!
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detunings: list of artificial_detunings (Hz) implemented
                              using phase
        cal_points:          whether to use calibration points or not
    '''
    seq_name = 'Ramsey_sequence_multiple_detunings'
    seq = sequence.Sequence(seq_name)
    ps.Pulsar.get_instance().update_channel_settings()
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['ref_point'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau
        art_det = artificial_detunings[i % len(artificial_detunings)]

        if art_det is not None:
            Dphase = ((tau-times[0]) * art_det * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X90'], pulse_pars_x2, RO_pars])
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name




def echo_seq(times, pulse_pars, RO_pars,
             artificial_detuning=None,
             cal_points=True, upload=True, return_seq=False):
    '''
    Echo sequence for a single qubit using the tektronix.
    Input pars:
        times:          array of times between (start of) pulses (s)
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        pulse_pars:     dict containing the pulse parameters
        RO_pars:        dict containing the RO parameters
        cal_points:     whether to use calibration points or not
    '''
    seq_name = 'Echo_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    center_X180 = deepcopy(pulses['X180'])
    final_X90 = deepcopy(pulses['X90'])
    center_X180['ref_point'] = 'start'
    final_X90['ref_point'] = 'start'

    for i, tau in enumerate(times):
        center_X180['pulse_delay'] = tau/2
        final_X90['pulse_delay'] = tau/2
        if artificial_detuning is not None:
            final_X90['phase'] = (tau-times[0]) * artificial_detuning * 360
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X90'], center_X180,
                                  final_X90, RO_pars])
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def single_state_active_reset(operation_dict, qb_name,
                              state='e', upload=True, prep_params={}):
    '''
    OffOn sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        pulse_pars_2nd:      dict containing the pulse parameters of ef transition.
                             Required if state is 'f'.
        Initialize:          adds an exta measurement before state preparation
                             to allow initialization by post-selection
        Post-measurement delay:  should be sufficiently long to avoid
                             photon-induced gate errors when post-selecting.
        state:               specifies for which state a pulse should be
                             generated (g,e,f)
        preselection:        adds an extra readout pulse before other pulses.
    '''
    seq_name = 'single_state_sequence'
    seq = sequence.Sequence(seq_name)

    # Create dicts with the parameters for all the pulses
    state_ops = dict(g=["I", "RO"], e=["X180", "RO"], f=["X180", "X180_ef", "RO"])
    pulses = [deepcopy(operation_dict[op])
              for op in add_suffix(state_ops[state], " " + qb_name)]

    #add preparation pulses
    pulses_with_prep = \
        add_preparation_pulses(pulses, operation_dict, [qb_name], **prep_params)

    seg = segment.Segment('segment_{}_level'.format(state), pulses_with_prep)
    seq.add(seg)

    # reuse sequencer memory by repeating readout pattern
    seq.repeat_ro(f"RO {qb_name}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())




def over_under_rotation_seq(qb_name, nr_pi_pulses_array, operation_dict,
                            pi_pulse_amp=None, cal_points=True, upload=True):
    seq_name = 'Over-under rotation sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    X90 = deepcopy(operation_dict['X90 ' + qb_name])
    X180 = deepcopy(operation_dict['X180 ' + qb_name])
    if pi_pulse_amp is not None:
        X90['amplitude'] = pi_pulse_amp/2
        X180['amplitude'] = pi_pulse_amp

    for i, N in enumerate(nr_pi_pulses_array):
        if cal_points and (i == (len(nr_pi_pulses_array)-4) or
                           i == (len(nr_pi_pulses_array)-3)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [operation_dict['I ' + qb_name],
                                   operation_dict['RO ' + qb_name]])
        elif cal_points and (i == (len(nr_pi_pulses_array)-2) or
                             i == (len(nr_pi_pulses_array)-1)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [operation_dict['X180 ' + qb_name],
                                   operation_dict['RO ' + qb_name]])
        else:
            pulse_list = [X90]
            pulse_list += N*[X180]
            pulse_list += [operation_dict['RO ' + qb_name]]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return


# Helper functions

def pulse_list_list_seq(pulse_list_list, name='pulse_list_list_sequence',
                        upload=True, fast_mode=False):
    seq = sequence.Sequence(name)
    for i, pulse_list in enumerate(pulse_list_list):
        seq.add(segment.Segment('segment_{}'.format(i), pulse_list,
                                fast_mode=fast_mode))
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq

def prepend_pulses(pulse_list, pulses_to_prepend):
    """
    Prepends a list of pulse to a list of pulses with correct referencing.
    :param pulse_list: initial pulse list
    :param pulses_to_prepend: pulse to prepend
    :return:
        list of pulses where prepended pulses are at the beginning of the
        returned list
    """
    all_pulses = deepcopy(pulse_list)
    for i, p in enumerate(reversed(pulses_to_prepend)):
        try:
            p['ref_pulse'] = all_pulses[0]['name']
        except KeyError:
            all_pulses[0]['name'] = 'fist_non_prepended_pulse'
            p['ref_pulse'] = all_pulses[0]['name']
        p['name'] = p.get('name',
                          f'prepended_pulse_{len(pulses_to_prepend) - i - 1}')
        p['ref_point'] = 'start'
        p['ref_point_new'] = 'end'
        all_pulses = [p] + all_pulses
    return all_pulses


def add_preparation_pulses(pulse_list, operation_dict, qb_names,
                           **prep_params):
    from pycqed.measurement.waveform_control import circuit_builder as cb_mod
    cb = cb_mod.CircuitBuilder(qubits=qb_names, operation_dict=operation_dict,
                               prep_params=prep_params)
    init_pulses = cb.initialize().build()
    init_pulses[0]['ref_pulse'] = 'init_start'
    return init_pulses + pulse_list


def sweep_pulse_params(pulses, params, pulse_not_found_warning=True):
    """
    Sweeps a list of pulses over specified parameters.
    Args:
        pulses (list): All pulses. Pulses which have to be swept over need to
            have a 'name' key.
        params (dict):  keys in format <pulse_name>.<pulse_param_name>,
            values are the sweep values. <pulse_name> can be formatted as
            exact name or '<pulse_starts_with>*<pulse_endswith>'. In that case
            all pulses with name starting with <pulse_starts_with> and ending
            with <pulse_endswith> will be modified. eg. "Rabi_*" will modify
            Rabi_1, Rabi_2 in [Rabi_1, Rabi_2, Other_Pulse]
        pulse_not_found_warning (bool, default: True) whether a warning
            should be issued if no pulse matches a given pulse name.

    Returns: a list of pulses_lists where each element is to be used
        for a single segment

    """

    def check_pulse_name(pulse, target_name):
        """
        Checks if an asterisk is found in the name, in that case only the first
        part of the name is compared
        """
        target_name_splitted = target_name.split("*")
        if len(target_name_splitted) == 1:
            return pulse.get('name', "") == target_name
        elif len(target_name_splitted) == 2:
            return pulse.get('name', "").startswith(target_name_splitted[0]) \
                   and pulse.get('name', "").endswith(target_name_splitted[1])
        else:
            raise Exception(f"Only one asterisk in pulse_name is allowed,"
                            f" more than one in {target_name}")

    swept_pulses = []
    if len(params.keys()) == 0:
        log.warning("No params to sweep. Returning unchanged pulses.")
        return pulses

    n_sweep_points = len(list(params.values())[0])

    assert np.all([len(v) == n_sweep_points for v in params.values()]), \
        "Parameter sweep values are not all of the same length: {}" \
            .format({n: len(v) for n, v in params.items()})

    for i in range(n_sweep_points):
        pulses_cp = deepcopy(pulses)
        for name, sweep_values in params.items():
            pulse_name, param_name = name.split('.')
            pulse_indices = [i for i, p in enumerate(pulses)
                             if check_pulse_name(p, pulse_name)]
            if len(pulse_indices) == 0 and pulse_not_found_warning:
                log.warning(f"No pulse with name {pulse_name} found in list:"
                            f"{[p.get('name', 'No Name') for p in pulses]}")
            for p_idx in pulse_indices:
                pulses_cp[p_idx][param_name] = sweep_values[i]
                # pulses_cp[p_idx].pop('name', 0)
        swept_pulses.append(pulses_cp)

    return swept_pulses


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

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude'] * pulse_pars['amp90_scale']

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] += 90
    pulses['mY180']['phase'] += 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] += 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] += 90

    pulses_sim = {key + 's': deepcopy(val) for key, val in pulses.items()}
    for val in pulses_sim.values():
        val['ref_point'] = 'start'

    pulses.update(pulses_sim)

    # Software Z-gate: apply phase offset to all subsequent X and Y pulses
    target_qubit = pulse_pars.get('basis', None)
    if target_qubit is not None:
        Z0 = {'pulse_type': 'VirtualPulse',
              'basis_rotation': {target_qubit: 0},
              'operation_type': 'Virtual'}
        pulses.update({'Z0': Z0,
                       'Z180': deepcopy(Z0),
                       'mZ180': deepcopy(Z0),
                       'Z90': deepcopy(Z0),
                       'mZ90': deepcopy(Z0)})
        pulses['Z180']['basis_rotation'][target_qubit] += 180
        pulses['mZ180']['basis_rotation'][target_qubit] += -180
        pulses['Z90']['basis_rotation'][target_qubit] += 90
        pulses['mZ90']['basis_rotation'][target_qubit] += -90

    return pulses


def Z(theta=0, pulse_pars=None):

    """
    Software Z-gate of arbitrary rotation.

    :param theta:           rotation angle
    :param pulse_pars:      pulse parameters (dict)

    :return: Pulse dict of the Z-gate
    """
    if pulse_pars is None:
        raise ValueError('Pulse_pars is None.')
    else:
        pulses = get_pulse_dict_from_pars(pulse_pars)

    Z_gate = deepcopy(pulses['Z180'])
    Z_gate['phase'] = theta

    return Z_gate


def add_suffix(operation_list, suffix):
    return [op + suffix for op in operation_list]