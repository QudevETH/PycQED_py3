import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control import sequence as sequence
from pycqed.measurement.waveform_control import segment as segment

import logging
log = logging.getLogger(__name__)


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


def add_preparation_pulses(pulse_list, operation_dict, qb_names,
                           preparation_type='wait', post_ro_wait=1e-6,
                           ro_separation=1.5e-6,
                           reset_reps=1, final_reset_pulse=True,
                           threshold_mapping=None):
    """
    Prepends to pulse_list the preparation pulses corresponding to preparation

    preparation:
        for active reset on |e>: ('active_reset_e', nr_resets)
        for active reset on |e> and |f>: ('active_reset_ef', nr_resets)
        for preselection: ('preselection', nr_readouts)
    """

    if threshold_mapping is None:
        threshold_mapping = {qbn: {0: 'g', 1: 'e'} for qbn in qb_names}

    # Calculate the length of a ge pulse, assumed the same for all qubits
    state_ops = dict(g=["I "], e=["X180 "], f=["X180_ef ", "X180 "])

    if 'ref_pulse' not in pulse_list[0]:
        first_pulse = deepcopy(pulse_list[0])
        first_pulse['ref_pulse'] = 'segment_start'
        pulse_list[0] = first_pulse

    if preparation_type == 'wait':
        return pulse_list
    elif 'active_reset' in preparation_type:
        reset_ro_pulses = []
        ops_and_codewords = {}
        for i, qbn in enumerate(qb_names):
            reset_ro_pulses.append(deepcopy(operation_dict['RO ' + qbn]))
            reset_ro_pulses[-1]['ref_point'] = 'start' if i != 0 else 'end'

            if preparation_type == 'active_reset_e':
                ops_and_codewords[qbn] = [
                    (state_ops[threshold_mapping[qbn][0]], 0),
                    (state_ops[threshold_mapping[qbn][1]], 1)]
            elif preparation_type == 'active_reset_ef':
                assert len(threshold_mapping[qbn]) == 4, \
                    "Active reset for the f-level requires a mapping of length 4" \
                    f" but only {len(threshold_mapping)} were given: " \
                    f"{threshold_mapping}"
                ops_and_codewords[qbn] = [
                    (state_ops[threshold_mapping[qbn][0]], 0),
                    (state_ops[threshold_mapping[qbn][1]], 1),
                    (state_ops[threshold_mapping[qbn][2]], 2),
                    (state_ops[threshold_mapping[qbn][3]], 3)]
            else:
                raise ValueError(f'Invalid preparation type {preparation_type}')

        reset_pulses = []
        for i, qbn in enumerate(qb_names):
            for ops, codeword in ops_and_codewords[qbn]:
                for j, op in enumerate(ops):
                    reset_pulses.append(deepcopy(operation_dict[op + qbn]))
                    reset_pulses[-1]['codeword'] = codeword
                    if j == 0:
                        reset_pulses[-1]['ref_point'] = 'start'
                        reset_pulses[-1]['pulse_delay'] = post_ro_wait
                    else:
                        reset_pulses[-1]['ref_point'] = 'start'
                        pulse_length = 0
                        for jj in range(1, j+1):
                            if 'pulse_length' in reset_pulses[-1-jj]:
                                pulse_length += reset_pulses[-1-jj]['pulse_length']
                            else:
                                pulse_length += reset_pulses[-1-jj]['sigma'] * \
                                                reset_pulses[-1-jj]['nr_sigma']
                        reset_pulses[-1]['pulse_delay'] = post_ro_wait+pulse_length

        prep_pulse_list = []
        for rep in range(reset_reps):
            ro_list = deepcopy(reset_ro_pulses)
            ro_list[0]['name'] = 'refpulse_reset_element_{}'.format(rep)

            for pulse in ro_list:
                pulse['element_name'] = 'reset_ro_element_{}'.format(rep)
            if rep == 0:
                ro_list[0]['ref_pulse'] = 'segment_start'
                ro_list[0]['pulse_delay'] = -reset_reps * ro_separation
            else:
                ro_list[0]['ref_pulse'] = 'refpulse_reset_element_{}'.format(
                    rep-1)
                ro_list[0]['pulse_delay'] = ro_separation
                ro_list[0]['ref_point'] = 'start'

            rp_list = deepcopy(reset_pulses)
            for j, pulse in enumerate(rp_list):
                pulse['element_name'] = 'reset_pulse_element_{}'.format(rep)
                pulse['ref_pulse'] = 'refpulse_reset_element_{}'.format(rep)
            prep_pulse_list += ro_list
            prep_pulse_list += rp_list

        if final_reset_pulse:
            rp_list = deepcopy(reset_pulses)
            for pulse in rp_list:
                pulse['element_name'] = f'reset_pulse_element_{reset_reps}'
            pulse_list += rp_list

        return prep_pulse_list + pulse_list

    elif preparation_type == 'preselection':
        preparation_pulses = []
        for i, qbn in enumerate(qb_names):
            preparation_pulses.append(deepcopy(operation_dict['RO ' + qbn]))
            preparation_pulses[-1]['ref_point'] = 'start'
            preparation_pulses[-1]['element_name'] = 'preselection_element'
        preparation_pulses[0]['ref_pulse'] = 'segment_start'
        preparation_pulses[0]['pulse_delay'] = -ro_separation

        return preparation_pulses + pulse_list


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


def add_suffix(operation_list, suffix):
    return [op + suffix for op in operation_list]