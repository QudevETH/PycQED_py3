import logging
log = logging.getLogger(__name__)
import itertools
import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
import pycqed.measurement.randomized_benchmarking.randomized_benchmarking as rb
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq, \
    prepend_pulses, add_suffix, sweep_pulse_params
from pycqed.measurement.gate_set_tomography.gate_set_tomography import \
    create_experiment_list_pyGSTi_qudev as get_exp_list
from pycqed.measurement.waveform_control import pulsar as ps
import pycqed.measurement.waveform_control.segment as segment
from pycqed.analysis_v2 import tomography_qudev as tomo

station = None
kernel_dir = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)
cached_kernels = {}


def n_qubit_off_on(pulse_pars_list, RO_pars_list, return_seq=False,
                   parallel_pulses=False, preselection=False, upload=True,
                   RO_spacing=2000e-9):
    n = len(pulse_pars_list)
    seq_name = '{}_qubit_OffOn_sequence'.format(n)
    seq = sequence.Sequence(seq_name)
    seg_list = []

    RO_pars_list_presel = deepcopy(RO_pars_list)
    
    for i, RO_pars in enumerate(RO_pars_list):
        RO_pars['name'] = 'RO_{}'.format(i)
        RO_pars['element_name'] = 'RO'
        if i != 0:
            RO_pars['ref_point'] = 'start'
    for i, RO_pars_presel in enumerate(RO_pars_list_presel):
        RO_pars_presel['ref_pulse'] = RO_pars_list[-1]['name']
        RO_pars_presel['ref_point'] = 'start'
        RO_pars_presel['element_name'] = 'RO_presel'
        RO_pars_presel['pulse_delay'] = -RO_spacing

    # Create a dict with the parameters for all the pulses
    pulse_dict = dict()
    for i, pulse_pars in enumerate(pulse_pars_list):
        pars = pulse_pars.copy()
        if i == 0 and parallel_pulses:
            pars['ref_pulse'] = 'segment_start'
        if i != 0 and parallel_pulses:
            pars['ref_point'] = 'start'
        pulses = add_suffix_to_dict_keys(
            get_pulse_dict_from_pars(pars), ' {}'.format(i))
        pulse_dict.update(pulses)

    # Create a list of required pulses
    pulse_combinations = []

    for pulse_list in itertools.product(*(n*[['I', 'X180']])):
        pulse_comb = (n)*['']
        for i, pulse in enumerate(pulse_list):
            pulse_comb[i] = pulse + ' {}'.format(i)
        pulse_combinations.append(pulse_comb)
    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for j, p in enumerate(pulse_comb):
            pulses += [pulse_dict[p]]
        pulses += RO_pars_list
        if preselection:
            pulses = pulses + RO_pars_list_presel

        seg = segment.Segment('segment_{}'.format(i), pulses)
        seg_list.append(seg)
        seq.add(seg)

    repeat_dict = {}
    repeat_pattern = ((1.0 + int(preselection))*len(pulse_combinations),1)
    for i, RO_pars in enumerate(RO_pars_list):
        repeat_dict = seq.repeat(RO_pars, None, repeat_pattern)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def n_qubit_tomo_seq(
        qubit_names, operation_dict, prep_sequence=None,
        prep_name=None,
        rots_basis=tomo.DEFAULT_BASIS_ROTS,
        upload=True, return_seq=False,
        preselection=False, ro_spacing=1e-6):
    """

    """

    # create the sequence
    if prep_name is None:
        seq_name = 'N-qubit tomography'
    else:
        seq_name = prep_name + ' tomography'
    seq = sequence.Sequence(seq_name)
    seg_list = []

    if prep_sequence is None:
        prep_sequence = ['Y90 ' + qubit_names[0]]

    # tomography elements
    tomography_sequences = get_tomography_pulses(*qubit_names,
                                                 basis_pulses=rots_basis)
    for i, tomography_sequence in enumerate(tomography_sequences):
        pulse_list = [operation_dict[pulse] for pulse in prep_sequence]
        # tomography_sequence.append('RO mux')
        # if preselection:
        #     tomography_sequence.append('RO mux_presel')
        #     tomography_sequence.append('RO presel_dummy')
        pulse_list.extend([operation_dict[pulse] for pulse in
                           tomography_sequence])
        ro_pulses = generate_mux_ro_pulse_list(qubit_names, operation_dict)
        pulse_list.extend(ro_pulses)

        if preselection:
            ro_pulses_presel = generate_mux_ro_pulse_list(
                qubit_names, operation_dict, 'RO_presel', 'start', -ro_spacing)
            pulse_list.extend(ro_pulses_presel)
        seg = segment.Segment('tomography_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def get_tomography_pulses(*qubit_names, basis_pulses=('I', 'X180', 'Y90',
                                                      'mY90', 'X90', 'mX90')):
    tomo_sequences = [[]]
    for i, qb in enumerate(qubit_names):
        if i == 0:
            qb = ' ' + qb
        else:
            qb = 's ' + qb
        tomo_sequences_new = []
        for sequence in tomo_sequences:
            for pulse in basis_pulses:
                tomo_sequences_new.append(sequence + [pulse+qb])
        tomo_sequences = tomo_sequences_new
    return tomo_sequences


def get_decoupling_pulses(*qubit_names, nr_pulses=4):
    if nr_pulses % 2 != 0:
        logging.warning('Odd number of dynamical decoupling pulses')
    echo_sequences = []
    for pulse in nr_pulses*['X180']:
        for i, qb in enumerate(qubit_names):
            if i == 0:
                qb = ' ' + qb
            else:
                qb = 's ' + qb
            echo_sequences.append(pulse+qb)
    return echo_sequences


def n_qubit_ref_seq(qubit_names, operation_dict, ref_desc, upload=True,
                    return_seq=False, preselection=False, ro_spacing=1e-6):
    """
        Calibration points for arbitrary combinations

        Arguments:
            qubits: List of calibrated qubits for obtaining the pulse
                dictionaries.
            ref_desc: Description of the calibration sequence. Dictionary
                name of the state as key, and list of pulses names as values.
    """


    # create the elements
    seq_name = 'Calibration'
    seq = sequence.Sequence(seq_name)
    seg_list = []

    # calibration elements
    # calibration_sequences = []
    # for pulses in ref_desc:
    #     calibration_sequences.append(
    #         [pulse+' '+qb for qb, pulse in zip(qubit_names, pulses)])

    calibration_sequences = []
    for pulses in ref_desc:
        calibration_sequence_new = []
        for i, pulse in enumerate(pulses):
            if i == 0:
                qb = ' ' + qubit_names[i]
            else:
                qb = 's ' + qubit_names[i]
            calibration_sequence_new.append(pulse+qb)
        calibration_sequences.append(calibration_sequence_new)

    for i, calibration_sequence in enumerate(calibration_sequences):
        pulse_list = []
        pulse_list.extend(
            [operation_dict[pulse] for pulse in calibration_sequence])
        ro_pulses = generate_mux_ro_pulse_list(qubit_names, operation_dict)
        pulse_list.extend(ro_pulses)

        if preselection:
            ro_pulses_presel = generate_mux_ro_pulse_list(
                qubit_names, operation_dict, 'RO_presel', 'start', -ro_spacing)
            pulse_list.extend(ro_pulses_presel)
        seg = segment.Segment('calibration_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def n_qubit_ref_all_seq(qubit_names, operation_dict, upload=True,
                        return_seq=False, preselection=False, ro_spacing=1e-6):
    """
        Calibration points for all combinations
    """

    return n_qubit_ref_seq(qubit_names, operation_dict,
                           ref_desc=itertools.product(["I", "X180"],
                                                      repeat=len(qubit_names)),
                           upload=upload, return_seq=return_seq,
                           preselection=preselection, ro_spacing=ro_spacing)


def Ramsey_add_pulse_seq(times, measured_qubit_name,
                         pulsed_qubit_name, operation_dict,
                         artificial_detuning=None,
                         cal_points=True,
                         verbose=False,
                         upload=True, return_seq=False):
    raise NotImplementedError(
        'Ramsey_add_pulse_seq has not been '
        'converted to the latest waveform generation code and can not be used.')

    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_with_additional_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []


    pulse_pars_x1 = deepcopy(operation_dict['X90 ' + measured_qubit_name])
    pulse_pars_x1['refpoint'] = 'end'
    pulse_pars_x2 = deepcopy(pulse_pars_x1)
    pulse_pars_x2['refpoint'] = 'start'
    RO_pars = operation_dict['RO ' + measured_qubit_name]
    add_pulse_pars = deepcopy(operation_dict['X180 ' + pulsed_qubit_name])

    for i, tau in enumerate(times):
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [
                operation_dict['I ' + measured_qubit_name], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [
                operation_dict['X180 ' + measured_qubit_name], RO_pars])
        else:
            pulse_pars_x2['pulse_delay'] = tau
            if artificial_detuning is not None:
                Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
                pulse_pars_x2['phase'] = Dphase

            if i % 2 == 0:
                el = multi_pulse_elt(
                    i, station, [operation_dict['X90 ' + measured_qubit_name],
                                 pulse_pars_x2, RO_pars])
            else:
                el = multi_pulse_elt(i, station,
                                     [add_pulse_pars, pulse_pars_x1,
                                     # [pulse_pars_x1, add_pulse_pars,
                                      pulse_pars_x2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def ramsey_add_pulse_seq_active_reset(
        times, measured_qubit_name, pulsed_qubit_name,
        operation_dict, cal_points, n=1, artificial_detunings = 0,
        upload=True, for_ef=False, last_ge_pulse=False, prep_params=dict()):
    '''
     Azz sequence:  Ramsey on measured_qubit
                    pi-pulse on pulsed_qubit
     Input pars:
         times:           array of delays (s)
         n:               number of pulses (1 is conventional Ramsey)
     '''
    seq_name = 'Ramsey_with_additional_pulse_sequence'

    # Operations
    if for_ef:
        ramsey_ops_measured = ["X180"] + ["X90_ef"] * 2 * n
        ramsey_ops_pulsed = ["X180"]
        if last_ge_pulse:
            ramsey_ops_measured += ["X180"]
    else:
        ramsey_ops_measured = ["X90"] * 2 * n
        ramsey_ops_pulsed = ["X180"]

    ramsey_ops_measured += ["RO"]
    ramsey_ops_measured = add_suffix(ramsey_ops_measured, " " + measured_qubit_name)
    ramsey_ops_pulsed = add_suffix(ramsey_ops_pulsed, " " + pulsed_qubit_name)
    ramsey_ops_init = ramsey_ops_pulsed + ramsey_ops_measured
    ramsey_ops_det = ramsey_ops_measured

    # pulses
    ramsey_pulses_init = [deepcopy(operation_dict[op]) for op in ramsey_ops_init]
    ramsey_pulses_det = [deepcopy(operation_dict[op]) for op in ramsey_ops_det]

    # name and reference swept pulse
    for i in range(n):
        idx = -2 #(2 if for_ef else 1) + i * 2 + 1
        ramsey_pulses_init[idx]["name"] = f"Ramsey_x2_{i}"
        ramsey_pulses_init[idx]['ref_point'] = 'start'
        ramsey_pulses_det[idx]["name"] = f"Ramsey_x2_{i}"
        ramsey_pulses_det[idx]['ref_point'] = 'start'

    # compute dphase
    a_d = artificial_detunings if np.ndim(artificial_detunings) == 1 \
        else [artificial_detunings]
    dphase = [((t - times[0]) * a_d[i % len(a_d)] * 360) % 360
                 for i, t in enumerate(times)]

    # sweep pulses
    params = {f'Ramsey_x2_{i}.pulse_delay': times for i in range(n)}
    params.update({f'Ramsey_x2_{i}.phase': dphase for i in range(n)})
    swept_pulses_init = sweep_pulse_params(ramsey_pulses_init, params)
    swept_pulses_det = sweep_pulse_params(ramsey_pulses_det, params)
    swept_pulses = np.ravel((swept_pulses_init,swept_pulses_det), order='F')

    # add preparation pulses
    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict,
                                [pulsed_qubit_name, measured_qubit_name],
                                **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    # add calibration segments
    seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    # reuse sequencer memory by repeating readout pattern
    seq.repeat_ro(f"RO {measured_qubit_name}", operation_dict)

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def get_dd_pulse_list(operation_dict, qb_names, dd_time, nr_pulses=4, 
                      dd_scheme='cpmg', init_buffer=0, refpoint='end'):
    drag_pulse_length = (operation_dict['X180 ' + qb_names[-1]]['nr_sigma'] * \
                         operation_dict['X180 ' + qb_names[-1]]['sigma'])
    pulse_list = []
    def cpmg_delay(i, nr_pulses=nr_pulses):
        if i == 0 or i == nr_pulses:
            return 1/nr_pulses/2
        else:
            return 1/nr_pulses

    def udd_delay(i, nr_pulses=nr_pulses):
        delay = np.sin(0.5*np.pi*(i + 1)/(nr_pulses + 1))**2
        delay -= np.sin(0.5*np.pi*i/(nr_pulses + 1))**2
        return delay

    if dd_scheme == 'cpmg':
        delay_func = cpmg_delay
    elif dd_scheme == 'udd':
        delay_func = udd_delay
    else:
        raise ValueError('Unknown decoupling scheme "{}"'.format(dd_scheme))

    for i in range(nr_pulses+1):
        delay_pulse = {
            'pulse_type': 'SquarePulse',
            'channel': operation_dict['X180 ' + qb_names[0]]['I_channel'],
            'amplitude': 0.0,
            'length': dd_time*delay_func(i),
            'pulse_delay': 0}
        if i == 0:
            delay_pulse['pulse_delay'] = init_buffer
            delay_pulse['refpoint'] = refpoint
        if i == 0 or i == nr_pulses:
            delay_pulse['length'] -= drag_pulse_length/2
        else:
            delay_pulse['length'] -= drag_pulse_length
        if delay_pulse['length'] > 0:
            pulse_list.append(delay_pulse)
        else:
            raise Exception("Dynamical decoupling pulses don't fit in the "
                            "specified dynamical decoupling duration "
                            "{:.2f} ns".format(dd_time*1e9))
        if i != nr_pulses:
            for j, qbn in enumerate(qb_names):
                pulse_name = 'X180 ' if j == 0 else 'X180s '
                pulse_pulse = deepcopy(operation_dict[pulse_name + qbn])
                pulse_list.append(pulse_pulse)
    return pulse_list


def get_DD_pulse_list(operation_dict, qb_names, DD_time,
                      qb_names_DD=None,
                      pulse_dict_list=None, idx_DD_start=-1,
                      nr_echo_pulses=4, UDD_scheme=True):

    n = len(qb_names)
    if qb_names_DD is None:
        qb_names_DD = qb_names
    if pulse_dict_list is None:
        pulse_dict_list = []
    elif len(pulse_dict_list) < 2*n:
        raise ValueError('The pulse_dict_list must have at least two entries.')

    idx_DD_start *= n
    pulse_dict_list_end = pulse_dict_list[idx_DD_start::]
    pulse_dict_list = pulse_dict_list[0:idx_DD_start]

    X180_pulse_dict = operation_dict['X180 ' + qb_names_DD[0]]
    DRAG_length = X180_pulse_dict['nr_sigma']*X180_pulse_dict['sigma']
    X90_separation = DD_time - DRAG_length

    if UDD_scheme:
        pulse_positions_func = \
            lambda idx, N: np.sin(np.pi*idx/(2*N+2))**2
        pulse_delays_func = (lambda idx, N: X90_separation*(
                pulse_positions_func(idx, N) -
                pulse_positions_func(idx-1, N)) -
                ((0.5 if idx == 1 else 1)*DRAG_length))

        if nr_echo_pulses*DRAG_length > X90_separation:
            pass
        else:
            for p_nr in range(nr_echo_pulses):
                for qb_name in qb_names_DD:
                    if qb_name == qb_names_DD[0]:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180 ' + qb_name])
                        DD_pulse_dict['refpoint'] = 'end'
                        DD_pulse_dict['pulse_delay'] = pulse_delays_func(
                            p_nr+1, nr_echo_pulses)
                    else:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180s ' + qb_name])
                    pulse_dict_list.append(DD_pulse_dict)

            for j in range(n):
                if j == 0:
                    pulse_dict_list_end[j]['refpoint'] = 'end'
                    pulse_dict_list_end[j]['pulse_delay'] = pulse_delays_func(
                        1, nr_echo_pulses)
                else:
                    pulse_dict_list_end[j]['pulse_delay'] = 0
    else:
        echo_pulse_delay = (X90_separation -
                            nr_echo_pulses*DRAG_length) / \
                           nr_echo_pulses
        if echo_pulse_delay < 0:
            pass
        else:
            start_end_delay = echo_pulse_delay/2
            for p_nr in range(nr_echo_pulses):
                for qb_name in qb_names_DD:
                    if qb_name == qb_names_DD[0]:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180 ' + qb_name])
                        DD_pulse_dict['refpoint'] = 'end'
                        DD_pulse_dict['pulse_delay'] = \
                            (start_end_delay if p_nr == 0 else echo_pulse_delay)
                    else:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180s ' + qb_name])
                    pulse_dict_list.append(DD_pulse_dict)
            for j in range(n):
                if j == 0:
                    pulse_dict_list_end[j]['refpoint'] = 'end'
                    pulse_dict_list_end[j]['pulse_delay'] = start_end_delay
                else:
                    pulse_dict_list_end[j]['pulse_delay'] = 0

    pulse_dict_list += pulse_dict_list_end

    return pulse_dict_list


def generate_mux_ro_pulse_list(qubit_names, operation_dict, element_name='RO',
                               ref_point='end', pulse_delay=0.0):
    ro_pulses = []
    for j, qb_name in enumerate(qubit_names):
        ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
        ro_pulse['pulse_name'] = '{}_{}'.format(element_name, j)
        ro_pulse['element_name'] = element_name
        if j == 0:
            ro_pulse['pulse_delay'] = pulse_delay
            ro_pulse['ref_point'] = ref_point
        else:
            ro_pulse['ref_point'] = 'start'
        ro_pulses.append(ro_pulse)
    return ro_pulses


def interleaved_pulse_list_equatorial_seg(
        qubit_names, operation_dict, interleaved_pulse_list, phase, 
        pihalf_spacing=None, prep_params=None, segment_name='equatorial_segment'):
    prep_params = {} if prep_params is None else prep_params
    pulse_list = []
    for notfirst, qbn in enumerate(qubit_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbn])) 
        pulse_list[-1]['ref_point'] = 'start'
        if not notfirst:
            pulse_list[-1]['name'] = 'refpulse'
    pulse_list += interleaved_pulse_list
    for notfirst, qbn in enumerate(qubit_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbn])) 
        pulse_list[-1]['phase'] = phase
        if notfirst:
            pulse_list[-1]['ref_point'] = 'start'
        elif pihalf_spacing is not None:
            pulse_list[-1]['ref_pulse'] = 'refpulse'
            pulse_list[-1]['ref_point'] = 'start'
            pulse_list[-1]['pulse_delay'] = pihalf_spacing
    pulse_list += generate_mux_ro_pulse_list(qubit_names, operation_dict)
    pulse_list = add_preparation_pulses(pulse_list, operation_dict, qubit_names, **prep_params)
    return segment.Segment(segment_name, pulse_list)


def interleaved_pulse_list_list_equatorial_seq(
        qubit_names, operation_dict, interleaved_pulse_list_list, phases, 
        pihalf_spacing=None, prep_params=None, cal_points=None,
        sequence_name='equatorial_sequence', upload=True):
    prep_params = {} if prep_params is None else prep_params
    seq = sequence.Sequence(sequence_name)
    for i, interleaved_pulse_list in enumerate(interleaved_pulse_list_list):
        for j, phase in enumerate(phases):
            seg = interleaved_pulse_list_equatorial_seg(
                qubit_names, operation_dict, interleaved_pulse_list, phase,
                pihalf_spacing=pihalf_spacing, prep_params=prep_params, segment_name=f'segment_{i}_{j}')
            seq.add(seg)
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq, np.arange(seq.n_acq_elements())


def measurement_induced_dephasing_seq(
        measured_qubit_names, dephased_qubit_names, operation_dict, 
        ro_amp_scales, phases, pihalf_spacing=None, prep_params=None,
        cal_points=None, upload=True, sequence_name='measurement_induced_dephasing_seq'):
    interleaved_pulse_list_list = []
    for i, ro_amp_scale in enumerate(ro_amp_scales):
        interleaved_pulse_list = generate_mux_ro_pulse_list(
            measured_qubit_names, operation_dict, 
            element_name=f'interleaved_readout_{i}')
        for pulse in interleaved_pulse_list:
            pulse['amplitude'] *= ro_amp_scale
            pulse['operation_type'] = None
        interleaved_pulse_list_list.append(interleaved_pulse_list)
    return interleaved_pulse_list_list_equatorial_seq(
        dephased_qubit_names, operation_dict, interleaved_pulse_list_list, 
        phases, pihalf_spacing=pihalf_spacing, prep_params=prep_params,
        cal_points=cal_points, sequence_name=sequence_name, upload=upload)


def drive_cancellation_seq(
        drive_op_code, ramsey_qubit_names, operation_dict,
        sweep_points, n_pulses=1, pihalf_spacing=None, prep_params=None,
        cal_points=None, upload=True, sequence_name='drive_cancellation_seq'):
    """
    Sweep pulse cancellation parameters and measure Ramsey on qubits the
    cancellation is for.

    Args:
        drive_op_code: Operation code for the pulse to be cancelled, including
            the qubit name.
        ramsey_qubit_names: A list of qubit names corresponding to the
            undesired targets of the pulse that is being cancelled.
        sweep_points: A SweepPoints object that describes the pulse
            parameters to sweep. The sweep point keys should be of the form
            `qb.param`, where `qb` is the name of the qubit the cancellation
            is for and `param` is a parameter in the pulses
            cancellation_params dict. For example to sweep the amplitude of
            the cancellation pulse on qb1, you could configure the sweep
            points as `SweepPoints('qb1.amplitude', np.linspace(0, 1, 21))`.
            The Ramsey phases must be given in the second sweep dimension with
            sweep name 'phases'.
        n_pulses: Number of pulse repetitions done between the Ramsey
            pulses. Useful for amplification of small errors. Defaults to 1.
    Rest of the arguments are passed down to
        interleaved_pulse_list_list_equatorial_seq
    """

    len_sweep = len(list(sweep_points[0].values())[0][0])
    # create len_sweep instances of n pulses, where the n references correspond
    # to the same dictionary instance
    interleaved_pulse_list_list = \
        [n_pulses*[deepcopy(operation_dict[drive_op_code])]
         for _ in range(len_sweep)]
    for key, (values, unit, label) in sweep_points[0].items():
        assert len(values) == len_sweep
        tqb, param = key.split('.')
        iq = operation_dict[f'X180 {tqb}']['I_channel'], \
             operation_dict[f'X180 {tqb}']['Q_channel']
        for pulse_list, value in zip(interleaved_pulse_list_list, values):
            # since all n pulses in pulse_list are the same dict. we only need
            # to modify the first element.
            pulse_list[0]['cancellation_params'][iq][param] = value
    # make last segment a calibration segment
    interleaved_pulse_list_list[-1][0]['amplitude'] = 0

    return interleaved_pulse_list_list_equatorial_seq(
        ramsey_qubit_names, operation_dict, interleaved_pulse_list_list,
        sweep_points[1]['phase'][0], pihalf_spacing=pihalf_spacing,
        prep_params=prep_params, cal_points=cal_points,
        sequence_name=sequence_name, upload=upload)


def fluxline_crosstalk_seq(target_qubit_name, crosstalk_qubits_names,
                           crosstalk_qubits_amplitudes, sweep_points,
                           operation_dict, crosstalk_fluxpulse_length,
                           target_fluxpulse_length, prep_params,
                           cal_points, upload=True,
                           sequence_name='fluxline_crosstalk_seq'):
    """
    Applies a flux pulse on the target qubit with various amplitudes.
    Measure the phase shift due to these pulses on the crosstalk qubits which
    are measured in a Ramsey setting and fluxed to a more sensitive frequency.

    Args:
        target_qubit_name: the qubit to which a fluxpulse with varying amplitude
            is applied
        crosstalk_qubits_names: a list of qubits to do a Ramsey on.
        crosstalk_qubits_amplitudes: A dictionary from crosstalk qubit names
            to flux pulse amplitudes that are applied to them to increase their
            flux sensitivity. Missing amplitudes are set to 0.
        sweep_points: A SweepPoints object, where the first sweep dimension is
            over Ramsey phases and must be called 'phase' and the second sweep
            dimenstion is over the target qubit pulse amplitudes and must be
            called 'target_amp'.
        operation_dict: A dictionary of pulse dictionaries corresponding to the
            various operations that can be done.
        target_fluxpulse_length: length of the flux pulse on the target qubit.
        crosstalk_fluxpulse_length: length of the flux pulses on the crosstalk
            qubits
        prep_params: Perparation parameters dictionary specifying the type
            of state preparation.
        cal_points: CalibrationPoints object determining the used calibration
            points
        upload: Whether the experimental sequence should be uploaded.
            Defaults to True.
        sequence_name: Overwrite the sequence name. Defaults to
            'fluxline_crosstalk_seq'.
    """

    interleaved_pulse_list_list = []
    buffer_start = 0
    buffer_end = 0
    pi_len = 0
    for qbn in crosstalk_qubits_names:
        buffer_start = max(buffer_start,
                           operation_dict[f'FP {qbn}']['buffer_length_start'])
        buffer_end = max(buffer_end,
                           operation_dict[f'FP {qbn}']['buffer_length_end'])
        pi_len = max(pi_len, operation_dict[f'X180 {qbn}']['nr_sigma'] *
                             operation_dict[f'X180 {qbn}']['sigma'])

    for amp in sweep_points[1]['target_amp'][0]:
        interleaved_pulse_list = []
        for i, qbn in enumerate(crosstalk_qubits_names):
            pulse = deepcopy(operation_dict[f'FP {qbn}'])
            if i > 0:
                pulse['ref_point'] = 'middle'
                pulse['ref_point_new'] = 'middle'
            pulse['amplitude'] = crosstalk_qubits_amplitudes.get(qbn, 0)
            pulse['pulse_length'] = crosstalk_fluxpulse_length
            pulse['buffer_length_start'] = buffer_start
            pulse['buffer_length_end'] = buffer_end
            interleaved_pulse_list += [pulse]
        pulse = deepcopy(operation_dict[f'FP {target_qubit_name}'])
        pulse['amplitude'] = amp
        pulse['pulse_length'] = target_fluxpulse_length
        pulse['ref_point'] = 'middle'
        pulse['ref_point_new'] = 'middle'
        interleaved_pulse_list += [pulse]
        interleaved_pulse_list_list += [interleaved_pulse_list]

    pihalf_spacing = buffer_start + crosstalk_fluxpulse_length + buffer_end + \
        pi_len
    return interleaved_pulse_list_list_equatorial_seq(
        crosstalk_qubits_names, operation_dict, interleaved_pulse_list_list,
        sweep_points[0]['phase'][0], pihalf_spacing=pihalf_spacing,
        prep_params=prep_params, cal_points=cal_points,
        sequence_name=sequence_name, upload=upload)
