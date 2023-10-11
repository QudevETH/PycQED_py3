import logging
log = logging.getLogger(__name__)

import os
import time
import lmfit
import datetime
import traceback
import qutip as qt
import numpy as np
import scipy as sp
from copy import copy, deepcopy
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
from pycqed.analysis_v3 import *
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis import fitting_models as fit_mods

import pycqed.measurement.benchmarking.randomized_benchmarking as rb_meas


convert_to_mhz = lambda freq_angle, t: freq_angle/2/np.pi/t
convert_to_freq_angle = lambda freq_mhz, t: 2*np.pi*t*freq_mhz

# in the function "xentropy" below, pops0 and pops1 must be 1d arrays of
# size s**n, ie the basis state probabilities after one circuit
xentropy = lambda pops0, pops1, d: d*np.sum([p0*p1
                                          for p0, p1 in zip(pops0, pops1)]) - 1
xentropy_rhs = lambda pops_ideal, d: d*np.sum(
    [np.mean(pops_ideal[:, i]**2) for i in range(pops_ideal.shape[1])]) - 1
xentropy_lhs = lambda pops_meas, pops_ideal, d: np.mean(d*np.sum(
    np.array([pm*pi for pm, pi in zip(pops_meas, pops_ideal)]),
    axis=len(np.array(pops_meas).shape)-1) - 1)
xentropy_fidelity = lambda pops_meas, pops_ideal, d: \
    xentropy_lhs(pops_meas, pops_ideal, d) / xentropy_rhs(pops_ideal, d)


def crossEntropy(p, q):
    """
    Calculates the cross-entropy between p and q.

    Args:
        p (np.array): qubit state probabilities, typically the measured ones
        q (np.array): qubit state probabilities, typically the calculated ones

    Returns:
        the cross-entropy
    """
    # if any of the entries in q are 0, we need to set p*np.log(q) = 0
    idxs = np.where(q == 0)[0]
    prod = p*np.log(q)
    if len(idxs) > 0:
        prod[idxs] = 0
    return - np.sum(prod)
entropy = lambda p: crossEntropy(p, p)
def crossEntropyFidelity(pops_meas, pops_ideal, d):
    """
    Calculate cross entropy fidelity as described in this paper:
    https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.125.120504/supp.pdf

    Args:
        pops_meas (array): measured qubit populations for a fixed XEB sequence
            length. Should have shape (nr_xeb_seqs, n), where n is the number of
            states. The columns should correspond to the order [pg, pe] for
            one qubit, and [pgg, pge, peg, pee] for two qubits.
        pops_ideal (array): calculated qubit populations for a fixed XEB
            sequence length. Should have shape (nr_xeb_seqs, n), where n is the
            number of states. The columns should correspond to the order
            [pg, pe] for one qubit, and [pgg, pge, peg, pee] for two qubits.
        d (int): dimension of the Hilbert space (2**n for n qubits)

    Returns:
        the cross-entropy fidelity between pops_meas and pops_ideal
    """
    pm = deepcopy(pops_meas)
    pi = deepcopy(pops_ideal)
    pops_incoh = 1/d/len(pi)
    pm /= len(pm)
    pm = pm.flatten()
    pi /= len(pi)
    pi = pi.flatten()
    numerator = crossEntropy(pops_incoh, pi) - \
                crossEntropy(pm, pi)
    denominator = crossEntropy(pops_incoh, pi) - entropy(pi)
    return numerator/denominator



### Single qubit XEB ###

sqrtXqt = qt.qip.operations.gates.rx(np.pi/2, N=None, target=0)
sqrtYqt = qt.qip.operations.gates.ry(np.pi/2, N=None, target=0)
Tqt = qt.qip.operations.gates.rz(np.pi/4, N=None, target=0)
sqrtX = sqrtXqt.full()
sqrtY = sqrtYqt.full()
T = Tqt.full()

gqt, eqt = qt.states.basis(2, 0), qt.states.basis(2, 1)
g, e = gqt.full(), eqt.full()
M0 = np.outer(g, g)
M1 = np.outer(e, e)
povms1 = [M0, M1]


## Models
U1 = lambda da, df, t: sp.linalg.expm(
    -1j*((2*np.pi*df)*e*e.conj().T +
         (np.pi/2/t + da/t)*qt.sigmay()/2).full()*t)


def U1_qscale(da, df, dqs, dphixy, t):
    phi_xy = np.pi/2 + dphixy
    sigma_phi = np.exp(-1j*phi_xy)*g*e.conj().T + np.exp(1j*phi_xy)*e*g.conj().T
    H = (2*np.pi*df)*e*e.conj().T + (1 + dqs)*(np.pi/2/t + da/t)*sigma_phi/2
    return sp.linalg.expm(-1j*H*t)


def U2(phid1, phid2, delta_phic, #phis, phisa,
       t):
    delta1 = phid1/2/np.pi/t
    delta2 = phid2/2/np.pi/t
    # Js = phis/2/t
    Jc = (np.pi + delta_phic)/t
    # phisa = - phisa - np.pi
    eye = np.eye(2)
    H = 2*np.pi*delta1*np.kron(e @ e.conj().T, eye) + \
        2*np.pi*delta2*np.kron(eye, e @ e.conj().T) + \
        Jc*np.kron(e, e)@np.kron(e, e).conj().T
        # Js*(np.exp(1j*phisa/2)*np.kron(g, e)@np.kron(e, g).conj().T +
        #     np.exp(-1j*phisa/2)*np.kron(e, g)@np.kron(g, e).conj().T) + \
        # Jc*np.kron(e, e)@np.kron(e, e).conj().T
    return sp.linalg.expm(-1j*H*t)

## Simulation ##

def time_evolve_prop(H, t_gate, T1, T2s):
    c_ops = [np.sqrt(1/T1)*qt.destroy(2),
             np.sqrt((1/T2s - 1/(2*T1))/2)*qt.sigmaz()]
    tlist = np.linspace(0, t_gate, 100)
    output = qt.propagator(H, tlist, c_ops)
    return output[-1]


def time_evolve_gate_prop(angle, T1, T2s, t_gate, delta_ang=0.0, delta_f=0.0):
    amp = angle/t_gate + delta_ang/t_gate
    H =  amp * qt.sigmay()/2 + 2*np.pi*delta_f*eqt*eqt.dag()
    U = time_evolve_prop(H, t_gate, T1=T1, T2s=T2s)
    return U


def simulate_circuits_1qb(nr_cycles, nr_seq, rots_qt, T1, T2,
                          t_gate, delta_ang=0, delta_f=0, init_rotation=None):
    t0 = time.time()
    if init_rotation is None:
        init_rotation = sqrtYqt
    # get pops_sim
    pops_sim = np.zeros((nr_seq, 2))
    for i in range(nr_seq):
        rho = init_rotation*qt.thermal_dm(2, 0)*init_rotation.dag()
        Uy = time_evolve_gate_prop(np.pi/2, T1, T2, t_gate=t_gate,
                                   delta_ang=delta_ang/2, delta_f=delta_f)
        for c in range(nr_cycles):
            rho = qt.vector_to_operator(Uy*qt.operator_to_vector(rho))
            rot = rots_qt[i][c]
            rho = rot*rho*rot.dag()
        pops_sim[i, :] = [np.real(rho[0, 0]), 1 - np.real(rho[0, 0])]
    print(f'Time for sim: {time.time() - t0} s.')
    return pops_sim


## Calculation ##
def calculate_ideal_circuit_1qb(nr_cycles, nr_seq, rots,
                                Uvar=None, init_state=None):
    """
    Calculate circuits assuming ideal gate.
    :param nr_cycles (int): number of XEB cycles in circuit
    :param nr_seq (int): number of random samplings of the circuit
    :param rots (list of lists): of the form
        [[qt.operations.rotation(qt.sigmaz(), z_angle).full()
         for z_angle in z_angles[i]] for i in range(nr_seq)]
    :param init_state: length 2 numpy array for initial rotation
    :return: ideal populations
    """

    if init_state is None:
        init_state = g
    if Uvar is None:
        Uvar = sqrtY
    if nr_cycles == 0:
        pops_ideal = np.reshape(np.tile((np.abs(init_state)**2).T[0], nr_seq),
                                (nr_seq, 2))
    else:
        pops_ideal = np.zeros((nr_seq, 2))
        for i in range(nr_seq):
            Us = []
            for c in range(nr_cycles):
                Us.extend([rots[i][nr_cycles-c-1], Uvar])
            pops_ideal[i, :] = (np.abs(np.matmul(
                reduce(np.matmul, Us), init_state))**2).T[0]
    return pops_ideal


# ## Cost function
# # psi = (g+e)/np.sqrt(2)
# psi = (sqrtX*gqt).full()
# def avg_loss(ctrl_errs, grad):
#     """
#     ctrl_errs = [ampl_det_angle, freq_det_angle]
#     U_from_det(ampl_det_angle, freq_det_Mhz)
#     """
#     ampl_angle = ctrl_errs[0]
#     freq_det = ctrl_errs[1]/2/np.pi/t
#     Uvar = U1(ampl_angle, freq_det, t)
#     cost_values = np.zeros(nr_seq)
#     for i in range(nr_seq):
#         Us = []
#         for c in range(nr_cycles):
#             Us.append(z_rots[i][nr_cycles-c-1])
#             Us.append(Uvar)
#         cost_values[i] = xentropy(
#             pops_sim[i],
#             (np.abs(np.matmul(reduce(np.matmul, Us), psi))**2).T[0], d)
#     return -np.mean(cost_values)

# cost function - single qubit
def avg_loss(ctrl_errs, grad):
    """
    ctrl_errs = [ampl_det_angle, freq_det_angle]
    U_from_det(ampl_det_angle, freq_det_Mhz)
    """
    ampl_angle = ctrl_errs[0]
    freq_det = xeb_ana.convert_to_mhz(ctrl_errs[1], t)
    Uvar = xeb_ana.U1(ampl_angle, freq_det, t)
    cost_values = np.zeros(nr_seq)
    for i in range(nr_seq):
        pops_ideal = xeb_ana.calculate_ideal_circuit_1qb(
            nr_cycles, 1, [z_rots[i]], t, Uvar, init_state)[0]
        #         print()
        #         print(pops_sim[i])
        #         print(pops_ideal)
        cost_values[i] = xeb_ana.xentropy(pops_sim[i], pops_ideal, d)
    return -np.mean(cost_values)


# cost function
def avg_loss_fidelity(ctrl_errs, grad):
    """
    ctrl_errs = [ampl_det_angle, freq_det_angle]
    U_from_det(ampl_det_angle, freq_det_Mhz)
    """
    ampl_angle = ctrl_errs[0]
    freq_det = xeb_ana.convert_to_mhz(ctrl_errs[1], t)
    Uvar = xeb_ana.U1(ampl_angle, freq_det, t)
    pops_ideal = xeb_ana.calculate_ideal_circuit_1qb(
        nr_cycles, nr_seq, z_rots, t, Uvar, init_state)
    #     print()
    #     print('pops_sim', pops_sim)
    #     print('pops_ideal', pops_ideal)
    cost_values = xeb_ana.xentropy_fidelity(pops_sim, pops_ideal, 2)
    return 1-np.mean(cost_values)


# cost function - two qubit
def avg_loss(angles, grad):
    """
    angles = [phid1, phid2, delta_phic, phis, phisa]
    U_from_det(ampl_det_angle, freq_det_Mhz)
    """
    Uvar = xeb_ana.U2(*angles, t)
    cost_values = np.zeros(nr_seq)
    for i in range(nr_seq):
        pops_ideal = xeb_ana.calculate_ideal_circuit_2qb(
            nr_cycles, 1, [circuits_list[i]], Uvar)[0]
        #         print()
        #         print(pops_sim[i])
        #         print(pops_ideal)
        cost_values[i] = xeb_ana.xentropy(pops_sim[i], pops_ideal, d)
    return -np.mean(cost_values)


### Analysis functions ###
def single_qubit_xeb_analysis(timestamp=None, classifier_params=None,
                              meas_obj_names=None, state_prob_mtxs=None,
                              correct_readout=(True,),
                              sweep_type=None, save_processed_data=True,
                              probability_states=None, save_figures=True,
                              raw_keys_in=None, save=True, save_filename=None,
                              renormalize=True, shots_selection_map=None,
                              meas_data_dtype=None, **params):

    # (nr_seq, len(cycles), nr_cycles)
    # list with nr_seq lists; each sublist has length len(cycles)
    # and contains nr_cyles for nr_cycles in cycles
    # z_angles = sp.get_sweep_params_property('values', 1)
    # len(z_angles) == nr_seq
    # len(z_angles[i]) == len(cycles)
    # [len(z_angles[i][j]) == nr_cycles for nr_cycles in cycles]
    pp = pp_mod.ProcessingPipeline(add_param_method='replace')
    try:
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param_from_metadata_group(timestamp,
                                                                   'meas_objs')
        print(meas_obj_names)
        if raw_keys_in is None:
            raw_keys_in = {mobjn: 'raw' for mobjn in meas_obj_names}
        data_dict = dat_extr_mod.extract_data_hdf(
            timestamps=timestamp, meas_data_dtype=meas_data_dtype)
        swpts, movnm = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'movnm'])
        if sweep_type is None:
            sweep_type = {'cycles': 0, 'seqs': 1}

        cycles = swpts.get_sweep_params_property('values', 0)
        nr_seq = swpts.length(1)
        compression_factor = hlp_mod.get_param('compression_factor', data_dict,
                                               **params)
        n_shots = hlp_mod.get_instr_param_from_hdf_file(
            meas_obj_names[0], 'acq_shots', timestamp)
        prep_params = hlp_mod.get_param_from_metadata_group(
            timestamp, 'preparation_params')
        reset_reps = prep_params['reset_reps'] if 'reset' in prep_params[
            'preparation_type'] else 0

        nr_swpts0 = swpts.length(0)
        # nr_swpts1 = swpts.length(1)
        data_size = len(data_dict[meas_obj_names[0]][
                            list(data_dict[meas_obj_names[0]])[0]])
        nr_swpts1 = data_size // n_shots // nr_swpts0 // (reset_reps+1)
        n_segments = nr_swpts0 * compression_factor
        n_sequences = nr_swpts1 // compression_factor
        print(f'{nr_swpts0} segments, {compression_factor} hard sequences, '
              f'{n_sequences} soft sequences, {n_shots} shots')

        classifier_params = hlp_mod.get_clf_params_from_hdf_file(
            timestamp, meas_obj_names, classifier_params)
        state_prob_mtxs = hlp_mod.get_state_prob_mtxs_from_hdf_file(
            timestamp, meas_obj_names, state_prob_mtxs)
        for mobjn, mtx in state_prob_mtxs.items():
            if mtx is None:
                if any(correct_readout):
                    log.warning(f'The acq_state_prob_mtx was not provided '
                                f'for {mobjn}. The acq_state_prob_mtxs '
                                f'must be specified for both qubits in '
                                f'order to perform readout correction.')
                if False in correct_readout:
                    # only do the readout-uncorrected analysis if the user
                    # wanted this originally
                    correct_readout = (False,)
                else:
                    # no analysis to run
                    return pp, None, None, None

        # get probability_states
        if probability_states is None:
            probability_states = ['pg', 'pe']
            if len(classifier_params[meas_obj_names[0]]['weights_']) == 3:
                probability_states += ['pf']
        print('probability_states: ', probability_states)

        pp.add_node('extract_data_hdf', timestamps=timestamp)
        for mobjn in meas_obj_names:
            pp.add_node('filter_data', keys_in=raw_keys_in[mobjn],
                        data_filter=lambda x: x[reset_reps::reset_reps+1],
                        meas_obj_names=mobjn)
            pp.add_node('classify_gm',
                        keys_in=[f'{mobjn}.filter_data {movn}' for movn in
                                 movnm[mobjn]],  # keys set by filter_data
                        keys_out=[f'{mobjn}.classify_gm.{ps}'
                                  for ps in probability_states],
                        clf_params=classifier_params.get(mobjn, None),
                        meas_obj_names=mobjn)
            pp.add_node('average_data',
                        shape=(n_sequences, n_shots, n_segments),
                        final_shape=(n_sequences*n_segments),
                        averaging_axis=1,
                        selection_map=shots_selection_map,
                        keys_in='previous',
                        keys_out=[f'{mobjn}.average_data.{ps}'
                                  for ps in probability_states],
                        meas_obj_names=mobjn)
            if any(correct_readout):
                pp.add_node('correct_readout',
                            keys_in='previous',
                            state_prob_mtxs=state_prob_mtxs,
                            keys_out=[f'{mobjn}.correct_readout.{ps}'
                                      for ps in probability_states],
                            meas_obj_names=mobjn)

        pp.resolve(movnm)
        data_dict = {'dim_hilbert': 2,
                     'sg_qb_gate_lengths': get_sg_qb_gate_lengths(timestamp),
                     'sweep_type': sweep_type,
                     'meas_obj_names': meas_obj_names,
                     'renormalize': renormalize}
        pp(data_dict)
        if len(pp) and save:
            pp.save(save_processed_data=save_processed_data,
                    save_figures=save_figures, filename=save_filename)
        return pp, meas_obj_names, cycles, nr_seq
    except Exception:
        traceback.print_exc()
        return pp, None, None, None


def calculate_fidelities_purities_1qb(data_dict, data_key='correct_readout',
                                      init_rotation=None, renormalize=True,
                                      amp_sc_intlvd_gate=None, nr_seq=None,
                                      **params):

    d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True, **params)
    print('d', d)
    sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                   default_value={'cycles': 0, 'seqs': 1},
                                   **params)

    if init_rotation is None:
        init_rotation = hlp_mod.get_param('init_rotation', data_dict)
    if init_rotation is not None:
        init_state = standard_pulses[init_rotation] @ g
    else:
        init_state = None

    Uinterleaved = None
    if amp_sc_intlvd_gate is not None:
        Uinterleaved = qt.qip.operations.gates.rx(amp_sc_intlvd_gate*np.pi,
                                                  N=None, target=0).full()
    circuit_calc_func = hlp_mod.get_param('circuit_calc_func', data_dict,
                                          **params)
    if isinstance(circuit_calc_func, str):
        circuit_calc_func = eval(circuit_calc_func)
    if circuit_calc_func is None:
        circuit_calc_func = calculate_ideal_circuit_1qb

    meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(
            data_dict['timestamps'][0], 'meas_objs')
    print(meas_obj_names)
    swpts, mospm = hlp_mod.get_measurement_properties(data_dict,
                                               props_to_extract=['sp', 'mospm'])
    cycles = swpts.get_sweep_params_property('values', sweep_type['cycles'])
    if nr_seq is None:
        nr_seq = swpts.length(sweep_type['seqs'])

    apm = hlp_mod.get_param('add_param_method', data_dict, **params)
    if apm is None:
        params['add_param_method'] = 'replace'

    for mobjn in meas_obj_names:
        pops_meas_all = data_dict[mobjn][data_key]['pe']
        if hlp_mod.get_param('renormalize', data_dict,
                             default_value=renormalize):
            pops_meas_all /= (pops_meas_all + data_dict[mobjn][data_key]['pg'])
        # add pg = 1-pe
        pops_meas_all = np.concatenate([(1-pops_meas_all)[np.newaxis].T,
                                        pops_meas_all[np.newaxis].T], axis=1)
        pops_meas_all = pops_meas_all.reshape((nr_seq, swpts.length(0), 2))
        hlp_mod.add_param(f'{mobjn}.xeb_probabilities', pops_meas_all,
                          data_dict, **params)

        fidelities = np.zeros(len(cycles))
        purities = np.zeros(len(cycles))
        pops_ideal_all = np.zeros_like(pops_meas_all)
        for ii, nr_cycles in enumerate(cycles):
            index = np.where(np.array(cycles) == nr_cycles)[0][0]
            pops_meas = pops_meas_all[:, index, :]

            z_angles = swpts.get_sweep_params_property('values',
                                                       sweep_type['seqs'],
                                                       mospm[mobjn][1])
            z_angles = np.array([z_angles[i][index]
                                 for i in range(nr_seq)])*np.pi/180
            print(z_angles.shape)
            z_rots = get_z_rotations(nr_cycles, nr_seq, z_angles,
                                     qutip_type=False)

            # calculate assuming ideal gate
            pops_ideal = circuit_calc_func(
                nr_cycles, nr_seq, z_rots,
                Uvar=Uinterleaved, init_state=init_state)
            pops_ideal_all[:, index, :] = pops_ideal
            fidelities[ii] = crossEntropyFidelity(pops_meas, pops_ideal, d)
            purities[ii] = sqrt_purity(pops_meas, d)

        hlp_mod.add_param(f'{mobjn}.xeb_probabilities_ideal', pops_ideal_all,
                          data_dict, **params)
        hlp_mod.add_param(f'{mobjn}.fidelities', fidelities, data_dict, **params)
        # FIXME THESE ARE SQRT PURITIES !!!
        hlp_mod.add_param(f'{mobjn}.purities', purities, data_dict, **params)


def get_z_rotations(nr_cycles, nr_seq, z_angles=None, seed=None,
                    qutip_type=True):
    if z_angles is None:
        rng_seed = np.random.RandomState(seed)
        z_angles = rng_seed.uniform(0, 2, (nr_seq, nr_cycles))*np.pi
    print(z_angles.shape)
    rots = [''] * nr_seq
    if qutip_type:
        rots_qt = [''] * nr_seq
    for i in range(nr_seq):
        rots_cycle = [''] * nr_cycles
        rots_cycle_qt = [''] * nr_cycles
        for j, a in enumerate(z_angles[i]):
            z_rot = qt.operations.rotation(qt.sigmaz(), a)
            rots_cycle[j] = z_rot.full()
            if qutip_type:
                rots_cycle_qt[j] = z_rot
        rots[i] = rots_cycle
        if qutip_type:
            rots_qt[i] = rots_cycle_qt
    if qutip_type:
        return rots, rots_qt
    else:
        return rots


def get_sg_qb_gate_lengths(timestamp):
    meas_obj_names = hlp_mod.get_param_from_metadata_group(timestamp,
                                                           'meas_objs')
    params_dict = {f'{qbn}_ge_sigma': f'Instrument settings.{qbn}.ge_sigma'
                   for qbn in meas_obj_names}
    params_dict.update({f'{qbn}_ge_nr_sigma':
                            f'Instrument settings.{qbn}.ge_nr_sigma'
                        for qbn in meas_obj_names})
    dd = hlp_mod.get_params_from_hdf_file({}, params_dict,
                                          folder=a_tools.get_folder(timestamp))
    gate_lengths = {qbn: dd[f'{qbn}_ge_sigma']*dd[f'{qbn}_ge_nr_sigma']
                    for qbn in meas_obj_names}
    return gate_lengths


def get_two_qb_gate_length(timestamp, dev_name, gate_name='CZ_nztc',
                           meas_obj_names=None):
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(timestamp,
                                                               'meas_objs')
    assert len(meas_obj_names) == 2
    params_dict = {f'pulse_length':
                       f'Instrument settings.{dev_name}.{gate_name}_'
                       f'{meas_obj_names[0]}_{meas_obj_names[1]}_pulse_length'}
    dd = hlp_mod.get_params_from_hdf_file({}, params_dict,
                                          folder=a_tools.get_folder(timestamp))
    if dd['pulse_length'] == 0:
        params_dict = {f'pulse_length':
                           f'Instrument settings.{dev_name}.{gate_name}_'
                           f'{meas_obj_names[1]}_{meas_obj_names[0]}_pulse_length'}
        dd = hlp_mod.get_params_from_hdf_file(
            {}, params_dict, folder=a_tools.get_folder(timestamp))
    return dd['pulse_length']


### Two-qubit XEB ###

sqrtXqt_2qbs = qt.tensor(sqrtXqt, sqrtXqt)
sqrtYqt_2qbs = qt.tensor(sqrtYqt, sqrtYqt)
Tqt_2qbs = qt.tensor(Tqt, Tqt)
sqrtX_2qbs, sqrtY_2qbs = sqrtXqt_2qbs.full(), sqrtYqt_2qbs.full()
T_2qbs = Tqt_2qbs.full()
czqt = qt.operations.cphase(np.pi)
cz = czqt.full()

gg_qt, ge_qt, eg_qt, ee_qt = [qt.states.basis(4, i).full() for i in range(4)]
gg, ge, eg, ee = [qt.states.basis(4, i).full() for i in range(4)]
M00, M01, M10, M11 = [np.outer(qt.states.basis(4, i).full(),
                               qt.states.basis(4, i).full()) for i in range(4)]
povms2 = [M00, M01, M10, M11]


## Simulation @@
def time_evolve_prop_2qbs(H, t_gate, T1_list, T2s_list, nr_qubits=2):
    c_ops = []
    for i in range(nr_qubits):
        nqb_state = [qt.identity(2) for j in range(nr_qubits)]
        nqb_state[i] = qt.destroy(2)
        c_ops += [np.sqrt(1/T1_list[i])*qt.tensor(nqb_state)]
    for i in range(nr_qubits):
        nqb_state = [qt.identity(2) for j in range(nr_qubits)]
        nqb_state[i] = qt.sigmaz()
        c_ops += [np.sqrt((1/T2s_list[i] - 1/(2*T1_list[i]))/2)*qt.tensor(nqb_state)]
    tlist = np.linspace(0, t_gate, 100)
    #     c_ops = []
    output = qt.propagator(H, tlist, c_ops)
    return output[-1]


def time_evolve_gate_prop_2qbs(T1, T2, t_gate, phid1=0, phid2=0,
                               delta_phic=0, phis=0, phisa=0, nr_qubits=2):
    delta1 = phid1/2/np.pi/t_gate
    delta2 = phid2/2/np.pi/t_gate
    Js = phis/2/t_gate
    Jc = (np.pi + delta_phic)/t_gate
    phisa = - phisa - np.pi
    eye = np.eye(2)
    H = qt.Qobj(2*np.pi*delta1*np.kron(e @ e.conj().T, eye) +
                2*np.pi*delta2*np.kron(eye, e @ e.conj().T) +
                # Js*(np.exp(1j*phisa/2)*np.kron(g, e)@np.kron(e, g).conj().T +
                #     np.exp(-1j*phisa/2)*np.kron(e, g)@np.kron(g, e).conj().T) +
                Jc*np.kron(e, e)@np.kron(e, e).conj().T)
    H.dims = [[2, 2], [2, 2]]
    U = time_evolve_prop_2qbs(H, t_gate, T1, T2, nr_qubits)
    return U


def simulate_circuits_2qbs(nr_cycles, nr_seq, circuits_list, T1, T2, t_gate,
                           phid1=0, phid2=0, delta_phic=0, phis=0, phisa=0):
    t0 = time.time()
    # get pops_sim
    Ucz = time_evolve_gate_prop_2qbs(T1, T2, t_gate, phid1=phid1, phid2=phid2,
                                     delta_phic=delta_phic, phis=phis,
                                     phisa=phisa)
    pops_sim = np.zeros((nr_seq, 4))
    rho0 = sqrtYqt_2qbs*qt.tensor([qt.thermal_dm(2, 0)] * 2)*sqrtYqt_2qbs.dag()
    # rho0 = Tqt_2qbs*rho0*Tqt_2qbs.dag()
    # rho0 = qt.vector_to_operator(Ucz*qt.operator_to_vector(rho0))
    for i in range(nr_seq):
        rho = rho0
        # gates = [g for g in circuits_list[i] if len(g) == 2 + (nr_cycles+1)*3][0][5:]
        gates = [g for g in circuits_list[i] if len(g) == 2 + nr_cycles*3][0][2:]
        gates = np.reshape(gates, (nr_cycles, 3))
        for c in range(nr_cycles):
            sgqb_gates = qt.tensor(standard_pulses_qt[gates[c][0][:3]],
                                   standard_pulses_qt[gates[c][1][:3]])
            rho = sgqb_gates * rho * sgqb_gates.dag()
            rho = qt.vector_to_operator(Ucz*qt.operator_to_vector(rho))
        pops_sim[i, :] = [np.real(rho[j, j]) for j in range(4)]
    print(f'Time for sim: {time.time() - t0} s.')
    return pops_sim

# def simulate_circuits_2qbs(nr_cycles, nr_seq, z_rots, T1, T2, t_gate,
#                            phid1=0, phid2=0, delta_phic=0, phis=0, phisa=0):
#     # circuits with VZ gates
#     t0 = time.time()
#     # get pops_sim
#     Ucz = time_evolve_gate_prop_2qbs(T1, T2, t_gate, phid1=phid1, phid2=phid2,
#                                      delta_phic=delta_phic, phis=phis,
#                                      phisa=phisa)
#     pops_sim = np.zeros((nr_seq, 4))
#     rho0 = sqrtY_2qbs*qt.tensor([qt.thermal_dm(2, 0)] * 2)*sqrtY_2qbs.dag()
#     for i in range(nr_seq):
#         rho = rho0
#         for c in range(nr_cycles):
#             #             rho = sg_qb_unitaries[i][c]*rho*sg_qb_unitaries[i][c].dag()
#             #             rho = qt.vector_to_operator(Ucz*qt.operator_to_vector(rho))
#             rho = sqrtY_2qbs * rho * sqrtY_2qbs.dag()
#             rho = qt.Qobj(z_rots[i][c], dims=[[2, 2], [2, 2]]) * rho * qt.Qobj(z_rots[i][c], dims=[[2, 2], [2, 2]]).dag()
#             rho = qt.vector_to_operator(Ucz*qt.operator_to_vector(rho))
#         pops_sim[i, :] = [np.real(rho[j, j]) for j in range(4)]
#     print(f'Time for sim: {time.time() - t0} s.')
#     return pops_sim


## Calculation ##
def translate(info):
    s_gates = ["RX", "RY", "RZ"]
    if info[0][0] == 'Y':
        gate_name = s_gates[1]
        angle = np.pi / 2
    elif info[0][0] == 'X':
        gate_name = s_gates[0]
        angle = np.pi / 2
    elif info[0][0] == 'Z':
        gate_name = s_gates[2]
        angle = np.pi / 4
    else:  # C-phase gate
        gate_name = 'CPHASE'
        angle = 180 if info[0][2:]=='' else float(info[0][2:])
        angle = -angle*np.pi/180
    if int(info[1][3]) == 1:
        qubit = 0
    else:
        qubit = 1
    return gate_name, qubit, angle


def construct_from_op(op_lis):
    q = qt.qip.circuit.QubitCircuit(2, reverse_states=False)
    for op in op_lis:
        op_info = op.split(" ")
        info = translate(op_info)
        if len(op_info) == 2:
            q.add_gate(info[0], info[1], None, info[2], r"\pi/4")
        else:
            q.add_gate(info[0], 0, 1, info[2])
    return q


def transfer(data_dict, **params):
    sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                   default_value={'cycles': 0, 'seqs': 1},
                                   **params)
    meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(
            data_dict['timestamps'][0], 'meas_objs')
    swpts, mospm = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'mospm'])
    gates_list = swpts.get_sweep_params_property('values',
                                                 sweep_type['seqs'],
                                                 mospm[meas_obj_names[0]][1])
    lis = []
    for circuit_lis in gates_list:
        for count, circuit in enumerate(circuit_lis):
            q = construct_from_op(circuit)
            lis += [q]
    return lis


def manual_propagator(gate):
    """
    Efficiently computes the propagator for a parameterised quantum gate using
    analytic expressions.

    Calling this method for each gate in a qutip circuit is equivalent to
    calling circuit.propagators(). The latter is more generic, since it can
    deal with arbitrary gates by exponentiating the underlying Hamiltonians.
    The former is however approximately 2 orders of magnitude faster, because
    it is so simple, and could probably still be further optimised if needed.
    """
    m = np.zeros([4,4], 'complex128')
    if gate.name == 'RZ':
        e = np.exp(-1j*gate.arg_value/2)
        ec = np.exp(1j*gate.arg_value/2)
        if gate.targets[0] == 0:
            m[0,0] = e
            m[1,1] = e
            m[2,2] = ec
            m[3,3] = ec
        if gate.targets[0] == 1:
            m[0,0] = e
            m[1,1] = ec
            m[2,2] = e
            m[3,3] = ec
    elif gate.name == 'RY':
        cos = np.cos(gate.arg_value/2)
        sin = np.sin(gate.arg_value/2)
        if gate.targets[0] == 0:
            m[0,0] = cos
            m[1,1] = cos
            m[0,2] = -sin
            m[1,3] = -sin
            m[2,0] = sin
            m[3,1] = sin
            m[2,2] = cos
            m[3,3] = cos
        if gate.targets[0] == 1:
            m[0,0] = cos
            m[0,1] = -sin
            m[1,0] = sin
            m[1,1] = cos
            m[2,2] = cos
            m[2,3] = -sin
            m[3,2] = sin
            m[3,3] = cos
    elif gate.name == 'RX':
        cos = np.cos(gate.arg_value/2)
        misin = -1j*np.sin(gate.arg_value/2)
        if gate.targets[0] == 0:
            m[0,0] = cos
            m[1,1] = cos
            m[0,2] = misin
            m[1,3] = misin
            m[2,0] = misin
            m[3,1] = misin
            m[2,2] = cos
            m[3,3] = cos
        if gate.targets[0] == 1:
            m[0,0] = cos
            m[0,1] = misin
            m[1,0] = misin
            m[1,1] = cos
            m[2,2] = cos
            m[2,3] = misin
            m[3,2] = misin
            m[3,3] = cos
    elif gate.name == 'CPHASE':
        m[0,0] = 1
        m[1,1] = 1
        m[2,2] = 1
        m[3,3] = np.exp(1j*gate.arg_value)  # qutip convention: positive sign
    else:
        raise ValueError(f"Gate {gate.name} not implemented!")
    return m


def proba(qc):
    """
    Computes the output states probabilities for a qutip quantum circuit

    TODO: one can finish removing calls to qutip and make this method even
     faster if needed.
    """
    # Equivalent (only for basic gates) to
    # U = qt.qip.operations.gate_sequence_product(qc.propagators())
    M = np.eye(4)
    for g in qc.gates:
        M = np.matmul(manual_propagator(g), M)
    U = qt.Qobj(M)
    U.dims = [[2, 2], [2, 2]]

    gg = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    ge = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
    eg = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    ee = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    s = U * gg
    b = gg.dag() * s.data
    proba_gg = abs(b[0][0])**2
    d = ee.dag() * s.data
    proba_ee = abs(d[0][0])**2
    c = ge.dag() * s.data
    proba_ge = abs(c[0][0])**2
    a = eg.dag() * s.data
    proba_eg = abs(a[0][0])**2
    return [proba_gg, proba_ge, proba_eg, proba_ee]


def proba_from_all_circuits(circuit_list):
    lis = []
    for circ in circuit_list:
        pros = proba(circ)
        lis.append(np.array(pros))
    return lis


# def calculate_ideal_circuit_2qb(nr_cycles, nr_seq, circuits_list):
#     pops_ideal = np.zeros((nr_seq, 4))
#     for i in range(nr_seq):
#         gates = [g for g in circuits_list[i] if len(g) == 2 + (nr_cycles+1)*3][0]
#         circuit = construct_from_op(gates)
#         pops_ideal[i, :] = proba(circuit)
#     return pops_ideal

def calculate_ideal_circuit_2qb(nr_cycles, nr_seq, circuits_list, Uvar=None,
                                init_state=None, extra_cycle=False):
    """
    Calculate ideal 2QB XEB circuits based on gates in circuits_list.
    :param nr_cycles:
    :param nr_seq:
    :param circuits_list: list of lists such that len(circuits_list) == nr_seq
        and len(circuits_list[i]) == len(cycles).
        Assumes [g for g in circuits_list[i] if len(g) == 2 + (nr_cycles+1)*3]
        is not empty.
    :param Uvar:
    :param init_state:
    :return:
    """
    if init_state is None:
        init_state = gg
    if Uvar is None:
        Uvar = cz
    pops_ideal = np.zeros((nr_seq, 4))
    for i in range(nr_seq):
        if extra_cycle:
            Us = [sqrtY_2qbs, T_2qbs, Uvar]
            gates = [g for g in circuits_list[i] if
                     len(g) == 2 + (nr_cycles+1)*3][0][5:]
        else:
            Us = [sqrtY_2qbs]
            gates = [g for g in circuits_list[i] if
                     len(g) == 2 + nr_cycles*3][0][2:]
        gates = np.reshape(gates, (nr_cycles, 3))
        for c in range(nr_cycles):
            Us.extend([np.kron(standard_pulses[gates[c][0][:3]],
                               standard_pulses[gates[c][1][:3]]),
                       Uvar])
        pops_ideal[i, :] = (np.abs(np.matmul(
            reduce(np.matmul, Us[::-1]), init_state))**2).T[0]
    return pops_ideal


# def calculate_ideal_circuit_2qb(data_dict, nr_cycles=None, nr_cycles_idx=None,
#                                 **params):
#     sweep_type = hlp_mod.get_param('sweep_type', data_dict,
#                                    default_value={'cycles': 0, 'seqs': 1},
#                                    **params)
#     swpts = hlp_mod.get_measurement_properties(data_dict,
#                                                props_to_extract=['sp'])
#     cycles = swpts.get_sweep_params_property('values', sweep_type['cycles'])
#
#     if nr_cycles_idx is None:
#         if nr_cycles is None:
#             raise ValueError('Please specify either nr_cycles_index or '
#                              'nr_cycles.')
#         nr_cycles_idx = np.where(np.array(cycles) == nr_cycles)[0][0]
#     all_circuits = transfer(swpts)
#     return np.array(proba_from_all_circuits(
#         all_circuits[nr_cycles_idx::len(cycles)]))

## Analysis ##
def two_qubit_xeb_analysis(timestamp=None, classifier_params=None,
                           meas_obj_names=None, renormalize=True,
                           state_prob_mtxs=None, correct_readout=(True,),
                           sweep_type=None, save_processed_data=True,
                           probability_states=None, save_figures=True,
                           raw_keys_in=None, save_filename=None, save=True,
                           meas_data_dtype=None):

    pp = pp_mod.ProcessingPipeline(add_param_method='replace')
    try:
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param_from_metadata_group(timestamp,
                                                                   'meas_objs')
        if raw_keys_in is None:
            raw_keys_in = {mobjn: 'raw' for mobjn in meas_obj_names}
        print(meas_obj_names)
        data_dict = dat_extr_mod.extract_data_hdf(
            timestamps=timestamp, meas_data_dtype=meas_data_dtype)
        swpts, movnm = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'movnm'])
        if sweep_type is None:
            sweep_type = {'cycles': 0, 'seqs': 1}

        cycles = swpts.get_sweep_params_property('values', 0)
        nr_seq = swpts.length(1)
        compression_factor = hlp_mod.get_param('compression_factor', data_dict)
        n_shots = hlp_mod.get_instr_param_from_hdf_file(
            meas_obj_names[0], 'acq_shots', timestamp)
        prep_params = hlp_mod.get_param_from_metadata_group(
            timestamp, 'preparation_params')
        reset_reps = prep_params['reset_reps'] if 'reset' in prep_params[
            'preparation_type'] else 0

        nr_swpts0 = swpts.length(0)
        data_size = len(data_dict[meas_obj_names[0]][
                            list(data_dict[meas_obj_names[0]])[0]])
        nr_swpts1 = data_size // n_shots // nr_swpts0
        n_segments = nr_swpts0 * compression_factor
        n_sequences = nr_swpts1 // compression_factor // (reset_reps+1)
        print(f'{nr_swpts0} segments, {compression_factor} hard sequences, '
              f'{n_sequences} soft sequences, {n_shots} shots')

        classifier_params = hlp_mod.get_clf_params_from_hdf_file(
            timestamp, meas_obj_names, classifier_params)
        state_prob_mtxs = hlp_mod.get_state_prob_mtxs_from_hdf_file(
            timestamp, meas_obj_names, state_prob_mtxs)
        for mobjn, mtx in state_prob_mtxs.items():
            if mtx is None:
                if any(correct_readout):
                    log.warning(f'The acq_state_prob_mtx was not provided '
                                f'for {mobjn}. The acq_state_prob_mtxs '
                                f'must be specified for both qubits in '
                                f'order to perform readout correction.')
                if False in correct_readout:
                    # only do the readout-uncorrected analysis if the user
                    # wanted this originally
                    correct_readout = (False,)
                else:
                    # no analysis to run
                    return pp, None, None, None

        # get probability_states
        if probability_states is None:
            probability_states = ['pg', 'pe']
            if len(classifier_params[meas_obj_names[0]]['weights_']) == 3:
                probability_states += ['pf']
        print('probability_states: ', probability_states)
        nr_states = len(probability_states)**len(meas_obj_names)

        for mobjn in meas_obj_names:
            pp.add_node('filter_data', keys_in=raw_keys_in[mobjn],
                        data_filter=lambda x: x[reset_reps::reset_reps+1],
                        meas_obj_names=mobjn)
            pp.add_node('classify_gm',
                        keys_in=[f'{mobjn}.filter_data {movn}' for movn in
                                 movnm[mobjn]],  # keys set by filter_data
                        keys_out=[f'{mobjn}.classify_gm.{ps}'
                                  for ps in probability_states],
                        clf_params=classifier_params.get(mobjn, None),
                        meas_obj_names=mobjn)

        # gg, ge, gf, eg, ee, ef, fg, fe, ff
        pp.add_node('calculate_flat_multiqubit_shots',
                    keys_in=hlp_mod.flatten_list(
                        [[f'{mobjn}.classify_gm.{ps}'
                          for ps in probability_states]
                         for mobjn in meas_obj_names]),
                    keys_out=None,
                    joint_processing=True, do_preselection=False,
                    meas_obj_names=meas_obj_names)
        pp.resolve(movnm)
        data_dict.update({
            'dim_hilbert': len(meas_obj_names)**2,
            'sg_qb_gate_lengths': get_sg_qb_gate_lengths(timestamp),
            'sweep_type': sweep_type,
            'meas_obj_names': meas_obj_names,
            'renormalize': renormalize
        })
        pp(data_dict)
        data_dict = pp.data_dict

        pp = pp_mod.ProcessingPipeline(add_param_method='replace')
        container = ",".join(meas_obj_names)
        pp.add_node('average_data',
                    shape=(n_sequences, n_shots, n_segments, nr_states),
                    final_shape=(n_sequences * n_segments, nr_states),
                    averaging_axis=1,
                    keys_in=[f'{container}.calculate_flat_multiqubit_shots'],
                    keys_out=[f'{container}.average_data'],
                    joint_processing=True,
                    meas_obj_names=meas_obj_names)
        if any(correct_readout):
            pp.add_node('correct_readout',
                        keys_in=[f'{container}.average_data'],
                        keys_out=[f'{container}.correct_readout'],
                        state_prob_mtxs=state_prob_mtxs,
                        joint_processing=True,
                        meas_obj_names=meas_obj_names)
        pp(data_dict)
        if len(pp) and save:
            pp.save(save_processed_data=save_processed_data,
                    save_figures=save_figures, filename=save_filename)
        return pp, meas_obj_names, cycles, nr_seq
    except Exception:
        traceback.print_exc()
        return pp, None, None, None


def calculate_fidelities_purities_2qb(data_dict, data_key='correct_readout',
                                      renormalize=True, **params):

    d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True, **params)
    print('d', d)
    sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                   default_value={'cycles': 0, 'seqs': 1},
                                   **params)

    meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(
            data_dict['timestamps'][0], 'meas_objs')
    print(meas_obj_names)
    container = ",".join(meas_obj_names)
    swpts = hlp_mod.get_measurement_properties(data_dict,
                                               props_to_extract=['sp'])
    cycles = swpts.get_sweep_params_property('values', sweep_type['cycles'])

    proba_exp = data_dict[container][data_key]
    if proba_exp.shape[1] > 4:
        proba_exp = np.concatenate([proba_exp[:, :2], proba_exp[:, 3:5]], axis=1)
        if hlp_mod.get_param('renormalize', data_dict,
                             default_value=renormalize):
            proba_exp = proba_exp/np.reshape(np.sum(proba_exp, axis=1),
                                             (proba_exp.shape[0], 1))
    proba_ideal = np.zeros_like(proba_exp)

    circuits = transfer(data_dict, **params)
    xeb_data = dict()
    purity = dict()
    for depth in cycles:
        xeb_data[depth] = 0
    i = 0
    while i < len(cycles):
        print(cycles[i])
        pops_meas = proba_exp[i::len(cycles)]
        current_circuits = circuits[i::len(cycles)]
        p_m = sqrt_purity(pops_meas, d)
        purity[cycles[i]] = p_m
        prob_ideal = proba_from_all_circuits(current_circuits)
        proba_ideal[i::len(cycles)] = prob_ideal
        xeb = crossEntropyFidelity(pops_meas, np.array(prob_ideal), d)
        xeb_data[cycles[i]] = xeb
        i += 1

    y_xeb = []
    y_purity = []
    for depth, xeb in xeb_data.items():
        y_xeb.append(xeb)
    for depth, pur in purity.items():
        y_purity.append(pur)

    apm = hlp_mod.get_param('add_param_method', data_dict, **params)
    if apm is None:
        params['add_param_method'] = 'replace'

    hlp_mod.add_param(f'{container}.xeb_probabilities',
                      np.reshape(proba_exp, (swpts.length(1), swpts.length(0),
                                             proba_exp.shape[-1])),
                      data_dict, **params)
    hlp_mod.add_param(f'{container}.xeb_probabilities_ideal',
                      np.reshape(proba_ideal, (swpts.length(1), swpts.length(0),
                                               proba_ideal.shape[-1])),
                      data_dict, **params)
    hlp_mod.add_param(f'{container}.fidelities', y_xeb, data_dict, **params)
    hlp_mod.add_param(f'{container}.purities', y_purity, data_dict, **params)


pauli_error_from_average_error = lambda avg_err, d: (1 + 1/d) * avg_err
average_error_from_pauli_error = lambda pauli_err, d: pauli_err / (1 + 1/d)


def calculate_cz_error(data_dict1, data_dict2, **params):
    timestamp = data_dict1['timestamps'][0]
    meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict1, **params)
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(
            timestamp, 'meas_objs')
    container = ",".join(meas_obj_names)
    metric = params.get('metric', 'fidelity')
    type = params.get('error_type', 'pauli')
    s1qb = params.get('subtract_1qb_errors', True)

    try:
        e2 = data_dict2[container][f'fit_res_{metric}'].best_values['e']
        e2e = data_dict2[container][f'fit_res_{metric}'].params['e'].stderr
    except AttributeError:
        e2 = data_dict2[container][f'fit_res_{metric}'][
            'params']['e']['value']
        e2e = data_dict2[container][f'fit_res_{metric}'][
            'params']['e']['stderr']

    if s1qb:
        try:
            # FIXME This try except is used since the fits saved in the data
            #  dict have slightly different formats if the data has been
            #  extracted from a previous analysis run saved in an hdf file.
            #  This should be fixed in the data extraction instead.
            e1_1 = data_dict1[meas_obj_names[0]][
                f'fit_res_{metric}'].best_values['e']
            e1_2 = data_dict1[meas_obj_names[1]][
                f'fit_res_{metric}'].best_values['e']
            e1_1e = data_dict1[meas_obj_names[0]][
                f'fit_res_{metric}'].params['e'].stderr
            e1_2e = data_dict1[meas_obj_names[1]][
                f'fit_res_{metric}'].params['e'].stderr
        except AttributeError:
            e1_1 = data_dict1[meas_obj_names[0]][f'fit_res_{metric}'][
                'params']['e']['value']
            e1_2 = data_dict1[meas_obj_names[1]][f'fit_res_{metric}'][
                'params']['e']['value']
            e1_1e = data_dict1[meas_obj_names[0]][f'fit_res_{metric}'][
                'params']['e']['stderr']
            e1_2e = data_dict1[meas_obj_names[1]][f'fit_res_{metric}'][
                'params']['e']['stderr']
        e2 = e2 - e1_1 - e1_2
        e2e = np.sqrt(e2e**2 + e1_1e**2 + e1_2e**2)

    cz_error_rate = {'value': e2, 'stderr': e2e}
    if type == 'pauli':
        pass
    elif type == 'average':
        cz_error_rate = {k: average_error_from_pauli_error(v, 4)
                         for k, v in cz_error_rate.items()}
    else:
        raise NotImplementedError
    hlp_mod.add_param(f"{container}.cz_error_rate", cz_error_rate,
                      data_dict2, add_param_method='replace')
    return cz_error_rate


def get_1qb_multi_xeb_dd(timestamp, meas_data_dtype=None, meas_obj_names=None,
                         idx0f=0, idx0p=0,):
    """
    TODO
    """
    if timestamp is None:
        return {}
    try:
        dd1 = hlp_mod.read_analysis_file(timestamp, raise_errors=True)
        # raise FileNotFoundError
    except FileNotFoundError:
        pp, meas_obj_names1, cycles, nr_seq = single_qubit_xeb_analysis(
            timestamp,
            meas_obj_names=meas_obj_names,
            meas_data_dtype=meas_data_dtype,
            save=False)
        plot_porter_thomas_dist(pp.data_dict, savefig=True)
        calculate_fidelities_purities_1qb(
            pp.data_dict,
            data_key='correct_readout'  # 'average_data'
        )
        _ = fit_plot_fidelity_purity(
            pp.data_dict,
            idx0f=0, idx0p=0,
            savefig=True,
            log_scale=False
        )
        fit_plot_leakage_1qb(pp.data_dict, meas_obj_names,
                                     data_key='correct_readout',
                                     idx0f=idx0f, idx0p=idx0p,
                                     savefig=True, show=False)
        pp.save()
        plt.close('all')
        dd1 = pp.data_dict
    return dd1


def get_2qb_multi_xeb_dd(timestamp, clear_some_memory=True, timer=None,
                         meas_data_dtype=None, meas_obj_names=None,
                         idx0f=0, idx0p=0,):
    """
    TODO
    """
    from pycqed.measurement import sweep_points as sp_mod

    task_id = 0

    dd2 = []
    cphases = \
    hlp_mod.get_param_from_metadata_group(timestamp, 'task_list')[task_id][
        'cphases']

    if timer:
        timer.checkpoint('two_qubit_xeb_analysis.start')
    pp_full, meas_obj_names2, cycles1, nr_seq1 = two_qubit_xeb_analysis(
        timestamp,
        meas_obj_names=meas_obj_names,
        save=False,
        meas_data_dtype=meas_data_dtype,
        # timer=timer,
    )
    if timer:
        timer.checkpoint('two_qubit_xeb_analysis.end')
    sp_full = sp_mod.SweepPoints(
        pp_full.data_dict['exp_metadata']['sweep_points'])

    for idx in range(len(cphases)):  # loop over cphases
        pp = deepcopy(pp_full)
        # Trim sp
        sp = rb_meas.TwoQubitXEBMultiCphase.extract_combined_sweep_points(
            sp_full=sp_full, idx=idx, deep=True)
        pp.data_dict['exp_metadata']['sweep_points'] = sp

        # Set cphase
        pp.data_dict['exp_metadata']['cphase'] = cphases[idx]

        # Trim data and only keep what corresponds to one cphase
        data = pp.data_dict[','.join(meas_obj_names)]['correct_readout']
        data = data.reshape(
            [sp.length(1), len(cphases), sp.length(0),
             9])[:, idx, :, :].reshape([-1, 9])
        pp.data_dict[','.join(meas_obj_names)]['correct_readout'] = data

        # Set mospm
        mospm = sp.get_meas_obj_sweep_points_map(meas_obj_names)
        # for v in mospm.values():
        #     v.pop(1)
        print(f"mospm = {mospm}")
        pp.data_dict['exp_metadata']['meas_obj_sweep_points_map'] = mospm

        # Analysis
        if timer:
            timer.checkpoint('plot_porter_thomas_dist.start')
        plot_porter_thomas_dist(pp.data_dict, savefig=True)
        if timer:
            timer.checkpoint('plot_porter_thomas_dist.end')
        if timer:
            timer.checkpoint('calculate_fidelities_purities_2qb.start')
        calculate_fidelities_purities_2qb(
            pp.data_dict,
            data_key='correct_readout',
            timer=timer,
        )
        if timer:
            timer.checkpoint('calculate_fidelities_purities_2qb.end')
        if timer:
            timer.checkpoint('fit_plot_fidelity_purity.start')
        _ = fit_plot_fidelity_purity(
            pp.data_dict,
            idx0f=idx0f, idx0p=idx0p,
            joint_processing=True,
            savefig=True,
            log_scale=False
        )
        if timer:
            timer.checkpoint('fit_plot_fidelity_purity.end')
        if timer:
            timer.checkpoint('fit_plot_leakage_2qb.start')
        fit_plot_leakage_2qb(
            pp.data_dict, meas_obj_names,
            data_key='correct_readout',
            savefig=True, show=False, timer=timer
        )
        if timer:
            timer.checkpoint('fit_plot_leakage_2qb.end')
        plt.close('all')
        if timer:
            timer.checkpoint('pp.save.start')
        # pp.save()  # Removed for speed reasons. Could keep once works properly
        if timer:
            timer.checkpoint('pp.save.end')
        dd = pp.data_dict
        del pp
        if clear_some_memory:
            for mobjn in meas_obj_names:
                # Raw data seems to be the highest quantity of data
                # (from task manager: 80%?)
                del dd[mobjn]
        dd2.append(dd)

        # if idx >= 1:
        #     break

    del pp_full
    return dd2


def get_multi_xeb_results_from_dd(dd1, dd2, meas_obj_names=None, **kw):
    """
    TODO

    Args:
        meas_obj_names (list): mobj names in case they aren't saved by the
            experiment in the correct order, preventing to recalculate the
            quantum circuits here. FIXME this should be saved correctly instead
    """
    results = {}
    for dd in dd2:
        cphase = dd['exp_metadata']['cphase']
        results[cphase] = res = {}
        res['tot'] = calculate_cz_error(
            dd1, dd, meas_obj_names=meas_obj_names,
            metric='fidelity', error_type='average', **kw)
        res['inc'] = calculate_cz_error(
            dd1, dd, meas_obj_names=meas_obj_names,
            metric='purity', error_type='average', **kw)
        res['coh'] = {
            'value': res['tot']['value'] - res['inc']['value'],
            'stderr': np.linalg.norm(
                [res['tot']['stderr'], res['inc']['stderr']], 2),
        }
    return results


## Functions common to both 1 and 2 qubits ##
# standard_pulses = {
#     'I': qt.qeye(2),
#     'X0': qt.qeye(2),
#     'Z0': qt.qeye(2),
#     'mX180': qt.qip.operations.gates.rx(np.pi),
#     'X180': qt.qip.operations.gates.rx(-np.pi),
#     'mY180': qt.qip.operations.gates.ry(np.pi),
#     'Y180': qt.qip.operations.gates.ry(-np.pi),
#     'mX90': qt.qip.operations.gates.rx(np.pi/2),
#     'X90': qt.qip.operations.gates.rx(-np.pi/2),
#     'mY90': qt.qip.operations.gates.ry(np.pi/2),
#     'Y90': qt.qip.operations.gates.ry(-np.pi/2),
#     'mZ90': qt.qip.operations.gates.rz(np.pi/2),
#     'Z90': qt.qip.operations.gates.rz(-np.pi/2),
#     'mZ180': qt.qip.operations.gates.rz(np.pi),
#     'Z180': qt.qip.operations.gates.rz(-np.pi),
# }
standard_pulses_qt = {
    'I': qt.qeye(2),
    'X0': qt.qeye(2),
    'Z0': qt.qeye(2),
    'mX180': qt.qip.operations.gates.rx(-np.pi, N=None, target=0),
    'X180': qt.qip.operations.gates.rx(np.pi, N=None, target=0),
    'mY180': qt.qip.operations.gates.ry(-np.pi, N=None, target=0),
    'Y180': qt.qip.operations.gates.ry(np.pi, N=None, target=0),
    'mX90': qt.qip.operations.gates.rx(-np.pi/2, N=None, target=0),
    'X90': sqrtXqt,
    'mY90': qt.qip.operations.gates.ry(-np.pi/2, N=None, target=0),
    'Y90': sqrtYqt,
    'mZ90': qt.qip.operations.gates.rz(-np.pi/2, N=None, target=0),
    'Z90': qt.qip.operations.gates.rz(np.pi/2, N=None, target=0),
    'mZ180': qt.qip.operations.gates.rz(-np.pi, N=None, target=0),
    'Z180': qt.qip.operations.gates.rz(np.pi, N=None, target=0),
    'Z45': Tqt,
}

standard_pulses = {k: g.full() for k, g in standard_pulses_qt.items()}

sqrt_purity = lambda pops_meas, d: \
    np.sqrt(np.var(pops_meas) * (d**2) * (d+1) / (d-1))


def get_populations_unitary(U, init_state, povms):
    final_state = U @ init_state
    final_state_dag = final_state.T.conj()
    return [np.real((final_state_dag @ M @ final_state)[0][0]) for M in povms]


def fit_plot_fidelity_purity(data_dict, idx0f=0, idx0p=0, meas_obj_names=None,
                             fidelities=None, purities=None,
                             exclude_from_plot=False, cz_error_rate=None,
                             fit=True, plot=True, joint_processing=False,
                             savefig=True, fmts=None, log_scale=False,
                             filename=None, filename_prefix='', show=False,
                             legend_kw=None, text_position=None, text_kw=None,
                             **params):

    fig = None
    try:
        if legend_kw is None:
            legend_kw = {}

        d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True,
                              **params)
        print('d', d)
        timestamp = data_dict['timestamps'][0]
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
            if meas_obj_names is None:
                meas_obj_names = hlp_mod.get_param_from_metadata_group(
                    timestamp, 'meas_objs')

        if joint_processing:
            meas_obj_names = [",".join(meas_obj_names)]

        if fidelities is None:
            fidelities = {}
            for mobjn in meas_obj_names:
                fidelities[mobjn] = hlp_mod.get_param(f'{mobjn}.fidelities',
                                                      data_dict,
                                                      raise_error=True,
                                                      **params)
        else:
            fidelities = {meas_obj_names[0]: fidelities}

        if purities is None:
            purities = {}
            for mobjn in meas_obj_names:
                purities[mobjn] = hlp_mod.get_param(f'{mobjn}.purities',
                                                    data_dict,
                                                    raise_error=True, **params)
            if not len(purities):
                raise ValueError('No purities to plot.')
        else:
            purities = {meas_obj_names[0]: purities}

        sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                       default_value={'cycles': 0, 'seqs': 1},
                                       **params)
        swpts = hlp_mod.get_measurement_properties(data_dict,
                                                   props_to_extract=['sp'])
        cycles = copy(swpts.get_sweep_params_property('values', sweep_type['cycles']))
        start_idx = 1 if cycles[0] == 0 else 0
        cycles = cycles[start_idx:]
        nr_seq = swpts.length(sweep_type['seqs'])

        fit_guess_params = hlp_mod.get_param(
            'fit_guess_params', {}, default_value={}, **params)
        if text_position is None:
            text_position = [0.05, 0.075] if log_scale else [0.975, 0.7]
        return_dict = {}
        for mobjn in fidelities:
            return_dict[mobjn] = []
            fid = fidelities[mobjn][start_idx:]
            pur = purities[mobjn][start_idx:]
            user_guess_dict = fit_guess_params.get(mobjn, {})

            if fit:
                apm = hlp_mod.get_param('add_param_method', data_dict, **params)
                if apm is None:
                    params['add_param_method'] = 'replace'

                guess_pars = {'A': {'value': 1.0},
                              'e': {'value': 0.01},
                              'B': {'value': 0.0}}

                # fit fidelities
                guess_pars_to_use = deepcopy(guess_pars)
                guess_pars_to_use.update(user_guess_dict.get('fidelity', {}))
                model = lmfit.Model(
                    lambda m, e, A, B: A*(1-e/(1-1/(d**2)))**m + B)
                model.set_param_hint('e', **guess_pars_to_use['e'])
                model.set_param_hint('A', **guess_pars_to_use['A'])
                model.set_param_hint('B', **guess_pars_to_use['B'])
                fit_res_f = model.fit(data=fid[idx0f:], m=cycles[idx0f:],
                                    params=model.make_params())
                hlp_mod.add_param(f'{mobjn}.fit_res_fidelity',
                                  fit_res_f, data_dict, **params)

                # fit purities
                guess_pars_to_use = deepcopy(guess_pars)
                guess_pars_to_use.update(user_guess_dict.get('purity', {}))
                model = lmfit.Model(
                    lambda m, e, A, B: A*(1-e/(1-1/(d**2)))**m + B)
                model.set_param_hint('e', **guess_pars_to_use['e'])
                model.set_param_hint('A', **guess_pars_to_use['A'])
                model.set_param_hint('B', **guess_pars_to_use['B'])
                fit_res_p = model.fit(data=pur[idx0p:], m=cycles[idx0p:],
                                    params=model.make_params())
                hlp_mod.add_param(f'{mobjn}.fit_res_purity',
                                  fit_res_p, data_dict, **params)

                return_dict[mobjn] += [fit_res_f, fit_res_p]

            if plot:
                with mpl.rc_context(rc={}):
                    fig, ax = plt.subplots()
                    if cz_error_rate is None:
                        cz_error_rate = hlp_mod.get_param(
                            f'{mobjn}.cz_error_rate', data_dict)

                    if exclude_from_plot:
                        line_f, = ax.plot(cycles[idx0f:], fid[idx0f:], 'o',
                                          zorder=0)
                        line_p, = ax.plot(cycles[idx0p:], pur[idx0p:], 'o',
                                          zorder=1)
                    else:
                        line_f, = ax.plot(cycles, fid, 'o', zorder=0)
                        line_p, = ax.plot(cycles, pur, 'o', zorder=1)

                    if fit:
                        xfine = np.linspace(cycles[idx0f], cycles[-1], 100)
                        ax.plot(xfine, fit_res_f.model.func(xfine, **fit_res_f.best_values),
                                c=line_f.get_color(), zorder=0)
                        xfine = np.linspace(cycles[idx0p], cycles[-1], 100)
                        ax.plot(xfine, fit_res_p.model.func(xfine, **fit_res_p.best_values),
                                c=line_p.get_color(), zorder=1)

                        epc, epc_err = 100*fit_res_f.best_values["e"], \
                                       100*fit_res_f.params["e"].stderr
                        # hlp_mod.add_param(f'{mobjn}.fit_results.EPC',
                        #                   (epc/100, epc_err/100), data_dict,
                        #                   **params)
                        ppc, ppc_err = 100*fit_res_p.best_values["e"], \
                                       100*fit_res_p.params["e"].stderr
                        # hlp_mod.add_param(f'{mobjn}.fit_results.PPC',
                        #                   (ppc/100, ppc_err/100), data_dict,
                        #                   **params)
                        ctrl_err = epc-ppc
                        ctrl_err_err = np.sqrt(epc_err**2 + ppc_err**2)
                        # hlp_mod.add_param(f'{mobjn}.fit_results.CEPC',
                        #                   (ctrl_err/100, ctrl_err_err/100),
                        #                   data_dict, **params)
                        textstr = f'{epc:.3f}% $\\pm$ {epc_err:.3f}% error per c.' \
                                  f'\n{ppc:.3f}% $\\pm$ {ppc_err:.3f}% purity per c.' \
                                  f'\n{ctrl_err:.3f}% $\\pm$ ' \
                                  f'{ctrl_err_err:.3f}% ctrl. errors per c.'
                        if cz_error_rate is not None:
                            textstr += f'\nCZ error: ' \
                                       f'{100*cz_error_rate.get("value", "nan"):.3f}% ' \
                                       f'$\\pm$ {100*cz_error_rate.get("stderr", "nan"):.3f}%'
                        if text_kw is None:
                            text_kw = dict(va='bottom', ha='left') if log_scale \
                                else dict(va='top', ha='right')
                        ax.text(*text_position, textstr, transform=ax.transAxes,
                                **text_kw)

                    ax.plot([], [], 'o-', c=line_f.get_color(),  label='Fidelity')
                    ax.plot([], [], 'o-', c=line_p.get_color(),
                            label='$\\sqrt{\mathrm{Purity}}$')
                    ax.legend(frameon=False, **legend_kw)
                    cz_name = f"_CZ{data_dict['exp_metadata']['cphase']}" \
                        if 'cphase' in data_dict['exp_metadata'] else ''
                    ax.set_title(
                        f'{filename_prefix}XEB{cz_name} {mobjn} - {timestamp}')

                    ax.set_ylabel('XEB fidelity, $\\sqrt{\mathrm{Purity}}$')
                    ax.set_xlabel('Number of cycles, $m$')
                    if log_scale:
                        ax.set_yscale('log')
                        ax.set_ylim(None, 1.5)
                        fig.subplots_adjust(0.14, 0.16, 0.99, 0.9)
                    else:
                        ax.set_ylim(-0.1, 1.1)
                        fig.subplots_adjust(0.12, 0.16, 0.99, 0.9)

                    xlims = params.get('xlims', None)
                    if xlims is not None:
                        ax.set_xlim(xlims)
                    ylims = params.get('ylims', None)
                    if ylims is not None:
                        ax.set_ylim(ylims)
                    return_dict[mobjn] += [fig, ax]

                    if savefig:
                        if fmts is None:
                            fmts = ['png']
                        fn = copy(filename)
                        if fn is None:
                            fn = f'XEB_{mobjn}_{cycles[-1]}cycles_{nr_seq}seqs_' \
                                 f'{cz_name}_{timestamp}'
                        if log_scale:
                            fn += '_log'
                        fn = f'{filename_prefix}{fn}'
                        for ext in ['png']:
                            fn = f'{data_dict["folders"][0]}\\{fn}.{ext}'
                            fig.savefig(fn)
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
        return return_dict
    except Exception:
        if fig is not None:
            plt.close(fig)
        traceback.print_exc()
        return


def fit_plot_leakage_1qb(data_dict, meas_obj_names, data_key='correct_readout',
                         filename_prefix='', savefig=True, show=False, **params):
    """
    Fit and plot the f-state population measured in a single-qubit
    XEB experiment.

    Args:
        data_dict (dict): containing the f-state population and other needed
            metdata (sweep points
        meas_obj_names (list): of qubit names. One figure will be created for
            each qubit.
        data_key (str): 'correct_readout' to look at the readout-corrected data
            or 'average_data' to look at the data without readout correction
        savefig (bool): whether to save the figure or not. If True, the figure
            is saved to data_dict['timestamps'][0].
        show (bool): whether to show the figure or not
        **params: keyword arguments: passed to extract_leakage_classified_shots
    """
    for mobjn in meas_obj_names:
        fig, ax = plt.subplots()  # here such that we can close fig in case of error
        try:
            swpts = hlp_mod.get_measurement_properties(data_dict, ['sp'])
            cycles = swpts.get_sweep_params_property('values', 0)
            pf = hlp_mod.get_param(f'{mobjn}.{data_key}.pf', data_dict,
                                   raise_error=True, **params)

            legend_info = {}
            avg_leaked = np.mean(pf.reshape(swpts.length(1), swpts.length(0)),
                                 axis=0)

            # plot data
            data_line, = ax.plot(cycles, avg_leaked, 'o')

            # do fit
            rbleak_mod = lmfit.Model(fit_mods.RandomizedBenchmarkingLeakage)
            guess_pars = rbleak_mod.make_params(pu=0.01, pd=0.05, p0=0)
            fit_res = rbleak_mod.fit(data=avg_leaked, numCliff=cycles,
                                     params=guess_pars)

            # add to data dict
            hlp_mod.add_param(f'{mobjn}.leakage',
                              pf.reshape(swpts.length(1), swpts.length(0)),
                              data_dict, add_param_method='replace')
            hlp_mod.add_param(f'{mobjn}.fit_res_leakage',
                              fit_res, data_dict, **params)
            hlp_mod.add_param(f'{mobjn}.fit_results.leakage',
                              (fit_res.best_values["pu"],
                               fit_res.params["pu"].stderr), data_dict,
                              add_param_method='replace')

            # plot fit line
            cycles_fine = np.linspace(cycles[0], cycles[-1], 100)
            ax.plot(cycles_fine,
                    fit_res.model.func(cycles_fine, **fit_res.best_values),
                    '-', c=data_line.get_color())

            # add legend
            legend_info[data_line.get_color()] = \
                f'{mobjn.upper()}: {100 * fit_res.best_values["pu"]:.5f}%' \
                f'$\\pm${100 * fit_res.params["pu"].stderr:.5f}%'
            for c, label in legend_info.items():
                ax.plot([], [], '-o', c=c, label=label)
            ax.legend(frameon=False)
            cz_name = f"_CZ{data_dict['exp_metadata']['cphase']}"\
                if 'cphase' in data_dict['exp_metadata'] else ''

            ax.set_xlabel('Number of Cycles, $m$')
            ax.set_ylabel('Probability, $P(f)$')
            timestamp = data_dict['timestamps'][0]
            ax.set_title(f'{filename_prefix}Leakage{cz_name} {mobjn} - {timestamp}')

            if savefig:
                fig.savefig(data_dict['folders'][0] +
                            f'\\{filename_prefix}Leakage{cz_name}_{mobjn}_{timestamp}.png',
                            dpi=600, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            plt.close(fig)
            raise e


def fit_plot_leakage_2qb(data_dict, meas_obj_names, data_key='correct_readout',
                         filename_prefix='', savefig=True, show=False, **params):
    """
    Fit and plot the f-state population measured in a two-qubit
    XEB experiment.

    This function uses extract_leakage_classified_shots to calculate the
    leakage probability that one qubit has leaked or that either of the two
    qubits have leaked, creating a subplot for each case.

    Args:
        data_dict (dict): containing the f-state population and other needed
            metdata (sweep points
        meas_obj_names (list): of qubit names
        data_key (str): 'correct_readout' to look at the readout-corrected data
            or 'average_data' to look at the data without readout correction
        savefig (bool): whether to save the figure or not. If True, the figure
            is saved to data_dict['timestamps'][0].
        show (bool): whether to show the figure or not
        **params: keyword arguments: passed to extract_leakage_classified_shots
    """
    fig, axs = plt.subplots(
        3, sharex=True,
        figsize=([plt.rcParams['figure.figsize'][0],]*2)
    )
    try:
        fit_res_all = {}
        swpts = hlp_mod.get_measurement_properties(data_dict, ['sp'])
        cycles = swpts.get_sweep_params_property('values', 0)
        # get leakage
        mobjn_joined = ",".join(meas_obj_names)
        dat_proc_mod.extract_leakage_classified_shots(
            data_dict, [f'{mobjn_joined}.{data_key}'],
            meas_obj_names=meas_obj_names, add_param_method='replace', **params)

        for i, ax in enumerate(axs):
            keys = list(data_dict['extract_leakage_classified_shots'])
            leakage = data_dict['extract_leakage_classified_shots'][keys[i]].reshape(
                    swpts.length(1), swpts.length(0))
            hlp_mod.add_param(f'{mobjn_joined}.leakage.{keys[i]}', leakage,
                              data_dict, add_param_method='replace')

            avg_leak = np.mean(
                data_dict['extract_leakage_classified_shots'][keys[i]].reshape(
                    swpts.length(1), swpts.length(0)),
                axis=0)
            # plot data
            ax.plot(cycles, avg_leak, 'o')

            rbleak_mod = lmfit.Model(fit_mods.RandomizedBenchmarkingLeakage)
            guess_pars = rbleak_mod.make_params(pu=0.01, pd=0.05, p0=0)
            fit_res = rbleak_mod.fit(data=avg_leak, numCliff=cycles,
                                     params=guess_pars)
            fit_res_all[keys[i]] = (fit_res.best_values["pu"],
                                    fit_res.params["pu"].stderr)
            hlp_mod.add_param(f'{mobjn_joined}.fit_res_leakage.{keys[i]}',
                              fit_res, data_dict, add_param_method='replace')

            cycles_fine = np.linspace(cycles[0], cycles[-1], 100)
            ax.plot(cycles_fine, fit_res.model.func(
                cycles_fine, **fit_res.best_values), '-C0')

            textstr = f'{keys[i]}: '
            textstr += f'{100*fit_res.best_values["pu"]:.3f}%' \
                       f'$\\pm${100*fit_res.params["pu"].stderr:.3f}%'
            ax.text(0.95, 0.075, textstr,
                    ha='right', va='bottom', transform=ax.transAxes)

        axs[2].set_xlabel('Cycles, $m$')
        axs[1].set_ylabel('Probability, $P_f$')

        timestamp = data_dict['timestamps'][0]
        cz_name = f"_CZ{data_dict['exp_metadata']['cphase']}"\
            if 'cphase' in data_dict['exp_metadata'] else ''
        axs[0].set_title(f'{filename_prefix}Leakage{cz_name} {mobjn_joined} - {timestamp}')

        # add to data dict
        hlp_mod.add_param(f'{mobjn_joined}.fit_results.leakage',
                          fit_res_all, data_dict,
                          add_param_method='replace')

        if savefig:
            fig.savefig(data_dict['folders'][0] +
                        f'\\{filename_prefix}Leakage{cz_name}_{mobjn_joined}_{timestamp}.png',
                        dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:
        plt.close(fig)
        raise e


pdf_PT = lambda x, d: (d - 1) * (1 - x) ** (d - 2)
cdf_PT = lambda x, d: 1 - (1 - x) ** (d - 1)
def get_cdfs(pops, d):
    x_sample = pops
    x = np.sort(x_sample)
    y_cdf_experiment = np.linspace(0, 1, len(x)+1, endpoint=True)[1:]
    y_cdf_theory = cdf_PT(x, d)
    return x, y_cdf_experiment, y_cdf_theory, \
           np.mean(np.abs(y_cdf_experiment - y_cdf_theory))


def plot_porter_thomas_dist(data_dict, data_key='correct_readout',
                            savefig=True, fmts=None,
                            figure_width=None, figure_height=None,
                            filename=None, filename_prefix='',
                            numcols=None, numrows=None, renormalize=True,
                            show=False, nr_sp_2d=None, **params):
    try:
        d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True,
                              **params)
        print('d', d)
        sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                       default_value={'cycles': 0, 'seqs': 1},
                                       **params)

        timestamp = data_dict['timestamps'][0]
        meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param_from_metadata_group(
                timestamp, 'meas_objs')
        if d == 4:
            meas_obj_names = [",".join(meas_obj_names)]

        swpts = hlp_mod.get_measurement_properties(data_dict,
                                                   props_to_extract=['sp'])
        cycles = swpts.get_sweep_params_property('values', sweep_type['cycles'])
        if nr_sp_2d is None:
            nr_sp_2d = swpts.length(sweep_type['seqs'])
        renormalize = hlp_mod.get_param('renormalize', data_dict,
                                        default_value=renormalize)

        if numcols is None or numrows is None:
            numcols_list = [4, 3, 2, 1]
            for numr in numcols_list:
                if not len(cycles) % numr:
                    # numrows = numr
                    # numcols = len(cycles) // numr
                    numcols = numr
                    numrows = len(cycles) // numr
                    break
        print(numrows, numcols)
        if figure_width is None:
            figure_width = plt.rcParams['figure.figsize'][0]/2*numcols
        if figure_height is None:
            figure_height = plt.rcParams['figure.figsize'][1]/2*numrows
        return_dict = {}

        rc_params = {
            'figure.figsize': (figure_width, figure_height),
        }
        with mpl.rc_context(rc=rc_params):
            for mobjn in meas_obj_names:
                proba_exp_all = hlp_mod.get_param(f'{mobjn}.xeb_probabilities',
                                                  data_dict)
                xeb_proba_found = True
                if proba_exp_all is None:
                    proba_exp_all = data_dict[mobjn][data_key]
                    xeb_proba_found = False
                print('xeb_proba_found', xeb_proba_found)

                fig, axs = plt.subplots(numrows, numcols,
                                        sharex=True, sharey=True)
                big_ax = fig.add_subplot(111, frameon=False)
                big_ax.tick_params(
                    labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
                for i, nrc in enumerate(cycles):
                    proba_exp = proba_exp_all

                    if isinstance(axs, np.ndarray):
                        ax = axs.flatten()[i]
                    else:
                        ax = axs

                    if not xeb_proba_found:
                        if isinstance(proba_exp_all, dict):
                            # 1 qb case
                            proba_exp = proba_exp_all['pe']
                            if renormalize:
                                proba_exp /= (proba_exp + proba_exp_all['pg'])
                            # add pg = 1-pe
                            proba_exp = np.concatenate([(1 - proba_exp)[np.newaxis].T,
                                                        proba_exp[np.newaxis].T],
                                                       axis=1)
                            # proba_exp = np.array(list(proba_exp_all.values())).T
                        else:
                            # 2 qb case
                            proba_exp = np.concatenate([proba_exp_all[:, :2],
                                                        proba_exp_all[:, 3:5]],
                                                       axis=1)
                            if renormalize:
                                proba_exp = proba_exp/np.reshape(
                                    np.sum(proba_exp, axis=1),
                                    (proba_exp.shape[0], 1))
                        proba_exp = proba_exp.reshape(nr_sp_2d,
                                                      swpts.length(0),
                                                      proba_exp.shape[-1])
                    proba_exp = proba_exp[:, i, :]

                    # get cdfs
                    x, y_exp, _, distance = get_cdfs(proba_exp.flatten(), d)
                    # Plot data
                    ax.plot(x, y_exp, linewidth=2, label='Measured data')
                    # Porter-Thomas distrib:
                    xpt = np.linspace(0, 1, 100)
                    ax.plot(xpt, cdf_PT(xpt, d), 'k--', label='Porter Thomas')
                    ax.vlines(1/d, 0, 1, colors='gray', linestyles='--',
                              linewidth=1)
                    ax.text(0.95, 0.05, f'{nrc} cycles\n$\\xi$={distance:.3f}',
                            ha='right', va='bottom', transform=ax.transAxes)

                # add legend
                odd_nr_cols = numcols % 2 == 1
                if not hasattr(axs, "shape"):
                    ax = axs
                elif len(axs.shape) == 1:
                    ax = axs[(numcols // 2 + odd_nr_cols) - 1]
                else:
                    ax = axs[0, (numcols // 2 + odd_nr_cols) - 1]
                ax.legend(frameon=False, loc='lower center', ncol=2,
                          bbox_to_anchor=(0.5 if odd_nr_cols else 1.05, 0.965))

                big_ax.set_ylabel('Cumulative distribution')
                big_ax.set_xlabel('Basis-state prob., $p$')
                cz_name = f"_CZ{data_dict['exp_metadata']['cphase']}"\
                    if 'cphase' in data_dict['exp_metadata'] else ''
                fig.suptitle(f'{filename_prefix}XEB{cz_name} {mobjn} - '
                                 f'{timestamp}', y=1.0)
                fig.subplots_adjust(wspace=0.05, hspace=0.05)
                return_dict[mobjn] = [fig, axs]

                if savefig:
                    if fmts is None:
                        fmts = ['png']
                    fn = copy(filename)
                    if fn is None:
                        fn = f'Porter_Thomas_Cumulative_{mobjn}_' \
                             f'{cycles[-1]}cycles_{nr_sp_2d}seqs{cz_name}_' \
                             f'{timestamp}'
                    fn = f'{filename_prefix}{fn}'
                    for ext in fmts:
                        fn = f'{data_dict["folders"][0]}\\{fn}.{ext}'
                        fig.savefig(fn)
                if show:
                    plt.show()
                else:
                    plt.close(fig)
        return return_dict
    except Exception:
        plt.close(fig)
        traceback.print_exc()
        return


def calculate_optimal_nr_cycles(data_dict, idx0=0, joint_processing=False,
                                fidelities=None, exlude_from_plot=False,
                                plot=True, savefig=True, fmts=None, show=False,
                                log_scale=False, filename=None, **params):
    try:
        timestamp = data_dict['timestamps'][0]
        meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param_from_metadata_group(
                timestamp, 'meas_objs')

        if joint_processing:
            meas_obj_names = [",".join(meas_obj_names)]

        if fidelities is None:
            fidelities = {}
            for mobjn in meas_obj_names:
                fidelities[mobjn] = hlp_mod.get_param(f'{mobjn}.fidelities',
                                                      data_dict,
                                                      raise_error=True,
                                                      **params)
        else:
            fidelities = {meas_obj_names[0]: fidelities}

        swpts = hlp_mod.get_measurement_properties(data_dict, ['sp'])
        cycles = swpts.get_sweep_params_property('values', 0)

        apm = hlp_mod.get_param('add_param_method', data_dict, **params)
        if apm is None:
            params['add_param_method'] = 'replace'

        nrc_opt_all = {}
        for mobjn, fid in fidelities.items():
            fit_res = lmfit.Model(
                lambda m, p=0.98, A=0.5, B=0: A*p**m + B).fit(
                data=fid[idx0:], m=cycles[idx0:])

            # calculate optimal nr_cyles for optimization
            nrc_opt = int(np.floor(-1/np.log(fit_res.best_values['p'])))
            hlp_mod.add_param(f'{mobjn}.optimal_nr_cycles',
                              nrc_opt, data_dict, **params)
            nrc_opt_all[mobjn] = nrc_opt

            if plot:
                with mpl.rc_context(rc={}):
                    fig, ax = plt.subplots()

                    # plot data
                    if exlude_from_plot:
                        line, = ax.plot(cycles[idx0:], fid[idx0:], 'o', zorder=1)
                    else:
                        line, = ax.plot(cycles, fid, 'o', zorder=1)

                    # plot optimal nr_cycles lines
                    opt_point = fit_res.model.func(nrc_opt, **fit_res.best_values)
                    ax.plot(nrc_opt, opt_point, 'o', c='gray')
                    ax.vlines(nrc_opt, -0.1, opt_point,
                              linestyles='--', color='gray', zorder=2)
                    ax.hlines(opt_point, 0, nrc_opt,
                              linestyles='--', color='gray', zorder=2)

                    # plot fit
                    xfine = np.linspace(cycles[idx0], cycles[-1], 100)
                    ax.plot(xfine, fit_res.model.func(xfine, **fit_res.best_values),
                            c=line.get_color(), zorder=0)

                    p, p_err = 100*fit_res.best_values["p"], \
                                   100*fit_res.params["p"].stderr
                    textstr = f'Decay const.: {p:.3f}% $\\pm$ {p_err:.3f}%' \
                              f'\nOptimal nr. cycles: {nrc_opt}'

                    if log_scale:
                        ax.text(0.05, 0.075, textstr,
                                va='bottom', ha='left', transform=ax.transAxes)
                    else:
                        ax.text(0.975, 0.95, textstr,
                                va='top', ha='right', transform=ax.transAxes)

                    ax.set_title(f'XEB {mobjn} - {timestamp}')
                    ax.set_ylabel('XEB fidelity')
                    ax.set_xlabel('Number of cycles, $m$')
                    if log_scale:
                        ax.set_yscale('log')
                        ax.set_ylim(None, 1.5)
                        fig.subplots_adjust(0.14, 0.16, 0.99, 0.9)
                    else:
                        ax.set_ylim(-0.1, 1.1)
                        fig.subplots_adjust(0.12, 0.16, 0.99, 0.9)

                    if savefig:
                        if fmts is None:
                            fmts = ['png']
                        fn = copy(filename)
                        if fn is None:
                            fn = f'Opt_nr_cycles_{mobjn}_{swpts.length(1)}seqs_' \
                                 f'{timestamp}'
                        if log_scale:
                            fn += '_log'
                        for ext in ['png']:
                            fn = f'{data_dict["folders"][0]}\\{fn}.{ext}'
                            fig.savefig(fn)
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
        return nrc_opt_all
    except Exception:
        plt.close(fig)
        traceback.print_exc()
        return


def plot_ctrl_errors(data_dict, unitary_label, data_dict_cepc=None,
                     process_errs=None,
                     show_error_stripe=False, show=False,
                     savefig=True, fmts=None,
                     filename=None, filename_prefix='', fig_margins=None,
                     **params):
    if data_dict_cepc is None:
        data_dict_cepc = data_dict

    timestamp = data_dict['timestamps'][0]
    meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
    if meas_obj_names is None:
        meas_obj_names = hlp_mod.get_param_from_metadata_group(
            timestamp, 'meas_objs')
    d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True,
                          **params)
    if d == 4:
        meas_obj_names = [",".join(meas_obj_names)]

    sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                   default_value={'cycles': 0, 'seqs': 1})
    swpts = hlp_mod.get_measurement_properties(data_dict,
                                               props_to_extract=['sp'])
    cycles = swpts.get_sweep_params_property('values', sweep_type['cycles'])

    if fig_margins is None:
        fig_margins = [0.17, 0.2, 0.99, 0.90]
    try:
        with mpl.rc_context(rc={}):
            for mobjn in meas_obj_names:
                process_errs = hlp_mod.get_param(f'{mobjn}.optimization_result_process_errs',
                                                 data_dict, default_value=process_errs,
                                                 **params)
                fig, ax = plt.subplots()
                # colours = {'qb3': (253/255, 208/255, 162/255), 'qb7': (127/255, 39/255, 4/255)}
                ax.plot(cycles, 100*np.array(list(process_errs.values())), 'o')
                try:
                    ctrl_errs = 100*(data_dict_cepc[mobjn]['fit_res_fidelity'].best_values['e'] - \
                                     data_dict_cepc[mobjn]['fit_res_purity'].best_values['e'])
                    ctrl_errs_err = 100*np.sqrt(data_dict_cepc[mobjn]['fit_res_fidelity'].params['e'].stderr**2 + \
                                                data_dict_cepc[mobjn]['fit_res_purity'].params['e'].stderr**2)
                except AttributeError:
                    ctrl_errs = 100*(data_dict_cepc[mobjn]['fit_res_fidelity']['params']['e']['value'] - \
                                     data_dict_cepc[mobjn]['fit_res_purity']['params']['e']['value'])
                    ctrl_errs_err = 100*np.sqrt(data_dict_cepc[mobjn]['fit_res_fidelity']['params']['e']['stderr']**2 + \
                                                data_dict_cepc[mobjn]['fit_res_purity']['params']['e']['stderr']**2)
                if show_error_stripe:
                    ax.axhspan(ctrl_errs-ctrl_errs_err, ctrl_errs+ctrl_errs_err, 0,
                               cycles[-1]+2, color='gray', alpha=0.25)
                ax.hlines(ctrl_errs, cycles[0]-2, cycles[-1]+2, linestyles='--',
                          colors='gray', label='ctrl. err. per cycle')
                ax.legend(frameon=False)
                ax.set_xticks(cycles)
                ax.set_xticklabels(cycles, rotation=45)
                ax.set_ylabel(f'Error {unitary_label}, $\\varepsilon$ (%)')
                ax.set_xlabel('Number of cycles, $m$', labelpad=0.1)
                ax.set_title(f'{filename_prefix}Control errors {mobjn} - {timestamp}')
                fig.subplots_adjust(*fig_margins, hspace=0.05)

                if savefig:
                    if fmts is None:
                        fmts = ['png']
                    fn = copy(filename)
                    if fn is None:
                        fn = f'Opt_res_ctrl_errs_{mobjn}_' \
                             f'{swpts.length(sweep_type["seqs"])}seqs_{timestamp}'
                    fn = f'{filename_prefix}{fn}'
                    if f'{fn}.png' in os.listdir(data_dict["folders"][0]):
                        fn = '{}--{:%Y%m%d_%H%M%S}'.format(fn, datetime.datetime.now())
                    for ext in fmts:
                        fn = f'{data_dict["folders"][0]}\\{fn}.{ext}'
                        fig.savefig(fn, dpi=600)
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception:
        plt.close(fig)
        traceback.print_exc()
        return


def plot_opt_res(data_dict, param_labels, savefig=True, fmts=None,
                 figure_width='1col', figure_height=3, filename=None,
                 filename_prefix='', numcols=None, numrows=None,
                 fig_margins=None, show=False, **params):
    try:
        timestamp = data_dict['timestamps'][0]
        meas_obj_names = hlp_mod.get_param('meas_obj_names', data_dict, **params)
        if meas_obj_names is None:
            meas_obj_names = hlp_mod.get_param_from_metadata_group(
                timestamp, 'meas_objs')
        d = hlp_mod.get_param('dim_hilbert', data_dict, raise_error=True,
                              **params)
        if d == 4:
            meas_obj_names = [",".join(meas_obj_names)]

        # for title
        sweep_type = hlp_mod.get_param('sweep_type', data_dict,
                                       default_value={'cycles': 0, 'seqs': 1})
        swpts = hlp_mod.get_measurement_properties(data_dict,
                                                   props_to_extract=['sp'])

        if fig_margins is None:
            fig_margins = [0.2, 0.14, 0.99, 0.99]

        rc_params = {
            'figure.figsize': (figure_width, figure_height),
        }
        with mpl.rc_context(rc=rc_params):
            for mobjn in meas_obj_names:
                opt_res = hlp_mod.get_param(f'{mobjn}.optimization_result',
                                            data_dict, raise_error=True,
                                            **params)
                xvals = np.array([int(nc) for nc in opt_res])
                or_vals = np.array(list(opt_res.values()))
                nr_params = or_vals.shape[1]
                if numcols is None or numrows is None:
                    numrows_list = [4, 3, 2, 1]
                    for numr in numrows_list:
                        if not nr_params % numr:
                            numrows = numr
                            numcols = nr_params // numr
                            break
                print(mobjn, numrows, numcols)

                fig, axs = plt.subplots(numrows, numcols, sharex=True)
                for i, ax in enumerate(axs):
                    ax.plot(xvals, or_vals[:, i], 'o')
                    ax.set_ylabel(param_labels[i])

                ax.set_xlabel('Number of cycles, $m$', labelpad=0.1)
                ax.set_xticks(xvals)
                ax.set_xticklabels(xvals, rotation=45)
                fig.align_labels()
                fig.subplots_adjust(*fig_margins, hspace=0.05)
                if savefig:
                    if fmts is None:
                        fmts = ['png']
                    fn = copy(filename)
                    if fn is None:
                        fn = f'Opt_res_{mobjn}_' \
                             f'{swpts.length(sweep_type["seqs"])}seqs_{timestamp}'
                    fn = f'{filename_prefix}{fn}'
                    if f'{fn}.png' in os.listdir(data_dict["folders"][0]):
                        fn = '{}--{:%Y%m%d_%H%M%S}'.format(fn, datetime.datetime.now())
                    for ext in fmts:
                        fn = f'{data_dict["folders"][0]}\\{fn}.{ext}'
                        fig.savefig(fn, dpi=600)
        if show:
            plt.show()
        else:
            plt.close(fig)
    except Exception:
        plt.close(fig)
        traceback.print_exc()
        return

