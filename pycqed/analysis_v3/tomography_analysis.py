import logging
log = logging.getLogger(__name__)

import itertools
import scipy as sp
import numpy as np
import qutip as qtp
import matplotlib as mpl
from collections import OrderedDict
from pycqed.analysis_v2 import tomography_qudev as tomo
from pycqed.analysis_v3 import plotting as plot_mod
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.analysis_v3 import data_extraction as dat_extr_mod
from pycqed.analysis_v3 import data_processing as dat_proc_mod
from copy import deepcopy

import sys
pp_mod.search_modules.add(sys.modules[__name__])


def standard_qubit_pulses_to_rotations(pulse_list):
    """
    Converts lists of n-tuples of standard PycQED single-qubit pulse names to
    the corresponding rotation matrices on the n-qubit Hilbert space.

    :param pulse_list: list of n-tuples. The tuples contain strings that should
        match the keys in standard_pulses dict below.
    :return list of len(pulse_list) qutip quantum objects representing the
        products of the pulses in each n-tuple.
    """
    standard_pulses = {
        'I': qtp.qeye(2),
        'X0': qtp.qeye(2),
        'Z0': qtp.qeye(2),
        'X180': qtp.sigmax(),
        'mX180': qtp.rotation(qtp.sigmax(), -np.pi),
        'Y180': qtp.sigmay(),
        'mY180': qtp.rotation(qtp.sigmay(), -np.pi),
        'X90': qtp.rotation(qtp.sigmax(), np.pi/2),
        'mX90': qtp.rotation(qtp.sigmax(), -np.pi/2),
        'Y90': qtp.rotation(qtp.sigmay(), np.pi/2),
        'mY90': qtp.rotation(qtp.sigmay(), -np.pi/2),
        'Z90': qtp.rotation(qtp.sigmaz(), np.pi/2),
        'mZ90': qtp.rotation(qtp.sigmaz(), -np.pi/2),
        'Z180': qtp.sigmaz(),
        'mZ180': qtp.rotation(qtp.sigmaz(), -np.pi),
        'CZ': qtp.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]])
    }
    rotations = [qtp.tensor(*[standard_pulses[pulse] for pulse in qb_pulses])
                 for qb_pulses in pulse_list]
    for i in range(len(rotations)):
        rotations[i].dims = [[d] for d in rotations[i].shape]
    return rotations


def state_tomography_analysis(data_dict, keys_in,
                              estimation_types=('least_squares',
                                                'max_likelihood'), **params):
    """
    State tomography analysis. Extracts density matrices based on
        estimation_types, calculates purity, concurrence, and fidelity to
        rho_target, prepares probability table plot, density matrix plots,
        pauli basis plots.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
        data_dict for the data to be analyzed (expects thresholded shots)
    :param estimation_types: list of strings indicating the methods to use to
        construct the density matrix. It will do all the estimation types in
        this list.
    :param params: keyword argument.
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
            - basis_rots: list/tuple of strings specifying tomography rotations
                Ex: ('I', 'X90', 'Y90', 'X180'), ('I', 'X90', 'Y90')
            - n_readouts or CalibrationPoints + basis_rots + preselection
                condition. Number of segments including preselection.
                If n_readouts is not provided it will try to estimate it from
                CalibrationPoints + basis_rots + preselection condition.
                n_readouts is the total number of readouts including
                preselection.
        Other possible keyword arguments:
            - do_preselection or preparation_params. If the former is not
                provided, it will try to take it from preparation_params.
                If preparation_params are not found, it will default to False.
                Specifies whether to do preselection on the data.
            - observables: measurement observables, see docstring of hlp_mod.
                get_observables. If not provided, it will default to hlp_mod.
                get_observables. See required input params there.
            - rho_target (qutip Qobj; default: None): target density matrix or
                state vector as qutip object
            - prepare_plotting (bool, default: True): whether to prepare
                plot dicts
            - do_plotting (bool, default: True): whether to plot the
                plot dicts
            - do_bootstrapping (bool, default: False): whether to run the
                bootstrapping statistical error estimation (see the function
                bootstrapping_state_tomography)
                - Nbstrp (int): REQUIRED IF do_bootstrapping IS TRUE! Number of
                    bootstrapping cycles, ie sample size for estimating errors
    :return: adds to data_dict the following quantities:
        - if not already there: basis_rots, n_readouts, do_preselection,
            observables
        - probability_table, probability_table_filtered
        - measurement_ops, cov_matrix_meas_obs
        - all_measurement_results, all_measurement_operators,
            all_cov_matrix_meas_obs
        - est_type.rho, est_type.purity, (est_type.concurrence if
            len(meas_obj_names) == 2), (est_type.fidelity if rho_target
            is provided) for est_type in estimation_types
        - plot_dicts if prepare_plotting; figures, axes if do_plotting
        - Nbstrp and est_type.bootstrapping_fidelities for est_type in
            estimation_types if do_bootstrapping.

    Assumptions:
        - the data indicated by keys_in is assumed to be thresholded shots
    """
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)

    hlp_mod.pop_param('keys_out', data_dict, node_params=params)

    cp = hlp_mod.get_measurement_properties(data_dict, props_to_extract=['cp'],
                                            raise_error=False, **params)
    basis_rots = hlp_mod.get_param('basis_rots', data_dict,
                                   raise_error=True, **params)
    if hlp_mod.get_param('basis_rots', data_dict) is None:
        hlp_mod.add_param('basis_rots', basis_rots, data_dict, **params)

    do_preselection = hlp_mod.get_param('do_preselection', data_dict,
                                        **params)
    if do_preselection is None:
        prep_params = hlp_mod.get_param('preparation_params', data_dict,
                                        default_value={}, **params)
        do_preselection = \
            prep_params.get('preparation_type', 'wait') == 'preselection'
        hlp_mod.add_param('do_preselection', do_preselection, data_dict,
                          **params)

    # get number of readouts
    n_readouts = hlp_mod.get_param('n_readouts', data_dict, **params)
    if n_readouts is None:
        n_readouts = (do_preselection + 1) * (
                len(basis_rots)**len(meas_obj_names) +
                (len(cp.states) if cp is not None else 0))
        hlp_mod.add_param('n_readouts', n_readouts, data_dict, **params)

    # get observables
    observables = hlp_mod.get_param('observables', data_dict, **params)
    if observables is None:
        hlp_mod.get_observables(data_dict, keys_out=['observables'],
                                **params)
        observables = hlp_mod.get_param('observables', data_dict)

    # get probability table
    keys_in_extra = hlp_mod.get_param('keys_in_extra', data_dict,
                                      default_value=[], **params)
    dat_proc_mod.calculate_probability_table(data_dict, keys_in=keys_in+keys_in_extra,
                                             keys_out=['probability_table'],
                                             n_readouts=n_readouts,
                                             observables=observables, **params)

    # get measurement_ops and cov_matrix_meas_obs
    measurement_ops = hlp_mod.get_param('measurement_ops', data_dict, **params)
    correct_readout = hlp_mod.get_param('correct_readout', data_dict, **params)
    if measurement_ops is None:
        if cp is not None and (correct_readout or correct_readout is None):
            dat_proc_mod.calculate_meas_ops_and_covariations_cal_points(
                data_dict, keys_in, n_readouts=n_readouts,
                keys_out=['measurement_ops', 'cov_matrix_meas_obs'],
                observables=observables, **params)
        else:
            dat_proc_mod.calculate_meas_ops_and_covariations(
                data_dict, keys_out=['measurement_ops', 'cov_matrix_meas_obs'],
                meas_obj_names=meas_obj_names,
                observables=observables,
                correct_readout=correct_readout)
    else:
        if hlp_mod.get_param('measurement_ops', data_dict) is None:
            hlp_mod.add_param('measurement_ops', measurement_ops, data_dict,
                              **params)
        cov_matrix_meas_obs = hlp_mod.get_param('cov_matrix_meas_obs',
                                                data_dict, **params)
        if cov_matrix_meas_obs is None:
            hlp_mod.add_param('cov_matrix_meas_obs',
                              np.diag(np.ones(len(meas_obj_names)**2)),
                              data_dict, **params)
        else:
            if hlp_mod.get_param('cov_matrix_meas_obs', data_dict) is None:
                hlp_mod.add_param('cov_matrix_meas_obs', measurement_ops,
                                  data_dict, **params)

    # get all measurement ops, measurement results, and covariance matrices
    all_msmt_ops_results_omegas(data_dict, observables, **params)

    # get density matrices, purity, fidelity, concurrence
    density_matrices(data_dict, estimation_types, **params)

    # plotting
    prepare_plotting = hlp_mod.pop_param('prepare_plotting', data_dict,
                                        default_value=True, node_params=params)
    do_plotting = hlp_mod.pop_param('do_plotting', data_dict,
                                    default_value=True, node_params=params)
    if prepare_plotting:
        prepare_prob_table_plot(data_dict, do_preselection, **params)
        for i, estimation_type in enumerate(estimation_types):
            prepare_density_matrix_plot(data_dict, estimation_type,
                                        plot_rho_target=(i == 0), **params)
            prepare_pauli_basis_plot(data_dict, estimation_type, **params)
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict, keys_in=list(
            data_dict['plot_dicts']), **params)

    # error estimation with boostrapping
    if hlp_mod.get_param('do_bootstrapping', data_dict, default_value=False,
                         **params):
        hlp_mod.pop_param('do_bootstrapping', data_dict, default_value=False,
                          node_params=params)
        bootstrapping_state_tomography(data_dict, keys_in, **params)


def all_msmt_ops_results_omegas(data_dict, observables, probability_table=None,
                                **params):
    """
    Calculates all_measurement_results, all_measurement_operators, and
        all_cov_matrix_meas_obs from measurement_ops, cov_matrix_meas_obs
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param observables: measurement observables, see docstring of
        hlp_mod.get_observables.
    :param probability_table: dictionary with observables as keys and
        normalized counts for each segment (excluding preselection) as values
        (see dat_proc_mod.calculate_probability_table).
        IF NONE, IT MUST EXIST IN data_dict.
    :param params: keyword arguments
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
            - basis_rots (see docstring of state_tomography)
            - measurement_ops: list of array corresponding to measurement
                operators
            - cov_matrix_meas_obs: covariance matrix
        Other possible keyword arguments used if both probability_table and
        prob_table_filter are None:
            - do_preselection or preparation_params. If the former is not
                provided, it will try to take it from preparation_params.
                If preparation_params are not found, it will default to False.
                Specifies whether to do preselection on the data.
            - prob_table_filter: filter for the probability table. If not given
                it will calculate it from meas_obj_names + basis_rots +
                preselection condition
    :return: adds to data_dict:
        - all_measurement_results: itertools.chain of the probability table
        - all_measurement_operators: see tomography_qudev.
            rotated_measurement_operators
        - and all_cov_matrix_meas_obs: sp.linalg.block_diag(
            *[Omega] * len(probability_table[0]))
    """
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    basis_rots = hlp_mod.get_param('basis_rots', data_dict,
                                   raise_error=True, **params)
    if probability_table is None:
        prob_table_filter = hlp_mod.get_param('prob_table_filter', data_dict,
                                              **params)
        if prob_table_filter is None:
            do_preselection = hlp_mod.get_param(
                'do_preselection', data_dict, default_value=
                 hlp_mod.get_param('preparation_params', data_dict,
                                   default_value={}, **params).get(
                     'preparation_type', 'wait') == 'preselection', **params)
            def prob_table_filter(prob_table, pre=do_preselection,
                                  basis_rots=basis_rots, n=len(meas_obj_names)):
                prob_table = np.array(list(prob_table.values())).T
                return prob_table[pre: (pre+1)*len(basis_rots)**n: (pre+1)]
        dat_proc_mod.filter_data(data_dict, keys_in=['probability_table'],
                                 keys_out=['probability_table_filtered'],
                                 data_filter=prob_table_filter, **params)
        probability_table = hlp_mod.get_param('probability_table_filtered',
                                              data_dict)

    try:
        preselection_obs_idx = list(observables.keys()).index('pre')
    except ValueError:
        preselection_obs_idx = None
    observabele_idxs = [i for i in range(len(observables))
                        if i != preselection_obs_idx]
    prob_table = probability_table.T[observabele_idxs]
    prob_table = prob_table.T
    for i, v in enumerate(prob_table):
        prob_table[i] = v / v.sum()
    prob_table = prob_table.T


    pulse_list = list(itertools.product(basis_rots,
                                        repeat=len(meas_obj_names)))
    rotations = tomo.standard_qubit_pulses_to_rotations(pulse_list)
    rotations = [qtp.Qobj(U) for U in rotations]

    msmt_ops = hlp_mod.get_param('measurement_ops', data_dict)
    msmt_ops = [qtp.Qobj(F) for F in msmt_ops]
    all_msmt_ops = tomo.rotated_measurement_operators(rotations, msmt_ops)
    all_msmt_ops = list(itertools.chain(*np.array(
        all_msmt_ops, dtype=np.object).T))
    hlp_mod.add_param('all_measurement_operators', all_msmt_ops, data_dict,
                      **params)

    all_msmt_res = np.array(list(itertools.chain(*prob_table.T)))
    hlp_mod.add_param('all_measurement_results', all_msmt_res, data_dict,
                      **params)

    omegas = hlp_mod.get_param('cov_matrix_meas_obs', data_dict)
    all_omegas = sp.linalg.block_diag(*[omegas] * len(prob_table[0]))
    hlp_mod.add_param('all_cov_matrix_meas_obs', all_omegas, data_dict,
                      **params)


def density_matrices(data_dict,
                     estimation_types=('least_squares', 'max_likelihood'),
                     **params):
    """
    Estimates density matrices from all_measurement_results,
    all_measurement_operators, and all_cov_matrix_meas_obs using the
    estimation methods in estimation_types.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param estimation_types: list of strings indicating the methods to use to
        construct the density matrix. It will do all the estimation types in
        this list.
    :param params: keyword arguments:
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
            - all_measurement_results, all_measurement_operators
                (see all_msmt_ops_results_omegas)
            - all_cov_matrix_meas_obs if use_covariance_matrix is True
        Other possible keyword arguments:
            - rho_target (qutip Qobj; default: None): target density matrix or
                state vector as qutip object
            - use_covariance_matrix (bool; default: False): whether to use the
                covariance matrices in the estimations
            - tolerance (float; default: None): custom tolerance threshold for
                iterative maximum likelihood estimation.
            - iterations (int; default: None): custom iteration threshold for
                iterative maximum likelihood estimation.
            - rho_guess (qutip Qobj; default: None): initial rho used in Maximum
                likelihood estimation (mle) or iterative mle instead of default
    :return: adds to data_dict
        - rho_target if not already there
        - est_type.purity, (est_type.concurrence if len(meas_obj_names) == 2),
            (est_type.fidelity if rho_target is provided) for est_type in
            estimation_types
    """
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)

    rho_target = hlp_mod.get_param('rho_target', data_dict, **params)
    if 'rho_target' not in data_dict:
        hlp_mod.add_param('rho_target', rho_target, data_dict, **params)
    for estimation_type in estimation_types:
        if estimation_type == 'least_squares':
            rho_ls = tomo.least_squares_tomography(
                hlp_mod.get_param('all_measurement_results', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('all_measurement_operators', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('all_cov_matrix_meas_obs', data_dict)
                if hlp_mod.get_param('use_covariance_matrix', data_dict,
                                     default_value=False, **params) else None)
            hlp_mod.add_param('least_squares.rho', rho_ls, data_dict, **params)
        elif estimation_type == 'max_likelihood':
            rho_guess = hlp_mod.get_param('rho_guess', data_dict, **params)
            if rho_guess is None:
                rho_guess = hlp_mod.get_param(
                    'least_squares.rho', data_dict, raise_error=True,
                    error_message='Maximum likelihood estimation needs a guess '
                                  'rho but neither a rho_guess not a '
                                  'least_squares.rho was found.', **params)
            rho_mle = tomo.mle_tomography(
                hlp_mod.get_param('all_measurement_results', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('all_measurement_operators', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('all_cov_matrix_meas_obs', data_dict)
                if hlp_mod.get_param('use_covariance_matrix', data_dict,
                                     default_value=False, **params) else None,
                rho_guess=rho_guess)
            hlp_mod.add_param('max_likelihood.rho', rho_mle, data_dict, **params)
        elif estimation_type == 'iterative_mle':
            rho_imle = tomo.imle_tomography(
                hlp_mod.get_param('all_measurement_results', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('all_measurement_operators', data_dict,
                                  raise_error=True, **params),
                hlp_mod.get_param('iterations', data_dict, **params),
                hlp_mod.get_param('tolerance', data_dict, **params),
                hlp_mod.get_param('rho_guess', data_dict, **params))
            hlp_mod.add_param('iterative_mle.rho', rho_imle, data_dict, **params)
        elif estimation_type == 'pauli_values':
            rho_pauli = tomo.pauli_values_tomography(
                hlp_mod.get_param('all_measurement_results', data_dict,
                                  raise_error=True, **params),
                [qtp.Qobj(F) for F in hlp_mod.get_param('measurement_ops',
                                                        data_dict)],
                hlp_mod.get_param('basis_rots', data_dict))
            hlp_mod.add_param('pauli_values.rho', rho_pauli, data_dict, **params)
        else:
            raise ValueError(f'Unknown estimation_type "{estimation_type}."')

        rho_meas = hlp_mod.get_param(f'{estimation_type}.rho', data_dict,
                                     **params)
        if rho_meas is not None:
            hlp_mod.add_param(f'{estimation_type}.purity',
                              (rho_meas*rho_meas).tr().real, data_dict,
                              **params)
            if rho_target is not None:
                hlp_mod.add_param(f'{estimation_type}.fidelity',
                                  fidelity(rho_meas, rho_target), data_dict,
                                  **params)
            if len(meas_obj_names) == 2:
                hlp_mod.add_param(f'{estimation_type}.concurrence',
                                  concurrence(rho_meas), data_dict,
                                  **params)


def fidelity(rho1, rho2):
    """
    Returns the fidelity between the two quantum states rho1 and rho2.
    Uses the Jozsa definition (the smaller of the two), not the Nielsen-Chuang
    definition.

    F = Tr(√((√rho1) rho2 √(rho1)))^2

    :param rho1: qtp.Qobj of measured rho
    :param rho2: qtp.Qobj of target rho

    """
    rho1 = tomo.convert_to_density_matrix(rho1).full()
    rho2 = tomo.convert_to_density_matrix(rho2).full()
    return sp.linalg.sqrtm(
        sp.linalg.sqrtm(rho1) @ rho2 @ sp.linalg.sqrtm(rho1)).trace().real ** 2


def concurrence(rho):
    """
    Calculates the concurrence of the two-qubit state rho given in the
    qubits' basis according to https://doi.org/10.1103/PhysRevLett.78.5022

    :param rho: qtp.Qobj of rho
    """
    rho = tomo.convert_to_density_matrix(rho).full()
    # convert to bell basis
    b = [np.sqrt(0.5)*np.array(l) for l in
         [[1, 0, 0, 1], [1j, 0, 0, -1j], [0, 1j, 1j, 0], [0, 1, -1, 0]]]
    rhobell = np.zeros((4, 4), dtype=np.complex)
    for i in range(4):
        for j in range(4):
            rhobell[i, j] = b[j].conj().T @ rho @ b[i]
    R = sp.linalg.sqrtm(
        sp.linalg.sqrtm(rhobell) @ rhobell.conj() @ sp.linalg.sqrtm(rhobell))
    counter = 0
    while counter < 5:
        # hack needed because of a strange bug on my computer where
        # np.linalg.eigvals sometimes fails the first time.
        try:
            C = max(0, 2*np.linalg.eigvals(R).max() - R.trace())
            break
        except Exception as e:
            if counter != 4:
                pass
            else:
                raise e
        counter += 1
    if not isinstance(C, int):
        C = C.real
    return C


def prepare_prob_table_plot(data_dict, exclude_preselection=False, **params):
    """
    Prepares a plot of the probability table.
    :param data_dict: OrderedDict containing data to be plotted and where
        plot_dicts is to be stored
    :param exclude_preselection: whether to exclude preselection segments
    :param params: keyword arguments:
        Expects to find either in data_dict or in params:
            - probability_table: dictionary with observables as keys and
                normalized counts for each segment as values
                (see dat_proc_mod.calculate_probability_table).
            - meas_obj_names: list of measurement object names
            - observables: measurement observables (see docstring of hlp_mod.
                get_observables).
            - timestamps: list with the measurement timestamp
        Other possible keyword arguments:
            - prob_table_filter (func; default: [1::2] if exclude_preselection
                else no filter): function for filtering probability_table
            - obs_filter (func; default: np.arange(len(observables))): function
                for filtering observables
    :return: adds to data_dict: plot_dicts
    """
    plot_dicts = OrderedDict()
    figures_prefix = hlp_mod.get_param('figures_prefix', data_dict,
                                       default_value='', **params)
    if len(figures_prefix):
        figures_prefix += '_'

    probability_table = hlp_mod.get_param('probability_table', data_dict,
                                          raise_error=True, **params)
    probability_table = np.array(list(probability_table.values())).T
    observables = hlp_mod.get_param('observables', data_dict,
                                    raise_error=True, **params)
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)

    # colormap which has a lot of contrast for small and large values
    v = [0, 0.1, 0.2, 0.8, 1]
    c = [(1, 1, 1),
         (191/255, 38/255, 11/255),
         (155/255, 10/255, 106/255),
         (55/255, 129/255, 214/255),
         (0, 0, 0)]
    cdict = {'red':   [(v[i], c[i][0], c[i][0]) for i in range(len(v))],
             'green': [(v[i], c[i][1], c[i][1]) for i in range(len(v))],
             'blue':  [(v[i], c[i][2], c[i][2]) for i in range(len(v))]}
    cm = mpl.colors.LinearSegmentedColormap('customcmap', cdict)

    prob_table_filter = hlp_mod.get_param('prob_table_filter', data_dict,
                                          **params)
    if prob_table_filter is not None:
        plt_data = prob_table_filter(probability_table).T
    else:
        if exclude_preselection:
            plt_data = probability_table[1::2].T
        else:
            plt_data = probability_table.T
    ylist = list(range(len(plt_data.T)))

    obs_filter = hlp_mod.get_param(
        'obs_filter', data_dict, default_value=np.arange(len(observables)),
        **params)
    plt_data = plt_data[obs_filter]

    timestamps = hlp_mod.get_param('timestamps', data_dict, raise_error=True,
                                   **params)
    if len(timestamps) > 1:
        title = f'{timestamps[0]} - {timestamps[-1]} {",".join(meas_obj_names)}'
    else:
        title = f'{timestamps[-1]} {",".join(meas_obj_names)}'

    plot_dicts[f'{figures_prefix}counts_table_{"".join(meas_obj_names)}'] = {
        'axid': "ptable",
        'plotfn': 'plot_colorx',
        'plotsize': [6.4, 4.8],
        'xvals': np.arange(len(observables))[obs_filter],
        'yvals': np.array(len(observables)*[ylist]),
        'zvals': plt_data,
        'xlabel': "Channels",
        'ylabel': "Segments",
        'zlabel': "Counts",
        'zrange': [0, 1],
        'title': title,
        'xunit': None,
        'yunit': None,
        'xtick_loc': np.arange(len(observables))[obs_filter],
        'xtick_labels': list(np.array(list(observables.keys()))[obs_filter]),
        'origin': 'upper',
        'cmap': cm,
        'aspect': 'equal'
    }

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')


def prepare_density_matrix_plot(data_dict, estimation_type='least_squares',
                                plot_rho_target=True, leakage=None, **params):
    """
    Prepares plot of the density matrix estimated with method estimation_type.
    :param data_dict: OrderedDict containing data to be plotted and where
        plot_dicts is to be stored
    :param estimation_type: string indicating the method that was used to
        estimate the density matrix. Assumes estimation_type.rho exists in
        data_dict.
    :param plot_rho_target: whether to prepare a separate figure for rho_target
    :param leakage: dict of leakages as returned by
        data_processing.py/extract_leakage_classified_shots
    :param params: keyword arguments:
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
        Other possible keyword arguments:
            - rho_ticklabels (list of strings; default: kets of basis states):
                x- and y-ticklabels
            - rho_colormap (colormap; default: plot_mod.default_phase_cmap()):
                colormap
            - rho_target (qutip Qobj; default: None): target density matrix or
                state vector as qutip object
    :return: adds to data_dict: plot_dicts

    Assumptions:
        - estimation_type.rho exists in data_dict
    """
    plot_dicts = OrderedDict()
    figures_prefix = hlp_mod.get_param('figures_prefix', data_dict,
                                       default_value='', **params)
    if len(figures_prefix):
        figures_prefix += '_'

    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    d = 2**len(meas_obj_names)
    xtick_labels = hlp_mod.get_param('rho_ticklabels', data_dict, **params)
    ytick_labels = hlp_mod.get_param('rho_ticklabels', data_dict, **params)
    if 2 ** (d.bit_length() - 1) == d:
        nr_qubits = d.bit_length() - 1
        fmt_string = '{{:0{}b}}'.format(nr_qubits)
        labels = [fmt_string.format(i) for i in range(2 ** nr_qubits)]
        if xtick_labels is None:
            xtick_labels = ['$|' + lbl + r'\rangle$' for lbl in labels]
        if ytick_labels is None:
            ytick_labels = [r'$\langle' + lbl + '|$' for lbl in labels]

    cmap = hlp_mod.get_param('rho_colormap', data_dict,
                             default_value=plot_mod.default_phase_cmap(),
                             **params)

    rho_target = hlp_mod.get_param('rho_target', data_dict, **params)
    if rho_target is not None:
        rho_target = qtp.Qobj(rho_target)
        if rho_target.type == 'ket':
            rho_target = rho_target * rho_target.dag()
        elif rho_target.type == 'bra':
            rho_target = rho_target.dag() * rho_target
        if plot_rho_target:
            title = 'Target density matrix\n' + plot_mod.default_figure_title(
                data_dict, ','.join(meas_obj_names))
            plot_dicts[f'{figures_prefix}density_matrix_target'] = {
                'plotfn': 'plot_bar3D',
                '3d': True,
                '3d_azim': -35,
                '3d_elev': 35,
                'xvals': np.arange(d),
                'yvals': np.arange(d),
                'zvals': np.abs(rho_target.full()),
                'zrange': (0, 1),
                'color': (0.5 * np.angle(rho_target.full()) / np.pi) % 1.,
                'colormap': cmap,
                'bar_widthx': 0.5,
                'bar_widthy': 0.5,
                'xtick_loc': np.arange(d),
                'xtick_labels': xtick_labels,
                'ytick_loc': np.arange(d),
                'ytick_labels': ytick_labels,
                'ctick_loc': np.linspace(0, 1, 5),
                'ctick_labels': ['$0$', r'$\frac{1}{2}\pi$', r'$\pi$',
                                 r'$\frac{3}{2}\pi$', r'$2\pi$'],
                'clabel': 'Phase (rad)',
                'title': title,
                'bar_kws': dict(zorder=1),
            }

    rho_meas = hlp_mod.get_param(f'{estimation_type}.rho', data_dict,
                                 raise_error=True)
    if estimation_type == 'least_squares':
        base_title = 'Least squares fit of the density matrix\n'
    elif estimation_type == 'max_likelihood':
        base_title = 'Maximum likelihood fit of the density matrix\n'
    elif estimation_type == 'iterative_mle':
        base_title = 'Iterative maximum likelihood fit of the density matrix\n'
    elif estimation_type == 'pauli_values':
        base_title = 'Density matrix reconstructed from measured Pauli values\n'
    else:
        base_title = 'Density matrix\n'


    legend_entries = get_legend_artists_labels(data_dict,
                                               estimation_type=estimation_type,
                                               **params)

    title = base_title + plot_mod.default_figure_title(
        data_dict, ','.join(meas_obj_names))
    zvals = np.concatenate((np.abs(rho_target.full()),
                            np.abs(rho_meas.full())))
    color_tar = (0.5 * np.angle(rho_target.full()) / np.pi) % 1.
    color_meas = (0.5 * np.angle(rho_meas.full()) / np.pi) % 1.
    color = np.concatenate((1.1*np.ones_like(color_tar), color_meas))
    plot_dicts[f'{figures_prefix}density_matrix_' \
               f'{estimation_type}_{"".join(meas_obj_names)}'] = {
        'plotfn': 'plot_bar3D',
        '3d': True,
        '3d_azim': -35,
        '3d_elev': 35,
        'xvals': np.arange(d),
        'yvals': np.arange(d),
        'zvals': zvals,
        'zrange': (0, 1),
        'color': color,
        'colormap': cmap,
        'bar_widthx': 0.5,
        'bar_widthy': 0.5,
        'xtick_loc': np.arange(d),
        'xtick_labels': xtick_labels,
        'ytick_loc': np.arange(d),
        'ytick_labels': ytick_labels,
        'ctick_loc': np.linspace(0, 1, 5),
        'ctick_labels': ['$0$', r'$\frac{1}{2}\pi$', r'$\pi$',
                         r'$\frac{3}{2}\pi$', r'$2\pi$'],
        'clabel': 'Phase (rad)',
        'title': title,
        'do_legend': len(legend_entries),
        'legend_entries': legend_entries,
        'legend_kws': dict(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           ncol=2, frameon=False),
        'set_edgecolor': True
    }

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')


def prepare_pauli_basis_plot(data_dict, estimation_type='least_squares',
                             leakage=None, **params):
    """
    Prepares plot of the density matrix estimated with method estimation_type.
    :param data_dict: OrderedDict containing data to be plotted and where
        plot_dicts is to be stored
    :param estimation_type: string indicating the method that was used to
        estimate the density matrix. Assumes estimation_type.rho exists in
        data_dict.
    :param leakage: dict of leakages as returned by
        data_processing.py/extract_leakage_classified_shots
    :param params: keyword arguments:
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
        Other possible keyword arguments:
            - rho_target (qutip Qobj; default: None): target density matrix or
                state vector as qutip object
    :return: adds to data_dict: plot_dicts

    Assumptions:
        - estimation_type.rho exists in data_dict
    """
    plot_dicts = OrderedDict()
    figures_prefix = hlp_mod.get_param('figures_prefix', data_dict,
                                       default_value='', **params)
    if len(figures_prefix):
        figures_prefix += '_'

    rho_meas = hlp_mod.get_param(f'{estimation_type}.rho', data_dict,
                                 raise_error=True)
    rho_target = hlp_mod.get_param('rho_target', data_dict, **params)
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    nr_qubits = len(meas_obj_names)

    yexp = tomo.density_matrix_to_pauli_basis(rho_meas)
    labels = list(itertools.product(*[['I', 'X', 'Y', 'Z']]*nr_qubits))
    labels = [''.join(label_list) for label_list in labels]
    if nr_qubits == 1:
        order = [1, 2, 3]
    elif nr_qubits == 2:
        order = [1, 2, 3, 4, 8, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    elif nr_qubits == 3:
        order = [1, 2, 3, 4, 8, 12, 16, 32, 48] + \
                [5, 6, 7, 9, 10, 11, 13, 14, 15] + \
                [17, 18, 19, 33, 34, 35, 49, 50, 51] + \
                [20, 24, 28, 36, 40, 44, 52, 56, 60] + \
                [21, 22, 23, 25, 26, 27, 29, 30, 31] + \
                [37, 38, 39, 41, 42, 43, 45, 46, 47] + \
                [53, 54, 55, 57, 58, 59, 61, 62, 63]
    else:
        order = np.arange(4**nr_qubits)[1:]
    if estimation_type == 'least_squares':
        fit_type = 'least squares fit\n'
    elif estimation_type == 'max_likelihood':
        fit_type = 'maximum likelihood estimation\n'
    elif estimation_type == 'iterative_mle':
        fit_type = 'iterative maximum likelihood estimation'
    elif estimation_type == 'pauli_values':
        fit_type = 'measured pauli values'
    else:
        fit_type = '\n'

    legend_entries = get_legend_artists_labels(data_dict,
                                               estimation_type=estimation_type,
                                               **params)
    figure_name = f'{figures_prefix}' \
                  f'pauli_basis_{estimation_type}_{"".join(meas_obj_names)}'
    plot_dicts[figure_name] = {
        'plotfn': 'plot_bar',
        'plotsize': (4.5, 3),
        'xcenters': np.arange(len(order)),
        'xwidth': 0.4,
        'xrange': (-1, len(order)),
        'yvals': np.array(yexp)[order],
        'xlabel': r'Pauli operator, $\hat{O}$',
        'ylabel': r'Expectation value, $\mathrm{Tr}(\hat{O} \hat{\rho})$',
        'title': 'Pauli operators, ' + fit_type +
                 plot_mod.default_figure_title(data_dict,
                                               ','.join(meas_obj_names)),
        'yrange': (-1.1, 1.1),
        'xtick_loc': np.arange(4**nr_qubits - 1),
        'xtick_rotation': 90,
        'xtick_labels': np.array(labels)[order],
        'bar_kws': dict(zorder=10),
        'setlabel': 'Fit to experiment',
        'do_legend': True
    }
    if nr_qubits > 2:
        plot_dicts[figure_name]['plotsize'] = (10, 5)

    if rho_target is not None:
        rho_target = qtp.Qobj(rho_target)
        ytar = tomo.density_matrix_to_pauli_basis(rho_target)
        plot_dicts[f'{figures_prefix}pauli_basis_target_{estimation_type}_' \
                   f'{"".join(meas_obj_names)}'] = {
            'plotfn': 'plot_bar',
            'fig_id': figure_name,
            'xcenters': np.arange(len(order)),
            'xwidth': 0.8,
            'yvals': np.array(ytar)[order],
            'xtick_loc': np.arange(len(order)),
            'xtick_labels': np.array(labels)[order],
            'bar_kws': dict(color='0.8', zorder=0),
            'setlabel': 'Target values',
            'legend_entries': legend_entries,
            'legend_kws': dict(loc='center left', bbox_to_anchor=(1, 0.5),
                               ncol=1, frameon=False),
            'do_legend': True
        }
    else:
        plot_dicts[figure_name].update({
            'legend_entries': legend_entries,
            'legend_kws': dict(loc='center left', bbox_to_anchor=(1, 0.5),
                               ncol=2, frameon=False)})

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')


def get_legend_artists_labels(data_dict, estimation_type='least_squares',
                              leakage=None, **params):
    rho_target = hlp_mod.get_param('rho_target', data_dict, **params)
    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    d = len(meas_obj_names)**2

    empty_artist = mpl.patches.Rectangle((0, 0), 0, 0, visible=False)
    purity = hlp_mod.get_param(f'{estimation_type}.purity', data_dict)
    legend_entries = []
    if purity is not None:
        legend_entries += [(empty_artist,
                            r'Purity, $Tr(\rho^2) = {:.1f}\%$'.format(
                                100 * purity))]
    if rho_target is not None:
        fidelity = hlp_mod.get_param(f'{estimation_type}.fidelity', data_dict)
        if fidelity is not None:
            legend_entries += [
                (empty_artist, r'Fidelity, $F = {:.1f}\%$'.format(
                    100 * fidelity))]
    if d == 4:
        concurrence = hlp_mod.get_param(f'{estimation_type}.concurrence',
                                        data_dict)
        if concurrence is not None:
            legend_entries += [
                (empty_artist, r'Concurrence, $C = {:.2f}$'.format(
                    concurrence))]

    if leakage is None:
        keys_in_leakage = hlp_mod.get_param('keys_in_leakage', data_dict,
                                            **params)
        if keys_in_leakage is not None:
            leakage = hlp_mod.get_param(keys_in_leakage[0], data_dict)

        if leakage is not None:
            legend_entries += [
                (empty_artist, f'Leakage, $L_{{{key}}} = {100*leak:.2f}\%$')
                for key, leak in leakage.items()]
    return legend_entries


def process_tomography_analysis(data_dict, Uideal=None,
                                n_qubits=None, prep_pulses_list=None,
                                estimation_types=('least_squares',
                                                  'max_likelihood'),
                                verbose=False, **params):
    """
    Process tomography analysis. Extracts chi and error of gate_of_interest or
    by comparing to Uideal.
    :param data_dict: OrderedDict containing the keys [''.join(pp) for pp in
         prep_pulses_list]. The values corresponding to these keys must either:
         - be data_dicts from running state tomography analysis for each of
         these prep_pulses
         - or must contain est_type.rho for est_type in estimation_types (i.e.
         the density matrices from doing state tomography for each prep state)
         - or must contain the key measured_rhos containing a dict of the form
         {est_type.rho: (list of meas ops) for est_type in estimation_types}
    :param Uideal: qutip Qobj of the ideal unitary operator for the process.
        Can also be specified with process_name, see keyword arguments below.
    :param n_qubits: number of qubits
    :param prep_pulses_list: list of tuples with length nr_qubits contanining
         strings indicating the preparation pulses for each state tomography
         measurement. If not specified, it will be constructed from product of
         basis_rots.
    :param estimation_types: list of strings indicating the methods that were
         used to construct the density matrices stores in data_dict[prep_pulses].
         This function will do process tomo for all the estimation types in
         this list.
    :param verbose: whether to show progress print statements
    :param params: keyword arguments
        Expects to find either in data_dict or in params:
            - only if n_qubits is None:
                - meas_obj_names: list of measurement object names
            - only if prep_pulses_list is None:
                - basis_rots: list/tuple of strings specifying the list of
                pulse names used to construct the prep_pulses list
                Ex: ('I', 'X90', 'Y90', 'X180'), ('I', 'X90', 'Y90')
        Other possible keyword arguments:
            - measured_rhos as {est_type.rho: list of meas ops for est_type
            in estimation_types}
    :return: adds to data_dict:
        - chi_{gate_name}.{estimation_type} and
            measured_error_{gate_name}.{estimation_type} for estimation_type
            in estimation_types.
        -  process_name: name of the gate for which the error is estimated.
         MUST CORRESPOND TO Uideal IF THE LATTER IS PROVIDED, since gate_name
         will be used in the key name for storing the results
    """
    if n_qubits is None:
        meas_obj_names = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
            **params)
        n_qubits = len(meas_obj_names)
    if Uideal is None:
        process_name = hlp_mod.get_param('process_name', data_dict, **params)
        if process_name == 'CZ':
            Uideal = qtp.qip.operations.cphase(np.pi)
        elif process_name == 'CNOT':
            Uideal = qtp.qip.operations.cnot()
        else:
            raise ValueError(f'Unknown gate of interest {process_name}. '
                             f'Please provide the process unitary, Uideal.')
    print(Uideal)
    Uideal = qtp.to_chi(Uideal)/16

    if prep_pulses_list is None:
        basis_rots = hlp_mod.get_param(
            'basis_rots', data_dict, raise_error=True,
            error_message='Either prep_pulses_list or basis_rots needs to be '
                          'provided.', **params)
        if hlp_mod.get_param('basis_rots', data_dict) is None:
            hlp_mod.add_param('basis_rots', basis_rots, data_dict, **params)
        prep_pulses_list = list(itertools.product(basis_rots, repeat=n_qubits))

    meas_density_matrices = hlp_mod.get_param('measured_rhos', data_dict,
                                              default_value={}, **params)

    for estimation_type in estimation_types:
        # get lambda array
        if verbose:
            print()
            print(f'From {estimation_type} estimation')
            print('Getting lambda array')

        # get measured density matrices
        measured_rhos = meas_density_matrices.get(estimation_type, None)
        if measured_rhos is None:
            measured_rhos = len(prep_pulses_list) * ['']
            for i, prep_pulses in enumerate(prep_pulses_list):
                prep_str = ''.join(prep_pulses)
                if verbose:
                    print(prep_str)
                measured_rhos[i] = \
                    hlp_mod.get_param(
                        f'{prep_str}.{estimation_type}.rho',
                        data_dict, raise_error=True,
                        error_message=f'Data for preparation pulses '
                                      f'{prep_str} was not found in '
                                      f'data_dict.').full().flatten()
        else:
            for i, mrho in enumerate(measured_rhos):
                if isinstance(mrho, qtp.qobj.Qobj):
                    measured_rhos[i] = mrho.full().flatten()
                elif mrho.ndim > 1:
                    measured_rhos[i] = mrho.flatten()
        measured_rhos = np.asarray(measured_rhos)

        # get density matrices for the preparation states
        U1s = {pulse: standard_qubit_pulses_to_rotations([(pulse,)])[0]
               for pulse in ['X90', 'Y90', 'mX90', 'mY90', 'I',
                             'X180', 'Y180', 'mY180', 'mX180']}
        preped_rhos = len(prep_pulses_list) * ['']
        preped_rhos_flatten = len(prep_pulses_list) * ['']
        for i, prep_pulses in enumerate(prep_pulses_list):
            prep_str = prep_pulses
            if not isinstance(prep_pulses, str):
                prep_str = ''.join(prep_pulses)
            if verbose:
                print(prep_str, [U1s[pp] for pp in prep_pulses])
            psi_target = (qtp.tensor([U1s[pp] for pp in prep_pulses]) *
                          qtp.tensor(n_qubits*[qtp.basis(2)]))
            rho_target = psi_target*psi_target.dag()
            preped_rhos[i] = rho_target
            preped_rhos_flatten[i] = rho_target.full().flatten()

        preped_rhos_flatten = np.asarray(preped_rhos_flatten)
        lambda_array = np.dot(measured_rhos, np.linalg.inv(preped_rhos_flatten))

        # get beta array
        if verbose:
            print('Geting beta array')
        standard_nqb_pauli_labels = list(itertools.product(
            ['I', 'X180', 'Y180', 'Z180'], repeat=n_qubits))
        standard_nqb_pauli_ops = standard_qubit_pulses_to_rotations(
            standard_nqb_pauli_labels)
        if verbose:
            print('len(standard_nqb_pauli_labels) ',
                  len(standard_nqb_pauli_labels))
            print('len(standard_nqb_pauli_ops) ',
                  len(standard_nqb_pauli_ops))

        prepared_rhos_rotated = []
        cnt = 0
        for i, prepared_rho in enumerate(preped_rhos):
            for plhs in standard_nqb_pauli_ops:
                plhs.dims = 2*[n_qubits*[2]]
                for prhs in standard_nqb_pauli_ops:
                    prhs.dims = 2*[n_qubits*[2]]
                    cnt += 1
                    prepared_rhos_rotated += [
                        (plhs*prepared_rho*prhs.dag()).full().flatten()]
        prepared_rhos_rotated = np.asarray(prepared_rhos_rotated)
        if verbose:
            print('prepared_rhos_rotated.shape ', prepared_rhos_rotated.shape)
            print('preped_rhos_flatten.shape ', preped_rhos_flatten.shape)
        beta_array = np.dot(prepared_rhos_rotated,
                            np.linalg.inv(preped_rhos_flatten))

        # get chi matrix
        if verbose:
            print()
            print('Getting chi matrix')
        chunck_size = len(standard_nqb_pauli_ops)**2
        beta_array_reshaped = np.zeros(shape=(len(preped_rhos)**2, chunck_size),
                                       dtype='complex128')
        for i in range(len(preped_rhos)):
            beta_array_reshaped[i*len(preped_rhos): (i+1)*len(preped_rhos), :] = \
                beta_array[i*chunck_size:(i+1)*chunck_size, :].T
            if verbose:
                print(beta_array[i*chunck_size:(i+1)*chunck_size, :].T.shape)
        if verbose:
            print('beta_array_reshaped.shape ', beta_array_reshaped.shape)
            print('lambda_array.flatten().size ', lambda_array.flatten().size)

        chi = np.linalg.solve(beta_array_reshaped, lambda_array.flatten())
        chi = chi.reshape(Uideal.shape)
        chi_qtp = qtp.Qobj(chi, dims=Uideal.dims)

        # add found chi matrix to data_dict
        hlp_mod.add_param(f'chi_{process_name}.{estimation_type}',
                          chi_qtp.full(), data_dict, **params)

        # add gate error to data_dict
        hlp_mod.add_param(f'measured_error_{process_name}.{estimation_type}',
                          1-np.real(qtp.process_fidelity(chi_qtp, Uideal)),
                          data_dict, **params)


def bootstrapping(data_dict=None, keys_in=None, measured_data=None, **params):
    """
    Does one round of resampling of measured_data using a uniform distribution.
    :param data_dict: OrderedDict containing data to be processed
    :param keys_in: list of channel names or dictionary paths leading to
            data to be processed of the form (n_readouts*n_shots, n_qubits).
    :param measured_data: array of shape (n_readouts*n_shots, n_qubits)
        containing raw data shots
    :param params: keyword arguments:
        n_readouts (required): number of segments including preselection
        n_shots (required): number of data shots per segment
        preselection: whether preselection was used

    :return: array of shape (n_readouts*n_shots, n_qubits) with resampled raw
        data shots.
    """
    if measured_data is None:
        if data_dict is None or keys_in is None:
            raise ValueError('Make sure both data_dict and keys_in are '
                             'specified. Or provide measured_data.')
        data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
        measured_data = np.concatenate([arr[:, np.newaxis] for arr in
                                        list(data_to_proc_dict.values())],
                                       axis=1)

    if data_dict is None:
        data_dict = {}
    n_readouts = hlp_mod.get_param('n_readouts', data_dict, raise_error=True,
                                   **params)
    n_shots = hlp_mod.get_param('n_shots', data_dict, raise_error=True,
                                **params)
    preselection = hlp_mod.get_param('preselection', data_dict,
                                     default_value=False, **params)

    sample_i = np.zeros(measured_data.shape)
    for seg in range(n_readouts)[preselection::preselection+1]:
        sample = deepcopy(measured_data[seg::n_readouts, :])
        assert len(sample) == n_shots
        # resample with replacement the shots for the segment seg
        p = np.random.choice(np.arange(n_shots), n_shots)
        sample_i[seg::n_readouts, :] = sample[p]
        # preselection
        if preselection:
            sample = deepcopy(measured_data[seg-1::n_readouts, :])
            assert len(sample) == n_shots
            sample_i[seg-1::n_readouts, :] = sample[p]
    return sample_i


def bootstrapping_state_tomography(data_dict, keys_in, store_rhos=False,
                                   verbose=False, **params):
    """
    Computes bootstrapping statistics of the density matrix fidelity.
    :param data_dict: OrderedDict containing thresholded shots specified by
        keys_in, and where processed results will be stored
    :param keys_in: list of key names or dictionary keys paths in
        data_dict for the data to be analyzed (expects thresholded shots)
    :param store_rhos: whether to store the density matrices in addition to
        the bootstrapping fidelities.
    :param verbose: whether to show progress print statements
    :param params: keyword arguments
        Expects to find either in data_dict or in params:
            - Nbstrp: int specifying the number of bootstrapping cycles,
                i.e. sample size for estimating errors, the number of times
                the raw data is resampled
            - timestamps: list of with the timestamps of the state tomo msmt
    :return: stores in data_dict:
        - {estimation_type}.bootstrapping_fidelities
        - (optionally) {estimation_type}.bootstrapping_rhos
        for estimation_type in estimation_types
    Assumptions:
     - CURRENTLY ONLY SUPPORTS DATA FROM HDF FILES!
     - !! This function calls state_tomography_analysis so all required input
        params needed there must also be here

    """
    Nbstrp = hlp_mod.get_param('Nbstrp', data_dict, raise_error=True, **params)

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)

    prep_params = hlp_mod.get_param('preparation_params', data_dict,
                                    default_value={}, **params)
    preselection = prep_params.get('preparation_type', 'wait') == 'preselection'
    n_readouts = hlp_mod.get_param('n_readouts', data_dict, raise_error=True,
                                   **params)
    raw_data = np.concatenate([np.reshape(arr, (len(arr), 1))
                               for arr in data_to_proc_dict.values()],
                              axis=1)
    n_shots = len(raw_data[:, 1]) // n_readouts

    timestamp = hlp_mod.get_param('timestamps', data_dict, raise_error=True,
                                  **params)
    if len(timestamp) > 1:
        raise ValueError(f'Bootstrapping can only be done for one data file. '
                         f'{len(timestamp)} timestamps were found.')
    data_dict_temp = {}
    hlp_mod.add_param('cal_points',
                      hlp_mod.get_param('cal_points', data_dict, **params),
                      data_dict_temp)
    hlp_mod.add_param('meas_obj_value_names_map',
                      hlp_mod.get_param('meas_obj_value_names_map',
                                        data_dict, **params),
                      data_dict_temp)
    hlp_mod.add_param('preparation_params',
                      hlp_mod.get_param('preparation_params',
                                        data_dict, **params),
                      data_dict_temp)
    hlp_mod.add_param('rho_target',
                      hlp_mod.get_param('rho_target', data_dict),
                      data_dict_temp)
    data_dict_temp = dat_extr_mod.extract_data_hdf(timestamps=timestamp,
                                                   data_dict=data_dict_temp)

    estimation_types = hlp_mod.get_param('estimation_types', data_dict,
                                         default_value=('least_squares',
                                                        'max_likelihood'),
                                         **params)

    fidelities = {est_type: np.zeros(Nbstrp) for est_type in estimation_types}
    if store_rhos:
        rhos = {est_type: Nbstrp*[''] for est_type in estimation_types}

    params.pop('do_plotting', False)
    params.pop('prepare_plotting', False)
    replace_value = params.pop('replace_value', False)
    # do bootstrapping Nbstrp times
    for n in range(Nbstrp):
        if verbose:
            print('Bootstrapping run state tomo: ', n)
        sample_i = bootstrapping(measured_data=raw_data, n_readouts=n_readouts,
                                 n_shots=n_shots, preselection=preselection)
        for i, keyi in enumerate(data_to_proc_dict):
            hlp_mod.add_param(keyi, sample_i[:, i], data_dict_temp,
                              add_param_method='replace')

        state_tomography_analysis(data_dict_temp, keys_in=keys_in,
                                  do_plotting=False, prepare_plotting=False,
                                  replace_value=True, **params)

        for estimation_type in estimation_types:
            fidelities[estimation_type][n] = hlp_mod.get_param(
                f'{estimation_type}.fidelity', data_dict_temp, raise_error=True)
            if store_rhos:
                rhos[estimation_type][n] = hlp_mod.get_param(
                    f'{estimation_type}.rho', data_dict_temp, raise_error=True)

    params['replace_value'] = replace_value
    hlp_mod.add_param('Nbstrp', Nbstrp, data_dict, **params)
    for estimation_type in fidelities:
        hlp_mod.add_param(f'{estimation_type}.bootstrapping_fidelities',
                          fidelities[estimation_type], data_dict, **params)
        if store_rhos:
            hlp_mod.add_param(f'{estimation_type}.bootstrapping_rhos',
                              rhos[estimation_type], data_dict, **params)


def bootstrapping_process_tomography(
        data_dict, keys_in, Nbstrp, gate_name='CZ', Uideal=None,
        estimation_types=('least_squares', 'max_likelihood'),
        prep_pulses_list=None, verbose=False, **params):
    """
    Computes bootstrapping statistics of the error of gate_name.
    :param data_dict: OrderedDict containing thresholded shots specified by
        keys_in, and where processed results will be stored
    :param keys_in: list of key names or dictionary keys paths in
        data_dict for the data to be analyzed (expects thresholded shots)
    :param Nbstrp: int specifying the number of bootstrapping cycles,
        i.e. sample size for estimating errors, the number of times the raw data
        is resampled
    :param gate_name: name of the gate for which the error is estimated.
         MUST CORRESPOND TO Uideal IF THE LATTER IS PROVIDED, since gate_name
         will be used in the key name for storing the results
    :param Uideal: qutip Qobj of the ideal unitary operator for gate_name
    :param estimation_types: list of strings indicating the methods that were
         used to construct the density matrices stores in data_dict[prep_pulses].
         This function will do process tomo for all the estimation types in
         this list.
    :param prep_pulses_list: list of tuples with length nr_qubits contanining
         strings indicating the preparation pulses for each state tomography
         measurement. If not specified, it will be constructed from product of
         basis_rots.
    :param verbose: whether to show progress print statements
    :param params: keyword_arguments
    :return: stores in data_dict:
        - bootstrapping_errors_{gate_name}.{estimation_type} for estimation_type
        in estimation_types

    Assumptions:
     - CURRENTLY ONLY SUPPORTS DATA FROM HDF FILES!
     - !! This function calls bootstrapping_state_tomography and
        process_tomography_analysis so all required input params needed
        there must also be here
    """
    if prep_pulses_list is None:
        meas_obj_names = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
            **params)
        n_qubits = len(meas_obj_names)
        basis_rots = hlp_mod.get_param(
            'basis_rots', data_dict, raise_error=True,
            error_message='Either prep_pulses_list or basis_rots needs to be '
                          'provided.', **params)
        if hlp_mod.get_param('basis_rots', data_dict) is None:
            hlp_mod.add_param('basis_rots', basis_rots, data_dict, **params)
        prep_pulses_list = list(itertools.product(basis_rots, repeat=n_qubits))

    replace_value = params.pop('replace_value', False)
    errors = {est_type: np.zeros(Nbstrp) for est_type in estimation_types}

    for n in range(Nbstrp):
        if verbose:
            print('Bootstrapping run process tomo: ', n)
        data_dict_temp = {}
        measured_rhos = {est_type: len(prep_pulses_list) * [''] for
                         est_type in estimation_types}
        for p, prep_pulses in enumerate(prep_pulses_list):
            data_dict_state_tomo = hlp_mod.get_param(''.join(prep_pulses),
                                                     data_dict,
                                                     raise_error=True)
            bootstrapping_state_tomography(data_dict_state_tomo, keys_in,
                                           estimation_types=estimation_types,
                                           Nbstrp=1, replace_value=True,
                                           store_rhos=True, **params)
            for estimation_type in estimation_types:
                measured_rhos[estimation_type][p] = hlp_mod.get_param(
                    f'{estimation_type}.bootstrapping_rhos',
                    data_dict_state_tomo)[0]

        process_tomography_analysis(data_dict_temp,
                                    prep_pulses_list=prep_pulses_list,
                                    measured_rhos=measured_rhos,
                                    estimation_types=estimation_types,
                                    gate_name=gate_name, Uideal=Uideal,
                                    replace_value=True, **params)

        for estimation_type in estimation_types:
            errors[estimation_type][n] = hlp_mod.get_param(
                f'measured_error_{gate_name}.{estimation_type}',
                data_dict_temp)

    params['replace_value'] = replace_value
    hlp_mod.add_param('Nbstrp', Nbstrp, data_dict, **params)
    for estimation_type in errors:
        hlp_mod.add_param(
            f'bootstrapping_errors_{gate_name}.{estimation_type}',
            errors[estimation_type], data_dict, **params)


def get_tomo_data_subset(data_dict, keys_in, preselection=True, **params):

    meas_obj_names = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    n = len(meas_obj_names)
    ignore_extra_seqs = hlp_mod.get_param(
        'ignore_extra_sequences', data_dict, **params, default_value=False)
    init_rots_basis = hlp_mod.get_param('init_rots_basis', data_dict, **params)
    prep_pulses_list = list(itertools.product(init_rots_basis, repeat=n))
    final_rots_basis = hlp_mod.get_param('final_rots_basis', data_dict, **params)
    cal_points = hlp_mod.get_measurement_properties(data_dict,
                                                    props_to_extract=['cp'])
    total_nr_segments = (len(final_rots_basis)**n + len(cal_points.states)) * \
                        (preselection + 1)

    nr_shots = hlp_mod.get_param('nr_shots', data_dict, **params)
    if nr_shots is None:
        detectors = hlp_mod.get_param('exp_metadata.Detector Metadata.detectors',
                                      data_dict)
        nr_shots = detectors[list(detectors)[0]].get('nr_shots', None)
        if nr_shots is None:
            raise ValueError('Please provide nr_shots.')

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    for keyi, data in data_to_proc_dict.items():
        if ignore_extra_seqs:
            data = data[:len(init_rots_basis)**n * nr_shots*total_nr_segments]
        data_reshaped = data.reshape((
            len(init_rots_basis)**n, nr_shots*total_nr_segments))
        if len(prep_pulses_list) != data_reshaped.shape[0]:
            raise ValueError(f'len(prep_pulses_list)={len(prep_pulses_list)} '
                             f'does not match data shape {data_reshaped.shape}).')

        for prep_pulses, row in zip(prep_pulses_list, data_reshaped):
            hlp_mod.add_param(f'{"".join(prep_pulses)}.{keyi}',
                              row, data_dict, **params)
