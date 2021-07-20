import logging
import itertools
import numpy as np
from numpy.linalg import inv
from typing import List, Optional, Tuple
import scipy as sp
try:
    import qutip as qtp
except ImportError as e:
    logging.warning('Could not import qutip, tomo code will not work')

DEFAULT_BASIS_ROTS = ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
# General state tomography functions

def least_squares_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                             Omega: Optional[np.ndarray]=None) -> qtp.Qobj:
    """
    Executes generalized linear least squares fit of the density matrix to
    the measured observables.

    The resulting density matrix could be unphysical, as the positive
    semi-definiteness is not ensured. The resulting density matrix has unit
    trace.

    To enforce the unit trace of the resulting density matrix, only the
    traceless part of the density matrix is fitted. For this we also need to
    subtract the trace of the measurement operators from the expectation values
    before fitting.

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus.
        Omega: The covariance matrix of the expectation values mu.
               If a 1-dimensional array is passed, the values are interpreted
               as the variations of the mus and the correlations are assumed to
               be zero.
               If `None` is passed, then all measurements are assumed to have
               equal variances.
    Returns: The found density matrix as a qutip operator.
    """
    d = Fs[0].shape[0]
    if Omega is None:
        Omega = np.diag(np.ones(len(mus)))
    elif len(Omega.shape) == 1:
        Omega = np.diag(Omega)
    OmegaInv = inv(Omega)

    Os = hermitian_traceless_basis(d)
    Fmtx = np.array([[(F * O).tr().real for O in Os] for F in Fs])
    F0s = np.array([F.tr().real / d for F in Fs])

    cs = inv((Fmtx.T).dot(OmegaInv).dot(Fmtx)).dot(Fmtx.T).dot(OmegaInv).dot(
        mus - F0s)

    rho = qtp.qeye(d) / d + sum([c * O for c, O in zip(cs, Os)])
    return rho


def hermitian_traceless_basis(d: int) -> List[qtp.Qobj]:
    """
    Generates a list of basis-vector matrices for the group of traceless
    Hermitian matrices.
    Args:
        d: dimension of the Hilbert space.
    Returns: A list of basis-vector matrices.
    """
    Os = []
    for i in range(d):
        for j in range(d):
            if i == d - 1 and j == d - 1:
                continue
            elif i == j:
                O = np.diag(-np.ones(d, dtype=np.complex) / d)
                O[i, j] += 1
            elif i > j:
                O = np.zeros((d, d), dtype=np.complex)
                O[i, j] = 1
                O[j, i] = 1
            else:
                O = np.zeros((d, d), dtype=np.complex)
                O[i, j] = 1j
                O[j, i] = -1j
            Os.append(qtp.Qobj(O))
    return Os

def imle_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                             iterations: Optional[int]=None,
                             tolerance: Optional[float]=None,
                             rho_guess: Optional[qtp.Qobj]=None) -> qtp.Qobj:
    """
    Executes iterative maximum likelihood based on a complete set of POVMs

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus. Must be a set able to
            fully characterize any density matrix of the corresponding
            dimension.
        tolerance (float; default: None): tolerance threshold. If not set, it
            will use default -15 for 2 qubits and -18 for 3 qubits.
        iterations (int; default: None): iteration threshold. If not set, it
            will use 1000 by default, which is larger than mean it. for
            convergence (~100 on 3 qubits) to give preference to tolerance as a
            stopping point.
        rho_guess: Initial rho for the iterative method. If none is provided, it
              starts on the identity matrix corresponding to the fully
              mixed state. (The method could be improved with a better
              initial choice? but this is something yet to look into)
    Returns: The found density matrix as a qutip operator.
    """

    dim = Fs[0].shape[0]
    n = int(np.log2(dim))
    new_shape_unty = [[2]*n,[2]*n]
    new_shape_ket = [[2]*n,[1]*n]

    # Set default values if they are None. Inital rho should not have a big
    # effect in the algorithm as long as it is positive semidefinite.
    if rho_guess == None:
        rho_guess = qtp.Qobj(np.diag(np.ones(dim)/dim),new_shape_unty)
    if iterations == None:
        iterations = 1000
    if tolerance == None:
        tolerance = (-15 if n==2 else -18)

    # Format POVM operators properly and get g matrix (sum of all used POVMs)
    povm_list = []
    g = 0
    for i in range(len(mus)):
        povm_list.append([qtp.Qobj(Fs[i].data,new_shape_unty), mus[i]])
        g += qtp.Qobj(Fs[i].data,new_shape_unty)
    g_inv = g.inv()

    # start loop
    rho = rho_guess
    for it in range(iterations):
        R = 0
        # build current trace to calculate R
        for povm in povm_list:
            trace_step = ((povm[0] * rho).tr()).real
            if trace_step>0:
                R += povm[1]/trace_step * povm[0]
        next_rho = g_inv * R * rho * R * g_inv
        # check for trace to renormalize
        next_rho = next_rho / next_rho.tr()

        # Tolerance checks that we're not stuck from one step to another
        tol = np.log(1-fidelity(next_rho,rho))
        rho = next_rho
        if tol < tolerance:
            print('Stopped by step-improvment under given tolerance (' + str(tolerance) + ')')
            print('Stopped with ' + str(it+1) + ' iterations')
            break
    # If we reach maximum iterations the algorithm also stops
    if it == (iterations-1):
        print('Stopped with maximum iterations (' + str(iterations) + ')')

    return next_rho

def mle_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                   Omega: Optional[np.ndarray]=None,
                   rho_guess: Optional[qtp.Qobj]=None) -> qtp.Qobj:
    """
    Executes a maximum likelihood fit to the measured observables, respecting
    the physicality constraints of the density matrix.

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus.
        Omega: The covariance matrix of the expectation values mu.
               If a 1-dimensional array is passed, the values are interpreted
               as the variations of the mus and the correlations are assumed to
               be zero.
               If `None` is passed, then all measurements are assumed to have
               equal variances.
        rho_guess: The initial value of the density matrix for the iterative
                   optimization algorithm.
    Returns: The found density matrix as a qutip operator.
    """
    d = Fs[0].shape[0]
    if Omega is None:
        Omega = np.diag(np.ones(len(mus)))
    elif len(Omega.shape) == 1:
        Omega = np.diag(Omega)
    OmegaInv = inv(Omega)

    if rho_guess is None:
        params0 = np.ones(d * d)
    else:
        rho_positive = rho_guess - min(rho_guess.eigenenergies().min(), 0)
        rho_positive += 1e-9
        T = np.linalg.cholesky(rho_positive.data.toarray())
        params0 = np.real(np.concatenate((T[np.tril_indices(d)],
                                          -1j*T[np.tril_indices(d, -1)])))

    def min_func(params, mus, Fs, OmegaInv, d):
        T = ltriag_matrix(params, d)
        rho = T * T.dag()
        dmus = mus.copy()
        for i, F in enumerate(Fs):
            dmus[i] -= (rho * F).tr().real
        return (dmus.T.dot(OmegaInv).dot(dmus)).real

    def constr_func(params, d):
        T = ltriag_matrix(params, d)
        rho = T * T.dag()
        return rho.tr().real - 1

    params = sp.optimize.minimize(min_func, params0, (mus, Fs, OmegaInv, d),
                                  method='SLSQP', constraints={'type': 'eq',
                                      'fun': constr_func, 'args': (d, )}).x
    T = ltriag_matrix(params, d)
    return T * T.dag()

def pauli_values_tomography(mus: np.ndarray, Fs: List[qtp.Qobj],
                            basis_rots: List[str]) -> qtp.Qobj:
    """
    Extracts the measured pauli values considering the assignment correction
    from the calibration points, then uses them as the weight for all
    combinations of Pauli operator products to construct a density matrix.
    It will not be physical in general.
    Extracting pauli values with 'density_matrix_to_pauli_basis' from the
    output density matrix will return the measured pauli values.

    Args:
        mus: 1-dimensional numpy ndarray containing the measured expectation
             values for the measurement operators Fs.
        Fs: A list of the measurement operators (as qutip operators) that
            correspond to the expectation values in mus. Must receive the
            original ones before applying the rotation basis.
        rotation_basis: list of standard PycQED pulse names that were
                        applied to qubits before measurement
    Returns: The found density matrix as a qutip operator.
    """

    nr_qubits = int(np.log2(Fs[0].shape[0]))
    pulse_list = list(itertools.product(basis_rots,repeat=nr_qubits))
    rotations = standard_qubit_pulses_to_rotations(pulse_list)

    Fs_corr = []
    assign_corr = []
    for i,F in enumerate(Fs):
        new_op = np.zeros(2**nr_qubits)
        new_op[i] = 1
        Fs_corr.append(qtp.Qobj(np.diag(new_op)))
        assign_corr.append(np.diag(F.full()))
    pauli_Fs = rotated_measurement_operators(rotations, Fs_corr)
    pauli_Fs = list(itertools.chain(*np.array(pauli_Fs, dtype=np.object).T))

    pauli_mus = np.reshape(mus,[-1,2**nr_qubits])
    for i,raw_mus in enumerate(pauli_mus):
        pauli_mus[i] = np.matmul(np.linalg.inv(assign_corr),np.array(raw_mus))
    pauli_mus = pauli_mus.flatten()

    pauli_values = []
    for op in generate_pauli_set(nr_qubits)[1]:
        nr_terms = 0
        sum_terms = 0.
        for meas_op, meas_res in zip(pauli_Fs,pauli_mus):
            trace = (meas_op*op).tr().real
            clss = int(trace*2)
            if clss < 0:
                sum_terms -= meas_res
                nr_terms += 1
            elif clss > 0:
                sum_terms += meas_res
                nr_terms += 1
        pauli_values.append(2**nr_qubits*sum_terms/nr_terms)

    rho_pauli = pauli_set_to_density_matrix(pauli_values)

    return rho_pauli


def ltriag_matrix(params: np.ndarray, d: int):
    """
    Creates a lower-triangular matrix of dimension d from an array of d**2
    real numbers. The elements on the diagonal of the resulting matrix
    are real and on the off-2diagonal they can be complex.

    The Cholesky decomposition of a positive-definite matrix is of this form.
    """
    T = np.zeros((d, d), dtype=np.complex)
    T[np.tril_indices(d)] += params[:((d * d + d) // 2)]
    T[np.tril_indices(d, -1)] += 1j * params[((d * d + d) // 2):]
    return qtp.Qobj(T)


def fidelity(rho1: qtp.Qobj, rho2: qtp.Qobj) -> float:
    """
    Returns the fidelity between the two quantum states rho1 and rho2.
    Uses the Jozsa definition (the smaller of the two), not the Nielsen-Chuang
    definition.

    F = Tr(√((√rho1) rho2 √(rho1)))^2
    """

    rho1 = convert_to_density_matrix(rho1)
    rho2 = convert_to_density_matrix(rho2)

    return (rho1.sqrtm()*rho2*rho1.sqrtm()).sqrtm().tr().real ** 2

def max_fidelity(rho1: qtp.Qobj, rho2: qtp.Qobj, thetas1, thetas2):

    fid_vec = np.zeros((len(thetas1), len(thetas2)))
    rho1 = qtp.Qobj(rho1, dims=[[2, 2], [2, 2]], shape=(4, 4))
    target_bell = qtp.Qobj(rho2.flatten(), dims=[[2, 2], [1, 1]], shape=(4, 1))

    for i, theta1 in enumerate(thetas1):
        for j, theta2 in enumerate(thetas2):
            state_rotation = qtp.tensor(qtp.rotation(qtp.sigmaz(), theta1),
                                        qtp.rotation(qtp.sigmaz(), theta2))
            target_state = state_rotation*target_bell
            fid_vec[i][j] = float(
                np.real((target_state.dag()*rho1*target_state).data[0, 0]))


    idxs_f_max = np.unravel_index(np.argmax(fid_vec),
                                  dims=(len(thetas1),
                                        len(thetas1)))

    max_F = 100*np.max(fid_vec)
    phase1 = thetas1[idxs_f_max[0]]*180./np.pi
    phase2 = thetas2[idxs_f_max[1]]*180./np.pi

    return max_F, phase1, phase2


def concurrence(rho):
    """
    Calculates the concurrence of the two-qubit state rho given in the
    qubits' basis according to https://doi.org/10.1103/PhysRevLett.78.5022
    """
    rho = convert_to_density_matrix(rho)
    # convert to bell basis
    b = [np.sqrt(0.5)*qtp.Qobj(np.array(l)) for l in
            [[1, 0, 0, 1], [1j, 0, 0, -1j], [0, 1j, 1j, 0], [0, 1, -1, 0]]]
    rhobell = np.zeros((4, 4), dtype=np.complex)
    for i in range(4):
        for j in range(4):
            rhobell[i, j] = (b[j].dag()*rho*b[i])[0, 0]
    rhobell = qtp.Qobj(rhobell)
    R = (rhobell.sqrtm()*rhobell.conj()*rhobell.sqrtm()).sqrtm()
    try:
        C = max(0, 2*R.eigenenergies().max() - R.tr())
    except Exception:
        # Steph, 29.09.2020: there is a bug numpy that causes this call to
        # sometimes fail the first time it is called.
        C = max(0, 2*R.eigenenergies().max() - R.tr())
    return C


def purity(rho: qtp.Qobj) -> float:
    rho = convert_to_density_matrix(rho)
    return (rho*rho).tr().real


def density_matrix_to_pauli_basis(rho):
    """
    Returns the expectation values for all combinations of Pauli operator
    products. The dimension of the density matrix must be a power of two.
    """
    rho = convert_to_density_matrix(rho)
    d = rho.shape[0]
    if 2 ** (d.bit_length() - 1) == d:
        nr_qubits = d.bit_length() - 1
    else:
        raise ValueError(
            'Dimension of the density matrix is not a power of '
            'two.')

    O1 = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    Os = [qtp.Qobj(qtp.tensor(*O1s).full()) for O1s in
          itertools.product(*[O1] * nr_qubits)]
    return np.array([(rho * O).tr().real for O in Os])

def pauli_set_to_density_matrix(paulis):
    """
    Returns a density matrix constructed with all combinations of Pauli operator
    products weighted with the corresponding expectation values given.
    The dimension of output density matrix will be a power of two.
    """
    d2 = len(paulis)
    if 4 ** ((d2.bit_length() - 1)//2) == d2:
        nr_qubits = (d2.bit_length() - 1)//2
        d = 2**nr_qubits
    else:
        raise ValueError(
            'Dimension of the density matrix is not a power of '
            'two.')

    O1 = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    Os = [qtp.Qobj(qtp.tensor(*O1s).full()) for O1s in
          itertools.product(*[O1] * nr_qubits)]
    rho = qtp.Qobj(dims=Os[0].dims)
    for pauli, O in zip(paulis, Os):
        rho += pauli*O
    return rho/d

def convert_to_density_matrix(rho):
    if not isinstance(rho, qtp.Qobj):
        rho = qtp.Qobj(rho)
    if rho.type == 'ket':
        rho = rho * rho.dag()
    elif rho.type == 'bra':
        rho = rho.dag() * rho
    return qtp.Qobj(rho.full())


# PycQED specific functions


def measurement_operator_from_calpoints(
        calpoints: np.ndarray, repetitions:int=1) -> Tuple[qtp.Qobj, float]:
    """
    Calculates the measurement operator and its expected variation from
    a list of calibration points corresponding to the preparations of the
    diagonal elements of the density matrix.

    For example for a two-qubit measurement, the calibration points would
    correspond to the prepared states gg, ge, eg, ee.
    """
    d = len(calpoints) // repetitions
    means = np.array([np.mean(calpoints[-repetitions*i:][:repetitions])
                      for i in range(d, 0, -1)])
    F = qtp.Qobj(np.diag(means))
    variation = np.std(calpoints - means.repeat(repetitions))**2 \
        if repetitions > 1 else 1
    return F, variation


def rotated_measurement_operators(rotations: List[qtp.Qobj],
                                  Fs: List[qtp.Qobj]) -> List[List[qtp.Qobj]]:
    """
    For each measurement operator in Fs, calculates the measurement operators
    when first applying the rotations in the rotations parameter to the system.
    """
    return [[U.dag() * F * U for U in rotations] for F in Fs]


def standard_qubit_pulses_to_rotations(pulse_list: List[Tuple]) \
        -> List[qtp.Qobj]:
    """
    Converts lists of n-tuples of standard PycQED single-qubit pulse names to
    the corresponding rotation matrices on the n-qubit Hilbert space.
    """
    standard_pulses = {
        'I': qtp.qeye(2),
        'X0': qtp.qeye(2),
        'Z0': qtp.qeye(2),
        'X180': qtp.rotation(qtp.sigmax(), np.pi),
        'mX180': qtp.rotation(qtp.sigmax(), -np.pi),
        'Y180': qtp.rotation(qtp.sigmay(), np.pi),
        'mY180': qtp.rotation(qtp.sigmay(), -np.pi),
        'X90': qtp.rotation(qtp.sigmax(), np.pi/2),
        'mX90': qtp.rotation(qtp.sigmax(), -np.pi/2),
        'Y90': qtp.rotation(qtp.sigmay(), np.pi/2),
        'mY90': qtp.rotation(qtp.sigmay(), -np.pi/2),
        'Z90': qtp.rotation(qtp.sigmaz(), np.pi/2),
        'mZ90': qtp.rotation(qtp.sigmaz(), -np.pi/2),
        'Z180': qtp.rotation(qtp.sigmaz(), np.pi),
        'mZ180': qtp.rotation(qtp.sigmaz(), -np.pi),
    }
    rotations = [qtp.tensor(*[standard_pulses[pulse] for pulse in qb_pulses])
                 for qb_pulses in pulse_list]
    for i in range(len(rotations)):
        rotations[i].dims = [[d] for d in rotations[i].shape]
    return rotations


def standard_qubit_pulses_to_pauli(pulse_list: List[Tuple]) \
        -> List[qtp.Qobj]:
    """
    Converts lists of n-tuples of standard PycQED single-qubit pulse names to
    the corresponding measurement operators.
    """
    standard_pulses = {
        'I': 'Z',
        'X0': 'Z',
        'Z0': 'Z',
        'X180': 'mZ',
        'mX180': 'mZ',
        'Y180': 'mZ',
        'mY180': 'mZ',
        'X90': 'Y',
        'mX90': 'mY',
        'Y90': 'X',
        'mY90': 'mX',
        'Z90': 'Z',
        'mZ90': 'Z',
        'Z180': 'Z',
        'mZ180': 'Z',
    }
    paulis = [[''.join(standard_pulses[pulse] for pulse in qb_pulses)][0]
              for qb_pulses in pulse_list]
    return paulis

def generate_pauli_set(nr_qubits):
    paulis = {
        'I': qtp.qeye(2),
        'X': qtp.sigmax(),
        'Y': qtp.sigmay(),
        'Z': qtp.sigmaz(),
    }
    labels = [''.join(ops) for ops in itertools.product(['I','X','Y','Z'],
              repeat=nr_qubits)]
    operators = []
    for label in labels:
        op = qtp.Qobj([[1]])
        for c in label:
            op = qtp.tensor(op, paulis[c])
        op.dims = [[2**nr_qubits], [2**nr_qubits]]
        operators.append(op)
    return labels, operators

def generate_povm_set(nr_qubits):
    """
    Generates povm set for n qubits (nr_qubits) based on the pauli operator set.
    """
    Id = qtp.Qobj([[1,0],[0,1]])
    X = qtp.Qobj([[0,1],[1,0]])
    Y = qtp.Qobj([[0,-1j],[1j,0]])
    Z = qtp.Qobj([[1,0],[0,-1]])
    povms = { 'X': (Id+X)/2, 'x': (Id-X)/2, 'Y': (Id+Y)/2,
              'y': (Id-Y)/2, 'Z': (Id+Z)/2, 'z': (Id-Z)/2 }

    labels = [''.join(ops) for ops in itertools.product(['X','x','Y','y',
                'Z','z'], repeat=nr_qubits)]
    operators = []
    for label in labels:
        op = qtp.Qobj([[1]])
        for c in label:
            op = qtp.tensor(op, paulis[c])
        op.dims = [[2**nr_qubits], [2**nr_qubits]]
        operators.append(op)
    return labels, operators

def pauli_to_povm(nr_qubits,mus):
    """
    Converts pauli expected values to expected values of the povm set generated
    by generate_povm_set.
    """
    base_conv = [[1/2,1/2,0,0],[1/2,-1/2,0,0],[1/2,0,1/2,0],[1/2,0,-1/2,0],
                    [1/2,0,0,1/2],[1/2,0,0,-1/2]]
    full_conv = [[1]]
    for n in range(nr_qubits):
        next_conv = []
        for povm_ind_new in base_conv:
            for povm_ind_prev in full_conv:
                next_conv.append(np.array([ind*np.array(povm_ind_prev)
                for ind in povm_ind_new]).flatten())
        full_conv = next_conv

    return np.dot(full_conv,mus)

def povm_to_pauli(nr_qubits,mus):
    """
    Converts expected values of the povm set generated by generate_povm_set to
    pauli expected values.
    """
    base_conv = [[1/6,1/6,1/6,1/6,1/6,1/6],[3/6,-3/6,0,0,0,0],
                    [0,0,3/6,-3/6,0,0],[0,0,0,0,3/6,-3/6]]
    full_conv = [[1]]
    for n in range(nr_qubits):
        next_conv = []
        for povm_ind_new in base_conv:
            for povm_ind_prev in full_conv:
                next_conv.append(np.array([ind*np.array(povm_ind_prev)
                for ind in povm_ind_new]).flatten())
        full_conv = next_conv

    return np.dot(full_conv,mus)
