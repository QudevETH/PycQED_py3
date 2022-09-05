import numpy as np
import scipy as sp
import scipy.optimize
import pycqed.simulations.transmon as tm


def kron(*args):
    """Variable input number Kronecker product.
    See also numpy.kron
    """
    assert len(args) > 0
    res = args[0]
    for x in args[1:]:
        res = np.kron(res, x)
    return res


def calculate_idle_spec_fidelity(fqb, anh, specs, t, angular_units=False):
    """Calculates process fidelity of an idling qubit due to residual couplings

    Args:
         fqb: float
            qubit frequency
        anh: float
            qubit anharmonicity
        specs: list[tuple[fqb_spec, anh_spec, j, ef]]
            fqb_spec: float
                spectator qubit frequency
            anh_spec: float
                spectator qubit anharmonicity
            j: float
                coupling rate to the spectator qubit
            ef: bool
                if the spectator qubit can also be in the f-state
        t: float
            interaction time
        angular_units: bool
            True if the inputs are specified in angular frequency, False for
            regular frequency (default False)
    Returns:
        Worst case (over spectator qubit states) process fidelity.
    """
    zeta = np.array([0])
    for fqb_spec, anh_spec, j, ef in specs:
        zeta_spec_e = tm.transition_dispersive_shift(1, fqb, anh, fqb_spec,
                                                     anh_spec, j)
        if ef:
            zeta_spec_f = tm.transition_dispersive_shift(1, fqb, anh, fqb_spec,
                                                         anh_spec, j, 2)
            zeta = kron(zeta, [1, 1, 1]) + kron(np.ones_like(zeta),
                                                [0, zeta_spec_e, zeta_spec_f])
        else:
            zeta = kron(zeta, [1, 1]) + kron(np.ones_like(zeta),
                                             [0, zeta_spec_e])
    z_angle = (1 if angular_units else 2 * np.pi) * zeta * t
    return tm.cz_process_fidelity(z_angle).min()


def calculate_gate_spec_fidelity(fint, anh1, anh2, specs1, specs2, t_gate,
                                 angular_units=False):
    """Calculates process fidelity of a CZ gate due to residual couplings

    Args:
        fint: float
            lower gate qubit frequency during the gate
        anh1: float
            lower gate qubit anharmonicity
        anh2: float
            higher gate qubit anharmonicity
        specs1: list[tuple[fqb_spec, anh_spec, j, ef]]
            fqb_spec: float
                frequency of spectator qubit to lower gate qubit
            anh_spec: float
                anharmonicity of spectator qubit to lower gate qubit
            j: float
                coupling rate to the spectator qubit to the lower gate qubit
            ef: bool
                if the spectator qubit can also be in the f-state
        specs2: list[tuple[fqb_spec, anh_spec, j, ef]]
            fqb_spec: float
                frequency of spectator qubit to higher gate qubit
            anh_spec: float
                anharmonicity of spectator qubit to higher gate qubit
            j: float
                coupling rate to the spectator qubit to the higher gate qubit
            ef: bool
                if the spectator qubit can also be in the f-state
        t_gate: float
            interaction time
        angular_units: bool
            True if the inputs are specified in angular frequency, False for
            regular frequency (default False)
    Returns:
        Worst case (over spectator qubit states) process fidelity.
    """
    fqb1 = fint
    fqb2 = fint - anh2
    zeta1_ge = np.array([0])
    zeta2_ge = np.array([0])
    zeta2_ef = np.array([0])

    for fqb_spec, anh_spec, j_spec, ef in specs1:
        zeta_ge_spec_e = tm.transition_dispersive_shift(1, fqb1, anh1, fqb_spec,
                                                        anh_spec, j_spec, 1)
        if ef:
            zeta_ge_spec_f = tm.transition_dispersive_shift(1, fqb1, anh1,
                                                            fqb_spec, anh_spec,
                                                            j_spec, 2)
            zeta1_ge = kron(zeta1_ge, [1, 1, 1]) + kron(np.ones_like(zeta1_ge),
                                                        [0, zeta_ge_spec_e,
                                                         zeta_ge_spec_f])
            zeta2_ge = kron(zeta2_ge, [1, 1, 1])
            zeta2_ef = kron(zeta2_ef, [1, 1, 1])
        else:
            zeta1_ge = kron(zeta1_ge, [1, 1]) + kron(np.ones_like(zeta1_ge),
                                                     [0, zeta_ge_spec_e])
            zeta2_ge = kron(zeta2_ge, [1, 1])
            zeta2_ef = kron(zeta2_ef, [1, 1])

    for fqb_spec, anh_spec, j_spec, ef in specs2:
        zeta_ge_spec_e = tm.transition_dispersive_shift(1, fqb2, anh2, fqb_spec,
                                                        anh_spec, j_spec, 1)
        zeta_ef_spec_e = tm.transition_dispersive_shift(2, fqb2, anh2, fqb_spec,
                                                        anh_spec, j_spec, 1)
        if ef:
            zeta_ge_spec_f = tm.transition_dispersive_shift(1, fqb2, anh2,
                                                            fqb_spec, anh_spec,
                                                            j_spec, 2)
            zeta_ef_spec_f = tm.transition_dispersive_shift(2, fqb2, anh2,
                                                            fqb_spec, anh_spec,
                                                            j_spec, 2)
            zeta1_ge = kron(zeta1_ge, [1, 1, 1])
            zeta2_ge = kron(zeta2_ge, [1, 1, 1]) + kron(np.ones_like(zeta2_ge),
                                                        [0, zeta_ge_spec_e,
                                                         zeta_ge_spec_f])
            zeta2_ef = kron(zeta2_ef, [1, 1, 1]) + kron(np.ones_like(zeta2_ef),
                                                        [0, zeta_ef_spec_e,
                                                         zeta_ef_spec_f])
        else:
            zeta1_ge = kron(zeta1_ge, [1, 1])
            zeta2_ge = kron(zeta2_ge, [1, 1]) + kron(np.ones_like(zeta2_ge),
                                                     [0, zeta_ge_spec_e])
            zeta2_ef = kron(zeta2_ef, [1, 1]) + kron(np.ones_like(zeta2_ef),
                                                     [0, zeta_ef_spec_e])

    z1 = (1 if angular_units else 2 * np.pi) * zeta1_ge * t_gate
    z2 = (1 if angular_units else 2 * np.pi) * zeta2_ge * t_gate
    zc = (0.5 if angular_units else np.pi) * (zeta2_ef - zeta1_ge) * t_gate

    return tm.cz_process_fidelity(z1, z2, zc).min()


def calculate_step_fidelity(qubits, couplings, wints):
    """Calculates the process fidelity for a step of parallel gates

    Args:
        qubits: dict[str, dict[str, any]]
            A dictionary from qubit names to properties. The following
            properties should be defined:
                neighbors: a set of qubit names that neighbor this qubit
                wq: qubit parking frequency
                anh: qubit anharmonicity (negative for transmon qubits)
        couplings: dict[frozenset[str], float]
            Qubit-qubit coupling rates. Keys are frozensets of coupled qubit
            names and values are the 01-10 coupling rates.
        wints: dict[frozenset[str], float]
            Interaction frequencies of  gates that will be executed in parallel
    Return: dict[str, float]
        Worst-case process fidelity of each qubit in the parallel gate step.
        For qubits involved in gates, the total error is equally divided
        between the two qubits.
    """
    active_qubits = {qn for g in wints for qn in g}
    inactive_qubits = {qn for qn in qubits if qn not in active_qubits}
    qubit_freqs = {qn: qubits[qn]['wq'] for qn in inactive_qubits}
    qubit_ef = {qn: False for qn in inactive_qubits}
    step_time = 0
    fidelities = {}
    for g in wints:
        step_time = max(step_time, 1 / (np.sqrt(8) * couplings[g]))
        for qn, k_anh, ef in zip(sorted(g, key=lambda qn: qubits[qn]['wq']),
                                 [0, 1], [False, True]):
            qubit_freqs[qn] = wints[g] - k_anh * qubits[qn]['anh']
            qubit_ef[qn] = ef
    for qn in inactive_qubits:
        specs = [
            (qubit_freqs[qn2], qubits[qn2]['anh'],
             couplings[frozenset({qn, qn2})], qubit_ef[qn2])
            for qn2 in qubits[qn]['neighbors']
        ]
        fidelities[qn] = \
            calculate_idle_spec_fidelity(qubit_freqs[qn], qubits[qn]['anh'],
                                         specs, step_time)

    for g in wints:
        qn1, qn2 = sorted(g, key=lambda qn: qubits[qn]['wq'])
        specs1, specs2 = [
            [
                (qubit_freqs[qn3], qubits[qn3]['anh'],
                 couplings[frozenset({qn, qn3})], qubit_ef[qn3])
                for qn3 in qubits[qn]['neighbors'] if qn3 not in g
            ] for qn in [qn1, qn2]
        ]
        fid = calculate_gate_spec_fidelity(wints[g], qubits[qn1]['anh'],
                                           qubits[qn2]['anh'],
                                           specs1, specs2, step_time)
        fidelities[qn1] = np.sqrt(fid)
        fidelities[qn2] = np.sqrt(fid)
    return fidelities


class InvalidParallelGatesError(Exception):
    """Parallel gate interaction frequency finding problem has no solution"""
    pass


def find_interaction_frequencies(qubits, couplings, gates, method='optimize'):
    """
    Find optimal interaction frequencies

    Args:
        qubits: dict[str, dict[str, any]]
            A dictionary from qubit names to properties. The following
            properties should be defined:
                neighbors: a set of qubit names that neighbor this qubit
                wq: qubit parking frequency
                anh: qubit anharmonicity (negative for transmon qubits)
        couplings: dict[frozenset[str], float]
            Qubit-qubit coupling rates. Keys are frozensets of coupled qubit
            names and values are the 01-10 coupling rates.
        gates: collection[frozenset[str]]
            A collection of gates (frozenset of involved qubit names) that will
            be executed in parallel.
        method: str
            Method for choosing the interaction frequencies. Valid options:
                'equal_spacing': maximizes the minimal detuning from all
                    spurious resonant interactions
                'optimize': minimizes total residual interaction error
    Returns: dict[frozenset[str], float]
        For each gate (key), the optimal interaction frequency
    """
    # connected subsets of qubits
    not_assigned_to_group = {qn for g in gates for qn in g}
    connected_groups = set()
    while len(not_assigned_to_group) != 0:
        # find a set of connected qubits (group)
        to_visit = {(next(iter(not_assigned_to_group)), None)}
        group = []  # all active qubits involved in the step and their neighbors
        while len(to_visit) != 0:
            qni, qn_from = to_visit.pop()
            group.append((qni, qn_from))
            for qnj in qubits[qni]['neighbors']:
                if qnj in not_assigned_to_group and qnj != qn_from:
                    if qnj in group:
                        raise InvalidParallelGatesError(f'Cycle: {group}')
                    to_visit.add((qnj, qni))
        not_assigned_to_group -= {g[0] for g in group}
        connected_groups.add(tuple(group))

    # neighboring gate graphs
    gate_graphs = []
    inv_gate_graphs = []
    for group in connected_groups:
        unordered_gates = {tuple(sorted(g, key=lambda qn: qubits[qn]['wq'])) for
                           g in group if frozenset(g) in gates}
        unordered_connections = {frozenset(g) for g in group if
                                 frozenset(g) not in gates and None not in g}
        gate_graph = {}  # from low int. freq gates to high int. freq gates
        inv_gate_graph = {}  # from high int. freq gates to low int. freq gates
        for g in unordered_gates:
            gate_graph[g] = set()
            inv_gate_graph[g] = set()
            for c in unordered_connections:
                if g[0] in c:
                    qn = [qn for qn in c if qn != g[0]][0]
                    for g2 in unordered_gates:
                        if g2[0] == qn:
                            raise InvalidParallelGatesError(
                                f'Opposing gate directions: {g}, {g2}')
                        if g2[1] == qn:
                            gate_graph[g].add(g2)
                if g[1] in c:
                    qn = [qn for qn in c if qn != g[1]][0]
                    for g2 in unordered_gates:
                        if g2[1] == qn:
                            raise InvalidParallelGatesError(
                                f'Opposing gate directions: {g}, {g2}')
                        if g2[0] == qn:
                            inv_gate_graph[g].add(g2)
        gate_graphs.append(gate_graph)
        inv_gate_graphs.append(inv_gate_graph)

    # optimize each gate graph
    gate_wints = {}
    if method == 'equal_spacing':
        # interactions to avoid:
        #   for high-freq int. qubits: 2x anh below (1x anh above)
        #   for low-freq int. qubits: 1x anh above (1x anh below)

        # strategy: start traversing gate graph from low to high, placing each
        # interaction frequency as low as possible and remember, how much
        # higher it could be placed, and in which layer this gate is
        # finally distribute the minimal slack equally
        for gate_graph, inv_gate_graph in zip(gate_graphs, inv_gate_graphs):
            layer = 0
            gate_layers_wints = {}
            next_gates = {g for g, conns in inv_gate_graph.items() if
                          len(conns) == 0}
            while len(next_gates) != 0:
                new_next_gates = set()
                for g in next_gates:
                    wintmin = qubits[g[0]]['wq']
                    wintmax = qubits[g[1]]['wq'] + qubits[g[1]]['anh']
                    # low qubit interactions with parked spectator qubits
                    for qn in qubits[g[0]]['neighbors']:
                        if qn != g[1]:
                            wintmax = min(wintmax,
                                          qubits[qn]['wq'] + qubits[qn]['anh'])
                    # high qubit interactions with parked spectator qubits
                    for qn in qubits[g[1]]['neighbors']:
                        if qn != g[0]:
                            wintmin = max(wintmin,
                                          qubits[qn]['wq'] + qubits[g[1]][
                                              'anh'])
                    # interactions with neighboring gates
                    for g2 in inv_gate_graph[g]:
                        wintmin = max(wintmin,
                                      gate_layers_wints[g2][1] - qubits[g[1]][
                                          'anh'])
                    gate_layers_wints[g] = (layer, wintmin, wintmax)
                    new_next_gates |= gate_graph[g]
                next_gates = new_next_gates
                layer += 1
            slack = min([(wintmax - wintmin) / (layer + 2) for
                         g, (layer, wintmin, wintmax) in
                         gate_layers_wints.items()])
            for g, (layer, wintmin, wintmax) in gate_layers_wints.items():
                gate_wints[frozenset(g)] = wintmin + (layer + 1) * slack
    elif method == 'optimize':
        gate_wints = find_interaction_frequencies(qubits, couplings, gates,
                                                  method='equal_spacing')
        x0 = [gate_wints[g] for g in gates]

        def cost_func(x):
            fid = calculate_step_fidelity(qubits, couplings,
                                          dict(zip(gates, x)))
            return -np.sum(np.log([f for f in fid.values()]))

        x1 = sp.optimize.minimize(cost_func, x0).x
        x2 = sp.optimize.minimize(cost_func, x1).x
        gate_wints = dict(zip(gates, x2))
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid options are: "
            "['equal_spacing', 'optimize']")
    return gate_wints
