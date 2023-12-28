import numpy as np
import matplotlib.pyplot as plt


def compute_energy_levels_from_flux_pulse(t, pulse, qubits, states,
                                          interpolations=None):
    """Computes the frequency of the specified energy levels for the times
    specified in t taking the flux pulse into account.

    Args:
        t (numpy.array): numpy array specifying the times at which the frequency
            will be computed.
        pulse (Pulse): Pulse object that defines the flux waveforms. If
            `pulse.channels` does not contain the `flux_pulse_channel` of a
            given qubit, it is assumed that no flux pulse is applied to that
            qubit (i.e., it is fixed at `qb.flux_parking`).
        qubits (list[QuDev_transmon]): Qubit objects. The order specifies the
            basis for the state vectors (used in argument `states`).
            They do not need to be part of the flux pulse.
        states (list[str]): List of strings. Each string specifying a mutli
            qubit state for which the frequency will be calculated from the
            provided flux pulse, e.g.: states=['01', '20']. Valid single qb
            states include 0, 1, 2.
        interpolations (list[callable]): List of callable objects that has the
            same signature as QuDev_transmon.calculate_frequency. This can be
            used to pass an interpolated Hamiltonian model
            (`InterpolatedHamiltonianModel`) to speed up the computation. If
            set to None, qb.calculate_frequency is used instead.
            Defaults to None.

    Returns
        numpy.array of shape (len(states), len(t)) with frequency values.
    """
    # get pulse waveforms for involved qubits:
    wf = [pulse.chan_wf(qb.flux_pulse_channel(), t)
          if qb.flux_pulse_channel() in pulse.channels else np.zeros_like(t)
          for qb in qubits]
    # choose methods to compute frequency evolutions:
    if interpolations is not None:
        calculate_frequency = [interpolations[i] for i, _ in enumerate(qubits)]
    else:
        calculate_frequency = [qb.calculate_frequency
                               for i, qb in enumerate(qubits)]
    # compute evolution of ge and ef transition frequencies for all qubits:
    # For each time in t, we compute the ge and ef transition frequency of each
    # qubit. In the end, we want an array of the same length as t with each
    # element in this array providing the frequencies at that time in the
    # shape [f_ge_qb1, f_ef_qb1, f_ge_qb2, f_ef_qb2] for the example of two
    # qubits. This format is achieved by the reshape.
    qb_freqs = np.reshape([calculate_frequency[i](amplitude=wf[i],
                                                  transition=['ge', 'ef'])
                           for i, qb in enumerate(qubits)],
                          (2 * len(qubits), len(t)))
    # We define a helper matrix to map from ge & ef transitions to energy
    # levels 0 (gg), 1 (ge), 2 (gf = ge + ef):
    transition2energy_level_matrix = np.array([[0, 0], [1, 0], [1, 1]])
    # The `state_matrix` maps the ge and ef transition frequencies of each
    # qubit to the multi-qubit states in the list `states`, e.g.,
    #    states = ['ee', 'ef']
    # will result in
    #    state_matrix = [[1, 0, 1, 0], [1, 0, 1, 1]]
    # with each row refering to [f_ge_qb1, f_ef_qb1, f_ge_qb2, f_ef_qb2].
    state_matrix = np.zeros(shape = (len(states), 2 * len(qubits)))
    for i, state in enumerate(states):
        for j, state_qb_j in enumerate(state):
            state_matrix[i, 2 * j : 2 * j + 2] += \
                transition2energy_level_matrix[int(state_qb_j)]
    # We apply the state_matrix element-wise to the the qb_freqs array. The
    # method np.einsum is used only to implement the operation element-wise.
    state_freqs = np.einsum('ij,jk', state_matrix, qb_freqs)
    return state_freqs


def plot_state_freqs(t, states, state_freqs, additional_states_matrix=None,
                     only_plot_additional_states=False, colors=dict()):
    """Plots the time evolution of the state frequencies in one plot.

    Args:
        t (numpy.array): numpy array specifying the time for which the frequency
            will be computed.
        states (list[str]): Labels of the states.
        state_freqs: numpy.array of shape (len(states), len(t)) with frequency
            values computed, e.g., using `compute_energy_levels_from_flux_pulse`
        additional_states_matrix: list of additional states to plot, each
            expressed as a linear combination of states present in
            state_freqs.
            For example:
                With `state_freqs` holding the 11 and 20 state frequencies and
                with `additional_states_matrix` being [[1, 1], [1, -1]], the
                additionally computed energies would be 11+20 and 11-20.
        only_plot_additional_states (bool): Whether to plot only the states
            computed from `additional_states_matrix`. Defaults to `False`.
        colors (dict[matplotlib colors]): Dict mapping state labels to valid
            matplotlib colors.

    Returns
        plt.figure object of the plot
    """
    fig = plt.figure()
    if not only_plot_additional_states:
        for i, state in enumerate(states):
            plt.plot(1e9*t, 1e-9*state_freqs[i], label=state,
                color=colors.get(states[i], None))
    if additional_states_matrix is not None:
        additional_states_freqs = np.einsum('ij,jk', additional_states_matrix,
                                            state_freqs)
        for i, v in enumerate(additional_states_matrix):
            freqs = additional_states_freqs[i]
            label = ''
            for j, factor in enumerate(v):
                if factor == 0:
                    continue
                elif abs(factor) == 1:
                    label += (f'{factor:+}')[0]
                    label += f'{states[j]}'
                else:
                    label += f' {factor:+} * {states[j]}'
            plt.plot(1e9*t, 1e-9*freqs, label=label)
    plt.ylabel('Frequency, $f$ (GHz)')
    plt.xlabel('Time, $t$ (ns)')
    plt.legend()
    return fig


def compute_accumulated_phase(t, states, state_freqs):
    """Uses np.trapz to integrate the accumulated phase between states.

    Args:
        t (numpy.array): numpy array specifying the sample points in time.
        states (tuple[str]): Tuple of 1 or 2 multi qb state strings for
            which the accumulated phase will be calculated. See docstring of
            compute_energy_levels_from_flux_pulse for syntax. If only one state is provided the phase will be computed from the frequency difference relative to the first frequency in state_freqs[0].
        state_freqs (np.array): Must have shape
            (len(states), len(t)).

    Returns:
        float: accumulated phase in deg
    """
    if len(states) == 1:
        diff_freq = state_freqs[0] - state_freqs[0][0]
    else:
        diff_freq = state_freqs[1] - state_freqs[0]
    return np.trapz(diff_freq, t) * 360