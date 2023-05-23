import numpy as np
import matplotlib.pyplot as plt


def compute_energy_levels_from_flux_pulse(t, pulse, qubits, states,
                                          interpolations=None):
    """Computes the frequency of the specified energy levels for the times
    specified in t taking the flux pulse into account.

    Args:
        t (numpy.array): numpy array specifying the time for which the frequency
            will be computed.
        pulse (Pulse): Pulse object that contains defined the flux waveforms.
            If the flux_pulse_channel of a qubit is not contained in
            pulse.channels it is assumed that this qubit has a fixed flux
            defined by flux_parking.
        qubits (list[QuDev_transmon]): Qubit objects. The order specifies the
            basis for the state vectors. They do not need to be part of the
            flux pulse.
        states (list[str]): List of strings. Each string specifying a mutli
            qubit state for which the frequency will be calculated from the
            provided flux pulse, e.g.: states=['01', '20']. Valid single qb
            states include 0 (g), 1 (e) & 2 (f)
        interpolations (list[callable]): List of callable objects that has the
            same signature as QuDev_transmon.calculate_frequency. This can be
            used to pass an interpolated Hamiltonian model to speed up the
            computation. If set to None, qb.calculate_frequency is used instead.
            Defaults to None.

    Returns
        numpy.array of shape (len(states), len(t)) with frequency values.
    """
    # mapping from basis = [ge, ef] to energy levels 0, 1, 2:
    sg_qb_mat = np.array([[0, 0], [1, 0], [1, 1]])
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
    qb_freqs = np.reshape(np.array([np.reshape(calculate_frequency[i](
                                                    amplitude=wf[i],
                                                    return_ge_and_ef=True),
                                               (2, -1))
                                    * np.ones((2, len(t)))
                                    for i, qb in enumerate(qubits)]),
                         (2 * len(qubits), len(t)))
    # The `state_matrix` maps the ge and ef transition freuqency of each qubit
    # to the states in the list `states`.
    state_matrix = np.zeros(shape = (len(states), 2 * len(qubits)))
    for i, state in enumerate(states):
        for j, state_qb_j in enumerate(state):
            state_matrix[i, 2 * j : 2 * j + 2] += sg_qb_mat[int(state_qb_j)]
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
        additional_states_matrix: Matrix to compute additional states from
            `state_freqs`, e.g., the difference between two states in `state_freqs`.
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
        additional_states_labels = []
        for i, v in enumerate(additional_states_matrix):
            additional_states_labels.append('')
            for j, factor in enumerate(v):
                if factor == 0:
                    continue
                elif abs(factor) == 1:
                    additional_states_labels[i] += (f'{factor:+}')[0]
                    additional_states_labels[i] += f'{states[j]}'
                else:
                    additional_states_labels[i] += f' {factor:+} * {states[j]}'
        for i, freqs in enumerate(additional_states_freqs):
            plt.plot(1e9*t, 1e-9*freqs, label=additional_states_labels[i],
                color=colors.get(states[i], None))
    plt.ylabel('Frequency, $f$ (GHz)')
    plt.xlabel('Time, $t$ (ns)')
    plt.legend()
    return fig


def compute_accumulated_phase(t, pulse, qubits, states=('11', '20'),
                              state_freqs=None, interpolations=None):
    """Uses np.trapz to integrate the accumulated phase between states.

    Args:
        t (numpy.array): numpy array specifying the sample points in time.
        pulse (Pulse): Pulse object that contains defined the flux waveforms.
            If the flux_pulse_channel of a qubit is not contained in
            pulse.channels it is assumed that this qubit has a fixed flux
            defined by flux_parking.
        qubits (list[QuDev_transmon]): Qubit objects. The order specifies the
            basis for the state vectors. They do not need to be part of the
            flux pulse.
        states (tuple[str], optional): Tuple of 2 multi qb state strings for
            which the accumulated phase will be calculated. See docstring of
            compute_energy_levels_from_flux_pulse for syntax.
            Defaults to ('11', '20').
        state_freqs (np.array, optional): If None
            compute_energy_levels_from_flux_pulse is used to compute the state
            frequencies during the pulse. If provided it must have shape
            (len(states), len(t)). Defaults to None.
        interpolations (list[callable]): See docstring of
            `compute_energy_levels_from_flux_pulse`.

    Returns:
        float: accumulated phase in deg
    """
    if state_freqs is None:
        state_freqs = compute_energy_levels_from_flux_pulse(t, pulse, qubits,
                                                            states,
                                                            interpolations)
    if len(states) == 1:
        diff_freq = state_freqs[0][1] - state_freqs[0][0]
    else:
        diff_freq = state_freqs[1] - state_freqs[0]
    return np.trapz(diff_freq, t) * 360