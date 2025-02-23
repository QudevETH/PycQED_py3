from copy import deepcopy

import numpy as np
import itertools
import logging

from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import \
    generate_mux_ro_pulse_list
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqtse
from pycqed.measurement.waveform_control import segment

log = logging.getLogger(__name__)


class CalibrationPoints:
    def __init__(self, qb_names, states, **kwargs):
        self.qb_names = qb_names
        self.states = states
        default_map = dict(g=['I '], e=["X180 "], f=['X180 ', "X180_ef "],
                           h=['X180 ', "X180_ef ", "X180_fh "])
        self.pulse_label_map = kwargs.get("pulse_label_map", default_map)
        self.pulse_modifs = kwargs.get('pulse_modifs', None)

    def create_segments(self, operation_dict, pulse_modifs=None,
                        segment_prefix='calibration_',
                        reset_params=None):
        segments = []
        if pulse_modifs is None:
            pulse_modifs = dict()
        if self.pulse_modifs is not None:
            pm = deepcopy(self.pulse_modifs)
            pm.update(pulse_modifs)
            pulse_modifs = pm

        for i, seg_states in enumerate(self.states):
            pulse_list = []
            for j, qbn in enumerate(self.qb_names):
                unique, counts = np.unique(self.get_states(qbn)[qbn],
                                           return_counts=True)
                cal_pt_idx = i % np.squeeze(counts[np.argwhere(unique ==
                                                               seg_states[j])])
                for k, pulse_name in enumerate(self.pulse_label_map[seg_states[j]]):
                    pulse = deepcopy(operation_dict[pulse_name + qbn])
                    pulse['name'] = f"{seg_states[j]}_{pulse_name + qbn}_" \
                                    f"{cal_pt_idx}"

                    if k == 0:
                        pulse['ref_pulse'] = 'segment_start'
                    if len(pulse_modifs) > 0:
                        # The pulse(s) to which the pulse_modifs refer might
                        # not be present in all calibration segments. We
                        # thus disable the pulse_not_found_warning.
                        pulse = sqtse.sweep_pulse_params(
                            [pulse], pulse_modifs,
                            pulse_not_found_warning=False)[0][0]
                        # reset the name as sweep_pulse_params deletes it
                        pulse['name'] = f"{seg_states[j]}_{pulse_name + qbn}_" \
                                        f"{cal_pt_idx}"
                    pulse_list.append(pulse)
            state_prep_pulse_names = [p['name'] for p in pulse_list]
            pulse_list = sqtse.add_preparation_pulses(pulse_list,
                                                operation_dict,
                                                [qbn for qbn in self.qb_names],
                                                reset_params=reset_params)

            ro_pulses = generate_mux_ro_pulse_list(self.qb_names,
                                                     operation_dict)
            # reference all readout pulses to all pulses of the pulse list, to ensure
            # readout happens after the last pulse (e.g. if doing "f" on some qubits
            # and "e" on others). In the future we could use the circuitBuilder
            # and Block here
            [rp.update({'ref_pulse': state_prep_pulse_names, "ref_point":'end'})
             for rp in ro_pulses]

            pulse_list += ro_pulses

            seg = segment.Segment(segment_prefix + f'{i}', pulse_list)
            segments.append(seg)

        return segments

    def get_states(self, qb_names=None):
        """
        Get calibrations states given a (subset) of qubits of self.qb_names.
        This function is a helper for the analysis which works with information
        per qubit.
        Args:
            qb_names: list of qubit names

        Returns: dict where keys are qubit names and values are the calibration
            states for this particular qubit
        """

        qb_names = self._check_qb_names(qb_names)

        return {qbn: [s[self.qb_names.index(qbn)] for s in self.states]
                for qbn in qb_names}

    def get_indices(self, qb_names=None, prep_params=None):
        """
        Get calibration indices
        Args:
            qb_names: qubit name or list of qubit names to retrieve
                the indices of. Defaults to all.
            prep_params: QuDev_transmon preparation_params attribute

        Returns: dict where keys are qb_names and values dict of {state: ind}

        """
        if prep_params is None:
            prep_params = {}
        prep_type = prep_params.get('preparation_type', 'wait')

        qb_names = self._check_qb_names(qb_names)
        indices = dict()
        states = self.get_states(qb_names)

        for qbn in qb_names:
            unique, idx, inv = np.unique(states[qbn], return_inverse=True,
                                        return_index=True)
            if prep_type == 'preselection':
                indices[qbn] = {s: [-2*len(states[qbn]) + 2*j + 1
                                    for j in range(len(inv)) if i == inv[j]]
                                for i, s in enumerate(unique)}
            elif 'active_reset' in prep_type:
                reset_reps = prep_params['reset_reps']
                indices[qbn] = {s: [-(reset_reps+1)*len(states[qbn]) +
                                    reset_reps*(j + 1)+j for j in
                                    range(len(inv)) if i == inv[j]]
                                for i, s in enumerate(unique)}
            else:
                indices[qbn] = {s: [-len(states[qbn]) + j
                                    for j in range(len(inv)) if i == inv[j]]
                                for i, s in enumerate(unique)}

        log.info(f"Calibration Points Indices: {indices}")
        return indices

    def _check_qb_names(self, qb_names):
        if qb_names is None:
            qb_names = self.qb_names
        elif np.ndim(qb_names) == 0:
            qb_names = [qb_names]
        for qbn in qb_names:
            assert qbn in self.qb_names, f"{qbn} not in Calibrated Qubits: " \
                f"{self.qb_names}"

        return qb_names

    def get_rotations(self, last_ge_pulses=False, qb_names=None,
                      enforce_two_cal_states=False, **kw):
        """
        Get rotation dictionaries for each qubit in qb_names,
        as used by the analysis for plotting.
        Args:
            qb_names (list or string): qubit names. Defaults to all.
            last_ge_pulses (list or bool): one for each qb in the same order as
                specified in qb_names
            kw: keyword arguments (to allow pass through kw even if it
                contains entries that are not needed)
        Returns:
             dict where keys are qb_names and values are dict specifying
             rotations.

        """
        qb_names = self._check_qb_names(qb_names)
        states = self.get_states(qb_names)
        if isinstance(last_ge_pulses, bool) and len(qb_names) > 1:
            last_ge_pulses = len(qb_names)*[last_ge_pulses]
        rotations = dict()

        if len(qb_names) == 0:
            return rotations
        if 'f' in [s for v in states.values() for s in v]:
            if len(qb_names) == 1:
                last_ge_pulses = [last_ge_pulses] if \
                    isinstance(last_ge_pulses, bool) else last_ge_pulses
            else:
                i, j = len(qb_names), \
                       1 if isinstance(last_ge_pulses, bool) else \
                           len(last_ge_pulses)
                assert i == j, f"Size of qb_names and last_ge_pulses don't " \
                    f"match: {i} vs {j}"

        for i, qbn in enumerate(qb_names):
            # get unique states in the order specified below
            order = {"g": 0, "e": 1, "f": 2, "h": 3}
            unique = list(np.unique(states[qbn]))
            unique.sort(key=lambda s: order[s])
            if len(unique) == 3 and enforce_two_cal_states:
                unique = np.delete(unique, 1 if last_ge_pulses[i] else 0)
            rotations[qbn] = {unique[i]: i for i in range(len(unique))}
        log.info(f"Calibration Points Rotation: {rotations}")
        return rotations

    @staticmethod
    def single_qubit(qubit_name, states, n_per_state=2):
        return CalibrationPoints.multi_qubit([qubit_name], states, n_per_state)

    @staticmethod
    def multi_qubit(qb_names, states, n_per_state=2, all_combinations=False):
        """
        Creates calibration points for multiple qubits. See docstring of
        MultiTaskingExperiment.create_cal_points() for details.
        """
        n_qubits = len(qb_names)
        if n_qubits == 0:
            return CalibrationPoints(qb_names, [])

        if np.ndim(states) == 2:  # handle custom list of cal_states
            # check if cal_states list has as many states as meas_objs
            if len(qb_names) != len(states[0]):
                raise ValueError(
                    f"{len(qb_names)} measurement objects were "
                    f" given but custom states were specified for "
                    f"{len(states[0])} measurement objects (qubits).")
            if all_combinations:
                log.warning(f"Provided custom cal_states, thus ignoring "
                            f"all_states_combinations=True.")
            labels = states
        elif all_combinations:
            labels_array = np.tile(
                list(itertools.product(states, repeat=n_qubits)), n_per_state)
            labels = [tuple(seg_label)
                      for seg_label in labels_array.reshape((-1, n_qubits))]
        else:
            labels =[tuple(np.repeat(tuple([state]), n_qubits))
                     for state in states for _ in range(n_per_state)]

        return CalibrationPoints(qb_names, labels)

    @staticmethod
    def from_string(cal_points_string):
        """
        Recreates a CalibrationPoints object from a string representation.
        Avoids having "eval" statements throughout the codebase.
        Args:
            cal_points_string: string representation of the CalibrationPoints

        Returns: CalibrationPoint object
        Examples:
            >>> cp = CalibrationPoints(['qb1'], ['g', 'e'])
            >>> cp_repr = repr(cp) # create string representation
            >>> # do other things including saving cp_str somewhere
            >>> cp = CalibrationPoints.from_string(cp_repr)
        """
        return eval(cal_points_string)

    def extend_sweep_points(self, sweep_points, qb_name):
        """
        Extends the sweep_points array for plotting calibration points after
        data for a particular qubit.
        Args:
            sweep_points (array): physical sweep_points
            qb_name (str): qubit name
        Returns:
            sweep_points + calib_fake_sweep points
        """
        n_cal_pts = len(self.get_states(qb_name)[qb_name])
        return self.extend_sweep_points_by_n_cal_pts(n_cal_pts, sweep_points)

    @staticmethod
    def extend_sweep_points_by_n_cal_pts(n_cal_pts, sweep_points):
        """
        Extends the sweep_points array by n_cal_pts for the calibration points.
        Args:
            n_cal_pts (int): number of calibration points
            sweep_points (array): physical sweep_points
        Returns:
            sweep_points + calib_fake_sweep points
        """
        if len(sweep_points) == 0:
            log.warning("No sweep points, returning a range.")
            return np.arange(n_cal_pts)
        if n_cal_pts == 0:
            return sweep_points
        try:
            step = sweep_points[-1] - sweep_points[-2]
        except IndexError:
            # This fallback is used to have a step value in the same order
            # of magnitude as the value of the single sweep point
            step = np.abs(sweep_points[0])
        except Exception:
            return np.arange(len(sweep_points) + n_cal_pts)
        plot_sweep_points = \
            np.concatenate([sweep_points, [sweep_points[-1] + i * step
                                           for i in range(1, n_cal_pts + 1)]])

        return plot_sweep_points

    def __str__(self):
        return "Calibration:\n    Qubits: {}\n    States: {}" \
            .format(self.qb_names, self.states)

    def __repr__(self):
        return "CalibrationPoints(\n    qb_names={},\n    states={}," \
               "\n    pulse_label_map={})" \
            .format(self.qb_names, self.states, self.pulse_label_map)

    @staticmethod
    def combine_parallel(first, second):
        """Combines two CalibrationPoints objects into a new CalibrationPoints
        object that represents the two calibration point sets played in
        parallel.

        Args:
            first, second:
                The two CalibrationPoints objects to be combined
        Returns:
            The combined CalibrationPoints object.
        """

        if first.pulse_label_map != second.pulse_label_map:
            raise ValueError("pulse_label_map's of combined CalibrationPoints "
                             "must be identical")
        if first.pulse_modifs != second.pulse_modifs:
            raise ValueError("pulse_modifs's of combined CalibrationPoints "
                             "must be identical")
        # dicts preserve insertion order, sets do not, therefore we use dicts
        qb_names = list(dict.fromkeys(first.qb_names + second.qb_names))
        first_states = first.states.copy()
        second_states = second.states.copy()
        nstates = max(len(first_states), len(second_states))
        while len(first_states) < nstates:
            first_states += [len(first.qb_names) * ['I ']]
        while len(second_states) < nstates:
            second_states += [len(second.qb_names) * ['I ']]
        states = []
        for first_state, second_state in zip(first_states, second_states):
            # loop over calibration segments
            states.append([])
            for qb in qb_names:
                # determine state for each qubit in this calibration segment
                idx_first = first.qb_names.index(qb) \
                    if qb in first.qb_names else None
                idx_second = second.qb_names.index(qb) \
                    if qb in second.qb_names else None
                if idx_first is not None and idx_second is not None:
                    if first_state[idx_first] != second_state[idx_second]:
                        raise ValueError("Same qubit should be prepared in "
                                         "different states in same segment in "
                                         "CalibrationPoints.combine_parallel")
                    states[-1].append(first_state[idx_first])
                elif idx_first is not None:
                    states[-1].append(first_state[idx_first])
                else:
                    states[-1].append(second_state[idx_second])
        return CalibrationPoints(qb_names, states,
                                 pulse_label_map=first.pulse_label_map,
                                 pulse_modifs=first.pulse_modifs)

    @staticmethod
    def guess_cal_states(cal_states, for_ef=False, **kw):
        """
        Generate calibration states to be passed to CalibrationPoints
        :param cal_states: str or list of str with state names. If 'auto', it
            will generate default states based on for_ef and transition_names.
        :param for_ef: bool specifying whether to add the 'f' state.
            This flag is here for legacy reasons (Steph, 07.10.2020).
        :param kw: keyword_arguments (to allow pass-through kw even if it
                    contains entries that are not needed)
        :return: tuple of calibration states or cal_states from the user
        """
        if cal_states == "auto":
            cal_states = ('g', 'e')
            if for_ef:
                cal_states += ('f',)
        return cal_states

