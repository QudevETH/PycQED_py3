from copy import deepcopy

import numpy as np
import itertools
import logging

from pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts import \
    generate_mux_ro_pulse_list
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    add_preparation_pulses, sweep_pulse_params
from pycqed.measurement.waveform_control import segment

log = logging.getLogger(__name__)


class CalibrationPoints:
    def __init__(self, qb_names, states, **kwargs):
        self.qb_names = qb_names
        self.states = states
        default_map = dict(g=['I '], e=["X180 "], f=['X180 ', "X180_ef "])
        self.pulse_label_map = kwargs.get("pulse_label_map", default_map)
        self.pulse_modifs = kwargs.get('pulse_modifs', None)

    def create_segments(self, operation_dict, pulse_modifs=None,
                        segment_prefix='calibration_',
                        **prep_params):
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
                        pulse = sweep_pulse_params(
                            [pulse], pulse_modifs,
                            pulse_not_found_warning=False)[0][0]
                        # reset the name as sweep_pulse_params deletes it
                        pulse['name'] = f"{seg_states[j]}_{pulse_name + qbn}_" \
                                        f"{cal_pt_idx}"
                    pulse_list.append(pulse)
            state_prep_pulse_names = [p['name'] for p in pulse_list]
            pulse_list = add_preparation_pulses(pulse_list,
                                                operation_dict,
                                                [qbn for qbn in self.qb_names],
                                                **prep_params)

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
            # get unique states in reversed alphabetical order: g, [e, f]
            order = {"g": 0, "e": 1, "f": 2}
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
        n_qubits = len(qb_names)
        if n_qubits == 0:
            return CalibrationPoints(qb_names, [])

        if all_combinations:
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
        try:
            step = np.abs(sweep_points[-1] - sweep_points[-2])
        except IndexError:
            # This fallback is used to have a step value in the same order
            # of magnitude as the value of the single sweep point
            step = np.abs(sweep_points[0])
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
    def guess_cal_states(cal_states, for_ef=False):
        if cal_states == "auto":
            cal_states = ('g', 'e')
            if for_ef:
                cal_states += ('f',)
        return cal_states


