import lmfit
import numpy as np
from numpy.linalg import inv
import scipy as sp
import itertools
import matplotlib as mpl
from collections import OrderedDict, defaultdict

from sklearn.mixture import GaussianMixture as GM
from sklearn.tree import DecisionTreeClassifier as DTC

from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.readout_analysis as roa
from pycqed.analysis_v2.readout_analysis import \
    Singleshot_Readout_Analysis_Qutrit as SSROQutrit
import pycqed.analysis_v2.tomography_qudev as tomo
import re
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from copy import deepcopy
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration_points import CalibrationPoints
import matplotlib.pyplot as plt
from pycqed.analysis.three_state_rotation import predict_proba_avg_ro
import logging

from pycqed.utilities import math

log = logging.getLogger(__name__)
try:
    import qutip as qtp
except ImportError as e:
    log.warning('Could not import qutip, tomography code will not work')


class AveragedTimedomainAnalysis(ba.BaseDataAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_timestamp = True
        self.params_dict = {
            'value_names': 'value_names',
            'measured_values': 'measured_values',
            'measurementstring': 'measurementstring',
            'exp_metadata': 'exp_metadata'}
        self.numeric_params = []
        if kwargs.get('auto', True):
            self.run_analysis()

    def process_data(self):
        self.metadata = self.raw_data_dict.get('exp_metadata', {})
        if self.metadata is None:
            self.metadata = {}
        cal_points = self.metadata.get('cal_points', None)
        cal_points = self.options_dict.get('cal_points', cal_points)
        cal_points_list = roa.convert_channel_names_to_index(
            cal_points, len(self.raw_data_dict['measured_values'][0]),
            self.raw_data_dict['value_names'])
        self.proc_data_dict['cal_points_list'] = cal_points_list
        measured_values = self.raw_data_dict['measured_values']
        cal_idxs = self._find_calibration_indices()
        scales = [np.std(x[cal_idxs]) for x in measured_values]
        observable_vectors = np.zeros((len(cal_points_list),
                                       len(measured_values)))
        observable_vector_stds = np.ones_like(observable_vectors)
        for i, observable in enumerate(cal_points_list):
            for ch_idx, seg_idxs in enumerate(observable):
                x = measured_values[ch_idx][seg_idxs] / scales[ch_idx]
                if len(x) > 0:
                    observable_vectors[i][ch_idx] = np.mean(x)
                if len(x) > 1:
                    observable_vector_stds[i][ch_idx] = np.std(x)
        Omtx = (observable_vectors[1:] - observable_vectors[0]).T
        d0 = observable_vectors[0]
        corr_values = np.zeros(
            (len(cal_points_list) - 1, len(measured_values[0])))
        for i in range(len(measured_values[0])):
            d = np.array([x[i] / scale for x, scale in zip(measured_values,
                                                           scales)])
            corr_values[:, i] = inv(Omtx.T.dot(Omtx)).dot(Omtx.T).dot(d - d0)
        self.proc_data_dict['corr_values'] = corr_values

    def measurement_operators_and_results(self):
        """
        Converts the calibration points to measurement operators. Assumes that
        the calibration points are ordered the same as the basis states for
        the tomography calculation (e.g. for two qubits |gg>, |ge>, |eg>, |ee>).
        Also assumes that each calibration in the passed cal_points uses
        different segments.

        Returns:
            A tuple of
                the measured values with outthe calibration points;
                the measurement operators corresponding to each channel;
                and the expected covariation matrix between the operators.
        """
        d = len(self.proc_data_dict['cal_points_list'])
        cal_point_idxs = [set() for _ in range(d)]
        for i, idxs_lists in enumerate(self.proc_data_dict['cal_points_list']):
            for idxs in idxs_lists:
                cal_point_idxs[i].update(idxs)
        cal_point_idxs = [sorted(list(idxs)) for idxs in cal_point_idxs]
        cal_point_idxs = np.array(cal_point_idxs)
        raw_data = self.raw_data_dict['measured_values']
        means = [None] * d
        residuals = [list() for _ in raw_data]
        for i, cal_point_idx in enumerate(cal_point_idxs):
            means[i] = [np.mean(ch_data[cal_point_idx]) for ch_data in raw_data]
            for j, ch_residuals in enumerate(residuals):
                ch_residuals += list(raw_data[j][cal_point_idx] - means[i][j])
        means = np.array(means)
        residuals = np.array(residuals)
        Fs = [np.diag(ms) for ms in means.T]
        Omega = residuals.dot(residuals.T) / len(residuals.T)
        data_idxs = np.setdiff1d(np.arange(len(raw_data[0])),
                                 cal_point_idxs.flatten())
        data = np.array([ch_data[data_idxs] for ch_data in raw_data])
        return data, Fs, Omega

    def _find_calibration_indices(self):
        cal_indices = set()
        cal_points = self.options_dict['cal_points']
        nr_segments = self.raw_data_dict['measured_values'].shape[-1]
        for observable in cal_points:
            if isinstance(observable, (list, np.ndarray)):
                for idxs in observable:
                    cal_indices.update({idx % nr_segments for idx in idxs})
            else:  # assume dictionaries
                for idxs in observable.values():
                    cal_indices.update({idx % nr_segments for idx in idxs})
        return list(cal_indices)


def all_cal_points(d, nr_ch, reps=1):
    """
    Generates a list of calibration points for a Hilbert space of dimension d,
    with nr_ch channels and reps reprtitions of each calibration point.
    """
    return [[list(range(-reps*i, -reps*(i-1)))]*nr_ch for i in range(d, 0, -1)]


class Single_Qubit_TimeDomainAnalysis(ba.BaseDataAnalysis):

    def process_data(self):
        """
        This takes care of rotating and normalizing the data if required.
        this should work for several input types.
            - I/Q values (2 quadratures + cal points)
            - weight functions (1 quadrature + cal points)
            - counts (no cal points)

        There are several options possible to specify the normalization
        using the options dict.
            cal_points (tuple) of indices of the calibrati  on points

            zero_coord, one_coord
        """

        cal_points = self.options_dict.get('cal_points', None)
        zero_coord = self.options_dict.get('zero_coord', None)
        one_coord = self.options_dict.get('one_coord', None)

        if cal_points is None:
            # default for all standard Timedomain experiments
            cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if len(self.raw_data_dict['measured_values']) == 1:
            # if only one weight function is used rotation is not required
            self.proc_data_dict['corr_data'] = a_tools.rotate_and_normalize_data_1ch(
                self.raw_data_dict['measured_values'][0],
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1])
        else:
            self.proc_data_dict['corr_data'], zero_coord, one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.raw_data_dict['measured_values'][0:2],
                    zero_coord=zero_coord,
                    one_coord=one_coord,
                    cal_zero_points=cal_points[0],
                    cal_one_points=cal_points[1])

        # This should be added to the hdf5 datafile but cannot because of the
        # way that the "new" analysis works.

        # self.add_dataset_to_analysisgroup('Corrected data',
        #                                   self.proc_data_dict['corr_data'])


class MultiQubit_TimeDomain_Analysis(ba.BaseDataAnalysis):

    def __init__(self,
                 qb_names: list=None, label: str='',
                 t_start: str=None, t_stop: str=None, data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True,
                 params_dict=None, numeric_params=None, **kwargs):

        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting, **kwargs)

        self.qb_names = qb_names
        if self.qb_names is None:
            raise ValueError('Provide the "qb_names."')

        self.params_dict = params_dict
        if self.params_dict is None:
            self.params_dict = {}
        self.numeric_params = numeric_params
        if self.numeric_params is None:
            self.numeric_params = []

        if not hasattr(self, "job"):
            self.create_job(qb_names=qb_names, t_start=t_start, t_stop=t_stop,
                            label=label, data_file_path=data_file_path,
                            do_fitting=do_fitting, options_dict=options_dict,
                            extract_only=extract_only, params_dict=params_dict,
                            numeric_params=numeric_params, **kwargs)
        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()

        self.channel_map = self.get_param_value('meas_obj_value_names_map')
        if self.channel_map is None:
            # if the new name meas_obj_value_names_map is not found, try with
            # the old name channel_map
            self.channel_map = self.get_param_value('channel_map')
            if self.channel_map is None:
                value_names = self.raw_data_dict['value_names']
                if np.ndim(value_names) > 0:
                    value_names = value_names
                if 'w' in value_names[0]:
                    self.channel_map = a_tools.get_qb_channel_map_from_hdf(
                        self.qb_names, value_names=value_names,
                        file_path=self.raw_data_dict['folder'])
                else:
                    self.channel_map = {}
                    for qbn in self.qb_names:
                        self.channel_map[qbn] = value_names

        if len(self.channel_map) == 0:
            raise ValueError('No qubit RO channels have been found.')

        # creates self.sp
        self.get_sweep_points()

    def get_sweep_points(self):
        self.sp = self.get_param_value('sweep_points')
        if self.sp is not None:
            self.sp = SweepPoints(from_dict_list=self.sp)

    def create_sweep_points_dict(self):
        sweep_points_dict = self.get_param_value('sweep_points_dict')
        hard_sweep_params = self.get_param_value('hard_sweep_params')
        if sweep_points_dict is not None:
            # assumed to be of the form {qbn1: swpts_array1, qbn2: swpts_array2}
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': sweep_points_dict[qbn]}
                 for qbn in self.qb_names}
        elif hard_sweep_params is not None:
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': list(hard_sweep_params.values())[0][
                    'values']} for qbn in self.qb_names}
        elif self.sp is not None:
            self.mospm = self.get_param_value('meas_obj_sweep_points_map')
            if self.mospm is None:
                raise ValueError('Please provide "meas_obj_sweep_points_map."')
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': self.sp.get_sweep_params_property(
                    'values', 0, self.mospm[qbn])[0]}
                 for qbn in self.qb_names}
        else:
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': self.data_filter(
                    self.raw_data_dict['hard_sweep_points'])}
                    for qbn in self.qb_names}

    def create_sweep_points_2D_dict(self):
        soft_sweep_params = self.get_param_value('soft_sweep_params')
        if self.sp is not None:
            self.proc_data_dict['sweep_points_2D_dict'] = OrderedDict()
            for qbn in self.qb_names:
                self.proc_data_dict['sweep_points_2D_dict'][qbn] = \
                    OrderedDict()
                for pn in self.mospm[qbn]:
                    if pn in self.sp[1]:
                        self.proc_data_dict['sweep_points_2D_dict'][qbn][
                            pn] = self.sp[1][pn][0]
        elif soft_sweep_params is not None:
            self.proc_data_dict['sweep_points_2D_dict'] = \
                {qbn: {pn: soft_sweep_params[pn]['values'] for
                       pn in soft_sweep_params}
                 for qbn in self.qb_names}
        else:
            if len(self.raw_data_dict['soft_sweep_points'].shape) == 1:
                self.proc_data_dict['sweep_points_2D_dict'] = \
                    {qbn: {self.raw_data_dict['sweep_parameter_names'][1]:
                               self.raw_data_dict['soft_sweep_points']} for
                     qbn in self.qb_names}
            else:
                sspn = self.raw_data_dict['sweep_parameter_names'][1:]
                self.proc_data_dict['sweep_points_2D_dict'] = \
                    {qbn: {sspn[i]: self.raw_data_dict['soft_sweep_points'][i]
                           for i in range(len(sspn))} for qbn in self.qb_names}
        if self.get_param_value('percentage_done', 100) < 100:
            # This indicated an interrupted measurement.
            # Remove non-measured sweep points in that case.
            # raw_data_dict['soft_sweep_points'] is obtained in
            # BaseDataAnalysis.add_measured_data(), and its length should
            # always correspond to the actual number of measured soft sweep
            # points.
            ssl = len(self.raw_data_dict['soft_sweep_points'])
            for sps in self.proc_data_dict['sweep_points_2D_dict'].values():
                for k, v in sps.items():
                    sps[k] = v[:ssl]

    def create_meas_results_per_qb(self):
        measured_RO_channels = list(self.raw_data_dict['measured_data'])
        meas_results_per_qb_raw = {}
        meas_results_per_qb = {}
        for qb_name, RO_channels in self.channel_map.items():
            meas_results_per_qb_raw[qb_name] = {}
            meas_results_per_qb[qb_name] = {}
            if isinstance(RO_channels, str):
                meas_ROs_per_qb = [RO_ch for RO_ch in measured_RO_channels
                                   if RO_channels in RO_ch]
                for meas_RO in meas_ROs_per_qb:
                    meas_results_per_qb_raw[qb_name][meas_RO] = \
                        self.raw_data_dict[
                            'measured_data'][meas_RO]
                    meas_results_per_qb[qb_name][meas_RO] = \
                        self.data_filter(
                            meas_results_per_qb_raw[qb_name][meas_RO])

            elif isinstance(RO_channels, list):
                for qb_RO_ch in RO_channels:
                    meas_ROs_per_qb = [RO_ch for RO_ch in measured_RO_channels
                                       if qb_RO_ch in RO_ch]

                    for meas_RO in meas_ROs_per_qb:
                        meas_results_per_qb_raw[qb_name][meas_RO] = \
                            self.raw_data_dict[
                                'measured_data'][meas_RO]
                        meas_results_per_qb[qb_name][meas_RO] = \
                            self.data_filter(
                                meas_results_per_qb_raw[qb_name][meas_RO])
            else:
                raise TypeError('The RO channels for {} must either be a list '
                                'or a string.'.format(qb_name))
        self.proc_data_dict['meas_results_per_qb_raw'] = \
            meas_results_per_qb_raw
        self.proc_data_dict['meas_results_per_qb'] = \
            meas_results_per_qb

    def process_data(self):
        super().process_data()
        self.data_filter = self.get_param_value('data_filter')
        prep_params = self.get_param_value('preparation_params',
                                           default_value=dict())
        self.data_with_reset = False
        if self.data_filter is None:
            if 'active' in prep_params.get('preparation_type', 'wait'):
                reset_reps = prep_params.get('reset_reps', 1)
                self.data_filter = lambda x: x[reset_reps::reset_reps+1]
                self.data_with_reset = True
            elif "preselection" in prep_params.get('preparation_type', 'wait'):
                self.data_filter = lambda x: x[1::2]  # filter preselection RO
        if self.data_filter is None:
            self.data_filter = lambda x: x

        self.create_sweep_points_dict()
        self.create_meas_results_per_qb()

        # temporary fix for appending calibration points to x values but
        # without breaking sequences not yet using this interface.
        rotate = self.get_param_value('rotate', default_value=False)
        cal_points = self.get_param_value('cal_points')
        last_ge_pulses = self.get_param_value('last_ge_pulses',
                                              default_value=False)
        try:
            self.cp = CalibrationPoints.from_string(cal_points)
            # for now assuming the same for all qubits.
            self.cal_states_dict = self.cp.get_indices(
                self.qb_names, prep_params)[self.qb_names[0]]
            if rotate:
                cal_states_rots = self.cp.get_rotations(last_ge_pulses,
                        self.qb_names[0])[self.qb_names[0]] if rotate else None
                self.cal_states_rotations = self.get_param_value(
                    'cal_states_rotations', default_value=cal_states_rots)
            else:
                self.cal_states_rotations = None
            sweep_points_w_calpts = \
                {qbn: {'sweep_points': self.cp.extend_sweep_points(
                    self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'], qbn)} for qbn in self.qb_names}
            self.proc_data_dict['sweep_points_dict'] = sweep_points_w_calpts
        except TypeError as e:
            log.error(e)
            log.warning("Failed retrieving cal point objects or states. "
                        "Please update measurement to provide cal point object "
                        "in metadata. Trying to get them using the old way ...")
            if rotate:
                self.cal_states_rotations = self.get_param_value(
                    'cal_states_rotations', default_value=None)
            else:
                self.cal_states_rotations = None
            self.cal_states_dict = self.get_param_value('cal_states_dict',
                                                         default_value={})

        # create projected_data_dict
        self.data_to_fit = self.get_param_value('data_to_fit', {})
        global_PCA = self.get_param_value('global_PCA', default_value=False)
        if self.cal_states_rotations is not None or global_PCA:
            self.cal_states_analysis()
        else:
            # this assumes data obtained with classifier detector!
            # ie pg, pe, pf are expected to be in the value_names
            self.proc_data_dict['projected_data_dict'] = OrderedDict()
            for qbn, data_dict in self.proc_data_dict[
                    'meas_results_per_qb'].items():
                self.proc_data_dict['projected_data_dict'][qbn] = OrderedDict()
                for state_prob in ['pg', 'pe', 'pf']:
                    self.proc_data_dict['projected_data_dict'][qbn].update(
                        {state_prob: data for key, data in data_dict.items()
                         if state_prob in key})
            if self.cal_states_dict is None:
                self.cal_states_dict = {}
            self.num_cal_points = np.array(list(
                self.cal_states_dict.values())).flatten().size

            # correct probabilities given calibration matrix
            if self.get_param_value("correction_matrix") is not None:
                self.proc_data_dict['projected_data_dict_corrected'] = OrderedDict()
                for qbn, data_dict in self.proc_data_dict[
                    'meas_results_per_qb'].items():
                    self.proc_data_dict['projected_data_dict'][qbn] = OrderedDict()
                    probas_raw = np.asarray([data_dict[k] for k in data_dict
                                             for state_prob in ['pg', 'pe', 'pf'] if
                                             state_prob in k])
                    corr_mtx = self.get_param_value("correction_matrix")[qbn]
                    probas_corrected = np.linalg.inv(corr_mtx).T @ probas_raw
                    for state_prob in ['pg', 'pe', 'pf']:
                        self.proc_data_dict['projected_data_dict_corrected'][qbn].update(
                            {state_prob: data for key, data in
                             zip(["pg", "pe", "pf"], probas_corrected)})

        # get data_to_fit
        self.proc_data_dict['data_to_fit'] = OrderedDict()
        for qbn, prob_data in self.proc_data_dict[
                'projected_data_dict'].items():
            if qbn in self.data_to_fit:
                self.proc_data_dict['data_to_fit'][qbn] = prob_data[
                    self.data_to_fit[qbn]]

        # create msmt_sweep_points, sweep_points, cal_points_sweep_points
        for qbn in self.qb_names:
            if self.num_cal_points > 0:
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'msmt_sweep_points'] = \
                    self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points'][:-self.num_cal_points]
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'cal_points_sweep_points'] = \
                    self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-self.num_cal_points::]
            else:
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'msmt_sweep_points'] = self.proc_data_dict[
                    'sweep_points_dict'][qbn]['sweep_points']
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'cal_points_sweep_points'] = []
        if self.options_dict.get('TwoD', False):
            self.create_sweep_points_2D_dict()

    def get_cal_data_points(self):
        if len(self.cal_states_rotations) == 0:
            # "To use PCA (no cal states rotation), the measurement function
            # has to specify the following in the metadata:
            # - CalibrationPoints with qubit names but empty states
            # - cal_states_rotations = {}
            # - rotate = True
            # - data_to_fit defined ({qbn: 'prob_to_fit'})
            self.cal_states_dict = {qbn: [] for qbn in self.qb_names}
            self.cal_states_dict_for_rotation = self.cal_states_dict
        else:
            if self.cal_states_dict is None:
                log.info('Assuming two cal states, |g> and |e>, and using '
                      'sweep_points[-4:-2] as |g> cal points, and '
                      'sweep_points[-2::] as |e> cal points.')
                self.cal_states_dict = OrderedDict()
                indices = list(range(-len(self.cal_states_rotations)*2, 0))
                for state, rot_idx in self.cal_states_rotations.items():
                    self.cal_states_dict[self.get_latex_prob_label(state)] = \
                        indices[2*rot_idx: 2*rot_idx+2]
                self.cal_states_dict_for_rotation = OrderedDict()
                for qbn in self.qb_names:
                    self.cal_states_dict_for_rotation[qbn] = self.cal_states_dict
            else:
                self.cal_states_dict_for_rotation = OrderedDict()
                states = False
                cal_states_rotations = self.cal_states_rotations
                for key in cal_states_rotations.keys():
                    if key == 'g' or key == 'e' or key == 'f':
                        states = True
                for qbn in self.qb_names:
                    self.cal_states_dict_for_rotation[qbn] = OrderedDict()
                    if states:
                        cal_states_rot_qb = cal_states_rotations
                    else:
                        cal_states_rot_qb = cal_states_rotations[qbn]
                    for i in range(len(cal_states_rot_qb)):
                        cal_state = \
                            [k for k, idx in cal_states_rot_qb.items()
                             if idx == i][0]
                        self.cal_states_dict_for_rotation[qbn][cal_state] = \
                            self.cal_states_dict[cal_state]

        self.num_cal_points = np.array(list(
            self.cal_states_dict.values())).flatten().size

    def cal_states_analysis(self):
        self.get_cal_data_points()
        self.proc_data_dict['projected_data_dict'] = OrderedDict(
            {qbn: '' for qbn in self.qb_names})
        for qbn in self.qb_names:
            cal_states_dict = self.cal_states_dict_for_rotation[qbn]
            if len(cal_states_dict) not in [0, 2, 3]:
                raise NotImplementedError('Calibration states rotation is '
                                          'currently only implemented for 0, '
                                          '2, or 3 cal states per qubit.')

            if self.get_param_value('TwoD', default_value=False):
                if len(cal_states_dict) == 3:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_3_cal_states_TwoD(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map,
                            self.cal_states_dict_for_rotation))
                else:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_TwoD(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.cal_states_dict_for_rotation,
                            self.data_to_fit,
                            global_PCA=self.get_param_value(
                                'global_PCA', default_value=False),
                            data_mostly_g=self.get_param_value(
                                'data_mostly_g', default_value=True)))
            else:
                if len(cal_states_dict) == 3:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_3_cal_states(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map,
                            self.cal_states_dict_for_rotation))
                else:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.cal_states_dict_for_rotation,
                            self.data_to_fit))

    @staticmethod
    def rotate_data_3_cal_states(qb_name, meas_results_per_qb, channel_map,
                                 cal_states_dict):
        # FOR 3 CAL STATES
        rotated_data_dict = OrderedDict()
        meas_res_dict  =meas_results_per_qb[qb_name]
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_states_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            raw_data = np.array([v for v in meas_res_dict.values()]).T
            for i, cal_idx in enumerate(cal_pts_idxs):
                cal_points_data[i, :] = np.mean(raw_data[cal_idx, :],
                                                axis=0)
            rotated_data = predict_proba_avg_ro(raw_data, cal_points_data)
            for i, state in enumerate(list(cal_states_dict[qb_name])):
                rotated_data_dict[qb_name][f'p{state}'] = rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        return rotated_data_dict

    @staticmethod
    def rotate_data(qb_name, meas_results_per_qb, channel_map,
                    cal_states_dict, data_to_fit):
        # ONLY WORKS FOR 2 CAL STATES
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        if len(cal_states_dict[qb_name]) == 0:
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_zero_points = list(cal_states_dict[qb_name].values())[0]
            cal_one_points = list(cal_states_dict[qb_name].values())[1]
        rotated_data_dict[qb_name] = OrderedDict()
        if len(meas_res_dict) == 1:
            # one RO channel per qubit
            if cal_zero_points is None and cal_one_points is None:
                data = meas_res_dict[list(meas_res_dict)[0]]
                rotated_data_dict[qb_name][data_to_fit[qb_name]] = \
                    (data - np.min(data))/(np.max(data) - np.min(data))
            else:
                rotated_data_dict[qb_name][data_to_fit[qb_name]] = \
                    a_tools.rotate_and_normalize_data_1ch(
                        data=meas_res_dict[list(meas_res_dict)[0]],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            rotated_data_dict[qb_name][data_to_fit[qb_name]], _, _ = \
                a_tools.rotate_and_normalize_data_IQ(
                    data=np.array([v for v in meas_res_dict.values()]),
                    cal_zero_points=cal_zero_points,
                    cal_one_points=cal_one_points)
        else:
            # multiple readouts per qubit per channel
            if isinstance(channel_map[qb_name], str):
                qb_ro_ch0 = channel_map[qb_name]
            else:
                qb_ro_ch0 = channel_map[qb_name][0]
            ro_suffixes = [s[len(qb_ro_ch0)+1::] for s in
                           list(meas_res_dict) if qb_ro_ch0 in s]
            for i, ro_suf in enumerate(ro_suffixes):
                if len(ro_suffixes) == len(meas_res_dict):
                    # one RO ch per qubit
                    rotated_data_dict[qb_name][ro_suf] = \
                        a_tools.rotate_and_normalize_data_1ch(
                            data=meas_res_dict[list(meas_res_dict)[i]],
                            cal_zero_points=cal_zero_points,
                            cal_one_points=cal_one_points)
                else:
                    # two RO ch per qubit
                    keys = [k for k in meas_res_dict if ro_suf in k]
                    correct_keys = [k for k in keys
                                    if k[len(qb_ro_ch0)+1::] == ro_suf]
                    data_array = np.array([meas_res_dict[k]
                                           for k in correct_keys])
                    rotated_data_dict[qb_name][ro_suf], \
                    _, _ = \
                        a_tools.rotate_and_normalize_data_IQ(
                            data=data_array,
                            cal_zero_points=cal_zero_points,
                            cal_one_points=cal_one_points)
        return rotated_data_dict

    @staticmethod
    def rotate_data_3_cal_states_TwoD(qb_name, meas_results_per_qb,
                                      channel_map, cal_states_dict):
        # FOR 3 CAL STATES
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_states_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            for i, state in enumerate(list(cal_states_dict[qb_name])):
                rotated_data_dict[qb_name][f'p{state}'] = np.zeros(
                    raw_data_arr.shape)
            for col in range(raw_data_arr.shape[1]):
                raw_data = np.concatenate([
                    v[:, col].reshape(len(v[:, col]), 1) for
                    v in meas_res_dict.values()], axis=1)
                for i, cal_idx in enumerate(cal_pts_idxs):
                    cal_points_data[i, :] = np.mean(raw_data[cal_idx, :],
                                                    axis=0)
                # rotated data is (raw_data_arr.shape[0], 3)
                rotated_data = predict_proba_avg_ro(
                    raw_data, cal_points_data)

                for i, state in enumerate(list(cal_states_dict[qb_name])):
                    rotated_data_dict[qb_name][f'p{state}'][:, col] = \
                        rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        return rotated_data_dict

    @staticmethod
    def rotate_data_TwoD(qb_name, meas_results_per_qb, channel_map,
                         cal_states_dict, data_to_fit,
                         global_PCA=False, data_mostly_g=None):
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        if len(cal_states_dict[qb_name]) == 0:
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_zero_points = list(cal_states_dict[qb_name].values())[0]
            cal_one_points = list(cal_states_dict[qb_name].values())[1]
        rotated_data_dict[qb_name] = OrderedDict()
        if len(meas_res_dict) == 1:
            if global_PCA:
                raise NotImplementedError('Global PCA is not implemented \
                                        for one channel RO!')
            # one RO channel per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][data_to_fit[qb_name]] = \
                deepcopy(raw_data_arr.transpose())
            for col in range(raw_data_arr.shape[1]):
                rotated_data_dict[qb_name][data_to_fit[qb_name]][col] = \
                    a_tools.rotate_and_normalize_data_1ch(
                        data=raw_data_arr[:, col],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][data_to_fit[qb_name]] = \
                deepcopy(raw_data_arr.transpose())
            if not global_PCA:
                for col in range(raw_data_arr.shape[1]):
                    data_array = np.array(
                        [v[:, col] for v in meas_res_dict.values()])
                    rotated_data_dict[qb_name][
                            data_to_fit[qb_name]][col], _, _ = \
                        a_tools.rotate_and_normalize_data_IQ(
                            data=data_array,
                            cal_zero_points=cal_zero_points,
                            cal_one_points=cal_one_points)
            else:
                data_array = np.array(
                    [v.flatten() for v in meas_res_dict.values()])
                rotated_data_dict[qb_name][
                    data_to_fit[qb_name]], _, _ = \
                    a_tools.rotate_and_normalize_data_IQ(
                        data=data_array,
                        cal_zero_points=None,  # if cal_points are None, rotation via PCA
                        cal_one_points=None,
                        data_mostly_g=data_mostly_g  # True if most points are expected to be ground state
                    )
        else:
            if global_PCA:
                raise NotImplementedError('Global PCA is not implemented \
                                        for multiple RPs per qubit channel!')
            # multiple readouts per qubit per channel
            if isinstance(channel_map[qb_name], str):
                qb_ro_ch0 = channel_map[qb_name]
            else:
                qb_ro_ch0 = channel_map[qb_name][0]

            ro_suffixes = [s[len(qb_ro_ch0)+1::] for s in
                           list(meas_res_dict) if qb_ro_ch0 in s]

            for i, ro_suf in enumerate(ro_suffixes):
                if len(ro_suffixes) == len(meas_res_dict):
                    # one RO ch per qubit
                    raw_data_arr = meas_res_dict[list(meas_res_dict)[i]]
                    rotated_data_dict[qb_name][ro_suf] = \
                        deepcopy(raw_data_arr.transpose())
                    for col in range(raw_data_arr.shape[1]):
                        rotated_data_dict[qb_name][
                            ro_suf][col] = \
                            a_tools.rotate_and_normalize_data_1ch(
                                data=raw_data_arr[:, col],
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
                else:
                    # two RO ch per qubit
                    raw_data_arr = meas_res_dict[list(meas_res_dict)[i]]
                    rotated_data_dict[qb_name][ro_suf] = \
                        deepcopy(raw_data_arr.transpose())
                    for col in range(raw_data_arr.shape[1]):
                        data_array = np.array(
                            [v[:, col] for k, v in meas_res_dict.items()
                             if ro_suf in k])
                        rotated_data_dict[qb_name][ro_suf][col], _, _ = \
                            a_tools.rotate_and_normalize_data_IQ(
                                data=data_array,
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
        return rotated_data_dict

    def get_xaxis_label_unit(self, qb_name):
        hard_sweep_params = self.get_param_value('hard_sweep_params')
        sweep_name = self.get_param_value('sweep_name')
        sweep_unit = self.get_param_value('sweep_unit')
        if self.sp is not None:
            _, xunit, xlabel = self.sp.get_sweep_params_description(
                param_names=self.mospm[qb_name], dimension=0)[0]
        elif hard_sweep_params is not None:
            xlabel = list(hard_sweep_params)[0]
            xunit = list(hard_sweep_params.values())[0][
                'unit']
        elif (sweep_name is not None) and (sweep_unit is not None):
            xlabel = sweep_name
            xunit = sweep_unit
        else:
            xlabel = self.raw_data_dict['sweep_parameter_names']
            xunit = self.raw_data_dict['sweep_parameter_units']
        if np.ndim(xlabel) > 0:
            xlabel = xlabel[0]
        if np.ndim(xunit) > 0:
            xunit = xunit[0]
        return xlabel, xunit

    @staticmethod
    def get_cal_state_color(cal_state_label):
        if cal_state_label == 'g' or cal_state_label == r'$|g\rangle$':
            return 'k'
        elif cal_state_label == 'e' or cal_state_label == r'$|e\rangle$':
            return 'gray'
        elif cal_state_label == 'f' or cal_state_label == r'$|f\rangle$':
            return 'C8'
        else:
            return 'C4'

    @staticmethod
    def get_latex_prob_label(prob_label):
        if '$' in prob_label:
            return prob_label
        elif 'p' in prob_label.lower():
            return r'$|{}\rangle$'.format(prob_label[-1])
        else:
            return r'$|{}\rangle$'.format(prob_label)

    def prepare_plots(self):
        if self.get_param_value('plot_proj_data', default_value=True):
            for qb_name, corr_data in self.proc_data_dict[
                    'projected_data_dict'].items():
                fig_name = 'projected_plot_' + qb_name
                if isinstance(corr_data, dict):
                    for data_key, data in corr_data.items():
                        if self.cal_states_rotations is None:
                            data_label = data_key
                            title_suffix = ''
                            plot_name_suffix = data_key
                            plot_cal_points = False
                            data_axis_label = 'Population'
                        else:
                            fig_name = 'projected_plot_' + qb_name + \
                                       data_key
                            data_label = 'Data'
                            title_suffix = data_key
                            plot_name_suffix = ''
                            plot_cal_points = (
                                not self.options_dict.get('TwoD', False))
                            data_axis_label = '{} state population'.format(
                                self.get_latex_prob_label(data_key))
                        self.prepare_projected_data_plot(
                            fig_name, data, qb_name=qb_name,
                            data_label=data_label,
                            title_suffix=title_suffix,
                            plot_name_suffix=plot_name_suffix,
                            data_axis_label=data_axis_label,
                            plot_cal_points=plot_cal_points)

                else:
                    fig_name = 'projected_plot_' + qb_name
                    self.prepare_projected_data_plot(
                        fig_name, corr_data, qb_name=qb_name,
                        plot_cal_points=(
                            not self.options_dict.get('TwoD', False)))

        if self.get_param_value('plot_raw_data', default_value=True):
            # self.cal_states_rotations is not None:
            self.prepare_raw_data_plots(plot_filtered=False)
            if 'preparation_params' in self.metadata:
                if 'active' in self.metadata['preparation_params'].get(
                        'preparation_type', 'wait'):
                    self.prepare_raw_data_plots(plot_filtered=True)

    def prepare_raw_data_plots(self, plot_filtered=False):
        if plot_filtered or not self.data_with_reset:
            key = 'meas_results_per_qb'
            suffix = 'filtered' if self.data_with_reset else ''
            func_for_swpts = lambda qb_name: self.proc_data_dict[
                'sweep_points_dict'][qb_name]['sweep_points']
        else:
            key = 'meas_results_per_qb_raw'
            suffix = ''
            func_for_swpts = lambda qb_name: self.raw_data_dict[
                'hard_sweep_points']
        for qb_name, raw_data_dict in self.proc_data_dict[key].items():
            sweep_points = func_for_swpts(qb_name)
            if len(raw_data_dict) == 1:
                numplotsx = 1
                numplotsy = 1
            elif len(raw_data_dict) == 2:
                numplotsx = 1
                numplotsy = 2
            else:
                numplotsx = 2
                numplotsy = len(raw_data_dict) // 2 + len(raw_data_dict) % 2

            plotsize = self.get_default_plot_params(set=False)['figure.figsize']
            fig_title = (self.raw_data_dict['timestamp'] + ' ' +
                         self.raw_data_dict['measurementstring'] +
                         '\nRaw data ' + suffix + ' ' + qb_name)
            plot_name = 'raw_plot_' + qb_name + suffix
            xlabel, xunit = self.get_xaxis_label_unit(qb_name)

            for ax_id, ro_channel in enumerate(raw_data_dict):
                if self.get_param_value('TwoD', default_value=False):
                    if self.sp is None:
                        soft_sweep_params = self.get_param_value(
                            'soft_sweep_params')
                        if soft_sweep_params is not None:
                            yunit = list(soft_sweep_params.values())[0]['unit']
                        else:
                            yunit = self.raw_data_dict[
                                'sweep_parameter_units'][1]
                        if np.ndim(yunit) > 0:
                            yunit = yunit[0]
                    for pn, ssp in self.proc_data_dict['sweep_points_2D_dict'][
                            qb_name].items():
                        ylabel = pn
                        if self.sp is not None:
                            yunit = self.sp.get_sweep_params_property(
                                'unit', dimension=1, param_names=pn)
                            ylabel = self.sp.get_sweep_params_property(
                                'label', dimension=2, param_names=pn)
                        self.plot_dicts[f'{plot_name}_{ro_channel}_{pn}'] = {
                            'fig_id': plot_name + '_' + pn,
                            'ax_id': ax_id,
                            'plotfn': self.plot_colorxy,
                            'xvals': sweep_points,
                            'yvals': ssp,
                            'zvals': raw_data_dict[ro_channel].T,
                            'xlabel': xlabel,
                            'xunit': xunit,
                            'ylabel': ylabel,
                            'yunit': yunit,
                            'numplotsx': numplotsx,
                            'numplotsy': numplotsy,
                            'plotsize': (plotsize[0]*numplotsx,
                                         plotsize[1]*numplotsy),
                            'title': fig_title,
                            'clabel': '{} (Vpeak)'.format(ro_channel)}
                else:
                    self.plot_dicts[plot_name + '_' + ro_channel] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': sweep_points,
                        'xlabel': xlabel,
                        'xunit': xunit,
                        'yvals': raw_data_dict[ro_channel],
                        'ylabel': '{} (Vpeak)'.format(ro_channel),
                        'yunit': '',
                        'numplotsx': numplotsx,
                        'numplotsy': numplotsy,
                        'plotsize': (plotsize[0]*numplotsx,
                                     plotsize[1]*numplotsy),
                        'title': fig_title}
            if len(raw_data_dict) == 1:
                self.plot_dicts[
                    plot_name + '_' + list(raw_data_dict)[0]]['ax_id'] = None

    def prepare_projected_data_plot(
            self, fig_name, data, qb_name, title_suffix='', sweep_points=None,
            plot_cal_points=True, plot_name_suffix='', data_label='Data',
            data_axis_label='', do_legend_data=True, do_legend_cal_states=True):
        title_suffix = qb_name + title_suffix
        if data_axis_label == '':
            data_axis_label = '{} state population'.format(
                self.get_latex_prob_label(self.data_to_fit[qb_name]))
        plotsize = self.get_default_plot_params(set=False)['figure.figsize']
        plotsize = (plotsize[0], plotsize[0]/1.25)

        if sweep_points is None:
            sweep_points = self.proc_data_dict['sweep_points_dict'][qb_name][
                'sweep_points']
        plot_names_cal = []
        if plot_cal_points and self.num_cal_points != 0:
            yvals = data[:-self.num_cal_points]
            xvals = sweep_points[:-self.num_cal_points]
            # plot cal points
            for i, cal_pts_idxs in enumerate(
                    self.cal_states_dict.values()):
                plot_dict_name_cal = fig_name + '_' + \
                                     list(self.cal_states_dict)[i] + '_' + \
                                     plot_name_suffix
                plot_names_cal += [plot_dict_name_cal]
                self.plot_dicts[plot_dict_name_cal] = {
                    'fig_id': fig_name,
                    'plotfn': self.plot_line,
                    'plotsize': plotsize,
                    'xvals': self.proc_data_dict['sweep_points_dict'][qb_name][
                        'cal_points_sweep_points'][cal_pts_idxs],
                    'yvals': data[cal_pts_idxs],
                    'setlabel': list(self.cal_states_dict)[i],
                    'do_legend': do_legend_cal_states,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'linestyle': 'none',
                    'line_kws': {'color': self.get_cal_state_color(
                        list(self.cal_states_dict)[i])}}

                self.plot_dicts[plot_dict_name_cal+'_line'] = {
                    'fig_id': fig_name,
                    'plotsize': plotsize,
                    'plotfn': self.plot_hlines,
                    'y': np.mean(data[cal_pts_idxs]),
                    'xmin': self.proc_data_dict['sweep_points_dict'][qb_name][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qb_name][
                        'sweep_points'][-1],
                    'colors': 'gray'}

        else:
            yvals = data
            xvals = sweep_points
        title = (self.raw_data_dict['timestamp'] + ' ' +
                 self.raw_data_dict['measurementstring'])
        if title_suffix is not None:
            title += '\n' + title_suffix

        plot_dict_name = fig_name + '_' + plot_name_suffix
        xlabel, xunit = self.get_xaxis_label_unit(qb_name)

        if self.get_param_value('TwoD', default_value=False):
            if self.sp is None:
                soft_sweep_params = self.get_param_value(
                    'soft_sweep_params')
                if soft_sweep_params is not None:
                    yunit = list(soft_sweep_params.values())[0]['unit']
                else:
                    yunit = self.raw_data_dict['sweep_parameter_units'][1]
                if np.ndim(yunit) > 0:
                    yunit = yunit[0]
            for pn, ssp in self.proc_data_dict['sweep_points_2D_dict'][
                    qb_name].items():
                ylabel = pn
                if self.sp is not None:
                    yunit = self.sp.get_sweep_params_property(
                        'unit', dimension=1, param_names=pn)
                    ylabel = self.sp.get_sweep_params_property(
                        'label', dimension=2, param_names=pn)
                self.plot_dicts[f'{plot_dict_name}_{pn}'] = {
                    'plotfn': self.plot_colorxy,
                    'fig_id': fig_name + '_' + pn,
                    'xvals': xvals,
                    'yvals': ssp,
                    'zvals': yvals,
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'ylabel': ylabel,
                    'yunit': yunit,
                    'title': title,
                    'clabel': data_axis_label}
        else:
            self.plot_dicts[plot_dict_name] = {
                'plotfn': self.plot_line,
                'fig_id': fig_name,
                'plotsize': plotsize,
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,
                'yvals': yvals,
                'ylabel': data_axis_label,
                'yunit': '',
                'setlabel': data_label,
                'title': title,
                'linestyle': 'none',
                'do_legend': do_legend_data,
                'legend_bbox_to_anchor': (1, 0.5),
                'legend_pos': 'center left'}
        if len(plot_names_cal) > 0:
            if do_legend_data and not do_legend_cal_states:
                for plot_name in plot_names_cal:
                    plot_dict_cal = self.plot_dicts.pop(plot_name)
                    self.plot_dicts[plot_name] = plot_dict_cal


class Idling_Error_Rate_Analyisis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        post_sel_th = self.options_dict.get('post_sel_th', 0.5)
        raw_shots = self.raw_data_dict['measured_values'][0][0]
        post_sel_shots = raw_shots[::2]
        data_shots = raw_shots[1::2]
        data_shots[np.where(post_sel_shots > post_sel_th)] = np.nan

        states = ['0', '1', '+']
        self.proc_data_dict['xvals'] = np.unique(self.raw_data_dict['xvals'])
        for i, state in enumerate(states):
            self.proc_data_dict['shots_{}'.format(state)] =data_shots[i::3]

            self.proc_data_dict['yvals_{}'.format(state)] = \
                np.nanmean(np.reshape(self.proc_data_dict['shots_{}'.format(state)],
                               (len(self.proc_data_dict['xvals']), -1),
                               order='F'), axis=1)


    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            self.plot_dicts['Prepare in {}'.format(state)] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': yvals,
                'ylabel': 'Counts',
                'yrange': [0, 1],
                'xrange': self.options_dict.get('xrange', None),
                'yunit': 'frac',
                'setlabel': 'Prepare in {}'.format(state),
                'do_legend':True,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'legend_pos': 'upper right'}
        if self.do_fitting:
            for state in ['0', '1', '+']:
                self.plot_dicts['fit_{}'.format(state)] = {
                    'ax_id': 'main',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['fit {}'.format(state)]['fit_res'],
                    'plot_init': self.options_dict['plot_init'],
                    'setlabel': 'fit |{}>'.format(state),
                    'do_legend': True,
                    'legend_pos': 'upper right'}

                self.plot_dicts['fit_text']={
                    'ax_id':'main',
                    'box_props': 'fancy',
                    'xpos':1.05,
                    'horizontalalignment':'left',
                    'plotfn': self.plot_text,
                    'text_string': self.proc_data_dict['fit_msg']}



    def analyze_fit_results(self):
        fit_msg =''
        states = ['0', '1', '+']
        for state in states:
            fr = self.fit_res['fit {}'.format(state)]
            N1 = fr.params['N1'].value, fr.params['N1'].stderr
            N2 = fr.params['N2'].value, fr.params['N2'].stderr
            fit_msg += ('Prep |{}> : \n\tN_1 = {:.2g} $\pm$ {:.2g}'
                    '\n\tN_2 = {:.2g} $\pm$ {:.2g}\n').format(
                state, N1[0], N1[1], N2[0], N2[1])

        self.proc_data_dict['fit_msg'] = fit_msg

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            mod = lmfit.Model(fit_mods.idle_error_rate_exp_decay)
            mod.guess = fit_mods.idle_err_rate_guess.__get__(mod, mod.__class__)

            # Done here explicitly so that I can overwrite a specific guess
            guess_pars = mod.guess(N=xvals, data=yvals)
            vary_N2 = self.options_dict.get('vary_N2', True)

            if not vary_N2:
                guess_pars['N2'].value = 1e21
                guess_pars['N2'].vary = False
            self.fit_dicts['fit {}'.format(states[i])] = {
                'model': mod,
                'fit_xvals': {'N': xvals},
                'fit_yvals': {'data': yvals},
                'guess_pars': guess_pars}
            # Allows fixing the double exponential coefficient


class Grovers_TwoQubitAllStates_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        normalize_to_cal_points = self.options_dict.get('normalize_to_cal_points', True)
        cal_points = [
                        [[-4, -3], [-2, -1]],
                        [[-4, -2], [-3, -1]],
                       ]
        for idx in [0,1]:
            yvals = list(self.raw_data_dict['measured_data'].values())[idx][0]

            self.proc_data_dict['ylabel_{}'.format(idx)] = \
                self.raw_data_dict['value_names'][0][idx]
            self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.rotate_and_normalize_data_1ch(yvals,
                    cal_zero_points=cal_points[idx][0],
                    cal_one_points=cal_points[idx][1])
            self.proc_data_dict['yvals_{}'.format(idx)] = yvals

        y0 = self.proc_data_dict['yvals_0']
        y1 = self.proc_data_dict['yvals_1']
        p_success = ((y0[0]*y1[0]) +
                     (1-y0[1])*y1[1] +
                     (y0[2])*(1-y1[2]) +
                     (1-y0[3])*(1-y1[3]) )/4
        self.proc_data_dict['p_success'] = p_success


    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i in [0, 1]:
            yvals = self.proc_data_dict['yvals_{}'.format(i)]
            xvals =  self.raw_data_dict['xvals'][0]
            ylabel = self.proc_data_dict['ylabel_{}'.format(i)]
            self.plot_dicts['main_{}'.format(ylabel)] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['xvals'][0],
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': self.proc_data_dict['yvals_{}'.format(i)],
                'ylabel': ylabel,
                'yunit': self.proc_data_dict['yunit'],
                'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': False,
                'legend_pos': 'upper right'}


        self.plot_dicts['limit_text']={
            'ax_id':'main_{}'.format(ylabel),
            'box_props': 'fancy',
            'xpos':1.05,
            'horizontalalignment':'left',
            'plotfn': self.plot_text,
            'text_string': 'P succes = {:.3f}'.format(self.proc_data_dict['p_success'])}








class FlippingAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # Even though we expect an exponentially damped oscillation we use
        # a simple cosine as this gives more reliable fitting and we are only
        # interested in extracting the frequency of the oscillation
        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.raw_data_dict['sweep_points'][:-4],
            data=self.proc_data_dict['corr_data'][:-4])

        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars['amplitude'].value = 0.5
        guess_pars['amplitude'].vary = False
        guess_pars['offset'].value = 0.5
        guess_pars['offset'].vary = False

        self.fit_dicts['cos_fit'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

        # In the case there are very few periods we fall back on a small
        # angle approximation to extract the drive detuning
        poly_mod = lmfit.models.PolynomialModel(degree=1)
        # the detuning can be estimated using on a small angle approximation
        # c1 = d/dN (cos(2*pi*f N) ) evaluated at N = 0 -> c1 = -2*pi*f
        poly_mod.set_param_hint('frequency', expr='-c1/(2*pi)')
        guess_pars = poly_mod.guess(x=self.raw_data_dict['sweep_points'][:-4],
                                    data=self.proc_data_dict['corr_data'][:-4])
        # Constraining the line ensures that it will only give a good fit
        # if the small angle approximation holds
        guess_pars['c0'].vary = False
        guess_pars['c0'].value = 0.5

        self.fit_dicts['line_fit'] = {
            'model': poly_mod,
            'fit_xvals': {'x': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        sf_line = self._get_scale_factor_line()
        sf_cos = self._get_scale_factor_cos()
        self.proc_data_dict['scale_factor'] = self.get_scale_factor()

        msg = 'Scale fact. based on '
        if self.proc_data_dict['scale_factor'] == sf_cos:
            msg += 'cos fit\n'
        else:
            msg += 'line fit\n'
        msg += 'cos fit: {:.4f}\n'.format(sf_cos)
        msg += 'line fit: {:.4f}'.format(sf_line)

        self.raw_data_dict['scale_factor_msg'] = msg
        # TODO: save scale factor to file

    def get_scale_factor(self):
        """
        Returns the scale factor that should correct for the error in the
        pulse amplitude.
        """
        # Model selection based on the Bayesian Information Criterion (BIC)
        # as  calculated by lmfit
        if (self.fit_dicts['line_fit']['fit_res'].bic <
                self.fit_dicts['cos_fit']['fit_res'].bic):
            scale_factor = self._get_scale_factor_line()
        else:
            scale_factor = self._get_scale_factor_cos()
        return scale_factor

    def _get_scale_factor_cos(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['cos_fit']['fit_res'].params['frequency']

        # the square is needed to account for the difference between
        # power and amplitude
        scale_factor = (1+frequency)**2

        phase = np.rad2deg(self.fit_dicts['cos_fit']['fit_res'].params['phase']) % 360
        # phase ~90 indicates an under rotation so the scale factor
        # has to be larger than 1. A phase ~270 indicates an over
        # rotation so then the scale factor has to be smaller than one.
        if phase > 180:
            scale_factor = 1/scale_factor

        return scale_factor

    def _get_scale_factor_line(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['line_fit']['fit_res'].params['frequency']
        scale_factor = (1+frequency)**2
        # no phase sign check is needed here as this is contained in the
        # sign of the coefficient

        return scale_factor

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'],  # does not do anything yet
            'yvals': self.proc_data_dict['corr_data'],
            'ylabel': 'Excited state population',
            'yunit': '',
            'setlabel': 'data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'line fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'cos fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['text_msg'] = {
                'ax_id': 'main',
                'ypos': 0.15,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'text_string': self.raw_data_dict['scale_factor_msg']}


class Intersect_Analysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analysis to extract the intercept of two parameters.

    relevant options_dict parameters
        ch_idx_A (int) specifies first channel for intercept
        ch_idx_B (int) specifies second channel for intercept if same as first
            it will assume data was taken interleaved.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xvals': 'sweep_points',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()


    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_A" and "ch_idx_B"
        specified in the options dict. If ch_idx_A and ch_idx_B are the same
        it will unzip the data.
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        ch_idx_A = self.options_dict.get('ch_idx_A', 0)
        ch_idx_B = self.options_dict.get('ch_idx_B', 0)


        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx_A]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx_A]

        if ch_idx_A == ch_idx_B:
            yvals = list(self.raw_data_dict['measured_data'].values())[ch_idx_A][0]
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0][::2]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0][1::2]
            self.proc_data_dict['yvals_A'] = yvals[::2]
            self.proc_data_dict['yvals_B'] = yvals[1::2]
        else:
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0]

            self.proc_data_dict['yvals_A'] = list(self.raw_data_dict
                ['measured_data'].values())[ch_idx_A][0]
            self.proc_data_dict['yvals_B'] = list(self.raw_data_dict
                ['measured_data'].values())[ch_idx_B][0]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_A'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_A']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_A']}}

        self.fit_dicts['line_fit_B'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_B']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_B']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_A'].best_values
        fr_1 = self.fit_res['line_fit_B'].best_values

        c0 = (fr_0['c0'] - fr_1['c0'])
        c1 = (fr_0['c1'] - fr_1['c1'])
        c2 = (fr_0['c2'] - fr_1['c2'])
        poly_coeff = [c0, c1, c2]
        poly = np.polynomial.polynomial.Polynomial([fr_0['c0'],
                                                   fr_0['c1'], fr_0['c2']])
        ic = np.polynomial.polynomial.polyroots(poly_coeff)

        self.proc_data_dict['intersect_L'] = ic[0], poly(ic[0])
        self.proc_data_dict['intersect_R'] = ic[1], poly(ic[1])

        if (((np.min(self.proc_data_dict['xvals']))< ic[0]) and
                ( ic[0] < (np.max(self.proc_data_dict['xvals'])))):
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_L']
        else:
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_R']

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_A'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_A'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'A',
            'title': (self.proc_data_dict['timestamps'][0] + ' \n' +
                      self.proc_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_B'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_B'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'B',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_A'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_A']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit A',
                'do_legend': True}
            self.plot_dicts['line_fit_B'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_B']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit B',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['intersect'][0],
                 self.proc_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['intersect'][0]],
                'yvals': [self.proc_data_dict['intersect'][1]],
                'line_kws': {'alpha': .5, 'color':'gray',
                            'markersize':15},
                'marker': 'o',
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_intersect(self):

        return self.proc_data_dict['intersect']



class CZ_1QPhaseCal_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract the intercept for a single qubit phase calibration
    experiment

    N.B. this is a less generic version of "Intersect_Analysis" and should
    be deprecated (MAR Dec 2017)
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx" in options dict and
        then splits the data for th
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx = self.options_dict['ch_idx']

        yvals = list(self.raw_data_dict['measured_data'].values())[ch_idx][0]

        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx]
        self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
        self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]
        self.proc_data_dict['yvals_off'] = yvals[::2]
        self.proc_data_dict['yvals_on'] = yvals[1::2]


    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_off'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_off']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_off']}}

        self.fit_dicts['line_fit_on'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_on']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_on']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_off'].best_values
        fr_1 = self.fit_res['line_fit_on'].best_values
        ic = -(fr_0['c0'] - fr_1['c0'])/(fr_0['c1'] - fr_1['c1'])

        self.proc_data_dict['zero_phase_diff_intersect'] = ic


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_off'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_on'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['line_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['zero_phase_diff_intersect'],
                 self.raw_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['zero_phase_diff_intersect']],
                'yvals': [np.mean(self.proc_data_dict['xvals_on'])],
                'line_kws': {'alpha': 0},
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_zero_phase_diff_intersect(self):

        return self.proc_data_dict['zero_phase_diff_intersect']


class Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Very basic analysis to determine the phase of a single oscillation
    that has an assumed period of 360 degrees.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        idx = 1

        self.proc_data_dict['yvals'] = list(self.raw_data_dict['measured_data'].values())[idx][0]
        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        self.fit_dicts['cos_fit'] = {
            'model': cos_mod,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.raw_data_dict['xvals'][0]},
            'fit_yvals': {'data': self.proc_data_dict['yvals']}}

    def analyze_fit_results(self):
        fr = self.fit_res['cos_fit'].best_values
        self.proc_data_dict['phi'] =  np.rad2deg(fr['phase'])


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['xvals'][0],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit',
                'do_legend': True}


class Conditional_Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract quantities from a conditional oscillation.

    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_osc" and
        "ch_idx_spec" in the options dict and then splits the data for the
        off and on cases
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx_spec = self.options_dict.get('ch_idx_spec', 0)
        ch_idx_osc = self.options_dict.get('ch_idx_osc', 1)
        normalize_to_cal_points = self.options_dict.get('normalize_to_cal_points', True)
        cal_points = [
                        [[-4, -3], [-2, -1]],
                        [[-4, -2], [-3, -1]],
                       ]


        i = 0
        for idx, type_str in zip([ch_idx_osc, ch_idx_spec], ['osc', 'spec']):
            yvals = list(self.raw_data_dict['measured_data'].values())[idx][0]
            self.proc_data_dict['ylabel_{}'.format(type_str)] = self.raw_data_dict['value_names'][0][idx]
            self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.rotate_and_normalize_data_1ch(yvals,
                    cal_zero_points=cal_points[i][0],
                    cal_one_points=cal_points[i][1])
                i +=1

                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]
                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]

            else:
                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]


                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]



    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod0 = lmfit.Model(fit_mods.CosFunc)
        cos_mod0.guess = fit_mods.Cos_guess.__get__(cos_mod0, cos_mod0.__class__)
        self.fit_dicts['cos_fit_off'] = {
            'model': cos_mod0,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.proc_data_dict['xvals_off'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_off'][:-2]}}

        cos_mod1 = lmfit.Model(fit_mods.CosFunc)
        cos_mod1.guess = fit_mods.Cos_guess.__get__(cos_mod1, cos_mod1.__class__)
        self.fit_dicts['cos_fit_on'] = {
            'model': cos_mod1,
            'guess_dict': {'frequency': {'value': 1/360, 'vary': False}},
            'fit_xvals': {'t': self.proc_data_dict['xvals_on'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_on'][:-2]}}

    def analyze_fit_results(self):
        fr_0 = self.fit_res['cos_fit_off'].params
        fr_1 = self.fit_res['cos_fit_on'].params

        phi0 = np.rad2deg(fr_0['phase'].value)
        phi1 = np.rad2deg(fr_1['phase'].value)

        phi0_stderr = np.rad2deg(fr_0['phase'].stderr)
        phi1_stderr = np.rad2deg(fr_1['phase'].stderr)

        self.proc_data_dict['phi_0'] = phi0, phi0_stderr
        self.proc_data_dict['phi_1'] = phi1, phi1_stderr
        phi_cond_stderr = (phi0_stderr**2+phi1_stderr**2)**.5
        self.proc_data_dict['phi_cond'] = (phi1 -phi0), phi_cond_stderr


        osc_amp = np.mean([fr_0['amplitude'], fr_1['amplitude']])
        osc_amp_stderr = np.sqrt(fr_0['amplitude'].stderr**2 +
                                 fr_1['amplitude']**2)/2

        self.proc_data_dict['osc_amp_0'] = (fr_0['amplitude'].value,
                                            fr_0['amplitude'].stderr)
        self.proc_data_dict['osc_amp_1'] = (fr_1['amplitude'].value,
                                            fr_1['amplitude'].stderr)

        self.proc_data_dict['osc_offs_0'] = (fr_0['offset'].value,
                                            fr_0['offset'].stderr)
        self.proc_data_dict['osc_offs_1'] = (fr_1['offset'].value,
                                            fr_1['offset'].stderr)


        offs_stderr = (fr_0['offset'].stderr**2+fr_1['offset'].stderr**2)**.5
        self.proc_data_dict['offs_diff'] = (
            fr_1['offset'].value - fr_0['offset'].value, offs_stderr)

        # self.proc_data_dict['osc_amp'] = (osc_amp, osc_amp_stderr)
        self.proc_data_dict['missing_fraction'] = (
            np.mean(self.proc_data_dict['yvals_spec_on'][:-2]) -
            np.mean(self.proc_data_dict['yvals_spec_off'][:-2]))


    def prepare_plots(self):
        self._prepare_main_oscillation_figure()
        self._prepare_spectator_qubit_figure()

    def _prepare_main_oscillation_figure(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_off'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_on'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['cos_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}

            # offset as a guide for the eye
            y = self.fit_res['cos_fit_off'].params['offset'].value
            self.plot_dicts['cos_off_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C0', 'linestyle': 'dotted'}
                    }

            phase_message = (
                'Phase diff.: {:.1f} $\pm$ {:.1f} deg\n'
                'Phase off: {:.1f} $\pm$ {:.1f}deg\n'
                'Phase on: {:.1f} $\pm$ {:.1f}deg\n'
                'Osc. amp. off: {:.4f} $\pm$ {:.4f}\n'
                'Osc. amp. on: {:.4f} $\pm$ {:.4f}\n'
                'Offs. diff.: {:.4f} $\pm$ {:.4f}\n'
                'Osc. offs. off: {:.4f} $\pm$ {:.4f}\n'
                'Osc. offs. on: {:.4f} $\pm$ {:.4f}'.format(
                    self.proc_data_dict['phi_cond'][0],
                    self.proc_data_dict['phi_cond'][1],
                    self.proc_data_dict['phi_0'][0],
                    self.proc_data_dict['phi_0'][1],
                    self.proc_data_dict['phi_1'][0],
                    self.proc_data_dict['phi_1'][1],
                    self.proc_data_dict['osc_amp_0'][0],
                    self.proc_data_dict['osc_amp_0'][1],
                    self.proc_data_dict['osc_amp_1'][0],
                    self.proc_data_dict['osc_amp_1'][1],
                    self.proc_data_dict['offs_diff'][0],
                    self.proc_data_dict['offs_diff'][1],
                    self.proc_data_dict['osc_offs_0'][0],
                    self.proc_data_dict['osc_offs_0'][1],
                    self.proc_data_dict['osc_offs_1'][0],
                    self.proc_data_dict['osc_offs_1'][1]))
            self.plot_dicts['phase_message'] = {
                'ax_id': 'main',
                'ypos': 0.9,
                'xpos': 1.45,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': phase_message}

    def _prepare_spectator_qubit_figure(self):

        self.plot_dicts['spectator_qubit'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_off'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['spec_on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'spectator_qubit',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_on'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            leak_msg = (
                'Missing fraction: {:.2f} % '.format(
                    self.proc_data_dict['missing_fraction']*100))
            self.plot_dicts['leak_msg'] = {
                'ax_id': 'spectator_qubit',
                'ypos': 0.7,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': leak_msg}
            # offset as a guide for the eye
            y = self.fit_res['cos_fit_on'].params['offset'].value
            self.plot_dicts['cos_on_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C1', 'linestyle': 'dotted'}
                    }


class StateTomographyAnalysis(ba.BaseDataAnalysis):
    """
    Analyses the results of the state tomography experiment and calculates
    the corresponding quantum state.

    Possible options that can be passed in the options_dict parameter:
        cal_points: A data structure specifying the indices of the calibration
                    points. See the AveragedTimedomainAnalysis for format.
                    The calibration points need to be in the same order as the
                    used basis for the result.
        data_type: 'averaged' or 'singleshot'. For singleshot data each
                   measurement outcome is saved and arbitrary order correlations
                   between the states can be calculated.
        meas_operators: (optional) A list of qutip operators or numpy 2d arrays.
                        This overrides the measurement operators otherwise
                        found from the calibration points.
        covar_matrix: (optional) The covariance matrix of the measurement
                      operators as a 2d numpy array. Overrides the one found
                      from the calibration points.
        use_covariance_matrix (bool): Flag to define whether to use the
            covariance matrix
        basis_rots_str: A list of standard PycQED pulse names that were
                             applied to qubits before measurement
        basis_rots: As an alternative to single_qubit_pulses, the basis
                    rotations applied to the system as qutip operators or numpy
                    matrices can be given.
        mle: True/False, whether to do maximum likelihood fit. If False, only
             least squares fit will be done, which could give negative
             eigenvalues for the density matrix.
        rho_target (optional): A qutip density matrix that the result will be
                               compared to when calculating fidelity.
    """
    def __init__(self, *args, **kwargs):
        auto = kwargs.pop('auto', True)
        super().__init__(*args, **kwargs)
        kwargs['auto'] = auto
        self.single_timestamp = True
        self.params_dict = {'exp_metadata': 'exp_metadata'}
        self.numeric_params = []
        self.data_type = self.options_dict['data_type']
        if self.data_type == 'averaged':
            self.base_analysis = AveragedTimedomainAnalysis(*args, **kwargs)
        elif self.data_type == 'singleshot':
            self.base_analysis = roa.MultiQubit_SingleShot_Analysis(
                *args, **kwargs)
        else:
            raise KeyError("Invalid tomography data mode: '" + self.data_type +
                           "'. Valid modes are 'averaged' and 'singleshot'.")

        if kwargs.get('auto', True):
            self.run_analysis()

    def process_data(self):
        tomography_qubits = self.options_dict.get('tomography_qubits', None)
        data, Fs, Omega = self.base_analysis.measurement_operators_and_results(
                              tomography_qubits)
        if 'data_filter' in self.options_dict:
            data = self.options_dict['data_filter'](data.T).T

        data = data.T
        for i, v in enumerate(data):
            data[i] = v / v.sum()
        data = data.T

        Fs = self.options_dict.get('meas_operators', Fs)
        Fs = [qtp.Qobj(F) for F in Fs]
        d = Fs[0].shape[0]
        self.proc_data_dict['d'] = d
        Omega = self.options_dict.get('covar_matrix', Omega)
        if Omega is None:
            Omega = np.diag(np.ones(len(Fs)))
        elif len(Omega.shape) == 1:
            Omega = np.diag(Omega)

        metadata = self.raw_data_dict.get('exp_metadata',
                                          self.options_dict.get(
                                              'exp_metadata', {}))
        if metadata is None:
            metadata = {}
        self.raw_data_dict['exp_metadata'] = metadata
        basis_rots_str = metadata.get('basis_rots_str', None)
        basis_rots_str = self.options_dict.get('basis_rots_str', basis_rots_str)
        if basis_rots_str is not None:
            nr_qubits = int(np.round(np.log2(d)))
            pulse_list = list(itertools.product(basis_rots_str,
                                                repeat=nr_qubits))
            rotations = tomo.standard_qubit_pulses_to_rotations(pulse_list)
        else:
            rotations = metadata.get('basis_rots', None)
            rotations = self.options_dict.get('basis_rots', rotations)
            if rotations is None:
                raise KeyError("Either 'basis_rots_str' or 'basis_rots' "
                               "parameter must be passed in the options "
                               "dictionary or in the experimental metadata.")
        rotations = [qtp.Qobj(U) for U in rotations]

        all_Fs = tomo.rotated_measurement_operators(rotations, Fs)
        all_Fs = list(itertools.chain(*np.array(all_Fs, dtype=np.object).T))
        all_mus = np.array(list(itertools.chain(*data.T)))
        all_Omegas = sp.linalg.block_diag(*[Omega] * len(data[0]))


        self.proc_data_dict['meas_operators'] = all_Fs
        self.proc_data_dict['covar_matrix'] = all_Omegas
        self.proc_data_dict['meas_results'] = all_mus

        if self.options_dict.get('pauli_raw', False):
            pauli_raw = self.generate_raw_pauli_set()
            rho_raw = tomo.pauli_set_to_density_matrix(pauli_raw)
            self.proc_data_dict['rho_raw'] = rho_raw
            self.proc_data_dict['rho'] = rho_raw
        else:
            rho_ls = tomo.least_squares_tomography(
                all_mus, all_Fs,
                all_Omegas if self.get_param_value('use_covariance_matrix', False)
                else None )
            self.proc_data_dict['rho_ls'] = rho_ls
            self.proc_data_dict['rho'] = rho_ls
            if self.options_dict.get('mle', False):
                rho_mle = tomo.mle_tomography(
                    all_mus, all_Fs, None,
                    all_Omegas if self.get_param_value('use_covariance_matrix', False) else None,
                    rho_guess=rho_ls)
                self.proc_data_dict['rho_mle'] = rho_mle
                self.proc_data_dict['rho'] = rho_mle

        rho = self.proc_data_dict['rho']
        self.proc_data_dict['purity'] = (rho * rho).tr().real

        rho_target = metadata.get('rho_target', None)
        rho_target = self.options_dict.get('rho_target', rho_target)
        if rho_target is not None:
            self.proc_data_dict['fidelity'] = tomo.fidelity(rho, rho_target)
        if d == 4:
            self.proc_data_dict['concurrence'] = tomo.concurrence(rho)

    def prepare_plots(self):
        self.prepare_density_matrix_plot()
        d = self.proc_data_dict['d']
        if 2 ** (d.bit_length() - 1) == d:
            # dimension is power of two, plot expectation values of pauli
            # operators
            self.prepare_pauli_basis_plot()

    def prepare_density_matrix_plot(self):
        self.tight_fig = self.options_dict.get('tight_fig', False)
        rho_target = self.raw_data_dict['exp_metadata'].get('rho_target', None)
        rho_target = self.options_dict.get('rho_target', rho_target)
        d = self.proc_data_dict['d']
        xtick_labels = self.options_dict.get('rho_ticklabels', None)
        ytick_labels = self.options_dict.get('rho_ticklabels', None)
        if 2 ** (d.bit_length() - 1) == d:
            nr_qubits = d.bit_length() - 1
            fmt_string = '{{:0{}b}}'.format(nr_qubits)
            labels = [fmt_string.format(i) for i in range(2 ** nr_qubits)]
            if xtick_labels is None:
                xtick_labels = ['$|' + lbl + r'\rangle$' for lbl in labels]
            if ytick_labels is None:
                ytick_labels = [r'$\langle' + lbl + '|$' for lbl in labels]
        color = (0.5 * np.angle(self.proc_data_dict['rho'].full()) / np.pi) % 1.
        cmap = self.options_dict.get('rho_colormap', self.default_phase_cmap())
        if self.options_dict.get('pauli_raw', False):
            title = 'Density matrix reconstructed from the Pauli set\n'
        elif self.options_dict.get('mle', False):
            title = 'Maximum likelihood fit of the density matrix\n'
        else:
            title = 'Least squares fit of the density matrix\n'
        empty_artist = mpl.patches.Rectangle((0, 0), 0, 0, visible=False)
        legend_entries = [(empty_artist,
                           r'Purity, $Tr(\rho^2) = {:.1f}\%$'.format(
                               100 * self.proc_data_dict['purity']))]
        if rho_target is not None:
            legend_entries += [
                (empty_artist, r'Fidelity, $F = {:.1f}\%$'.format(
                    100 * self.proc_data_dict['fidelity']))]
        if d == 4:
            legend_entries += [
                (empty_artist, r'Concurrence, $C = {:.2f}$'.format(
                    self.proc_data_dict['concurrence']))]
        meas_string = self.base_analysis.\
            raw_data_dict['measurementstring']
        if isinstance(meas_string, list):
            if len(meas_string) > 1:
                meas_string = meas_string[0] + ' to ' + meas_string[-1]
            else:
                meas_string = meas_string[0]
        self.plot_dicts['density_matrix'] = {
            'plotfn': self.plot_bar3D,
            '3d': True,
            '3d_azim': -35,
            '3d_elev': 35,
            'xvals': np.arange(d),
            'yvals': np.arange(d),
            'zvals': np.abs(self.proc_data_dict['rho'].full()),
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
            'title': (title + self.raw_data_dict['timestamp'] + ' ' +
                      meas_string),
            'do_legend': True,
            'legend_entries': legend_entries,
            'legend_kws': dict(loc='upper left', bbox_to_anchor=(0, 0.94))
        }

        if rho_target is not None:
            rho_target = qtp.Qobj(rho_target)
            if rho_target.type == 'ket':
                rho_target = rho_target * rho_target.dag()
            elif rho_target.type == 'bra':
                rho_target = rho_target.dag() * rho_target
            self.plot_dicts['density_matrix_target'] = {
                'plotfn': self.plot_bar3D,
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
                'title': ('Target density matrix\n' +
                          self.raw_data_dict['timestamp'] + ' ' +
                          meas_string),
                'bar_kws': dict(zorder=1),
            }

    def generate_raw_pauli_set(self):
        nr_qubits = self.proc_data_dict['d'].bit_length() - 1
        pauli_raw_values = []
        for op in tomo.generate_pauli_set(nr_qubits)[1]:
            nr_terms = 0
            sum_terms = 0.
            for meas_op, meas_res in zip(self.proc_data_dict['meas_operators'],
                                         self.proc_data_dict['meas_results']):
                trace = (meas_op*op).tr().real
                clss = int(trace*2)
                if clss < 0:
                    sum_terms -= meas_res
                    nr_terms += 1
                elif clss > 0:
                    sum_terms += meas_res
                    nr_terms += 1
            pauli_raw_values.append(2**nr_qubits*sum_terms/nr_terms)
        return pauli_raw_values

    def prepare_pauli_basis_plot(self):
        yexp = tomo.density_matrix_to_pauli_basis(self.proc_data_dict['rho'])
        nr_qubits = self.proc_data_dict['d'].bit_length() - 1
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
        if self.options_dict.get('pauli_raw', False):
            fit_type = 'raw counts'
        elif self.options_dict.get('mle', False):
            fit_type = 'maximum likelihood estimation'
        else:
            fit_type = 'least squares fit'
        meas_string = self.base_analysis. \
            raw_data_dict['measurementstring']
        if np.ndim(meas_string) > 0:
            if len(meas_string) > 1:
                meas_string = meas_string[0] + ' to ' + meas_string[-1]
            else:
                meas_string = meas_string[0]
        self.plot_dicts['pauli_basis'] = {
            'plotfn': self.plot_bar,
            'xcenters': np.arange(len(order)),
            'xwidth': 0.4,
            'xrange': (-1, len(order)),
            'yvals': np.array(yexp)[order],
            'xlabel': r'Pauli operator, $\hat{O}$',
            'ylabel': r'Expectation value, $\mathrm{Tr}(\hat{O} \hat{\rho})$',
            'title': 'Pauli operators, ' + fit_type + '\n' +
                      self.raw_data_dict['timestamp'] + ' ' + meas_string,
            'yrange': (-1.1, 1.1),
            'xtick_loc': np.arange(4**nr_qubits - 1),
            'xtick_rotation': 90,
            'xtick_labels': np.array(labels)[order],
            'bar_kws': dict(zorder=10),
            'setlabel': 'Fit to experiment',
            'do_legend': True
        }
        if nr_qubits > 2:
            self.plot_dicts['pauli_basis']['plotsize'] = (10, 5)

        rho_target = self.raw_data_dict['exp_metadata'].get('rho_target', None)
        rho_target = self.options_dict.get('rho_target', rho_target)
        if rho_target is not None:
            rho_target = qtp.Qobj(rho_target)
            ytar = tomo.density_matrix_to_pauli_basis(rho_target)
            self.plot_dicts['pauli_basis_target'] = {
                'plotfn': self.plot_bar,
                'ax_id': 'pauli_basis',
                'xcenters': np.arange(len(order)),
                'xwidth': 0.8,
                'yvals': np.array(ytar)[order],
                'xtick_loc': np.arange(len(order)),
                'xtick_labels': np.array(labels)[order],
                'bar_kws': dict(color='0.8', zorder=0),
                'setlabel': 'Target values',
                'do_legend': True
            }

        purity_str = r'Purity, $Tr(\rho^2) = {:.1f}\%$'.format(
            100 * self.proc_data_dict['purity'])
        if rho_target is not None:
            fidelity_str = '\n' + r'Fidelity, $F = {:.1f}\%$'.format(
                100 * self.proc_data_dict['fidelity'])
        else:
            fidelity_str = ''
        if self.proc_data_dict['d'] == 4:
            concurrence_str = '\n' + r'Concurrence, $C = {:.1f}\%$'.format(
                100 * self.proc_data_dict['concurrence'])
        else:
            concurrence_str = ''
        self.plot_dicts['pauli_info_labels'] = {
            'ax_id': 'pauli_basis',
            'plotfn': self.plot_line,
            'xvals': [0],
            'yvals': [0],
            'line_kws': {'alpha': 0},
            'setlabel': purity_str + fidelity_str,
            'do_legend': True
        }

    def default_phase_cmap(self):
        cols = np.array(((41, 39, 231), (61, 130, 163), (208, 170, 39),
                         (209, 126, 4), (181, 28, 20), (238, 76, 152),
                         (251, 130, 242), (162, 112, 251))) / 255
        n = len(cols)
        cdict = {
            'red': [[i/n, cols[i%n][0], cols[i%n][0]] for i in range(n+1)],
            'green': [[i/n, cols[i%n][1], cols[i%n][1]] for i in range(n+1)],
            'blue': [[i/n, cols[i%n][2], cols[i%n][2]] for i in range(n+1)],
        }

        return mpl.colors.LinearSegmentedColormap('DMDefault', cdict)


class ReadoutROPhotonsAnalysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analyses the photon number in the RO based on the
    readout_photons_in_resonator function

    function specific options for options dict:
    f_qubit
    chi
    artif_detuning
    print_fit_results
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 close_figs: bool=False, options_dict: dict=None,
                 extract_only: bool=False, do_fitting: bool=False,
                 auto: bool=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs, label=label,
                         extract_only=extract_only, do_fitting=do_fitting)
        if self.options_dict.get('TwoD', None) is None:
            self.options_dict['TwoD'] = True
        self.label = label
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'sweep_points': 'sweep_points',
            'sweep_points_2D': 'sweep_points_2D',
            'value_names': 'value_names',
            'value_units': 'value_units',
            'measured_values': 'measured_values'}

        self.numeric_params = self.options_dict.get('numeric_params',
                                                   OrderedDict())

        self.kappa = self.options_dict.get('kappa_effective', None)
        self.chi = self.options_dict.get('chi', None)
        self.T2 = self.options_dict.get('T2echo', None)
        self.artif_detuning = self.options_dict.get('artif_detuning', 0)

        if (self.kappa is None) or (self.chi is None) or (self.T2 is None):
            raise ValueError('kappa_effective, chi and T2echo must be passed to '
                             'the options_dict.')

        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        self.proc_data_dict['qubit_state'] = [[],[]]
        self.proc_data_dict['delay_to_relax'] = self.raw_data_dict[
                                                    'sweep_points_2D'][0]
        self.proc_data_dict['ramsey_times'] = []

        for i,x in enumerate(np.transpose(self.raw_data_dict[
                        'measured_data']['raw w0 _measure'][0])):
            self.proc_data_dict['qubit_state'][0].append([])
            self.proc_data_dict['qubit_state'][1].append([])

            for j,y in enumerate(np.transpose(self.raw_data_dict[
                    'measured_data']['raw w0 _measure'][0])[i]):

                if j%2 == 0:
                    self.proc_data_dict['qubit_state'][0][i].append(y)

                else:
                    self.proc_data_dict['qubit_state'][1][i].append(y)
        for i,x in enumerate( self.raw_data_dict['sweep_points'][0]):
            if i % 2 == 0:
                self.proc_data_dict['ramsey_times'].append(x)

    #I STILL NEED to pass Chi
    def prepare_fitting(self):
        self.proc_data_dict['photon_number'] = [[],[]]
        self.proc_data_dict['fit_results'] = []
        self.proc_data_dict['ramsey_fit_results'] = [[],[]]


        for i,tau in enumerate(self.proc_data_dict['delay_to_relax']):

            self.proc_data_dict['ramsey_fit_results'][0].append(self.fit_Ramsey(
                            self.proc_data_dict['ramsey_times'][:-4],
                            self.proc_data_dict['qubit_state'][0][i][:-4]/
                            max(self.proc_data_dict['qubit_state'][0][i][:-4]),
                            state=0,
                            kw=self.options_dict))

            self.proc_data_dict['ramsey_fit_results'][1].append(self.fit_Ramsey(
                            self.proc_data_dict['ramsey_times'][:-4],
                            self.proc_data_dict['qubit_state'][1][i][:-4]/
                            max(self.proc_data_dict['qubit_state'][1][i][:-4]),
                            state=1,
                            kw=self.options_dict))

            n01 = self.proc_data_dict['ramsey_fit_results'
                                         ][0][i][0].params['n0'].value
            n02 = self.proc_data_dict['ramsey_fit_results'
                                         ][1][i][0].params['n0'].value

            self.proc_data_dict['photon_number'][0].append(n01)
            self.proc_data_dict['photon_number'][1].append(n02)


    def run_fitting(self):
        print_fit_results = self.params_dict.pop('print_fit_results',False)

        exp_dec_mod = lmfit.Model(fit_mods.ExpDecayFunc)
        exp_dec_mod.set_param_hint('n',
                                   value=1,
                                   vary=False)
        exp_dec_mod.set_param_hint('offset',
                                   value=0,
                                   min=0,
                                   vary=True)
        exp_dec_mod.set_param_hint('tau',
                                   value=self.proc_data_dict[
                                                'delay_to_relax'][-1],
                                   min=1e-11,
                                   vary=True)
        exp_dec_mod.set_param_hint('amplitude',
                                   value=1,
                                   min=0,
                                   vary=True)
        params = exp_dec_mod.make_params()
        self.fit_res = OrderedDict()
        self.fit_res['ground_state'] = exp_dec_mod.fit(
                                data=self.proc_data_dict['photon_number'][0],
                                params=params,
                                t=self.proc_data_dict['delay_to_relax'])
        self.fit_res['excited_state'] = exp_dec_mod.fit(
                                data=self.proc_data_dict['photon_number'][1],
                                params=params,
                                t=self.proc_data_dict['delay_to_relax'])
        if print_fit_results:
            print(self.fit_res['ground_state'].fit_report())
            print(self.fit_res['excited_state'].fit_report())

    def fit_Ramsey(self, x, y, state, **kw):

        x = np.array(x)

        y = np.array(y)

        exp_dec_p_mod = lmfit.Model(fit_mods.ExpDecayPmod)
        comb_exp_dec_mod = lmfit.Model(fit_mods.CombinedOszExpDecayFunc)

        average = np.mean(y)

        ft_of_data = np.fft.fft(y)
        index_of_fourier_maximum = np.argmax(np.abs(
            ft_of_data[1:len(ft_of_data) // 2])) + 1
        max_ramsey_delay = x[-1] - x[0]

        fft_axis_scaling = 1 / max_ramsey_delay
        freq_est = fft_axis_scaling * index_of_fourier_maximum

        n_est = (freq_est-self.artif_detuning)/(2 * self.chi)


        exp_dec_p_mod.set_param_hint('T2echo',
                                   value=self.T2,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('offset',
                                   value=average,
                                   min=0,
                                   vary=True)
        exp_dec_p_mod.set_param_hint('delta',
                                   value=self.artif_detuning,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('amplitude',
                                   value=1,
                                   min=0,
                                   vary=True)
        exp_dec_p_mod.set_param_hint('kappa',
                                   value=self.kappa[state],
                                   vary=False)
        exp_dec_p_mod.set_param_hint('chi',
                                   value=self.chi,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('n0',
                                      value=n_est,
                                      min=0,
                                      vary=True)
        exp_dec_p_mod.set_param_hint('phase',
                                       value=0,
                                       vary=True)


        comb_exp_dec_mod.set_param_hint('tau',
                                     value=self.T2,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('offset',
                                        value=average,
                                        min=0,
                                        vary=True)
        comb_exp_dec_mod.set_param_hint('oscillation_offset',
                                        value=average,
                                        min=0,
                                        vary=True)
        comb_exp_dec_mod.set_param_hint('amplitude',
                                     value=1,
                                     min=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('tau_gauss',
                                     value=self.kappa[state],
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('n0',
                                     value=n_est,
                                     min=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('phase',
                                     value=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('delta',
                                     value=self.artif_detuning,
                                     vary=False)
        comb_exp_dec_mod.set_param_hint('chi',
                                     value=self.chi,
                                     vary=False)

        if (np.average(y[:4]) >
                np.average(y[4:8])):
            phase_estimate = 0
        else:
            phase_estimate = np.pi
        exp_dec_p_mod.set_param_hint('phase',
                                     value=phase_estimate, vary=True)
        comb_exp_dec_mod.set_param_hint('phase',
                                     value=phase_estimate, vary=True)

        amplitude_guess = 0.5
        if np.all(np.logical_and(y >= 0, y <= 1)):
            exp_dec_p_mod.set_param_hint('amplitude',
                                         value=amplitude_guess,
                                         min=0.00,
                                         max=4.0,
                                         vary=True)
            comb_exp_dec_mod.set_param_hint('amplitude',
                                         value=amplitude_guess,
                                         min=0.00,
                                         max=4.0,
                                         vary=True)

        else:
            print('data is not normalized, varying amplitude')
            exp_dec_p_mod.set_param_hint('amplitude',
                                         value=max(y),
                                         min=0.00,
                                         max=4.0,
                                         vary=True)
            comb_exp_dec_mod.set_param_hint('amplitude',
                                        value=max(y),
                                        min=0.00,
                                        max=4.0,
                                        vary=True)

        fit_res_1 = exp_dec_p_mod.fit(data=y,
                                    t=x,
                                    params= exp_dec_p_mod.make_params())

        fit_res_2 = comb_exp_dec_mod.fit(data=y,
                                         t=x,
                                         params= comb_exp_dec_mod.make_params())


        if fit_res_1.chisqr > .35:
            log.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2*np.pi, 10):

                for i, del_amp in enumerate(np.linspace(
                        -max(y)/10, max(y)/10, 10)):
                    exp_dec_p_mod.set_param_hint('phase',
                                                 value=phase_estimate,
                                                 vary=False)
                    exp_dec_p_mod.set_param_hint('amplitude',
                                                 value=max(y)+ del_amp)

                    fit_res_lst += [exp_dec_p_mod.fit(
                        data=y,
                        t=x,
                        params= exp_dec_p_mod.make_params())]

            chisqr_lst = [fit_res_1.chisqr for fit_res_1 in fit_res_lst]
            fit_res_1 = fit_res_lst[np.argmin(chisqr_lst)]

        if fit_res_2.chisqr > .35:
            log.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2*np.pi, 10):

                for i, del_amp in enumerate(np.linspace(
                        -max(y)/10, max(y)/10, 10)):
                    comb_exp_dec_mod.set_param_hint('phase',
                                                 value=phase_estimate,
                                                 vary=False)
                    comb_exp_dec_mod.set_param_hint('amplitude',
                                                 value=max(y)+ del_amp)

                    fit_res_lst += [comb_exp_dec_mod.fit(
                        data=y,
                        t=x,
                        params= comb_exp_dec_mod.make_params())]

            chisqr_lst = [fit_res_2.chisqr for fit_res_2 in fit_res_lst]
            fit_res_2 = fit_res_lst[np.argmin(chisqr_lst)]

        if fit_res_1.chisqr < fit_res_2.chisqr:
            self.proc_data_dict['params'] = exp_dec_p_mod.make_params()
            return [fit_res_1,fit_res_1,fit_res_2]
        else:
            self.proc_data_dict['params'] = comb_exp_dec_mod.make_params()
            return [fit_res_2,fit_res_1,fit_res_2]


    def prepare_plots(self):
            self.prepare_2D_sweep_plot()
            self.prepare_photon_number_plot()
            self.prepare_ramsey_plots()

    def prepare_2D_sweep_plot(self):
        self.plot_dicts['off_full_data_'+self.label] = {
            'title': 'Raw data |g>',
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['ramsey_times'],
            'xlabel': 'Ramsey delays',
            'xunit': 's',
            'yvals': self.proc_data_dict['delay_to_relax'],
            'ylabel': 'Delay after first RO-pulse',
            'yunit': 's',
            'zvals': np.array(self.proc_data_dict['qubit_state'][0]) }

        self.plot_dicts['on_full_data_'+self.label] = {
            'title': 'Raw data |e>',
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['ramsey_times'],
            'xlabel': 'Ramsey delays',
            'xunit': 's',
            'yvals': self.proc_data_dict['delay_to_relax'],
            'ylabel': 'Delay after first RO-pulse',
            'yunit': 's',
            'zvals': np.array(self.proc_data_dict['qubit_state'][1])  }



    def prepare_ramsey_plots(self):
        x_fit = np.linspace(self.proc_data_dict['ramsey_times'][0],
                            max(self.proc_data_dict['ramsey_times']),101)
        for i in range(len(self.proc_data_dict['ramsey_fit_results'][0])):

            self.plot_dicts['off_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+\
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['ramsey_times'],
                'xlabel': 'Ramsey delays',
                'xunit': 's',
                'yvals': np.array(self.proc_data_dict['qubit_state'][0][i]/
                             max(self.proc_data_dict['qubit_state'][0][i][:-4])),
                'ylabel': 'Measured qubit state',
                'yunit': '',
                'marker': 'o',
                'setlabel': '|g> data_'+str(i),
                'do_legend': True }

            self.plot_dicts['off_fit_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][0][i][1].eval(
                    self.proc_data_dict['ramsey_fit_results'][0][i][1].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|g> fit_model'+str(i),
                'do_legend': True  }

            self.plot_dicts['off_fit_2_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][0][i][2].eval(
                    self.proc_data_dict['ramsey_fit_results'][0][i][2].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|g> fit_simpel_model'+str(i),
                'do_legend': True  }

            self.plot_dicts['hidden_g_'+str(i)] = {
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': [0],
                'yvals': [0],
                'color': 'w',
                'setlabel': 'Residual photon count = '
                             ''+str(self.proc_data_dict['photon_number'][0][i]),
                'do_legend': True }


            self.plot_dicts['on_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['ramsey_times'],
                'xlabel': 'Ramsey delays',
                'xunit': 's',
                'yvals':  np.array(self.proc_data_dict['qubit_state'][1][i]/
                             max(self.proc_data_dict['qubit_state'][1][i][:-4])),
                'ylabel': 'Measured qubit state',
                'yunit': '',
                'marker': 'o',
                'setlabel': '|e> data_'+str(i),
                'do_legend': True }

            self.plot_dicts['on_fit_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][1][i][1].eval(
                    self.proc_data_dict['ramsey_fit_results'][1][i][1].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|e> fit_model'+str(i),
                'do_legend': True }

            self.plot_dicts['on_fit_2_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][1][i][2].eval(
                    self.proc_data_dict['ramsey_fit_results'][1][i][2].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|e> fit_simpel_model'+str(i),
                'do_legend': True }

            self.plot_dicts['hidden_e_'+str(i)] = {
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': [0],
                'yvals': [0],
                'color': 'w',
                'setlabel': 'Residual photon count = '
                            ''+str(self.proc_data_dict['photon_number'][1][i]),
                'do_legend': True }


    def prepare_photon_number_plot(self):


        ylabel = 'Average photon number'
        yunit = ''

        x_fit = np.linspace(min(self.proc_data_dict['delay_to_relax']),
                            max(self.proc_data_dict['delay_to_relax']),101)
        minmax_data = [min(min(self.proc_data_dict['photon_number'][0]),
                           min(self.proc_data_dict['photon_number'][1])),
                       max(max(self.proc_data_dict['photon_number'][0]),
                           max(self.proc_data_dict['photon_number'][1]))]
        minmax_data[0] -= minmax_data[0]/5
        minmax_data[1] += minmax_data[1]/5

        self.proc_data_dict['photon_number'][1],

        self.fit_res['excited_state'].eval(
            self.fit_res['excited_state'].params,
            t=x_fit)
        self.plot_dicts['Photon number count'] = {
            'plotfn': self.plot_line,
            'xlabel': 'Delay after first RO-pulse',
            'ax_id': 'Photon number count ',
            'xunit': 's',
            'xvals': self.proc_data_dict['delay_to_relax'],
            'yvals': self.proc_data_dict['photon_number'][0],
            'ylabel': ylabel,
            'yunit': yunit,
            'yrange': minmax_data,
            'title': 'Residual photon number',
            'color': 'b',
            'linestyle': '',
            'marker': 'o',
            'setlabel': '|g> data',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main2'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'xvals': x_fit,
            'yvals': self.fit_res['ground_state'].eval(
                self.fit_res['ground_state'].params,
                t=x_fit),
            'yrange': minmax_data,
            'ax_id': 'Photon number count ',
            'color': 'b',
            'linestyle': '-',
            'marker': '',
            'setlabel': '|g> fit',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main3'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'xvals': self.proc_data_dict['delay_to_relax'],
            'yvals': self.proc_data_dict['photon_number'][1],
            'yrange': minmax_data,
            'ax_id': 'Photon number count ',
            'color': 'r',
            'linestyle': '',
            'marker': 'o',
            'setlabel': '|e> data',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main4'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'ax_id': 'Photon number count ',
            'xvals': x_fit,
            'yvals': self.fit_res['excited_state'].eval(
                self.fit_res['excited_state'].params,
                t=x_fit),
            'yrange': minmax_data,
            'ylabel': ylabel,
            'color': 'r',
            'linestyle': '-',
            'marker': '',
            'setlabel': '|e> fit',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['hidden_1'] = {
            'ax_id': 'Photon number count ',
            'plotfn': self.plot_line,
            'yrange': minmax_data,
            'xvals': [0],
            'yvals': [0],
            'color': 'w',
            'setlabel': 'tau_g = '
                        ''+str("%.3f" %
                        (self.fit_res['ground_state'].params['tau'].value*1e9))+''
                        ' ns',
            'do_legend': True }


        self.plot_dicts['hidden_2'] = {
            'ax_id': 'Photon number count ',
            'plotfn': self.plot_line,
            'yrange': minmax_data,
            'xvals': [0],
            'yvals': [0],
            'color': 'w',
            'setlabel': 'tau_e = '
                        ''+str("%.3f" %
                        (self.fit_res['excited_state'].params['tau'].value*1e9))+''
                        ' ns',
            'do_legend': True}


class RODynamicPhaseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names: list=None,  t_start: str=None, t_stop: str=None,
                 data_file_path: str=None, single_timestamp: bool=False,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(qb_names=qb_names, t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting,
                         auto=False)

        if auto:
            self.run_analysis()

    def process_data(self):

        super().process_data()

        if 'qbp_name' in self.metadata:
            self.pulsed_qbname = self.metadata['qbp_name']
        else:
            self.pulsed_qbname = self.options_dict.get('pulsed_qbname')
        self.measured_qubits = [qbn for qbn in self.channel_map if
                                qbn != self.pulsed_qbname]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.measured_qubits:
            ro_dict = self.proc_data_dict['projected_data_dict'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            for ro_suff, data in ro_dict.items():
                cos_mod = lmfit.Model(fit_mods.CosFunc)
                if self.num_cal_points != 0:
                    data = data[:-self.num_cal_points]
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod,
                    t=sweep_points,
                    data=data)
                guess_pars['amplitude'].vary = True
                guess_pars['offset'].vary = True
                guess_pars['frequency'].vary = True
                guess_pars['phase'].vary = True

                key = 'cos_fit_{}{}'.format(qbn, ro_suff)
                self.fit_dicts[key] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):

        self.dynamic_phases = OrderedDict()
        for meas_qbn in self.measured_qubits:
            self.dynamic_phases[meas_qbn] = \
                (self.fit_dicts['cos_fit_{}_measure'.format(meas_qbn)][
                    'fit_res'].best_values['phase'] -
                 self.fit_dicts['cos_fit_{}_ref_measure'.format(meas_qbn)][
                    'fit_res'].best_values['phase'])*180/np.pi

    def prepare_plots(self):

        super().prepare_plots()

        if self.do_fitting:
            for meas_qbn in self.measured_qubits:
                sweep_points_dict = self.proc_data_dict['sweep_points_dict'][
                    meas_qbn]
                if self.num_cal_points != 0:
                    yvals = [self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_ref_measure'][:-self.num_cal_points],
                             self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_measure'][:-self.num_cal_points]]
                    sweep_points = sweep_points_dict['msmt_sweep_points']

                    # plot cal points
                    for i, cal_pts_idxs in enumerate(
                            self.cal_states_dict.values()):
                        key = list(self.cal_states_dict)[i] + meas_qbn
                        self.plot_dicts[key] = {
                            'fig_id': 'dyn_phase_plot_' + meas_qbn,
                            'plotfn': self.plot_line,
                            'xvals': np.mean([
                                sweep_points_dict['cal_points_sweep_points'][
                                    cal_pts_idxs],
                                sweep_points_dict['cal_points_sweep_points'][
                                    cal_pts_idxs]],
                                axis=0),
                            'yvals': np.mean([
                                self.proc_data_dict['projected_data_dict'][meas_qbn][
                                    '_ref_measure'][cal_pts_idxs],
                                self.proc_data_dict['projected_data_dict'][meas_qbn][
                                    '_measure'][cal_pts_idxs]],
                                             axis=0),
                            'setlabel': list(self.cal_states_dict)[i],
                            'do_legend': True,
                            'legend_bbox_to_anchor': (1, 0.5),
                            'legend_pos': 'center left',
                            'linestyle': 'none',
                            'line_kws': {'color': self.get_cal_state_color(
                                list(self.cal_states_dict)[i])}}

                else:
                    yvals = [self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_ref_measure'],
                             self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_measure']]
                    sweep_points = sweep_points_dict['sweep_points']

                self.plot_dicts['dyn_phase_plot_' + meas_qbn] = {
                    'plotfn': self.plot_line,
                    'xvals': [sweep_points, sweep_points],
                    'xlabel': self.raw_data_dict['xlabel'][0],
                    'xunit': self.raw_data_dict['xunit'][0][0],
                    'yvals': yvals,
                    'ylabel': 'Excited state population',
                    'yunit': '',
                    'setlabel': ['with measurement', 'no measurement'],
                    'title': (self.raw_data_dict['timestamps'][0] + ' ' +
                              self.raw_data_dict['measurementstring'][0]),
                    'linestyle': 'none',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                self.plot_dicts['cos_fit_' + meas_qbn + '_ref_measure'] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['cos_fit_{}_ref_measure'.format(
                                    meas_qbn)]['fit_res'],
                    'setlabel': 'cos fit',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                self.plot_dicts['cos_fit_' + meas_qbn + '_measure'] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['cos_fit_{}_measure'.format(
                                    meas_qbn)]['fit_res'],
                    'setlabel': 'cos fit',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                textstr = 'Dynamic phase = {:.2f}'.format(
                    self.dynamic_phases[meas_qbn]) + r'$^{\circ}$'
                self.plot_dicts['text_msg_' + meas_qbn] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'ypos': -0.175,
                    'xpos': 0.5,
                    'horizontalalignment': 'center',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}

class FluxAmplitudeSweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def __init__(self, qb_names, *args, **kwargs):
        self.mask_freq = kwargs.pop('mask_freq', None)
        self.mask_amp = kwargs.pop('mask_amp', None)

        super().__init__(qb_names, *args, **kwargs)

    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_sp = {qb: len(pdd['sweep_points_dict'][qb]['sweep_points']) \
            for qb in self.qb_names}
        nr_sp2d = {qb: len(pdd['sweep_points_2D_dict'][qb][self.raw_data_dict['sweep_parameter_names'][1]])\
            for qb in self.qb_names}
        nr_cp = self.num_cal_points

        # make matrix out of vector
        data_reshaped = {qb: \
            np.reshape(deepcopy(pdd['data_to_fit'][qb]),(nr_sp[qb], nr_sp2d[qb])) \
            for qb in self.qb_names}

        pdd['data_reshaped'] = data_reshaped

        # remove calibration points from data to fit
        data_no_cp = {qb: \
            np.array([data_reshaped[qb][i,:] for i in range(nr_sp[qb]-nr_cp)]) \
            for qb in self.qb_names}

        # apply mask
        for qb in self.qb_names:
            if self.mask_freq is None:
                self.mask_freq = [True]*nr_sp2d[qb] # by default, no point is masked
            if self.mask_amp is None:
                self.mask_amp = [True]*(nr_sp[qb]-nr_cp)

        pdd['freqs_masked'] = {}
        pdd['amps_masked'] = {}
        pdd['data_masked'] = {}

        for qb in self.qb_names:
            pdd['freqs_masked'][qb] = \
                pdd['sweep_points_2D_dict'][qb][self.raw_data_dict['sweep_parameter_names'][1]][self.mask_freq]
            pdd['amps_masked'][qb] = pdd['sweep_points_dict'][qb]['sweep_points'][
                                     :-self.num_cal_points][self.mask_amp]
            data_masked = data_no_cp[qb][self.mask_amp,:]
            pdd['data_masked'][qb] = data_masked[:,self.mask_freq]


    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()

        # Gaussian fit of amplitude slices
        gauss_mod = fit_mods.GaussianModel_v2()
        for qb in self.qb_names:
            for i in range(len(pdd['amps_masked'][qb])):
                data = pdd['data_masked'][qb][i,:]
                self.fit_dicts[f'gauss_fit_{qb}_{i}'] = {
                    'model': gauss_mod,
                    'fit_xvals': {'x': pdd['freqs_masked'][qb]},
                    'fit_yvals': {'data': data}
                    }

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['gauss_center'] = {}
        pdd['gauss_center_err'] = {}
        pdd['filtered_center'] = {}
        pdd['filtered_amps'] = {}

        for qb in self.qb_names:
            pdd['gauss_center'][qb] = np.array([
                self.fit_res[f'gauss_fit_{qb}_{i}'].best_values['center']
                for i in range(len(pdd['amps_masked'][qb]))])
            pdd['gauss_center_err'][qb] = np.array([
                self.fit_res[f'gauss_fit_{qb}_{i}'].params['center'].stderr
                for i in range(len(pdd['amps_masked'][qb]))])

            # filter out points with stderr > 1e6 Hz
            pdd['filtered_center'][qb] = np.array([])
            pdd['filtered_amps'][qb] = np.array([])
            for i, stderr in enumerate(pdd['gauss_center_err'][qb]):
                try:
                    if stderr < 1e6:
                        pdd['filtered_center'][qb] = \
                            np.append(pdd['filtered_center'][qb],
                                  pdd['gauss_center'][qb][i])
                        pdd['filtered_amps'][qb] = \
                            np.append(pdd['filtered_amps'][qb],
                            pdd['sweep_points_dict'][qb]\
                            ['sweep_points'][:-self.num_cal_points][i])
                except:
                    continue

            # if gaussian fitting does not work (i.e. all points were filtered
            # out above) use max value of data to get an estimate of freq
            if len(pdd['filtered_amps'][qb]) == 0:
                for qb in self.qb_names:
                    freqs = np.array([])
                    for i in range(pdd['data_masked'][qb].shape[0]):
                        freqs = np.append(freqs, pdd['freqs_masked'][qb]\
                            [np.argmax(pdd['data_masked'][qb][i,:])])
                    pdd['filtered_center'][qb] = freqs
                    pdd['filtered_amps'][qb] = pdd['amps_masked'][qb]

            # fit the freqs to the qubit model
            freq_mod = lmfit.Model(fit_mods.Qubit_dac_to_freq)
            freq_mod.guess = fit_mods.Qubit_dac_arch_guess.__get__(freq_mod, freq_mod.__class__)

            self.fit_dicts[f'freq_fit_{qb}'] = {
                'model': freq_mod,
                'fit_xvals': {'dac_voltage': pdd['filtered_amps'][qb]},
                'fit_yvals': {'data': pdd['filtered_center'][qb]}}

            self.run_fitting()

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            self.plot_dicts[f'data_2d_{qb}'] = {
                'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                'ax_id': f'data_2d_{qb}',
                'plotfn': self.plot_colorxy,
                'xvals': pdd['sweep_points_dict'][qb]['sweep_points'],
                'yvals': pdd['sweep_points_2D_dict'][qb][self.raw_data_dict['sweep_parameter_names'][1]],
                'zvals': np.transpose(pdd['data_reshaped'][qb]),
                'xlabel': r'Flux pulse amplitude',
                'xunit': 'V',
                'ylabel': r'Qubit drive frequency',
                'yunit': 'Hz',
                'zlabel': 'Excited state population',
            }

            if self.do_fitting:
                if self.options_dict.get('scatter', False):
                    label = f'freq_scatter_{qb}'
                    self.plot_dicts[label] = {
                        'title': rdd['measurementstring'] +
                        '\n' + rdd['timestamp'],
                        'ax_id': f'data_2d_{qb}',
                        'plotfn': self.plot_line,
                        'linestyle': '',
                        'xvals': pdd['filtered_amps'][qb],
                        'yvals': pdd['filtered_center'][qb],
                        'xlabel': r'Flux pulse amplitude',
                        'xunit': 'V',
                        'ylabel': r'Qubit drive frequency',
                        'yunit': 'Hz',
                        'color': 'purple'
                        }

                amps = pdd['sweep_points_dict'][qb]['sweep_points'][
                                     :-self.num_cal_points]

                label = f'freq_scatter_{qb}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'data_2d_{qb}',
                    'plotfn': self.plot_line,
                    'linestyle': '-',
                    'xvals': amps,
                    'yvals': fit_mods.Qubit_dac_to_freq(amps,
                            **self.fit_res[f'freq_fit_{qb}'].best_values),
                    'color': 'red'
                }

class T1FrequencySweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_cp = self.num_cal_points
        nr_amps = len(self.metadata['amplitudes'])
        nr_lengths = len(self.metadata['flux_lengths'])

        # make matrix out of vector
        data_reshaped_no_cp = {qb: \
            np.reshape(deepcopy(pdd['data_to_fit'][qb]\
                [:len(pdd['data_to_fit'][qb])-nr_cp]),(\
                nr_amps,nr_lengths)) for qb in self.qb_names}

        pdd['data_reshaped_no_cp'] = data_reshaped_no_cp

    def prepare_fitting(self):
        pdd = self.proc_data_dict

        self.fit_dicts = OrderedDict()
        exp_mod = fit_mods.ExponentialModel()
        for qb in self.qb_names:
            for i, data in enumerate(pdd['data_reshaped_no_cp'][qb]):
                self.fit_dicts[f'exp_fit_{qb}_amp_{i}'] = {
                    'model': exp_mod,
                    'fit_xvals': {'x': self.metadata['flux_lengths']},
                    'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['T1'] = {}
        pdd['T1_err'] = {}
        pdd['mask'] = {}

        for qb in self.qb_names:
            pdd['T1'][qb] = np.array([
                abs(self.fit_res[f'exp_fit_{qb}_amp_{i}'].best_values['decay'])
                for i in range(len(self.metadata['amplitudes']))])

            pdd['T1_err'][qb] = np.array([
                self.fit_res[f'exp_fit_{qb}_amp_{i}'].params['decay'].stderr
                for i in range(len(self.metadata['amplitudes']))])

            pdd['mask'][qb] = []
            for i in range(len(self.metadata['amplitudes'])):
                try:
                    if pdd['T1_err'][qb][i] < 10 * pdd['T1'][qb][i]:
                        pdd['mask'][qb].append(True)
                    else:
                        pdd['mask'][qb].append(False)
                except TypeError:
                    pdd['mask'][qb].append(False)

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:

            # Plot T1 vs flux pulse amplitude
            mask = pdd['mask'][qb]
            label = f'T1_fit_{qb}'
            xvals = self.metadata['amplitudes'][mask] if \
                self.metadata['frequencies'] is None else \
                self.metadata['frequencies'][mask]
            xlabel = r'Flux pulse amplitude' if \
                self.metadata['frequencies'] is None else \
                r'Derived qubit frequency'
            self.plot_dicts[label] = {
                'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                'plotfn': self.plot_line,
                'linestyle': '-',
                'xvals': xvals,
                'yvals': pdd['T1'][qb][mask],
                'yerr': pdd['T1_err'][qb][mask],
                'xlabel': xlabel,
                'xunit': 'V' if self.metadata['frequencies'] is None else 'Hz',
                'ylabel': r'T1',
                'yunit': 's',
                'color': 'blue',
            }

            # Plot rotated integrated average in dependece of flux pulse
            # amplitude and length
            label = f'T1_color_plot_{qb}'
            xvals = self.metadata['amplitudes'][mask] if \
                self.metadata['frequencies'] is None else \
                self.metadata['frequencies'][mask]
            xlabel = r'Flux pulse amplitude' if \
                self.metadata['frequencies'] is None else \
                r'Derived qubit frequency'
            self.plot_dicts[label] = {
                'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                'plotfn': self.plot_colorxy,
                'linestyle': '-',
                'xvals': xvals,
                'yvals': self.metadata['flux_lengths'],
                'zvals': np.transpose(pdd['data_reshaped_no_cp'][qb]),
                'xlabel': xlabel,
                'xunit': 'V' if self.metadata['frequencies'] is None else 'Hz',
                'ylabel': r'Flux pulse length',
                'yunit': 's',
                'zlabel': r'Excited state population'
            }

            # Plot all fits in single figure
            if not self.options_dict.get('all_fits', False):
                continue

            colormap = self.options_dict.get('colormap', mpl.cm.plasma)
            for i in range(len(self.metadata['amplitudes'])):
                color = colormap(i/(len(self.metadata['frequencies'])-1))
                label = f'exp_fit_{qb}_amp_{i}'
                freqs = self.metadata['frequencies'] is not None
                fitid = self.metadata.get('frequencies',
                                          self.metadata['amplitudes'])[i]
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                    'ax_id': f'T1_fits_{qb}',
                    'xlabel': r'Flux pulse length',
                    'xunit': 's',
                    'ylabel': r'Excited state population',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_res[label],
                    'plot_init': self.options_dict.get('plot_init', False),
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                    }

                label = f'freq_scatter_{qb}_{i}'
                self.plot_dicts[label] = {
                    'ax_id': f'T1_fits_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': self.metadata['flux_lengths'],
                    'linestyle': '',
                    'yvals': pdd['data_reshaped_no_cp'][qb][i,:],
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                }

class T2FrequencySweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_cp = self.num_cal_points
        nr_amps = len(self.metadata['amplitudes'])
        nr_lengths = len(self.metadata['flux_lengths'])
        nr_phases = len(self.metadata['phases'])

        # make matrix out of vector
        data_reshaped_no_cp = {qb: \
            np.reshape(deepcopy(pdd['data_to_fit'][qb]\
                [:len(pdd['data_to_fit'][qb])-nr_cp]),(\
                nr_amps, nr_lengths, nr_phases)) for qb in self.qb_names}

        pdd['data_reshaped_no_cp'] = data_reshaped_no_cp
        if self.metadata['use_cal_points']:
            pdd['cal_point_data'] = {qb: deepcopy(pdd['data_to_fit'][qb]\
                    [len(pdd['data_to_fit'][qb])-nr_cp:]) for qb in self.qb_names}

    def prepare_fitting(self):
        pdd = self.proc_data_dict

        self.fit_dicts = OrderedDict()

        nr_amps = len(self.metadata['amplitudes'])


        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        for qb in self.qb_names:
            for i in range(nr_amps):
                for j, data in enumerate(pdd['data_reshaped_no_cp'][qb][i]):
                    self.fit_dicts[f'cos_fit_{qb}_{i}_{j}'] = {
                        'model': cos_mod,
                        'fit_xvals': {'t': self.metadata['phases']},
                        'guess_dict': {'frequency': {'value': 1/360,
                                                 'vary': False}},
                        'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['T2'] = {}
        pdd['T2_err'] = {}
        pdd['mask'] = {}
        pdd['phase_contrast'] = {}
        nr_lengths = len(self.metadata['flux_lengths'])
        nr_amps = len(self.metadata['amplitudes'])

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = {}
            exp_mod = fit_mods.ExponentialModel()
            for i in range(nr_amps):
                pdd['phase_contrast'][qb][f'amp_{i}'] = np.array([self.fit_res[
                                                        f'cos_fit_{qb}_{i}_{j}'
                                                    ].best_values['amplitude']
                                                    for j in
                                                    range(nr_lengths)])

                self.fit_dicts[f'exp_fit_{qb}_{i}'] = {
                    'model': exp_mod,
                    'fit_xvals': {'x': self.metadata['flux_lengths']},
                    'fit_yvals': {'data': np.array([self.fit_res[
                                                        f'cos_fit_{qb}_{i}_{j}'
                                                    ].best_values['amplitude']
                                                    for j in
                                                    range(nr_lengths)])}}

            self.run_fitting()

            pdd['T2'][qb] = np.array([
                abs(self.fit_res[f'exp_fit_{qb}_{i}'].best_values['decay'])
                for i in range(len(self.metadata['amplitudes']))])

            pdd['mask'][qb] = []
            for i in range(len(self.metadata['amplitudes'])):
                try:
                    if self.fit_res[f'exp_fit_{qb}_{i}']\
                                            .params['decay'].stderr < 1e-5:
                        pdd['mask'][qb].append(True)
                    else:
                        pdd['mask'][qb].append(False)
                except TypeError:
                    pdd['mask'][qb].append(False)

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            mask = pdd['mask'][qb]
            label = f'T2_fit_{qb}'
            xvals = self.metadata['amplitudes'][mask] if \
                self.metadata['frequencies'] is None else \
                self.metadata['frequencies'][mask]
            xlabel = r'Flux pulse amplitude' if \
                self.metadata['frequencies'] is None else \
                r'Derived qubit frequency'
            self.plot_dicts[label] = {
                'plotfn': self.plot_line,
                'linestyle': '-',
                'xvals': xvals,
                'yvals': pdd['T2'][qb][mask],
                'xlabel': xlabel,
                'xunit': 'V' if self.metadata['frequencies'] is None else 'Hz',
                'ylabel': r'T2',
                'yunit': 's',
                'color': 'blue',
            }

            # Plot all fits in single figure
            if not self.options_dict.get('all_fits', False):
                continue

            colormap = self.options_dict.get('colormap', mpl.cm.plasma)
            for i in range(len(self.metadata['amplitudes'])):
                color = colormap(i/(len(self.metadata['frequencies'])-1))
                label = f'exp_fit_{qb}_amp_{i}'
                freqs = self.metadata['frequencies'] is not None
                fitid = self.metadata.get('frequencies',
                                          self.metadata['amplitudes'])[i]
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                    'ax_id': f'T2_fits_{qb}',
                    'xlabel': r'Flux pulse length',
                    'xunit': 's',
                    'ylabel': r'Excited state population',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_res[label],
                    'plot_init': self.options_dict.get('plot_init', False),
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                    }

                label = f'freq_scatter_{qb}_{i}'
                self.plot_dicts[label] = {
                    'ax_id': f'T2_fits_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': self.metadata['phases'],
                    'linestyle': '',
                    'yvals': pdd['data_reshaped_no_cp'][qb][i,:],
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                }

class MeasurementInducedDephasingAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict

        pdd['data_reshaped'] = {qb: [] for qb in pdd['data_to_fit']}
        pdd['amps_reshaped'] = np.unique(self.metadata['hard_sweep_params']['ro_amp_scale']['values'])
        pdd['phases_reshaped'] = []
        for amp in pdd['amps_reshaped']:
            mask = self.metadata['hard_sweep_params']['ro_amp_scale']['values'] == amp
            pdd['phases_reshaped'].append(self.metadata['hard_sweep_params']['phase']['values'][mask])
            for qb in self.qb_names:
                pdd['data_reshaped'][qb].append(pdd['data_to_fit'][qb][:len(mask)][mask])

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        for qb in self.qb_names:
            for i, data in enumerate(pdd['data_reshaped'][qb]):
                self.fit_dicts[f'cos_fit_{qb}_{i}'] = {
                    'model': cos_mod,
                    'guess_dict': {'frequency': {'value': 1/360,
                                                 'vary': False}},
                    'fit_xvals': {'t': pdd['phases_reshaped'][i]},
                    'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['phase_contrast'] = {}
        pdd['phase_offset'] = {}
        pdd['sigma'] = {}
        pdd['sigma_err'] = {}
        pdd['a'] = {}
        pdd['a_err'] = {}
        pdd['c'] = {}
        pdd['c_err'] = {}

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['amplitude']
                for i, _ in enumerate(pdd['data_reshaped'][qb])])
            pdd['phase_offset'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['phase']
                for i, _ in enumerate(pdd['data_reshaped'][qb])])
            pdd['phase_offset'][qb] += np.pi * (pdd['phase_contrast'][qb] < 0)
            pdd['phase_offset'][qb] = (pdd['phase_offset'][qb] + np.pi) % (2 * np.pi) - np.pi
            pdd['phase_offset'][qb] = 180*np.unwrap(pdd['phase_offset'][qb])/np.pi
            pdd['phase_contrast'][qb] = np.abs(pdd['phase_contrast'][qb])

            gauss_mod = lmfit.models.GaussianModel()
            self.fit_dicts[f'phase_contrast_fit_{qb}'] = {
                'model': gauss_mod,
                'guess_dict': {'center': {'value': 0, 'vary': False}},
                'fit_xvals': {'x': pdd['amps_reshaped']},
                'fit_yvals': {'data': pdd['phase_contrast'][qb]}}

            quadratic_mod = lmfit.models.QuadraticModel()
            self.fit_dicts[f'phase_offset_fit_{qb}'] = {
                'model': quadratic_mod,
                'guess_dict': {'b': {'value': 0, 'vary': False}},
                'fit_xvals': {'x': pdd['amps_reshaped']},
                'fit_yvals': {'data': pdd['phase_offset'][qb]}}

            self.run_fitting()
            self.save_fit_results()

            pdd['sigma'][qb] = self.fit_res[f'phase_contrast_fit_{qb}'].best_values['sigma']
            pdd['sigma_err'][qb] = self.fit_res[f'phase_contrast_fit_{qb}'].params['sigma']. \
                stderr
            pdd['a'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].best_values['a']
            pdd['a_err'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].params['a'].stderr
            pdd['c'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].best_values['c']
            pdd['c_err'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].params['c'].stderr

            pdd['sigma_err'][qb] = float('nan') if pdd['sigma_err'][qb] is None \
                else pdd['sigma_err'][qb]
            pdd['a_err'][qb] = float('nan') if pdd['a_err'][qb] is None else pdd['a_err'][qb]
            pdd['c_err'][qb] = float('nan') if pdd['c_err'][qb] is None else pdd['c_err'][qb]

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        phases_equal = True
        for phases in pdd['phases_reshaped'][1:]:
            if not np.all(phases == pdd['phases_reshaped'][0]):
                phases_equal = False
                break

        for qb in self.qb_names:
            if phases_equal:
                self.plot_dicts[f'data_2d_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'plotfn': self.plot_colorxy,
                    'xvals': pdd['phases_reshaped'][0],
                    'yvals': pdd['amps_reshaped'],
                    'zvals': pdd['data_reshaped'][qb],
                    'xlabel': r'Pulse phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'yunit': '',
                    'zlabel': 'Excited state population',
                }

            colormap = self.options_dict.get('colormap', mpl.cm.plasma)
            for i, amp in enumerate(pdd['amps_reshaped']):
                color = colormap(i/(len(pdd['amps_reshaped'])-1))
                label = f'cos_data_{qb}_{i}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'amplitude_crossections_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['phases_reshaped'][i],
                    'yvals': pdd['data_reshaped'][qb][i],
                    'xlabel': r'Pulse phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': 'Excited state population',
                    'linestyle': '',
                    'color': color,
                    'setlabel': f'amp={amp:.4f}',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }
            if self.do_fitting:
                for i, amp in enumerate(pdd['amps_reshaped']):
                    color = colormap(i/(len(pdd['amps_reshaped'])-1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'amplitude_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.options_dict.get('plot_init', False),
                        'color': color,
                        'setlabel': f'fit, amp={amp:.4f}',
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': 200*pdd['phase_contrast'][qb],
                    'xlabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'xunit': '',
                    'ylabel': 'Phase contrast',
                    'yunit': '%',
                    'linestyle': '',
                    'color': 'k',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_fit_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': 200*self.fit_res[f'phase_contrast_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_labels_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': 200*pdd['phase_contrast'][qb],
                    'marker': '',
                    'linestyle': '',
                    'setlabel': r'$\sigma = ({:.5f} \pm {:.5f})$ V'.
                        format(pdd['sigma'][qb], pdd['sigma_err'][qb]),
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }

                # Phase offset
                self.plot_dicts[f'phase_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': pdd['phase_offset'][qb],
                    'xlabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'xunit': '',
                    'ylabel': 'Phase offset',
                    'yunit': 'deg',
                    'linestyle': '',
                    'color': 'k',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_fit_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': self.fit_res[f'phase_offset_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_labels_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
                    'yvals': pdd['phase_offset'][qb],
                    'marker': '',
                    'linestyle': '',
                    'setlabel': r'$a = {:.0f} \pm {:.0f}$ deg/V${{}}^2$'.
                        format(pdd['a'][qb], pdd['a_err'][qb]) + '\n' +
                                r'$c = {:.1f} \pm {:.1f}$ deg'.
                        format(pdd['c'][qb], pdd['c_err'][qb]),
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }


class DriveCrosstalkCancellationAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()
        if self.sp is None:
            raise ValueError('This analysis needs a SweepPoints '
                             'class instance.')

        pdd = self.proc_data_dict
        # get the ramsey phases as the values of the first sweep parameter
        # in the 2nd sweep dimension.
        # !!! This assumes all qubits have the same ramsey phases !!!
        pdd['ramsey_phases'] = self.sp.get_sweep_params_property('values', 1)
        pdd['qb_sweep_points'] = {}
        pdd['qb_sweep_param'] = {}
        for k, v in self.sp.get_sweep_dimension(0).items():
            if k == 'phase':
                continue
            qb, param = k.split('.')
            pdd['qb_sweep_points'][qb] = v[0]
            pdd['qb_sweep_param'][qb] = (param, v[1], v[2])
        pdd['qb_msmt_vals'] = {}
        pdd['qb_cal_vals'] = {}

        for qb, data in pdd['data_to_fit'].items():
            pdd['qb_msmt_vals'][qb] = data[:, :-self.num_cal_points].reshape(
                len(pdd['qb_sweep_points'][qb]), len(pdd['ramsey_phases']))
            pdd['qb_cal_vals'][qb] = data[0, -self.num_cal_points:]

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        for qb in self.qb_names:
            for i, data in enumerate(pdd['qb_msmt_vals'][qb]):
                self.fit_dicts[f'cos_fit_{qb}_{i}'] = {
                    'model': cos_mod,
                    'guess_dict': {'frequency': {'value': 1/360,
                                                 'vary': False}},
                    'fit_xvals': {'t': pdd['ramsey_phases']},
                    'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['phase_contrast'] = {}
        pdd['phase_offset'] = {}

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = np.array([
                2*self.fit_res[f'cos_fit_{qb}_{i}'].best_values['amplitude']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['phase']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] *= 180/np.pi
            pdd['phase_offset'][qb] += 180 * (pdd['phase_contrast'][qb] < 0)
            pdd['phase_offset'][qb] = (pdd['phase_offset'][qb] + 180) % 360 - 180
            pdd['phase_contrast'][qb] = np.abs(pdd['phase_contrast'][qb])

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            self.plot_dicts[f'data_2d_{qb}'] = {
                'title': rdd['measurementstring'] +
                         '\n' + rdd['timestamp'] + '\n' + qb,
                'plotfn': self.plot_colorxy,
                'xvals': pdd['ramsey_phases'],
                'yvals': pdd['qb_sweep_points'][qb],
                'zvals': pdd['qb_msmt_vals'][qb],
                'xlabel': r'Ramsey phase, $\phi$',
                'xunit': 'deg',
                'ylabel': pdd['qb_sweep_param'][qb][2],
                'yunit': pdd['qb_sweep_param'][qb][1],
                'zlabel': 'Excited state population',
            }

            colormap = self.options_dict.get('colormap', mpl.cm.plasma)
            for i, pval in enumerate(pdd['qb_sweep_points'][qb]):
                if i == len(pdd['qb_sweep_points'][qb]) - 1:
                    legendlabel='data, ref.'
                else:
                    legendlabel = f'data, {pdd["qb_sweep_param"][qb][0]}='\
                                  f'{pval:.4f}{pdd["qb_sweep_param"][qb][1]}'
                color = colormap(i/(len(pdd['qb_sweep_points'][qb])-1))
                label = f'cos_data_{qb}_{i}'

                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'param_crossections_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['ramsey_phases'],
                    'yvals': pdd['qb_msmt_vals'][qb][i],
                    'xlabel': r'Ramsey phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': 'Excited state population',
                    'linestyle': '',
                    'color': color,
                    'setlabel': legendlabel,
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }
            if self.do_fitting:
                for i, pval in enumerate(pdd['qb_sweep_points'][qb]):
                    if i == len(pdd['qb_sweep_points'][qb]) - 1:
                        legendlabel = 'fit, ref.'
                    else:
                        legendlabel = f'fit, {pdd["qb_sweep_param"][qb][0]}='\
                                      f'{pval:.4f}{pdd["qb_sweep_param"][qb][1]}'
                    color = colormap(i/(len(pdd['qb_sweep_points'][qb])-1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'param_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.options_dict.get('plot_init', False),
                        'color': color,
                        'do_legend': False,
                        # 'setlabel': legendlabel
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['qb_sweep_points'][qb][:-1],
                    'yvals': pdd['phase_contrast'][qb][:-1] * 100,
                    'xlabel': pdd['qb_sweep_param'][qb][2],
                    'xunit': pdd['qb_sweep_param'][qb][1],
                    'ylabel': 'Phase contrast',
                    'yunit': '%',
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'C0',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_ref_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_hlines,
                    'xmin': pdd['qb_sweep_points'][qb][:-1].min(),
                    'xmax': pdd['qb_sweep_points'][qb][:-1].max(),
                    'y': pdd['phase_contrast'][qb][-1] * 100,
                    'linestyle': '--',
                    'colors': '0.6',
                    'setlabel': 'ref',
                    'do_legend': True,
                }

                # Phase offset
                self.plot_dicts[f'phase_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['qb_sweep_points'][qb][:-1],
                    'yvals': pdd['phase_offset'][qb][:-1],
                    'xlabel': pdd['qb_sweep_param'][qb][2],
                    'xunit': pdd['qb_sweep_param'][qb][1],
                    'ylabel': 'Phase offset',
                    'yunit': 'deg',
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'C0',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_ref_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_hlines,
                    'xmin': pdd['qb_sweep_points'][qb][:-1].min(),
                    'xmax': pdd['qb_sweep_points'][qb][:-1].max(),
                    'y': pdd['phase_offset'][qb][-1],
                    'linestyle': '--',
                    'colors': '0.6',
                    'setlabel': 'ref',
                    'do_legend': True,
                }


class RabiAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.'+qbn
            for trans_name in ['ge', 'ef']:
                params_dict[f'{trans_name}_amp180_'+qbn] = \
                    s+f'.{trans_name}_amp180'
                params_dict[f'{trans_name}_amp90scale_'+qbn] = \
                    s+f'.{trans_name}_amp90_scale'
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            cos_mod = lmfit.Model(fit_mods.CosFunc)
            guess_pars = fit_mods.Cos_guess(
                model=cos_mod, t=sweep_points, data=data)
            guess_pars['amplitude'].vary = True
            guess_pars['amplitude'].min = -10
            guess_pars['offset'].vary = True
            guess_pars['frequency'].vary = True
            guess_pars['phase'].vary = True

            key = 'cos_fit_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': fit_mods.CosFunc,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            fit_res = self.fit_dicts['cos_fit_' + qbn]['fit_res']
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            self.proc_data_dict['analysis_params_dict'][qbn] = \
                self.get_amplitudes(fit_res=fit_res, sweep_points=sweep_points)
        self.save_processed_data(key='analysis_params_dict')

    def get_amplitudes(self, fit_res, sweep_points):
        # Extract the best fitted frequency and phase.
        freq_fit = fit_res.best_values['frequency']
        phase_fit = fit_res.best_values['phase']

        freq_std = fit_res.params['frequency'].stderr
        phase_std = fit_res.params['phase'].stderr

        # If fitted_phase<0, shift fitted_phase by 4. This corresponds to a
        # shift of 2pi in the argument of cos.
        if np.abs(phase_fit) < 0.1:
            phase_fit = 0

        # If phase_fit<1, the piHalf amplitude<0.
        if phase_fit < 1:
            log.info('The data could not be fitted correctly. '
                         'The fitted phase "%s" <1, which gives '
                         'negative piHalf '
                         'amplitude.' % phase_fit)

        stepsize = sweep_points[1] - sweep_points[0]
        if freq_fit > 2 * stepsize:
            log.info('The data could not be fitted correctly. The '
                         'frequency "%s" is too high.' % freq_fit)
        n = np.arange(-2, 10)

        piPulse_vals = (n*np.pi - phase_fit)/(2*np.pi*freq_fit)
        piHalfPulse_vals = (n*np.pi + np.pi/2 - phase_fit)/(2*np.pi*freq_fit)

        # find piHalfPulse
        try:
            piHalfPulse = \
                np.min(piHalfPulse_vals[piHalfPulse_vals >= sweep_points[1]])
            n_piHalf_pulse = n[piHalfPulse_vals==piHalfPulse]
        except ValueError:
            piHalfPulse = np.asarray([])

        if piHalfPulse.size == 0 or piHalfPulse > max(sweep_points):
            i = 0
            while (piHalfPulse_vals[i] < min(sweep_points) and
                   i<piHalfPulse_vals.size):
                i+=1
            piHalfPulse = piHalfPulse_vals[i]
            n_piHalf_pulse = n[i]

        # find piPulse
        try:
            if piHalfPulse.size != 0:
                piPulse = \
                    np.min(piPulse_vals[piPulse_vals >= piHalfPulse])
            else:
                piPulse = np.min(piPulse_vals[piPulse_vals >= 0.001])
            n_pi_pulse = n[piHalfPulse_vals == piHalfPulse]

        except ValueError:
            piPulse = np.asarray([])

        if piPulse.size == 0:
            i = 0
            while (piPulse_vals[i] < min(sweep_points) and
                   i < piPulse_vals.size):
                i += 1
            piPulse = piPulse_vals[i]
            n_pi_pulse = n[i]

        try:
            freq_idx = fit_res.var_names.index('frequency')
            phase_idx = fit_res.var_names.index('phase')
            if fit_res.covar is not None:
                cov_freq_phase = fit_res.covar[freq_idx, phase_idx]
            else:
                cov_freq_phase = 0
        except ValueError:
            cov_freq_phase = 0

        try:
            piPulse_std = self.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_num=n_pi_pulse,
                cov=cov_freq_phase)
            piHalfPulse_std = self.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_num=n_piHalf_pulse,
                cov=cov_freq_phase)
        except Exception as e:
            log.error(e)
            piPulse_std = 0
            piHalfPulse_std = 0

        rabi_amplitudes = {'piPulse': piPulse,
                           'piPulse_stderr': piPulse_std,
                           'piHalfPulse': piHalfPulse,
                           'piHalfPulse_stderr': piHalfPulse_std}

        return rabi_amplitudes

    def calculate_pulse_stderr(self, f, phi, f_err, phi_err,
                               period_num, cov=0):
        x = period_num + phi
        return np.sqrt((f_err*x/(2*np.pi*(f**2)))**2 +
                       (phi_err/(2*np.pi*f))**2 -
                       2*(cov**2)*x/((2*np.pi*(f**3))**2))[0]

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                base_plot_name = 'Rabi_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                fit_res = self.fit_dicts['cos_fit_' + qbn]['fit_res']
                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'cosine fit',
                    'color': 'r',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                rabi_amplitudes = self.proc_data_dict['analysis_params_dict']
                self.plot_dicts['piamp_marker_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([rabi_amplitudes[qbn]['piPulse']]),
                    'yvals': np.array([fit_res.model.func(
                        rabi_amplitudes[qbn]['piPulse'],
                        **fit_res.best_values)]),
                    'setlabel': '$\pi$-Pulse amp',
                    'color': 'r',
                    'marker': 'o',
                    'line_kws': {'markersize': 10},
                    'linestyle': '',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                self.plot_dicts['piamp_hline_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': [fit_res.model.func(
                        rabi_amplitudes[qbn]['piPulse'],
                        **fit_res.best_values)],
                    'xmin': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-1],
                    'colors': 'gray'}

                self.plot_dicts['pihalfamp_marker_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([rabi_amplitudes[qbn]['piHalfPulse']]),
                    'yvals': np.array([fit_res.model.func(
                        rabi_amplitudes[qbn]['piHalfPulse'],
                        **fit_res.best_values)]),
                    'setlabel': '$\pi /2$-Pulse amp',
                    'color': 'm',
                    'marker': 'o',
                    'line_kws': {'markersize': 10},
                    'linestyle': '',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                self.plot_dicts['pihalfamp_hline_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': [fit_res.model.func(
                        rabi_amplitudes[qbn]['piHalfPulse'],
                        **fit_res.best_values)],
                    'xmin': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-1],
                    'colors': 'gray'}

                trans_name = 'ef' if 'f' in self.data_to_fit[qbn] else 'ge'
                old_pipulse_val = self.raw_data_dict[
                    f'{trans_name}_amp180_'+qbn]
                if old_pipulse_val != old_pipulse_val:
                    old_pipulse_val = 0
                old_pihalfpulse_val = self.raw_data_dict[
                    f'{trans_name}_amp90scale_'+qbn]
                if old_pihalfpulse_val != old_pihalfpulse_val:
                    old_pihalfpulse_val = 0
                old_pihalfpulse_val *= old_pipulse_val

                textstr = ('  $\pi-Amp$ = {:.3f} V'.format(
                    rabi_amplitudes[qbn]['piPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                    rabi_amplitudes[qbn]['piPulse_stderr']) +
                           '\n$\pi/2-Amp$ = {:.3f} V '.format(
                    rabi_amplitudes[qbn]['piHalfPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                    rabi_amplitudes[qbn]['piHalfPulse_stderr']) +
                           '\n  $\pi-Amp_{old}$ = ' + '{:.3f} V '.format(
                    old_pipulse_val) +
                           '\n$\pi/2-Amp_{old}$ = ' + '{:.3f} V '.format(
                    old_pihalfpulse_val))
                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class T1Analysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.'+qbn
            for trans_name in ['ge', 'ef']:
                params_dict[f'{trans_name}_T1_'+qbn] = s+'.T1{}'.format(
                    '_ef' if trans_name == 'ef' else '')
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            exp_decay_mod = lmfit.Model(fit_mods.ExpDecayFunc)
            guess_pars = fit_mods.exp_dec_guess(
                model=exp_decay_mod, data=data, t=sweep_points)
            guess_pars['amplitude'].vary = True
            guess_pars['tau'].vary = True
            if self.options_dict.get('vary_offset', False):
                guess_pars['offset'].vary = True
            else:
                guess_pars['offset'].value = 0
                guess_pars['offset'].vary = False
            key = 'exp_decay_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': exp_decay_mod.func,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn]['T1'] = \
                self.fit_dicts['exp_decay_' + qbn]['fit_res'].best_values['tau']
            self.proc_data_dict['analysis_params_dict'][qbn]['T1_stderr'] = \
                self.fit_dicts['exp_decay_' + qbn]['fit_res'].params[
                    'tau'].stderr
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                # rename base plot
                base_plot_name = 'T1_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['exp_decay_' + qbn]['fit_res'],
                    'setlabel': 'exp decay fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                trans_name = 'ef' if 'f' in self.data_to_fit[qbn] else 'ge'
                old_T1_val = self.raw_data_dict[f'{trans_name}_T1_'+qbn]
                if old_T1_val != old_T1_val:
                    old_T1_val = 0
                T1_dict = self.proc_data_dict['analysis_params_dict']
                textstr = '$T_1$ = {:.2f} $\mu$s'.format(
                            T1_dict[qbn]['T1']*1e6) \
                          + ' $\pm$ {:.2f} $\mu$s'.format(
                            T1_dict[qbn]['T1_stderr']*1e6) \
                          + '\nold $T_1$ = {:.2f} $\mu$s'.format(old_T1_val*1e6)
                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class RamseyAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.'+qbn
            for trans_name in ['ge', 'ef']:
                params_dict[f'{trans_name}_freq_'+qbn] = s+f'.{trans_name}_freq'
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def prepare_fitting(self):
        if self.options_dict.get('fit_gaussian_decay', True):
            self.fit_keys = ['exp_decay_', 'gauss_decay_']
        else:
            self.fit_keys = ['exp_decay_']
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            for i, key in enumerate([k + qbn for k in self.fit_keys]):
                exp_damped_decay_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
                guess_pars = fit_mods.exp_damp_osc_guess(
                    model=exp_damped_decay_mod, data=data, t=sweep_points,
                    n_guess=i+1)
                guess_pars['amplitude'].vary = False
                guess_pars['amplitude'].value = 0.5
                guess_pars['frequency'].vary = True
                guess_pars['tau'].vary = True
                guess_pars['phase'].vary = True
                guess_pars['n'].vary = False
                guess_pars['oscillation_offset'].vary = \
                        'f' in self.data_to_fit[qbn]
                # guess_pars['exponential_offset'].value = 0.5
                guess_pars['exponential_offset'].vary = True
                self.fit_dicts[key] = {
                    'fit_fn': exp_damped_decay_mod .func,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        if 'artificial_detuning' in self.options_dict:
            artificial_detuning_dict = OrderedDict(
                [(qbn, self.options_dict['artificial_detuning'])
             for qbn in self.qb_names])
        elif 'artificial_detuning_dict' in self.metadata:
            artificial_detuning_dict = self.metadata[
                'artificial_detuning_dict']
        elif 'artificial_detuning' in self.metadata:
            artificial_detuning_dict = OrderedDict(
                [(qbn, self.metadata['artificial_detuning'])
                 for qbn in self.qb_names])
        else:
            raise ValueError('"artificial_detuning" not found.')

        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            for key in [k + qbn for k in self.fit_keys]:
                self.proc_data_dict['analysis_params_dict'][qbn][key] = \
                    OrderedDict()
                fit_res = self.fit_dicts[key]['fit_res']
                for par in fit_res.params:
                    if fit_res.params[par].stderr is None:
                        fit_res.params[par].stderr = 0

                trans_name = 'ef' if 'f' in self.data_to_fit[qbn] else 'ge'
                old_qb_freq = self.raw_data_dict[f'{trans_name}_freq_'+qbn]
                if old_qb_freq != old_qb_freq:
                    old_qb_freq = 0
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'old_qb_freq'] = old_qb_freq
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'new_qb_freq'] = old_qb_freq + \
                                     artificial_detuning_dict[qbn] - \
                                     fit_res.best_values['frequency']
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'new_qb_freq_stderr'] = fit_res.params['frequency'].stderr
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'T2_star'] = fit_res.best_values['tau']
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'T2_star_stderr'] = fit_res.params['tau'].stderr
                self.proc_data_dict['analysis_params_dict'][qbn][key][
                    'artificial_detuning'] = artificial_detuning_dict[qbn]
        hdf_group_name_suffix = self.options_dict.get(
            'hdf_group_name_suffix', '')
        self.save_processed_data(key='analysis_params_dict' +
                                     hdf_group_name_suffix)

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            ramsey_dict = self.proc_data_dict['analysis_params_dict']
            for qbn in self.qb_names:
                base_plot_name = 'Ramsey_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                exp_decay_fit_key = self.fit_keys[0] + qbn
                old_qb_freq = ramsey_dict[qbn][
                    exp_decay_fit_key]['old_qb_freq']
                textstr = ''
                T2_star_str = ''

                for i, key in enumerate([k + qbn for k in self.fit_keys]):

                    fit_res = self.fit_dicts[key]['fit_res']
                    self.plot_dicts['fit_' + key] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res,
                        'setlabel': 'exp decay fit' if i == 0 else
                            'gauss decay fit',
                        'do_legend': True,
                        'color': 'r' if i == 0 else 'C4',
                        'legend_bbox_to_anchor': (1, -0.15),
                        'legend_pos': 'upper right'}

                    if i != 0:
                        textstr += '\n'
                    textstr += \
                        ('$f_{{qubit \_ new \_ {{{key}}} }}$ = '.format(
                            key=('exp' if i == 0 else 'gauss')) +
                            '{:.6f} GHz '.format(
                            ramsey_dict[qbn][key]['new_qb_freq']*1e-9) +
                            '$\pm$ {:.2E} GHz '.format(
                            ramsey_dict[qbn][key][
                                'new_qb_freq_stderr']*1e-9))
                    T2_star_str += \
                        ('\n$T_{{2,{{{key}}} }}^\star$ = '.format(
                            key=('exp' if i == 0 else 'gauss')) +
                            '{:.2f} $\mu$s'.format(
                            fit_res.params['tau'].value*1e6) +
                            '$\pm$ {:.2f} $\mu$s'.format(
                            fit_res.params['tau'].stderr*1e6))

                textstr += '\n$f_{qubit \_ old}$ = '+'{:.6f} GHz '.format(
                    old_qb_freq*1e-9)
                textstr += ('\n$\Delta f$ = {:.4f} MHz '.format(
                    (ramsey_dict[qbn][exp_decay_fit_key]['new_qb_freq'] -
                    old_qb_freq)*1e-6) + '$\pm$ {:.2E} MHz'.format(
                    self.fit_dicts[exp_decay_fit_key]['fit_res'].params[
                        'frequency'].stderr*1e-6) +
                    '\n$f_{Ramsey}$ = '+'{:.4f} MHz $\pm$ {:.2E} MHz'.format(
                    self.fit_dicts[exp_decay_fit_key]['fit_res'].params[
                        'frequency'].value*1e-6,
                    self.fit_dicts[exp_decay_fit_key]['fit_res'].params[
                        'frequency'].stderr*1e-6))
                textstr += T2_star_str
                textstr += '\nartificial detuning = {:.2f} MHz'.format(
                    ramsey_dict[qbn][exp_decay_fit_key][
                        'artificial_detuning']*1e-6)

                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': -0.025,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}

                self.plot_dicts['half_hline_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': 0.5,
                    'xmin': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-1],
                    'colors': 'gray'}


class QScaleAnalysis(MultiQubit_TimeDomain_Analysis):
    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.'+qbn
            for trans_name in ['ge', 'ef']:
                params_dict[f'{trans_name}_qscale_'+qbn] = \
                    s+f'.{trans_name}_motzoi'
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def process_data(self):
        super().process_data()

        self.proc_data_dict['qscale_data'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['qscale_data'][qbn] = OrderedDict()
            sweep_points = deepcopy(self.proc_data_dict['sweep_points_dict'][
                                        qbn]['msmt_sweep_points'])
            # check if the sweep points are repeated 3 times as they have to be
            # for the qscale analysis:
            # Takes the first 3 entries and check if they are all the same or different.
            # Needed For backwards compatibility with QudevTransmon.measure_qscale()
            # that does not (yet) use Sweeppoints object.

            unique_sp = np.unique(sweep_points[:3])
            if unique_sp.size > 1:
                sweep_points = np.repeat(sweep_points, 3)
            # replace in proc_data_dict; otherwise plotting in base class fails
            self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points'] = sweep_points
            self.proc_data_dict['sweep_points_dict'][qbn][
                'sweep_points'] = np.concatenate([
                sweep_points, self.proc_data_dict['sweep_points_dict'][qbn][
                    'cal_points_sweep_points']])

            data = self.proc_data_dict['data_to_fit'][qbn]
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xx'] = \
                sweep_points[0::3]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xy'] = \
                sweep_points[1::3]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xmy'] = \
                sweep_points[2::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xx'] = \
                data[0::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xy'] = \
                data[1::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xmy'] = \
                data[2::3]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        for qbn in self.qb_names:
            for msmt_label in ['_xx', '_xy', '_xmy']:
                sweep_points = self.proc_data_dict['qscale_data'][qbn][
                    'sweep_points' + msmt_label]
                data = self.proc_data_dict['qscale_data'][qbn][
                    'data' + msmt_label]

                # As a workaround for a weird bug letting crash the analysis
                # every second time, we do not use lmfit.models.ConstantModel
                # and lmfit.models.LinearModel, but create custom models.
                if msmt_label == '_xx':
                    model = lmfit.Model(lambda x, c: c)
                    guess_pars = model.make_params(c=np.mean(data))
                else:
                    model = lmfit.Model(lambda x, slope, intercept:
                                        slope * x + intercept)
                    slope = (data[-1] - data[0]) / \
                            (sweep_points[-1] - sweep_points[0])
                    intercept = data[-1] - slope * sweep_points[-1]
                    guess_pars = model.make_params(slope=slope,
                                                   intercept=intercept)

                key = 'fit' + msmt_label + '_' + qbn
                self.fit_dicts[key] = {
                    'fit_fn': model.func,
                    'fit_xvals': {'x': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        # The best qscale parameter is the point where all 3 curves intersect.
        threshold = 0.02
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            fitparams0 = self.fit_dicts['fit_xx'+'_'+qbn]['fit_res'].params
            fitparams1 = self.fit_dicts['fit_xy'+'_'+qbn]['fit_res'].params
            fitparams2 = self.fit_dicts['fit_xmy'+'_'+qbn]['fit_res'].params

            intercept_diff_mean = fitparams1['intercept'].value - \
                                  fitparams2['intercept'].value
            slope_diff_mean = fitparams2['slope'].value - \
                              fitparams1['slope'].value
            optimal_qscale = intercept_diff_mean/slope_diff_mean

            # Warning if Xpi/2Xpi line is not within +/-threshold of 0.5
            if (fitparams0['c'].value > (0.5 + threshold)) or \
                    (fitparams0['c'].value < (0.5 - threshold)):
                log.warning('The trace from the X90-X180 pulses is '
                                'NOT within $\pm${} of the expected value '
                                'of 0.5.'.format(threshold))
            # Warning if optimal_qscale is not within +/-threshold of 0.5
            y_optimal_qscale = optimal_qscale * fitparams2['slope'].value + \
                                 fitparams2['intercept'].value
            if (y_optimal_qscale > (0.5 + threshold)) or \
                    (y_optimal_qscale < (0.5 - threshold)):
                log.warning('The optimal qscale found gives a population '
                                'that is NOT within $\pm${} of the expected '
                                'value of 0.5.'.format(threshold))

            # Calculate standard deviation
            intercept_diff_std_squared = \
                fitparams1['intercept'].stderr**2 + \
                fitparams2['intercept'].stderr**2
            slope_diff_std_squared = \
                fitparams2['slope'].stderr**2 + fitparams1['slope'].stderr**2

            optimal_qscale_stderr = np.sqrt(
                intercept_diff_std_squared*(1/slope_diff_mean**2) +
                slope_diff_std_squared*(intercept_diff_mean /
                                        (slope_diff_mean**2))**2)

            self.proc_data_dict['analysis_params_dict'][qbn]['qscale'] = \
                optimal_qscale
            self.proc_data_dict['analysis_params_dict'][qbn][
                'qscale_stderr'] = optimal_qscale_stderr

    def prepare_plots(self):
        super().prepare_plots()

        color_dict = {'_xx': '#365C91',
                      '_xy': '#683050',
                      '_xmy': '#3C7541'}
        label_dict = {'_xx': r'$X_{\pi/2}X_{\pi}$',
                      '_xy': r'$X_{\pi/2}Y_{\pi}$',
                      '_xmy': r'$X_{\pi/2}Y_{-\pi}$'}
        for qbn in self.qb_names:
            base_plot_name = 'Qscale_' + qbn
            for msmt_label in ['_xx', '_xy', '_xmy']:
                sweep_points = self.proc_data_dict['qscale_data'][qbn][
                    'sweep_points' + msmt_label]
                data = self.proc_data_dict['qscale_data'][qbn][
                    'data' + msmt_label]
                if msmt_label == '_xx':
                    plot_name = base_plot_name
                else:
                    plot_name = 'data' + msmt_label + '_' + qbn
                xlabel, xunit = self.get_xaxis_label_unit(qbn)
                self.plot_dicts[plot_name] = {
                    'plotfn': self.plot_line,
                    'xvals': sweep_points,
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': data,
                    'ylabel': '{} state population'.format(
                        self.get_latex_prob_label(self.data_to_fit[qbn])),
                    'yunit': '',
                    'setlabel': 'Data\n' + label_dict[msmt_label],
                    'title': (self.raw_data_dict['timestamp'] + ' ' +
                              self.raw_data_dict['measurementstring'] +
                              '\n' + qbn),
                    'linestyle': 'none',
                    'color': color_dict[msmt_label],
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}
                if msmt_label != '_xx':
                    self.plot_dicts[plot_name]['fig_id'] = base_plot_name

                if self.do_fitting:
                    # plot fit
                    xfine = np.linspace(sweep_points[0], sweep_points[-1], 1000)
                    fit_key = 'fit' + msmt_label + '_' + qbn
                    fit_res = self.fit_dicts[fit_key]['fit_res']
                    yvals = fit_res.model.func(xfine, **fit_res.best_values)
                    if not hasattr(yvals, '__iter__'):
                        yvals = np.array(len(xfine)*[yvals])
                    self.plot_dicts[fit_key] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_line,
                        'xvals': xfine,
                        'yvals': yvals,
                        'marker': '',
                        'setlabel': 'Fit\n' + label_dict[msmt_label],
                        'do_legend': True,
                        'color': color_dict[msmt_label],
                        'legend_bbox_to_anchor': (1, 0.5),
                        'legend_pos': 'center left'}

                    trans_name = 'ef' if 'f' in self.data_to_fit[qbn] else 'ge'
                    old_qscale_val = self.raw_data_dict[
                        f'{trans_name}_qscale_'+qbn]
                    if old_qscale_val != old_qscale_val:
                        old_qscale_val = 0
                    textstr = 'Qscale = {:.4f} $\pm$ {:.4f}'.format(
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'qscale'],
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'qscale_stderr']) + \
                            '\nold Qscale= {:.4f}'.format(old_qscale_val)

                    self.plot_dicts['text_msg_' + qbn] = {
                        'fig_id': base_plot_name,
                        'ypos': -0.175,
                        'xpos': 0.5,
                        'horizontalalignment': 'center',
                        'verticalalignment': 'top',
                        'plotfn': self.plot_text,
                        'text_string': textstr}

            # plot cal points
            if self.num_cal_points != 0:
                for i, cal_pts_idxs in enumerate(
                        self.cal_states_dict.values()):
                    plot_dict_name = list(self.cal_states_dict)[i] + \
                                     '_' + qbn
                    self.plot_dicts[plot_dict_name] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_line,
                        'xvals': np.mean([
                            self.proc_data_dict['sweep_points_dict'][qbn]
                            ['cal_points_sweep_points'][cal_pts_idxs],
                            self.proc_data_dict['sweep_points_dict'][qbn]
                            ['cal_points_sweep_points'][cal_pts_idxs]],
                            axis=0),
                        'yvals': self.proc_data_dict[
                            'data_to_fit'][qbn][cal_pts_idxs],
                        'setlabel': list(self.cal_states_dict)[i],
                        'do_legend': True,
                        'legend_bbox_to_anchor': (1, 0.5),
                        'legend_pos': 'center left',
                        'linestyle': 'none',
                        'line_kws': {'color': self.get_cal_state_color(
                            list(self.cal_states_dict)[i])}}

                    self.plot_dicts[plot_dict_name + '_line'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_hlines,
                        'y': np.mean(
                            self.proc_data_dict[
                                'data_to_fit'][qbn][cal_pts_idxs]),
                        'xmin': self.proc_data_dict['sweep_points_dict'][
                            qbn]['sweep_points'][0],
                        'xmax': self.proc_data_dict['sweep_points_dict'][
                            qbn]['sweep_points'][-1],
                        'colors': 'gray'}


class EchoAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        auto = kwargs.pop('auto', True)
        super().__init__(*args, auto=False, **kwargs)
        if self.options_dict.get('artificial_detuning', None) is not None:
            self.echo_analysis = RamseyAnalysis(*args, auto=False, **kwargs)
        else:
            if 'options_dict' in kwargs:
                # kwargs.pop('options_dict')
                kwargs['options_dict'].update({'vary_offset': True})
            else:
                kwargs['options_dict'] = {'vary_offset': True}
            self.echo_analysis = T1Analysis(*args, auto=False, **kwargs)

        if auto:
            self.echo_analysis.extract_data()
            self.echo_analysis.process_data()
            self.echo_analysis.prepare_fitting()
            self.echo_analysis.run_fitting()
            self.echo_analysis.save_fit_results()
            self.analyze_fit_results()
            self.prepare_plots()

    def analyze_fit_results(self):
        self.echo_analysis.analyze_fit_results()
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()

            params_dict = self.echo_analysis.proc_data_dict[
                'analysis_params_dict'][qbn]
            if 'T1' in params_dict:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo'] = params_dict['T1']
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo_stderr'] = params_dict['T1_stderr']
            else:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo'] = params_dict['exp_decay_'+qbn][
                    'T2_star']
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo_stderr'] = params_dict['exp_decay_'+qbn][
                    'T2_star_stderr']

    def prepare_plots(self):
        self.echo_analysis.prepare_plots()
        for qbn in self.qb_names:
            # rename base plot
            figure_name = 'Echo_' + qbn
            echo_plot_key_t1 = [key for key in self.echo_analysis.plot_dicts if
                                'T1_'+qbn in key]
            echo_plot_key_ram = [key for key in self.echo_analysis.plot_dicts if
                                 'Ramsey_'+qbn in key]
            if len(echo_plot_key_t1) != 0:
                echo_plot_name = echo_plot_key_t1[0]
            elif len(echo_plot_key_ram) != 0:
                echo_plot_name = echo_plot_key_ram[0]
            else:
                raise ValueError('Neither T1 nor Ramsey plots were found.')

            self.echo_analysis.plot_dicts[echo_plot_name][
                'legend_pos'] = 'upper right'
            self.echo_analysis.plot_dicts[echo_plot_name][
                'legend_bbox_to_anchor'] = (1, -0.15)

            for plot_label in self.echo_analysis.plot_dicts:
                if qbn in plot_label:
                    if 'raw' not in plot_label and 'projected' not in plot_label:
                        self.echo_analysis.plot_dicts[plot_label]['fig_id'] = \
                            figure_name

            old_T2e_val = a_tools.get_instr_setting_value_from_file(
                file_path=self.echo_analysis.raw_data_dict['folder'],
                instr_name=qbn, param_name='T2{}'.format(
                    '_ef' if 'f' in self.echo_analysis.data_to_fit[qbn]
                    else ''))
            T2_dict = self.proc_data_dict['analysis_params_dict']
            textstr = '$T_2$ echo = {:.2f} $\mu$s'.format(
                T2_dict[qbn]['T2_echo']*1e6) \
                      + ' $\pm$ {:.2f} $\mu$s'.format(
                T2_dict[qbn]['T2_echo_stderr']*1e6) \
                      + '\nold $T_2$ echo = {:.2f} $\mu$s'.format(
                old_T2e_val*1e6)

            self.echo_analysis.plot_dicts['text_msg_' + qbn][
                'text_string'] = textstr

        self.echo_analysis.plot(key_list='auto')
        self.echo_analysis.save_figures(close_figs=True)


class RamseyAddPulseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        auto = kwargs.pop('auto', True)
        super().__init__(*args, auto=False, **kwargs)
        options_dict = kwargs.pop('options_dict', OrderedDict())
        options_dict_no = deepcopy(options_dict)
        options_dict_no.update(dict(
            data_filter=lambda raw: np.concatenate([
                raw[:-4][1::2], raw[-4:]]),
            hdf_group_name_suffix='_no_pulse'))
        self.ramsey_analysis = RamseyAnalysis(
            *args, auto=False, options_dict=options_dict_no,
            **kwargs)
        options_dict_with = deepcopy(options_dict)
        options_dict_with.update(dict(
            data_filter=lambda raw: np.concatenate([
                raw[:-4][0::2], raw[-4:]]),
            hdf_group_name_suffix='_with_pulse'))
        self.ramsey_add_pulse_analysis = RamseyAnalysis(
            *args, auto=False, options_dict=options_dict_with,
            **kwargs)


        if auto:
            self.ramsey_analysis.extract_data()
            self.ramsey_analysis.process_data()
            self.ramsey_analysis.prepare_fitting()
            self.ramsey_analysis.run_fitting()
            self.ramsey_analysis.save_fit_results()
            self.ramsey_add_pulse_analysis.extract_data()
            self.ramsey_add_pulse_analysis.process_data()
            self.ramsey_add_pulse_analysis.prepare_fitting()
            self.ramsey_add_pulse_analysis.run_fitting()
            self.ramsey_add_pulse_analysis.save_fit_results()
            self.raw_data_dict = self.ramsey_analysis.raw_data_dict
            self.analyze_fit_results()
            self.prepare_plots()
            keylist = []
            for qbn in self.qb_names:
                figure_name = 'CrossZZ_' + qbn
                keylist.append(figure_name+'with')
                keylist.append(figure_name+'no')
            self.plot()
            self.save_figures(close_figs=True)

    def analyze_fit_results(self):
        self.cross_kerr = 0.0
        self.ramsey_analysis.analyze_fit_results()
        self.ramsey_add_pulse_analysis.analyze_fit_results()

        self.proc_data_dict['analysis_params_dict'] = OrderedDict()


        for qbn in self.qb_names:

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()

            self.params_dict_ramsey = self.ramsey_analysis.proc_data_dict[
                'analysis_params_dict'][qbn]
            self.params_dict_add_pulse = \
                self.ramsey_add_pulse_analysis.proc_data_dict[
                    'analysis_params_dict'][qbn]
            self.cross_kerr = self.params_dict_ramsey[
                                  'exp_decay_'+str(qbn)]['new_qb_freq'] \
                            - self.params_dict_add_pulse[
                                  'exp_decay_'+str(qbn)]['new_qb_freq']
            self.cross_kerr_error = np.sqrt(
                (self.params_dict_ramsey[
                    'exp_decay_'+str(qbn)]['new_qb_freq_stderr'])**2 +
                (self.params_dict_add_pulse[
                    'exp_decay_' + str(qbn)]['new_qb_freq_stderr'])**2)

    def prepare_plots(self):
        self.ramsey_analysis.prepare_plots()
        self.ramsey_add_pulse_analysis.prepare_plots()

        self.ramsey_analysis.plot(key_list='auto')
        self.ramsey_analysis.save_figures(close_figs=True, savebase='Ramsey_no')

        self.ramsey_add_pulse_analysis.plot(key_list='auto')
        self.ramsey_add_pulse_analysis.save_figures(close_figs=True,
                                                    savebase='Ramsey_with')

        self.options_dict['plot_proj_data'] = False
        self.metadata = {'plot_proj_data': False, 'plot_raw_data': False}
        super().prepare_plots()

        try:
            xunit = self.metadata["sweep_unit"]
            xlabel = self.metadata["sweep_name"]
        except KeyError:
            xlabel = self.raw_data_dict['sweep_parameter_names'][0]
            xunit = self.raw_data_dict['sweep_parameter_units'][0]
        if np.ndim(xunit) > 0:
            xunit = xunit[0]
        title = (self.raw_data_dict['timestamp'] + ' ' +
                 self.raw_data_dict['measurementstring'])

        for qbn in self.qb_names:
            data_no = self.ramsey_analysis.proc_data_dict['data_to_fit'][
                          qbn][:-self.ramsey_analysis.num_cal_points]
            data_with = self.ramsey_add_pulse_analysis.proc_data_dict[
                            'data_to_fit'][
                            qbn][:-self.ramsey_analysis.num_cal_points]
            delays = self.ramsey_analysis.proc_data_dict['sweep_points_dict'][
                         qbn]['sweep_points'][
                     :-self.ramsey_analysis.num_cal_points]

            figure_name = 'CrossZZ_' + qbn
            self.plot_dicts[figure_name+'with'] = {
                'fig_id': figure_name,
                'plotfn': self.plot_line,
                'xvals': delays,
                'yvals': data_with,
                'xlabel': xlabel,
                'xunit': xunit,
                'ylabel': '|e> state population',
                'setlabel': 'with $\\pi$-pulse',
                'title': title,
                'color': 'r',
                'marker': 'o',
                'line_kws': {'markersize': 5},
                'linestyle': 'none',
                'do_legend': True,
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1, -0.15),
                'legend_pos': 'upper right'}

            if self.do_fitting:
                fit_res_with = self.ramsey_add_pulse_analysis.fit_dicts[
                    'exp_decay_' + qbn]['fit_res']
                self.plot_dicts['fit_with_'+qbn] = {
                    'fig_id': figure_name,
                    'plotfn': self.plot_fit,
                    'xlabel': 'Ramsey delay',
                    'xunit': 's',
                    'fit_res': fit_res_with,
                    'setlabel': 'with $\\pi$-pulse - fit',
                    'title': title,
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

            self.plot_dicts[figure_name+'no'] = {
                'fig_id': figure_name,
                'plotfn': self.plot_line,
                'xvals': delays,
                'yvals': data_no,
                'setlabel': 'no $\\pi$-pulse',
                'title': title,
                'color': 'g',
                'marker': 'o',
                'line_kws': {'markersize': 5},
                'linestyle': 'none',
                'do_legend': True,
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1, -0.15),
                'legend_pos': 'upper right'}

            if self.do_fitting:
                fit_res_no = self.ramsey_analysis.fit_dicts[
                    'exp_decay_' + qbn]['fit_res']
                self.plot_dicts['fit_no_'+qbn] = {
                    'fig_id': figure_name,
                    'plotfn': self.plot_fit,
                    'xlabel': 'Ramsey delay',
                    'xunit': 's',
                    'fit_res': fit_res_no,
                    'setlabel': 'no $\\pi$-pulse - fit',
                    'title': title,
                    'do_legend': True,
                    'color': 'g',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

            textstr = r'$\alpha ZZ$ = {:.2f} +- {:.2f}'.format(
               self.cross_kerr*1e-3, self.cross_kerr_error*1e-3) + ' kHz'

            self.plot_dicts['text_msg_' + qbn] = {'fig_id': figure_name,
                                                  'text_string': textstr,
                                                  'ypos': -0.2,
                                                  'xpos': -0.075,
                                                  'horizontalalignment': 'left',
                                                  'verticalalignment': 'top',
                                                  'plotfn': self.plot_text}




class OverUnderRotationAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['projected_data_dict'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            model = lmfit.models.LinearModel()
            guess_pars = model.guess(data=data, x=sweep_points)
            guess_pars['intercept'].value = 0.5
            guess_pars['intercept'].vary = False
            key = 'fit_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': model.func,
                'fit_xvals': {'x': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            try:
                old_amp180 = a_tools.get_instr_setting_value_from_file(
                    file_path=self.raw_data_dict['folder'][0],
                    instr_name=qbn, param_name='amp180{}'.format(
                        '_ef' if 'f' in self.data_to_fit[qbn] else ''))
            except KeyError:
                old_amp180 = a_tools.get_instr_setting_value_from_file(
                    file_path=self.raw_data_dict['folder'][0],
                    instr_name=qbn, param_name='{}_amp180'.format(
                        'ef' if 'f' in self.data_to_fit[qbn] else 'ge'))

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn][
                'corrected_amp'] = old_amp180 - self.fit_dicts[
                'fit_' + qbn]['fit_res'].best_values['slope']*old_amp180
            self.proc_data_dict['analysis_params_dict'][qbn][
                'corrected_amp_stderr'] = self.fit_dicts[
                'fit_' + qbn]['fit_res'].params['slope'].stderr*old_amp180

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                # rename base plot
                if self.fit_dicts['fit_' + qbn][
                        'fit_res'].best_values['slope'] >= 0:
                    base_plot_name = 'OverRotation_' + qbn
                else:
                    base_plot_name = 'UnderRotation_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['fit_' + qbn]['fit_res'],
                    'setlabel': 'linear fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                try:
                    old_amp180 = a_tools.get_instr_setting_value_from_file(
                        file_path=self.raw_data_dict['folder'][0],
                        instr_name=qbn, param_name='amp180{}'.format(
                            '_ef' if 'f' in self.data_to_fit[qbn] else ''))
                except KeyError:
                    old_amp180 = a_tools.get_instr_setting_value_from_file(
                        file_path=self.raw_data_dict['folder'][0],
                        instr_name=qbn, param_name='{}_amp180'.format(
                            'ef' if 'f' in self.data_to_fit[qbn] else 'ge'))
                correction_dict = self.proc_data_dict['analysis_params_dict']
                fit_res = self.fit_dicts['fit_' + qbn]['fit_res']
                textstr = '$\pi$-Amp = {:.4f} mV'.format(
                    correction_dict[qbn]['corrected_amp']*1e3) \
                          + ' $\pm$ {:.1e} mV'.format(
                    correction_dict[qbn]['corrected_amp_stderr']*1e3) \
                          + '\nold $\pi$-Amp = {:.4f} mV'.format(
                    old_amp180*1e3) \
                          + '\namp. correction = {:.4f} mV'.format(
                              fit_res.best_values['slope']*old_amp180*1e3) \
                          + '\nintercept = {:.2f}'.format(
                              fit_res.best_values['intercept'])
                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}

                self.plot_dicts['half_hline_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': 0.5,
                    'xmin': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-1],
                    'colors': 'gray'}


class CPhaseLeakageAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self):
        super().process_data()

        self.leakage_qbname = self.get_param_value('leakage_qbname')
        self.cphase_qbname = self.get_param_value('cphase_qbname')
        if self.leakage_qbname is None and self.cphase_qbname is None:
            raise ValueError('Please provide either leakage_qbname or '
                             'cphase_qbname.')
        elif self.cphase_qbname is None:
            self.cphase_qbname = [qbn for qbn in self.qb_names if
                                  qbn != self.leakage_qbname][0]
        elif self.leakage_qbname is None:
            self.leakage_qbname = [qbn for qbn in self.qb_names if
                                   qbn != self.cphase_qbname]
            if len(self.leakage_qbname) > 0:
                self.leakage_qbname = self.leakage_qbname[0]
            else:
                self.leakage_qbname = None

        for qbn, data in self.proc_data_dict['data_to_fit'].items():
            if data.shape[1] != self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points'].size:
                self.proc_data_dict['data_to_fit'][qbn] = data.T

        # convert phases to radians
        for qbn in self.qb_names:
            sweep_dict = self.proc_data_dict['sweep_points_dict'][qbn]
            sweep_dict['sweep_points'] *= np.pi/180

        # reshape data for ease of use
        self.proc_data_dict['data_to_fit_reshaped'] = {
            qbn: np.reshape(
                self.proc_data_dict['data_to_fit'][qbn][
                    :, :-self.num_cal_points],
                (2*self.proc_data_dict['data_to_fit'][qbn][
                    :, :-self.num_cal_points].shape[0],
                self.proc_data_dict['data_to_fit'][qbn][
                    :, :-self.num_cal_points].shape[1]//2))
            for qbn in self.qb_names}

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        self.leakage_values = np.array([])
        labels = ['e', 'g']
        for i, qbn in enumerate(self.qb_names):
            for row in range(self.proc_data_dict['data_to_fit_reshaped'][
                                 qbn].shape[0]):
                phases = np.unique(self.proc_data_dict['sweep_points_dict'][
                    qbn]['msmt_sweep_points'])
                data = self.proc_data_dict['data_to_fit_reshaped'][qbn][row, :]
                key = 'fit_{}{}_{}'.format(labels[row % 2], row, qbn)
                if qbn == self.cphase_qbname:
                    # fit cphase qb results to a cosine
                    model = lmfit.Model(fit_mods.CosFunc)
                    guess_pars = fit_mods.Cos_guess(
                        model=model,
                        t=phases,
                        data=data)
                    guess_pars['amplitude'].vary = True
                    guess_pars['offset'].vary = True
                    guess_pars['frequency'].value = 1/(2*np.pi)
                    guess_pars['frequency'].vary = False
                    guess_pars['phase'].vary = True

                    self.fit_dicts[key] = {
                        'fit_fn': fit_mods.CosFunc,
                        'fit_xvals': {'t': phases},
                        'fit_yvals': {'data': data},
                        'guess_pars': guess_pars}
                else:
                    if self.get_param_value('classified_ro', False):
                        self.leakage_values = np.append(self.leakage_values,
                                                        np.mean(data))
                    else:
                        # fit leakage qb results to a constant
                        model = lmfit.models.ConstantModel()
                        guess_pars = model.guess(data=data, x=phases)

                        self.fit_dicts[key] = {
                            'fit_fn': model.func,
                            'fit_xvals': {'x': phases},
                            'fit_yvals': {'data': data},
                            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        # get cphases population losses
        keys = [k for k in list(self.fit_dicts.keys()) if
                self.cphase_qbname in k]
        fit_res_objs = [self.fit_dicts[k]['fit_res'] for k in keys]
        # cphases
        phases = np.array([fr.best_values['phase'] for fr in fit_res_objs])
        phases_errs = np.array([fr.params['phase'].stderr
                                for fr in fit_res_objs])
        phases_errs[phases_errs == None] = 0.0

        cphases = phases[0::2] - phases[1::2]
        cphases[cphases < 0] += 2*np.pi
        cphases_stderrs = np.sqrt(np.array(phases_errs[0::2]**2 +
                                           phases_errs[1::2]**2,
                                           dtype=np.float64))
        self.proc_data_dict['analysis_params_dict'][
            'cphase'] = {'val': cphases, 'stderr': cphases_stderrs}

        # population losses
        amps = np.array([fr.best_values['amplitude'] for fr in fit_res_objs])
        amps_errs = np.array([fr.params['amplitude'].stderr
                                for fr in fit_res_objs])
        amps_errs[amps_errs == None] = 0.0

        population_loss = (amps[0::2] - amps[1::2])/amps[1::2]
        x   = amps[0::2] - amps[1::2]
        x_err = np.array(amps_errs[0::2]**2 + amps_errs[1::2]**2,
                         dtype=np.float64)
        y = amps[1::2]
        y_err = amps_errs[1::2]
        try:
            population_loss_stderrs = np.sqrt(np.array(
                ((y * x_err) ** 2 + (x * y_err) ** 2) / (y ** 4),
                dtype=np.float64))
        except:
            population_loss_stderrs = float("nan")
        self.proc_data_dict['analysis_params_dict'][
            'population_loss'] = {'val': population_loss,
                                  'stderr': population_loss_stderrs}

        if self.leakage_qbname is not None:
            # get leakage
            if self.get_param_value('classified_ro', False):
                leakage = self.leakage_values[0::2]
                leakage_errs = np.zeros(len(leakage))
                leakage_increase = self.leakage_values[0::2] - \
                                   self.leakage_values[1::2]
                leakage_increase_errs = np.zeros(len(leakage))
            else:
                keys = [k for k in list(self.fit_dicts.keys()) if
                        self.leakage_qbname in k]
                fit_res_objs = [self.fit_dicts[k]['fit_res'] for k in keys]

                lines = np.array([fr.best_values['c'] for fr in fit_res_objs])
                lines_errs = np.array([fr.params['c'].stderr for
                                       fr in fit_res_objs])
                lines_errs[lines_errs == None] = 0.0

                leakage = lines[0::2]
                leakage_errs = np.array(np.sqrt(lines_errs[0::2]**2 +
                                                lines_errs[1::2]**2),
                                        dtype=np.float64)
                leakage_increase = lines[0::2] - lines[1::2]
                leakage_increase_errs = np.array(lines_errs[0::2],
                                                 dtype=np.float64)

            self.proc_data_dict['analysis_params_dict'][
                'leakage'] = {'val': leakage, 'stderr': leakage_errs}
            self.proc_data_dict['analysis_params_dict'][
                'leakage_increase'] = {'val': leakage_increase,
                                       'stderr': leakage_increase_errs}

        self.save_processed_data(key='analysis_params_dict')

    def plot_traces(self, prob_label, data_2d, qbn):
        plotsize = self.get_default_plot_params(set=False)[
            'figure.figsize']
        plotsize = (plotsize[0], plotsize[0]/1.25)
        if data_2d.shape[1] != self.proc_data_dict[
                'sweep_points_dict'][qbn]['sweep_points'].size:
            data_2d = data_2d.T

        data_2d_reshaped = np.reshape(
            data_2d[:, :-self.num_cal_points],
            (2*data_2d[:, :-self.num_cal_points].shape[0],
             data_2d[:, :-self.num_cal_points].shape[1]//2))

        data_2d_cal_reshaped = [[data_2d[:, -self.num_cal_points:]]] * \
                               (2*data_2d[:, :-self.num_cal_points].shape[0])

        ref_states_plot_dicts = {}
        for row in range(data_2d_reshaped.shape[0]):
            phases = np.unique(self.proc_data_dict['sweep_points_dict'][qbn][
                    'msmt_sweep_points'])
            data = data_2d_reshaped[row, :]
            legend_bbox_to_anchor = (1, -0.15)
            legend_pos = 'upper right'
            legend_ncol = 2

            if qbn == self.cphase_qbname and \
                    self.get_latex_prob_label(prob_label) == \
                    self.get_latex_prob_label(self.data_to_fit[qbn]):
                figure_name = 'Cphase_{}_{}'.format(qbn, prob_label)
            elif qbn == self.leakage_qbname and \
                    self.get_latex_prob_label(prob_label) == \
                    self.get_latex_prob_label(self.data_to_fit[qbn]):
                figure_name = 'Leakage_{}_{}'.format(qbn, prob_label)
            else:
                figure_name = 'projected_plot_' + qbn + '_' + \
                              prob_label

            # plot cal points
            if self.num_cal_points > 0:
                data_w_cal = data_2d_cal_reshaped[row][0][0]
                for i, cal_pts_idxs in enumerate(
                        self.cal_states_dict.values()):
                    s = '{}_{}_{}'.format(row, qbn, prob_label)
                    ref_state_plot_name = list(
                        self.cal_states_dict)[i] + '_' + s
                    ref_states_plot_dicts[ref_state_plot_name] = {
                        'fig_id': figure_name,
                        'plotfn': self.plot_line,
                        'plotsize': plotsize,
                        'xvals': self.proc_data_dict[
                            'sweep_points_dict'][qbn][
                            'cal_points_sweep_points'][
                            cal_pts_idxs],
                        'yvals': data_w_cal[cal_pts_idxs],
                        'setlabel': list(
                            self.cal_states_dict)[i] if
                        row == 0 else '',
                        'do_legend': row == 0,
                        'legend_bbox_to_anchor':
                            legend_bbox_to_anchor,
                        'legend_pos': legend_pos,
                        'legend_ncol': legend_ncol,
                        'linestyle': 'none',
                        'line_kws': {'color':
                            self.get_cal_state_color(
                                list(self.cal_states_dict)[i])}}
                # phases = phases[:-self.num_cal_points]
                # data = data[:-self.num_cal_points]

            if self.leakage_qbname is not None:
                legend_label = '{} in $|g\\rangle$'.format(
                    self.leakage_qbname) if row % 2 != 0 else \
                    '{} in $|e\\rangle$'.format(
                        self.leakage_qbname)
            else:
                legend_label = 'qbc in $|g\\rangle$' if \
                    row % 2 != 0 else 'qbc in $|e\\rangle$'
            hard_sweep_params = self.get_param_value('hard_sweep_params')
            sweep_name = self.get_param_value('sweep_name')
            sweep_unit = self.get_param_value('sweep_unit')
            if hard_sweep_params is not None:
                xlabel = list(hard_sweep_params)[0]
                xunit = list(hard_sweep_params.values())[0][
                    'unit']
            elif (sweep_name is not None) and (sweep_unit is not None):
                xlabel = sweep_name
                xunit = sweep_unit
            else:
                xlabel = self.raw_data_dict['sweep_parameter_names']
                xunit = self.raw_data_dict['sweep_parameter_units']
            if np.ndim(xunit) > 0:
                xunit = xunit[0]
            self.plot_dicts['data_{}_{}_{}'.format(
                row, qbn, prob_label)] = {
                'plotfn': self.plot_line,
                'fig_id': figure_name,
                'plotsize': plotsize,
                'xvals': phases,
                'xlabel': xlabel,
                'xunit': xunit,
                'yvals': data,
                'ylabel': '{} state population'.format(
                    self.get_latex_prob_label(prob_label)),
                'yunit': '',
                'yscale': self.get_param_value("yscale", "linear"),
                'setlabel': 'Data - ' + legend_label
                if row in [0, 1] else '',
                'title': self.raw_data_dict['timestamp'] + ' ' +
                         self.raw_data_dict['measurementstring'] + '-' + qbn,
                'linestyle': 'none',
                'color': 'C0' if row % 2 == 0 else 'C2',
                'do_legend': row in [0, 1],
                'legend_ncol': legend_ncol,
                'legend_bbox_to_anchor': legend_bbox_to_anchor,
                'legend_pos': legend_pos}

            if self.do_fitting and 'projected' not in figure_name:
                k = 'fit_{}{}_{}'.format(
                    'e' if row % 2 == 0 else 'g', row, qbn)

                if qbn == self.cphase_qbname:
                    fit_res = self.fit_dicts[k]['fit_res']
                    self.plot_dicts[k + '_' + prob_label] = {
                        'fig_id': figure_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res,
                        'setlabel': 'Fit - ' + legend_label
                        if row in [0, 1] else '',
                        'color': 'C0' if row % 2 == 0 else 'C2',
                        'do_legend': row in [0, 1],
                        'legend_ncol': legend_ncol,
                        'legend_bbox_to_anchor':
                            legend_bbox_to_anchor,
                        'legend_pos': legend_pos}
                else:
                    if self.get_param_value('classified_ro', False):
                        pass
                    else:
                        fit_res = self.fit_dicts[k]['fit_res']
                        xvals = fit_res.userkws[
                            fit_res.model.independent_vars[0]]
                        xfine = np.linspace(min(xvals), max(xvals), 100)
                        yvals = fit_res.model.func(
                            xfine, **fit_res.best_values)
                        if not hasattr(yvals, '__iter__'):
                            yvals = np.array(len(xfine)*[yvals])

                        self.plot_dicts[k] = {
                            'fig_id': figure_name,
                            'plotfn': self.plot_line,
                            'xvals': xfine,
                            'yvals': yvals,
                            'marker': '',
                            'setlabel': 'Fit - ' + legend_label
                            if row in [0, 1] else '',
                            'do_legend': row in [0, 1],
                            'legend_ncol': legend_ncol,
                            'color': 'C0' if row % 2 == 0 else 'C2',
                            'legend_bbox_to_anchor':
                                legend_bbox_to_anchor,
                            'legend_pos': legend_pos}

        # ref state plots need to be added at the end, otherwise the
        # legend for |g> and |e> is added twice (because of the
        # condition do_legend = (row in [0,1]) in the plot dicts above
        if self.num_cal_points > 0:
            self.plot_dicts.update(ref_states_plot_dicts)
        return figure_name

    def prepare_plots(self):
        if self.options_dict.get('plot_all_traces', True):
            for j, qbn in enumerate(self.qb_names):
                if self.options_dict.get('plot_all_probs', True):
                    for prob_label, data_2d in self.proc_data_dict[
                            'projected_data_dict'][qbn].items():
                        figure_name = self.plot_traces(prob_label, data_2d, qbn)
                else:
                    figure_name = self.plot_traces(
                        self.data_to_fit[qbn], self.proc_data_dict[
                            'data_to_fit'][qbn], qbn)

                if self.do_fitting and len(self.proc_data_dict[
                               'analysis_params_dict']['cphase']['val']) == 1:
                        if qbn == self.cphase_qbname:
                            textstr = 'Cphase = {:.2f}'.format(
                                self.proc_data_dict['analysis_params_dict'][
                                    'cphase']['val'][0]*180/np.pi) + \
                                      r'$^{\circ}$' + \
                                    '$\\pm${:.2f}'.format(self.proc_data_dict[
                                          'analysis_params_dict']['cphase'][
                                          'stderr'][0] * 180 / np.pi) + \
                                      r'$^{\circ}$'
                            textstr += '\nPopulation loss = ' + \
                                       '{:.3f} $\\pm$ {:.3f}'.format(
                                self.proc_data_dict[
                                    'analysis_params_dict'][
                                    'population_loss']['val'][0],
                                                self.proc_data_dict[
                                    'analysis_params_dict'][
                                    'population_loss']['stderr'][0])
                            self.plot_dicts['text_msg_' + qbn] = {
                                'fig_id': 'Cphase_{}_pe'.format(qbn),
                                'ypos': -0.2,
                                'xpos': -0.05,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'plotfn': self.plot_text,
                                'text_string': textstr}
                        else:
                            textstr = 'Leakage = {:.5f} $\\pm$ {:.5f}'.format(
                                self.proc_data_dict['analysis_params_dict'][
                                    'leakage']['val'][0],
                                self.proc_data_dict['analysis_params_dict'][
                                    'leakage']['stderr'][0])
                            textstr += '\nLeakage increase = \n\t' \
                                       '{:.5f} $\\pm$ {:.5f}'.format(
                                self.proc_data_dict['analysis_params_dict'][
                                    'leakage_increase']['val'][0],
                                self.proc_data_dict['analysis_params_dict'][
                                    'leakage_increase']['stderr'][0])
                            self.plot_dicts['text_msg_' + qbn] = {
                                'fig_id': figure_name,
                                'ypos': -0.2,
                                'xpos': -0.05,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'plotfn': self.plot_text,
                                'text_string': textstr}

        # plot analysis results
        if self.do_fitting and len(self.proc_data_dict[
                'analysis_params_dict']['cphase']['val']) > 1:
            # unique_swpts2d = [np.unique(arr) for arr in self.raw_data_dict[
            #     'sweep_points_2D_dict'][self.qb_names[0]]]
            # swpts2d_lengths = np.array([len(np.unique(arr)) for arr in
            #                             unique_swpts2d])
            # swpts2d_idxs = np.where(swpts2d_lengths > 1)[0]
            assert ('soft_sweep_params' in self.metadata)
            ss_pars = self.metadata['soft_sweep_params']

            for idx, ss_pname in enumerate(ss_pars):
                for param_name, results_dict in self.proc_data_dict[
                        'analysis_params_dict'].items():
                    reps = len(results_dict['val']) / \
                           len(ss_pars[ss_pname]['values'])
                    plot_name = '{}_vs_{}'.format(param_name, ss_pname)
                    if param_name == 'cphase':
                        yvals = results_dict['val']*180/np.pi - 180
                        yerr = results_dict['stderr']*180/np.pi
                        ylabel = param_name + '-$180^{\\circ}$'
                        self.plot_dicts[plot_name+'_hline'] = {
                            'fig_id': plot_name,
                            'plotfn': self.plot_hlines,
                            'y': 0,
                            'xmin': np.min(ss_pars[ss_pname]['values']),
                            'xmax': np.max(ss_pars[ss_pname]['values']),
                            'colors': 'gray'}
                    else:
                        yvals = results_dict['val']
                        yerr = results_dict['stderr']
                        ylabel = param_name
                    self.plot_dicts[plot_name] = {
                        'plotfn': self.plot_line,
                        'xvals': np.repeat(ss_pars[ss_pname]['values'], reps),
                        'xlabel': ss_pname,
                        'xunit': ss_pars[ss_pname]['unit'],
                        'yvals': yvals,
                        'yerr': yerr if param_name != 'leakage' else None,
                        'ylabel': ylabel,
                        'yunit': 'deg' if param_name == 'cphase' else '',
                        'linestyle': 'none',
                        'do_legend': False}


class CZDynamicPhaseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self):
        super().process_data()
        # convert phases to radians
        for qbn in self.qb_names:
            sweep_dict = self.proc_data_dict['sweep_points_dict'][qbn]
            sweep_dict['sweep_points'] *= np.pi/180

        # get data with flux pulse and w/o flux pulse
        self.data_with_fp = OrderedDict()
        self.data_no_fp = OrderedDict()
        for qbn in self.qb_names:
            all_data = self.proc_data_dict['data_to_fit'][qbn]
            if self.num_cal_points != 0:
                all_data = all_data[:-self.num_cal_points]
            self.data_with_fp[qbn] = all_data[0: len(all_data)//2]
            self.data_no_fp[qbn] = all_data[len(all_data)//2:]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            sweep_points = np.unique(
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'msmt_sweep_points'])
            for i, data in enumerate([self.data_with_fp[qbn],
                                      self.data_no_fp[qbn]]):
                cos_mod = lmfit.Model(fit_mods.CosFunc)
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod,
                    t=sweep_points,
                    data=data)
                guess_pars['amplitude'].vary = True
                guess_pars['offset'].vary = True
                guess_pars['frequency'].value = 1/(2*np.pi)
                guess_pars['frequency'].vary = False
                guess_pars['phase'].vary = True

                key = 'cos_fit_{}_{}'.format(qbn, 'wfp' if i == 0 else 'nofp')
                self.fit_dicts[key] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn][
                'dynamic_phase'] = {
                'val': (self.fit_dicts[f'cos_fit_{qbn}_wfp'][
                            'fit_res'].best_values['phase'] -
                        self.fit_dicts[f'cos_fit_{qbn}_nofp'][
                            'fit_res'].best_values['phase']),
                'stderr': np.sqrt(
                    self.fit_dicts[f'cos_fit_{qbn}_wfp'][
                        'fit_res'].params['phase'].stderr**2 +
                    self.fit_dicts[f'cos_fit_{qbn}_nofp'][
                        'fit_res'].params['phase'].stderr**2)
                }
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()
        for qbn in self.qb_names:
            for i, data in enumerate([self.data_with_fp[qbn],
                                      self.data_no_fp[qbn]]):
                fit_key = f'cos_fit_{qbn}_wfp' if i == 0 else \
                    f'cos_fit_{qbn}_nofp'
                plot_name_suffix = 'fit_'+'wfp' if i == 0 else 'nofp'
                cal_pts_data = self.proc_data_dict['data_to_fit'][qbn][
                               -self.num_cal_points:]
                base_plot_name = 'Dynamic_phase_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=np.concatenate((data,cal_pts_data)),
                    sweep_points=np.unique(
                        self.proc_data_dict['sweep_points_dict'][qbn][
                            'sweep_points']),
                    data_label='with flux pulse' if i == 0 else 'no flux pulse',
                    plot_name_suffix=qbn + plot_name_suffix,
                    qb_name=qbn,
                    do_legend_cal_states=(i == 0))
                if self.do_fitting:
                    fit_res = self.fit_dicts[fit_key]['fit_res']
                    self.plot_dicts[plot_name_suffix + '_' + qbn] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res ,
                        'setlabel': 'cosine fit',
                        'color': 'r',
                        'do_legend': i == 0}

                    textstr = 'Dynamic phase {}:\n\t{:.2f}'.format(
                        qbn,
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'dynamic_phase']['val']*180/np.pi) + \
                              r'$^{\circ}$' + \
                            '$\\pm${:.2f}'.format(
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'dynamic_phase']['stderr']*180/np.pi) + \
                              r'$^{\circ}$'

                    fpl = self.get_param_value('flux_pulse_length')
                    if fpl is not None:
                        textstr += '\n length: {:.2f} ns'.format(fpl*1e9)
                    fpa = self.get_param_value('flux_pulse_amp')
                    if fpa is not None:
                        textstr += '\n amp: {:.4f} V'.format(fpa)

                    self.plot_dicts['text_msg_' + qbn] = {
                        'fig_id': base_plot_name,
                        'ypos': -0.15,
                        'xpos': -0.05,
                        'horizontalalignment': 'left',
                        'verticalalignment': 'top',
                        'plotfn': self.plot_text,
                        'text_string': textstr}
            for plot_name in list(self.plot_dicts)[::-1]:
                if self.plot_dicts[plot_name].get('do_legend', False):
                    break
            self.plot_dicts[plot_name].update(
                {'legend_ncol': 2,
                 'legend_bbox_to_anchor': (1, -0.15),
                 'legend_pos': 'upper right'})

class MultiQutrit_Timetrace_Analysis(ba.BaseDataAnalysis):
    """
    Analysis class for timetraces, in particular use to compute
    Optimal SNR integration weights.
    """
    def __init__(self, qb_names=None, auto=True, **kwargs):
        """
        Initializes the timetrace analysis class.
        Args:
            qb_names (list): name of the qubits to analyze (can be a subset
                of the measured qubits)
            auto (bool): Start analysis automatically
            **kwargs:
                t_start: timestamp of the first timetrace
                t_stop: timestamp of the last timetrace to analyze
                options_dict (dict): relevant parameters:
                    acq_weights_basis (list, dict):
                        list of basis vectors used to compute optimal weight.
                        e.g. ["ge", 'gf'], the first basis vector will be the
                        "e" timetrace minus the "g" timetrace and the second basis
                        vector is f - g. The first letter in each basis state is the
                        "reference state", i.e. the one of which the timetrace
                         is substracted. Can also be passed as a dictionary where
                         keys are the qubit names and the values are lists of basis states
                         in case different bases should be used for different qubits.
                    orthonormalize (bool): Whether or not to orthonormalize the
                        weight basis
                    tmax (float): time boundary for the plot (not the weights)
                        in seconds.
                    scale_weights (bool): scales the weights near unity to avoid
                        loss of precision on FPGA if weights are too small

        """

        if qb_names is not None:
            self.params_dict = {}
            for qbn in qb_names:
                s = 'Instrument settings.' + qbn
                for trans_name in ['ge', 'ef']:
                    self.params_dict[f'ro_mod_freq_' + qbn] = \
                        s + f'.ro_mod_freq'
            self.numeric_params = list(self.params_dict)

        self.qb_names = qb_names
        super().__init__(auto=auto, **kwargs)

    def extract_data(self):
        super().extract_data()

        if self.qb_names is None:
            # get all qubits from cal_points of first timetrace
            cp = CalibrationPoints.from_string(
                self.get_param_value('cal_points', None, 0))
            self.qb_names = deepcopy(cp.qb_names)

        self.channel_map = self.get_param_value('channel_map', None,
                                                metadata_index=0)
        if self.channel_map is None:
            # assume same channel map for all timetraces (pick 0th)
            value_names = self.raw_data_dict[0]['value_names']
            if np.ndim(value_names) > 0:
                value_names = value_names
            if 'w' in value_names[0]:
                self.channel_map = a_tools.get_qb_channel_map_from_hdf(
                    self.qb_names, value_names=value_names,
                    file_path=self.raw_data_dict['folder'])
            else:
                self.channel_map = {}
                for qbn in self.qb_names:
                    self.channel_map[qbn] = value_names

        if len(self.channel_map) == 0:
            raise ValueError('No qubit RO channels have been found.')

    def process_data(self):
        super().process_data()
        pdd = self.proc_data_dict

        pdd['analysis_params_dict'] = dict()
        ana_params = pdd['analysis_params_dict']
        ana_params['timetraces'] = defaultdict(dict)
        ana_params['optimal_weights'] = defaultdict(dict)
        ana_params['optimal_weights_basis_labels'] = defaultdict(dict)
        for qbn in self.qb_names:
            # retrieve time traces
            for i, rdd in enumerate(self.raw_data_dict):
                ttrace_per_ro_ch = [rdd["measured_data"][ch]
                                    for ch in self.channel_map[qbn]]
                if len(ttrace_per_ro_ch) != 2:
                    raise NotImplementedError(
                        'This analysis does not support optimal weight '
                        f'measurement based on {len(ttrace_per_ro_ch)} ro channels.'
                        f' Try again with 2 RO channels.')
                cp = CalibrationPoints.from_string(
                    self.get_param_value('cal_points', None, i))
                # get state of qubit. There can be only one cal point per sequence
                # when using uhf for time traces so it is the 0th state
                qb_state = cp.states[0][cp.qb_names.index(qbn)]
                # store all timetraces in same pdd for convenience
                ana_params['timetraces'][qbn].update(
                    {qb_state: ttrace_per_ro_ch[0] + 1j *ttrace_per_ro_ch[1]})

            timetraces = ana_params['timetraces'][qbn] # for convenience
            basis_labels = self.get_param_value('acq_weights_basis', None, 0)
            if basis_labels is None:
                # guess basis labels from # states measured
                basis_labels = ["ge", "gf"] \
                    if len(ana_params['timetraces'][qbn]) > 2 else ['ge']

            if isinstance(basis_labels, dict):
                # if different basis for qubits, then select the according one
                basis_labels = basis_labels[qbn]

            # check that states from the basis are included in mmnt
            for bs in basis_labels:
                for qb_s in bs:
                     assert qb_s in timetraces,\
                         f'State: {qb_s} on {qbn} was not provided in the given ' \
                         f'timestamps but was requested as part of the basis' \
                         f' {basis_labels}. Please choose another weight basis.'
            basis = np.array([timetraces[b[1]] - timetraces[b[0]]
                              for b in basis_labels])

            # orthonormalize if required
            if self.get_param_value("orthonormalize", False):
                basis = math.gram_schmidt(basis.T).T
                basis_labels = [bs + "_ortho" if bs != basis_labels[0] else bs
                                for bs in basis_labels]

            # scale if required
            if self.get_param_value('scale_weights', True):
                k = np.amax([(np.max(np.abs(b.real)),
                              np.max(np.abs(b.imag))) for b in basis])
                basis /= k
            ana_params['optimal_weights'][qbn] = basis
            ana_params['optimal_weights_basis_labels'][qbn] = basis_labels

            self.save_processed_data()

    def prepare_plots(self):

        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        ana_params = self.proc_data_dict['analysis_params_dict']
        for qbn in self.qb_names:
            mod_freq = float(
                rdd[0].get(f'ro_mod_freq_{qbn}',
                           self.get_hdf_param_value(f"Instrument settings/{qbn}",
                                                    'ro_mod_freq')))
            tbase = rdd[0]['hard_sweep_points']
            basis_labels = pdd["analysis_params_dict"][
                'optimal_weights_basis_labels'][qbn]
            title = 'Optimal SNR weights ' + qbn + \
                    "".join(['\n' + rddi["timestamp"] for rddi in rdd]) \
                            + f'\nWeight Basis: {basis_labels}'
            plot_name = f"weights_{qbn}"
            xlabel = "Time, $t$"
            modulation = np.exp(2j * np.pi * mod_freq * tbase)

            for ax_id, (state, ttrace) in \
                enumerate(ana_params["timetraces"][qbn].items()):
                for func, label in zip((np.real, np.imag), ('I', "Q")):
                    # plot timetraces for each state, I and Q channels
                    self.plot_dicts[f"{plot_name}_{state}_{label}"] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': tbase,
                        'xunit': 's',
                        "marker": "",
                        'yvals': func(ttrace*modulation),
                        'ylabel': 'Voltage, $V$',
                        'yunit': 'V',
                        "sharex": True,
                        "xrange": (0, self.get_param_value('tmax', 400e-9, 0)),
                        "setdesc": label + f"_{state}",
                        "setlabel": "",
                        "do_legend":True,
                        "legend_pos": "upper right",
                        'numplotsx': 1,
                        'numplotsy': len(rdd) + 1, # #states + 1 for weights
                        'plotsize': (10,
                                     (len(rdd) + 1) * 3), # 3 inches per plot
                        'title': title if ax_id == 0 else ""}
            ax_id = len(ana_params["timetraces"][qbn]) # id plots for weights
            for i, weights in enumerate(ana_params['optimal_weights'][qbn]):
                for func, label in zip((np.real, np.imag), ('I', "Q")):
                    self.plot_dicts[f"{plot_name}_weights_{label}_{i}"] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': tbase,
                        'xlabel': xlabel,
                        "setlabel": "",
                        "marker": "",
                        'xunit': 's',
                        'yvals': func(weights * modulation),
                        'ylabel': 'Voltage, $V$ (arb.u.)',
                        "sharex": True,
                        "xrange": (0, self.get_param_value('tmax', 400e-9, 0)),
                        "setdesc": label + f"_{i+1}",
                        "do_legend": True,
                        "legend_pos": "upper right",
                        }


class MultiQutrit_Singleshot_Readout_Analysis(MultiQubit_TimeDomain_Analysis):
    """
    Analysis class for parallel SSRO qutrit/qubit calibration. It is a child class
    from the tda.MultiQubit_Timedomain_Analysis as it uses the same functions to
    - preprocess the data to remove active reset/preselection
    - extract the channel map
    - reorder the data per qubit
    Note that in the future, it might be useful to transfer these functionalities
    to the base analysis.
    """

    def __init__(self,
                 options_dict: dict = None, auto=True, **kw):
        '''
        options dict options:
            'nr_bins' : number of bins to use for the histograms
            'post_select' :
            'post_select_threshold' :
            'nr_samples' : amount of different samples (e.g. ground and excited = 2)
            'sample_0' : index of first sample (ground-state)
            'sample_1' : index of second sample (first excited-state)
            'max_datapoints' : maximum amount of datapoints for culumative fit
            'log_hist' : use log scale for the y-axis of the 1D histograms
            'verbose' : see BaseDataAnalysis
            'presentation_mode' : see BaseDataAnalysis
            'classif_method': how to classify the data.
                'ncc' : default. Nearest Cluster Center
                'gmm': gaussian mixture model.
                'threshold': finds optimal vertical and horizontal thresholds.
            'classif_kw': kw to pass to the classifier
            see BaseDataAnalysis for more.
        '''
        super().__init__(options_dict=options_dict, auto=False,
                         **kw)
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_data': 'measured_data',
            'value_names': 'value_names',
            'value_units': 'value_units'}
        self.numeric_params = []
        self.DEFAULT_CLASSIF = "gmm"
        self.classif_method = self.options_dict.get("classif_method",
                                                    self.DEFAULT_CLASSIF)

        self.create_job(options_dict=options_dict, auto=auto, **kw)

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        self.preselection = \
            self.get_param_value("preparation_params",
                                 {}).get("preparation_type", "wait") == "preselection"
        default_states_info = defaultdict(dict)
        default_states_info.update({"g": {"label": r"$|g\rangle$"},
                               "e": {"label": r"$|e\rangle$"},
                               "f": {"label": r"$|f\rangle$"}
                               })

        self.states_info = \
            self.get_param_value("states_info",
                                {qbn: deepcopy(default_states_info)
                                 for qbn in self.qb_names})

    def process_data(self):
        """
        Create the histograms based on the raw data
        """
        ######################################################
        #  Separating data into shots for each level         #
        ######################################################
        super().process_data()
        del self.proc_data_dict['data_to_fit'] # not used in this analysis
        n_states = len(self.cp.states)

        # prepare data in convenient format, i.e. arrays per qubit and per state
        # e.g. {'qb1': {'g': np.array of shape (n_shots, n_ro_ch}, ...}, ...}
        shots_per_qb = dict()        # store shots per qb and per state
        presel_shots_per_qb = dict() # store preselection ro
        means = defaultdict(OrderedDict)    # store mean per qb for each ro_ch
        pdd = self.proc_data_dict    # for convenience of notation

        for qbn in self.qb_names:
            # shape is (n_shots, n_ro_ch) i.e. one column for each ro_ch
            shots_per_qb[qbn] = \
                np.asarray(list(
                    pdd['meas_results_per_qb'][qbn].values())).T
            # make 2D array in case only one channel (1D array)
            if len(shots_per_qb[qbn].shape) == 1:
                shots_per_qb[qbn] = np.expand_dims(shots_per_qb[qbn],
                                                   axis=-1)
            for i, qb_state in enumerate(self.cp.get_states(qbn)[qbn]):
                means[qbn][qb_state] = np.mean(shots_per_qb[qbn][i::n_states],
                                               axis=0)
            if self.preselection:
                # preselection shots were removed so look at raw data
                # and look at only the first out of every two readouts
                presel_shots_per_qb[qbn] = \
                    np.asarray(list(
                        pdd['meas_results_per_qb_raw'][qbn].values())).T[::2]
                # make 2D array in case only one channel (1D array)
                if len(presel_shots_per_qb[qbn].shape) == 1:
                    presel_shots_per_qb[qbn] = \
                        np.expand_dims(presel_shots_per_qb[qbn], axis=-1)

        # create placeholders for analysis data
        pdd['analysis_params'] = dict()
        pdd['data'] = defaultdict(dict)
        pdd['analysis_params']['state_prob_mtx'] = defaultdict(dict)
        pdd['analysis_params']['classifier_params'] = defaultdict(dict)
        pdd['analysis_params']['means'] = defaultdict(dict)
        pdd['analysis_params']["n_shots"] = len(shots_per_qb[qbn])
        self.clf_ = defaultdict(dict)
        # create placeholders for analysis with preselection
        if self.preselection:
            pdd['data_masked'] = defaultdict(dict)
            pdd['analysis_params']['state_prob_mtx_masked'] = defaultdict(dict)
            pdd['analysis_params']['n_shots_masked'] = defaultdict(dict)

        n_shots = len(shots_per_qb[qbn]) // n_states

        for qbn, qb_shots in shots_per_qb.items():
            # create mapping to integer following ordering in cal_points.
            # Notes:
            # 1) the state_integer should to the order of pdd[qbn]['means'] so that
            # when passing the init_means to the GMM model, it is ensured that each
            # gaussian component will predict the state_integer associated to that state
            # 2) the mapping cannot be preestablished because the GMM predicts labels
            # in range(n_components). For instance, if a qubit has states "g", "f"
            # then the model will predicts 0's and 1's, so the typical g=0, e=1, f=2
            # mapping would fail. The number of different states can be different
            # for each qubit and therefore the mapping should also be done per qubit.
            state_integer = 0
            for state in means[qbn].keys():
                self.states_info[qbn][state]["int"] = state_integer
                state_integer += 1

            # note that if some states are repeated, they are assigned the same label
            qb_states_integer_repr = \
                [self.states_info[qbn][s]["int"]
                 for s in self.cp.get_states(qbn)[qbn]]
            prep_states = np.tile(qb_states_integer_repr, n_shots)

            pdd['analysis_params']['means'][qbn] = deepcopy(means[qbn])
            pdd['data'][qbn] = dict(X=deepcopy(qb_shots),
                                    prep_states=prep_states)
            # self.proc_data_dict['keyed_data'] = deepcopy(data)

            assert np.ndim(qb_shots) == 2, "Data must be a two D array. " \
                                    "Received shape {}, ndim {}"\
                                    .format(qb_shots.shape, np.ndim(qb_shots))
            pred_states, clf_params, clf = \
                self._classify(qb_shots, prep_states,
                               method=self.classif_method, qb_name=qbn,
                               **self.options_dict.get("classif_kw", dict()))
            # order "unique" states to have in usual order "gef" etc.
            state_labels_ordered = self._order_state_labels(
                list(means[qbn].keys()))
            # translate to corresponding integers
            state_labels_ordered_int = [self.states_info[qbn][s]['int'] for s in
                                        state_labels_ordered]
            fm = self.fidelity_matrix(prep_states, pred_states,
                                      labels=state_labels_ordered_int)

            # save fidelity matrix and classifier
            pdd['analysis_params']['state_prob_mtx'][qbn] = fm
            pdd['analysis_params']['classifier_params'][qbn] = clf_params
            self.clf_[qbn] = clf
            if self.preselection:
                #re do with classification first of preselection and masking
                pred_presel = self.clf_[qbn].predict(presel_shots_per_qb[qbn])
                presel_filter = \
                    pred_presel == self.states_info[qbn]['g']['int']
                if np.sum(presel_filter) == 0:
                    log.warning(f"{qbn}: No data left after preselection! "
                                f"Skipping preselection data & figures.")
                    continue
                qb_shots_masked = qb_shots[presel_filter]
                prep_states = prep_states[presel_filter]
                pred_states = self.clf_[qbn].predict(qb_shots_masked)
                fm = self.fidelity_matrix(prep_states, pred_states,
                                          labels=state_labels_ordered_int)

                pdd['data_masked'][qbn] = dict(X=deepcopy(qb_shots_masked),
                                          prep_states=deepcopy(prep_states))
                pdd['analysis_params']['state_prob_mtx_masked'][qbn] = fm
                pdd['analysis_params']['n_shots_masked'][qbn] = \
                    qb_shots_masked.shape[0]

        self.save_processed_data()

    def _classify(self, X, prep_state, method, qb_name, **kw):
        """

        Args:
            X: measured data to classify
            prep_state: prepared states (true values)
            type: classification method
            qb_name: name of the qubit to classify

        Returns:

        """
        if np.ndim(X) == 1:
            X = X.reshape((-1,1))
        params = dict()

        if method == 'ncc':
            ncc = SSROQutrit.NCC(
                self.proc_data_dict['analysis_params']['means'][qb_name])
            pred_states = ncc.predict(X)
            # self.clf_ = ncc
            return pred_states, dict(), ncc

        elif method == 'gmm':
            cov_type = kw.pop("covariance_type", "tied")
            # full allows full covariance matrix for each level. Other options
            # see GM documentation
            # assumes if repeated state, should be considered of the same component
            # this classification method should not be used for multiplexed SSRO
            # analysis
            n_qb_states = len(np.unique(self.cp.get_states(qb_name)[qb_name]))
            gm = GM(n_components=n_qb_states,
                    covariance_type=cov_type,
                    random_state=0,
                    means_init=[mu for _, mu in
                                self.proc_data_dict['analysis_params']
                                    ['means'][qb_name].items()])
            gm.fit(X)
            pred_states = np.argmax(gm.predict_proba(X), axis=1)

            params['means_'] = gm.means_
            params['covariances_'] = gm.covariances_
            params['covariance_type'] = gm.covariance_type
            params['weights_'] = gm.weights_
            params['precisions_cholesky_'] = gm.precisions_cholesky_
            return pred_states, params, gm

        elif method == "threshold":
            tree = DTC(max_depth=kw.pop("max_depth", X.shape[1]),
                       random_state=0, **kw)
            tree.fit(X, prep_state)
            pred_states = tree.predict(X)
            params["thresholds"], params["mapping"] = \
                self._extract_tree_info(tree, self.cp.get_states(qb_name)[qb_name])
            if len(params["thresholds"]) != X.shape[1]:
                msg = "Best 2 thresholds to separate this data lie on axis {}" \
                    ", most probably because the data is not well separated." \
                    "The classifier attribute clf_ can still be used for " \
                    "classification (which was done to obtain the state " \
                    "assignment probability matrix), but only the threshold" \
                    " yielding highest gini impurity decrease was returned." \
                    "\nTo circumvent this problem, you can either choose" \
                    " a second threshold manually (fidelity will likely be " \
                    "worse), make the data more separable, or use another " \
                    "classification method."
                logging.warning(msg.format(list(params['thresholds'].keys())[0]))
            return pred_states, params, tree
        elif method == "threshold_brute":
            raise NotImplementedError()
        else:
            raise NotImplementedError("Classification method: {} is not "
                                      "implemented. Available methods: {}"
                                      .format(method, ['ncc', 'gmm',
                                                       'threshold']))
    @staticmethod
    def _get_covariances(gmm, cov_type=None):
       return SSROQutrit._get_covariances(gmm, cov_type=cov_type)

    @staticmethod
    def fidelity_matrix(prep_states, pred_states, levels=('g', 'e', 'f'),
                        plot=False, labels=None, normalize=True):

        return SSROQutrit.fidelity_matrix(prep_states, pred_states,
                                          levels=levels, plot=plot,
                                          normalize=normalize, labels=labels)

    @staticmethod
    def plot_fidelity_matrix(fm, target_names,
                             title="State Assignment Probability Matrix",
                             auto_shot_info=True, ax=None,
                             cmap=None, normalize=True, show=False):
        return SSROQutrit.plot_fidelity_matrix(
            fm, target_names, title=title, ax=ax,
            auto_shot_info=auto_shot_info,
            cmap=cmap, normalize=normalize, show=show)

    @staticmethod
    def _extract_tree_info(tree_clf, class_names=None):
        return SSROQutrit._extract_tree_info(tree_clf,
                                             class_names=class_names)

    @staticmethod
    def _to_codeword_idx(tuple):
        return SSROQutrit._to_codeword_idx(tuple)

    @staticmethod
    def plot_scatter_and_marginal_hist(data, y_true=None, plot_fitting=False,
                                       **kwargs):
        return SSROQutrit.plot_scatter_and_marginal_hist(
            data, y_true=y_true, plot_fitting=plot_fitting, **kwargs)

    @staticmethod
    def plot_clf_boundaries(X, clf, ax=None, cmap=None):
        return SSROQutrit.plot_clf_boundaries(X, clf, ax=ax, cmap=cmap)

    @staticmethod
    def plot_std(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
        return SSROQutrit.plot_std(mean, cov, ax,n_std=n_std,
                                   facecolor=facecolor, **kwargs)

    @staticmethod
    def plot_1D_hist(data, y_true=None, plot_fitting=True,
                     **kwargs):
        return SSROQutrit.plot_1D_hist(data, y_true=y_true,
                                       plot_fitting=plot_fitting, **kwargs)

    @staticmethod
    def _order_state_labels(states_labels,
                            order="gefhabcdijklmnopqrtuvwxyz0123456789"):
        """
        Orders state labels according to provided ordering. e.g. for default
        ("f", "e", "g") would become ("g", "e", "f")
        Args:
            states_labels (list, tuple): list of states_labels
            order (str): custom string order

        Returns:

        """
        try:
            indices = [order.index(s) for s in states_labels]
            order_for_states = np.argsort(indices).astype(np.int32)
            return np.array(states_labels)[order_for_states]

        except Exception as e:
            log.error(f"Could not find order in state_labels:"
                      f"{states_labels}. Probably because one or several "
                      f"states are not part of '{order}'. Error: {e}."
                      f" Returning same as input order")
            return states_labels


    def plot(self, **kwargs):
        if not self.get_param_value("plot", True):
            return # no plotting if "plot" is False
        cmap = plt.get_cmap('tab10')
        show = self.options_dict.get("show", False)
        pdd = self.proc_data_dict
        for qbn in self.qb_names:
            n_qb_states = len(np.unique(self.cp.get_states(qbn)[qbn]))
            tab_x = a_tools.truncate_colormap(cmap, 0,
                                              n_qb_states/10)

            kwargs = {
                "states": list(pdd["analysis_params"]['means'][qbn].keys()),
                "xlabel": "Integration Unit 1, $u_1$",
                "ylabel": "Integration Unit 2, $u_2$",
                "scale":self.options_dict.get("hist_scale", "linear"),
                "cmap":tab_x}
            data_keys = [k for k in list(pdd.keys()) if
                            k.startswith("data") and qbn in pdd[k]]

            for dk in data_keys:
                data = pdd[dk][qbn]
                title =  self.raw_data_dict['timestamp'] + f" {qbn} " + dk + \
                    "\n{} classifier".format(self.classif_method)
                kwargs.update(dict(title=title))

                # plot data and histograms
                n_shots_to_plot = self.get_param_value('n_shots_to_plot', None)
                if n_shots_to_plot is not None:
                    n_shots_to_plot *= n_qb_states
                if data['X'].shape[1] == 1:
                    if self.classif_method == "gmm":
                        kwargs['means'] = pdd['analysis_params']['means'][qbn]
                        kwargs['std'] = np.sqrt(self._get_covariances(self.clf_[qbn]))
                    kwargs['colors'] = cmap(np.unique(data['prep_states']))
                    fig, main_ax = self.plot_1D_hist(data['X'][:n_shots_to_plot],
                                            data["prep_states"][:n_shots_to_plot],
                                            **kwargs)
                else:
                    fig = self.plot_scatter_and_marginal_hist(
                        data['X'][:n_shots_to_plot],
                        data["prep_states"][:n_shots_to_plot],
                        **kwargs)

                    # plot clf_boundaries
                    main_ax = fig.get_axes()[0]
                    self.plot_clf_boundaries(data['X'], self.clf_[qbn], ax=main_ax,
                                             cmap=tab_x)
                    # plot means and std dev
                    means = pdd['analysis_params']['means'][qbn]
                    try:
                        covs = self._get_covariances(self.clf_[qbn])
                    except Exception as e: # not a gmm model--> no cov.
                        covs = []

                    for i, mean in enumerate(means.values()):
                        main_ax.scatter(mean[0], mean[1], color='w', s=80)
                        if len(covs) != 0:
                            self.plot_std(mean, covs[i],
                                          n_std=1, ax=main_ax,
                                          edgecolor='w', linestyle='--',
                                          linewidth=1)

                # plot thresholds and mapping
                plt_fn = {0: main_ax.axvline, 1: main_ax.axhline}
                thresholds = pdd['analysis_params'][
                    'classifier_params'][qbn].get("thresholds", dict())
                mapping = pdd['analysis_params'][
                    'classifier_params'][qbn].get("mapping", dict())
                for k, thres in thresholds.items():
                    plt_fn[k](thres, linewidth=2,
                              label="threshold i.u. {}: {:.5f}".format(k, thres),
                              color='k', linestyle="--")
                    main_ax.legend(loc=[0.2,-0.62])

                ax_frac = {0: (0.07, 0.1), # locations for codewords
                           1: (0.83, 0.1),
                           2: (0.07, 0.9),
                           3: (0.83, 0.9)}
                for cw, state in mapping.items():
                    main_ax.annotate("0b{:02b}".format(cw) + f":{state}",
                                     ax_frac[cw], xycoords='axes fraction')

                self.figs[f'{qbn}_{self.classif_method}_classifier_{dk}'] = fig
            if show:
                plt.show()

            # state assignment prob matrix
            title = self.raw_data_dict['timestamp'] + "\n{} State Assignment" \
                " Probability Matrix\nTotal # shots:{}"\
                .format(self.classif_method,
                        self.proc_data_dict['analysis_params']['n_shots'])
            fig = self.plot_fidelity_matrix(
                self.proc_data_dict['analysis_params']['state_prob_mtx'][qbn],
                self._order_state_labels(kwargs['states']),
                title=title,
                show=show,
                auto_shot_info=False)
            self.figs[f'{qbn}_state_prob_matrix_{self.classif_method}'] = fig

            if self.preselection and \
                    len(pdd['analysis_params']['state_prob_mtx_masked'][qbn]) != 0:
                title = self.raw_data_dict['timestamp'] + \
                    "\n{} State Assignment Probability Matrix Masked"\
                    "\nTotal # shots:{}".format(
                        self.classif_method,
                        self.proc_data_dict['analysis_params']['n_shots_masked'][qbn])

                fig = self.plot_fidelity_matrix(
                    pdd['analysis_params']['state_prob_mtx_masked'][qbn],
                    self._order_state_labels(kwargs['states']),
                    title=title, show=show, auto_shot_info=False)
                fig_key = f'{qbn}_state_prob_matrix_masked_{self.classif_method}'
                self.figs[fig_key] = fig
