import logging
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

import lmfit
import numpy as np
import scipy as sp
import itertools
import matplotlib as mpl
from collections import OrderedDict
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.readout_analysis as roa
from pycqed.analysis_v3 import fitting_and_plotting as fit_plt
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from sklearn.mixture import GaussianMixture as GM
from copy import deepcopy

from pycqed.analysis import fitting_models as fit_mods
from pycqed.measurement.calibration_points import CalibrationPoints


def get_data_to_process(data_dict, data_keys_in):
    """
    Finds data to be processed in unproc_data_dict based on data_keys_in.

    :param unproc_data_dict: OrderedDict containing data to be processed
    :param data_keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: measured_data.raw w0.
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in}
    """
    # if data_keys_in is None:
    #     if 'measured_data' in data_dict:
    #         data_to_proc_dict = data_dict['measured_data']
    #     else:
    #         data_to_proc_dict = list(data_dict.values())[-1]
    # else:
    data_to_proc_dict = OrderedDict()
    key_found = True
    for keyi in data_keys_in:
        all_keys = keyi.split('.')
        if len(all_keys) == 1:
            try:
                if isinstance(data_dict[all_keys[0]], dict):
                    data_to_proc_dict.update(data_dict[all_keys[0]])
                else:
                    data_to_proc_dict[keyi] = data_dict[all_keys[0]]
            except KeyError:
                try:
                    data_to_proc_dict[keyi] = data_dict[
                        'measured_data'][keyi]
                except KeyError:
                    key_found = False
        else:
            try:
                data = deepcopy(data_dict)
                for k in all_keys:
                    data = data[k]
                if isinstance(data, dict):
                    data_to_proc_dict.update({k: data[k] for k in data})
                else:
                    data_to_proc_dict[all_keys[-1]] = data
            except KeyError:
                key_found = False
        if not key_found:
            raise ValueError(f'Channel {keyi} was not found.')
    return data_to_proc_dict


def get_param(name, data_dict, default_value=None, **func_pars):
    value = func_pars.get(name,
                          data_dict.get('exp_metadata',
                                        dict()).get(name,
                                                    default_value))
    return value


def filter(data_dict, data_keys_in, data_keys_out, **params):
    """
    Filters data in raw_data_dict for each ch_in according to data_filter
    in params. Puts the filtered data in ch_out

    To be used for example for filtering:
        - reset readouts
        - data with and without flux pulse/ pi pulse etc.

    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param data_keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param data_keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments:
        data_filter (str, default: 'lambda x: x'): filtering condition passed
            as a string that will be evaluated with eval.

    Assumptions:
        - len(data_keys_out) == len(data_keys_in)
    """
    data_to_proc_dict = get_data_to_process(data_dict, data_keys_in)
    if len(data_keys_out) != len(data_to_proc_dict):
        raise ValueError('data_keys_out and data_keys_in do not have '
                         'the same length.')
    data_filter_func = get_param('data_filter', data_dict,
                                  default_value=lambda data: data, **params)
    for keyo, keyi in zip(data_keys_out, list(data_to_proc_dict)):
        data = data_dict
        all_keys = keyo.split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]
        data[all_keys[-1]] = eval(data_filter_func)(data_to_proc_dict[keyi])
    return data_dict


def classify_gm(data_dict, data_keys_in, data_keys_out, **params):
    """
    BROKEN
    TODO: need to correctly handle channel tuples

    Predict gaussian mixture posterior probabilities for single shots
    of different levels of a qudit. Data to be classified expected in the
    shape (n_datapoints, n_channels).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param data_keys_in:
                    qubit: list of key names or dictionary keys paths
                    qutrit: list of tuples of key names or dictionary keys paths
                        in data_dict for the data to be processed
    :param data_keys_out: list of tuples of key names or dictionary keys paths
                        in data_dict for the processed data to be saved into
    :param params: keyword arguments:
        clf_params: list of dictionaries with parameters for
            Gaussian Mixture classifier:
                means_: array of means of each component of the GM
                covariances_: covariance matrix
                covariance_type: type of covariance matrix
                weights_: array of priors of being in each level. (n_levels,)
                precisions_cholesky_: array of precision_cholesky
            For more info see about parameters see :
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.
            GaussianMixture.html
    For each item in data_keys_out, stores int data_dict an
    (n_datapoints, n_levels) array of posterior probability of being
    in each level.

    Assumptions:
        - data_keys_in is a list of tuples for qutrit and
            list of strings for qubit
        - data_keys_out is a list of tuples
        - len(data_keys_out) == len(data_keys_in) + 1
        - clf_params exist in **params
    """
    pass
    # clf_params = get_param('clf_params', data_dict, **params)
    # if clf_params is None:
    #     raise ValueError('clf_params is not specified.')
    # if len(data_keys_out) != len(data_keys_in) + 1:
    #     raise ValueError('Condition len(data_keys_out) == len(data_keys_in) + 1 '
    #                      'is not satisfied.')
    #
    # reqs_params = ['means_', 'covariances_', 'covariance_type',
    #                'weights_', 'precisions_cholesky_']
    # for k, key_tup_in in enumerate(data_keys_in):
    #     data_to_proc_dict = get_data_to_process(data_dict, key_tup_in)
    #
    #     data = data_dict
    #     all_keys = data_keys_out[k].split('.')
    #     for i in range(len(all_keys)-1):
    #         if all_keys[i] not in data:
    #             data[all_keys[i]] = OrderedDict()
    #         else:
    #             data = data[all_keys[i]]
    #
    #     clf_params_temp = deepcopy(clf_params[k])
    #     for r in reqs_params:
    #         assert r in clf_params_temp, "Required Classifier parameter {} " \
    #                                      "not given.".format(r)
    #     gm = GM(covariance_type=clf_params_temp.pop('covariance_type'))
    #     for param_name, param_value in clf_params_temp.items():
    #         setattr(gm, param_name, param_value)
    #     data[all_keys[-1]] = gm.predict_proba(data_to_proc_dict[keyi])
    # return data_dict


def do_preselection(data_dict, classified_data, data_keys_out, **params):
    """
    Keeps only the data for which the preselection readout data in
    classified_data satisfies the preselection condition.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param classified_data: list of arrays of 0,1 for qubit, and
                    0,1,2 for qutrit, or list of keys pointing to the binary
                    (or trinary) arrays in the data_dict
    :param data_keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        presel_ro_idxs (function, default: lambda idx: idx % 2 == 0):
            specifies which (classified) data entry is a preselection ro
        data_keys_in (list): list of key names or dictionary keys paths in
            data_dict for the data to be processed
        presel_condition (int, default: 0): 0, 1 (, or 2 for qutrit). Keeps
            data for which the data in classified data corresponding to
            preselection readouts satisfies presel_condition.

    Assumptions:
        - len(data_keys_out) == len(classified_data)
        - if data_keys_in are given, len(data_keys_in) == len(classified_data)
        - classified_data either list of arrays or list of strings
        - if classified_data contains strings, assumes they are keys in
            data_dict
    """
    if len(data_keys_out) != len(classified_data):
        raise ValueError('classified_data and data_keys_out do not have '
                         'the same length.')

    data_keys_in = params.get('data_keys_in', None)
    presel_ro_idxs = get_param('presel_ro_idxs', data_dict,
                               default_value=lambda idx: idx % 2 == 0, **params)
    presel_condition = get_param('presel_condition', data_dict,
                                 default_value=0, **params)
    if data_keys_in is not None:
        if len(data_keys_in) != len(classified_data):
            raise ValueError('classified_data and data_keys_in do not have '
                             'the same length.')
        data_to_proc_dict = get_data_to_process(data_dict, data_keys_in)
        for i, keyi in enumerate(data_to_proc_dict):
            # Check if the entry in classified_data is an array or a string
            # denoting a key in the data_dict
            if isinstance(classified_data[i], str):
                if classified_data[i] in data_dict:
                    classif_data = data_dict[classified_data[i]]
                else:
                    raise KeyError(
                        f'{classified_data[i]} not found in data_dict.')
            else:
                classif_data = classified_data[i]

            mask = np.zeros(len(data_to_proc_dict[keyi]))
            val = True
            for idx in range(len(data_to_proc_dict[keyi])):
                if presel_ro_idxs(idx):
                    val = (classif_data[idx] == presel_condition)
                    mask[idx] = False
                else:
                    mask[idx] = val
            preselected_data = data_to_proc_dict[keyi][mask]
            data = data_dict
            all_keys = data_keys_out[i].split('.')
            for k in range(len(all_keys)-1):
                data[all_keys[k]] = OrderedDict()
                data = data[all_keys[k]]
            data[all_keys[-1]] = preselected_data
    else:
        for i, keyo in enumerate(data_keys_out):
            # Check if the entry in classified_data is an array or a string
            # denoting a key in the data_dict
            if isinstance(classified_data[i], str):
                if classified_data[i] in data_dict:
                    classif_data = data_dict[classified_data[i]]
                else:
                    raise KeyError(
                        f'{classified_data[i]} not found in data_dict.')
            else:
                classif_data = classified_data[i]

            mask = np.zeros(len(classif_data))
            val = True
            for idx in range(len(classif_data)):
                if presel_ro_idxs(idx):
                    val = (classif_data[idx] == 0)
                    mask[idx] = False
                else:
                    mask[idx] = val
            data_dict[keyo] = classif_data[mask]
    return data_dict


def average(data_dict, data_keys_in, data_keys_out, **params):
    """
    Averages data in data_dict specified by data_keys_in into num_avg_bins.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param data_keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param data_keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        num_avg_bins (list): list with number of averaging bins for each entry
            in data_keys_in

    Assumptions:
        - len(data_keys_out) == len(data_to_proc_dict)
        - num_avg_bins exists in params
        - num_avg_bins[i] exactly divides data_dict[data_keys_in[i]]
        - len(data_keys_in) == len(num_avg_bins)
    """
    num_avg_bins = get_param('num_avg_bins', data_dict, **params)
    if num_avg_bins is None:
        raise ValueError('num_avg_bins is not specified.')
    if len(data_keys_in) != len(num_avg_bins):
        raise ValueError('data_keys_in and num_avg_bins do not have '
                         'the same length.')
    data_to_proc_dict = get_data_to_process(data_dict, data_keys_in)
    if len(data_keys_out) != len(data_to_proc_dict):
        raise ValueError('data_keys_out and data_keys_in do not have '
                         'the same length.')
    for k, keyi in enumerate(data_to_proc_dict):
        if len(data_to_proc_dict[keyi]) % num_avg_bins[k] == 0:
            raise ValueError(f'{num_avg_bins[k]} does not exactly divide '
                             f'len(data_dict[{keyi}]).')
        data = data_dict
        all_keys = data_keys_out[k].split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]
        averages = len(data_to_proc_dict[keyi]) // num_avg_bins[k]
        data[all_keys[-1]] = np.mean(np.reshape(
            data_to_proc_dict[keyi], (averages, num_avg_bins[k])), axis=0)
    return data_dict


def get_qb_channel_map_from_file(data_dict, data_keys, **params):
    file_type = params.get('file_type', 'hdf')
    qb_names = get_param('qb_names', data_dict, **params)
    if qb_names is None:
        raise ValueError('Either channel_map or qb_names must be specified.')

    folder = get_param('folder', data_dict, **params)
    if folder is None:
        if 'folder' in data_dict:
            folder = data_dict['folder']
        else:
            raise ValueError('Path to file must be saved in '
                             'data_dict[folder] in order to extract '
                             'channel_map.')

    if file_type == 'hdf':
        qb_channel_map = a_tools.get_qb_channel_map_from_hdf(
            qb_names, value_names=data_keys, file_path=folder)
    else:
        raise ValueError('Only "hdf" files supported at the moment.')
    return qb_channel_map


def rotate_iq(data_dict, data_keys_in, data_keys_out, **params):
    """
    Rotates IQ data based on information in the CalibrationPoints objects.
    The number of CalibrationPoints objects should equal the number of
    tuples in data_keys_in.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param data_keys_in: list of length-2 tuples of key names or dictionary
                keys paths in data_dict for the data to be processed
    :param data_keys_out: list of key names or dictionary keys paths in
                data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        cal_points_list (list): list of CalibrationPoints objects.
        last_ge_pulses (list): list of booleans
        qb_value_names_map (dict): {qbn: [value_names]}.

    Assumptions:
        - data_keys_in is a list of tuples; each tuple has length 2 (IQ data)
        - one output key per keys in tuple
        - len(data_keys_in) == len(data_keys_out)
        - cal_points_list exists in **params
        - len(cp_list) == len(data_keys_in)
        - one CalibrationPoints object per keys in tuple
        - assumes the dicts returned by CalibrationPoints.get_indices(),
        CalibrationPoints.get_rotations() are keyed by qubit names
        - data_keys_in exists in qb_value_names_map
    """
    if len(data_keys_in) != len(data_keys_out):
        raise ValueError('data_keys_in and data_keys_out do not have '
                         'the same length.')

    cp_list = get_param('cal_points_list', data_dict, **params)
    if cp_list is not None:
        cp_list = [eval(cp) for cp in cp_list]
    else:
        raise ValueError('cal_points_list not found.')
    if len(cp_list) != len(data_keys_in):
        raise ValueError('cal_points_list and data_keys_in do not have '
                         'the same length.')

    last_ge_pulses = get_param('last_ge_pulses', data_dict, default_value=[],
                               **params)
    qb_value_names_map = get_param('qb_value_names_map', data_dict, **params)
    if qb_channel_map is None:
        raise ValueError('Unknown qb_value_names_map.')

    for j, cp in enumerate(cp_list):
        data_to_proc_dict = get_data_to_process(data_dict, data_keys_in[j])

        data = data_dict
        all_keys = data_keys_out[j].split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]

        qbn = [k for k, v in qb_value_names_map.items() if
               data_keys_in[j] == v][0]
        rotations = cp.get_rotations(last_ge_pulses=last_ge_pulses)
        ordered_cal_states = []
        for ii in range(len(rotations[qbn])):
            ordered_cal_states += \
                [k for k, idx in rotations[qbn].items() if idx == ii]
        rotated_data, _, _ = \
            a_tools.rotate_and_normalize_data_IQ(
                data=list(data_to_proc_dict.values()),
                cal_zero_points=None if len(ordered_cal_states) == 0 else
                    cp.get_indices()[qbn][ordered_cal_states[0]],
                cal_one_points=None if len(ordered_cal_states) == 0 else
                    cp.get_indices()[qbn][ordered_cal_states[0]])
        data[all_keys[-1]+'_msmt'] = cp.get_msmt_array(rotated_data, qbn)
        data[all_keys[-1]+'_cal'] = cp.get_cal_array(rotated_data, qbn)
    return data_dict


def rotate_1d_array(data_dict, data_keys_in, data_keys_out, **params):
    """
    Rotates 1d array based on information in the CalibrationPoints objects.
    The number of CalibrationPoints objects should equal the number of
    key names in data_keys_in.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param data_keys_in: list of key names or dictionary keys paths in
                data_dict for the data to be processed
    :param data_keys_out: list of key names or dictionary keys paths in
                data_dict for the processed data to be saved into
    :param params: keyword arguments.:
        cal_points_list (list): list of CalibrationPoints objects.
        last_ge_pulses (list): list of booleans

    Assumptions:
        - one output key per input key
        - len(data_keys_in) == len(data_keys_out)
        - cal_points_list exists in **params
        - len(cp_list) == len(data_keys_in)
        - one CalibrationPoints object input key
        - assumes the dicts returned by CalibrationPoints.get_indices(),
        CalibrationPoints.get_rotations() are keyed by channel number strings;
        ex: indices for ch 0: {'0': {'g': [-4, -3], 'e': [-2, -1]}}
    """
    cp_list = get_param('cal_points_list', data_dict, **params)
    if cp_list is not None:
        cp_list = [eval(cp) for cp in cp_list]
    else:
        raise ValueError('cal_points object not found.')

    last_ge_pulses = get_param('last_ge_pulses', data_dict, default_value=[],
                               **params)
    qb_value_names_map = get_param('qb_value_names_map', data_dict, **params)
    if qb_value_names_map is None:
        raise ValueError('Unknown qb_value_names_map.')

    data_to_proc_dict = get_data_to_process(data_dict, data_keys_in)
    if len(data_keys_out) != len(data_to_proc_dict):
        raise ValueError('data_keys_out and data_keys_in do not have '
                         'the same length.')
    if len(cp_list) != len(data_to_proc_dict):
        raise ValueError('cal_points_list and data_keys_in do not have '
                         'the same length.')

    for j, keyi in enumerate(data_to_proc_dict):
        data = data_dict
        all_keys = data_keys_out[j].split('.')
        for i in range(len(all_keys)-1):
            if all_keys[i] not in data:
                data[all_keys[i]] = OrderedDict()
            else:
                data = data[all_keys[i]]

        cp = cp_list[j]
        qbn = [k for k, v in qb_value_names_map.items() if keyi == v][0]
        rotations = cp.get_rotations(last_ge_pulses=last_ge_pulses)
        ordered_cal_states = []
        for ii in range(len(rotations[qbn])):
            ordered_cal_states += \
                [k for k, idx in rotations[qbn].items() if idx == ii]
        rotated_data = \
            a_tools.rotate_and_normalize_data_1ch(
                data=data_to_proc_dict[keyi],
                cal_zero_points=None if len(ordered_cal_states) == 0 else
                    cp.get_indices()[qbn][ordered_cal_states[0]],
                cal_one_points=None if len(ordered_cal_states) == 0 else
                    cp.get_indices()[qbn][ordered_cal_states[0]])
        data[all_keys[-1]+'_msmt'] = cp.get_msmt_array(rotated_data, qbn)
        data[all_keys[-1]+'_cal'] = cp.get_cal_array(rotated_data, qbn)
    return data_dict


## Helper functions ##
def get_msmt_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the measurement
    points (without calibration points data).
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: measured data without calibration points data
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
    return deepcopy(all_data[:-n_cal_pts])


def get_cal_data(all_data, cal_points, qb_name):
    """
    Extracts data points from all_data that correspond to the calibration points
    data.
    :param all_data: array containing both measurement and calibration
                     points data
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    :return: Calibration points data
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
    return deepcopy(all_data[-n_cal_pts:])


def get_cal_sweep_points(sweep_points_array, cal_points, qb_name):
    """
    Creates the sweep points corresponding to the calibration points data as
    equally spaced number_of_cal_states points, with the spacing given by the
    spacing in sweep_points_array.
    :param sweep_points_array: array of physical sweep points
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    """
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
    step = np.abs(sweep_points_array[-1] - sweep_points_array[-2])
    return np.array([sweep_points_array[-1] + i * step for
                     i in range(1, n_cal_pts + 1)])


## Plotting nodes ##
def get_cal_state_color(cal_state_label):
    if cal_state_label == 'g' or cal_state_label == r'$|g\rangle$':
        return 'k'
    elif cal_state_label == 'e' or cal_state_label == r'$|e\rangle$':
        return 'gray'
    elif cal_state_label == 'f' or cal_state_label == r'$|f\rangle$':
        return 'C8'
    else:
        return 'C4'


def get_latex_prob_label(prob_label):
    if '$' in prob_label:
        return prob_label
    elif 'p' in prob_label.lower():
        return r'$|{}\rangle$'.format(prob_label[prob_label.index('p')+1])
    else:
        return r'$|{}\rangle$'.format(prob_label)


def prepare_1d_plot(data_dict, data_keys_in, fig_name, **params):

    data_to_proc_dict = get_data_to_process(data_dict, data_keys_in)

    data_axis_label = params.get('data_axis_label', None)
    data_label = params.get('data_label', 'Data')
    do_legend_data = params.get('do_legend_data', True)
    do_legend_cal_states = params.get('do_legend_cal_states', True)
    plot_name_suffix = params.get('plot_name_suffix', '')
    title_suffix = params.get('title_suffix', '')

    plotsize = fit_plt.get_default_plot_params(set=False)['figure.figsize']
    plotsize = (plotsize[0], plotsize[0]/1.25)

    cp = get_param('cal_points', data_dict, **params)
    if cp is not None:
        cp = eval(cp)
        qb_name = params.get('qb_name', None)
        if qb_name is None:
            raise ValueError('Unknwn qb_name.')
        title_suffix = qb_name + title_suffix
    physical_swpts = params.get('physical_swpts', None)
    if physical_swpts is None:
        raise ValueError('physical_swpts not found.')

    plot_dicts = OrderedDict()
    plot_names_cal = []
    for i, keyi in enumerate(data_to_proc_dict):
        data = data_to_proc_dict[keyi]
        if cp is not None:
            cal_data = get_cal_data(data, cp, qb_name)
            data = get_msmt_data(data, cp, qb_name)
            cal_swpts = get_cal_sweep_points(physical_swpts, cp, qb_name)

            qb_cal_indxs = cp.get_indices()[qb_name]
            # plot cal points
            for ii, cal_pts_idxs in enumerate(qb_cal_indxs.values()):
                plot_dict_name_cal = list(qb_cal_indxs)[ii] + \
                                     '_' + qb_name + '_' + plot_name_suffix
                plot_names_cal += [plot_dict_name_cal]
                plot_dicts[plot_dict_name_cal] = {
                    'fig_id': fig_name,
                    'plotfn': 'plot_line',
                    'plotsize': plotsize,
                    'xvals': cal_swpts[cal_pts_idxs],
                    'yvals': cal_data[cal_pts_idxs],
                    'setlabel': list(qb_cal_indxs)[ii],
                    'do_legend': do_legend_cal_states,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'linestyle': 'none',
                    'line_kws': {'color': get_cal_state_color(
                        list(qb_cal_indxs)[ii])}}

                plot_dicts[plot_dict_name_cal+'_line'] = {
                    'fig_id': fig_name,
                    'plotsize': plotsize,
                    'plotfn': 'plot_hlines',
                    'y': np.mean(cal_data[cal_pts_idxs]),
                    'xmin': physical_swpts[0],
                    'xmax': cal_swpts[-1],
                    'colors': 'gray'}

        title = (data_dict['timestamp'] + ' ' + data_dict['measurementstring'])
        if title_suffix is not None:
            title += '\n' + title_suffix

        plot_dict_name = fig_name + '_' + plot_name_suffix
        # get x info (try for old sequences which do not have info in metadata
        hard_sweep_params = get_param('hard_sweep_params', data_dict)
        sweep_name = get_param('sweep_name', data_dict)
        sweep_unit = get_param('sweep_unit', data_dict)
        if hard_sweep_params is not None:
            xlabel = list(hard_sweep_params)[0]
            xunit = list(hard_sweep_params.values())[0][
                'unit']
        elif (sweep_name is not None) and (sweep_unit is not None):
            xlabel = sweep_name
            xunit = sweep_unit
        else:
            xlabel = data_dict['sweep_parameter_names']
            xunit = data_dict['sweep_parameter_units']
        if np.ndim(xunit) > 0:
            xunit = xunit[0]

        if data_axis_label is None:
            data_axis_label = '{} state population'.format(
                get_latex_prob_label(keyi))

        plot_dicts[plot_dict_name] = {
            'plotfn': 'plot_line',
            'fig_id': fig_name,
            'plotsize': plotsize,
            'xvals': physical_swpts,
            'xlabel': xlabel,
            'xunit': xunit,
            'yvals': data,
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
                plot_dict_cal = plot_dicts.pop(plot_name)
                plot_dicts[plot_name] = plot_dict_cal

    if 'plot_dicts' in data_dict:
        data_dict['plot_dicts'].update(plot_dicts)
    else:
        data_dict['plot_dicts'] = plot_dicts


## Nodes that are classes ##

class RabiAnalysis(object):

    def __init__(self, data_dict, data_keys_in, **params):
        """
        Does Rabi analysis. Prepares fits and plot, and extracts
        pi-pulse and pi-half pulse amplitudes.
        :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
        :param data_keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed

        Assumptions:
            - cal_points, sweep_points, qb_sweep_points_map, qb_name exist in
            metadata or params
            - expects a 1d sweep, ie takes sweep_points[0][
            qb_sweep_points_map[qb_name]][0] as sweep points
        """
        self.data_dict = data_dict
        self.data_keys_in = data_keys_in
        self.data_to_proc_dict = get_data_to_process(self.data_dict,
                                                     self.data_keys_in)

        if params.get('auto', True):
            self.process_data(**params)
            if params.get('do_fitting', True):
                self.prepare_fitting()
                getattr(fit_plt, 'run_fitting')(self.data_dict, **params)
                self.analyze_fit_results()
            if params.get('prepare_plots', True):
                self.prepare_plots()
            if params.get('do_plotting', True):
                getattr(fit_plt, 'plot')(self.data_dict, **params)

    def __call__(self, *args, **kwargs):
        return self.data_dict

    def process_data(self, **params):
        self.qb_name = get_param('qubit_name', self.data_dict, **params)

        self.cp = get_param('cal_points', self.data_dict, **params)
        if self.cp is not None:
            self.cp = eval(self.cp)
        else:
            raise ValueError('cal_points not found.')

        self.sp = get_param('sweep_points', self.data_dict, **params)
        if self.sp is None:
            raise ValueError('sweep_points not found.')
        if isinstance(self.sp, str):
            self.sp = eval(self.sp)

        self.qb_sweep_points_map = get_param('qb_sweep_points_map',
                                             self.data_dict, **params)
        if self.qb_sweep_points_map is None:
            raise ValueError('qb_sweep_points_map not found.')
        self.physical_swpts = self.sp[0][self.qb_sweep_points_map[
            self.qb_name][0]][0]

        if 'folder' in self.data_dict:
            self.folder = self.data_dict['folder']
        else:
            self.folder = get_param('folder', self.data_dict, **params)
        if self.folder is None:
            log.warning(r'Unspecified folder. Old $\pi$- and $\pi/2$-pulse '
                        r'amplitudes will not be extracted.')

    def prepare_fitting(self):
        fit_dicts = OrderedDict()
        for keyi, data in self.data_to_proc_dict.items():
            data_fit = get_msmt_data(data, self.cp, self.qb_name)
            cos_mod = lmfit.Model(fit_mods.CosFunc)
            guess_pars = fit_mods.Cos_guess(
                model=cos_mod, t=self.physical_swpts, data=data_fit)
            guess_pars['amplitude'].vary = True
            guess_pars['amplitude'].min = -10
            guess_pars['offset'].vary = True
            guess_pars['frequency'].vary = True
            guess_pars['phase'].vary = True

            key = 'cos_fit_' + self.qb_name + keyi
            fit_dicts[key] = {
                'fit_fn': fit_mods.CosFunc,
                'fit_xvals': {'t': self.physical_swpts},
                'fit_yvals': {'data': data_fit},
                'guess_pars': guess_pars}

        if 'fit_dicts' in self.data_dict:
            self.data_dict['fit_dicts'].update(fit_dicts)
        else:
            self.data_dict['fit_dicts'] = fit_dicts

    def analyze_fit_results(self):
        if 'fit_dicts' in self.data_dict:
            fit_dicts = self.data_dict['fit_dicts']
        else:
            raise KeyError('data_dict does not contain fit_dicts.')
        rabi_amplitudes = OrderedDict()
        for keyi in self.data_to_proc_dict:
            fit_res = fit_dicts['cos_fit_' + self.qb_name + keyi]['fit_res']
            rabi_amplitudes[self.qb_name] = self.get_amplitudes(
                fit_res=fit_res, sweep_points=self.physical_swpts)

        if 'analysis_params_dict' in self.data_dict:
            self.data_dict['analysis_params_dict'].update(rabi_amplitudes)
        else:
            self.data_dict['analysis_params_dict'] = rabi_amplitudes

    def prepare_plots(self):
        if 'fit_dicts' in self.data_dict:
            fit_dicts = self.data_dict['fit_dicts']
        else:
            raise KeyError('data_dict does not contain fit_dicts.')
        plot_dicts = OrderedDict()
        for keyi, data in self.data_to_proc_dict.items():
            base_plot_name = 'Rabi_' + self.qb_name + keyi
            prepare_1d_plot(
                data_dict=self.data_dict,
                data_keys_in=[keyi],
                physical_swpts=self.physical_swpts,
                fig_name=base_plot_name,
                data=data,
                plot_name_suffix=self.qb_name+'fit',
                qb_name=self.qb_name)

            fit_res = fit_dicts['cos_fit_' + self.qb_name + keyi]['fit_res']
            plot_dicts['fit_' + self.qb_name + keyi] = {
                'fig_id': base_plot_name,
                'plotfn': 'plot_fit',
                'fit_res': fit_res,
                'setlabel': 'cosine fit',
                'color': 'r',
                'do_legend': True,
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1, -0.15),
                'legend_pos': 'upper right'}

            rabi_amplitudes = self.data_dict['analysis_params_dict']
            plot_dicts['piamp_marker_' + self.qb_name + keyi] = {
                'fig_id': base_plot_name,
                'plotfn': 'plot_line',
                'xvals': np.array([rabi_amplitudes[self.qb_name]['piPulse']]),
                'yvals': np.array([fit_res.model.func(
                    rabi_amplitudes[self.qb_name]['piPulse'],
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

            plot_dicts['piamp_hline_' + self.qb_name + keyi] = {
                'fig_id': base_plot_name,
                'plotfn': 'plot_hlines',
                'y': [fit_res.model.func(
                    rabi_amplitudes[self.qb_name]['piPulse'],
                    **fit_res.best_values)],
                'xmin': self.physical_swpts[0],
                'xmax': get_cal_sweep_points(self.physical_swpts, self.cp,
                                             self.qb_name)[-1],
                'colors': 'gray'}

            plot_dicts['pihalfamp_marker_' + self.qb_name + keyi] = {
                'fig_id': base_plot_name,
                'plotfn': 'plot_line',
                'xvals': np.array([rabi_amplitudes[self.qb_name]['piHalfPulse']]),
                'yvals': np.array([fit_res.model.func(
                    rabi_amplitudes[self.qb_name]['piHalfPulse'],
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

            plot_dicts['pihalfamp_hline_' + self.qb_name + keyi] = {
                'fig_id': base_plot_name,
                'plotfn': 'plot_hlines',
                'y': [fit_res.model.func(
                    rabi_amplitudes[self.qb_name]['piHalfPulse'],
                    **fit_res.best_values)],
                'xmin': self.physical_swpts[0],
                'xmax': get_cal_sweep_points(self.physical_swpts, self.cp,
                                             self.qb_name)[-1],
                'colors': 'gray'}

            if self.folder is not None:
                old_pipulse_val = a_tools.get_param_value_from_file(
                    file_path=self.folder, instr_name=self.qb_name,
                    param_name='{}_amp180'.format(
                        'ef' if 'f' in keyi else 'ge'))
                old_pihalfpulse_val = old_pipulse_val * \
                    a_tools.get_param_value_from_file(
                      file_path=self.data_dict['folder'],
                      instr_name=self.qb_name, param_name='{}_amp90_scale'.format(
                          'ef' if 'f' in keyi else 'ge'))

                textstr = ('  $\pi-Amp$ = {:.3f} V'.format(
                    rabi_amplitudes[self.qb_name]['piPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                               rabi_amplitudes[self.qb_name]['piPulse_stderr']) +
                           '\n$\pi/2-Amp$ = {:.3f} V '.format(
                               rabi_amplitudes[self.qb_name]['piHalfPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                               rabi_amplitudes[self.qb_name]['piHalfPulse_stderr']) +
                           '\n  $\pi-Amp_{old}$ = ' + '{:.3f} V '.format(
                            old_pipulse_val) +
                           '\n$\pi/2-Amp_{old}$ = ' + '{:.3f} V '.format(
                            old_pihalfpulse_val))
                plot_dicts['text_msg_' + self.qb_name] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': -0.05,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': 'plot_text',
                    'text_string': textstr}

        if 'plot_dicts' in self.data_dict:
            self.data_dict['plot_dicts'].update(plot_dicts)
        else:
            self.data_dict['plot_dicts'] = plot_dicts

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
            print(e)
            piPulse_std = 0
            piHalfPulse_std = 0

        rabi_amplitudes = {'piPulse': piPulse,
                           'piPulse_stderr': piPulse_std,
                           'piHalfPulse': piHalfPulse,
                           'piHalfPulse_stderr': piHalfPulse_std}

        return rabi_amplitudes

    @staticmethod
    def calculate_pulse_stderr(f, phi, f_err, phi_err,
                               period_num, cov=0):
        x = period_num + phi
        return np.sqrt((f_err*x/(2*np.pi*(f**2)))**2 +
                       (phi_err/(2*np.pi*f))**2 -
                       2*(cov**2)*x/((2*np.pi*(f**3))**2))[0]


