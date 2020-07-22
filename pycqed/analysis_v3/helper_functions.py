import logging
log = logging.getLogger(__name__)
import re
import os
import h5py
import itertools
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from more_itertools import unique_everseen
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.hdf5_data import read_dict_from_hdf5
from pycqed.measurement.calibration_points import CalibrationPoints


def get_hdf_param_value(group, param_name):
    '''
    Returns an attribute "key" of the group "Experimental Data"
    in the hdf5 datafile.
    '''
    s = group.attrs[param_name]
    # converts byte type to string because of h5py datasaving
    if type(s) == bytes:
        s = s.decode('utf-8')
    # If it is an array of value decodes individual entries
    if type(s) == np.ndarray:
        s = [s.decode('utf-8') for s in s]
    try:
        return eval(s)
    except Exception:
        return s


def get_value_names_from_timestamp(timestamp):
    """
    Returns value_names from the HDF5 file specified by timestamp.
    :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
    :return: list of value_names
    """
    folder = a_tools.get_folder(timestamp)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, 'r+')
    try:
        channel_names = get_hdf_param_value(data_file['Experimental Data'],
                                            'value_names')
        data_file.close()
        return channel_names
    except Exception as e:
        data_file.close()
        raise e


def get_param_from_metadata_group(timestamp=None, param_name=None,
                                  data_file=None, close_file=True):
    """
    Get a parameter with param_name from the Experimental Metadata group in
    the HDF5 file specified by timestamp, or return the whole group if
    param_name is None.
    :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
    :param param_name: (str) name of a key in Experimental Metadata group
    :param data_file: (HDF file) opened HDF5 file
    :param close_file: (bool) whether to close the HDF5 file
    :return: the value of the param_name or the whole experimental metadata
    dictionary
    """
    if data_file is None:
        if timestamp is None:
            raise ValueError('Please provide either timestamp or data_file.')
        folder = a_tools.get_folder(timestamp)
        h5filepath = a_tools.measurement_filename(folder)
        data_file = h5py.File(h5filepath, 'r+')

    try:
        if param_name is None:
            group = data_file['Experimental Data']
            return read_dict_from_hdf5({}, group['Experimental Metadata'])

        group = data_file['Experimental Data']['Experimental Metadata']
        if param_name in group:
            group = group[param_name]
            param_value = OrderedDict()
            param_value = read_dict_from_hdf5(param_value, group)
        elif param_name in group.attrs:
            param_value = get_hdf_param_value(group, param_name)
        else:
            raise KeyError(f'{param_name} was not found in metadata.')
        if close_file:
            data_file.close()
    except Exception as e:
        data_file.close()
        raise e
    return param_value


def get_data_from_hdf_file(timestamp=None, data_file=None,
                           close_file=True):
    """
    Return the measurement data stored in Experimental Data group of the file
    specified by timestamp.
    :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
    :param data_file: (HDF file) opened HDF5 file
    :param close_file: (bool) whether to close the HDF5 file
    :return: numpy array with measurement data
    """
    if data_file is None:
        if timestamp is None:
            raise ValueError('Please provide either timestamp or data_file.')
        folder = a_tools.get_folder(timestamp)
        h5filepath = a_tools.measurement_filename(folder)
        data_file = h5py.File(h5filepath, 'r+')
    try:
        group = data_file['Experimental Data']
        if 'Data' in group:
            dataset = np.array(group['Data'])
        else:
            raise KeyError(f'{Data} was not found in Experimental Data.')
        if close_file:
            data_file.close()
    except Exception as e:
        data_file.close()
        raise e
    return dataset


def open_data_file_from_timestamp(timestamp, mode='r+'):
    """
    Return the opened HDF5 file specified by timestamp.
    ! File is not closed !
    :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
    :param mode: (str) in what mode to open the file
    :return: open HDF5 file
    """
    folder = a_tools.get_folder(timestamp)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, mode)
    return data_file


def get_params_from_hdf_file(data_dict, params_dict=None, numeric_params=None,
                             append_key=False, update_key=False,
                             replace_value=False, folder=None, **params):
    """
    Extracts the parameter provided in params_dict from an HDF file
    and saves them in data_dict.
    :param data_dict: OrderedDict where parameters and their values are saved
    :param params_dict: OrderedDict with key being the parameter name that will
        be used as key in data_dict for this parameter, and value being a
        parameter name or a path + parameter name indie the HDF file.
    :param numeric_params: list of parameter names from amount the keys of
        params_dict. This specifies that those parameters are numbers and will
        be converted to floats.
    :param folder: path to file from which data will be read
    :param params: keyword arguments:
        append_value (bool, default: True): whether to append an
            already-existing key
        update_value (bool, default: False): whether to replace an
            already-existing key
        h5mode (str, default: 'r+'): reading mode of the HDF file
        close_file (bool, default: True): whether to close the HDF file(s)
    """
    if params_dict is None:
        params_dict = get_param('params_dict', data_dict, raise_error=True,
                                **params)
    if numeric_params is None:
        numeric_params = get_param('numeric_params', data_dict,
                                   default_value=[], **params)
    append_value = get_param('append_value', data_dict, default_value=True,
                             **params)
    update_value = get_param('update_value', data_dict, default_value=False,
                             **params)
    if append_value is True and update_value is True:
        raise ValueError('"append_value" and "update_value" '
                         'cannot both be True.')

    # if folder is not specified, will take the last folder in the list from
    # data_dict['folders']
    if folder is None:
        folder = get_param('folders', data_dict, raise_error=True, **params)
        if len(folder) > 0:
            folder = folder[-1]

    h5mode = get_param('h5mode', data_dict, default_value='r+', **params)
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, h5mode)

    try:
        if 'measurementstrings' in params_dict:
            # assumed data_dict['measurementstrings'] is a list
            if 'measurementstrings' in data_dict:
                data_dict['measurementstrings'] += [os.path.split(folder)[1][7:]]
            else:
                data_dict['measurementstrings'] = [os.path.split(folder)[1][7:]]


        for save_par, file_par in params_dict.items():
            epd = data_dict
            all_keys = save_par.split('.')
            for i in range(len(all_keys)-1):
                if all_keys[i] not in epd:
                    epd[all_keys[i]] = OrderedDict()
                else:
                    epd = epd[all_keys[i]]

            if len(file_par.split('.')) == 1:
                par_name = file_par.split('.')[0]
                for group_name in data_file.keys():
                    if par_name in list(data_file[group_name].attrs):
                        add_param(all_keys[-1],
                                  get_hdf_param_value(data_file[group_name],
                                                      par_name),
                                  epd, append_key=append_key,
                                  update_key=update_key,
                                  replace_value=replace_value)
            else:
                group_name = '/'.join(file_par.split('.')[:-1])
                par_name = file_par.split('.')[-1]
                if group_name in data_file:
                    if par_name in list(data_file[group_name].attrs):
                        add_param(all_keys[-1],
                                  get_hdf_param_value(data_file[group_name],
                                                      par_name),
                                  epd, append_key=append_key,
                                  update_key=update_key,
                                  replace_value=replace_value)
                    elif par_name in list(data_file[group_name].keys()):
                        add_param(all_keys[-1],
                                  read_dict_from_hdf5(
                                      {}, data_file[group_name][par_name]),
                                  epd, append_key=append_key,
                                  update_key=update_key,
                                  replace_value=replace_value)

            if all_keys[-1] not in epd:
                log.warning(f'Parameter {file_par} was not found.')
                epd[all_keys[-1]] = 0
        data_file.close()
    except Exception as e:
        data_file.close()
        raise e

    for par_name in data_dict:
        if par_name in numeric_params:
            if hasattr(data_dict[par_name], '__iter__'):
                data_dict[par_name] = [np.float(p) for p
                                       in data_dict[par_name]]
                data_dict[par_name] = np.asarray(data_dict[par_name])
            else:
                data_dict[par_name] = np.float(data_dict[par_name])

    if get_param('close_file', data_dict, default_value=True, **params):
        data_file.close()
    else:
        if 'data_files' in data_dict:
            data_dict['data_files'] += [data_file]
        else:
            data_dict['data_files'] = [data_file]
    return data_dict


def get_data_to_process(data_dict, keys_in):
    """
    Finds data to be processed in unproc_data_dict based on keys_in.

    :param data_dict: OrderedDict containing data to be processed
    :param keys_in: list of channel names or dictionary paths leading to
            data to be processed. For example: raw w1, filtered_data.raw w0
    :return:
        data_to_proc_dict: dictionary {ch_in: data_ch_in}
    """
    data_to_proc_dict = OrderedDict()
    key_found = True
    for keyi in keys_in:
        all_keys = keyi.split('.')
        if len(all_keys) == 1:
            try:
                if isinstance(data_dict[all_keys[0]], dict):
                    data_to_proc_dict = {f'{keyi}.{k}': deepcopy(v) for k, v
                                         in data_dict[all_keys[0]].items()}
                else:
                    data_to_proc_dict[keyi] = data_dict[all_keys[0]]
            except KeyError:
                key_found = False
        else:
            try:
                data = data_dict
                for k in all_keys:
                    data = data[k]
                if isinstance(data, dict):
                    data_to_proc_dict = {f'{keyi}.{k}': deepcopy(data[k])
                                         for k in data}
                else:
                    data_to_proc_dict[keyi] = deepcopy(data)
            except KeyError:
                key_found = False
        if not key_found:
            raise ValueError(f'Channel {keyi} was not found.')
    return data_to_proc_dict


def get_param(param, data_dict, default_value=None,
              raise_error=False, error_message=None, **params):
    """
    Get the value of the parameter "param" from params, data_dict, or metadata.
    :param name: name of the parameter being sought
    :param data_dict: OrderedDict where param is to be searched
    :param default_value: default value for the parameter being sought in case
        it is not found.
    :param raise_error: whether to raise error if the parameter is not found
    :param params: keyword args where parameter is to be sough
    :return: the value of the parameter
    """
    p = params
    md = data_dict.get('exp_metadata', dict())
    dd = data_dict
    value = p.get(param,
                  dd.get(param,
                         md.get(param, 'not found')))

    # the check isinstance(valeu, str) is necessary because if value is an array
    # or list then the check value == 'not found' raises an "elementwise
    # comparison failed" warning in the notebook
    if isinstance(value, str) and value == 'not found':
        all_keys = param.split('.')
        if len(all_keys) > 1:
            for i in range(len(all_keys)-1):
                if all_keys[i] not in p:
                    p[all_keys[i]] = OrderedDict()
                if all_keys[i] not in md:
                    md[all_keys[i]] = OrderedDict()
                if all_keys[i] not in dd:
                    dd[all_keys[i]] = OrderedDict()
                p = p[all_keys[i]]
                md = md[all_keys[i]]
                dd = dd[all_keys[i]]
        value = p.get(all_keys[-1],
                      dd.get(all_keys[-1],
                             md.get(all_keys[-1], default_value)))

    if raise_error and value is None:
        if error_message is None:
            error_message = f'{param} was not found in either data_dict, or ' \
                            f'exp_metadata or input params.'
        raise ValueError(error_message)
    return value


def pop_param(param, data_dict, default_value=None,
              raise_error=False, error_message=None, node_params=None):
    """
    Pop the value of the parameter "param" from params, data_dict, or metadata.
    :param name: name of the parameter being sought
    :param data_dict: OrderedDict where param is to be searched
    :param default_value: default value for the parameter being sought in case
        it is not found.
    :param raise_error: whether to raise error if the parameter is not found
    :param params: keyword args where parameter is to be sough
    :return: the value of the parameter
    """
    if node_params is None:
        node_params = OrderedDict()

    p = node_params
    md = data_dict.get('exp_metadata', dict())
    dd = data_dict
    value = p.pop(param,
                  dd.pop(param,
                         md.pop(param, 'not found')))

    # the check isinstance(valeu, str) is necessary because if value is an array
    # or list then the check value == 'not found' raises an "elementwise
    # comparison failed" warning in the notebook
    if isinstance(value, str) and value == 'not found':
        all_keys = param.split('.')
        if len(all_keys) > 1:
            for i in range(len(all_keys)-1):
                if all_keys[i] not in p:
                    p[all_keys[i]] = OrderedDict()
                if all_keys[i] not in md:
                    md[all_keys[i]] = OrderedDict()
                if all_keys[i] not in dd:
                    dd[all_keys[i]] = OrderedDict()
                p = p[all_keys[i]]
                md = md[all_keys[i]]
                dd = dd[all_keys[i]]
        value = p.pop(all_keys[-1],
                      dd.pop(all_keys[-1],
                             md.pop(all_keys[-1], default_value)))

    if raise_error and value is None:
        if error_message is None:
            error_message = f'{param} was not found in either data_dict, or ' \
                            f'exp_metadata or input params.'
        raise ValueError(error_message)
    return value


def add_param(name, value, data_dict, update_value=False, append_value=False,
              replace_value=False, **params):
    """
    Adds a new key-value pair to the data_dict, with key = name.
    If update, it will try data_dict[name].update(value), else raises KeyError.
    :param name: key of the new parameter in the data_dict
    :param value: value of the new parameter
    :param data_dict: OrderedDict containing data to be processed
    :param update_value: whether to try data_dict[name].update(value).
        Both value and the already-existing entry in data_dict need to be dicts.
    :param append_value: whether to try data_dict[name].extend(value). If either
        value or already-existing entry in data_dict are not lists, they will be
        converted to lists.
    :param replace_value: whether to replaced the already-existing key in
        data_dict
    :param params: keyword arguments

    Assumptions:
        - if update_value == True, both value and the already-existing entry in
            data_dict need to be dicts.
    """
    if any([append_value, update_value]) and replace_value:
        raise ValueError('"replace_value" cannot be True when either '
                         '"append_value" or "update_value" is True.')

    dd = data_dict
    all_keys = name.split('.')
    if len(all_keys) > 1:
        for i in range(len(all_keys)-1):
            if all_keys[i] not in dd:
                dd[all_keys[i]] = OrderedDict()
            dd = dd[all_keys[i]]

    if all_keys[-1] in dd:
        if update_value:
            if not isinstance(value, dict):
                raise ValueError(f'The value corresponding to {all_keys[-1]} '
                                 f'is not a dict. Cannot update_value in '
                                 f'data_dict')
            dd[all_keys[-1]].update(value)
        elif append_value:
            v = dd[all_keys[-1]]
            if not isinstance(v, list):
                dd[all_keys[-1]] = [v]
            else:
                dd[all_keys[-1]] = v
            if not isinstance(value, list):
                dd[all_keys[-1]].extend([value])
            else:
                dd[all_keys[-1]].extend(value)
        elif replace_value:
            dd[all_keys[-1]] = value
        else:
            raise KeyError(f'{all_keys[-1]} already exists in data_dict and it'
                           f' is unclear how to add it to the data_dict.')
    else:
        dd[all_keys[-1]] = value


def get_measurement_properties(data_dict, props_to_extract='all',
                               raise_error=True, **params):
    """
    Extracts the items listed in props_to_extract from experiment metadata
    or from params.
    :param data_dict: OrderedDict containing experiment metadata (exp_metadata)
    :param: props_to_extract: list of items to extract. Can be
        'cp' for CalibrationPoints object
        'sp' for SweepPoints object
        'mospm' for meas_obj_sweep_points_map = {mobjn: [sp_names]}
        'movnm' for meas_obj_value_names_map = {mobjn: [value_names]}
        'rev_movnm' for the reversed_meas_obj_value_names_map =
            {value_name: mobjn}
        'mobjn' for meas_obj_names = the measured objects names
        If 'all', then all of the above are extracted.
    :param params: keyword arguments
        enforce_one_meas_obj (default True): checks if meas_obj_names contains
            more than one element. If True, raises an error, else returns
            meas_obj_names[0].
    :return: cal_points, sweep_points, meas_obj_sweep_points_map and
    meas_obj_names

    Assumptions:
        - if cp or sp are strings, then it assumes they can be evaluated
    """
    if props_to_extract == 'all':
        props_to_extract = ['cp', 'sp', 'mospm', 'movnm', 'rev_movnm', 'mobjn']

    props_to_return = []
    if 'cp' in props_to_extract:
        cp = get_param('cal_points', data_dict, raise_error=raise_error,
                       **params)
        if isinstance(cp, str):
            cp = CalibrationPoints.from_string(cp)
        props_to_return += [cp]

    if 'sp' in props_to_extract:
        sp = get_param('sweep_points', data_dict, raise_error=raise_error,
                       **params)
        if isinstance(sp, str):
            sp = eval(sp)
        props_to_return += [sp]

    if 'mospm' in props_to_extract:
        meas_obj_sweep_points_map = get_param(
            'meas_obj_sweep_points_map', data_dict, raise_error=raise_error,
            **params)
        props_to_return += [meas_obj_sweep_points_map]

    if 'movnm' in props_to_extract:
        meas_obj_value_names_map = get_param(
            'meas_obj_value_names_map', data_dict, raise_error=raise_error,
            **params)
        props_to_return += [meas_obj_value_names_map]

    if 'rev_movnm' in props_to_extract:
        meas_obj_value_names_map = get_param(
            'meas_obj_value_names_map', data_dict, raise_error=raise_error,
            **params)
        rev_movnm = OrderedDict()
        for mobjn, value_names in meas_obj_value_names_map.items():
            rev_movnm.update({vn: mobjn for vn in value_names})
        props_to_return += [rev_movnm]

    if 'mobjn' in props_to_extract:
        mobjn = get_param('meas_obj_names', data_dict,
                          raise_error=raise_error, **params)
        if params.get('enforce_one_meas_obj', True):
            if isinstance(mobjn, list):
                if len(mobjn) > 1:
                    raise ValueError(f'This node expects one measurement '
                                     f'object, {len(mobjn)} were given.')
                else:
                    mobjn = mobjn[0]
        props_to_return += [mobjn]

    if len(props_to_return) == 1:
        props_to_return = props_to_return[0]

    return props_to_return


def get_qb_channel_map_from_file(data_dict, data_keys, **params):
    file_type = params.get('file_type', 'hdf')
    qb_names = get_param('qb_names', data_dict, **params)
    if qb_names is None:
        raise ValueError('Either channel_map or qb_names must be specified.')

    folder = get_param('folders', data_dict, **params)[-1]
    if folder is None:
        raise ValueError('Path to file must be saved in '
                         'data_dict[folders] in order to extract '
                         'channel_map.')

    if file_type == 'hdf':
        qb_channel_map = a_tools.get_qb_channel_map_from_hdf(
            qb_names, value_names=data_keys, file_path=folder)
    else:
        raise ValueError('Only "hdf" files supported at the moment.')
    return qb_channel_map


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
    if cal_points is None:
        return all_data

    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return all_data
        else:
            return deepcopy(all_data[:-n_cal_pts])
    else:
        return all_data


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
    if cal_points is None:
        return np.array([])

    if isinstance(cal_points, str):
        cal_points = repr(cal_points)
    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            return deepcopy(all_data[-n_cal_pts:])
    else:
        return np.array([])


def get_cal_sweep_points(sweep_points_array, cal_points, qb_name):
    """
    Creates the sweep points corresponding to the calibration points data as
    equally spaced number_of_cal_states points, with the spacing given by the
    spacing in sweep_points_array.
    :param sweep_points_array: array of physical sweep points
    :param cal_points: CalibrationPoints instance or its repr
    :param qb_name: qubit name
    """
    if cal_points is None:
        return np.array([])
    if isinstance(cal_points, str):
        cal_points = repr(cal_points)

    if qb_name in cal_points.qb_names:
        n_cal_pts = len(cal_points.get_states(qb_name)[qb_name])
        if n_cal_pts == 0:
            return np.array([])
        else:
            try:
                step = np.abs(sweep_points_array[-1] - sweep_points_array[-2])
            except IndexError:
                # This fallback is used to have a step value in the same order
                # of magnitude as the value of the single sweep point
                step = np.abs(sweep_points_array[0])
            return np.array([sweep_points_array[-1] + i * step for
                             i in range(1, n_cal_pts + 1)])
    else:
        return np.array([])


def get_reset_reps_from_data_dict(data_dict):
    reset_reps = 0
    metadata = data_dict.get('exp_metadata', {})
    if 'preparation_params' in metadata:
        if 'active' in metadata['preparation_params'].get(
                'preparation_type', 'wait'):
            reset_reps = metadata['preparation_params'].get(
                'reset_reps', 0)
    return reset_reps


def get_observables(data_dict, keys_out, preselection_shift=-1,
                    use_preselection=False, **params):

    assert len(keys_out) == 1

    mobj_names = get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
        **params)
    combination_list = list(itertools.product([False, True],
                                              repeat=len(mobj_names)))
    preselection_condition = dict(zip(
        [(qb, preselection_shift) for qb in mobj_names],  # keys contain shift
        combination_list[0]  # first comb has all ground
    ))
    observables = OrderedDict()

    # add preselection condition also as an observable
    if use_preselection:
        observables["pre"] = preselection_condition
    # add all combinations
    for i, states in enumerate(combination_list):
        name = ''.join(['e' if s else 'g' for s in states])
        obs_name = '$\| ' + name + '\\rangle$'
        observables[obs_name] = dict(zip(mobj_names, states))
        # add preselection condition
        if use_preselection:
            observables[obs_name].update(preselection_condition)

    add_param(keys_out[0], observables, data_dict)


### functions that do NOT have the ana_v3 format for input parameters ###

def observable_product(*observables):
    """
    Finds the product-observable of the input observables.
    If the observable conditions are contradicting, returns None. For the
    format of the observables, see the docstring of `probability_table`.
    """
    res_obs = {}
    for obs in observables:
        for k in obs:
            if k in res_obs:
                if obs[k] != res_obs[k]:
                    return None
            else:
                res_obs[k] = obs[k]
    return res_obs


def convert_channel_names_to_index(cal_points, nr_segments, value_names):
    """
    Converts the calibration points list from the format
    cal_points = [{'ch1': [-4, -3], 'ch2': [-4, -3]},
                  {0: [-2, -1], 1: [-2, -1]}]
    to the format (for a 100-segment dataset)
    cal_points_list = [[[96, 97], [96, 97]],
                       [[98, 99], [98, 99]]]

    Args:
        cal_points: the list of calibration points to convert
        nr_segments: number of segments in the dataset to convert negative
                     indices to positive indices.
        value_names: a list of channel names that is used to determine the
                     index of the channels
    Returns:
        cal_points_list in the converted format
    """

    cal_points_list = []
    for observable in cal_points:
        if isinstance(observable, (list, np.ndarray)):
            observable_list = [[]] * len(value_names)
            for i, idxs in enumerate(observable):
                observable_list[i] = \
                    [idx % nr_segments for idx in idxs]
            cal_points_list.append(observable_list)
        else:
            observable_list = [[]] * len(value_names)
            for channel, idxs in observable.items():
                if isinstance(channel, int):
                    observable_list[channel] = \
                        [idx % nr_segments for idx in idxs]
                else:  # assume str
                    ch_idx = value_names.index(channel)
                    observable_list[ch_idx] = \
                        [idx % nr_segments for idx in idxs]
            cal_points_list.append(observable_list)
    return cal_points_list


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
    if 'pg ' in prob_label.lower():
        return r'$|g\rangle$ state population'
    elif 'pe ' in prob_label.lower():
        return r'$|e\rangle$ state population'
    elif 'pf ' in prob_label.lower():
        return r'$|f\rangle$ state population'
    else:
        return prob_label


def flatten_list(lst_of_lsts):
    """
    Flattens the list of lists lst_of_lsts.
    :param lst_of_lsts: a list of lists
    :return: flattened list
    """
    if all([isinstance(e, list) for e in lst_of_lsts]):
        return [e for l1 in lst_of_lsts for e in l1]
    elif any([isinstance(e, list) for e in lst_of_lsts]):
        l = []
        for e in lst_of_lsts:
            if isinstance(e, list):
                l.extend(e)
            else:
                l.append(e)
        return l
    else:
        return lst_of_lsts


def is_string_in(string, lst_to_search):
    """
    Checks whether string is in the list lst_to_serach
    :param string: a string
    :param lst_to_search: list of strings or list of lists of strings
    :return: True or False
    """
    lst_to_search_flat = flatten_list(lst_to_search)
    found = False
    for el in lst_to_search_flat:
        if string in el:
            found = True
            break
    return found


def get_sublst_with_all_strings_of_list(lst_to_search, lst_to_match):
    """
    Finds all string elements in lst_to_search that contain the
    string elements of lst_to_match.
    :param lst_to_search: list of strings to search
    :param lst_to_match: list of strings to match
    :return: list of strings from lst_to_search that contain all string
    elements in lst_to_match
    """
    lst_w_matches = []
    for etm in lst_to_match:
        for ets in lst_to_search:
            r = re.search(etm, ets)
            if r is not None:
                lst_w_matches += [ets]
    # unique_everseen takes unique elements while also keeping the original
    # order of the elements
    return list(unique_everseen(lst_w_matches))