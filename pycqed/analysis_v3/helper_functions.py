import logging
log = logging.getLogger(__name__)
import re
import os
import h5py
import itertools
import numpy as np
from numpy import array  # Needed for eval. Do not remove.
from copy import deepcopy
from collections import OrderedDict
from more_itertools import unique_everseen
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.measurement.hdf5_data import read_dict_from_hdf5
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement import sweep_points as sp_mod


def convert_attribute(attr_val):
    """
    Converts byte type to string because of h5py datasaving
    :param attr_val: the raw value of the attribute as retrieved from the HDF
        file
    :return: the converted attribute value
    """
    if isinstance(attr_val, bytes):
        attr_val = attr_val.decode('utf-8')
    # If it is an array of value decodes individual entries
    if isinstance(attr_val, np.ndarray) or isinstance(attr_val, list):
        attr_val = [av.decode('utf-8') if isinstance(av, bytes)
                    else av for av in attr_val]
    try:
        return eval(attr_val)
    except Exception:
        return attr_val


def get_hdf_param_value(group, param_name):
    '''
    Returns an attribute "key" of the group "Experimental Data"
    in the hdf5 datafile.
    '''
    s = group.attrs[param_name]
    return convert_attribute(s)


def get_value_names_from_timestamp(timestamp, file_id=None, mode='r'):
    """
    Returns value_names from the HDF5 file specified by timestamp.
    :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
    :return: list of value_names
    """
    folder = a_tools.get_folder(timestamp)
    h5filepath = a_tools.measurement_filename(folder, file_id=file_id)
    data_file = h5py.File(h5filepath, mode)
    try:
        channel_names = get_hdf_param_value(data_file['Experimental Data'],
                                            'value_names')
        data_file.close()
        return channel_names
    except Exception as e:
        data_file.close()
        raise e


def get_param_from_metadata_group(timestamp=None, param_name=None, file_id=None,
                                  data_file=None, close_file=True, mode='r'):
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
        h5filepath = a_tools.measurement_filename(folder, file_id=file_id)
        data_file = h5py.File(h5filepath, mode)

    try:
        if param_name is None:
            group = data_file['Experimental Data']
            return read_dict_from_hdf5({}, group['Experimental Metadata'])

        group = data_file['Experimental Data']['Experimental Metadata']
        if param_name in group:
            group = group[param_name]
            param_value = OrderedDict()
            if isinstance(group, h5py._hl.dataset.Dataset):
                param_value = list(np.array(group).flatten())
            else:
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
                           close_file=True, file_id=None, mode='r'):
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
        h5filepath = a_tools.measurement_filename(folder, file_id=file_id)
        data_file = h5py.File(h5filepath, mode)
    try:
        group = data_file['Experimental Data']
        if 'Data' in group:
            dataset = np.array(group['Data'])
        else:
            raise KeyError('Data was not found in Experimental Data.')
        if close_file:
            data_file.close()
    except Exception as e:
        data_file.close()
        raise e
    return dataset


def open_hdf_file(timestamp=None, folder=None, filepath=None, mode='r', file_id=None):
    """
    Opens the hdf5 file with flexible input parameters. If no parameter is given,
    opens the  hdf5 of the last measurement in reading mode.
    Args:
        :param timestamp: (str) measurement timestamp of form YYYYMMDD_hhmmsss
        :param folder: (str) path to file location
        :param mode filepath: (str) path to hdf5 file. Overwrites timestamp
            and folder
        :param mode: (str) mode to open the file ('r' for read),
            ('r+' for read/write)
        :param file_id: (str) file id
    :return: opened HDF5 file

    """
    if filepath is None:
        if folder is None:
            assert timestamp is not None
            folder = a_tools.get_folder(timestamp)
        filepath = a_tools.measurement_filename(folder, file_id=file_id)
    return h5py.File(filepath, mode)


def get_instr_param_from_hdf_file(instr_name, param_name, timestamp=None,
                                  folder=None, **params):
    """
    Extracts the value for the parameter specified by param_name for the
    instrument specified by instr_name from an HDF file.
    :param instr_name: str specifying the instrument name in the HDF file
    :param param_name: str specifyin the name of the parameter to extract
    :param timestamp: str of the form YYYYMMDD_hhmmss.
    :param folder: path to HDF file
    :param params: keyword arguments
    :return: value corresponding to param_name
    """
    if folder is None:
        if timestamp is None:
            raise ValueError('Please provide either timestamp or folder.')
        folder = a_tools.get_folder(timestamp)

    d = {}
    get_params_from_hdf_file(
        d, {'instr_param_val': f'Instrument settings.{instr_name}.{param_name}'},
        folder=folder)

    if d['instr_param_val'] == 0:
        raise KeyError(f'Parameter {param_name} not found for instrument '
                       f'{instr_name}.')
    return d['instr_param_val']


def get_clf_params_from_hdf_file(timestamp, meas_obj_names, for_ge=True,
                                 **params):
    """
    Extracts the acquistion classifier parameters for the meas_obj_names from
    an HDF file.
    First tries to take them from the Analysis group (timestamp corresponds to
    an ssro calibration measurement). If a key error is raised, it will take
    them from the qubit parameter acq_classifier_params.
    :param timestamp: timestamp string
    :param meas_obj_names: list of measured object names
    :param for_ge: flag indicating whether to ignore the f-level classification
        (sets means to 1000 for the f-level)
    :param params: keyword arguments passed to get_params_from_hdf_file
    :return: dict with meas_obj_names as keys and classifier params as values
    """

    try:
        params_dict = {}
        params_dict.update({f'{qbn}':
                                f'Analysis.Processed data.analysis_params.'
                                f'classifier_params.{qbn}'
                            for qbn in meas_obj_names})
        classifier_params = get_params_from_hdf_file(
            {},  params_dict, folder=a_tools.get_folder(timestamp), **params)
    except KeyError:
        params_dict = {}
        params_dict.update({f'{qbn}':
                                f'Instrument settings.{qbn}.'
                                f'acq_classifier_params'
                            for qbn in meas_obj_names})
        classifier_params = get_params_from_hdf_file(
            {},  params_dict, folder=a_tools.get_folder(timestamp), **params)

    if for_ge:
        for qbn in classifier_params:
            classifier_params[qbn]['means_'][2, :] = [1000, 1000]

    return classifier_params


def get_params_from_hdf_file(data_dict, params_dict=None, numeric_params=None,
                             add_param_method=None, folder=None, **params):
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

    # if folder is not specified, will take the last folder in the list from
    # data_dict['folders']
    if folder is None:
        folder = get_param('folders', data_dict, raise_error=True, **params)
        if len(folder) > 0:
            folder = folder[-1]

    h5mode = get_param('h5mode', data_dict, default_value='r', **params)
    h5filepath = a_tools.measurement_filename(folder, **params)
    data_file = h5py.File(h5filepath, h5mode)

    try:
        for save_par, file_par in params_dict.items():
            epd = data_dict
            all_keys = save_par.split('.')
            for i in range(len(all_keys)-1):
                if all_keys[i] not in epd:
                    epd[all_keys[i]] = OrderedDict()
                epd = epd[all_keys[i]]

            if isinstance(epd, list):
                epd = epd[-1]

            if file_par == 'measurementstring':
                add_param(all_keys[-1],
                          [os.path.split(folder)[1][7:]],
                          epd, add_param_method='append')
                continue

            group_name = '/'.join(file_par.split('.')[:-1])
            par_name = file_par.split('.')[-1]
            if group_name == '':
                group = data_file
                attrs = []
            else:
                group = data_file[group_name]
                attrs = list(group.attrs)

            if group_name in data_file or group_name == '':
                if par_name in attrs:
                    add_param(all_keys[-1],
                              get_hdf_param_value(group,
                                                  par_name),
                              epd, add_param_method=add_param_method)
                elif par_name in list(group.keys()) or file_par == '':
                    par = group[par_name] if par_name != '' else group
                    if isinstance(par,
                                  h5py._hl.dataset.Dataset):
                        add_param(all_keys[-1],
                                  np.array(par),
                                  epd, add_param_method=add_param_method)
                    else:
                        add_param(all_keys[-1],
                                  read_dict_from_hdf5(
                                      {}, par),
                                  epd, add_param_method=add_param_method)

            if all_keys[-1] not in epd:
                # search through the attributes of all groups
                for group_name in data_file.keys():
                    if par_name in list(data_file[group_name].attrs):
                        add_param(all_keys[-1],
                                  get_hdf_param_value(data_file[group_name],
                                                      par_name),
                                  epd, add_param_method=add_param_method)

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
                # if isinstance(data_dict[all_keys[0]], dict):
                #     data_to_proc_dict = {f'{keyi}.{k}': deepcopy(v) for k, v
                #                          in data_dict[all_keys[0]].items()}
                # else:
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
    dd = data_dict
    md = data_dict.get('exp_metadata', dict())
    if isinstance(md, list):
        # this should only happen when extracting metadata from a list of
        # timestamps. Hence, this extraction should be done separate from
        # from other parameter extractions, and one should call
        # combine_metadata_list in pipeline_analysis.py afterwards.
        md = md[0]
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
                if all_keys[i] in p:
                    p = p[all_keys[i]]
                if all_keys[i] in dd:
                    dd = dd[all_keys[i]]
                if all_keys[i] in md:
                    md = md[all_keys[i]]
                p = p if isinstance(p, dict) else OrderedDict()
                if isinstance(dd, list) or isinstance(dd, np.ndarray):
                    all_keys[i + 1] = int(all_keys[i + 1])
                else:
                    dd = dd if isinstance(dd, dict) else OrderedDict()
                md = md if isinstance(md, dict) else OrderedDict()
        if isinstance(dd, list) or isinstance(dd, np.ndarray):
            value = dd[all_keys[-1]]
        else:
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
    dd = data_dict
    md = data_dict.get('exp_metadata', dict())
    if isinstance(md, list):
        # this should only happen when extracting metadata from a list of
        # timestamps. Hence, this extraction should be done separate from
        # from other parameter extractions, and one should call
        # combine_metadata_list in pipeline_analysis.py afterwards.
        md = md[0]
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
                if all_keys[i] in p:
                    p = p[all_keys[i]]
                if all_keys[i] in dd:
                    dd = dd[all_keys[i]]
                if all_keys[i] in md:
                    md = md[all_keys[i]]
                p = p if isinstance(p, dict) else OrderedDict()
                dd = dd if isinstance(dd, dict) else OrderedDict()
                md = md if isinstance(md, dict) else OrderedDict()

        value = p.pop(all_keys[-1],
                      dd.pop(all_keys[-1],
                             md.pop(all_keys[-1], default_value)))

    if raise_error and value is None:
        if error_message is None:
            error_message = f'{param} was not found in either data_dict, or ' \
                            f'exp_metadata or input params.'
        raise ValueError(error_message)
    return value


def add_param(name, value, data_dict, add_param_method=None, **params):
    """
    Adds a new key-value pair to the data_dict, with key = name.
    If update, it will try data_dict[name].update(value), else raises KeyError.
    :param name: key of the new parameter in the data_dict
    :param value: value of the new parameter
    :param data_dict: OrderedDict containing data to be processed
    :param add_param_method: str specifying how to add the value if name
        already exists in data_dict:
            'skip': skip adding this parameter without raising an error
            'replace': replace the old value corresponding to name with value
            'update': whether to try data_dict[name].update(value).
                Both value and the already-existing entry in data_dict have got
                to be dicts.
            'append': whether to try data_dict[name].extend(value). If either
                value or already-existing entry in data_dict are not lists,
                they will be converted to lists.
    :param params: keyword arguments

    Assumptions:
        - if update_value == True, both value and the already-existing entry in
            data_dict need to be dicts.
    """
    dd = data_dict
    all_keys = name.split('.')
    if len(all_keys) > 1:
        for i in range(len(all_keys)-1):
            if isinstance(dd, list):
                all_keys[i] = int(all_keys[i])
            if not isinstance(dd, list) and all_keys[i] not in dd:
                dd[all_keys[i]] = OrderedDict()
            dd = dd[all_keys[i]]

    if isinstance(dd, list) or isinstance(dd, np.ndarray):
        all_keys[-1] = int(all_keys[-1])
    if isinstance(dd, list) or all_keys[-1] in dd:
        if add_param_method == 'skip':
            return
        elif add_param_method == 'update':
            if not isinstance(value, dict):
                raise ValueError(f'The value corresponding to {all_keys[-1]} '
                                 f'is not a dict. Cannot update_value in '
                                 f'data_dict')
            if isinstance(dd[all_keys[-1]], list):
                for k, v in value.items():
                    dd[all_keys[-1]][int(k)] = v
            else:
                dd[all_keys[-1]].update(value)
        elif add_param_method == 'append':
            v = dd[all_keys[-1]]
            if not isinstance(v, list):
                dd[all_keys[-1]] = [v]
            else:
                dd[all_keys[-1]] = v
            if not isinstance(value, list):
                dd[all_keys[-1]].extend([value])
            else:
                dd[all_keys[-1]].extend(value)
        elif add_param_method == 'replace':
            dd[all_keys[-1]] = value
        else:
            raise KeyError(f'{all_keys[-1]} already exists in data_dict and it'
                           f' is unclear how to add it.')
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
    for prop in props_to_extract:
        if 'cp' == prop:
            cp = get_param('cal_points', data_dict, raise_error=raise_error,
                           **params)
            if isinstance(cp, str):
                cp = CalibrationPoints.from_string(cp)
            props_to_return += [cp]
        elif 'sp' == prop:
            sp = get_param('sweep_points', data_dict, raise_error=raise_error,
                           **params)
            props_to_return += [sp_mod.SweepPoints(sp)]
        elif 'mospm' == prop:
            meas_obj_sweep_points_map = get_param(
                'meas_obj_sweep_points_map', data_dict, raise_error=raise_error,
                **params)
            props_to_return += [meas_obj_sweep_points_map]
        elif 'movnm' == prop:
            meas_obj_value_names_map = get_param(
                'meas_obj_value_names_map', data_dict, raise_error=raise_error,
                **params)
            props_to_return += [meas_obj_value_names_map]
        elif 'rev_movnm' == prop:
            meas_obj_value_names_map = get_param(
                'meas_obj_value_names_map', data_dict, raise_error=raise_error,
                **params)
            rev_movnm = OrderedDict()
            for mobjn, value_names in meas_obj_value_names_map.items():
                rev_movnm.update({vn: mobjn for vn in value_names})
            props_to_return += [rev_movnm]
        elif 'mobjn' == prop:
            mobjn = get_param('meas_obj_names', data_dict,
                              raise_error=raise_error, **params)
            if params.get('enforce_one_meas_obj', True):
                if isinstance(mobjn, list):
                    if len(mobjn) > 1:
                        raise ValueError(f'This node expects one measurement '
                                         f'object, {len(mobjn)} were given.')
                    else:
                        mobjn = mobjn[0]
            else:
                if isinstance(mobjn, str):
                    mobjn = [mobjn]
            props_to_return += [mobjn]
        else:
            raise KeyError(f'Extracting {prop} is not implemented in this '
                           f'function. Please use get_params_from_hdf_file.')

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
                'reset_reps', 3)
    return reset_reps


def get_observables(data_dict, keys_out=None, preselection_shift=-1,
                    do_preselection=False, **params):
    """
    Creates the observables dictionary from meas_obj_names, preselection_shift,
        and do_preselection.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param keys_out: list with one entry specifying the key name or dictionary
        key path in data_dict for the processed data to be saved into
    :param preselection_shift: integer specifying which readout prior to the
        current readout to be considered for preselection
    :param do_preselection: bool specifying whether to do preselection on
        the data.
    :param params: keyword arguments
        Expects to find either in data_dict or in params:
            - meas_obj_names: list of measurement object names
    :return: a dictionary with
        name of the qubit as key and boolean value indicating if it is
        selecting exited states. If the qubit is missing from the list
        of states it is averaged out. Instead of just the qubit name, a
        tuple of qubit name and a shift value can be passed, where the
        shift value specifies the relative readout index for which the
        state is checked.
        Example qb2-qb4 state tomo with preselection:
            {'pre': {('qb2', -1): False,
                    ('qb4', -1): False}, # preselection conditions
             '$\\| gg\\rangle$': {'qb2': False,
                                  'qb4': False,
                                  ('qb2', -1): False,
                                  ('qb4', -1): False},
             '$\\| ge\\rangle$': {'qb2': False,
                                  'qb4': True,
                                  ('qb2', -1): False,
                                  ('qb4', -1): False},
             '$\\| eg\\rangle$': {'qb2': True,
                                  'qb4': False,
                                  ('qb2', -1): False,
                                  ('qb4', -1): False},
             '$\\| ee\\rangle$': {'qb2': True,
                                  'qb4': True,
                                  ('qb2', -1): False,
                                  ('qb4', -1): False}}
    """
    mobj_names = None
    legacy_channel_map = get_param('channel_map', data_dict, **params)
    task_list = get_param('task_list', data_dict, **params)
    if legacy_channel_map is not None:
        mobj_names = list(legacy_channel_map)
    else:
        mobj_names = get_measurement_properties(
            data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
            **params)
    # elif task_list is not None:
    #     mobj_names = get_param('qubits', task_list[0])

    # if mobj_names is None:
    #     # make sure the qubits are in the correct order here when we take a
    #     # tomo measurement in new framework
    #     mobj_names = get_measurement_properties(
    #         data_dict, props_to_extract=['mobjn'], enforce_one_meas_obj=False,
    #         **params)

    combination_list = list(itertools.product([False, True],
                                              repeat=len(mobj_names)))
    preselection_condition = dict(zip(
        [(qb, preselection_shift) for qb in mobj_names],  # keys contain shift
        combination_list[0]  # first comb has all ground
    ))
    observables = OrderedDict()

    # add preselection condition also as an observable
    if do_preselection:
        observables["pre"] = preselection_condition
    # add all combinations
    for i, states in enumerate(combination_list):
        name = ''.join(['e' if s else 'g' for s in states])
        obs_name = '$\| ' + name + '\\rangle$'
        observables[obs_name] = dict(zip(mobj_names, states))
        # add preselection condition
        if do_preselection:
            observables[obs_name].update(preselection_condition)

    if keys_out is None:
        keys_out = ['observables']
    if len(keys_out) != 1:
        raise ValueError(f'keys_out must have length one. {len(keys_out)} '
                         f'entries were given.')
    add_param(keys_out[0], observables, data_dict, **params)


def select_data_from_nd_array(data_dict, keys_in, keys_out, **params):
    """
    Select subset of data from an n-d array along any of the axes.
    :param data_dict: OrderedDict containing data to be processed and where
        processed data is to be stored
    :param keys_in: key names or dictionary keys paths in data_dict for shots
        (with preselection) classified into pg, pe, pf
    :param keys_out: list of key names or dictionary keys paths in
        data_dict for the processed data to be saved into
    :param params: keyword arguments
        - selection_map (dict, default: must be provided): dict of the form
            {axis: index_list} where axis is any axis in the original data array.
            index_list is a list of tuples specifying indices or ranges as:
            - [2, 3, 4]: array[2] and array[3] and array[4]
            - [(n, m)]: array[n:m]
            - [(n, 'end')]: array[n:]
            - [(n, m, k)]: array[n:m:k]
            - can also be [2, (n, end), (m, k, l)] etc.

    A new entry in data_dict is added for each keyi in keys_in, under
    keyo in keys_out.

    Assumptions:
        - len(keys_in) == len(keys_out)
        - if len(keys_in) > 1, the same selection_map is used for all
    """
    if len(keys_out) != len(keys_in):
        raise ValueError('keys_out and keys_in do not have '
                         'the same length.')

    data_to_proc_dict = get_data_to_process(data_dict, keys_in)
    selection_map = get_param('selection_map', data_dict, raise_error=True,
                              **params)

    for keyi, keyo in zip(keys_in, keys_out):
        selected_data = deepcopy(data_to_proc_dict[keyi])
        for axis, sel_info in selection_map.items():
            indices = np.array([], dtype=int)
            arange_axis = np.arange(selected_data.shape[axis])
            for sl in sel_info:
                if hasattr(sl, '__iter__'):
                    if len(sl) == 2:
                        if sl[1] == 'end':
                            indices = np.append(indices, arange_axis[sl[0]:])
                        else:
                            indices = np.append(indices,
                                                arange_axis[sl[0]:sl[1]])
                    elif len(sl) == 3:
                        if sl[1] == 'end':
                            indices = np.append(indices,
                                                arange_axis[sl[0]::sl[2]])
                        else:
                            indices = np.append(indices,
                                                arange_axis[sl[0]:sl[1]:sl[2]])
                else:
                    # sl is a number
                    indices = np.append(indices, sl)

            if len(indices):
                indices = np.sort(indices)
                selected_data = np.take(selected_data, indices, axis=axis)
            else:
                log.warning('No data selected in select_data_from_nd_array.')

        add_param(keyo, selected_data, data_dict, **params)


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
    if all([isinstance(e, (list, tuple)) for e in lst_of_lsts]):
        return [e for l1 in lst_of_lsts for e in l1]
    elif any([isinstance(e, (list, tuple)) for e in lst_of_lsts]):
        l = []
        for e in lst_of_lsts:
            if isinstance(e, (list, tuple)):
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


def check_equal(value1, value2):
    """
    Check if value1 is the same as value2.
    :param value1: dict, list, tuple, str, np.ndarray; dict, list, tuple can
        contain further dict, list, tuple
    :param value2: dict, list, tuple, str, np.ndarray; dict, list, tuple can
        contain further dict, list, tuple
    :return: True if value1 is the same as value2, else False
    """
    if not isinstance(value1, (float, int, bool, np.number,
                               np.float_, np.int_, np.bool_)):
        assert type(value1) == type(value2)

    if not hasattr(value1, '__iter__'):
        return value1 == value2
    else:
        if isinstance(value1, dict):
            if len(value1) != len(value2):
                return False
            for k, v in value1.items():
                if k not in value2:
                    return False
                else:
                    if not check_equal(v, value2[k]):
                        return False
            # if it reached this point, then all key-vals are the same
            return True
        if isinstance(value1, (list, tuple)):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(value1, value2):
                if not check_equal(v1, v2):
                    return False
            return True
        else:
            try:
                # numpy array
                if value1.shape != value2.shape:
                    return False
                else:
                    return np.all(np.isclose(value1, value2))
            except AttributeError:
                if len(value1) != len(value2):
                    return False
                else:
                    return value1 == value2


def read_analysis_file(timestamp=None, filepath=None, data_dict=None,
                       file_id=None, ana_file=None, close_file=True, mode='r'):
    """
    Creates a data_dict from an AnalysisResults file as generated by analysis_v3
    :param timestamp: str with a measurement timestamp
    :param filepath: (str) path to file
    :param data_dict: dict where to store the file entries
    :param file_id: suffix to the usual HDF measurement file found from giving
        a measurement timestamp. Defaults to '_AnalysisResults,' the standard
        suffix created by analysis_v3
    :param ana_file: HDF file instance
    :param close_file: whether to close the HDF file at the end
    :param mode: str specifying the HDF read mode (if ana_file is None)
    :return: the data dictionary
    """
    if data_dict is None:
        data_dict = {}
    try:
        if ana_file is None:
            if filepath is None:
                if file_id is None:
                    file_id = 'AnalysisResults'
                folder = a_tools.get_folder(timestamp)
                filepath = a_tools.measurement_filename(folder,
                                                        file_id=f'_{file_id}')
            ana_file = h5py.File(filepath, mode)
        read_from_hdf(data_dict, ana_file)
        if close_file:
            ana_file.close()
    except Exception as e:
        if close_file:
            try:
                ana_file.close()
            except AttributeError:
                # there was an exception before reaching the line above where
                # ana_file is opened
                pass
        raise e
    return data_dict


def read_from_hdf(data_dict, hdf_group):
    """
    Adds to data_dict everything found in the HDF group or file hdf_group.
    :param data_dict: dict where the entries will be stored
    :param hdf_group: HDF group or file
    :return: nothing but updates data_dict with all values from hdf_group
    """
    if not len(hdf_group) and not len(hdf_group.attrs):
        path = hdf_group.name.split('/')[1:]
        add_param('.'.join(path), {}, data_dict)

    for key, value in hdf_group.items():
        if isinstance(value, h5py.Group):
            read_from_hdf(data_dict, value)
        else:
            path = value.name.split('/')[1:]
            if 'list_type' not in value.attrs:
                val_to_store = value[()]
            elif value.attrs['list_type'] == 'str':
                # lists of strings needs some special care, see also
                # the writing part in the writing function above.
                val_to_store = [x[0] for x in value[()]]
            else:
                val_to_store = list(value[()])
            if path[-2] == path[-1]:
                path = path[:-1]
            add_param('.'.join(path), val_to_store, data_dict)

    path = hdf_group.name.split('/')[1:]
    for key, value in hdf_group.attrs.items():
        if isinstance(value, str):
            # Extracts "None" as an exception as h5py does not support
            # storing None, nested if statement to avoid elementwise
            # comparison warning
            if value == 'NoneType:__None__':
                value = None
            elif value == 'NoneType:__emptylist__':
                value = []

        temp_path = deepcopy(path)
        if temp_path[-1] != key:
            temp_path += [key]
        if 'list_type' not in hdf_group.attrs:
            value = convert_attribute(value)
            if key == 'cal_points' and not isinstance(value, str):
                value = repr(value)
        add_param('.'.join(temp_path), value, data_dict)

    if 'list_type' in hdf_group.attrs:
        if (hdf_group.attrs['list_type'] == 'generic_list' or
                hdf_group.attrs['list_type'] == 'generic_tuple'):
            list_dict = pop_param('.'.join(path), data_dict)
            data_list = []
            for i in range(list_dict['list_length']):
                data_list.append(list_dict[f'list_idx_{i}'])
            if hdf_group.attrs['list_type'] == 'generic_tuple':
                data_list = tuple(data_list)
            if path[-1] == 'sweep_points':
                data_list = sp_mod.SweepPoints(data_list)
            add_param('.'.join(path), data_list, data_dict,
                      add_param_method='replace')
        else:
            raise NotImplementedError('cannot read "list_type":"{}"'.format(
                hdf_group.attrs['list_type']))




