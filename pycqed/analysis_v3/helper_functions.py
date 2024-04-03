import logging
log = logging.getLogger(__name__)
import re
import os
import h5py
import traceback
import itertools
import numpy as np
import pycqed.analysis_v2.base_analysis as ba
from numpy import array  # Needed for eval. Do not remove.
from copy import deepcopy
from collections import OrderedDict
from more_itertools import unique_everseen
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.io.hdf5 import read_dict_from_hdf5, decode_attribute_value
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement import sweep_points as sp_mod
import pycqed.utilities.settings_manager as setman
from pycqed.instrument_drivers.mock_qcodes_interface import \
    ParameterNotFoundError


def convert_attribute(attr_val):
    log.warning("Deprecated function, will be removed in a future MR;"
                " Please use utilities.io.hdf5.decode_attribute_value instead.")
    return decode_attribute_value(attr_val)


def decode_parameter_value(param_value):
    """
    Converts byte type to the true type of a parameter loaded from a file.

    Args:
        param_value: the raw value of the parameter as retrieved from the HDF
            file

    Returns:
        the converted parameter value
    """
    log.warning("Deprecated function, will be removed in a future MR;"
                " Please use utilities.io.hdf5.decode_attribute_value instead.")
    return decode_attribute_value(param_value)


def get_hdf_param_value(group, param_name):
    """
    Function will be removed in future MR.
    Returns an attribute "key" of the group "Experimental Data"
    in the hdf5 datafile.
    """
    log.warning("Deprecated function, will be removed in a future MR; "
                "Please use get_param_from_group() instead.")
    s = group.attrs[param_name]
    return decode_attribute_value(s)


def get_value_names_from_timestamp(timestamp, **params):
    """
    Returns value_names from an HDF5 file.

    Args:
        timestamp (str): with a measurement timestamp (YYYYMMDD_hhmmss)
        **params: keyword arguments passed to a_tools.open_hdf_file,
            see docstring there for acceptable input parameters

    Returns:
        list of value_names
    """

    data_file = a_tools.open_hdf_file(timestamp, **params)
    try:
        channel_names = decode_attribute_value(
            data_file['Experimental Data'].attrs['value_names'])
        data_file.close()
        return channel_names
    except Exception as e:
        data_file.close()
        raise e


def get_param_from_group(group_name, param_name=None, timestamp=None,
                         folder=None, file=None, find_all_matches=True,
                         **params):
    """
    Extract the value of a parameter from any group in the data files.

    This functions returns the value of the parameters whose paths in the
    file end with param_name, i.e. 'Analysis.group1.group2.param_name'.

    If only one match is found, the function returns the value corresponding
    to that path.

    If more than one match is found, this function returns a dict with all
    matches if find_all_matches==True, or raises a ValueError otherwise.

    Args:
        group_name (str): name of the group to search in, ex:
            Analysis, Experimental Data, qb1
        param_name (str): name of the parameter to extract
        timestamp (str): timestamp (YYYYMMDD_hhmmss) of the measurement. Will
            be used to find the location of the HDF file
        folder (str): path to the HDF file
        file (open file): open file where to search
        find_all_matches (bool): whether to return the values of all the
            parameters in the group that contain the substring param_name in
            their path. If this flag is False and more than one parameter
            contains the string param_name, a ValueError will be raised.
        **params: keyword arguments: not used but are here to allow pass-through

    Returns:
        if only one match was found: the value corresponding to param_name
        if more than one match found: dict with parameters paths as keys and
            values corresponding to those paths as values
    """
    if file is None:
        if folder is None:
            if timestamp is None:
                raise ValueError('Please provide either timestamp or folder.')
            folder = a_tools.get_folder(timestamp)

        try:
            group = get_params_from_files(
                {}, {'group': group_name}, folder=folder)
        except ParameterNotFoundError:
            raise KeyError(f'Group "{group_name}" was not found.')
        group = group['group']
    else:
        if group_name not in file:
            raise KeyError(f'Group "{group_name}" was not found.')
        group = file[group_name]

    if param_name is None:
        return group
    else:
        all_matches = find_all_in_dict(param_name, group)
        param_value = {}
        for key_paths in all_matches:
            split_kp = key_paths.split('.')
            if param_name == split_kp[-1]:
                param_value[key_paths] = decode_attribute_value(
                    all_matches[key_paths])

        if len(param_value) > 1 and not find_all_matches:
            raise ValueError(f'{len(param_value)} parameters were found that '
                             f'contain "{param_name}":\n{list(param_value)}.')
        if len(param_value) == 0:
            raise KeyError(f'Parameter {param_name} was not '
                           f'found in group "{group_name}."')
        elif len(param_value) == 1:
            param_value = list(param_value.values())[0]
        return param_value


def get_param_from_metadata_group(timestamp=None, param_name=None,
                                  folder=None, **params):
    """
    Returns the value of param_name from the group "Experimental Metadata."

    See docstring of get_param_from_group for more details.
    """
    return get_param_from_group('Experimental Metadata', param_name, timestamp,
                                folder, **params)


def get_param_from_analysis_group(timestamp=None, param_name=None,
                                  folder=None, **params):
    """
    Returns the value of param_name from the group "Analysis."

    See docstring of get_param_from_group for more details.
    """
    return get_param_from_group('Analysis', param_name, timestamp,
                                folder, **params)


def get_instr_param_from_file(instr_name, param_name=None, timestamp=None,
                              folder=None, **params):
    """
    Returns the value of an instrument parameter of an hdf file.

    Args
        instr_name (str): name of a instrument, ex: qb1, Pulsar, TWPA

    See docstring of get_param_from_group for more details.
    """
    SETTINGS_PREFIX = 'Instrument settings.'
    return get_param_from_group(SETTINGS_PREFIX+instr_name, param_name,
                                timestamp, folder, **params)


def get_param_from_fit_res(param, fit_res, split_char='.'):
    """
    Extract the value of a parameter from a fit result object or dict.

    When a fit result is loaded from an HDF file, it is loaded as a dict, and
    the way to access the relevant fit parameters is therefore slightly
    different compared to accessing them from a lmfit.ModelResult instance. This
    function returns the desired parameters from either a ModelResult instance
    or a dict, so it is convenient to use in analyses notebooks for example,
    where the data you process can be either a ModelResult instance or a
    loaded fit results dict.

    Args:
        param (str): of the form param_name + split_char + param_attribute where
            param_name is the name of a fit parameter in fit_res, and
            para_attribute is a fit attribute stored by lmfit, ex: value,
            stderr, min, max, correl, init_value etc.
        fit_res (instance of lmfit.model.ModelResult or dict):
            contains the fit results. This parameter is a dict whenever it
            is loaded from an HDF file.
        split_char (str): the character around which to split param

    Returns:
        value corresponding to param
    """
    param_split = param.split(split_char)
    try:
        p = fit_res.params[param_split[0]]
        p_name = param_split[1]
        if p_name == 'value':
            p_name = '_val'
        return p.__dict__[p_name]
    except AttributeError:
        return get_param(f'params.{param}', fit_res, split_char=split_char,
                         raise_error=True)


def get_data_from_hdf_file(timestamp=None, data_file=None, close_file=True,
                           **params):
    log.warning("Deprecated function, will be removed in a future MR;"
                "use hlp_mod.get_dataset_from_hdf_file() instead.")
    return get_dataset_from_hdf_file(timestamp=timestamp, data_file=data_file,
                                     close_file=close_file, **params)


def get_dataset_from_hdf_file(timestamp=None, data_file=None, close_file=True,
                              **params):
    """
    Return the measurement data stored in the Experimental Data group of an
    HDF file.

    Args:
        timestamp (str): with a measurement timestamp (YYYYMMDD_hhmmss)
        data_file (open HDF file): measurement file from which to load data
        close_file (bool): whether to close the ana_file at the end
        **params: keyword arguments passed to a_tools.open_hdf_file,
            see docstring there for acceptable input parameters

    Returns:
        numpy array with the dataset
    """

    if data_file is None:
        data_file = a_tools.open_hdf_file(timestamp, **params)
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


def open_hdf_file(timestamp=None, folder=None, filepath=None, mode='r',
                  file_id=None, **params):
    """
    Opens an HDF file.

    Args:
        timestamp (str): with a measurement timestamp (YYYYMMDD_hhmmss)
        folder (str): path to file location without the filename + extension
        filepath (str): path to HDF file, including the filename + extension.
            Overwrites timestamp and folder.
        mode (str): mode in which to open the file ('r' for read,
            'w' for write, 'r+' for read/write).
        file_id (str): suffix of the file name
        **params: keyword arguments passed to measurement_filename,
            see docstring there for acceptable input parameters

    Returns:
        open HDF file
    """
    log.warning("Deprecated function, will be removed in a future MR;"
                "use a_tools.open_hdf_file() instead.")
    return a_tools.open_hdf_file(timestamp=timestamp, folder=folder,
                                 filepath=filepath, mode=mode, file_id=file_id,
                                 **params)


def get_qb_channel_map_from_file(qb_names, file_path, value_names, **kw):
    """
    Construct the qubit channel map based from a config file.

    Args:
        qb_names (list): list of qubit names
        file_path (str): path to the HDF file
        value_names (list): list of detector function value names
        h5mode (str): HDF opening mode

    Returns
        qubit channel map as a dict with qubit names as keys and the list of
        value names corresponding to each qubit as values
    """
    channel_map = {}

    if 'raw' in value_names[0]:
        ro_type = 'raw w'
    elif 'digitized' in value_names[0]:
        ro_type = 'digitized w'
    elif 'lin_trans' in value_names[0]:
        ro_type = 'lin_trans w'
    else:
        ro_type = 'w'

    for qbn in qb_names:
        SETTINGS_PREFIX = 'Instrument settings.'
        params_dict = {'acq_I_channel': f'{SETTINGS_PREFIX}{qbn}.acq_I_channel',
                       'acq_Q_channel': f'{SETTINGS_PREFIX}{qbn}.acq_Q_channel',
                       'acq_weights_type': f'{SETTINGS_PREFIX}{qbn}.acq_weights_type',
                       }
        try:
            uhf = get_params_from_files(
                {}, {'instr_acq': f'{SETTINGS_PREFIX}{qbn}.instr_acq'},
                folder=file_path, **kw)['instr_acq']
        except KeyError:
            # Compatibility with older setting files where key
            # instr_acq was named instr_uhf
            uhf = get_params_from_files(
                {}, {'instr_uhf': f'{SETTINGS_PREFIX}{qbn}.instr_uhf'},
                folder=file_path, **kw)['instr_uhf']
        try:
            acq_unit = get_params_from_files(
                {}, {'acq_unit': f'{SETTINGS_PREFIX}{qbn}.acq_unit'},
                folder=file_path, **kw)
            acq_unit = acq_unit['acq_unit']
        except KeyError:
            # compatibility with older setting files
            acq_unit = None
        params = get_params_from_files({}, params_dict, folder=file_path, **kw)
        qbchs = [str(params['acq_I_channel'])]
        ro_acq_weight_type = params['acq_weights_type']

        if ro_acq_weight_type in ['SSB', 'DSB', 'DSB2', 'custom_2D',
                                  'optimal_qutrit']:
            qbchs += [str(params['acq_Q_channel'])]
        if acq_unit is None:  # compatibility with older setting files
            vn_string = uhf + '_' + ro_type
        else:
            vn_string = uhf + '_' + str(acq_unit) + '_' + ro_type
        channel_map[qbn] = [vn for vn in value_names for nr in qbchs
                            if vn_string+nr in vn]

    all_values_empty = np.all([len(v) == 0 for v in channel_map.values()])
    if len(channel_map) == 0 or all_values_empty:
        raise ValueError('Did not find any channels. '
                         'qb_channel_map is empty.')
    return channel_map


def get_qb_thresholds_from_hdf_file(meas_obj_names, timestamp=None,
                                    acq_dev_name=None, th_path=None,
                                    th_scaling=1, **params):
    """
    Extracts the state classification threshold value for the qubits in
    meas_obj_names.
    :param meas_obj_names: list of measured object names
    :param timestamp: timestamp string of an HDF file from which to extract
    :param acq_dev_name: name of acquisition device
    :param th_path: path inside HDF file to the threshold attribute of acq dev
    :param th_scaling: float by which to scale the threshold values
    :param params: passed to get_instr_param_from_hdf_file. See docstring there.
    :return: thresholds of the form
        {meas_obj_name: classification threshold value).
    """
    log.warning("Deprecated function, will be removed in a future MR;"
                "use get_qb_thresholds_from_file() instead.")
    return get_qb_thresholds_from_file(meas_obj_names, timestamp=timestamp,
                                    acq_dev_name=acq_dev_name, th_path=th_path,
                                    th_scaling=th_scaling, **params)


def get_qb_thresholds_from_file(meas_obj_names, timestamp=None,
                                acq_dev_name=None, th_path=None,
                                th_scaling=1, **params):
    """
    Extracts the state classification threshold value for the qubits in
    meas_obj_names.
    :param meas_obj_names: list of measured object names
    :param timestamp: timestamp string of an HDF file from which to extract
    :param acq_dev_name: name of acquisition device
    :param th_path: path inside HDF file to the threshold attribute of acq dev
    :param th_scaling: float by which to scale the threshold values
    :param params: passed to get_instr_param_from_file. See docstring there.
    :return: thresholds of the form
        {meas_obj_name: classification threshold value).
    """
    use_default_th_path = th_path is None
    thresholds = {}
    for mobjn in meas_obj_names:
        if acq_dev_name is None:
            # take the one defined in mobjn
            try:
                acq_dev_name = get_instr_param_from_file(mobjn, 'instr_acq',
                                                         timestamp, **params)
            except KeyError:
                acq_dev_name = get_instr_param_from_file(mobjn, 'instr_uhf',
                                                         timestamp, **params)
        if use_default_th_path:
            # try to figure out the correct path for the acquisition instrument
            if 'uhf' in acq_dev_name.lower():
                # acquisition device is a UHFQA
                ro_ch = get_instr_param_from_file(mobjn, 'acq_I_channel',
                                                  timestamp, **params)
                th_path = f'qas_0_thresholds_{ro_ch}_level'
            else:
                raise NotImplementedError('Currently only UHFQA is supported.')

        thresholds[mobjn] = get_instr_param_from_file(acq_dev_name, th_path,
                                                      timestamp, **params)
        if thresholds[mobjn] is not None:
            thresholds[mobjn] *= th_scaling

    return thresholds


def get_clf_params_from_hdf_file(timestamp, meas_obj_names,
                                 classifier_params=None, for_ge=False,
                                 from_analysis_group=False, **params):
    """
    Extracts the acquisition classifier parameters for the meas_obj_names from
    an HDF file.

    If the classifier params are not found for a measurement object, then a
    KeyError is raised.

    Args:
        timestamp (str): timestamp string
        meas_obj_names (list): of measured object names
        classifier_params (dict): with meas_obj_names as keys and classifier
            params as values. This function will only extract the classifier
            params for a measurement object if they don't already exist in
            classifier_params.
        for_ge (bool): indicating whether to ignore the f-level classification
            (sets means to 1000 for the f-level)
        from_analysis_group (bool): if True, it will take the
            classifier_params from the Analysis group (timestamp corresponds
            to an ssro calibration measurement). If False, it will take the
            qubit parameter acq_classifier_params.
        **params: keyword arguments to allow pass-through

    Returns:
        dict with meas_obj_names as keys and classifier params or None as values
    """
    if classifier_params is None:
        classifier_params = {}
    for mobjn in meas_obj_names:
        if classifier_params.get(mobjn, None) is None:
            if from_analysis_group:
                # get_param_from_analysis_group will return {qbn: clf_pars}
                # for whichever qubit was in the Analysis group (might not
                # be part of meas_obj_names).
                classifier_params[mobjn] = get_param_from_analysis_group(
                    'classifier_params', timestamp).get(mobjn, None)
            else:
                classifier_params[mobjn] = get_instr_param_from_file(
                    mobjn, 'acq_classifier_params', timestamp)

        if for_ge and classifier_params[mobjn] is not None and \
                len(classifier_params[mobjn]['weights_']) == 3:
            classifier_params[mobjn]['means_'][2, :] = [1000, 1000]
    return classifier_params


def get_state_prob_mtxs_from_hdf_file(timestamp, meas_obj_names,
                                      state_prob_mtxs=None,
                                      from_analysis_group=False, **params):
    """
    Extracts the state assignment probability matrices for the meas_obj_names
    from an HDF file.

    If the state assignment probability matrix is not found for a measurement
    object, then this function returns None for that measurement object.

    Args:
        timestamp (str): timestamp string
        meas_obj_names (list): of measured object names
        state_prob_mtxs (dict): with meas_obj_names as keys and state assignment
            probability matrices as values. This function will only extract the
            state assignment probability matrix for a measurement object if
            it doesn't already exist in state_prob_mtxs.
        from_analysis_group (bool): if True, it will take the
            state_prob_mtx_masked from the Analysis group (timestamp corresponds
            to an ssro calibration measurement). If False, it will take the
            qubit parameter acq_state_prob_mtx.

        **params: keyword arguments to allow pass-through

    Returns:
        dict with meas_obj_names as keys and state assignment probability
        matrices or None as values
    """
    if state_prob_mtxs is None:
        state_prob_mtxs = {}
    for mobjn in meas_obj_names:
        if state_prob_mtxs.get(mobjn, None) is None:
            if from_analysis_group:
                # get_param_from_analysis_group will return {qbn: clf_pars}
                # for whichever qubit was in the Analysis group (might not
                # be part of meas_obj_names).
                state_prob_mtxs[mobjn] = get_param_from_analysis_group(
                    timestamp, 'state_prob_mtx_masked').get(mobjn, None)
            else:
                try:
                    state_prob_mtxs[mobjn] = get_instr_param_from_file(
                        mobjn, 'acq_state_prob_mtx', timestamp)
                except KeyError:
                    state_prob_mtxs[mobjn] = None
    return state_prob_mtxs


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
    log.warning("Deprecated function, will be removed in a future MR;"
                "use hlp_mod.get_instr_param_from_file() instead.")
    return get_instr_param_from_file(instr_name, param_name=param_name,
                                     timestamp=timestamp, folder=folder,
                                     **params)


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
    log.warning("Deprecated function, will be removed in a future MR;"
                "use hlp_mod.get_params_from_files() instead.")
    return get_params_from_files(data_dict, params_dict=params_dict,
                                 numeric_params=numeric_params,
                                 add_param_method=add_param_method,
                                 folder=folder,
                                 **params)


def get_params_from_files(data_dict, params_dict=None, numeric_params=None,
                          add_param_method=None, folder=None, **params):
    """
    Extracts the parameters provided in params_dict.values() from the data and
    config files corresponding to a measurement, and saves them in data_dict
    under the keys of params_dict.

    The values of params_dict correspond to the paths leading to the desired
    parameters inside the files:
    params_dict = {storing key for param: path to param}, where the path to
    param is the key hierarchy inside a nested dict, separated by "."
    Example:
        params_dict = {
            'qb1_amp180': 'qb1.amp180',
            'awg': 'Pulsar.AWG1_ch1_awg',
            'uhf_idn': 'UHF.IDN',
            'qscale': 'Analysis.Processed data.analysis_params_dict.qb3.qscale',
            'data_to_fit': 'Experimental Data.Experimental Metadata.data_to_fit'
            }

    Args
        data_dict (dict): where the parameter values are saved under the keys
            of params_dict.
        params_dict (dict): with keys being the keys under which the parameter
            values will be stored in data_dict, and the values being
            parameter names or path + parameter name inside a file.
        numeric_params: list of parameter names from among the keys of
            params_dict. This specifies that those parameters are numbers and
            will be converted to floats.
        add_param_method: passed to add_param; see docstring there
        folder: path to files
        params: keyword arguments:
            passed to get_param, add_param, open_hdf_file, open_config_file
            close_file (bool; default: True): whether to close the files
    """

    def _get_data_dict_location(data_dict, save_par):
        """
        Helper function to get the location inside the data dictionary where to
        append the parameter.
        Args:
            data_dict(dict): dictionary where to save the parameter
            save_par (str): Name of the key where the parameter should be saved.
                Separations by '.' correspond to an additional layer in the
                nested dictionary.

        Returns: The location of the dictionary where the parameter should be
            saved in and save_par as a list split by '.'.
        """
        epd = data_dict
        all_keys = save_par.split('.')
        for i in range(len(all_keys) - 1):
            if all_keys[i] not in epd:
                epd[all_keys[i]] = OrderedDict()
            epd = epd[all_keys[i]]

        if isinstance(epd, list):
            epd = epd[-1]
        return epd, all_keys

    if params_dict is None:
        params_dict = get_param('params_dict', data_dict, raise_error=True,
                                **params)
    if numeric_params is None:
        numeric_params = get_param('numeric_params', data_dict,
                                   default_value=[], **params)

    # if folder is not specified, will take the last folder in the list from
    if folder is None:
        folder = get_param('folders', data_dict, raise_error=True, **params)
        if len(folder) > 0:
            folder = folder[-1]

    SETTINGS_PREFIX = 'Instrument settings.'
    settings_keys = [k for k, v in params_dict.items() if
                     v.startswith(SETTINGS_PREFIX)]
    settings_dict = {k: v[len(SETTINGS_PREFIX):] for k, v in
                     params_dict.items()
                     if k in settings_keys}
    params_dict = {k: v for k, v in params_dict.items()
                   if k not in settings_keys}

    h5mode = get_param('h5mode', data_dict, default_value='r', **params)
    data_file = a_tools.open_hdf_file(folder=folder, mode=h5mode, **params)

    if settings_keys:
        # Some hdf files only contain analysis information without instrument
        # settings. In that case trying to open the file leads to a KeyError
        # because group ['Instrument settings'] does not exist.
        config_station = setman.get_station_from_file(
            folder=folder, param_path=settings_dict.values(), **params)
    else:
        config_station = None

    try:
        for save_par, file_par in params_dict.items():
            epd, all_keys = _get_data_dict_location(data_dict, save_par)

            if file_par == 'measurementstring':
                add_param(all_keys[-1],
                          [os.path.split(folder)[1][7:]],
                          epd, add_param_method='append')
                continue
            tmp_param = extract_from_data_files([data_file], file_par)
            add_param(all_keys[-1], tmp_param,
                      epd, add_param_method=add_param_method)
        a_tools.close_files([data_file])
    except Exception as e:
        a_tools.close_files([data_file])
        raise e

    # adding instrument settings parameter
    for save_par, file_par in settings_dict.items():
        tmp_param = config_station.get(file_par)
        epd, all_keys = _get_data_dict_location(data_dict, save_par)
        add_param(all_keys[-1], tmp_param,
                  epd, add_param_method=add_param_method)

    for par_name in data_dict:
        if par_name in numeric_params:
            if hasattr(data_dict[par_name], '__iter__'):
                data_dict[par_name] = [float(p) for p
                                       in data_dict[par_name]]
                data_dict[par_name] = np.asarray(data_dict[par_name])
            else:
                data_dict[par_name] = float(data_dict[par_name])

    return data_dict


def extract_from_hdf_file(file, path_to_param):
    """
    Extracts the value of a parameter from an open HDF file. If a group name is
    given as path_to_param, it returns the hdf structure as a dictionary.

    Args:
        file (h5py._hl.files.File, h5py._hl.group.Group): an open HDF file or
            a group within an open HDF file
        path_to_param (str): key hierarchy inside a nested dict, separated by "."
            pointing to the parameter to be extracted
            Ex: 'Analysis.Processed data.analysis_params_dict.qb3.qscale'
                In this example, the qscale parameter will be returned.

    Returns:
        the value of the parameter
        If the parameter is not found, a ParameterNotFoundError is raised
    """
    group_name = '/'.join(path_to_param.split('.')[:-1])
    par_name = path_to_param.split('.')[-1]

    if group_name == '':
        # if path_to_param is already as the lowest layer
        group = file
        attrs = []
    else:
        try:
            # tries to go to the subgroup which contains the desired
            # parameter/group
            group = file[group_name]
        except KeyError:
            raise ParameterNotFoundError(path_to_param)
        attrs = list(group.attrs)

    if par_name in attrs:
        return decode_attribute_value(group.attrs[par_name])
    elif par_name in list(group.keys()) or path_to_param == '':
        # if parameter is a group itself
        par = group[par_name] if par_name != '' else group
        if isinstance(par, h5py._hl.dataset.Dataset):
            # if the parameter is a dataset, e.g. measurement data, it returns
            # the data as a numpy array
            return np.array(par)
        else:
            try:
                # It tries to return the entire subgroup as a dictionary.
                param_value = {}
                read_from_hdf(param_value, par, raise_exceptions=True)
                param_value = get_param(f'{group_name}.{par_name}', param_value)
                return param_value
            except Exception:
                # This is a legacy fallback. For instance, read_from_hdf
                # tries to instantiate a SweepPoints class but this might
                # fail for some legacy sweep points
                return read_dict_from_hdf5({}, par)

    # search through the keys and attributes of all groups
    # Like that one can directly request for attributes or groups from the
    # second-highest layer, e.g. path_to_param='value_names' yields the same
    # results as path_to_param='Experimental Data.value_names' for an attribute
    # or 'Processed Data' instead of 'Analysis.Processed Data'.
    for group_name in file.keys():
        if par_name in list(file[group_name].keys()):
            try:
                param_value = {}
                read_from_hdf(param_value, file[group_name][par_name],
                              raise_exceptions=True)
                return get_param(f'{group_name}.{par_name}',
                                        param_value)
            except Exception:
                # This is a legacy fallback. For instance, read_from_hdf
                # tries to instantiate a SweepPoints class but this might
                # fail for some legacy sweep points
                return read_dict_from_hdf5(
                    {}, file[group_name][par_name])
        if par_name in list(file[group_name].attrs):
            return decode_attribute_value(
                file[group_name].attrs[par_name])

    # param_value could not be found
    raise ParameterNotFoundError(path_to_param)


def extract_from_data_files(files, path_to_param):
    """
    Extracts the value of a parameter from the open files.

    Args:
        files (list): of open files in which to search for the parameter
        path_to_param (str): key hierarchy inside a nested dict, separated by "."
            pointing to the parameter to be extracted
            Ex: 'Analysis.Processed data.analysis_params_dict.qb3.qscale'
                In this example, the qscale parameter will be returned.

    Returns:
        the value of the parameter
        If the parameter is not found, ParameterNotFoundError is raised.
    """
    for i, file in enumerate(files):
        # searches in all files
        try:
            if isinstance(file, (h5py._hl.files.File, h5py._hl.group.Group)):
                # HDF file
                param_value = extract_from_hdf_file(file, path_to_param)
                # if no ParameterNotFoundError is raise, parameter is found and
                # search is completed.
                return param_value
            else:
                raise NotImplementedError('Currently only HDF files '
                                      'are supported.')
        except ParameterNotFoundError:
            if i == len(files)-1:
                raise


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
                data_to_proc_dict[keyi] = data_dict[all_keys[0]]
            except KeyError:
                key_found = False
        else:
            try:
                data = data_dict
                for k in all_keys:
                    data = data[k]
                data_to_proc_dict[keyi] = deepcopy(data)
            except KeyError:
                key_found = False
        if not key_found:
            raise ValueError(f'Channel {keyi} was not found.')
    return data_to_proc_dict


def find_all_in_dict(str_to_match, data_dict, split_char='.', key_prev_lev=''):
    """
    Find all keys in data_dict that contain the string str_to_match.

    Args:
        str_to_match (str): substring of a key that this function tries to match
        data_dict (dict): dict to be searched
        split_char (str): the character inserted between keys of data_dict to
            construct the key path
        key_prev_lev (str): string to prepend to the found key, separated by
            split_char

    Returns:
        dict of the form
        {f'{key_prev_lev}{split_char}{found_key}': data_dict[found_key]}
    """
    if len(key_prev_lev) > 0:
        key_prev_lev += split_char
    search_res = {}
    for k, v in data_dict.items():
        if isinstance(k, str) and str_to_match in k:
            search_res[f'{key_prev_lev}{k}'] = data_dict[k]

        if isinstance(v, dict):
            search_res.update(find_all_in_dict(
                str_to_match, v, key_prev_lev=f'{key_prev_lev}{k}'))
    return search_res


def get_param(param, data_dict, default_value=None, split_char='.',
              raise_error=False, error_message=None, find_all=False, **params):
    """
    Get the value of the parameter "param" from params, data_dict, or metadata.
    :param name: name of the parameter being sought
    :param data_dict: OrderedDict where param is to be searched
    :param default_value: default value for the parameter being sought in case
        it is not found.
    :param split_char: the character around which to split param
    :param raise_error: whether to raise error if the parameter is not found
    :param find_all: whether to return values for all the keys in data_dict that
        contain param
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
        all_keys = param.split(split_char)
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
    if value is None:
        if find_all:
            value = find_all_in_dict(param, data_dict, split_char=split_char)
            if len(value) == 0:
                # find_all_in_dict returns empty list if no matches are found
                value = None
    if value is None and raise_error:
        if error_message is None:
            error_message = f'{param} was not found in either data_dict, or ' \
                            f'exp_metadata or input params.'
        raise ValueError(error_message)
    return value


def pop_param(param, data_dict, default_value=None, split_char='.',
              raise_error=False, error_message=None, node_params=None):
    """
    Pop the value of the parameter "param" from params, data_dict, or metadata.
    :param name: name of the parameter being sought
    :param data_dict: OrderedDict where param is to be searched
    :param default_value: default value for the parameter being sought in case
        it is not found.
    :param split_char: the character around which to split param
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
        all_keys = param.split(split_char)
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


def add_param(name, value, data_dict, split_char='.',
              add_param_method=None, **params):
    """
    Adds a new key-value pair to the data_dict, with key = name.
    If update, it will try data_dict[name].update(value), else raises KeyError.
    :param name: key of the new parameter in the data_dict
    :param value: value of the new parameter
    :param data_dict: OrderedDict containing data to be processed
    :param split_char: the character around which to split param
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
    all_keys = name.split(split_char)
    if len(all_keys) > 1:
        for i in range(len(all_keys)-1):
            if isinstance(dd, list):
                all_keys[i] = int(all_keys[i])
            if not isinstance(dd, list) and all_keys[i] not in dd:
                dd[all_keys[i]] = {}
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
                           f'function. Please use get_params_from_files.')

    if len(props_to_return) == 1:
        props_to_return = props_to_return[0]

    return props_to_return


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
    """
    Retrieves the number of reset repetitions from a data dictionary.

    This function parses a data dictionary to find the number of reset
    repetitions. It supports both new and legacy formats for reset parameters.
    If 'reset_params' is found, it translates to the legacy format. If neither
    is set, it defaults to 0 reset repetitions.

    Args:
        data_dict (dict): The data dictionary containing experiment information,
            including metadata under 'exp_metadata'.

    Returns:
        int: The number of repetitions for reset operations, defaulting to 0
            if not explicitly specified in the data dictionary.
    """
    reset_reps = 0
    metadata = data_dict.get('exp_metadata', {})

    # Convert new active reset to legacy format
    if "reset_params" in metadata:
        prep_params = ba.BaseDataAnalysis.translate_reset_to_prep_params(
            metadata["reset_params"], default_value={}
        )

    # Extract reset_reps
    if "active" in prep_params.get("preparation_type", "wait"):
        reset_reps = prep_params.get("reset_reps", 3)

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


def read_analysis_file(timestamp=None, data_dict=None, ana_file=None,
                       close_file=True, file_id='_AnalysisResults',
                       **params):
    """
    Creates a data_dict from an AnalysisResults file as generated by analysis_v3

    Args:
        timestamp (str): with a measurement timestamp (YYYYMMDD_hhmmss)
        data_dict (dict): where to store the file entries
        ana_file (open HDF file): AnalysisResults file from which to load
        close_file (bool): whether to close the ana_file at the end
        file_id (str): suffix of the file name
        **params: keyword arguments passed to a_tools.open_hdf_file,
            see docstring there for acceptable input parameters

    Returns:
        the data_dict
    """

    if data_dict is None:
        data_dict = {}
    try:
        if ana_file is None:
            ana_file = a_tools.open_hdf_file(timestamp, file_id=file_id,
                                             **params)
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


def read_from_hdf(data_dict, hdf_group, split_char='.', raise_exceptions=False):
    """
    Adds to data_dict everything found in the HDF group or file hdf_group.
    :param data_dict: dict where the entries will be stored
    :param hdf_group: HDF group or file
    :return: nothing but updates data_dict with all values from hdf_group
    """
    try:
        if not len(hdf_group) and not len(hdf_group.attrs):
            path = hdf_group.name.split('/')[1:]
            add_param(split_char.join(path), {}, data_dict, split_char=split_char)

        for key, value in hdf_group.items():
            if isinstance(value, h5py.Group):
                read_from_hdf(data_dict, value, split_char, raise_exceptions)
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
                was_array = isinstance(val_to_store, np.ndarray)
                val_to_store = decode_attribute_value(val_to_store)
                if was_array:
                    val_to_store = np.array(val_to_store)
                try:
                    add_param(split_char.join(path), val_to_store, data_dict,
                              split_char=split_char)
                except Exception:
                    log.warning(f'Could not load path {split_char.join(path)}.')

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
                value = decode_attribute_value(value)
                if key == 'cal_points' and not isinstance(value, str):
                    value = repr(value)
            try:
                add_param(split_char.join(temp_path), value, data_dict,
                          split_char=split_char)
            except Exception:
                log.warning(f'Could not load path {split_char.join(path)}.')

        if 'list_type' in hdf_group.attrs:
            if (hdf_group.attrs['list_type'] == 'generic_list' or
                    hdf_group.attrs['list_type'] == 'generic_tuple'):
                list_dict = pop_param(split_char.join(path), data_dict)
                data_list = []
                for i in range(list_dict['list_length']):
                    data_list.append(list_dict[f'list_idx_{i}'])
                if hdf_group.attrs['list_type'] == 'generic_tuple':
                    data_list = tuple(data_list)
                if path[-1] == 'sweep_points':
                    # instantiate a SweepPoints class
                    data_list = sp_mod.SweepPoints(data_list)
                try:
                    add_param(split_char.join(path), data_list, data_dict,
                              add_param_method='replace', split_char=split_char)
                except Exception:
                    log.warning(f'Could not load path {split_char.join(path)}.')
            else:
                raise NotImplementedError('cannot read "list_type":"{}"'.format(
                    hdf_group.attrs['list_type']))
    except Exception as e:
        log.error(f"Unable to load: {hdf_group.name}.")
        if raise_exceptions:
            raise e
        else:
            log.error(traceback.format_exc())
