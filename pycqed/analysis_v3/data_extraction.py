import h5py
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import logging
log = logging.getLogger(__name__)

from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.measurement import sweep_points as sp_mod

import sys
pp_mod.search_modules.add(sys.modules[__name__])


def get_timestamps(data_dict=None, t_start=None, t_stop=None,
                   label='', data_file_path=None, **params):
    """
    Get timestamps (YYYYMMDD_hhmmss) of HDF files from a specified location.

    Args:
        data_dict (dict or OrderedDict): the extracted timestamps will be
            stored here
        t_start (str): timestamp of the form YYYYMMDD_hhmmss. This timestamp
            is returned if t_stop is None, and otherwise it is the first
            timestamp of the range [t_start, t_stop]
        t_stop (str): timestamp of the form YYYYMMDD_hhmmss. The last timestamp
            to be extracted, starting at t_start
        label (str): if specified, only those timestamps are returned for which
            this label is contained in the filename
        data_file_path (str): full path to a datafile for which the timestamp
            will be returned

    Keyword args (**params)
        passed to analysis_tools.py/latest_data and
            analysis_tools.py/get_timestamps_in_range

    Returns
        data dict containing the timestamps
    """
    # If I put data_dict = OrderedDict() in the input params, somehow this
    # function sees the data_dict I have in my notebook. How???
    if data_dict is None:
        data_dict = OrderedDict()

    timestamps = None
    if data_file_path is None:
        if t_start is None:
            if isinstance(label, list):
                timestamps = [a_tools.latest_data(
                    contains=l, return_timestamp=True, **params)[0]
                              for l in label]
            else:
                timestamps = [a_tools.latest_data(
                    contains=label, return_timestamp=True, **params)[0]]
        elif t_stop is None:
            if isinstance(t_start, list):
                timestamps = t_start
            else:
                timestamps = [t_start]
        else:
            timestamps = a_tools.get_timestamps_in_range(
                t_start, timestamp_end=t_stop,
                label=label if label != '' else None, **params)

    if timestamps is None or len(timestamps) == 0:
        raise ValueError('No data file found.')

    data_dict['timestamps'] = timestamps
    return data_dict


def extract_data_hdf(data_dict=None, timestamps=None,
                     params_dict=OrderedDict(), numeric_params=None,
                     append_data=False, replace_data=False, **params):
    """
    Extracts the data specified in params_dict and pumeric_params
    from each timestamp in timestamps and stores it into data_dict

    Args:
        data_dict (dict): place where extracted data will be stored, under the
            keys of params_dict
        timestamps (list): list of timestamps from where to extract data. If
            not specified, they will be taken from data_dict, and, if not found
            there, from get_timestamps
        params_dict (dict): if the form {storing_key: path_to_data}, where
            storing_key will be created in data_dict for storing the data
            indicated by path_to_data as a parameter name or a
            path + parameter name inside an HDF file.
        numeric_params (list or tuple): passed to get_params_from_hdf_file, see
            docstring there.
        append_data (bool): passed to add_measured_data_hdf, see docstring there
        replace_data (bool): passed to add_measured_data_hdf, see docstring
            there

    Keyword args (**params)
        passed to get_timestamps and add_measured_data_dict

    Returns
        data_dict containing the extracted data
    """
    if data_dict is None:
        data_dict = OrderedDict()

    # Add flag that this is an analysis_v3 data_dict. This is used by the
    # Saving class.
    data_dict['is_data_dict'] = True

    if timestamps is None:
        timestamps = hlp_mod.get_param('timestamps', data_dict)
    if timestamps is None:
        get_timestamps(data_dict, **params)
        timestamps = hlp_mod.get_param('timestamps', data_dict)
    if isinstance(timestamps, str):
        timestamps = [timestamps]
    hlp_mod.add_param('timestamps', timestamps, data_dict,
                      add_param_method='replace')

    data_dict['folders'] = []

    for i, timestamp in enumerate(timestamps):
        folder = a_tools.get_folder(timestamp)
        data_dict['folders'] += [folder]

        # extract the data array and add it as data_dict['measured_data'].
        add_measured_data_hdf(data_dict, folder, append_data, replace_data)

        # extract exp_metadata separately, then call
        # combine_metadata_list, then extract all other parameters.
        # Otherwise, data_dict['exp_metadata'] is a list and it is unclear
        # from where to extract parameters.
        hlp_mod.get_params_from_hdf_file(
            data_dict,
            params_dict={'exp_metadata':
                             'Experimental Data.Experimental Metadata'},
            folder=folder, add_param_method='append')

    if len(timestamps) > 1:
        # If data_dict['exp_metadata'] is a list, then the following functions
        # defines exp_metadata in data_dict as the combined version of the list
        # of metadata dicts extracted above for each timestamp
        combine_metadata_list(data_dict, **params)

    # call get_params_from_hdf_file which gets values for params
    # in params_dict and adds them to the dictionary data_dict
    params_dict_temp = params_dict
    params_dict = OrderedDict(
        {'exp_metadata.sweep_parameter_names': 'sweep_parameter_names',
         'exp_metadata.sweep_parameter_units': 'sweep_parameter_units',
         'exp_metadata.value_names': 'value_names',
         'exp_metadata.value_units': 'value_units',
         'exp_metadata.measurementstrings': 'measurementstring'})
    params_dict.update(params_dict_temp)
    hlp_mod.get_params_from_hdf_file(
        data_dict, params_dict=params_dict, numeric_params=numeric_params,
        folder=data_dict['folders'][-1],
        add_param_method=params.get('add_param_method', 'replace'))

    # add entries in data_dict for each readout channel and its corresponding
    # data array.
    add_measured_data_dict(data_dict, **params)

    return data_dict


def combine_metadata_list(data_dict, update_value=True, append_value=True,
                          replace_value=False, **params):
    """
    Combines a list of metadata dictionaries into one dictionary. Whenever
    entries are the same it keep only one copy. Whenever entries are different,
    it can append, update, or replace them.
    :param data_dict: OrderedDict that contains 'exp_metadata' as a list of
        dicts
    :param update_value: bool that applies in the case where two keys that are
        dicts are the same and specifies whether to update one of the dicts
        with the other.
    :param append_value: bool that applies in the case where two keys are the
        same and specifies whether to append them in a list.
    :param replace_value: bool that applies in the case where two keys are the
        same and specifies whether to replace the value of the key.
    :param params:
    :return: nothing but saves exp_metadata_list (original list of metadata
        dicts) and exp_metadata (the combined metadata dict) to data_dict
    """
    metadata = hlp_mod.pop_param('exp_metadata', data_dict,
                                  default_value=OrderedDict())
    data_dict['exp_metadata_list'] = metadata
    if isinstance(metadata, list):
        metadata_list = deepcopy(metadata)
        metadata0 = metadata_list[0]
        metadata = deepcopy(metadata0)
        for i, md_dict in enumerate(metadata_list[1:]):
            for key, value in md_dict.items():
                if key in metadata:
                    if not hlp_mod.check_equal(metadata0[key], value):
                        if isinstance(metadata0[key], dict) and update_value:
                            if not isinstance(value, dict):
                                raise ValueError(
                                    f'The value corresponding to {key} in  '
                                    f'metadata list {i+1} is not a dict. '
                                    f'Cannot update_value.')
                            metadata[key].update(value)
                        elif append_value:
                            if i == 0:
                                metadata[key] = [metadata[key]]
                            if not isinstance(value, list):
                                value = [value]
                            metadata[key].append(value)
                        elif replace_value:
                            metadata[key] = value
                        else:
                            raise KeyError(f'{key} already exists in '
                                           f'combined metadata and it"s unclear'
                                           f' how to add it.')
                    else:
                        metadata[key] = value
                else:
                    metadata[key] = value
    data_dict['exp_metadata'] = metadata


def add_measured_data_hdf(data_dict, folder=None, append_data=False,
                          replace_data=False, **params):
    """
    Extracts the dataset from the "Experimental Data" data of an HDF file and
    stores it in data_dict under "measured_data".

    Args:
        data_dict (dict): extracted dataset will be stored here under the key
            "measured_data"
        folder (str): full path to an hdf file
        append_data (bool): if True, and "measured_data" exists in data_dict,
            the new dataset will be appended
        replace_data (bool): if True, and "measured_data" exists in data_dict,
            it will be replaced by the new dataset

    Keyword args (**params)
        passed to helper_functions.py/add_param, see docstring there

    Returns
        data_dict containing the extracted dataset
    """
    if folder is None:
        folder = hlp_mod.get_param('folders', data_dict, raise_error=True,
                                   **params)
        if len(folder) > 0:
            folder = folder[-1]

    h5mode = hlp_mod.get_param('h5mode', data_dict, default_value='r+',
                               **params)
    dtype = hlp_mod.get_param('meas_data_dtype', data_dict, default_value=None,
                               **params)
    if dtype is not None:
        log.warning(f'Setting Experimental data type: {dtype}')
    h5filepath = a_tools.measurement_filename(folder)
    data_file = h5py.File(h5filepath, h5mode)
    meas_data_array = np.array(data_file['Experimental Data']['Data'],
                               dtype=dtype).T
    if 'measured_data' in data_dict:
        if replace_data:
            data_dict['measured_data'] = meas_data_array
            data_dict['data_replaced'] = True
        elif append_data:
            if not isinstance(data_dict['measured_data'], list):
                data_dict['measured_data'] = [data_dict['measured_data']]
            data_dict['measured_data'] += [meas_data_array]
            data_dict['data_appended'] = True
        else:
            raise ValueError('Both "append_data" and "replace_data" are False. '
                             'Unclear how to add "measured_data" to data_dict.')
    else:
        data_dict['measured_data'] = meas_data_array
    return data_dict


def add_measured_data_dict(data_dict, **params):
    """
    Adds to the data_dict the acquisition channels with the corresponding data
    array taken from "measured_data," which must exist in data_dict.

    Args:
        data_dict (dict): the measured data dict will be stored here

    Keyword args (**params)
        passed to helper_functions.py/get_param, helper_functions.py/pop_param,
        and helper_functions.py/add_param, see docstrings there

        This function expects to find the following parameters:
         - exp_metadata must exist in data_dict
         - value_names must exist in exp_metadata
         - meas_obj_value_names_map must exist in data_dict

        Other possible keywargs
         - TwoD (bool; default: False) whether the measurement had two
            sweep dimensions
         - compression_factor (int; default: 1) sequence compression factor
            (see Sequence.compress_2D_sweep)
         - percentage_done (int; default: 100) for interrupted measurements;
            indicates percentage of the total data that was acquired

    Returns
        data_dict containing the following entries:
            - acquisition channels (keys of the meas_obj_value_names_map) with
                the corresponding raw data
            - sweep points (if not already in data_dict)
    """
    metadata = hlp_mod.get_param('exp_metadata', data_dict, raise_error=True)
    TwoD = hlp_mod.get_param('TwoD', data_dict, default_value=False,
                             **params)
    compression_factor = hlp_mod.get_param('compression_factor',
                                           data_dict, default_value=1,
                                           **params)
    if 'measured_data' in data_dict and 'value_names' in metadata:
        value_names = metadata['value_names']
        rev_movnm = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['rev_movnm'])

        if rev_movnm is not None:
            data_key = lambda ro_ch, rev_movnm=rev_movnm: f'{rev_movnm[ro_ch]}.{ro_ch}'
        else:
            data_key = lambda ro_ch: ro_ch
        [hlp_mod.add_param(data_key(ro_ch), [], data_dict)
         for ro_ch in value_names]
        measured_data = data_dict.pop('measured_data')

        if not isinstance(measured_data, list):
            measured_data = [measured_data]

        for meas_data in measured_data:
            data = meas_data[-len(value_names):]
            if data.shape[0] != len(value_names):
                raise ValueError('Shape mismatch between data and ro channels.')

            mc_points = meas_data[:-len(value_names)]
            hsp = np.unique(mc_points[0])
            if mc_points.shape[0] > 1:
                ssp, counts = np.unique(mc_points[1:], return_counts=True)
                if counts[0] != len(hsp):
                    # ssro data
                    hsp = np.tile(hsp, counts[0]//len(hsp))
                # if needed, decompress the data
                # (assumes hsp and ssp are indices)
                if compression_factor != 1:
                    hsp = hsp[:int(len(hsp) / compression_factor)]
                    ssp = np.arange(len(ssp) * compression_factor)
            for i, ro_ch in enumerate(value_names):
                if TwoD:
                    meas_data = np.reshape(data[i], (len(ssp), len(hsp))).T
                else:
                    meas_data = data[i]
                hlp_mod.add_param(data_key(ro_ch), [meas_data], data_dict,
                                  add_param_method='append')

        for ro_ch in value_names:
            data = hlp_mod.pop_param(data_key(ro_ch), data_dict)
            if len(data) == 1:
                hlp_mod.add_param(data_key(ro_ch), data[0], data_dict)
            else:
                hlp_mod.add_param(data_key(ro_ch), np.array(data), data_dict)

        # check for and deal with interrupted measurements
        perc_done = hlp_mod.get_param('percentage_done', data_dict,
                                      default_value=100, **params)
        if perc_done < 100 and mc_points.shape[0] > 1:
            sp = hlp_mod.get_measurement_properties(data_dict,
                                                    props_to_extract=['sp'],
                                                    raise_error=False)
            if len(sp):  # above call returns [] if sp not found
                sp_new = sp_mod.SweepPoints([sp.get_sweep_dimension(0)])
                sp_new.add_sweep_dimension()
                for param_name, props_tup in sp.get_sweep_dimension(1).items():
                    sp_new.add_sweep_parameter(param_name, props_tup[0][:len(ssp)],
                                               props_tup[1], props_tup[2])
                hlp_mod.add_param('sweep_points', sp_new, data_dict, **params)
            else:
                log.warning('sweep_points not found. Cannot deal with '
                            'interrupted measurements.')
    else:
        raise ValueError('Either "measured_data" was not found in data_dict '
                         'or "value_names" was not found in metadata. '
                         '"measured_data" was not added.')
    return data_dict
