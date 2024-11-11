import logging
log = logging.getLogger(__name__)

import numpy as np
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod

import sys
pp_mod.search_modules.add(sys.modules[__name__])


def mean_squared_error(data_dict, keys_in, keys_out, labels,
                       sorted_by_label=False, **params):
    """
    Computes the MSE for a set of data points and their correspoding labels.

    Args:
        data_dict: OrderedDict containing data to be processed and where
            processed data is to be stored
        keys_in: list of key names or dictionary keys paths in
            data_dict for the data to be processed
        keys_out: list of key names or dictionary keys paths in
            data_dict for the processed data to be saved into
        labels: list of labels used to compute the error.
        sorted_by_label (str, default=False): Whether the data is sorted by
            label first (True) or by parameter (False).
        params: keyword arguments

    Assumptions:
        - if any keyo in keys_out contains a '.' string, keyo is assumed to
        indicate a path in the data_dict.
        - len(keys_out) == len(keys_in)
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    nr_param_sets = int(len(data_to_proc_dict[keys_in[0]])/len(labels))
    if sorted_by_label:
        data_array = np.reshape(data_to_proc_dict[keys_in[0]],
                                (-1, nr_param_sets)).T
    else:
        data_array = np.reshape(data_to_proc_dict[keys_in[0]],
                                (-1, len(labels)))
    hlp_mod.add_param(
            keys_out[0],
            np.array([np.mean((data - labels)**2) for data in data_array]),
            data_dict, **params)
    return data_dict