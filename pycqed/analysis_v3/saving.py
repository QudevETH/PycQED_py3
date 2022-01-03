import logging
log = logging.getLogger(__name__)

import os
import sys
import h5py
import lmfit
import datetime
import traceback
import numpy as np
import qutip as qtp
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import OrderedDict
from pycqed.measurement import hdf5_data as h5d
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import helper_functions as hlp_mod


class Save:
    """
    Saves figures and creates the HDF5 file
    measurement_name + "_AnalysisResults.hdf in the last folder in the list
    data_dict['folders'].
    The new file will contain everything in data_dict execept values
    corresponding to the keys "plot_dicts", "axes", "figures", "data_files."

    WARNING: this function will probably not work if data_dict contains
        more than one level of nested data_dicts.
        Ex: data_dict = {key: data_dict0} will work.
            But if data_dict0 = {key: data_dict1}, then it will not work.
    """
    def __init__(self, data_dict, savedir=None, save_processed_data=True,
                 save_figures=True, filename=None, extension='hdf5',
                 filter_keys=None, add_timestamp=False, **save_figs_params):

        opt = np.get_printoptions()
        np.set_printoptions(threshold=sys.maxsize)
        try:
            self.data_dict = data_dict
            if filter_keys is None:
                filter_keys = []
            self.filter_keys = filter_keys + ['fit_dicts', 'plot_dicts', 'axes',
                                              'figures', 'data_files']
            if savedir is None:
                savedir = hlp_mod.get_param('folders', data_dict)
                if savedir is None:
                    savedir = hlp_mod.get_param(
                        'timestamps', data_dict, raise_error=True,
                        error_message='Either folders or timestamps must be '
                                      'in data_dict if save_dir is not '
                                      'specified.')
                    savedir = a_tools.get_folder(savedir[-1])
                else:
                    savedir = savedir[-1]
            self.savedir = savedir

            if filename is None:
                filename = 'AnalysisResults'
            filename = self.savedir.split('\\')[-1] + f'_{filename}.{extension}'
            if add_timestamp:
                filename = '{:%Y%m%d_%H%M%S}--{}'.format(
                    datetime.datetime.now(), filename)
            self.filepath = self.savedir + '\\' + filename
            if save_processed_data:
                self.save_data_dict()
            if save_figures and hlp_mod.get_param('figures', self.data_dict) \
                    is not None:
                self.save_figures(**save_figs_params)

            np.set_printoptions(**opt)
        except Exception:
            np.set_printoptions(**opt)
            log.warning("Unhandled error during saving!")
            log.warning(traceback.format_exc())

    def save_data_dict(self):
        """
        Saves to the HDF5 file AnalysisResults.hdf everything in data_dict
        except values corresponding to the keys "plot_dicts", "axes",
        "figures", "data_files"
        """
        with h5py.File(self.filepath, 'a') as analysis_file:
            self.dump_to_file(self.data_dict, analysis_file)

    def dump_to_file(self, data_dict, entry_point):
        if 'fit_dicts' in data_dict:
            self.dump_fit_results(entry_point)

        # Iterate over all the fit result dicts as not to overwrite
        # old/other analysis
        for key, value in data_dict.items():
            if key not in self.filter_keys:
                try:
                    group = entry_point.create_group(key)
                except ValueError:
                    del entry_point[key]
                    group = entry_point.create_group(key)

                if isinstance(value, dict):
                    if value.get('is_data_dict', False):
                        self.dump_to_file(value, group)
                    else:
                        value_to_save = {}
                        for k, v in value.items():
                            if k not in self.filter_keys:
                                if isinstance(v, lmfit.model.ModelResult):
                                    frd = {f'{k}': _convert_dict_rec(v)}
                                    self.dump_to_file(frd,
                                                      entry_point=group)
                                else:
                                    value_to_save[k] = v
                        h5d.write_dict_to_hdf5(value_to_save,
                                               entry_point=group)
                elif isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value)
                elif isinstance(value, qtp.qobj.Qobj):
                    group.create_dataset(key, data=value.full())
                elif isinstance(value, lmfit.model.ModelResult):
                    h5d.write_dict_to_hdf5(_convert_dict_rec(value),
                                           entry_point=group)
                else:
                    try:
                        val = repr(value)
                    except KeyError:
                        val = ''
                    group.attrs[key] = val

    def dump_fit_results(self, entry_point):
        """
        Saves the fit results from data_dict['fit_dicts']
        :param entry_point: HDF5 file object to save to or a group within
            this file
        """
        try:
            group = entry_point.create_group('Fit Results')
        except ValueError:
            # If the analysis group already exists.
            group = entry_point['Fit Results']

        # Iterate over all the fit result dicts as not to overwrite
        # old/other analysis
        fit_dicts = hlp_mod.get_param('fit_dicts', self.data_dict)
        for fr_key, fit_dict in fit_dicts.items():
            fit_res = fit_dict['fit_res']
            try:
                fr_group = group.create_group(fr_key)
            except ValueError:
                # If the analysis sub group already exists
                # (each fr_key should be unique)
                # Delete the old group and create a new group
                # (overwrite).
                del group[fr_key]
                fr_group = group.create_group(fr_key)

            d = _convert_dict_rec(fit_res)
            h5d.write_dict_to_hdf5(d, entry_point=fr_group)

    def save_figures(self, **params):
        """
        Saves figures from self.data_dict['figures'].
        :param params: keyword arguments
            keys_in (list; default 'auto'): list of keys in data_dict['figures']
                denoting which figures to save. If "auto", all figures will be
                saved.
            fmt (str; default: 'png'): figures format
            dpi (int; default: 300): figures dpi
            tag_tstamp(bool; default: True): whether to add the last timestamp
                in data_dict['timestamps'] to the figure title
            savebase (str; default: None): base of the figure name
            presentation_mode (bool; default: False): whether to save and 'svg
                file in addition to the figure
        """
        keys_in = params.get('keys_in', 'auto')
        fmt = params.get('fmt', 'png')
        dpi = params.get('dpi', 300)
        tag_tstamp = params.get('tag_tstamp', True)
        savebase = params.get('savebase', None)

        if savebase is None:
            savebase = ''
        if tag_tstamp:
            timestamps = hlp_mod.get_param(
                'timestamps', self.data_dict, raise_error=True,
                error_message='"tag_tstamp" == True but "timestamps" not found '
                              'in data_dict.')
            tstag = '_' + timestamps[-1]
        else:
            tstag = ''

        # get figures from data_dict
        figs = hlp_mod.get_param('figures', self.data_dict, raise_error=True)
        if isinstance(figs, list):
            figs_dicts = OrderedDict()
            for fig_dict in figs:
                figs_dicts.update(fig_dict)
        else:
            figs_dicts = figs

        if keys_in == 'auto' or keys_in is None:
            keys_in = list(figs_dicts)

        for key in keys_in:
            if params.get('presentation_mode', False):
                savename = os.path.join(self.savedir, savebase + key + tstag +
                                        'presentation' + '.' + fmt)
                figs_dicts[key].savefig(savename, bbox_inches='tight',
                                        format=fmt, dpi=dpi)
                savename = os.path.join(self.savedir, savebase + key + tstag +
                                        'presentation' + '.svg')
                figs_dicts[key].savefig(savename, bbox_inches='tight',
                                        format='svg')
            else:
                savename = os.path.join(self.savedir, savebase + key + tstag
                                        + '.' + fmt)
                figs_dicts[key].savefig(savename, bbox_inches='tight',
                                        format=fmt, dpi=dpi)
            if params.get('close_figs', True):
                plt.close(figs_dicts[key])


def _convert_dict_rec(obj):
    try:
        obj_conv = deepcopy(obj)
        # is iterable?
        for k in obj_conv:
            obj_conv[k] = _convert_dict_rec(obj_conv[k])
    except TypeError:
        if isinstance(obj, lmfit.model.ModelResult):
            obj_conv = _flatten_lmfit_modelresult(obj)
        else:
            obj_conv = deepcopy(str(obj))
    return obj_conv


def _flatten_lmfit_modelresult(model):
    assert type(model) is lmfit.model.ModelResult
    dic = OrderedDict()
    dic['success'] = model.success
    dic['message'] = model.message
    dic['params'] = {}
    for param_name in model.params:
        dic['params'][param_name] = {}
        param = model.params[param_name]
        for k in param.__dict__:
            if k == '_val':
                dic['params'][param_name]['value'] = getattr(param, k)
            else:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    dic['params'][param_name][k] = getattr(param, k)
    return dic
