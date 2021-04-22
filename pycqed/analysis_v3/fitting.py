import logging
log = logging.getLogger(__name__)

from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.analysis import fitting_models as fit_mods
from collections import OrderedDict
import numpy as np
import lmfit

import sys
pp_mod.search_modules.add(sys.modules[__name__])

#####################################
### Functions related to Fitting ###
#####################################

def run_fitting(data_dict, keys_in='all', **params):
    """
    Fits the data dicts in dat_dict['fit_dicts'] specified by keys_in.
    Only model fitting is implemented here. Minimizing fitting should
    be implemented here.
    """
    fit_res_dict = {}
    if 'fit_dicts' not in data_dict:
        raise ValueError('fit_dicts not found in data_dict.')

    if keys_in == 'all':
        fit_dicts = data_dict['fit_dicts']
    else:
        fit_dicts = {fk: fd for fk, fd in data_dict['fit_dicts'].items() if
                     fk in keys_in}

    for fit_name, fit_dict in fit_dicts.items():
        fit_one_dict(fit_dict)
        for par in fit_dict['fit_res'].params:
            if fit_dict['fit_res'].params[par].stderr is None:
                fit_dict['fit_res'].params[par].stderr = 0
        fit_res_dict[fit_name] = fit_dict['fit_res']


def fit_one_dict(fit_dict, **params):
    """
    Does fitting to one fit_dict. Updates the fit_dict with the entry 'fit_res.'
    """
    guess_dict = fit_dict.get('guess_dict', None)
    guess_pars = fit_dict.get('guess_pars', None)
    guessfn_pars = fit_dict.get('guessfn_pars', {})
    fit_yvals = fit_dict['fit_yvals']
    fit_xvals = fit_dict['fit_xvals']

    model = fit_dict.get('model', None)
    if model is None:
        fit_fn = fit_dict.get('fit_fn', None)
        model = fit_dict.get('model', lmfit.Model(fit_fn))
    fit_guess_fn = fit_dict.get('fit_guess_fn', None)
    if fit_guess_fn is None and fit_dict.get('fit_guess', True):
        fit_guess_fn = model.guess

    fit_kwargs = fit_dict.get('fit_kwargs', {})
    if guess_pars is None:
        if fit_guess_fn is not None:
            # a fit function should return lmfit parameter
            # objects but can also work by returning a
            # dictionary of guesses
            guess_pars = fit_guess_fn(**fit_yvals, **fit_xvals,
                                      **guessfn_pars)
            if not isinstance(guess_pars, lmfit.Parameters):
                for gd_key, val in list(guess_pars.items()):
                    model.set_param_hint(gd_key, **val)
                guess_pars = model.make_params()

            if guess_dict is not None:
                for gd_key, val in guess_dict.items():
                    for attr, attr_val in val.items():
                        # e.g. setattr(guess_pars['frequency'],
                        # 'value', 20e6)
                        setattr(guess_pars[gd_key], attr,
                                attr_val)
            # A guess can also be specified as a dictionary.
            # additionally this can be used to overwrite values
            # from the guess functions.
        elif guess_dict is not None:
            for gd_key, val in list(guess_dict.items()):
                model.set_param_hint(gd_key, **val)
            guess_pars = model.make_params()
    fit_dict['fit_res'] = model.fit(**fit_xvals, **fit_yvals,
                                    params=guess_pars, **fit_kwargs)


def prepare_cos_fit_dict(data_dict, keys_in=None, **params):
    fit_dicts = OrderedDict()
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'], **params)
    indep_var_array = hlp_mod.get_param('indep_var_array', data_dict,
                                              raise_error=False, **params)
    if indep_var_array is None:
        indep_var_array = sp[0][mospm[mobjn][0]][0]
    plot_params = hlp_mod.get_param('plot_params', data_dict, default_value={},
                                    **params)
    if 'setlabel' not in plot_params:
        plot_params['setlabel'] = 'CosFit'
    params_to_print = hlp_mod.get_param(
        'params_to_print', data_dict, default_value=None, **params)

    fit_name = hlp_mod.get_param('fit_name', data_dict,
                                raise_error=False, **params)
    for keyi, data in data_to_proc_dict.items():
        data_fit = hlp_mod.get_msmt_data(data, cp, mobjn)
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=indep_var_array, data=data_fit)
        guess_pars['amplitude'].vary = True
        guess_pars['amplitude'].min = -10
        guess_pars['offset'].vary = True
        guess_pars['frequency'].vary = True
        guess_pars['phase'].vary = True

        fit_name_to_set = fit_name
        if fit_name_to_set is None:
            fit_name_to_set = 'CosFit'
        fit_name_to_set += keyi
        fit_dicts[fit_name_to_set] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': indep_var_array},
            'fit_yvals': {'data': data_fit},
            'guess_pars': guess_pars,
            'params_to_print': params_to_print, **plot_params}

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')


def prepare_joint_residzz_fit_dict(data_dict, keys_in=None, **params):
    """
    This function does a joint fit to Ramsey data without and with the other
    qubit in the |e> state.
    keys_in should have two entries corresponding to two 1d arrays for the
    data mentioned above, IN THAT ORDER.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param params: keyword args
        do_fitting (bool, default: False): whether to perform the fit
        guess_params (dict, default: dict()): dict of guess pars for fit
    :return: adds fit_dicts to data_dict
    """
    if len(keys_in) != 2:
        raise ValueError('keys_in must have two entries.')
    fit_dicts = OrderedDict()
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'], **params)
    indep_var_array = hlp_mod.get_param('indep_var_array', data_dict,
                                              raise_error=False, **params)
    if indep_var_array is None:
        indep_var_array = sp[0][mospm[mobjn][0]][0]
    plot_params = hlp_mod.get_param('plot_params', data_dict, default_value={},
                                    **params)
    if 'setlabel' not in plot_params:
        plot_params['setlabel'] = 'JointResidZZFit'
    params_to_print = hlp_mod.get_param(
        'params_to_print', data_dict, default_value=None, **params)

    data_wo_pulse = hlp_mod.get_msmt_data(
        list(data_to_proc_dict.values())[0], cp, mobjn)
    data_w_pulse = hlp_mod.get_msmt_data(
        list(data_to_proc_dict.values())[1], cp, mobjn)

    residzz_mod = lmfit.Model(fit_mods.ResidZZFuncJoint)
    guess_pars = fit_mods.exp_damp_osc_guess(
        model=residzz_mod, t=indep_var_array,
        data=data_wo_pulse)

    guess_pars['alpha'].value = -50e3
    guess_pars['alpha'].max = 0
    # guess_pars['x'].value = 12e-6*guess_pars['alpha'].value
    guess_pars['t11'].value = 12e-6
    guess_pars['t11'].min = 0
    guess_pars['offset'].value = np.mean(data_wo_pulse)
    guess_pars['amplitude1'].value = guess_pars['amplitude'].value
    # guess_pars['phase1'].value = guess_pars['phase'].value + np.pi/2
    # guess_pars['amplitude'].min = -1
    # guess_pars['amplitude1'].min = -1
    # guess_pars['amplitude'].max = 1
    # guess_pars['amplitude1'].max = 1
    for par in guess_pars:
        guess_pars[par].vary = True
    guess_pars['offset'].vary = False
    guess_params_new = hlp_mod.get_param('guess_params', data_dict,
                                                default_value=dict(),
                                                raise_error=False, **params)
    update_fit_guess_pars(guess_params_new, guess_pars)

    fit_name = hlp_mod.get_param('fit_name', data_dict,
                                 raise_error=False, **params)
    fit_name_to_set = fit_name
    if fit_name_to_set is None:
        fit_name_to_set = 'residzz_fit'
    fit_name_to_set += ','.join(mobjn)
    fit_dicts[fit_name] = {
        'fit_fn': fit_mods.ResidZZFuncJoint,
        'fit_xvals': {'t': indep_var_array},
        'fit_yvals': {'data': (data_wo_pulse, data_w_pulse)},
        'guess_pars': guess_pars,
        'params_to_print': params_to_print, **plot_params}

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')
    if params.get('do_fitting', False):
        run_fitting(data_dict, keys_in=list(fit_dicts), **params)


def prepare_residzz_fit_dict(data_dict, keys_in=None, **params):
    """
    This function does a fit to Ramsey data with the other
    qubit in the |e> state.
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param params: keyword args
        do_fitting (bool, default: False): whether to perform the fit
        guess_params (dict, default: dict()): dict of guess pars for fit
    :return: adds fit_dicts to data_dict
    """
    fit_dicts = OrderedDict()
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'], **params)
    indep_var_array = hlp_mod.get_param('indep_var_array', data_dict,
                                              raise_error=False, **params)
    if indep_var_array is None:
        indep_var_array = sp[0][mospm[mobjn][0]][0]
    plot_params = hlp_mod.get_param('plot_params', data_dict, default_value={},
                                    **params)
    if 'setlabel' not in plot_params:
        plot_params['setlabel'] = 'ResidZZFit'
    params_to_print = hlp_mod.get_param(
        'params_to_print', data_dict, default_value=None, **params)

    fit_name = hlp_mod.get_param('fit_name', data_dict,
                                raise_error=False, **params)
    for keyi, data in data_to_proc_dict.items():
        data_fit = hlp_mod.get_msmt_data(data, cp, mobjn)
        residzz_mod = lmfit.Model(fit_mods.ResidZZFunc)
        guess_pars = fit_mods.exp_damp_osc_guess(
            model=residzz_mod, t=indep_var_array, data=data_fit)
        guess_pars['alpha'].value = -50e3
        guess_pars['x'].value = 12e-6*guess_pars['alpha'].value
        guess_pars['offset'].value = np.mean(data_fit)
        # guess_pars['amplitude'].min = -1
        # guess_pars['amplitude'].max = 1

        guess_params_new = hlp_mod.get_param('guess_params', data_dict,
                                                   default_value=dict(),
                                                   raise_error=False, **params)
        update_fit_guess_pars(guess_params_new, guess_pars)

        fit_name_to_set = fit_name
        if fit_name_to_set is None:
            fit_name_to_set = 'residzz_fit'
        fit_name_to_set += keyi
        fit_dicts[fit_name_to_set] = {
            'fit_fn': fit_mods.ResidZZFunc,
            'fit_xvals': {'t': indep_var_array},
            'fit_yvals': {'data': data_fit},
            'guess_pars': guess_pars,
            'params_to_print': params_to_print, **plot_params}

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')
    if params.get('do_fitting', False):
        run_fitting(data_dict, keys_in=list(fit_dicts), **params)


def prepare_expdamposc_fit_dict(data_dict, keys_in=None, **params):
    """
    This function does a to Ramsey data
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param params: keyword args
        do_fitting (bool, default: False): whether to perform the fit
        guess_params (dict, default: dict()): dict of guess pars for fit
        fit_name
        indep_var_array
        plot_params
        params_to_print
    :return: adds fit_dicts to data_dict
    """
    fit_dicts = OrderedDict()
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'], **params)
    indep_var_array = hlp_mod.get_param('indep_var_array', data_dict,
                                        raise_error=False, **params)
    if indep_var_array is None:
        indep_var_array = sp[0][mospm[mobjn][0]][0]
    plot_params = hlp_mod.get_param('plot_params', data_dict, default_value={},
                                    **params)
    if 'setlabel' not in plot_params:
        plot_params['setlabel'] = 'ExpDampOscFit'
    params_to_print = hlp_mod.get_param(
        'params_to_print', data_dict, default_value=None, **params)

    fit_name = hlp_mod.get_param('fit_name', data_dict,
                                raise_error=False, **params)
    for keyi, data in data_to_proc_dict.items():
        data_fit = hlp_mod.get_msmt_data(data, cp, mobjn)
        exp_damped_decay_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
        guess_pars = fit_mods.exp_damp_osc_guess(
            model=exp_damped_decay_mod, data=data_fit, t=indep_var_array,
            n_guess=1)
        guess_pars['amplitude'].vary = False
        guess_pars['amplitude'].value = 0.5
        guess_pars['frequency'].vary = True
        guess_pars['tau'].vary = True
        guess_pars['phase'].vary = True
        guess_pars['n'].vary = False
        guess_pars['oscillation_offset'].vary = False
        guess_pars['exponential_offset'].vary = True
        guess_params_new = hlp_mod.get_param('guess_params', data_dict,
                                             default_value=dict(),
                                             raise_error=False, **params)
        update_fit_guess_pars(guess_params_new, guess_pars)

        fit_name_to_set = fit_name
        if fit_name_to_set is None:
            fit_name_to_set = 'expdamposc_fit'
        fit_name_to_set += keyi
        fit_dicts[fit_name_to_set] = {
            'fit_fn': fit_mods.ExpDampOscFunc,
            'fit_xvals': {'t': indep_var_array},
            'fit_yvals': {'data': data_fit},
            'guess_pars': guess_pars,
            'params_to_print': params_to_print,
            'plot_params': plot_params}

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_fitting', False):
        run_fitting(data_dict, keys_in=list(fit_dicts), **params)


def prepare_rbleakage_fit_dict(data_dict, keys_in=None, **params):
    """
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param params: keyword args
        do_fitting (bool, default: False): whether to perform the fit
        guess_params (dict, default: dict()): dict of guess pars for fit
    :return: adds fit_dicts to data_dict
    """
    fit_dicts = OrderedDict()
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'], **params)
    indep_var_array = hlp_mod.get_param('indep_var_array', data_dict,
                                              raise_error=False, **params)
    if indep_var_array is None:
        indep_var_array = sp[0][mospm[mobjn][0]][0]
    plot_params = hlp_mod.get_param('plot_params', data_dict, default_value={},
                                    **params)
    if 'setlabel' not in plot_params:
        plot_params['setlabel'] = 'RBLeakageFit'
    params_to_print = hlp_mod.get_param(
        'params_to_print', data_dict, default_value=None, **params)

    fit_name = hlp_mod.get_param('fit_name', data_dict,
                                raise_error=False, **params)
    for keyi, data in data_to_proc_dict.items():
        data_fit = hlp_mod.get_msmt_data(data, cp, mobjn)
        rbleak_mod = lmfit.Model(fit_mods.RandomizedBenchmarkingLeakage)
        guess_pars = rbleak_mod.make_params(pu=0.01, pd=0.05, p0=0)

        guess_params_new = hlp_mod.get_param('guess_params', data_dict,
                                                   default_value=dict(),
                                                   raise_error=False, **params)
        update_fit_guess_pars(guess_params_new, guess_pars)

        fit_name_to_set = fit_name
        if fit_name_to_set is None:
            fit_name_to_set = 'rbleak_fit'
        fit_name_to_set += keyi
        fit_dicts[fit_name_to_set] = {
                'fit_fn': fit_mods.RandomizedBenchmarkingLeakage,
                'fit_xvals': {'numCliff': indep_var_array},
                'fit_yvals': {'data': data_fit},
                'guess_pars': guess_pars,
                'params_to_print': params_to_print, **plot_params}

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')
    if params.get('do_fitting', False):
        run_fitting(data_dict, keys_in=list(fit_dicts), **params)


def update_fit_guess_pars(guess_params_new, guess_params_old):
    if len(guess_params_new) != 0:
        for par, val in guess_params_new.items():
            if isinstance(val, dict):
                if 'value' in val:
                    guess_params_old[par].value = val['value']
                if 'min' in val:
                    guess_params_old[par].min = val['min']
                if 'max' in val:
                    guess_params_old[par].max = val['max']
                if 'vary' in val:
                    guess_params_old[par].max = val['vary']
            else:
                # assumes the value corresponding to par is an int or float
                guess_params_old[par].value = val