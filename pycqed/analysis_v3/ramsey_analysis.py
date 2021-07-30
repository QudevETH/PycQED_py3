import logging
log = logging.getLogger(__name__)

import numpy as np
from collections import OrderedDict
from pycqed.analysis_v3 import fitting as fit_mod
from pycqed.analysis_v3 import plotting as plot_mod
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from copy import deepcopy

import sys
pp_mod.search_modules.add(sys.modules[__name__])

# Create pipelines
def ramsey_iq_pipeline(meas_object_name):
    pp = pp_mod.ProcessingPipeline()
    pp.add_node('rotate_iq', keys_in='raw', meas_obj_names=meas_object_name)
    pp.add_node('ramsey_analysis',
                keys_in=f'previous {meas_object_name}.rotate_iq',
                keys_out=None,
                meas_obj_names=meas_object_name)
    return pp


# run ramsey analysis
def ramsey_analysis(data_dict, keys_in, **params):
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)

    prep_fit_dicts = params.pop('prep_fit_dicts', True)
    do_fitting = params.pop('do_fitting', True)
    prepare_plotting = params.pop('prepare_plotting', True)
    do_plotting = params.pop('do_plotting', True)

    # prepare fitting
    if prep_fit_dicts:
        fit_gaussian_decay = hlp_mod.get_param('fit_gaussian_decay', data_dict,
                                               default_value=True, **params)
        if fit_gaussian_decay:
            fit_names = ['exp_decay', 'gauss_decay']
        else:
            fit_names = ['exp_decay']
        params.update({'fit_names': fit_names})
        prepare_ramsey_fitting(data_dict, keys_in, **params)

    if do_fitting:
        getattr(fit_mod, 'run_fitting')(data_dict, keys_in=list(
                data_dict['fit_dicts']),**params)
        # calculate new qubit frequecy, extract T2 star
        analyze_ramsey_fit_results(data_dict, keys_in, **params)

    # prepare plots
    if prepare_plotting:
        prepare_ramsey_plots(data_dict, data_to_proc_dict, **params)
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict, keys_in=list(
            data_dict['plot_dicts']), **params)


def prepare_ramsey_fitting(data_dict, keys_in, **params):
    fit_names = hlp_mod.get_param('fit_names', params, raise_error=True)
    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    physical_swpts = sp[0][mospm[mobjn][0]][0]
    for i, fit_name in enumerate(fit_names):
        fit_mod.prepare_expdamposc_fit_dict(
            data_dict, keys_in=keys_in,
            meas_obj_names=mobjn, fit_name=fit_name,
            indep_var_array=physical_swpts,
            guess_params={'n': i+1},
            plot_params={'color': 'r'if i == 0 else 'C4',
                         'setlabel': 'expFit' if i == 0 else 'gaussFit'})


def analyze_ramsey_fit_results(data_dict, keys_in, **params):
    # Get from the hdf5 file any parameters specified in
    # params_dict and numeric_params.
    mobjn = hlp_mod.get_measurement_properties(data_dict,
                                               props_to_extract=['mobjn'],
                                               **params)
    params_dict = {}
    s = 'Instrument settings.' + mobjn
    for trans_name in ['ge', 'ef']:
        params_dict[f'{mobjn}.{trans_name}_freq'] = s + f'.{trans_name}_freq'
    hlp_mod.get_params_from_hdf_file(data_dict, params_dict=params_dict,
                                     numeric_params=list(params_dict), **params)
    fit_names = hlp_mod.get_param('fit_names', params, raise_error=True)
    artificial_detuning_dict = hlp_mod.get_param('artificial_detuning_dict',
                                                 data_dict, raise_error=True,
                                                 **params)

    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    for keyi in keys_in:
        trans_name = 'ef' if 'f' in keyi else 'ge'
        old_qb_freq = hlp_mod.get_param(f'{mobjn}.{trans_name}_freq', data_dict)
        if old_qb_freq != old_qb_freq:
            old_qb_freq = 0
        hlp_mod.add_param(f'{mobjn}.old_freq', old_qb_freq, data_dict,
                          add_param_method='replace')
        for fit_name in fit_names:
            key = fit_name + keyi
            fit_res = fit_dicts[key]['fit_res']
            hlp_mod.add_param(f'{mobjn}.new_freq ' + fit_name,
                              old_qb_freq + artificial_detuning_dict[mobjn] -
                              fit_res.best_values['frequency'],
                              data_dict, add_param_method='replace')
            hlp_mod.add_param(f'{mobjn}.new_freq ' + fit_name + '_stderr',
                              fit_res.params['frequency'].stderr,
                              data_dict, add_param_method='replace')
            hlp_mod.add_param(f'{mobjn}.T2_star ' + fit_name,
                              fit_res.best_values['tau'],
                              data_dict, add_param_method='replace')
            hlp_mod.add_param(f'{mobjn}.T2_star ' + fit_name + '_stderr',
                              fit_res.params['tau'].stderr,
                              data_dict, add_param_method='replace')


def prepare_ramsey_plots(data_dict, data_to_proc_dict, **params):

    cp, sp, mospm, mobjn = \
        hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'],
            **params)
    physical_swpts = sp[0][mospm[mobjn][0]][0]

    # check whether active reset was used
    reset_reps = hlp_mod.get_reset_reps_from_data_dict(data_dict)
    # prepare raw data plot
    if reset_reps != 0:
        swpts = deepcopy(physical_swpts)
        swpts = np.concatenate([
            swpts, hlp_mod.get_cal_sweep_points(
                physical_swpts, cp, mobjn)])
        swpts = np.repeat(swpts, reset_reps+1)
        swpts = np.arange(len(swpts))
        plot_mod.prepare_1d_raw_data_plot_dicts(
            data_dict, xvals=swpts, **params)

        filtered_raw_keys = [k for k in data_dict.keys() if 'filter' in k]
        if len(filtered_raw_keys) > 0:
            plot_mod.prepare_1d_raw_data_plot_dicts(
                data_dict=data_dict, keys_in=filtered_raw_keys,
                figure_name='raw_data_filtered', **params)
    else:
        plot_mod.prepare_1d_raw_data_plot_dicts(data_dict, **params)

    fit_names = hlp_mod.pop_param('fit_names', params, raise_error=True)
    artificial_detuning_dict = hlp_mod.get_param('artificial_detuning_dict',
                                                 data_dict, raise_error=True,
                                                 **params)
    plot_dicts = OrderedDict()
    # the prepare plot dict functions below also iterate over data_to_proc_dict,
    # however the prepare functions add all the data corresponding to keys_in
    # to the same figure.
    # Here we want a figure for each keyi
    for keyi, data in data_to_proc_dict.items():
        figure_name = 'Ramsey_' + keyi
        sp_name = mospm[mobjn][0]
        # plot data
        plot_mod.prepare_1d_plot_dicts(data_dict=data_dict, keys_in=[keyi],
                                          figure_name=figure_name,
                                          sp_name=sp_name, do_plotting=False,
                                          **params)

        if len(cp.states) != 0:
            # plot cal states
            plot_mod.prepare_cal_states_plot_dicts(data_dict=data_dict,
                                                      keys_in=[keyi],
                                                      figure_name=figure_name,
                                                      sp_name=sp_name,
                                                      do_plotting=False,
                                                      **params)

        if 'fit_dicts' in data_dict:
            textstr = ''
            T2_star_str = ''
            for i, fit_name in enumerate(fit_names):
                plot_mod.prepare_fit_plot_dicts(
                    data_dict=data_dict,
                    figure_name=figure_name,
                    fit_names=[fit_name + keyi],
                    plot_params={'legend_bbox_to_anchor': (1, -0.55),
                                 'legend_ncol': 1},
                    do_plotting=False, **params)

                fit_res = data_dict['fit_dicts'][
                    fit_name + keyi]['fit_res']
                if i != 0:
                    textstr += '\n'
                textstr += \
                    ('$f_{{qubit \_ new \_ {{{key}}} }}$ = '.format(
                        key=('exp' if i == 0 else 'gauss')) +
                     '{:.6f} GHz '.format(hlp_mod.get_param(
                         f'{mobjn}.new_freq '+fit_name, data_dict)*1e-9) +
                     '$\pm$ {:.2E} GHz '.format(hlp_mod.get_param(
                         f'{mobjn}.new_freq ' + fit_name + '_stderr',
                         data_dict)*1e-9))
                T2_star_str += \
                    ('\n$T_{{2,{{{key}}} }}^\star$ = '.format(
                        key=('exp' if i == 0 else 'gauss')) +
                     '{:.2f} $\mu$s'.format(
                         fit_res.params['tau'].value*1e6) +
                     ' $\pm$ {:.2f} $\mu$s'.format(
                         fit_res.params['tau'].stderr*1e6))

            fit_name = 'exp_decay'
            fit_res = data_dict['fit_dicts'][
                fit_name + keyi]['fit_res']
            old_qb_freq = hlp_mod.get_param(f'{mobjn}.old_freq', data_dict)
            textstr += '\n$f_{qubit \_ old}$ = '+'{:.6f} GHz '.format(
                old_qb_freq*1e-9)
            textstr += ('\n$\Delta f$ = {:.4f} MHz '.format(
                (hlp_mod.get_param(f'{mobjn}.new_freq ' + fit_name, data_dict) -
                 old_qb_freq)*1e-6) +
                        '$\pm$ {:.2E} MHz'.format(
                fit_res.params['frequency'].stderr*1e-6) +
                        '\n$f_{Ramsey}$ = '+
                        '{:.4f} MHz $\pm$ {:.2E} MHz'.format(
                        fit_res.params['frequency'].value*1e-6,
                        fit_res.params['frequency'].stderr*1e-6))
            textstr += T2_star_str
            textstr += '\nartificial detuning = {:.2f} MHz'.format(
                artificial_detuning_dict[mobjn]*1e-6)

            plot_dicts['text_msg_' + keyi] = {
                'fig_id': figure_name,
                'ypos': -0.3,
                'xpos': -0.125,
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'plotfn': 'plot_text',
                'text_string': textstr}

        plot_dicts['half_hline_' + keyi] = {
            'fig_id': figure_name,
            'plotfn': 'plot_hlines',
            'y': 0.5,
            'xmin': physical_swpts[0],
            'xmax': hlp_mod.get_cal_sweep_points(
                physical_swpts, cp, mobjn)[-1],
            'colors': 'gray'}

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

