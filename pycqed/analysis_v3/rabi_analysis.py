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
def rabi_iq_pipeline(meas_object_name):
    pp = pp_mod.ProcessingPipeline()
    pp.add_node('rotate_iq', keys_in='raw', meas_obj_names=meas_object_name)
    pp.add_node('rabi_analysis',
                keys_in=f'previous {meas_object_name}.rotate_iq',
                keys_out=None,
                meas_obj_names=meas_object_name)
    return pp


# run rabi analysis
def rabi_analysis(data_dict, keys_in, **params):
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)

    prep_fit_dicts = params.pop('prep_fit_dicts', True)
    do_fitting = params.pop('do_fitting', True)
    prepare_plotting = params.pop('prepare_plotting', True)
    do_plotting = params.pop('do_plotting', True)

    # prepare fitting
    if prep_fit_dicts:
        prepare_rabi_fitting(data_dict, keys_in, **params)

    if do_fitting:
        getattr(fit_mod, 'run_fitting')(data_dict, keys_in=list(
                data_dict['fit_dicts']),**params)
        # calculate new pi-pulse amplitude
        analyze_rabi_fit_results(data_dict, keys_in, **params)

    # prepare plots
    if prepare_plotting:
        prepare_plotting(data_dict, data_to_proc_dict, **params)
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict, keys_in=list(
            data_dict['plot_dicts']), **params)


def prepare_rabi_fitting(data_dict, keys_in, **params):
    fit_mod.prepare_cos_fit_dict(data_dict, keys_in=keys_in,
                                    fit_name='rabi_fit', **params)


def analyze_rabi_fit_results(data_dict, keys_in, **params):
    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    physical_swpts = sp[0][mospm[mobjn][0]][0]
    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    for keyi in keys_in:
        fit_res = fit_dicts['rabi_fit' + keyi]['fit_res']
        rabi_amps_dict = extract_rabi_amplitudes(fit_res=fit_res,
                                                 sweep_points=physical_swpts)
        for k, v in rabi_amps_dict.items():
            hlp_mod.add_param(f'{mobjn}.{k}', v, data_dict,
                              add_param_method='replace')


def prepare_rabi_plots(data_dict, data_to_proc_dict, **params):

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

    plot_dicts = OrderedDict()
    # the prepare plot dict functions below also iterate over data_to_proc_dict,
    # however the prepare functions add all the data corresponding to keys_in
    # to the same figure.
    # Here we want a figure for each keyi
    for keyi, data in data_to_proc_dict.items():
        figure_name = 'Rabi_' + keyi
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
            # plot fits
            fit_dicts = data_dict['fit_dicts']
            fit_name = 'rabi_fit' + keyi
            textstr = ''
            plot_mod.prepare_fit_plot_dicts(data_dict=data_dict,
                                               figure_name=figure_name,
                                               fit_names=[fit_name],
                                               do_plotting=False, **params)

            fit_res = fit_dicts[fit_name]['fit_res']
            piPulse_amp = hlp_mod.get_param(f'{mobjn}.piPulse_value', data_dict,
                                            default_value=0.0)
            # pi-pulse marker
            plot_dicts['piamp_marker_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_line',
                'xvals': np.array([piPulse_amp]),
                'yvals': np.array([fit_res.model.func(piPulse_amp,
                                                      **fit_res.best_values)]),
                'setlabel': '$\pi$ amp',
                'color': 'r',
                'marker': 'o',
                'line_kws': {'markersize': plot_mod.get_default_plot_params(
                    set_params=False).get('lines.markersize', 2) + 2},
                'linestyle': '',
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1.02, -0.2),
                'do_legend': True}

            # pi-pulse dashed line
            plot_dicts['piamp_hline_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_hlines',
                'y': np.array([fit_res.model.func(piPulse_amp,
                                                  **fit_res.best_values)]),
                'xmin': physical_swpts[0],
                'xmax': hlp_mod.get_cal_sweep_points(physical_swpts,
                                                     cp, mobjn)[-1],
                'colors': 'gray'}

            piHalfPulse_amp = hlp_mod.get_param(f'{mobjn}.piHalfPulse_value',
                                                data_dict, default_value=0.0)
            # piHalf-pulse marker
            plot_dicts['pihalfamp_marker_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_line',
                'xvals': np.array([piHalfPulse_amp]),
                'yvals': np.array([fit_res.model.func(piHalfPulse_amp,
                                                      **fit_res.best_values)]),
                'setlabel': '$\pi /2$ amp',
                'color': 'm',
                'marker': 'o',
                'line_kws': {'markersize': plot_mod.get_default_plot_params(
                    set_params=False).get('lines.markersize', 2) + 2},
                'linestyle': '',
                'do_legend': True,
                'legend_bbox_to_anchor': (1.02, -0.2),#'right',
                'legend_ncol': 2}

            # piHalf-pulse dashed line
            plot_dicts['pihalfamp_hline_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_hlines',
                'y': np.array([fit_res.model.func(piHalfPulse_amp,
                                                  **fit_res.best_values)]),
                'xmin': physical_swpts[0],
                'xmax': hlp_mod.get_cal_sweep_points(physical_swpts,
                                                     cp, mobjn)[-1],
                'colors': 'gray'}

            # plot textbox
            textstr, _, _ = get_rabi_textbox_properties(
                data_dict, textstr, transition='ef' if 'f' in keyi else 'ge',
                **params)
            plot_dicts['text_msg_' + keyi] = {
                'fig_id': figure_name,
                'ypos': -0.15,
                'xpos': -0.13,
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'plotfn': 'plot_text',
                'text_string': textstr}

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')


def get_rabi_textbox_properties(data_dict, textstr='',
                                transition='ge', **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)

    # Get from the hdf5 file any parameters specified in
    # params_dict and numeric_params.
    params_dict = {}
    s = 'Instrument settings.' + mobjn
    params_dict[f'{mobjn}.{transition}_amp180'] = \
        s + f'.{transition}_amp180'
    params_dict[f'{mobjn}.{transition}_amp90scale'] = \
        s + f'.{transition}_amp90_scale'
    hlp_mod.get_params_from_hdf_file(data_dict, params_dict=params_dict,
                                     numeric_params=list(params_dict), **params)

    # create textstring
    old_pipulse_val = hlp_mod.get_param(f'{mobjn}.{transition}_amp180',
                                        data_dict, default_value=0.0)
    # if old_pipulse_val != old_pipulse_val:
    #     old_pipulse_val = 0
    old_pihalfpulse_val = hlp_mod.get_param(f'{mobjn}.{transition}_amp90scale',
                                            data_dict, default_value=0.0)
    # if old_pihalfpulse_val != old_pihalfpulse_val:
    #     old_pihalfpulse_val = 0
    old_pihalfpulse_val *= old_pipulse_val
    textstr += ('  $\pi-Amp$:\n   {:.3f} V'.format(
        hlp_mod.get_param(f'{mobjn}.piPulse_value', data_dict)) +
               '$\pm$ {:.3f} V '.format(
        hlp_mod.get_param(f'{mobjn}.piPulse_stderr', data_dict)) +
               '\n$\pi/2-Amp$:\n   {:.3f} V '.format(
        hlp_mod.get_param(f'{mobjn}.piHalfPulse_value', data_dict)) +
               ' $\pm$ {:.3f} V '.format(
        hlp_mod.get_param(f'{mobjn}.piHalfPulse_stderr', data_dict)) +
               '\n  $\pi-Amp_{old}$ = ' + '{:.3f} V '.format(
        old_pipulse_val) +
               '\n$\pi/2-Amp_{old}$ = ' + '{:.3f} V '.format(
        old_pihalfpulse_val))

    hp = -0.135
    vp = -0.3
    return textstr, hp, vp


def extract_rabi_amplitudes(fit_res, sweep_points):
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
        n_piHalf_pulse = n[piHalfPulse_vals == piHalfPulse]
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
        piPulse_std = calculate_pulse_stderr(
            f=freq_fit,
            phi=phase_fit,
            f_err=freq_std,
            phi_err=phase_std,
            period_num=n_pi_pulse,
            cov=cov_freq_phase)
        piHalfPulse_std = calculate_pulse_stderr(
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

    rabi_amplitudes = {'piPulse_value': piPulse,
                       'piPulse_stderr': piPulse_std,
                       'piHalfPulse_value': piHalfPulse,
                       'piHalfPulse_stderr': piHalfPulse_std}

    return rabi_amplitudes


def calculate_pulse_stderr(f, phi, f_err, phi_err,
                           period_num, cov=0):
    x = period_num + phi
    return np.sqrt((f_err*x/(2*np.pi*(f**2)))**2 +
                   (phi_err/(2*np.pi*f))**2 -
                   2*(cov**2)*x/((2*np.pi*(f**3))**2))[0]