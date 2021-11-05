import logging
log = logging.getLogger(__name__)

import re
import numpy as np
import numbers
from inspect import signature
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import copy, deepcopy
from mpl_toolkits.mplot3d import Axes3D

from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.analysis.analysis_toolbox import get_color_order as gco
from pycqed.analysis.analysis_toolbox import get_color_list
from pycqed.analysis.tools.plotting import (
    set_axis_label, flex_colormesh_plot_vs_xy, flex_color_plot_vs_x,
    SI_prefix_and_scale_factor)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

import sys
this_module = sys.modules[__name__]
pp_mod.search_modules.add(this_module )

prx_single_column_width = 3.404
prx_two_column_width = 7.057
# Default to PRX style, change these global variables according to journal
FIGURE_WIDTH_1COL = prx_single_column_width
FIGURE_WIDTH_2COL = prx_two_column_width

############################################
### Helper functions related to Plotting ###
############################################
def default_figure_height(figure_width):
    return figure_width*2/(1 + np.sqrt(5))


def legend_unique_labels(ax, **kwargs):
    '''
    Generates a legend without duplicate labels.
    Args:
        ax: matplotlib axis.
        **kwargs: keyword arguments passed to ax.legend()

    Returns:

    '''
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    return ax.legend(*zip(*unique), **kwargs)


def get_default_plot_params(set_params=True, figure_width='1col',
                            figure_height=None, params=None, **kw):
    """
    Generates the rcParams that produce nice paper-style figures.
    Optionally updates the rcParams if set_pars == True.
    :param set_pars: whether to update the rcParams with the ones generated here
    :param figure_width: wight of the figure. Can be a float representing the
        width in inches, can be the string "2col" for a two column PRX style
        figure, or can be a string of the form "Xcol" with X a float <= 1
        representing a fraction of 1 column PRX style.
    :param figure_height: height of the figure in inches. If None, uses
        the function default_figure_height defined above.
    :param params: keyword arguments
    :return: rcParams dictionary
    """
    if hasattr(figure_width, '__iter__'):
        # assumes figure_width is of the form "Xcol" where X is either a
        # fraction of a column width, or X = 2.
        if figure_width == '2col':
            FIGURE_WIDTH = FIGURE_WIDTH_2COL
        else:
            X = float(figure_width[:-3])
            FIGURE_WIDTH = X*FIGURE_WIDTH_1COL
    else:
        FIGURE_WIDTH = figure_width

    FIGURE_HEIGHT = figure_height
    if FIGURE_HEIGHT is None:
        FIGURE_HEIGHT = default_figure_height(FIGURE_WIDTH)

    rc_params = {
        'font.size': 8,
        'lines.markersize': 2.0,
        'figure.titlesize': 'medium',
        'axes.titlesize': 'medium',
        'figure.dpi': 300,
        'figure.figsize': (FIGURE_WIDTH, FIGURE_HEIGHT),
        'axes.axisbelow': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 'small',
        'ytick.direction': 'in',
        'ytick.labelsize': 'small',
        'image.interpolation': 'none',
    }
    if params is None:
        params = {}
    rc_params.update(params)

    if set_params:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(rc_params)
    return plt.rcParams


def add_letter_to_subplots(fig, axes, xoffset=0.0, yoffset=0.0,
                           ha='left', va='top', labels=None):
    """
    Adds letters to top left corner of subplots corresponding to axes from fig.
    :param fig: figure object
    :param axes: subplots axes on fig
    :param xoffset: the x location of the letter is the left margin of the axis.
        This might not always work well. xoffset is added to this position.
    :param yoffset: the y location of the letter is the top margin of the axis.
        This might not always work well. yoffset is added to this position.
    :param ha: horizontal alignment of letters
    :param va: vertical alignment of letters
    :param labels: list of labels to be places next to each subplot
    :return: fig, axes

    Attention!
    Needs to be called after fig.subplots_adjust() or fig.tight_layout()

    """
    # ax_geom = get_axes_geometry_from_figure(fig)
    if labels is None:
        labels = [f'({chr(x+97)})'
                   for x in range(len(np.array(axes).flatten()))]
    for i, ax in enumerate(np.array(axes).flatten()):
        label = labels[i]
        ax.text(ax.bbox.transformed(fig.transFigure.inverted()).x0 + xoffset,
                ax.bbox.transformed(fig.transFigure.inverted()).y1 + yoffset,
                label,
                ha=ha, va=va, transform=fig.transFigure)
    return fig, axes


def get_axes_geometry_from_figure(fig):
    return fig.axes[0].get_subplotspec().get_topmost_subplotspec().\
        get_gridspec().get_geometry()


def default_figure_title(data_dict, meas_obj_name, **params):
    timestamps = hlp_mod.get_param('timestamps', data_dict, raise_error=True,
                                   **params)
    measurementstrings = hlp_mod.get_param('measurementstrings', data_dict,
                                           raise_error=True, **params)
    if len(timestamps) > 1:
        title = f'{timestamps[0]} - {timestamps[-1]}' \
                f'\n{measurementstrings[-1]}'
    else:
        title = f'{timestamps[-1]}\n{measurementstrings[-1]}'

    if len(meas_obj_name.split('_')) > 1:
        title += f'\n{meas_obj_name}'
    else:
        title += f' {meas_obj_name}'
    return title


def default_phase_cmap():
    cols = np.array(((41, 39, 231), (61, 130, 163), (208, 170, 39),
                     (209, 126, 4), (181, 28, 20), (238, 76, 152),
                     (251, 130, 242), (162, 112, 251))) / 255
    n = len(cols)
    cdict = {
        'red': [[i/n, cols[i%n][0], cols[i%n][0]] for i in range(n+1)],
        'green': [[i/n, cols[i%n][1], cols[i%n][1]] for i in range(n+1)],
        'blue': [[i/n, cols[i%n][2], cols[i%n][2]] for i in range(n+1)],
        'alpha': [[i/n, 1.0, 1.0] for i in range(n+1)],
    }

    cmap = mpl.colors.LinearSegmentedColormap('DMDefault', cdict)
    cmap.set_over((0,0,0,0))
    return cmap


## Prepare plot dicts functions ##
def prepare_cal_states_plot_dicts(data_dict, figure_name=None,
                                  keys_in=None, **params):
    """
    Prepares plot for cal_states and adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param figure_name: name of the figure on which all the plot dicts created
            will be plotted
    :param params:
        sp_name (str, default: 'none): name of the sweep parameter in
            sweep_points. To be used on x-axis.
        ylabel (str, default: None): y-axis label
        yunit (str, default: ''): y-axis unit
        do_legend (bool, default: True): whether to show the legend
        ncols (int, default: 2 if len(data_to_proc_dict) > 2 else 1):
            number of subplots along x
        nrows (int, default: 2 if len(data_to_proc_dict) == 2 else
            len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2):
            number of subplots along y
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved

    Assumptions:
        - if len(keys_in) > 1, this function will plot the data corresponding to
        each key_in on a separate subplot. To plot on same axis,
        set ncols=1, nrows=1.
        - cal_points, sweep_points, meas_obj_names exist in
        exp_metadata or params
        - expects 1d arrays
        - meas_obj_names is defined in cal_points
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    cp, sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'sp', 'mospm', 'mobjn'],
        **params)
    if len(cp.states) == 0:
        print(f'There are no cal_states to plot for {mobjn}.')
        return

    # get the sweep points information
    sp_name = params.get('sp_name', mospm[mobjn][0])
    sweep_info = [v for d in sp for k, v in d.items() if sp_name == k]
    if len(sweep_info) == 0:
        raise KeyError(f'{sp_name} not found.')
    physical_swpts = sweep_info[0][0]

    # get x-axis information
    xlabel = sweep_info[0][2]
    xunit = sweep_info[0][1]
    # xvals needs to be extracted in the for loop below

    # get y-axis information
    ylabel = params.get('ylabel', None)
    if ylabel is None:
        ylabel = r'$|f\rangle$ state population' if 'f,' in cp.states else \
                 r'$|e\rangle$ state population'
    yunit = params.get('yunit', '')

    # get more plotting aspect information
    data_labels = params.get('data_labels', ['Data']*len(data_to_proc_dict))
    if len(data_labels) != len(data_to_proc_dict):
        raise ValueError('Lenght of "data_labels" does not equal the number '
                         'of traces to plot')
    plotsize = get_default_plot_params(set=False)['figure.figsize']
    plotsize = (plotsize[0], 1.5*plotsize[1])
    ncols = hlp_mod.get_param(
        'ncols', params, default_value=2 if len(data_to_proc_dict) > 2 else 1)
    nrows = hlp_mod.get_param(
        'nrows', params, default_value=2 if len(data_to_proc_dict) == 2 else
        len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2)
    axids = np.arange(ncols*nrows)
    if len(axids) == 1:
        axids = [None]*len(data_to_proc_dict)

    # get figure name
    if figure_name is None:
        figure_name = 'cal_states'

    # start to iterate over data_to_proc_dict
    plot_dicts = OrderedDict()
    plot_names_cal = []
    for i, keyi in enumerate(data_to_proc_dict):
        data = data_to_proc_dict[keyi]
        if ylabel is None:
            ylabel = hlp_mod.get_latex_prob_label(keyi)

        cal_swpts = hlp_mod.get_cal_sweep_points(physical_swpts, cp, mobjn)
        cal_data = hlp_mod.get_cal_data(data, cp, mobjn)
        qb_cal_indxs = cp.get_indices()[mobjn]

        # plot cal points
        for ii, cal_pts_idxs in enumerate(qb_cal_indxs.values()):
            plot_dict_name_cal = list(qb_cal_indxs)[ii] + '_' + keyi + \
                                 hlp_mod.get_param('key_suffix', data_dict,
                                                   default_value='', **params)
            plot_dicts[plot_dict_name_cal+'_line'] = {
                'fig_id': figure_name,
                'ax_id': axids[i],
                'numplotsx': ncols,
                'numplotsy': nrows,
                'plotfn': 'plot_hlines',
                'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
                'y': np.mean(cal_data[cal_pts_idxs]),
                'xmin': physical_swpts[0],
                'xmax': cal_swpts[-1],
                'colors': 'gray'}
            plot_dicts[plot_dict_name_cal] = {
                'fig_id': figure_name,
                'ax_id': axids[i],
                'numplotsx': ncols,
                'numplotsy': nrows,
                'plotfn': 'plot_line',
                'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
                'xvals': cal_swpts[cal_pts_idxs],
                'xlabel': xlabel,
                'xunit': xunit,
                'yvals': cal_data[cal_pts_idxs],
                'ylabel': ylabel,
                'yunit': yunit,
                'setlabel': f'prep {list(qb_cal_indxs)[ii]}',
                'title': default_figure_title(data_dict, mobjn),
                'legend_bbox_to_anchor': (1, 0.5),
                'legend_pos': 'center left',
                'linestyle': 'none',
                'line_kws': {'color': hlp_mod.get_cal_state_color(
                    list(qb_cal_indxs)[ii])}}

            plot_names_cal += [plot_dict_name_cal, plot_dict_name_cal + '_line']

    # add plot_params to each plot dict
    plot_params = hlp_mod.get_param('plot_params', params, default_value={})
    for plt_name in plot_names_cal:
        plot_dicts[plt_name].update(plot_params)

    # add plot_dicts to the data_dict
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_plotting', False):
        # do plotting
        plot(data_dict, keys_in=plot_names_cal, **params)
    return plot_dicts


def prepare_1d_plot_dicts(data_dict, figure_name, keys_in, **params):
    """
    Prepares A SINGLE plot for 1d data arrays and adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param figure_name: name of the figure on which all the plot dicts created
            will be plotted
    :param params:
        sp_name (str, default: 'none): name of the sweep parameter in
            sweep_points. To be used on x-axis.
        xvals (array, default: None): array of points to be plotted on x-axis.
            If None, will be taken from sweep_points
        ylabel (str, default: None): y-axis label
        yunit (str, default: ''): y-axis unit
        data_labels (list of str, default: ['Data']): legend labels
            corresponding to the each of the keys_in
        ncols (int, default: 2 if len(data_to_proc_dict) > 2 else 1):
            number of subplots along x
        nrows (int, default: 2 if len(data_to_proc_dict) == 2 else
            len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2):
            number of subplots along y
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved

    Assumptions:
        - all the data corresponding to keys_in are plotted on the same figure!
        - automatically excludes cal points if len(cp.states) != 0.
        - if len(keys_in) > 1, this function will plot the data corresponding to
        each key_in on a separate subplot. To plot on same axis,
        set ncols=1, nrows=1.
        - cal_points, sweep_points, meas_obj_names exist in
        exp_metadata or params
        - expects 1d arrays
        - meas_obj_names is defined in cal_points
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in=keys_in)
    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    # we extract cp separately because we do not want to raise an error if cp
    # is not found; we do not strictly need them here
    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=False, **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)

    # get the sweep points information
    sp_name = params.get('sp_name', mospm[mobjn][0])
    # get x-axis information
    xvals = params.get('xvals', None)
    if xvals is None:
        xvals = deepcopy(sp.get_sweep_params_property('values', 'all', sp_name))
    xlabel = sp.get_sweep_params_property('label', 'all', sp_name)
    xunit = sp.get_sweep_params_property('unit', 'all', sp_name)
    xerr_key = params.get('xerr_key', '')
    xerr = hlp_mod.get_param(xerr_key, data_dict)  if len(xerr_key) else None

    # get y-axis information
    ylabel = params.get('ylabel', None)
    if ylabel is None and cp is not None:
        if len(cp.states) != 0:
            ylabel = r'$|f\rangle$ state population' if 'f,' in cp.states else \
                     r'$|e\rangle$ state population'
    yunit = params.get('yunit', '')
    yerr_key = params.get('yerr_key', '')
    yerr = hlp_mod.get_param(yerr_key, data_dict) if len(yerr_key) else None

    # get more plotting aspect information
    data_labels = params.get('data_labels', ['data']*len(data_to_proc_dict))
    if len(data_labels) != len(data_to_proc_dict):
        raise ValueError('Length of "data_labels" does not equal the number '
                         'of traces to plot')
    plotsize = get_default_plot_params(set=False)['figure.figsize']
    plotsize = (plotsize[0], 1.5*plotsize[1])
    ncols = hlp_mod.get_param(
        'ncols', params, default_value=2 if len(data_to_proc_dict) > 2 else 1)
    nrows = hlp_mod.get_param(
        'nrows', params, default_value=2 if len(data_to_proc_dict) == 2 else
        len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2)
    axids = np.arange(ncols*nrows)
    if len(axids) == 1:
        axids = [None]*len(data_to_proc_dict)

    # start to iterate over data_to_proc_dict
    plot_dicts = OrderedDict()
    plot_dict_names = []
    for i, keyi in enumerate(data_to_proc_dict):
        yvals = data_to_proc_dict[keyi]
        yvals = hlp_mod.get_msmt_data(yvals, cp, mobjn)
        if ylabel is None:
            ylabel = hlp_mod.get_latex_prob_label(keyi)
        smax = 40
        if len(ylabel) > smax:
            k = len(ylabel) // smax
            ylabel = '\n'.join([ylabel[i*smax:(i+1)*smax] for i in range(k)]
                               + [ylabel[-(len(ylabel) % smax):]])

        plot_dict_name = figure_name + keyi + hlp_mod.get_param('key_suffix',
                                                                data_dict,
                                                                default_value='',
                                                                **params)
        plot_dicts[plot_dict_name] = {
            'plotfn': 'plot_line',
            'fig_id': figure_name,
            'ax_id': axids[i],
            'numplotsx': ncols,
            'numplotsy': nrows,
            'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
            'xvals': xvals,
            'xlabel': xlabel,
            'xunit': xunit,
            'xerr': xerr,
            'yvals': yvals,
            'ylabel': ylabel,
            'yunit': yunit,
            'yerr': yerr,
            'setlabel': data_labels[i],
            'title': default_figure_title(data_dict, mobjn),
            'linestyle': params.get('linestyle', 'none'),
            'do_legend': True,
            'legend_bbox_to_anchor': (1, 0.5),
            'legend_pos': 'center left'}
        plot_dict_names += [plot_dict_name]

    # add plot_params to each plot dict
    plot_params = hlp_mod.get_param('plot_params', params, default_value={})
    for plt_name in plot_dict_names:
        plot_dicts[plt_name].update(plot_params)

    # add plot_dicts to the data_dict
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_plotting', False):
        # do plotting
        plot(data_dict, keys_in=plot_dict_names, **params)
    return plot_dicts


def prepare_2d_plot_dicts(data_dict, figure_name, keys_in, **params):
    """
    Prepares A SINGLE plot for 1d data arrays and adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param figure_name: name of the figure on which all the plot dicts created
            will be plotted
    :param params:
        sp_names (str, default: 'none): name of the sweep parameters in
            sweep_points. To be used on x- and y-axes.
        xvals (array, default: None): array of points to be plotted on x-axis
        zlabel (str, default: None): z-axis label
        zunit (str, default: ''): z-axis unit
        ncols (int, default: 2 if len(data_to_proc_dict) > 2 else 1):
            number of subplots along x
        nrows (int, default: 2 if len(data_to_proc_dict) == 2 else
            len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2):
            number of subplots along y
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved

    Assumptions:
        - all the data corresponding to keys_in are plotted on the same figure!
        - automatically excludes cal points if len(cp.states) != 0.
        - if len(keys_in) > 1, this function will plot the data corresponding to
        each key_in on a separate subplot. To plot on same axis,
        set ncols=1, nrows=1.
        - cal_points, sweep_points, meas_obj_names exist in
        exp_metadata or params
        - expects 1d arrays
        - meas_obj_names is defined in cal_points
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in=keys_in)
    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    # we extract cp separately because we do not want to raise an error if cp
    # is not found; we do not strictly need them here
    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=False, **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)

    # get the sweep points information
    sp_names = params.get('sp_names', mospm[mobjn])
    sweep_info = [v for d in sp for k, v in d.items() if k in sp_names]
    if len(sweep_info) == 0:
        raise KeyError(f'sp_names={sp_names} not found.')

    # get x-axis information
    xvals = params.get('xvals', None)
    if xvals is None:
        xvals = deepcopy(sweep_info[0][0])
    xlabel = sweep_info[0][2]
    xunit = sweep_info[0][1]

    # get Z-axis information
    zlabel = params.get('zlabel', None)
    if zlabel is None and cp is not None:
        if len(cp.states) != 0:
            zlabel = r'$|f\rangle$ state population' if 'f,' in cp.states else \
                     r'$|e\rangle$ state population'
    zunit = params.get('zunit', '')

    # get more plotting aspect information
    plotsize = get_default_plot_params(set=False)['figure.figsize']
    plotsize = (plotsize[0], 1.5*plotsize[1])
    ncols = hlp_mod.get_param(
        'ncols', params, default_value=2 if len(data_to_proc_dict) > 2 else 1)
    nrows = hlp_mod.get_param(
        'nrows', params, default_value=2 if len(data_to_proc_dict) == 2 else
        len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2)
    axids = np.arange(ncols*nrows)
    if len(axids) == 1:
        axids = [None]*len(data_to_proc_dict)

    # start to iterate over data_to_proc_dict
    plot_dicts = OrderedDict()
    plot_dict_names = []
    for i, keyi in enumerate(data_to_proc_dict):
        zvals = data_to_proc_dict[keyi]
        zvals = hlp_mod.get_msmt_data(zvals, cp, mobjn)
        if zlabel is None:
            zlabel = hlp_mod.get_latex_prob_label(keyi)
        zlabel = f'{zlabel} {zunit}'

        plot_dict_name = figure_name + keyi + hlp_mod.get_param('key_suffix',
                                                                data_dict,
                                                                default_value='',
                                                                **params)
        for sp_info in sweep_info:
            plot_dict_name += '_' + sp_info[2]
            plot_dicts[plot_dict_name] = {
                'plotfn': 'plot_colorxy',
                'fig_id': figure_name + '_' + sp_info[2],
                'ax_id': axids[i],
                'numplotsx': ncols,
                'numplotsy': nrows,
                'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
                'xvals': xvals,
                'yvals': sp_info[0],
                'zvals': zvals.T,
                'xlabel': xlabel,
                'xunit': xunit,
                'ylabel': sp_info[2],
                'yunit': sp_info[1],
                'title': default_figure_title(data_dict, mobjn),
                'clabel': zlabel}
            plot_dict_names += [plot_dict_name]

    # add plot_params to each plot dict
    plot_params = hlp_mod.get_param('plot_params', params, default_value={})
    for plt_name in plot_dict_names:
        plot_dicts[plt_name].update(plot_params)

    # add plot_dicts to the data_dict
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_plotting', False):
        # do plotting
        plot(data_dict, keys_in=plot_dict_names, **params)
    return plot_dicts


def prepare_1d_raw_data_plot_dicts(data_dict, keys_in=None, figure_name=None,
                                   **params):
    """
    Prepares A SINGLE plot for raw data and adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param figure_name: name of the figure on which all the plot dicts created
            will be plotted
    :param params:
        sp_name (str, default: 'none): name of the sweep parameter in
            sweep_points. To be used on x-axis.
        xvals (array, default: None): array of points to be plotted on x-axis.
            If None, will be taken from sweep_points
        yunit (str, default: taken from values_names if they exist else 'arb.'):
            y-axis unit
        data_labels (list of str, default: ['Data']): legend labels
            corresponding to the each of the keys_in
        ncols (int, default: 2 if len(data_to_proc_dict) > 2 else 1):
            number of subplots along x
        nrows (int, default: 2 if len(data_to_proc_dict) == 2 else
            len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2):
            number of subplots along y
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved
        figname_suffix (str, default: ''): suffix for the figure name name

    Assumptions:
        - all the data corresponding to keys_in are plotted on the same figure!
        - does NOT exclude cal points if len(cp.states) != 0; instead it extends
        the physical sweep points (if user does not provide xvals)
        - if len(keys_in) > 1, this function will plot the data corresponding to
        each key_in on a separate subplot. To plot on same axis,
        set ncols=1, nrows=1.
        - cal_points, sweep_points, meas_obj_name exist in
        exp_metadata or params
        - expects 1d arrays
        - meas_obj_name is defined in cal_points
    """
    sp, mospm, movnm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'mospm', 'movnm', 'mobjn'],
        **params)
    # we extract cp separately because we do not want to raise an error if cp
    # is not found; we do not strictly need them here
    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=False, **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)

    # get data_to_proc_dict
    if keys_in is None:
        keys_in = movnm[mobjn]
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in=keys_in)

    # get the sweep points information
    sp_name = params.get('sp_name', mospm[mobjn][0])
    # get x-axis information
    xvals = hlp_mod.get_param('xvals', params)
    xlabel = sp.get_sweep_params_property('label', 'all', sp_name)
    xunit = sp.get_sweep_params_property('unit', 'all', sp_name)

    # get more plotting aspect information
    data_labels = params.get('data_labels', ['data']*len(data_to_proc_dict))
    if len(data_labels) != len(data_to_proc_dict):
        raise ValueError('Lenght of "data_labels" does not equal the number '
                         'of traces to plot')
    plotsize = get_default_plot_params(set=False)['figure.figsize']
    ncols = hlp_mod.get_param(
        'ncols', params, default_value=2 if len(data_to_proc_dict) > 2 else 1)
    nrows = hlp_mod.get_param(
        'nrows', params, default_value=2 if len(data_to_proc_dict) == 2 else
        len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2)
    axids = np.arange(ncols*nrows)
    if len(axids) == 1:
        axids = [None]*len(data_to_proc_dict)

    # get figure name
    if figure_name is None:
        figure_name = 'raw_data'
    figure_name += '_' + hlp_mod.get_param(
        'figname_suffix', data_dict, default_value='', **params)

    data_transform_func = hlp_mod.get_param('data_transform_func',
                                            data_dict,
                                            default_value=lambda x: x,
                                            **params)

    # start to iterate over data_to_proc_dict
    plot_dicts = OrderedDict()
    plot_dict_names = []
    for i, keyi in enumerate(data_to_proc_dict):
        if xvals is None:
            physical_swpts = deepcopy(sp.get_sweep_params_property(
                'values', 'all', sp_name))
            cal_swpts = hlp_mod.get_cal_sweep_points(physical_swpts, cp, mobjn)
            xvals = np.concatenate([physical_swpts, cal_swpts])
        yvals = data_to_proc_dict[keyi]
        if yvals.ndim == 2:
            yvals = yvals.T
        yvals = data_transform_func(yvals)
        ylabel = hlp_mod.get_param('ylabel', data_dict, **params)
        if ylabel is None:
            ylabel = ','.join(hlp_mod.flatten_list([re.findall(ch, keyi)
                for ch in movnm[mobjn]]))
        yunit = hlp_mod.get_param('yunit', params,
                                  default_value=hlp_mod.get_param(
                                      'value_units', data_dict,
                                      default_value='arb.'))
        if isinstance(yunit, list):
            yunit = yunit[0]

        title = default_figure_title(data_dict, mobjn) + hlp_mod.get_param(
            'title_suffix', data_dict, default_value='', **params)
        plot_dict_name = figure_name + '_' + keyi + hlp_mod.get_param(
            'key_suffix', data_dict, default_value='', **params)
        plot_dicts[plot_dict_name] = {
            'plotfn': 'plot_line',
            'fig_id': figure_name,
            'ax_id': axids[i],
            'numplotsx': ncols,
            'numplotsy': nrows,
            'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
            'xvals': xvals,
            'xlabel': xlabel,
            'xunit': xunit,
            'yvals': yvals,
            'ylabel': ylabel,
            'yunit': yunit,
            'setlabel': data_labels[i],
            'title': title,
            'linestyle': params.get('linestyle', '-'),
            'color': params.get('color', None),
            'do_legend': False,
            'legend_bbox_to_anchor': (1, 0.5),
            'legend_pos': 'center left'}
        plot_dict_names += [plot_dict_name]

    # add plot_params to each plot dict
    plot_params = hlp_mod.get_param('plot_params', params, default_value={})
    for plt_name in plot_dict_names:
        plot_dicts[plt_name].update(plot_params)

    # add plot_dicts to the data_dict
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_plotting', False):
        # do plotting
        plot(data_dict, keys_in=plot_dict_names, **params)
    return plot_dicts


def prepare_2d_raw_data_plot_dicts(data_dict, keys_in=None, figure_name=None,
                                   **params):
    """
    Prepares A SINGLE plot for raw data and adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param figure_name: name of the figure on which all the plot dicts created
                        will be plotted
    :param params:
        sp_name (str, default: mospm[mobjn]): name of the sweep parameter in
            sweep_points. To be used on x-axis.
        xvals (numpy array or list, default: None): x values
        zunit (str, default: 'arb.'): z-axis unit
        do_plotting (bool, default: False): wheter to plot
        ncols (int, default: 2 if len(data_to_proc_dict) > 2 else 1):
            number of subplots along x
        nrows (int, default: 2 if len(data_to_proc_dict) == 2 else
            len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2):
            number of subplots along y
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved

    Assumptions:
        - all the data corresponding to keys_in are plotted on the same figure!
        - does NOT exclude cal points if len(cp.states) != 0; instead it extends
        the physical sweep points (if user does not provide xvals)
        - if len(keys_in) > 1, this function will plot the data corresponding to
        each key_in on a separate subplot. To plot on same axis,
        set ncols=1, nrows=1.
        - cal_points, sweep_points, meas_obj_name exist in
        exp_metadata or params
        - expects 1d arrays
        - meas_obj_name is defined in cal_points
    """
    sp, mospm, movnm, mobjn = hlp_mod.get_measurement_properties(
            data_dict, props_to_extract=['sp', 'mospm', 'movnm', 'mobjn'],
            **params)
    # we extract cp separately because we do not want to raise an error if cp
    # is not found; we do not strictly need them here
    cp = hlp_mod.get_param('cal_points', data_dict, raise_error=False, **params)
    if isinstance(cp, str):
        cp = CalibrationPoints.from_string(cp)

    # get data_to_proc_dict
    if keys_in is None:
        keys_in = movnm[mobjn]
    data_to_proc_dict = hlp_mod.get_data_to_process(
        data_dict, keys_in=keys_in)

    # get the sweep points information
    sp_names = params.get('sp_names', mospm[mobjn])
    sweep_info = [v for d in sp for k, v in d.items() if k in sp_names]
    if len(sweep_info) == 0:
        raise KeyError(f'sp_names={sp_names} not found.')

    # get x-axis information
    xvals = params.get('xvals', None)
    xlabel = sweep_info[0][2]
    xunit = sweep_info[0][1]

    # get more plotting aspect information
    plotsize = get_default_plot_params(set=False)['figure.figsize']
    ncols = hlp_mod.get_param(
        'ncols', params, default_value=2 if len(data_to_proc_dict) > 2 else 1)
    nrows = hlp_mod.get_param(
        'nrows', params, default_value=2 if len(data_to_proc_dict) == 2 else
        len(data_to_proc_dict) // 2 + len(data_to_proc_dict) % 2)
    axids = np.arange(ncols*nrows)
    if len(axids) == 1:
        axids = [None]*len(data_to_proc_dict)

    # get figure name
    if figure_name is None:
        figure_name = 'raw_data'

    # start to iterate over data_to_proc_dict
    plot_dicts = OrderedDict()
    plot_dict_names = []
    for i, keyi in enumerate(data_to_proc_dict):
        if xvals is None:
            physical_swpts = deepcopy(sweep_info[0][0])
            cal_swpts = hlp_mod.get_cal_sweep_points(physical_swpts, cp, mobjn)
            xvals = np.concatenate([physical_swpts, cal_swpts])
        zvals = data_to_proc_dict[keyi]
        zlabel = hlp_mod.get_param('ylabel', data_dict, **params)
        if zlabel is None:
            zlabel = ','.join(hlp_mod.flatten_list([re.findall(ch, keyi)
                                                    for ch in movnm[mobjn]]))
        zunit = hlp_mod.get_param('zunit', params,
                                  default_value=hlp_mod.get_param(
                                      'value_units', data_dict,
                                      default_value='arb.'))
        if isinstance(zunit, list):
            zunit = zunit[0]
        zlabel = f'{zlabel} {zunit}'

        plot_dict_name = figure_name + '_' + keyi + hlp_mod.get_param(
            'key_suffix', data_dict, default_value='', **params)
        for sp_info in sweep_info:
            plot_dict_name += '_' + sp_info[2]
            plot_dicts[plot_dict_name] = {
                'plotfn': 'plot_colorxy',
                'fig_id': figure_name + '_' + sp_info[2],
                'ax_id': axids[i],
                'numplotsx': ncols,
                'numplotsy': nrows,
                'plotsize': (plotsize[0]*ncols, plotsize[1]*nrows),
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,
                'yvals': sp_info[0],
                'ylabel': sp_info[2],
                'yunit': sp_info[1],
                'zvals': zvals.T,
                'title': default_figure_title(data_dict, mobjn),
                'clabel': zlabel}
            plot_dict_names += [plot_dict_name]

    # add plot_params to each plot dict
    plot_params = hlp_mod.get_param('plot_params', params, default_value={})
    for plt_name in plot_dict_names:
        plot_dicts[plt_name].update(plot_params)

    # add plot_dicts to the data_dict
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')

    if params.get('do_plotting', False):
        # do plotting
        plot(data_dict, keys_in=plot_dict_names, **params)
    return plot_dicts


def prepare_fit_plot_dicts(data_dict, figure_name, fit_names='all', **params):
    """
    Prepares plot dict for from a fit result object. Adds the plot dicts to the
    data_dict['plot_dicts'].
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param figure_name: name of the figure on which all the plot dicts created
            will be plotted
    :param fit_names: list of fit_names that exist in fit_dicts. Can also be
        'all' in which case it plots all the dictionaries in fit_dicts
    :param params:
        plot_params (dict, default: {}): dictionary of plot parameters and
            their values compatible with the plotting framework
        params_to_print (dict, default: None): list of parameter names that
            exist in the fit_res object and whose value should be displayed as
            textbox in the plot.
        key_suffix (str, default: ''): suffix for the key name under which
            the created plot dict is saved

    Assumptions:
        - fit_dicts exists in fit_dicts
        - meas_obj_names exists in either params, data_dict, or exp_metadata
    """
    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    plot_dicts = {}
    if fit_names == 'all':
        fit_names = list(fit_dicts)
    for fit_name in fit_names:
        fit_dict = hlp_mod.get_param(fit_name, fit_dicts, raise_error=True)
        dict_key = fit_name + hlp_mod.get_param(
            'key_suffix', data_dict, default_value='', **params)
        fit_res = fit_dict['fit_res']
        plot_params = hlp_mod.get_param('plot_params', fit_dict,
                                        default_value={})
        plot_params.update(hlp_mod.get_param('plot_params', params,
                                             default_value={}))
        plot_dicts[dict_key] = {
            'fig_id': figure_name,
            'plotfn': 'plot_fit',
            'fit_res': fit_res,
            'setlabel': 'fit',
            'zorder': 0,
            'color': 'r',
            'do_legend': True,
            'legend_ncol': 2,
            'legend_bbox_to_anchor': (1, -0.15),
            'legend_pos': 'upper right'}
        plot_dicts[dict_key].update(plot_params)

        pois = hlp_mod.get_param('params_to_print', fit_dict, **params)
        if pois is not None:
            textstr = ''
            for i, poi in enumerate(pois):
                params_val = fit_res.params[poi[0]].value
                params_stderr = fit_res.params[poi[0]].stderr
                scale_factor, unit = SI_prefix_and_scale_factor(
                    params_val, poi[1])
                textstr += '' if i == 0 else '\n'
                textstr += f'{poi[2]}={params_val*scale_factor:.3f}' \
                           f'$\\pm${params_stderr*scale_factor:.3f} {unit}'
            plot_dicts[dict_key + '_textstr'] = {
                'fig_id': figure_name,
                'ypos': -0.2,
                'xpos': -0.05,
                'box_props': None,
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'plotfn': 'plot_text',
                'text_string': textstr}

    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')
    if params.get('do_plotting', False):
        plot(data_dict, keys_in=list(plot_dicts), **params)
    return plot_dicts


## Plotting functions ##

def plot(data_dict, keys_in='all', axs_dict=None, **params):
    """
    Fits based on the information in proc_dat_dict[pipe_name]['fit_dicts']
    for each pipe_name, if 'fit_dicts' exists.
    Goes over the plots defined in the plot_dicts in
    proc_dat_dict[pipe_name]['fit_dicts'] for each pipe_name,
    if 'fit_dicts' exists, and creates the desired figures.
    """
    axs = OrderedDict()
    figs = OrderedDict()
    plot_dicts = data_dict['plot_dicts']
    no_label = params.get('no_label', False)
    if axs_dict is not None:
        for key, val in list(axs_dict.items()):
            axs[key] = val
    if keys_in is 'all':
        keys_in = plot_dicts.keys()
    if type(keys_in) is str:
        keys_in = [keys_in]

    for key in keys_in:
        # go over all the plot_dicts
        pdict = plot_dicts[key]
        pdict['no_label'] = no_label
        # Use the key of the plot_dict if no ax_id is specified
        pdict['fig_id'] = pdict.get('fig_id', key)
        pdict['ax_id'] = pdict.get('ax_id', None)

        if isinstance(pdict['ax_id'], str):
            pdict['fig_id'] = pdict['ax_id']
            pdict['ax_id'] = None

        if pdict['fig_id'] not in axs:
            # This fig variable should perhaps be a different
            # variable for each plot!!
            # This might fix a bug.
            figs[pdict['fig_id']], axs[pdict['fig_id']] = \
                plt.subplots(pdict.get('numplotsy', 1),
                             pdict.get('numplotsx', 1),
                             sharex=pdict.get('sharex', False),
                             sharey=pdict.get('sharey', False),
                             figsize=pdict.get('plotsize', None))
            if pdict.get('3d', False):
                axs[pdict['fig_id']].remove()
                axs[pdict['fig_id']] = Axes3D(
                    figs[pdict['fig_id']],
                    azim=pdict.get('3d_azim', -35),
                    elev=pdict.get('3d_elev', 35))
                axs[pdict['fig_id']].patch.set_alpha(0)

            # transparent background around axes for presenting data
            figs[pdict['fig_id']].patch.set_alpha(0)

    for fig_name in figs:
        figs[fig_name].tight_layout()

    for key in keys_in:
        pdict = plot_dicts[key]
        plot_touching = pdict.get('touching', False)

        if type(pdict['plotfn']) is str:
            plotfn = getattr(this_module, pdict['plotfn'])
        else:
            plotfn = pdict['plotfn']

        # used to ensure axes are touching
        if plot_touching:
            axs[pdict['fig_id']].figure.subplots_adjust(
                wspace=0, hspace=0)

        try:
            pdict['ax_geom'] = get_axes_geometry_from_figure(
                figs[pdict['fig_id']])
        except AttributeError:
            pass

        # Check if pdict is one of the accepted arguments,
        # these are the plotting functions in this module.
        if 'pdict' in signature(plotfn).parameters:
            if pdict['ax_id'] is None:
                plotfn(pdict=pdict, axs=axs[pdict['fig_id']])
            else:
                plotfn(pdict=pdict,
                       axs=axs[pdict['fig_id']].flatten()[
                           pdict['ax_id']])
                axs[pdict['fig_id']].flatten()[
                    pdict['ax_id']].figure.subplots_adjust(
                    hspace=0.4, wspace=0.25)

        # most normal plot functions also work, it is required
        # that these accept an "ax" argument to plot on and
        # **kwargs the pdict is passed in as kwargs to such
        # a function
        elif 'ax' in signature(plotfn).parameters:
            # Calling the function passing along anything
            # defined in the specific plot dict as kwargs
            if pdict['ax_id'] is None:
                plotfn(ax=axs[pdict['fig_id']], **pdict)
            else:
                plotfn(pdict=pdict,
                       axs=axs[pdict['fig_id']].flatten()[
                           pdict['ax_id']])
                axs[pdict['fig_id']].flatten()[
                    pdict['ax_id']].figure.subplots_adjust(
                    hspace=0.4, wspace=0.25)
        else:
            raise ValueError(
                f'"{plotfn}" is not a valid plot function')

    format_datetime_xaxes(data_dict, keys_in, axs)

    # add_letter_to_subplots
    for plot_name, axes in axs.items():
        if hasattr(axes, '__iter__'):
            # figs[plot_name].tight_layout()
            add_letter_to_subplots(figs[plot_name], axes.flatten(),
                                   xoffset=-0.07, yoffset=0.01, va='bottom')

    # close figures
    for fig_name in figs:
        try:
            figs[fig_name].align_ylabels()
        except AttributeError:
            pass
        plt.close(figs[fig_name])

    # add figures and axes to data_dict
    hlp_mod.add_param('figures', figs, data_dict, add_param_method='append')
    hlp_mod.add_param('axes', axs, data_dict, add_param_method='append')


def plot_vlines_auto(pdict, axs):
    xs = pdict.get('xdata')
    for i, x in enumerate(xs):
        d = {}
        for k in pdict:
            lk = k[:-1]
            if k not in ['xdata', 'plotfn', 'ax_id', 'do_legend']:
                try:
                    d[lk] = pdict[k][i]
                except:
                    pass
        axs.axvline(x=x, **d)


def plot_for_presentation(data_dict, key_list=None, **params):
    if key_list is None:
        key_list = list(data_dict['plot_dicts'].keys())
    for key in key_list:
        data_dict['plot_dicts'][key]['title'] = None
    plot(data_dict, key_list=key_list, **params)


def format_datetime_xaxes(data_dict, key_list, axs):
    for key in key_list:
        pdict = data_dict['plot_dicts'][key]
        # this check is needed as not all plots have xvals e.g., plot_text
        if 'xvals' in pdict.keys() and len(pdict['xvals']):
            if (type(pdict['xvals'][0]) is datetime.datetime and
                    key in axs.keys()):
                axs[key].figure.autofmt_xdate()

        
def plot_bar(pdict, axs, tight_fig=True):
    pfunc = getattr(axs, pdict.get('func', 'bar'))
    # xvals interpreted as edges for a bar plot
    plot_xedges = pdict.get('xvals', None)
    if plot_xedges is None:
        plot_centers = pdict['xcenters']
        plot_xwidth = pdict['xwidth']
    else:
        plot_xwidth = (plot_xedges[1:] - plot_xedges[:-1])
        # center is left edge + width/2
        plot_centers = plot_xedges[:-1] + plot_xwidth / 2
    plot_yvals = pdict['yvals']
    plot_xlabel = pdict.get('xlabel', None)
    plot_ylabel = pdict.get('ylabel', None)
    plot_xunit = pdict.get('xunit', None)
    plot_yunit = pdict.get('yunit', None)
    plot_xtick_loc = pdict.get('xtick_loc', None)
    plot_ytick_loc = pdict.get('ytick_loc', None)
    plot_xtick_rotation = pdict.get('xtick_rotation', None)
    plot_ytick_rotation = pdict.get('ytick_rotation', None)
    plot_xtick_labels = pdict.get('xtick_labels', None)
    plot_ytick_labels = pdict.get('ytick_labels', None)
    plot_title = pdict.get('title', None)
    plot_xrange = pdict.get('xrange', None)
    plot_yrange = pdict.get('yrange', None)
    plot_barkws = pdict.get('bar_kws', {})
    plot_multiple = pdict.get('multiple', False)
    dataset_desc = pdict.get('setdesc', '')
    dataset_label = pdict.get('setlabel', list(range(len(plot_yvals))))
    do_legend = pdict.get('do_legend', False)
    plot_touching = pdict.get('touching', False)

    if plot_multiple:
        p_out = []
        for ii, this_yvals in enumerate(plot_yvals):
            p_out.append(pfunc(plot_centers, this_yvals, width=plot_xwidth,
                               color=gco(ii, len(plot_yvals) - 1),
                               label='%s%s' % (dataset_desc, dataset_label[ii]),
                               **plot_barkws))

    else:
        p_out = pfunc(plot_centers, plot_yvals, width=plot_xwidth,
                      label='%s%s' % (dataset_desc, dataset_label),
                      **plot_barkws)

    if plot_xrange is not None:
        axs.set_xlim(*plot_xrange)
    if plot_yrange is not None:
        axs.set_ylim(*plot_yrange)
    if plot_xlabel is not None:
        set_axis_label('x', axs, plot_xlabel, plot_xunit)
    if plot_ylabel is not None:
        set_axis_label('y', axs, plot_ylabel, plot_yunit)
    if plot_xtick_labels is not None:
        axs.xaxis.set_ticklabels(plot_xtick_labels)
    if plot_ytick_labels is not None:
        axs.yaxis.set_ticklabels(plot_ytick_labels)
    if plot_xtick_loc is not None:
        axs.xaxis.set_ticks(plot_xtick_loc)
    if plot_ytick_loc is not None:
        axs.yaxis.set_ticks(plot_ytick_loc)
    if plot_xtick_rotation is not None:
        for tick in axs.get_xticklabels():
            tick.set_rotation(plot_xtick_rotation)
    if plot_ytick_rotation is not None:
        for tick in axs.get_yticklabels():
            tick.set_rotation(plot_ytick_rotation)

    if plot_title is not None:
        axs.set_title(plot_title)

    if do_legend:
        legend_kws = {'loc': pdict.get('legend_pos', 'best'),
                      'ncol': pdict.get('legend_ncol', 1),
                      'title': pdict.get('legend_title', None)}
        legend_kws.update(pdict.get('legend_kws', {}))

        legend_entries = pdict.get('legend_entries', [])
        legend_artists = [entry[0] for entry in legend_entries]
        legend_labels = [entry[1] for entry in legend_entries]
        handles, labels = axs.get_legend_handles_labels()
        handles += legend_artists
        labels += legend_labels
        axs.legend(handles, labels, **legend_kws)

    if plot_touching:
        axs.figure.subplots_adjust(wspace=0, hspace=0)

    if tight_fig:
        axs.figure.tight_layout()

    pdict['handles'] = p_out


def plot_bar3D(pdict, axs, tight_fig=True):
    pfunc = axs.bar3d
    plot_xvals = pdict['xvals']
    plot_yvals = pdict['yvals']
    plot_zvals = pdict['zvals']
    plot_xlabel = pdict.get('xlabel', None)
    plot_ylabel = pdict.get('ylabel', None)
    plot_zlabel = pdict.get('zlabel', None)
    plot_xunit = pdict.get('xunit', None)
    plot_yunit = pdict.get('yunit', None)
    plot_zunit = pdict.get('zunit', None)
    plot_color = pdict.get('color', None)
    plot_colormap = pdict.get('colormap', None)
    plot_title = pdict.get('title', None)
    plot_xrange = pdict.get('xrange', None)
    plot_yrange = pdict.get('yrange', None)
    plot_zrange = pdict.get('zrange', None)
    plot_barkws = pdict.get('bar_kws', {})
    plot_barwidthx = pdict.get('bar_widthx', None)
    plot_barwidthy = pdict.get('bar_widthy', None)
    plot_xtick_rotation = pdict.get('xtick_rotation', None)
    plot_ytick_rotation = pdict.get('ytick_rotation', None)
    plot_xtick_loc = pdict.get('xtick_loc', None)
    plot_ytick_loc = pdict.get('ytick_loc', None)
    plot_xtick_labels = pdict.get('xtick_labels', None)
    plot_ytick_labels = pdict.get('ytick_labels', None)
    do_legend = pdict.get('do_legend', False)
    set_edgecolor = pdict.get('set_edgecolor', False)

    xpos, ypos = np.meshgrid(plot_xvals, plot_yvals)
    xpos = xpos.T.flatten()
    ypos = ypos.T.flatten()
    xpos = np.concatenate(len(plot_zvals.flatten()) // len(xpos) * [xpos])
    ypos = np.concatenate(len(plot_zvals.flatten()) // len(ypos) * [ypos])
    zpos = np.zeros_like(xpos)
    if plot_barwidthx is None:
        plot_barwidthx = plot_xvals[1] - plot_xvals[0]
    if not hasattr(plot_barwidthx, '__iter__'):
        plot_barwidthx = np.ones_like(plot_zvals.flatten())*plot_barwidthx
    if plot_barwidthy is None:
        plot_barwidthy = plot_yvals[1] - plot_yvals[0]
    if not hasattr(plot_barwidthy, '__iter__'):
        plot_barwidthy = np.ones_like(plot_zvals.flatten()) * plot_barwidthy
    plot_barheight = plot_zvals.flatten()

    if 'color' in plot_barkws:
        plot_color = plot_barkws.pop('color')
    else:
        if plot_colormap is not None:
            # plot_color assumed to be floats
            if hasattr(plot_color, '__iter__') and \
                    hasattr(plot_color[0], '__iter__'):
                plot_color = np.array(plot_color).flatten()
            plot_color = plot_colormap(plot_color)
        else:
            # plot_color assumed to be RGBA tuple(s)
            if hasattr(plot_color[0], '__iter__') and \
                    hasattr(plot_color[0][0], '__iter__'):
                plot_color = np.array(plot_color)
                plot_color = plot_color.reshape((-1, plot_color.shape[-1]))
            elif not hasattr(plot_color[0], '__iter__'):
                plot_color = np.array(plot_color)
                n = plot_zvals.size
                plot_color = np.repeat(plot_color, n).reshape(-1, n).T

    zsort = plot_barkws.pop('zsort', 'max')
    edgecolor = None
    if set_edgecolor:
        edgecolor = len(xpos)*[(0,0,0,1)] + len(xpos)*[(0,0,0,1)]
    p_out = pfunc(xpos - plot_barwidthx/2, ypos - plot_barwidthy/2, zpos,
                  plot_barwidthx, plot_barwidthy, plot_barheight,
                  color=plot_color, edgecolor=edgecolor,
                  zsort=zsort, **plot_barkws)

    if plot_xtick_labels is not None:
        axs.xaxis.set_ticklabels(plot_xtick_labels)
    if plot_ytick_labels is not None:
        axs.yaxis.set_ticklabels(plot_ytick_labels)
    if plot_xtick_loc is not None:
        axs.xaxis.set_ticks(plot_xtick_loc)
    if plot_ytick_loc is not None:
        axs.yaxis.set_ticks(plot_ytick_loc)
    if plot_xtick_rotation is not None:
        for tick in axs.get_xticklabels():
            tick.set_rotation(plot_xtick_rotation)
    if plot_ytick_rotation is not None:
        for tick in axs.get_yticklabels():
            tick.set_rotation(plot_ytick_rotation)

    if plot_xrange is not None:
        axs.set_xlim(*plot_xrange)
    if plot_yrange is not None:
        axs.set_ylim(*plot_yrange)
    if plot_zrange is not None:
        axs.set_zlim3d(*plot_zrange)
    if plot_xlabel is not None:
        set_axis_label('x', axs, plot_xlabel, plot_xunit)
    if plot_ylabel is not None:
        set_axis_label('y', axs, plot_ylabel, plot_yunit)
    if plot_zlabel is not None:
        set_axis_label('z', axs, plot_zlabel, plot_zunit)
    if plot_title is not None:
        axs.set_title(plot_title)

    if do_legend:
        legend_kws = pdict.get('legend_kws', {})
        legend_entries = pdict.get('legend_entries', [])
        legend_artists = [entry[0] for entry in legend_entries]
        legend_labels = [entry[1] for entry in legend_entries]
        axs.legend(legend_artists, legend_labels, **legend_kws)

    if tight_fig:
        axs.figure.tight_layout()

    if pdict.get('colorbar', True) and plot_colormap is not None:
        plot_colorbar(axs=axs, pdict=pdict)

    pdict['handles'] = p_out


def plot_line(pdict, axs, tight_fig=True):
    """
    Basic line plotting function.
    Takes either an x and y array or a list of x and y arrays.
    Detection happens based on types of the data
    """

    # if a y or xerr is specified, used the errorbar-function
    plot_linekws = pdict.get('line_kws', {})
    xerr = pdict.get('xerr', None)
    yerr = pdict.get('yerr', None)
    plot_errorbars = xerr is not None or yerr is not None
    if plot_errorbars:
        pdict['func'] = pdict.get('func', 'errorbar')
        if yerr is not None:
            plot_linekws['yerr'] = plot_linekws.get('yerr', yerr)
        if xerr is not None:
            plot_linekws['xerr'] = plot_linekws.get('xerr', xerr)

    pdict['line_kws'] = plot_linekws
    plot_xvals = pdict['xvals']
    plot_yvals = pdict['yvals']
    plot_xlabel = pdict.get('xlabel', None)
    plot_ylabel = pdict.get('ylabel', None)
    plot_xunit = pdict.get('xunit', None)
    plot_yunit = pdict.get('yunit', None)
    plot_title = pdict.get('title', None)
    plot_xrange = pdict.get('xrange', None)
    plot_yrange = pdict.get('yrange', None)
    zorder = pdict.get('zorder', None)
    ax_id = pdict.get('ax_id', None)
    ax_geom = pdict.get('ax_geom', (1, 1))

    if pdict.get('color', False):
        plot_linekws['color'] = pdict.get('color')

    # plot_multiple = pdict.get('multiple', False)
    plot_linestyle = pdict.get('linestyle', '-')
    plot_marker = pdict.get('marker', 'o')
    dataset_desc = pdict.get('setdesc', '')
    if np.ndim(plot_yvals) == 2:
        default_labels = list(range(len(plot_yvals)))
    elif np.ndim(plot_yvals) == 1:
        default_labels = [0]
    else:
        raise ValueError("number of plot_yvals not understood")
    dataset_label = pdict.get('setlabel', default_labels)
    do_legend = pdict.get('do_legend', False)

    plot_multiple = False
    if len(plot_xvals):
        # Detect if two arrays/lists of x and yvals are passed or a list
        # of x-arrays and a list of y-arrays
        if (isinstance(plot_xvals[0], numbers.Number) or
                isinstance(plot_xvals[0], datetime.datetime)):
            plot_multiple = False
        else:
            plot_multiple = True
            assert (len(plot_xvals) == len(plot_yvals))
            assert (len(plot_xvals[0]) == len(plot_yvals[0]))

    alpha_errorbars = plot_linekws.pop('alpha_errorbars', 1)
    if plot_multiple:
        p_out = []
        len_color_cycle = pdict.get('len_color_cycle', len(plot_yvals))
        # Default gives max contrast
        cmap = pdict.get('cmap', 'tab10')  # Default matplotlib cycle
        colors = get_color_list(len_color_cycle, cmap)
        if cmap == 'tab10':
            len_color_cycle = min(10, len_color_cycle)

        # plot_*vals is the list of *vals arrays
        pfunc = getattr(axs, pdict.get('func', 'plot'))
        for i, (xvals, yvals) in enumerate(zip(plot_xvals, plot_yvals)):
            p_out.append(pfunc(xvals, yvals,
                               linestyle=plot_linestyle,
                               marker=plot_marker,
                               color=plot_linekws.pop(
                                   'color', colors[i % len_color_cycle]),
                               label='%s%s' % (
                                   dataset_desc, dataset_label[i]),
                               zorder=zorder,
                               **plot_linekws))

    else:
        pfunc = getattr(axs, pdict.get('func', 'plot'))
        p_out = pfunc(plot_xvals, plot_yvals, zorder=zorder,
                                 linestyle=plot_linestyle, marker=plot_marker,
                                 label='%s%s' % (dataset_desc, dataset_label),
                                 **plot_linekws)
    plot_linekws['alpha_errorbars'] = alpha_errorbars

    if plot_errorbars:
        # loop through bars and caps and set the alpha value
        [bar.set_alpha(alpha_errorbars) for bar in p_out[2]]
        [cap.set_alpha(alpha_errorbars) for cap in p_out[1]]

    if plot_xrange is None:
        pass  # Do not set xlim if xrange is None as the axs gets reused
    else:
        xmin, xmax = plot_xrange
        axs.set_xlim(xmin, xmax)

    if plot_title is not None:
        if ax_geom[1] > 1:
            axs.figure.text(0.5, 1.02, plot_title,
                            horizontalalignment='center',
                            verticalalignment='bottom')
        elif ax_id in [0, None]:
            axs.set_title(plot_title)

    if do_legend:
        legend_ncol = pdict.get('legend_ncol', 1)
        legend_title = pdict.get('legend_title', None)
        legend_pos = pdict.get('legend_pos', 'best')
        legend_frameon = pdict.get('legend_frameon', False)
        legend_labelspacing = pdict.get('legend_labelspacing', None)
        legend_columnspacing = pdict.get('legend_columnspacing', None)
        legend_bbox_to_anchor = pdict.get('legend_bbox_to_anchor', None)
        legend_bbox_transform = pdict.get('legend_bbox_transform',
                                          axs.transAxes)

        axs.legend(title=legend_title,
                   loc=legend_pos,
                   ncol=legend_ncol,
                   bbox_to_anchor=legend_bbox_to_anchor,
                   bbox_transform=legend_bbox_transform,
                   labelspacing=legend_labelspacing,
                   columnspacing=legend_columnspacing ,
        frameon=legend_frameon)

    if plot_xlabel is not None:
        set_axis_label('x', axs, plot_xlabel, plot_xunit)
    if plot_ylabel is not None:
        set_axis_label('y', axs, plot_ylabel, plot_yunit)
    if plot_yrange is not None:
        ymin, ymax = plot_yrange
        axs.set_ylim(ymin, ymax)

    if tight_fig:
        axs.figure.tight_layout()

        # Need to set labels again, because tight_layout can screw them up
        if plot_xlabel is not None:
            set_axis_label('x', axs, plot_xlabel, plot_xunit)
        if plot_ylabel is not None:
            set_axis_label('y', axs, plot_ylabel, plot_yunit)

    pdict['handles'] = p_out


def plot_yslices(pdict, axs, tight_fig=True):
    pfunc = getattr(axs, pdict.get('func', 'plot'))
    plot_xvals = pdict['xvals']
    plot_yvals = pdict['yvals']
    plot_slicevals = pdict['slicevals']
    plot_xlabel = pdict['xlabel']
    plot_ylabel = pdict['ylabel']
    plot_nolabel = pdict.get('no_label', False)
    plot_title = pdict['title']
    slice_idxs = pdict['sliceidxs']
    slice_label = pdict.get('slicelabel', '')
    slice_units = pdict.get('sliceunits', '')
    do_legend = pdict.get('do_legend', True)
    plot_xrange = pdict.get('xrange', None)
    plot_yrange = pdict.get('yrange', None)

    plot_xvals_step = plot_xvals[1] - plot_xvals[0]

    for ii, idx in enumerate(slice_idxs):
        if len(slice_idxs) == 1:
            pfunc(plot_xvals, plot_yvals[idx], '-bo',
                  label='%s = %.2f %s' % (
                      slice_label, plot_slicevals[idx], slice_units))
        else:
            if ii == 0 or ii == len(slice_idxs) - 1:
                pfunc(plot_xvals, plot_yvals[idx], '-o',
                      color=gco(ii, len(slice_idxs) - 1),
                      label='%s = %.2f %s' % (
                          slice_label, plot_slicevals[idx], slice_units))
            else:
                pfunc(plot_xvals, plot_yvals[idx], '-o',
                      color=gco(ii, len(slice_idxs) - 1))
    if plot_xrange is None:
        xmin, xmax = np.min(plot_xvals) - plot_xvals_step / \
                     2., np.max(plot_xvals) + plot_xvals_step / 2.
    else:
        xmin, xmax = plot_xrange
    axs.set_xlim(xmin, xmax)

    if not plot_nolabel:
        axs.set_axis_label('x', plot_xlabel)
        axs.set_axis_label('y', plot_ylabel)

    if plot_yrange is not None:
        ymin, ymax = plot_yrange
        axs.set_ylim(ymin, ymax)

    if plot_title is not None:
        axs.set_title(plot_title)

    if do_legend:
        legend_ncol = pdict.get('legend_ncol', 1)
        legend_title = pdict.get('legend_title', None)
        legend_pos = pdict.get('legend_pos', 'best')
        axs.legend(title=legend_title, loc=legend_pos, ncol=legend_ncol)
        legend_pos = pdict.get('legend_pos', 'best')
        # box_props = dict(boxstyle='Square', facecolor='white', alpha=0.6)
        legend = axs.legend(loc=legend_pos, frameon=1)
        frame = legend.get_frame()
        frame.set_alpha(0.8)
        frame.set_linewidth(0)
        frame.set_edgecolor(None)
        legend_framecol = pdict.get('legend_framecol', 'white')
        frame.set_facecolor(legend_framecol)

    if tight_fig:
        axs.figure.tight_layout()


def plot_colorxy(pdict, axs):
    """
    This wraps flex_colormesh_plot_vs_xy which excepts data of shape
        x -> 1D array
        y -> 1D array
        z -> 2D array (shaped (xl, yl))
    """
    plot_color2D(flex_colormesh_plot_vs_xy, pdict, axs)


def plot_colorx(pdict, axs):
    """
    This wraps flex_color_plot_vs_x which excepts data of shape
        x -> 1D array
        y -> list "xl" 1D arrays
        z -> list "xl" 1D arrays
    """

    plot_color2D(flex_color_plot_vs_x, pdict, axs)


def plot_color2D_grid_idx(pfunc, pdict, axs, idx):
    pfunc(pdict, np.ravel(axs)[idx])


def plot_color2D_grid(pdict, axs):
    color2D_pfunc = pdict.get('pfunc', plot_colorxy)
    num_elements = len(pdict['zvals'])
    num_axs = axs.size
    if num_axs > num_elements:
        max_plot = num_elements
    else:
        max_plot = num_axs
    plot_idxs = pdict.get('plot_idxs', None)
    if plot_idxs is None:
        plot_idxs = list(range(max_plot))
    else:
        plot_idxs = plot_idxs[:max_plot]
    this_pdict = {key: val for key, val in list(pdict.items())}
    if pdict.get('sharex', False):
        this_pdict['xlabel'] = ''
    if pdict.get('sharey', False):
        this_pdict['ylabel'] = ''

    box_props = dict(boxstyle='Square', facecolor='white', alpha=0.7)
    plot_axlabels = pdict.get('axlabels', None)

    for ii, idx in enumerate(plot_idxs):
        this_pdict['zvals'] = np.squeeze(pdict['zvals'][idx])
        if ii != 0:
            this_pdict['title'] = None
        else:
            this_pdict['title'] = pdict['title']
        plot_color2D_grid_idx(color2D_pfunc, this_pdict, axs, ii)
        if plot_axlabels is not None:
            np.ravel(axs)[idx].text(
                0.95, 0.9, plot_axlabels[idx],
                transform=np.ravel(axs)[idx].transAxes, fontsize=16,
                verticalalignment='center', horizontalalignment='right',
                bbox=box_props)
    if pdict.get('sharex', False):
        for ax in axs[-1]:
            ax.set_axis_label('x', pdict['xlabel'])
    if pdict.get('sharey', False):
        for ax in axs:
            ax[0].set_axis_label('y', pdict['ylabel'])


def plot_color2D(pfunc, pdict, axs, verbose=False, do_individual_traces=False):
    """

    """
    plot_xvals = pdict['xvals']
    plot_yvals = pdict['yvals']
    plot_cbar = pdict.get('plotcbar', True)
    plot_cmap = pdict.get('cmap', 'viridis')
    plot_aspect = pdict.get('aspect', None)
    plot_zrange = pdict.get('zrange', None)
    plot_yrange = pdict.get('yrange', None)
    plot_xrange = pdict.get('xrange', None)
    plot_xwidth = pdict.get('xwidth', None)
    plot_xtick_labels = pdict.get('xtick_labels', None)
    plot_ytick_labels = pdict.get('ytick_labels', None)
    plot_xtick_loc = pdict.get('xtick_loc', None)
    plot_ytick_loc = pdict.get('ytick_loc', None)
    plot_transpose = pdict.get('transpose', False)
    plot_nolabel = pdict.get('no_label', False)
    plot_normalize = pdict.get('normalize', False)
    plot_logzscale = pdict.get('logzscale', False)
    plot_origin = pdict.get('origin', 'lower')

    if plot_logzscale:
        plot_zvals = np.log10(pdict['zvals'] / plot_logzscale)
    else:
        plot_zvals = pdict['zvals']

    if plot_xwidth is not None:
        plot_xvals_step = 0
        plot_yvals_step = 0
    else:
        plot_xvals_step = (abs(np.max(plot_xvals) - np.min(plot_xvals)) /
                           len(plot_xvals))
        plot_yvals_step = (abs(_globalmax(plot_yvals) -
                               _globalmin(plot_yvals)) /
                           len(plot_yvals))
        # plot_yvals_step = plot_yvals[1]-plot_yvals[0]

    if plot_zrange is not None:
        fig_clim = plot_zrange
    else:
        fig_clim = [None, None]

    trace = {}
    block = {}
    if do_individual_traces:
        trace['xvals'] = plot_xvals
        trace['yvals'] = plot_yvals
        trace['zvals'] = plot_zvals
    else:
        trace['yvals'] = [plot_yvals]
        trace['xvals'] = [plot_xvals]
        trace['zvals'] = [plot_zvals]

    block['xvals'] = [trace['xvals']]
    block['yvals'] = [trace['yvals']]
    block['zvals'] = [trace['zvals']]

    for ii in range(len(block['zvals'])):
        traces = {}
        for key, vals in block.items():
            traces[key] = vals[ii]
        for tt in range(len(traces['zvals'])):
            if verbose:
                (print(t_vals[tt].shape) for key, t_vals in traces.items())
            if plot_xwidth is not None:
                xwidth = plot_xwidth[tt]
            else:
                xwidth = None
            out = pfunc(ax=axs,
                        xwidth=xwidth,
                        clim=fig_clim, cmap=plot_cmap,
                        xvals=traces['xvals'][tt],
                        yvals=traces['yvals'][tt],
                        zvals=traces['zvals'][tt],
                        transpose=plot_transpose,
                        normalize=plot_normalize)

    if plot_xrange is None:
        if plot_xwidth is not None:
            xmin, xmax = min([min(xvals) - plot_xwidth[tt] / 2
                              for tt, xvals in enumerate(plot_xvals)]), \
                         max([max(xvals) + plot_xwidth[tt] / 2
                              for tt, xvals in enumerate(plot_xvals)])
        else:
            xmin = np.min(plot_xvals) - plot_xvals_step / 2
            xmax = np.max(plot_xvals) + plot_xvals_step / 2
    else:
        xmin, xmax = plot_xrange
    if plot_transpose:
        axs.set_ylim(xmin, xmax)
    else:
        axs.set_xlim(xmin, xmax)

    if plot_yrange is None:
        if plot_xwidth is not None:
            ymin_list, ymax_list = [], []
            for ytraces in block['yvals']:
                ymin_trace, ymax_trace = [], []
                for yvals in ytraces:
                    ymin_trace.append(min(yvals))
                    ymax_trace.append(max(yvals))
                ymin_list.append(min(ymin_trace))
                ymax_list.append(max(ymax_trace))
            ymin = min(ymin_list)
            ymax = max(ymax_list)
        else:
            ymin = _globalmin(plot_yvals) - plot_yvals_step / 2.
            ymax = _globalmax(plot_yvals) + plot_yvals_step / 2.
    else:
        ymin, ymax = plot_yrange
    if plot_transpose:
        axs.set_xlim(ymin, ymax)
    else:
        axs.set_ylim(ymin, ymax)

    # FIXME Ignores thranspose option. Is it ok?
    if plot_xtick_labels is not None:
        axs.xaxis.set_ticklabels(plot_xtick_labels, rotation=90)
    if plot_ytick_labels is not None:
        axs.yaxis.set_ticklabels(plot_ytick_labels)
    if plot_xtick_loc is not None:
        axs.xaxis.set_ticks(plot_xtick_loc)
    if plot_ytick_loc is not None:
        axs.yaxis.set_ticks(plot_ytick_loc)
    if plot_origin == 'upper':
        axs.invert_yaxis()

    if plot_aspect is not None:
        axs.set_aspect(plot_aspect)

    if not plot_nolabel:
        label_color2D(pdict, axs)

    axs.cmap = out['cmap']
    if plot_cbar:
        plot_colorbar(axs=axs, pdict=pdict)


def label_color2D(pdict, axs):
    plot_transpose = pdict.get('transpose', False)
    plot_xlabel = pdict['xlabel']
    plot_xunit = pdict['xunit']
    plot_ylabel = pdict['ylabel']
    plot_yunit = pdict['yunit']
    plot_title = pdict.get('title', None)
    if plot_transpose:
        # transpose switches X and Y
        set_axis_label('x', axs, plot_ylabel, plot_yunit)
        set_axis_label('y', axs, plot_xlabel, plot_xunit)
    else:
        set_axis_label('x', axs, plot_xlabel, plot_xunit)
        set_axis_label('y', axs, plot_ylabel, plot_yunit)
    if plot_title is not None:
        axs.figure.text(0.5, 1, plot_title,
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        transform=axs.transAxes)
        # axs.set_title(plot_title)


def plot_colorbar(pdict=None, axs=None, cax=None,
                  orientation='vertical', tight_fig=True):
    if pdict is None or axs is None:
        raise ValueError('pdict and axs must be specified'
                         ' when no key is specified.')
    plot_nolabel = pdict.get('no_label', False)
    plot_clabel = pdict.get('clabel', None)
    plot_cbarwidth = pdict.get('cbarwidth', '10%')
    plot_cbarpad = pdict.get('cbarpad', '5%')
    plot_ctick_loc = pdict.get('ctick_loc', None)
    plot_ctick_labels = pdict.get('ctick_labels', None)
    if cax is None:
        if not isinstance(axs, Axes3D):
            axs.ax_divider = make_axes_locatable(axs)
            axs.cax = axs.ax_divider.append_axes(
                'right', size=plot_cbarwidth, pad=plot_cbarpad)
            cmap = axs.cmap
        else:
            plot_cbarwidth = str_to_float(plot_cbarwidth)
            plot_cbarpad = str_to_float(plot_cbarpad)
            axs.cax, _ = mpl.colorbar.make_axes(
                axs, shrink=1-plot_cbarwidth-plot_cbarpad, pad=plot_cbarpad,
                orientation=orientation)
            cmap = pdict.get('colormap')
    else:
        axs.cax = cax
    if hasattr(cmap, 'autoscale_None'):
        axs.cbar = plt.colorbar(cmap, cax=axs.cax, orientation=orientation)
    else:
        norm = mpl.colors.Normalize(0, 1)
        axs.cbar = mpl.colorbar.ColorbarBase(axs.cax, cmap=cmap, norm=norm)
    if plot_ctick_loc is not None:
        axs.cbar.set_ticks(plot_ctick_loc)
    if plot_ctick_labels is not None:
        axs.cbar.set_ticklabels(plot_ctick_labels)
    if not plot_nolabel and plot_clabel is not None:
        axs.cbar.set_label(plot_clabel)

    if tight_fig:
        axs.figure.tight_layout()


def plot_fit(pdict, axs):
    """
    Plots an lmfit fit result object using the plot_line function.
    """
    xvals, yvals = 'xvals', 'yvals'
    if pdict.get('swap_xy', False):
        xvals, yvals = yvals, xvals
    model = pdict['fit_res'].model
    plot_data = pdict.get('plot_data', False)  # plot the data
    plot_data_params = pdict.get('plot_data_params', {})
    plot_init = pdict.get('plot_init', False)  # plot the initial guess
    plot_init_params = pdict.get('plot_init_params', {})
    pdict['marker'] = pdict.get('marker', '')  # different default
    plot_numpoints = pdict.get('num_points', 1000)

    if len(model.independent_vars) == 1:
        independent_var = model.independent_vars[0]
    else:
        raise ValueError('Fit can only be plotted if the model function'
                         ' has one independent variable.')

    x_arr = pdict['fit_res'].userkws[independent_var]
    pdict[xvals] = np.linspace(np.min(x_arr), np.max(x_arr),
                                 plot_numpoints)
    pdict[yvals] = model.eval(pdict['fit_res'].params,
                                **{independent_var: pdict[xvals]})
    if not hasattr(pdict[yvals], '__iter__'):
        pdict[yvals] = np.array([pdict[yvals]])
    if isinstance(pdict[yvals], list) or isinstance(pdict[yvals], tuple):
        pdict[xvals] = len(pdict[yvals])*[pdict[xvals]]
    if 'zorder' not in pdict:
        pdict['zorder'] = 0
    plot_line(pdict, axs)
    if 'setlabel' not in pdict:
        pdict['setlabel'] = ''

    if plot_data:
        # The data
        pdict_data = copy(pdict)
        pdict_data.update(plot_data_params)
        if 'linestyle' not in pdict_data:
            pdict_data['linestyle'] = ''
        if pdict_data['marker'] == '':
            pdict_data['marker'] = 'o'
        fit_color = pdict.get('color', '')
        data_color = pdict_data.get('color', '')
        pdict_data['color'] = data_color
        if pdict_data['zorder'] == 0:
            pdict_data['zorder'] = 1
        pdict_data[xvals] = pdict['fit_res'].userkws[independent_var]
        pdict_data[yvals] = pdict['fit_res'].userargs[0]
        if pdict_data['setlabel'] == pdict['setlabel']:
            pdict_data['setlabel'] += ' data'
        plot_line(pdict_data, axs)

    if plot_init:
        # The initial guess
        pdict_init = copy(pdict)
        pdict_init.update(plot_init_params)
        if 'linestyle' not in pdict_init:
            pdict_init['linestyle'] = '--'
        if pdict_init['zorder'] == 0:
            pdict_init['zorder'] = 3
        pdict_init[yvals] = model.eval(
            **pdict['fit_res'].init_values,
            **{independent_var: pdict[xvals]})
        pdict_init['setlabel'] += ' init'
        plot_line(pdict_init, axs)


def plot_text(pdict, axs):
    """
    Helper function that adds text to a plot
    """
    pfunc = getattr(axs, pdict.get('func', 'text'))
    plot_text_string = pdict['text_string']
    plot_xpos = pdict.get('xpos', -0.125)
    plot_ypos = pdict.get('ypos', -0.225)
    verticalalignment = pdict.get('verticalalignment', 'top')
    horizontalalignment = pdict.get('horizontalalignment', 'right')
    color = pdict.get('color', 'k')

    # fancy box props is based on the matplotlib legend
    box_props = pdict.get('box_props', None)
    if box_props == 'fancy':
        box_props = dict(boxstyle='round', pad=.4,
                         facecolor='white', alpha=0.5)

    # pfunc is expected to be ax.text
    pfunc(x=plot_xpos, y=plot_ypos, s=plot_text_string,
          transform=axs.transAxes,
          verticalalignment=verticalalignment,
          horizontalalignment=horizontalalignment,
          color=color, bbox=box_props)


def plot_vlines(pdict, axs):
    """
    Helper function to add vlines to a plot
    """
    pfunc = getattr(axs, pdict.get('func', 'vlines'))
    x = pdict['x']
    ymin = pdict['ymin']
    ymax = pdict['ymax']
    label = pdict.get('setlabel', None)
    colors = pdict.get('colors', None)
    linestyles = pdict.get('linestyles', '--')

    axs.vlines(x, ymin, ymax, colors,
               linestyles=linestyles, label=label,
               **pdict.get('line_kws', {}))
    if pdict.get('do_legend', False):
        axs.legend()


def plot_hlines(pdict, axs):
    """
    Helper function to add vlines to a plot
    """
    y = pdict['y']
    xmin = pdict['xmin']
    xmax = pdict['xmax']
    label = pdict.get('setlabel', None)
    colors = pdict.get('colors', None)
    linestyles = pdict.get('linestyles', '--')

    axs.hlines(y, xmin, xmax, colors,
               linestyles=linestyles, label=label,
               **pdict.get('line_kws', {}))
    if pdict.get('do_legend', False):
        axs.legend()


def plot_matplot_ax_method(pdict, axs):
    """
    Used to use any of the methods of a matplotlib axis object through
    the pdict interface.

    An example pdict would be:
        {'func': 'axhline',
         'plot_kw': {'y': 0.5, 'mfc': 'green'}}
    which would call
        ax.axhline(y=0.5, mfc='green')
    to plot a horizontal green line at y=0.5

    """
    pfunc = getattr(axs, pdict.get('func'))
    pfunc(**pdict['plot_kws'])


def str_to_float(s):
    if s[-1] == '%':
        return float(s.strip('%'))/100
    else:
        return float(s)


def _globalmin(array):
    '''
    Gives the global minimum of an array (possibly a list of unequally long lists)
    :param array: array (possibly a list of unequally long lists)
    :return: Global minimum
    '''
    return np.min([np.min(v) for v in array])


def _globalmax(array):
    '''
    Gives the global maximum of an array (possibly a list of unequally long lists)
    :param array: array (possibly a list of unequally long lists)
    :return: Global maximum
    '''
    return np.max([np.max(v) for v in array])