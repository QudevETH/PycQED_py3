import logging
log = logging.getLogger(__name__)

import lmfit
import numpy as np
from scipy import optimize
from collections import OrderedDict
from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis_v3 import fitting as fit_mod
from pycqed.analysis_v3 import plotting as plot_mod
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod
from copy import deepcopy

import sys
pp_mod.search_modules.add(sys.modules[__name__])

# Create pipelines


def pipeline_single_qubit_rb_ssro(meas_obj_names, mospm, sweep_points,
                                  n_shots, dim_hilbert, cal_points=None,
                                  ro_thresholds=None, nreps=1,
                                  plot_all_shots=False, sweep_type=None,
                                  processing_pipeline=None):

    """
    Wrapper to create the standard processing pipeline for an single qubit RB
        measurement, measured in SSRO.
    WARNING: if you use plot_all_shots=True, disable data saving. It will try
        to save a huge string of the large numpy array this node will generate.
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param n_shots: number of shots
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param ro_thresholds: optional (the threshold_data node can also extract
        them from the data_dict. See docstring there).
        Dict with meas_obj_names as keys and their readout thresholds as values.
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :param plot_all_shots: bool specifying whether to produce a raw plot of
        of all the shots vs cliffords. SEE WARNING ABOVE.
    :param sweep_type: dict of the form
        {'cliffords': sweep_dim, 'seeds': sweep_dim} where sweep_dim is either
        0 or 1 and specifies whether the measurement was run with seeds in the
        fast dimension (0) and cliffords in the slow dimensino (1), or the other
        way around.
    :param processing_pipeline: ProcessingPipeline instance to which this
        function will append.
    :return: the unresolved ProcessingPipeline
    """
    if sweep_type is None:
        sweep_type = {'cliffords': 0, 'seeds': 1}
    slow_cliffords = sweep_type['cliffords'] == 1

    sweep_points = sp_mod.SweepPoints(sweep_points)
    if cal_points is None:
        num_cal_states = 0
    else:
        if isinstance(cal_points, str):
            cal_points = cp_mod.CalibrationPoints.from_string(cal_points)
        num_cal_states = len(cal_points.states)
    if slow_cliffords:
        # n_segments = nr_seeds + nr_cal_segments
        n_segments = nreps*(sweep_points.length(sweep_type['seeds']) +
                            num_cal_states)
        # n_sequences = nr_cliffords
        n_sequences = sweep_points.length(sweep_type['cliffords'])
    else:
        # n_segments = nr_cliffords + nr_cal_segments
        n_segments = nreps*(sweep_points.length(sweep_type['cliffords']) +
                            num_cal_states)
        # n_sequences = nr_seeds
        n_sequences = sweep_points.length(sweep_type['seeds'])

    if processing_pipeline is None:
        processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_split_by_seeds',
                                     keys_in='raw',
                                     n_shots=n_shots,
                                     meas_obj_names=meas_obj_names)
    keys_in = 'previous combine_datafiles_split_by_seeds' if nreps > 1 else 'raw'
    processing_pipeline.add_node('threshold_data',
                                 keys_in=keys_in,
                                 ro_thresholds=ro_thresholds,
                                 meas_obj_names=meas_obj_names)
    processing_pipeline.add_node('average_data',
                                 # shape=(n_shots, n_segments*n_sequences),
                                 # averaging_axis=0,
                                 shape=(n_sequences, n_shots, n_segments),
                                 averaging_axis=1,
                                 keys_in='previous threshold_data',
                                 meas_obj_names=meas_obj_names)
    for label in ['rb']:
        pp = pp_mod.ProcessingPipeline(keys_out_container=label)
        pp.add_node('average_data',
                    shape=(n_sequences, n_segments),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in='previous average_data',
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(n_sequences, n_segments) ,
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in='previous average_data',
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    sweep_type=sweep_type,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_in_all_seeds_data='previous average_data',
                    do_plotting=False,
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', sweep_type['cliffords'], mospm[mobjn][
                    sweep_type['cliffords']])
            if plot_all_shots:
                pp.add_node('prepare_1d_raw_data_plot_dicts',
                            sp_name=mospm[mobjn][sweep_type['cliffords']],
                            xvals=np.repeat(cliffords, n_segments*n_shots
                                if slow_cliffords else n_sequences*n_shots),
                            do_plotting=False,
                            figname_suffix=f'shots_{label}',
                            title_suffix=' - All shots',
                            plot_params={'linestyle': 'none'},
                            keys_in=keys_in,
                            keys_out=None,
                            meas_obj_names=mobjn)
            if slow_cliffords:
                xvals = np.repeat(cliffords, n_segments)
            else:
                xvals = np.tile(cliffords, n_sequences)
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][sweep_type['cliffords']],
                        xvals=xvals,
                        do_plotting=False,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        plot_params={'linestyle': 'none'},
                        ylabel='Probability, $P(|e\\rangle)$',
                        yunit='',
                        keys_in='previous average_data',
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp

    # do plotting of all plot_dicts in the data_dict
    processing_pipeline.add_node('plot')

    return processing_pipeline


def pipeline_interleaved_rb_irb_classif(meas_obj_names, mospm, sweep_points,
                                        dim_hilbert, cal_points=None, nreps=1,
                                        sweep_type=None,
                                        processing_pipeline=None):
    """
    Wrapper to create the standard processing pipeline for an interleaved RB/RIB
        measurement, measured with a the classifier detector with qutrit readout
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :param sweep_type: dict of the form
        {'cliffords': sweep_dim, 'seeds': sweep_dim} where sweep_dim is either
        0 or 1 and specifies whether the measurement was run with seeds in the
        fast dimension (0) and cliffords in the slow dimensino (1), or the other
        way around.
    :param processing_pipeline: ProcessingPipeline instance to which this
        function will append.
    :return: the unresolved ProcessingPipeline
    """
    if sweep_type is None:
        sweep_type = {'cliffords': 0, 'seeds': 1}
    slow_cliffords = sweep_type['cliffords'] == 1

    sweep_points = sp_mod.SweepPoints(sweep_points)
    if cal_points is None:
        num_cal_states = 0
    else:
        if isinstance(cal_points, str):
            cal_points = cp_mod.CalibrationPoints.from_string(cal_points)
        num_cal_states = len(cal_points.states)

    if slow_cliffords:
        # n_segments = nr_seeds + nr_cal_segments
        n_segments = nreps*(sweep_points.length(sweep_type['seeds'])
                                   + num_cal_states)
        # n_sequences = nr_cliffords
        n_sequences = sweep_points.length(sweep_type['cliffords'])
    else:
        # n_segments = nr_cliffords + nr_cal_segments
        n_segments = nreps*(sweep_points.length(sweep_type['cliffords'])
                                   + num_cal_states)
        # n_sequences = nr_seeds
        n_sequences = sweep_points.length(sweep_type['seeds'])

    if processing_pipeline is None:
        processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_split_by_seeds',
                                     keys_in='raw',
                                     interleaved_irb=True,
                                     sweep_type=sweep_type,
                                     meas_obj_names=meas_obj_names)
    for label in ['rb', 'irb']:
        pp = pp_mod.ProcessingPipeline(global_keys_out_container=label)
        keys_in = 'previous combine_datafiles_split_by_seeds' if \
            nreps > 1 else 'raw'
        pp.add_node('submsmt_data_from_interleaved_msmt', msmt_name=label,
                    keys_in=keys_in, meas_obj_names=meas_obj_names)
        pp.add_node('average_data',
                    shape=(n_sequences, n_segments),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in=f'previous {label}.submsmt_'
                            f'data_from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(n_sequences, n_segments),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in=f'previous {label}.submsmt_'
                             f'data_from_interleaved_msmt',
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_in_all_seeds_data=f'previous {label}.submsmt_'
                                           f'data_from_interleaved_msmt',
                    do_plotting=False,
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', sweep_type['cliffords'], mospm[mobjn][
                    sweep_type['cliffords']])
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][-1],
                        xvals=np.repeat(cliffords, n_segments),
                        do_plotting=False,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        plot_params={'linestyle': 'none'},
                        ylabel='Probability, $P(|ee\\rangle)$' if
                            mobjn=='correlation_object' else None,
                        yunit='',
                        keys_in=f'previous {label}.submsmt_'
                                f'data_from_interleaved_msmt',
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp

    # calculate interleaved gate error
    processing_pipeline.add_node('irb_gate_error',
                                 meas_obj_names='correlation_object',
                                 d=dim_hilbert)

    # do plotting of all plot_dicts in the data_dict
    processing_pipeline.add_node('plot')

    return processing_pipeline


def pipeline_ssro_measurement(meas_obj_names, mospm, sweep_points, n_shots,
                              dim_hilbert, ro_thresholds=None,
                              nreps=1, interleaved_irb=False, sweep_type=None,
                              plot_all_shots=False, processing_pipeline=None,
                              compression_factor=1, **params):

    """
    Wrapper to create the standard processing pipeline for an interleaved RB/RIB
        measurement, measured in SSRO.
    WARNING: if you use plot_all_shots=True, disable data saving. It will try
        to save a huge string of the large numpy array this node will generate.
    :param meas_obj_names: list of measured object names
    :param mospm: meas_obj_sweep_points_map
    :param sweep_points: SweepPoints object (of one file if the measurement
        was split into several files)
    :param n_shots: number of shots
    :param dim_hilbert: dimension of Hilebert space. 4 for 2QB RB, 2 for 1QB RB
    :param cal_points: CalibrationPoints object
    :param ro_thresholds: optional (the threshold_data node can also extract
        them from the data_dict. See docstring there).
        Dict with meas_obj_names as keys and their readout thresholds as values.
    :param nreps: int specifying the number of files to combine into one
        measurement. IMPORTANT! This feature only works if the measurement was
        split by seeds, not by cliffords. Meaning that each measurement file
        contains data for all the Cliffords in sweep_points, but for a subset
        of the total seeds.
    :param interleaved_irb: bool specifying whether the measurement was
        IRB with RB and IRB interleaved.
    :param plot_all_shots: bool specifying whether to produce a raw plot of
        of all the shots vs cliffords. SEE WARNING ABOVE.
    :param sweep_type: dict of the form
        {'cliffords': sweep_dim, 'seeds': sweep_dim} where sweep_dim is either
        0 or 1 and specifies whether the measurement was run with seeds in the
        fast dimension (0) and cliffords in the slow dimensino (1), or the other
        way around.
    :param compression_factor: sequence compression factor
    :param processing_pipeline: ProcessingPipeline instance to which this
        function will append.
    :return: the unresolved ProcessingPipeline
    """
    if sweep_type is None:
        sweep_type = {'cliffords': 0, 'seeds': 1}
    slow_cliffords = sweep_type['cliffords'] == 1

    nr_swpts0 = sweep_points.length(0)
    nr_swpts1 = sweep_points.length(1)
    n_segments = nr_swpts0 * compression_factor
    n_sequences = (nr_swpts1 * (interleaved_irb + 1)) // compression_factor

    if processing_pipeline is None:
        processing_pipeline = pp_mod.ProcessingPipeline()
    if nreps > 1:
        processing_pipeline.add_node('combine_datafiles_split_by_seeds',
                                     keys_in='raw',
                                     n_shots=n_shots,
                                     sweep_type=sweep_type,
                                     interleaved_irb=interleaved_irb,
                                     meas_obj_names=meas_obj_names)
    keys_in = 'previous combine_datafiles_split_by_seeds' if nreps > 1 else 'raw'
    processing_pipeline.add_node('threshold_data',
                                 keys_in=keys_in,
                                 ro_thresholds=ro_thresholds,
                                 meas_obj_names=meas_obj_names)
    processing_pipeline.add_node('average_data',
                                 shape=(n_sequences, n_shots, n_segments),
                                 final_shape=(n_sequences*n_segments),
                                 averaging_axis=1,
                                 keys_in='previous threshold_data',
                                 meas_obj_names=meas_obj_names)
    if plot_all_shots:
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', sweep_type['cliffords'], mospm[mobjn])[0]
            keys_in = 'previous combine_datafiles_split_by_seeds' \
                if nreps > 1 else 'raw'
            if slow_cliffords:
                xvals = np.repeat(cliffords, 2*n_segments*n_shots if
                interleaved_irb else n_segments*n_shots)
            else:
                xvals = np.repeat(cliffords, n_sequences*n_shots)
            processing_pipeline.add_node('prepare_1d_raw_data_plot_dicts',
                                         sp_name=mospm[mobjn][-1],
                                         xvals=xvals,
                                         do_plotting=False,
                                         figname_suffix=f'shots',
                                         title_suffix=' - All shots',
                                         plot_params={'linestyle': 'none'},
                                         keys_in=keys_in,
                                         keys_out=None,
                                         meas_obj_names=mobjn)

    if dim_hilbert == 4:
        processing_pipeline.add_node('correlate_qubits',
                                     keys_in='previous threshold_data',
                                     meas_obj_names=meas_obj_names,
                                     joint_processing=True, num_keys_out=1,
                                     keys_out_container='correlation_object',
                                     add_mobjn_container=False)
        processing_pipeline.add_node('average_data',
                                     shape=(n_sequences, n_shots, n_segments),
                                     final_shape=(n_sequences*n_segments),
                                     averaging_axis=1,
                                     keys_in='previous correlate_qubits',
                                     meas_obj_names=['correlation_object'])

        meas_obj_names = deepcopy(meas_obj_names)
        meas_obj_names += ['correlation_object']
        mospm['correlation_object'] = list(mospm.values())[0]
    labels = ['rb', 'irb'] if interleaved_irb else ['rb']
    for label in labels:
        pp = pp_mod.ProcessingPipeline(keys_out_container=label)
        keys_in_0 = 'previous average_data'
        if interleaved_irb:
            pp.add_node('submsmt_data_from_interleaved_msmt',
                        msmt_name=label,
                        keys_in='previous average_data',
                        meas_obj_names=meas_obj_names)
            keys_in_0 = f'previous {label}.submsmt_data_from_interleaved_msmt'
        pp.add_node('average_data',
                    shape=(nr_swpts1, nr_swpts0),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in=keys_in_0,
                    meas_obj_names=meas_obj_names)
        pp.add_node('get_std_deviation',
                    shape=(nr_swpts1, nr_swpts0),
                    averaging_axis=-1 if slow_cliffords else 0,
                    keys_in=keys_in_0,
                    meas_obj_names=meas_obj_names)
        pp.add_node('rb_analysis',
                    d=dim_hilbert,
                    sweep_type=sweep_type,
                    msmt_type=label,
                    state_prob_name='e' if dim_hilbert==2 else None,
                    keys_in=f'previous {label}.average_data',
                    keys_in_std=f'previous {label}.get_std_deviation',
                    keys_in_all_seeds_data=keys_in_0,
                    do_plotting=False,
                    keys_out=None,
                    meas_obj_names=meas_obj_names)
        for mobjn in meas_obj_names:
            cliffords = sweep_points.get_sweep_params_property(
                'values', sweep_type['cliffords'], mospm[mobjn])[0]
            xvals = np.repeat(cliffords, nr_swpts0) if slow_cliffords else \
                np.tile(cliffords, nr_swpts1)
            pp.add_node('prepare_1d_raw_data_plot_dicts',
                        sp_name=mospm[mobjn][-1],
                        xvals=xvals,
                        do_plotting=False,
                        figname_suffix=f'{label}',
                        title_suffix=' - All seeds',
                        plot_params={'linestyle': 'none'},
                        ylabel='Probability, ' + ('$P(|ee\\rangle)$' if
                            mobjn=='correlation_object' else '$P(|e\\rangle)$'),
                        yunit='',
                        keys_in=keys_in_0,
                        keys_out=None,
                        meas_obj_names=mobjn)
        processing_pipeline += pp

    if interleaved_irb:
        # calculate interleaved gate error
        processing_pipeline.add_node(
            'irb_gate_error', meas_obj_names='correlation_object' if
            dim_hilbert == 4 else meas_obj_names, d=dim_hilbert)

    # do plotting of all plot_dicts in the data_dict
    if params.get('do_plotting', True):
        processing_pipeline.add_node('plot')

    return processing_pipeline


# nodes related to extracting data
def combine_datafiles_split_by_seeds(data_dict, keys_in, keys_out,
                                     interleaved_irb=False, **params):
    """
    NOT FULLY IMPLEMENTED FOR slow_cliffords == True!!!
    Combines the data from an (interleaved) RB/IRB measurement that was saved in
    multiple files into one data set that would look as if it had all been
    taken in one measurement (one file).
    :param data_dict: OrderedDict containing data to be processed and where
                    processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                    data_dict for the data to be processed
    :param keys_out: list of key names or dictionary keys paths in
                    data_dict for the processed data to be saved into
    :param interleaved_irb: bool specifying whether the measurement was
        IRB with RB and IRB interleaved.
    :param params: keyword arguments:
        Should contain 'exp_metadata_list', 'n_shots', 'mospm', 'rev_movnm',
        'cp' if they are not in data_dict
        ToDo: put n_shots info in the metadata (27.07.2020)
    :return:

    Assumptions:
        - ASSUMES MEASUREMENT WAS SPLIT BY SEEDS NOT BY CLIFFORDS. Meaning that
        each measurement file contains data for all the Cliffords in
        sweep_points, but for a subset of the total seeds.

    """
    assert len(keys_in) == len(keys_out)

    n_shots = hlp_mod.get_param('n_shots', data_dict, default_value=1, **params)
    mospm, rev_movnm, cp, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mospm', 'rev_movnm', 'cp', 'mobjn'],
        **params)
    metadata_list = hlp_mod.get_param('exp_metadata_list', data_dict,
                                      raise_error=True, **params)
    sp_list = [hlp_mod.get_param('sweep_points', mdl, raise_error=True)
               for mdl in metadata_list]
    sp0 = sp_mod.SweepPoints(sp_list[0])

    nr_segments = sp0.length(0) + len(cp.states)
    nr_uploads = sp0.length(1)
    chunk = nr_segments*n_shots

    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    for keyi, keyo in zip(keys_in, keys_out):
        data = data_to_proc_dict[keyi]
        if np.ndim(data) != 2:
            raise ValueError(f'Data corresponding to {keyi} is not 2D.')
        # take the segment_chunk * n_shots for each clifford from each row
        # (corresponding to data from one data file) in data and concatenate
        # them. Put all the nr_cliffords concatenations in the
        # list data_combined
        data_combined = [np.concatenate(
            [d[m * chunk + j * nr_segments: m * chunk + (j + 1) * nr_segments]
             for d in data])
            for m in np.arange((interleaved_irb + 1)*nr_uploads)
            for j in np.arange(n_shots)]
        # concatenate all the lists in data_combined to get one complete
        # array of data
        data_combined = np.concatenate(data_combined)
        hlp_mod.add_param(keyo, data_combined, data_dict, **params)

    # update the sweep_points if they were a list
    nr_sp0 = sp0.length(0)
    nr_exp = len(sp_list)
    sp_all_vals_list = [np.zeros(nr_exp*nr_sp0, dtype=int) for _
                        in range(len(sp0.get_sweep_dimension(0)))]

    for i, sp in enumerate(sp_list):
        sp = sp_mod.SweepPoints(sp)
        sp_vals_list = sp.get_sweep_params_property('values', 0, 'all')
        for j, sp_vals in enumerate(sp_vals_list):
            sp_all_vals_list[j][i::nr_exp] = sp_vals

    sweep_points = sp_mod.SweepPoints()
    for i, sp_name in enumerate(sp0.get_sweep_dimension(0)):
        sweep_points.add_sweep_parameter(
            sp_name, sp_all_vals_list[i],
            sp0.get_sweep_params_property('unit', 0, sp_name),
            sp0.get_sweep_params_property('label', 0, sp_name))
    sweep_points += [sp0.get_sweep_dimension(1)]
    hlp_mod.add_param('exp_metadata.sweep_points', sweep_points,
                      data_dict, add_param_method='replace')


def submsmt_data_from_interleaved_msmt(data_dict, keys_in, msmt_name,
                                       keys_out=None, sweep_type=None,
                                       **params):
    start_index = (msmt_name.lower() != 'rb')
    if sweep_type is None:
        sweep_type = {'cliffords': 0, 'seeds': 1}
    slow_cliffords = sweep_type['cliffords'] == 1

    n_shots = hlp_mod.get_param('n_shots', data_dict, default_value=1, **params)
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    sp, cp = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'cp'], **params)
    nr_seeds = sp.length(sweep_type['seeds']) + len(cp.states)
    nr_cliffords = sp.length(sweep_type['cliffords'])
    nr_segments = (nr_seeds if slow_cliffords else nr_cliffords) + len(cp.states)
    nr_uploads = (nr_cliffords if slow_cliffords else nr_seeds)

    if keys_out is None:
        keys_out = [f'{msmt_name}_data_from_interleaved_msmt.{s}'
                    for s in keys_in]
    for keyi, keyo in zip(keys_in, keys_out):
        data = data_to_proc_dict[keyi]
        if len(data) != nr_segments * (2 * nr_uploads):
            raise ValueError(f'The data has the wrong size of {len(data)}, '
                             f'which is not expected for {nr_segments} '
                             f'segments  and {nr_uploads} uploads.')
        selected_data = np.concatenate([
            data[j*nr_segments*n_shots:(j+1)*nr_segments*n_shots]
            for j in np.arange(2*nr_uploads)[start_index::2]])
        hlp_mod.add_param(
            keyo, selected_data, data_dict, **params)


def rb_analysis(data_dict, keys_in, sweep_type=None, **params):
    """
    Does single qubit RB analysis. Prepares fits and plots, and extracts
    errors per clifford.
    :param data_dict: OrderedDict containing data to be processed and where
                processed data is to be stored
    :param keys_in: list of key names or dictionary keys paths in
                data_dict for the data to be processed

    Assumptions:
        - cal_points, sweep_points, qb_sweep_points_map, qb_name exist in
        metadata or params
        - expects a 2d sweep with nr_seeds on innermost sweep and cliffords
        on outermost
        - if active reset was used, 'filter' must be in the key names of the
        filtered data if you want the filtered raw data to be plotted
    """
    data_to_proc_dict = hlp_mod.get_data_to_process(data_dict, keys_in)
    keys_in = list(data_to_proc_dict)

    do_fitting = hlp_mod.pop_param('do_fitting', data_dict,
                                   default_value=True, node_params=params)
    prepare_plotting = hlp_mod.pop_param('prepare_plotting', data_dict,
                                         default_value=True, node_params=params)
    do_plotting = hlp_mod.pop_param('do_plotting', data_dict,
                                    default_value=True, node_params=params)

    sp, mospm, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'mospm', 'mobjn'], **params)
    if sweep_type is None:
        sweep_type = {'cliffords': 0, 'seeds': 1}
    nr_seeds = sp.length(sweep_type['seeds'])
    if len(data_dict['timestamps']) > 1:
        nr_seeds *= len(data_dict['timestamps'])
    cliffords = sp.get_sweep_params_property('values', sweep_type['cliffords'],
                                             mospm[mobjn])[0]

    # prepare fitting
    if do_fitting:
        prepare_rb_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                        **params)

        getattr(fit_mod, 'run_fitting')(data_dict, keys_in=list(
                data_dict['fit_dicts']),**params)
        # extract EPC, leakage, and seepage from fits and save to
        # data_dict[meas_obj_name]
        analyze_rb_fit_results(data_dict, keys_in, **params)

    # prepare plots
    if prepare_plotting:
        prepare_rb_plots(data_dict, keys_in, sweep_type, **params)
        if do_plotting:
            getattr(plot_mod, 'plot')(data_dict, keys_in=list(
                data_dict['plot_dicts']), **params)


def prepare_rb_fitting(data_dict, data_to_proc_dict, cliffords, nr_seeds,
                       **params):
    cp, mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['cp', 'mobjn'], **params)
    conf_level = hlp_mod.get_param('conf_level', data_dict,
                                   default_value=0.68, **params)
    do_simple_fit = hlp_mod.get_param(
        'do_simple_fit', data_dict, default_value=True, **params)
    d = hlp_mod.get_param('d', data_dict, raise_error=True, **params)
    log.info(f'd: {d}')
    guess_pars = {'A': {'value': 1},
                  'p': {'value': 0.99},
                  'B': {'value': 0}}
    fit_guess_params = hlp_mod.get_param('fit_guess_params', data_dict,
                                         default_value={}, **params)
    guess_pars.update(fit_guess_params)

    fit_dicts = OrderedDict()
    rb_mod = lmfit.Model(fit_mods.RandomizedBenchmarkingDecay)
    rb_mod.set_param_hint('Amplitude', **guess_pars['A'])
    rb_mod.set_param_hint('p', **guess_pars['p'])
    rb_mod.set_param_hint('offset', **guess_pars['B'])
    rb_mod.set_param_hint('fidelity_per_Clifford',
                          expr=f'1-(({d}-1)*(1-p)/{d})')
    rb_mod.set_param_hint('error_per_Clifford',
                          expr='1-fidelity_per_Clifford')
    gate_decomp = hlp_mod.get_param('gate_decomp', data_dict,
                                    default_value='HZ', **params)
    if gate_decomp == 'XY':
        rb_mod.set_param_hint('fidelity_per_gate',
                              expr='fidelity_per_Clifford**(1./1.875)')
    elif gate_decomp == 'HZ':
        rb_mod.set_param_hint('fidelity_per_gate',
                              expr='fidelity_per_Clifford**(1./1.125)')
    else:
        raise ValueError('Gate decomposition not recognized.')
    rb_mod.set_param_hint('error_per_gate', expr='1-fidelity_per_gate')
    guess_pars = rb_mod.make_params()

    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict, raise_error=False,
                                    **params)
    if keys_in_std is None:
        keys_in_std = [''] * len(data_to_proc_dict)
    if len(keys_in_std) != len(data_to_proc_dict):
        raise ValueError('keys_in_std and keys_in do not have '
                         'the same length.')
    for keyi, keys in zip(data_to_proc_dict, keys_in_std):
        if 'pf' in keyi:
            # if this is the |f> state population data, then do an additional
            # fit based on the Google style
            fit_mod.prepare_rbleakage_fit_dict(
                data_dict, [keyi], indep_var_array=cliffords,
                fit_name='rbleak_fit', **params)

        # do standard fit to A*p**m + B
        key = 'rb_fit' + keyi
        data_fit = hlp_mod.get_msmt_data(data_to_proc_dict[keyi], cp, mobjn)

        model = deepcopy(rb_mod)
        fit_dicts[key] = {
            'fit_fn': fit_mods.RandomizedBenchmarkingDecay,
            'fit_xvals': {'numCliff': cliffords},
            'fit_yvals': {'data': np.array(data_fit).flatten()},
            'guess_pars': guess_pars}

        if do_simple_fit:
            fit_kwargs = {}
        elif keys is not None:
            stds = np.array(hlp_mod.get_param(keys, data_dict)).flatten()
            fit_kwargs = {'scale_covar': False,
                          'weights': 1/stds}
        else:
            # Run once to get an estimate for the error per Clifford
            fit_res = model.fit(data_fit, numCliff=cliffords,
                                params=guess_pars)
            # Use the found error per Clifford to standard errors for
            # the data points fro Helsen et al. (2017)
            epsilon_guess = hlp_mod.get_param('epsilon_guess', data_dict,
                                              default_value=0.01, **params)
            epsilon = calculate_rb_confidence_intervals(
                nr_seeds=nr_seeds,
                nr_cliffords=cliffords,
                depolariz_param=fit_res.best_values['p'],
                conf_level=conf_level,
                epsilon_guess=epsilon_guess, d=2)

            hlp_mod.add_param(
                keys, epsilon, data_dict,
                add_param_method=params.get('add_param_method', None))
            # Run fit again with scale_covar=False, and
            # weights = 1/epsilon if an entry in epsilon_sqrd is 0,
            # replace it with half the minimum value in the epsilon_sqrd
            # array
            idxs = np.where(epsilon == 0)[0]
            epsilon[idxs] = min([eps for eps in epsilon if eps != 0])/2
            fit_kwargs = {'scale_covar': False, 'weights': 1/epsilon}
        fit_dicts[key]['fit_kwargs'] = fit_kwargs

    hlp_mod.add_param('fit_dicts', fit_dicts, data_dict,
                      add_param_method='update')


def analyze_rb_fit_results(data_dict, keys_in, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    msmt_type = hlp_mod.get_param('msmt_type', data_dict, **params)
    keys_out_container = hlp_mod.get_param('keys_out_container', data_dict,
                                           default_value='', **params)
    if not len(keys_out_container) or keys_out_container is None:
        keys_out_container = f'{mobjn}.{msmt_type}'

    fit_dicts = hlp_mod.get_param('fit_dicts', data_dict, raise_error=True)
    for keyi in keys_in:
        fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
        hlp_mod.add_param(f'{keys_out_container}.EPC value',
                          fit_res.params['error_per_Clifford'].value,
                          data_dict, add_param_method='replace')
        hlp_mod.add_param(f'{keys_out_container}.EPC stderr',
                          fit_res.params['fidelity_per_Clifford'].stderr,
                          data_dict, add_param_method='replace')
        hlp_mod.add_param(
            f'{keys_out_container}.depolarization parameter value',
            fit_res.params['p'].value, data_dict,
            add_param_method='replace')
        hlp_mod.add_param(
            f'{keys_out_container}.depolarization parameter stderr',
            fit_res.params['p'].stderr, data_dict,
            add_param_method='replace')

        if 'pf' in keyi:
            A = fit_res.best_values['Amplitude']
            Aerr = fit_res.params['Amplitude'].stderr
            p = fit_res.best_values['p']
            perr = fit_res.params['p'].stderr
            # IBM-style leakage and seepage:
            # https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.032306
            hlp_mod.add_param(f'{keys_out_container}.IBM-style leakage value',
                              A*(1-p),
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.IBM-style leakage stderr',
                              np.sqrt((A*perr)**2 + (Aerr*(p-1))**2),
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.IBM-style seepage value',
                              (1-A)*(1-p),
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.IBM-style seepage stderr',
                              np.sqrt((Aerr*(p-1))**2 + ((A-1)*perr)**2),
                              data_dict,
                              add_param_method='replace')

            # Google-style leakage and seepage:
            # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.116.020501
            fit_res = fit_dicts['rbleak_fit' + keyi]['fit_res']
            hlp_mod.add_param(f'{keys_out_container}.Google-style leakage value',
                              fit_res.best_values['pu'],
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.Google-style leakage stderr',
                              fit_res.params['pu'].stderr,
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.Google-style seepage value',
                              fit_res.best_values['pd'],
                              data_dict,
                              add_param_method='replace')
            hlp_mod.add_param(f'{keys_out_container}.Google-style seepage stderr',
                              fit_res.params['pd'].stderr,
                              data_dict,
                              add_param_method='replace')

    if hlp_mod.get_param('plot_T1_lim', data_dict, default_value=False,
                         **params):
        # get T1, T2, gate length from HDF file
        get_meas_obj_coh_times(data_dict, **params)
        F_T1, p_T1 = calc_rb_coherence_limited_fidelity(
            hlp_mod.get_param(f'{mobjn}.T1', data_dict, **params),
            hlp_mod.get_param(f'{mobjn}.T2', data_dict, **params),
            hlp_mod.get_param(f'{mobjn}.ge_sigma', data_dict, **params) *
            hlp_mod.get_param(f'{mobjn}.ge_nr_sigma', data_dict, **params),
            hlp_mod.get_param('gate_decomp', data_dict,
                              default_value='HZ', **params))
        hlp_mod.add_param(f'{keys_out_container}.EPC coh_lim', 1-F_T1,
                          data_dict, add_param_method='replace')
        hlp_mod.add_param(
            f'{keys_out_container}.depolarization parameter coh_lim', p_T1,
            data_dict, add_param_method='replace')


def prepare_rb_plots(data_dict, keys_in, sweep_type, **params):
    sp, cp, mospm, mobjn, movnm = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['sp', 'cp', 'mospm', 'mobjn', 'movnm'],
        **params)

    plot_dicts = OrderedDict()
    keys_in_std = hlp_mod.get_param('keys_in_std', data_dict, raise_error=False,
                                    **params)
    stpn = hlp_mod.get_param(
        'state_prob_name', data_dict,
        default_value='gg' if 'corr' in mobjn else 'e', **params)
    classified_msmt = any([v == 3 for v in [len(chs) for chs in movnm.values()]])
    lw = plot_mod.get_default_plot_params(
        set_params=False, return_full_rc_params=True)['lines.linewidth']
    ms = plot_mod.get_default_plot_params(
        set_params=False, return_full_rc_params=True)['lines.markersize']
    llsp = plot_mod.get_default_plot_params(
        set_params=False, return_full_rc_params=True)['legend.labelspacing']
    lcsp = plot_mod.get_default_plot_params(
        set_params=False, return_full_rc_params=True)['legend.columnspacing']

    ylabel = hlp_mod.pop_param('ylabel', data_dict, node_params=params)
    if ylabel is None:
        if isinstance(stpn, (tuple, list)):
            # assumed of the form ('gg', '+', 'ee', '-', 'ge', '-', 'eg')
            prob_states = stpn[0::2]
            prob_labels = [f'$P(|{{{p}}}\\rangle)$' for p in prob_states]
            ylabel = (2*len(prob_states)-1)*['']
            ylabel[0::2] = prob_labels
            ylabel[1::2] = list(stpn[1::2])
            ylabel = ''.join(ylabel)
        else:
            ylabel = f'Probability, $P(|{{{stpn}}}\\rangle)$'
    figure_name_suffix = hlp_mod.get_param('figure_name_suffix', data_dict,
                                           default_value='', **params)
    for keyi, keys in zip(keys_in, keys_in_std):
        figure_name = f'RB_{keyi}_{mobjn}{figure_name_suffix}'
        sp_name = [p for p in mospm[mobjn] if 'clifford' in p][0]

        # plot data
        pd = \
            plot_mod.prepare_1d_plot_dicts(data_dict=data_dict, keys_in=[keyi],
                                           figure_name=figure_name,
                                           ylabel=ylabel,
                                           sp_name=sp_name,
                                           yerr_key=keys,
                                           data_labels=['avg.'],
                                           plot_params={
                                               'zorder': 2, 'marker': 'o',
                                               'legend_ncol': 3,
                                               'line_kws': {
                                                   'elinewidth': lw+3,
                                                   'markersize': ms+1,
                                                   'alpha_errorbars': 0.25}},
                                           do_plotting=False, **params)
        plot_dicts.update(pd)

        # plot all seeds
        keys_in_all_seeds_data = hlp_mod.get_param('keys_in_all_seeds_data',
                                                   data_dict, **params)
        clf_dim = sweep_type['cliffords']
        seeds_dim = sweep_type['seeds']
        cliffords = sp.get_sweep_params_property('values', clf_dim, sp_name)
        xvals = np.repeat(cliffords, sp.length(seeds_dim)) if clf_dim == 1 \
            else np.tile(cliffords, sp.length(seeds_dim))
        if keys_in_all_seeds_data is not None:

            pd = \
                plot_mod.prepare_1d_plot_dicts(data_dict=data_dict,
                                               keys_in=keys_in_all_seeds_data,
                                               figure_name=figure_name,
                                               xvals=xvals,
                                               ylabel=ylabel,
                                               sp_name=sp_name,
                                               data_labels=['seeds'],
                                               plot_params={
                                                   'linestyle': 'none',
                                                   'marker': '.',
                                                   'color': 'gray',
                                                   'line_kws': {'alpha': 0.5},
                                                   'zorder': 1},
                                               do_plotting=False, **params)
            plot_dicts.update(pd)

        if len(cp.states) != 0:
            # plot cal states
            plot_dicts.update(
                plot_mod.prepare_cal_states_plot_dicts(data_dict=data_dict,
                                                       keys_in=[keyi],
                                                       figure_name=figure_name,
                                                       sp_name=sp_name,
                                                       do_plotting=False,
                                                       **params))

        if 'fit_dicts' in data_dict:
            # plot fits
            fit_dicts = data_dict['fit_dicts']
            textstr = ''
            if 'pf' in keyi:
                # plot Google-style leakage fit + textbox
                plot_dicts.update(plot_mod.prepare_fit_plot_dicts(
                    data_dict=data_dict,
                    figure_name=figure_name,
                    fit_names=['rbleak_fit' + keyi],
                    plot_params={'color': 'C1',
                                 'setlabel': 'fit - Google',
                                 'legend_ncol': 3},
                    do_plotting=False, **params))
                textstr += get_rb_textbox_properties(
                    data_dict, fit_dicts['rbleak_fit' + keyi]['fit_res'],
                    textstr_style=['leakage_google'],
                    **params)[0]

            # plot fit trace
            pd = plot_mod.prepare_fit_plot_dicts(
                data_dict=data_dict,
                figure_name=figure_name,
                fit_names=['rb_fit' + keyi],
                plot_params={'color': 'C0',
                             'setlabel': 'fit - IBM' if 'pf' in keyi else 'fit',
                             'legend_ncol': 3},
                do_plotting=False, **params)
            plot_dicts.update(pd)

            # plot coherence-limit
            fit_res = fit_dicts['rb_fit' + keyi]['fit_res']
            if hlp_mod.get_param('plot_T1_lim', data_dict,
                    default_value=False, **params) and 'pf' not in keyi:
                keys_out_container = hlp_mod.get_param('keys_out_container',
                                                       data_dict,
                                                       default_value=mobjn,
                                                       **params)
                epc_T1 = hlp_mod.get_param(f'{keys_out_container}.EPC coh_lim',
                                         data_dict,  **params)
                p_T1 = hlp_mod.get_param(
                    f'{keys_out_container}.depolarization parameter coh_lim',
                    data_dict,  **params)
                clfs_fine = np.linspace(cliffords[0], cliffords[-1], 1000)
                T1_limited_curve = fit_res.model.func(
                    clfs_fine, fit_res.best_values['Amplitude'], p_T1,
                    fit_res.best_values['offset'])
                plot_dicts['t1Lim_' + keyi] = {
                    'fig_id': figure_name,
                    'plotfn': 'plot_line',
                    'xvals': clfs_fine,
                    'yvals': T1_limited_curve,
                    'setlabel': 'coh-lim',
                    'do_legend': True,
                    'linestyle': '--',
                    'line_kws': {'linewidth': lw-0.5},
                    'zorder': 0,
                    'marker': ''}
            else:
                epc_T1 = None

            # add texbox
            va_text = hlp_mod.get_param('va_text', data_dict, **params)
            if va_text is None:
                va_text = 'top' if 'g' in stpn else 'bottom'
            textstr, ha, hp, va, vp = get_rb_textbox_properties(
                data_dict, fit_res, epc_T1=None if 'pf' in keyi else epc_T1,
                va=va_text,
                textstr_style='leakage_ibm' if 'pf' in keyi else 'regular',
                textstr=textstr if 'pf' in keyi else '', **params)
            plot_dicts['text_msg_' + keyi] = {
                'fig_id': figure_name,
                'plotfn': 'plot_text',
                'ypos': vp,
                'xpos': hp,
                'horizontalalignment': ha,
                'verticalalignment': va,
                'box_props': None,
                'text_string': textstr}

        plot_dicts[list(plot_dicts)[-2]].update({
            'legend_bbox_to_anchor': (1.025, -0.15),
            'legend_pos': 'upper right',
            'legend_labelspacing': llsp-0.25,
            'legend_columnspacing': lcsp-1,
            'legend_ncol': 1 if 'pf' in keyi else 2,
            'yrange': hlp_mod.get_param('yrange', data_dict, **params)
        })
    hlp_mod.add_param('plot_dicts', plot_dicts, data_dict,
                      add_param_method='update')


def prepare_irb_plot(data_dict, plot_dict_names_irb_plot=None,
                     figure_name=None, **params):

    plot_dicts_updated = OrderedDict()
    do_plotting = params.pop('do_plotting', False)
    if figure_name is None:
        figure_name = 'IRB'

    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'],
        **params)

    if plot_dict_names_irb_plot is None:
        plot_dict_names_irb_plot = hlp_mod.get_param(
            'plot_dict_names_irb_plot', data_dict, **params)

    plot_dicts = hlp_mod.get_param('plot_dicts', data_dict, **params)
    for label in ['rb', 'irb']:
        epc_value = hlp_mod.get_param(f'{mobjn}.{label}.EPC value',
                                      data_dict, **params)
        log.info(epc_value)
        leg_label = ''
        if epc_value is not None:
            epc_stderr = hlp_mod.get_param(f'{mobjn}.{label}.EPC stderr',
                                           data_dict, **params)
            leg_label = f'{label.upper()}:\t' \
                        f'{100*epc_value:.2f}%$\\pm${100*epc_stderr:.2f}% EPC'
        log.info(leg_label)
        plot_dicts_updated[f'legend_data_IRB_{label}'] = {
            'fig_id': figure_name,
            'plotfn': 'plot_line',
            'xvals': [],
            'yvals': [],
            'color': 'C0' if label == 'rb' else 'C1',
            'marker': 'o',
            'linestyle': '-',
            'setlabel': leg_label,
        }

    pd_plot_type = [pdn for pdn in plot_dict_names_irb_plot['rb']
                    if 'seeds' in pdn]
    if len(pd_plot_type):
        pd_name = plot_dict_names_irb_plot['rb'][pd_plot_type[0]]
        plot_dicts_updated['legend_seeds_IRB'] = \
            deepcopy(plot_dicts[f'{pd_name}'])
        plot_dicts_updated['legend_seeds_IRB'].update({
            'xvals': [], 'yvals': [], 'yerr': None,
            'setlabel': 'all seeds'})

    cz_err = hlp_mod.get_param('cz_err_value', data_dict, **params)
    if cz_err is None:
        cz_err = hlp_mod.get_param(
            'correlation_object.average_gate_error_CZ value', data_dict)
    cz_err_stderr = hlp_mod.get_param('cz_err_stderr', data_dict, **params)
    if cz_err_stderr is None:
        cz_err_stderr = hlp_mod.get_param(
            'correlation_object.average_gate_error_CZ stderr', data_dict)
    if cz_err is not None:
        textstr = \
            f'Gate error:\n{100*cz_err:.2f}%$\\pm${100*cz_err_stderr:.2f}%'
        plot_dicts_updated['text_msg_IRB'] = {
            'fig_id': figure_name,
            'plotfn': 'plot_text',
            'ypos': 0.05,
            'xpos': 0.4,
            'horizontalalignment': 'left',
            'verticalalignment': 'bottom',
            'box_props': None,
            'text_string': textstr}

    for label in ['rb', 'irb']:
        for plot_type in list(plot_dict_names_irb_plot[label])[::-1]:
            pd_name = plot_dict_names_irb_plot[label][plot_type]
            plot_dicts_updated[f'{pd_name} IRB'] = deepcopy(plot_dicts[pd_name])
            updated_vals = {'fig_id': figure_name,
                            'color': 'C0' if label == 'rb' else 'C1',
                            'setlabel': '', 'legend_ncol': 1}
            plot_dicts_updated[f'{pd_name} IRB'].update(updated_vals)

    plotsize = plot_mod.get_default_plot_params(
        set_params=False, return_full_rc_params=True)['figure.figsize']
    plotsize = (plotsize[0], 3*plotsize[1])
    last_pd = plot_dicts_updated[list(plot_dicts_updated)[-1]]
    last_pd.update({'legend_bbox_to_anchor': (0.35, 0.08),
                    'legend_ncol': 1,
                    'legend_pos': 'lower left',
                    'plotsize': plotsize})

    hlp_mod.add_param('plot_dicts', plot_dicts_updated, data_dict,
                      add_param_method='update')
    if do_plotting:
        getattr(plot_mod, 'plot')(data_dict, keys_in=list(plot_dicts),
                                  **params)


def get_rb_leakage_ibm_textstr(data_dict, fit_res=None, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    msmt_type = hlp_mod.get_param('msmt_type', data_dict, **params)
    keys_out_container = hlp_mod.get_param('keys_out_container', data_dict,
                                           default_value='', **params)
    if not len(keys_out_container) or keys_out_container is None:
        keys_out_container = f'{mobjn}.{msmt_type}'
    textstr = 'IBM style:'
    p_value = hlp_mod.get_param(
        f'{keys_out_container}.depolarization parameter value', data_dict,
        raise_error=True)
    p_stderr = hlp_mod.get_param(
        f'{keys_out_container}.depolarization parameter stderr', data_dict)
    textstr += f'\np = {100*p_value:.4f}%'
    if p_stderr is not None:
        textstr += f'$\pm$ {100*p_stderr:.3f}%'

    L_value = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style leakage value', data_dict,
        raise_error=True)
    textstr += f'\nL = {100*L_value:.4f}%'
    L_stderr = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style leakage stderr', data_dict)
    if L_stderr is not None:
        textstr += f'$\pm$ {100*L_stderr:.3f}%'

    S_value = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style seepage value', data_dict,
        raise_error=True)
    textstr += f'\nS = {100*S_value:.4f}%'
    S_stderr = hlp_mod.get_param(
        f'{keys_out_container}.IBM-style seepage stderr', data_dict)
    if S_stderr is not None:
        textstr += f'$\pm$ {100*S_stderr:.3f}%'
    return textstr


def get_rb_leakage_google_textstr(fit_res, **params):
    textstr = 'Google style:'
    textstr += ('\n$p_{\\uparrow}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pu'].value*100,
                    fit_res.params['pu'].stderr*100) +
                '\n$p_{\\downarrow}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pd'].value*100,
                    fit_res.params['pd'].stderr*100) +
                '\n$p_0$' + ' = {:.2f}% $\pm$ {:.2f}%\n'.format(
                fit_res.params['p0'].value,
                fit_res.params['p0'].stderr))
    return textstr


def get_rb_regular_textstr(fit_res, epc_T1=None, **params):
    textstr = ('$r_{\mathrm{Cl}}$' + ' = {:.4f}% $\pm$ {:.3f}%'.format(
        (1-fit_res.params['fidelity_per_Clifford'].value)*100,
        fit_res.params['fidelity_per_Clifford'].stderr*100))
    if epc_T1 is not None:
        textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                    '{:.3f}%'.format(epc_T1*100))
    textstr += ('\n' + 'p = {:.4f}% $\pm$ {:.3f}%'.format(
        fit_res.params['p'].value*100, fit_res.params['p'].stderr*100))
    textstr += ('\n' + r'$\langle \sigma_z \rangle _{m=0}$ = ' +
                '{:.2f} $\pm$ {:.2f}'.format(
                    fit_res.params['Amplitude'].value +
                    fit_res.params['offset'].value,
                    np.sqrt(fit_res.params['offset'].stderr**2 +
                            fit_res.params['Amplitude'].stderr**2)))
    return textstr


def get_cz_irb_textstr(fit_res,  epc_T1=None, **params):
    suffix = params.get('suffix', 'RB')
    textstr = (f'$r_{{\mathrm{{Cl}}, {{{suffix}}}}}$' +
               ' = {:.4f}% $\pm$ {:.3f}%'.format(
        (1-fit_res.params['fidelity_per_Clifford'].value)*100,
        fit_res.params['fidelity_per_Clifford'].stderr*100))
    if epc_T1 is not None:
        textstr += ('\n$r_{\mathrm{coh-lim}}$  = ' +
                    '{:.3f}%'.format(epc_T1*100))
    textstr += (f'\n$p_{{\\uparrow, {suffix}}}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pu'].value*100,
                    fit_res.params['pu'].stderr*100) +
                f'\n$p_{{\\downarrow, {suffix}}}$' +
                ' = {:.4f}% $\pm$ {:.3f}%'.format(
                    fit_res.params['pd'].value*100,
                    fit_res.params['pd'].stderr*100))
    return textstr


def get_rb_textbox_properties(data_dict, fit_res, epc_T1=None,
                              textstr_style=(), textstr='', **params):
    if len(textstr_style) != 0:
        textstr += '\n'
        if 'regular' in textstr_style:
            textstr += get_rb_regular_textstr(fit_res, epc_T1, **params)
        if 'leakage_google' in textstr_style:
            textstr += get_rb_leakage_google_textstr(fit_res, **params)
        if 'leakage_ibm' in textstr_style:
            textstr += get_rb_leakage_ibm_textstr(data_dict, **params)
        if 'irb' in textstr_style:
            textstr += get_cz_irb_textstr(fit_res, **params)
        if len(textstr) == 0:
            raise NotImplementedError(f'The textstring style {textstr_style} '
                                      f'has not been implemented yet.')

    va = 'top'
    vp = -0.15
    ha = 'left'
    hp = -0.12

    return textstr, ha, hp, va, vp


def irb_gate_error(data_dict, keys_container_rb, keys_container_irb, **params):
    """
    Calculates the average gate error from a set of RB-IRB measurements and
    saves it in data_dict.
    :param data_dict: OrderedDict containing the results of running rb_analysis
        node.
    :param params: keyword arguments:
        meas_obj_names (str): name of the measurement object
            for which to calculate average gate error.
            Should be correlation_object for a two-qubit RB.
        d (int): dimension of the Hilbert space
        interleaved_gate (str or int): the interleaved gate for which to
            calculate average gate error.

    Assumptions:
        - meas_obj_names, d, interleaved_gate must exist wither in data_dict,
        metadata, or params
    """
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    d = hlp_mod.get_param('d', data_dict, raise_error=True, **params)
    interleaved_gate = hlp_mod.get_param(
        'interleaved_gate', data_dict, raise_error=True, **params)
    if interleaved_gate == 4368:
        interleaved_gate = 'CZ'

    keys_out_container = hlp_mod.get_param('keys_out_container', data_dict,
                                           default_value='', **params)
    prb = hlp_mod.get_param(
        f'{keys_container_rb}.depolarization parameter value', data_dict,
        raise_error=True, **params)
    prb_err = hlp_mod.get_param(
        f'{keys_container_rb}.depolarization parameter stderr', data_dict,
        raise_error=True, **params)
    pirb = hlp_mod.get_param(
        f'{keys_container_irb}.depolarization parameter value', data_dict,
        raise_error=True, **params)
    pirb_err = hlp_mod.get_param(
        f'{keys_container_irb}.depolarization parameter stderr', data_dict,
        raise_error=True, **params)

    if not len(keys_out_container) or keys_out_container is None:
        keys_out_container = f'{mobjn}.average_gate_error_{interleaved_gate}'
    if mobjn not in keys_out_container:
        keys_out_container = f'{mobjn}.{keys_out_container}'
    hlp_mod.add_param(f'{keys_out_container}.value',
                      ((d-1)/d)*(1 - pirb/prb),
                      data_dict, **params)
    hlp_mod.add_param(f'{keys_out_container}.stderr',
                      ((d-1)/d)*np.sqrt((pirb_err*prb)**2 +
                                        (prb_err*pirb)**2)/(prb**2),
                      data_dict, **params)


def calc_rb_coherence_limited_fidelity(T1, T2, pulse_length, gate_decomp='HZ'):
    """
    Formula from Asaad et al. (2016):
    https://www.nature.com/articles/npjqi201629

    Returns:
        F_cl (float): decoherence limited fildelity
        p (float): decoherence limited depolarization parameter
    """
    # Np = 1.875  # Avg. number of gates per Clifford for XY decomposition
    # Np = 1.125  # Avg. number of gates per Clifford for HZ decomposition
    if gate_decomp == 'HZ':
        Np = 1.125
    elif gate_decomp == 'XY':
        Np = 1.875
    else:
        raise ValueError('Gate decomposition not recognized.')

    F_cl = (1/6*(3 + 2*np.exp(-1*pulse_length/(T2)) +
                 np.exp(-pulse_length/T1)))**Np
    p = 2*F_cl - 1

    return F_cl, p


def get_meas_obj_coh_times(data_dict, extract_T2s=True, **params):
    mobjn = hlp_mod.get_measurement_properties(
        data_dict, props_to_extract=['mobjn'], **params)
    # Get from the hdf5 file any parameters specified in
    # params_dict and numeric_params.
    params_dict = {}
    s = 'Instrument settings.' + mobjn
    for trans_name in ['', '_ef']:
        if hlp_mod.get_param(f'{mobjn}.T1{trans_name}', data_dict) is None:
            params_dict[f'{mobjn}.T1{trans_name}'] = s + f'.T1{trans_name}'
        if hlp_mod.get_param(f'{mobjn}.T2{trans_name}', data_dict) is None:
            params_dict[f'{mobjn}.T2{trans_name}'] = s + (
                f'.T2_star{trans_name}' if extract_T2s else f'.T2{trans_name}')
    for trans_name in ['ge', 'ef']:
        if hlp_mod.get_param(f'{mobjn}.T1{trans_name}', data_dict) is None and \
                hlp_mod.get_param(f'{mobjn}.T1{trans_name}', data_dict) is None:
            params_dict[f'{mobjn}.{trans_name}_sigma'] = \
                s + f'.{trans_name}_sigma'
            params_dict[f'{mobjn}.{trans_name}_nr_sigma'] = \
                s + f'.{trans_name}_nr_sigma'
    if len(params_dict) > 0:
        hlp_mod.get_params_from_files(data_dict, params_dict=params_dict,
                                      numeric_params=list(params_dict),
                                      **params)


def calculate_rb_confidence_intervals(
        nr_seeds, nr_cliffords, conf_level=0.68, depolariz_param=1,
        epsilon_guess=0.01, d=2):

    # From Helsen et al. (2017)
    # For each number of cliffords in nr_cliffords (array), finds epsilon
    # such that with probability greater than conf_level, the true value of
    # the survival probability, p_N_m, for a given N=nr_seeds and
    # m=nr_cliffords, is in the interval
    # [p_N_m_measured-epsilon, p_N_m_measured+epsilon]
    # See Helsen et al. (2017) for more details.

    # eta is the SPAM-dependent prefactor defined in Helsen et al. (2017)
    epsilon = []
    delta = 1-conf_level
    infidelity = (d-1)*(1-depolariz_param)/d

    for n_cl in nr_cliffords:
        if n_cl == 0:
            epsilon.append(0)
        else:
            if d == 2:
                V_short_n_cl = (13*n_cl*infidelity**2)/2
                V_long_n_cl = 7*infidelity/2
                V = min(V_short_n_cl, V_long_n_cl)
            else:
                V_short_n_cl = \
                    (0.25*(-2+d**2)/((d-1)**2)) * (infidelity**2) + \
                    (0.5*n_cl*(n_cl-1)*(d**2)/((d-1)**2)) * (infidelity**2)
                V1 = 0.25*((-2+d**2)/((d-1)**2))*n_cl*(infidelity**2) * \
                     depolariz_param**(n_cl-1) + ((d/(d-1))**2) * \
                     (infidelity**2)*(
                             (1+(n_cl-1)*(depolariz_param**(2*n_cl)) -
                              n_cl*(depolariz_param**(2*n_cl-2))) /
                             (1-depolariz_param**2)**2 )
                V = min(V1, V_short_n_cl)
            H = lambda eps: (1/(1-eps))**((1-eps)/(V+1)) * \
                            (V/(V+eps))**((V+eps)/(V+1)) - \
                            (delta/2)**(1/nr_seeds)
            epsilon.append(optimize.fsolve(H, epsilon_guess)[0])
    return np.asarray(epsilon)


