import numpy as np
import itertools
from copy import deepcopy
import logging
log = logging.getLogger(__name__)

import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.awg_sweep_functions_multi_qubit as awg_swf2
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as mqs
import pycqed.measurement.pulse_sequences.fluxing_sequences as fsqs
import pycqed.measurement.detector_functions as det
import pycqed.analysis.fitting_models as fms
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
from pycqed.measurement.waveform_control import pulsar as ps
import pycqed.analysis.measurement_analysis as ma
import pycqed.analysis_v2.readout_analysis as ra
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.analysis_v3 import helper_functions as hlp_mod
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.utilities.general import temporary_value
from pycqed.analysis_v2 import tomography_qudev as tomo
import pycqed.analysis.analysis_toolbox as a_tools

try:
    import \
        pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQuantumController as uhfqc
except ModuleNotFoundError:
    log.warning('"UHFQuantumController" not imported.')


def get_operation_dict(qubits):
    operation_dict = dict()
    for qb in qubits:
        operation_dict.update(qb.get_operation_dict())
    return operation_dict


def get_correlation_channels(qubits, self_correlated, **kw):
    """
    Creates the correlations input parameter for the UHFQC_correlation_detector.
    :param qubits: list of QuDev_transmon instrances
    :param self_correlated: whether to do also measure self correlations
    :return: list of tuples with the channels to correlate; only looks at the
        acq_I_channel of each qubit!
    """
    if self_correlated:
        return list(itertools.combinations_with_replacement(
            [qb.acq_I_channel() for qb in qubits], r=2))
    else:
        return list(itertools.combinations(
            [qb.acq_I_channel() for qb in qubits], r=2))


def get_multiplexed_readout_detector_functions(qubits, nr_averages=None,
                                               nr_shots=None,
                                               used_channels=None,
                                               correlations=None,
                                               add_channels=None,
                                               det_get_values_kws=None,
                                               nr_samples=4096,
                                               **kw):
    if nr_averages is None:
        nr_averages = max(qb.acq_averages() for qb in qubits)
    if nr_shots is None:
        nr_shots = max(qb.acq_shots() for qb in qubits)

    uhfs = set()
    uhf_instances = {}
    max_int_len = {}
    channels = {}
    acq_classifier_params = {}
    acq_state_prob_mtxs = {}
    for qb in qubits:
        uhf = qb.instr_uhf()
        uhfs.add(uhf)
        uhf_instances[uhf] = qb.instr_uhf.get_instr()

        if uhf not in max_int_len:
            max_int_len[uhf] = 0
        max_int_len[uhf] = max(max_int_len[uhf], qb.acq_length())

        if uhf not in channels:
            channels[uhf] = []
        channels[uhf] += [qb.acq_I_channel()]
        if qb.acq_weights_type() in ['SSB', 'DSB', 'DSB2', 'optimal_qutrit']:
            if qb.acq_Q_channel() is not None:
                channels[uhf] += [qb.acq_Q_channel()]

        if uhf not in acq_classifier_params:
            acq_classifier_params[uhf] = []
        acq_classifier_params[uhf] += [qb.acq_classifier_params()]
        if uhf not in acq_state_prob_mtxs:
            acq_state_prob_mtxs[uhf] = []
        acq_state_prob_mtxs[uhf] += [qb.acq_state_prob_mtx()]

    if det_get_values_kws is None:
        det_get_values_kws = {}
        det_get_values_kws_in = None
    else:
        det_get_values_kws_in = deepcopy(det_get_values_kws)
        for uhf in acq_state_prob_mtxs:
            det_get_values_kws_in.pop(uhf, False)
    for uhf in acq_state_prob_mtxs:
        if uhf not in det_get_values_kws:
            det_get_values_kws[uhf] = {}
        det_get_values_kws[uhf].update({
            'classifier_params': acq_classifier_params[uhf],
            'state_prob_mtx': acq_state_prob_mtxs[uhf]})
        if det_get_values_kws_in is not None:
            det_get_values_kws[uhf].update(det_get_values_kws_in)
    if add_channels is None:
        add_channels = {uhf: [] for uhf in uhfs}
    elif isinstance(add_channels, list):
        add_channels = {uhf: add_channels for uhf in uhfs}
    else:  # is a dict
        pass
    for uhf in add_channels:
        channels[uhf] += add_channels[uhf]

    if correlations is None:
        correlations = {uhf: [] for uhf in uhfs}
    elif isinstance(correlations, list):
        correlations = {uhf: correlations for uhf in uhfs}
    else:  # is a dict
        for uhf in uhfs:
            if uhf not in correlations:
                correlations[uhf] = []

    if used_channels is None:
        used_channels = {uhf: None for uhf in uhfs}
    elif isinstance(used_channels, list):
        used_channels = {uhf: used_channels for uhf in uhfs}
    else:  # is a dict
        for uhf in uhfs:
            if uhf not in used_channels:
                used_channels[uhf] = None

    AWG = None
    for qb in qubits:
        qbAWG = qb.instr_pulsar.get_instr()
        if AWG is not None and qbAWG is not AWG:
            raise Exception('Multi qubit detector can not be created with '
                            'multiple pulsar instances')
        AWG = qbAWG

    individual_detectors = {uhf: {
            'int_log_det': det.UHFQC_integration_logging_det(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                result_logging_mode='raw', **kw),
            'dig_log_det': det.UHFQC_integration_logging_det(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                result_logging_mode='digitized', **kw),
            'int_avg_det': det.UHFQC_integrated_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages, **kw),
            'int_avg_classif_det': det.UHFQC_classifier_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_shots=nr_shots,
                get_values_function_kwargs=det_get_values_kws[uhf],
                result_logging_mode='raw', **kw),
            'dig_avg_det': det.UHFQC_integrated_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                result_logging_mode='digitized', **kw),
            'inp_avg_det': det.UHFQC_input_average_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, nr_averages=nr_averages,
                nr_samples=nr_samples,
                **kw),
            'int_corr_det': det.UHFQC_correlation_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                used_channels=used_channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                correlations=correlations[uhf], **kw),
            'dig_corr_det': det.UHFQC_correlation_detector(
                UHFQC=uhf_instances[uhf], AWG=AWG, channels=channels[uhf],
                used_channels=used_channels[uhf],
                integration_length=max_int_len[uhf], nr_averages=nr_averages,
                correlations=correlations[uhf], thresholding=True, **kw),
        } for uhf in uhfs}

    combined_detectors = {det_type: det.UHFQC_multi_detector([
        individual_detectors[uhf][det_type] for uhf in uhfs])
        for det_type in ['int_log_det', 'dig_log_det',
                         'int_avg_det', 'dig_avg_det', 'inp_avg_det',
                         'int_avg_classif_det', 'int_corr_det', 'dig_corr_det']}

    return combined_detectors


def get_multi_qubit_prep_params(qubits):
    prep_params_list = [qb.preparation_params() for qb in qubits]
    if len(prep_params_list) == 0:
        raise ValueError('prep_params_list is empty.')

    thresh_map = {}
    for i, prep_params in enumerate(prep_params_list):
        if 'threshold_mapping' in prep_params:
            thresh_map.update({qubits[i].name:
                                   prep_params['threshold_mapping']})

    prep_params = deepcopy(prep_params_list[0])
    prep_params['threshold_mapping'] = thresh_map
    return prep_params


def get_meas_obj_value_names_map(mobjs, multi_uhf_det_func):
    # we cannot just use the value_names from the qubit detector functions
    # because the UHF_multi_detector function adds suffixes

    if multi_uhf_det_func.detectors[0].name == 'raw_UHFQC_classifier_det':
        meas_obj_value_names_map = {
            qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                multi_uhf_det_func.value_names,
                qb.int_avg_classif_det.value_names)
            for qb in mobjs}
    elif multi_uhf_det_func.detectors[0].name == 'UHFQC_input_average_detector':
        meas_obj_value_names_map = {
            qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                multi_uhf_det_func.value_names, qb.inp_avg_det.value_names)
            for qb in mobjs}
    else:
        meas_obj_value_names_map = {
            qb.name: hlp_mod.get_sublst_with_all_strings_of_list(
                multi_uhf_det_func.value_names, qb.int_avg_det.value_names)
            for qb in mobjs}

    meas_obj_value_names_map.update({
        name + '_object': [name] for name in
        [vn for vn in multi_uhf_det_func.value_names if vn not in
         hlp_mod.flatten_list(list(meas_obj_value_names_map.values()))]})

    return meas_obj_value_names_map


def measure_multiplexed_readout(dev, qubits, liveplot=False,
                                shots=5000,
                                RO_spacing=None, preselection=True,
                                thresholds=None, thresholded=False,
                                analyse=True, upload=True):
    for qb in qubits:
        MC = qb.instr_mc.get_instr()

    for qb in qubits:
        qb.prepare(drive='timedomain')

    if RO_spacing is None:
        UHFQC = qubits[0].instr_uhf.get_instr()
        RO_spacing = UHFQC.qas_0_delay() * 2 / 1.8e9
        RO_spacing += UHFQC.qas_0_integration_length() / 1.8e9
        RO_spacing += 50e-9  # for slack
        RO_spacing = np.ceil(RO_spacing * 225e6 / 3) / 225e6 * 3
    
    operation_dict = dev.get_operation_dict(qubits=qubits)
    sf = awg_swf2.n_qubit_off_on(
        [operation_dict['X180 ' + qb.name] for qb in qubits],
        [operation_dict['RO ' + qb.name] for qb in qubits],
        preselection=preselection,
        parallel_pulses=True,
        RO_spacing=RO_spacing,
        upload=upload)

    m = 2 ** (len(qubits))
    if preselection:
        m *= 2
    if thresholded:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(qubits,
                                                        nr_shots=shots)[
            'int_log_det']

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(m))
    MC.set_detector_function(df)
    MC.run('{}_multiplexed_ssro'.format('-'.join(
        [qb.name for qb in qubits])))

    if analyse and thresholds is not None:
        channel_map = {qb.name: qb.int_log_det.value_names[0]+' '+qb.instr_uhf()
                       for qb in qubits}
        return ra.Multiplexed_Readout_Analysis(options_dict=dict(
            n_readouts=(2 if preselection else 1) * 2 ** len(qubits),
            thresholds=thresholds,
            channel_map=channel_map,
            use_preselection=preselection
        ))

def measure_ssro(dev, qubits, states=('g', 'e'), n_shots=10000, label=None,
                 preselection=True, all_states_combinations=False, upload=True,
                 exp_metadata=None, analyze=True, analysis_kwargs=None,
                 delegate_plotting=False, update=True):
    """
    Measures in single shot readout the specified states and performs
    a Gaussian mixture fit to calibrate the state classfier and provide the
    single shot readout probability assignment matrix
    Args:
        dev (Device): device object
        qubits (list): list of qubits to calibrate in parallel
        states (tuple, str, list of tuples): if tuple, each entry will be interpreted
            as a state. if string (e.g. "gef"), each letter will be interpreted
            as a state. All qubits will be prepared simultaneously in each given state.
            If list of tuples is given, then each tuple should be of length = qubits
            and the ith tuple should represent the state that each qubit should have
            in the ith segment. In the latter case, all_state_combinations is ignored.
        n_shots (int): number of shots
        label (str): measurement label
        preselection (bool, None): If True, force preselection even if not
            in preparation params. If False, then removes preselection even if in prep_params.
            if None, then takes prep_param of first qubit.

        all_states_combinations (bool): if False, then all qubits are prepared
            simultaneously in the first state and then read out, then all qubits
            are prepared in the second state, etc. If True, then all combinations
            are measured, which allows to characterize the multiplexed readout of
            each basis state. e.g. say qubits = [qb1, qb2], states = "ge" and
            all_states_combinations = False, then the different segments will be "g, g"
            and "e, e" for "qb1, qb2" respectively. all_states_combinations=True would
            yield "g,g", "g, e", "e, g" , "e,e".
        upload (bool): upload waveforms to AWGs
        exp_metadata (dict): experimental metadata
        analyze (bool): analyze data
        analysis_kwargs (dict): arguments for the analysis. Defaults to all qb names
        delegate_plotting (bool): Whether or not to create a job for an analysisDaemon
            and skip the plotting during the analysis.
        update (bool): update readout classifier parameters.
            Does not update the readout correction matrix (i.e. qb.acq_state_prob_mtx),
            as we ended up using this a lot less often than the update for readout
            classifier params. The user can still access the state_prob_mtx through
            the analysis object and set the corresponding parameter manually if desired.


    Returns:

    """
    # combine operations and preparation dictionaries
    qubits = dev.get_qubits(qubits)
    qb_names = dev.get_qubits(qubits, "str")
    operation_dict = dev.get_operation_dict(qubits=qubits)
    prep_params = dev.get_prep_params(qubits)

    if preselection is None:
        pass
    elif preselection: # force preselection for this measurement if desired by user
        prep_params['preparation_type'] = "preselection"
    else:
        prep_params['preparation_type'] = "wait"

    # create and set sequence
    if np.ndim(states) == 2: # list of custom states provided
        if len(qb_names) != len(states[0]):
            raise ValueError(f"{len(qb_names)} qubits were given but custom "
                             f"states were "
                             f"specified for {len(states[0])} qubits.")
        cp = CalibrationPoints(qb_names, states)
    else:
        cp = CalibrationPoints.multi_qubit(qb_names, states, n_per_state=1,
                                       all_combinations=all_states_combinations)
    seq = sequence.Sequence("SSRO_calibration",
                            cp.create_segments(operation_dict, **prep_params))

    # prepare measurement
    for qb in qubits:
        qb.prepare(drive='timedomain')
    label = f"SSRO_calibration_{states}{get_multi_qubit_msmt_suffix(qubits)}" if \
        label is None else label
    channel_map = {qb.name: [vn + ' ' + qb.instr_uhf()
                             for vn in qb.int_log_det.value_names]
                   for qb in qubits}
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({"cal_points": repr(cp),
                         "preparation_params": prep_params,
                         "all_states_combinations": all_states_combinations,
                         "n_shots": n_shots,
                         "channel_map": channel_map,
                         "data_to_fit": {}
                         })
    df = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=n_shots)['int_log_det']
    MC = dev.instr_mc.get_instr()
    MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq,
                                                   upload=upload))
    MC.set_sweep_points(np.arange(seq.n_acq_elements()))
    MC.set_detector_function(df)

    # run measurement
    temp_values = [(MC.soft_avg, 1)]

    # required to ensure having original prep_params after mmnt
    # in case preselection=True
    temp_values += [(qb.preparation_params, prep_params) for qb in qubits]
    with temporary_value(*temp_values):
        MC.run(name=label, exp_metadata=exp_metadata)

    # analyze
    if analyze:
        if analysis_kwargs is None:
            analysis_kwargs = dict()
        if "qb_names" not in analysis_kwargs:
            analysis_kwargs["qb_names"] = qb_names # all qubits by default
        if "options_dict" not in analysis_kwargs:
            analysis_kwargs["options_dict"] = \
                dict(delegate_plotting=delegate_plotting)
        else:
            analysis_kwargs["options_dict"].update(
                                dict(delegate_plotting=delegate_plotting))
        a = tda.MultiQutrit_Singleshot_Readout_Analysis(**analysis_kwargs)
        for qb in qubits:
            classifier_params = a.proc_data_dict[
                'analysis_params']['classifier_params'][qb.name]
            if update:
                qb.acq_classifier_params().update(classifier_params)
                if 'state_prob_mtx_masked' in a.proc_data_dict[
                        'analysis_params']:
                    qb.acq_state_prob_mtx(a.proc_data_dict['analysis_params'][
                        'state_prob_mtx_masked'][qb.name])
                else:
                    log.warning('Measurement was not run with preselection. '
                                'state_prob_matx updated with non-masked one.')
                    qb.acq_state_prob_mtx(a.proc_data_dict['analysis_params'][
                        'state_prob_mtx'][qb.name])
        return a

def find_optimal_weights(dev, qubits, states=('g', 'e'), upload=True,
                         acq_length=4096/1.8e9, exp_metadata=None,
                         analyze=True, analysis_kwargs=None,
                         acq_weights_basis=None, orthonormalize=True,
                         update=True, measure=True):
    """
    Measures time traces for specified states and
    Args:
        dev (Device): quantum device object
        qubits: qubits on which traces should be measured
        states (tuple, list, str): if str or tuple of single character strings,
            then interprets each letter as a state and does it on all qubits
             simultaneously. e.g. "ge" or ('g', 'e') --> measures all qbs
             in g then all in e.
             If list/tuple of tuples, then interprets the list as custom states:
             each tuple should be of length equal to the number of qubits
             and each state is calibrated individually. e.g. for 2 qubits:
             [('g', 'g'), ('e', 'e'), ('f', 'g')] --> qb1=qb2=g then qb1=qb2=e
             and then qb1 = "f" != qb2 = 'g'

        upload: upload waveforms to AWG
        acq_length: length of timetrace to record
        exp_metadata: experimental metadata
        acq_weights_basis (list): shortcut for analysis parameter.
            list of basis vectors used for computing the weights.
            (see Timetrace Analysis). e.g. ["ge", "gf"] yields basis vectors e - g
            and f - g. If None, defaults to  ["ge", "ef"] when more than 2
            traces are passed to the analysis and to ['ge'] if 2 traces are
            measured.
        orthonormalize (bool): shortcut for analysis parameter. Whether or not to
            orthonormalize the optimal weights (see MultiQutrit Timetrace Analysis)
        update (bool): update weights


    Returns:

    """
    # check whether timetraces can be compute simultaneously
    qubits = dev.get_qubits(qubits)
    qb_names = dev.get_qubits(qubits, "str")

    if measure:
        uhf_names = np.array([qubit.instr_uhf.get_instr().name for qubit in qubits])
        unique, counts = np.unique(uhf_names, return_counts=True)
        for u, c in zip(unique, counts):
            if c != 1:
                log.warning(f"{np.array(qubits)[uhf_names == u]} "
                            f"share the same UHF ({u}) and therefore their "
                            f"timetraces should not be measured simultaneously, "
                            f"except if you know what you are doing.")

        # combine operations and preparation dictionaries
        operation_dict = dev.get_operation_dict(qubits=qubits)
        prep_params = dev.get_prep_params(qubits)
        MC = qubits[0].instr_mc.get_instr()

        if exp_metadata is None:
            exp_metadata = dict()
        temp_val = [(qb.acq_length, acq_length) for qb in qubits]
        with temporary_value(*temp_val):
            [qb.prepare(drive='timedomain') for qb in qubits]
            npoints = qubits[0].inp_avg_det.nr_samples # same for all qubits
            sweep_points = np.linspace(0, npoints / 1.8e9, npoints,
                                                endpoint=False)
            channel_map = {qb.name: [vn + ' ' + qb.instr_uhf()
                            for vn in qb.inp_avg_det.value_names]
                            for qb in qubits}
            exp_metadata.update(
                {'sweep_name': 'time',
                 'sweep_unit': ['s'],
                 'sweep_points': sweep_points,
                 'acq_length': acq_length,
                 'channel_map': channel_map,
                 'orthonormalize': orthonormalize,
                 "acq_weights_basis": acq_weights_basis})

            for state in states:
                # create sequence
                name = f'timetrace_{state}{get_multi_qubit_msmt_suffix(qubits)}'
                if isinstance(state, str) and len(state) == 1:
                    # same state for all qubits, e.g. "e"
                    cp = CalibrationPoints.multi_qubit(qb_names, state,
                                                       n_per_state=1)
                else:
                    # ('g','e','f') as qb1=g, qb2=e, qb3=f
                    if len(qb_names) != len(state):
                        raise ValueError(f"{len(qb_names)} qubits were given "
                                         f"but custom states were "
                                         f"specified for {len(state)} qubits.")
                    cp = CalibrationPoints(qb_names, state)
                exp_metadata.update({'cal_points': repr(cp)})
                seq = sequence.Sequence("timetrace",
                                        cp.create_segments(operation_dict,
                                                           **prep_params))
                # set sweep function and run measurement
                MC.set_sweep_function(awg_swf.SegmentHardSweep(sequence=seq,
                                                               upload=upload))
                MC.set_sweep_points(sweep_points)
                df = get_multiplexed_readout_detector_functions(
                    qubits, nr_samples=npoints)["inp_avg_det"]
                MC.set_detector_function(df)
                MC.run(name=name, exp_metadata=exp_metadata)

    if analyze:
        tps = [a_tools.latest_data(
            contains=f'timetrace_{s}{get_multi_qubit_msmt_suffix(qubits)}',
            n_matches=1, return_timestamp=True)[0][0] for s in states]
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 't_start' not in analysis_kwargs:
            analysis_kwargs.update({"t_start": tps[0],
                                    "t_stop": tps[-1]})

        options_dict = dict(orthonormalize=orthonormalize,
                            acq_weights_basis=acq_weights_basis)
        options_dict.update(analysis_kwargs.pop("options_dict", {}))
        a = tda.MultiQutrit_Timetrace_Analysis(options_dict=options_dict,
                                               **analysis_kwargs)

        if update:
            for qb in qubits:
                weights = a.proc_data_dict['analysis_params_dict'
                    ]['optimal_weights'][qb.name]
                if np.ndim(weights) == 1:
                    # single channel
                    qb.acq_weights_I(weights.real)
                    qb.acq_weights_Q(weights.imag)
                elif np.ndim(weights) == 2 and len(weights) == 1:
                    # single channels
                    qb.acq_weights_I(weights[0].real)
                    qb.acq_weights_Q(weights[0].imag)
                elif np.ndim(weights) == 2 and len(weights) == 2:
                    # two channels
                    qb.acq_weights_I(weights[0].real)
                    qb.acq_weights_Q(weights[0].imag)
                    qb.acq_weights_I2(weights[1].real)
                    qb.acq_weights_Q2(weights[1].imag)
                else:
                    log.warning(f"{qb.name}: Number of weight vectors > 2: "
                                f"{len(weights)}. Cannot update weights "
                                f"automatically.")
                qb.acq_weights_basis(a.proc_data_dict['analysis_params_dict'
                    ]['optimal_weights_basis_labels'][qb.name])
        return a


def measure_tomography(dev, qubits, prep_sequence, state_name,
                       rots_basis=tomo.DEFAULT_BASIS_ROTS,
                       use_cal_points=True,
                       preselection=True,
                       rho_target=None,
                       shots=4096,
                       ro_spacing=1e-6,
                       thresholded=False,
                       liveplot=True,
                       nreps=1, run=True,
                       operation_dict=None,
                       upload=True):
    exp_metadata = {}

    MC = dev.instr_mc.get_instr()

    qubits = [dev.get_qb(qb) if isinstance(qb, str) else qb for qb in qubits]

    for qb in qubits:
        qb.prepare(drive='timedomain')

    if operation_dict is None:
        operation_dict = dev.get_operation_dict()

    qubit_names = [qb.name for qb in qubits]
    if preselection:
        label = '{}_tomography_ssro_preselection_{}'.format(state_name, '-'.join(
            [qb.name for qb in qubits]))
    else:
        label = '{}_tomography_ssro_{}'.format(state_name, '-'.join(
            [qb.name for qb in qubits]))

    seq_tomo, seg_list_tomo = mqs.n_qubit_tomo_seq(
        qubit_names, operation_dict, prep_sequence=prep_sequence,
        rots_basis=rots_basis, return_seq=True, upload=False,
        preselection=preselection, ro_spacing=ro_spacing)
    seg_list = seg_list_tomo

    if use_cal_points:
        seq_cal, seg_list_cal = mqs.n_qubit_ref_all_seq(
            qubit_names, operation_dict, return_seq=True, upload=False,
            preselection=preselection, ro_spacing=ro_spacing)
        seg_list += seg_list_cal

    seq = sequence.Sequence(label)
    for seg in seg_list:
        seq.add(seg)

    # reuse sequencer memory by repeating readout pattern
    for qbn in qubit_names:
        seq.repeat_ro(f"RO {qbn}", operation_dict)

    n_segments = seq.n_acq_elements()
    sf = awg_swf2.n_qubit_seq_sweep(seq_len=n_segments)
    if shots > 1048576:
        shots = 1048576 - 1048576 % n_segments
    if thresholded:
        df = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=shots)['dig_log_det']
    else:
        df = get_multiplexed_readout_detector_functions(
            qubits, nr_shots=shots)['int_log_det']

    # get channel map
    channel_map = get_meas_obj_value_names_map(qubits, df)
    # the above function returns channels in a list, but the state tomo analysis
    # expects a single string as values, not list
    for qb in qubits:
        if len(channel_map[qb.name]) == 1:
            channel_map[qb.name] = channel_map[qb.name][0]

    # todo Calibration point description code should be a reusable function
    #   but where?
    if use_cal_points:
        # calibration definition for all combinations
        cal_defs = []
        for i, name in enumerate(itertools.product("ge", repeat=len(qubits))):
            cal_defs.append({})
            for qb in qubits:
                if preselection:
                    cal_defs[i][channel_map[qb.name]] = \
                        [2 * len(seg_list) + 2 * i + 1]
                else:
                    cal_defs[i][channel_map[qb.name]] = \
                        [len(seg_list) + i]
    else:
        cal_defs = None

    exp_metadata["n_segments"] = n_segments
    exp_metadata["rots_basis"] = rots_basis
    if rho_target is not None:
        exp_metadata["rho_target"] = rho_target
    exp_metadata["cal_points"] = cal_defs
    exp_metadata["channel_map"] = channel_map
    exp_metadata["use_preselection"] = preselection

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    MC.live_plot_enabled(liveplot)
    MC.soft_avg(1)
    MC.set_sweep_function(sf)
    MC.set_sweep_points(np.arange(n_segments))
    MC.set_sweep_function_2D(swf.None_Sweep())
    MC.set_sweep_points_2D(np.arange(nreps))
    MC.set_detector_function(df)
    if run:
        MC.run_2D(label, exp_metadata=exp_metadata)


def measure_measurement_induced_dephasing(qb_dephased, qb_targeted, phases, amps,
                                          readout_separation, nr_readouts=1,
                                          label=None, n_cal_points_per_state=1,
                                          cal_states='auto', prep_params=None,
                                          exp_metadata=None, analyze=True,
                                          upload=True, **kw):
    classified = kw.get('classified', False)
    predictive_label = kw.pop('predictive_label', False)
    if prep_params is None:
        prep_params = get_multi_qubit_prep_params(qb_dephased)

    if label is None:
        label = 'measurement_induced_dephasing_x{}_{}_{}'.format(
            nr_readouts,
            ''.join([qb.name for qb in qb_dephased]),
            ''.join([qb.name for qb in qb_targeted]))

    hard_sweep_params = {
        'phase': {'unit': 'deg',
            'values': np.tile(phases, len(amps))},
        'ro_amp_scale': {'unit': 'deg',
            'values': np.repeat(amps, len(phases))}
    }

    for qb in set(qb_targeted) | set(qb_dephased):
        MC = qb.instr_mc.get_instr()
        qb.prepare(drive='timedomain')

    cal_states = CalibrationPoints.guess_cal_states(cal_states)
    cp = CalibrationPoints.multi_qubit([qb.name for qb in qb_dephased], cal_states,
                                       n_per_state=n_cal_points_per_state)

    operation_dict = get_operation_dict(list(set(qb_dephased + qb_targeted)))
    seq, sweep_points = mqs.measurement_induced_dephasing_seq(
        [qb.name for qb in qb_targeted], [qb.name for qb in qb_dephased], operation_dict,
        amps, phases, pihalf_spacing=readout_separation, prep_params=prep_params,
        cal_points=cp, upload=False, sequence_name=label)

    hard_sweep_func = awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload,
        parameter_name='readout_idx', unit='')
    MC.set_sweep_function(hard_sweep_func)
    MC.set_sweep_points(sweep_points)

    det_name = 'int_avg{}_det'.format('_classif' if classified else '')
    det_func = get_multiplexed_readout_detector_functions(
        qb_dephased, nr_averages=max(qb.acq_averages() for qb in qb_dephased)
    )[det_name]
    MC.set_detector_function(det_func)

    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({'qb_dephased': [qb.name for qb in qb_dephased],
                         'qb_targeted': [qb.name for qb in qb_targeted],
                         'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'classified_ro': classified,
                         'rotate': len(cal_states) != 0 and not classified,
                         'data_to_fit': {qb.name: 'pe' for qb in qb_dephased},
                         'hard_sweep_params': hard_sweep_params})

    MC.run(label, exp_metadata=exp_metadata)

    tda.MeasurementInducedDephasingAnalysis(qb_names=[qb.name for qb in qb_dephased])


def measure_drive_cancellation(
        dev, driven_qubit, ramsey_qubits, sweep_points,
        phases=None, n_pulses=1, pulse='X180',
        n_cal_points_per_state=2, cal_states='auto', prep_params=None,
        exp_metadata=None, label=None, upload=True, analyze=True):
        """
        Sweep pulse cancellation parameters and measure Ramsey on qubits the
        cancellation is for.

        Args:
            dev: The Device object used for the measurement
            driven_qubit: The qubit object corresponding to the desired
                target of the pulse that is being cancelled.
            ramsey_qubits: A list of qubit objects corresponding to the
                undesired targets of the pulse that is being cancelled.
            sweep_points: A SweepPoints object that describes the pulse
                parameters to sweep. The sweep point keys should be of the form
                `qb.param`, where `qb` is the name of the qubit the cancellation
                is for and `param` is a parameter in the pulses
                cancellation_params dict. For example to sweep the amplitude of
                the cancellation pulse on qb1, you could configure the sweep
                points as `SweepPoints('qb1.amplitude', np.linspace(0, 1, 21))`.
            phases: An array of Ramsey phases in degrees.
            n_pulses: Number of pulse repetitions done between the Ramsey
                pulses. Useful for amplification of small errors. Defaults to 1.
            pulse: Operation name (without qb name) that will be done between
                the Ramsey pulses. Defaults to 'X180'.
            n_cal_points_per_state: Number of calibration measurements per
                calibration state. Defaults to 2.
            cal_states:
                List of qubit states to use for calibration. Defaults to 'auto'.
            prep_params: Perparation parameters dictionary specifying the type
                of state preparation.
            exp_metadata: A dictionary of extra metadata to save with the
                experiment.
            label: Overwrite the default measuremnt label.
            upload: Whether the experimental sequence should be uploaded.
                Defaults to true.
            analyze: Whether the analysis will be run. Defaults to True.

        """
        if phases is None:
            phases = np.linspace(0, 360, 3, endpoint=False)

        if isinstance(driven_qubit, str):
            driven_qubit = dev.get_qb(driven_qubit)
        ramsey_qubits = [dev.get_qb(qb) if isinstance(qb, str) else qb
                         for qb in ramsey_qubits]
        ramsey_qubit_names = [qb.name for qb in ramsey_qubits]

        MC = dev.instr_mc.get_instr()
        if label is None:
            label = f'drive_{driven_qubit.name}_cancel_'\
                    f'{list(sweep_points[0].keys())}'

        if prep_params is None:
            prep_params = dev.get_prep_params(ramsey_qubits)

        sweep_points = deepcopy(sweep_points)
        sweep_points.add_sweep_dimension()
        sweep_points.add_sweep_parameter('phase', phases, 'deg', 'Ramsey phase')

        if exp_metadata is None:
            exp_metadata = {}

        for qb in [driven_qubit] + ramsey_qubits:
            qb.prepare(drive='timedomain')

        cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                        for_ef=False)
        cp = CalibrationPoints.multi_qubit(
            [qb.name for qb in ramsey_qubits], cal_states,
            n_per_state=n_cal_points_per_state)
        operation_dict = dev.get_operation_dict()

        drive_op_code = pulse + ' ' + driven_qubit.name
        # We get sweep_vals for only one dimension since drive_cancellation_seq
        # turns 2D sweep points into 1D-SegmentHardSweep.
        # FIXME: in the future, this should rather be implemented via
        # sequence.compress_2D_sweep
        seq, sweep_vals = mqs.drive_cancellation_seq(
            drive_op_code, ramsey_qubit_names, operation_dict, sweep_points,
            n_pulses=n_pulses, prep_params=prep_params, cal_points=cp,
            upload=False)

        [seq.repeat_ro(f"RO {qbn}", operation_dict)
         for qbn in ramsey_qubit_names]

        sweep_func = awg_swf.SegmentHardSweep(
                sequence=seq, upload=upload,
                parameter_name='segment_index')
        MC.set_sweep_function(sweep_func)
        MC.set_sweep_points(sweep_vals)

        det_func = get_multiplexed_readout_detector_functions(
            ramsey_qubits,
            nr_averages=max([qb.acq_averages() for qb in ramsey_qubits]))\
            ['int_avg_det']
        MC.set_detector_function(det_func)

        # !!! Watch out with the call below. See docstring for this function
        # to see the assumptions it makes !!!
        meas_obj_sweep_points_map = sweep_points.get_meas_obj_sweep_points_map(
            [qb.name for qb in ramsey_qubits])
        exp_metadata.update({
            'ramsey_qubit_names': ramsey_qubit_names,
            'preparation_params': prep_params,
            'cal_points': repr(cp),
            'sweep_points': sweep_points,
            'meas_obj_sweep_points_map': meas_obj_sweep_points_map,
            'meas_obj_value_names_map':
                get_meas_obj_value_names_map(ramsey_qubits, det_func),
            'rotate': len(cp.states) != 0,
            'data_to_fit': {qbn: 'pe' for qbn in ramsey_qubit_names}
        })

        MC.run(label, exp_metadata=exp_metadata)

        if analyze:
            return tda.DriveCrosstalkCancellationAnalysis(
                qb_names=ramsey_qubit_names, options_dict={'TwoD': True})


def measure_fluxline_crosstalk(
        dev, target_qubit, crosstalk_qubits, amplitudes,
        crosstalk_qubits_amplitudes=None, phases=None,
        target_fluxpulse_length=500e-9, crosstalk_fluxpulse_length=None,
        skip_qb_freq_fits=False, n_cal_points_per_state=2,
        cal_states='auto', prep_params=None, label=None, upload=True,
        analyze=True):
    """
    Applies a flux pulse on the target qubit with various amplitudes.
    Measure the phase shift due to these pulses on the crosstalk qubits which
    are measured in a Ramsey setting and fluxed to a more sensitive frequency.

    Args:
        dev: The Device object used for the measurement
        target_qubit: the qubit to which a fluxpulse with varying amplitude
            is applied
        crosstalk_qubits: a list of qubits to do a Ramsey on.
        amplitudes: A list of flux pulse amplitudes to apply to the target qubit
        crosstalk_qubits_amplitudes: A dictionary from crosstalk qubit names
            to flux pulse amplitudes that are applied to them to increase their
            flux sensitivity. Missing amplitudes are set to 0.
        phases: An array of Ramsey phases in degrees.
        target_fluxpulse_length: length of the flux pulse on the target qubit.
            Default: 500 ns.
        crosstalk_fluxpulse_length: length of the flux pulses on the crosstalk
            qubits. Default: target_fluxpulse_length + 50 ns.
        n_cal_points_per_state: Number of calibration measurements per
            calibration state. Defaults to 2.
        cal_states:
            List of qubit states to use for calibration. Defaults to 'auto'.
        prep_params: Perparation parameters dictionary specifying the type
            of state preparation.
        label: Overwrite the default measuremnt label.
        upload: Whether the experimental sequence should be uploaded.
            Defaults to True.
        analyze: Whether the analysis will be run. Defaults to True.

    """
    if phases is None:
        phases = np.linspace(0, 360, 3, endpoint=False)
    if crosstalk_fluxpulse_length is None:
        crosstalk_fluxpulse_length = target_fluxpulse_length + 50e-9
    if crosstalk_qubits_amplitudes is None:
        crosstalk_qubits_amplitudes = {}

    if isinstance(target_qubit, str):
        target_qubit = dev.get_qb(target_qubit)
    target_qubit_name = target_qubit.name
    crosstalk_qubits = [dev.get_qb(qb) if isinstance(qb, str) else qb
                     for qb in crosstalk_qubits]
    crosstalk_qubits_names = [qb.name for qb in crosstalk_qubits]

    MC = dev.instr_mc.get_instr()
    if label is None:
        label = f'fluxline_crosstalk_{target_qubit_name}_' + \
                ''.join(crosstalk_qubits_names)

    if prep_params is None:
        prep_params = dev.get_prep_params(crosstalk_qubits)

    sweep_points = SweepPoints('phase', phases, 'deg', 'Ramsey phase')
    sweep_points.add_sweep_dimension()
    sweep_points.add_sweep_parameter('target_amp', amplitudes, 'V',
                                     'Target qubit flux pulse amplitude')

    exp_metadata = {}

    for qb in set(crosstalk_qubits) | {target_qubit}:
        qb.prepare(drive='timedomain')

    cal_states = CalibrationPoints.guess_cal_states(cal_states,
                                                    for_ef=False)
    cp = CalibrationPoints.multi_qubit(
        [qb.name for qb in crosstalk_qubits], cal_states,
        n_per_state=n_cal_points_per_state)
    operation_dict = dev.get_operation_dict()

    # We get sweep_vals for only one dimension since drive_cancellation_seq
    # turns 2D sweep points into 1D-SegmentHardSweep.
    # FIXME: in the future, this should rather be implemented via
    # sequence.compress_2D_sweep
    seq, sweep_vals = mqs.fluxline_crosstalk_seq(
        target_qubit_name, crosstalk_qubits_names,
        crosstalk_qubits_amplitudes, sweep_points, operation_dict,
        crosstalk_fluxpulse_length=crosstalk_fluxpulse_length,
        target_fluxpulse_length=target_fluxpulse_length,
        prep_params=prep_params, cal_points=cp, upload=False)

    [seq.repeat_ro(f"RO {qbn}", operation_dict)
     for qbn in crosstalk_qubits_names]

    sweep_func = awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload,
        parameter_name='segment_index')
    MC.set_sweep_function(sweep_func)
    MC.set_sweep_points(sweep_vals)

    det_func = get_multiplexed_readout_detector_functions(
        crosstalk_qubits,
        nr_averages=max([qb.acq_averages() for qb in crosstalk_qubits])) \
        ['int_avg_det']
    MC.set_detector_function(det_func)

    # !!! Watch out with the call below. See docstring for this function
    # to see the assumptions it makes !!!
    meas_obj_sweep_points_map = sweep_points.get_meas_obj_sweep_points_map(
        [qb.name for qb in crosstalk_qubits])
    exp_metadata.update({
        'target_qubit_name': target_qubit_name,
        'crosstalk_qubits_names': crosstalk_qubits_names,
        'crosstalk_qubits_amplitudes': crosstalk_qubits_amplitudes,
        'target_fluxpulse_length': target_fluxpulse_length,
        'crosstalk_fluxpulse_length': crosstalk_fluxpulse_length,
        'skip_qb_freq_fits': skip_qb_freq_fits,
        'preparation_params': prep_params,
        'cal_points': repr(cp),
        'sweep_points': sweep_points,
        'meas_obj_sweep_points_map': meas_obj_sweep_points_map,
        'meas_obj_value_names_map':
            get_meas_obj_value_names_map(crosstalk_qubits, det_func),
        'rotate': len(cp.states) != 0,
        'data_to_fit': {qbn: 'pe' for qbn in crosstalk_qubits_names}
    })

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        return tda.FluxlineCrosstalkAnalysis(
            qb_names=crosstalk_qubits_names, options_dict={
                'TwoD': True,
                'skip_qb_freq_fits': skip_qb_freq_fits,
            })


def measure_J_coupling(dev, qbm, qbs, freqs, cz_pulse_name,
                       label=None, cal_points=False, prep_params=None,
                       cal_states='auto', n_cal_points_per_state=1,
                       freq_s=None, f_offset=0, exp_metadata=None,
                       upload=True, analyze=True):

    """
    Measure the J coupling between the qubits qbm and qbs at the interaction
    frequency freq.

    :param qbm:
    :param qbs:
    :param freq:
    :param cz_pulse_name:
    :param label:
    :param cal_points:
    :param prep_params:
    :return:
    """

    # check whether qubits are connected
    dev.check_connection(qbm, qbs)

    if isinstance(qbm, str):
        qbm = dev.get_qb(qbm)
    if isinstance(qbs, str):
        qbs = dev.get_qb(qbs)

    if label is None:
        label = f'J_coupling_{qbm.name}{qbs.name}'
    MC = dev.instr_mc.get_instr()

    for qb in [qbm, qbs]:
        qb.prepare(drive='timedomain')

    if cal_points:
        cal_states = CalibrationPoints.guess_cal_states(cal_states)
        cp = CalibrationPoints.single_qubit(
            qbm.name, cal_states, n_per_state=n_cal_points_per_state)
    else:
        cp = None
    if prep_params is None:
        prep_params = dev.get_prep_params([qbm, qbs])

    operation_dict = dev.get_operation_dict()

    # Adjust amplitude of stationary qubit
    if freq_s is None:
        freq_s = freqs.mean()

    amp_s = fms.Qubit_freq_to_dac(freq_s,
                                  **qbs.fit_ge_freq_from_flux_pulse_amp())

    fit_paras = qbm.fit_ge_freq_from_flux_pulse_amp()

    amplitudes = fms.Qubit_freq_to_dac(freqs,
                                       **fit_paras)

    amplitudes = np.array(amplitudes)

    if np.any((amplitudes > abs(fit_paras['V_per_phi0']) / 2)):
        amplitudes -= fit_paras['V_per_phi0']
    elif np.any((amplitudes < -abs(fit_paras['V_per_phi0']) / 2)):
        amplitudes += fit_paras['V_per_phi0']

    for [qb1, qb2] in [[qbm, qbs], [qbs, qbm]]:
        operation_dict[cz_pulse_name + f' {qb1.name} {qb2.name}'] \
            ['amplitude2'] = amp_s

    freqs += f_offset

    cz_pulse_name += f' {qbm.name} {qbs.name}'

    seq, sweep_points, sweep_points_2D = \
        fsqs.fluxpulse_amplitude_sequence(
            amplitudes=amplitudes, freqs=freqs, qb_name=qbm.name,
            operation_dict=operation_dict,
            cz_pulse_name=cz_pulse_name, cal_points=cp,
            prep_params=prep_params, upload=False)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload, parameter_name='Amplitude', unit='V'))

    MC.set_sweep_points(sweep_points)
    MC.set_sweep_function_2D(swf.Offset_Sweep(
        qbm.instr_ge_lo.get_instr().frequency,
        -qbm.ge_mod_freq(),
        name='Drive frequency',
        parameter_name='Drive frequency', unit='Hz'))
    MC.set_sweep_points_2D(sweep_points_2D)
    MC.set_detector_function(qbm.int_avg_det)
    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update({'sweep_points_dict': {qbm.name: amplitudes},
                         'sweep_points_dict_2D': {qbm.name: freqs},
                         'use_cal_points': cal_points,
                         'preparation_params': prep_params,
                         'cal_points': repr(cp),
                         'rotate': cal_points,
                         'data_to_fit': {qbm.name: 'pe'},
                         "sweep_name": "Amplitude",
                         "sweep_unit": "V",
                         "rotation_type": 'global_PCA'})
    MC.run_2D(label, exp_metadata=exp_metadata)

    if analyze:
        ma.MeasurementAnalysis(TwoD=True)


def measure_ramsey_add_pulse(measured_qubit, pulsed_qubit, times=None,
                             artificial_detuning=0, label='', analyze=True,
                             cal_states="auto", n_cal_points_per_state=2,
                             n=1, upload=True,  last_ge_pulse=False, for_ef=False,
                             classified_ro=False, prep_params=None,
                             exp_metadata=None):
    if times is None:
        raise ValueError("Unspecified times for measure_ramsey")
    if artificial_detuning is None:
        log.warning('Artificial detuning is 0.')
    if np.abs(artificial_detuning) < 1e3:
        log.warning('The artificial detuning is too small. The units'
                    'should be Hz.')
    if np.any(times > 1e-3):
        log.warning('The values in the times array might be too large.'
                    'The units should be seconds.')

    for qb in [pulsed_qubit, measured_qubit]:
        qb.prepare(drive='timedomain')
    MC = measured_qubit.instr_mc.get_instr()
    if prep_params is None:
        prep_params = measured_qubit.preparation_params()

    # Define the measurement label
    if label == '':
        label = 'Ramsey_add_pulse_{}'.format(pulsed_qubit.name) + \
                measured_qubit.msmt_suffix

    # create cal points
    cal_states = CalibrationPoints.guess_cal_states(cal_states, for_ef)
    cp = CalibrationPoints.single_qubit(measured_qubit.name, cal_states,
                                        n_per_state=n_cal_points_per_state)
    # create sequence
    seq, sweep_points = mqs.ramsey_add_pulse_seq_active_reset(
        times=times, measured_qubit_name=measured_qubit.name,
        pulsed_qubit_name=pulsed_qubit.name,
        operation_dict=get_operation_dict([measured_qubit, pulsed_qubit]),
        cal_points=cp, n=n, artificial_detunings=artificial_detuning,
        upload=False, for_ef=for_ef, last_ge_pulse=False, prep_params=prep_params)

    MC.set_sweep_function(awg_swf.SegmentHardSweep(
        sequence=seq, upload=upload, parameter_name='Delay', unit='s'))
    MC.set_sweep_points(sweep_points)

    MC.set_detector_function(
        measured_qubit.int_avg_classif_det if classified_ro else
        measured_qubit.int_avg_det)

    if exp_metadata is None:
        exp_metadata = {}
    exp_metadata.update(
        {'sweep_points_dict': {measured_qubit.name: times},
         'sweep_name': 'Delay',
         'sweep_unit': 's',
         'cal_points': repr(cp),
         'preparation_params': prep_params,
         'last_ge_pulses': [last_ge_pulse],
         'artificial_detuning': artificial_detuning,
         'rotate': len(cp.states) != 0,
         'data_to_fit': {measured_qubit.name: 'pf' if for_ef else 'pe'},
         'measured_qubit': measured_qubit.name,
         'pulsed_qubit': pulsed_qubit.name})

    MC.run(label, exp_metadata=exp_metadata)

    if analyze:
        tda.RamseyAddPulseAnalysis(qb_names=[measured_qubit.name])


def get_multi_qubit_msmt_suffix(qubits):
    """
    Function to get measurement label suffix from the measured qubit names.
    :param qubits: list of QuDev_transmon instances.
    :return: string with the measurement label suffix
    """
    # TODO: this was also added in Device. Remove from here when all the
    # functions that call it have been upgraded to use the Device class
    # (Steph 15.06.2020)
    qubit_names = [qb.name for qb in qubits]
    if len(qubit_names) == 1:
        msmt_suffix = qubits[0].msmt_suffix
    elif len(qubit_names) > 5:
        msmt_suffix = '_{}qubits'.format(len(qubit_names))
    else:
        msmt_suffix = '_{}'.format(''.join([qbn for qbn in qubit_names]))
    return msmt_suffix
