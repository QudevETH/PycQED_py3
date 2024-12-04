import matplotlib.pyplot as plt
import numpy as np
import lmfit
from copy import deepcopy

from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.measurement.calibration import two_qubit_gates as twoqbcal
import pycqed.measurement.sweep_points as sp_mod
from pycqed.utilities.general import temporary_value

#############################################################################
#                   CZ related
#############################################################################

def update_cz_amplitude(qbc, qbt, phases, amplitudes, target_phase=np.pi,
                        update=True):
    print(f"old amplitude: {qbc.get('upCZ_{}_amplitude'.format(qbt.name))}")
    print(f"amplitudes: {amplitudes}")
    phases %= 2*np.pi
    print(f"phases: {phases}")
    fit_res = lmfit.Model(lambda x, m, b: m*np.tan(x/2-np.pi/2) + b).fit(
        x=phases, data=amplitudes, m=1, b=np.mean(amplitudes))
    new_ampl = fit_res.model.func(target_phase, **fit_res.best_values)
    print('BEST {} '.format('amplitude'), new_ampl)
    if update:
        qbc.set('upCZ_{}_amplitude'.format(qbt.name), new_ampl)


def get_optimal_amp(qbc, qbt, timestamp=None,
                    classified_ro=False, tangent_fit=False,
                    parfit=False, phi=180,
                    analysis_object=None, **kw):

    if analysis_object is None:
        if classified_ro:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_acq() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in [qbc, qbt]}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_acq() for vn in
                                     qb.int_avg_det.value_names]
                           for qb in [qbc, qbt]}
        tdma = tda.CPhaseLeakageAnalysis(
            t_start=timestamp,
            qb_names=[qbc.name, qbt.name],
            options_dict={'TwoD': True, 'plot_all_traces': False,
                          'plot_all_probs': False,
                          'delegate_plotting': False,
                          'channel_map': channel_map})
    else:
        tdma = analysis_object
    cphases = tdma.proc_data_dict[
        'analysis_params_dict'][f'cphase_{qbt.name}']['val']
    cphases = np.unwrap(cphases, period=2 * np.pi)
    # Add an integer multiple of 2*pi to the cphases, such that
    # cphases[len(cphases) // 2] is close to phi. This should roughly avoid
    # that the gate jump around by 2*pi (depending on the measurement range).
    cphases = cphases - 2*np.pi *\
        np.round((cphases[len(cphases) // 2] - phi*np.pi/180) / (2*np.pi))

    # -1 to get the soft (last) dimension
    soft_sweep_param = tdma.mospm[qbc.name][-1]
    soft_sweep_points = tdma.sp[1]
    sweep_pts = soft_sweep_points[soft_sweep_param][0]
    if tangent_fit:
        fit_res = lmfit.Model(lambda x, m, b: m*np.tan(x/2-np.pi/2) + b).fit(
            x=cphases, data=sweep_pts,
            m=(max(sweep_pts)-min(sweep_pts))/((max(cphases)-min(cphases))),
            b=np.min(sweep_pts))
    elif parfit:
        fit_res = lmfit.Model(lambda x, m, b, c: m*x + c*x**2 + b).fit(
            x=cphases, data=sweep_pts,
            m=(max(sweep_pts)-min(sweep_pts))/((max(cphases)-min(cphases))),
            c=0.001,
            b=np.min(sweep_pts))
    else:
        fit_res = lmfit.Model(lambda x, m, b: m*x + b).fit(
            x=cphases, data=sweep_pts,
            m=(max(sweep_pts)-min(sweep_pts))/((max(cphases)-min(cphases))),
            b=np.min(sweep_pts))
    plot_and_save_cz_amp_sweep(cphases=cphases, timestamp=timestamp,
                               soft_sweep_params_dict=soft_sweep_points,
                               sweep_param_name=soft_sweep_param,
                               fit_res=fit_res, save_fig=True, plot_guess=False,
                               qbc_name=qbc.name, qbt_name=qbt.name, phi=phi,
                               **kw)
    best_val = fit_res.model.func(phi * np.pi / 180, **fit_res.best_values)
    converged = best_val == np.clip(best_val, sweep_pts[0], sweep_pts[-1])
    return best_val, converged


def plot_and_save_cz_amp_sweep(cphases, soft_sweep_params_dict,
                               sweep_param_name, fit_res,
                               qbc_name, qbt_name, save_fig=True, show=True,
                               plot_guess=False, timestamp=None, phi=180):

    sweep_points = soft_sweep_params_dict[sweep_param_name][0]
    unit = soft_sweep_params_dict[sweep_param_name][1]
    best_val = fit_res.model.func(phi*np.pi/180, **fit_res.best_values)
    fit_points_init = fit_res.model.func(cphases, **fit_res.init_values)
    fit_points = fit_res.model.func(cphases, **fit_res.best_values)

    fig, ax = plt.subplots()
    ax.plot(cphases*180/np.pi, sweep_points, 'o-')
    ax.plot(cphases*180/np.pi, fit_points, '-r')
    if plot_guess:
        ax.plot(cphases*180/np.pi, fit_points_init, '--k')
    ax.hlines(best_val, cphases[0]*180/np.pi, cphases[-1]*180/np.pi)
    ax.vlines(phi, sweep_points.min(), sweep_points.max())
    ax.set_ylabel('Flux pulse {} ({})'.format(sweep_param_name, unit))
    ax.set_xlabel('Conditional phase (deg)')
    ax.set_title('CZ {}-{}'.format(qbc_name, qbt_name))

    ax.text(0.5, 0.95, 'Best {} = {:.6f} ({})'.format(
        sweep_param_name, best_val*1e9 if unit == 's' else best_val,
        'ns' if unit == 's' else unit),
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    if save_fig:
        import datetime
        import os
        fig_title = 'CPhase_amp_sweep_{}_{}'.format(qbc_name, qbt_name)
        fig_title = '{}--{:%Y%m%d_%H%M%S}'.format(
            fig_title, datetime.datetime.now())
        if timestamp is None:
            save_folder = a_tools.latest_data()
        else:
            save_folder = a_tools.get_folder(timestamp)
        filename = os.path.abspath(os.path.join(save_folder, fig_title+'.png'))
        fig.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()


unit_dict = dict(amplitude='V', amplitude2='V', pulse_length='s',
                 amplitude_offset='V', amplitude_offset2='V',
                 trans_amplitude='V', trans_amplitude2='V',
                 gaussian_filter_sigma='s', trans_length='s')


def get_spectators(dev, anc_data_qb_map, gate_list,
                   include_spectators_which_are_uss = True, include_spectators_which_are_lss = True):
    
    if isinstance(anc_data_qb_map, list):
        return dev.get_qubits(anc_data_qb_map, 'obj')
    gate_list_ancqb = [qb[0].name for qb in gate_list]
    gate_list_dataqb = [qb[1].name for qb in gate_list]
    dd_qubit_list = list(
        np.unique([d for qb in gate_list_ancqb for d in anc_data_qb_map[qb] if
                   d not in gate_list_dataqb]))
    anc_qubit_list = []
    for data_qb in gate_list_dataqb:
        anc_qubit_list += [qb for qb, data_qb_list in anc_data_qb_map.items()
                           if data_qb in data_qb_list]
    anc_qubit_list = list(set([qb for qb in anc_qubit_list if qb not in
                               gate_list_ancqb]))

    qubit_list = []
    if include_spectators_which_are_lss:
        qubit_list += dd_qubit_list
    if include_spectators_which_are_uss:
        qubit_list += anc_qubit_list

    return dev.get_qubits(qubit_list,'obj')


def get_spectator_pulses(dev, spectators, opcode='X90'):
    return [dev.get_operation_dict()[f'{opcode}{"s" if i != 0 else ""} {qb.name}'] for
            i, qb in enumerate(spectators)]


def cal_two_qubit_gates(
        sweep_params, gate_list, dev,
        sweep_range_dict=None, n_cz=1,
        cz_pulse_name='CZ_nztc', phi=None, phi_target=None,
        do_check_msmt=False, only_check_msmt=False,
        nr_phases=8, measure=True, update=True, optimize=False,
        cphase_acq_avg=2**13, chevron_acq_avg=2**16,
        cphase_soft_avg=1, chevron_soft_avg=1,
        acq_weights_type=None, task_kw=None,
        spectator_map=False, include_spec_of_data_qubits=False,
        spectator_pulse_opcode="X90", extra_prepend_pulses=None,
        **kw
):
    optimal_values = {}
    phi_target = phi_target if phi_target is not None else\
        phi if phi is not None else 180
    if extra_prepend_pulses is None:
        extra_prepend_pulses = []
    if task_kw is None:
        task_kw = {}

    # Configure Sweep Params
    default_sweep_range_dict = dict(amplitude=0.01,
                                    amplitude2=0.0025,
                                    pulse_length=1.5e-9,
                                    gaussian_filter_sigma=1.5e-9,
                                    trans_amplitude=0.075,
                                    trans_amplitude2=0.075,
                                    amplitude_offset2=0.002,
                                    amplitude_offset = 0.002)
    if sweep_range_dict is None:
        sweep_range_dict = dict()

    default_sweep_range_dict.update(sweep_range_dict)
    sweep_range_dict = default_sweep_range_dict
    dev.prepare_mwg()
    tmp_vals = []
    if acq_weights_type is not None:
        tmp_vals += [
            (dev.acq_weights_type, acq_weights_type),
        ]
    for qbh, qbl in gate_list:
        if qbh.ge_freq() < qbl.ge_freq():
            print(f'Did you mix up qbH and qbL? {qbh}, {qbl}')

    chevron_mnt_params = ('pulse_length', 'amplitude_offset2',
                          'amplitude_offset')
    mmnts = []
    meas_index = 0
    try_index = 0
    sweep_params = deepcopy(sweep_params)
    if only_check_msmt:
        sweep_params = []
    if do_check_msmt:
        # Will be used to run a CPhase without sweeping any param
        sweep_params.append('do_check')
    while meas_index < len(sweep_params):
        # Param name, e.g. 'amplitude-chevron'
        param = sweep_params[meas_index]
        pulse_params = []  # Real pulse param for each gate
        for i, (qbh, qbl) in enumerate(gate_list):
            pp = param.split('-')[0]
            # Can be different for each gate
            if pp == 'amp_ctrl_param':
                pp = dev.get_pulse_par(
                cz_pulse_name, qbh, qbl, 'amp_ctrl_param')()
            pulse_params.append(pp)
        task_list = []

        if param=='do_check':
            pass
        elif np.ndim(sweep_range_dict[param]) != 0:
            # A tuple was passed: (sweep_range, num_soft_swpts)
            sweep_range, num_soft_swpts = sweep_range_dict[param]
        else:
            # Only sweep_range was passed
            sweep_range = sweep_range_dict[param]
            num_soft_swpts = 11
        if param in chevron_mnt_params or param.endswith('-chevron'):
            experiment_name = f'Chevron_{param}_sweep'

            for i, (qbh, qbl) in enumerate(gate_list):
                sweep_values = dev.get_pulse_par(
                    cz_pulse_name, qbh, qbl, pulse_params[i])() +\
                    np.linspace(-sweep_range, sweep_range, num_soft_swpts)
                sweep_points = sp_mod.SweepPoints(
                    pulse_params[i], sweep_values,
                    unit_dict[pulse_params[i]],
                    dimension=0)

                task = dict(
                    qbc=qbh, qbt=qbl,
                    sweep_points=sweep_points,
                    # qbr=qbl,
                    num_cz_gates=n_cz,
                    cz_pulse_name=cz_pulse_name+('' if phi is None
                                                 else str(phi)),
                )
                if i == 0 and spectator_map:
                    task['prepend_pulse_dicts'] = \
                        get_spectator_pulses(
                            dev,
                            get_spectators(
                                dev,
                                spectator_map,
                                gate_list,
                                include_spec_of_data_qubits=\
                                    include_spec_of_data_qubits
                            ),
                            opcode=spectator_pulse_opcode,
                        )

                if i == 0 and extra_prepend_pulses:
                    print('pushaway')
                    print(extra_prepend_pulses)
                    task['prepend_pulse_dicts'] = extra_prepend_pulses \
                        + task.get('prepend_pulse_dicts', [])
                task.update(task_kw)
                task_list.append(task)
            with temporary_value(
                *tmp_vals,
                (dev.acq_averages, chevron_acq_avg),
                (dev.acq_shots, chevron_acq_avg),
                (dev.instr_mc.get_instr().soft_avg, chevron_soft_avg),
            ):
                mmnt = twoqbcal.Chevron(
                    task_list,
                    dev=dev,
                    cz_pulse_name=cz_pulse_name,
                    cal_states="gef",
                    experiment_name=experiment_name,
                    measure=measure, analyze=False,
                    **kw,
                )
            mmnt.analysis = tda.SingleRowChevronAnalysis()
        else:
            if param=='do_check':
                experiment_name = 'CPhase_measurement_check'
            else:
                experiment_name = f'CPhase_measurement_{param}_sweep'
            for i, (qbh, qbl) in enumerate(gate_list):
                if param=='do_check':
                    sweep_param_dict = {'nothing': {'values': [0]}}
                else:
                    sweep_param_dict = {
                        pulse_params[i]: {
                            'values':
                                dev.get_pulse_par(
                                    cz_pulse_name, qbh, qbl,
                                    pulse_params[i])() + \
                                np.linspace(-sweep_range, sweep_range,
                                            num_soft_swpts),
                            'unit': unit_dict[pulse_params[i]],
                        }
                    }
                task_list.append(dict(
                    qbl=qbh, qbr=qbl,
                    sweep_points=[{}, sweep_param_dict],
                    cz_pulse_name=cz_pulse_name,
                    cphase=phi,
                ))
                if i == 0 and spectator_map:
                    task_list[-1]['prepend_pulse_dicts'] = \
                        get_spectator_pulses(
                            dev,
                            get_spectators(
                                dev,
                                spectator_map,
                                gate_list,
                                include_spec_of_data_qubits=\
                                    include_spec_of_data_qubits
                            ),
                            opcode=spectator_pulse_opcode,
                        )
                if i == 0 and extra_prepend_pulses:
                    print('pushaway')
                    print(extra_prepend_pulses)
                    task_list[-1]['prepend_pulse_dicts'] =\
                        extra_prepend_pulses \
                        + task_list[-1].get('prepend_pulse_dicts', [])
                task_list[-1].update(task_kw)
            with temporary_value(
                *tmp_vals,
                (dev.acq_averages, cphase_acq_avg),
                (dev.acq_shots, cphase_acq_avg),
                (dev.instr_mc.get_instr().soft_avg, cphase_soft_avg),
            ):
                mmnt = twoqbcal.CPhase(
                    task_list=task_list,
                    dev=dev,
                    nr_phases=nr_phases,
                    cz_pulse_name=cz_pulse_name,
                    measure=measure,
                    experiment_name=experiment_name,
                    num_cz_gates=n_cz,
                    ref_pi_half=True,
                    **kw,
                )
        mmnts.append(mmnt)
        if param == 'do_check':
            meas_index += 1
            continue

        # Extract best sweep parameter value
        converged = []
        for i, (qbh, qbl) in enumerate(gate_list):
            if 'CPhase' in experiment_name:
                best_val, c = get_optimal_amp(
                    qbh, qbl, timestamp=None,
                    # parfit=True,
                    analysis_object=mmnt.analysis,
                    phi=phi_target,
                )
                converged.append(c)
            else:
                best_val, c = mmnt.analysis.get_leakage_best_val(
                    qbh.name, qbl.name,
                    minimize='auto',
                )
                converged.append(c)
                mmnt.analysis.plot()
                mmnt.analysis.save_figures()
            print(f'The optimal {param} is: {best_val}')
            optimal_values[param] = best_val
            if update:
                dev.get_pulse_par(cz_pulse_name, qbh, qbl,
                                  pulse_params[i])(best_val)
        if all(converged) or not optimize:
            try_index = 0
            meas_index += 1
        else:
            try_index += 1
            if try_index < 10:
                print("Did not converge! Retrying...")
            else:
                print("Optimization failed. Starting next measurement")
                try_index = 0
                meas_index += 1
    return mmnts


def cal_dyn_phase(
        gate_list, dev,
        update=True, n_cz=1, cz_pulse_name='CZ_nzbasic',
        reset_phases_before_measurement=True, nr_phases=5,
        spectator_map=False, include_spec_of_data_qubits=False,
        spectator_pulse_opcode="X90", extra_prepend_pulses=None,
        acq_avg=2**13,
        **kw):
    if extra_prepend_pulses is None:
        extra_prepend_pulses = []
    task_list = []
    tmp_vals = []
    if acq_avg is not None:
        tmp_vals += [
            (dev.acq_averages, acq_avg),
            (dev.acq_shots, acq_avg),
        ]
    for i, (qbh, qbl) in enumerate(gate_list):
        task_list.append({'op_code': f'CZ {qbh.name} {qbl.name}',
                          'qubits_to_measure': [qbh, qbl],
                          'num_cz_gates': n_cz,
                          })
        # add qubits that have pushaway flux pulses
        flux_channels = {
            qb.flux_pulse_channel(): qb for qb in dev.get_qubits()}
        try:
            add_qubits = [flux_channels[ch] for ch in [
                pd[p] for pd in dev.get_pulse_par(cz_pulse_name, qbh, qbl,
                                                  'aux_pulses_list')()
                for p in pd if 'channel' in p]]
        except ValueError:
            add_qubits = []
        task_list[-1]['qubits_to_measure'] += add_qubits
        if i == 0 and spectator_map:
            task_list[-1]['prepend_pulse_dicts'] = \
                get_spectator_pulses(
                    dev,
                    get_spectators(
                        dev,
                        spectator_map,
                        gate_list,
                        include_spec_of_data_qubits=\
                            include_spec_of_data_qubits
                    ),
                    opcode=spectator_pulse_opcode,
                )
        if i == 0 and extra_prepend_pulses:
            print('pushaway')
            print(extra_prepend_pulses)
            task_list[-1]['prepend_pulse_dicts'] = extra_prepend_pulses \
                                                   + task_list[-1].get(
                'prepend_pulse_dicts', [])
    with temporary_value(*tmp_vals):
        dev.prepare_mwg()
        dynphase_obj = twoqbcal.DynamicPhase(
            task_list,
            dev=dev,
            cz_pulse_name=cz_pulse_name,
            nr_phases=nr_phases,
            reset_phases_before_measurement=reset_phases_before_measurement,
            update=update,
            analyze=True,
            **kw
        )
    if update:
        dynphase_obj.run_update()

    for qbh, qbl in gate_list:
        if update:
            dyn_phase = dynphase_obj.dyn_phases[
                f'{cz_pulse_name} {qbh.name} {qbl.name}']
            dev.get_pulse_par(cz_pulse_name, qbh, qbl, 'basis_rotation')(
                dyn_phase)
            print(
                dev.get_pulse_par(cz_pulse_name, qbh, qbl, 'basis_rotation')())
    return dynphase_obj