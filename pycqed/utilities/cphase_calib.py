from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import timedomain_analysis as tda
import matplotlib.pyplot as plt
import numpy as np
import lmfit

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


def get_optimal_amp(qbc, qbt, soft_sweep_points, timestamp=None,
                    classified_ro=False, tangent_fit=False,
                    parfit=False,
                    analysis_object=None, **kw):

    if analysis_object is None:
        if classified_ro:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in [qbc, qbt]}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_uhf() for vn in
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

    sweep_pts = list(soft_sweep_points.values())[0]['values']
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
                               fit_res=fit_res, save_fig=True, plot_guess=False,
                               qbc_name=qbc.name, qbt_name=qbt.name, **kw)
    return fit_res


def plot_and_save_cz_amp_sweep(cphases, soft_sweep_params_dict, fit_res,
                               qbc_name, qbt_name, save_fig=True, show=True,
                               plot_guess=False, timestamp=None):

    sweep_param_name = list(soft_sweep_params_dict)[0]
    sweep_points = soft_sweep_params_dict[sweep_param_name]['values']
    unit = soft_sweep_params_dict[sweep_param_name]['unit']
    best_val = fit_res.model.func(np.pi, **fit_res.best_values)
    fit_points_init = fit_res.model.func(cphases, **fit_res.init_values)
    fit_points = fit_res.model.func(cphases, **fit_res.best_values)

    fig, ax = plt.subplots()
    ax.plot(cphases*180/np.pi, sweep_points, 'o-')
    ax.plot(cphases*180/np.pi, fit_points, '-r')
    if plot_guess:
        ax.plot(cphases*180/np.pi, fit_points_init, '--k')
    ax.hlines(best_val, cphases[0]*180/np.pi, cphases[-1]*180/np.pi)
    ax.vlines(180, sweep_points.min(), sweep_points.max())
    ax.set_ylabel('Flux pulse {} ({})'.format(sweep_param_name, unit))
    ax.set_xlabel('Conditional phase (rad)')
    ax.set_title('CZ {}-{}'.format(qbc_name, qbt_name))

    ax.text(0.5, 0.95, 'Best {} = {:.6f} ({})'.format(
        sweep_param_name, best_val*1e9 if unit=='s' else best_val, unit),
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
