"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of all
the spectroscopy measurement analyses.
"""

from copy import copy
import logging
import re
import numpy as np
import lmfit
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.timedomain_analysis as tda
import pycqed.analysis.fitting_models as fit_mods
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
import pycqed.analysis.fit_toolbox.geometry as geo
from collections import OrderedDict
from scipy import integrate

log = logging.getLogger(__name__)


class SpectroscopyOld(ba.BaseDataAnalysis):

    def __init__(self, t_start: str = None,
                 t_stop: str = None,
                 options_dict: dict = None,
                 label: str = None,
                 extract_only: bool = False,
                 auto: bool = True,
                 do_fitting: bool = False):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting)
        self.extract_fitparams = self.options_dict.get('fitparams', False)
        self.params_dict = {'freq_label': 'sweep_name',
                            'freq_unit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'freq': 'sweep_points',
                            'amp': 'amp',
                            'phase': 'phase'}
        
        self.options_dict.get('xwidth', None)
        # {'xlabel': 'sweep_name',
        # 'xunit': 'sweep_unit',
        # 'measurementstring': 'measurementstring',
        # 'sweep_points': 'sweep_points',
        # 'value_names': 'value_names',
        # 'value_units': 'value_units',
        # 'measured_values': 'measured_values'}

        if self.extract_fitparams:
            self.params_dict.update({'fitparams': self.options_dict.get('fitparams_key', 'fit_params')})

        self.numeric_params = ['freq', 'amp', 'phase']
        if 'qubit_label' in self.options_dict:
            self.labels.extend(self.options_dict['qubit_label'])
        sweep_param = self.options_dict.get('sweep_param', None)
        if sweep_param is not None:
            self.params_dict.update({'sweep_param': sweep_param})
            self.numeric_params.append('sweep_param')
        if auto is True:
            self.run_analysis()

    def process_data(self):
        proc_data_dict = self.proc_data_dict
        proc_data_dict['freq_label'] = 'Frequency (GHz)'
        proc_data_dict['amp_label'] = 'Transmission amplitude (arb. units)'

        proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        proc_data_dict['freq_range'] = self.options_dict.get(
            'freq_range', None)
        proc_data_dict['amp_range'] = self.options_dict.get('amp_range', None)
        proc_data_dict['phase_range'] = self.options_dict.get(
            'phase_range', None)
        proc_data_dict['plotsize'] = self.options_dict.get('plotsize', (8, 5))

        # FIXME: Nathan : I still don't think using raw_data_dict as a tuple
        #  in case of multi timestamps is a good idea, unless it is also
        #  a tuple of length 1 in the case of 1 timestamp. otherwise we
        #  have to add checks like this one everywhere
        if not isinstance(self.raw_data_dict, (tuple, list)):
            proc_data_dict['plot_frequency'] = np.squeeze(
                self.raw_data_dict['freq'])
            proc_data_dict['plot_amp'] = np.squeeze(self.raw_data_dict['amp'])
            proc_data_dict['plot_phase'] = np.squeeze(
                self.raw_data_dict['phase'])
        else:
            # TRANSPOSE ALSO NEEDS TO BE CODED FOR 2D
            sweep_param = self.options_dict.get('sweep_param', None)
            if sweep_param is not None:
                proc_data_dict['plot_xvals'] = np.array(
                    self.raw_data_dict['sweep_param'])
                proc_data_dict['plot_xvals'] = np.reshape(proc_data_dict['plot_xvals'],
                                                          (len(proc_data_dict['plot_xvals']), 1))
                proc_data_dict['plot_xlabel'] = self.options_dict.get(
                    'xlabel', sweep_param)
            else:

                xvals = np.array([[tt] for tt in range(
                    len(self.raw_data_dict))])
                proc_data_dict['plot_xvals'] = self.options_dict.get(
                    'xvals', xvals)
                proc_data_dict['plot_xlabel'] = self.options_dict.get(
                    'xlabel', 'Scan number')
            proc_data_dict['plot_xwidth'] = self.options_dict.get(
                'xwidth', None)
            if proc_data_dict['plot_xwidth'] == 'auto':
                x_diff = np.diff(np.ravel(proc_data_dict['plot_xvals']))
                dx1 = np.concatenate(([x_diff[0]], x_diff))
                dx2 = np.concatenate((x_diff, [x_diff[-1]]))
                proc_data_dict['plot_xwidth'] = np.minimum(dx1, dx2)
                proc_data_dict['plot_frequency'] = np.array([self.raw_data_dict[i]['hard_sweep_points']
                                                    for i in
                                                    range(len(self.raw_data_dict))])
                proc_data_dict['plot_phase'] = np.array([self.raw_data_dict[i][
                                                    'measured_data']['Phase']
                                                    for i in
                                                    range(len(
                                                        self.raw_data_dict))])
                proc_data_dict['plot_amp'] = np.array([self.raw_data_dict[i][
                                                  'measured_data']['Magn']
                                                    for i in
                                                    range(len(
                                                        self.raw_data_dict))])

            else:
                # manual setting of plot_xwidths
                proc_data_dict['plot_frequency'] = [self.raw_data_dict[i]['hard_sweep_points']
                                                    for i in
                                                    range(len(self.raw_data_dict))]
                proc_data_dict['plot_phase'] = [self.raw_data_dict[i][
                                                    'measured_data']['Phase']
                                                    for i in
                                                    range(len(self.raw_data_dict))]
                proc_data_dict['plot_amp'] = [self.raw_data_dict[i][
                                                  'measured_data']['Magn']
                                                    for i in
                                                    range(len(self.raw_data_dict))]


    def prepare_plots(self):
        proc_data_dict = self.proc_data_dict
        plotsize = self.options_dict.get('plotsize')
        if len(self.raw_data_dict['timestamps']) == 1:
            plot_fn = self.plot_line
            self.plot_dicts['amp'] = {'plotfn': plot_fn,
                                      'xvals': proc_data_dict['plot_frequency'],
                                      'yvals': proc_data_dict['plot_amp'],
                                      'title': 'Spectroscopy amplitude: %s' % (self.timestamps[0]),
                                      'xlabel': proc_data_dict['freq_label'],
                                      'ylabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize
                                      }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                                        'xvals': proc_data_dict['plot_frequency'],
                                        'yvals': proc_data_dict['plot_phase'],
                                        'title': 'Spectroscopy phase: %s' % (self.timestamps[0]),
                                        'xlabel': proc_data_dict['freq_label'],
                                        'ylabel': proc_data_dict['phase_label'],
                                        'yrange': proc_data_dict['phase_range'],
                                        'plotsize': plotsize
                                        }
        else:
            self.plot_dicts['amp'] = {'plotfn': self.plot_colorx,
                                      'xvals': proc_data_dict['plot_xvals'],
                                      'xwidth': proc_data_dict['plot_xwidth'],
                                      'yvals': proc_data_dict['plot_frequency'],
                                      'zvals': proc_data_dict['plot_amp'],
                                      'title': 'Spectroscopy amplitude: %s' % (self.timestamps[0]),
                                      'xlabel': proc_data_dict['plot_xlabel'],
                                      'ylabel': proc_data_dict['freq_label'],
                                      'zlabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['freq_range'],
                                      'zrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize,
                                      'plotcbar': self.options_dict.get('colorbar', False),
                                      }

            self.plot_dicts['amp'] = {'plotfn': self.plot_colorx,
                                      'xvals': proc_data_dict['plot_xvals'],
                                      'yvals': proc_data_dict['plot_frequency'],
                                      'zvals': proc_data_dict['plot_amp'],
                                      }

    def plot_for_presentation(self, key_list=None, no_label=False):
        super().plot_for_presentation(
            key_list=key_list, no_label=no_label)
        for key in key_list:
            pdict = self.plot_dicts[key]
            if key == 'amp':
                if pdict['plotfn'] == self.plot_line:
                    ymin, ymax = 0, 1.2 * np.max(np.ravel(pdict['yvals']))
                    self.axs[key].set_ylim(ymin, ymax)
                    self.axs[key].set_ylabel('Transmission amplitude (V rms)')


class ResonatorSpectroscopy(SpectroscopyOld):
    def __init__(self, t_start,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True):
        super(ResonatorSpectroscopy, self).__init__(t_start, t_stop=t_stop,
                                                    options_dict=options_dict,
                                                    extract_only=extract_only,
                                                    auto=False,
                                                    do_fitting=do_fitting)
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess', {})
        self.simultan = self.options_dict.get('simultan', False)

        if self.simultan:
            if not (len(t_start) == 2 and t_stop is None):
                raise ValueError('Exactly two timestamps need to be passed for'
                             ' simultan resonator spectroscopy in ground '
                             'and excited state as: t_start = [t_on, t_off]')


        if auto is True:
            self.run_analysis()


    def process_data(self):
        super(ResonatorSpectroscopy, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        if len(self.raw_data_dict) == 1:
            self.proc_data_dict['plot_phase'] = np.unwrap(np.pi / 180. *
                              self.proc_data_dict['plot_phase']) * 180 / np.pi
            self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        else:
            pass

    def prepare_fitting(self):
        super().prepare_fitting()
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex, hanger_with_pf.
        fit_options = self.options_dict.get('fit_options', None)
        subtract_background = self.options_dict.get(
            'subtract_background', False)
        if fit_options is None:
            fitting_model = 'hanger'
        else:
            fitting_model = fit_options['model']
        if subtract_background:
            self.do_subtract_background(thres=self.options_dict['background_thres'],
                                        back_dict=self.options_dict['background_dict'])

        if fitting_model == 'hanger':
            fit_fn = fit_mods.SlopedHangerFuncAmplitude
            fit_guess_fn = fit_mods.SlopedHangerFuncAmplitudeGuess
            guess_pars = None
        elif fitting_model == 'simple_hanger':
            fit_fn = fit_mods.HangerFuncAmplitude
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            # TODO HangerFuncAmplitude Guess
        elif fitting_model == 'lorentzian':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.Lorentzian
            # TODO LorentzianGuess
        elif fitting_model == 'complex':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.HangerFuncComplex
            # TODO HangerFuncComplexGuess
        elif fitting_model == 'hanger_with_pf':
            if self.simultan:
                fit_fn = fit_mods.simultan_hanger_with_pf
                self.sim_fit = fit_mods.fit_hanger_with_pf(
                            fit_mods.SimHangerWithPfModel,[
                            np.transpose([self.proc_data_dict['plot_frequency'][0],
                                          self.proc_data_dict['plot_amp'][0]]),
                            np.transpose([self.proc_data_dict['plot_frequency'][1],
                                          self.proc_data_dict['plot_amp'][1]])],
                            simultan=True)
                guess_pars = None
                fit_guess_fn = None
                x_fit_0 = self.proc_data_dict['plot_frequency'][0]

                self.chi = (self.sim_fit[1].params['omega_ro'].value -
                            self.sim_fit[0].params['omega_ro'].value)/2
                self.f_RO_res = (self.sim_fit[0].params['omega_ro'].value+
                                 self.sim_fit[1].params['omega_ro'].value)/2
                self.f_PF = self.sim_fit[0].params['omega_pf'].value
                self.kappa = self.sim_fit[0].params['kappa_pf'].value
                self.J_ = self.sim_fit[0].params['J'].value

               
            else:
                fit_fn = fit_mods.hanger_with_pf
                fit_temp = fit_mods.fit_hanger_with_pf(
                            fit_mods.HangerWithPfModel,
                            np.transpose([self.proc_data_dict['plot_frequency'],
                                          self.proc_data_dict['plot_amp']]))
                guess_pars = fit_temp.params
                self.proc_data_dict['fit_params'] = fit_temp.params
                fit_guess_fn = None

        if (len(self.raw_data_dict['timestamps']) == 1) or self.simultan:
            self.fit_dicts['reso_fit'] = {
                              'fit_fn': fit_fn,
                              'fit_guess_fn': fit_guess_fn,
                              'guess_pars': guess_pars,
                              'fit_yvals': {
                                  'data': self.proc_data_dict['plot_amp']
                                           },
                              'fit_xvals': {
                                  'f': self.proc_data_dict['plot_frequency']}
                                         }
        else:
            self.fit_dicts['reso_fit'] = {
                              'fit_fn': fit_fn,
                              'fit_guess_fn': fit_guess_fn,
                              'guess_pars': guess_pars,
                              'fit_yvals': [{'data': np.squeeze(tt)}
                                               for tt in self.plot_amp],
                              'fit_xvals': np.squeeze([{'f': tt[0]}
                                               for tt in self.plot_frequency])}

    def run_fitting(self):
        if not self.simultan:
            super().run_fitting()

    def do_subtract_background(self, thres=None, back_dict=None, ):
        if len(self.raw_data_dict['timestamps']) == 1:
            pass
        else:
            x_filtered = []
            y_filtered = []
            for tt in range(len(self.raw_data_dict['timestamps'])):
                y = np.squeeze(self.plot_amp[tt])
                x = np.squeeze(self.plot_frequency)[tt]
                guess_dict = SlopedHangerFuncAmplitudeGuess(y, x)
                Q = guess_dict['Q']['value']
                f0 = guess_dict['f0']['value']
                df = 2 * f0 / Q
                fmin = f0 - df
                fmax = f0 + df
                indices = np.logical_or(x < fmin * 1e9, x > fmax * 1e9)
                
                x_filtered.append(x[indices])
                y_filtered.append(y[indices])
            self.background = pd.concat([pd.Series(y_filtered[tt], index=x_filtered[tt])
                                         for tt in range(len(self.raw_data_dict['timestamps']))], axis=1).mean(axis=1)
            background_vals = self.background.reset_index().values
            freq = background_vals[:, 0]
            amp = background_vals[:, 1]
            # thres = 0.0065
            indices = amp < thres
            freq = freq[indices] * 1e-9
            amp = amp[indices]
            fit_fn = double_cos_linear_offset
            model = lmfit.Model(fit_fn)
            fit_yvals = amp
            fit_xvals = {'t': freq}
            for key, val in list(back_dict.items()):
                model.set_param_hint(key, **val)
            params = model.make_params()
            fit_res = model.fit(fit_yvals,
                                params=params,
                                **fit_xvals)
            self.background_fit = fit_res

            for tt in range(len(self.raw_data_dict['timestamps'])):
                divide_vals = fit_fn(np.squeeze(self.plot_frequency)[tt] * 1e-9, **fit_res.best_values)
                self.plot_amp[tt] = np.array(
                    [np.array([np.divide(np.squeeze(self.plot_amp[tt]), divide_vals)])]).transpose()

    def prepare_plots(self):
        if not self.simultan:
            super(ResonatorSpectroscopy, self).prepare_plots()
        else:
            proc_data_dict = self.proc_data_dict
            plotsize = self.options_dict.get('plotsize')
            plot_fn = self.plot_line
            amp_diff = np.abs(proc_data_dict['plot_amp'][0]*np.exp(
                                  1j*np.pi*proc_data_dict['plot_phase'][0]/180)-
                              proc_data_dict['plot_amp'][1]*np.exp(
                                  1j*np.pi*proc_data_dict['plot_phase'][1]/180))
            # FIXME: Nathan 2019.05.08 I don't think this is the right place to adapt
            #  the ro fequency (i.e. in prepare_plot)... I had a hard time finding
            #  where it happened !
            self.f_RO = proc_data_dict['plot_frequency'][0][np.argmax(amp_diff)]
            self.plot_dicts['amp1'] = {'plotfn': plot_fn,
                                      'ax_id': 'amp',
                                      'xvals': proc_data_dict['plot_frequency'][0],
                                      'yvals': proc_data_dict['plot_amp'][0],
                                      'title': 'Spectroscopy amplitude: \n'
                                               '%s-%s' % (
                                          self.raw_data_dict[0][
                                              'measurementstring'],
                                          self.timestamps[0]),
                                      'xlabel': proc_data_dict['freq_label'],
                                      'xunit': 'Hz',
                                      'ylabel': proc_data_dict['amp_label'],
                                      'yrange': proc_data_dict['amp_range'],
                                      'plotsize': plotsize,
                                      'color': 'b',
                                      'linestyle': '',
                                      'marker': 'o',
                                      'setlabel': '|g> data',
                                      'do_legend': True
                                       }
            self.plot_dicts['amp2'] = {'plotfn': plot_fn,
                                       'ax_id': 'amp',
                                       'xvals': proc_data_dict['plot_frequency'][1],
                                       'yvals': proc_data_dict['plot_amp'][1],
                                       'color': 'r',
                                       'linestyle': '',
                                       'marker': 'o',
                                       'setlabel': '|e> data',
                                       'do_legend': True
                                       }
            self.plot_dicts['diff'] = {'plotfn': plot_fn,
                                       'ax_id': 'amp',
                                       'xvals': proc_data_dict['plot_frequency'][0],
                                       'yvals': amp_diff,
                                       'color': 'g',
                                       'linestyle': '',
                                       'marker': 'o',
                                       'setlabel': 'diff',
                                       'do_legend': True
                                       }
            self.plot_dicts['phase'] = {'plotfn': plot_fn,
                                        'xvals': proc_data_dict['plot_frequency'],
                                        'yvals': proc_data_dict['plot_phase'],
                                        'title': 'Spectroscopy phase: '
                                                 '%s' % (self.timestamps[0]),
                                        'xlabel': proc_data_dict['freq_label'],
                                        'ylabel': proc_data_dict['phase_label'],
                                        'yrange': proc_data_dict['phase_range'],
                                        'plotsize': plotsize
                                        }


    def plot_fitting(self):
        if self.do_fitting:
            fit_options = self.options_dict.get('fit_options', None)
            if fit_options is None:
                fitting_model = 'hanger'
            else:
                fitting_model = fit_options['model']
            for key, fit_dict in self.fit_dicts.items():
                if not self.simultan:
                    fit_results = fit_dict['fit_res']
                else:
                    fit_results = self.sim_fit
                ax = self.axs['amp']
                if len(self.raw_data_dict['timestamps']) == 1 or self.simultan:
                    if fitting_model == 'hanger':
                        ax.plot(list(fit_dict['fit_xvals'].values())[0],
                                fit_results.best_fit, 'r-', linewidth=1.5)
                        textstr = 'f0 = %.5f $\pm$ %.1g GHz' % (
                              fit_results.params['f0'].value,
                              fit_results.params['f0'].stderr) + '\n' \
                                           'Q = %.4g $\pm$ %.0g' % (
                              fit_results.params['Q'].value,
                              fit_results.params['Q'].stderr) + '\n' \
                                           'Qc = %.4g $\pm$ %.0g' % (
                              fit_results.params['Qc'].value,
                              fit_results.params['Qc'].stderr) + '\n' \
                                           'Qi = %.4g $\pm$ %.0g' % (
                              fit_results.params['Qi'].value,
                              fit_results.params['Qi'].stderr)
                        box_props = dict(boxstyle='Square',
                                         facecolor='white', alpha=0.8)
                        self.box_props = {key: val for key,
                                                       val in box_props.items()}
                        self.box_props.update({'linewidth': 0})
                        self.box_props['alpha'] = 0.
                        ax.text(0.03, 0.95, textstr, transform=ax.transAxes,
                                verticalalignment='top', bbox=self.box_props)
                    elif fitting_model == 'simple_hanger':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'lorentzian':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'complex':
                        raise NotImplementedError(
                            'This functions guess function is not coded up yet')
                    elif fitting_model == 'hanger_with_pf':
                        if not self.simultan:
                            ax.plot(list(fit_dict['fit_xvals'].values())[0],
                                    fit_results.best_fit, 'r-', linewidth=1.5)

                            par = ["%.3f" %(fit_results.params['omega_ro'].value*1e-9),
                                   "%.3f" %(fit_results.params['omega_pf'].value*1e-9),
                                   "%.3f" %(fit_results.params['kappa_pf'].value*1e-6),
                                   "%.3f" %(fit_results.params['J'].value*1e-6),
                                   "%.3f" %(fit_results.params['gamma_ro'].value*1e-6)]
                            textstr = str('f_ro = '+par[0]+' GHz'
                                      +'\n\nf_pf = '+par[1]+' GHz'
                                      +'\n\nkappa = '+par[2]+' MHz'
                                      +'\n\nJ = '+par[3]+' MHz'
                                      +'\n\ngamma_ro = '+par[4]+' MHz')
                            ax.plot([0],
                                    [0],
                                    'w',
                                    label=textstr)
                        else:
                            x_fit_0 = np.linspace(min(
                                self.proc_data_dict['plot_frequency'][0][0],
                                self.proc_data_dict['plot_frequency'][1][0]),
                                max(self.proc_data_dict['plot_frequency'][0][-1],
                                    self.proc_data_dict['plot_frequency'][1][-1]),
                                len(self.proc_data_dict['plot_frequency'][0]))
                            x_fit_1 = np.linspace(min(
                                self.proc_data_dict['plot_frequency'][0][0],
                                self.proc_data_dict['plot_frequency'][1][0]),
                                max(self.proc_data_dict['plot_frequency'][0][-1],
                                    self.proc_data_dict['plot_frequency'][1][-1]),
                                len(self.proc_data_dict['plot_frequency'][1]))

                            ax.plot(x_fit_0,
                                    fit_results[0].eval(
                                        fit_results[0].params,
                                        f=x_fit_0),
                                    'b--', linewidth=1.5, label='|g> fit')
                            ax.plot(x_fit_1,
                                    fit_results[1].eval(
                                        fit_results[1].params,
                                        f=x_fit_1),
                                    'r--', linewidth=1.5, label='|e> fit')
                            f_RO = self.f_RO
                            ax.plot([f_RO, f_RO],
                                    [0,max(max(self.raw_data_dict['amp'][0]),
                                           max(self.raw_data_dict['amp'][1]))],
                                    'k--', linewidth=1.5)

                            par = ["%.3f" %(fit_results[0].params['gamma_ro'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['omega_pf'].value*1e-9),
                                   "%.3f" %(fit_results[0].params['kappa_pf'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['J'].value*1e-6),
                                   "%.3f" %(fit_results[0].params['omega_ro'].value*1e-9),
                                   "%.3f" %(fit_results[1].params['omega_ro'].value*1e-9),
                                   "%.3f" %((fit_results[1].params['omega_ro'].value-
                                             fit_results[0].params['omega_ro'].value)
                                             /2*1e-6)]
                            textstr = str('\n\nkappa = '+par[2]+' MHz'
                                          +'\n\nJ = '+par[3]+' MHz'
                                          +'\n\nchi = '+par[6]+' MHz'
                                          +'\n\nf_pf = '+par[1]+' GHz'
                                          +'\n\nf_rr |g> = '+par[4]+' GHz'
                                          +'\n\nf_rr |e> = '+par[5]+' GHz'
                                          +'\n\nf_RO = '+"%.3f" %(f_RO*1e-9)+''
                                          ' GHz'
                                         )
                            ax.plot([f_RO],
                                    [0],
                                    'w--', label=textstr)
                        # box_props = dict(boxstyle='Square',
                        #                  facecolor='white', alpha=0.8)
                        # self.box_props = {key: val for key,
                        #                                val in box_props.items()}
                        # self.box_props.update({'linewidth': 0})
                        # self.box_props['alpha'] = 0.
                        #
                        ax.legend(loc='upper left', bbox_to_anchor=[1, 1])

                else:
                    reso_freqs = [fit_results[tt].params['f0'].value *
                                  1e9 for tt in range(len(self.raw_data_dict['timestamps']))]
                    ax.plot(np.squeeze(self.plot_xvals),
                            reso_freqs,
                            'o',
                            color='m',
                            markersize=3)

    def plot(self, key_list=None, axs_dict=None, presentation_mode=None, no_label=False):
        super(ResonatorSpectroscopy, self).plot(key_list=key_list,
                                                axs_dict=axs_dict,
                                                presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()


class ResonatorSpectroscopy_v2(SpectroscopyOld):
    def __init__(self, t_start=None,
                 options_dict=None,
                 t_stop=None,
                 do_fitting=False,
                 extract_only=False,
                 auto=True, **kw):
        """
        FIXME: Nathan: the dependency on the # of timestamps is carried
         through the entire class and is horrible. We should loop and make fits
         separately, instead of using the simultan parameter.
         It would be much simpler!
        Args:
            t_start:
            options_dict:
                ref_state: reference state timestamp when comparing several
                    spectra. Most of the time it will be timestamp of ground
                    state.
                # TODO: Nathan: merge with fit_options (?)
                qutrit_fit_options: dict with options for qutrit RO frequency
                    fitting.
                        sigma_init: initial noise standard deviation assumed
                            for distribution of point in IQ plane. Assumed to
                            be large and algorithm will reduce it.
                        target_fidelity: target fidelity
                        max_width_at_max_fid: maximum width (in Hz) when
                        searching for appropriate sigma
            t_stop:
            do_fitting:
            extract_only:
            auto:
        """
        super(ResonatorSpectroscopy_v2, self).__init__(t_start=t_start,
                                                       t_stop=t_stop,
                                                    options_dict=options_dict,
                                                    extract_only=extract_only,
                                                    auto=False,
                                                    do_fitting=do_fitting,
                                                       **kw)
        self.do_fitting = do_fitting
        self.fitparams_guess = self.options_dict.get('fitparams_guess', {})

        if auto is True:
            self.run_analysis()

    def process_data(self):
        super(ResonatorSpectroscopy_v2, self).process_data()
        self.proc_data_dict['amp_label'] = 'Transmission amplitude (V rms)'
        self.proc_data_dict['phase_label'] = 'Transmission phase (degrees)'
        # now assumes the raw data dict is a tuple due to aa1ed4cdf546
        n_spectra = len(self.raw_data_dict)
        self.proc_data_dict['plot_xlabel'] = 'Readout Frequency (Hz)'
        if self.options_dict.get('ref_state', None) is None:
            default_ref_state = 'g'
            message = "Analyzing spectra of {} states but no ref_state " \
                      "was passed. Assuming timestamp[0]: {} is the " \
                      "timestamp of reference state with label {}"
            log.warning(
                message.format(n_spectra, self.raw_data_dict[0]['timestamp'],
                               default_ref_state))
            self.ref_state = default_ref_state
        else:
            self.ref_state = self.options_dict['ref_state']

        spectra_mapping = \
            self.options_dict.get("spectra_mapping",
                                  self._default_spectra_mapping())

        spectra = {state: self.raw_data_dict[i]["measured_data"]['Magn'] *
                      np.exp(1j * np.pi *
                           self.raw_data_dict[i]["measured_data"]['Phase'] / 180.)
                    for i, state in enumerate(spectra_mapping.keys())}

        iq_distance = {state + self.ref_state:
                           np.abs(spectra[state] - spectra[self.ref_state])
                       for state in spectra_mapping.keys()
                       if state != self.ref_state}
        for state_i in spectra_mapping:
            for state_j in spectra_mapping:
                if not state_i + state_j  in iq_distance and \
                        state_i != state_j:
                    # both ij and ji will have entries which will have
                    # the same values but this is not a problem per se.
                    iq_distance[state_i + state_j] = \
                        np.abs(spectra[state_i] - spectra[state_j])

        self.proc_data_dict["spectra_mapping"] = spectra_mapping
        self.proc_data_dict["spectra"] = spectra
        self.proc_data_dict["iq_distance"] = iq_distance
        self.proc_data_dict["fit_raw_results"] = OrderedDict()

    def _default_spectra_mapping(self):
        default_levels_order = ('g', 'e', 'f')
        # assumes raw_data_dict is tuple
        tts = [d['timestamp'] for d in self.raw_data_dict]
        spectra_mapping = {default_levels_order[i]: tt
                           for i, tt in enumerate(tts)}
        msg = "Assuming following mapping templates of spectra: {}." \
              "\nspectra_mapping can be used in options_dict to modify" \
              "this behavior."
        log.warning(msg.format(spectra_mapping))
        return spectra_mapping


    def prepare_fitting(self):
        super().prepare_fitting()
        # Fitting function for one data trace. The fitted data can be
        # either complex, amp(litude) or phase. The fitting models are
        # HangerFuncAmplitude, HangerFuncComplex,
        # PolyBgHangerFuncAmplitude, SlopedHangerFuncAmplitude,
        # SlopedHangerFuncComplex, hanger_with_pf.
        fit_options = self.options_dict.get('fit_options', dict())
        subtract_background = \
            self.options_dict.get('subtract_background', False)
        fitting_model = fit_options.get('model', 'hanger')
        self.proc_data_dict['fit_results'] = OrderedDict()
        self.fit_res = dict()
        if subtract_background:
            log.warning("Substract background might not work and has "
                            "not been tested.")
            self.do_subtract_background(
                thres=self.options_dict['background_thres'],
                back_dict=self.options_dict['background_dict'])

        if fitting_model == 'hanger':
            fit_fn = fit_mods.SlopedHangerFuncAmplitude
            fit_guess_fn = fit_mods.SlopedHangerFuncAmplitudeGuess
            guess_pars = None
        elif fitting_model == 'simple_hanger':
            fit_fn = fit_mods.HangerFuncAmplitude
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            # TODO HangerFuncAmplitude Guess
        elif fitting_model == 'lorentzian':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.Lorentzian
            # TODO LorentzianGuess
        elif fitting_model == 'complex':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
            fit_fn = fit_mods.HangerFuncComplex
            # TODO HangerFuncComplexGuess
        elif fitting_model == 'hanger_with_pf':
            if not isinstance(self.raw_data_dict, tuple):
                # single fit
                fit_fn = fit_mods.hanger_with_pf
                fit_temp = fit_mods.fit_hanger_with_pf(
                    fit_mods.HangerWithPfModel,
                    np.transpose([self.proc_data_dict['plot_frequency'],
                                  self.proc_data_dict['plot_amp']]))
                guess_pars = fit_temp.params
                self.proc_data_dict['fit_params'] = fit_temp.params
                self.proc_data_dict['fit_raw_results'][self.ref_state] = \
                    fit_temp.params
                fit_guess_fn = None
            else:
                pass
                # comparative fit to reference state
                # FIXME: Nathan: I guess here only fit dicts should be created
                #  and then passed to run_fitting() of basis class but this is
        #         #  not done here. Instead, fitting seems to be done here.
        #         ref_spectrum = self.proc_data_dict['spectra'][self.ref_state]
        #         for state, spectrum in self.proc_data_dict['spectra'].items():
        #             if state == self.ref_state:
        #                 continue
        #             key = self.ref_state + state
        #             fit_fn = fit_mods.simultan_hanger_with_pf
        #             fit_results = fit_mods.fit_hanger_with_pf(
        #                 fit_mods.SimHangerWithPfModel, [
        #                     np.transpose(
        #                         [self.proc_data_dict['plot_frequency'][0],
        #                          np.abs(ref_spectrum)]),
        #                     np.transpose(
        #                         [self.proc_data_dict['plot_frequency'][0],
        #                          np.abs(spectrum)])],
        #                 simultan=True)
        #             self.proc_data_dict['fit_raw_results'][key] = fit_results
        #             guess_pars = None
        #             fit_guess_fn = None
        #
        #             chi = (fit_results[1].params['omega_ro'].value -
        #                         fit_results[0].params['omega_ro'].value) / 2
        #             f_RO_res = (fit_results[0].params['omega_ro'].value +
        #                              fit_results[1].params['omega_ro'].value) / 2
        #             f_PF = fit_results[0].params['omega_pf'].value
        #             kappa = fit_results[0].params['kappa_pf'].value
        #             J_ = fit_results[0].params['J'].value
        #             f_RO = self.find_f_RO([self.ref_state, state])
        #             self.fit_res[key] = \
        #                 dict(chi=chi, f_RO_res=f_RO_res, f_PF=f_PF,
        #                      kappa=kappa, J_=J_, f_RO=f_RO)
        #
        # if not isinstance(self.raw_data_dict, tuple ):
        #     self.fit_dicts['reso_fit'] = {
        #         'fit_fn': fit_fn,
        #         'fit_guess_fn': fit_guess_fn,
        #         'guess_pars': guess_pars,
        #         'fit_yvals': {'data': self.proc_data_dict['plot_amp']},
        #         'fit_xvals': { 'f': self.proc_data_dict['plot_frequency']}}

    def find_f_RO(self, states):
        """
        Finds the best readout frequency of the list of states.
            If one state is passed, the resonator frequency is returned.
            If two states are passed, the frequency with maximal difference
        between the two states in IQ plane is returned (optimal qubit nRO freq).
            If three states are passed, optimal frequency is found by finding
            the highest variance allowing a target fidelity to be reached on a
            narrow frequency interval. (optimal qutrit RO_freq)
        Args:
            states: list of states between which readout frequency
                should be found

        Returns:

        """
        key = "".join(states)
        if len(states) == 1:
            f_RO = self.proc_data_dict['plot_frequency'][0][
            np.argmax(self.proc_data_dict['spectra'][key])]
        elif len(states) == 2:
            f_RO = self.proc_data_dict['plot_frequency'][0][
                np.argmax(self.proc_data_dict['iq_distance'][key])]
        elif len(states) == 3:
            f_RO, raw_results = self._find_f_RO_qutrit(
                self.proc_data_dict['spectra'],
                self.proc_data_dict['plot_frequency'][0],
                **self.options_dict.get('qutrit_fit_options', dict()))
            self.proc_data_dict["fit_raw_results"][key] = raw_results
        else:
            raise ValueError("{} states were given but method expects 1, "
                             "2 or 3 states.")
        return f_RO

    @staticmethod
    def _find_f_RO_qutrit(spectra, freqs, sigma_init=0.01,
                          return_full=True, **kw):
        n_iter = 0
        avg_fidelities = OrderedDict()
        single_level_fidelities = OrderedDict()
        optimal_frequency = []
        sigmas = [sigma_init]

        log.debug("###### Starting Analysis to find qutrit f_RO ######")

        while ResonatorSpectroscopy_v2.update_sigma(avg_fidelities, sigmas, freqs,
                                optimal_frequency, n_iter, **kw):
            log.debug("Iteration {}".format(n_iter))
            sigma = sigmas[-1]
            if sigma in avg_fidelities.keys():
                continue
            else:
                avg_fidelity, single_level_fidelity = \
                    ResonatorSpectroscopy_v2.three_gaussians_overlap(spectra, sigma)
                avg_fidelities[sigma] = avg_fidelity
                single_level_fidelities[sigma] = single_level_fidelity
                n_iter += 1
        raw_results = dict(avg_fidelities=avg_fidelities,
                           single_level_fidelities=single_level_fidelities,
                           sigmas=sigmas, optimal_frequency=optimal_frequency)
        qutrit_key = "".join(list(spectra))


        log.debug("###### Finished Analysis. Optimal f_RO: {} ######"
                      .format(optimal_frequency[-1]))

        return optimal_frequency[-1], raw_results if return_full else \
            optimal_frequency[-1]

    @staticmethod
    def update_sigma(avg_fidelities, sigmas, freqs,
                     optimal_frequency, n_iter, n_iter_max=20,
                     target_fidelity=0.99, max_width_at_max_fid=0.2e6, **kw):
        continue_search = True
        if n_iter >= n_iter_max:
            log.warning("Could not converge to a proper RO frequency" \
                  "within {} iterations. Returning best frequency found so far. "
                  "Consider changing log_bounds".format(n_iter_max))
            continue_search = False
        elif len(avg_fidelities.keys()) == 0:
            # search has not started yet
            continue_search = True
        else:
            delta_freq = freqs[1] - freqs[0]
            if max_width_at_max_fid < delta_freq:
                msg = "max_width_at_max_fid cannot be smaller than the " \
                      "difference between two frequency data points.\n" \
                      "max_width_at_max_fid: {}\nDelta freq: {}"
                log.warning(msg.format(max_width_at_max_fid, delta_freq))
                max_width_at_max_fid = delta_freq

            sigma_current = sigmas[-1]
            fid, idx_width = ResonatorSpectroscopy_v2.fidelity_and_width(
                avg_fidelities[sigma_current], target_fidelity)
            width = idx_width * delta_freq
            log.debug("sigmas " + str(sigmas) + " width (MHz): "
                          + str(width / 1e6))
            f_opt = freqs[np.argmax(avg_fidelities[sigma_current])]
            optimal_frequency.append(f_opt)

            if len(sigmas) == 1:
                sigma_previous = 10 ** (np.log10(sigma_current) + 1)
            else:
                sigma_previous = sigmas[-2]
            log_diff = np.log10(sigma_previous) - np.log10(sigma_current)

            if fid >= target_fidelity and width <= max_width_at_max_fid:
                # succeeded
                continue_search = False
            elif fid >= target_fidelity and width > max_width_at_max_fid:
                # sigma is too small, update lower bound
                if log_diff < 0:
                    sigma_new = \
                        10 ** (np.log10(sigma_current) - np.abs(log_diff) / 2)
                else:
                    sigma_new = \
                        10 ** (np.log10(sigma_current) + np.abs(log_diff))
                msg = "Width > max_width, update sigma to: {}"
                log.debug(msg.format(sigma_new))
                sigmas.append(sigma_new)
            elif fid < target_fidelity:
                # sigma is too high, update higher bound
                if np.all(np.diff(sigmas) < 0):
                    sigma_new = 10 ** (np.log10(sigma_current) - log_diff)
                else:
                    sigma_new = 10 ** (np.log10(sigma_current) -
                                       np.abs(log_diff) / 2)
                msg = "Fidelity < target fidelity, update sigma to: {}"
                log.debug(msg.format(sigma_new))
                sigmas.append(sigma_new)

        return continue_search

    @staticmethod
    def fidelity_and_width(avg_fidelity, target_fidelity):
        avg_fidelity = np.array(avg_fidelity)
        max_fid = np.max(avg_fidelity)
        idx_width = np.sum(
            (avg_fidelity >= target_fidelity) * (avg_fidelity <= 1.))
        return max_fid, idx_width

    @staticmethod
    def _restricted_angle(angle):
        entire_div = angle // (np.sign(angle) * np.pi)
        return angle - np.sign(angle) * entire_div * 2 * np.pi

    @staticmethod
    def three_gaussians_overlap(spectrums, sigma):
        """
        Evaluates the overlap of 3 gaussian distributions for each complex
        point given in spectrums.
        Args:
            spectrums: dict with resonnator response of each state
            sigma: standard deviation of gaussians used for computing overlap

        Returns:

        """
        def g(x, d, sigma=0.1):
            x = ResonatorSpectroscopy_v2._restricted_angle(x)
            return np.exp(-d ** 2 / np.cos(x) ** 2 / (2 * sigma ** 2))

        def f(gamma, val1=0, val2=1 / (2 * np.pi)):
            gamma = ResonatorSpectroscopy_v2._restricted_angle(gamma)
            return val1 if gamma > -np.pi / 2 and gamma < np.pi / 2 else val2

        def integral(angle, dist, sigma):
            const = 1 / (2 * np.pi)
            p1 = const * \
                 integrate.quad(lambda x: f(x, g(x, dist, sigma=sigma), 0),
                                angle - np.pi,
                                angle)[0]
            return -p1 + integrate.quad(lambda x: f(x),
                                        angle - np.pi, angle)[0] + \
                         integrate.quad(lambda x: f(x, 1 / (2 * np.pi), 0),
                                        angle - np.pi,
                                        angle)[0]

        assert len(spectrums) == 3, "3 spectrums required for qutrit F_RO " \
                                  "analysis. Found {}".format((len(spectrums)))
        i1s, i2s, i3s = [], [], []
        # in most cases, states will be ['g', 'e', 'f'] but to ensure not to
        # be dependent on labels we take indices of keys
        states = list(spectrums.keys())
        for i in range(len(spectrums[states[0]])):
            pt1 = (spectrums[states[0]][i].real, spectrums[states[0]][i].imag)
            pt2 = (spectrums[states[1]][i].real, spectrums[states[1]][i].imag)
            pt3 = (spectrums[states[2]][i].real, spectrums[states[2]][i].imag)
            d1 = geo.distance(pt1, pt2) / 2
            d2 = geo.distance(pt2, pt3) / 2
            d3 = geo.distance(pt1, pt3) / 2
            # translate to point1
            pt2 = tuple(np.asarray(pt2) - np.asarray(pt1))
            pt3 = tuple(np.asarray(pt3) - np.asarray(pt1))
            pt1 = (0., 0.)
            c, R = geo.circumcenter(pt2, pt3, pt1, show=False)
            gamma1 = np.arccos(d1 / R)
            gamma2 = np.arccos(d2 / R)
            gamma3 = np.arccos(d3 / R)
            i1 = integral(gamma1, d1, sigma)
            i2 = integral(gamma2, d2, sigma)
            i3 = integral(gamma3, d3, sigma)
            i1s.append(i1)
            i2s.append(i2)
            i3s.append(i3)

        i1s, i2s, i3s = np.array(i1s), np.array(i2s), np.array(i3s)
        total_area = 2 * i1s + 2 * i2s + 2 * i3s
        avg_fidelity = total_area / 3
        fid_state_0 = i1s + i3s
        not0 = 1 - fid_state_0
        fid_state_1 = i1s + i2s
        not1 = 1 - i1s + i2s
        fid_state_2 = i2s + i3s
        not2 = 1 - fid_state_2

        single_level_fid = {states[0]: fid_state_0,
                            states[1]: fid_state_1,
                            states[2]: fid_state_2}

        return avg_fidelity, single_level_fid

    def run_fitting(self):
        # FIXME: Nathan: for now this is left as written previously but
        #  ultimately all fitting should be done in base class if possible
        states = list(self.proc_data_dict['spectra'])
        if len(states) == 1:
            super().run_fitting()

        if len(states) == 3:
            f_RO_qutrit =  self.find_f_RO(states)
            self.fit_res["".join(states)] = dict(f_RO=f_RO_qutrit)


    def prepare_plots(self):
        self.get_default_plot_params(set_pars=True)
        proc_data_dict = self.proc_data_dict
        spectra = proc_data_dict['spectra']
        plotsize = self.options_dict.get('plotsize')
        plot_fn = self.plot_line
        for i, (state, spectrum) in enumerate(spectra.items()):
            all_freqs = proc_data_dict['plot_frequency']
            freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
            self.plot_dicts['amp_{}'
                .format(state)] = {
                'plotfn': plot_fn,
                'ax_id': 'amp',
                'xvals': freqs,
                'yvals': np.abs(spectrum),
                'title': 'Spectroscopy amplitude: \n'
                        '%s-%s' % (
                            self.raw_data_dict[i]['measurementstring'],
                            self.raw_data_dict[i]['timestamp']),
                'xlabel': proc_data_dict['freq_label'],
                'xunit': 'Hz',
                'ylabel': proc_data_dict['amp_label'],
                'yrange': proc_data_dict['amp_range'],
                'plotsize': plotsize,
                # 'color': 'b',
                'linestyle': '',
                'marker': 'o',
                'setlabel': '$|{}\\rangle$'.format(state),
                'do_legend': True }
            if state != self.ref_state and len(spectra) == 2.:
                # if comparing two stattes we are interested in the
                # difference between the two responses
                label = "iq_distance_{}{}".format(state, self.ref_state)
                self.plot_dicts[label] = {
                    'plotfn': plot_fn,
                    'ax_id': 'amp',
                    'xvals': proc_data_dict['plot_frequency'][0],
                    'yvals': proc_data_dict['iq_distance'][
                        state + self.ref_state],
                    #'color': 'g',
                    'linestyle': '',
                    'marker': 'o',
                    'markersize': 5,
                    'setlabel': label,
                    'do_legend': True}
            fig = self.plot_difference_iq_plane()
            self.figs['difference_iq_plane'] = fig
            fig2 = self.plot_gaussian_overlap()
            self.figs['gaussian_overlap'] = fig2
            fig3 = self.plot_max_area()
            self.figs['area_in_iq_plane'] = fig3

    def plot_fitting(self):
        fit_options = self.options_dict.get('fit_options', None)
        if fit_options is None:
            fitting_model = 'hanger'
        else:
            fitting_model = fit_options['model']

        if not isinstance(self.raw_data_dict, tuple):
            fit_results = self.fit_dict['fit_res']
        else:
            fit_results = self.proc_data_dict['fit_raw_results']
        ax = self.axs['amp']
        if fitting_model == 'hanger':
            raise NotImplementedError(
                'Plotting hanger is not supported in this class.')
        elif fitting_model == 'simple_hanger':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'lorentzian':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'complex':
            raise NotImplementedError(
                'This functions guess function is not coded up yet')
        elif fitting_model == 'hanger_with_pf':
            label = "$|{}\\rangle$ {}"
            all_freqs = self.proc_data_dict['plot_frequency']
            freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
            for state, spectrum in self.proc_data_dict['spectra'].items():
                if len(self.proc_data_dict['spectra']) == 1:
                    # then also add single fit parameters to the legend
                    # else the coupled params will be added from fit results
                    textstr = "f_ro = {:.3f} GHz\nf_pf = {:3f} GHz\n" \
                        "kappa = {:3f} MHz\nJ = {:3f} MHz\ngamma_ro = " \
                        "{:3f} MHz".format(
                            fit_results.params['omega_ro'].value * 1e-9,
                            fit_results.params['omega_pf'].value * 1e-9,
                            fit_results.params['kappa_pf'].value * 1e-6,
                            fit_results.params['J'].value * 1e-6,
                            fit_results.params['gamma_ro'].value * 1e-6)
                    # note: next line will have to be removed when
                    # cleaning up the # timestamps dependency
                    ax.plot(freqs,
                            fit_results.best_fit, 'r-', linewidth=1.5)
                    ax.plot([], [], 'w', label=textstr)

            if len(self.proc_data_dict['spectra']) != 1 :
                for states, params in self.fit_res.items():
                    f_r = fit_results[states]
                    if len(states) == 3:
                        ax.plot([params["f_RO"], params["f_RO"]],
                                [0, np.max(np.abs(np.asarray(
                                    list(self.proc_data_dict['spectra'].values()))))],
                                'm--', linewidth=1.5, label="F_RO_{}"
                                .format(states))
                        ax2 = ax.twinx()
                        last_fit_key = list(f_r["avg_fidelities"].keys())[-1]
                        ax2.scatter(freqs, f_r["avg_fidelities"][last_fit_key],
                                    color='c',
                                    label= "{} fidelity".format(states),
                                    marker='.')
                        ax2.set_ylabel("Fidelity")
                        label = "f_RO_{} = {:.5f} GHz".format(states,
                                                          params['f_RO'] * 1e-9)
                        ax.plot([],[], label=label)
                        fig, ax3 = plt.subplots()
                        for sigma, avg_fid in f_r['avg_fidelities'].items():
                            ax3.plot(self.proc_data_dict['plot_frequency'][0],
                                     avg_fid, label=sigma)
                        ax3.plot([f_r["optimal_frequency"][-1]],
                                 [f_r["optimal_frequency"][-1]], "k--")
                        t_f = self.options_dict.get('qutrit_fit_options', dict())
                        ax3.set_ylim([0.9, 1])

                    elif len(states) == 2:
                        c = "r--"
                        c2 = "k--"
                        ax.plot(freqs, f_r[0].eval(f_r[0].params, f=freqs),
                                c, label=label.format(states[0], "fit"),
                                linewidth=1.5)
                        ax.plot(freqs, f_r[1].eval(f_r[1].params, f=freqs),
                                c2, label=label.format(states[1], "fit"),
                                linewidth=1.5)
                        ax.plot([params['f_RO'], params['f_RO']],
                                [0, np.max(np.abs(np.asarray(list(self.proc_data_dict['spectra'].values()))))],
                                'r--', linewidth=2)

                        params_str = 'states: {}' \
                            '\n kappa = {:.3f} MHz\n J = {:.3f} MHz' \
                            '\n chi = {:.3f} MHz\n f_pf = {:.3f} GHz' \
                            '\n f_rr $|{}\\rangle$ = {:.3f} GHz' \
                            '\n f_rr $|{}\\rangle$ = {:.3f} GHz' \
                            '\n f_RO = {:.3f} GHz'.format(
                            states,
                            f_r[0].params['kappa_pf'].value * 1e-6,
                            f_r[0].params['J'].value * 1e-6,
                            (f_r[1].params['omega_ro'].value -
                             f_r[0].params['omega_ro'].value) / 2 * 1e-6,
                            f_r[0].params['omega_pf'].value * 1e-9,

                            states[0], f_r[0].params['omega_ro'].value * 1e-9,
                            states[1], f_r[1].params['omega_ro'].value * 1e-9,
                            params['f_RO'] * 1e-9)
                        ax.plot([],[], 'w', label=params_str)
            ax.legend(loc='upper left', bbox_to_anchor=[1.1, 1])

    def plot_difference_iq_plane(self, fig=None):
        spectrums = self.proc_data_dict['spectra']
        all_freqs = self.proc_data_dict['plot_frequency']
        freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
        total_dist = np.abs(spectrums['e'] - spectrums['g']) + \
                     np.abs(spectrums['f'] - spectrums['g']) + \
                     np.abs(spectrums['f'] - spectrums['e'])
        fmax = freqs[np.argmax(total_dist)]
        # FIXME: just as debug plotting for now
        if fig is None:
            fig, ax = plt.subplots(2, figsize=(10,14))
        else:
            ax = fig.get_axes()
        ax[0].plot(freqs, np.abs(spectrums['g']), label='g')
        ax[0].plot(freqs, np.abs(spectrums['e']), label='e')
        ax[0].plot(freqs, np.abs(spectrums['f']), label='f')
        ax[0].set_ylabel('Amplitude')
        ax[0].legend()
        ax[1].plot(freqs, np.abs(spectrums['e'] - spectrums['g']), label='eg')
        ax[1].plot(freqs, np.abs(spectrums['f'] - spectrums['g']), label='fg')
        ax[1].plot(freqs, np.abs(spectrums['e'] - spectrums['f']), label='ef')
        ax[1].plot(freqs, total_dist, label='total distance')
        ax[1].set_xlabel("Freq. [Hz]")
        ax[1].set_ylabel('Distance in IQ plane')
        ax[0].set_title(f"Max Diff Freq: {fmax*1e-9} GHz")
        ax[1].legend(loc=[1.01, 0])
        return fig

    def plot_gaussian_overlap(self, fig=None):
        states = list(self.proc_data_dict['spectra'])
        all_freqs = self.proc_data_dict['plot_frequency']
        freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
        if len(states) == 3:
            f_RO_qutrit = self.find_f_RO(states)
            f_r = self.proc_data_dict["fit_raw_results"]["".join(states)]
            if fig is None:
                fig, ax = plt.subplots(2, figsize=(10,14))
            else:
                ax = fig.get_axes()
            ax[0].plot([f_RO_qutrit, f_RO_qutrit],
                    [0, 1],
                    'm--', linewidth=1.5, label="F_RO_{}"
                    .format(states))

            last_fit_key = list(f_r["avg_fidelities"].keys())[-1]
            ax[0].scatter(freqs, f_r["avg_fidelities"][last_fit_key],
                        color='c',
                        label="{} fidelity".format(states),
                        marker='.')
            ax[0].set_ylabel("Expected Fidelity")
            label = "f_RO_{} = {:.6f} GHz".format(states,
                                                  f_RO_qutrit * 1e-9)
            ax[0].plot([], [], label=label)
            ax[0].legend()

            for sigma, avg_fid in f_r['avg_fidelities'].items():
                ax[1].plot(self.proc_data_dict['plot_frequency'][0],
                         avg_fid, label=sigma)

            ax[1].axvline(f_r["optimal_frequency"][-1],linestyle="--", )
            #ax.set_ylim([0.9, 1])
            return fig

    def plot_max_area(self, fig=None):
        spectrums = self.proc_data_dict['spectra']
        states = list(self.proc_data_dict['spectra'])
        all_freqs = self.proc_data_dict['plot_frequency']
        freqs = all_freqs if np.ndim(all_freqs) == 1 else all_freqs[0]
        if len(states) == 3:
            # Area of triangle in IQ plane using Heron formula
            s1, s2, s3 = np.abs(spectrums['e'] - spectrums['g']), \
                         np.abs(spectrums['f'] - spectrums['g']),\
                         np.abs(spectrums['f'] - spectrums['e'])
            s = (s1 + s2 + s3)/2
            qutrit_triangle_area = np.sqrt(s * (s - s1) * (s - s2) * (s - s3))
            f_max_area = freqs[np.argmax(qutrit_triangle_area)]
            if fig is None:
                fig, ax = plt.subplots(1, figsize=(14, 8))
            else:
                ax = fig.get_axes()
            ax.plot([f_max_area, f_max_area],
                       [0, np.max(qutrit_triangle_area)],
                       'm--', linewidth=1.5, label="F_RO_{}"
                       .format(states))


            ax.scatter(freqs, qutrit_triangle_area,
                          label="{} area in IQ".format(states))
            ax.set_ylabel("qutrit area in IQ")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_title( "f_RO_{}_area = {:.6f} GHz".format(states,
                                                  f_max_area * 1e-9))

        return fig
            # ax.set_ylim([0.9, 1])

    def plot(self, key_list=None, axs_dict=None, presentation_mode=None, no_label=False):
        super(ResonatorSpectroscopy_v2, self).plot(key_list=key_list,
                                                axs_dict=axs_dict,
                                                presentation_mode=presentation_mode)
        if self.do_fitting:
            self.plot_fitting()


class Spectroscopy(ba.BaseDataAnalysis):
    """ A baseclass for spectroscopic measurements.

    Supports analyzing data from 2d sweeps and also combining data from multiple
    timestamps.

    Args:
        t_start, t_stop, options_dict, label, extract_only, do_fitting:
            See dodcstring of `BaseDataAnalysis`.
        auto: bool
            Run the analysis as the last step of initialization.

    Parameters used from the options_dict:
        param_2d: A path to a parameter in the hdf5 file that is interpreted
            as the second sweep dimension in case the sweep is split into
            multiple 1d sweep files. Optional.

    Parameters used either from metadata or options_dict:
        calc_pca: Whether to calculate the principal component of the spectrum,
            combining amplitude and phase. Default False.
        global_pca: If calculating the principal component, whether to do it
            globally or per-second-sweep-dimension-point. Default False.

    Plotting related parameters either from metadata or options_dict:
        plot_lines: Whether to do a line plots. Defaults to True if nr of 2d
            sweep points is smaller than 4, False otherwise.
        plot_color: Whether to do a 2d coulour-plots. Defaults to True if nr of
            2d sweep points is larger than 3, False otherwise.
        plot_amp: Whether to plot transmission amplitude. Default True.
        plot_phase: Whether to plot transmission phase. Default True.
        plot_pca: Whether to plot principal component of the spectrum.
            Default False.
        label_1d: Label for the first sweep dimension. Default 'Frequency'.
        unit_1d: Unit for the first sweep dimension. Default 'Hz'.
        label_2d: Label for the second sweep dimension. Default 'Frequency'.
        unit_2d: Unit for the second sweep dimension. Default 'Frequency'.
        label_amp: Label for the amplitude output. Default 'Amplitude'.
        unit_amp: Unit for the amplitude output. Default 'V'.
        range_amp: Range for the amplitude output. Default min-to-max.
        label_phase: Label for the phase output. Default 'Phase'.
        unit_phase: Unit for the phase output. Default 'deg'.
        range_phase: Range for the phase output. Default Default min-to-max.
        label_pca: Label for the principal component output.
            Default 'Principal component'.
        unit_pca: Unit for the principal component output. Default 'V'.
        range_pca: Range for the principal component output. Default min-to-max.
    """
    def __init__(self, t_start: str = None,
                 t_stop: str = None,
                 options_dict: dict = None,
                 label: str = None,
                 extract_only: bool = False,
                 auto: bool = True,
                 do_fitting: bool = False):
        if options_dict is None:
            options_dict = {}
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict,
                         label=label,
                         extract_only=extract_only,
                         do_fitting=do_fitting)
        self.params_dict = {'measurementstring': 'measurementstring'}
        self.param_2d = options_dict.get('param_2d', None)
        if self.param_2d is not None:
            pname = 'Instrument settings.' + self.param_2d
            self.params_dict.update({'param_2d': pname})
            self.numeric_params = ['param_2d']

        if auto:
            self.run_analysis()

    def process_data(self):
        pdd = self.proc_data_dict
        rdds = self.raw_data_dict
        if not isinstance(self.raw_data_dict, (tuple, list)):
            rdds = (rdds,)

        pdd['freqs'] = []  # list of lists of floats
        pdd['amps'] = []  # list of lists of floats
        pdd['phases'] = []  # list of lists of floats
        pdd['values_2d'] = []  # list of floats

        for rdd in rdds:
            f, a, p, v = self._process_spec_rdd(rdd)
            pdd['freqs'] += f
            pdd['amps'] += a
            pdd['phases'] += p
            pdd['values_2d'] += v
        next_idx = 0
        for i in range(len(pdd['values_2d'])):
            if pdd['values_2d'][i] is None:
                pdd['values_2d'][i] = next_idx
                next_idx += 1

        spn = rdds[0]['sweep_parameter_names']
        pdd['label_2d'] = '2D index' if isinstance(spn, str) else spn[1]
        pdd['label_2d'] = self.get_param_value('name_2d', pdd['label_2d'])
        spu = rdds[0]['sweep_parameter_units']
        pdd['unit_2d'] = '' if isinstance(spu, str) else spu[1]
        pdd['unit_2d'] = self.get_param_value('unit_2d', pdd['unit_2d'])
        pdd['ts_string'] = self.timestamps[0]
        if len(self.timestamps) > 1:
            pdd['ts_string'] = pdd['ts_string'] + ' to ' + self.timestamps[-1]

        if self.get_param_value('calc_pca', False):
            if self.get_param_value('global_pca', False):
                # find global transformation
                amp = np.array([a for amps in pdd['amps'] for a in amps])
                phase = np.array([p for ps in pdd['phases'] for p in ps])
                _, pca_basis = self._transform_pca(amp, phase)

                # apply found transform to data
                pdd['pcas'] = []
                for amp, phase in zip(pdd['amps'], pdd['phases']):
                    pca, _ = self._transform_pca(amp, phase, basis=pca_basis)
                    pdd['pcas'].append(pca)

                # subtract offset and fix sign
                pca = np.array([p for pcas in pdd['pcas'] for p in pcas])
                median = np.median(pca)
                sign = np.sign(pca[np.argmax(np.abs(pca - median))])
                for i in range(len(pdd['pcas'])):
                    pdd['pcas'][i] = sign * (pdd['pcas'][i] - median)
            else:
                pdd['pcas'] = []
                for amp, phase in zip(pdd['amps'], pdd['phases']):
                    pca, _ = self._transform_pca(amp, phase)
                    pdd['pcas'].append(pca)

    @staticmethod
    def _transform_pca(amp, phase, basis=None):
        i = amp * np.cos(np.pi * phase / 180)
        q = amp * np.sin(np.pi * phase / 180)
        pca = np.array([i, q]).T
        if basis is None:
            pca -= pca.mean(axis=0)
            pca_basis = np.linalg.eigh(pca.T @ pca)[1]
        else:
            pca_basis = basis
        pca = (pca_basis @ pca.T)[1]
        if basis is None:
            pca -= np.median(pca)
            pca *= np.sign(pca[np.argmax(np.abs(pca))])
        return pca, pca_basis

    @staticmethod
    def _process_spec_rdd(rdd):
        if 'soft_sweep_points' in rdd:
            # 2D sweep
            v = list(rdd['soft_sweep_points'])
            f = len(v) * [rdd['hard_sweep_points']]
            a = list(rdd['measured_data']['Magn'].T)
            p = list(rdd['measured_data']['Phase'].T)
        else:
            # 1D sweep
            v = [rdd.get('param_2d', None)]
            f = [rdd['hard_sweep_points']]
            a = [rdd['measured_data']['Magn']]
            p = [rdd['measured_data']['Phase']]
        return f, a, p, v

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        if isinstance(rdd, (tuple, list)):
            rdd = rdd[0]

        def calc_range(values):
            return (min([np.min(x) for x in values]),
                    max([np.max(x) for x in values]))

        plot_lines = self.get_param_value('plot_lines', len(pdd['amps']) <= 3)
        plot_color = self.get_param_value('plot_color', len(pdd['amps']) > 3)
        plot_amp = self.get_param_value('plot_amp', True)
        plot_phase = self.get_param_value('plot_phase', True)
        plot_pca = self.get_param_value('plot_pca',
                                        self.get_param_value('calc_pca', False))
        label1 = self.get_param_value('label_1d', 'Frequency')
        unit1 = self.get_param_value('unit_1d', 'Hz')
        label2 = self.get_param_value('label_2d', pdd['label_2d'])
        unit2 = self.get_param_value('unit_2d', pdd['unit_2d'])
        label_amp = self.get_param_value('label_amp', 'Amplitude')
        unit_amp = self.get_param_value('unit_amp', 'V')
        range_amp = self.get_param_value('range_amp', calc_range(pdd['amps']))
        label_phase = self.get_param_value('label_phase', 'Phase')
        unit_phase = self.get_param_value('unit_phase', 'deg')
        range_phase = self.get_param_value('range_phase',
                                           calc_range(pdd['phases']))
        label_pca = self.get_param_value('label_pca', 'Principal component')
        unit_pca = self.get_param_value('unit_pca', 'V')
        range_pca = calc_range(pdd['pcas']) if 'pcas' in pdd else (0, 1)
        range_pca = self.get_param_value('range_pca', range_pca)

        fig_title_suffix = ' ' + rdd['measurementstring'] + '\n' + \
                           pdd['ts_string']

        if plot_lines:
            for enable, param, plot_name, ylabel, yunit, yrange in [
                (plot_amp, 'amps', 'amp_1d', label_amp, unit_amp,
                 range_amp),
                (plot_phase, 'phases', 'phase_1d', label_phase, unit_phase,
                 range_phase),
                (plot_pca, 'pcas', 'pca_1d', label_pca, unit_pca,
                 range_pca),
            ]:
                if enable:
                    self.plot_dicts[plot_name] = {
                        'fig_id': plot_name,
                        'plotfn': self.plot_line,
                        'xvals': pdd['freqs'],
                        'yvals': pdd[param],
                        'xlabel': label1,
                        'xunit': unit1,
                        'ylabel': ylabel,
                        'yunit': yunit,
                        'yrange': yrange,
                        'title': plot_name + fig_title_suffix,
                    }
        if plot_color:
            for enable, param, plot_name, zlabel, zunit, zrange in [
                (plot_amp, 'amps', 'amp_2d', label_amp, unit_amp,
                 range_amp),
                (plot_phase, 'phases', 'phase_2d', label_phase, unit_phase,
                 range_phase),
                (plot_pca, 'pcas', 'pca_2d', label_pca, unit_pca,
                 range_pca),
            ]:
                if enable:
                    self.plot_dicts[plot_name] = {
                        'fig_id': plot_name,
                        'plotfn': self.plot_colorx,
                        'xvals': pdd['values_2d'],
                        'yvals': pdd['freqs'],
                        'zvals': pdd[param],
                        'zrange': zrange,
                        'xlabel': label2,
                        'xunit': unit2,
                        'ylabel': label1,
                        'yunit': unit1,
                        'clabel': f'{zlabel} ({zunit})',
                        'title': plot_name + fig_title_suffix,
                    }


class QubitTrackerSpectroscopy(Spectroscopy):
    """A class for peak-tracking 2d spectroscopy.

    Fits the spectroscopy data to a Gaussian model and can extrapolate a
    polynomial model of the peak frequency as a function of the second sweep
    parameter to guess a frequency range for the next sweep.

    Args: Same as for `Spectroscopy`.

    Parameters used from the options_dict: Same as for `Spectroscopy`.

    Parameters used either from metadata or options_dict:
        calc_pca: Hard-coded to True, as the amplitude-phase data needs to be
            reduced for fitting.
        global_pca: Whether to do principal component analysis globally or
            per-second-sweep-dimension-point. Default False.
        tracker_fit_order: Polynomial order for extrapolating the measurement
            range. Default 1.
        tracker_fit_points: Number of 2d sweep points to use for the polynomial
            fit. The points are taken evenly from the entire range. Default 4.

    Plotting related parameters either from metadata or options_dict:
        Same as for `Spectroscopy`.
    """
    def __init__(self, t_start: str = None,
                 t_stop: str = None,
                 options_dict: dict = None,
                 label: str = None,
                 extract_only: bool = False,
                 auto: bool = True,
                 do_fitting: bool = True):
        if options_dict is None:
            options_dict = {}
        options_dict['calc_pca'] = True
        super().__init__(t_start=t_start, t_stop=t_stop,
                         options_dict=options_dict, label=label,
                         extract_only=extract_only, auto=auto,
                         do_fitting=do_fitting)

    def prepare_fitting(self):
        super().prepare_fitting()
        pdd = self.proc_data_dict
        fit_order = self.get_param_value('tracker_fit_order', 1)
        fit_pts = self.get_param_value('tracker_fit_points', 4)
        if fit_pts < fit_order + 1:
            raise ValueError(f"Can't fit {fit_pts} points to order {fit_order} "
                             "polynomial")
        idxs = np.round(
            np.linspace(0, len(pdd['pcas']) - 1, fit_pts)).astype(np.int)
        pdd['fit_idxs'] = idxs
        model = fit_mods.GaussianModel
        model.guess = fit_mods.Gaussian_guess.__get__(model, model.__class__)
        for i in idxs:
            self.fit_dicts[f'tracker_fit_{i}'] = {
                'model': model,
                'fit_xvals': {'freq': pdd['freqs'][i]},
                'fit_yvals': {'data': pdd['pcas'][i]},
            }

    def analyze_fit_results(self):
        super().analyze_fit_results()
        pdd = self.proc_data_dict
        fit_order = self.get_param_value('tracker_fit_order', 1)
        model = lmfit.models.PolynomialModel(degree=fit_order)
        xpoints = [pdd['values_2d'][i] for i in pdd['fit_idxs']]
        ypoints = [self.fit_res[f'tracker_fit_{i}'].best_values['mu']
                   for i in pdd['fit_idxs']]
        self.fit_dicts['tracker_fit'] = {
            'model': model,
            'fit_xvals': {'x': xpoints},
            'fit_yvals': {'data': ypoints},
        }

        self.run_fitting()
        self.save_fit_results()

    def prepare_plots(self):
        super().prepare_plots()
        pdd = self.proc_data_dict
        plot_color = self.get_param_value('plot_color', len(pdd['amps']) > 3)
        if self.do_fitting and plot_color:
            xpoints = [pdd['values_2d'][i] for i in pdd['fit_idxs']]
            ypoints = [self.fit_res[f'tracker_fit_{i}'].best_values['mu']
                       for i in pdd['fit_idxs']]
            self.plot_dicts['pca_2d_fit1'] = {
                'fig_id': 'pca_2d',
                'plotfn': self.plot_line,
                'xvals': xpoints,
                'yvals': ypoints,
                'marker': 'o',
                'linestyle': '',
                'color': 'red',
            }

            xpoints = np.linspace(min(xpoints), max(xpoints), 101)
            fr = self.fit_res[f'tracker_fit']
            ypoints = fr.model.func(xpoints, **fr.best_values)
            self.plot_dicts['pca_2d_fit2'] = {
                'fig_id': 'pca_2d',
                'plotfn': self.plot_line,
                'xvals': xpoints,
                'yvals': ypoints,
                'marker': '',
                'linestyle': '-',
                'color': 'green',
            }

    def next_round_limits(self, freq_slack=0):
        """Calculate 2d-parameter and frequency ranges for next tracker sweep.

        The 2d parameter range is calculated that it spans the same range as
        the current sweep, but starts one mean step-size after the current
        sweep.

        The frequency range is calculated such that the extrapolated polynomial
        fits inside the range within the 2d parameter range, with some optional
        extra margin that is passed as an argument.

        Args:
            freq_slack: float
                Extra frequency margin for the output frequency range. The
                output range is extended by this value on each side.
                Default 0.

        Returns:
            v2d_next: (float, float)
                Range for the 2d sweep parameter for the next sweep.
            f_next: (float, float)
                Range for the frequency sweep for the next sweep.
        """
        if 'tracker_fit' not in self.fit_res:
            raise KeyError('Tracker fit not yet run.')
        pdd = self.proc_data_dict
        fr = self.fit_res['tracker_fit']
        v2d = pdd['values_2d']
        v2d_next = (v2d[-1] + (v2d[-1] - v2d[0])/(len(v2d)-1),
                    2*v2d[-1] - v2d[0] + (v2d[-1] - v2d[0])/(len(v2d)-1))
        x = np.linspace(v2d_next[0], v2d_next[1], 101)
        y = fr.model.func(x, **fr.best_values)
        f_next = (y.min() - freq_slack, y.max() + freq_slack)
        return v2d_next, f_next


class MultiQubit_Spectroscopy_Analysis(tda.MultiQubit_TimeDomain_Analysis):
    """Base class for the analysis of `MultiTaskingSpectroscopyExperiment`.

    Transforms the IQ data provided by the detector function into magnitude and
    phase and overwrites specific methods of tda.MultiQubit_TimeDomain_Analysis.
    """
    def process_data(self):
        super().process_data()

        mdata_per_qb = self.proc_data_dict['meas_results_per_qb_raw']
        self.proc_data_dict['projected_data_dict'] = {
            qb: self._transform(raw_data, True) for qb, raw_data in mdata_per_qb.items()
        }

    def _transform(self, data, transpose=True):
        polar_data = dict()
        values = list(data.values())
        S21 = values[0] + 1j * values[1] # vector in complex plane
        polar_data["Magnitude"] = np.abs(S21).T if transpose else np.abs(S21)
        polar_data["Phase"] = np.angle(S21).T if transpose else np.angle(S21)
        return polar_data

    def get_yaxis_label(self, qb_name, data_key=None):
        if data_key is None:
            return 'Measured Data (arb.)'
        if data_key == 'Phase':
            # FIXME a cleaner version would be to implemented this via yunit
            #  in tda.MultiQubit_TimeDomain_Analysis.prepare_projected_data_plot
            return 'Phase (rad)'
        if data_key == 'Magnitude':
            # FIXME a cleaner version would be to implemented this via yunit
            #  in tda.MultiQubit_TimeDomain_Analysis.prepare_projected_data_plot
            return 'Magnitude (Vpeak)'
        return data_key


class MultiQubit_AvgRoCalib_Analysis(MultiQubit_Spectroscopy_Analysis):
    """Analysis to find the RO frequency that maximizes distance in IQ plane.

    Compatible with `MultiTaskingSpectroscopyExperiment`.
    """
    def process_data(self):
        super().process_data()

        mdata_per_qb = self.proc_data_dict['meas_results_per_qb_raw']
        for qb, raw_data in mdata_per_qb.items():
            sp2d_key = f'{qb}_initialize'
            if sp2d_key not in self.proc_data_dict['sweep_points_2D_dict'][qb].keys():
                sp2d_key = 'initialize'
            states = self.proc_data_dict['sweep_points_2D_dict'][qb][sp2d_key]
            self.proc_data_dict['projected_data_dict'][qb]['distance'] = \
                self._compute_s21_distance(raw_data, states)

    def _compute_s21_distance(self, data, states):
        distances = dict()
        if len(states) < 2:
            log.error('At least two states need to be measured to compute a'
                      'distance in transmission.')
        state_index_map = {state: i for i, state in enumerate(states)}
        for state_1, state_2 in combinations(states, 2):
            ind_1 = state_index_map[state_1]
            ind_2 = state_index_map[state_2]
            I, Q = list(data.values())
            S21_1 = I[:, ind_1] + 1j * Q[:, ind_1]
            S21_2 = I[:, ind_2] + 1j * Q[:, ind_2]
            distance = np.abs(S21_1-S21_2)
            argmax = np.argmax(distance)
            distances[f'{state_1}-{state_2}'] = (distance, argmax)
        return distances

    def prepare_plots(self):
        pdd = self.proc_data_dict
        plotsize = self.get_default_plot_params(set=False)['figure.figsize']
        for qb_name in self.qb_names:
            fig_title = f'Magnitude and Phase Plot {qb_name}'
            plot_name = f'raw_mag_phase_{qb_name}'
            frequency = pdd['sweep_points_dict'][qb_name]['sweep_points']
            for ax_id, key in enumerate(['Magnitude', 'Phase']):
                data = pdd['projected_data_dict'][qb_name][key]
                sp2d_key = f'{qb_name}_initialize'
                if sp2d_key not in pdd['sweep_points_2D_dict'][qb_name].keys():
                    sp2d_key = 'initialize'
                for i, state in enumerate(pdd['sweep_points_2D_dict'][qb_name] \
                                             [sp2d_key]):
                    yvals = data[i, :]
                    self.plot_dicts[f'raw_{key}_{state}_{qb_name}'] = {
                            'fig_id': plot_name,
                            'ax_id': ax_id,
                            'plotfn': self.plot_line,
                            'xvals': frequency,
                            'xlabel': 'RO frequency',
                            'xunit': 'Hz',
                            'yvals': yvals,
                            'ylabel': key,
                            'yunit': 'rad' if key == 'Phase' else 'Vpeak',
                            'line_kws': {'color': self.get_state_color(state)},
                            'numplotsx': 1,
                            'numplotsy': 2,
                            'plotsize': (plotsize[0],
                                        plotsize[1]*2),
                            'title': fig_title if not ax_id else None,
                            'setlabel': f'|{state}>',
                            'do_legend': True,
                    }
            fig_title = f'S21 distance {qb_name}'
            plot_name = f's21_distance_{qb_name}'
            for dkey, val in pdd['projected_data_dict'][qb_name]['distance'].items():
                distance, argmax = val
                self.plot_dicts[f's21_distance_{dkey}_{qb_name}'] = {
                        'fig_id': plot_name,
                        'plotfn': self.plot_line,
                        'xvals': frequency,
                        'xlabel': 'RO frequency',
                        'xunit': 'Hz',
                        'yvals': distance,
                        'ylabel': 'S21 distance',
                        'yunit': 'Vpeak',
                        'label': dkey,
                        'title': fig_title,
                        'setlabel': f'S21 distance {dkey}',
                }
                self.plot_dicts[f's21_distance_{dkey}_{qb_name}_max'] = {
                        'fig_id': plot_name,
                        'plotfn': self.plot_line,
                        'xvals': frequency[argmax] * np.ones(2),
                        'yvals': [0.95 * np.min(distance),
                                  1.05 * np.max(distance)],
                        'color': 'red',
                        'linestyle': 'dotted',
                        'marker': 'None',
                        'setlabel': '$f_{RO}$ = ' \
                                    + f'{(frequency[argmax]/1e9):.3f} GHz',
                        'do_legend': True,
                }

    def get_state_color(self, state):
        state_colors = {'g': 'blue', 'e': 'orange', 'f': 'green'}
        INT_TO_STATE = 'gef'
        if isinstance(state, int):
            state = INT_TO_STATE[state]
        if state not in state_colors.keys():
            state = INT_TO_STATE[int(state)]
        return state_colors[state]
