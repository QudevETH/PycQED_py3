"""
Spectroscopy class

This file contains the Spectroscopy class that forms the basis analysis of
all the spectroscopy measurement analyses.
"""

import logging
import numpy as np
import lmfit
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis.fitting_models as fit_mods

log = logging.getLogger(__name__)


class Spectroscopy(ba.BaseDataAnalysis):

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
            f, a, p, v = self.process_spec_rdd(rdd)
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
                amp = np.array([a for amps in pdd['amps'] for a in amps])
                phase = np.array([p for ps in pdd['phases'] for p in ps])
                I = amp * np.cos(np.pi * phase / 180)
                Q = amp * np.sin(np.pi * phase / 180)
                PCA = np.array([I, Q]).T
                PCA -= PCA.mean(axis=0)
                PCA = np.linalg.eigh(PCA.T @ PCA)[1]

                pdd['pcas'] = []
                for amp, phase in zip(pdd['amps'], pdd['phases']):
                    i = amp * np.cos(np.pi * phase / 180)
                    q = amp * np.sin(np.pi * phase / 180)
                    pca = np.array([i, q])
                    pca = (PCA @ pca)[1]
                    pdd['pcas'].append(pca)

                pca = np.array([p for pcas in pdd['pcas'] for p in pcas])
                median = np.median(pca)
                sign = np.sign(pca[np.argmax(np.abs(pca - median))])
                for i in range(len(pdd['pcas'])):
                    pdd['pcas'][i] = sign * (pdd['pcas'][i] - median)
            else:
                pdd['pcas'] = []
                for amp, phase in zip(pdd['amps'], pdd['phases']):
                    i = amp * np.cos(np.pi * phase / 180)
                    q = amp * np.sin(np.pi * phase / 180)
                    pca = np.array([i, q]).T
                    pca -= pca.mean(axis=0)
                    pca = (np.linalg.eigh(pca.T @ pca)[1] @ pca.T)[1]
                    pca -= np.median(pca)
                    pca *= np.sign(pca[np.argmax(np.abs(pca))])
                    pdd['pcas'].append(pca)

    def process_spec_rdd(self, rdd):
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
            # cg = pdd['freqs'][i][np.argmax(pdd['pcas'][i])]
            self.fit_dicts[f'tracker_fit_{i}'] = {
                'model': model,
                # 'guess_dict': {'freq': {'value': cg}},
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
