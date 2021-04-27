"""
File containing the analysis for characterizing an amplifier.
"""

import numpy as np
import pycqed.analysis_v2.base_analysis as ba

class Amplifier_Characterization_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='',
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.do_timestamp_blocks = False
        self.single_timestamp = False

        self.params_dict = {
            'sweep_parameter_names': 'Experimental Data.sweep_parameter_names',
            'sweep_parameter_units': 'Experimental Data.sweep_parameter_units',
            'measurementstring': 'measurementstring',
            'measured_values': 'measured_values',
            'value_names': 'value_names',
            'value_units': 'value_units',
        }

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        # Extract sweep points
        self.proc_data_dict['dim'] = 2 if 'soft_sweep_points' in self.raw_data_dict[0] else 1
        self.proc_data_dict['sweep_label'] = \
            self.raw_data_dict[0]['sweep_parameter_names']
        self.proc_data_dict['sweep_unit'] = \
            self.raw_data_dict[0]['sweep_parameter_units']
        if self.proc_data_dict['dim'] > 1:
            self.proc_data_dict['sweep_points'] = \
                unsorted_unique(self.raw_data_dict[0]['hard_sweep_points'])
            self.proc_data_dict['sweep_points_2D'] = \
                unsorted_unique(self.raw_data_dict[0]['soft_sweep_points'])
            self.proc_data_dict['sweep_label'], \
                self.proc_data_dict['sweep_label_2D'] = \
                self.proc_data_dict['sweep_label'][:2]
            self.proc_data_dict['sweep_unit'], \
                self.proc_data_dict['sweep_unit_2D'] = \
                self.proc_data_dict['sweep_unit'][:2]
            self.proc_data_dict['sweep_label'] = \
                self.proc_data_dict['sweep_label']
            self.proc_data_dict['sweep_label_2D'] = \
                self.proc_data_dict['sweep_label_2D']
            self.proc_data_dict['sweep_unit'] = \
                self.proc_data_dict['sweep_unit']
            self.proc_data_dict['sweep_unit_2D'] = \
                self.proc_data_dict['sweep_unit_2D']
        else:
            self.proc_data_dict['sweep_points'] = \
                self.raw_data_dict[0]['hard_sweep_points']

        # Extract signal and noise powers
        self.proc_data_dict['signal_power'] = \
            self.raw_data_dict[0]['measured_data']['I'] ** 2 + \
            self.raw_data_dict[0]['measured_data']['Q'] ** 2
        self.proc_data_dict['signal_power_ref'] = \
            self.raw_data_dict[1]['measured_data']['I'] ** 2 + \
            self.raw_data_dict[1]['measured_data']['Q'] ** 2
        correlator_scale = self.options_dict.get('correlator_scale', 1)
        self.proc_data_dict['total_power'] = \
            (self.raw_data_dict[0]['measured_data']['corr (0,0)'] +
             self.raw_data_dict[0]['measured_data']['corr (1,1)']) * correlator_scale
        self.proc_data_dict['total_power_ref'] = \
            (self.raw_data_dict[1]['measured_data']['corr (0,0)'] +
             self.raw_data_dict[1]['measured_data']['corr (1,1)']) * correlator_scale
        if self.proc_data_dict['dim'] > 1:
            for key in ['signal_power', 'total_power']:
                self.proc_data_dict[key] = np.reshape(
                    self.proc_data_dict[key],
                    (len(self.proc_data_dict['sweep_points']),
                     len(self.proc_data_dict['sweep_points_2D']))).T
        self.proc_data_dict['noise_power'] = \
            self.proc_data_dict['total_power'] - \
            self.proc_data_dict['signal_power']
        self.proc_data_dict['noise_power_ref'] = \
            self.proc_data_dict['total_power_ref'] - \
            self.proc_data_dict['signal_power_ref']

        # Extract signal gain and snr2 gain
        self.proc_data_dict['signal_power_gain_dB'] = \
            10 * np.log10(self.proc_data_dict['signal_power'] /
                          self.proc_data_dict['signal_power_ref'])
        self.proc_data_dict['noise_power_gain_dB'] = \
            10 * np.log10(self.proc_data_dict['noise_power'] /
                          self.proc_data_dict['noise_power_ref'])
        self.proc_data_dict['snr2_gain_dB'] = \
            10 * np.log10(self.proc_data_dict['signal_power'] *
                          self.proc_data_dict['noise_power_ref'] /
                          self.proc_data_dict['noise_power'] /
                          self.proc_data_dict['signal_power_ref'])
        self.proc_data_dict['snr2_gain_dB'] = \
            np.ma.array(self.proc_data_dict['snr2_gain_dB'],
                        mask=np.isnan(self.proc_data_dict['snr2_gain_dB']))

    def prepare_plots(self):
        if self.proc_data_dict['dim'] > 1:
            self.prepare_plots_2D()
        else:
            self.prepare_plots_1D()

    def prepare_plots_1D(self):
        if len(self.proc_data_dict['sweep_points']) < 40:
            marker = '.'
        else:
            marker = ''
        self.plot_dicts['signal_power_gain'] = {
            'title': 'Signal power gain\n' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'fig_id': 'signal_power_gain',
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['signal_power_gain_dB'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': 'Signal power gain',
            'yunit': 'dB',
            'setlabel': 'Signal gain',
            'do_legend': True,
            'line_kws': {'color': 'C0'},
            'marker': marker}
        self.plot_dicts['signal_power_gain_2'] = {
            'fig_id': 'signal_power_gain',
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['noise_power_gain_dB'],
            'setlabel': 'Noise gain',
            'do_legend': True,
            'line_kws': {'color': 'C1', 'alpha': 0.5},
            'marker': marker}
        self.plot_dicts['snr2_gain'] = {
            'title': 'SNR${}^2$ gain ' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['snr2_gain_dB'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': 'SNR${}^2$ gain',
            'yunit': 'dB',
            'line_kws': {'color': 'C0'},
            'marker': marker}
        self.plot_dicts['noise_power'] = {
            'fig_id': 'noise_power',
            'title': 'Noise power ' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['noise_power'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': 'Noise power',
            'yunit': 'a.u.',
            'yscale': 'log',
            'setlabel': 'TWPA On',
            'do_legend': True,
            'line_kws': {'color': 'C0'},
            'marker': marker}
        self.plot_dicts['noise_power_2'] = {
            'fig_id': 'noise_power',
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': np.repeat(self.proc_data_dict['noise_power_ref'],
                               len(self.proc_data_dict['sweep_points']) //
                               len(self.proc_data_dict['noise_power_ref'])),
            'setlabel': 'TWPA Off',
            'do_legend': True,
            'line_kws': {'color': 'C1'},
            'marker': marker}
        self.plot_dicts['noise_power_gain'] = {
            'title': 'Noise power rise \n' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['noise_power_gain_dB'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': 'Noise power rise',
            'yunit': 'dB',
            'line_kws': {'color': 'C0'},
            'marker': marker}

    def prepare_plots_2D(self):
        cmap = self.options_dict.get('colormap', 'viridis')
        zmin = self.options_dict.get('sig_min',
            max(0, self.proc_data_dict['signal_power_gain_dB'].min()))
        zmax = self.options_dict.get('sig_max',
            self.proc_data_dict['signal_power_gain_dB'].max())
        self.plot_dicts['signal_power_gain'] = {
            'title': 'Signal power gain\n' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['sweep_points_2D'],
            'zvals': self.proc_data_dict['signal_power_gain_dB'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': self.proc_data_dict['sweep_label_2D'],
            'yunit': self.proc_data_dict['sweep_unit_2D'],
            'clabel': 'Signal power gain (dB)',
            'zrange': (zmin, zmax),
            'cmap': cmap}

        zmin = self.options_dict.get('snr_min',
            max(0, self.proc_data_dict['snr2_gain_dB'].min()))
        zmax = self.options_dict.get('snr_max',
            self.proc_data_dict['snr2_gain_dB'].max())
        self.plot_dicts['snr2_gain'] = {
            'title': 'SNR${}^2$ gain\n' +
                     self.timestamps[0] + ', ' + self.timestamps[1],
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['sweep_points'],
            'yvals': self.proc_data_dict['sweep_points_2D'],
            'zvals': self.proc_data_dict['snr2_gain_dB'],
            'xlabel': self.proc_data_dict['sweep_label'],
            'xunit': self.proc_data_dict['sweep_unit'],
            'ylabel': self.proc_data_dict['sweep_label_2D'],
            'yunit': self.proc_data_dict['sweep_unit_2D'],
            'clabel': 'SNR${}^2$ gain (dB)',
            'zrange': (zmin, zmax),
            'cmap': cmap}

def unsorted_unique(x):
    return x.flatten()[np.sort(np.unique(x, return_index=True)[1])]