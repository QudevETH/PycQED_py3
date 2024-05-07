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
            (self.raw_data_dict[0]['measured_data']['I^2'] +
             self.raw_data_dict[0]['measured_data']['Q^2']) * correlator_scale
        self.proc_data_dict['total_power_ref'] = \
            (self.raw_data_dict[1]['measured_data']['I^2'] +
             self.raw_data_dict[1]['measured_data']['Q^2']) * correlator_scale
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


class MultiTWPA_SNR_Analysis(ba.NDim_BaseDataAnalysis):
    """
    Analysis class for SNR (gain + noise) measurements for TWPAs.
    Will group data to reconstruct 4 successive NDimMultiTaskingExperiment:
    - num_ts times NoisePower (N-dim sweep over TWPA params + RO frequency)
    - num_ts times ResonatorSpectroscopy (same parameters)
    - 1 NoisePower (sweep over RO frequency with TWPA off)
    - 1 ResonatorSpectroscopy (same parameters)
    In addition to the datasets specified in get_measurement_groups,
    an 'SNR_rise' dataset is computed and added to self.proc_data_dict.

    Args:
        options_dict (dict): in addition to the keys used by the base class:
        - ro_freqs: dict of the form {TWPA.name: [f1, f2, ...]} specifying
            the readout frequencies of the qubits which are on a given feedline,
            which will be indicated as vertical white lines on the
            automatically generated plots
        - TWPAs, qubits: respectively a list of MeasurementObject and qubit
            objects, from which ro_freqs will be extracted if it does not exist

    FIXME for now mobj_names must be of the form
     [TWPA1.name, qb1.name, TWPA2.name...]
     where each characterized TWPA is followed by a qubit pertaining to the
     same feedline. This is because the spectroscopy (gain) measurements are
     for now only feasible with qubits.
    """

    def tile_data_ndim(self, ana_ids, post_proc_func, on_qb=False):

        # FIXME: this function is specific to MultiTWPA_SNR_Analysis.
        #  Once spectroscopies can be done without a qb, this can be cleaned
        #  up and replaced by the method from the base class, since we can
        #  remove on_qb and replace the interleaved list of mobjs and qb by a
        #  proper list of mobj

        data_ndim = {}
        sweep_points_ndim = {}

        for mobj_id in range(len(self.mobj_names) // 2):
            mobjn = self.mobj_names[2 * mobj_id]  # FIXME see main FIXME
            data_ndim[mobjn] = {}
            sweep_points_ndim[mobjn] =\
                self.get_ndim_sweep_points_from_mobj_name(
                    ana_ids, self.mobj_names[2 * mobj_id + on_qb])
            sweep_lengths = sweep_points_ndim[mobjn].length()
            data_ndim[mobjn] = np.nan * np.ones(sweep_lengths)
            for ana_id in ana_ids:
                # idxs: see NDimQuantumExperiment
                # idxs are for now the same for all tasks, they are just stored
                # in the task list for convenience
                idxs = \
                self.raw_data_dict[ana_id]['exp_metadata']['task_list'][0][
                    'ndim_current_idxs']
                ch_map = self.raw_data_dict[ana_id]['exp_metadata'][
                    'meas_obj_value_names_map'][
                    self.mobj_names[2 * mobj_id + on_qb]]
                ana_data_dict = self.raw_data_dict[ana_id]['measured_data']
                spectrum = post_proc_func(ana_data_dict, ch_map)
                for i in range(sweep_lengths[0]):
                    for j in range(sweep_lengths[1]):
                        data_ndim[mobjn][(i, j, *idxs)] = spectrum[i, j]
        return data_ndim, sweep_points_ndim

    def get_measurement_groups(self):
        post_proc_noise = lambda data_dict, ch_map: data_dict[ch_map[0]]
        post_proc_gain = lambda data_dict, ch_map: 20 * np.log10(
            np.sqrt(data_dict[ch_map[0]] ** 2 + data_dict[ch_map[1]] ** 2))

        # We have a set of num_ts+num_ts+1+1 experiments
        num_ts = len(self.raw_data_dict) // 2 - 1

        # Assumes that all experiments are ordered as follows:
        # num_ts for PSD, num_ts for gain, plus two reference
        # measurements with TWPA off
        measurement_groups = {
            'noise_on': dict(
                ana_ids=range(0, num_ts),
                post_proc_func=post_proc_noise,
            ),
            'gain_on': dict(
                ana_ids=range(num_ts, 2 * num_ts),
                post_proc_func=post_proc_gain,
                on_qb=True,
            ),
            'noise_off': dict(
                ana_ids=[-2],
                post_proc_func=post_proc_noise,
            ),
            'gain_off': dict(
                ana_ids=[-1],
                post_proc_func=post_proc_gain,
                on_qb=True,
            ),
        }
        return measurement_groups

    def process_data(self):
        super().process_data()
        pdd = self.proc_data_dict
        pdd['SNR_rise'] = {}
        pdd['sweep_points'] = pdd['group_sweep_points']['noise_on']
        for mobj_id in range(len(self.mobj_names) // 2):
            mobjn = self.mobj_names[
                2 * mobj_id]  # FIXME after removing qb objects
            reshape_1d = [-1] + [1] * (
                        len(pdd['sweep_points'][mobjn].length()) - 1)
            pdd['noise_off'][mobjn] = pdd['noise_off'][mobjn].reshape(
                *reshape_1d)
            pdd['gain_off'][mobjn] = pdd['gain_off'][mobjn].reshape(*reshape_1d)
            pdd['SNR_rise'][mobjn] = pdd['gain_on'][mobjn] - pdd['gain_off'][
                mobjn] - (pdd['noise_on'][mobjn] - pdd['noise_off'][mobjn])
        self.save_processed_data()

    def prepare_plots(self):
        """
        Prepares plotting dicts for a gain/SNR comparison

        In addition to gain and SNR measurements, this function can add vertical
        lines on all plots at ro_freqs, which can be passed in self.options_dict
        as a dict of the form {TWPA.name: [f1, f2, ...]}. Alternatively,
        these can be extracted automatically from entries 'TWPAs' and
        'qubits' if specified in self.options_dict.
        """
        pdd = self.proc_data_dict

        for mobj_id in range(len(self.mobj_names) // 2):
            mobjn = self.mobj_names[2 * mobj_id]  # FIXME see main FIXME

            # Will draw vertical lines at these values
            ro_freqs = self.options_dict.get('ro_freqs', {}).get(mobjn, [])
            if not ro_freqs:
                # An alternative way to define ro_freqs to plot is to pass
                # TWPAs and qubits in the options_dict
                if 'TWPAs' in self.options_dict:
                    TWPA = [TWPA for TWPA in self.options_dict['TWPAs']
                            if TWPA.name==mobjn]
                    assert len(TWPA)==1, f"Please pass {mobjn} if providing " \
                                         f"TWPAs in the options_dict"
                    TWPA = TWPA[0]
                    ro_freqs = [qb.ro_freq()
                                for qb in self.options_dict.get('qubits', [])
                                if qb.acq_unit() == TWPA.acq_unit()]

            sp = pdd['sweep_points'][mobjn]
            plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
            gain_min = self.options_dict.get('gain_min', 0)
            gain_max = self.options_dict.get('gain_max', 35)
            snr_min = self.options_dict.get('snr_min', 0)
            snr_max = self.options_dict.get('snr_max', 25)
            fig_key = f"{mobjn}_gain_SNR"
            gridspec_kw = {'height_ratios': [0.2] + [1]*len(sp['pump_freq'])}

            for idx_freq, pump_freq in enumerate(sp['pump_freq']):
                x = sp['freq'] / 1e9
                y = sp['pump_power']
                Z1, _ = self.get_slice(
                    pdd['gain_on'][mobjn] - pdd['gain_off'][mobjn],
                    sp,
                    {'pump_freq': pump_freq},
                )
                Z2, _ = self.get_slice(
                    pdd['SNR_rise'][mobjn],
                    sp,
                    {'pump_freq': pump_freq},
                )

                plot_key = f"{mobjn}_{pump_freq}"
                # Gain data
                self.plot_dicts[plot_key + "_gain"] = {
                    'plotsize': (plotsize[0] * 2, len(sp['pump_freq'])*2+2),
                    'gridspec_kw': gridspec_kw,
                    'numplotsx': 2,
                    'numplotsy': len(sp['pump_freq']) + 1,
                    'sharex': False,
                    'sharey': False,
                    'ax_id': 2 + 2 * idx_freq,
                    'fig_id': fig_key,
                    'plotfn': self.plot_colorxy,
                    'xvals': x,
                    'yvals': y,
                    'zvals': Z1.T,
                    'zrange': (gain_min, gain_max),
                    'xlabel': '',
                    'xtick_labels': [],
                    'xunit': '',
                    'xtick_rotation': 0,
                    'ylabel': 'Pump\npwr.',
                    'yunit': 'dBm',
                    'plotcbar': False,
                    'cmap': 'CMRmap',
                    'title': f'Pump frequency {pump_freq / 1e9:.3f} GHz',
                }
                # SNR data
                self.plot_dicts[plot_key + "_SNR"] = {
                    'ax_id':  2 + 2 * idx_freq + 1,
                    'fig_id': fig_key,
                    'plotfn': self.plot_colorxy,
                    'xvals': x,
                    'yvals': y,
                    'zvals': Z2.T,
                    'zrange': (snr_min, snr_max),
                    'xlabel': '',
                    'xtick_labels': [],
                    'xunit': '',
                    'xtick_rotation': 0,
                    'ylabel': '',
                    'ytick_labels': [],
                    'yunit': '',
                    'plotcbar': False,
                    'cmap': 'CMRmap',
                    # 'title': '',
                }
                # Colorbar
                if idx_freq == len(sp['pump_freq']) - 1:
                    self.plot_dicts[plot_key + "_gain"].update({
                        'plotcbar': True,
                        'cmap_levels': 100,
                        'clabel': 'Signal gain, $G$ (dB)',
                        'cax_id': 0,
                        'orientation': 'horizontal',
                        # 'location': 'top',
                        'xlabel': 'Signal frequency',
                        'xunit': 'GHz',
                    })
                    self.plot_dicts[plot_key + "_SNR"].update({
                        'plotcbar': True,
                        'cmap_levels': 100,
                        'clabel': 'SNR gain, $G_{SNR}$ (dB)',
                        'cax_id': 1,
                        'orientation': 'horizontal',
                        # 'location': 'top',
                        'xlabel': 'Signal frequency',
                        'xunit': 'GHz',
                    })
                # RO freqs
                self.plot_dicts[plot_key + "_gain_ro_freqs"] = {
                    'ax_id': 2 + 2 * idx_freq + 0,
                    'fig_id': fig_key,
                    'plotfn': self.plot_vlines,
                    'x': np.array(ro_freqs)/1e9,
                    'ymin': np.min(y),
                    'ymax': np.max(y),
                    'colors': 'white',
                    'linestyles': '-',
                }
                self.plot_dicts[plot_key + "_SNR_ro_freqs"] = {
                    'ax_id': 2 + 2 * idx_freq + 1,
                    'fig_id': fig_key,
                    'plotfn': self.plot_vlines,
                    'x': np.array(ro_freqs)/1e9,
                    'ymin': np.min(y),
                    'ymax': np.max(y),
                    'colors': 'white',
                    'linestyles': '-',
                }