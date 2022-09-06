from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.measurement import sweep_functions as swf
from copy import deepcopy
import itertools
import numpy as np
import traceback


class NoisePower(qe_mod.QuantumExperiment):
    """
    Noise PSD measurement for TWPA objects
    """

    def __init__(self, meas_objs, sweep_functions_dict=None, sweep_points=None,
                 **kw):
        assert len(meas_objs) == 1
        try:
            super().__init__(meas_objs=meas_objs,
                             df_name='psd_avg_det',
                             **kw)
            self.sweep_points = sp_mod.SweepPoints(sweep_points, min_length=2)
            self.sweep_functions_dict = {} if sweep_functions_dict is None \
                                        else sweep_functions_dict
            self.sequences = [self.seq_from_ops(
                operations=[f'Acq {meas_objs[0].name}'], seq_name='PSD',
                ro_kwargs={'qb_names': []},
            )]
            self.autorun()
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def run_measurement(self, **kw):
        mobj = self.meas_objs[0]
        if self.sweep_points.find_parameter('freq') is not None:
            self.sweep_points.remove_sweep_parameter('freq')
        freqs = mobj.instr_acq.get_instr().get_sweep_points_spectrum(
            mobj.acq_length(), mobj.ro_fixed_lo_freq())
        self.sweep_points.add_sweep_parameter(
            param_name='freq', values=freqs,
            unit='Hz', dimension=0,
        )

        for dim in range(1, len(self.sweep_points)):
            swfs = []
            for param in self.sweep_points[dim]:
                if param in self.sweep_functions_dict:
                    swfs.append(swf.Indexed_Sweep(
                        self.sweep_functions_dict[param],
                        self.sweep_points[param]))
            self.sweep_functions[dim] = swf.multi_sweep_function(swfs)

        self.mc_points = [self.sweep_points['freq'],
                          range(self.sweep_points.length(1))]
        self.exp_metadata.update({
             'meas_obj_sweep_points_map':
                 self.sweep_points.get_meas_obj_sweep_points_map(
                     self.meas_objs),
             'cal_points': repr(cp_mod.CalibrationPoints([mobj.name], []))
        })
        super().run_measurement(**kw)

    def run_analysis(self, analysis_kwargs=None, **kw):
        if analysis_kwargs is None:
            analysis_kwargs = {}
        analysis_kwargs.setdefault('qb_names', self.meas_obj_names)
        analysis_kwargs.setdefault('options_dict', {})
        analysis_kwargs['options_dict'].setdefault('plot_raw_data', True)
        analysis_kwargs['options_dict'].setdefault('TwoD', True)
        analysis_kwargs['options_dict'].setdefault('plot_proj_data', False)
        analysis_kwargs['options_dict'].setdefault('rotate', False)
        super().run_analysis(
            analysis_class=tda.MultiQubit_TimeDomain_Analysis,
            analysis_kwargs=analysis_kwargs
        )


class NDimNoisePower(qe_mod.NDimQuantumExperiment):
    QuantumExperiment = NoisePower

    def __init__(self, *args, **kwargs):
        try:
            self.data_ND = {}
            super().__init__(*args, **kwargs)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def run_ndim_analysis(self):
        for idxs, qe in self.experiments.items():
            if qe.analysis is None:
                continue
            if not self.sweep_lengths[0]:
                self.sweep_lengths[0] = qe.sweep_points.length(0)
            for qbn, ana_data_dict in qe.analysis.proc_data_dict[
                    'meas_results_per_qb_raw'].items():
                ch_map = qe.analysis.raw_data_dict['exp_metadata'][
                    'meas_obj_value_names_map'][qbn]
                spectrum = ana_data_dict[ch_map[0]]
                self.data_ND.setdefault(qbn,
                    np.nan * np.ones(self.sweep_lengths))
                for i in range(self.sweep_lengths[0]):
                    for j in range(self.sweep_lengths[1]):
                        self.data_ND[qbn][(i, j, *idxs)] = spectrum[(i, j)]
