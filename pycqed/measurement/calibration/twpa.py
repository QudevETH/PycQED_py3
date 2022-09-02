from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod
from pycqed.analysis_v2 import timedomain_analysis as tda
from copy import deepcopy
import itertools
import numpy as np


class NoisePower(qe_mod.QuantumExperiment):
    """
    Noise PSD measurement for TWPA objects
    """

    def __init__(self, meas_objs, sweep_functions_dict=None, sweep_points=None,
                 **kw):

        assert len(meas_objs) == 1
        super().__init__(
            meas_objs=meas_objs,
            df_name='psd_avg_det',
            force_2D_sweep=True,
            **kw,
        )
        mobj = self.meas_objs[0]

        if sweep_points is None:
            self.sweep_points = sp_mod.SweepPoints([{}, {}])
        else:
            # This should have sp in dimension > 0 only
            self.sweep_points = deepcopy(sweep_points)

        freqs = mobj.instr_acq.get_instr().get_sweep_points_spectrum(
            mobj.acq_length(), mobj.ro_fixed_lo_freq())
        self.sequences = [
            self.seq_from_ops(operations=[f'Acq {mobj.name}'],
                              seq_name='PSD')]
        self.sweep_points.add_sweep_parameter(
            param_name='freq', values=freqs,
            unit='Hz', dimension=0,
        )
        if len(list(self.sweep_points[1])):
            swf_2d = sweep_functions_dict[list(self.sweep_points[1])[0]]
            self.sweep_functions = [self.sweep_functions[0], swf_2d]
            self.mc_points = [freqs, list(self.sweep_points[1].values())[0][0]]
        else:
            self.mc_points = [freqs, []]

        self.exp_metadata.update({
             'meas_obj_sweep_points_map':
                 self.sweep_points.get_meas_obj_sweep_points_map(
                     self.meas_objs),
             'cal_points': repr(cp_mod.CalibrationPoints([mobj.name], []))
        })

        autorun_kwargs = {
            'analysis_class': tda.MultiQubit_TimeDomain_Analysis,
            'analysis_kwargs': {
                'qb_names': [mobj.name],
                'options_dict': {
                    'plot_raw_data': True,
                    # 'data_type': 'singleshot',
                    # 'predict_proba': False,
                    # 'classifier_params': {},
                    'TwoD': True,
                    'plot_proj_data': False,
                    'rotate': False,
                    # 'rotation_type': 'PCA',
                    # 'sweep_points': sweep_points,
                },
            },
        }

        # This for loop jointly sweeps over the sweep points in dims >= 2
        param_names = [list(sp)[0] for sp in sweep_points[2:]]
        array_of_sp_vals = [list(sp.values())[0][0] for sp in
                            sweep_points[2:]]  # except 2 first dims
        data_3D = []
        for idx in itertools.product(*array_of_sp_vals):
            print('Setting sweep functions: ', end='')
            for i, param in enumerate(param_names):
                # sweep the corresponding sweep functions
                sweep_functions_dict[param](idx[i])
                print(f'{sweep_functions_dict[param].name}({idx[i]}) ', end='')
            print()
            self.autorun(**autorun_kwargs)
            qbn = mobj.name
            ana_data_dict = self.analysis.proc_data_dict[
                'meas_results_per_qb_raw'][qbn]
            ch_map = self.analysis.raw_data_dict['exp_metadata'][
                'meas_obj_value_names_map'][qbn]
            spectrum = ana_data_dict[ch_map[0]].T
            data_3D.append(spectrum)  # indexed by [sp>2d, sp_2d, sp_1d]

        len_sp = [len(freqs)] + [len(list(sp.values())[0][0]) for sp in
                                 sweep_points[1:]]
        # indexed by [sp_1d, sp_2d, sp_3d, sp_4d...]
        self.data_ND = np.array(data_3D).T.reshape(len_sp)
