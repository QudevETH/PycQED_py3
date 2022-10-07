from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement.calibration import two_qubit_gates as twoqbcal
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod
from pycqed.analysis_v2 import timedomain_analysis as tda
from pycqed.analysis_v2 import base_analysis as ba
from pycqed.measurement import sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
from copy import deepcopy
import numpy as np
import traceback


class NoisePower(twoqbcal.MultiTaskingExperiment):
    """
    Noise PSD measurement for TWPA objects
    """
    task_mobj_keys = ['mobj']

    def __init__(self, meas_objs, sweep_functions_dict=None,
                 **kw):
        try:
            super().__init__(meas_objs=meas_objs,
                             df_name='psd_avg_det',
                             cal_states=[],
                             **kw)

            for task in self.task_list:
                mobj = self.find_qubits_in_tasks(self.meas_objs, [task])
                assert len(mobj) == 1
                mobj = mobj[0]
                sp = deepcopy(task['sweep_points'])
                if sp.find_parameter('freq') is not None:
                    sp.remove_sweep_parameter('freq')
                freqs = mobj.instr_acq.get_instr().get_sweep_points_spectrum(
                    mobj.acq_length(), mobj.ro_fixed_lo_freq())
                sp.add_sweep_parameter(
                    param_name='freq', values=freqs,
                    unit='Hz', dimension=0,
                )
                task['sweep_points'] = sp_mod.SweepPoints(sp, min_length=2)

            self.sweep_functions_dict = {} if sweep_functions_dict is None \
                                        else sweep_functions_dict
            self.preprocessed_task_list = self.preprocess_task_list()

            self.autorun()
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, mobj, **kw):
        return self.block_from_ops(
                block_name="acq_block",
                operations=[f'Acq {mobj}'],
                pulse_modifs={0: dict(element_name='acq_element')},
            )

    def create_meas_objs_list(self, task_list=None, **kw):
        super(twoqbcal.MultiTaskingExperiment, self).create_meas_objs_list(**kw)
        if task_list is None:
            task_list = self.task_list
        if task_list is None:
            task_list = [{}]
        for task in task_list:
            for k in self.task_mobj_keys:
                if not isinstance(m := task.get(k, ''), str):
                    if m not in self.meas_objs:
                        self.meas_objs += [m]
                        self.meas_obj_names += [m.name]

    def get_meas_objs_from_task(self, task):
        return [task['mobj']]

    def run_measurement(self, **kw):
        # Rerun in case params of the measurement objects changed since the init
        self.preprocessed_task_list = self.preprocess_task_list()

        self.sequences, self.mc_points = self.parallel_sweep(
            self.preprocessed_task_list, self.sweep_block, **kw)
        # FIXME is this needed?
        self.mc_points[0] = range(self.preprocessed_task_list[0][
                                      'sweep_points'].length(0))

        # self.generate_sweep_functions()
        for dim in range(1, len(self.sweep_points)):
            self.sweep_functions[dim] = swf.multi_sweep_function(
                [], parameter_name=f"{dim}. dim multi sweep"
            )
            for task in self.preprocessed_task_list:
                param_names = list(task['sweep_points'][dim])
                for param in param_names:
                    if param not in task['sweep_functions_dict']:
                        continue
                    sf = swf.Indexed_Sweep(
                        sweep_function = task['sweep_functions_dict'][param],
                        values=task['sweep_points'][param],
                        name=param,
                        parameter_name=param,
                    )
                    self.sweep_functions[dim].add_sweep_function(sf)

        self.exp_metadata.update({
             'meas_obj_sweep_points_map':
                 self.sweep_points.get_meas_obj_sweep_points_map(
                     self.meas_objs),
             # 'cal_points': repr(cp_mod.CalibrationPoints([mobj.name], []))
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
