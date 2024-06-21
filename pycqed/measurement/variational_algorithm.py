import numpy as np
import logging
import h5py
import traceback

from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import awg_sweep_functions as awg_swf
# import pycqed.analysis_v3 as ana_v3
import pycqed.analysis_v3.processing_pipeline as pp_mod
import pycqed.analysis_v3.helper_functions as hlp_mod
# ana_v3.reload_anav3()
from pycqed.analysis_v2 import timedomain_analysis as tda
import pycqed.measurement.sweep_points as sp_mod
from pycqed.analysis_v2.timedomain_analysis import (
    VariationalAlgorithmAnalysis as vaa)

log = logging.getLogger(__name__)


class VariationalAlgorithm(qe_mod.QuantumExperiment):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'VariationalAlgorithm'
    # Sp will be stored at the end by MC from the result of the optimiser
    # Apparently if MC stores them once at the beginning in MC.run,
    # this prevents them from being overriden in the file
    _metadata_params = {'cal_points', 'channel_map', 'meas_objs'}

    def __init__(self, optimize=True, optimizer=None,
                 classified=False, df_name='int_log_det',
                 sweep_points=None, fixed_params_values=None, **kw):
        self.default_experiment_name += '_opt' if optimize else ''
        try:
            self.optimize = optimize
            if self.optimize:
                sweep_points = None
                if None in [optimizer]:
                    raise ValueError("optimizer not provided")
            else:
                if sweep_points is None:
                    raise ValueError('No sweep points')

            super().__init__(
                classified=classified, df_name=df_name,
                sequence_kwargs=dict(sweep_points=sweep_points), **kw
            )
            self.set_block_and_params()
            self.resolve_fixed_block_params(fixed_params_values)
            self.exp_metadata.update({
                'predict_proba': True,
                'rotate': False,
                'thresholding': True,
                # 'meas_obj_sweep_points_map': self.sweep_points.get_meas_obj_sweep_points_map(
                #     [qb.name for qb in self.meas_objs]),
                'data_to_fit': {},  # FIXME understand why this is needed
                'training_settings': optimizer.training_settings,
                'optimize': self.optimize,
                'qb_names': self.qb_names,  # FIXME needed?
            })

            if self.optimize:
                self.optimizer = optimizer  # TODO or pass kw and instantiate here?
                self.sweep_functions = [awg_swf.BlockSoftHardSweep(
                    self,
                    self.params,
                    block=self.block,
                    parameter_name='Iteration',
                    sweep_kwargs=kw.get('sweep_kwargs', {})
                    )]
                self.mc_mode = 'adaptive'
                self.mc_store_sweep_indices = True
                self.force_2D_sweep = False  # TODO is this needed?
                self.mc_points = [[0]]
                self.sequences = []  # Will be filled by the SF
                self._set_MC()  # Used in the next line
                # TODO check usage and possibly modify
                self.MC.set_adaptive_function_parameters(dict(
                    adaptive_function=self.optimizer,
                    data_processing_function=self._data_processing_function,
                    indexed_sweep=True,
                ))
                self.exp_metadata.update({
                    'hybrid': self.optimizer.hybrid,
                    'plot_raw_data': False,
                    'optim_param_names': self.params,
                })
            else:
                self.exp_metadata.update({
                    # add sweep_points here (removed from _metadata_params)
                    'sweep_points': self.sweep_points,
                    'meas_obj_sweep_points_map':
                        self.sweep_points.get_meas_obj_sweep_points_map(
                            [qb.name for qb in self.meas_objs]),
                })
                self.sequences, self.mc_points = self.sweep_n_dim(
                    sweep_points, body_block=self.block, **kw)

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def set_block_and_params(self):
        self.params = [f"prep_{qb.name}" for qb in self.qubits]
        self.params += [f"angle_{qb.name}" for qb in self.qubits]
        state_prep_block = self.simultaneous_blocks(
            block_name=f'state_prep',
            blocks=[self.block_from_anything(
                f"Y:prep_{qb.name} {qb.name}", f"prep_{qb.name}")
                for qb in self.qubits],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
        )
        single_qb_gates_block = self.simultaneous_blocks(
            block_name='single_qb_gates',
            blocks=[self.block_from_anything(
                f"Y:angle_{qb.name} {qb.name}", f"rot_{qb.name}")
                for qb in self.qubits],
            block_align='middle',
            destroy=True,
        )
        self.block = self.sequential_blocks('VQA',
                                            [state_prep_block,
                                             single_qb_gates_block],
                                            set_end_after_all_pulses=True,
                                            destroy=True)

    def resolve_fixed_block_params(self, fixed_params_values=None):
        """Partially resolves self.block, to set params which are not swept

        Args:
            fixed_params_values: dict of the form {'param_name': value,}
        """
        if fixed_params_values is None:
            return
        # Convert into sweep_points format
        sweep_dicts_list = sp_mod.SweepPoints()
        for key, val in fixed_params_values.items():
            sweep_dicts_list.add_sweep_parameter(key, [val])
        # Update self.block: fix parameters contained in sweep_dicts_list
        self.block.pulses = self.block.pulses_sweepcopy(sweep_dicts_list, [0])
        self.params = [p for p in self.params if p not in fixed_params_values]

    # here data should be in the flattened shape
    # @staticmethod
    # def classical_postprocessing(shots):
    #     raise ValueError("Refactor and move to tda!")
    #     # TODO maybe discard f state
    #     return shots
    #
    # @staticmethod
    # def cost_function(cpp_output, targets):
    #     raise ValueError("Refactor and move to tda!")
    #     # cpp_output: (n_shots, trainable params, non trainable params)
    #     targets = np.array(targets)
    #     cost_func = np.average(
    #         np.array([
    #             [np.mean((row-targets)**2) for row in single_sweep] for
    #             single_sweep in cpp_output
    #         ]),
    #         axis=0,
    #     )
    #     # cost_func shape: (trainable parameter number in one batch,) or scalar
    #     # reshape cost_func to 2D: for EGO
    #     return cost_func.reshape((-1, 1))

    # @staticmethod  # FIXME?
    def _data_processing_function(self, vals,
                                  dset=None  # TODO remove
                                  ):

        meas_objs = self.meas_objs  # Only non-static variable

        # FIXME this won't exist when running a separate analysis offline
        classifier_params = {mobj.name: mobj.acq_classifier_params()
                             for mobj in meas_objs}
        # # FIXME using these as a hack for now
        # classifier_params = hlp_mod.get_clf_params_from_hdf_file(
        #     '20240415_182839', [mobj.name for mobj in meas_objs])
        # Could do readout correction here:
        # state_prob_mtxs = qb.acq_state_prob_mtx() ...

        probability_states = ['pg', 'pe', 'pf']

        # Setup pipeline
        pp = pp_mod.ProcessingPipeline()
        # FIXME: creating a dummy meas_obj_value_names_map since this is
        #  only used in the first node to re-extract the data (keys_in='raw')
        #  Can this create any problems? How to re-run offline?
        movnm = {mobj.name: [f'{mobj.name}_{i}' for i in range(2)]  # I,Q
                 for mobj in meas_objs}
        for mobj in meas_objs:
            # FIXME get this from the metadata instead?
            reset_reps = mobj.reset.feedback.repetitions() if hasattr(
                mobj.reset, 'feedback') else 1
            pp.add_node('filter_data', keys_in='raw',
                        data_filter=lambda x: x[reset_reps::reset_reps+1],
                        meas_obj_names=mobj.name)
            pp.add_node('classify_gm', keys_in='previous',
                        keys_out=[f'{mobj.name}.classify_gm.{ps}'
                                  for ps in probability_states],
                        clf_params=classifier_params.get(mobj.name, None),
                        meas_obj_names=mobj.name)

            pp.add_node('do_postselection_f_level', keys_in='previous',
                        keys_out=[f'{mobj.name}.post_selected'],
                        meas_obj_names=mobj.name)
        pp.resolve(meas_obj_value_names_map=movnm)

        # Run pipeline with raw data
        vals = np.atleast_2d(vals)
        # Construct an initial data dict with the raw data (vals, TODO rename)
        # data_dict = { TODO
        data_dict = {
            mobj.name: {
                movnm[mobj.name][ch_i]: vals[:, 2*mobj_i+ch_i]
                for ch_i in [0, 1]
            } for mobj_i, mobj in enumerate(meas_objs)
        }
        pp.run(data_dict, overwrite_data_dict=True)

        # data shape: {qb.name: flattened three state readout}
        data = {qb.name: np.array([v for v in pp.data_dict[qb.name][
            'classify_gm'].values()]).T for qb in meas_objs}
        # Transpose: to have the 3 states as last dimension
        # TODO write exact shape of the data here
        return data

    def _prepare_sequences(self, sequences=None, sequence_function=None,
                           sequence_kwargs=None):
        """Preparing the sequences is taken care of by the `BlockSoftHardSweep` sweep function.
        """
        # FIXME: this means that the logic in QuantumExperiment._prepare_sequences
        #  cannot be used here
        pass

    def run_analysis(self, analysis_class=None, analysis_kwargs=None, **kw):
        if analysis_class is None:
            analysis_class = tda.VariationalAlgorithmAnalysis
        return super().run_analysis(analysis_class=analysis_class,
                                    analysis_kwargs=analysis_kwargs, **kw)


class VariationalAlgorithmCZ(VariationalAlgorithm):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'VariationalAlgorithmCZ'

    def set_block_and_params(self):
        self.params = [f"prep_{qb.name}" for qb in self.qubits]
        self.params += ['theta']
        state_prep_block = self.simultaneous_blocks(
            block_name=f'state_prep',
            blocks=[self.block_from_anything(
                f"Y:prep_{qb.name} {qb.name}", f"prep_{qb.name}")
                for qb in self.qubits],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
        )
        # in combination the cz block is a controlled-Y180 gate
        cz_gate_block = self.block_from_ops(
            block_name=f'cz_gate',
            operations=[
                f'X90 {self.qubits[1].name}',
                f'CZ:theta {self.qubits[0].name} {self.qubits[1].name}',
                f'X270 {self.qubits[1].name}',
            ],
        )
        self.block = self.sequential_blocks('VQACZ',
                                            [state_prep_block,
                                             cz_gate_block],
                                            set_end_after_all_pulses=True,
                                            destroy=True)


class QCNN4(VariationalAlgorithm):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'QCNN_4_qubit'

    def __init__(self, prep_params_filename=None, *args, **kw):
        self.prep_params_filename = prep_params_filename
        super().__init__(*args, **kw)

    def pp(self, h_index, param_index):
        """Get a preparation parameter

        Short name for convenience when using in an op code
        """
        # h_index = 0 ~ 20, parameters from h5 file, h = 0 ~ 2
        # h_index = 21, prepare TP state with explicit parameters below
        # h_index = 22, prepare |0000> state by setting all params to 0
        if h_index == 21:
            gs_prep_param = np.array([0,np.pi/2,np.pi/2,0,\
                       np.pi,\
                       -np.pi/2,0,np.pi/2,-np.pi/2,\
                       np.pi,\
                        np.pi,\
                       -np.pi,np.pi/2,-np.pi/2,0]) * 180 / np.pi
            return gs_prep_param[param_index]
        elif h_index == 22:
            gs_prep_param = np.zeros(15)
            return gs_prep_param[param_index]
        if not hasattr(self, 'prep_params_vs_h'):
            if self.prep_params_filename is None:
                raise ValueError("self.prep_params_filename is None!")
            try:
                with h5py.File(self.prep_params_filename, 'r') as fileObject:
                    self.prep_params_vs_h = np.array(fileObject['angles_opt'])*180/np.pi
            except FileNotFoundError:
                log.warning("Can't find prep params file! Using zeros instead")
                self.prep_params_vs_h = np.zeros((21, 15))
        param_index = [3,0,1,2,4,8,5,6,7,9,10,14,11,12,13][param_index]
        h_index = int(round(h_index))
        return self.prep_params_vs_h[h_index, param_index]

    def _add_ry_block(self, prefix, qbns, params=None):
        if params is None:
            params = [f"{prefix}_{qbn}" for qbn in qbns]
        self.params += [p for p in params if isinstance(p, str)]
        op_code_params = [':'+p if isinstance(p, str) else p for p in params]
        self._blocks.append(self.simultaneous_blocks(
                block_name=prefix,
                blocks=[self.block_from_anything(
                    f"Y{op_code_params[i]} {qbns[i]}",
                    f"{prefix}_{qbns[i]}")
                    for i in range(len(qbns))],
                block_align='middle',
                set_end_after_all_pulses=True,
                destroy=True,
            ))

    def _add_cz_block(self, prefix, qubit_lists, params=None):
        if params is None:
            params = [f"{prefix}_{qbns[0]}_{qbns[1]}"
                       for i, qbns in enumerate(qubit_lists)]
        self.params += [p for p in params if isinstance(p, str)]
        op_code_params = [':'+p if isinstance(p, str) else p for p in params]
        self._blocks.append(self.simultaneous_blocks(
            block_name=prefix,
            blocks=[
                self.block_from_ops(
                    block_name=f'{prefix}_{qbns[0]}_{qbns[1]}',
                    operations=[f'CZ{op_code_params[i]} {qbns[0]} {qbns[1]}']
                ) for i, qbns in enumerate(qubit_lists)
            ],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
            ))

    def set_block_and_params(self): # TODO call the super!
        self._blocks = []
        self.params = []

        if len(self.qubits) == 2:
            self._add_ry_block('RYp1', range(len(self.qubits)),
                               [90, '[theta_p]/2'])
            self._add_cz_block('CZp1', [[0, 1]],
                               [180])
            self._add_ry_block('RYp2', range(len(self.qubits)),
                                [0, '[basis]',])
        elif len(self.qubits) == 4:
            op_code = "cb.pp([h_index],{i})"
            # Prep circuit. Only one parameter determines whether to prepare
            # the all zero state (theta_p=0) or the ground state (theta_p=180)
            # TODO maybe remove prefix if not needed
            self._add_ry_block('RYp1', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [0, 1, 2, 3]])
            self._add_cz_block('CZp1', [[1, 2]],
                               [op_code.format(i=4)])
            self._add_ry_block('RYp2', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [5, 6, 7, 8]])
            self._add_cz_block('CZp2', [[2, 3], [0, 1]],
                               [op_code.format(i=i) for i in [9, 10]])
            self._add_ry_block('RYp3', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [11, 12, 13, 14]])
            # QCNN. Each gate has an independent parameter.
            self._add_ry_block('RY1', range(len(self.qubits)),
                               ['RY1_0', 'RY1_1', 'RY1_2', 'RY1_3'])
                               # ['theta_b', 'theta_b', 'theta_b', 'theta_b'])
            self._add_cz_block('CZ1', [[0, 1], [2, 3]], ['CZ1', 'CZ2'])
            self._add_ry_block('RY2', range(len(self.qubits)),
                               ['RY2_0', 'RY2_1', 'RY2_2', 'RY2_3'])
            self._add_cz_block('CZ3', [[1, 2]], ['CZ3'])
            self._add_ry_block('RY3', range(len(self.qubits)),
                               ['RY3_0', 'RY3_1', 'RY3_2', 'RY3_3'])
        elif len(self.qubits) == 9:
            pass  # TODO
        elif len(self.qubits) == 1:
            self._add_ry_block('RYp', [qb.name for qb in self.qubits],
                               ['theta_p'])
            self._add_ry_block('RYt', [qb.name for qb in self.qubits],
                               ['theta_t'])
        else:
            raise ValueError("Only 4 or 9 qubits are supported!")
        self.params = [self._parse_param(p)[0] for p in self.params]
        _, idx = np.unique(self.params, return_index=True)
        self.params = list(np.array(self.params)[np.sort(idx)])
        self.block = self.sequential_blocks('QCNN',
                                            self._blocks,
                                            set_end_after_all_pulses=True,
                                            destroy=True)

    #
    # def set_block_and_params(self):
    #     self._blocks = []
    #     self.params = []
    #     if len(self.qubits) == 2:
    #         # Prep circuit
    #         self._add_ry_block('RYp1', range(len(self.qubits)))
    #         self._add_cz_block('CZp1', [[0, 1]])
    #         self._add_ry_block('RY2', [0])
    #     self.block = self.sequential_blocks('QCNN',
    #                                         self._blocks,
    #                                         set_end_after_all_pulses=True,
    #                                         destroy=True)


    # def set_block_and_params(self):
    #     self._blocks = []
    #     self.params = []
    #     if len(self.qubits) == 4:
    #         # Prep circuit
    #         self._add_ry_block('RYp1', [0, 1, 2, 3])
    #         # self._add_ry_block('RYp1', [1])
    #         self._add_ry_block('RY1', [0, 1, 2, 3])
    #         # self._add_ry_block('RY1', [1])
    #     else:
    #         raise ValueError("Only 4 or 9 qubits are supported!")
    #     self.block = self.sequential_blocks('QCNN',
    #                                         self._blocks,
    #                                         set_end_after_all_pulses=True,
    #                                         destroy=True)

class QCNNExperiment(VariationalAlgorithm):
    """QuantumExperiment to perform training of the 3qb spin chain QCNN for quantum phase recognition.
    """

    default_experiment_name = 'QCNNExperiment'

    def set_block_and_params(self):

        state_prep_blocks = list()

        state_prep_blocks.append(self.simultaneous_blocks(
            block_name=f'first_layer_state_prep',
            blocks=[self.block_from_anything(op, op) for op in
                    ['Y:prep0 0', 'Y:prep1 1']],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
        ))

        state_prep_blocks.append(
            self.block_from_anything('CZ180 0 1', f'second_layer_state_prep'))

        state_prep_blocks.append(self.simultaneous_blocks(
            block_name=f'third_layer_state_prep',
            blocks=[self.block_from_anything(op, op) for op in
                    ['Y:prep2 0', 'Y:prep3 1', 'Y:prep4 2']],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
        ))

        state_prep_blocks.append(
            self.block_from_anything('CZ180 1 2', f'fourth_layer_state_prep'))

        state_prep_blocks.append(self.simultaneous_blocks(
            block_name=f'fifth_layer_state_prep',
            blocks=[self.block_from_anything(op, op) for op in
                    ['Y:prep5 1', 'Y:prep6 2']],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
        ))

        qcnn_block = self.block_from_ops(
            block_name=f'qcnn',
            operations=['CZ:theta0 0 1',
                        'CZ:theta1 2 1',
                        'Y270 1', ],
        )

        self.block = self.sequential_blocks('VQA',
                                            state_prep_blocks + [qcnn_block],
                                            set_end_after_all_pulses=True,
                                            destroy=True)

        self.params = [f'prep{i}' for i in range(7)]
        self.params += [f'theta{i}' for i in range(2)]


class VQAOptimizer:
    """
    Wrapper

    TODO ensure this can be instantiated and inspected

    TODO clarify which methods are used and in which order:
        MC.measure_soft_adaptive
            adaptive_function = VQAOptimizer
                optimizer_function
                    _full_circuit
                        get_batch_params
                        optimization_function?
                            measurement_function
                            data_processing_function
                        cost_function
                self.callback?
            MC.af_pars['callback']? (passed to optimizer, to remove)

    Args:
        training_settings: settings, in a format understood by get_batch_params
    """

    def __init__(self, optimizer_function, optimizer_kw, cost_function,
                 training_settings, hybrid=False,
                 classical_optimizer_function_name=None,
                 classical_optimizer_kw=None):
        self._set_optimizer_function(optimizer_function, optimizer_kw)
        self._set_cost_function(cost_function)
        self.training_settings = training_settings
        self.measurement_function = None
        self.sweep_points = None
        self.optim_param_values = []
        self.cost_function_values = []
        self.hybrid = hybrid
        self.iterations = 0
        if self.hybrid:
            self._set_classical_optimizer_function(
                classical_optimizer_function_name, classical_optimizer_kw)
            self.classical_params_list = []
            self.classical_params_result = []

    def __call__(self, fun, **kw):
        # in MeasurementControl.measure_soft_adaptive:
        # self.adaptive_function(self.optimization_function, **self.af_pars)
        if self.iterations:
            log.warning("Reusing an optimizer which has previously run! "
                        "We should first reset all relevant parameters here.")
        self.measurement_function = fun
        try:
            self.optimizer_function(self._full_circuit,
                                    **self.optimizer_kw)
        except KeyboardInterrupt:
            log.warning(
                'Caught a KeyboardInterrupt and there is unsaved data. '
                'Trying clean exit to save data.')
        self.cost_function_values = np.concatenate(
            self.cost_function_values, axis=1)  # TODO very confusing
        # cf.shape: (n sets of train. pars (hard), n batches (soft))
        self.optim_param_values = np.array(self.optim_param_values)
        self.optim_param_values = np.swapaxes(
            self.optim_param_values, 0, 2)  # TODO clean up?
        # pv.shape: (
        #   n trainable parameters,
        #   n sets of trainable parameters (hard),
        #   n batches (soft)
        # )
        self.create_sweep_points()
        result_dict = {
            'optim_param_values': self.optim_param_values,
            'cost_function_values': self.cost_function_values,
            'sweep_points': self.sweep_points,
            'batch_shape': np.array(self.batch_shape),
            'targets': self.targets,
        }
        if self.hybrid:
            result_dict.update({
                'classical_params_list': self.classical_params_list,
                'classical_params_result': self.classical_params_result,
            })
        return result_dict

    def _full_circuit(self, params):
        all_params, batch_shape, targets = self.get_batch_params(params)
        data = self.measurement_function(all_params)
        self.iterations += 1
        data = np.array([
            data[key].reshape((-1, *batch_shape, 3)) for key in data.keys()
        ])
        # shape: (n_qb, n_shots, sets of trainable params (batch size),
        #   sets of non trainable params (prep circuit), 3 states)
        # Take the e state probability (now array contains 0s and 1s)
        data = data[..., 1]
        # shape: (n_qb, n_shots, sets_trainable_params, sets_non_trainable)

        # if self.hybrid:
        #     costs = []
        #     for i in range(batch_shape[0]):
        #         data_batch = data[:, :, i, :, :]
        #         # data batch shape: (n_qb, n_shots, n_non_trainable_params,
        #         # 3 states)
        #         # cost below is scalar
        #         cost, classical_params, classical_param = \
        #             self._classical_training(data_batch, targets)
        #         costs.append([cost])
        #         self.classical_params_list.append(np.array(classical_params))
        #         self.classical_params_result.append(np.array(classical_param))
        #     costs = np.array(costs)

        # batch_shape = (sets_trainable, sets_non_trainable)
        costs = self.cost_function(data, targets).reshape((-1, 1))  # TODO
        # cost must be 2D list of values for EGO to work
        # [[value_1], [value_2], ... [value_n_trainable]]
        self.optim_param_values.append(np.atleast_2d(params))
        self.cost_function_values.append(costs)  # Will be concatenated
        self.batch_shape = batch_shape
        self.targets = targets
        return costs

    # def _classical_training(self, data_batch, targets):
    #     # return: cost (scalar)
    #     classical_params = []
    #     # There are two ways to define the cost function. The first one
    #     # calculate the cost function value for each single shot readout and
    #     # then take the average, while the second one take the average of
    #     # the single shot readout result and then calculate the cost function.
    #
    #     # def to_optimize(c_para):
    #     #     # c_para_vector: 1D array conforms the multi-qubit single-shot
    #     #     # readout
    #     #     data_batch_shape = data_batch.shape
    #     #     c_para_vector = np.array(
    #     #       [1-c_para[0], c_para[0], 0, 1-c_para[0], c_para[0], 0]) * 1/2
    #     #     cpp_output = np.zeros((data_batch_shape[1], data_batch_shape[2]))
    #     #     # cpp_output shape = (n_shots, n_non_trainable_params)
    #     #     for i in range(data_batch_shape[1]):
    #     #         for j in range(data_batch_shape[2]):
    #     #             cpp_output[i, j] = np.dot(data_batch[:, i, j, :].reshape(
    #     #                 -1), c_para_vector)
    #     #     cost = np.average(np.array([
    #     #         np.mean((row - targets) ** 2) for row in cpp_output
    #     #     ]), axis=0)
    #     #     classical_params.append(c_para[0])
    #     #     return cost
    #
    #     def to_optimize_(c_para):
    #         # c_para_vector: 1D array conforms the multi-qubit single-shot
    #         # readout
    #         c_para_vector = np.array(
    #             [1-c_para[0], c_para[0], 0, 1-c_para[0], c_para[0], 0]) * 1/2
    #         data_batch_test = np.concatenate(
    #             (data_batch[0], data_batch[1]), axis=-1)
    #         data_batch_test = np.average(data_batch_test, axis=0)
    #         cpp_output = np.matmul(data_batch_test,
    #                                c_para_vector.T).reshape(-1)
    #         cost = np.mean((cpp_output - targets) ** 2)
    #         classical_params.append(c_para[0])
    #         return cost
    #     result = self.classical_optimizer_function(to_optimize_,
    #                                             **self.classical_optimizer_kw)
    #     optimized_cost = result.fun
    #     classical_param = result.x
    #     return optimized_cost, classical_params, classical_param

    def get_batch_params(self, trainable_params_values):
        """

        This is the only method which knows about the format/shape of both
        training_settings and data (FIXME for now mean_square_error also does)

        Args:
            trainable_params_values: TODO

        training_settings = {  TODO should these belong to the QE?
            'params': [''],
            'trainable_params': int,  # Could be generalised to a list of
            bool of the same length as 'params'. For now, this method
            assumes that params are ordered (non trainable then trainable).
            This is used in the list comprehension.
            'non_trainable_params_values': [[x0, x1 ...] ...],
            'targets': [y ...],  # corresponding target outputs
            TODO unused. Use, and generate random choice if None?
            'trainable_params_init_values': [x0, x1 ...],
        }

        Returns:

        """
        trainable_params_values = np.atleast_2d(trainable_params_values)
        non_trainable_params_values = self.training_settings.get(
            'non_trainable_params_values', [[]])
        non_trainable_params_values = np.atleast_2d(
            non_trainable_params_values)
        targets = np.array(self.training_settings['targets'])
        assert len(targets.shape) == 1, "targets is expected to be 1D"
        assert targets.shape[0] == non_trainable_params_values.shape[0], \
            ("Inconsistent shape of targets and non_trainable_params_values! "
             "While the measurement should run, this might indicate that the "
             "measurement has not been configured properly. For now we "
             "prevent this possibility to keep things simple.")
        params_values = np.array([
            [
                np.append(vnt, vt)
                for vnt in non_trainable_params_values
            ] for vt in trainable_params_values
        ])
        # Shape at this point: (
        #  number of sets of trainable params,
        #  number of sets of non trainable params,
        #  number of params (= number of parametrised gates)
        # )
        # Extract the first 2 dimensions: this is the real shape of the data
        # (which will be returned flattened by the experiment, see next line).
        batch_shape = params_values.shape[:-1]
        # Flatten the first 2 dimensions, to iterate jointly over vf and vt
        # in the experiment (single sweep). The last dimension just
        # corresponds to the number of params, which are swept jointly.
        params_values = params_values.reshape(-1, params_values.shape[-1])
        return params_values, batch_shape, targets

    def create_sweep_points(self):
        # Create a posteriori sweep points based on the optimisation run
        self.sweep_points = sp_mod.SweepPoints()
        self.sweep_points.add_sweep_parameter(
            'optimizer_hard_sweep_index',
            np.array(range(np.prod(self.batch_shape))),
        )
        self.sweep_points.add_sweep_dimension()
        self.sweep_points.add_sweep_parameter(
            'optimizer_soft_sweep_index',
            np.array(range(self.iterations)),
        )

    def _set_classical_optimizer_function(self,
                                          classical_optimizer_function_name,
                                          classical_optimizer_kw):
        assert classical_optimizer_function_name == 'scipy'  #  FIXME
        from scipy.optimize import minimize
        self.classical_optimizer_function = minimize
        self.classical_optimizer_kw = classical_optimizer_kw

    def _set_optimizer_function(self, optimizer_function, optimizer_kw):
        self.optimizer_function = None
        self.optimizer_callback = None
        if callable(optimizer_function):
            self.optimizer_function = optimizer_function
        elif isinstance(optimizer_function, str):
            if optimizer_function == 'scipy':
                from scipy.optimize import minimize
                self.optimizer_function = minimize
                self.optimizer_kw = optimizer_kw
            elif optimizer_function == 'ego':
                # TODO could use to filter what ends up in MC.adaptive_result
                # def callback(opt_result):
                #     x_opt, y_opt, _, x_data, y_data = opt_result
                #     self.optimizer_result = dict(x_opt=x_opt, y_opt=y_opt,
                #                                  x_data=x_data, y_data=y_data)
                #     return x_opt
                # self.optimizer_callback = callback
                from smt.applications import EGO
                if 'xlimits' in optimizer_kw:
                    xlimits = optimizer_kw.pop('xlimits')
                    # needed to specify bounds on parameters
                    from smt.surrogate_models import KRG, DesignSpace
                    optimizer_kw['surrogate'] = KRG(
                        design_space=DesignSpace(xlimits),
                        print_global=False)
                # Here the kw are used to instantiate the optimiser
                ego = EGO(**optimizer_kw)
                self.ego = ego  # FIXME store this somewhere
                self.optimizer_function = ego.optimize
                self.optimizer_kw = {}
            elif optimizer_function == 'evolutionary':
                def _evolutionary_strategy(cost_function, angles_init, npop,
                                           sigma, rate, Nsteps):
                    length = len(angles_init)

                    angles_ev = np.zeros([Nsteps + 1, length])
                    angles_ev[0] = angles_init  # initial guess

                    for i in range(Nsteps):
                        seed = sigma * np.random.randn(npop-1, length)
                        # TODO comment
                        seed = np.concatenate((np.zeros((1, length)), seed),
                                              axis=0)
                        angles_try = angles_ev[i] + seed

                        # cost = Parallel(n_jobs = num_cores)(delayed(\
                        # cost_function)(angles) for angles in angles_try)
                        cost = cost_function(angles_try).reshape(npop)
                        # cost = np.array(
                        #     [cost_function(angles) for angles in
                        #      angles_try]).reshape(npop)

                        cost_diff = (cost - np.mean(cost)) / np.std(cost)
                        gradient = rate * np.dot(seed.T, cost_diff) / npop

                        angles_ev[i + 1] = angles_ev[i] - gradient
                        print(f'iteration: {i}/{Nsteps}', ', cost:',
                              np.mean(cost))
                self.optimizer_function = _evolutionary_strategy
                self.optimizer_kw = optimizer_kw
        if self.optimizer_function is None:
            raise ValueError

    def _set_cost_function(self, cost_function):
        if callable(cost_function):
            self.cost_function = cost_function
        elif cost_function == 'binary_cross_entropy':
            self.cost_function = vaa.cpp_bxe_cost_function
        elif isinstance(cost_function, str):
            self.cost_function = getattr(self, cost_function)
        else:
            raise ValueError

    # TODO in cost_functions.py? Or keep here and delete that module?
    # TODO this is currently specific to the application (which dimensions
    #  to average and reshape on). How to make this generic and integrate in
    #  the rest of the framework?
    @staticmethod
    def mean_square_error(vals, targets):
        # shape: [mobj, shots, trainable pars, non trainable pars]
        vals = np.average(vals, (0, 1))
        vals = np.array([(val-targets)**2 for val in vals])
        vals = np.average(vals, 1)
        vals = np.reshape(vals, (-1, 1))  # TODO why 2nd dim needed?
        # hint for the above question: might because of EGO. See
        # documentation "Usage with parallel options"
        return vals
