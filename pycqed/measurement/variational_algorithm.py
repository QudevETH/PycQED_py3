import numpy as np
import logging

from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import awg_sweep_functions as awg_swf
# import pycqed.analysis_v3 as ana_v3
import pycqed.analysis_v3.processing_pipeline as pp_mod
import pycqed.analysis_v3.helper_functions as hlp_mod
# ana_v3.reload_anav3()
from pycqed.analysis_v2 import timedomain_analysis as tda
import pycqed.measurement.sweep_points as sp_mod

log = logging.getLogger(__name__)


class VariationalAlgorithm(qe_mod.QuantumExperiment):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'VariationalAlgorithm'

    def __init__(self, optimize=True, optimizer=None,
                 classified=False, df_name='int_log_det',
                 sweep_points=None, fixed_params_values=None, **kw):
        # TODO add try except around the whole init
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
            'optimize': optimize,
            'qb_names': self.qb_names,  # FIXME needed?
        })

        if optimize:
            if None in [optimizer]:
                raise ValueError("Not all parameters provided")
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
            self.sequences = [[None]]
            self._set_MC()  # FIXME needed?
            # TODO check usage and possibly modify
            self.MC.set_adaptive_function_parameters(dict(
                adaptive_function=self.optimizer,
                data_processing_function=self._data_processing_function,
                indexed_sweep=True,
            ))
            self.exp_metadata.update({'hybrid': self.optimizer.hybrid})
        else:
            if sweep_points is None:
                raise ValueError('No sweep points')
            self.exp_metadata.update({
                'meas_obj_sweep_points_map':
                    self.sweep_points.get_meas_obj_sweep_points_map(
                        [qb.name for qb in self.meas_objs]),
            })
            self.sequences, self.mc_points = self.sweep_n_dim(
                sweep_points, body_block=self.block, **kw)

        self.autorun()

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

    # here data should be in the flattened shape
    @staticmethod
    def classical_postprocessing(shots):
        raise ValueError("Refactor and move to tda!")
        # TODO maybe discard f state
        return shots

    @staticmethod
    def cost_function(cpp_output, targets):
        raise ValueError("Refactor and move to tda!")
        # cpp_output: (n_shots, trainable params, non trainable params)
        targets = np.array(targets)
        cost_func = np.average(
            np.array([
                [np.mean((row-targets)**2) for row in single_sweep] for
                single_sweep in cpp_output
            ]),
            axis=0,
        )
        # cost_func shape: (trainable parameter number in one batch,) or scalar
        # reshape cost_func to 2D: for EGO
        return cost_func.reshape((-1, 1))

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
        print(f"pp.data_dict[qb.name]['classify_gm'] = "
              f"{pp.data_dict['qb2']['classify_gm']}")

        # data shape: {qb.name: flattened three state readout}
        data = {qb.name: np.array([v for v in pp.data_dict[qb.name][
            'classify_gm'].values()]).T for qb in meas_objs}
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
        return super().run_analysis(analysis_class=analysis_class, analysis_kwargs=analysis_kwargs, **kw)


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

    default_experiment_name = 'VariationalAlgorithmCZ'

    def _add_ry_block(self, prefix, qbns, params=None):
        if params is None:
            params = [f"{prefix}_{qbn}" for qbn in qbns]
        self.params += params
        self._blocks.append(self.simultaneous_blocks(
                block_name=prefix,
                blocks=[self.block_from_anything(
                    f"Y:{params[i]} {qbns[i]}",
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
        self.params += params
        self._blocks.append(self.simultaneous_blocks(
            block_name=prefix,
            blocks=[
                self.block_from_ops(
                    block_name=f'CZ:{prefix}_{qbns[0]}_{qbns[1]}',
                    operations=[f'CZ:{prefix}_{qbns[0]}_{qbns[1]} '
                                f'{qbns[0]} {qbns[1]}']
                ) for i, qbns in enumerate(qubit_lists)
            ],
            block_align='middle',
            set_end_after_all_pulses=True,
            destroy=True,
            ))

    def set_block_and_params(self):
        self._blocks = []
        self.params = []
        if len(self.qubits) == 4:
            # Prep circuit
            self._add_ry_block('RYp1', range(len(self.qubits)),)
                               # ['[theta_prep]', '3*[theta_prep]-90', 0, 0])
            self._add_cz_block('CZp1', [[0, 1]])
            self._add_ry_block('RYp2', range(len(self.qubits)))
            # self._add_cz_block('CZp2', [[1, 2], [0, 3]])
            self._add_cz_block('CZp2', [[1, 2]])
            self._add_cz_block('CZp3', [[0, 3]])
            self._add_ry_block('RYp3', range(len(self.qubits)))
            # QCNN
            self._add_ry_block('RY1', range(len(self.qubits)))
            # # self._add_cz_block('CZ1', [[1, 2], [0, 3]])
            # self._add_cz_block('CZ1', [[1, 2]])
            # self._add_cz_block('CZ1', [[0, 3]])
            # self._add_ry_block('RY2', self.qubits)
            # # self._add_cz_block('CZ2', [[0, 1], [2, 3]])
            # self._add_cz_block('CZ2', [[0, 1]])
            # self._add_cz_block('CZ2', [[2, 3]])
            # self._add_ry_block('RY3', self.qubits)
        elif len(self.qubits) == 9:
            pass  # TODO
        else:
            raise ValueError("Only 4 or 9 qubits are supported!")
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
        # FIXME maybe this should not be called sweep_points
        self.sweep_points = []
        self.cost_function_values = []
        self.hybrid = hybrid
        if self.hybrid:
            self._set_classical_optimizer_function(
                classical_optimizer_function_name, classical_optimizer_kw)
            self.classical_params_list = []
            self.classical_params_result = []

    def __call__(self, fun, **kw):
        # in MeasurementControl.measure_soft_adaptive:
        # self.adaptive_function(self.optimization_function, **self.af_pars)
        self.measurement_function = fun
        result = self.optimizer_function(self._full_circuit,
                                         **self.optimizer_kw)
        # if self.optimizer_callback is not None:
        #     result = self.optimizer_callback(result)
        result_dict = {'opt_result': result, 'sweep_points': self.sweep_points,
                'cost_function_values': self.cost_function_values,
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
        data = np.array([
            data[key].reshape((-1, *batch_shape, 3)) for key in data.keys()
        ])
        # shape: (n_qb, n_shots, n_trainable_params,
        #   n_non_trainable_params, 3 states)
        # Take the e state probability (now array contains 0s and 1s)
        data = data[..., 1]
        # shape: (n_qb, n_shots, n_trainable_params, n_non_trainable_params)
        if self.hybrid:
            costs = []
            for i in range(batch_shape[0]):
                data_batch = data[:, :, i, :, :]
                # data batch shape: (n_qb, n_shots, n_non_trainable_params,
                # 3 states)
                # cost below is scalar
                cost, classical_params, classical_param = \
                    self._classical_training(data_batch, targets)
                costs.append([cost])
                self.classical_params_list.append(np.array(classical_params))
                self.classical_params_result.append(np.array(classical_param))
            costs = np.array(costs)
        else:
            cpp_output = VariationalAlgorithm.classical_postprocessing(data)
            # batch_shape = (n_trainable, n_non_trainable)
            costs = self.cost_function(cpp_output, targets)
        # cost: 2D list of values
        # [[value_1], [value_2], ... [value_n_trainable]]
        self.sweep_points.append(np.atleast_2d(params))
        self.cost_function_values.append(costs)
        return costs

    def _classical_training(self, data_batch, targets):
        # return: cost (scalar)
        classical_params = []
        # There are two ways to define the cost function. The first one
        # calculate the cost function value for each single shot readout and
        # then take the average, while the second one take the average of
        # the single shot readout result and then calculate the cost function.

        # def to_optimize(c_para):
        #     # c_para_vector: 1D array conforms the multi-qubit single-shot
        #     # readout
        #     data_batch_shape = data_batch.shape
        #     c_para_vector = np.array(
        #       [1-c_para[0], c_para[0], 0, 1-c_para[0], c_para[0], 0]) * 1/2
        #     cpp_output = np.zeros((data_batch_shape[1], data_batch_shape[2]))
        #     # cpp_output shape = (n_shots, n_non_trainable_params)
        #     for i in range(data_batch_shape[1]):
        #         for j in range(data_batch_shape[2]):
        #             cpp_output[i, j] = np.dot(data_batch[:, i, j, :].reshape(
        #                 -1), c_para_vector)
        #     cost = np.average(np.array([
        #         np.mean((row - targets) ** 2) for row in cpp_output
        #     ]), axis=0)
        #     classical_params.append(c_para[0])
        #     return cost

        def to_optimize_(c_para):
            # c_para_vector: 1D array conforms the multi-qubit single-shot
            # readout
            c_para_vector = np.array(
                [1-c_para[0], c_para[0], 0, 1-c_para[0], c_para[0], 0]) * 1/2
            data_batch_test = np.concatenate(
                (data_batch[0], data_batch[1]), axis=-1)
            data_batch_test = np.average(data_batch_test, axis=0)
            cpp_output = np.matmul(data_batch_test,
                                   c_para_vector.T).reshape(-1)
            cost = np.mean((cpp_output - targets) ** 2)
            classical_params.append(c_para[0])
            return cost
        result = self.classical_optimizer_function(to_optimize_,
                                                **self.classical_optimizer_kw)
        optimized_cost = result.fun
        classical_param = result.x
        return optimized_cost, classical_params, classical_param

    def get_batch_params(self, trainable_params_values):
        """

        This is the only method which knows about the format/shape of both
        training_settings and data (FIXME for now mean_square_error also does)

        Args:
            trainable_params_values: TODO

        training_settings = {  TODO should these belong to the QE?
            'params': [''],  TODO unused
            'trainable_params': int,  # Could be generalised to a list of
            bool of the same length as 'params'. For now, this method
            assumes that params are ordered (non trainable then trainable).
            This is used in the list comprehension.
            'non_trainable_params_values': [[x0, x1 ...] ...],
            'out_targets': [y ...],  # corresponding target outputs
            TODO unused. Use, and generate random choice if None?
            'trainable_params_init_values': [x0, x1 ...],
        }

        Returns:

        """
        non_trainable_params_values = self.training_settings.get(
            'non_trainable_params_values')
        if non_trainable_params_values is None:
            non_trainable_params_values = [[]]
        trainable_params_values = np.atleast_2d(trainable_params_values)
        out_targets = self.training_settings['out_targets']
        params_values = np.array([
            [
                np.append(vf, vt)
                for vf in non_trainable_params_values
            ] for vt in trainable_params_values
        ])
        # Shape at this point: (
        #  number of sets of non trainable params,
        #  number of sets of trainable params,
        #  number of params (= number of parametrised gates)
        # )
        # Extract the first 2 dimensions: this is the real shape of the data
        # (which will be returned flattened by the experiment, see next line).
        batch_shape = params_values.shape[:-1]
        # Flatten the first 2 dimensions, to iterate jointly over vf and vt
        # in the experiment (single sweep). The last dimension just
        # corresponds to the number of params, which are swept jointly.
        params_values = params_values.reshape(-1, params_values.shape[-1])
        return params_values, batch_shape, out_targets

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
        if self.optimizer_function is None:
            raise ValueError

    def _set_cost_function(self, cost_function):
        if callable(cost_function):
            self.cost_function = cost_function
        elif cost_function == 'va_cost_function':
            self.cost_function = VariationalAlgorithm.cost_function
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
