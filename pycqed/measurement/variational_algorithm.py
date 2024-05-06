import numpy as np
import logging

from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import awg_sweep_functions as awg_swf
# import pycqed.analysis_v3 as ana_v3
import pycqed.analysis_v3.processing_pipeline as pp_mod
import pycqed.analysis_v3.helper_functions as hlp_mod
# ana_v3.reload_anav3()
from pycqed.analysis_v2 import timedomain_analysis as tda

log = logging.getLogger(__name__)


class VariationalAlgorithm(qe_mod.QuantumExperiment):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'VariationalAlgorithm'

    def __init__(self, optimize=True, optimizer=None, classified=False, df_name='int_log_det', sweep_points=None, **kw):
        super().__init__(
            classified=classified, df_name=df_name,
            sequence_kwargs=dict(sweep_points=sweep_points), **kw
        )
        self.set_block_and_params()
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
        self.params = [f"angle_{qb.name}" for qb in self.qubits]
        self.block = self.simultaneous_blocks(
            block_name='single_qb_gates',
            blocks=[self.block_from_anything(
                f"Y:angle_{qb.name} {qb.name}", f"rot_{qb.name}")
                for qb in self.qubits],
            block_align='middle',
            destroy=True,
        )

    # here data should be in the flattered shape
    @staticmethod
    def classical_postprocessing(single_shots_per_qb_thresholded,
                                 classical_params=1.0):
        # returns the sum of g state population
        # data shape: dictionary of flattened single shot measurement
        e_state_data = [single_shots_per_qb_thresholded[qbn][:, 1] for qbn
                        in single_shots_per_qb_thresholded.keys()]
        # return shape: (flattened_len,)
        return np.average(e_state_data, axis=0)

    @staticmethod
    def cost_function(cpp_output):
        label = np.ones(cpp_output.shape[0])
        return np.square(cpp_output - label)

    @staticmethod
    def classical_postprocessing_train(e_state_data, classical_params=1.0):
        # returns the sum of e state population over qubits
        # e_state_data shape: (qubits, single_shots)
        # return shape: (flattened_len,)
        return np.sum(e_state_data, axis=0)

    @staticmethod
    def cost_function_analysis(cpp_output):
        label = np.ones(cpp_output.shape[1]) * 2
        output = np.zeros((cpp_output.shape[0], cpp_output.shape[-1]))
        for i in range(cpp_output.shape[0]):
            for j in range(cpp_output.shape[-1]):
                output[i, j] = np.mean(np.square(cpp_output[i, :, j] - label))
        return output

    @staticmethod
    def cost_function_train(cpp_output):
        label = np.ones(cpp_output.shape[0]) * 2
        output = np.mean(np.square(cpp_output - label))
        return output

    @staticmethod
    def cost_function_train_analysis(cpp_output):
        label = np.ones(cpp_output.shape[-1]) * 2
        output = np.mean(np.square(cpp_output - label), axis=1)
        return output

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
            pp.add_node('classify_gm', keys_in='raw',
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

        data = {qb.name: np.vstack(
            (pp.data_dict[qb.name]['classify_gm']['pg'], pp.data_dict[qb.name][
                'classify_gm']['pe'], pp.data_dict[qb.name]['classify_gm'][
                'pf'])
        ).T for qb in meas_objs}
        # Shape at this point:
        # {qb.name: flattened three state readout}
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
                 training_settings, ):
        self._set_optimizer_function(optimizer_function, optimizer_kw)
        self._set_cost_function(cost_function)
        self.training_settings = training_settings
        self.measurement_function = None
        # FIXME maybe this should not be called sweep_points
        self.sweep_points = []

    def __call__(self, fun, **kw):
        # in MeasurementControl.measure_soft_adaptive:
        # self.adaptive_function(self.optimization_function, **self.af_pars)
        self.measurement_function = fun
        result = self.optimizer_function(self._full_circuit,
                                         **self.optimizer_kw)
        # if self.optimizer_callback is not None:
        #     result = self.optimizer_callback(result)
        return {'opt_result': result, 'sweep_points': self.sweep_points}

    def _full_circuit(self, params):
        self.sweep_points.append(params)
        all_params, batch_shape, targets = self.get_batch_params(params)
        data = self.measurement_function(all_params)
        # shape = [len(mobj), n_shots, *data_shape]
        data = VariationalAlgorithm.classical_postprocessing(data)
        cost = self.cost_function(data)
        # FIXME: consider batch size
        return np.average(cost, axis=0)

    def get_batch_params(self, trainable_params_values):
        """

        This is the only method which knows about the format/shape of both
        training_settings and data (FIXME for now mean_square_error also does)

        Args:
            params:

        training_settings = {  TODO should these belong to the QE?
            'params': [''],  TODO unused
            'trainable_params': int,  # Could be generalised to a list of
            bool of the same length as 'params'. For now, this method
            assumes that params are ordered (fixed then trainable). This is
            used in the list comprehension.
            'fixed_params_values': [[x0, x1 ...] ...],
            'out_targets': [y ...],  # corresponding target outputs
            TODO unused. Use, and generate random choice if None?
            'trainable_params_init_values': [x0, x1 ...],
        }

        Returns:

        """
        fixed_params_values = self.training_settings.get('fixed_params_values')
        if fixed_params_values is None:
            fixed_params_values = [[]]
        trainable_params_values = np.atleast_2d(trainable_params_values)
        out_targets = self.training_settings['out_targets']
        params_values = np.array([
            [
                np.append(vf, vt)
                for vf in fixed_params_values
            ] for vt in trainable_params_values
        ])
        # Shape at this point: (
        #  number of sets of fixed params,
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
        # shape: [mobj, shots, trainable pars, fixed pars]
        vals = np.average(vals, (0, 1))
        vals = np.array([(val-targets)**2 for val in vals])
        vals = np.average(vals, 1)
        vals = np.reshape(vals, (-1, 1))  # TODO why 2nd dim needed?
        return vals
