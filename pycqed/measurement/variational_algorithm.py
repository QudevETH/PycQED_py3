import numpy as np
import logging

import pycqed.measurement.calibration.two_qubit_gates as twoqbcal
from pycqed.measurement import awg_sweep_functions as awg_swf
# import pycqed.analysis_v3 as ana_v3
# import pycqed.analysis_v3.processing_pipeline as pp_mod
# import pycqed.analysis_v3.helper_functions as hlp_mod
# ana_v3.reload_anav3()

log = logging.getLogger(__name__)


class VariationalAlgorithm(twoqbcal.MultiTaskingExperiment):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'VariationalAlgorithm'

    def __init__(self, qubits, task_list=None, sweep_points=None,
                 optimize=True, optimizer=None, **kw):
        super().__init__(**kw)

        self.set_block_and_params()

        if optimize:
            if None in [optimizer]:
                raise ValueError("Not all parameters provided")
            self.optimizer = optimizer  # TODO or pass kw and instantiate here?
            self.sweep_functions = [
                awg_swf.BlockSoftHardSweep(self,
                                           self.params,
                                           block=self.block,
                                           sweep_kwargs=kw.get('sweep_kwargs', {}))
            ]
            self.mc_mode = 'adaptive'
            self.force_2D_sweep = False  # TODO is this needed?
            self.mc_points = [[0]]
            self.sequences = [[None]]
            self._set_MC()
            # TODO check usage and possibly modify
            self.MC.set_adaptive_function_parameters(dict(
                adaptive_function=self.optimizer,
                # TODO either auto-generate here or process in base QE
                data_processing_function=self._data_processing_function,
            ))
        else:
            pass
            # TODO pass sweep_points/task_list to normal super init

        self.autorun()

    def set_block_and_params(self):

        self.params = ['angle0']

        self.block

    @staticmethod
    def _data_processing_function(vals, dset=None):
        timestamp = '20230209_014813'

        meas_obj_names = ['qb2']

        pp = pp_mod.ProcessingPipeline()

        classifier_params = hlp_mod.get_clf_params_from_hdf_file(
            timestamp, meas_obj_names)
        state_prob_mtxs = hlp_mod.get_state_prob_mtxs_from_hdf_file(
            timestamp, meas_obj_names)
        for mobjn, mtx in state_prob_mtxs.items():
            if mtx is None:
                if any(correct_readout):
                    log.warning(f'The acq_state_prob_mtx was not provided '
                                f'for {mobjn}. The acq_state_prob_mtxs '
                                f'must be specified for both qubits in '
                                f'order to perform readout correction.')
                if False in correct_readout:
                    # only do the readout-uncorrected analysis if the user
                    # wanted this originally
                    correct_readout = (False,)
                else:
                    raise Exception

        mobjn = meas_obj_names[0]

        probability_states = ['pg', 'pe', 'pf']

        pp.add_node('classify_gm', keys_in='raw',
                    keys_out=[f'{mobjn}.classify_gm.{ps}'
                              for ps in probability_states],
                    clf_params=classifier_params.get(mobjn, None),
                    meas_obj_names=mobjn)

        pp.add_node('do_postselection_f_level', keys_in='previous',
                    keys_out=[f'{mobjn}.post_selected'],
                    meas_obj_names=mobjn)

        labels = list(training_state_labels.values())
        n_shots = qb2.acq_shots()
        n_segments = len(labels)

        pp.add_node('average_data',
                    shape=(n_shots, n_segments),
                    final_shape=(n_segments),
                    averaging_axis=0,
                    selection_map=None,
                    keys_in='previous',
                    keys_out=[f'{mobjn}.expectation_value'],
                    meas_obj_names=mobjn)

        pp.add_node('mean_squared_error',
                    keys_in='previous',
                    keys_out=[f'{mobjn}.MSE'],
                    sorted_by_label=False,
                    # specifies the order in which the measurements were performed
                    labels=labels,
                    meas_obj_names=mobjn,
                    )

        # WARNING: meas_obj_value_names_map is somewhat hard coded to match
        # the value names generated in IntegratingAveragingPollDetector
        data_type = "raw"
        meas_obj_value_names_map = {qb.name: [
            f'{qb.instr_acq()}_{qb.acq_unit()}_{data_type} w{ch} {qb.instr_acq()}'
            for ch in [qb.acq_I_channel(), qb.acq_Q_channel()]] for i, qb in
                                    enumerate(qcnn_qubits)}

        pp.resolve(meas_obj_value_names_map=meas_obj_value_names_map)

        data_dict = dict()
        vals = np.atleast_2d(vals)
        channels = meas_obj_value_names_map[mobjn]
        data_dict[mobjn] = {channels[0]: vals[:, 0], channels[1]: vals[:, 1]}
        pp.run(data_dict, overwrite_data_dict=True)
        MSE = pp.data_dict[mobjn]['MSE'][
            0]  # remove index `[0]` when using batch sampling (EGO)
        return MSE

    def _prepare_sequences(self, sequences=None, sequence_function=None,
                           sequence_kwargs=None):
        """Preparing the sequences is taken care of by the `BlockSoftHardSweep` sweep function.
        """
        # FIXME: this means that the logic in QuantumExperiment._prepare_sequences
        #  cannot be used here
        pass


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


class VQAOptimizer:
    """
    Wrapper

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
                callback?
            callback?
    Args:
        training_settings: settings, in a format understood by get_batch_params
    """

    def __init__(self, optimizer_function, optimizer_kw, cost_function,
                 training_settings, ):
        self._set_optimizer_function(optimizer_function, optimizer_kw)
        self._set_cost_function(cost_function)
        self.training_settings = training_settings
        self.measurement_function = None

    def __call__(self, fun, **kw):
        self.measurement_function = fun
        return self.optimizer_function(self._full_circuit, self.optimizer_kw)

    def _full_circuit(self, params):
        all_params, targets = self.get_batch_params(params)
        meas = self.measurement_function(all_params)
        cost = self.cost_function(meas, targets)
        return cost

    def get_batch_params(self, trainable_params_values):
        """

        Args:
            params:

        training_settings = {  TODO should these belong to the QE?
            'params': [''],
            'trainable_params': int,  # Could be generalised to a list of
            bool of the same length as 'params'. For now, this method
            assumes that params are ordered (fixed then trainable). This is
            used in the list comprehension.
            'fixed_params_values': [[x0, x1 ...] ...],
            'out_targets': [y ...],  # corresponding target outputs
            'trainable_params_init_values': [x0, x1 ...],  TODO here or in optimizer_kw?
        }

        Returns:

        """
        fixed_params_values = self.training_settings['fixed_params_values']
        trainable_params_values = np.atleast_2d(trainable_params_values)
        out_targets = self.training_settings['out_targets']
        all_params_values = np.array([
            [
                np.append(vf, vt)
                for vf in fixed_params_values
            ] for vt in trainable_params_values
        ])
        return all_params_values, out_targets

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
                def callback(opt_result):
                    log.warning('TODO check what is in opt_result')
                    x_opt, y_opt, _, x_data, y_data = opt_result
                    self.optimizer_result = dict(x_opt=x_opt, y_opt=y_opt,
                                                 x_data=x_data, y_data=y_data)
                    return x_opt
                from smt.applications import EGO
                if 'xlimits' in optimizer_kw:
                    xlimits = optimizer_kw.pop('xlimits')
                    # needed to specify bounds on parameters
                    from smt.surrogate_models import KRG
                    optimizer_kw['surrogate'] = KRG(
                        xlimits=xlimits,
                        print_global=False)
                # Here the kw are used to instantiate the optimiser
                ego = EGO(**optimizer_kw)
                self.optimizer_function = ego.optimize
                self.optimizer_kw = {}
                self.optimizer_callback = callback
        if self.optimizer_function is None:
            raise ValueError

    def _set_cost_function(self, cost_function):
        if callable(cost_function):
            self.cost_function = cost_function
        elif isinstance(cost_function, str):
            self.cost_function = getattr(self, cost_function)
        else:
            raise ValueError

    # TODO in cost_functions.py? Or keep here and delete that module?
    @staticmethod
    def mean_square_error(self, vals, targets):
        return np.mean((vals-targets)**2)
