import numpy as np
import logging
import h5py
import traceback
from copy import deepcopy

from pycqed.utilities import general as gen
from pycqed.measurement import quantum_experiment as qe_mod
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.analysis_v2 import timedomain_analysis as tda
import pycqed.measurement.sweep_points as sp_mod
from pycqed.analysis_v2.timedomain_analysis import (
    VariationalAlgorithmAnalysis as vaa)
from pycqed.analysis_v3 import helper_functions as hlp_mod

log = logging.getLogger(__name__)


class VariationalAlgorithm(qe_mod.QuantumExperiment):
    """Experiment to train a variational quantum algorithm.

    Base class, meant to be extended when implementing a concrete algorithm.
    Child classes must create self.block, a block parameterised by arbitrarily
    many parameters listed in self.params, see set_block_and_params.

    Args:
        optimize (bool): If False, does a normal measurement with sweep points.
            If True, runs an adaptive measurement, optimising parameters of the
            measurement (hard sweep) at each iteration (soft sweep). Sweep
            points are created a posteriori by the optimiser.
        optimizer (VQAOptimizer): only used if optimize
        fixed_params_values (dict): Values to pre-resolve some of the
            parameters of self.block, such that it is only parameterised by
            the remaining parameters in self.params. Useful if only a few
            parameters are to be swept or optimised over.
        sweep_points: only used if not optimize
        other args: see QuantumExperiment
    """

    default_experiment_name = 'VariationalAlgorithm'

    def __init__(
            self,
            optimize=False,
            optimizer=None,
            fixed_params_values=None,
            df_name='int_log_det',
            sweep_points=None,
            **kw
    ):
        try:
            self.default_experiment_name += '_opt' if optimize else ''
            self.add_default_kw(kw, df_name=df_name)
            self.optimize = optimize
            if self.optimize:
                sweep_points = None
                if None in [optimizer]:
                    raise ValueError("optimizer not provided")
            else:
                if None in [sweep_points]:
                    raise ValueError('No sweep points')

            super().__init__(
                df_name=df_name,
                sequence_kwargs=dict(sweep_points=sweep_points), **kw
            )
            self.set_block_and_params()
            self.resolve_fixed_block_params(fixed_params_values)

            if self.optimize:
                # Sweep points will be stored at the end by MC from the
                # optimiser result. Apparently if MC stores them once at the
                # beginning in MC.run (if they are listed in _metadata_params),
                # this prevents them from being later overridden in the file
                self._metadata_params.remove("sweep_points")
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
                # Needed since there are no 2D sweep points and so on
                self.force_2D_sweep = False
                self.mc_points = [[0]]
                self.sequences = []  # Will be filled by the SF
                self._set_MC()  # Used in the next line
                self.MC.set_adaptive_function_parameters(dict(
                    adaptive_function=self.optimizer,
                    data_processing_function=self._data_processing_function,
                ))
            else:
                self.sequences, self.mc_points = self.sweep_n_dim(
                    self.sweep_points, body_block=self.block, **kw)

            self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_kw(self, kw, df_name=None):
        # Options for simulated experiment
        kw.setdefault('temporary_values', [])
        # Here prepending is equivalent to setdefault
        kw['temporary_values'].insert(0,(
            kw['dev'].instr_mc.get_instr().live_plot_enabled, False))
        if df_name=='sim_int_avg_classif_det':
            default_kw = dict(
                fast_mode=False,
                df_kwargs=dict(
                    qutrit=False,
                    det_get_values_kws=dict(
                        classified=True,
                        correlated=True,
                        thresholded=True,
                        averaged=True,
                    ),
                ),
            )
            gen.setdefault_nested(kw, default_kw)
            if kw['df_kwargs']['det_get_values_kws']['averaged']:
                # Avoid measuring unneeded shots since the detector
                # anyway overwrites the data with a simulation. If not
                # averaged, keep the number of shots (might not be
                # needed for data shapes, but allows the det to know it)
                kw['temporary_values'].insert(0, (kw['dev'].acq_shots, 1))
        kw.setdefault('fast_mode', True)

    def update_metadata(self):
        super().update_metadata()
        # Assuming a MultiPollDetector. predict_proba if not classified
        self.predict_proba = not getattr(
            self.df.detectors[0], 'classified', False)
        self.exp_metadata.update({
            'predict_proba': self.predict_proba,
            'rotate': False,
            'thresholding': self.predict_proba,
            'optimize': self.optimize,
        })
        if self.optimize:
            self.exp_metadata.update({
                'batching_settings': self.optimizer.batching_settings,
                'hybrid': self.optimizer.hybrid,
                'plot_raw_data': False,
                'optim_param_names': self.params,
            })

    def set_block_and_params(self):
        # Minimal basic example with 2 layers of Y gates
        # FIXME should this be an abstract method which adds no blocks,
        #  to be implemented by children? Currently overridden by children
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

    def _extract_variational_param_names(self):
        """Extracts the actual sweep parameters from the parameterised angles

        e.g. ["cb.pp([h_index],{i})", ...] -> ["h_index", ...]
        """
        self.params = [self._parse_param(p)[0] for p in self.params]
        _, idx = np.unique(self.params, return_index=True)
        self.params = list(np.array(self.params)[np.sort(idx)])

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
            assert isinstance(key, str)
            sweep_dicts_list.add_sweep_parameter(key, [val])
        # Update self.block: fix parameters contained in sweep_dicts_list
        self.block.pulses = self.block.pulses_sweepcopy(sweep_dicts_list, [0])
        self.params = [p for p in self.params if p not in fixed_params_values]

    def _data_processing_function(self, vals):

        vals = np.atleast_2d(vals)
        meas_objs = self.meas_objs

        classifier_params = {mobj.name: mobj.acq_classifier_params()
                             for mobj in meas_objs}

        analysis_instructions = self.get_reset_params()[
            'analysis_instructions']
        reset_reps = sum([
            step.get('reset_reps', 0)
            for step in analysis_instructions[meas_objs[0].name]])
        data_filter = lambda x: x[reset_reps::reset_reps + 1]
        vals = data_filter(vals)

        # Construct an initial data dict with the raw data
        value_names = self.df.value_names
        movnm = self.df.get_meas_obj_value_names_map()
        inverse_movnm = {
            vn: mobjn
            for mobjn, vns in movnm.items()
            for vn in vns
        }
        data_dict = {}
        for i, vn in enumerate(value_names):
            qbn = inverse_movnm[vn]
            if qbn not in data_dict:
                data_dict[qbn] = {}
            data_dict[qbn][vn] = vals[:, i:i+1]

        self.pdd = {'meas_results_per_qb_raw': data_dict}
        tda.MultiQubit_TimeDomain_Analysis._process_single_shots(
            pdd=self.pdd,
            n_shots=self.meas_objs[0].acq_shots(),
            predict_proba=self.predict_proba,
            classifier_params=classifier_params,
            thresholding=self.predict_proba,
            preselection_qbs=None,
            preselection=False,
            twoD=True,
            data_filter=lambda x: x,
        )
        return self.pdd

    def _prepare_sequences(self, sequences=None, sequence_function=None,
                           sequence_kwargs=None):
        """Preparing the sequences is done by the `BlockSoftHardSweep` sweep function.
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


class HNNExperiment(VariationalAlgorithm):
    """Experiment to train a variational quantum algorithm.

    The blocks are hard coded at the moment because this was the easiest way to implement parallel
    single-qubit gates during the state preparation. Next step would be to generalize to arbitrary
    parameterized quantum circuits. TODO
    """

    default_experiment_name = 'HNN'

    # TODO could make use of global options here

    def __init__(
            self,
            ts_reload=None,
            optimize=False,  # nsteps
            optimizer=None,
            fixed_params_values=None,
            weights=None,
            prep_params_filename=None,
            do_hnn=True,
            angles_init=None,  # array or number
            v_opt=None,  # dict
            sweep_points=None,
            *args, **kw
    ):
        self.prep_params_filename = prep_params_filename
        self.do_hnn = do_hnn

        if ts_reload:
            # Read trained parameters from training data file in sweep mode
            apd = self._get_apd(ts_reload)
            weights = apd['weights_opt']
            fixed_params_values = apd['params_opt']

        if sweep_points is not None:
            sweep_points = self.update_sweep_points(
                sweep_points,
                fixed_params_values,
            )
        if optimize and optimizer is None:
            if not hasattr(angles_init, '__iter__'):
                if angles_init is None:
                    angles_init = 360
                angles_init = list(v_opt.values()) +\
                    (np.random.random(len(v_opt)) - 0.5) * angles_init
            optimizer = VQAOptimizer(
                optimizer_function='evolutionary',
                optimizer_kw={
                    'angles_init': angles_init,
                    'npop': 10,
                    'sigma': 0.03 * 180,
                    'rate': 1,
                    'nsteps': optimize,
                },
                cost_function='binary_cross_entropy',
                batching_settings={
                    'non_trainable_params_values':
                        # shape = (sets of params, params values)
                        np.reshape(sweep_points['h_index'], (-1, 1)),
                    'trainable_params': len(angles_init),
                    'targets': sweep_points['targets'],
                    'trainable_params_init_values': None,
                }
            )

        super().__init__(
            optimize=optimize,
            optimizer=optimizer,
            fixed_params_values=fixed_params_values,
            weights=weights,  # for the analysis
            do_hnn=do_hnn,
            sweep_points=sweep_points,
            *args, **kw
        )

    def _get_apd(self, timestamp):
        try:
            return hlp_mod.get_param_from_analysis_group(
                timestamp=timestamp, param_name='analysis_params_dict')
        except KeyError:
            # weights_opt doesn't exist yet
            tda.VariationalAlgorithmAnalysis(
                t_start=timestamp, extract_only=True, raise_exceptions=True,
            )
            return self._get_apd(timestamp)

    def update_sweep_points(self, sweep_points, fixed_params_values):
        if fixed_params_values is None:
            fixed_params_values = {}
        sweep_points = deepcopy(sweep_points)
        for sp_dim in sweep_points:
            for sp_name in list(sp_dim):
                vals = sp_dim[sp_name][0]
                if sp_name in fixed_params_values:
                    vals += fixed_params_values.pop(sp_name)
                sp_dim[sp_name] = (
                    vals, sp_dim[sp_name][1], sp_dim[sp_name][2]
                )
        if len(sweep_points)==1:
            sweep_points.append({'dummy': (np.array([0]), '', '')})
        return sweep_points

    def _add_rxy_block(self, prefix, qbns, params=None, rot='Y'):
        # FIXME this kind of functionality could be moved to CircuitBuilder,
        #  e.g. by extending get_pulses.
        #  This would also allow replacing this unnecessary prefix with
        #  e.g. a simple counter.
        # Add a rotation-X or -Y gate to the qubits specified by qbns.
        #
        # Input arguments
        #   prefix: name of the gate parameter
        #   qbns: index of the qubits
        #   params: values of the gate parameter or op_code

        if params is None:
            params = [f"{prefix}_{qbn}" for qbn in qbns]
        qbns = [qbn if isinstance(qbn, str) else self.qubits[qbn].name
                for qbn in qbns]
        self.params += [p for p in params if isinstance(p, str)]
        op_code_params = [':'+p if isinstance(p, str) else p for p in params]
        self._blocks.append(self.simultaneous_blocks(
                block_name=prefix,
                blocks=[self.block_from_anything(
                    f"{rot}{op_code_params[i]} {qbns[i]}",
                    f"{prefix}_{qbns[i]}")
                    for i in range(len(qbns))],
                block_align='middle',
                set_end_after_all_pulses=True,
                destroy=True,
            ))

    def _add_cz_block(self, prefix, qubit_lists, params=None):
        # Add an arbitrary-phase controlled-Z gate to the qubit pairs specified
        # by qubits_lists.
        #
        # Input arguments
        #   prefix: name of the gate parameter
        #   qubits_lists: indices or names of the qubits
        #   params: values of the gate parameter or op_code

        if params is None:
            params = [f"{prefix}_{qbns[0]}_{qbns[1]}"
                       for i, qbns in enumerate(qubit_lists)]
        qubit_lists = [[qbn if isinstance(qbn, str) else self.qubits[qbn].name
                        for qbn in qbns] for qbns in qubit_lists]
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

    def set_block_and_params(self):
        self._blocks = []
        self.params = []

        if len(self.qubits) == 2:
            self._add_rxy_block('RYp1', range(len(self.qubits)),
                               [90, '[theta_p]/2'])
            self._add_cz_block('CZp1', [[0, 1]],
                               [180])
            self._add_rxy_block('RYp2', range(len(self.qubits)),
                                [0, '[basis]',])
        elif len(self.qubits) == 4:
            op_code = "cb.pp([h_index],{i})"
            self._add_rxy_block('RYp1', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [0, 1, 2, 3]])
            self._add_cz_block('CZp1', [[1, 2]],
                               [op_code.format(i=4)])
            self._add_rxy_block('RYp2', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [5, 6, 7, 8]])
            self._add_cz_block('CZp2', [[2, 3], [0, 1]],
                               [op_code.format(i=i) for i in [9, 10]])
            self._add_rxy_block('RYp3', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [11, 12, 13, 14]])
            # temporary X gate to switch measurement bases for state
            # preparation check
            # self._add_rxy_block('RY', range(len(self.qubits)), [90]*4)
            if self.do_hnn:
                # HNN. Each gate has an independent parameter.
                self._add_rxy_block('RY1', range(len(self.qubits)),
                                   ['RY1_0', 'RY1_1', 'RY1_2', 'RY1_3'])
                self._add_cz_block('CZ1', [[0, 1], [2, 3]], ['CZ1', 'CZ2'])
                self._add_rxy_block('RY2', range(len(self.qubits)),
                                   ['RY2_0', 'RY2_1', 'RY2_2', 'RY2_3'])
                self._add_cz_block('CZ3', [[1, 2]], ['CZ3'])
                self._add_rxy_block('RY3', range(len(self.qubits)),
                                   ['RY3_0', 'RY3_1', 'RY3_2', 'RY3_3'])
            else:
                self._add_rxy_block('RYb', range(len(self.qubits)),
                                    ['theta_b']*4)
        elif len(self.qubits) == 9:
            op_code = "cb.pp([h_index],{i})"
            self._add_rxy_block('RYp1', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(0, 9)])
            self._add_cz_block('CZp1', [[2, 3], [5, 6]],
                               [op_code.format(i=i) for i in range(9, 11)])
            self._add_rxy_block('RYp2', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(11, 20)])
            self._add_cz_block('CZp2', [[1, 2], [4, 5], [7, 8]],
                               [op_code.format(i=i) for i in range(20, 23)])
            self._add_rxy_block('RYp3', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(23, 32)])
            self._add_cz_block('CZp3', [[0, 1], [3, 4], [6, 7]],
                               [op_code.format(i=i) for i in range(32, 35)])
            self._add_rxy_block('RYp4', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(35, 44)])
            # temporary X gate to switch measurement bases for state
            # preparation check
            # self._add_rxy_block('RY', range(len(self.qubits)), [90]*9)
            if self.do_hnn:
                # HNN. Each gate has an independent parameter.
                self._add_rxy_block('RY1', range(len(self.qubits)))
                self._add_cz_block('CZ1', [[1, 2], [4, 5], [8, 7]])
                self._add_rxy_block('RY2', range(len(self.qubits)))
                self._add_cz_block('CZ2', [[0, 1], [3, 4], [6, 7]])
                self._add_rxy_block('RY3', range(len(self.qubits)))
                self._add_cz_block('CZ3', [[2, 3], [5, 6]])
                self._add_rxy_block('RY4', range(len(self.qubits)))
            else:
                self._add_rxy_block('RYb', range(len(self.qubits)),
                                    ['theta_b']*9)
        elif len(self.qubits) == 1:
            self._add_rxy_block('RYp', [qb.name for qb in self.qubits],
                               ['theta_p'])
            self._add_rxy_block('RYt', [qb.name for qb in self.qubits],
                               ['theta_t'])
        else:
            raise ValueError("Only 4 or 9 qubits are supported!")
        self._extract_variational_param_names()
        self.block = self.sequential_blocks('HNN',
                                            self._blocks,
                                            set_end_after_all_pulses=True,
                                            destroy=True)
        self.create_pp_vals()

    def create_pp_vals(self):
        """Creates the array containing state preparation parameters

        pp_vals.shape = (n_indices = 28, n_params (depends on n_qubits))
        where the first 21 indices correspond to a sweep over h and 21 to 28
        correspond to trivial states
        TODO maybe find a smart way to remove unused 0 gates in the circuit

        """
        if hasattr(self, 'pp_vals'):
            return
        n_qubits = len(self.qubits)
        # Has to be hardcoded, to allow fallback to 0s if no file below
        n_params = {
            4: 15,
            9: 44,
        }[n_qubits]
        # h_index = 0 ~ 20: parameters from h5 file.
        # For n_qubits = 4 and 9, h = 0 ~ 2 and 0 ~ 1 respectively
        try:
            if self.prep_params_filename is None:
                raise FileNotFoundError
            with h5py.File(self.prep_params_filename, 'r') as fileObject:
                self.pp_vals = np.array(
                    fileObject['angles_opt']) * 180 / np.pi
        except FileNotFoundError:
            log.warning("No prep params file! Using zeros for the h sweep.")
            self.pp_vals = np.zeros((21, n_params))
        # Create the trivial state preparation parameters and append them
        # to self.pp_vals
        # h_index = 21, ..., 28: validation set
        self.pp_vals = np.concatenate((self.pp_vals, np.zeros((8, n_params))))
        # self.pp_vals[21] = 0  # h_index = 21, |0000...>
        self.pp_vals[22][0:n_qubits] = 180  # |1111...>
        self.pp_vals[23][0:n_qubits] = 90  # |++++...>
        self.pp_vals[24][0:n_qubits] = -90  # |----...>
        self.pp_vals[25][1:n_qubits:2] = 180  # |0101...>
        self.pp_vals[26][0:n_qubits:2] = 180  # |1010...>
        self.pp_vals[27][0:n_qubits] = \
            - 2 * (np.arange(n_qubits) % 2 - 1 / 2) * 90  # |+-+-...>
        self.pp_vals[28][0:n_qubits] = \
            2 * (np.arange(n_qubits) % 2 - 1 / 2) * 90  # |-+-+...>

    def pp(self, h_index, param_index):
        """Get a preparation parameter

        Short name for convenience when using in an op code
        """
        # TODO h_index is a misnomer, should rename/clean up
        h_index = int(round(h_index))
        return self.pp_vals[h_index, param_index]


class QCNNExperiment_old(VariationalAlgorithm):
    """3qb spin chain QCNN for quantum phase recognition.
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


class StatePrepExperiment(HNNExperiment):  # TODO

    def set_block_and_params(self):
        self._blocks = []
        self.params = []

        if len(self.qubits) == 1:
            self._add_rxy_block('RY0', range(len(self.qubits)),
                                [90]*9)
        elif len(self.qubits) == 4:
            op_code = "cb.pp([h_index],{i})"
            self._add_rxy_block('RYp1', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [0, 1, 2, 3]])
            self._add_cz_block('CZp1', [[1, 2]],
                               [op_code.format(i=4)])
            self._add_rxy_block('RYp2', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [5, 6, 7, 8]])
            self._add_cz_block('CZp2', [[2, 3], [0, 1]],
                               [op_code.format(i=i) for i in [9, 10]])
            self._add_rxy_block('RYp3', range(len(self.qubits)),
                               [op_code.format(i=i) for i in [11, 12, 13, 14]])
        elif len(self.qubits) == 9:
            op_code = "cb.pp([h_index],{i})"
            self._add_rxy_block('RYp1', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(0, 9)])
            self._add_cz_block('CZp1', [[2, 3], [5, 6]],
                               [op_code.format(i=i) for i in range(9, 11)])
            self._add_rxy_block('RYp2', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(11, 20)])
            self._add_cz_block('CZp2', [[1, 2], [4, 5], [7, 8]],
                               [op_code.format(i=i) for i in range(20, 23)])
            self._add_rxy_block('RYp3', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(23, 32)])
            self._add_cz_block('CZp3', [[0, 1], [3, 4], [6, 7]],
                               [op_code.format(i=i) for i in range(32, 35)])
            self._add_rxy_block('RYp4', range(len(self.qubits)),
                               [op_code.format(i=i) for i in range(35, 44)])
        else:
            raise ValueError("Only 4 or 9 qubits are supported!")
        self._add_rxy_block('RYb', range(len(self.qubits)),
                            ['theta_b']*9)
        self._extract_variational_param_names()
        self.block = self.sequential_blocks('StatePrep',
                                            self._blocks,
                                            set_end_after_all_pulses=True,
                                            destroy=True)
        self.create_pp_vals()


class VQAOptimizer:
    """
    Wrapper for an optimiser, to be used by MeasurementControl in adaptive mode
    FIXME some functionalities still to be extended (e.g. hybrid optimisation)

    Methods call each other as follows (convoluted to keep small changes in MC)
    MC.measure_soft_adaptive
        MC.adaptive_function = self
            self.optimizer_function  # this is the actual optimiser
                self._full_circuit  # f to be optimised (circuit + post-proc.)
                    self.get_batch_params  # prepares measurement params
                    self.measurement_wrapper = MC.measurement_function_wrapper
                        MC.measurement_function  # quantum circuit
                        MC.data_processing_function  # see VariationalAlgorithm
                    self.cost_function  # raw state probabilities -> cost

    Other args:
        optimizer_kw: passed to optimizer_function (some kwargs can be reserved
            to create optimizer_function, see _set_optimizer_function)
        batching_settings: settings, in a format understood by get_batch_params
        hybrid, classical_optimizer_function_name, classical_optimizer_kw: TODO
    """

    def __init__(self, optimizer_function, optimizer_kw, cost_function,
                 batching_settings, hybrid=False,
                 classical_optimizer_function_name=None,
                 classical_optimizer_kw=None):
        self._set_optimizer_function(optimizer_function, optimizer_kw)
        self._set_cost_function(cost_function)
        self.batching_settings = batching_settings
        self.measurement_wrapper = None
        self.sweep_points = None
        self.optim_param_values = []
        self.cost_function_values = []
        self.hybrid = hybrid
        self.iterations = 0
        if self.hybrid:
            self._set_classical_optimizer_function(
                classical_optimizer_function_name, classical_optimizer_kw)
            self.classical_optim_param_values = []
            self.classical_cost_function_values = []

    def __call__(self, fun, **kw):
        # in MeasurementControl.measure_soft_adaptive:
        #   __call__(MC.measurement_function_wrapper, **MC.af_pars)
        if self.iterations:
            log.warning("Reusing an optimizer which has previously run! "
                        "We should first reset all relevant parameters here.")
        self.measurement_wrapper = fun
        try:
            self.optimizer_function(self._full_circuit,
                                    **self.optimizer_kw)
        except KeyboardInterrupt:
            log.warning(
                'Caught a KeyboardInterrupt and there is unsaved data. '
                'Trying clean exit to save data.')
        self.cost_function_values = np.array(self.cost_function_values)
        self.optim_param_values = np.array(self.optim_param_values)
        # These lists were created over iterations (soft sweep) -> reshape
        # to match normal shape of sweep points (hard, soft)
        self.cost_function_values = self.cost_function_values.T
        # cost_function_values.shape: (
        #   n sets of trainable parameters (hard),
        #   n batches (soft)
        # )
        self.optim_param_values = np.swapaxes(self.optim_param_values, 0, 2)
        # optim_param_values.shape: (
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
                'classical_optim_param_values':
                    self.classical_optim_param_values,
                'classical_cost_function_values':
                    self.classical_cost_function_values,
            })
        return result_dict

    def _full_circuit(self, params):
        all_params, batch_shape, targets = self.get_batch_params(params)
        # Same format as analysis.proc_data_dict
        pdd = self.measurement_wrapper(all_params)
        freqs = pdd['meas_results_per_qb']['all_qubits']
        # Turn into list and keep qubit subspace only
        freqs = np.array([
            val for key, val in freqs.items() if not 'f' in key])
        freqs = freqs.reshape((freqs.shape[0], *batch_shape))
        # shape: (n_states, sets of trainable params (batch size),
        #   sets of non trainable params (prep circuit))
        # with batch_shape = (sets_trainable, sets_non_trainable)

        if self.hybrid:  # TODO test
            from qml_training_utils.utils import neural_network_post_processing
            freqs = pdd['TODO']
            cost, weights = neural_network_post_processing(
                freqs, targets, optimize=True)
            self.classical_optim_param_values.append(cost)
            self.classical_cost_function_values.append(weights)

        costs = self.cost_function(freqs, targets)
        self.iterations += 1
        # shape: (sets of trainable params (batch size))
        self.optim_param_values.append(np.atleast_2d(params))
        self.cost_function_values.append(costs)
        self.batch_shape = batch_shape
        self.targets = targets
        return costs

    def get_batch_params(self, trainable_params_values):
        """

        This is the only method which knows about the format/shape of both
        batching_settings and data

        Args:
            trainable_params_values: sweep values tried by the optimiser

        batching_settings = {
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
        non_trainable_params_values = self.batching_settings.get(
            'non_trainable_params_values', [[]])
        non_trainable_params_values = np.atleast_2d(
            non_trainable_params_values)
        targets = np.array(self.batching_settings['targets'])
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
        if callable(optimizer_function):
            self.optimizer_function = optimizer_function
        elif isinstance(optimizer_function, str):
            if optimizer_function == 'scipy':
                from scipy.optimize import minimize
                self.optimizer_function = minimize
                self.optimizer_kw = optimizer_kw
            elif optimizer_function == 'ego':
                # cost must be a 2D list of values for EGO to work
                # [[value_1], [value_2], ... [value_n_trainable]]
                raise NotImplementedError("_full_circuit must return 2D "
                                          "values in order to use EGO!")
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
                                           sigma, rate, nsteps):
                    length = len(angles_init)

                    angles_ev = np.zeros([nsteps + 1, length])
                    angles_ev[0] = angles_init  # initial guess

                    for i in range(nsteps):
                        seed = sigma * np.random.randn(npop-1, length)
                        seed = np.concatenate((np.zeros((1, length)), seed),
                                              axis=0)
                        # seed contains the random step taken by the
                        # optimizer in a single iteration. The first entry
                        # is set to zero to evaluate the current best guess
                        # (angles_ev)
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
                self.optimizer_function = _evolutionary_strategy
                self.optimizer_kw = optimizer_kw
        if self.optimizer_function is None:
            raise ValueError

    def _set_cost_function(self, cost_function):
        if callable(cost_function):
            self.cost_function = cost_function
        elif cost_function == 'binary_cross_entropy':
            self.cost_function = vaa.cpp_bxe_cost_function_training
        elif isinstance(cost_function, str):
            self.cost_function = getattr(self, cost_function)
        else:
            raise ValueError
