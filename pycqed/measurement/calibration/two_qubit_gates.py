import numpy as np
from collections import OrderedDict as odict
from copy import copy
from copy import deepcopy
from itertools import zip_longest
import traceback
from pycqed.utilities.general import assert_not_none
from pycqed.measurement.quantum_experiment import QuantumExperiment
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.waveform_control.block import Block, ParametricValue
from pycqed.measurement.waveform_control.segment import UnresolvedPulse
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
import pycqed.measurement.awg_sweep_functions as awg_swf
import pycqed.measurement.sweep_functions as swf
import pycqed.analysis_v2.timedomain_analysis as tda
from pycqed.measurement import multi_qubit_module as mqm
import logging
import qcodes
import pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon\
    as qb_mod

log = logging.getLogger(__name__)

# TODO: docstrings (list all kw at the highest level with reference to where
#  they are explained, explain all kw where they are processed)
# TODO: add some comments that explain the way the code works


class MultiTaskingExperiment(QuantumExperiment):
    """
    This class adds the concept of tasks to the QuantumExperiment class and
    allows to run multiple tasks in parallel. There are no checks whether a
    parallel execution of tasks makes sense on the used hardware
    (connectivity, crosstalk etc.), i.e., it is up to the experimentalist
    to ensure this by passing a reasonable task_list.

    The concept is that each experiment inherited from this class should
    define a method that creates a block based on the parameters specified
    in a task. The method parallel_sweep can then be used to assemble these
    blocks in parallel.

    :param task_list: list of dicts, where each dict contains the parameters of
        a task (= keyword arguments for the block creation function)
    :param dev: device object, see QuantumExperiment
    :param qubits: list of qubit objects, see QuantumExperiment
    :param operation_dict: operations dictionary, see QuantumExperiment
    :param kw: keyword arguments. Some are processed directly in the init or
        the parent init, some are processed in other functions, e.g.,
        create_cal_points. (FIXME further documentation would help). The
        contents of kw are also stored to metadata and thus be used to pass
        options to the analysis.
    """

    default_experiment_name = 'MultitaskingExperiment'
    # The following dictionary can be overwritten by child classes to
    # specify keyword arguments from which sweep_points should be generated
    # automatically (see docstring of generate_kw_sweep_points).
    kw_for_sweep_points = {}
    # The following list can be overwritten by child classes to specify keyword
    # arguments that should be automatically copied into each task (list of
    # str, each being a key to be searched in kw).
    kw_for_task_keys = ()
    # The following list can be overwritten by child classes to specify keys
    # inside tasks that refer to measurement objects. The respective values
    # will then automatically replaced by measurement object names in case
    # the actual objects were provided.
    task_mobj_keys = ()

    @assert_not_none('task_list')
    def __init__(self, task_list, dev=None, qubits=None,
                 operation_dict=None, **kw):

        for task in task_list:
            # convert qubit objects to qubit names
            for k in self.task_mobj_keys:
                if not isinstance(task[k], str):
                    task[k] = task[k].name
            # generate an informative task prefix
            if 'prefix' not in task and len(self.task_mobj_keys):
                task['prefix'] = '_'.join([v for k, v in task.items() if k
                                           in self.task_mobj_keys]) + '_'

        self.task_list = task_list
        # Process kw_for_sweep_points for the global keyword arguments kw
        self.generate_kw_sweep_points(kw)

        # Try to get qubits or at least qb_names
        _, qb_names = self.extract_qubits(dev, qubits, operation_dict)
        # Filter to the ones that are needed
        qb_names = self.find_qubits_in_tasks(qb_names, task_list + [kw])
        # Initialize the QuantumExperiment
        super().__init__(dev=dev, qubits=qubits,
                         operation_dict=operation_dict,
                         filter_qb_names=qb_names, **kw)

        if 'sweep_points' in kw:
            # Note that sweep points generated due to kw_for_sweep_points are
            # already part of kw['sweep_points'] at this point.
            self.sweep_points = kw.pop('sweep_points')

        self.cal_points = None
        self.cal_states = None
        self.exception = None
        self.all_main_blocks = []
        self.data_to_fit = {}

        # The following is done because the respective call in the init of
        # QuantumExperiment does not capture all kw since many are explicit
        # arguments of the init there.
        kw.pop('exp_metadata', None)
        self.exp_metadata.update(kw)

        # Create calibration points based on settings in kw (see docsring of
        # create_cal_points)
        self.create_cal_points(**kw)

        # The following is only relevant for child classes that make use of
        # the sweep_functions_dict and call generate_sweep_functions
        # (see docstring of generate_sweep_functions)
        self.sweep_functions_dict = kw.get('sweep_functions_dict', {})

    def add_to_meas_obj_sweep_points_map(self, meas_objs, sweep_point):
        """
        Add an entry to the meas_obj_sweep_points_map, which will later be
        stored to the metadata. Makes sure to not add entries twice.
        :param meas_objs: (str or list of str) name(s) of the measure
            object(s) for which the sweep_point should be added
        :param sweep_point: (str) name of the sweep_point that should be added
        """
        if 'meas_obj_sweep_points_map' not in self.exp_metadata:
            self.exp_metadata['meas_obj_sweep_points_map'] = {}
        if not isinstance(meas_objs, list):
            meas_objs = [meas_objs]
        for mo in meas_objs:
            # get name from object if an object was given
            mo = mo if isinstance(mo, str) else mo.name
            if mo not in self.exp_metadata['meas_obj_sweep_points_map']:
                self.exp_metadata['meas_obj_sweep_points_map'][mo] = []
            if sweep_point not in self.exp_metadata[
                    'meas_obj_sweep_points_map'][mo]:
                # if the entry does not exist yet
                self.exp_metadata['meas_obj_sweep_points_map'][mo].append(
                    sweep_point)

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects (e.g., qubits) of a task.
        Should be overloaded in child classes if the default behavior
        of returning all qubits found in the task is not desired.
        :param task: a task dictionary
        :return: list of all qubit objects (if available) or names
        """
        return self.find_qubits_in_tasks(self.qb_names, [task])

    def run_measurement(self, **kw):
        """
        Run the actual measurement. Stores some additional settings and
            then calls the respective method in QuantumExperiment.
        :param kw: keyword arguments
        """
        # update the nr_averages based on the settings in the user measure
        # objects
        self.df_kwargs.update(
            {'nr_averages': max(qb.acq_averages() for qb in self.meas_objs)})

        # Store metadata that is not part of QuantumExperiment.
        self.exp_metadata.update({
            'preparation_params': self.get_prep_params(),
            'rotate': len(self.cal_states) != 0 and not self.classified,
            'sweep_points': self.sweep_points,
            'ro_qubits': self.meas_obj_names,
        })
        if len(self.data_to_fit):
            self.exp_metadata.update({'data_to_fit': self.data_to_fit})

        def replace_qc_params(obj):
            obj = copy(obj)
            if isinstance(obj, list):
                ind = range(len(obj))
            elif isinstance(obj, dict):
                ind = obj.keys()
            for i in ind:
                if isinstance(obj[i], (dict, list)):
                    obj[i] = replace_qc_params(obj[i])
                elif isinstance(obj[i], qcodes.Parameter):
                    obj[i] = repr(obj[i])
            return(obj)

        if kw.get('store_preprocessed_task_list', False) and hasattr(
                self, 'preprocessed_task_list'):
            tl = replace_qc_params(self.preprocessed_task_list)
            self.exp_metadata.update({'preprocessed_task_list': tl})
        if self.task_list is not None:
            tl = replace_qc_params(self.task_list)
            self.exp_metadata.update({'task_list': tl})

        super().run_measurement(**kw)

    def create_cal_points(self, n_cal_points_per_state=1, cal_states='auto',
                          for_ef=False, **kw):
        """
        Creates a CalibrationPoints object based on the given parameters and
            saves it to self.cal_points.

        :param n_cal_points_per_state: number of segments for each
            calibration state
        :param cal_states: str or tuple of str; the calibration states
            to measure
        :param for_ef: (deprecated) bool indicating whether to measure the
            |f> calibration state for each qubit
        :param kw: keyword arguments (to allow pass-through kw even if it
            contains entries that are not needed)
        """
        if for_ef:
            log.warning('for_ef is deprecated, use cal_states instead.')
        self.cal_states = CalibrationPoints.guess_cal_states(
            cal_states, for_ef=for_ef)
        self.cal_points = CalibrationPoints.multi_qubit(
            self.meas_obj_names, self.cal_states,
            n_per_state=n_cal_points_per_state)
        self.exp_metadata.update({'cal_points': repr(self.cal_points)})

    def preprocess_task_list(self, **kw):
        """
        Calls preprocess task for all tasks in self.task_list. This adds
        prefixed sweep points to self.sweep_points and returns a
        preprocessed task list, for details see preprocess_task.

        :param kw: keyword arguments
        :return: the preprocessed task list
        """
        # keep a reference to the original sweep_points object
        given_sweep_points = self.sweep_points
        # Store a copy of the sweep_points (after ensuring that they are a
        # SweepPoints object). This copy will then be extended with prefixed
        # task-specific sweep_points.
        self.sweep_points = SweepPoints(given_sweep_points)
        # Internally, all sweeps need to be handled as 2D sweeps (i.e.,
        # _num_sweep_dims = 2) if force_2D_sweep is True. Otherwise,
        # the number of sweep dimensions _num_sweep_dims is the largest
        # number of dimensions in any of the (global or task-specific)
        # SweepPoints objects.
        self._num_sweep_dims = 2 if self.force_2D_sweep else max(
            [len(t.get('sweep_points', [])) for t in self.task_list]
            + [len(self.sweep_points)])
        while len(self.sweep_points) < self._num_sweep_dims:
            self.sweep_points.add_sweep_dimension()
        preprocessed_task_list = []
        for task in self.task_list:
            # preprocessed_task_list requires both the sweep point that
            # should be modified and the original version of the sweep
            # points (to see which sweep points are valid for all tasks)
            preprocessed_task_list.append(
                self.preprocess_task(task, self.sweep_points,
                                     given_sweep_points, **kw))
        return preprocessed_task_list

    def preprocess_task(self, task, global_sweep_points, sweep_points=None,
                        **kw):
        """
        Preprocesses a task, which includes the following actions. The
        original task is not modified, but instead a new, preprocessed task
        is returned.
        - Create or cleanup task prefix.
        - Copy kwargs listed in kw_for_task_keys to the task.
        - Generate task-specific sweep points based on generate_kw_sweep_points
          if the respective keys are found as parameters of the task.
        - Copies sweep points valid for all tasks to the task.
        - Adds prefixed versions of task-specific sweep points to the global
          sweep points
        - Generate a list of sweep points whose names have to be prefixed
          when used as ParametricValue during block creation.
        - Update meas_obj_sweep_points_map for qubits involved in the task

        :param task: (dict) the task
        :param global_sweep_points: (SweepPoints object) global sweep points
            containing the sweep points valid for all tasks plus prefixed
            versions of task-specific sweep points. The object is updated
            by this method.
        :param sweep_points: (SweepPoints object or list of dicts or None)
            sweep points valid for all tasks. Remains unchanged in this method.
        :param kw: keyword arguments
        :return: the preprocessed task
        """
        # copy the task in order to not modify the original task
        task = copy(task)  # no deepcopy: might contain qubit objects
        # Create a prefix if it does not exist. Otherwise clean it up (add "_")
        prefix = task.get('prefix', None)
        if prefix is None:  # try to guess one based on contained qubits
            if self.qb_names:
                prefix = '_'.join(self.find_qubits_in_tasks(self.qb_names,
                                                            [task]))
            else:
                prefix = '_'.join(self.find_qubits_in_tasks(self.meas_obj_names,
                                                            [task]))
        prefix += ('_' if prefix[-1] != '_' else '')
        task['prefix'] = prefix

        # Get measure objects needed involved in this task. Will be used
        # below to generate entries for the meas_obj_sweep_points_map.
        mo = self.get_meas_objs_from_task(task)

        # Copy kwargs listed in kw_for_task_keys to the task.
        for param in self.kw_for_task_keys:
            if param not in task and param in kw:
                task[param] = kw.get(param)

        # Start with sweep points valid for all tasks
        current_sweep_points = SweepPoints(sweep_points)

        # Generate kw sweep points for the task
        self.generate_kw_sweep_points(task)
        # check whether new sweep points increase the total number of sweep
        # dimensions
        if (l := len(task.get('sweep_points', []))) > self._num_sweep_dims:
            self._num_sweep_dims = l
            while len(global_sweep_points) < self._num_sweep_dims:
                global_sweep_points.add_sweep_dimension()

        # Add all task sweep points to the current_sweep_points object.
        # If a task-specific sweep point has the same name as a sweep point
        # valid for all tasks, the task-specific one is used for this task.
        current_sweep_points.update(SweepPoints(task['sweep_points']))
        # Create a list of lists containing for each dimension the names of
        # the task-specific sweep points. These sweep points have to be
        # prefixed with the task prefix later on (in the global sweep
        # points, see below, and when used as ParametricValue during block
        # creation).
        params_to_prefix = [list(d) for d in task['sweep_points']]
        task['params_to_prefix'] = params_to_prefix
        # Save the current_sweep_points object to the preprocessed task
        task['sweep_points'] = current_sweep_points

        while len(current_sweep_points) < self._num_sweep_dims:
            current_sweep_points.add_sweep_dimension()
        while len(params_to_prefix) < self._num_sweep_dims:
            params_to_prefix.append([])
        # for all sweep dimensions
        for gsp, csp, params in zip(global_sweep_points,
                                    current_sweep_points,
                                    params_to_prefix):
            # for all sweep points in this dimension (both task-specific and
            # valid for all tasks)
            for k in csp.keys():
                if k in params and '=' not in k:
                    # task-specific sweep point. Add prefixed version to
                    # global sweep points and to meas_obj_sweep_points_map
                    gsp[prefix + k] = csp[k]
                    self.add_to_meas_obj_sweep_points_map(mo, prefix + k)
                else:
                    # sweep point valid for all tasks.
                    if k not in gsp:
                        # Pulse modifier sweep point from a task. Copy it to
                        # global sweep points, assuming that the expert user
                        # has made sure that there are no conflicts of pulse
                        # modifier sweep points across tasks.
                        gsp[k] = csp[k]
                    # Add without prefix to meas_obj_sweep_points_map
                    self.add_to_meas_obj_sweep_points_map(mo, k)

        # The following is only relevant for child classes that make use of
        # the sweep_functions_dict and call generate_sweep_functions
        # (see docstring of generate_sweep_functions)
        for k, v in task.get('sweep_functions_dict', {}).items():
            # add task sweep functions to the global sweep_functions_dict with
            # the appropriately prefixed key
            self.sweep_functions_dict[prefix + k] = v

        return task

    def generate_sweep_functions(self):
        """Loops over all sweep points and adds the according sweep function to
        self.sweep_functions. The appropriate sweep function is taken from
        self.sweep_function_dict. For multiple sweep points in one dimension a
        multi_sweep is used.

        This method is (for now) not used in this base class, but is only
        provided to allow child classes to make use of it, in which case
        the child class needs to populate self.sweep_function_dict before
        calling this method.

        Caution: Special behaviour if the sweep point param_name is not found in
        self.sweep_function_dict.keys(): We assume that this sweep point is a
        pulse parameter and insert the class SegmentSoftSweep as placeholder
        that will be replaced by an instance in QE._configure_mc.
        """
        # The following dict of lists will store the mapping between
        # (local/global) sweep parameters and (local/global) sweep functions
        class ListsDict(dict):
            def append(self, key, value):
                if key not in self:
                    self[key] = []
                self[key].append(value)
        sf_for_sp = ListsDict()
        # helper lists needed to determine the mapping
        all_sp = [sp for d in self.sweep_points for sp in d]
        prefixes = [t['prefix'] for t in self.preprocessed_task_list]
        # for each sweep function, find for which sweep params it is needed
        for sf in self.sweep_functions_dict.keys():
            if sf in all_sp:  # exact match
                # local swf for local sp or global swf for global sp
                sf_for_sp.append(sf, sf)
            else:
                # it might be a local sweep function for a global sweep point
                sp_p = list(set([(sp, p) for sp in all_sp for p in prefixes
                                 if sf == p + sp]))
                if len(sp_p):
                    if len(sp_p) > 1:
                        raise ValueError(
                            f'Sweep function matches multiple combinations '
                            f'of prefix and sweep parameter: {sp_p}.')
                    sf_for_sp.append(sp_p[0][0], sf)
        # for each sweep param that does not have a sweep function yet
        for sp in all_sp:
            if sp not in sf_for_sp:
                # It might be a global sweep param that is overridden in all
                # tasks, in which case we can ignore it.
                if all([sf_for_sp.get(p + sp, []) for p in prefixes]):
                    sf_for_sp[sp] = []  # no sweep function needed

        # check if the child class has a separate dict for sweeping pulse
        # parameters
        sp_pulses = self.get_sweep_points_for_sweep_n_dim()
        if sp_pulses == self.sweep_points:
            sp_pulses = None  # update of a separate dict not needed below

        # We can now add the needed sweep functions to self.sweep_functions.
        # loop over all sweep_points dimensions
        for i in range(len(self.sweep_points)):
            sw_ctrl = None
            if i >= len(self.sweep_functions):
                # add new dimension with empty multi_sweep_function
                # in case i is out of range
                self.sweep_functions.append(
                    swf.multi_sweep_function(
                        [],
                        parameter_name=f"{i}. dim multi sweep"
                    )
                )
            elif not isinstance(self.sweep_functions[i],
                                swf.multi_sweep_function):
                # refactor current sweep function into multi_sweep_function
                self.sweep_functions[i] = swf.multi_sweep_function(
                    [self.sweep_functions[i]],
                    parameter_name=f"{i}. dim multi sweep"
                )

            for param in self.sweep_points[i].keys():
                # Add sweep functions according to the mapping sf_for_sp.
                for sf_key in sf_for_sp.get(param, []):
                    sf = self.sweep_functions_dict[sf_key]
                    if sf is None:
                        continue
                    sf_sw_ctrl = getattr(sf, 'sweep_control', 'soft')
                    if sw_ctrl is not None and sf_sw_ctrl != sw_ctrl:
                        raise ValueError(
                            f'Cannot combine soft sweep and hard sweep in '
                            f'dimension {i}.')
                    if sf_sw_ctrl == 'hard':
                        if i != 0:
                            raise ValueError(
                                f'Hard sweeps are only allowed in dimension '
                                f'0. Cannot perform hard sweep for {param} '
                                f'in dim {i}.')
                        # hard sweep is not encapsulated by Indexed_Sweep
                        # and we only need one hard sweep per dimension
                        if sw_ctrl is None:
                            self.sweep_functions[i] = sf
                    else:
                        sweep_function = swf.Indexed_Sweep(
                            sf, values=self.sweep_points[i][param][0],
                            name=self.sweep_points[i][param][2],
                            parameter_name=param
                        )
                        self.sweep_functions[i].add_sweep_function(
                            sweep_function
                        )
                    sw_ctrl = sf_sw_ctrl
                # Params for which no sweep function is present are treated as
                # pulse parameter sweeps below.
                if param in sf_for_sp:
                    continue  # was treated above already
                elif i == 1:  # dimension 1
                    # assuming that this parameter is a pulse parameter and we
                    # therefore need a SegmentSoftSweep as the first sweep
                    # function in our multi_sweep_function
                    if not self.sweep_functions[i].sweep_functions \
                            or self.sweep_functions[i].sweep_functions[0] != \
                                awg_swf.SegmentSoftSweep:
                        self.sweep_functions[i].insert_sweep_function(
                            pos=0,
                            sweep_function=awg_swf.SegmentSoftSweep
                        )
                    if sp_pulses:
                        sp_pulses[i][param] = self.sweep_points[i][param]
                else:  # dimension 0
                    # assuming that this parameter is a pulse parameter, and we
                    # therefore need a SegmentHardSweep as the first sweep
                    # function in our multi_sweep_function
                    if not self.sweep_functions[i].sweep_functions:
                        # no previous sweep functions defined in dim 0
                        self.sweep_functions[i].insert_sweep_function(
                            pos=0,
                            sweep_function=awg_swf.SegmentHardSweep)
                    elif self.sweep_functions[i].sweep_functions[
                             0].sweep_control != 'hard':
                        # sweep_functinos[0] is not empty but what is there
                        # isn't a hard sweep;
                        raise NotImplementedError(
                            'Combined sweeping of pulse parameters and other '
                            'parameters for which a sweep function is provided '
                            'is not supported in dimension 0.')
                    if sp_pulses:
                        sp_pulses[i][param] = self.sweep_points[i][param]
                    sw_ctrl = 'hard'

    def parallel_sweep(self, preprocessed_task_list=(), block_func=None,
                       block_align=None, **kw):
        """
        Calls a block creation function for each task in a task list,
        puts these blocks in parallel and sweeps over the given sweep points.

        Note that this method automatically adds final readout pulses for
        qubits that are listed in meas_obj_names, but do not have any
        RO or Acq operation inside the block generated by the block_func.

        :param preprocessed_task_list: a list of dictionaries, each containing
            keyword arguments for block_func, plus a key 'prefix' with a unique
            prefix string, plus optionally a key 'params_to_prefix' created
            by preprocess_task indicating which sweep parameters have to be
            prefixed with the task prefix, plus optionally a key
            pulse_modifs with pulse modifiers for the created blocks.
        :param block_func: a handle to a function that creates a block. As
            an alternative, a task-specific block_func can be given as a
            parameter of the task. If the block creation function instead
            returns a list of blocks, the i-th blocks of all tasks are
            assembled in parallel to each other, and the resulting
            multitask blocks are then assembled sequentially in the order of
            the list index.
        :param block_align: (str or list) alignment of the parallel blocks, see
            CircuitBuilder.simultaneous_blocks, default: center. If the block
            creation function creates a list of N blocks, block_align can be
            a list of N strings (otherwise the same alignment is used for
            all parallel blocks).
        :param kw: keyword arguments are passed to sweep_n_dim, except:
            - 'ro_qubits'. This keyword argument is overwritten by the list
              of qubits in self.meas_obj_names for which there are no
              readout pulses in the parallel blocks.
        :return: see sweep_n_dim
        """
        parallel_blocks = []
        for task in preprocessed_task_list:
            # copy the task in order to not modify the original task
            task = copy(task)  # no deepcopy: might contain qubit objects
            # pop prefix and params_to_prefix since they are not needed by
            # the block creation function
            prefix = task.pop('prefix')
            params_to_prefix = task.pop('params_to_prefix', None)
            pulse_modifs = task.pop('pulse_modifs', None)
            # the block_func passed as argument is used for all tasks that
            # do not define their own block_func
            if not 'block_func' in task:
                task['block_func'] = block_func
            # Call the block creation function. The items in the task dict
            # are used as kwargs for this function.
            new_block = task['block_func'](**task)
            # If a single block was returned, create a single-entry list to
            # have a unified treatment afterwards.
            if not isinstance(new_block, list):
                new_block = [new_block]
            for b in new_block:
                if pulse_modifs is not None:
                    b.pulses = b.pulses_sweepcopy([pulse_modifs], [None])
                # prefix the block names to avoid naming conflicts later on
                b.name = prefix + b.name
                # For the sweep points that need to be prefixed (see
                # preprocess_task), call the respective method of the block
                # object.
                if params_to_prefix is not None:
                    # params_to_prefix is a list of lists (per dimension) and
                    # needs to be flattened
                    b.prefix_parametric_values(
                        prefix, [k for l in params_to_prefix for k in l])
            # add the new blocks to the lists of blocks
            parallel_blocks.append(new_block)

        # We currently require that all block functions must return the
        # same number of blocks.
        if not isinstance(block_align, list):
            block_align = [block_align] * len(parallel_blocks[0])
        # assemble all i-th blocks in parallel
        self.all_main_blocks = [
            self.simultaneous_blocks(
                f'all{i}', [l[i] for l in parallel_blocks],
                block_align=block_align[i])
            for i in range(len(parallel_blocks[0]))]
        if len(parallel_blocks[0]) > 1:
            # assemble the multitask blocks sequentially
            self.all_main_blocks = self.sequential_blocks(
                'all', self.all_main_blocks)
        else:
            self.all_main_blocks = self.all_main_blocks[0]
        sweep_points = self.get_sweep_points_for_sweep_n_dim()
        self._create_dummy_sweep_params(self.sweep_points)
        if sweep_points is not self.sweep_points:
            self._create_dummy_sweep_params(sweep_points)
        # Generate kw['ro_qubits'] as explained in the docstring
        op_codes = [p['op_code'] for p in self.all_main_blocks.pulses if
                    'op_code' in p]
        kw['ro_qubits'] = [m for m in self.meas_obj_names
                           if f'RO {m}' not in op_codes
                           and f'Acq {m}' not in op_codes]
        # call sweep_n_dim to perform the actual sweep
        return self.sweep_n_dim(sweep_points,
                                body_block=self.all_main_blocks,
                                cal_points=self.cal_points, **kw)

    def _create_dummy_sweep_params(self, sweep_points):
        if len(sweep_points[0]) == 0:
            # Create a single segement if no hard sweep points are provided.
            sweep_points.add_sweep_parameter('dummy_hard_sweep', [0],
                                             dimension=0)
        if self._num_sweep_dims == 2 and len(sweep_points[1]) == 0:
            # With this dummy soft sweep, exactly one sequence will be created
            # and the data format will be the same as for a true soft sweep.
            sweep_points.add_sweep_parameter('dummy_soft_sweep', [0],
                                             dimension=1)

    def get_sweep_points_for_sweep_n_dim(self):
        """Return the sweep_points list that is passed to sweep_n_dim.

        This method can be implemented by child classes to modify which sweep
        points are used to generate segments and sequences, e.g. only one RO
        segment is needed in a feedline spectroscopy and not one per fequency
        sweep point.
        """
        return self.sweep_points

    @staticmethod
    def find_qubits_in_tasks(qubits, task_list, search_in_operations=True):
        """
        Searches for qubit objects and all mentions of qubit names in the
        provided tasks.
        :param qubits: (list of str or objects) list of qubits whose mentions
            should be searched
        :param task_list: (list of dicts) list of tasks in which qubit objects
            and mentions of qubit names should be searched.
        :param search_in_operations: (bool) whether qubits should also be
            searched inside op_codes, default: True
        :return: list of a qubit object for each found qubit (if objects are
            available, otherwise list of qubit names)
        """
        # This dict maps from qubit names to qubit object if qubit objects
        # are available. Otherwise it is a trivial map from qubit names to
        # qubit names.
        qbs_dict = {qb if isinstance(qb, str) else qb.name: qb for qb in
                    qubits}
        found_qubits = []

        # helper function that checks candidates and calls itself recursively
        # if a candidate is a list
        def append_qbs(found_qubits, candidate):
            if isinstance(candidate, qb_mod.QuDev_transmon):
                if candidate.name in qbs_dict:
                    # avoid duplicates by adding it exactly in the form
                    # contained in the qbs_dict
                    append_qbs(found_qubits, candidate.name)
                elif candidate not in found_qubits:
                    found_qubits.append(candidate)
            elif isinstance(candidate, str):
                if candidate in qbs_dict.keys():
                    # it is a mention of a qubit
                    if qbs_dict[candidate] not in found_qubits:
                        found_qubits.append(qbs_dict[candidate])
                elif ' ' in candidate and search_in_operations:
                    # If it contains spaces, it could be an op_code. To
                    # search in operations, we just split the potential op_code
                    # at the spaces and search again in the resulting list
                    append_qbs(found_qubits, candidate.split(' '))
            elif isinstance(candidate, list):
                # search inside each list element
                for v in candidate:
                    append_qbs(found_qubits, v)
            elif isinstance(candidate, dict):
                for v in candidate.values():
                    append_qbs(found_qubits, v)
            else:
                return None

        # search in all tasks
        append_qbs(found_qubits, task_list)
        
        return found_qubits

    def create_meas_objs_list(self, task_list=None, **kw):
        """
        Creates a list of all measure objects used in the measurement. The
        following measure objects are added:
        - qubits listed in kw['ro_qubits']
        - qubits listed in the parameter ro_qubits of each task
        - if kw['ro_qubits'] is not provided and a task does not have a
          parameter ro_qubits: the result of get_meas_objs_from_task for
          this task
        Stores two lists:
        - self.meas_objs: list of measure objects (None if not available)
        - self.meas_obj_names: and list of measure object names

        :param task_list: (list of dicts) the task list
        :param kw: keyword arguments
        """
        if task_list is None:
            task_list = self.task_list
        if task_list is None:
            task_list = [{}]
        ro_qubits = kw.get('ro_qubits')
        if ro_qubits is None:
            # Combine for all tasks, fall back to get_meas_objs_from_task if
            # ro_qubits does not exist in a task.
            ro_qubits = [qb for task in task_list for qb in task.pop(
                'ro_qubits', self.get_meas_objs_from_task(task))]
        else:
            # Add ro_qubits from all tasks without falling back to
            # get_meas_objs_from_task.
            ro_qubits += [qb for task in task_list for qb in
                          task.pop('ro_qubits', [])]
        # Unique and sort. To make this possible, convert to str.
        ro_qubits = [qb if isinstance(qb, str) else qb.name for qb in
                     ro_qubits]
        ro_qubits = list(np.unique(ro_qubits))
        ro_qubits.sort()
        # Get the objects again if available, and store the lists.
        self.meas_objs, self.meas_obj_names = self.get_qubits(
            'all' if len(ro_qubits) == 0 else ro_qubits)

    def generate_kw_sweep_points(self, task):
        """
        Generates sweep_points based on task parameters (or kwargs if kw is
        passed instead of a task) according to the specification in the
        property kw_for_sweep_points. The generated sweep points are added to
        sweep_points in the task (or in kw). If needed, sweep_points is
        converted to a SweepPoints object before.

        Format of kw_for_sweep_points: dict with
         - key: key to be searched in kw and in tasks
         - val: dict of kwargs for SweepPoints.add_sweep_parameter, with the
           additional possibility of specifying 'values_func', a lambda
           function that processes the values in kw before using them as
           sweep values
        or a list of such dicts to create multiple sweep points based on a
        single keyword argument.

        :param task: a task dictionary ot the kw dictionary
        """
        # make sure that sweep_points is a SweepPoints object
        task['sweep_points'] = SweepPoints(task.get('sweep_points', None))
        for k, sp_dict_list in self.kw_for_sweep_points.items():
            if isinstance(sp_dict_list, dict):
                sp_dict_list = [sp_dict_list]
            # This loop can create multiple sweep points based on a single
            # keyword argument.
            for v in sp_dict_list:
                # copy to allow popping the values_func, which should not be
                # passed to SweepPoints.add_sweep_parameter
                v = copy(v)
                values_func = v.pop('values_func', None)
                if isinstance(values_func, str):
                    # assumes the string is the name of a self method
                    values_func = getattr(self, values_func, None)

                # comma-separated strings correspond to different keys in task
                # whose corresponding values can be used as input parameters
                # for values_func
                k_list = k.split(',')
                # if the respective task parameters (or keyword arguments) exist
                if all([k in task and task[k] is not None for k in k_list]):
                    if values_func is not None:
                        # the entries in k_list point to input parameters
                        # for values_func
                        values = values_func(*[task[key] for key in k_list])
                    elif isinstance(task[k_list[0]], int):
                        # A single int N as sweep value will be interpreted as
                        # a sweep over N indices.
                        values = np.arange(task[k_list[0]])
                    else:
                        # Otherwise it is assumed that list-like sweep
                        # values are provided.
                        values = task[k_list[0]]
                    task['sweep_points'].add_sweep_parameter(
                        values=values, **v)

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'].update({
            MultiTaskingExperiment.__name__: odict({
                'n_cal_points_per_state': (int, 1),
                'cal_states': (str, None),
                'ro_qubits': ((qb_mod.QuDev_transmon, 'multi_select'), None),
            })
        })
        d['task_list_fields'].update({
            MultiTaskingExperiment.__name__: odict({
                'sweep_points': (SweepPoints, None),
            })
        })
        return d


class CalibBuilder(MultiTaskingExperiment):
    """
    This class extends MultiTaskingExperiment with some methods that are
    useful for calibration measurements.

    :param task_list: see MultiTaskingExperiment
    :param kw: kwargs passed to MultiTaskingExperiment, plus in addition:
        update: (bool) whether instrument settings should be updated based on
            analysis results of the calibration measurement, default: False
    """
    def __init__(self, task_list, **kw):
        super().__init__(task_list=task_list, **kw)
        self.set_update_callback(**kw)

    def set_update_callback(self, update=False, **kw):
        """
        Configures QuantumExperiement to run the function run_update()
        (or a user-specified callback function) in autorun after measurement
        and analysis, conditioned on the flag self.update. The flag is
        intialized to True if update=True was passed, and False otherwise.

        """
        self.update = update
        self.callback = kw.get('callback', self.run_update)
        self.callback_condition = lambda : self.update and self.analyze

    def run_update(self, **kw):
        # must be overriden by child classes to update the
        # relevant calibration parameters
        pass

    def max_pulse_length(self, pulse, sweep_points=None,
                         given_pulse_length=None):
        """
        Determines the maximum time duration of a pulse during a sweep,
        where the pulse length could be modified by a sweep parameter.
        Currently, this is implemented only for up to 2-dimensional sweeps.

        :param pulse: a pulse dictionary (which could contain params that
            are ParametricValue objects)
        :param sweep_points: a SweepPoints object describing the sweep
        :param given_pulse_length: overwrites the pulse_length determined by
            the sweep points with the given value (i.e., no actual sweep is
            performed). This is useful to conveniently process a
            user-provided fixed value for the maximum pulse length.
        """
        pulse = copy(pulse)
        # the following parameters are required to create an UnresolvedPulse
        pulse['name'] = 'tmp'
        pulse['element_name'] = 'tmp'

        if given_pulse_length is not None:
            log.debug(f'maximum pulse length set by the user: '
                      f'{given_pulse_length * 1e9:.2f} ns')
            pulse['pulse_length'] = given_pulse_length
            # generate a pulse object to extend the given length with buffer
            # times etc
            p = UnresolvedPulse(pulse)
            return p.pulse_obj.length

        # Even if we only need a single pulse, creating a block allows
        # us to easily perform a sweep.
        b = Block('tmp', [pulse])
        # Clean up sweep points
        sweep_points = deepcopy(sweep_points)
        if sweep_points is None:
            sweep_points = SweepPoints([{}, {}])
        while len(sweep_points) < 2:
            sweep_points.add_sweep_dimension()
        for i in range(len(sweep_points)):
            if len(sweep_points[i]) == 0:
                # Make sure that there exists at least a single sweep point
                # that does not overwrite default values of the pulse params.
                sweep_points[i].update({'dummy': ([0], '', 'dummy')})

        # determine number of sweep values per dimension
        nr_sp_list = [len(list(d.values())[0][0]) for d in sweep_points]
        max_length = 0
        for i in range(nr_sp_list[1]):
            for j in range(nr_sp_list[0]):
                # Perform sweep
                pulses = b.build(
                    sweep_dicts_list=sweep_points, sweep_index_list=[j, i])
                # generate a pulse object to extend the pulse length with
                # buffer times etc. The pulse with index 1 is needed because
                # the virtual block start pulse has index 0.
                p = UnresolvedPulse(pulses[1])
                max_length = max(p.pulse_obj.length, max_length)
        return max_length

    @staticmethod
    def add_default_ramsey_sweep_points(sweep_points, tile=2,
                                        repeat=0, **kw):
        """
        Adds phase sweep points for Ramsey-type experiments to the provided
        sweep_points. Assumes that each phase is required twice (to measure a
        comparison between two scenarios, e.g., with flux pulses on and off
        in a dynamic phase measurement).

        :param sweep_points: (SweepPoints object, list of dicts, or None) the
            existing sweep points
        :param tile: (int) TODO, default: 2
        :param repeat: (int) TODO, default: 0
        :param kw: keyword arguments
            nr_phases: how many phase sweep points should be added, default: 6.
                If there already exist sweep points in dimension 0, this
                parameter is ignored and the number of phases is adapted to
                the number of existing sweep points.
            endpoint_phases: (bool, default True) whether the endpoint (360 deg.)
                should be included in the linspace for the phase sweep points.
        :return: sweep_points with the added phase sweep points
        """
        if tile > 0 and repeat > 0:
            raise ValueError('"repeat" and "tile" cannot both be > 0.')
        # ensure that sweep_points is a SweepPoints object with at least one
        # dimension
        sweep_points = SweepPoints(sweep_points, min_length=1)
        # If there already exist sweep points in dimension 0, this adapt the
        # number of phases to the number of existing sweep points.
        if len(sweep_points[0]) > 0:
            nr_phases = sweep_points.length(0) // 2
        else:
            nr_phases = kw.get('nr_phases', 6)
        # create the phase sweep points (with each phase twice)
        hard_sweep_dict = SweepPoints()
        if 'phase' not in sweep_points[0]:
            phases = np.linspace(0, 360, nr_phases,
                                 endpoint=kw.get('endpoint', True))
            if tile > 0:
                phases = np.tile(phases, tile)
            elif repeat > 0:
                phases = np.repeat(phases, repeat)
            hard_sweep_dict.add_sweep_parameter('phase', phases, 'deg')
        # add phase sweep points to the existing sweep points (overwriting
        # them if they exist already)
        sweep_points.update(hard_sweep_dict)
        return sweep_points

    @classmethod
    def gui_kwargs(cls, device):
        d = super().gui_kwargs(device)
        d['kwargs'].update({
            CalibBuilder.__name__: odict({
                'update': (bool, False),
            })
        })
        return d


class CPhase(CalibBuilder):
    """
    Class to measure the phase acquired by a qubit (qbr) during a flux pulse
    conditioned on the state of another qubit (qbl). Also measures the
    leakage of qbl.
    In this measurement, the phase from two Ramsey type measurements
    on qbr is measured, once with the control qubit qbl in the excited state
    and once in the ground state. The conditional phase is calculated as the
    difference.

    Args:
        FIXME: add further args
        TODO
        :param cz_pulse_name: see CircuitBuilder
        :param n_cal_points_per_state: see CalibBuilder.get_cal_points()
        :param kw:
            cal_states_rotations: (dict) Overwrite the default choice of
                cal_states_rotations written to the meta data. The keys are
                qubit names, and the values are dictionaries mapping state
                names to state indices, e.g., {'g': 0, 'e': 1, 'f': 2}. For
                qubits that are not contained in the dict, the value
                generated in cphase_block will be used.
            TODO further kws
    ...
    """
    kw_for_task_keys = ['ref_pi_half', 'num_cz_gates']
    task_mobj_keys = ['qbl', 'qbr']
    default_experiment_name = 'CPhase_measurement'

    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            # By default, include f-level cal points (for measuring leakage).
            kw['cal_states'] = kw.get('cal_states', 'gef')

            super().__init__(task_list, sweep_points=sweep_points, **kw)

            # initialize properties specific to the CPhase measurement
            self.cphases = None
            self.contrast_losses = None
            self.leakage = None
            self.delta_leakage = None
            self.swap_errors = None
            self.cz_durations = {}
            self.cal_states_rotations = {}

            # Preprocess sweep points and tasks before creating the sequences
            self.add_default_sweep_points(**kw)
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            # the block alignments are for: prepended pulses, initial
            # rotations, flux pulse, final rotations
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.cphase_block,
                block_align=['center', 'end', 'center', 'start'], **kw)

            # allow the user to overwrite entries in cal_states_rotations
            self.cal_states_rotations.update(
                kw.get('cal_states_rotations', {}))
            # save CPhase-specific metadata
            self.exp_metadata.update({
                'cz_durations': self.cz_durations,
                'cal_states_rotations': self.cal_states_rotations,
            })

            self.autorun(**kw)  # run measurement & analysis if requested in kw
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def add_default_sweep_points(self, **kw):
        """
        Adds default sweep points for the CPhase experiment. These are:
        - Ramsey phases (in dimension 0)
        - pi pulse off (for control qubit, in dimension 0)
        :param kw: kwargs passed to add_default_ramsey_sweep_points
        """
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, **kw)
        nr_phases = self.sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'pi_pulse_off', [0] * nr_phases + [1] * nr_phases)
        self.sweep_points.update(hard_sweep_dict)

    def cphase_block(self, sweep_points,
                     qbl, qbr, num_cz_gates=1, max_flux_length=None,
                     prepend_pulse_dicts=None, cphase=None, **kw):
        """
        This function creates the blocks for a single CPhase measurement task.
        :param sweep_points: sweep points
        :param qbl: control qubit (= qubit on which leakage is measured)
        :param qbr: Ramsey qubit (= qubit on which the cphase is measured)
        :param num_cz_gates: number of sequential CZ gates, default: 1
        :param max_flux_length: determines the time to wait before the final
            rotations to the, default: None, in which case it will be
            determined automatically
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            CircuitBuilder.block_from_pulse_dicts
        :param cphase: (float) conditional phase of the CZ gate (degrees),
            if allowed by the gate. Defaults to None (180 degrees).
        :param kw: further keyword arguments:
            cz_pulse_name: task-specific prefix of CZ gates (overwrites
                global choice passed to the class init)
            spectator_op_codes: op_code for adding initializations of
                spectator qubits. Will be assembled in parallel with the
                initial rotations.
            ref_pi_half: (bool) TODO, default: False
        """
        ref_pi_half = kw.get('ref_pi_half', False)

        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        pulse_modifs = {'all': {'element_name': 'cphase_initial_rots_el'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'X180 {qbl}', f'X90 {qbr}'] +
                                 kw.get('spectator_op_codes', []),
                                 pulse_modifs=pulse_modifs)
        for p in ir.pulses[1:]:
            p['ref_point_new'] = 'end'
        ir.pulses[0]['pulse_off'] = ParametricValue(param='pi_pulse_off')

        fp = self.block_from_ops('flux', [f"{kw.get('cz_pulse_name', 'CZ')} "
                                          f"{qbl} {qbr}"] * num_cz_gates)
        for p in fp.pulses:
            p['cphase'] = cphase
        # TODO here, we could do DD pulses (CH 2020-06-19)

        for k in sweep_points.get_sweep_dimension(1, default={}):
            for p in fp.pulses:
                p[k] = ParametricValue(k)
        max_flux_length = self.max_pulse_length(fp.pulses[0], sweep_points,
                                                max_flux_length)
        w = self.block_from_ops('wait', [])
        w.block_end.update({'pulse_delay': max_flux_length * num_cz_gates})
        fp_w = self.simultaneous_blocks('sim', [fp, w], block_align='center')

        pulse_modifs = {'all': {'element_name': 'cphase_final_rots_el'}}
        fr = self.block_from_ops('final_rots',
                                 [f"{'X90' if ref_pi_half else 'X180'} {qbl}",
                                  f'X90s {qbr}'],
                                 pulse_modifs=pulse_modifs)
        fr.set_end_after_all_pulses()
        if not ref_pi_half:
            fr.pulses[0]['pulse_off'] = ParametricValue(param='pi_pulse_off')
        for k in sweep_points[0].keys():
            if k != 'pi_pulse_on' and '=' not in k:
                if ref_pi_half:
                    fr.pulses[0][k] = ParametricValue(k)
                fr.pulses[1][k] = ParametricValue(k)

        self.cz_durations.update({
            fp.pulses[0]['op_code']: fr.pulses[0]['pulse_delay']})
        self.cal_states_rotations.update({qbl: {'g': 0, 'e': 1, 'f': 2},
                                          qbr: {'g': 0, 'e': 1}})
        self.data_to_fit.update({qbl: ['pg', 'pf'] if ref_pi_half else 'pf',
                                 qbr: 'pe'})

        return [pb, ir, fp_w, fr]

    def guess_label(self, **kw):
        """
        Default label with CPhase-specific information
        :param kw: keyword arguments
        """
        predictive_label = kw.pop('predictive_label', False)
        if self.label is None:
            if predictive_label:
                self.label = 'Predictive_' + self.experiment_name
            else:
                self.label = self.experiment_name
            if self.classified:
                self.label += '_classified'
            if 'active' in self.get_prep_params()['preparation_type']:
                self.label += '_reset'
            for t in self.task_list:
                self.label += f"_{t['qbl']}{t['qbr']}"
                num_cz_gates = t.get('num_cz_gates', 1)
                if num_cz_gates > 1:
                    self.label += f'_{num_cz_gates}_gates'

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects of a task. In case of CPhase
        this is qbl and qbr.
        :param task: a task dictionary
        :return: list of qubit objects (if available) or names
        """
        qbs = self.get_qubits([task['qbl'], task['qbr']])
        return qbs[0] if qbs[0] is not None else qbs[1]

    def run_analysis(self, **kw):
        """
        Runs analysis, and stores analysis results and analysis instance.
        :param kw: keyword arguments
             plot_all_traces: (bool) TODO, default: True
             plot_all_probs: (bool) TODO, default: True
             ref_pi_half: (bool) TODO, default: False
        :return: cphases, contrast_losses, leakage, and the analysis instance
        """
        plot_all_traces = kw.get('plot_all_traces', True)
        plot_all_probs = kw.get('plot_all_probs', True)
        ref_pi_half = kw.get('ref_pi_half', False)
        if self.classified:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_acq() for vn in
                                     qb.int_avg_classif_det.value_names]
                           for qb in self.meas_objs}
        else:
            channel_map = {qb.name: [vn + ' ' +
                                     qb.instr_acq() for vn in
                                     qb.int_avg_det.value_names]
                           for qb in self.meas_objs}
        self.analysis = tda.CPhaseLeakageAnalysis(
            qb_names=self.meas_obj_names,
            options_dict={'TwoD': (len(self.sweep_points) == 2),
                          'plot_all_traces': plot_all_traces,
                          'plot_all_probs': plot_all_probs,
                          'channel_map': channel_map,
                          'ref_pi_half': ref_pi_half})
        self.cphases = {}
        self.contrast_losses = {}
        self.leakage = {}
        self.delta_leakage = {}
        self.swap_errors = {}
        for task in self.task_list:
            self.cphases.update({task['prefix'][:-1]: self.analysis.proc_data_dict[
                'analysis_params_dict'][f"cphase_{task['qbr']}"]['val']})
            self.contrast_losses.update(
                {task['prefix'][:-1]: self.analysis.proc_data_dict[
                    'analysis_params_dict'][
                    f"contrast_loss_{task['qbr']}"]['val']})
            if ref_pi_half:
                self.swap_errors.update(
                    {task['prefix'][:-1]: self.analysis.proc_data_dict[
                        'analysis_params_dict'][
                        f"amps_{task['qbl']}"]['val']})
            self.leakage.update(
                {task['prefix'][:-1]: self.analysis.proc_data_dict[
                    'analysis_params_dict'][
                    f"leakage_{task['qbl']}"]['val']})
            self.delta_leakage.update(
                {task['prefix'][:-1]: self.analysis.proc_data_dict[
                    'analysis_params_dict'][
                    f"leakage_increase_{task['qbl']}"]['val']})

        return self.cphases, self.contrast_losses, self.leakage, \
               self.analysis, self.swap_errors


class DynamicPhase(CalibBuilder):
    """
    Dynamic Phase Measurement for CZ gates by performing a Ramsey with
    interleaved flux pulse. This is a multitasking experiment, see docstrings
    of MultiTaskingExperiment and of CalibBuilder for general information.
    Each task corresponds to the characterization of a particular CZ gate
    (specified by the op_code key in the task), i.e., multiple gates can be
    characterized in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method dynamic_phase_block):

        |X90**|  ---  |fluxpulse***|  ---  |X90*|  ---  |RO*|
                                         sweep phase

        * = in parallel on all qubits_to_measure (key in the task)
        ** = in parallel on all qubits_to_measure (key in the task), unless
             qubits_to_drive is provided (key in the task), in which case
             this pulse will be in parallel on all qubits_to_drive
        *** = specified by the op_code (key in the task). The measurement is
              performed once with this pulse enabled and, for reference,
              once with this pulse disabled (setting the pulse_off property
              of the pulse to True).

        Note: in case of a FLIP gate or in case of push-away pulses,
        flux pulses are on multiple qubits.

        Note: the class can measure dynamic phases also for other kind
        of gates and pulses (single-qb gates, swap gates, ...),
        by specifying an op_code that does not correspond to a CZ gates.
        Since each kind of gate might bring its own subtleties, this is
        an experimental feature that should be used with care by expert
        users (a future version might include more stable code for other
        kinds of gates). In this context, also note the parameters
        init_for_swap and qubits_to_drive of the method dynamic_phase_block.
        Also note that parameter names like num_cz_gates need to be
        understood as "number of gates specified by the op_code" in this case.

    Sweep points passed in dimension 1 (if any) are interpreted as
    parameters of the flux pulse (CZ gate) while sweep points in dimension 0
    are interpreted as parameters of the final rotation pulses. The latter
    are generated by add_default_ramsey_sweep_points if they are not passed
    explicitly.

    The class will automatically decide to run multiple separate measurements
    if the  dynamic phases of multiple qubits are to be measured in some
    task(s) and a simultaneous measurement of these dynamical phases is not
    possible. This behavior can be controlled by the keyword arguments
    simultaneous and simultaneous_groups (see below).

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to set_update_callback,
        add_default_sweep_points, sweep_n_dim, autorun, and to the parent
        class.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        dynamic_phase_block):
        - num_cz_gates
        - init_for_swap

        Moreover, the following keyword arguments are understood:
        simultaneous: (bool, default: False) measure all phases simultaneously
            (not possible if phases of both gate qubits should be measured).
        simultaneous_groups: (list of list of qubit objects or names)
            specifies that the phases of all qubits within each sublist can
            be measured simultaneously.
            If simultaneous=False and no simultaneous_groups are specified,
            only one qubit per task will be measured in parallel.
        reset_phases_before_measurement: (bool, default: True) If True,
            resets the basis_rotation parameter to {} before measurement(s).
            If False, keeps the dict stored in this parameter and updates
            only the entries in this dict that were measured.
    """

    kw_for_task_keys = ['num_cz_gates', 'init_for_swap']
    default_experiment_name = 'Dynamic_phase_measurement'

    @assert_not_none('task_list')
    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            self.simultaneous = kw.get('simultaneous', False)
            self.simultaneous_groups = kw.get('simultaneous_groups', None)
            if self.simultaneous_groups is not None:
                kw['simultaneous_groups'] = [
                    [qb if isinstance(qb, str) else qb.name for qb in group]
                    for group in self.simultaneous_groups]
            self.reset_phases_before_measurement = kw.get(
                'reset_phases_before_measurement', True)

            self.dynamic_phase_analysis = {}
            self.dyn_phases = {}
            for task in task_list:
                if task.get('qubits_to_measure', None) is None:
                    task['qubits_to_measure'] = task['op_code'].split(' ')[1:]
                else:
                    # copy to not modify the caller's list
                    task['qubits_to_measure'] = copy(task['qubits_to_measure'])

                for k, v in enumerate(task['qubits_to_measure']):
                    if not isinstance(v, str):
                        task['qubits_to_measure'][k] = v.name

                if 'prefix' not in task:
                    task['prefix'] = task['op_code'].replace(' ', '')

            qbm_all = [task['qubits_to_measure'] for task in task_list]
            if not self.simultaneous and max([len(qbs) for qbs in qbm_all]) > 1:
                # create a child for each measurement
                self.parent = None
                task_lists = []
                if self.simultaneous_groups is not None:
                    for group in self.simultaneous_groups:
                        new_task_list = []
                        for task in task_list:
                            group = [qb if isinstance(qb, str) else qb.name
                                     for qb in group]
                            new_task = copy(task)
                            new_task['qubits_to_measure'] = [
                                qb for qb in new_task['qubits_to_measure']
                                if qb in group]
                            new_task_list.append(new_task)
                        task_lists.append(new_task_list)
                    # the children measure simultaneously within each group
                    kw['simultaneous'] = True
                else:
                    # the number of required children is the length of the
                    # longest qubits_to_measure
                    for z in zip_longest(*qbm_all):
                        new_task_list = []
                        for task, new_qb in zip(task_list, z):
                            if new_qb is not None:
                                new_task = copy(task)
                                new_task['qubits_to_measure'] = [new_qb]
                                new_task_list.append(new_task)
                        task_lists.append(new_task_list)

                # We call the init of super() only for the spawned child
                # measurements. We need special treatment of some properties
                # in the following lines.
                self.MC = None
                # extract device object, which will be needed for update
                self.dev = kw.get('dev', None)
                # Configure the update callback for the parent. It will be
                # called after all children have analyzed.
                self.set_update_callback(**kw)
                # pop to ensure that children do not update
                kw.pop('update', None)
                # spawn the child measurements
                self.measurements = [DynamicPhase(tl, sweep_points,
                                                  parent=self, **kw)
                                     for tl in task_lists]
                # Use the choices for measure and analyze that were extracted
                # from kw by the children.
                self.measure = self.measurements[0].measure
                self.analyze = self.measurements[0].analyze
            else:
                # this happens if we are in child or if simultaneous=True or
                # if only one qubit per task is measured
                self.measurements = [self]
                self.parent = kw.pop('parent', None)
                super().__init__(task_list, sweep_points=sweep_points, **kw)

                if self.reset_phases_before_measurement:
                    for task in task_list:
                        self.operation_dict[self.get_cz_operation_name(
                            **task)]['basis_rotation'] = {}

                # Preprocess sweep points and tasks before creating the sequences
                self.add_default_sweep_points(**kw)
                self.preprocessed_task_list = self.preprocess_task_list(**kw)
                # the block alignments are for: prepended pulses, initial
                # rotations, flux pulse, final rotations
                self.sequences, self.mc_points = self.parallel_sweep(
                    self.preprocessed_task_list, self.dynamic_phase_block,
                    block_align=['center', 'end', 'center', 'start'], **kw)
            # run measurement & analysis & update if requested in kw
            # (unless the parent takes care of it)
            if self.parent is None:
                self.autorun(**kw)
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def __repr__(self):
        if self.measurements[0] != self:  # we have spawned child measurements
            return 'DynamicPhase: ' + repr(self.measurements)
        else:
            return super().__repr__()

    def add_default_sweep_points(self, **kw):
        """
        Adds default sweep points for the DynamicPhase experiment. These are:
        - Ramsey phases (in dimension 0)
        - flux pulse off (in dimension 0)
        :param kw: kwargs passed to add_default_ramsey_sweep_points
        """
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, **kw)
        nr_phases = self.sweep_points.length(0) // 2
        hard_sweep_dict = SweepPoints(
            'flux_pulse_off', [0] * nr_phases + [1] * nr_phases)
        self.sweep_points.update(hard_sweep_dict)

    def guess_label(self, **kw):
        """
        Default label with DynamicPhase-specific information
        :param kw: keyword arguments
        """
        if self.label is None:
            self.label = self.experiment_name
            for task in self.task_list:
                self.label += "_" + task['prefix'] + "_"
                for qb_name in task['qubits_to_measure']:
                    self.label += f"{qb_name}"

    def dynamic_phase_block(self, sweep_points, op_code, qubits_to_measure,
                            qubits_to_drive=None, prepend_pulse_dicts=None,
                            num_cz_gates=1, init_for_swap=False, **kw):
        """
        This function creates the blocks for a single DynamicPhase measurement
        task, see the pulse sequence in the class docstring.
        :param sweep_points: sweep points
        :param op_code: (str) the op_code of the CZ gate (flux pulse)
        :param qubits_to_measure: (list of str) the qubits on which the
            dynamic phase should be measured (Ramsey qubits). Typically,
            these are all qubits affected directly or indirectly (e.g.,
            via push-away pulses) by the CZ gate.
        :param qubits_to_drive: (list of str) the qubits on which the
            initial X90 rotation (see class docstring) shall be applied
            (default=None, in which case qubits_to_measure will be used).
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            CircuitBuilder.block_from_pulse_dicts
        :param num_cz_gates: number of sequential CZ gates, default: 1
        :param init_for_swap: (bool, default: False) Expert-level feature for
            swap gate characterization (see the note in the class docstring).
            If this is True, it will invert the meaning of qubits_to_drive in
            reference measurement (where the flux pulse has
            pulse_off=True), i.e., in the reference measurement, the initial
            X90 pulses will be applied to exactly those qubits_to_measure
            that are *not* part of qubits_to_drive.
        :param kw: keyword arguments passed to get_cz_operation_name
            cz_pulse_name: task-specific prefix of CZ gates (overwrites
                global choice passed to the class init)
        """
        if 'CZ' in op_code and ((sum([qb in op_code.split(' ')[1:]
                                      for qb in qubits_to_measure]) > 1)):
            raise ValueError(
                f"Dynamic phases of control and target qubit of a CZ gate "
                f"cannot be measured simultaneously ({op_code}).")

        # create prepended pulses (pb)
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        # create pulses for initial rotations (ir)
        if qubits_to_drive is None:
            qubits_to_drive = qubits_to_measure
        pulse_modifs = {
            'all': {'element_name': 'pi_half_start', 'ref_pulse': 'start'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'X90 {qb}' for qb in qubits_to_drive],
                                 pulse_modifs=pulse_modifs)
        for p in ir.pulses[1:]:
            p['ref_point_new'] = 'end'
        if init_for_swap:
            # switch off the original initial rotations if the flux pulse is
            # off
            for p in ir.pulses:
                p['pulse_off'] = ParametricValue('flux_pulse_off')
            # create a new set of initial rotations, which will be switched
            # on only if the flux pulse is off
            qubits_to_drive2 = [qb for qb in qubits_to_measure if qb not in
                                qubits_to_drive]
            ir2 = self.block_from_ops('initial_rots2',
                                     [f'X90 {qb}' for qb in qubits_to_drive2],
                                     pulse_modifs=pulse_modifs)
            for p in ir2.pulses[1:]:
                p['ref_point_new'] = 'end'
            # switch off the new initial rotations if the flux pulse is on
            for p in ir2.pulses:
                p['pulse_off'] = ParametricValue('flux_pulse_off',
                                                 func=lambda x : not x)
            # put the two sets of initial rotations in parallel (noting
            # that only one of them will be active at a time)
            ir = self.simultaneous_blocks('initial_rots', [ir, ir2],
                                          block_align='end')

        # create flux pulse (fp)
        if len(op_code.split(' ')) == 3:
            # For two-qubit operations (op_code consisting of three parts),
            # calling get_cz_operation_name() allows to take into account a
            # custom cz_pulse_name that was potentially provided in kw.
            proc_op_code = self.get_cz_operation_name(op_code=op_code, **kw)
        else:  # not a 2-qubit gate
            proc_op_code = op_code
        fp = self.block_from_ops('flux', [proc_op_code] * num_cz_gates)
        for p in fp.pulses:
            p['pulse_off'] = ParametricValue('flux_pulse_off')
        # All soft sweep points (sweep dimension 1) are interpreted as
        # parameters of the flux pulse, except if they are pulse modifier
        # sweep points (see docstring of Block.build).
        for k in sweep_points.get_sweep_dimension(1, default={}):
            if '=' not in k:  # '=' indicates a pulse modifier sweep point
                for p in fp.pulses:
                    p[k] = ParametricValue(k)

        # create pulses for final rotations (fr)
        pulse_modifs = {
            'all': {'element_name': 'pi_half_end', 'ref_pulse': 'start'}}
        fr = self.block_from_ops('final_rots',
                                 [f'X90 {qb}' for qb in qubits_to_measure],
                                 pulse_modifs=pulse_modifs)
        # reserve enough time for the longest X90 pulse
        fr.set_end_after_all_pulses()
        # All hard sweep points (sweep dimension 0) apart from flux_pulse_off
        # are interpreted as parameters of the final rotation pulses,
        # except if they are pulse modifier sweep points.
        for p in fr.pulses:
            for k in sweep_points[0].keys():
                # '=' indicates a pulse modifier sweep point
                if '=' not in k and k != 'flux_pulse_off':
                    p[k] = ParametricValue(k)

        # add the qubits measured in this task to the data_to_fit dictionary
        self.data_to_fit.update({qb: 'pe' for qb in qubits_to_measure})
        # return all generated blocks (parallel_sweep will arrange them)
        return [pb, ir, fp, fr]

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects of a task. In case of
        DynamicPhase, this list is the parameter qubits_to_measure.
        :param task: a task dictionary
        :return: list of qubit objects (if available) or names
        """
        qbs = self.get_qubits(task['qubits_to_measure'])
        return qbs[0] if qbs[0] is not None else qbs[1]

    def run_measurement(self, **kw):
        """
        Overloads the method from QuantumExperiment to deal with child
        measurements.
        """
        if self.measurements[0] != self:  # we have spawned child measurements
            for m in self.measurements:
                m.run_measurement(**kw)
        else:
            super().run_measurement(**kw)

    def run_analysis(self, **kw):
        """
        Runs analysis, stores analysis instance in self.dynamic_phase_analysis
        and stores dynamic phase in self.dyn_phases
        :param kw: keyword arguments
             extract_only: (bool) if True, do not plot, default: False
        :return: the dynamic phases dict and the analysis instance
        """
        if self.measurements[0] != self:  # we have spawned child measurements
            for m in self.measurements:
                m.run_analysis(**kw)
            return  # the rest of the function is executed in the children

        qb_names = [l1 for l2 in [task['qubits_to_measure'] for task in
                                  self.task_list] for l1 in l2]
        self.dynamic_phase_analysis = tda.DynamicPhaseAnalysis(
            qb_names=qb_names, t_start=self.timestamp)

        for task in self.task_list:
            if len(task['op_code'].split(' ')) == 3:
                op = self.get_cz_operation_name(**task)
            else:  # not a 2-qubit gate
                op = task['op_code']
            self.dyn_phases[op] = {}
            for qb_name in task['qubits_to_measure']:
                self.dyn_phases[op][qb_name] = \
                    (self.dynamic_phase_analysis.proc_data_dict[
                        'analysis_params_dict'][f"dynamic_phase_{qb_name}"][
                        'val'] * 180 / np.pi)[0]
        if self.parent is not None:
            for k, v in self.dyn_phases.items():
                if k not in self.parent.dyn_phases:
                    self.parent.dyn_phases[k] = {}
                self.parent.dyn_phases[k].update(v)
        return self.dyn_phases, self.dynamic_phase_analysis

    def run_update(self, **kw):
        assert self.measurements[0].dev is not None, \
            "Update only works with device object provided."
        assert len(self.dyn_phases) > 0, \
            "Update is only allowed after running the analysis."
        assert len(self.measurements[0].mc_points[1]) == 1, \
            "Update is only allowed without a soft sweep."
        assert self.parent is None, \
            "Update has to be run for the parent object, not for the " \
            "individual child measurements."

        for op, dp in self.dyn_phases.items():
            op_split = op.split(' ')
            basis_rot_par = self.dev.get_pulse_par(
                *op_split, param='basis_rotation')

            if self.reset_phases_before_measurement:
                basis_rot_par(dp)
            else:
                not_updated = {k: v for k, v in basis_rot_par().items()
                               if k not in dp}
                basis_rot_par().update(dp)
                if len(not_updated) > 0:
                    log.warning(f'Not all basis_rotations stored in the '
                                f'pulse settings for {op} have been '
                                f'measured. Keeping the following old '
                                f'value(s): {not_updated}')


class MeasurementInducedDephasing(DynamicPhase):
    """
    Measures the measurement-induced dephasing of a qubit while applying a
    readout pulse

    The measurement consists of a readout pulse (no data acquisition)
    interleaved in a Ramsey pulse sequence, followed by a readout pulse and
    acquisition. This allows extracting the dephasing induced by the first
    readout pulse, e.g. while sweeping its amplitude.
    """

    default_experiment_name = 'measurement_induced_dephasing'
    kw_for_sweep_points = {
        'amps': dict(param_name='amplitude', unit='V',
                     label='Readout Amplitude', dimension=1)
    }

    def __init__(self, task_list=None, qubits=None, buffer_length_end=1e-6,
                 **kw):
        if task_list is None:
            if qubits is None:
                raise ValueError('Please provide either "qubits" or '
                                 '"task_list"')
            # Create task_list from qubits
            task_list = [{'qb': qb.name} for qb in qubits]
        for task in task_list:
            qb = task.pop('qb')
            task['qubits_to_measure'] = [qb]
            qb_name = qb.name if not isinstance(qb, str) else qb
            p_mod_key = f'op_code=RO {qb_name}, occurrence=0'
            task.update(dict(
                op_code=f'RO {qb_name}',
                pulse_modifs={  # Applied in parallel_sweep to the first RO
                    # Set operation_type of the RO pulse to None such that no
                    # acquisition is performed
                    f'attr=operation_type, {p_mod_key}': None,
                    # Avoids that the pulses are deactivated by the sweep
                    # points added by self.add_default_ramsey_sweep_points
                    f'attr=pulse_off, {p_mod_key}': 0,
                    # Additional delay to account for ring-down of the pulse
                    f'attr=buffer_length_end, {p_mod_key}': buffer_length_end,
                    # The following renaming avoids that parallel_sweep thinks
                    # that there already is a RO in the sweep block.
                    # (Note that this modification has to be the last one
                    # because p_mod_key will no longer match the pulse
                    # afterwards.)
                    f'attr=op_code, {p_mod_key}': f'noRO {qb_name}',
                },
            ))

        super().__init__(
            qubits=qubits,
            task_list=task_list,
            nr_phases=kw.get('nr_phases', 4),  # fitting needs >=4 phases
            endpoint=kw.get('endpoint', False),
            reset_phases_before_measurement=False,  # Only needed for DynPhase
            repeat_ro=False,  # Repeat patterns don't work if we sweep RO params
            **kw,
        )

    def add_default_sweep_points(self, **kw):
        """
        Adds default sweep points for the measurement-induced dephasing
        experiment. These are:
        - Ramsey phases (in dimension 0)
        :param kw: kwargs passed to add_default_ramsey_sweep_points
        """
        self.sweep_points = self.add_default_ramsey_sweep_points(
            self.sweep_points, tile=1, **kw)

    def run_analysis(self, **kw):
        self.analysis = tda.MeasurementInducedDephasingAnalysis(
            t_start=self.timestamp)

    def run_update(self, **kw):
        pass  # Bypass the update performed in the DynPhase experiment


def measure_flux_pulse_timing_between_qubits(task_list, pulse_length,
                                             analyze=True, label=None, **kw):
    '''
    uses the Chevron measurement to sweep the delay between the two flux pulses
    in the FLIP gate, finds symmmetry point and
    :param task_list:
    :param pulse_length: single float
    :param analyze:
    :param label:
    :param kw:
    :return:
    '''
    if label is None:
        label = 'Flux_pulse_timing_between_qubits_{}_{}'.format(task_list[0][
                                                                    'qbc'].name,
                                                               task_list[0][
                                                                   'qbt'].name)
    pulse_lengths = np.array([pulse_length])
    sweep_points = SweepPoints('pulse_length', pulse_lengths, 's',
                                      dimension=1)
    qe = Chevron(task_list, sweep_points=sweep_points, analyze=False,
             label=label, **kw)
    if analyze:
        qe.analysis = tda.FluxPulseTimingBetweenQubitsAnalysis(
            qb_names=[task_list[0]['qbr']])
    return qe


class Chevron(CalibBuilder):
    """
    Chevron Measurement for CZ gates by performing a flux pulse after
    bringing the qubits to an init state (by default '11). This is a
    multitasking experiment, see docstrings of MultiTaskingExperiment and of
    CalibBuilder for general information. Each task corresponds to the
    characterization of a particular CZ gate specified by the control qubit
    qbc and the target qubit qbt (keys in the task), i.e., multiple gates
    can be characterized in parallel.

    Sequence for each task (for further info and possible parameters of
    the task, see the docstring of the method sweep_block):

        qbc:    |X180|  ---   |  fluxpulse   |
        qbt:    |X180|  --------------------------------------  |RO|

        Note: in case of a FLIP gate or in case of push-away pulses,
        flux pulses are on multiple qubits.

        Note: the class can also be used to characterize other kinds of
        two-qubit gates (e.g., SWAP gates) by setting the parameter
        cz_pulse_name (see docstring of CircuitBuilder) to point to the
        desired two qubit gates. Since each kind of gate might bring its own
        subtleties, this is an experimental feature that should be used with
        care by expert users (a future version might include more stable
        code for other kinds of gates). In this context, note that parameter
        names like qbc, qbt, num_cz_gates, etc. need to be re-interpreted
        in an appropriate way if a non-CZ gate is characterized.

    Sweep points in dimension 0 and 1 will be interpreted as parameters of
    the flux pulse (CZ gate).

    :param kw: keyword arguments.
        Can be used to provide keyword arguments to sweep_n_dim, autorun,
        and to the parent class.

        The following keyword arguments will be copied as a key to tasks
        that do not have their own value specified (see docstring of
        sweep_block):
        - num_cz_gates
        - init_state
    """
    kw_for_task_keys = ['num_cz_gates', 'init_state']
    task_mobj_keys = ['qbc', 'qbt']
    default_experiment_name = 'Chevron'

    @assert_not_none('task_list')
    def __init__(self, task_list, sweep_points=None, **kw):
        try:
            for d in task_list + [kw]:
                if 'qbr' in d:
                    log.warning(
                        "Chevron: the argument qbr is deprecated and will be "
                        "ignored. The argument ro_qubits can be used to restrict"
                        "the readout to a subset of qubits.")
                    d.pop('qbr')

            super().__init__(task_list, sweep_points=sweep_points, **kw)

            # Preprocess sweep points and tasks before creating the sequences
            self.preprocessed_task_list = self.preprocess_task_list(**kw)
            # Chevron takes care of the init state inside the task, so we
            # have to make sure that we do not pass init_state to parallel_sweep.
            kw.pop('init_state', None)
            # the block alignments are for: prepended pulses, initial
            # rotations, flux pulse
            self.sequences, self.mc_points = self.parallel_sweep(
                self.preprocessed_task_list, self.sweep_block,
                block_align = ['center', 'end', 'center'], **kw)

            self.autorun(**kw)  # run measurement & analysis if requested in kw
        except Exception as x:
            self.exception = x
            traceback.print_exc()

    def sweep_block(self, sweep_points, qbc, qbt, num_cz_gates=1,
                    init_state='11', max_flux_length=None,
                    prepend_pulse_dicts=None, **kw):
        """
        This function creates the blocks for a single Chevron measurement
        task, see the pulse sequence in the class docstring.

        :param sweep_points: SweepPoints object
        :param qbc: control qubit (= 1st gate qubit)
        :param qbt: target qubit (= 2nd gate qubit)
        :param num_cz_gates: number of sequential CZ gates, default: 1
        :param init_state: initial states of qbc and qbt (default: '11')
            Init in f level is currently not supported!
        :param max_flux_length: (float, default: None) the duration of the
            longest possible flux pulse. If None, it will be determined
            automatically. This length is used to reserve a fixed time window
            for the flux pulse in order to have a uniform sequence length,
            no matter how long the individual flux pulse is. Note that the
            parameter is understood as a net pulse length, and buffer times
            are automatically added if applicable.
        :param prepend_pulse_dicts: (dict) prepended pulses, see
            CircuitBuilder.block_from_pulse_dicts
        :param kw: further keyword arguments:
            cz_pulse_name: task-specific prefix of CZ gates (overwrites
                global choice passed to the class init)
        """

        # create prepended pulses (pb)
        pb = self.block_from_pulse_dicts(prepend_pulse_dicts)

        # create pulses for initial rotations (ir)
        pulse_modifs = {'all': {'element_name': 'initial_rots_el'}}
        ir = self.block_from_ops('initial_rots',
                                 [f'{self.STD_INIT[init_state[0]][0]} {qbc}',
                                  f'{self.STD_INIT[init_state[1]][0]} {qbt}'],
                                 pulse_modifs=pulse_modifs)
        ir.pulses[1]['ref_point_new'] = 'end'

        # create flux pulse (fp)
        # Calling get_cz_operation_name() allows to take into account a
        # custom cz_pulse_name that was potentially provided in kw.
        fp = self.block_from_ops('flux', [f"{kw.get('cz_pulse_name', 'CZ')} "
                                          f"{qbc} {qbt}"] * num_cz_gates)

        # All sweep points are interpreted as parameters of the flux pulse,
        # except if they are pulse modifier sweep points (see docstring of
        # Block.build).
        for k in list(sweep_points[0].keys()) + list(
                sweep_points.get_sweep_dimension(1, default={}).keys()):
            if '=' not in k:  # '=' indicates a pulse modifier sweep point
                for p in fp.pulses:
                    p[k] = ParametricValue(k)

        # Reserve a time window (w), whose duration is specified by
        # max_flux_length.
        max_flux_length = self.max_pulse_length(fp.pulses[0], sweep_points,
                                                max_flux_length)
        w = self.block_from_ops('wait', [])
        w.block_end.update({'pulse_delay': max_flux_length * num_cz_gates})
        # Center-align the current flux pulse within the avaiable time window.
        fp_w = self.simultaneous_blocks('sim', [fp, w], block_align='center')

        # return all generated blocks (parallel_sweep will arrange them)
        return [pb, ir, fp_w]

    def guess_label(self, **kw):
        """
        Default label with Chevron-specific information
        :param kw: keyword arguments
        """
        if self.label is None:
            self.label = self.experiment_name
            for t in self.task_list:
                self.label += f"_{t['qbc']}{t['qbt']}"

    def get_meas_objs_from_task(self, task):
        """
        Returns a list of all measure objects of a task. In case of
        Chevron, this list includes qbc and qbt.
        :param task: a task dictionary
        :return: list of qubit objects (if available) or names
        """
        qbs = self.get_qubits([task['qbc'], task['qbt']])
        return qbs[0] if qbs[0] is not None else qbs[1]

    def run_analysis(self, analysis_kwargs=None, **kw):
        """
        Runs analysis and stores analysis instance in self.analysis.
        :param analysis_kwargs: (dict) keyword arguments for analysis
        :param kw: currently ignored
        :return: the analysis instance
        """
        if analysis_kwargs is None:
            analysis_kwargs = {}
        if 'options_dict' not in analysis_kwargs:
            analysis_kwargs['options_dict'] = {}
        if 'TwoD' not in analysis_kwargs['options_dict']:
            if len(self.sweep_points) == 2:
                analysis_kwargs['options_dict']['TwoD'] = True
        self.analysis = tda.MultiQubit_TimeDomain_Analysis(
            qb_names=self.meas_obj_names,
            t_start=self.timestamp, **analysis_kwargs)
        return self.analysis

    @classmethod
    def gui_kwargs(cls, device):
        pulse_pars = odict({
            'pulse_length': 's',
            'amplitude': 'V',
            'amplitude2': 'V',
            'trans_amplitude': 'V',
            'trans_amplitude2': 'V',
            'amplitude_offset': 'V',
            'amplitude_offset2': 'V',
            'trans_length': 's',
            'buffer_length_start': 's',
            'buffer_length_end': 's',
            'extra_buffer_aux_pulse': 's',
            'channel_relative_delay': 's',
            'gaussian_filter_sigma': 's',
        })
        # move first param to the end for the second sweep dimension
        first_param = list(pulse_pars.keys())[0]
        pulse_pars2 = deepcopy(pulse_pars)
        pulse_pars2.pop(first_param)
        pulse_pars2[first_param] = pulse_pars[first_param]
        d = super().gui_kwargs(device)
        d['kwargs'][MultiTaskingExperiment.__name__]['cal_states'] = (str, 'gef')
        d['task_list_fields'].update({
            Chevron.__name__: odict({
                'qbc': ((qb_mod.QuDev_transmon, 'single_select'), None),
                'qbt': ((qb_mod.QuDev_transmon, 'single_select'), None),
                'num_cz_gates': (int, 1),
                'init_state': (CircuitBuilder.STD_INIT, '11'),
                'max_flux_length': (float, None),
            })
        })
        d['sweeping_parameters'].update({
            Chevron.__name__: {
                0: pulse_pars,
                1: pulse_pars2,
            }
        })
        return d
