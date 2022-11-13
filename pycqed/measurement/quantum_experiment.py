import traceback

import time
import numpy as np
from pycqed.analysis_v3 import helper_functions

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.utilities.general import temporary_value
from pycqed.utilities.timer import Timer, TimedMetaClass
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement import sweep_functions as swf
import pycqed.measurement.awg_sweep_functions as awg_swf
from pycqed.measurement import multi_qubit_module as mqm
import pycqed.analysis_v2.base_analysis as ba
import pycqed.utilities.general as general
from copy import copy, deepcopy
from collections import OrderedDict as odict
from pycqed.measurement.sweep_points import SweepPoints
import itertools
import logging
from pycqed.gui.waveform_viewer import WaveformViewer
log = logging.getLogger(__name__)


class QuantumExperiment(CircuitBuilder, metaclass=TimedMetaClass):
    """
    Base class for Experiments with pycqed. A QuantumExperiment consists of
    3 main parts:
    - The __init__(), which takes care of initializing the parent class
     (CircuitBuilder) and setting all the attributes of the quantum experiment
    - the run_measurement(), which is the skeleton of any measurement in pycqed.
      This function should *not* be modified by child classes
    - the run_analysis(), which defaults to calling BaseDataAnalysis. This function
      may be overwritten by child classes to start measurement-specific analysis

    """
    TIMED_METHODS = ["run_analysis"]
    _metadata_params = {'cal_points', 'preparation_params', 'sweep_points',
                        'channel_map', 'meas_objs'}
    # The following string can be overwritten by child classes to provide a
    # default value for the kwarg experiment_name. None means that the name
    # of the first sequences will be used.
    default_experiment_name = None

    def __init__(self, dev=None, qubits=None, operation_dict=None,
                 meas_objs=None, classified=False, MC=None,
                 label=None, exp_metadata=None, upload=True, measure=True,
                 analyze=True, temporary_values=(), drive="timedomain",
                 sequences=(), sequence_function=None, sequence_kwargs=None,
                 plot_sequence=False, filter_segments_mask=None, df_kwargs=None, df_name=None,
                 timer_kwargs=None, mc_points=None, sweep_functions=(awg_swf.SegmentHardSweep,
                                                  awg_swf.SegmentSoftSweep),
                 harmonize_element_lengths=False,
                 compression_seg_lim=None, force_2D_sweep=True, callback=None,
                 callback_condition=lambda : True, **kw):
        """
        Initializes a QuantumExperiment.

        Args:
            dev (Device): Device object used for the experiment. Defaults to None.
            qubits (list): list of qubits used for the experiment (e.g. a subset of
                qubits on the device). Defaults to None. (see circuitBuilder for more
                details).
            operation_dict (dict): dictionary with operations. Defaults to None.
                (see circuitBuilder for more details).
            meas_objs (list): list of measure object (e.g., qubits) to be read
                out (i.e. for which the detector functions will be
                prepared). Defaults to self.qubits (attribute set by
                CircuitBuilder). Required for run_measurement() when qubits
                is None.
            classified (bool): whether
            MC (MeasurementControl): MeasurementControl object. Required for
                run_measurement() if qubits is None and device is None.
            label (str): Measurement label
            exp_metadata (dict): experimental metadata saved in hdf5 file
            upload (bool): whether or not to upload the sequences to the AWGs
            measure (bool): whether or not to measure
            analyze (bool): whether or not to analyze
            temporary_values (list): list of temporary values with the form:
                [(Qcode_param_1, value_1), (Qcode_param_2, value_2), ...]
            drive (str): qubit configuration.
            sequences (list): list of sequences for the experiment. Note that
                even in the case of a single sequence, a list is required.
                Required if sequence_function is None.
            sequence_function (callable): functions returning the sequences,
                see self._prepare_sequences() for more details. Required for
                run_measurement if sequences is None
            sequence_kwargs (dict): keyword arguments passed to the sequence_function.
                see self._prepare_sequences()
            filter_segments_mask (array of bool): An array with dimension
                n_0 x n_1, indicating which segments need to be measured.
                Here, n_1 is the number of sweep points in dimension 1 (soft sweep)
                and n_0 <= N_0, where N_0 is the number of sweep points in
                dimension 0 (hard sweep) including calibration points. If
                n_0 < N_0, segments with index larger than n_0 will always
                be measured (typical use case: calibration points, i.e.,
                let n_0 be the number of sweep pooints without calibration
                points).
                In case of a hard sweep in dimension 0, the lower-level
                implementation in FilteredSegmentSweep and Pulsar
                currently only supports a single consecutive range of
                segments to be measured in each row of this array (plus the
                segments with index >n_0, which are always measured). To
                fulfill this requirement, QuantumExperiment will internally
                change False-values to True in this array if needed, i.e.,
                it can happen that segments are measured even though their
                entry in this array is set to False.
            df_kwargs (dict): detector function keyword arguments.
            timer_kwargs (dict): keyword arguments for timer. See pycqed.utilities.timer.
                Timer.
            df_name (str): detector function name.
            mc_points (tuple): tuple of 2 lists with first and second dimension
                measurement control points (previously also called sweep_points,
                but name has changed to avoid confusion with SweepPoints):
                [first_dim_mc_points, second_dim_mc_points]. MC points
                correspond to measurement_control sweep points i.e. sweep points
                directly related to the instruments, e.g. segment readout index.
                Not required when using sweep_functions SegmentSoftSweep and
                SegmentHardSweep as these may be inferred from the sequences objects.
                In case other sweep functions are used (e.g. for sweeping instrument
                parameters), then the sweep points must be specified. Note that the list
                must always have two entries. E.g. for a 1D sweep of LO frequencies,
                mc_points should be of the form: (freqs, [])
            sweep_functions (tuple): tuple of sweepfunctions. Similarly to mc_points,
                sweep_functions has 2 entries, one for each dimension. Defaults to
                SegmentHardSweep for the first sweep dimensions and SegmentSoftSweep
                for the second dimension.
            harmonize_element_lengths (bool, default False): whether it
                should be ensured for all AWGs and all elements that the
                element length is the same in all sequences. Use case: If
                pulsar.use_sequence_cache and pulsar.AWGX_use_placeholder_waves
                are activated for a ZI HDAWG, harmonized element lengths across
                a soft sweep avoid recompilation of SeqC code during the sweep
                (replacing binary waveform data is sufficient in this case).
            compression_seg_lim (int): maximal number of segments that can be in a
                single sequence. If not None and the QuantumExperiment is a 2D sweep
                with more than 1 sequence, and the sweep_functions are
                (SegmentHardSweep, SegmentSoftsweep), then the quantumExperiment
                will try to compress the sequences, see Sequence.compress_2D_sweep.
            force_2D_sweep (bool): whether or not to force a two-dimensional sweep.
                In that case, even if there is only one sequence, a second
                sweep_function dimension is added. The idea is to use this more
                and more to generalize data format passed to the analysis.
            callback (func): optional function to call after run_analysis() in
                autorun(). All arguments passed to autorun will be passed down to
                the callback.
            callback_condition (func): function returning a bool to decide whether or
                not the callback function should be executed. Defaults to always True.
            **kw:
                further keyword arguments are passed to the CircuitBuilder __init__
        """
        self.timer = Timer('QuantumExperiment', **timer_kwargs if timer_kwargs is
                                                                  not None else {})
        if qubits is None and dev is None and operation_dict is None:
            # if no qubits/devive/operation_dict are provided, use empty
            # list to skip iterations over qubit lists
            qubits = []
            if meas_objs is not None:
                operation_dict = mqm.get_operation_dict(meas_objs)
        super().__init__(dev=dev, qubits=qubits, operation_dict=operation_dict,
                         **kw)

        self.exp_metadata = exp_metadata
        if self.exp_metadata is None:
            self.exp_metadata = {}

        self.create_meas_objs_list(**kw, meas_objs=meas_objs)
        self.MC = MC

        self.classified = classified
        self.label = label
        self.upload = upload
        self.measure = measure
        self.temporary_values = list(temporary_values)
        self.analyze = analyze
        self.drive = drive
        self.callback = callback
        self.callback_condition = callback_condition
        self.plot_sequence = plot_sequence

        self.sequences = list(sequences)
        self.sequence_function = sequence_function
        self.sequence_kwargs = {} if sequence_kwargs is None else sequence_kwargs
        self.filter_segments_mask = filter_segments_mask
        self.sweep_points = self.sequence_kwargs.get("sweep_points", None)
        self.mc_points = mc_points if mc_points is not None else [[], []]
        self.sweep_functions = list(sweep_functions)
        self.force_2D_sweep = force_2D_sweep
        self.compression_seg_lim = compression_seg_lim
        self.harmonize_element_lengths = harmonize_element_lengths
        # The experiment_name might have been set by the user in kw or by a
        # child class as an attribute. Otherwise, the default None will
        # trigger guess_label to use the sequence name.
        self.experiment_name = kw.pop(
            'experiment_name', getattr(self, 'experiment_name',
                                       self.default_experiment_name))
        self.timestamp = None
        self.analysis = None

        # detector and sweep functions
        default_df_kwargs = {'det_get_values_kws':
                                 {'classified': self.classified,
                                  'correlated': False,
                                  'thresholded': True,
                                  'averaged': True}}
        self.df_kwargs = default_df_kwargs if df_kwargs is None else df_kwargs
        if df_name is not None:
            self.df_name = df_name
            if 'classif' in df_name:
                self.classified = True
        else:
            self.df_name = 'int_avg{}_det'.format('_classif' if self.classified else '')
        self.df = None

        # determine data type
        if "log" in self.df_name or not \
                self.df_kwargs.get("det_get_values_kws",
                                   {}).get('averaged', True):
            data_type = "singleshot"
        else:
            data_type = "averaged"

        self.exp_metadata.update(kw)
        self.exp_metadata.update({'classified_ro': self.classified,
                                  'cz_pulse_name': self.cz_pulse_name,
                                  'data_type': data_type})
        self.waveform_viewer = None

    def create_meas_objs_list(self, meas_objs=None, **kwargs):
        """
        Creates a default list for self.meas_objs if meas_objs is not provided,
        and creates the list self.meas_obj_names.
        Args:
            meas_objs (list): a list of measurement objects (or None for
                default, which is self.qubits)
        """
        self.meas_objs = self.qubits if meas_objs is None else meas_objs
        self.meas_obj_names = [m.name for m in self.meas_objs]

    def _update_parameters(self, overwrite_dicts=True, **kwargs):
        """
        Update all attributes of the quantumExperiment class.
        Args:
            overwrite_dicts (bool): whether or not to overwrite
                attributes that are dictionaries. If False,
                then dictionaries are updated.
            **kwargs: any attribute of the QuantumExperiment class


        """
        for param_name, param_value in kwargs.items():
            if hasattr(self, param_name):
                if isinstance(param_value, dict) and not overwrite_dicts:
                    getattr(self, param_name).update(param_value)
                else:
                    setattr(self, param_name, param_value)

    @Timer()
    def run_measurement(self, save_timers=True, **kw):
        """
        Runs a measurement. Any keyword argument passes to this function that
        is also an attribute of the QuantumExperiment class will be updated
        before starting the experiment

        Args:
            save_timers (bool): whether timers should be saved to the hdf
            file at the end of the measurement (default: True).
        Returns:

        """
        self._update_parameters(**kw)
        assert self.meas_objs is not None, 'Cannot run measurement without ' \
                                           'measure objects.'
        if len(self.mc_points) == 1:
            self.mc_points = [self.mc_points[0], []]

        exception = None
        with temporary_value(*self.temporary_values):
            # Perpare all involved qubits. If not available, prepare
            # all measure objects.
            mos = self.qubits if self.qubits else self.meas_objs
            for m in mos:
                m.prepare(drive=self.drive)

            # create/retrieve sequence to run
            self._prepare_sequences(self.sequences, self.sequence_function,
                                    self.sequence_kwargs)

            # configure measurement control (mc_points, detector functions)
            mode = self._configure_mc()

            self.guess_label(**kw)

            self.update_metadata()

            # run measurement
            try:
                self.MC.run(name=self.label, exp_metadata=self.exp_metadata,
                            mode=mode)
            except (Exception, KeyboardInterrupt) as e:
                exception = e  # exception will be raised below
        self.extract_timestamp()
        if save_timers:
            self.save_timers()
        if exception is not None:
            raise exception

    def update_metadata(self):
        # make sure that all metadata params are up to date
        for name in self._metadata_params:
            if hasattr(self, name):
                value = getattr(self, name)
                try:
                    if name in ('cal_points', 'sweep_points') and \
                            value is not None:
                        old_val = np.get_printoptions()['threshold']
                        np.set_printoptions(threshold=np.inf)
                        self.exp_metadata.update({name: repr(value)})
                        np.set_printoptions(threshold=old_val)
                    elif name in ('meas_objs', "qubits") and value is not None:
                        self.exp_metadata.update(
                            {name: [qb.name for qb in value]})
                    else:
                        self.exp_metadata.update({name: value})
                except Exception as e:
                    log.error(
                        f"Could not add {name} with value {value} to the "
                        f"metadata")
                    raise e

    def extract_timestamp(self):
        try:
            self.timestamp = self.MC.data_object._datemark + '_' \
                             + self.MC.data_object._timemark
        except Exception:
            pass  # if extraction fails, keep the old value (None from init)

    def guess_label(self, **kwargs):
        """
        Creates a default label.

        Returns:

        """
        if self.label is None:
            if self.experiment_name is None:
                self.experiment_name = self.sequences[0].name
            self.label = self.experiment_name
            _, qb_names = self.get_qubits(self.qubits)
            if self.dev is not None:
                self.label += self.dev.get_msmt_suffix(self.meas_obj_names)
            else:
                # guess_label is called from run_measurement -> we have qubits
                self.label += mqm.get_multi_qubit_msmt_suffix(self.meas_objs)

    @Timer()
    def run_analysis(self, analysis_class=None, analysis_kwargs=None, **kw):
        """
        Launches the analysis.
        Args:
            analysis_class: Class to use for the analysis
            analysis_kwargs: keyword arguments passed to the analysis class

        Returns: analysis object

        """
        if analysis_class is None:
            analysis_class = ba.BaseDataAnalysis
        if analysis_kwargs is None:
            analysis_kwargs = {}
        analysis_kwargs.setdefault('t_start', self.timestamp)
        self.analysis = analysis_class(**analysis_kwargs)
        return self.analysis

    def autorun(self, **kw):
        if self.measure:
            try:
                # Do not save timers here since they will be saved below.
                self.run_measurement(save_timers=False, **kw)
            except (Exception, KeyboardInterrupt) as e:
                self.save_timers()
                raise e
            # analyze and call callback only when measuring
            if self.analyze:
                self.run_analysis(**kw)
            if self.callback is not None and self.callback_condition():
                self.callback(**kw)
            self.save_timers()  # for now store timers only if creating new file
        return self

    def serialize(self, omitted_attrs=('MC', 'device', 'qubits')):
        """
        Map a Quantum experiment to a large dict for hdf5 storage/pickle object,
        etc.
        Returns:

        """
        raise NotImplementedError()

    @Timer()
    def _prepare_sequences(self, sequences=None, sequence_function=None,
                           sequence_kwargs=None):
        """
        Prepares/build sequences for a measurement.
        Args:
            sequences (list): list of sequences to run. Optional. If not given
                then a sequence_function from which the sequences can be created
                is required.
            sequence_function (callable): sequence function to generate sequences..
                Should return with one of the following formats:
                    - a list of sequences: valid if the first and second
                        sweepfunctions are  SegmentHardSweep and SegmentSoftsweep
                        respectively.
                    - a sequence: valid if the sweepfunction is SegmentHardsweep
                    - One of the following tuples:
                        (sequences, mc_points_tuple), where mc_points_tuple is a
                        tuple in which each entry corresponds to a dimension
                        of the sweep. This is the preferred option.
                        For backwards compatibility, the following two tuples are
                        also accepted:
                        (sequences, mc_points_first_dim, mc_points_2nd_dim)
                        (sequences, mc_points_first_dim)

            sequence_kwargs (dict): arguments to pass to the sequence
                function if sequence_function is not None. If
                sequence_function is None, the following entries in this
                dict are supported:
                - extra_sequences (list): a list of additional sequences to
                    measure. This is useful for combining sequences that are
                    automatically generated by a child-class of
                    QuantumExperiment with user-provided sequences into a
                    single experiment (e.g., for measuring them in a single
                    upload by specifying a sufficiently high
                    compression_seg_lim). The user has to ensure that the
                    extra sequences are compatible with the normal sequences
                    of the QuantumExperiment, e.g., in terms of number of
                    acquisition elements.
                - aux_triggers (dict): a dict where each key is a tuple of a
                    sequence index and segment index, and the corresponding
                    value is a list of channel names. This will add trigger
                    pulses on the given channels for the first pulse of
                    the segment indicated by the key.
        Returns:

        """

        if sequence_kwargs is None:
            sequence_kwargs = {}
        if sequence_function is not None:
            # build sequence from function
            seq_info = sequence_function(**sequence_kwargs)

            if isinstance(seq_info, list):
                self.sequences = seq_info
            elif isinstance(seq_info, Sequence):
                self.sequences = [seq_info]
            elif len(seq_info) == 3: # backwards compatible 2D sweep
                self.sequences, \
                    (self.mc_points[0], self.mc_points[1]) = seq_info
            elif len(seq_info) == 2:
                if np.ndim(seq_info[1]) == 1:
                    # backwards compatible 1D sweep
                    self.sequences, self.mc_points[0] = seq_info
                else:
                    self.sequences, self.mc_points = seq_info

            # ensure self.sequences is a list
            if np.ndim(self.sequences) == 0:
                self.sequences = [self.sequences]
        elif sequences is not None:
            extra_seqs = deepcopy(sequence_kwargs.get('extra_sequences', []))
            for seq in extra_seqs:
                seq.name = 'Extra' + seq.name
            self.sequences = sequences + extra_seqs
            if len(self.mc_points) > 1 and len(self.mc_points[1]):
                # mc_points are set and won't be generated automatically.
                # We have to add additional points for the extra sequences.
                self.mc_points[1] = np.concatenate([
                    self.mc_points[1],
                    np.arange(len(extra_seqs)) + self.mc_points[1][-1] + 1])
            aux_triggers = sequence_kwargs.get('aux_triggers', None)
            if aux_triggers is not None:
                for (i, j), v in aux_triggers.items():
                    self.sequences[i][j].unresolved_pulses[
                        0].pulse_obj.trigger_channels = v

        # check sequence
        assert len(self.sequences) != 0, "No sequence found."

        if self.plot_sequence:
            self.plot()

    @Timer()
    def _configure_mc(self, MC=None):
        """
        Configure the measurement control (self.MC) for the measurement.
        This includes setting the sweep points and the detector function.
        By default, SegmentHardSweep is the sweepfunction used for the first
        dimension and SegmentSoftSweep is the sweepfunction used for the second
        dimension. In case other sweepfunctions should be used, self.sweep_functions
        should be modified prior to the call of this function.

        Returns:
            mmnt_mode (str): "1D" or "2D"
        """
        # ensure measurement control is set
        self._set_MC(MC)

        # configure mc_points
        if len(self.mc_points[0]) == 0: # first dimension mc_points not yet set
            if self.sweep_functions[0] == awg_swf.SegmentHardSweep:
                # first dimension mc points can be retrieved as
                # ro_indices from sequence
                self.mc_points[0] = np.arange(self.sequences[0].n_acq_elements())
            else:
                raise ValueError("The first dimension of mc_points must be provided "
                                 "with sequence if the sweep function isn't "
                                 "'SegmentHardSweep'.")

        if len(self.sequences) > 1 and len(self.mc_points[1]) == 0:
            if self.sweep_functions[1] == awg_swf.SegmentSoftSweep:
                # 2nd dimension mc_points can be retrieved as sequence number
                self.mc_points[1] = np.arange(len(self.sequences))
            elif self.sweep_points is not None and len(self.sweep_points) > 1:
                # second dimension can be inferred from sweep points
                self.mc_points[1] = self.sweep_points.get_sweep_params_property(
                    'values', 1)
            else:
                raise ValueError("The second dimension of mc_points must be provided "
                                 "if the sweep function isn't 'SegmentSoftSweep' and"
                                 "no sweep_point object is given.")

        # force 2D sweep if needed (allow 1D sweep for backwards compatibility)
        if len(self.mc_points[1]) == 0 and self.force_2D_sweep:
            self.mc_points[1] = np.array([0]) # force 2d with singleton

        # set mc points
        if len(self.sequences) > 1:
            # compress 2D sweep
            if self.compression_seg_lim is not None:
                if self.sweep_functions == [awg_swf.SegmentHardSweep,
                                            awg_swf.SegmentSoftSweep]:
                    self.sequences, self.mc_points[0], \
                    self.mc_points[1], cf = \
                        self.sequences[0].compress_2D_sweep(self.sequences,
                                                            self.compression_seg_lim,
                                                            True,
                                                            self.mc_points[0])
                    self.exp_metadata.update({'compression_factor': cf})
                else:
                    log.warning("Sequence compression currently does not support"
                                "sweep_functions different than (SegmentHardSweep,"
                                " SegmentSoftSweep). This could easily be implemented"
                                "by modifying Sequence.compress_2D_sweep to accept"
                                "mc_points and do the appropriate reshaping. Feel"
                                "free to make a pull request ;). Skipping compression"
                                "for now.")

        if self.harmonize_element_lengths:
            Sequence.harmonize_element_lengths(self.sequences)
        try:
            sweep_param_name = list(self.sweep_points[0])[0]
            unit = self.sweep_points.get_sweep_params_property(
                'unit', 0, param_names=sweep_param_name)
        except TypeError:
            sweep_param_name, unit = "None", ""
        if self.sweep_functions[0] == awg_swf.SegmentHardSweep:
            sweep_func_1st_dim = self.sweep_functions[0](
                sequence=self.sequences[0], upload=self.upload,
                parameter_name=sweep_param_name, unit=unit)
        elif isinstance(self.sweep_functions[0], swf.UploadingSweepFunction):
            sweep_func_1st_dim = self.sweep_functions[0]
            sweep_func_1st_dim.sequence = self.sequences[0]
        else:
            # Check whether it is a nested sweep function whose first
            # sweep function is a SegmentHardSweep class as placeholder.
            swfs = getattr(self.sweep_functions[0], 'sweep_functions',
                            [None])
            if (len(swfs) != 0 and swfs[0] == awg_swf.SegmentHardSweep):
                # Replace the SegmentHardSweep placeholder by a properly
                # configured instance of SegmentHardSweep.
                if len(swfs) > 1:
                    # make sure that units are compatible
                    unit = getattr(swfs[1], 'unit', unit)
                swfs[0] = awg_swf.SegmentHardSweep(
                    sequence=self.sequences[0], upload=self.upload,
                    parameter_name=sweep_param_name, unit=unit)
            # In case of an unknown sweep function type, it is assumed
            # that self.sweep_functions[0] has already been initialized
            # with all required parameters and can be directly passed to
            # MC.
            sweep_func_1st_dim = self.sweep_functions[0]

        self.MC.set_sweep_function(sweep_func_1st_dim)
        self.MC.set_sweep_points(self.mc_points[0])

        swf_prep_pulsar = None
        # set second dimension sweep function
        if len(self.mc_points[1]) > 0: # second dimension exists
            try:
                sweep_param_name = list(self.sweep_points[1])[0]
                unit = self.sweep_points.get_sweep_params_property(
                    'unit', 1, param_names=sweep_param_name)
            except TypeError:
                sweep_param_name, unit = "None", ""
            if self.sweep_functions[1] == awg_swf.SegmentSoftSweep:
                sweep_func_2nd_dim = awg_swf.SegmentSoftSweep(
                    self.sequences, sweep_param_name, unit)
                # Indicate that df.prepare_pulsar needs to be configured as
                # an upload_finished_callback for this sweep function (to
                # let the detector function restart pulsar if needed).
                swf_prep_pulsar = sweep_func_2nd_dim
            else:
                # Check whether it is a nested sweep function whose first
                # sweep function is a SegmentSoftSweep class as placeholder.
                swfs = getattr(self.sweep_functions[1], 'sweep_functions',
                               [None])
                if (len(swfs) != 0 and swfs[0] == awg_swf.SegmentSoftSweep):
                    # Replace the SegmentSoftSweep placeholder by a properly
                    # configured instance of SegmentSoftSweep.
                    if len(swfs) > 1:
                        # make sure that units are compatible
                        unit = getattr(swfs[1], 'unit', unit)
                    swfs[0] = awg_swf.SegmentSoftSweep(
                        self.sequences,
                        sweep_param_name, unit)
                    # Indicate that df.prepare_pulsar needs to be configured as
                    # an upload_finished_callback for this sweep function (to
                    # let the detector function restart pulsar if needed).
                    swf_prep_pulsar = swfs[0]
                # In case of an unknown sweep function type, it is assumed
                # that self.sweep_functions[1] has already been initialized
                # with all required parameters and can be directly passed to
                # MC.
                sweep_func_2nd_dim = self.sweep_functions[1]

            if self.filter_segments_mask is not None and \
                    self.compression_seg_lim is not None:
                log.warning("Combining compression_seg_lim and "
                            "filter_segments_mask is not supported. Ignoring "
                            "filter_segments_mask.")
            elif (self.filter_segments_mask is not None and
                  sweep_func_1st_dim.sweep_control == 'hard'):
                mask = np.array(self.filter_segments_mask)
                # Only segments with indices included in the mask can be
                # filtered out. The others will always be measured.
                for seq in self.sequences:
                    for i, seg in enumerate(seq.segments.values()):
                        if i < mask.shape[0]:
                            seg.allow_filter = True
                # Create filter lookup table in the format expected by
                # FilteredSweep: each key is a soft sweep point and the
                # respective value is a tuple two of segment indices
                # indicating the range of segments to be measured.
                # The conversion to a tuple of start index and end index may
                # require to measure segments that have a False-entry in the
                # filter_segments_mask, see the class docstring.
                filter_lookup = {}
                for i, sp in enumerate(self.mc_points[1]):
                    if i >= mask.shape[1]:
                        # measure everything
                        filter_lookup[sp] = (0, 32767)
                    elif True in mask[:, i]:
                        # measure from the first True up to the last True
                        filter_lookup[sp] = (
                            list(mask[:, i]).index(True),
                            mask.shape[0] - list(mask[:, i])[::-1].index(
                                True) - 1)
                    else:
                        # measure nothing (by setting last < first)
                        filter_lookup[sp] = (1, 0)
                sweep_func_2nd_dim = swf.FilteredSegmentSweep(
                        self.sequences[0], filter_lookup, [sweep_func_2nd_dim])
            elif self.filter_segments_mask is not None:
                mask = np.array(self.filter_segments_mask)
                filter_lookup = {
                    mcp: mask[:, i] if i < mask.shape[1] else []
                    for i, mcp in enumerate(self.mc_points[1])}
                sweep_func_2nd_dim = swf.FilteredSoftSweep(
                    filter_lookup, [sweep_func_2nd_dim])

            self.MC.set_sweep_function_2D(sweep_func_2nd_dim)
            self.MC.set_sweep_points_2D(self.mc_points[1])

        # Below, start_pulsar is set to False since pulsar usually gets
        # started by the detector function (either via df.AWG or via
        # df.prepare_and_finish_pulsar).
        if sweep_func_1st_dim.configure_upload(start_pulsar=False):
            # sweep_func_1st_dim takes care of first upload
            pass
        elif len(self.mc_points[1]) > 0 \
                and sweep_func_2nd_dim.configure_upload(start_pulsar=False):
            # sweep_func_2nd_dim takes care of first upload
            pass
        else:
            self._upload_first_sequence()
            # separate method so that children can override
            # see docstring for default behavior

        # check whether there is at least one measure object
        if len(self.meas_objs) == 0:
            raise ValueError('No measure objects provided. Cannot '
                             'configure detector functions')

        # Configure detector function
        # FIXME: this should be extended to meas_objs that are not qubits
        if sweep_func_1st_dim.sweep_control == 'hard':
            # The following ensures that we use a hard detector if the acq
            # dev provided a sweep function for a hardware IF sweep.
            # Used by IntegratingAveragingPollDetector and its childen,
            # detectors that don't implement this kwarg will ignore it
            self.df_kwargs['single_int_avg'] = False
        self.df = mqm.get_multiplexed_readout_detector_functions(
            self.df_name, self.meas_objs, **self.df_kwargs)
        # Set an upload_finished_callback if the above logic determined that
        # this is needed (i.e., for SegmentSoftSweep).
        if swf_prep_pulsar is not None:
            swf_prep_pulsar.upload_finished_callback = self.df.prepare_pulsar
        self.MC.set_detector_function(self.df)
        meas_obj_value_names_map = self.df.get_meas_obj_value_names_map()
        self.exp_metadata.update(
            {'meas_obj_value_names_map': meas_obj_value_names_map})
        if 'meas_obj_sweep_points_map' not in self.exp_metadata:
            self.exp_metadata['meas_obj_sweep_points_map'] = {}
        if self.MC.soft_repetitions() != 1:
            self.exp_metadata['soft_repetitions'] = self.MC.soft_repetitions()

        if len(self.mc_points[1]) > 0:
            mmnt_mode = "2D"
        else:
            mmnt_mode = "1D"
        return mmnt_mode

    def _set_MC(self, MC=None):
        """
        Sets the measurement control and raises an error if no MC
        could be retrieved from device/qubits objects
        Args:
            MC (MeasurementControl):

        Returns:

        """
        if MC is not None:
            self.MC = MC
        elif self.MC is None:
            try:
                self.MC = self.dev.instr_mc.get_instr()
            except AttributeError:
                try:
                    self.MC = self.meas_objs[0].instr_mc.get_instr()
                except (AttributeError, IndexError):
                    raise ValueError("The Measurement Control (MC) could not "
                                     "be retrieved because no Device/measure "
                                     "objects were found. Pass the MC to "
                                     "run_measurement() or set the MC attribute"
                                     " of the QuantumExperiment instance.")

    # def __setattr__(self, name, value):
    #     """
    #     Observes attributes which are set to this class. If they are in the
    #     _metadata_params then they are automatically added to the experimental
    #     metadata
    #     Args:
    #         name:
    #         value:
    #
    #     Returns:
    #
    #     """
    #     if name in self._metadata_params:
    #         try:
    #             if name in 'cal_points' and value is not None:
    #                 self.exp_metadata.update({name: repr(value)})
    #             elif name in ('meas_objs', "qubits") and value is not None:
    #                 self.exp_metadata.update({name: [qb.name for qb in value]})
    #             else:
    #                 self.exp_metadata.update({name: value})
    #         except Exception as e:
    #             log.error(f"Could not add {name} with value {value} to the "
    #                       f"metadata")
    #             raise e
    #
    #     self.__dict__[name] = value

    def _upload_first_sequence(self):
        if self.upload:
            if hasattr(self, 'sequences') and self.sequences is not None \
                    and len(self.sequences) > 0:
                self.sequences[0].upload()
            else:
                raise ValueError(f'QuantumExperiment needs to have attribute '
                                 f'self.sequences not None and not empty.')

    def save_timers(self, quantum_experiment=True, sequence=True, segments=True, filepath=None):
        if self.MC is None or self.MC.skip_measurement():
            return
        data_file = helper_functions.open_hdf_file(self.timestamp, filepath=filepath, mode="r+")
        try:
            timer_group = data_file.get(Timer.HDF_GRP_NAME)
            if timer_group is None:
                timer_group = data_file.create_group(Timer.HDF_GRP_NAME)
            if quantum_experiment:
                self.timer.save(timer_group)

            if sequence:
                seq_group = timer_group.create_group('Sequences')
                for s in self.sequences:
                    # save sequence timers
                    try:
                        timer_seq_name = s.timer.name
                        # check that name doesn't exist and it case it does, append an index
                        # Note: normally that should not happen (not desirable)
                        if timer_seq_name in seq_group.keys():
                            log.warning(f"Timer with name {timer_seq_name} already "
                                        f"exists in Sequences timers. "
                                        f"Only last instance will be kept")
                        s.timer.save(seq_group)

                        if segments:
                            seg_group = seq_group[timer_seq_name].create_group(timer_seq_name + ".segments")
                            for _, seg in s.segments.items():
                                try:
                                    timer_seg_name = seg.timer.name
                                    # check that name doesn't exist and it case it does, append an index
                                    # Note: normally that should not happen (not desirable)
                                    if timer_seg_name in seg_group.keys():
                                        log.warning(f"Timer with name {timer_seg_name} already "
                                                    f"exists in Segments timers. "
                                                    f"Only last instance will be kept")
                                    seg.timer.save(seg_group)
                                except AttributeError:
                                    pass

                    except AttributeError:
                        pass # in case some sequences don't have timers
        except Exception as e:
            data_file.close()
            raise e


    def plot(self, sequences=0, segments=0, qubits=None,
             save=False, legend=True, **plot_kwargs):
        """
        Plots (a subset of) sequences / segments of the QuantumExperiment
        :param sequences (int, list, "all"): sequences to plot. Can be "all"
        (plot all sequences),
        an integer (index of sequence to plot),  or a list of
        integers/str. If strings are in the list, then plots only sequences
        with the corresponding name.
        :param segments (int, list, "all"): Segments to be plotted.
            If a single index i is provided, then the ith segment will be plot-
            ted for each sequence in `sequences`. Otherwise a list of list of
            indices must be provided: the outer list corresponds to each
            sequence and the inner list to the indices of the segments to plot.
            E.g. segments=[[0,1],[3]] will plot segment 0 and 1 of
                sequence 0 and segment 3 of sequence 1.
            If the string 'all' is provided, then all segments are plotted.
            Plots segment 0 by default.
        :param qubits (list): list of qubits to plot.
            Defaults to self.meas_objs. Qubits can be specified as qubit names
            or qubit objects.
        :param save (bool): whether or not to save the figures in the
            measurement folder.
        :param legend (bool): whether or not to show the legend.
        :param plot_kwargs: kwargs passed on to segment.plot(). By default,
             channel_map is taken from dev.get_channel_map(qubits) if available.
        :return:
        """
        plot_kwargs = deepcopy(plot_kwargs)
        if sequences == "all":
            # plot all sequences
            sequences = self.sequences
        # if the provided sequence is not it a list or tuple, make it a list
        if np.ndim(sequences) == 0:
            sequences = [sequences]
        # get sequence objects from sequence name or index
        sequences = np.ravel([[s for i, s in enumerate(self.sequences)
                               if i == ind or s.name == ind]
                              for ind in sequences])
        if qubits is None:
            qubits = self.meas_objs
        qubits, _ = self.get_qubits(qubits) # get qubit objects
        default_ch_map = \
            self.dev.get_channel_map(qubits) if self.dev is not None else \
                {qb.name: qb.get_channels() for qb in qubits}
        plot_kwargs.update(dict(channel_map=plot_kwargs.pop('channel_map',
                           default_ch_map)))
        plot_kwargs.update(dict(legend=legend))

        if segments == "all":
            # plot all segments
            segments = [range(len(seq.segments)) for seq in sequences]
        elif isinstance(segments, int):
            # single segment from index
            segments = [[segments] for _ in sequences]

        figs_and_axs = []
        for seq, segs in zip(sequences, segments):
            for s in segs:
                s = list(seq.segments.keys())[s]
                if save:
                    try:
                        from pycqed.analysis import analysis_toolbox as a_tools
                        folder = a_tools.data_from_time(self.timestamp,
                                                        folder=self.MC.datadir(),
                                                        auto_fetch=False)
                    except:
                        log.warning('Could not determine folder of current '
                                    'experiment. Sequence plot will be saved in '
                                    'current directory.')
                        folder = "."
                    import os
                    save_path = os.path.join(folder,
                                             "_".join((seq.name, s)) + ".png")
                    save_kwargs = dict(fname=save_path,
                                       bbox_inches="tight")
                    plot_kwargs.update(dict(save_kwargs=save_kwargs,
                                            savefig=True))
                figs_and_axs.append(seq.segments[s].plot(**plot_kwargs))
        # avoid returning a list of Nones (if show_and_close is True)
        return [v for v in figs_and_axs if v is not None] or None

    def __repr__(self):
        return f"QuantumExperiment(dev={getattr(self, 'dev', None)}, " \
               f"qubits={getattr(self, 'qubits', None)})"

    def spawn_waveform_viewer(self, **kwargs):
        """
        Spawns a WaveformViewer window to interactively inspect the control,
        readout and trigger pulses of the QuantumExperiment instance.
        Args:
            **kwargs: Are passed to the __init__ method of WaveformViewer,
                see WaveformViewer docstring.
        """
        if self.waveform_viewer is None:
            self.waveform_viewer = WaveformViewer(self, **kwargs)
        else:
            self.waveform_viewer.spawn_waveform_viewer(**kwargs)

    @classmethod
    def gui_kwargs(cls, device):
        """
        Determines which options of a quantum experiment can be configured
        in the QuantumExperimentGUI. Returns a dictionary with keys
        'kwargs', 'task_list_fields' and 'sweeping_parameters', where the
        corresponding values are themselves ordered dictionaries.

        The 'kwargs', 'task_list_fields' and 'sweeping_parameters' ordered
        dictionaries can be extended in the subclasses of QuantumExperiment.
        The keys of the ordered dictionaries should indicate, in which class
        the configuration options were added to the corresponding ordered
        dictionary. The values of the ordered dictionaries are regular
        dictionaries and contain the name, type and default value of the
        configuration options. Only the configuration options that
        are added to these dictionaries are available in the
        QuantumExperimentGUI.

        The 'kwargs' ordered dict contains the keyword arguments to
        configure/instantiate QuantumExperiment objects. Quantum experiments
        of type MultiTaskingExperiment can be configured by specifying a
        task_list. The 'task_list_fields' ordered dict contains the keyword
        arguments that can be used to configure task lists in the GUI. Finally,
        the 'sweeping_parameters' ordered dict contains the sweep parameters
        that can be selected in the QuantumExperimentGUI (e.g. pulse
        amplitude in the case of a Rabi experiment).

        For the lowest level dict, the keys correspond to the keyword
        argument names of the configuration options, while the values are
        tuples. The first entry of a tuple indicates the field type in
        which the configuration option can be set (e.g. if the first entry
        is bool, the GUI will provide a checkbox to set the value of the
        configuration option). See the docstring of the
        create_field_from_field_information method of the
        QuantumExperimentGUI class to see, which value corresponds to what
        field type. The second entry specifies the default value of the field,
        i.e. the value that is initially displayed in the field. If None,
        no value is displayed by default.

        Args:
            device (Device): Device object containing information about the
                currently installed superconducting device.

        Returns:
            Dictionary containing the configuration options available in
                the QuantumExperimentGUI.
        """
        two_qb_gates = device.two_qb_gates()
        return {
            'kwargs': odict({
                QuantumExperiment.__name__: {
                    # kwarg: (fieldtype, default_value),
                    'label': (str, None),
                    'sweep_points': (SweepPoints, None),
                    'upload': (bool, True),
                    'measure': (bool, True),
                    'analyze': (bool, True),
                    'delegate_plotting': (bool, False),
                    'compression_seg_lim': (int, None),
                    'cz_pulse_name': (
                        set(two_qb_gates),
                        two_qb_gates[0] if len(two_qb_gates) else None)
                },
            }),
            'task_list_fields': odict({}),
            'sweeping_parameters': odict({}),
        }


class NDimQuantumExperiment():
    """
    Wrapper class to allow running an N-dimensional QuantumExperiment.

    Instantiates M times self.QuantumExperiment, where M is the product of
    lengths of self.sweep_points[2:], each initialised with
    sweep_points=self.sweep_points[0:2].
    Each of the M experiments is given a single value of each sweep points in
    self.sweep_points[2:] (passed in self.sweep_points[self.DUMMY_DIM]),
    such that sweeping over all M experiments is then equivalent to sweeping
    over self.sweep_points[2:].
    self.experiments are indexed by self.get_experiment_indices(),
    such that one can reconstruct an N-dim dataset afterwards from the
    separate 2-dim datasets of each experiment.

    Example:
        Initial sweep points passed to this class:
        pp=[1,2,3]
        dv=[1,2]
        sweep_points=sp_mod.SweepPoints([
            {'freq': (..., 'Hz', 'Readout frequency')},
            {'pump_freq': (..., 'Hz', 'TWPA pump frequency')},
            {'pump_power': (pp, 'dBm', 'TWPA pump power')},
            {'dc_voltage': (dv, 'V', 'TWPA DC voltage')},
        ])
        Resulting experiments (psueudo-code):
        for j in range(2):
            for i in range(3):
                idxs = (i,j)
                sweep_points=sp_mod.SweepPoints([
                    {'freq': (..., 'Hz', 'Readout frequency')},
                    {  # dimension number self.DUMMY_DIM
                        'pump_freq': (..., 'Hz', 'TWPA pump frequency'),
                        'pump_power': ([pp[i]]*3, 'dBm', 'TWPA pump power'),
                        'dc_voltage': ([dv[j]]*2, 'V', 'TWPA DC voltage'),
                    },
                ])

    Args:
        sweep_points (SweepPoints): SweepPoints to be used by the experiment
        clear_experiments (bool): whether to keep or clear previously
            instantiated experiments from memory during runtime
        QuantumExperiment (class): experiment class to be instantiated and
        run several times as a sub-experiment
        DUMMY_DIM: dimension of self.sweep_points to which the (non-swept)
            values of sweep_points from dimensions >= 2 should be collapsed
    """

    QuantumExperiment = QuantumExperiment
    DUMMY_DIM = 1

    def __init__(self, *args, sweep_points=None, clear_experiments=False,
                 QuantumExperiment=None, **kwargs):
        self.experiments = {}
        if QuantumExperiment is not None:
            self.QuantumExperiment = QuantumExperiment
        self.clear_experiments = clear_experiments
        self.sweep_points = SweepPoints(sweep_points)
        self._generate_sweep_lengths()
        self.args = args
        self.kwargs = kwargs

        for idxs in self.get_experiment_indices():
            self.create_experiment(idxs)

        if kwargs.get('measure') and kwargs.get('analyze', True):
            self.run_ndim_analysis()

    def _generate_sweep_lengths(self, sweep_points=None):
        """
        Extracts the length of sweep_points (useful for
        NDimMultiTaskingExperiment). Note: this could be made unnecessary by
        simply checking that sweep_points lengths are consistent over tasks.
        """

        sp = self.sweep_points if sweep_points is None else sweep_points
        self.sweep_lengths = sp.length()

    def get_experiment_indices(self):
        """
        Returns:
            idxs: list of indices indexing self.experiments, where each
                index is a tuple of int indicating the position of
                the experiment within self.sweep_points[2:]. E.g., as in the
                docstring of the init, for
                self.sweep_points.length() = [a, b, 3, 2],
                idxs = [(0, 0),(1, 0),(2, 0),(0, 1),(1, 1),(2, 1)]
        """
        idxs = itertools.product(
            *[range(i) for i in self.sweep_lengths[:1:-1]])
        return [idx[::-1] for idx in idxs]

    def make_2d_sweep_points(self, sweep_points, idxs):
        """
        Extracts 2-dim SweepPoints for an experiment indexed by idxs,
        by slicing the dimensions >=2 of sweep_points at indices idxs.

        Args:
            sweep_points: N-dim sweep points to slice down to 2 dimensions.
            idxs: indices of the experiments to instantiate,
            see self.get_experiment_indices
        """
        # Extract sweep_point dimensions beyond the second dimension
        extra_sp = SweepPoints(sweep_points[2:])
        # Extract first two dimensions of N dimensional sweep points
        current_sp = SweepPoints(sweep_points[:2])
        length = self.sweep_lengths[self.DUMMY_DIM]
        for dim, idx in enumerate(idxs):
            for k in extra_sp.get_sweep_dimension(dim, default={}):
                current_sp.add_sweep_parameter(
                    k, [extra_sp[k][idx] for i in range(length)],
                    extra_sp.get_sweep_params_property('unit', param_names=k),
                    extra_sp.get_sweep_params_property('label', param_names=k),
                    dimension=self.DUMMY_DIM,
                )
        return current_sp

    def create_experiment(self, idxs):
        """
        Instantiates sub-experiments.

        Args:
            idxs: indices of the experiments to instantiate,
                see self.get_experiment_indices
        """
        current_sp = self.make_2d_sweep_points(self.sweep_points, idxs)
        exp_metadata = dict(
            ndim_sweep_points=self.sweep_points,
            ndim_current_idxs=idxs,
        )
        self.experiments[idxs] = self.QuantumExperiment(
            *self.args, sweep_points=current_sp, exp_metadata=exp_metadata,
            **self.kwargs)
        if self.clear_experiments:
            del self.experiments[idxs]

    def run_measurement(self, **kw):
        for qe in self.experiments.values():
            qe.run_measurement(**kw)

    def run_analysis(self, **kw):
        for qe in self.experiments.values():
            qe.run_analysis(**kw)
        self.run_ndim_analysis()

    def run_ndim_analysis(self):
        """
        Instantiates and runs an analysis class on the whole N-dimensional
        dataset
        """
        pass


class NDimMultiTaskingExperiment(NDimQuantumExperiment):
    """
    Extension of NDimQuantumExperiment capable of running MultiTaskingExperiment

    Similar class to NDimQuantumExperiment, but self.QuantumExperiment should be
    a MultiTaskingExperiment.
    See NDimQuantumExperiment for details on the high-level logic, and see
    MultiTaskingExperiment for details on the measurement.

    Args:
        task_list: see MultiTaskingExperiment

    FIXME should a MultiTasking-like experiment be listed in this module?
    """

    def __init__(self, task_list, *args, sweep_points=None,
                 QuantumExperiment=None, **kwargs):
        self.task_list = list(task_list)
        super().__init__(*args, sweep_points=sweep_points,
                         QuantumExperiment=QuantumExperiment, **kwargs)


    def _generate_sweep_lengths(self, sweep_points=None):
        sp = self.sweep_points if sweep_points is None else sweep_points
        sp = deepcopy(sp)
        for task in self.task_list:
            sp.update(SweepPoints(task.get('sweep_points', [])))
        super()._generate_sweep_lengths(sp)


    def create_experiment(self, idxs):
        current_sp = self.make_2d_sweep_points(self.sweep_points, idxs)
        current_tl = [copy(t) for t in self.task_list]
        for task in current_tl:
            # Just to ease recovering the full sp in the analysis
            task['ndim_sweep_points'] = task['sweep_points']
            task['ndim_current_idxs'] = idxs
            # Make sp that will actually be used in the experiment
            task['sweep_points'] = self.make_2d_sweep_points(
                task.get('sweep_points', []), idxs)
        self.experiments[idxs] = self.QuantumExperiment(
            *self.args, sweep_points=current_sp,
            task_list=current_tl, **self.kwargs)
        if self.clear_experiments:
            del self.experiments[idxs]

