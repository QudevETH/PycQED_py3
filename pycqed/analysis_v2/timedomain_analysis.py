import lmfit
import numpy as np
from numpy.linalg import inv
import scipy as sp
import itertools
import matplotlib as mpl
import cmath
from collections import OrderedDict, defaultdict
import re

from pycqed.utilities import timer as tm_mod
from sklearn.mixture import GaussianMixture as GM
from sklearn.tree import DecisionTreeClassifier as DTC

from pycqed.analysis import fitting_models as fit_mods
from pycqed.analysis import analysis_toolbox as a_tools
import pycqed.analysis_v2.base_analysis as ba
import pycqed.analysis_v2.readout_analysis as roa
from pycqed.analysis_v2.readout_analysis import \
    Singleshot_Readout_Analysis_Qutrit as SSROQutrit
import pycqed.analysis_v2.tomography_qudev as tomo
from pycqed.analysis.tools.plotting import SI_val_to_msg_str
from copy import deepcopy
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration.calibration_points import CalibrationPoints
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cols
from pycqed.analysis.three_state_rotation import predict_proba_avg_ro
import pycqed.analysis_v3.helper_functions as hlp_mod
import logging

from pycqed.utilities import math
from pycqed.utilities.general import find_symmetry_index
from pycqed.utilities.state_and_transition_translation import STATE_ORDER
import pycqed.measurement.waveform_control.segment as seg_mod
import datetime as dt
log = logging.getLogger(__name__)
try:
    import qutip as qtp
except ImportError as e:
    log.warning('Could not import qutip, tomography code will not work')


# Mixin classes
class ArtificialDetuningMixin():
    """
    Extract the information about the artificial_detuning and
    create the artificial_detuning_dict for each qubit.
    """

    def get_artificial_detuning_dict(self, raise_error=True):
        """
        - first checks whether artificial_detuning_dict was passed in
            options_dict, metadata, or default_options
        - then tries to create it based on the information in
            preprocessed_task_list
        - lastly, it falls back to the legacy version: searching for
            artificial_detuning in options_dict, metadata, or default_options

        Args:
            raise_error (bool; default: True): whether to raise ValueError
                if no information about artificial detuning is found

        Returns:
            artificial_detuning_dict: dict with qb names as keys and
                value for the artificial detuning as values
        """
        artificial_detuning_dict = self.get_param_value(
            'artificial_detuning_dict')
        if artificial_detuning_dict is None:
            artificial_detuning = self.get_param_value('artificial_detuning')
            if 'preprocessed_task_list' in self.metadata:
                pptl = self.metadata['preprocessed_task_list']
                artificial_detuning_dict = OrderedDict([
                    (t['qb'], t['artificial_detuning']) for t in pptl
                ])
            elif artificial_detuning is not None:
                # legacy case
                if isinstance(artificial_detuning, dict):
                    artificial_detuning_dict = artificial_detuning
                else:
                    artificial_detuning_dict = OrderedDict(
                        [(qbn, artificial_detuning) for qbn in self.qb_names])
        if raise_error and artificial_detuning_dict is None:
            raise ValueError('"artificial_detuning" not found.')
        return artificial_detuning_dict


class PhaseErrorsAnalysisMixin():
    """Mixin containing utility functions needed by the QScaleAnalysis and
    NPulsePhaseErrorCalibAnalysis classes.

    Classes deriving from this mixin must have the following attributes
        - sp
        - qb_names
        - raw_data_dict
        - proc_data_dict
    and the following methods:
        - get_transition_name
        - get_data_from_timestamp_list
    """

    @property
    def pulse_par_name(self):
        """
        Pulse parameter that was swept (either motzoi or env_mod_frequency).
        """
        spars0 = self.sp.get_parameters(0)
        if any(['motzoi' in e for e in spars0]):
            # for the standard QScale measurement
            return 'motzoi'
        elif any(['env_mod_frequency' in e for e in spars0]):
            # for the FrequencyDetuningCalib and NPulsePhaseErrorCalibAnalysis
            # measurements
            return 'env_mod_freq'
        else:
            raise ValueError('QScale pulse parameter not recognised. Accepted '
                             'parameters are "motzoi" and "env_mod_frequency."')

    def _extract_current_pulse_par_value(self):
        """
        Takes the value of self.pulse_par_name for each qubit from the HDF file
        and stores the values in self.raw_data_dict.
        """
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_{self.pulse_par_name}_'+qbn] = \
                s+f'.{trans_name}_{self.pulse_par_name}'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def create_textstr(self, qb_name, fit_val, fit_stderr, break_lines=False):
        """
        Creates the text string that will show up on the plot.

        Args:
            qb_name (str): name of the qubit
            fit_val (float): fitted value of self.pulse_par_name
            fit_stderr (float): standard error from fit of self.pulse_par_name
            break_lines (bool): wither to introduce line breaks before the
                 fit_val and fit_stderr in the text string.

        Returns:
            text string
        """
        chr = '\n' if break_lines else ' '
        scaling_factor = 1 if self.pulse_par_name == 'motzoi' \
            else 1e-6
        trans_name = self.get_transition_name(qb_name)
        old_pulse_par_val = self.raw_data_dict[
            f'{trans_name}_{self.pulse_par_name}_' + qb_name]
        # FIXME: the following condition is always False, isn't it?
        if old_pulse_par_val != old_pulse_par_val:
            old_pulse_par_val = 0  # FIXME: explain why
        old_pulse_par_val *= scaling_factor
        fit_val *= scaling_factor
        fit_stderr *= scaling_factor

        if self.pulse_par_name == 'motzoi':
            return f'Qscale ={chr}{fit_val:.4f} $\pm$ {fit_stderr:.4f}' + \
                   f'\nold Qscale ={chr}{old_pulse_par_val:.4f}'
        else:
            return f'Envelope mod. freq. ={chr}' \
                   f'{fit_val:.4f} MHz $\pm$ {fit_stderr:.4f} MHz' + \
                   f'\nold envelope mod. freq. ={chr}{old_pulse_par_val:.4f} MHz'


# Analysis classes
class AveragedTimedomainAnalysis(ba.BaseDataAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_timestamp = True
        self.params_dict = {
            'value_names': 'value_names',
            'measured_values': 'measured_values',
            'measurementstring': 'measurementstring',
            'exp_metadata': 'exp_metadata'}
        self.numeric_params = []
        if kwargs.get('auto', True):
            self.run_analysis()

    def process_data(self):
        self.metadata = self.raw_data_dict.get('exp_metadata', {})
        if self.metadata is None:
            self.metadata = {}
        cal_points = self.metadata.get('cal_points', None)
        cal_points = self.get_param_value('cal_points', cal_points)
        cal_points_list = roa.convert_channel_names_to_index(
            cal_points, len(self.raw_data_dict['measured_values'][0]),
            self.raw_data_dict['value_names'])
        self.proc_data_dict['cal_points_list'] = cal_points_list
        measured_values = self.raw_data_dict['measured_values']
        cal_idxs = self._find_calibration_indices()
        scales = [np.std(x[cal_idxs]) for x in measured_values]
        observable_vectors = np.zeros((len(cal_points_list),
                                       len(measured_values)))
        observable_vector_stds = np.ones_like(observable_vectors)
        for i, observable in enumerate(cal_points_list):
            for ch_idx, seg_idxs in enumerate(observable):
                x = measured_values[ch_idx][seg_idxs] / scales[ch_idx]
                if len(x) > 0:
                    observable_vectors[i][ch_idx] = np.mean(x)
                if len(x) > 1:
                    observable_vector_stds[i][ch_idx] = np.std(x)
        Omtx = (observable_vectors[1:] - observable_vectors[0]).T
        d0 = observable_vectors[0]
        corr_values = np.zeros(
            (len(cal_points_list) - 1, len(measured_values[0])))
        for i in range(len(measured_values[0])):
            d = np.array([x[i] / scale for x, scale in zip(measured_values,
                                                           scales)])
            corr_values[:, i] = inv(Omtx.T.dot(Omtx)).dot(Omtx.T).dot(d - d0)
        self.proc_data_dict['corr_values'] = corr_values

    def measurement_operators_and_results(self):
        """
        Converts the calibration points to measurement operators. Assumes that
        the calibration points are ordered the same as the basis states for
        the tomography calculation (e.g. for two qubits |gg>, |ge>, |eg>, |ee>).
        Also assumes that each calibration in the passed cal_points uses
        different segments.

        Returns:
            A tuple of
                the measured values with outthe calibration points;
                the measurement operators corresponding to each channel;
                and the expected covariation matrix between the operators.
        """
        d = len(self.proc_data_dict['cal_points_list'])
        cal_point_idxs = [set() for _ in range(d)]
        for i, idxs_lists in enumerate(self.proc_data_dict['cal_points_list']):
            for idxs in idxs_lists:
                cal_point_idxs[i].update(idxs)
        cal_point_idxs = [sorted(list(idxs)) for idxs in cal_point_idxs]
        cal_point_idxs = np.array(cal_point_idxs)
        raw_data = self.raw_data_dict['measured_values']
        means = [None] * d
        residuals = [list() for _ in raw_data]
        for i, cal_point_idx in enumerate(cal_point_idxs):
            means[i] = [np.mean(ch_data[cal_point_idx]) for ch_data in raw_data]
            for j, ch_residuals in enumerate(residuals):
                ch_residuals += list(raw_data[j][cal_point_idx] - means[i][j])
        means = np.array(means)
        residuals = np.array(residuals)
        Fs = [np.diag(ms) for ms in means.T]
        Omega = residuals.dot(residuals.T) / len(residuals.T)
        data_idxs = np.setdiff1d(np.arange(len(raw_data[0])),
                                 cal_point_idxs.flatten())
        data = np.array([ch_data[data_idxs] for ch_data in raw_data])
        return data, Fs, Omega

    def _find_calibration_indices(self):
        cal_indices = set()
        cal_points = self.get_param_value('cal_points')
        nr_segments = self.raw_data_dict['measured_values'].shape[-1]
        for observable in cal_points:
            if isinstance(observable, (list, np.ndarray)):
                for idxs in observable:
                    cal_indices.update({idx % nr_segments for idx in idxs})
            else:  # assume dictionaries
                for idxs in observable.values():
                    cal_indices.update({idx % nr_segments for idx in idxs})
        return list(cal_indices)


class MultiQubit_TimeDomain_Analysis(ba.BaseDataAnalysis):
    """
    Base class for multi-qubit time-domain analyses.

    Parameters that can be specified in the options_dict:
     - rotation_type: type of rotation to be done on the raw data.
       Types of rotations supported by this class:
        - 'cal_states' (default, no need to specify): rotation based on
            CalibrationPoints for 1D and TwoD data. Supports 2 and 3 cal states
            per qubit
        - 'fixed_cal_points' (only for TwoD, with 2 cal states):
            does PCA on the columns corresponding to the highest cal state
            to find the indices of that cal state in the columns, then uses
            those to get the data points for the other cal state. Does
            rotation using the mean of the data points corresponding to the
            two cal states as the zero and one coordinates to rotate
            the data.
        - 'PCA': ignores cal points and does pca; in the case of TwoD data it
            does PCA row by row
        - 'column_PCA': ignores cal points and does pca; in the case of TwoD
            data it does PCA column by column
        - 'global_PCA' (only for TwoD): does PCA on the whole 2D array
     - main_sp (default: None): dict with keys qb_name used to specify which
        sweep parameter should be used as axis label in plot
     - functionality to split measurements with tiled sweep_points:
         - split_params (default: None): list of strings with sweep parameters
            names expected to be found in SweepPoints. Groups data by these
            parameters and stores it in proc_data_dict['split_data_dict'].
         - select_split (default: None): dict with keys qb_names and values
            a tuple (sweep_param_name, value) or (sweep_param_name, index).
            Stored in self.measurement_strings which specify the plot title.
            The selected parameter must also be part of the split_params for
            that qubit.
     - functionality to plot 1d slices from a 2D plot: enabled by passing the
        following parameters:
         - slice_idxs_1d_raw_plot (dict; default: None): slices indices of the
            raw data. If None, no slices are plotted.
         - slice_idxs_1d_proj_plot (dict; default: None): slices indices of the
            projected data. If None, no slices are plotted.
         The two dicts above are of the form {qb_name: [(idxs, axis)]}, where
            - axis (str) can be either 'row' or 'col', specifying whether idxs
                are row or column indices
            - idxs can be an int (data index) or a str of the form
                'idx_start:idx_end' interpreted as standard list/array indexing
                arr[idx_start:idx_end]
            Example: {'qb14': [('8:13', 'row'), (0, 'col')]}.
        Note:
            - to plot only 1D slices of 2D data, the standard plotting of raw
            and projected data can be disabled via the flags `plot_raw_data` and
            `plot_proj_data.`
            - to plot 1D slices of 2D data using an instance of this class that
            has already run self.process_data(), use the methods
            'plot_raw_1d_slices' and 'plot_projected_1d_slices'

    If an instance of SweepPoints (or its repr) is provided, then the
    corresponding meas_obj_sweep_points_map must also be specified in
    options_dict.

    To analyse data obtained with classifier detector, pass rotate = False
    in options_dict.
    """
    def __init__(self,
                 qb_names: list=None, label: str='',
                 t_start: str=None, t_stop: str=None, data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True,
                 params_dict=None, numeric_params=None, **kwargs):

        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting, **kwargs)

        self.qb_names = qb_names
        self.params_dict = params_dict
        if self.params_dict is None:
            self.params_dict = {}
        self.numeric_params = numeric_params
        self.measurement_strings = {}
        if self.numeric_params is None:
            self.numeric_params = []

        if self.job is None:
            self.create_job(qb_names=qb_names, t_start=t_start, t_stop=t_stop,
                            label=label, data_file_path=data_file_path,
                            do_fitting=do_fitting, options_dict=options_dict,
                            extract_only=extract_only, params_dict=params_dict,
                            numeric_params=numeric_params, **kwargs)
        if auto:
            self.run_analysis()

    def extract_data(self):
        """
        Extracts data and other relevant metadata.

        Creates the following attributes:
            - self.data_to_fit: {qbn: proj_data_name}
                ! NOTE: This attribute will be modified later in
                rotate_and_project_data !
            - self.measurement_strings: {qbn: measurement_string}
            - self.prep_params: preparation parameters dict
            - self.rotate: bool; whether to do data rotation and projection.
                Must be False if raw data is already classified.
            - self.predict_proba: bool (only used for SSRO data);
                whether to classify shots

        Calls the following methods (see docstrings there):
            - self.get_sweep_points()
            - self.get_cal_points()
            - self.get_rotation_type()
            - self._get_default_data_to_fit()
            - self.update_data_to_fit()
            - self.get_data_filter()
            - self.create_meas_results_per_qb()
            - self.create_sweep_points_dict()
            - self.get_num_cal_points()
            - self.update_sweep_points_dict()
            - (if TwoD) self.create_sweep_points_2D_dict()
        These methods must be called in the order given above because they
        depend on attributes created in the calls preceding them.
        """
        super().extract_data()

        if self.qb_names is None:
            self.qb_names = self.get_param_value(
                'ro_qubits', default_value=self.get_param_value('qb_names'))
            if self.qb_names is None:
                raise ValueError('Provide the "qb_names."')
        self.measurement_strings = {
            qbn: self.raw_data_dict['measurementstring'] for qbn in
            self.qb_names}

        self.prep_params = self.get_param_value('preparation_params',
                                                default_value=dict())

        # creates self.channel_map
        self.get_channel_map()

        # whether to rotate data or not
        self.rotate = self.get_param_value('rotate', default_value=True)

        # to be used if data_type == singleshot (see process_data)
        self.predict_proba = self.get_param_value("predict_proba", False)
        if self.predict_proba and self.get_param_value("classified_ro", False):
            log.warning("predict_proba set to 'False' as probabilities are "
                        "already obtained from classified readout")
            self.predict_proba = False
        # ensure rotation is removed when single shots yield probabilities
        if self.get_param_value("classified_ro", False) or self.predict_proba:
            self.rotate = False

        # creates self.sp
        self.get_sweep_points()

        # creates self.cp if found else defaults to legacy extraction
        self.get_cal_points()

        # creates self.rotation_type
        self.get_rotation_type()

        # create self.data_to_fit
        # here we don't create the attribute directly inside the method because
        # we want to retain access to the data_to_fit returned by this method
        self.data_to_fit = self._get_default_data_to_fit()
        # update self.data_to_fit based on the rotation_type
        self.update_data_to_fit()

        # creates self.data_filter and self.data_with_filter
        self.get_data_filter()

        # These calls need to stay here because otherwise QScaleAnalysis will
        # fail: in process_data, this child needs the correctly created
        # sweep_points_dict.
        self.create_meas_results_per_qb()
        self.create_sweep_points_dict()
        self.get_num_cal_points()
        self.update_sweep_points_dict()
        if self.get_param_value('TwoD', False):
            self.create_sweep_points_2D_dict()

    def get_channel_map(self):
        """
        Creates the channel map as {qbn: [ro channels (usually value_names]}.
        This is the same as the meas_obj_value_names_map, so this function first
        tries to extract this parameter, which can thus be passed by the user in
        the options_dict or metadata under that name.

        Creates self.channel_map

        """
        self.channel_map = self.get_param_value('meas_obj_value_names_map')
        if self.channel_map is None:
            # if the new name meas_obj_value_names_map is not found, try with
            # the old name channel_map
            self.channel_map = self.get_param_value('channel_map')
            if self.channel_map is None:
                # if channel_map also not found, construct channel map from
                # value names
                value_names = self.raw_data_dict['value_names']
                if np.ndim(value_names) > 0:
                    value_names = value_names
                if 'w' in value_names[0]:
                    # FIXME: avoid dependency ana_v2 - ana_v3
                    self.channel_map = hlp_mod.get_qb_channel_map_from_file(
                        self.qb_names, value_names=value_names,
                        file_path=self.raw_data_dict['folder'])
                else:
                    self.channel_map = {}
                    for qbn in self.qb_names:
                        self.channel_map[qbn] = value_names

        if len(self.channel_map) == 0:
            raise ValueError('No qubit RO channels have been found.')

    def get_sweep_points(self):
        """
        Create the SweepPoints class instance if sweep points or their repr
        were passed to the analysis.

        Creates self.sp
        """
        self.sp = self.get_param_value('sweep_points')
        if self.sp is not None:
            self.sp = SweepPoints(self.sp)

    def get_cal_points(self):
        """
        Extracts information about calibration points.

        First checks if an instance of Calibration points (or its repr) was
        passed and extract information from there.
        If this fails (no such instance was passed), it falls back to the legacy
        way of passing calibration points information: by directly specifying
        cal_states_rotations and cal_states_dict.

        Creates the following attributes:
            - self.cp: CalibrationPoints instance
            - self.cal_states_dict: dict of the form
                {qbn: {
                    'cal_state': [data array indices for this state]
                    }
                }
                Ex: {'qb13': {'e': [-1, -2], 'g': [-3, -4]},
                      'qb5': {'e': [-1, -2], 'g': [-3, -4]}}

                Note: this dict has an entry for each qubit in order to adhere
                to the general structure of this class, but the inner dicts must
                contain all the cal states specified in CalibrationPoints for
                each qubit.
            - self.cal_states_rotations: dict of the form
                {qbn: {
                    'cal_state': int specifying the transmon state
                        level (0, 1, ...)
                    }
                }
                Ex.: {'qb12': {'g': 0, 'e': 1, 'f': 2},
                      'qb13': {'g': 0, 'e': 1}}
                This attribute tells the analysis what kind of rotation to do
                for each qubit. In the above example, qb12 data will undergo
                a 3-state rotation, while qb13 data will be projected along
                the line between g and e (for this qubit, the f-state cal point
                is ignored).
        """
        # Looks for instance of CalibrationPoints (or its repr)
        cal_points = self.get_param_value('cal_points')
        last_ge_pulses = self.get_param_value('last_ge_pulses',
                                              default_value=False)
        if hasattr(last_ge_pulses, '__iter__') and \
                len(last_ge_pulses) != len(self.qb_names):
            last_ge_pulses = len(self.qb_names) * list(last_ge_pulses)

        # default value for cal_states_rotations
        cal_states_rotations = {qbn: {} for qbn in self.qb_names}
        try:
            # try to take cal points information from CalibrationPoints instance
            # or its repr
            self.cp = CalibrationPoints.from_string(cal_points)
            # get cal_states_dict from self.cp
            # The cal point indices in cal_points_dict are used in MQTDA for
            # plots only on data for which any preparation readout (e.g. active
            # reset or preselection) has already been removed. Therefore the
            # indices should only consider filtered data
            self.cal_states_dict = self.cp.get_indices(self.qb_names)
            # if self.rotate (do rotation and projection, i.e. the raw data
            # is not classified), get cal_states_rots from self.cp. If rotation
            # should not be done, take the default value defined above
            cal_states_rots = self.cp.get_rotations(
                last_ge_pulses, self.qb_names) if \
                self.rotate else cal_states_rotations
            # get cal_states_rotations from options_dict or metadata and
            # default to cal_states_rots above if not found
            self.cal_states_rotations = self.get_param_value(
                'cal_states_rotations', default_value=cal_states_rots)
        except TypeError as e:
            # Handles measurements that do not use CalibrationPoints.
            # Look for cal_states_dict and cal_states_rotations in metadata
            # or options_dict
            if cal_points is not None:
                # This means cal_points were provided by something went wrong
                # with their extraction
                log.error(e)
                log.warning("Failed retrieving cal point objects or states. "
                            "Please update measurement to provide cal point object "
                            "in metadata. Trying to get them using the old way ...")
            # Get cal_states_rotations from options_dict or metadata and
            # default to cal_states_rotations above if not found.
            # Also set to the default cal_states_rotations defined above if
            # rotation and projection should not be done
            # (self.rotation == False, i.e. raw data is already classified).
            self.cal_states_rotations = self.get_param_value(
                'cal_states_rotations', default_value=cal_states_rotations) \
                if self.rotate else cal_states_rotations
            # get cal_states_dict from options_dict or metadata
            self.cal_states_dict = self.get_param_value('cal_states_dict')

        if self.cal_states_rotations is None:
            # this variable may have been set to None by mistake in the metadata
            # or by the user
            log.warning('cal_states_rotations cannot be None. Setting it to '
                        '{qbn: {} for qbn in self.qb_names}.')
            self.cal_states_rotations = cal_states_rotations

        if self.cal_states_dict is None or not len(self.cal_states_dict):
            # no cal points information: do principle component analysis
            self.cal_states_dict = {qbn: {} for qbn in self.qb_names}

    def get_rotation_type(self):
        """
        Extracts the rotation_type parameter from the options_dict or metadata
        and updates it if necessary (no cal points information provided but
        rotation type indicates these are needed).

        Creates the following attribute:
            - self.rotation_type: string or dict with qubit names as keys and
            strings indicating rotation type as values. See class docstring
            for the recognized rotation types.
        """
        if self.get_param_value('global_PCA') is not None:
            log.warning('Parameter "global_PCA" is deprecated. Please set '
                        'rotation_type="global_PCA" instead.')
        # Get the rotation_type. See class docstring
        # Without deepcopy, the value in the options_dict/metadata would be
        # modified as well.
        self.rotation_type = deepcopy(self.get_param_value(
            'rotation_type',
            default_value='cal_states' if self.rotate else 'no_rotation'))

        if isinstance(self.rotation_type, str):
            self.rotation_type = {qbn: self.rotation_type
                                  for qbn in self.qb_names}
        for qbn in self.qb_names:
            if self.rotation_type[qbn] != 'no_rotation' and \
                    len(self.cal_states_dict[qbn]) == 0:
                if 'pca' not in self.rotation_type[qbn].lower():
                    # If no cal states information was provided
                    # (i.e. self.cal_states_dict = {qbn: {} for qbn in
                    # self.qb_names}, see self.get_cal_points()), the analysis
                    # class will do pca (the only options to project the raw
                    # data), so here we update self.rotation_type to reflect
                    # this if not already set to some form of PCA.
                    orig_rot_type = self.rotation_type[qbn]
                    self.rotation_type[qbn] = 'global_PCA' if \
                        self.get_param_value('TwoD', default_value=False) \
                        else 'PCA'
                    log.warning(f'rotation_type is set to {orig_rot_type} but '
                                f'no calibration points information was found. '
                                f'Setting rotation_type for {qbn} to '
                                f'{self.rotation_type[qbn]}.')

    def _get_default_data_to_fit(self):
        """
        Extracts the data_to_fit parameter from the options_dict or metadata
        and assigns it a reasonable default value based on the cal points
        information.

        Returns:
            - data_to_fit: dict of the form {qbn: proj_data_name} where
            proj_data_name is a string corresponding to a key in
            self.proc_data_dict['projected_data_dict'] indicating which
            projected data the children should fit.

        ! If proj_data_name is a list or tuple of string, this method takes the
        first entry only !
        These lists are currently only supported by MultiCZgate_CalibAnalysis
        and allowing them in this class would break all children.
        It remains as a to do to upgrade this module in that
        direction (13.08.2021).

        """
        # Without deepcopy, the value in the options_dict/metadata would be
        # modified as well. This would break MultiCZgate_Calib_Analysis which
        # compares what this class sets for self.data_to_fit and what was given
        # in options_dict/metadata/default_options
        data_to_fit = deepcopy(self.get_param_value('data_to_fit'))
        if data_to_fit is None:
            # It could happen that it was passed as None or was not specified in the
            # metadata. In this case, it makes sense to still check the
            # default option because data_to_fit must be specified.
            # Note that passing an empty dict as data_to_fit will keep
            # data_to_fit empty as expected.
            data_to_fit = deepcopy(self.default_options.get('data_to_fit'))
            if data_to_fit is None:
                # If we have cal points, but data_to_fit is not specified,
                # choose a reasonable default value.
                data_to_fit = {}
                for qbn in self.qb_names:
                    if not len(self.cal_states_dict[qbn]) or \
                            'pca' in self.rotation_type[qbn].lower():
                        # cal states do not exist or the rotation type is some
                        # kind of pca: assign rotation type
                        # to the data to fit such that the rotation type appears
                        # in the plot title and figure names. This allows for
                        # plots with unique names if the same data is analysed
                        # with different rotation types.
                        data_to_fit[qbn] = self.rotation_type[qbn]
                    else:
                        # cal states exist or rotation_type is cal_states but
                        # cal states weren't specified:
                        # the highest transmon state will be fitted
                        csr = [(k, v) for k, v in
                               self.cal_states_rotations[qbn].items()]
                        # sort by transmon state (lowest to highest)
                        csr.sort(key=lambda t: t[1])
                        # take letter of the highest transmon state
                        data_to_fit[qbn] = f'p{csr[-1][0]}'

        # make sure no extra qubit names exist in data_to_fit compared to
        # self.qb_names (can happen if user passes qb_names)
        qbns = list(data_to_fit)
        for qbn in qbns:
            if qbn not in self.qb_names:
                del data_to_fit[qbn]

        # This is a hack to allow list inside data_to_fit.
        # A nicer solution is needed at some point, but for now this feature is
        # only needed by the MultiCZgate_Calib_Analysis to allow the same data
        # to be fitted in two ways (to extract SWAP errors).
        for qbn in data_to_fit:
            if isinstance(data_to_fit[qbn], (list, tuple)):
                data_to_fit[qbn] = data_to_fit[qbn][0]

        return data_to_fit

    def update_data_to_fit(self):
        """
        Updates self.data_to_fit based on the rotation_type.
        """
        for qbn in self.data_to_fit:
            if self.get_param_value('TwoD', default_value=False):
                if self.rotation_type[qbn].lower() == 'global_pca':
                    self.data_to_fit[qbn] = self.rotation_type[qbn]
                elif self.rotation_type[qbn].lower() == 'fixed_cal_points':
                    self.data_to_fit[qbn] += '_fixed_cp'
                else:
                    if 'pca' in self.rotation_type[qbn].lower():
                        self.data_to_fit[qbn] = self.rotation_type[qbn]
            else:
                if 'pca' in self.rotation_type[qbn].lower():
                    self.data_to_fit[qbn] = self.rotation_type[qbn]

    def get_data_filter(self):
        """
        Extracts the data_filter parameter from the options_dict or metadata
        and assigns it a reasonable default value based on the information in
        self.prep_params.

        Creates the following attribute:
            - self.data_filter: function that will be used in
                create_meas_results_per_qb to filter/process the data
            - self.data_with_filter: bool that is checked in
                prepare_raw_data_plots
        """
        # flag to be used in prepare_raw_data_plots
        self.data_with_filter = False
        self.data_filter = self.get_param_value('data_filter')
        if self.data_filter is None:
            if 'active' in self.prep_params.get('preparation_type', 'wait'):
                reset_reps = self.prep_params.get('reset_reps', 3)
                self.data_filter = lambda x: x[reset_reps::reset_reps+1]
                self.data_with_filter = True
            elif "preselection" in self.prep_params.get('preparation_type',
                                                        'wait'):
                self.data_filter = lambda x: x[1::2]  # filter preselection RO
                self.data_with_filter = True
            else:
                self.data_filter = lambda x: x
        else:
            self.data_with_filter = True

    def create_sweep_points_dict(self):
        """
        Creates self.proc_data_dict['sweep_points_dict'][qbn]['sweep_points']
        containing the hard sweep points (1st sweep dimension).
        It can be created from the following given parameters (the priority in
        which these parameters are considered is the following):
            - 1D sweep points taken from a SweepPoints instance or its repr
            based on the meas_obj_sweep_points_map or main_sp (used with split
            data, see self.split_data).
            ! If SweepPoints are given then the meas_obj_sweep_points_map must
            also be specified !
            - sweep_points_dict of the form {qbn: swpts_1d_array}
            - swpts_1d_array from hard_sweep_params of the form
            {sweep_param_name: {'values': swpts_1d_array, 'unit': unit_str}}
            - self.raw_data_dict['hard_sweep_points'] created by the base class
                Only for this case, the self.data_filter is applied to the
                sweep_points. TODO: understand why! (Steph, 12.08.2021)

        Creates the following attributes if SweepPoints are given:
            - self.mospm (the meas_ibj_sweep_points_map)

        Possible keyword arguments taken from metadata or options_dict:
            - sp1d_filter: function to filter or process the sweep points. Can
                be specified either as a callable or a string. In the later
                case, the string will be evaluated
            - sweep_points_dict: dict of the form {qbn: swpts_1d_array}
            - hard_sweep_params: dict of the form
                {sweep_param_name: {'values': swpts_1d_array, 'unit': unit_str}}
            - meas_obj_sweep_points_map: dict of the form {qbn: sp_param_names}
                where sp_param_names is a list of the sweep parameters in
                SweepPoints corresponding to qbn
            - main_sp: dict of the form {qbn: sp_param_name} where sp_param_name
                if a sweep param name inside SweepPoints indicating which sweep
                points values to be taken.
        """
        sp1d_filter = self.get_param_value('sp1d_filter', lambda x: x)
        if isinstance(sp1d_filter, str):
            sp1d_filter = eval(sp1d_filter)
        sweep_points_dict = self.get_param_value('sweep_points_dict')
        hard_sweep_params = self.get_param_value('hard_sweep_params')
        if self.sp is not None:
            self.mospm = self.get_param_value('meas_obj_sweep_points_map')
            main_sp = self.get_param_value('main_sp')
            if self.mospm is None:
                raise ValueError('When providing "sweep_points", '
                                 '"meas_obj_sweep_points_map" has to be '
                                 'provided in addition.')
            self.proc_data_dict['sweep_points_dict'] = {}
            for qbn in self.qb_names:
                param_names = self.mospm[qbn]
                if main_sp is not None and (p := main_sp.get(qbn)):
                    dim = self.sp.find_parameter(p)
                    if dim is None:
                        log.warning(
                            f'Main sweep point {p} for {qbn} not found.')
                    elif dim == 1:
                        log.warning(f"main_sp is only implemented for sweep "
                                    f"dimension 0, but {p} is in dimension 1.")
                    else:
                        param_names = [p]
                sp_qb = self.sp.get_sweep_params_property(
                    'values', 0, param_names)[0]
                self.proc_data_dict['sweep_points_dict'][qbn] = \
                    {'sweep_points': sp1d_filter(sp_qb),
                     'param_names': param_names}
        elif sweep_points_dict is not None:
            # assumed to be of the form {qbn1: swpts_array1, qbn2: swpts_array2}
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': sp1d_filter(sweep_points_dict[qbn])}
                 for qbn in self.qb_names}
        elif hard_sweep_params is not None:
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': sp1d_filter(
                    list(hard_sweep_params.values())[0][
                    'values'])} for qbn in self.qb_names}
        else:
            hard_sp = self.raw_data_dict['hard_sweep_points']
            # The data filter is applied to the hard_sp in this case because
            # these correspond to the mc_points (i.e. range(len(sweep points))),
            # which have the same length as the data. So if the data is filtered
            # the mc_points must also be filtered for dimension matching.
            # We end up here, for example, when calling
            # MultiQutrit_Singleshot_Readout_Analysis with preselection.
            self.proc_data_dict['sweep_points_dict'] = \
                {qbn: {'sweep_points': sp1d_filter((self.data_filter(
                    hard_sp)))} for qbn in self.qb_names}

    def create_sweep_points_2D_dict(self):
        """
        Creates self.proc_data_dict['sweep_points_2D_dict'][qbn][2d_sp_par_name]
        containing the soft sweep points (2st sweep dimension).
        It can be created from the following given parameters (the priority in
        which these parameters are considered is the following):
            - 2D sweep points taken from a SweepPoints instance or its repr
            based on the meas_obj_sweep_points_map.
            ! If SweepPoints are given then the meas_obj_sweep_points_map must
            also be specified !
            - sweep_points_dict of the form {qbn: swpts_1d_array}
            - swpts_1d_array from hard_sweep_params of the form
            {sweep_param_name: {'values': swpts_1d_array, 'unit': unit_str}}
            - self.raw_data_dict['hard_sweep_points'] created by the base class

        Creates the following attributes if SweepPoints are given:
            - self.mospm (the meas_ibj_sweep_points_map)

        Possible keyword arguments taken from metadata or options_dict:
            - soft_sweep_params: dict of the form
                {sweep_param_name: {'values': swpts_2d_array, 'unit': unit_str}}
            - percentage_done: int between 0 and 100 indicating for what
                percentage of an interrupted measurement data was acquired and
                stored.
        """
        soft_sweep_params = self.get_param_value('soft_sweep_params')
        if self.sp is not None:
            self.proc_data_dict['sweep_points_2D_dict'] = OrderedDict()
            for qbn in self.qb_names:
                self.proc_data_dict['sweep_points_2D_dict'][qbn] = \
                    OrderedDict()
                for pn in self.mospm[qbn]:
                    if pn in self.sp[1]:
                        self.proc_data_dict['sweep_points_2D_dict'][qbn][
                            pn] = self.sp[1][pn][0]
        elif soft_sweep_params is not None:
            self.proc_data_dict['sweep_points_2D_dict'] = \
                {qbn: {pn: soft_sweep_params[pn]['values'] for
                       pn in soft_sweep_params}
                 for qbn in self.qb_names}
        else:
            if len(self.raw_data_dict['soft_sweep_points'].shape) == 1:
                self.proc_data_dict['sweep_points_2D_dict'] = \
                    {qbn: {self.raw_data_dict['sweep_parameter_names'][1]:
                               self.raw_data_dict['soft_sweep_points']} for
                     qbn in self.qb_names}
            else:
                sspn = self.raw_data_dict['sweep_parameter_names'][1:]
                self.proc_data_dict['sweep_points_2D_dict'] = \
                    {qbn: {sspn[i]: self.raw_data_dict['soft_sweep_points'][i]
                           for i in range(len(sspn))} for qbn in self.qb_names}

        if self.get_param_value('percentage_done', 100) < 100:
            # This indicated an interrupted measurement.
            # Remove non-measured sweep points in that case.
            # raw_data_dict['soft_sweep_points'] is obtained in
            # BaseDataAnalysis.add_measured_data(), and its length should
            # always correspond to the actual number of measured soft sweep
            # points.
            ssl = len(self.raw_data_dict['soft_sweep_points'])
            for sps in self.proc_data_dict['sweep_points_2D_dict'].values():
                for k, v in sps.items():
                    sps[k] = v[:ssl]

    def create_meas_results_per_qb(self):
        """
        Creates
         - self.proc_data_dict['meas_results_per_qb_raw']: dict of the form
            {qbn: {ro_channel: data}
         - self.proc_data_dict['meas_results_per_qb']: same as
            meas_results_per_qb_raw but with self.data_filter applied
        These are created from self.raw_data_dict['measured_data" created by
        the base class using the self.channel_map.
        """

        measured_RO_channels = list(self.raw_data_dict['measured_data'])
        meas_results_per_qb_raw = {}
        meas_results_per_qb = {}
        for qb_name, RO_channels in self.channel_map.items():
            meas_results_per_qb_raw[qb_name] = {}
            meas_results_per_qb[qb_name] = {}
            if isinstance(RO_channels, str):
                meas_ROs_per_qb = [RO_ch for RO_ch in measured_RO_channels
                                   if RO_channels in RO_ch]
                for meas_RO in meas_ROs_per_qb:
                    meas_results_per_qb_raw[qb_name][meas_RO] = \
                        self.raw_data_dict[
                            'measured_data'][meas_RO]
                    meas_results_per_qb[qb_name][meas_RO] = \
                        self.data_filter(
                            meas_results_per_qb_raw[qb_name][meas_RO])

            elif isinstance(RO_channels, list):
                for qb_RO_ch in RO_channels:
                    meas_ROs_per_qb = [RO_ch for RO_ch in measured_RO_channels
                                       if qb_RO_ch in RO_ch]

                    for meas_RO in meas_ROs_per_qb:
                        meas_results_per_qb_raw[qb_name][meas_RO] = \
                            self.raw_data_dict[
                                'measured_data'][meas_RO]
                        meas_results_per_qb[qb_name][meas_RO] = \
                            self.data_filter(
                                meas_results_per_qb_raw[qb_name][meas_RO])
            else:
                raise TypeError('The RO channels for {} must either be a list '
                                'or a string.'.format(qb_name))
        self.proc_data_dict['meas_results_per_qb_raw'] = \
            meas_results_per_qb_raw
        self.proc_data_dict['meas_results_per_qb'] = \
            meas_results_per_qb

    def process_data(self):
        """
        Handles the following data processing, if applicable:
            - single shot processing
            - data rotation and projection
            - creation of self.proc_data_dict['data_to_fit'] based on
                self.data_to_fit. This contains the data to be fitted by the
                children.
            - correction of probabilities by calibration matrix
            - data splitting
        """
        super().process_data()

        # handle single shot data
        if self.get_param_value("data_type", "averaged") == "singleshot":
            self.process_single_shots(
                predict_proba=self.predict_proba,
                classifier_params=self.get_param_value("classifier_params"),
                states_map=self.get_param_value("states_map"),
                thresholding=self.get_param_value("thresholding", False),)

        # create projected_data_dict
        if self.rotate:
            self.rotate_and_project_data()
        else:
            # this assumes data obtained with classifier detector!
            # ie pg, pe, pf are expected to be in the value_names
            self.proc_data_dict['projected_data_dict'] = OrderedDict()

            for qbn, data_dict in self.proc_data_dict[
                    'meas_results_per_qb'].items():
                self.proc_data_dict['projected_data_dict'][qbn] = OrderedDict()
                for state_prob in ['pg', 'pe', 'pf']:
                    self.proc_data_dict['projected_data_dict'][qbn].update(
                        {state_prob: data for key, data in data_dict.items()
                         if state_prob in key})

            # correct probabilities given calibration matrix
            if self.get_param_value("correction_matrix") is not None:
                self.proc_data_dict['projected_data_dict_corrected'] = \
                    OrderedDict()
                for qbn, data_dict in self.proc_data_dict[
                    'meas_results_per_qb'].items():
                    self.proc_data_dict['projected_data_dict_corrected'][qbn] = \
                        OrderedDict()
                    probas_raw = np.asarray([
                        data_dict[k] for k in data_dict for state_prob in
                        ['pg', 'pe', 'pf'] if state_prob in k])
                    corr_mtx = self.get_param_value("correction_matrix")[qbn]

                    if np.ndim(probas_raw) == 3:
                        assert self.get_param_value("TwoD", False) == True, \
                            "'TwoD' is False but data seems to be 2D"
                        # temporarily put 2D sweep into 1d for readout correction
                        sh = probas_raw.shape
                        probas_raw = probas_raw.reshape(sh[0], -1)
                        probas_corrected = np.linalg.inv(corr_mtx).T @ probas_raw
                        probas_corrected = probas_corrected.reshape(sh)
                    else:
                        probas_corrected = np.linalg.inv(corr_mtx).T @ probas_raw
                    self.proc_data_dict['projected_data_dict_corrected'][
                        qbn] = {key: data for key, data in
                                zip(["pg", "pe", "pf"], probas_corrected)}

        # add data_to_fit to proc_data_dict based on self.data_to_fit
        suffix = "_corrected" if self.get_param_value("correction_matrix")\
                                 is not None else ""
        self.proc_data_dict['data_to_fit'] = OrderedDict()
        for qbn, prob_data in self.proc_data_dict[
                'projected_data_dict' + suffix].items():
            if len(prob_data) and qbn in self.data_to_fit:
                self.proc_data_dict['data_to_fit'][qbn] = prob_data[
                    self.data_to_fit[qbn]]

        # handle data splitting if needed
        self.split_data()

    def split_data(self):
        def unique(l):
            try:
                return np.unique(l, return_inverse=True)
            except Exception:
                h = [repr(a) for a in l]
                _, i, j = np.unique(h, return_index=True, return_inverse=True)
                return l[i], j

        split_params = self.get_param_value('split_params', [])
        if not len(split_params):
            return

        pdd = self.proc_data_dict
        pdd['split_data_dict'] = {}

        for qbn in self.qb_names:
            pdd['split_data_dict'][qbn] = {}

            for p in split_params:
                dim = self.sp.find_parameter(p)
                sv = self.sp.get_sweep_params_property(
                    'values', param_names=p, dimension=dim)
                usp, ind = unique(sv)
                if len(usp) <= 1:
                    continue

                svs = [self.sp.subset(ind == i, dim) for i in
                          range(len(usp))]
                [s.remove_sweep_parameter(p) for s in svs]

                sdd = {}
                pdd['split_data_dict'][qbn][p] = sdd
                for i in range(len(usp)):
                    subset = (np.concatenate(
                        [ind == i,
                         [True] * len(pdd['sweep_points_dict'][qbn][
                                          'cal_points_sweep_points'])]))
                    sdd[i] = {}
                    sdd[i]['value'] = usp[i]
                    sdd[i]['sweep_points'] = svs[i]

                    d = pdd['sweep_points_dict'][qbn]
                    if dim == 0:
                        sdd[i]['sweep_points_dict'] = {
                            'sweep_points': d['sweep_points'][subset],
                            'msmt_sweep_points':
                                d['msmt_sweep_points'][ind == i],
                            'cal_points_sweep_points':
                                d['cal_points_sweep_points'],
                        }
                        sdd[i]['sweep_points_2D_dict'] = pdd[
                            'sweep_points_2D_dict'][qbn]
                    else:
                        sdd[i]['sweep_points_dict'] = \
                            pdd['sweep_points_dict'][qbn]
                        sdd[i]['sweep_points_2D_dict'] = {
                            k: v[ind == i] for k, v in pdd[
                            'sweep_points_2D_dict'][qbn].items()}
                    for d in ['projected_data_dict', 'data_to_fit']:
                        if isinstance(pdd[d][qbn], dict):
                            if dim == 0:
                                sdd[i][d] = {k: v[:, subset] for
                                             k, v in pdd[d][qbn].items()}
                            else:
                                sdd[i][d] = {k: v[ind == i, :] for
                                             k, v in pdd[d][qbn].items()}
                        else:
                            if dim == 0:
                                sdd[i][d] = pdd[d][qbn][:, subset]
                            else:
                                sdd[i][d] = pdd[d][qbn][ind == i, :]

        select_split = self.get_param_value('select_split')
        if select_split is not None:
            for qbn, select in select_split.items():
                p, v = select
                if p not in pdd['split_data_dict'][qbn]:
                    log.warning(f"Split parameter {p} for {qbn} not "
                                f"found. Ignoring this selection.")
                try:
                    ind = [a['value'] for a in pdd['split_data_dict'][
                        qbn][p].values()].index(v)
                except ValueError:
                    ind = v
                    try:
                        pdd['split_data_dict'][qbn][p][ind]
                    except ValueError:
                        log.warning(f"Value {v} for split parameter {p} "
                                    f"of {qbn} not found. Ignoring this "
                                    f"selection.")
                        continue
                for d in ['projected_data_dict', 'data_to_fit',
                          'sweep_points_2D_dict']:
                    pdd[d][qbn] = pdd['split_data_dict'][qbn][p][ind][d]
                for d in ['sweep_points_dict']:
                    # use update to preserve additional information in
                    # sweep_points_dict (in particular: param_names)
                    pdd[d][qbn].update(pdd['split_data_dict'][qbn][p][ind][d])
                self.measurement_strings[qbn] += f' ({p}: {v})'

    def get_num_cal_points(self):
        """
        Figures out how many calibration segments were used in the experiment.

        First tries to get this information from the cal_states_dict (i.e. from
        provided cal points info, see get_cal_points method).

        Then checks whether the number of sweep points matches the size of the
        data. If not, then either:
            1. no cal points information was provided but cal points were used
            in the experiment. In this case, the no_cp_but_cp_in_data is set
            to True (will be checked in update_sweep_points_dict).
            2. the data_type is "singleshot", and there is obviously a mismatch
             between the number of sweep points and the length of the data array.
            In that case, num_cal_points is assumed to be 0 and if
            there are any cal pointstreated correctly by the single_shot
            processing methods.

        Creates the attributes
            - self.num_cal_points
            - self.no_cp_but_cp_in_data
        """

        # Count num_cal_points from self.cal_states_dict
        self.num_cal_points = np.hstack(list(self.cal_states_dict[
                list(self.cal_states_dict)[0]].values())).size

        self.no_cp_but_cp_in_data = False
        spd = self.proc_data_dict['sweep_points_dict']
        num_sp = len(spd[list(spd)[0]]['sweep_points'])
        mrpq = self.proc_data_dict['meas_results_per_qb']
        mrpq_raw_dict = mrpq[list(mrpq)[0]]
        num_data_points = len(mrpq_raw_dict[list(mrpq_raw_dict)[0]])
        if self.num_cal_points == 0 and num_data_points != num_sp and \
                self.get_param_value('data_type', 'averaged') != 'singleshot':
            # No cal_points information was provided but cal points were part
            # of the measurement.
            self.num_cal_points = num_data_points - num_sp
            # Will be checked in update_sweep_points_dict
            self.no_cp_but_cp_in_data = True

    def update_sweep_points_dict(self):
        """
        Updates self.proc_data_dict['sweep_points_dict']:
            - 'sweep_points' are updated to include calibration points if
            these were part of the measurement (sweep points are extended
            with CalibrationPoints.extend_sweep_points_by_n_cal_pts)
            - 'msm_sweep_points' is added: sweep points corresponding to the
            data (i.e. without cal points); same as 'sweep_points' if no
            cal points were part of the measurement
            - 'cal_points_sweep_points' is added: sweep points corresponding
            to cal points; [] if no cal points were part of the measurement
        """
        cp_obj = None
        if hasattr(self, 'cp'):
            cp_obj = self.cp
        elif self.no_cp_but_cp_in_data:
            # self.no_cp_but_cp_in_data created in get_num_cal_points
            cp_obj = CalibrationPoints

        for qbn in self.qb_names:
            spd_qb = self.proc_data_dict['sweep_points_dict'][qbn]
            if cp_obj is not None:
                spd_qb['sweep_points'] = \
                    cp_obj.extend_sweep_points_by_n_cal_pts(
                        self.num_cal_points,
                        spd_qb['sweep_points'])
                # slicing with aux variable ind to be robust to the case
                # of 0 cal points
                ind = len(spd_qb['sweep_points']) - self.num_cal_points
                spd_qb['msmt_sweep_points'] = spd_qb['sweep_points'][:ind]
                spd_qb['cal_points_sweep_points'] = \
                    spd_qb['sweep_points'][ind:]
            else:
                spd_qb['msmt_sweep_points'] = spd_qb['sweep_points']
                spd_qb['cal_points_sweep_points'] = []

    def get_cal_states_dict_for_rotation(self):
        """
        Prepares for data rotation and projection by resolving what type of
        rotation/projection should be done on each qubit based on the
        information in self.cal_states_rotations, self.cal_states_dict, and
        self.rotation_type.

        Creates the attribute:
            - self.cal_states_dict_for_rotation: same as self.cal_states_dict
                but ordered based on transmon state levels.
                Will be used by self.rotate_and_project_data.
                Ex: self.cal_states_dict =
                        {'qb2': {'e': [-2], 'f': [-1], 'g': [-3]}}
                    self.cal_states_dict_for_rotation =
                        {'qb2': {'g': [-3], 'e': [-2], 'f': [-1]}}
                If pca is to be done on a qubit, then {qbn: None}.
                A None entry will tell rotate_and_normalize_data_IQ that pca
                should be done in the following methods:
                - rotate_data
                - rotate_data_TwoD
                - rotate_data_TwoD_same_fixed_cal_idxs
        """
        self.cal_states_dict_for_rotation = OrderedDict()
        cal_states_rotations = self.cal_states_rotations
        for qbn in self.qb_names:
            do_PCA = 'pca' in self.rotation_type[qbn].lower()
            self.cal_states_dict_for_rotation[qbn] = OrderedDict()
            cal_states_rot_qb = cal_states_rotations.get(qbn, {})
            cal_states = list(cal_states_rot_qb)
            # sort cal_states
            cal_states.sort(key=lambda s: cal_states_rot_qb[s])
            for cal_state in cal_states:
                if do_PCA or not len(self.cal_states_dict[qbn]):
                    # rotation_type is some form of pca or no cal points
                    # information was given --> do pca
                    self.cal_states_dict_for_rotation[qbn][cal_state] = None
                else:
                    # take data array index for cal_state
                    self.cal_states_dict_for_rotation[qbn][cal_state] = \
                        self.cal_states_dict[qbn][cal_state]

    def rotate_and_project_data(self):
        """
        Handles data rotation and projection based on calibration points
        information and self.rotation_type (see class docstring for what
        rotations types are recognized by the class).

        Can handle rotation and projection based on 0, 2, 3 (not for pca)
        calibration states.

        Creates self.proc_data_dict['projected_data_dict'].
        """
        # creates self.cal_states_dict_for_rotation
        self.get_cal_states_dict_for_rotation()
        self.proc_data_dict['projected_data_dict'] = OrderedDict(
            {qbn: '' for qbn in self.qb_names})

        for qbn in self.qb_names:
            cal_states_dict = self.cal_states_dict_for_rotation[qbn]
            if len(cal_states_dict) not in [0, 2, 3]:
                raise NotImplementedError('Calibration states rotation is '
                                          'currently only implemented for 0, '
                                          '2, or 3 cal states per qubit.')
            data_mostly_g = self.get_param_value('data_mostly_g',
                                                 default_value=True)
            if self.get_param_value('TwoD', default_value=False):
                if self.rotation_type[qbn].lower() == 'global_pca':
                    self.proc_data_dict['projected_data_dict'].update(
                        self.global_pca_TwoD(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.data_to_fit,
                            data_mostly_g=data_mostly_g))
                elif self.rotation_type[qbn].lower() == 'cal_states' and \
                        len(cal_states_dict) == 3:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_3_cal_states_TwoD(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map,
                            self.cal_states_dict_for_rotation))
                elif self.rotation_type[qbn].lower() == 'fixed_cal_points':
                    rotated_data_dict, zero_coord, one_coord = \
                        self.rotate_data_TwoD_same_fixed_cal_idxs(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.cal_states_dict_for_rotation,
                            self.data_to_fit)
                    self.proc_data_dict['projected_data_dict'].update(
                        rotated_data_dict)
                    self.proc_data_dict['rotation_coordinates'] = \
                        [zero_coord, one_coord]
                else:
                    column_PCA = self.rotation_type[qbn].lower() == 'column_pca'
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_TwoD(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.cal_states_dict_for_rotation,
                            self.data_to_fit, data_mostly_g=data_mostly_g,
                            column_PCA=column_PCA))

            else:
                if self.rotation_type[qbn].lower() == 'cal_states' and \
                        len(cal_states_dict) == 3:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data_3_cal_states(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map,
                            self.cal_states_dict_for_rotation))
                else:
                    self.proc_data_dict['projected_data_dict'].update(
                        self.rotate_data(
                            qbn, self.proc_data_dict['meas_results_per_qb'],
                            self.channel_map, self.cal_states_dict_for_rotation,
                            self.data_to_fit, data_mostly_g=data_mostly_g))

    @staticmethod
    def rotate_data_3_cal_states(qb_name, meas_results_per_qb, channel_map,
                                 cal_idxs_dict):
        # FOR 3 CAL STATES
        rotated_data_dict = OrderedDict()
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_idxs_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            raw_data = np.array([v for v in meas_res_dict.values()]).T
            for i, cal_idx in enumerate(cal_pts_idxs):
                cal_points_data[i, :] = np.mean(raw_data[cal_idx, :],
                                                axis=0)
            rotated_data = predict_proba_avg_ro(raw_data, cal_points_data)
            for i, state in enumerate(list(cal_idxs_dict[qb_name])):
                rotated_data_dict[qb_name][f'p{state}'] = rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        return rotated_data_dict

    @staticmethod
    def rotate_data(qb_name, meas_results_per_qb, channel_map,
                    cal_idxs_dict, storing_keys, data_mostly_g=True):
        # ONLY WORKS FOR 2 CAL STATES
        # We assign to main_cs the cal state that is found in storing_keys,
        # which is typically that passed by the user in data_to_fit
        main_cs = storing_keys[qb_name]
        if 'pca' not in main_cs.lower():
            # Add the other state probability
            # ex if main_cs == 'pe' add data for 'pg' as 1-pe
            qb_cal_states = cal_idxs_dict[qb_name].keys()
            if len(qb_cal_states) != 2:
                raise ValueError(f'Expected two cal states for {qb_name} but '
                                 f'found {len(qb_cal_states)}: {qb_cal_states}')
            other_cs = [cs for cs in qb_cal_states if cs != main_cs[-1]]
            if len(other_cs) == 0:
                raise ValueError(f'There are no other cal states except for '
                                 f'{main_cs[-1]} from storing_keys.')
            elif len(other_cs) > 1:
                raise ValueError(f'There is more than one other cal state in '
                                 f'addition to {main_cs[-1]} from'
                                 f' storing_keys. Not clear which one to use.')
            other_cs = f'p{other_cs[0]}'

            # rotate_and_normalize_data_IQ and rotate_and_normalize_data_1ch
            # return the qubit population corresponding to the highest cal state
            # (f>e>g). So the main_cs must be higher than the other_cs.
            # For example if the user passes g in data_to_fit main_cs=g
            # (main_cs just gets copied from data_to_fit). The bit of code here
            # swaps main_cs and other_cs only locally in this, so that main_cs
            # becomes e (or f), while data_to_fit stays g. Like this, we respect
            # the user's choice (by keeping data_to_fit) while handling the
            # rotation correctly (by changing the main_cs).
            # Below, states will have the order f, e, g because
            # cal_idxs_dict[qb_name] was ordered as g, e, f in
            # get_cal_states_dict_for_rotation
            states = list(cal_idxs_dict[qb_name])[::-1]
            probs = [main_cs, other_cs]
            # sort probs by order pf, pe, pg
            probs.sort(key=lambda p: states.index(p[-1]))
            # reassign to main_cs, other_cs such that main_cs > other_cs
            main_cs, other_cs = probs[0], probs[1]

        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        vals = list(cal_idxs_dict[qb_name].values())
        if len(cal_idxs_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_idxs_dict[qb_name].values())
            cal_pts_idxs.sort()
            cal_zero_points = cal_pts_idxs[0]
            cal_one_points = cal_pts_idxs[1]
        rotated_data_dict[qb_name] = OrderedDict()
        if len(meas_res_dict) == 1:
            # one RO channel per qubit
            if cal_zero_points is None and cal_one_points is None:
                data = meas_res_dict[list(meas_res_dict)[0]]
                data = (data - np.min(data))/(np.max(data) - np.min(data))
                data = a_tools.set_majority_sign(
                    data, -1 if data_mostly_g else 1)
                rotated_data_dict[qb_name][main_cs] = data
            else:
                rotated_data_dict[qb_name][main_cs] = \
                    a_tools.rotate_and_normalize_data_1ch(
                        data=meas_res_dict[list(meas_res_dict)[0]],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
            if 'pca' not in main_cs.lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][main_cs]
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            data, _, _ = a_tools.rotate_and_normalize_data_IQ(
                data=np.array([v for v in meas_res_dict.values()]),
                cal_zero_points=cal_zero_points,
                cal_one_points=cal_one_points)
            if cal_zero_points is None:
                data = a_tools.set_majority_sign(
                    data, -1 if data_mostly_g else 1)
            rotated_data_dict[qb_name][main_cs] = data
            if 'pca' not in main_cs.lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][main_cs]
        else:
            # multiple readouts per qubit per channel
            if isinstance(channel_map[qb_name], str):
                qb_ro_ch0 = channel_map[qb_name]
            else:
                qb_ro_ch0 = channel_map[qb_name][0]
            ro_suffixes = [s[len(qb_ro_ch0)+1::] for s in
                           list(meas_res_dict) if qb_ro_ch0 in s]
            for i, ro_suf in enumerate(ro_suffixes):
                rotated_data_dict[qb_name][ro_suf] = OrderedDict()
                if len(ro_suffixes) == len(meas_res_dict):
                    # one RO ch per qubit
                    if cal_zero_points is None and cal_one_points is None:
                        data = meas_res_dict[list(meas_res_dict)[i]]
                        data = (data - np.min(data))/(np.max(data) - np.min(data))
                        data = a_tools.set_majority_sign(
                            data, -1 if data_mostly_g else 1)
                        rotated_data_dict[qb_name][ro_suf][main_cs] = data
                    else:
                        rotated_data_dict[qb_name][ro_suf][main_cs] = \
                            a_tools.rotate_and_normalize_data_1ch(
                                data=meas_res_dict[list(meas_res_dict)[i]],
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
                else:
                    # two RO ch per qubit
                    keys = [k for k in meas_res_dict if ro_suf in k]
                    correct_keys = [k for k in keys
                                    if k[len(qb_ro_ch0)+1::] == ro_suf]
                    data_array = np.array([meas_res_dict[k]
                                           for k in correct_keys])
                    data, _, _ = a_tools.rotate_and_normalize_data_IQ(
                            data=data_array,
                            cal_zero_points=cal_zero_points,
                            cal_one_points=cal_one_points)
                    if cal_zero_points is None:
                        data = a_tools.set_majority_sign(
                            data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][ro_suf][main_cs] = data
                if 'pca' not in main_cs.lower():
                    rotated_data_dict[qb_name][ro_suf][other_cs] = \
                        1 - rotated_data_dict[qb_name][ro_suf][main_cs]
        return rotated_data_dict

    @staticmethod
    def rotate_data_3_cal_states_TwoD(qb_name, meas_results_per_qb,
                                      channel_map, cal_idxs_dict):
        # FOR 3 CAL STATES
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_idxs_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            for i, state in enumerate(list(cal_idxs_dict[qb_name])):
                rotated_data_dict[qb_name][f'p{state}'] = np.zeros(
                    raw_data_arr.shape)
            for col in range(raw_data_arr.shape[1]):
                raw_data = np.concatenate([
                    v[:, col].reshape(len(v[:, col]), 1) for
                    v in meas_res_dict.values()], axis=1)
                for i, cal_idx in enumerate(cal_pts_idxs):
                    cal_points_data[i, :] = np.mean(raw_data[cal_idx, :],
                                                    axis=0)
                # rotated data is (raw_data_arr.shape[0], 3)
                rotated_data = predict_proba_avg_ro(
                    raw_data, cal_points_data)

                for i, state in enumerate(list(cal_idxs_dict[qb_name])):
                    rotated_data_dict[qb_name][f'p{state}'][:, col] = \
                        rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        # transpose data
        for i, state in enumerate(list(cal_idxs_dict[qb_name])):
            rotated_data_dict[qb_name][f'p{state}'] = \
                rotated_data_dict[qb_name][f'p{state}'].T
        return rotated_data_dict

    @staticmethod
    def global_pca_TwoD(qb_name, meas_results_per_qb, channel_map,
                        storing_keys, data_mostly_g=True):
        meas_res_dict = meas_results_per_qb[qb_name]
        if list(meas_res_dict) != channel_map[qb_name]:
            raise NotImplementedError('Global PCA is only implemented '
                                      'for two-channel RO!')

        raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
        rotated_data_dict = OrderedDict({qb_name: OrderedDict()})
        rotated_data_dict[qb_name][storing_keys[qb_name]] = \
            deepcopy(raw_data_arr.transpose())
        data_array = np.array(
            [v.T.flatten() for v in meas_res_dict.values()])
        rot_flat_data, _, _ = \
            a_tools.rotate_and_normalize_data_IQ(
                data=data_array)
        data = np.reshape(rot_flat_data, raw_data_arr.T.shape)
        data = a_tools.set_majority_sign(data, -1 if data_mostly_g else 1)
        rotated_data_dict[qb_name][storing_keys[qb_name]] = data
        return rotated_data_dict

    @staticmethod
    def rotate_data_TwoD(qb_name, meas_results_per_qb, channel_map,
                         cal_idxs_dict, storing_keys,
                         column_PCA=False, data_mostly_g=True):
        # ONLY WORKS FOR 2 CAL STATES
        # We assign to main_cs the cal state that is found in storing_keys,
        # which is typically that passed by the user in data_to_fit
        main_cs = storing_keys[qb_name]
        if 'pca' not in main_cs.lower():
            # Add the other state probability
            # ex if main_cs == 'pe' add data for 'pg' as 1-pe
            qb_cal_states = cal_idxs_dict[qb_name].keys()
            if len(qb_cal_states) != 2:
                raise ValueError(f'Expected two cal states for {qb_name} but '
                                 f'found {len(qb_cal_states)}: {qb_cal_states}')
            other_cs = [cs for cs in qb_cal_states if cs != main_cs[-1]]
            if len(other_cs) == 0:
                raise ValueError(f'There are no other cal states except for '
                                 f'{main_cs[-1]} from '
                                 f'storing_keys.')
            elif len(other_cs) > 1:
                raise ValueError(f'There is more than one other cal state in '
                                 f'addition to {main_cs[-1]} from'
                                 f' storing_keys. Not clear which one to use.')
            other_cs = f'p{other_cs[0]}'

            # rotate_and_normalize_data_IQ and rotate_and_normalize_data_1ch
            # return the qubit population corresponding to the highest cal state
            # (f>e>g). So the main_cs must be higher than the other_cs. See
            # the comment in rotate_data for a more detailed explanation of the
            # lines of code below.
            states = list(cal_idxs_dict[qb_name])[::-1]
            probs = [main_cs, other_cs]
            probs.sort(key=lambda p: states.index(p[-1]))
            main_cs, other_cs = probs[0], probs[1]

        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        vals = list(cal_idxs_dict[qb_name].values())
        if len(cal_idxs_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_idxs_dict[qb_name].values())
            cal_pts_idxs.sort()
            cal_zero_points = cal_pts_idxs[0]
            cal_one_points = cal_pts_idxs[1]
        rotated_data_dict[qb_name] = OrderedDict()
        if len(meas_res_dict) == 1:
            # one RO channel per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][main_cs] = \
                deepcopy(raw_data_arr.transpose())
            if column_PCA:
                for row in range(raw_data_arr.shape[0]):
                    data = a_tools.rotate_and_normalize_data_1ch(
                        data=raw_data_arr[row, :],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
                    data = a_tools.set_majority_sign(
                        data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][main_cs][
                        :, row] = data
            else:
                for col in range(raw_data_arr.shape[1]):
                    data = a_tools.rotate_and_normalize_data_1ch(
                        data=raw_data_arr[:, col],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
                    if cal_zero_points is None:
                        data = a_tools.set_majority_sign(
                            data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][main_cs][col] = data
            if 'pca' not in main_cs.lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][main_cs]
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][main_cs] = \
                deepcopy(raw_data_arr.transpose())
            if column_PCA:
                for row in range(raw_data_arr.shape[0]):
                    data_array = np.array(
                        [v[row, :] for v in meas_res_dict.values()])
                    data, _, _ = \
                        a_tools.rotate_and_normalize_data_IQ(
                            data=data_array,
                            cal_zero_points=cal_zero_points,
                            cal_one_points=cal_one_points)
                    data = a_tools.set_majority_sign(
                        data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][main_cs][:, row] = data
            else:
                for col in range(raw_data_arr.shape[1]):
                    data_array = np.array(
                        [v[:, col] for v in meas_res_dict.values()])
                    data, _, _ = a_tools.rotate_and_normalize_data_IQ(
                        data=data_array,
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
                    if cal_zero_points is None:
                        data = a_tools.set_majority_sign(
                            data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][main_cs][col] = data
            if 'pca' not in main_cs.lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][main_cs]
        else:
            # multiple readouts per qubit per channel
            if isinstance(channel_map[qb_name], str):
                qb_ro_ch0 = channel_map[qb_name]
            else:
                qb_ro_ch0 = channel_map[qb_name][0]

            ro_suffixes = [s[len(qb_ro_ch0)+1::] for s in
                           list(meas_res_dict) if qb_ro_ch0 in s]

            for i, ro_suf in enumerate(ro_suffixes):
                rotated_data_dict[qb_name][ro_suf] = OrderedDict()
                if len(ro_suffixes) == len(meas_res_dict):
                    # one RO ch per qubit
                    raw_data_arr = meas_res_dict[list(meas_res_dict)[i]]
                    rotated_data_dict[qb_name][ro_suf][main_cs] = \
                        deepcopy(raw_data_arr.transpose())
                    for col in range(raw_data_arr.shape[1]):
                        data = a_tools.rotate_and_normalize_data_1ch(
                                data=raw_data_arr[:, col],
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
                        if cal_zero_points is None:
                            data = a_tools.set_majority_sign(
                                data, -1 if data_mostly_g else 1)
                        rotated_data_dict[qb_name][ro_suf][main_cs][col] = data
                else:
                    # two RO ch per qubit
                    raw_data_arr = meas_res_dict[list(meas_res_dict)[i]]
                    rotated_data_dict[qb_name][ro_suf][main_cs] = \
                        deepcopy(raw_data_arr.transpose())
                    for col in range(raw_data_arr.shape[1]):
                        data_array = np.array(
                            [v[:, col] for k, v in meas_res_dict.items()
                             if ro_suf in k])
                        data, _, _ = a_tools.rotate_and_normalize_data_IQ(
                                data=data_array,
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
                        if cal_zero_points is None:
                            data = a_tools.set_majority_sign(
                                data, -1 if data_mostly_g else 1)
                        rotated_data_dict[qb_name][ro_suf][main_cs][col] = data
                if 'pca' not in main_cs.lower():
                    rotated_data_dict[qb_name][ro_suf][other_cs] = \
                        1 - rotated_data_dict[qb_name][ro_suf][main_cs]
        return rotated_data_dict

    @staticmethod
    def rotate_data_TwoD_same_fixed_cal_idxs(qb_name, meas_results_per_qb,
                                             channel_map, cal_idxs_dict,
                                             storing_keys):
        meas_res_dict = meas_results_per_qb[qb_name]
        if list(meas_res_dict) != channel_map[qb_name]:
            raise NotImplementedError('rotate_data_TwoD_same_fixed_cal_idxs '
                                      'only implemented for two-channel RO!')

        vals = list(cal_idxs_dict[qb_name].values())
        if len(cal_idxs_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_idxs_dict[qb_name].values())
            cal_pts_idxs.sort()
            cal_zero_points = cal_pts_idxs[0]
            cal_one_points = cal_pts_idxs[1]

        # do pca on the one cal states
        raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
        rot_dat_e = np.zeros(raw_data_arr.shape[1])
        for row in cal_one_points:
            rot_dat_e += a_tools.rotate_and_normalize_data_IQ(
                data=np.array([v[row, :] for v in meas_res_dict.values()]),
                cal_zero_points=None, cal_one_points=None)[0]
        rot_dat_e /= len(cal_one_points)

        # find the values of the zero and one cal points
        col_idx = np.argmax(np.abs(rot_dat_e))
        zero_coord = [np.mean([v[r, col_idx] for r in cal_zero_points])
                      for v in meas_res_dict.values()]
        one_coord = [np.mean([v[r, col_idx] for r in cal_one_points])
                     for v in meas_res_dict.values()]

        # rotate all data based on the fixed zero_coord and one_coord
        rotated_data_dict = OrderedDict({qb_name: OrderedDict()})
        rotated_data_dict[qb_name][storing_keys[qb_name]] = \
            deepcopy(raw_data_arr.transpose())
        for col in range(raw_data_arr.shape[1]):
            data_array = np.array(
                [v[:, col] for v in meas_res_dict.values()])
            rotated_data_dict[qb_name][
                storing_keys[qb_name]][col], _, _ = \
                a_tools.rotate_and_normalize_data_IQ(
                    data=data_array,
                    zero_coord=zero_coord,
                    one_coord=one_coord)

        return rotated_data_dict, zero_coord, one_coord

    def get_transition_name(self, qb_name):
        """
        Extracts the transition_name:
            - first by taking transition_name_input from the task in task_list
                for qb_name
            - then from options_dict/metadata
        If not found in any of the above, it is inferred from data_to_fit.
        :param qb_name: qubit name
        :return: string indicating the transition name ("ge", "ef", etc.)
        """
        task_list = self.get_param_value('preprocessed_task_list')
        trans_name = self.get_param_value('transition_name')
        if task_list is not None:
            task = [t for t in task_list if t['qb'] == qb_name][0]
            trans_name = task.get('transition_name_input', trans_name)

        if trans_name is None:
            # This is a fallback but not ideal because data_to_fit gets
            # overwritten by this class for certain rotation types.
            if 'pca' in self.data_to_fit.get(qb_name, '').lower():
                dtf = self._get_default_data_to_fit()
            else:
                dtf = self.data_to_fit
            if 'h' in dtf.get(qb_name, ''):
                trans_name = 'fh'
            elif 'f' in dtf.get(qb_name, ''):
                trans_name = 'ef'
            else:
                trans_name = 'ge'
        return trans_name

    def get_xaxis_label_unit(self, qb_name):
        hard_sweep_params = self.get_param_value('hard_sweep_params')
        sweep_name = self.get_param_value('sweep_name')
        sweep_unit = self.get_param_value('sweep_unit')
        if self.sp is not None:
            param_names = self.proc_data_dict['sweep_points_dict'][qb_name][
                'param_names']
            _, xunit, xlabel = self.sp.get_sweep_params_description(
                param_names=param_names, dimension=0)[0]
        elif hard_sweep_params is not None:
            xlabel = list(hard_sweep_params)[0]
            xunit = list(hard_sweep_params.values())[0][
                'unit']
        elif (sweep_name is not None) and (sweep_unit is not None):
            xlabel = sweep_name
            xunit = sweep_unit
        else:
            xlabel = self.raw_data_dict['sweep_parameter_names']
            xunit = self.raw_data_dict['sweep_parameter_units']
        if np.ndim(xlabel) > 0:
            xlabel = xlabel[0]
        if np.ndim(xunit) > 0:
            xunit = xunit[0]
        return xlabel, xunit

    @staticmethod
    def get_cal_state_color(cal_state_label):
        if cal_state_label == 'g' or cal_state_label == r'$|g\rangle$':
            return 'k'
        elif cal_state_label == 'e' or cal_state_label == r'$|e\rangle$':
            return 'gray'
        elif cal_state_label == 'f' or cal_state_label == r'$|f\rangle$':
            return 'C8'
        elif cal_state_label == 'h' or cal_state_label == r'$|h\rangle$':
            return 'C5'
        else:
            return 'C6'

    @staticmethod
    def get_latex_prob_label(prob_label):
        if '$' in prob_label:
            return prob_label
        else:
            # search for "p" plus a letter between a and z, enclosed by
            # underscores (or at beginning/end of string)
            res = re.search('_p([a-z])_', '_' + prob_label.lower() + '_')
            if res:
                return r'$|{}\rangle$'.format(res.expand(r'\1'))
            else:
                return r'$|{}\rangle$'.format(prob_label)

    def get_yaxis_label(self, qb_name, data_key=None):
        if self.rotate and (self.rotation_type[qb_name].lower() not in
                            ['cal_states', 'fixed_cal_points']
                            or not len(self.cal_states_dict[
                                        list(self.cal_states_dict)[0]])):
            # some kind of pca was done
            return 'Strongest principal component (arb.)'
        else:
            if data_key is None:
                if self.data_to_fit.get(qb_name, None) is not None:
                    return '{} state population'.format(
                        self.get_latex_prob_label(self.data_to_fit[qb_name]))
                else:
                    return 'Measured data'
            else:
                return '{} state population'.format(
                        self.get_latex_prob_label(data_key))

    def get_soft_sweep_label_unit(self, param_name):
        """
        Finds the label and unit of a soft sweep parameter.

        If a SweepPoints instance exists, takes the soft sweep parameter
        corresponding to param_name. Otherwise, takes the first soft sweep
        parameter.

        Args:
            param_name (str): name of a soft sweep parameter in SweepPoints

        Returns:
            label and unit of a soft sweep parameter
        """
        if self.sp is not None:
            unit = self.sp.get_sweep_params_property(
                'unit', dimension=1, param_names=param_name)
            label = self.sp.get_sweep_params_property(
                'label', dimension=1, param_names=param_name)
        else:
            soft_sweep_params = self.get_param_value(
                'soft_sweep_params')
            if soft_sweep_params is not None:
                label = list(soft_sweep_params.values())[0]['label']
                unit = list(soft_sweep_params.values())[0]['unit']
            else:
                label = self.raw_data_dict['sweep_parameter_names'][1]
                unit = self.raw_data_dict['sweep_parameter_units'][1]
            if np.ndim(label) > 0:
                label = label[0]
            if np.ndim(unit) > 0:
                unit = unit[0]
        return label, unit

    def _get_single_shots_per_qb(self, raw=False, qb_names=None):
        """
        Gets single shots from the proc_data_dict and arranges
        them as arrays per qubit
        Args:
            raw (bool): whether or not to return raw shots (before
            data filtering)
            qb_names (list): Qubits for which to extract the data. If None
            (default): uses self.qb_names. This can be useful if extracting
            data for qubits which are not in self.qb_names.
            E.g.: Doing an experiment on qb1 (self.qb_names = ['qb1']) but
            preselecting on qb2, meaning that we need the data from both.
        Returns: shots_per_qb: dict where keys are qb_names and
            values are arrays of shape (n_shots, n_value_names) for
            1D measurements and (n_shots*n_soft_sp, n_value_names) for
            2D measurements

        """
        # prepare data in convenient format, i.e. arrays per qubit
        shots_per_qb = dict()        # store shots per qb and per state
        pdd = self.proc_data_dict    # for convenience of notation
        key = 'meas_results_per_qb'
        if raw:
            key += "_raw"
        if qb_names is None:
            qb_names = self.qb_names
        for qbn in qb_names:
            # if "1D measurement" , shape is (n_shots, n_vn) i.e. one
            # column for each value_name (often equal to n_ro_ch)
            shots_per_qb[qbn] = \
                np.asarray(list(
                    pdd[key][qbn].values())).T
            # if "2D measurement" reshape from (n_soft_sp, n_shots, n_vn)
            #  to ( n_shots * n_soft_sp, n_ro_ch)
            if np.ndim(shots_per_qb[qbn]) == 3:
                assert self.get_param_value("TwoD", False) == True, \
                    "'TwoD' is False but single shot data seems to be 2D"
                n_vn = shots_per_qb[qbn].shape[-1]
                # put softsweep as inner most loop for easier processing
                shots_per_qb[qbn] = np.swapaxes(shots_per_qb[qbn], 0, 1)
                # reshape to 2D array
                shots_per_qb[qbn] = shots_per_qb[qbn].reshape((-1, n_vn))
            # make 2D array in case only one channel (1D array)
            elif np.ndim(shots_per_qb[qbn]) == 1:
                shots_per_qb[qbn] = np.expand_dims(shots_per_qb[qbn],
                                                   axis=-1)

        return shots_per_qb

    def _get_preselection_masks(self, presel_shots_per_qb, preselection_qbs=None,
                                predict_proba=True,
                                classifier_params=None,
                                preselection_state_int=0):
        """
        Prepares preselection masks for each qubit considered in the keys of
        "preselection_qbs" using the preslection readouts of presel_shots_per_qb.
        Note: this function replaces the use of the "data_filter" lambda function
        in the case of single_shot readout.
        TODO: in the future, it might make sense to merge this function
         with the data_filter.
        Args:
            presel_shots_per_qb (dict): {qb_name: preselection_shot_readouts}
            preselection_qbs (dict): keys are the qubits for which the masks have to be
                computed and values are list of qubit to consider jointly for preselection.
                e.g. {"qb1": ["qb1", "qb2"], "qb2": ["qb2"]}. In this case shots of qb1 will
                only be kept if both qb1 and qb2 are in the state specified by
                preselection_state_int (usually, the ground state), while qb2 is preselected
                independently of qb1.
                If None (by default) or empty: in this case each qubit is
                preselected independently from others.
            predict_proba (bool): whether or not to consider input as raw voltages shots.
                Should be false if input shots are already probabilities, e.g. when using
                classified readout.

            classifier_params (dict): classifier params
            preselection_state_int (int): integer corresponding to the state of the classifier
                on which preselection should be performed. Defaults to 0 (i.e. ground state
                in most cases).

        Returns:
            preselection_masks (dict): dictionary of boolean arrays of shots to keep
            (indicated with True) for each qubit

        """
        presel_mask_single_qb = {}
        for qbn, presel_shots in presel_shots_per_qb.items():
            if not predict_proba:
                # shots were obtained with classifier detector and
                # are already probas
                presel_proba = presel_shots_per_qb[qbn]
            else:
                # use classifier calibrated to classify preselection readouts
                presel_proba = a_tools.predict_gm_proba_from_clf(
                    presel_shots_per_qb[qbn], classifier_params[qbn])
            presel_classified = np.argmax(presel_proba, axis=1)
            # create boolean array of shots to keep.
            # each time ro is the ground state --> true otherwise false
            presel_mask_single_qb[qbn] = presel_classified == preselection_state_int

            if np.sum(presel_mask_single_qb[qbn]) == 0:
                # FIXME: Nathan should probably not be error but just continue
                #  without preselection ?
                raise ValueError(f"{qbn}: No data left after preselection!")

        # compute final mask taking into account all qubits in presel_qubits for each qubit
        presel_mask = {}

        if preselection_qbs is None or len(preselection_qbs) == 0:
            # default is each qubit preselected individually
            preselection_qbs = {qbn: [qbn] for qbn in presel_shots_per_qb}

        for qbn, presel_qbs in preselection_qbs.items():
            if len(presel_qbs) == 1:
                presel_qbs = [presel_qbs[0], presel_qbs[0]]
            presel_mask[qbn] = np.logical_and.reduce(
                [presel_mask_single_qb[qb] for qb in presel_qbs])

        return presel_mask

    def process_single_shots(self, predict_proba=False,
                             classifier_params=None,
                             states_map=None,
                             thresholding=False):
        """
        Processes single shots from proc_data_dict("meas_results_per_qb")
        This includes assigning probabilities to each shot (optional),
        preselect shots on the ground state if there is a preselection readout,
        average the shots/probabilities.

        Args:
            predict_proba (bool): whether or not to assign probabilities to shots.
                If True, it assumes that shots in the proc_data_dict are the
                raw voltages on n channels. If False, it assumes either that
                shots were acquired with the classifier detector (i.e. shots
                are the probabilities of being in each state of the classifier)
                or that they are raw voltages. Note that when preselection
                the function checks for "classified_ro" and if it is false,
                 (i.e. the input are raw voltages and not probas) then it uses
                  the classifier on the preselection readouts regardless of the
                  "predict_proba" flag (preselection requires classif of ground state).
            classifier_params (dict): dict where keys are qb_names and values
                are dictionaries of classifier parameters passed to
                a_tools.predict_proba_from_clf(). Defaults to
                qb.acq_classifier_params(). Note: it
            states_map (dict):
                list of states corresponding to the different integers output
                by the classifier. Defaults to  {0: "g", 1: "e", 2: "f", 3: "h"}
            thresholding (bool):
                whether or not to threshold (i.e. classify) the shots. If True,
                it will transform [0.01, 0.97, 0.02] into [0, 1, 0].

        Other parameters taken from self.get_param_value:
            use_preselection (bool): whether or not preselection should be used
                before averaging. If true, then checks if there is a preselection
                readout in prep_params and if so, performs preselection on the
                ground state
            n_shots (int): number of shots per readout. Used to infer the number
                of readouts. Defaults to qb.acq_shots. WATCH OUT, sometimes
                for mutli-qubit detector uses max(qb.acq_shots() for qb in qbs),
                such that acq_shots found in the hdf5 file might be different than
                the actual number of shots used for the experiment.
                it is therefore safer to pass the number of shots in the metadata.
            TwoD (bool): Whether data comes from a 2D sweep, i.e. several concatenated
                sequences. Used for proper reshaping when using preselection
        Returns:

        """
        if states_map is None:
            states_map = {0: "g", 1: "e", 2: "f", 3: "h"}

        # get preselection information
        prep_params_presel = self.prep_params.get('preparation_type', "wait") \
                             == "preselection"
        use_preselection = self.get_param_value("use_preselection", True)
        # activate preselection flag only if preselection is in prep_params
        # and the user wants to use the preselection readouts
        preselection = prep_params_presel and use_preselection

        # returns for each qb: (n_shots, n_ch) or (n_soft_sp* n_shots, n_ch)
        # where n_soft_sp is the inner most loop i.e. the first dim is ordered as
        # (shot0_ssp0, shot0_ssp1, ... , shot1_ssp0, shot1_ssp1, ...)
        shots_per_qb = self._get_single_shots_per_qb()

        # save single shots in proc_data_dict, as they will be overwritten in
        # 'meas_results_per_qb' with their averaged values for the rest of the
        # analysis to work.
        self.proc_data_dict['single_shots_per_qb'] = deepcopy(shots_per_qb)

        # determine number of shots
        n_shots = self.get_param_value("n_shots")
        if n_shots is None:
            # FIXME: this extraction of number of shots won't work with soft repetitions.
            # FIXME: refactor to use settings manager instead of raw_data_dict
            n_shots_from_hdf = [
                int(self.get_data_from_timestamp_list({
                    f'sh': f"Instrument settings.{qbn}.acq_shots"})['sh']) for qbn in self.qb_names]
            if len(np.unique(n_shots_from_hdf)) > 1:
                log.warning("Number of shots extracted from hdf are not all the same:"
                            "assuming n_shots=max(qb.acq_shots() for qb in qb_names)")
            n_shots = np.max(n_shots_from_hdf)

        # determine number of readouts per sequence
        if self.get_param_value("TwoD", False):
            n_seqs = self.sp.length(1)  # corresponds to number of soft sweep points
        else:
            n_seqs = 1
        # n_readouts refers to the number of readouts per sequence after
        # filtering out e.g. preselection readouts
        n_readouts = list(shots_per_qb.values())[0].shape[0] // (n_shots *
                                                                 n_seqs)

        # get qubits for which data should be extracted
        preselection_qbs = self.get_param_value("preselection_qbs")
        if preselection_qbs is None:
            # by default, extract data for all qubits in self.qb_names
            qb_names_to_extract = self.qb_names
        else:
            # Data should be extracted both for qubits in self.qb_names
            # (qubits on which the experiment is performed), and for qubits
            # used in preselection. These two sets may or not be distinct.
            # Preselection will not work if data is not extracted for the
            # preselection qubits as well.
            # E.g. self.qb_names = ['qb1'] (only qubit to analyse) and
            # preselection_qbs = {"qb1": ["qb1", "qb2"]},
            # i.e. preselection is done for qb1 using ['qb1', 'qb2'] so we
            # should exctract data no only for qb1 but also qb2.
            qb_names_to_extract = [qbn for qbns in list(
                preselection_qbs.values()) for qbn in qbns]
            qb_names_to_extract = qb_names_to_extract + [
                qbn for qbn in self.qb_names if qbn not in qb_names_to_extract]

        # get classification parameters
        if classifier_params is None:
            classifier_params = self.get_data_from_timestamp_list({
                f'{qbn}': f"Instrument settings.{qbn}.acq_classifier_params"
                for qbn in qb_names_to_extract})

        # prepare preselection mask
        if preselection:

            # get preselection readouts for selected qubits
            shots_per_qb_before_filtering = self._get_single_shots_per_qb(
                raw=True, qb_names=qb_names_to_extract)

            n_ro_before_filtering = \
                list(shots_per_qb_before_filtering.values())[0].shape[0] // \
                (n_shots * n_seqs)
            # The following lines assume a single preselection readout and
            # an arbitrary number n_readouts_per_segment of other readouts.
            # n_readouts = n_readouts_per_segment * n_segs
            # n_ro_before_filtering = (1+n_readouts_per_segment) * n_segs
            n_readouts_per_segment = n_readouts // (n_ro_before_filtering -
                                                    n_readouts)
            n_segs = n_readouts // n_readouts_per_segment
            # FIXME follows the data format of _get_single_shots_per_qb,
            #  hence the tiling by n_seqs. Array shapes should probably be
            #  cleaned up, after fixing transposed arrays in proc_data_dict
            #  as well, see FIXME in BaseDataAnalysis.__init__
            preselection_ro_mask = \
                np.tile([True] * n_seqs +
                        [False] * n_readouts_per_segment * n_seqs,
                        n_shots * n_segs)
            presel_shots_per_qb = \
                {qbn: ps[preselection_ro_mask]
                 for qbn, ps in shots_per_qb_before_filtering.items()}
            # create boolean array of shots to keep.
            # each time ro is the ground state --> true otherwise false
            g_state_int = [k for k, v in states_map.items() if v == "g"][0]
            preselection_masks = self._get_preselection_masks(
                presel_shots_per_qb,
                preselection_qbs=self.get_param_value("preselection_qbs"),
                predict_proba=not self.get_param_value('classified_ro', False),
                classifier_params=classifier_params,
                preselection_state_int=g_state_int)
            self.proc_data_dict['percent_data_after_presel'] = {} #initialize
        else:
            # keep all shots
            preselection_masks = {qbn: np.ones(len(shots), dtype=bool)
                                  for qbn, shots in shots_per_qb.items()}
        self.proc_data_dict['preselection_masks'] = preselection_masks

        # process single shots per qubit
        for qbn, shots in shots_per_qb.items():
            if predict_proba:
                # shots become probabilities with shape (n_shots, n_states)
                try:
                    shots = a_tools.predict_gm_proba_from_clf(
                        shots, classifier_params[qbn])
                except ValueError as e:
                    log.error(f'If the following error relates to number'
                              ' of features, probably wrong classifer'
                              ' parameters were passed (e.g. a classifier'
                              ' trained with a different number of channels'
                              ' than in the current measurement): {e}')
                    raise e

            if thresholding:
                # shots become one-hot encoded arrays with length n_states
                # shots has shape (n_shots, n_states)

                shots = a_tools.threshold_shots(shots)

                if 'single_shots_per_qb_thresholded' not in self.proc_data_dict:
                    self.proc_data_dict['single_shots_per_qb_thresholded'] = {}
                self.proc_data_dict['single_shots_per_qb_thresholded'][qbn] = \
                    shots

            averaged_shots = [] # either raw voltage shots or probas
            preselection_percentages = []
            for ro in range(n_readouts*n_seqs):
                shots_single_ro = shots[ro::n_readouts*n_seqs]
                presel_mask_single_ro = preselection_masks[qbn][ro::n_readouts*n_seqs]
                preselection_percentages.append(100*np.sum(presel_mask_single_ro)/
                                                len(presel_mask_single_ro))
                averaged_shots.append(
                    np.mean(shots_single_ro[presel_mask_single_ro], axis=0))
            if self.get_param_value("TwoD", False):
                # Shape: (n_readouts*n_seqs, n), with n = n_prob or n_ch or 1
                averaged_shots = np.reshape(averaged_shots, (n_readouts, n_seqs, -1))
                # Shape: (n_readouts, n_seqs, n)
                averaged_shots = np.swapaxes(averaged_shots, 0, 1) # return to original 2D shape
                # Shape: (n_seqs, n_readouts, n)
            averaged_shots = np.array(averaged_shots).T
            # Shape: (n, n_readouts) if 1d, or (n, n_readouts, n_ssp) if 2d

            if preselection:
                self.proc_data_dict['percent_data_after_presel'][qbn] = \
                    f"{np.mean(preselection_percentages):.2f} $\\pm$ " \
                    f"{np.std(preselection_percentages):.2f}%"
            if predict_proba:
                # value names are different from what was previously in
                # meas_results_per_qb and therefore "artificial" values
                # are made based on states
                self.proc_data_dict['meas_results_per_qb'][qbn] = \
                    {"p" + states_map[i]: p for i, p in enumerate(averaged_shots)}
            else:
                # reuse value names that were already there if did not classify
                for i, k in enumerate(
                        self.proc_data_dict['meas_results_per_qb'][qbn]):
                    self.proc_data_dict['meas_results_per_qb'][qbn][k] = \
                        averaged_shots[i]

    def prepare_plots(self):
        """
        Prepares the plot dicts for the raw data and the projected data.
        """
        self.prepare_raw_data_plots()
        self.prepare_projected_data_plots()

    def prepare_raw_data_plots(self):
        """
        Prepares plots of the raw data for each qubit, stored in
        proc_data_dict['meas_results_per_qb(_raw)'].

        Calls
            - _prepare_raw_data_plots if plot_raw_data (passed in the
             options_dict, metadata, or default_options) is True.
            - _prepare_raw_1d_slices_plots if the data is TwoD and
             slice_idxs_1d_raw_plot was passed in options_dict, metadata, or
             default_options

        If the measurement had active reset readouts (specified in
        preparation_params), the two functions above are called twice: once
        for the data in meas_results_per_qb (data points corresponding to
        active reset readouts are filtered out), and once for the data in
        meas_results_per_qb_raw (active reset readouts included).
        """
        TwoD = self.get_param_value('TwoD', False)
        slice_idxs_1d_raw_plot = self.get_param_value(
            'slice_idxs_1d_raw_plot', {})
        plot_raw_data = self.get_param_value('plot_raw_data', True)
        for qb_name in self.qb_names:
            slice_idxs_list = slice_idxs_1d_raw_plot.get(qb_name, [])
            # Plot all raw data (including potential active reset readouts)
            if self.data_with_filter:
                # With, then without the active reset readouts
                keys = ['meas_results_per_qb_raw', 'meas_results_per_qb']
                fig_suffixes = ['', 'filtered']
            else:
                # Raw data (no active reset readouts)
                keys = ['meas_results_per_qb']
                fig_suffixes = ['']
            # For each key, create a figure with corresponding fig_suffix
            for key, fig_suffix in zip(keys, fig_suffixes):
                raw_data_dict = self.proc_data_dict[key][qb_name]
                if key == 'meas_results_per_qb_raw':
                    sweep_points = self.raw_data_dict['hard_sweep_points']
                elif key == 'meas_results_per_qb':
                    sweep_points = self.proc_data_dict[
                        'sweep_points_dict'][qb_name]['sweep_points']
                else:
                    raise ValueError
                if plot_raw_data:
                    # standard raw data plot
                    self._prepare_raw_data_plots(qb_name, raw_data_dict,
                                                 xvals=sweep_points,
                                                 fig_suffix=fig_suffix)
                if TwoD and len(slice_idxs_list) > 0:
                    # plot slices of the 2D raw data
                    self._prepare_raw_1d_slices_plots(qb_name, raw_data_dict,
                                                      slice_idxs_list)

    def _prepare_raw_1d_slices_plots(self, qb_name, raw_data_dict,
                                     slice_idxs_list):
        """
        Prepares 1d plots of slices from a TwoD raw data plot.

        Args:
            qb_name (str): name of the qubit
            raw_data_dict (dict): the dictionary containing the raw data, with
                qubit names as keys.
                Typically proc_data_dict['meas_results_per_qb(_raw)'].
            slice_idxs_list (list): list of tuples of the form (idxs, axis)
                (see class docstring).
                Example: [('8:13', 'row'), (0, 'col')]
        """
        for slice_idxs in slice_idxs_list:
            idxs, axis, xvals, xlabel, xunit = \
                self.get_1d_slice_params(qb_name, slice_idxs)
            for idx in idxs:
                fig_suffix = \
                    f'{"_row" if axis == 0 else "_col"}_{idx}'
                self._prepare_raw_data_plots(qb_name, raw_data_dict,
                                             xvals, idx, axis,
                                             fig_suffix=fig_suffix,
                                             TwoD=False,
                                             xlabel=xlabel, xunit=xunit)

    def _prepare_raw_data_plots(self, qb_name, raw_data_dict, xvals,
                                twod_data_idx=None, twod_data_axis=None,
                                fig_name='raw_plot', fig_suffix='',
                                xlabel=None, xunit=None, TwoD=None):
        """
        Prepares plots of the raw data in raw_data_dict.

        Args:
            qb_name (str): qubit name
            raw_data_dict (dict): the dictionary containing the raw data, where
                the keys are names of acquisition channels (usually the value
                names created by the detector functions).
                Typically, this dict corresponds to
                proc_data_dict['meas_results_per_qb(_raw)'].
            xvals (list/array): x-axis values. Typically, the hard/1d sweep
                points
            twod_data_idx (int): index of a slice in a 2D data array. Will be
                ignored if `None`. More details below argument list. Defaults
                to `None`.
            twod_data_axis (int): axis of a 2D data array (0 for row, 1 for
                col). Will be ignored if `None`. More details below argument
                list. Defaults to `None`.
            fig_name (str): name of the figure
            fig_suffix (str): suffix to the figure name. qb_name is
                automatically inserted between fig_name and fig_suffix.
            xlabel (str): x-axis label, typically corresponding to the hard/1d
                sweep parameter
            xunit (str): x-axis unit, typically corresponding to the hard/1d
                sweep parameter
            TwoD (bool): whether to prepare a 2D plot

        If the data is 2D, and twod_data_idx and twod_data_axis are passed,
        this function will prepare a 1D plot of the slice in the 2D data
        at index twod_data_idx along axis twod_data_axis. Parameter "TwoD"
        should be False in this case in order to avoid preparing a 2D plot
        in addition.

        If xlabel, xunit are None, they are taken from the SweepPoints or from
        the metadata as the ones corresponding to the hard/1d sweep parameter.
        They can also correspond to a soft sweep parameter if twod_data_idx and
        twod_data_axis are passed, and twod_data_axis is 1 (column).
        """
        if len(raw_data_dict) == 1:
            numplotsx = 1
            numplotsy = 1
        elif len(raw_data_dict) == 2:
            numplotsx = 1
            numplotsy = 2
        else:
            numplotsx = 2
            numplotsy = len(raw_data_dict) // 2 + len(raw_data_dict) % 2

        plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
        fig_title = (self.raw_data_dict['timestamp'] + ' ' +
                     self.raw_data_dict['measurementstring'] +
                     '\nRaw data ' + fig_suffix + ' ' + qb_name)
        plot_name = f'{fig_name}_{qb_name}_{fig_suffix}'
        xl, xu = self.get_xaxis_label_unit(qb_name)
        if xlabel is None:
            xlabel = xl
        if xunit is None:
            xunit = xu

        value_units = deepcopy(self.raw_data_dict['value_units'])
        if not isinstance(value_units, list):
            value_units = [value_units]
        value_units = {vn: vu for vn, vu in zip(
            self.raw_data_dict['measured_data'], value_units)}

        if TwoD is None:
            TwoD = self.get_param_value('TwoD', False)
        prep_1d_plot = True
        for ax_id, ro_channel in enumerate(raw_data_dict):
            ro_unit = value_units.get(ro_channel, 'a.u.')
            if TwoD:
                sp2dd = self.proc_data_dict['sweep_points_2D_dict'][qb_name]
                if len(sp2dd) >= 1 and len(sp2dd[list(sp2dd)[0]]) > 1:
                    # Only prepare 2D plots when there is more than one soft
                    # sweep point. When there is only one soft sweep point
                    # we want to do 1D plots which are more meaningful
                    prep_1d_plot = False
                    for pn, ssp in sp2dd.items():
                        ylabel, yunit = self.get_soft_sweep_label_unit(pn)
                        self.plot_dicts[f'{plot_name}_{ro_channel}_{pn}'] = {
                            'fig_id': plot_name + '_' + pn,
                            'ax_id': ax_id,
                            'plotfn': self.plot_colorxy,
                            'xvals': xvals,
                            'yvals': ssp,
                            'zvals': raw_data_dict[ro_channel].T,
                            'xlabel': xlabel,
                            'xunit': xunit,
                            'ylabel': ylabel,
                            'yunit': yunit,
                            'numplotsx': numplotsx,
                            'numplotsy': numplotsy,
                            'plotsize': (plotsize[0]*numplotsx,
                                         plotsize[1]*numplotsy),
                            'title': fig_title,
                            'clabel': f'{ro_channel} ({ro_unit})'}

            if prep_1d_plot:
                yvals = raw_data_dict[ro_channel]
                if len(yvals.shape) > 1 and yvals.shape[1] == 1:
                    # only one soft sweep point: prepare 1D plot which is
                    # more meaningful
                    yvals = np.squeeze(yvals, axis=1)
                elif yvals.ndim == 2:
                    # Plot 1D slice of raw data at data_idx along data_axis
                    if twod_data_idx is None or twod_data_axis is None:
                        raise ValueError('Both "twod_data_idx" and '
                                         '"twod_data_axis" must be specified '
                                         'in order to plot 1D a slice of the '
                                         'TwoD raw data.')
                    yvals = np.take_along_axis(
                        yvals.T,
                        np.array([[twod_data_idx]]), twod_data_axis).flatten()
                self.plot_dicts[plot_name + '_' + ro_channel] = {
                    'fig_id': plot_name,
                    'ax_id': ax_id,
                    'plotfn': self.plot_line,
                    'xvals': xvals,
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': yvals,
                    'ylabel': f'{ro_channel} ({ro_unit})',
                    'yunit': '',
                    'numplotsx': numplotsx,
                    'numplotsy': numplotsy,
                    'plotsize': (plotsize[0]*numplotsx,
                                 plotsize[1]*numplotsy),
                    'title': fig_title}
        if len(raw_data_dict) == 1:
            self.plot_dicts[
                plot_name + '_' + list(raw_data_dict)[0]]['ax_id'] = None

    def prepare_projected_data_plots(self):
        """
        Prepares plots of the projected data for each qubit, stored in
        proc_data_dict['projected_data_dict'].

        Calls
            - prepare_projected_data_plot if plot_proj_data (passed in the
             options_dict, metadata, or default_options) is True.
            - prepare_projected_1d_slices_plots if the data is TwoD and
             slice_idxs_1d_proj_plot was passed in options_dict, metadata, or
             default_options
        """
        plot_proj_data = self.get_param_value('plot_proj_data', True)
        select_split = self.get_param_value('select_split')
        fig_name_suffix = self.get_param_value('fig_name_suffix', '')
        title_suffix = self.get_param_value('title_suffix', '')
        TwoD = self.get_param_value('TwoD', False)
        slice_idxs_1d_proj_plot = self.get_param_value(
            'slice_idxs_1d_proj_plot', {})
        for qb_name, corr_data in self.proc_data_dict[
                'projected_data_dict'].items():
            slice_idxs_list = slice_idxs_1d_proj_plot.get(qb_name, [])
            fig_name = f'projected_plot_{qb_name}'
            title_suf = title_suffix
            if select_split is not None:
                param, idx = select_split[qb_name]
                # remove qb_name from param
                p = '_'.join([e for e in param.split('_') if e != qb_name])
                # create suffix
                suf = f'({p}, {str(np.round(idx, 3))})'
                # add suffix
                fig_name += f'_{suf}'
                title_suf = f'{suf}_{title_suf}' if \
                    len(title_suf) else suf
            if isinstance(corr_data, dict):
                # the most typical case, where we have data for 'pg', 'pe'
                # (and 'pf')
                for data_key, data in corr_data.items():
                    # data_key is usually 'pg', 'pe', 'pf'
                    fn = f'{fig_name}_{data_key}'
                    if not self.rotate:
                        data_label = ''
                        plot_name_suffix = data_key
                        plot_cal_points = False
                    else:
                        data_label = 'Data'
                        plot_name_suffix = ''
                        plot_cal_points = (
                            not self.get_param_value('TwoD', False))
                    data_axis_label = self.get_yaxis_label(qb_name,
                                                           data_key)
                    tf = f'{data_key}_{title_suf}' if \
                        len(title_suf) else data_key
                    if plot_proj_data:
                        # standard projected data plot
                        self.prepare_projected_data_plot(
                            fn, data, qb_name=qb_name,
                            data_label=data_label,
                            title_suffix=tf,
                            plot_name_suffix=plot_name_suffix,
                            fig_name_suffix=fig_name_suffix,
                            data_axis_label=data_axis_label,
                            plot_cal_points=plot_cal_points)
                    if TwoD and len(slice_idxs_list) > 0:
                        # plot slices of the 2D projected data
                        self.prepare_projected_1d_slices_plots(
                            fn, data, qb_name, slice_idxs_list,
                            title_suffix=tf, data_label=data_label,
                            fig_name_suffix=fig_name_suffix,
                            data_axis_label=data_axis_label)
            else:
                fig_name = 'projected_plot_' + qb_name
                if plot_proj_data:
                    # standard projected data plot
                    self.prepare_projected_data_plot(
                        fig_name, corr_data, qb_name=qb_name,
                        plot_cal_points=(not TwoD))
                if TwoD and len(slice_idxs_list) > 0:
                    # plot slices of the 2D projected data
                    self.prepare_projected_1d_slices_plots(
                        fig_name, qb_name, corr_data, slice_idxs_list)

    def prepare_projected_1d_slices_plots(self, fig_name, data, qb_name,
                                          slice_idxs_list, title_suffix='',
                                          **kw):
        """
        Prepares 1d plots of slices from a TwoD projected data plot.

        Args:
            fig_name (str): figure name
            data (array): 2D array of the projected data
            qb_name (str): qubit name
            slice_idxs_list (list): list of tuples of the form (idxs, axis)
                referring to slices of the 2D data array (see class docstring).
                Example: [('8:13', 'row'), (0, 'col')]
            title_suffix (str): suffix to add to the plot title which is created
                automatically in prepare_projected_data_plot
            **kw: passed to prepare_projected_data_plot
        """
        for slice_idxs in slice_idxs_list:
            idxs, axis, xvals, xlabel, xunit = self.get_1d_slice_params(
                qb_name, slice_idxs)
            for idx in idxs:
                data_slice = np.take_along_axis(
                    data, np.array([[idx]]), axis).flatten()
                plot_name_suffix = \
                    f'{"_row" if axis == 0 else "_col"}_{idx}'
                fn_slice = f'{fig_name}{plot_name_suffix}'
                ts_slice = f'{title_suffix}{plot_name_suffix}'
                self.prepare_projected_data_plot(
                    fn_slice, data_slice, qb_name=qb_name,
                    sweep_points=xvals,
                    title_suffix=ts_slice, TwoD=False,
                    plot_name_suffix=plot_name_suffix,
                    xlabel=xlabel, xunit=xunit,
                    plot_cal_points=axis == 0, **kw)

    def prepare_projected_data_plot(
            self, fig_name, data, qb_name, title_suffix='', sweep_points=None,
            plot_cal_points=True, plot_name_suffix='', fig_name_suffix='',
            data_label='Data', data_axis_label='', do_legend_data=True,
            do_legend_cal_states=True, TwoD=None, yrange=None,
            linestyle='none', xlabel=None, xunit=None):
        """
        Prepares one projected data plot, typically one of the keys in
        proc_data_dict['projected_data_dict'].

        Args:
            fig_name: string with name of the figure
            data: numpy array of data to plot, typically from
                proc_data_dict['projected_data_dict']
            qb_name: string with name of the qubit to which the data corresponds
            title_suffix: string with axis title suffix (without the
                underscore character '_' which is added by this method)
            sweep_points: numpy array of sweep points corresponding to the data
                If None, will be taken from proc_data_dict['sweep_points_dict']
            plot_cal_points: bool specifying whether to prepare separate plot
                dicts for the cal points (with individual colors taken from
                self.get_cal_state_color)
            plot_name_suffix: string with suffix for the key name under which
                the plot dict will be added to self.plots_dicts (should not
                contain the underscore character '_' which is added by this
                method)
            fig_name_suffix: string with figure title suffix (without the
                underscore character '_' which is added by this method)
            data_label: string with the legend label corresponding to the data
            data_axis_label: string with the yaxis label. If not specified, it
                will be taken from self.get_yaxis_label.
            do_legend_data: bool specifying whether to set to True the do_legend
                key of the plot dict corresponding to the data
            do_legend_cal_states: bool specifying whether to set to True the
                do_legend key of the plot dict corresponding to the cal points.
                Only has an effect if plot_cal_points is True.
            TwoD: bool specifying whether to prepare a plot dict for 2D (True)
                or 1D data (False).
            yrange: tuple/list of floats for the plot yrange
            linestyle: string with recognised matplotlib linestyle. If not set,
                only markers for the data points will be shows. Only has an
                effects if TwoD is False.
            xlabel: string with the x-axis label, typically corresponding to the
                hard/1d sweep parameter
            xunit: string with the x-axis unit, typically corresponding to the
                hard/1d sweep parameter
        """
        if len(fig_name_suffix):
            fig_name = f'{fig_name}_{fig_name_suffix}'

        if data_axis_label == '':
            data_axis_label = self.get_yaxis_label(qb_name=qb_name)
        plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
        plotsize = (plotsize[0], plotsize[0]/1.25)

        if sweep_points is None:
            sweep_points = self.proc_data_dict['sweep_points_dict'][qb_name][
                'sweep_points']
        plot_names_cal = []
        if plot_cal_points and self.num_cal_points != 0 and \
                len(self.cal_states_dict[qb_name]):
            # the cal points are part of the dataset and we want to indicate
            # them in the plot with their own colour
            # We need to check both num_cal_points and cal_states dict because
            # the former counts how many calibration segments were used in the
            # experiment. So it will be nonzero even if no calibration point
            # information was provided (reflected by
            # self.cal_states_dict = {qbn: {} for qbn in self.qb_names})
            yvals = data[:-self.num_cal_points]
            xvals = sweep_points[:-self.num_cal_points]
            # plot cal points
            for i, cal_pts_idxs in enumerate(
                    self.cal_states_dict[qb_name].values()):
                plot_dict_name_cal = fig_name + '_' + \
                                     list(self.cal_states_dict[qb_name])[i] + '_' + \
                                     plot_name_suffix
                plot_names_cal += [plot_dict_name_cal]
                self.plot_dicts[plot_dict_name_cal] = {
                    'fig_id': fig_name,
                    'plotfn': self.plot_line,
                    'plotsize': plotsize,
                    'xvals': sweep_points[cal_pts_idxs],
                    'yvals': data[cal_pts_idxs],
                    'setlabel': list(self.cal_states_dict[qb_name])[i],
                    'do_legend': do_legend_cal_states,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'linestyle': 'none',
                    'line_kws': {'color': self.get_cal_state_color(
                        list(self.cal_states_dict[qb_name])[i])},
                    'yrange': yrange,
                }

                self.plot_dicts[plot_dict_name_cal+'_line'] = {
                    'fig_id': fig_name,
                    'plotsize': plotsize,
                    'plotfn': self.plot_hlines,
                    'y': np.mean(data[cal_pts_idxs]),
                    'xmin': sweep_points[0],
                    'xmax': sweep_points[-1],
                    'colors': 'gray'}

        else:
            yvals = data
            xvals = sweep_points
        title = (self.raw_data_dict['timestamp'] + ' ' +
                 self.raw_data_dict['measurementstring'])
        title += '\n' + f'{qb_name}_{title_suffix}' if len(title_suffix) else \
            ' ' + qb_name

        plot_dict_name = f'{fig_name}_{plot_name_suffix}'
        xl, xu = self.get_xaxis_label_unit(qb_name)
        if xlabel is None:
            xlabel = xl
        if xunit is None:
            xunit = xu

        prep_1d_plot = True
        if TwoD is None:
            TwoD = self.get_param_value('TwoD', default_value=False)
        if TwoD:
            sp2dd = self.proc_data_dict['sweep_points_2D_dict'][qb_name]
            if len(sp2dd) >= 1 and len(sp2dd[list(sp2dd)[0]]) > 1:
                # Only prepare 2D plots when there is more than one soft
                # sweep points. When there is only one soft sweep point
                # we want to do 1D plots which are more meaningful
                prep_1d_plot = False
                for pn, ssp in sp2dd.items():
                    ylabel, yunit = self.get_soft_sweep_label_unit(pn)
                    self.plot_dicts[f'{plot_dict_name}_{pn}'] = {
                        'plotfn': self.plot_colorxy,
                        'fig_id': fig_name + '_' + pn,
                        'xvals': xvals,
                        'yvals': ssp,
                        'zvals': yvals,
                        'xlabel': xlabel,
                        'xunit': xunit,
                        'ylabel': ylabel,
                        'yunit': yunit,
                        'yrange': yrange,
                        'zrange': self.get_param_value('zrange', None),
                        'title': title,
                        'clabel': data_axis_label}

        if prep_1d_plot:
            if len(yvals.shape) > 1 and yvals.shape[0] == 1:
                # only one soft sweep point: prepare 1D plot which is
                # more meaningful
                yvals = np.squeeze(yvals, axis=0)
            self.plot_dicts[plot_dict_name] = {
                'plotfn': self.plot_line,
                'fig_id': fig_name,
                'plotsize': plotsize,
                'xvals': xvals,
                'xlabel': xlabel,
                'xunit': xunit,
                'yvals': yvals,
                'ylabel': data_axis_label,
                'yunit': '',
                'yrange': yrange,
                'setlabel': data_label,
                'title': title,
                'linestyle': linestyle,
                'do_legend': do_legend_data and len(data_label),
                'legend_bbox_to_anchor': (1, 0.5),
                'legend_pos': 'center left'}

        # add plot_params to each plot dict
        plot_params = self.get_param_value('plot_params', default_value={})
        for plt_name in self.plot_dicts:
            self.plot_dicts[plt_name].update(plot_params)

        if len(plot_names_cal) > 0:
            if do_legend_data and not do_legend_cal_states:
                for plot_name in plot_names_cal:
                    plot_dict_cal = self.plot_dicts.pop(plot_name)
                    self.plot_dicts[plot_name] = plot_dict_cal

    def _plot_1d_slices_of_2d_data(self, plot_type, slice_idxs_1d_plot):
        """
        Prepares figures for 1D slices of 2D data, and plots them.

        Args:
            plot_type (str): can be either "raw" or "proj", indicating whether
                to plot 1D slices of raw data or projected data
            slice_idxs_1d_plot (dict): slices indices of the 2D data. See
                class docstring for more details.
        """
        if plot_type == 'raw':
            key1, key2 = 'slice_idxs_1d_raw_plot', 'plot_raw_data'
            prepare_plots_func = self.prepare_raw_data_plots
        elif plot_type == 'proj':
            key1, key2 = 'slice_idxs_1d_proj_plot', 'plot_proj_data'
            prepare_plots_func = self.prepare_projected_data_plots
        else:
            raise ValueError(f'Unrecognized plot_type "{plot_type}." Please '
                             f'use either "raw" or "proj."')
        # Store self.default_options
        default_options = deepcopy(self.default_options)
        # Update self.default_options to ensure that only 1D slice plots
        # are prepared. It is important to update this attribute instead of
        # overwriting it, since the TwoD flag is added there by BaseDataAnalysis
        self.default_options.update({key1: slice_idxs_1d_plot, key2: False})
        # Store self.plot_dicts (deepcopy not allowed here because the plot
        # dicts contain functions)
        plot_dicts = {}
        plot_dicts.update(self.plot_dicts)
        # Remove existing entries from self.plot_dicts such that we only plot
        # the 1D slices
        self.plot_dicts = {}
        # Prepare the 1D slice plots
        prepare_plots_func()
        # Plot 1D slices
        self.plot()
        # Save the figures
        self.save_figures(close_figs=self.options_dict.get('close_figs', False))
        # Update the self.plot_dicts with what was there originally
        self.plot_dicts.update(plot_dicts)
        # Reset the self.default_options
        self.default_options = default_options

    def plot_raw_1d_slices(self, slice_idxs_1d_raw_plot):
        """
        Helper function to prepare and plot figures of 1D slices of
        2D raw data.

        'projected_data_dict' must exist in self.proc_data_dict.

        Args:
            slice_idxs_1d_raw_plot (dict): slices indices of the 2D data. See
                class docstring for more details.
        """
        self._plot_1d_slices_of_2d_data('raw', slice_idxs_1d_raw_plot)

    def plot_projected_1d_slices(self, slice_idxs_1d_proj_plot):
        """
        Helper function to prepare and plot figures of 1D slices of
        2D projected data.

        'meas_results_per_qb_raw' or 'meas_results_per_qb' must exist
        in self.proc_data_dict.

        Args:
            slice_idxs_1d_proj_plot (dict): slices indices of the 2D data. See
                class docstring for more details.
        """
        self._plot_1d_slices_of_2d_data('proj', slice_idxs_1d_proj_plot)

    def get_1d_slice_params(self, qb_name, slice_idxs):
        """
        Translates the information in slice_idxs into the relevant plot
        parameters used by the functions that prepare plots.

        Args:
            qb_name (str): qubit name
            slice_idxs (tuple): of the form (idxs, axis) referring to slices
                of a 2D data array (see class docstring).
                Examples: ('8:13', 'row'), (0, 'col')

        Returns:
            idx (list): list of indices of the 2D data array
            axis (int): axis along which to take indices (0 for row, 1 for col)
            xvals (array): x-axis values
            xlabel (str): x-axis label
            xunit (str): x-axis unit
        """
        axis = 0 if slice_idxs[1] == 'row' else 1
        if axis == 0:
            xvals = self.proc_data_dict[
                'sweep_points_dict'][qb_name][
                'sweep_points']
            yvals = list(
                self.proc_data_dict[
                    'sweep_points_2D_dict'][
                    qb_name].values())[0]
            xlabel, xunit = None, None
        else:
            param_name = list(self.proc_data_dict[
                                  'sweep_points_2D_dict'][qb_name])[0]
            xvals = list(
                self.proc_data_dict[
                    'sweep_points_2D_dict'][
                    qb_name].values())[0]
            yvals = self.proc_data_dict[
                'sweep_points_dict'][qb_name][
                'sweep_points']
            xlabel, xunit = \
                self.get_soft_sweep_label_unit(
                    param_name)

        idxs = slice_idxs[0]
        if isinstance(idxs, str):
            if idxs == ':':
                # take all slices along axis
                idxs = np.arange(len(yvals))
            else:
                # idxs of the form 'int:int' or ':'
                idxs = np.arange(int(idxs.split(':')[0]),
                                 int(idxs.split(':')[-1]))
        else:
            idxs = [idxs]

        return idxs, axis, xvals, xlabel, xunit

    def get_first_sweep_param(self, qbn=None, dimension=0):
        """
        Get properties of the first sweep param in the given dimension
        (potentially for the given qubit).
        :param qbn: (str) qubit name. If None, all sweep params are considered.
        :param dimension: (float, default: 0) sweep dimension to be considered.
        :return: a 3-tuple of label, unit, and array of values
        """
        if not hasattr(self, 'mospm'):
            return None

        if qbn is None:
            param_name = [p for v in self.mospm.values() for p in v
                          if self.sp.find_parameter(p) == 1]
        else:
            param_name = [p for p in self.mospm[qbn]
                          if self.sp.find_parameter(p)]
        if not len(param_name):
            return None

        param_name = param_name[0]
        label = self.sp.get_sweep_params_property(
            'label', dimension=dimension, param_names=param_name)
        unit = self.sp.get_sweep_params_property(
            'unit', dimension=dimension, param_names=param_name)
        vals = self.sp.get_sweep_params_property(
            'values', dimension=dimension, param_names=param_name)
        return label, unit, vals

    @staticmethod
    def get_qbs_from_task_list(task_list):
    # TODO: there is a more elaborate method find_qubits_in_tasks in MultiTaskingExperiment.
    #  We could consider moving it to a place where it can be used from both measurement and analysis classes and
    #  then remove this function here
        all_qubits = set()
        for task in task_list:
            all_qubits.update([v for k,v in task.items() if 'qb' in k])
        return list(all_qubits)


class MultiQubit_HistogramAnalysis(MultiQubit_TimeDomain_Analysis):
    def extract_data(self):
        self.default_options['rotation'] = False
        super().extract_data()

    def get_channel_map(self):
        # FIXME: delete this method once it has been tested that
        #  - the super method handles single-qubit histograms correctly
        #  - multi-qubit histograms are analyzed correctly
        if len(self.qb_names) > 1:
            log.warning('Analysis of multi-qubit histograms has not been '
                        'tested yet.')
            return super().get_channel_map()
        self.channel_map = {
            self.qb_names[0]: self.raw_data_dict['value_names']}

    def process_data(self):
        super().process_data()
        import re
        hsp = self.raw_data_dict['hard_sweep_points']
        for qbn in self.qb_names:
            hist = self.proc_data_dict['meas_results_per_qb'][qbn]
            bins = {}
            for vn in hist:
                res = re.findall(r'_hist_(\([0-9, ]*\))', vn)
                if len(res) == 1:
                    bins[eval(res[0])] = vn
            nr_bins = [max([b[i] for b in bins]) + 1 for i in
                       range(len(list(bins)[0]))]
            all_bins = list(itertools.product(*[range(i) for i in nr_bins]))
            missing = [b for b in all_bins if b not in bins]
            if len(missing):
                log.warning(f'Bin data missing for {qbn}: {missing}')
            hist_proc = dict()
            for i in range(len(hsp)):
                hist_proc[i] = np.zeros(nr_bins)
                for b, vn in bins.items():
                    hist_proc[i][b] = hist[vn][i]
            self.proc_data_dict.setdefault('histogram_per_qb', {})
            self.proc_data_dict['histogram_per_qb'][qbn] = hist_proc
            self.proc_data_dict.setdefault('nr_bins', {})
            self.proc_data_dict['nr_bins'][qbn] = nr_bins

    def prepare_plots(self):
        for qbn in self.qb_names:
            self.prepare_hist_plot(qbn)

    def prepare_hist_plot(self, qb_name):
        nr_bins = self.proc_data_dict['nr_bins'][qb_name]
        if len(nr_bins) != 2:
            return
        hists = self.proc_data_dict['histogram_per_qb'][qb_name]
        for seg, hist in hists.items():
            plot_name = f'histogram_{qb_name}_{seg}'
            self.plot_dicts[plot_name] = {
                'plotfn': self.plot_colorxy,
                'fig_id': plot_name,
                'xvals': list(range(nr_bins[0])),
                'yvals': list(range(nr_bins[0])),
                'zvals': hist,
                'xlabel': 'bin index dim. 0',
                'xunit': '',
                'ylabel': 'bin index dim. 1',
                'yunit': '',
                'zrange': self.get_param_value('zrange', None),
                'title': f'Histogram {qb_name} Segment {seg}',
                'clabel': 'counts'}


class StateTomographyAnalysis(ba.BaseDataAnalysis):
    """
    Analyses the results of the state tomography experiment and calculates
    the corresponding quantum state.

    Possible options that can be passed in the options_dict parameter:
        cal_points: A data structure specifying the indices of the calibration
                    points. See the AveragedTimedomainAnalysis for format.
                    The calibration points need to be in the same order as the
                    used basis for the result.
        data_type: 'averaged' or 'singleshot'. For singleshot data each
                   measurement outcome is saved and arbitrary order correlations
                   between the states can be calculated.
        meas_operators: (optional) A list of qutip operators or numpy 2d arrays.
                        This overrides the measurement operators otherwise
                        found from the calibration points.
        covar_matrix: (optional) The covariance matrix of the measurement
                      operators as a 2d numpy array. Overrides the one found
                      from the calibration points.
        use_covariance_matrix (bool): Flag to define whether to use the
            covariance matrix
        basis_rots_str: A list of standard PycQED pulse names that were
                             applied to qubits before measurement
        basis_rots: As an alternative to single_qubit_pulses, the basis
                    rotations applied to the system as qutip operators or numpy
                    matrices can be given.
        mle: True/False, whether to do maximum likelihood fit. If False, only
             least squares fit will be done, which could give negative
             eigenvalues for the density matrix.
        imle: True/False, whether to do iterative maximum likelihood fit. If
             True, it takes preference over maximum likelihood method. Otherwise
             least squares fit will be done, then 'mle' option will be checked.
        pauli_raw: True/False, extracts Pauli expected values from a measurement
             without assignment correction based on calibration data. If True,
             takes preference over other methods except pauli_corr.
        pauli_values: True/False, extracts Pauli expected values from a
             measurement with assignment correction based on calibration data.
             If True, takes preference over other methods.
        iterations (optional): maximum number of iterations allowed in imle.
             Tomographies with more qubits require more iterations to converge.
        tolerance (optional): minimum change across iterations allowed in imle.
             The iteration will stop if it goes under this value. Tomographies
             with more qubits require smaller tolerance to converge.
        rho_target (optional): A qutip density matrix that the result will be
                               compared to when calculating fidelity.
    """
    def __init__(self, *args, **kwargs):
        auto = kwargs.pop('auto', True)
        super().__init__(*args, **kwargs)
        kwargs['auto'] = auto
        self.single_timestamp = True
        self.params_dict = {'exp_metadata': 'exp_metadata'}
        self.numeric_params = []
        self.data_type = self.get_param_value('data_type')
        if self.data_type == 'averaged':
            self.base_analysis = AveragedTimedomainAnalysis(*args, **kwargs)
        elif self.data_type == 'singleshot':
            self.base_analysis = roa.MultiQubit_SingleShot_Analysis(
                *args, **kwargs)
        else:
            raise KeyError("Invalid tomography data mode: '" + self.data_type +
                           "'. Valid modes are 'averaged' and 'singleshot'.")
        if kwargs.get('auto', True):
            self.run_analysis()

    def process_data(self):
        tomography_qubits = self.get_param_value('tomography_qubits')
        data, Fs, Omega = self.base_analysis.measurement_operators_and_results(
                              tomography_qubits)
        data_filter = self.get_param_value('data_filter')
        if data_filter is not None:
            data = data_filter(data.T).T

        data = data.T
        for i, v in enumerate(data):
            data[i] = v / v.sum()
        data = data.T

        Fs = self.get_param_value('meas_operators', Fs)
        Fs = [qtp.Qobj(F) for F in Fs]
        d = Fs[0].shape[0]
        self.proc_data_dict['d'] = d
        Omega = self.get_param_value('covar_matrix', Omega)
        if Omega is None:
            Omega = np.diag(np.ones(len(Fs)))
        elif len(Omega.shape) == 1:
            Omega = np.diag(Omega)

        metadata = self.raw_data_dict.get('exp_metadata',
                                          self.options_dict.get(
                                              'exp_metadata', {}))
        if metadata is None:
            metadata = {}
        self.raw_data_dict['exp_metadata'] = metadata
        basis_rots_str = metadata.get('basis_rots_str', None)
        basis_rots_str = self.options_dict.get('basis_rots_str', basis_rots_str)
        if basis_rots_str is not None:
            nr_qubits = int(np.round(np.log2(d)))
            pulse_list = list(itertools.product(basis_rots_str,
                                                repeat=nr_qubits))
            rotations = tomo.standard_qubit_pulses_to_rotations(pulse_list)
        else:
            rotations = metadata.get('basis_rots', None)
            rotations = self.options_dict.get('basis_rots', rotations)
            if rotations is None:
                raise KeyError("Either 'basis_rots_str' or 'basis_rots' "
                               "parameter must be passed in the options "
                               "dictionary or in the experimental metadata.")
        rotations = [qtp.Qobj(U) for U in rotations]

        all_Fs = tomo.rotated_measurement_operators(rotations, Fs)
        all_Fs = [all_Fs[i][j]
                  for j in range(len(all_Fs[0]))
                  for i in range(len(all_Fs))]
        all_mus = np.array(list(itertools.chain(*data.T)))
        all_Omegas = sp.linalg.block_diag(*[Omega] * len(data[0]))


        self.proc_data_dict['meas_operators'] = all_Fs
        self.proc_data_dict['covar_matrix'] = all_Omegas
        self.proc_data_dict['meas_results'] = all_mus

        if self.get_param_value('pauli_values', False):
            rho_pauli = tomo.pauli_values_tomography(all_mus,Fs,basis_rots_str)
            self.proc_data_dict['rho_raw'] = rho_pauli
            self.proc_data_dict['rho'] = rho_pauli
        elif self.get_param_value('pauli_raw', False):
            pauli_raw = self.generate_raw_pauli_set()
            rho_raw = tomo.pauli_set_to_density_matrix(pauli_raw)
            self.proc_data_dict['rho_raw'] = rho_raw
            self.proc_data_dict['rho'] = rho_raw
        elif self.get_param_value('imle', False):
            it = metadata.get('iterations', None)
            it = self.get_param_value('iterations', it)
            tol = metadata.get('tolerance', None)
            tol = self.get_param_value('tolerance', tol)
            rho_imle = tomo.imle_tomography(
                all_mus, all_Fs, it, tol)
            self.proc_data_dict['rho_imle'] = rho_imle
            self.proc_data_dict['rho'] = rho_imle
        else:
            rho_ls = tomo.least_squares_tomography(
                all_mus, all_Fs,
                all_Omegas if self.get_param_value('use_covariance_matrix', False)
                else None )
            self.proc_data_dict['rho_ls'] = rho_ls
            self.proc_data_dict['rho'] = rho_ls
            if self.get_param_value('mle', False):
                rho_mle = tomo.mle_tomography(
                    all_mus, all_Fs,
                    all_Omegas if self.get_param_value('use_covariance_matrix', False) else None,
                    rho_guess=rho_ls)
                self.proc_data_dict['rho_mle'] = rho_mle
                self.proc_data_dict['rho'] = rho_mle

        rho = self.proc_data_dict['rho']
        self.proc_data_dict['purity'] = (rho * rho).tr().real

        rho_target = metadata.get('rho_target', None)
        rho_target = self.get_param_value('rho_target', rho_target)
        if rho_target is not None:
            self.proc_data_dict['fidelity'] = tomo.fidelity(rho, rho_target)
        if d == 4:
            self.proc_data_dict['concurrence'] = tomo.concurrence(rho)
        else:
            self.proc_data_dict['concurrence'] = 0

    def prepare_plots(self):
        self.prepare_density_matrix_plot()
        d = self.proc_data_dict['d']
        if 2 ** (d.bit_length() - 1) == d:
            # dimension is power of two, plot expectation values of pauli
            # operators
            self.prepare_pauli_basis_plot()

    def prepare_density_matrix_plot(self):
        self.tight_fig = self.get_param_value('tight_fig', False)
        rho_target = self.raw_data_dict['exp_metadata'].get('rho_target', None)
        rho_target = self.get_param_value('rho_target', rho_target)
        d = self.proc_data_dict['d']
        xtick_labels = self.get_param_value('rho_ticklabels', None)
        ytick_labels = self.get_param_value('rho_ticklabels', None)
        if 2 ** (d.bit_length() - 1) == d:
            nr_qubits = d.bit_length() - 1
            fmt_string = '{{:0{}b}}'.format(nr_qubits)
            labels = [fmt_string.format(i) for i in range(2 ** nr_qubits)]
            if xtick_labels is None:
                xtick_labels = ['$|' + lbl + r'\rangle$' for lbl in labels]
            if ytick_labels is None:
                ytick_labels = [r'$\langle' + lbl + '|$' for lbl in labels]
        color = (0.5 * np.angle(self.proc_data_dict['rho'].full()) / np.pi) % 1.
        cmap = self.get_param_value('rho_colormap', self.default_phase_cmap())
        if self.get_param_value('pauli_raw', False):
            title = 'Density matrix reconstructed from the Pauli (raw) set\n'
        elif self.get_param_value('pauli_values', False):
            title = 'Density matrix reconstructed from the Pauli set\n'
        elif self.get_param_value('mle', False):
            title = 'Maximum likelihood fit of the density matrix\n'
        elif self.get_param_value('it_mle', False):
            title = 'Iterative maximum likelihood fit of the density matrix\n'
        else:
            title = 'Least squares fit of the density matrix\n'
        empty_artist = mpl.patches.Rectangle((0, 0), 0, 0, visible=False)
        legend_entries = [(empty_artist,
                           r'Purity, $Tr(\rho^2) = {:.1f}\%$'.format(
                               100 * self.proc_data_dict['purity']))]
        if rho_target is not None:
            legend_entries += [
                (empty_artist, r'Fidelity, $F = {:.1f}\%$'.format(
                    100 * self.proc_data_dict['fidelity']))]
        if d == 4:
            legend_entries += [
                (empty_artist, r'Concurrence, $C = {:.2f}$'.format(
                    self.proc_data_dict['concurrence']))]
        meas_string = self.base_analysis.\
            raw_data_dict['measurementstring']
        if isinstance(meas_string, list):
            if len(meas_string) > 1:
                meas_string = meas_string[0] + ' to ' + meas_string[-1]
            else:
                meas_string = meas_string[0]
        self.plot_dicts['density_matrix'] = {
            'plotfn': self.plot_bar3D,
            '3d': True,
            '3d_azim': -35,
            '3d_elev': 35,
            'xvals': np.arange(d),
            'yvals': np.arange(d),
            'zvals': np.abs(self.proc_data_dict['rho'].full()),
            'zrange': (0, 1),
            'color': color,
            'colormap': cmap,
            'bar_widthx': 0.5,
            'bar_widthy': 0.5,
            'xtick_loc': np.arange(d),
            'xtick_labels': xtick_labels,
            'ytick_loc': np.arange(d),
            'ytick_labels': ytick_labels,
            'ctick_loc': np.linspace(0, 1, 5),
            'ctick_labels': ['$0$', r'$\frac{1}{2}\pi$', r'$\pi$',
                             r'$\frac{3}{2}\pi$', r'$2\pi$'],
            'clabel': 'Phase (rad)',
            'title': (title + self.raw_data_dict['timestamp'] + ' ' +
                      meas_string),
            'do_legend': True,
            'legend_entries': legend_entries,
            'legend_kws': dict(loc='upper left', bbox_to_anchor=(0, 0.94))
        }

        if rho_target is not None:
            rho_target = qtp.Qobj(rho_target)
            if rho_target.type == 'ket':
                rho_target = rho_target * rho_target.dag()
            elif rho_target.type == 'bra':
                rho_target = rho_target.dag() * rho_target
            self.plot_dicts['density_matrix_target'] = {
                'plotfn': self.plot_bar3D,
                '3d': True,
                '3d_azim': -35,
                '3d_elev': 35,
                'xvals': np.arange(d),
                'yvals': np.arange(d),
                'zvals': np.abs(rho_target.full()),
                'zrange': (0, 1),
                'color': (0.5 * np.angle(rho_target.full()) / np.pi) % 1.,
                'colormap': cmap,
                'bar_widthx': 0.5,
                'bar_widthy': 0.5,
                'xtick_loc': np.arange(d),
                'xtick_labels': xtick_labels,
                'ytick_loc': np.arange(d),
                'ytick_labels': ytick_labels,
                'ctick_loc': np.linspace(0, 1, 5),
                'ctick_labels': ['$0$', r'$\frac{1}{2}\pi$', r'$\pi$',
                                 r'$\frac{3}{2}\pi$', r'$2\pi$'],
                'clabel': 'Phase (rad)',
                'title': ('Target density matrix\n' +
                          self.raw_data_dict['timestamp'] + ' ' +
                          meas_string),
                'bar_kws': dict(zorder=1),
            }

    def generate_raw_pauli_set(self):
        nr_qubits = self.proc_data_dict['d'].bit_length() - 1
        pauli_raw_values = []
        for op in tomo.generate_pauli_set(nr_qubits)[1]:
            nr_terms = 0
            sum_terms = 0.
            for meas_op, meas_res in zip(self.proc_data_dict['meas_operators'],
                                         self.proc_data_dict['meas_results']):
                trace = (meas_op*op).tr().real
                clss = int(trace*2)
                if clss < 0:
                    sum_terms -= meas_res
                    nr_terms += 1
                elif clss > 0:
                    sum_terms += meas_res
                    nr_terms += 1
            pauli_raw_values.append(2**nr_qubits*sum_terms/nr_terms)
        return pauli_raw_values

    def generate_corr_pauli_set(self,Fs,rotations):
        nr_qubits = self.proc_data_dict['d'].bit_length() - 1

        Fs_corr = []
        assign_corr = []
        for i,F in enumerate(Fs):
            new_op = np.zeros(2**nr_qubits)
            new_op[i] = 1
            Fs_corr.append(qtp.Qobj(np.diag(new_op)))
            assign_corr.append(np.diag(F.full()))
        pauli_Fs = tomo.rotated_measurement_operators(rotations, Fs_corr)
        pauli_Fs = list(itertools.chain(*np.array(pauli_Fs, dtype=object).T))

        mus = self.proc_data_dict['meas_results']
        pauli_mus = np.reshape(mus,[-1,2**nr_qubits])
        for i,raw_mus in enumerate(pauli_mus):
            pauli_mus[i] = np.matmul(np.linalg.inv(assign_corr),np.array(raw_mus))
        pauli_mus = pauli_mus.flatten()

        pauli_values = []
        for op in tomo.generate_pauli_set(nr_qubits)[1]:
            nr_terms = 0
            sum_terms = 0.
            for meas_op, meas_res in zip(pauli_Fs,pauli_mus):
                trace = (meas_op*op).tr().real
                clss = int(trace*2)
                if clss < 0:
                    sum_terms -= meas_res
                    nr_terms += 1
                elif clss > 0:
                    sum_terms += meas_res
                    nr_terms += 1
            pauli_values.append(2**nr_qubits*sum_terms/nr_terms)

        return pauli_values

    def prepare_pauli_basis_plot(self):
        yexp = tomo.density_matrix_to_pauli_basis(self.proc_data_dict['rho'])
        nr_qubits = self.proc_data_dict['d'].bit_length() - 1
        labels = list(itertools.product(*[['I', 'X', 'Y', 'Z']]*nr_qubits))
        labels = [''.join(label_list) for label_list in labels]
        if nr_qubits == 1:
            order = [1, 2, 3]
        elif nr_qubits == 2:
            order = [1, 2, 3, 4, 8, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        elif nr_qubits == 3:
            order = [1, 2, 3, 4, 8, 12, 16, 32, 48] + \
                    [5, 6, 7, 9, 10, 11, 13, 14, 15] + \
                    [17, 18, 19, 33, 34, 35, 49, 50, 51] + \
                    [20, 24, 28, 36, 40, 44, 52, 56, 60] + \
                    [21, 22, 23, 25, 26, 27, 29, 30, 31] + \
                    [37, 38, 39, 41, 42, 43, 45, 46, 47] + \
                    [53, 54, 55, 57, 58, 59, 61, 62, 63]
        else:
            order = np.arange(4**nr_qubits)[1:]
        if self.get_param_value('pauli_raw', False):
            fit_type = 'raw counts'
        elif self.get_param_value('pauli_values', False):
            fit_type = 'corrected counts'
        elif self.get_param_value('mle', False):
            fit_type = 'maximum likelihood estimation'
        elif self.get_param_value('imle', False):
            fit_type = 'iterative maximum likelihood estimation'
        else:
            fit_type = 'least squares fit'
        meas_string = self.base_analysis. \
            raw_data_dict['measurementstring']
        if np.ndim(meas_string) > 0:
            if len(meas_string) > 1:
                meas_string = meas_string[0] + ' to ' + meas_string[-1]
            else:
                meas_string = meas_string[0]
        self.plot_dicts['pauli_basis'] = {
            'plotfn': self.plot_bar,
            'xcenters': np.arange(len(order)),
            'xwidth': 0.4,
            'xrange': (-1, len(order)),
            'yvals': np.array(yexp)[order],
            'xlabel': r'Pauli operator, $\hat{O}$',
            'ylabel': r'Expectation value, $\mathrm{Tr}(\hat{O} \hat{\rho})$',
            'title': 'Pauli operators, ' + fit_type + '\n' +
                      self.raw_data_dict['timestamp'] + ' ' + meas_string,
            'yrange': (-1.1, 1.1),
            'xtick_loc': np.arange(4**nr_qubits - 1),
            'xtick_rotation': 90,
            'xtick_labels': np.array(labels)[order],
            'bar_kws': dict(zorder=10),
            'setlabel': 'Fit to experiment',
            'do_legend': True
        }
        if nr_qubits > 2:
            self.plot_dicts['pauli_basis']['plotsize'] = (10, 5)

        rho_target = self.raw_data_dict['exp_metadata'].get('rho_target', None)
        rho_target = self.get_param_value('rho_target', rho_target)
        if rho_target is not None:
            rho_target = qtp.Qobj(rho_target)
            ytar = tomo.density_matrix_to_pauli_basis(rho_target)
            self.plot_dicts['pauli_basis_target'] = {
                'plotfn': self.plot_bar,
                'ax_id': 'pauli_basis',
                'xcenters': np.arange(len(order)),
                'xwidth': 0.8,
                'yvals': np.array(ytar)[order],
                'xtick_loc': np.arange(len(order)),
                'xtick_labels': np.array(labels)[order],
                'bar_kws': dict(color='0.8', zorder=0),
                'setlabel': 'Target values',
                'do_legend': True
            }

        purity_str = r'Purity, $Tr(\rho^2) = {:.1f}\%$'.format(
            100 * self.proc_data_dict['purity'])
        if rho_target is not None:
            fidelity_str = '\n' + r'Fidelity, $F = {:.1f}\%$'.format(
                100 * self.proc_data_dict['fidelity'])
        else:
            fidelity_str = ''
        if self.proc_data_dict['d'] == 4:
            concurrence_str = '\n' + r'Concurrence, $C = {:.1f}\%$'.format(
                100 * self.proc_data_dict['concurrence'])
        else:
            concurrence_str = ''
        self.plot_dicts['pauli_info_labels'] = {
            'ax_id': 'pauli_basis',
            'plotfn': self.plot_line,
            'xvals': [0],
            'yvals': [0],
            'line_kws': {'alpha': 0},
            'setlabel': purity_str + fidelity_str,
            'do_legend': True
        }

    def default_phase_cmap(self):
        cols = np.array(((41, 39, 231), (61, 130, 163), (208, 170, 39),
                         (209, 126, 4), (181, 28, 20), (238, 76, 152),
                         (251, 130, 242), (162, 112, 251))) / 255
        n = len(cols)
        cdict = {
            'red': [[i/n, cols[i%n][0], cols[i%n][0]] for i in range(n+1)],
            'green': [[i/n, cols[i%n][1], cols[i%n][1]] for i in range(n+1)],
            'blue': [[i/n, cols[i%n][2], cols[i%n][2]] for i in range(n+1)],
        }

        return mpl.colors.LinearSegmentedColormap('DMDefault', cdict)


class FluxAmplitudeSweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def __init__(self, qb_names, *args, **kwargs):
        self.mask_freq = kwargs.pop('mask_freq', None)
        self.mask_amp = kwargs.pop('mask_amp', None)

        super().__init__(qb_names, *args, **kwargs)

    def extract_data(self):
        self.default_options['rotation_type'] = 'global_PCA'
        super().extract_data()

    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_sp = {qb: len(pdd['sweep_points_dict'][qb]['sweep_points'])
                 for qb in self.qb_names}
        nr_sp2d = {qb: len(list(pdd['sweep_points_2D_dict'][qb].values())[0])
                           for qb in self.qb_names}
        nr_cp = self.num_cal_points

        # make matrix out of vector
        data_reshaped = {qb: np.reshape(deepcopy(
            pdd['data_to_fit'][qb]).T.flatten(), (nr_sp[qb], nr_sp2d[qb]))
                         for qb in self.qb_names}
        pdd['data_reshaped'] = data_reshaped

        # remove calibration points from data to fit
        data_no_cp = {qb: np.array([pdd['data_reshaped'][qb][i, :]
                                    for i in range(nr_sp[qb]-nr_cp)])
            for qb in self.qb_names}

        # apply mask
        for qb in self.qb_names:
            if self.mask_freq is None:
                self.mask_freq = [True]*nr_sp2d[qb] # by default, no point is masked
            if self.mask_amp is None:
                self.mask_amp = [True]*(nr_sp[qb]-nr_cp)

        pdd['freqs_masked'] = {}
        pdd['amps_masked'] = {}
        pdd['data_masked'] = {}
        for qb in self.qb_names:
            sp_param = [k for k in self.mospm[qb] if 'freq' in k][0]
            pdd['freqs_masked'][qb] = \
                pdd['sweep_points_2D_dict'][qb][sp_param][self.mask_freq]
            pdd['amps_masked'][qb] = \
                pdd['sweep_points_dict'][qb]['sweep_points'][
                :-self.num_cal_points][self.mask_amp]
            data_masked = data_no_cp[qb][self.mask_amp,:]
            pdd['data_masked'][qb] = data_masked[:, self.mask_freq]

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()

        # Gaussian fit of amplitude slices
        gauss_mod = fit_mods.GaussianModel_v2()
        for qb in self.qb_names:
            for i in range(len(pdd['amps_masked'][qb])):
                data = pdd['data_masked'][qb][i,:]
                self.fit_dicts[f'gauss_fit_{qb}_{i}'] = {
                    'model': gauss_mod,
                    'fit_xvals': {'x': pdd['freqs_masked'][qb]},
                    'fit_yvals': {'data': data}
                    }

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['gauss_center'] = {}
        pdd['gauss_center_err'] = {}
        pdd['filtered_center'] = {}
        pdd['filtered_amps'] = {}

        for qb in self.qb_names:
            pdd['gauss_center'][qb] = np.array([
                self.fit_res[f'gauss_fit_{qb}_{i}'].best_values['center']
                for i in range(len(pdd['amps_masked'][qb]))])
            pdd['gauss_center_err'][qb] = np.array([
                self.fit_res[f'gauss_fit_{qb}_{i}'].params['center'].stderr
                for i in range(len(pdd['amps_masked'][qb]))])

            # filter out points with stderr > 1e6 Hz
            pdd['filtered_center'][qb] = np.array([])
            pdd['filtered_amps'][qb] = np.array([])
            for i, stderr in enumerate(pdd['gauss_center_err'][qb]):
                try:
                    if stderr < 1e6:
                        pdd['filtered_center'][qb] = \
                            np.append(pdd['filtered_center'][qb],
                                  pdd['gauss_center'][qb][i])
                        pdd['filtered_amps'][qb] = \
                            np.append(pdd['filtered_amps'][qb],
                            pdd['sweep_points_dict'][qb]\
                            ['sweep_points'][:-self.num_cal_points][i])
                except:
                    continue

            # if gaussian fitting does not work (i.e. all points were filtered
            # out above) use max value of data to get an estimate of freq
            if len(pdd['filtered_amps'][qb]) == 0:
                for qb in self.qb_names:
                    freqs = np.array([])
                    for i in range(pdd['data_masked'][qb].shape[0]):
                        freqs = np.append(freqs, pdd['freqs_masked'][qb]\
                            [np.argmax(pdd['data_masked'][qb][i,:])])
                    pdd['filtered_center'][qb] = freqs
                    pdd['filtered_amps'][qb] = pdd['amps_masked'][qb]

            # fit the freqs to the qubit model
            self.fit_func = self.get_param_value('fit_func', fit_mods.Qubit_dac_to_freq)

            if self.fit_func == fit_mods.Qubit_dac_to_freq_precise:
                fit_guess_func = fit_mods.Qubit_dac_arch_guess_precise
            else:
                fit_guess_func = fit_mods.Qubit_dac_arch_guess
            freq_mod = lmfit.Model(self.fit_func)
            fixed_params = \
                self.get_param_value("fixed_params_for_fit", {}).get(qb, None)
            if fixed_params is None:
                fixed_params = dict(E_c=0)
            freq_mod.guess = fit_guess_func.__get__(
                freq_mod, freq_mod.__class__)

            self.fit_dicts[f'freq_fit_{qb}'] = {
                'model': freq_mod,
                'fit_xvals': {'dac_voltage': pdd['filtered_amps'][qb]},
                'fit_yvals': {'data': pdd['filtered_center'][qb]},
                "guessfn_pars": {"fixed_params": fixed_params}}

            self.run_fitting()

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            sp_param = [k for k in self.mospm[qb] if 'freq' in k][0]
            self.plot_dicts[f'data_2d_{qb}'] = {
                'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                'ax_id': f'data_2d_{qb}',
                'plotfn': self.plot_colorxy,
                'xvals': pdd['sweep_points_dict'][qb]['sweep_points'],
                'yvals': pdd['sweep_points_2D_dict'][qb][sp_param],
                'zvals': np.transpose(pdd['data_reshaped'][qb]),
                'xlabel': r'Flux pulse amplitude',
                'xunit': 'V',
                'ylabel': r'Qubit drive frequency',
                'yunit': 'Hz',
                'zlabel': 'Excited state population',
            }

            if self.do_fitting:
                if self.get_param_value('scatter', True):
                    label = f'freq_scatter_{qb}_scatter'
                    self.plot_dicts[label] = {
                        'title': rdd['measurementstring'] +
                        '\n' + rdd['timestamp'],
                        'ax_id': f'data_2d_{qb}',
                        'plotfn': self.plot_line,
                        'linestyle': '',
                        'marker': 'o',
                        'xvals': pdd['filtered_amps'][qb],
                        'yvals': pdd['filtered_center'][qb],
                        'xlabel': r'Flux pulse amplitude',
                        'xunit': 'V',
                        'ylabel': r'Qubit drive frequency',
                        'yunit': 'Hz',
                        'color': 'white',
                    }

                amps = pdd['sweep_points_dict'][qb]['sweep_points'][
                                     :-self.num_cal_points]

                label = f'freq_scatter_{qb}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'data_2d_{qb}',
                    'plotfn': self.plot_line,
                    'linestyle': '-',
                    'marker': '',
                    'xvals': amps,
                    'yvals': self.fit_func(amps,
                            **self.fit_res[f'freq_fit_{qb}'].best_values),
                    'color': 'red',
                }


class T1FrequencySweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_cp = self.num_cal_points
        self.lengths = OrderedDict()
        self.amps = OrderedDict()
        self.freqs = OrderedDict()
        for qbn in self.qb_names:
            len_key = [pn for pn in self.mospm[qbn] if 'length' in pn]
            if len(len_key) == 0:
                raise KeyError('Couldn"t find sweep points corresponding to '
                               'flux pulse length.')
            self.lengths[qbn] = self.sp.get_sweep_params_property(
                'values', 0, len_key[0])

            amp_key = [pn for pn in self.mospm[qbn] if 'amp' in pn]
            if len(len_key) == 0:
                raise KeyError('Couldn"t find sweep points corresponding to '
                               'flux pulse amplitude.')
            self.amps[qbn] = self.sp.get_sweep_params_property(
                'values', 1, amp_key[0])

            freq_key = [pn for pn in self.mospm[qbn] if 'freq' in pn]
            if len(freq_key) == 0:
                self.freqs[qbn] = None
            else:
                self.freqs[qbn] =self.sp.get_sweep_params_property(
                    'values', 1, freq_key[0])

        nr_amps = len(self.amps[self.qb_names[0]])
        nr_lengths = len(self.lengths[self.qb_names[0]])

        # make matrix out of vector
        data_reshaped_no_cp = {qb: np.reshape(deepcopy(
                pdd['data_to_fit'][qb][
                :, :pdd['data_to_fit'][qb].shape[1]-nr_cp]).flatten(),
                (nr_amps, nr_lengths)) for qb in self.qb_names}

        pdd['data_reshaped_no_cp'] = data_reshaped_no_cp

        pdd['mask'] = {qb: np.ones(nr_amps, dtype=bool)
                           for qb in self.qb_names}

    def prepare_fitting(self):
        pdd = self.proc_data_dict

        self.fit_dicts = OrderedDict()
        exp_mod = fit_mods.ExponentialModel()
        for qb in self.qb_names:
            for i, data in enumerate(pdd['data_reshaped_no_cp'][qb]):
                self.fit_dicts[f'exp_fit_{qb}_amp_{i}'] = {
                    'model': exp_mod,
                    'fit_xvals': {'x': self.lengths[qb]},
                    'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['T1'] = {}
        pdd['T1_err'] = {}

        for qb in self.qb_names:
            pdd['T1'][qb] = np.array([
                abs(self.fit_res[f'exp_fit_{qb}_amp_{i}'].best_values['decay'])
                for i in range(len(self.amps[qb]))])

            pdd['T1_err'][qb] = np.array([
                self.fit_res[f'exp_fit_{qb}_amp_{i}'].params['decay'].stderr
                for i in range(len(self.amps[qb]))])

            for i in range(len(self.amps[qb])):
                try:
                    if pdd['T1_err'][qb][i] >= 10 * pdd['T1'][qb][i]:
                        pdd['mask'][qb][i] = False
                except TypeError:
                    pdd['mask'][qb][i] = False

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            for p, param_values in enumerate([self.amps, self.freqs]):
                if param_values is None:
                    continue
                suffix = '_amp' if p == 0 else '_freq'
                mask = pdd['mask'][qb]
                xlabel = r'Flux pulse amplitude' if p == 0 else \
                    r'Derived qubit ge frequency'

                if self.do_fitting:
                    # Plot T1 vs flux pulse amplitude
                    label = f'T1_fit_{qb}{suffix}_{self.data_to_fit[qb]}'
                    self.plot_dicts[label] = {
                        'title': rdd['measurementstring'] + '\n' + rdd['timestamp'],
                        'plotfn': self.plot_line,
                        'linestyle': '-',
                        'xvals': param_values[qb][mask],
                        'yvals': pdd['T1'][qb][mask],
                        'yerr': pdd['T1_err'][qb][mask],
                        'xlabel': xlabel,
                        'xunit': 'V' if p == 0 else 'Hz',
                        'ylabel': r'T1',
                        'yunit': 's',
                        'color': 'blue',
                    }

                # Plot rotated integrated average in dependece of flux pulse
                # amplitude and length
                label = f'T1_color_plot_{qb}{suffix}_{self.data_to_fit[qb]}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] + '\n' + rdd['timestamp'],
                    'plotfn': self.plot_colorxy,
                    'linestyle': '-',
                    'xvals': param_values[qb][mask],
                    'yvals': self.lengths[qb],
                    'zvals': np.transpose(pdd['data_reshaped_no_cp'][qb][mask]),
                    'xlabel': xlabel,
                    'xunit': 'V' if p == 0 else 'Hz',
                    'ylabel': r'Flux pulse length',
                    'yunit': 's',
                    'clabel': self.get_yaxis_label(qb)
                }

                # Plot population loss for the first flux pulse length as a
                # function of flux pulse amplitude
                label = f'Pop_loss_{qb}{suffix}_{self.data_to_fit[qb]}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] + '\n' + rdd['timestamp'],
                    'plotfn': self.plot_line,
                    'linestyle': '-',
                    'xvals': param_values[qb][mask],
                    'yvals': 1 - pdd['data_reshaped_no_cp'][qb][:, 0][mask],
                    'xlabel': xlabel,
                    'xunit': 'V' if p == 0 else 'Hz',
                    'ylabel': r'Pop. loss {} @ {:.0f} ns'.format(
                        self.data_to_fit[qb],
                        self.lengths[qb][0]/1e-9
                    ),
                    'yunit': '',
                }

            # Plot all fits in single figure
            if self.get_param_value('all_fits', False) and self.do_fitting:
                colormap = self.get_param_value('colormap', mpl.cm.Blues)
                for i in range(len(self.amps[qb])):
                    color = colormap(i/(len(self.amps[qb])-1))
                    label = f'exp_fit_{qb}_amp_{i}'
                    fitid = param_values[qb][i]
                    self.plot_dicts[label] = {
                        'title': rdd['measurementstring'] + '\n' + rdd['timestamp'],
                        'fig_id': f'T1_fits_{qb}_{self.data_to_fit[qb]}',
                        'xlabel': r'Flux pulse length',
                        'xunit': 's',
                        'ylabel': r'Excited state population',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.get_param_value('plot_init', False),
                        'color': color,
                        'setlabel': f'freq={fitid:.4f}' if p == 1
                                            else f'amp={fitid:.4f}',
                        'do_legend': False,
                        'legend_bbox_to_anchor': (1, 1),
                        'legend_pos': 'upper left',
                        }

                    label = f'freq_scatter_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'fig_id': f'T1_fits_{qb}_{self.data_to_fit[qb]}',
                        'plotfn': self.plot_line,
                        'xvals': self.lengths[qb],
                        'linestyle': '',
                        'yvals': pdd['data_reshaped_no_cp'][qb][i, :],
                        'color': color,
                        'setlabel': f'freq={fitid:.4f}' if p == 1
                                            else f'amp={fitid:.4f}',
                    }


class T2FrequencySweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        pdd = self.proc_data_dict
        nr_cp = self.num_cal_points
        nr_amps = len(self.metadata['amplitudes'])
        nr_lengths = len(self.metadata['flux_lengths'])
        nr_phases = len(self.metadata['phases'])

        # make matrix out of vector
        data_reshaped_no_cp = {qb: np.reshape(
            deepcopy(pdd['data_to_fit'][qb][:-nr_cp]).flatten(),
            (nr_amps, nr_lengths, nr_phases)) for qb in self.qb_names}

        pdd['data_reshaped_no_cp'] = data_reshaped_no_cp
        if self.metadata['use_cal_points']:
            pdd['cal_point_data'] = {qb: deepcopy(
                pdd['data_to_fit'][qb][
                len(pdd['data_to_fit'][qb])-nr_cp:]) for qb in self.qb_names}

        pdd['mask'] = {qb: np.ones(nr_amps, dtype=bool)
                           for qb in self.qb_names}

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()
        nr_amps = len(self.metadata['amplitudes'])

        for qb in self.qb_names:
            for i in range(nr_amps):
                for j, data in enumerate(pdd['data_reshaped_no_cp'][qb][i]):
                    cos_mod = fit_mods.CosModel
                    guess_pars = fit_mods.Cos_guess(
                        model=cos_mod, t=self.metadata['phases'],
                        data=data,
                        freq_guess=1/360)
                    guess_pars['frequency'].value = 1/360
                    guess_pars['frequency'].vary = False
                    self.fit_dicts[f'cos_fit_{qb}_{i}_{j}'] = {
                        'fit_fn': fit_mods.CosFunc,
                        'fit_xvals': {'t': self.metadata['phases']},
                        'fit_yvals': {'data': data},
                        'guess_pars': guess_pars}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['T2'] = {}
        pdd['T2_err'] = {}
        pdd['phase_contrast'] = {}
        pdd['phase_contrast_stderr'] = {}
        nr_lengths = len(self.metadata['flux_lengths'])
        nr_amps = len(self.metadata['amplitudes'])

        guess_pars_dict_default = {
            qb: dict(amplitude=dict(value=0.5, vary=True),
                     decay=dict(value=1e-6, vary=True),
                     # FIXME vary=True seems to prevent convergence
                     n=dict(value=2, vary=False, min=1, max=2),
                     ) for qb in self.qb_names}

        gaussian_decay_func = \
            lambda x, amplitude, decay, n: amplitude * np.exp(-(x / decay) ** n)

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = {}
            pdd['phase_contrast_stderr'][qb] = {}
            pdd['mask'][qb] = [True]*len(self.metadata['amplitudes'])
            exp_mod = self.get_param_value('exp_fit_mod',
                                           lmfit.Model(gaussian_decay_func))
            guess_pars_dict = self.get_param_value("guess_pars_dict",
                                                   guess_pars_dict_default)[qb]
            for i in range(nr_amps):
                pdd['phase_contrast'][qb][f'amp_{i}'] = np.array(
                    [self.fit_res[f'cos_fit_{qb}_{i}_{j}'].best_values[
                         'amplitude'] for j in range(nr_lengths)])
                pdd['phase_contrast_stderr'][qb][f'amp_{i}'] = np.array(
                    [self.fit_res[f'cos_fit_{qb}_{i}_{j}'].params[
                         'amplitude'].stderr for j in range(nr_lengths)])
                for key in ['phase_contrast', 'phase_contrast_stderr']:
                    if None in pdd[key][qb][f'amp_{i}']:
                        log.warning(f"None values in {key} cos fit for "
                                    f"amplitude {i}! Skipping.")
                        pdd['mask'][qb][i] = False
                        continue  # No need to check all keys in that case


                for par, params in guess_pars_dict.items():
                    exp_mod.set_param_hint(par, **params)
                guess_pars = exp_mod.make_params()

                self.fit_dicts[f'exp_fit_{qb}_{i}'] = {
                    'fit_fn': exp_mod.func,
                    'guess_pars': guess_pars,
                    'fit_xvals': {'x': self.get_param_value('flux_lengths')},
                    'fit_yvals': {
                        'data': pdd['phase_contrast'][qb][f'amp_{i}']},
                }


            self.run_fitting()

            pdd['T2'][qb] = np.array([
                self.fit_res[f'exp_fit_{qb}_{i}'].best_values['decay']
                if f'exp_fit_{qb}_{i}' in self.fit_res else None
                for i in range(len(self.metadata['amplitudes']))])
            pdd['T2_err'][qb] = np.array([
                self.fit_res[f'exp_fit_{qb}_{i}'].params['decay'].stderr
                if f'exp_fit_{qb}_{i}' in self.fit_res else None
                for i in range(len(self.metadata['amplitudes']))])

            for i in range(len(self.metadata['amplitudes'])):
                try:
                    if (pdd['T2_err'][qb][i] is None
                            or pdd['T2_err'][qb][i] >= 1e-5):
                        pdd['mask'][qb][i] = False
                except TypeError:
                    pdd['mask'][qb][i] = False

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        colormap = self.get_param_value('colormap', mpl.cm.viridis)
        freqs = self.get_param_value('frequencies')
        delays = self.get_param_value('flux_lengths')

        for qb in self.qb_names:
            mask = pdd['mask'][qb]

            for i in range(len(self.metadata['amplitudes'])):
                color = colormap(i/(len(self.metadata['amplitudes'])-1))
                fitid = self.get_param_value('amplitudes')[i] if\
                    freqs is None else self.get_param_value('frequencies')[i]

                if self.get_param_value('all_fits', False):
                    label = f'cos_fit_{qb}_{i}'
                    # Raw data plot for pulse amplitude i, as a function of
                    # phase, coloured by pulse length
                    self.plot_dicts[label + '_data'] = {
                        'ax_id': label,
                        'plotfn': self.plot_line,
                        'xvals': [self.metadata['phases']]*len(delays),
                        'linestyle': '',
                        'yvals': pdd['data_reshaped_no_cp'][qb][i],
                        'color': [colormap(j/(len(delays)-1)) for j
                                  in range(len(delays))],
                        'xlabel': r'Phase',
                        'xunit': '°',
                        'ylabel': r'Excited state population',
                        }
                    # Corresponding cos fit
                    for j in range(len(delays)):  # FIXME no loop would be nice
                        self.plot_dicts[label + f"_{j}"] = {
                            'ax_id': label,
                            'plotfn': self.plot_fit,
                            'color': colormap(j/(len(delays)-1)),
                            'fit_res': self.fit_res[label + f"_{j}"],
                            }
                    # Same raw data, as a 2D plot
                    self.plot_dicts[label + '_data_2D'] = {
                        # 'ax_id': f'T2_fits_{qb}_phases',
                        'plotfn': self.plot_colorxy,
                        'yvals': self.metadata['phases'],
                        'xvals': self.metadata['flux_lengths'],
                        'zvals': pdd['data_reshaped_no_cp'][qb][i].T,
                        'zrange': [0, 1],
                        'xlabel': 'Flux pulse length',
                        'xunit': 's',
                        'ylabel': 'Phase',
                        'yunit': '°',
                        'plotcbar': True,
                        'clabel': f"Excited state population (%)",
                    }

                if mask[i]:
                    # Contrast (amplitude) from the cos fits, as a function of
                    # pulse length, coloured by pulse amplitude
                    label = f'exp_fit_{qb}_{i}'
                    self.plot_dicts[label + 'data'] = {
                        'ax_id': f'T2_fits_{qb}',
                        'plotfn': self.plot_line,
                        "xvals": delays,
                        "yvals": pdd['phase_contrast'][qb][f'amp_{i}'],
                        "yerr": pdd['phase_contrast_stderr'][qb][f'amp_{i}'],
                        "marker": "o",
                        'plot_init': self.options_dict.get('plot_init', False),
                        'color': color,
                        'setlabel': f'freq={fitid:.4f}' if freqs
                            else f'amp={fitid:.4f}',
                        'do_legend': False,
                        'legend_bbox_to_anchor': (1, 1),
                        'legend_pos': 'upper left',
                        'linestyle': "none",
                    }
                    # Corresponding exponential fit
                    self.plot_dicts[label] = {
                        'title': rdd['measurementstring'] +
                                '\n' + rdd['timestamp'],
                        'ax_id': f'T2_fits_{qb}',
                        'xlabel': r'Flux pulse length',
                        'xunit': 's',
                        'ylabel': r'Excited state population',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.get_param_value('plot_init', False),
                        'color': color,
                        'setlabel': f'freq={fitid:.4f}' if freqs
                                            else f'amp={fitid:.4f}',
                        'do_legend': False,
                        'legend_bbox_to_anchor': (1, 1),
                        'legend_pos': 'upper left',
                        }

            if np.sum(mask):
                label = f'T2_fit_{qb}'
                xvals = self.metadata['amplitudes'][mask] if \
                    self.metadata['frequencies'] is None else \
                    self.metadata['frequencies'][mask]
                xlabel = r'Flux pulse amplitude' if \
                    self.metadata['frequencies'] is None else \
                    r'Derived qubit ge frequency'
                # Final T2(freq or amp) plot
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                    'plotfn': self.plot_line,
                    'linestyle': '-',
                    'xvals': xvals,
                    'yvals': pdd['T2'][qb][mask],
                    'yerr': pdd['T2_err'][qb][mask],
                    'xlabel': xlabel,
                    'xunit': 'V' if self.metadata['frequencies'] is None else 'Hz',
                    'ylabel': r'T2',
                    'yunit': 's',
                    'color': 'tab:green',
                }


class MeasurementInducedDephasingAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict

        try: # Extract data
            pdd['amps_reshaped'] = {qbn: pdd['sweep_points_2D_dict'][qbn][
                'amplitude'] for qbn in self.qb_names}
            pdd['phases_reshaped'] = [pdd['sweep_points_dict'][
                self.qb_names[0]]['msmt_sweep_points']] * len(pdd[
                    'amps_reshaped'][self.qb_names[0]])
            pdd['data_reshaped'] = {
                qbn: np.array(pdd['data_to_fit'][qbn])[:, :-2]
                for qbn in self.qb_names}
        except Exception:
            # If a problem occurred, this might come from old data
            # incompatible with the QuantumExperiment framework, for instance
            # missing SweepPoints per qubit. In this case, try to extract
            # data the old way:
            pdd['data_reshaped'] = {qb: [] for qb in pdd['data_to_fit']}
            pdd['amps_reshaped'] = {qb: np.unique(
                self.metadata['hard_sweep_params']['ro_amp_scale']['values'])
                                    for qb in pdd['data_to_fit']}
            pdd['phases_reshaped'] = []
            for amp in pdd['amps_reshaped'][self.qb_names[0]]:
                mask = self.metadata['hard_sweep_params']['ro_amp_scale'][
                           'values'] == amp
                pdd['phases_reshaped'].append(
                    self.metadata['hard_sweep_params']['phase']['values'][mask])
                for qb in self.qb_names:
                    pdd['data_reshaped'][qb].append(
                        pdd['data_to_fit'][qb][:len(mask)][mask])

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        self.fit_dicts = OrderedDict()
        for qb in self.qb_names:
            for i, data in enumerate(pdd['data_reshaped'][qb]):
                cos_mod = fit_mods.CosModel
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod, t=pdd['phases_reshaped'][i],
                    data=data, freq_guess=1/360)
                guess_pars['frequency'].value = 1/360
                guess_pars['frequency'].vary = False
                self.fit_dicts[f'cos_fit_{qb}_{i}'] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': pdd['phases_reshaped'][i]},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['phase_contrast'] = {}
        pdd['phase_offset'] = {}
        pdd['sigma'] = {}
        pdd['sigma_err'] = {}
        pdd['a'] = {}
        pdd['a_err'] = {}
        pdd['c'] = {}
        pdd['c_err'] = {}

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['amplitude']
                for i, _ in enumerate(pdd['data_reshaped'][qb])])
            pdd['phase_offset'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['phase']
                for i, _ in enumerate(pdd['data_reshaped'][qb])])
            pdd['phase_offset'][qb] += np.pi * (pdd['phase_contrast'][qb] < 0)
            pdd['phase_offset'][qb] = (pdd['phase_offset'][qb] + np.pi) % (2 * np.pi) - np.pi
            pdd['phase_offset'][qb] = 180*np.unwrap(pdd['phase_offset'][qb])/np.pi
            pdd['phase_contrast'][qb] = np.abs(pdd['phase_contrast'][qb])

            gauss_mod = lmfit.models.GaussianModel()
            self.fit_dicts[f'phase_contrast_fit_{qb}'] = {
                'model': gauss_mod,
                'guess_dict': {'center': {'value': 0, 'vary': False}},
                'fit_xvals': {'x': pdd['amps_reshaped'][qb]},
                'fit_yvals': {'data': pdd['phase_contrast'][qb]}}

            quadratic_mod = lmfit.models.QuadraticModel()
            self.fit_dicts[f'phase_offset_fit_{qb}'] = {
                'model': quadratic_mod,
                'guess_dict': {'b': {'value': 0, 'vary': False}},
                'fit_xvals': {'x': pdd['amps_reshaped'][qb]},
                'fit_yvals': {'data': pdd['phase_offset'][qb]}}

            self.run_fitting()
            self.save_fit_results()

            pdd['sigma'][qb] = self.fit_res[f'phase_contrast_fit_{qb}'].best_values['sigma']
            pdd['sigma_err'][qb] = self.fit_res[f'phase_contrast_fit_{qb}'].params['sigma']. \
                stderr
            pdd['a'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].best_values['a']
            pdd['a_err'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].params['a'].stderr
            pdd['c'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].best_values['c']
            pdd['c_err'][qb] = self.fit_res[f'phase_offset_fit_{qb}'].params['c'].stderr

            pdd['sigma_err'][qb] = float('nan') if pdd['sigma_err'][qb] is None \
                else pdd['sigma_err'][qb]
            pdd['a_err'][qb] = float('nan') if pdd['a_err'][qb] is None else pdd['a_err'][qb]
            pdd['c_err'][qb] = float('nan') if pdd['c_err'][qb] is None else pdd['c_err'][qb]

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        phases_equal = True
        for phases in pdd['phases_reshaped'][1:]:
            if not np.all(phases == pdd['phases_reshaped'][0]):
                phases_equal = False
                break

        for qb in self.qb_names:
            if phases_equal:
                self.plot_dicts[f'data_2d_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'plotfn': self.plot_colorxy,
                    'xvals': pdd['phases_reshaped'][0],
                    'yvals': pdd['amps_reshaped'][qb],
                    'zvals': pdd['data_reshaped'][qb],
                    'xlabel': r'Pulse phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'yunit': '',
                    'zlabel': 'Excited state population',
                }

            colormap = self.get_param_value('colormap', mpl.cm.Blues)
            for i, amp in enumerate(pdd['amps_reshaped'][qb]):
                color = colormap(i/(len(pdd['amps_reshaped'][qb])-1))
                label = f'cos_data_{qb}_{i}'
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'amplitude_crossections_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['phases_reshaped'][i],
                    'yvals': pdd['data_reshaped'][qb][i],
                    'xlabel': r'Pulse phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': 'Excited state population',
                    'linestyle': '',
                    'color': color,
                    'setlabel': f'amp={amp:.4f}',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }
            if self.do_fitting:
                for i, amp in enumerate(pdd['amps_reshaped'][qb]):
                    color = colormap(i/(len(pdd['amps_reshaped'][qb])-1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'amplitude_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.get_param_value('plot_init', False),
                        'color': color,
                        'setlabel': f'fit, amp={amp:.4f}',
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': 200*pdd['phase_contrast'][qb],
                    'xlabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'xunit': '',
                    'ylabel': 'Phase contrast',
                    'yunit': '%',
                    'linestyle': '',
                    'color': 'k',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_fit_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': 200*self.fit_res[f'phase_contrast_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_labels_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': 200*pdd['phase_contrast'][qb],
                    'marker': '',
                    'linestyle': '',
                    'setlabel': r'$\sigma = ({:.5f} \pm {:.5f})$ V'.
                        format(pdd['sigma'][qb], pdd['sigma_err'][qb]),
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }

                # Phase offset
                self.plot_dicts[f'phase_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': pdd['phase_offset'][qb],
                    'xlabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'xunit': '',
                    'ylabel': 'Phase offset',
                    'yunit': 'deg',
                    'linestyle': '',
                    'color': 'k',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_fit_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': self.fit_res[f'phase_offset_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_labels_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'][qb],
                    'yvals': pdd['phase_offset'][qb],
                    'marker': '',
                    'linestyle': '',
                    'setlabel': r'$a = {:.0f} \pm {:.0f}$ deg/V${{}}^2$'.
                        format(pdd['a'][qb], pdd['a_err'][qb]) + '\n' +
                                r'$c = {:.1f} \pm {:.1f}$ deg'.
                        format(pdd['c'][qb], pdd['c_err'][qb]),
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }


class DriveCrosstalkCancellationAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()
        if self.sp is None:
            raise ValueError('This analysis needs a SweepPoints '
                             'class instance.')

        pdd = self.proc_data_dict
        # get the ramsey phases as the values of the first sweep parameter
        # in the 2nd sweep dimension.
        # !!! This assumes all qubits have the same ramsey phases !!!
        pdd['ramsey_phases'] = self.sp.get_sweep_params_property('values', 1)
        pdd['qb_sweep_points'] = {}
        pdd['qb_sweep_param'] = {}
        for k, v in self.sp.get_sweep_dimension(0).items():
            if k == 'phase':
                continue
            qb, param = k.split('.')
            pdd['qb_sweep_points'][qb] = v[0]
            pdd['qb_sweep_param'][qb] = (param, v[1], v[2])
        pdd['qb_msmt_vals'] = {}
        pdd['qb_cal_vals'] = {}

        for qb, data in pdd['data_to_fit'].items():
            pdd['qb_msmt_vals'][qb] = data[:, :-self.num_cal_points].reshape(
                len(pdd['qb_sweep_points'][qb]), len(pdd['ramsey_phases']))
            pdd['qb_cal_vals'][qb] = data[0, -self.num_cal_points:]

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()
        for qb in self.qb_names:
            for i, data in enumerate(pdd['qb_msmt_vals'][qb]):
                cos_mod = fit_mods.CosModel
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod, t=pdd['ramsey_phases'],
                    data=data, freq_guess=1/360)
                guess_pars['frequency'].value = 1/360
                guess_pars['frequency'].vary = False
                self.fit_dicts[f'cos_fit_{qb}_{i}'] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': pdd['ramsey_phases']},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['phase_contrast'] = {}
        pdd['phase_offset'] = {}

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = np.array([
                2*self.fit_res[f'cos_fit_{qb}_{i}'].best_values['amplitude']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['phase']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] *= 180/np.pi
            pdd['phase_offset'][qb] += 180 * (pdd['phase_contrast'][qb] < 0)
            pdd['phase_offset'][qb] = (pdd['phase_offset'][qb] + 180) % 360 - 180
            pdd['phase_contrast'][qb] = np.abs(pdd['phase_contrast'][qb])

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            self.plot_dicts[f'data_2d_{qb}'] = {
                'title': rdd['measurementstring'] +
                         '\n' + rdd['timestamp'] + '\n' + qb,
                'plotfn': self.plot_colorxy,
                'xvals': pdd['ramsey_phases'],
                'yvals': pdd['qb_sweep_points'][qb],
                'zvals': pdd['qb_msmt_vals'][qb],
                'xlabel': r'Ramsey phase, $\phi$',
                'xunit': 'deg',
                'ylabel': pdd['qb_sweep_param'][qb][2],
                'yunit': pdd['qb_sweep_param'][qb][1],
                'zlabel': 'Excited state population',
            }

            colormap = self.get_param_value('colormap', mpl.cm.Blues)
            for i, pval in enumerate(pdd['qb_sweep_points'][qb]):
                if i == len(pdd['qb_sweep_points'][qb]) - 1:
                    legendlabel='data, ref.'
                else:
                    legendlabel = f'data, {pdd["qb_sweep_param"][qb][0]}='\
                                  f'{pval:.4f}{pdd["qb_sweep_param"][qb][1]}'
                color = colormap(i/(len(pdd['qb_sweep_points'][qb])-1))
                label = f'cos_data_{qb}_{i}'

                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'param_crossections_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['ramsey_phases'],
                    'yvals': pdd['qb_msmt_vals'][qb][i],
                    'xlabel': r'Ramsey phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': 'Excited state population',
                    'linestyle': '',
                    'color': color,
                    'setlabel': legendlabel,
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }
            if self.do_fitting:
                for i, pval in enumerate(pdd['qb_sweep_points'][qb]):
                    if i == len(pdd['qb_sweep_points'][qb]) - 1:
                        legendlabel = 'fit, ref.'
                    else:
                        legendlabel = f'fit, {pdd["qb_sweep_param"][qb][0]}='\
                                      f'{pval:.4f}{pdd["qb_sweep_param"][qb][1]}'
                    color = colormap(i/(len(pdd['qb_sweep_points'][qb])-1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'param_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.get_param_value('plot_init', False),
                        'color': color,
                        'do_legend': False,
                        # 'setlabel': legendlabel
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['qb_sweep_points'][qb][:-1],
                    'yvals': pdd['phase_contrast'][qb][:-1] * 100,
                    'xlabel': pdd['qb_sweep_param'][qb][2],
                    'xunit': pdd['qb_sweep_param'][qb][1],
                    'ylabel': 'Phase contrast',
                    'yunit': '%',
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'C0',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_ref_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_hlines,
                    'xmin': pdd['qb_sweep_points'][qb][:-1].min(),
                    'xmax': pdd['qb_sweep_points'][qb][:-1].max(),
                    'y': pdd['phase_contrast'][qb][-1] * 100,
                    'linestyle': '--',
                    'colors': '0.6',
                    'setlabel': 'ref',
                    'do_legend': True,
                }

                # Phase offset
                self.plot_dicts[f'phase_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['qb_sweep_points'][qb][:-1],
                    'yvals': pdd['phase_offset'][qb][:-1],
                    'xlabel': pdd['qb_sweep_param'][qb][2],
                    'xunit': pdd['qb_sweep_param'][qb][1],
                    'ylabel': 'Phase offset',
                    'yunit': 'deg',
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'C0',
                    'setlabel': 'data',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_ref_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_hlines,
                    'xmin': pdd['qb_sweep_points'][qb][:-1].min(),
                    'xmax': pdd['qb_sweep_points'][qb][:-1].max(),
                    'y': pdd['phase_offset'][qb][-1],
                    'linestyle': '--',
                    'colors': '0.6',
                    'setlabel': 'ref',
                    'do_legend': True,
                }


class FluxlineCrosstalkAnalysis(MultiQubit_TimeDomain_Analysis):
    """Analysis for the measure_fluxline_crosstalk measurement.

    The measurement involves Ramsey measurements on a set of crosstalk qubits,
    which have been brought to a flux-sensitive position with a flux pulse.
    The first dimension is the ramsey-phase of these qubits.

    In the second sweep dimension, the amplitude of a flux pulse on another
    (target) qubit is swept.

    The analysis extracts the change in Ramsey phase offset, which gets
    converted to a frequency offset due to the flux pulse on the target qubit.
    The frequency offset is then converted to a flux offset, which is a measure
    of the crosstalk between the target fluxline and the crosstalk qubit.

    The measurement is hard-compressed, meaning the raw data is inherently 1d,
    with one set of calibration points as the final segments. The experiment
    part of the measured values are reshaped to the correct 2d shape for
    the analysis. The sweep points passed into the analysis should still reflect
    the 2d nature of the measurement, meaning the ramsey phase values should be
    passed in the first dimension and the target fluxpulse amplitudes in the
    second sweep dimension.
    """


    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for param in ['fit_ge_freq_from_flux_pulse_amp',
                      'fit_ge_freq_from_dc_offset',
                      'flux_amplitude_bias_ratio',
                      'flux_parking']:
            params_dict.update({
                f'{qbn}.{param}': f'Instrument settings.{qbn}.{param}'
                for qbn in qb_names})
        kwargs['params_dict'] = kwargs.get('params_dict', {})
        kwargs['params_dict'].update(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def process_data(self):
        super().process_data()
        if self.sp is None:
            raise ValueError('This analysis needs a SweepPoints '
                             'class instance.')

        pdd = self.proc_data_dict

        pdd['ramsey_phases'] = self.sp.get_sweep_params_property('values', 0)
        pdd['target_amps'] = self.sp.get_sweep_params_property('values', 1)
        pdd['target_fluxpulse_length'] = \
            self.get_param_value('target_fluxpulse_length')
        pdd['crosstalk_qubits_amplitudes'] = \
            self.get_param_value('crosstalk_qubits_amplitudes')

        pdd['qb_msmt_vals'] = {qb:
            pdd['data_to_fit'][qb][:, :-self.num_cal_points].reshape(
                len(pdd['target_amps']), len(pdd['ramsey_phases']))
            for qb in self.qb_names}
        pdd['qb_cal_vals'] = {
            qb: pdd['data_to_fit'][qb][0, -self.num_cal_points:]
            for qb in self.qb_names}

    def prepare_fitting(self):
        pdd = self.proc_data_dict
        self.fit_dicts = OrderedDict()
        cos_mod = lmfit.Model(fit_mods.CosFunc)
        cos_mod.guess = fit_mods.Cos_guess.__get__(cos_mod, cos_mod.__class__)
        for qb in self.qb_names:
            for i, data in enumerate(pdd['qb_msmt_vals'][qb]):
                self.fit_dicts[f'cos_fit_{qb}_{i}'] = {
                    'model': cos_mod,
                    'guess_dict': {'frequency': {'value': 1 / 360,
                                                 'vary': False}},
                    'fit_xvals': {'t': pdd['ramsey_phases']},
                    'fit_yvals': {'data': data}}

    def analyze_fit_results(self):
        pdd = self.proc_data_dict

        pdd['phase_contrast'] = {}
        pdd['phase_offset'] = {}
        pdd['freq_offset'] = {}
        pdd['freq'] = {}

        self.skip_qb_freq_fits = self.get_param_value('skip_qb_freq_fits', False)
        self.vfc_method = self.get_param_value('vfc_method', 'transmon_res')

        if not self.skip_qb_freq_fits:
            pdd['flux'] = {}

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = np.array([
                2 * self.fit_res[f'cos_fit_{qb}_{i}'].best_values['amplitude']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] = np.array([
                self.fit_res[f'cos_fit_{qb}_{i}'].best_values['phase']
                for i, _ in enumerate(pdd['qb_msmt_vals'][qb])])
            pdd['phase_offset'][qb] *= 180 / np.pi
            pdd['phase_offset'][qb] += 180 * (pdd['phase_contrast'][qb] < 0)
            pdd['phase_offset'][qb] = (pdd['phase_offset'][qb] + 180) % 360 - 180
            pdd['phase_offset'][qb] = \
                np.unwrap(pdd['phase_offset'][qb] / 180 * np.pi) * 180 / np.pi
            pdd['phase_contrast'][qb] = np.abs(pdd['phase_contrast'][qb])
            pdd['freq_offset'][qb] = pdd['phase_offset'][qb] / 360 / pdd[
                'target_fluxpulse_length']
            startval_slope = (pdd['freq_offset'][qb][-1] - pdd['freq_offset'][
                qb][0]) / (pdd['target_amps'][-1] - pdd['target_amps'][0])
            startval_offset = pdd['freq_offset'][qb][
                len(pdd['freq_offset'][qb]) // 2]
            fr = lmfit.Model(lambda a,
                                    f_a=startval_slope,
                                    f0=startval_offset: a * f_a + f0).fit(
                data=pdd['freq_offset'][qb], a=pdd['target_amps'])
            pdd['freq_offset'][qb] -= fr.best_values['f0']
            if not self.skip_qb_freq_fits:
                if self.vfc_method == 'approx':
                    mpars = self.raw_data_dict[
                        f'{qb}.fit_ge_freq_from_flux_pulse_amp']
                    freq_pulsed_no_crosstalk = fit_mods.Qubit_dac_to_freq(
                        pdd['crosstalk_qubits_amplitudes'].get(qb, 0), **mpars)
                    pdd['freq'][qb] = pdd['freq_offset'][
                                          qb] + freq_pulsed_no_crosstalk
                    mpars.update({'V_per_phi0': 1, 'dac_sweet_spot': 0})
                    pdd['flux'][qb] = fit_mods.Qubit_freq_to_dac(
                        pdd['freq'][qb], **mpars)
                else:
                    mpars = self.get_param_value(
                        f'{qb}.fit_ge_freq_from_dc_offset')
                    ratio = self.get_param_value(
                        f'{qb}.flux_amplitude_bias_ratio')
                    flux_parking = self.get_param_value(
                        f'{qb}.flux_parking')
                    bias = (mpars['dac_sweet_spot']
                            + mpars['V_per_phi0'] * flux_parking)
                    amp = pdd['crosstalk_qubits_amplitudes'].get(qb, 0)
                    freq_pulsed_no_crosstalk = fit_mods.Qubit_dac_to_freq_res(
                        (bias + amp / ratio), **mpars)
                    pdd['freq'][qb] = pdd['freq_offset'][qb] + freq_pulsed_no_crosstalk
                    # mpars.update({'V_per_phi0': 1, 'dac_sweet_spot': 0})
                    volt = fit_mods.Qubit_freq_to_dac_res(
                        pdd['freq'][qb], **mpars,
                        branch=(bias + amp / ratio))
                    pdd['flux'][qb] = (volt - mpars['dac_sweet_spot']) \
                                      / mpars['V_per_phi0']  # convert volt to flux
        # fit fitted results to linear models
        lin_mod = lmfit.Model(lambda x, a=1, b=0: a*x + b)
        def guess(model, data, x, **kwargs):
            a_guess = (data[-1] - data[0])/(x[-1] - x[0])
            b_guess = data[0] - x[0]*a_guess
            return model.make_params(a=a_guess, b=b_guess)
        lin_mod.guess = guess.__get__(lin_mod, lin_mod.__class__)

        keys_to_fit = []
        for qb in self.qb_names:
            for param in ['phase_offset', 'freq_offset', 'flux']:
                if param == 'flux' and self.skip_qb_freq_fits:
                    continue
                key = f'{param}_fit_{qb}'
                self.fit_dicts[key] = {
                    'model': lin_mod,
                    'fit_xvals': {'x': pdd['target_amps']},
                    'fit_yvals': {'data': pdd[param][qb]}}
                keys_to_fit.append(key)
        self.run_fitting(keys_to_fit=keys_to_fit)

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            self.plot_dicts[f'data_2d_{qb}'] = {
                'title': rdd['measurementstring'] +
                         '\n' + rdd['timestamp'] + '\n' + qb,
                'plotfn': self.plot_colorxy,
                'xvals': pdd['ramsey_phases'],
                'yvals': pdd['target_amps'],
                'zvals': pdd['qb_msmt_vals'][qb],
                'xlabel': r'Ramsey phase, $\phi$',
                'xunit': 'deg',
                'ylabel': self.sp.get_sweep_params_property('label', 1,
                                                            'target_amp'),
                'yunit': self.sp.get_sweep_params_property('unit', 1,
                                                           'target_amp'),
                'zlabel': 'Excited state population',
            }

            colormap = self.get_param_value('colormap', mpl.cm.plasma)
            for i, pval in enumerate(pdd['target_amps']):
                legendlabel = f'data, amp. = {pval:.4f} V'
                color = colormap(i / (len(pdd['target_amps']) - 1))
                label = f'cos_data_{qb}_{i}'

                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'param_crossections_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['ramsey_phases'],
                    'yvals': pdd['qb_msmt_vals'][qb][i],
                    'xlabel': r'Ramsey phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': 'Excited state population',
                    'linestyle': '',
                    'color': color,
                    'setlabel': legendlabel,
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                }
            if self.do_fitting:
                for i, pval in enumerate(pdd['target_amps']):
                    legendlabel = f'fit, amp. = {pval:.4f} V'
                    color = colormap(i / (len(pdd['target_amps']) - 1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'param_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.get_param_value('plot_init', False),
                        'color': color,
                        'setlabel': legendlabel,
                        'do_legend': False,
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['target_amps'],
                    'yvals': pdd['phase_contrast'][qb] * 100,
                    'xlabel':self.sp.get_sweep_params_property('label', 1,
                                                               'target_amp'),
                    'xunit': self.sp.get_sweep_params_property('unit', 1,
                                                               'target_amp'),
                    'ylabel': 'Phase contrast',
                    'yunit': '%',
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'C0',
                }

                # Phase offset
                self.plot_dicts[f'phase_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['target_amps'],
                    'yvals': pdd['phase_offset'][qb],
                    'xlabel':self.sp.get_sweep_params_property('label', 1,
                                                               'target_amp'),
                    'xunit': self.sp.get_sweep_params_property('unit', 1,
                                                               'target_amp'),
                    'ylabel': 'Phase offset',
                    'yunit': 'deg',
                    'linestyle': 'none',
                    'marker': 'o',
                    'color': 'C0',
                }

                # Frequency offset
                self.plot_dicts[f'freq_offset_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'] + '\n' + qb,
                    'ax_id': f'freq_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['target_amps'],
                    'yvals': pdd['freq_offset'][qb],
                    'xlabel':self.sp.get_sweep_params_property('label', 1,
                                                               'target_amp'),
                    'xunit': self.sp.get_sweep_params_property('unit', 1,
                                                               'target_amp'),
                    'ylabel': 'Freq. offset, $\\Delta f$',
                    'yunit': 'Hz',
                    'linestyle': 'none',
                    'marker': 'o',
                    'color': 'C0',
                }

                if not self.skip_qb_freq_fits:
                    # Flux
                    self.plot_dicts[f'flux_data_{qb}'] = {
                        'title': rdd['measurementstring'] +
                                 '\n' + rdd['timestamp'] + '\n' + qb,
                        'ax_id': f'flux_{qb}',
                        'plotfn': self.plot_line,
                        'xvals': pdd['target_amps'],
                        'yvals': pdd['flux'][qb],
                        'xlabel': self.sp[1]['target_amp'][2],
                        'xunit': self.sp[1]['target_amp'][1],
                        'ylabel': 'Flux, $\\Phi$',
                        'yunit': '$\\Phi_0$',
                        'linestyle': 'none',
                        'marker': 'o',
                        'color': 'C0',
                    }

                for param in ['phase_offset', 'freq_offset', 'flux']:
                    if param == 'flux' and self.skip_qb_freq_fits:
                        continue
                    self.plot_dicts[f'{param}_fit_{qb}'] = {
                        'ax_id': f'{param}_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[f'{param}_fit_{qb}'],
                        'plot_init': self.get_param_value('plot_init', False),
                        'linestyle': '-',
                        'marker': '',
                        'color': 'C1',
                    }


class RabiAnalysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        self.default_options['base_plot_name'] = 'Rabi'

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        def add_fit_dict(qbn, data, key, scalex=1):
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            reduction_arr = np.invert(np.isnan(data))
            data = data[reduction_arr]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points'][reduction_arr] * scalex
            cos_mod = lmfit.Model(fit_mods.CosFunc)
            guess_pars = fit_mods.Cos_guess(
                model=cos_mod, t=sweep_points, data=data)
            guess_pars['amplitude'].vary = True
            guess_pars['amplitude'].min = -10
            guess_pars['offset'].vary = True
            guess_pars['frequency'].vary = True
            guess_pars['phase'].vary = True
            self.set_user_guess_pars(guess_pars)

            self.fit_dicts[key] = {
                'fit_fn': fit_mods.CosFunc,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

        for qbn in self.qb_names:
            all_data = self.proc_data_dict['data_to_fit'][qbn]
            if self.get_param_value('TwoD'):
                daa = self.metadata.get('drive_amp_adaptation', {}).get(
                    qbn, None)
                for i, data in enumerate(all_data):
                    key = f'cos_fit_{qbn}_{i}'
                    add_fit_dict(qbn, data, key,
                                 scalex=1 if daa is None else daa[i])
            else:
                add_fit_dict(qbn, all_data, 'cos_fit_' + qbn)

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for k, fit_dict in self.fit_dicts.items():
            # k is of the form cos_fit_qbn_i if TwoD else cos_fit_qbn
            # replace k with qbn_i or qbn
            k = k.replace('cos_fit_', '')
            # split into qbn and i. (k + '_') is needed because if k = qbn
            # doing k.split('_') will only have one output and assignment to
            # two variables will fail.
            qbn, i = (k + '_').split('_')[:2]
            fit_res = fit_dict['fit_res']
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            self.proc_data_dict['analysis_params_dict'][k] = \
                self.get_amplitudes(fit_res=fit_res, sweep_points=sweep_points)
        self.save_processed_data(key='analysis_params_dict')

    @staticmethod
    def get_amplitudes(fit_res, sweep_points):
        # Extract the best fitted frequency and phase.
        freq_fit = fit_res.best_values['frequency']
        phase_fit = fit_res.best_values['phase']

        freq_std = fit_res.params['frequency'].stderr
        phase_std = fit_res.params['phase'].stderr

        # If fitted_phase<0, shift fitted_phase by 4. This corresponds to a
        # shift of 2pi in the argument of cos.
        if np.abs(phase_fit) < 0.1:
            phase_fit = 0

        # If phase_fit<1, the piHalf amplitude<0.
        if phase_fit < 1:
            log.info('The data could not be fitted correctly. '
                         'The fitted phase "%s" <1, which gives '
                         'negative piHalf '
                         'amplitude.' % phase_fit)

        stepsize = sweep_points[1] - sweep_points[0]
        if freq_fit > 2 * stepsize:
            log.info('The data could not be fitted correctly. The '
                         'frequency "%s" is too high.' % freq_fit)
        n = np.arange(-10, 10)

        piPulse_vals = (n*np.pi - phase_fit)/(2*np.pi*freq_fit)
        piHalfPulse_vals = (n*np.pi + np.pi/2 - phase_fit)/(2*np.pi*freq_fit)

        # find piHalfPulse
        try:
            piHalfPulse = \
                np.min(piHalfPulse_vals[piHalfPulse_vals >= sweep_points[0]])
            n_piHalf_pulse = n[piHalfPulse_vals==piHalfPulse][0]
        except ValueError:
            piHalfPulse = np.asarray([])

        if piHalfPulse.size == 0 or piHalfPulse > max(sweep_points):
            i = 0
            while (piHalfPulse_vals[i] < min(sweep_points) and
                   i<piHalfPulse_vals.size):
                i+=1
            piHalfPulse = piHalfPulse_vals[i]
            n_piHalf_pulse = n[i]

        # find piPulse
        try:
            if piHalfPulse.size != 0:
                piPulse = \
                    np.min(piPulse_vals[piPulse_vals >= piHalfPulse])
            else:
                piPulse = np.min(piPulse_vals[piPulse_vals >= 0.001])
            n_pi_pulse = n[piHalfPulse_vals == piHalfPulse][0]

        except ValueError:
            piPulse = np.asarray([])

        if piPulse.size == 0:
            i = 0
            while (piPulse_vals[i] < min(sweep_points) and
                   i < piPulse_vals.size):
                i += 1
            piPulse = piPulse_vals[i]
            n_pi_pulse = n[i]

        try:
            freq_idx = fit_res.var_names.index('frequency')
            phase_idx = fit_res.var_names.index('phase')
            if fit_res.covar is not None:
                cov_freq_phase = fit_res.covar[freq_idx, phase_idx]
            else:
                cov_freq_phase = 0
        except ValueError:
            cov_freq_phase = 0

        try:
            piPulse_std = RabiAnalysis.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_const=n_pi_pulse*np.pi,
                cov=cov_freq_phase)
            piHalfPulse_std = RabiAnalysis.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_const=n_piHalf_pulse*np.pi + np.pi/2,
                cov=cov_freq_phase)
        except Exception:
            log.warning(f'Some stderrs from fit are None, setting stderr '
                        f'of pi and pi/2 pulses to 0!')
            piPulse_std = 0
            piHalfPulse_std = 0

        # Calculate mask to extract the PiPulse_vals and piHalfPulse_vals that
        # are within the range of the sweep_points
        mask1 = np.logical_and(piPulse_vals > min(sweep_points),
                               piPulse_vals < max(sweep_points))
        mask2 = np.logical_and(piHalfPulse_vals > min(sweep_points),
                               piHalfPulse_vals < max(sweep_points))
        # Return the piPulse and piHalfPulse (needed by the RabiAnalysis), but
        # also all the remaining piPulse and piHalfPulse values within the range
        # of the sweep points (needed by the NPulsePhaseErrorCalibAnalysis).
        rabi_amplitudes = {'piPulse': piPulse,
                           'piPulse_stderr': piPulse_std,
                           'piHalfPulse': piHalfPulse,
                           'piHalfPulse_stderr': piHalfPulse_std,
                           'piPulse_vals': piPulse_vals[mask1],
                           'piHalfPulse_vals': piHalfPulse_vals[mask2]}

        return rabi_amplitudes

    @staticmethod
    def calculate_pulse_stderr(f, phi, f_err, phi_err,
                               period_const, cov=0):
        jacobian = np.array([-1 / (2 * np.pi * f),
                             - (period_const - phi) / (2 * np.pi * f**2)])
        cov_matrix = np.array([[phi_err**2, cov], [cov, f_err**2]])
        return np.sqrt(jacobian @ cov_matrix @ jacobian.T)

    def prepare_plots(self):
        super().prepare_plots()
        bpn = self.get_param_value('base_plot_name')
        if self.do_fitting:
            for k, fit_dict in self.fit_dicts.items():
                if k.startswith('amplitude_fit'):
                    # This is only for RabiFrequencySweepAnalysis.
                    # It is handled by prepare_amplitude_fit_plots of that class
                    continue

                # k is of the form cos_fit_qbn_i if TwoD else cos_fit_qbn
                # replace k with qbn_i or qbn
                k = k.replace('cos_fit_', '')
                # split into qbn and i. (k + '_') is needed because if k = qbn
                # doing k.split('_') will only have one output and assignment to
                # two variables will fail.
                qbn, i = (k + '_').split('_')[:2]
                sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points']
                first_sweep_param = self.get_first_sweep_param(
                    qbn, dimension=1)
                if len(i) and first_sweep_param is not None:
                    # TwoD
                    label, unit, vals = first_sweep_param
                    title_suffix = (f'{i}: {label} = ' + ' '.join(
                        SI_val_to_msg_str(vals[int(i)], unit,
                                          return_type=lambda x : f'{x:0.4f}')))
                    daa = self.metadata.get('drive_amp_adaptation', {}).get(
                        qbn, None)
                    if daa is not None:
                        sweep_points = sweep_points * daa[int(i)]
                else:
                    # OneD
                    title_suffix = ''
                fit_res = fit_dict['fit_res']
                base_plot_name = f'{bpn}_{k}_{self.data_to_fit[qbn]}'
                dtf = self.proc_data_dict['data_to_fit'][qbn]
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=dtf[int(i)] if i != '' else dtf,
                    sweep_points=sweep_points,
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn, TwoD=False,
                    title_suffix=title_suffix
                )

                self.plot_dicts['fit_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'cosine fit',
                    'color': 'r',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                rabi_amplitudes = self.proc_data_dict['analysis_params_dict']
                self.plot_dicts['piamp_marker_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([rabi_amplitudes[k]['piPulse']]),
                    'yvals': np.array([fit_res.model.func(
                        rabi_amplitudes[k]['piPulse'],
                        **fit_res.best_values)]),
                    'setlabel': '$\pi$-Pulse amp',
                    'color': 'r',
                    'marker': 'o',
                    'line_kws': {'markersize': 10},
                    'linestyle': '',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                self.plot_dicts['piamp_hline_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': [fit_res.model.func(
                        rabi_amplitudes[k]['piPulse'],
                        **fit_res.best_values)],
                    'xmin': sweep_points[0],
                    'xmax': sweep_points[-1],
                    'colors': 'gray'}

                self.plot_dicts['pihalfamp_marker_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([rabi_amplitudes[k]['piHalfPulse']]),
                    'yvals': np.array([fit_res.model.func(
                        rabi_amplitudes[k]['piHalfPulse'],
                        **fit_res.best_values)]),
                    'setlabel': '$\pi /2$-Pulse amp',
                    'color': 'm',
                    'marker': 'o',
                    'line_kws': {'markersize': 10},
                    'linestyle': '',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                self.plot_dicts['pihalfamp_hline_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': [fit_res.model.func(
                        rabi_amplitudes[k]['piHalfPulse'],
                        **fit_res.best_values)],
                    'xmin': sweep_points[0],
                    'xmax': sweep_points[-1],
                    'colors': 'gray'}

                trans_name = self.get_transition_name(qbn)
                old_pipulse_val = self.get_instrument_setting(
                    f'{qbn}.{trans_name}_amp180')
                # FIXME: the following condition is always False, isn't it?
                if old_pipulse_val != old_pipulse_val:
                    old_pipulse_val = 0  # FIXME: explain why
                old_pihalfpulse_val = self.get_instrument_setting(
                    f'{qbn}.{trans_name}_amp90_scale')
                # FIXME: the following condition is always False, isn't it?
                if old_pihalfpulse_val != old_pihalfpulse_val:
                    old_pihalfpulse_val = 0  # FIXME: explain why
                old_pihalfpulse_val *= old_pipulse_val

                textstr = ('  $\pi-Amp$ = {:.3f} V'.format(
                    rabi_amplitudes[k]['piPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                    rabi_amplitudes[k]['piPulse_stderr']) +
                           '\n$\pi/2-Amp$ = {:.3f} V '.format(
                    rabi_amplitudes[k]['piHalfPulse']) +
                           ' $\pm$ {:.3f} V '.format(
                    rabi_amplitudes[k]['piHalfPulse_stderr']) +
                           '\n  $\pi-Amp_{old}$ = ' + '{:.3f} V '.format(
                    old_pipulse_val) +
                           '\n$\pi/2-Amp_{old}$ = ' + '{:.3f} V '.format(
                    old_pihalfpulse_val))
                self.plot_dicts['text_msg_' + k] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class NPulsePhaseErrorCalibAnalysis(RabiAnalysis, PhaseErrorsAnalysisMixin):

    def extract_data(self):
        super().extract_data()
        self.default_options['base_plot_name'] = 'NPulsePhaseErrorCalib'
        self._extract_current_pulse_par_value()

    def prepare_plots(self):
        super().prepare_plots()
        if self.do_fitting:
            # We look for the plot dicts which prepare text boxes. Their names
            # start with text_msg_ (see RabiAnalysis.prepare_plots())
            txt_pdns = [k for k in self.plot_dicts if k.startswith('text_msg_')]
            for txt_pd_name in txt_pdns:
                # take everything after text_msg_
                suff_split = txt_pd_name.split('_')[2:]
                suff = '_'.join(suff_split)
                # to get the qubit name, we assume that the suffix contains the
                # qubit name and that it is separated by '_' from the rest of
                # the characters
                qbn = [s for s in suff_split if s.startswith('qb')][0]
                amplitudes = self.proc_data_dict['analysis_params_dict'][suff]
                val = amplitudes['piPulse']
                stderr = amplitudes['piPulse_stderr']
                textstr = self.create_textstr(qbn, val, stderr,
                                              break_lines=True)
                self.plot_dicts[txt_pd_name]['text_string'] = textstr


class RabiFrequencySweepAnalysis(RabiAnalysis):

    def extract_data(self):
        super().extract_data()
        # Extract additional parameters from the HDF file.
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            params_dict[f'drive_ch_{qbn}'] = \
                f'Instrument settings.{qbn}.ge_I_channel'
            params_dict[f'ge_freq_{qbn}'] = \
                f'Instrument settings.{qbn}.ge_freq'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def analyze_fit_results(self):
        super().analyze_fit_results()
        amplitudes = {qbn: np.array([[
            self.proc_data_dict[
                'analysis_params_dict'][f'{qbn}_{i}']['piPulse'],
            self.proc_data_dict[
                'analysis_params_dict'][f'{qbn}_{i}']['piPulse_stderr']]
            for i in range(self.sp.length(1))]) for qbn in self.qb_names}
        self.proc_data_dict['analysis_params_dict']['amplitudes'] = amplitudes

        fit_dict_keys = self.prepare_fitting_pulse_amps()
        self.run_fitting(keys_to_fit=fit_dict_keys)

        lo_freqsX = self.get_param_value('allowed_lo_freqs')
        mid_freq = np.mean(lo_freqsX)
        self.proc_data_dict['analysis_params_dict']['rabi_model_lo'] = {}
        func_repr = lambda a, b, c: \
            f'{a} * (x / 1e9) ** 2 + {b} * x/ 1e9 + {c}'
        for qbn in self.qb_names:
            drive_ch = self.raw_data_dict[f'drive_ch_{qbn}']
            pd = self.get_data_from_timestamp_list({
                f'ch_amp': f'Instrument settings.Pulsar.{drive_ch}_amp'})
            fit_res_L = self.fit_dicts[f'amplitude_fit_left_{qbn}']['fit_res']
            fit_res_R = self.fit_dicts[f'amplitude_fit_right_{qbn}']['fit_res']
            rabi_model_lo = \
                f'lambda x : np.minimum({pd["ch_amp"]}, ' \
                f'({func_repr(**fit_res_R.best_values)}) * (x >= {mid_freq})' \
                f'+ ({func_repr(**fit_res_L.best_values)}) * (x < {mid_freq}))'
            self.proc_data_dict['analysis_params_dict']['rabi_model_lo'][
                qbn] = rabi_model_lo

    def prepare_fitting_pulse_amps(self):
        exclude_freq_indices = self.get_param_value('exclude_freq_indices', {})
        # TODO: generalize the code for len(allowed_lo_freqs) > 2
        lo_freqsX = self.get_param_value('allowed_lo_freqs')
        if lo_freqsX is None:
            raise ValueError('allowed_lo_freqs not found.')
        fit_dict_keys = []
        self.proc_data_dict['analysis_params_dict']['optimal_vals'] = {}
        for i, qbn in enumerate(self.qb_names):
            excl_idxs = exclude_freq_indices.get(qbn, [])
            param = [p for p in self.mospm[qbn] if 'freq' in p][0]
            freqs = self.sp.get_sweep_params_property('values', 1, param)
            ampls = deepcopy(self.proc_data_dict['analysis_params_dict'][
                'amplitudes'][qbn])
            if len(excl_idxs):
                mask = np.array([i in excl_idxs for i in np.arange(len(freqs))])
                ampls = ampls[np.logical_not(mask)]
                freqs = freqs[np.logical_not(mask)]
            if 'cal_data' not in self.proc_data_dict['analysis_params_dict']:
                self.proc_data_dict['analysis_params_dict']['cal_data'] = {}
            self.proc_data_dict['analysis_params_dict']['cal_data'][qbn] = \
                [freqs, ampls[:, 0]]

            optimal_idx = np.argmin(np.abs(
                freqs - self.raw_data_dict[f'ge_freq_{qbn}']))
            self.proc_data_dict['analysis_params_dict']['optimal_vals'][qbn] = \
                (freqs[optimal_idx], ampls[optimal_idx, 0], ampls[optimal_idx, 1])

            mid_freq = np.mean(lo_freqsX)
            fit_func = lambda x, a, b, c: a * x ** 2 + b * x + c

            # fit left range
            model = lmfit.Model(fit_func)
            guess_pars = model.make_params(a=1, b=1, c=0)
            self.fit_dicts[f'amplitude_fit_left_{qbn}'] = {
                'fit_fn': fit_func,
                'fit_xvals': {'x': freqs[freqs < mid_freq]/1e9},
                'fit_yvals': {'data': ampls[freqs < mid_freq, 0]},
                'fit_yvals_stderr': ampls[freqs < mid_freq, 1],
                'guess_pars': guess_pars}

            # fit right range
            model = lmfit.Model(fit_func)
            guess_pars = model.make_params(a=1, b=1, c=0)
            self.fit_dicts[f'amplitude_fit_right_{qbn}'] = {
                'fit_fn': fit_func,
                'fit_xvals': {'x': freqs[freqs >= mid_freq]/1e9},
                'fit_yvals': {'data': ampls[freqs >= mid_freq, 0]},
                'fit_yvals_stderr': ampls[freqs >= mid_freq, 1],
                'guess_pars': guess_pars}

            fit_dict_keys += [f'amplitude_fit_left_{qbn}',
                              f'amplitude_fit_right_{qbn}']
        return fit_dict_keys

    def prepare_plots(self):
        if self.get_param_value('plot_all_traces', True):
            super().prepare_plots()
        if self.do_fitting:
            for qbn in self.qb_names:
                base_plot_name = f'Rabi_amplitudes_{qbn}_{self.data_to_fit[qbn]}'
                title = f'{self.raw_data_dict["timestamp"]} ' \
                        f'{self.raw_data_dict["measurementstring"]}\n{qbn}'
                plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
                plotsize = (plotsize[0], plotsize[0]/1.25)
                param = [p for p in self.mospm[qbn] if 'freq' in p][0]
                xlabel = self.sp.get_sweep_params_property('label', 1, param)
                xunit = self.sp.get_sweep_params_property('unit', 1, param)
                lo_freqsX = self.get_param_value('allowed_lo_freqs')

                # plot upper sideband
                fit_dict = self.fit_dicts[f'amplitude_fit_left_{qbn}']
                fit_res = fit_dict['fit_res']
                xmin = min(fit_dict['fit_xvals']['x'])
                self.plot_dicts[f'{base_plot_name}_left_data'] = {
                    'plotfn': self.plot_line,
                    'fig_id': base_plot_name,
                    'plotsize': plotsize,
                    'xvals': fit_dict['fit_xvals']['x'],
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': fit_dict['fit_yvals']['data'],
                    'ylabel': '$\\pi$-pulse amplitude, $A$',
                    'yunit': 'V',
                    'setlabel': f'USB, LO at {np.min(lo_freqsX)/1e9:.3f} GHz',
                    'title': title,
                    'linestyle': 'none',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'yerr':  fit_dict['fit_yvals_stderr'],
                    'color': 'C0'
                }

                self.plot_dicts[f'{base_plot_name}_left_fit'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'USB quadratic fit',
                    'color': 'C0',
                    'do_legend': True,
                    # 'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                # plot lower sideband
                fit_dict = self.fit_dicts[f'amplitude_fit_right_{qbn}']
                fit_res = fit_dict['fit_res']
                xmax = max(fit_dict['fit_xvals']['x'])
                self.plot_dicts[f'{base_plot_name}_right_data'] = {
                    'plotfn': self.plot_line,
                    'fig_id': base_plot_name,
                    'xvals': fit_dict['fit_xvals']['x'],
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': fit_dict['fit_yvals']['data'],
                    'ylabel': '$\\pi$-pulse amplitude, $A$',
                    'yunit': 'V',
                    'setlabel': f'LSB, LO at {np.max(lo_freqsX)/1e9:.3f} GHz',
                    'title': title,
                    'linestyle': 'none',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'yerr':  fit_dict['fit_yvals_stderr'],
                    'color': 'C1'
                }

                self.plot_dicts[f'{base_plot_name}_right_fit'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'LSB quadratic fit',
                    'color': 'C1',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                # max ch amp line
                drive_ch = self.raw_data_dict[f'drive_ch_{qbn}']
                pd = self.get_data_from_timestamp_list({
                    f'ch_amp': f'Instrument settings.Pulsar.{drive_ch}_amp'})
                self.plot_dicts[f'ch_amp_line_{qbn}'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': pd['ch_amp'],
                    'xmin': xmax,
                    'xmax': xmin,
                    'colors': 'k'}


class ThermalPopulationAnalysis(RabiAnalysis):

    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            s = 'Instrument settings.'+qbn
            params_dict[f'ge_freq_'+qbn] = \
                s + '.ge_freq'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def _get_default_data_to_fit(self):
        return {qbn: 'pf' for qbn in self.qb_names}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()

        for qbn in self.qb_names:
            # k is of the form cos_fit_qbn_i if TwoD else cos_fit_qbn
            # replace k with qbn_i or qbn
            fit_res_th = self.fit_dicts[f'cos_fit_{qbn}_0']['fit_res']
            fit_res_prep = self.fit_dicts[f'cos_fit_{qbn}_1']['fit_res']
            self.proc_data_dict['analysis_params_dict'][qbn] = \
                self.get_thermal_population(qbn, fit_res_th, fit_res_prep)
        self.save_processed_data(key='analysis_params_dict')

    def get_thermal_population(self, qbn, fit_res_th, fit_res_prep):
        ratio = fit_res_th.best_values['amplitude'] \
                    / fit_res_prep.best_values['amplitude']
        peth = ratio / (1 + ratio)
        ge_freq = self.raw_data_dict[f'ge_freq_'+qbn]
        # We set E_g = 0 and assume p_fth = 0. This implies
        # 1 = p_gth + p_eth + p_fth = 1/Z exp(0) + p_eth = 1/Z + p_eth,
        # where Z is the partition function, Z=\sum_j exp(- E_j/(k_B T)).
        # => Z = 1/(1-p_eth) => p_eth = (1 - p_eth) * exp(- h * f_ge / (k * T))
        # and therefore:
        T = ge_freq * sp.constants.h / (sp.constants.k * np.log(1/peth-1))
        return {'peth': peth, 'temperature': T}

    def prepare_plots(self):
        if self.do_fitting:
            for k, fit_dict in self.fit_dicts.items():
                # k is of the form cos_fit_qbn_i if TwoD else cos_fit_qbn
                # replace k with qbn_i or qbn
                k = k.replace('cos_fit_', '')
                # split into qbn and i. (k + '_') is needed because if k = qbn
                # doing k.split('_') will only have one output and assignment to
                # two variables will fail.
                qbn, i = (k + '_').split('_')[:2]
                sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points']
                first_sweep_param = self.get_first_sweep_param(
                    qbn, dimension=1)
                if len(i) and first_sweep_param is not None:
                    # TwoD
                    label, unit, vals = first_sweep_param
                    title_suffix = (f'{i}: {label} = ' + ' '.join(
                        SI_val_to_msg_str(vals[int(i)], unit,
                                          return_type=lambda x : f'{x:0.4f}')))
                    daa = self.metadata.get('drive_amp_adaptation', {}).get(
                        qbn, None)
                    if daa is not None:
                        sweep_points = sweep_points * daa[int(i)]
                else:
                    # OneD
                    title_suffix = ''
                fit_res = fit_dict['fit_res']
                base_plot_name = f'Rabi_{k}_{self.data_to_fit[qbn]}'
                dtf = self.proc_data_dict['data_to_fit'][qbn]
                plot_cal_pts=int(i) if i != '' else True
                data=dtf[int(i)] if i != '' else dtf
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=data if plot_cal_pts else data[:-self.num_cal_points],
                    sweep_points=sweep_points if plot_cal_pts else sweep_points[:-self.num_cal_points],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn, TwoD=False,
                    title_suffix=title_suffix,
                    plot_cal_points=plot_cal_pts,
                )

                self.plot_dicts['fit_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'cosine fit',
                    'color': 'r',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                ana_params_qb = self.proc_data_dict['analysis_params_dict'][qbn]
                textstr = ('amplitude = {:.2f} %'.format(
                    fit_dict['fit_res'].best_values['amplitude'] * 1e2) +
                           '\noffset = {:.2f} %'.format(
                    fit_dict['fit_res'].best_values['offset'] * 1e2) +
                           '\n$P_{e, th.}$' + ' = {:.2f} %'.format(
                    ana_params_qb['peth'] * 1e2) +
                           '\n$T$ = {:.2f} mK'.format(
                    ana_params_qb['temperature'] * 1e3))
                self.plot_dicts['text_msg_' + k] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class T1Analysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        self.default_options['base_plot_name'] = 'T1'
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_T1_'+qbn] = \
                s + ('.T1' if trans_name == 'ge' else f'.T1_{trans_name}')
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        def add_fit_dict(qbn, data, key):
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            exp_decay_mod = lmfit.Model(fit_mods.ExpDecayFunc)
            guess_pars = fit_mods.exp_dec_guess(
                model=exp_decay_mod, data=data, t=sweep_points)
            guess_pars['amplitude'].vary = True
            guess_pars['tau'].vary = True
            if self.get_param_value('vary_offset', False):
                guess_pars['offset'].vary = True
            else:
                guess_pars['offset'].value = 0
                guess_pars['offset'].vary = False
            self.set_user_guess_pars(guess_pars)
            self.fit_dicts[key] = {
                'fit_fn': exp_decay_mod.func,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

        for qbn in self.qb_names:
            all_data = self.proc_data_dict['data_to_fit'][qbn]
            if self.get_param_value('TwoD'):
                for i, data in enumerate(all_data):
                    key = f'exp_decay_{qbn}_{i}'
                    add_fit_dict(qbn, data, key,)
            else:
                add_fit_dict(qbn, all_data, 'exp_decay_' + qbn)


    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for k, fit_dict in self.fit_dicts.items():
            # k is of the form exp_decay_qbn_i if TwoD else exp_decay_qbn
            # replace k with qbn_i or qbn
            k = k.replace('exp_decay_', '')
            fit_res = fit_dict['fit_res']
            for par in fit_res.params:
                if fit_res.params[par].stderr is None:
                    log.warning(f'Stderr for {par} is None. Setting it to 0.')
                    fit_res.params[par].stderr = 0
            # stores as qbn if OneD or qbn_i if TwoD
            self.proc_data_dict['analysis_params_dict'][k] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][k]['T1'] = \
                fit_res.best_values['tau']
            self.proc_data_dict['analysis_params_dict'][k]['T1_stderr'] = \
                fit_res.params['tau'].stderr
        self.save_processed_data(key='analysis_params_dict')


    def prepare_plots(self):
        super().prepare_plots()
        bpn = self.get_param_value('base_plot_name')

        if self.do_fitting:
            for k, fit_dict in self.fit_dicts.items():
                # k is of the form exp_decay_qbn_i if TwoD else exp_decay_qbn
                # replace k with qbn_i or qbn
                k = k.replace('exp_decay_', '')
                # split into qbn and i. (k + '_') is needed because if k = qbn
                # doing k.split('_') will only have one output and assignment to
                # two variables will fail.
                qbn, i = (k + '_').split('_')[:2]
                sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points']
                first_sweep_param = self.get_first_sweep_param(
                    qbn, dimension=1)
                if len(i) and first_sweep_param is not None:
                    # TwoD
                    label, unit, vals = first_sweep_param
                    title_suffix = (f'{i}: {label} = ' + ' '.join(
                        SI_val_to_msg_str(vals[int(i)], unit,
                                          return_type=lambda x: f'{x:0.4f}')))
                else:
                    # OneD
                    title_suffix = ''
                fit_res = fit_dict['fit_res']
                base_plot_name = f'{bpn}_{k}_{self.data_to_fit[qbn]}'
                dtf = self.proc_data_dict['data_to_fit'][qbn]
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=dtf[int(i)] if i != '' else dtf,
                    sweep_points=sweep_points,
                    plot_name_suffix=qbn + 'fit',
                    qb_name=qbn, TwoD=False,
                    title_suffix=title_suffix
                )
                self.plot_dicts['fit_' + k] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'exp decay fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}
                trans_name = self.get_transition_name(qbn)
                old_T1_val = self.raw_data_dict[f'{trans_name}_T1_' + qbn]
                # FIXME: the following condition is always False, isn't it?
                if old_T1_val != old_T1_val:
                    old_T1_val = 0  # FIXME: explain why
                T1_dict = self.proc_data_dict['analysis_params_dict']
                textstr = '$T_1$ = {:.2f} $\mu$s'.format(
                    T1_dict[k]['T1'] * 1e6) \
                          + ' $\pm$ {:.2f} $\mu$s'.format(
                    T1_dict[k]['T1_stderr'] * 1e6) \
                          + '\nold $T_1$ = {:.2f} $\mu$s'.format(
                    old_T1_val * 1e6)
                self.plot_dicts['text_msg_' + k] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class RamseyAnalysis(MultiQubit_TimeDomain_Analysis, ArtificialDetuningMixin):
    """
    Analysis for a Ramsey measurement.

    Parameters recognized in the options_dict:
     - artificial_detuning_dict (dict; default: None): has the form
        {qbn: artificial detuning value}
    - artificial_detuning (float or dict; default: None): accepted parameter
        for legacy reasons. Can be the same as artificial_detuning_dict or just
        a single value which will be used for all qubits.
    - fit_gaussian_decay (bool; default: True): whether to fit with a Gaussian
        envelope for the oscillations in addition to the exponential decay
        envelope.
    Note: if cal points have been measured, the fit amplitude is fixed to 0.5,
        otherwise it is optimised.
    """
    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_freq_'+qbn] = s+f'.{trans_name}_freq'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def prepare_fitting(self):
        if self.get_param_value('fit_gaussian_decay', default_value=True):
            self.fit_keys = ['exp_decay_', 'gauss_decay_']
        else:
            self.fit_keys = ['exp_decay_']
        self.fit_dicts = OrderedDict()
        def add_fit_dict(qbn, data, fit_keys):
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            for i, key in enumerate(fit_keys):
                exp_damped_decay_mod = lmfit.Model(fit_mods.ExpDampOscFunc)
                guess_pars = fit_mods.exp_damp_osc_guess(
                    model=exp_damped_decay_mod, data=data, t=sweep_points,
                    n_guess=i+1)
                guess_pars['amplitude'].value = 0.5
                if len(self.options_dict.get('cal_states', [])):
                    # If there are cal states, we expect a 0.5 amplitude
                    guess_pars['amplitude'].vary = False
                else:
                    # If no cal states, the oscillation amplitude in the IQ
                    # plane depends on readout
                    guess_pars['amplitude'].vary = True
                guess_pars['frequency'].vary = True
                guess_pars['tau'].vary = True
                guess_pars['tau'].min = 0
                guess_pars['phase'].vary = True
                guess_pars['n'].vary = False
                guess_pars['oscillation_offset'].vary = \
                    'f' in self.data_to_fit[qbn]
                guess_pars['exponential_offset'].vary = True
                self.set_user_guess_pars(guess_pars)
                self.fit_dicts[key] = {
                    'fit_fn': exp_damped_decay_mod .func,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

        for qbn in self.qb_names:
            all_data = self.proc_data_dict['data_to_fit'][qbn]
            if self.get_param_value('TwoD'):
                for i, data in enumerate(all_data):
                    fit_keys = [f'{fk}{qbn}_{i}' for fk in self.fit_keys]
                    add_fit_dict(qbn, data, fit_keys)
            else:
                fit_keys = [f'{fk}{qbn}' for fk in self.fit_keys]
                add_fit_dict(qbn, all_data, fit_keys)

    def analyze_fit_results(self):
        # get _get_artificial_detuning_dict
        self.artificial_detuning_dict = self.get_artificial_detuning_dict()
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for k, fit_dict in self.fit_dicts.items():
            # k is of the form fot_type_qbn_i if TwoD else fit_type_qbn
            split_key = k.split('_')
            fit_type = '_'.join(split_key[:2])
            qbn = split_key[2]
            if len(split_key[2:]) == 1:
                outer_key = qbn
            else:
                # TwoD: out_key = qbn_i
                outer_key = '_'.join(split_key[2:])

            if outer_key not in self.proc_data_dict['analysis_params_dict']:
                self.proc_data_dict['analysis_params_dict'][outer_key] = \
                    OrderedDict()
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type] = \
                OrderedDict()

            fit_res = fit_dict['fit_res']
            for par in fit_res.params:
                if fit_res.params[par].stderr is None:
                    log.warning(f'Stderr for {par} is None. Setting it to 0.')
                    fit_res.params[par].stderr = 0

            trans_name = self.get_transition_name(qbn)
            old_qb_freq = self.raw_data_dict[f'{trans_name}_freq_'+qbn]
            # FIXME: the following condition is always False, isn't it?
            if old_qb_freq != old_qb_freq:
                old_qb_freq = 0  # FIXME: explain why
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'old_qb_freq'] = old_qb_freq
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'new_qb_freq'] = old_qb_freq + \
                                 self.artificial_detuning_dict[qbn] - \
                                 fit_res.best_values['frequency']
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'new_qb_freq_stderr'] = fit_res.params['frequency'].stderr
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'T2_star'] = fit_res.best_values['tau']
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'T2_star_stderr'] = fit_res.params['tau'].stderr
            self.proc_data_dict['analysis_params_dict'][outer_key][fit_type][
                'artificial_detuning'] = self.artificial_detuning_dict[qbn]

        hdf_group_name_suffix = self.get_param_value(
            'hdf_group_name_suffix', '')
        self.save_processed_data(key='analysis_params_dict' +
                                     hdf_group_name_suffix)

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            apd = self.proc_data_dict['analysis_params_dict']
            for outer_key, ramsey_pars_dict in apd.items():
                if outer_key in ['qubit_frequencies', 'reparking_params',
                                 'residual_ZZs']:
                    # This is only for ReparkingRamseyAnalysis and
                    # ResidualZZAnalysis. It is handled those classes directly.
                    continue
                # outer_key is of the form qbn_i if TwoD else qbn.
                # split into qbn and i. (outer_key + '_') is needed because if
                # outer_key = qbn doing outer_key.split('_') will only have one
                # output and assignment to two variables will fail.
                qbn, ii = (outer_key + '_').split('_')[:2]
                sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points']
                first_sweep_param = self.get_first_sweep_param(
                    qbn, dimension=1)
                if len(ii) and first_sweep_param is not None:
                    # TwoD
                    label, unit, vals = first_sweep_param
                    title_suffix = (f'{ii}: {label} = ' + ' '.join(
                        SI_val_to_msg_str(vals[int(ii)], unit,
                                          return_type=lambda x: f'{x:0.1f}')))
                    daa = self.metadata.get('drive_amp_adaptation', {}).get(
                        qbn, None)
                    if daa is not None:
                        sweep_points = sweep_points * daa[int(ii)]
                else:
                    # OneD
                    title_suffix = ''
                base_plot_name = f'Ramsey_{outer_key}_{self.data_to_fit[qbn]}'
                dtf = self.proc_data_dict['data_to_fit'][qbn]
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=dtf[int(ii)] if ii != '' else dtf,
                    sweep_points=sweep_points,
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn, TwoD=False,
                    title_suffix=title_suffix)

                exp_dec_k = self.fit_keys[0][:-1]
                old_qb_freq = ramsey_pars_dict[exp_dec_k]['old_qb_freq']
                textstr = ''
                T2_star_str = ''

                for i, fit_type in enumerate(ramsey_pars_dict):
                    fit_res = self.fit_dicts[f'{fit_type}_{outer_key}']['fit_res']
                    self.plot_dicts[f'fit_{outer_key}_{fit_type}'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res,
                        'setlabel': 'exp decay fit' if i == 0 else
                            'gauss decay fit',
                        'do_legend': True,
                        'color': 'r' if i == 0 else 'C4',
                        'legend_bbox_to_anchor': (1, -0.15),
                        'legend_pos': 'upper right'}

                    if i != 0:
                        textstr += '\n'
                    textstr += \
                        ('$f_{{qubit \_ new \_ {{{key}}} }}$ = '.format(
                            key=('exp' if i == 0 else 'gauss')) +
                            '{:.6f} GHz '.format(
                            ramsey_pars_dict[fit_type]['new_qb_freq']*1e-9) +
                                '$\pm$ {:.3f} MHz '.format(
                            ramsey_pars_dict[fit_type][
                                'new_qb_freq_stderr']*1e-6))
                    T2_star_str += \
                        ('\n$T_{{2,{{{key}}} }}^\star$ = '.format(
                            key=('exp' if i == 0 else 'gauss')) +
                            '{:.2f} $\mu$s'.format(
                            fit_res.params['tau'].value*1e6) +
                            '$\pm$ {:.2f} $\mu$s'.format(
                            fit_res.params['tau'].stderr*1e6))

                textstr += '\n$f_{qubit \_ old}$ = '+'{:.6f} GHz '.format(
                    old_qb_freq*1e-9)
                art_det = ramsey_pars_dict[exp_dec_k][
                              'artificial_detuning']*1e-6
                delta_f = (ramsey_pars_dict[exp_dec_k]['new_qb_freq'] -
                           old_qb_freq)*1e-6
                textstr += ('\n$\Delta f$ = {:.4f} MHz '.format(delta_f) +
                            '$\pm$ {:.3f} kHz'.format(
                    self.fit_dicts[f'{exp_dec_k}_{outer_key}']['fit_res'].params[
                        'frequency'].stderr*1e-3) +
                    '\n$f_{Ramsey}$ = '+'{:.4f} MHz $\pm$ {:.3f} kHz'.format(
                    self.fit_dicts[f'{exp_dec_k}_{outer_key}']['fit_res'].params[
                        'frequency'].value*1e-6,
                    self.fit_dicts[f'{exp_dec_k}_{outer_key}']['fit_res'].params[
                        'frequency'].stderr*1e-3))
                textstr += T2_star_str
                textstr += '\nartificial detuning = {:.2f} MHz'.format(art_det)

                color = 'k'
                if np.abs(delta_f) > np.abs(art_det):
                    # We don't want this: if the qubit detuning is larger than
                    # the artificial detuning, the sign of the qubit detuning
                    # cannot be determined from a single Ramsey measurement.
                    # Save a warning image and highlight in red
                    # the Delta f and artificial detuning rows in textstr
                    self._warning_message += (f'\nQubit {qbn} frequency change '
                                        f'({np.abs(delta_f):.5f} MHz) is larger'
                                        f' than the artificial detuning of '
                                        f'{art_det:.5f} MHz. In this case, the '
                                        f'sign of the qubit detuning cannot be '
                                        f'determined from a single Ramsey '
                                        f'measurement.')
                    self._raise_warning_image = True
                    textstr = textstr.split('\n')
                    color = ['black']*len(textstr)
                    idx = [i for i, s in enumerate(textstr) if 'Delta f' in s][0]
                    color[idx] = 'red'
                    idx = [i for i, s in enumerate(textstr) if
                           'artificial detuning' in s][0]
                    color[idx] = 'red'

                self.plot_dicts['text_msg_' + outer_key] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': -0.025,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'color': color,
                    'plotfn': self.plot_text,
                    'text_string': textstr}

                if 'pca' not in self.rotation_type[qbn].lower():
                    self.plot_dicts['half_hline_' + outer_key] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_hlines,
                        'y': 0.5,
                        'xmin': sweep_points[0],
                        'xmax': sweep_points[-1],
                        'colors': 'gray'}


class ReparkingRamseyAnalysis(RamseyAnalysis):

    def analyze_fit_results(self):
        super().analyze_fit_results()
        freqs = OrderedDict()

        if self.get_param_value('freq_from_gaussian_fit', False):
            self.fit_type = self.fit_keys[1][:-1]
        else:
            self.fit_type = self.fit_keys[0][:-1]

        apd = self.proc_data_dict['analysis_params_dict']
        for qbn in self.qb_names:
            freqs[qbn] = \
                {'val': np.array([d[self.fit_type]['new_qb_freq']
                                     for k, d in apd.items() if qbn in k]),
                 'stderr': np.array([d[self.fit_type]['new_qb_freq_stderr']
                                     for k, d in apd.items() if qbn in k])}
        self.proc_data_dict['analysis_params_dict']['qubit_frequencies'] = freqs

        fit_dict_keys = self.prepare_fitting_qubit_freqs()
        self.run_fitting(keys_to_fit=fit_dict_keys)

        self.proc_data_dict['analysis_params_dict']['reparking_params'] = {}
        for qbn in self.qb_names:
            fit_dict = self.fit_dicts[f'frequency_fit_{qbn}']
            fit_res = fit_dict['fit_res']
            new_ss_freq = fit_res.best_values['f0']
            new_ss_volt = fit_res.best_values['V0']

            par_name = \
                [p for p in self.proc_data_dict['sweep_points_2D_dict'][qbn]
                 if 'offset' not in p][0]
            voltages = self.sp.get_sweep_params_property('values', 1, par_name)
            if new_ss_volt < min(voltages) or new_ss_volt > max(voltages):
                # if the fitted voltage is outside the sweep points range take
                # the max or min of range depending on where the fitted point is
                idx = np.argmin(voltages) if new_ss_volt < min(voltages) else \
                    np.argmax(voltages)
                new_ss_volt = min(voltages) if new_ss_volt < min(voltages) else \
                        max(voltages)
                freqs = self.proc_data_dict['analysis_params_dict'][
                    'qubit_frequencies'][qbn]['val']
                new_ss_freq = freqs[idx]

                log.warning(f"New sweet spot voltage suggested by fitting "
                            f"is {fit_res.best_values['V0']:.6f} and exceeds "
                            f"the voltage range [{min(voltages):.6f}, "
                            f"{max(voltages):.6f}] that is swept. New sweet "
                            f"spot voltage set to {new_ss_volt:.6f}.")

            self.proc_data_dict['analysis_params_dict'][
                'reparking_params'][qbn] = {
                    'new_ss_vals': {'ss_freq': new_ss_freq,
                                    'ss_volt': new_ss_volt},
                    'fitted_vals': {'ss_freq': fit_res.best_values['f0'],
                                    'ss_volt': fit_res.best_values['V0']}}

        self.save_processed_data(key='analysis_params_dict')

    def prepare_fitting_qubit_freqs(self):
        fit_dict_keys = []
        ss_type = self.get_param_value('sweet_spot_type')
        for qbn in self.qb_names:
            freqs = self.proc_data_dict['analysis_params_dict'][
                'qubit_frequencies'][qbn]
            par_name = \
                [p for p in self.proc_data_dict['sweep_points_2D_dict'][qbn]
                 if 'offset' not in p][0]
            voltages, _, label = self.sp.get_sweep_params_description(par_name,
                                                                      1)
            fit_func = lambda V, V0, f0, fv: f0 - fv * (V - V0)**2
            model = lmfit.Model(fit_func)

            if ss_type is None:
                # define secant from outermost points to check
                # convexity and decide for USS or LSS
                secant_gradient = ((freqs['val'][-1] - freqs['val'][0])
                                   / (voltages[-1] - voltages[0]))
                secant = lambda x: secant_gradient * x + freqs['val'][-1] \
                                   - secant_gradient * voltages[-1]
                # compute convexity as trapezoid integral of difference to
                # secant
                delta_secant = np.array(freqs['val'] - secant(voltages))
                convexity = np.sum((delta_secant[:-1] + delta_secant[1:]) / 2
                                   * (voltages[1:] - voltages[:-1]))
                self.fit_uss = convexity >= 0
            else:
                self.fit_uss = ss_type == 'upper'

            # set initial values of fitting parameters depending on USS or LSS
            if self.fit_uss:  # USS
                guess_pars_dict = {'V0': voltages[np.argmax(freqs['val'])],
                                   'f0': np.max(np.array(freqs['val'])),
                                   'fv': 2.5e9}
            else:  # LSS
                guess_pars_dict = {'V0': voltages[np.argmin(freqs['val'])],
                                   'f0': np.min(np.array(freqs['val'])),
                                   'fv': -2.5e9}
            guess_pars = model.make_params(**guess_pars_dict)
            self.fit_dicts[f'frequency_fit_{qbn}'] = {
                'fit_fn': fit_func,
                'fit_xvals': {'V': voltages},
                'fit_yvals': {'data': freqs['val']},
                'fit_yvals_stderr': freqs['stderr'],
                'guess_pars': guess_pars}
            fit_dict_keys += [f'frequency_fit_{qbn}']
        return fit_dict_keys

    def prepare_plots(self):
        if self.get_param_value('plot_all_traces', True):
            super().prepare_plots()
        if self.do_fitting:
            current_voltages = self.get_param_value('current_voltages', {})
            for qbn in self.qb_names:
                base_plot_name = f'reparking_{qbn}_{self.data_to_fit[qbn]}'
                title = f'{self.raw_data_dict["timestamp"]} ' \
                        f'{self.raw_data_dict["measurementstring"]}\n{qbn}'
                plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
                plotsize = (plotsize[0], plotsize[0]/1.25)
                par_name = \
                    [p for p in self.proc_data_dict['sweep_points_2D_dict'][qbn]
                     if 'offset' not in p][0]
                voltages, xunit, xlabel = self.sp.get_sweep_params_description(
                    par_name, 1)
                fit_dict = self.fit_dicts[f'frequency_fit_{qbn}']
                fit_res = fit_dict['fit_res']
                self.plot_dicts[base_plot_name] = {
                    'plotfn': self.plot_line,
                    'fig_id': base_plot_name,
                    'plotsize': plotsize,
                    'xvals': fit_dict['fit_xvals']['V'],
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': fit_dict['fit_yvals']['data'],
                    'ylabel': 'Qubit frequency, $f$',
                    'yunit': 'Hz',
                    'setlabel': 'Data',
                    'title': title,
                    'linestyle': 'none',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left',
                    'yerr':  fit_dict['fit_yvals_stderr'],
                    'color': 'C0'
                }

                self.plot_dicts[f'{base_plot_name}_fit'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'Fit',
                    'color': 'C0',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                # old qb freq is the same for all keys in
                # self.proc_data_dict['analysis_params_dict'] so take qbn_0
                old_qb_freq = self.proc_data_dict['analysis_params_dict'][
                    f'{qbn}_0'][self.fit_type]['old_qb_freq']
                # new ss values
                ss_vals = self.proc_data_dict['analysis_params_dict'][
                    'reparking_params'][qbn]['new_ss_vals']
                textstr = \
                    "SS frequency: " \
                        f"{ss_vals['ss_freq']/1e9:.6f} GHz " \
                    f"\nSS DC voltage: " \
                        f"{ss_vals['ss_volt']:.6f} V " \
                    f"\nPrevious SS frequency: {old_qb_freq/1e9:.6f} GHz "
                if qbn in current_voltages:
                    old_voltage = current_voltages[qbn]
                    textstr += f"\nPrevious SS DC voltage: {old_voltage:.6f} V"
                self.plot_dicts[f'{base_plot_name}_text'] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': -0.1,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}

                self.plot_dicts[f'{base_plot_name}_marker'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': [ss_vals['ss_volt']],
                    'yvals': [ss_vals['ss_freq']],
                    'color': 'r',
                    'marker': 'o',
                    'line_kws': {'markersize': 10},
                    'linestyle': ''}


class ResidualZZAnalysis(RamseyAnalysis):
    """
    Analysis for a ResidualZZ measurement.

    Extracts the residual ZZ couling by comparing the frequencies of two Ramsey
    measurements and adds one additional plot for each residual ZZ measurement
    to display the extracted residual ZZ couling.
    """

    def __init__(self, echo=False, **kwargs):
        """
        Args:
            echo (bool, optional): Whether the Ramsey sequence included an echo
                pulse or not. This is relevant for the analysis as this causes a
                factor of 1/2 in the frequency difference between the two Ramsey
                measurements compared to the measurements without echo pulse.
                Defaults to False.
        """
        self.echo = echo
        super().__init__(**kwargs)

    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        task_list = self.get_param_value('task_list', default_value=[])
        for task in task_list:
            params_dict[f'qbc_of_' + task['qb']] = task['qbc']
        self.raw_data_dict.update(params_dict)

    def analyze_fit_results(self):
        super().analyze_fit_results()
        residual_ZZs_dicts = OrderedDict()
        for qbn in self.qb_names:
            residual_ZZs_dicts[qbn] = OrderedDict()
            for fk in self.fit_keys:
                fit_res_base = self.fit_dicts[f'{fk}{qbn}_0']['fit_res']
                fit_res_prep = self.fit_dicts[f'{fk}{qbn}_1']['fit_res']
                residual_ZZs_dicts[qbn][f'{fk}'] = \
                    self.get_residual_zz(fit_res_base, fit_res_prep)
        self.proc_data_dict['analysis_params_dict']['residual_ZZs'] = \
            residual_ZZs_dicts
        hdf_group_name_suffix = self.get_param_value(
            'hdf_group_name_suffix', '')
        self.save_processed_data(key='analysis_params_dict' +
                                     hdf_group_name_suffix)

    def get_residual_zz(self, fit_res_base, fit_res_prep):
        """Compute the residual ZZ coupling from the two Ramsey fits.

        Args:
            fit_res_base (lmfit.ModelResult): Ramsey fit for the measurement
                with the control qubit in the ground state the whole time.
            fit_res_prep (lmfit.ModelResult): Ramsey fit for the measurement
                with the control qubit excited.

        Returns:
            dict with the individual detunings ('freq_g', 'freq_e') and the
            extracted residual ZZ ('shift_Hz') and its standard error
            ('shift_Hz_stderr').
        """
        freq_g = fit_res_base.best_values['frequency']
        freq_e = fit_res_prep.best_values['frequency']
        shift = freq_e - freq_g
        stderr = np.sqrt(fit_res_prep.params['frequency'].stderr**2
                         + fit_res_base.params['frequency'].stderr**2)
        # When the measurement was implemented using an echo pulse, the control
        # qubit only got excited during the second half of the Ramsey sequence.
        # The resulting shift between the fitted detuning frequencies therefore
        # corresponds to only half the residual ZZ coupling. This is compensated
        # by the following factor:
        x = 2 if self.echo else 1
        return {'freq_g': freq_g, 'freq_e': freq_e,
                'shift_Hz': x*shift, 'shift_Hz_stderr': x*stderr}

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            apd = self.proc_data_dict['analysis_params_dict']
            for qb, residual_zz_dict in apd['residual_ZZs'].items():
                qbc = self.raw_data_dict[f'qbc_of_{qb}']
                plot_name = f'ResidualZZ_{qb}_{qbc}'

                sweep_points = self.proc_data_dict['sweep_points_dict'][qb][
                    'sweep_points']
                dtf = self.proc_data_dict['data_to_fit'][qb]
                self.prepare_projected_data_plot(
                    fig_name=plot_name,
                    data=dtf[0],
                    data_label=f'{qbc} in g',
                    do_legend_data=False,
                    do_legend_cal_states=False,
                    sweep_points=sweep_points,
                    plot_name_suffix=qb+'fit0',
                    qb_name=qb, TwoD=False,
                    title_suffix="")
                self.prepare_projected_data_plot(
                    fig_name=plot_name,
                    data=dtf[1][:-self.num_cal_points],
                    plot_cal_points=False,
                    data_label=f'{qbc} in e',
                    do_legend_data=False,
                    do_legend_cal_states=False,
                    sweep_points=sweep_points[:-self.num_cal_points],
                    plot_name_suffix=qb+'fit1',
                    qb_name=qb, TwoD=False,
                    title_suffix="")

                textstr = ''
                for fk, residual_zz in residual_zz_dict.items():
                    key = fk.split('_')[0]
                    textstr += f'$\Delta f_{{0_{{{key}}}}}$ = ' \
                               + f'{1e-3*residual_zz["freq_g"]:.3f} kHz\n'
                    textstr += f'$\Delta f_{{1_{{{key}}}}}$ = ' \
                               + f'{1e-3*residual_zz["freq_e"]:.3f} kHz\n'
                    textstr += f'$\Omega_{{ZZ_{{{key}}}}}/2\pi$ = ' \
                               + f'{1e-3*residual_zz["shift_Hz"]:.3f} $\pm$ ' \
                               + f'{1e-3*residual_zz["shift_Hz_stderr"]:.3f} ' \
                               + f'kHz\n'
                textstr += f'target: {qb}, control: {qbc}'
                self.plot_dicts['text_msg'] = {
                    'fig_id': plot_name,
                    'ypos': -0.2,
                    'xpos': -0.025,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'color': 'black',
                    'plotfn': self.plot_text,
                    'text_string': textstr,}


            # helper variable to prevent having labels twice in the legend
            labels_set = False
            for outer_key, ramsey_pars_dict in apd.items():
                if outer_key in ['qubit_frequencies', 'reparking_params',
                                 'residual_ZZs']:
                    # We only care about the fit types saved in the apd.
                    continue
                # outer_key is of the form qbn_i if TwoD else qbn.
                # split into qbn and i. (outer_key + '_') is needed because if
                # outer_key = qbn doing outer_key.split('_') will only have one
                # output and assignment to two variables will fail.
                qbn = (outer_key + '_').split('_')[0]

                qbc = self.raw_data_dict[f'qbc_of_{qbn}']
                plot_name = f'ResidualZZ_{qbn}_{qbc}'

                for i, fit_type in enumerate(ramsey_pars_dict):
                    fit_res = self.fit_dicts[f'{fit_type}_{outer_key}'][
                        'fit_res']
                    label = 'exp decay fit' if i == 0 else 'gauss decay fit'
                    self.plot_dicts[
                        f'ResidualZZ_fit_{outer_key}_{fit_type}'] = {
                            'fig_id': plot_name,
                            'plotfn': self.plot_fit,
                            'fit_res': fit_res,
                            'color': 'r' if i == 0 else 'C4',
                            'setlabel': label if not labels_set else '',
                        }
                    if not labels_set:
                        self.plot_dicts[
                            f'ResidualZZ_fit_{outer_key}_{fit_type}'].update({
                                'do_legend': True,
                                'legend_bbox_to_anchor': (1.0, -0.15),
                                'legend_pos': 'upper right',
                            })
                labels_set = True


class QScaleAnalysis(MultiQubit_TimeDomain_Analysis, PhaseErrorsAnalysisMixin):

    def extract_data(self):
        super().extract_data()
        self._extract_current_pulse_par_value()

    def process_data(self):
        for qbn in self.qb_names:
            sweep_points = deepcopy(self.proc_data_dict['sweep_points_dict'][
                                        qbn]['msmt_sweep_points'])
            # check if the sweep points are repeated 3 times as they have to be
            # for the qscale analysis:
            # Takes the first 3 entries and check if they are all the same or
            # different. Needed For backwards compatibility with
            # QudevTransmon.measure_qscale()  that does not (yet) use
            # SweepPoints class.
            unique_sp = np.unique(sweep_points[:3])
            if unique_sp.size > 1:
                if self.get_param_value('sp1d_filter') is not None:
                    log.warning('Passing sp1d_filter might not work '
                                'for this QScaleAnalysis because the sweep '
                                'points need to be repeated 3x for it to '
                                'match the data size.')
                else:
                    # repeat each sweep points 3x
                    self.default_options['sp1d_filter'] = \
                        lambda sp: np.repeat(sp, 3)
                    # update self.num_cal_points and
                    # self.proc_data_dict['sweep_points_dict']
                    self.create_sweep_points_dict()
                    self.get_num_cal_points()
                    self.update_sweep_points_dict()

        super().process_data()
        # Separate data and sweep points into those corresponding to the
        # xx, xy, xmy lines
        self.proc_data_dict['qscale_data'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['qscale_data'][qbn] = OrderedDict()
            sweep_points = deepcopy(self.proc_data_dict['sweep_points_dict'][
                                        qbn]['msmt_sweep_points'])
            data = self.proc_data_dict['data_to_fit'][qbn]
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xx'] = \
                sweep_points[0::3]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xy'] = \
                sweep_points[1::3]
            self.proc_data_dict['qscale_data'][qbn]['sweep_points_xmy'] = \
                sweep_points[2::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xx'] = \
                data[0::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xy'] = \
                data[1::3]
            self.proc_data_dict['qscale_data'][qbn]['data_xmy'] = \
                data[2::3]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        for qbn in self.qb_names:
            for msmt_label in ['_xx', '_xy', '_xmy']:
                sweep_points = self.proc_data_dict['qscale_data'][qbn][
                    'sweep_points' + msmt_label]
                data = self.proc_data_dict['qscale_data'][qbn][
                    'data' + msmt_label]

                # As a workaround for a weird bug letting crash the analysis
                # every second time, we do not use lmfit.models.ConstantModel
                # and lmfit.models.LinearModel, but create custom models.
                if msmt_label == '_xx':
                    model = lmfit.Model(lambda x, c: c)
                    guess_pars = model.make_params(c=np.mean(data))
                else:
                    model = lmfit.Model(lambda x, slope, intercept:
                                        slope * x + intercept)
                    slope = (data[-1] - data[0]) / \
                            (sweep_points[-1] - sweep_points[0])
                    intercept = data[-1] - slope * sweep_points[-1]
                    guess_pars = model.make_params(slope=slope,
                                                   intercept=intercept)
                self.set_user_guess_pars(guess_pars)
                key = 'fit' + msmt_label + '_' + qbn
                self.fit_dicts[key] = {
                    'fit_fn': model.func,
                    'fit_xvals': {'x': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        # The best qscale parameter is the point where all 3 curves intersect.
        threshold = 0.02
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            fitparams0 = self.fit_dicts['fit_xx'+'_'+qbn]['fit_res'].params
            fitparams1 = self.fit_dicts['fit_xy'+'_'+qbn]['fit_res'].params
            fitparams2 = self.fit_dicts['fit_xmy'+'_'+qbn]['fit_res'].params

            intercept_diff_mean = fitparams1['intercept'].value - \
                                  fitparams2['intercept'].value
            slope_diff_mean = fitparams2['slope'].value - \
                              fitparams1['slope'].value
            optimal_qscale = intercept_diff_mean/slope_diff_mean

            # Warning if Xpi/2Xpi line is not within +/-threshold of 0.5
            if (fitparams0['c'].value > (0.5 + threshold)) or \
                    (fitparams0['c'].value < (0.5 - threshold)):
                log.warning('The trace from the X90-X180 pulses is '
                            'NOT within ±{} of the expected value '
                            'of 0.5.'.format(threshold))
            # Warning if optimal_qscale is not within +/-threshold of 0.5
            y_optimal_qscale = optimal_qscale * fitparams2['slope'].value + \
                                 fitparams2['intercept'].value
            if (y_optimal_qscale > (0.5 + threshold)) or \
                    (y_optimal_qscale < (0.5 - threshold)):
                log.warning('The optimal qscale found gives a population '
                            'that is NOT within ±{} of the expected '
                            'value of 0.5.'.format(threshold))

            # Calculate standard deviation
            intercept_diff_std_squared = \
                fitparams1['intercept'].stderr**2 + \
                fitparams2['intercept'].stderr**2
            slope_diff_std_squared = \
                fitparams2['slope'].stderr**2 + fitparams1['slope'].stderr**2

            optimal_qscale_stderr = np.sqrt(
                intercept_diff_std_squared*(1/slope_diff_mean**2) +
                slope_diff_std_squared*(intercept_diff_mean /
                                        (slope_diff_mean**2))**2)

            self.proc_data_dict['analysis_params_dict'][qbn]['qscale'] = \
                optimal_qscale
            self.proc_data_dict['analysis_params_dict'][qbn][
                'qscale_stderr'] = optimal_qscale_stderr

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        color_dict = {'_xx': '#365C91',
                      '_xy': '#683050',
                      '_xmy': '#3C7541'}
        label_dict = {'_xx': r'$X_{\pi/2}X_{\pi}$',
                      '_xy': r'$X_{\pi/2}Y_{\pi}$',
                      '_xmy': r'$X_{\pi/2}Y_{-\pi}$'}
        for qbn in self.qb_names:
            base_plot_name = f'Qscale_{qbn}_{self.data_to_fit[qbn]}'
            for msmt_label in ['_xx', '_xy', '_xmy']:
                sweep_points = self.proc_data_dict['qscale_data'][qbn][
                    'sweep_points' + msmt_label]
                data = self.proc_data_dict['qscale_data'][qbn][
                    'data' + msmt_label]
                if msmt_label == '_xx':
                    plot_name = base_plot_name
                else:
                    plot_name = 'data' + msmt_label + '_' + qbn
                xlabel, xunit = self.get_xaxis_label_unit(qbn)
                self.plot_dicts[plot_name] = {
                    'plotfn': self.plot_line,
                    'xvals': sweep_points,
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'yvals': data,
                    'ylabel': self.get_yaxis_label(qb_name=qbn),
                    'yunit': '',
                    'setlabel': 'Data\n' + label_dict[msmt_label],
                    'title': (self.raw_data_dict['timestamp'] + ' ' +
                              self.raw_data_dict['measurementstring'] +
                              '\n' + qbn),
                    'linestyle': 'none',
                    'color': color_dict[msmt_label],
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}
                if msmt_label != '_xx':
                    self.plot_dicts[plot_name]['fig_id'] = base_plot_name

                if self.do_fitting:
                    # plot fit
                    xfine = np.linspace(sweep_points[0], sweep_points[-1], 1000)
                    fit_key = 'fit' + msmt_label + '_' + qbn
                    fit_res = self.fit_dicts[fit_key]['fit_res']
                    yvals = fit_res.model.func(xfine, **fit_res.best_values)
                    if not hasattr(yvals, '__iter__'):
                        yvals = np.array(len(xfine)*[yvals])
                    self.plot_dicts[fit_key] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_line,
                        'xvals': xfine,
                        'yvals': yvals,
                        'marker': '',
                        'setlabel': 'Fit\n' + label_dict[msmt_label],
                        'do_legend': True,
                        'color': color_dict[msmt_label],
                        'legend_bbox_to_anchor': (1, 0.5),
                        'legend_pos': 'center left'}

                    val = self.proc_data_dict['analysis_params_dict'][qbn][
                                'qscale']
                    stderr = self.proc_data_dict['analysis_params_dict'][qbn][
                                'qscale_stderr']
                    textstr = self.create_textstr(qbn, val, stderr)
                    self.plot_dicts['text_msg_' + qbn] = {
                        'fig_id': base_plot_name,
                        'ypos': -0.225,
                        'xpos': 0.0,
                        'horizontalalignment': 'left',
                        'verticalalignment': 'top',
                        'plotfn': self.plot_text,
                        'text_string': textstr}

            # plot cal points
            if self.num_cal_points != 0:
                for i, cal_pts_idxs in enumerate(
                        self.cal_states_dict[qbn].values()):
                    plot_dict_name = list(self.cal_states_dict[qbn])[i] + \
                                     '_' + qbn
                    self.plot_dicts[plot_dict_name] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_line,
                        'xvals': np.mean([
                            self.proc_data_dict['sweep_points_dict'][qbn]
                            ['cal_points_sweep_points'][cal_pts_idxs],
                            self.proc_data_dict['sweep_points_dict'][qbn]
                            ['cal_points_sweep_points'][cal_pts_idxs]],
                            axis=0),
                        'yvals': self.proc_data_dict[
                            'data_to_fit'][qbn][cal_pts_idxs],
                        'setlabel': list(self.cal_states_dict[qbn])[i],
                        'do_legend': True,
                        'legend_bbox_to_anchor': (1, 0.5),
                        'legend_pos': 'center left',
                        'linestyle': 'none',
                        'line_kws': {'color': self.get_cal_state_color(
                            list(self.cal_states_dict[qbn])[i])}}

                    self.plot_dicts[plot_dict_name + '_line'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_hlines,
                        'y': np.mean(
                            self.proc_data_dict[
                                'data_to_fit'][qbn][cal_pts_idxs]),
                        'xmin': self.proc_data_dict['sweep_points_dict'][
                            qbn]['sweep_points'][0],
                        'xmax': self.proc_data_dict['sweep_points_dict'][
                            qbn]['sweep_points'][-1],
                        'colors': 'gray'}


class EchoAnalysis(MultiQubit_TimeDomain_Analysis, ArtificialDetuningMixin):

    def __init__(self, *args, extract_only=False, **kwargs):
        """
        This class is different to the other single qubit calib analysis classes
        (Rabi, Ramsey, QScale, T1).
        The analysis for an Echo measurement is identical to the T1 analysis
        if no artificial_detuing was used, and identical to the Ramsey analysis
        if an artificial_detuning was used. Hence, this class contains the
        attribute self.echo_analysis which is an instance of either T1 or Ramsey
        analysis.
        """
        auto = kwargs.pop('auto', True)
        super().__init__(*args, auto=False, extract_only=extract_only, **kwargs)

        # get experimental metadata from file
        self.metadata = self.get_data_from_timestamp_list(
            {'md': 'Experimental Data.Experimental Metadata'})['md']

        # get _get_artificial_detuning_dict
        self.artificial_detuning_dict = self.get_artificial_detuning_dict(
            raise_error=False)

        # Decide whether to do a RamseyAnalysis or a T1Analysis
        self.run_ramsey = self.artificial_detuning_dict is not None and \
                any(list(self.artificial_detuning_dict.values()))

        # Define options_dict for call to RamseyAnalysis or T1Analysis
        options_dict = deepcopy(kwargs.pop('options_dict', dict()))
        options_dict['save_figs'] = False  # plots will be made by EchoAnalysis

        if self.run_ramsey:
            # artificial detuning was used and it is not 0
            self.echo_analysis = RamseyAnalysis(*args, auto=auto,
                                                extract_only=True,
                                                options_dict=options_dict,
                                                **kwargs)
        else:
            options_dict['vary_offset'] = True  # pe saturates at 0.5 not 0
            self.echo_analysis = T1Analysis(*args, auto=auto,
                                            extract_only=True,
                                            options_dict=options_dict,
                                            **kwargs)

        if auto:
            self.qb_names = self.echo_analysis.qb_names
            # Run analysis of this class
            super().run_analysis()

    def extract_data(self):
        """Skip for this class. Take raw_data_dict from self.echo_analysis
        which is needed for a check in the BaseDataAnalsis.save_fit_results."""
        self.raw_data_dict = self.echo_analysis.raw_data_dict
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            params_dict[f'{trans_name}_T2_{qbn}'] = \
                'Instrument settings.' + qbn + \
                ('.T2' if trans_name == 'ge' else f'.T2_{trans_name}')
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def process_data(self):
        """Skip for this class. All relevant processing is done in
        self.echo_analysis."""
        pass

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()

        # If OneD, qbn_idx will be of the form qb2
        # If TwoD, qbn_idx wil lbe of the form qb2_1
        for qbn_idx in (
                self.echo_analysis.proc_data_dict[
                    'analysis_params_dict'].keys()
        ):
            self.proc_data_dict['analysis_params_dict'][qbn_idx] = OrderedDict()
            params_dict = self.echo_analysis.proc_data_dict[
                'analysis_params_dict'][qbn_idx]
            if 'T1' in params_dict:
                self.proc_data_dict['analysis_params_dict'][qbn_idx][
                    'T2_echo'] = params_dict['T1']
                self.proc_data_dict['analysis_params_dict'][qbn_idx][
                    'T2_echo_stderr'] = params_dict['T1_stderr']
            else:
                self.proc_data_dict['analysis_params_dict'][qbn_idx][
                    'T2_echo'] = params_dict['exp_decay']['T2_star']
                self.proc_data_dict['analysis_params_dict'][qbn_idx][
                    'T2_echo_stderr'] = params_dict['exp_decay'][
                    'T2_star_stderr']
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        if self.do_fitting:
            # If OneD, qbn_idx will be of the form qb2
            # If TwoD, qbn_idx wil lbe of the form qb2_1
            for qbn_idx in (
                    self.echo_analysis.proc_data_dict[
                        'analysis_params_dict'].keys()
            ):
                qbn = (qbn_idx + "_").split("_")[0]
                # rename base plot
                figure_name = (f'Echo_{qbn_idx}_'
                               f'{self.echo_analysis.data_to_fit[qbn]}')
                echo_plot_key_t1 = [key for key in self.echo_analysis.plot_dicts
                                    if 'T1_'+qbn_idx in key]
                echo_plot_key_rm = [key for key in self.echo_analysis.plot_dicts
                                    if 'Ramsey_'+qbn_idx in key]
                if len(echo_plot_key_t1) != 0:
                    echo_plot_name = echo_plot_key_t1[0]
                elif len(echo_plot_key_rm) != 0:
                    echo_plot_name = echo_plot_key_rm[0]
                else:
                    raise ValueError('Neither T1 nor Ramsey plots were found.')

                self.echo_analysis.plot_dicts[echo_plot_name][
                    'legend_pos'] = 'upper right'
                self.echo_analysis.plot_dicts[echo_plot_name][
                    'legend_bbox_to_anchor'] = (1, -0.15)

                for plot_label in self.echo_analysis.plot_dicts:
                    if qbn_idx in plot_label:
                        if 'raw' not in plot_label and \
                                'projected' not in plot_label:
                            self.echo_analysis.plot_dicts[
                                plot_label]['fig_id'] = figure_name

                trans_name = self.get_transition_name(qbn)
                old_T2e_val = self.raw_data_dict[f'{trans_name}_T2_{qbn}']
                T2_dict = self.proc_data_dict['analysis_params_dict']
                textstr = 'Echo Measurement with'
                art_det = self.artificial_detuning_dict[qbn]*1e-6
                textstr += '\nartificial detuning = {:.2f} MHz'.format(art_det)
                textstr += '\n$T_2$ echo = {:.2f} $\mu$s'.format(
                    T2_dict[qbn_idx]['T2_echo']*1e6) \
                          + ' $\pm$ {:.2f} $\mu$s'.format(
                    T2_dict[qbn_idx]['T2_echo_stderr']*1e6) \
                          + '\nold $T_2$ echo = {:.2f} $\mu$s'.format(
                    old_T2e_val*1e6)

                self.echo_analysis.plot_dicts['text_msg_' + qbn_idx][
                    'text_string'] = textstr
                # Set text colour to black.
                # When the change in qubit frequency is larger than artificial
                # detuning, the qubit freq estimation is unreliable and the
                # RamseyAnalysis will alert the user by producing a
                # multi-coloured text string, in which case both the "color" and
                # the "text_string" entries of the plot dict are lists with the
                # same length. This warning is not relevant for the EchoAnalysis
                # since we do not use it to estimate the qubit frequency.
                # Not resetting the colour here will cause an error in the case
                # of multi-coloured text strings.
                self.echo_analysis.plot_dicts['text_msg_' + qbn_idx]['color'] \
                    = 'k'

    def plot(self, **kw):
        # Overload base method to run the method in echo_analysis
        self.echo_analysis.plot(key_list='auto')

    def save_figures(self, **kw):
        # Overload base method to run the method in echo_analysis
        self.echo_analysis.save_figures(
            close_figs=self.get_param_value('close_figs', True))


class RamseyAddPulseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        auto = kwargs.pop('auto', True)
        super().__init__(*args, auto=False, **kwargs)
        options_dict = kwargs.pop('options_dict', OrderedDict())
        options_dict_no = deepcopy(options_dict)
        options_dict_no.update(dict(
            data_filter=lambda raw: np.concatenate([
                raw[:-4][1::2], raw[-4:]]),
            hdf_group_name_suffix='_no_pulse'))
        self.ramsey_analysis = RamseyAnalysis(
            *args, auto=False, options_dict=options_dict_no,
            **kwargs)
        options_dict_with = deepcopy(options_dict)
        options_dict_with.update(dict(
            data_filter=lambda raw: np.concatenate([
                raw[:-4][0::2], raw[-4:]]),
            hdf_group_name_suffix='_with_pulse'))
        self.ramsey_add_pulse_analysis = RamseyAnalysis(
            *args, auto=False, options_dict=options_dict_with,
            **kwargs)


        if auto:
            self.ramsey_analysis.extract_data()
            self.ramsey_analysis.process_data()
            self.ramsey_analysis.prepare_fitting()
            self.ramsey_analysis.run_fitting()
            self.ramsey_analysis.save_fit_results()
            self.ramsey_add_pulse_analysis.extract_data()
            self.ramsey_add_pulse_analysis.process_data()
            self.ramsey_add_pulse_analysis.prepare_fitting()
            self.ramsey_add_pulse_analysis.run_fitting()
            self.ramsey_add_pulse_analysis.save_fit_results()
            self.raw_data_dict = self.ramsey_analysis.raw_data_dict
            self.analyze_fit_results()
            self.prepare_plots()
            keylist = []
            for qbn in self.qb_names:
                figure_name = 'CrossZZ_' + qbn
                keylist.append(figure_name+'with')
                keylist.append(figure_name+'no')
            self.plot()
            self.save_figures(close_figs=True)

    def analyze_fit_results(self):
        self.cross_kerr = 0.0
        self.ramsey_analysis.analyze_fit_results()
        self.ramsey_add_pulse_analysis.analyze_fit_results()

        self.proc_data_dict['analysis_params_dict'] = OrderedDict()


        for qbn in self.qb_names:

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()

            self.params_dict_ramsey = self.ramsey_analysis.proc_data_dict[
                'analysis_params_dict'][qbn]
            self.params_dict_add_pulse = \
                self.ramsey_add_pulse_analysis.proc_data_dict[
                    'analysis_params_dict'][qbn]
            self.cross_kerr = self.params_dict_ramsey[
                                  'exp_decay']['new_qb_freq'] \
                            - self.params_dict_add_pulse[
                                  'exp_decay']['new_qb_freq']
            self.cross_kerr_error = np.sqrt(
                (self.params_dict_ramsey[
                    'exp_decay']['new_qb_freq_stderr'])**2 +
                (self.params_dict_add_pulse[
                    'exp_decay']['new_qb_freq_stderr'])**2)

    def prepare_plots(self):
        self.ramsey_analysis.prepare_plots()
        self.ramsey_add_pulse_analysis.prepare_plots()

        self.ramsey_analysis.plot(key_list='auto')
        self.ramsey_analysis.save_figures(close_figs=True, savebase='Ramsey_no')

        self.ramsey_add_pulse_analysis.plot(key_list='auto')
        self.ramsey_add_pulse_analysis.save_figures(close_figs=True,
                                                    savebase='Ramsey_with')

        self.default_options['plot_proj_data'] = False
        self.metadata = {'plot_proj_data': False, 'plot_raw_data': False}
        super().prepare_plots()

        try:
            xunit = self.metadata["sweep_unit"]
            xlabel = self.metadata["sweep_name"]
        except KeyError:
            xlabel = self.raw_data_dict['sweep_parameter_names'][0]
            xunit = self.raw_data_dict['sweep_parameter_units'][0]
        if np.ndim(xunit) > 0:
            xunit = xunit[0]
        title = (self.raw_data_dict['timestamp'] + ' ' +
                 self.raw_data_dict['measurementstring'])

        for qbn in self.qb_names:
            data_no = self.ramsey_analysis.proc_data_dict['data_to_fit'][
                          qbn][:-self.ramsey_analysis.num_cal_points]
            data_with = self.ramsey_add_pulse_analysis.proc_data_dict[
                            'data_to_fit'][
                            qbn][:-self.ramsey_analysis.num_cal_points]
            delays = self.ramsey_analysis.proc_data_dict['sweep_points_dict'][
                         qbn]['sweep_points'][
                     :-self.ramsey_analysis.num_cal_points]

            figure_name = 'CrossZZ_' + qbn
            self.plot_dicts[figure_name+'with'] = {
                'fig_id': figure_name,
                'plotfn': self.plot_line,
                'xvals': delays,
                'yvals': data_with,
                'xlabel': xlabel,
                'xunit': xunit,
                'ylabel': '|e> state population',
                'setlabel': 'with $\\pi$-pulse',
                'title': title,
                'color': 'r',
                'marker': 'o',
                'line_kws': {'markersize': 5},
                'linestyle': 'none',
                'do_legend': True,
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1, -0.15),
                'legend_pos': 'upper right'}

            if self.do_fitting:
                fit_res_with = self.ramsey_add_pulse_analysis.fit_dicts[
                    'exp_decay_' + qbn]['fit_res']
                self.plot_dicts['fit_with_'+qbn] = {
                    'fig_id': figure_name,
                    'plotfn': self.plot_fit,
                    'xlabel': 'Ramsey delay',
                    'xunit': 's',
                    'fit_res': fit_res_with,
                    'setlabel': 'with $\\pi$-pulse - fit',
                    'title': title,
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

            self.plot_dicts[figure_name+'no'] = {
                'fig_id': figure_name,
                'plotfn': self.plot_line,
                'xvals': delays,
                'yvals': data_no,
                'setlabel': 'no $\\pi$-pulse',
                'title': title,
                'color': 'g',
                'marker': 'o',
                'line_kws': {'markersize': 5},
                'linestyle': 'none',
                'do_legend': True,
                'legend_ncol': 2,
                'legend_bbox_to_anchor': (1, -0.15),
                'legend_pos': 'upper right'}

            if self.do_fitting:
                fit_res_no = self.ramsey_analysis.fit_dicts[
                    'exp_decay_' + qbn]['fit_res']
                self.plot_dicts['fit_no_'+qbn] = {
                    'fig_id': figure_name,
                    'plotfn': self.plot_fit,
                    'xlabel': 'Ramsey delay',
                    'xunit': 's',
                    'fit_res': fit_res_no,
                    'setlabel': 'no $\\pi$-pulse - fit',
                    'title': title,
                    'do_legend': True,
                    'color': 'g',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

            textstr = r'$\alpha ZZ$ = {:.2f} +- {:.2f}'.format(
               self.cross_kerr*1e-3, self.cross_kerr_error*1e-3) + ' kHz'

            self.plot_dicts['text_msg_' + qbn] = {'fig_id': figure_name,
                                                  'text_string': textstr,
                                                  'ypos': -0.2,
                                                  'xpos': -0.075,
                                                  'horizontalalignment': 'left',
                                                  'verticalalignment': 'top',
                                                  'plotfn': self.plot_text}


class InPhaseAmpCalibAnalysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_amp180_'+qbn] = \
                s+f'.{trans_name}_amp180'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            model = lmfit.models.LinearModel()
            guess_pars = model.guess(data=data, x=sweep_points)
            guess_pars['intercept'].value = 0.5
            guess_pars['intercept'].vary = False
            key = 'fit_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': model.func,
                'fit_xvals': {'x': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            old_amp180 = self.raw_data_dict[
                f'{trans_name}_amp180_'+qbn]
            # FIXME: the following condition is always False, isn't it?
            if old_amp180 != old_amp180:
                old_amp180 = 0  # FIXME explain why

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn][
                'corrected_amp'] = old_amp180 - self.fit_dicts[
                'fit_' + qbn]['fit_res'].best_values['slope']*old_amp180
            self.proc_data_dict['analysis_params_dict'][qbn][
                'corrected_amp_stderr'] = self.fit_dicts[
                'fit_' + qbn]['fit_res'].params['slope'].stderr*old_amp180

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                # rename base plot
                if self.fit_dicts['fit_' + qbn][
                        'fit_res'].best_values['slope'] >= 0:
                    base_plot_name = 'OverRotation_' + qbn
                else:
                    base_plot_name = 'UnderRotation_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['fit_' + qbn]['fit_res'],
                    'setlabel': 'linear fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                trans_name = self.get_transition_name(qbn)
                old_amp180 = self.raw_data_dict[
                    f'{trans_name}_amp180_'+qbn]
                # FIXME: the following condition is always False, isn't it?
                if old_amp180 != old_amp180:
                    old_amp180 = 0  # FIXME: explain why
                correction_dict = self.proc_data_dict['analysis_params_dict']
                fit_res = self.fit_dicts['fit_' + qbn]['fit_res']
                textstr = '$\pi$-Amp = {:.4f} mV'.format(
                    correction_dict[qbn]['corrected_amp']*1e3) \
                          + ' $\pm$ {:.1e} mV'.format(
                    correction_dict[qbn]['corrected_amp_stderr']*1e3) \
                          + '\nold $\pi$-Amp = {:.4f} mV'.format(
                    old_amp180*1e3) \
                          + '\namp. correction = {:.4f} mV'.format(
                              fit_res.best_values['slope']*old_amp180*1e3) \
                          + '\nintercept = {:.2f}'.format(
                              fit_res.best_values['intercept'])
                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}

                self.plot_dicts['half_hline_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_hlines,
                    'y': 0.5,
                    'xmin': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][0],
                    'xmax': self.proc_data_dict['sweep_points_dict'][qbn][
                        'sweep_points'][-1],
                    'colors': 'gray'}


class MultiCZgate_Calib_Analysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        self.phase_key = 'phase_diffs'
        self.legend_label_func = lambda qbn, row: ''
        super().__init__(*args, **kwargs)

    def extract_data(self):
        super().extract_data()

        # Find leakage and ramsey qubit names
        self.leakage_qbnames = self.get_param_value('leakage_qbnames',
                                                    default_value=[])
        self.ramsey_qbnames = self.get_param_value('ramsey_qbnames',
                                                   default_value=[])
        self.gates_list = self.get_param_value('gates_list', default_value=[])
        if not len(self.gates_list):
            # self.gates_list must exist as a list of tuples where the first
            # entry in each tuple is a leakage qubit name, and the second is
            # a ramsey qubit name.
            self.gates_list = [(qbl, qbr) for qbl, qbr in
                               zip(self.leakage_qbnames, self.ramsey_qbnames)]

        # prepare list of qubits on which must be considered simultaneously
        # for preselection. Default: preselect on all qubits in the gate = ground
        default_preselection_qbs = defaultdict(list)
        for qbn in self.qb_names:
            for gate_qbs in self.gates_list:
                if qbn in gate_qbs:
                    default_preselection_qbs[qbn].extend(gate_qbs)
        self.default_options.update(
            {"preselection_qbs": default_preselection_qbs})

    def process_data(self):
        super().process_data()

        # Deal with the case when self.data_to_fit[qbn] are lists with multiple
        # entries. The parent class takes the first entry in
        # self.data_to_fit[qbn] and forces its type to str. Then it figures out
        # the appropriate self.data_to_fit based on cal_points (ex. 'pca').
        # Here we want to allow lists with multiple entries so.

        # TODO: Steph 15.09.2020
        # This is a hack. MultiQubit_TimeDomain_Analysis should be upgraded to
        # allow lists of multiple entries here but this would break every
        # analysis inheriting from it.
        # For now, we just needed it to work for this analysis to allow the same
        # data to be fitted in two ways (allows to extract SWAP errors).

        # Take the data_to_fit provided by the user:
        data_to_fit = self.get_param_value('data_to_fit')
        if data_to_fit is None or not(len(data_to_fit)):
            data_to_fit = {qbn: [] for qbn in self.qb_names}
        for qbn in self.data_to_fit:
            # The entries can be different for each qubit so make sure
            # entries are lists. As mentioned above, the parent class sets them
            # to strings.
            if isinstance(self.data_to_fit[qbn], str):
                self.data_to_fit[qbn] = [self.data_to_fit[qbn]]
            if isinstance(data_to_fit[qbn], str):
                data_to_fit[qbn] = [data_to_fit[qbn]]

        # Overwrite data_to_fit in proc_data_dict, as well as self.data_to_fit
        # if needed.
        self.proc_data_dict['data_to_fit'] = OrderedDict()
        for qbn, prob_data in self.proc_data_dict[
                'projected_data_dict'].items():
            if qbn in self.data_to_fit:
                if len(self.data_to_fit[qbn]) < len(data_to_fit[qbn]) and \
                        'pca' not in self.data_to_fit[qbn][0].lower():
                    # The entry that the parent class assigned is shorter than
                    # the one specified by the user. We only want to keep the
                    # former if it was 'pca', otherwise overwrite the entry to
                    # allow lists with several values.
                    self.data_to_fit[qbn] = data_to_fit[qbn]
                # Add the data from projected_data_dict specified by
                # self.data_to_fit[qbn]
                self.proc_data_dict['data_to_fit'][qbn] = {
                    prob_label: prob_data[prob_label] for prob_label in
                    self.data_to_fit[qbn]}

        # Make sure data has the right shape (len(hard_sp), len(soft_sp))
        for qbn, prob_data in self.proc_data_dict['data_to_fit'].items():
            for prob_label, data in prob_data.items():
                if data.shape[1] != self.proc_data_dict[
                        'sweep_points_dict'][qbn]['sweep_points'].size:
                    self.proc_data_dict['data_to_fit'][qbn][prob_label] = data.T

        # reshape data for ease of use
        qbn = self.qb_names[0]
        phase_sp_param_name = [p for p in self.mospm[qbn] if 'phase' in p][0]
        phases = self.sp.get_sweep_params_property('values', 0,
                                                   phase_sp_param_name)
        self.dim_scale_factor = len(phases) // len(np.unique(phases))

        self.proc_data_dict['data_to_fit_reshaped'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['data_to_fit_reshaped'][qbn] = {
                prob_label: np.reshape(
                    self.proc_data_dict['data_to_fit'][qbn][prob_label][
                    :, :-self.num_cal_points],
                    (self.dim_scale_factor * \
                     self.proc_data_dict['data_to_fit'][qbn][prob_label][
                       :, :-self.num_cal_points].shape[0],
                     self.proc_data_dict['data_to_fit'][qbn][prob_label][
                     :, :-self.num_cal_points].shape[1]//self.dim_scale_factor))
                for prob_label in self.proc_data_dict['data_to_fit'][qbn]}

        # convert phases to radians
        for qbn in self.qb_names:
            sweep_dict = self.proc_data_dict['sweep_points_dict'][qbn]
            sweep_dict['sweep_points'] *= np.pi/180

    def plot_traces(self, prob_label, data_2d, qbn):
        plotsize = self.get_default_plot_params(set_pars=False)[
            'figure.figsize']
        plotsize = (plotsize[0], plotsize[0]/1.25)
        if data_2d.shape[1] != self.proc_data_dict[
                'sweep_points_dict'][qbn]['sweep_points'].size:
            data_2d = data_2d.T

        data_2d_reshaped = np.reshape(
            data_2d[:, :-self.num_cal_points],
            (self.dim_scale_factor*data_2d[:, :-self.num_cal_points].shape[0],
             data_2d[:, :-self.num_cal_points].shape[1]//self.dim_scale_factor))

        data_2d_cal_reshaped = [[data_2d[:, -self.num_cal_points:]]] * \
                               (self.dim_scale_factor *
                                data_2d[:, :-self.num_cal_points].shape[0])

        ref_states_plot_dicts = {}
        for row in range(data_2d_reshaped.shape[0]):
            phases = np.unique(self.proc_data_dict['sweep_points_dict'][qbn][
                                   'msmt_sweep_points'])
            data = data_2d_reshaped[row, :]
            legend_bbox_to_anchor = (1, -0.15)
            legend_pos = 'upper right'
            legend_ncol = 2

            if qbn in self.ramsey_qbnames and self.get_latex_prob_label(
                    prob_label) in [self.get_latex_prob_label(pl)
                                    for pl in self.data_to_fit[qbn]]:
                figure_name = '{}_{}_{}'.format(self.phase_key, qbn, prob_label)
            elif qbn in self.leakage_qbnames and self.get_latex_prob_label(
                    prob_label) in [self.get_latex_prob_label(pl)
                                    for pl in self.data_to_fit[qbn]]:
                figure_name = 'Leakage_{}_{}'.format(qbn, prob_label)
            else:
                figure_name = 'projected_plot_' + qbn + '_' + \
                              prob_label

            # plot cal points
            if self.num_cal_points > 0:
                data_w_cal = data_2d_cal_reshaped[row][0][0]
                for i, cal_pts_idxs in enumerate(
                        self.cal_states_dict[qbn].values()):
                    s = '{}_{}_{}'.format(row, qbn, prob_label)
                    ref_state_plot_name = list(
                        self.cal_states_dict[qbn])[i] + '_' + s
                    ref_states_plot_dicts[ref_state_plot_name] = {
                        'fig_id': figure_name,
                        'plotfn': self.plot_line,
                        'plotsize': plotsize,
                        'xvals': self.proc_data_dict[
                            'sweep_points_dict'][qbn][
                            'cal_points_sweep_points'][
                            cal_pts_idxs],
                        'yvals': data_w_cal[cal_pts_idxs],
                        'setlabel': list(
                            self.cal_states_dict[qbn])[i] if
                        row == 0 else '',
                        'do_legend': row == 0,
                        'legend_bbox_to_anchor':
                            legend_bbox_to_anchor,
                        'legend_pos': legend_pos,
                        'legend_ncol': legend_ncol,
                        'linestyle': 'none',
                        'line_kws': {'color':
                            self.get_cal_state_color(
                                list(self.cal_states_dict[qbn])[i])}}

            xlabel, xunit = self.get_xaxis_label_unit(qbn)
            self.plot_dicts['data_{}_{}_{}'.format(
                row, qbn, prob_label)] = {
                'plotfn': self.plot_line,
                'fig_id': figure_name,
                'plotsize': plotsize,
                'xvals': phases,
                'xlabel': xlabel,
                'xunit': 'rad',  # overriden from deg to rad in process_data
                'yvals': data,
                'ylabel': self.get_yaxis_label(qbn, prob_label),
                'yunit': '',
                'yscale': self.get_param_value("yscale", "linear"),
                'setlabel': 'Data - ' + self.legend_label_func(qbn, row)
                    if row in [0, 1] else '',
                'title': self.raw_data_dict['timestamp'] + ' ' +
                         self.raw_data_dict['measurementstring'] + '-' + qbn,
                'linestyle': 'none',
                'color': 'C0' if row % 2 == 0 else 'C2',
                'do_legend': row in [0, 1],
                'legend_ncol': legend_ncol,
                'legend_bbox_to_anchor': legend_bbox_to_anchor,
                'legend_pos': legend_pos}

            if self.do_fitting and 'projected' not in figure_name:
                if qbn in self.leakage_qbnames and self.get_param_value(
                        'classified_ro', False):
                    continue

                k = 'fit_{}{}_{}_{}'.format(
                    'on' if row % 2 == 0 else 'off', row, prob_label, qbn)
                if f'Cos_{k}' in self.fit_dicts:
                    fit_res = self.fit_dicts[f'Cos_{k}']['fit_res']
                    self.plot_dicts[k + '_' + prob_label] = {
                        'fig_id': figure_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res,
                        'setlabel': 'Fit - ' + self.legend_label_func(qbn, row)
                            if row in [0, 1] else '',
                        'color': 'C0' if row % 2 == 0 else 'C2',
                        'do_legend': row in [0, 1],
                        'legend_ncol': legend_ncol,
                        'legend_bbox_to_anchor':
                            legend_bbox_to_anchor,
                        'legend_pos': legend_pos}
                elif f'Linear_{k}' in self.fit_dicts:
                    fit_res = self.fit_dicts[f'Linear_{k}']['fit_res']
                    xvals = fit_res.userkws[
                        fit_res.model.independent_vars[0]]
                    xfine = np.linspace(min(xvals), max(xvals), 100)
                    yvals = fit_res.model.func(
                        xfine, **fit_res.best_values)
                    if not hasattr(yvals, '__iter__'):
                        yvals = np.array(len(xfine)*[yvals])

                    self.plot_dicts[k] = {
                        'fig_id': figure_name,
                        'plotfn': self.plot_line,
                        'xvals': xfine,
                        'yvals': yvals,
                        'marker': '',
                        'setlabel': 'Fit - ' + self.legend_label_func(
                            qbn, row) if row in [0, 1] else '',
                        'do_legend': row in [0, 1],
                        'legend_ncol': legend_ncol,
                        'color': 'C0' if row % 2 == 0 else 'C2',
                        'legend_bbox_to_anchor':
                            legend_bbox_to_anchor,
                        'legend_pos': legend_pos}

        # ref state plots need to be added at the end, otherwise the
        # legend for |g> and |e> is added twice (because of the
        # condition do_legend = (row in [0,1]) in the plot dicts above
        if self.num_cal_points > 0:
            self.plot_dicts.update(ref_states_plot_dicts)
        return figure_name

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        self.leakage_values = np.array([])
        labels = ['on', 'off']
        for i, qbn in enumerate(self.qb_names):
            for prob_label in self.data_to_fit[qbn]:
                for row in range(self.proc_data_dict['data_to_fit_reshaped'][
                                     qbn][prob_label].shape[0]):
                    phases = np.unique(self.proc_data_dict['sweep_points_dict'][
                                           qbn]['msmt_sweep_points'])
                    data = self.proc_data_dict['data_to_fit_reshaped'][qbn][
                        prob_label][row, :]
                    key = 'fit_{}{}_{}_{}'.format(labels[row % 2], row,
                                                   prob_label, qbn)
                    if qbn in self.leakage_qbnames and (prob_label == 'pf'
                            or 'pca' in prob_label.lower()):
                        if self.get_param_value('classified_ro', False):
                            self.leakage_values = np.append(self.leakage_values,
                                                            np.mean(data))
                        else:
                            # fit leakage qb results to a constant
                            model = lmfit.models.ConstantModel()
                            guess_pars = model.guess(data=data, x=phases)
                            self.fit_dicts[f'Linear_{key}'] = {
                                'fit_fn': model.func,
                                'fit_xvals': {'x': phases},
                                'fit_yvals': {'data': data},
                                'guess_pars': guess_pars}
                    else:
                        # fit ramsey qb results to a cosine
                        model = lmfit.Model(fit_mods.CosFunc)
                        guess_pars = fit_mods.Cos_guess(
                            model=model,
                            t=phases,
                            data=data, freq_guess=1/(2*np.pi))
                        guess_pars['frequency'].value = 1/(2*np.pi)
                        guess_pars['frequency'].vary = False

                        self.fit_dicts[f'Cos_{key}'] = {
                            'fit_fn': fit_mods.CosFunc,
                            'fit_xvals': {'t': phases},
                            'fit_yvals': {'data': data},
                            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()

        for qbn in self.qb_names:
            # Cos fits
            keys = [k for k in list(self.fit_dicts.keys()) if
                    (k.startswith('Cos') and k.endswith(qbn))]
            if len(keys) > 0:
                fit_res_objs = [self.fit_dicts[k]['fit_res'] for k in keys]
                # cosine amplitudes
                amps = np.array([fr.best_values['amplitude'] for fr
                                 in fit_res_objs])
                amps_errs = np.array([fr.params['amplitude'].stderr
                                      for fr in fit_res_objs], dtype=np.float64)
                amps_errs = np.nan_to_num(amps_errs)
                # amps_errs.dtype = amps.dtype
                if qbn in self.ramsey_qbnames:
                    # phase_diffs
                    phases = np.array([fr.best_values['phase'] for fr in
                                       fit_res_objs])
                    phases_errs = np.array([fr.params['phase'].stderr for fr in
                                            fit_res_objs], dtype=np.float64)
                    phases_errs = np.nan_to_num(phases_errs)
                    self.proc_data_dict['analysis_params_dict'][
                        f'phases_{qbn}'] = {
                        'val': phases, 'stderr': phases_errs}

                    # compute phase diffs
                    if getattr(self, 'delta_tau', 0) is not None:
                        # this can be false for Cyroscope with
                        # estimation_window == None and odd nr of trunc lengths
                        phase_diffs = phases[0::2] - phases[1::2]
                        phase_diffs %= (2*np.pi)
                        phase_diffs_stderrs = np.sqrt(
                            np.array(phases_errs[0::2]**2 +
                                     phases_errs[1::2]**2, dtype=np.float64))
                        self.proc_data_dict['analysis_params_dict'][
                            f'{self.phase_key}_{qbn}'] = {
                            'val': phase_diffs, 'stderr': phase_diffs_stderrs}

                        # contrast = (cos_amp_g + cos_amp_e)/ 2
                        contrast = (amps[1::2] + amps[0::2])/2
                        contrast_stderr = 0.5*np.sqrt(
                            np.array(amps_errs[0::2]**2 + amps_errs[1::2]**2,
                                     dtype=np.float64))

                        self.proc_data_dict['analysis_params_dict'][
                            f'mean_contrast_{qbn}'] = {
                            'val': contrast, 'stderr': contrast_stderr}

                        # contrast_loss = (cos_amp_g - cos_amp_e)/ cos_amp_g
                        contrast_loss = (amps[1::2] - amps[0::2])/amps[1::2]
                        x = amps[1::2] - amps[0::2]
                        x_err = np.array(amps_errs[0::2]**2 + amps_errs[1::2]**2,
                                         dtype=np.float64)
                        y = amps[1::2]
                        y_err = amps_errs[1::2]
                        try:
                            contrast_loss_stderrs = np.sqrt(np.array(
                                ((y * x_err) ** 2 + (x * y_err) ** 2) / (y ** 4),
                                dtype=np.float64))
                        except:
                            contrast_loss_stderrs = float("nan")
                        self.proc_data_dict['analysis_params_dict'][
                            f'contrast_loss_{qbn}'] = \
                            {'val': contrast_loss,
                             'stderr': contrast_loss_stderrs}

                else:
                    self.proc_data_dict['analysis_params_dict'][
                        f'amps_{qbn}'] = {
                        'val': amps[1::2], 'stderr': amps_errs[1::2]}

            # Linear fits
            keys = [k for k in list(self.fit_dicts.keys()) if
                    (k.startswith('Linear') and k.endswith(qbn))]
            if len(keys) > 0:
                fit_res_objs = [self.fit_dicts[k]['fit_res'] for k in keys]
                # get leakage
                lines = np.array([fr.best_values['c'] for fr
                                  in fit_res_objs])
                lines_errs = np.array([fr.params['c'].stderr for
                                       fr in fit_res_objs], dtype=np.float64)
                lines_errs = np.nan_to_num(lines_errs)

                leakage = lines[0::2]
                leakage_errs = np.array(lines_errs[0::2], dtype=np.float64)
                leakage_increase = lines[0::2] - lines[1::2]
                leakage_increase_errs = np.array(np.sqrt(lines_errs[0::2]**2,
                                                         lines_errs[1::2]**2),
                                                 dtype=np.float64)
                self.proc_data_dict['analysis_params_dict'][
                    f'leakage_{qbn}'] = \
                    {'val': leakage, 'stderr': leakage_errs}
                self.proc_data_dict['analysis_params_dict'][
                    f'leakage_increase_{qbn}'] = {'val': leakage_increase,
                                                  'stderr': leakage_increase_errs}

            # special case: if classified detector was used, we get leakage
            # for free
            if qbn in self.leakage_qbnames and self.get_param_value(
                    'classified_ro', False):
                leakage = self.leakage_values[0::2]
                leakage_errs = np.zeros(len(leakage))
                leakage_increase = self.leakage_values[0::2] - \
                                   self.leakage_values[1::2]
                leakage_increase_errs = np.zeros(len(leakage))
                self.proc_data_dict['analysis_params_dict'][
                    f'leakage_{qbn}'] = \
                    {'val': leakage, 'stderr': leakage_errs}
                self.proc_data_dict['analysis_params_dict'][
                    f'leakage_increase_{qbn}'] = {'val': leakage_increase,
                                                  'stderr': leakage_increase_errs}

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        if self.do_fitting:
            len_ssp = len(
                self.proc_data_dict['analysis_params_dict'][
                    f'{self.phase_key}_{self.ramsey_qbnames[0]}']['val'])
        if self.get_param_value('plot_all_traces', True):
            for j, qbn in enumerate(self.qb_names):
                if self.get_param_value('plot_all_probs', True):
                    for prob_label, data_2d in self.proc_data_dict[
                            'projected_data_dict'][qbn].items():
                        self.plot_traces(prob_label, data_2d, qbn)
                else:
                    for prob_label, data_2d in self.proc_data_dict[
                            'data_to_fit'][qbn]:
                        self.plot_traces(prob_label, data_2d, qbn)

                if self.do_fitting and len_ssp == 1:
                    # only plot raw data here; the projected data plots were
                    # prepared above
                    self.prepare_raw_data_plots()

                    if qbn in self.ramsey_qbnames:
                        # add the cphase + leakage textboxes to the
                        # cphase_qbr_pe figure
                        figure_name = f'{self.phase_key}_{qbn}_'
                        figure_name += self.data_to_fit[qbn][0] if \
                            'pca' in self.data_to_fit[qbn][0].lower() else 'pe'
                        textstr = '{} = \n{:.2f}'.format(
                            self.phase_key,
                            self.proc_data_dict['analysis_params_dict'][
                                f'{self.phase_key}_{qbn}']['val'][0]*180/np.pi) + \
                                  r'$^{\circ}$' + \
                                  '$\\pm${:.2f}'.format(
                                      self.proc_data_dict[
                                          'analysis_params_dict'][
                                          f'{self.phase_key}_{qbn}'][
                                          'stderr'][0] * 180 / np.pi) + \
                                  r'$^{\circ}$'
                        textstr += '\nMean contrast = \n' + \
                                   '{:.3f} $\\pm$ {:.3f}'.format(
                                       self.proc_data_dict[
                                           'analysis_params_dict'][
                                           f'mean_contrast_{qbn}']['val'][0],
                                       self.proc_data_dict[
                                           'analysis_params_dict'][
                                           f'mean_contrast_{qbn}'][
                                           'stderr'][0])
                        textstr += '\nContrast loss = \n' + \
                                   '{:.3f} $\\pm$ {:.3f}'.format(
                                       self.proc_data_dict[
                                           'analysis_params_dict'][
                                           f'contrast_loss_{qbn}']['val'][0],
                                       self.proc_data_dict[
                                           'analysis_params_dict'][
                                           f'contrast_loss_{qbn}'][
                                           'stderr'][0])
                        pdap = self.proc_data_dict.get(
                            'percent_data_after_presel', False)
                        if pdap:
                            textstr += "\nPreselection = \n {" + ', '.join(
                                f"{qbn}: {v}" for qbn, v in pdap.items()) + '}'

                        self.plot_dicts['cphase_text_msg_' + qbn] = {
                            'fig_id': figure_name,
                            'ypos': -0.2,
                            'xpos': -0.1,
                            'horizontalalignment': 'left',
                            'verticalalignment': 'top',
                            'box_props': None,
                            'plotfn': self.plot_text,
                            'text_string': textstr}

                        qbl = [gl[0] for gl in self.gates_list
                               if qbn == gl[1]]
                        if len(qbl):
                            qbl = qbl[0]
                            textstr = 'Leakage =\n{:.5f} $\\pm$ {:.5f}'.format(
                                self.proc_data_dict['analysis_params_dict'][
                                    f'leakage_{qbl}']['val'][0],
                                self.proc_data_dict['analysis_params_dict'][
                                    f'leakage_{qbl}']['stderr'][0])
                            textstr += '\n\n$\\Delta$Leakage = \n' \
                                       '{:.5f} $\\pm$ {:.5f}'.format(
                                self.proc_data_dict['analysis_params_dict'][
                                    f'leakage_increase_{qbl}']['val'][0],
                                self.proc_data_dict['analysis_params_dict'][
                                    f'leakage_increase_{qbl}']['stderr'][0])
                            self.plot_dicts['cphase_text_msg_' + qbl] = {
                                'fig_id': figure_name,
                                'ypos': -0.2,
                                'xpos': 0.175,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'box_props': None,
                                'plotfn': self.plot_text,
                                'text_string': textstr}

                    else:
                        if f'amps_{qbn}' in self.proc_data_dict[
                                'analysis_params_dict']:
                            figure_name = f'Leakage_{qbn}_pg'
                            textstr = 'Amplitude CZ int. OFF = \n' + \
                                       '{:.3f} $\\pm$ {:.3f}'.format(
                                           self.proc_data_dict[
                                               'analysis_params_dict'][
                                               f'amps_{qbn}']['val'][0],
                                           self.proc_data_dict[
                                               'analysis_params_dict'][
                                               f'amps_{qbn}']['stderr'][0])
                            self.plot_dicts['swap_text_msg_' + qbn] = {
                                'fig_id': figure_name,
                                'ypos': -0.2,
                                'xpos': -0.1,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'box_props': None,
                                'plotfn': self.plot_text,
                                'text_string': textstr}

        # plot analysis results
        if self.do_fitting and len_ssp > 1:
            for qbn in self.qb_names:
                ss_pars = self.proc_data_dict['sweep_points_2D_dict'][qbn]
                for idx, ss_pname in enumerate(ss_pars):
                    xvals = self.sp.get_sweep_params_property('values', 1,
                                                              ss_pname)
                    xvals_to_use = deepcopy(xvals)
                    xlabel = self.sp.get_sweep_params_property('label', 1,
                                                               ss_pname)
                    xunit = self.sp.get_sweep_params_property('unit', 1,
                                                               ss_pname)
                    for param_name, results_dict in self.proc_data_dict[
                            'analysis_params_dict'].items():
                        if qbn in param_name:
                            reps = 1
                            if len(results_dict['val']) >= len(xvals):
                                reps = len(results_dict['val']) / len(xvals)
                            else:
                                # cyroscope case
                                if hasattr(self, 'xvals_reduction_func'):
                                    xvals_to_use = self.xvals_reduction_func(
                                        xvals)
                                else:
                                    log.warning(f'Length mismatch between xvals'
                                                ' and analysis param for'
                                                ' {param_name}, and no'
                                                ' xvals_reduction_func has been'
                                                ' defined. Unclear how to'
                                                ' reduce xvals.')

                            plot_name = f'{param_name}_vs_{xlabel}'
                            if 'phase' in param_name:
                                yvals = results_dict['val']*180/np.pi - (180 if
                                    len(self.leakage_qbnames) > 0 else 0)
                                yerr = results_dict['stderr']*180/np.pi
                                ylabel = param_name + ('-$180^{\\circ}$' if
                                    len(self.leakage_qbnames) > 0 else '')
                                self.plot_dicts[plot_name+'_hline'] = {
                                    'fig_id': plot_name,
                                    'plotfn': self.plot_hlines,
                                    'y': 0,
                                    'xmin': np.min(xvals_to_use),
                                    'xmax': np.max(xvals_to_use),
                                    'colors': 'gray'}
                            else:
                                yvals = results_dict['val']
                                yerr = results_dict['stderr']
                                ylabel = param_name

                            if 'phase' in param_name:
                                yunit = 'deg'
                            elif 'freq' in param_name:
                                yunit = 'Hz'
                            else:
                                yunit = ''

                            self.plot_dicts[plot_name] = {
                                'plotfn': self.plot_line,
                                'xvals': np.repeat(xvals_to_use, reps),
                                'xlabel': xlabel,
                                'xunit': xunit,
                                'yvals': yvals,
                                'yerr': yerr if param_name != 'leakage'
                                    else None,
                                'ylabel': ylabel,
                                'yunit': yunit,
                                'title': self.raw_data_dict['timestamp'] + ' ' +
                                         self.raw_data_dict['measurementstring']
                                         + '-' + qbn,
                                'linestyle': 'none',
                                'do_legend': False}


class CPhaseLeakageAnalysis(MultiCZgate_Calib_Analysis):

    def extract_data(self):
        super().extract_data()
        # Find leakage and ramsey qubit names
        # first try the legacy code
        leakage_qbname = self.get_param_value('leakage_qbname')
        ramsey_qbname = self.get_param_value('ramsey_qbname')
        if leakage_qbname is not None and ramsey_qbname is not None:
            self.gates_list += [(leakage_qbname, ramsey_qbname)]
            self.leakage_qbnames = [leakage_qbname]
            self.ramsey_qbnames = [ramsey_qbname]
        else:
            # new measurement framework
            task_list = self.get_param_value('task_list', default_value=[])
            for task in task_list:
                self.gates_list += [(task['qbl'], task['qbr'])]
                self.leakage_qbnames += [task['qbl']]
                self.ramsey_qbnames += [task['qbr']]

        if len(self.leakage_qbnames) == 0 and len(self.ramsey_qbnames) == 0:
            raise ValueError('Please provide either leakage_qbnames or '
                             'ramsey_qbnames.')
        elif len(self.ramsey_qbnames) == 0:
            self.ramsey_qbnames = [qbn for qbn in self.qb_names if
                                  qbn not in self.leakage_qbnames]
        elif len(self.leakage_qbnames) == 0:
            self.leakage_qbnames = [qbn for qbn in self.qb_names if
                                   qbn not in self.ramsey_qbnames]
            if len(self.leakage_qbnames) == 0:
                self.leakage_qbnames = None


    def process_data(self):
        super().process_data()


        self.phase_key = 'cphase'
        if len(self.leakage_qbnames) > 0:
            def legend_label_func(qbn, row, gates_list=self.gates_list):
                leakage_qbnames = [qb_tup[0] for qb_tup in gates_list]
                if qbn in leakage_qbnames:
                    return f'{qbn} in $|g\\rangle$' if row % 2 != 0 else \
                        f'{qbn} in $|e\\rangle$'
                else:
                    qbln = [qb_tup for qb_tup in gates_list
                            if qbn == qb_tup[1]][0][0]
                    return f'{qbln} in $|g\\rangle$' if row % 2 != 0 else \
                        f'{qbln} in $|e\\rangle$'
        else:
            legend_label_func = lambda qbn, row: \
                'qbc in $|g\\rangle$' if row % 2 != 0 else \
                    'qbc in $|e\\rangle$'
        self.legend_label_func = legend_label_func


class DynamicPhaseAnalysis(MultiCZgate_Calib_Analysis):

    def process_data(self):
        super().process_data()

        if len(self.ramsey_qbnames) == 0:
            self.ramsey_qbnames = self.qb_names

        self.phase_key = 'dynamic_phase'
        self.legend_label_func = lambda qbn, row: 'no FP' \
            if row % 2 != 0 else 'with FP'


class CryoscopeAnalysis(DynamicPhaseAnalysis):

    def __init__(self, qb_names, *args, **kwargs):
        options_dict = kwargs.get('options_dict', {})
        unwrap_phases = options_dict.pop('unwrap_phases', True)
        options_dict['unwrap_phases'] = unwrap_phases
        kwargs['options_dict'] = options_dict
        params_dict = {}
        for qbn in qb_names:
            s = f'Instrument settings.{qbn}'
            params_dict[f'ge_freq_{qbn}'] = s+f'.ge_freq'
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        super().__init__(qb_names, *args, **kwargs)

    def process_data(self):
        super().process_data()
        self.phase_key = 'delta_phase'

    def analyze_fit_results(self):
        global_delta_tau = self.get_param_value('estimation_window')
        task_list = self.get_param_value('task_list')
        for qbn in self.qb_names:
            delta_tau = deepcopy(global_delta_tau)
            if delta_tau is None:
                if task_list is None:
                    log.warning(f'estimation_window is None and task_list '
                                f'for {qbn} was not found. Assuming no '
                                f'estimation_window was used.')
                else:
                    task = [t for t in task_list if t['qb'] == qbn]
                    if not len(task):
                        raise ValueError(f'{qbn} not found in task_list.')
                    delta_tau = task[0].get('estimation_window', None)
        self.delta_tau = delta_tau

        if self.get_param_value('analyze_fit_results_super', True):
            super().analyze_fit_results()
        self.proc_data_dict['tvals'] = OrderedDict()

        for qbn in self.qb_names:
            if delta_tau is None:
                trunc_lengths = self.sp.get_sweep_params_property(
                    'values', 1, f'{qbn}_truncation_length')
                delta_tau = np.diff(trunc_lengths)
                m = delta_tau > 0
                delta_tau = delta_tau[m]
                phases = self.proc_data_dict['analysis_params_dict'][
                    f'phases_{qbn}']
                delta_phases_vals = -np.diff(phases['val'])[m]
                delta_phases_vals = (delta_phases_vals + np.pi) % (
                            2 * np.pi) - np.pi
                delta_phases_errs = (np.sqrt(
                    np.array(phases['stderr'][1:] ** 2 +
                             phases['stderr'][:-1] ** 2, dtype=np.float64)))[m]

                self.xvals_reduction_func = lambda xvals: \
                    ((xvals[1:] + xvals[:-1]) / 2)[m]

                self.proc_data_dict['analysis_params_dict'][
                    f'{self.phase_key}_{qbn}'] = {
                    'val': delta_phases_vals, 'stderr': delta_phases_errs}

                # remove the entries in analysis_params_dict that are not
                # relevant for Cryoscope (pop_loss), since
                # these will cause a problem with plotting in this case.
                self.proc_data_dict['analysis_params_dict'].pop(
                    f'population_loss_{qbn}', None)
            else:
                delta_phases = self.proc_data_dict['analysis_params_dict'][
                    f'{self.phase_key}_{qbn}']
                delta_phases_vals = delta_phases['val']
                delta_phases_errs = delta_phases['stderr']

            if self.get_param_value('unwrap_phases', False):
                if hasattr(delta_tau, '__iter__'):
                    # unwrap in frequency such that we don't jump more than half
                    # the nyquist band at any step
                    df = []
                    prev_df = 0
                    for dp, dt in zip(delta_phases_vals, delta_tau):
                        df.append(dp / (2 * np.pi * dt))
                        df[-1] += np.round((prev_df - df[-1]) * dt) / dt
                        prev_df = df[-1]
                    delta_phases_vals = np.array(df)*(2*np.pi*delta_tau)
                else:
                    delta_phases_vals = np.unwrap((delta_phases_vals + np.pi) %
                                                  (2*np.pi) - np.pi)

            self.proc_data_dict['analysis_params_dict'][
                f'{self.phase_key}_{qbn}']['val'] = delta_phases_vals

            delta_freqs = delta_phases_vals/2/np.pi/delta_tau
            delta_freqs_errs = delta_phases_errs/2/np.pi/delta_tau
            self.proc_data_dict['analysis_params_dict'][f'delta_freq_{qbn}'] = \
                {'val': delta_freqs, 'stderr': delta_freqs_errs}

            qb_freqs = self.raw_data_dict[f'ge_freq_{qbn}'] + delta_freqs
            self.proc_data_dict['analysis_params_dict'][f'freq_{qbn}'] = \
                {'val':  qb_freqs, 'stderr': delta_freqs_errs}

            if hasattr(self, 'xvals_reduction_func') and \
                    self.xvals_reduction_func is not None:
                self.proc_data_dict['tvals'][f'{qbn}'] = \
                    self.xvals_reduction_func(
                    self.proc_data_dict['sweep_points_2D_dict'][qbn][
                        f'{qbn}_truncation_length'])
            else:
                self.proc_data_dict['tvals'][f'{qbn}'] = \
                    self.proc_data_dict['sweep_points_2D_dict'][qbn][
                    f'{qbn}_truncation_length']

        self.save_processed_data(key='analysis_params_dict')
        self.save_processed_data(key='tvals')

    def get_generated_and_measured_pulse(self, qbn=None):
        """
        Args:
            qbn: specifies for which qubit to calculate the quantities for.
                Defaults to the first qubit in qb_names.

        Returns: A tuple (tvals_gen, volts_gen, tvals_meas, freqs_meas,
                freq_errs_meas, volt_freq_conv)
            tvals_gen: time values for the generated fluxpulse
            volts_gen: voltages of the generated fluxpulse
            tvals_meas: time-values for the measured qubit frequencies
            freqs_meas: measured qubit frequencies
            freq_errs_meas: errors of measured qubit frequencies
            volt_freq_conv: dictionary of fit params for frequency-voltage
                conversion
        """
        if qbn is None:
            qbn = self.qb_names[0]

        tvals_meas = self.proc_data_dict['tvals'][qbn]
        freqs_meas = self.proc_data_dict['analysis_params_dict'][
            f'freq_{qbn}']['val']
        freq_errs_meas = self.proc_data_dict['analysis_params_dict'][
            f'freq_{qbn}']['stderr']

        tvals_gen, volts_gen, volt_freq_conv = self.get_generated_pulse(qbn)

        return tvals_gen, volts_gen, tvals_meas, freqs_meas, freq_errs_meas, \
               volt_freq_conv

    def get_generated_pulse(self, qbn=None, tvals_gen=None, pulse_params=None):
        """
        Args:
            qbn: specifies for which qubit to calculate the quantities for.
                Defaults to the first qubit in qb_names.

        Returns: A tuple (tvals_gen, volts_gen, tvals_meas, freqs_meas,
                freq_errs_meas, volt_freq_conv)
            tvals_gen: time values for the generated fluxpulse
            volts_gen: voltages of the generated fluxpulse
            volt_freq_conv: dictionary of fit params for frequency-voltage
                conversion
        """
        if qbn is None:
            qbn = self.qb_names[0]

        # Flux pulse parameters
        # Needs to be changed when support for other pulses is added.
        op_dict = {
            'pulse_type': f'Instrument settings.{qbn}.flux_pulse_type',
            'channel': f'Instrument settings.{qbn}.flux_pulse_channel',
            'aux_channels_dict': f'Instrument settings.{qbn}.'
                                 f'flux_pulse_aux_channels_dict',
            'amplitude': f'Instrument settings.{qbn}.flux_pulse_amplitude',
            'frequency': f'Instrument settings.{qbn}.flux_pulse_frequency',
            'phase': f'Instrument settings.{qbn}.flux_pulse_phase',
            'pulse_length': f'Instrument settings.{qbn}.'
                            f'flux_pulse_pulse_length',
            'truncation_length': f'Instrument settings.{qbn}.'
                                 f'flux_pulse_truncation_length',
            'buffer_length_start': f'Instrument settings.{qbn}.'
                                   f'flux_pulse_buffer_length_start',
            'buffer_length_end': f'Instrument settings.{qbn}.'
                                 f'flux_pulse_buffer_length_end',
            'extra_buffer_aux_pulse': f'Instrument settings.{qbn}.'
                                      f'flux_pulse_extra_buffer_aux_pulse',
            'pulse_delay': f'Instrument settings.{qbn}.'
                           f'flux_pulse_pulse_delay',
            'basis_rotation': f'Instrument settings.{qbn}.'
                              f'flux_pulse_basis_rotation',
            'gaussian_filter_sigma': f'Instrument settings.{qbn}.'
                                     f'flux_pulse_gaussian_filter_sigma',
        }

        params_dict = {
            'volt_freq_conv': f'Instrument settings.{qbn}.'
                              f'fit_ge_freq_from_flux_pulse_amp',
            'flux_channel': f'Instrument settings.{qbn}.'
                            f'flux_pulse_channel',
            'instr_pulsar': f'Instrument settings.{qbn}.'
                            f'instr_pulsar',
            **op_dict
        }

        dd = self.get_data_from_timestamp_list(params_dict)
        if pulse_params is not None:
            dd.update(pulse_params)
        dd['element_name'] = 'element'

        pulse = seg_mod.UnresolvedPulse(dd).pulse_obj
        pulse.algorithm_time(0)

        if tvals_gen is None:
            clk = self.clock(channel=dd['channel'], pulsar=dd['instr_pulsar'])
            tvals_gen = np.arange(0, pulse.length, 1 / clk)
        volts_gen = pulse.chan_wf(dd['flux_channel'], tvals_gen)
        volt_freq_conv = dd['volt_freq_conv']

        return tvals_gen, volts_gen, volt_freq_conv


class MultiQutrit_Timetrace_Analysis(ba.BaseDataAnalysis):
    """
    Analysis class for timetraces, in particular use to compute
    Optimal SNR integration weights.
    """
    def __init__(self, qb_names=None, auto=True, **kwargs):
        """
        Initializes the timetrace analysis class.
        Args:
            qb_names (list): name of the qubits to analyze (can be a subset
                of the measured qubits)
            auto (bool): Start analysis automatically (default: True)
            **kwargs:
                t_start: timestamp of the first timetrace
                t_stop: timestamp of the last timetrace to analyze
                options_dict: Dictionary for analysis options (see below)

        options_dict keywords (dict): relevant parameters:
            acq_weights_basis (list, dict):
                list of basis vectors used to compute optimal weight.
                e.g. ['ge', 'gf'], the first basis vector will be the
                "e" timetrace minus the "g" timetrace and the second basis
                vector is f - g. The first letter in each basis state is the
                "reference state", i.e. the one of which the timetrace
                 is substracted. Can also be passed as a dictionary where
                 keys are the qubit names and the values are lists of basis states
                 in case different bases should be used for different qubits.
                 (default:  ["ge", "ef"] when more than 2 traces are passed to
                 the analysis, ['ge'] if 2 traces are measured.)
            orthonormalize (bool): Whether to orthonormalize the weight basis
                (default: True)
            scale_weights (bool): scales the weights near unity to avoid
                loss of precision on FPGA if weights are too small (default: True)
            filter_residual_tones (bool): Whether to filter the measured
                weights. Specify filter using ``residual_tone_filter_fcn``.
                Creates new field ``'optimal_weights_unfiltered'`` in
                ``analysis_params_dict`` to save weights before filtering
                (default: True)
            residual_tone_filter_fcn (fcn, str): function (``freq``, ``ro_freq``) ->
                float. For a given value of the parameter ``ro_freq`` (in Hz),
                the function value indicates the complex scaling factor by
                which a frequency component at frequency ``freq`` (in Hz) is
                multiplied. Can be a str that parses into a function using
                eval(). (default: Gaussian centered at ``ro_freq`` with sigma
                ``residual_tone_filter_sigma``)
            residual_tone_filter_sigma (float): specifies the width
                of the Gaussian filter. Ignored if a ``residual_tone_filter_fcn``
                is provided. (default: 1e7 (10 MHz))
            plot_end_time (float): specifies the time up to which the
                timetraces and weights are plotted, no effect if None (default:
                None)
        """
        self.qb_names = qb_names
        super().__init__(**kwargs)

        if self.job is None:
            self.create_job(qb_names=qb_names, **kwargs)

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        if isinstance(self.raw_data_dict, dict):
            self.raw_data_dict = (self.raw_data_dict, )

        if self.qb_names is None:
            # get all qubits from cal_points of first timetrace
            cp = CalibrationPoints.from_string(
                self.get_param_value('cal_points', None, 0))
            self.qb_names = deepcopy(cp.qb_names)

        self.channel_map = self.get_param_value(
            'meas_obj_value_names_map', self.get_param_value('channel_map'))
        if self.channel_map is None:
            # assume same channel map for all timetraces (pick 0th)
            value_names = self.raw_data_dict[0]['value_names']
            if np.ndim(value_names) > 0:
                value_names = value_names
            if 'w' in value_names[0]:
                # FIXME: avoid dependency ana_v2 - ana_v3
                self.channel_map = hlp_mod.get_qb_channel_map_from_file(
                    self.qb_names, value_names=value_names,
                    file_path=self.raw_data_dict['folder'])
            else:
                self.channel_map = {}
                for qbn in self.qb_names:
                    self.channel_map[qbn] = value_names

        if len(self.channel_map) == 0:
            raise ValueError('No qubit RO channels have been found.')

    def process_data(self):
        super().process_data()
        pdd = self.proc_data_dict

        pdd['analysis_params_dict'] = dict()
        ana_params = pdd['analysis_params_dict']
        ana_params['timetraces'] = defaultdict(dict)
        ana_params['optimal_weights'] = defaultdict(dict)
        ana_params['optimal_weights_basis_labels'] = defaultdict(dict)
        ana_params['means'] = defaultdict(dict)

        if self.get_param_value('filter_residual_tones', True):
            filter_fcn = self._get_residual_tone_filter_fcn()

        for qb_indx, qbn in enumerate(self.qb_names):
            # retrieve time traces

            if len(self.channel_map[qbn]) != 2:
                raise NotImplementedError(
                    'This analysis does not support optimal weight '
                    f'measurement based on {len(self.channel_map[qbn])} '
                    f'ro channels. Try again with 2 RO channels.')

            twoD = self.get_param_value('TwoD', False)

            if twoD: # called by measure.OptimalWeights
                rdd = self.raw_data_dict[0]
                ttrace_per_ro_ch = [rdd["measured_data"][ch]
                                    for ch in self.channel_map[qbn]]
                states = eval(self.get_param_value('states'))
                if len(states[0]) == 1: # [('g',), ('e',), ]
                    states = [s[0] for s in states]
                else: # [('g', 'e'), ('e', 'f'), ]
                    states = [s[qb_indx] for s in states]
                for i, state in enumerate(states):
                    ana_params['timetraces'][qbn].update(
                        {state: ttrace_per_ro_ch[0][:,i] +
                                1j * ttrace_per_ro_ch[1][:,i]})

            else: # called by mqm.find_optimal_weights
                for i, rdd in enumerate(self.raw_data_dict):
                    ttrace_per_ro_ch = [rdd["measured_data"][ch]
                                        for ch in self.channel_map[qbn]]
                    cp = CalibrationPoints.from_string(
                        self.get_param_value('cal_points', None, i))
                    # get state of qubit. There can be only one cal point per
                    # sequence when using uhf for time traces, so it is the
                    # 0th state
                    qb_state = cp.states[0][cp.qb_names.index(qbn)]
                    # store all timetraces in same pdd for convenience
                    ana_params['timetraces'][qbn].update(
                        {qb_state: ttrace_per_ro_ch[0] + 1j *ttrace_per_ro_ch[1]})

            timetraces = ana_params['timetraces'][qbn] # for convenience
            basis_labels = self.get_param_value('acq_weights_basis', None, 0)
            n_labels = min(len(ana_params['timetraces'][qbn]) - 1, 2)
            if basis_labels is None:
                # guess basis labels from # states measured
                basis_labels = ["ge", "ef"][:n_labels]

            if isinstance(basis_labels, dict):
                # if different basis for qubits, then select the according one
                basis_labels = basis_labels[qbn]
            # check that states from the basis are included in mmnt
            for bs in basis_labels:
                for qb_s in bs:
                     assert qb_s in timetraces,\
                         f'State: {qb_s} on {qbn} was not provided in the given ' \
                         f'timestamps but was requested as part of the basis' \
                         f' {basis_labels}. Please choose another weight basis.'
            basis = np.array([timetraces[b[1]] - timetraces[b[0]]
                              for b in basis_labels])

            # orthonormalize if required
            if self.get_param_value("orthonormalize", False) and len(basis):
                # We need to consider the integration weights as a vector of
                # real numbers to ensure the Gram-Schmidt transformation of the
                # weights leads to a linear transformation of the integrated
                # readout results (relates to how integration is done on UHF,
                # see One Note: Surface 17/ATC75 M136 S17HW02 Cooldown 5/
                # 210330 Notes on orthonormalizing readout weights
                basis_real = np.hstack((basis.real, basis.imag), )
                basis_real = math.gram_schmidt(basis_real.T).T
                basis =    basis_real[:,:basis_real.shape[1]//2] + \
                        1j*basis_real[:,basis_real.shape[1]//2:]
                basis_labels = [bs + "_ortho" if bs != basis_labels[0] else bs
                                for bs in basis_labels]

            # scale if required
            if self.get_param_value('scale_weights', True) and len(basis):
                k = np.amax([(np.max(np.abs(b.real)),
                              np.max(np.abs(b.imag))) for b in basis])
                basis /= k

            if self.get_param_value('filter_residual_tones', twoD):
                basis = self._get_filtered_basis(basis, filter_fcn, qbn)

            ana_params['optimal_weights'][qbn] = basis
            ana_params['optimal_weights_basis_labels'][qbn] = basis_labels

            # calculation of centroids
            ana_params['means'][qbn] = [[
                np.sum(np.real(timetraces[state]) * np.real(basis[i,:]))
                + np.sum(np.imag(timetraces[state]) * np.imag(basis[i,:]))
                for i, _ in enumerate(basis_labels)]
                for state in 'gef'[:n_labels + 1]]

            self.save_processed_data()

    def _get_residual_tone_filter_fcn(self):
        pdd = self.proc_data_dict
        ana_params = pdd['analysis_params_dict']

        ana_params['optimal_weights_unfiltered'] = defaultdict(dict)
        filter_fcn = self.get_param_value(
            'residual_tone_filter_fcn', None)
        if filter_fcn is None:
            # if no filter is given, use gaussian filter with width sigma
            sigma = self.get_param_value('residual_tone_filter_sigma', 1e7)
            filter_fcn = lambda f, f0: np.exp(
                -.5 * (f - f0) ** 2 / sigma ** 2)
        else:
            try:
                filter_fcn = eval(filter_fcn) if \
                    isinstance(filter_fcn, str) else filter_fcn
            except SyntaxError:
                log.warning(
                    'Could not parse the custom filter function. '
                    'Either pass a valid lambda function '
                    'directly or as a string')
        return filter_fcn

    def _get_filtered_basis(self, basis, filter_fcn, qbn):
        """
        Residual/spurious tone filtering, only supported if called by
        OptimalWeights.

        Applies ``filter_fcn`` to the time trace in ``basis`` with the
        RO-frequency and RO modulation frequency of the qubit ``qbn``, returns
        the modified basis.
        """
        pdd = self.proc_data_dict
        ana_params = pdd['analysis_params_dict']

        # safe copy for comparison
        ana_params['optimal_weights_unfiltered'][qbn] = deepcopy(basis)
        # TODO: Allow for per qb/acq_instr sampling rates. This change
        #  requires the respective changes in readout.OptimalWeights.
        sampling_rate = self.get_param_value('acq_sampling_rate', None)
        if sampling_rate is None:
            raise ValueError("Please provide acq_sampling_rate.")
        ro_freq = self.get_instrument_setting(f'{qbn}.ro_freq')
        ro_mod_freq = self.get_instrument_setting(f'{qbn}.ro_mod_freq')

        if np.ndim(basis) == 1:
            basis = self._filter_residual_tones(
                basis, ro_freq, ro_mod_freq, sampling_rate, filter_fcn)
        else:
            assert np.ndim(basis) == 2, "Basis arr should only be 2D."
            for i in range(len(basis)):
                basis[i] = self._filter_residual_tones(
                    basis[i], ro_freq, ro_mod_freq, sampling_rate,
                    filter_fcn)
        return basis

    @staticmethod
    def _filter_residual_tones(w, ro_freq, ro_mod_freq, sampling_rate,
                               filter_fcn):
        """
        Filtering of timetraces to reduce residual tones and noise. Filtering
        is carried out in fourier space in units of MHz.

        Args:
            w: complex weigths I + .j * Q to be filtered
            ro_freq: ro freq of qb (in Hz)
            ro_mod_freq: ro modulation freq of qb (in Hz)
            sampling_rate: sampling rate of timetrace acq_instr
            filter_fcn: filter to be applied around ro_freq, see docstring of
            __init__ for more details.

        Returns:
            filtered complex weights in time space
        """
        w_spec = np.fft.fft(np.conj(w))
        w_spec = np.fft.fftshift(w_spec)
        f = np.fft.fftfreq(len(w_spec), 1e6 / sampling_rate)
        f = np.fft.fftshift(f) + ro_freq / 1e6 - ro_mod_freq / 1e6
        w_spec *= filter_fcn(f * 1e6, ro_freq)
        w = np.fft.ifftshift(w_spec)
        w = np.fft.ifft(w)
        w /= np.max(np.abs(w))
        return np.conj(w)

    def prepare_plots(self):

        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        twoD = self.get_param_value('TwoD', False)
        states = self.get_param_value('states', None)
        num_states = len(eval(states)) if twoD else len(rdd)
        ana_params = self.proc_data_dict['analysis_params_dict']
        plot_end_time = self.get_param_value('plot_end_time', None)
        for qbn in self.qb_names:
            mod_freq = self.get_instrument_setting(f'{qbn}.ro_mod_freq')
            tbase = rdd[0]['hard_sweep_points']
            plot_end_idx = np.where(tbase < plot_end_time)[0][-1] \
                if plot_end_time is not None else None
            basis_labels = pdd["analysis_params_dict"][
                'optimal_weights_basis_labels'][qbn]
            title = 'Optimal SNR weights ' + qbn + \
                    "".join(['\n' + rddi["timestamp"] for rddi in rdd]) \
                            + f'\nWeight Basis: {basis_labels}'
            plot_name = f"weights_{qbn}"
            xlabel = "Time, $t$"
            modulation = np.exp(2j * np.pi * mod_freq * tbase)

            for ax_id, (state, ttrace) in \
                enumerate(ana_params["timetraces"][qbn].items()):
                for func, label in zip((np.real, np.imag), ('I', "Q")):
                    # plot timetraces for each state, I and Q channels
                    self.plot_dicts[f"{plot_name}_{state}_{label}"] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': tbase[:plot_end_idx],
                        "marker": "",
                        'yvals': func(ttrace*modulation)[:plot_end_idx],
                        'ylabel': 'Voltage, $V$',
                        'yunit': 'V',
                        "sharex": True,
                        "setdesc": label + rf"$_{{\vert {state}\rangle}}$",
                        "setlabel": "",
                        "do_legend":True,
                        "legend_pos": "best",
                        'legend_ncol': 2,
                        'numplotsx': 1,
                        'numplotsy': num_states + 1, # #states + 1 for weights
                        'plotsize': (10,
                                     (num_states + 1) * 3), # 3 inches per plot
                        'title': title if ax_id == 0 else ""}
            ax_id = len(ana_params["timetraces"][qbn]) # id plots for weights
            for i, weights in enumerate(ana_params['optimal_weights'][qbn]):
                for func, label in zip((np.real, np.imag), ('I', "Q")):
                    self.plot_dicts[f"{plot_name}_weights_{label}_{i}"] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': tbase[:plot_end_idx],
                        'xlabel': xlabel,
                        "setlabel": "",
                        "marker": "",
                        'xunit': 's',
                        'yvals': func(weights * modulation)[:plot_end_idx],
                        'ylabel': 'Voltage, $V$ (arb.u.)',
                        "sharex": True,
                        "xrange": [self.get_param_value('tmin', min(tbase[:plot_end_idx]), 0),
                                   self.get_param_value('tmax', max(tbase[:plot_end_idx]), 0)],
                        "setdesc": label + rf"$_{i+1}$",
                        "do_legend": True,
                        "legend_pos": "best",
                        'legend_ncol': 2,
                        }


class MultiQutrit_Singleshot_Readout_Analysis(MultiQubit_TimeDomain_Analysis):
    """
    Analysis class for parallel SSRO qutrit/qubit calibration. It is a child class
    from the tda.MultiQubit_Timedomain_Analysis as it uses the same functions to
    - preprocess the data to remove active reset/preselection
    - extract the channel map
    - reorder the data per qubit
    Note that in the future, it might be useful to transfer these functionalities
    to the base analysis.
    """

    def __init__(self,
                 options_dict: dict = None, auto=True, **kw):
        """
        Initializes the SSRO analysis class.

        Args:
            qb_names (list): name of the qubits to analyze (can be a subset
                of the measured qubits)
            auto (bool): Start analysis automatically (default: True)
            **kw:
                options_dict: Dictionary for analysis options (see below)

        options_dict keywords:
            hist_scale (str) : scale for the y-axis of the 1D histograms:
                "linear" or "log" (default: 'log')
            verbose (bool) : see BaseDataAnalysis
            presentation_mode (bool) : see BaseDataAnalysis
            classif_method (str): how to classify the data.
                'ncc' : default. Nearest Cluster Center
                'gmm': gaussian mixture model (default)
                'threshold': finds optimal vertical and horizontal thresholds.
            retrain_classifier (bool): whether to retrain the classifier
                (default) or to use the classifier stored in instrument_setting
            classif_kw (dict): kw to pass to the classifier.
            multiplexed_ssro (bool): whether to perform analysis for a
                multiplexed measurement (default: False)
            plot (bool): Whether to plot any plots (default: True)
            plot_single_qb_plots (bool): Whether to plot the state assignment
                probability matrices and classification plots (for every qb and
                sweep point) (default: True if no sweep was performed)
            plot_mux_plots (bool): Whether to plot the multiplexed state
                probability matrix (for every sweep point) if applicable
                (default: True if ``multiplexed_ssro``)
            plot_sweep_plots (bool): Whether to plot single qb trend plots in
                sweeps if applicable (default: True if a sweep was performed)
            plot_mux_sweep_plots (bool): Whether to plot multiplexed trend
                plots in sweeps if applicable (default: True if a multiplexed
                sweep was performed)
            plot_metrics (list of dicts): Custom metrics of the state
                assignment probability matrix to be plotted when sweeping. The
                dictionaries contain the keys:
                'metric' (str, required): string can be parsed into a lambda
                    expression using eval(). The lambda takes the state assignment
                    probability matrices as a 2D-np.array and returns a float.
                'plot_name' (str): Title of the custom plot
                'yscale' (str): 'linear' or 'log'
                Ignored, if no sweep was performed (defaults to plotting
                fidelity and infidelity plots)
            multiplexed_plot_metrics (list of dict): Same as ``plot_metrics``,
                but 'metric' lambda takes the multiplexed state assignment
                probability matrices. Ignored if ``multiplexed_ssro`` is False
                and no sweep was performed.
            plot_init_columns (bool): Whether to plot additional column
                representing the percentage of shots that were not filtered in
                preselection (default: True if ``multiplexed_ssro``)
            n_shots_to_plot (int): Truncates the number of shots to be
                plotted if not None. Only affects the plotting (default: None)
        """
        super().__init__(options_dict=options_dict, auto=False,
                         **kw)
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'measured_data': 'measured_data',
            'value_names': 'value_names',
            'value_units': 'value_units'}
        self.numeric_params = []
        self.DEFAULT_CLASSIF = "gmm"
        self.classif_method = self.options_dict.get("classif_method",
                                                    self.DEFAULT_CLASSIF)

        self.create_job(options_dict=options_dict, auto=auto, **kw)

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        self.preselection = \
            self.get_param_value("preparation_params",
                                 {}).get("preparation_type", "wait") == "preselection"
        default_states_info = defaultdict(dict)
        default_states_info.update({"g": {"label": r"$|g\rangle$"},
                               "e": {"label": r"$|e\rangle$"},
                               "f": {"label": r"$|f\rangle$"}
                               })

        self.states_info = \
            self.get_param_value("states_info",
                                {qbn: deepcopy(default_states_info)
                                 for qbn in self.qb_names})

    def process_data(self):
        """
        Create the histograms based on the raw data
        """
        ######################################################
        #  Separating data into shots for each level         #
        ######################################################

        # remove 'data_type' from metadata before calling super, as this is
        # required to circumvent the postprocessing to get the ssro data.
        self.metadata.pop('data_type', None)

        super().process_data()
        del self.proc_data_dict['data_to_fit'] # not used in this analysis

        # prepare data in convenient format, i.e. arrays per qubit and per state
        # e.g. {'qb1': {'g': np.array of shape (n_shots, n_ro_ch}, ...}, ...}
        shots_per_qb = dict()        # store shots per qb and per state
        presel_shots_per_qb = dict() # store preselection ro
        pdd = self.proc_data_dict    # for convenience of notation

        for qbn in self.qb_names:
            # shape is (2d_sweep_indx, n_shots, n_ro_ch) i.e. one column for
            # each ro_ch
            shots_per_qb[qbn] = np.array(
                [vals for ch, vals in
                 pdd['meas_results_per_qb'][qbn].items()]).T
            # make 3D array in case mqm.measure_ssro is used
            if len(shots_per_qb[qbn].shape) == 2:
                shots_per_qb[qbn] = np.expand_dims(shots_per_qb[qbn], axis=0)
            if self.preselection:
                # preselection shots were removed so look at raw data
                # and look at only the first out of every two readouts
                presel_shots_per_qb[qbn] = np.array(
                [vals[::2] for ch, vals in
                 pdd['meas_results_per_qb_raw'][qbn].items()]).T
                # make 3D array in case mqm.measure_ssro is used
                if len(presel_shots_per_qb[qbn].shape) == 2:
                    presel_shots_per_qb[qbn] = \
                        np.expand_dims(presel_shots_per_qb[qbn], axis=0)

        n_sweep_pts_dim1 = len(self.cp.states)
        n_sweep_pts_dim2 = np.shape(shots_per_qb[self.qb_names[0]])[0]
        n_shots = np.shape(shots_per_qb[self.qb_names[0]])[1]//n_sweep_pts_dim1

        # entry for every measured qubit and sweep point
        qb_dict = {qbn: [None]*n_sweep_pts_dim2 for qbn in self.qb_names}

        # create placeholders for analysis data
        pdd['data'] = {'X': deepcopy(qb_dict),
                       'prep_states': deepcopy(qb_dict),
                       'pred_states': deepcopy(qb_dict)}
        pdd['n_dim_2_sweep_points'] = n_sweep_pts_dim2
        pdd['avg_fidelities'] = deepcopy(qb_dict)
        pdd['best_fidelity'] = {}

        pdd['analysis_params'] = {
            'state_prob_mtx': deepcopy(qb_dict),
            'classifier_params': deepcopy(qb_dict),
            'means': deepcopy(qb_dict),
            'snr': deepcopy(qb_dict),
            'n_shots': np.shape(shots_per_qb[qbn])[1],
            'slopes': deepcopy(qb_dict),
        }
        pdd_ap = pdd['analysis_params']
        self.clf_ = deepcopy(qb_dict) # classifier

        # create placeholders for analysis with preselection
        if self.preselection:
            pdd['data_masked'] = {'X': deepcopy(qb_dict),
                                  'prep_states': deepcopy(qb_dict),
                                  'pred_states': deepcopy(qb_dict)}
            pdd['avg_fidelities_masked'] = deepcopy(qb_dict)
            pdd_ap['state_prob_mtx_masked'] = deepcopy(qb_dict)
            pdd_ap['n_shots_masked'] = deepcopy(qb_dict)
            pdd_ap['presel_fraction_per_state'] = deepcopy(qb_dict)

        for qbn, qb_shots in shots_per_qb.items():
            # iteration over 2nd sweep dim

            # assign every state the qb is prepared in a unique integer
            masks = {state: np.array(self.cp.get_states(qbn)[qbn]) == state
                     for state in self.states_info[qbn]}
            measured_states = [state for state in self.states_info[qbn]
                               if sum(masks[state]) > 0]
            state_integer = 0
            for state in measured_states:
                self.states_info[qbn][state]["int"] = state_integer
                state_integer += 1

            # note that if some states are repeated, they are assigned the
            # same label
            qb_states_integer_repr = \
                [self.states_info[qbn][s]["int"]
                 for s in self.cp.get_states(qbn)[qbn]]
            prep_states = np.tile(qb_states_integer_repr, n_shots)

            for dim2_sp_idx in range(n_sweep_pts_dim2):
                # create mapping to integer following ordering in cal_points.
                # Notes:
                # 1) the state_integer should to the order of pdd[qbn]['means']
                # so that when passing the init_means to the GMM model, it is
                # ensured that each gaussian component will predict the
                # state_integer associated to that state
                # 2) the mapping cannot be preestablished because the GMM
                # predicts labels in range(n_components). For instance, if a
                # qubit has states "g", "f" then the model will predicts 0's
                # and 1's, so the typical g=0, e=1, f=2 mapping would fail.
                # The number of different states can be different for each
                # qubit and therefore the mapping should also be done per qubit

                qb_means = {state:
                    np.mean(shots_per_qb[qbn][dim2_sp_idx,
                    np.tile(masks[state], n_shots)], axis=0)
                    for state in measured_states
                }

                assert np.ndim(qb_shots) == 3, \
                    f"Data must be a 3D array. Received shape " \
                    f"{qb_shots.shape}, ndim {np.ndim(qb_shots)}"\

                pred_states, clf_params, clf = \
                    self._classify(qb_shots[dim2_sp_idx], prep_states,
                                   method=self.classif_method, qb_name=qbn,
                                   dim2_sweep_indx=dim2_sp_idx,
                                   train_new_classifier=self.get_param_value(
                                       "retrain_classifier", True),
                                   means=list(qb_means.values()),
                                   **self.options_dict.get("classif_kw", dict()))
                # order "unique" states to have in usual order "gef" etc.
                state_labels_ordered = self._order_state_labels(
                    list(measured_states))
                # translate to corresponding integers
                state_labels_ordered_int = [self.states_info[qbn][s]['int'] for s in
                                            state_labels_ordered]
                fm = self.fidelity_matrix(prep_states, pred_states,
                                          labels=state_labels_ordered_int)

                # save processed data
                pdd_ap['means'][qbn][dim2_sp_idx] = deepcopy(qb_means)
                pdd['data']['X'][qbn][dim2_sp_idx] = \
                    deepcopy(qb_shots[dim2_sp_idx])
                pdd['data']['prep_states'][qbn][dim2_sp_idx] = \
                    deepcopy(prep_states)
                pdd['data']['pred_states'][qbn][dim2_sp_idx] = \
                    deepcopy(pred_states)
                pdd['avg_fidelities'][qbn][dim2_sp_idx] = \
                    np.trace(fm) / float(np.sum(fm))
                pdd_ap['state_prob_mtx'][qbn][dim2_sp_idx] = fm
                pdd_ap['classifier_params'][qbn][dim2_sp_idx] = clf_params

                if 'means_' in clf_params:
                    pdd_ap['snr'][qbn][dim2_sp_idx] = \
                        self._extract_snr(clf, state_labels_ordered)
                    pdd_ap['slopes'][qbn][dim2_sp_idx] = \
                        self._extract_slopes(clf, state_labels_ordered)

                self.clf_[qbn][dim2_sp_idx] = clf

                if self.preselection:
                    # redo with classification 1st of preselection and masking
                    pred_presel = self.clf_[qbn][dim2_sp_idx].predict(
                        presel_shots_per_qb[qbn][dim2_sp_idx])
                    try:
                        presel_filter = \
                            pred_presel == self.states_info[qbn]['g']['int']
                    except KeyError:
                        log.warning(f"{qbn}: Classifier not trained on g-state"
                                    f" to classify for preselection! "
                                    f"Skipping preselection data & figures.")
                        continue

                    if np.sum(presel_filter) == 0:
                        log.warning(f"{qbn}: No data left after preselection! "
                                    f"Skipping preselection data & figures.")
                        continue
                    qb_shots_masked = qb_shots[dim2_sp_idx][presel_filter]
                    prep_states_masked = prep_states[presel_filter]
                    pred_states = self.clf_[qbn][dim2_sp_idx].predict(
                        qb_shots_masked)
                    fm_masked = self.fidelity_matrix(
                        prep_states_masked, pred_states,
                        labels=state_labels_ordered_int)
                    pdd['avg_fidelities_masked'][qbn][dim2_sp_idx] = \
                        np.trace(fm_masked) / float(np.sum(fm_masked))
                    pdd['data_masked']['X'][qbn][dim2_sp_idx] = \
                        deepcopy(qb_shots_masked)
                    pdd['data_masked']['prep_states'][qbn][dim2_sp_idx] = \
                        deepcopy(prep_states_masked)
                    pdd['data_masked']['pred_states'][qbn][dim2_sp_idx] = \
                        deepcopy(pred_states)
                    pdd_ap['state_prob_mtx_masked'][qbn][dim2_sp_idx] = \
                        fm_masked
                    pdd_ap['n_shots_masked'][qbn][dim2_sp_idx] = \
                        qb_shots_masked.shape[0]

                    presel_frac = np.array([np.sum(prep_states_masked == s) /
                                            (np.sum(prep_states == s))
                                            for s in state_labels_ordered_int])

                    pdd_ap['presel_fraction_per_state'][qbn][dim2_sp_idx] = \
                        presel_frac

            fids = pdd['avg_fidelities_masked'][qbn] if \
                    self.preselection else pdd['avg_fidelities'][qbn]
            pdd['best_fidelity'][qbn] = {
                'fidelity': np.nanmax(fids),
                'sweep_index': np.nanargmax(fids)}

        if self.get_param_value('multiplexed_ssro', False):
            # perform data analysis for multiplexed SSRO measurements
            self.process_data_multiplexed(shots_per_qb, presel_shots_per_qb)

        self.save_processed_data()

    def process_data_multiplexed(self, shots_per_qb, presel_shots_per_qb):
        pdd = self.proc_data_dict    # for convenience of notation
        pdd_ap = pdd['analysis_params']
        n_sweep_pts_dim1 = len(self.cp.states)
        n_sweep_pts_dim2 = np.shape(shots_per_qb[self.qb_names[0]])[0]
        n_shots = np.shape(shots_per_qb[self.qb_names[0]])[1]//n_sweep_pts_dim1

        # have a list of unique multiplexed states
        unique_states = self._order_multiplexed_state_labels(
            np.unique(np.array(self.cp.states), axis=0))
        # assign every multiplexed state in unique_states an index
        state_idx = np.arange(len(unique_states))
        # prepared states as a list of multiplexed state indexes
        prep_states = np.array([
            np.argmax(np.all(unique_states == state, axis=-1))
            for state in np.array(self.cp.states)])
        prep_states = np.tile(prep_states, n_shots)
        # a list of which single qb states correspond to which mltplxed state
        states_int = np.array(
            [[self.states_info[self.cp.qb_names[i]][s]['int']
              for i, s in enumerate(state)] for state in unique_states])

        # create placeholders for analysis data
        pdd['mux_data'] = {'prep_states': prep_states,
                             'pred_states':
                                 np.zeros((n_sweep_pts_dim2,
                                           n_sweep_pts_dim1 * n_shots)),
                             'pred_states_raw':
                                 np.zeros((n_sweep_pts_dim2,
                                           n_sweep_pts_dim1 * n_shots,
                                           len(self.qb_names))),
                             'unique_states': list(unique_states)
                             }
        pdd_ap['mux_state_prob_mtx'] = [None] * n_sweep_pts_dim2
        pdd_ap['mux_n_shots'] = [None] * n_sweep_pts_dim2
        pdd['mux_avg_fidelities'] = np.zeros(n_sweep_pts_dim2)

        # create placeholders for analysis with preselection
        if self.preselection:
            pdd['mux_data_masked'] = \
                {'prep_states': [None] * n_sweep_pts_dim2,
                 'pred_states': [None] * n_sweep_pts_dim2,
                 'pred_presel_states_raw':
                     np.zeros((n_sweep_pts_dim2,
                               n_sweep_pts_dim1 * n_shots,
                               len(self.qb_names))),
                 'presel_filter': [None] * n_sweep_pts_dim2,
                 }
            pdd['mux_avg_fidelities_masked'] = [None] * n_sweep_pts_dim2
            pdd_ap['mux_state_prob_mtx_masked'] = [None] * n_sweep_pts_dim2
            pdd_ap['mux_n_shots_masked'] = [None] * n_sweep_pts_dim2
            pdd_ap['mux_presel_fraction_per_state'] = [None]*n_sweep_pts_dim2

        for dim2_sp_idx in range(n_sweep_pts_dim2):
            for qbn, qb_shots in shots_per_qb.items():
                assert np.ndim(qb_shots) == 3, \
                    f"Data must be a 3D array. Received shape " \
                    f"{qb_shots.shape}, ndim {np.ndim(qb_shots)} for {qbn}"

            # single qb predictions (each row contains single qb predictions)
            pred_qb_states = np.array([self.clf_[qbn][dim2_sp_idx].predict(
                shots_per_qb[qbn][dim2_sp_idx]) for qbn in self.cp.qb_names]).T
            # find corresponding multiplexed state int
            pred_state_bools = np.array([np.all(states_int == pred, axis=-1)
                                         for pred in pred_qb_states])
            # we expect the measured state to correspond to one or none of the
            # prepared multiplexed states
            sum_state_bools = np.sum(pred_state_bools, axis=-1)
            assert np.all(sum_state_bools <= 1), 'A measurement result ' \
                                                 'could not be uniquely ' \
                                                 'assigned to a state'
            # create a mask for the shots that were not assigned to a state in
            # unique_states
            unkn_state_mask = sum_state_bools == 0
            if np.sum(unkn_state_mask) > 0:
                log.warning(f"{np.sum(unkn_state_mask)} measurements were "
                            f"assigned a state not given in 'states'. "
                            f"Ignoring these measurements in plots.")

            # count the shots that were assigned a state in unique_states
            pdd_ap['mux_n_shots'][dim2_sp_idx] = \
                np.sum(unkn_state_mask == False)
            # per shot array containing assigned multiplexed state int
            pred_states = np.array([np.argmax(np.all(states_int == pred,
                                                     axis=-1))
                                    for pred in pred_qb_states])
            # assign state -1 to states not in unique_states
            pred_states[unkn_state_mask] = -1

            pdd['mux_data']['pred_states_raw'] \
                [dim2_sp_idx] = pred_qb_states
            pdd['mux_data']['pred_states'][dim2_sp_idx] = pred_states

            fm = self.fidelity_matrix(prep_states, pred_states,
                                      labels=state_idx)
            pdd['mux_avg_fidelities'][dim2_sp_idx] = np.trace(
                fm) / float(np.sum(fm))

            # save fidelity matrix
            pdd_ap['mux_state_prob_mtx'][dim2_sp_idx] = fm

            if self.preselection:
                # redo with classification first of preselection & masking
                pred_presel_qb_states = np.array(
                    [self.clf_[qbn][dim2_sp_idx].predict(
                        presel_shots_per_qb[qbn][dim2_sp_idx]) for qbn in
                        self.qb_names]).T
                try:
                    init_state = np.array(
                        [self.states_info[qbn]['g']['int']
                         for qbn in self.qb_names])
                except KeyError:
                    log.warning(f"{qbn}: Classifier not trained on g-state"
                                f" to classify for preselection! "
                                f"Skipping multiplexed preselection data "
                                f"& figures.")
                    continue

                presel_filter = np.all(pred_presel_qb_states == init_state,
                                       axis=-1)

                pdd['mux_data_masked']['pred_presel_states_raw'] \
                    [dim2_sp_idx] = pred_presel_qb_states
                pdd['mux_data_masked']['presel_filter'] \
                    [dim2_sp_idx] = presel_filter

                if np.sum(presel_filter) == 0:
                    log.warning(
                        f"Sweep point {dim2_sp_idx} (dim 2): "
                        f"No data left after preselection! "
                        f"Skipping preselection data & figures.")
                    continue

                prep_states_masked = prep_states[presel_filter]
                pred_states_masked = pred_states[presel_filter]

                pdd['mux_data_masked']['pred_states'][dim2_sp_idx] = \
                    pred_states_masked
                pdd['mux_data_masked']['prep_states'][dim2_sp_idx] = \
                    prep_states_masked

                presel_frac = np.array([np.sum(prep_states_masked == s) /
                                        (np.sum(prep_states == s))
                                        for s in state_idx])

                pdd_ap['mux_presel_fraction_per_state'][dim2_sp_idx] = \
                    presel_frac

                fm_masked = self.fidelity_matrix(prep_states_masked,
                                                 pred_states_masked,
                                                 labels=state_idx)
                pdd_ap['mux_state_prob_mtx_masked'][
                    dim2_sp_idx] = fm_masked
                pdd['mux_avg_fidelities_masked'][dim2_sp_idx] = \
                    np.trace(np.nan_to_num(fm_masked)) \
                    / float(np.nansum(fm_masked))
                pdd_ap['mux_n_shots_masked'][dim2_sp_idx] = \
                    sum(presel_filter)

        fids = pdd['mux_avg_fidelities_masked'] if \
            self.preselection else pdd['mux_avg_fidelities']
        pdd['mux_best_fidelity'] = {
            'fidelity': np.nanmax(fids),
            'sweep_index': np.nanargmax(fids)}
        return

    @staticmethod
    def _extract_snr(gmm=None,  state_labels=None, clf_params=None,):
        """
        Extracts SNR between pairs of states. SNR is defined as dist(m1,
        m2)/mean(std1, std2), where dist = L2 norm, m1, m2 are the means of the
        pair of states and std1, std2 are the "standard deviation" (obtained
        from the confidence ellipse of the covariance if 2D).
        :param gmm: Gaussian mixture model
        :param clf_params: Classifier parameters. Not implemented but could
        reconstruct gmm from clf params. Would be more analysis friendly.
        :param state_labels (list): state labels for the SNR dict. If not provided,
            tuples indicating the index of the state pairs is used.
        :return: snr (dict): e.g. {"ge": 2.4} or  {"ge": 3, "ef": 2, "gf": 4}
        """
        snr = {}
        if clf_params is not None:
            raise NotImplementedError("Look in a_tools.predict_probas to "
                                      "recreate GMM from clf_params")
        means = MultiQutrit_Singleshot_Readout_Analysis._get_means(gmm)
        covs = MultiQutrit_Singleshot_Readout_Analysis._get_covariances(gmm)
        n_states = len(means)
        if n_states >= 2:
            state_pairs = list(itertools.combinations(np.arange(n_states), 2))
            for sp in state_pairs:
                m0, m1 = means[sp[0]], means[sp[1]]
                if len(m0) == 1:
                    # pad second element to treat as 2d
                    m0, m1 = np.concatenate([m0, [0]]), np.concatenate([m1, [0]])
                dist = np.linalg.norm(m0 - m1)
                std0_candidates = math.find_intersect_line_ellipse(
                    math.slope(m0- m1),
                    *math.get_ellipse_radii_and_rotation(covs[sp[0]]))
                idx = np.argmin([np.linalg.norm(std0_candidates[0] - m1),
                                 np.linalg.norm(std0_candidates[1] -
                                                m1)]).flatten()[0]
                std0 = np.linalg.norm(std0_candidates[idx])
                std1_candidates = math.find_intersect_line_ellipse(
                    math.slope(m0 - m1),
                    *math.get_ellipse_radii_and_rotation(covs[sp[1]]))
                idx = np.argmin([np.linalg.norm(std0_candidates[0] - m0),
                                 np.linalg.norm(std0_candidates[1] -
                                                m1)]).flatten()[0]
                std1 = np.linalg.norm(std1_candidates[idx])
                label = state_labels[sp[0]] + state_labels[sp[1]] \
                    if state_labels is not None else sp

                snr.update({label: dist/np.mean([std0, std1])})
        return snr

    @staticmethod
    def _extract_slopes(gmm=None,  state_labels=None, clf_params=None, means=None):
        """
        Extracts slopes of line connecting two means of different states.
        :param gmm: Gaussian mixture model from which means are extracted
        :param clf_params: Classifier parameters from which means are extracted.
        :param state_labels (list): state labels for the SNR dict. If not provided,
            tuples indicating the index of the state pairs is used.
        :param means (array):
        :return: slopes (dict): e.g. {"ge": 0.1} or  {"ge": 0.1, "ef": 2,
        "gf": 0.4}
        """
        slopes = {}
        if clf_params is not None:
            if not 'means_' in clf_params:
                raise ValueError(f"could not find 'means_' in clf_params:"
                                 f" {clf_params}. Please pass in means directly "
                                 f"provide a classifier that fits means.")
            means = clf_params.get('means_')
        if gmm is not None:
            means = MultiQutrit_Singleshot_Readout_Analysis._get_means(gmm)
        if means is None:
            raise ValueError('Please provide one of kwarg gmm, clf_params or '
                             'means to extract the means of the different '
                             'distributions')
        n_states = len(means)
        if n_states >= 2:
            state_pairs = list(itertools.combinations(np.arange(n_states), 2))
            for sp in state_pairs:
                m0, m1 = means[sp[0]], means[sp[1]]
                if len(m0) == 1:
                    # pad second element to treat as 2d
                    m0, m1 = np.concatenate([m0, [0]]), np.concatenate([m1, [0]])

                label = state_labels[sp[0]] + state_labels[sp[1]] \
                    if state_labels is not None else sp
                slopes.update({label: math.slope(m0 - m1)})
        return slopes

    def _classify(self, X, prep_state, method, qb_name, dim2_sweep_indx=None,
                  train_new_classifier=True, means=None, **kw):
        """

        Args:
            X: measured data to classify
            prep_state: prepared states (true values)
            type: classification method
            qb_name: name of the qubit to classify
            dim2_sweep_indx: specifies which slice of self.proc_data_dict is to
            be used.
            train_new_classifier: Whether to fit a new classifier or to use the
            one specified in instrument_settings. Only implemented for gmm.
            means: 2D array for the initialisation of the GMM means
        Returns:

        """
        if np.ndim(X) == 1:
            X = X.reshape((-1,1))
        params = dict()

        if method == 'ncc':
            if not train_new_classifier:
                raise NotImplementedError("NCC classification doesn't "
                                          "support classification from "
                                          "trained classifier.")
            ncc = SSROQutrit.NCC(
                self.proc_data_dict['analysis_params']['means'][qb_name])
            pred_states = ncc.predict(X)
            # self.clf_ = ncc
            return pred_states, dict(), ncc

        elif method == 'gmm':
            if train_new_classifier:
                cov_type = kw.pop("covariance_type", "tied")
                # full allows full covariance matrix for each level. Other
                # options see GM documentation
                # assumes if repeated state, should be considered of the same
                # component this classification method should not be used for
                # multiplexed SSRO analysis
                n_qb_states = len(np.unique(self.cp.get_states(qb_name)[qb_name]))
                # give same weight to each class by default
                weights_init = kw.pop("weights_init",
                                      np.ones(n_qb_states)/n_qb_states)

                # calculate delta of means and set tol and cov based on this
                delta_means = np.array([[np.linalg.norm(mu_i - mu_j) for mu_i in means]
                                        for mu_j in means]).flatten().max()

                tol = delta_means/10 if delta_means > 1e-5 else 1e-6
                tol = kw.pop("tol", tol)
                reg_covar = tol**2

                gm = GM(n_components=n_qb_states,
                        covariance_type=cov_type,
                        random_state=0,
                        tol=tol,
                        reg_covar=reg_covar,
                        weights_init=weights_init,
                        means_init=means, **kw)
                gm.fit(X)
            else:  # restore GMM from instrument_setting
                gm = GM()
                params = self.get_instrument_setting(
                    f"{qb_name}.acq_classifier_params")
                if not isinstance(params, dict):
                    raise ValueError(f"Please make sure that {qb_name} has a "
                                     f"trained GMM classifier.")
                for name in ['means_', 'covariances_', 'covariance_type',
                             'weights_', 'precisions_cholesky_']:
                    if name not in params.keys():
                        raise ValueError(f"Classifier stored in instrument "
                                         f"setting doesn't contain the "
                                         f"parameter {name}.")
                    setattr(gm, name, params[name])
                setattr(gm, 'n_components', params['means_'].shape[0])

            pred_states = np.argmax(gm.predict_proba(X), axis=1)

            params['means_'] = gm.means_
            params['covariances_'] = gm.covariances_
            params['covariance_type'] = gm.covariance_type
            params['weights_'] = gm.weights_
            params['precisions_cholesky_'] = gm.precisions_cholesky_
            return pred_states, params, gm

        elif method == "threshold":
            if not train_new_classifier:
                raise NotImplementedError("Threshold classification doesn't "
                                          "support classification from "
                                          "trained classifier.")
            tree = DTC(max_depth=kw.pop("max_depth", X.shape[1]),
                       random_state=0, **kw)
            tree.fit(X, prep_state)
            pred_states = tree.predict(X)
            params["thresholds"], params["mapping"] = \
                self._extract_tree_info(tree, self.cp.get_states(qb_name)[qb_name])
            if len(params["thresholds"]) != X.shape[1]:
                msg = "Best 2 thresholds to separate this data lie on axis {}" \
                    ", most probably because the data is not well separated." \
                    "The classifier attribute clf_ can still be used for " \
                    "classification (which was done to obtain the state " \
                    "assignment probability matrix), but only the threshold" \
                    " yielding highest gini impurity decrease was returned." \
                    "\nTo circumvent this problem, you can either choose" \
                    " a second threshold manually (fidelity will likely be " \
                    "worse), make the data more separable, or use another " \
                    "classification method."
                log.warning(msg.format(list(params['thresholds'].keys())[0]))
            return pred_states, params, tree
        elif method == "threshold_brute":
            raise NotImplementedError()
        else:
            raise NotImplementedError("Classification method: {} is not "
                                      "implemented. Available methods: {}"
                                      .format(method, ['ncc', 'gmm',
                                                       'threshold']))

    @staticmethod
    def _get_covariances(gmm, cov_type=None):
       return SSROQutrit._get_covariances(gmm, cov_type=cov_type)

    @staticmethod
    def _get_means(gmm):
        return gmm.means_

    @staticmethod
    def fidelity_matrix(prep_states, pred_states, levels=('g', 'e', 'f'),
                        plot=False, labels=None, normalize=True):

        return SSROQutrit.fidelity_matrix(prep_states, pred_states,
                                          levels=levels, plot=plot,
                                          normalize=normalize, labels=labels)

    @staticmethod
    def plot_fidelity_matrix(fm, target_names, prep_names=None,
                             title="State Assignment Probability Matrix",
                             auto_shot_info=True, ax=None,
                             cmap=None, normalize=True, show=False,
                             plot_compact=False, presel_column=None,
                             plot_norm=None):
        return SSROQutrit.plot_fidelity_matrix(
            fm, target_names, prep_names=prep_names, title=title, ax=ax,
            auto_shot_info=auto_shot_info,
            cmap=cmap, normalize=normalize, show=show,
            plot_compact=plot_compact, presel_column=presel_column,
            plot_norm=plot_norm)

    @staticmethod
    def _extract_tree_info(tree_clf, class_names=None):
        return SSROQutrit._extract_tree_info(tree_clf,
                                             class_names=class_names)

    @staticmethod
    def _to_codeword_idx(tuple):
        return SSROQutrit._to_codeword_idx(tuple)

    @staticmethod
    def plot_scatter_and_marginal_hist(data, y_true=None, plot_fitting=False,
                                       **kwargs):
        return SSROQutrit.plot_scatter_and_marginal_hist(
            data, y_true=y_true, plot_fitting=plot_fitting, **kwargs)

    @staticmethod
    def plot_clf_boundaries(X, clf, ax=None, cmap=None, spacing=None):
        return SSROQutrit.plot_clf_boundaries(X, clf, ax=ax, cmap=cmap,
                                              spacing=spacing)

    @staticmethod
    def plot_std(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
        return SSROQutrit.plot_std(mean, cov, ax,n_std=n_std,
                                   facecolor=facecolor, **kwargs)

    @staticmethod
    def plot_1D_hist(data, y_true=None, plot_fitting=True,
                     **kwargs):
        return SSROQutrit.plot_1D_hist(data, y_true=y_true,
                                       plot_fitting=plot_fitting, **kwargs)

    @staticmethod
    def _order_state_labels(states_labels, order=STATE_ORDER):
        """
        Orders state labels according to provided ordering. e.g. for default
        ("f", "e", "g") would become ("g", "e", "f")
        Args:
            states_labels (list, tuple): list of states_labels
            order (str): custom string order

        Returns:

        """
        try:
            indices = [order.index(s) for s in states_labels]
            order_for_states = np.argsort(indices).astype(np.int32)
            return np.array(states_labels)[order_for_states]

        except Exception as e:
            log.error(f"Could not find order in state_labels:"
                      f"{states_labels}. Probably because one or several "
                      f"states are not part of '{order}'. Error: {e}."
                      f" Returning same as input order")
            return states_labels

    @staticmethod
    def _order_multiplexed_state_labels(states_labels, order=STATE_ORDER,
                                        most_significant_state_first=True):
        """
        Orders multiplexed state labels according to provided ordering. e.g.
        for default ordering [('e', 'g'), ('g', 'g'), ('e', 'e'), ('g', 'e')]
        becomes [['g', 'g'], ['e', 'g'], ['g', 'e'], ['e', 'e']].
        Args:
            states_labels (2D list, tuple): list of states_labels to be sorted
            order (str): custom string order
            most_significant_state_first: setting this to False orders by the
            last state in the array first, default True.

        Returns:
            ordered list of multiplexed states
        """
        try:
            key = lambda x: [order.index(c) for c in x][
                            ::1 if most_significant_state_first else -1]
            return np.array(sorted(states_labels, key=key))
        except Exception as e:
            log.error(f"Could not find order in state_labels:"
                      f"{states_labels}. Probably because one or several "
                      f"states are not part of '{order}'. Error: {e}."
                      f" Returning same as input order")
            return states_labels

    def plot(self, **kwargs):
        """
        Main plotting function for MeasureSSRO.

        Infers which measurement was run (multiplexed and/or sweep) and which
        single, multiplexed and trend plots should be prepared. See docstring
        of __init__ for details on plotting keywords.

        Args:
            **kwargs: forwarded to the super

        Returns:

        """
        if not self.get_param_value("plot", True):
            return  # no plotting if "plot" is False

        # prepares the processed data before plotting
        n_sweep_points = self.proc_data_dict['n_dim_2_sweep_points']

        # set which plots to plot depending on whether it was a sweep and/or
        # multiplexed readout
        was_sweep = self.get_param_value('TwoD', False) and n_sweep_points > 1
        was_mux = self.get_param_value('multiplexed_ssro', False)

        plot_single_qb = self.get_param_value('plot_single_qb_plots',
                                              not was_sweep)
        plot_mux = self.get_param_value('plot_mux_plots',
                                          was_mux and not was_sweep)

        if plot_single_qb:
            for qbn in self.qb_names:
                # iterates over slices if 2nd dimension was swept
                if was_sweep:
                    sp_dict = self.proc_data_dict['sweep_points_2D_dict'][qbn]
                    # use only first sweep parameter to indicate slice
                    sp_name, sp_vals = next(iter(sp_dict.items()))
                    _, sp_unit, sp_hrs = self.sp.get_sweep_params_description(sp_name)
                    for i in range(n_sweep_points):
                        slice_title = f"\nSweep Point {i}: {sp_hrs} = {sp_vals[i]}{sp_unit}"
                        self.plot_single_qb_plots(qbn=qbn,
                                                  slice_title=slice_title,
                                                  sweep_indx=i, **kwargs)
                else:
                    self.plot_single_qb_plots(qbn=qbn, **kwargs)

        # multiplexed ssro plots
        if plot_mux and was_mux:
            # iterates over slices if 2nd dimension was swept
            if was_sweep:
                sp_dict = self.proc_data_dict['sweep_points_2D_dict']
                main_qbn = list(sp_dict.keys())[0]
                main_sp = self.get_param_value('main_sp')
                main_sp_names = [(sp_name, qbn) for qbn in self.qb_names
                                 if main_sp is not None
                                 and (sp_name := main_sp.get(qbn))
                                 and sp_name not in self.sp.get_parameters(dimension=0)]
                if len(sp_dict) > 1 and len(main_sp_names):
                    # use main sp of first qb that has it specified to indicate
                    # slice
                    sp_vals, sp_unit, sp_hrs = \
                        self.sp.get_sweep_params_description(main_sp_names[0][0])
                    main_qbn = main_sp_names[0][1]
                else:
                    # use only first sweep parameter to indicate slice
                    sp_name, sp_vals = next(iter(sp_dict[main_qbn].items()))
                    _, sp_unit, sp_hrs = self.sp.get_sweep_params_description(
                        sp_name)
                for i in range(n_sweep_points):
                    slice_title = f"\nSweep Point {i}: {main_qbn} {sp_hrs} " \
                                  f"= {sp_vals[i]}{sp_unit}"
                    self.plot_multiplexed_plots(slice_title=slice_title,
                                                sweep_indx=i,
                                                **kwargs)
            else:
                self.plot_multiplexed_plots(**kwargs)

        # plots fidelity trend plot
        super().plot(**kwargs)

    def plot_single_qb_plots(self, qbn, slice_title=None, sweep_indx=0, **kw):
        """
        Plots IQ plane scatter plots and state assignment probability matrices
        for a given qbn and sweep index (2nd dimension).

        Plots all the data in ``self.proc_data_dict['analysis_params']`` which
        keys start with 'data' as scatter plots. Plots the keys
        ``state_prob_mtx`` and ``state_prob_mtx_masked`` as fidelity matrices.

        Args:
            qbn (str): qubit name
            slice_title: additional info if sweep in 2nd dimension was
                performed
            sweep_indx: sweep point (in the 2nd dim) to be plotted
            **kw: not used

        Returns:
            ``None``
        """
        clf_ = self.clf_
        pdd = self.proc_data_dict
        pdd_ap = pdd['analysis_params']

        data_keys = [k for k in list(pdd.keys()) if k.startswith("data")]
        data_dict = {dk: pdd[dk] for dk in data_keys}

        cmap = plt.get_cmap('tab10')
        show = self.options_dict.get("show", False)

        n_qb_states = len(np.unique(self.cp.get_states(qbn)[qbn]))
        tab_x = a_tools.truncate_colormap(cmap, 0,
                                          n_qb_states/10)

        kwargs = {
            "states": list(pdd_ap['means'][qbn][sweep_indx].keys()),
            "xlabel": "Integration Unit 1, $u_1$",
            "ylabel": "Integration Unit 2, $u_2$",
            "scale": self.options_dict.get("hist_scale", "log"),
            "cmap":tab_x}

        for dk, data in data_dict.items():
            if qbn not in data['X']: continue
            if data['X'][qbn][sweep_indx] is None: continue

            title = f"{self.raw_data_dict['timestamp']} {qbn} {dk}\n " \
                    f"{self.classif_method} classifier" \
                    f"{slice_title if slice_title is not None else ''}"

            kwargs.update(dict(title=title))

            # plot data and histograms
            n_shots_to_plot = self.get_param_value('n_shots_to_plot', None)
            if n_shots_to_plot is not None:
                n_shots_to_plot *= n_qb_states
            if data['X'][qbn][sweep_indx].shape[1] == 1:
                if self.classif_method == "gmm":
                    kwargs['means'] = self._get_means(clf_[qbn][sweep_indx])
                    kwargs['std'] = np.sqrt(self._get_covariances(clf_[qbn][sweep_indx]))
                else:
                    # no Gaussian distribution can be plotted
                    kwargs['plot_fitting'] = False
                kwargs['colors'] = cmap(np.unique(data['prep_states'][qbn][sweep_indx]))
                fig, main_ax = self.plot_1D_hist(data['X'][qbn][sweep_indx][:n_shots_to_plot],
                                        data["prep_states"][qbn][sweep_indx][:n_shots_to_plot],
                                        **kwargs)
            else:
                fig, axes = self.plot_scatter_and_marginal_hist(
                    data['X'][qbn][sweep_indx][:n_shots_to_plot],
                    data["prep_states"][qbn][sweep_indx][:n_shots_to_plot],
                    **kwargs)

                # FIXME HACK
                # With Matplotlib 3.8.3, this plot ends up with an extra
                # blank axis as the first one which breaks the logic below
                # I did not hunt through the mess to find the root cause
                # of the change; instead, the lines of code below check if
                # this first blank axis was created, and, if so, deletes it
                if fig.get_axes()[0].get_xlabel() == "":
                    fig.delaxes(fig.get_axes()[0])

                # plot clf_boundaries
                main_ax = fig.get_axes()[0]
                self.plot_clf_boundaries(data['X'][qbn][sweep_indx],
                                         clf_[qbn][sweep_indx],
                                         ax=main_ax,
                                         cmap=tab_x)
                # plot means and std dev
                data_means = pdd_ap['means'][qbn][sweep_indx]
                try:
                    clf_means = self._get_means(clf_[qbn][sweep_indx])
                except Exception as e: # not a gmm model--> no clf_means.
                    clf_means = []
                try:
                    covs = self._get_covariances(clf_[qbn][sweep_indx])
                except Exception as e: # not a gmm model--> no cov.
                    covs = []

                for i, data_mean in enumerate(data_means.values()):
                    main_ax.scatter(data_mean[0], data_mean[1], color='w', s=80)
                    if len(clf_means):
                        main_ax.scatter(clf_means[i][0], clf_means[i][1],
                                        color='k', s=80)
                    if len(covs) != 0:
                        self.plot_std(clf_means[i] if len(clf_means)
                                      else data_mean,
                                      covs[i],
                                      n_std=1, ax=main_ax,
                                      edgecolor='k', linestyle='--',
                                      linewidth=1)

            # plot thresholds and mapping
            plt_fn = {0: main_ax.axvline, 1: main_ax.axhline}
            thresholds = pdd_ap[
                'classifier_params'][qbn][sweep_indx].get("thresholds", dict())
            mapping = pdd_ap[
                'classifier_params'][qbn][sweep_indx].get("mapping", dict())
            for k, thres in thresholds.items():
                plt_fn[k](thres, linewidth=2,
                          label="threshold i.u. {}: {:.5f}".format(k, thres),
                          color='k', linestyle="--")
                main_ax.legend(loc=[0.2,-0.62])

            ax_frac = {0: (0.07, 0.1), # locations for codewords
                       1: (0.83, 0.1),
                       2: (0.07, 0.9),
                       3: (0.83, 0.9)}
            for cw, state in mapping.items():
                main_ax.annotate("0b{:02b}".format(cw) + f":{state}",
                                 ax_frac[cw], xycoords='axes fraction')
            fig_key = f'{qbn}_{self.classif_method}_classifier_{dk}' \
                      f'{f"_sp_{sweep_indx}" if slice_title is not None else ""}'
            self.figs[fig_key] = fig
        if show:
            plt.show()

        # state assignment prob matrix
        title = f"{self.raw_data_dict['timestamp']} \n" \
                f"{self.classif_method} State Assignment Probability Matrix {qbn}\n" \
                f"Total # shots:{pdd_ap['n_shots']}"\
                f"{slice_title if slice_title is not None else ''}"

        fig = self.plot_fidelity_matrix(
            pdd_ap['state_prob_mtx'][qbn][sweep_indx],
            [rf'$\vert {"".join(state)} \rangle$'
             for state in self._order_state_labels(kwargs['states'])],
            title=title,
            show=show,
            auto_shot_info=False)
        fig_key = f'{qbn}_state_prob_matrix_{self.classif_method}'\
                  f'{f"_sp_{sweep_indx}" if slice_title is not None else ""}'
        self.figs[fig_key] = fig

        if self.preselection and \
                pdd_ap['state_prob_mtx_masked'][qbn][sweep_indx] is not None and \
                len(pdd_ap['state_prob_mtx_masked'][qbn][sweep_indx]) != 0:
            title = f"{self.raw_data_dict['timestamp']}\n"\
                f"{self.classif_method} State Assignment Probability " \
                    f"Matrix Masked {qbn}\n"\
                f"Total # shots:{pdd_ap['n_shots_masked'][qbn][sweep_indx]}"\
                f"{slice_title if slice_title is not None else ''}"
            presel_col = pdd_ap['presel_fraction_per_state'][qbn][sweep_indx] \
                if self.get_param_value('plot_init_columns', False) else None

            fig = self.plot_fidelity_matrix(
                pdd_ap['state_prob_mtx_masked'][qbn][sweep_indx],
                [rf'$\vert {"".join(state)} \rangle$'
                 for state in self._order_state_labels(kwargs['states'])],
                title=title, show=show, auto_shot_info=False,
                presel_column=presel_col)
            fig_key = f'{qbn}_state_prob_matrix_masked_{self.classif_method}'\
                f'{f"_sp_{sweep_indx}" if slice_title is not None else ""}'
            self.figs[fig_key] = fig

    def plot_multiplexed_plots(self, slice_title=None, sweep_indx=0, **kw):
        """
        Plots the state assignment probability matrices for a given qbn and
        sweep index (2nd dimension).

        Plots the keys ``mux_state_prob_mtx`` and ``mux_state_prob_mtx_masked``
        as fidelity matrices.

        Args:
            slice_title (str): additional info if sweep in 2nd dimension was
                performed
            sweep_indx (int): sweep point (in the 2nd dim) to be plotted
            **kw: not used

        Returns:

        """
        cmap = roa.MultiQubit_SingleShot_Analysis.get_highcontrast_colormap()
        show = self.options_dict.get("show", False)
        plot_norm = plt_cols.Normalize(vmin=0., vmax=1.)
        plot_compact = len(np.unique(self.cp.states, axis=0)) > 6

        pdd = self.proc_data_dict
        pdd_ap = pdd['analysis_params']

        target_names = [rf'$\vert {"".join(state)} \rangle$'
                        for state in pdd['mux_data']['unique_states']]

        title = f"{self.raw_data_dict['timestamp']} \n" \
                f"{self.classif_method} Multiplexed State Assignment " \
                f"Probability Matrix\n" \
                rf"States $\vert${', '.join(self.cp.qb_names)}$\rangle$, " \
                f"Total # shots:{pdd_ap['mux_n_shots'][sweep_indx]}" \
                f"{slice_title if slice_title is not None else ''}"

        fig = self.plot_fidelity_matrix(
            pdd_ap['mux_state_prob_mtx'][sweep_indx],
            target_names=target_names, title=title, show=show, cmap=cmap,
            auto_shot_info=False, plot_norm=plot_norm, plot_compact=plot_compact)
        fig_key = f'mux_state_prob_matrix_{self.classif_method}' \
                  f'{f"_sp_{sweep_indx}" if slice_title is not None else ""}'
        fig.set_size_inches((max(3 + 0.4*len(target_names), 10), ) * 2)
        self.figs[fig_key] = fig

        if self.preselection and \
                pdd_ap['mux_state_prob_mtx_masked'][sweep_indx] is not None and \
                len(pdd_ap['mux_state_prob_mtx_masked'][sweep_indx]) != 0:
            title = f"{self.raw_data_dict['timestamp']}\n" \
                    f"{self.classif_method} Multiplexed State Assignment " \
                    f"Probability Matrix Masked \n" \
                    rf"States $\vert${', '.join(self.cp.qb_names)}$\rangle$, " \
                    f"Total # shots:{pdd_ap['mux_n_shots_masked'][sweep_indx]} " \
                    f"out of {pdd_ap['mux_n_shots'][sweep_indx]}" \
                    f"{slice_title if slice_title is not None else ''}"
            presel_col = pdd_ap['mux_presel_fraction_per_state'][sweep_indx] \
                if self.get_param_value('plot_init_columns', True) else None
            fig = self.plot_fidelity_matrix(
                pdd_ap['mux_state_prob_mtx_masked'][sweep_indx],
                target_names=target_names, title=title, show=show, cmap=cmap,
                auto_shot_info=False, plot_norm=plot_norm,
                plot_compact=plot_compact, presel_column=presel_col)
            fig_key = f'mux_state_prob_matrix_masked_{self.classif_method}' \
                      f'{f"_sp_{sweep_indx}" if slice_title is not None else ""}'
            fig.set_size_inches((max(3 + 0.4*len(target_names), 10), ) * 2)
            self.figs[fig_key] = fig

    def prepare_plots(self):
        """
        Prepares sweep plots: Prepare fidelity (linear) and infidelity (log)
        plots as well as parse custom plotting metrics given in ``plot_metrics``
        or ``multiplexed_plot_metrics``. See __init__ docstring for details
        on custom plotting metrics.
        """
        # don't prepare sweep plots if there was no sweep
        if self.proc_data_dict['n_dim_2_sweep_points'] == 1: return
        was_mux = self.get_param_value('multiplexed_ssro', False)

        plot_sweep = self.get_param_value('plot_sweep_plots', not was_mux)
        plot_mux_sweep = self.get_param_value(
            'plot_mux_sweep_plots', was_mux) and was_mux

        for should_plot, multiplexed in zip([plot_sweep, plot_mux_sweep],
                                            [False, True]):
            if not should_plot:
                continue

            plot_metrics = self.get_param_value(
                'multiplexed_plot_metrics' if multiplexed
                else 'plot_metrics', [])

            # adding 'fidelity' and 'infidelity' plots (default)
            if isinstance(plot_metrics, (dict, str)):
                plot_metrics = [plot_metrics]
            plot_metrics.append({
                'metric': 'lambda fm: 100 * np.trace(fm) / float(np.sum(fm))',
                'plot_name': 'fidelity',
                'yunit': '%',
                'yscale': 'linear',
                'ymax': 100
            })
            plot_metrics.append({
                'metric': 'lambda fm: 1 - np.trace(fm) / float(np.sum(fm))',
                'plot_name': 'infidelity',
                'yunit': '',
                'yscale': 'log'
            })
            if multiplexed:
                for pm in plot_metrics:
                    # here the qbn is used to identify the plot in
                    # plot_dicts, and to specify from which qb the best pulse
                    # parameters are listed below a multiplexed sweep plot
                    sp_dict = self.proc_data_dict['sweep_points_2D_dict']
                    main_qbn = list(sp_dict.keys())[0]
                    main_sp = self.get_param_value('main_sp')
                    main_sp_qbns = [qbn for qbn in self.qb_names
                                    if main_sp is not None
                                    and (sp_name := main_sp.get(qbn))
                                    and sp_name not in self.sp.get_parameters(
                                    dimension=0)]
                    main_qbn = main_sp_qbns[0] if len(main_sp_qbns) \
                        else main_qbn

                    self.prepare_sweep_plot(main_qbn, pm, multiplexed=True)
            else:
                for qbn in self.qb_names:
                    for pm in plot_metrics:
                        self.prepare_sweep_plot(qbn, pm)

    def prepare_sweep_plot(self, qbn, plot_settings, multiplexed=False):
        """
        Helper function for ``prepare_plots``: Prepares (single qb) trend plots
        and custom (single qb) trend plots when sweeping in 2nd dimension.

        Args:
            qbn (str): Name of qb
            plot_settings (dict): Same as 'metrics' kwarg in __init__ docstring
                without the outer list.
            multiplexed (bool): Whether the plot shows multiplexed data

        Returns:

        """
        metric = plot_settings.get('metric', None)
        try:
            metric = eval(metric) if isinstance(metric, str) else metric
        except SyntaxError:
            log.warning('Could not parse the custom plot metric into a '
                        'function. Either pass a valid lambda function '
                        'directly or as a string')
        if not callable(metric):
            log.warning('Every metric must contain a function taking '
                        'the state probability matrix as an argument and '
                        'returning a single value, e.g. the fidelity.')

        plot_name = plot_settings.get('plot_name', '')
        if multiplexed:
            yvals = np.array([metric(fm) for fm in self.proc_data_dict[
                'analysis_params']['mux_state_prob_mtx']])
            raw_fig_key = f'multiplexed_ssro_{plot_name}'
        else:
            yvals = np.array([metric(fm) for fm in self.proc_data_dict[
                'analysis_params']['state_prob_mtx'][qbn]])
            raw_fig_key = f'ssro_{plot_name}_{qbn}'

        ymin = np.nanmin(yvals)
        ymax = plot_settings.get('ymax', np.nanmax(yvals))

        base_plot_dict, vline_key = \
            self.get_base_sweep_plot_options(qbn=qbn, plot_name=plot_name,
                                             multiplexed=multiplexed)

        plot_label = plot_settings.get('setlabel',
                        plot_settings.get('ylabel', plot_name)).capitalize()

        self.plot_dicts[raw_fig_key] = base_plot_dict
        self.plot_dicts[raw_fig_key].update({
            'yvals': yvals,
            'ylabel': plot_settings.get('ylabel', plot_name).capitalize(),
            'setlabel': f'{plot_label} (raw)',
            'yunit': plot_settings.get('yunit', ''),
            'yscale': plot_settings.get('yscale', 'linear'),
            'color': 'C1',
            'linestyle': '--',
        })

        if self.preselection:
            if multiplexed:
                yvals_masked = np.array([metric(fm) for fm in self.proc_data_dict[
                    'analysis_params']['mux_state_prob_mtx_masked']])
                masked_fig_key = f'multiplexed_ssro_{plot_name}_masked'
            else:
                yvals_masked = np.array([metric(fm) for fm in self.proc_data_dict[
                    'analysis_params']['state_prob_mtx_masked'][qbn]])
                masked_fig_key = f'ssro_{plot_name}_masked_{qbn}'

            self.plot_dicts[masked_fig_key] = {}
            self.plot_dicts[masked_fig_key].update(
                self.plot_dicts[raw_fig_key])

            self.plot_dicts[masked_fig_key].update({
                'yvals':  yvals_masked,
                'setlabel': f'{plot_label} (preselected)',
                'color': 'C0',
                'linestyle': '-',
            })

            ymin = min(ymin, np.nanmin(yvals_masked))
            ymax = max(ymax, np.nanmax(yvals_masked))

        self.plot_dicts[vline_key].update({
            'ymin': ymin,
            'ymax': ymax, })

    def get_base_sweep_plot_options(self, qbn, plot_name, multiplexed=False):
        """
        Helper function for prepare_sweep_plot. Prepares default trend plots.

        Args:
            qbn (str): Name of qb
            plot_name (str): Name of plot
            multiplexed (bool): Whether the plot shows multiplexed data

        Returns:

        """
        pdd = self.proc_data_dict
        sp_dict = pdd['sweep_points_2D_dict'][qbn]

        main_sp = self.get_param_value('main_sp')
        if len(sp_dict) > 1 and main_sp is not None and \
                (sp_name := main_sp.get(qbn)) and \
                sp_name not in self.sp.get_parameters(dimension=0):
            sp_vals, sp_unit, sp_hrs = self.sp.get_sweep_params_description(
                sp_name)
        elif len(sp_dict) != 1:
            # fall back to sweep point indx if this qb is not swept or if there
            # are more than parameters that are swept
            sp_unit = None
            sp_vals = np.arange(self.proc_data_dict['n_dim_2_sweep_points'])
            sp_hrs = 'Sweep point index'
        else:
            sp_name, sp_vals = next(iter(sp_dict.items()))
            _, sp_unit, sp_hrs = self.sp.get_sweep_params_description(sp_name)

        if multiplexed:
            title = f"{self.raw_data_dict['timestamp']} multiplexed sweep " \
                    f"{plot_name}\n {self.classif_method} classifier"
            fig_id = f'multiplexed_ssro_{plot_name}'

            best_fidelity = pdd['mux_best_fidelity']['fidelity']
            best_slice = pdd['mux_best_fidelity']['sweep_index']
            vline_key = f'best_slice_vline_{plot_name}_multiplexed'
            text_msg_key = f'text_msg_{plot_name}_multiplexed'
        else:
            title = f"{self.raw_data_dict['timestamp']} {qbn} sweep " \
                    f"{plot_name}\n {self.classif_method} classifier"
            fig_id = f'ssro_{plot_name}_{qbn}'

            best_fidelity = pdd['best_fidelity'][qbn]['fidelity']
            best_slice = pdd['best_fidelity'][qbn]['sweep_index']
            vline_key = f'best_slice_vline_{plot_name}_{qbn}'
            text_msg_key = f'text_msg_{plot_name}_{qbn}'

        fid_plot_options = {
            'plotfn': self.plot_line,
            'title': title,
            'fig_id': fig_id,
            'xvals': sp_vals,
            'xlabel': sp_hrs,
            'xunit': sp_unit,
            'legend_ncol': 1,
            'do_legend': True,
            'legend_bbox_to_anchor': (1, -0.15),
            'legend_pos': 'upper right'
        }
        
        textstr = f'best fidelity: {best_fidelity * 100:.2f}%'

        self.plot_dicts[vline_key] = {
            'fig_id': fig_id,
            'plotfn': self.plot_vlines,
            'x': sp_vals[best_slice],
            'linestyle': '--',
            'colors': 'gray'}

        for sp_name, sp_vals in sp_dict.items():
            _, sp_unit, sp_hrs = self.sp.get_sweep_params_description(sp_name)
            textstr += f'\n{qbn}: {sp_hrs} = {sp_vals[best_slice]} ' \
                       f'{sp_unit if sp_unit is not None else ""}'

        self.plot_dicts[text_msg_key] = {
            'fig_id': fig_id,
            'ypos': -0.2,
            'xpos': -0.025,
            'horizontalalignment': 'left',
            'verticalalignment': 'top',
            'plotfn': self.plot_text,
            'text_string': textstr}

        return fid_plot_options, vline_key


class MultiQutritActiveResetAnalysis(MultiQubit_TimeDomain_Analysis):
    """
    Analyzes the performance of (two- or three-level) active reset
    (Measured via pycqed.measurement.calibration.single_qubit_gates.ActiveReset,
    see the corresponding doc string for details about the sequence).

    Extracts the reset rate (how fast is the reset) and the residual excited
    state population.

    Helps to choose the number of reset repetitions for experiments making use
    of active reset, by considering the tradeoff between the time required
    for reset and the residual excited state population.

    """

    def extract_data(self):
        super().extract_data()
        params_dict = {
            'pulse_period': 'Instrument settings.TriggerDevice.pulse_period',
        }
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

        if self.qb_names is None:
            # try to get qb_names from cal_points
            try:
                cp = CalibrationPoints.from_string(
                    self.get_param_value('cal_points', None))
                self.qb_names = deepcopy(cp.qb_names)
            except:
                # try to get them from metadata
                self.qb_names = self.get_param_value('ro_qubits', None)
                if self.qb_names is None:
                    raise ValueError('Could not find qb_names. Please'
                                     'provide qb_names to the analysis'
                                     'or ensure they are in calibration points'
                                     'or the metadata under "qb_names" '
                                     'or "ro_qubits"')

    def process_data(self):
        super().process_data()

        # reshape data per prepared state before reset for each pg, pe, (pf),
        # for the projected data dict and possibly the readout-corrected version
        pdd = 'projected_data_dict'
        # self.proc_data_dict[pdd]["qb10"]["pe"] = self.proc_data_dict[pdd]["qb10"]["pe"].T
        # self.proc_data_dict[pdd]["qb10"]["pg"] = (1 - self.proc_data_dict[pdd]["qb10"]["pe"])
        for suffix in ["", "_corrected"]:
            projdd_per_prep_state = \
                deepcopy(self.proc_data_dict.get(pdd + suffix, {}))
            for qbn, data_qbi in \
                    self.proc_data_dict.get(pdd + suffix, {}).items():
                prep_states = self.sp.get_values("initialize")
                for j, (state, data) in enumerate(data_qbi.items()):
                    n_ro = data.shape[0] # infer number of readouts per sequence
                    projdd_per_prep_state[qbn][state] = dict()
                    for i, prep_state in enumerate(prep_states):
                        projdd_per_prep_state[qbn][state].update(
                            {f"prep_{prep_state}":
                                 data[i*n_ro//len(prep_states):
                                      (i+1)*n_ro//len(prep_states),
                                 :]})

            if len(projdd_per_prep_state):
                self.proc_data_dict[pdd + '_per_prep_state' + suffix] = \
                    projdd_per_prep_state

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        if "ro_separation" in self.get_param_value("preparation_params"):
            ro_sep = \
                self.get_param_value("preparation_params")["ro_separation"]
        else:
            return

        base_data_key = 'projected_data_dict_per_prep_state'
        data_keys = [base_data_key]
        if self.proc_data_dict.get(base_data_key + '_corrected', False):
            data_keys += [base_data_key + '_corrected']

        for dk, suffix in zip(data_keys, ('', '_corrected')):
            for qbn in self.qb_names:
                probs = self.proc_data_dict[dk][qbn]
                for prep_state, g_pop in probs.get('pg', {}).items():
                    if "g" in prep_state:
                        continue # no need to fit reset on ground state
                    for seq_nr, g_pop_per_seq in enumerate(g_pop.T):
                        excited_pop = 1 - g_pop_per_seq
                        # excited_pop = np.exp(-np.arange(len(g_pop_per_seq)))
                        if self.num_cal_points != 0:
                            # do not fit data with cal points
                            excited_pop = excited_pop[:-self.num_cal_points]

                        if len(excited_pop) < 3:
                            log.warning('Not enough reset pulses to fit a reset '
                                        'rate, increase the number of reset '
                                        'pulses to 3 or more ')
                            continue
                        time = np.arange(len(excited_pop)) * ro_sep
                        # linear rate approx
                        rate_guess = (excited_pop[0] - excited_pop[-1]) / time[-1]
                        decay = lambda time, a, rate, offset: \
                            a * np.exp(-2 * np.pi * rate * time) + offset
                        decay_model =  lmfit.Model(decay)
                        decay_model.set_param_hint('a', value=excited_pop[0])
                        decay_model.set_param_hint('rate', value=rate_guess)

                        decay_model.set_param_hint('n', value=1, vary=False)
                        decay_model.set_param_hint('offset', value=0)

                        params = decay_model.make_params()

                        key = f'fit_rate_{qbn}_{prep_state}_seq_{seq_nr}{suffix}'
                        self.fit_dicts[key] = {
                            'fit_fn': decay_model.func,
                            'fit_xvals': {'time': time},
                            'fit_yvals': {'data': excited_pop},
                            'guess_pars': params}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        apd = self.proc_data_dict['analysis_params_dict']

        base_data_key = 'projected_data_dict_per_prep_state'
        data_keys = [base_data_key]
        if self.proc_data_dict.get(base_data_key + '_corrected', False):
            data_keys += [base_data_key + '_corrected']

        for dk, suffix in zip(data_keys, ('', '_corrected')):
            for qbn in self.qb_names:
                probs = self.proc_data_dict[dk][qbn]
                for prep_state, g_pop in probs.get('pg', {}).items():
                    if "g" in prep_state:
                        continue # no fit for reset on ground state
                    for seq_nr in range(len((g_pop.T))):
                        key = f'fit_rate_{qbn}_{prep_state}_seq_{seq_nr}{suffix}'
                        for param, param_key in zip(('rate', 'offset'),
                                                    ("reset_rate",
                                                     "residual_population")):
                            pk = param_key + suffix
                            res = self.fit_res[key]
                            param_val = res.params[param].value
                            param_stderr = res.params[param].stderr
                            if not pk in apd:
                                apd[pk] = defaultdict(dict)
                            if not prep_state in apd[pk][qbn]:
                                apd[pk][qbn][prep_state] = \
                                    defaultdict(dict)

                            apd[pk][qbn][prep_state]["val"] = \
                                apd[pk][qbn][prep_state].get("val", []) + \
                                [param_val]
                            apd[pk][qbn][prep_state]["stderr"] = \
                                apd[pk][qbn][prep_state].get("stderr", []) + \
                                [param_stderr]
        self.save_processed_data(key="analysis_params_dict")

    def prepare_plots(self):
        # prepare raw population plots
        legend_bbox_to_anchor = (1, -0.20)
        legend_pos = 'upper right'
        legend_ncol = 2 #len(self.sp.get_values("initialize"))
        # overwrite baseAnalysis plots
        self.plot_dicts = OrderedDict()
        basekey = 'projected_data_dict_per_prep_state'
        suffixes = ('', '_corrected')
        keys = {basekey + suffix: suffix for suffix in suffixes
                 if basekey + suffix in self.proc_data_dict}
        for k in keys:
            for qbn, data_qbi in self.proc_data_dict[k].items():
                for i, (state, data) in enumerate(data_qbi.items()):
                    for j, (prep_state, data_prep_state) in \
                            enumerate(data.items()):
                        for seq_nr, pop in enumerate(data_prep_state.T):
                            plt_key = 'data_{}_{}_{}_{}_{}'.format(
                                 k, qbn, state, prep_state, seq_nr)
                            fig_key = f"populations_{qbn}_{prep_state}{keys[k]}"
                            self.plot_dicts[plt_key] = {
                                'plotfn': self.plot_line,
                                'fig_id': fig_key,
                                'xvals': np.arange(len(pop)),
                                'xlabel': "Reset cycle, $n$",
                                'xunit': "",
                                'yvals': pop,
                                'yerr': self._std_error(
                                    pop, self.get_param_value('n_shots')),
                                'ylabel': 'Population, $P$',
                                'yunit': '',
                                'yscale': self.get_param_value("yscale", "log"),
                                'yrange': self.get_param_value("yrange", None),
                                'grid': True,
                                'setlabel': self._get_pop_label(state, k,
                                                                not self._has_reset_pulses(seq_nr),
                                                                ),
                                'title': self.raw_data_dict['timestamp'] + ' ' +
                                         self.raw_data_dict['measurementstring']
                                         + " " + prep_state,
                                'titlepad': 0.25,
                                'linestyle': '-',
                                'color': f'C{i}',
                                'alpha': 0.5 if seq_nr == 0 else 1,
                                'do_legend': True,
                                'legend_ncol': legend_ncol,
                                'legend_bbox_to_anchor': legend_bbox_to_anchor,
                                'legend_pos': legend_pos,
                            }

                            # add feedback params info to plot
                            textstr = self._get_feedback_params_text_str(qbn)
                            self.plot_dicts[f'text_msg_{qbn}_' \
                                            f'{prep_state}{keys[k]}'] = {
                                'fig_id': f"populations_{qbn}_{prep_state}{keys[k]}",
                                'ypos': -0.32,
                                'xpos': 0,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'plotfn': self.plot_text,
                                'text_string': textstr}

                            # add thermal population line (for each plot j)
                            if i == 0:
                                g_state_prep_g = \
                                    data_qbi.get("pg", {}).get('prep_g', None)
                                # taking first ro of first sequence as estimate
                                # for thermal population
                                if g_state_prep_g is not None and seq_nr == 0:
                                    p_therm = 1 - g_state_prep_g.flatten()[0]
                                    self.plot_dicts[plt_key + "_thermal"] = {
                                        'plotfn': self.plot_line,
                                        'fig_id': fig_key,
                                        'xvals': np.arange(len(pop)),
                                        'yvals': p_therm * np.ones_like(pop),
                                        'setlabel': "$P_\\mathrm{therm}$",
                                        'linestyle': '--',
                                        'marker': "",
                                        'color': 'k',
                                        'do_legend': True,
                                        'legend_ncol': legend_ncol,
                                        'legend_bbox_to_anchor':
                                            legend_bbox_to_anchor,
                                        'legend_pos': legend_pos,
                                    }

                            # plot fit results
                            fit_key = \
                                f'fit_rate_{qbn}_{prep_state}_seq_{seq_nr}{keys[k]}'
                            if fit_key in self.fit_res and \
                                    not fit_key in self.plot_dicts:
                                res = self.fit_res[fit_key]
                                rate = res.best_values['rate'] * 1e-6
                                residual_pop = res.best_values['offset']
                                superscript = "{NR}" if seq_nr == 0 \
                                    else f"{{c {seq_nr}}}" if "corrected" \
                                                in fit_key else f"{{{seq_nr}}}"
                                label = f'fit: $\Gamma_{prep_state[-1]}^{superscript}' \
                                        f' = {rate:.3f}$ MHz'
                                if seq_nr != 0:
                                    # add residual population if not no reset
                                    label += f", $P_\mathrm{{exc}}^\mathrm{{res}}$" \
                                             f" = {residual_pop*100:.2f} %"
                                self.plot_dicts[fit_key] = {
                                    'plotfn': self.plot_fit,
                                    'fig_id':
                                        f"rate_{qbn}{keys[k]}",
                                    'xvals': res.userkws['time'],
                                    'xlabel': "Reset cycle, $n$",
                                    'fit_res': res,
                                    'xunit': "s",
                                    'ylabel': 'Population, $P$',
                                    'yscale': self.get_param_value("yscale", "log"),
                                    'setlabel': label,
                                    'title': self.raw_data_dict['timestamp'] + ' ' +
                                             f"Reset rates {qbn}{keys[k]}",
                                    'color': f'C{j}',
                                    'alpha': 1 if self._has_reset_pulses(seq_nr) else 0.5,
                                    'do_legend': seq_nr in [0, 1],
                                    'legend_ncol': legend_ncol,
                                    'legend_bbox_to_anchor': legend_bbox_to_anchor,
                                    'legend_pos': legend_pos,
                                }

                                self.plot_dicts[fit_key + 'data'] = {
                                    'plotfn': self.plot_line,
                                    'fig_id':
                                        f"rate_{qbn}{keys[k]}",
                                    'xvals': res.userkws['time'],
                                    'xlabel': "Time, $t$",
                                    'xunit': "s",
                                    'yvals': res.data,
                                    'yerr': self._std_error(
                                        res.data, self.get_param_value('n_shots')),
                                    'ylabel': 'Excited Pop., $P_\mathrm{exc}$',
                                    'yunit': '',
                                    'setlabel':
                                        "data" if
                                        self._has_reset_pulses(seq_nr)
                                        else "data NR",
                                    'linestyle': 'none',
                                    'color': f'C{j}',
                                    'alpha': 1 if self._has_reset_pulses(seq_nr) else 0.5,
                                    "do_legend": True,
                                    'legend_ncol': legend_ncol,
                                    'legend_bbox_to_anchor': legend_bbox_to_anchor,
                                    'legend_pos': legend_pos,
                                    }

    def _has_reset_pulses(self, seq_nr):
        return not self.sp.get_values('pulse_off')[seq_nr]


    def plot(self, **kw):
        super().plot(**kw)

        # add second axis to population figures
        from matplotlib.ticker import MaxNLocator
        for axname, ax in self.axs.items():
            if "populations" in axname:
                if "ro_separation" in self.get_param_value("preparation_params"):
                    ro_sep = \
                        self.get_param_value("preparation_params")["ro_separation"]
                    timeax = ax.twiny()
                    timeax.set_xlabel(r"Time ($\mu s$)")
                    timeax.set_xlim(0, ax.get_xlim()[1] * ro_sep * 1e6)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # plot raw readouts
        if self.get_param_value('plot_raw_shots'):
            prep_states = self.sp.get_values("initialize")
            n_seqs = self.sp.length(1)
            for qbn, shots in self.proc_data_dict['single_shots_per_qb'].items():
                # shots are organized as follow, from outer to inner loop:
                # shot_prep-state_reset-ro_seq-nr
                n_ro = len(shots) // self.get_param_value("n_shots")
                n_ro_per_prep_state = n_ro // (n_seqs * len(prep_states))
                for i, prep_state in enumerate(prep_states):
                    for j in range(n_ro_per_prep_state):
                        for seq_nr in range(n_seqs):
                            ro = i * n_ro_per_prep_state * len(prep_states) \
                                 + j * len(prep_states) + seq_nr
                            shots_single_ro = shots[ro::n_ro]
                            # first sequence is "no reset"
                            seq_label = 'NR' if seq_nr == 0 else seq_nr
                            fig_key = \
                                f"histograms_seq_{seq_label}_reset_cycle_{j}"
                            if fig_key not in self.figs:
                                self.figs[fig_key], _ = plt.subplots()

                            if shots.shape[1] == 2:
                                plot_func = \
                                    MultiQutrit_Singleshot_Readout_Analysis.\
                                        plot_scatter_and_marginal_hist
                                kwargs = dict(create_axes=not bool(i))
                            elif shots.shape[1] == 1:
                                plot_func = \
                                    MultiQutrit_Singleshot_Readout_Analysis.\
                                        plot_1D_hist
                                kwargs = {}
                            else:
                                raise NotImplementedError(
                                    "Raw shot plotting not implemented for"
                                    f" {shots.shape[1]} dimensions")
                            colors = [f'C{i}']
                            fig, _ = plot_func(shots_single_ro,
                                       y_true=[i]*shots_single_ro.shape[0],
                                        colors=colors,
                                        legend=True,
                                        legend_labels={i: "prep " + prep_state},
                                        fig=self.figs[fig_key], **kwargs)
                            fig.suptitle(f'Reset cycle: {j}')

    def _get_feedback_params_text_str(self, qbn):
        str = "Reset cycle time: "
        ro_sep = self.prep_params.get("ro_separation", None)
        str += f"{1e6 * ro_sep:.2f} $\mu s$" if ro_sep is not None else \
            "Unknown"
        str += "\n"

        str += "RO to feedback time: "
        prow = self.prep_params.get("post_ro_wait", None)
        str += f"{1e6 * prow:.2f} $\mu s$" if ro_sep is not None else "Unknown"
        str += "\n"
        str += "Trigger rate: "
        pp = self.raw_data_dict['pulse_period']
        str += f"{1e6 * pp:.2f} $\mu s$" if pp is not None else "Unknown"
        str += "\n"
        thresholds = self.get_param_value('thresholds', {})
        str += "Threshold(s):\n{}".format(
            "\n".join([f"{i}: {t:0.5f}" for i, t in
                       thresholds.get(qbn, {}).items()]))
        return str

    @staticmethod
    def _get_pop_label(state, key, no_reset=False):
        superscript = "{NR}" if no_reset else "{c}" \
            if "corrected" in key else "{}"
        return f'$P_{state[-1]}^{superscript}$'

    @staticmethod
    def _std_error(p, nshots=10000):
        return np.sqrt(np.abs(p)*(1-np.abs(p))/nshots)


class FluxPulseTimingAnalysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            params_dict[f'fp_length_{qbn}'] = \
                f'Instrument settings.{qbn}.flux_pulse_pulse_length'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

    def process_data(self):
        super().process_data()

        # Make sure data has the right shape (len(hard_sp), len(soft_sp))
        for qbn, data in self.proc_data_dict['data_to_fit'].items():
            if data.shape[1] != self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points'].size:
                self.proc_data_dict['data_to_fit'][qbn] = data.T

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn][0]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            TwoErrorFuncModel = lmfit.Model(fit_mods.TwoErrorFunc)
            guess_pars = fit_mods.TwoErrorFunc_guess(
                model=TwoErrorFuncModel, data=data, delays=sweep_points)
            guess_pars['amp'].vary = True
            guess_pars['mu_A'].vary = True
            guess_pars['mu_B'].vary = True
            guess_pars['sigma'].vary = True
            guess_pars['offset'].vary = True
            key = 'two_error_func_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': TwoErrorFuncModel.func,
                'fit_xvals': {'x': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            mu_A = self.fit_dicts['two_error_func_' + qbn][
                'fit_res'].best_values['mu_A']
            mu_B = self.fit_dicts['two_error_func_' + qbn][
                'fit_res'].best_values['mu_B']
            fp_length = self.raw_data_dict[f'fp_length_{qbn}']

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn]['delay'] = \
                mu_A + 0.5 * (mu_B - mu_A) - fp_length / 2
            try:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'delay_stderr'] = 1 / 2 * np.sqrt(
                        self.fit_dicts['two_error_func_' + qbn][
                            'fit_res'].params['mu_A'].stderr ** 2
                        + self.fit_dicts['two_error_func_' + qbn][
                            'fit_res'].params['mu_B'].stderr ** 2)
            except TypeError:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'delay_stderr'] = 0
            self.proc_data_dict['analysis_params_dict'][qbn][
                'fp_length'] = (mu_B - mu_A)
            try:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'fp_length_stderr'] = np.sqrt(
                        self.fit_dicts['two_error_func_' + qbn][
                            'fit_res'].params['mu_A'].stderr ** 2
                        + self.fit_dicts['two_error_func_' + qbn][
                            'fit_res'].params['mu_B'].stderr ** 2)
            except TypeError:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'fp_length_stderr'] = 0
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                # rename base plot
                base_plot_name = 'Pulse_timing_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn][0],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts[
                        'two_error_func_' + qbn]['fit_res'],
                    'setlabel': 'two error func. fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 1,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                apd = self.proc_data_dict['analysis_params_dict']
                textstr = 'delay = {:.2f} ns'.format(apd[qbn]['delay']*1e9) + \
                          ' $\pm$ {:.2f} ns'.format(apd[qbn]['delay_stderr']
                                                      * 1e9)
                textstr += '\n\nflux_pulse_length:\n  ' \
                           'fitted = {:.2f} ns'.format(
                    apd[qbn]['fp_length'] * 1e9) + \
                           ' $\pm$ {:.2f} ns'.format(
                    apd[qbn]['fp_length_stderr'] * 1e9)
                textstr += '\n  set = {:.2f} ns'.format(
                    1e9 * self.raw_data_dict[f'fp_length_{qbn}'])

                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class FluxPulseTimingBetweenQubitsAnalysis(MultiQubit_TimeDomain_Analysis):

    def process_data(self):
        super().process_data()

        # Make sure data has the right shape (len(hard_sp), len(soft_sp))
        for qbn, data in self.proc_data_dict['data_to_fit'].items():
            if data.shape[1] != self.proc_data_dict['sweep_points_dict'][qbn][
                    'sweep_points'].size:
                self.proc_data_dict['data_to_fit'][qbn] = data.T

        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn][0]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            delays = np.zeros(len(sweep_points) * 2 - 1)
            delays[0::2] = sweep_points
            delays[1::2] = sweep_points[:-1] + np.diff(sweep_points) / 2
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            symmetry_idx, corr_data = find_symmetry_index(data)
            delay = delays[symmetry_idx]

            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn]['delays'] = delays
            self.proc_data_dict['analysis_params_dict'][qbn]['delay'] = delay
            self.proc_data_dict['analysis_params_dict'][qbn][
                'delay_stderr'] = np.diff(delays).mean()
            self.proc_data_dict['analysis_params_dict'][qbn][
                'corr_data'] = np.array(corr_data)
        self.save_processed_data(key='analysis_params_dict')

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn][0]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]

            model = lmfit.Model(lambda t, slope, offset, delay:
                                slope*np.abs((t-delay)) + offset)

            delay_guess = sweep_points[np.argmin(data)]
            offset_guess = np.min(data)
            slope_guess = (data[-1] - offset_guess) / (sweep_points[-1] -
                                                       delay_guess)

            guess_pars = model.make_params(slope=slope_guess,
                                           delay=delay_guess,
                                           offset=offset_guess)

            key = 'delay_fit_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': model.func,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):

        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn]['delay_fit'] = \
                self.fit_dicts['delay_fit_' + qbn]['fit_res'].best_values[
                    'delay']
            try:
                stderr = self.fit_dicts['delay_fit_' + qbn]['fit_res'].params[
                            'delay'].stderr
                stderr = np.nan if stderr is None else stderr
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'delay_fit_stderr'] = stderr

            except TypeError:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'delay_fit_stderr'] \
                    = 0
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        apd = self.proc_data_dict['analysis_params_dict']
        super().prepare_plots()
        rdd = self.raw_data_dict
        for qbn in self.qb_names:
            # rename base plot
            base_plot_name = 'Pulse_timing_' + qbn
            self.prepare_projected_data_plot(
                fig_name=base_plot_name,
                data=self.proc_data_dict['data_to_fit'][qbn][0],
                plot_name_suffix=qbn + 'fit',
                qb_name=qbn)

            if self.do_fitting:
                self.plot_dicts['fit_' + base_plot_name] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_res[ 'delay_fit_' + qbn],
                    'setlabel': 'fit',
                    'color': 'r',
                    'do_legend': True,
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                textstr = 'delay = {:.2f} ns'.format(
                    apd[qbn]['delay_fit'] * 1e9) + ' $\pm$ {:.2f} ns'.format(
                    apd[qbn]['delay_fit_stderr'] * 1e9)
                self.plot_dicts['text_msg_fit' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}
            corr_data = self.proc_data_dict['analysis_params_dict'][qbn][
                'corr_data']
            delays = self.proc_data_dict['analysis_params_dict'][qbn]['delays']

            self.plot_dicts['Autoconvolution_' + qbn] = {
                'title': rdd['measurementstring'] +
                         '\n' + rdd['timestamp'] + '\n' + qbn,
                'fig_name': f'Autoconvolution_{qbn}',
                'fig_id': f'Autoconvolution_{qbn}',
                'plotfn': self.plot_line,
                'xvals': delays[0::2] / 1e-9,
                'yvals': corr_data[0::2],
                'xlabel': r'Delay time',
                'xunit': 'ns',
                'ylabel': 'Autoconvolution function',
                'linestyle': '-',
                'color': 'k',
                'do_legend': False,
                'legend_bbox_to_anchor': (1, 1),
                'legend_pos': 'upper left',
            }

            self.plot_dicts['Autoconvolution2_' + qbn] = {
                'fig_id': f'Autoconvolution_{qbn}',
                'plotfn': self.plot_line,
                'xvals': delays[1::2] / 1e-9,
                'yvals': corr_data[1::2],
                'color': 'r'}

            self.plot_dicts['corr_vline_' + qbn] = {
                'fig_id': f'Autoconvolution_{qbn}',
                'plotfn': self.plot_vlines,
                'x': self.proc_data_dict['analysis_params_dict'][qbn][
                         'delay'] / 1e-9,
                'ymin': corr_data.min(),
                'ymax': corr_data.max(),
                'colors': 'gray'}

            textstr = 'delay = {:.2f} ns'.format(apd[qbn]['delay'] * 1e9) + \
                      ' $\pm$ {:.2f} ns'.format(apd[qbn]['delay_stderr']
                                                  * 1e9)
            self.plot_dicts['text_msg_' + qbn] = {
                'fig_id': f'Autoconvolution_{qbn}',
                'ypos': -0.2,
                'xpos': 0,
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'plotfn': self.plot_text,
                'text_string': textstr}


class FluxPulseScopeAnalysis(MultiQubit_TimeDomain_Analysis):
    """
    Analysis class for a flux pulse scope measurement.
    options_dict parameters specific to this class:
    - freq_ranges_remove/delay_ranges_remove: dict with keys qubit names and
        values list of length-2 lists/tuples that specify frequency/delays
        ranges to completely exclude (from both the fit and the plots)
        Ex: delay_ranges_remove = {'qb1': [ [5e-9, 72e-9] ]}
            delay_ranges_remove = {'qb1': [ [5e-9, 20e-9], [50e-9, 72e-9] ]}
            freq_ranges_remove = {'qb1': [ [5.42e9, 5.5e9] ]}
    - freq_ranges_to_fit/delay_ranges_to_fit: dict with keys qubit names and
        values list of length-2 lists/tuples that specify frequency/delays
        ranges that should be fitted (only these will be fitted!).
        Plots will still show the full data.
        Ex: delays_ranges_to_fit = {'qb1': [ [5e-9, 72e-9] ]}
            delays_ranges_to_fit = {'qb1': [ [5e-9, 20e-9], [50e-9, 72e-9] ]}
            freq_ranges_to_fit = {'qb1': [ [5.42e9, 5.5e9] ]}
    - rectangles_exclude: dict with keys qubit names and
        values list of length-4 lists/tuples that specify delays and frequency
        ranges that should be excluded from  the fit (these will not be
        fitted!). Plots will still show the full data.
        Ex: {'qb1': [ [-10e-9, 5e-9, 5.42e9, 5.5e9], [...] ]}
    - fit_first_cal_state: dict with keys qubit names and values booleans
        specifying whether to fit the delay points corresponding to the first
        cal state (usually g) for that qubit
    - sigma_guess: dict with keys qubit names and values floats specifying the
        fit guess value for the Gaussian sigma
    - sign_of_peaks: dict with keys qubit names and values floats specifying the
        the sign of the peaks used for setting the amplitude guess in the fit
    - from_lower: unclear; should be cleaned up (TODO, Steph 07.10.2020)
    - ghost: unclear; should be cleaned up (TODO, Steph 07.10.2020)
    """
    def extract_data(self):
        self.default_options['rotation_type'] = 'fixed_cal_points'
        super().extract_data()

    def process_data(self):
        super().process_data()

        # dictionaries with keys qubit names and values a list of tuples of
        # 2 numbers specifying ranges to exclude
        freq_ranges_remove = self.get_param_value('freq_ranges_remove')
        delay_ranges_remove = self.get_param_value('delay_ranges_remove')

        self.proc_data_dict['proc_data_to_fit'] = deepcopy(
            self.proc_data_dict['data_to_fit'])
        self.proc_data_dict['proc_sweep_points_2D_dict'] = deepcopy(
            self.proc_data_dict['sweep_points_2D_dict'])
        self.proc_data_dict['proc_sweep_points_dict'] = deepcopy(
            self.proc_data_dict['sweep_points_dict'])
        if freq_ranges_remove is not None:
            for qbn, freq_range_list in freq_ranges_remove.items():
                if freq_range_list is None:
                    continue
                # find name of 1st sweep point in sweep dimension 1
                param_name = [p for p in self.mospm[qbn]
                              if self.sp.find_parameter(p)][0]
                for freq_range in freq_range_list:
                    freqs = self.proc_data_dict['proc_sweep_points_2D_dict'][
                        qbn][param_name]
                    data = self.proc_data_dict['proc_data_to_fit'][qbn]
                    reduction_arr = np.logical_not(
                        np.logical_and(freqs > freq_range[0],
                                       freqs < freq_range[1]))
                    freqs_reshaped = freqs[reduction_arr]
                    self.proc_data_dict['proc_data_to_fit'][qbn] = \
                        data[reduction_arr]
                    self.proc_data_dict['proc_sweep_points_2D_dict'][qbn][
                        param_name] = freqs_reshaped

        # remove delays
        if delay_ranges_remove is not None:
            for qbn, delay_range_list in delay_ranges_remove.items():
                if delay_range_list is None:
                    continue
                for delay_range in delay_range_list:
                    delays = self.proc_data_dict['proc_sweep_points_dict'][qbn][
                        'msmt_sweep_points']
                    data = self.proc_data_dict['proc_data_to_fit'][qbn]
                    reduction_arr = np.logical_not(
                        np.logical_and(delays > delay_range[0],
                                       delays < delay_range[1]))
                    delays_reshaped = delays[reduction_arr]
                    self.proc_data_dict['proc_data_to_fit'][qbn] = \
                        np.concatenate([
                            data[:, :-self.num_cal_points][:, reduction_arr],
                            data[:, -self.num_cal_points:]], axis=1)
                    self.proc_data_dict['proc_sweep_points_dict'][qbn][
                        'msmt_sweep_points'] = delays_reshaped
                    self.proc_data_dict['proc_sweep_points_dict'][qbn][
                        'sweep_points'] = self.cp.extend_sweep_points(
                        delays_reshaped, qbn)

        self.sign_of_peaks = self.get_param_value('sign_of_peaks',
                                                  default_value=None)
        if self.sign_of_peaks is None:
            self.sign_of_peaks = {qbn: None for qbn in self.qb_names}
        for qbn in self.qb_names:
            if self.sign_of_peaks.get(qbn, None) is None:
                if self.rotation_type[qbn] == 'fixed_cal_points'\
                        or 'pca' in self.rotation_type[qbn].lower():
                    # e state corresponds to larger values than g state
                    # (either due to cal points or due to set_majority_sign)
                    self.sign_of_peaks[qbn] = 1
                else:
                    msmt_data = self.proc_data_dict['proc_data_to_fit'][qbn][
                        :, :-self.num_cal_points]
                    self.sign_of_peaks[qbn] = np.sign(np.mean(msmt_data) -
                                                      np.median(msmt_data))

        self.sigma_guess = self.get_param_value('sigma_guess')
        if self.sigma_guess is None:
            self.sigma_guess = {qbn: 10e6 for qbn in self.qb_names}

        self.from_lower = self.get_param_value('from_lower')
        if self.from_lower is None:
            self.from_lower = {qbn: False for qbn in self.qb_names}
        self.ghost = self.get_param_value('ghost')
        if self.ghost is None:
            self.ghost = {qbn: False for qbn in self.qb_names}

    def prepare_fitting_slice(self, freqs, qbn, mu_guess,
                              slice_idx=None, data_slice=None,
                              mu0_guess=None, do_double_fit=False):
        if slice_idx is None:
            raise ValueError('"slice_idx" cannot be None. It is used '
                             'for unique names in the fit_dicts.')
        if data_slice is None:
            data_slice = self.proc_data_dict['proc_data_to_fit'][qbn][
                         :, slice_idx]
        GaussianModel = lmfit.Model(fit_mods.DoubleGaussian) if do_double_fit \
            else lmfit.Model(fit_mods.Gaussian)
        ampl_guess = (data_slice.max() - data_slice.min()) / \
                     0.4 * self.sign_of_peaks[qbn] * self.sigma_guess[qbn]
        offset_guess = data_slice[0]
        GaussianModel.set_param_hint('sigma',
                                     value=self.sigma_guess[qbn],
                                     vary=True)
        GaussianModel.set_param_hint('mu',
                                     value=mu_guess,
                                     vary=True)
        GaussianModel.set_param_hint('ampl',
                                     value=ampl_guess,
                                     vary=True)
        GaussianModel.set_param_hint('offset',
                                     value=offset_guess,
                                     vary=True)
        if do_double_fit:
            GaussianModel.set_param_hint('sigma0',
                                         value=self.sigma_guess[qbn],
                                         vary=True)
            GaussianModel.set_param_hint('mu0',
                                         value=mu0_guess,
                                         vary=True)
            GaussianModel.set_param_hint('ampl0',
                                         value=ampl_guess/2,
                                         vary=True)
        guess_pars = GaussianModel.make_params()
        self.set_user_guess_pars(guess_pars)

        key = f'gauss_fit_{qbn}_slice{slice_idx}'
        self.fit_dicts[key] = {
            'fit_fn': GaussianModel.func,
            'fit_xvals': {'freq': freqs},
            'fit_yvals': {'data': data_slice},
            'guess_pars': guess_pars}

    def prepare_fitting(self):
        self.rectangles_exclude = self.get_param_value('rectangles_exclude')
        self.delays_double_fit = self.get_param_value('delays_double_fit')
        self.delay_ranges_to_fit = self.get_param_value(
            'delay_ranges_to_fit', default_value={})
        self.freq_ranges_to_fit = self.get_param_value(
            'freq_ranges_to_fit', default_value={})
        fit_first_cal_state = self.get_param_value(
            'fit_first_cal_state', default_value={})

        self.fit_dicts = OrderedDict()
        self.delays_for_fit = OrderedDict()
        self.freqs_for_fit = OrderedDict()
        for qbn in self.qb_names:
            # find name of 1st sweep point in sweep dimension 1
            param_name = [p for p in self.mospm[qbn]
                          if self.sp.find_parameter(p)][0]
            data = self.proc_data_dict['proc_data_to_fit'][qbn]
            delays = self.proc_data_dict['proc_sweep_points_dict'][qbn][
                'sweep_points']
            self.delays_for_fit[qbn] = np.array([])
            self.freqs_for_fit[qbn] = []
            dr_fit = self.delay_ranges_to_fit.get(qbn, [(min(delays),
                                                        max(delays))])
            fr_fit = self.freq_ranges_to_fit.get(qbn, [])
            if not fit_first_cal_state.get(qbn, True):
                first_cal_state = list(self.cal_states_dict_for_rotation[qbn])[0]
                first_cal_state_idxs = self.cal_states_dict[qbn][first_cal_state]
                if first_cal_state_idxs is None:
                    first_cal_state_idxs = []
            for i, delay in enumerate(delays):
                do_double_fit = False
                if not fit_first_cal_state.get(qbn, True) and \
                        i-len(delays) in first_cal_state_idxs:
                    continue
                if any([t[0] <= delay <= t[1] for t in dr_fit]):
                    data_slice = data[:, i]
                    freqs = self.proc_data_dict['proc_sweep_points_2D_dict'][
                        qbn][param_name]
                    if len(fr_fit):
                        mask = [np.logical_and(t[0] < freqs, freqs < t[1])
                                for t in fr_fit]
                        if len(mask) > 1:
                            mask = np.logical_or(*mask)
                        freqs = freqs[mask]
                        data_slice = data_slice[mask]

                    if self.rectangles_exclude is not None and \
                            self.rectangles_exclude.get(qbn, None) is not None:
                        for rectangle in self.rectangles_exclude[qbn]:
                            if rectangle[0] < delay < rectangle[1]:
                                reduction_arr = np.logical_not(
                                    np.logical_and(freqs > rectangle[2],
                                                   freqs < rectangle[3]))
                                freqs = freqs[reduction_arr]
                                data_slice = data_slice[reduction_arr]

                    if self.delays_double_fit is not None and \
                            self.delays_double_fit.get(qbn, None) is not None:
                        rectangle = self.delays_double_fit[qbn]
                        do_double_fit = rectangle[0] < delay < rectangle[1]

                    reduction_arr = np.invert(np.isnan(data_slice))
                    freqs = freqs[reduction_arr]
                    data_slice = data_slice[reduction_arr]

                    self.freqs_for_fit[qbn].append(freqs)
                    self.delays_for_fit[qbn] = np.append(
                        self.delays_for_fit[qbn], delay)

                    if do_double_fit:
                        peak_indices = sp.signal.find_peaks(
                            data_slice, distance=50e6/(freqs[1] - freqs[0]))[0]
                        peaks = data_slice[peak_indices]
                        srtd_idxs = np.argsort(np.abs(peaks))
                        mu_guess = freqs[peak_indices[srtd_idxs[-1]]]
                        mu0_guess = freqs[peak_indices[srtd_idxs[-2]]]
                    else:
                        mu_guess = freqs[np.argmax(
                            data_slice * self.sign_of_peaks[qbn])]
                        mu0_guess = None

                    self.prepare_fitting_slice(freqs, qbn, mu_guess, i,
                                               data_slice=data_slice,
                                               mu0_guess=mu0_guess,
                                               do_double_fit=do_double_fit)

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            delays = self.proc_data_dict['proc_sweep_points_dict'][qbn][
                'sweep_points']
            fit_keys = [k for k in self.fit_dicts if qbn in k.split('_')]
            fitted_freqs = np.zeros(len(fit_keys))
            fitted_freqs_errs = np.zeros(len(fit_keys))
            deep = False
            for i, fk in enumerate(fit_keys):
                fit_res = self.fit_dicts[fk]['fit_res']
                mu_param = 'mu'
                if 'mu0' in fit_res.best_values:
                    mu_param = 'mu' if fit_res.best_values['mu'] > \
                                       fit_res.best_values['mu0'] else 'mu0'

                fitted_freqs[i] = fit_res.best_values[mu_param]
                fitted_freqs_errs[i] = fit_res.params[mu_param].stderr
                if self.from_lower[qbn]:
                    if self.ghost[qbn]:
                        if (fitted_freqs[i - 1] - fit_res.best_values['mu']) / \
                                fitted_freqs[i - 1] > 0.05 and i > len(delays)-4:
                            deep = False
                        condition1 = ((fitted_freqs[i-1] -
                                     fit_res.best_values['mu']) /
                                     fitted_freqs[i-1]) < -0.015
                        condition2 = (i > 1 and i < (len(fitted_freqs) -
                                                     len(delays)))
                        if condition1 and condition2:
                            if deep:
                                mu_guess = fitted_freqs[i-1]
                                self.prepare_fitting_slice(
                                    self.freqs_for_fit[qbn][i], qbn, mu_guess, i)
                                self.run_fitting(keys_to_fit=[fk])
                                fitted_freqs[i] = self.fit_dicts[fk][
                                    'fit_res'].best_values['mu']
                                fitted_freqs_errs[i] = self.fit_dicts[fk][
                                    'fit_res'].params['mu'].stderr
                            deep = True
                else:
                    if self.ghost[qbn]:
                        if (fitted_freqs[i - 1] - fit_res.best_values['mu']) / \
                                fitted_freqs[i - 1] > -0.05 and \
                                i > len(delays) - 4:
                            deep = False
                        if (fitted_freqs[i - 1] - fit_res.best_values['mu']) / \
                                fitted_freqs[i - 1] > 0.015 and i > 1:
                            if deep:
                                mu_guess = fitted_freqs[i - 1]
                                self.prepare_fitting_slice(
                                    self.freqs_for_fit[qbn][i], qbn, mu_guess, i)
                                self.run_fitting(keys_to_fit=[fk])
                                fitted_freqs[i] = self.fit_dicts[fk][
                                    'fit_res'].best_values['mu']
                                fitted_freqs_errs[i] = self.fit_dicts[fk][
                                    'fit_res'].params['mu'].stderr
                            deep = True

            self.proc_data_dict['analysis_params_dict'][
                f'fitted_freqs_{qbn}'] = {'val': fitted_freqs,
                                          'stderr': fitted_freqs_errs}
            self.proc_data_dict['analysis_params_dict'][f'delays_{qbn}'] = \
                self.delays_for_fit[qbn]

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                base_plot_name = f'FluxPulseScope_{qbn}_{self.data_to_fit[qbn]}'
                xlabel, xunit = self.get_xaxis_label_unit(qbn)
                # find name of 1st sweep point in sweep dimension 1
                param_name = [p for p in self.mospm[qbn]
                              if self.sp.find_parameter(p)][0]
                ylabel = self.sp.get_sweep_params_property(
                    'label', dimension=1, param_names=param_name)
                yunit = self.sp.get_sweep_params_property(
                    'unit', dimension=1, param_names=param_name)
                xvals = self.proc_data_dict['proc_sweep_points_dict'][qbn][
                    'sweep_points']
                self.plot_dicts[f'{base_plot_name}_main'] = {
                    'plotfn': self.plot_colorxy,
                    'fig_id': base_plot_name,
                    'xvals': xvals,
                    'yvals': self.proc_data_dict['proc_sweep_points_2D_dict'][
                        qbn][param_name],
                    'zvals': self.proc_data_dict['proc_data_to_fit'][qbn],
                    'xlabel': xlabel,
                    'xunit': xunit,
                    'ylabel': ylabel,
                    'yunit': yunit,
                    'title': (self.raw_data_dict['timestamp'] + ' ' +
                              self.measurement_strings[qbn]),
                    'clabel': self.get_yaxis_label(qb_name=qbn)}

                self.plot_dicts[f'{base_plot_name}_fit'] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': self.delays_for_fit[qbn],
                    'yvals': self.proc_data_dict['analysis_params_dict'][
                                                 f'fitted_freqs_{qbn}']['val'],
                    'yerr': self.proc_data_dict['analysis_params_dict'][
                        f'fitted_freqs_{qbn}']['stderr'],
                    'color': 'r',
                    'linestyle': '-',
                    'marker': 'x'}

                # plot with log scale on x-axis
                self.plot_dicts[f'{base_plot_name}_main_log'] = {
                    'plotfn': self.plot_colorxy,
                    'fig_id': f'{base_plot_name}_log',
                    'xvals': xvals*1e6,
                    'yvals': self.proc_data_dict['proc_sweep_points_2D_dict'][
                        qbn][param_name]/1e9,
                    'zvals': self.proc_data_dict['proc_data_to_fit'][qbn],
                    'xlabel': f'{xlabel} ($\\mu${xunit})',
                    'ylabel': f'{ylabel} (G{yunit})',
                    'logxscale': True,
                    'xrange': [min(xvals*1e6), max(xvals*1e6)],
                    'no_label_units': True,
                    'no_label': True,
                    'clabel': self.get_yaxis_label(qb_name=qbn)}

                self.plot_dicts[f'{base_plot_name}_fit_log'] = {
                    'fig_id': f'{base_plot_name}_log',
                    'plotfn': self.plot_line,
                    'xvals': self.delays_for_fit[qbn]*1e6,
                    'yvals': self.proc_data_dict['analysis_params_dict'][
                        f'fitted_freqs_{qbn}']['val']/1e9,
                    'yerr': self.proc_data_dict['analysis_params_dict'][
                        f'fitted_freqs_{qbn}']['stderr']/1e9,
                    'title': (self.raw_data_dict['timestamp'] + ' ' +
                              self.measurement_strings[qbn]),
                    'color': 'r',
                    'linestyle': '-',
                    'marker': 'x'}


class RunTimeAnalysis(ba.BaseDataAnalysis):
    """
    Provides elementary analysis of Run time by plotting all timers
    saved in the hdf5 file of a measurement.
    """
    def __init__(self,
                 label: str = '',
                 t_start: str = None, t_stop: str = None, data_file_path: str = None,
                 options_dict: dict = None, extract_only: bool = False,
                 do_fitting: bool = True, auto=True,
                 params_dict=None, numeric_params=None, **kwargs):

        super().__init__(t_start=t_start, t_stop=t_stop, label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting, **kwargs)
        self.timers = {}

        if self.job is None:
            self.create_job(t_start=t_start, t_stop=t_stop,
                            label=label, data_file_path=data_file_path,
                            do_fitting=do_fitting, options_dict=options_dict,
                            extract_only=extract_only, params_dict=params_dict,
                            numeric_params=numeric_params, **kwargs)
        self.params_dict = {
            f"{tm_mod.Timer.HDF_GRP_NAME}": f"{tm_mod.Timer.HDF_GRP_NAME}",
            "repetition_rate": "Instrument settings.TriggerDevice.pulse_period",
                            }

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        timers_dicts = self.raw_data_dict.get('Timers', {})
        for t, v in timers_dicts.items():
            self.timers[t] = tm_mod.Timer(name=t, **v)

        self.extract_nr_runs_per_segment()

        # Extract and build raw measurement timer
        self.timers['BareMeasurement'] = self.bare_measurement_timer(
            ref_time=self.get_param_value("ref_time")
        )

        # Extract bare experiment timer
        self.timers['BareExperiment'] = self.bare_experiment_timer(
            ref_time=self.get_param_value("ref_time")
        )

    def process_data(self):
        pass

    def plot(self, **kwargs):
        timers = [t for t in self.timers.values()]
        plot_kws = self.get_param_value('plot_kwargs', {})
        for t in timers:
            try:
                if len(t):
                    self.figs["timer_" + t.name] = t.plot(**plot_kws)
                # plot combined plot of children
                if self.get_param_value('children_timer', True) and len(t.children):
                    self.figs['timer_' + t.name + "_children"] = \
                        tm_mod.multi_plot(list(t.children.values()), **plot_kws)
                    # recursive plot
                    for n, subtimer in t.children.items():
                        if len(subtimer.children):
                            self.figs[f'timer_{n}'] = subtimer.plot(**plot_kws)
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                log.error(f'Could not plot Timer: {t.name}: {e}')

        if self.get_param_value('combined_timer', True):
            self.figs['timer_all'] = tm_mod.multi_plot(timers,
                                                       **plot_kws)

    def bare_measurement_timer(self, ref_time=None,
                               checkpoint='bare_measurement', **kw):
        """
        Creates and returns an additional timer for the bare measurement time,
        which is defined as the repetition period between two master triggers
        times the total number of runs

        Args:
            ref_time: Reference time for the start of the bare measurement timer.
                If None, the earliest time found in self.timers is used.
            checkpoint (str): name of the bare measurement checkpoint
            **kw: remaining keywords are passed to bare_measurement_time

        """
        bmtime = self.bare_measurement_time(**kw)
        bmtimer = tm_mod.Timer('BareMeasurement', auto_start=False)
        if ref_time is None:
            try:
                ts = [t.find_earliest() for t in self.timers.values()]
                ts = [t[-1] for t in ts if len(t)]
                arg_sorted = sorted(range(len(ts)),
                                    key=list(ts).__getitem__)
                ref_time = ts[arg_sorted[0]]
            except Exception as e:
                log.error('Failed to extract reference time for bare'
                          f'Measurement timer. Please fix the error'
                          f'or pass in a reference time manually.')
                raise e

        # TODO add more options of how to distribute the bm time in the timer
        #  (not only start stop to get the correct duration but
        #  add it when the acquisition device is actually acquiring data)
        bmtimer.checkpoint(f"BareMeasurement.{checkpoint}.start",
                           values=[ref_time], log_init=False)
        bmtimer.checkpoint(f"BareMeasurement.{checkpoint}.end",
                           values=[ ref_time + dt.timedelta(seconds=bmtime)],
                           log_init=False)

        return bmtimer

    def bare_experiment_timer(self, ref_time=None,
                                   checkpoint='bare_experiment', **kw):
        tot_sec = self.bare_experiment_time(**kw)
        tm = tm_mod.Timer('BareExperiment', auto_start=False)
        if ref_time is None:
            try:
                ts = [t.find_earliest() for t in self.timers.values()]
                ts = [t[-1] for t in ts if len(t)]
                arg_sorted = sorted(range(len(ts)),
                                    key=list(ts).__getitem__)
                ref_time = ts[arg_sorted[0]]
            except Exception as e:
                log.error('Failed to extract reference time for bare'
                          f'Experiment timer. Please fix the error'
                          f'or pass in a reference time manually.')
                raise e

        # TODO add more options of how to distribute the bm time in the timer
        #  (not only start stop but e.g. distribute it)
        tm.checkpoint(f"BareExperiment.{checkpoint}.start",
                           values=[ref_time], log_init=False)
        tm.checkpoint(f"BareExperiment.{checkpoint}.end",
                           values=[ref_time + dt.timedelta(seconds=tot_sec)],
                           log_init=False)

        return tm

    def extract_nr_runs_per_segment(self):
        """Extracts the total number of times each segment has been measured

        Sets self.nr_averages and self.nr_shots, by extracting them either
        from the metadata (see self.get_param_value), or from the options_dict.
        Note that for average readout, the number of averages is > 1 in the
        detectors while nr_shots is =1 (and vice-versa for single-shot readout).
        Warning: this code currently does not support soft averages or soft
        repetitions /!\
        """
        try:
            sa = self.get_instrument_setting('MC.soft_avg')
            if sa is not None and sa != 1:
                log.warning('Currently, soft averages are not taken into account'
                          'when extracting the number of averages. This might lead'
                          'to unexpected timings.')
            sr = self.get_instrument_setting('MC.soft_repetitions')
            if sr is not None and sr != 1:
                log.warning('Currently, soft repetitions are not taken into account'
                          'when extracting the number of shots. This might lead'
                          'to unexpected timings.')
        except:
            # in case some of the attributes do not exist
            pass
        #TODO Currently does not support soft averages or soft repetitions
        for param in ['nr_averages', 'nr_shots']:
            val = self.get_param_value(
                param, self._extract_param_from_det(param))
            if val is None:
                # Note that both nr_averages and nr_shots are required,
                # see self.bare_experiment_time.
                raise ValueError(
                    f'Could not extract "{param}" from hdf file. Please make '
                    f'sure that your measurement stores it (or pass it '
                    f'via the options_dict).')
            setattr(self, param, val)

    def _extract_param_from_det(self, param, default=None):
        det_metadata = self.metadata.get("Detector Metadata", None)
        val = None
        if det_metadata is not None:
            # multi detector function: look for child "detectors"
            # assumes at least 1 child and that all children have the same
            # number of averages
            val = det_metadata.get(param, None)
            if val is None:
                det = list(det_metadata.get('detectors', {}).values())[0]
                val = det.get(param, None)
        if val is None:
            val = default
        return val

    def bare_measurement_time(self, nr_averages=None, repetition_rate=None,
                              count_nan_measurements=False):
        """
        Computes the measurement time i.e. the number of segments * the repetition rate
        * number of averages.
        Args:
            nr_averages: number of averages for each segment
            repetition_rate: time interval between two main triggers of
                subsequent segments
            count_nan_measurements: whether or not to account for segments in
                unfinished measurements.

        Returns:

        """
        if nr_averages is None:
            nr_averages = self.nr_averages
        det_metadata = self.metadata.get("Detector Metadata", None)
        # metadata['detectors'] is a dict for MultiPollDetector and else a list
        df_names = list(det_metadata['detectors'])
        # Testing the first detector, since detectors in a MultiPollDetector
        # should all be the same
        if 'scope' in df_names[0]:
            # No scaling needed, since all hsp are contained in one hardware run
            n_hsp = 1
        else:
            # Note that the number of shots is already included in n_hsp
            n_hsp = len(self.raw_data_dict['hard_sweep_points'])
            prep_params = self.metadata['preparation_params']
            if 'active' in prep_params['preparation_type']:
                # If reset: n_hsp already includes the number of shots
                # and the final readout is interleaved with n_reset readouts
                n_resets = prep_params['reset_reps']
                n_hsp = n_hsp // (1 + n_resets)
        n_ssp = len(self.raw_data_dict.get('soft_sweep_points', [0]))
        if repetition_rate is None:
            repetition_rate = self.raw_data_dict["repetition_rate"]
        if count_nan_measurements:
            perc_meas = 1
        else:
            # When sweep points are skipped, data is missing in all columns
            # Thus, we can simply check in the first column.
            vals = list(self.raw_data_dict['measured_data'].values())[0]
            perc_meas = 1 - np.sum(np.isnan(vals)) / np.prod(vals.shape)
        return self._bare_measurement_time(n_ssp, n_hsp, repetition_rate,
                                           nr_averages, perc_meas)

    def bare_experiment_time(self, nr_averages=None, nr_shots=None):
        """
        Computes the bare experiment time from the segments durations,
        which are extracted from the segment timers.
        Note: unlike bare_measurement_time,
            for now this function always accounts for the full experiment time
            even if the measurement was interrupted.
        Note: Typically nr_shots = 1 for an averaged measurement, and
            nr_averages = 1 for a single-shot measurement, so the total
            number of runs is their product. This assumes that both were
            passed in the hdf file by the detector function.
        Args:
            nr_averages: Number of averages for each segment. Defaults to
                self.nr_averages
            nr_shots: Number of shots for each segment. Defaults to
                self.nr_shots

        Returns:
            Total experiment time (in seconds)
        """

        if nr_averages is None:
            nr_averages = self.nr_averages
        if nr_shots is None:
            nr_shots = self.nr_shots
        try:
            # duration of segments are stored with the .dt checkpoint end
            ckpts = self.timers['Sequences'].find_keys('.dt',
                                                       search_children=True)
            # since ns are not supported by datetime, the durations are stored
            # in micro seconds, so we need to set them back to nanoseconds
            segment_durations = [
                (self.timers['Sequences'][ck][0] -
                 dt.datetime.utcfromtimestamp(0)).total_seconds() / 1e3
                for ck in ckpts]
            self.proc_data_dict["segment_durations"] = \
                {ck:d for ck, d in zip(ckpts, segment_durations)}

            tot_secs = np.sum(segment_durations) * nr_averages * nr_shots
            return tot_secs
        except Exception as e:
            log.error(f'Could not compute bare experiment time : {e}.'
                      f'Returning timer with 0s duration.')
            return 0

    @staticmethod
    def _bare_measurement_time(n_ssp, n_hsp, repetition_rate, nr_averages,
                               percentage_measured):
        return n_ssp * n_hsp * repetition_rate * nr_averages \
               * percentage_measured


class MixerCarrierAnalysis(MultiQubit_TimeDomain_Analysis):
    """Analysis for the :py:meth:~'QuDev_transmon.calibrate_drive_mixer_carrier_model' measurement.

    The class extracts the DC biases on the I and Q channel inputs of the
    measured IQ mixer that minimize the LO leakage.
    """
    def process_data(self):
        super().process_data()

        hsp = self.raw_data_dict['hard_sweep_points']
        ssp = self.raw_data_dict['soft_sweep_points']
        mdata = self.raw_data_dict['measured_data']

        # Conversion from V_peak -> V_RMS
        V_RMS = list(mdata.values())[0]/np.sqrt(2)
        # Conversion to P (dBm):
        #   P = V_RMS^2 / 50 Ohms
        #   P (dBm) = 10 * log10(P / 1 mW)
        #   P (dBm) = 10 * log10(V_RMS^2 / 50 Ohms / 1 mW)
        #   P (dBm) = 20 * log10(V_RMS) - 10 * log10(50 Ohms * 1 mW)
        LO_dBm = 20*np.log10(V_RMS) - 10 * np.log10(50 * 1e-3)

        if len(hsp) * len(ssp) == len(LO_dBm.flatten()):
            # sweep points are aligned on grid

            # The arrays hsp and ssp define the edges of a grid of measured 
            # points. We reshape the arrays such that each data point 
            # LO_dBm[i] corresponds to the sweep point VI[i], VQ[i]
            self.proc_data_dict['sweeppoints_are_grid'] = True
            # save raw format of data for plotting with plot_colorxy
            self.proc_data_dict['V_I_raw_format'] = hsp
            self.proc_data_dict['V_Q_raw_format'] = ssp
            self.proc_data_dict['LO_leakage_raw_format'] = LO_dBm.T

            VI, VQ = np.meshgrid(hsp, ssp)
            VI = VI.flatten()
            VQ = VQ.flatten()
            LO_dBm = LO_dBm.T.flatten()            
        else:
            # sweep points are random
            self.proc_data_dict['sweeppoints_are_grid'] = False
            VI = hsp
            VQ = ssp

        self.proc_data_dict['V_I'] = VI
        self.proc_data_dict['V_Q'] = VQ
        self.proc_data_dict['LO_leakage'] = LO_dBm
        self.proc_data_dict['data_to_fit'] = LO_dBm

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        VI = self.proc_data_dict['V_I']
        VQ = self.proc_data_dict['V_Q']
        data = self.proc_data_dict['data_to_fit']

        mixer_lo_leakage_mod = lmfit.Model(fit_mods.mixer_lo_leakage,
                                           independent_vars=['vi', 'vq'])
        # Use two lowest values in measurements to choose
        # initial model parameters.
        VI_two_lowest = VI[np.argpartition(data, 2)][0:2]
        VQ_two_lowest = VQ[np.argpartition(data, 2)][0:2]
        minimum = - np.mean(VI_two_lowest) + 1j * np.mean(VQ_two_lowest)
        li_guess = np.abs(minimum)
        theta_i_guess = cmath.phase(minimum)
        guess_pars = fit_mods.mixer_lo_leakage_guess(mixer_lo_leakage_mod,
                                                     li=li_guess,
                                                     theta_i=theta_i_guess)

        self.fit_dicts['mixer_lo_leakage'] = {
            'model': mixer_lo_leakage_mod,
            'fit_xvals': {'vi': VI,
                          'vq': VQ},
            'fit_yvals': {'data': data},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        fit_dict = self.fit_dicts['mixer_lo_leakage']
        best_values = fit_dict['fit_res'].best_values
        # compute values that minimize the fitted model:
        leakage = best_values['li'] * np.exp(1j* best_values['theta_i']) \
                  - 1j * best_values['lq'] * np.exp(1j*best_values['theta_q'])
        adict = self.proc_data_dict['analysis_params_dict']
        adict['V_I'] = -leakage.real
        adict['V_Q'] = leakage.imag

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        pdict = self.proc_data_dict
        V_I = pdict['V_I']
        V_Q = pdict['V_Q']

        timestamp = self.timestamps[0]

        leakage = pdict['LO_leakage']
        leakage_dBm_amp_zrange = [1.1*np.min(leakage), 
                                  0.9*np.max(leakage)]

        if pdict['sweeppoints_are_grid']:
            # If the sweeppoints are aligned in a grid we can plot a 2D
            # histogram of the measured data

            hist_plot_name = 'mixer_lo_leakage_histogram'
            self.plot_dicts['hist_measurement'] = {
                'fig_id': hist_plot_name,
                'plotfn': self.plot_colorxy,
                'xvals': pdict['V_I_raw_format'],
                'yvals': pdict['V_Q_raw_format'],
                'zvals': pdict['LO_leakage_raw_format'],
                'zrange': leakage_dBm_amp_zrange,
                'xlabel': 'Offset, $V_\\mathrm{I}$',
                'ylabel': 'Offset, $V_\\mathrm{Q}$',
                'xunit': 'V',
                'yunit': 'V',
                'setlabel': 'lo leakage magnitude',
                'cmap': 'plasma',
                'cmap_levels': 100,
                'clabel': 'Carrier Leakage $V_\\mathrm{LO}$ (dBm)',
                'title': f'{timestamp} calibrate_drive_mixer_carrier_'
                         f'{self.qb_names[0]}'
            }

            V_I_opt = pdict['analysis_params_dict']['V_I']
            V_Q_opt = pdict['analysis_params_dict']['V_Q']
            self.plot_dicts['hist_minimum'] = {
                'fig_id': hist_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([V_I_opt]),
                'yvals': np.array([V_Q_opt]),
                'setlabel': '$V_\\mathrm{I}$' + f' ={V_I_opt*1e3:.1f}$\,$mV\n'
                            '$V_\\mathrm{Q}$' + f' ={V_Q_opt*1e3:.1f}$\,$mV',
                'color': 'red',
                'marker': 'o',
                'linestyle': 'None',
                'do_legend': True,
                'legend_pos': 'upper right',
                'legend_title': None,
                'legend_frameon': True
            }

        plot_name_dict = {ch: f'V_{ch}_vs_LO_magn' for ch in ['I', 'Q']} 
        for ch in ['I', 'Q']:
            self.plot_dicts[f'raw_V_{ch}_vs_LO_magn'] = {
                'fig_id': plot_name_dict[ch],
                'plotfn': self.plot_line,
                'xvals': pdict[f'V_{ch}'],
                'yvals': leakage,
                'color': 'blue',
                'marker': '.',
                'linestyle': 'None',
                'xlabel': f'Offset, $V_\\mathrm{{{ch}}}$',
                'ylabel': 'Carrier Leakage $V_\\mathrm{LO}$',
                'xunit': 'V',
                'yunit': 'dBm',
                'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO}}$ '
                         f'projected onto offset $V_\\mathrm{{{ch}}}$',
                'do_legend': True,
                'setlabel': 'measurement',
            }

        if self.do_fitting:
            # interpolate data for plot,
            # define grid with limits based on measurement
            # points and make it 10 % larger in both axes
            size_offset_vi = 0.05 * (np.max(V_I) - np.min(V_I))
            size_offset_vq = 0.05 * (np.max(V_Q) - np.min(V_Q))
            vi = np.linspace(np.min(V_I) - size_offset_vi,
                             np.max(V_I) + size_offset_vi, 250)
            vq = np.linspace(np.min(V_Q) - size_offset_vq,
                             np.max(V_Q) + size_offset_vq, 250)
            V_I_plot, V_Q_plot = np.meshgrid(vi, vq)
            fit_dict = self.fit_dicts['mixer_lo_leakage']
            fit_res = fit_dict['fit_res']
            best_values = fit_res.best_values
            model_func = fit_dict['model'].func
            z = model_func(V_I_plot, V_Q_plot, **best_values)

            base_plot_name = 'mixer_lo_leakage'
            self.plot_dicts['base_contour'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_contourf,
                'xvals': V_I_plot,
                'yvals': V_Q_plot,
                'zvals': z,
                'zrange': leakage_dBm_amp_zrange,
                'xlabel': 'Offset, $V_\\mathrm{I}$',
                'ylabel': 'Offset, $V_\\mathrm{Q}$',
                'xunit': 'V',
                'yunit': 'V',
                'setlabel': 'lo leakage magnitude',
                'cmap': 'plasma',
                'cmap_levels': 100,
                'clabel': 'Carrier Leakage $V_\\mathrm{LO}$ (dBm)',
                'title': f'{timestamp} calibrate_drive_mixer_carrier_'
                         f'{self.qb_names[0]}'
            }

            self.plot_dicts['base_measurement_points'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_line,
                'xvals': V_I,
                'yvals': V_Q,
                'color': 'white',
                'marker': '.',
                'linestyle': 'None',
                'setlabel': ''
            }

            V_I_opt = pdict['analysis_params_dict']['V_I']
            V_Q_opt = pdict['analysis_params_dict']['V_Q']
            self.plot_dicts['base_minimum'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([V_I_opt]),
                'yvals': np.array([V_Q_opt]),
                'setlabel': '$V_\\mathrm{I}$' + f' ={V_I_opt*1e3:.1f}$\,$mV\n'
                            '$V_\\mathrm{Q}$' + f' ={V_Q_opt*1e3:.1f}$\,$mV',
                'color': 'red',
                'marker': 'o',
                'linestyle': 'None',
                'do_legend': True,
                'legend_pos': 'upper right',
                'legend_title': None,
                'legend_frameon': True
            }

            self.plot_dicts[f'optimum_V_I_vs_LO_magn'] = {
                'fig_id': plot_name_dict['I'],
                'plotfn': self.plot_line,
                'xvals': np.array([V_I_opt, V_I_opt]),
                'yvals': leakage_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '--',
                'setlabel': '$V_\\mathrm{I}$' + f' ={V_I_opt*1e3:.1f}$\,$mV',
                'do_legend': True,
            }
       
            self.plot_dicts[f'fit_V_I_vs_LO_magn'] = {
                'fig_id': plot_name_dict['I'],
                'plotfn': self.plot_line,
                'xvals': vi,
                'yvals': model_func(vi, V_Q_opt, **best_values),
                'yrange': leakage_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '-',
                'setlabel': '\nfitted model\n'
                            '@ $V_\\mathrm{Q}$'
                            f'={V_Q_opt*1e3:.1f}$\,$mV',
                'do_legend': True,
            }

            self.plot_dicts[f'optimum_V_Q_vs_LO_magn'] = {
                'fig_id': plot_name_dict['Q'],
                'plotfn': self.plot_line,
                'xvals': np.array([V_Q_opt, V_Q_opt]),
                'yvals': leakage_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '--',
                'setlabel': '$V_\\mathrm{Q}$' + f' ={V_Q_opt*1e3:.1f}$\,$mV',
                'do_legend': True,
            }

            self.plot_dicts[f'fit_V_Q_vs_LO_magn'] = {
                'fig_id': plot_name_dict['Q'],
                'plotfn': self.plot_line,
                'xvals': vq,
                'yvals': model_func(V_I_opt, vq, **best_values),
                'yrange': leakage_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '-',
                'setlabel': '\nfitted model\n'
                            '@ $V_\\mathrm{I}$'
                            f'={V_I_opt*1e3:.1f}$\,$mV',
                'do_legend': True,
            }
            
                


class MixerSkewnessAnalysis(MultiQubit_TimeDomain_Analysis):
    """Analysis for the :py:meth:~'QuDev_transmon.calibrate_drive_mixer_skewness_model' measurement.

    The class extracts the phase and amplitude correction settings of the Q
    channel input of the measured IQ mixer that maximize the suppression of the
    unwanted sideband.
    """
    def process_data(self):
        super().process_data()

        hsp = self.raw_data_dict['hard_sweep_points']
        ssp = self.raw_data_dict['soft_sweep_points']
        mdata = self.raw_data_dict['measured_data']

        sideband_I, sideband_Q = list(mdata.values())

        if len(hsp) * len(ssp) == len(sideband_I.flatten()):
            # sweep points are aligned on grid

            # The arrays hsp and ssp define the edges of a grid of measured
            # points. We reshape the arrays such that each data point
            # sideband_I/Q[i] corresponds to the sweep point alpha[i], phase[i]
            self.proc_data_dict['sweeppoints_are_grid'] = True
            alpha, phase = np.meshgrid(hsp, ssp)
            alpha = alpha.flatten()
            phase = phase.flatten()
            sideband_I = sideband_I.T.flatten()
            sideband_Q = sideband_Q.T.flatten()
        else:
            # sweep points are random
            self.proc_data_dict['sweeppoints_are_grid'] = False
            alpha = hsp
            phase = ssp

        # Conversion from V_peak -> V_RMS
        #   V_RMS = sqrt(V_peak_I^2 + V_peak_Q^2)/sqrt(2)
        # Conversion to P (dBm):
        #   P = V_RMS^2 / 50 Ohms
        #   P (dBm) = 10 * log10(P / 1 mW)
        #   P (dBm) = 10 * log10(V_RMS^2 / 50 Ohms / 1 mW)
        #   P (dBm) = 10 * log10(V_RMS^2) - 10 * log10(50 Ohms * 1 mW)
        #   P (dBm) = 10 * log10(V_peak_I^2 + V_peak_Q^2)
        #             - 10 * log10(2 * 50 Ohms * 1 mW)
        sideband_dBm_amp = 10 * np.log10(sideband_I**2 + sideband_Q**2) \
                           - 10 * np.log10(2 * 50 * 1e-3)

        self.proc_data_dict['alpha'] = alpha
        self.proc_data_dict['phase'] = phase
        self.proc_data_dict['sideband_I'] = sideband_I
        self.proc_data_dict['sideband_Q'] = sideband_Q
        self.proc_data_dict['sideband_dBm_amp'] = sideband_dBm_amp
        self.proc_data_dict['data_to_fit'] = sideband_dBm_amp

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        data = self.proc_data_dict['data_to_fit']

        mixer_imbalance_sideband_mod = lmfit.Model(
            fit_mods.mixer_imbalance_sideband,
            independent_vars=['alpha', 'phi_skew']
            )
        # Use two lowest values in measurements to choose
        # initial model parameters.
        alpha_two_lowest = self.proc_data_dict['alpha'][np.argpartition(data, 2)][0:2]
        phi_two_lowest = self.proc_data_dict['phase'][np.argpartition(data, 2)][0:2]
        g_guess = np.mean(alpha_two_lowest)
        phi_guess = - np.mean(phi_two_lowest)
        guess_pars = fit_mods.mixer_imbalance_sideband_guess(
            mixer_imbalance_sideband_mod,
            g=g_guess,
            phi=phi_guess
            )

        self.fit_dicts['mixer_imbalance_sideband'] = {
            'model': mixer_imbalance_sideband_mod,
            'fit_xvals': {'alpha': self.proc_data_dict['alpha'],
                          'phi_skew': self.proc_data_dict['phase']},
            'fit_yvals': {'data': self.proc_data_dict['data_to_fit']},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        fit_dict = self.fit_dicts['mixer_imbalance_sideband']
        best_values = fit_dict['fit_res'].best_values
        self.proc_data_dict['analysis_params_dict']['alpha'] = best_values['g']
        self.proc_data_dict['analysis_params_dict']['phase'] = -best_values['phi']

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        pdict = self.proc_data_dict

        alpha = pdict['alpha']
        phase = pdict['phase']

        sideband_dBm_amp = pdict['sideband_dBm_amp']
        sideband_dBm_amp_zrange = [1.1*np.min(sideband_dBm_amp), 
                                   0.9*np.max(sideband_dBm_amp)]

        timestamp = self.timestamps[0]

        if pdict['sweeppoints_are_grid']:
            # If the sweeppoints are aligned in a grid we can plot a 2D
            # histogram of the measured data

            # Here we use the raw sweep points as they have the correct format
            # for the plotting function plot_colorxy
            alpha_raw = self.raw_data_dict['hard_sweep_points']
            phi_raw = self.raw_data_dict['soft_sweep_points']
            mdata = self.raw_data_dict['measured_data']

            sideband_I, sideband_Q = list(mdata.values())

            # See comment in process_data for the derivation of the conversion.
            sideband_dBm_amp = 10 * np.log10(sideband_I**2 + sideband_Q**2) \
                               - 10 * np.log10(2 * 50 * 1e-3)

            hist_plot_name = 'mixer_sideband_suppression_histogram'
            self.plot_dicts['raw_histogram'] = {
                'fig_id': hist_plot_name,
                'plotfn': self.plot_colorxy,
                'xvals': alpha_raw,
                'yvals': phi_raw,
                'zvals': sideband_dBm_amp.T,
                'zrange': sideband_dBm_amp_zrange,
                'xlabel': 'Ampl., Ratio, $\\alpha$',
                'ylabel': 'Phase Off., $\\Delta\\phi$',
                'xunit': '',
                'yunit': 'deg',
                'setlabel': 'sideband magnitude',
                'cmap': 'plasma',
                'cmap_levels': 100,
                'clabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$ (dBm)',
                'title': f'{timestamp} calibrate_drive_mixer_skewness_'
                        f'{self.qb_names[0]}'
            }

            if self.do_fitting:
                alpha_min = pdict['analysis_params_dict']['alpha']
                phase_min = pdict['analysis_params_dict']['phase']
                self.plot_dicts['raw_histogram_fit_result'] = {
                    'fig_id': hist_plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([alpha_min]),
                    'yvals': np.array([phase_min]),
                    'setlabel': f'$\\alpha$ ={alpha_min:.2f}\n'
                                f'$\phi$ ={phase_min:.2f}$^\\circ$',
                    'color': 'red',
                    'marker': 'o',
                    'linestyle': 'None',
                    'do_legend': True,
                    'legend_pos': 'upper right',
                    'legend_title': None,
                    'legend_frameon': True
                }

        raw_alpha_plot_name = 'alpha_vs_sb_magn'
        self.plot_dicts['raw_alpha_vs_sb_magn'] = {
            'fig_id': raw_alpha_plot_name,
            'plotfn': self.plot_line,
            'xvals': alpha,
            'yvals': pdict['sideband_dBm_amp'],
            'color': 'blue',
            'marker': '.',
            'linestyle': 'None',
            'xlabel': 'Ampl., Ratio, $\\alpha$',
            'ylabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$',
            'xunit': '',
            'yunit': 'dBm',
            'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO-IF}}$ '
                     f'projected onto ampl. ratio $\\alpha$',
            'do_legend': True,
            'setlabel': 'measurement',
        }

        raw_phase_plot_name = 'phase_vs_sb_magn'
        self.plot_dicts['raw_phase_vs_sb_magn'] = {
            'fig_id': raw_phase_plot_name,
            'plotfn': self.plot_line,
            'xvals': phase,
            'yvals': pdict['sideband_dBm_amp'],
            'color': 'blue',
            'marker': '.',
            'linestyle': 'None',
            'xlabel': 'Phase Off., $\\Delta\\phi$',
            'ylabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$',
            'xunit': 'deg',
            'yunit': 'dBm',
            'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO-IF}}$ '
                     f'projected onto phase offset $\\Delta\\phi$',
            'do_legend': True,
            'setlabel': 'measurement',
        }

        if self.do_fitting:
            # define grid with limits based on measurement points 
            # and make it 10 % larger in both axes
            size_offset_alpha = 0.05*(np.max(alpha)-np.min(alpha))
            size_offset_phase = 0.05*(np.max(phase)-np.min(phase))
            alpha_edge = np.linspace(np.min(alpha) - size_offset_alpha, 
                            np.max(alpha) + size_offset_alpha, 250)
            phase_edge = np.linspace(np.min(phase) - size_offset_phase, 
                            np.max(phase) + size_offset_phase, 250)
            alpha_plot, phase_plot = np.meshgrid(alpha_edge, phase_edge)

            fit_dict = self.fit_dicts['mixer_imbalance_sideband']
            fit_res = fit_dict['fit_res']
            best_values = fit_res.best_values
            model_func = fit_dict['model'].func
            z = model_func(alpha_plot, phase_plot, **best_values)

            base_plot_name = 'mixer_sideband_suppression'
            self.plot_dicts['base_contour'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_contourf,
                'xvals': alpha_plot,
                'yvals': phase_plot,
                'zvals': z,
                'zrange': sideband_dBm_amp_zrange,
                'xlabel': 'Ampl., Ratio, $\\alpha$',
                'ylabel': 'Phase Off., $\\Delta\\phi$',
                'xunit': '',
                'yunit': 'deg',
                'setlabel': 'sideband magnitude',
                'cmap': 'plasma',
                'cmap_levels': 100,
                'clabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$ (dBm)',
                'title': f'{timestamp} calibrate_drive_mixer_skewness_'
                        f'{self.qb_names[0]}'
            }

            self.plot_dicts['base_measurement_points'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_line,
                'xvals': alpha,
                'yvals': phase,
                'color': 'white',
                'marker': '.',
                'linestyle': 'None',
                'setlabel': '',
            }

            alpha_min = pdict['analysis_params_dict']['alpha']
            phase_min = pdict['analysis_params_dict']['phase']
            self.plot_dicts['base_minimum'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([alpha_min]),
                'yvals': np.array([phase_min]),
                'setlabel': f'$\\alpha$ ={alpha_min:.2f}\n'
                            f'$\phi$ ={phase_min:.2f}$^\\circ$',
                'color': 'red',
                'marker': 'o',
                'linestyle': 'None',
                'do_legend': True,
                'legend_pos': 'upper right',
                'legend_title': None,
                'legend_frameon': True
            }

            self.plot_dicts['optimum_in_alpha_vs_sb_magn'] = {
                'fig_id': raw_alpha_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([alpha_min, alpha_min]),
                'yvals': sideband_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '--',
                'setlabel': f'$\\alpha$ ={alpha_min:.2f}',
                'do_legend': True,
            }

            self.plot_dicts['fit_in_alpha_vs_sb_magn'] = {
                'fig_id': raw_alpha_plot_name,
                'plotfn': self.plot_line,
                'xvals': alpha_edge,
                'yvals': model_func(alpha_edge, phase_min, **best_values),
                'yrange': sideband_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '-',
                'setlabel': f'\nfitted model\n'
                            f'@ $\phi$ ={phase_min:.2f}$^\\circ$',
                'do_legend': True,
            }

            self.plot_dicts['optimum_in_phase_vs_sb_magn'] = {
                'fig_id': raw_phase_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([phase_min, phase_min]),
                'yvals': sideband_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '--',
                'setlabel': f'$\phi$ ={phase_min:.2f}$^\\circ$',
                'do_legend': True,
            }

            self.plot_dicts['fit_in_phase_vs_sb_magn'] = {
                'fig_id': raw_phase_plot_name,
                'plotfn': self.plot_line,
                'xvals': phase_edge,
                'yvals': model_func(alpha_min, phase_edge, **best_values),
                'yrange': sideband_dBm_amp_zrange,
                'color': 'red',
                'marker': 'None',
                'linestyle': '-',
                'setlabel': f'\nfitted model\n'
                            f'@ $\\alpha$ ={alpha_min:.2f}',
                'do_legend': True,
            }


class NPulseAmplitudeCalibAnalysis(MultiQubit_TimeDomain_Analysis):
    """
    Analysis class for the DriveAmpCalib measurement.
    The typical accepted input parameters are described in the parent class.

    Additional parameters that this class recognises, which can be passed in the
    options_dict:
        - nr_pulses_pi (int; default: None): specifying into how many identical
            pulses a pi rotation was divided ( nr_pulses_pi pulses with rotation
            angle pi/nr_pulses_pi ). See docstring of the measurement class.
            Can also be dict with qb names as keys.
        - fixed scaling (int; default: None): specifying the amplitude scaling
            for the pulse that wasn't swept. See docstring of the measurement
            class. Can also be dict with qb names as keys.
        - fitted_scaling_errors (dict; default: None): the keys are qubit names
            and the values are arrays of fit results for each soft sweep
            ex: {'qb10':  np.array([-0.00077432, -0.00055945])}
        - maxeval (int; default: 400): number of evaluations for the optimiser
        - T1 (float; default: value from hdf): qubit T1 to use for fit (fixed)
        - T2 (float; default: value from hdf): qubit T2 to use as starting value
            for the fit (dimensionless fraction T2/T2_guess is varied)
    """
    def extract_data(self):
        super().extract_data()
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_amp180_{qbn}'] = \
                s+f'.{trans_name}_amp180'
            params_dict[f'{trans_name}_sigma_{qbn}'] = \
                s+f'.{trans_name}_sigma'
            params_dict[f'{trans_name}_nr_sigma_{qbn}'] = \
                s+f'.{trans_name}_nr_sigma'
            params_dict[f'T1_{qbn}'] = s+f'.T1'
            params_dict[f'T2_{qbn}'] = s+f'.T2'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

        self.nr_pulses_pi = self.get_param_value('n_pulses_pi')
        if self.nr_pulses_pi is None:
            raise ValueError('Please specify nr_pulses_pi.')
        if not hasattr(self.nr_pulses_pi, '__iter__'):
            # it is a number: convert to dict
            self.nr_pulses_pi = {qbn: self.nr_pulses_pi
                                 for qbn in self.qb_names}
        self.fixed_scaling = self.get_param_value('fixed_scaling')
        if self.fixed_scaling is None:
            task_list = self.get_param_value('preprocessed_task_list')
            self.fixed_scaling = {}
            for qbn in self.qb_names:
                task = [t for t in task_list if t['qb']==qbn][0]
                self.fixed_scaling[qbn] = task.get('fixed_scaling')
        if not isinstance(self.fixed_scaling, dict):
            # it is a number or None: convert to dict
            self.fixed_scaling = {qbn: self.fixed_scaling
                                  for qbn in self.qb_names}
        # the fixed scaling used in the experiment is potentially corrected
        # from the ideal value. Ex: to calibrate 3pi/4, we apply groups of
        # [Xpi/4, X3pi/4] where the fixed scaling applies to the first pulse
        # and should ideally be pi/4 but in practice it is usually pi/4-eps
        # to correct for small miscalibrations. However, in the analysis,
        # we need to take the correct theoretical scaling (pi/4) for the
        # calculation done in the fit, where only the amplitude scaling of
        # the second pulse (X3pi/4) is varied.
        for qbn in self.qb_names:
            if self.fixed_scaling[qbn] is not None:
                self.fixed_scaling[qbn] = 1/self.nr_pulses_pi[qbn]

        self.ideal_scalings = {}
        for qbn in self.qb_names:
            if self.fixed_scaling[qbn] is None:
                self.ideal_scalings[qbn] = 1/self.nr_pulses_pi[qbn]
            else:
                self.ideal_scalings[qbn] = 1 - 1/self.nr_pulses_pi[qbn]

    @staticmethod
    def apply_gate(y, z, ang_scaling, t_gate, gamma_1, gamma_phi, zth=1):
        """
        Calculates the time evolution of the y and z components of the qubit
        state vector under the application of an X gate described by the
        time-independent Hamiltonian (ang_scaling*pi/t_gate)*sigma_x/2.
        https://arxiv.org/src/1711.01208v2/anc/Supmat-Ficheux.pdf

        This function is used in sim_func when fixed_scaling is not None.

        Args:
            y (float): y coordinate of the qubit state vector at the
                start of the evolution
            z (float): z coordinate of the qubit state vector at the
                start of the evolution
            ang_scaling (float or array): fraction of a pi rotation
                (see Hamiltonian above)
            t_gate (float): gate length (s)
            gamma_1 (float): qubit energy relaxation rate
            gamma_phi (float): qubit dephasing rate
            zth (float; default=1): z coordinate at equilibrium

        Returns
            y, z: coordinates of the qubit state vector after the evolution
        """
        Omega = ang_scaling*np.pi/t_gate
        f_rabi = np.sqrt(Omega**2 - (1/16)*(gamma_1-2*gamma_phi)**2)
        yinf = 2*zth*Omega*gamma_1 / \
               (gamma_1*(gamma_1 + 2*gamma_phi) + 2*Omega**2)
        zinf = zth*gamma_1*(gamma_1 + 2*gamma_phi) / \
               (gamma_1*(gamma_1 + 2*gamma_phi) + 2*Omega**2)

        y_t = yinf + np.exp(-(3*gamma_1 + 2*gamma_phi)*t_gate/4) * \
              ((y - yinf)*(np.cos(f_rabi*t_gate) + np.sin(f_rabi*t_gate) *
                           (gamma_1 - 2*gamma_phi)/(4*f_rabi)) +
               (z - zinf)*np.sin(f_rabi*t_gate)*Omega/f_rabi)
        z_t = zinf + np.exp(-(3*gamma_1 + 2*gamma_phi)*t_gate/4) * \
              ((z - zinf)*(np.cos(f_rabi*t_gate) - np.sin(f_rabi*t_gate) *
                           (gamma_1 - 2*gamma_phi)/(4*f_rabi)) -
               (y - yinf)*np.sin(f_rabi*t_gate)*Omega/f_rabi)
        return y_t, z_t

    @staticmethod
    def apply_gate_mtx(y, z, ang_scaling, t_gate, gamma_1, gamma_phi, nreps=1):
        """
        Calculates the time evolution of the y and z components of the qubit
        state vector under the application of an X gate described by the
        time-independent Hamiltonian (ang_scaling*pi/t_gate)*sigma_x/2.
        https://arxiv.org/src/1711.01208v2/anc/Supmat-Ficheux.pdf
        This function implements the matrix version of apply_gate: y and z
        here are y - yinf and z - zinf in apply_gate

        This function is used in sim_func when fixed_scaling is None.

        Args:
            y (float): scaled y coordinate of the qubit state vector at the
                start of the evolution
            z (float): scaled z coordinate of the qubit state vector at the
                start of the evolution
            ang_scaling (float or array): fraction of a pi rotation
                (see Hamiltonian above)
            t_gate (float): gate length (s)
            gamma_1 (float): qubit energy relaxation rate
            gamma_phi (float): qubit dephasing rate
            nreps (int; default: 1): number of times the gate is applied

        Returns
            y, z: scaled coordinates of the qubit state vector after the
                evolution
        """
        Omega = ang_scaling * np.pi / t_gate
        f_rabi = np.sqrt(Omega ** 2 - (1 / 16) * (gamma_1 - 2 * gamma_phi) ** 2)
        prefactor = np.exp(-(3 * gamma_1 + 2 * gamma_phi) * t_gate / 4)
        mtx = prefactor * np.array([
            [np.cos(f_rabi * t_gate) + np.sin(f_rabi * t_gate) * \
                (gamma_1 - 2 * gamma_phi) / (4 * f_rabi),
             np.sin(f_rabi * t_gate) * Omega / f_rabi],
            [-np.sin(f_rabi * t_gate) * Omega / f_rabi,
             np.cos(f_rabi * t_gate) - np.sin(f_rabi * t_gate) * \
                (gamma_1 - 2 * gamma_phi) / (4 * f_rabi)]])
        mtx = np.linalg.matrix_power(mtx, nreps)
        res = mtx @ np.array([[y], [z]])
        return res[0][0], res[1][0]

    @staticmethod
    def sim_func(nr_pi_pulses, sc_error, ideal_scaling,
                 T2, t2_r=1, nr_pulses_pi=None,
                 y0=0, z0=1, zth=1, fixed_scaling=None,
                 T1=None, t_gate=None, mobjn=None, ts=None):
        """
        Simulation function for the excited qubit state populations for a trace
        of the N-pulse calibration experiment:
            - X90 - [ repeated groups of pulses ]^nr_pi_pulses -

        The repeated groups of pulses are either:
         - nr_pulses_pi x R(pi/nr_pulses_pi)
         or
         - R(fixed_scaling*pi)-R(pi-pi/nr_pulses_pi)
            if fixed_scaling is not None

         See also the docstring of the measurement class DriveAmpCalib.

        Args:
            nr_pi_pulses (array): number of repeated pulses applied to the qubit
                prepared in the + state
            sc_error (float or array): error(s) on the amplitude scaling
                away from the ideal scaling. This error will be fitted
            ideal_scaling (float): ideal amplitude scaling factor
            T2 (float): qubit decoherence time in seconds to be used as a guess
            t2_r (float; default=1): ratio T2_varied/T2. This ratio will be fitted
            nr_pulses_pi (int; default=None): the number of pulses that together
                implement a pi rotation.
            y0 (float; default=0): y coordinate of the initial state
            z0 (float; default=1): z coordinate of the initial state
            zth (float; default=1): z coordinate at equilibrium
            fixed_scaling (float; default: None): the amplitude scaling of
                the first rotation in the description above
            T1 (float; default: None): quit lifetime (s)
            t_gate (float): gate length (s)
            mobjn (str): name of the qubit
            ts (str): measurement timestamp
            The last two parameers will be used to extract T1/t_gate if the
            latter are not specified (see docstring of sim_func)

        Returns
            e_pops (array): same length as nr_pi_pulses and containing the
                qubit excited state populations after the application of
                nr_pi_pulses repeated groups of pulses
        """

        if any([v is None for v in [T1, T2, t_gate]]):
            assert mobjn is not None
            assert ts is not None
        if ts is not None:
            from pycqed.utilities.settings_manager import SettingsManager
            sm = SettingsManager()
        if t_gate is None:
            t_gate = sm.get_parameter(mobjn + '.ge_sigma', ts) * \
                     sm.get_parameter(mobjn + '.ge_nr_sigma', ts)
        if t_gate == 0:
            raise ValueError('Please specify t_gate.')
        if T1 is None:
            T1 = sm.get_parameter(mobjn + '.T1', ts)
        if T1 == 0:
            raise ValueError('Please specify T1.')

        T2 = t2_r * T2
        if nr_pulses_pi is None and fixed_scaling is None:
            raise ValueError('Please specify either nr_pulses_pi or '
                             'fixed_scaling.')

        gamma_1 = 1/T1
        gamma_2 = 1/T2
        gamma_phi = gamma_2 - 0.5*gamma_1

        # apply initial pi/2 gate
        y00, z00 = NPulseAmplitudeCalibAnalysis.apply_gate(
            y0, z0, 0.5, t_gate, gamma_1, gamma_phi, zth=zth)

        # calculate yinf, zinf with amp_sc
        amp_sc = sc_error + ideal_scaling
        if hasattr(amp_sc, '__iter__'):
            amp_sc = amp_sc[0]
        Omega = amp_sc * np.pi / t_gate
        yinf = 2 * zth * Omega * gamma_1 / \
               (gamma_1 * (gamma_1 + 2 * gamma_phi) + 2 * Omega ** 2)
        zinf = zth * gamma_1 * (gamma_1 + 2 * gamma_phi) / \
               (gamma_1 * (gamma_1 + 2 * gamma_phi) + 2 * Omega ** 2)

        e_pops = np.zeros(len(nr_pi_pulses))
        for i, n in enumerate(nr_pi_pulses):
            if fixed_scaling is None:
                # start in superposition state (after pi/2 pulse application)
                # redefine variables y -> y - yinf; z -> z - zinf;
                # adds offset to initial values
                y, z = y00 - yinf, z00 - zinf
                y, z = NPulseAmplitudeCalibAnalysis.apply_gate_mtx(
                    y, z, amp_sc, t_gate, gamma_1, gamma_phi,
                    nreps=nr_pulses_pi * n)
                # get back the true y and z
                y += yinf
                z += zinf
            else:
                y, z = y00, z00
                for j in range(n):
                    # apply pulse with varying scaling
                    y, z = NPulseAmplitudeCalibAnalysis.apply_gate(
                        y, z, amp_sc, t_gate, gamma_1, gamma_phi, zth=zth)
                    # apply pulse with fixed scaling
                    y, z = NPulseAmplitudeCalibAnalysis.apply_gate(
                        y, z, fixed_scaling, t_gate, gamma_1, gamma_phi, zth=zth)
            e_pops[i] = 0.5 * (1 - z)
        return e_pops

    @staticmethod
    def fit_trace(data_to_fit, nr_pi_pulses, amp_sc_err, T2,
                 ideal_scaling, maxeval=500, **kw):
        """
        Fit the excited state population as a function of nr_pi_pulses for the
        amplitude scalnig error amp_sc_err.

        Args:
            data_to_fit (array): excited state population measured after
                the application of the pulses in nr_pi_pulses
            nr_pi_pulses (array): number of repeated pulses applied to the qubit
                prepared in the + state
            amp_sc_err (float or array): error(s) on the amplitude scaling
                away from the ideal scaling. This error will be fitted
            T2 (float): qubit decoherence time (s)
            ideal_scaling (float): ideal amplitude scaling factor
            maxeval (int): maximum number of iterations

        Kwargs:
            - must contain either
                - T1 (float): quit lifetime (s)
                - t_gate (float): gate length (s)
                or
                - mobjn (str): name of the qubit
                - ts (str): measurement timestamp
                The last two will be used to extract the first two if T1/t_gate
                are not specified (see docstring of sim_func)
        """
        import nlopt
        fit_t2_r = kw.pop('fit_t2_r', True)
        def cost_func(vars, grad):
            sc_error = vars[0]
            t2_r = vars[1] if fit_t2_r else 1
            calc = NPulseAmplitudeCalibAnalysis.sim_func(
                nr_pi_pulses, sc_error, ideal_scaling, T2, t2_r, **kw)
            return np.sqrt(np.sum((calc - data_to_fit)**2))

        opt = nlopt.opt(nlopt.GN_DIRECT, 1 + fit_t2_r)
        opt.set_xtol_rel(1e-8)
        opt.set_maxeval(maxeval)
        opt.set_min_objective(cost_func)
        opt.set_lower_bounds([-0.05] + ([0.01] if fit_t2_r else []))
        opt.set_upper_bounds([0.1] + ([1.0] if fit_t2_r else []))
        x = opt.optimize([amp_sc_err] + ([1] if fit_t2_r else []))
        return x

    def run_fitting(self, **kw):
        fit_t2_r = self.get_param_value('fit_t2_r', True)
        nr_pi_p = self.proc_data_dict['sweep_points_dict'][self.qb_names[0]][
            'msmt_sweep_points']
        sp2dd = self.proc_data_dict['sweep_points_2D_dict'][self.qb_names[0]]
        len_sp2dd = len(sp2dd[list(sp2dd)[0]])
        # check if the fit results were pass by the user
        self.fitted_amp_sc_errs = self.get_param_value('fitted_scaling_errors')
        run_fit = False
        if self.fitted_amp_sc_errs is None:
            run_fit = True
            # store fit results {qbn: len_nr_soft_sp}
            self.fitted_amp_sc_errs = {qbn: np.zeros(len_sp2dd)
                                       for qbn in self.qb_names}
        # store fit results T1, T2 {qbn: len_nr_soft_sp}
        # FIXME: stores fit results as {qbn: {soft_sp_index: {'Tx': val, ...}}}
        #  while other analyses typically store as {qbn: {'Tx': [vals], ...}}.
        #  Also, it probably does not make sense to index a dict with integers.
        self.fitted_coh_times = {qbn: {i: '' for i in range(len_sp2dd)}
                                 for qbn in self.qb_names}
        # store fit lines {qbn: 2D array (each row is a fit line)}
        self.fit_lines = {qbn:  np.zeros((len_sp2dd, len(nr_pi_p)))
                          for qbn in self.qb_names}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            T1 = self.get_param_value(
                'T1', default_value=self.raw_data_dict[f'T1_{qbn}'])
            T2 = self.get_param_value(
                'T2', default_value=self.raw_data_dict[f'T2_{qbn}'])
            t_gate = self.raw_data_dict[f'{trans_name}_sigma_{qbn}'] * \
                     self.raw_data_dict[f'{trans_name}_nr_sigma_{qbn}']
            data = self.proc_data_dict['data_to_fit'][qbn]
            idx = data.shape[1] - self.num_cal_points
            data = data[:, :idx]
            nr_pi_pulses = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            sp2dd = self.proc_data_dict['sweep_points_2D_dict'][qbn]
            amp_scalings = sp2dd[list(sp2dd)[0]]
            for i, amp_sc in enumerate(amp_scalings):
                if run_fit:
                    maxeval = 5000 if self.fixed_scaling[qbn] is None else 500
                    introduced_amp_sc_err = amp_sc - self.ideal_scalings[qbn]
                    opt_res = NPulseAmplitudeCalibAnalysis.fit_trace(
                            data[i, :], nr_pi_pulses,
                            introduced_amp_sc_err, T2, self.ideal_scalings[qbn],
                            nr_pulses_pi=self.nr_pulses_pi[qbn],
                            fixed_scaling=self.fixed_scaling[qbn],
                            T1=T1, t_gate=t_gate, fit_t2_r=fit_t2_r,
                            maxeval=self.get_param_value('maxeval', maxeval))
                    self.opt_res = opt_res
                    amp_sc_err = opt_res[0]
                    t2_r = 1
                    self.fitted_amp_sc_errs[qbn][i] = amp_sc_err
                    if fit_t2_r:
                        t2_r = opt_res[1]
                        self.fitted_coh_times[qbn][i] = {'T2 ratio': t2_r,
                                                         'T2 fit': t2_r*T2,
                                                         'T2 guess': T2}
                    self.fit_lines[qbn][i, :] = \
                        NPulseAmplitudeCalibAnalysis.sim_func(
                            nr_pi_pulses, amp_sc_err, T2=T2, t2_r=t2_r,
                            ideal_scaling=self.ideal_scalings[qbn],
                            nr_pulses_pi=self.nr_pulses_pi[qbn],
                            fixed_scaling=self.fixed_scaling[qbn],
                            T1=T1, t_gate=t_gate)

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict(
            {qbn: {} for qbn in self.qb_names})
        for qbn in self.qb_names:
            fit_sc_errs = self.fitted_amp_sc_errs[qbn]
            sp2dd = self.proc_data_dict['sweep_points_2D_dict'][qbn]
            amp_scalings = sp2dd[list(sp2dd)[0]]
            ideal_sc = self.ideal_scalings[qbn]
            # calculate the correction that must be applied to the scaling
            # used for this experiment.
            # fit_sc_errs is the scaling error away from the ideal scaling
            # (ex: 0.5 for pi/2). But we have nonlinearity, so the scaling
            # used in this measurement (amp_scalings), which might already have
            # been corrected once, is the closes scaling that gives the
            # intended rotation in practice. I.e. our model fits
            # (ideal_sc + sc_err)pi/tau, but ideal scaling is in practice
            # the one used in the measurement. So the correct scaling that
            # this analysis must report is not ideal_sc - sc_err, but
            # amp_scalings - sc_err.
            sc_corrs = fit_sc_errs - (amp_scalings - ideal_sc)

            self.proc_data_dict['analysis_params_dict'][qbn][
                'fitted_scaling_errors'] = fit_sc_errs
            self.proc_data_dict['analysis_params_dict'][qbn][
                'scaling_corrections'] = sc_corrs
            self.proc_data_dict['analysis_params_dict'][qbn][
                'scaling_corrections_mean'] = np.mean(sc_corrs)
            self.proc_data_dict['analysis_params_dict'][qbn][
                'correct_scalings'] = ideal_sc - sc_corrs
            self.proc_data_dict['analysis_params_dict'][qbn][
                'correct_scalings_mean'] = ideal_sc - np.mean(sc_corrs)
            self.proc_data_dict['analysis_params_dict'][qbn][
                'fitted_coh_times'] = self.fitted_coh_times[qbn]

            trans_name = self.get_transition_name(qbn)
            amp180 = self.raw_data_dict[f'{trans_name}_amp180_{qbn}']
            self.proc_data_dict['analysis_params_dict'][qbn][
                'set_amplitude'] = ideal_sc*amp180
            self.proc_data_dict['analysis_params_dict'][qbn][
                'correct_amplitude'] = (ideal_sc - np.mean(sc_corrs))*amp180

        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()
        if self.do_fitting:
            for qbn in self.qb_names:
                data_pe = self.proc_data_dict['data_to_fit'][qbn]
                nr_pi_pulses = self.proc_data_dict['sweep_points_dict'][
                    qbn]['msmt_sweep_points']
                sp1d = self.proc_data_dict['sweep_points_dict'][
                    qbn]['sweep_points']
                sp2dd = self.proc_data_dict['sweep_points_2D_dict'][qbn]
                amp_scalings = sp2dd[list(sp2dd)[0]]
                for i, amp_sc in enumerate(amp_scalings):
                    base_plot_name = f'DriveAmpCalib_{qbn}_' \
                                     f'{self.data_to_fit[qbn]}_idx{i}_' \
                                     f'{amp_sc:.4f}'
                    # data
                    self.prepare_projected_data_plot(
                        fig_name=base_plot_name,
                        data=data_pe[i, :],
                        title_suffix=f'set_amp_sc={amp_sc:.4f}',
                        plot_name_suffix=f'{qbn}_data',
                        data_label='Data',
                        linestyle='--',
                        TwoD=False,
                        qb_name=qbn)

                    # line at 0.5: start with this so it has the lowest zorder
                    self.plot_dicts[f'0.5_hline_{qbn}_{i}'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_hlines,
                        'y': [0.5],
                        'xmin': sp1d[0],
                        'xmax': sp1d[-1],
                        'linestyle': '--',
                        'colors': 'gray'}

                    # fit
                    fit_line = self.fit_lines[qbn][i, :]
                    self.plot_dicts[f'fit_{qbn}_{i}'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_line,
                        'xvals': nr_pi_pulses,
                        'yvals': fit_line,
                        'setlabel': 'fit',
                        'marker': None,
                        'do_legend': True,
                        'color': 'r',
                        'legend_ncol': 2,
                        'legend_bbox_to_anchor': (1, -0.15),
                        'legend_pos': 'upper right'}

                    # textbox
                    trans_name = self.get_transition_name(qbn)
                    amp180 = self.raw_data_dict[f'{trans_name}_amp180_{qbn}']
                    ana_d = self.proc_data_dict['analysis_params_dict'][qbn]
                    corr_sc = ana_d["correct_scalings_mean"]
                    sc_corr = ana_d["scaling_corrections_mean"]
                    ideal_sc = self.ideal_scalings[qbn]
                    textstr = f'Corrected scaling: {corr_sc:.7f}\n'
                    textstr += f'Used scaling: {amp_sc:.6f}\n'
                    textstr += f'Ideal scaling: {ideal_sc:.3f}\n'
                    textstr += f'Scaling corr. fit: {sc_corr:.7f}\n'
                    # correction to amp used in measurement
                    da = (corr_sc - amp_sc) * amp180
                    # correction to the amplitude calculated with ideal scaling
                    da_ideal = (corr_sc - ideal_sc) * amp180
                    textstr += f'Diff. Amp. fit - Amp. set: {da*1000:.3f} mV\n'
                    textstr += 'Diff. Amp. fit - Amp. ideal: ' + \
                               f'{da_ideal*1000:.3f} mV'
                    if self.get_param_value('fit_t2_r', True):
                        t2f = self.fitted_coh_times[qbn][i]['T2 fit']
                        t2s = self.get_param_value(
                            'T2', default_value=self.raw_data_dict[f'T2_{qbn}'])
                        textstr += f'\nT2 fit: {t2f*1e6:.1f} $\mu$s; ' \
                                   f'T2 meas: {t2s*1e6:.1f} $\mu$s'
                    self.plot_dicts[f'text_msg_{qbn}_{i}'] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_text,
                        'ypos': -0.2,
                        'xpos': -0.05,
                        'horizontalalignment': 'left',
                        'verticalalignment': 'top',
                        'text_string': textstr}

                    # pf without cal points
                    if 'pf' in self.proc_data_dict['projected_data_dict'][qbn]:
                        data_pf = self.proc_data_dict[
                            'projected_data_dict'][qbn]['pf']
                        # remove cal points from data_pf
                        idx = data_pf.shape[1] - self.num_cal_points
                        data_pf = data_pf[:, :idx]
                        leak_plot_name = f'Leakage_{qbn}_pf_{i}_{amp_sc:.4f}'
                        data_axis_label = self.get_yaxis_label(qbn, 'pf')
                        self.prepare_projected_data_plot(
                            fig_name=leak_plot_name,
                            data=data_pf[i, :],
                            sweep_points=nr_pi_pulses,
                            data_axis_label=data_axis_label,
                            title_suffix=f'amp_sc={amp_sc:.4f}',
                            do_legend_data=False,
                            linestyle='-',
                            plot_cal_points=False,
                            TwoD=False,
                            qb_name=qbn)


class DriveAmplitudeNonlinearityCurveAnalysis(ba.BaseDataAnalysis):
    """
    Class to analyse an experiment run with the DriveAmpNonlinearityCurve, or a
    set of DriveAmpCalib experiments for measuring the drive amplitude
    non-linearity curve to calibrate the non-linearity produced by the control
    electronics.

    Args:
        t_start (str): timestamp of the first measurement
        t_stop (str): timestamp of the last measurement; all timestamps
            between t_start and t_stop will be taken
        qb_names (list of str): names of the qubits in the experiment
        For do_fitting and auto, see docstring of parent class.

    Keyword args:
        Passed to the parent class.

        Specific to this class:
            - fit_mask (array of bools; default: all True): specified whether to
                exclude any points of the non-linearity curve from the fit.
                Can be passed in options_dict or metadata.
    """

    def __init__(self, t_start, t_stop, qb_names=None, do_fitting=True,
                 auto=True, **kwargs):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label='Drive_amp_calib', do_fitting=do_fitting,
                         **kwargs)
        self.qb_names = qb_names
        self.auto = auto

        if self.auto:
            self.run_analysis()

    def extract_data(self):
        """
        Extracts the nonlinearity_curve for each qubit from the individual
        DriveAmpCalib measurements specified by self.timestamps.
        First tries to extract analysis parameters from the data file. If
        not found, the NPulseAmplitudeCalibAnalysis will be run for that
        timestamp.

        First sorts the timestamps in ascending order of the amplitude scaling
        that was calibrated in the experiment.

        Stores the nonlinearity curves in self.proc_data_dict and saves them
        to file.
        """
        super().extract_data()

        if self.qb_names is None:
            self.qb_names = self.get_param_value(
                'ro_qubits', default_value=self.get_param_value('qb_names'))
            if self.qb_names is None:
                raise ValueError('Provide the "qb_names."')

        # sort timestamps by measured amplitude scaling factor (given by
        # 1/n_pulses pi and fixed_scaling; see docstring of measurement class
        # DriveAmpCalib.)
        self.sorted_tss = []
        ideal_scalings = []
        for i, ts in enumerate(self.timestamps):
            # assumes n_pulses_pi is the same for all qubits
            n_pulses_pi = self.get_param_value('n_pulses_pi', index=i)
            range_center = 1 / n_pulses_pi
            fixed_scaling = self.get_param_value('fixed_scaling', index=i)
            if fixed_scaling is None:
                task_list = self.get_param_value('preprocessed_task_list', index=i)
                fixed_scaling = {}
                for qbn in self.qb_names:
                    task = [t for t in task_list if t['qb'] == qbn][0]
                    fixed_scaling[qbn] = task.get('fixed_scaling')
            if not isinstance(fixed_scaling, dict):
                # it is a number or None: convert to dict
                fixed_scaling = {qbn: fixed_scaling for qbn in self.qb_names}
            if any([fs is not None for fs in list(fixed_scaling.values())]):
                range_center = 1 - fixed_scaling[self.qb_names[0]]
                ideal_scalings += [1 - 1 / n_pulses_pi]
            else:
                ideal_scalings += [range_center]
            self.sorted_tss += [(ts, range_center)]
        self.sorted_tss.sort(key=lambda t: t[1])
        ideal_scalings.sort()

        nonlinearity_curves = {qbn: {label: np.zeros(len(self.sorted_tss))
                              for label in ['set_scalings',
                                            'corrected_scalings',
                                            'set_amps', 'corrected_amps']}
                               for qbn in self.qb_names}

        trans_name = self.get_param_value('transition_name')
        for ii, tup in enumerate(self.sorted_tss):
            ts = tup[0]
            idx = self.timestamps.index(ts)
            task_list = self.get_param_value('preprocessed_task_list', index=idx)
            # get corrected scalings
            params_dict = {qbn: f'Analysis.Processed data.'
                                f'analysis_params_dict.{qbn}.'
                                f'correct_scalings_mean'
                           for qbn in self.qb_names}
            corr_sc = self.get_data_from_timestamp_list(params_dict,
                                                        timestamps=[ts])

            # Check if the corrected scalings were found in the data file
            run_ana = any([
                hasattr(corr_sc[qbn], '__iter__') and len(corr_sc[qbn]) == 0
                for qbn in self.qb_names])
            if run_ana:
                # NPulseAmplitudeCalibAnalysis has not been run. Run it here
                log.info(f'NPulseAmplitudeCalibAnalysis was not run for {ts} '
                         f'(amplitude scaling of {ideal_scalings[ii]}). '
                         f'Running it now ... ')
                a = NPulseAmplitudeCalibAnalysis(
                    t_start=ts, options_dict=self.options_dict)
                for qbn in self.qb_names:
                    nonlinearity_curves[qbn]['corrected_scalings'][ii] = \
                        a.proc_data_dict['analysis_params_dict'][qbn][
                            'correct_scalings_mean']
            else:
                # corrected scalings were found in the data file
                for qbn in self.qb_names:
                    nonlinearity_curves[qbn]['corrected_scalings'][ii] = \
                        corr_sc[qbn]

            # get the ideal scalings, and the set and corrected amps
            for qbn in self.qb_names:
                # take the qubit amp180 from the data file
                task = [t for t in task_list if t['qb'] == qbn][0]
                trans_name = task.get('transition_name_input', trans_name)
                s = 'Instrument settings.' + qbn
                params_dict = {'amp180': s + f'.{trans_name}_amp180'}
                amp180 = self.get_data_from_timestamp_list(
                    params_dict, timestamps=[ts])['amp180']
                # ideal scaling
                nonlinearity_curves[qbn]['set_scalings'][ii] = \
                    ideal_scalings[ii]
                # convert set_scaling and corrected_scaling to amps
                nonlinearity_curves[qbn]['set_amps'][ii] = \
                    nonlinearity_curves[qbn]['set_scalings'][ii] * amp180
                nonlinearity_curves[qbn]['corrected_amps'][ii] = \
                    nonlinearity_curves[qbn]['corrected_scalings'][ii] * amp180

        for qbn in self.qb_names:
            # add the point (1,1) is not part of the dataset
            if nonlinearity_curves[qbn]['set_scalings'][-1] != 1:
                nonlinearity_curves[qbn]['set_scalings'] = np.concatenate([
                    nonlinearity_curves[qbn]['set_scalings'], [1]])
                nonlinearity_curves[qbn]['corrected_scalings'] = np.concatenate([
                    nonlinearity_curves[qbn]['corrected_scalings'], [1]])

        # store nonlinearity_curves in proc_data_dict
        self.proc_data_dict['nonlinearity_curves'] = nonlinearity_curves
        # save proc_data_dict to file
        self.save_processed_data()

    def prepare_fitting(self):
        """
        Prepare the fit_dict for fitting the non-linearity curves to a
        fifth-order odd polynomial with two free coefficients 'a' and 'b.'
        """
        nonlinearity_curves = self.proc_data_dict['nonlinearity_curves']
        mask = self.get_param_value('fit_mask', None)
        if mask is None:
            length = len(nonlinearity_curves[self.qb_names[0]]['set_scalings'])
            mask = {qbn: np.ones(length, dtype=np.bool_) for qbn in self.qb_names}
        if not isinstance(mask, dict):
            mask = {qbn: mask for qbn in self.qb_names}

        fit_func_poly = lambda x, a, b, c: x * (a * (x ** 4 - 1) +
                                                b * (x ** 2 - 1) + c)
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            model = lmfit.Model(fit_func_poly)
            model.set_param_hint('a', value=0.008, vary=True)
            model.set_param_hint('b', value=0.008, vary=True)
            model.set_param_hint('c', value=1, vary=False)
            guess_pars = model.make_params()
            self.set_user_guess_pars(guess_pars)
            x = nonlinearity_curves[qbn]['set_scalings'][mask[qbn]]
            y = nonlinearity_curves[qbn]['corrected_scalings'][mask[qbn]]
            if x[-1] == 1:
                y[-1] = 1
            self.fit_dicts[f'fit_scalings_{qbn}'] = {
                'fit_fn': fit_func_poly,
                'fit_xvals': {'x': x},
                'fit_yvals': {'data': y},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        """
        Extract the fitted coefficients 'a' and 'b.' store them in
        proc_data_dict under the key "nonlinearity_fit_pars," and save them to
        file.
        """
        # If we store the results in the usual way under
        # self.proc_data_dict['analysis_params_dict'], saving this entry will
        # overwrite the existing entry in self.timestamps[0] which came from
        # the NPulseAmplitudeCalibAnalysis. So we must use a different key.
        self.proc_data_dict['nonlinearity_fit_pars'] = OrderedDict()
        for k, fit_dict in self.fit_dicts.items():
            qbn = k.split('_')[-1]
            fit_res = fit_dict['fit_res']
            self.proc_data_dict['nonlinearity_fit_pars'][qbn] = \
                {'a': fit_res.best_values['a'], 'b': fit_res.best_values['b']}
            self.save_processed_data(key='nonlinearity_fit_pars')

    def prepare_plots(self):
        nonlinearity_curves = self.proc_data_dict['nonlinearity_curves']
        # Figure with 2 rows of axes
        plotsize = self.get_default_plot_params(set_pars=False)['figure.figsize']
        numplotsx, numplotsy = 1, 2
        for qbn in self.qb_names:
            fig_title = f'Non-linearity curve {qbn}:\n' \
                        f'{self.timestamps[0]} - {self.timestamps[-1]}'
            figname = f'Nonlinearity_curve_{qbn}'

            xvals = nonlinearity_curves[qbn]['set_scalings']
            yvals = nonlinearity_curves[qbn]['corrected_scalings']
            if xvals[-1] == 1:
                yvals[-1] = 1

            # Plot line interpolated between (0, 0) and (1, 1)
            # we plot this before the data in order to have the data on top of
            # the line in the plot
            x_with_zero = np.concatenate([[0], xvals])
            y_with_zero = np.concatenate([[0], yvals])
            line = x_with_zero * max(y_with_zero) / max(x_with_zero)
            self.plot_dicts[f'{figname}_data_line'] = {
                'fig_id': figname,
                'ax_id': 0,
                'plotfn': self.plot_line,
                'xvals': x_with_zero,
                'xlabel': 'Set amp. scaling',
                'xunit': '',
                'xrange': [-0.05, 1.05],
                'yvals': line,
                'ylabel': 'Corrected amp. scaling',
                'yunit': '',
                'setlabel': 'linear',
                'marker': '',
                'line_kws': {'color': 'k', 'linewidth': 1},
                'numplotsx': numplotsx,
                'numplotsy': numplotsy,
                'plotsize': (plotsize[0] * 0.65, plotsize[1] * 1.5),
                'title': fig_title
                }

            # plot data
            self.plot_dicts[f'{figname}_data'] = {
                'fig_id': figname,
                'ax_id': 0,
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': 'Set amp. scaling',
                'xunit': '',
                'yvals': yvals,
                'ylabel': 'Corrected amp. scaling',
                'yunit': '',
                'setlabel': 'data',
                'linestyle': 'none'}

            # Calculate and plot nonlinearity: difference between data and line
            # interpolated between (0, 0) and (1, 1)
            nonlinearity = line[1:] - yvals
            self.plot_dicts[f'{figname}_data_diff'] = {
                'fig_id': figname,
                'ax_id': 1,
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': 'Set amp. scaling',
                'xunit': '',
                'xrange': [-0.05, 1.05],
                'yvals': nonlinearity,
                'ylabel': 'Diff. to linear',
                'yunit': '',
                'linestyle': 'none'}

            if self.do_fitting:
                fit_res = self.fit_dicts[f'fit_scalings_{qbn}']['fit_res']

                # plot fit
                self.plot_dicts[f'{figname}_fit'] = {
                    'fig_id': figname,
                    'ax_id': 0,
                    'plotfn': self.plot_fit,
                    'fit_res': fit_res,
                    'setlabel': 'poly fit',
                    'color': 'C0',
                    'do_legend': True,
                    # 'legend_ncol': 2,
                    # 'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper left'}

                # plot difference between fit and linear
                xfine = np.linspace(0, x_with_zero[-1], 100)
                line_fine = xfine * max(y_with_zero) / max(x_with_zero)
                poly_line = fit_res.model.func(xfine, **fit_res.best_values)
                self.plot_dicts[f'{figname}_fit_diff'] = {
                    'fig_id': figname,
                    'ax_id': 1,
                    'plotfn': self.plot_line,
                    'xvals': xfine,
                    'yvals': line_fine-poly_line,
                    'color': 'C0',
                    'marker': ''}


class ChevronAnalysis(MultiQubit_TimeDomain_Analysis):
    """ Class for automated chevron analysis

    For this function a model for the Hamiltonian of the qubits is required. The function will use the fit_ge_from_...
    parameters to calculate the frequencies for the given amplitudes.
    Saves the fit parameters and the optimal CZ time in self.proc_data_dict['analysis_params_dict']:
        J: coupling between |20> and |11> in Hz
        offset_freq: frequency by which the two energy levels were off from the calculation in Hz
    Options_dict options:
        'num_curves': number of curves of full state recovery that should be plotted using the fit results, default: 5
        'model': model used to calculate voltage to qubit frequency
        'J_guess_boundary_scale': the limits for the fit of J are: (default: 2)
            [J_fft/J_guess_boundary_scale, J_fft*J_guess_boundary_scale], where J_fft is obtained by the function J_fft
        'offset_guess_boundary': boundaries for the offset_freq fit in Hz (default: 2e8)
        'steps': number of evaulations used for the fitting, default 1e8, dual_annealing default 1e7
    """
    def extract_data(self):
        super().extract_data()
        self.task_list = self.get_param_value('task_list')
        self.qb_names = self.get_param_value('qb_names', self.get_qbs_from_task_list(self.task_list))
        # FIXME: refactor to use settings manager instead of raw_data_dict
        params = ['ge_freq', 'fit_ge_freq_from_dc_offset', 'fit_ge_freq_from_flux_pulse_amp',
                  'flux_amplitude_bias_ratio', 'flux_parking', 'anharmonicity']
        for qbn in self.qb_names:
            params_dict = {f"{p}_{qbn}": f'Instrument settings.{qbn}.{p}'
                           for p in params}
            self.raw_data_dict.update(
                self.get_data_from_timestamp_list(params_dict))


    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        self.proc_data_dict['Delta'] = OrderedDict()

        def pe_function(t, Delta, J=10e6, offset_freq=0):
            # From Nathan's master's thesis Eq. 2.6 - fitting function
            t = t*1e9
            J = 2*np.pi*J/1e9
            offset_freq = offset_freq/1e9
            Delta = Delta/1e9
            Delta_off = 2 * np.pi * (
                    Delta + offset_freq)  # multiplied with 2pi because needs to be in angular frequency,
            return (Delta_off ** 2 + 2 * J ** 2 * (np.cos(t * np.sqrt(4 * J ** 2 + Delta_off ** 2)) + 1)) / (
                    4 * J ** 2 + Delta_off ** 2) # J is already in angular frequency (see J_fft)

        def J_fft(t, Delta, data):
            # Fourier transform for all Delta. Smallest period corresponds to J (assuming actual Delta=0 in that case)
            # Adopted from https://docs.scipy.org/doc/scipy/tutorial/fft.html
            J_min = None
            for Delta_index in range(len(Delta)):
                S = sp.fft.fft(data[Delta_index])
                S[0] = 0 # Ignore DC component
                xf = sp.fft.fftfreq(len(t), (t.max()-t.min())/len(t))[0:len(t)//2]
                J_fft = xf[np.abs(S[0:len(t)//2]).argmax()]/2 # 2J = 2pi/T (see thesis), but we want it in Hz
                if J_min == None or J_min > J_fft:
                    J_min = J_fft
            return J_min

        def calculate_voltage_from_flux(vfc, flux):
            return vfc['dac_sweet_spot'] + vfc['V_per_phi0'] * flux

        def calculate_qubit_frequency(flux_amplitude_bias_ratio, amplitude, vfc,
                                      model='transmon_res', bias = None):
            # TODO: possibly regroup this function as well as the one in qudev transmon into a separate module that
            #  can be called both from the measurement and analysis side, to avoid code dupplication
            if flux_amplitude_bias_ratio is None:
                if np.ndim(amplitude) > 0:
                    if ((model in ['transmon', 'transmon_res'] and amplitude.any() != 0) or
                            (model == ['approx'] and bias is not None and bias != 0)):
                        raise ValueError('flux_amplitude_bias_ratio is None, but is '
                                         'required for this calculation.')
                else:
                    if ((model in ['transmon', 'transmon_res'] and amplitude != 0) or
                            (model == ['approx'] and bias is not None and bias != 0)):
                        raise ValueError('flux_amplitude_bias_ratio is None, but is '
                                         'required for this calculation.')


            if model == 'approx':
                ge_freq = fit_mods.Qubit_dac_to_freq(
                    amplitude + (0 if bias is None or np.all(bias == 0) else
                                 bias * flux_amplitude_bias_ratio), **vfc)
            elif model == 'transmon':
                kw = deepcopy(vfc)
                kw.pop('coupling', None)
                kw.pop('fr', None)
                ge_freq = fit_mods.Qubit_dac_to_freq_precise(bias + (
                    0 if np.all(amplitude == 0)
                    else amplitude / flux_amplitude_bias_ratio), **kw)
            elif model == 'transmon_res':
                ge_freq = fit_mods.Qubit_dac_to_freq_res(bias + (
                    0 if np.all(amplitude == 0)
                    else amplitude / flux_amplitude_bias_ratio), **vfc)
            else:
                raise NotImplementedError(
                    "Currently, only the models 'approx', 'transmon', and"
                    "'transmon_res' are implemented.")
            return ge_freq

        def add_fit_dict(qbH_name, qbL_name, data, key):
            """ Creates the dictionary used for fitting

             The dictionary includes the fitting-model, the function be fitted to, the x and y data, the method used for
             fitting and the guess parameters. First extracts the relevant data from the raw_data_dict.
             Converts the amplitude to frequencies and the swept frequency (amplitude) to a detuning (Delta) of the
             qubits. Additionally removes all data points with NaN excited state population. The fit parameters are
             the coupling of the qubits J and the frequency offset of the actual detuning compared to what was
             calculated. J is obtained from doing a FFT on all measured detunings and taking the minimum of it, which
             corresponds to the smallest detuning (ideally 0). Finally flattens the data for the fit.


            Parameters
            ----------
            qbH_name: Higher-frequency qubit
            qbL_name: Lower-frequency qubit
            data: excited state population
            key: key for the dictionary

            Returns
            -------
            A fit-dictionary
            """
            model = self.get_param_value('model', 'transmon_res')
            J_guess_boundary_scale = self.get_param_value('guess_paramater_scale', 2)
            offset_guess_boundary = self.get_param_value('offset_guess_boundary', 2e8)
            hdf_file_index = self.get_param_value('hdf_file_index', 0)

            qbH_flux_amplitude_bias_ratio = self.raw_data_dict[f'flux_amplitude_bias_ratio_{qbH_name}']
            qbL_flux_amplitude_bias_ratio = self.raw_data_dict[f'flux_amplitude_bias_ratio_{qbL_name}']
            qbH_flux = self.raw_data_dict[f'flux_parking_{qbH_name}']
            qbL_flux = self.raw_data_dict[f'flux_parking_{qbL_name}']

            if model in ['transmon', 'transmon_res']:
                qbH_vfc = self.raw_data_dict[f'fit_ge_freq_from_dc_offset_{qbH_name}']
                qbL_vfc = self.raw_data_dict[f'fit_ge_freq_from_dc_offset_{qbL_name}']
                qbH_bias = calculate_voltage_from_flux(qbH_vfc, qbH_flux)
                qbL_bias = calculate_voltage_from_flux(qbL_vfc, qbL_flux)
            else:
                qbH_vfc = self.raw_data_dict[f'fit_ge_freq_from_flux_pulse_amp_{qbH_name}']
                qbL_vfc = self.raw_data_dict[f'fit_ge_freq_from_flux_pulse_amp_{qbL_name}']
                qbH_bias = None
                qbL_bias = None

            if self.num_cal_points != 0:
                data = np.array([element[:-self.num_cal_points] for element in data])

            t = self.proc_data_dict['sweep_points_dict'][qbH_name]['msmt_sweep_points']

            # Not sure according to which rule the qbs are ordered in the string, maybe high q.number to low
            sweep_point_name = qbH_name + '_' + qbL_name + '_amplitude2'
            if sweep_point_name in self.proc_data_dict['sweep_points_2D_dict'][qbL_name]:
                amp2 = self.proc_data_dict['sweep_points_2D_dict'][qbL_name][sweep_point_name]
            else:
                sweep_point_name = qbL_name + '_' + qbH_name + '_amplitude2'
                amp2 = self.proc_data_dict['sweep_points_2D_dict'][qbL_name][sweep_point_name]

            cz_name = self.get_param_value("exp_metadata")["cz_pulse_name"]
            device_name = self.get_param_value('device_name')
            if device_name is None:
                try:
                    device_name = self.get_instruments_by_class(
                        'pycqed.instrument_drivers.meta_instrument.device.Device',
                        hdf_file_index)[0]
                except KeyError:
                    raise KeyError('For old data, the device name has to be given as an input "device_name" ')

            path = f"{device_name}.{cz_name}_{qbH_name}_{qbL_name}_amplitude"
            amp = self.get_instrument_setting(path)
            qbL_tuned_freq_arr = calculate_qubit_frequency(
                flux_amplitude_bias_ratio=qbL_flux_amplitude_bias_ratio, amplitude=amp2,
                vfc=qbL_vfc, model= model, bias=qbL_bias)
            qbH_tuned_ef_freq = calculate_qubit_frequency(
                flux_amplitude_bias_ratio=qbH_flux_amplitude_bias_ratio, amplitude=amp,
                vfc=qbH_vfc, model= model, bias=qbH_bias) + self.raw_data_dict[f'anharmonicity_{qbH_name}']

            Delta = qbL_tuned_freq_arr - qbH_tuned_ef_freq

            reduction_arr = np.invert(np.isnan(data))
            indices = np.where(reduction_arr == False)
            reduction_arr[indices[0]] = False # All columns with the faulty fields get set false
            for element in reduction_arr:
                element[indices[1]] = False # All rows with the faulty fields get set false

            # Remove faulty fields
            t_mod = np.delete(t, indices[1])
            Delta_mod = np.delete(Delta, indices[0])
            pe = data[reduction_arr].reshape(len(Delta_mod), len(t_mod))

            pe_model = lmfit.Model(pe_function, independent_vars=['t', 'Delta'])

            J_guess = J_fft(t_mod, Delta_mod, pe)
            pe_model.set_param_hint('J', value=J_guess, min=J_guess/J_guess_boundary_scale,
                                    max=J_guess_boundary_scale*J_guess)
            pe_model.set_param_hint('offset_freq', value=0, min=-offset_guess_boundary,
                                    max=offset_guess_boundary)
            guess_pars = pe_model.make_params()
            self.set_user_guess_pars(guess_pars)

            # Data needs to be flat for fitting
            t_mod_flat = np.tile(t_mod, len(Delta_mod))
            Delta_mod_flat = np.repeat(Delta_mod, len(t_mod))
            pe_flat = pe.flatten()

            self.proc_data_dict['Delta'][qbH_name] = Delta


            self.fit_dicts[key] = {
                'model' : pe_model, # if this is missing, Model is None and Delta is assigned as parameter..
                'fit_fn': pe_model.func,
                'fit_xvals': {'t': t_mod_flat, 'Delta': Delta_mod_flat},
                'fit_yvals': {'data': pe_flat},
                'method': 'dual_annealing',
                'guess_pars': guess_pars,
                'steps': self.get_param_value('steps', 1e8), # default for dual annealing is 1e7
            }

        for task in self.get_param_value('task_list'):
            qbH_name, qbL_name = self._get_qbH_qbL(task)
            if qbH_name is not None:
                data = self.proc_data_dict['projected_data_dict'][qbH_name]['pe']
                add_fit_dict(qbH_name=qbH_name, qbL_name=qbL_name, data=data, key= f'chevron_fit_{qbH_name}_{qbL_name}')

    def _get_qbH_qbL(self, task):
        qbH_name, qbL_name = task['qbc'], task['qbt']

        if qbH_name not in self.qb_names or qbL_name not in self.qb_names:
            return None, None

        # Safety check on whether the above qbH/L assignment is correct
        if self.raw_data_dict[f'ge_freq_{qbH_name}'] < self.raw_data_dict[
                f'ge_freq_{qbL_name}']:
            qbH_name, qbL_name = qbL_name, qbH_name

        return qbH_name, qbL_name

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for k, fit_dict in self.fit_dicts.items():
            # k is of the form chevron_fit_qbH_qbL
            # Replace k with qbH_qbL
            k = k.replace('chevron_fit_', '')
            qbH = k.split('_')[0]
            fit_res = fit_dict['fit_res']
            self.proc_data_dict['analysis_params_dict'][k] = \
                fit_res.best_values
            self.proc_data_dict['analysis_params_dict'][k]['t_CZ_'+self.measurement_strings[qbH].partition('_')[0]] = \
                self.t_CARB(fit_res.best_values['J'], 0, 1) # could e.g. be changed to t_CZ_up with if statements
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        num_curves = self.options_dict.get('num_curves', 5)
        steps = 500 # How many plotting points you want for the fit

        if self.do_fitting:
            for task in self.get_param_value('task_list'):
                qbH, qbL = self._get_qbH_qbL(task)
                if qbH is not None:
                    for actual_detuning in [True, False]:
                        base_plot_name = f'Chevron_{qbH}_{qbL}_pe_actual' if actual_detuning else \
                            f'Chevron_{qbH}_{qbL}_pe_expected'
                        xlabel, xunit = self.get_xaxis_label_unit(qbH)
                        ylabel = r'Detuning $\Delta$' if actual_detuning else r'Expected detuning $\Delta_{exp}$'
                        yunit = 'Hz'
                        # ylabel = self.get_yaxis_label(qbH)
                        xvals = self.proc_data_dict['sweep_points_dict'][qbH][
                            'msmt_sweep_points']
                        Delta = self.proc_data_dict['Delta'][qbH]
                        offset_freq = self.fit_dicts[f'chevron_fit_{qbH}_{qbL}']['fit_res'].best_values['offset_freq']
                        Delta_fine = np.linspace(-2*abs(min(Delta)-abs(
                            offset_freq)), 2*(max(Delta)+offset_freq),
                                                 steps) # for fit plotting
                        J = self.proc_data_dict['analysis_params_dict'][f'{qbH}_{qbL}']['J']
                        offset = self.proc_data_dict['analysis_params_dict'][f'{qbH}_{qbL}']['offset_freq']
                        mmt_string = self.measurement_strings[
                                        qbH].partition('_')[0]
                        t_CZ = self.proc_data_dict['analysis_params_dict'][
                                f'{qbH}_{qbL}']['t_CZ_'+mmt_string]
                        textstr = r'$J = ${:.2f} MHz'.format(J/1e6)  + '\n'
                        textstr += r'Detuning_offset = {:.2f} MHz'.format(offset/1e6) + '\n'
                        textstr += r'$t_\mathrm{CZ} = $' +  '{:.2f} ns'.format(t_CZ*1e9)
                        self.plot_dicts[f'{base_plot_name}_main'] = {
                            'plotfn': self.plot_colorxy,
                            'fig_id': base_plot_name,
                            'xvals': xvals,
                            'yvals': Delta+offset_freq if actual_detuning else Delta,
                            'zvals': self.proc_data_dict['projected_data_dict'][qbH]['pe'],
                            'xlabel': xlabel,
                            'xunit': xunit,
                            'ylabel': ylabel,
                            'yunit': yunit,
                            'title': (self.raw_data_dict['timestamp'] + ' ' +
                                      self.measurement_strings[qbH]),
                            'clabel': self.get_yaxis_label(qb_name=qbH, data_key='e'),}
                        for n in range(num_curves+1):
                            fit_plot_name = f'fit_Chevron_{qbH}_{qbL}_pe_{n}_actual' if actual_detuning else \
                                f'fit_Chevron_{qbH}_{qbL}_pe_{n}_expected'
                            J = self.fit_dicts[f'chevron_fit_{qbH}_{qbL}']['fit_res'].best_values['J']
                            xvals_fit = self.t_CARB(J, Delta_fine, n) if actual_detuning else \
                                self.t_CARB(J, Delta_fine + offset_freq, n)
                            self.plot_dicts[f'{fit_plot_name}_main'] = {
                                'plotfn': self.plot_line,
                                'fig_id': base_plot_name,
                                'xvals': xvals_fit,
                                'yvals': Delta_fine,
                                'color': 'red',
                                'marker': 'None',
                                # 'datalabel': None,
                                         }
                        self.plot_dicts[f'text_msg_Chevron_{base_plot_name}]'] = {
                            'fig_id': base_plot_name,
                            'ypos': -0.2,
                            'xpos': -0.025,
                            'horizontalalignment': 'left',
                            'verticalalignment': 'top',
                            'color': 'k',
                            'plotfn': self.plot_text,
                            'text_string': textstr}

                        for negative_J in [True, False]:
                            self.plot_dicts[f'{fit_plot_name}_plot_J_{negative_J}'] = {
                                'plotfn': self.plot_line,
                                'fig_id': base_plot_name,
                                'xvals': [0, 2*xvals[-1]-xvals[-2]], # make
                                # sure the horizontal lines indicating J are
                                # wide enough to cover the entire plot
                                'yvals': J*np.ones(2) * (-1) ** negative_J - \
                                         offset_freq * (1-actual_detuning),
                                'color': 'white',
                                'marker': 'None',
                                # 'setlabel': 'J',
                                # 'do_legend': True, #FIXME: this also makes a legend for the fit lines
                            }

    @staticmethod
    def t_CARB(J, Delta, n):
        """
        Function to calculate the time needed for an arbitrary C-Phase gate.

        The mathemtical description of this function can be found in Nathan's master thesis for example.
        Parameters
        ----------
        J: coupling of the 20 and 11 transition in Hz
        Delta: Detuning of the qubits during the interaction in Hz
        n: Number of oscillation for which the interaction time should be calculated (usually you want the shortest
        (n=1)). This is mainly used to allow to plot the different traces in Delta-t-plane

        Returns
        -------
        The interaction time required for an arbitrary C-Phase gate for a given detuning and qubit coupling.
        """
        return n * 2 * np.pi / np.sqrt(4 * (2*np.pi*J) ** 2 + (2 * np.pi * Delta) ** 2)


class SingleRowChevronAnalysis(ChevronAnalysis):
    """Analysis for 1-dimensional Chevron QuantumExperiment

    This is used for instance for the 1-D measurements performed as part of
    CZ gate calibration
    """

    def extract_data(self):
        # Necessary for data processing and plotting since sweep_points are 2D
        self.default_options['TwoD'] = True
        super().extract_data()

    def prepare_projected_data_plots(self):
        # FIXME this overrides the super method but does almost the same. One
        #  should clean up the logic in the super to get rid of this, see below.
        for qbn, dd in self.proc_data_dict['projected_data_dict'].items():
            for k, v in dd.items():
                # FIXME this plot is almost the same as in the super method,
                #  but with plot_cal_points=True. The problem is that TwoD is
                #  True, meaning that the super does not plot the cal points,
                #  even though it correctly resolves that the data is really 1D.
                self.prepare_projected_data_plot(
                    fig_name=f'SingleRowChevron_{qbn}_{k}',
                    data=v[0],
                    data_axis_label=f'|{k[1:]}> state population',
                    qb_name=qbn, TwoD=False, plot_cal_points=True,
                )
                # FIXME this is the same as the projected plot, just rescaled.
                if k == 'pf':
                    self.prepare_projected_data_plot(
                        fig_name=f'Leakage_{qbn}',
                        data=v[0],
                        data_axis_label=f'|{k[1:]}> state population',
                        qb_name=qbn, TwoD=False, plot_cal_points=True,
                        yrange=(min(0,np.min(v[0][:-self.num_cal_points]))-0.01,
                                max(0,np.max(v[0][:-self.num_cal_points]))+0.01)
                    )

    def get_leakage_best_val(self, qbH_name, qbL_name, minimize='auto',
                             xtransform=None, xlabel=None, xunit=None,
                             colors=None):
        """Gets sweep parameter value corresponding to minimum/maximum leakage

        Fits a second-order polynomial and extracts extremum qbH f population.
        Args:
            qbH_name, qbL_name: names of high- and low-frequency qubits
            minimize (bool or 'auto'): Whether to explicitly look for a
            minimum or maximum, by setting initial fit parameters to help the
                fit converge. If minimize=='auto', this is guessed by comparing
                the middle point to a linear fit through the end points.
            save_fig: Whether to save the figure.
            show_fig: Whether to show the figure.
            fig: Optionally pass a figure on which to plot the data.
            ax: Optional axis of fig on which to plot the data.
            xtransform: Optional transformation to apply to sweep parameters in
                the plot (e.g. change of units).
            xlabel: x axis label. If None: extracted from the sweep points.
            colors: Optionally set colours of the plot.
        Returns:
            Sweep parameter corresponding to the extremum of the fitted model.
        Note that this method could be generalised to fit populations from
            other states than the f state of the high-frequency qubit.
        """

        if colors is None:
            colors = ['C0', 'C1']
        data = self.proc_data_dict['projected_data_dict'][qbH_name][
                   'pf'][0, :-3]
        x = self.sp.get_sweep_params_property('values',
                                              dimension=0).copy()
        if minimize == 'auto':
            minimize = data[len(data) // 2] < (data[0] + data[-1])/2
        if minimize:
            a = 100
            c = 0
        else:
            a = -100
            c = 1
        if xtransform:
            x = xtransform(x)
        xlabel = xlabel or self.sp.get_sweep_params_property('label')
        xunit = xunit or self.sp.get_sweep_params_property('unit')
        fact = 1
        while abs(max(x)) < 1e-3:
            x *= 1e3
            fact *= 1e3
        model_func = lambda x, a, b, c: a * ((x - b) ** 2) + c
        model = lmfit.Model(model_func)
        self.fit_res = model.fit(data, x=x, a=a, c=c, b=np.mean(x))
        best_val = self.fit_res.best_values['b'] / fact
        x_resampled = np.linspace(x[0], x[-1], 100)
        fig_title = f'Leakage_sweep_{qbH_name}_{qbL_name}'

        self.plot_dicts[fig_title + '_fit'] = {
            'fig_id': fig_title,
            'title': fig_title,
            'plotfn': self.plot_line,
            'xvals': x_resampled / fact,
            'yvals': model_func(x_resampled, **self.fit_res.best_values),
            'xlabel': xlabel,
            'xunit': '',
            'ylabel': '$|2\\rangle$ state pop.',
            'yunit': '',
            'label': 'Fit',
            'color': colors[1],
            'marker': '',
        }
        self.plot_dicts[fig_title + '_data'] = {
            'fig_id': fig_title,
            'plotfn': self.plot_line,
            'xvals': x / fact,
            'yvals': data,
            'label': 'Meas.',
            'color': colors[0],
            'linestyle': '',
        }
        self.plot_dicts[fig_title + '_best'] = {
            'fig_id': fig_title,
            'plotfn': self.plot_vlines,
            'x': best_val,
            'ymin': np.min(data),
            'ymax': np.max(data),
            'colors': 'gray',
        }
        return best_val
