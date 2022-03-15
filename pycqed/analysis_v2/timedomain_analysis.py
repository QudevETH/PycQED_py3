import lmfit
import numpy as np
from numpy.linalg import inv
import scipy as sp
import itertools
import matplotlib as mpl
import cmath
from collections import OrderedDict, defaultdict

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
from pycqed.analysis.three_state_rotation import predict_proba_avg_ro
import traceback
import logging

from pycqed.utilities import math
from pycqed.utilities.general import find_symmetry_index
import pycqed.measurement.waveform_control.segment as seg_mod
import datetime as dt
log = logging.getLogger(__name__)
try:
    import qutip as qtp
except ImportError as e:
    log.warning('Could not import qutip, tomography code will not work')


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
        cal_points = self.options_dict.get('cal_points', cal_points)
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
        cal_points = self.options_dict['cal_points']
        nr_segments = self.raw_data_dict['measured_values'].shape[-1]
        for observable in cal_points:
            if isinstance(observable, (list, np.ndarray)):
                for idxs in observable:
                    cal_indices.update({idx % nr_segments for idx in idxs})
            else:  # assume dictionaries
                for idxs in observable.values():
                    cal_indices.update({idx % nr_segments for idx in idxs})
        return list(cal_indices)


def all_cal_points(d, nr_ch, reps=1):
    """
    Generates a list of calibration points for a Hilbert space of dimension d,
    with nr_ch channels and reps reprtitions of each calibration point.
    """
    return [[list(range(-reps*i, -reps*(i-1)))]*nr_ch for i in range(d, 0, -1)]


class Single_Qubit_TimeDomainAnalysis(ba.BaseDataAnalysis):

    def process_data(self):
        """
        This takes care of rotating and normalizing the data if required.
        this should work for several input types.
            - I/Q values (2 quadratures + cal points)
            - weight functions (1 quadrature + cal points)
            - counts (no cal points)

        There are several options possible to specify the normalization
        using the options dict.
            cal_points (tuple) of indices of the calibrati  on points

            zero_coord, one_coord
        """

        cal_points = self.options_dict.get('cal_points', None)
        zero_coord = self.options_dict.get('zero_coord', None)
        one_coord = self.options_dict.get('one_coord', None)

        if cal_points is None:
            # default for all standard Timedomain experiments
            cal_points = [list(range(-4, -2)), list(range(-2, 0))]

        if len(self.raw_data_dict['measured_values']) == 1:
            # if only one weight function is used rotation is not required
            self.proc_data_dict['corr_data'] = a_tools.rotate_and_normalize_data_1ch(
                self.raw_data_dict['measured_values'][0],
                cal_zero_points=cal_points[0],
                cal_one_points=cal_points[1])
        else:
            self.proc_data_dict['corr_data'], zero_coord, one_coord = \
                a_tools.rotate_and_normalize_data(
                    data=self.raw_data_dict['measured_values'][0:2],
                    zero_coord=zero_coord,
                    one_coord=one_coord,
                    cal_zero_points=cal_points[0],
                    cal_one_points=cal_points[1])

        # This should be added to the hdf5 datafile but cannot because of the
        # way that the "new" analysis works.

        # self.add_dataset_to_analysisgroup('Corrected data',
        #                                   self.proc_data_dict['corr_data'])


class MultiQubit_TimeDomain_Analysis(ba.BaseDataAnalysis):
    """
    Base class for multi-qubit time-domain analyses.

    Parameters that can be specified in the options dict:
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

        if not hasattr(self, "job"):
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

        # creates self.data_filter and self.data_with_reset
        self.get_data_filter()

        # These calls need to stay here because otherwise QScaleAnalysis will
        # fail: in process_data, this child needs the correctly created
        # sweep_points_dict.
        self.create_meas_results_per_qb()
        self.create_sweep_points_dict()
        self.get_num_cal_points()
        self.update_sweep_points_dict()
        if self.options_dict.get('TwoD', False):
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
                    self.channel_map = a_tools.get_qb_channel_map_from_hdf(
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
        # in options_dict/metadata
        data_to_fit = deepcopy(self.get_param_value('data_to_fit'))
        if data_to_fit is None or not len(data_to_fit):
            # If we have cal points, but data_to_fit is not specified,
            # choose a reasonable default value.
            data_to_fit = {}
            for qbn in self.qb_names:
                if not len(self.cal_states_dict[qbn]) or \
                        'pca' in self.rotation_type[qbn].lower():
                    data_to_fit[qbn] = self.rotation_type[qbn]
                else:
                    csr = [(k, v) for k, v in
                           self.cal_states_rotations[qbn].items()]
                    csr.sort(key=lambda t: t[1])
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
            - self.data_with_reset: bool that is checked in
                prepare_raw_data_plots
        """
        # flag to be used in prepare_raw_data_plots
        self.data_with_reset = False
        self.data_filter = self.get_param_value('data_filter')
        if self.data_filter is None:
            if 'active' in self.prep_params.get('preparation_type', 'wait'):
                reset_reps = self.prep_params.get('reset_reps', 3)
                self.data_filter = lambda x: x[reset_reps::reset_reps+1]
                self.data_with_reset = True
            elif "preselection" in self.prep_params.get('preparation_type',
                                                        'wait'):
                self.data_filter = lambda x: x[1::2]  # filter preselection RO
            else:
                self.data_filter = lambda x: x

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
            if main_sp is not None:
                self.proc_data_dict['sweep_points_dict'] = {}
                for qbn, p in main_sp.items():
                    dim = self.sp.find_parameter(p)
                    if dim == 1:
                        log.warning(f"main_sp is only implemented for sweep "
                                    f"dimension 0, but {p} is in dimension 1.")
                    self.proc_data_dict['sweep_points_dict'][qbn] = \
                        {'sweep_points': sp1d_filter(
                            self.sp.get_sweep_params_property('values', dim, p))}
            else:
                self.proc_data_dict['sweep_points_dict'] = \
                    {qbn: {'sweep_points': sp1d_filter(
                        self.sp.get_sweep_params_property(
                        'values', 0, self.mospm[qbn])[0])}
                     for qbn in self.qb_names}
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
                states_map=self.get_param_value("states_map"))

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
                          'sweep_points_dict', 'sweep_points_2D_dict']:
                    pdd[d][qbn] = pdd['split_data_dict'][qbn][p][ind][d]
                self.measurement_strings[qbn] += f' ({p}: {v})'

    def get_num_cal_points(self):
        """
        Figures out how many calibration segments were used in the experiment.

        First tries to get this information from the cal_states_dict (i.e. from
        provided cal points info, see get_cal_points method).

        Then checks whether the number of sweep points matches the size of the
        data. If not, then no cal points information was provided but cal
        points were used in the experiment.
            In this case, the no_cp_but_cp_in_data is set to True (will be
            checked in update_sweep_points_dict).

        Creates the attributes
            - self.num_cal_points
            - self.no_cp_but_cp_in_data
        """

        # Count num_cal_points from self.cal_states_dict
        self.num_cal_points = np.array(list(
            self.cal_states_dict[
                list(self.cal_states_dict)[0]].values())).flatten().size

        self.no_cp_but_cp_in_data = False
        spd = self.proc_data_dict['sweep_points_dict']
        num_sp = len(spd[list(spd)[0]]['sweep_points'])
        mrpq = self.proc_data_dict['meas_results_per_qb']
        mrpq_raw_dict = mrpq[list(mrpq)[0]]
        num_data_points = len(mrpq_raw_dict[list(mrpq_raw_dict)[0]])
        if self.num_cal_points == 0 and num_data_points != num_sp:
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

        sweep_points_dict = {}
        for qbn in self.qb_names:
            sweep_points_dict[qbn] = {}
            if cp_obj is not None:
                sweep_points_dict[qbn]['sweep_points'] = \
                    cp_obj.extend_sweep_points_by_n_cal_pts(
                        self.num_cal_points,
                        self.proc_data_dict['sweep_points_dict'][qbn][
                            'sweep_points'])
                sweep_points_dict[qbn]['msmt_sweep_points'] = \
                    sweep_points_dict[qbn]['sweep_points'][:-self.num_cal_points]
                sweep_points_dict[qbn]['cal_points_sweep_points'] = \
                    sweep_points_dict[qbn]['sweep_points'][-self.num_cal_points::]
            else:
                sweep_points_dict[qbn]['sweep_points'] = \
                    self.proc_data_dict['sweep_points_dict'][qbn]['sweep_points']
                sweep_points_dict[qbn]['msmt_sweep_points'] = \
                    sweep_points_dict[qbn]['sweep_points']
                sweep_points_dict[qbn]['cal_points_sweep_points'] = []

        self.proc_data_dict['sweep_points_dict'] = sweep_points_dict

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
            for i in list(cal_states_rot_qb.values()):
                # cal state corresponding to transmon level i
                cal_state = \
                    [k for k, idx in cal_states_rot_qb.items()
                     if idx == i][0]
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
                                 cal_states_dict):
        # FOR 3 CAL STATES
        rotated_data_dict = OrderedDict()
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_states_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            raw_data = np.array([v for v in meas_res_dict.values()]).T
            for i, cal_idx in enumerate(cal_pts_idxs):
                cal_points_data[i, :] = np.mean(raw_data[cal_idx, :],
                                                axis=0)
            rotated_data = predict_proba_avg_ro(raw_data, cal_points_data)
            for i, state in enumerate(list(cal_states_dict[qb_name])):
                rotated_data_dict[qb_name][f'p{state}'] = rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        return rotated_data_dict

    @staticmethod
    def rotate_data(qb_name, meas_results_per_qb, channel_map,
                    cal_states_dict, storing_keys, data_mostly_g=True):
        # ONLY WORKS FOR 2 CAL STATES
        if 'pca' not in storing_keys[qb_name].lower():
            # Add the other state probability
            # ex if storing_keys[qb_name] == 'pe' add data for 'pg' as 1-pe
            qb_cal_states = cal_states_dict[qb_name].keys()
            if len(qb_cal_states) != 2:
                raise ValueError(f'Expected two cal states for {qb_name} but '
                                 f'found {len(qb_cal_states)}: {qb_cal_states}')
            other_cs = [cs for cs in qb_cal_states
                        if cs != storing_keys[qb_name][-1]]
            if len(other_cs) == 0:
                raise ValueError(f'There are no other cal states except for '
                                 f'{storing_keys[qb_name][-1]} from '
                                 f'storing_keys.')
            elif len(other_cs) > 1:
                raise ValueError(f'There is more than one other cal state in '
                                 f'addition to {storing_keys[qb_name][-1]} from'
                                 f' storing_keys. Not clear which one to use.')
            other_cs = f'p{other_cs[0]}'

        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        vals = list(cal_states_dict[qb_name].values())
        if len(cal_states_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_states_dict[qb_name].values())
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
                rotated_data_dict[qb_name][storing_keys[qb_name]] = data
            else:
                rotated_data_dict[qb_name][storing_keys[qb_name]] = \
                    a_tools.rotate_and_normalize_data_1ch(
                        data=meas_res_dict[list(meas_res_dict)[0]],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
            if 'pca' not in storing_keys[qb_name].lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][storing_keys[qb_name]]
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            data, _, _ = a_tools.rotate_and_normalize_data_IQ(
                data=np.array([v for v in meas_res_dict.values()]),
                cal_zero_points=cal_zero_points,
                cal_one_points=cal_one_points)
            if cal_zero_points is None:
                data = a_tools.set_majority_sign(
                    data, -1 if data_mostly_g else 1)
            rotated_data_dict[qb_name][storing_keys[qb_name]] = data
            if 'pca' not in storing_keys[qb_name].lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][storing_keys[qb_name]]
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
                        rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]] = data
                    else:
                        rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]] = \
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
                    rotated_data_dict[qb_name][ro_suf][
                        storing_keys[qb_name]] = data
                if 'pca' not in storing_keys[qb_name].lower():
                    rotated_data_dict[qb_name][ro_suf][other_cs] = \
                        1 - rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]]
        return rotated_data_dict

    @staticmethod
    def rotate_data_3_cal_states_TwoD(qb_name, meas_results_per_qb,
                                      channel_map, cal_states_dict):
        # FOR 3 CAL STATES
        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        rotated_data_dict[qb_name] = OrderedDict()
        cal_pts_idxs = list(cal_states_dict[qb_name].values())
        cal_points_data = np.zeros((len(cal_pts_idxs), 2))
        if list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            for i, state in enumerate(list(cal_states_dict[qb_name])):
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

                for i, state in enumerate(list(cal_states_dict[qb_name])):
                    rotated_data_dict[qb_name][f'p{state}'][:, col] = \
                        rotated_data[:, i]
        else:
            raise NotImplementedError('Calibration states rotation with 3 '
                                      'cal states only implemented for '
                                      '2 readout channels per qubit.')
        # transpose data
        for i, state in enumerate(list(cal_states_dict[qb_name])):
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
                         cal_states_dict, storing_keys,
                         column_PCA=False, data_mostly_g=True):
        # ONLY WORKS FOR 2 CAL STATES
        if 'pca' not in storing_keys[qb_name].lower():
            # Add the other state probability
            # ex if storing_keys[qb_name] == 'pe' add data for 'pg' as 1-pe
            qb_cal_states = cal_states_dict[qb_name].keys()
            if len(qb_cal_states) != 2:
                raise ValueError(f'Expected two cal states for {qb_name} but '
                                 f'found {len(qb_cal_states)}: {qb_cal_states}')
            other_cs = [cs for cs in qb_cal_states
                        if cs != storing_keys[qb_name][-1]]
            if len(other_cs) == 0:
                raise ValueError(f'There are no other cal states except for '
                                 f'{storing_keys[qb_name][-1]} from '
                                 f'storing_keys.')
            elif len(other_cs) > 1:
                raise ValueError(f'There is more than one other cal state in '
                                 f'addition to {storing_keys[qb_name][-1]} from'
                                 f' storing_keys. Not clear which one to use.')
            other_cs = f'p{other_cs[0]}'

        meas_res_dict = meas_results_per_qb[qb_name]
        rotated_data_dict = OrderedDict()
        vals = list(cal_states_dict[qb_name].values())
        if len(cal_states_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_states_dict[qb_name].values())
            cal_pts_idxs.sort()
            cal_zero_points = cal_pts_idxs[0]
            cal_one_points = cal_pts_idxs[1]
        rotated_data_dict[qb_name] = OrderedDict()
        if len(meas_res_dict) == 1:
            # one RO channel per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][storing_keys[qb_name]] = \
                deepcopy(raw_data_arr.transpose())
            if column_PCA:
                for row in range(raw_data_arr.shape[0]):
                    data = a_tools.rotate_and_normalize_data_1ch(
                        data=raw_data_arr[row, :],
                        cal_zero_points=cal_zero_points,
                        cal_one_points=cal_one_points)
                    data = a_tools.set_majority_sign(
                        data, -1 if data_mostly_g else 1)
                    rotated_data_dict[qb_name][storing_keys[qb_name]][
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
                    rotated_data_dict[qb_name][storing_keys[qb_name]][col] = data
            if 'pca' not in storing_keys[qb_name].lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][storing_keys[qb_name]]
        elif list(meas_res_dict) == channel_map[qb_name]:
            # two RO channels per qubit
            raw_data_arr = meas_res_dict[list(meas_res_dict)[0]]
            rotated_data_dict[qb_name][storing_keys[qb_name]] = \
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
                    rotated_data_dict[qb_name][storing_keys[qb_name]][
                        :, row] = data
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
                    rotated_data_dict[qb_name][
                        storing_keys[qb_name]][col] = data
            if 'pca' not in storing_keys[qb_name].lower():
                rotated_data_dict[qb_name][other_cs] = \
                    1 - rotated_data_dict[qb_name][storing_keys[qb_name]]
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
                    rotated_data_dict[qb_name][ro_suf][storing_keys[qb_name]] = \
                        deepcopy(raw_data_arr.transpose())
                    for col in range(raw_data_arr.shape[1]):
                        data = a_tools.rotate_and_normalize_data_1ch(
                                data=raw_data_arr[:, col],
                                cal_zero_points=cal_zero_points,
                                cal_one_points=cal_one_points)
                        if cal_zero_points is None:
                            data = a_tools.set_majority_sign(
                                data, -1 if data_mostly_g else 1)
                        rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]][col] = data
                else:
                    # two RO ch per qubit
                    raw_data_arr = meas_res_dict[list(meas_res_dict)[i]]
                    rotated_data_dict[qb_name][ro_suf][storing_keys[qb_name]] = \
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
                        rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]][col] = data
                if 'pca' not in storing_keys[qb_name].lower():
                    rotated_data_dict[qb_name][ro_suf][other_cs] = \
                        1 - rotated_data_dict[qb_name][ro_suf][
                            storing_keys[qb_name]]
        return rotated_data_dict

    @staticmethod
    def rotate_data_TwoD_same_fixed_cal_idxs(qb_name, meas_results_per_qb,
                                             channel_map, cal_states_dict,
                                             storing_keys):
        meas_res_dict = meas_results_per_qb[qb_name]
        if list(meas_res_dict) != channel_map[qb_name]:
            raise NotImplementedError('rotate_data_TwoD_same_fixed_cal_idxs '
                                      'only implemented for two-channel RO!')

        vals = list(cal_states_dict[qb_name].values())
        if len(cal_states_dict[qb_name]) == 0 or any([v is None for v in vals]):
            cal_zero_points = None
            cal_one_points = None
        else:
            cal_pts_idxs = list(cal_states_dict[qb_name].values())
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
            main_sp = self.get_param_value('main_sp', None)
            if main_sp is not None and qb_name in main_sp:
                param_names = [main_sp[qb_name]]
            else:
                param_names = self.mospm[qb_name]
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
        elif 'p' in prob_label.lower():
            return r'$|{}\rangle$'.format(prob_label[-1])
        else:
            return r'$|{}\rangle$'.format(prob_label)

    def get_yaxis_label(self, qb_name, data_key=None):
        if self.rotate and ('pca' in self.rotation_type[qb_name].lower() or
                            not len(self.cal_states_dict[
                                        list(self.cal_states_dict)[0]])):
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

    def _get_single_shots_per_qb(self, raw=False):
        """
        Gets single shots from the proc_data_dict and arranges
        them as arrays per qubit
        Args:
            raw (bool): whether or not to return raw shots (before
            data filtering)

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
        for qbn in self.qb_names:
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
                 Defaults to None: in this case each qubit is preselected independently from others
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

        if preselection_qbs is None:
            # default is each qubit preselected individually
            # note that the list includes the qubit name twice as the minimal
            # number of arguments in logical_and.reduce() is 2.
            preselection_qbs = {qbn: [qbn] for qbn in presel_shots_per_qb}

        for qbn, presel_qbs in preselection_qbs.items():
            if len(presel_qbs) == 1:
                presel_qbs = [presel_qbs[0], presel_qbs[0]]
            presel_mask[qbn] = np.logical_and.reduce(
                [presel_mask_single_qb[qb] for qb in presel_qbs])

        return presel_mask

    def process_single_shots(self, predict_proba=True,
                             classifier_params=None,
                             states_map=None):
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
            n_shots_from_hdf = [
                int(self.get_hdf_param_value(f"Instrument settings/{qbn}",
                                             "acq_shots")) for qbn in self.qb_names]
            if len(np.unique(n_shots_from_hdf)) > 1:
                log.warning("Number of shots extracted from hdf are not all the same:"
                            "assuming n_shots=max(qb.acq_shots() for qb in qb_names)")
            n_shots = np.max(n_shots_from_hdf)

        # determine number of readouts per sequence
        if self.get_param_value("TwoD", False):
            n_seqs = self.sp.length(1)  # corresponds to number of soft sweep points
        else:
            n_seqs = 1
        # n_reaouds  refers to the number of readouts per sequence after filtering out e.g.
        # preselection readouts
        n_readouts = list(shots_per_qb.values())[0].shape[0] // (n_shots * n_seqs)

        # get classification parameters
        if classifier_params is None:
            classifier_params = {}
            from numpy import array  # for eval
            for qbn in self.qb_names:
                classifier_params[qbn] = eval(self.get_hdf_param_value(
                f'Instrument settings/{qbn}', "acq_classifier_params"))

        # prepare preselection mask
        if preselection:
            # get preselection readouts
            shots_per_qb_before_filtering = self._get_single_shots_per_qb(raw=True)
            n_ro_before_filtering = \
                list(shots_per_qb_before_filtering.values())[0].shape[0] // \
                (n_shots * n_seqs)
            preselection_ro_mask = \
                np.tile([True] * n_seqs +
                        [False] * (n_ro_before_filtering - n_readouts) * n_seqs,
                        n_shots * n_readouts)
            presel_shots_per_qb = \
                {qbn: presel_shots[preselection_ro_mask] for qbn, presel_shots in
                 shots_per_qb_before_filtering.items()}
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
                              ' of features, probably wrong classifer parameters'
                              ' were passed (e.g. a classifier trained with'
                              ' a different number of channels than in the'
                              f' current measurement): {e}')
                    raise e
                if not 'single_shots_per_qb_probs' in self.proc_data_dict:
                    self.proc_data_dict['single_shots_per_qb_probs'] = {}
                self.proc_data_dict['single_shots_per_qb_probs'][qbn] = shots


            # TODO: Nathan: if predict_proba is activated then we should
            #  first classify, then do a count table and thereby estimate
            #  average proba
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
                averaged_shots = np.reshape(averaged_shots, (n_readouts, n_seqs, -1))
                averaged_shots = np.swapaxes(averaged_shots, 0, 1) # return to original 2D shape
            # reshape to (n_prob or n_ch or 1, n_readouts) if 1d
            # or (n_prob or n_ch or 1, n_readouts, n_ssp) if 2d
            averaged_shots = np.array(averaged_shots).T

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
        if self.get_param_value('plot_proj_data', default_value=True):
            select_split = self.get_param_value('select_split')
            fig_name_suffix = self.get_param_value('fig_name_suffix', '')
            title_suffix = self.get_param_value('title_suffix', '')
            for qb_name, corr_data in self.proc_data_dict[
                    'projected_data_dict'].items():
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
                    for data_key, data in corr_data.items():
                        fn = f'{fig_name}_{data_key}'
                        if not self.rotate:
                            data_label = ''
                            plot_name_suffix = data_key
                            plot_cal_points = False
                        else:
                            data_label = 'Data'
                            plot_name_suffix = ''
                            plot_cal_points = (
                                not self.options_dict.get('TwoD', False))
                        data_axis_label = self.get_yaxis_label(qb_name,
                                                               data_key)
                        tf = f'{data_key}_{title_suf}' if \
                            len(title_suf) else data_key
                        self.prepare_projected_data_plot(
                            fn, data, qb_name=qb_name,
                            data_label=data_label,
                            title_suffix=tf,
                            plot_name_suffix=plot_name_suffix,
                            fig_name_suffix=fig_name_suffix,
                            data_axis_label=data_axis_label,
                            plot_cal_points=plot_cal_points)

                else:
                    fig_name = 'projected_plot_' + qb_name
                    self.prepare_projected_data_plot(
                        fig_name, corr_data, qb_name=qb_name,
                        plot_cal_points=(
                            not self.options_dict.get('TwoD', False)))

        if self.get_param_value('plot_raw_data', default_value=True):
            self.prepare_raw_data_plots(plot_filtered=False)
            if 'preparation_params' in self.metadata:
                if 'active' in self.metadata['preparation_params'].get(
                        'preparation_type', 'wait'):
                    self.prepare_raw_data_plots(plot_filtered=True)

    def prepare_raw_data_plots(self, plot_filtered=False):
        if plot_filtered or not self.data_with_reset:
            key = 'meas_results_per_qb'
            suffix = 'filtered' if self.data_with_reset else ''
            func_for_swpts = lambda qb_name: self.proc_data_dict[
                'sweep_points_dict'][qb_name]['sweep_points']
        else:
            key = 'meas_results_per_qb_raw'
            suffix = ''
            func_for_swpts = lambda qb_name: self.raw_data_dict[
                'hard_sweep_points']
        for qb_name, raw_data_dict in self.proc_data_dict[key].items():
            if qb_name not in self.qb_names:
                continue
            sweep_points = func_for_swpts(qb_name)
            if len(raw_data_dict) == 1:
                numplotsx = 1
                numplotsy = 1
            elif len(raw_data_dict) == 2:
                numplotsx = 1
                numplotsy = 2
            else:
                numplotsx = 2
                numplotsy = len(raw_data_dict) // 2 + len(raw_data_dict) % 2

            plotsize = self.get_default_plot_params(set=False)['figure.figsize']
            fig_title = (self.raw_data_dict['timestamp'] + ' ' +
                         self.raw_data_dict['measurementstring'] +
                         '\nRaw data ' + suffix + ' ' + qb_name)
            plot_name = 'raw_plot_' + qb_name + suffix
            xlabel, xunit = self.get_xaxis_label_unit(qb_name)

            for ax_id, ro_channel in enumerate(raw_data_dict):
                if self.get_param_value('TwoD', default_value=False):
                    if self.sp is None:
                        soft_sweep_params = self.get_param_value(
                            'soft_sweep_params')
                        if soft_sweep_params is not None:
                            yunit = list(soft_sweep_params.values())[0]['unit']
                        else:
                            yunit = self.raw_data_dict[
                                'sweep_parameter_units'][1]
                        if np.ndim(yunit) > 0:
                            yunit = yunit[0]
                    for pn, ssp in self.proc_data_dict['sweep_points_2D_dict'][
                            qb_name].items():
                        ylabel = pn
                        if self.sp is not None:
                            yunit = self.sp.get_sweep_params_property(
                                'unit', dimension=1, param_names=pn)
                            ylabel = self.sp.get_sweep_params_property(
                                'label', dimension=1, param_names=pn)
                        self.plot_dicts[f'{plot_name}_{ro_channel}_{pn}'] = {
                            'fig_id': plot_name + '_' + pn,
                            'ax_id': ax_id,
                            'plotfn': self.plot_colorxy,
                            'xvals': sweep_points,
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
                            'clabel': '{} (Vpeak)'.format(ro_channel)}
                else:
                    self.plot_dicts[plot_name + '_' + ro_channel] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': sweep_points,
                        'xlabel': xlabel,
                        'xunit': xunit,
                        'yvals': raw_data_dict[ro_channel],
                        'ylabel': '{} (Vpeak)'.format(ro_channel),
                        'yunit': '',
                        'numplotsx': numplotsx,
                        'numplotsy': numplotsy,
                        'plotsize': (plotsize[0]*numplotsx,
                                     plotsize[1]*numplotsy),
                        'title': fig_title}
            if len(raw_data_dict) == 1:
                self.plot_dicts[
                    plot_name + '_' + list(raw_data_dict)[0]]['ax_id'] = None

    def prepare_projected_data_plot(
            self, fig_name, data, qb_name, title_suffix='', sweep_points=None,
            plot_cal_points=True, plot_name_suffix='', fig_name_suffix='',
            data_label='Data', data_axis_label='', do_legend_data=True,
            do_legend_cal_states=True, TwoD=None, yrange=None):

        if len(fig_name_suffix):
            fig_name = f'{fig_name}_{fig_name_suffix}'

        if data_axis_label == '':
            data_axis_label = self.get_yaxis_label(qb_name=qb_name)
        plotsize = self.get_default_plot_params(set=False)['figure.figsize']
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
        xlabel, xunit = self.get_xaxis_label_unit(qb_name)

        if TwoD is None:
            TwoD = self.get_param_value('TwoD', default_value=False)
        if TwoD:
            if self.sp is None:
                soft_sweep_params = self.get_param_value(
                    'soft_sweep_params')
                if soft_sweep_params is not None:
                    yunit = list(soft_sweep_params.values())[0]['unit']
                else:
                    yunit = self.raw_data_dict['sweep_parameter_units'][1]
                if np.ndim(yunit) > 0:
                    yunit = yunit[0]
            for pn, ssp in self.proc_data_dict['sweep_points_2D_dict'][
                    qb_name].items():
                ylabel = pn
                if self.sp is not None:
                    yunit = self.sp.get_sweep_params_property(
                        'unit', dimension=1, param_names=pn)
                    ylabel = self.sp.get_sweep_params_property(
                        'label', dimension=1, param_names=pn)
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
                    'zrange': self.get_param_value('zrange', None),
                    'title': title,
                    'clabel': data_axis_label}
        else:
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
                'setlabel': data_label,
                'title': title,
                'linestyle': 'none',
                'do_legend': do_legend_data,
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


class Idling_Error_Rate_Analyisis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        post_sel_th = self.options_dict.get('post_sel_th', 0.5)
        raw_shots = self.raw_data_dict['measured_values'][0][0]
        post_sel_shots = raw_shots[::2]
        data_shots = raw_shots[1::2]
        data_shots[np.where(post_sel_shots > post_sel_th)] = np.nan

        states = ['0', '1', '+']
        self.proc_data_dict['xvals'] = np.unique(self.raw_data_dict['xvals'])
        for i, state in enumerate(states):
            self.proc_data_dict['shots_{}'.format(state)] =data_shots[i::3]

            self.proc_data_dict['yvals_{}'.format(state)] = \
                np.nanmean(np.reshape(self.proc_data_dict['shots_{}'.format(state)],
                               (len(self.proc_data_dict['xvals']), -1),
                               order='F'), axis=1)


    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            self.plot_dicts['Prepare in {}'.format(state)] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': xvals,
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': yvals,
                'ylabel': 'Counts',
                'yrange': [0, 1],
                'xrange': self.options_dict.get('xrange', None),
                'yunit': 'frac',
                'setlabel': 'Prepare in {}'.format(state),
                'do_legend':True,
                'title': (self.raw_data_dict['timestamps'][0]+' - ' +
                          self.raw_data_dict['timestamps'][-1] + '\n' +
                          self.raw_data_dict['measurementstring'][0]),
                'legend_pos': 'upper right'}
        if self.do_fitting:
            for state in ['0', '1', '+']:
                self.plot_dicts['fit_{}'.format(state)] = {
                    'ax_id': 'main',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['fit {}'.format(state)]['fit_res'],
                    'plot_init': self.options_dict['plot_init'],
                    'setlabel': 'fit |{}>'.format(state),
                    'do_legend': True,
                    'legend_pos': 'upper right'}

                self.plot_dicts['fit_text']={
                    'ax_id':'main',
                    'box_props': 'fancy',
                    'xpos':1.05,
                    'horizontalalignment':'left',
                    'plotfn': self.plot_text,
                    'text_string': self.proc_data_dict['fit_msg']}



    def analyze_fit_results(self):
        fit_msg =''
        states = ['0', '1', '+']
        for state in states:
            fr = self.fit_res['fit {}'.format(state)]
            N1 = fr.params['N1'].value, fr.params['N1'].stderr
            N2 = fr.params['N2'].value, fr.params['N2'].stderr
            fit_msg += ('Prep |{}> : \n\tN_1 = {:.2g} $\pm$ {:.2g}'
                    '\n\tN_2 = {:.2g} $\pm$ {:.2g}\n').format(
                state, N1[0], N1[1], N2[0], N2[1])

        self.proc_data_dict['fit_msg'] = fit_msg

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        states = ['0', '1', '+']
        for i, state in enumerate(states):
            yvals = self.proc_data_dict['yvals_{}'.format(state)]
            xvals =  self.proc_data_dict['xvals']

            mod = lmfit.Model(fit_mods.idle_error_rate_exp_decay)
            mod.guess = fit_mods.idle_err_rate_guess.__get__(mod, mod.__class__)

            # Done here explicitly so that I can overwrite a specific guess
            guess_pars = mod.guess(N=xvals, data=yvals)
            vary_N2 = self.options_dict.get('vary_N2', True)

            if not vary_N2:
                guess_pars['N2'].value = 1e21
                guess_pars['N2'].vary = False
            self.fit_dicts['fit {}'.format(states[i])] = {
                'model': mod,
                'fit_xvals': {'N': xvals},
                'fit_yvals': {'data': yvals},
                'guess_pars': guess_pars}
            # Allows fixing the double exponential coefficient


class Grovers_TwoQubitAllStates_Analysis(ba.BaseDataAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        normalize_to_cal_points = self.options_dict.get('normalize_to_cal_points', True)
        cal_points = [
                        [[-4, -3], [-2, -1]],
                        [[-4, -2], [-3, -1]],
                       ]
        for idx in [0,1]:
            yvals = list(self.raw_data_dict['measured_data'].values())[idx][0]

            self.proc_data_dict['ylabel_{}'.format(idx)] = \
                self.raw_data_dict['value_names'][0][idx]
            self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.rotate_and_normalize_data_1ch(yvals,
                    cal_zero_points=cal_points[idx][0],
                    cal_one_points=cal_points[idx][1])
            self.proc_data_dict['yvals_{}'.format(idx)] = yvals

        y0 = self.proc_data_dict['yvals_0']
        y1 = self.proc_data_dict['yvals_1']
        p_success = ((y0[0]*y1[0]) +
                     (1-y0[1])*y1[1] +
                     (y0[2])*(1-y1[2]) +
                     (1-y0[3])*(1-y1[3]) )/4
        self.proc_data_dict['p_success'] = p_success


    def prepare_plots(self):
        # assumes that value names are unique in an experiment
        for i in [0, 1]:
            yvals = self.proc_data_dict['yvals_{}'.format(i)]
            xvals =  self.raw_data_dict['xvals'][0]
            ylabel = self.proc_data_dict['ylabel_{}'.format(i)]
            self.plot_dicts['main_{}'.format(ylabel)] = {
                'plotfn': self.plot_line,
                'xvals': self.raw_data_dict['xvals'][0],
                'xlabel': self.raw_data_dict['xlabel'][0],
                'xunit': self.raw_data_dict['xunit'][0][0],
                'yvals': self.proc_data_dict['yvals_{}'.format(i)],
                'ylabel': ylabel,
                'yunit': self.proc_data_dict['yunit'],
                'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                          self.raw_data_dict['measurementstring'][0]),
                'do_legend': False,
                'legend_pos': 'upper right'}


        self.plot_dicts['limit_text']={
            'ax_id':'main_{}'.format(ylabel),
            'box_props': 'fancy',
            'xpos':1.05,
            'horizontalalignment':'left',
            'plotfn': self.plot_text,
            'text_string': 'P succes = {:.3f}'.format(self.proc_data_dict['p_success'])}








class FlippingAnalysis(Single_Qubit_TimeDomainAnalysis):

    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = True

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'sweep_points': 'sweep_points',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}
        # This analysis makes a hardcoded assumption on the calibration points
        self.options_dict['cal_points'] = [list(range(-4, -2)),
                                           list(range(-2, 0))]

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        # Even though we expect an exponentially damped oscillation we use
        # a simple cosine as this gives more reliable fitting and we are only
        # interested in extracting the frequency of the oscillation
        cos_mod = lmfit.Model(fit_mods.CosFunc)

        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.raw_data_dict['sweep_points'][:-4],
            data=self.proc_data_dict['corr_data'][:-4])

        # This enforces the oscillation to start at the equator
        # and ensures that any over/under rotation is absorbed in the
        # frequency
        guess_pars['amplitude'].value = 0.5
        guess_pars['amplitude'].vary = False
        guess_pars['offset'].value = 0.5
        guess_pars['offset'].vary = False

        self.fit_dicts['cos_fit'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

        # In the case there are very few periods we fall back on a small
        # angle approximation to extract the drive detuning
        poly_mod = lmfit.models.PolynomialModel(degree=1)
        # the detuning can be estimated using on a small angle approximation
        # c1 = d/dN (cos(2*pi*f N) ) evaluated at N = 0 -> c1 = -2*pi*f
        poly_mod.set_param_hint('frequency', expr='-c1/(2*pi)')
        guess_pars = poly_mod.guess(x=self.raw_data_dict['sweep_points'][:-4],
                                    data=self.proc_data_dict['corr_data'][:-4])
        # Constraining the line ensures that it will only give a good fit
        # if the small angle approximation holds
        guess_pars['c0'].vary = False
        guess_pars['c0'].value = 0.5

        self.fit_dicts['line_fit'] = {
            'model': poly_mod,
            'fit_xvals': {'x': self.raw_data_dict['sweep_points'][:-4]},
            'fit_yvals': {'data': self.proc_data_dict['corr_data'][:-4]},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        sf_line = self._get_scale_factor_line()
        sf_cos = self._get_scale_factor_cos()
        self.proc_data_dict['scale_factor'] = self.get_scale_factor()

        msg = 'Scale fact. based on '
        if self.proc_data_dict['scale_factor'] == sf_cos:
            msg += 'cos fit\n'
        else:
            msg += 'line fit\n'
        msg += 'cos fit: {:.4f}\n'.format(sf_cos)
        msg += 'line fit: {:.4f}'.format(sf_line)

        self.raw_data_dict['scale_factor_msg'] = msg
        # TODO: save scale factor to file

    def get_scale_factor(self):
        """
        Returns the scale factor that should correct for the error in the
        pulse amplitude.
        """
        # Model selection based on the Bayesian Information Criterion (BIC)
        # as  calculated by lmfit
        if (self.fit_dicts['line_fit']['fit_res'].bic <
                self.fit_dicts['cos_fit']['fit_res'].bic):
            scale_factor = self._get_scale_factor_line()
        else:
            scale_factor = self._get_scale_factor_cos()
        return scale_factor

    def _get_scale_factor_cos(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['cos_fit']['fit_res'].params['frequency']

        # the square is needed to account for the difference between
        # power and amplitude
        scale_factor = (1+frequency)**2

        phase = np.rad2deg(self.fit_dicts['cos_fit']['fit_res'].params['phase']) % 360
        # phase ~90 indicates an under rotation so the scale factor
        # has to be larger than 1. A phase ~270 indicates an over
        # rotation so then the scale factor has to be smaller than one.
        if phase > 180:
            scale_factor = 1/scale_factor

        return scale_factor

    def _get_scale_factor_line(self):
        # 1/period of the oscillation corresponds to the (fractional)
        # over/under rotation error per gate
        frequency = self.fit_dicts['line_fit']['fit_res'].params['frequency']
        scale_factor = (1+frequency)**2
        # no phase sign check is needed here as this is contained in the
        # sign of the coefficient

        return scale_factor

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['sweep_points'],
            'xlabel': self.raw_data_dict['xlabel'],
            'xunit': self.raw_data_dict['xunit'],  # does not do anything yet
            'yvals': self.proc_data_dict['corr_data'],
            'ylabel': 'Excited state population',
            'yunit': '',
            'setlabel': 'data',
            'title': (self.raw_data_dict['timestamp'] + ' ' +
                      self.raw_data_dict['measurementstring']),
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'line fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'cos fit',
                'do_legend': True,
                'legend_pos': 'upper right'}

            self.plot_dicts['text_msg'] = {
                'ax_id': 'main',
                'ypos': 0.15,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'text_string': self.raw_data_dict['scale_factor_msg']}


class Intersect_Analysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analysis to extract the intercept of two parameters.

    relevant options_dict parameters
        ch_idx_A (int) specifies first channel for intercept
        ch_idx_B (int) specifies second channel for intercept if same as first
            it will assume data was taken interleaved.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xvals': 'sweep_points',
                            'xunit': 'sweep_unit',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()


    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_A" and "ch_idx_B"
        specified in the options dict. If ch_idx_A and ch_idx_B are the same
        it will unzip the data.
        """
        self.proc_data_dict = deepcopy(self.raw_data_dict)
        # The channel containing the data must be specified in the options dict
        ch_idx_A = self.options_dict.get('ch_idx_A', 0)
        ch_idx_B = self.options_dict.get('ch_idx_B', 0)


        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx_A]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx_A]

        if ch_idx_A == ch_idx_B:
            yvals = list(self.raw_data_dict['measured_data'].values())[ch_idx_A][0]
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0][::2]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0][1::2]
            self.proc_data_dict['yvals_A'] = yvals[::2]
            self.proc_data_dict['yvals_B'] = yvals[1::2]
        else:
            self.proc_data_dict['xvals_A'] = self.raw_data_dict['xvals'][0]
            self.proc_data_dict['xvals_B'] = self.raw_data_dict['xvals'][0]

            self.proc_data_dict['yvals_A'] = list(self.raw_data_dict
                ['measured_data'].values())[ch_idx_A][0]
            self.proc_data_dict['yvals_B'] = list(self.raw_data_dict
                ['measured_data'].values())[ch_idx_B][0]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_A'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_A']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_A']}}

        self.fit_dicts['line_fit_B'] = {
            'model': lmfit.models.PolynomialModel(degree=2),
            'fit_xvals': {'x': self.proc_data_dict['xvals_B']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_B']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_A'].best_values
        fr_1 = self.fit_res['line_fit_B'].best_values

        c0 = (fr_0['c0'] - fr_1['c0'])
        c1 = (fr_0['c1'] - fr_1['c1'])
        c2 = (fr_0['c2'] - fr_1['c2'])
        poly_coeff = [c0, c1, c2]
        poly = np.polynomial.polynomial.Polynomial([fr_0['c0'],
                                                   fr_0['c1'], fr_0['c2']])
        ic = np.polynomial.polynomial.polyroots(poly_coeff)

        self.proc_data_dict['intersect_L'] = ic[0], poly(ic[0])
        self.proc_data_dict['intersect_R'] = ic[1], poly(ic[1])

        if (((np.min(self.proc_data_dict['xvals']))< ic[0]) and
                ( ic[0] < (np.max(self.proc_data_dict['xvals'])))):
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_L']
        else:
            self.proc_data_dict['intersect'] =self.proc_data_dict['intersect_R']

    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_A'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_A'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'A',
            'title': (self.proc_data_dict['timestamps'][0] + ' \n' +
                      self.proc_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_B'],
            'xlabel': self.proc_data_dict['xlabel'][0],
            'xunit': self.proc_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_B'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'B',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_A'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_A']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit A',
                'do_legend': True}
            self.plot_dicts['line_fit_B'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_B']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit B',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['intersect'][0],
                 self.proc_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['intersect'][0]],
                'yvals': [self.proc_data_dict['intersect'][1]],
                'line_kws': {'alpha': .5, 'color':'gray',
                            'markersize':15},
                'marker': 'o',
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_intersect(self):

        return self.proc_data_dict['intersect']



class CZ_1QPhaseCal_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract the intercept for a single qubit phase calibration
    experiment

    N.B. this is a less generic version of "Intersect_Analysis" and should
    be deprecated (MAR Dec 2017)
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx" in options dict and
        then splits the data for th
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx = self.options_dict['ch_idx']

        yvals = list(self.raw_data_dict['measured_data'].values())[ch_idx][0]

        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][ch_idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][ch_idx]
        self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
        self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]
        self.proc_data_dict['yvals_off'] = yvals[::2]
        self.proc_data_dict['yvals_on'] = yvals[1::2]


    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()

        self.fit_dicts['line_fit_off'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_off']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_off']}}

        self.fit_dicts['line_fit_on'] = {
            'model': lmfit.models.PolynomialModel(degree=1),
            'fit_xvals': {'x': self.proc_data_dict['xvals_on']},
            'fit_yvals': {'data': self.proc_data_dict['yvals_on']}}


    def analyze_fit_results(self):
        fr_0 = self.fit_res['line_fit_off'].best_values
        fr_1 = self.fit_res['line_fit_on'].best_values
        ic = -(fr_0['c0'] - fr_1['c0'])/(fr_0['c1'] - fr_1['c1'])

        self.proc_data_dict['zero_phase_diff_intersect'] = ic


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_off'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_on'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['line_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['line_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['line_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}


            ic, ic_unit = SI_val_to_msg_str(
                self.proc_data_dict['zero_phase_diff_intersect'],
                 self.raw_data_dict['xunit'][0][0], return_type=float)
            self.plot_dicts['intercept_message'] = {
                'ax_id': 'main',
                'plotfn': self.plot_line,
                'xvals': [self.proc_data_dict['zero_phase_diff_intersect']],
                'yvals': [np.mean(self.proc_data_dict['xvals_on'])],
                'line_kws': {'alpha': 0},
                'setlabel': 'Intercept: {:.1f} {}'.format(ic, ic_unit),
                'do_legend': True}

    def get_zero_phase_diff_intersect(self):

        return self.proc_data_dict['zero_phase_diff_intersect']


class Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Very basic analysis to determine the phase of a single oscillation
    that has an assumed period of 360 degrees.
    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        idx = 1

        self.proc_data_dict['yvals'] = list(self.raw_data_dict['measured_data'].values())[idx][0]
        self.proc_data_dict['ylabel'] = self.raw_data_dict['value_names'][0][idx]
        self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod = fit_mods.CosModel
        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.raw_data_dict['xvals'][0],
            data=self.proc_data_dict['yvals'], freq_guess=1/360)
        guess_pars['frequency'].value = 1/360
        guess_pars['frequency'].vary = False
        self.fit_dicts['cos_fit'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.raw_data_dict['xvals'][0]},
            'fit_yvals': {'data': self.proc_data_dict['yvals']},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        fr = self.fit_res['cos_fit'].best_values
        self.proc_data_dict['phi'] =  np.rad2deg(fr['phase'])


    def prepare_plots(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.raw_data_dict['xvals'][0],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals'],
            'ylabel': self.proc_data_dict['ylabel'],
            'yunit': self.proc_data_dict['yunit'],
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit',
                'do_legend': True}


class Conditional_Oscillation_Analysis(ba.BaseDataAnalysis):
    """
    Analysis to extract quantities from a conditional oscillation.

    """
    def __init__(self, t_start: str=None, t_stop: str=None,
                 data_file_path: str=None,
                 label: str='',
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         label=label,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only, do_fitting=do_fitting)
        self.single_timestamp = False

        self.params_dict = {'xlabel': 'sweep_name',
                            'xunit': 'sweep_unit',
                            'xvals': 'sweep_points',
                            'measurementstring': 'measurementstring',
                            'value_names': 'value_names',
                            'value_units': 'value_units',
                            'measured_values': 'measured_values'}

        self.numeric_params = []
        if auto:
            self.run_analysis()

    def process_data(self):
        """
        selects the relevant acq channel based on "ch_idx_osc" and
        "ch_idx_spec" in the options dict and then splits the data for the
        off and on cases
        """
        self.proc_data_dict = OrderedDict()
        # The channel containing the data must be specified in the options dict
        ch_idx_spec = self.options_dict.get('ch_idx_spec', 0)
        ch_idx_osc = self.options_dict.get('ch_idx_osc', 1)
        normalize_to_cal_points = self.options_dict.get('normalize_to_cal_points', True)
        cal_points = [
                        [[-4, -3], [-2, -1]],
                        [[-4, -2], [-3, -1]],
                       ]


        i = 0
        for idx, type_str in zip([ch_idx_osc, ch_idx_spec], ['osc', 'spec']):
            yvals = list(self.raw_data_dict['measured_data'].values())[idx][0]
            self.proc_data_dict['ylabel_{}'.format(type_str)] = self.raw_data_dict['value_names'][0][idx]
            self.proc_data_dict['yunit'] = self.raw_data_dict['value_units'][0][idx]

            if normalize_to_cal_points:
                yvals = a_tools.rotate_and_normalize_data_1ch(yvals,
                    cal_zero_points=cal_points[i][0],
                    cal_one_points=cal_points[i][1])
                i +=1

                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]
                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]

            else:
                self.proc_data_dict['yvals_{}_off'.format(type_str)] = yvals[::2]
                self.proc_data_dict['yvals_{}_on'.format(type_str)] = yvals[1::2]


                self.proc_data_dict['xvals_off'] = self.raw_data_dict['xvals'][0][::2]
                self.proc_data_dict['xvals_on'] = self.raw_data_dict['xvals'][0][1::2]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        cos_mod = fit_mods.CosModel
        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.proc_data_dict['xvals_off'][:-2],
            data=self.proc_data_dict['yvals_osc_off'][:-2],
            freq_guess=1/360)
        guess_pars['frequency'].value = 1/360
        guess_pars['frequency'].vary = False
        self.fit_dicts['cos_fit_off'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.proc_data_dict['xvals_off'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_off'][:-2]},
            'guess_pars': guess_pars}


        cos_mod = fit_mods.CosModel
        guess_pars = fit_mods.Cos_guess(
            model=cos_mod, t=self.proc_data_dict['xvals_on'][:-2],
            data=self.proc_data_dict['yvals_osc_on'][:-2],
            freq_guess=1/360)
        guess_pars['frequency'].value = 1/360
        guess_pars['frequency'].vary = False
        self.fit_dicts['cos_fit_on'] = {
            'fit_fn': fit_mods.CosFunc,
            'fit_xvals': {'t': self.proc_data_dict['xvals_on'][:-2]},
            'fit_yvals': {'data': self.proc_data_dict['yvals_osc_on'][:-2]},
            'guess_pars': guess_pars}

    def analyze_fit_results(self):
        fr_0 = self.fit_res['cos_fit_off'].params
        fr_1 = self.fit_res['cos_fit_on'].params

        phi0 = np.rad2deg(fr_0['phase'].value)
        phi1 = np.rad2deg(fr_1['phase'].value)

        phi0_stderr = np.rad2deg(fr_0['phase'].stderr)
        phi1_stderr = np.rad2deg(fr_1['phase'].stderr)

        self.proc_data_dict['phi_0'] = phi0, phi0_stderr
        self.proc_data_dict['phi_1'] = phi1, phi1_stderr
        phi_cond_stderr = (phi0_stderr**2+phi1_stderr**2)**.5
        self.proc_data_dict['phi_cond'] = (phi1 -phi0), phi_cond_stderr


        osc_amp = np.mean([fr_0['amplitude'], fr_1['amplitude']])
        osc_amp_stderr = np.sqrt(fr_0['amplitude'].stderr**2 +
                                 fr_1['amplitude']**2)/2

        self.proc_data_dict['osc_amp_0'] = (fr_0['amplitude'].value,
                                            fr_0['amplitude'].stderr)
        self.proc_data_dict['osc_amp_1'] = (fr_1['amplitude'].value,
                                            fr_1['amplitude'].stderr)

        self.proc_data_dict['osc_offs_0'] = (fr_0['offset'].value,
                                            fr_0['offset'].stderr)
        self.proc_data_dict['osc_offs_1'] = (fr_1['offset'].value,
                                            fr_1['offset'].stderr)


        offs_stderr = (fr_0['offset'].stderr**2+fr_1['offset'].stderr**2)**.5
        self.proc_data_dict['offs_diff'] = (
            fr_1['offset'].value - fr_0['offset'].value, offs_stderr)

        # self.proc_data_dict['osc_amp'] = (osc_amp, osc_amp_stderr)
        self.proc_data_dict['missing_fraction'] = (
            np.mean(self.proc_data_dict['yvals_spec_on'][:-2]) -
            np.mean(self.proc_data_dict['yvals_spec_off'][:-2]))


    def prepare_plots(self):
        self._prepare_main_oscillation_figure()
        self._prepare_spectator_qubit_figure()

    def _prepare_main_oscillation_figure(self):
        self.plot_dicts['main'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_off'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'main',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_osc_on'],
            'ylabel': self.proc_data_dict['ylabel_osc'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            self.plot_dicts['cos_fit_off'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_off']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ off',
                'do_legend': True}
            self.plot_dicts['cos_fit_on'] = {
                'ax_id': 'main',
                'plotfn': self.plot_fit,
                'fit_res': self.fit_dicts['cos_fit_on']['fit_res'],
                'plot_init': self.options_dict['plot_init'],
                'setlabel': 'Fit CZ on',
                'do_legend': True}

            # offset as a guide for the eye
            y = self.fit_res['cos_fit_off'].params['offset'].value
            self.plot_dicts['cos_off_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C0', 'linestyle': 'dotted'}
                    }

            phase_message = (
                'Phase diff.: {:.1f} $\pm$ {:.1f} deg\n'
                'Phase off: {:.1f} $\pm$ {:.1f}deg\n'
                'Phase on: {:.1f} $\pm$ {:.1f}deg\n'
                'Osc. amp. off: {:.4f} $\pm$ {:.4f}\n'
                'Osc. amp. on: {:.4f} $\pm$ {:.4f}\n'
                'Offs. diff.: {:.4f} $\pm$ {:.4f}\n'
                'Osc. offs. off: {:.4f} $\pm$ {:.4f}\n'
                'Osc. offs. on: {:.4f} $\pm$ {:.4f}'.format(
                    self.proc_data_dict['phi_cond'][0],
                    self.proc_data_dict['phi_cond'][1],
                    self.proc_data_dict['phi_0'][0],
                    self.proc_data_dict['phi_0'][1],
                    self.proc_data_dict['phi_1'][0],
                    self.proc_data_dict['phi_1'][1],
                    self.proc_data_dict['osc_amp_0'][0],
                    self.proc_data_dict['osc_amp_0'][1],
                    self.proc_data_dict['osc_amp_1'][0],
                    self.proc_data_dict['osc_amp_1'][1],
                    self.proc_data_dict['offs_diff'][0],
                    self.proc_data_dict['offs_diff'][1],
                    self.proc_data_dict['osc_offs_0'][0],
                    self.proc_data_dict['osc_offs_0'][1],
                    self.proc_data_dict['osc_offs_1'][0],
                    self.proc_data_dict['osc_offs_1'][1]))
            self.plot_dicts['phase_message'] = {
                'ax_id': 'main',
                'ypos': 0.9,
                'xpos': 1.45,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': phase_message}

    def _prepare_spectator_qubit_figure(self):

        self.plot_dicts['spectator_qubit'] = {
            'plotfn': self.plot_line,
            'xvals': self.proc_data_dict['xvals_off'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_off'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ off',
            'title': (self.raw_data_dict['timestamps'][0] + ' \n' +
                      self.raw_data_dict['measurementstring'][0]),
            'do_legend': True,
            # 'yrange': (0,1),
            'legend_pos': 'upper right'}

        self.plot_dicts['spec_on'] = {
            'plotfn': self.plot_line,
            'ax_id': 'spectator_qubit',
            'xvals': self.proc_data_dict['xvals_on'],
            'xlabel': self.raw_data_dict['xlabel'][0],
            'xunit': self.raw_data_dict['xunit'][0][0],
            'yvals': self.proc_data_dict['yvals_spec_on'],
            'ylabel': self.proc_data_dict['ylabel_spec'],
            'yunit': self.proc_data_dict['yunit'],
            'setlabel': 'CZ on',
            'do_legend': True,
            'legend_pos': 'upper right'}

        if self.do_fitting:
            leak_msg = (
                'Missing fraction: {:.2f} % '.format(
                    self.proc_data_dict['missing_fraction']*100))
            self.plot_dicts['leak_msg'] = {
                'ax_id': 'spectator_qubit',
                'ypos': 0.7,
                'plotfn': self.plot_text,
                'box_props': 'fancy',
                'line_kws': {'alpha': 0},
                'text_string': leak_msg}
            # offset as a guide for the eye
            y = self.fit_res['cos_fit_on'].params['offset'].value
            self.plot_dicts['cos_on_offset'] ={
                'plotfn': self.plot_matplot_ax_method,
                'ax_id':'main',
                'func': 'axhline',
                'plot_kws': {
                    'y': y, 'color': 'C1', 'linestyle': 'dotted'}
                    }


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
        self.data_type = self.options_dict['data_type']
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
        tomography_qubits = self.options_dict.get('tomography_qubits', None)
        data, Fs, Omega = self.base_analysis.measurement_operators_and_results(
                              tomography_qubits)
        if 'data_filter' in self.options_dict:
            data = self.options_dict['data_filter'](data.T).T

        data = data.T
        for i, v in enumerate(data):
            data[i] = v / v.sum()
        data = data.T

        Fs = self.options_dict.get('meas_operators', Fs)
        Fs = [qtp.Qobj(F) for F in Fs]
        d = Fs[0].shape[0]
        self.proc_data_dict['d'] = d
        Omega = self.options_dict.get('covar_matrix', Omega)
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

        if self.options_dict.get('pauli_values', False):
            rho_pauli = tomo.pauli_values_tomography(all_mus,Fs,basis_rots_str)
            self.proc_data_dict['rho_raw'] = rho_pauli
            self.proc_data_dict['rho'] = rho_pauli
        elif self.options_dict.get('pauli_raw', False):
            pauli_raw = self.generate_raw_pauli_set()
            rho_raw = tomo.pauli_set_to_density_matrix(pauli_raw)
            self.proc_data_dict['rho_raw'] = rho_raw
            self.proc_data_dict['rho'] = rho_raw
        elif self.options_dict.get('imle', False):
            it = metadata.get('iterations', None)
            it = self.options_dict.get('iterations', it)
            tol = metadata.get('tolerance', None)
            tol = self.options_dict.get('tolerance', tol)
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
            if self.options_dict.get('mle', False):
                rho_mle = tomo.mle_tomography(
                    all_mus, all_Fs,
                    all_Omegas if self.get_param_value('use_covariance_matrix', False) else None,
                    rho_guess=rho_ls)
                self.proc_data_dict['rho_mle'] = rho_mle
                self.proc_data_dict['rho'] = rho_mle

        rho = self.proc_data_dict['rho']
        self.proc_data_dict['purity'] = (rho * rho).tr().real

        rho_target = metadata.get('rho_target', None)
        rho_target = self.options_dict.get('rho_target', rho_target)
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
        self.tight_fig = self.options_dict.get('tight_fig', False)
        rho_target = self.raw_data_dict['exp_metadata'].get('rho_target', None)
        rho_target = self.options_dict.get('rho_target', rho_target)
        d = self.proc_data_dict['d']
        xtick_labels = self.options_dict.get('rho_ticklabels', None)
        ytick_labels = self.options_dict.get('rho_ticklabels', None)
        if 2 ** (d.bit_length() - 1) == d:
            nr_qubits = d.bit_length() - 1
            fmt_string = '{{:0{}b}}'.format(nr_qubits)
            labels = [fmt_string.format(i) for i in range(2 ** nr_qubits)]
            if xtick_labels is None:
                xtick_labels = ['$|' + lbl + r'\rangle$' for lbl in labels]
            if ytick_labels is None:
                ytick_labels = [r'$\langle' + lbl + '|$' for lbl in labels]
        color = (0.5 * np.angle(self.proc_data_dict['rho'].full()) / np.pi) % 1.
        cmap = self.options_dict.get('rho_colormap', self.default_phase_cmap())
        if self.options_dict.get('pauli_raw', False):
            title = 'Density matrix reconstructed from the Pauli (raw) set\n'
        elif self.options_dict.get('pauli_values', False):
            title = 'Density matrix reconstructed from the Pauli set\n'
        elif self.options_dict.get('mle', False):
            title = 'Maximum likelihood fit of the density matrix\n'
        elif self.options_dict.get('it_mle', False):
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
        pauli_Fs = list(itertools.chain(*np.array(pauli_Fs, dtype=np.object).T))

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
        if self.options_dict.get('pauli_raw', False):
            fit_type = 'raw counts'
        elif self.options_dict.get('pauli_values', False):
            fit_type = 'corrected counts'
        elif self.options_dict.get('mle', False):
            fit_type = 'maximum likelihood estimation'
        elif self.options_dict.get('imle', False):
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
        rho_target = self.options_dict.get('rho_target', rho_target)
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


class ReadoutROPhotonsAnalysis(Single_Qubit_TimeDomainAnalysis):
    """
    Analyses the photon number in the RO based on the
    readout_photons_in_resonator function

    function specific options for options dict:
    f_qubit
    chi
    artif_detuning
    print_fit_results
    """

    def __init__(self, t_start: str=None, t_stop: str=None,
                 label: str='', data_file_path: str=None,
                 close_figs: bool=False, options_dict: dict=None,
                 extract_only: bool=False, do_fitting: bool=False,
                 auto: bool=True):
        super().__init__(t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         close_figs=close_figs, label=label,
                         extract_only=extract_only, do_fitting=do_fitting)
        if self.options_dict.get('TwoD', None) is None:
            self.options_dict['TwoD'] = True
        self.label = label
        self.params_dict = {
            'measurementstring': 'measurementstring',
            'sweep_points': 'sweep_points',
            'sweep_points_2D': 'sweep_points_2D',
            'value_names': 'value_names',
            'value_units': 'value_units',
            'measured_values': 'measured_values'}

        self.numeric_params = self.options_dict.get('numeric_params',
                                                   OrderedDict())

        self.kappa = self.options_dict.get('kappa_effective', None)
        self.chi = self.options_dict.get('chi', None)
        self.T2 = self.options_dict.get('T2echo', None)
        self.artif_detuning = self.options_dict.get('artif_detuning', 0)

        if (self.kappa is None) or (self.chi is None) or (self.T2 is None):
            raise ValueError('kappa_effective, chi and T2echo must be passed to '
                             'the options_dict.')

        if auto:
            self.run_analysis()

    def process_data(self):
        self.proc_data_dict = OrderedDict()
        self.proc_data_dict['qubit_state'] = [[],[]]
        self.proc_data_dict['delay_to_relax'] = self.raw_data_dict[
                                                    'sweep_points_2D'][0]
        self.proc_data_dict['ramsey_times'] = []

        for i,x in enumerate(np.transpose(self.raw_data_dict[
                        'measured_data']['raw w0 _measure'][0])):
            self.proc_data_dict['qubit_state'][0].append([])
            self.proc_data_dict['qubit_state'][1].append([])

            for j,y in enumerate(np.transpose(self.raw_data_dict[
                    'measured_data']['raw w0 _measure'][0])[i]):

                if j%2 == 0:
                    self.proc_data_dict['qubit_state'][0][i].append(y)

                else:
                    self.proc_data_dict['qubit_state'][1][i].append(y)
        for i,x in enumerate( self.raw_data_dict['sweep_points'][0]):
            if i % 2 == 0:
                self.proc_data_dict['ramsey_times'].append(x)

    #I STILL NEED to pass Chi
    def prepare_fitting(self):
        self.proc_data_dict['photon_number'] = [[],[]]
        self.proc_data_dict['fit_results'] = []
        self.proc_data_dict['ramsey_fit_results'] = [[],[]]


        for i,tau in enumerate(self.proc_data_dict['delay_to_relax']):

            self.proc_data_dict['ramsey_fit_results'][0].append(self.fit_Ramsey(
                            self.proc_data_dict['ramsey_times'][:-4],
                            self.proc_data_dict['qubit_state'][0][i][:-4]/
                            max(self.proc_data_dict['qubit_state'][0][i][:-4]),
                            state=0,
                            kw=self.options_dict))

            self.proc_data_dict['ramsey_fit_results'][1].append(self.fit_Ramsey(
                            self.proc_data_dict['ramsey_times'][:-4],
                            self.proc_data_dict['qubit_state'][1][i][:-4]/
                            max(self.proc_data_dict['qubit_state'][1][i][:-4]),
                            state=1,
                            kw=self.options_dict))

            n01 = self.proc_data_dict['ramsey_fit_results'
                                         ][0][i][0].params['n0'].value
            n02 = self.proc_data_dict['ramsey_fit_results'
                                         ][1][i][0].params['n0'].value

            self.proc_data_dict['photon_number'][0].append(n01)
            self.proc_data_dict['photon_number'][1].append(n02)


    def run_fitting(self):
        print_fit_results = self.params_dict.pop('print_fit_results',False)

        exp_dec_mod = lmfit.Model(fit_mods.ExpDecayFunc)
        exp_dec_mod.set_param_hint('n',
                                   value=1,
                                   vary=False)
        exp_dec_mod.set_param_hint('offset',
                                   value=0,
                                   min=0,
                                   vary=True)
        exp_dec_mod.set_param_hint('tau',
                                   value=self.proc_data_dict[
                                                'delay_to_relax'][-1],
                                   min=1e-11,
                                   vary=True)
        exp_dec_mod.set_param_hint('amplitude',
                                   value=1,
                                   min=0,
                                   vary=True)
        params = exp_dec_mod.make_params()
        self.fit_res = OrderedDict()
        self.fit_res['ground_state'] = exp_dec_mod.fit(
                                data=self.proc_data_dict['photon_number'][0],
                                params=params,
                                t=self.proc_data_dict['delay_to_relax'])
        self.fit_res['excited_state'] = exp_dec_mod.fit(
                                data=self.proc_data_dict['photon_number'][1],
                                params=params,
                                t=self.proc_data_dict['delay_to_relax'])
        if print_fit_results:
            print(self.fit_res['ground_state'].fit_report())
            print(self.fit_res['excited_state'].fit_report())

    def fit_Ramsey(self, x, y, state, **kw):

        x = np.array(x)

        y = np.array(y)

        exp_dec_p_mod = lmfit.Model(fit_mods.ExpDecayPmod)
        comb_exp_dec_mod = lmfit.Model(fit_mods.CombinedOszExpDecayFunc)

        average = np.mean(y)

        ft_of_data = np.fft.fft(y)
        index_of_fourier_maximum = np.argmax(np.abs(
            ft_of_data[1:len(ft_of_data) // 2])) + 1
        max_ramsey_delay = x[-1] - x[0]

        fft_axis_scaling = 1 / max_ramsey_delay
        freq_est = fft_axis_scaling * index_of_fourier_maximum

        n_est = (freq_est-self.artif_detuning)/(2 * self.chi)


        exp_dec_p_mod.set_param_hint('T2echo',
                                   value=self.T2,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('offset',
                                   value=average,
                                   min=0,
                                   vary=True)
        exp_dec_p_mod.set_param_hint('delta',
                                   value=self.artif_detuning,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('amplitude',
                                   value=1,
                                   min=0,
                                   vary=True)
        exp_dec_p_mod.set_param_hint('kappa',
                                   value=self.kappa[state],
                                   vary=False)
        exp_dec_p_mod.set_param_hint('chi',
                                   value=self.chi,
                                   vary=False)
        exp_dec_p_mod.set_param_hint('n0',
                                      value=n_est,
                                      min=0,
                                      vary=True)
        exp_dec_p_mod.set_param_hint('phase',
                                       value=0,
                                       vary=True)


        comb_exp_dec_mod.set_param_hint('tau',
                                     value=self.T2,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('offset',
                                        value=average,
                                        min=0,
                                        vary=True)
        comb_exp_dec_mod.set_param_hint('oscillation_offset',
                                        value=average,
                                        min=0,
                                        vary=True)
        comb_exp_dec_mod.set_param_hint('amplitude',
                                     value=1,
                                     min=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('tau_gauss',
                                     value=self.kappa[state],
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('n0',
                                     value=n_est,
                                     min=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('phase',
                                     value=0,
                                     vary=True)
        comb_exp_dec_mod.set_param_hint('delta',
                                     value=self.artif_detuning,
                                     vary=False)
        comb_exp_dec_mod.set_param_hint('chi',
                                     value=self.chi,
                                     vary=False)

        if (np.average(y[:4]) >
                np.average(y[4:8])):
            phase_estimate = 0
        else:
            phase_estimate = np.pi
        exp_dec_p_mod.set_param_hint('phase',
                                     value=phase_estimate, vary=True)
        comb_exp_dec_mod.set_param_hint('phase',
                                     value=phase_estimate, vary=True)

        amplitude_guess = 0.5
        if np.all(np.logical_and(y >= 0, y <= 1)):
            exp_dec_p_mod.set_param_hint('amplitude',
                                         value=amplitude_guess,
                                         min=0.00,
                                         max=4.0,
                                         vary=True)
            comb_exp_dec_mod.set_param_hint('amplitude',
                                         value=amplitude_guess,
                                         min=0.00,
                                         max=4.0,
                                         vary=True)

        else:
            print('data is not normalized, varying amplitude')
            exp_dec_p_mod.set_param_hint('amplitude',
                                         value=max(y),
                                         min=0.00,
                                         max=4.0,
                                         vary=True)
            comb_exp_dec_mod.set_param_hint('amplitude',
                                        value=max(y),
                                        min=0.00,
                                        max=4.0,
                                        vary=True)

        fit_res_1 = exp_dec_p_mod.fit(data=y,
                                    t=x,
                                    params= exp_dec_p_mod.make_params())

        fit_res_2 = comb_exp_dec_mod.fit(data=y,
                                         t=x,
                                         params= comb_exp_dec_mod.make_params())


        if fit_res_1.chisqr > .35:
            log.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2*np.pi, 10):

                for i, del_amp in enumerate(np.linspace(
                        -max(y)/10, max(y)/10, 10)):
                    exp_dec_p_mod.set_param_hint('phase',
                                                 value=phase_estimate,
                                                 vary=False)
                    exp_dec_p_mod.set_param_hint('amplitude',
                                                 value=max(y)+ del_amp)

                    fit_res_lst += [exp_dec_p_mod.fit(
                        data=y,
                        t=x,
                        params= exp_dec_p_mod.make_params())]

            chisqr_lst = [fit_res_1.chisqr for fit_res_1 in fit_res_lst]
            fit_res_1 = fit_res_lst[np.argmin(chisqr_lst)]

        if fit_res_2.chisqr > .35:
            log.warning('Fit did not converge, varying phase')
            fit_res_lst = []

            for phase_estimate in np.linspace(0, 2*np.pi, 10):

                for i, del_amp in enumerate(np.linspace(
                        -max(y)/10, max(y)/10, 10)):
                    comb_exp_dec_mod.set_param_hint('phase',
                                                 value=phase_estimate,
                                                 vary=False)
                    comb_exp_dec_mod.set_param_hint('amplitude',
                                                 value=max(y)+ del_amp)

                    fit_res_lst += [comb_exp_dec_mod.fit(
                        data=y,
                        t=x,
                        params= comb_exp_dec_mod.make_params())]

            chisqr_lst = [fit_res_2.chisqr for fit_res_2 in fit_res_lst]
            fit_res_2 = fit_res_lst[np.argmin(chisqr_lst)]

        if fit_res_1.chisqr < fit_res_2.chisqr:
            self.proc_data_dict['params'] = exp_dec_p_mod.make_params()
            return [fit_res_1,fit_res_1,fit_res_2]
        else:
            self.proc_data_dict['params'] = comb_exp_dec_mod.make_params()
            return [fit_res_2,fit_res_1,fit_res_2]


    def prepare_plots(self):
            self.prepare_2D_sweep_plot()
            self.prepare_photon_number_plot()
            self.prepare_ramsey_plots()

    def prepare_2D_sweep_plot(self):
        self.plot_dicts['off_full_data_'+self.label] = {
            'title': 'Raw data |g>',
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['ramsey_times'],
            'xlabel': 'Ramsey delays',
            'xunit': 's',
            'yvals': self.proc_data_dict['delay_to_relax'],
            'ylabel': 'Delay after first RO-pulse',
            'yunit': 's',
            'zvals': np.array(self.proc_data_dict['qubit_state'][0]) }

        self.plot_dicts['on_full_data_'+self.label] = {
            'title': 'Raw data |e>',
            'plotfn': self.plot_colorxy,
            'xvals': self.proc_data_dict['ramsey_times'],
            'xlabel': 'Ramsey delays',
            'xunit': 's',
            'yvals': self.proc_data_dict['delay_to_relax'],
            'ylabel': 'Delay after first RO-pulse',
            'yunit': 's',
            'zvals': np.array(self.proc_data_dict['qubit_state'][1])  }



    def prepare_ramsey_plots(self):
        x_fit = np.linspace(self.proc_data_dict['ramsey_times'][0],
                            max(self.proc_data_dict['ramsey_times']),101)
        for i in range(len(self.proc_data_dict['ramsey_fit_results'][0])):

            self.plot_dicts['off_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+\
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['ramsey_times'],
                'xlabel': 'Ramsey delays',
                'xunit': 's',
                'yvals': np.array(self.proc_data_dict['qubit_state'][0][i]/
                             max(self.proc_data_dict['qubit_state'][0][i][:-4])),
                'ylabel': 'Measured qubit state',
                'yunit': '',
                'marker': 'o',
                'setlabel': '|g> data_'+str(i),
                'do_legend': True }

            self.plot_dicts['off_fit_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][0][i][1].eval(
                    self.proc_data_dict['ramsey_fit_results'][0][i][1].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|g> fit_model'+str(i),
                'do_legend': True  }

            self.plot_dicts['off_fit_2_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |g> state',
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][0][i][2].eval(
                    self.proc_data_dict['ramsey_fit_results'][0][i][2].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|g> fit_simpel_model'+str(i),
                'do_legend': True  }

            self.plot_dicts['hidden_g_'+str(i)] = {
                'ax_id':'ramsey_off_'+str(i),
                'plotfn': self.plot_line,
                'xvals': [0],
                'yvals': [0],
                'color': 'w',
                'setlabel': 'Residual photon count = '
                             ''+str(self.proc_data_dict['photon_number'][0][i]),
                'do_legend': True }


            self.plot_dicts['on_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict['ramsey_times'],
                'xlabel': 'Ramsey delays',
                'xunit': 's',
                'yvals':  np.array(self.proc_data_dict['qubit_state'][1][i]/
                             max(self.proc_data_dict['qubit_state'][1][i][:-4])),
                'ylabel': 'Measured qubit state',
                'yunit': '',
                'marker': 'o',
                'setlabel': '|e> data_'+str(i),
                'do_legend': True }

            self.plot_dicts['on_fit_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][1][i][1].eval(
                    self.proc_data_dict['ramsey_fit_results'][1][i][1].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|e> fit_model'+str(i),
                'do_legend': True }

            self.plot_dicts['on_fit_2_'+str(i)] = {
                'title': 'Ramsey w t_delay = '+ \
                         str(self.proc_data_dict['delay_to_relax'][i])+ \
                         ' s, in |e> state',
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': x_fit,
                'yvals':  self.proc_data_dict['ramsey_fit_results'][1][i][2].eval(
                    self.proc_data_dict['ramsey_fit_results'][1][i][2].params,
                    t=x_fit),
                'linestyle': '-',
                'marker': '',
                'setlabel': '|e> fit_simpel_model'+str(i),
                'do_legend': True }

            self.plot_dicts['hidden_e_'+str(i)] = {
                'ax_id':'ramsey_on_'+str(i),
                'plotfn': self.plot_line,
                'xvals': [0],
                'yvals': [0],
                'color': 'w',
                'setlabel': 'Residual photon count = '
                            ''+str(self.proc_data_dict['photon_number'][1][i]),
                'do_legend': True }


    def prepare_photon_number_plot(self):


        ylabel = 'Average photon number'
        yunit = ''

        x_fit = np.linspace(min(self.proc_data_dict['delay_to_relax']),
                            max(self.proc_data_dict['delay_to_relax']),101)
        minmax_data = [min(min(self.proc_data_dict['photon_number'][0]),
                           min(self.proc_data_dict['photon_number'][1])),
                       max(max(self.proc_data_dict['photon_number'][0]),
                           max(self.proc_data_dict['photon_number'][1]))]
        minmax_data[0] -= minmax_data[0]/5
        minmax_data[1] += minmax_data[1]/5

        self.proc_data_dict['photon_number'][1],

        self.fit_res['excited_state'].eval(
            self.fit_res['excited_state'].params,
            t=x_fit)
        self.plot_dicts['Photon number count'] = {
            'plotfn': self.plot_line,
            'xlabel': 'Delay after first RO-pulse',
            'ax_id': 'Photon number count ',
            'xunit': 's',
            'xvals': self.proc_data_dict['delay_to_relax'],
            'yvals': self.proc_data_dict['photon_number'][0],
            'ylabel': ylabel,
            'yunit': yunit,
            'yrange': minmax_data,
            'title': 'Residual photon number',
            'color': 'b',
            'linestyle': '',
            'marker': 'o',
            'setlabel': '|g> data',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main2'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'xvals': x_fit,
            'yvals': self.fit_res['ground_state'].eval(
                self.fit_res['ground_state'].params,
                t=x_fit),
            'yrange': minmax_data,
            'ax_id': 'Photon number count ',
            'color': 'b',
            'linestyle': '-',
            'marker': '',
            'setlabel': '|g> fit',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main3'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'xvals': self.proc_data_dict['delay_to_relax'],
            'yvals': self.proc_data_dict['photon_number'][1],
            'yrange': minmax_data,
            'ax_id': 'Photon number count ',
            'color': 'r',
            'linestyle': '',
            'marker': 'o',
            'setlabel': '|e> data',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['main4'] = {
            'plotfn': self.plot_line,
            'xunit': 's',
            'ax_id': 'Photon number count ',
            'xvals': x_fit,
            'yvals': self.fit_res['excited_state'].eval(
                self.fit_res['excited_state'].params,
                t=x_fit),
            'yrange': minmax_data,
            'ylabel': ylabel,
            'color': 'r',
            'linestyle': '-',
            'marker': '',
            'setlabel': '|e> fit',
            'func': 'semilogy',
            'do_legend': True}

        self.plot_dicts['hidden_1'] = {
            'ax_id': 'Photon number count ',
            'plotfn': self.plot_line,
            'yrange': minmax_data,
            'xvals': [0],
            'yvals': [0],
            'color': 'w',
            'setlabel': 'tau_g = '
                        ''+str("%.3f" %
                        (self.fit_res['ground_state'].params['tau'].value*1e9))+''
                        ' ns',
            'do_legend': True }


        self.plot_dicts['hidden_2'] = {
            'ax_id': 'Photon number count ',
            'plotfn': self.plot_line,
            'yrange': minmax_data,
            'xvals': [0],
            'yvals': [0],
            'color': 'w',
            'setlabel': 'tau_e = '
                        ''+str("%.3f" %
                        (self.fit_res['excited_state'].params['tau'].value*1e9))+''
                        ' ns',
            'do_legend': True}


class RODynamicPhaseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names: list=None,  t_start: str=None, t_stop: str=None,
                 data_file_path: str=None, single_timestamp: bool=False,
                 options_dict: dict=None, extract_only: bool=False,
                 do_fitting: bool=True, auto=True):

        super().__init__(qb_names=qb_names, t_start=t_start, t_stop=t_stop,
                         data_file_path=data_file_path,
                         options_dict=options_dict,
                         extract_only=extract_only,
                         do_fitting=do_fitting,
                         auto=False)

        if auto:
            self.run_analysis()

    def process_data(self):

        super().process_data()

        if 'qbp_name' in self.metadata:
            self.pulsed_qbname = self.metadata['qbp_name']
        else:
            self.pulsed_qbname = self.options_dict.get('pulsed_qbname')
        self.measured_qubits = [qbn for qbn in self.channel_map if
                                qbn != self.pulsed_qbname]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.measured_qubits:
            ro_dict = self.proc_data_dict['projected_data_dict'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            for ro_suff, data in ro_dict.items():
                cos_mod = lmfit.Model(fit_mods.CosFunc)
                if self.num_cal_points != 0:
                    data = data[:-self.num_cal_points]
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod,
                    t=sweep_points,
                    data=data)
                guess_pars['amplitude'].vary = True
                guess_pars['offset'].vary = True
                guess_pars['frequency'].vary = True
                guess_pars['phase'].vary = True

                key = 'cos_fit_{}{}'.format(qbn, ro_suff)
                self.fit_dicts[key] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):

        self.dynamic_phases = OrderedDict()
        for meas_qbn in self.measured_qubits:
            self.dynamic_phases[meas_qbn] = \
                (self.fit_dicts['cos_fit_{}_measure'.format(meas_qbn)][
                    'fit_res'].best_values['phase'] -
                 self.fit_dicts['cos_fit_{}_ref_measure'.format(meas_qbn)][
                    'fit_res'].best_values['phase'])*180/np.pi

    def prepare_plots(self):

        super().prepare_plots()

        if self.do_fitting:
            for meas_qbn in self.measured_qubits:
                sweep_points_dict = self.proc_data_dict['sweep_points_dict'][
                    meas_qbn]
                if self.num_cal_points != 0:
                    yvals = [self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_ref_measure'][:-self.num_cal_points],
                             self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_measure'][:-self.num_cal_points]]
                    sweep_points = sweep_points_dict['msmt_sweep_points']

                    # plot cal points
                    for i, cal_pts_idxs in enumerate(
                            self.cal_states_dict[meas_qbn].values()):
                        key = list(self.cal_states_dict[meas_qbn])[i] + meas_qbn
                        self.plot_dicts[key] = {
                            'fig_id': 'dyn_phase_plot_' + meas_qbn,
                            'plotfn': self.plot_line,
                            'xvals': np.mean([
                                sweep_points_dict['cal_points_sweep_points'][
                                    cal_pts_idxs],
                                sweep_points_dict['cal_points_sweep_points'][
                                    cal_pts_idxs]],
                                axis=0),
                            'yvals': np.mean([
                                self.proc_data_dict['projected_data_dict'][meas_qbn][
                                    '_ref_measure'][cal_pts_idxs],
                                self.proc_data_dict['projected_data_dict'][meas_qbn][
                                    '_measure'][cal_pts_idxs]],
                                             axis=0),
                            'setlabel': list(self.cal_states_dict[meas_qbn])[i],
                            'do_legend': True,
                            'legend_bbox_to_anchor': (1, 0.5),
                            'legend_pos': 'center left',
                            'linestyle': 'none',
                            'line_kws': {'color': self.get_cal_state_color(
                                list(self.cal_states_dict[meas_qbn])[i])}}

                else:
                    yvals = [self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_ref_measure'],
                             self.proc_data_dict['projected_data_dict'][meas_qbn][
                                 '_measure']]
                    sweep_points = sweep_points_dict['sweep_points']

                self.plot_dicts['dyn_phase_plot_' + meas_qbn] = {
                    'plotfn': self.plot_line,
                    'xvals': [sweep_points, sweep_points],
                    'xlabel': self.raw_data_dict['xlabel'][0],
                    'xunit': self.raw_data_dict['xunit'][0][0],
                    'yvals': yvals,
                    'ylabel': 'Excited state population',
                    'yunit': '',
                    'setlabel': ['with measurement', 'no measurement'],
                    'title': (self.raw_data_dict['timestamps'][0] + ' ' +
                              self.raw_data_dict['measurementstring'][0]),
                    'linestyle': 'none',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                self.plot_dicts['cos_fit_' + meas_qbn + '_ref_measure'] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['cos_fit_{}_ref_measure'.format(
                                    meas_qbn)]['fit_res'],
                    'setlabel': 'cos fit',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                self.plot_dicts['cos_fit_' + meas_qbn + '_measure'] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['cos_fit_{}_measure'.format(
                                    meas_qbn)]['fit_res'],
                    'setlabel': 'cos fit',
                    'do_legend': True,
                    'legend_bbox_to_anchor': (1, 0.5),
                    'legend_pos': 'center left'}

                textstr = 'Dynamic phase = {:.2f}'.format(
                    self.dynamic_phases[meas_qbn]) + r'$^{\circ}$'
                self.plot_dicts['text_msg_' + meas_qbn] = {
                    'fig_id': 'dyn_phase_plot_' + meas_qbn,
                    'ypos': -0.175,
                    'xpos': 0.5,
                    'horizontalalignment': 'center',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class FluxAmplitudeSweepAnalysis(MultiQubit_TimeDomain_Analysis):
    def __init__(self, qb_names, *args, **kwargs):
        self.mask_freq = kwargs.pop('mask_freq', None)
        self.mask_amp = kwargs.pop('mask_amp', None)

        super().__init__(qb_names, *args, **kwargs)

    def extract_data(self):
        super().extract_data()
        # Set some default values specific to FluxPulseScopeAnalysis if the
        # respective options have not been set by the user or in the metadata.
        # (We do not do this in the init since we have to wait until
        # metadata has been extracted.)
        if self.get_param_value('rotation_type', default_value=None) is None:
            self.options_dict['rotation_type'] = 'global_PCA'
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

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
                if self.options_dict.get('scatter', True):
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

        pdd['mask'] = {qb: np.ones(nr_amps, dtype=np.bool)
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
                    r'Derived qubit frequency'

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
                    'ylabel': r'Pop. loss @ {:.0f} ns'.format(
                        self.lengths[qb][0]/1e-9
                    ),
                    'yunit': '',
                }

            # Plot all fits in single figure
            if self.options_dict.get('all_fits', False) and self.do_fitting:
                colormap = self.options_dict.get('colormap', mpl.cm.Blues)
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
                        'plot_init': self.options_dict.get('plot_init', False),
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
            deepcopy(pdd['data_to_fit'][qb][
                     :, :pdd['data_to_fit'][qb].shape[1]-nr_cp]).flatten(),
            (nr_amps, nr_lengths, nr_phases)) for qb in self.qb_names}

        pdd['data_reshaped_no_cp'] = data_reshaped_no_cp
        if self.metadata['use_cal_points']:
            pdd['cal_point_data'] = {qb: deepcopy(
                pdd['data_to_fit'][qb][
                len(pdd['data_to_fit'][qb])-nr_cp:]) for qb in self.qb_names}

        pdd['mask'] = {qb: np.ones(nr_amps, dtype=np.bool)
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
        nr_lengths = len(self.metadata['flux_lengths'])
        nr_amps = len(self.metadata['amplitudes'])

        for qb in self.qb_names:
            pdd['phase_contrast'][qb] = {}
            exp_mod = fit_mods.ExponentialModel()
            for i in range(nr_amps):
                pdd['phase_contrast'][qb][f'amp_{i}'] = np.array([self.fit_res[
                                                        f'cos_fit_{qb}_{i}_{j}'
                                                    ].best_values['amplitude']
                                                    for j in
                                                    range(nr_lengths)])

                self.fit_dicts[f'exp_fit_{qb}_{i}'] = {
                    'model': exp_mod,
                    'fit_xvals': {'x': self.metadata['flux_lengths']},
                    'fit_yvals': {'data': np.array([self.fit_res[
                                                        f'cos_fit_{qb}_{i}_{j}'
                                                    ].best_values['amplitude']
                                                    for j in
                                                    range(nr_lengths)])}}

            self.run_fitting()

            pdd['T2'][qb] = np.array([
                abs(self.fit_res[f'exp_fit_{qb}_{i}'].best_values['decay'])
                for i in range(len(self.metadata['amplitudes']))])

            pdd['mask'][qb] = []
            for i in range(len(self.metadata['amplitudes'])):
                try:
                    if self.fit_res[f'exp_fit_{qb}_{i}']\
                                            .params['decay'].stderr >= 1e-5:
                        pdd['mask'][qb][i] = False
                except TypeError:
                    pdd['mask'][qb][i] = False

    def prepare_plots(self):
        pdd = self.proc_data_dict
        rdd = self.raw_data_dict

        for qb in self.qb_names:
            mask = pdd['mask'][qb]
            label = f'T2_fit_{qb}'
            xvals = self.metadata['amplitudes'][mask] if \
                self.metadata['frequencies'] is None else \
                self.metadata['frequencies'][mask]
            xlabel = r'Flux pulse amplitude' if \
                self.metadata['frequencies'] is None else \
                r'Derived qubit frequency'
            self.plot_dicts[label] = {
                'plotfn': self.plot_line,
                'linestyle': '-',
                'xvals': xvals,
                'yvals': pdd['T2'][qb][mask],
                'xlabel': xlabel,
                'xunit': 'V' if self.metadata['frequencies'] is None else 'Hz',
                'ylabel': r'T2',
                'yunit': 's',
                'color': 'blue',
            }

            # Plot all fits in single figure
            if not self.options_dict.get('all_fits', False):
                continue

            colormap = self.options_dict.get('colormap', mpl.cm.Blues)
            for i in range(len(self.metadata['amplitudes'])):
                color = colormap(i/(len(self.metadata['frequencies'])-1))
                label = f'exp_fit_{qb}_amp_{i}'
                freqs = self.metadata['frequencies'] is not None
                fitid = self.metadata.get('frequencies',
                                          self.metadata['amplitudes'])[i]
                self.plot_dicts[label] = {
                    'title': rdd['measurementstring'] +
                            '\n' + rdd['timestamp'],
                    'ax_id': f'T2_fits_{qb}',
                    'xlabel': r'Flux pulse length',
                    'xunit': 's',
                    'ylabel': r'Excited state population',
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_res[label],
                    'plot_init': self.options_dict.get('plot_init', False),
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                    'do_legend': False,
                    'legend_bbox_to_anchor': (1, 1),
                    'legend_pos': 'upper left',
                    }

                label = f'freq_scatter_{qb}_{i}'
                self.plot_dicts[label] = {
                    'ax_id': f'T2_fits_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': self.metadata['phases'],
                    'linestyle': '',
                    'yvals': pdd['data_reshaped_no_cp'][qb][i,:],
                    'color': color,
                    'setlabel': f'freq={fitid:.4f}' if freqs
                                        else f'amp={fitid:.4f}',
                }


class MeasurementInducedDephasingAnalysis(MultiQubit_TimeDomain_Analysis):
    def process_data(self):
        super().process_data()

        rdd = self.raw_data_dict
        pdd = self.proc_data_dict

        pdd['data_reshaped'] = {qb: [] for qb in pdd['data_to_fit']}
        pdd['amps_reshaped'] = np.unique(self.metadata['hard_sweep_params']['ro_amp_scale']['values'])
        pdd['phases_reshaped'] = []
        for amp in pdd['amps_reshaped']:
            mask = self.metadata['hard_sweep_params']['ro_amp_scale']['values'] == amp
            pdd['phases_reshaped'].append(self.metadata['hard_sweep_params']['phase']['values'][mask])
            for qb in self.qb_names:
                pdd['data_reshaped'][qb].append(pdd['data_to_fit'][qb][:len(mask)][mask])

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
                'fit_xvals': {'x': pdd['amps_reshaped']},
                'fit_yvals': {'data': pdd['phase_contrast'][qb]}}

            quadratic_mod = lmfit.models.QuadraticModel()
            self.fit_dicts[f'phase_offset_fit_{qb}'] = {
                'model': quadratic_mod,
                'guess_dict': {'b': {'value': 0, 'vary': False}},
                'fit_xvals': {'x': pdd['amps_reshaped']},
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
                    'yvals': pdd['amps_reshaped'],
                    'zvals': pdd['data_reshaped'][qb],
                    'xlabel': r'Pulse phase, $\phi$',
                    'xunit': 'deg',
                    'ylabel': r'Readout pulse amplitude scale, $V_{RO}/V_{ref}$',
                    'yunit': '',
                    'zlabel': 'Excited state population',
                }

            colormap = self.options_dict.get('colormap', mpl.cm.Blues)
            for i, amp in enumerate(pdd['amps_reshaped']):
                color = colormap(i/(len(pdd['amps_reshaped'])-1))
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
                for i, amp in enumerate(pdd['amps_reshaped']):
                    color = colormap(i/(len(pdd['amps_reshaped'])-1))
                    label = f'cos_fit_{qb}_{i}'
                    self.plot_dicts[label] = {
                        'ax_id': f'amplitude_crossections_{qb}',
                        'plotfn': self.plot_fit,
                        'fit_res': self.fit_res[label],
                        'plot_init': self.options_dict.get('plot_init', False),
                        'color': color,
                        'setlabel': f'fit, amp={amp:.4f}',
                    }

                # Phase contrast
                self.plot_dicts[f'phase_contrast_data_{qb}'] = {
                    'title': rdd['measurementstring'] +
                             '\n' + rdd['timestamp'],
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
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
                    'xvals': pdd['amps_reshaped'],
                    'yvals': 200*self.fit_res[f'phase_contrast_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_contrast_labels_{qb}'] = {
                    'ax_id': f'phase_contrast_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
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
                    'xvals': pdd['amps_reshaped'],
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
                    'xvals': pdd['amps_reshaped'],
                    'yvals': self.fit_res[f'phase_offset_fit_{qb}'].best_fit,
                    'color': 'r',
                    'marker': '',
                    'setlabel': 'fit',
                    'do_legend': True,
                }
                self.plot_dicts[f'phase_offset_labels_{qb}'] = {
                    'ax_id': f'phase_offset_{qb}',
                    'plotfn': self.plot_line,
                    'xvals': pdd['amps_reshaped'],
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

            colormap = self.options_dict.get('colormap', mpl.cm.Blues)
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
                        'plot_init': self.options_dict.get('plot_init', False),
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

            colormap = self.options_dict.get('colormap', mpl.cm.plasma)
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
                        'plot_init': self.options_dict.get('plot_init', False),
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
                        'plot_init': self.options_dict.get('plot_init', False),
                        'linestyle': '-',
                        'marker': '',
                        'color': 'C1',
                    }


class RabiAnalysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_amp180_'+qbn] = \
                s+f'.{trans_name}_amp180'
            params_dict[f'{trans_name}_amp90scale_'+qbn] = \
                s+f'.{trans_name}_amp90_scale'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

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

    def get_amplitudes(self, fit_res, sweep_points):
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
            piPulse_std = self.calculate_pulse_stderr(
                f=freq_fit,
                phi=phase_fit,
                f_err=freq_std,
                phi_err=phase_std,
                period_const=n_pi_pulse*np.pi,
                cov=cov_freq_phase)
            piHalfPulse_std = self.calculate_pulse_stderr(
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

        rabi_amplitudes = {'piPulse': piPulse,
                           'piPulse_stderr': piPulse_std,
                           'piHalfPulse': piHalfPulse,
                           'piHalfPulse_stderr': piHalfPulse_std}

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
                base_plot_name = f'Rabi_{k}_{self.data_to_fit[qbn]}'
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
                old_pipulse_val = self.raw_data_dict[
                    f'{trans_name}_amp180_'+qbn]
                if old_pipulse_val != old_pipulse_val:
                    old_pipulse_val = 0
                old_pihalfpulse_val = self.raw_data_dict[
                    f'{trans_name}_amp90scale_'+qbn]
                if old_pihalfpulse_val != old_pihalfpulse_val:
                    old_pihalfpulse_val = 0
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


class RabiFrequencySweepAnalysis(RabiAnalysis):

    def extract_data(self):
        super().extract_data()
        # Set some default values specific to RabiFrequencySweepAnalysis if the
        # respective options have not been set by the user or in the metadata.
        # (We do not do this in the init since we have to wait until
        # metadata has been extracted.)
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

        # Extract additional parameters from the HDF file.
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
                plotsize = self.get_default_plot_params(set=False)['figure.figsize']
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


class T1Analysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
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
        for qbn in self.qb_names:
            data = self.proc_data_dict['data_to_fit'][qbn]
            sweep_points = self.proc_data_dict['sweep_points_dict'][qbn][
                'msmt_sweep_points']
            if self.num_cal_points != 0:
                data = data[:-self.num_cal_points]
            exp_decay_mod = lmfit.Model(fit_mods.ExpDecayFunc)
            guess_pars = fit_mods.exp_dec_guess(
                model=exp_decay_mod, data=data, t=sweep_points)
            guess_pars['amplitude'].vary = True
            guess_pars['tau'].vary = True
            if self.options_dict.get('vary_offset', False):
                guess_pars['offset'].vary = True
            else:
                guess_pars['offset'].value = 0
                guess_pars['offset'].vary = False
            self.set_user_guess_pars(guess_pars)
            key = 'exp_decay_' + qbn
            self.fit_dicts[key] = {
                'fit_fn': exp_decay_mod.func,
                'fit_xvals': {'t': sweep_points},
                'fit_yvals': {'data': data},
                'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            fit_res = self.fit_dicts['exp_decay_' + qbn]['fit_res']
            for par in fit_res.params:
                if fit_res.params[par].stderr is None:
                    log.warning(f'Stderr for {par} is None. Setting it to 0.')
                    fit_res.params[par].stderr = 0
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn]['T1'] = \
                fit_res.best_values['tau']
            self.proc_data_dict['analysis_params_dict'][qbn]['T1_stderr'] = \
                fit_res.params['tau'].stderr
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            for qbn in self.qb_names:
                # rename base plot
                base_plot_name = f'T1_{qbn}_{self.data_to_fit[qbn]}'
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=self.proc_data_dict['data_to_fit'][qbn],
                    plot_name_suffix=qbn+'fit',
                    qb_name=qbn)

                self.plot_dicts['fit_' + qbn] = {
                    'fig_id': base_plot_name,
                    'plotfn': self.plot_fit,
                    'fit_res': self.fit_dicts['exp_decay_' + qbn]['fit_res'],
                    'setlabel': 'exp decay fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 2,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                trans_name = self.get_transition_name(qbn)
                old_T1_val = self.raw_data_dict[f'{trans_name}_T1_'+qbn]
                if old_T1_val != old_T1_val:
                    old_T1_val = 0
                T1_dict = self.proc_data_dict['analysis_params_dict']
                textstr = '$T_1$ = {:.2f} $\mu$s'.format(
                            T1_dict[qbn]['T1']*1e6) \
                          + ' $\pm$ {:.2f} $\mu$s'.format(
                            T1_dict[qbn]['T1_stderr']*1e6) \
                          + '\nold $T_1$ = {:.2f} $\mu$s'.format(old_T1_val*1e6)
                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class RamseyAnalysis(MultiQubit_TimeDomain_Analysis):
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
    """
    def extract_data(self):
        super().extract_data()
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
                guess_pars['amplitude'].vary = False
                guess_pars['amplitude'].value = 0.5
                guess_pars['frequency'].vary = True
                guess_pars['tau'].vary = True
                guess_pars['phase'].vary = True
                guess_pars['n'].vary = False
                guess_pars['oscillation_offset'].vary = \
                        'f' in self.data_to_fit[qbn]
                # guess_pars['exponential_offset'].value = 0.5
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
        self.artificial_detuning_dict = self.get_param_value(
            'artificial_detuning_dict')
        if self.artificial_detuning_dict is None:
            artificial_detuning = self.get_param_value('artificial_detuning')
            if 'preprocessed_task_list' in self.metadata:
                pptl = self.metadata['preprocessed_task_list']
                self.artificial_detuning_dict = OrderedDict([
                    (t['qb'], t['artificial_detuning']) for t in pptl
                ])
            elif artificial_detuning is not None:
                # legacy case
                if isinstance(artificial_detuning, dict):
                    self.artificial_detuning_dict = artificial_detuning
                else:
                    self.artificial_detuning_dict = OrderedDict(
                        [(qbn, artificial_detuning) for qbn in self.qb_names])
        if self.artificial_detuning_dict is None:
            raise ValueError('"artificial_detuning" not found.')

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
            if old_qb_freq != old_qb_freq:
                old_qb_freq = 0
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

        hdf_group_name_suffix = self.options_dict.get(
            'hdf_group_name_suffix', '')
        self.save_processed_data(key='analysis_params_dict' +
                                     hdf_group_name_suffix)

    def prepare_plots(self):
        super().prepare_plots()

        if self.do_fitting:
            apd = self.proc_data_dict['analysis_params_dict']
            for outer_key, ramsey_pars_dict in apd.items():
                if outer_key in ['qubit_frequencies', 'reparking_params']:
                    # This is only for ReparkingRamseyAnalysis.
                    # It is handled by prepare_fitting_qubit_freqs of that class
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

    def extract_data(self):
        super().extract_data()
        # Set some default values specific to ReparkingRamseyAnalysis if the
        # respective options have not been set by the user or in the metadata.
        # (We do not do this in the init since we have to wait until
        # metadata has been extracted.)
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

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
                plotsize = self.get_default_plot_params(set=False)['figure.figsize']
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


class QScaleAnalysis(MultiQubit_TimeDomain_Analysis):

    def extract_data(self):
        super().extract_data()
        params_dict = {}
        for qbn in self.qb_names:
            trans_name = self.get_transition_name(qbn)
            s = 'Instrument settings.'+qbn
            params_dict[f'{trans_name}_qscale_'+qbn] = \
                s+f'.{trans_name}_motzoi'
        self.raw_data_dict.update(
            self.get_data_from_timestamp_list(params_dict))

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
                if 'sp1d_filter' in self.options_dict:
                    log.warning('Passing sp1d_filter might not work '
                                'for this QScaleAnalysis because the sweep '
                                'points need to be repeated 3x for it to '
                                'match the data size.')
                else:
                    # repeat each sweep points 3x
                    self.options_dict['sp1d_filter'] = \
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
                                'NOT within $\pm${} of the expected value '
                                'of 0.5.'.format(threshold))
            # Warning if optimal_qscale is not within +/-threshold of 0.5
            y_optimal_qscale = optimal_qscale * fitparams2['slope'].value + \
                                 fitparams2['intercept'].value
            if (y_optimal_qscale > (0.5 + threshold)) or \
                    (y_optimal_qscale < (0.5 - threshold)):
                log.warning('The optimal qscale found gives a population '
                                'that is NOT within $\pm${} of the expected '
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

                    trans_name = self.get_transition_name(qbn)
                    old_qscale_val = self.raw_data_dict[
                        f'{trans_name}_qscale_'+qbn]
                    if old_qscale_val != old_qscale_val:
                        old_qscale_val = 0
                    textstr = 'Qscale = {:.4f} $\pm$ {:.4f}'.format(
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'qscale'],
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'qscale_stderr']) + \
                            '\nold Qscale= {:.4f}'.format(old_qscale_val)

                    self.plot_dicts['text_msg_' + qbn] = {
                        'fig_id': base_plot_name,
                        'ypos': -0.225,
                        'xpos': 0.5,
                        'horizontalalignment': 'center',
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


class EchoAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, auto=False, **kwargs)
        if self.options_dict.get('artificial_detuning', None) is not None:
            self.echo_analysis = RamseyAnalysis(*args, auto=False, **kwargs)
        else:
            if 'options_dict' in kwargs:
                # kwargs.pop('options_dict')
                kwargs['options_dict'].update({'vary_offset': True})
            else:
                kwargs['options_dict'] = {'vary_offset': True}
            self.echo_analysis = T1Analysis(*args, auto=False, **kwargs)

        if auto:
            try:
                self.echo_analysis.extract_data()
                self.qb_names = self.echo_analysis.qb_names
                self.echo_analysis.process_data()
                self.echo_analysis.prepare_fitting()
                self.echo_analysis.run_fitting()
                self.echo_analysis.save_fit_results()
                self.analyze_fit_results()
                self.prepare_plots()
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    log.error("Unhandled error during analysis!")
                    log.error(traceback.format_exc())

    def analyze_fit_results(self):
        self.echo_analysis.analyze_fit_results()
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()

            params_dict = self.echo_analysis.proc_data_dict[
                'analysis_params_dict'][qbn]
            if 'T1' in params_dict:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo'] = params_dict['T1']
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo_stderr'] = params_dict['T1_stderr']
            else:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo'] = params_dict['exp_decay']['T2_star']
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'T2_echo_stderr'] = params_dict['exp_decay'][
                    'T2_star_stderr']

    def prepare_plots(self):
        self.echo_analysis.prepare_plots()
        for qbn in self.qb_names:
            # rename base plot
            figure_name = f'Echo_{qbn}_{self.echo_analysis.data_to_fit[qbn]}'
            echo_plot_key_t1 = [key for key in self.echo_analysis.plot_dicts if
                                'T1_'+qbn in key]
            echo_plot_key_ram = [key for key in self.echo_analysis.plot_dicts if
                                 'Ramsey_'+qbn in key]
            if len(echo_plot_key_t1) != 0:
                echo_plot_name = echo_plot_key_t1[0]
            elif len(echo_plot_key_ram) != 0:
                echo_plot_name = echo_plot_key_ram[0]
            else:
                raise ValueError('Neither T1 nor Ramsey plots were found.')

            self.echo_analysis.plot_dicts[echo_plot_name][
                'legend_pos'] = 'upper right'
            self.echo_analysis.plot_dicts[echo_plot_name][
                'legend_bbox_to_anchor'] = (1, -0.15)

            for plot_label in self.echo_analysis.plot_dicts:
                if qbn in plot_label:
                    if 'raw' not in plot_label and 'projected' not in plot_label:
                        self.echo_analysis.plot_dicts[plot_label]['fig_id'] = \
                            figure_name

            old_T2e_val = a_tools.get_instr_setting_value_from_file(
                file_path=self.echo_analysis.raw_data_dict['folder'],
                instr_name=qbn, param_name='T2{}'.format(
                    '_ef' if 'f' in self.echo_analysis.data_to_fit[qbn]
                    else ''))
            T2_dict = self.proc_data_dict['analysis_params_dict']
            textstr = '$T_2$ echo = {:.2f} $\mu$s'.format(
                T2_dict[qbn]['T2_echo']*1e6) \
                      + ' $\pm$ {:.2f} $\mu$s'.format(
                T2_dict[qbn]['T2_echo_stderr']*1e6) \
                      + '\nold $T_2$ echo = {:.2f} $\mu$s'.format(
                old_T2e_val*1e6)

            self.echo_analysis.plot_dicts['text_msg_' + qbn][
                'text_string'] = textstr

        self.echo_analysis.plot(key_list='auto')
        self.echo_analysis.save_figures(close_figs=True)


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

        self.options_dict['plot_proj_data'] = False
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
            if old_amp180 != old_amp180:
                old_amp180 = 0

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
                if old_amp180 != old_amp180:
                    old_amp180 = 0
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
        options_dict = kwargs.pop('options_dict', {})
        options_dict.update({'TwoD': True})
        kwargs.update({'options_dict': options_dict})
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
        preselection_qbs = self.get_param_value("preselection_qbs",
                                                default_preselection_qbs)
        self.options_dict.update({"preselection_qbs": preselection_qbs})

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
        plotsize = self.get_default_plot_params(set=False)[
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
                'xunit': xunit,
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
                        population_loss = (amps[1::2] - amps[0::2])/amps[1::2]
                        x = amps[1::2] - amps[0::2]
                        x_err = np.array(amps_errs[0::2]**2 + amps_errs[1::2]**2,
                                         dtype=np.float64)
                        y = amps[1::2]
                        y_err = amps_errs[1::2]
                        try:
                            population_loss_stderrs = np.sqrt(np.array(
                                ((y * x_err) ** 2 + (x * y_err) ** 2) / (y ** 4),
                                dtype=np.float64))
                        except:
                            population_loss_stderrs = float("nan")
                        self.proc_data_dict['analysis_params_dict'][
                            f'population_loss_{qbn}'] = \
                            {'val': population_loss,
                             'stderr': population_loss_stderrs}

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
        len_ssp = len(self.proc_data_dict['analysis_params_dict'][
                          f'{self.phase_key}_{self.ramsey_qbnames[0]}']['val'])
        if self.options_dict.get('plot_all_traces', True):
            for j, qbn in enumerate(self.qb_names):
                if self.options_dict.get('plot_all_probs', True):
                    for prob_label, data_2d in self.proc_data_dict[
                            'projected_data_dict'][qbn].items():
                        figure_name = self.plot_traces(prob_label, data_2d, qbn)
                else:
                    for prob_label, data_2d in self.proc_data_dict[
                            'data_to_fit'][qbn]:
                        figure_name = self.plot_traces(prob_label, data_2d, qbn)

                if self.do_fitting and len_ssp == 1:
                    self.options_dict.update({'TwoD': False,
                                              'plot_proj_data': False})
                    super().prepare_plots()

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
                                           f'population_loss_{qbn}']['val'][0],
                                       self.proc_data_dict[
                                           'analysis_params_dict'][
                                           f'population_loss_{qbn}'][
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class CZDynamicPhaseAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self):
        super().process_data()
        # convert phases to radians
        for qbn in self.qb_names:
            sweep_dict = self.proc_data_dict['sweep_points_dict'][qbn]
            sweep_dict['sweep_points'] *= np.pi/180

        # get data with flux pulse and w/o flux pulse
        self.data_with_fp = OrderedDict()
        self.data_no_fp = OrderedDict()
        for qbn in self.qb_names:
            all_data = self.proc_data_dict['data_to_fit'][qbn]
            if self.num_cal_points != 0:
                all_data = all_data[:-self.num_cal_points]
            self.data_with_fp[qbn] = all_data[0: len(all_data)//2]
            self.data_no_fp[qbn] = all_data[len(all_data)//2:]

    def prepare_fitting(self):
        self.fit_dicts = OrderedDict()
        for qbn in self.qb_names:
            sweep_points = np.unique(
                self.proc_data_dict['sweep_points_dict'][qbn][
                    'msmt_sweep_points'])
            for i, data in enumerate([self.data_with_fp[qbn],
                                      self.data_no_fp[qbn]]):
                cos_mod = lmfit.Model(fit_mods.CosFunc)
                guess_pars = fit_mods.Cos_guess(
                    model=cos_mod,
                    t=sweep_points,
                    data=data, freq_guess=1/(2*np.pi))
                guess_pars['frequency'].value = 1/(2*np.pi)
                guess_pars['frequency'].vary = False

                key = 'cos_fit_{}_{}'.format(qbn, 'wfp' if i == 0 else 'nofp')
                self.fit_dicts[key] = {
                    'fit_fn': fit_mods.CosFunc,
                    'fit_xvals': {'t': sweep_points},
                    'fit_yvals': {'data': data},
                    'guess_pars': guess_pars}

    def analyze_fit_results(self):
        self.proc_data_dict['analysis_params_dict'] = OrderedDict()
        for qbn in self.qb_names:
            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn][
                'dynamic_phase'] = {
                'val': (self.fit_dicts[f'cos_fit_{qbn}_wfp'][
                            'fit_res'].best_values['phase'] -
                        self.fit_dicts[f'cos_fit_{qbn}_nofp'][
                            'fit_res'].best_values['phase']),
                'stderr': np.sqrt(
                    self.fit_dicts[f'cos_fit_{qbn}_wfp'][
                        'fit_res'].params['phase'].stderr**2 +
                    self.fit_dicts[f'cos_fit_{qbn}_nofp'][
                        'fit_res'].params['phase'].stderr**2)
            }
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        super().prepare_plots()
        for qbn in self.qb_names:
            for i, data in enumerate([self.data_with_fp[qbn],
                                      self.data_no_fp[qbn]]):
                fit_key = f'cos_fit_{qbn}_wfp' if i == 0 else \
                    f'cos_fit_{qbn}_nofp'
                plot_name_suffix = 'fit_'+'wfp' if i == 0 else 'nofp'
                cal_pts_data = self.proc_data_dict['data_to_fit'][qbn][
                               -self.num_cal_points:]
                base_plot_name = 'Dynamic_phase_' + qbn
                self.prepare_projected_data_plot(
                    fig_name=base_plot_name,
                    data=np.concatenate((data,cal_pts_data)),
                    sweep_points=np.unique(
                        self.proc_data_dict['sweep_points_dict'][qbn][
                            'sweep_points']),
                    data_label='with flux pulse' if i == 0 else 'no flux pulse',
                    plot_name_suffix=qbn + plot_name_suffix,
                    qb_name=qbn,
                    do_legend_cal_states=(i == 0))
                if self.do_fitting:
                    fit_res = self.fit_dicts[fit_key]['fit_res']
                    self.plot_dicts[plot_name_suffix + '_' + qbn] = {
                        'fig_id': base_plot_name,
                        'plotfn': self.plot_fit,
                        'fit_res': fit_res ,
                        'setlabel': 'cosine fit',
                        'color': 'r',
                        'do_legend': i == 0}

                    textstr = 'Dynamic phase {}:\n\t{:.2f}'.format(
                        qbn,
                        self.proc_data_dict['analysis_params_dict'][qbn][
                            'dynamic_phase']['val']*180/np.pi) + \
                              r'$^{\circ}$' + \
                              '$\\pm${:.2f}'.format(
                                  self.proc_data_dict['analysis_params_dict'][qbn][
                                      'dynamic_phase']['stderr']*180/np.pi) + \
                              r'$^{\circ}$'

                    fpl = self.get_param_value('flux_pulse_length')
                    if fpl is not None:
                        textstr += '\n length: {:.2f} ns'.format(fpl*1e9)
                    fpa = self.get_param_value('flux_pulse_amp')
                    if fpa is not None:
                        textstr += '\n amp: {:.4f} V'.format(fpa)

                    self.plot_dicts['text_msg_' + qbn] = {
                        'fig_id': base_plot_name,
                        'ypos': -0.15,
                        'xpos': -0.05,
                        'horizontalalignment': 'left',
                        'verticalalignment': 'top',
                        'plotfn': self.plot_text,
                        'text_string': textstr}
            for plot_name in list(self.plot_dicts)[::-1]:
                if self.plot_dicts[plot_name].get('do_legend', False):
                    break
            self.plot_dicts[plot_name].update(
                {'legend_ncol': 2,
                 'legend_bbox_to_anchor': (1, -0.15),
                 'legend_pos': 'upper right'})


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
            auto (bool): Start analysis automatically
            **kwargs:
                t_start: timestamp of the first timetrace
                t_stop: timestamp of the last timetrace to analyze
                options_dict (dict): relevant parameters:
                    acq_weights_basis (list, dict):
                        list of basis vectors used to compute optimal weight.
                        e.g. ["ge", 'gf'], the first basis vector will be the
                        "e" timetrace minus the "g" timetrace and the second basis
                        vector is f - g. The first letter in each basis state is the
                        "reference state", i.e. the one of which the timetrace
                         is substracted. Can also be passed as a dictionary where
                         keys are the qubit names and the values are lists of basis states
                         in case different bases should be used for different qubits.
                    orthonormalize (bool): Whether or not to orthonormalize the
                        weight basis
                    tmax (float): time boundary for the plot (not the weights)
                        in seconds.
                    scale_weights (bool): scales the weights near unity to avoid
                        loss of precision on FPGA if weights are too small

        """

        if qb_names is not None:
            self.params_dict = {}
            for qbn in qb_names:
                s = 'Instrument settings.' + qbn
                for trans_name in ['ge', 'ef']:
                    self.params_dict[f'ro_mod_freq_' + qbn] = \
                        s + f'.ro_mod_freq'
            self.numeric_params = list(self.params_dict)

        self.qb_names = qb_names
        super().__init__(**kwargs)
        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()

        if self.qb_names is None:
            # get all qubits from cal_points of first timetrace
            cp = CalibrationPoints.from_string(
                self.get_param_value('cal_points', None, 0))
            self.qb_names = deepcopy(cp.qb_names)

        self.channel_map = self.get_param_value('channel_map', None,
                                                index=0)
        if self.channel_map is None:
            # assume same channel map for all timetraces (pick 0th)
            value_names = self.raw_data_dict[0]['value_names']
            if np.ndim(value_names) > 0:
                value_names = value_names
            if 'w' in value_names[0]:
                self.channel_map = a_tools.get_qb_channel_map_from_hdf(
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
        for qbn in self.qb_names:
            # retrieve time traces
            for i, rdd in enumerate(self.raw_data_dict):
                ttrace_per_ro_ch = [rdd["measured_data"][ch]
                                    for ch in self.channel_map[qbn]]
                if len(ttrace_per_ro_ch) != 2:
                    raise NotImplementedError(
                        'This analysis does not support optimal weight '
                        f'measurement based on {len(ttrace_per_ro_ch)} ro channels.'
                        f' Try again with 2 RO channels.')
                cp = CalibrationPoints.from_string(
                    self.get_param_value('cal_points', None, i))
                # get state of qubit. There can be only one cal point per sequence
                # when using uhf for time traces so it is the 0th state
                qb_state = cp.states[0][cp.qb_names.index(qbn)]
                # store all timetraces in same pdd for convenience
                ana_params['timetraces'][qbn].update(
                    {qb_state: ttrace_per_ro_ch[0] + 1j *ttrace_per_ro_ch[1]})

            timetraces = ana_params['timetraces'][qbn] # for convenience
            basis_labels = self.get_param_value('acq_weights_basis', None, 0)
            if basis_labels is None:
                # guess basis labels from # states measured
                basis_labels = ["ge", "ef"] \
                    if len(ana_params['timetraces'][qbn]) > 2 else ['ge']

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
            if self.get_param_value("orthonormalize", False):
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
            if self.get_param_value('scale_weights', True):
                k = np.amax([(np.max(np.abs(b.real)),
                              np.max(np.abs(b.imag))) for b in basis])
                basis /= k
            ana_params['optimal_weights'][qbn] = basis
            ana_params['optimal_weights_basis_labels'][qbn] = basis_labels

            self.save_processed_data()

    def prepare_plots(self):

        pdd = self.proc_data_dict
        rdd = self.raw_data_dict
        ana_params = self.proc_data_dict['analysis_params_dict']
        for qbn in self.qb_names:
            mod_freq = float(
                rdd[0].get(f'ro_mod_freq_{qbn}',
                           self.get_hdf_param_value(f"Instrument settings/{qbn}",
                                                    'ro_mod_freq')))
            tbase = rdd[0]['hard_sweep_points']
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
                        'xvals': tbase,
                        "marker": "",
                        'yvals': func(ttrace*modulation),
                        'ylabel': 'Voltage, $V$',
                        'yunit': 'V',
                        "sharex": True,
                        "setdesc": label + f"_{state}",
                        "setlabel": "",
                        "do_legend":True,
                        "legend_pos": "upper right",
                        'numplotsx': 1,
                        'numplotsy': len(rdd) + 1, # #states + 1 for weights
                        'plotsize': (10,
                                     (len(rdd) + 1) * 3), # 3 inches per plot
                        'title': title if ax_id == 0 else ""}
            ax_id = len(ana_params["timetraces"][qbn]) # id plots for weights
            for i, weights in enumerate(ana_params['optimal_weights'][qbn]):
                for func, label in zip((np.real, np.imag), ('I', "Q")):
                    self.plot_dicts[f"{plot_name}_weights_{label}_{i}"] = {
                        'fig_id': plot_name,
                        'ax_id': ax_id,
                        'plotfn': self.plot_line,
                        'xvals': tbase,
                        'xlabel': xlabel,
                        "setlabel": "",
                        "marker": "",
                        'xunit': 's',
                        'yvals': func(weights * modulation),
                        'ylabel': 'Voltage, $V$ (arb.u.)',
                        "sharex": True,
                        "xrange": [min(tbase), max(tbase)],
                        "setdesc": label + f"_{i+1}",
                        "do_legend": True,
                        "legend_pos": "upper right",
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
        '''
        options dict options:
            'nr_bins' : number of bins to use for the histograms
            'post_select' :
            'post_select_threshold' :
            'nr_samples' : amount of different samples (e.g. ground and excited = 2)
            'sample_0' : index of first sample (ground-state)
            'sample_1' : index of second sample (first excited-state)
            'max_datapoints' : maximum amount of datapoints for culumative fit
            'hist_scale' : scale for the y-axis of the 1D histograms: "linear" or "log"
            'verbose' : see BaseDataAnalysis
            'presentation_mode' : see BaseDataAnalysis
            'classif_method': how to classify the data.
                'ncc' : default. Nearest Cluster Center
                'gmm': gaussian mixture model.
                'threshold': finds optimal vertical and horizontal thresholds.
            'classif_kw': kw to pass to the classifier
            see BaseDataAnalysis for more.
        '''
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
        super().process_data()
        del self.proc_data_dict['data_to_fit'] # not used in this analysis
        n_states = len(self.cp.states)

        # prepare data in convenient format, i.e. arrays per qubit and per state
        # e.g. {'qb1': {'g': np.array of shape (n_shots, n_ro_ch}, ...}, ...}
        shots_per_qb = dict()        # store shots per qb and per state
        presel_shots_per_qb = dict() # store preselection ro
        means = defaultdict(OrderedDict)    # store mean per qb for each ro_ch
        pdd = self.proc_data_dict    # for convenience of notation

        for qbn in self.qb_names:
            # shape is (n_shots, n_ro_ch) i.e. one column for each ro_ch
            shots_per_qb[qbn] = \
                np.asarray(list(
                    pdd['meas_results_per_qb'][qbn].values())).T
            # make 2D array in case only one channel (1D array)
            if len(shots_per_qb[qbn].shape) == 1:
                shots_per_qb[qbn] = np.expand_dims(shots_per_qb[qbn],
                                                   axis=-1)
            for i, qb_state in enumerate(self.cp.get_states(qbn)[qbn]):
                means[qbn][qb_state] = np.mean(shots_per_qb[qbn][i::n_states],
                                               axis=0)
            if self.preselection:
                # preselection shots were removed so look at raw data
                # and look at only the first out of every two readouts
                presel_shots_per_qb[qbn] = \
                    np.asarray(list(
                        pdd['meas_results_per_qb_raw'][qbn].values())).T[::2]
                # make 2D array in case only one channel (1D array)
                if len(presel_shots_per_qb[qbn].shape) == 1:
                    presel_shots_per_qb[qbn] = \
                        np.expand_dims(presel_shots_per_qb[qbn], axis=-1)

        # create placeholders for analysis data
        pdd['analysis_params'] = dict()
        pdd['data'] = defaultdict(dict)
        pdd['analysis_params']['state_prob_mtx'] = defaultdict(dict)
        pdd['analysis_params']['classifier_params'] = defaultdict(dict)
        pdd['analysis_params']['means'] = defaultdict(dict)
        pdd['analysis_params']['snr'] = defaultdict(dict)
        pdd['analysis_params']["n_shots"] = len(shots_per_qb[qbn])
        pdd['analysis_params']['slopes'] = defaultdict(dict)
        self.clf_ = defaultdict(dict)
        # create placeholders for analysis with preselection
        if self.preselection:
            pdd['data_masked'] = defaultdict(dict)
            pdd['analysis_params']['state_prob_mtx_masked'] = defaultdict(dict)
            pdd['analysis_params']['n_shots_masked'] = defaultdict(dict)

        n_shots = len(shots_per_qb[qbn]) // n_states

        for qbn, qb_shots in shots_per_qb.items():
            # create mapping to integer following ordering in cal_points.
            # Notes:
            # 1) the state_integer should to the order of pdd[qbn]['means'] so that
            # when passing the init_means to the GMM model, it is ensured that each
            # gaussian component will predict the state_integer associated to that state
            # 2) the mapping cannot be preestablished because the GMM predicts labels
            # in range(n_components). For instance, if a qubit has states "g", "f"
            # then the model will predicts 0's and 1's, so the typical g=0, e=1, f=2
            # mapping would fail. The number of different states can be different
            # for each qubit and therefore the mapping should also be done per qubit.
            state_integer = 0
            for state in means[qbn].keys():
                self.states_info[qbn][state]["int"] = state_integer
                state_integer += 1

            # note that if some states are repeated, they are assigned the same label
            qb_states_integer_repr = \
                [self.states_info[qbn][s]["int"]
                 for s in self.cp.get_states(qbn)[qbn]]
            prep_states = np.tile(qb_states_integer_repr, n_shots)

            pdd['analysis_params']['means'][qbn] = deepcopy(means[qbn])
            pdd['data'][qbn] = dict(X=deepcopy(qb_shots),
                                    prep_states=prep_states)
            # self.proc_data_dict['keyed_data'] = deepcopy(data)

            assert np.ndim(qb_shots) == 2, "Data must be a two D array. " \
                                    "Received shape {}, ndim {}"\
                                    .format(qb_shots.shape, np.ndim(qb_shots))
            pred_states, clf_params, clf = \
                self._classify(qb_shots, prep_states,
                               method=self.classif_method, qb_name=qbn,
                               **self.options_dict.get("classif_kw", dict()))
            # order "unique" states to have in usual order "gef" etc.
            state_labels_ordered = self._order_state_labels(
                list(means[qbn].keys()))
            # translate to corresponding integers
            state_labels_ordered_int = [self.states_info[qbn][s]['int'] for s in
                                        state_labels_ordered]
            fm = self.fidelity_matrix(prep_states, pred_states,
                                      labels=state_labels_ordered_int)

            # save fidelity matrix and classifier
            pdd['analysis_params']['state_prob_mtx'][qbn] = fm
            pdd['analysis_params']['classifier_params'][qbn] = clf_params
            if 'means_' in clf_params:
                pdd['analysis_params']['snr'][qbn] = \
                    self._extract_snr(clf, state_labels_ordered)
                pdd['analysis_params']['slopes'][qbn] = self._extract_slopes(
                    clf, state_labels_ordered)

            self.clf_[qbn] = clf
            if self.preselection:
                #re do with classification first of preselection and masking
                pred_presel = self.clf_[qbn].predict(presel_shots_per_qb[qbn])
                presel_filter = \
                    pred_presel == self.states_info[qbn]['g']['int']
                if np.sum(presel_filter) == 0:
                    log.warning(f"{qbn}: No data left after preselection! "
                                f"Skipping preselection data & figures.")
                    continue
                qb_shots_masked = qb_shots[presel_filter]
                prep_states = prep_states[presel_filter]
                pred_states = self.clf_[qbn].predict(qb_shots_masked)
                fm = self.fidelity_matrix(prep_states, pred_states,
                                          labels=state_labels_ordered_int)

                pdd['data_masked'][qbn] = dict(X=deepcopy(qb_shots_masked),
                                          prep_states=deepcopy(prep_states))
                pdd['analysis_params']['state_prob_mtx_masked'][qbn] = fm
                pdd['analysis_params']['n_shots_masked'][qbn] = \
                    qb_shots_masked.shape[0]

        self.save_processed_data()

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

    def _classify(self, X, prep_state, method, qb_name, **kw):
        """

        Args:
            X: measured data to classify
            prep_state: prepared states (true values)
            type: classification method
            qb_name: name of the qubit to classify

        Returns:

        """
        if np.ndim(X) == 1:
            X = X.reshape((-1,1))
        params = dict()

        if method == 'ncc':
            ncc = SSROQutrit.NCC(
                self.proc_data_dict['analysis_params']['means'][qb_name])
            pred_states = ncc.predict(X)
            # self.clf_ = ncc
            return pred_states, dict(), ncc

        elif method == 'gmm':
            cov_type = kw.pop("covariance_type", "tied")
            # full allows full covariance matrix for each level. Other options
            # see GM documentation
            # assumes if repeated state, should be considered of the same component
            # this classification method should not be used for multiplexed SSRO
            # analysis
            n_qb_states = len(np.unique(self.cp.get_states(qb_name)[qb_name]))
            # give same weight to each class by default
            weights_init = kw.pop("weights_init",
                                  np.ones(n_qb_states)/n_qb_states)

            gm = GM(n_components=n_qb_states,
                    covariance_type=cov_type,
                    random_state=0,
                    weights_init=weights_init,
                    means_init=[mu for _, mu in
                                self.proc_data_dict['analysis_params']
                                    ['means'][qb_name].items()], **kw)
            gm.fit(X)
            pred_states = np.argmax(gm.predict_proba(X), axis=1)

            params['means_'] = gm.means_
            params['covariances_'] = gm.covariances_
            params['covariance_type'] = gm.covariance_type
            params['weights_'] = gm.weights_
            params['precisions_cholesky_'] = gm.precisions_cholesky_
            return pred_states, params, gm

        elif method == "threshold":
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
    def plot_fidelity_matrix(fm, target_names,
                             title="State Assignment Probability Matrix",
                             auto_shot_info=True, ax=None,
                             cmap=None, normalize=True, show=False):
        return SSROQutrit.plot_fidelity_matrix(
            fm, target_names, title=title, ax=ax,
            auto_shot_info=auto_shot_info,
            cmap=cmap, normalize=normalize, show=show)

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
    def _order_state_labels(states_labels,
                            order="gefhabcdijklmnopqrtuvwxyz0123456789"):
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


    def plot(self, **kwargs):
        if not self.get_param_value("plot", True):
            return # no plotting if "plot" is False
        cmap = plt.get_cmap('tab10')
        show = self.options_dict.get("show", False)
        pdd = self.proc_data_dict
        for qbn in self.qb_names:
            n_qb_states = len(np.unique(self.cp.get_states(qbn)[qbn]))
            tab_x = a_tools.truncate_colormap(cmap, 0,
                                              n_qb_states/10)

            kwargs = {
                "states": list(pdd["analysis_params"]['means'][qbn].keys()),
                "xlabel": "Integration Unit 1, $u_1$",
                "ylabel": "Integration Unit 2, $u_2$",
                "scale": self.options_dict.get("hist_scale", "log"),
                "cmap":tab_x}
            data_keys = [k for k in list(pdd.keys()) if
                            k.startswith("data") and qbn in pdd[k]]

            for dk in data_keys:
                data = pdd[dk][qbn]
                title =  self.raw_data_dict['timestamp'] + f" {qbn} " + dk + \
                    "\n{} classifier".format(self.classif_method)
                kwargs.update(dict(title=title))

                # plot data and histograms
                n_shots_to_plot = self.get_param_value('n_shots_to_plot', None)
                if n_shots_to_plot is not None:
                    n_shots_to_plot *= n_qb_states
                if data['X'].shape[1] == 1:
                    if self.classif_method == "gmm":
                        kwargs['means'] = self._get_means(self.clf_[qbn])
                        kwargs['std'] = np.sqrt(self._get_covariances(self.clf_[qbn]))
                    else:
                        # no Gaussian distribution can be plotted
                        kwargs['plot_fitting'] = False
                    kwargs['colors'] = cmap(np.unique(data['prep_states']))
                    fig, main_ax = self.plot_1D_hist(data['X'][:n_shots_to_plot],
                                            data["prep_states"][:n_shots_to_plot],
                                            **kwargs)
                else:
                    fig, axes = self.plot_scatter_and_marginal_hist(
                        data['X'][:n_shots_to_plot],
                        data["prep_states"][:n_shots_to_plot],
                        **kwargs)

                    # plot clf_boundaries
                    main_ax = fig.get_axes()[0]
                    self.plot_clf_boundaries(data['X'], self.clf_[qbn], ax=main_ax,
                                             cmap=tab_x)
                    # plot means and std dev
                    data_means = pdd['analysis_params']['means'][qbn]
                    try:
                        clf_means = self._get_means(self.clf_[qbn])
                    except Exception as e: # not a gmm model--> no clf_means.
                        clf_means = []
                    try:
                        covs = self._get_covariances(self.clf_[qbn])
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
                thresholds = pdd['analysis_params'][
                    'classifier_params'][qbn].get("thresholds", dict())
                mapping = pdd['analysis_params'][
                    'classifier_params'][qbn].get("mapping", dict())
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

                self.figs[f'{qbn}_{self.classif_method}_classifier_{dk}'] = fig
            if show:
                plt.show()

            # state assignment prob matrix
            title = self.raw_data_dict['timestamp'] + "\n{} State Assignment" \
                " Probability Matrix\nTotal # shots:{}"\
                .format(self.classif_method,
                        self.proc_data_dict['analysis_params']['n_shots'])
            fig = self.plot_fidelity_matrix(
                self.proc_data_dict['analysis_params']['state_prob_mtx'][qbn],
                self._order_state_labels(kwargs['states']),
                title=title,
                show=show,
                auto_shot_info=False)
            self.figs[f'{qbn}_state_prob_matrix_{self.classif_method}'] = fig

            if self.preselection and \
                    len(pdd['analysis_params']['state_prob_mtx_masked'][qbn]) != 0:
                title = self.raw_data_dict['timestamp'] + \
                    "\n{} State Assignment Probability Matrix Masked"\
                    "\nTotal # shots:{}".format(
                        self.classif_method,
                        self.proc_data_dict['analysis_params']['n_shots_masked'][qbn])

                fig = self.plot_fidelity_matrix(
                    pdd['analysis_params']['state_prob_mtx_masked'][qbn],
                    self._order_state_labels(kwargs['states']),
                    title=title, show=show, auto_shot_info=False)
                fig_key = f'{qbn}_state_prob_matrix_masked_{self.classif_method}'
                self.figs[fig_key] = fig

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
    def __init__(self, options_dict: dict = None, auto=True, **kw):
        '''
        options dict options:
            plot_raw_shots (bool): whether or not to plot histograms/scatter
                plots of raw shots. False by default. Slows down the analysis as
                it creates one plot per readout.
            see BaseDataAnalysis for more.
        '''
        if options_dict is None:
            options_dict = {}
        options_dict.update({"TwoD": True})
        super().__init__(options_dict=options_dict, auto=False,
                         **kw)

        self.create_job(options_dict=options_dict, auto=auto, **kw)

        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
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
                                'fig_id':fig_key,
                                'xvals': np.arange(len(pop)),
                                'xlabel': "Reset cycle, $n$",
                                'xunit': "",
                                'yvals': pop,
                                'yerr': self._std_error(
                                    pop, self.get_param_value('n_shots')),
                                'ylabel': 'Population, $P$',
                                'yunit': '',
                                'yscale': self.get_param_value("yscale", "log"),
                                'setlabel': self._get_pop_label(state, k,
                                                                not self._has_reset_pulses(seq_nr),
                                                                ),
                                'title': self.raw_data_dict['timestamp'] + ' ' +
                                         self.raw_data_dict['measurementstring']
                                         + " " + prep_state,
                                'titlepad': 0.2,
                                'linestyle': '-',
                                'color': f'C{i}',
                                'alpha': 0.5 if seq_nr == 0 else 1,
                                'do_legend': True,
                                'legend_ncol': legend_ncol,
                                'legend_bbox_to_anchor': legend_bbox_to_anchor,
                                'legend_pos': legend_pos,
                                'legend_fontsize': 5}

                            # add feedback params info to plot
                            textstr = self._get_feedback_params_text_str(qbn)
                            self.plot_dicts[f'text_msg_{qbn}_' \
                                            f'{prep_state}{keys[k]}'] = {
                                'fig_id': f"populations_{qbn}_{prep_state}{keys[k]}",
                                'ypos': -0.21,
                                'xpos': 0,
                                'horizontalalignment': 'left',
                                'verticalalignment': 'top',
                                'plotfn': self.plot_text,
                                "fontsize": "x-small",
                                'text_string': textstr}

                            # add thermal population line
                            g_state_prep_g = \
                                data_qbi.get("pg", {}).get('prep_g', None)
                            # taking first ro of first sequence as estimate for
                            # thermal population
                            if g_state_prep_g is not None and seq_nr == 0:
                                p_therm = 1 - g_state_prep_g.flatten()[0]
                                self.plot_dicts[plt_key + "_thermal"] = {
                                    'plotfn': self.plot_line,
                                    'fig_id': fig_key,
                                    'xvals': np.arange(len(pop)),
                                    'yvals': p_therm * np.ones_like(pop),
                                    'setlabel': "$P_\mathrm{therm}$",
                                    'linestyle': '--',
                                    'marker': "",
                                    'color': 'k',
                                    'do_legend': True,
                                    'legend_ncol': legend_ncol,
                                    'legend_bbox_to_anchor': legend_bbox_to_anchor,
                                    'legend_pos': legend_pos,
                                    'legend_fontsize': 5}

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
                                'legend_fontsize': 5}

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
                                    'legend_fontsize': 5
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
        str += f"{1e6 * ro_sep:.2f}$\mu s$" if ro_sep is not None else "Unknown"
        str += "\n"

        str += "RO to feedback time: "
        prow = self.prep_params.get("post_ro_wait", None)
        str += f"{1e6 * prow:.2f}$\mu s$" if ro_sep is not None else "Unknown"
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

    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.'+qbn
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        # super().__init__(qb_names, *args, **kwargs)

        options_dict = kwargs.pop('options_dict', {})
        options_dict['TwoD'] = True
        kwargs['options_dict'] = options_dict
        super().__init__(qb_names, *args, **kwargs)

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
            guess_pars = fit_mods.TwoErrorFunc_guess(model=TwoErrorFuncModel,
                                               data=data, \
                                            delays=sweep_points)
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
            mu_A = self.fit_dicts['two_error_func_' + qbn]['fit_res'].best_values[
                'mu_A']
            mu_B = self.fit_dicts['two_error_func_' + qbn]['fit_res'].best_values[
                'mu_B']
            fp_length = a_tools.get_instr_setting_value_from_file(
                file_path=self.raw_data_dict['folder'],
                instr_name=qbn, param_name='flux_pulse_pulse_length')


            self.proc_data_dict['analysis_params_dict'][qbn] = OrderedDict()
            self.proc_data_dict['analysis_params_dict'][qbn]['delay'] = \
                mu_A + 0.5 * (mu_B - mu_A) - fp_length / 2
            try:
                self.proc_data_dict['analysis_params_dict'][qbn]['delay_stderr'] = \
                    1 / 2 * np.sqrt(
                        self.fit_dicts['two_error_func_' + qbn]['fit_res'].params[
                            'mu_A'].stderr ** 2
                        + self.fit_dicts['two_error_func_' + qbn]['fit_res'].params[
                            'mu_B'].stderr ** 2)
            except TypeError:
                self.proc_data_dict['analysis_params_dict'][qbn]['delay_stderr']\
                    = 0
            self.proc_data_dict['analysis_params_dict'][qbn]['fp_length'] = \
                (mu_B - mu_A)
            try:
                self.proc_data_dict['analysis_params_dict'][qbn]['fp_length_stderr'] = \
                    np.sqrt(
                        self.fit_dicts['two_error_func_' + qbn]['fit_res'].params[
                            'mu_A'].stderr ** 2
                        + self.fit_dicts['two_error_func_' + qbn]['fit_res'].params[
                            'mu_B'].stderr ** 2)
            except TypeError:
                self.proc_data_dict['analysis_params_dict'][qbn][
                    'fp_length_stderr'] = 0
        self.save_processed_data(key='analysis_params_dict')

    def prepare_plots(self):
        self.options_dict.update({'TwoD': False,
                                  'plot_proj_data': False})
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
                    'fit_res': self.fit_dicts['two_error_func_' + qbn]['fit_res'],
                    'setlabel': 'two error func. fit',
                    'do_legend': True,
                    'color': 'r',
                    'legend_ncol': 1,
                    'legend_bbox_to_anchor': (1, -0.15),
                    'legend_pos': 'upper right'}

                apd = self.proc_data_dict['analysis_params_dict']
                textstr = 'delay = {:.2f} ns'.format(apd[qbn]['delay']*1e9) \
                          + ' $\pm$ {:.2f} ns'.format(apd[qbn]['delay_stderr']
                                                      * 1e9)
                textstr += '\n\nflux_pulse_length:\n  fitted = {:.2f} ns'.format(
                    apd[qbn]['fp_length'] * 1e9) \
                           + ' $\pm$ {:.2f} ns'.format(
                    apd[qbn]['fp_length_stderr'] * 1e9)
                textstr += '\n  set = {:.2f} ns'.format(
                    1e9 * a_tools.get_instr_setting_value_from_file(
                        file_path=self.raw_data_dict['folder'],
                        instr_name=qbn, param_name='flux_pulse_pulse_length'))

                self.plot_dicts['text_msg_' + qbn] = {
                    'fig_id': base_plot_name,
                    'ypos': -0.2,
                    'xpos': 0,
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'plotfn': self.plot_text,
                    'text_string': textstr}


class FluxPulseTimingBetweenQubitsAnalysis(MultiQubit_TimeDomain_Analysis):

    def __init__(self, qb_names, *args, **kwargs):
        params_dict = {}
        for qbn in qb_names:
            s = 'Instrument settings.' + qbn
        kwargs['params_dict'] = params_dict
        kwargs['numeric_params'] = list(params_dict)
        # super().__init__(qb_names, *args, **kwargs)

        options_dict = kwargs.pop('options_dict', {})
        options_dict['TwoD'] = True
        kwargs['options_dict'] = options_dict
        super().__init__(qb_names, *args, **kwargs)

    #         self.analyze_results()

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
                self.fit_dicts['delay_fit_' + qbn]['fit_res'].best_values['delay']
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
        self.options_dict.update({'TwoD': False,
                                  'plot_proj_data': False})
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

                textstr = 'delay = {:.2f} ns'.format(apd[qbn]['delay_fit'] *
                                                     1e9) \
                          + ' $\pm$ {:.2f} ns'.format(apd[qbn][
                                                          'delay_fit_stderr']
                                                      * 1e9)
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
                #                                     'setlabel': legendlabel,
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

            textstr = 'delay = {:.2f} ns'.format(apd[qbn]['delay'] * 1e9) \
                      + ' $\pm$ {:.2f} ns'.format(apd[qbn]['delay_stderr']
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
    def __init__(self, *args, **kwargs):
        options_dict = kwargs.pop('options_dict', {})
        options_dict['TwoD'] = True
        kwargs['options_dict'] = options_dict
        super().__init__(*args, **kwargs)

    def extract_data(self):
        super().extract_data()
        # Set some default values specific to FluxPulseScopeAnalysis if the
        # respective options have not been set by the user or in the metadata.
        # (We do not do this in the init since we have to wait until
        # metadata has been extracted.)
        if self.get_param_value('rotation_type', default_value=None) is None:
            self.options_dict['rotation_type'] = 'fixed_cal_points'
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

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

        if not hasattr(self, "job"):
            self.create_job(t_start=t_start, t_stop=t_stop,
                            label=label, data_file_path=data_file_path,
                            do_fitting=do_fitting, options_dict=options_dict,
                            extract_only=extract_only, params_dict=params_dict,
                            numeric_params=numeric_params, **kwargs)
        self.params_dict = {f"{tm_mod.Timer.HDF_GRP_NAME}":
                                f"{tm_mod.Timer.HDF_GRP_NAME}",
                            "repetition_rate":
                                "Instrument settings/TriggerDevice.pulse_period",
                            }


        if auto:
            self.run_analysis()

    def extract_data(self):
        super().extract_data()
        timers_dicts = self.raw_data_dict.get('Timers', {})
        for t, v in timers_dicts.items():
            self.timers[t] = tm_mod.Timer(name=t, **v)

        # Extract and build raw measurement timer
        self.timers['BareMeasurement'] = self.bare_measurement_timer(
            ref_time=self.get_param_value("ref_time")
        )

    def process_data(self):
        pass

    def plot(self, **kwargs):
        timers = [t for t in self.timers.values() if len(t)]
        plot_kws = self.get_param_value('plot_kwargs', {})
        for t in timers:
            try:
                self.figs["timer_" + t.name] = t.plot(**plot_kws)
            except Exception as e:
                if self.raise_exceptions:
                    raise
                log.error(f'Could not plot Timer: {t.name}: {e}')

        if self.get_param_value('combined_timer', True):
            self.figs['timer_all'] = tm_mod.multi_plot(timers,
                                                       **plot_kws)

    def bare_measurement_timer(self, ref_time=None,
                               checkpoint='bare_measurement', **kw):
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
        #  (not only start stop but e.g. distribute it)
        bmtimer.checkpoint(f"BareMeasurement.{checkpoint}.start",
                           values=[ref_time], log_init=False)
        bmtimer.checkpoint(f"BareMeasurement.{checkpoint}.end",
                           values=[ ref_time + dt.timedelta(seconds=bmtime)],
                           log_init=False)

        return bmtimer

    def bare_measurement_time(self, nr_averages=None, repetition_rate=None,
                              count_nan_measurements=False):
        det_metadata = self.metadata.get("Detector Metadata", None)
        if nr_averages is None:
            nr_averages = self.get_param_value('nr_averages', None)
        if det_metadata is not None and nr_averages is None:
            # multi detector function: look for child "detectors"
            # assumes at least 1 child and that all children have the same
            # number of averages
            det = list(det_metadata.get('detectors', {}).values())[0]
            nr_averages = det.get('nr_averages', det.get('nr_shots', None))
        if nr_averages is None:
            raise ValueError('Could not extract nr_averages/nr_shots from hdf file.'
                             'Please specify "nr_averages" in options_dict.')
        self.nr_averages = nr_averages
        n_hsp = len(self.raw_data_dict['hard_sweep_points'])
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
    def extract_data(self):
        super().extract_data()
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

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

        VI = hsp
        VQ = ssp

        if len(hsp) * len(ssp) == len(LO_dBm.flatten()):
            VI, VQ = np.meshgrid(hsp, ssp)
            VI = VI.flatten()
            VQ = VQ.flatten()
            LO_dBm = LO_dBm.T.flatten()

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
        V_I = self.proc_data_dict['V_I']
        V_Q = self.proc_data_dict['V_Q']

        timestamp = self.timestamps[0]

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

            V_I_opt = self.proc_data_dict['analysis_params_dict']['V_I']
            V_Q_opt = self.proc_data_dict['analysis_params_dict']['V_Q']
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

        for ch in ['I', 'Q']:
            plot_name = f'V_{ch}_vs_LO_magn'
            leakage = self.proc_data_dict['LO_leakage']
            self.plot_dicts[f'raw_V_{ch}_vs_LO_magn'] = {
                'fig_id': plot_name,
                'plotfn': self.plot_line,
                'xvals': self.proc_data_dict[f'V_{ch}'],
                'yvals': leakage,
                'color': 'blue',
                'marker': '.',
                'linestyle': 'None',
                'xlabel': f'Offset, $V_\\mathrm{{{ch}}}$',
                'ylabel': 'Carrier Leakage $V_\\mathrm{LO}$',
                'xunit': 'V',
                'yunit': 'dBm',
                'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO}}$ '
                         f'projected onto offset $V_\\mathrm{{{ch}}}$'
            }

            if self.do_fitting:
                optimum =self.proc_data_dict['analysis_params_dict']['V_'+ch]
                y_min = np.min(leakage)
                y_max = np.max(leakage)
                self.plot_dicts[f'optimum_V_{ch}_vs_LO_magn'] = {
                    'fig_id': plot_name,
                    'plotfn': self.plot_line,
                    'xvals': np.array([optimum, optimum]),
                    'yvals': np.array([y_min, y_max]),
                    'color': 'red',
                    'marker': 'None',
                    'linestyle': '--'
                }


class MixerSkewnessAnalysis(MultiQubit_TimeDomain_Analysis):
    """Analysis for the :py:meth:~'QuDev_transmon.calibrate_drive_mixer_skewness_model' measurement.

    The class extracts the phase and amplitude correction settings of the Q
    channel input of the measured IQ mixer that maximize the suppression of the
    unwanted sideband.
    """
    def extract_data(self):
        super().extract_data()
        if self.get_param_value('TwoD', default_value=None) is None:
            self.options_dict['TwoD'] = True

    def process_data(self):
        super().process_data()

        hsp = self.raw_data_dict['hard_sweep_points']
        ssp = self.raw_data_dict['soft_sweep_points']
        mdata = self.raw_data_dict['measured_data']

        sideband_I, sideband_Q = list(mdata.values())

        if len(hsp) * len(ssp) == len(sideband_I.flatten()):
            # The arrays hsp and ssp define the edges of a grid of measured
            # points. We reshape the arrays such that each data point
            # sideband_I/Q[i] corresponds to the sweep point alpha[i], phase[i]
            alpha, phase = np.meshgrid(hsp, ssp)
            alpha = alpha.flatten()
            phase = phase.flatten()
            sideband_I = sideband_I.T.flatten()
            sideband_Q = sideband_Q.T.flatten()
        else:
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

        timestamp = self.timestamps[0]

        if self.do_fitting:
            # define grid with limits based on measurement points
            # and make it 10 % larger in both axes
            size_offset_alpha = 0.05*(np.max(alpha)-np.min(alpha))
            size_offset_phase = 0.05*(np.max(phase)-np.min(phase))
            xi = np.linspace(np.min(alpha) - size_offset_alpha,
                            np.max(alpha) + size_offset_alpha, 250)
            yi = np.linspace(np.min(phase) - size_offset_phase,
                            np.max(phase) + size_offset_phase, 250)
            x, y = np.meshgrid(xi, yi)

            fit_dict = self.fit_dicts['mixer_imbalance_sideband']
            fit_res = fit_dict['fit_res']
            best_values = fit_res.best_values
            model_func = fit_dict['model'].func
            z = model_func(x, y, **best_values)

            base_plot_name = 'mixer_sideband_suppression'
            self.plot_dicts['base_contour'] = {
                'fig_id': base_plot_name,
                'plotfn': self.plot_contourf,
                'xvals': x,
                'yvals': y,
                'zvals': z,
                'xlabel': 'Ampl., Ratio, $\\alpha_\\mathrm{IQ}$',
                'ylabel': 'Phase Off., $\\Delta\\phi_\\mathrm{IQ}$',
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

        raw_alpha_plot_name = 'alpha_vs_sb_magn'
        self.plot_dicts['raw_alpha_vs_sb_magn'] = {
            'fig_id': raw_alpha_plot_name,
            'plotfn': self.plot_line,
            'xvals': alpha,
            'yvals': pdict['sideband_dBm_amp'],
            'color': 'blue',
            'marker': '.',
            'linestyle': 'None',
            'xlabel': 'Ampl., Ratio, $\\alpha_\\mathrm{IQ}$',
            'ylabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$',
            'xunit': '',
            'yunit': 'dBm',
            'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO-IF}}$ '
                     f'projected onto ampl. ratio $\\alpha_\\mathrm{{IQ}}$'
        }

        if self.do_fitting:
            self.plot_dicts['optimum_in_alpha_vs_sb_magn'] = {
                'fig_id': raw_alpha_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([alpha_min, alpha_min]),
                'yvals': np.array([np.min(pdict['sideband_dBm_amp']),
                                np.max(pdict['sideband_dBm_amp'])]),
                'color': 'red',
                'marker': 'None',
                'linestyle': '--'
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
            'xlabel': 'Phase Off., $\\Delta\\phi_\\mathrm{IQ}$',
            'ylabel': 'Sideband Leakage $V_\\mathrm{LO-IF}$',
            'xunit': 'deg',
            'yunit': 'dBm',
            'title': f'{timestamp} {self.qb_names[0]}\n$V_\\mathrm{{LO-IF}}$ '
                     f'projected onto phase offset $\\Delta\\phi_\\mathrm{{IQ}}$'
        }

        if self.do_fitting:
            self.plot_dicts['optimum_in_phase_vs_sb_magn'] = {
                'fig_id': raw_phase_plot_name,
                'plotfn': self.plot_line,
                'xvals': np.array([phase_min, phase_min]),
                'yvals': np.array([np.min(pdict['sideband_dBm_amp']),
                                np.max(pdict['sideband_dBm_amp'])]),
                'color': 'red',
                'marker': 'None',
                'linestyle': '--'
            }