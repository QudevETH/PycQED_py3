from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    Step,
    IntermediateStep,
    AutomaticCalibrationRoutine,
    ROUTINES
)

from pycqed.measurement.calibration.automatic_calibration_routines.base. \
    base_automatic_calibration_routine import _device_db_client_module_missing

if not _device_db_client_module_missing:
    from pycqed.utilities.devicedb import utils as db_utils

from pycqed.measurement.spectroscopy import spectroscopy as spec
from pycqed.utilities.general import temporary_value

from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.device import Device
import time

import numpy as np
import logging
from typing import Dict, List, Any

log = logging.getLogger(ROUTINES)


class QubitSpectroscopy1DStep(spec.QubitSpectroscopy1D, Step):
    """Wrapper for QubitSpectroscopy1D experiment."""

    def __init__(self, routine: AutomaticCalibrationRoutine, **kwargs):
        """
        Initializes the QubitSpectroscopy1DStep class, which also includes
        initialization of the QubitSpectroscopy1D experiment.

        Args:
            routine (Step): The parent routine

        Keyword Arguments (for the Step constructor):
            step_label (str): A unique label for this step to be used in the
                configuration parameters files.
            settings (SettingsDictionary obj): The configuration parameters
                passed down from its parent. if None, the dictionary is taken
                from the Device object.
            qubits (list): A list with the Qubit objects which should be part of
                the step.
            settings_user (dict): A dictionary from the user to update the
                configuration parameters with.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            freq_center (float): Frequency around which the spectroscopy should
                be centered. It is possible to specify a string value that will
                 be parsed. It is possible to include "{current}" to use the
                 current qb.ge_freq() value. Defaults to qb.ge_freq() (if no
                 value is found in the dictionary).
            freq_range (float): Range of the qubit spectroscopy. The sweep
                points will extend from freq_center-freq_range/2 to
                freq_center+freq_range/2. Defaults to 200 MHz (if no value
                is found in the dictionary).
            pts (int): Number of points for the sweep. Defaults to 500 (if no
                value is found in the dictionary).
            spec_power (float): Spectroscopy power to be used. Defaults to
                15 dBm (if no value is found in the dictionary).
        """
        self.kw = kwargs

        Step.__init__(self, routine=routine, **kwargs)

        self.spec_powers = {}
        self.freq_ranges = {}
        self.freq_centers = {}
        self.pts = {}

        self.experiment_settings = self.parse_settings(
            self.get_requested_settings())

        spec.QubitSpectroscopy1D.__init__(self,
                                          dev=self.dev,
                                          **self.experiment_settings)

    def parse_settings(self, requested_kwargs) -> Dict[str, Any]:
        """
        Searches the keywords given in requested_kwargs in the configuration
        parameter dictionary and prepares the keywords to be passed to the
        QubitSpectroscopy1D class.

        Args:
            requested_kwargs (dict): Dictionary containing the names and the
            default values of the keywords needed for the QubitSpectroscopy1D
            class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the QubitSpectroscopy1D class.
        """
        kwargs = super().parse_settings(requested_kwargs)
        task_list = []

        for qb in self.qubits:
            task = {}
            # Extract the parameters necessary to build the task_list from
            # the configuration parameter dictionary.
            freq_center = self.get_param_value('freq_center',
                                               qubit=qb.name,
                                               default=qb.ge_freq())
            if isinstance(freq_center, str):
                freq_center = eval(
                    freq_center.format(current=qb.ge_freq()))
            freq_range = self.get_param_value('freq_range',
                                              qubit=qb.name,
                                              default=self.DEFAULT_FREQ_RANGE)
            pts = self.get_param_value('pts',
                                       qubit=qb.name,
                                       default=self.DEFAULT_PTS)
            spec_power = self.get_param_value('spec_power',
                                              qubit=qb.name,
                                              default=self.DEFAULT_SPEC_POWER)

            # Store the read values in class attributes so that they can be
            # accessed by a parent routine.
            self.freq_ranges[qb.name] = freq_range
            self.freq_centers[qb.name] = freq_center
            self.pts[qb.name] = pts
            self.spec_powers[qb.name] = (qb.spec_power, spec_power)

            # Frequency sweep points
            freqs = np.linspace(freq_center - freq_range / 2,
                                freq_center + freq_range / 2, pts)

            # Create the task list for the measurement
            task['qb'] = qb.name
            task['freqs'] = freqs
            task_list.append(task)

        kwargs['task_list'] = task_list
        return kwargs

    def run(self):
        """
        Runs the QubitSpectroscopy1D experiment and the analysis for it.
        The specified spectroscopy powers are used within the temporary_value
        context manager.
        """
        with temporary_value(*self.spec_powers.values()):
            # Set 'measure' and 'analyze' of QubitSpectroscopy1D to True in
            # order to use its autorun() function
            self.experiment_settings['measure'] = True
            self.experiment_settings['analyze'] = True
            self._update_parameters(**self.experiment_settings)
            self.autorun(**self.experiment_settings)

    DEFAULT_FREQ_RANGE = 200e6
    DEFAULT_PTS = 500
    DEFAULT_SPEC_POWER = 15


class AdaptiveQubitSpectroscopy(AutomaticCalibrationRoutine):
    """
    Routine to find the ge transition frequency via qubit spectroscopy.
    A series of qubit spectroscopies is performed. A Decision step decides
    whether a fit failed for some qubits and whether to rerun the spectroscopy
    for them.

    The user can specify the number of spectroscopies via the keyword
    "n_spectroscopies" in the configuration parameter dictionary. The settings
    of each spectroscopy can be specified by using their unique label.

    For example, a routine with 2 spectroscopies can have the following routine
     steps:

    1) QubitSpectroscopy1DStep (qubit_spectroscopy_<n>): Performs a qubit
        spectroscopy and fits the result with a Lorentzian to extract the ge
        transition frequency.
    2) Decision (decision_spectroscopy_<n>): Checks whether the fit was
        successful. The decision is made comparing the reduced chi-squared of
        the fit with the variance of the data.
        If the fit was not successful for certain qubits, another qubit
        spectroscopy is run (only for the unsuccessful qubits), followed by
        another Decision step. For example:

        3) QubitSpectroscopy1DStep (qubit_spectroscopy_<n>_repetition_<k>):
        4) Decision (decision_spectroscopy_<n>_repetition_<k>):
        5) QubitSpectroscopy1DStep (qubit_spectroscopy_<n>_repetition_<k+1>):
        6) Decision (decision_spectroscopy_<n>_repetition_<k+1>):
        (...)

        Additional steps will be added until the maximum number of repetitions
        (specified in the configuration parameter dictionary as
        "max_iterations") is reached. If this happens, no additional
        spectroscopies will be run on the qubits whose spectroscopy failed.

        The settings of these additional steps can be set manually by specifying
        the settings for each step in the configuration parameter dictionary
        (using the unique label of each step)
        Alternatively, by setting '"auto_repetition_settings": false' in the
        configuration parameter dictionary, the settings for the repetitions
        will be automatically chosen. Namely, at each repetition the range of
        the sweep and the density of the sweep points will be doubled.

    7) QubitSpectroscopy1DStep (qubit_spectroscopy_<n+1>): ...
    8) Decision (decision_spectroscopy_<n+1>): ...
        9) QubitSpectroscopy1DStep (qubit_spectroscopy_<n+1>_repetition_<k>): ...
        10) Decision (decision_spectroscopy_<n+1>_repetition_<k>): ...
        (...)
    """

    def __init__(
            self,
            dev: Device,
            qubits: List[QuDev_transmon],
            **kw,
    ):
        """
        Initialize the AdaptiveQubitSpectroscopy routine.

        Args:
            dev (Device): Device to be used for the routine
            qubits (list): The qubits which should be calibrated. By default,
                all qubits of the device are selected.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            n_spectroscopies (int): Number of (successful) spectroscopies that
                will be run.
            max_iterations (int): Maximum number of iterations that will be
                performed if a spectroscopy fails.
            auto_repetition_settings (bool): Whether the settings of the
                repeated spectroscopy should be automatically set. If True,
                the range of the sweep and the density of the sweep points will
                be doubled at each repetition.
            max_waiting_seconds (int): Maximum number of seconds to wait before
                running the Decision step. This is necessary because it might
                take some time before the analysis results are available.
        """

        super().__init__(
            dev=dev,
            qubits=qubits,
            **kw,
        )
        self.index_iteration = 1
        self.index_spectroscopy = 1

        # Store initial frequency of the qubits
        self.previous_freqs = {qb.name: qb.ge_freq() for qb in qubits}
        self.results = {}

        self.final_init(**kw)

    class Decision(IntermediateStep):
        """
        Decision step that decides to add another qubit spectroscopy if
        the Lorentzian fit of the previous spectroscopy was not successful.
        The fit is considered not successful if the reduced chi-squared is
        greater than the variance of the data minus the standard deviation
        of the variance.
        Additionally, it checks if the maximum number of iterations has been
        reached.
        """

        def __init__(self, routine: AutomaticCalibrationRoutine, **kw):
            """
            Initialize the Decision step.

            Args:
                routine (Step): AdaptiveQubitSpectroscopy routine.

            Keyword args:
                Keyword arguments that will be passed to :obj:`IntermediateStep`

            Configuration parameters (coming from the configuration parameter
            dictionary):
                max_kappa_fraction_sweep_range (float): Maximum value of kappa
                    as a fraction of the whole sweep range. If the kappa found
                    with the fit is bigger than this value, the fit will be
                    considered unsuccessful.
                max_kappa_absolute (float): Maximum value of kappa in Hz.
                    If the kappa found with the fit is bigger than this value,
                    the fit will be considered unsuccessful.
                min_kappa_fraction_sweep_range (float): Minimum value of kappa
                    as a fraction of the whole sweep range. If the kappa found
                    with the fit is smaller than this value, the fit will be
                    considered unsuccessful.
                min_kappa_absolute (float): Minimum value of kappa in Hz.
                    If the kappa found with the fit is smaller than this value,
                    the fit will be considered unsuccessful.
                max_waiting_seconds (float): Maximum number of seconds to wait
                    before running the Decision step. This is necessary because
                    it might take some time before the analysis results are
                    available.
            """
            super().__init__(routine=routine, **kw)

        def run(self):
            """Executes the decision step."""
            routine: AdaptiveQubitSpectroscopy = self.routine
            qubits_to_rerun = []
            qubits_failed = []
            for qb in self.qubits:
                max_iterations = self.get_param_value("max_iterations",
                                                      qubit=qb,
                                                      default=3)
                # Retrieve the QubitSpectroscopy1DStep run last
                qb_spec: QubitSpectroscopy1DStep = routine.routine_steps[-1]

                # Retrieve the necessary data from the analysis results
                max_waiting_seconds = self.get_param_value(
                    "max_waiting_seconds", default=5)
                for i in range(max_waiting_seconds):
                    try:
                        data = qb_spec.analysis.proc_data_dict[
                            'projected_data_dict'][qb.name]['PCA'][0]
                        red_chi = qb_spec.analysis.fit_res[qb.name].redchi
                        break
                    except AttributeError:
                        # FIXME: Unsure if this can also happen on real set-up
                        log.warning(
                            "Analysis not yet run on last QubitSpectroscopy1D "
                            "measurement, frequency difference not updated")
                        time.sleep(1)

                # Calculate the upper bound of the reduced chi-squared, namely
                # var(data) - std(var(data))
                n = len(data)
                mean = np.mean(data)
                std_err = np.std(data)
                var = std_err ** 2
                # FIXME Calculates the variance of the entire sweep.
                #  This is basically a the noise and only works when the
                #  majority of the data point are expected to have approximately
                #  the same value.
                #  See https://en.wikipedia.org/wiki/Variance#Distribution_of_the_sample_variance
                var_of_var = 1 / n * (np.mean(
                    (data - mean) ** 4) - (n - 3) / (n - 1) * var ** 2)
                red_chi_upper_bound = var - np.sqrt(var_of_var)

                # Check whether the fit was successful and store the qubits
                # in the corresponding list for further processing.
                if red_chi_upper_bound < red_chi:
                    fail_message = f'Fit error too high.'
                    success = False
                else:
                    success = True

                # Check whether the width of the resonance is within the
                # specified limits
                max_kappa_fraction_sweep_range = self.get_param_value(
                    "max_kappa_fraction_sweep_range", qubit=qb
                )
                min_kappa_fraction_sweep_range = self.get_param_value(
                    "min_kappa_fraction_sweep_range", qubit=qb
                )
                sweep_points_freq = qb_spec.analysis.sp[f"{qb.name}_freq"]
                sweep_range = np.max(sweep_points_freq) - np.min(
                    sweep_points_freq)

                max_kappa_absolute = self.get_param_value(
                    "max_kappa_absolute", qubit=qb
                )
                min_kappa_absolute = self.get_param_value(
                    "min_kappa_absolute", qubit=qb
                )
                kappa = qb_spec.analysis.fit_res[qb.name].values['kappa']
                if max_kappa_absolute is not None:
                    if kappa > max_kappa_absolute:
                        fail_message = f'f{kappa=}>{max_kappa_absolute}'
                        success = False
                if max_kappa_fraction_sweep_range is not None:
                    if kappa > max_kappa_fraction_sweep_range * sweep_range:
                        fail_message = (
                            f'f{kappa=}>'
                            f'{max_kappa_fraction_sweep_range * sweep_range}')
                        success = False
                if min_kappa_absolute is not None:
                    if kappa < min_kappa_absolute:
                        fail_message = f'f{kappa=}<{min_kappa_absolute}'
                        success = False
                if min_kappa_fraction_sweep_range is not None:
                    if kappa < min_kappa_fraction_sweep_range * sweep_range:
                        fail_message = (
                            f'f{kappa=}<'
                            f'{min_kappa_fraction_sweep_range * sweep_range}')
                        success = False

                if success:
                    if self.get_param_value('verbose'):
                        print(f"Lorentzian fit for {qb.name} was successful.")
                    # Update the frequency in the dictionary for next steps
                    self.routine.results[qb.name] = {
                        "measured_ge_freq": qb.ge_freq(),
                        "iterations_needed": routine.index_iteration
                    }
                    self.routine.previous_freqs[qb.name] = qb.ge_freq()
                elif routine.index_iteration < max_iterations:
                    if self.get_param_value('verbose'):
                        print(f"Lorentzian fit failed for {qb.name} since "
                              f"{fail_message}. Trying again.")
                    qubits_to_rerun.append(qb)
                    # Restore the previous frequency
                    qb.ge_freq(self.routine.previous_freqs[qb.name])
                else:
                    if self.get_param_value('verbose'):
                        print(f"Lorentzian fit failed for {qb.name}. since "
                              f"{fail_message}. Maximum number of iterations "
                              f"reached.")
                    qubits_failed.append(qb)
                    # Restore the previous frequency
                    qb.ge_freq(self.routine.previous_freqs[qb.name])

            # Decide how to modify the routine steps
            if len(qubits_to_rerun) > 0:
                # If there are some qubits whose spectroscopy should be rerun,
                # add another QubitSpectroscopy1DStep followed by another
                # Decision step
                routine.index_iteration += 1
                routine.add_rerun_qubit_spectroscopy_step(
                    index_spectroscopy=routine.index_spectroscopy,
                    index_iteration=routine.index_iteration,
                    qubits=qubits_to_rerun)
            else:
                # If there are no qubits whose spectroscopy should be rerun, it
                # means either that the fit was successful or that the maximum
                # number of iterations was exceeded. In the latter case,
                # we remove the qubit from the following spectroscopy steps
                indices_steps_to_remove = []
                for i, step in enumerate(routine.routine_template):
                    label, index = step[1].rsplit("_", 1)
                    index = int(index)
                    # Retrieve the steps with label
                    # "qubits_spectroscopy_<index>", where index is greater the
                    # index of the current spectroscopy
                    if label == "qubit_spectroscopy" \
                            and index > routine.index_spectroscopy:
                        # Get the settings of such steps and remove all the
                        # qubits_failed from the qubits that are supposed to
                        # be measured again
                        qubits = step[2]['qubits']
                        for qb in qubits_failed:
                            if qb in qubits:
                                qubits.remove(qb)
                        # If there are no more qubits left to measure, store the
                        # index of the spectroscopy and the following decision
                        # step in order to remove them afterwards
                        if not qubits:
                            indices_steps_to_remove += [i, i + 1]
                            log.warning(
                                f"All the qubits of step {step[1]} failed."
                                f"It will be removed together with the "
                                f"following Decision step.")

                # Remove the steps with only qubits whose spectroscopy failed
                # and exceeded the maximum number of iterations
                for index in reversed(indices_steps_to_remove):
                    routine.routine_template.pop(index)

                # Restore the iteration index to 1 for the next spectroscopy
                # round
                routine.index_iteration = 1
                routine.index_spectroscopy += 1

    def create_routine_template(self):
        """
        Creates the routine template for the AdaptiveQubitSpectroscopy
        routine.
        """
        super().create_routine_template()

        # Add the requested number of QubitSpectroscopy1DStep followed by
        # Decision steps
        for i in range(self.get_param_value("n_spectroscopies", default=2)):
            qb_spec_settings = {'qubits': self.qubits}
            self.add_step(QubitSpectroscopy1DStep,
                          f'qubit_spectroscopy_{i + 1}',
                          qb_spec_settings)
            decision_settings = {'qubits': self.qubits}
            self.add_step(self.Decision, f'decision_spectroscopy_{i + 1}',
                          decision_settings)

    def add_rerun_qubit_spectroscopy_step(self, index_spectroscopy,
                                          index_iteration, qubits):
        """
        Adds a next QubitSpectroscopy1DStep followed by a Decision step

        Args:
            index_spectroscopy (int): Index of the spectroscopy whose fit
                failed.
            index_iteration (int): Index of the iteration for the spectroscopy
                with index `index_spectroscopy`.
            qubits (list): List of qubits (QuDev_transmon objects) whose
                spectroscopy should be run again.
        """

        settings = {'QubitSpectroscopy1D': {"qubits": {}}}

        # If requested, retrieve the previous measurement settings and adapt
        # them before trying again (increase both range and points density)
        if self.get_param_value("auto_repetition_settings", default=False):
            previous_qb_spec: QubitSpectroscopy1DStep = self.routine_steps[-1]
            for qb in qubits:
                settings['QubitSpectroscopy1D']['qubits'][qb.name] = {
                    'freq_range': previous_qb_spec.freq_ranges[qb.name] * 2,
                    'freq_center': previous_qb_spec.freq_centers[qb.name],
                    'spec_power': previous_qb_spec.spec_powers[qb.name][1],
                    'pts': previous_qb_spec.pts[qb.name] * 4
                }

        # Add the steps right after the current one
        self.add_step(
            *[
                QubitSpectroscopy1DStep,
                f'qubit_spectroscopy_{index_spectroscopy}_'
                f'repetition_{index_iteration}',
                {
                    'settings': settings,
                    'qubits': qubits
                },
            ],
            index=self.current_step_index + 1,
        )
        self.add_step(
            *[
                self.Decision,
                f'decision_spectroscopy_{index_spectroscopy}_'
                f'repetition_{index_iteration}',
                {
                    'qubits': qubits
                },
            ],
            index=self.current_step_index + 2
        )
