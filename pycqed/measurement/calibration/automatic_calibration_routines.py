from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.utilities import hamiltonian_fitting_analysis as hfa
from pycqed.utilities.state_and_transition_translation import *
from pycqed.utilities.general import temporary_value
from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals
from pycqed.utilities.reload_settings import reload_settings

import pycqed.analysis.analysis_toolbox as a_tools
import numpy as np
import copy
import logging
import time
import pprint
import inspect
import collections.abc

log = logging.getLogger(__name__)


class RoutineTemplate(list):
    """
    Class to describe templates for (calibration) routines.

    The class is essentially a list of tuples that contain a class and
    corresponding settings of a step in a routine. Steps may be measurements,
    calibration routines, or intermediate steps.

    FIXME it is confusing but right now I the parameter is part of the routine
    and the routine template does not. In a sense, the parameters form the
    instructions to construct the template. I am not sure if this is the best
    way to do it. - Joost
    """

    def __init__(
        self,
        routine_template,
        global_settings=None,
        routine=None,
        **kw,
    ):
        # initializing routine template as list object
        super().__init__(routine_template)

        if routine is not None:
            self.routine = routine

        if global_settings is not None:
            self.global_settings = global_settings
        else:
            self.global_settings = {}

    def get_step_class_at_index(self, index):
        """
        Returns the step class for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self[index][0]

    def get_step_settings_at_index(self, index):
        """
        Returns the settings for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        settings = {}
        settings.update(copy.copy(self.global_settings))
        settings.update(copy.copy(self[index][1]))
        return settings

    def get_step_tmp_vals_at_index(self, index):
        """
        Returns the temporary values of the step at index.
        """
        try:
            return self[index][2]
        except IndexError:
            return []

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """
        Extends the temporary values of the step at index. If the step does not
        have any temporary values, it sets the temporary values to the passed
        temporary values.
        """
        try:
            self[index][2].extend(tmp_vals)
        except IndexError:
            self[index].append(tmp_vals)

    def update_settings_at_index(self, settings, index):
        """
        Updates the settings of the step at index.
        """
        self[index][1].update(settings)

    def update_all_step_settings(self, settings):
        """
        Updates all settings of all steps in the routine.
        """
        for i, x in enumerate(self):
            self.update_settings_at_index(settings, index=i)

    def update_settings(self, settings_list):
        """
        Updates all settings of the routine. Settings_list must be a list of
        dictionaries of the same length as the routine.
        """
        for i, x in enumerate(settings_list):
            self.update_settings_at_index(settings=x, index=i)

    def view(
        self,
        print_global_settings=True,
        print_parameters=True,
        print_tmp_vals=False,
    ):
        """
        Prints a user-friendly representation of the routine template

        Args:
            print_global_settings (bool): If True, prints the global settings
                of the routine.
            print_parameters (bool): If True, prints the parameters of the
                routine.
        """
        try:
            print(self.routine.name)
        except AttributeError:
            pass

        if print_global_settings:
            print("Global settings:")
            pprint.pprint(self.global_settings)
            print()

        if print_parameters:
            try:
                print("Parameters:")
                pprint.pprint(self.routine.parameters)
                print()
            except AttributeError:
                pass

        for i, x in enumerate(self):
            print(f"Step {i}, {x[0].__name__}")
            print("Settings:")
            pprint.pprint(x[1], indent=4)

            if print_tmp_vals:
                try:
                    print("Temporary values:")
                    pprint.pprint(x[2], indent=4)
                except IndexError:
                    pass
            print()

    def __str__(self):
        """
        Returns a string representation of the routine template.

        FIXME this representation does not include the tmp_vals of the steps.
        """
        try:
            s = self.routine.name + "\n"
        except AttributeError:
            s = ""

        if self.global_settings:
            s += "Global settings:\n"
            s += pprint.pformat(self.global_settings) + "\n"

        try:
            if self.routine.settings:
                s += "Parameters:\n"
                s += pprint.pformat(self.global_settings) + "\n"
        except AttributeError:
            pass

        for i, x in enumerate(self):
            s += f"Step {i}, {x[0].__name__}, {x[1]}\n"
        return s

    def step_name(self, index):
        """
        Returns the name of the step at index.b
        """
        return self.get_step_class_at_index(index).__name__

    def add_step(
        self, step_class, step_settings, step_tmp_vals=None, index=None
    ):
        """
        Adds a step to the routine.
        """
        if step_tmp_vals is None:
            step_tmp_vals = []

        if index is None:
            super().append([step_class, step_settings, step_tmp_vals])
        else:
            super().insert(index, [step_class, step_settings, step_tmp_vals])

    @staticmethod
    def check_step(step):
        assert isinstance(step, list), "Step must be a list"
        assert (
            len(step) == 2 or len(step) == 3
        ), "Step must be a list of length 2 or 3 (to include temporary values)"
        assert isinstance(step[0], type), (
            "The first element of the step "
            "must be a class (e.g. measurement or a calibration routine)"
        )
        assert isinstance(step[1], dict), (
            "The second element of the step "
            "must be a dictionary containing settings"
        )

    def __getitem__(self, i):
        """
        Overloading of List.__getitem__ to ensure type RoutineTemplate is
        preserved.

        Arguments:
            i: index or slice

        Returns: element or new RoutineTemplate instance
        """
        new_data = super().__getitem__(i)
        if isinstance(i, slice):
            new_data = self.__class__(new_data)
            new_data.global_settings = copy.copy(self.global_settings)
        return new_data


class IntermediateStep:
    """
    Class used for defining intermediate steps between automatic calibration
    steps.
    """

    def __init__(self, routine, **kw):
        self.routine = routine
        self.kw = kw

    def run(self):
        """
        Intermediate processing step to be overridden by Children (which are
        routine specific).
        """
        pass


class AutomaticCalibrationRoutine:
    """
    Base class for general automated calibration routines
    """

    def __init__(
        self,
        qubits,
        autorun=True,
        update=True,
        save_instrument_settings=False,
        **kw,
    ):
        """
        Initializes the routine.

        NOTE that currently only one qubit is supported. If a list of multiple
            qubits is passed, only the first qubit is used.

        Args:
            qubits (list): list of qubits to be used in the routine
            autorun (bool): if True, the routine will be run immediately after
                initialization.
            update (bool): if True, the routine will overwrite qubit attributes
                with the values found in the routine. Note that if the routine
                is used as subroutine, this should be set to True.
            save_instrument_settings (bool): if True, the routine will save the
                instrument settings before and after the routine.

        kw (passed to the routine template):
            dev: device to be used for the routine
            parameters_device: device parameters. See _DEFAULT_PARAMETERS
                attribute of the routine of interest for the desired structure.
            parameters_user: user parameters. See _DEFAULT_PARAMETERS
                attribute of the routine of interest for the desired structure.
            get_parameters_from_qubit_object: if True, the routine will try to
                get the parameters from the qubit object. Default is False.
            verbose: if True, the routine will print out the progress of the
                routine. Default is True.

        FIXME add support for loading the device settings directly from a file.
        """
        # attributes from args
        self.qubits = qubits
        self.save_instrument_settings = save_instrument_settings

        # attributes from kwargs
        self.verbose = kw.get("verbose", True)
        self.dev = kw.pop("dev", None)
        self.DCSources = kw.pop("DCSources", None)

        # storing kwargs
        self.kw = kw
        self.kw["update"] = update

        # only one qubit is supported
        if len(qubits) > 1:
            log.warning(
                "Only one qubit is currently supported. Choosing first qubit."
            )
            self.qubits = [qubits[0]]
        self.qubit = qubits[0]

        # MC - trying to get it from either the device or the qubits
        for source in [self.dev] + self.qubits:
            try:
                self.MC = source.instr_mc.get_instr()
                break
            except:
                pass

        # loading hierarchical settings and creating initial routine
        self.create_initial_routine()

        # autorun
        if autorun:
            # FIXME: if the init does not finish the object does not exist and
            # the routine results are not accesible
            try:
                self.run()
            except:
                log.error(
                    "Autorun failed to fully run, concluded routine steps"
                    "are stored in the routine_steps attribute.",
                    exc_info=1,
                )

    def create_routine_template(self):
        """
        Creates routine template. Can be overwritten or extended by children
        for more complex routines that require adaptive creation.
        """
        # create RoutineTemplate based on _default_routine_template
        self.routine_template = copy.deepcopy(self._DEFAULT_ROUTINE_TEMPLATE)
        self.routine_template.routine = self
        self.routine_template.parameters = self.parameters

        # standard global settings
        try:
            delegate_plotting = self.parameters["Routine"]["delegate_plotting"]
        except KeyError:
            delegate_plotting = False

        self.routine_template.global_settings.update(
            {
                "dev": self.dev,
                "qubits": self.qubits,
                "update": True,  # all subroutines should update relevant params
                "delegate_plotting": delegate_plotting,
            }
        )

        # add user specified global settings
        update_nested_dictionary(
            self.routine_template.global_settings,
            self.kw.get("global_settings", {}),
        )

    def prepare_step(self, i=None):
        """
        Prepares the next step in the routine. That is, it
        initializes the measurement object.

        Args:
            qubits: qubits on which to perform the measurement
            i: index of the step to be prepared. If None, the default is set to
                the current_step_index
        """

        if i is None:
            i = self.current_step_index

        # Setting step class and settings
        step_class = self.get_step_class_at_index(i)
        step_settings = self.get_step_settings_at_index(i)

        # Setting the temporary values
        self.current_step_tmp_vals = self.get_step_tmp_vals_at_index(i)

        # Update print
        if self.verbose:
            print(
                f"{self.name}, step {i} "
                f"({self.routine_template.step_name(index=i)}), preparing..."
            )

        # Executing the step with corresponding settings
        if issubclass(step_class, qbcal.SingleQubitGateCalibExperiment):
            step = step_class(measure=False, analyze=False, **step_settings)
        elif issubclass(step_class, IntermediateStep):
            step = step_class(routine=self, **step_settings)
        elif issubclass(step_class, AutomaticCalibrationRoutine):
            step = step_class(autorun=False, **step_settings)
        else:
            log.error(
                f"automatic subroutine is not compatible (yet) with the "
                f"current step class {step_class}"
            )
        self.current_step = step
        self.current_step_settings = step_settings

    def execute_step(self):
        """
        Executes the current step (routine.current_step) in the routine and
        writes the result in the routine_steps list.
        """
        if self.verbose:
            j = self.current_step_index
            print(
                f"{self.name}, step {j} "
                f"({self.routine_template.step_name(index=j)}), executing..."
            )

        settings = self.current_step_settings

        if isinstance(self.current_step, AutomaticCalibrationRoutine):
            self.current_step.run()
        elif isinstance(self.current_step, IntermediateStep):
            self.current_step.run()
        elif isinstance(
            self.current_step, qbcal.SingleQubitGateCalibExperiment
        ):
            self.current_step.run_measurement(**settings)
            self.current_step.run_analysis(**settings)

        self.routine_steps.append(self.current_step)
        self.current_step_index += 1

    def run(self, start_index=None, stop_index=None):
        """
        Runs the complete automatic calibration routine. In case the routine was
        already completed, the routine is reset and run again. In case the
        routine was interrupted, it will run from the last completed step, the
        index of which is saved in the current_step_index attribute of the routine.
        Additionally, it is possible to start the routine from a specific step.

        Args:
            start_index: index of the step to start with.
            stop_index: index of the step to stop before. The step at this index
                will NOT be executed. Indeces start at 0.

                For example, if a routine consists of 3 steps, [step0, step1,
                step2], then the method will stop before step2 (and thus after
                step1), if stop_index is set to 2.

        FIXME There's an issue when starting from a given start index. The
        routine_steps is only wiped if the routine ran completely and is reran
        from the start. In the future, it might be good to implement a way so
        the user can choose if previous results should be wiped or not (that is,
        if routine_steps should be wiped or not).
        """
        routine_name = self.name

        # saving instrument settings pre-routine
        if (
            self.save_instrument_settings
            or not self.parameters["Routine"]["update"]
        ):
            # saving instrument settings before the routine
            self.MC.create_instrument_settings_file(
                f"pre-{self.name}_routine-settings"
            )
            self.preroutine_timestamp = a_tools.get_last_n_timestamps(
                1,
            )[0]
        else:
            # registering start of routine so all data in measurement period can
            # be retrieved later to determine the Hamiltonian model
            self.preroutine_timestamp = self.MC.get_datetimestamp()

        # rerun routine if already finished
        if (len(self.routine_template) != 0) and (
            self.current_step_index >= len(self.routine_template)
        ):
            self.create_initial_routine(load_parameters=False)
            self.run()
            return

        # start and stop indeces
        if start_index is not None:
            self.current_step_index = start_index
        elif self.current_step_index >= len(self.routine_template):
            self.current_step_index = 0

        if stop_index is None:
            stop_index = np.Inf

        # running the routine
        while self.current_step_index < len(self.routine_template):
            j = self.current_step_index
            step_name = self.routine_template.step_name(index=j)

            # preparing the next step (incl. temporary values)
            self.prepare_step()

            # interrupting if we reached the stop condition
            if self.current_step_index >= stop_index:
                if self.verbose:
                    print(
                        f"Partial routine {routine_name} stopped before "
                        f"executing step {j} ({step_name})."
                    )
                return

            # executing the step
            with temporary_value(*self.current_step_tmp_vals):
                self.execute_step()

            if self.verbose:
                print(f"{routine_name}, step {j} ({step_name}), done!", "\n")

        if self.verbose:
            print(f"Routine {routine_name} finished!")

        # saving instrument settings post-routine
        if (
            self.save_instrument_settings
            or not self.parameters["Routine"]["update"]
        ):
            # saving instrument settings after the routine
            self.MC.create_instrument_settings_file(
                f"post-{routine_name}_routine-settings"
            )

        # reloading instrument settings if update is False
        if not self.parameters["Routine"]["update"]:
            if self.verbose:
                print(
                    f"Reloading instrument settings from before routine "
                    f"(ts {self.preroutine_timestamp})"
                )

            reload_settings(
                self.preroutine_timestamp,
                qubits=self.qubits,
                dev=self.dev,
                DCSources=self.DCSources,
            )

    def create_initial_parameters(self):
        """
        Creates the initial parameters for the routine.

        Keyword arguments are accessed through self. Relevant kwargs stored in
        self are:
            get_parameters_from_qubit_object: if True, initial guesses and
                estimates for the transmon parameters are retrieved from the
                (transmon) qubit object.
            parameters_device: nested dictionary containing relevant parameters
                of the device for the routine (second highest priority).
            parameters_user: nested dictionary containing relevant parameters
                from the user (highest priority).

        FIXME: In the future, we might want to swap the hierarchy and give
        priority to the settings in the qubit object, but that would require
        implementing plausibility checks and falling back to the config file if
        there is no reasonable value set in the qubit object.
        """

        # Adjusting settings with hierarchy
        # user input > set-up specific settings > default settings
        self.parameters = {}

        # layer 1: bare default settings (device independent)
        self.parameters = copy.deepcopy(self._DEFAULT_PARAMETERS)

        # layer 2.1: device specific settings (from qubit object)
        if self.kw.get("get_parameters_from_qubit_object", False):
            update_nested_dictionary(
                self.parameters, {"Transmon": self.parameters_qubit}
            )

        # layer 2.2: device specific settings (from configuration file)
        parameters_device = self.kw.get("parameters_device", {})
        update_nested_dictionary(self.parameters, parameters_device)

        # layer 3.1: user settings
        parameters_user = self.kw.get("parameters_user", {})
        update_nested_dictionary(self.parameters, parameters_user)

        update_nested_dictionary(
            self.parameters,
            {
                "Routine": {"update": self.kw["update"]},
            },
        )

        # layer 3.2: user settings from init
        update_nested_dictionary(
            self.parameters, {"Routine": self.parameters_init}
        )

    def create_initial_routine(self, load_parameters=True):
        """
        Creates (or recreates) initial routine by defining the routine
        template, set routine_steps to an empty array, and setting the
        current step to 0.

        NOTE this method wipes the results of the previous run stored in
        routine_steps.
        """
        self.routine_steps = []
        self.current_step_index = 0

        # loading initial parameters. Note that if load_parameters=False,
        # the parameters are not reloaded and thus remain the same. This is
        # desired when wanting to rerun a routine
        if load_parameters:
            self.create_initial_parameters()

        self.create_routine_template()

        # making sure all subroutines update relevant parameters
        self.routine_template.update_all_step_settings({"update": True})

    @property
    def parameters_qubit(self):
        """
        Returns: the parameters of the qubit, including the read-out frequency,
            the anharmonicity and (if present) the latest Hamiltonian model
            parameters containing the total Josephson energy, the charging
            energy, voltage per phi0, the dac voltage, the asymmetry, the
            coupling constant and bare read-out resonator frequency (overwriting
            the previous frb value).

        FIXME: The selection of parameters extracted from the qb is currently
        tailored to the first example use cases. This either needs to be
        generalized to extract more parameters here, or we could decide the
        concrete routines could override the method to extract their specific
        parameters.
        """
        qb = self.qubit

        parameters = {}

        hamfit_model = qb.fit_ge_freq_from_dc_offset()

        # extracting parameters from the qubit
        parameters.update(
            {
                "fr": hamfit_model.get("fr", qb.ro_freq()),
                "anharmonicity": qb.anharmonicity(),
            }
        )

        # getting transmon parameters from present Hamiltonian model if it exists
        parameters.update(hamfit_model)

        return parameters

    @property
    def parameters_init(self):
        parameters = copy.copy(self.kw)
        return parameters

    def view(self, **kw):
        """
        Prints a user friendly representation of the routine settings
        """
        self.routine_template.view(**kw)

    def update_settings_at_index(self, settings: dict, index):
        """
        Updates the calibration settings for a specific step in the routine.

        Args:
            settings: dictionary of settings to be updated
            index: index of the step settings to be updated.
        """
        self.routine_template.update_settings_at_index(settings, index)

    def get_step_class_at_index(self, index):
        """
        Returns the step class for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_class_at_index(index)

    def get_step_settings_at_index(self, index):
        """
        Returns the settings for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_settings_at_index(index)

    def get_step_tmp_vals_at_index(self, index):
        """
        Returns the temporary values for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_tmp_vals_at_index(index)

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """
        Sets the temporary values for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        self.routine_template.extend_step_tmp_vals_at_index(
            tmp_vals=tmp_vals, index=index
        )

    def add_step(
        self, step_class, step_settings, step_tmp_vals=None, index=None
    ):
        self.routine_template.add_step(
            step_class, step_settings, step_tmp_vals=step_tmp_vals, index=index
        )

    @property
    def global_settings(self):
        return self.routine_template.global_settings

    @property
    def name(self):
        """
        Returns the name of the routine.
        """
        # Name depends on whether or not the object is initialized.
        if type(self) is not type:
            return type(self).__name__
        else:
            try:
                return self.__name__
            except:
                return "AutomaticCalibration"

    # initializing necessary attributes, should/can be overridden by children
    _DEFAULT_PARAMETERS = {}
    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([])


# Special Automatic calibration routines


class PiPulseCalibration(AutomaticCalibrationRoutine):
    """
    Pi-pulse calibration consisting of a Rabi experiment followed by a Ramsey
    experiment
    """

    def __init__(
        self,
        qubits,
        rabi_amps=None,
        ramsey_delays=None,
        artificial_detuning=None,
        transition="ge",
        **kw,
    ):
        """
        Pipulse calibration routine consisting of one Rabi and one Ramsey

        Args:
            qubits: qubits on which to perform the measurement
            rabi_amps: list of amplitudes for the Rabi experiment
            ramsey_delays: list of delays for the Ramsey experiment
            artificial_detuning: artificial detuning for the Ramsey
            transition: transition used for the Pi-pulse calibration experiment
            autorun: whether to automatically run the routine

        Keyword Arguments:
            pts: number of points to use for the Rabi experiment
            v_high: high voltage for the Rabi experiment
            v_low: low voltage for the Rabi experiment
            delta_t: difference between final and initial time in Ramsey
                measurement
            t0: initial time in Ramsey measurement
            n_periods: number of periods
            pts_per_period: number of points per period
        """
        transition = transition_to_str(transition)

        # arguments that are not arguments of the super init will be considered
        # as key words for the super init.
        super().__init__(
            qubits=qubits,
            rabi_amps=rabi_amps,
            ramsey_delays=ramsey_delays,
            artificial_detuning=artificial_detuning,
            transition=transition,
            **kw,
        )

    def create_routine_template(self):
        """
        Creates the routine template for the PiPulseCalibration routine.
        """
        super().create_routine_template()

        # Rabi
        rabi_amps = self.kw.get("rabi_amps", None)
        if rabi_amps is None:
            rabi_amps = self.rabiParameters(**self.parameters["Rabi"])

        self.add_step(
            qbcal.Rabi,
            {
                "transition_name": self.parameters["Routine"]["transition"],
                "amps": rabi_amps,
                "update": True,
            },
        )

        # Ramsey
        ramsey_delays = self.kw.get("ramsey_delays", None)
        artificial_detuning = self.kw.get("artificial_detuning", None)
        if ramsey_delays is None or artificial_detuning is None:
            (
                ramsey_delays,
                artificial_detuning,
            ) = self.ramseyParameters(**self.parameters["Ramsey"])

        self.add_step(
            qbcal.Ramsey,
            {
                "transition_name": self.parameters["Routine"]["transition"],
                "artificial_detuning": artificial_detuning,
                "delays": ramsey_delays,
                "update": True,
            },
        )

    _DEFAULT_PARAMETERS = {
        "Routine": {
            "delegate_plotting": False,
            "transition": "ge",
        },
        # default parameters for the Rabi experiment
        "Rabi": {
            "v_low": 0,  # lowest voltage
            "v_high": 1,  # highest voltage
            "pts": 31,  # number of measurement points
        },
        # default parameters for Ramsey measurement
        "Ramsey": {
            "delta_t": 210e-9,  # difference between final and initial time
            "t0": 60e-9,  # initial time
            "n_periods": 6,  # number of periods
            "pts_per_period": 5,  # points per period
        },
    }

    @staticmethod
    def rabiParameters(
        v_low=_DEFAULT_PARAMETERS["Rabi"]["v_low"],
        v_high=_DEFAULT_PARAMETERS["Rabi"]["v_high"],
        pts=_DEFAULT_PARAMETERS["Rabi"]["pts"],
    ):
        """
        Returns the parameter amps for a Rabi experiment, based on the low and
        high voltage, plus the number of points.

        Args:
            v_low: lowest voltage
            v_high: highest voltage
            pts: number of measurement points

        Returns:
            amps: list of amplitudes for the Rabi experiment
        """
        amps = np.linspace(v_low, v_high, pts)
        return amps

    @staticmethod
    def ramseyParameters(
        delta_t=_DEFAULT_PARAMETERS["Ramsey"]["delta_t"],
        t0=_DEFAULT_PARAMETERS["Ramsey"]["t0"],
        n_periods=_DEFAULT_PARAMETERS["Ramsey"]["n_periods"],
        pts_per_period=_DEFAULT_PARAMETERS["Ramsey"]["pts_per_period"],
    ):
        """
        Returns the parameters delays and artificial detuning for a Ramsey
            experiment.

        Args:
            delta_t: difference between final and initial time in Ramsey
                measurement
            t0: initial time in Ramsey measurement
            n_periods: number of periods
            pts_per_period: number of points per period
        """

        delays = np.linspace(t0, t0 + delta_t, pts_per_period * n_periods)
        artificial_detuning = n_periods / delta_t

        return delays, artificial_detuning


class FindFrequency(AutomaticCalibrationRoutine):
    """
    Routine to find frequency of a given transmon transition.
    """

    def __init__(
        self,
        qubits,
        transition="ge",
        adaptive=False,
        allowed_delta_f=0.2e6,
        max_iterations=3,
        **kw,
    ):
        """
        Routine to find frequency of a given transmon transition.

        Args:
            qubits: list of qubits to be calibrated
            transition: transition to be calibrated
            adaptive: whether to use adaptive rabi and ramsey settings
            allowed_difference: allowed frequency difference in Hz between
                old and new frequency (convergence criterion)
            max_iterations: maximum number of iterations

        Keyword Arguments:
            autorun: whether to run the routine automatically
            rabi_amps: list of amplitudes for the (initial) Rabi experiment
            ramsey_delays: list of delays for the (initial) Ramsey experiment
            delta_t: time duration for (initial) Ramsey
            t0: time for (initial) Ramsey
            n_points: number of points per period for the (initial) Ramsey
            pts_per_period: number of points per period to use for the (initial)
                Ramsey
            artificial_detuning: artificial detuning for the initial Ramsey
                experiment
            f_factor: factor to multiply the frequency by (only relevant
                for displaying results)
            f_unit: unit of the frequency (only relevant for displaying
                results)
            delta_f_factor: factor to multiply the frequency difference by
                (only relevant for displaying results)
            delta_f_unit: unit of the frequency difference (only relevant for
                displaying results)

        For key words of super().__init__(), see AutomaticCalibrationRoutine for
         more details.
        """
        super().__init__(
            qubits=qubits,
            transition=transition,
            adaptive=adaptive,
            allowed_delta_f=allowed_delta_f,
            max_iterations=max_iterations,
            **kw,
        )

        # defining initial and allowed frequency difference
        self.delta_f = np.Infinity
        self.iteration = 1

    @property
    def parameters_init(self):
        # writing keyword arguments into correct categroty of parameters dict
        for x in ["rabi_amps", "ramsey_delays", "artificial_detuning"]:
            self.parameters["PiPulseCalibration"][x] = self.kw.pop(
                x, self.parameters["PiPulseCalibration"][x]
            )
        parameters = super().parameters_init
        return parameters

    class Decision(IntermediateStep):
        def __init__(self, routine, index, **kw):
            """
            Decision step that decides to add another round of Rabi-Ramsey to
            the FindFrequency routine based on the difference between the
            results of the previous and current Ramsey experiments.
            Additionally, it checks if the maximum number of iterations has been
            reached.

            Args:
                routine: FindFrequency routine
                index: index of the decision step (necessary to find the
                    position of the Ramsey measurement in the routine)

            Keyword Arguments:
                max_waiting_seconds: maximum number of seconds to wait for the
                    results of the previous Ramsey experiment to arrive.
            """
            super().__init__(routine, index=index, **kw)

        def run(self):
            """
            Executes the decision step.
            """
            routine = self.routine
            qubit = self.routine.qubit
            index = self.kw["index"]

            # saving some typing for parameters that are only read ;)
            allowed_delta_f = routine.parameters["Routine"]["allowed_delta_f"]
            f_unit = routine.parameters["Routine"]["f_unit"]
            f_factor = routine.parameters["Routine"]["f_factor"]
            delta_f_unit = routine.parameters["Routine"]["delta_f_unit"]
            delta_f_factor = routine.parameters["Routine"]["delta_f_factor"]
            max_iterations = routine.parameters["Routine"]["max_iterations"]
            transition = routine.parameters["Routine"]["transition"]

            # finding the ramsey experiment in the pipulse calibration
            pipulse_calib = routine.routine_steps[index - 1]
            ramsey = pipulse_calib.routine_steps[-1]

            # transition frequency from last Ramsey
            freq = qubit[f"{transition}_freq"]()

            # retrieving the frequency difference
            max_waiting_seconds = self.kw.get("max_waiting_seconds", 1)
            for i in range(max_waiting_seconds):
                try:
                    routine.delta_f = (
                        ramsey.analysis.proc_data_dict["analysis_params_dict"][
                            qubit.name
                        ]["exp_decay"]["new_qb_freq"]
                        - ramsey.analysis.proc_data_dict[
                            "analysis_params_dict"
                        ][qubit.name]["exp_decay"]["old_qb_freq"]
                    )
                    break
                except KeyError:
                    log.warning(
                        "Could not find frequency difference between current "
                        "and last Ramsey measurement, delta_f not updated"
                    )
                    break
                except AttributeError:
                    # FIXME Unsure if this can also happen on real set-up
                    log.warning(
                        "Analysis not yet run on last Ramsey measurement, "
                        "frequency difference not updated"
                    )
                    time.sleep(1)

            # progress update
            if self.routine.verbose:
                print(
                    f"Iteration {routine.iteration}, {transition}-freq "
                    f"{freq/f_factor} {f_unit}, frequency "
                    f"difference = {routine.delta_f/delta_f_factor} "
                    f"{delta_f_unit}"
                )

            # check if the absolute frequency difference is small enough
            if np.abs(routine.delta_f) < allowed_delta_f:
                # success
                if self.routine.verbose:
                    print(
                        f"{transition}-frequency found to be"
                        f"{freq/f_factor} {f_unit} within "
                        f"{allowed_delta_f/delta_f_factor} "
                        f"{delta_f_unit} of previous value."
                    )

            elif routine.iteration < max_iterations:
                # no success yet, adding a new rabi-ramsey and decision step
                if self.routine.verbose:
                    print(
                        f"Allowed error ("
                        f"{allowed_delta_f/delta_f_factor} "
                        f"{delta_f_unit}) not yet achieved, adding new"
                        " round of PiPulse calibration..."
                    )

                routine.add_next_pipulse_step(index=index + 1)

                routine.add_step(
                    FindFrequency.Decision,
                    {"index": index + 2},
                    index=index + 2,
                )

                routine.iteration += 1
                return

            else:
                # no success yet, passed max iterations
                msg = (
                    f"FindFrequency routine finished for {qubit.name}, "
                    "desired precision not necessarily achieved within the "
                    f"maximum number of iterations ({max_iterations})."
                )
                log.warning(msg)

                if self.routine.verbose:
                    print(msg)

            if self.routine.verbose:
                # printing termination update
                print(
                    f"FindFrequency routine finished: "
                    f"{transition}-frequencies for {qubit.name} "
                    f"is {freq/f_factor} {f_unit}."
                )

    def create_routine_template(self):
        """
        Creates the routine template for the FindFrequency routine.
        """
        super().create_routine_template()

        # PiPulse calibration
        pipulse_settings = {
            "transition": self.parameters["Routine"]["transition"]
        } | self.parameters["PiPulseCalibration"]

        self.add_step(PiPulseCalibration, pipulse_settings)

        # Decision step
        decision_settings = {"index": 1}
        self.add_step(self.Decision, decision_settings)

    def add_next_pipulse_step(self, index, adaptive=True):
        """
        Adds a next pipulse step at the specified index in the FindFrequency
        routine.
        """
        qubit = self.qubit
        transition = self.parameters["Routine"]["transition"]

        rabi_amps = self.parameters["PiPulseCalibration"].get("rabi_amps", None)
        ramsey_delays = self.parameters["PiPulseCalibration"].get(
            "ramsey_delays", None
        )
        artificial_detuning = self.parameters["PiPulseCalibration"].get(
            "artificial_detuning", None
        )

        if not adaptive:
            if ramsey_delays is None or artificial_detuning is None:
                raise ValueError(
                    "If adaptive is False, rabi_amps, ramsey_delays"
                    "and artificial_detuning must be specified as key words"
                )

        if adaptive:
            # Retrieving T2_star and pi-pulse amplitude
            if transition == "ge":
                T2_star = qubit.T2_star() if qubit.T2_star() else 0
                amp180 = qubit.ge_amp180() if qubit.ge_amp180() else 0
            elif transition == "ef":
                T2_star = qubit.T2_star_ef() if qubit.T2_star_ef() else 0
                amp180 = qubit.ef_amp180() if qubit.ef_amp180() else 0
            else:
                raise ValueError('transition must either be "ge" or "ef"')

            # Amplitudes for Rabi
            # 1) if passed in init
            # 2) v_high based on current pi-pulse amplitude
            # 3) v_high based on default value
            if rabi_amps is None:
                rabi_pts = self.parameters["PiPulseCalibration"]["pts"]
                if amp180:
                    rabi_amps = np.linspace(0, amp180, rabi_pts)
                else:
                    rabi_amps = np.linspace(
                        0,
                        self.parameters["PiPulseCalibration"]["v_high"],
                        rabi_pts,
                    )

            # Delays and artificial detuning for Ramsey
            if ramsey_delays is None or artificial_detuning is None:
                # defining delta_t for Ramsey
                # 1) if passed in init
                # 2) based on T2_star
                # 3) based on default
                if self.parameters["Routine"]["use_T2_star"]:
                    delta_t = T2_star
                else:
                    delta_t = self.parameters["PiPulseCalibration"]["delta_t"]

                (
                    ramsey_delays,
                    artificial_detuning,
                ) = PiPulseCalibration.ramseyParameters(
                    delta_t=delta_t,
                    t0=self.parameters["PiPulseCalibration"]["t0"],
                    n_periods=self.parameters["PiPulseCalibration"][
                        "n_periods"
                    ],
                    pts_per_period=self.parameters["PiPulseCalibration"][
                        "pts_per_period"
                    ],
                )

        self.add_step(
            *[
                PiPulseCalibration,
                {
                    "qubits": [qubit],
                    "rabi_amps": rabi_amps,
                    "ramsey_delays": ramsey_delays,
                    "artificial_detuning": artificial_detuning,
                    "transition": transition,
                    "update": True,
                },
            ],
            index=index,
        )

    _DEFAULT_PARAMETERS = {
        "Routine": {
            "delegate_plotting": False,
            "max_iterations": 3,
            "allowed_delta_f": 1e6,
            "transition": "ge",
            "f_factor": 1e9,
            "f_unit": "GHz",
            "delta_f_factor": 1e6,
            "delta_f_unit": "MHz",
            "adaptive": False,
            "use_T2_star": False,
        },
        "PiPulseCalibration": {
            "rabi_amps": None,
            "ramsey_delays": None,
            "artificial_detuning": None,
        }
        | copy.deepcopy(PiPulseCalibration._DEFAULT_PARAMETERS["Rabi"])
        | copy.deepcopy(PiPulseCalibration._DEFAULT_PARAMETERS["Ramsey"]),
    }


class HamiltonianFitting(
    AutomaticCalibrationRoutine, hfa.HamiltonianFittingAnalysis
):
    def __init__(
        self,
        qubit,
        fluxlines_dict,
        use_prior_model,
        measurements=None,
        flux_to_voltage_and_freq_guess=None,
        save_instrument_settings=True,
        **kw,
    ):
        """
        Constructs a HamiltonianFitting routine used to determine a Hamiltonian
        model for a transmon qubit.

        Args:
            qubits: qubits to perform the calibration on. Note! Only supports
                one qubit input.
            fluxlines_dict: fluxlines_dict object for accessing and changing
                the dac voltages (needed in reparking ramsey)
            use_prior_model: whether to use the prior model (stored in qubit
                object) to determine the guess frequencies and voltages for the
                flux values of the measurement. If True,
                flux_to_voltage_and_freq_guess will automatically be generated
                using this model (and can be left None).
            measurements: dictionary containing the fluxes and transitions to
                measure for determining the Hamiltonian model. See default
                below on how to input the dictionary. The measurements must
                include at least two sweet spots (meaning the corresponding
                flux should be divisible by 0.5*phi0), which should be the first
                two keys of the dictionary. If None, a default measurement
                dictionary is chosen containing ge-frequency measurements at the
                zeroth upper sweet spot (phi = 0*phi0), the left lower sweet
                spot (phi = -0.5*phi0) and the mid voltage (phi = -0.25*phi0).
                Additionally the ef-frequency is measured at the upper sweet
                spot.

                default:
                    {0: ('ge', 'ef')),
                    -0.5: ('ge',),
                    -0.25: ('ge',)}

            flux_to_voltage_and_freq_guess: Guessed values for voltage and
                ge-frequencies for the fluxes in the measurement dictionary in
                case no Hamiltonian model is available yet. If passed, the
                dictionary must include guesses for the first two fluxes of
                `measurements` (which should be sweet spots). Default is None,
                in which case it is required to have use_prior_model==True. This
                means that the guesses will automatically be calculated with the
                existing model.

                Note, for succesful execution of the routine either a
                flux_to_voltage_and_freq_guess must be passed or a
                flux-voltage model must be specified in the qubit object.

                For example, `flux_to_voltage_and_freq_guess = {0:(-0.5,
                5.8e9), -0.5:(-4.1, 4.20e9)}` designates measurements at two
                sweet spots; one at flux = 0 (zeroth uss) and one at
                flux = -0.5 (left lss), both in units of phi0. The
                guessed voltages are -0.5V and -4.1V, respectively. The guessed
                frequencies are the last entry of the tuple: 5.8 GHz at flux = 0
                and 4.2 GHz at flux = -0.5 (the first left lss)
            save_instrument_settings: boolean to designate if instrument
                settings should be saved before and after this (sub)routine

        Keyword Arguments:
            delegate_plotting: whether to delegate plotting to the
                `plotting` module. The variable is stored in the global settings
                and the default value is False.
            anharmonicity: guess for the anharmonicity. Default -175
                MHz. Note the sign convention, the anharmonicity alpha is
                defined as alpha = f_ef - f_ge.
            method: optimization method to use. Default is Nelder-
                Mead.
            include_mixer_calib_carrier: If True, include mixer calibration
                for the carrier.
            mixer_calib_carrier_settings: Settings for the mixer calibration
                for the carrier.
            include_mixer_calib_skewness: If True, include mixer calibration
                for the skewness.
            mixer_calib_skewness_settings: Settings for the mixer calibration
                for the skewness.

        FIXME If the guess is very rough, it might be good to have an option to
        run a qubit spectroscopy before. This could be included in the future
        either directly here or, maybe even better, as an optional step in
        FindFrequency.

        FIXME There needs to be a general framework allowing the user to decide
        which optional steps should be included (e.g., mixer calib, qb spec,
        FindFreq before ReparkingRamsey, etc.).

        FIXME The routine relies on fp-assisted read-out which assumes
        that the qubit object has a model containing the dac_sweet_spot and
        V_per_phi0. While these values are not strictly needed, they are needed
        for the current implementation of the ro_flux_tmp_vals. This is
        detrimental for the routine if use_prior_model = False and the qubit
        doesn't contain a prior model.
        """

        if not use_prior_model:
            assert (
                flux_to_voltage_and_freq_guess is not None
            ), "If use_prior_model is False, flux_to_voltage_and_freq_guess must be specified"

        # flux to voltage and frequency dictionaries (guessed and measured)
        self.flux_to_voltage_and_freq = {}
        if not flux_to_voltage_and_freq_guess is None:
            flux_to_voltage_and_freq_guess = (
                flux_to_voltage_and_freq_guess.copy()
            )
        self.flux_to_voltage_and_freq_guess = flux_to_voltage_and_freq_guess

        # routine attributes
        self.fluxlines_dict = fluxlines_dict

        # Measurements
        if measurements is None:
            # default measurements
            self.measurements = {
                0: (
                    "ge",
                    "ef",
                ),  # ge- and ef-frequency at zeroth upper ss
                -0.5: ("ge",),  # ge-frequency at first left lower ss
                -0.25: ("ge",),  # ge-frequency at mid point
            }
        else:
            # user specified measurements
            self.measurements = {
                k: tuple(transition_to_str(t) for t in v)
                for k, v in measurements.items()
            }

        # Validity of measurement dictionary
        for flux, transitions in self.measurements.items():
            for t in transitions:
                if not t in ["ge", "ef"]:
                    raise ValueError(
                        "Measurements must only include 'ge' and "
                        "'ef' (transitions). Currently, other transitions are "
                        "not supported."
                    )

        # Guesses for voltage and frequency
        # determine sweet spot fluxes
        self.ss1_flux, self.ss2_flux, *rest = self.measurements.keys()
        assert self.ss1_flux % 0.5 == 0 and self.ss2_flux % 0.5 == 0, (
            "First entries of the measurement dictionary must be sweet spots"
            "(flux must be half integer)"
        )

        # If use_prior_model is True, generate guess from prior Hamiltonian
        # model
        if use_prior_model:
            assert len(qubit.fit_ge_freq_from_dc_offset()) > 0, (
                "To use the prior Hamiltonian model, a model must be present in"
                " the qubit object"
            )

            self.flux_to_voltage_and_freq_guess = {
                self.ss1_flux: (
                    qubit.calculate_voltage_from_flux(self.ss1_flux),
                    qubit.calculate_frequency(flux=self.ss1_flux),
                ),
                self.ss2_flux: (
                    qubit.calculate_voltage_from_flux(self.ss2_flux),
                    qubit.calculate_frequency(flux=self.ss2_flux),
                ),
            }

        # Validity of flux_to_voltage_and_freq_guess dictionary
        x = set(self.flux_to_voltage_and_freq_guess.keys())
        y = set(self.measurements.keys())
        z = set([self.ss1_flux, self.ss2_flux])

        assert x.issubset(y), (
            "Fluxes in flux_to_voltage_and_freq_guess must be a subset of the "
            "fluxes in measurements"
        )

        self.other_fluxes_with_guess = list(x - z)
        self.fluxes_without_guess = list(y - x)

        super().__init__(
            qubits=[qubit],
            save_instrument_settings=save_instrument_settings,
            use_prior_model=use_prior_model,
            **kw,
        )

    @property
    def parameters_init(self):
        # writing keyword arguments into correct categroty of parameters dict
        # and removing the from self.kw. Note that the parameters should already
        # be present in the parameters dict (so there should be no KeyError),
        # the parameters passed in the init simply take precedence.

        # This is the structure: FIXME - add a generalized way to extract and
        # store the parameters in the parameters dict from a structure dict like
        # the one below (possibly of varying depth).
        #
        # {
        #     "Transmon": {"anharmonicity"},
        #     "DetermineModel": {"method"},
        # }

        # Transmon
        for x in ["anharmonicity"]:
            self.parameters["Transmon"][x] = self.kw.pop(
                x, self.parameters["Transmon"][x]
            )

        # DetermineModel
        for x in ["method"]:
            self.parameters["DetermineModel"][x] = self.kw.pop(
                x, self.parameters["DetermineModel"][x]
            )

        if self.kw.get("use_prior_model", False):
            self.parameters["DetermineModel"]["preliminary"][
                "use_prior_model"
            ] = True
            self.parameters["DetermineModel"]["preliminary"][
                "include_resonator"
            ] = True

        return super().parameters_init

    def create_routine_template(self):
        """
        Creates the routine template for the HamiltonianFitting routine using
            the specified parameters.
        """
        super().create_routine_template()
        qubit = self.qubit

        # Measurements at fluxes with provided guess voltage/freq
        for flux in [
            self.ss1_flux,
            self.ss2_flux,
            *self.other_fluxes_with_guess,
        ]:
            # Getting the guess voltage and frequency
            voltage_guess, ge_freq_guess = self.flux_to_voltage_and_freq_guess[
                flux
            ]

            # Setting bias voltage to the guess voltage
            self.add_step(self.SetBiasVoltage, {"voltage": voltage_guess})

            for transition in self.measurements[flux]:
                # ge-transitions
                if transition == "ge":

                    # Updating ge-frequency at this voltage to guess value
                    self.add_step(
                        self.UpdateFrequency,
                        {
                            "frequency": ge_freq_guess,
                            "transition": transition,
                            "use_prior_model": self.kw["use_prior_model"],
                        },
                    )

                    # Finding the ge-transition frequency at this voltage
                    find_freq_settings = {
                        "transition": transition,
                    }
                    find_freq_settings.update(
                        self.parameters["FindFrequency"][transition]
                    )
                    self.add_step(
                        FindFrequency,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(
                            qubit, v_park=voltage_guess, use_ro_flux=True
                        ),
                    )

                    # If this flux is one of the two sweet spot fluxes, we also
                    # perform a Reparking Ramsey and update the flux-voltage
                    # relation stored in routine.
                    if flux in [self.ss1_flux, self.ss2_flux]:
                        # Reparking ramsey
                        task_list = [
                            {
                                "qb": qubit,
                                "fluxline": self.fluxlines_dict[qubit.name],
                            }
                        ]

                        self.add_step(
                            qbcal.ReparkingRamsey,
                            {"update": True, "task_list": task_list}
                            | self.parameters["ReparkingRamsey"],
                            step_tmp_vals=ro_flux_tmp_vals(
                                qubit, v_park=voltage_guess, use_ro_flux=True
                            ),
                        )

                        # Updating voltage to flux with Reparking ramsey results
                        self.add_step(
                            self.UpdateFluxToVoltage,
                            {
                                "index_reparking": len(self.routine_template)
                                - 1,
                                "flux": flux,
                            },
                        )

                elif transition == "ef":
                    # Updating ef-frequency at this voltage to guess value
                    self.add_step(
                        self.UpdateFrequency,
                        {
                            "flux": flux,
                            "transition": transition,
                            "use_prior_model": self.kw["use_prior_model"],
                        },
                    )

                    # Finding the ef-frequency
                    find_freq_settings = {"transition": transition}
                    find_freq_settings.update(
                        self.parameters["FindFrequency"][transition]
                    )
                    self.add_step(
                        FindFrequency,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(
                            qubit, v_park=voltage_guess, use_ro_flux=True
                        ),
                    )

        if not self.kw["use_prior_model"]:
            # Determining preliminary model - based on partial data and routine
            # parameters (e.g. fabrication parameters or user input)
            self.add_step(
                self.DetermineModel,
                self.parameters["DetermineModel"]["preliminary"]
                | {"method": self.parameters["DetermineModel"]["method"]},
            )

        # Measurements at other flux values (using preliminary or prior model)
        for flux in self.fluxes_without_guess:
            # Updating bias voltage using earlier reparking measurements
            self.add_step(self.SetBiasVoltage, {"flux": flux})

            # Looping over all transitions
            for transition in self.measurements[flux]:

                # Updating transition frequency of the qubit object to the value
                # calculated by prior or preliminary model
                update_freq_settings = {
                    "flux": flux,
                    "transition": transition,
                    "use_prior_model": True,
                }
                self.add_step(self.UpdateFrequency, update_freq_settings)

                # Set temporary values for Find Frequency
                set_tmp_vals_settings = {
                    "flux_park": flux,
                    "index": len(self.routine_template) + 1,
                }
                self.add_step(
                    SetTemporaryValuesFluxPulseReadOut,
                    set_tmp_vals_settings,
                )

                # Finding frequency
                find_freq_settings = {"transition": transition}
                find_freq_settings.update(
                    self.parameters["FindFrequency"][transition]
                )
                self.add_step(
                    FindFrequency,
                    find_freq_settings,
                    step_tmp_vals=ro_flux_tmp_vals(
                        qubit, v_park=voltage_guess, use_ro_flux=True
                    ),
                )

        # Determining final model based on all data
        self.add_step(
            self.DetermineModel,
            self.parameters["DetermineModel"]["final"]
            | {"method": self.parameters["DetermineModel"]["method"]},
        )

        # Interweave routine if the user wants to include mixer calibration
        self.add_mixer_calib_steps(**self.kw)

    def add_mixer_calib_steps(self, **kw):
        """
        Add steps to calibrate the mixer after the rest of the routine is
        defined. Mixer calibrations are put after every UpdateFrequency step.

        Keyword arguments:
            include_mixer_calib_carrier: If True, include mixer calibration
                for the carrier.
            mixer_calib_carrier_settings: Settings for the mixer calibration
                for the carrier.
            include_mixer_calib_skewness: If True, include mixer calibration
                for the skewness.
            mixer_calib_skewness_settings: Settings for the mixer calibration
                for the skewness.
        """

        # carrier settings
        include_mixer_calib_carrier = kw.get(
            "include_mixer_calib_carrier", False
        )
        mixer_calib_carrier_settings = kw.get(
            "mixer_calib_carrier_settings", {}
        )
        mixer_calib_carrier_settings.update(
            {"qubit": self.qubit, "update": True}
        )

        # skewness settings
        include_mixer_calib_skewness = kw.get(
            "include_mixer_calib_skewness", False
        )
        mixer_calib_skewness_settings = kw.get(
            "mixer_calib_skewness_settings", {}
        )
        mixer_calib_skewness_settings.update(
            {"qubit": self.qubit, "update": True}
        )

        if include_mixer_calib_carrier or include_mixer_calib_skewness:
            i = 0

            while i < len(self.routine_template):
                step_class = self.get_step_class_at_index(i)

                if step_class == self.UpdateFrequency:

                    # include mixer calibration skewness
                    if include_mixer_calib_skewness:

                        self.add_step(
                            MixerCalibrationSkewness,
                            mixer_calib_carrier_settings,
                            index=i + 1,
                        )
                        i += 1

                    # include mixer calibration carrier
                    if include_mixer_calib_carrier:

                        self.add_step(
                            MixerCalibrationCarrier,
                            mixer_calib_skewness_settings,
                            index=i + 1,
                        )
                        i += 1
                i += 1

    class UpdateFluxToVoltage(IntermediateStep):
        def __init__(self, routine, flux, index_reparking, **kw):
            """
            Intermediate step that updates the flux_to_voltage_and_freq
            dictionary using prior ReparkingRamsey measurements.

            Arguments:
                routine: the routine to which this step belongs
                flux: the flux to update the voltage and frequency for using the
                    ReparkingRamsey results

            FIXME it might be useful to also include results from normal ramsey
            experiments. For example, this could be used in UpdateFrequency if
            there is no model but a measurement of the ge-frequency was done.
            """
            # arguments that are not arguments of the super init will be considered
            # as key words for the super init.
            super().__init__(
                routine, flux=flux, index_reparking=index_reparking, **kw
            )

        def run(self):
            kw = self.kw
            qb = self.routine.qubit

            # flux passed on in settings
            flux = kw["flux"]
            index_reparking = kw["index_reparking"]

            # voltage found by reparking routine
            reparking_ramsey = self.routine.routine_steps[index_reparking]
            try:
                apd = reparking_ramsey.analysis.proc_data_dict[
                    "analysis_params_dict"
                ]
                voltage = apd["reparking_params"][qb.name]["new_ss_vals"][
                    "ss_volt"
                ]
                frequency = apd["reparking_params"][qb.name]["new_ss_vals"][
                    "ss_freq"
                ]
            except KeyError:
                log.error(
                    "Analysis reparking ramsey routine failed, flux to "
                    "voltage mapping can not be updated (guess values will be "
                    "used in the rest of the routine)"
                )

                (
                    voltage,
                    frequency,
                ) = self.routine.flux_to_voltage_and_freq_guess[flux]

            self.routine.flux_to_voltage_and_freq.update(
                {flux: (voltage, frequency)}
            )

    class UpdateFrequency(IntermediateStep):
        def __init__(self, routine, frequency=None, transition="ge", **kw):
            """
            Updates the frequency of the specified transition.

            Args:
                routine (Routine): the routine to which this step belongs
                frequency (float): frequency to which the qubit should be set
                transition (str): transition to be updated

            key word arguments:
                use_prior_model: if True, the frequency is updated using the
                    Hamiltonian model. If False, the frequency is updated using
                    the specified frequency.
                flux (float): flux to which the qubit frequency should
                    correspond. Useful if the frequency is not known a priori
                    while there a Hamiltonian model or flux-frequency
                    relationship is known.
                voltage (float): the dac voltage to which the qubit frequency
                    should correspond. Useful if the frequency is not known a
                    priori while there a Hamiltonian model or flux-frequency
                    relationship is known.

            FIXME the flux-frequency relationship stored in
            flux_to_voltage_and_freq should/could be used here.
            """
            # arguments that are not arguments of the super init will be
            # considered as key words for the super init.
            use_prior_model = kw.pop("use_prior_model", False)

            # transition and frequency
            self.transition = transition  # default is "ge"
            self.frequency = frequency  # default is None
            self.flux = kw.get("flux", None)
            self.voltage = kw.get("voltage", None)

            assert not (
                self.frequency is None
                and self.flux is None
                and self.voltage is None
            ), "No transition, frequency or voltage specified. At least one of "
            "these should be specified."

            super().__init__(
                routine,
                frequency=frequency,
                transition=transition,
                use_prior_model=use_prior_model,
                **kw,
            )

        def run(self):
            """
            Updates frequency of the qubit for a given transition. This can
            either be done by passing the frequency directly, or in case a
            model exists by passing the voltage or flux.
            """
            qb = self.routine.qubit
            frequency = self.frequency

            # If no explicit frequency is given, try to find it for given flux
            # or voltage using the Hamiltonian model stored in qubit
            if frequency is None:
                if self.transition == "ge" or "ef":
                    # A (possibly preliminary) Hamiltonian model exists
                    if (
                        self.kw["use_prior_model"]
                        and len(qb.fit_ge_freq_from_dc_offset()) > 0
                    ):
                        frequency = qb.calculate_frequency(
                            flux=self.flux,
                            bias=self.voltage,
                            transition=self.transition,
                        )

                    # No Hamiltonian model exists, but we want to know the ef-
                    # frequency when the ge-frequency is known at this flux and
                    # there is a guess for the anharmonicity

                    # FIXME instead of qb.ge_freq() we should use the frequency
                    # stored in the flux_to_voltage_and_freq dictionary. The
                    # current implementation assumes that the ef measurement is
                    # preceded by the ge-measurement at the same voltage (and
                    # thus flux).
                    elif self.transition == "ef":
                        frequency = (
                            qb.ge_freq()
                            + self.routine.parameters["Transmon"][
                                "anharmonicity"
                            ]
                        )

                    # No Hamiltonian model exists
                    else:
                        raise NotImplementedError(
                            "Can not estimate frequency with incomplete model, "
                            "make sure to save a (possibly preliminary) model "
                            "first"
                        )

            if self.routine.verbose:
                print(
                    f"{self.transition}-frequency updated to ", frequency, "Hz"
                )

            qb[f"{self.transition}_freq"](frequency)

    class SetBiasVoltage(IntermediateStep):
        def __init__(self, routine, voltage=None, flux=None, **kw):
            """
            Intermediate step that updates the bias voltage of the qubit. This
            can be done by simply specifying the voltage, or by specifying the
            flux. If the flux is given, the corresponding bias is calculated
            using the Hamiltonian model stored in the qubit object.
            """
            # arguments that are not arguments of the super init will be
            # considered as key words for the super init.
            super().__init__(routine, voltage=voltage, flux=flux, **kw)

        def run(self):
            kw = self.kw
            qb = self.routine.qubit

            if kw["flux"] is not None:
                flux = kw["flux"]
                voltage = qb.calculate_voltage_from_flux(flux)
            elif kw["voltage"] is not None:
                voltage = kw["voltage"]
            else:
                raise ValueError("No voltage or flux specified")
            self.routine.fluxlines_dict[qb.name](voltage)

    class DetermineModel(IntermediateStep):
        def __init__(
            self,
            routine,
            include_resonator=True,
            use_prior_model=True,
            method="Nelder-Mead",
            **kw,
        ):
            """
            Intermediate step that determines the model of the qubit based on
            the measured data. Can be used for both estimating the model and
            determining the final model.

            Args:
                include_resonator (bool): if True, the model includes the effect
                    of the resonator. If False, the coupling is set to zero and
                    essentially only the transmon parameters are optimized (the
                    resonator has no effect on the transmon).
                use_prior_model (bool): if True, the prior model is used to
                    determine the model, if False, the measured data is used.
                method (str): the optimization method to be used
                    when determining the model. Default is Nelder-Mead.

            Keyword Arguments:
                include_reparkings (bool): if True, the data from the
                    ReparkingRamsey measurements are used to help determine a
                    Hamiltonian model.
            """

            # arguments that are not arguments of the super init will be
            # considered as key words for the super init.
            super().__init__(
                routine,
                include_resonator=include_resonator,
                use_prior_model=use_prior_model,
                method=method,
                **kw,
            )

        def run(self):
            kw = self.kw

            # using all experimental values
            self.experimental_values = (
                HamiltonianFitting.get_experimental_values(
                    qubit=self.routine.qubit,
                    fluxlines_dict=self.routine.fluxlines_dict,
                    timestamp_start=self.routine.preroutine_timestamp,
                    include_reparkings=kw["include_reparkings"],
                )
            )

            log.info(f"Experimental values: {self.experimental_values}")

            # preparing guess parameters and choosing parameters to optimize
            p_guess, parameters_to_optimize = self.make_model_guess(
                use_prior_model=kw["use_prior_model"],
                include_resonator=kw["include_resonator"],
            )

            log.info(f"Parameters guess: {p_guess}")
            log.info(f"Parameters to optimize: {parameters_to_optimize}")

            # determining the model
            f = self.routine.optimizer(
                experimental_values=self.experimental_values,
                parameters_to_optimize=parameters_to_optimize,
                parameters_guess=p_guess,
                method=kw["method"],
            )

            # extracting results
            result_dict = self.routine.fit_parameters_from_optimization_results(
                f, parameters_to_optimize, p_guess
            )

            log.info(f"Result from fit: {result_dict}")

            # saving model to qubit from routine
            self.routine.qubit.fit_ge_freq_from_dc_offset(result_dict)

        def make_model_guess(
            self, use_prior_model=True, include_resonator=True
        ):
            """
            Constructing parameters for the for the Hamiltonian model
            optimization. This includes defining values for the fixed parameters
            as well as initial guesses for the parameters to be optimized.

            Args:
                use_prior_model (bool): if True, the prior model parameters are
                    used as initial guess. Note that the voltage parameters
                    (dac_sweet_spot and V_per_phi0) are fixed according to the
                    sweet spot voltage measurements. If False, the guessed
                    parameters are determined through routine parameters or key
                    word inputs.
                include_resonator (bool): if True, the model includes the effect
                    of the resonator. If False, the coupling is set to zero and
                    essentially only the transmon parameters are optimized (the
                    resonator has no effect on the transmon).
            """
            # using prior model to determine the model or not
            if use_prior_model:
                p_guess = self.routine.qubit.fit_ge_freq_from_dc_offset()

            # using guess parameters instead
            else:
                p_guess = {
                    "Ej_max": self.routine.parameters["Transmon"]["Ej_max"],
                    "E_c": self.routine.parameters["Transmon"]["E_c"],
                    "asymmetry": self.routine.parameters["Transmon"][
                        "asymmetry"
                    ],
                    "coupling": self.routine.parameters["Transmon"]["coupling"]
                    * include_resonator,
                    "fr": self.routine.parameters["Transmon"]["fr"],
                }

            # using sweet spot measurements determined by the reparking routine
            flux_to_voltage_and_freq = self.routine.flux_to_voltage_and_freq
            ss1_flux, ss2_flux = self.routine.ss1_flux, self.routine.ss2_flux

            ss1_voltage, ss1_frequency = flux_to_voltage_and_freq[ss1_flux]
            ss2_voltage, ss2_frequency = flux_to_voltage_and_freq[ss2_flux]

            # calculating voltage parameters based on ss-measurements and fixing
            # the corresponding parameters to these values
            V_per_phi0 = (ss1_voltage - ss2_voltage) / (ss1_flux - ss2_flux)
            dac_sweet_spot = ss1_voltage - V_per_phi0 * ss1_flux
            p_guess.update(
                {"dac_sweet_spot": dac_sweet_spot, "V_per_phi0": V_per_phi0}
            )

            # including coupling (resonator) into model optimization
            if include_resonator:
                parameters_to_optimize = [
                    "Ej_max",
                    "E_c",
                    "asymmetry",
                    "coupling",
                ]
            else:
                parameters_to_optimize = ["Ej_max", "E_c", "asymmetry"]

            log.debug(f"Parameters guess {p_guess}")
            log.debug(f"Parameters to optimize {parameters_to_optimize}")

            return p_guess, parameters_to_optimize

    @staticmethod
    def verification_measurement(
        qubit, fluxlines_dict, fluxes=None, voltages=None, verbose=True, **kw
    ):
        """
        Performs a verification measurement of the model for given fluxes
        (or voltages). The read-out is done using flux assisted read-out and
        the read-out should be configured beforehand.

        Note, this measurement requires the qubit(s) to contain a
        fit_ge_freq_from_dc_offset model.

        FIXME: the current implementation forces the user to use flux-pulse
        assisted read-out. In the future, the routine should also work on
        set-ups that only have DC sources, but no flux AWG. Possible solutions:
        - have the user specify the RO frequencies that are to be used at the
            various flux biases
        - use the Hamiltonian model to determine the RO frequencies at the
            various flux biases. Note, the usual Hamiltonian model does not
            take the Purcell filter into account which might cause problems.

        Args:
            qubit: qubit to perform the verification measurement on
            fluxlines_dict:
            fluxes: fluxes to perform the verification measurement at.
                If None, fluxes is set to default np.linspace(0, -0.5, 11)
            voltages: voltages to perform the verification measurement at.
                If None, voltages are set to correspond to the fluxes array.
                Note that if both fluxes and voltages are specified, only
                the values for `voltages` is used.
            verbose: if True, prints updates on the progress of the
                measurement.

        Keyword Arguments:
            reset_fluxline: bool for resetting to fluxline to initial value
                after the measurement.
            Default is True.
            plot: bool for plotting the results at the end, default False


        Returns:
            dictionary of verification measurements.

        FIXME: In an automated routine that is supposed to run without user
        interaction, plots should rather be stored in a timestamp folder rather
        than being shown on the screen. get_experimental_values_from_timestamps
        (or get_experimental_values?) could keep track of which timestamps
        belong to the current analysis, and you could then use get_folder
        (from analysis_toolbox) to get the folder of the latest among these
        timestamps. However, this might require refactoring the methods to not
        be static methods.
        """

        result_dict = qubit.fit_ge_freq_from_dc_offset()
        assert len(result_dict) > 0, (
            "This measurement requires the "
            "qubit(s) to contain a fit_ge_freq_from_dc_offset model"
        )

        fluxline = fluxlines_dict[qubit.name]
        initial_voltage = fluxline()

        if fluxes is None and voltages is None:
            fluxes = np.linspace(0, -0.5, 11)
        if voltages is None:
            voltages = qubit.calculate_voltage_from_flux(fluxes)

        experimental_values = {}  # empty dictionary to store results

        for index, voltage in enumerate(voltages):
            with temporary_value(
                *ro_flux_tmp_vals(qubit, v_park=voltage, use_ro_flux=True)
            ):

                if verbose:
                    print(
                        f"Verification measurement  step {index} / "
                        f"{len(voltages)} of {qubit.name}"
                    )

                # setting the frequency and fluxline
                qubit.calculate_frequency(voltage, update=True)
                fluxline(voltage)

                # finding frequency
                ff = FindFrequency(
                    [qubit], calibration_settings=None, update=True
                )

                # storing experimental result
                experimental_values[voltage] = {"ge": qubit.ge_freq()}

        if kw.pop("reset_fluxline", True):
            # resetting fluxline to initial value
            fluxline(initial_voltage)

        # plotting
        if kw.pop("plot", False):
            HamiltonianFitting.plot_model_and_experimental_values(
                result_dict=result_dict, experimental_values=experimental_values
            )
            HamiltonianFitting.calculate_residuals(
                result_dict=result_dict,
                experimental_values=experimental_values,
                plot_residuals=True,
            )

        return experimental_values

    _DEFAULT_PARAMETERS = {
        "Routine": {
            "delegate_plotting": False,
        },
        "Transmon": {
            "anharmonicity": -175e6,
            "Ej_max": 20e9,
            "E_c": 175e6,
            "asymmetry": 0.5,
            "coupling": 250e6,
            "fr": 7e9,
        },
        "ReparkingRamsey": {
            "delays": np.linspace(100e-9, 250e-9, 21),
            "dc_voltage_offsets": np.linspace(-0.03, 0.03, 5),
            "artificial_detuning": 40e6,
        },
        "FindFrequency": {
            "ge": {
                "allowed_delta_f": 0.05e6,
                "max_iterations": 2,
            },
            "ef": {
                "allowed_delta_f": 0.05e6,
                "max_iterations": 2,
            },
        },
        "DetermineModel": {
            "method": "Nelder-Mead",
            "preliminary": {
                "include_reparkings": True,
                "include_resonator": False,
                "use_prior_model": False,
            },
            "final": {
                "include_reparkings": False,
                "include_resonator": True,
                "use_prior_model": True,
            },
        },
    }


class MixerCalibrationSkewness(IntermediateStep):
    def __init__(self, routine, **kw):
        """
        Mixer calibration step that calibrates the skewness of the mixer.

        Args:
            routine: Routine object

        Keyword Arguments:
            calibrate_drive_mixer_skewness_function: method for calibrating to
                be used. Default is to use calibrate_drive_mixer_skewness_model.
        """
        super().__init__(routine, **kw)

    def run(self):
        kw = self.kw

        calibrate_drive_mixer_skewness_function = kw.get(
            "calibrate_drive_mixer_skewness_function",
            "calibrate_drive_mixer_skewness_model",
        )

        function = getattr(
            self.routine.qubit, calibrate_drive_mixer_skewness_function
        )
        new_kw = keyword_subset_for_function(kw, function)

        function(**new_kw)


class MixerCalibrationCarrier(IntermediateStep):
    def __init__(self, routine, **kw):
        """
        Mixer calibration step that calibrates the carrier of the mixer.

        Args:
            routine: Routine object

        Keyword Arguments:
            calibrate_drive_mixer_carrier_function: method for calibrating to
                be used. Default is to use calibrate_drive_mixer_carrier_model.
        """
        super().__init__(routine, **kw)

    def run(self):
        kw = self.kw

        calibrate_drive_mixer_carrier_function = kw.get(
            "calibrate_drive_mixer_carrier_function",
            "calibrate_drive_mixer_carrier_model",
        )

        function = getattr(
            self.routine.qubit, calibrate_drive_mixer_carrier_function
        )
        new_kw = keyword_subset_for_function(kw, function)

        function(**new_kw)


class SetTemporaryValuesFluxPulseReadOut(IntermediateStep):
    def __init__(
        self,
        routine,
        index,
        voltage_park=None,
        flux_park=None,
        **kw,
    ):
        """
        Intermediate step that sets ro-temporary values for a step of the
        routine.

        Args:
            routine: Routine object
            index: index of the step in the routine that requires temporary
                values for flux-pulse assisted read-out
            v_park: value of the ro-temporary value to set

        """
        super().__init__(
            routine=routine,
            index=index,
            voltage_park=voltage_park,
            flux_park=flux_park,
            **kw,
        )

    def run(self):
        kw = self.kw
        qb = self.routine.qubit
        index = self.kw["index"]

        if kw["flux_park"] is not None:
            flux = kw["flux_park"]
            v_park = qb.calculate_voltage_from_flux(flux)
        elif kw["voltage_park"] is not None:
            v_park = kw["voltage_park"]
        else:
            raise ValueError("No voltage or flux specified")

        # temporary values for ro
        ro_tmp_vals = ro_flux_tmp_vals(qb, v_park, use_ro_flux=True)

        # extending temporary values
        self.routine.extend_step_tmp_vals_at_index(
            tmp_vals=ro_tmp_vals, index=index
        )


def keyword_subset(keyword_arguments, allowed_keywords):
    """
    Returns a dictionary with only the keywords that are used by the
    function.
    """
    keywords = set(keyword_arguments.keys())
    keyswords_to_extract = keywords.intersection(allowed_keywords)

    new_kw = {key: keyword_arguments[key] for key in keyswords_to_extract}

    return new_kw


def keyword_subset_for_function(keyword_arguments, function):
    """
    Returns a dictionary with only the keywords that are used by the
    function.
    """
    allowed_keywords = inspect.getfullargspec(function)[0]

    return keyword_subset(keyword_arguments, allowed_keywords)


def update_nested_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d
