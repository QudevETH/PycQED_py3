from pycqed.measurement.calibration.automatic_calibration_routines.base import\
    update_nested_dictionary

from .base_step import Step, IntermediateStep
from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.measurement.quantum_experiment import QuantumExperiment
from pycqed.utilities.general import temporary_value
from pycqed.utilities.reload_settings import reload_settings

import pycqed.analysis.analysis_toolbox as a_tools
from typing import List, Any, Tuple, Dict, Type, Optional
from warnings import warn
import numpy as np
import copy
import logging
import pprint
import inspect

log = logging.getLogger('Routines')
log.setLevel('INFO')

try:
    from pycqed.utilities import devicedb
except ModuleNotFoundError:
    log.info("The module 'device-db-client' was not successfully imported. "
             "The device database features will not be available.")
    _device_db_client_module_missing = True
else:
    _device_db_client_module_missing = False


class RoutineTemplate(list):
    """Class to describe templates for (calibration) routines.

    The class is essentially a list of lists that contain a class, a label, and
    the corresponding settings of a step in a routine. Steps may be
    measurements, calibration routines, or intermediate steps.
    """

    def __init__(
            self,
            steps,
            global_settings=None,
            routine=None,
    ):
        """Initialize the routine template.

        Args:
            steps (list of :obj:`Step`): List of steps that define the routine.
                Each step consists of a list of three elements. Namely, the step
                class, the step label (a string), and the step settings (a
                dictionary). It can optionally also have a fourth element to
                give the experiment temporary values.
                For example:
                steps = [
                    [StepClass1, step_label_1, step_settings_1],
                    [StepClass2, step_label_2, step_settings_2,
                    step_tmp_vals_2],
                ]
            global_settings (dict, optional): Dictionary containing global
                settings for each step of the routine (e.g., "dev", "update",
                "delegate_plotting"). Defaults to None.
            routine (:obj:`AutomaticCalibrationRoutine`, optional): Routine that
                the RoutineTemplate defines. Defaults to None.
        """
        super().__init__(steps)

        if routine is not None:
            self.routine = routine

        if global_settings is not None:
            self.global_settings = global_settings
        else:
            self.global_settings = {}

    def get_step_class_at_index(self, index):
        """Returns the step class for a specific step in the routine template.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            class: The class of the step at position 'index' in the routine
                template.
        """
        return self[index][0]

    def get_step_label_at_index(self, index):
        """Returns the step label for a specific step in the routine template.

        Args:
            index (int): Index of the step for which the step label is to be
                returned.
        Returns:
            str: The label of the step at position 'index' in the routine
                template.
        """
        return self[index][1]

    def get_step_settings_at_index(self, index):
        """Returns the settings for a specific step in the routine template.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            dict: The settings dictionary of the step at position 'index' in
                the routine template.
        """
        settings = {}
        settings.update(copy.copy(self.global_settings))
        settings.update(copy.copy(self[index][2]))
        return settings

    def get_step_tmp_vals_at_index(self, index):
        """Returns the temporary values of the step at index.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            list: The temporary values for the step at position 'index' in the
                routine template. Each entry is a tuple made of a
                QCoDeS parameter and its temporary value.
        """
        try:
            return self[index][3]
        except IndexError:
            return []

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """Extends the temporary values of the step at index. If the step does
        not have any temporary values, it sets the temporary values to the
        passed temporary values.

        Args:
            tmp_vals (list): The temporary values for the step at position
                'index' in the routine template. Each entry should be a tuple
                made of a QCoDeS parameter and its temporary value.
            index (int): Index of the step for which the temporary values should
                be used.
        """
        try:
            self[index][3].extend(tmp_vals)
        except IndexError:
            self[index].append(tmp_vals)

    def update_settings_at_index(self, settings, index):
        """Updates the settings of the step at position 'index'.

        Args:
            settings (dict): The new settings that will update the existing
                settings of the step at position 'index' in the routine
                template.
            index (int): Index of the step for which the temporary values should
                be used.
        """
        self[index][2].update(settings)

    def update_all_step_settings(self, settings):
        """Updates all settings of all steps in the routine.

        Args:
            settings (dict): The new settings that will update the existing
                settings of all the steps.
        """
        for i, x in enumerate(self):
            self.update_settings_at_index(settings, index=i)

    def update_settings(self, settings_list):
        """Updates all settings of the routine. Settings_list must be a list of
        dictionaries of the same length as the routine.

        Args:
            settings_list (list): List of dictionaries, where each entry will
                update the existing settings of the corresponding step of the
                routine. Must be of the same length as the routine.
        """
        for i, x in enumerate(settings_list):
            self.update_settings_at_index(settings=x, index=i)

    def view(
            self,
            **kws
    ):
        """DEPRECATED."""
        warn('This method is deprecated, use `Routine.view()` instead',
             DeprecationWarning, stacklevel=2)

    def __str__(self):
        """Returns a string representation of the routine template.

        FIXME: this representation does not include the tmp_vals of the steps.
        """

        s = ""

        for i, x in enumerate(self):
            s += f"Step {i}, {x[0].__name__}, {x[1]}\n"
        return s

    def step_name(self, index):
        """Returns the name of the step at position 'index'.

        Args:
            index (int): Index of the step whose name will be returned.

        Returns:
            str: The label of the step or the name of its class.
        """
        step_label = self.get_step_label_at_index(index)
        if step_label is not None:
            return step_label
        return self.get_step_class_at_index(index).get_lookup_class().__name__

    def add_step(self,
                 step_class,
                 step_label,
                 step_settings,
                 step_tmp_vals=None,
                 index=None):
        """Adds a step to the routine template.

        Args:
            step_class (class): Class of the step
            step_label (str): Label of the step
            step_settings (dict): Settings of the step.
            step_tmp_vals (list, optional): Temporary values for the step. Each
                entry should be a tuple made of a QCoDeS parameter and its
                temporary value. Defaults to None.
            index (int, optional): Index of the routine template at which the
                step should be added. If None, the step will be added at the end
                of the routine. Defaults to None.
        """
        if step_tmp_vals is None:
            step_tmp_vals = []

        if index is None:
            super().append(
                [step_class, step_label, step_settings, step_tmp_vals])
        else:
            super().insert(
                index, [step_class, step_label, step_settings, step_tmp_vals])

    @staticmethod
    def check_step(step):
        """Check that a step is properly built.

        Args:
            step (list): Routine template step.
        """
        assert isinstance(step, list), "Step must be a list"
        assert (len(step) == 3 or len(step) == 4), \
            "Step must be a list of length 3 or 4 (to include temporary values)"
        assert isinstance(step[0], type), (
            "The first element of the step "
            "must be a class (e.g. measurement or a calibration routine)")
        assert isinstance(step[2],
                          dict), ("The second element of the step "
                                  "must be a dictionary containing settings")

    def __getitem__(self, i):
        """Overloading of List.__getitem__ to ensure type RoutineTemplate is
        preserved.

        Args:
            i: index or slice

        Returns:
            Element or new RoutineTemplate instance
        """
        new_data = super().__getitem__(i)
        if isinstance(i, slice):
            new_data = self.__class__(new_data)
            new_data.global_settings = copy.copy(self.global_settings)
        return new_data


class AutomaticCalibrationRoutine(Step):
    """Base class for general automated calibration routines

    NOTE: In the children classes, it is necessary to call final_init at
    the end of their constructor. It is not possible to do this in the
    parent class because some routines need to do some further initialization
    after the base class __init__ and before the routine is actually created
    (which happens in final_init).

    In the children classes, the initialization follows this hierarchy:
        ChildRoutine.__init__
            AutomaticCalibrationRoutine.__init__
            final_init
                create_initial_routine
                    create_routine_template

    Afterwards, the routine is ready to run.

    If a routine contains some subroutines, the subroutines will be initialized
    at runtime when the parent routine is running the subroutine step.
    """

    def __init__(
            self,
            dev,
            routine=None,
            autorun=True,
            **kw,
    ):
        """Initializes the routine.

        Args:
            dev (Device): Device to be used for the routine
            autorun (bool): If True, the routine will be run immediately after
                initialization.
            routine (Step): The parent routine of the routine.

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter
        dictionary):
            update (bool): If True, the routine will overwrite qubit attributes
                with the values found in the routine. Note that if the routine
                is used as subroutine, this should be set to True.
            save_instrument_settings (bool): If True, the routine will save the
                instrument settings before and after the routine.
            verbose: If True, the routine will print out the progress of the
                routine. Default is True.
        """

        self.kw = kw
        self.autorun = autorun
        # Call Step constructor
        super().__init__(dev, routine, **kw)

        self.parameter_sublookups = ['General']
        self.leaf = False
        self.step_label = self.step_label or self.name

        self.DCSources = self.kw.pop("DCSources", None)

        self.routine_steps: List[Step] = []
        self.current_step_index = 0

        self.routine_template: Optional[RoutineTemplate] = None
        self.current_step: Optional[Step] = None
        self.current_step_settings: Optional[Dict] = None
        self.current_step_tmp_vals: Optional[List[Tuple[Any, Any]]] = None

        # MC - trying to get it from either the device or the qubits
        for source in [self.dev] + self.qubits:
            try:
                self.MC = source.instr_mc.get_instr()
                break
            except KeyError:  # instr_mc not a valid instrument (e.g., None)
                pass

        self.create_initial_parameters()

    def merge_settings(self, lookups, sublookups):
        """Merges all scopes relevant for a particular child step. The settings
        are retrieved and merged recursively to ensure that the priority
        specified in lookups and sublookups is respected.

        Example of how the settings are merged in chronological order
        with the following routine:
            Routine [None, "Routine"]
                SubRoutine ["subroutine_label", "SubRoutine"]
                    ExperimentStep ["experiment_label", "Experiment"]

        a) Initialization of Routine's steps. This will extract the settings
        of SubRoutine from the configuration parameter dictionary.

        Call: Routine.merge_settings(lookups=[None,"Routine"],
                               sublookups=["subroutine_label", "SubRoutine"])

            Look for the relevant settings in this order:
            1) Routine.settings["SubRoutine"]
            2) Routine.settings["subroutine_label"]
            3) Routine.settings["Routine"]["SubRoutine"]
            4) Routine.settings["Routine"]["subroutine_label"]

        At the end, SubRoutine.settings will be updated according to the
        hierarchy specified in the lookups.

        b) Initialization of SubRoutine's steps (occurs at runtime). This will
        extract the settings of ExperimentStep from the configuration parameter
        dictionary.

        Call: SubRoutine.merge_settings(lookups=["subroutine_label","SubRoutine"],
                                sublookups=["experiment_label","Experiment"])

            Call: Routine.merge_settings(lookups=["subroutine_label","SubRoutine"],
                                sublookups=["experiment_label","Experiment"])

                Look for the relevant settings in this order:
                1) Routine.settings["Experiment"]
                2) Routine.settings["experiment_label"]
                3) Routine.settings["SubRoutine"]["Experiment"]
                4) Routine.settings["SubRoutine"]["experiment_label"]
                5) Routine.settings["subroutine_label"]["Experiment"]
                6) Routine.settings["subroutine_label"]["experiment_label"]

            7) SubRoutine.settings["Experiment"]
            8) SubRoutine.settings["experiment_label"]
            9) SubRoutine.settings["SubRoutine"]["Experiment"]
            10) SubRoutine.settings["SubRoutine"]["experiment_label"]
            11) SubRoutine.settings["subroutine_label"]["Experiment"]
            12) SubRoutine.settings["subroutine_label"]["experiment_label"]

        The dictionary of settings that were merged according to the
        hierarchy specified in the lookups can be used to update 
        :obj:`Step.settings`.

        Arguments:
            lookups (list): A list of all scopes for the parent routine
                of the step whose settings need to be merged. The elements
                of the list will be interpreted in descending order of priority.
            sublookups (list): A list of scopes for the step whose settings need
                to be merged. The elements of the list will be interpreted in
                descending order of priority.

        Returns:
            dict: The dictionary containing the merged settings.
        """
        if self.routine is not None:
            # If the current step has a parent routine, call its merge_settings
            # recursively
            settings = self.routine.merge_settings(lookups, sublookups)
        else:
            # If the root routine is calling the function, then initialize
            # an empty dictionary for the settings of the child step
            settings = {}

        for sublookup in reversed(sublookups):
            # Looks for the sublookups directly in the settings. If self is the
            # root routine, this corresponds to looking in the first layer of
            # the configuration parameter dictionary, where the most general
            # settings are stored.
            # E.g., ['experiment_label', 'Experiment'] will first be looked
            # up in the most general settings.
            if sublookup in self.settings:
                update_nested_dictionary(settings, self.settings[sublookup])

        # Look for the entries settings[lookup][sublookup] (if both the lookup 
        # and the sublookup entries exist) or settings[lookup] (if only the 
        # lookup entry exist, but not the sublookup one)
        for lookup in reversed(lookups):
            if lookup in self.settings:
                if sublookups is not None:
                    for sublookup in reversed(sublookups):
                        if sublookup in self.settings[lookup]:
                            update_nested_dictionary(
                                settings, self.settings[lookup][sublookup])
                else:
                    update_nested_dictionary(settings,
                                                self.settings[lookup])

        return settings

    def extract_step_settings(self,
                              step_class: Type[Step],
                              step_label: str,
                              step_settings=None,
                              lookups=None,
                              sublookups=None):
        """Extract the settings of a step from the configuration parameter
        dictionary that was loaded and built from the JSON config files. The
        entry 'settings' of step_settings is also included in the returned
        settings.

        Args:
            step_class (Step): The class of the step whose settings need to be
                extracted.
            step_label (str): The label of the step whose settings need to be
                extracted.
            step_settings (dict, optional): Additional settings of the step
                whose settings need to be extracted. The entry
                step_settings['settings'] will be included in the returned
                settings. The settings contained in step_settings['settings']
                will have priority over those found in the configuration
                parameter dictionary. If None, an empty dictionary is used.
                Defaults to None.
            lookups (list, optional): A list of all scopes for the parent
                routine of the step whose settings need to be merged. The
                elements of the list will be interpreted in descending order of
                priority. If None, [routine_label, RoutineClass] will be used.
                Defaults to None.
            sublookups (list, optional): A list of scopes for the step whose
                settings need to be merged. The elements of the list will be
                interpreted in descending order of priority. If None,
                [step_label, StepClass] will be used. Defaults to None.
                Defaults to None.

        Returns:
            dict: A dictionary containing the settings extracted from the
                configuration parameter dictionary.
        """
        if step_settings is None:
            step_settings = {}
        # No 'General' lookup since at this point we are only interested
        # in retrieving the settings of each step of a routine, not the settings
        # of the routine itself
        if lookups is None:
            lookups = [self.step_label, self.get_lookup_class().__name__]
        if sublookups is None:
            sublookups = [step_label, step_class.get_lookup_class().__name__]

        autocalib_settings = self.settings.copy({
            step_class.get_lookup_class().__name__:
                self.merge_settings(lookups, sublookups)
        })
        update_nested_dictionary(autocalib_settings,
                                 step_settings.get('settings', {}))
        return autocalib_settings

    def create_routine_template(self):
        """Creates routine template. Can be overwritten or extended by children
        for more complex routines that require adaptive creation. The settings
        for each step are extracted from the configuration parameters
        dictionary.
        """
        # Create RoutineTemplate based on _DEFAULT_ROUTINE_TEMPLATE
        self.routine_template = copy.deepcopy(self._DEFAULT_ROUTINE_TEMPLATE)

        for step in self.routine_template:
            # Retrieve the step settings from the configuration parameter
            # dictionary. The settings will be merged according to the correct
            # hierarchy (more specific settings will overwrite less specific
            # settings)
            step_settings = self.extract_step_settings(step[0], step[1],
                                                       step[2])
            step[2]['settings'] = step_settings

        # standard global settings
        delegate_plotting = self.get_param_value('delegate_plotting')

        self.routine_template.global_settings.update({
            "dev": self.dev,
            "update": True,  # all subroutines should update relevant params
            "delegate_plotting": delegate_plotting,
        })

        # add user specified global settings
        update_nested_dictionary(
            self.routine_template.global_settings,
            self.kw.get("global_settings", {}),
        )

    def split_step_for_parallel_groups(self, index):
        """Replace the step at the given index with multiple steps according
        to the parallel groups defined in the configuration parameter
        dictionary.
        The multiple steps will be added starting from the given index and after
        it (the first one at the given index, the second one at index + 1 and so
        on).
        If no parallel groups are found, the step is left unchanged.

        Args:
            index (int): Index of the step to be replaced with the rearranged
                steps.
        """
        # Get the details of the step to be replaced
        step = self.routine_template[index]
        step_class = step[0]
        step_label = step[1]
        step_settings = step[2]
        try:
            step_tmp_settings = step[3]
        except IndexError:
            step_tmp_settings = []

        # Look for the keyword 'parallel_groups' in the settings
        lookups = [
            step_label,
            step_class.get_lookup_class().__name__, 'General'
        ]
        parallel_groups = self.get_param_value('parallel_groups',
                                               sublookups=lookups,
                                               leaf=True)
        if parallel_groups is not None:
            new_step_index = index
            # Remove existing step
            self.routine_template.pop(index)
            for parallel_group in parallel_groups:
                # Find the qubits belonging to parallel_group
                qubits_filtered = [
                    qb for qb in self.qubits if
                    (qb.name is parallel_group or
                     parallel_group in self.get_qubit_groups(qb.name))
                ]
                # Create a new step for qubits_filtered only and add it to the
                # routine template
                if len(qubits_filtered) != 0:
                    new_settings = copy.deepcopy(step_settings)
                    new_settings['qubits'] = qubits_filtered
                    self.add_step(step_class,
                                  step_label,
                                  new_settings,
                                  step_tmp_settings,
                                  index=new_step_index)
                    new_step_index += 1

    def prepare_step(self, i=None):
        """Prepares the next step in the routine. That is, it initializes the
        measurement object. The steps of the routine are instantiated here.

        Args:
            i (int): Index of the step to be prepared. If None, the default is
                set to the current_step_index.
        """

        if i is None:
            i = self.current_step_index

        # Setting step class and settings
        step_class = self.get_step_class_at_index(i)
        step_settings = self.get_step_settings_at_index(i)
        step_label = self.get_step_label_at_index(i)

        # Setting the temporary values
        self.current_step_tmp_vals = self.get_step_tmp_vals_at_index(i)

        # Update print
        if self.get_param_value('verbose'):
            print(f"{self.name}, step {i} "
                  f"({self.routine_template.step_name(index=i)}), preparing...")
        qubits = step_settings.pop('qubits', self.qubits)
        dev = step_settings.pop('dev', self.dev)
        autocalib_settings = self.settings.copy(
            overwrite_dict=step_settings.pop('settings', {}))
        # Executing the step with corresponding settings
        if issubclass(step_class, qbcal.SingleQubitGateCalibExperiment) or \
                issubclass(step_class, QuantumExperiment):
            step = step_class(qubits=qubits,
                              routine=self,
                              dev=dev,
                              step_label=step_label,
                              settings=autocalib_settings,
                              **step_settings)
        elif issubclass(step_class, IntermediateStep):
            step = step_class(routine=self,
                              dev=dev,
                              step_label=step_label,
                              qubits=qubits,
                              autorun=False,
                              settings=autocalib_settings,
                              **step_settings)
        elif issubclass(step_class, AutomaticCalibrationRoutine):
            step = step_class(routine=self,
                              dev=dev,
                              step_label=step_label,
                              qubits=qubits,
                              autorun=False,
                              settings=autocalib_settings,
                              **step_settings)
        else:
            raise ValueError(f"automatic subroutine is not compatible (yet)"
                             f"with the current step class {step_class}")
        self.current_step = step
        self.current_step_settings = step_settings

    def execute_step(self):
        """
        Executes the current step (routine.current_step) in the routine and
        writes the result in the routine_steps list.
        """
        if self.get_param_value('verbose'):
            j = self.current_step_index
            print(f"{self.name}, step {j} "
                  f"({self.routine_template.step_name(index=j)}), executing...")

        self.current_step.run()

        self.routine_steps.append(self.current_step)
        self.current_step_index += 1

    def run(self, start_index=None, stop_index=None):
        """Runs the complete automatic calibration routine. In case the routine
        was already completed, the routine is reset and run again. In case the
        routine was interrupted, it will run from the last completed step, the
        index of which is saved in the current_step_index attribute of the
        routine.
        Additionally, it is possible to start the routine from a specific step.

        Args:
            start_index (int): Index of the step to start with.
            stop_index (int): Index of the step to stop before. The step at this
                index will NOT be executed. Indices start at 0.

                For example, if a routine consists of 3 steps, [step0, step1,
                step2], then the method will stop before step2 (and thus after
                step1), if stop_index is set to 2.

        FIXME: There's an issue when starting from a given start index. The
         routine_steps is only wiped if the routine ran completely and is reran
         from the start. In the future, it might be good to implement a way so
         the user can choose if previous results should be wiped or not (that is,
         if routine_steps should be wiped or not).
        """
        routine_name = self.name

        # Saving instrument settings pre-routine
        if (self.get_param_value('save_instrument_settings') or
                not self.get_param_value("update")):
            # saving instrument settings before the routine
            self.MC.create_instrument_settings_file(
                f"pre-{self.name}_routine-settings")
            self.preroutine_timestamp = a_tools.get_last_n_timestamps(1)[0]
        else:
            # Registering start of routine so all data in measurement period can
            # be retrieved later to determine the Hamiltonian model
            self.preroutine_timestamp = self.MC.get_datetimestamp()

        # Rerun routine if already finished
        if (len(self.routine_template) != 0) and (self.current_step_index >=
                                                  len(self.routine_template)):
            self.create_initial_routine(load_parameters=False)
            self.run()
            return

        # Start and stop indices
        if start_index is not None:
            self.current_step_index = start_index
        elif self.current_step_index >= len(self.routine_template):
            self.current_step_index = 0

        if stop_index is None:
            stop_index = np.Inf

        # Running the routine
        while self.current_step_index < len(self.routine_template):
            j = self.current_step_index
            step_name = self.routine_template.step_name(index=j)

            # Preparing the next step (incl. temporary values)
            self.prepare_step()

            # Interrupting if we reached the stop condition
            if self.current_step_index >= stop_index:
                if self.get_param_value('verbose'):
                    print(f"Partial routine {routine_name} stopped before "
                          f"executing step {j} ({step_name}).")
                return

            # Executing the step
            with temporary_value(*self.current_step_tmp_vals):
                self.execute_step()

            if self.get_param_value('verbose'):
                print(f"{routine_name}, step {j} ({step_name}), done!", "\n")

        if self.get_param_value('verbose'):
            print(f"Routine {routine_name} finished!")

        # Saving instrument settings post-routine
        if (self.get_param_value('save_instrument_settings') or
                not self.get_param_value("update")):
            # Saving instrument settings after the routine
            self.MC.create_instrument_settings_file(
                f"post-{routine_name}_routine-settings")

        # Reloading instrument settings if update is False
        if not self.get_param_value("update"):
            if self.get_param_value('verbose'):
                print(f"Reloading instrument settings from before routine "
                      f"(ts {self.preroutine_timestamp})")

            reload_settings(self.preroutine_timestamp,
                            qubits=self.qubits,
                            dev=self.dev,
                            DCSources=self.DCSources,
                            fluxlines_dict=self.kw.get("fluxlines_dict")
                            )

    def create_initial_parameters(self):
        """Adds any keyword passed to the routine constructor to the
        configuration parameter dictionary. For an AutomaticCalibrationRoutine,
        these keyword will be added to the 'General' scope. These settings
        would have priority over the settings specified in the keyword
        'settings_user'.
        """
        update_nested_dictionary(
            self.settings,
            {self.highest_lookup: {
                self.highest_sublookup: self.kw
            }})

    def create_initial_routine(self, load_parameters=True):
        """Creates (or recreates) initial routine by defining the routine
        template, set routine_steps to an empty array, and setting the
        current step to 0.

        Args:
            load_parameters (bool): Whether to reload the initial parameters.
                Defaults to True.

        NOTE: This method wipes the results of the previous run stored in
        routine_steps.
        """

        # Loading initial parameters. Note that if load_parameters=False,
        # the parameters are not reloaded and thus remain the same. This is
        # desired when wanting to rerun a routine
        if load_parameters:
            self.create_initial_parameters()

        self.create_routine_template()

        # making sure all subroutines update relevant parameters
        self.routine_template.update_all_step_settings({"update": True})

    def final_init(self, **kwargs):
        """A function to be called after the initialization of all base classes,
        since some functionality in the init of a routine needs the base
        classes already initialized.
        """
        # Loading hierarchical settings and creating initial routine
        self.create_initial_routine(load_parameters=False)
        if self.autorun:
            # FIXME: if the init does not finish the object does not exist and
            #  the routine results are not accessible
            try:
                self.run()
            except:
                log.error(
                    "Autorun failed to fully run, concluded routine steps "
                    "are stored in the routine_steps attribute.",
                    exc_info=True,
                )

    @property
    def parameters_qubit(self):
        """
        Returns:
            dict: The parameters of the qubit, including the read-out frequency,
                the anharmonicity and (if present) the latest Hamiltonian model
                parameters containing the total Josephson energy, the charging
                energy, voltage per phi0, the dac voltage, the asymmetry, the
                coupling constant and bare read-out resonator frequency
                (overwriting the previous frb value).

        FIXME: The selection of parameters extracted from the qb is currently
         tailored to the first example use cases. This either needs to be
         generalized to extract more parameters here, or we could decide the
         concrete routines could override the method to extract their specific
         parameters.
        """
        qb = self.qubit

        settings = {}

        hamfit_model = qb.fit_ge_freq_from_dc_offset()

        # Extracting settings from the qubit
        settings.update({
            "fr": hamfit_model.get("fr", qb.ro_freq()),
            "anharmonicity": qb.anharmonicity(),
        })

        # Getting transmon settings from present Hamiltonian model if it exists
        settings.update(hamfit_model)

        return settings

    def view(self,
             print_global_settings=True,
             print_general_settings=True,
             print_tmp_vals=False,
             print_results=True,
             **kws
             ):
        """Prints a user-friendly representation of the routine template.

        Args:
            print_global_settings (bool): If True, prints the global settings
                of the routine. Defaults to True.
            print_general_settings (bool): If True, prints the 'General' scope
                of the routine settings. Defaults to True.
            print_tmp_vals (bool): If True, prints the temporary values of the
                routine. Defaults to False.
            print_results (bool): If True, prints the results dicts of all the
                steps of the routine.
        """

        print(self.name)

        if print_global_settings:
            print("Global settings:")
            pprint.pprint(self.global_settings)
            print()

        if print_general_settings:
            print("General settings:")
            pprint.pprint(self.settings[self.name]['General'])
            print()

        for i, x in enumerate(self.routine_template):
            print(f"Step {i}, {x[0].__name__} ({x[1]})")
            print("Settings:")
            pprint.pprint(x[2], indent=4)

            if print_tmp_vals:
                try:
                    print("Temporary values:")
                    pprint.pprint(x[3], indent=4)
                except IndexError:
                    pass
            print()

        if print_results:
            print_step_results(self)

    def update_settings_at_index(self, settings: dict, index):
        """Updates the settings of the step at position 'index'. Wrapper of
        the method of RoutineTemplate.

        Args:
            settings (dict): The new settings that will update the existing
                settings of the step at position 'index' in the routine
                template.
            index (int): Index of the step for which the temporary values should
                be used.
        """
        self.routine_template.update_settings_at_index(settings, index)

    def get_step_class_at_index(self, index):
        """Returns the step class for a specific step in the routine template.
        Wrapper of the method of RoutineTemplate.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            class: The class of the step at position 'index' in the routine
                template.
        """
        return self.routine_template.get_step_class_at_index(index)

    def get_step_label_at_index(self, index):
        """Returns the step label for a specific step in the routine template.

        Args:
            index (int): Index of the step for which the step label is to be
                returned.
        Returns:
            str: The label of the step at position 'index' in the routine
                template.
        """
        return self.routine_template.get_step_label_at_index(index)

    def get_step_settings_at_index(self, index):
        """Returns the settings for a specific step in the routine template.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            dict: The settings dictionary of the step at position 'index' in
                the routine template.
        """
        return self.routine_template.get_step_settings_at_index(index)

    def get_step_tmp_vals_at_index(self, index):
        """Returns the temporary values of the step at index.

        Args:
            index (int): Index of the step for which the settings are to be
                returned.

        Returns:
            list: The temporary values for the step at position 'index' in the
                routine template. Each entry is a tuple made of a
                QCoDeS parameter and its temporary value.
        """
        return self.routine_template.get_step_tmp_vals_at_index(index)

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """Extends the temporary values of the step at index. If the step does
        not have any temporary values, it sets the temporary values to the
        passed temporary values.

        Args:
            tmp_vals (list): The temporary values for the step at position
                'index' in the routine template. Each entry should be a tuple
                made of a QCoDeS parameter and its temporary value.
            index (int): Index of the step for which the temporary values should
                be used.
        """
        self.routine_template.extend_step_tmp_vals_at_index(tmp_vals=tmp_vals,
                                                            index=index)

    def add_step(self,
                 step_class: Type[Step],
                 step_label: str,
                 step_settings: Optional[Dict[str, Any]] = None,
                 step_tmp_vals=None,
                 index=None):
        """Adds a step to the routine template. The settings of the step are
        extracted from the configuration parameter dictionary.

        Args:
            step_class (Step): Class of the step
            step_label (str): Label of the step
            step_settings (dict, optional): Settings of the step. If any settings
                are found in step_settings['settings'], they will have priority
                over those found in the configuration parameter dictionary.
            step_tmp_vals (list, optional): Temporary values for the step. Each
                entry is a tuple made of a QCoDeS parameter and its
                temporary value. Defaults to None.
            index (int, optional): Index of the routine template at which the
                step should be added. If None, the step will be added at the end
                of the routine. Defaults to None.
        """
        updated_step_settings = self.extract_step_settings(
            step_class, step_label, step_settings)
        step_settings['settings'] = updated_step_settings
        self.routine_template.add_step(step_class,
                                       step_label,
                                       step_settings,
                                       step_tmp_vals=step_tmp_vals,
                                       index=index)

    def get_empty_device_properties_dict(self, step_type=None):
        """Returns an empty dictionary of the following structure, for use with
        `get_device_property_values`

        Example:
            .. code-block:: python
                {
                    'step_type': step_type,
                    'property_values': [],
                    'timestamp': '20220101_161403',
                }
        Args:
            step_type (str, optional): The name of the step. Defaults to the
                class name.

        Returns:
            dict: An empty results dictionary (i.e., no results)
        """
        return {
            'step_type':
                step_type if step_type is not None else str(
                    type(self).__name__),
            'property_values': [],
            'timestamp':
                self.preroutine_timestamp,
        }

    def get_device_property_values(self, **kwargs):
        """Returns a dictionary of high-level device property values from
        running this routine, and all of its steps.

        `qubit_sweet_spots` can be used to prefix `property_type` based on the
        sweet-spots of the qubit. An example for `qubit_sweet_spots` is given
        below. `None` as a sweet-spot will not add a prefix.

        .. code-block:: python
            {
                'qb1': 'uss',
                'qb4': 'lss',
                'qb7': None,
            }


        An example of what is returned is given below.

        Example:
            .. code-block:: python
                {
                    'step_type':
                    'AutomaticCalibrationRoutine',
                    'timestamp':
                    '20220101_163859',  # from self.preroutine_timestamp
                    'property_values': [
                        {
                            'step_type': 'T1Step',
                            'property_values': [
                                {
                                    'qubits': ['qb1'],
                                    'component_type': 'qb',
                                    'property_type': 'ge_T1_time',
                                    'value': 1.6257518120474107e-05,
                                    'timestamp': '20220101_163859',
                                    'folder_name': 'Q:\\....',
                                },
                            ]
                        },
                        {
                            'step_type': 'RamseyStep',
                            'property_values': [
                                {
                                    'qubits': ['qb1'],
                                    'component_type': 'qb',
                                    'property_type': 'ge_t2_echo',
                                    'value': 7.1892927355629493e-06,
                                    'timestamp': '20220101_163859',
                                    'folder_name': 'Q:\\....',
                                },
                            ]
                        }
                    ]
                }


        Returns:
            dict: dictionary of high-level device property_values determined by
                this routine
        """
        results = self.get_empty_device_properties_dict()
        for _, step in enumerate(self.routine_steps):
            step_i_results = step.get_device_property_values(**kwargs)
            results['property_values'].append({
                "step_type":
                    str(type(step).__name__)
                    if step_i_results.get('step_type') is None else
                    step_i_results.get('step_type'),
                "property_values":
                    step_i_results['property_values'],
            })
        return results

    @property
    def global_settings(self):
        """
        Returns:
            dict: The global settings of the routine
        """
        return self.routine_template.global_settings

    @property
    def name(self):
        """Returns the name of the routine.
        """
        # Name depends on whether the object is initialized.
        if type(self) is not type:
            return type(self).__name__
        else:
            try:
                return self.__name__
            except:
                return "AutomaticCalibration"

    # Initializing necessary attributes, should/can be overridden by children
    _DEFAULT_PARAMETERS = {}
    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([])


def keyword_subset(keyword_arguments, allowed_keywords):
    """Returns a dictionary with only the keywords that are specified in
    allowed_keywords.

    Args:
        keyword_arguments (dict): Original dictionary from which the allowed
            keywords will be extracted.
        allowed_keywords (list): List of keywords to pick from the original
            dictionary.

    Returns:
        dict: The new dictionary containing only the allowed keywords and the
            corresponding values found in keyword_arguments.
    """
    keywords = set(keyword_arguments.keys())
    keyswords_to_extract = keywords.intersection(allowed_keywords)

    new_kw = {key: keyword_arguments[key] for key in keyswords_to_extract}

    return new_kw


def keyword_subset_for_function(keyword_arguments, function):
    """Returns a dictionary with only the keywords that are used by the
    function.

    Args:
        keyword_arguments (dict): Original dictionary from which the allowed
            keywords will be extracted.
        function (function): Function from which the allowed arguments are
            extracted.

    Returns:
        dict: The new dictionary containing only the keywords arguments
            extracted from the given function.
    """
    allowed_keywords = inspect.getfullargspec(function)[0]

    return keyword_subset(keyword_arguments, allowed_keywords)


def print_step_results(step: Step, routine_name: str = ''):
    """Recursively print the results of the step and its sub-steps."""
    if step.results:  # Do not print None or empty dictionaries
        print(f'{routine_name} Step {step.step_label} results:')
        pprint.pprint(step.results)
        print()
    if hasattr(step, 'routine_steps'):  # A routine with sub-steps
        for sub_step in step.routine_steps:
            print_step_results(sub_step, routine_name=step.step_label)
    else:
        pass
