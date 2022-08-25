from pycqed.measurement.calibration.single_qubit_gates import (
    SingleQubitGateCalibExperiment
)
from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.measurement.quantum_experiment import QuantumExperiment
from pycqed.utilities.state_and_transition_translation import *
from pycqed.utilities.general import temporary_value
from pycqed.utilities.reload_settings import reload_settings
from collections import OrderedDict as odict

import pycqed.analysis.analysis_toolbox as a_tools
import numpy as np
import copy
import logging
import pprint
import inspect
import collections.abc
import os
import json
import re

log = logging.getLogger(__name__)

try:
    from pycqed.utilities import devicedb
except ModuleNotFoundError:
    log.info("The module 'device-db-client' was not successfully imported. "
             "The device database features will not be available.")
    _device_db_client_module_missing = True
else:
    _device_db_client_module_missing = False


class SettingsDictionary(dict):
    """This class represents the configuration parameters specified in default,
    setup, and sample folder as a dictionary.
    The hierarchy (in descending significance) is User - Sample - Setup - Default
    """

    _USE_DB_STRING = "USE_DB"

    def __init__(self,
                 init_dict={},
                 db_client=None,
                 db_client_config=None,
                 dev_name=None):
        """
        Initializes the dictionary.

        Args:
            init_dict (dict): A dictionary to initialize the custom dictionary
                with.
            db_client (devicedb.Client): A client to connect to the device
                database.
            db_client_config (devicedb.Config): A configuration object for the
                device database as an alternative way to initialize the database
                connection.
            dev_name (str): The name of the device in use. Used to tell the
                device database which device it should use.
        """

        # Copies items, no deep copy
        super().__init__(init_dict)

        self.dev_name = dev_name

        self.db_client = None
        if db_client is not None:
            if _device_db_client_module_missing:
                log.warning(
                    "Can't use the given client to connect to the "
                    "device database. The module 'device-db-client' was not "
                    "successfully imported.")
            else:
                self.db_client = db_client
        elif db_client_config:
            if _device_db_client_module_missing:
                log.warning(
                    "Can't use the given configuration to connect to "
                    "the device database. The module 'device-db-client' was "
                    "not successfully imported.")
            else:
                self.enable_use_database(db_client_config)

    def update_user_settings(self, settings_user):
        """Updates the current configuration parameter dictionary with the
        provided user parameters.

        Args:
            settings_user (dict): A dictionary with the user configuration
                parameters.
        """
        update_nested_dictionary(self, settings_user)

    def _get_unprocessed_param_value(self,
                                     param,
                                     lookups,
                                     sublookups=None,
                                     qubit=None,
                                     groups=None,
                                     leaf=True,
                                     associated_component_type_hint=None):
        """Looks for the requested parameter recursively in the configuration
        parameter dictionary. It is used as a helper function for
        get_param_value, but the actual search in the nested dictionary is
        done here.

        Args:
            param (str): The name of the parameter to look up.
            lookups (list of str): The scopes in which the parameter should be
                looked up.
            sublookups (list of str): The subscopes to be looked up. The
                parameter is then looked up in self[lookup][sublookup]
            qubit (str): The name of the qubit, if the parameter is
                qubit-specific.
            groups (list of str): The groups the qubit is in, as specified in
                the dictionary.
            leaf (boolean): True if the scope to search the parameter for is a
                leaf node (e.g. a measurement or intermediate step, not a
                routine)
            associated_component_type_hint (str): A hint for the device
                database, if the parameter does not belong to the qubit.

        Returns:
            A tuple with the raw fetched parameter from the dictionary and a
            boolean indicating success.
        """
        for lookup in lookups:
            if lookup is None:
                continue
            if lookup in self:
                if sublookups:
                    # If a scope in lookups is found and there are sublookups,
                    # search again for the parameter in self[lookup] using
                    # the previous sublookups as lookups.
                    val, success = SettingsDictionary.get_param_value(
                        self[lookup],
                        param,
                        lookups=sublookups,
                        qubit=qubit,
                        groups=groups,
                        leaf=leaf,
                        associated_component_type_hint=
                        associated_component_type_hint)
                    if success:
                        return val, success
                elif not leaf and lookup != 'General':
                    # If there are no sublookups, use 'General' as a sublookup
                    # scope and the lookup node is not a leaf node (e.g., not
                    # a QuantumExperiment step)
                    val, success = SettingsDictionary.get_param_value(
                        self,
                        param,
                        lookups=[lookup],
                        sublookups=['General'],
                        qubit=qubit,
                        groups=groups,
                        associated_component_type_hint=
                        associated_component_type_hint)
                    if success:
                        return val, success
                else:
                    # Look if param is defined for a specific qubit or a group
                    # of qubits
                    if qubit and 'qubits' in self[lookup]:
                        for group, v in self[lookup]['qubits'].items():
                            if group == qubit or group in groups:
                                if param in v:
                                    return v[param], True
                    # Return the parameter if it is found in self[lookup]
                    if param in self[lookup]:
                        return self[lookup][param], True

        return None, False

    def get_param_value(self,
                        param,
                        lookups,
                        sublookups=None,
                        qubit=None,
                        groups=None,
                        leaf=True,
                        associated_component_type_hint=None):
        """Looks up the requested parameter in the configuration parameters.
        If the fetched value is a request to query the database, the queried
        value is returned.

        Args:
            param (str): The name of the parameter to look up.
            lookups (list of str): The scopes in which the parameter should be
                looked up.
            sublookups (list of str): The subscopes to be looked up. The
                parameter is then looked up in self[lookup][sublookup]
            qubit (str): The name of the qubit, if the parameter is
                qubit-specific.
            groups (list of str): The groups the qubit is in, as specified in
                the dictionary.
            leaf (boolean): True if the scope to search the parameter for is a
                leaf node (e.g. a measurement or intermediate step, not a
                routine).
            associated_component_type_hint (str): A hint for the device
                database, if the parameter does not belong to the qubit.

        Returns
            A tuple with the postprocessed fetched parameter from the
            dictionary and a boolean indicating success.
        """

        val, success = SettingsDictionary._get_unprocessed_param_value(
            self, param, lookups, sublookups, qubit, groups, leaf)

        # Use database value
        if isinstance(val, list) and len(
                val) and val[0] == SettingsDictionary._USE_DB_STRING:
            if _device_db_client_module_missing:
                log.warning(
                    "Can't read values from the device database. "
                    "The module 'device-db-client' was not successfully "
                    "imported.")
            else:
                if qubit is None:
                    raise ValueError(
                        "When using the database, only parameters associated "
                        "with a qubit are allowed. Provide qubit as a keyword.")
                db_value = self.db_client.get_property_value_from_param_args(
                    qubit.name, param, associated_component_type_hint)
                success = db_value is not None
                if success:
                    return db_value.value, success
                elif len(val) > 1:
                    return val[1], True
                else:
                    return None, False

        return val, success

    def get_qubit_groups(self, qubit, lookups):
        """Gets the groups the specified qubit belongs to out of the
        configuration parameter dictionary.

        Args:
            qubit (str): The name of the qubit.
            lookups (list of str): The scopes in which to search for the qubits
                group definitions. Default is ['Groups'].

        Returns:
            set: A set of strings with the group names the qubit is part of.
        """
        groups = set()
        for lookup in lookups:
            if lookup in self:
                for group_name, group in self[lookup].items():
                    if qubit in group:
                        groups.add(group_name)
        return groups

    def load_settings_from_file(self,
                                settings_default_folder=None,
                                settings_setup_folder=None,
                                settings_sample_folder=None,
                                settings_user=None):
        """Loads the device settings from the folders storing Default, Setup and
        Sample parameters and puts it into the configuration parameter
        dictionary as a nested dictionary. The folders should contain JSON files.
        Order in the hierarchy: Sample overwrites Setup overwrites Default values.
        Additionally, when settings_sample_folder is specified, it overwrites
        the dictionary loaded from the files.

        Since JSON only supports strings as keys, postprocessing is applied.
        Therefore, it is also possible to use a string with a tuple inside as a
        key, which is converted to a tuple in the dictionary, e.g.
        "('SFHQA', 1)" is converted to ('SFHQA', 1).

        Args:
            settings_default_folder (string): Full path to the folder for
                default settings. If None, uses the default PycQED settings
                in "autocalib_default_settings".
            settings_setup_folder (string): Full path to the folder for setup
                settings.
            settings_sample_folder (string): Full path to the folder for sample
                settings.
            settings_default_folder (string): User settings as a dictionary.
        """
        if settings_default_folder == None:
            dirname = os.path.dirname(os.path.abspath(__file__))
            settings_default_folder = os.path.join(
                dirname, "autocalib_default_settings")
        if settings_setup_folder is None:
            log.warning("No settings_setup_folder specified.")
        if settings_sample_folder is None:
            log.warning("No settings_sample_folder specified.")

        for settings_folder in [
                settings_default_folder, settings_setup_folder,
                settings_sample_folder
        ]:
            if settings_folder is not None:
                settings_files = os.listdir(settings_folder)
                for file in settings_files:
                    with open(os.path.join(settings_folder, file)) as f:
                        update_nested_dictionary(
                            self, {os.path.splitext(file)[0]: json.load(f)})

        if settings_user is not None:
            self.update_user_settings(settings_user, reload_database=False)

        self._postprocess_settings_from_file()

    def _postprocess_settings_from_file(self):
        """Since JSON only supports strings as keys, postprocessing is applied.
        Therefore, it is also possible to use a string with a tuple inside as a
        key, which is converted to a tuple in the dictionary, e.g.
        "('SFHQA', 1)" is converted to ('SFHQA', 1).
        """
        for k in list(self.keys()):
            if isinstance(self[k], collections.abc.Mapping):
                SettingsDictionary._postprocess_settings_from_file(self[k])
            if re.search('^\(.*\)$', k):  # represents a tuple
                self[eval(k)] = self.pop(k)

    def enable_use_database(self, db_client_config):
        """Can be called to enable querying the database with configuration
        parameters. Either this function is called or the database client is
        already specified in the init of the dictionary.

        Args:
            db_client_config (devicedb.Config): A configuration object for the
                device.
        """
        if self.dev_name is not None:
            db_client_config.device_name = self.dev_name
        self.db_client = devicedb.Client(db_client_config)

    def copy(self, overwrite_dict=None):
        """Creates a deepcopy of the current dictionary.

        Args:
            overwrite_dict (dict): When this is not None, the content of the
                dictionary is cleared and replaced with the dict specified in
                overwrite_dict. Other arguments of the dictionary like the
                device database client stay the same.

        Returns:
            SettingsDictionary: the deepcopied dictionary.
        """

        if overwrite_dict is None:
            overwrite_dict = self
        settings_copy = SettingsDictionary(copy.deepcopy(overwrite_dict),
                                           db_client=self.db_client,
                                           dev_name=self.dev_name)

        return settings_copy

    def __deepcopy__(self, memo):
        """Overloads the standard deepcopying function to enable
        deepcopying the dictionary by disabling deepcopying of certain argument
        objects.
        """
        return self.__class__(
            {k: copy.deepcopy(v, memo) for k, v in self.items()},
            db_client=self.db_client,
            dev_name=self.dev_name)


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
            steps (list): List of steps that define the routine. Each step
                consists of a list of three elements. Namely, the step class,
                the step label (a string), and the step settings (a dictionary).
                For example:
                steps = [
                    [StepClass1, step_label_1, step_settings_1],
                    [StepClass2, step_label_2, step_settings_2],
                ]
            global_settings (dict, optional): Dictionary containing global
                settings for each step of the routine (e.g., "dev", "update",
                "delegate_plotting"). Defaults to None.
            routine (AutomaticCalibrationRoutine, optional): Routine that the
                RoutineTemplate defines. Defaults to None.
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
        print_global_settings=True,
        print_general_settings=True,
        print_tmp_vals=False,
    ):
        """Prints a user-friendly representation of the routine template.

        Args:
            print_global_settings (bool): If True, prints the global settings
                of the routine. Defaults to True.
            print_general_settings (bool): If True, prints the 'General' scope
                of the routine settings. Defaults to True.
            print_tmp_vals (bool): If True, prints the temporary values of the
                routine. Defaults to False.
        """
        try:
            print(self.routine.name)
        except AttributeError:
            pass

        if print_global_settings:
            print("Global settings:")
            pprint.pprint(self.global_settings)
            print()

        if print_general_settings:
            try:
                print("General settings:")
                pprint.pprint(
                    self.routine.settings[self.routine.name]['General'])
                print()
            except AttributeError:
                pass

        for i, x in enumerate(self):
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

    def __str__(self):
        """Returns a string representation of the routine template.

        FIXME: this representation does not include the tmp_vals of the steps.
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
                s += "General settings:\n"
                s += pprint.pformat(
                    self.routine.settings[self.routine.name]['General']) + "\n"
        except AttributeError:
            pass

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
        assert (
            len(step) == 3 or len(step) == 4
        ), "Step must be a list of length 3 or 4 (to include temporary values)"
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


class Step:
    """A class to collect functionality used for each step in a routine.
    A step can be an AutomaticCalibrationRoutine, an IntermediateStep
    or a measurements. Measurements are wrapped with a wrapper class
    which inherits from this class to give access to its functions.
    """

    def __init__(self, dev, routine=None, **kw):
        """Initializes the Step class.

        Arguments:
            dev (Device): The device which is currently measured.
            routine (Step): The parent of the step. If this step is the root
                routine, this should be None.

        Keyword Arguments:
            step_label (str): A unique label for this step to be used in the
                configuration parameters files.
            settings (SettingsDictionary): The configuration parameters
                passed down from its parent. If None, the dictionary is taken
                from the Device object.
            qubits (list): A list with the Qubit objects which should be part of
                the step.
            settings_user (dict): A dictionary from the user to update the
                configuration parameters with.

        """
        self.routine = routine
        self.step_label = self.kw.pop('step_label', None)
        self.dev = dev
        # Copy default settings from autocalib if this is the root routine, else
        # create an empty SettingsDictionary
        default_settings = self.dev.autocalib_settings.copy(
        ) if self.routine is None else self.routine.settings.copy({})
        self.settings = self.kw.pop("settings", default_settings)
        self.qubits = self.kw.pop("qubits", self.dev.get_qubits())

        # FIXME: this is there to make the current one-qubit-only implementation
        # of HamiltionianFitting work easily
        # remove dependency on self.qubit
        self.qubit = self.qubits[0]

        settings_user = self.kw.pop('settings_user', None)
        if settings_user:
            self.settings.update_user_settings(settings_user)
        self.parameter_lookups = [
            self.step_label,
            self.get_lookup_class().__name__, 'General'
        ]
        self.parameter_sublookups = None
        self.leaf = True

    class NotFound:
        """This class is used in get_param_value to identify the cases where
        a keyword could not be found in the configuration parameter dictionary.
        It is necessary to distinguish between the cases when None is explicitly
        specified for a keyword argument and when no keyword argument was found.
        """

        def __bool__(self):
            """Return False by default for the truth value of an instance of
            NotFound.
            """
            return False

    def get_param_value(self,
                        param,
                        qubit=None,
                        sublookups=None,
                        default=None,
                        leaf=None,
                        associated_component_type_hint=None):
        """Looks up the requested parameter in the own configuration parameter
        dictionary. If no value was found, the parent routine's function is
        called recursively up to the root routine.

        This effectively implements the priority hierarchy for the settings: the
        settings specified for the most specific step will have priority over
        those specified for the more general ones. For instance, the settings
        of a QuantumExperiment step specified within the settings of an
        AutomaticRoutine step will have priority over the generic settings of
        the same QuantumExperiment Step specified in the root node of the
        dictionary.

        The lookups used for the search are those defined in
        self.parameters_lookups, i.e., ['step_label', 'StepClass', 'General']
        (sorted by priority).

        Args:
            param (str): The name of the parameter to look up.
            sublookups (list of str): Optional subscopes to be looked up. The
                sublookups will be assumed to be sorted by priority (highest
                priority first).
            default: The default value the parameters falls back to if no value
                was found for the parameter in the whole dictionary. By setting
                it to NotFound() it is possible to detect whether the value
                was found in the configuration parameter dictionary. Defaults to
                None.
                FIXME: A better solution would be to change the function and
                return also a bool indicating whether a parameter was found.
                This would require minimal changes in this function, but it
                would require to go through the code and fix all the lines
                containing a call to get_param_value.
            leaf (boolean): True if the scope to search the parameter for is a
                leaf node (e.g. a measurement or intermediate step, not a
                routine)
            associated_component_type_hint (str): A hint for the device database,
                if the parameter does not belong to the qubit.

        Returns:
            The value found in the dictionary. If no value was found, either the
            default value is returned if specified or otherwhise None.
        """
        # Get the groups the specified qubit belongs to. This allows searching
        # for qubit-specific settings
        groups = None
        if qubit is not None:
            groups = self.get_qubit_groups(qubit)
        # Look for the parameter in  self.settings with the initial default
        # lookups. Note that self.settings is different for different steps.
        lookups = self.parameter_lookups
        if leaf is None:
            leaf = self.leaf
        val, success = self.settings.get_param_value(
            param,
            lookups=lookups,
            sublookups=sublookups,
            qubit=qubit,
            groups=groups,
            leaf=leaf,
            associated_component_type_hint=associated_component_type_hint)

        if not success:
            # If the initial search failed, repeat it by calling the
            # parent routine's function (if there is a parent routine). Keep the
            # sublookups if they were specified, otherwise use the initial
            # lookups as sublookups.
            if self.routine is not None:
                sublookups = sublookups if sublookups else lookups
                val = self.routine.get_param_value(
                    param,
                    qubit=qubit,
                    sublookups=sublookups,
                    leaf=leaf,
                    associated_component_type_hint=associated_component_type_hint
                )
                if val is None or type(val) == self.NotFound:
                    success = False
                else:
                    success = True
            # If the initial search failed and there is no parent routine,
            # look for the parameter in the settings using the sublookups as
            # lookups. Basically, search for the given parameter within the
            # given sublookups in the root node of the settings.
            elif sublookups:
                val, success = self.settings.get_param_value(
                    param,
                    lookups=sublookups,
                    sublookups=None,
                    qubit=qubit,
                    groups=groups,
                    leaf=leaf,
                    associated_component_type_hint=associated_component_type_hint
                )

        return val if success else default

    def get_qubit_groups(self, qubit):
        """Gets the groups the specified qubit belongs to out of the
        configuration parameter dictionary.

        Args:
            qubit (str): The name of the qubit.

        Returns:
            set: A set of strings with the group names the qubit is part of.
        """
        # FIXME: When a qubit is removed from a group with the same
        # name in higher hierarchy, it will still remain.
        lookups = ['Groups']
        groups = self.settings.get_qubit_groups(qubit, lookups)
        if self.routine is not None:
            groups.update(self.routine.get_qubit_groups(qubit))
        return groups

    def run(self):
        """Run the Step. To be implemented by subclasses.
        """
        pass

    def get_empty_device_properties_dict(self, step_type=None):
        """Returns an empty dictionary of the following structure, for use with
        `get_device_property_values`

        Example:
            .. code-block:: python
                {
                    'step_type': step_type,
                    'property_values': []
                }
        Args:
            step_type (str, optional): The name of the step. Defaults to the
                class name.

        Returns:
            dict: An empty property value dictionary (i.e., no results)
        """
        return {
            'step_type':
                step_type if step_type is not None else str(
                    type(self).__name__),
            'property_values': [],
        }

    def get_device_property_values(self, **kwargs):
        """Returns a dictionary of high-level property values from running this
        step. To be overridden by children classes.

        Here is an example of the output dictionary

        Example:
            .. code-block:: python
                {
                    'step_type': 'RamseyStep',
                    'property_values': [{
                        'qubits': 'qb2',
                        'component_type': 'qb',
                        'property_type': 'ge_T2_star',
                        'value': 6.216582600129854e-05,
                        'timestamp': '20220101_101500',
                        'rawdata_folder_path': 'Q:\\....\\20220101\\101500_...',
                    }, {
                        'qubits': 'qb7',
                        'component_type': 'qb',
                        'property_type': 'ge_T2_star',
                        'value': 1.9795263942036515e-05,
                        'timestamp': '20220101_101500',
                        'rawdata_folder_path': 'Q:\\....\\20220101\\101500_...',
                    }]
                }


        Returns:
            dict: dictionary of high-level property values (may be empty)
        """
        # Default return is an empty dictionary
        return self.get_empty_device_properties_dict()

    @classmethod
    def gui_kwargs(cls, device):
        """Returns the kwargs necessary to run a QuantumExperiment. Every
        QuantumExperiment should implement them. The keywords returned by
        this method will be included in the requested settings (see
        get_requested_settings) and eventually extracted from the configuration
        parameter dictionary (see parse_settings).

        NOTE: The name of the function could be confusing. This function was
        first implemented to retrieve the kwargs to be specified with a GUI.
        The same name was maintained to make use of the already implemented
        functions.

        Args:
            device (Device): The device that is being measured.

        Returns:
            dict: Dictionary of kwargs necessary to run a QuantumExperiment.
        """
        return {
            'kwargs':
                odict({
                    Step.__name__: {
                        # kwarg: (fieldtype, default_value),
                        # 'delegate_plotting': (bool, False),
                    },
                })
        }

    def get_requested_settings(self):
        """Gets a set of keyword arguments which are needed for the
        initialization of the current step. The keywords are retrieved via the
        gui_kwargs method that is implemented in each QuantumExperiment.

        Returns:
            dict: A dictionary containing names and default values of keyword
                arguments which are needed for the current step.
        """
        gui_kwargs = self.__class__.gui_kwargs(self.dev)
        # Remove second layer of dict
        requested_kwargs = {
            k: {ky: vy for kx, vx in v.items()
                for ky, vy in vx.items()}
            for k, v in gui_kwargs.items()
        }
        return requested_kwargs

    def parse_settings(self, requested_kwargs):
        """Resolves the keyword arguments from get_requested_settings to calls
        within the parameter dictionary.

        Args:
            requested_kwargs (dict): A dictionary containing the names and
            default values of the keyword arguments for an experiment.
        Returns:
            The exact keyword arguments to pass to the experiment class.
        """
        kwargs = {}
        for k, v in requested_kwargs['kwargs'].items():
            kwargs[k] = self.get_param_value(k, default=self.NotFound())
            # If the keyword was not found in the configuration parameter
            # dictionary, it will not be passed to the QuantumExperiment.
            # This prevents problem when the gui_kwargs default values raise
            # errors
            if type(kwargs[k]) == self.NotFound:
                kwargs.pop(k)

        kwargs['measure'] = False
        kwargs['analyze'] = False
        kwargs['qubits'] = self.qubits
        return kwargs

    @property
    def highest_lookup(self):
        """Returns the highest scope for this step which is not None
        in the order step_label, class name, General.

        Returns:
            str: A string with the highest lookup of this step which is not
                None.
        """
        return self._get_first_not_none(self.parameter_lookups)

    @property
    def highest_sublookup(self):
        """Returns the highest subscope for this step which is not None.

        Returns:
            str: A string with the highest sublookup of this step which is not
                None. If the step is a leaf, this is None, otherwhise it is
                "General".
        """
        return None if self.parameter_sublookups is None else self._get_first_not_none(
            self.parameter_sublookups)

    def _get_first_not_none(self, lookup_list):
        """Returns the first subscope in the lookup list that is not None.

        Args:
            lookup_list (list): List of scopes to look up.

        Returns:
            str: The first subscope in lookup_list that is not None. If all the
                entries in lookup_list are None, returns None.
        """
        return next((item for item in lookup_list if item is not None), None)

    @classmethod
    def get_lookup_class(cls):
        """
        Returns:
            class: The class corresponding to the experiment that the Step is
                based on. If there is no experiment, the class itself is
                returned.
        """
        if issubclass(cls, SingleQubitGateCalibExperiment):
            return cls.__bases__[0]
        if issubclass(cls, QuantumExperiment):
            return cls.__bases__[0]
        return cls


class IntermediateStep(Step):
    """Class used for defining intermediate steps between automatic calibration
    steps.

    NOTE: Currently, there is no difference between an IntermediateStep and a
    Step. A different class was implemented just in case future modifications
    will make it necessary.
    """

    def __init__(self, **kw):
        self.kw = kw
        super().__init__(**kw)

    def run(self):
        """Intermediate processing step to be overridden by Children (which are
        routine specific).
        """
        pass


class AutomaticCalibrationRoutine(Step):
    """Base class for general automated calibration routines

    NOTE: In the children classes, it is necessary to call final_init at
    the end of their constructor. It is not possible to do this in the
    parent class because some routines need to do some further initialization
    after the base class __init__ and before the routine is actually created
    (which happens in final_init).

    In the children classes, the intialization follows this hierarchy:
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

        Configuration parameters (coming from the configuration parameter dictionary):
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

        self.DCSources = self.kw.pop("DCSources", None)

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

        At the end, ExperimentStep.settings will be updated according to the
        hierarchy specified in the lookups.

        Arguements:
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

        # The control statement prevents looking for the step scopes inside
        # the step settings. This is to avoid considering the following
        # settings:
        #       9) SubRoutine.settings["SubRoutine"]["Experiment"]
        #       10) SubRoutine.settings["SubRoutine"]["experiment_label"]
        #       11) SubRoutine.settings["subroutine_label"]["Experiment"]
        #       12) SubRoutine.settings["subroutine_label"]["experiment_label"]

        # The only exception is when self is the root routine, in this case
        # the routine settings are whole configuration parameter dictionary.
        # Hence, we need to look for the routine scopes within the routine
        # settings (this is why it is included or self.routine is None).
        if self.get_lookup_class(
        ).__name__ not in lookups or self.routine is None:
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
                              step_class,
                              step_label,
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
        if not issubclass(step_class, Step):
            raise NotImplementedError("Steps have to inherit from class Step.")
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
        self.routine_template.routine = self

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
                    qb for qb in self.qubits if qb.name is parallel_group or
                    parallel_group in self.get_qubit_groups(qb.name)
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
            step_settings.pop('settings', {}))
        # Executing the step with corresponding settings
        if issubclass(step_class, qbcal.SingleQubitGateCalibExperiment):
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
            log.error(f"automatic subroutine is not compatible (yet) with the "
                      f"current step class {step_class}")
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
        index of which is saved in the current_step_index attribute of the routine.
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
            self.preroutine_timestamp = a_tools.get_last_n_timestamps(1,)[0]
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

        # Start and stop indeces
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
        self.routine_steps = []
        self.current_step_index = 0

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
            # the routine results are not accesible
            try:
                self.run()
            except:
                log.error(
                    "Autorun failed to fully run, concluded routine steps"
                    "are stored in the routine_steps attribute.",
                    exc_info=1,
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

    def view(self, **kw):
        """Prints a user friendly representation of the routine settings

        Keyword Arguments:
            print_global_settings (bool): If True, prints the global settings
                of the routine. Defaults to True.
            print_parameters (bool): If True, prints the parameters of the
                routine. Defaults to True.
            print_tmp_vals (bool): If True, prints the temporary values of the
                routine. Defaults to False.
        """
        self.routine_template.view(**kw)

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
                 step_class,
                 step_label,
                 step_settings,
                 step_tmp_vals=None,
                 index=None):
        """Adds a step to the routine template. The settings of the step are
        extracted from the configuration parameter dictionary.

        Args:
            step_class (Step): Class of the step
            step_label (str): Label of the step
            step_settings (dict): Settings of the step. If any settings
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
                    'timestamp': '20220101_161403', # from self.preroutine_timestamp
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

        Args:
            qubit_sweet_spots (dict, optional): a dictionary mapping qubits to
                sweet-spots ('uss', 'lss', or None)


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
        # Name depends on whether or not the object is initialized.
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


def update_nested_dictionary(d, u):
    """Updates a nested dictionary. Each value of 'u' will update the
    corresponding entry of 'd'. If an entry of 'u' is a dictionary itself,
    then the function is called recursively, and the subdictionary of 'd' will
    be the dictionary to be updated.

    If 'u' contains a key that does not exist in 'd', it will be added to 'd'.

    Args:
        d (dict): Dictionary to be updated.
        u (dict): Dictionary whose items will update the dictionary 'd'.

    Returns:
        dict: The updated dictionary.
    """
    for k, v in u.items():
        # Check whether the value 'v' is a dictionary. In this case,
        # updates_nested_dictionary is called again recursively. The
        # subdictionary d[k] will be updated with v.
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d
