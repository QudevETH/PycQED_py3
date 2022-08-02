from pycqed.measurement.sweep_points import SweepPoints

from pycqed.measurement.calibration.single_qubit_gates import SingleQubitGateCalibExperiment
from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.utilities import hamiltonian_fitting_analysis as hfa
from pycqed.utilities.state_and_transition_translation import *
from pycqed.utilities.general import temporary_value, configure_qubit_mux_drive, configure_qubit_mux_readout
from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals
from pycqed.utilities.reload_settings import reload_settings
from collections import OrderedDict as odict

import pycqed.analysis.analysis_toolbox as a_tools
import numpy as np
import copy
import logging
import time
import pprint
import inspect
import collections.abc
import os
import json
import re

log = logging.getLogger(__name__)

try:
    from pycqed.utilities import devicedb
    from pycqed.utilities.devicedb import utils as db_utils
except ModuleNotFoundError:
    log.warning("The module 'device-db-client' was not successfully imported. "
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
            init_dict (dict): a dictionary to initialize the custom dictionary
                with
            db_client (devicedb.Client): a client to connect to the device
                database
            db_client_config (devicedb.Config): a configuration object for the
                device database as an alternative way to initialize the database
                connection
            dev_name (str): The name of the device in use. Used to tell the
                device database which device it should use.
        """

        # Copies items, no deep copy
        super().__init__(init_dict)

        self.dev_name = dev_name

        self.db_client = None
        if db_client is not None:
            if _device_db_client_module_missing:
                log.warning("Can't use the given client to connect to the "
                    "device database. The module 'device-db-client' was not "
                    "successfully imported.")
            else:
                self.db_client = db_client
        elif db_client_config:
            if _device_db_client_module_missing:
                log.warning("Can't use the given configuration to connect to "
                    "the device database. The module 'device-db-client' was "
                    "not successfully imported.")
            else:
                self.enable_use_database(db_client_config)

    def update_user_settings(self, settings_user):
        """Updates the current configuration parameter dictionary with the
        provided user parameters.

        Args:
            settings_user (dict): a dictionary with the user configuration
                parameters
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
        """A helper function for get_param_value. It does not include the
        postprocessing of the fetched value by get_param_value.

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
                    val, success = SettingsDictionary.get_param_value(
                        self[lookup],
                        param,
                        sublookups,
                        qubit=qubit,
                        groups=groups,
                        leaf=leaf,
                        associated_component_type_hint=
                        associated_component_type_hint)
                    if success:
                        return val, success
                elif not leaf and lookup != 'General':
                    val, success = SettingsDictionary.get_param_value(
                        self,
                        param, [lookup], ['General'],
                        qubit=qubit,
                        groups=groups,
                        associated_component_type_hint=
                        associated_component_type_hint)
                    if success:
                        return val, success
                else:
                    # look if param is defined for a group
                    if qubit and 'qubits' in self[lookup]:
                        for group, v in self[lookup]['qubits'].items():
                            if group == qubit or group in groups:
                                if param in v:
                                    return v[param], True
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
                routine)
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
                log.warning("Can't read values from the device database. "
                    "The module 'device-db-client' was not successfully "
                    "imported.")
            else:
                if qubit is None:
                    raise ValueError(
                        "When using the database, only parameters associated "
                        "with a qubit are allowed. Provide qubit as a keyword."
                    )
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
                group definitions. Default is ['Groups']

        Returns:
            A set of strings with the group names the qubit is part of.
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
            settings_default_folder (string): full path to the folder for
                default settings
            settings_setup_folder (string): full path to the folder for setup
                settings
            settings_sample_folder (string): full path to the folder for sample
                settings
            settings_default_folder (string): user settings as a dictionary
        """
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

        Args:
            settings_default_folder (string): full path to the folder for
                default settings
            settings_setup_folder (string): full path to the folder for setup
                settings
            settings_sample_folder (string): full path to the folder for sample
                settings
            settings_default_folder (string): user settings as a dictionary
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
            db_client_config (devicedb.Config): a configuration object for the
                device
        """
        if self.dev_name is not None:
            db_client_config.device_name = self.dev_name
        self.db_client = devicedb.Client(db_client_config)

    def copy(self, overwrite_dict=None):
        """Creates a deepcopy of the current dictionary.
        Returns the deepcopied dictionary.

        Args:
            overwrite_dict (dict): When this is ntt None, the content of the
                dictionary is cleared and replaced with the dict specified in
                overwrite_dict. Other arguments of the dictionary like the
                device database client stay the same.
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
        """Returns the step class for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self[index][0]

    def get_step_label_at_index(self, index):
        """Returns the step label for a specific step in the routine template.

        Args:
            index: index of the step for which the step label is to be returned.
        """
        return self[index][1]

    def get_step_settings_at_index(self, index):
        """Returns the settings for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        settings = {}
        settings.update(copy.copy(self.global_settings))
        settings.update(copy.copy(self[index][2]))
        return settings

    def get_step_tmp_vals_at_index(self, index):
        """Returns the temporary values of the step at index.
        """
        try:
            return self[index][3]
        except IndexError:
            return []

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """Extends the temporary values of the step at index. If the step does
        not have any temporary values, it sets the temporary values to the
        passed temporary values.
        """
        try:
            self[index][3].extend(tmp_vals)
        except IndexError:
            self[index].append(tmp_vals)

    def update_settings_at_index(self, settings, index):
        """Updates the settings of the step at index.
        """
        self[index][2].update(settings)

    def update_all_step_settings(self, settings):
        """Updates all settings of all steps in the routine.
        """
        for i, x in enumerate(self):
            self.update_settings_at_index(settings, index=i)

    def update_settings(self, settings_list):
        """Updates all settings of the routine. Settings_list must be a list of
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
        """Prints a user-friendly representation of the routine template

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
        """Returns the name of the step at index.
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
        """Adds a step to the routine.
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

        Arguments:
            i: index or slice

        Returns: element or new RoutineTemplate instance
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
            dev (Device obj): The device which is currently measured.
            routine (Step obj): The parent of the step. If this step is the root
                routine, this should be None.

        Keywords:
            step_label (str): A unique label for this step to be used in the
                configuration parameters files.
            settings (SettingsDictionary obj): The configuration parameters
                passed down from its parent. if None, the dictionary is taken
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

        # FIXME this is there to make the current one-qubit-only implementation
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

    def get_param_value(self,
                        param,
                        qubit=None,
                        sublookups=None,
                        default=None,
                        leaf=None,
                        associated_component_type_hint=None):
        """Looks up the requested parameter in the own configuration parameter
        dictionary. If no value was found, the parent's function is called
        recursively up to the root routine.

        Args:
            param (str): The name of the parameter to look up.
            sublookups (list of str): Optional subscopes to be looked up.
            default: The default value the parameters falls back to if no value
                was found for the parameter in the whole dictionary.
            leaf (boolean): True if the scope to search the parameter for is a
                leaf node (e.g. a measurement or intermediate step, not a
                routine)
            associated_component_type_hint (str): A hint for the device database,
                if the parameter does not belong to the qubit.

        Returns:
            The value found in the dictionary. If no value was found, either the
            default value is returned if specified or otherwhise None.
        """
        groups = None
        if qubit is not None:
            groups = self.get_qubit_groups(qubit)

        lookups = self.parameter_lookups
        if leaf is None:
            leaf = self.leaf
        val, success = self.settings.get_param_value(
            param,
            lookups,
            sublookups,
            qubit,
            groups,
            leaf=leaf,
            associated_component_type_hint=associated_component_type_hint)

        if not success:
            if self.routine is not None:
                sublookups = sublookups if sublookups else lookups
                val = self.routine.get_param_value(
                    param,
                    qubit,
                    sublookups=sublookups,
                    leaf=leaf,
                    associated_component_type_hint=associated_component_type_hint
                )
                success = True
            elif sublookups:
                val, success = self.settings.get_param_value(
                    param,
                    sublookups,
                    None,
                    qubit,
                    groups,
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
            A set of strings with the group names the qubit is part of.
        """

        # Possible problem: when a qubit is removed from a group with the same
        # name in higher hierarchy, it will still remain
        lookups = ['Groups']
        groups = self.settings.get_qubit_groups(qubit, lookups)
        if self.routine is not None:
            groups.update(self.routine.get_qubit_groups(qubit))
        return groups

    def run(self):
        # to be overwritten by subclasses
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
        step.

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
        initialization of the current step. This is important for experiments,
        as they have to be initialized with these keywords and cannot use the
        get_param_value function.

        Returns:
            A dictionary containing names and default values of keyword
            arguments which are needed for the current step.
        """
        gui_kwargs = self.__class__.gui_kwargs(self.dev)
        # remove second layer of dict
        requested_kwargs = {
            k: {ky: vy for kx, vx in v.items()
                for ky, vy in vx.items()}
            for k, v in gui_kwargs.items()
        }
        requested_kwargs['kwargs'].pop('sweep_points',
                                       None)  # pop kw that is not needed
        requested_kwargs['kwargs'].pop('cal_states', None)
        requested_kwargs['kwargs'].pop('n_cal_points_per_state', None)
        requested_kwargs['kwargs'].pop('cz_pulse_name', None)
        requested_kwargs['kwargs'].pop('compression_seg_lim', None)
        return requested_kwargs

    def parse_settings(self, requested_kwargs):
        """Resolves the keyword arguments from get_requested_settings to calls
        within the parameter dictionary.

        Arguments:
            requested_kwargs (dict): A dictionary containing the names and
            default values of the keyword arguments for an experiment.
        Returns:
            The exact keyword arguments to pass to the experiment class.
        """
        kwargs = {}
        for k, v in requested_kwargs['kwargs'].items():
            # Default value in kwargs, should the default value always be found
            # in settings?
            kwargs[k] = self.get_param_value(k, default=v[1])
        kwargs.pop('ro_qubits', None)
        # kwargs.pop('compression_seg_lim',None)
        # kwargs.pop('cz_pulse_name',None)
        kwargs['measure'] = False
        kwargs['analyze'] = False
        kwargs['qubits'] = self.qubits
        return kwargs

    @property
    def highest_lookup(self):
        """Returns the highest scope for this step which is not None
        in the order step_label, class name, General

        Returns:
            A string with the highest lookup of this step which is not None.
        """
        return self._get_first_not_none(self.parameter_lookups)

    @property
    def highest_sublookup(self):
        """Returns the highest subscope for this step which is not None.

        Returns:
            A string with the highest sublookup of this step which is not None.
            If the step is a leaf, this is None, otherwhise it is "General".
        """
        return None if self.parameter_sublookups is None else self._get_first_not_none(
            self.parameter_sublookups)

    def _get_first_not_none(self, lookup_list):
        return next((item for item in lookup_list if item is not None), None)

    @classmethod
    def get_lookup_class(cls):
        if issubclass(cls, SingleQubitGateCalibExperiment):
            return cls.__bases__[0]
        return cls


class IntermediateStep(Step):
    """Class used for defining intermediate steps between automatic calibration
    steps.
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
    """

    def __init__(
        self,
        dev,
        routine=None,
        autorun=True,
        **kw,
    ):
        """Initializes the routine.

        NOTE that currently only one qubit is supported. If a list of multiple
            qubits is passed, only the first qubit is used.

        Args:
            dev: device to be used for the routine
            autorun (bool): if True, the routine will be run immediately after
                initialization.
            routine (Step obj): The parent routine of the routine.

        Keywords:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter dictionary):
            update (bool): if True, the routine will overwrite qubit attributes
                with the values found in the routine. Note that if the routine
                is used as subroutine, this should be set to True.
            save_instrument_settings (bool): if True, the routine will save the
                instrument settings before and after the routine.
            verbose: if True, the routine will print out the progress of the
                routine. Default is True.

        """

        self.kw = kw
        self.autorun = autorun
        # call step constructor
        super().__init__(dev, routine, **kw)

        self.parameter_sublookups = ['General']
        self.leaf = False

        self.DCSources = self.kw.pop("DCSources", None)

        # MC - trying to get it from either the device or the qubits
        for source in [self.dev] + self.qubits:
            try:
                self.MC = source.instr_mc.get_instr()
                break
            except:
                pass

        self.create_initial_parameters()

    def merge_settings(self, lookups, sublookups=None):
        """Merges all subscopes relevant for a particular child step.

        Arguements:
            lookups (list): A list of all subscopes of the routine relevant for
                the child step.
            sublookups (list): A list of subscopes of the lookups (relevant if
                the child step is a routine)

        """
        settings = {}
        for lookup in reversed(lookups):
            if lookup in self.settings:
                if sublookups is not None:
                    for sublookup in reversed(sublookups):
                        if sublookup in self.settings[lookup]:
                            update_nested_dictionary(
                                settings, self.settings[lookup][sublookup])
                else:
                    update_nested_dictionary(settings, self.settings[lookup])

        return settings

    def create_routine_template(self):
        """Creates routine template. Can be overwritten or extended by children
        for more complex routines that require adaptive creation.
        """
        # create RoutineTemplate based on _default_routine_template
        self.routine_template = copy.deepcopy(self._DEFAULT_ROUTINE_TEMPLATE)
        self.routine_template.routine = self
        self.routine_template.settings = self.settings

        for step in self.routine_template:
            # Convert basic experiments into Autoroutine Steps
            if not issubclass(step[0], Step):
                raise NotImplementedError(
                    "Steps have to inherit from class Step.")
                # class TmpStep(Step, step[0]):
                #    pass
                #step[0] = TmpStep

            # no 'General' lookup since in 'General' should only be raw
            # parameters and no step descriptions for children
            lookups = [self.step_label, self.get_lookup_class().__name__]
            # no 'General' lookup
            sublookups = [step[1], step[0].get_lookup_class().__name__]

            autocalib_settings = self.settings.copy({
                step[0].get_lookup_class().__name__:
                    self.merge_settings(lookups, sublookups)
            })
            update_nested_dictionary(autocalib_settings,
                                     step[2].get('settings', {}))
            step[2]['settings'] = autocalib_settings

        # standard global settings
        delegate_plotting = self.get_param_value('delegate_plotting')

        self.routine_template.global_settings.update({
            "dev": self.dev,
            # "qubits": self.qubits,
            "update": True,  # all subroutines should update relevant params
            "delegate_plotting": delegate_plotting,
        })

        # add user specified global settings
        update_nested_dictionary(
            self.routine_template.global_settings,
            self.kw.get("global_settings", {}),
        )

    def prepare_step(self, i=None):
        """Prepares the next step in the routine. That is, it initializes the
        measurement object.

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

        settings = self.current_step_settings

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
        if (self.get_param_value('save_instrument_settings') or
                not self.get_param_value("update")):
            # saving instrument settings before the routine
            self.MC.create_instrument_settings_file(
                f"pre-{self.name}_routine-settings")
            self.preroutine_timestamp = a_tools.get_last_n_timestamps(1,)[0]
        else:
            # registering start of routine so all data in measurement period can
            # be retrieved later to determine the Hamiltonian model
            self.preroutine_timestamp = self.MC.get_datetimestamp()

        # rerun routine if already finished
        if (len(self.routine_template) != 0) and (self.current_step_index >=
                                                  len(self.routine_template)):
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
                if self.get_param_value('verbose'):
                    print(f"Partial routine {routine_name} stopped before "
                          f"executing step {j} ({step_name}).")
                return

            # executing the step
            with temporary_value(*self.current_step_tmp_vals):
                self.execute_step()

            if self.get_param_value('verbose'):
                print(f"{routine_name}, step {j} ({step_name}), done!", "\n")

        if self.get_param_value('verbose'):
            print(f"Routine {routine_name} finished!")

        # saving instrument settings post-routine
        if (self.get_param_value('save_instrument_settings') or
                not self.get_param_value("update")):
            # saving instrument settings after the routine
            self.MC.create_instrument_settings_file(
                f"post-{routine_name}_routine-settings")

        # reloading instrument settings if update is False
        if not self.get_param_value("update"):
            if self.get_param_value('verbose'):
                print(f"Reloading instrument settings from before routine "
                      f"(ts {self.preroutine_timestamp})")

            reload_settings(
                self.preroutine_timestamp,
                qubits=self.qubits,
                dev=self.dev,
                DCSources=self.DCSources,
            )

    def create_initial_parameters(self):
        """Creates the initial parameters for the routine.

        Keyword arguments are accessed through self. Relevant kwargs stored in
        self are:
            get_parameters_from_qubit_object: if True, initial guesses and
                estimates for the transmon parameters are retrieved from the
                (transmon) qubit object.
            parameters_device: nested dictionary containing the device settings
                folder. If non-existent, the device folder is loaded (second
                highest priority).
            parameters_user: nested dictionary containing relevant parameters
                from the user (highest priority).

        FIXME: In the future, we might want to swap the hierarchy and give
        priority to the settings in the qubit object, but that would require
        implementing plausibility checks and falling back to the config file if
        there is no reasonable value set in the qubit object.
        """
        self.process_kw()

    def create_initial_routine(self, load_parameters=True):
        """Creates (or recreates) initial routine by defining the routine
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

    def final_init(self, **kwargs):
        """A function to be called after initialization of all base classes,
        since some functionality in the init of a routine needs the base
        classes already initialized.

        """
        # loading hierarchical settings and creating initial routine
        # if not isinstance(self,AutomaticCalibrationRoutine)
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

        settings = {}

        hamfit_model = qb.fit_ge_freq_from_dc_offset()

        # extracting settings from the qubit
        settings.update({
            "fr": hamfit_model.get("fr", qb.ro_freq()),
            "anharmonicity": qb.anharmonicity(),
        })

        # getting transmon settings from present Hamiltonian model if it exists
        settings.update(hamfit_model)

        return settings

    @property
    def parameters_init(self):
        settings = copy.copy(self.kw)
        return settings

    def process_kw(self):
        update_nested_dictionary(self.settings, {
            self.highest_lookup: {
                self.highest_sublookup: self.parameters_init
            }
        })

    def view(self, **kw):
        """Prints a user friendly representation of the routine settings
        """
        self.routine_template.view(**kw)

    def update_settings_at_index(self, settings: dict, index):
        """Updates the calibration settings for a specific step in the routine.

        Args:
            settings: dictionary of settings to be updated
            index: index of the step settings to be updated.
        """
        self.routine_template.update_settings_at_index(settings, index)

    def get_step_class_at_index(self, index):
        """Returns the step class for a specific step in the routine template.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_class_at_index(index)

    def get_step_label_at_index(self, index):
        """Returns the step label for a specific step in the routine.

        Args:
            index: index of the step for which the step label is to be returned.
        """
        return self.routine_template.get_step_label_at_index(index)

    def get_step_settings_at_index(self, index):
        """Returns the settings for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_settings_at_index(index)

    def get_step_tmp_vals_at_index(self, index):
        """Returns the temporary values for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        return self.routine_template.get_step_tmp_vals_at_index(index)

    def extend_step_tmp_vals_at_index(self, tmp_vals, index):
        """Sets the temporary values for a specific step in the routine.

        Args:
            index: index of the step for which the settings are to be returned.
        """
        self.routine_template.extend_step_tmp_vals_at_index(tmp_vals=tmp_vals,
                                                            index=index)

    def add_step(self,
                 step_class,
                 step_label,
                 step_settings,
                 step_tmp_vals=None,
                 index=None):
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

    # initializing necessary attributes, should/can be overridden by children
    _DEFAULT_PARAMETERS = {}
    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([])


# FIXME these wrappers can probably be collected with an ExperimentStep class to
# avoid repetition
class RabiStep(qbcal.Rabi, Step):
    """A wrapper class for the Rabi experiment.
    """

    def __init__(self, routine, **kwargs):
        """
        Initializes the RabiStep class, which also includes initialization
        of the Rabi experiment.

        Arguments:
            routine (Step obj): The parent routine

        Keywords:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): a list of all groups of qubits on which the
                Rabi measurement can be conducted in parallel
            v_low: minimum voltage sweep range for the Rabi experiment. Can
                either be a float or an expression to be evaluated. The
                expression can contain the following values:
                    current - the current π pulse amplitude of the qubit
                    default - the default π pulse amplitude specified in the
                        configuration parameters, e.g. default ge amp180 for the
                        |g⟩ ↔ |e⟩ transition
                    max - the maximum drive amplitude specified in the
                        configuration parameters as max drive amp
                    n - the keyword for the number of π pulses applied during
                        the Rabi experiment, specified in the configuration
                        parameters
                Example:
                    "v_high": "min(({n} + 0.45) * {current} / {n}, {max})"
            v_high: maximum voltage sweep range for the Rabi experiment. Can
                either be a float or an expression to be evaluated. See above.
            pts: Number of points for the sweep voltage. Can either be a float
                or an expression to be evaluated. See above.
            max_drive_amp (float): The maximum drive amplitude.
            default_<transition>_amp180: A default value for the pi pulse to be
                set back to.
            clip_drive_amp (bool): If True, and the determined pi pulse amp is
                higher than max_drive_amp, it is reset to
                default_<transition>_amp180
        """
        self.kw = kwargs
        Step.__init__(self, routine=routine, **kwargs)
        rabi_settings = self.parse_settings(self.get_requested_settings())
        qbcal.Rabi.__init__(self, dev=self.dev, **rabi_settings)

    def parse_settings(self, requested_kwargs):
        """Searches the keywords for the Rabi experiment given in
        requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
            keywords needed for the Rabi class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the Rabi class
        """
        kwargs = {}
        task_list = []
        for qb in self.qubits:
            task = {}
            task_list_fields = requested_kwargs['task_list_fields']

            transition_name_v = task_list_fields.get('transition_name')
            tr_name = self.get_param_value('transition_name',
                                           qubit=qb.name,
                                           default=transition_name_v[1])
            task['transition_name'] = tr_name

            value_params = {
                'v_low': None,
                'v_high': None,
                'pts': None
            }  # the information about these custom parametes could be
            # saved somewhere else to generalize all wrappers

            default = self.get_param_value(f'default_{tr_name}_amp180',
                                           qubit=qb.name)
            current = qb.parameters[f'{tr_name}_amp180']()
            max = self.get_param_value('max_drive_amp', qubit=qb.name)
            n = self.get_param_value('n', qubit=qb.name)

            for name, value in value_params.items():
                value = self.get_param_value(name, qubit=qb.name)
                if isinstance(value, str):
                    value = eval(
                        value.format(current=current,
                                     max=max,
                                     default=default,
                                     n=n))
                value_params[name] = value

            sweep_points_v = task_list_fields.get('sweep_points', None)
            if sweep_points_v is not None:
                # get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(value_params['v_low'],
                                     value_params['v_high'],
                                     value_params['pts'])
                task['sweep_points'] = SweepPoints()
                task['sweep_points'].add_sweep_parameter(values=values,
                                                         **sweep_points_kws)
            qb_v = task_list_fields.get('qb', None)
            if qb_v is not None:
                task['qb'] = qb.name

            for k, v in task_list_fields.items():
                if k not in task:
                    task[k] = self.get_param_value(k,
                                                   qubit=qb.name,
                                                   default=v[1])

            task_list.append(task)

        kwargs['task_list'] = task_list

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def run(self):
        """Runs the Rabi experiment, the analysis for it and additional
        postprocessing.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()

        if self.get_param_value('clip_drive_amp'):
            for qb in self.qubits:
                tr_name = self.get_param_value('transition_name', qubit=qb.name)
                max_drive_amp = self.get_param_value('max_drive_amp',
                                                     qubit=qb.name)
                if tr_name == 'ge' and qb.ge_amp180() > max_drive_amp:
                    qb.ge_amp180(
                        self.get_param_value('default_ge_amp180',
                                             qubit=qb.name))
                elif tr_name == 'ef' and qb.ef_amp180() > max_drive_amp:
                    qb.ef_amp180(
                        self.get_param_value('default_ef_amp180',
                                             qubit=qb.name))


class RamseyStep(qbcal.Ramsey, Step):

    def __init__(self, routine, *args, **kwargs):
        """Initializes the RamseyStep class, which also includes initialization
        of the Ramsey experiment.

        Arguments:
            routine (Step obj): The parent routine

        Keywords:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): a list of all groups of qubits on which the
                Ramsey measurement can be conducted in parallel
            t0: Minimum delay time for the Ramsey experiment.
            delta_t: Duration of the delay time for the Ramsey experiment.
            n_periods (int): Number of expected oscillation periods in the delay
                time given with t0 and delta_t.
            pts_per_period (int): Number of points per period of oscillation.
                The total points for the sweep range are n_periods*pts_per_period+1,
                the artificial detuning is n_periods/delta_t.
            configure_mux_drive (bool): If the LO frequencies and IFs should of
                the qubits measured in this step should be updated afterwards.
        """
        self.kw = kwargs
        Step.__init__(self, routine=routine, *args, **kwargs)
        ramsey_settings = self.parse_settings(self.get_requested_settings())
        qbcal.Ramsey.__init__(self, dev=self.dev, **ramsey_settings)

    def parse_settings(self, requested_kwargs):
        """
        Searches the keywords for the Ramsey experiment given in
        requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
                keywords needed for the Ramsey class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the Ramsey class
        """
        kwargs = {}
        task_list = []
        for qb in self.qubits:
            task = {}
            task_list_fields = requested_kwargs['task_list_fields']

            value_params = {
                'delta_t': None,
                't0': None,
                'n_periods': None,
                'pts_per_period': None
            }
            for name, value in value_params.items():
                value = self.get_param_value(name, qubit=qb.name)
                value_params[name] = value

            sweep_points_v = task_list_fields.get('sweep_points', None)
            if sweep_points_v is not None:
                # get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(
                    value_params['t0'],
                    value_params['t0'] + value_params['delta_t'],
                    value_params['pts_per_period'] * value_params['n_periods'] +
                    1)
                task['sweep_points'] = SweepPoints()
                task['sweep_points'].add_sweep_parameter(values=values,
                                                         **sweep_points_kws)

            ad_v = task_list_fields.get('artificial_detuning', None)
            if ad_v is not None:
                task['artificial_detuning'] = value_params['n_periods'] / \
                    value_params['delta_t']
            qb_v = task_list_fields.get('qb', None)
            if qb_v is not None:
                task['qb'] = qb.name

            for k, v in task_list_fields.items():
                if k not in task:
                    task[k] = self.get_param_value(k,
                                                   qubit=qb.name,
                                                   default=v[1])

            task_list.append(task)

        kwargs['task_list'] = task_list

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def run(self):
        """Runs the Ramsey experiment, the analysis for it and additional
        postprocessing.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()
            self.dev.update_cancellation_params()

        if self.get_param_value('configure_mux_drive'):
            drive_lo_freqs = self.get_param_value('drive_lo_freqs')
            configure_qubit_mux_drive(self.qubits, drive_lo_freqs)

    def get_device_property_values(self, **kwargs):
        """Returns a dictionary of high-level device property values from
        running this RamseyStep

        Args:
            qubit_sweet_spots (dict, optional): a dictionary mapping qubits to
                sweet-spots ('uss', 'lss', or None)

        Returns:
            results (dict, optional): dictionary of high-level results (may be
                empty)
        """

        results = self.get_empty_device_properties_dict()
        sweet_spots = kwargs.get('qubit_sweet_spots', {})
        if _device_db_client_module_missing:
            log.warning("Assemblying the dictionary of high-level device "
            "property values requires the module 'device-db-client', which was "
            "not imported successfully.")
        elif self.analysis:
            # Get the analysis parameters dictionary
            analysis_params_dict = self.analysis.proc_data_dict[
                'analysis_params_dict']
            # For RamseyStep, the keys in `analysis_params_dict` are qubit names
            for qubit_name, qubit_results in analysis_params_dict.items():
                # This transition is not stored in RamseyAnalysis, so we must
                # get it from the settings parameters
                transition = self.get_param_value('transition_name',
                                                  qubit=qubit_name)
                node_creator = db_utils.ValueNodeCreator(
                    qubits=qubit_name,
                    timestamp=self.analysis.timestamps[0],
                    sweet_spots=sweet_spots.get(qubit_name),
                    transition=transition,
                )
                # T2 Star Time for the exponential decay
                if 'exp_decay' in qubit_results.keys(
                ) and 'T2_star' in qubit_results['exp_decay'].keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='T2_star',
                            value=qubit_results['exp_decay']['T2_star']))

                # Updated qubit frequency
                if 'exp_decay' in qubit_results.keys(
                ) and f"new_{transition}_freq" in qubit_results[
                        'exp_decay'].keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='freq',
                            value=qubit_results['exp_decay']
                            ['new_{transition}_freq']))

                if 'T2_echo' in qubit_results.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='T2_echo',
                            value=qubit_results['T2_echo']))
        return results


class ReparkingRamseyStep(qbcal.ReparkingRamsey, Step):

    def __init__(self, routine, *args, **kwargs):
        """Initializes the ReparkingRamseyStep class, which also includes
        initialization of the ReparkingRamsey experiment.

        Arguments:
            routine (Step obj): The parent routine

        Keywords:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): a list of all groups of qubits on which the
                ReparkingRamsey experiment can be conducted in parallel
            t0: Minimum delay time for the ReparkingRamsey experiment.
            delta_t: Duration of the delay time for the ReparkingRamsey
                experiment.
            n_periods (int): Number of expected oscillation periods in the delay
                time given with t0 and delta_t.
            pts_per_period (int): Number of points per period of oscillation.
                The total points for the sweep range are n_periods*pts_per_period+1,
                the artificial detuning is n_periods/delta_t.
    """
        self.kw = kwargs
        Step.__init__(self, routine=routine, *args, **kwargs)
        settings = self.parse_settings(self.get_requested_settings())
        qbcal.ReparkingRamsey.__init__(self, dev=self.dev, **settings)

    def parse_settings(self, requested_kwargs):
        """Searches the keywords for the ReparkingRamsey experiment given in
        requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
            keywords needed for the ReparkingRamsey class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
            the ReparkingRamsey class
        """
        kwargs = {}
        task_list = []
        for qb in self.qubits:
            task = {}
            task_list_fields = requested_kwargs['task_list_fields']

            # FIXME can this be combined with RamseyStep to avoid code
            # replication?
            value_params = {
                'delta_t': None,
                't0': None,
                'n_periods': None,
                'pts_per_period': None,
                'dc_voltage_offsets': []
            }
            for name, value in value_params.items():
                value = self.get_param_value(name, qubit=qb.name)
                value_params[name] = value
            dc_voltage_offsets = value_params['dc_voltage_offsets']
            if isinstance(dc_voltage_offsets, dict):
                dc_voltage_offsets = np.linspace(dc_voltage_offsets['low'],
                                                 dc_voltage_offsets['high'],
                                                 dc_voltage_offsets['pts'])
            task['dc_voltage_offsets'] = dc_voltage_offsets

            sweep_points_v = task_list_fields.get('sweep_points', None)
            if sweep_points_v is not None:
                # get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(
                    value_params['t0'],
                    value_params['t0'] + value_params['delta_t'],
                    value_params['pts_per_period'] * value_params['n_periods'] +
                    1)
                task['sweep_points'] = SweepPoints()
                task['sweep_points'].add_sweep_parameter(values=values,
                                                         **sweep_points_kws)

            ad_v = task_list_fields.get('artificial_detuning', None)
            if ad_v is not None:
                task['artificial_detuning'] = value_params['n_periods'] / \
                    value_params['delta_t']
            qb_v = task_list_fields.get('qb', None)
            if qb_v is not None:
                task['qb'] = qb.name
                task['fluxline'] = self.get_param_value('fluxlines_dict')[
                    qb.name]

            for k, v in task_list_fields.items():
                if k not in task:
                    task[k] = self.get_param_value(k,
                                                   qubit=qb.name,
                                                   default=v[1])

            task_list.append(task)

        kwargs['task_list'] = task_list

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def run(self):
        """Runs the Ramsey experiment and the analysis for it.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()


class T1Step(qbcal.T1, Step):

    def __init__(self, routine, *args, **kwargs):
        """Initializes the T1Step class, which also includes initialization
        of the T1 experiment.

        Arguments:
            routine (Step obj): The parent routine

        Keywords:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): a list of all groups of qubits on which the
                T1 experiment can be conducted in parallel
            t0: Minimum delay time for the T1 experiment.
            delta_t: Duration of the delay time for the T1 experiment.
            pts (int): Number of points for the sweep range of the delay time.
        """
        self.kw = kwargs
        Step.__init__(self, routine=routine, *args, **kwargs)
        t1_settings = self.parse_settings(self.get_requested_settings())
        qbcal.T1.__init__(self, dev=self.dev, **t1_settings)

    def parse_settings(self, requested_kwargs):
        """
        Searches the keywords for the T1 experiment given in requested_kwargs
        in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
            keywords needed for the T1 class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the T1 class
        """
        kwargs = {}
        task_list = []
        for qb in self.qubits:
            task = {}
            task_list_fields = requested_kwargs['task_list_fields']

            value_params = {'t0': None, 'delta_t': None, 'pts': None}

            for name, value in value_params.items():
                value = self.get_param_value(name, qubit=qb.name)
                value_params[name] = value

            sweep_points_v = task_list_fields.get('sweep_points', None)
            if sweep_points_v is not None:
                # get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(
                    value_params['t0'],
                    value_params['t0'] + value_params['delta_t'],
                    value_params['pts'])
                task['sweep_points'] = SweepPoints()
                task['sweep_points'].add_sweep_parameter(values=values,
                                                         **sweep_points_kws)

            qb_v = task_list_fields.get('qb', None)
            if qb_v is not None:
                task['qb'] = qb.name

            for k, v in task_list_fields.items():
                if k not in task:
                    task[k] = self.get_param_value(k,
                                                   qubit=qb.name,
                                                   default=v[1])

            task_list.append(task)

        kwargs['task_list'] = task_list

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def run(self):
        """Runs the T1 experiment and the analysis for it.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()

    def get_device_property_values(self, **kwargs):
        """Returns a dictionary of high-level device property values from
        running this T1Step.

        Args:
            qubit_sweet_spots (dict, optional): a dictionary mapping qubits to
                sweet-spots ('uss', 'lss', or None)

        Returns:
            dict: dictionary of high-level device property values
        """
        results = self.get_empty_device_properties_dict()
        sweet_spots = kwargs.get('qubit_sweet_spots', {})
        if _device_db_client_module_missing:
            log.warning("Assemblying the dictionary of high-level device "
            "property values requires the module 'device-db-client', which was "
            "not imported successfully.")
        elif self.analysis:
            analysis_params_dict = self.analysis.proc_data_dict[
                'analysis_params_dict']

            # For T1Step, the keys in `analysis_params_dict` are qubit names
            for qubit_name, qubit_results in analysis_params_dict.items():
                transition = self.get_param_value('transition_name',
                                                  qubit=qubit_name)
                node_creator = db_utils.ValueNodeCreator(
                    qubits=qubit_name,
                    timestamp=self.analysis.timestamps[0],
                    sweet_spots=sweet_spots.get(qubit_name),
                    transition=transition,
                )
                results['property_values'].append(
                    node_creator.create_node(property_type='T1',
                                             value=qubit_results['T1']))

        return results


class QScaleStep(qbcal.QScale, Step):
    """A wrapper class for the QScale experiment.

    Arguments:
        routine (Step obj): The parent routine

    Keywords:
        qubits (list): list of qubits to be used in the routine

    Configuration parameters (coming from the configuration parameter dictionary):
        transition_name (str): The transition of the experiment
        parallel_groups (list): a list of all groups of qubits on which the T1
            experiment can be conducted in parallel
        v_low: Minimum of the sweep range for the qscale parameter.
        v_high:  Maximum of the sweep range for the qscale parameter.
        pts (int): Number of points for the sweep range.
        configure_mux_drive (bool): If the LO frequencies and IFs of the
            qubits measured in this step should be updated afterwards.
        """

    def __init__(self, routine, *args, **kwargs):
        self.kw = kwargs
        Step.__init__(self, routine=routine, *args, **kwargs)
        qscale_settings = self.parse_settings(self.get_requested_settings())
        qbcal.QScale.__init__(self, dev=self.dev, **qscale_settings)

    def parse_settings(self, requested_kwargs):
        """Searches the keywords for the QScale experiment given in
        requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
                keywords needed for the QScale class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the QScale class
        """
        kwargs = {}
        task_list = []
        for qb in self.qubits:
            task = {}
            task_list_fields = requested_kwargs['task_list_fields']

            value_params = {'v_low': None, 'v_high': None, 'pts': None}

            for name, value in value_params.items():
                value = self.get_param_value(name, qubit=qb.name)
                value_params[name] = value

            sweep_points_v = task_list_fields.get('sweep_points', None)
            if sweep_points_v is not None:
                # get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(value_params['v_low'],
                                     value_params['v_high'],
                                     value_params['pts'])
                task['sweep_points'] = SweepPoints()
                # FIXME:why is values_func an invalid paramteter, if it is in
                # kw_for_sweep_points?
                sweep_points_kws.pop('values_func', None)
                task['sweep_points'].add_sweep_parameter(values=values,
                                                         **sweep_points_kws)

            qb_v = task_list_fields.get('qb', None)
            if qb_v is not None:
                task['qb'] = qb.name

            for k, v in task_list_fields.items():
                if k not in task:
                    task[k] = self.get_param_value(k,
                                                   qubit=qb.name,
                                                   default=v[1])

            task_list.append(task)

        kwargs['task_list'] = task_list

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def run(self):
        """Runs the QScale experiment, the analysis for it and some
        postprocessing.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()
            self.dev.update_cancellation_params()

        if self.get_param_value('configure_mux_drive'):
            drive_lo_freqs = self.get_param_value('drive_lo_freqs')
            configure_qubit_mux_drive(self.qubits, drive_lo_freqs)


class InPhaseAmpCalibStep(qbcal.InPhaseAmpCalib, Step):

    def __init__(self, routine, *args, **kwargs):
        self.kw = kwargs
        Step.__init__(self, routine=routine, *args, **kwargs)
        ip_calib_settings = self.parse_settings(self.get_requested_settings())
        qbcal.InPhaseAmpCalib.__init__(self, dev=self.dev, **ip_calib_settings)

    def parse_settings(self, requested_kwargs):
        """Searches the keywords for the InPhaseAmpCalib experiment given in
        requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names of the
            keywords needed for the InPhaseAmpCalib class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
            the InPhaseAmpCalib class
        """
        kwargs = {}

        kwargs_super = super().parse_settings(requested_kwargs)
        kwargs_super.update(kwargs)

        return kwargs_super

    def get_requested_settings(self):
        """Add additional keywords to be passed to the InPhaseAmpCalib class.

        Returns:
            dict: Dictionary containing names  and default values
                of the keyword arguments to be passed to the InPhaseAmpCalib
                class.
        """
        settings = super().get_requested_settings()
        settings['kwargs']['n_pulses'] = (int, 100)
        return settings

    def run(self):
        """Runs the InPhaseAmpCalib experiment and the analysis for it.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()


# Special Automatic calibration routines


class PiPulseCalibration(AutomaticCalibrationRoutine):
    """Pi-pulse calibration consisting of a Rabi experiment followed by a Ramsey
    experiment.
    """

    def __init__(
        self,
        dev,
        **kw,
    ):
        """Pi-pulse calibration routine consisting of one Rabi and one Ramsey.

        Args:
            dev (Device obj): the device which is currently measured.

        Keyword Arguments:
            qubits: qubits on which to perform the measurement.

        """

        # arguments that are not arguments of the super init will be considered
        # as key words for the super init.
        super().__init__(
            dev=dev,
            **kw,
        )
        self.final_init(**kw)

    def create_routine_template(self):
        """Creates routine template.
        """
        super().create_routine_template()

        detailed_routine_template = copy.copy(self.routine_template)
        detailed_routine_template.clear()

        for step in self.routine_template:
            step_class = step[0]
            settings = copy.deepcopy(step[2])
            label = step[1]

            lookups = [label, step_class.get_lookup_class().__name__, 'General']
            for parallel_group in self.get_param_value('parallel_groups',
                                                       sublookups=lookups,
                                                       leaf=True):
                qubits_filtered = [
                    qb for qb in self.qubits if qb.name is parallel_group or
                    parallel_group in self.get_qubit_groups(qb.name)
                ]
                if len(qubits_filtered) != 0:
                    temp_settings = copy.deepcopy(settings)
                    temp_settings['qubits'] = qubits_filtered
                    qubit_label = label + "_"
                    for qb in qubits_filtered:
                        qubit_label += qb.name

                    detailed_routine_template.add_step(step_class, qubit_label,
                                                       temp_settings)

        self.routine_template = detailed_routine_template

    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
        [RabiStep, 'rabi', {}],
        [RamseyStep, 'ramsey', {}],
    ])


class FindFrequency(AutomaticCalibrationRoutine):
    """Routine to find frequency of a given transmon transition.
    """

    def __init__(
        self,
        dev,
        qubits,
        **kw,
    ):
        """Routine to find frequency of a given transmon transition.

        Args:
            dev (Device obj): the device which is currently measured
            qubits: list of qubits to be calibrated

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


        Configuration parameters (coming from the configuration parameter dictionary):
            transition_name: transition to be calibrated
            adaptive: whether to use adaptive rabi and ramsey settings
            allowed_difference: allowed frequency difference in Hz between
                old and new frequency (convergence criterion)
            max_iterations: maximum number of iterations
            autorun: whether to run the routine automatically
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
            dev=dev,
            qubits=qubits,
            **kw,
        )
        if len(qubits) > 1:
            raise ValueError("Currently only one qubit is allowed.")

        # defining initial and allowed frequency difference
        self.delta_f = np.Infinity
        self.iteration = 1

        self.final_init(**kw)

    @property
    def parameters_init(self):
        # FIXME the current implementation uses different parameters

        # writing keyword arguments into correct categroty of parameters dict
        # for x in ["rabi_amps", "ramsey_delays", "artificial_detuning"]:
        #    self.settings["PiPulseCalibration"][x] = self.kw.pop(
        #        x, self.settings["PiPulseCalibration"][x]
        #    )
        parameters = super().parameters_init
        return parameters

    class Decision(IntermediateStep):

        def __init__(self, routine, index, **kw):
            """Decision step that decides to add another round of Rabi-Ramsey to
            the FindFrequency routine based on the difference between the
            results of the previous and current Ramsey experiments.
            Additionally, it checks if the maximum number of iterations has been
            reached.

            Args:
                routine: FindFrequency routine
                index: index of the decision step (necessary to find the
                    position of the Ramsey measurement in the routine)

           Configuration parameters (coming from the configuration parameter dictionary):
                max_waiting_seconds: maximum number of seconds to wait for the
                    results of the previous Ramsey experiment to arrive.
            """
            super().__init__(routine=routine, index=index, **kw)
            # FIXME use general parameters from FindFrequency for now
            self.parameter_lookups = self.routine.parameter_lookups
            self.parameter_sublookups = self.routine.parameter_sublookups
            self.leaf = self.routine.leaf

        def run(self):
            """Executes the decision step.
            """
            qubit = self.qubit

            routine = self.routine
            index = self.kw.get("index")

            # saving some typing for parameters that are only read ;)
            allowed_delta_f = self.get_param_value("allowed_delta_f")
            f_unit = self.get_param_value("f_unit")
            f_factor = self.get_param_value("f_factor")
            delta_f_unit = self.get_param_value("delta_f_unit")
            delta_f_factor = self.get_param_value("delta_f_factor")
            max_iterations = self.get_param_value("max_iterations")
            transition = self.get_param_value("transition_name")

            # finding the ramsey experiment in the pipulse calibration
            pipulse_calib = routine.routine_steps[index - 1]
            ramsey = pipulse_calib.routine_steps[-1]

            # transition frequency from last Ramsey
            freq = qubit[f"{transition}_freq"]()

            # retrieving the frequency difference
            max_waiting_seconds = self.get_param_value("max_waiting_seconds")
            for i in range(max_waiting_seconds):
                try:
                    routine.delta_f = (
                        ramsey.analysis.proc_data_dict["analysis_params_dict"][
                            qubit.name]["exp_decay"]["new_qb_freq"] -
                        ramsey.analysis.proc_data_dict["analysis_params_dict"][
                            qubit.name]["exp_decay"]["old_qb_freq"])
                    break
                except KeyError:
                    log.warning(
                        "Could not find frequency difference between current "
                        "and last Ramsey measurement, delta_f not updated")
                    break
                except AttributeError:
                    # FIXME Unsure if this can also happen on real set-up
                    log.warning(
                        "Analysis not yet run on last Ramsey measurement, "
                        "frequency difference not updated")
                    time.sleep(1)

            # progress update
            if self.get_param_value('verbose'):
                print(f"Iteration {routine.iteration}, {transition}-freq "
                      f"{freq/f_factor} {f_unit}, frequency "
                      f"difference = {routine.delta_f/delta_f_factor} "
                      f"{delta_f_unit}")

            # check if the absolute frequency difference is small enough
            if np.abs(routine.delta_f) < allowed_delta_f:
                # success
                if self.get_param_value('verbose'):
                    print(f"{transition}-frequency found to be"
                          f"{freq/f_factor} {f_unit} within "
                          f"{allowed_delta_f/delta_f_factor} "
                          f"{delta_f_unit} of previous value.")

            elif routine.iteration < max_iterations:
                # no success yet, adding a new rabi-ramsey and decision step
                if self.get_param_value('verbose'):
                    print(f"Allowed error ("
                          f"{allowed_delta_f/delta_f_factor} "
                          f"{delta_f_unit}) not yet achieved, adding new"
                          " round of PiPulse calibration...")

                routine.add_next_pipulse_step(index=index + 1)

                step_settings = {'index': index + 2, 'qubits': self.qubits}
                routine.add_step(
                    FindFrequency.Decision,
                    'decision',
                    step_settings,
                    index=index + 2,
                )

                routine.iteration += 1
                return

            else:
                # no success yet, passed max iterations
                msg = (f"FindFrequency routine finished for {qubit.name}, "
                       "desired precision not necessarily achieved within the "
                       f"maximum number of iterations ({max_iterations}).")
                log.warning(msg)

                if self.get_param_value('verbose'):
                    print(msg)

            if self.get_param_value('verbose'):
                # printing termination update
                print(f"FindFrequency routine finished: "
                      f"{transition}-frequencies for {qubit.name} "
                      f"is {freq/f_factor} {f_unit}.")

    def create_routine_template(self):
        """Creates the routine template for the FindFrequency routine.
        """
        super().create_routine_template()

        pipulse_settings = {'qubits': self.qubits}
        self.add_step(PiPulseCalibration, 'pi_pulse_calibration',
                      pipulse_settings)

        # Decision step
        decision_settings = {"index": 1}
        self.add_step(self.Decision, 'decision', decision_settings)

    def add_next_pipulse_step(self, index):
        """Adds a next pipulse step at the specified index in the FindFrequency
        routine.
        """
        qubit = self.qubit

        adaptive = self.get_param_value('adaptive')
        transition = self.get_param_value('transition_name')

        #rabi_amps = self.settings["PiPulseCalibration"].get("rabi_amps", None)
        # ramsey_delays = self.settings["PiPulseCalibration"].get(
        #    "ramsey_delays", None
        # )
        # artificial_detuning = self.settings["PiPulseCalibration"].get(
        #    "artificial_detuning", None
        # )

        # if not adaptive:
        #    if ramsey_delays is None or artificial_detuning is None:
        #        raise ValueError(
        #            "If adaptive is False, rabi_amps, ramsey_delays"
        #            "and artificial_detuning must be specified as key words"
        #        )

        settings = self.settings.copy({})

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

            # This has to be solved differently now
            # Amplitudes for Rabi
            # 1) if passed in init
            # 2) v_high based on current pi-pulse amplitude
            # 3) v_high based on default value
            # if rabi_amps is None:
            if amp180:
                settings['Rabi']['v_max'] = amp180

            # Delays and artificial detuning for Ramsey
            # if ramsey_delays is None or artificial_detuning is None:
            # defining delta_t for Ramsey
            # 1) if passed in init
            # 2) based on T2_star
            # 3) based on default
            if self.get_param_value("use_T2_star"):
                settings['Ramsey']['delta_t'] = T2_star

        self.add_step(
            *[
                PiPulseCalibration,
                'pi_pulse_calibration_' + str(index),
                {
                    'settings': settings
                },
            ],
            index=index,
        )


class HamiltonianFitting(AutomaticCalibrationRoutine,
                         hfa.HamiltonianFittingAnalysis):

    def __init__(
        self,
        dev,
        qubit,
        fluxlines_dict,
        **kw,
    ):
        """Constructs a HamiltonianFitting routine used to determine a
        Hamiltonian model for a transmon qubit.

        Args:
            qubit: qubit to perform the calibration on. Note! Only supports
                one qubit input.
            fluxlines_dict: fluxlines_dict object for accessing and changing
                the dac voltages (needed in reparking ramsey)
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

        Configuration parameters (coming from the configuration parameter dictionary):
            use_prior_model: whether to use the prior model (stored in qubit
                object) to determine the guess frequencies and voltages for the
                flux values of the measurement. If True,
                flux_to_voltage_and_freq_guess will automatically be generated
                using this model (and can be left None).
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

            get_parameters_from_qubit_object: if True, the routine will try to
                get the parameters from the qubit object. Default is False.

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

        super().__init__(
            dev=dev,
            qubits=[qubit],
            fluxlines_dict=fluxlines_dict,
            **kw,
        )

        use_prior_model = self.get_param_value('use_prior_model')
        measurements = self.get_param_value('measurements')
        flux_to_voltage_and_freq_guess = self.get_param_value(
            'flux_to_voltage_and_freq_guess')

        if not use_prior_model:
            assert (
                flux_to_voltage_and_freq_guess is not None
            ), "If use_prior_model is False, flux_to_voltage_and_freq_guess"\
                "must be specified"

        # flux to voltage and frequency dictionaries (guessed and measured)
        self.flux_to_voltage_and_freq = {}
        if not flux_to_voltage_and_freq_guess is None:
            flux_to_voltage_and_freq_guess = (
                flux_to_voltage_and_freq_guess.copy())
        self.flux_to_voltage_and_freq_guess = flux_to_voltage_and_freq_guess

        # routine attributes
        self.fluxlines_dict = fluxlines_dict

        self.measurements = {
            float(k): tuple(transition_to_str(t) for t in v)
            for k, v in measurements.items()
        }

        # Validity of measurement dictionary
        for flux, transitions in self.measurements.items():
            for t in transitions:
                if not t in ["ge", "ef"]:
                    raise ValueError(
                        "Measurements must only include 'ge' and "
                        "'ef' (transitions). Currently, other transitions are "
                        "not supported.")

        # Guesses for voltage and frequency
        # determine sweet spot fluxes

        self.ss1_flux, self.ss2_flux, *rest = self.measurements.keys()
        assert self.ss1_flux % 0.5 == 0 and self.ss2_flux % 0.5 == 0, (
            "First entries of the measurement dictionary must be sweet spots"
            "(flux must be half integer)")

        # If use_prior_model is True, generate guess from prior Hamiltonian
        # model
        if use_prior_model:
            assert len(qubit.fit_ge_freq_from_dc_offset()) > 0, (
                "To use the prior Hamiltonian model, a model must be present in"
                " the qubit object")

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
            "fluxes in measurements")

        self.other_fluxes_with_guess = list(x - z)
        self.fluxes_without_guess = list(y - x)

        if self.get_param_value("get_parameters_from_qubit_object", False):
            update_nested_dictionary(
                self.settings,
                {self.highest_lookup: {
                    "General": self.parameters_qubit
                }})

        self.final_init(**kw)

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
        # for x in ["anharmonicity"]:
        #     v = self.kw.pop(x, None)
        #     if v:
        #         update_nested_dictionary(
        #             self.settings, {
        #                 self.highest_lookup: {
        #                     self.highest_sublookup: {
        #                         "Transmon": {
        #                             x: v
        #                         }
        #                     }
        #                 }
        #             })

        # DetermineModel
        for x in ["method"]:
            v = self.kw.pop(x, None)
            if v:
                update_nested_dictionary(
                    self.settings,
                    {self.highest_lookup: {
                        "DetermineModel": {
                            x: v
                        }
                    }})

        return super().parameters_init

    def create_routine_template(self):
        """Creates the routine template for the HamiltonianFitting routine using
        the specified parameters.
        """
        super().create_routine_template()
        qubit = self.qubit

        # Measurements at fluxes with provided guess voltage/freq
        for i, flux in enumerate([
                self.ss1_flux,
                self.ss2_flux,
                *self.other_fluxes_with_guess,
        ], 1):
            # Getting the guess voltage and frequency
            voltage_guess, ge_freq_guess = self.flux_to_voltage_and_freq_guess[
                flux]

            # Setting bias voltage to the guess voltage
            step_label = 'set_bias_voltage_' + str(i)
            settings = {step_label: {"voltage": voltage_guess}}
            step_settings = {'settings': settings}
            self.add_step(self.SetBiasVoltage, step_label, step_settings)

            for transition in self.measurements[flux]:
                # ge-transitions
                if transition == "ge":

                    step_label = 'update_frequency_' + \
                        transition + '_' + str(i)
                    # Updating ge-frequency at this voltage to guess value
                    settings = {
                        step_label: {
                            "frequency": ge_freq_guess,
                            "transition_name": transition,
                            # "use_prior_model": self.get_param_value('use_prior_model')
                        }
                    }
                    step_settings = {'settings': settings}
                    self.add_step(self.UpdateFrequency, step_label,
                                  step_settings)

                    # Finding the ge-transition frequency at this voltage
                    find_freq_settings = {
                        "transition_name": transition,
                    }
                    # Commented this block out as it should work without this
                    # find_freq_settings.update(
                    #    self.settings["FindFrequency"][transition]
                    # )
                    step_label = 'find_frequency_' + transition + '_' + str(i)
                    self.add_step(
                        FindFrequency,
                        step_label,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                       v_park=voltage_guess,
                                                       use_ro_flux=True),
                    )

                    # If this flux is one of the two sweet spot fluxes, we also
                    # perform a Reparking Ramsey and update the flux-voltage
                    # relation stored in routine.
                    if flux in [self.ss1_flux, self.ss2_flux]:

                        self.add_step(
                            ReparkingRamseyStep,
                            'reparking_ramsey_' +
                            ('1' if flux == self.ss1_flux else '2'),
                            {},
                            step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                           v_park=voltage_guess,
                                                           use_ro_flux=True),
                        )

                        # Updating voltage to flux with Reparking ramsey results
                        step_label = 'update_flux_to_voltage'
                        self.add_step(
                            self.UpdateFluxToVoltage,
                            step_label,
                            {
                                "index_reparking":
                                    len(self.routine_template) - 1,
                                "settings": {
                                    step_label: {
                                        "flux": flux
                                    }
                                },
                            },
                        )

                elif transition == "ef":
                    # Updating ef-frequency at this voltage to guess value
                    step_label = 'update_frequency_' + transition
                    settings = {
                        step_label: {
                            "flux": flux,
                            "transition_name": transition,
                            # "use_prior_model": self.get_param_value('use_prior_model')
                        }
                    }
                    step_settings = {'settings': settings}
                    self.add_step(self.UpdateFrequency, step_label,
                                  step_settings)

                    find_freq_settings = {
                        "transition_name": transition,
                    }
                    # Finding the ef-frequency
                    step_label = "find_frequency_" + transition
                    self.add_step(
                        FindFrequency,
                        step_label,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                       v_park=voltage_guess,
                                                       use_ro_flux=True),
                    )

        if not self.get_param_value("use_prior_model"):
            # Determining preliminary model - based on partial data and routine
            # parameters (e.g. fabrication parameters or user input)
            step_label = 'determine_model_preliminary'
            settings = {
                step_label: {
                    # should never be True
                    "include_resonator": self.get_param_value('use_prior_model')
                }
            }
            step_settings = {'settings': settings}
            self.add_step(self.DetermineModel, step_label, step_settings)

        # Measurements at other flux values (using preliminary or prior model)
        for i, flux in enumerate(self.fluxes_without_guess,
                                 len(self.other_fluxes_with_guess) + 2):
            # Updating bias voltage using earlier reparking measurements
            step_label = 'set_bias_voltage_' + str(i)
            self.add_step(self.SetBiasVoltage, step_label,
                          {'settings': {
                              step_label: {
                                  "flux": flux
                              }
                          }})

            # Looping over all transitions
            for transition in self.measurements[flux]:

                # Updating transition frequency of the qubit object to the value
                # calculated by prior or preliminary model
                step_label = 'update_frequency_' + transition + '_' + str(i)
                settings = {
                    'settings': {
                        step_label: {
                            "flux": flux,
                            "transition": transition,
                            "use_prior_model": True
                        }
                    }
                }
                self.add_step(self.UpdateFrequency, step_label, settings)

                # Set temporary values for Find Frequency
                step_label = 'set_tmp_values_flux_pulse_ro_' + \
                    transition+'_'+str(i)
                settings = {step_label: {"flux_park": flux,}}
                set_tmp_vals_settings = {
                    "settings": settings,
                    "index": len(self.routine_template) + 1,
                }
                self.add_step(
                    SetTemporaryValuesFluxPulseReadOut,
                    step_label,
                    set_tmp_vals_settings,
                )

                # Finding frequency
                step_label = 'find_frequency_' + transition + '_' + str(i)
                self.add_step(
                    FindFrequency,
                    step_label,
                    {"settings": {
                        step_label: {
                            'transition_name': transition
                        }
                    }},
                    step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                   v_park=voltage_guess,
                                                   use_ro_flux=True),
                )

        # Determining final model based on all data
        self.add_step(self.DetermineModel, 'determine_model_final', {})

        # Interweave routine if the user wants to include mixer calibration
        # FIXME taken out right now
        # self.add_mixer_calib_steps(**self.kw)

    def add_mixer_calib_steps(self, **kw):
        """Add steps to calibrate the mixer after the rest of the routine is
        defined. Mixer calibrations are put after every UpdateFrequency step.

        Configuration parameters (coming from the configuration parameter dictionary):
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
        include_mixer_calib_carrier = kw.get("include_mixer_calib_carrier",
                                             False)
        mixer_calib_carrier_settings = kw.get("mixer_calib_carrier_settings",
                                              {})
        mixer_calib_carrier_settings.update({
            "qubit": self.qubit,
            "update": True
        })

        # skewness settings
        include_mixer_calib_skewness = kw.get("include_mixer_calib_skewness",
                                              False)
        mixer_calib_skewness_settings = kw.get("mixer_calib_skewness_settings",
                                               {})
        mixer_calib_skewness_settings.update({
            "qubit": self.qubit,
            "update": True
        })

        if include_mixer_calib_carrier or include_mixer_calib_skewness:
            i = 0

            while i < len(self.routine_template):
                step_class = self.get_step_class_at_index(i)

                if step_class == self.UpdateFrequency:

                    # include mixer calibration skewness
                    if include_mixer_calib_skewness:

                        self.add_step(
                            MixerCalibrationSkewness,
                            'mixer_calibration_skewness',
                            mixer_calib_carrier_settings,
                            index=i + 1,
                        )
                        i += 1

                    # include mixer calibration carrier
                    if include_mixer_calib_carrier:

                        self.add_step(
                            MixerCalibrationCarrier,
                            'mixer_calibration_carrier',
                            mixer_calib_skewness_settings,
                            index=i + 1,
                        )
                        i += 1
                i += 1

    class UpdateFluxToVoltage(IntermediateStep):

        def __init__(self, index_reparking, **kw):
            """Intermediate step that updates the flux_to_voltage_and_freq
            dictionary using prior ReparkingRamsey measurements.

            Arguments:
                index_reparking: The index of the ReparkingRamsey step before

            Configuration parameters (coming from the configuration parameter dictionary):
                flux: the flux to update the voltage and frequency for using the
                    ReparkingRamsey results

            FIXME it might be useful to also include results from normal ramsey
            experiments. For example, this could be used in UpdateFrequency if
            there is no model but a measurement of the ge-frequency was done.
            """
            # arguments that are not arguments of the super init will be considered
            # as key words for the super init.
            super().__init__(index_reparking=index_reparking, **kw)

        def run(self):
            kw = self.kw

            qb = self.qubit

            # flux passed on in settings
            flux = self.get_param_value("flux")
            index_reparking = kw["index_reparking"]

            # voltage found by reparking routine
            reparking_ramsey = self.routine.routine_steps[index_reparking]
            try:
                apd = reparking_ramsey.analysis.proc_data_dict[
                    "analysis_params_dict"]
                voltage = apd["reparking_params"][
                    qb.name]["new_ss_vals"]["ss_volt"]
                frequency = apd["reparking_params"][
                    qb.name]["new_ss_vals"]["ss_freq"]
            except KeyError:
                log.error(
                    "Analysis reparking ramsey routine failed, flux to "
                    "voltage mapping can not be updated (guess values will be "
                    "used in the rest of the routine)")

                (
                    voltage,
                    frequency,
                ) = self.routine.flux_to_voltage_and_freq_guess[flux]

            self.routine.flux_to_voltage_and_freq.update(
                {flux: (voltage, frequency)})

    class UpdateFrequency(IntermediateStep):

        def __init__(self, **kw):
            """Updates the frequency of the specified transition.

            Keywords:
                routine (Routine): the routine to which this step belongs


            Configuration parameters (coming from the configuration parameter dictionary):

                frequency (float): frequency to which the qubit should be set
                transition_name (str): transition to be updated
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
            super().__init__(**kw,)

            # transition and frequency
            self.transition = self.get_param_value(
                'transition_name')  # default is "ge"
            self.frequency = self.get_param_value(
                'frequency')  # default is None
            self.flux = self.get_param_value('flux')
            self.voltage = self.get_param_value('voltage')

            assert not (
                self.frequency is None and self.flux is None and
                self.voltage is None
            ), "No transition, frequency or voltage specified. At least one of "
            "these should be specified."

        def run(self):
            """Updates frequency of the qubit for a given transition. This can
            either be done by passing the frequency directly, or in case a
            model exists by passing the voltage or flux.
            """
            qb = self.qubit
            frequency = self.frequency

            # If no explicit frequency is given, try to find it for given flux
            # or voltage using the Hamiltonian model stored in qubit
            if frequency is None:
                if self.transition == "ge" or "ef":
                    # A (possibly preliminary) Hamiltonian model exists
                    if (self.get_param_value('use_prior_model') and
                            len(qb.fit_ge_freq_from_dc_offset()) > 0):
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
                        frequency = (qb.ge_freq() +
                                     self.get_param_value("anharmonicity"))

                    # No Hamiltonian model exists
                    else:
                        raise NotImplementedError(
                            "Can not estimate frequency with incomplete model, "
                            "make sure to save a (possibly preliminary) model "
                            "first")

            if self.get_param_value('verbose'):
                print(f"{self.transition}-frequency updated to ", frequency,
                      "Hz")

            qb[f"{self.transition}_freq"](frequency)

    class SetBiasVoltage(IntermediateStep):

        def __init__(self, **kw):
            """Intermediate step that updates the bias voltage of the qubit.
            This can be done by simply specifying the voltage, or by specifying
            the flux. If the flux is given, the corresponding bias is calculated
            using the Hamiltonian model stored in the qubit object.
            """
            # arguments that are not arguments of the super init will be
            # considered as key words for the super init.
            super().__init__(**kw)

        def run(self):
            for qb in self.qubits:
                flux = self.get_param_value("flux", qubit=qb.name)
                if flux is not None:
                    voltage = qb.calculate_voltage_from_flux(flux)
                else:
                    voltage = self.get_param_value("voltage", qubit=qb.name)
                if voltage is None:
                    raise ValueError("No voltage or flux specified")
                self.routine.fluxlines_dict[qb.name](voltage)

    class DetermineModel(IntermediateStep):

        def __init__(
            self,
            **kw,
        ):
            """
            Intermediate step that determines the model of the qubit based on
            the measured data. Can be used for both estimating the model and
            determining the final model.

            Configuration parameters (coming from the configuration parameter dictionary):
                include_reparkings (bool): if True, the data from the
                    ReparkingRamsey measurements are used to help determine a
                    Hamiltonian model.
                include_resonator (bool): if True, the model includes the effect
                    of the resonator. If False, the coupling is set to zero and
                    essentially only the transmon parameters are optimized (the
                    resonator has no effect on the transmon).
                use_prior_model (bool): if True, the prior model is used to
                    determine the model, if False, the measured data is used.
                method (str): the optimization method to be used
                    when determining the model. Default is Nelder-Mead.
            """

            # arguments that are not arguments of the super init will be
            # considered as key words for the super init.
            super().__init__(**kw,)

        def run(self):
            kw = self.kw

            # using all experimental values
            self.experimental_values = (
                HamiltonianFitting.get_experimental_values(
                    qubit=self.routine.qubit,
                    fluxlines_dict=self.routine.fluxlines_dict,
                    timestamp_start=self.routine.preroutine_timestamp,
                    include_reparkings=self.get_param_value(
                        "include_reparkings"),
                ))

            log.info(f"Experimental values: {self.experimental_values}")

            # preparing guess parameters and choosing parameters to optimize
            p_guess, parameters_to_optimize = self.make_model_guess(
                use_prior_model=self.get_param_value("use_prior_model"),
                include_resonator=self.get_param_value("include_resonator"),
            )

            log.info(f"Parameters guess: {p_guess}")
            log.info(f"Parameters to optimize: {parameters_to_optimize}")

            # determining the model
            f = self.routine.optimizer(
                experimental_values=self.experimental_values,
                parameters_to_optimize=parameters_to_optimize,
                parameters_guess=p_guess,
                method=self.get_param_value("method"),
            )

            # extracting results, store results dictionary for use in
            # get_device_property_values.
            self.__result_dict = self.routine.fit_parameters_from_optimization_results(
                f, parameters_to_optimize, p_guess)

            log.info(f"Result from fit: {self.__result_dict}")

            # saving model to qubit from routine
            self.routine.qubit.fit_ge_freq_from_dc_offset(self.__result_dict)

            # Save timestamp from previous run, to use for get_device_property_values
            self.__end_of_run_timestamp = a_tools.get_last_n_timestamps(1)[0]

        def get_device_property_values(self, **kwargs):
            """Returns a dictionary of high-level device property values from
            running this DetermineModel step

            Args:
                qubit_sweet_spots (dict, optional): a dictionary mapping qubits
                    to sweet-spots ('uss', 'lss', or None)

            Returns:
                results (dict, optional): dictionary of high-level results (may
                    be empty)
            """

            results = self.get_empty_device_properties_dict()
            sweet_spots = kwargs.get('qubit_sweet_spots', {})
            if _device_db_client_module_missing:
                log.warning("Assemblying the dictionary of high-level device "
                "property values requires the module 'device-db-client', which "
                "was not imported successfully.")
            elif self.__result_dict is not None:
                # For DetermineModel, the results are in
                # self.__result_dict from self.run()
                # The timestamp is the end of run timestamp or the one
                # immediately after. If we set `save_instrument_settings=True`,
                # we will have a timestamp after __end_of_run_timestamp.
                if self.get_param_value('save_instrument_settings'
                                       ) or not self.get_param_value("update"):
                    timestamps = a_tools.get_timestamps_in_range(
                        self.__end_of_run_timestamp)
                    # one after __end_of_run_timestamp
                    timestamp = timestamps[1]
                else:
                    timestamp = self.__end_of_run_timestamp
                node_creator = db_utils.ValueNodeCreator(
                    qubits=self.routine.qubit.name,
                    timestamp=timestamp,
                )

                # Total Josephson Energy
                if 'Ej_max' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='Ej_max',
                            value=self.__result_dict['Ej_max'],
                        ),)

                # Charging Energy
                if 'E_c' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='E_c',
                            value=self.__result_dict['E_c'],
                        ),)

                # asymmetry
                if 'asymmetry' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='asymmetry',
                            value=self.__result_dict['asymmetry'],
                        ),)

                # coupling
                if 'coupling' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='coupling',
                            value=self.__result_dict['coupling'],
                        ),)

                # Bare Readout Resonator Frequency
                if 'fr' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='fr',
                            component_type='ro_res',  # Not a qubit property
                            value=self.__result_dict['fr'],
                        ),)

                # Anharmonicity
                if 'anharmonicity' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='anharmonicity',
                            value=self.__result_dict['anharmonicity'],
                        ),)
            return results

        def make_model_guess(self,
                             use_prior_model=True,
                             include_resonator=True):
            """Constructing parameters for the for the Hamiltonian model
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
                p_guess = self.qubit.fit_ge_freq_from_dc_offset()

            # using guess parameters instead
            else:
                p_guess = {
                    "Ej_max":
                        self.get_param_value("Ej_max"),
                    "E_c":
                        self.get_param_value("E_c"),
                    "asymmetry":
                        self.get_param_value("asymmetry"),
                    "coupling":
                        self.get_param_value("coupling") * include_resonator,
                    "fr":
                        self.get_param_value(
                            "fr", associated_component_type_hint="ro_res"),
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
            p_guess.update({
                "dac_sweet_spot": dac_sweet_spot,
                "V_per_phi0": V_per_phi0
            })

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
    def verification_measurement(qubit,
                                 fluxlines_dict,
                                 fluxes=None,
                                 voltages=None,
                                 verbose=True,
                                 **kw):
        """Performs a verification measurement of the model for given fluxes
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
            "qubit(s) to contain a fit_ge_freq_from_dc_offset model")

        fluxline = fluxlines_dict[qubit.name]
        initial_voltage = fluxline()

        if fluxes is None and voltages is None:
            fluxes = np.linspace(0, -0.5, 11)
        if voltages is None:
            voltages = qubit.calculate_voltage_from_flux(fluxes)

        experimental_values = {}  # empty dictionary to store results

        for index, voltage in enumerate(voltages):
            with temporary_value(
                    *ro_flux_tmp_vals(qubit, v_park=voltage, use_ro_flux=True)):

                if verbose:
                    print(f"Verification measurement  step {index} / "
                          f"{len(voltages)} of {qubit.name}")

                # setting the frequency and fluxline
                qubit.calculate_frequency(voltage, update=True)
                fluxline(voltage)

                # finding frequency
                ff = FindFrequency([qubit], dev=kw.get('dev'), update=True)

                # storing experimental result
                experimental_values[voltage] = {"ge": qubit.ge_freq()}

        if kw.pop("reset_fluxline", True):
            # resetting fluxline to initial value
            fluxline(initial_voltage)

        # plotting
        if kw.pop("plot", False):
            HamiltonianFitting.plot_model_and_experimental_values(
                result_dict=result_dict,
                experimental_values=experimental_values)
            HamiltonianFitting.calculate_residuals(
                result_dict=result_dict,
                experimental_values=experimental_values,
                plot_residuals=True,
            )

        return experimental_values

    def get_device_property_values(self, **kwargs):
        """Returns a property values dictionary of the fitted Hamiltonian
        parameters

        Returns:
            dict: the property values dictionary for this routine
        """
        results = self.get_empty_device_properties_dict()
        # Only return property values from DetermineModel steps
        for _, step in enumerate(self.routine_steps):
            # FIXME: for some reason isinstance doesn't work here.
            if type(step).__name__ == 'DetermineModel':
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


class MixerCalibrationSkewness(IntermediateStep):

    def __init__(self, routine, **kw):
        """Mixer calibration step that calibrates the skewness of the mixer.

        Args:
            routine: Routine object

        Keyword Arguments:
            calibrate_drive_mixer_skewness_function: method for calibrating to
                be used. Default is to use calibrate_drive_mixer_skewness_model.
        """
        super().__init__(routine, **kw)

    def run(self):
        kw = self.kw

        # FIXME used only default right now, kw is not passed
        calibrate_drive_mixer_skewness_function = kw.get(
            "calibrate_drive_mixer_skewness_function",
            "calibrate_drive_mixer_skewness_model",
        )

        function = getattr(self.qubit, calibrate_drive_mixer_skewness_function)
        new_kw = keyword_subset_for_function(kw, function)

        function(**new_kw)


class MixerCalibrationCarrier(IntermediateStep):

    def __init__(self, routine, **kw):
        """Mixer calibration step that calibrates the carrier of the mixer.

        Args:
            routine: Routine object

        Keyword Arguments:
            calibrate_drive_mixer_carrier_function: method for calibrating to
                be used. Default is to use calibrate_drive_mixer_carrier_model.
        """
        super().__init__(routine, **kw)

    def run(self):
        kw = self.kw

        # FIXME used only default right now, kw is not passed
        calibrate_drive_mixer_carrier_function = kw.get(
            "calibrate_drive_mixer_carrier_function",
            "calibrate_drive_mixer_carrier_model",
        )

        function = getattr(self.qubit, calibrate_drive_mixer_carrier_function)
        new_kw = keyword_subset_for_function(kw, function)

        function(**new_kw)


class SetTemporaryValuesFluxPulseReadOut(IntermediateStep):

    def __init__(
        self,
        index,
        **kw,
    ):
        """Intermediate step that sets ro-temporary values for a step of the
        routine.

        Args:
            routine: Routine object
            index: index of the step in the routine that requires temporary
                values for flux-pulse assisted read-out

        Configuration parameters (coming from the configuration parameter dictionary):
            flux_park:  flux to calculate the the ro-temporary value to set
            voltage_park: value of the ro-temporary value to set

        """
        super().__init__(
            index=index,
            **kw,
        )

    def run(self):
        kw = self.kw
        qb = self.qubit
        index = self.kw["index"]

        if flux := self.get_param_value("flux_park") is not None:
            v_park = qb.calculate_voltage_from_flux(flux)
        elif v_park_tmp := self.get_param_value("voltage_park") is not None:
            v_park = v_park_tmp
        else:
            raise ValueError("No voltage or flux specified")

        # temporary values for ro
        ro_tmp_vals = ro_flux_tmp_vals(qb, v_park, use_ro_flux=True)

        # extending temporary values
        self.routine.extend_step_tmp_vals_at_index(tmp_vals=ro_tmp_vals,
                                                   index=index)


class SingleQubitCalib(AutomaticCalibrationRoutine):
    """Single qubit calibration brings the setup into a default state and
    calibrates the specified qubits. It consists of several steps:
        <Class name> (<step_label>)

        SQCPreparation (sqc_preparation)
        RabiStep (rabi)
        RamseyStep (ramsey_large_AD)
        RamseyStep (ramsey_small_AD)
        RabiStep (rabi_after_ramsey)
        QScaleStep (qscale)
        RabiStep (rabi_after_qscale)
        T1Step (t1)
        RamseyStep (echo_large_AD)
        RamseyStep (echo_small_AD)
        InPhaseAmpCalibStep (in_phase_calib)

    Args:
        dev: device to be used for the routine

    Keyword args:
        qubits: The qubits which should be calibrated. By default,
            all qubits of the device are selected.

    Configuration parameters (coming from the configuration parameter dictionary):
        autorun (bool): If True, the routine runs automatically. Otherwise,
            run() has to called.
        transition_names (list): List of transitions for which the steps in the
            routine should be conducted. Example:
                "transition_names": ["ge"]
        rabi (bool): Enables the respective step. Enabled by default.
        ramsey_large_AD (bool): Enables the respective step.
        ramsey_small_AD (bool): Enables the respective step. Enabled by default.
        rabi_after_ramsey (bool): Enables the respective step.
        qscale (bool): Enables the respective step. Enabled by default.
        rabi_after_qscale (bool): Enables the respective step.
        t1 (bool): Enables the respective step.
        echo_large_AD (bool): Enables the respective step.
        echo_small_AD (bool): Enables the respective step.
        in_phase_calib (bool): Enables the respective step.

        nr_rabis (dict): nested dictionary containing the transitions and a
            respective list containing numbers of Rabi pulses. For each entry in
            the list, the Rabi experiment is repeated with the respective number
            of pulses.
            Example:
                "nr_rabis": {
                    "ge": [1,3],
                    "ef": [1]
                },
    """

    def __init__(self, dev, **kw):

        AutomaticCalibrationRoutine.__init__(
            self,
            dev=dev,
            **kw,
        )

        self.final_init(**kw)

    def create_routine_template(self):
        """
        Creates routine template.
        """
        super().create_routine_template()

        detailed_routine_template = copy.copy(self.routine_template)
        detailed_routine_template.clear()

        for transition_name in self.get_param_value('transition_names'):
            for step in self.routine_template:
                step_class = step[0]
                settings = copy.deepcopy(step[2])
                label = step[1]
                if issubclass(step_class, SingleQubitGateCalibExperiment):

                    if 'qscale' in label or 'echo' in label:
                        if transition_name != 'ge':
                            continue
                    if not self.get_param_value(label):
                        continue
                    new_label = label + "_" + transition_name
                    update_nested_dictionary(
                        settings['settings'], {
                            step_class.get_lookup_class().__name__: {
                                'transition_name': transition_name
                            }
                        })
                    # FIXME search for parallel groups in settings
                    lookups = [
                        label,
                        step_class.get_lookup_class().__name__, 'General'
                    ]
                    for parallel_group in self.get_param_value(
                            'parallel_groups', sublookups=lookups, leaf=True):
                        qubits_filtered = [
                            qb for qb in self.qubits
                            if qb.name is parallel_group or
                            parallel_group in self.get_qubit_groups(qb.name)
                        ]
                        if len(qubits_filtered) != 0:
                            temp_settings = copy.deepcopy(settings)
                            temp_settings['qubits'] = qubits_filtered
                            qubit_label = new_label + "_"
                            for qb in qubits_filtered:
                                qubit_label += qb.name

                            if issubclass(step_class, qbcal.Rabi):
                                for n in self.get_param_value(
                                        'nr_rabis')[transition_name]:
                                    qubits = temp_settings.pop('qubits')
                                    temp_settings['settings'] = copy.deepcopy(
                                        temp_settings['settings'])
                                    temp_settings['qubits'] = qubits
                                    update_nested_dictionary(
                                        temp_settings['settings'], {
                                            step_class.get_lookup_class().__name__:
                                                {
                                                    'n': n
                                                }
                                        })
                                    detailed_routine_template.add_step(
                                        step_class, qubit_label, temp_settings)

                            else:
                                detailed_routine_template.add_step(
                                    step_class, qubit_label, temp_settings)
                else:
                    detailed_routine_template.add_step(step_class, label,
                                                       settings)
        self.routine_template = detailed_routine_template

    class SQCPreparation(IntermediateStep):

        def __init__(
            self,
            routine,
            **kw,
        ):
            """Intermediate step that configures qubits for Mux drive and
            readout.

            Args:
                routine: the parent routine

            Configuration parameters (coming from the configuration parameter dictionary):
                configure_mux_readout (bool): Whether to configure the qubits
                    for multiplexed readout or not.
                configure_mux_drive (bool): Specifies if the LO frequencies and
                    IFs of the qubits measured in this step should be updated.
                reset_to_defaults (bool): Whether to reset the sigma values to
                    the default or not.
                default_<transition>_sigma (float): The default with of the
                    pulse in units of the sigma of the pulse.
                ro_lo_freqs (dict): Dictionary containing MWG names and readout
                    local oscillator frequencies.
                drive_lo_freqs (dict):  Dictionary containing MWG names and
                    drive locas oscillator frequencies
                acq_averages (int): The number of averages for each measurement
                    for each qubit.
                acq_weights_type (str): The weight type for the readout for each
                    qubit.
                preparation_type (str): The preparation type of the qubit.
                trigger_pulse_period (float): The delay between individual
                    measurements.
            """
            super().__init__(
                routine=routine,
                **kw,
            )

        def run(self):

            kw = self.kw

            if self.get_param_value('configure_mux_readout'):
                ro_lo_freqs = self.get_param_value('ro_lo_freqs')
                configure_qubit_mux_readout(self.qubits, ro_lo_freqs)
            if self.get_param_value('configure_mux_drive'):
                drive_lo_freqs = self.get_param_value('drive_lo_freqs')
                configure_qubit_mux_drive(self.routine.qubits, drive_lo_freqs)

            if (self.get_param_value('reset_to_defaults')):
                for qb in self.qubits:
                    qb.ge_sigma(self.get_param_value('default_ge_sigma'))
                    qb.ef_sigma(self.get_param_value('default_ef_sigma'))

            self.dev.set_default_acq_channels()

            self.dev.preparation_params().update(
                {'preparation_type': self.get_param_value('preparation_type')})
            for qb in self.qubits:
                qb.preparation_params(
                )['preparation_type'] = self.get_param_value('preparation_type')
                qb.acq_averages(
                    kw.get('acq_averages',
                           self.get_param_value('acq_averages')))
                qb.acq_weights_type(self.get_param_value('acq_weights_type'))

            trigger_device = self.qubits[0].instr_trigger.get_instr()
            trigger_device.pulse_period(
                self.get_param_value('trigger_pulse_period'))

    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
        [SQCPreparation, 'sqc_preparation', {}],
        [RabiStep, 'rabi', {}],
        [RamseyStep, 'ramsey_large_AD', {}],
        [RamseyStep, 'ramsey_small_AD', {}],
        [RabiStep, 'rabi_after_ramsey', {}],
        [QScaleStep, 'qscale', {}],
        [RabiStep, 'rabi_after_qscale', {}],
        [T1Step, 't1', {}],
        [RamseyStep, 'echo_large_AD', {}],
        [RamseyStep, 'echo_small_AD', {}],
        [InPhaseAmpCalibStep, 'in_phase_calib', {}],
    ])


def keyword_subset(keyword_arguments, allowed_keywords):
    """Returns a dictionary with only the keywords that are used by the
    function.
    """
    keywords = set(keyword_arguments.keys())
    keyswords_to_extract = keywords.intersection(allowed_keywords)

    new_kw = {key: keyword_arguments[key] for key in keyswords_to_extract}

    return new_kw


def keyword_subset_for_function(keyword_arguments, function):
    """Returns a dictionary with only the keywords that are used by the
    function.
    """
    allowed_keywords = inspect.getfullargspec(function)[0]

    return keyword_subset(keyword_arguments, allowed_keywords)


def update_nested_dictionary(d, u):
    """Returns a dictionary with only the keywords that are used by the
    function.

    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d
