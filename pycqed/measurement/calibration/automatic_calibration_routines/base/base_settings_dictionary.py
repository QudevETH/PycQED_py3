import copy
import logging
from pathlib import Path
import json
import re
from collections.abc import Mapping
from typing import Union, Dict, Any, Tuple

log = logging.getLogger('Routines')
# log.setLevel('INFO')

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

    The hierarchy (in descending significance) is
    User > Sample > Setup > Default.
    Routine settings can contain separate dictionaries for each step and even
    dictionaries with specific step-labels (whenever a step can repeat itself).
    In addition, they can contain a 'General' dictionary with more settings.

    Examples::

        adaptive_qubit_spectroscopy_settings = {
            "General":{
                "n_spectroscopies": 2,
                "max_iterations": 3,
                "auto_repetition_settings": false,
                "max_waiting_seconds": 60
            },

            "QubitSpectroscopy1D":{
                "spec_power": -10,
                "freq_range": 400e6,
                "pts": 200,
                "pulsed": false,
                "modulated": false
            },

            "Decision": {
                "max_kappa_fraction_sweep_range": 0.2,
                "min_kappa_absolute": 0
            },

            "qubit_spectroscopy_1": {
                "spec_power": -5,
                "freq_range": 1.5e9,
                "pts": 1000,
                "freq_center": "{current}"
            },

            "qubit_spectroscopy_1_repetition_2": {
                "spec_power": 0,
                "freq_range": 2e9,
                "pts": 1500,
                "freq_center": "{current}"
            },

            "qubit_spectroscopy_1_repetition_3": {
                "spec_power": 5,
                "freq_range": 2.5e9,
                "pts": 2500,
                "freq_center": "{current}"
            },

            "qubit_spectroscopy_2": {
                "spec_power": -15,
                "freq_range": 50e6,
                "pts": 200
            },

            "decision_spectroscopy_2": {
                "max_kappa_absolute": 1e6
            }
        }
    """

    _USE_DB_STRING = "USE_DB"

    def __init__(self,
                 init_dict=None,
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
        init_dict = init_dict or {}
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
                                     associated_component_type_hint=None) -> \
            Tuple[Any, bool]:
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
            groups (set): The groups the qubit is in, as specified in
                the dictionary.
            leaf (boolean): True if the scope to search the parameter for is a
                leaf node (e.g. a measurement or intermediate step, not a
                routine).
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
                        # Ignore the qubits list of the experiment
                        if isinstance(self[lookup]['qubits'], dict):
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
                        associated_component_type_hint=None) -> Tuple[
            Any, bool]:
        """Looks up the requested parameter in the configuration parameters.
        If the fetched value is a request to query the database, the queried
        value is returned.

        Args:
            param (str): The name of the parameter to look up.
            lookups (list of str): The scopes in which the parameter should be
                looked up.
            sublookups (list of str, optional): The subscopes to be looked up.
                The parameter is then looked up in self[lookup][sublookup].
            qubit (str): The name of the qubit, if the parameter is
                qubit-specific.
            groups (set): The groups the qubit is in, as specified in
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

    def get_qubit_groups(self, qubit, lookups) -> set:
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
                                settings_default_folder: Union[
                                    Path, str] = None,
                                settings_setup_folder: Union[Path, str] = None,
                                settings_sample_folder: Union[Path, str] = None,
                                settings_user: Dict[str, Any] = None):
        """Loads the device settings from the folders storing Default, Setup and
        Sample parameters and puts it into the configuration parameter
        dictionary as a nested dictionary. The folders should contain JSON
        files.
        Order in the hierarchy: Sample overwrites Setup overwrites Default
        values.
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
            settings_user (dict): User settings as a dictionary.
        """
        if settings_default_folder is None:
            dirname = Path(__file__).parent.parent
            # path to the directory `automatic_calibration_routines`
            settings_default_folder = Path(dirname,
                                           "autocalib_default_settings")
        if settings_setup_folder is None:
            log.warning("No settings_setup_folder specified.")
        if settings_sample_folder is None:
            log.warning("No settings_sample_folder specified.")

        for settings_folder in [
            settings_default_folder,
            settings_setup_folder,
            settings_sample_folder
        ]:
            if settings_folder is not None:
                for file in Path.iterdir(settings_folder):
                    with open(file) as f:
                        update_nested_dictionary(
                            self, {file.stem: json.load(f)})

        if settings_user is not None:
            self.update_user_settings(settings_user)

        self._postprocess_settings_from_file()

    def _postprocess_settings_from_file(self):
        """Since JSON only supports strings as keys, postprocessing is applied.
        Therefore, it is also possible to use a string with a tuple inside as a
        key, which is converted to a tuple in the dictionary, e.g.
        "('SFHQA', 1)" is converted to ('SFHQA', 1).
        """
        for k in list(self.keys()):
            if isinstance(self[k], Mapping):
                SettingsDictionary._postprocess_settings_from_file(self[k])
            if re.search(r'^\(.*\)$', k):  # represents a tuple
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


def update_nested_dictionary(d, u: Mapping) -> dict:
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
        # update_nested_dictionary is called again recursively. The
        # subdictionary d[k] will be updated with v.
        if isinstance(v, Mapping):
            d[k] = update_nested_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d
