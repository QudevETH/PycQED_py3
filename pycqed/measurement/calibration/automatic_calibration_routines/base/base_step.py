from .base_settings_dictionary import SettingsDictionary
from typing import List, Dict, Any, Optional
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon\
    import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.device import Device
from pycqed.measurement.quantum_experiment import QuantumExperiment
from pycqed.measurement.calibration.single_qubit_gates import \
    SingleQubitGateCalibExperiment
from collections import OrderedDict


class Step:
    """
    A class to collect functionality used for each step in a routine.
    A step can be an AutomaticCalibrationRoutine, an IntermediateStep
    or a measurements. Measurements are wrapped with a wrapper class
    which inherits from this class to give access to its functions.
    """

    def __init__(self,
                 dev: Device,
                 routine: Optional[Any] = None,
                 step_label: Optional[str] = None,
                 settings: Optional[SettingsDictionary] = None,
                 qubits: Optional[List[QuDev_transmon]] = None,
                 settings_user: Optional[Dict[str, Any]] = None,
                 **kw):
        """
        Initializes the Step class.

        Arguments:
            dev (Device): The device which is currently measured.
            routine (AutomaticCalibrationRoutine): The parent of the step. If
                this step is the root routine, this should be None.
            step_label (str): A unique label for this step to be used in the
                configuration parameters files.
            settings (SettingsDictionary): The configuration parameters
                passed down from its parent. If None, the dictionary is taken
                from the Device object.
            qubits (list): A list with the Qubit objects which should be part of
                the step.
            settings_user (dict): A dictionary from the user to update the
                configuration parameters. The structure of the dictionary must
                be compatible with that of a general settings dictionary.

        Keyword args:
            Shouldn't be any, this is here for backwards compatibility.

        """
        self.routine = routine
        self.step_label = step_label
        self.dev = dev
        # Copy default settings from autocalib if this is the root routine, else
        # create an empty SettingsDictionary
        default_settings = self.dev.autocalib_settings.copy(
        ) if self.routine is None else self.routine.settings.copy({})
        self.settings = settings or default_settings
        self.qubits = qubits or self.dev.get_qubits()

        # FIXME: Remove dependency on self.qubit. This is there to make the
        #  current one-qubit-only implementation of HamiltonianFitting work.
        self.qubit = self.qubits[0]

        if settings_user:
            self.settings.update_user_settings(settings_user)
        self.parameter_lookups = [
            self.step_label, self.get_lookup_class().__name__, 'General'
        ]
        self.parameter_sublookups = None
        self.leaf = True

        # Store results with qubit names as keys
        self.results: Optional[Dict[str, Dict[str, Any]]] = {}

    class NotFound:
        """
        This class is used in get_param_value to identify the cases where
        a keyword could not be found in the configuration parameter dictionary.
        It is necessary to distinguish between the cases when None is explicitly
        specified for a keyword argument and when no keyword argument was found.
        """

        def __bool__(self):
            """
            Return False by default for the truth value of an instance of
            NotFound.
            """
            return False

    def get_param_value(self,
                        param: str,
                        qubit: str = None,
                        sublookups=None,
                        default=None,
                        leaf=None,
                        associated_component_type_hint=None):
        """
        Looks up the requested parameter in the own configuration parameter
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
            qubit (str): Qubit name to which the param is related
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
            associated_component_type_hint (str): A hint for the device
                database, if the parameter does not belong to the qubit.

        Returns:
            The value found in the dictionary. If no value was found, either the
            default value is returned if specified or otherwise None.
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
                    associated_component_type_hint=
                    associated_component_type_hint
                )

        return val if success else default

    def get_qubit_groups(self, qubit):
        """
        Gets the groups the specified qubit belongs to out of the
        configuration parameter dictionary.

        Args:
            qubit (str): The name of the qubit.

        Returns:
            set: A set of strings with the group names the qubit is part of.
        """
        # FIXME: When a qubit is removed from a group with the same
        #  name in higher hierarchy, it will still remain.
        lookups = ['Groups']
        groups = self.settings.get_qubit_groups(qubit, lookups)
        if self.routine is not None:
            groups.update(self.routine.get_qubit_groups(qubit))
        return groups

    def run(self):
        """
        Run the Step. To be implemented by child classes."""
        pass

    def post_run(self):
        """
        Execute after run. To be implemented by child classes.
        Update results, for example."""
        pass

    def get_empty_device_properties_dict(self, step_type=None):
        """
        Returns an empty dictionary of the following structure, for use with
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
        """
        Returns a dictionary of high-level property values from running this
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
        """
        Returns the kwargs necessary to run a QuantumExperiment. Every
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
                OrderedDict({
                    Step.__name__: {
                        # kwarg: (fieldtype, default_value),
                        # 'delegate_plotting': (bool, False),
                    },
                })
        }

    def get_requested_settings(self):
        """
        Gets a set of keyword arguments which are needed for the
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
        """
        Resolves the keyword arguments from get_requested_settings to calls
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
        kwargs['experiment_name'] = self.step_label
        return kwargs

    @property
    def highest_lookup(self):
        """
        Returns the highest scope for this step which is not None
        in the order step_label, class name, General.

        Returns:
            str: A string with the highest lookup of this step which is not
                None.
        """
        return self._get_first_not_none(self.parameter_lookups)

    @property
    def highest_sublookup(self):
        """
        Returns the highest subscope for this step which is not None.

        Returns:
            str: A string with the highest sublookup of this step which is not
                None. If the step is a leaf, this is None, otherwhise it is
                "General".
        """
        return None if self.parameter_sublookups is None else self._get_first_not_none(
            self.parameter_sublookups)

    @staticmethod
    def _get_first_not_none(lookup_list):
        """
        Returns the first subscope in the lookup list that is not None.

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
    """
    Class used for defining intermediate steps between automatic calibration
    steps.

    NOTE: Currently, there is no difference between an IntermediateStep and a
    Step. A different class was implemented just in case future modifications
    will make it necessary.
    """

    def __init__(self, **kw):
        self.kw = kw
        super().__init__(**kw)

    def run(self):
        """
        Intermediate processing step to be overridden by Children (which are
        routine specific).
        """
        pass
