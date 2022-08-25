from .autocalib_framework import (
    Step,
    IntermediateStep,
    RoutineTemplate,
    AutomaticCalibrationRoutine
)
from .autocalib_framework import update_nested_dictionary
from .autocalib_framework import (
    _device_db_client_module_missing
)
if not _device_db_client_module_missing:
    from pycqed.utilities.devicedb import utils as db_utils

from pycqed.measurement.sweep_points import SweepPoints

from pycqed.measurement.calibration.single_qubit_gates import (
    SingleQubitGateCalibExperiment
)
from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.utilities.state_and_transition_translation import *
from pycqed.utilities.general import (
    configure_qubit_mux_drive,
    configure_qubit_mux_readout
)

import numpy as np
import copy
import logging
import time

log = logging.getLogger(__name__)


# FIXME: these wrappers can probably be collected with an ExperimentStep class
# to avoid repetition
class RabiStep(qbcal.Rabi, Step):
    """A wrapper class for the Rabi experiment.
    """

    def __init__(self, routine, **kwargs):
        """
        Initializes the RabiStep class, which also includes initialization
        of the Rabi experiment.

        Arguments:
            routine (Step): The parent routine

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter
        dictionary):
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
            pts: Number of points for the sweep voltage. Can either be an int
                or an expression to be evaluated. See above.
            max_drive_amp (float): The maximum drive amplitude.
            default_<transition>_amp180 (float): A default value for the pi
                pulse to be set back to.
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

            value_params = {'v_low': None, 'v_high': None, 'pts': None}
            # The information about the custom parameters above could be
            # Saved somewhere else to generalize all wrappers

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
                # Get first dimension (there is only one)
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
    """A wrapper class for the Ramsey experiment.
    """

    def __init__(self, routine, *args, **kwargs):
        """Initializes the RamseyStep class, which also includes initialization
        of the Ramsey experiment.

        Args:
            routine (Step obj): The parent routine

        Keyword Arguments:
            qubits (list): list of qubits to be used in the routine

        Configuration parameters (coming from the configuration parameter
        dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): a list of all groups of qubits on which the
                Ramsey measurement can be conducted in parallel
            t0 (float): Minimum delay time for the Ramsey experiment.
            delta_t (float): Duration of the delay time for the Ramsey
                experiment.
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
                # Get first dimension (there is only one)
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
        running this RamseyStep.

        Keyword Arguments:
            qubit_sweet_spots (dict, optional): a dictionary mapping qubits to
                sweet-spots ('uss', 'lss', or None).

        Returns:
            dict: dictionary of high-level results (may be empty).
        """

        results = self.get_empty_device_properties_dict()
        sweet_spots = kwargs.get('qubit_sweet_spots', {})
        if _device_db_client_module_missing:
            log.warning(
                "Assemblying the dictionary of high-level device "
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
    """A wrapper class for the ReparkingRamsey experiment.
    """

    def __init__(self, routine, *args, **kwargs):
        """Initializes the ReparkingRamseyStep class, which also includes
        initialization of the ReparkingRamsey experiment.

        Arguments:
            routine (Step): The parent routine.

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            transition_name (str): The transition of the experiment
            parallel_groups (list): A list of all groups of qubits on which the
                ReparkingRamsey experiment can be conducted in parallel
            t0 (float): Minimum delay time for the ReparkingRamsey experiment.
            delta_t (float): Duration of the delay time for the ReparkingRamsey
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

            # FIXME: can this be combined with RamseyStep to avoid code
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
                # Get first dimension (there is only one)
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
    """A wrapper class for the T1 experiment.
    """

    def __init__(self, routine, *args, **kwargs):
        """Initializes the T1Step class, which also includes initialization
        of the T1 experiment.

        Arguments:
            routine (Step): The parent routine.

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            transition_name (str): The transition of the experiment.
            parallel_groups (list): A list of all groups of qubits on which the
                T1 experiment can be conducted in parallel.
            t0 (float): Minimum delay time for the T1 experiment.
            delta_t (float): Duration of the delay time for the T1 experiment.
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
                the T1 class.
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
                sweet-spots ('uss', 'lss', or None).

        Returns:
            dict: Dictionary of high-level device property values.
        """
        results = self.get_empty_device_properties_dict()
        sweet_spots = kwargs.get('qubit_sweet_spots', {})
        if _device_db_client_module_missing:
            log.warning(
                "Assemblying the dictionary of high-level device "
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
    """

    def __init__(self, routine, *args, **kwargs):
        """Initializes the QScaleStep class, which also includes initialization
        of the QScale experiment.

        Arguments:
            routine (Step): The parent routine.

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            transition_name (str): The transition of the experiment.
            parallel_groups (list): A list of all groups of qubits on which the
                QScale experiment can be conducted in parallel.
            v_low (float): Minimum of the sweep range for the qscale parameter.
            v_high (float):  Maximum of the sweep range for the qscale parameter.
            pts (int): Number of points for the sweep range.
            configure_mux_drive (bool): If the LO frequencies and IFs of the
                qubits measured in this step should be updated afterwards.
        """

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
                the QScale class.
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
                # Get first dimension (there is only one)
                # TODO: support for more dimensions?
                sweep_points_kws = next(iter(
                    self.kw_for_sweep_points.items()))[1]
                values = np.linspace(value_params['v_low'],
                                     value_params['v_high'],
                                     value_params['pts'])
                task['sweep_points'] = SweepPoints()
                # FIXME: why is values_func an invalid paramteter, if it is in
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
    """A wrapper class for the InPhaseAmpCalibStep experiment.
    """

    def __init__(self, routine, *args, **kwargs):
        """
        Arguments:
            routine (Step): The parent routine.

        Keyword Arguments:
            qubits (list): List of qubits to be used in the routine.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            n_pulses (int): Max number of pi pulses to be applied.
        """
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
            dict: Dictionary containing names and default values of the keyword
                arguments to be passed to the InPhaseAmpCalib class.
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

    Routine steps:
    1) Rabi
    2) Ramsey
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
        super().__init__(
            dev=dev,
            **kw,
        )
        self.final_init(**kw)

    def create_routine_template(self):
        """Creates routine template.
        """
        super().create_routine_template()
        # Loop in reverse order so that the correspondence between the index
        # of the loop and the index of the routine_template steps is preserved
        # when new steps are added
        for i, step in reversed(list(enumerate(self.routine_template))):
            self.split_step_for_parallel_groups(index=i)

    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
        [RabiStep, 'rabi', {}],
        [RamseyStep, 'ramsey', {}],
    ])


class FindFrequency(AutomaticCalibrationRoutine):
    """Routine to find frequency of a given transmon transition.

    Routine steps:
    1) PiPulseCalibration (Rabi + Ramsey)
    2) Decision: checks whether the new frequency of the qubit found after the
    Ramsey experiment is within a specified threshold with respect to the
    previous one. If not, it adds another PiPulseCalibration and Decision step.
    """

    def __init__(
        self,
        dev,
        qubits,
        **kw,
    ):
        """Routine to find frequency of a given transmon transition.

        Args:
            dev (Device): The device which is currently measured.
            qubits (list): List of qubits to be calibrated. FIXME: currently
                only one qubit is supported. E.g., qubits = [qb1].

        Keyword Arguments:
            autorun (bool): Whether to run the routine automatically
            rabi_amps (list): List of amplitudes for the (initial) Rabi
                experiment
            ramsey_delays (list): List of delays for the (initial) Ramsey
                experiment
            delta_t (float): Time duration for (initial) Ramsey
            t0 (float): Time for (initial) Ramsey
            n_points (int): Number of points per period for the (initial) Ramsey
            pts_per_period (int): Number of points per period to use for the
                (initial) Ramsey
            artificial_detuning (float): Artificial detuning for the initial
                Ramsey experiment


        Configuration parameters (coming from the configuration parameter
        dictionary):
            transition_name (str): Transition to be calibrated
            adaptive (bool): Whether to use adaptive rabi and ramsey settings
            allowed_difference (float): Allowed frequency difference in Hz
                between old and new frequency (convergence criterion)
            max_iterations (int): Maximum number of iterations
            autorun (bool): Whether to run the routine automatically
            f_factor (float): Factor to multiply the frequency by (only relevant
                for displaying results)
            f_unit (str): Unit of the frequency (only relevant for displaying
                results)
            delta_f_factor (float): Factor to multiply the frequency difference
                by (only relevant for displaying results)
            delta_f_unit (str): Unit of the frequency difference (only relevant
                for displaying results)

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

        # Defining initial and allowed frequency difference
        self.delta_f = np.Infinity
        self.iteration = 1

        self.final_init(**kw)

    class Decision(IntermediateStep):

        def __init__(self, routine, index, **kw):
            """Decision step that decides to add another round of Rabi-Ramsey to
            the FindFrequency routine based on the difference between the
            results of the previous and current Ramsey experiments.
            Additionally, it checks if the maximum number of iterations has been
            reached.

            Args:
                routine (Step): FindFrequency routine
                index (int): Index of the decision step (necessary to find the
                    position of the Ramsey measurement in the routine)

           Configuration parameters (coming from the configuration parameter
           dictionary):
                max_waiting_seconds (float): maximum number of seconds to wait
                    for the results of the previous Ramsey experiment to arrive.
            """
            super().__init__(routine=routine, index=index, **kw)
            # FIXME: use general parameters from FindFrequency for now
            self.parameter_lookups = self.routine.parameter_lookups
            self.parameter_sublookups = self.routine.parameter_sublookups
            self.leaf = self.routine.leaf

        def run(self):
            """Executes the decision step.
            """
            qubit = self.qubit

            routine = self.routine
            index = self.kw.get("index")

            # Saving some typing for parameters that are only read ;)
            allowed_delta_f = self.get_param_value("allowed_delta_f")
            f_unit = self.get_param_value("f_unit")
            f_factor = self.get_param_value("f_factor")
            delta_f_unit = self.get_param_value("delta_f_unit")
            delta_f_factor = self.get_param_value("delta_f_factor")
            max_iterations = self.get_param_value("max_iterations")
            transition = self.get_param_value("transition_name")

            # Finding the ramsey experiment in the pipulse calibration
            pipulse_calib = routine.routine_steps[index - 1]
            ramsey = pipulse_calib.routine_steps[-1]

            # Transition frequency from last Ramsey
            freq = qubit[f"{transition}_freq"]()

            # Retrieving the frequency difference
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
                    # FIXME: Unsure if this can also happen on real set-up
                    log.warning(
                        "Analysis not yet run on last Ramsey measurement, "
                        "frequency difference not updated")
                    time.sleep(1)

            # Progress update
            if self.get_param_value('verbose'):
                print(f"Iteration {routine.iteration}, {transition}-freq "
                      f"{freq/f_factor} {f_unit}, frequency "
                      f"difference = {routine.delta_f/delta_f_factor} "
                      f"{delta_f_unit}")

            # Check if the absolute frequency difference is small enough
            if np.abs(routine.delta_f) < allowed_delta_f:
                # Success
                if self.get_param_value('verbose'):
                    print(f"{transition}-frequency found to be"
                          f"{freq/f_factor} {f_unit} within "
                          f"{allowed_delta_f/delta_f_factor} "
                          f"{delta_f_unit} of previous value.")

            elif routine.iteration < max_iterations:
                # No success yet, adding a new rabi-ramsey and decision step
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
                # No success yet, reached max iterations
                msg = (f"FindFrequency routine finished for {qubit.name}, "
                       "desired precision not necessarily achieved within the "
                       f"maximum number of iterations ({max_iterations}).")
                log.warning(msg)

                if self.get_param_value('verbose'):
                    print(msg)

            if self.get_param_value('verbose'):
                # Printing termination update
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
    """

    def __init__(self, dev, **kw):
        """Initialize the SingleQubitCalib routine.

        Args:
            dev (Device): Device to be used for the routine.

        Keyword args:
            qubits (list): The qubits which should be calibrated. By default,
                all qubits of the device are selected.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            autorun (bool): If True, the routine runs automatically. Otherwise,
                run() has to called.
            transition_names (list): List of transitions for which the steps in
                the routine should be conducted. Example:
                    "transition_names": ["ge"]
            rabi (bool): Enables the respective step. Enabled by default.
            ramsey_large_AD (bool): Enables the respective step.
            ramsey_small_AD (bool): Enables the respective step. Enabled by
                default.
            rabi_after_ramsey (bool): Enables the respective step.
            qscale (bool): Enables the respective step. Enabled by default.
            rabi_after_qscale (bool): Enables the respective step.
            t1 (bool): Enables the respective step.
            echo_large_AD (bool): Enables the respective step.
            echo_small_AD (bool): Enables the respective step.
            in_phase_calib (bool): Enables the respective step.

            nr_rabis (dict): nested dictionary containing the transitions and a
                respective list containing numbers of Rabi pulses. For each
                entry in the list, the Rabi experiment is repeated with the
                respective number of pulses.
                Example:
                    "nr_rabis": {
                        "ge": [1,3],
                        "ef": [1]
                    },
        """
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
                step_label = step[1]
                step_settings = copy.deepcopy(step[2])
                try:
                    step_tmp_settings = step[3]
                except IndexError:
                    step_tmp_settings = []

                if issubclass(step_class, SingleQubitGateCalibExperiment):
                    if 'qscale' in step_label or 'echo' in step_label:
                        if transition_name != 'ge':
                            continue
                    # Check whether the step is supposed to be included
                    if not self.get_param_value(step_label):
                        continue

                    update_nested_dictionary(
                        step_settings['settings'], {
                            step_class.get_lookup_class().__name__: {
                                'transition_name': transition_name
                            }
                        })

                    if issubclass(step_class, qbcal.Rabi):
                        for n in self.get_param_value(
                                'nr_rabis')[transition_name]:
                            update_nested_dictionary(
                                step_settings['settings'], {
                                    step_class.get_lookup_class().__name__: {
                                        'n': n
                                    }
                                })

                    sublookups = [
                        f"{step_label}_{transition_name}", step_label,
                        step_class.get_lookup_class().__name__
                    ]

                    new_step_settings = self.extract_step_settings(
                        step_class, step_label, sublookups=sublookups)
                    update_nested_dictionary(step_settings['settings'],
                                             new_step_settings)

                    step_label = step_label + "_" + transition_name

                detailed_routine_template.add_step(step_class, step_label,
                                                   step_settings,
                                                   step_tmp_settings)

        self.routine_template = detailed_routine_template

        # Loop in reverse order so that the correspondence between the index
        # of the loop and the index of the routine_template steps is preserved
        # when new steps are added
        for i, step in reversed(list(enumerate(self.routine_template))):
            self.split_step_for_parallel_groups(index=i)

    class SQCPreparation(IntermediateStep):
        """Intermediate step that configures qubits for Mux drive and
        readout.
        """

        def __init__(
            self,
            routine,
            **kw,
        ):
            """Initialize the SQCPreparation step.

            Args:
                routine (Step): the parent routine.

            Configuration parameters (coming from the configuration parameter
            dictionary):
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