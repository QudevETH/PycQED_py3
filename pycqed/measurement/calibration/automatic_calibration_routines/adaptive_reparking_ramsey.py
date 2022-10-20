from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    Step,
    IntermediateStep,
    AutomaticCalibrationRoutine
)

from pycqed.measurement.calibration.automatic_calibration_routines.base. \
    base_automatic_calibration_routine import _device_db_client_module_missing

if not _device_db_client_module_missing:
    from pycqed.utilities.devicedb import utils as db_utils

from pycqed.measurement.sweep_points import SweepPoints
from pycqed.measurement.calibration import single_qubit_gates as qbcal
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.device import Device
from pycqed.measurement.calibration.automatic_calibration_routines import (
    routines_utils, ROUTINES)

import numpy as np
import logging
from typing import Dict, List, Any

log = logging.getLogger(ROUTINES)


class ReparkingRamseyStep(qbcal.ReparkingRamsey, Step):
    """A wrapper class for the ReparkingRamsey experiment."""

    def __init__(self, routine,
                 fluxlines_dict: Dict[str, Any],
                 qubits: List[QuDev_transmon] = None,
                 **kwargs):
        """
        Initializes the ReparkingRamseyStep class, which also includes
        initialization of the ReparkingRamsey experiment.

        Arguments:
            routine (AdaptiveReparkingRamsey): The parent routine.
            fluxlines_dict (dict): dictionary containing the qubits names as
                keys and the flux lines QCoDeS parameters as values.
            qubits (list): List of qubits to be used in the routine. If None is
                given the step will acquire the qubits of the parent routine.

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
        self.qubits = qubits or self.routine.qubits
        Step.__init__(self, routine=routine, qubits=self.qubits, **kwargs)
        self.fluxlines_dict = fluxlines_dict

        settings = self.parse_settings(self.get_requested_settings())
        qbcal.ReparkingRamsey.__init__(self, dev=self.dev, **settings)

    def parse_settings(self, requested_kwargs):
        """
        Searches the keywords for the ReparkingRamsey experiment given in
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
            #  duplication?
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
            sweep_volts = dc_voltage_offsets + self.fluxlines_dict[qb.name]()
            self.results[qb.name] = {'sweep_volts': sweep_volts}

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
                task['fluxline'] = self.fluxlines_dict[qb.name]

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
        """
        Runs the Ramsey experiment and the analysis for it.
        """
        self.run_measurement()
        self.run_analysis()
        if self.get_param_value('update'):
            self.run_update()
    
    def post_run(self):
        for qb in self.qubits:
            if self.analysis.fit_uss:
                # Update the 'dac_sweet_spot' attribute such that the found new
                # voltage will correspond to the designated flux
                dac_sweet_spot = self.fluxlines_dict[qb.name]()
                qb.fit_ge_freq_from_dc_offset()[
                    'dac_sweet_spot'] = dac_sweet_spot
                self.results[qb.name]['dac_sweet_spot'] = dac_sweet_spot
                log.info(f"{qb.name} 'dac_sweet_spot' value updated to "
                         f"{dac_sweet_spot} V")

            else:
                if (np.sign((lss_voltage := self.fluxlines_dict[qb.name]()) -
                            (uss_voltage := qb.fit_ge_freq_from_dc_offset()[
                                'dac_sweet_spot']))
                        != np.sign(lss_flux := qb.flux_parking())):
                    log.critical("The measured lower sweet spot does not "
                                 f"correspond to {qb.name} designated sweet"
                                 f" spot ({lss_flux})")

                # Update the 'V_per_phi0' attribute such that the found new
                # voltage will correspond to the designated flux
                V_per_phi0 = 2 * np.abs(uss_voltage - lss_voltage)
                qb.fit_ge_freq_from_dc_offset()['V_per_phi0'] = V_per_phi0
                self.results[qb.name]['V_per_phi0'] = V_per_phi0
                log.info(f"{qb.name} 'V_per_phi0' value updated to "
                         f"{V_per_phi0}")


class AdaptiveReparkingRamsey(AutomaticCalibrationRoutine):
    """
    Routine to find the sweet-spot of a qubit using Ramsey experiments at
    different voltages.
    A series of Ramsey experiments is performed. A Decision step decides
    whether the found sweet spot is outside the swept voltages range, and if so
    another series of iterations is run.
    Notice that this is the only criterion for now, and the fit itself is not
    taken into consideration.

    For example, a routine with up to 3 iterations can have the following
    routine steps:

    1) ReparkingRamseyStep (reparking_ramsey): Performs a reparking Ramsey and
        fits the result with a Parabola to extract the voltage of the
        sweet-spot.
    2) Decision (decision_reparking_ramsey): Checks whether the sweet spot found
        according to the fit is inside the swept voltage range.
        If the fit was not successful for certain qubits, another reparking
        Ramsey is run with the found voltage as the new sweep center (only for
        the unsuccessful qubits), followed by another Decision step.
        For example:

        3) ReparkingRamseyStep (reparking_ramsey_repetition_<k>):
        4) Decision (decision_reparking_ramsey_repetition_<k>):
        5) ReparkingRamseyStep (reparking_ramsey_repetition_<k+1>):
        6) Decision (decision_reparking_ramsey_repetition_<k+1>):
        (...)

        Additional steps will be added until the maximum number of repetitions
        (specified in the configuration parameter dictionary as
        "max_iterations", default is 3) is reached.

        The settings of these additional steps can be set manually by specifying
        the settings for each step in the configuration parameter dictionary
        (using the unique label of each step)

    Notes:
        When the `dc_voltage_offsets` are specified, the center voltage will be
        the current one, so the user must make sure that it is indeed
        corresponding to the aimed flux point.
    """

    def __init__(
            self,
            dev: Device,
            fluxlines_dict: Dict[str, Any],
            qubits: List[QuDev_transmon],
            **kw,
    ):
        """
        Initialize the AdaptiveReparkingRamsey routine.

        Args:
            dev (Device): Device to be used for the routine
            fluxlines_dict (dict): dictionary containing the qubits names as
                keys and the flux lines QCoDeS parameters as values.
            qubits (list): The qubits which should be calibrated. By default,
                all qubits of the device are selected.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            max_iterations (int, default 3): Maximum number of iterations that
                will be performed if a spectroscopy fails.
        """

        super().__init__(
            dev=dev,
            qubits=qubits,
            **(kw | {'fluxlines_dict': fluxlines_dict}),
        )
        self.fluxlines_dict = fluxlines_dict
        self.index_iteration = 1

        for qb in qubits:
            self.results[qb.name] = dict(
                previous_freq=qb.ge_freq(),
                previous_volt=fluxlines_dict[qb.name]()
            )

        routines_utils.append_DCsources(self)

        self.final_init(**kw)

    class Decision(IntermediateStep):
        """
        Decision step that decides to add another reparking Ramsey step if
        the reparking Ramsey experiment found a fit value for the sweet-spot
        voltage that is outside the range of the swept voltages.
        """

        def __init__(self, routine: AutomaticCalibrationRoutine, **kw):
            """Initialize the Decision step.

            Args:
                routine (Step): AdaptiveReparkingRamsey routine.

            Kwargs:
                Arguments that will be passes to :obj:`IntermediateStep`.
            """
            super().__init__(routine=routine, **kw)
            self.precision = 1e-4  # 0.1 mV. Used for comparing QDAC values.

        def run(self):
            """Executes the decision step."""
            routine: AdaptiveReparkingRamsey = self.routine
            qubits_to_rerun = []
            qubits_failed = []
            for qb in self.qubits:
                max_iterations = self.get_param_value("max_iterations",
                                                      qubit=qb,
                                                      default=3)
                # Retrieve the ReparkingRamseyStep run last
                reparking_ramsey: ReparkingRamseyStep = \
                    routine.routine_steps[-1]

                # Retrieve the necessary data from the analysis results
                swept_voltages = reparking_ramsey.results[qb.name][
                    'sweep_volts']
                fit_voltage = self.routine.fluxlines_dict[qb.name]()

                min_swept_voltage = np.min(swept_voltages)
                max_swept_voltage = np.max(swept_voltages)
                if any([np.isclose(fit_voltage, range_edge, rtol=0,
                                   atol=self.precision) for range_edge in
                        [min_swept_voltage, max_swept_voltage]]):
                    fail_message = f'Extremum voltage of fit outside range.'
                    success = False
                elif min_swept_voltage < fit_voltage < max_swept_voltage:
                    success = True
                else:
                    # Bug in the setting of voltage of the qubit.
                    # ReparkingRamsey should have set it to one of the range
                    # edges.
                    log.warning("The set voltage value is outside the given"
                                "voltage range")
                    success = False

                if success:
                    if self.get_param_value('verbose'):
                        print(f"Reparking Ramsey for {qb.name} was successful.")
                elif routine.index_iteration < max_iterations:
                    if self.get_param_value('verbose'):
                        print(f"Reparking Ramsey failed for {qb.name} since "
                              f"{fail_message}. Trying again.")
                    qubits_to_rerun.append(qb)
                else:
                    if self.get_param_value('verbose'):
                        print(f"Reparking Ramsey failed for {qb.name}. since "
                              f"{fail_message}. Maximum number of iterations "
                              f"reached.")
                    qubits_failed.append(qb)

            # Decide how to modify the routine steps
            if len(qubits_to_rerun) > 0:
                # If there are some qubits whose reparking ramsey should be
                # rerun, add another ReparkingRamseyStep followed by another
                # Decision step
                routine.index_iteration += 1
                routine.add_rerun_reparking_ramsey_step(
                    index_iteration=routine.index_iteration,
                    qubits=qubits_to_rerun)

    def create_routine_template(self):
        """
        Creates the routine template for the AdaptiveReparkingRamsey
        routine.
        """
        super().create_routine_template()

        # Add the requested number of ReparkingRamseyStep followed by
        # Decision steps
        reparking_ramsey_settings = {'qubits': self.qubits,
                                     'fluxlines_dict': self.fluxlines_dict}
        self.add_step(ReparkingRamseyStep,
                      f'reparking_ramsey',
                      reparking_ramsey_settings)
        decision_settings = {'qubits': self.qubits}
        self.add_step(self.Decision, f'decision_reparking_ramsey',
                      decision_settings)

    def add_rerun_reparking_ramsey_step(self,
                                        index_iteration: int,
                                        qubits: List[QuDev_transmon]):
        """
        Adds a next ReparkingRamseyStep followed by a Decision step

        Args:
            index_iteration (int): Index of the iteration for the spectroscopy
                with index `index_spectroscopy`.
            qubits (list): List of qubits (QuDev_transmon objects) whose
                reparking ramsey should be run again.
        """

        # Add the steps right after the current one
        self.add_step(
            ReparkingRamseyStep,
            f'reparking_ramsey_repetition_{index_iteration}',
            {
                'qubits': qubits,
                'fluxlines_dict': self.fluxlines_dict
            },
            index=self.current_step_index + 1,
        )
        self.add_step(
            self.Decision,
            f'decision_reparking_ramsey_repetition_{index_iteration}',
            {
                'qubits': qubits
            },
            index=self.current_step_index + 2
        )

    def post_run(self):
        for qb in self.qubits:
            self.results[qb.name].update({
                "ss_freq": qb.ge_freq(),
                "ss_volt": self.fluxlines_dict[qb.name](),
                "iterations_needed": self.index_iteration
            })

            if self.routine is None:
                if not routines_utils.qb_is_at_designated_sweet_spot(
                        qb=qb,
                        fluxlines_dict=self.fluxlines_dict):
                    # Do not update if the qubit is not at its designated sweet
                    # spot and the routine is not a step of a bigger routine
                    log.warning("The routine results will not update since this"
                                " is not the designated sweet spot")
                    self.settings[self.step_label]["General"]["update"] = False

        super().post_run()
