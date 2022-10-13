from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, AutomaticCalibrationRoutine, update_nested_dictionary)
from pycqed.measurement.calibration.automatic_calibration_routines.base. \
    base_automatic_calibration_routine import (_device_db_client_module_missing,
                                               keyword_subset_for_function)
from pycqed.measurement.calibration.automatic_calibration_routines import (
    routines_utils, AdaptiveReparkingRamsey, UpdateFrequency, SetBiasVoltage)

if not _device_db_client_module_missing:
    from pycqed.utilities.devicedb import utils as db_utils

from pycqed.measurement.calibration.automatic_calibration_routines.\
    single_qubit_routines import FindFrequency

from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals

import numpy as np
import logging
from typing import Dict, Tuple, Any, List, Literal, Optional, Union
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.instrument_drivers.meta_instrument.device import Device
from dataclasses import dataclass

log = logging.getLogger('Routines')


@dataclass
class QubitParkingResults:
    # Store results for a single qubit
    flux: float = None
    initial_voltage: float = None
    measured_voltage: float = None
    initial_ge_freq: float = None
    measured_ge_freq: float = None


class QubitParking(AutomaticCalibrationRoutine):
    """Qubit parking for a single qubit, to better calibrate a sweet spot
    ge_frequency and voltage.

    This routine usually acts as a step in the :obj:`MultiQubitParking` routine,
     where decision steps are present as well (even if only one qubit is used).

     The routine steps are:
        1) :obj:`SetBiasVoltage`.
        2) :obj:`UpdateFrequency`
        3) :obj:`FindFrequency`
        4) :obj:`AdaptiveReparkingRamsey`

     """

    def __init__(
            self,
            dev: Device,
            qubit: QuDev_transmon,
            fluxlines_dict: Dict[str, Any],
            flux: Optional[Union[float, Literal['{designated}',
                                                '{opposite}',
                                                '{mid}']]] = None,
            **kw,
    ):
        """
        Args:
            dev: Device.
            qubit: qubit to be parked.
            fluxlines_dict: dictionary with the qubit names as keys and Qcodes
                parameters of their flux line as value.

        Keyword Args:
            Additional keyword arguments that will be transferred to
            :obj:`AutomaticCalibrationRoutine`

        """
        kw.pop('qubits', None)  # Do not pass 'qubits' twice
        super().__init__(
            dev=dev,
            qubits=[qubit],
            fluxlines_dict=fluxlines_dict,
            **kw,
        )

        # Routine attributes
        self.fluxlines_dict = fluxlines_dict
        routines_utils.append_DCsources(self)

        self.qubit = qubit
        self.iteration_index = 1

        # Store initial values so that the user can retrieve them if overwritten
        self.results: Dict[str, QubitParkingResults] = {}
        flux, initial_voltage = routines_utils.get_qubit_flux_and_voltage(
            qb=qubit,
            fluxlines_dict=self.fluxlines_dict,
            flux=flux or self.get_param_value("flux", qubit=qubit.name),
            voltage=self.get_param_value("voltage", qubit=qubit.name)
        )
        self.results[qubit.name] = QubitParkingResults(
            **dict(initial_voltage=initial_voltage, flux=flux))

        if flux != qubit.flux_parking() or not qubit.ge_freq():
            transmon_freq_model = \
                routines_utils.get_transmon_freq_model(qubit)
            updated_frequency = qubit.calculate_frequency(
                flux=flux, model=transmon_freq_model)
            qubit.ge_freq(updated_frequency)
            self.results[qubit.name].initial_ge_freq = updated_frequency
            self.settings[type(self).__name__]['General']['update'] = False
        else:
            self.results[qubit.name].initial_ge_freq = qubit.ge_freq()

        self.final_init(**kw)

    def create_routine_template(self):
        """Creates the routine template for the QubitParking routine using
        the specified parameters.
        """
        super().create_routine_template()  # Create empty routine template
        qb = self.qubit

        # Setting bias voltage to the guess voltage
        step_label = f'set_bias_voltage'
        set_voltage = self.results[qb.name].initial_voltage
        step_settings = {"qubit": self.qubit,
                         "transition": 'ge',
                         "voltage": set_voltage}
        self.add_step(SetBiasVoltage, step_label, step_settings)

        # Updating ge-frequency at this voltage to guess value
        step_label = f'update_frequency'
        step_settings = {"qubits": [self.qubit],
                         "transition": 'ge',
                         "frequencies": [self.results[qb.name].initial_ge_freq]}
        self.add_step(UpdateFrequency, step_label, step_settings)

        # Finding the ge-transition frequency at this voltage
        step_label = 'find_frequency'
        self.add_step(
            FindFrequency,
            step_label,
            step_settings={},
            step_tmp_vals=ro_flux_tmp_vals(qb=qb,
                                           v_park=set_voltage,
                                           use_ro_flux=True),
        )

        # Reparking Ramsey
        self.add_step(
            AdaptiveReparkingRamsey,
            'adaptive_reparking_ramsey',
            {"fluxlines_dict": self.fluxlines_dict},
            step_tmp_vals=ro_flux_tmp_vals(qb=qb,
                                           v_park=set_voltage,
                                           use_ro_flux=True),
        )

    def post_run(self):
        qb = self.qubit
        last_results = self.routine_steps[-1].results[qb.name]
        self.results[qb.name].measured_ge_freq = last_results['ss_freq']
        self.results[qb.name].measured_voltage = last_results['ss_volt']

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


class MultiQubitParking(AutomaticCalibrationRoutine):
    """Routine to park several (or one) qubits at a sweet spot.
    """

    def __init__(
            self,
            dev: Device,
            fluxlines_dict: Dict[str, Any],
            qubits: List[QuDev_transmon],
            **kw,
    ):
        """
        Args:
            dev (Device): Device to be used for the routine
            fluxlines_dict (dict): dictionary containing the qubits names as
                keys and the flux lines QCoDeS parameters as values.
            qubits (list): The qubits which should be calibrated. By default,
                all qubits of the device are selected.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            max_delta (float): Convergence threshold. Maximum allowed frequency
                difference between two consecutive qubit parkings series.
            max_iterations (int, default 3): Maximum number of iterations that
                will be performed if the parking fails.
        """
        super().__init__(
            dev=dev,
            qubits=qubits,
            **(kw | {'fluxlines_dict': fluxlines_dict}),
        )
        self.fluxlines_dict = fluxlines_dict
        self.index_iteration = 1

        for qb in qubits:
            current_ge_freq = qb.ge_freq()
            current_voltage = fluxlines_dict[qb.name]()
            self.results[qb.name] = QubitParkingResults(**dict(
                initial_ge_freq=current_ge_freq,
                measured_ge_freq=current_ge_freq,
                initial_voltage=current_voltage,
                measured_voltage=current_voltage
            ))

        routines_utils.append_DCsources(self)

        self.final_init(**kw)

    class Decision(IntermediateStep):
        """Decision step that decides to add another reparking Ramsey step if
            the reparking Ramsey experiment found a fit value for the sweet-spot
            voltage that is outside the range of the swept voltages.
        """

        def __init__(self,
                     routine: AutomaticCalibrationRoutine,
                     max_delta: float = 0.1e6,
                     max_iterations: int = 3,
                     **kw):
            """Initialize the Decision step.

            Args:
                routine (Step): AdaptiveReparkingRamsey routine.
                max_delta (float): The maximum allowed frequency difference.
                    Above this threshold the routine will be repeated.
                max_iterations (int) The maximum number of iterations for
                    the routine.

            Keyword Args:
                kw: Arguments that will be passes to :obj:`IntermediateStep`
            """
            super().__init__(routine=routine, **kw)
            self.max_delta = max_delta
            self.max_iterations = max_iterations

        def run(self):
            """Executes the decision step.
            """
            routine: MultiQubitParking = self.routine
            success = True
            for qb_step_index, qb in enumerate(self.qubits, -len(self.qubits)):
                qb_res = routine.routine_steps[qb_step_index].results[qb.name]
                previous_res = self.routine.results[qb.name]
                delta_freq = np.abs(qb_res.measured_ge_freq -
                                    previous_res.measured_ge_freq)
                if delta_freq > self.max_delta:
                    log.info(f"The frequency difference for {qb.name} is "
                             f"{delta_freq / 1e6:.2f} MHz, which is higher "
                             f"than the allowed threshold (="
                             f"{self.max_delta / 1e6:.2f} MHz).")
                    success = False

            if not success:
                if self.routine.index_iteration < self.max_iterations:
                    self.routine.add_qubits_parkings_and_decision()

                self.routine.index_iteration += 1

    def create_routine_template(self):
        super().create_routine_template()  # Create empty routine template
        self.add_qubits_parkings_and_decision()

    def add_qubits_parkings_and_decision(self):
        """Add a series of qubit parkings for all the qubits"""
        # Add qubit parkings for all qubits
        for qb in self.qubits:
            flux, _ = routines_utils.get_qubit_flux_and_voltage(
                qb=qb,
                fluxlines_dict=self.fluxlines_dict,
                flux=self.get_param_value('flux', qubit=qb,
                                          default='{designated}'),
                voltage=self.get_param_value('voltage', qubit=qb)
            )
            step_label = f'{qb.name}_parking_iteration_{self.index_iteration}'
            step_settings = {"qubit": qb,
                             "fluxlines_dict": self.fluxlines_dict,
                             "flux": flux}
            self.add_step(QubitParking, step_label, step_settings)

        # Add decision step
        step_label = f'decision_iteration_{self.index_iteration}'
        step_settings = {"max_delta": self.get_param_value("max_delta"),
                         "max_iterations": self.get_param_value(
                             "max_iterations", default=5)}
        self.add_step(self.Decision, step_label, step_settings)
