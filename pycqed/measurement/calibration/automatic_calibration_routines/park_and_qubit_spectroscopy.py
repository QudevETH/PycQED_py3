from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, RoutineTemplate, AutomaticCalibrationRoutine)
from pycqed.measurement.calibration.automatic_calibration_routines import \
    AdaptiveQubitSpectroscopy, UpdateFrequency
from pycqed.measurement.calibration.automatic_calibration_routines.base.\
    base_automatic_calibration_routine import _device_db_client_module_missing
from pycqed.measurement.calibration.automatic_calibration_routines \
    import routines_utils, ROUTINES

if not _device_db_client_module_missing:
    pass

from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals
import logging
import numpy as np
from typing import List, Dict, Tuple
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from dataclasses import dataclass

log = logging.getLogger(ROUTINES)


@dataclass
class ParkAndQubitSpectroscopyResults:
    # Store results for a single qubit
    flux: float = None
    voltage: float = None
    initial_ge_freq: float = None
    measured_ge_freq: float = None


class ParkAndQubitSpectroscopy(AutomaticCalibrationRoutine):
    """
    AutomaticRoutine that parks a qubit at the specified spot where it
    performs an AdaptiveQubitSpectroscopy routine to find its ge_freq.

    The flux and voltage, together with initial and measured values for ge_freq,
    can be retrieved from the `results` attribute.
    In the case where the requested parking point is different from the
    designated one (`qubit.flux_parking()`):
     1. The initial qubit frequency will be calculated according to the flux.
     2. The measured qubit frequency will not be updated at the end of
    the routine.


    Examples::

        settings_user = {
            'ParkAndQubitSpectroscopy': {'General': {
                'flux': '{designated}'}},
            'AdaptiveQubitSpectroscopy': {'General': {'n_spectroscopies': 1,
                                                      'max_iterations': 2}},
            'QubitSpectroscopy1D': {'pts': 500}
        }

        park_and_qubit_spectroscopy = ParkAndQubitSpectroscopy(dev=dev,
                                            fluxlines_dict=fluxlines_dict,
                                            settings_user=settings_user,
                                            qubits=[qb1, qb6],
                                            autorun=False)
        park_and_qubit_spectroscopy.view()
        park_and_qubit_spectroscopy.run()
    """

    def __init__(self,
                 dev,
                 qubits: List[QuDev_transmon],
                 fluxlines_dict,
                 **kw):
        """
        Initializes the ParkAndQubitSpectroscopy routine.
        The fluxes and/or voltages at which the AdaptiveQubitSpectroscopies are
        run are specified with the settings of
        SetBiasVoltageAndFluxPulseAssistedReadOut. If no settings are specified
        there, it is possible to use the 'General' scope of
        ParkAndQubitSpectroscopy. In this case (i.e., no settings are specified
        for SetBiasVoltageAndFluxPulseAssistedReadOut), "flux" or "voltage" can
        be specified as a keyword argument of ParkAndQubitSpectroscopy.

        Args:
            dev (Device): Device that is being measured.
            qubits (list[QuDev_transmon]): List of qubits that should be
                measured.
            fluxlines_dict (dict): fluxlines_dict object for accessing and
                changing the dac voltages.
            **kw: keyword arguments that will be passed to the `__init__()` and
                `final_init()` functions of :obj:`AutomaticCalibrationRoutine`.
        """
        super().__init__(dev=dev,
                         qubits=qubits,
                         fluxlines_dict=fluxlines_dict,
                         **kw)

        # Routine attributes
        self.fluxlines_dict = fluxlines_dict
        routines_utils.append_DCsources(self)

        # Store initial values so that the user can retrieve them if overwritten
        self.results: Dict[str, ParkAndQubitSpectroscopyResults] = {}
        for qb in self.qubits:
            flux, voltage = routines_utils.get_qubit_flux_and_voltage(
                qb=qb,
                fluxlines_dict=self.fluxlines_dict,
                flux=self.get_param_value("flux", qubit=qb.name),
                voltage=self.get_param_value("voltage", qubit=qb.name)
            )
            self.results[qb.name] = ParkAndQubitSpectroscopyResults(
                **dict(voltage=voltage, flux=flux))

            self.qubits_frequencies = []

            if flux != qb.flux_parking() or not qb.ge_freq():
                transmon_freq_model = \
                    routines_utils.get_transmon_freq_model(qb)
                updated_frequency = qb.calculate_frequency(
                    flux=flux, model=transmon_freq_model)
            else:
                updated_frequency = qb.ge_freq()

            if flux != qb.flux_parking():
                self.settings[self.step_label]['General']['update'] = False

            self.results[qb.name].initial_ge_freq = updated_frequency
            self.qubits_frequencies.append(updated_frequency)

        self._DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
            [self.SetBiasVoltageAndFluxPulseAssistedReadOut,
             'set_bias_voltage_and_fp_assisted_ro', {}],
            [UpdateFrequency, 'update_frequency',
             {'transition': 'ge', 'frequencies': self.qubits_frequencies}],
            [AdaptiveQubitSpectroscopy, 'adaptive_qubit_spectroscopy', {}]
        ])

        self.final_init(**kw)

    def create_routine_template(self):
        """Creates routine template."""
        super().create_routine_template()
        # Loop in reverse order so that the correspondence between the index
        # of the loop and the index of the routine_template steps is preserved
        # when new steps are added
        for i, step in reversed(list(enumerate(self.routine_template))):
            self.split_step_for_parallel_groups(index=i)

    def post_run(self):
        """Save the results of the routine."""
        for qb in self.qubits:
            self.results[qb.name].measured_ge_freq = qb.ge_freq()

        # Do not update if the qubit is not at its designated sweet
        # spot and the routine is not a step of a bigger routine
        if self.routine is None:
            for qb in self.qubits:
                if qb.flux_parking() != self.results[qb.name].flux:

                    log.warning(f"The routine results will not be updated since"
                                f" {qb.name} was not measured at its "
                                f"designated sweet spot")
                    self.settings[self.step_label]["General"]["update"] = False

        super().post_run()

    class SetBiasVoltageAndFluxPulseAssistedReadOut(IntermediateStep):
        """
        Intermediate step that updates the bias voltage of the qubit and the
        temporary values of the following :obj:`AdaptiveQubitSpectroscopy` for
        flux-pulse-assisted RO.

        It is possible to specify the voltage or the flux. If the flux is given,
        the corresponding bias is calculated using the Hamiltonian model stored
        in the qubit object.

        The fluxes and voltages at which the measurement are done can be
        retrieved from the parent routine's results attribute.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            flux (float or str): Flux at which the qubit should be parked. The
                corresponding voltage is calculated using
                qb.calculate_voltage_from_flux(flux).
                It is possible to specify the following strings as well:
                - "{designated}", for the designated sweet spot.
                - "{opposite}", for the opposite sweet spot.
                - "{mid}", for the mid-point between the designated sweet spot
                    and the opposite sweet spot.
            voltage (float): Voltage at which the qubits should be parked. This
                is used only if flux is not specified.

        NOTE: qubit-specific settings can be specified using the "qubits"
        keyword. For example:
        "qubits":{
            "qb1":{
                "flux": 0.5
            },
            "qb2":{
                "flux": 0.25
            }
        }

        TODO: The purpose of this class is similar to SetBiasVoltage (of
         HamiltonianFitting) and SetTemporaryValuesFluxPulseReadOut. However,
         this intermediate step has some additional features (e.g., possibility
         of specifying strings as settings and the fact that it saves the fluxes
         and voltages in a dictionary). It could be worth writing a generic
         intermediate step that sets a bias voltage and the temporary values for
         FP-assisted RO.
        """

        def run(self):
            """Execute the step."""
            for qb in self.qubits:
                flux = self.routine.results[qb.name].flux
                voltage = self.routine.results[qb.name].voltage

                # Temporary values for ro
                ro_tmp_vals = ro_flux_tmp_vals(qb, voltage, use_ro_flux=True)
                # Extending temporary values for qubit spectroscopy
                index = 2
                assert (self.routine.routine_template[index][0].__name__ ==
                        AdaptiveQubitSpectroscopy.__name__)
                self.routine.extend_step_tmp_vals_at_index(tmp_vals=ro_tmp_vals,
                                                           index=index)
                log.info(f"Setting {qb.name} voltage bias to {voltage:.6f} V. "
                         f"Corresponding flux: {flux} Phi0")

                # Set the voltage on the corresponding flux line
                self.routine.fluxlines_dict[qb.name](voltage)
