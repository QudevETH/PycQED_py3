from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, RoutineTemplate, AutomaticCalibrationRoutine)
from pycqed.measurement.calibration.automatic_calibration_routines.base.\
    base_automatic_calibration_routine import (_device_db_client_module_missing)

if not _device_db_client_module_missing:
    pass

from .single_qubit_routines import AdaptiveQubitSpectroscopy

from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals
import logging
import numpy as np
from typing import List, Dict
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from dataclasses import dataclass

log = logging.getLogger('Routines')


@dataclass
class ParkAndQubitSpectroscopyResults:
    # Store results for a single qubit
    initial_ge_freq: float = None
    initial_flux: float = None
    initial_voltage: float = None

    measured_ge_freq: float = None
    measured_flux: float = None
    measured_voltage: float = None


class ParkAndQubitSpectroscopy(AutomaticCalibrationRoutine):
    """AutomaticRoutine that parks a qubit at the specified spot where it
    performs an AdaptiveQubitSpectroscopy routine to find its ge_freq.

    The initial and measured values for ge_freqs, fluxes, and voltages can be
    retrieved from the `results` attribute.

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

    def __init__(
            self,
            dev,
            qubits: List[QuDev_transmon],
            fluxlines_dict,
            **kw,
    ):
        """Initializes the ParkAndQubitSpectroscopy routine.
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
        super().__init__(
            dev=dev,
            qubits=qubits,
            fluxlines_dict=fluxlines_dict,
            **kw,
        )

        # Routine attributes
        self.fluxlines_dict = fluxlines_dict
        # Retrieve the DCSources from the fluxlines_dict. These are necessary
        # to reload the pre-routine settings when update=False
        self.DCSources = []
        for qb in self.qubits:
            dc_source = self.fluxlines_dict[qb.name].instrument
            if dc_source not in self.DCSources:
                self.DCSources.append(dc_source)

        # Store initial values so that the user can retrieve them if overwritten
        self.results: Dict[str, ParkAndQubitSpectroscopyResults] = {}
        for qb in self.qubits:
            uss = qb.fit_ge_freq_from_dc_offset()['dac_sweet_spot']
            V_per_phi0 = qb.fit_ge_freq_from_dc_offset()['V_per_phi0']
            self.results[qb.name] = ParkAndQubitSpectroscopyResults(**dict(
                initial_ge_freq=qb.ge_freq(),
                initial_voltage=self.fluxlines_dict[qb.name](),
                initial_flux=(self.fluxlines_dict[qb.name]() - uss) / V_per_phi0
            ))

        self.final_init(**kw)

    def create_routine_template(self):
        """Creates routine template."""
        super().create_routine_template()
        # Loop in reverse order so that the correspondence between the index
        # of the loop and the index of the routine_template steps is preserved
        # when new steps are added
        for i, step in reversed(list(enumerate(self.routine_template))):
            self.split_step_for_parallel_groups(index=i)

    class SetBiasVoltageAndFluxPulseAssistedReadOut(IntermediateStep):
        """Intermediate step that updates the bias voltage of the qubit and the
        temporary values of the following AdaptiveQubitSpectroscopy for
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
            for qb in self.qubits:
                designated_ss_flux = qb.flux_parking()
                designated_ss_volt = qb.calculate_voltage_from_flux(
                    designated_ss_flux)
                if designated_ss_flux == 0:
                    # Qubit parked at the USS.
                    # LSS will be with opposite voltage sign
                    opposite_ss_flux = -0.5 * np.sign(designated_ss_volt)
                elif np.abs(designated_ss_flux) == 0.5:
                    # Qubit parked at the LSS
                    opposite_ss_flux = 0
                else:
                    raise ValueError("Only Sweet Spots are supported!")
                mid_flux = (designated_ss_flux + opposite_ss_flux) / 2

                # Allow user to specify flux using the following strings:
                # "{designated}", "{opposite}", and "{mid}"
                flux = self.get_param_value("flux", qubit=qb.name)
                if isinstance(flux, str):
                    flux = eval(
                        flux.format(designated=designated_ss_flux,
                                    opposite=opposite_ss_flux,
                                    mid=mid_flux))

                if flux is not None:
                    voltage = qb.calculate_voltage_from_flux(flux)
                else:
                    voltage = self.get_param_value("voltage", qubit=qb.name)
                    uss = qb.fit_ge_freq_from_dc_offset()['dac_sweet_spot']
                    V_per_phi0 = qb.fit_ge_freq_from_dc_offset()['V_per_phi0']
                    flux = (self.routine.fluxlines_dict[
                                qb.name]() - uss) / V_per_phi0
                if voltage is None:
                    raise ValueError("No voltage or flux specified")
                self.routine.results[qb.name].measured_flux = flux
                self.routine.results[qb.name].measured_voltage = voltage

                # Temporary values for ro
                ro_tmp_vals = ro_flux_tmp_vals(qb, voltage, use_ro_flux=True)
                # Extending temporary values
                self.routine.extend_step_tmp_vals_at_index(tmp_vals=ro_tmp_vals,
                                                           index=1)
                if self.get_param_value("verbose"):
                    log.info(f"Setting {qb.name} voltage bias to {voltage} V. "
                             f"Corresponding flux: {flux} Phi0")

                self.routine.fluxlines_dict[qb.name](voltage)

    class StoreMeasuredValues(IntermediateStep):
        """Stores the current ge_freq of the measured qubits in a dictionary of
        the parent routine, so that the user can retrieve them even if the
        initial values are restored.
        """
        def run(self):
            for qb in self.qubits:
                self.routine.results[qb.name].measured_ge_freq = qb.ge_freq()

    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
        [SetBiasVoltageAndFluxPulseAssistedReadOut,
         'set_bias_voltage_and_fp_assisted_ro', {}],
        [AdaptiveQubitSpectroscopy, 'adaptive_qubit_spectroscopy', {}],
        [StoreMeasuredValues, 'store_measured_values', {}]
    ])
