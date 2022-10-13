from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, AutomaticCalibrationRoutine)
from pycqed.measurement.calibration.automatic_calibration_routines import \
    routines_utils

from .park_and_qubit_spectroscopy import (ParkAndQubitSpectroscopy,
                                          ParkAndQubitSpectroscopyResults)

import numpy as np
import logging
from typing import List, Any, Dict, Optional
from pycqed.instrument_drivers.meta_instrument.device import Device
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from dataclasses import dataclass, asdict

log = logging.getLogger('Routines')


@dataclass
class QubitHamiltonianParameters:
    # Parameters that must be provided using, e.g., `InitialQubitParking`
    dac_sweet_spot: float
    V_per_phi0: float

    # Parameter that need to be extracted from design
    E_c: float = None

    # Parameters that will be estimated during the routine
    Ej_max: float = None
    asymmetry: float = None

    # Optional parameters that might exist for the qubit
    fr: float = 0
    coupling: float = 0


class PopulateInitialHamiltonianModel(AutomaticCalibrationRoutine):
    r"""This routine populates a first guess for the Hamiltonian model of a
    transmon.

    It is meant to run after the `InitialQubitParking` routine and before the
    full `HamiltonianFitting` routine.
    In practice, it measures the transmon ge-frequency at the two sweet-spots,
    and uses these values together with the design E_c value to give a first
    estimation of the EJ_max and asymmetry (d) values of the transmon. These
    values are updated in the qubit's `fit_ge_freq_from_dc_offset()` dictionary,
    as well as in the `results` attribute of the routine.

    Following Koch et al. 2017 (DOI: 10.1103/PhysRevA.76.042319, Eqs. 2.11 and
    2.18), the following equations are used:
    .. math::
        \Phi = 0:
            hf_{ge} = -E_c + \sqrt{8 E_c  E_{J,max}} \\
        \Phi = \pm \frac{1}{2} \Phi_0:
            hf_{ge} = -E_c + \sqrt{8 d E_c E_{J,max}}

    Examples::

        settings_user = {
            'ParkAndQubitSpectroscopy': {'General': {
                'flux': '{designated}'}},
            'AdaptiveQubitSpectroscopy': {'General': {'n_spectroscopies': 1,
                                                      'max_iterations': 2}},
            'QubitSpectroscopy1D': {'pts': 500}
        }

        initial_hamiltonian_model = PopulateInitialHamiltonianModel(dev=dev,
                                            fluxlines_dict=fluxlines_dict,
                                            settings_user=settings_user,
                                            qubits=[qb1, qb6],
                                            autorun=False)
        initial_hamiltonian_model.view()
        initial_hamiltonian_model.run()
    """

    def __init__(self,
                 dev: Device,
                 qubits: List[QuDev_transmon],
                 fluxlines_dict: Dict[str, Any],
                 **kw,
                 ):
        """
        Args:
            dev: The device that is measured.
            qubits: List of qubit instances that will be measured.
            fluxlines_dict:  Dict with the qubit names as keys and a
                `qcodes.Parameter` that sets the flux that is applied on the
                corresponding qubit.
            **kw: keyword arguments that will be passed to the `__init__()` and
                `final_init()` functions of :obj:`AutomaticCalibrationRoutine`.

        Configuration parameters:
            park_qubit_after_routine: If True the qubit flux and
                frequency will be set to its parking sweet-spot.
        """

        super().__init__(
            dev=dev,
            qubits=qubits,
            fluxlines_dict=fluxlines_dict,
            **kw,
        )

        self.fluxlines_dict = fluxlines_dict
        # Retrieve the DCSources from the fluxlines_dict. These are necessary
        # to reload the pre-routine settings when update=False
        self.DCSources = []
        for qb in self.qubits:
            dc_source = self.fluxlines_dict[qb.name].instrument
            if dc_source not in self.DCSources:
                self.DCSources.append(dc_source)

        self.results = {}
        for qubit in self.qubits:
            existing_hamiltonian_params = qubit.fit_ge_freq_from_dc_offset()
            if 'E_c' not in existing_hamiltonian_params:
                existing_hamiltonian_params['E_c'] = self.extract_qubit_E_c(
                    qubit)

            self.results[qubit.name] = QubitHamiltonianParameters(
                **existing_hamiltonian_params)
        self.final_init(**kw)

    @staticmethod
    def extract_qubit_E_c(qubit: QuDev_transmon) -> float:
        # TODO Implement this method to give a meaningful value! (from the
        #  design DB?)
        log.warning("Implement the `extract_qubit_E_c()` method to give a"
                    "meaningful value!")
        return 165e6

    @staticmethod
    def get_park_and_spectroscopy_step_label(qubit: QuDev_transmon,
                                             flux: float):
        return f'park_and_qubit_{qubit.name}_spectroscopy_flux_{float(flux)}'

    def create_initial_routine(self, load_parameters=True):
        super().create_routine_template()  # Create empty routine template
        for qubit in self.qubits:
            designated_flux = routines_utils.flux_to_float(
                qubit, '{designated}')
            opposite_flux = routines_utils.flux_to_float(qubit, '{opposite}')
            for flux in [designated_flux, opposite_flux]:
                # Add park and spectroscopy step
                step_label = self.get_park_and_spectroscopy_step_label(
                    qubit=qubit, flux=flux)
                step_settings = {'fluxlines_dict': self.fluxlines_dict,
                                 'settings': {
                                     step_label: {'General': {'flux': flux}}}}
                self.add_step(ParkAndQubitSpectroscopy,
                              step_label=step_label,
                              step_settings=step_settings)

                # Add model update step
                step_label = f'update_hamiltonian_model_{qubit.name}_' \
                             f'flux_{flux}'
                step_settings = {'flux': flux}
                self.add_step(self.UpdateHamiltonianModel,
                              step_label=step_label,
                              step_settings=step_settings)

    class UpdateHamiltonianModel(IntermediateStep):
        """Updates the parameters of the initial Hamiltonian with the values
         extracted from the last step.
        """

        def __init__(self,
                     flux: float,
                     **kw):
            """

            Args:
                flux: The flux at which the qubit was measured, in Phi0 unit.

            Keyword Args:
                Keyword arguments that will be passed to the `__init__` function
                of :obj:`IntermediateStep`.
            """
            self.routine: Optional[PopulateInitialHamiltonianModel] = None
            super().__init__(**kw)
            self.flux = flux

        def run(self):
            for qubit in self.qubits:
                qubit_parameters: QubitHamiltonianParameters = \
                    self.routine.results[qubit.name]
                qb_results: ParkAndQubitSpectroscopyResults = \
                    self.routine.routine_steps[-1].results[qubit.name]
                E_c = qubit_parameters.E_c
                if self.flux == 0:
                    Ej_max = (np.square(qb_results.measured_ge_freq + E_c) /
                              (8 * E_c))
                    qubit_parameters.Ej_max = Ej_max
                else:
                    assert (Ej_max := qubit_parameters.Ej_max) is not None, (
                        f'The calculation of the asymmetry (d) relies on '
                        f'Ej_max.')
                    d = (np.square(qb_results.measured_ge_freq + E_c) /
                         (8 * E_c * Ej_max))
                    qubit_parameters.asymmetry = d

                    # Update the qubit with the final parameters
                    qubit.fit_ge_freq_from_dc_offset(asdict(qubit_parameters))

    def post_run(self):
        for qubit in self.qubits:
            park_qubit_after_routine = self.get_param_value(
                "park_qubit_after_routine", qubit=qubit.name)
            if park_qubit_after_routine:
                # Repark the qubit at its sweet-spot
                updated_qb_results = None
                for i, step in enumerate(self.routine_steps):
                    if step.step_label == \
                            self.get_park_and_spectroscopy_step_label(
                                qubit=qubit, flux=qubit.flux_parking()):
                        updated_qb_results = step.results[qubit.name]
                        break

                if updated_qb_results is None:
                    log.warning(f"Could not find the correct step to load the"
                                f"updated parameters for {qubit.name}")
                else:
                    voltage = updated_qb_results.voltage
                    ge_freq = updated_qb_results.measured_ge_freq
                    log.info(f"Updating {qubit.name} to ge_frequency {ge_freq} "
                             f"Hz and voltage {voltage} V")
                    self.fluxlines_dict[qubit.name](voltage)
                    qubit.ge_freq(ge_freq)
