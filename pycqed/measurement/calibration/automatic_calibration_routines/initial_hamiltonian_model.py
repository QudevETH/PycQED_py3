from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, AutomaticCalibrationRoutine)

from .park_and_qubit_spectroscopy import ParkAndQubitSpectroscopy

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

    # Parameter that can be extracted from design
    E_c: float = None

    # Parameters that will be estimated during the routine
    Ej_max: float = None
    asymmetry: float = None


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

        self.qubits_to_skip = []
        self.results = {}
        for qubit in self.qubits:
            existing_hamiltonian_params = qubit.fit_ge_freq_from_dc_offset()
            if 'E_c' not in existing_hamiltonian_params:
                existing_hamiltonian_params['E_c'] = self.extract_qubit_E_c(
                    qubit)
            if all([k in existing_hamiltonian_params for k in
                    ['Ej_max', 'asymmetry']]):
                log.info(f"Ej_max and asymmetry values already exist in the "
                         f"attributes of qubit {qubit}. The routine will skip "
                         f"it.")
                self.qubits_to_skip.append(qubit)

            self.results[qubit.name] = QubitHamiltonianParameters(
                **existing_hamiltonian_params)

        self.final_init(**kw)

    @staticmethod
    def extract_qubit_E_c(qubit: QuDev_transmon) -> float:
        # TODO Implement this method to give a meaningful value! (from the
        #  design DB?)
        log.warning("Implement the `extract_qubit_E_c()` method to give a"
                    "meaningful value!")
        return 0.2e9

    def create_initial_routine(self, load_parameters=True):
        super().create_routine_template()  # Create empty routine template
        qubit: QuDev_transmon = self.qubit
        uss_flux = 0
        lss_flux = -0.5 * np.sign(qubit.calculate_voltage_from_flux(flux=0.0))
        for flux in [uss_flux, lss_flux]:
            # Add park and spectroscopy step
            step_label = f'park_and_qubit_spectroscopy_flux_{flux}'
            step_settings = {'fluxlines_dict': self.fluxlines_dict,
                             'settings': {
                                 step_label: {'General': {'flux': flux}}}}
            self.add_step(ParkAndQubitSpectroscopy,
                          step_label=step_label,
                          step_settings=step_settings)

            # Add model update step
            step_label = f'update_hamiltonian_model_flux_{flux}'
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
            self.routine: Optional[PopulateInitialHamiltonianModel] = None
            super().__init__(**kw)
            self.flux = flux

        def run(self):
            for qubit in self.qubits:
                if qubit in self.routine.qubits_to_skip:
                    continue

                qubit_parameters: QubitHamiltonianParameters = \
                    self.routine.results[qubit.name]
                qubit_park_and_spec: ParkAndQubitSpectroscopy = \
                    self.routine.routine_steps[-1]
                ge_freq = qubit_park_and_spec.results[
                    qubit.name].measured_ge_freq
                E_c = qubit_parameters.E_c
                if self.flux == 0:
                    Ej_max = np.square(ge_freq + E_c) / (8 * E_c)
                    qubit_parameters.Ej_max = Ej_max
                else:
                    assert (Ej_max := qubit_parameters.Ej_max) is not None, (
                        f'The calculation of the asymmetry (d) relies on '
                        f'Ej_max, which should be evaluated at Phi=0.')
                    d = np.square(ge_freq + E_c) / (8 * E_c * Ej_max)
                    qubit_parameters.asymmetry = d

                    # Update the qubit with the final parameters
                    qubit.fit_ge_freq_from_dc_offset(asdict(qubit_parameters))
