"""Generic intermediate steps that are used in several routines"""

from pycqed.measurement.calibration.automatic_calibration_routines.base import \
    IntermediateStep
from pycqed.measurement.calibration.automatic_calibration_routines import \
    routines_utils
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from typing import Literal


class UpdateFrequency(IntermediateStep):
    """Updates the frequency of the specified transition at a specified
        flux and voltage bias.
    """

    def __init__(self,
                 transition: Literal['ge', 'ef'],
                 qubit: QuDev_transmon = None,
                 frequency: float = None,
                 voltage: float = None,
                 flux: float = None,
                 **kw):
        """Initialize the UpdateFrequency step.

        Args:
            transition: The frequency will be updated for this transition.
            qubit: The qubit whose flux line needs to be set. If None the qubit
                will be taken from the parent routine.
            frequency (float): Frequency to which the qubit should be set.
                transition_name (str): Transition to be updated.
            voltage (float): the dac voltage to which the qubit frequency
                should correspond. Useful if the frequency is not known a
                priori and there is a Hamiltonian model or a
                flux-frequency relationship is known.
            flux (float): Flux (in units of Phi0) to which the qubit
                frequency should correspond. Useful if the frequency is not
                known a priori and there is a Hamiltonian model or a
                flux-frequency relationship is known.

        Keyword Arguments:
            routine (Step): the routine to which this step belongs to.
        """
        super().__init__(**kw, )

        # Transition and frequency
        self.transition = transition
        self.qubit = qubit or self.routine.qubit
        self.frequency = frequency
        self.voltage = voltage
        self.flux = flux

        if self.frequency is None:
            assert any([self.flux is not None,
                        self.voltage is not None]), \
                f"Could not calculate the frequency of transition " \
                f"{self.transition} without flux or voltage specified."

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
                freq_model = routines_utils.get_transmon_freq_model(qb)
                try:
                    frequency = qb.calculate_frequency(
                        model=freq_model,
                        flux=self.flux,
                        bias=self.voltage,
                        transition=self.transition,
                    )
                except NotImplementedError:
                    assert (self.transition == "ef")
                    frequency = (
                            qb.ge_freq() +
                            routines_utils.get_transmon_anharmonicity(
                                qb))

        if self.get_param_value('verbose'):
            print(f"{self.transition}-frequency updated to {frequency:0f} Hz")

        qb[f"{self.transition}_freq"](frequency)


class SetBiasVoltage(IntermediateStep):
    """Intermediate step that updates the bias voltage of the qubit.
    This can be done by simply specifying the voltage, or by specifying
    the flux. If the flux is given, the corresponding bias is calculated
    using the Hamiltonian model stored in the qubit object.
    """

    def __init__(self,
                 qubit: QuDev_transmon = None,
                 voltage: float = None,
                 flux: float = None,
                 **kw):
        """Initialize the SetBiasVoltage step.

        Args:
            qubit: The qubit whose flux line needs to be set. If None the qubit
                will be taken from the parent routine.
            voltage: The set value for the flux line of the qubit. If None the
                value will be calculated with from the given flux.
            flux: If voltage is None this value will

        Keyword Arguments:
            routine (Step): the parent routine to which this step belongs.
        """

        super().__init__(**kw)
        self.qubit = qubit or self.routine.qubit
        self.voltage = voltage
        self.flux = flux

    def run(self):
        qb = self.qubit
        if self.flux is not None:
            self.voltage = qb.calculate_voltage_from_flux(self.flux)
        else:
            if self.voltage is None:
                raise ValueError("No voltage or flux specified")
            self.routine.fluxlines_dict[qb.name](self.voltage)
