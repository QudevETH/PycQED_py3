"""Generic intermediate steps that are used in several routines"""

from pycqed.measurement.calibration.automatic_calibration_routines.base import \
    IntermediateStep
from pycqed.measurement.calibration.automatic_calibration_routines import \
    routines_utils


class UpdateFrequency(IntermediateStep):
    """Updates the frequency of the specified transition at a specified
        flux and voltage bias.
    """

    def __init__(self, **kw):
        """Initialize the UpdateFrequency step.

        Keyword Arguments:
            routine (Step): the routine to which this step belongs to.

        Configuration parameters (coming from the configuration parameter
         dictionary):
            frequency (float): Frequency to which the qubit should be set.
            transition_name (str): Transition to be updated.
            use_prior_model (bool): If True, the frequency is updated using
                the Hamiltonian model. If False, the frequency is updated
                using the specified frequency.
            flux (float): Flux (in units of Phi0) to which the qubit
                frequency should correspond. Useful if the frequency is not
                known a priori and there is a Hamiltonian model or a
                flux-frequency relationship is known.
            voltage (float): the dac voltage to which the qubit frequency
                should correspond. Useful if the frequency is not known a
                priori and there is a Hamiltonian model or a
                flux-frequency relationship is known.

        FIXME: the flux-frequency relationship stored in
         flux_to_voltage_and_freq should/could be used here.
        """
        super().__init__(**kw, )

        # Transition and frequency
        self.transition = self.get_param_value(
            'transition_name')  # default is "ge"
        self.frequency = self.get_param_value(
            'frequency')  # default is None
        self.flux = self.get_param_value('flux')
        self.voltage = self.get_param_value('voltage')

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
                # A (possibly preliminary) Hamiltonian model exists
                if (self.get_param_value('use_prior_model') and
                        len(qb.fit_ge_freq_from_dc_offset()) > 0):
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

                # No Hamiltonian model exists, but we want to know the ef-
                # frequency when the ge-frequency is known at this flux and
                # there is a guess for the anharmonicity

                # FIXME: instead of qb.ge_freq() we should use the frequency
                #  stored in the flux_to_voltage_and_freq dictionary. The
                #  current implementation assumes that the ef measurement is
                #  preceded by the ge-measurement at the same voltage (and
                #  thus flux).
                elif self.transition == "ef":
                    frequency = (
                            qb.ge_freq() +
                            routines_utils.get_transmon_anharmonicity(
                                qb))

                # No Hamiltonian model exists
                else:
                    raise NotImplementedError(
                        "Can not estimate frequency with incomplete model, "
                        "make sure to save a (possibly preliminary) model "
                        "first")

        if self.get_param_value('verbose'):
            print(f"{self.transition}-frequency updated to ", frequency,
                  "Hz")

        qb[f"{self.transition}_freq"](frequency)


class SetBiasVoltage(IntermediateStep):
    """Intermediate step that updates the bias voltage of the qubit.
    This can be done by simply specifying the voltage, or by specifying
    the flux. If the flux is given, the corresponding bias is calculated
    using the Hamiltonian model stored in the qubit object.
    """

    def __init__(self, **kw):
        """Initialize the SetBiasVoltage step.

        Keyword Arguments:
            routine (Step): the routine to which this step belongs to.

        Configuration parameters (coming from the configuration parameter
         dictionary):
            flux (float): Flux (in units of Phi0) at which the qubit should
                be parked. The voltage is calculated from this value
                using qb.calculate_voltage_from_flux(flux).
            voltage (float): Voltage bias at which the qubit should be
                parked. This is used only if the flux is not specified.
        """
        super().__init__(**kw)

    def run(self):
        for qb in self.qubits:
            flux = self.get_param_value("flux", qubit=qb.name)
            if flux is not None:
                voltage = qb.calculate_voltage_from_flux(flux)
            else:
                voltage = self.get_param_value("voltage", qubit=qb.name)
            if voltage is None:
                raise ValueError("No voltage or flux specified")
            self.routine.fluxlines_dict[qb.name](voltage)
