from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    IntermediateStep, AutomaticCalibrationRoutine, update_nested_dictionary,
    ROUTINES)
from pycqed.measurement.calibration.automatic_calibration_routines.base. \
    base_automatic_calibration_routine import (_device_db_client_module_missing,
                                               keyword_subset_for_function)
from pycqed.measurement.calibration.automatic_calibration_routines import (
    routines_utils, AdaptiveReparkingRamsey, UpdateFrequency, SetBiasVoltage)

if not _device_db_client_module_missing:
    from pycqed.utilities.devicedb import utils as db_utils

from pycqed.measurement.calibration.automatic_calibration_routines.\
    single_qubit_routines import FindFrequency, RabiStep

from pycqed.utilities import hamiltonian_fitting_analysis as hfa
from pycqed.utilities.state_and_transition_translation import *
from pycqed.utilities.general import temporary_value
from pycqed.utilities.flux_assisted_readout import ro_flux_tmp_vals

import pycqed.analysis.analysis_toolbox as a_tools
import numpy as np
import logging
from typing import Dict, Tuple
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon

log = logging.getLogger(ROUTINES)


class HamiltonianFitting(AutomaticCalibrationRoutine,
                         hfa.HamiltonianFittingAnalysis):
    """
    Constructs a HamiltonianFitting routine used to determine a
    Hamiltonian model for a transmon qubit.

    Routine steps:

    1) SetBiasVoltage (set_bias_voltage_<i>): sets the bias voltage at either
        the USS or LSS.
    2) UpdateFrequency (update_frequency_<tr_name>_<i>): updates the frequency
        of the qubit at current bias voltage. The bias voltage is calculated
        from the previous Hamiltonian model (if given), otherwise the guessed
        one will be used.
    3) :obj:`FindFrequency` (find_frequency_<tr_name>_<i>): see corresponding
        routine.
    4) :obj:`AdaptiveReparkingRamsey` (reparking_ramsey_<i>): see corresponding
        routine.
    5) :obj:`RabiStep` (rabi_step_ge): see corresponding step.

    Steps 1), 2), 3), and 4) are repeated for the ge transition for both the
    upper and the lower sweet spot.
    Step 5) is done only for the designated sweet spot.
    Steps 2) and 3) are repeated also for the ef transition at the designated
    sweet spot.

    6) DetermineModel (determine_model_preliminary): fits a Hamiltonian
        model based on the three transition frequencies
        i) |g⟩ ↔ |e⟩ (USS)
        ii) |g⟩ ↔ |e⟩ (LSS)
        iii) |e⟩ ↔ |f⟩ (USS).

    For the remaining fluxes and transitions specified in "measurements" the
    following step will be run:
        7) SetBiasVoltage (set_bias_voltage_<i>): sets the bias voltage at the
            midpoint.
        8) UpdateFrequency (update_frequency_ge_<i>): updates the frequency of
            the qubit at the midpoint by using the preliminary Hamiltonian
            model.
        9) SetTemporaryValuesFluxPulseReadout (set_tmp_values_flux_pulse_ro_ge):
            sets temporary bias voltage for flux-pulse-assisted readout.
        10) FindFrequency (find_frequency_ge_<i>): see corresponding routine.

    Afterwards, the final model is determined:
    11) DetermineModel (determine_model_final): fits a Hamiltonian model based
        on the transition frequencies at the fluxes specified in "measurements".
        By default, they are:
        i) |g⟩ ↔ |e⟩ (USS)
        ii) |g⟩ ↔ |e⟩ (LSS)
        iii) |e⟩ ↔ |f⟩ (USS)
        iv) |g⟩ ↔ |e⟩ (midpoint).
    """

    def __init__(self,
                 dev,
                 qubit,
                 fluxlines_dict,
                 **kw):
        """
        Constructs a HamiltonianFitting routine used to determine a
        Hamiltonian model for a transmon qubit.

        Args:
            qubit (QuDev_transmon): qubit to perform the calibration on.
                NOTE: Only supports one qubit input.
            fluxlines_dict (dict): fluxlines_dict object for accessing and
                changing the dac voltages (needed in reparking ramsey).
            measurements (dict): dictionary containing the fluxes and
                transitions to measure tp determine the Hamiltonian model. See
                default below for an example. The measurements must
                include at least two sweet spots (meaning the corresponding
                flux should be divisible by 0.5*phi0), which should be the first
                two keys of the dictionary. If None, a default measurement
                dictionary is chosen containing ge-frequency measurements at the
                zeroth upper sweet spot (phi = 0*phi0), the left lower sweet
                spot (phi = -0.5*phi0) and the mid voltage (phi = -0.25*phi0).
                Additionally the ef-frequency is measured at the upper sweet
                spot.

                default:
                {"{designated}": ["ge", "ef"],
                 "{opposite}": ["ge"],
                 "{mid}": ["ge"]}

            flux_to_voltage_and_freq_guess (dict(float, tuple(float, float))):
                Guessed values for voltage and ge-frequencies for the fluxes in
                the measurement dictionary in case no Hamiltonian model is
                available yet. If passed, the dictionary must include guesses
                for the first two fluxes of `measurements` (which should be
                sweet spots). Default is None, in which case it is required to
                have use_prior_model==True. This means that the guesses will
                automatically be calculated with the existing model.

                Note, for successful execution of the routine either a
                `flux_to_voltage_and_freq_guess` must be passed or a
                flux-voltage model must be specified in the qubit object.

                For example, `flux_to_voltage_and_freq_guess = {0:(-0.5,
                5.8e9), -0.5:(-4.1, 4.20e9)}` designates measurements at two
                sweet spots; one at flux = 0 (zeroth uss) and one at
                flux = -0.5 (left lss), both in units of phi0. The
                guessed voltages are -0.5V and -4.1V, respectively. The guessed
                frequencies are the last entry of the tuple: 5.8 GHz at flux = 0
                and 4.2 GHz at flux = -0.5 (the first left lss)
            save_instrument_settings (bool): boolean to designate if instrument
                settings should be saved before and after this (sub)routine

        Configuration parameters (coming from the configuration parameter
        dictionary):
            use_prior_model (bool): whether to use the prior model (stored in
                qubit object) to determine the guess frequencies and voltages
                for the flux values of the measurement. If True,
                flux_to_voltage_and_freq_guess will automatically be generated
                using this model (and can be left None).
            delegate_plotting (bool): whether to delegate plotting to the
                `plotting` module. The variable is stored in the global settings
                and the default value is False.
            method (str): optimization method to use. Default is Nelder-
                Mead.
            include_mixer_calib (bool): If True, include mixer
                calibration for both carrier and skewness.
            mixer_calib_carrier_settings (bool): Settings for the mixer
                calibration for the carrier.
            mixer_calib_skewness_settings (bool): Settings for the mixer
                calibration for the skewness.
            get_parameters_from_qubit_object (bool): if True, the routine will
                try to get the parameters from the qubit object. Default is
                False.

        FIXME: There needs to be a general framework allowing the user to decide
         which optional steps should be included (e.g., mixer calib, qb spec,
         FindFreq before ReparkingRamsey, etc.).

        FIXME: The routine relies on fp-assisted read-out which assumes
         that the qubit object has a model containing the dac_sweet_spot and
         V_per_phi0. While these values are not strictly needed, they are needed
         for the current implementation of the ro_flux_tmp_vals. This is
         detrimental for the routine if use_prior_model = False and the qubit
         doesn't contain a prior model.

        FIXME: Remove the dependence on `use_prior_model`, which should always
         be True. The preliminary values before any measurement can be estimated
         from the design values.
        """

        super().__init__(
            dev=dev,
            qubits=[qubit],
            fluxlines_dict=fluxlines_dict,
            **kw,
        )

        use_prior_model = self.get_param_value('use_prior_model')
        measurements = self.get_param_value('measurements')
        flux_to_voltage_and_freq_guess = self.get_param_value(
            'flux_to_voltage_and_freq_guess')

        if not use_prior_model:
            assert (flux_to_voltage_and_freq_guess is not None
                    ), "If use_prior_model is False, " \
                       "flux_to_voltage_and_freq_guess must be specified."

        # Flux to voltage and frequency dictionaries (guessed and measured)
        self.flux_to_voltage_and_freq = {}
        if flux_to_voltage_and_freq_guess is not None:
            self.flux_to_voltage_and_freq_guess: Dict[float, Tuple[
                float, float]] = flux_to_voltage_and_freq_guess.copy()

        # Routine attributes
        self.fluxlines_dict = fluxlines_dict
        routines_utils.append_DCsources(self)

        self.measurements = {
            routines_utils.flux_to_float(qb=self.qubit, flux=k):
                tuple(transition_to_str(t) for t in v) for k, v in
            measurements.items()
        }

        # Validity of measurement dictionary
        for flux, transitions in self.measurements.items():
            for t in transitions:
                if t not in ["ge", "ef"]:
                    raise ValueError(
                        "Measurements must only include 'ge' and "
                        "'ef' (transitions). Currently, other transitions are "
                        "not supported.")

        # Guesses for voltage and frequency
        # determine sweet spot fluxes

        self.ss1_flux, self.ss2_flux, *rest = self.measurements.keys()
        assert self.ss1_flux % 0.5 == 0 and self.ss2_flux % 0.5 == 0, (
            "First entries of the measurement dictionary must be sweet spots"
            "(flux must be half integer)")

        # If `use_prior_model` is True, generate guess from prior Hamiltonian
        # model
        if use_prior_model:
            freq_model = routines_utils.get_transmon_freq_model(qubit)
            # Use measured values for designated sweet spot or estimated ones
            # for other fluxes
            self.flux_to_voltage_and_freq_guess = {
                flux: (self.fluxlines_dict[qubit.name](),
                       qubit.ge_freq()) if flux == qubit.flux_parking() else (
                    qubit.calculate_voltage_from_flux(flux=flux),
                    qubit.calculate_frequency(model=freq_model,
                                              flux=flux))
                for flux in [self.ss1_flux, self.ss2_flux]
            }

        # Validity of flux_to_voltage_and_freq_guess dictionary
        x = set(self.flux_to_voltage_and_freq_guess.keys())
        y = set(self.measurements.keys())
        z = {self.ss1_flux, self.ss2_flux}

        assert x.issubset(y), (
            "Fluxes in flux_to_voltage_and_freq_guess must be a subset of the "
            "fluxes in measurements")

        self.other_fluxes_with_guess = list(x - z)
        self.fluxes_without_guess = list(y - x)

        if self.get_param_value("get_parameters_from_qubit_object",
                                default=False):
            update_nested_dictionary(
                self.settings,
                {self.highest_lookup: {
                    "General": self.parameters_qubit
                }})

        self.results[self.qubit.name] = {}
        self.final_init(**kw)

    def create_routine_template(self):
        """
        Creates the routine template for the HamiltonianFitting routine using
        the specified parameters.
        """
        super().create_routine_template()  # Create empty routine template
        qubit = self.qubit

        # # Counter for mixer calibration steps
        # n_update_frequency = 0

        # Measurements at fluxes with provided guess voltage/freq
        for i, flux in enumerate([
            self.ss1_flux,
            self.ss2_flux,
            *self.other_fluxes_with_guess,
        ], 1):
            # Getting the guess voltage and frequency
            voltage_guess, ge_freq_guess = self.flux_to_voltage_and_freq_guess[
                flux]

            # Setting bias voltage to the guess voltage
            step_label = f'set_bias_voltage_{i}'
            step_settings = {"qubit": qubit,
                             "voltage": voltage_guess}
            self.add_step(SetBiasVoltage, step_label, step_settings)

            for transition in self.measurements[flux]:
                # ge-transitions
                if transition == "ge":

                    step_label = f'update_frequency_{transition}_{i}'
                    # Updating ge-frequency at this voltage to guess value
                    step_settings = {"qubits": [qubit],
                                     "frequencies": [ge_freq_guess],
                                     "transition": transition}
                    self.add_step(UpdateFrequency, step_label, step_settings)
                    # n_update_frequency += 1

                    # Finding the ge-transition frequency at this voltage
                    step_label = f'find_frequency_{transition}_{i}'
                    find_freq_settings = {
                        'settings': {
                            step_label: {
                                "General": {
                                    "transition_name": transition,
                                }
                            }
                        }
                    }
                    self.add_step(
                        FindFrequency,
                        step_label,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                       v_park=voltage_guess,
                                                       use_ro_flux=True),
                    )

                    # If this flux is one of the two sweet spot fluxes, we also
                    # perform a Reparking Ramsey and update the flux-voltage
                    # relation stored in routine.
                    if flux in [self.ss1_flux, self.ss2_flux]:
                        self.add_step(
                            AdaptiveReparkingRamsey,
                            'reparking_ramsey_' +
                            ('1' if flux == self.ss1_flux else '2'),
                            {"fluxlines_dict": self.fluxlines_dict},
                            step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                           v_park=voltage_guess,
                                                           use_ro_flux=True),
                        )

                        # Updating voltage to flux with ReparkingRamsey results
                        step_label = 'update_flux_to_voltage'
                        self.add_step(
                            self.UpdateFluxToVoltage,
                            step_label,
                            {
                                "index_reparking":
                                    # len(self.routine_template) - 1 + 2*n_update_frequency,
                                    len(self.routine_template) - 1,
                            "settings": {
                                    step_label: {
                                        "flux": flux
                                    }
                                },
                            },
                        )

                elif transition == "ef":
                    # First run a ge-Rabi measurement
                    step_label = f'rabi_step_ge'
                    step_settings = {
                        "qubits": [self.qubit],
                        'settings': {
                            step_label: {
                                "transition_name": 'ge',
                            }
                        }
                    }
                    self.add_step(RabiStep, step_label, step_settings)

                    # Updating ef-frequency at this voltage to guess value

                    step_label = f'update_frequency_{transition}'
                    step_settings = {"qubits": [qubit],
                                     "fluxes": [flux],
                                     "transition": transition,
                                     "use_model": False}
                    self.add_step(UpdateFrequency, step_label, step_settings)
                    # n_update_frequency += 1

                    # Finding the ef-frequency
                    step_label = f"find_frequency_{transition}"
                    find_freq_settings = {
                        'settings': {
                            step_label: {
                                "General": {
                                    "transition_name": transition,
                                }
                            }
                        }
                    }
                    self.add_step(
                        FindFrequency,
                        step_label,
                        find_freq_settings,
                        step_tmp_vals=ro_flux_tmp_vals(qubit,
                                                       v_park=voltage_guess,
                                                       use_ro_flux=True),
                    )

        if not self.get_param_value("use_prior_model"):
            # Determining preliminary model - based on partial data and routine
            # parameters (e.g. fabrication parameters or user input)
            step_label = 'determine_model_preliminary'
            settings = {
                step_label: {
                    # should never be True
                    "include_resonator": self.get_param_value('use_prior_model')
                }
            }
            step_settings = {'settings': settings}
            self.add_step(self.DetermineModel, step_label, step_settings)

        # Measurements at other flux values (using preliminary or prior model)
        for i, flux in enumerate(self.fluxes_without_guess,
                                 len(self.other_fluxes_with_guess) + 2):
            # Updating bias voltage using earlier reparking measurements
            step_label = f'set_bias_voltage_{i}'
            self.add_step(SetBiasVoltage, step_label,
                          step_settings={"qubit": qubit,
                                         "flux": flux})

            # Looping over all transitions
            for transition in self.measurements[flux]:
                # Updating transition frequency of the qubit object to the value
                # calculated by prior or preliminary model
                step_label = f'update_frequency_{transition}_{i}'
                step_settings = {"qubits": [qubit],
                                 "fluxes": [flux],
                                 "transition": transition}
                self.add_step(UpdateFrequency, step_label, step_settings)
                # n_update_frequency += 1

                # Set temporary values for Find Frequency
                step_label = 'set_tmp_values_flux_pulse_ro_' + \
                             transition + '_' + str(i)
                settings = {step_label: {"flux_park": flux, }}
                set_tmp_vals_settings = {
                    "settings": settings,
                    "index": len(self.routine_template) + 1,
                }
                self.add_step(
                    SetTemporaryValuesFluxPulseReadOut,
                    step_label,
                    set_tmp_vals_settings,
                )

                # Finding frequency
                step_label = 'find_frequency_' + transition + '_' + str(i)
                self.add_step(
                    FindFrequency,
                    step_label,
                    {"settings": {
                        step_label: {
                            'transition_name': transition
                        }
                    }}
                )

        # Determining final model based on all data
        self.add_step(self.DetermineModel, 'determine_model_final', {})

        # # Mixer calibration
        # self.add_mixer_calib_steps(**self.kw)

    def post_run(self):
        """
        Park the qubit at its designated sweet spot after the routine is over.
        """
        qb = self.qubit

        # Setting bias voltage
        voltage = qb.calculate_voltage_from_flux(flux := qb.flux_parking())
        self.fluxlines_dict[qb.name](voltage)
        log.info(f'{qb.name} updated with voltage {voltage} V, corresponding to'
                 f' flux {flux}')

        # Updating ge-frequency at this voltage to guess value
        qb.ge_freq(ge_freq := qb.calculate_frequency(flux=flux))
        log.info(f'{qb.name} updated with ge-frequency {ge_freq} Hz.')

        AutomaticCalibrationRoutine.post_run(self)

    # def add_mixer_calib_steps(self, **kw):
    #     """
    #     Add steps to calibrate the mixer after the rest of the routine is
    #     defined. Mixer calibrations are put after every UpdateFrequency step.
    #
    #     Configuration parameters (coming from the configuration parameter
    #     dictionary):
    #         include_mixer_calib (bool): If True, include mixer
    #             calibration for the carrier and for the skewness.
    #         mixer_calib_carrier_settings (dict): Settings for the mixer
    #             calibration for the carrier.
    #         mixer_calib_skewness_settings (dict): Settings for the mixer
    #             calibration for the skewness.
    #     """
    #
    #     include_mixer_calib = kw.get("include_mixer_calib",
    #                                          False)
    #
    #     # Carrier settings
    #     mixer_calib_carrier_settings = kw.get("mixer_calib_carrier_settings",
    #                                           {})
    #     mixer_calib_carrier_settings.update({
    #         "qubit": self.qubit,
    #         "update": True
    #     })
    #
    #     # Skewness settings
    #     mixer_calib_skewness_settings = kw.get("mixer_calib_skewness_settings",
    #                                            {})
    #     mixer_calib_skewness_settings.update({
    #         "qubit": self.qubit,
    #         "update": True
    #     })
    #
    #     if include_mixer_calib:
    #         i = 0
    #
    #         while i < len(self.routine_template):
    #             step_class = self.get_step_class_at_index(i)
    #
    #             if step_class == UpdateFrequency:
    #
    #                 # Include mixer calibration
    #                 if include_mixer_calib:
    #                     # Skewness calibration
    #                     self.add_step(
    #                         MixerCalibrationSkewness,
    #                         'mixer_calibration_skewness',
    #                         mixer_calib_skewness_settings,
    #                         index=i + 1,
    #                     )
    #                     i += 1
    #
    #                     # Carrier calibration
    #                     self.add_step(
    #                         MixerCalibrationCarrier,
    #                         'mixer_calibration_carrier',
    #                         mixer_calib_carrier_settings,
    #                         index=i + 1,
    #                     )
    #                     i += 1
    #             i += 1

    class UpdateFluxToVoltage(IntermediateStep):
        """
        Intermediate step that updates the flux_to_voltage_and_freq
        dictionary using prior ReparkingRamsey measurements.
        """

        def __init__(self, index_reparking, **kw):
            """Initialize UpdateFluxToVoltage step.

            Args:
                index_reparking (int): The index of the previous ReparkingRamsey
                    step.

            Configuration parameters (coming from the configuration parameter
            dictionary):
                flux (float): the flux (in Phi0 units) to update the voltage and
                    frequency for using the ReparkingRamsey results.

            FIXME: it might be useful to also include results from normal ramsey
             experiments. For example, this could be used in UpdateFrequency if
             there is no model but a measurement of the ge-frequency was done.
            """
            super().__init__(index_reparking=index_reparking, **kw)

        def run(self):
            kw = self.kw

            qb = self.qubit

            # flux passed on in settings
            flux = self.get_param_value("flux")
            index_reparking = kw["index_reparking"]

            # voltage found by reparking routine
            reparking_ramsey = self.routine.routine_steps[index_reparking]
            try:
                voltage = reparking_ramsey.results[qb.name]["ss_volt"]
                frequency = reparking_ramsey.results[qb.name]["ss_freq"]
            except KeyError:
                log.error(
                    "Analysis reparking ramsey routine failed, flux to "
                    "voltage mapping can not be updated (guess values will be "
                    "used in the rest of the routine)")
                voltage, frequency = \
                    self.routine.flux_to_voltage_and_freq_guess[flux]

            self.routine.flux_to_voltage_and_freq.update(
                {flux: (voltage, frequency)})

    class DetermineModel(IntermediateStep):
        """
        Intermediate step that determines the model of the qubit based on
        the measured data. Can be used for both estimating the model and
        determining the final model.
        """

        def __init__(self,
                     **kw):
            """
            Initialize the DetermineModel step.

            Configuration parameters (coming from the configuration parameter
            dictionary):
                include_reparkings (bool): if True, the data from the
                    ReparkingRamsey measurements are used to help determine a
                    Hamiltonian model.
                include_resonator (bool): if True, the model includes the effect
                    of the resonator. If False, the coupling is set to zero and
                    essentially only the transmon parameters are optimized (the
                    resonator has no effect on the transmon).
                use_prior_model (bool): if True, the prior model is used to
                    determine the model, if False, the measured data is used.
                method (str): the optimization method to be used
                    when determining the model. Default is Nelder-Mead.
            """
            super().__init__(**kw, )
            self.experimental_values: Dict[str, float] = {}
            self.__result_dict: Dict[str, float] = {}
            self.__end_of_run_timestamp: str = None

        def run(self):
            """
            Runs the optimization and extract the fit parameters of the
            model.
            The extracted parameters are saved in the qubit object.
            """
            kw = self.kw

            # Using all experimental values
            self.experimental_values = (
                HamiltonianFitting.get_experimental_values(
                    qubit=self.routine.qubit,
                    fluxlines_dict=self.routine.fluxlines_dict,
                    timestamp_start=self.routine.preroutine_timestamp,
                    include_reparkings=self.get_param_value(
                        "include_reparkings"),
                ))

            log.info(f"Experimental values: {self.experimental_values}")
            self.routine.results[self.qubit.name][
                "experimental_values"] = self.experimental_values

            # Preparing guess parameters and choosing parameters to optimize
            p_guess, parameters_to_optimize = self.make_model_guess(
                use_prior_model=self.get_param_value("use_prior_model"),
                include_resonator=self.get_param_value("include_resonator"),
            )

            log.info(f"Parameters guess: {p_guess}")
            log.info(f"Parameters to optimize: {parameters_to_optimize}")

            # Determining the model
            f = self.routine.optimizer(
                experimental_values=self.experimental_values,
                parameters_to_optimize=parameters_to_optimize,
                parameters_guess=p_guess,
                method=self.get_param_value("method"),
            )

            # Extracting results, store results dictionary for use in
            # get_device_property_values.
            self.__result_dict = self.routine.fit_parameters_from_optimization_results(
                f, parameters_to_optimize, p_guess)

            log.info(f"Result from fit: {self.__result_dict}")

            # Saving model to qubit from routine
            self.routine.qubit.fit_ge_freq_from_dc_offset(self.__result_dict)

            # Save timestamp from previous run, to use for
            # get_device_property_values
            self.__end_of_run_timestamp = a_tools.get_last_n_timestamps(1)[0]

        def get_device_property_values(self, **kwargs):
            """
            Returns a dictionary of high-level device property values from
            running this DetermineModel step

            Args:
                qubit_sweet_spots (dict, optional): a dictionary mapping qubits
                    to sweet-spots ('uss', 'lss', or None)

            Returns:
                dict: dictionary of high-level results (may be empty)
            """

            results = self.get_empty_device_properties_dict()
            sweet_spots = kwargs.get('qubit_sweet_spots', {})
            if _device_db_client_module_missing:
                log.warning("Assembling the dictionary of high-level device "
                            "property values requires the module "
                            "'device-db-client', which was not imported "
                            "successfully.")
            elif self.__result_dict is not None:
                # For DetermineModel, the results are in
                # self.__result_dict from self.run()
                # The timestamp is the end of run timestamp or the one
                # immediately after. If we set `save_instrument_settings=True`,
                # we will have a timestamp after __end_of_run_timestamp.
                if self.get_param_value('save_instrument_settings'
                                        ) or not self.get_param_value("update"):
                    timestamps = a_tools.get_timestamps_in_range(
                        self.__end_of_run_timestamp)
                    # one after __end_of_run_timestamp
                    timestamp = timestamps[1]
                else:
                    timestamp = self.__end_of_run_timestamp
                node_creator = db_utils.ValueNodeCreator(
                    qubits=self.routine.qubit.name,
                    timestamp=timestamp,
                )

                # Total Josephson Energy
                if 'Ej_max' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='Ej_max',
                            value=self.__result_dict['Ej_max'],
                        ), )

                # Charging Energy
                if 'E_c' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='E_c',
                            value=self.__result_dict['E_c'],
                        ), )

                # Asymmetry
                if 'asymmetry' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='asymmetry',
                            value=self.__result_dict['asymmetry'],
                        ), )

                # Coupling
                if 'coupling' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='coupling',
                            value=self.__result_dict['coupling'],
                        ), )

                # Bare Readout Resonator Frequency
                if 'fr' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='fr',
                            component_type='ro_res',  # Not a qubit property
                            value=self.__result_dict['fr'],
                        ), )

                # Anharmonicity
                if 'anharmonicity' in self.__result_dict.keys():
                    results['property_values'].append(
                        node_creator.create_node(
                            property_type='anharmonicity',
                            value=self.__result_dict['anharmonicity'],
                        ), )
            return results

        def make_model_guess(self,
                             use_prior_model=True,
                             include_resonator=True):
            """
            Constructing parameters for the Hamiltonian model optimization.
            This includes defining values for the fixed parameters as well as
            initial guesses for the parameters to be optimized.

            Args:
                use_prior_model (bool): If True, the prior model parameters are
                    used as initial guess. Note that the voltage parameters
                    (dac_sweet_spot and V_per_phi0) are fixed according to the
                    sweet spot voltage measurements. If False, the guessed
                    parameters are determined through routine parameters or key
                    word inputs.
                include_resonator (bool): If True, the model includes the effect
                    of the resonator. If False, the coupling is set to zero and
                    essentially only the transmon parameters are optimized (the
                    resonator has no effect on the transmon).
            """
            # Using prior model to determine the model or not
            if use_prior_model:
                p_guess = self.qubit.fit_ge_freq_from_dc_offset()

                if "coupling" not in p_guess.keys():
                    p_guess["coupling"] = self.get_param_value(
                        "coupling") * include_resonator
                if "fr" not in p_guess.keys():
                    p_guess["fr"] = self.get_param_value(
                        "fr", associated_component_type_hint="ro_res")

            # Using new guess parameters instead
            else:
                p_guess = {
                    "Ej_max":
                        self.get_param_value("Ej_max"),
                    "E_c":
                        self.get_param_value("E_c"),
                    "asymmetry":
                        self.get_param_value("asymmetry"),
                    "coupling":
                        self.get_param_value("coupling") * include_resonator,
                    "fr":
                        self.get_param_value(
                            "fr", associated_component_type_hint="ro_res"),
                }

            # Using sweet spot measurements determined by the reparking routine
            flux_to_voltage_and_freq = self.routine.flux_to_voltage_and_freq
            ss1_flux, ss2_flux = self.routine.ss1_flux, self.routine.ss2_flux

            ss1_voltage, ss1_frequency = flux_to_voltage_and_freq[ss1_flux]
            ss2_voltage, ss2_frequency = flux_to_voltage_and_freq[ss2_flux]

            # Calculating voltage parameters based on ss-measurements and fixing
            # the corresponding parameters to these values
            V_per_phi0 = (ss1_voltage - ss2_voltage) / (ss1_flux - ss2_flux)
            dac_sweet_spot = ss1_voltage - V_per_phi0 * ss1_flux
            p_guess.update({
                "dac_sweet_spot": dac_sweet_spot,
                "V_per_phi0": V_per_phi0
            })

            # Including coupling (resonator) into model optimization
            if include_resonator:
                parameters_to_optimize = [
                    "Ej_max",
                    "E_c",
                    "asymmetry",
                    "coupling",
                ]
            else:
                parameters_to_optimize = ["Ej_max", "E_c", "asymmetry"]

            log.debug(f"Parameters guess {p_guess}")
            log.debug(f"Parameters to optimize {parameters_to_optimize}")

            return p_guess, parameters_to_optimize

    @staticmethod
    def verification_measurement(qubit,
                                 fluxlines_dict,
                                 fluxes=None,
                                 voltages=None,
                                 verbose=True,
                                 **kw):
        """
        Performs a verification measurement of the model for given fluxes
        (or voltages). The read-out is done using flux assisted read-out and
        the read-out should be configured beforehand.

        Note, this measurement requires the qubit(s) to contain a
        fit_ge_freq_from_dc_offset model.

        FIXME: the current implementation forces the user to use flux-pulse
         assisted read-out. In the future, the routine should also work on
         set-ups that only have DC sources, but no flux AWG. Possible solutions:
         - have the user specify the RO frequencies that are to be used at the
             various flux biases
         - use the Hamiltonian model to determine the RO frequencies at the
             various flux biases. Note, the usual Hamiltonian model does not
             take the Purcell filter into account which might cause problems.

        Args:
            qubit (QuDev_transmon): Qubit to perform the verification
                measurement on
            fluxlines_dict (dict): Dictionary containing the QCoDeS parameters
                of the fluxlines.
            fluxes (np.array): Fluxes to perform the verification measurement
                at. If None, fluxes is set to default np.linspace(0, -0.5, 11).
            voltages (np.array): Voltages to perform the verification
                measurement at. If None, voltages are set to correspond to the
                fluxes array. Note that if both fluxes and voltages are
                specified, only the values for `voltages` is used.
            verbose (bool): If True, prints updates on the progress of the
                measurement.

        Keyword args:
            reset_fluxline (bool): bool for resetting to fluxline to initial
                value after the measurement. Default is True.
            plot (bool): bool for plotting the results at the end, default False.

        Returns:
            dict: dictionary of verification measurements.

        FIXME: In an automated routine that is supposed to run without user
         interaction, plots should rather be stored in a timestamp folder rather
         than being shown on the screen. get_experimental_values_from_timestamps
         (or get_experimental_values?) could keep track of which timestamps
         belong to the current analysis, and you could then use get_folder
         (from analysis_toolbox) to get the folder of the latest among these
         timestamps. However, this might require refactoring the methods to not
         be static methods.
        """

        result_dict = qubit.fit_ge_freq_from_dc_offset()
        assert len(result_dict) > 0, (
            "This measurement requires the "
            "qubit(s) to contain a fit_ge_freq_from_dc_offset model")

        fluxline = fluxlines_dict[qubit.name]
        initial_voltage = fluxline()

        if fluxes is None and voltages is None:
            fluxes = np.linspace(0, -0.5, 11)
        if voltages is None:
            voltages = qubit.calculate_voltage_from_flux(fluxes)

        experimental_values = {}  # Empty dictionary to store results

        for index, voltage in enumerate(voltages):
            with temporary_value(
                    *ro_flux_tmp_vals(qubit, v_park=voltage, use_ro_flux=True)):

                if verbose:
                    print(f"Verification measurement  step {index} / "
                          f"{len(voltages)} of {qubit.name}")

                # Setting the frequency and fluxline
                qubit.calculate_frequency(voltage, update=True)
                fluxline(voltage)

                # Finding frequency
                ff = FindFrequency([qubit], dev=kw.get('dev'), update=True)

                # Storing experimental result
                experimental_values[voltage] = {"ge": qubit.ge_freq()}

        if kw.pop("reset_fluxline", True):
            # Resetting fluxline to initial value
            fluxline(initial_voltage)

        # Plotting
        if kw.pop("plot", False):
            HamiltonianFitting.plot_model_and_experimental_values(
                result_dict=result_dict,
                experimental_values=experimental_values)
            HamiltonianFitting.calculate_residuals(
                result_dict=result_dict,
                experimental_values=experimental_values,
                plot_residuals=True,
            )

        return experimental_values

    def get_device_property_values(self, **kwargs):
        """
        Returns a property values dictionary of the fitted Hamiltonian
        parameters

        Returns:
            dict: the property values dictionary for this routine
        """
        results = self.get_empty_device_properties_dict()
        # Only return property values from DetermineModel steps
        for _, step in enumerate(self.routine_steps):
            # FIXME: for some reason isinstance doesn't work here.
            if type(step).__name__ == 'DetermineModel':
                step_i_results = step.get_device_property_values(**kwargs)
                results['property_values'].append({
                    "step_type":
                        str(type(step).__name__)
                        if step_i_results.get('step_type') is None else
                        step_i_results.get('step_type'),
                    "property_values":
                        step_i_results['property_values'],
                })
        return results


# class MixerCalibrationSkewness(IntermediateStep):
#     """Mixer calibration step that calibrates the skewness of the mixer."""
#
#     def __init__(self, routine, **kw):
#         """
#         Initialize the MixerCalibrationSkewness step.
#
#         Args:
#             routine (Step): Routine object.
#
#         Keyword args:
#             calibrate_drive_mixer_skewness_function: method for calibrating to
#                 be used. Default is to use calibrate_drive_mixer_skewness_model.
#         """
#         super().__init__(routine=routine, **kw)
#
#     def run(self):
#         kw = self.kw
#
#         # FIXME: used only default right now, kw is not passed
#         calibrate_drive_mixer_skewness_function = kw.get(
#             "calibrate_drive_mixer_skewness_function",
#             "calibrate_drive_mixer_skewness_model",
#         )
#
#         function = getattr(self.qubit, calibrate_drive_mixer_skewness_function)
#         new_kw = keyword_subset_for_function(kw, function)
#         function(**new_kw)
#
#
# class MixerCalibrationCarrier(IntermediateStep):
#     """Mixer calibration step that calibrates the carrier of the mixer."""
#
#     def __init__(self, routine, **kw):
#         """
#         Initialize the MixerCalibrationCarrier step.
#
#         Args:
#             routine (Step): Routine object.
#
#         Keyword args:
#             calibrate_drive_mixer_carrier_function: method for calibrating to
#                 be used. Default is to use calibrate_drive_mixer_carrier_model.
#         """
#         super().__init__(routine=routine, **kw)
#
#     def run(self):
#         kw = self.kw
#
#         # FIXME: Used only default right now, kw is not passed
#         calibrate_drive_mixer_carrier_function = kw.get(
#             "calibrate_drive_mixer_carrier_function",
#             "calibrate_drive_mixer_carrier_model",
#         )
#
#         function = getattr(self.qubit, calibrate_drive_mixer_carrier_function)
#         new_kw = keyword_subset_for_function(kw, function)
#         function(**new_kw)


class SetTemporaryValuesFluxPulseReadOut(IntermediateStep):
    """
    Intermediate step that sets the temporary values for flux-pulse-assisted
    readout for a step of the routine. The step will modify the temporary
    values of the step at the specified index.
    """

    def __init__(self,
                 index,
                 **kw):
        """Initialize the SetTemporaryValuesFluxPulseReadOut step.

        Args:
            routine (Step): Routine object. The routine should be the parent of
                the step that requires temporary values for flux-pulse-assisted
                readout.
            index (int): Index of the step in the routine that requires
                temporary values for flux-pulse-assisted readout.

        Configuration parameters (coming from the configuration parameter
        dictionary):
            flux_park (float): Flux at which the qubit is going to be parked
                at the moment the readout happens. The corresponding voltage
                will be calculated using qb.calculate_voltage_from_flux.
                The voltage value will be used to calculate the temporary value
                for flux-pulse-assisted readout using ro_flux_tmp_vals.
                If None, voltage_park will be used.
            voltage_park (float): Voltage at which the qubit will be parked
                at the moment the readout happens. This is only used if
                flux_park is None. If voltage_park is also None, a
                ValueError will be raised.
        """
        super().__init__(
            index=index,
            **kw,
        )

    def run(self):
        kw = self.kw
        qb = self.qubit
        index = self.kw["index"]

        if (flux := self.get_param_value("flux_park")) is not None:
            v_park = qb.calculate_voltage_from_flux(flux)
        elif (v_park_tmp := self.get_param_value("voltage_park")) is not None:
            v_park = v_park_tmp
        else:
            raise ValueError("No voltage or flux specified")

        # Temporary values for ro
        ro_tmp_vals = ro_flux_tmp_vals(qb, v_park, use_ro_flux=True)

        # Extending temporary values
        self.routine.extend_step_tmp_vals_at_index(tmp_vals=ro_tmp_vals,
                                                   index=index)
