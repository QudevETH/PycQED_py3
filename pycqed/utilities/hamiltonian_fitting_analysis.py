import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.simulations.transmon as transmon
import scipy
from pycqed.utilities.state_and_transition_translation import *
import h5py

log = logging.getLogger(__name__)

# FIXME: usage of the code could maybe be simplified by using a real
# object-oriented approach. For instance, you pass along dicts of
# experimental_values between various functions. Analogous idea for result_dict.


class HamiltonianFittingAnalysis:
    _PARAMETERS_ALL = [
        "dac_sweet_spot",
        "V_per_phi0",
        "Ej_max",
        "E_c",
        "asymmetry",
        "coupling",
        "fr",
    ]

    @staticmethod
    def eigenenergies_transmon_with_resonator(
        phi,
        Ej_max,
        E_c,
        asymmetry,
        coupling,
        fr,
        states=((1, 0), (2, 0), (0, 1), (1, 1)),
        **kw,
    ):
        """
        Calculates eigenenergies of transmon coupled with resonator system.

        Key words will be passed on to transmon.transmon_resonator_levels

        Arguments:
            phi: dimensionless flux
            Ejmax: Josephson energy
            Ec: Charge energy
            d: assymmetry
            g: coupling strength
            fr: bare resonator spectroscopy
            states: tuple (or list) containing the states of interest. Each
            state is given as (n_transmon, n_resonator)
            in their respective energy eigenbases.

        Keyword Arguments:
            ng: bias charge of the transmon
            dim_charge: dimension of the truncated transmon Hilbert space
            dim_resonator: dimension of the truncated resonator Hilbert space

        Returns:
            Eigenenergies of the transmon plus resonator system as numpy array
        """
        states = tuple(
            [transmon_resonator_state_to_tuple(state) for state in states]
        )

        Ej = (
            Ej_max
            * np.cos(np.pi * phi)
            * np.sqrt(1 + asymmetry ** 2 * np.tan(np.pi * phi) ** 2)
        )

        for x in list(kw.keys()):
            if x not in ["ng", "dim_charge", "dim_resonator"]:
                kw.pop(x)

        return transmon.transmon_resonator_levels(
            ec=E_c, ej=Ej, frb=fr, gb=coupling, states=states, **kw
        )

    @staticmethod
    def transitions_transmon_with_resonator(
        phi,
        Ej_max,
        E_c,
        asymmetry,
        coupling,
        fr,
        transitions=(((0, 0), (1, 0)), ((0, 0), (0, 1))),
        **kw,
    ):
        """
        Calculates transition frequencies.

        Arguments:
            phi: dimensionless flux
            Ejmax: Josephson energy
            Ec: Charge energy
            d: assymmetry
            g: coupling strength
            fr: bare resonator spectroscopy
            transitions: tuple (or list) containing the transitions of
            interest. Each state is given as a 2-tuple
            containing two states. Each state in this tuple is given as (
            n_transmon, n_resonator) in their respective
            energy eigenbases.

        Keyword Arguments:
            ng: bias charge of the transmon
            dim_charge: dimension of the truncated transmon Hilbert space
            dim_resonator: dimension of the truncated resonator Hilbert space

        Returns:
            Transition frequencies between the states as numpy array
        """
        transitions = HamiltonianFittingAnalysis._translate_transitions(
            transitions
        )

        states = list(set([s for t in transitions for s in t]))
        eigen_energies = (
            HamiltonianFittingAnalysis.eigenenergies_transmon_with_resonator(
                phi=phi,
                Ej_max=Ej_max,
                E_c=E_c,
                asymmetry=asymmetry,
                coupling=coupling,
                fr=fr,
                states=states,
                **kw,
            )
        )

        freqs = []
        for t in transitions:
            i0 = states.index(t[0])
            i1 = states.index(t[1])
            freqs.append(eigen_energies[i1] - eigen_energies[i0])
        return freqs

    @staticmethod
    def plot_transitions(
        Ej_max,
        E_c,
        asymmetry,
        coupling,
        fr,
        phis=np.linspace(-1.0, 1.0, 101),
        transitions=(((0, 0), (1, 0)), ((0, 0), (0, 1))),
        **kw,
    ):
        """
        Plots transition frequencies of transmon-resonator system given model
        parameters. The transition frequencies are either plotted against flux
        or against voltage. In case the voltage is given, you also have to pass
        in the dac_sweet_spot and V_per_phi0 parameters as key word arguments.

        Arguments:
            Ejmax: max Josephson energy
            Ec: charge energy
            d: assymmetry
            g: coupling strength
            fr: bare resonator frequency
            phis: (numpy) array, dimensionless flux
            transitions: tuple of transitions

        Keyword Arguments:
            voltages: (numpy) array, voltages
            dac_sweet_spot: dac sweet spot in V. Needed if voltages are
                given.
            V_per_phi0: voltage per flux in V. Needed if voltages are given.

            frequency_unit: unit of the frequency. For example Hz, GHz,
                or bpm if that's what you like. Default is GHz.
            frequency_factor: factor compared to Hz. For example, MHz ->
                frequency_factor = 1e6. Default is 1e9 (corresponding to GHz).

        Returns:
            None
        """

        frequency_unit = kw.pop("frequency_unit", "GHz")
        frequency_factor = kw.pop("frequency_factor", 1e9)

        voltages = kw.pop("voltages", None)
        plot_voltages = False
        if voltages is not None:
            plot_voltages = True
            phis = (voltages - kw["dac_sweet_spot"]) / kw["V_per_phi0"]

        transitions = HamiltonianFittingAnalysis._translate_transitions(
            transitions
        )

        transitionfrequencies = np.array(
            [
                HamiltonianFittingAnalysis.transitions_transmon_with_resonator(
                    phi,
                    Ej_max=Ej_max,
                    E_c=E_c,
                    asymmetry=asymmetry,
                    coupling=coupling,
                    fr=fr,
                    transitions=transitions,
                )
                for phi in phis
            ]
        )

        for i in range(transitionfrequencies.shape[1]):
            plt.plot(
                (phis if not plot_voltages else voltages),
                transitionfrequencies[:, i] / frequency_factor,
                label=(
                    transmon_resonator_state_to_str(transitions[i][0])
                    + r"$\rightarrow$"
                    + transmon_resonator_state_to_str(transitions[i][1])
                    + " model"
                ),
            )
        if not plot_voltages:
            plt.xlabel(r"flux $\phi$ ($\phi_0$)")
        else:
            plt.xlabel(r"DC bias voltage $U$ (V)")
        plt.ylabel(f"transition frequency $f$ ({frequency_unit})")
        plt.legend()

    @staticmethod
    def plot_experimental_values(
        experimental_values,
        print_values=False,
        flux_dimension="voltage",
        dac_sweet_spot=None,
        V_per_phi0=None,
        transitions=None,
        phi_limits=None,
        voltage_limits=None,
        **kw,
    ):
        """
        Plots the measured transition frequencies to the bias voltage or
        dimensionless flux. Multiple transitions are allowed.

        Arguments:
            experimental_values: The data should come as
                dict(voltage1: dict(transition1: freq1, transition2, freq2,
                ...), voltage2: ...)
            Here, the keys are bias voltage values, and the keys of the inner
            dictionary are the transitions given
            as tuple. The corresponding items are the measured frequencies of
            that transitions. The order of the transition
            is (transmon, resonator) and thus the ((0,0), (1,0)) transition is
            the ge-frequency of the transmon.

            print_values: Boolean if user wants to (user friendly) print the
            experimental values
            flux_dimension: 'voltage' or 'dimensionless',
            dac_sweet_spot: voltage of the upper sweet spot
            V_per_phi0: voltage difference between two neighbouring upper and
            lower sweet spots
            transitions: transitions that will be plotted.
            voltage_limits: plot limits for voltage. Default will be based on
            the experimental values.
            phi_limits: plot limits for the dimensionless flux. If None the
            limits will be based on the experimental values.

        Keyword Arguments:
            frequency_unit: unit of the frequency. For example Hz, GHz,
                or bpm if that's what you like.
            Default is GHz.
            frequency_factor: factor compared to Hz. For example, MHz ->
                frequency_factor = 1e6. Default is 1e9 (corresponding to GHz).

        Example::
            experimental_values = {
                0.531:{
                    ((0,0),(1, 0)):6.143162354737646e9,
                    ((1,0),(2, 0)):5.971582479578413e9,
                    ((0,0),(0, 1)):7.1543e9,
                },
                1.068:{
                    ((0,0),(1, 0)):6.000116129419129e9,
                    ((1,0),(2, 0)):5.822948252320393e9,
                    ((0,0),(0, 1)):7.1505e9,
                }
            }

            plot_experimental_values(experimental_values) # plots experimental
            values marked with respective transition
        """
        experimental_values = (
            HamiltonianFittingAnalysis._translate_experimental_values(
                experimental_values
            )
        )

        frequency_unit = kw.pop("frequency_unit", "GHz")
        frequency_factor = kw.pop("frequency_factor", 1e9)

        if flux_dimension == "dimensionless":
            if dac_sweet_spot is None or V_per_phi0 is None:
                raise ValueError(
                    "provide both the dac_sweet_spot and the V_per_phi0 if you "
                    "want to plot the "
                    "frequencies to the dimensionless flux"
                )

        list_of_bias = list(set(experimental_values.keys()))
        list_of_bias.sort()

        # plot transitions
        if transitions is None:
            transitions = [
                list(experimental_values[b].keys()) for b in list_of_bias
            ]
            transitions = list(set([b for a in transitions for b in a]))
        else:
            transitions = HamiltonianFittingAnalysis._translate_transitions(
                transitions
            )

        # plot limits for voltage or flux
        voltages = np.array(list_of_bias)
        if flux_dimension == "voltage":
            if voltage_limits is None:
                voltage_margin = kw.pop("voltage_margin", 0.05)
                voltage_limits = (
                    np.min(voltages) - voltage_margin,
                    np.max(voltages) + voltage_margin,
                )
        elif flux_dimension == "dimensionless":
            if phi_limits is None:
                phis = (voltages - dac_sweet_spot) / V_per_phi0
                phi_margin = kw.pop("phi_margin", 0.01)
                phi_limits = (
                    np.min(phis) - phi_margin,
                    np.max(phis) + phi_margin,
                )

        # plotting per transition
        for t in transitions:
            # extracting and ordering data from experimental_results
            datasetx = []
            datasety = []
            for b in list_of_bias:
                try:
                    datasety.append(experimental_values[b][t])
                    if flux_dimension == "voltage":
                        datasetx.append(b)
                    elif flux_dimension == "dimensionless":
                        datasetx.append((b - dac_sweet_spot) / V_per_phi0)
                    else:
                        raise ValueError(
                            "Flux dimension must either be 'voltage' or "
                            "'dimensionless'"
                        )
                except:
                    pass

            # printing
            if print_values:
                print("transistion:", transition_to_str(t))
                if flux_dimension == "voltage":
                    print("bias (V):", datasetx)
                elif flux_dimension == "dimensionless":
                    print("dimensionless flux:", datasetx)
                print("measured transition frequency (Hz):", datasety, "\n")

            # plotting
            plt.plot(
                datasetx,
                np.array(datasety) / frequency_factor,
                "+",
                label=transmon_resonator_state_to_str(t[0])
                + r"$\rightarrow$"
                + transmon_resonator_state_to_str(t[1])
                + " data",
                markersize=kw.get("markersize", 10),
                markeredgewidth=kw.get("markeredgewidth", 2),
            )
            if flux_dimension == "voltage":
                plt.xlabel("DC bias voltage $U$ (V)")
                plt.xlim([voltage_limits[0], voltage_limits[1]])
            elif flux_dimension == "dimensionless":
                plt.xlabel(r"flux $\phi$ ($\phi_0$)")
                plt.xlim([phi_limits[0], phi_limits[1]])
            plt.ylabel(f"transition frequency $f$ ({frequency_unit})")
            plt.legend()
        plt.show()

    @staticmethod
    def cost_function(
        experimental_values, model, model_parameters, mode="relative"
    ):
        """

        Args:
            experimental_values
            model
            model_parameters
            mode: mode for how to evaluate costs and can be set to 'relative'
            or 'absolute'. This parameter comes into
            play when one looks at different transitions that naturally have
            different scales. Setting mode to 'relative'
            ensures that the optimization does not prioritize naturally higher
            frequencies but rather looks at the relative
            deviation. Default is 'relative'.

        Returns:
            cost: float that specifies the cost given the set of experimental
            values and the set of model parameters

        """
        experimental_values = (
            HamiltonianFittingAnalysis._translate_experimental_values(
                experimental_values
            )
        )
        list_of_bias = experimental_values.keys()

        cost = 0
        for b in list_of_bias:
            transitions = experimental_values[b].keys()
            eigenfrequencies = model(
                bias=b, parameters=model_parameters, transitions=transitions
            )
            for i, t in enumerate(transitions):
                experimental_freq = experimental_values[b][t]
                if mode == "relative":
                    cost += (
                        (experimental_freq - eigenfrequencies[i])
                        / experimental_freq
                    ) ** 2
                if mode == "absolute":
                    cost += (experimental_freq - eigenfrequencies[i]) ** 2
        return cost

    @staticmethod
    def model(bias, parameters, transitions):
        """ "
        Model function that will be used to calculate transition frequencies.
        """
        transitions = HamiltonianFittingAnalysis._translate_transitions(
            transitions
        )
        phi = (bias - parameters["dac_sweet_spot"]) / parameters["V_per_phi0"]
        return HamiltonianFittingAnalysis.transitions_transmon_with_resonator(
            phi,
            parameters["Ej_max"],
            parameters["E_c"],
            parameters["asymmetry"],
            parameters["coupling"],
            parameters["fr"],
            transitions=transitions,
        )

    @staticmethod
    def optimizer(
        experimental_values,
        parameters_to_optimize,
        parameters_guess,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            experimental_values
            parameters_to_optimize
            parameters_guess
            model
            cost_function
            args

        Keyword Arguments:
            model:
            cost_function:
            method: ... The default method is 'Nelder-Mead'.

        Returns:
            optimization object (scipy.optimize.OptimizeResult) containing the
            information of the optimization

        """

        model = kwargs.pop("model", HamiltonianFittingAnalysis.model)
        cost_function = kwargs.pop(
            "cost_function", HamiltonianFittingAnalysis.cost_function
        )
        mode = kwargs.pop("mode", "relative")

        fixed_parameters = HamiltonianFittingAnalysis._PARAMETERS_ALL.copy()
        for param in parameters_to_optimize:
            fixed_parameters.remove(param)

        def min_func(x):
            param = {
                **{p: x0 for p, x0 in zip(parameters_to_optimize, x)},
                **{k: parameters_guess[k] for k in fixed_parameters},
            }
            return cost_function(
                experimental_values=experimental_values,
                model=model,
                model_parameters=param,
                mode=mode,
            )

        optimization = scipy.optimize.minimize(
            min_func,
            [parameters_guess[k] for k in parameters_to_optimize],
            method=kwargs.pop("method", "Nelder-Mead"),
            *args,
            **kwargs,
        )
        return optimization

    @staticmethod
    def fit_parameters_from_optimization_results(
        optimization: scipy.optimize.OptimizeResult,
        parameters_to_optimize,
        parameters_guess,
    ):
        """
        Extracts the fit parameters from an optimization results (of type
        OptimizeResult)

        Arguments
            optimization
            parameters_to_optimize
            parameters_guess

        Returns
            resulting dictionary containing ['dac_sweet_spot', 'V_per_phi0',
            'Ej_max', 'E_c', 'asymmetry', 'coupling', 'fr']
            calculated during the optimization.

        """

        fixed_parameters = HamiltonianFittingAnalysis._PARAMETERS_ALL.copy()
        for param in parameters_to_optimize:
            fixed_parameters.remove(param)

        result_dict = {
            **{
                k: optimization.x[i]
                for i, k in enumerate(parameters_to_optimize)
            },
            **{k: parameters_guess[k] for k in fixed_parameters},
        }
        return result_dict

    @staticmethod
    def fit_parameters(
        experimental_values,
        parameters_to_optimize,
        parameters_guess,
        **kwargs,
    ):
        """
        Arguments:
            experimental_values: dictionary containing the experimental values
            parameters_to_optimize: list of parameters to optimize
            parameters_guess: dictionary containing the initial guess for the
                parameters to optimize and the values for the fixed parameters.

        Keyword Arguments:
            model: function that will be used to calculate the transition
            cost_function: function that will be used to calculate the cost.

        Additional kwargs are passed to the optimizer function.

        Returns:
            resulting dictionary containing ['dac_sweet_spot', 'V_per_phi0',
            'Ej_max', 'E_c', 'asymmetry', 'coupling', 'fr']
            calculated during an optimization.
        """
        experimental_values = (
            HamiltonianFittingAnalysis._translate_experimental_values(
                experimental_values
            )
        )

        model = kwargs.pop("model", HamiltonianFittingAnalysis.model)
        cost_function = kwargs.pop(
            "cost_function", HamiltonianFittingAnalysis.cost_function
        )

        optimization_results = HamiltonianFittingAnalysis.optimizer(
            experimental_values,
            parameters_to_optimize,
            parameters_guess,
            model=model,
            cost_function=cost_function,
            **kwargs,
        )

        return (
            HamiltonianFittingAnalysis.fit_parameters_from_optimization_results(
                optimization_results, parameters_to_optimize, parameters_guess
            )
        )

    @staticmethod
    def plot_model_and_experimental_values(
        result_dict,
        experimental_values,
        voltage_limits=None,
        transitions=None,
        **kw,
    ):
        """
        Plots transition frequencies as function of flux or voltage based model
        together with experimentally found transition frequencies.

        Arguments:
            result_dict
            experimental_values
            voltage_limits
            transitions

        Keyword Arguments:
            voltage_stepsize: voltage step size for the plot (default: 0.1)
            voltage_margin: voltage margin for the plot (default: 0.1)

            phi_limits: limits of the flux axis
            phi_stepsize: flux step size for the plot (default: 0.01)
            phi_margin: flux margin for the plot (default: 0.01)

            frequency_unit: unit of the frequency. For example Hz, GHz,
            or bpm if that's what you like.	Default is GHz.
            frequency_factor: factor compared to Hz. For example, MHz ->
            frequency_factor = 1e6.
            Default is 1e9 (corresponding to GHz)
        """
        # converting input into consistent internal format
        experimental_values = (
            HamiltonianFittingAnalysis._translate_experimental_values(
                experimental_values
            )
        )

        phi_limits = kw.pop("phi_limits", None)
        plot_phi = kw.pop("plot_phi", (phi_limits is not None))

        phis = None
        voltages = None

        voltages_experiment = np.array(list(experimental_values.keys()))
        if not plot_phi:
            if voltage_limits is None:
                voltage_margin = kw.pop("voltage_margin", 0.1)
                voltage_limits = [
                    voltages_experiment.min() - voltage_margin,
                    voltages_experiment.max() + voltage_margin,
                ]
            voltages = np.arange(
                voltage_limits[0],
                voltage_limits[1],
                kw.pop("voltage_step", 0.01),
            )

        elif plot_phi:
            if phi_limits is None:
                phis = (voltages_experiment - result_dict["dac_sweet_spot"]) / (
                    result_dict["V_per_phi0"]
                )
                phi_min, phi_max = np.min(phis), np.max(phis)

                phi_margin = kw.pop("phi_margin", 0.01)
                phi_limits = (phi_min - phi_margin, phi_max + phi_margin)
            phis = np.arange(
                phi_limits[0], phi_limits[1], kw.pop("phi_step", 0.005)
            )

        if transitions is None:
            list_of_bias = list(set(experimental_values.keys()))
            list_of_bias.sort()

            transitions = [
                list(experimental_values[b].keys()) for b in list_of_bias
            ]
            transitions = list(set([b for a in transitions for b in a]))
        else:
            transitions = HamiltonianFittingAnalysis._translate_transitions(
                transitions
            )

        HamiltonianFittingAnalysis.plot_transitions(
            **result_dict,
            transitions=transitions,
            phis=phis,
            voltages=voltages,
            **kw,
        )

        HamiltonianFittingAnalysis.plot_experimental_values(
            experimental_values,
            print_values=False,
            flux_dimension=("voltage" if not plot_phi else "dimensionless"),
            transitions=transitions,
            phi_limits=phi_limits,
            voltage_limits=voltage_limits,
            **result_dict,
            **kw,
        )

    @staticmethod
    def calculate_residuals(
        result_dict,
        experimental_values,
        transitions=None,
        plot_residuals=False,
        **kw,
    ):
        """

        Arguments:
            result_dict: dictionary containing 'dac_sweet_spot', 'V_per_phi0',
                'Ej_max', 'E_c', 'asymmetry', 'coupling' and 'fr'
            experimental_values: dictionary containing experimentally found
                transition frequencies
            transitions: transitions that will be included in the residual
            plot_residuals: Boolean to allow the user to visualize the
                residual frequencies in a plot

        Keyword Arguments:
            frequency_unit: unit of the frequency. For example Hz, GHz,
                or bpm if that's what you like. Default is MHz.
            frequency_factor: factor compared to Hz. For example, MHz ->
                frequency_factor = 1e6. Default is 1e6 (corresponding to MHz)
                frequencies as a function of flux or voltage.

            You can pass additional key words for plotting

        Returns
            Dictionary containing residual frequencies (difference between
            model and experimental values) with as key the transition

        """
        experimental_values = (
            HamiltonianFittingAnalysis._translate_experimental_values(
                experimental_values
            )
        )

        list_of_bias = list(set(experimental_values.keys()))
        list_of_bias.sort()

        frequency_prefix = kw.pop("frequency_unit", "MHz")
        frequency_factor = kw.pop("frequency_factor", 1e6)

        # if no list of transitions if given, use all transitions found in
        # experimental_values
        if transitions is None:
            transitions = [
                list(experimental_values[b].keys()) for b in list_of_bias
            ]
            transitions = list(set([b for a in transitions for b in a]))

        residuals = {}
        for t in transitions:
            dataset_phi = []
            dataset_experiment = []
            dataset_model = []
            for b in list_of_bias:
                try:
                    phi = (b - result_dict["dac_sweet_spot"]) / result_dict[
                        "V_per_phi0"
                    ]
                    dataset_experiment.append(experimental_values[b][t])
                    dataset_model.append(
                        HamiltonianFittingAnalysis.transitions_transmon_with_resonator(
                            phi=phi, **result_dict, transitions=[t]
                        )[
                            0
                        ]
                    )
                    dataset_phi.append(phi)
                except:
                    pass

            residuals[t] = np.array(dataset_model) - np.array(
                dataset_experiment
            )

            if plot_residuals:
                fig, ax1 = plt.subplots()
                ax_twin = ax1.twiny()

                markersize = kw.get("markersize", 10)
                markeredgewidth = kw.get("markeredgewidth", 2)

                def V_to_Phi1(V):
                    print(result_dict)
                    V_0 = result_dict["dac_sweet_spot"]
                    V_phi0 = result_dict["V_per_phi0"]
                    return (V - V_0) / V_phi0  # Phi in units of Phi0

                def V_to_Phi2(ax1):
                    x1, x2 = ax1.get_xlim()
                    ax_twin.set_xlim(V_to_Phi1(x1), V_to_Phi1(x2))
                    ax_twin.figure.canvas.draw()

                ax1.callbacks.connect("xlim_changed", V_to_Phi2)

                ax1.set_xlabel(r"DC bias voltage $U$ (V)")
                ax_twin.set_xlabel("Flux $\Phi$ ($\Phi_0$)")

                plt.axhline(y=0, color="grey", linestyle="dotted", linewidth=1)
                ax1.plot(
                    list_of_bias,
                    residuals[t] / frequency_factor,
                    "+",
                    label=transmon_resonator_state_to_str(t[0])
                    + r"$\rightarrow$"
                    + transmon_resonator_state_to_str(t[1]),
                    markersize=markersize,
                    markeredgewidth=markeredgewidth,
                    **kw,
                )
                ax1.set_ylabel(
                    r"residual frequency $f_{\mathrm{model}}"
                    + r" - f_{\mathrm{meas}}$ "
                    + f"({frequency_prefix})"
                )

        if plot_residuals:
            ax1.legend()

        return residuals

    @staticmethod
    def get_experimental_values(
        qubit,
        fluxlines_dict,
        timestamp_start,
        timestamp_end=None,
        transitions=(
            "ge",
            "ef",
        ),
        experiments=("Ramsey",),
        **kw,
    ):
        """
        Gets the experimental values from the database in a particular time
        period (between timestamp_start and timestamp_end) and returns them in
        the usual format (see docstring of plot_experimental_values)

        Arguments:
            qubit: qubit object or qubit name
            fluxlines_dict: dictionary containing the qubits and their
                corresponding fluxline ids (necessary to determine voltage)
            timestamp_start: start timestamp
            timestamp_end: end timestamp. This timestamp should come
                chronologically after timestamp_start. Default is None, which
                means that the end timestamp is the latest timestamp in the
                database.
            transitions: transitions that will be included in the
                experimental_values
            experiments: experiments that will be included in the
                experimental_values. Default is ('Ramsey',).

        Keyword Arguments:
            include_reparkings: Boolean to include reparkings in the
                experimental_values. Default is False.
            overwrite_warning: Boolean to display the warning message when
                overwriting a transition frequency at a particular voltage.
                Default is True.
            datadir: path to the directory containing the desired data.
                Default is D:pydata.

        #FIXME code replication could have been avoided by letting
        get_experimental_values call get_experimental_values_from_timestamps
        with the list of timestamps returned by get_timestamps_in_range
        """
        transitions = [
            transition_to_str(transition) for transition in transitions
        ]
        experimental_values = {}

        # datadirectory
        default_datadir = a_tools.datadir
        a_tools.datadir = kw.get("datadir", default_datadir)

        if not set(transitions).issubset(set(("ge", "ef"))):
            log.warning(
                "Only ge and ef transitions are supported now. Setting "
                "transitions to ('ge', 'ef')"
            )
            transitions = ("ge", "ef")

        # easily add ReparkingRamseys to the experiments list
        if kw.get("include_reparkings", False):
            experiments = list(experiments)
            experiments.append("ReparkingRamsey")

        for transition in transitions:
            for experiment in experiments:
                label = f"_{experiment}_{transition}_{qubit.name}"
                timestamps = a_tools.get_timestamps_in_range(
                    timestamp_start, timestamp_end=timestamp_end, label=label
                )
                for timestamp in timestamps:
                    if experiment == "Ramsey":
                        HamiltonianFittingAnalysis._fill_experimental_values_with_Ramsey(
                            experimental_values,
                            timestamp,
                            qubit,
                            fluxlines_dict,
                        )

                    elif experiment == "ReparkingRamsey":
                        HamiltonianFittingAnalysis._fill_experimental_values_with_ReparkingRamsey(
                            experimental_values, timestamp, qubit
                        )

        a_tools.datadir = default_datadir
        return experimental_values

    @staticmethod
    def get_experimental_values_from_timestamps(
        qubit, fluxlines_dict, timestamps, **kw
    ):
        """
        Gets the experimental values from the database from a list of timestamps
        and returns them in the usual format.

        Arguments:
            timestamps: timestamps
            qubit: qubit object or qubit name (str)
            transition: transition

        Keyword Arguments:
            include_reparkings: Boolean to include reparkings in the
            datadir: path to the directory containing the desired data.
        """
        experimental_values = {}

        # datadirectory
        default_datadir = a_tools.datadir
        a_tools.datadir = kw.get("datadir", default_datadir)

        include_reparkings = kw.get("include_reparkings", False)

        for timestamp in timestamps:
            path = a_tools.data_from_time(timestamp)

            if "_Ramsey_" in path:
                if fluxlines_dict is None:
                    raise ValueError(
                        "fluxlines_dict must be specified for to"
                        "read the experimental values from Ramsey experiments"
                    )
                HamiltonianFittingAnalysis._fill_experimental_values_with_Ramsey(
                    experimental_values, timestamp, qubit, fluxlines_dict
                )

            elif ("_ReparkingRamsey_" in path) and include_reparkings:
                HamiltonianFittingAnalysis._fill_experimental_values_with_ReparkingRamsey(
                    experimental_values, timestamp, qubit
                )

        a_tools.datadir = default_datadir
        return experimental_values

    @staticmethod
    def _fill_experimental_values(
        experimental_values, voltage, transition, freq
    ):
        """
        Fills the experimental_values dictionary with the experimental values.

        Note that if there are multiple transitions at the same voltage, the last
        one will be used and the other(s) overwritten.
        """
        if voltage not in experimental_values:
            experimental_values[voltage] = {}
        if transition not in experimental_values[voltage]:
            experimental_values[voltage][transition] = {}
        experimental_values[voltage][transition] = freq

    @staticmethod
    def _fill_experimental_values_with_Ramsey(
        experimental_values, timestamp, qubit, fluxlines_dict
    ):
        """
        Fills the experimental_values dictionary with the experimental values
        from Ramsey experiments.

        NOTE if there are multiple measurements of the same transition at the
        same voltage, the (chronological) last one will be used.

        Arguments:
            experimental_values: dictionary containing the experimental values
            timestamp: timestamp
            qubit: qubit (qubit object) or qubit name (str)
            fluxlines_dict: dictionary containing the fluxline ids (necessary
                to determine voltage)
        """
        path = a_tools.data_from_time(timestamp)
        filepath = a_tools.measurement_filename(path)
        data = h5py.File(filepath, "r")

        if "_ge_" in filepath:
            transition = "ge"
        elif "_ef_" in filepath:
            transition = "ef"
        else:
            raise ValueError(
                "Transition not recognized. Only ge and ef "
                "transitions are supported."
            )

        if type(qubit) is str:
            qubit_name = qubit
        else:
            qubit_name = qubit.name
        if not qubit_name in filepath:
            return

        try:
            freq = data["Analysis"]["Processed data"]["analysis_params_dict"][
                qubit_name
            ]["exp_decay"].attrs["new_qb_freq"]
            freq_std = data["Analysis"]["Processed data"][
                "analysis_params_dict"
            ][qubit_name]["exp_decay"].attrs["new_qb_freq_stderr"]
            voltage = float(
                data["Instrument settings"]["DCSource"].attrs[
                    fluxlines_dict[qubit_name].name
                ]
            )
            HamiltonianFittingAnalysis._fill_experimental_values(
                experimental_values, voltage, transition, freq
            )
        except KeyError:
            log.warning(f"Could not get ramsey data from file {filepath}")

    @staticmethod
    def _fill_experimental_values_with_ReparkingRamsey(
        experimental_values, timestamp, qubit
    ):
        """
        Fills the experimental_values dictionary with the experimental values
        from ReparkingRamsey experiments.

        NOTE if there are multiple measurements of the same transition at the
        same voltage, the (chronological) last one will be used.

        Arguments:
            experimental_values: dictionary to fill with experimental data
            timestamp: timestamp
            qubit: qubit (qubit object) or qubit name (str)
        """
        path = a_tools.data_from_time(timestamp)
        filepath = a_tools.measurement_filename(path)
        data = h5py.File(filepath, "r")

        if "_ge_" in filepath:
            transition = "ge"
        elif "_ef_" in filepath:
            transition = "ef"
        else:
            log.warning(
                "Transition not recognized. Only ge and ef transitions "
                "are supported."
            )
            return

        if type(qubit) is str:
            qubit_name = qubit
        else:
            qubit_name = qubit.name
        if not qubit_name in filepath:
            return

        try:
            dc_voltages = np.array(
                data["Experimental Data"]["Experimental Metadata"][
                    "dc_voltages"
                ]
            )
            for i in range(len(dc_voltages)):
                freq = float(
                    np.array(
                        data["Analysis"]["Processed data"][
                            "analysis_params_dict"
                        ][qubit_name + f"_" f"{i}"]["exp_decay"].attrs[
                            "new_qb_freq"
                        ]
                    )
                )
                freq_std = float(
                    np.array(
                        data["Analysis"]["Processed data"][
                            "analysis_params_dict"
                        ][qubit_name + f"_" f"{i}"]["exp_decay"].attrs[
                            "new_qb_freq"
                        ]
                    )
                )
                voltage = dc_voltages[i]
                HamiltonianFittingAnalysis._fill_experimental_values(
                    experimental_values, voltage, transition, freq
                )
        except:
            log.warning(
                "Could not get reparking data from file {}".format(filepath)
            )

    @staticmethod
    def _translate_experimental_values(experimental_values):
        d = {}
        for v in experimental_values.keys():
            d[v] = {}
            for t in experimental_values[v]:
                transition_to_tuple(t)
                d[v][transition_to_tuple(t)] = experimental_values[v][t]
        return d

    @staticmethod
    def _translate_transitions(transitions):
        return [transition_to_tuple(t) for t in transitions]
