from pycqed.measurement.calibration.automatic_calibration_routines.base import (
    Step,
    RoutineTemplate,
    AutomaticCalibrationRoutine
)

from pycqed.measurement.calibration.automatic_calibration_routines import \
    routines_utils
from pycqed.measurement.spectroscopy import spectroscopy as spec
from pycqed.instrument_drivers.meta_instrument.device import Device
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon

import numpy as np
import logging
from typing import Tuple, Dict, Any

log = logging.getLogger(__name__)


class FeedlineSpectroscopyStep(spec.FeedlineSpectroscopy, Step):
    """Wrapper for FeedlineSpectroscopy experiment. Performs a 1D spectroscopy
    of the feedlines to which the given qubits belong to.

    Args:
        routine (Step): The parent routine

    Keyword Arguments (for the Step constructor):
        step_label (str): A unique label for this step to be used in the
            configuration parameters files.
        settings (:obj:`SettingsDictionary`): The configuration parameters
            passed down from its parent. if None, the dictionary is taken from
            the Device object.
        qubits (list): A list with the Qubit objects which should be part of
            the step.
        settings_user (dict): A dictionary from the user to update the
            configuration parameters with.

    Configuration parameters (coming from the configuration parameter dictionary):
        freq_start (float): Starting frequency for the frequency sweep.
        freq_stop (float): Stopping frequency for the frequency sweep.
        pts (int): Number of points for the sweep range.
        feedlines (list[str]): list of qubit groups that will be used as
            feedlines. The feedlines to which the given qubits belong will be
            measured.
    """

    def __init__(self, routine, **kwargs):
        self.kw = kwargs

        Step.__init__(self, routine=routine, **kwargs)
        self.experiment_settings = self.parse_settings(
            self.get_requested_settings())

        spec.FeedlineSpectroscopy.__init__(self,
                                           dev=self.dev,
                                           **self.experiment_settings)

    def get_requested_settings(self):
        """Add additional keywords and default values to be passed to the
        FeedlineSpectroscopy class. These are the keywords that are going to be
        looked up in the configuration parameter dictionary.

        Returns:
            dict: Dictionary containing names and default values
                of the keyword arguments needed for the FeedlineSpectroscopy
                class
        """
        settings = super().get_requested_settings()
        # Settings to be read from the settings files
        settings['kwargs']['freq_start'] = (float, 6e9)
        settings['kwargs']['freq_stop'] = (float, 7.5e9)
        settings['kwargs']['pts'] = (int, 1000)
        settings['kwargs']['feedlines'] = (list, [])
        settings['kwargs']['analysis_kwargs'] = (dict, {})
        return settings

    def parse_settings(self, requested_kwargs):
        """
        Searches the keywords given in requested_kwargs in the configuration
        parameter dictionary and prepares the keywords to be passed to the
        FeedlineSpectroscopy class.

        Args:
            requested_kwargs (dict): Dictionary containing the names and the
            default values of the keywords needed for the FeedlineSpectroscopy
            class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the FeedlineSpectroscopy class.
        """
        kwargs = super().parse_settings(requested_kwargs)

        # Frequency sweep points
        freqs = np.linspace(kwargs['freq_start'], kwargs['freq_stop'],
                            kwargs['pts'])

        # Create the feedlines as list of qubits, reading them from the settings
        # which contain them as lists of qubit names
        self.feedlines = []
        for feedline_name in kwargs.pop('feedlines'):
            feedline = []
            for qb in self.dev.qubits:
                if feedline_name in self.get_qubit_groups(qb.name):
                    feedline.append(qb)
            self.feedlines.append(feedline)

        # Create the task list for the measurement. Measure the feedlines of the
        # given qubits.
        kwargs['task_list'] = []
        for feedline in self.feedlines:
            for qb in feedline:
                if qb in self.qubits:
                    task = {'feedline': feedline, 'freqs': freqs}
                    kwargs['task_list'].append(task)
                    break

        # Remove qubits keyword because FeedlineSpectroscopy takes care of it
        # internally
        kwargs.pop('qubits')

        return kwargs

    def run(self):
        """Runs the FeedlineSpectroscopy experiment and the analysis for it.
        """
        # Set 'measure', 'analyze' and `update` of ResonatorSpectroscopy to True
        # in order to use its autorun() function
        self.experiment_settings['measure'] = True
        self.experiment_settings['analyze'] = True
        self.experiment_settings['update'] = True
        self._update_parameters(**self.experiment_settings)
        self.autorun(**self.experiment_settings)
        self.results: Dict[str, Dict[str, float]] = self.analysis.fit_res


class ResonatorSpectroscopyFluxSweepStep(spec.ResonatorSpectroscopyFluxSweep,
                                         Step):
    """Wrapper for ResonatorSpectroscopyFluxSweep experiment. Performs a 2D
    resonator spectroscopy sweeping the bias voltage.

    Args:
        routine (Step): The parent routine

    Keyword Arguments (for the Step constructor):
        step_label (str): A unique label for this step to be used in the
            configuration parameters files.
        settings (SettingsDictionary obj): The configuration parameters passed
            down from its parent. if None, the dictionary is taken from the
            Device object.
        qubits (list): A list with the Qubit objects which should be part of
            the step.
        settings_user (dict): A dictionary from the user to update the
            configuration parameters with.

    Configuration parameters (coming from the configuration parameter dictionary):
        freq_range (float): Range of the frequency sweep points. The center of
            the sweep points is the RO frequency of the qubit. It is possible to
            specify "{adaptive}" to give a smart estimation of the good range.
        freq_pts (int): Number of points for the frequency sweep.
        freq_center (float): Center of the frequency sweep points. The sweep
            points will extend from freq_center - freq_range/2 to
            freq_center + freq_range/2. It is possible to specify "{current}" to
            use the current RO frequency as the center value.
        volt_range (float): Range of the bias voltage sweep points. The center
            of the sweep points is 0 V.
        volt_pts (int): Number of points for the bias voltage sweep.
        volt_center (float): Center of the voltage sweep points. The sweep
            points will extend from volt_center - volt_range/2 to
            volt_center + volt_range/2. It is possible to specify "{current}" to
            use the current voltage bias as the center value.
        expected_dips_width (float): Expected width of the dips (in Hz). This
            value is used for the calculation of prominence (for more information
            see analysis_v2.spectroscopy_analysis.ResonatorSpectroscopy1DAnalysis).
    """

    def __init__(self, routine, **kwargs):
        self.kw = kwargs

        Step.__init__(self, routine=routine, **kwargs)
        self.experiment_settings = self.parse_settings(
            self.get_requested_settings())
        spec.ResonatorSpectroscopyFluxSweep.__init__(self,
                                                     dev=self.dev,
                                                     **self.experiment_settings)
        self.DEFAULT_FREQ_RANGE = 150e6

    def parse_settings(self, requested_kwargs):
        """
        Searches the keywords for the ResonatorSpectroscopyFluxSweep
        given in requested_kwargs in the configuration parameter dictionary.

        Args:
            requested_kwargs (dict): Dictionary containing the names and the
            default values of the keywords needed for the
            ResonatorSpectroscopyFluxSweep class.

        Returns:
            dict: Dictionary containing all keywords with values to be passed to
                the ResonatorSpectroscopyFluxSweep class.
        """
        kwargs = super().parse_settings(requested_kwargs)
        kwargs['task_list'] = []
        # Build the tasks for the remaining qubits to be measured using the
        # default settings
        for qb in self.qubits:
            # Build frequency sweep points
            freq_pts = self.get_param_value('freq_pts', qubit=qb.name)

            freq_center = self.get_param_value('freq_center', qubit=qb.name)
            if isinstance(freq_center, str):
                freq_center = eval(
                    freq_center.format(current=qb.ro_freq(),
                                       between_dips=self.get_freq_between_dips(
                                           qb)))

            freq_range = self.get_param_value('freq_range', qubit=qb.name)
            if isinstance(freq_range, str):
                freq_range = eval(
                    freq_range.format(
                        adaptive=self.get_adaptive_freq_range(qb, freq_center)))

            log.debug(f'{self.step_label}: '
                      f'{freq_center=}, {freq_range=}, {freq_pts=}')
            freqs = np.linspace(freq_center - freq_range / 2,
                                freq_center + freq_range / 2, freq_pts)

            # Build voltage sweep points
            current_voltage = self.routine.fluxlines_dict[qb.name]
            volt_range = self.get_param_value('volt_range', qubit=qb.name)
            volt_pts = self.get_param_value('volt_pts', qubit=qb.name)
            volt_center = self.get_param_value('volt_center', qubit=qb.name)
            if isinstance(volt_center, str):
                volt_center = eval(
                    volt_center.format(current=current_voltage))
            volts = np.linspace(volt_center - volt_range / 2,
                                volt_center + volt_range / 2, volt_pts)

            kwargs['task_list'].append({
                'qb': qb.name,
                'freqs': freqs,
                'volts': volts
            })
        kwargs['fluxlines_dict'] = self.routine.fluxlines_dict

        # If the parameter expected_dips_width was specified, add it to
        # the analysis_kwargs dictionary
        expected_dips_width = self.get_param_value(
            'expected_dips_width', default=self.routine.NotFound())
        if type(expected_dips_width) != self.routine.NotFound:
            try:
                # Add expected_dips_width to analysis_kwargs if it already
                # exists
                kwargs['analysis_kwargs'][
                    'expected_dips_width'] = expected_dips_width
            except KeyError:
                # Otherwise create the analysis_kwargs dictionary
                kwargs['analysis_kwargs'] = {
                    "expected_dips_width": expected_dips_width
                }

        return kwargs

    def _get_two_dips_freqs(self, qubit: QuDev_transmon) -> \
            Tuple[float, float, Dict[str, float]]:
        """Returns:
            ro_frea (float): The designated frequency for the readout.
            mode2_freq (float): The other mode of the RO-Purcell system.
            feedline_results (Dict[str, float]): The full results dictionary
                of the feedline of the qubit."""
        fit_res = self.routine.routine_steps[-1].results
        for feedline_results in fit_res.values():
            for k in feedline_results.keys():
                if k.startswith(f'{qubit.name}'):
                    ro_freq = feedline_results[f'{qubit.name}_RO_frequency']
                    mode2_freq = feedline_results[
                        f'{qubit.name}_mode2_frequency']
                    return ro_freq, mode2_freq, feedline_results

    def get_freq_between_dips(self, qubit: QuDev_transmon) -> float:
        """Find the frequency between the RO-Purcell dips."""
        try:
            ro_freq, mode2_freq, _ = self._get_two_dips_freqs(qubit)
            return np.mean([ro_freq, mode2_freq])
        except (AttributeError, KeyError, TypeError):
            return qubit.ro_freq()

    def get_adaptive_freq_range(self, qubit: QuDev_transmon,
                                freq_center: float) -> float:
        """Calculates a reasonable freq_range such that two flux-oscillating
        dips will be contained in it, but not neighboring dips.
        The full range is determined as the minimum between this distance and
        twice the distance between the two qubit's freqs.
        """

        try:
            ro_freq, mode2_freq, feedline_results = self._get_two_dips_freqs(
                qubit)
            qubit_dips_distance = np.abs(ro_freq - mode2_freq)
            all_other_dips = [v for k, v in feedline_results.items() if
                              all([k.endswith('frequency'),
                                   k.startswith('qb'),
                                   not k.startswith(qubit.name)])]
            if len(all_other_dips) == 0:  # No more qubits on the feedline
                return 3 * qubit_dips_distance
            distance_from_nearest_dip = np.abs(np.array(all_other_dips) -
                                               freq_center).min()
            if distance_from_nearest_dip < qubit_dips_distance:
                log.warning("The neighboring dip is very close to the two dips."
                            "Frequency range might not be appropriate.")
            return np.min([distance_from_nearest_dip,
                           3 * qubit_dips_distance])
        except (AttributeError, KeyError, TypeError):
            return self.DEFAULT_FREQ_RANGE

    def run(self):
        """Runs the ResonatorSpectroscopyFluxSweep experiment and the analysis
        for it.
        """
        # Set 'measure' and 'analyze' of ResonatorSpectroscopyFluxSweep to True
        # in order to use its autorun() function
        self.experiment_settings['measure'] = True
        self.experiment_settings['analyze'] = True
        self.experiment_settings['update'] = True
        self._update_parameters(**self.experiment_settings)
        self.autorun(**self.experiment_settings)
        self.results: Dict[str, Dict[str, float]] = self.analysis.fit_res
        self.update_qubits_fr_and_coupling()

    def update_qubits_fr_and_coupling(self):
        """Update the keys 'fr' and 'coupling' of the qubit Hamiltonian model.

        Update the `fit_ge_freq_from_dc_offset()` dictionary with an estimation
        from the step results.
        Note that this estimation uses the qubit
        frequencies at the sweet spots, which are estimated from the parameters
        'E_c', 'Ej_max' and 'asymmetry' which might be only design values. """

        for qb in self.qubits:
            hamiltonian_fit_params = qb.fit_ge_freq_from_dc_offset()
            if not all([k in hamiltonian_fit_params.keys() for k in [
                    'E_c', 'Ej_max', 'asymmetry']]):
                log.warning("Could not estimate 'coupling' and 'fr' for qubit"
                            f"{qb.name} since not all required parameters were"
                            f"provided.")
                continue
            else:
                fit_results = self.results[qb.name]
                uss_transmon_freq = qb.calculate_frequency(model='transmon',
                                                           flux=0)
                lss_transmon_freq = qb.calculate_frequency(model='transmon',
                                                           flux=0.5)
                sides = ('left', 'right')
                uss_readout_freq = np.mean([fit_results[f'{side}_uss_freq'] for
                                            side in sides])
                lss_readout_freq = np.mean([fit_results[f'{side}_lss_freq'] for
                                            side in sides])
                coupling = routines_utils.get_transmon_resonator_coupling(
                    qubit=qb,
                    uss_transmon_freq=uss_transmon_freq,
                    lss_transmon_freq=lss_transmon_freq,
                    uss_readout_freq=uss_readout_freq,
                    lss_readout_freq=lss_readout_freq
                )
                dips = ('left_uss', 'right_uss', 'left_lss', 'right_lss')
                fr = np.mean([fit_results[f'{dip}_freq'] for dip in dips])
                self.results[qb.name].update({'fr': fr, 'coupling': coupling})
                hamiltonian_fit_params.update({'fr': fr, 'coupling': coupling})


class InitialQubitParking(AutomaticCalibrationRoutine):
    """
    Routine to find the RO frequency of the qubits and park them at their sweet
    spot. It consists of two steps:

    1) FeedlineSpectroscopy, to find the frequency of the resonators. The
        feedlines to which the given qubits belong to will be measured.
        The RO frequency of all the qubits in the measured feedlines will be
        updated.
        FIXME: Only update the RO frequency of the qubits passed to the routine.
    2) ResonatorSpectroscopyFluxSweep, to find the USS and LSS of each given
        qubit, as well as the resonator frequency at finite bias voltage.
        The RO frequency of the given qubits will be updated, as well as their
        bias voltage, and the entries 'dac_sweet_spot' and 'V_per_phi0' of the
        fit_ge_from_dc_offset dictionary.

    Args:
        dev (Device obj): the device which is currently measured
        fluxlines_dict (dict): dictionary containing the qubits names as
            keys and the flux lines QCoDeS parameters as values.

    Keyword Arguments:
        qubits (list(Qudev_transmon)): qubits on which to perform the
            measurement.

    Examples::

        settings_user = {
            'InitialQubitParking': {'General': {
                                     'save_instrument_settings': True}},
            'FeedlineSpectroscopy': {'pts': 1000},
            'ResonatorSpectroscopyFluxSweep': {'freq_pts': 500},
        }

        initial_qubit_parking = InitialQubitParking(dev=dev,
                                                fluxlines_dict=fluxlines_dict,
                                                settings_user=settings_user,
                                                qubits=[qb1, qb6],
                                                autorun=False)
        initial_qubit_parking.view()
        initial_qubit_parking.run()

    """

    def __init__(
            self,
            dev: Device,
            fluxlines_dict: Dict[str, Any],
            **kw,
    ):
        # FIXME: fluxlines_dict has to be passed as an argument because the
        #  fluxlines are not available directly from the qubit objects.
        self.fluxlines_dict = fluxlines_dict

        super().__init__(
            dev=dev,
            **kw,
        )
        self.final_init(**kw)

    def create_routine_template(self):
        """
        Creates routine template.
        """
        super().create_routine_template()
        # Loop in reverse order so that the correspondence between the index
        # of the loop and the index of the routine_template steps is preserved
        # when new steps are added
        for i, step in reversed(list(enumerate(self.routine_template))):
            self.split_step_for_parallel_groups(index=i)

    _DEFAULT_ROUTINE_TEMPLATE = RoutineTemplate([
        [FeedlineSpectroscopyStep, 'feedline_spectroscopy', {}],
        # TODO: add 2D resonator spectroscopy amplitude sweep to find the
        #  optimal RO amplitude
        [ResonatorSpectroscopyFluxSweepStep, 'resonator_spectroscopy_flux', {}]
    ])
