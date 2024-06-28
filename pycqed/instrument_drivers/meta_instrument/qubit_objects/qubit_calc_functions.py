import numpy as np
from copy import deepcopy
import pycqed.analysis.fitting_models as fit_mods

import logging
log = logging.getLogger(__name__)

class QubitCalcFunctionsMixIn:
    """MixIn with methods for calculations based on qubit parameters"""

    def calculate_frequency(self, bias=None, amplitude=0, transition='ge',
                            model='transmon_res', flux=None, update=False):
        """
        Calculates the transition frequency for a given DC bias and flux
        pulse amplitude using fit parameters stored in the qubit object.
        Note that the qubit parameter flux_amplitude_bias_ratio is used for
        conversion between bias values and amplitudes.

        :param bias: (float) DC bias. If model='approx' is used, the bias is
            optional, and is understood relative to the parking position at
            which the  model was measured. Otherwise, it mandatory and is
            interpreted as voltage of the DC source.
        :param amplitude: (float, default: 0) flux pulse amplitude
        :param transition: (str or list of str, default: 'ge') the transition
            or transitions whose frequency should be calculated.
        :param model: (str, default: 'transmon_res') the model to use.
            'approx': Qubit_dac_to_freq with parameters from
                the qubit parameter fit_ge_freq_from_flux_pulse_amp.
                bias is understood as relative to the parking position.
            'transmon': Qubit_dac_to_freq_precise with parameters from
                the qubit parameter fit_ge_freq_from_dc_offset.
                bias is understood as the voltage of the DC source.
            'transmon_res': Qubit_dac_to_freq_res with parameters from
                the qubit parameter fit_ge_freq_from_dc_offset.
                bias is understood as the voltage of the DC source.
        :param flux: (float, default None) if this is not None, the frequency
            is calculated for the given flux (in units of phi_0) instead of
            for the given bias (for models 'transmon' and 'transmon_res') or
            instead of the given amplitude (for model 'approx'). If both bias
            and flux are None and the model is 'transmon' or 'transmon_res',
            the flux value from self.flux_parking() is used.
        :param update: (bool, default False) whether the result should be
            stored as {transition}_freq parameter of the qubit object.
        :return: calculated transition frequency/frequencies

        TODO: Add feature to automatically make use of
              `InterpolatedHamiltonianModel` to speed up the computation in
              certain use cases.
        """

        if isinstance(transition, (list, tuple)):
            return_list = True
        else:
            transition = [transition]
            return_list = False

        for t in transition:
            if t not in ['ge', 'ef', 'gf']\
                    or (t != 'ge' and model not in ['transmon_res']):
                raise NotImplementedError(
                    f'calculate_frequency: Currently, transition {t} '
                    f'is not implemented for model {model}.')
        flux_amplitude_bias_ratio = self.flux_amplitude_bias_ratio()
        if flux_amplitude_bias_ratio is None:
            if ((model in ['transmon', 'transmon_res'] and amplitude != 0) or
                    (model == ['approx'] and bias is not None and bias != 0)):
                raise ValueError('flux_amplitude_bias_ratio is None, but is '
                                 'required for this calculation.')

        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
            if bias is None and flux is None:
                flux = self.flux_parking()
            if flux is not None:
                bias = self.calculate_voltage_from_flux(flux, model)
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()
            if flux is not None:
                amplitude = self.calculate_voltage_from_flux(flux, model)

        if model == 'approx':
            freqs = [fit_mods.Qubit_dac_to_freq(
                amplitude + (0 if bias is None or np.all(bias == 0) else
                             bias * flux_amplitude_bias_ratio), **vfc)]
        elif model == 'transmon':
            kw = deepcopy(vfc)
            kw.pop('coupling', None)
            # FIXME: 'fr' refers to the bare readout-resonator frequency,
            #  this is not a very descriptive name. Should it be changed to
            #  'bare_ro_res_freq'? This is relevant to the device database.
            kw.pop('fr', None)
            freqs = [fit_mods.Qubit_dac_to_freq_precise(bias + (
                0 if np.all(amplitude == 0)
                else amplitude / flux_amplitude_bias_ratio), **kw)]
        elif model == 'transmon_res':
            freqs = fit_mods.Qubit_dac_to_freq_res(
                bias + (0 if np.all(amplitude == 0)
                        else amplitude / flux_amplitude_bias_ratio),
                return_ef=True, **vfc)
            freqs = [
                {'ge': freqs[0], 'ef': freqs[1], 'gf': freqs[0]+freqs[1]}[t]
                for t in transition]
        else:
            raise NotImplementedError(
                "Currently, only the models 'approx', 'transmon', and"
                "'transmon_res' are implemented.")
        if update:
            for t, f in zip(transition, freqs):
                if f'{t}_freq' in self.parameters:
                    self.parameters[f'{t}_freq'](f)
                else:
                    log.warning(f'Cannot set the frequency of transition {t}!')
        if return_list:
            return freqs
        else:
            return freqs[0]

    def calculate_flux_voltage(self, frequency=None, bias=None,
                               amplitude=None, transition='ge',
                               model='transmon_res', flux=None,
                               branch=None):
        """
        Calculates the flux pulse amplitude or DC bias required to reach a
        transition frequency using fit parameters stored in the qubit
        object. Note that the qubit parameter flux_amplitude_bias_ratio is
        used for conversion between bias values and amplitudes.
        :param frequency: (float, default: None = use self.ge_freq())
            transition frequency
        :param bias: (float, default; None) DC bias. If None, the function
            calculates the required DC bias to reach the target frequency
            (potentially taking into account the given flux pulse amplitude).
            Otherwise, it fixes the DC bias and calculates the required pulse
            amplitude. See note below.
        :param amplitude: (float, default: None) flux pulse amplitude. If None,
            the function calculates the required pulse amplitude to reach
            the target frequency (taking into account the given bias).
            Otherwise, it fixes the pulse amplitude and calculates the
            required bias. See note below.
        :param transition: (str, default: 'ge') the transition whose
            frequency should be calculated. Currently, only 'ge' is
            implemented for all models. The model 'transmon_res' also allows to
            compute the 'ef' and 'gf' transition.
        :param model: (str, default: 'transmon_res') the model to use.
            Currently 'transmon_res' and 'approx' are supported. See
            docstring of self.calculate_frequency
        :param flux: (float, default None) if this is not None, the bias
            parameter is overwritten with the bias corresponding to the given
            flux (in units of phi_0) for models 'transmon' and 'transmon_res'.
            This parameter is ignored if the model is 'approx'.
        :param branch: which branch of the flux-to-frequency curve should be
            used. See the meaning of this parameter in Qubit_freq_to_dac
            and Qubit_freq_to_dac_res. If None, this is set to the bias (if
            not None)
        :return: calculated bias or amplitude, depending on which parameters
            are passed in (see above and notes below).

        Notes:
        If model='approx' is used, the bias (parameter or return
            value) is understood relative to the parking position at
            which the model was measured. Otherwise, it is interpreted as
            voltage of the DC source.
        If both bias and amplitude are None, an amplitude is returned if the
            model is 'approx'. For the other models, a bias is returned in
            this case.
        """

        if frequency is None:
            frequency = self.ge_freq()
        if model != 'transmon_res' and transition not in ['ge']:
            raise NotImplementedError(
                'Currently, only ge transition is implemented.')
        elif transition not in ['ge', 'ef', 'gf']:
            raise NotImplementedError(
                'Currently, only the ge, ef & gf transitions are implemented.')
        flux_amplitude_bias_ratio = self.flux_amplitude_bias_ratio()

        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
            if flux is not None:
                bias = self.calculate_voltage_from_flux(flux, model)
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()

        if flux_amplitude_bias_ratio is None:
            if bias is not None and amplitude is not None:
                raise ValueError(
                    'flux_amplitude_bias_ratio is None, but is '
                    'required for this calculation.')

        if branch is None:
            if bias is None and flux is None:
                branch = 'negative'
            else:
                # select well-defined branch close to requested flux
                if flux is None:
                    flux = (bias - vfc['dac_sweet_spot']) / vfc['V_per_phi0']
                if flux % 0.5:
                    pass  # do not shift (well-defined branch)
                elif flux != self.flux_parking():
                    # shift slightly in the direction of flux parking
                    flux += np.sign(self.flux_parking()-flux) * 0.25
                elif flux != 0:
                    # shift slightly in the direction of 0
                    flux += -np.sign(flux) * 0.25
                else:
                    # shift slightly to the left to use rising branch as default
                    flux = -0.25
                branch = flux * vfc['V_per_phi0'] + vfc['dac_sweet_spot']

        if model == 'approx':
            val = fit_mods.Qubit_freq_to_dac(frequency, **vfc, branch=branch)
        elif model == 'transmon_res':
            val = fit_mods.Qubit_freq_to_dac_res(
                frequency, **vfc, branch=branch, single_branch=True,
                transition=transition)
        else:
            raise NotImplementedError(
                "Currently, only the models 'approx' and"
                "'transmon_res' are implemented.")

        if model in ['transmon', 'transmon_res'] and bias is not None:
            # return amplitude
            val = (val - bias) * flux_amplitude_bias_ratio
        elif model in ['approx'] and bias is not None:
            # return amplitude
            val = val - bias * flux_amplitude_bias_ratio
        elif model in ['transmon', 'transmon_res'] and amplitude is not None:
            # return bias, corrected for amplitude
            val = val - amplitude / flux_amplitude_bias_ratio
        elif model in ['approx'] and amplitude is not None:
            # return bias
            val = (val - amplitude) / flux_amplitude_bias_ratio
        # If both bias and amplitude are None, the bare result is returned,
        # see note in the doctring.
        return val

    def calculate_voltage_from_flux(self, flux, model='transmon_res'):
        """
        Calculates the DC bias for a given target flux.

        :param flux: (float) flux in units of phi_0
        :param model: (str, default: 'transmon_res') the model to use,
            see calculate_frequency.
        :return: calculated DC bias if model is transmon or transmon_res,
            calculated flux pulse amplitude otherwise
        """
        if model in ['transmon', 'transmon_res']:
            vfc = self.fit_ge_freq_from_dc_offset()
        else:
            vfc = self.fit_ge_freq_from_flux_pulse_amp()
        return vfc['dac_sweet_spot'] + vfc['V_per_phi0'] * flux

    def calc_flux_amplitude_bias_ratio(self, amplitude, ge_freq, bias=None,
                                       flux=None, update=False):
        """
        Calculates the conversion factor between flux pulse amplitudes and bias
        voltage changes that lead to the same qubit detuning. The calculation is
        done based on the model Qubit_freq_to_dac_res and the parameters stored
        in the qubit parameter fit_ge_freq_from_dc_offset.

        :param amplitude: (float) flux pulse amplitude
        :param ge_freq: (float) measured ge transition frequency
        :param bias: (float) DC bias, i.e., voltage of the DC source.
        :param flux: (float) if this is not None, the value of the bias
            is overwritten with the voltage corresponding to the given flux
            (in units of phi_0). If both bias and flux are None, the flux
            value from self.flux_parking() is used.
        :param update: (bool, default False) whether the result should be
            stored as flux_amplitude_bias_ratio parameter of the qubit object.
        :return: calculated conversion factor
        """
        if bias is None and flux is None:
            flux = self.flux_parking()
        if flux is not None:
            bias = self.calculate_voltage_from_flux(flux)
        v = fit_mods.Qubit_freq_to_dac_res(
            ge_freq, **self.fit_ge_freq_from_dc_offset(), branch=bias)
        flux_amplitude_bias_ratio = amplitude / (v - bias)
        if flux_amplitude_bias_ratio < 0:
            log.warning('The extracted flux_amplitude_bias_ratio is negative, '
                        'please check your input values.')
        if update:
            self.flux_amplitude_bias_ratio(flux_amplitude_bias_ratio)
        return flux_amplitude_bias_ratio

    def generate_scaled_volt_freq_conv(self, scaling=None, flux=None,
                                       bias=None):
        """
        Generates a scaled and shifted version of the voltage frequency
        conversion dictionary (self.fit_ge_freq_from_dc_offset). This can,
        e.g., be used to calculate flux pulse amplitude to ge frequency
        conversion using fit_mods.Qubit_dac_to_freq_res. This shift is done
        relative to obtain a model that is relative to a flux offset (
        parking position) indicated by either flux or bias.
        :param scaling: the scaling factor. Default: use
            self.flux_amplitude_bias_ratio()
        :param flux: parking position in unit of Phi_0. If both bias and flux
            are None, the flux value from self.flux_parking() is used.
        :param bias: If not None, overwrite flux with the flux resulting from
            the given DC voltage.
        :return: the scaled and shifed voltage frequency conversion dictionary
        """
        vfc = deepcopy(self.fit_ge_freq_from_dc_offset())
        if scaling is None:
            scaling = self.flux_amplitude_bias_ratio()
        if bias is not None:
            flux = (bias - vfc['dac_sweet_spot']) / vfc['V_per_phi0']
        elif flux is None:
            flux = self.flux_parking()
        vfc['V_per_phi0'] *= scaling
        vfc['dac_sweet_spot'] = -flux * vfc['V_per_phi0']
        return vfc


