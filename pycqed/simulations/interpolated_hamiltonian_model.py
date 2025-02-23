import numpy as np
from scipy.interpolate import interp1d

class InterpolatedHamiltonianModel:
    """Provides a simple class to wrap the scipy 1D interpolation method to
    speed-up the computation of the Hamiltonian model when this is called
    several times, e.g., when called in a loop as part of a fitting routine.

    Currently, this class is meant to be used by manually initiallizing an
    instance and manually calling the instance instead of
    `qb.calculate_frequency`.
    TODO: In future, we might extend the qb.calculate_frequency
    method to automatically use the interpolated model in appropriate use
    cases.
    """

    def __init__(self, qb, n_steps=1001, flux=None, **kw):
        """
        Args:
            qb (QuDev_transmon): Qb to which the Hamiltonian model corresponds
            n_steps (int): Number of linear spaced points within
                [0, 1] to be used as flux points for the interpolation data
                points. Defaults to 1001.
            flux (array, optional): Initial flux points to compute the data
                points used in the interpolation. In units of phi_0. If None
                one needs to specify n_steps instead. Defaults to None.
            kw (optional):
                - kind: Specifies the interpolation type, see `scipy.interp1d`.
                        Defaults to 'cubic'.
                - all other kwargs are directly passed to `scipy.interp1d`.
        """
        self._qb = qb
        self._flux = np.linspace(0, 1.0, n_steps) if flux is None else flux
        self._data_points = []
        self.update_frequency_points()
        self.model = interp1d(self._flux, self._data_points,
                              kind=kw.pop('kind', 'cubic'),
                              copy=False,
                              **kw)

    def update_frequency_points(self, flux=None):
        """Update the data points used for interpolation.

        Needs to be called if the actual Hamiltonian model in the qubit changed

        Args:
            flux (array, optional): New flux points that should be used for
                computing the interpolation data. If None, the old flux
                points (self._flux) will be used. Defaults to None.
        """
        if flux is not None:
            self._flux = flux
        self._data_points = self._qb.calculate_frequency(
            flux=self._flux, transition=['ge', 'ef'])

    def __call__(self, flux=None, bias=None, amplitude=None,
                 transition=('ge',)):
        """Compute transition frequencies at the specified flux-bias-amplitude
        points using interpolation.

        Emulates behavior of QuDev_transmon.calculate_frequency.

        Args:
            flux (array, optional): Flux points to be evaluated.
                Defaults to None.
            bias (array, optional): Bias points to be evaluated. Overwrites flux
                if specified. Defaults to None. In case flux and bias are both
                None (default), the method will assume the parking position
                and make flux excursion around this flux point according to
                the amplitudes.
            amplitude (array, optional): Flux pulse amplitudes that are added
                ontop of the specified flux or dc bias points. Needs to have the
                same length as flux or bias. Defaults to None.
            transition (list, optional): Which transition frequencies should be
                calculated. Defaults to ('ge'). Note that this is called
                transition and not transitions to match the signature of
                QuDev_transmon.calculate_frequency.

        Raises:
            ValueError: Raised in case flux pulse amplitude is provided but the
                flux_amplitude_bias_ratio saved in the qubit is None.

        Returns:
            list: List containing the computed frequencies for each transition
                (or list of arrays if flux, bias or amplitude are arrays).
        """

        flux_amplitude_bias_ratio = self._qb.flux_amplitude_bias_ratio()
        if flux_amplitude_bias_ratio is None:
            if amplitude != 0:
                raise ValueError('flux_amplitude_bias_ratio is None, but is '
                                 'required for this calculation.')

        vfc = self._qb.fit_ge_freq_from_dc_offset()
        if bias is None and flux is None:
            flux = self._qb.flux_parking()
        if bias is not None:
            flux = (bias - vfc['dac_sweet_spot']) / vfc['V_per_phi0']

        if amplitude is not None and not np.all(amplitude == 0):
            flux += (amplitude / flux_amplitude_bias_ratio) / vfc['V_per_phi0']
        # Note: the interpolation model is computed for flux points in [0, 1],
        # so the flux points are here passed modulo 1 to the model.
        freqs = self.model(flux % 1.0)
        freqs = [
            {'ge': freqs[0], 'ef': freqs[1], 'gf': freqs[0] + freqs[1]}[t]
            for t in transition]
        return freqs
