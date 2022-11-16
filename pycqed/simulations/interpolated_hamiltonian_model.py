import numpy as np
from scipy.interpolate import interp1d

class InterpolatedHamiltonianModel:
    """Provides as simple class to wrap the scipy 1D interpolation method to
    speed-up the computation of the Hamiltonian model when this is called
    several times, e.g. when called in a loop as part of a fitting routine.
    """

    def __init__(self, qb, n_steps=1001, flux=None, **kw):
        """
        Args:
            qb (QuDev_transmon): Qb to which the Hamiltonian model correspondss
            flux (array): Initial flux points to compute the data points used in
                the interpolation. Will be mapped to domain [0, 1]. If None one
                needs to specify n_steps instead. Defaults to None.
            n_steps (int, optional): Number of linear spaced points within
                [0, 1] to be used as flux points for the interpolation data
                points. Defaults to 1001.
            kw (optional):
                - kind: Specifies the interpolation type, see `scipy.interp1d`.
                        Defaults to 'cubic'.
                - all other kwargs are directly passed to `scipy.interp1d`.
        """
        self._qb = qb
        self._flux = np.linspace(0, 1.0, n_steps) if flux is None else flux
        self._data_points = []
        self.update_data_points()
        self.model = interp1d(self._flux, self._data_points,
                              kind=kw.pop('kind', 'cubic'),
                              copy=False,
                              **kw)

    def update_data_points(self, flux=None):
        """Update the data points used for interpolation.

        Needs to be called if the actual Hamiltonian model in the qubit changed

        Args:
            flux (array, optional): New flux points that should be used for
                computing th einterpolation data. If None, the previously
                specified flux points will be used. Defaults to None.
        """
        if flux is not None:
            self._flux = flux
        self._data_points = self._qb.calculate_frequency(flux=self._flux,
                                                         return_ge_and_ef=True)

    def __call__(self, flux=None, bias=None, amplitude=None, transition='ge',
                 return_ge_and_ef=False):
        """Compute transition frequencies at the specified flux-bias-amplitude
        points using interpolation.

        Emulates behavior of QuDev_transmon.calculate_frequency.

        Args:
            flux (array, optional): Flux points to be evaluated.
                Defaults to None.
            bias (array, optional): Bias points to be evaluated. Overwrites flux
                if specified. Defaults to None.
            amplitude (array, optional): Flux pulse amplitudes that are added
                ontop of the specified flux or dc bias points. Needs to have the
                same length as flux or bias. Defaults to None.
            transition (str, optional): Which transition frequency should be
                calculated. Either 'ge' or 'ef'. Defaults to 'ge'.
            return_ge_and_ef (bool, optional): Whether to return both
                transitions, 'ge' or 'ef'. Overwrites the choice taken in
                argument transition. Defaults to False.

        In case flux and bias are both None (default), the method will assume
        the parking position and make flux excursion around this flux point
        according to amplitudes.

        Note: Before calling the interpolation, the flux-bias-amplitude points
        will be mapped to the [0, 1] flux domain.

        Raises:
            ValueError: Raised in case flux pulse amplitude is provided but the
                flux_amplitude_bias_ratio saved in the qubit is None.

        Returns:
            array: Array containing the computed frequencies.
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
        freqs = self.model(flux % 1.0)
        if return_ge_and_ef:
            return freqs
        elif transition=='ge':
            return freqs[0]
        elif transition=='ef':
            return freqs[1]
        else:
            raise NotImplementedError('transition either needs to be one of '
                                      '["ge", "ef"] or return_ge_and_ef needs '
                                      'to be True.')