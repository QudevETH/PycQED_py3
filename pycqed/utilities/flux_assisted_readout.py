from pycqed.utilities.general import temporary_value

def ro_flux_tmp_vals(qubit, v_park=None, use_ro_flux=None, **kw):
    """
    Creates temporary values for flux assisted read-out

    Args:
        qubit: qubit to be read-out
        v_park: voltage where the read-out happens (usually the sweet spot is a good place to do this)
        use_ro_flux: ...
        kw:
            ro_flux_extend_start
            ro_buffer_length_start
            ro_flux_extend_end
            ro_buffer_length_end
            ro_flux_gaussian_filter_sigma

            Note that the above key word arguments will be used as temporary value

    Example:
        with temporary_value(*ro_flux_tmp_vals(qubit,
                                               v_park=v_park,
                                               use_ro_flux=True)):

            [some code that reads out qubit state]

    """

    if use_ro_flux is None:
        use_ro_flux = qubit.ro_pulse_type() == 'GaussFilteredCosIQPulseWithFlux'

    tmp_vals = []

    if use_ro_flux:
        if qubit.ro_pulse_type() == 'GaussFilteredCosIQPulse':
            tmp_vals += [
                (qubit.ro_pulse_type, kw.pop('ro_pulse_type', 'GaussFilteredCosIQPulseWithFlux')),
                (qubit.ro_flux_extend_start, kw.pop('ro_flux_extend_start', 30e-9)),
                (qubit.ro_buffer_length_start, kw.pop('ro_buffer_length_start', 40e-9)),
                (qubit.ro_flux_extend_end, kw.pop('ro_flux_extend_end', 120e-9)),
                (qubit.ro_buffer_length_end, kw.pop('ro_buffer_length_end', 130e-9)),
                (qubit.ro_flux_gaussian_filter_sigma, kw.pop('ro_flux_gaussian_filter_sigma', .5e-9)),
                (qubit.ro_flux_channel, qubit.flux_pulse_channel()),
            ]
            if v_park is not None:
                ro_flux_amplitude = 0

            else:
                tmp_vals += [
                    (qubit.ro_flux_amplitude, 0),
                ]

        elif v_park is not None:
            ro_flux_amplitude = qubit.ro_flux_amplitude()

        if v_park is not None:
            delta_v = v_park - qubit.calculate_voltage_from_flux(qubit.flux_parking())
            ro_flux_amp = ro_flux_amplitude - delta_v * qubit.flux_amplitude_bias_ratio()
            tmp_vals += [(qubit.ro_flux_amplitude, ro_flux_amp)]

    elif (not use_ro_flux) and qubit.ro_pulse_type() == 'GaussFilteredCosIQPulseWithFlux':
        tmp_vals += [
            (qubit.ro_pulse_type, 'GaussFilteredCosIQPulse'),
        ]

    return tmp_vals