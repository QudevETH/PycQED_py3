def ro_flux_tmp_vals(qb, v_park=None, use_ro_flux=None, **kw):
    """Create temporary values for flux-pulse-assisted read-out

    This function can calculate a temporary ro_flux_amplitude for a
    temporary parking position v_park, based on the information provided in
    the following qcodes parameters of the qubit object, which need to be
    set properly prior to calling the function:
    - flux_parking
    - flux_amplitude_bias_ratio
    - fit_ge_freq_from_dc_offset
    - ro_flux_amplitude if the qubit was already configured for fp-assisted RO

    If the qubit was not yet configured for fp-assisted RO, it can be
    temporarily configured for it (if use_ro_flux is True) based on default
    parameter settings or on settings provided as kw. Note that the parameter
    flux_pulse_channel needs to configured before calling the function in this
    case.

    The function can also be used to temporarily deactivate fp-assisted RO.

    Args:
        qb (QuDev_transmon): qubit for which the fp-assisted RO should be
            configured temporarily
        v_park (float, None): DC bias voltage applied to the qubit at the
            moment the read-out happens. The default value None means that
            the the usual DC bias corresponding to qb.flux_parking() is
            applied.
        use_ro_flux (bool, None): whether fp-assisted RO should be used. True
            (False) means that the fp-assisted RO will be temporarily activated
            (deactivated) if it is usually inactive (active). The default
            value None means that the temporary values will not change
            whether or not fp-assisted RO is active, but only adjust the
            amplitude if it is active.
        kw: If use_ro_flux is True for a qubit for which fp-assisted RO is
            usually not used, the following keyword arguments can be used
            to overwrite the default temporary settings:
            - ro_flux_extend_start
            - ro_buffer_length_start
            - ro_flux_extend_end
            - ro_buffer_length_end
            - ro_flux_gaussian_filter_sigma

    Example::
        with temporary_value(*ro_flux_tmp_vals(qubit,
                                               v_park=v_park,
                                               use_ro_flux=True)):
            # code that runs experiments using the temporary RO settings

    """
    tmp_vals = []
    if use_ro_flux is None:
        use_ro_flux = qb.ro_pulse_type() == 'GaussFilteredCosIQPulseWithFlux'

    if use_ro_flux:
        if qb.ro_pulse_type() == 'GaussFilteredCosIQPulse':
            # temporarily activate and configure fp-assisted RO
            tmp_vals += [
                (qb.ro_pulse_type, 'GaussFilteredCosIQPulseWithFlux'),
                (qb.ro_flux_extend_start, kw.pop(
                    'ro_flux_extend_start', 30e-9)),
                (qb.ro_buffer_length_start, kw.pop(
                    'ro_buffer_length_start', 40e-9)),
                (qb.ro_flux_extend_end, kw.pop(
                    'ro_flux_extend_end', 120e-9)),
                (qb.ro_buffer_length_end, kw.pop(
                    'ro_buffer_length_end', 130e-9)),
                (qb.ro_flux_gaussian_filter_sigma, kw.pop(
                    'ro_flux_gaussian_filter_sigma', .5e-9)),
                (qb.ro_flux_channel, qb.flux_pulse_channel()),
            ]
            ro_flux_amp = 0
            if v_park is None:
                # no new DC bias: configure a zero-amplitude flux pulse
                tmp_vals += [(qb.ro_flux_amplitude, ro_flux_amp)]
        else:  # fp-assisted RO already used
            ro_flux_amp = qb.ro_flux_amplitude()

        if v_park is not None:
            # calculate and configure amplitude for new DC bias
            delta_v = v_park - qb.calculate_voltage_from_flux(
                qb.flux_parking())
            ro_flux_amp -= delta_v * qb.flux_amplitude_bias_ratio()
            tmp_vals += [(qb.ro_flux_amplitude, ro_flux_amp)]

    elif qb.ro_pulse_type() == 'GaussFilteredCosIQPulseWithFlux':
        # temporarily deactivate fp-assisted RO
        tmp_vals += [(qb.ro_pulse_type, 'GaussFilteredCosIQPulse')]

    return tmp_vals