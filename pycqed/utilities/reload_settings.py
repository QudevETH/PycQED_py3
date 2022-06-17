import pycqed.utilities.general as gen
import logging

log = logging.getLogger(__name__)

def reload_settings(timestamp=None, timestamp_filters=None, load_flux_bias=True,
                    virtual_setup=False,
                    qubits=None, dev=None,
                    DCSources=None,
                    **kw):
    """
    Reload settings from the database.
    """
    if dev is None:
        raise ValueError('dev must be specified')
    else:
        qubits = dev.get_qubits()

    qbs = kw.get('qubits', kw.get('qbs', None))
    if qbs is None:
        qbs = qubits

    gen.load_settings(dev, timestamp=timestamp)
    for qb in qbs:
        gen.load_settings(qb, timestamp=timestamp)
        if timestamp_filters is not None:
            gen.load_settings(qb, timestamp=timestamp_filters,
                              params_to_set=[f'flux_distortion'])
        # set offsets and create filters in pulsar based on settings in the qubits
        try:
            qb.configure_pulsar()
        except Exception:
            if not virtual_setup:
                raise
    try:
        dev.configure_pulsar()
    except Exception:
        if not virtual_setup:
            raise

    if load_flux_bias:  # reload and set flux bias

        if DCSources is None:
            ts = f"({timestamp}) " if timestamp is not None else ""
            log.warning(f"DCSources must be specified if user wants to load flux bias. DCSources NOT reloaded. You can retry loading the settings {ts}by either passing the DCSources or by reloading settings manually with the reload_settings method from init script")
            return

        for DCSource in DCSources:
            DCSource_params = gen.load_settings(DCSource, timestamp=timestamp,
                                                update=False)
            set_fluxline_dict = {'fluxline_' + qb.name[2:]:
                                     eval(DCSource_params['volt_fluxline_' + qb.name[2:]])
                                 for qb in qubits if qb in qbs}
            print(f"Setting flux line voltages to {set_fluxline_dict}")
            if 'Virtual' in DCSource.__class__.__name__:
                for k, v in set_fluxline_dict.items():
                    DCSource.set(f'volt_{k}', v)
            else:
                DCSource.set_smooth(set_fluxline_dict)
