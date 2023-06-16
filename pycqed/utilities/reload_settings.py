import pycqed.utilities.general as gen
import logging

log = logging.getLogger(__name__)


def reload_settings(timestamp=None, timestamp_filters=None, load_flux_bias=True,
                    qubits=None, dev=None,
                    DCSources=None,
                    fluxlines_dict=None,
                    **kw):
    """
    Reload settings from the database.

    Attributes:
        timestamp: Timestamp from which to reload settings
        timestamp_filters: Timestamp from which to reload predistortion filters
        load_flux_bias: If True, sets the DC source biases for the flux lines
        qubits: Qubits for which to reload settings
        dev: Device object to reload. qubits are taken from dev if None
        DCSources: DCSources to reload if load_flux_bias is True
        fluxlines_dict: Dictionary containing the flux lines qcodes parameters
    """
    if qubits is None:
        qubits = kw.get('qbs')  # allow qbs as an alias for qubits
    if qubits is None:
        qubits = dev.get_qubits()  # take all qubits from the device object

    for qb in qubits:
        gen.load_settings(qb, timestamp=timestamp)
        if timestamp_filters is not None:
            gen.load_settings(qb, timestamp=timestamp_filters,
                              params_to_set=[f'flux_distortion'])
        # set offsets and create filters in pulsar based on settings in the qubits
        qb.configure_pulsar()
    if dev is not None:
        gen.load_settings(dev, timestamp=timestamp)
        dev.configure_pulsar()

    if load_flux_bias:  # reload and set flux bias

        if DCSources is None or fluxlines_dict is None:
            ts = f"({timestamp}) " if timestamp is not None else ""
            log.warning(
                f"DCSources and fluxlines_dict must be specified if user "
                f"wants to load flux bias. DCSources NOT reloaded. You can "
                f"retry loading the settings {ts}by passing the DCSources and "
                f"fluxlines_dict.")
            return

        for DCSource in DCSources:
            DCSource_params = gen.load_settings(DCSource, timestamp=timestamp,
                                                update=False)
            set_fluxline_dict = {
                fluxlines_dict[qb.name].name[5:]:  # strip "volt_"
                eval(DCSource_params[fluxlines_dict[qb.name].name])
                for qb in qubits
                if fluxlines_dict[qb.name].instrument == DCSource
            }
            print(f"Setting flux line voltages to {set_fluxline_dict}")
            if 'Virtual' in DCSource.__class__.__name__:
                for k, v in set_fluxline_dict.items():
                    DCSource.set(f'volt_{k}', v)
            else:
                DCSource.set_smooth(set_fluxline_dict)
