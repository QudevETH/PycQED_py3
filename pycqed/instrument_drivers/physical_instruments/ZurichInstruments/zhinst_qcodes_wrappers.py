"""
    Qudev specific driver for the HDAWG instrument.
"""

import logging

from zhinst.qcodes import HDAWG as HDAWG_core
from zhinst.qcodes import UHFQA as UHFQA_core
log = logging.getLogger(__name__)


class HDAWG8(HDAWG_core):
    """QuDev-specific PycQED driver for the ZI HDAWG
    """

    def __init__(self, *args, **kwargs):
        self._check_server(kwargs)
        super().__init__(*args, **kwargs)

    def _check_server(self, kwargs):
        if kwargs.pop('server', '') == 'emulator':
            from pycqed.instrument_drivers.physical_instruments \
                .ZurichInstruments import ZI_base_qudev as zibase
            from zhinst.qcodes import session as ziqcsess
            daq = zibase.MockDAQServer.get_instance(
                kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 8004))
            self._session = ziqcsess.ZISession(
                server_host=kwargs.get('host', 'localhost'),
                connection=daq, new_session=False)
            return daq


class UHFQA(UHFQA_core):
    """QuDev-specific PycQED driver for the ZI UHFQA
    """

    def __init__(self, *args, **kwargs):
        self._check_server(kwargs)
        super().__init__(*args, **kwargs)

    def _check_server(self, kwargs):
        if kwargs.pop('server', '') == 'emulator':
            from pycqed.instrument_drivers.physical_instruments \
                .ZurichInstruments import ZI_base_qudev as zibase
            from zhinst.qcodes import session as ziqcsess
            daq = zibase.MockDAQServer.get_instance(
                kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 8004))
            self._session = ziqcsess.ZISession(
                server_host=kwargs.get('host', 'localhost'),
                connection=daq, new_session=False)
            return daq
