import os
import logging
import pycqed as pq
from uuid import getnode as get_mac

def get_default_datadir():
    """
    Returns the default datadir by first looking in the setup dict and
    searching for a location defined by the mac address. If there is no
    location specified, it will fall back to pycqed/data as a default location.
    """
    # If the mac_address is unknown
    # Stores data in the default data location (pycqed_py3/data/)
    datadir = os.path.abspath(os.path.join(
        os.path.dirname(pq.__file__), os.pardir, 'data'))
    print(datadir)
    logging.info('Setting datadir to default location: {}'.format(
        datadir))
    return datadir
