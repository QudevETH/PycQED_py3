import os
import logging
import pycqed


def get_default_datadir():
    """
    Returns the default datadir in the repostory folder.
    """
    # Stores data in the default data location (pycqed_py3/data/)
    datadir = os.path.abspath(os.path.join(os.path.dirname(pycqed.__file__),
                                           os.pardir, 'data'))
    logging.info('Setting datadir to default location: {}'.format(datadir))
    return datadir
