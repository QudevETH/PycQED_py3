import os
import logging
import pycqed


def get_default_datadir():
    """
    Returns the default datadir in the repository folder.

    This function attempts to locate the default data directory for PyCQED.
    It first checks if the data directory exists in the PyCQED installation folder.
    If not found, it searches for a 'data' directory in the current working directory
    and its parent directories (up to two levels deep).

    Returns:
        str: The path to the default data directory.

    Note:
        The default data location is typically 'pycqed_py3/data/'.
    """

    # Depending upon how pcyqed was installed, pycqed.__file__ can evaluate to None
    if pycqed.__file__ is not None:
        datadir = os.path.abspath(os.path.join(os.path.dirname(pycqed.__file__), os.pardir, 'data'))
    else:
        # Search the directory tree from which this script was called 
        # for a depth of two directories for a directory called "data".
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if 'data' in dirs:
                return os.path.join(root, 'data')
            if root.count(os.sep) - current_dir.count(os.sep) == 2:
                break
    
    # If no 'data' directory is found, return a default path
    datadir = os.path.join(current_dir, 'data')

    # Show the result
    logging.warn('Setting datadir to location: {}'.format(datadir))
    return datadir