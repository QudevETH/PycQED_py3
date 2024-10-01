import pytest
import numpy as np
from pycqed.analysis import analysis_toolbox as a_tools
import os
import tempfile

# We overwrite datadir and restore it after this file to not disturb other tests
@pytest.fixture
def setup_datadir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_datadir = a_tools.datadir
        a_tools.datadir = tmpdirname
        yield tmpdirname
        a_tools.datadir = old_datadir

def test_atools_get_folder_happy_path(setup_datadir):
    """Create a mock data folder and run happy test."""
    timestamp = '20230101_120000'
    os.makedirs(os.path.join(setup_datadir, '20230101', '120000_test_measurement'))
    
    folder = a_tools.get_folder(timestamp)
    assert folder == os.path.join(setup_datadir, '20230101', '120000_test_measurement')

def test_atools_get_folder_bad_path():
    """Set datadir to None to simulate the bad path."""
    old_datadir = a_tools.datadir
    a_tools.datadir = None

    # Expect an error here
    with pytest.raises(ValueError):
        result = a_tools.get_folder('20230101_120000')
        print(result)
    
    a_tools.datadir = old_datadir