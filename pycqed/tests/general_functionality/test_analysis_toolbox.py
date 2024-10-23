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
    """Create a mock data folder and check for existing folder."""
    timestamp = '20230101_120000'
    os.makedirs(os.path.join(setup_datadir, '20230101', '120000_test_measurement'))
    
    folder = a_tools.get_folder(timestamp)
    assert folder == os.path.join(setup_datadir, '20230101', '120000_test_measurement')

def test_atools_get_folder_bad_path(setup_datadir):
    """Set datadir to None to simulate a nonexistant path."""

    # Test that accessing a non-existent folder raises ValueError
    with pytest.raises(Exception):
        a_tools.get_folder('20230101_120000')