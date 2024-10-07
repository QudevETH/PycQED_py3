import pytest
import pycqed
from pathlib import Path
import os

# We assume `pycqed_testdata` is next to `pycqed_py3`
@pytest.fixture(scope="session")
def test_data_base_dir():
    env_test_data_dir = os.environ.get('PYCQED_TESTDATA_DIR')

    if env_test_data_dir:
        test_data_dir = Path(env_test_data_dir)
    else:
        test_data_dir = Path(pycqed.__file__).parent.parent.parent / 'pycqed_testdata/'

    if not test_data_dir.exists():
        pytest.skip(f"Test data directory not found: {test_data_dir}")

    return test_data_dir