import pytest
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import timedomain_analysis as tda

# Navigate to the folder where the testdata of this test is contained
@pytest.fixture(scope="module", autouse=True)
def set_module_data_dir(test_data_base_dir):
    sub_directory = test_data_base_dir / 'setups/xld'
    a_tools.datadir = str(sub_directory)
    return sub_directory

def test_Rabi():
    for t_start in [
        '20241119_110543',  # avg
        '20241120_080117',  # avg with reset
        '20241118_141159',  # SSRO
        '20241120_080101',  # SSRO with reset
    ]:
        tda.RabiAnalysis(
            t_start=t_start,
            do_fitting=True,
            options_dict={'delegate_plotting': False},
                extract_only=True,
                raise_exceptions=True,
        )