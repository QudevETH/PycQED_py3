import pytest
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import timedomain_analysis as tda

# Navigate to the folder where the testdata of this test is contained
@pytest.fixture(scope="module", autouse=True)
def set_module_data_dir(test_data_base_dir):
    sub_directory = test_data_base_dir / 'setups/xld'
    a_tools.datadir = str(sub_directory)
    return sub_directory

def add_default_kw(kw=None):
    if kw is None:
        kw = {}
    kw.setdefault('options_dict', {})
    kw.setdefault('extract_only', False)
    kw.setdefault('raise_exceptions', True)
    kw['options_dict'].setdefault('delegate_plotting', False)
    return kw

def test_Rabi():
    for t_start in [
        '20241120_155114',  # avg
        '20241120_163229',  # avg with reset
        '20241120_155158',  # SSRO
        '20241120_163256',  # SSRO with reset
    ]:
        tda.RabiAnalysis(
            t_start=t_start,
            do_fitting=True,
            **add_default_kw(),
        )

def test_Chevron():
    for t_start, kw in [
        # avg
        ('20241108_012425',
            {
                'do_fitting': True,
            }
        ),
        # SSRO
        (
            '20241107_212929',
            {
                'do_fitting': False,
            }
        ),
        (
            '20241107_212929',
            {
                'do_fitting': False,
                'options_dict': {
                    'predict_proba': True,
                },
            }
        ),
        (
            '20241107_212929',
            {
                'do_fitting': False,
                'options_dict': {
                    'predict_proba': True,
                    'thresholding': True,
                },
            }
        ),
        (
            '20241107_212929',
            {
                'do_fitting': False,
                'options_dict': {
                    'predict_proba': True,
                    'correlate_proba': True,
                },
            }
        ),
    ]:
        add_default_kw(kw)
        tda.ChevronAnalysis(
            qb_names=['qb14', 'qb13'],
            t_start=t_start,
            **kw,
            # options_dict={  # TODO could still vary these
            #     'plot_raw_data': True,
            #     'plot_proj_data': True,
            #     'rotate': True,
            # },
        )

def test_CPhase():
    for t_start, kw in [
        ('20241116_160921', {}),  # 1D
        ('20241116_155729', {}),  # 2D
    ]:
        add_default_kw(kw)
        tda.CPhaseLeakageAnalysis(
            qb_names=['qb5', 'qb10'],
            t_start=t_start,
            **kw,
        )
