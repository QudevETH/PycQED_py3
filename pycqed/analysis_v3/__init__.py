# This file initializes all the node modules in analysis_v3.

# pipeline_analysis.py must be initialized first in order to create the global
# variable search_modules, a set with all modules in analysis_v3 that contain
# node functions. Each of these modules will populate search_modules with
# the functions inside the module.

from pycqed.analysis_v3 import processing_pipeline as pp_mod
from pycqed.analysis_v3 import pipeline_analysis as pla
from pycqed.analysis_v3 import helper_functions as hlp_mod
from pycqed.analysis_v3 import data_extraction as dat_extr_mod
from pycqed.analysis_v3 import flux_distortion as flux_dist_mod
from pycqed.analysis_v3 import fitting as fit_mod
from pycqed.analysis_v3 import plotting as plot_mod
from pycqed.analysis_v3 import saving as save_mod
from pycqed.analysis_v3 import data_processing as dat_proc_mod
from pycqed.analysis_v3 import rabi_analysis as rabi_ana
from pycqed.analysis_v3 import ramsey_analysis as ramsey_ana
from pycqed.analysis_v3 import randomized_benchmarking_analysis as rb_ana
from pycqed.analysis_v3 import tomography_analysis as tomo_ana


from importlib import reload
def reload_anav3():
    reload(pp_mod)
    reload(pla)
    reload(hlp_mod)
    reload(dat_extr_mod)
    reload(fit_mod)
    reload(plot_mod)
    reload(save_mod)
    reload(dat_proc_mod)
    reload(flux_dist_mod)
    reload(rabi_ana)
    reload(ramsey_ana)
    reload(rb_ana)
    reload(tomo_ana)