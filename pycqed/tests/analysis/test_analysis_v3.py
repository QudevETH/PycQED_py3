import pytest
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v3 import plotting as plot_mod

# Navigate to the folder where the testdata of this test is contained
@pytest.fixture(scope="module", autouse=True)
def set_module_data_dir(test_data_base_dir):
    sub_directory = test_data_base_dir / 'setups/xld'
    a_tools.datadir = str(sub_directory)
    a_tools.fetch_data_dir = (r'Q:\Archive\Qudev107 - QComp PycQED data from '
                              r'XLD\pydata')
    return sub_directory

@pytest.mark.skip(reason="FIXME: Move to integration tests")
def test_1qb_IRB_analysis():
    from pycqedscripts.scripts.characterization import \
        randomized_benchmarking as rb_ana_full
    timestamp = '20220203_183543'
    plot_mod.get_default_plot_params()
    pp = rb_ana_full.single_qubit_rb_analysis(
        timestamp=timestamp,
        extract_T2s=True,
        renormalize_qb=(False,),
        save_figures=True,
        save_processed_data=False,
    )

@pytest.mark.skip(reason="FIXME: Move to integration tests")
def test_2qb_IRB_analysis():
    from pycqedscripts.scripts.characterization import \
        randomized_benchmarking as rb_ana_full
    timestamp = '20220722_131009'
    plot_mod.get_default_plot_params()
    pp = rb_ana_full.two_qubit_irb_analysis(
        timestamp=timestamp,
        save_figures=True,
        save_processed_data=False,
    )

def test_1qb_XEB_analysis():
    import \
        pycqed.analysis_v3.cross_entropy_benchmarking_analysis as xeb_ana
    timestamp = '20210722_120747'
    plot_mod.get_default_plot_params()
    renormalize = True  # renormalise qubit subspace pg + pg = 1
    pp, meas_obj_names, cycles, nr_seq = xeb_ana.single_qubit_xeb_analysis(
        timestamp, renormalize=renormalize, save=False)
    xeb_ana.plot_porter_thomas_dist(pp.data_dict, renormalize=renormalize,
                                    savefig=True)
    xeb_ana.calculate_fidelities_purities_1qb(
        pp.data_dict,
        renormalize=renormalize,
        # on the ro-corrected data; pass 'average_data' otherwise
        data_key='correct_readout',
    )
    results_dict = xeb_ana.fit_plot_fidelity_purity(
        pp.data_dict,
        idx0f=0,  # fit fidelity starting at cycle index 0
        idx0p=1,
        # fit purity starting at 1; usually needs more randomisation
        joint_processing=True,
        log_scale=False,
        savefig=True,
    )
    xeb_ana.fit_plot_leakage_1qb(pp.data_dict, meas_obj_names)

def test_2qb_XEB_analysis():
    import pycqed.analysis_v3.cross_entropy_benchmarking_analysis as xeb_ana
    timestamp = '20220722_141951'
    plot_mod.get_default_plot_params()
    renormalize = True  # renormalise qubit subspace pgg + pge + peg + pee = 1
    pp, meas_obj_names, cycles, nr_seq = xeb_ana.two_qubit_xeb_analysis(
        timestamp, renormalize=renormalize, save=False)
    xeb_ana.plot_porter_thomas_dist(pp.data_dict, renormalize=renormalize,
                                    savefig=True)
    xeb_ana.calculate_fidelities_purities_2qb(
        pp.data_dict,
        renormalize=renormalize,
        # on the ro-corrected data; pass 'average_data' otherwise
        data_key='correct_readout',
    )
    results_dict = xeb_ana.fit_plot_fidelity_purity(
        pp.data_dict,
        idx0f=0,  # fit fidelity starting at cycle index 0
        idx0p=1,  # fit purity starting at 1; usually needs more randomisation
        joint_processing=True,
        log_scale=False,
        savefig=True,
    )
    xeb_ana.fit_plot_leakage_2qb(pp.data_dict, meas_obj_names)

# TODO 1qb State Tomography
# from pycqedscripts.scripts.characterization.state_tomo import\
#     do_state_tomo_analysis
# timestamp = '20220723_171601'
# plot_mod.get_default_plot_params()
# psi_target = qt.qip.operations.gates.rx(np.pi)*qt.states.basis(2, 0)
# rho_target = psi_target*psi_target.dag()
# pp = do_state_tomo_analysis(timestamp=timestamp, rho_target=rho_target,
#                             save_figures=True, save_processed_data=False)
# os.startfile(a_tools.get_folder(timestamp))

# TODO 2qb State Tomography
# timestamp = '20220722_124116'
# plot_mod.get_default_plot_params()
# import qutip as qtp
# from pycqed.analysis_v2 import tomography_qudev as tomo_ana
# U1s = {pulse: tomo_ana.standard_qubit_pulses_to_rotations([(pulse,)])[0]
#        for pulse in
#        ['X90', 'Y90', 'mX90', 'mY90', 'I', 'X180', 'Y180', 'mY180',
#         'mX180']}
# UCZ = qtp.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]])
# rho_target = (qtp.tensor([U1s['I'], U1s['mY90']]) * UCZ * qtp.tensor(
#                      [U1s['mY90'], U1s['Y90']]) *
#                   qtp.tensor(qtp.basis(2), qtp.basis(2))).full()
# pp = do_state_tomo_analysis(timestamp=timestamp, rho_target=rho_target,
#                             save_figures=True, save_processed_data=False)
# os.startfile(a_tools.get_folder(timestamp))

# TODO Process Tomography
# from pycqedscripts.scripts.characterization.process_tomo import\
#     do_process_tomo_analysis
# # Produces very many plots!
# timestamp = '20220722_124212'
# os.startfile(a_tools.get_folder(timestamp))
# plot_mod.get_default_plot_params()
# pp = do_process_tomo_analysis(
#     timestamp=timestamp,
#     meas_obj_names=['qb10', 'qb11'],
#     # save_figures_state_tomo=False,  # reduces nr of plots that are saved
#     process_name='CZ',
#     save_processed_data=False,
# )