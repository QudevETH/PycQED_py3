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

def test_RabiAnalysis():
    for t_start in [
        '20210716_171937',  # ge
        '20210716_165406',  # ef
        '20210716_002652',  # ef - multi-qubit
        '20241120_155114',  # avg
        '20241120_163229',  # avg with reset
        '20241120_155158',  # SSRO
        # '20241120_163256',  # SSRO with reset
    ]:
        tda.RabiAnalysis(
            t_start=t_start,
            do_fitting=True,
            **add_default_kw(),
        )

def test_ChevronAnalysis():
    for t_start, kw in [
        # avg
        ('20241108_012425',
            {
                'do_fitting': True,
            }
        ),
        ('20241108_012425',
            {
                'options_dict': {
                    'slice_idxs_1d_proj_plot': {
                        'qb14': [('8:10', 'row'), (0, 'col')]},
                    'slice_idxs_1d_raw_plot': {
                        'qb14': [('8:10', 'row'), (0, 'col')]},
                },
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

def test_CPhaseLeakageAnalysis():
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

def test_QScaleAnalysis():
    for t_start, kw in [
        ('20210716_161150', {}),
        ('20241115_002806', {}),
    ]:
        add_default_kw(kw)
        tda.QScaleAnalysis(
            t_start=t_start,
            **kw,
        )

def test_T1Analysis():
    for t_start, kw in [
        ('20210716_002204', {}),  # 1 qb
        ('20210716_235757', {}),  # n qbs
    ]:
        add_default_kw(kw)
        tda.T1Analysis(
            t_start=t_start,
            **kw,
        )

def test_RamseyAnalysis():
    for t_start, kw in [
        ('20210716_160058', {}),  # Ramsey ge - 1qb
        ('20210716_174638', {}),  # Ramsey ef - 1qb
        ('20210716_235343', {}),  # Ramsey ge - nqb
        ('20211221_114300', {}),  # Ramsey ge with different artificial dets
    ]:
        add_default_kw(kw)
        tda.RamseyAnalysis(
            t_start=t_start,
            **kw,
        )

def test_EchoAnalysis():
    for t_start, kw in [
        ('20220520_191131', {}),
    ]:
        add_default_kw(kw)
        tda.EchoAnalysis(
            t_start=t_start,
            **kw,
        )

def test_DynamicPhaseAnalysis():
    for t_start, kw in [
        ('20210721_090259', {}),  # cphase check
        ('20210721_062113', {}),  # cphase sweep
        ('20210721_083238', {}),  # dyn phase
    ]:
        add_default_kw(kw)
        tda.DynamicPhaseAnalysis(
            t_start=t_start,
            **kw,
        )

def test_MultiQubitTDA():
    for t_start, kw in [
        ('20210809_123011', {}),  # chevron
    ]:
        add_default_kw(kw)
        tda.MultiQubit_TimeDomain_Analysis(
            t_start=t_start,
            **kw,
        )

def test_FluxPulseScopeAnalysis():
    for t_start, kw in [
        ('20210317_210331', {}),  # non-tracked
        ('20211210_222258', {}),  # tracked
    ]:
        add_default_kw(kw)
        tda.FluxPulseScopeAnalysis(
            t_start=t_start,
            **kw,
        )

def test_CryoscopeAnalysis():
    for t_start, kw in [
        ('20211219_081948', {}),
    ]:
        add_default_kw(kw)
        tda.CryoscopeAnalysis(
            t_start=t_start,
            **kw,
        )

def test_RabiFrequencySweepAnalysis():
    for t_start, kw in [
        ('20211202_175810', {}),
    ]:
        add_default_kw(kw)
        tda.RabiFrequencySweepAnalysis(
            t_start=t_start,
            **kw,
        )

def test_ReparkingRamseyAnalysis():
    for t_start, kw in [
        ('20220118_134248', {}),
    ]:
        add_default_kw(kw)
        tda.ReparkingRamseyAnalysis(
            t_start=t_start,
            **kw,
        )

def test_T1FrequencySweepAnalysis():
    for t_start, kw in [
        # ('20210825_225159', {}),  # TODO r'Q:\Archive\Qudev86 - QComp PycQED data from BF1\pydata'
    ]:
        add_default_kw(kw)
        tda.T1FrequencySweepAnalysis(
            t_start=t_start,
            **kw,
        )

def test_MixerCarrierAnalysis():
    for t_start, kw in [  # TODO
        # (
        #     '20211116_113037',
        #     {
        #         'options_dict': {
        #             'qb_names': ['qb4'],
        #             'rotate': False,
        #         },
        #     }
        # ),  # LO with random grid
        # (
        #     '20211127_152630',
        #     {
        #         'options_dict': {
        #             'qb_names': ['qb4'],
        #             'rotate': False,
        #         },
        #     }
        # ),  # LO with rectangular grid
    ]:
        add_default_kw(kw)
        tda.MixerCarrierAnalysis(
            t_start=t_start,
            **kw,
        )

def test_MixerSkewnessAnalysis():
    for t_start, kw in [
        ('20220317_182154', {}),
    ]:
        add_default_kw(kw)
        tda.MixerSkewnessAnalysis(
            t_start=t_start,
            **kw,
        )

def test_FluxPulseTimingAnalysis():
    for t_start, kw in [
        (
            '20211216_010723',
            {
                'qb_names': ['qb2'],
            }
        ),
        # ('', {}),
    ]:
        add_default_kw(kw)
        tda.FluxPulseTimingAnalysis(
            t_start=t_start,
            **kw,
        )

def test_FluxPulseTimingBetweenQubitsAnalysis():
    for t_start, kw in [
        (
            '20211216_011402',
            {
                'options_dict': {
                    'data_to_fit': {'qb2': 'pe', 'qb4': 'pe'}  # FIXME remove?
                },
            }
        ),
        # ('', {}),
    ]:
        add_default_kw(kw)
        tda.FluxPulseTimingBetweenQubitsAnalysis(
            t_start=t_start,
            **kw,
        )

def test_FluxlineCrosstalkAnalysis():
    for t_start, kw in [
        ('20220202_024818', {}),  # multi qb
        ('20220202_024222', {}),  # single qb
    ]:
        add_default_kw(kw)
        # FIXME v2 shouldn't rely on v3. Imported only locally here
        from pycqed.analysis_v3 import helper_functions as hlp_mod
        crosstalk_qubits_names = hlp_mod.get_param_from_metadata_group(
            t_start, 'crosstalk_qubits_names')
        tda.FluxlineCrosstalkAnalysis(
            qb_names=crosstalk_qubits_names,
            t_start=t_start,
            **kw,
        )

def test_MeasurementInducedDephasingAnalysis():
    for t_start, kw in [
        ('20221013_140524', {}),
    ]:
        add_default_kw(kw)
        tda.MeasurementInducedDephasingAnalysis(
            t_start=t_start,
            **kw,
        )

def test_FluxAmplitudeSweepAnalysis():
    for t_start, kw in [
        ('20230828_170441', {}),
    ]:
        add_default_kw(kw)
        # FIXME v2 shouldn't rely on v3. Imported only locally here
        from pycqed.analysis_v3 import helper_functions as hlp_mod
        ro_qubits = hlp_mod.get_param_from_metadata_group(t_start, 'ro_qubits')
        tda.FluxAmplitudeSweepAnalysis(
            qb_names=ro_qubits,
            t_start=t_start,
            **kw,
        )

def test_NPulseAmplitudeCalibAnalysis():
    for t_start, kw in [
        ('20220521_113929', {}),
    ]:
        add_default_kw(kw)
        tda.NPulseAmplitudeCalibAnalysis(
            t_start=t_start,
            **kw,
        )

def test_DriveAmplitudeNonlinearityCurveAnalysis():
    for ts, kw in [
        (('20220515_164605', '20220515_175316'), {}),
    ]:
        add_default_kw(kw)
        tda.DriveAmplitudeNonlinearityCurveAnalysis(
            t_start=ts[0],
            t_stop=ts[1],
            **kw,
        )

# TODO State Tomography Analysis
# ts = '20200422_215530'
# gate = ('qb2', 'qb4')
# import pycqed.utilities.qutip_compat as qt
# qtp = qt
# U1s = {pulse: tomo_ana.standard_qubit_pulses_to_rotations([(pulse,)])[0] for
#        pulse in
#        ['X90', 'Y90', 'mX90', 'mY90', 'I', 'X180', 'Y180', 'mY180', 'mX180']}
# UCZ = qtp.Qobj(np.diag([1, 1, 1, -1]), dims=[[2, 2], [2, 2]])
# cz_pulse_name = 'CZ'
# state_dict = {
#     'phi+': ([f'mY90 {gate[0]}', f'Y90s {gate[1]}',
#               f'{cz_pulse_name} {gate[0]} {gate[1]}',
#               f'mY90 {gate[1]}'],
#              (qtp.tensor([U1s['I'], U1s['mY90']]) * UCZ * qtp.tensor(
#                  [U1s['mY90'], U1s['Y90']]) *
#               qtp.tensor(qtp.basis(2), qtp.basis(2))).full()),
#     'mY90Y90': ([f'mY90 {gate[0]}', f'Y90s {gate[1]}'],
#                 (qtp.tensor([U1s['mY90'], U1s['Y90']]) * qtp.tensor(
#                     qtp.basis(2), qtp.basis(2))).full()),
#     'CZ': ([f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#            (UCZ * qtp.tensor(qtp.basis(2), qtp.basis(2))).full()),
#     'IX90CZ': ([f'X90 {gate[1]}', f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#                (UCZ * qtp.tensor([U1s['I'], U1s['X90']]) * qtp.tensor(
#                    qtp.basis(2), qtp.basis(2))).full()),
#     'X90ICZ': ([f'X90 {gate[0]}', f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#                (UCZ * qtp.tensor([U1s['X90'], U1s['I']]) * qtp.tensor(
#                    qtp.basis(2), qtp.basis(2))).full()),
#     'IX180CZ': ([f'X180 {gate[1]}', f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#                 (UCZ * qtp.tensor([U1s['I'], U1s['X180']]) * qtp.tensor(
#                     qtp.basis(2), qtp.basis(2))).full()),
#     'X180ICZ': ([f'X180 {gate[0]}', f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#                 (UCZ * qtp.tensor([U1s['X180'], U1s['I']]) * qtp.tensor(
#                     qtp.basis(2), qtp.basis(2))).full()),
#     'X180X180CZ': ([f'X180 {gate[1]}', f'X180s {gate[0]}',
#                     f'{cz_pulse_name} {gate[0]} {gate[1]}'],
#                    (UCZ * qtp.tensor([U1s['X180'], U1s['X180']]) * qtp.tensor(
#                        qtp.basis(2), qtp.basis(2))).full()),
# }
# reload(ba)
# reload(tda)
# state_name = 'X180X180CZ'
# _, rho_target = state_dict[state_name]
#
# preselection = hlp_mod.get_param_from_metadata_group(ts,
#                                                      'use_preselection')  # True
# use_cal_points = True
# rots_basis = hlp_mod.get_param_from_metadata_group(ts, 'rots_basis')
# channel_map = hlp_mod.get_param_from_metadata_group(ts, 'channel_map')
# if hasattr(hlp_mod, 'get_instr_param_from_file'):
#     get_instr_settings_func = hlp_mod.get_instr_param_from_file
# else:
#     get_instr_settings_func = hlp_mod.get_instr_param_from_hdf_file
# try:
#     thresholds = {qbn: get_instr_settings_func(qbn, 'acq_classifier_params',
#                                                ts)['thresholds'][0]
#                   for qbn in gate}
# except KeyError:
#     #     thresholds = hlp_mod.get_qb_thresholds_from_hdf_file(gate, ts)
#     thresholds = {}
#     for qbn in gate:
#         acq_I_channel = get_instr_settings_func(qbn, 'acq_I_channel', ts)
#         instr_uhf = get_instr_settings_func(qbn, 'instr_uhf', ts)
#         thresholds[qbn] = get_instr_settings_func(instr_uhf,
#                                                   f'qas_0_thresholds_{acq_I_channel}_level',
#                                                   ts)
# n_segments = len(rots_basis) ** 2 + (2 ** 2 if use_cal_points else 0)
# # n_segments = hlp_mod.get_param_from_metadata_group(ts, 'n_segments')
# MA = tda.StateTomographyAnalysis(t_start=ts, options_dict=dict(
#     n_readouts=(2 if preselection else 1) * n_segments,
#     thresholds=OrderedDict([(qb, thresholds[qb]) for qb in gate]),
#     channel_map=OrderedDict([(qb, channel_map[qb]) for qb in gate]),
#     cal_points=[
#         OrderedDict([(channel_map[qb], [2 * i + 1 if preselection else i])
#                      for qb in gate]) for i in np.arange(-4, 0)],
#     #         meas_operators=[np.diag(1*(np.arange(4) == idx)) for idx in range(4)],
#     data_type='singleshot',
#     rho_target=qtp.Qobj(rho_target),
#     basis_rots_str=rots_basis,
#     covar_matrix=np.diag(np.ones(4)),
#     mle=True,
#     concurrence=False,
#     use_preselection=preselection,
#     data_filter=(
#         lambda data: data[1:2 * len(rots_basis) ** 2 + 1:2]) if preselection \
#         else (lambda data: data[:len(rots_basis) ** 2])
# ))
# os.startfile(a_tools.get_folder(ts))