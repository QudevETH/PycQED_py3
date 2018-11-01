import numpy as np
import logging
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as sqs2

class two_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'two_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                  q1_pulse_pars=self.q1_pulse_pars,
                                  RO_pars=self.RO_pars,
                                  return_seq=self.return_seq,
                                  verbose=self.verbose)


class three_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'three_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.three_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                    q1_pulse_pars=self.q1_pulse_pars,
                                    q2_pulse_pars=self.q2_pulse_pars,
                                    RO_pars=self.RO_pars,
                                    return_seq=self.return_seq,
                                    verbose=self.verbose)


class four_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars,  q3_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.q3_pulse_pars = q3_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'four_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.four_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                   q1_pulse_pars=self.q1_pulse_pars,
                                   q2_pulse_pars=self.q2_pulse_pars,
                                   q3_pulse_pars=self.q3_pulse_pars,
                                   RO_pars=self.RO_pars,
                                   return_seq=self.return_seq,
                                   verbose=self.verbose)


class five_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, q2_pulse_pars,
                 q3_pulse_pars, q4_pulse_pars, RO_pars, upload=True,
                 return_seq=False, nr_samples=4, verbose=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.q2_pulse_pars = q2_pulse_pars
        self.q3_pulse_pars = q3_pulse_pars
        self.q4_pulse_pars = q4_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(nr_samples)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'five_qubit_off_on'

    def prepare(self, **kw):
        if self.upload:
            sqs2.four_qubit_off_on(q0_pulse_pars=self.q0_pulse_pars,
                                   q1_pulse_pars=self.q1_pulse_pars,
                                   q2_pulse_pars=self.q2_pulse_pars,
                                   q3_pulse_pars=self.q3_pulse_pars,
                                   q4_pulse_pars=self.q4_pulse_pars,
                                   RO_pars=self.RO_pars,
                                   return_seq=self.return_seq,
                                   verbose=self.verbose)


class n_qubit_seq_sweep(swf.Hard_Sweep):
    """
    Allows an arbitrary sequence.
    """

    def __init__(self, seq_len, #upload=True,
                 verbose=False, sweep_name=""):
        super().__init__()
        self.parameter_name = 'segment'
        self.unit = '#'
        self.sweep_points = np.arange(seq_len)
        self.verbose = verbose
        self.name = sweep_name

    def prepare(self, **kw):
        pass


class n_qubit_off_on(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars, upload=True,
                 preselection=False, parallel_pulses=False,
                 verbose=False, RO_spacing=200e-9):
        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        samples = 2**len(pulse_pars_list)
        if preselection:
            samples *= 2
        # self.sweep_points = np.arange(samples)
        self.verbose = verbose
        self.preselection = preselection
        self.parallel_pulses = parallel_pulses
        self.RO_spacing = RO_spacing
        self.name = '{}_qubit_off_on'.format(len(pulse_pars_list))

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_off_on(pulse_pars_list=self.pulse_pars_list,
                                RO_pars=self.RO_pars,
                                preselection=self.preselection,
                                parallel_pulses=self.parallel_pulses,
                                RO_spacing=self.RO_spacing,
                                verbose=self.verbose)

class n_qubit_reset(swf.Hard_Sweep):
    def __init__(self, qubit_names, operation_dict, reset_cycle_time,
                 nr_resets=1, upload=True, verbose=False,
                 codeword_indices=None):
        super().__init__()
        self.qubit_names = qubit_names
        self.operation_dict = operation_dict
        self.reset_cycle_time = reset_cycle_time
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.nr_resets = nr_resets
        samples = nr_resets*2**len(qubit_names)
        self.sweep_points = np.arange(samples)
        self.verbose = verbose
        self.name = '{}_reset_x{}'.format(','.join(qubit_names), nr_resets)

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_reset(qubit_names=self.qubit_names,
                               operation_dict=self.operation_dict,
                               reset_cycle_time=self.reset_cycle_time,
                               nr_resets=self.nr_resets,
                               verbose=self.verbose)


class n_qubit_Simultaneous_RB_sequence_lengths(swf.Soft_Sweep):
    def __init__(self, sweep_control='soft',
                 n_qubit_RB_sweepfunction=None):
        super().__init__()

        self.sweep_control = sweep_control
        # self.sweep_points = nr_cliffords
        self.n_qubit_RB_sweepfunction = n_qubit_RB_sweepfunction
        self.name = 'two_qubit_Simultaneous_RB_sequence_lengths'
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'


    def set_parameter(self, val):
        self.n_qubit_RB_sweepfunction.nr_cliffords_value = val
        self.n_qubit_RB_sweepfunction.upload = True
        self.n_qubit_RB_sweepfunction.prepare()


class n_qubit_Simultaneous_RB_fixed_length(swf.Hard_Sweep):

    def __init__(self, qubit_names_list, operation_dict,
                 nr_seeds_array, #array
                 nr_cliffords_value, #int
                 gate_decomposition='HZ', interleaved_gate=None,
                 upload=True, return_seq=False, seq_name=None,
                 verbose=False, cal_points=False):

        super().__init__()
        self.qubit_names_list = qubit_names_list
        self.operation_dict = operation_dict
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.nr_seeds_array = nr_seeds_array
        self.seq_name = seq_name
        self.return_seq = return_seq
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate
        self.verbose = verbose
        self.cal_points = cal_points

        self.parameter_name = 'Nr of Seeds'
        self.unit = '#'
        self.name = 'two_qubit_Simultaneous_RB_fixed_length'

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_simultaneous_randomized_benchmarking_seq(
                self.qubit_names_list, self.operation_dict,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=self.nr_seeds_array,
                seq_name=self.seq_name,
                return_seq=self.return_seq,
                verbose=self.verbose,
                upload=self.upload,
                cal_points=self.cal_points)


class n_qubit_Simultaneous_RB_fixed_seeds(swf.Hard_Sweep):

    def __init__(self, pulse_pars_list, RO_pars, nr_cliffords_value,
                 gate_decomposition='HZ', interleaved_gate=None,
                 upload=True, return_seq=False, seq_name=None,
                 CxC_RB=True, idx_for_RB=0,
                 verbose=False):

        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars = RO_pars
        self.upload = upload
        self.nr_cliffords_value = nr_cliffords_value
        self.CxC_RB = CxC_RB
        self.seq_name = seq_name
        self.return_seq = return_seq
        self.gate_decomposition = gate_decomposition
        self.interleaved_gate = interleaved_gate
        self.idx_for_RB = idx_for_RB
        self.verbose = verbose

        self.parameter_name = 'samples'
        self.unit = '#'
        self.name = 'n_qubit_Simultaneous_RB_fixed_seed'

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_simultaneous_randomized_benchmarking_seq(
                self.pulse_pars_list, self.RO_pars,
                nr_cliffords_value=self.nr_cliffords_value,
                gate_decomposition=self.gate_decomposition,
                interleaved_gate=self.interleaved_gate,
                nr_seeds=np.array([1]),
                CxC_RB=self.CxC_RB,
                idx_for_RB=self.idx_for_RB,
                seq_name=self.seq_name,
                return_seq=self.return_seq,
                verbose=self.verbose)


class two_qubit_randomized_benchmarking_nr_cliffords(swf.Soft_Sweep):

    def __init__(self, sweep_control='soft',
                 two_qubit_RB_sweepfunction=None):
        super().__init__()

        self.sweep_control = sweep_control
        self.two_qubit_RB_sweepfunction = two_qubit_RB_sweepfunction
        self.name = 'Two_Qubit_Randomized_Benchmarking_nr_cliffords'
        self.parameter_name = 'Nr of Cliffords'
        self.unit = '#'


    def set_parameter(self, val):
        self.two_qubit_RB_sweepfunction.nr_cliffords_value = val
        self.two_qubit_RB_sweepfunction.upload = True
        self.two_qubit_RB_sweepfunction.prepare()


class two_qubit_randomized_benchmarking_one_length(swf.Hard_Sweep):

    def __init__(self, qb1n, qb2n, operation_dict,
                 nr_cliffords_value,
                 CZ_pulse_name=None,
                 net_clifford=0,
                 clifford_decomposition_name='HZ',
                 interleaved_gate=None,
                 seq_name=None, upload=True,
                 return_seq=False, verbose=False):

        super().__init__()
        self.qb1n = qb1n
        self.qb2n = qb2n
        self.operation_dict = operation_dict
        self.nr_cliffords_value = nr_cliffords_value
        self.CZ_pulse_name = CZ_pulse_name
        self.net_clifford = net_clifford
        self.clifford_decomposition_name = clifford_decomposition_name
        self.interleaved_gate = interleaved_gate
        self.seq_name = seq_name
        self.upload = upload
        self.return_seq = return_seq
        self.verbose = verbose

        self.parameter_name = 'Nr of Seeds'
        self.unit = '#'
        self.name = 'Two_Qubit_Randomized_Benchmarking_one_length'

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_randomized_benchmarking_seq(
                qb1n=self.qb1n, qb2n=self.qb2n,
                operation_dict=self.operation_dict,
                nr_cliffords_value=self.nr_cliffords_value,
                nr_seeds=self.sweep_points,
                CZ_pulse_name=self.CZ_pulse_name,
                net_clifford=self.net_clifford,
                clifford_decomposition_name=self.clifford_decomposition_name,
                interleaved_gate=self.interleaved_gate,
                seq_name=self.seq_name, upload=self.upload,
                return_seq=self.return_seq, verbose=self.verbose)


class two_qubit_AllXY(swf.Hard_Sweep):

    def __init__(self, q0_pulse_pars, q1_pulse_pars, RO_pars, upload=True,
                 return_seq=False, verbose=False, X_mid=False, simultaneous=False):
        super().__init__()
        self.q0_pulse_pars = q0_pulse_pars
        self.q1_pulse_pars = q1_pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.sweep_points = np.arange(42*2)
        self.verbose = verbose
        self.return_seq = return_seq
        self.name = 'two_qubit_AllXY'
        self.X_mid = X_mid
        self.simultaneous = simultaneous

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_AllXY(q0_pulse_pars=self.q0_pulse_pars,
                                  q1_pulse_pars=self.q1_pulse_pars,
                                  RO_pars=self.RO_pars,
                                  double_points=True,
                                  return_seq=self.return_seq,
                                  verbose=self.verbose,
                                  X_mid=self.X_mid,
                                  simultaneous=self.simultaneous)


class tomo_Bell(swf.Hard_Sweep):

    def __init__(self, bell_state, qb_c, qb_t, RO_pars,
                 num_flux_pulses=0, basis_pulses=None,
                 cal_state_repeats=7,
                 CZ_disabled=False, spacing=100e9,
                 verbose=False, upload=True, return_seq=False):
        super().__init__()
        self.bell_state = bell_state
        self.qb_c = qb_c
        self.qb_t = qb_t
        self.RO_pars = RO_pars
        self.num_flux_pulses = num_flux_pulses
        self.cal_state_repeats = cal_state_repeats
        self.basis_pulses = basis_pulses
        self.upload = upload
        self.CZ_disabled = CZ_disabled
        self.parameter_name = 'sample'
        self.unit = '#'
        self.verbose = verbose
        self.spacing = spacing
        self.return_seq = return_seq
        self.name = 'tomo_Bell'

    def prepare(self, **kw):
        if self.upload:
            sqs2.two_qubit_tomo_bell_qudev_seq(
                bell_state=self.bell_state,
                qb_c=self.qb_c, qb_t=self.qb_t,
                RO_pars=self.RO_pars,
                basis_pulses=self.basis_pulses,
                num_flux_pulses=self.num_flux_pulses,
                cal_state_repeats=self.cal_state_repeats,
                CZ_disabled=self.CZ_disabled,
                verbose=self.verbose,
                upload=self.upload,
                spacing=self.spacing)


class three_qubit_GHZ_tomo(swf.Hard_Sweep):

    def __init__(self, qubits, RO_pars,
                 CZ_qubit_dict,
                 basis_pulses=None,
                 cal_state_repeats=2,
                 spacing=100e-9,
                 verbose=False, upload=True, return_seq=False):
        super().__init__()
        self.qubits = qubits
        self.RO_pars = RO_pars
        self.CZ_qubit_dict = CZ_qubit_dict
        self.basis_pulses = basis_pulses
        self.cal_state_repeats = cal_state_repeats
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.verbose = verbose
        self.spacing = spacing
        self.return_seq = return_seq
        self.name = 'tomo_3_qubit_GHZ'

    def prepare(self, **kw):
        if self.upload:
            sqs2.three_qubit_GHZ_tomo_seq(
                qubits=self.qubits,
                RO_pars=self.RO_pars,
                CZ_qubit_dict=self.CZ_qubit_dict,
                basis_pulses=self.basis_pulses,
                cal_state_repeats=self.cal_state_repeats,
                spacing=self.spacing,
                verbose=self.verbose,
                upload=self.upload)


class parity_correction(swf.Hard_Sweep):
    def __init__(self, q0n, q1n, q2n, operation_dict, feedback_delay,
                 prep_sequence=None, nr_echo_pulses=4, cpmg_scheme=True,
                 tomography_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
                 reset=True, upload=True, verbose=False, preselection=False,
                 ro_spacing=1e-6):
        super().__init__()
        self.q0n = q0n
        self.q1n = q1n
        self.q2n = q2n
        self.operation_dict = operation_dict
        self.feedback_delay = feedback_delay
        self.tomography_basis = tomography_basis
        self.prep_sequence = prep_sequence
        self.nr_echo_pulses = nr_echo_pulses
        self.cpmg_scheme = cpmg_scheme
        self.preselection = preselection
        self.ro_spacing = ro_spacing
        self.reset = reset
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        self.verbose = verbose
        self.name = 'two_qubit_parity'

    def prepare(self, **kw):
        if self.upload:
            sqs2.parity_correction_seq(
                self.q0n, self.q1n, self.q2n,
                self.operation_dict,
                feedback_delay=self.feedback_delay,
                prep_sequence=self.prep_sequence,
                tomography_basis=self.tomography_basis,
                reset=self.reset,
                verbose=self.verbose,
                preselection=self.preselection,
                ro_spacing=self.ro_spacing,
                nr_echo_pulses=self.nr_echo_pulses,
                cpmg_scheme=self.cpmg_scheme
                )


class calibrate_n_qubits(swf.Hard_Sweep):
    def __init__(self, sweep_params, sweep_points,
                 qubit_names, operation_dict,
                 cal_points=True, no_cal_points=4,
                 nr_echo_pulses=0, idx_DD_start=-1, UDD_scheme=True,
                 parameter_name='sample', unit='#',
                 upload=False, return_seq=False):

        super().__init__()
        self.name = 'n_Qubits_Time_Domain'
        self.sweep_params = sweep_params
        self.sweep_pts = sweep_points
        self.qubit_names = qubit_names
        self.operation_dict = operation_dict
        self.cal_points = cal_points
        self.no_cal_points = no_cal_points
        self.nr_echo_pulses = nr_echo_pulses
        self.idx_DD_start = idx_DD_start
        self.UDD_scheme = UDD_scheme
        self.upload = upload
        self.return_seq = return_seq
        self.parameter_name = parameter_name
        self.unit = unit

    def prepare(self, **kw):
        if self.upload:
            sqs2.general_multi_qubit_seq(sweep_params=self.sweep_params,
                                         sweep_points=self.sweep_pts,
                                         qb_names=self.qubit_names,
                                         operation_dict=self.operation_dict,
                                         cal_points=self.cal_points,
                                         no_cal_points=self.no_cal_points,
                                         nr_echo_pulses=self.nr_echo_pulses,
                                         idx_DD_start=self.idx_DD_start,
                                         UDD_scheme=self.UDD_scheme,
                                         upload=self.upload,
                                         return_seq=self.return_seq)


class calibrate_n_qubits_change_sweep_points(swf.Soft_Sweep):

    def __init__(self, hard_sweep, sweep_control='soft', upload=True):


        self.sweep_control = sweep_control
        self.hard_sweep = hard_sweep
        self.upload = upload
        self.name = 'n_Qubit_Time_Domain_Change_Sweep_Points'
        self.parameter_name = self.hard_sweep.parameter_name
        self.unit = self.hard_sweep.unit

    def set_parameter(self, val):
        self.hard_sweep.sweep_pts = val
        self.hard_sweep.upload = True
        self.hard_sweep.prepare()


