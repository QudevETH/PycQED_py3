import numpy as np
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as sqs2
import time
from pycqed.analysis_v2 import tomography_qudev as tomo


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

    def __init__(self, pulse_pars_list, RO_pars_list, upload=True,
                 preselection=False, parallel_pulses=False, RO_spacing=200e-9):
        super().__init__()
        self.pulse_pars_list = pulse_pars_list
        self.RO_pars_list = RO_pars_list
        self.upload = upload
        self.parameter_name = 'sample'
        self.unit = '#'
        samples = 2**len(pulse_pars_list)
        if preselection:
            samples *= 2
        self.preselection = preselection
        self.parallel_pulses = parallel_pulses
        self.RO_spacing = RO_spacing
        self.name = '{}_qubit_off_on'.format(len(pulse_pars_list))

    def prepare(self, **kw):
        if self.upload:
            sqs2.n_qubit_off_on(pulse_pars_list=self.pulse_pars_list,
                                RO_pars_list=self.RO_pars_list,
                                preselection=self.preselection,
                                parallel_pulses=self.parallel_pulses,
                                RO_spacing=self.RO_spacing)


class Ramsey_add_pulse_swf(swf.Hard_Sweep):

    def __init__(self, measured_qubit_name,
                 pulsed_qubit_name, operation_dict,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.measured_qubit_name = measured_qubit_name
        self.pulsed_qubit_name = pulsed_qubit_name
        self.operation_dict = operation_dict
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning

        self.name = 'Ramsey Add Pulse'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.Ramsey_add_pulse_seq(
                times=self.sweep_points,
                measured_qubit_name=self.measured_qubit_name,
                pulsed_qubit_name=self.pulsed_qubit_name,
                operation_dict=self.operation_dict,
                artificial_detuning=self.artificial_detuning,
                cal_points=self.cal_points)


class Measurement_Induced_Dephasing_Phase_Swf(swf.Hard_Sweep):
    def __init__(self, qbn_dephased, ro_op, operation_dict, readout_separation,
                 nr_readouts, cal_points=((-4,-3), (-2,-1)), upload=True):
        super().__init__()
        self.qbn_dephased = qbn_dephased
        self.ro_op = ro_op
        self.operation_dict = operation_dict
        self.readout_separation = readout_separation
        self.nr_readouts = nr_readouts
        self.cal_points = cal_points
        self.upload = upload

        self.name = 'Measurement induced dephasing phase'
        self.parameter_name = 'theta'
        self.unit = 'rad'

    def prepare(self, **kw):
        if self.upload:
            sqs2.measurement_induced_dephasing_seq(
                phases=self.sweep_points, 
                qbn_dephased=self.qbn_dephased, 
                ro_op=self.ro_op, 
                operation_dict=self.operation_dict, 
                readout_separation=self.readout_separation, 
                nr_readouts=self.nr_readouts, 
                cal_points=self.cal_points)


class Measurement_Induced_Dephasing_Amplitude_Swf(swf.Soft_Sweep):
    class DummyQubit:
        _params = ['f_RO', 'f_RO_mod', 'RO_pulse_length', 'RO_amp',
                   'ro_pulse_shape', 'ro_pulse_filter_sigma', 
                   'ro_pulse_nr_sigma', 'ro_CLEAR_delta_amp_segment', 
                   'ro_CLEAR_segment_length']
        
        def __init__(self, qb):
            self._values = {}
            self.name = qb.name
            self.UHFQC = qb.UHFQC
            self.readout_DC_LO = qb.readout_DC_LO
            self.readout_UC_LO = qb.readout_UC_LO
            for param in self._params:
                self.make_param(param, qb.get(param))

        def make_param(self, name, val):
            self._values[name] = val
            def accessor(v=None):
                if v is None:
                    return self._values[name]
                else:
                    self._values[name] = v
            setattr(self, name, accessor)
    
    def __init__(self, qb_dephased, qb_targeted, nr_readouts, 
                 multiplexed_pulse_fn, f_LO):
        super().__init__()
        self.qb_dephased = qb_dephased
        self.qb_targeted = qb_targeted
        self.nr_readouts = nr_readouts
        self.multiplexed_pulse_fn = multiplexed_pulse_fn
        self.f_LO = f_LO

        self.name = 'Measurement induced dephasing amplitude'
        self.parameter_name = 'amp'
        self.unit = 'max'

    def prepare(self, **kw):
        pass

    def set_parameter(self, val):
        qb_targeted_dummy = self.DummyQubit(self.qb_targeted)
        qb_targeted_dummy.RO_amp(val)
        readouts = [(qb_targeted_dummy,)]*self.nr_readouts + \
                   [(self.qb_dephased,)]
        time.sleep(0.1)
        self.multiplexed_pulse_fn(readouts, self.f_LO, upload=True)
        time.sleep(0.1)
