import numpy as np
import pycqed.measurement.sweep_functions as swf
import pycqed.measurement.pulse_sequences.multi_qubit_tek_seq_elts as sqs2


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
                 preselection=False, parallel_pulses=False, RO_spacing=2000e-9):
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
