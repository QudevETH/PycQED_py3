import numpy as np
import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sqs
from pycqed.measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sqs2
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import calibration_elements as csqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mq_sqs
from pycqed.measurement.waveform_control import pulsar as ps

import time

log = logging.getLogger(__name__)

class File(swf.Hard_Sweep):

    def __init__(self, filename, AWG, title=None, NoElements=None, upload=True):
        self.upload = upload
        self.AWG = AWG
        if title:
            self.name = title
        else:
            self.name = filename
        self.filename = filename + '_FILE'
        self.upload = upload
        self.parameter_name = 'amplitude'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            self.AWG.set_setup_filename(self.filename)


class awg_seq_swf(swf.Hard_Sweep):
    def __init__(self, awg_seq_func, awg_seq_func_kwargs,
                 parameter_name=None, unit='a.u.',
                 AWG=None,
                 fluxing_channels=[],
                 upload=True, return_seq=False):
        """
        A wrapper for awg sequence generating functions.
        Works as a general awg sweep function.
        """
        super().__init__()
        self.upload = upload
        self.awg_seq_func = awg_seq_func
        self.awg_seq_func_kwargs = awg_seq_func_kwargs
        self.unit = unit
        self.name = 'swf_' + awg_seq_func.__name__
        self.fluxing_channels = fluxing_channels
        self.AWG = AWG

        if parameter_name is not None:
            self.parameter_name = parameter_name
        else:
            self.parameter_name = 'points'

    def prepare(self, **kw):
        if self.parameter_name != 'points':
            self.awg_seq_func_kwargs[self.parameter_name] = self.sweep_points

        if self.upload:
            old_vals = np.zeros(len(self.fluxing_channels))
            for i, ch in enumerate(self.fluxing_channels):
                old_vals[i] = self.AWG.get('{}_amp'.format(ch))
                self.AWG.set('{}_amp'.format(ch), 2)

            self.awg_seq_func(**self.awg_seq_func_kwargs)

            for i, ch in enumerate(self.fluxing_channels):
                self.AWG.set('{}_amp'.format(ch), old_vals[i])

    def set_parameter(self, val, **kw):
        # exists for compatibility reasons with 2D sweeps
        pass


class mixer_skewness_calibration_swf(swf.Hard_Sweep):
    """
    Based on the "UHFQC_integrated_average" detector.
    generates an AWG seq to measure sideband transmission.
    """

    def __init__(self, pulseIch, pulseQch, alpha, phi_skew,
                 f_mod, RO_trigger_channel, RO_pars,
                 amplitude=0.1, RO_trigger_separation=5e-6,
                 verbose=False,  data_points=1, upload=True):
        super().__init__()
        self.name = 'mixer_skewness_calibration_swf'
        self.parameter_name = 'alpha'
        self.unit = 'a.u'
        self.pulseIch = pulseIch
        self.pulseQch = pulseQch
        self.alpha = alpha
        self.phi_skew = phi_skew
        self.f_mod = f_mod
        self.amplitude = amplitude
        self.verbose = verbose
        self.RO_trigger_separation = RO_trigger_separation
        self.RO_trigger_channel = RO_trigger_channel
        self.RO_pars = RO_pars
        self.data_points = data_points
        self.n_measured = 0
        self.seq = None
        self.elts = []
        self.finished = False
        self.upload = upload

    def prepare(self):
        if self.upload:
            sqs.mixer_skewness_cal_sqs(pulseIch=self.pulseIch,
                              pulseQch=self.pulseQch,
                              alpha=self.alpha,
                              phi_skew=self.phi_skew,
                              f_mod=self.f_mod,
                              RO_trigger_channel=self.RO_trigger_channel,
                              RO_pars=self.RO_pars,
                              amplitude=self.amplitude,
                              RO_trigger_separation=self.RO_trigger_separation,
                              data_points=self.data_points)


class arbitrary_variable_swf(swf.Hard_Sweep):
    def __init__(self,control=None,parameter = None):

        super().__init__()
        self.name = 'arbitrary_variable_swf'
        self.parameter_name = 'phi_skew'
        self.unit = 'Deg'
        if control is not None:
            if not hasattr(control,'children'):
                control.children = [self]
            if not hasattr(control,'prepared'):
                control.prepared = [False]
        self.control_swf = control

    def set_parameter(self,value):
        pass

    def prepare(self):
        if self.control_swf is not None:
            #here there has to be a name map in the control swf in order to find
            #the correct swf. in control_swf and set it's prepared value to True.
          pass


class SegmentHardSweep(swf.UploadingSweepFunctionMixin, swf.Hard_Sweep):

    def __init__(self, sequence, upload=True, parameter_name='None', unit='',
                 start_pulsar=True, start_exclude_awgs=(), **kw):
        """A hardware sweep over segments in a sequence.

        Args:
            sequence (:class:`~pycqed.measurement.waveform_control.sequence.Sequence`):
                Sequence of segments to sweep over.
            upload (bool, optional):
                Whether to upload the sequences before measurement.
                Defaults to True.
            parameter_name (str, optional):
                Name for the sweep parameter. Defaults to 'None'.
            unit (str, optional):
                Unit for the sweep parameter. Defaults to ''.
            start_pulsar (bool, optional):
                Whether (a sub set of) the used AWGs will be started directly
                after upload. This can be used, e.g., to start AWGs that have
                only one unique segment and do not need to be synchronized to
                other AWGs and therefore do not need to be stopped when
                switching to the next segment in the sweep. Defaults to False.
            start_exclude_awgs (collection[str], optional):
                A collection of AWG names that will not be started directly
                after upload in case start_pulsar is True. Defaults to ().
        """
        super().__init__(sequence=sequence, upload=upload,
                         upload_first=True,
                         start_pulsar=start_pulsar,
                         start_exclude_awgs=start_exclude_awgs,
                         parameter_name=parameter_name, unit=unit, **kw)
        self.name = 'Segment hard sweep'

    def set_parameter(self, value):
        pass


class SegmentSoftSweep(swf.UploadingSweepFunctionMixin, swf.Soft_Sweep):
    # The following allows adding the class as placeholder in a
    # multi_sweep_function
    unit = ''

    def __init__(self, sequence_list, parameter_name='None', unit='',
                 upload_first=False, upload=True, **kw):
        super().__init__(sequence=sequence_list[0], upload=upload,
                         parameter_name=parameter_name, unit=unit, **kw)
        self.name = 'Segment soft sweep'
        self.sequence_list = sequence_list
        self.upload_next = upload_first

    def set_parameter(self, val, **kw):
        self.sequence = self.sequence_list[val]
        if self.upload_next:
            self.upload_sequence()
        self.upload_next = True


class T1_2nd_exc(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars, no_cal_points=6,
                 cal_points=True, upload=True, last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.cal_points = cal_points
        self.no_cal_points=no_cal_points
        self.last_ge_pulse = last_ge_pulse
        self.upload = upload
        self.name = 'T1 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.T1_2nd_exc_seq(times=self.sweep_points,
                                no_cal_points=self.no_cal_points,
                                pulse_pars=self.pulse_pars,
                                pulse_pars_2nd=self.pulse_pars_2nd,
                                RO_pars=self.RO_pars,
                                cal_points=self.cal_points,
                                last_ge_pulse=self.last_ge_pulse)


class Ramsey_multiple_detunings(swf.Hard_Sweep):

    def __init__(self, pulse_pars, RO_pars,
                 artificial_detunings=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detunings = artificial_detunings

        self.name = 'Ramsey_mult_det'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.ramsey_seq_multiple_detunings(times=self.sweep_points,
                           pulse_pars=self.pulse_pars,
                           RO_pars=self.RO_pars,
                           artificial_detunings=self.artificial_detunings,
                           cal_points=self.cal_points)


class Ramsey_2nd_exc_multiple_detunings(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 artificial_detunings=None, return_seq=False,
                 n=1, cal_points=True, upload=True, no_cal_points=6,
                 last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.artificial_detunings = artificial_detunings
        self.n = n
        self.cal_points = cal_points
        self.upload = upload
        self.name = 'Rabi 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'
        self.return_seq = return_seq
        self.last_ge_pulse = last_ge_pulse
        self.no_cal_points = no_cal_points

    def prepare(self, **kw):
        if self.upload:
            sqs2.ramsey_2nd_exc_seq_multiple_detunings(times=self.sweep_points,
                                                       pulse_pars=self.pulse_pars,
                                                       pulse_pars_2nd=self.pulse_pars_2nd,
                                                       RO_pars=self.RO_pars,
                                                       n=self.n,
                                                       cal_points=self.cal_points,
                                                       artificial_detunings=self.artificial_detunings,
                                                       upload=self.upload,
                                                       return_seq=self.return_seq,
                                                       last_ge_pulse=self.last_ge_pulse)


class Echo(swf.Hard_Sweep):
    def __init__(self, pulse_pars, RO_pars,
                 artificial_detuning=None,
                 cal_points=True,
                 upload=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.name = 'Echo'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs.echo_seq(times=self.sweep_points,
                         pulse_pars=self.pulse_pars,
                         RO_pars=self.RO_pars,
                         artificial_detuning=self.artificial_detuning,
                         cal_points=self.cal_points)


class Echo_2nd_exc(swf.Hard_Sweep):
    def __init__(self, pulse_pars, pulse_pars_2nd, RO_pars,
                 artificial_detuning=None, return_seq=False,
                 cal_points=True, upload=True, no_cal_points=6,
                 last_ge_pulse=True):
        super().__init__()
        self.pulse_pars = pulse_pars
        self.pulse_pars_2nd = pulse_pars_2nd
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.no_cal_points = no_cal_points
        self.artificial_detuning = artificial_detuning
        self.last_ge_pulse = last_ge_pulse
        self.return_seq = return_seq
        self.name = 'Echo 2nd excited state'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            sqs2.echo_2nd_exc_seq(times=self.sweep_points,
                                  pulse_pars=self.pulse_pars,
                                  pulse_pars_2nd=self.pulse_pars_2nd,
                                  RO_pars=self.RO_pars,
                                  artificial_detuning=self.artificial_detuning,
                                  cal_points=self.cal_points,
                                  no_cal_points=self.no_cal_points,
                                  upload=self.upload,
                                  return_seq=self.return_seq,
                                  last_ge_pulse=self.last_ge_pulse)


class Readout_pulse_scope_swf(swf.Hard_Sweep):
    def __init__(self, delays, pulse_pars, RO_pars, RO_separation,
                 cal_points=((-4, -3), (-2, -1)), prep_pulses=None,
                 comm_freq=225e6, verbose=False, upload=True):
        super().__init__()
        self.delays = delays
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.RO_separation = RO_separation
        self.cal_points = cal_points
        self.prep_pulses = prep_pulses
        self.comm_freq = comm_freq
        self.verbose = verbose
        self.upload = upload

        self.name = 'Readout pulse scope'
        self.parameter_name = 'delay'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            csqs.readout_pulse_scope_seq(
                delays=self.delays,
                pulse_pars=self.pulse_pars,
                RO_pars=self.RO_pars,
                RO_separation=self.RO_separation,
                cal_points=self.cal_points,
                prep_pulses=self.prep_pulses,
                comm_freq=self.comm_freq,
                verbose=self.verbose)


class readout_photons_in_resonator_swf(swf.Hard_Sweep):
    def __init__(self, delay_to_relax, delay_buffer, ramsey_times,
                 RO_pars, pulse_pars, cal_points=((-4, -3), (-2, -1)),
                 verbose=False, upload=True, return_seq=False,
                 artificial_detuning=None):
        super().__init__()
        self.delay_to_relax = delay_to_relax
        self.delay_buffer = delay_buffer
        self.ramsey_times = ramsey_times
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.verbose = verbose
        self.upload = upload
        self.return_seq = return_seq

        self.name = 'read out photons in RO'
        self.parameter_name = 'ramsey_times'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            csqs.readout_photons_in_resonator_seq(
                delay_to_relax=self.delay_to_relax,
                pulse_pars=self.pulse_pars,
                RO_pars=self.RO_pars,
                delay_buffer=self.delay_buffer,
                cal_points=self.cal_points,
                ramsey_times=self.ramsey_times,
                return_seq=self.return_seq,
                artificial_detuning=self.artificial_detuning,
                verbose=self.verbose)


class readout_photons_in_resonator_soft_swf(swf.Soft_Sweep):
    def __init__(self, hard_sweep):

        super().__init__()
        self.hard_sweep = hard_sweep
        self.name = 'read out photons in RO'
        self.parameter_name = 'delay_to_relax'
        self.unit = 's'

    def set_parameter(self, val):
        self.hard_sweep.upload = True 
        self.hard_sweep.delay_to_relax = val
        self.hard_sweep.prepare()


class Ramsey_decoupling_swf(swf.Hard_Sweep):
    def __init__(self, seq_func, pulse_pars, RO_pars,
                 artificial_detuning=None, nr_echo_pulses=4,
                 cpmg_scheme=True,
                 upload=True, cal_points=True):
        super().__init__()
        self.seq_func = seq_func
        self.pulse_pars = pulse_pars
        self.RO_pars = RO_pars
        self.upload = upload
        self.cal_points = cal_points
        self.artificial_detuning = artificial_detuning
        self.nr_echo_pulses = nr_echo_pulses
        self.cpmg_scheme = cpmg_scheme

        self.name = 'Ramsey_decoupling'
        self.parameter_name = 't'
        self.unit = 's'

    def prepare(self, **kw):
        if self.upload:
            self.seq_func(times=self.sweep_points,
                          pulse_pars=self.pulse_pars,
                          RO_pars=self.RO_pars,
                          artificial_detuning=self.artificial_detuning,
                          cal_points=self.cal_points,
                          nr_echo_pulses=self.nr_echo_pulses,
                          cpmg_scheme=self.cpmg_scheme)
