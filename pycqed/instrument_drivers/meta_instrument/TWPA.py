import qcodes as qc
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from pycqed.measurement import detector_functions as det
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import amplifier_characterization as ca

class TWPAObject(qc.Instrument):
    """
    A meta-instrument containing the microwave generators needed for operating
    and characterizing the TWPA and the corresponding helper functions.
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.msmt_suffix = '_' + self.name

        # Add instrument reference parameters
        self.add_parameter('instr_pump', parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_signal',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_lo', parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_uhf',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_mc', parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_pulsar',
                           parameter_class=InstrumentRefParameter)

        # Add pump control parameters
        self.add_parameter('pump_freq', label='Pump frequency', unit='Hz',
                           get_cmd=(lambda self=self:
                                    self.instr_pump.get_instr().frequency()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_pump.get_instr().frequency(val)))
        self.add_parameter('pump_power', label='Pump power', unit='dBm',
                           get_cmd=(lambda self=self:
                                    self.instr_pump.get_instr().power()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_pump.get_instr().power(val)))
        self.add_parameter('pump_status', label='Pump status',
                           get_cmd=(lambda self=self:
                                    self.instr_pump.get_instr().status()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_pump.get_instr().status(val)))

        # Add signal control parameters
        def set_freq(val, self=self):
            if self.pulsed():
                self.instr_signal.get_instr().frequency(val - self.acq_mod_freq())
                if self.instr_lo() != self.instr_signal():
                    self.instr_lo.get_instr().frequency(val - self.acq_mod_freq())
            else:
                self.instr_signal.get_instr().frequency(val)
                self.instr_lo.get_instr().frequency(val - self.acq_mod_freq())

        # Add signal control parameters
        def get_freq(self=self):
            if self.pulsed():
                return self.instr_signal.get_instr().frequency() + \
                       self.acq_mod_freq()
            else:
                return self.instr_signal.get_instr().frequency()

        self.add_parameter('signal_freq', label='Signal frequency', unit='Hz',
                           get_cmd=get_freq, set_cmd=set_freq)
        self.add_parameter('signal_power', label='Signal power', unit='dBm',
                           get_cmd=(lambda self=self:
                                    self.instr_signal.get_instr().power()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_signal.get_instr().power(val)))
        self.add_parameter('signal_status', label='Signal status',
                           get_cmd=(lambda self=self:
                                    self.instr_signal.get_instr().status()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_signal.get_instr().status(val)))

        # Add acquisition parameters
        self.add_parameter('acq_length', parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 2.275e-6), unit='s',
                           initial_value=2.275e-6)
        self.add_parameter('acq_mod_freq', parameter_class=ManualParameter,
                           vals=vals.Numbers(-900e6, 900e6), unit='Hz',
                           label='Intermediate frequency',
                           initial_value=225e6)
        self.add_parameter('acq_averages', parameter_class=ManualParameter,
                           vals=vals.Ints(0), initial_value=2**10)
        self.add_parameter('acq_weights_type', parameter_class=ManualParameter,
                           vals=vals.Enum('DSB', 'SSB'), initial_value='SSB')

        # add pulse parameters
        self.add_parameter('pulsed', parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=True)
        self.add_parameter('pulse_length', parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 2.5e-6), unit='s',
                           initial_value=2.5e-6)
        self.add_parameter('pulse_amplitude', parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 1.0), unit='V',
                           initial_value=0.01)

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def on(self):
        self.instr_pump.get_instr().on()

    def off(self):
        self.instr_pump.get_instr().off()

    def prepare_readout(self):
        UHF = self.instr_uhf.get_instr()
        pulsar = self.instr_pulsar.get_instr()

        # Prepare MWG states
        self.instr_pump.get_instr().pulsemod_state('Off')
        self.instr_signal.get_instr().pulsemod_state('Off')
        self.instr_lo.get_instr().pulsemod_state('Off')
        self.instr_signal.get_instr().on()
        self.instr_lo.get_instr().on()
        # make sure that the lo frequency is set correctly
        self.signal_freq(self.signal_freq())

        # Prepare integration weights
        if self.acq_weights_type() == 'SSB':
            UHF.prepare_SSB_weight_and_rotation(IF=self.acq_mod_freq())
        else:
            UHF.prepare_DSB_weight_and_rotation(IF=self.acq_mod_freq())

        # Program the AWG
        if self.pulsed():
            pulse = {'pulse_type': 'GaussFilteredCosIQPulse',
                     'I_channel': pulsar._id_channel('ch1', UHF.name),
                     'Q_channel': pulsar._id_channel('ch2', UHF.name),
                     'amplitude': self.pulse_amplitude(),
                     'pulse_length': self.pulse_length(),
                     'gaussian_filter_sigma': 1e-08,
                     'mod_frequency': self.acq_mod_freq(),
                     'operation_type': 'RO'}
        else:  # dummy_pulse
            pulse = {'pulse_type': 'SquarePulse',
                     'channels': pulsar.find_awg_channels(UHF.name),
                     'amplitude': 0,
                     'length': 100e-9,
                     'operation_type': 'RO'}
        sq.pulse_list_list_seq([[pulse]])
        pulsar.start(exclude=[UHF.name])

        # Create the detector
        return det.UHFQC_correlation_detector(
            UHFQC=UHF,
            AWG=UHF,
            integration_length=self.acq_length(),
            nr_averages=self.acq_averages(),
            channels=[0, 1],
            correlations=[(0, 0), (1, 1)],
            value_names=['I', 'Q'],
            single_int_avg=True)

    def _measure_1D(self, parameter, values, label, analyze=True):

        MC = self.instr_mc.get_instr()

        initial_value = parameter()

        detector = self.prepare_readout()

        MC.set_sweep_function(parameter)
        MC.set_sweep_points(values)
        MC.set_detector_function(detector)

        MC.run(name=label + self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(auto=True)

        parameter(initial_value)

    def _measure_2D(self, parameter1, parameter2, values1, values2,
                    label, analyze=True):

        MC = self.instr_mc.get_instr()

        detector = self.prepare_readout()

        initial_value1 = parameter1()
        initial_value2 = parameter2()

        MC.set_sweep_function(parameter1)
        MC.set_sweep_function_2D(parameter2)
        MC.set_sweep_points(values1)
        MC.set_sweep_points_2D(values2)
        MC.set_detector_function(detector)
        MC.run_2D(name=label + self.msmt_suffix)
        if analyze:
            ma.MeasurementAnalysis(TwoD=True, auto=True)

        parameter1(initial_value1)
        parameter2(initial_value2)

    def measure_vs_pump_freq(self, pump_freqs, analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'pump_freq_scan_pp{self.pump_power():.2f}dB_' + \
                f'sf{self.signal_freq()/1e9:.3f}G'
        self._measure_1D(self.pump_freq, pump_freqs, label, analyze)
        self.off()
        label = f'pump_freq_scan_off_sf{self.signal_freq() / 1e9:.3f}G'
        self._measure_1D(self.pump_freq, pump_freqs[:1], label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='pump_freq_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)

    def measure_vs_signal_freq(self, signal_freqs, analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'signal_freq_scan_pp{self.pump_power():.2f}dB_' + \
                f'pf{self.pump_freq() / 1e9:.3f}G'
        self._measure_1D(self.signal_freq, signal_freqs, label, analyze)
        self.off()
        label = 'signal_freq_scan_off'
        self._measure_1D(self.signal_freq, signal_freqs, label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='signal_freq_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)

    def measure_vs_pump_power(self, pump_powers, analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'pump_power_scan_pf{self.pump_freq() / 1e9:.3f}G_' + \
                f'sf{self.signal_freq() / 1e9:.3f}G'
        self._measure_1D(self.pump_power, pump_powers, label, analyze)
        self.off()
        label = f'pump_power_scan_off_sf{self.signal_freq() / 1e9:.3f}G'
        self._measure_1D(self.pump_power, pump_powers[:1], label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='pump_power_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)

    def measure_vs_signal_freq_pump_freq(self, signal_freqs, pump_freqs,
                                         analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'signal_freq_pump_freq_scan_pp{self.pump_power():.2f}dB'
        self._measure_2D(self.signal_freq, self.pump_freq, signal_freqs,
                         pump_freqs, label, analyze)
        self.off()
        label = f'signal_freq_pump_freq_scan_off'
        self._measure_1D(self.signal_freq, signal_freqs, label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='signal_freq_pump_freq_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)

    def measure_vs_signal_freq_pump_power(self, signal_freqs, pump_powers,
                                         analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'signal_freq_pump_power_scan_pf{self.pump_freq() / 1e9:.3f}G'
        self._measure_2D(self.signal_freq, self.pump_power, signal_freqs,
                         pump_powers, label, analyze)
        self.off()
        label = f'signal_freq_pump_power_scan_off'
        self._measure_1D(self.signal_freq, signal_freqs, label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='signal_freq_pump_power_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)

    def measure_vs_pump_freq_pump_power(self, pump_freqs, pump_powers,
                                        analyze=True):
        timestamp_start = a_tools.current_timestamp()
        self.on()
        label = f'pump_freq_pump_power_scan_sf{self.signal_freq() / 1e9:.3f}G'
        self._measure_2D(self.pump_freq, self.pump_power, pump_freqs,
                         pump_powers, label, analyze)
        self.off()
        label = f'pump_freq_pump_power_scan_off_' + \
                f'sf{self.signal_freq() / 1e9:.3f}G'
        self._measure_1D(self.pump_freq, pump_freqs[:1], label, analyze)
        if analyze:
            timestamps = a_tools.get_timestamps_in_range(
                timestamp_start, label='pump_freq_pump_power_scan')
            ca.Amplifier_Characterization_Analysis(timestamps)
