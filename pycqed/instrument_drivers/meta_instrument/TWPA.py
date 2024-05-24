import qcodes as qc
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
from qcodes.utils import validators as vals
from pycqed.measurement import detector_functions as det
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis_v2 import amplifier_characterization as ca
from pycqed.instrument_drivers.meta_instrument.MeasurementObject import \
    MeasurementObject

class TWPAObject(MeasurementObject):
    """
    A meta-instrument containing the microwave generators needed for operating
    and characterizing the TWPA and the corresponding helper functions.
    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Add instrument reference parameters
        self.add_parameter('instr_pump',
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
                self.instr_ro_lo.get_instr().frequency(
                    val - self.acq_mod_freq())
            else:
                raise NotImplementedError("Continuous spectroscopy not "
                                          "implemented!")
                # TODO use instr_acq and instr_ro_lo from the base class
                # self.instr_signal.get_instr().frequency(val)
                # self.instr_lo.get_instr().frequency(val - self.acq_mod_freq())

        # Add signal control parameters
        def get_freq(self=self):
            if self.pulsed():
                return self.instr_ro_lo.get_instr().frequency() + \
                       self.acq_mod_freq()
            else:
                return self.instr_ro_lo.get_instr().frequency()

        self.add_parameter('signal_freq', label='Signal frequency', unit='Hz',
                           get_cmd=get_freq, set_cmd=set_freq)
        self.add_parameter('signal_power', label='Signal power', unit='dBm',
                           get_cmd=(lambda self=self:
                                    self.instr_ro_lo.get_instr().power()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_ro_lo.get_instr().power(val)))
        self.add_parameter('signal_status', label='Signal status',
                           get_cmd=(lambda self=self:
                                    self.instr_ro_lo.get_instr().status()),
                           set_cmd=(lambda val, self=self:
                                    self.instr_ro_lo.get_instr().status(val)))

        # add pulse parameters
        self.add_parameter('pulsed', parameter_class=ManualParameter,
                           vals=vals.Bool(), initial_value=True)
        self.add_parameter('pulse_length', parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 2.5e-6), unit='s',
                           initial_value=2.5e-6)
        self.add_parameter('pulse_amplitude', parameter_class=ManualParameter,
                           vals=vals.Numbers(0, 1.0), unit='V',
                           initial_value=0.01)


    def on(self):
        self.instr_pump.get_instr().on()

    def off(self):
        self.instr_pump.get_instr().off()

    def get_operation_dict(self, operation_dict=None):
        operation_dict = super().get_operation_dict(operation_dict)
        operation_dict['I ' + self.name] = {
            'pulse_type': 'VirtualPulse',
            'operation_type': 'Virtual',
        }
        for code, op in operation_dict.items():
            op['op_code'] = code
        return operation_dict

    def prepare_readout(self):
        # FIXME most of this could be done using MeasurementObject.prepare()
        #  including features like automatic switching for pulsed or
        #  not-pulsed (modulated or spec, respectively), and gated pulsed
        #  measurements
        UHF = self.instr_acq.get_instr()
        if not UHF.IDN()['model'].startswith('UHF'):
            raise NotImplementedError(
                'The UHFQC_correlation_detector used for TWPA tuneup '
                'measurements is not implemented for '
                '{acq_dev.name}, but only for ZI UHF devices.')
        pulsar = self.instr_pulsar.get_instr()

        # Prepare MWG states
        # There are (currently) 3 possible use cases:
        # - Continuous spectroscopy (self.pulsed() == False)
        # - Pulsed spectroscopy by gating the MWG
        # - Pulsed spectroscopy using readout pulse modulation
        # Currently the code below only works in the third case
        self.instr_pump.get_instr().pulsemod_state('Off')
        self.instr_ro_lo.get_instr().pulsemod_state('Off')
        self.instr_ro_lo.get_instr().on()
        # make sure that the lo frequency is set correctly
        self.signal_freq(self.signal_freq())

        # Prepare integration weights
        UHF.acquisition_set_weights(
            channels=[(0, 0), (0, 1)],
            weights_type=self.acq_weights_type(),
            mod_freq=self.acq_mod_freq(),
        )

        # FIXME this should use the RO operation from MeasurementObject
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
            acq_dev=UHF,
            AWG=UHF,
            integration_length=self.acq_length(),
            nr_averages=self.acq_averages(),
            channels=[(0, 0), (0, 1)],
            correlations=[((0, 0), (0, 0)), ((0, 1), (0, 1))],
            value_names=['I', 'Q', 'I^2', 'Q^2'],
            single_int_avg=True)

    def _measure_1D(self, parameter, values, label, analyze=True):
        # FIXME should be unified as a QuantumExperiment measurement
        #  (currently only works with a UHFQA acquisition device)

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
        # FIXME should be unified as a QuantumExperiment measurement
        #  (currently only works with a UHFQA acquisition device)

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