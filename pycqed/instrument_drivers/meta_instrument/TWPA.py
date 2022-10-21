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
        self.add_parameter('instr_pump', parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_signal',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('instr_lo', parameter_class=InstrumentRefParameter)

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

    def get_idn(self):
        return {'driver': str(self.__class__), 'name': self.name}

    def on(self):
        self.instr_pump.get_instr().on()

    def off(self):
        self.instr_pump.get_instr().off()
