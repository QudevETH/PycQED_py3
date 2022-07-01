from qcodes import Instrument
import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ManualParameter


class RTB2000FunctionGeneratorTriggerDevice(Instrument):
    """Meta instrument to control the RTB2000 function generator as trigger
    device

    Args:
        name (str): name of the meta instrument
        instrument (Instrument): RTB2000 oscilloscope qcodes instrument
    """

    DUTY_CYCLE = .2

    def __init__(self, name, instrument):
        super().__init__(name=name)
        self.instr = instrument

        self.add_parameter(
            'pulse_length',
            get_cmd=(lambda self=self: self.instr.ask('WGENerator:FREQuency?')),
            set_cmd=(
                lambda val, self=self: self.instr.write(
                    f'WGENerator:FREQuency {val}')),
            get_parser=lambda f, d=self.DUTY_CYCLE: d / float(f),
            set_parser=lambda T, d=self.DUTY_CYCLE: d / T,
            unit='s',
            vals=vals.Numbers(min_value=20e-9, max_value=1e6),
            initial_value=25e-9)

        self.add_parameter(
            'pulse_period',
            get_cmd=(lambda self=self: self._period_getter()),
            set_cmd=(
                lambda val, self=self: self._period_setter(val)),
            unit='s',
            vals=vals.Numbers(min_value=50e-9, max_value=17),
            initial_value=200e-6)

        self.add_parameter(
            'amplitude',
            get_cmd=(lambda self=self: self.instr.ask('WGENerator:VOLTage?')),
            set_cmd=(
                lambda val, self=self: self.instr.write(
                    f'WGENerator:VOLTage {val}')),
            get_parser=float,
            unit='V',
            vals=vals.Numbers(min_value=6e-3, max_value=6),
            initial_value=1)

        self.add_parameter(
            'offset',
            get_cmd=(
                lambda self=self: self.instr.ask('WGENerator:VOLTage:OFFSet?')),
            set_cmd=(
                lambda val, self=self: self.instr.write(
                    f'WGENerator:VOLTage:OFFSet {val}')),
            get_parser=float,
            unit='V',
            vals=vals.Numbers(min_value=-3, max_value=3),
            initial_value=.5)

        self.configure()

    def _period_setter(self, value):
        t = self.pulse_length()
        self.instr.write(f'WGENerator:BURSt:ITIMe {value - t}')
        # FIXME: this needs to be updated every time after changing pulse length

    def _period_getter(self):
        t = self.pulse_length()
        return float(self.instr.ask('WGENerator:BURSt:ITIMe?')) + t

    def start(self, **kw):
        """Start the playback of trigger pulses at the RTB2000 function generator
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        self.instr.write('WGENerator:OUTPut ON')

    def stop(self):
        """Stop the playback of trigger pulses at the RTB2000 function generator"""
        self.instr.write('WGENerator:OUTPut OFF')

    def configure(self):
        self.instr.write('WGENerator:FUNCtion PULSe')
        self.instr.write(
            f'WGENerator:FUNCtion:PULSe:DCYCle {self.DUTY_CYCLE * 100}')
        self.instr.write(f'WGENerator:BURSt:NCYCle 1')
        self.instr.write(f'WGENerator:BURSt:TRIGger CONTinuous')
        self.instr.write(f'WGENerator:BURSt:PHASe 0')
        self.instr.write(f'WGENerator:BURSt ON')


class RTB2000PatternGeneratorTriggerDevice(Instrument):
    """Meta instrument to control the RTB2000 pattern generator as trigger
    device

    Args:
        name (str): name of the meta instrument
        instrument (Instrument): RTB2000 oscilloscope qcodes instrument
    """

    def __init__(self, name, instrument):
        super().__init__(name=name)
        self.instr = instrument

        self.add_parameter(
            'pulse_length_target',
            unit='s',
            initial_value=30e-9,
            parameter_class=ManualParameter,
            vals=vals.Numbers()
        )

        self.add_parameter(
            'pulse_period',
            label='Pulse period',
            get_cmd=(
                lambda self=self: self.instr.ask('PGENerator:PATTern:PERiod?')),
            set_cmd=(
                lambda val, self=self: self._period_setter(val)),
            get_parser=float,
            unit='s',
            vals=vals.Numbers(min_value=2e-6, max_value=10416),
            initial_value=2e-6)

        self.add_parameter(
            'pulse_length',
            get_cmd=(
                lambda self=self: self._pulse_length_getter()),
            unit='s',
        )

    def _period_setter(self, value):
        self.instr.write(f'PGENerator:PATTern:PERiod {value}')
        dutyc = self.pulse_length_target() / value
        self.instr.write(f'PGENerator:PATTern:SQUarewave:DCYCle {dutyc * 100}')

    def _pulse_length_getter(self):
        dutyc = float(
            self.instr.ask(f'PGENerator:PATTern:SQUarewave:DCYCle?')) / 100
        print(dutyc)
        print(self.pulse_period())
        return dutyc * self.pulse_period()

    def start(self, **kw):
        """Start the playback of trigger pulses at the RTB2000 pattern generator
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        self.instr.write('PGENerator:FUNCtion SQUarewave')
        self.instr.write('PGENerator:PATTern:SQUarewave:POLarity NORMal')
        self.instr.write('PGENerator:PATTern:STATe ON')

    def stop(self):
        """Stop the playback of trigger pulses at the RTB2000 pattern generator"""

        self.instr.write('PGENerator:PATTern:STATe OFF')
