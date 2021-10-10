from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Enum, MultiType


class Agilent_33250A(VisaInstrument):
    """
    This is the driver code for the Agilent 33250A AWG.

    Status: Not tested

    Only most commonly used commands of the device integrated at this stage.
    """
    def __init__(self, name ,address,**kwargs):

        super().__init__(name,address,**kwargs)

        # set output units to voltage peak-to-peak
        self.visa_handle.write('VOLT:UNIT VPP')

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-6,
                                        max_value=80e6),
                           docstring=('Command for setting the pulse frequency. Min value: 1 uHz, Max value: 80 MHz'))
        self.add_parameter(name='pulse_shape',
                           label='Pulse shape',
                           get_cmd='FUNC?',
                           set_cmd='FUNC {}',
                           vals=Enum('SIN','SQU'),
                           docstring=('Command for setting the desired pulse shape. Currently supported: SIN, SQU.'))
        self.add_parameter(name='pulse_width',
                           label='Pulse width',
                           get_cmd='PULS:WIDT?',
                           set_cmd='PULS:WIDT {}',
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(min_value=8e-9, max_value=2e3),
                           docstring=('Command for setting the desired pulse width. Min value: 8 ns, Max value: 2000 s.'))
        self.add_parameter('pulse_period',
                           label='Pulse period',
                           get_cmd='PULS:PER?',
                           set_cmd='PULS:PER {}',
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(min_value=20e-9, max_value=2e3),
                           docstring=('Command for setting the desired pulse period. Min value: 20 ns, Max value: 2000 s.'))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='Vpp',
                           get_cmd='VOLT?',
                           set_cmd='VOLT {}',
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-3, max_value=20),
                           docstring=('Command for setting the desired pulse amplitude. Min value for 50 Ohm load: 1 mVpp, Max value for 50 Ohm load: 10 Vpp. Min value for high impedance load: 2 mVpp, Max value for high impedance load: 20 Vpp.'))
        self.add_parameter('offset',
                           label='Offset',
                           unit='V',
                           get_cmd='VOLT:OFFS?',
                           set_cmd='VOLT:OFFS {}',
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=-10, max_value=10),
                           docstring=('Command for setting the desired dc offset. Min value for 50 Ohm load: -10 V, Max value for 50 Ohm load: 10 V. Min value for high impedance load: -10 V, Max value for high impedance load: 10 Vpp.'))
        self.add_parameter(name='output',
                           get_cmd='OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring=('Command for switching on/off the device output.'))
        self.add_parameter(name='output_sync',
                           get_cmd='OUTP:SYNC?',
                           set_cmd='OUTP:SYNC {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring='Command for switching on/off the device output synchronization.')
        self.add_parameter(name='load_impedance',
						   label='Load impedance',
						   unit='Ohm',
						   get_cmd='OUTP:LOAD?',
						   set_cmd='OUTP:LOAD {}',	
                           get_parser=float,
                           set_parser=float,
                           vals=MultiType(Numbers(min_value=1, max_value=10e3), Enum('INF')),
                           docstring=("Command for setting the load impedance in Ohms. Min value: 1 Ohm, Max value: 10 kOhm or 'INF'"))						   
        
        self.connect_message()
    
    def reset(self):
        self.write('*RST')

    def start(self, **kw):
        """
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        self.write('OUTP OFF')
        self.write('BURS:STAT OFF')
        self.write('BURS:MODE TRIG')
        self.write('BURS:NCYC INF')
        self.write('TRIG:SOUR BUS')

        self.write('BURS:STAT ON')
        self.write('OUTP ON')
        self.write('TRIG')
    
    def stop(self):
        self.write('OUTP OFF')
    