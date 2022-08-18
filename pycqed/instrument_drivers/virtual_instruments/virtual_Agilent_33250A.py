import logging
from pycqed.instrument_drivers.instrument import Instrument
from qcodes.utils.validators import Numbers, Enum, MultiType

log = logging.getLogger(__name__)

class Virtual_Agilent_33250A(Instrument):  
    """
    Driver code for virtual Agilent 33250A trigger.
    Based on Agilent_33250A class
    Only most commonly used commands of the device integrated at this stage.
    """
    def __init__(self, name, address=None, **kwargs):

        super().__init__(name, **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd=None,
                           set_cmd=None,
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-6,
                                        max_value=80e6),
                           docstring=('Command for setting the pulse frequency. Min value: 1 uHz, Max value: 80 MHz'))
        self.add_parameter(name='pulse_shape',
                           label='Pulse shape',
                           get_cmd=None,
                           set_cmd=None,
                           vals=Enum('SIN','SQU'),
                           docstring=('Command for setting the desired pulse shape. Currently supported: SIN, SQU.'))
        self.add_parameter(name='pulse_width',
                           label='Pulse width',
                           get_cmd=None,
                           set_cmd=None,
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(min_value=8e-9, max_value=2e3),
                           docstring=('Command for setting the desired pulse width. Min value: 8 ns, Max value: 2000 s.'))
        self.add_parameter('pulse_period',
                           label='Pulse period',
                           get_cmd=None,
                           set_cmd=None,
                           get_parser=float,
                           set_parser=float,
                           unit='s',
                           vals=Numbers(min_value=20e-9, max_value=2e3),
                           docstring=('Command for setting the desired pulse period. Min value: 20 ns, Max value: 2000 s.'))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='Vpp',
                           get_cmd=None,
                           set_cmd=None,
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=1e-3, max_value=20),
                           docstring=('Command for setting the desired pulse amplitude. Min value for 50 Ohm load: 1 mVpp, Max value for 50 Ohm load: 10 Vpp. Min value for high impedance load: 2 mVpp, Max value for high impedance load: 20 Vpp.'))
        self.add_parameter('offset',
                           label='Offset',
                           unit='V',
                           get_cmd=None,
                           set_cmd=None,
                           get_parser=float,
                           set_parser=float,
                           vals=Numbers(min_value=-10, max_value=10),
                           docstring=('Command for setting the desired dc offset. Min value for 50 Ohm load: -10 V, Max value for 50 Ohm load: 10 V. Min value for high impedance load: -10 V, Max value for high impedance load: 10 Vpp.'))
        self.add_parameter(name='output',
                           get_cmd=None,
                           set_cmd=None,
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring=('Command for switching on/off the device output.'))
        self.add_parameter(name='output_sync',
                           get_cmd=None,
                           set_cmd=None,
                           val_mapping={'OFF': 0,
                                        'ON': 1},
                           docstring='Command for switching on/off the device output synchronization.')
        self.add_parameter(name='load_impedance',
						   label='Load impedance',
						   unit='Ohm',
                           get_cmd=None,
                           set_cmd=None,	
                           get_parser=float,
                           set_parser=float,
                           vals=MultiType(Numbers(min_value=1, max_value=10e3), Enum('INF')),
                           docstring=("Command for setting the load impedance in Ohms. Min value: 1 Ohm, Max value: 10 kOhm or 'INF'"))						   
        
        self.connect_message()

    def reset(self):
        """
        pass
        """
        pass

    def start(self, **kw):
        """
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        self.output('ON')
    
    def stop(self):
        self.output('OFF')
    