import logging

from qcodes import VisaInstrument
from qcodes.utils.validators import Enum, Ints

"""
Requires
    PySerial
"""

log = logging.getLogger(__name__)


class NanotecSMI33(VisaInstrument):
    """
    This is a driver for the Nanotec SMI33 Motor controller

    Status: tested with physical hardware

    Only a subset of all features have been included
    """
    def __init__(self, name: str, address, controller_id: str, **kwargs) -> None:
        """
        Args:
            name: The name of this instance
            address: The address of the controller
            controller_id: The id of the motor which can be either '*' or a number
                between 1 and 254
            kwargs: Additional keyword arguments
        """
        super().__init__(name, address, terminator='\r', **kwargs)

        # Set correct serial parameters
        self.visa_handle.baud_rate = 115200

        if controller_id not in [str(i) for i in range(1, 255)] + ['*']:
            raise ValueError('Controller_id must be * or a number from'
                             '1 to 254')
        self.controller_id = controller_id
        self._start_character = '#'

        self.add_parameter(
            'acceleration',
            label='Acceleration',
            unit='',
            get_cmd='b',
            set_cmd='b{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=1,
                      max_value=65535),
            docstring=('Acceleration rate'
                       'Min value: 1'
                       'Max value: 65535'))

        self.add_parameter(
            'acceleration_mode',
            label='Acceleration Mode',
            unit='',
            get_cmd=':ramp_mode',
            set_cmd=':ramp_mode={}',
            max_val_age=0,
            val_mapping={'Trapezoidal': 0,
                         'Sinusoidal': 1,
                         'Jerk-free': 2},
            docstring='Set the acceleration mode')

        self.add_parameter(
            'acceleration_jerk',
            label='Acceleration Jerk',
            unit='',
            get_cmd=':b',
            set_cmd=':b{}',
            get_parser=int,
            vals=Ints(min_value=1,
                      max_value=100000000),
            docstring=('Acceleration jerk'
                       'Min value: 1'
                       'Max value: 100000000'))

        self.add_parameter(
            'braking',
            label='Breaking',
            unit='',
            get_cmd='B',
            set_cmd='B{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=65535),
            docstring=('Braking rate'
                       'Min value: 1'
                       'Max value: 65535'))

        self.add_parameter(
            'braking_jerk',
            label='Braking Jerk',
            unit='',
            get_cmd=':B',
            set_cmd=':B{}',
            get_parser=int,
            vals=Ints(min_value=1,
                      max_value=100000000),
            docstring=('Breaking jerk'
                       'Min value: 1'
                       'Max value: 100000000'))

        self.add_parameter(
            'continuation_record',
            label='Continuation Record',
            unit='',
            get_cmd='N',
            set_cmd='N{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=32),
            docstring=('Record in EEPROM to continue with after finishing'
                       'the current record.'
                       'Min value is 0'
                       'Max value is 32'
                       'If the value is set to 0, this setting has no'
                       'effect.'))

        self.add_parameter(
            'command_response',
            label='Command Response',
            unit='',
            get_cmd=False,
            set_cmd='|{}',
            val_mapping={'Disabled': 0,
                         'Enabled': 1},
            docstring=('Enable or disable command response'
                       'If disabled, the controller will obey all commands'
                       'that are sent without responding.'
                       'Note that this disables response to any commands'
                       'including those reading out other settings'))

        for i in range(1, 7):
            self.add_parameter(
                f'digital_input_{i}_function',
                label=f'Digital Input {i} Function',
                unit='',
                get_cmd=f':port_in_{chr(i+96)}',
                set_cmd=f':port_in_{chr(i+96)}={{}}',
                max_val_age=0,
                val_mapping={'UserDefined': 0,
                             'StartRecord/ErrorReset': 1,
                             'RecordSelectBit0': 2,
                             'RecordSelectBit1': 3,
                             'RecordSelectBit2': 4,
                             'RecordSelectBit3': 5,
                             'RecordSelectBit4': 6,
                             'ExternalLimitSwitch': 7,
                             'Trigger': 8,
                             'Direction': 9,
                             'Enable': 10,
                             'Clock': 11,
                             'ClockDirectionMode1': 12,
                             'ClockDirectionMode2': 13},
                docstring=f'Function of digital input {i}')

        for i in range(1, 4):
            self.add_parameter(
                f'digital_output_{i}_function',
                label=f'Digital Output {i} Function',
                unit='',
                get_cmd=f':port_out_{chr(i+96)}',
                set_cmd=f':port_out_{chr(i+96)}={{}}',
                max_val_age=0,
                val_mapping={'UserDefined': 0,
                             'Ready': 1,
                             'Running': 2},
                docstring=f'Function of digital output {i}')

        self.add_parameter(
            'direction',
            label='Direction',
            unit='',
            get_cmd='d',
            set_cmd='d{}',
            max_val_age=0,
            val_mapping={'Left': 0,
                         'Right': 1},
            docstring='Direction of rotation (left or right)')

        self.add_parameter(
            'direction_change_on_repeat',
            label='Direction Change on Repeat',
            unit='',
            get_cmd='t',
            set_cmd='t{}',
            val_mapping={False: 0,
                         True: 1},
            docstring='Change direction of rotation on repeat')

        self.add_parameter(
            'error_correction',
            label='Error Correction',
            unit='',
            get_cmd='U',
            set_cmd='U{}',
            val_mapping={'Off': 0,
                         'CorrectionAfterTravel': 1,
                         'CorrectionDuringTravel': 2},
            docstring=('Error correction mode'
                       'Can be Off, CorrectionAfterTravel, or'
                       'CorrectionDuringTravel. CorrectionDuringTravel'
                       'is included only for compatibility reasons and'
                       'is implemented by CorrectionAfterTravel.'))

        self.add_parameter(
            'firmware_version',
            label='Firmware Version',
            unit='',
            get_cmd='v',
            set_cmd=False,
            docstring='Firmware version')

        self.add_parameter(
            'input_debounce_time',
            label='Input Debounce Time',
            unit='ms',
            get_cmd='K',
            set_cmd='K{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=250),
            docstring=('Debounce time for inputs in ms'
                       'Min value is 0'
                       'Max value is 250'))

        self.add_parameter(
            'io_input_mask',
            label='IO Input Mask',
            unit='',
            get_cmd='L',
            set_cmd='L{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=458815),
            docstring=('Set the IO input mask (32 bits)'
                       'If a bit of the mask is 1, the input or output is used'
                       'by the controller. If the bit is 0, the input or output'
                       'is available to the user.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'io_output_mask',
            label='IO Output Mask',
            unit='',
            get_cmd='Y',
            set_cmd='Y{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=196671),
            docstring=('Set the IO output mask (32 bits)'
                       'If a bit of the mask is 1, the input or output is used'
                       'by the controller (as long as it free in the IO input'
                       'mask). If the bit is 0, the input or output'
                       'is available to the user.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'io_polarity',
            label='IO Polarity',
            unit='',
            get_cmd='h',
            set_cmd='h{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=458815),
            docstring=('Set the IO polarity mask (32 bits)'
                       'If a bit of the mask is 1, the polarity is retained.'
                       'If the bit is 0, the polarity is reversed.'
                       'Bit 0: Input 1'
                       'Bit 1: Input 2'
                       'Bit 2: Input 3'
                       'Bit 3: Input 4'
                       'Bit 4: Input 5'
                       'Bit 5: Input 6'
                       'Bit 16: Output 1'
                       'Bit 17: Output 2'
                       'Invalid masks are discarded even if echoed by'
                       'the controller.'))

        self.add_parameter(
            'limit_switch_behavior',
            label='Limit Switch Behavior',
            unit='',
            get_cmd='l',
            set_cmd='l{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=4294967295),
            docstring=('Set the limit switch behavior'
                       '16-bit mask'))

        self.add_parameter(
            'maximum_frequency',
            label='Maximum Frequency',
            unit='Steps/s',
            get_cmd='o',
            set_cmd='o{}',
            get_parser=int,
            vals=Ints(min_value=1,
                      max_value=1000000),
            docstring=('Maximum frequency in steps per second'
                       'Min value: 1'
                       'Max value: 1000000'
                       'Maximum value depends on stepping mode'))

        self.add_parameter(
            'maximum_frequency2',
            label='Maximum Frequency 2',
            unit='Steps/s',
            get_cmd='n',
            set_cmd='n{}',
            get_parser=int,
            vals=Ints(min_value=1,
                      max_value=1000000),
            docstring=('Maximum frequency 2 in steps per second'
                       'Min value: 1'
                       'Max value: 1000000'
                       'Maximum value depends on stepping mode.'
                       'This speed is only used in flag positioning mode'))

        self.add_parameter(
            'minimum_frequency',
            label='Minimum Frequency',
            unit='Steps/s',
            get_cmd='u',
            set_cmd='u{}',
            get_parser=int,
            vals=Ints(min_value=1,
                      max_value=160000),
            docstring=('Minimum frequency in steps per second'
                       'Min value: 1'
                       'Max value: 160000'
                       'Motor starts moving at this speed at the beginning'
                       'of a record and then accelerates at the set rate'
                       'to the maximum frequency'))

        self.add_parameter(
            'motor_referenced',
            label='Motor Referenced',
            unit='',
            get_cmd=':is_referenced',
            set_cmd=False,
            get_parser=(lambda x: bool(int(x))),
            max_val_age=0,
            docstring=('Set the limit switch behavior'
                       '16-bit mask'))

        self.add_parameter(
            'pause',
            label='Pause',
            unit='ms',
            get_cmd='P',
            set_cmd='P{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=65535),
            docstring=('Pause time between current record and continuation'
                       'record in ms.'
                       'Min value is 0'
                       'Max value is 65535'
                       'Has no effect if the current record specifies no'
                       'continuation record'))

        self.add_parameter(
            'phase_current',
            label='Phase Current',
            unit='%',
            get_cmd='i',
            set_cmd='i{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=150),
            docstring=('Set the phase current (in %)'
                       'Min value: 0'
                       'Max value: 150'
                       'Should not be set above 100'))

        self.add_parameter(
            'phase_current_standstill',
            label='Phase Current at Standstill',
            unit='%',
            get_cmd='r',
            set_cmd='r{}',
            get_parser=int,
            max_val_age=0,
            vals=Ints(min_value=0,
                      max_value=150),
            docstring=('Set the phase current at standstill (in %)'
                       'Min value: 0'
                       'Max value: 150'
                       'Should not be set above 100'))

        self.add_parameter(
            'position',
            label='Position',
            unit='Steps',
            get_cmd='C',
            set_cmd=False,
            get_parser=int,
            max_val_age=0,
            docstring=('Current motor position relative to last reference'
                       'run'))

        self.add_parameter(
            'positioning_mode',
            label='Positioning Mode',
            unit='',
            get_cmd='p',
            set_cmd='p{}',
            val_mapping={'Relative': 1,
                         'Absolute': 2,
                         'InternalReferenceRun': 3,
                         'ExternalReferenceRun': 4,
                         'Speed': 5,
                         'Flag': 6,
                         'ClockManualLeft': 7,
                         'ClockManualRight': 8,
                         'ClockIntRefRun': 9,
                         'ClockExtRefRun': 10,
                         'AnalogSpeed': 11,
                         'Joystick': 12,
                         'AnalogPosition': 13,
                         'HWReference': 14,
                         'Torque': 15,
                         'CLQuickTest': 16,
                         'ClTest': 17,
                         'CLAutotune': 18,
                         'CLQuickTest2': 19},
            docstring='Positioning Mode')

        self.add_parameter(
            'quickstop',
            label='Quick Stop',
            unit='',
            get_cmd='H',
            set_cmd='H{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=8000),
            docstring=('Quickstop rate'
                       'Min value: 0'
                       'Max value: 8000'
                       'A value of 0 corresponds to an abrupt stop.'))

        self.add_parameter(
            'repetitions',
            label='Repetitions',
            unit='',
            get_cmd='W',
            set_cmd='W{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=254),
            docstring=('Repetitions of the current record'
                       'A value of zero means infinite repetitions'))

        self.add_parameter(
            'reset_position_error',
            label='Reset Position Error',
            unit='Steps',
            get_cmd=False,
            set_cmd='D{}',
            vals=Ints(min_value=-100000000,
                      max_value=+100000000),
            docstring=('Reset position error.'
                       'Used to clear speed errors and set position to'
                       'the given value'))

        self.add_parameter(
            'reverse_clearance',
            label='Reverse Clearance',
            unit='Steps',
            get_cmd='z',
            set_cmd='z{}',
            get_parser=int,
            vals=Ints(min_value=0,
                      max_value=9999),
            max_val_age=0,
            docstring=('Set the reverse clearance (in steps)'
                       'This is the number of steps that is added to'
                       'a movement command when changing directions to'
                       'compensate for play in the system'
                       'Min value: 0'
                       'Max value: 9999'))

        self.add_parameter(
            'status',
            label='Status',
            unit='',
            get_cmd='$',
            set_cmd=False,
            get_parser=int,
            max_val_age=0,
            docstring=('Controller status, 8-bit mask'
                       'Bit 0 == 1 means the controller is ready'
                       'Bit 1 == 1 means that zero position has been reached'
                       'Bit 2 == 1 means that there has been a position error'
                       'Bit 3 == 1 means that input 1 was set while the'
                       'controller was ready'
                       'Bit 4 is always set to 1'
                       'Bit 5 is always set to 0'
                       'Bit 6 is always set to 1'
                       'Bit 7 is always set to 0'))

        self.add_parameter(
            'step_mode',
            label='Step Mode',
            unit='',
            get_cmd='g',
            set_cmd='g{}',
            get_parser=int,
            vals=Enum(1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255),
            docstring=('Step mode'
                       'Must be one of '
                       '[1, 2, 4, 5, 8, 10, 16, 32, 64, 254, 255]'
                       'Values from 1 to 64 set the number of microsteps'
                       'per step. Value 254 selects the feed rate mode'
                       'and value 255 selects the adaptive step mode.'))

        self.add_parameter(
            'travel_distance',
            label='Travel Distance',
            unit='Steps',
            get_cmd='s',
            set_cmd='s{}',
            get_parser=int,
            vals=Ints(min_value=-100000000,
                      max_value=+100000000),
            docstring=('Travel distance in steps'
                       'Min value: -100000000'
                       'Max value: +100000000'
                       'Can only be positive when in relative positioning mode'
                       'In absolute positioning mode, this sets the target'
                       'position'))

        self.connect_message()

    def ask(self, cmd: str) -> str:
        # Case for 'long' commands
        if cmd[0] == ':' and len(cmd) > 2:
            motor_cmd = self._start_character + self.controller_id + cmd
            motor_response = super().ask(motor_cmd)
            response = motor_response.lstrip('0').split(cmd)[1].lstrip('=')
            return response
        # Case for all other normal commands
        else:
            motor_cmd = self._start_character + self.controller_id + 'Z' + cmd
            motor_response = super().ask(motor_cmd)
            response = motor_response.lstrip('0').split(cmd)[1].lstrip()
            return response


    def get_idn(self):
        # Make sure that the controller responds by placing it back into transmit
        # mode
        response = super().ask(self._start_character + self.controller_id + '|1')
        info = self.ask('v')
        info = info.split('_')
        return {'vendor': 'Nanotec',
                'model': info[0],
                'serial': None,
                'firmware': info[2]}

    def load_record_from_eeprom(self, index: int) -> None:
        """
        Load a record from the EEPROM
        Args
            index: integer between 1 and 32
        :return:
        """
        self.write('y' + str(index))

    def save_record_to_eeprom(self, index: int) -> None:
        """
        Save the current record to the EEPROM
        Args
            index: integer between 1 and 32
        :return:
        """
        self.write('>' + str(index))

    def start_motor(self):
        self.write('A')

    def stop_motor(self, ramp='Quickstop'):
        types = {
            'Quickstop': 0,
            'Brake': 1,
        }
        # TODO: add error checking to ramp type
        self.write('S' + str(types[ramp]))

    def write(self, cmd: str) -> None:
        # Handle case of command that disables responding to commands
        if cmd == '|0':
            motor_cmd = self._start_character + self.controller_id + cmd
            super().write(motor_cmd)
        else:
            motor_cmd = self._start_character + self.controller_id + cmd
            response =  super().ask(motor_cmd)
