import time
import logging

from pycqed.instrument_drivers.physical_instruments.NanotecSMI33\
    import NanotecSMI33
from qcodes.utils.validators import Numbers

log = logging.getLogger(__name__)


class QudevMechDisplacerMotor(NanotecSMI33):
    def __init__(self, name: str, address: str, controller_id: str = '*', **kwargs) -> None:
        super().__init__(name, address, controller_id, **kwargs)
        self._lower_bound = None
        self._upper_bound = None
        self._is_initialized = False
        self.initialize()

    def _drive_absolute(self, position: int, direction: str,
                        speed: int = 600) -> None:
        """
        Drives the motor in absolute mode

        :param position:
        :param direction:
        :param speed:
        :return:
        """
        assert self._is_initialized, 'Motor is not initialized'
        assert self.motor_referenced, 'Motor is not referenced. Run init'
        # self.command_response('Disabled')
        self.acceleration(65535)
        self.acceleration_jerk(1)
        self.braking(65535)
        self.braking_jerk(100000000)
        self.continuation_record(0)
        self.direction(direction)
        self.direction_change_on_repeat(False)
        self.maximum_frequency(speed)
        self.maximum_frequency2(100)
        self.minimum_frequency(30)
        self.pause(0)
        self.positioning_mode('Absolute')
        self.quickstop(0)
        self.repetitions(1)
        self.travel_distance(position)
        # self.command_response('Enabled')

        self.start_motor()

    def drive_motor(self, position: int, speed: int = 600,
                    minimum_steps_to_limit: int = 50,
                    mode: str = 'Normal',
                    clearance_steps: int = 30) -> None:
        """
        Drives the motor to a given position (in steps)

        Works in either Normal mode where it drives directly to the target
        in terms of number of steps or in LeftToRight mode where it always
        approaches the target position from below
        :param position:
        :param speed:
        :param minimum_steps_to_limit:
        :param mode:
        :param clearance_steps:
        :return:
        """
        if not self._is_initialized:
            raise ValueError('Motor is not initialized. Run initialize')
        if not self.motor_referenced:
            raise ValueError('Motor is not referenced. Run initialize')
        self.safety_check()
        if (position < self._lower_bound + minimum_steps_to_limit or
                position > self._upper_bound - minimum_steps_to_limit):
            log.warning('Position outside of safe region. Value clamped.')
        position = max(position, self._lower_bound + minimum_steps_to_limit)
        position = min(position, self._upper_bound - minimum_steps_to_limit)
        if mode == 'Normal':
            if position < self.position():
                direction = 'Left'
            else:
                direction = 'Right'
            self._drive_absolute(position, direction, speed)
            # wait until limit is reached or controller is finished
            self.wait_until_status(5)
        elif mode == 'LeftToRight':
            if (position - clearance_steps) < self.position():
                direction = 'Left'
            else:
                direction = 'Right'
            self._drive_absolute(position - clearance_steps, direction, speed)
            self.wait_until_status(5)
            if direction == 'Left':
                direction = 'Right'
            else:
                direction = 'Left'
            self._drive_absolute(position, direction, speed)
            self.wait_until_status(5)

    def _escape_limit(self, direction: str, steps: int) -> None:
        """
        Drives the motor in relative mode to escape the limit region

        :param direction:
        :param steps:
        :return:
        """
        # self.command_response('Disabled')
        self.acceleration(65535)
        self.acceleration_jerk(1)
        self.braking(65535)
        self.braking_jerk(100000000)
        self.continuation_record(0)
        self.direction(direction)
        self.direction_change_on_repeat(False)
        self.maximum_frequency(100)
        self.maximum_frequency2(100)
        self.minimum_frequency(30)
        self.pause(0)
        self.positioning_mode('Relative')
        self.quickstop(0)
        self.repetitions(1)
        self.reset_position_error(0)
        self.travel_distance(steps)
        # self.command_response('Enabled')

        self.start_motor()
        # Wait until controller reaches limit
        # (LabVIEW code had a hard coded wait of 3s)
        self.wait_until_status(4)

    def _find_limits(self) -> None:
        """
        Finds limits of motor travel

        Drives the motor to the lower limit in referenced mode and then
        commands the motor to drive towards the upper limit until it
        reaches the limit switch
        :return:
        """
        # Prepare first record
        # An external reference run means that the motor will run until
        # the limit switch is triggered (in this case, due to the limit
        # switch behavior and IO polarity, it will run until the limit
        # switch is released).
        # After the limit switch is released,
        # self.command_response('Disabled')
        self.acceleration(6000)
        self.acceleration_jerk(1)
        self.braking(1)
        self.braking_jerk(100000000)
        self.continuation_record(0)
        self.direction('Left')
        self.direction_change_on_repeat(False)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.minimum_frequency(30)
        self.pause(200)
        self.positioning_mode('ExternalReferenceRun')
        self.quickstop(0)
        self.repetitions(1)
        self.travel_distance(100)
        # self.command_response('Enabled')
        self.save_record_to_eeprom(1)
        self.load_record_from_eeprom(1)
        self.start_motor()
        # Wait until motor reaches limit switch
        self.wait_until_status(5)
        self.reset_position_error(0)
        self._lower_bound = 0

        # Prepare second record
        # self.command_response('Disabled')
        self.acceleration(6000)
        self.acceleration_jerk(1)
        self.braking(1)
        self.braking_jerk(100000000)
        self.continuation_record(0)
        self.direction('Right')
        self.direction_change_on_repeat(False)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.minimum_frequency(30)
        self.pause(200)
        self.positioning_mode('Relative')
        self.quickstop(0)
        self.repetitions(254)
        self.travel_distance(100000000)
        # self.command_response('Enabled')
        self.save_record_to_eeprom(2)
        self.load_record_from_eeprom(2)
        # Start sequence
        self.start_motor()
        # Wait until motor reaches limit switch
        self.wait_until_status(4)
        self._upper_bound = self.position()

    def initialize(self, reverse_clearance: int = 0) -> None:
        """
        Prepares motor for operation

        Configure the motor settings and perform mechanical limit seeking
        Digital input 6 is configured as the external limit switch with
        inverted polarity (such that the controller is triggered when
        the limit switch turns off).
        The external limit switch is configured for free run backwards
        mode during an external reference run and stop mode during normal
        runs.
        """
        self.command_response('Enabled')
        self.firmware_version()
        self.phase_current(20)
        self.phase_current_standstill(0)
        self.limit_switch_behavior(int(0b0010010000100010))
        self.io_polarity(int(0b110000000000011111))
        self.digital_input_6_function('ExternalLimitSwitch')
        self.error_correction('Off')
        self.reverse_clearance(reverse_clearance)
        self.acceleration_mode('Jerk-free')
        self.acceleration_jerk(1)
        self.braking_jerk(100000000)
        self.step_mode(4)

        # Escape the limit if we are currently at one
        previous_direction = self.direction()
        if self.limit_switch_on():
            for i in range(1, 3):
                if previous_direction == 'Left':
                    direction = 'Right'
                else:
                    direction = 'Left'
                self._escape_limit(direction, steps=30*i)
                previous_direction = self.direction()

        # Find the limits by driving to the upper limit in external reference
        # run mode and then driving to lower limit in relative mode until
        # the limit switch is reached and the motor stops
        print(f'Finding {self.name} motor travel limits')
        self._find_limits()
        # Travel away from the limit
        self._travel_away_from_limit()
        self.limit_switch_behavior(0b10010000100010)
        self._is_initialized = True

        self.add_parameter(
            'setting_normalized',
            label='Normalized Setting',
            unit='',
            get_cmd=(lambda: self.position()),
            set_cmd=(lambda x: self.drive_motor(x)),
            get_parser=(lambda x: float(x) / self._upper_bound),
            set_parser=(lambda x: round(self._upper_bound * x)),
            vals=Numbers(min_value=0.0,
                         max_value=1.0),
            docstring=('Normalized setting (position)'
                       'Min value: 0'
                       'Max value: 1'
                       'Automatically mapped between lower and upper'
                       'bound of limit switch'))

    def limit_switch_on(self) -> bool:
        """
        Checks whether limit switch is activated

        :return:
        """
        return self.io_output_mask() & 32 > 0

    def safety_check(self) -> None:
        """
        Checks that all critical parameters are set correctly

        :return:
        """
        self.command_response('Enabled')
        error_correction = self.error_correction()
        input_6_function = self.digital_input_6_function()
        input_config = self.io_output_mask()
        io_polarity = self.io_polarity()
        limit_switch_config = self.limit_switch_behavior()
        microsteps = self.step_mode()
        phase_current = self.phase_current()
        standstill_current = self.phase_current_standstill()

        if error_correction != 'Off':
            raise ValueError('Error correction not disabled')
        if input_6_function != 'ExternalLimitSwitch':
            raise ValueError('Input 6 not configured as external limit switch')
        if input_config & 32 != 0:
            raise ValueError('Bit 5 of IO input mask not set to 0')
        if io_polarity & 32 != 0:
            raise ValueError('Bit 5 of IO polarity not set to 0')
        if microsteps not in [1, 2, 4, 8]:
            raise ValueError('Unusual number of microsteps')
        if limit_switch_config != 0b10010000100010:
            raise ValueError(f'Limit switch behavior not set to {9250:b}')
        if not 1 <= phase_current <= 20:
            raise ValueError('Phase current note between 1 and 20%')
        if not 0 <= standstill_current <= 1:
            raise ValueError('Standstill current not between 0 and 1%')

    def _travel_away_from_limit(self) -> None:
        """
        Travels away from the limit after finding the limit

        :return:
        """
        if self.direction() == 'Left':
            reverse_direction = 'Right'
        else:
            reverse_direction = 'Left'
        # self.command_response('Disabled')
        self.acceleration(6000)
        self.acceleration_jerk(1)
        self.braking(65535)
        self.braking_jerk(100000000)
        self.continuation_record(0)
        self.direction(reverse_direction)
        self.direction_change_on_repeat(False)
        self.limit_switch_behavior(0b100010000100010)
        self.maximum_frequency(250)
        self.maximum_frequency2(250)
        self.minimum_frequency(30)
        self.pause(200)
        self.positioning_mode('Relative')
        self.quickstop(0)
        self.repetitions(1)
        self.travel_distance(300)
        # self.command_response('Enabled')

        self.start_motor()
        # wait until controller finishes moving or the limit is reached
        self.wait_until_status(5)

    def wait_until_status(self, mask: int = 5, timeout: int = 30) -> None:
        """
        Waits until controller has a status given by the mask

        :param mask:
        :param timeout:
        :return:
        """
        self.command_response('Enabled')
        t0 = time.time()
        while (time.time() - t0) < timeout and (self.status() & mask == 0):
            # TODO: add warning when timeout occurs
            time.sleep(0.05)
