from pycqed.instrument_drivers.instrument import Instrument
import qcodes.utils.validators as vals
import numpy as np
import logging

log = logging.getLogger(__name__)


class HDAWG_TriggerDevice(Instrument):
    """Meta instrument to control an HDAWG channel pair as main trigger"""
    USER_REG_REPETITIONS = 0
    USER_REG_SEPARATION = 1
    GRANULARITY = 16
    PULSE_LENGTH = 64  # samples

    TEMPLATE = (
        "var reps = getUserReg({reps});\n"
        "var sep = getUserReg({sep});\n"
        "\n"
        "wave w_pulse = marker({pulse_length}, 1);\n"
        "repeat (reps) {{\n"
        "  playWave(w_pulse, w_pulse);\n"
        "  playZero(sep);\n"
        "}}\n"
    )

    def __init__(self, name, awg, awg_nr):

        super().__init__(name=name)
        self.awg = awg
        self._awg_nr = None  # to avoid that the next line programs the AWG
        self.pulse_length = self.PULSE_LENGTH
        self.awg_nr = awg_nr  # this programs the AWG

        self.add_parameter(
            'pulse_period',
            label='Pulse period',
            get_cmd=(lambda self=self: self.awg.get(
                f'awgs_{self.awg_nr}_userregs_{self.USER_REG_SEPARATION}')),
            set_cmd=(lambda val, self=self: self.awg.set(
                f'awgs_{self.awg_nr}_userregs_{self.USER_REG_SEPARATION}',
                int(val))),
            get_parser=self._pulse_period_get_parser,
            set_parser=self._pulse_period_set_parser,
            unit='s',
            vals=vals.Numbers(min_value=20e-9, max_value=2e3),
            initial_value=20e-9,
            docstring=('Min value: 20 ns, Max value: 2000 s.'))
        self.add_parameter(
            'repetitions',
            label='Number of repetitions',
            get_cmd=(lambda self=self: self.awg.get(
                f'awgs_{self.awg_nr}_userregs_{self.USER_REG_REPETITIONS}')),
            set_cmd=(lambda val, self=self: self.awg.set(
                f'awgs_{self.awg_nr}_userregs_{self.USER_REG_REPETITIONS}',
                int(val))),
            unit='',
            vals=vals.Ints(1, int(1e15)))

    @property
    def pulse_length(self):
        return self._pulse_length

    @pulse_length.setter
    def pulse_length(self, value):
        # TODO this could become a qcodes parameter in unit of seconds
        if np.abs(value % self.GRANULARITY) > 1e-12:
            raise ValueError(
                f'Pulse length {value} samples is not a multiple of the '
                f'waveform granularity {self.GRANULARITY} samples.')
        self._pulse_length = value
        self.program_awg()

    @property
    def awg_nr(self):
        return self._awg_nr

    @awg_nr.setter
    def awg_nr(self, value):
        self._awg_nr = value
        self.awg.exclude_from_stop = [self._awg_nr]
        self.program_awg()

    def program_awg(self):
        if self._awg_nr is None:  # during init
            return
        awg_str = self.TEMPLATE.format(
            reps=self.USER_REG_REPETITIONS,
            sep=self.USER_REG_SEPARATION,
            pulse_length=self.pulse_length
        )
        self.awg.configure_awg_from_string(awg_nr=self.awg_nr,
                                           program_string=awg_str)

    def _pulse_period_set_parser(self, pulse_period):
        samples = pulse_period * self.awg.clock_freq()
        if np.abs(samples % self.GRANULARITY) > 1e-12:
            raise ValueError(
                f'Pulse period {pulse_period}s is not a multiple of the '
                f'waveform granularity {self.GRANULARITY} samples.')
        self.stop()
        return samples - self.pulse_length

    def _pulse_period_get_parser(self, pulse_distance):
        samples = pulse_distance + self.pulse_length
        return samples / self.awg.clock_freq()

    def start(self, **kw):
        """Start the playback of trigger pulses
        :param kw: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        self.awg.set(f'awgs_{self.awg_nr}_enable', 1)

    def stop(self):
        """Start the playback of trigger pulses"""
        self.awg.set(f'awgs_{self.awg_nr}_enable', 0)
