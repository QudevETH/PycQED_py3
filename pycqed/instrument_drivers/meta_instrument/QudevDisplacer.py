import time
import logging

from qcodes import Instrument
from qcodes.instrument.parameter import InstrumentRefParameter
from qcodes.utils.validators import Enum, Ints, Numbers


class QudevDisplacer(Instrument):
    """
    A meta-instrument containing the two sub-modules that make up a
    displacer board along with routines for optimizing the parameters
    for a TWPA
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # Add instrument references
        self.add_parameter('attenuator_instr',
                           parameter_class=InstrumentRefParameter)
        self.add_parameter('phase_shifter_instr',
                           parameter_class=InstrumentRefParameter)

        # Add displacer board control parameters
        self.add_parameter(
            'attenuator_setting',
            label='Attenuator Setting',
            unit='',
            get_cmd=(
                lambda: self.attenuator.instrument.setting_normalized),
            set_cmd=(
                lambda: self.attenuator.instrument.setting_normalized),
            docstring=('Attenuator setting (normalized)'
                       'Min value: 0.0'
                       'Max value: 1.0'))

        self.add_parameter(
            'phase_shifter_setting',
            label='Phase Shifter Setting',
            unit='',
            get_cmd=(
                lambda: self.phase_shifter.instrument.setting_normalized),
            set_cmd=(
                lambda: self.phase_shifter.instrument.setting_normalized),
            docstring=('Phase shifter setting (normalized)'
                       'Min value: 0.0'
                       'Max value: 1.0'))

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {'driver': str(self.__class__), 'name': self.name}