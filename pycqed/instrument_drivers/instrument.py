from qcodes.instrument.base import Instrument as QcodesInstrument


class Instrument(QcodesInstrument):
    """
    Class for all QCodes instruments.
    """

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {'driver': str(self.__class__), 'name': self.name}

    def get(self, param_name, *args):
        """Shortcut for getting a parameter from its name or a default value.

        Extends the super method to allow specifying a default value as
        second argument, which is returned if the parameter does not exist.

        Args:
            param_name: The name of a parameter of this instrument.
            *args: accepts a single unnamed argument, which, if provided, is
                used as default value if the parameter does not exist.

        Returns:
            The current value of the parameter.

        Examples:
        >>> # returns 'default_value'
        >>> instr.get('nonexistent_parameter', 'default_value')
        >>> # raises a KeyError
        >>> instr.get('nonexistent_parameter')
        """
        if len(args) > 1:
            raise ValueError(f'Pulsar.get accepts 1 or 2 arguments, but '
                             f'{len(args) + 1} were provided.')
        if param_name not in self.parameters and len(args) == 1:
            return args[0]  # interpret second argument as default value
        return super().get(param_name)
