from qcodes.instrument.base import Instrument as QcodesInstrument
import weakref

class FurtherInstrumentsDictMixIn:
    _further_instruments = weakref.WeakValueDictionary()


class Instrument(QcodesInstrument, FurtherInstrumentsDictMixIn):
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

    @classmethod
    def find_instrument(cls, name, instrument_class=None):
        # This overrides the super method to allow normal qcodes instruments
        # and other kinds of instruments inheriting from
        # FurtherInstrumentsDictMixIn (e.g., remote instruments) to find each
        # other. There is no docstring here since the docstring of the super
        # method remains valid.
        try:
            # First try to find it among the qcodes instruments.
            return super().find_instrument(
                name, instrument_class=instrument_class)
        except KeyError:
            # Try to find it in the dict of further instruments.
            if name not in cls._further_instruments:
                raise KeyError(f"Instrument with name {name} does not exist")
            # By default, allow qcodes instruments and objects from classes
            # that include the FurtherInstrumentsDictMixIn.
            internal_instrument_class = instrument_class or (
                QcodesInstrument, FurtherInstrumentsDictMixIn)
            ins = cls._further_instruments[name]
            if not isinstance(ins, internal_instrument_class):
                raise TypeError(
                    f"Instrument {name} is {type(ins)} but "
                    f"{internal_instrument_class} was requested"
                )
            return ins


class DummyVisaHandle:
    """Dummy handle for virtual visa instruments to avoid crash in snapshot
    """
    class DummyTimeOut:
        def get(self):
            return None
    read_termination = None
    write_termination = None
    timeout = DummyTimeOut()

    def close(self):
        pass

