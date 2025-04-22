from qcodes.instrument.base import Instrument as QcodesInstrument
from qcodes.instrument.channel import InstrumentModule as QcodesInstrumentModule
import weakref
from abc import ABC


class FurtherInstrumentsDictMixIn:
    _further_instruments = weakref.WeakValueDictionary()


class PycqedInstrumentMixin(ABC):
    """
    Mixin to be used for both QCodes-based Instrument and InstrumentModule
    """

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {'driver': self.__class__.__name__, 'name': self.name}

    def get(self, param_name, *args, cache=False):
        """Shortcut for getting a parameter from its name or a default value.

        Extends the super method to allow specifying a default value as
        second argument, which is returned if the parameter does not exist.

        Args:
            param_name (str): The name of a parameter of this instrument.
            *args: accepts a single unnamed argument, which, if provided, is
                used as default value if the parameter does not exist.
            cache (bool): if True, enforces that the value is retrieved
                from the cache (default: False).

        Returns:
            The current value of the parameter.

        Examples:
        >>> # returns 'default_value'
        >>> instr.get('nonexistent_parameter', 'default_value')
        >>> # raises a KeyError
        >>> instr.get('nonexistent_parameter')
        """
        if len(args) > 1:
            raise ValueError(f'{self.name}.get accepts 1 or 2 arguments, but '
                             f'{len(args) + 1} were provided.')
        if param_name not in self.parameters and len(args) == 1:
            return args[0] # interpret second argument as default value
        elif cache:
            return self.parameters[param_name].cache.get()
        else:
            # qcodes 0.49 deprecated self.get()/self.set() for params;
            # use the form below instead
            return self.parameters[param_name].get()

    def set(self, param_name, value):
        """Shortcut for setting a parameter from its name.


        Args:
            param_name: The name of a parameter of this instrument.
            value: The value to set.
        """
        # qcodes 0.49 deprecated self.get()/self.set() for params;
        # use the form below instead
        self.parameters[param_name].set(value)


class Instrument(PycqedInstrumentMixin, QcodesInstrument,
                 FurtherInstrumentsDictMixIn):
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


class InstrumentModule(PycqedInstrumentMixin, QcodesInstrumentModule):
    """
    Extends QcodesInstrumentModule to use the compatibility fixes in
    PycqedInstrumentMixin

    FIXME: this is currently only used in
     measurement.waveform_control.reset_schemes, is this needed?
    """
    pass


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

