import types
from qcodes.instrument.base import Instrument
from pycqed.measurement.waveform_control.pulsar import Pulsar
from copy import deepcopy


class CallableValue():
    """
    Helper Class to emulate the syntax of getting the values of QCoDeS
    parameters.
    """
    def __init__(self, val):
        self.value = val

    def __call__(self):
        return self.value


class InstrumentShadow():
    """
    Pickle-able class containing the parameters of a QCoDeS instrument
    object as attributes. Class is used to enable Qt GUI applications to run
    in a separate python process.
    The QApplication (managing the control flow of the GUI application)
    blocks the execution of other tasks in the Python process, in which the
    application was instantiated, until the last GUI window is closed.
    Hence, users cannot use PycQED as long as there's an open GUI window. To
    prevent the blocking, the application can be started in a new process.
    However, the GUI application (e.g. when trying to spawn a WaveformViewer
    GUI window) may require certain QCoDeS instrument objects as resources.
    But as QCoDeS instruments are not pickle-able, they cannot simply be
    copied to the new process. Hence, the QCoDes instrument objects are
    replaced by pickle-able dummy objects of this class prior to starting
    the GUI application in a new process.
    """
    def __init__(self, instr):
        """
        Copies all parameters of the instr Instrument and adds them as a
        CallableValue to the parameters dict and as an object attribute
        Args:
            instr (Instrument): QCoDeS instrument of which the parameters
                are copied
        """
        self.snapshot = instr.snapshot()['parameters']
        setattr(self, 'parameters', dict())
        for k, v in self.snapshot.items():
            # CallableValue emulates the syntax of getting the value of a
            # QCoDeS parameter. Cannot use a lambda function, as lambda
            # functions are not pickle-able when spawning a child process.
            self.parameters[k] = CallableValue(v['value'])
            setattr(self, k, self.parameters[k])

    def get(self, k):
        return self.parameters[k]()


class PulsarShadow(InstrumentShadow):
    """
    Pickle-able dummy class containing the parameters, methods and
    attributes needed for using the plot method of a QuantumExperiment
    object, where the Pulsar object has been replaced by a PulsarShadow
    object (see docstring of parent class).
    """
    def __init__(self, instr):
        """
        Copies the parameters and the relevant attributes and methods of
        instr to the PulsarShadow object.
        Args:
            instr (Pulsar): Pulsar object of which the parameters, methods
                and attributes are copied
        """
        super().__init__(instr)
        self.channels = deepcopy(instr.channels)
        self.awgs = deepcopy(instr.awgs)

        # for k in ['find_awg_channels']:
        #     setattr(self, k, types.MethodType(getattr(instr.__class__, k), self))

        # for clock:
        instr.awgs_prequeried = True
        self._clocks = instr._clocks

    def clock(self, channel=None, awg=None):
        if channel is not None:
            awg = self.get('{}_awg'.format(channel))
        return self._clocks[awg]

    def find_awg_channels(self, awg):
        channel_list = []
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) == awg:
                channel_list.append(channel)

        return channel_list

