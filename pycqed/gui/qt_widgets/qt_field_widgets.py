from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder


class QLineEditInt(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(QtGui.QIntValidator())


class QLineEditDouble(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(QtGui.QDoubleValidator())


class QLineEditInitStateSelection(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        allowed_states = list(CircuitBuilder.STD_INIT.keys())
        # expects two characters of allowed_states. Inside character range []
        # all characters are interpreted as literals, except the dash,
        # hence it has to be escaped
        regex = QtCore.QRegularExpression('[%s]{2}' % ''.join(
            allowed_states).replace('-', '\-'))
        validator = QtGui.QRegularExpressionValidator(regex)
        self.setPlaceholderText('e.g. "1+"')
        self.setToolTip(f"two chars from {allowed_states}, first should be "
                        "control qubit state and second should be target "
                        "qubit state")
        self.setValidator(validator)

    def input_is_acceptable(self):
        return self.validator().validate(self.text(), 0)[0] == \
               QtGui.QValidator.State.Acceptable
