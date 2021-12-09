from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox


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

class GlobalChoiceOptionWidget(QtWidgets.QWidget):
    def __init__(self, widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(QtWidgets.QHBoxLayout())
        self.widget = widget
        self.global_choice_checkbox = QtWidgets.QCheckBox("use global choice")
        self.global_choice_checkbox.stateChanged.connect(
            lambda: connect_widget_and_checkbox(widget,
                                                self.global_choice_checkbox))
        self.global_choice_checkbox.setChecked(True)
        self.layout().addWidget(widget)
        self.layout().addWidget(self.global_choice_checkbox)


class SingleQubitSelectionWidget(QtWidgets.QComboBox):
    """
    Qt form widget for single Qubit selection

    """
    def __init__(self, qubit_list, *args, **kwargs):
        """
        Initializes the qubit selection form

        Args:
            qubit_list (list): list of qubit names that should be dispalyed
                in the dropdown menu of the widget
            *args: passed to the QComboBox init
            **kwargs: passed to the QComboBox init
        """
        super().__init__(*args, **kwargs)
        self.addItems(qubit_list)

    def get_selected_qubit_from_device(self, device):
        """
        Returns the qubit object corresponding to the currently selected qubit
        name.

        Args:
            device (Device): Device object for which the qubit object with
                the selected name should be returned

        Returns:
            Qubit object associated to device and with the currently
            selected name.
        """
        return getattr(device, self.currentText())


class MultiQubitSelectionWidget(CheckableComboBox):
    def __init__(self, qubit_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_display_text = "Select..."
        self.addItems(qubit_list)

    def get_selected_qubits_from_device(self, device):
        return device.get_qubits(
            [qubitname for qubitname in self.currentData()]) if \
            self.currentData() != [] else None


class ScrollLabelFixedLineHeight(QtWidgets.QScrollArea):
    def __init__(self, number_of_lines=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        content = QtWidgets.QWidget(self)
        self.setWidget(content)
        lay = QtWidgets.QVBoxLayout(content)
        self.label = QtWidgets.QLabel(content)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        font_metrics = QtGui.QFontMetrics(self.font())
        # TODO: find better solution than adding a small number of pixels
        #  manually
        self.setMaximumHeight((font_metrics.lineSpacing()+3)*number_of_lines)
        lay.addWidget(self.label)

    def clear_and_set_text(self, text):
        self.label.clear()
        self.label.setText(text)


def connect_widget_and_checkbox(widget, checkbox):
    if checkbox.isChecked():
        widget.setEnabled(False)
    else:
        widget.setEnabled(True)

