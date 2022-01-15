from pycqed.gui import qt_compat as qt
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox
from pycqed.gui import gui_utilities as g_utils
import traceback
import numpy as np


class QLineEditInt(qt.QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(qt.QtGui.QIntValidator())


class QLineEditDouble(qt.QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(qt.QtGui.QDoubleValidator())


class QLineEditInitStateSelection(qt.QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        allowed_states = list(CircuitBuilder.STD_INIT.keys())
        # expects two characters of allowed_states. Inside character range []
        # all characters are interpreted as literals, except the dash,
        # hence it has to be escaped
        regex = qt.QtCore.QRegularExpression('[%s]{2}' % ''.join(
            allowed_states).replace('-', '\-'))
        validator = qt.QtGui.QRegularExpressionValidator(regex)
        self.setPlaceholderText('e.g. "1+"')
        self.setToolTip(f"two chars from {allowed_states}, first should be "
                        "control qubit state and second should be target "
                        "qubit state")
        self.setValidator(validator)

    def input_is_acceptable(self):
        return self.validator().validate(self.text(), 0)[0] == \
               qt.QtGui.QValidator.State.Acceptable

class GlobalChoiceOptionWidget(qt.QtWidgets.QWidget):
    def __init__(self, widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(qt.QtWidgets.QHBoxLayout())
        self.widget = widget
        self.global_choice_checkbox = qt.QtWidgets.QCheckBox("use global choice")
        self.global_choice_checkbox.stateChanged.connect(
            lambda: connect_widget_and_checkbox(widget,
                                                self.global_choice_checkbox))
        self.global_choice_checkbox.setChecked(True)
        self.layout().addWidget(widget)
        self.layout().addWidget(self.global_choice_checkbox)


class SingleQubitSelectionWidget(qt.QtWidgets.QComboBox):
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


class ScrollLabelFixedLineHeight(qt.QtWidgets.QScrollArea):
    def __init__(self, number_of_lines=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        content = qt.QtWidgets.QWidget(self)
        self.setWidget(content)
        lay = qt.QtWidgets.QVBoxLayout(content)
        self.label = qt.QtWidgets.QLabel(content)
        self.label.setAlignment(qt.QtCore.Qt.AlignmentFlag.AlignLeft |
                                qt.QtCore.Qt.AlignmentFlag.AlignTop)
        self.label.setWordWrap(True)
        font_metrics = qt.QtGui.QFontMetrics(self.font())
        # TODO: find better solution than adding a small number of pixels
        #  manually
        self.setMaximumHeight((font_metrics.lineSpacing()+3)*number_of_lines)
        lay.addWidget(self.label)
        self.label.setTextInteractionFlags(
            qt.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setCursor(qt.QtGui.QCursor(
            qt.QtCore.Qt.CursorShape.IBeamCursor))
        self.setStyleSheet("""
        background-color: white;
        """)

    def clear_and_set_text(self, text):
        self.label.clear()
        self.label.setText(text)


class ConfigureDialogWidget(qt.QtWidgets.QWidget):
    def __init__(self, gui, dialog_widget, on_ok_method, cancel_message,
                 on_cancel_reset_method=None, on_open_dialog_method=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_message = cancel_message
        self.on_cancel_reset_method = on_cancel_reset_method
        self.on_ok_method = on_ok_method
        self.gui = gui
        self.on_open_dialog_method = on_open_dialog_method

        self.setLayout(qt.QtWidgets.QHBoxLayout())
        self.configure_button = qt.QtWidgets.QPushButton('Configure')
        self.configure_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.configure_button.clicked.connect(self.show_config_dialog)
        self.layout().addWidget(self.configure_button)
        self.dialog_widget = dialog_widget(parent=self, gui=gui)

    def show_config_dialog(self):
        if self.on_open_dialog_method is not None:
            getattr(self.dialog_widget, self.on_open_dialog_method)()
        clicked_button = self.dialog_widget.exec_()
        if clicked_button == 1:
            getattr(self.dialog_widget, self.on_ok_method)()
        else:
            self.gui.message_textedit.clear_and_set_text(
                self.cancel_message
            )
            if self.on_cancel_reset_method is not None:
                getattr(self.dialog_widget, self.on_cancel_reset_method)()


class SimpleDialog(qt.QtWidgets.QDialog):
    def __init__(self, parent, window_title=""):
        super().__init__(parent=parent)
        self.setSizeGripEnabled(True)
        self.setWindowTitle(window_title)
        self.setLayout(qt.QtWidgets.QVBoxLayout())
        self.configuration_layout = qt.QtWidgets.QHBoxLayout()

        accept_cancel_buttons = \
            qt.QtWidgets.QDialogButtonBox.StandardButton.Ok | \
            qt.QtWidgets.QDialogButtonBox.StandardButton.Cancel
        self.buttonBox = qt.QtWidgets.QDialogButtonBox(accept_cancel_buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addLayout(self.configuration_layout)
        self.layout().addWidget(self.buttonBox)


class AcqAvgDialog(SimpleDialog):
    def __init__(self, parent, gui):
        super().__init__(
            parent=parent, window_title="Configure Acquisition Average")
        self.gui = gui

        self.qubit_selection_cbox = MultiQubitSelectionWidget(
            [qb.name for qb in self.gui.device.get_qubits()])
        self.value_lineedit = QLineEditInt()
        self.qubit_selection_cbox.model().dataChanged.connect(
            self.show_current_value)
        self.configuration_layout.addLayout(g_utils.add_label_to_widget(
            self.qubit_selection_cbox, "For Qubits "))
        self.configuration_layout.addLayout(g_utils.add_label_to_widget(
            self.value_lineedit, "set Value"))

        self.select_all_button = qt.QtWidgets.QPushButton('Select All')
        self.select_all_button.clicked.connect(
            lambda: self.set_all_qubits(True))
        self.unselect_all_button = qt.QtWidgets.QPushButton('Unselect All')
        self.unselect_all_button.clicked.connect(
            lambda: self.set_all_qubits(False))
        self.select_choices_layout = qt.QtWidgets.QHBoxLayout()
        self.select_choices_layout.addWidget(self.select_all_button)
        self.select_choices_layout.addWidget(self.unselect_all_button)
        self.layout().insertLayout(1, self.select_choices_layout)

    def on_open_dialog(self):
        self.qubit_selection_cbox.blockSignals(True)
        self.set_all_qubits(True)
        self.qubit_selection_cbox.blockSignals(False)
        qubits = self.qubit_selection_cbox.get_selected_qubits_from_device(
            self.gui.device)
        max_set_value = max([qubit.acq_averages() for qubit in qubits])
        self.value_lineedit.setText(str(max_set_value))

    def set_all_qubits(self, state):
        check_state = qt.QtCore.Qt.CheckState.Checked if state else \
            qt.QtCore.Qt.CheckState.Unchecked
        for i in range(self.qubit_selection_cbox.model().rowCount()):
            self.qubit_selection_cbox.model().item(i).setCheckState(
                check_state)

    def show_current_value(self):
        qubits = self.qubit_selection_cbox.get_selected_qubits_from_device(
            self.gui.device)
        # qubits is either None or a list with length >= 1
        if qubits is not None and (
                len(set([qubit.acq_averages() for qubit in qubits])) == 1):
            self.value_lineedit.setText(str(qubits[0].acq_averages()))
        else:
            self.value_lineedit.setText("")

    def update_acq_avg_for_selected_qubit(self):
        qubits = self.qubit_selection_cbox.get_selected_qubits_from_device(
            self.gui.device)
        value = None
        if self.value_lineedit.text() != "":
            value = int(self.value_lineedit.text())
            [qubit.acq_averages(value)
             for qubit in qubits]
        self.display_stored_setting(qubits, value)

    def display_stored_setting(self, qubits, value):
        if value is not None:
            self.gui.message_textedit.clear_and_set_text(
                "saved the configuration:\n"
                f"acq_averages: {value} for qubits "
                f"{', '.join([qubit.name for qubit in qubits])}")
        else:
            "acq_averages configuration was not saved, no value chosen"


class PulsePeriodDialog(SimpleDialog):
    def __init__(self, parent, gui):
        super().__init__(parent=parent, window_title="Configure Pulse Period")
        self.gui = gui

        self.value_lineedit = QLineEditDouble()
        first_qubit = self.gui.device.get_qubits()[0]
        self.value_lineedit.setText(
            np.format_float_scientific(
                first_qubit.instr_trigger.get_instr().pulse_period()))
        self.configuration_layout.addLayout(g_utils.add_label_to_widget(
            self.value_lineedit, "Set Pulse Period "))
        self.configuration_layout.addWidget(qt.QtWidgets.QLabel("s"))

    def update_pulse_period(self):
        first_qubit = self.gui.device.get_qubits()[0]
        value = None
        if self.value_lineedit.text() != "":
            value = float(self.value_lineedit.text())
            try:
                first_qubit.instr_trigger.get_instr().pulse_period(value)
            except Exception as exception:
                traceback.print_exc()
                self.gui.message_textedit.clear_and_set_text(
                    "Exception occurred:\n"
                    f"{type(exception).__name__}: {str(exception)}\n")
                self.reset_value_lineedit()
                return
        else:
            self.reset_value_lineedit()
        self.display_stored_setting(value)

    def display_stored_setting(self, value):
        if value is not None:
            self.gui.message_textedit.clear_and_set_text(
                "saved the configuration:\n"
                f"pulse_period: {value} ")
        else:
            "pulse_period configuration was not saved, no value chosen"

    def reset_value_lineedit(self):
        first_qubit = self.gui.device.get_qubits()[0]
        self.value_lineedit.setText(
            np.format_float_scientific(
                first_qubit.instr_trigger.get_instr().pulse_period()))


def connect_widget_and_checkbox(widget, checkbox):
    if checkbox.isChecked():
        widget.setEnabled(False)
    else:
        widget.setEnabled(True)

