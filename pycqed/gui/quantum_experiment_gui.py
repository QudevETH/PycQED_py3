import sys
import matplotlib

matplotlib.use("Qt5Agg")

# TODO: simplify imports
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox
from pycqed.gui.qt_widgets.qt_field_widgets import QLineEditDouble, \
    QLineEditInt, QLineEditInitStateSelection
from pycqed.measurement.calibration \
    import single_qubit_gates, two_qubit_gates, calibration_points
from pycqed.measurement import quantum_experiment
import numpy as np
from pycqed.gui.waveform_viewer import add_label_to_widget
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.sweep_points import SweepPoints
import logging
log = logging.getLogger(__name__)
import io

from enum import Enum

# TODO: remove importlib
import importlib
importlib.reload(single_qubit_gates)
importlib.reload(two_qubit_gates)
importlib.reload(calibration_points)
importlib.reload(quantum_experiment)


def convert_string_to_number_if_possible(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def convert_field_value_to_string(value):
    return str(value) if value is not None else ""


def connect_widget_and_checkbox(widget, checkbox):
    if checkbox.isChecked():
        widget.setEnabled(False)
    else:
        widget.setEnabled(True)


class ExperimentTypes(Enum):
    RABI = single_qubit_gates.Rabi
    RAMSEY = single_qubit_gates.Ramsey
    CHEVRON = two_qubit_gates.Chevron


class SweepPointsValueSelectionTypes(Enum):
    LINSPACE = "np.linspace"
    ARANGE = "np.arange"
    LIST = "list"


def get_members_by_experiment_class_name(experiment_class_name):
    return [ExperimentType for ExperimentType in ExperimentTypes 
            if ExperimentType.value.__name__ == experiment_class_name]


def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
        elif child.layout():
            clear_layout(child.layout())


def clear_QFormLayout(QFormLayout):
    while QFormLayout.rowCount():
        QFormLayout.removeRow(0)


class QuantumExperimentGUI:

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
            **kwargs:
        """
        QuantumExperimentGUI._instance = self
        self.device = device
        self.quantum_experiment = None
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        w = QuantumExperimentGUIMainWindow(device, **kwargs)
        w.setStyleSheet("""
        QPushButton {
            min-width:75px;
            max-width:75px;
            min-height:20px;
            border:1px solid black;
            border-radius:5px;
        }
        QGroupBox {
            font-weight: bold;
        }
        """)
        w.showMaximized()
        app.exec_()

    @staticmethod
    def get_instance():
        return QuantumExperimentGUI._instance


class QuantumExperimentGUIMainWindow(QtWidgets.QMainWindow):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Quantum Experiment GUI")
        self.device = device
        self.experiment = None

        self.boldfont = QtGui.QFont()
        self.boldfont.setBold(True)

        self.scroll = QtWidgets.QScrollArea()
        self.setCentralWidget(self.scroll)
        self.mainWidget = QtWidgets.QWidget()
        self.scroll.setLayout(QtWidgets.QVBoxLayout())
        self.scroll.layout().addWidget(self.mainWidget)
        self.mainWidget.setLayout(QtWidgets.QVBoxLayout())

        self.cbox_experiment_options = QtWidgets.QComboBox()
        self.cbox_experiment_options.addItems(
            [member.value.__name__ for member in ExperimentTypes])
        self.cbox_experiment_options.setCurrentIndex(-1)

        self.selectbox_qubits = MultiQubitSelectionWidget(
            [qb.name for qb in self.device.get_qubits()])

        self.add_task_form_button = QtWidgets.QPushButton("&Add")
        self.add_task_form_button.setEnabled(False)
        self.add_task_form_button.clicked.connect(self.add_task_form)

        self.experiment_widget_container = QtWidgets.QFormLayout()
        self.general_options_field_container = QtWidgets.QFormLayout()
        self.general_options_field_container.addRow(
            "Choose Experiment: ", self.cbox_experiment_options)
        self.general_options_field_container.addRow(
            "Choose Qubits: ", self.selectbox_qubits)
        self.general_options_field_container.addRow(
            "Add Task Form: ", self.add_task_form_button)

        self.tasks_configuration_container = QtWidgets.QGroupBox(
            "Configure Tasks")
        self.tasks_configuration_container.setLayout(QtWidgets.QHBoxLayout())
        self.tasks_configuration_container.hide()

        self.run_experiment_pushbutton = QtWidgets.QPushButton(
            "&Run Experiment")
        self.run_experiment_pushbutton.clicked.connect(self.run_experiment)
        self.run_experiment_pushbutton.hide()

        self.main_vbox_elements = [
            self.general_options_field_container,
            self.experiment_widget_container,
        ]
        [self.mainWidget.layout().addLayout(element) 
         for element in self.main_vbox_elements]

        self.mainWidget.layout().insertWidget(
            1, self.tasks_configuration_container)
        self.mainWidget.layout().addWidget(self.run_experiment_pushbutton)

        self.cbox_experiment_options.currentIndexChanged.connect(
            self.handle_experiment_choice)
        print("11")

    def handle_experiment_choice(self):
        self.add_task_form_button.setEnabled(True)
        clear_QFormLayout(self.experiment_widget_container)
        clear_layout(self.tasks_configuration_container.layout())
        self.tasks_configuration_container.hide()
        self.add_experiment_fields()
        # TODO: dynamic sizing of window does not yet work
        self.mainWidget.adjustSize()
        self.scroll.adjustSize()

    def add_experiment_fields(self):
        input_field_dict = self.get_selected_experiment().gui_kwargs()["kwargs"]
        for class_name, kwarg_dict in reversed(input_field_dict.items()):
            class_name_label = QtWidgets.QLabel(f"{class_name} Options")
            class_name_label.setFont(self.boldfont)
            self.experiment_widget_container.addRow(class_name_label)
            for kwarg, field_information in kwarg_dict.items():
                self.add_widget_to_experiment_section(kwarg, field_information)
        self.run_experiment_pushbutton.show()

    def add_widget_to_experiment_section(self, kwarg, field_information):
        widget = self.create_field_from_field_information(field_information)
        if widget is None:
            return
        self.experiment_widget_container.addRow(kwarg, widget)

    def add_task_form(self):
        self.tasks_configuration_container.show()
        experiment_kwargs = self.get_selected_experiment().gui_kwargs()
        self.tasks_configuration_container.layout().addWidget(
            TaskForm(parent=self, experiment_kwargs=experiment_kwargs))

    def get_selected_experiment(self):
        experiment_name = self.cbox_experiment_options.currentText()
        experiment_list = get_members_by_experiment_class_name(experiment_name)
        if experiment_list:
            return experiment_list[0].value
        else:
            return None

    def create_field_from_field_information(self, field_information):
        """

        Args:
            field_information: tuple holding information about the field that
                should be created.

                First entry must specify the field type, which must be one
                of the following:
                str -> QLineEdit;
                bool -> QCheckBox
                int -> QLineEditInt (only allows integer number input)
                float -> QLineEditDouble (only allows floating number input)
                CircuitBuilder.STD_INIT -> QLineEditInitStateSelection (only
                allows characters that represent qubit states such as g, 1, +)
                (Qubit, "multi_select") -> MultiQubitSelectionWidget (allows
                selection of multiple qubits associated to the device object)
                (Qubit, "single_select") -> SingleQubitSelectionWidget (allows
                selection of a single qubit associated to the device object)
                instance of list -> QComboBox (creates a single selection
                dropdown menu) composed of the elements from the list,
                returns the displayed value when experiment is run
                instance of dict -> QComboBox (creates a single selection
                dropdown list that displays the keys of the dictionary and
                returns the values of the dictionary when experiment is run)
                SweepPoints -> SweepPointsWidget (creates a button that
                spawns a dedicated SweepPointsForm dialog when clicked,
                which allows configuration of sweep points)

                Second entry specifies the default value of the field, i.e. the
                value that is initially displayed in the field. If None,
                no value is displayed by default.

        Returns:
            An instance of the specified widget type
        """
        experiment_kwargs = self.get_selected_experiment().gui_kwargs()
        if field_information[0] is str:
            widget = QtWidgets.QLineEdit()
            if field_information[1] is not None:
                widget.setText(field_information[1])
        elif field_information[0] is bool:
            widget = QtWidgets.QCheckBox()
            if field_information[1] is not None:
                widget.setChecked(field_information[1])
        elif field_information[0] is int:
            widget = QLineEditInt()
            if field_information[1] is not None:
                widget.insert(str(field_information[1]))
        elif field_information[0] is float:
            widget = QLineEditDouble()
            if field_information[1] is not None:
                widget.insert(str(field_information[1]))
        elif field_information[0] == CircuitBuilder.STD_INIT:
            widget = QLineEditInitStateSelection()
            if field_information[1] is not None:
                widget.insert(str(field_information[1]))
        elif field_information[0] == (Qubit, "multi_select"):
            widget = MultiQubitSelectionWidget(
                [qb.name for qb in self.device.get_qubits()])
            # TODO: think about implementing default value (same for
            #  SweepPointsWidget)
        elif field_information[0] == (Qubit, "single_select"):
            widget = SingleQubitSelectionWidget(
                [qb.name for qb in self.device.get_qubits()])
            if field_information[1] is not None:
                widget.setCurrentText(field_information[1])
        elif isinstance(field_information[0], list):
            widget = QtWidgets.QComboBox()
            widget.addItems(field_information[0])
            if field_information[1] is not None:
                widget.setCurrentText(field_information[1])
            else:
                widget.setCurrentIndex(-1)
        elif isinstance(field_information[0], dict):
            widget = QtWidgets.QComboBox()
            for display_value, value in field_information[0].items():
                widget.addItem(display_value, value)
            if field_information[1] is not None:
                widget.setCurrentText(field_information[1])
        elif field_information[0] is SweepPoints:
            widget = SweepPointsWidget(
                sweeping_parameters=experiment_kwargs["sweeping_parameters"],
                parent=self)
        else:
            log.warning(
                f"could not create field for keyword {field_information[0]}")
            return None
        return widget

    def get_argument_value_from_widget(self, widget, kwarg):
        if isinstance(widget, QLineEditInitStateSelection):
            return widget.text() if widget.hasAcceptableInput() else None
        elif isinstance(widget, GlobalChoiceOptionWidget):
            if widget.global_choice_checkbox.isChecked():
                return None
            else:
                return self.get_argument_value_from_widget(widget.widget, kwarg)
        elif isinstance(widget, SingleQubitSelectionWidget):
            return widget.get_selected_qubit_from_device(self.device)
        elif isinstance(widget, MultiQubitSelectionWidget):
            return widget.get_selected_qubits_from_device(self.device)
        elif isinstance(widget, CheckableComboBox):
            return widget.currentData()
        elif isinstance(widget, QLineEditInt):
            return int(widget.text()) if widget.text() != "" else None
        elif isinstance(widget, QLineEditDouble):
            return float(widget.text()) if widget.text() != "" else None
        elif isinstance(widget, QtWidgets.QLineEdit):
            return widget.text() if widget.text() != "" else None
        elif isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText() if widget.currentText() != "" else None
        elif isinstance(widget, SweepPointsWidget):
            if widget.chosen_sweep_points_kwargs is None:
                return None
            else:
                return SweepPoints(
                    **widget.chosen_sweep_points_kwargs["kwargs"])
        else:
            log.warning(
                f"could not get argument value of field {kwarg} with widget "
                f"of type {type(widget)}")
            return None

    def run_experiment(self):
        QtWidgets.QApplication.instance().setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.WaitCursor))
        experiment = self.get_selected_experiment()
        experiment_settings_kwargs = self.get_QFormLayout_settings(
            self.experiment_widget_container)
        qubits = self.selectbox_qubits.get_selected_qubits_from_device(
            self.device)
        task_list = self.get_task_list()
        # out = io.StringIO()
        # sys.stdout = out
        # self.experiment_output.setText(out.getvalue())
        # self.experiment_output.show()
        print("task list:\n", task_list)
        print("experiment settings:\n", experiment_settings_kwargs)
        # TODO: add qubits keyword
        self.experiment = experiment(task_list=task_list,
                                     qubits=qubits,
                                     dev=self.device,
                                     **experiment_settings_kwargs)
        QtWidgets.QApplication.instance().restoreOverrideCursor()
        if self.experiment.exception is not None:
            # TODO: notify user
            message_box = QtWidgets.QMessageBox(parent=self)
            message_box.setText("Unable to run experiment")
            message_box.setDetailedText("Check the output of your python "
                                        "console to read the error message")
            message_box.setWindowTitle("Error")
            message_box.setDefaultButton(message_box.Ok)
            message_box.setIcon(message_box.Icon.Warning)
            message_box.exec_()
        # sys.stdout = sys.__stdout__

    def get_QFormLayout_settings(self, QFormLayoutInstance):
        settings_dict = {}
        kwarg_widget_pairs = [(
            QFormLayoutInstance.itemAt(i, QtWidgets.QFormLayout.LabelRole).widget().text(),
            QFormLayoutInstance.itemAt(i, QtWidgets.QFormLayout.FieldRole).widget())
            for i in range(QFormLayoutInstance.rowCount())
            if QFormLayoutInstance.itemAt(i, QtWidgets.QFormLayout.LabelRole) is not None
            and QFormLayoutInstance.itemAt(i, QtWidgets.QFormLayout.FieldRole) is not None]
        for kwarg, widget in kwarg_widget_pairs:
            value = self.get_argument_value_from_widget(widget, kwarg)
            print(value)
            if value is None:
                continue
            settings_dict.update({kwarg: value})
        return settings_dict

    def get_task_list(self):
        task_list = []
        for task_form in self.tasks_configuration_container.findChildren(
                TaskForm):
            task_list.append(self.get_QFormLayout_settings(task_form.layout()))
        return task_list

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_W \
                and event.modifiers() == QtCore.Qt.ControlModifier:
            self.close()
            event.accept()

    def closeEvent(self, event):
        quantum_experiment_gui = QuantumExperimentGUI.get_instance()
        quantum_experiment_gui.experiment = self.experiment
        event.accept()


class SweepPointsForm(QtWidgets.QDialog):
    def __init__(self, sweep_parameters, parent=None, previous_choices=None):
        super().__init__(parent=parent)
        self.previous_choices = previous_choices
        self.setWindowTitle("Configure Sweep Points")
        self.resize(300, 200)
        self.setSizeGripEnabled(True)
        self.sweep_points_dimensions = [0, 1]
        self.sweep_parameters = {}
        for dimension in self.sweep_points_dimensions:
            self.sweep_parameters.update({dimension: {}})
        for class_name, sweep_parameter_dict in sweep_parameters.items():
            for dimension in self.sweep_points_dimensions:
                self.sweep_parameters[dimension].update(
                    sweep_parameter_dict[dimension])
        accept_cancel_buttons = \
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(accept_cancel_buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        message = QtWidgets.QLabel(
            "Choose the sweep points for the selected dimension")
        self.layout.addWidget(message)

        self.dimension_combobox = QtWidgets.QComboBox()
        self.dimension_combobox.addItems(
            [str(dimension) for dimension in self.sweep_points_dimensions])
        self.layout.addLayout(
            add_label_to_widget(self.dimension_combobox, "Dimension: "))

        self.parameter_name_cbox = QtWidgets.QComboBox()
        self.parameter_name_cbox.setEditable(True)
        self.parameter_name_cbox.setInsertPolicy(
            self.parameter_name_cbox.NoInsert)
        self.layout.addLayout(
            add_label_to_widget(self.parameter_name_cbox, "Parameter Name: "))

        self.unit_lineedit = QtWidgets.QLineEdit()
        self.layout.addLayout(
            add_label_to_widget(self.unit_lineedit, "Base unit: "))

        self.label_lineedit = QtWidgets.QLineEdit()
        self.layout.addLayout(
            add_label_to_widget(self.label_lineedit, "Label: "))

        self.values_selection_type_cbox = QtWidgets.QComboBox()
        self.values_selection_type_cbox.addItems(
            [SweepPointsValueSelectionType.value for
             SweepPointsValueSelectionType in
             SweepPointsValueSelectionTypes])
        self.values_selection_type_cbox.setCurrentIndex(0)
        self.layout.addLayout(
            add_label_to_widget(self.values_selection_type_cbox,
                                "Values Selection Type: "))

        self.values_layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.values_layout)

        self.setup_dialog()

        self.dimension_combobox.currentIndexChanged.connect(
            lambda: self.set_parameter_list(initialization=False))
        self.parameter_name_cbox.currentIndexChanged.connect(
            lambda: self.set_standard_unit(initialization=False))
        self.values_selection_type_cbox.currentIndexChanged.connect(
            lambda: self.configure_values_selection_field(
                set_previous_choices=False))

        self.layout.addWidget(self.buttonBox)

    def setup_dialog(self):
        if self.previous_choices is not None:
            self.dimension_combobox.setCurrentText(
                str(self.previous_choices["kwargs"]["dimension"]))
            if "label" in self.previous_choices["kwargs"].keys():
                self.label_lineedit.setText(
                    self.previous_choices["kwargs"]["label"])
            self.configure_values_selection_field(set_previous_choices=True)
        else:
            self.configure_values_selection_field(set_previous_choices=False)
        self.set_parameter_list(initialization=True)

    def set_parameter_list(self, initialization=False):
        self.parameter_name_cbox.blockSignals(True)
        dimension = int(self.dimension_combobox.currentText())
        self.parameter_name_cbox.clear()
        self.parameter_name_cbox.addItems(
            list(self.sweep_parameters[dimension].keys()))
        if self.previous_choices is not None and initialization:
            self.parameter_name_cbox.setCurrentText(
                self.previous_choices["kwargs"]["param"])
        else:
            self.parameter_name_cbox.setCurrentIndex(0)
        self.parameter_name_cbox.blockSignals(False)
        self.set_standard_unit(initialization=initialization)

    def set_standard_unit(self, initialization=False):
        if self.previous_choices is not None and initialization:
            self.unit_lineedit.setText(self.previous_choices["kwargs"]["unit"])
        else:
            selected_sweep_parameter = self.parameter_name_cbox.currentText()
            current_dimension = int(self.dimension_combobox.currentText())
            if selected_sweep_parameter in [
                self.parameter_name_cbox.itemText(i) for i in range(
                    self.parameter_name_cbox.count())]:
                self.unit_lineedit.setText(
                    self.sweep_parameters[current_dimension][selected_sweep_parameter])

    def configure_values_selection_field(self, set_previous_choices=False):
        clear_layout(self.values_layout)
        if not set_previous_choices:
            values_selection_type = self.values_selection_type_cbox.currentText()
        else:
            values_selection_type = self.previous_choices["_field_values"][
                "values_selection_type"]
            self.values_selection_type_cbox.setCurrentText(values_selection_type)

        fields = []
        if values_selection_type == SweepPointsValueSelectionTypes.LINSPACE.value:
            self.values_start_value = QLineEditDouble()
            self.values_end_value = QLineEditDouble()
            self.values_number_of_steps = QLineEditInt()
            if set_previous_choices:
                self.values_start_value.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["start_value"]))
                self.values_end_value.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["end_value"]))
                self.values_number_of_steps.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["number_of_steps"]))
            fields.extend([
                (self.values_start_value, "Start Value: "),
                (self.values_end_value, "End Value: "),
                (self.values_number_of_steps, "Step Number: ")
            ])

        elif values_selection_type == SweepPointsValueSelectionTypes.ARANGE.value:
            self.values_start_value = QLineEditDouble()
            self.values_end_value = QLineEditDouble()
            self.values_step_size = QLineEditDouble()
            if set_previous_choices:
                self.values_start_value.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["start_value"]))
                self.values_end_value.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["end_value"]))
                self.values_step_size.setText(convert_field_value_to_string(
                    self.previous_choices["_field_values"]["step_size"]))
            fields.extend([
                (self.values_start_value, "Start Value: "),
                (self.values_end_value, "End Value: "),
                (self.values_step_size, "Step Size: ")
            ])

        elif values_selection_type == SweepPointsValueSelectionTypes.LIST.value:
            self.values_list = QtWidgets.QLineEdit()
            self.values_list.setPlaceholderText(
                "separate items by comma, e.g. '1, 2, 3' or '\"foo\", \"bar\"")
            if set_previous_choices:
                self.values_list.setText(
                    self.previous_choices["_field_values"]["list_string"])
            fields.extend([
                (self.values_list, "List Items: "),
            ])

        else:
            return

        [self.values_layout.addLayout(add_label_to_widget(widget, label))
         for widget, label in fields]

    def get_sweep_points_config(self):
        sweep_points_config = {
            "kwargs": {
                "param": self.parameter_name_cbox.currentText(),
                "unit": self.unit_lineedit.text(),
                "dimension": int(self.dimension_combobox.currentText()),
            },
            "_field_values": {
                "values_selection_type":
                    self.values_selection_type_cbox.currentText()
            },
        }
        if self.label_lineedit.text() != "":
            sweep_points_config["kwargs"].update(
                {"label": self.label_lineedit.text()})

        selection_type = self.values_selection_type_cbox.currentText()
        values = None

        if selection_type == SweepPointsValueSelectionTypes.LINSPACE.value:
            start_value = float(self.values_start_value.text()) \
                if self.values_start_value.text() != "" else None
            end_value = float(self.values_end_value.text()) \
                if self.values_end_value.text() != "" else None
            number_of_steps = int(self.values_number_of_steps.text()) \
                if self.values_number_of_steps.text() != "" else None
            sweep_points_config["_field_values"].update({
                "start_value": start_value,
                "end_value": end_value,
                "number_of_steps": number_of_steps,
            })
            # np.linspace requires start_value and end_value as argument,
            # num is 50 by default
            if not (start_value is None or end_value is None):
                if number_of_steps is not None:
                    values = np.linspace(
                        start_value, end_value, num=number_of_steps)
                else:
                    values = np.linspace(start_value, end_value)

        elif selection_type == SweepPointsValueSelectionTypes.ARANGE.value:
            start_value = float(self.values_start_value.text()) \
                if self.values_start_value.text() != "" else None
            end_value = float(self.values_end_value.text()) \
                if self.values_end_value.text() != "" else None
            step_size = float(self.values_step_size.text()) \
                if self.values_step_size.text() != "" else None
            sweep_points_config["_field_values"].update({
                "start_value": start_value,
                "end_value": end_value,
                "step_size": step_size,
            })
            # np.arange only requires end_value as argument, default for
            # start value is 0 and default for step is 1
            if end_value is not None:
                if start_value is not None:
                    values = np.arange(start_value, end_value, step=step_size)
                else:
                    values = np.arange(end_value, step=step_size)

        elif selection_type == SweepPointsValueSelectionTypes.LIST.value:
            sweep_points_config["_field_values"].update({
                "list_string": self.values_list.text(),
            })
            if self.values_list.text() != "":
                values = [convert_string_to_number_if_possible(value)
                          for value in self.values_list.text().split(",")]
        sweep_points_config["kwargs"].update({"values": values})
        return sweep_points_config


class TaskForm(QtWidgets.QWidget):
    def __init__(self, parent, experiment_kwargs):
        super(TaskForm, self).__init__(parent)
        self.parent = parent
        self.experiment_kwargs = experiment_kwargs
        self.experiment_name = self.parent.cbox_experiment_options.currentText()
        self.setLayout(QtWidgets.QFormLayout())
        self.layout().setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.setStyleSheet("""
        background-color:rgb(239, 230, 178 );
        """)
        for class_name, kwarg_dict in self.experiment_kwargs["task_list_fields"].items():
            for kwarg, field_information in kwarg_dict.items():
                self.add_task_widget(kwarg, field_information)

        delete_task_button = QtWidgets.QPushButton("Delete Task")
        delete_task_button.clicked.connect(self.delete_task_form)
        self.layout().addRow("", delete_task_button)

    def add_task_widget(self, kwarg, field_information):
        widget = self.parent.create_field_from_field_information(
            field_information)
        if widget is None:
            return
        if kwarg in self.parent.get_selected_experiment().kw_for_task_keys:
            global_option_widget = GlobalChoiceOptionWidget(widget)
            self.layout().addRow(kwarg, global_option_widget)
        else:
            self.layout().addRow(kwarg, widget)

    def delete_task_form(self):
        if sum([isinstance(widget, TaskForm) for widget in
                self.parent.tasks_configuration_container.children()]) == 1:
            self.parent.tasks_configuration_container.hide()
        self.deleteLater()


# TODO: move to qt_field_widgets.py
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


class SweepPointsWidget(QtWidgets.QWidget):
    def __init__(self, sweeping_parameters, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.sweeping_parameters = sweeping_parameters
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.configure_sweep_points_button = QtWidgets.QPushButton("Configure")
        self.chosen_sweep_points_kwargs = None
        self.configure_sweep_points_button.clicked.connect(
            self.spawn_sweep_points_dialog)
        self.stored_sweep_parameter = QtWidgets.QLabel("")
        self.layout.addWidget(self.configure_sweep_points_button,
                              QtCore.Qt.AlignLeft)
        self.layout.addWidget(self.stored_sweep_parameter, QtCore.Qt.AlignLeft)
        self.setStyleSheet("""
        background-color:rgb(239, 189, 178);
        """)

    def spawn_sweep_points_dialog(self):
        sweep_points_configuration_dialog = SweepPointsForm(
            self.sweeping_parameters,
            parent=self.parent,
            previous_choices=self.chosen_sweep_points_kwargs
        )
        clicked_button = sweep_points_configuration_dialog.exec_()
        if clicked_button == 1:
            self.chosen_sweep_points_kwargs = \
                sweep_points_configuration_dialog.get_sweep_points_config()
            print(
                "saved the configuration: \n",
                repr(sweep_points_configuration_dialog.get_sweep_points_config()["kwargs"])
            )
            self.stored_sweep_parameter.setText(
                f"Parameter: "
                f"{self.chosen_sweep_points_kwargs['kwargs']['param']}")
        else:
            print("Configuration was not saved (click ok to save)")
        sweep_points_configuration_dialog.deleteLater()
