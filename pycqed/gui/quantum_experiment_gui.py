import sys
from pycqed.gui.qt_widgets.qt_field_widgets import *
from pycqed.measurement.calibration.two_qubit_gates import Chevron
from pycqed.measurement.calibration.single_qubit_gates import Rabi, Ramsey
import numpy as np
from collections import OrderedDict as odict
from pycqed.gui.waveform_viewer import add_label_to_widget
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.sweep_points import SweepPoints
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
import logging
from enum import Enum

log = logging.getLogger(__name__)


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


class ExperimentTypes(Enum):
    RABI = Rabi
    RAMSEY = Ramsey
    CHEVRON = Chevron


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
        self.device = device
        self.experiments = []
        self.spawn_gui(**kwargs)

    def spawn_gui(self, **kwargs):
        QuantumExperimentGUI._instance = self
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        w = QuantumExperimentGUIMainWindow(self.device, self.experiments,
                                           **kwargs)
        w.setStyleSheet("""
        QPushButton {
            min-width:75px;
            max-width:100px;
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



class QuantumExperimentGUIMainWindow(QtWidgets.QMainWindow):
    def __init__(self, device, experiments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Quantum Experiment GUI")
        self.device = device
        self.experiments = experiments

        self.boldfont = QtGui.QFont()
        self.boldfont.setBold(True)

        QtWidgets.QApplication.instance().restoreOverrideCursor()

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
            "Add Task: ", self.add_task_form_button)
        self.general_options_field_container.addRow(QtWidgets.QLabel("OR"))
        self.general_options_field_container.addRow(
            "Choose Qubits: ", self.selectbox_qubits)

        self.tasks_configuration_container = QtWidgets.QGroupBox(
            "Configure Tasks")
        self.tasks_configuration_container.setLayout(QtWidgets.QHBoxLayout())
        self.tasks_configuration_container.hide()

        self.run_experiment_pushbutton = QtWidgets.QPushButton(
            "&Run Experiment")
        self.run_experiment_pushbutton.clicked.connect(self.run_experiment)
        self.run_experiment_pushbutton.hide()

        self.message_textedit = ScrollLabelFixedLineHeight(number_of_lines=6)

        self.main_vbox_elements = [
            self.general_options_field_container,
            self.experiment_widget_container,
        ]
        [self.mainWidget.layout().addLayout(element) 
         for element in self.main_vbox_elements]

        self.mainWidget.layout().insertWidget(
            1, self.tasks_configuration_container)
        self.mainWidget.layout().addWidget(self.run_experiment_pushbutton)
        self.mainWidget.layout().addWidget(self.message_textedit)

        self.cbox_experiment_options.currentIndexChanged.connect(
            self.handle_experiment_choice)

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
        exp = self.get_selected_experiment()
        input_field_dict = exp.gui_kwargs()["kwargs"]
        # automatically add global fields for task fields that are listed in
        # kw_for_task_keys
        # 
        # if hierarchy of classes is out of order in gui, most likely it's
        # because there's a keyword in kw_for_task_keys that corresponds to
        # a keyword argument of a parent class of the selected experiment which
        # is not registered in exp.gui_kwargs()['kwargs']. In this case the
        # order can be restored by adding an empty dictionary to the
        # gui_kwargs()['kwargs'] dictionary in the gui_kwargs classmethod of
        # the parent class, i.e. adding the parent class name as key and an
        # empty dictionary as value to the gui_kwargs()['kwargs'] dictionary.
        task_list_fields = exp.gui_kwargs()["task_list_fields"]
        for class_name, kwarg_dict in reversed(task_list_fields.items()):
            for kwarg, field_information in kwarg_dict.items():
                if kwarg in getattr(exp, 'kw_for_task_keys', []) and \
                        kwarg not in input_field_dict.get(class_name, {}):
                    if class_name not in input_field_dict:
                        input_field_dict[class_name] = odict({})
                    input_field_dict[class_name][kwarg] = field_information
        # create the widgets
        for class_name, kwarg_dict in reversed(input_field_dict.items()):
            class_name_label = QtWidgets.QLabel(f"{class_name} Options")
            class_name_label.setFont(self.boldfont)
            if len(kwarg_dict):
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
                gui=self)
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
                sweep_points = SweepPoints()
                for sweep_info in widget.chosen_sweep_points_kwargs:
                    sweep_points.add_sweep_parameter(**sweep_info["kwargs"])
                return sweep_points
        else:
            log.warning(
                f"could not get argument value of field {kwarg} with widget "
                f"of type {type(widget)}")
            return None

    def run_experiment(self):
        experiment = self.get_selected_experiment()
        experiment_settings_kwargs = self.get_QFormLayout_settings(
            self.experiment_widget_container)
        qubits = self.selectbox_qubits.get_selected_qubits_from_device(
            self.device)
        task_list = self.get_task_list()
        if not len(task_list):
            task_list = None
        argument_string = (f"Arguments:\n"
                           f"  task list:\n"
                           f"  {task_list}\n"
                           f"  experiment settings:\n"
                           f"  {experiment_settings_kwargs}\n"
                           f"  qubits:\n"
                           f"  {qubits}")
        self.message_textedit.clear_and_set_text(
            "Running Experiment...\n"
            "\n"
            f"{argument_string}"
            )
        QtWidgets.QApplication.instance().setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.WaitCursor))
        experiment = experiment(task_list=task_list,
                                     qubits=qubits,
                                     dev=self.device,
                                     **experiment_settings_kwargs)
        QtWidgets.QApplication.instance().restoreOverrideCursor()
        if experiment.exception is not None:
            self.message_textedit.clear_and_set_text(
                "Experiment could not be performed"
                "\n"
                f"{argument_string}")
            message_box = QtWidgets.QMessageBox(parent=self)
            message_box.setText("Unable to run experiment. Check the output "
                                "of your python console to read the error "
                                "message.")
            message_box.setWindowTitle("Error")
            message_box.setDefaultButton(message_box.Ok)
            message_box.setIcon(message_box.Icon.Warning)
            message_box.exec_()

        else:
            self.message_textedit.clear_and_set_text(
                "Successfully ran the experiment!\n"
                "\n"
                f"{argument_string}")
            self.experiments.append(experiment)

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


class SweepPointsDialog(QtWidgets.QDialog):
    def __init__(self, sweep_parameters, parent, gui=None):
        super().__init__(parent=parent)
        self.sweep_parameters = sweep_parameters
        self.parent = parent
        self.gui = gui

        # self.resize(300, 200)
        self.setSizeGripEnabled(True)
        self.setWindowTitle("Configure Sweep Points")
        self.setLayout(QtWidgets.QVBoxLayout())

        message = QtWidgets.QLabel(
            "Choose the sweep points for the selected dimension")
        self.layout().addWidget(message)
        accept_cancel_buttons = \
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(accept_cancel_buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.add_sweep_points_button = QtWidgets.QPushButton('&Add')
        self.layout().addWidget(self.add_sweep_points_button)
        self.sweep_points_list = []
        self.add_sweep_points_button.clicked.connect(self.add_sweep_points)
        self.layout().addWidget(self.buttonBox)
        self.resize(self.minimumSizeHint())

    def add_sweep_points(self):
        self.sweep_points_list.append(SweepPointsForm(
            self.sweep_parameters, dialog=self, gui=self.gui)
        )
        self.layout().insertWidget(
            self.layout().count() - 1, self.sweep_points_list[-1])
        self.resize(self.minimumSizeHint())

    def get_and_store_sweep_points_config(self):
        sweep_points_configurations = []
        [sweep_points_configurations.append(
            sweep_points.get_and_store_sweep_points_config())
            for sweep_points in self.sweep_points_list]
        return sweep_points_configurations if sweep_points_configurations \
            else None

    def reset_to_saved_choices(self):
        [sweep_points.reset_to_saved_choices()
         for sweep_points in self.sweep_points_list]


class SweepPointsForm(QtWidgets.QGroupBox):
    def __init__(self, sweep_parameters, dialog, parent=None, gui=None):
        super().__init__(parent=parent)
        self.previous_choice = None
        self.dialog = dialog
        self.gui = gui
        self.kw_for_sweep_points = self.gui.get_selected_experiment().kw_for_sweep_points

        self.sweep_points_dimensions = [0, 1]
        self.sweep_parameters = {}
        for dimension in self.sweep_points_dimensions:
            self.sweep_parameters.update({dimension: {}})
        for class_name, sweep_parameter_dict in sweep_parameters.items():
            for dimension in self.sweep_points_dimensions:
                self.sweep_parameters[dimension].update(
                    sweep_parameter_dict[dimension])
        self.row_layouts = {
            "row_1": QtWidgets.QHBoxLayout(),
            "row_2": QtWidgets.QHBoxLayout(),
            "row_3": QtWidgets.QHBoxLayout(),
        }
        self.layout = QtWidgets.QVBoxLayout()
        [self.layout.addLayout(row_layout) 
            for row_layout in self.row_layouts.values()]
        self.values_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.dimension_combobox = QtWidgets.QComboBox()
        self.dimension_combobox.addItems(
            [str(dimension) for dimension in self.sweep_points_dimensions])
        self.parameter_name_cbox = QtWidgets.QComboBox()
        self.parameter_name_cbox.setEditable(True)
        self.parameter_name_cbox.setInsertPolicy(
            self.parameter_name_cbox.NoInsert)
        self.unit_lineedit = QtWidgets.QLineEdit()
        self.label_lineedit = QtWidgets.QLineEdit()
        self.values_selection_type_cbox = QtWidgets.QComboBox()
        self.values_selection_type_cbox.addItems(
            [SweepPointsValueSelectionType.value for
             SweepPointsValueSelectionType in
             SweepPointsValueSelectionTypes])
        self.values_selection_type_cbox.setCurrentIndex(0)
        self.delete_sweep_points_button = QtWidgets.QPushButton('Delete')
        self.setup_dialog()

    def configure_layout(self):
        self.row_layouts["row_1"].addLayout(add_label_to_widget(
            self.dimension_combobox, "Dimension: "))
        self.row_layouts["row_1"].addLayout(add_label_to_widget(
            self.parameter_name_cbox, "Parameter Name: "))
        self.row_layouts["row_1"].addLayout(
            add_label_to_widget(self.unit_lineedit, "Base unit: "))
        self.row_layouts["row_2"].addLayout(
            add_label_to_widget(self.label_lineedit, "Label: "))
        self.row_layouts["row_2"].addLayout(
            add_label_to_widget(self.values_selection_type_cbox,
                                "Values Selection Type: "))

        self.row_layouts["row_3"].addLayout(self.values_layout)
        self.row_layouts["row_3"].addWidget(self.delete_sweep_points_button)

    def connect_widgets(self):
        self.dimension_combobox.currentIndexChanged.connect(
            lambda: self.set_parameter_list(set_previous_choice=False))
        self.parameter_name_cbox.currentTextChanged.connect(
            lambda: self.set_fields_dependent_on_param(set_previous_choice=False))
        self.values_selection_type_cbox.currentIndexChanged.connect(
            lambda: self.configure_values_selection_field(
                set_previous_choice=False))
        self.delete_sweep_points_button.clicked.connect(
            self.delete_sweep_points)

    def setup_dialog(self):
        self.configure_layout()
        self.configure_values_selection_field(set_previous_choice=False)
        self.set_parameter_list(set_previous_choice=False)
        self.connect_widgets()

    def reset_to_saved_choices(self):
        if self.previous_choice is not None:
            self.dimension_combobox.blockSignals(True)
            self.dimension_combobox.setCurrentText(
                str(self.previous_choice["kwargs"]["dimension"]))
            self.dimension_combobox.blockSignals(False)
            if "label" in self.previous_choice["kwargs"].keys():
                self.label_lineedit.setText(
                    self.previous_choice["kwargs"]["label"])
            else: 
                # if label is not in the saved kwargs, it must have been empty 
                # when the configuration was saved
                self.label_lineedit.setText('')
            self.configure_values_selection_field(set_previous_choice=True)
            self.set_parameter_list(set_previous_choice=True)

    def set_parameter_list(self, set_previous_choice=False):
        self.parameter_name_cbox.blockSignals(True)
        dimension = int(self.dimension_combobox.currentText())
        self.parameter_name_cbox.clear()
        self.parameter_name_cbox.addItems(
            list(self.sweep_parameters[dimension].keys()))
        if self.previous_choice is not None and set_previous_choice:
            self.parameter_name_cbox.setCurrentText(
                self.previous_choice["kwargs"]["param_name"])
        else:
            self.parameter_name_cbox.setCurrentIndex(0)
        self.parameter_name_cbox.blockSignals(False)
        self.set_fields_dependent_on_param(set_previous_choice=set_previous_choice)

    def set_fields_dependent_on_param(self, set_previous_choice=False):
        self.set_standard_unit(set_previous_choice=set_previous_choice)
        if not set_previous_choice:
            lookup = {v['param_name']: v['label']
                    for v in self.kw_for_sweep_points.values()}
            param_name = self.parameter_name_cbox.currentText()
            if param_name in lookup:
                self.label_lineedit.setText(lookup[param_name])
            else:
                self.label_lineedit.setText("")

    def set_standard_unit(self, set_previous_choice=False):
        if self.previous_choice is not None and set_previous_choice:
            self.unit_lineedit.setText(self.previous_choice["kwargs"]["unit"])
        else:
            selected_sweep_parameter = self.parameter_name_cbox.currentText()
            current_dimension = int(self.dimension_combobox.currentText())
            if selected_sweep_parameter in [
                self.parameter_name_cbox.itemText(i) for i in range(
                    self.parameter_name_cbox.count())]:
                self.unit_lineedit.setText(
                    self.sweep_parameters[current_dimension][selected_sweep_parameter])

    def configure_values_selection_field(self, set_previous_choice=False):
        clear_layout(self.values_layout)
        if not set_previous_choice:
            values_selection_type = self.values_selection_type_cbox.currentText()
        else:
            self.values_selection_type_cbox.blockSignals(True)
            values_selection_type = self.previous_choice["_field_values"][
                "values_selection_type"]
            self.values_selection_type_cbox.setCurrentText(values_selection_type)
            self.values_selection_type_cbox.blockSignals(False)
        
        fields = []
        if values_selection_type == SweepPointsValueSelectionTypes.LINSPACE.value:
            self.values_start_value = QLineEditDouble()
            self.values_end_value = QLineEditDouble()
            self.values_number_of_steps = QLineEditInt()
            if set_previous_choice:
                self.values_start_value.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["start_value"]))
                self.values_end_value.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["end_value"]))
                self.values_number_of_steps.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["number_of_steps"]))
            fields.extend([
                (self.values_start_value, "Start Value: "),
                (self.values_end_value, "End Value: "),
                (self.values_number_of_steps, "Step Number: ")
            ])

        elif values_selection_type == SweepPointsValueSelectionTypes.ARANGE.value:
            self.values_start_value = QLineEditDouble()
            self.values_end_value = QLineEditDouble()
            self.values_step_size = QLineEditDouble()
            if set_previous_choice:
                self.values_start_value.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["start_value"]))
                self.values_end_value.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["end_value"]))
                self.values_step_size.setText(convert_field_value_to_string(
                    self.previous_choice["_field_values"]["step_size"]))
            fields.extend([
                (self.values_start_value, "Start Value: "),
                (self.values_end_value, "End Value: "),
                (self.values_step_size, "Step Size: ")
            ])

        elif values_selection_type == SweepPointsValueSelectionTypes.LIST.value:
            self.values_list = QtWidgets.QLineEdit()
            self.values_list.setPlaceholderText(
                "delimit items by comma, e.g. '1, 2, 3' or '\"foo\", \"bar\"'")
            if set_previous_choice:
                self.values_list.setText(
                    self.previous_choice["_field_values"]["list_string"])
            fields.extend([
                (self.values_list, "List Items: "),
            ])

        else:
            return

        [self.values_layout.addLayout(add_label_to_widget(widget, label))
         for widget, label in fields]

    def get_and_store_sweep_points_config(self):
        sweep_points_config = {
            "kwargs": {
                "param_name": self.parameter_name_cbox.currentText(),
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
        self.previous_choice = sweep_points_config
        return sweep_points_config

    def delete_sweep_points(self):
        self.dialog.sweep_points_list.remove(self)
        self.deleteLater()
        self.dialog.resize(self.dialog.minimumSizeHint())



class TaskForm(QtWidgets.QWidget):
    def __init__(self, parent, experiment_kwargs):
        super(TaskForm, self).__init__()
        self.parent = parent
        self.experiment_kwargs = experiment_kwargs
        self.experiment_name = self.parent.cbox_experiment_options.currentText()
        self.setLayout(QtWidgets.QFormLayout())
        self.layout().setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
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


class SweepPointsWidget(QtWidgets.QWidget):
    def __init__(self, sweeping_parameters, gui=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_values_maximum = 3
        self.gui = gui
        self.sweeping_parameters = sweeping_parameters
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.configure_sweep_points_button = QtWidgets.QPushButton("Configure")
        self.chosen_sweep_points_kwargs = None
        self.configure_sweep_points_button.clicked.connect(
            self.spawn_sweep_points_dialog)
        self.stored_sweep_parameter = QtWidgets.QLabel()
        self.stored_sweep_parameter.setWordWrap(True)
        self.stored_sweep_parameter.setText('None')
        self.layout.addWidget(self.configure_sweep_points_button,
                              QtCore.Qt.AlignLeft)
        self.layout.addWidget(self.stored_sweep_parameter, QtCore.Qt.AlignLeft)
        self.sweep_points_dialog = SweepPointsDialog(
            self.sweeping_parameters, parent=self, gui=self.gui)
        self.sweep_points_dialog.setModal(True)

    def spawn_sweep_points_dialog(self):
        clicked_button = self.sweep_points_dialog.exec_()
        if clicked_button == 1:
            self.chosen_sweep_points_kwargs = \
                self.sweep_points_dialog.get_and_store_sweep_points_config()
            if self.chosen_sweep_points_kwargs:
                spkw = [sweep_points_dict['kwargs'] for sweep_points_dict
                        in self.chosen_sweep_points_kwargs]

                self.gui.message_textedit.clear_and_set_text(
                    "saved the configuration:\n"
                    f"{spkw}")
                self.stored_sweep_parameter.setText(
                    "\n".join(
                        [f"{kw['dimension']}, {kw['param_name']}:"
                         + self.get_string_rep_of_values(kw['values'])
                         for kw in spkw])
                )
            else:
                self.gui.message_textedit.clear_and_set_text(
                    "No SweepPoints configured")
                self.stored_sweep_parameter.setText("None")
        else:
            self.gui.message_textedit.clear_and_set_text(
                "Configuration was not saved (click ok to save)")
            self.sweep_points_dialog.reset_to_saved_choices()

    def get_string_rep_of_values(self, values):

        if values is None:
            return str(values)
        else:
            if len(values) <= self.display_values_maximum:
                return f"[{','.join([str(v) for v in values])}]"
            else:
                truncated_values = values[:self.display_values_maximum]
                return f"[{','.join([str(v) for v in truncated_values])},...]"

