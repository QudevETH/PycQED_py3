import sys
import os
import types
from pycqed.gui.qt_widgets.qt_field_widgets import *
from pycqed.gui import gui_utilities as g_utils
from pycqed.measurement.calibration import two_qubit_gates
from pycqed.measurement.calibration import single_qubit_gates
import numpy as np
from collections import OrderedDict as odict
from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
    import Qubit
from pycqed.measurement.waveform_control.circuit_builder import CircuitBuilder
from pycqed.measurement.sweep_points import SweepPoints
from pycqed.gui import qt_compat as qt
import logging
from enum import Enum
from pycqed.analysis import analysis_toolbox as a_tools
import traceback

log = logging.getLogger(__name__)


class ExperimentTypes(Enum):
    RABI = single_qubit_gates.Rabi
    RAMSEY = single_qubit_gates.Ramsey
    QSCALE = single_qubit_gates.QScale
    T1 = single_qubit_gates.T1
    CHEVRON = two_qubit_gates.Chevron


def get_members_by_experiment_class_name(experiment_class_name):
    return [ExperimentType for ExperimentType in ExperimentTypes 
            if ExperimentType.value.__name__ == experiment_class_name]


class QuantumExperimentGUI:

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
            **kwargs:
        """
        if qt.QtWidgets.__package__ not in ['PySide2']:
            log.warning('This GUI is optimized to run with the PySide2 Qt '
                        'binding')
        self.device = device
        self.experiments = []
        self.experiments_failed_in_init = []
        if not qt.QtWidgets.QApplication.instance():
            self.app = qt.QtWidgets.QApplication(sys.argv)
        else:
            self.app = qt.QtWidgets.QApplication.instance()
        g_utils.handle_matplotlib_backends(self.app)
        self.main_window = QuantumExperimentGUIMainWindow(
            self.device, self.experiments, self.experiments_failed_in_init,
            **kwargs
        )
        self.main_window.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                }
                """)
        self.spawn_gui()

    def spawn_gui(self):
        # update the matplotlib backend identifier stored in
        # self.app._matplotlib_backend such that the backend can be properly
        # reset after the last gui window has been closed
        self.app._matplotlib_backend = \
            sys.modules.get('matplotlib').get_backend()
        # The Agg backend has to be used because otherwise the plotting
        # backend might try to spawn a matplotlib gui in a separate thread.
        # As matplotlib is not thread-safe, this might cause a crash
        sys.modules.get('matplotlib').use('Agg')
        qt.QtWidgets.QApplication.restoreOverrideCursor()
        self.main_window.showMaximized()
        self.app.exec_()


class QuantumExperimentGUIMainWindow(qt.QtWidgets.QMainWindow):
    def __init__(self, device, experiments, experiments_failed_in_init,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Quantum Experiment GUI")
        self.device = device
        self.experiments = experiments
        self.experiments_failed_in_init = experiments_failed_in_init

        self.boldfont = qt.QtGui.QFont()
        self.boldfont.setBold(True)

        # container for general options (experiment class dropdown, add tasks
        # button, qubit selection dropdown)
        self.scroll = qt.QtWidgets.QScrollArea()
        self.setCentralWidget(self.scroll)
        self.mainWidget = qt.QtWidgets.QWidget()
        self.scroll.setWidget(self.mainWidget)
        self.mainWidget.setLayout(qt.QtWidgets.QVBoxLayout())

        self.cbox_experiment_options = qt.QtWidgets.QComboBox()
        self.cbox_experiment_options.addItems(
            [member.value.__name__ for member in ExperimentTypes])
        self.cbox_experiment_options.setCurrentIndex(-1)

        self.add_task_form_button = qt.QtWidgets.QPushButton("&Add")
        self.add_task_form_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.add_task_form_button.setEnabled(False)

        self.selectbox_qubits = MultiQubitSelectionWidget(
            [qb.name for qb in self.device.get_qubits()])
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        self.logo_pixmap = qt.QtGui.QPixmap(
            os.path.join(self.file_path, "assets", "qudev_logo.png"))

        self.logo_label = qt.QtWidgets.QLabel()
        self.logo_label.setPixmap(self.logo_pixmap)

        self.header_container = qt.QtWidgets.QWidget()
        self.header_container.setLayout(qt.QtWidgets.QHBoxLayout())

        self.general_options_field_container = qt.QtWidgets.QWidget()
        self.general_options_field_container.setLayout(
            qt.QtWidgets.QFormLayout())

        # container for general qubit settings
        self.qubit_settings_container = qt.QtWidgets.QGroupBox("Qubit Settings")
        self.qubit_settings_container.setLayout(qt.QtWidgets.QVBoxLayout())
        self.qubit_pulse_period = ConfigureDialogWidget(
            gui=self, dialog_widget=PulsePeriodDialog,
            on_ok_method="update_pulse_period",
            cancel_message="No pulse_period configured\n",
            on_cancel_reset_method="reset_value_lineedit"
        )
        self.qubit_acq_averages = ConfigureDialogWidget(
            gui=self, dialog_widget=AcqAvgDialog,
            on_ok_method="update_acq_avg_for_selected_qubit",
            cancel_message="No acq_averages configured\n",
            on_cancel_reset_method="show_current_value",
            on_open_dialog_method="on_open_dialog"
        )

        # container for configuring the task list
        self.tasks_configuration_container = qt.QtWidgets.QGroupBox(
            "Configure Tasks")
        self.tasks_configuration_container.setLayout(qt.QtWidgets.QHBoxLayout())
        self.tasks_configuration_container.hide()

        # container for the experiment dependent configuration options
        self.experiment_configuration_container = qt.QtWidgets.QFormLayout()

        # run experiment button and spinning wheel
        self.create_experiment_pushbutton = qt.QtWidgets.QPushButton(
            "&Create Experiment")
        self.create_experiment_pushbutton.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.create_experiment_pushbutton.hide()
        self.run_waiting_label = qt.QtWidgets.QLabel()
        self.run_waiting_animation = qt.QtGui.QMovie(
            os.path.join(self.file_path, "assets", "spinner_animation.gif"))
        self.run_waiting_animation.setScaledSize(qt.QtCore.QSize(40, 40))
        self.run_waiting_label.setMovie(self.run_waiting_animation)
        self.run_waiting_label.hide()

        # actions for the performed experiments
        self.performed_experiments_cbox = qt.QtWidgets.QComboBox()
        self.performed_experiments_cbox.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Minimum,
            qt.QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.performed_experiments_cbox.hide()
        self.spawn_waveform_viewer_button = qt.QtWidgets.QPushButton(
            "View Waveforms")
        self.spawn_waveform_viewer_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.run_measurement_button = qt.QtWidgets.QPushButton(
            "Run Measurement")
        self.run_measurement_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.run_analysis_button = qt.QtWidgets.QPushButton(
            "Run Analysis")
        self.run_analysis_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.run_update_button = qt.QtWidgets.QPushButton(
            "Run Update")
        self.run_update_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.open_in_explorer_button = qt.QtWidgets.QPushButton(
            "Open in Explorer")
        self.open_in_explorer_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.spawn_waveform_viewer_button.hide()
        self.run_measurement_button.hide()
        self.run_analysis_button.hide()
        self.run_update_button.hide()
        self.open_in_explorer_button.hide()

        # container for actions below the experiment configuration widgets
        self.experiment_actions_container = qt.QtWidgets.QGroupBox("Actions")
        self.experiment_actions_container.hide()

        # message box
        self.message_textedit = ScrollLabelFixedLineHeight(number_of_lines=6)

        self.set_layout()
        self.connect_widgets()
        # self.active_threads is used as manual thread counter, because
        # self.threadpool.activeThreadCount() behaves inconsistently
        self.active_threads = 0
        self.threadpool = qt.QtCore.QThreadPool.globalInstance()

    def connect_widgets(self):
        self.cbox_experiment_options.currentIndexChanged.connect(
            self.handle_experiment_choice)
        self.add_task_form_button.clicked.connect(self.add_task_form)
        self.create_experiment_pushbutton.clicked.connect(
            self.create_experiment)
        self.performed_experiments_cbox.currentIndexChanged.connect(
            self.handle_experiment_action_buttons)
        self.spawn_waveform_viewer_button.clicked.connect(
            lambda: self.perform_experiment_action("spawn_waveform_viewer")
        )
        self.run_measurement_button.clicked.connect(
            lambda: self.perform_experiment_action("run_measurement")
        )
        self.run_analysis_button.clicked.connect(
            lambda: self.perform_experiment_action("run_analysis")
        )
        self.run_update_button.clicked.connect(
            lambda: self.perform_experiment_action("run_update")
        )
        self.open_in_explorer_button.clicked.connect(
            lambda: self.perform_experiment_action("open_in_explorer")
        )

    def set_layout(self):
        self.general_options_field_container.layout().addRow(
            "Choose Experiment: ", self.cbox_experiment_options)
        self.general_options_field_container.layout().addRow(
            "Add Task: ", self.add_task_form_button)
        self.general_options_field_container.layout().addRow(
            qt.QtWidgets.QLabel("OR"))
        self.general_options_field_container.layout().addRow(
            "Choose Qubits: ", self.selectbox_qubits)
        self.general_options_field_container.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Expanding,
            qt.QtWidgets.QSizePolicy.Policy.Preferred)

        self.qubit_settings_container.layout().addLayout(
            g_utils.add_label_to_widget(
                self.qubit_pulse_period, "pulse_period: "))
        self.qubit_settings_container.layout().addLayout(
            g_utils.add_label_to_widget(
                self.qubit_acq_averages, "acq_averages: "))
        self.qubit_settings_container.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Expanding,
            qt.QtWidgets.QSizePolicy.Policy.Preferred)

        self.header_container.layout().addWidget(
            self.general_options_field_container)
        self.header_container.layout().addWidget(
            self.qubit_settings_container)
        self.header_container.layout().addWidget(
            self.logo_label)

        self.experiment_actions_container.setLayout(qt.QtWidgets.QVBoxLayout())
        self.experiment_actions_container.layout().addWidget(
            self.create_experiment_pushbutton)
        self.experiment_actions_container.layout().addStretch()
        self.experiment_actions_container.layout().addWidget(
            self.performed_experiments_cbox)
        self.experiment_actions_container.layout().addWidget(
            self.spawn_waveform_viewer_button)
        self.experiment_actions_container.layout().addWidget(
            self.run_measurement_button)
        self.experiment_actions_container.layout().addWidget(
            self.run_analysis_button)
        self.experiment_actions_container.layout().addWidget(
            self.run_update_button)
        self.experiment_actions_container.layout().addWidget(
            self.open_in_explorer_button)
        self.experiment_actions_container.layout().addStretch()
        self.experiment_actions_container.layout().addWidget(
            self.run_waiting_label)

        lower_container = qt.QtWidgets.QHBoxLayout()
        lower_container.addLayout(self.experiment_configuration_container)
        lower_container.addWidget(self.experiment_actions_container)

        # main vertical layout
        main_vbox_elements = [
            (self.header_container, 'widget'),
            (None, 'stretch'),
            (self.tasks_configuration_container, 'widget'),
            (lower_container, 'layout'),
            (None, 'stretch'),
            (self.message_textedit, 'widget')
        ]
        for element, type in main_vbox_elements:
            if type == 'widget':
                self.mainWidget.layout().addWidget(element)
            elif type == 'layout':
                self.mainWidget.layout().addLayout(element)
            elif type == 'stretch':
                self.mainWidget.layout().addStretch()

        # configure sensible default size for non-maximized view
        screen_geometry = self.window().screen().availableGeometry()
        title_bar_height = self.style().pixelMetric(
            qt.QtWidgets.QStyle.PixelMetric.PM_TitleBarHeight)
        # manual subtraction of 10 pixels is heuristic countermeasure to the
        # suboptimal placement of window by the window manager
        self.resize(
            self.width(), screen_geometry.height() - title_bar_height - 10)
        self.scroll.setWidgetResizable(True)

    def handle_experiment_choice(self):
        self.add_task_form_button.setEnabled(True)
        g_utils.clear_QFormLayout(self.experiment_configuration_container)
        g_utils.clear_layout(self.tasks_configuration_container.layout())
        self.tasks_configuration_container.hide()
        self.add_experiment_fields()

    def add_experiment_fields(self):
        exp = self.get_selected_experiment()
        input_field_dict = exp.gui_kwargs(self.device)["kwargs"]
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
        task_list_fields = exp.gui_kwargs(self.device)["task_list_fields"]
        for class_name, kwarg_dict in reversed(task_list_fields.items()):
            for kwarg, field_information in kwarg_dict.items():
                if kwarg in getattr(exp, 'kw_for_task_keys', []) and \
                        kwarg not in input_field_dict.get(class_name, {}):
                    if class_name not in input_field_dict:
                        input_field_dict[class_name] = odict({})
                    input_field_dict[class_name][kwarg] = field_information
        # create the widgets
        for class_name, kwarg_dict in reversed(input_field_dict.items()):
            class_name_label = qt.QtWidgets.QLabel(f"{class_name} Options")
            class_name_label.setFont(self.boldfont)
            if len(kwarg_dict):
                self.experiment_configuration_container.addRow(class_name_label)
            for kwarg, field_information in kwarg_dict.items():
                self.add_widget_to_experiment_section(kwarg, field_information)
        self.experiment_actions_container.show()
        self.create_experiment_pushbutton.show()

    def show_experiment_action_buttons(self):
        self.spawn_waveform_viewer_button.show()
        self.run_measurement_button.show()
        self.run_analysis_button.show()
        self.run_update_button.show()
        self.open_in_explorer_button.show()

    def add_widget_to_experiment_section(self, kwarg, field_information):
        widget = self.create_field_from_field_information(field_information)
        if widget is None:
            return
        self.experiment_configuration_container.addRow(kwarg, widget)

    def add_task_form(self):
        self.tasks_configuration_container.show()
        experiment_kwargs = self.get_selected_experiment().gui_kwargs(
            self.device)
        self.tasks_configuration_container.layout().addWidget(
            TaskForm(parent=self, experiment_kwargs=experiment_kwargs))
        if sum([isinstance(widget, TaskForm) for widget in
                self.tasks_configuration_container.children()]) == 1:
            check_state = qt.QtCore.Qt.CheckState.Unchecked
            for i in range(self.selectbox_qubits.model().rowCount()):
                self.selectbox_qubits.model().item(i).setCheckState(
                    check_state)

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
                dropdown menu composed of the elements from the list,
                returns the displayed value when experiment is run)
                instance of set -> QComboBox with QLineEdit (creates a
                single selection dropdown menu composed of the elements
                from the set while also allowing custom text input by user,
                returns the displayed value when experiment is run)
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
        experiment_kwargs = self.get_selected_experiment().gui_kwargs(
            self.device)
        if field_information[0] is str:
            widget = qt.QtWidgets.QLineEdit()
            if field_information[1] is not None:
                widget.setText(field_information[1])
        elif field_information[0] is bool:
            widget = qt.QtWidgets.QCheckBox()
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
            widget = qt.QtWidgets.QComboBox()
            widget.addItems(field_information[0])
            if field_information[1] is not None:
                widget.setCurrentText(field_information[1])
            else:
                widget.setCurrentIndex(-1)
        elif isinstance(field_information[0], set):
            widget = qt.QtWidgets.QComboBox()
            widget.setEditable(True)
            widget.setInsertPolicy(widget.InsertPolicy.NoInsert)
            widget.addItems(field_information[0])
            if field_information[1] is not None:
                widget.setCurrentText(field_information[1])
            else:
                widget.setCurrentIndex(-1)
        elif isinstance(field_information[0], dict):
            widget = qt.QtWidgets.QComboBox()
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
        elif isinstance(widget, qt.QtWidgets.QLineEdit):
            return widget.text() if widget.text() != "" else None
        elif isinstance(widget, qt.QtWidgets.QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, qt.QtWidgets.QComboBox):
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

    def create_experiment(self):
        self.create_experiment_pushbutton.setEnabled(False)
        self.run_waiting_animation.start()
        self.run_waiting_label.show()
        experiment = self.get_selected_experiment()
        # The creation of a SweepPoints instance while trying to retrieve
        # the argument values for the instantiation of the quantum experiment
        # might raise an error if the sweep points configuration by the user
        # was faulty.
        # If not handled, the GUI will be stuck in create_experiment mode.
        try:
            experiment_settings_kwargs = self.get_QFormLayout_settings(
                self.experiment_configuration_container)
            qubits = self.selectbox_qubits.get_selected_qubits_from_device(
                self.device)
            task_list = self.get_task_list()
        except Exception as exception:
            pass_exception = types.SimpleNamespace()
            pass_exception.exception = exception
            self.handle_experiment_result(
                pass_exception, "Unable to set the experiment arguments.\n"
            )
            return

        if not len(task_list):
            task_list = None
        argument_string = (f"Arguments:\n"
                                f"  task list:\n"
                                f"    {task_list}\n"
                                f"  experiment settings:\n"
                                f"    {experiment_settings_kwargs}\n"
                                f"  qubits:\n"
                                f"    {qubits}")
        self.message_textedit.clear_and_set_text(
            "Creating Experiment...\n"
            "\n"
            f"{argument_string}"
            )
        # performing the experiment in a separate thread has the advantage
        # of making the gui main window responsive while the experiment is
        # being performed
        create_experiment_worker = CreateExperimentWorker()
        create_experiment_worker.signals.finished_experiment.connect(
            self.handle_experiment_result)
        create_experiment_worker.update_parameters(
            experiment=experiment,
            task_list=task_list,
            qubits=qubits,
            dev=self.device,
            experiment_settings_kwargs=experiment_settings_kwargs
        )
        self.active_threads += 1
        self.threadpool.start(create_experiment_worker)
        self.handle_experiment_action_buttons()

    @qt.QtCore.Slot(object, str)
    def handle_experiment_result(self, experiment, argument_string):
        self.active_threads -= 1
        self.run_waiting_animation.stop()
        self.run_waiting_label.hide()
        added_experiment = False
        if experiment.exception is not None:
            self.handle_exception(
                experiment.exception,
                pre_error_message="Unable to create experiment:\n",
                post_error_message=f"\n{argument_string}")
            if len(getattr(experiment, "sequences", [])):
                self.experiments.append(experiment)
                experiment.guess_label()
                self.performed_experiments_cbox.blockSignals(True)
                self.performed_experiments_cbox.insertItem(
                    0, f"{experiment.label} (failed)", False)
                self.performed_experiments_cbox.blockSignals(False)
                added_experiment = True
            else:
                self.experiments_failed_in_init.append(experiment)

        else:
            self.message_textedit.clear_and_set_text(
                "Successfully created the experiment!\n"
                "\n"
                f"{argument_string}")
            experiment.guess_label()
            self.experiments.append(experiment)
            self.performed_experiments_cbox.blockSignals(True)
            self.performed_experiments_cbox.insertItem(
                0, f"{experiment.label} ({experiment.timestamp})", True)
            self.performed_experiments_cbox.blockSignals(False)
            added_experiment = True
        if added_experiment:
            self.performed_experiments_cbox.blockSignals(True)
            self.performed_experiments_cbox.setCurrentIndex(0)
            self.performed_experiments_cbox.blockSignals(False)
            self.performed_experiments_cbox.show()
            self.show_experiment_action_buttons()
        self.handle_experiment_action_buttons()
        self.create_experiment_pushbutton.setEnabled(True)

    def get_QFormLayout_settings(self, QFormLayoutInstance):
        settings_dict = {}
        kwarg_widget_pairs = [(
            QFormLayoutInstance.itemAt(
                i,qt.QtWidgets.QFormLayout.ItemRole.LabelRole).widget().text(),
            QFormLayoutInstance.itemAt(
                i, qt.QtWidgets.QFormLayout.ItemRole.FieldRole).widget())
            for i in range(QFormLayoutInstance.rowCount())
            if QFormLayoutInstance.itemAt(
                i, qt.QtWidgets.QFormLayout.ItemRole.LabelRole) is not None
            and QFormLayoutInstance.itemAt(
                i, qt.QtWidgets.QFormLayout.ItemRole.FieldRole) is not None]
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

    def get_selected_performed_experiment_index(self):
        list_index = self.performed_experiments_cbox.currentIndex()
        list_length = self.performed_experiments_cbox.count()
        return list_length - list_index - 1

    def perform_experiment_action(self, action):
        experiment_index = self.get_selected_performed_experiment_index()
        experiment = self.experiments[experiment_index]
        experiment_methods = {
            "run_measurement":
                ("Running measurement...",
                 RunMeasurementWorker,
                 "finished_measurement"),
            "run_analysis":
                ("Running analysis...",
                 RunAnalysisWorker,
                 "finished_analysis"),
            "run_update":
                ("Running update...",
                 RunUpdateWorker,
                 "finished_update")
        }
        if action in experiment_methods.keys():
            message, worker_class, signal = experiment_methods[action]
            self.run_waiting_animation.start()
            self.run_waiting_label.show()
            self.set_enabled_buttons_experiment_methods(False)
            self.message_textedit.clear_and_set_text(
                message
            )
            worker = worker_class()
            getattr(worker.signals, signal).connect(
                self.handle_experiment_actions
            )
            worker.signals.exception.connect(
                self.handle_experiment_action_exception)
            worker.update_parameters(
                experiment=experiment
            )
            self.active_threads += 1
            self.threadpool.start(worker)
        elif action == "spawn_waveform_viewer":
            self.spawn_waveform_viewer_button.setEnabled(False)
            experiment.spawn_waveform_viewer(
                new_process=False, active_qapp=True)
            self.spawn_waveform_viewer_button.setEnabled(True)
        elif action == "open_in_explorer":
            filepath = a_tools.get_folder(
                experiment.timestamp)
            g_utils.open_file_in_explorer(filepath)

    @qt.QtCore.Slot(object, str)
    def handle_experiment_action_exception(self, exception, pre_error_message):
        self.active_threads -= 1
        pre_error_message = pre_error_message
        self.handle_exception(
            exception,
            pre_error_message=pre_error_message)
        self.run_waiting_animation.stop()
        self.run_waiting_label.hide()
        self.handle_experiment_action_buttons()

    def handle_exception(self, exception, pre_error_message="",
                         post_error_message=""):
        traceback.print_exception(
            type(exception), exception, exception.__traceback__)
        self.message_textedit.clear_and_set_text(
            pre_error_message +
            f"{type(exception).__name__}: {str(exception)}\n"
            + post_error_message)

    @qt.QtCore.Slot(str, object)
    def handle_experiment_actions(self, action, experiment):
        self.active_threads -= 1
        message = None
        if action == "run_measurement":
            message = "Successfully ran the measurement!\n"
            experiment_index = self.experiments.index(experiment)
            list_index = len(self.experiments) - experiment_index - 1
            experiment.guess_label()
            self.performed_experiments_cbox.setItemText(
                list_index, f"{experiment.label} ({experiment.timestamp})")
        elif action == "run_analysis":
            message = "Successfully ran the analysis!\n"
        elif action == "run_update":
            message = "Successfully updated the experiment!\n"
        if message is not None:
            self.message_textedit.clear_and_set_text(message)
        self.run_waiting_animation.stop()
        self.run_waiting_label.hide()
        self.handle_experiment_action_buttons()

    def handle_experiment_action_buttons(self):
        if self.performed_experiments_cbox.count() == 0:
            return
        # index of experiment in performed experiments dropdown list
        dropdown_experiment_index = \
            self.performed_experiments_cbox.currentIndex()
        # index of experiment in self.experiments
        experiment_index = self.get_selected_performed_experiment_index()
        chosen_experiment = self.experiments[experiment_index]
        # checks
        experiment_successful = self.performed_experiments_cbox.itemData(
            dropdown_experiment_index)
        timestamp_not_none = getattr(
            chosen_experiment, 'timestamp', None) is not None
        run_update_possible = \
            getattr(chosen_experiment, 'analysis', None) is not None and \
            getattr(chosen_experiment, 'run_update', None) is not None
        # disable all buttons
        self.set_enabled_buttons_experiment_methods(False)
        self.open_in_explorer_button.setEnabled(False)
        # handle buttons associated with methods of experiment. They should
        # only be available if no thread is running
        if not self.active_threads > 0:
            if experiment_successful:
                self.run_measurement_button.setEnabled(True)
            if timestamp_not_none:
                self.run_analysis_button.setEnabled(True)
            if run_update_possible:
                self.run_update_button.setEnabled(True)
        # handle open in explorer button
        if timestamp_not_none:
            self.open_in_explorer_button.setEnabled(True)

    def set_enabled_buttons_experiment_methods(self, enabled=True):
        self.run_measurement_button.setEnabled(enabled)
        self.run_analysis_button.setEnabled(enabled)
        self.run_update_button.setEnabled(enabled)

    def keyPressEvent(self, event):
        if event.key() == qt.QtCore.Qt.Key.Key_W \
                and event.modifiers() == \
                qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            self.close()
            event.accept()


class CreateExperimentWorker(g_utils.SimpleWorker):
    @qt.QtCore.Slot()
    def run(self):
        argument_string = (f"Arguments:\n"
                           f"  task list:\n"
                           f"    {self.task_list}\n"
                           f"  experiment settings:\n"
                           f"    {self.experiment_settings_kwargs}\n"
                           f"  qubits:\n"
                           f"    {self.qubits}")
        try:
            return_experiment = self.experiment(
                task_list=self.task_list,
                qubits=self.qubits,
                dev=self.dev,
                **self.experiment_settings_kwargs)
        except Exception as exception:
            return_experiment = types.SimpleNamespace()
            return_experiment.exception = exception
        self.signals.finished_experiment.emit(
            return_experiment, argument_string)


class RunMeasurementWorker(g_utils.SimpleWorker):
    @qt.QtCore.Slot()
    def run(self):
        try:
            self.experiment.run_measurement()
            self.signals.finished_measurement.emit(
                "run_measurement", self.experiment)
        except Exception as exception:
            pre_error_message = "Unable to perform measurement\n"
            self.signals.exception.emit(exception, pre_error_message)


class RunAnalysisWorker(g_utils.SimpleWorker):
    @qt.QtCore.Slot()
    def run(self):
        try:
            self.experiment.run_analysis(
                analysis_kwargs={"raise_exceptions": True})
            self.signals.finished_analysis.emit(
                "run_analysis", self.experiment)
        except Exception as exception:
            pre_error_message = "Unable to perform analysis\n"
            self.signals.exception.emit(exception, pre_error_message)


class RunUpdateWorker(g_utils.SimpleWorker):
    @qt.QtCore.Slot()
    def run(self):
        try:
            self.experiment.run_update()
            self.signals.finished_update.emit("run_update", self.experiment)
        except Exception as exception:
            pre_error_message = "Unable to update the experiment\n"
            self.signals.exception.emit(exception, pre_error_message)


class SweepPointsValueSelectionTypes(Enum):
    LINSPACE = "np.linspace"
    ARANGE = "np.arange"
    LIST = "list"


class SweepPointsDialog(qt.QtWidgets.QDialog):
    def __init__(self, sweep_parameters, parent, gui=None):
        super().__init__(parent=parent)
        self.sweep_parameters = sweep_parameters
        self.parent = parent
        self.gui = gui

        self.setSizeGripEnabled(True)
        self.setWindowTitle("Configure Sweep Points")
        self.setLayout(qt.QtWidgets.QVBoxLayout())

        message = qt.QtWidgets.QLabel(
            "Choose the sweep points for the selected dimension")
        self.layout().addWidget(message)
        accept_cancel_buttons = \
            qt.QtWidgets.QDialogButtonBox.StandardButton.Ok | \
            qt.QtWidgets.QDialogButtonBox.StandardButton.Cancel
        self.buttonBox = qt.QtWidgets.QDialogButtonBox(accept_cancel_buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.add_sweep_points_button = qt.QtWidgets.QPushButton('&Add')
        self.add_sweep_points_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Fixed,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.layout().addWidget(self.add_sweep_points_button)
        self.sweep_points_list = []
        self.add_sweep_points_button.clicked.connect(self.add_sweep_points)
        self.layout().addWidget(self.buttonBox)
        self.add_sweep_points()
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


class SweepPointsForm(qt.QtWidgets.QGroupBox):
    def __init__(self, sweep_parameters, dialog, parent=None, gui=None):
        super().__init__(parent=parent)
        self.previous_choice = None
        self.dialog = dialog
        self.gui = gui
        self.kw_for_sweep_points = \
            self.gui.get_selected_experiment().kw_for_sweep_points

        self.sweep_points_dimensions = [0, 1]
        self.sweep_parameters = {}
        for dimension in self.sweep_points_dimensions:
            self.sweep_parameters.update({dimension: {}})
        for class_name, sweep_parameter_dict in sweep_parameters.items():
            for dimension in self.sweep_points_dimensions:
                self.sweep_parameters[dimension].update(
                    sweep_parameter_dict[dimension])
        self.row_layouts = {
            "row_1": qt.QtWidgets.QHBoxLayout(),
            "row_2": qt.QtWidgets.QHBoxLayout(),
            "row_3": qt.QtWidgets.QHBoxLayout(),
        }
        self.layout = qt.QtWidgets.QVBoxLayout()
        [self.layout.addLayout(row_layout) 
            for row_layout in self.row_layouts.values()]
        self.values_layout = qt.QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.dimension_combobox = qt.QtWidgets.QComboBox()
        self.dimension_combobox.addItems(
            [str(dimension) for dimension in self.sweep_points_dimensions])
        self.parameter_name_cbox = qt.QtWidgets.QComboBox()
        self.parameter_name_cbox.setEditable(True)
        self.parameter_name_cbox.setInsertPolicy(
            self.parameter_name_cbox.InsertPolicy.NoInsert)
        self.unit_lineedit = qt.QtWidgets.QLineEdit()
        self.label_lineedit = qt.QtWidgets.QLineEdit()
        self.values_selection_type_cbox = qt.QtWidgets.QComboBox()
        self.values_selection_type_cbox.addItems(
            [SweepPointsValueSelectionType.value for
             SweepPointsValueSelectionType in
             SweepPointsValueSelectionTypes])
        self.values_selection_type_cbox.setCurrentIndex(0)
        self.delete_sweep_points_button = qt.QtWidgets.QPushButton('Delete')
        self.delete_sweep_points_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.setup_dialog()

    def configure_layout(self):
        self.row_layouts["row_1"].addLayout(g_utils.add_label_to_widget(
            self.dimension_combobox, "Dimension: "))
        self.row_layouts["row_1"].addLayout(g_utils.add_label_to_widget(
            self.parameter_name_cbox, "Parameter Name: "))
        self.row_layouts["row_1"].addLayout(
            g_utils.add_label_to_widget(self.unit_lineedit, "Base unit: "))
        self.row_layouts["row_2"].addLayout(
            g_utils.add_label_to_widget(self.label_lineedit, "Label: "))
        self.row_layouts["row_2"].addLayout(
            g_utils.add_label_to_widget(self.values_selection_type_cbox,
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
        g_utils.clear_layout(self.values_layout)
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
                self.values_start_value.setText(
                    g_utils.convert_field_value_to_string(
                        self.previous_choice["_field_values"]["start_value"]))
                self.values_end_value.setText(
                    g_utils.convert_field_value_to_string(
                        self.previous_choice["_field_values"]["end_value"]))
                self.values_number_of_steps.setText(
                    g_utils.convert_field_value_to_string(
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
                self.values_start_value.setText(
                    g_utils.convert_field_value_to_string(
                        self.previous_choice["_field_values"]["start_value"]))
                self.values_end_value.setText(
                    g_utils.convert_field_value_to_string(
                        self.previous_choice["_field_values"]["end_value"]))
                self.values_step_size.setText(
                    g_utils.convert_field_value_to_string(
                        self.previous_choice["_field_values"]["step_size"]))
            fields.extend([
                (self.values_start_value, "Start Value: "),
                (self.values_end_value, "End Value: "),
                (self.values_step_size, "Step Size: ")
            ])

        elif values_selection_type == SweepPointsValueSelectionTypes.LIST.value:
            self.values_list = qt.QtWidgets.QLineEdit()
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

        [self.values_layout.addLayout(g_utils.add_label_to_widget(widget, label))
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
                values = [g_utils.convert_string_to_number_if_possible(value)
                          for value in self.values_list.text().split(",")]
        sweep_points_config["kwargs"].update({"values": values})
        self.previous_choice = sweep_points_config
        return sweep_points_config

    def delete_sweep_points(self):
        self.dialog.sweep_points_list.remove(self)
        self.deleteLater()
        self.dialog.resize(self.dialog.minimumSizeHint())


class TaskForm(qt.QtWidgets.QWidget):
    def __init__(self, parent, experiment_kwargs):
        super(TaskForm, self).__init__()
        self.parent = parent
        self.experiment_kwargs = experiment_kwargs
        self.experiment_name = self.parent.cbox_experiment_options.currentText()
        self.setLayout(qt.QtWidgets.QFormLayout())
        self.layout().setFieldGrowthPolicy(
            qt.QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for class_name, kwarg_dict \
                in self.experiment_kwargs["task_list_fields"].items():
            for kwarg, field_information in kwarg_dict.items():
                self.add_task_widget(kwarg, field_information)

        delete_task_button = qt.QtWidgets.QPushButton("Delete Task")
        delete_task_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
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


class SweepPointsWidget(qt.QtWidgets.QWidget):
    def __init__(self, sweeping_parameters, gui=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_values_maximum = 3
        self.gui = gui
        self.sweeping_parameters = sweeping_parameters
        self.layout = qt.QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.configure_sweep_points_button = qt.QtWidgets.QPushButton("Configure")
        self.configure_sweep_points_button.setSizePolicy(
            qt.QtWidgets.QSizePolicy.Policy.Preferred,
            qt.QtWidgets.QSizePolicy.Policy.Fixed)
        self.chosen_sweep_points_kwargs = None
        self.configure_sweep_points_button.clicked.connect(
            self.spawn_sweep_points_dialog)
        self.stored_sweep_parameter = qt.QtWidgets.QLabel()
        self.stored_sweep_parameter.setWordWrap(True)
        self.stored_sweep_parameter.setText('None')
        self.layout.addWidget(self.configure_sweep_points_button,
                              qt.QtCore.Qt.AlignmentFlag.AlignLeft)
        self.layout.addWidget(self.stored_sweep_parameter,
                              qt.QtCore.Qt.AlignmentFlag.AlignTop)
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
                values = [np.format_float_scientific(v, precision=3)
                          for v in values]
                return f"[{','.join(values)}]"
            else:
                truncated_values = values[:self.display_values_maximum]
                truncated_values = [np.format_float_scientific(v, precision=3)
                                    for v in truncated_values]
                return f"[{','.join([str(v) for v in truncated_values])},...]"

