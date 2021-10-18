import sys
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
from copy import deepcopy
from pycqed.gui.rc_params import gui_rc_params
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox
from pycqed.utilities.general import get_channel_map

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


class WaveformViewer:
    def __init__(self, quantum_experiment, **kwargs):
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        w = MainWindow(quantum_experiment, **kwargs)
        app.exec()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, quantum_experiment=None, sequence_index=0, segment_index=0, normalized_amplitudes=False, plot_kwargs=None, parent=None, sharex=True, *args, **kwargs):
        # super().__init__(parent)
        QtWidgets.QMainWindow.__init__(self)
        self.quantum_experiment = quantum_experiment
        self.current_sequence_index = sequence_index
        self.current_segment_index = segment_index

        self.frame = QtWidgets.QWidget()
        self.setCentralWidget(self.frame)
        self.frame.setLayout(QtWidgets.QVBoxLayout())
        self.frame.layout().setContentsMargins(0, 5, 0, 5)
        self.frame.layout().setSpacing(0)

        self.cbox_sequence_indices = QtWidgets.QComboBox()
        self.cbox_sequence_indices.addItems([seq.name for seq in self.quantum_experiment.sequences])
        if self.cbox_sequence_indices.count() != 0:
            self.cbox_sequence_indices.setCurrentIndex(self.current_sequence_index)
        self.cbox_segment_indices = QtWidgets.QComboBox()
        self.cbox_segment_indices.addItems(list(self.quantum_experiment.sequences[self.current_sequence_index].segments.keys()))
        if self.cbox_segment_indices.count() != 0:
            self.cbox_segment_indices.setCurrentIndex(self.current_segment_index)

        self.checkbox_normalized = QtWidgets.QCheckBox('normalized amp')
        self.checkbox_normalized.setChecked(normalized_amplitudes)

        self.cbox_seq_label = QtWidgets.QLabel("Sequence: ")
        self.cbox_seq_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cbox_seg_label = QtWidgets.QLabel("Segment: ")
        self.cbox_seg_label.setAlignment(QtCore.Qt.AlignCenter)

        self.selectbox_qubits = CheckableComboBox()
        self.selectbox_qubits.default_display_text = 'Select...'
        self.selectbox_qubits.addItems([qubit.name for qubit in self.quantum_experiment.qubits])
        self.selectbox_qubits_label = QtWidgets.QLabel("Qubits: ")
        self.selectbox_qubits_label.setAlignment(QtCore.Qt.AlignCenter)

        plot_kwargs = deepcopy(plot_kwargs)
        self.plot_kwargs = plot_kwargs
        self.sharex = sharex

        self.gui_kwargs = {
            'show_and_close': False,
            'normalized_amplitudes': self.checkbox_normalized.isChecked(),
            'figtitle_kwargs': {'y': 0.98},
            'sharex': self.sharex,
            'plot_kwargs': self.plot_kwargs,
        }
        self.fig, self.axes = self.get_experiment_plot()
        self.view = FigureCanvasQTAgg(self.fig)
        self.view.draw()
        self.scroll = QtWidgets.QScrollArea(self.frame)
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(True)
        self.view.setMinimumSize(800, 400)
        self.toolbar = NavigationToolbar2QT(self.view, self.frame)

        seg_layout = QtWidgets.QHBoxLayout()
        seg_layout.addWidget(self.cbox_seg_label)
        seg_layout.addWidget(self.cbox_segment_indices)
        seg_layout.setSpacing(0)

        seq_layout = QtWidgets.QHBoxLayout()
        seq_layout.addWidget(self.cbox_seq_label)
        seq_layout.addWidget(self.cbox_sequence_indices)
        seq_layout.setSpacing(0)

        qub_layout = QtWidgets.QHBoxLayout()
        qub_layout.addWidget(self.selectbox_qubits_label)
        qub_layout.addWidget(self.selectbox_qubits)
        qub_layout.setSpacing(0)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addLayout(seq_layout)
        input_layout.addLayout(seg_layout)
        input_layout.addLayout(qub_layout)
        input_layout.addWidget(self.checkbox_normalized)
        input_layout.setSpacing(50)

        self.frame.layout().addWidget(self.toolbar)
        self.frame.layout().addWidget(self.scroll)
        self.frame.layout().addLayout(input_layout)

        self.setWindowTitle(self.get_window_title_string())

        self.cbox_sequence_indices.currentIndexChanged.connect(lambda: self.on_change(caller_id="sequence_change"))
        self.cbox_segment_indices.currentIndexChanged.connect(self.on_change)
        self.selectbox_qubits.currentIndexChanged.connect(self.on_change)
        self.checkbox_normalized.stateChanged.connect(self.on_change)
        self.selectbox_qubits.model().dataChanged.connect(self.on_change)
        self.showMaximized()

    def on_change(self, caller_id=None):
        self.current_segment_index = self.cbox_segment_indices.currentText()
        self.current_sequence_index = self.cbox_sequence_indices.currentIndex()
        if caller_id == "sequence_change":
            self.cbox_segment_indices.blockSignals(True)
            self.cbox_segment_indices.clear()
            self.cbox_segment_indices.addItems(list(self.quantum_experiment.sequences[self.current_sequence_index].segments.keys()))
            if self.cbox_segment_indices.count() != 0:
                self.cbox_segment_indices.setCurrentIndex(0)
            self.current_segment_index = self.cbox_segment_indices.currentText()
            self.cbox_segment_indices.blockSignals(False)
            # upon clearing the previous entries and adding the new ones, the first element of the list that was added
            # is automatically selected in the segment combo box (i.e. segment 0)
        oldfig = self.view.figure
        with plt.rc_context(gui_rc_params):
            fig, axes = self.get_experiment_plot()
        self.view.figure = fig
        plt.close(oldfig)
        self.view.draw()
        self.setWindowTitle(self.get_window_title_string())
        self.trigger_resize_event()

    def get_window_title_string(self):
        title_string = 'Sequence ' + str(self.current_sequence_index) + ', Segment ' + str(self.current_segment_index)
        if len(self.selectbox_qubits.currentData()) != 0:
            for qubit in self.selectbox_qubits.currentData():
                title_string += ", "+qubit
        if self.checkbox_normalized.isChecked():
            title_string += ' (normalized amplitude)'
        return title_string

    def trigger_resize_event(self):
        # ugly hack to trigger a resize event (otherwise the figure in the canvas is not displayed properly)
        size = self.size()
        self.resize(self.width() + 1, self.height())
        self.resize(size)

    def get_experiment_plot(self, **kwargs):
        self.gui_kwargs = {
            'show_and_close': False,
            'normalized_amplitudes': self.checkbox_normalized.isChecked(),
            'plot_kwargs': self.plot_kwargs,
            'figtitle_kwargs': {'y': 0.98},
            'sharex': self.sharex,
        }
        with plt.rc_context(gui_rc_params):
            if len(self.selectbox_qubits.currentData()) == 0:
                return self.quantum_experiment.sequences[self.current_sequence_index].segments[self.current_segment_index].plot(**self.gui_kwargs, **kwargs)
            else:
                qubits = [qubit for qubit in self.quantum_experiment.qubits if qubit.name in self.selectbox_qubits.currentData()]
                return self.quantum_experiment.sequences[self.current_sequence_index].segments[self.current_segment_index].plot(channel_map=get_channel_map(qubits), **self.gui_kwargs, **kwargs)
