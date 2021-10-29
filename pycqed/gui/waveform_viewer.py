import sys
import matplotlib

# matplotlib.use('Qt5Agg')

from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.gui.rc_params import gui_rc_params
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox

import matplotlib.figure
# simpler import
import importlib
from pycqed.gui import pulsar_shadow
importlib.reload(pulsar_shadow)

from pycqed.utilities.general import get_channel_map
import multiprocessing as mp
from copy import deepcopy
import os

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class WaveformViewer:
    def __init__(self, quantum_experiment, **kwargs):
        try:
            mp.set_start_method('spawn')
        except:
            if mp.get_start_method() != 'spawn':
                warnings.warn("Child process should be spawned")

        if QtWidgets.__package__ not in ['PySide2']:
            warnings.warn("This GUI is optimized to run with the PySide2 Qt binding")
        # qubit_name_combinations = {', '.join([q.name for q in qset]): get_channel_map(qset) for qset in powerset(quantum_experiment.qubits) if len(qset) != 0}
        qubit_channel_maps = [get_channel_map(qset) for qset in powerset(quantum_experiment.qubits) if len(qset) != 0]
        sequences = deepcopy(quantum_experiment.sequences)
        # p_shadow = pulsar_shadow.PulsarShadow(Pulsar.get_instance())
        p_shadow = pulsar_shadow.PulsarShadow(Pulsar.get_instance())
        for seq in sequences:
            seq.pulsar = p_shadow
            for segname, seg in seq.segments.items():
                seg.pulsar = p_shadow
        # sequences = {sequence.name: deepcopy(sequence.segments) for sequence in quantum_experiment.sequences}

        # one can obtain a list of the keys of the channel maps (i.e. list of associated qubits) via
        # qubit_names = [list(chmap.keys()) for chmap in qubit_channel_maps]

        self.process = mp.Process(name='pycqed_gui', target=self.start_qapp, args=(sequences, qubit_channel_maps), kwargs=kwargs)
        self.process.daemon = False
        self.process.start()

    def start_qapp(self, sequences, qubit_channel_maps, **kwargs):
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()
        w = MainWindow(sequences, qubit_channel_maps, **kwargs)
        self.app.exec_()


class DummyWindow(QtWidgets.QMainWindow):
    def __init__(self, qubit_channel_maps):
        QtWidgets.QMainWindow.__init__(self)
        self.dummy_label = QtWidgets.QLabel(', '.join(['. '.join(list(chmap.keys())) for chmap in qubit_channel_maps]))
        self.setCentralWidget(self.dummy_label)
        self.show()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, sequences=None, qubit_channel_maps=None, sequence_index=0, segment_index=0, normalized_amplitudes=False, plot_kwargs=None, parent=None, sharex=True, *args, **kwargs):
        super().__init__(parent)
        # QtWidgets.QMainWindow.__init__(self)
        self.sequences = sequences
        self.qubit_channel_maps = qubit_channel_maps
        self.current_sequence_index = sequence_index
        self.current_segment_index = segment_index

        self.frame = QtWidgets.QWidget()
        self.setCentralWidget(self.frame)
        self.frame.setLayout(QtWidgets.QVBoxLayout())
        self.frame.layout().setContentsMargins(0, 5, 0, 5)
        self.frame.layout().setSpacing(0)

        self.cbox_sequence_indices = QtWidgets.QComboBox()
        self.cbox_sequence_indices.addItems([seq.name for seq in self.sequences])
        if self.cbox_sequence_indices.count() != 0:
            self.cbox_sequence_indices.setCurrentIndex(self.current_sequence_index)
        self.cbox_segment_indices = QtWidgets.QComboBox()
        self.cbox_segment_indices.addItems(list(self.sequences[self.current_sequence_index].segments.keys()))
        if self.cbox_segment_indices.count() != 0:
            self.cbox_segment_indices.setCurrentIndex(self.current_segment_index)

        self.checkbox_normalized = QtWidgets.QCheckBox('normalized amp')
        self.checkbox_normalized.setChecked(normalized_amplitudes)

        self.cbox_seq_label = QtWidgets.QLabel("Sequence: ")
        self.cbox_seq_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cbox_seg_label = QtWidgets.QLabel("Segment: ")
        self.cbox_seg_label.setAlignment(QtCore.Qt.AlignCenter)

        self.qubit_list = set(qkey for q_ch_map in self.qubit_channel_maps for qkey in list(q_ch_map.keys()))
        self.selectbox_qubits = CheckableComboBox()
        self.selectbox_qubits.default_display_text = 'Select...'
        self.selectbox_qubits.addItems([qubit_name for qubit_name in self.qubit_list])
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

        self.dummy_widget = QtWidgets.QLabel("hello world")
        self.dummy_widget.setVisible(False)
        self.cid = self.view.mpl_connect('button_press_event', self.on_click)

        self.set_layout()
        self.setWindowTitle(self.get_window_title_string())

        self.cbox_sequence_indices.currentIndexChanged.connect(lambda: self.on_change(caller_id="sequence_change"))
        self.cbox_segment_indices.currentIndexChanged.connect(self.on_change)
        self.selectbox_qubits.currentIndexChanged.connect(self.on_change)
        self.checkbox_normalized.stateChanged.connect(self.on_change)
        self.selectbox_qubits.model().dataChanged.connect(self.on_change)




        self.showMaximized()

    def on_click(self, caller_id=None):
        self.dummy_widget.setVisible(True)

    def on_change(self, caller_id=None):
        self.current_segment_index = self.cbox_segment_indices.currentIndex()
        self.current_sequence_index = self.cbox_sequence_indices.currentIndex()
        if caller_id == "sequence_change":
            self.cbox_segment_indices.blockSignals(True)
            self.cbox_segment_indices.clear()
            self.cbox_segment_indices.addItems(list(self.sequences[self.current_sequence_index].segments.keys()))
            if self.cbox_segment_indices.count() != 0:
                self.cbox_segment_indices.setCurrentIndex(self.current_segment_index)
            self.cbox_segment_indices.blockSignals(False)
        oldfig = self.view.figure
        with plt.rc_context(gui_rc_params):
            fig, axes = self.get_experiment_plot()
        self.view.figure = fig
        # plt.close(oldfig)
        self.view.draw()
        self.toolbar = NavigationToolbar2QT(self.view, self.frame)
        self.toolbar.update()

        self.setWindowTitle(self.get_window_title_string())
        self.trigger_resize_event()

    def set_layout(self):
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
        input_layout.addWidget(self.dummy_widget)
        input_layout.setSpacing(50)

        self.frame.layout().addWidget(self.toolbar)
        self.frame.layout().addWidget(self.scroll)
        self.frame.layout().addLayout(input_layout)

    def get_window_title_string(self):
        title_string = ''.join(['Sequence ', str(self.current_sequence_index), ', Segment ', str(self.current_segment_index)])
        if len(self.selectbox_qubits.currentData()) != 0:
            for qubit in self.selectbox_qubits.currentData():
                ''.join([title_string, ", ", qubit])
        if self.checkbox_normalized.isChecked():
            ''.join([title_string, ' (normalized amplitude)'])
        return title_string

    #  figure out what part of the resize event is relevant
    def trigger_resize_event(self):
        # ugly hack to trigger a resize event (otherwise the figure in the canvas is not displayed properly)
        size = self.size()
        self.resize(self.width() + 1, self.height())
        self.resize(size)

    def get_experiment_plot(self, **kwargs):
        self.gui_kwargs.update({
            'normalized_amplitudes': self.checkbox_normalized.isChecked(),
            'sharex': self.sharex,
            'plot_kwargs': self.plot_kwargs,
        })
        with plt.rc_context(gui_rc_params):
            if len(self.selectbox_qubits.currentData()) == 0:
                return self.sequences[self.current_sequence_index].segments[self.current_segment_index].plot(**self.gui_kwargs, **kwargs)
            else:
                channel_map = None
                for chmap in self.qubit_channel_maps:
                    if set(self.selectbox_qubits.currentData()) == set(chmap.keys()):
                        channel_map = chmap
                # qubits = [qubit for qubit in self.qubit_channel_maps if qubit.name in self.selectbox_qubits.currentData()]
                return self.sequences[self.current_sequence_index].segments[self.current_segment_index].plot(channel_map=channel_map, **self.gui_kwargs, **kwargs)
