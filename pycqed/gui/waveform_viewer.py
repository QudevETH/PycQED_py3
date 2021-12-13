import sys
import matplotlib
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.gui.rc_params import GUI_RC_PARAMS
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox
from matplotlib.backend_bases import _Mode
from pycqed.gui import pulsar_shadow
from matplotlib.backend_tools import cursors
import multiprocessing as mp
from copy import deepcopy
from itertools import chain, combinations
from enum import Enum
import logging
log = logging.getLogger(__name__)

def powerset(iterable):
    """powerset([1,2,3]) --> (), (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class PLOT_OPTIONS(Enum):
    NORMALIZED = "normalized amp"
    SHAREX = "share x-axis"
    DEMODULATE = "demodulate"


class Object(object):
    pass


class WaveformViewer:
    """
    Spawns a GUI to interactively inspect the waveforms associated to the passed QuantumExperiment object on
    initialization

    """
    def __init__(self, quantum_experiment, sequence_index=0, segment_index=0,  rc_params=None, new_process=True, **kwargs):
        """
        Initialization of the WaveformViewer object. As we can only pass pickle-able variables to the child process,
        shadow objects that have the relevant properties of the QCoDeS instruments are created and passed
        instead.

        Args:
            quantum_experiment (QuantumExperiment): the experiment for which the waveforms should be plotted
            sequence_index (int): index of initially displayed sequence, default is 0
            segment_index (int): index of initially displayed segment, default is 0
            rc_params (dict): modify the rc parameters of the matplotlib plotting backend. By default (if rc_params=None
                is passed) the rc parameters in pycqed.gui.rc_params.GUI_RC_PARAMS are loaded, but they are updated by
                the parameters passed in the rc_params dictionary
            new_process (bool): if true the QApplication hosting the GUI will be spawned in a separate process
            **kwargs: are passed as plotting kwargs to the segment plotting method

        """
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            if mp.get_start_method() != 'spawn':
                log.warning('Child process should be spawned')
        if QtWidgets.__package__ not in ['PySide2']:
            log.warning('This GUI is optimized to run with the PySide2 Qt '
                     'binding')
        qubit_channel_maps = []
        for qset in powerset(quantum_experiment.qubits):
            if len(qset) != 0:
                channel_map = {}
                [channel_map.update(qb.get_channel_map()) for qb in qset]
                qubit_channel_maps.append(channel_map)
        sequences = deepcopy(quantum_experiment.sequences)
        pass_kwargs = deepcopy(kwargs)
        pass_kwargs.update({'sequence_index': sequence_index,
                            'segment_index': segment_index,
                            'rc_params': rc_params})

        if new_process:
            # pulsar object can't be pickled (and thus can't be passed to new process), pass PulsarShadow object instead
            p_shadow = pulsar_shadow.PulsarShadow(Pulsar.get_instance())
            for seq in sequences:
                seq.pulsar = p_shadow
                for segname, seg in seq.segments.items():
                    seg.pulsar = p_shadow
            self.process = mp.Process(name='pycqed_gui', target=self.start_qapp,args=(sequences, qubit_channel_maps, quantum_experiment.experiment_name), kwargs=pass_kwargs)
            self.process.daemon = False
            self.process.start()
        else:
            self.start_qapp(sequences, qubit_channel_maps, quantum_experiment.experiment_name, **pass_kwargs)

    def start_qapp(self, sequences, qubit_channel_maps, experiment_name, **kwargs):
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication.instance()
        w = WaveformViewerMainWindow(sequences, qubit_channel_maps, experiment_name, **kwargs)
        app.exec_()


def add_label_to_widget(widget, labeltext, parent=None):
    layout = QtWidgets.QHBoxLayout(parent=parent)
    label = QtWidgets.QLabel(labeltext, parent=parent)
    label.setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(label)
    layout.addWidget(widget)
    layout.setSpacing(0)
    return layout


class WaveformViewerMainWindow(QtWidgets.QMainWindow):

    def __init__(self, sequences, qubit_channel_maps, experiment_name, sequence_index=0, segment_index=0,  rc_params=None, *args, **kwargs):
        super().__init__(None)
        self.pulse_information_window = PulseInformationWindow()
        self.sequences = sequences
        self.qubit_channel_maps = qubit_channel_maps
        self.experiment_name = experiment_name
        self.current_sequence_index = sequence_index
        self.current_segment_index = segment_index
        self.rc_params = deepcopy(GUI_RC_PARAMS)
        if rc_params is not None:
            self.rc_params.update(rc_params)
        self._toolbar_memory = _Mode.NONE

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

        self.qubit_list = set(qkey for q_ch_map in self.qubit_channel_maps for qkey in list(q_ch_map.keys()))
        self.selectbox_qubits = CheckableComboBox()
        self.selectbox_qubits.default_display_text = 'Select...'
        self.selectbox_qubits.addItems(list(self.qubit_list))

        self.get_current_segment().resolve_segment()
        self.get_current_segment().gen_elements_on_awg()
        self.instrument_list = set(self.get_current_segment().elements_on_awg)
        self.selectbox_instruments = CheckableComboBox()
        self.selectbox_instruments.default_display_text = 'Select...'
        self.selectbox_instruments.addItems(list(self.instrument_list))

        self.plot_options = CheckableComboBox()
        self.plot_options.default_display_text = 'Select...'
        self.plot_options.addItems([PLOT_OPTIONS.DEMODULATE.value, PLOT_OPTIONS.NORMALIZED.value, PLOT_OPTIONS.SHAREX.value])
        self.plot_options.model().item(self.plot_options.findText(PLOT_OPTIONS.SHAREX.value)).setCheckState(QtCore.Qt.Checked)

        if kwargs:
            self.plot_kwargs = deepcopy(kwargs)
        else:
            self.plot_kwargs = None

        self.gui_kwargs = {
            'show_and_close': False,
            'figtitle_kwargs': {'y': 0.98},
        }
        fig, axes = self.get_experiment_plot()
        self.view = FigureCanvasQTAgg(fig)
        self.view.draw()
        self.scroll = QtWidgets.QScrollArea(self.frame)
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(True)
        self.view.setMinimumSize(800, 400)
        self.toolbar = NavigationToolbar2QT(self.view, self.frame)

        self.toggle_selection_button = QtWidgets.QPushButton('&Select Waveform')
        self.toggle_selection_button.setCheckable(True)
        self.toggle_selection_button.toggled.connect(self.on_toggle_selection)
        self.cid_bpe = self.view.mpl_connect('pick_event', self.on_pick)

        self.set_layout()
        self.setWindowTitle(self.get_window_title_string())

        self.cbox_sequence_indices.currentIndexChanged.connect(lambda: self.on_change(caller_id='sequence_change'))
        self.cbox_segment_indices.currentIndexChanged.connect(lambda: self.on_change(caller_id='segment_change'))
        self.toolbar.actionTriggered[QtWidgets.QAction].connect(self.on_toolbar_selection)
        self.plot_options.model().dataChanged.connect(self.on_change)
        self.selectbox_qubits.model().dataChanged.connect(lambda: self.on_change(caller_id='qubits_change'))
        self.selectbox_instruments.model().dataChanged.connect(self.on_change)

        self.showMaximized()

    def on_toolbar_selection(self):
        if self.toggle_selection_button.isChecked():
            self.toggle_selection_button.setChecked(False)

    def on_toggle_selection(self):
        if not self.toggle_selection_button.isChecked():
            self.view.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.toolbar.mode = self._toolbar_memory
            if self.toolbar.mode in [_Mode.PAN, _Mode.ZOOM]:
                self.view.widgetlock(self.toolbar)
            self.toolbar._update_buttons_checked()
            return

        # deselect the current toolbar action
        # if the _lastCursor is not set to cursor.POINTER the mouse will be changed to ArrowCursor after the next
        # mouse move event
        self.toolbar._lastCursor = cursors.POINTER
        self.view.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._toolbar_memory = self.toolbar.mode
        self.toolbar.mode = _Mode.NONE
        self.view.widgetlock.release(self.toolbar)
        self.toolbar._update_buttons_checked()

    def on_pick(self, event):
        if not self.toggle_selection_button.isChecked():
            return
        t = event.mouseevent.xdata/1e6
        plist = [p for p in self.get_current_segment().resolved_pulses]
        for p in self.get_current_segment().extra_pulses:
            plist.append(Object())
            plist[-1].pulse_obj = p
        plist = [p for p in plist if p.pulse_obj.algorithm_time() < t < p.pulse_obj.algorithm_time() + p.pulse_obj.length]
        pulse_information = []
        for p in plist:
            artist_metadata_dict = event.artist.pycqed_metadata
            checks = [
                any([ch.find(artist_metadata_dict['channel']) >= 0 for ch in p.pulse_obj.channels]),
                any([ch.find(artist_metadata_dict['instrument']) >= 0 for ch in p.pulse_obj.channels]),
                artist_metadata_dict['codeword'] == (p.pulse_obj.codeword if p.pulse_obj.codeword != 'no_codeword' else ''),
                artist_metadata_dict['element_name'] == p.pulse_obj.element_name,
            ]
            if all(checks):
                pulse_dict = {**p.__dict__, **{('pulse_obj.' + k): v for k, v in p.pulse_obj.__dict__.items()}}
                pulse_dict.update({'pulse_obj': p.pulse_obj.__class__.__name__})
                pulse_information.append(pulse_dict)

        self.pulse_information_window.display_window(pulse_information)
        self.toggle_selection_button.setChecked(False)

    def on_change(self, caller_id=None):
        QtWidgets.QApplication.instance().setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.current_segment_index = self.cbox_segment_indices.currentIndex()
        self.current_sequence_index = self.cbox_sequence_indices.currentIndex()
        if caller_id == 'sequence_change':
            self.reload_segment_keys()
            self.reload_instrument_keys()
        if caller_id == 'segment_change' or caller_id == 'qubits_change':
            self.reload_instrument_keys()

        oldfig = self.view.figure
        fig, axes = self.get_experiment_plot()
        self.view.figure = fig
        fig.set_canvas(self.view)
        plt.close(oldfig)
        self.view.draw()
        self.cid_bpe = self.view.mpl_connect('pick_event', self.on_pick)
        self.reinitialize_toolbar()

        self.setWindowTitle(self.get_window_title_string())
        self.trigger_resize_event()
        QtWidgets.QApplication.instance().restoreOverrideCursor()

    def reload_segment_keys(self):
        self.cbox_segment_indices.blockSignals(True)
        self.cbox_segment_indices.clear()
        self.cbox_segment_indices.addItems(list(self.sequences[self.current_sequence_index].segments.keys()))
        if self.cbox_segment_indices.count() != 0:
            self.cbox_segment_indices.setCurrentIndex(self.current_segment_index)
        self.cbox_segment_indices.blockSignals(False)

    def reload_instrument_keys(self):
        selected_instruments = self.selectbox_instruments.currentData()
        self.selectbox_instruments.blockSignals(True)
        self.selectbox_instruments.clear()
        self.instrument_list = set(self.get_current_segment().elements_on_awg)
        active_channel_map = self.get_active_channel_map()
        if active_channel_map:
            qubit_instrument_set = set(instrument_channel.split('_')[0] for qubit, instrument_channels in active_channel_map.items() for instrument_channel in instrument_channels)
            self.instrument_list = self.instrument_list.intersection(qubit_instrument_set)
        self.selectbox_instruments.addItems(list(self.instrument_list))
        if self.selectbox_instruments.count() != 0:
            for selected_instrument in selected_instruments:
                if selected_instrument in self.instrument_list:
                    index = self.selectbox_instruments.findText(selected_instrument)
                    self.selectbox_instruments.model().item(index).setCheckState(QtCore.Qt.Checked)
        self.selectbox_instruments.updateText()
        self.selectbox_instruments.blockSignals(False)

    def reinitialize_toolbar(self):
        # upon changing the figure associated to the FigureCanvasQTAgg instance self.view the zoom and pan tools of the
        # toolbar can't be used to manipulate the figure unless the corresponding callback methods are reinitialized.
        # Reinitialization of callback methods corresponds to the __init__ method of the FigureCanvasQTAgg class.
        self.toolbar._id_press = self.toolbar.canvas.mpl_connect(
            'button_press_event', self.toolbar._zoom_pan_handler)
        self.toolbar._id_release = self.toolbar.canvas.mpl_connect(
            'button_release_event', self.toolbar._zoom_pan_handler)
        self.toolbar._id_drag = self.toolbar.canvas.mpl_connect(
            'motion_notify_event', self.toolbar.mouse_move)
        self.toolbar._pan_info = None
        self.toolbar._zoom_info = None
        self.toolbar.update()

    def set_layout(self):
        input_widgets = [add_label_to_widget(self.cbox_sequence_indices, 'Sequence: '),
                         add_label_to_widget(self.cbox_segment_indices, 'Segment: '),
                         add_label_to_widget(self.selectbox_qubits, 'Qubits: '),
                         add_label_to_widget(self.selectbox_instruments, 'Instruments: '),
                         add_label_to_widget(self.plot_options, 'Plot Options: '),
                         ]

        input_layout = QtWidgets.QHBoxLayout()
        for input_widget in input_widgets:
            input_layout.addLayout(input_widget)
        input_layout.addWidget(self.toggle_selection_button)
        input_layout.setSpacing(50)

        self.frame.layout().addWidget(self.toolbar)
        self.frame.layout().addWidget(self.scroll)
        self.frame.layout().addLayout(input_layout)

    def get_window_title_string(self):
        return self.experiment_name

    def trigger_resize_event(self):
        # ugly hack to trigger a resize event (otherwise the figure in the canvas is not displayed properly)
        size = self.size()
        self.resize(self.width() + 1, self.height())
        self.resize(size)

    def get_experiment_plot(self, **kwargs):
        self.gui_kwargs.update({
            'normalized_amplitudes': self.is_checked(PLOT_OPTIONS.NORMALIZED),
            'sharex': self.is_checked(PLOT_OPTIONS.SHAREX),
            'demodulate': self.is_checked(PLOT_OPTIONS.DEMODULATE),
            'plot_kwargs': self.plot_kwargs,
        })
        with plt.rc_context(self.rc_params):

            instruments = set(self.selectbox_instruments.currentData())
            if not instruments:
                instruments = None
            channel_map = self.get_active_channel_map()
            fig, axes = self.get_current_segment().plot(channel_map=channel_map, instruments=instruments, **self.gui_kwargs, **kwargs)

            for ax in axes:
                for ch in ax[0].get_children():
                    if isinstance(ch, matplotlib.lines.Line2D):
                        ch.set_picker(3)
            return fig, axes

    def get_current_segment(self):
        return self.sequences[self.current_sequence_index][self.current_segment_index]

    def get_active_channel_map(self):
        channel_map = None
        for chmap in self.qubit_channel_maps:
            if set(self.selectbox_qubits.currentData()) == set(chmap.keys()):
                channel_map = chmap
                break
        return channel_map

    def is_checked(self, plot_option):
        if not isinstance(plot_option, PLOT_OPTIONS):
            raise TypeError('plot_option must be of type PLOT_OPTIONS')
        return self.plot_options.model().item(self.plot_options.findText(plot_option.value)).checkState() == QtCore.Qt.Checked

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.toggle_selection_button.setChecked(True)
            event.accept()
        elif event.key() == QtCore.Qt.Key_W and event.modifiers() == QtCore.Qt.ControlModifier:
            self.close()
            event.accept()

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.toggle_selection_button.setChecked(False)
            event.accept()


class PulseInformationWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._data = []
        self.setWindowTitle('Pulse Information')
        layout = QtWidgets.QVBoxLayout()
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setHorizontalHeaderLabels(['label', 'value'])
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(450, 800)

    def display_window(self, pass_information=None):
        if pass_information is not None:
            self._data = deepcopy(pass_information)
            self.set_table()
        self.show()
        self.activateWindow()

    def set_table(self):
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        if len(self._data) != 0:
            self.table.setRowCount(len(self._data)*len(self._data[0]))
            self.setWindowTitle('Pulse Information '+', '.join([pulse_dict['pulse_obj.element_name'] for pulse_dict in self._data]))
        else:
            self.table.setRowCount(0)
            self.setWindowTitle('Pulse Information')
        for i, pulse_dict in enumerate(self._data):
            for j, key in enumerate(pulse_dict):
                col1 = QtWidgets.QTableWidgetItem(key)
                col2 = QtWidgets.QTableWidgetItem(repr(pulse_dict[key]))
                self.table.setItem(i*len(pulse_dict)+j, 0, col1)
                self.table.setItem(i*len(pulse_dict)+j, 1, col2)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_W and event.modifiers() == QtCore.Qt.ControlModifier:
            self.close()
            event.accept()
