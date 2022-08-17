import sys
import matplotlib
from pycqed.gui import qt_compat as qt
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from pycqed.measurement.waveform_control.pulsar import Pulsar
from pycqed.gui.rc_params import GUI_RC_PARAMS
from pycqed.gui.qt_widgets.checkable_combo_box import CheckableComboBox
from pycqed.gui import gui_utilities as g_utils
from matplotlib.backend_bases import _Mode
from pycqed.gui import pulsar_shadow
from matplotlib.backend_tools import cursors
import multiprocessing as mp
from copy import deepcopy
from enum import Enum
from pycqed.gui import gui_process
import logging
log = logging.getLogger(__name__)


class PLOT_OPTIONS(Enum):
    NORMALIZED = "normalized amp"
    SHAREX = "share x-axis"
    DEMODULATE = "demodulate"


class WaveformViewer:
    """
    Spawns a GUI to interactively inspect the waveforms associated to the
    passed QuantumExperiment object on initialization

    """
    def __init__(self, quantum_experiment, sequence_index=0,
                 segment_index=0, rc_params=None, new_process=False,
                 active_qapp=False, **kwargs):
        """
        Initialization of the WaveformViewer object. As we can only pass
        pickle-able variables to the child process, shadow objects that have
        the relevant properties of the QCoDeS instruments are created and
        passed instead.

        Args:
            quantum_experiment (QuantumExperiment): the experiment for
            which the waveforms should be plotted
            sequence_index (int): index of initially displayed sequence,
            default is 0
            segment_index (int): index of initially displayed segment,
            default is 0
            rc_params (dict): modify the rc parameters of the matplotlib
            plotting backend. By default (if rc_params=None is passed) the
            rc parameters in pycqed.gui.rc_params.GUI_RC_PARAMS are loaded,
            but they are updated with the parameters passed in the rc_params
            dictionary
            new_process (bool): if True the QApplication hosting the GUI
            will be spawned in a separate process
            active_qapp (bool): if True, the exec method to start the qt
            event loop won't be called. Useful if a window should be spawned
            from a running qt application
            **kwargs: are passed as plotting kwargs to the segment
            plotting method

        """
        if qt.QtWidgets.__package__ not in ['PySide2']:
            log.warning('This GUI is optimized to run with the PySide2 Qt '
                        'binding')
        if not qt.QtWidgets.QApplication.instance():
            self.app = qt.QtWidgets.QApplication(sys.argv)
        else:
            self.app = qt.QtWidgets.QApplication.instance()
        g_utils.handle_matplotlib_backends(self.app)
        self.experiment_name = quantum_experiment.experiment_name
        self.qubit_channel_maps = []
        # get all qubit objects corresponding to the qubit names returned by
        # quantum_experiment.get_qubits[1]
        qubits = quantum_experiment.dev.get_qubits(
            quantum_experiment.get_qubits()[1])
        for qset in g_utils.powerset(qubits):
            if len(qset) != 0:
                channel_map = {}
                [channel_map.update(qb.get_channel_map()) for qb in qset]
                self.qubit_channel_maps.append(channel_map)
        self.sequences = deepcopy(quantum_experiment.sequences)
        self.pass_kwargs = {}
        self.pass_kwargs.update(kwargs)
        self.pass_kwargs.update({
            'sequence_index': sequence_index,
            'segment_index': segment_index,
            'rc_params': rc_params if rc_params is not None else {}
        })
        self.new_process = new_process
        self._is_init = True
        if self.new_process:
            self._prepare_new_process()
            self.main_window = None
        else:
            self._prepare_main_window()
            self.p_shadow = None
        self.spawn_waveform_viewer(active_qapp=active_qapp)

    def _prepare_main_window(self):
        self.main_window = WaveformViewerMainWindow(
            self.sequences, self.qubit_channel_maps,
            self.experiment_name, **self.pass_kwargs
        )

    def _prepare_new_process(self):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            if mp.get_start_method() != 'spawn':
                log.warning('Child process should be spawned')
        # pulsar object can't be pickled (and thus can't be passed to
        # new process), pass PulsarShadow object instead
        self.p_shadow = pulsar_shadow.PulsarShadow(Pulsar.get_instance())
        for seq in self.sequences:
            seq.pulsar = self.p_shadow
            for seg in seq.segments.values():
                seg.pulsar = self.p_shadow

    def _update_kwargs(self, rc_params, **kwargs):
        if rc_params is not None:
            if self.main_window is not None:
                self.main_window.rc_params.update(rc_params)
            self.pass_kwargs['rc_params'].update(rc_params)
        if kwargs:
            if self.main_window is not None:
                self.main_window.plot_kwargs = deepcopy(kwargs)
            self.pass_kwargs.update(kwargs)
        if any([kwargs, rc_params is not None]) \
                and self.main_window is not None:
            self.main_window.on_change()

    def spawn_waveform_viewer(self, rc_params=None, new_process=None,
                              active_qapp=False, **kwargs):
        if new_process is not None:
            self.new_process = new_process
        self._update_kwargs(rc_params=rc_params, **kwargs)
        if self.new_process:
            if self.p_shadow is None:
                self._prepare_new_process()
            gui_process.create_waveform_viewer_process(
                args=(self.sequences, self.qubit_channel_maps,
                      self.experiment_name, self.pass_kwargs),
                qt_lib=qt.QtWidgets.__package__
            )
        else:
            if self.main_window is None:
                self._prepare_main_window()
            self._start_qapp(active_qapp=active_qapp)

    def _start_qapp(self, active_qapp=False):
        self.main_window.showMaximized()
        if self._is_init:
            # plots are not properly displayed if they are added to the
            # FigureCanvasQTAgg before the main_window is shown. To display
            # the plots correctly one needs to add them after the gui window
            # has been spawned
            self.main_window.on_change()
            self._is_init = False
        self.main_window.activateWindow()
        qt.QtWidgets.QApplication.instance().processEvents()
        self.main_window.trigger_resize_event()
        qt.QtWidgets.QApplication.restoreOverrideCursor()
        if not active_qapp:
            self.app._matplotlib_backend = \
                sys.modules.get('matplotlib').get_backend()
            sys.modules.get('matplotlib').use('Agg')
            self.app.exec_()


class WaveformViewerMainWindow(qt.QtWidgets.QWidget):

    def __init__(self, sequences, qubit_channel_maps, experiment_name,
                 sequence_index=0, segment_index=0,  rc_params=None,
                 *args, **kwargs):
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
        if kwargs:
            self.plot_kwargs = kwargs
        else:
            self.plot_kwargs = None
        self.gui_kwargs = {
            'show_and_close': False,
            'figtitle_kwargs': {'y': 0.98},
        }

        self.setLayout(qt.QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0, 5, 0, 5)
        self.layout().setSpacing(0)
        self.cbox_sequence_indices = qt.QtWidgets.QComboBox()
        self.cbox_sequence_indices.addItems(
            [seq.name for seq in self.sequences])
        if self.cbox_sequence_indices.count() != 0:
            self.cbox_sequence_indices.setCurrentIndex(
                self.current_sequence_index)

        self.cbox_segment_indices = qt.QtWidgets.QComboBox()
        self.cbox_segment_indices.addItems(
            list(self.sequences[self.current_sequence_index].segments.keys()))
        if self.cbox_segment_indices.count() != 0:
            self.cbox_segment_indices.setCurrentIndex(
                self.current_segment_index)

        self.qubit_list = set(
            qkey for q_ch_map in self.qubit_channel_maps
            for qkey in list(q_ch_map.keys())
        )
        self.selectbox_qubits = CheckableComboBox()
        self.selectbox_qubits.default_display_text = 'Select...'
        self.selectbox_qubits.addItems(list(self.qubit_list))

        self.get_current_segment().resolve_segment(allow_overlap=True)
        self.get_current_segment().gen_elements_on_awg()
        self.instrument_list = set(self.get_current_segment().elements_on_awg)
        self.selectbox_instruments = CheckableComboBox()
        self.selectbox_instruments.default_display_text = 'Select...'
        self.selectbox_instruments.addItems(list(self.instrument_list))

        self.plot_options = CheckableComboBox()
        self.plot_options.default_display_text = 'Select...'
        self.plot_options.addItems(
            [PLOT_OPTIONS.DEMODULATE.value,
             PLOT_OPTIONS.NORMALIZED.value,
             PLOT_OPTIONS.SHAREX.value]
        )
        self.plot_options.model().item(self.plot_options.findText(
            PLOT_OPTIONS.SHAREX.value)).setCheckState(
            qt.QtCore.Qt.CheckState.Checked)

        # add an empty figure on initialization, because the FigureCanvas
        # does not display the figure properly if the main window has not
        # yet been made visible. The correct figure is added by the
        # WaveformViewer class
        fig = plt.figure()
        self.view = qt.FigureCanvasQTAgg(fig)
        self.view.draw()
        self.scroll = qt.QtWidgets.QScrollArea(self)
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(True)
        self.view.setMinimumSize(800, 400)
        self.toolbar = qt.NavigationToolbar2QT(self.view, self)

        self.toggle_selection_button = qt.QtWidgets.QPushButton(
            '&Select Waveform')
        self.toggle_selection_button.setCheckable(True)
        self.toggle_selection_button.toggled.connect(self.on_toggle_selection)
        self.cid_bpe = self.view.mpl_connect('pick_event', self.on_pick)

        self.input_layout_widget = qt.QtWidgets.QWidget()

        self.set_layout()
        self.setWindowTitle(self.get_window_title_string())

        self.cbox_sequence_indices.currentIndexChanged.connect(
            lambda: self.on_change(caller_id='sequence_change'))
        self.cbox_segment_indices.currentIndexChanged.connect(
            lambda: self.on_change(caller_id='segment_change'))
        self.toolbar.actionTriggered[qt.QAction].connect(
            self.on_toolbar_selection)
        self.plot_options.model().dataChanged.connect(self.on_change)
        self.selectbox_qubits.model().dataChanged.connect(
            lambda: self.on_change(caller_id='qubits_change'))
        self.selectbox_instruments.model().dataChanged.connect(self.on_change)

    def set_layout(self):
        input_widgets = [
            g_utils.add_label_to_widget(
                self.cbox_sequence_indices, 'Sequence: '),
            g_utils.add_label_to_widget(
                self.cbox_segment_indices, 'Segment: '),
            g_utils.add_label_to_widget(
                self.selectbox_qubits, 'Qubits: '),
            g_utils.add_label_to_widget(
                self.selectbox_instruments, 'Instruments: '),
            g_utils.add_label_to_widget(
                self.plot_options, 'Plot Options: '),
        ]

        self.input_layout_widget.setLayout(qt.QtWidgets.QHBoxLayout())
        for input_widget in input_widgets:
            self.input_layout_widget.layout().addLayout(input_widget)
        self.input_layout_widget.layout().addWidget(
            self.toggle_selection_button)
        self.input_layout_widget.layout().setSpacing(50)

        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.scroll)
        self.layout().addWidget(self.input_layout_widget)

    def on_toolbar_selection(self):
        if self.toggle_selection_button.isChecked():
            self.toggle_selection_button.setChecked(False)

    def on_toggle_selection(self):
        if not self.toggle_selection_button.isChecked():
            self.view.setCursor(qt.QtGui.QCursor(
                qt.QtCore.Qt.CursorShape.ArrowCursor))
            self.toolbar.mode = self._toolbar_memory
            if self.toolbar.mode in [_Mode.PAN, _Mode.ZOOM]:
                self.view.widgetlock(self.toolbar)
            self.toolbar._update_buttons_checked()
            return

        # deselect the current toolbar action
        # if the _lastCursor is not set to cursor.POINTER the mouse will be
        # changed to ArrowCursor after the next mouse move event
        self.toolbar._lastCursor = cursors.POINTER
        self.view.setCursor(qt.QtGui.QCursor(
            qt.QtCore.Qt.CursorShape.PointingHandCursor))
        self._toolbar_memory = self.toolbar.mode
        self.toolbar.mode = _Mode.NONE
        self.view.widgetlock.release(self.toolbar)
        self.toolbar._update_buttons_checked()

    def on_pick(self, event):
        if not (self.toggle_selection_button.isChecked() and
                event.mouseevent.button == MouseButton.LEFT):
            return
        t = event.mouseevent.xdata/1e6
        plist = [p for p in self.get_current_segment().resolved_pulses]
        for p in self.get_current_segment().extra_pulses:
            plist.append(g_utils.Object())
            plist[-1].pulse_obj = p
        plist = [p for p in plist
                 if p.pulse_obj.algorithm_time() < t <
                 p.pulse_obj.algorithm_time() + p.pulse_obj.length]
        pulse_information = []
        for p in plist:
            artist_metadata_dict = event.artist.pycqed_metadata
            checks = [
                any([ch.find(artist_metadata_dict['channel']) >= 0
                     for ch in p.pulse_obj.channels]),
                any([ch.find(artist_metadata_dict['instrument']) >= 0
                     for ch in p.pulse_obj.channels]),
                artist_metadata_dict['codeword'] == (p.pulse_obj.codeword
                                                     if p.pulse_obj.codeword
                                                     != 'no_codeword'
                                                     else ''),
                artist_metadata_dict['element_name'] == p.pulse_obj.element_name,
            ]
            if all(checks):
                pulse_dict = {
                    **p.__dict__,
                    **{('pulse_obj.' + k): v
                       for k, v in p.pulse_obj.__dict__.items()}}
                pulse_dict.update({'pulse_obj': p.pulse_obj.__class__.__name__})
                pulse_information.append(pulse_dict)

        self.pulse_information_window.display_window(pulse_information)
        self.toggle_selection_button.setChecked(False)

    def on_change(self, caller_id=None):
        with g_utils.set_wait_cursor(self):
            self.input_layout_widget.setEnabled(False)
            self.current_segment_index = self.cbox_segment_indices.currentIndex()
            self.current_sequence_index = self.cbox_sequence_indices.currentIndex()
            if caller_id == 'sequence_change':
                self.reload_segment_keys()
                self.reload_instrument_keys()
            if caller_id == 'segment_change' or caller_id == 'qubits_change':
                self.reload_instrument_keys()
            fig, axes = self.get_experiment_plot()
            qt.QtWidgets.QApplication.instance().processEvents()
            oldfig = self.view.figure
            self.view.figure = fig
            fig.set_canvas(self.view)
            plt.close(oldfig)
            self.view.draw()
            self.cid_bpe = self.view.mpl_connect('pick_event', self.on_pick)
            self.reinitialize_toolbar()

            self.setWindowTitle(self.get_window_title_string())
            self.trigger_resize_event()
            self.input_layout_widget.setEnabled(True)

    def reload_segment_keys(self):
        self.cbox_segment_indices.blockSignals(True)
        self.cbox_segment_indices.clear()
        self.cbox_segment_indices.addItems(
            list(self.sequences[self.current_sequence_index].segments.keys()))
        if self.cbox_segment_indices.count() != 0:
            self.cbox_segment_indices.setCurrentIndex(
                self.current_segment_index)
        self.cbox_segment_indices.blockSignals(False)

    def reload_instrument_keys(self):
        selected_instruments = self.selectbox_instruments.currentData()
        self.selectbox_instruments.blockSignals(True)
        self.selectbox_instruments.clear()
        self.get_current_segment().gen_elements_on_awg()
        self.instrument_list = set(self.get_current_segment().elements_on_awg)
        active_channel_map = self.get_active_channel_map()
        if active_channel_map is not None:
            qubit_instrument_set = set(
                instrument_channel.split('_')[0]
                for qubit, instrument_channels in active_channel_map.items()
                for instrument_channel in instrument_channels
            )
            self.instrument_list = self.instrument_list.intersection(
                qubit_instrument_set)
        self.selectbox_instruments.addItems(list(self.instrument_list))
        if self.selectbox_instruments.count() != 0:
            for selected_instrument in selected_instruments:
                if selected_instrument in self.instrument_list:
                    index = self.selectbox_instruments.findText(
                        selected_instrument)
                    self.selectbox_instruments.model().item(
                        index).setCheckState(qt.QtCore.Qt.CheckState.Checked)
        self.selectbox_instruments.updateText()
        self.selectbox_instruments.blockSignals(False)

    def reinitialize_toolbar(self):
        # upon changing the figure associated to the FigureCanvasQTAgg
        # instance self.view, the zoom and pan tools of the toolbar can't be
        # used to manipulate the figure unless the corresponding callback
        # methods are reinitialized. Reinitialization of callback methods
        # corresponds to the __init__ method of the FigureCanvasQTAgg class.
        self.toolbar._id_press = self.toolbar.canvas.mpl_connect(
            'button_press_event', self.toolbar._zoom_pan_handler)
        self.toolbar._id_release = self.toolbar.canvas.mpl_connect(
            'button_release_event', self.toolbar._zoom_pan_handler)
        self.toolbar._id_drag = self.toolbar.canvas.mpl_connect(
            'motion_notify_event', self.toolbar.mouse_move)
        self.toolbar._pan_info = None
        self.toolbar._zoom_info = None
        self.toolbar.update()

    def get_window_title_string(self):
        return self.experiment_name

    def trigger_resize_event(self):
        # ugly hack to trigger a resize event (otherwise the figure in the
        # canvas is not displayed properly)
        size = self.size()
        self.resize(self.width() + 1, self.height())
        self.resize(size)

    def get_experiment_plot(self):
        qt.QtWidgets.QApplication.instance().processEvents()
        channel_map, instruments = self.prepare_experiment_plot()
        with plt.rc_context(self.rc_params):
            fig, axes = self.get_current_segment().plot(
                channel_map=channel_map,
                trigger_groups=instruments,
                **self.gui_kwargs
            )
            add_picker_to_line_artists(axes)
            return fig, axes

    def prepare_experiment_plot(self):
        self.gui_kwargs.update({
            'normalized_amplitudes': self.is_checked(PLOT_OPTIONS.NORMALIZED),
            'sharex': self.is_checked(PLOT_OPTIONS.SHAREX),
            'demodulate': self.is_checked(PLOT_OPTIONS.DEMODULATE),
            'plot_kwargs': self.plot_kwargs,
        })
        instruments = set(self.selectbox_instruments.currentData())
        if not instruments:
            instruments = None
        channel_map = self.get_active_channel_map()
        return channel_map, instruments

    def get_current_segment(self):
        return self.sequences[self.current_sequence_index][
            self.current_segment_index]

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
        return self.plot_options.model().item(self.plot_options.findText(
            plot_option.value)).checkState() == qt.QtCore.Qt.CheckState.Checked

    def keyPressEvent(self, event):
        if event.key() == qt.QtCore.Qt.Key.Key_Control:
            self.toggle_selection_button.setChecked(True)
            event.accept()
        elif event.key() == qt.QtCore.Qt.Key.Key_W \
                and event.modifiers() == \
                qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            self.close()
            event.accept()

    def keyReleaseEvent(self, event):
        if event.key() == qt.QtCore.Qt.Key.Key_Control:
            self.toggle_selection_button.setChecked(False)
            event.accept()


def add_picker_to_line_artists(axes):
    for ax in axes:
        for ch in ax[0].get_children():
            if isinstance(ch, matplotlib.lines.Line2D):
                ch.set_picker(True)
                ch.set_pickradius(3)


class PulseInformationWindow(qt.QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._data = []
        self.setWindowTitle('Pulse Information')
        layout = qt.QtWidgets.QVBoxLayout()
        self.table = qt.QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            0, qt.QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            1, qt.QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setHorizontalHeaderLabels(['label', 'value'])
        self.table.setEditTriggers(
            qt.QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(600, 800)

    def display_window(self, pass_information=None):
        if pass_information is not None:
            self._data = pass_information
            self.set_table()
        self.show()
        self.activateWindow()

    def set_table(self):
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        if len(self._data) != 0:
            self.table.setRowCount(len(self._data)*len(self._data[0]))
            self.setWindowTitle(
                'Pulse Information '
                + ', '.join([pulse_dict['pulse_obj.element_name']
                             for pulse_dict in self._data])
            )
        else:
            self.table.setRowCount(0)
            self.setWindowTitle('Pulse Information')
        for i, pulse_dict in enumerate(self._data):
            for j, key in enumerate(pulse_dict):
                col1 = qt.QtWidgets.QTableWidgetItem(key)
                col2 = qt.QtWidgets.QTableWidgetItem(repr(pulse_dict[key]))
                self.table.setItem(i*len(pulse_dict)+j, 0, col1)
                self.table.setItem(i*len(pulse_dict)+j, 1, col2)

    def keyPressEvent(self, event):
        if event.key() == qt.QtCore.Qt.Key.Key_W \
                and event.modifiers() == qt.QtCore.Qt.ControlModifier:
            self.close()
            event.accept()
