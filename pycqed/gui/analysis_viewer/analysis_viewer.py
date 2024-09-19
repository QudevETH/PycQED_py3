"""GUI module to view plots from `base_analysis.BaseDataAnalysis` objects.

Enables scrolling through all experiments in the data directory that have job
string saved in their HDF5 file (skips unsupported experiments).
"""
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from pycqed.analysis import analysis_toolbox
from pycqed.analysis_v2.base_analysis import BaseDataAnalysis
from pycqed.gui import gui_utilities
from pycqed.gui.analysis_viewer.iterators import (ExperimentIterator,
                                                  BidirectionalIterator)
from pycqed.gui.gui_utilities import TriggerResizeEventMixin, \
    reinitialize_toolbar
from pycqed.gui.qt_compat import QtWidgets, QtCore, QtGui
from pycqed.gui.rc_params import gui_rc_params


class AnalysisViewer(object):
    """Class that instantiates the GUI and displays it.

    This class sets up `MainWindow` and displays is. It acts more like a
    wrapper while `MainWindow` is responsible for the actual functionality.

    Typical usage example 1:
        viewer = AnalysisViewer()
        viewer.show()

    Typical usage example 2:
        viewer = AnalysisViewer('20230912_202101')
        viewer.show()

    Attributes:
        _app: `QtWidgets.QApplication` object that runs the GUI.
        _main_window: `MainWindow` object that governs all the functionality.
    """

    def __init__(self, timestamp: str = None, rc_params: dict = None):
        """Initializes the instance for GUI.

        Args:
            timestamp: str of format "YYYYmmdd_HHMMSS" ("%Y%m%d_%H%M%S"). If
                provided, `AnalysisViewer` grabs that specific experiment to
                show first. Otherwise, it shows the most recent one.
            rc_params: dict of matplotlib rcParams. Modifies rcParams only
                for the `AnalysisViewer` plots.
        """
        # Set GUI app.
        if not QtWidgets.QApplication.instance():
            self._app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
        else:
            self._app: QtWidgets.QApplication = (
                QtWidgets.QApplication.instance())

        # Set the current analysis object.
        file_path = self.resolve_file_path(timestamp)
        base_data_analysis = (
            BaseDataAnalysis.get_analysis_object_from_hdf5_file_path(file_path)
        )

        # Get daystamps and setup iterator with them.
        daystamps = analysis_toolbox.get_all_daystamps(analysis_toolbox.datadir)
        experiment_iterator = ExperimentIterator(daystamps)

        # Create MainWindow which is responsible for the whole functionality.
        if isinstance(base_data_analysis, BaseDataAnalysis):
            self._main_window: MainWindow = MainWindow(base_data_analysis,
                                                       experiment_iterator,
                                                       rc_params)
        else:
            raise RuntimeError("Variable base_data_analysis should be of "
                               "instance BaseDataAnalysis. Got "
                               f"{type(base_data_analysis)} instead.")

    @staticmethod
    def resolve_file_path(timestamp: str = None) -> str:
        """Gets the path of measurement for provided or the most recent
        `timestamp`.

        Args:
            timestamp: str of format "YYYYmmdd_HHMMSS" ("%Y%m%d_%H%M%S").

        Returns:
            str: path to measurement data HDF5 file.
        """
        if timestamp:
            timestamp = '_'.join(analysis_toolbox.verify_timestamp(timestamp))
            file_path = analysis_toolbox.measurement_filename(
                analysis_toolbox.data_from_time(timestamp)
            )
        else:
            file_path = analysis_toolbox.measurement_filename(
                analysis_toolbox.latest_data()
            )
        return file_path

    def show(self):
        """Displays the GUI and handles matplotlib backend change and reset."""
        # Setup matplotlib backend
        gui_utilities.handle_matplotlib_backends(self._app)
        sys.modules.get('matplotlib').use('Agg')
        # Show the window
        self._main_window.show()
        self._app.exec_()


class MainWindow(TriggerResizeEventMixin, QtWidgets.QMainWindow):
    """MainWindow of `AnalysisViewer` responsible for displaying plots.

    Iterates over `BaseDataAnalysis` objects from given `ExperimentIterator` and
    displays plots from current `BaseDataAnalysis` one by one. User can
    navigate through plots with arrow keys: up/down navigates between
    experiments, right/left navigates between plots in the current experiment.

    Attributes:
        rc_params: dict of matplotlib rcParams. Modifies rcParams only
            for the `AnalysisViewer` plots.
        _base_data_analysis: `BaseDataAnalysis` object whose plots will be
            displayed upon initialization.
        _experiment_iterator: `ExperimentIterator` object which is responsible
            for iterating through `BaseDataAnalysis` object whose plots are
            displayed.
        _plot_iterator: `BidirectionalIterator` object to iterate over plots
            from current experiment (`BaseDataAnalysis` object).
        _canvas: `FigureCanvasQTAgg` object which is used to display plots.
        _toolbar: `NavigationToolbar` object for plot control (like zooming,
            panning, etc.)
        _figure: `matplotlib.figure.Figure` object of the current plot.
        _axes: `matplotlib.axes.Axes` object or `numpy.ndarray` of those
            objects that are axes of the current plot.
    """

    FIGURE_TOP: float = 0.9
    """Position of the top edge of the subplots,
    as a fraction of the figure height. As defined in
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    """

    DPI: int = 100
    """Dpi used for all plots."""

    def __init__(
            self,
            base_data_analysis: BaseDataAnalysis,
            experiment_iterator: ExperimentIterator,
            rc_params: dict = None,
            *args,
            **kwargs
    ):
        """Initializes `MainWindow` object and sets attributes.

        Args:
            base_data_analysis: `BaseDataAnalysis` object whose plots will be
                displayed upon initialization.
            experiment_iterator: `ExperimentIterator` object which is
                responsible for iterating through `BaseDataAnalysis` object
                whose plots are displayed.
            rc_params: dict of matplotlib rcParams. Modifies rcParams only
                for the `AnalysisViewer` plots.
            *args: optional arguments passed to
                `QtWidgets.QMainWindow.__init__()`.
            **kwargs: optional keyword arguments passed to
                `QtWidgets.QMainWindow.__init__()`.
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        # Set up matplotlib rcParams.
        self.rc_params: dict = gui_rc_params()
        if rc_params is not None:
            self.rc_params.update(rc_params)

        # Set attributes for choosing plot to display.
        self._base_data_analysis: BaseDataAnalysis = base_data_analysis
        self._plot_iterator: BidirectionalIterator = BidirectionalIterator(
            self._base_data_analysis.get_fig_ids())
        self._experiment_iterator: ExperimentIterator = experiment_iterator
        self._experiment_iterator.set_pointer_to_timestamp(
            self._base_data_analysis.raw_data_dict.get('folder', '')
        )

        # Set attributes for GUI and plot display parts of functionality.
        self._canvas: FigureCanvasQTAgg = FigureCanvasQTAgg()
        self._toolbar: NavigationToolbar = NavigationToolbar(self._canvas, self)
        self._figure: Optional[Figure] = None
        self._axes: Optional[Axes] = None

        self.plot(self._plot_iterator.next())

        # Set up layout for toolbar and canvas (which holds the plot).
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def plot(self, fig_id: str):
        """Plot figure specified with `fig_id`.

        Args:
            fig_id: str id of a figure stored in `BaseDataAnalysis.plot_dicts`.
        """
        with plt.rc_context(self.rc_params):
            # Save the old figure to close it after plotting the new one.
            old_figure = self._figure

            self.clear_axes()

            # Plot the figure.
            self._figure, self._axes = self._base_data_analysis.plot_for_gui(
                fig_id=fig_id
            )

            # Set figure properties and associate it with the canvas.
            self._figure.set_canvas(self._canvas)
            self._figure.dpi = MainWindow.DPI
            self._canvas.figure = self._figure

            # Close the old figure saved at the start of the function.
            plt.close(old_figure)

            # Trigger the canvas to update and redraw.
            self._canvas.draw()
            self._adjust_gui_after_redraw()

    def clear_axes(self):
        """Clears axes for next plot."""
        if isinstance(self._axes, Axes):
            self._axes.cla()
        elif isinstance(self._axes, numpy.ndarray):
            for axis in self._axes:
                axis.cla()

    # noinspection PyPep8Naming
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Qt GUI slot for `QtGui.QKeyEvent`. Processes arrow keys presses.

        Args:
            event: `QtGui.QKeyEvent` object to be processed.
        """
        if event.key() == QtCore.Qt.Key.Key_Left:
            self._handle_key_left_event(event)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._handle_key_right_event(event)
        elif event.key() == QtCore.Qt.Key.Key_Up:
            self._handle_key_up_event(event)
        elif event.key() == QtCore.Qt.Key.Key_Down:
            self._handle_key_down_event(event)

    def _handle_key_down_event(self, event: QtGui.QKeyEvent):
        """Goes to the next experiment and accepts the event.

        Args:
            event: key press event (assumed to be
                `QtCore.Qt.Key.Key_Down`).
        """
        with gui_utilities.set_wait_cursor(self):
            try:
                self._base_data_analysis = self._experiment_iterator.next()
                self._plot_iterator = BidirectionalIterator(
                    self._base_data_analysis.get_fig_ids())
                self.plot(self._plot_iterator.next())
            except StopIteration:
                pass
            event.accept()

    def _handle_key_up_event(self, event: QtGui.QKeyEvent):
        """Goes to the next previous and accepts the event.

        Args:
            event: key press event (assumed to be
                `QtCore.Qt.Key.Key_Up`).
        """
        with gui_utilities.set_wait_cursor(self):
            try:
                self._base_data_analysis = self._experiment_iterator.prev()
                self._plot_iterator = BidirectionalIterator(
                    self._base_data_analysis.get_fig_ids())
                self.plot(self._plot_iterator.next())
            except StopIteration:
                pass
            event.accept()

    def _handle_key_right_event(self, event: QtGui.QKeyEvent):
        """Goes to the next plot in current experiment and accepts the event.

        Args:
            event: key press event (assumed to be
                `QtCore.Qt.Key.Key_Right`).
        """
        with gui_utilities.set_wait_cursor(self):
            try:
                self.plot(self._plot_iterator.next())
            except StopIteration:
                pass
            event.accept()

    def _handle_key_left_event(self, event: QtGui.QKeyEvent):
        """Goes to the previous plot in current experiment and accepts the
        event.

        Args:
            event: key press event (assumed to be
                `QtCore.Qt.Key.Key_Left`).
        """
        with gui_utilities.set_wait_cursor(self):
            try:
                self.plot(self._plot_iterator.prev())
            except StopIteration:
                pass
            event.accept()

    def _adjust_gui_after_redraw(self):
        """Performs GUI adjustments to preserve nice and working GUI."""
        self._adjust_subplots()
        reinitialize_toolbar(self._toolbar)
        self._trigger_resize_event()

    def _adjust_subplots(self):
        """Adjust some subplot and figure values for nicer display."""
        self._figure.subplots_adjust(top=MainWindow.FIGURE_TOP)
        # in case there are multiple subplots, more padding is needed
        if isinstance(self._axes, numpy.ndarray):
            self._figure.tight_layout(pad=5.0)
