import platform
import os
import sys
import subprocess
from pycqed.gui import qt_compat as qt
from itertools import chain, combinations
from contextlib import contextmanager
import traceback


class GUIWorkerSignals(qt.QtCore.QObject):
    """
    QtCore Signals to notify the main thread when a task has been completed
    in a worker thread.
    """
    finished_experiment = qt.QtCore.Signal(object, str)
    finished_measurement = qt.QtCore.Signal(str, object)
    finished_analysis = qt.QtCore.Signal(str, object)
    finished_update = qt.QtCore.Signal(str, object)
    finished_plots = qt.QtCore.Signal(object, object)
    exception = qt.QtCore.Signal(object, str)


class SimpleWorker(qt.QtCore.QRunnable):
    """
    Base class of runnable tasks used in GUI classes. Running tasks in a
    separate thread prevents the GUI from freezing while the task is being
    performed.
    """
    def __init__(self):
        super().__init__()
        self.signals = GUIWorkerSignals()

    def update_parameters(self, **kwargs):
        """
        Convenience method to add attributes to the class instance.
        Args:
            **kwargs: keyword argument names and values are used as attribute
                names and values respectively.
        """
        [setattr(self, argument_name, value)
         for argument_name, value in kwargs.items()]


def reset_matplotlib_backend(matplotlib_backend):
    """
    Checks if matplotlib has been imported and if so, sets the matplotlib
    backend to the backend specified in argument matplotlib_backend.

    Args:
        matplotlib_backend (str): name of the matplotlib backend to be
            activated.

    """
    if 'matplotlib' in sys.modules:
        sys.modules.get('matplotlib').use(matplotlib_backend)


def get_meta_signal_from_q_object(q_object, signal_name):
    """
    Returns the signal of the QApplication instance, whose name matches
    signal_name.
    Args:
        q_object (QApplication): QApplication instance managing the GUI windows
            signal_name (str): name of the signal to be returned
    Returns: Signal of the QApplication instance, matching the signal_name
        string. Returns None, if no matching signal is found

    """
    meta_object = q_object.metaObject()
    for i in range(meta_object.methodCount()):
        meta_method = meta_object.method(i)
        if not meta_method.isValid():
            continue
        if meta_method.methodType() == meta_method.MethodType.Signal and \
            meta_method.name() == signal_name:
            return meta_method
    return None


def handle_matplotlib_backends(app):
    """
    Stores the selected matplotlib backend at the time of opening a GUI
    window and resets the backend once the last GUI window has been closed.
    Args:
        app (QApplication): QApplication instance managing the GUI windows
    """
    if 'matplotlib' not in sys.modules:
        import matplotlib
    app._matplotlib_backend = \
        sys.modules.get('matplotlib').get_backend()
    if not app.isSignalConnected(get_meta_signal_from_q_object(
                app, "lastWindowClosed")):
        app.lastWindowClosed.connect(
            lambda: reset_matplotlib_backend(
                app._matplotlib_backend))


@contextmanager
def set_wait_cursor(widget):
    try:
        qt.QtWidgets.QApplication.setOverrideCursor(qt.QtGui.QCursor(
            qt.QtCore.Qt.CursorShape.WaitCursor))
        yield
    except Exception as exception:
        traceback.print_exception(
            type(exception), exception, exception.__traceback__)
    finally:
        qt.QtWidgets.QApplication.restoreOverrideCursor()


def clear_layout(layout):
    """
    Removes all child widgets and clears all child layouts from a Qt layout
    object.
    Args:
        layout (QLayout): Layout object to be cleared.
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
        elif child.layout():
            clear_layout(child.layout())


def clear_QFormLayout(QFormLayout):
    """
    Removes all rows from a Qt QFormLayout object.
    Args:
        QFormLayout (QFormLayout):  QFormLayout to be cleared.
    """
    while QFormLayout.rowCount():
        QFormLayout.removeRow(0)


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


def open_file_in_explorer(path):
    """
    Opens the directory containing the file specified by path in the file
    explorer application of the user's operating system.
    Args:
        path (str): Path to the file to be opened.
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def add_label_to_widget(widget, labeltext):
    """
    Adds a label to the left of the passed widget and returns the
    corresponding layout object.
    Args:
        widget (qt.QtWidgets.QWidget): Widget to which the label is added.
        labeltext (str): Label text.

    Returns: Layout object consisting of the passed widget and a label
        object placed to the left of the widget.

    """
    layout = qt.QtWidgets.QHBoxLayout()
    label = qt.QtWidgets.QLabel(labeltext)
    label.setAlignment(qt.QtCore.Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    layout.addWidget(widget)
    layout.setSpacing(0)
    return layout


class Object(object):
    pass


def powerset(iterable):
    """
    Returns a chain of all possible combinations of the
    elements in iterable.
    e.g. powerset([1,2,3]) returns a chain consisting of elements
    (), (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
