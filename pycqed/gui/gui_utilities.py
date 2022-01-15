import platform
import os
import sys
import subprocess
from pycqed.gui import qt_compat as qt
from itertools import chain, combinations


class GUIWorkerSignals(qt.QtCore.QObject):
    finished_experiment = qt.QtCore.Signal(object, str)
    finished_measurement = qt.QtCore.Signal(str, object)
    finished_analysis = qt.QtCore.Signal(str, object)
    finished_update = qt.QtCore.Signal(str, object)
    finished_plots = qt.QtCore.Signal(object, object)
    exception = qt.QtCore.Signal(object, str)


class SimpleWorker(qt.QtCore.QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = GUIWorkerSignals()

    def update_parameters(self, **kwargs):
        [setattr(self, argument_name, value)
         for argument_name, value in kwargs.items()]


def reset_matplotlib_backend(matplotlib_backend):
    """
    Resets the matplotlib backend to the backend specified by the passed
    string. Make sure that matplotlib has been imported when using this
    function, otherwise nothing will happen.

    Args:
        matplotlib_backend (str): Specifier string of the matplotlib backend
        that should be activated.

    """
    if 'matplotlib' in sys.modules:
        sys.modules.get('matplotlib').use(matplotlib_backend)


def get_meta_signal_from_q_object(q_object, signal_name):
    meta_object = q_object.metaObject()
    for i in range(meta_object.methodCount()):
        meta_method = meta_object.method(i)
        if not meta_method.isValid():
            continue
        if meta_method.methodType () == meta_method.MethodType.Signal and \
            meta_method.name() == signal_name:
            return meta_method
    return None


def handle_matplotlib_backends(app):
    if 'matplotlib' not in sys.modules:
        import matplotlib
    app._matplotlib_backend = \
        sys.modules.get('matplotlib').get_backend()
    if not app.isSignalConnected(get_meta_signal_from_q_object(
                app, "lastWindowClosed")):
        app.lastWindowClosed.connect(
            lambda: reset_matplotlib_backend(
                app._matplotlib_backend))


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
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def add_label_to_widget(widget, labeltext):
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
    """powerset([1,2,3]) --> (), (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
