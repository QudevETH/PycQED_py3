import platform
import os
import subprocess
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from itertools import chain, combinations


class GUIWorkerSignals(QtCore.QObject):
    finished_experiment = QtCore.Signal(object, str)
    finished_measurement = QtCore.Signal(str)
    finished_analysis = QtCore.Signal(str)
    finished_update = QtCore.Signal(str)
    finished_plots = QtCore.Signal(object, object)
    exception = QtCore.Signal(object)


class SimpleWorker(QtCore.QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = GUIWorkerSignals()

    def update_parameters(self, **kwargs):
        [setattr(self, argument_name, value)
         for argument_name, value in kwargs.items()]


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
    layout = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(labeltext)
    label.setAlignment(QtCore.Qt.AlignCenter)
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
