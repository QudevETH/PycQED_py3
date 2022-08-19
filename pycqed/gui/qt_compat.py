from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

QtWidgets = QtWidgets
QtGui = QtGui
QtCore = QtCore
FigureCanvasQTAgg = FigureCanvasQTAgg
NavigationToolbar2QT = NavigationToolbar2QT

"""
PyQt and PySide (the two python bindings of Qt) generally follow the same 
naming convention for their modules, methods, class names, constants, 
etc. However, there are small differences (e.g. to start the QApplication, 
exec is called in PyQt, whereas exec_ is called in PySide). 
To account for these small differences, we create aliases where the names 
of the methods / classes / modules differ. Generally, the PySide2 
conventions are followed in the codebase.
"""
if QtWidgets.__package__ in ['PyQt6']:
    QAction = QtGui.QAction

    class QApplication(QtWidgets.QApplication):
        def exec_(self):
            return self.exec()


    class QDialog(QtWidgets.QDialog):
        def exec_(self):
            return self.exec()

    QtWidgets.QApplication = QApplication
    QtWidgets.QDialog = QDialog
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Signal = QtCore.pyqtSignal
else:
    QAction = QtWidgets.QAction



