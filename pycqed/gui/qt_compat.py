from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

QtWidgets = QtWidgets
QtGui = QtGui
QtCore = QtCore
FigureCanvasQTAgg = FigureCanvasQTAgg
NavigationToolbar2QT = NavigationToolbar2QT

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



