import sys

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


    class QMenu(QtWidgets.QMenu):
        def exec_(self, pos=None, at=None):
            return self.exec(pos, at)


    QtWidgets.QApplication = QApplication
    QtWidgets.QDialog = QDialog
    QtWidgets.QMenu = QMenu
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Signal = QtCore.pyqtSignal

    # To ensure compatibility of QApplication the easiest way is to create
    # its instance. Some code creates the instance disregarding the updated
    # QtWidgets.QApplication
    if QtWidgets.QApplication.instance() is None:

        # It is important to assign this to some variable.
        # Without it the instance() method wouldn't have a pointer to the
        # instance and return None on the next call
        app = QtWidgets.QApplication(sys.argv)

    # If there is already an instance of QApplication, and it is not an
    # instance of qt_compat.QApplication, we need to add the missing exec_
    # method to it.
    elif not isinstance(QtWidgets.QApplication.instance(), QApplication):
        QtWidgets.QApplication.instance().exec_ =\
            QtWidgets.QApplication.instance().exec

else:
    QAction = QtWidgets.QAction
