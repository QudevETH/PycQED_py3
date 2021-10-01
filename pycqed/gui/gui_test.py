import sys
import matplotlib
from pycqed.measurement.quantum_experiment import QuantumExperiment

matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import QMainWindow, QApplication

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, quantum_experiment=None, sequence_index=0, segment_index=0, parent=None, width=5, height=4, dpi=100):
        figures = quantum_experiment.sequences[sequence_index].plot(show_and_close=False)
        # fig = Figure(figsize=(width, height), dpi=dpi)
        # segment.plot(show_and_close=False) returns as follows: return fig, ax
        self.axes = figures[segment_index][1]
        super(MplCanvas, self).__init__(figures[segment_index][1])


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
        self.setCentralWidget(sc)

        self.show()


if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()
w = MainWindow()
app.exec_()

