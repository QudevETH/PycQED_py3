import multiprocessing as mp
import sys


def create_waveform_viewer_process(args, qt_lib):
    """
    Creates a new process to run the qt eventloop, allowing the usage of the
    WaveformViewer without blocking the parent process.

    WARNING:
        Do not move this function to another file - the new process will
        automatically import the libraries of the module, in which this
        function is located. This may lead to a situation, where a
        different qt binding than the one in the main process is selected,
        which may result in a crash.

    Args:
        args (tuple): Arguments to initialize a WaveformViewerMainWindow
            object. The arguments must be pickle-able, which means that QCoDeS
            instrument objects cannot be passed. In particular, the Pulsar
            object (which inherits from the QCoDeS Instrument class) referenced
            by the pulsar attribute of the Sequence and Segment objects that
            need to be passed to the WaveformViewerMainWindow has to be
            replaced by a PulsarShadow object (which does not inherit from
            the QCoDeS Instrument class and is pickle-able).
        qt_lib (str): Name of the qt binding, that the new process should
            use to run the WaveformViewer.

    """
    args += (qt_lib,)
    process = mp.Process(name="pycqed_waveform_viewer",
                         target=_start_qapp_in_new_process,
                         args=args)
    process.daemon = False
    process.start()


def _start_qapp_in_new_process(sequences, qubit_channel_maps,
                               experiment_name, pass_kwargs, qt_lib):
    """
    Helper function to spawn a WaveformViewerMainWindow instance in a new
    process.

    Args:
        sequences (list): List of Sequence objects of the QuantumExperiment.
        qubit_channel_maps (list): List of channel maps of the qubits.
        experiment_name (str): Name of the QuantumExperiment, will be used
            as window title of the WaveformViewerMainWindow.
        pass_kwargs (dict): Keyword arguments for the
            WaveformViewerMainWindow class.
        qt_lib (str): Name of the qt binding to be used in the new process

    """
    qt = __import__(qt_lib, fromlist=['QtWidgets'])
    if not qt.QtWidgets.QApplication.instance():
        app = qt.QtWidgets.QApplication(sys.argv)
    else:
        app = qt.QtWidgets.QApplication.instance()
    from pycqed.gui.waveform_viewer import WaveformViewerMainWindow
    main_window = WaveformViewerMainWindow(
        sequences, qubit_channel_maps,
        experiment_name, **pass_kwargs
    )
    main_window.showMaximized()
    main_window.on_change()
    from pycqed.gui.gui_utilities import handle_matplotlib_backends
    handle_matplotlib_backends(app)
    app._matplotlib_backend = \
        sys.modules.get('matplotlib').get_backend()
    sys.modules.get('matplotlib').use('Agg')
    app.exec_()


def dict_viewer_process(snapshot, timestamp, qt_lib):
    """
    Helper function to spawn the dict viewer in a new process.
    Function needs to be in a different file than the mp.process object.
    For more information see:
    https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    Args:
        snapshot (dict): Snapshot which is displayed in the gui
        timestamp (str): Timestamp of the snapshot which is used as the header
        qt_lib (str): Name of the qt binding to be used in the new process

    """
    qt = __import__(qt_lib, fromlist=['QtWidgets'])
    if not qt.QtWidgets.QApplication.instance():
        app = qt.QtWidgets.QApplication(sys.argv)
    else:
        app = qt.QtWidgets.QApplication.instance()
    from pycqed.gui.dict_viewer import DictViewerWindow
    screen = app.primaryScreen()
    snap_viewer = DictViewerWindow(
        dic=snapshot,
        title='Snapshot timestamp: %s' % timestamp,
        screen=screen)
    app.exec()


def comparison_viewer_process(comparison_dict, timestamps, qt_lib):
    """
    Helper function to spawn the comparison viewer in a new process.
    Function needs to be in a different file than the mp.process object.
    For more information see:
    https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    Args:
        comparison_dict (dict): Dictionary of the compared snapshots
        timestamps (list of str): List of timestamps of the compared
            dictionaries
        qt_lib (str): Name of the qt binding to be used in the new process

    """
    qt = __import__(qt_lib, fromlist=['QtWidgets'])
    if not qt.QtWidgets.QApplication.instance():
        app = qt.QtWidgets.QApplication(sys.argv)
    else:
        app = qt.QtWidgets.QApplication.instance()
    from pycqed.gui.dict_viewer import ComparisonViewerWindow
    screen = app.primaryScreen()
    comparison_viewer = ComparisonViewerWindow(
        dic=comparison_dict,
        title='Comparison of %s snapshots' % len(timestamps),
        timestamps=timestamps,
        screen=screen)
    app.exec()
