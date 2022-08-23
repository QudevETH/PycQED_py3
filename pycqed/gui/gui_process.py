import multiprocessing as mp
import sys


def create_waveform_viewer_process(args, qt_lib):
    args += (qt_lib,)
    process = mp.Process(name="pycqed_waveform_viewer",
                         target=start_qapp_in_new_process,
                         args=args)
    process.daemon = False
    process.start()


def start_qapp_in_new_process(sequences, qubit_channel_maps,
                              experiment_name, pass_kwargs, qt_lib):
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