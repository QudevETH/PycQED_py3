# run the init script
from pycqedscripts.init.demo.virtual_ATC75_M136_S17HW02_PQSC import *
datadir_path = r'C:\Users\Kuno Knapp\Documents\pydata'
MC.datadir(datadir_path)
a_tools.datadir = MC.datadir()

#%%
from pycqed.measurement.calibration.two_qubit_gates import Chevron
qbc = qb2
qbt = qb1
cz_pulse_name = 'CZ_nzbasic'
pl = np.linspace(10e-9, 150e-9, 6)
sweep_points = sp_mod.SweepPoints('pulse_length', pl, 's', dimension=0)
dev.get_pulse_par(cz_pulse_name, qbc, qbt, 'amplitude')(0.6)
amp = 0.5 + np.linspace(-0.4, 0.4, 5)
sweep_points.add_sweep_parameter('amplitude2', amp, 'V', dimension=1)
chevron_task = {
    'qbc': qbc,
    'qbt': qbt,
    'sweep_points': sweep_points
}
task_list = [chevron_task]

# run the Chevron measurement
chev = Chevron(task_list,
               dev=dev,
               cal_states=('g', 'e', 'f'),  # use calibration states g, e, f
               cz_pulse_name=cz_pulse_name,
               sweep_points=sweep_points,
               )
#%%
import importlib
from pycqed.gui import waveform_viewer
from pycqed.gui import rc_params

#%%
importlib.reload(rc_params)
importlib.reload(waveform_viewer)
wpqe = waveform_viewer.WaveformViewer(chev, new_process=False)

#%%
# example of waveform viewer with keywords
wpqe = waveform_viewer.WaveformViewer(chev,
                                      sequence_index=1,
                                      segment_index=2,
                                      rc_params=
                                      {'figure.facecolor': '#383838',
                                       'axes.facecolor': '#505050',
                                       'text.color': 'white',
                                       'axes.labelcolor': 'white',
                                       'xtick.color': 'white',
                                       'ytick.color': 'white'},
                                      linewidth='1.2',
                                      )

#%%
chev.spawn_waveform_viewer()
