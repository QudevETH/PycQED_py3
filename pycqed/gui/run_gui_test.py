# run the init script
from pycqedscripts.init.demo.virtual_ATC75_M136_S17HW02_PQSC import *
MC.datadir('C:\\Users\\Kuno Knapp\\Documents\\pydata')
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
               cal_states=('g', 'e', 'f'), # use calibration states g, e, f
               cz_pulse_name=cz_pulse_name,
               sweep_points=sweep_points,
               label='Chevron',
              )
#%%
from pycqed.gui import waveform_viewer
import importlib
from pycqed.gui import rc_params
#%%
ppv2 = ba.BaseDataAnalysis.get_default_plot_params(set_pars=False)  # v2
ppv3 = plot_mod.get_default_plot_params(set_params=False)  # v3

#%%
importlib.reload(rc_params)
importlib.reload(waveform_viewer)
wpqe = waveform_viewer.WaveformViewer(chev)

