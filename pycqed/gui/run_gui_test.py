# run the init script
from pycqedscripts.init.demo.virtual_ATC75_M136_S17HW02_PQSC import *

# MeasurementControl (MC) is in charge of running the measurement and saving data to an HDF file it will create in the
# data dir specified below. Change it to a local folder which should be used as data dir of your virtual setup.
MC.datadir('C:\\Users\\Kuno Knapp\\Documents\\pydata')  # data dir for measurements.
# The analysis_toolbox module (a_tools) contains the functions to load the information from an HDF file that are used
# by the analysis classes.
a_tools.datadir = MC.datadir()  # data dir for analysis

#%%
from pycqed.measurement.calibration.single_qubit_gates import Rabi
qubit = qb1
amps = np.linspace(.05, .25, 5)  # in Volts

# perform the rabi measurement on qubit 1
rabi = Rabi(qubits=[qubit],  # must be a list because we could do the msmt on multiple qubits (see 2.4 below)
            amps=amps,
            transition_name='ge',  # transmon transition on which to do Rabi; accepts "ge", "ef", "fh"
            n=1,  # how many rabi pulses to apply
            update=True,  # whether to update pi-pulse amplitude with value from the analysis
           )
#%%
