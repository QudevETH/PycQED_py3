# General imports
import time
t0 = time.time()  # to print how long init takes
from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qcodes
import qcodes as qc
qc.set_mp_method('spawn')  # force Windows behavior on mac

# Globally defined config
qc_config = {'datadir': 'D:\Experiments\Simultaneous_Driving\data',
             'PycQEDdir': 'D:\GitHubRepos\PycQED_py3'}


# General PycQED modules
from modules.measurement import measurement_control as mc
from modules.measurement import sweep_functions as swf
from modules.measurement import awg_sweep_functions as awg_swf
from modules.measurement import detector_functions as det
from modules.measurement import composite_detector_functions as cdet
from modules.measurement import calibration_toolbox as cal_tools
from modules.measurement import mc_parameter_wrapper as pw
from modules.measurement import CBox_sweep_functions as cb_swf
from modules.analysis import measurement_analysis as ma
from modules.analysis import analysis_toolbox as a_tools

from modules.utilities import general as gen
# Standarad awg sequences
from modules.measurement.waveform_control import pulsar as ps
from modules.measurement.pulse_sequences import standard_sequences as st_seqs

# Instrument drivers
from qcodes.instrument_drivers.rohde_schwarz import SGS100A as rs
import qcodes.instrument_drivers.signal_hound.USB_SA124B as sh
import qcodes.instrument_drivers.QuTech.IVVI as iv
from qcodes.instrument_drivers.tektronix import AWG5014 as tek
from instrument_drivers.physical_instruments import QuTech_ControlBoxdriver as qcb
import instrument_drivers.meta_instrument.qubit_objects.Tektronix_driven_transmon as qbt
from instrument_drivers.meta_instrument import heterodyne as hd
import instrument_drivers.meta_instrument.CBox_LookuptableManager as lm

from instrument_drivers.meta_instrument.qubit_objects import CBox_driven_transmon as qb


# Initializing instruments

# SH = sh.SignalHound_USB_SA124B('Signal hound') #commented because of 8s load time
CBox = qcb.QuTech_ControlBox('CBox', address='Com3', run_tests=False)
S1 = rs.RohdeSchwarz_SGS100A('S1', address='GPIB0::11::INSTR')  # located on top of rack
LO = rs.RohdeSchwarz_SGS100A(name='LO', address='TCPIP0::192.168.0.77')  # left of s2
S2 = rs.RohdeSchwarz_SGS100A(name='S2', address='TCPIP0::192.168.0.78')  # right
AWG = tek.Tektronix_AWG5014(name='AWG', setup_folder=None,
                            address='TCPIP0::192.168.0.9')
IVVI = iv.IVVI('IVVI', address='ASRL1', numdacs=16)

# Meta-instruments
HS = hd.LO_modulated_Heterodyne('HS', LO=LO, CBox=CBox, AWG=AWG)
LutMan = lm.QuTech_ControlBox_LookuptableManager('LutMan', CBox)

MC = mc.MeasurementControl('MC')

VIP_mon_2 = qb.CBox_driven_transmon('VIP_mon_2',
                                    LO=LO, cw_source=S1, td_source=S2,
                                    IVVI=IVVI,
                                    AWG=AWG, LutMan=LutMan,
                                    CBox=CBox, heterodyne_instr=HS, MC=MC)
VIP_mon_4 = qb.CBox_driven_transmon('VIP_mon_4',
                                    LO=LO, cw_source=S1, td_source=S2,
                                    IVVI=IVVI,
                                    AWG=AWG, LutMan=LutMan,
                                    CBox=CBox, heterodyne_instr=HS, MC=MC)
VIP_mon_6 = qb.CBox_driven_transmon('VIP_mon_6',
                                    LO=LO, cw_source=S1, td_source=S2,
                                    IVVI=IVVI,
                                    AWG=AWG, LutMan=LutMan,
                                    CBox=CBox, heterodyne_instr=HS, MC=MC)


VIP_mon_4_tek = qbt.Tektronix_driven_transmon('VIP_mon_4_tek',
                                              LO=LO,
                                              cw_source=S1, td_source=S2,
                                              IVVI=IVVI,
                                              AWG=AWG,
                                              CBox=CBox, heterodyne_instr=HS,
                                              MC=MC)

gen.load_settings_onto_instrument(VIP_mon_2, label='VIP_mon_2')
gen.load_settings_onto_instrument(VIP_mon_4, label='VIP_mon_4')
gen.load_settings_onto_instrument(VIP_mon_4_tek)
gen.load_settings_onto_instrument(VIP_mon_6, label='VIP_mon_6')

station = qc.Station(LO, S1, S2, IVVI,
                     AWG, HS, CBox, LutMan,
                     VIP_mon_2, VIP_mon_4, VIP_mon_4_tek, VIP_mon_6)
MC.station = station
station.MC = MC
nested_MC = mc.MeasurementControl('nested_MC')
nested_MC.station = station

# The AWG sequencer
station.pulsar = ps.Pulsar()
station.pulsar.AWG = station.instruments['AWG']
for i in range(4):
    # Note that these are default parameters and should be kept so.
    # the channel offset is set in the AWG itself. For now the amplitude is
    # hardcoded. You can set it by hand but this will make the value in the
    # sequencer different.
    station.pulsar.define_channel(id='ch{}'.format(i+1),
                                  name='ch{}'.format(i+1), type='analog',
                                  # max safe IQ voltage
                                  high=.5, low=-.5,
                                  offset=0.0, delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker1'.format(i+1),
                                  name='ch{}_marker1'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)
    station.pulsar.define_channel(id='ch{}_marker2'.format(i+1),
                                  name='ch{}_marker2'.format(i+1),
                                  type='marker',
                                  high=2.0, low=0, offset=0.,
                                  delay=0, active=True)
# to make the pulsar available to the standard awg seqs
st_seqs.station = station

IVVI.dac1.set(-40)
IVVI.dac2.set(70)
IVVI.dac5.set(0)

IF = -20e6        # RO modulation frequency

LO.off()

# S1.off()
S2.off()


# Calibrated at 6.6GHz (22-2-2016)
CBox.set_dac_offset(0, 0, -38.8779296875)  # Q channel
CBox.set_dac_offset(0, 1,  16.1220703125)  # I channel qubit drive AWG

CBox.set_dac_offset(1, 1, 0)  # I channel
CBox.set_dac_offset(1, 0, 0)  # Q channel readout AWG

# LO offsets calibrated at 23-2-2016 at f = 7.15350 GHz
AWG.ch3_offset.set(0.002)
AWG.ch4_offset.set(0.018)
AWG.clock_freq.set(1e9)

def set_CBox_cos_sine_weigths(IF):
    '''
    Maybe I should add this to the CBox driver
    '''
    t_base = np.arange(512)*5e-9

    cosI = np.cos(2*np.pi * t_base*IF)
    sinI = np.sin(2*np.pi * t_base*IF)
    w0 = np.round(cosI*120)
    w1 = np.round(sinI*120)

    CBox.set('sig0_integration_weights', w0)
    CBox.set('sig1_integration_weights', w1)
set_CBox_cos_sine_weigths(IF)

CBox.set('nr_averages', 2048)
# this is the max nr of averages that does not slow down the heterodyning
CBox.set('nr_samples', 75)  # Shorter because of min marker spacing
CBox.set('integration_length', 140)
CBox.set('acquisition_mode', 0)
CBox.set('lin_trans_coeffs', [1, 0, 0, 1])
CBox.set('log_length', 8000)

CBox.set('AWG0_mode', 'Tape')
CBox.set('AWG1_mode', 'Tape')
CBox.set('AWG0_tape', [1, 1])
CBox.set('AWG1_tape', [1, 1])

t1 = time.time()


print('Ran initialization in %.2fs' % (t1-t0))
