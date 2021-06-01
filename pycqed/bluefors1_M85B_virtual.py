# General imports
import time
t0 = time.time()  # to print how long init takes
import logging
import pprint

from importlib import reload  # Useful for reloading while testing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = [7, 4]

# Qcodes
import sys
import qcodes as qc

qc_config = {'datadir': r'E:\Control software\data',
             'PycQEDdir': r'E:\Control software\PycQED_py3'}

# makes sure logging messages show up in the notebook
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement.waveform_control import pulsar as ps

# Instrument drivers
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon
from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 \
    import VirtualAWG5014
from pycqed.instrument_drivers.virtual_instruments.virtual_RSSGS100A \
    import virtualRohdeSchwarz_SGS100A

################################################################################
########################### Instrument creation ################################
################################################################################

station = qc.Station()
MWG2 = virtualRohdeSchwarz_SGS100A(name='MWG2', address='TCPIP0::192.168.1.32')
MWG3 = virtualRohdeSchwarz_SGS100A(name='MWG3', address='TCPIP0::192.168.1.33')
MWG4 = virtualRohdeSchwarz_SGS100A(name='MWG4', address='TCPIP0::192.168.1.34')
MWG5 = virtualRohdeSchwarz_SGS100A(name='MWG5', address='TCPIP0::192.168.1.35')
MWG6 = virtualRohdeSchwarz_SGS100A(name='MWG6', address='TCPIP0::192.168.1.36')
MWG7 = virtualRohdeSchwarz_SGS100A(name='MWG7', address='TCPIP0::192.168.1.37')
MWG8 = virtualRohdeSchwarz_SGS100A(name='MWG8', address='TCPIP0::192.168.1.38')
MWG9 = virtualRohdeSchwarz_SGS100A(name='MWG9', address='TCPIP0::192.168.1.39')
AWG1 = VirtualAWG5014(name='AWG1')
AWG2 = VirtualAWG5014(name='AWG2')
AWG3 = VirtualAWG5014(name='AWG3')
UHF1 = VirtualAWG5014(name='UHF1')
UHF2 = VirtualAWG5014(name='UHF2')

MC = mc.MeasurementControl('MC')
MC.plotting_interval(5)
MC.station = station
pulsar = ps.Pulsar(master_awg=AWG2.name)

station.add_component(MC)

station.add_component(MWG2)
station.add_component(MWG3)
station.add_component(MWG4)
station.add_component(MWG5)
station.add_component(MWG6)
station.add_component(MWG7)
station.add_component(MWG8)
station.add_component(MWG9)

station.add_component(AWG1)
station.add_component(AWG2)
station.add_component(AWG3)
station.add_component(UHF1)
station.add_component(UHF2)

station.pulsar = pulsar

station.sequencer_config = {'RO_fixed_point': 3/0.225e9,
                            'Buffer_Flux_Flux': 0,
                            'Buffer_Flux_MW': 0,
                            'Buffer_Flux_RO': 0,
                            'Buffer_MW_Flux': 0,
                            'Buffer_MW_MW': 0,
                            'Buffer_MW_RO': 0,
                            'Buffer_RO_Flux': 0,
                            'Buffer_RO_MW': 0,
                            'Buffer_RO_RO': 0,
                            'Flux_comp_dead_time': 3e-6,
                            'slave_AWG_trig_channels': ['AWG2_ch1m1',
                                                        'AWG2_ch1m2'],
                            }

from pycqed.measurement.pulse_sequences import standard_sequences as st_seqs
from pycqed.measurement.pulse_sequences import calibration_elements as cal_elts
from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
from pycqed.measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sq2
st_seqs.station = station
sq.station = station
cal_elts.station = station
fsqs.station = station
mqs.station = station
sq2.station = station

################################################################################
########################## Parameter configuration #############################
################################################################################

# configure AWGs
delays = {
    'AWG1': 0,
    'AWG2': -137e-9,
    'AWG3': -137e-9,
    'UHF1': -270e-9,
    'UHF2': -270e-9,
}
# trigger_channel_map = {
#     'AWG2': [],
#     'AWG1': ['AWG2_ch1m1'],
#     'AWG3': ['AWG2_ch1m2'],
#     'AWG4': ['AWG2_ch2m1'],
# }

trigger_channel_map = {
    'AWG1': [],
    'AWG2': ['AWG1_ch1m', 'AWG1_ch3m', 'AWG1_ch5m', 'AWG1_ch7m'],
    'AWG3': ['AWG1_ch2m', 'AWG1_ch4m', 'AWG1_ch6m'],
    'UHF1': ['AWG1_ch8'],
    'UHF2': ['AWG1_ch8m']
}
for AWG in [AWG1, AWG2, AWG3, UHF1, UHF2]:
    # pulsar.define_awg_channels(AWG)
    # pulsar.set('{}_delay'.format(AWG.name), delays[AWG.name])
    # pulsar.set('{}_trigger_channels'.format(AWG.name),
    #            trigger_channel_map[AWG.name])
    pulsar.define_awg_channels(AWG)
    pulsar.set('{}_delay'.format(AWG.name), delays[AWG.name])
    pulsar.set('{}_compensation_pulse_min_length'.format(AWG.name), 4/1.2e9)
    pulsar.set('{}_active'.format(AWG.name), True)
    pulsar.set('{}_trigger_channels'.format(AWG.name),
               trigger_channel_map[AWG.name])

qubits = [QuDev_transmon('qb{}'.format(i+1)) for i in range(7)]
qb1, qb2, qb3, qb4, qb5, qb6, qb7 = qubits

design_mw_freqs = {
    'qb1': 5.75e9,
    'qb2': 5.90e9,
    'qb3': 4.90e9,
    'qb4': 5.05e9,
    'qb5': 5.20e9,
    'qb6': 4.20e9,
    'qb7': 4.35e9,
}
design_ro_freqs = {
    'qb1': 7.0e9,
    'qb2': 7.2e9,
    'qb3': 6.55e9,
    'qb4': 6.3e9,
    'qb5': 6.6e9,
    'qb6': 6.1e9,
    'qb7': 6.35e9,
}

for i, qb in enumerate(qubits):
    qb.instr_mc('MC')
    qb.instr_pulsar('Pulsar')

    qb.instr_ro_lo('MWG9')
    qb.instr_ge_lo(f'MWG{i+1}')

    qb.ro_lo_power(25)
    qb.ge_lo_power(22)
    qb.ge_freq(design_mw_freqs[qb.name])
    qb.ro_freq(design_ro_freqs[qb.name])

    station.add_component(qb)

t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
