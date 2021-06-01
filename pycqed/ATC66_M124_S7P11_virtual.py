# General imports
import time
t0 = time.time()  # to print how long init takes
import logging

import collections
odict = collections.OrderedDict

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [7, 4]

import os; os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'

# Qcodes
import qcodes as qc

# makes sure logging messages show up in the notebook
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# Import a_tools first so that the datadir is correct in other modules that
# import it.
from pycqed.analysis import analysis_toolbox as a_tools
a_tools.datadir = r'D:\pydata'

# General PycQED modules
from pycqed.measurement import measurement_control as mc
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.analysis_v2 import base_analysis as ba
# set good plotting parameters
default_plot_params_dict = ba.BaseDataAnalysis.get_default_plot_params(
    set_pars=True)
from pycqed.measurement import multi_qubit_module as mqm

# Instrument drivers
from pycqed.instrument_drivers.virtual_instruments import virtual_RSSGS100A as \
    rs
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments \
    import UHFQuantumController as ziuhfqa
from pycqed.instrument_drivers.physical_instruments.ZurichInstruments \
    import ZI_HDAWG8 as zihdawg
from pycqed.instrument_drivers.meta_instrument.qubit_objects.QuDev_transmon \
    import QuDev_transmon

from IPython.display import clear_output

###############################################################################
########################### Instrument creation ###############################
###############################################################################

station = qc.Station()
qc.station = station

MWG_LO_RO1 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_RO1',
                                     address='TCPIP0::192.168.1.10')
MWG_TWPA_RO1 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_TWPA_RO1',
                                       address='TCPIP0::192.168.1.11')
MWG_LO_RO2 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_RO2',
                                     address='TCPIP0::192.168.1.12')
MWG_TWPA_RO2 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_TWPA_RO2',
                                       address='TCPIP0::192.168.1.13')
MWG_LO_QB1 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB1',
                                    address='TCPIP0::192.168.1.14')
MWG_LO_QB2 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB2',
                                    address='TCPIP0::192.168.1.15')
MWG_LO_QB45 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB45',
                                    address='TCPIP0::192.168.1.16')
MWG_LO_QB3 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB3',
                                    address='TCPIP0::192.168.1.17')
MWG_LO_QB6 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB6',
                                    address='TCPIP0::192.168.1.18')
MWG_LO_QB7 = rs.virtualRohdeSchwarz_SGS100A(name='MWG_LO_QB7',
                                    address='TCPIP0::192.168.1.19')


UHF1 = ziuhfqa.UHFQC('UHF1', device='dev2121', interface='1GbE',
                     server='emulator')
UHF2 = ziuhfqa.UHFQC('UHF2', device='dev2268', interface='1GbE',
                     server='emulator')
AWG1 = zihdawg.ZI_HDAWG8('AWG1', device='dev8098', num_codewords=2,
                          interface='1GbE', server='emulator')
AWG2 = zihdawg.ZI_HDAWG8('AWG2', device='dev8097', num_codewords=2,
                          interface='1GbE', server='emulator')
AWG3 = zihdawg.ZI_HDAWG8('AWG3', device='dev8084', num_codewords=2,
                          interface='1GbE',server='emulator')
MC = mc.MeasurementControl('MC')
MC.plotting_interval(5)
MC.station = station
MC.datadir(r'D:\pydata')

pulsar = ps.Pulsar('Pulsar')

station.add_component(MC)
station.add_component(pulsar)

station.add_component(MWG_LO_RO1)
station.add_component(MWG_TWPA_RO1)
station.add_component(MWG_LO_RO1)
station.add_component(MWG_TWPA_RO2)
station.add_component(MWG_LO_QB1)
station.add_component(MWG_LO_QB2)
station.add_component(MWG_LO_QB45)
station.add_component(MWG_LO_QB3)
station.add_component(MWG_LO_QB6)
station.add_component(MWG_LO_QB7)

station.add_component(UHF1, update_snapshot=False)
station.add_component(UHF2, update_snapshot=False)
station.add_component(AWG1, update_snapshot=False)
station.add_component(AWG2, update_snapshot=False)
station.add_component(AWG3, update_snapshot=False)

station.pulsar = pulsar

# from pycqed.measurement.pulse_sequences import calibration_elements as cal_elts
# from pycqed.measurement.pulse_sequences import single_qubit_tek_seq_elts as sq
# from pycqed.measurement.pulse_sequences import fluxing_sequences as fsqs
# from pycqed.measurement.pulse_sequences import multi_qubit_tek_seq_elts as mqs
# from pycqed.measurement.pulse_sequences import single_qubit_2nd_exc_seqs as sq2
#
# sq.station = station
# cal_elts.station = station
# fsqs.station = station
# mqs.station = station
# sq2.station = station
# mqm.station = station

###############################################################################
########################## Parameter configuration ############################
###############################################################################

# configure AWGs
delays = {
    'AWG1': 0,
    'AWG2': -137e-9,
    'AWG3': -137e-9,
    'UHF1': -270e-9,
    'UHF2': -270e-9,
}

trigger_channel_map = {
    'AWG1': [],
    'AWG2': ['AWG1_ch1m', 'AWG1_ch3m', 'AWG1_ch5m', 'AWG1_ch7m'],
    'AWG3': ['AWG1_ch2m', 'AWG1_ch4m', 'AWG1_ch6m'],
    'UHF1': ['AWG1_ch8'],
    'UHF2': ['AWG1_ch8m']
}

# define AWG channels, set default values
for AWG in [AWG1, AWG2, AWG3, UHF1, UHF2]:
    pulsar.define_awg_channels(AWG)
    pulsar.set('{}_delay'.format(AWG.name), delays[AWG.name])
    pulsar.set('{}_compensation_pulse_min_length'.format(AWG.name), 4/1.2e9)
    pulsar.set('{}_active'.format(AWG.name), True)
    pulsar.set('{}_trigger_channels'.format(AWG.name),
               trigger_channel_map[AWG.name])

    if isinstance(AWG, zihdawg.ZI_HDAWG8):
        pulsar.set('{}_min_length'.format(AWG.name), 8/2.4e9)
        AWG.set('system_clocks_referenceclock_source', 1)
        for ch in pulsar.find_awg_channels(AWG.name):
            cid = pulsar.get(ch + '_id')
            AWG.set('sigouts_{}_direct'.format(int(cid[2]) - 1), 0)
            pulsar.set('{}_amp'.format(ch), 2.5)

            # We use AWG1_ch8 sa marker channel
            if AWG.name == 'AWG1' and ch == 8:
                pulsar.set('{}_amp'.format(ch), 1.0)

            AWG.set('sigouts_{}_on'.format(int(cid[2]) - 1), True)
            AWG.set('sines_{}_enables_0'.format(int(cid[2]) - 1), 0)
            AWG.set('sines_{}_enables_1'.format(int(cid[2]) - 1), 0)
            AWG.set('triggers_in_{}_imp50'.format(int(cid[2]) - 1), 1)
            AWG.set('triggers_in_{}_level'.format(int(cid[2]) - 1), 0.5)
            AWG.set('triggers_out_{}_source'.format(int(cid[2]) - 1),
                    4 + 2*((int(cid[2]) - 1) % 2))
        for awg_nr in range(4):
            # First the DIO input (same as seen by HDAWG.awgs_0_dio_data() 
            # parameter) is bit shifted to the right by 
            # HDAWG.awgs_0_dio_mask_shift() bits. Then it is bitwise ANDed with 
            # HDAWG.awgs_0_dio_mask_value(). The result is what the playWaveDIO() 
            # command in the awg sequencer sees.
            AWG.set('awgs_{}_dio_mask_value'.format(awg_nr), 1)
            AWG.set('awgs_{}_dio_mask_shift'.format(awg_nr), 1)
            # The UHF outputs a 25MHz signal on pin 15
            AWG.set('awgs_{}_dio_strobe_index'.format(awg_nr), 15)
            # 1: rising edge, 2: falling edge or 3: both edges
            AWG.set('awgs_{}_dio_strobe_slope'.format(awg_nr), 3)
            # Choose any pin here
            AWG.set('awgs_{}_dio_valid_index'.format(awg_nr), 0)
            # 0: no valid bit needed, 1: low, 2: high
            AWG.set('awgs_{}_dio_valid_polarity'.format(awg_nr), 0)
            AWG.set('awgs_{}_auxtriggers_0_slope'.format(awg_nr), 1)
    elif isinstance(AWG, ziuhfqa.UHFQC):
        UHF = AWG

        pulsar.set('{}_element_start_granularity'.format(AWG.name), None)

        # set awg digital trigger 1 to trigger input 1
        UHF.awgs_0_auxtriggers_0_channel(0)
        # set outputs to 50 Ohm
        UHF.sigouts_0_imp50(1)
        UHF.sigouts_1_imp50(1)

        UHF.sigouts_0_on(1)
        UHF.sigouts_1_on(1)

        # set inputs to 1.5 V range
        UHF.sigins_0_range(1.5)
        UHF.sigins_1_range(1.5)

        # set awg output to 1:1 scale
        UHF.awgs_0_outputs_0_amplitude(1)
        UHF.awgs_0_outputs_1_amplitude(1)

        # set trigger level to 500 mV
        UHF.triggers_in_0_level({'UHF1': 0.25, 'UHF2': 0.5}[UHF.name])
        UHF.triggers_in_0_imp50(1)

        UHF.awgs_0_auxtriggers_0_slope(1)
        UHF.awgs_0_auxtriggers_1_slope(1)

        UHF.qas_0_delay(350)

        # Set up UHF dio port
        UHF.dios_0_drive(0xf)  # enable driving all 4 bytes
        UHF.dios_0_mode(2)  # thresholded readout results to dio output

        UHF.set('system_extclk', 1)

        UHF.timeout(60000)


# Setting trigger channels manually
AWG1.awgs_0_auxtriggers_0_channel(0)
AWG1.awgs_1_auxtriggers_0_channel(2)
AWG1.awgs_2_auxtriggers_0_channel(4)
AWG1.awgs_3_auxtriggers_0_channel(6)

AWG2.awgs_0_auxtriggers_0_channel(0)
AWG2.awgs_1_auxtriggers_0_channel(3)
AWG2.awgs_2_auxtriggers_0_channel(4)
AWG2.awgs_3_auxtriggers_0_channel(7)

AWG3.awgs_0_auxtriggers_0_channel(0)
AWG3.awgs_1_auxtriggers_0_channel(3)
AWG3.awgs_2_auxtriggers_0_channel(4)
AWG3.awgs_3_auxtriggers_0_channel(7)


pulsar.master_awg('TriggerDevice')
# Reuse waveforms
pulsar.reuse_waveforms(True)

MWG_LO_RO1.power(25)
MWG_LO_RO2.power(25)

qubits = [QuDev_transmon('qb{}'.format(i+1))
          for i in range(7)]
qb1, qb2, qb3, qb4, qb5, qb6, qb7 = qubits

design_ge_freqs = {
    'qb1': 5.65e9,
    'qb2': 5.80e9,
    'qb3': 4.80e9,
    'qb4': 4.95e9,
    'qb5': 5.10e9,
    'qb6': 4.10e9,
    'qb7': 4.25e9,
}
design_ro_freqs = {
    'qb1': 6.63e9,
    'qb2': 6.85e9,
    'qb3': 6.28e9,
    'qb4': 6.04e9,
    'qb5': 6.3e9,
    'qb6': 5.85e9,
    'qb7': 6.07e9,
}
for qb in qubits:
    qb.instr_mc('MC')
    qb.instr_pulsar('Pulsar')
    qb.instr_trigger('TriggerDevice')

    if qb in [qb1, qb2, qb4, qb5]:
        qb.instr_ro_lo('MWG_LO_RO1')
        qb.instr_uhf('UHF1')
        qb.ro_I_channel('UHF1_ch1')
        qb.ro_Q_channel('UHF1_ch2')
    else:
        qb.instr_ro_lo('MWG_LO_RO2')
        qb.instr_uhf('UHF2')
        qb.ro_I_channel('UHF2_ch1')
        qb.ro_Q_channel('UHF2_ch2')
    
    qb.ro_lo_power(24)
    qb.ge_lo_power(22)

    # QB4 and QB5 shares the LO, so we need 3dB extra power!
    if qb in [qb4, qb5]:
        qb.ge_lo_power(24)

    qb.ge_freq(design_ge_freqs[qb.name])
    qb.ro_freq(design_ro_freqs[qb.name])

    qb.ge_amp180(0.1)

    station.add_component(qb)

UCDevMap = {
    'UC3': {'LO': MWG_LO_QB1,
            'AWGCP I': 'AWG2_ch1',
            'AWGCP Q': 'AWG2_ch2',
            'spec_marker': 'AWG2_ch1m',
            'mixer_cal_port': 5},
    'UC4': {'LO': MWG_LO_QB2,
            'AWGCP I': 'AWG2_ch3',
            'AWGCP Q': 'AWG2_ch4',
            'spec_marker': 'AWG2_ch4m',
            'mixer_cal_port': 5},
    'UC5': {'LO': MWG_LO_QB45,
            'AWGCP I': 'AWG2_ch5',
            'AWGCP Q': 'AWG2_ch6',
            'spec_marker': 'AWG2_ch5m',
            'mixer_cal_port': 4},
    'UC6': {'LO': MWG_LO_QB45,
            'AWGCP I': 'AWG2_ch7',
            'AWGCP Q': 'AWG2_ch8',
            'spec_marker': 'AWG2_ch5m',
            'mixer_cal_port': 4},
    'UC7': {'LO': MWG_LO_QB3,
            'AWGCP I': 'AWG3_ch1',
            'AWGCP Q': 'AWG3_ch2',
            'spec_marker': 'AWG3_ch1m',
            'mixer_cal_port': 2},
    'UC8': {'LO': MWG_LO_QB6,
            'AWGCP I': 'AWG3_ch3',
            'AWGCP Q': 'AWG3_ch4',
            'spec_marker': 'AWG3_ch4m',
            'mixer_cal_port': 2},
    'UC9': {'LO': MWG_LO_QB7,
            'AWGCP I': 'AWG3_ch5',
            'AWGCP Q': 'AWG3_ch6',
            'spec_marker': 'AWG3_ch5m',
            'mixer_cal_port': 1},
    'UC1': {'LO': MWG_LO_RO1,
            'mixer_cal_port': 5},
    'UC2': {'LO': MWG_LO_RO2,
            'mixer_cal_port': 5},
    'DUT': {'mixer_cal_port': 6}
}

qbUCMap = {
    qb1: 'UC3',
    qb2: 'UC4',
    qb3: 'UC7',
    qb4: 'UC5',
    qb5: 'UC6',
    qb6: 'UC8',
    qb7: 'UC9',
}

for qb in qubits:
    qb.instr_ge_lo(UCDevMap[qbUCMap[qb]]['LO'].name)
    qb.ge_I_channel(UCDevMap[qbUCMap[qb]]['AWGCP I'])
    qb.ge_Q_channel(UCDevMap[qbUCMap[qb]]['AWGCP Q'])
    qb.spec_marker_channel(UCDevMap[qbUCMap[qb]]['spec_marker'])

t1 = time.time()
print('Ran initialization in %.2fs' % (t1-t0))
