from qcodes.instrument_drivers.tektronix import AWG70002A
from qcodes.instrument.parameter import ManualParameter
from qcodes import validators as vals
from qcodes.instrument.base import Instrument
from qcodes.tests.test_visa import MockVisaHandle
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes.utils.validators import Validator


##################################################
#
# SETTINGS
#
_fg_path_val = {'direct': 'DIR',
                'DCamplified': 'DCAM',
                 'AC': 'AC'}

# number of channels
_num_of_channels = 2

# number of markers per channel
_num_of_markers = 2

# channel resolution
_chan_resolutions = [8, 9, 10]

# channel resolution docstrings
_chan_resolution_docstring = "8 bit resolution allows for two " \
                             "markers, 9 bit resolution " \
                             "allows for one, and 10 bit " \
                             "does NOT allow for markers "

# channel amplitudes
_chan_amp = 0.5

# marker ranges
_marker_high = (-1.4, 1.4)
_marker_low = (-1.4, 1.4)
##################################################


class SRValidator(Validator[float]):
    """
    Validator to validate the AWG clock sample rate
    """
    def __init__(self, awg: 'AWG70002A') -> None:
        """
        Args:
            awg: The parent instrument instance. We need this since sample
                rate validation depends on many clock settings
        """
        self.awg = awg
        self._internal_validator = vals.Numbers(1.49e3, 25e9)
        self._freq_multiplier = 2

    def validate(self, value: float, context: str='') -> None:
        if 'Internal' in self.awg.clock_source():
            self._internal_validator.validate(value)
        else:
            ext_freq = self.awg.clock_external_frequency()
            validator = vals.Numbers(1.49e3, self._freq_multiplier*ext_freq)
            validator.validate(value)


class VirtualAWGChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str,
                 channel: int) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The name used in the DataSet
            channel: The channel number, either 1 or 2.
        """

        super().__init__(parent, name)

        self.channel = channel

        num_channels = self.root_instrument.num_channels

        if channel not in list(range(1, num_channels + 1)):
            raise ValueError('Illegal channel value.')

        self.add_parameter('state',
                           label=f'Channel {channel} state',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Ints(0, 1),
                           get_parser=int)

        self.add_parameter(
            'awg_amplitude',
            label=f'Channel {channel} AWG peak-to-peak amplitude',
            parameter_class=ManualParameter,
            initial_value=0.5,
            unit='V',
            get_parser=float,
            vals=vals.Numbers(0.250, _chan_amp))

        # markers
        for mrk in range(1, _num_of_markers + 1):
            self.add_parameter(
                f'marker{mrk}_high',
                initial_value=0,
                label=f'Channel {channel} marker {mrk} high level',
                parameter_class=ManualParameter,
                unit='V',
                vals=vals.Numbers(*_marker_high),
                get_parser=float)

            self.add_parameter(
                f'marker{mrk}_low',
                initial_value=0,
                label=f'Channel {channel} marker {mrk} low level',
                parameter_class=ManualParameter,
                unit='V',
                vals=vals.Numbers(*_marker_low),
                get_parser=float)

            self.add_parameter(
                f'marker{mrk}_waitvalue',
                initial_value='LOW',
                label=f'Channel {channel} marker {mrk} wait state',
                parameter_class=ManualParameter,
                vals=vals.Enum('FIRST', 'LOW', 'HIGH'))

            self.add_parameter(
                name=f'marker{mrk}_stoppedvalue',
                initial_value='OFF',
                label=f'Channel {channel} marker {mrk} stopped value',
                parameter_class=ManualParameter,
                vals=vals.Enum('OFF', 'LOW'))

        self.add_parameter('resolution',
                           label=f'Channel {channel} bit resolution',
                           parameter_class=ManualParameter,
                           vals=vals.Enum(*_chan_resolutions),
                           get_parser=int,
                           docstring=_chan_resolution_docstring)

    def setSequenceTrack(self, seqname: str, tracknr: int) -> None:
        pass


class VirtualAWG70002A(AWG70002A.AWG70002A):
    """
    This is the PycQED driver for the virtual Tektronix AWG70002A
    Arbitrary Waveform Generator.
    It is not compatible with other instruments of the series Tektronix
    AWG70000.
    """

    def __init__(self, name: str, address='',
                 num_channels=_num_of_channels, timeout = 10,
                 model='70002A') -> None:
        Instrument.__init__(self, name)
        if model != '70002A':
            raise NotImplementedError('The virtual driver is not implemented'\
                                      'for other models than 70002A.')
        self.num_channels = num_channels
        self._address = address
        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=timeout,
            parameter_class=ManualParameter,
            vals=vals.MultiType(vals.Numbers(min_value=0), vals.Enum(None)))
        self.add_parameter(
            'address',
            unit='',
            initial_value='',
            parameter_class=ManualParameter,
            vals=vals.Strings())

        # We deem 2 channels too few for a channel list
        if self.num_channels > 2:
            chanlist = ChannelList(self, 'Channels', VirtualAWGChannel,
                                   snapshotable=False)

        for ch_num in range(1, num_channels+1):
            ch_name = f'ch{ch_num}'
            channel = VirtualAWGChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            if self.num_channels > 2:
                chanlist.append(channel)

        if self.num_channels > 2:
            self.add_submodule("channels", chanlist.to_channel_tuple())

        #clocking parameters
        self.add_parameter('clock_source',
                           label='Clock source',
                           initial_value='External',
                           parameter_class=ManualParameter,
                           val_mapping={'Internal': 'INT',
                                        'Internal, 10 MHZ ref.': 'EFIX',
                                        'Internal, variable ref.': 'EVAR',
                                        'External': 'EXT'})
        self.add_parameter('clock_external_frequency',
                           label='External clock frequency',
                           initial_value=12.5e9,
                           parameter_class=ManualParameter,
                           get_parser=float,
                           unit='Hz',
                           vals=vals.Numbers(6.25e9, 12.5e9))
        self.add_parameter('sample_rate',
                           label='Clock sample rate',
                           initial_value=25e9,
                           parameter_class=ManualParameter,
                           unit='Sa/s',
                           get_parser=float,
                           vals=SRValidator(self))
        self.add_parameter('run_state',
                           label='Run state',
                           initial_value='Stopped',
                           parameter_class=ManualParameter,
                           val_mapping={'Stopped': '0',
                                        'Waiting for trigger': '1',
                                        'Running': '2'})
        self.add_parameter(
            'clock_freq',
            label='Clock frequency',
            unit='Hz',
            vals=vals.Numbers(1e6, 25e9),
            parameter_class=ManualParameter,
            initial_value=25e9)

        self.add_parameter('trigger_interval',
                           label='Internal trigger interval',
                           parameter_class=ManualParameter,
                           initial_value=1e-6,
                           vals=vals.Numbers(1e-6, 10e-6))
        self.add_parameter('trigger_level_A',
                           label='External trigger level',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Numbers(-5, 5))
        self.add_parameter('trigger_level_B',
                           label='External trigger level',
                           parameter_class=ManualParameter,
                           initial_value=1,
                           vals=vals.Numbers(-5, 5))
        self.add_parameter('trigger_polarity_A',
                           label='Trigger polarity',
                           parameter_class=ManualParameter,
                           initial_value="POS",
                           vals=vals.Strings())
        self.add_parameter('trigger_polarity_B',
                           label='Trigger polarity',
                           parameter_class=ManualParameter,
                           initial_value="POS",
                           vals=vals.Strings())
        self.add_parameter('trigger_impedance_A',
                           label='Trigger impedance',
                           parameter_class=ManualParameter,
                           initial_value=50,
                           vals=vals.Enum(50, 1000))
        self.add_parameter('trigger_impedance_B',
                           label='Trigger impedance',
                           parameter_class=ManualParameter,
                           initial_value=50,
                           vals=vals.Enum(50, 1000))
        self.add_parameter('trigger_timing_A',
                           label="External trigger timing mode",
                           parameter_class=ManualParameter,
                           initial_value="SYNC",
                           vals=vals.Strings())
        self.add_parameter('trigger_timing_B',
                           label="External trigger timing mode",
                           parameter_class=ManualParameter,
                           initial_value="SYNC",
                           vals=vals.Strings())

        self.visa_log = MockVisaHandle()
        self.awg_files = {}
        self.file = None

    def stop(self):
        self.run_state.set('Stopped')

    def start(self, **kwargs) -> None:
        self.run_state.set('Waiting for trigger')

    def makeSEQXFile(self, trig_waits, nreps, event_jumps, event_jump_to,
                          go_to, wfms, seqname, sequence):
        return {
            'trig_waits': trig_waits,
            'nreps': nreps,
            'event_jumps': event_jumps,
            'event_jump_to': event_jump_to,
            'go_to': go_to,
            'wfms': wfms,
            'seqname': seqname,
            'sequence': sequence
        }

    def sendSEQXFile(self, awg_file, filename):
        self.awg_files[filename] = awg_file

    def clearSequenceList(self):
        pass

    def clearWaveformList(self):
        pass

    def loadSEQXFile(self, filename):
        self.file = self.awg_files[filename]

    def wait_for_operation_to_complete(self):
        """
        Waits for the latest issued overlapping command to finish
        """
        return 1

