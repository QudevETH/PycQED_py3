import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
import copy
from datetime import datetime
from functools import partial

from qcodes.instrument.base import Instrument
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter

import zhinst.ziPython as zi

log = logging.getLogger(__name__)

##########################################################################
# Module level functions
##########################################################################


def gen_waveform_name(ch, cw):
    """
    Return a standard waveform name based on channel and codeword number.

    Note the use of 1-based indexing of the channels. To clarify, the
    'ch' argument to this function is 0-based, but the naming of the actual
    waveforms as well as the signal outputs of the instruments are 1-based.
    The function will map 'logical' channel 0 to physical channel 1, and so on.

    """
    return 'wave_ch{}_cw{:03}'.format(ch+1, cw)


def gen_partner_waveform_name(ch, cw):
    """
    Return a standard waveform name for the partner waveform of a dual-channel
    waveform. The physical channel indexing is 1-based where as the logical channel
    indexing (i.e. the argument to this function) is 0-based. To clarify, the
    'ch' argument to this function is 0-based, but the naming of the actual
    waveforms as well as the signal outputs of the instruments are 1-based.
    The function will map 'logical' channel 0 to physical channel 1, and so on.
    """
    return gen_waveform_name(2*(ch//2) + ((ch + 1) % 2), cw)


def merge_waveforms(chan0=None, chan1=None, marker=None):
    """
    Merges waveforms for channel 0, channel 1 and marker bits into a single
    numpy array suitable for being written to the instrument. Channel 1 and marker
    data is optional. Use named arguments to combine, e.g. channel 0 and marker data.
    """
    chan0_uint = None
    chan1_uint = None
    marker_uint = None

    # The 'array_format' variable is used internally in this function in order to
    # control the order and number of uint16 words that we put together for each
    # sample of the final array. The variable is essentially interpreted as a bit
    # mask where each bit indicates which channels/marker values to include in
    # the final array. Bit 0 for chan0 data, 1 for chan1 data and 2 for marker data.
    array_format = 0

    if chan0 is not None:
        chan0_uint = np.array((np.power(2, 15)-1)*chan0, dtype=np.uint16)
        array_format += 1
    if chan1 is not None:
        chan1_uint = np.array((np.power(2, 15)-1)*chan1, dtype=np.uint16)
        array_format += 2
    if marker is not None:
        marker_uint = np.array(marker, dtype=np.uint16)
        array_format += 4

    if array_format == 1:
        return chan0_uint
    elif array_format == 2:
        return chan1_uint
    elif array_format == 3:
        return np.vstack((chan0_uint, chan1_uint)).reshape((-2,), order='F')
    elif array_format == 4:
        return marker_uint
    elif array_format == 5:
        return np.vstack((chan0_uint, marker_uint)).reshape((-2,), order='F')
    elif array_format == 6:
        return np.vstack((chan1_uint, marker_uint)).reshape((-2,), order='F')
    elif array_format == 7:
        return np.vstack((chan0_uint, chan1_uint, marker_uint)).reshape((-2,), order='F')
    else:
        return []


def plot_timing_diagram(data, bits, line_length=30):
    """
    Takes list of 32-bit integer values as read from the 'raw/dios/0/data' device nodes and creates
    a timing diagram of the result. The timing diagram can be used for verifying that e.g. the
    strobe signal (A.K.A the toggle signal) is periodic.
    """
    def _plot_lines(ax, pos, *args, **kwargs):
        if ax == 'x':
            for p in pos:
                plt.axvline(p, *args, **kwargs)
        else:
            for p in pos:
                plt.axhline(p, *args, **kwargs)

    def _plot_timing_diagram(data, bits):
        plt.figure(figsize=(20, 0.5*len(bits)))

        t = np.arange(len(data))
        _plot_lines('y', 2*np.arange(len(bits)), color='.5', linewidth=2)
        _plot_lines('x', t[0:-1:2], color='.5', linewidth=0.5)

        for n, i in enumerate(reversed(bits)):
            line = [((x >> i) & 1) for x in data]
            plt.step(t, np.array(line) + 2*n, 'r', linewidth=2, where='post')
            plt.text(-0.5, 2*n, str(i))

        plt.xlim([t[0], t[-1]])
        plt.ylim([0, 2*len(bits)+1])

        plt.gca().axis('off')
        plt.show()

    while len(data) > 0:
        if len(data) > line_length:
            d = data[0:line_length]
            data = data[line_length:]
        else:
            d = data
            data = []

        _plot_timing_diagram(d, bits)


def plot_codeword_diagram(ts, cws, range=None):
    """
    Takes a list of timestamps (X) and codewords (Y) and produces a simple 'stem' plot of the two.
    The plot is useful for visually checking that the codewords are detected at regular intervals.
    Can also be used for visual verification of standard codeword patterns such as the staircase used
    for calibration.
    """
    plt.figure(figsize=(20, 10))
    plt.stem((np.array(ts)-ts[0])*10.0/3, np.array(cws))
    if range is not None:
        plt.xlim(range[0], range[1])
        xticks = np.arange(range[0], range[1], step=20)
        while len(xticks) > 20:
            xticks = xticks[::2]
        plt.xticks(xticks)
    plt.xlabel('Time (ns)')
    plt.ylabel('Codeword (#)')
    plt.grid()
    plt.show()


def _gen_set_cmd(dev_set_func, node_path: str):
    """
    Generates a set function based on the dev_set_type method (e.g., seti)
    and the node_path (e.g., '/dev8003/sigouts/1/mode'
    """
    def set_cmd(val):
        return dev_set_func(node_path, val)
    return set_cmd


def _gen_get_cmd(dev_get_func, node_path: str):
    """
    Generates a get function based on the dev_set_type method (e.g., geti)
    and the node_path (e.g., '/dev8003/sigouts/1/mode'
    """
    def get_cmd():
        return dev_get_func(node_path)
    return get_cmd

##########################################################################
# Exceptions
##########################################################################


class ziDAQError(Exception):
    """Exception raised when no DAQ has been connected."""
    pass


class ziModuleError(Exception):
    """Exception raised when a module generates an error."""
    pass


class ziValueError(Exception):
    """Exception raised when a wrong or empty value is returned."""
    pass


class ziCompilationError(Exception):
    """Exception raised when an AWG program fails to compile."""
    pass


class ziDeviceError(Exception):
    """Exception raised when a class is used with the wrong device type."""
    pass


class ziOptionsError(Exception):
    """Exception raised when a device does not have the right options installed."""
    pass


class ziVersionError(Exception):
    """Exception raised when a device does not have the right firmware versions."""
    pass


class ziReadyError(Exception):
    """Exception raised when a device was started which is not ready."""
    pass


class ziRuntimeError(Exception):
    """Exception raised when a device detects an error at runtime."""
    pass


class ziConfigurationError(Exception):
    """Exception raised when a wrong configuration is detected."""
    pass

##########################################################################
# Mock classes
##########################################################################


class MockDAQServer():
    """
    This class implements a mock version of the DAQ object used for
    communicating with the instruments. It contains dummy declarations of
    the most important methods implemented by the server and used by
    the instrument drivers.

    Important: The Mock server creates some default 'nodes' (basically
    just entries in a 'dict') based on the device name that is used when
    connecting to a device. These nodes differ depending on the instrument
    type, which is determined by the number in the device name: dev2XXX are
    UHFQA instruments, dev8XXX are HDAWG8 instruments, dev10XXX are PQSC
    instruments.
    """

    def __init__(self, server, port, apilevel, verbose=False):
        self.server = server
        self.port = port
        self.apilevel = apilevel
        self.device = None
        self.interface = None
        self.nodes = {'/zi/devices/connected': {'type': 'String', 'value': ''}}
        self.devtype = None
        self.poll_nodes = []
        self.verbose = verbose
        self.async_nodes = []

    def awgModule(self):
        return MockAwgModule(self)

    def setDebugLevel(self, debuglevel: int):
        log.info(f'MockDAQServer: Setting debug level to {debuglevel}')

    def connectDevice(self, device, interface):
        if self.device is not None:
            raise ziDAQError(
                'Trying to connect to a device that is already connected!')

        if self.interface is not None and self.interface != interface:
            raise ziDAQError(
                'Trying to change interface on an already connected device!')

        self.device = device
        self.interface = interface

        if self.device.lower().startswith('dev2'):
            self.devtype = 'UHFQA'
        elif self.device.lower().startswith('dev8'):
            self.devtype = 'HDAWG8'
        elif self.device.lower().startswith('dev10'):
            self.devtype = 'PQSC'

        # Add paths
        filename = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'zi_parameter_files', 'node_doc_{}.json'.format(self.devtype))
        if not os.path.isfile(filename):
            raise ziRuntimeError(
                'No parameter file available for devices of type ' + self.devtype)
        # NB: defined in parent class
        self._load_parameter_file(filename=filename)

        # Update connected status
        self.nodes['/zi/devices/connected']['value'] = self.device

        # Set the LabOne revision
        self.nodes['/zi/about/revision'] = {'type': 'Integer', 'value': 200802104}

        self.nodes[f'/{self.device}/features/devtype'] = {'type': 'String', 'value': self.devtype}
        self.nodes[f'/{self.device}/system/fwrevision'] = {'type': 'Integer', 'value': 99999}
        self.nodes[f'/{self.device}/system/fpgarevision'] = {'type': 'Integer', 'value': 99999}
        self.nodes[f'/{self.device}/system/slaverevision'] = {'type': 'Integer', 'value': 99999}
        self.nodes[f'/{self.device}/raw/error/json/errors'] = {
                'type': 'String', 'value': '{"sequence_nr" : 0, "new_errors" : 0, "first_timestamp" : 0, "timestamp" : 0, "timestamp_utc" : "2019-08-07 17 : 33 : 55", "messages" : []}'}
        # 1 line added to Delft version: create error/clear node
        self.nodes[f'/{self.device}/raw/error/clear'] = {'type': 'Integer', 'value': 0}

        if self.devtype == 'UHFQA':
            self.nodes[f'/{self.device}/features/options'] = {'type': 'String', 'value': 'QA\nAWG'}
            for i in range(16):
                self.nodes[f'/{self.device}/awgs/0/waveform/waves/{i}'] = {'type': 'ZIVectorData', 'value': np.array([])}
            for i in range(10):
                self.nodes[f'/{self.device}/qas/0/integration/weights/{i}/real'] = {'type': 'ZIVectorData', 'value': np.array([])}
                self.nodes[f'/{self.device}/qas/0/integration/weights/{i}/imag'] = {'type': 'ZIVectorData', 'value': np.array([])}
                self.nodes[f'/{self.device}/qas/0/result/data/{i}/wave'] = {'type': 'ZIVectorData', 'value': np.array([])}
            self.nodes[f'/{self.device}/raw/dios/0/delay'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/extclk'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/drive'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/mode'] = {'type': 'Integer', 'value': 0}
        elif self.devtype == 'HDAWG8':
            self.nodes[f'/{self.device}/features/options'] = {'type': 'String', 'value': 'PC\nME'}
            for i in range(32):
                self.nodes['/' + self.device +
                        '/raw/dios/0/delays/' + str(i) + '/value'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/raw/error/blinkseverity'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/raw/error/blinkforever'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/extclk'] = {'type': 'Integer', 'value': 0}
            # 1 line added to Delft version: allow for changing HDAWG clock_freq
            self.nodes[f'/{self.device}/system/clocks/sampleclock/freq'] = {'type': 'Double', 'value': 2.4e9}
            for awg_nr in range(4):
                # 1 line different from Delft version: more waves
                for i in range(2048):
                    self.nodes[f'/{self.device}/awgs/{awg_nr}/waveform/waves/{i}'] = {
                        'type': 'ZIVectorData', 'value': np.array([])}
                    self.nodes[f'/{self.device}/awgs/{awg_nr}/waveform/waves/{i}'] = {
                        'type': 'ZIVectorData', 'value': np.array([])}
                    self.nodes[f'/{self.device}/awgs/{awg_nr}/waveform/waves/{i}'] = {
                        'type': 'ZIVectorData', 'value': np.array([])}
                    self.nodes[f'/{self.device}/awgs/{awg_nr}/waveform/waves/{i}'] = {
                        'type': 'ZIVectorData', 'value': np.array([])}
                self.nodes[f'/{self.device}/awgs/{awg_nr}/commandtable/data'] = {
                    'type': 'ZIVectorData', 'value': np.array([])}
            for sigout_nr in range(8):
                self.nodes[f'/{self.device}/sigouts/{sigout_nr}/precompensation/fir/coefficients'] = {
                    'type': 'ZIVectorData', 'value': np.array([])}
            self.nodes[f'/{self.device}/dios/0/mode'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/extclk'] = {'type': 'Integer', 'value': 0}
            self.nodes[f'/{self.device}/dios/0/drive'] = {'type': 'Integer', 'value': 0}
            for dio_nr in range(32):
                self.nodes[f'/{self.device}/raw/dios/0/delays/{dio_nr}/value'] = {'type': 'Integer', 'value': 0}
        # 5 lines removed from Delft version: unneeded (merge artifact)

    def listNodesJSON(self, path):
        pass

    def getString(self, path):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.nodes[path]['type'] != 'String':
            raise ziRuntimeError(
                "Trying to node '" + path + "' as string, but the type is '" + self.nodes[path]['type'] + "'!")

        return self.nodes[path]['value']

    def getInt(self, path):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.verbose:
            print('getInt', path, int(self.nodes[path]['value']))

        return int(self.nodes[path]['value'])

    def getDouble(self, path):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.verbose:
            print('getDouble', path, float(self.nodes[path]['value']))

        return float(self.nodes[path]['value'])

    def setInt(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.verbose:
            print('setInt', path, value)

        self.nodes[path]['value'] = value

    def asyncSetInt(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.verbose:
            print('asyncSetInt', path, value)

        self.async_nodes.append(partial(self.setInt, path, value))

    def setDouble(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")
        if self.verbose:
            print('setDouble', path, value)
        self.nodes[path]['value'] = value

    def asyncSetDouble(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")
        if self.verbose:
            print('setDouble', path, value)

        self.async_nodes.append(partial(self.setDouble, path, value))

    def setVector(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if self.nodes[path]['type'] != 'ZIVectorData':
            raise ziRuntimeError("Unable to set node '" + path + "' of type " +
                                 self.nodes[path]['type'] + " using setVector!")

        self.nodes[path]['value'] = value

    def setComplex(self, path, value):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if not self.nodes[path]['type'].startswith('Complex'):
            raise ziRuntimeError("Unable to set node '" + path + "' of type " +
                                 self.nodes[path]['type'] + " using setComplex!")

        if self.verbose:
            print('setComplex', path, value)
        self.nodes[path]['value'] = value

    def getComplex(self, path):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        if not self.nodes[path]['type'].startswith('Complex'):
            raise ziRuntimeError("Unable to get node '" + path + "' of type " +
                                 self.nodes[path]['type'] + " using getComplex!")

        if self.verbose:
            print('getComplex', path, self.nodes[path]['value'])
        return self.nodes[path]['value']

    def get(self, path, flat, flags):
        if path not in self.nodes:
            raise ziRuntimeError("Unknown node '" + path +
                                 "' used with mocked server and device!")

        return {path: [{'vector': self.nodes[path]['value']}]}

    def getAsEvent(self, path):
        self.poll_nodes.append(path)

    def poll(self, poll_time, timeout, flags, flat):
        poll_data = {}

        for path in self.poll_nodes:
            if self.verbose:
                print('poll', path)
            m = re.match(r'/(\w+)/qas/0/result/data/(\d+)/wave', path)
            if m:
                poll_data[path] = [{'vector': np.random.rand(
                    self.getInt('/' + m.group(1) + '/qas/0/result/length'))}]
                continue

            m = re.match(r'/(\w+)/qas/0/monitor/inputs/(\d+)/wave', path)
            if m:
                poll_data[path] = [{'vector': np.random.rand(
                    self.getInt('/' + m.group(1) + '/qas/0/monitor/length'))}]
                continue

            m = re.match(r'/(\w+)/awgs/(\d+)/ready', path)
            if m:
                poll_data[path] = {'value': [1]}
                continue

            poll_data[path] = {'value': [0]}

        return poll_data

    def subscribe(self, path):
        if self.verbose:
            print('subscribe', path)

        self.poll_nodes.append(path)

    def unsubscribe(self, path):
        if self.verbose:
            print('unsubscribe', path)

        if path in self.poll_nodes:
            self.poll_nodes.remove(path)

    def sync(self):
        """The sync method does not need to do anything except goes through
        the list of nodes set asynchronously and executes those.
        """
        for p in self.async_nodes:
            p()

    def _load_parameter_file(self, filename: str):
        """
        Takes in a node_doc JSON file auto generates paths based on
        the contents of this file.
        """
        # 2 lines different from Delft version: properly close file
        with open(filename) as fo:
            f = fo.read()
        node_pars = json.loads(f)
        for par in node_pars.values():
            node = par['Node'].split('/')
            # The parfile is valid for all devices of a certain type
            # so the device name has to be split out.
            parpath = '/' + self.device + '/' + '/'.join(node)
            if par['Type'].startswith('Integer'):
                self.nodes[parpath.lower()] = {'type': par['Type'], 'value': 0}
            elif par['Type'].startswith('Double'):
                self.nodes[parpath.lower()] = {
                    'type': par['Type'], 'value': 0.0}
            elif par['Type'].startswith('Complex'):
                self.nodes[parpath.lower()] = {
                    'type': par['Type'], 'value': 0 + 0j}
            elif par['Type'].startswith('String'):
                self.nodes[parpath.lower()] = {
                    'type': par['Type'], 'value': ''}
        # 1 line added to Delft version: return for futher actions in child
        return node_pars


class MockAwgModule():
    """
    This class implements a mock version of the awgModule object used for
    compiling and uploading AWG programs. It doesn't actually compile anything, but
    only maintains a counter of how often the compilation method has been executed.

    For the future, the class could be updated to allow the user to select whether
    the next compilation should be successful or not in order to enable more
    flexibility in the unit tests of the actual drivers.
    """

    def __init__(self, daq):
        self._daq = daq
        self._device = None
        self._index = None
        self._sourcestring = None
        self._compilation_count = {}
        # 2 lines deleted from Delft version: do not create waves dir

    def get_compilation_count(self, index):
        if index not in self._compilation_count:
            raise ziModuleError(
                'Trying to access compilation count of invalid index ' + str(index) + '!')

        return self._compilation_count[index]

    def set(self, path, value):
        if path == 'awgModule/device':
            self._device = value
        elif path == 'awgModule/index':
            self._index = value
            if self._index not in self._compilation_count:
                self._compilation_count[self._index] = 0
        elif path == 'awgModule/compiler/sourcestring':
            # The compiled program is stored in _sourcestring
            self._sourcestring = value
            if self._index not in self._compilation_count:
                raise ziModuleError(
                    'Trying to compile AWG program, but no AWG index has been configured!')

            if self._device is None:
                raise ziModuleError(
                    'Trying to compile AWG program, but no AWG device has been configured!')

            self._compilation_count[self._index] += 1
            self._daq.setInt('/' + self._device + '/' +
                             'awgs/' + str(self._index) + '/ready', 1)

    def get(self, path):
        if path == 'awgModule/device':
            value = [self._device]
        elif path == 'awgModule/index':
            value = [self._index]
        elif path == 'awgModule/compiler/statusstring':
            value = ['File successfully uploaded']
        else:
            value = ['']

        for elem in reversed(path.split('/')[1:]):
            rv = {elem: value}
            value = rv

        return rv

    def execute(self):
        pass

##########################################################################
# Class
##########################################################################


class ZI_base_instrument(Instrument):
    """
    This is a base class for Zurich Instruments instrument drivers.
    It includes functionality that is common to all instruments. It maintains
    a list of available nodes as JSON files in the 'zi_parameter_files'
    subfolder. The parameter files should be regenerated when newer versions
    of the firmware are installed on the instrument.

    The base class also manages waveforms for the instruments. The waveforms
    are kept in a table, which is kept synchronized with CSV files in the
    awg/waves folder belonging to LabOne. The base class will select whether
    to compile and configure an instrument based on changes to the waveforms
    and to the requested AWG program. Basically, if a waveform changes length
    or if the AWG program changes, then the program will be compiled and
    uploaded the next time the user executes the 'start' method. If a waveform
    has changed, but the length is the same, then the waveform will simply
    be updated on the instrument using a a fast waveform upload technique. Again,
    this is triggered when the 'start' method is called.
    """

    ##########################################################################
    # Constructor
    ##########################################################################

    def __init__(self,
                 name: str,
                 device: str,
                 interface: str= '1GbE',
                 server: str= 'localhost',
                 port: int= 8004,
                 # 1 line different from Delft version: apilevel
                 apilevel: int= 6,
                 num_codewords: int= 0,
                 awg_module: bool=True,
                 logfile: str = None,
                 **kw) -> None:
        """
        Input arguments:
            name:           (str) name of the instrument as seen by the user
            device          (str) the name of the device e.g., "dev8008"
            interface       (str) the name of the interface to use ('1GbE' or 'USB')
            server          (str) the host where the ziDataServer is running
            port            (int) the port to connect to for the ziDataServer (don't change)
            apilevel        (int) the API version level to use (don't change unless you know what you're doing)
            awg_module      (bool) create an awgModule
            num_codewords   (int) the number of codeword-based waveforms to prepare
            logfile         (str) file name where all commands should be logged
        """
        t0 = time.time()
        super().__init__(name=name, **kw)

        # Decide which server to use based on name
        if server == 'emulator':
            log.info(f'{device}: Connecting to mock DAQ server')
            self.daq = MockDAQServer(server, port, apilevel)
        else:
            log.info(f'{device}: Connecting to DAQ server')
            self.daq = zi.ziDAQServer(server, port, apilevel)

        if not self.daq:
            raise(ziDAQError())

        # 1 line different from Delft version: DAQ debug level (more verbose)
        self.daq.setDebugLevel(4)

        # Handle absolute path
        self.use_setVector = "setVector" in dir(self.daq)

        # Connect a device
        if not self._is_device_connected(device):
            log.info(f'{device}: Connecting to device')
            self.daq.connectDevice(device, interface)
        self.devname = device
        self.devtype = self.gets('features/devtype')

        # We're now connected, so do some sanity checking
        self._check_devtype()
        self._check_versions()
        self._check_options()

        # Default waveform length used when initializing waveforms to zero
        self._default_waveform_length = 32

        # add qcodes parameters based on JSON parameter file
        # FIXME: we might want to skip/remove/(add  to _params_to_skip_update) entries like AWGS/*/ELF/DATA,
        #       AWGS/*/SEQUENCER/ASSEMBLY, AWGS/*/DIO/DATA
        filename = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'zi_parameter_files', 'node_doc_{}.json'.format(self.devtype))
        if not os.path.isfile(filename):
            log.info(f"{self.devname}: Parameter file not found, creating  '{filename}''")
            self._create_parameter_file(filename=filename)

        try:
            # NB: defined in parent class
            log.info(f'{self.devname}: Loading parameter file')
            self._load_parameter_file(filename=filename)
        except FileNotFoundError:
            # Should never happen as we just created the file above
            log.error(f"{self.devname}: parameter file for data parameters {filename} not found")
            raise

        # Create modules
        if awg_module:
            self._awgModule = self.daq.awgModule()
            self._awgModule.set('awgModule/device', device)
            self._awgModule.execute()

            # Will hold information about all configured waveforms
            self._awg_waveforms = {}

            # Asserted when AWG needs to be reconfigured
            self._awg_needs_configuration = [False]*(self._num_awgs())
            self._awg_program = [None]*(self._num_awgs())

            # Create waveform parameters
            self._num_codewords = 0
            self._add_codeword_waveform_parameters(num_codewords)
        else:
            self._awgModule = None

        # Create other neat parameters
        self._add_extra_parameters()
        # A list of all subscribed paths
        self._subscribed_paths = []

        # Structure for storing errors
        self._errors = None
        # Structure for storing errors that should be demoted to warnings
        self._errors_to_ignore = []
        # Make initial error check
        self.check_errors()

        # Default is not to use async mode
        self._async_mode = False

        # Optionally setup log file
        if logfile is not None:
            self._logfile = open(logfile, 'w')
        else:
            self._logfile = None

        # Show some info
        serial = self.get('features_serial')
        options = self.get('features_options')
        fw_revision = self.get('system_fwrevision')
        fpga_revision = self.get('system_fpgarevision')
        log.info('{}: serial={}, options={}, fw_revision={}, fpga_revision={}'
                 .format(self.devname, serial, options.replace('\n', '|'), fw_revision, fpga_revision))

        self.connect_message(begin_time=t0)

    ##########################################################################
    # Private methods: Abstract Base Class methods
    ##########################################################################

    def _check_devtype(self):
        """
        Checks that the driver is used with the correct device-type.
        """
        raise NotImplementedError('Virtual method with no implementation!')

    def _check_options(self):
        """
        Checks that the correct options are installed on the instrument.
        """
        raise NotImplementedError('Virtual method with no implementation!')

    def _check_versions(self):
        """
        Checks that sufficient versions of the firmware are available.
        """
        raise NotImplementedError('Virtual method with no implementation!')

    def _check_awg_nr(self, awg_nr):
        """
        Checks that the given AWG index is valid for the device.
        """
        raise NotImplementedError('Virtual method with no implementation!')

    def _update_num_channels(self):
        raise NotImplementedError('Virtual method with no implementation!')

    def _update_awg_waveforms(self):
        raise NotImplementedError('Virtual method with no implementation!')

    def _num_channels(self):
        raise NotImplementedError('Virtual method with no implementation!')

    def _num_awgs(self):
        return self._num_channels()//2

    def _get_waveform_table(self, awg_nr: int) -> list:
        return dict()

    def _add_extra_parameters(self) -> None:
        """
        Adds extra useful parameters to the instrument.
        """
        log.info(f'{self.devname}: Adding extra parameters')
        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=30,
            parameter_class=ManualParameter,
            vals=validators.Ints())

    ##########################################################################
    # Private methods
    ##########################################################################

    def _add_codeword_waveform_parameters(self, num_codewords) -> None:
        """
        Adds parameters that are used for uploading codewords.
        It also contains initial values for each codeword to ensure
        that the "upload_codeword_program" works.
        """
        docst = ('Specifies a waveform for a specific codeword. ' +
                 'The waveforms must be uploaded using ' +
                 '"upload_codeword_program". The channel number corresponds' +
                 ' to the channel as indicated on the device (1 is lowest).')

        self._params_to_skip_update = []
        log.info(f'{self.devname}: Adding codeword waveform parameters')
        for ch in range(self._num_channels()):
            for cw in range(max(num_codewords, self._num_codewords)):
                # NB: parameter naming identical to QWG
                wf_name = gen_waveform_name(ch, cw)

                if cw >= self._num_codewords and wf_name not in self.parameters:
                    # Add parameter
                    self.add_parameter(
                        wf_name,
                        label='Waveform channel {} codeword {:03}'.format(
                            ch+1, cw),
                        vals=validators.Arrays(),  # min_value, max_value = unknown
                        set_cmd=self._gen_write_waveform(ch, cw),
                        get_cmd=self._gen_read_waveform(ch, cw),
                        docstring=docst)
                    self._params_to_skip_update.append(wf_name)
                    # Make sure the waveform data is up-to-date
                    self._gen_read_waveform(ch, cw)()
                elif cw >= num_codewords:
                    # Delete parameter as it's no longer needed
                    if wf_name in self.parameters:
                        self.parameters.pop(wf_name)
                        self._awg_waveforms.pop(wf_name)

        # Update the number of codewords
        self._num_codewords = num_codewords

    def _load_parameter_file(self, filename: str):
        """
        Takes in a node_doc JSON file auto generates parameters based on
        the contents of this file.
        """
        # 2 lines different from Delft version: properly close file
        with open(filename) as fo:
            f = fo.read()
        node_pars = json.loads(f)
        for par in node_pars.values():
            node = par['Node'].split('/')
            # The parfile is valid for all devices of a certain type
            # so the device name has to be split out.
            parname = '_'.join(node).lower()
            parpath = '/' + self.devname + '/' + '/'.join(node)

            # This block provides the mapping between the ZI node and QCoDes
            # parameter.
            par_kw = {}
            par_kw['name'] = parname
            if par['Unit'] != 'None':
                par_kw['unit'] = par['Unit']
            else:
                par_kw['unit'] = 'arb. unit'

            par_kw['docstring'] = par['Description']
            if "Options" in par.keys():
                # options can be done better, this is not sorted
                par_kw['docstring'] += '\nOptions:\n' + str(par['Options'])

            # Creates type dependent get/set methods
            if par['Type'] == 'Integer (64 bit)':
                par_kw['set_cmd'] = _gen_set_cmd(self.seti, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.geti, parpath)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = validators.Ints()

            elif par['Type'] == 'Integer (enumerated)':
                par_kw['set_cmd'] = _gen_set_cmd(self.seti, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.geti, parpath)
                par_kw['vals'] = validators.Ints()

            elif par['Type'] == 'Double':
                par_kw['set_cmd'] = _gen_set_cmd(self.setd, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.getd, parpath)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = validators.Numbers()

            elif par['Type'] == 'Complex Double':
                par_kw['set_cmd'] = _gen_set_cmd(self.setc, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.getc, parpath)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = validators.Anything()

            elif par['Type'] == 'ZIVectorData':
                par_kw['set_cmd'] = _gen_set_cmd(self.setv, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.getv, parpath)
                # min/max not implemented yet for ZI auto docstrings #352
                par_kw['vals'] = validators.Arrays(valid_types=(complex, np.integer, np.floating))

            elif par['Type'] == 'String':
                par_kw['set_cmd'] = _gen_set_cmd(self.sets, parpath)
                par_kw['get_cmd'] = _gen_get_cmd(self.gets, parpath)
                par_kw['vals'] = validators.Strings()

            elif par['Type'] == 'CoreString':
                par_kw['get_cmd'] = _gen_get_cmd(self.getd, parpath)
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = validators.Strings()

            elif par['Type'] == 'ZICntSample':
                par_kw['get_cmd'] = None  # Not implemented
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = None  # Not implemented

            elif par['Type'] == 'ZITriggerSample':
                par_kw['get_cmd'] = None  # Not implemented
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = None  # Not implemented

            elif par['Type'] == 'ZIDIOSample':
                par_kw['get_cmd'] = None  # Not implemented
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = None  # Not implemented

            elif par['Type'] == 'ZIAuxInSample':
                par_kw['get_cmd'] = None  # Not implemented
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = None  # Not implemented

            elif par['Type'] == 'ZIScopeWave':
                par_kw['get_cmd'] = None  # Not implemented
                par_kw['set_cmd'] = None  # Not implemented
                par_kw['vals'] = None  # Not implemented

            else:
                raise NotImplementedError(
                    "Parameter '{}' of type '{}' not supported".format(
                        parname, par['Type']))

            # If not readable/writable the methods are removed after the type
            # dependent loop to keep this more readable.
            if 'Read' not in par['Properties']:
                par_kw['get_cmd'] = None
            if 'Write' not in par['Properties']:
                par_kw['set_cmd'] = None
            self.add_parameter(**par_kw)

    def _create_parameter_file(self, filename: str):
        """
        This generates a json file Containing the node_docs as extracted
        from the ZI instrument API.

        Replaces the use of the s_node_pars and d_node_pars files.
        """
        # Get all interesting nodes
        nodes = json.loads(self.daq.listNodesJSON('/' + self.devname))

        modified_nodes = {}

        # Do some name mangling
        for name, node in nodes.items():
            name = name.replace('/' + self.devname.upper() + '/', '')
            node['Node'] = name
            modified_nodes[name] = node

        # Dump the nodes
        with open(filename, "w") as json_file:
            json.dump(modified_nodes, json_file, indent=4, sort_keys=True)

    def _is_device_connected(self, device):
        """
        Return true if the given device is already connected to the server.
        """
        if device.lower() in [x.lower() for x in self.daq.getString('/zi/devices/connected').split(',')]:
            return True
        else:
            return False

    def _get_full_path(self, paths):
        """
        Concatenates the device name with one or more paths to create a fully
        qualified path for use in the server.
        """
        if type(paths) is list:
            for p, n in enumerate(paths):
                if p[0] != '/':
                    paths[n] = ('/' + self.devname + '/' + p).lower()
                else:
                    paths[n] = paths[n].lower()
        else:
            if paths[0] != '/':
                paths = ('/' + self.devname + '/' + paths).lower()
            else:
                paths = paths.lower()

        return paths

    def _get_awg_directory(self):
        """
        Returns the AWG directory where waveforms should be stored.
        # 2 lines added to Delft version: additional explanation
        Note that this directory is not relevant when programming
        via configure_awg_from_string.
        """
        return os.path.join(self._awgModule.get('awgModule/directory')['directory'][0], 'awg')

    def _initialize_waveform_to_zeros(self):
        """
        Generates all zeros waveforms for all codewords.
        """
        t0 = time.time()
        wf = np.zeros(self._default_waveform_length)
        waveform_params = [value for key, value in self.parameters.items()
                           if 'wave_ch' in key.lower()]
        for par in waveform_params:
            par(wf)
        t1 = time.time()
        log.debug(
            'Set all waveforms to zeros in {:.1f} ms'.format(1.0e3*(t1-t0)))

    def _gen_write_waveform(self, ch, cw):
        def write_func(waveform):
            log.debug(f"{self.devname}: Writing waveform (len {len(waveform)}) to ch{ch} cw{cw}")
            # Determine which AWG this waveform belongs to
            awg_nr = ch//2

            # Name of this waveform
            wf_name = gen_waveform_name(ch, cw)

            # Check that we're allowed to modify this waveform
            if self._awg_waveforms[wf_name]['readonly']:
                raise ziConfigurationError(
                    'Trying to modify read-only waveform on '
                    'codeword {}, channel {}'.format(cw, ch))

            # The length of HDAWG waveforms should be a multiple of 8 samples.
            if (len(waveform) % 8) != 0:
                log.debug(f"{self.devname}: waveform is not a multiple of 8 samples, appending zeros.")
                extra_zeros = 8-(len(waveform) % 8)
                waveform = np.concatenate([waveform, np.zeros(extra_zeros)])

            # If the length has changed, we need to recompile the AWG program
            if len(waveform) != len(self._awg_waveforms[wf_name]['waveform']):
                log.debug(f"{self.devname}: Length of waveform has changed. Flagging awg as requiring recompilation.")
                self._awg_needs_configuration[awg_nr] = True

            # Update the associated CSV file
            log.debug(f"{self.devname}: Updating csv waveform {wf_name}, for ch{ch}, cw{cw}")
            self._write_csv_waveform(ch=ch, cw=cw, wf_name=wf_name,
                                     waveform=waveform)

            # And the entry in our table and mark it for update
            self._awg_waveforms[wf_name]['waveform'] = waveform
            log.debug(f"{self.devname}: Marking waveform as dirty.")
            self._awg_waveforms[wf_name]['dirty'] = True

        return write_func

    def _write_csv_waveform(self, ch: int, cw: int, wf_name: str, waveform) -> None:
        filename = os.path.join(
            self._get_awg_directory(), 'waves',
            self.devname + '_' + wf_name + '.csv')
        np.savetxt(filename, waveform, delimiter=",")

    def _gen_read_waveform(self, ch, cw):
        def read_func():
            # AWG
            awg_nr = ch//2

            # Name of this waveform
            wf_name = gen_waveform_name(ch, cw)
            log.debug(f"{self.devname}: Reading waveform {wf_name} for ch{ch} cw{cw}")
            # Check if the waveform data is in our dictionary
            if wf_name not in self._awg_waveforms:
                log.debug(f"{self.devname}: Waveform not in self._awg_waveforms: reading from csv file.")
                # Initialize elements
                self._awg_waveforms[wf_name] = {
                    'waveform': None, 'dirty': False, 'readonly': False}
                # Make sure everything gets recompiled
                log.debug(f"{self.devname}: Flagging awg as requiring recompilation.")
                self._awg_needs_configuration[awg_nr] = True
                # It isn't, so try to read the data from CSV
                waveform = self._read_csv_waveform(ch, cw, wf_name)
                # Check whether  we got something
                if waveform is None:
                    log.debug(f"{self.devname}: Waveform CSV does not exist, initializing to zeros.")
                    # Nope, initialize to zeros
                    waveform = np.zeros(32)
                    self._awg_waveforms[wf_name]['waveform'] = waveform
                    # write the CSV file
                    self._write_csv_waveform(ch, cw, wf_name, waveform)
                else:
                    # Got data, update dictionary
                    self._awg_waveforms[wf_name]['waveform'] = waveform

            # Get the waveform data from our dictionary, which must now
            # have the data
            return self._awg_waveforms[wf_name]['waveform']

        return read_func

    def _read_csv_waveform(self, ch: int, cw: int, wf_name: str):
        filename = os.path.join(
            self._get_awg_directory(), 'waves',
            self.devname + '_' + wf_name + '.csv')
        try:
            log.debug(f"{self.devname}: reading waveform from csv '{filename}'")
            return np.genfromtxt(filename, delimiter=',')
        except OSError as e:
            # if the waveform does not exist yet dont raise exception
            log.warning(e)
            return None

    def _length_match_waveforms(self, awg_nr):
        """
        Adjust the length of a codeword waveform such that each individual
        waveform of the pair has the same length
        """
        log.info(f'{self.devname}: Length matching waveforms for dynamic waveform upload.')
        wf_table = self._get_waveform_table(awg_nr)

        matching_updated = False
        iter_id = 0
        # We iterate over the waveform table
        while(matching_updated or iter_id == 0):

            iter_id += 1
            if iter_id > 10:
                raise StopIteration
            log.info(f'{self.devname}: Length matching iteration {iter_id}.')
            matching_updated = False

            for wf_name, other_wf_name in wf_table:
                len_wf = len(self._awg_waveforms[wf_name]['waveform'])
                len_other_wf = len(self._awg_waveforms[other_wf_name]['waveform'])

                # First one is shorter
                if len_wf < len_other_wf:
                    log.info(f"{self.devname}: Modifying {wf_name} for length matching.")
                    # Temporarily unset the readonly flag to be allowed to append zeros
                    readonly = self._awg_waveforms[wf_name]['readonly']
                    self._awg_waveforms[wf_name]['readonly'] = False
                    self.set(wf_name, np.concatenate(
                        (self._awg_waveforms[wf_name]['waveform'], np.zeros(len_other_wf-len_wf))))
                    self._awg_waveforms[wf_name]['dirty'] = True
                    self._awg_waveforms[wf_name]['readonly'] = readonly
                    matching_updated = True
                elif len_other_wf < len_wf:
                    log.info(f"{self.devname}: Modifying {other_wf_name} for length matching.")
                    readonly = self._awg_waveforms[other_wf_name]['readonly']
                    self._awg_waveforms[other_wf_name]['readonly'] = False
                    self.set(other_wf_name, np.concatenate(
                        (self._awg_waveforms[other_wf_name]['waveform'], np.zeros(len_wf-len_other_wf))))
                    self._awg_waveforms[other_wf_name]['dirty'] = True
                    self._awg_waveforms[other_wf_name]['readonly'] = readonly
                    matching_updated = True

    def _clear_dirty_waveforms(self, awg_nr):
        """
        Adjust the length of a codeword waveform such that each individual
        waveform of the pair has the same length
        """
        log.info(f"{self.devname}: Clearing dirty waveform tag for AWG {awg_nr}")
        for cw in range(self._num_codewords):
            wf_name = gen_waveform_name(2*awg_nr+0, cw)
            self._awg_waveforms[wf_name]['dirty'] = False

            other_wf_name = gen_waveform_name(2*awg_nr+1, cw)
            self._awg_waveforms[other_wf_name]['dirty'] = False

    def _clear_readonly_waveforms(self, awg_nr):
        """
        Clear the read-only flag of all configured waveforms. Typically used when switching
        configurations (i.e. programs).
        """
        for cw in range(self._num_codewords):
            wf_name = gen_waveform_name(2*awg_nr+0, cw)
            self._awg_waveforms[wf_name]['readonly'] = False

            other_wf_name = gen_waveform_name(2*awg_nr+1, cw)
            self._awg_waveforms[other_wf_name]['readonly'] = False

    def _set_readonly_waveform(self, ch: int, cw: int):
        """
        Mark a waveform as being read-only. Typically used to limit which waveforms the user
        is allowed to change based on the overall configuration of the instrument and the type
        of AWG program being executed.
        """
        # Sanity check
        if cw >= self._num_codewords:
            raise ziConfigurationError(
                'Codeword {} is out of range of the configured number of codewords ({})!'.format(cw, self._num_codewords))

        if ch >= self._num_channels():
            raise ziConfigurationError(
                'Channel {} is out of range of the configured number of channels ({})!'.format(ch, self._num_channels()))

        # Name of this waveform
        wf_name = gen_waveform_name(ch, cw)

        # Check if the waveform data is in our dictionary
        if wf_name not in self._awg_waveforms:
            raise ziConfigurationError(
                'Trying to mark waveform {} as read-only, but the waveform has not been configured yet!'.format(wf_name))

        self._awg_waveforms[wf_name]['readonly'] = True

    def _upload_updated_waveforms(self, awg_nr):
        """
        Loop through all configured waveforms and use dynamic waveform uploading
        to update changed waveforms on the instrument as needed.
        """
        # Fixme. the _get_waveform_table should also be implemented for the UFH
        log.info(f"{self.devname}: Using dynamic waveform update for AWG {awg_nr}.")
        wf_table = self._get_waveform_table(awg_nr)

        for dio_cw, (wf_name, other_wf_name) in enumerate(wf_table):
            if self._awg_waveforms[wf_name]['dirty'] or self._awg_waveforms[other_wf_name]['dirty']:
                # Combine the waveforms and upload
                wf_data = merge_waveforms(self._awg_waveforms[wf_name]['waveform'],
                                          self._awg_waveforms[other_wf_name]['waveform'])
                # Write the new waveform
                # print('DEBUG::upload_updated_waveforms awg_nr={}; dio_cw={}\n'.format(awg_nr,dio_cw))
                # print('DEBUG::upload_updated_waveforms {}'.format(wf_data))
                self.setv(
                    'awgs/{}/waveform/waves/{}'.format(awg_nr, dio_cw), wf_data)

    def _codeword_table_preamble(self, awg_nr):
        """
        Defines a snippet of code to use in the beginning of an AWG program in order to define the waveforms.
        The generated code depends on the instrument type. For the HDAWG instruments, we use the setDIOWaveform
        function. For the UHF-QA we simply define the raw waveforms.
        """
        raise NotImplementedError('Virtual method with no implementation!')

    def _configure_awg_from_variable(self, awg_nr):
        """
        Configures an AWG with the program stored in the object in the self._awg_program[awg_nr] member.
        """
        if self._awg_program[awg_nr] is not None:
            log.info(f"{self.devname}: Configuring AWG {awg_nr} with predefined codeword program")
            full_program = \
                '// Start of automatically generated codeword table\n' + \
                self._codeword_table_preamble(awg_nr) + \
                '// End of automatically generated codeword table\n' + \
                self._awg_program[awg_nr]

            self.configure_awg_from_string(awg_nr, full_program)
        else:
            # 1 line different from Delft version: reduced log level
            log.info(f"{self.devname}: No program configured for awg_nr {awg_nr}.")

    def _write_cmd_to_logfile(self, cmd):
        if self._logfile is not None:
            now = datetime.now()
            now_str = now.strftime("%d/%m/%Y %H:%M:%S")
            self._logfile.write(f'#{now_str}\n')
            self._logfile.write(f'{self.name}.{cmd}\n')

    def _flush_logfile(self):
        if self._logfile is not None:
            self._logfile.flush()

    ##########################################################################
    # Public methods: node helpers
    ##########################################################################

    def setd(self, path, value) -> None:
        self._write_cmd_to_logfile(f'daq.setDouble("{path}", {value})')
        if self._async_mode:
            self.daq.asyncSetDouble(self._get_full_path(path), value)
        else:
            self.daq.setDouble(self._get_full_path(path), value)

    def getd(self, path):
        return self.daq.getDouble(self._get_full_path(path))

    def seti(self, path, value) -> None:
        self._write_cmd_to_logfile(f'daq.setDouble("{path}", {value})')
        if self._async_mode:
            self.daq.asyncSetInt(self._get_full_path(path), value)
        else:
            self.daq.setInt(self._get_full_path(path), value)

    def geti(self, path):
        return self.daq.getInt(self._get_full_path(path))

    def sets(self, path, value) -> None:
        self._write_cmd_to_logfile(f'daq.setString("{path}", {value})')
        if self._async_mode:
            self.daq.asyncSetString(self._get_full_path(path), value)
        else:
            self.daq.setString(self._get_full_path(path), value)

    def gets(self, path):
        return self.daq.getString(self._get_full_path(path))

    def setc(self, path, value) -> None:
        self._write_cmd_to_logfile(f'daq.setComplex("{path}", {value})')
        self.daq.setComplex(self._get_full_path(path), value)

    def getc(self, path):
        return self.daq.getComplex(self._get_full_path(path))

    def setv(self, path, value) -> None:
        # Handle absolute path
        # print('DEBUG::setv {} {}'.format(path,value))
        if self.use_setVector:
            # self._write_cmd_to_logfile(f'daq.setVector("{path}", np.array({np.array2string(value, separator=",")}))')
            self.daq.setVector(self._get_full_path(path), value)
        else:
            self._write_cmd_to_logfile(f'daq.vectorWrite("{path}", np.array({np.array2string(value, separator=",")}))')
            self.daq.vectorWrite(self._get_full_path(path), value)

    def getv(self, path):
        path = self._get_full_path(path)
        value = self.daq.get(path, True, 0)
        if path not in value:
            raise ziValueError('No value returned for path ' + path)
        else:
            return value[path][0]['vector']

    def getdeep(self, path, timeout=5.0):
        path = self._get_full_path(path)

        self.daq.getAsEvent(path)
        while timeout > 0.0:
            value = self.daq.poll(0.01, 500, 4, True)
            if path in value:
                return value[path]
            else:
                timeout -= 0.01

        return None

    def subs(self, path:str) -> None:
        full_path = self._get_full_path(path)
        if full_path not in self._subscribed_paths:
            self._subscribed_paths.append(full_path)
        self.daq.subscribe(full_path)

    def unsubs(self, path:str=None) -> None:
        if path is None:
            for path in self._subscribed_paths:
                self.daq.unsubscribe(path)
            self._subscribed_paths.clear()
        else:
            full_path = self._get_full_path(path)
            if full_path in self._subscribed_paths:
                del self._subscribed_paths[self._subscribed_paths.index(full_path)]
            self.daq.unsubscribe(full_path)

    def poll(self, poll_time=0.1):
        return self.daq.poll(poll_time, 500, 4, True)

    def sync(self) -> None:
        self.daq.sync()

    ##########################################################################
    # Public methods
    ##########################################################################

    def start(self):
        log.info(f"{self.devname}: Starting '{self.name}'")
        self.check_errors()

        # Loop through each AWG and check whether to reconfigure it
        for awg_nr in range(self._num_awgs()):
            self._length_match_waveforms(awg_nr)

            # If the reconfiguration flag is set, upload new program
            if self._awg_needs_configuration[awg_nr]:
                log.debug(f"{self.devname}: Detected awg configuration tag for AWG {awg_nr}.")
                self._configure_awg_from_variable(awg_nr)
                self._awg_needs_configuration[awg_nr] = False
                self._clear_dirty_waveforms(awg_nr)
            else:
                log.debug(f"{self.devname}: Did not detect awg configuration tag for AWG {awg_nr}.")
                # Loop through all waveforms and update accordingly
                self._upload_updated_waveforms(awg_nr)
                self._clear_dirty_waveforms(awg_nr)

        # Start all AWG's
        for awg_nr in range(self._num_awgs()):
            # Skip AWG's without programs
            if self._awg_program[awg_nr] is None:
                # to configure all awgs use "upload_codeword_program" or specify
                # another program
                # 1 line different from Delft version: reduced log level
                log.info(f"{self.devname}: Not starting awg_nr {awg_nr}.")
                continue
            # Check that the AWG is ready
            if not self.get('awgs_{}_ready'.format(awg_nr)):
                raise ziReadyError(
                    'Tried to start AWG {} that is not ready!'.format(awg_nr))
            # Enable it
            self.set('awgs_{}_enable'.format(awg_nr), 1)
        log.info(f"{self.devname}: Started '{self.name}'")

    def stop(self):
        log.info(f"{self.devname}: Stopping '{self.name}'")
        # Stop all AWG's
        for awg_nr in range(self._num_awgs()):
            self.set('awgs_{}_enable'.format(awg_nr), 0)

        self.check_errors()

    # FIXME: temporary solution for issue
    def FIXMEclose(self) -> None:
        try:
            # Disconnect application server
            self.daq.disconnect()
        except AttributeError:
            pass
        super().close()

    def check_errors(self, errors_to_ignore=None):
        errors = json.loads(self.getv('raw/error/json/errors'))

        # If this is the first time we are called, log the detected errors, but don't raise
        # any exceptions
        if self._errors is None:
            raise_exceptions = False
            self._errors = {}
        else:
            raise_exceptions = True

        # Asserted in case errors were found
        found_errors = False

        # Combine errors_to_ignore with commandline
        _errors_to_ignore = copy.copy(self._errors_to_ignore)
        if errors_to_ignore is not None:
            _errors_to_ignore += errors_to_ignore

        # Go through the errors and update our structure, raise exceptions if anything changed
        for m in errors['messages']:
            code     = m['code']
            count    = m['count']
            severity = m['severity']
            message  = m['message']

            # 3 lines added to Delft version: severity 0 is no error
            if severity == 0:
                log.info(f'{self.devname}: Code {code}: "{message}" ({severity})')
                continue
            if not raise_exceptions:
                self._errors[code] = {
                    'count'   : count,
                    'severity': severity,
                    'message' : message}
                log.warning(f'{self.devname}: Code {code}: "{message}" ({severity})')
            else:
                # Check if there are new errors
                if code not in self._errors or count > self._errors[code]['count']:
                    if code in _errors_to_ignore:
                        # 1 line different from Delft version: reduced log level
                        log.info(f'{self.devname}: {message} ({code}/{severity})')
                    else:
                        log.error(f'{self.devname}: {message} ({code}/{severity})')
                        found_errors = True

                if code in self._errors:
                    self._errors[code]['count'] = count
                else:
                    self._errors[code] = {
                        'count'   : count,
                        'severity': severity,
                        'message' : message}

        if found_errors:
            log.error('Errors detected during run-time!')

    def clear_errors(self):
        self.seti('raw/error/clear', 1)

    def demote_error(self, code: str):
        """
        Demote a ZIRuntime error to a warning.

        Arguments
            code (str)
            The error code of the exception to ignore.
            The error code gets logged as an error before the exception
            is raised. The code is a string like "DIOCWCASE".
        """
        self._errors_to_ignore.append(code)

    def reset_waveforms_zeros(self):
        """
        Sets all waveforms to an array of 48 zeros.
        """
        t0 = time.time()

        wf = np.zeros(48)
        waveform_params = [value for key, value in self.parameters.items()
                           if 'wave_ch' in key.lower()]
        for par in waveform_params:
            par(wf)

        t1 = time.time()
        log.info(f"{self.devname}: Set all waveforms to zeros in {1.0e3 * (t1 - t0):.1f} ms")

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  timeout: float=15):
        """
        Uploads a program string to one of the AWGs in a UHF-QA or AWG-8.

        This function is tested to work and give the correct error messages
        when compilation fails.
        """
        log.info(f'{self.devname}: Configuring AWG {awg_nr} from string.')
        # Check that awg_nr is set in accordance with devtype
        self._check_awg_nr(awg_nr)

        t0 = time.time()
        success_and_ready = False
        # 2 lines added to Delft version: store statusstring
        if not hasattr(self, 'compiler_statusstring'):
            self.compiler_statusstring = ''

        # This check (and while loop) is added as a workaround for #9
        while not success_and_ready:
            # 3 lines different from Delft version: store and log statusstring
            new_statusstring = f'{self.devname}: Configuring AWG {awg_nr}...'
            log.info(new_statusstring)
            self.compiler_statusstring += new_statusstring

            self._awgModule.set('awgModule/index', awg_nr)
            self._write_cmd_to_logfile(f"_awgModule.set('awgModule/index', {awg_nr})")
            self._awgModule.set(
                'awgModule/compiler/sourcestring', program_string)
            self._write_cmd_to_logfile(f"_awgModule.set('awgModule/compiler/sourcestring', \'\'\'{program_string}\'\'\')")

            succes_msg = 'File successfully uploaded'

            # Success is set to False when either a timeout or a bad compilation
            # message is encountered.
            success = True
            while len(self._awgModule.get('awgModule/compiler/sourcestring')
                      ['compiler']['sourcestring'][0]) > 0:
                time.sleep(0.01)

                if (time.time()-t0 >= timeout):
                    success = False
                    raise TimeoutError(
                        'Timeout while waiting for compilation to finish!')

            comp_msg = (self._awgModule.get(
                'awgModule/compiler/statusstring')['compiler']
                ['statusstring'][0])

            if not comp_msg.endswith(succes_msg):
                success = False

            if not success:
                print("Compilation failed, printing program:")
                for i, line in enumerate(program_string.splitlines()):
                    print(i+1, '\t', line)
                print('\n')
                raise ziCompilationError(comp_msg)

            # Give the device one second to respond
            for i in range(10):
                ready = self.getdeep(
                    'awgs/{}/ready'.format(awg_nr))['value'][0]
                if ready != 1:
                    log.warning('AWG {} not ready'.format(awg_nr))
                    time.sleep(1)
                else:
                    success_and_ready = True
                    break

        t1 = time.time()
        # 5 lines different from Delft version: store and log statusstring (do not print)
        new_statusstring = (self._awgModule.get(
            'awgModule/compiler/statusstring')
              ['compiler']['statusstring'][0] + ' in {:.2f}s'.format(t1-t0))
        log.info(new_statusstring)
        self.compiler_statusstring += new_statusstring

        # Check status
        if self.get('awgs_{}_waveform_memoryusage'.format(awg_nr)) > 1.0:
            log.warning(f'{self.devname}: Waveform memory usage exceeds available internal memory!')

        if self.get('awgs_{}_sequencer_memoryusage'.format(awg_nr)) > 1.0:
            log.warning(f'{self.devname}: Sequencer memory usage exceeds available instruction memory!')

    def plot_dio_snapshot(self, bits=range(32)):
        raise NotImplementedError('Virtual method with no implementation!')

    def plot_awg_codewords(self, awg_nr=0, range=None):
        raise NotImplementedError('Virtual method with no implementation!')

    def get_idn(self) -> dict:
        idn_dict = {}
        idn_dict['vendor'] = 'ZurichInstruments'
        idn_dict['model'] = self.devtype
        idn_dict['serial'] = self.devname
        idn_dict['firmware'] = self.geti('system/fwrevision')
        idn_dict['fpga_firmware'] = self.geti('system/fpgarevision')

        return idn_dict

    def load_default_settings(self):
        raise NotImplementedError('Virtual method with no implementation!')

    def assure_ext_clock(self) -> None:
        raise NotImplementedError('Virtual method with no implementation!')

    def asyncBegin(self):
        self._async_mode = True

    def asyncEnd(self):
        self.daq.sync()
        self._async_mode = False
