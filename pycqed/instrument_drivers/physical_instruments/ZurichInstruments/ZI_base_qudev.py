import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibase
import json
import os
import time
import numpy as np
import re
import copy
import fnmatch
import types

class ZI_base_instrument_qudev(zibase.ZI_base_instrument):
    """
    Class to override functionality of ZI_base_instrument using
    multi-inheritance in ZI_HDAWG_qudev and acquisition_devices.uhfqa.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._awg_source_strings = {}

    def start(self, **kw):
        """
        Start the sequencer
        :param kw: currently ignored, added for compatibilty with other
        instruments that accept kwargs in start().
        """
        super().start()  # ZI_base_instrument.start() does not expect kwargs

    def store_awg_source_string(self, awg_nr, awg_str):
        """Save awg source string in a private attribute for debugging

        Args:
            awg_nr (int): index of the channel pair (0-indexed)
            awg_str (str): source string to be stored
        """
        self._awg_source_strings[awg_nr] = awg_str

    def configure_awg_from_string(self, awg_nr: int, program_string: str,
                                  *args, **kwargs):
        self.store_awg_source_string(awg_nr, program_string)
        super().configure_awg_from_string(awg_nr, program_string,
                                          *args, **kwargs)

    def _add_codeword_waveform_parameters(self, num_codewords) -> None:
        """
        Override to remove Delft-specific functionality.
        """
        pass


class MockDAQServer(zibase.MockDAQServer):
    _instances = {}
    """Dict of existing instances, keys are tuples of host (str) and 
    port (int)."""

    @classmethod
    def get_instance(cls, host='localhost', port=8004):
        """
        Provides a MockDAQServer instance for the given host and port.
        A new instance is automatically created if no existing instance for
        the given combination of host and port is found.
        Collects new instances in cls._instances.

        Instantiating this class with this method allows to have multiple
        devices running on the same server/session.

        Arguments:
            host (str): the host where the ziDataServer is running.
            port (int): the port to connect to for the ziDataServer.
        """
        if cls._instances.get((host, port)) is None:
            cls._instances[(host, port)] = cls(host, port=port, apilevel=5)
        return cls._instances[(host, port)]

    def __init__(self, server, port, apilevel, verbose=False):
        """
        Mock ZI daq server.
        This class should be instantiated via the class method get_instance
        to ensure that all instruments belong to the same server/session.

        Arguments:
            server (str): the host where the ziDataServer is running.
            port (int): the port to connect to for the ziDataServer.
            apilevel (int) the API version level to use.
            verbose (bool): the port to connect to for the ziDataServer
                (default: False).
        """
        super().__init__(server, port, apilevel, verbose=verbose)
        self.host = server
        self.port = port
        self.devices = set()  # devices on the same server
        self.nodes['/zi/about/dataserver'] = {
            'type': 'String', 'value': self.__class__.__name__}
        self._device_types = {}
        # create syncSet methods
        for k in dir(self):
            if k.startswith('set') and len(k) > 3:
                if not hasattr(self, 'get' + k[3:]):
                    continue
                def syncset(self, path, value,
                            setter=getattr(self, k),
                            getter=getattr(self, 'get' + k[3:])):
                    setter(path, value)
                    return getter(path)
                setattr(self, f'syncS{k[1:]}',
                        types.MethodType(syncset, self))

    def listNodes(self, path):
        return json.loads(self.listNodesJSON(path + "/*"))

    def set_device_type(self, device, devtype):
        self._device_types[device] = devtype

    def awgModule(self):
        return MockAwgModule(self)

    def connectDevice(self, device, interface):
        if device in self._device_types:
            self.devtype = self._device_types[device]
        elif device.lower().startswith('dev12'):
            self.devtype = 'SHFQA4'
            # FIXME: it could also be 'SHFQC', but not clear how to
            #  distinguish by serial. A virtual SHFQC currently has to set it
            #  beforehand using set_device_type.

        # The line below is a hack to trick this class to allow multiple
        # devices. We set self.device to None here because it will be
        # overwritten in the super call for the current device. Then we add
        # the self.device to self.devices.
        self.device = None
        super().connectDevice(device, interface)
        pattern = f'/{self.device}/sgchannels/*'
        awg_nrs = list(np.unique([n[len(pattern) - 1:][:1] for n in
                                  self.nodes if fnmatch.fnmatch(n, pattern)]))
        for awg_nr in awg_nrs:
            for i in range(2048):
                self.nodes[f'/{self.device}/sgchannels/{awg_nr}/awg/' \
                           f'waveform/waves/{i}'] = {
                    'type': 'ZIVectorData', 'value': np.array([])}
        # The 3 lines below are a hack to allow multiple devices.
        self.devices.add(self.device)
        self._device_types[device] = self.devtype
        self.nodes['/zi/devices/connected']['value'] = ','.join(self.devices)

    def listNodesJSON(self, path):
        dd = {k: copy.copy(v) for k, v in self.nodes.items()
              if fnmatch.fnmatch(k, path)}
        defaults = {
            'Description': 'Description not available.',
            'Properties': 'Read, Write, Setting',
            'Unit': 'None',
        }
        for name, node in dd.items():
            node.pop('value')
            node['Type'] = node.pop('type')
            node['Node'] = name
            for k in defaults:
                if k not in node:
                    node[k] = defaults[k]
        return json.dumps(dd)

    def getInt(self, path):
        if '/qachannels/' in path and '/result/' in path:
            m = re.match(
                r'/(\w+)/qachannels/(\d+)/spectroscopy/result/acquired', path)
            if m:
                p = ('/' + m.group(1) + '/qachannels/' + m.group(2) +
                     '/spectroscopy/result')
                return (self.getInt(f'{p}/length')
                        * self.getInt(f'{p}/averages'))
        if '/scopes/' in path:
            m = re.match(r'/(\w+)/scopes/(\d+)/enable', path)
            if m and self.getInt('/' + m.group(1) + '/scopes/' + m.group(2)
                                 + '/single') and (
                    time.time() - self.nodes[path].get('timestamp', np.inf) >
                    .1):
                return 0  # emulate that single run finishes after 0.1s
        if '/sgchannels/' in path and '/awg/ready' in path:
            return 1
        elif '/sgchannels/' in path:
            m = re.match(r'/(\w+)/sgchannels/(\d+)/synthesizer', path)
            if m:
                return (1 + int(m.group(2)) // 2)
        if '/awgs/' in path and '/ready' in path:
            return 1

        if 'Options' in self.nodes[path]:
            raw_val = self.nodes[path]['value']
            vals = [k for k, v in self.nodes[path]['Options'].items()
                    if v.startswith(f'"{raw_val}"')]
            if len(vals):
                return int(vals[0])

        return super().getInt(path)

    def setInt(self, path, value):
        super().setInt(path, value)
        if fnmatch.fnmatch(path, '/*/qachannels/*/generator/reset'):
            self.nodes[path[:-5] + 'ready']['value'] = 0
        elif re.match(r'/(\w+)/scopes/(\d+)/enable', path):
            self.nodes[path]['timestamp'] = time.time()

    def get(self, path, flat=None, flags=None, settingsonly=True,
            excludevectors=False, **kw):
        # FIXME: settingsonly and excludevectors are currently only taken into
        #  account if the wildcard '*' is contained in the path. This behavior
        #  might be different from the behavior of the original DAQ server.
        if path not in self.nodes:
            paths = [p for prefix in ['', '/'] for p, v in self.nodes.items()
                     if fnmatch.fnmatch(p, prefix + path)
                     and ((not settingsonly)
                          or 'Setting' in v.get('Properties', ''))
                     and ((not excludevectors)
                          or v.get('type') != 'ZIVectorData')
                     ]
            if not len(paths):
                raise zibase.ziRuntimeError(
                    "Unknown node '" + path +
                    "' used with mocked server and device!")
            ret = {}
            for p in paths:
                ret.update(self.get(p, flat=flat, flags=flags))
            return ret

        l = None
        if '/qachannels/' in path and '/result/' in path:
            m = re.match(r'/(\w+)/qachannels/(\d+)/readout/result/data/'
                         r'(\d+)/wave', path)
            if m:
                l = self.getInt('/' + m.group(1) + '/qachannels/'
                                + m.group(2) + '/readout/result/length')
            m = re.match(
                r'/(\w+)/qachannels/(\d+)/spectroscopy/result/data/wave', path)
            if m:
                l = self.getInt('/' + m.group(1) + '/qachannels/' + m.group(2)
                                + '/spectroscopy/result/length')
        if '/scopes/' in path:
            m = re.match(r'/(\w+)/scopes/(\d+)/channels/(\d+)/wave', path)
            if m:
                l = self.getInt(
                    '/' + m.group(1) + '/scopes/' + m.group(2) + '/length')
        if l is not None:
            return {path: [{
                'vector': np.random.rand(l) + 1j * np.random.rand(l),
                'timestamp': 0,
            }]}
        if '/sgchannels/' in path:
            m = re.match(r'/(\w+)/sgchannels/(\d+)/awg/waveform/descriptors',
                         path)
            if m:
                val = []
                paths = [p for p in self.nodes if fnmatch.fnmatch(
                    p, '/' + m.group(1) + '/sgchannels/' + m.group(2) + '/awg'
                       '/waveform/waves/*')]
                for p in paths:
                    val.append({
                        'name': p.replace('/', '_'),
                        "filename": "",
                        "function": "",
                        "channels": "2",
                        "marker_bits": "0;0",
                        "length": "0",
                        "timestamp": "0",
                        "play_config": "0"
                    })
                return {path: [{'vector': json.dumps(
                    {'waveforms': val}), 'timestamp': 0}]}
        ret = super().get(path, flat, flags)
        ret[path][0]['timestamp'] = 0
        return ret

    def set(self, path, value=None, **kwargs):
        if value is None:
            [self.set(*v) for v in path]
            return
        if path not in self.nodes:
            paths = [p for p in self.nodes if fnmatch.fnmatch(p, path)]
            if len(paths):
                [self.set(p, value) for p in paths]
                return
            else:
                raise zibase.ziRuntimeError(
                    "Unknown node '" + path +
                    "' used with mocked server and device!")
        self.nodes[path]['value'] = value
        if re.match(r'/(\w+)/scopes/(\d+)/enable', path):
            self.nodes[path]['timestamp'] = time.time()

    def _load_parameter_file(self, filename: str):
        node_pars = super()._load_parameter_file(filename)
        for par in node_pars.values():
            node = par['Node'].split('/')
            parpath = '/' + self.device + '/' + '/'.join(node)
            if par['Type'] == 'ZIVectorData':
                self.nodes[parpath.lower()] = {
                    'type': par['Type'], 'value': np.array([])}
            if parpath.lower() in self.nodes:
                for k in ['Description', 'Options', 'Properties', 'Unit']:
                    if k in par:
                        self.nodes[parpath.lower()][k] = par[k]


class MockAwgModule(zibase.MockAwgModule):
    def __init__(self, daq):
        super().__init__(daq)
        self._sequencertype = None

    def set(self, path, value):
        if path[0] == '/':
            path = path[1:]
        if not path.startswith('awgModule/'):
            path = 'awgModule/' + path
        if path == 'awgModule/sequencertype':
            self._sequencertype = value
        elif path == 'awgModule/compiler/sourcestring':
            if self._sequencertype == "qa":
                # The compiled program is stored in _sourcestring
                self._sourcestring = value
                if self._index not in self._compilation_count:
                    raise zibase.ziModuleError(
                        'Trying to compile AWG program, but no AWG index has been configured!')

                if self._device is None:
                    raise zibase.ziModuleError(
                        'Trying to compile AWG program, but no AWG device has been configured!')

                self._compilation_count[self._index] += 1
                self._daq.setInt('/' + self._device + '/qachannels/'
                                 + str(self._index) + '/generator/ready', 1)
            else:
                # create directories for dumping elf files
                base_dir = self.getString('/directory')
                base_dir = os.path.join(base_dir, "awg/elf")
                os.makedirs(base_dir, exist_ok=True)
                fn = os.path.join(base_dir, f'{self._device}_{self._index}_awg_default.elf')
                with open(fn, "w") as f:
                    f.write('')
                path = f'/{self._device}/awgs/{self._index}/ready'
                if path in self._daq.nodes:
                    self._daq.setInt(path, 1)
        else:
            super().set(path, value)

    def getInt(self, path):
        if path == 'compiler/status':
            return 0
        elif path == 'elf/status':
            return 0
        else:
            return 0

    def getString(self, path):
        if path == 'compiler/statusstring':
            return 'File successfully uploaded'
        elif path == '/directory':
            from pycqed.utilities.general import get_pycqed_appdata_dir
            return get_pycqed_appdata_dir()
        else:
            return ''

    def getDouble(self, path):
        if path == '/progress':
            return 1
        else:
            return 0

    def listNodesJSON(self, path):
        filename = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'zi_parameter_files', 'node_doc_awgModule.json')
        with open(filename) as fo:
            f = fo.read()
        dd = json.loads(f)

        defaults = {
            'Description': 'Description not available.',
            'Properties': 'Read, Write, Setting',
            'Unit': 'None',
        }
        for name, node in dd.items():
            node.pop('value', None)
            node['Type'] = node.pop('type', node.get('Type', 'String'))
            node['Node'] = name
            for k in defaults:
                if k not in node:
                    node[k] = defaults[k]
        return json.dumps(dd)

    def finish(self):
        pass