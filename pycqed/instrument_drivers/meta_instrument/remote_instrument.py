"""Preliminary (hacky) code to access remote qcodes instruments

CAUTION: Use with care and see concerns below. This was just a quick and
    hacky draft, but was added to the master because it works well on multiple
    setups.

FIXME: Clean up and address concerns about security and about potential
    problems from version mismatches. Then build a proper module and
    add documentation.
    Concerns are, e.g., related to unpickling:
    - can be a security flaw (might be OK on local networks with trusted PCs)
    - will pickling and unpickling Exceptions lead to problems if different
      software versions run on client and server?
"""

import socket
import pickle
import traceback
import qcodes as qc
import time
from pycqed.instrument_drivers.instrument import FurtherInstrumentsDictMixIn

import logging
log = logging.getLogger(__name__)


Instrument = qc.instrument.base.Instrument
try:
    Metadatable = qc.metadatable.Metadatable
except AttributeError:
    # older qcode versions (pre 0.37)
    Metadatable = qc.utils.metadata.Metadatable
try:
    Parameter = qc.instrument.parameter.Parameter
except AttributeError:
    # older qcode versions (pre 0.37)
    Parameter = qc.parameters.Parameter

# The following default port was chosen arbitrarily (same default is used for
# the client class and the server function).
PORT = 65432
SUBMODULE_CACHE_TIME = 60  # seconds

class RemoteParameter():
    """Helper class to represent qcodes parameters of remote instruments

    Not meant to be instantiated manually by the user.
    """
    @property
    def __class__(self):
        return Parameter

    def __init__(self, instr, param):
        self.instrument = instr
        self.name = param

    def __call__(self, *args):
        return self.instrument.__getattr__(self.name)(*args)

    def get(self):
        return self.instrument.get(self.name)

    def set(self, value):
        self.instrument.set(self.name, value)


class RemoteInstrument(FurtherInstrumentsDictMixIn):
    """Allows to access a qcodes instrument on a remote server.

    CAUTION: Note the warning in the module docstring.
    """
    _remote_instruments = {}

    def __init__(self, name,
                 host, port=PORT,
                 id=None,
                 ):
        """Connect to a remote instrument.

        Args:
            name (str): name of the instrument (currently needs to be the same
                name on client and server)
            host (str): hostname or IP address of the server
            port (int): port on which the server is listening
            id: a unique identifier of the instrument. Not meant to be passed
                by the user. Used in recursive calls to avoid re-querying
                information that is already available on the client.
        """
        self._name = name
        self.host = host
        self.port = port
        if id is not None:
            self._id = id
        else:
            self._id = self.remote_call([self._name, 'id', '', [], []])
        self._remote_instruments[self._id] = self
        self._further_instruments[name] = self
        self._submodules = (None, 0)

    def _create_instr(self, name, id=None):
        if id is None:
            id = self.remote_call([name, 'id', '', [], []])
        if id in self._remote_instruments:
            return self._remote_instruments[id]
        return RemoteInstrument(name, self.host, self.port, id=id)

    def __getattr__(self, p):
        if self._name in Instrument._all_instruments:
            return getattr(Instrument._all_instruments[self._name], p)
        if p == 'submodules' and time.time() - self._submodules[1] < \
                SUBMODULE_CACHE_TIME:
            return self._submodules[0]
        if p == '__getstate__':
            if self._name != 'Pulsar':
                traceback.print_stack()

            def f():
                return self.__dict__

            return f
        result = self.remote_call([self._name, 'type', p, [], []])
        if result.startswith('submodule'):
            return self._create_instr(f'{self._name}.{p}',
                                       id=int(result.split(' ')[1]))
        elif result.startswith('channellist '):
            submods = self.submodules
            if p in submods:
                return submods[p]
            length = int(result[len('channellist '):])
            return [self._create_instr(f'{self._name}.{p}.{i}')
                    for i in range(length)]
        elif result == 'callable':
            def f(*args, **kwargs):
                return self.remote_call([self._name, None, p, args, kwargs])

            def get_instr(p=p):
                remote_instr = self.remote_call([self._name, None, p, [], {}])
                if remote_instr in Instrument._all_instruments:
                    return Instrument._all_instruments[remote_instr]
                else:
                    return self._create_instr(remote_instr)

            f.get_instr = get_instr
            return f
        else:
            result = self.remote_call([self._name, None, p])
            if p == 'parameters':
                result_new = {k: RemoteParameter(self, k) for k in result}
                for param, param_dict in result.items():
                    for k, v in param_dict.items():
                        setattr(result_new[param], k, v)
                result = result_new
            elif p == 'submodules':
                result_new = {k: self._create_instr(
                    f'{self._name}.{k}', id=v.get('_id'))
                    for k, v in result.items()}
                for subinst, attr_dict in result.items():
                    for k, v in attr_dict.items():
                        setattr(result_new[subinst], k, v)
                result = result_new
                self._submodules = (result, time.time())
            elif p in ['_all_instruments', 'components']:
                result = {k: self._create_instr(k) for k in result}
            return result

    def __setattr__(self, p, v):
        if p not in ['_name', 'host', 'port', '_id', '_submodules']:
            self.remote_call([self._name, 'set', p, v, []])
        else:
            super().__setattr__(p, v)

    def __getitem__(self, item):
        return self._create_instr(f'{self._name}.{item}')

    def remote_call(self, cmd):
        """Send a command to the server

        Args:
              cmd (str) the command in a format understood by the server

        Returns:
              the data returned by the server
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            data = pickle.dumps(cmd)
            step = 1024
            for i in range(0, len(data), step):
                new_data = data[i:i+step]
                s.sendall(new_data)
            s.sendall(b'done')
            data = b''
            while True:
                data_count = 0
                data += s.recv(1024 * 1024)
                more_data = False
                try:
                    data = pickle.loads(data)
                except pickle.UnpicklingError as e:
                    more_data = True
                    data_count += 1
                if not more_data:
                    if data_count > 0:
                        log.info(f'Data from the remote instrument has been '
                                 f'acquired in {data_count} blocks')
                    break
            if isinstance(data, Exception):
                if str(data).endswith('QCoDeS instruments can not be pickled.'):
                    log.warning(f'Error while accessing remote parameter '
                                f'{cmd[0]} {cmd[2]}: {str(data)}')
                    raise AttributeError(str(data))
                raise data
            else:
                return data


def server(port=PORT, host='127.0.0.1'):
    """Runs a server that allows clients to access qcodes instruments

    CAUTION: Note the warning in the module docstring.

    Args:
        port (int): port to which the listening socket should be bound
        host (str): identified the network interface to which the listening
            socket should be bound, see (host, port) in
            https://docs.python.org/3/library/socket.html#socket-families
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        while True:
            try:
                s.listen()
                s.setblocking(True)
                conn, addr = s.accept()
                with conn:
                    data = b''
                    while True:
                        new_data = conn.recv(1024)
                        if new_data == b'quit':
                            log.info('quit')
                            conn.sendall(b'Exiting')
                            return
                        data += new_data
                        if new_data[-4:] == b'done':
                            data = data[:-4]
                            break

                    cmd = pickle.loads(data)

                    try:
                        instr = cmd[0].split('.')
                        if len(instr[0]) == 0:
                            obj = qc.station
                        else:
                            obj = Instrument._all_instruments[instr[0]]
                        for k in instr[1:]:
                            if k.isnumeric():
                                obj = obj[int(k)]
                            else:
                                obj = getattr(obj, k)
                        if cmd[2] == '':
                            attr = obj
                        else:
                            attr = getattr(obj, cmd[2])
                        if cmd[1] == 'id':
                            result = id(attr)
                        elif cmd[1] == 'type':
                            if (isinstance(attr, Metadatable)
                                    and not isinstance(attr, Parameter)):
                                if isinstance(attr, qc.instrument.ChannelList):
                                    result = f'channellist {len(attr)}'
                                else:
                                    result = f'submodule {id(attr)}'
                            elif callable(attr):
                                result = 'callable'
                            else:
                                result = 'value'
                        elif cmd[1] == 'set':
                            result = setattr(obj, cmd[2], cmd[3])
                        elif callable(attr):
                            result = attr(*cmd[3], **cmd[4])
                        elif cmd[2] in ['parameters']:
                            attrs_to_copy = ['label', 'vals', 'unit']
                            result = {k: {a: getattr(p, a) for a
                                    in attrs_to_copy} for k, p in attr.items()}
                        elif cmd[2] in ['submodules']:
                            attrs_to_copy = []
                            result = {
                                k: {a: getattr(p, a) for a in
                                    attrs_to_copy} | {'_id': id(p)}
                                for k, p in attr.items()}
                        elif cmd[2] in ['_channels']:
                            attrs_to_copy = []
                            result = {k.name: {a: getattr(p, a) for a in
                                               attrs_to_copy}
                                      for k in attr}
                        elif cmd[2] in ['_all_instruments', 'components']:
                            result = list(attr)
                        else:
                            result = attr
                        data_out = pickle.dumps(result)
                    except Exception as e:
                        data_out = pickle.dumps(e)
                    conn.sendall(data_out)
            except ConnectionResetError:
                log.info('Client disconnected')
                pass
            except:
                raise
