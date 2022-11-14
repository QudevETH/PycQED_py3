import socket
import pickle
import traceback
import qcodes as qc
import time

import logging
log = logging.getLogger(__name__)


Instrument = qc.instrument.base.Instrument

PORT = 65432
SUBMODULE_CACHE_TIME = 60  # second

class RemoteParameter():
    @property
    def __class__(self):
        return qc.instrument.parameter.Parameter

    def __init__(self, instr, param):
        self.instrument = instr
        self.name = param

    def __call__(self, *args):
        return self.instrument.__getattr__(self.name)(*args)

    def get(self):
        return self.instrument.get(self.name)

    def set(self, value):
        self.instrument.set(self.name, value)


class RemoteInstrument():
    _remote_instruments = {}

    def __init__(self, name,
                 host, port=PORT,
                 id=None,
                 # instr_ref=None
                 ):
        self._name = name
        self.host = host
        self.port = port
        if id is not None:
            self._id = id
        else:
            self._id = self.remote_call([self._name, 'id', '', [], []])
        self._remote_instruments[self._id] = self
        self._submodules = (None, 0)
        # self.instr_ref = instr_ref

    #         if self.instr_ref:
    #             self.__repr__ = self.__str__

    def _create_instr(self, name, id=None):
        if id is None:
            id = self.remote_call([name, 'id', '', [], []])
        if id in self._remote_instruments:
            return self._remote_instruments[id]
        return RemoteInstrument(name, self.host, self.port, id=id)

    def __getattr__(self, p):
        # print(self._name, 'get', p, flush=True)
        if self._name in Instrument._all_instruments:
            return getattr(Instrument._all_instruments[self._name], p)
        if p == 'submodules' and  time.time() - self._submodules[1] < \
                SUBMODULE_CACHE_TIME:
            return self._submodules[0]
        if p == '__getstate__':
            if self._name != 'Pulsar':
                traceback.print_stack()

            def f():
                return self.__dict__

            return f
        result = self.remote_call([self._name, 'type', p, [], []])
        # print(result)
        if result.startswith('submodule'):
            return self._create_instr(f'{self._name}.{p}',
                                       id=int(result.split(' ')[1]))
        #         elif result == 'instr_ref':
        #             return RemoteInstrument(f'{self._name}.{p}', p)
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
                # print(remote_instr)
                if remote_instr in Instrument._all_instruments:
                    return Instrument._all_instruments[remote_instr]
                else:
                    return self._create_instr(remote_instr)

            f.get_instr = get_instr
            return f
        else:
            result = self.remote_call([self._name, None, p])
            #             print(result)
            if p == 'parameters':
                # result_new = {k: lambda *args, k=k: self.__getattr__(k)(*args)
                #           for k in result}
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

    #     @property
    #     def _instance(self):
    #         return self

    def get_instance(self):
        if self._name in Instrument._all_instruments:
            return Instrument._all_instruments[self._name].get_instance()
        return self

    #     def get_instr(self):
    #         print(self._name)
    #         if self.instr_ref:
    #             remote_instr = self.remote_call([self._name, None, self.instr_ref])
    #             print(remote_instr)
    #             all_ins = qc.instrument.base.Instrument._all_instruments
    #             if remote_instr in all_ins:
    #                 return all_ins[remote_instr]
    #             else:
    #                 return RemoteInstrument(remote_instr)
    #         else:
    #             super().get_instr()

    def __setattr__(self, p, v):
        if p not in ['_name', 'host', 'port', '_id', '_submodules'
                     # 'instr_ref'
                     ]:
            if self._name in Instrument._all_instruments:
                return setattr(Instrument._all_instruments[self._name], p, v)
            # print('x', p)
            self.remote_call([self._name, 'set', p, v, []])
        else:
            super().__setattr__(p, v)

    #     def __str__(self):
    #         if self.instr_ref:
    #             return self.instr_ref
    #         else:
    #             super().__str__()

    def __getitem__(self, item):
        # result = self.remote_call([self._name, '', '_channels', [], []])
        # if item not in result:
        #     raise KeyError(f'Channel {item} not found in remote instrument '
        #                    f'{self._name}.')
        return self._create_instr(f'{self._name}.{item}')
        # if result == 'value':


    def remote_call(self, cmd):
        # print(cmd, flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            data = pickle.dumps(cmd)
            step = 1024
            for i in range(0, len(data), step):
                new_data = data[i:i+step]
                # print(new_data, flush=True)
                s.sendall(new_data)
            s.sendall(b'done')
            data = b''
            more_data = False
            while True:
                data += s.recv(1024 * 1024)
                more_data = False
                try:
                    data = pickle.loads(data)
                except pickle.UnpicklingError as e:
                    print('exception')
                    print(e)
                    more_data = True
                if not more_data:
                    break
                else:
                    print('more data')
                    print(data)
            if isinstance(data, Exception):
                if str(data).endswith('QCoDeS instruments can not be pickled.'):
                    log.warning(f'Error while accessing remote parameter '
                                f'{cmd[0]} {cmd[2]}: {str(data)}')
                    raise AttributeError(str(data))
                raise data
            else:
                # print(f"data = {data}")
                return data

    # print(f"Received {data!r}")


def server(port=PORT, host='127.0.0.1'):
    i = 1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        while True:
            try:
                s.listen()
                s.setblocking(True)
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    data = b''
                    while True:
                        new_data = conn.recv(1024)
                        # print(new_data, flush=True)
                        if new_data == b'quit':
                            print('quit')
                            conn.sendall(b'Exiting')
                            return
                        data += new_data
                        if new_data[-4:] == b'done':
                            data = data[:-4]
                            break

                    # if len(data) > 1000:
                    #     time.sleep(1)
                    #     print('test', flush=True)
                    #     time.sleep(1)
                    #     print('test2', flush=True)
                    #     time.sleep(1)
                    #
                    cmd = pickle.loads(data)
                    #
                    # if len(data) > 1000:
                    #     time.sleep(1)
                    #     print('test3', flush=True)
                    #     time.sleep(1)
                    #     print('test4', flush=True)
                    #     time.sleep(1)

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
                        print(cmd[:3])
                        if cmd[1] == 'id':
                            result = id(attr)
                        elif cmd[1] == 'type':
                            # print('type')
                            if (isinstance(attr, qc.utils.metadata.Metadatable)
                                    and not isinstance(attr,
                                                       qc.instrument.parameter.Parameter)):
                                if isinstance(attr, qc.instrument.ChannelList):
                                    result = f'channellist {len(attr)}'
                                else:
                                    result = f'submodule {id(attr)}'
                            #                                 elif isinstance(attr, qc.instrument.parameter.InstrumentRefParameter):
                            #                                     result = 'instr_ref'
                            elif callable(attr):
                                result = 'callable'
                            else:
                                result = 'value'
                        elif cmd[1] == 'set':
                            # print('set')
                            result = setattr(obj, cmd[2], cmd[3])
                        elif callable(attr):
                            # print('method')
                            result = attr(*cmd[3], **cmd[4])
                        elif cmd[2] in ['parameters']:
                            attrs_to_copy = ['label', 'vals', 'unit']
                            result = {k: {a: getattr(p, a) for a in attrs_to_copy} for k, p in attr.items()}
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
                        #                             elif isinstance(attr, qc.instrument.parameter.InstrumentRefParameter):
                        #                                 result = attr()
                        else:
                            # print('property', attr)
                            result = attr
                        data_out = pickle.dumps(result)
                    except Exception as e:
                        data_out = pickle.dumps(e)
                    conn.sendall(data_out)
                    i += 1
            except ConnectionResetError:
                print('Client disconnected')
                pass
            except:
                raise



