import types
from copy import deepcopy


class CallableValue():
    def __init__(self, val):
        self.value = val

    def __call__(self):
        return self.value


class InstrumentShadow():
    def __init__(self, instr):
        self.snapshot = instr.snapshot()['parameters']
        # self.parameters = dict()
        setattr(self, 'parameters', dict())
        for k, v in self.snapshot.items():
            # can't pickle a lambda function when spawning child process
            self.parameters[k] = CallableValue(v['value'])
            setattr(self, k, self.parameters[k])

    def get(self, k):
        return self.parameters[k]()


class PulsarShadow(InstrumentShadow):
    def __init__(self, instr):
        super().__init__(instr)
        self.channels = deepcopy(instr.channels)
        self.awgs = deepcopy(instr.awgs)

        # for k in ['find_awg_channels']:
        #     setattr(self, k, types.MethodType(getattr(instr.__class__, k), self))

        # for clock:
        instr.AWGs_prequeried(True)
        self._clocks = instr._clocks

    def clock(self, channel=None, awg=None):
        if channel is not None:
            awg = self.get('{}_awg'.format(channel))
        return self._clocks[awg]

    def find_awg_channels(self, awg):
        channel_list = []
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) == awg:
                channel_list.append(channel)

        return channel_list

