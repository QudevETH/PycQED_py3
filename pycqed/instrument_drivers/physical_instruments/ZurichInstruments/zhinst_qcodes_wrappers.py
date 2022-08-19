"""
    Qudev specific drivers ZI Qcodes instruments.
"""

import numpy as np
import logging
log = logging.getLogger(__name__)

from zhinst.qcodes import SHFSG as SHFSG_core
from qcodes.utils import validators
from qcodes.instrument.parameter import ManualParameter


class ZHInstMixin:

    @property
    def devname(self):
        return self.serial

    @property
    def daq(self):
        """Returns the ZI data server (DAQ).
        """
        return self.session.daq_server

    def check_server(self, kwargs):
        # Note: kwargs are passed without ** in order to allow modifying them.
        if kwargs.pop('server', None) == 'emulator':
            from pycqed.instrument_drivers.physical_instruments \
                .ZurichInstruments import ZI_base_qudev as zibase
            from zhinst.qcodes import session as ziqcsess
            daq = zibase.MockDAQServer.get_instance(
                kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 8004))
            self._session = ziqcsess.ZISession(
                server_host=kwargs.get('host', 'localhost'),
                connection=daq, new_session=False)
            return daq


class ZHInstSHFMixin:

    def configure_sine_generation(self, chid, enable=True, osc_index=0, freq=None,
                                  phase=0.0, gains=(0.0, 1.0, 1.0, 0.0),
                                  sine_generator_index=0,
                                  force_enable=False):
        """Configure the direct output of the SHF sine generators.

        Unless `force_enable` is set tor True, this method does not enable the
        sine output and instead only sets the corresponding flag in
        'self._sgchannel_sine_enable' which is used in :meth:`self.start`.

        Args:
            chid (str): Name of the SGChannel to configure
            enable (bool, optional): Enable of the sine generator. Also see
                comment above and parameter description of `force_enable`.
                Defaults to True.
            osc_index (int, optional): Index of the digital oscillator to be
                used. Defaults to 0.
            freq (float, optional): Frequency of the digital oscillator. If
                `None` the frequency of the oscillator will
                not be changed. Defaults to `None`.
            phase (float, optional): Phase of the sine generator.
                Defaults to 0.0.
            gains (tuple, optional): Tuple of floats of length 4. Structure:
                (sin I, cos I, sin Q, cos Q). Defaults to `(0.0, 1.0, 1.0, 0.0)`.
            sine_generator_index (int, optional): index of the sine generator to
                be used. Defaults to 0.
            force_enable (bool): In combination with `enable` this parameter
                determines whether the sine output is enabled. Defaults to False
        """
        if freq is None:
            freq = self.sgchannels[int(chid[2]) - 1].oscs[osc_index].freq()
        self.sgchannels[int(chid[2]) - 1].configure_sine_generation(
            enable=(enable and force_enable),
            osc_index=osc_index,
            osc_frequency=freq,
            phase=phase,
            gains=gains,
            sine_generator_index=sine_generator_index,
        )
        self._sgchannel_sine_enable[int(chid[2]) - 1] = enable

    def configure_internal_mod(self, chid, enable=True, osc_index=0, phase=0.0,
                               global_amp=0.5, gains=(1.0, - 1.0, 1.0, 1.0),
                               sine_generator_index=0):
        """Configure the internal modulation of the SG channel.

        Args:
            chid (str): Name of the SGChannel to configure
            enable (bool, optional): Enable of the digital modulation.
                Defaults to True.
            osc_index (int, optional): Index of the digital oscillator to be
                used. Defaults to 0.
            phase (float, optional): Phase of the digital modulation.
                Defaults to 0.0.
            global_amp (float, optional): Defaults to 0.5.
            gains (tuple, optional): Tuple of floats of length 4. Structure:
                (sin I, cos I, sin Q, cos Q). Defaults to (1.0, -1.0, 1.0, 1.0).
            sine_generator_index (int, optional): index of the sine generator to
                be used. Defaults to 0.
        """
        self.sgchannels[int(chid[2]) - 1].configure_pulse_modulation(
            enable=enable,
            osc_index=osc_index,
            osc_frequency=self.sgchannels[int(chid[2]) - 1].oscs[osc_index].freq(),
            phase=phase,
            global_amp=global_amp,
            gains=gains,
            sine_generator_index=sine_generator_index,
        )
        self.configure_sine_generation(chid,
            enable=False, # do not turn on the output of the sine
                          # generator for internal modulation
            osc_index=osc_index,
            sine_generator_index=sine_generator_index)

    def start(self):
        first_sg_awg = len(getattr(self, 'qachannels', []))
        for awg_nr, sgchannel in enumerate(self.sgchannels):
            if self._awg_program[awg_nr + first_sg_awg] is not None:
                sgchannel.awg.enable(1)
            if self._sgchannel_sine_enable[awg_nr]:
                sgchannel.sines[0].i.enable(1)
                sgchannel.sines[0].q.enable(1)

    def stop(self):
        for awg_nr, sgchannel in enumerate(self.sgchannels):
            sgchannel.awg.enable(0)
            if self._sgchannel_sine_enable[awg_nr]:
                sgchannel.sines[0].i.enable(0)
                sgchannel.sines[0].q.enable(0)


class SHFSG(SHFSG_core, ZHInstSHFMixin, ZHInstMixin):
    """QuDev-specific PycQED driver for the ZI SHFSG
    """

    def __init__(self, serial, *args, **kwargs):
        daq = self.check_server(kwargs)
        if daq is not None:
            daq.set_device_type(serial, 'SHFSG4')
        self._awg_source_strings = {}
        super().__init__(serial, *args, **kwargs)
        self._awg_program = [None] * len(self.sgchannels)
        self._sgchannel_sine_enable = [False] * len(self.sgchannels)

        self.add_parameter(
            'allowed_lo_freqs',
            initial_value=np.arange(1e9, 8.1e9, 100e6),
            parameter_class=ManualParameter,
            docstring='List of values that the center frequency (LO) is '
                      'allowed to take. As of now this is limited to steps '
                      'of 100 MHz.',
            set_parser=lambda x: list(np.atleast_1d(x).flatten()),
            vals=validators.MultiType(validators.Lists(), validators.Arrays(),
                                      validators.Numbers()))

    def store_awg_source_string(self, channel, awg_str):
        """
        Store AWG source strings to a private property for debugging.

        This function is called automatically when programming a QA
        channel via set_awg_program and currently still needs to be called
        manually after programming an SG channel. The source strings get
        stored in the dict self._awg_source_strings.

        Args:
             channel: the QA or SG channel object for which the AWG was
                programmed
            awg_str: the source string that was programmed to the AWG
        """
        key = channel.short_name[:2] + channel.short_name[-1:]
        self._awg_source_strings[key] = awg_str

