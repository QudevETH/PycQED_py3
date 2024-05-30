# A Segment is the building block of Sequence Class. They are responsible
# for resolving pulse timing, Z gates, generating trigger pulses and adding
# charge compensation
#
# author: Michael Kerschbaum
# created: 4/2019

import numpy as np
import math
import logging

import datetime
from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse as bpl
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.waveform_control.block as block_mod
import pycqed.measurement.waveform_control.fluxpulse_predistortion as flux_dist
from collections import OrderedDict as odict
import re
from pycqed.utilities.general import temporary_value
import functools


def _with_pulsar_tmp_vals(f):
    """A decorator enabling the usage of temporary values for plotting & hashing
       
       The temporary values are collected from self.pulsar_tmp_vals.
    """
    @functools.wraps(f)
    def wrapped_func(self, *args, **kwargs):
        tmp_vals = [(self.pulsar.parameters[p], v)
                    for p, v in self.pulsar_tmp_vals]
        with temporary_value(*tmp_vals):
            return f(self, *args, **kwargs)
    wrapped_func.__name__ = f.__name__
    return wrapped_func


class Segment:
    """
    Consists of a list of UnresolvedPulses, each of which contains information 
    about in which element the pulse is played and when it is played 
    (reference point + delay) as well as an instance of class Pulse.

    Property distortion_dicts: a key of the form {AWG}_{channel} specifies
        that the respective val should be used as distortion dict instead of
        self.pulsar.{AWG}_{channel}_distortion_dict.
    """

    trigger_pulse_length = 20e-9
    trigger_pulse_amplitude = 0.5
    trigger_pulse_start_buffer = 25e-9

    # When internal modulation of a channel is turned on, the following
    # parameters should be the same for all pulses on this channel.
    internal_mod_pulse_params_to_check = [
        "phi_skew",
        "alpha"
    ]

    PHASE_ROUNDING_DIGITS = 5
    """Specifies the rounding precision when processing phases 
    in resolve_Z_gates method. If this parameter has value n, then the 
    waveform phase will be rounded to the n-th digit of degree."""

    FREQUENCY_ROUNDING_DIGITS = 3
    """Specifies the rounding precision when processing frequencies 
    in _internal_mod_update_params method. If this parameter has value n, then 
    the waveform frequency will be rounded to the n-th digit of Hz."""

    def __init__(self, name, pulse_pars_list=(), acquisition_mode='default',
                 fast_mode=False, **kw):
        """
        Initiate instance of Segment class.

        Args:
            name: Name of segment
            pulse_pars_list: list of pulse parameters in the form
                of dictionaries
            acquisition_mode (dict or string): This will be copied into the acq
                key of the element metadata of acquisition elements to inform
                Pulsar that waveforms need to be programmed in a way that is
                compatible with the given acquisition mode. Note:
                - Pulsar may fall back to the default acquisition mode if
                  the given mode is not available or not needed on the used
                  hardware.
                - If applicable, higher layer code of experiments that use a
                  special acquisition mode need to ensure that other parts of
                  pycqed (e.g., sweep_function) get configured in a
                  compatible manner.
                If acquisition_mode is a dict, allowed items currently include:
                - 'sweeper': use sweeper mode if available (for RO frequency
                  sweeps, e.g., resonator spectroscopy).
                - 'f_start', 'f_step' and 'n_step': Sweep parameters, in the
                  case of a hardware sweep on the SHFQA.
                - 'seqtrigger': if True, let the sequencer output an auxiliary
                  trigger when starting the acquisition
                - 'default' (default value): if this key is present in
                  acquisition_mode, indicates a normal acquisition element
                It can also be a string in older code (note that conditions
                such as "'sweeper' in acquisition_mode" work in both cases)
                See
                :class:`pycqed.measurement.waveform_control.pulsar.SHFQAPulsar`
                for allowed values.
            fast_mode (bool): Activates the following features to save
                execution time (default: False):
                - Avoiding copying pulses. In this case, the pulse_pars_list
                  passed to Segment will be modified
                - "Destroying" segments after resolving them, meaning that they
                  cannot be resolved again. Not assuming that segments can be
                  reused allows to use less deepcopy operations.
                - Not copying time values between calls to Pulse.waveforms.
                  This might be an issue in case someone has the weird idea
                  to modify tvals in Pulse.waveforms.
            kw (dict): Keyword arguments:

                * ``resolve_overlapping_elements``: flag that, if true, lets the
                  segment automatically resolve overlapping elements by combining
                  them in a single element.
        """
        self.name = name
        self.pulsar = ps.Pulsar.get_instance()
        self.unresolved_pulses = []
        self.resolved_pulses = []
        self.destroyed = False  # Used if fast_mode==True
        self.extra_pulses = []  # trigger and charge compensation pulses
        self.previous_pulse = None
        self._init_end_name = None  # to detect end of init block in self.add
        self._algo_start = {'search': self.pulsar.algorithm_start(),
                            'name': None, 'occ_counter': [0]}
        if self._algo_start['search'] in ['segment_start', 'init_start']:
            self._algo_start['name'] = self._algo_start['search']
            self._algo_start['search'] = None
        self.elements = odict()
        self.element_start_end = {}
        self._element_start_end_raw = {}
        self.elements_on_awg = {}
        self.elements_on_channel = {}
        self.element_metadata = {}
        self.distortion_dicts = {}
        self.pulsar_tmp_vals = []
        """temporary values for pulsar, specific to this segment, in the
        format [('param_name', val), ...]. This should only be used for
        virtual parameters that influence waveform generation (e.g., software
        channel delay) and not for physical device parameters."""
        self._channel_amps = {}
        # The sweep_params dict is processed by generate_waveforms_sequences
        # and allows to sweep values of nodes of ZI HDAWGs in a hard sweep.
        # Keys are of the form awgname_chid_nodename (with _ instead of / in
        # the node name) and values are lists of hard sweep values.
        # The segment will be repeated as many times as there are hard
        # sweep values given in this property.
        # FIXME: This is an experimental feature and needs to be further
        #  cleaned up and documented in the future.
        self.sweep_params = kw.pop('sweep_params', dict())
        # allow_filter specifies whether the segment can be filtered out in
        # a FilteredSweep
        self.allow_filter = False
        self.trigger_pars = {
            'pulse_length': self.trigger_pulse_length,
            'amplitude': self.trigger_pulse_amplitude,
            'buffer_length_start': self.trigger_pulse_start_buffer,
        }
        self.trigger_pars['length'] = self.trigger_pars['pulse_length'] + \
                                      self.trigger_pars['buffer_length_start']
        self._pulse_names = set()
        self.acquisition_elements = dict()
        self.acquisition_mode = acquisition_mode
        self.mod_config = kw.pop('mod_config', dict())
        self.sine_config = kw.pop('sine_config', dict())
        self.timer = Timer(self.name)
        self.pulse_pars = []
        self.is_first_segment = False
        self.fast_mode = fast_mode
        self.resolve_overlapping_elements = \
            kw.pop('resolve_overlapping_elements',
                   self.pulsar.resolve_overlapping_elements())
        for pulse_pars in pulse_pars_list:
            self.add(pulse_pars)

    def add(self, pulse_pars):
        """
        Checks if all entries of the passed pulse_pars dictionary are valid
        and sets default values where necessary. After that an UnresolvedPulse
        is instantiated.
        """
        if self.fast_mode:
            pars_copy = pulse_pars
        else:
            self.pulse_pars.append(deepcopy(pulse_pars))
            pars_copy = deepcopy(pulse_pars)

        # Makes sure that pulse name is unique
        if pars_copy.get('name') in self._pulse_names:
            raise ValueError(f'Name of added pulse already exists: '
                             f'{pars_copy.get("name")}')
        if pars_copy.get('name', None) is None:
            pars_copy['name'] = pulse_pars['pulse_type'] + '_' + str(
                len(self.unresolved_pulses))
        self._pulse_names.add(pars_copy['name'])

        # Makes sure that element name is unique within sequence of
        # segments by appending the segment name to the element name
        # and that RO pulses are put into an appropriate element if no
        # element_name was provided. In particular, if a RO pulse for a
        # measurement object does not have an element_name specified,
        # the pulse will be put into the first acquisition element not yet
        # used for this measurement object (or a new acquisition element is
        # created if needed).
        suffix = '_' + self.name
        if pars_copy.get('operation_type', None) == 'RO':
            # get measurement object by removing first part of op_code
            mobj = ' '.join(pars_copy.get('op_code', '').split(' ')[1:])
            if (elname := pars_copy.get('element_name')) is None:
                if not mobj:
                    log.warning(f"RO pulse {pars_copy['name']} has neither "
                                f"element_name nor op_code. This can lead to "
                                f"unexpected behavior.")
                # find the first acquisition element not yet used for this mobj
                for i in range(len(self.acquisition_elements) + 1):
                    elname = f'RO_element_{i + 1}'
                    if mobj not in self.acquisition_elements.get(
                            elname + suffix, []):
                        break
                pars_copy['element_name'] = elname
            # add element to dict of acquisition elements
            self.acquisition_elements.setdefault(elname + suffix, [])
            self.acquisition_elements[elname + suffix].append(mobj)
        elif pars_copy.get('element_name') is None:
            pars_copy['element_name'] = 'default'
        pars_copy['element_name'] += suffix

        new_pulse = UnresolvedPulse(pars_copy)

        if new_pulse.ref_pulse == 'previous_pulse':
            if self.previous_pulse != None:
                if self.previous_pulse.pulse_obj.name == self._init_end_name:
                    # end of the init block detected, reference following
                    # pulse to segment_start
                    new_pulse.ref_pulse = 'segment_start'
                else:
                    new_pulse.ref_pulse = self.previous_pulse.pulse_obj.name
            # if the first pulse added to the segment has no ref_pulse
            # it is reference to segment_start by default
            elif self.previous_pulse == None and \
                 len(self.unresolved_pulses) == 0:
                new_pulse.ref_pulse = 'segment_start'
            else:
                raise ValueError('No previous pulse has been added!')
        elif new_pulse.ref_pulse == 'init_start' and \
                new_pulse.pulse_obj.name.endswith('-|-start'):
            # generate name to automatically detect the end of the init block
            self._init_end_name = new_pulse.pulse_obj.name[:-len('start')] + \
                                   'end'
        if self._algo_start['search'] is not None:
            if block_mod.check_pulse_by_pattern(
                    pars_copy, self._algo_start['search'],
                    self._algo_start['occ_counter']):
                if self._algo_start['name'] is not None:
                    log.warning(
                        f"{self.name}: The algorithm start search pattern "
                        f"{self._algo_start['search']} had matched "
                        f"{self._algo_start['name']} before, but also matches "
                        f"{new_pulse.pulse_obj.name}.")
                else:
                    self._algo_start['name'] = new_pulse.pulse_obj.name

        self.unresolved_pulses.append(new_pulse)

        self.previous_pulse = new_pulse
        # if self.elements is odict(), the resolve_timing function has to be
        # called prior to generating the waveforms
        self.elements = odict()
        self.resolved_pulses = []

    def extend(self, pulses):
        """
        Adds sequentially all pulses to the segment
        :param pulses: list of pulses to add
        :return:
        """
        for p in pulses:
            self.add(p)

    @Timer()
    @_with_pulsar_tmp_vals
    def resolve_segment(self, allow_overlap=False,
                        store_segment_length_timer=True):
        """
        Top layer method of Segment class. After having addded all pulses,
            * pulse elements are updated to enforce single element per segment
                for the that AWGs configured this way.
            * the timing is resolved
            * the virtual Z gates are resolved
            * the trigger pulses are generated
            * the charge compensation pulses are added

        :param allow_overlap: (bool, default: False) see _test_overlap
        :param store_segment_length_timer: (bool, default: True) whether
            the segment length should be stored in the segment's Timer object
        """
        self._check_acquisition_elements()
        self.join_or_split_elements()
        self.resolve_timing()
        self.resolve_mirror()
        self.resolve_Z_gates()
        self.add_flux_crosstalk_cancellation_channels()
        if self.resolve_overlapping_elements:
            self.resolve_overlap()
        self.extra_pulses = []
        self.resolve_internal_modulation()
        self.gen_trigger_el(allow_overlap=allow_overlap)
        self.add_charge_compensation()
        if store_segment_length_timer:
            try:
                # FIXME: we currently store 1e3*length because datetime
                #  does not support nanoseconds. Find a cleaner solution.
                self.timer.checkpoint(
                    'length.dt', log_init=False, values=[
                        datetime.datetime.utcfromtimestamp(0)
                        + datetime.timedelta(microseconds=1e9*np.diff(
                                self.get_segment_start_end())[0])])
            except Exception as e:
                # storing segment length is not crucial for the measurement
                log.warning(f"Could not store segment length timer: {e}")

    def _check_acquisition_elements(self):
        mobjs = [set(m) for m in self.acquisition_elements.values()]
        if len(mobjs) and any([m != mobjs[0] for m in mobjs]):
            log.warning(
                'Inconsistent acquisition elements can lead to unexpected '
                'behavior. Usually, all acquisition elements should contain '
                'pulses for the same set of measurement objects.')

    def join_or_split_elements(self):
        self.resolved_pulses = []
        if self.destroyed:
            raise Exception(
                f'The unresolved_pulses list of segment {self.name} has been '
                'destroyed in a previous resolution in fast mode. The segment '
                'cannot be resolved again.')
        if self.fast_mode:
            self.destroyed = True
        default_ese_element = f'default_ese_{self.name}'
        index = 1
        join_or_split = {}
        for p in self.unresolved_pulses:
            channels = p.pulse_obj.masked_channels()
            chs_ese = set()
            chs_split = set()
            chs_def = set()
            is_RO = (p.operation_type == "RO")
            for ch in channels:
                if ch not in join_or_split:
                    join_or_split[ch] = self.pulsar.get_join_or_split_elements(ch)
                if join_or_split[ch] == 'ese':
                    chs_ese.add(ch)
                elif join_or_split[ch] == 'split':
                    if is_RO:
                        log.info(f'Splitting elements is not implemented'
                                    f'with RO pulses. Not splitting '
                                    f'{p.pulse_obj.name} on channel {ch}.')
                        chs_def.add(ch)
                    else:
                        chs_split.add(ch)
                else:
                    chs_def.add(ch)

            # add the pulses as default pulse if it has a default channel or
            # if it is not part of any channel (i.e. only used as a flag)
            p_def = None
            ch_mask_def = p.pulse_obj.channel_mask | (channels - chs_def)
            if len(chs_def) != 0 or \
                    len(chs_def) == 0 and len(chs_ese) == 0 and len(chs_split) == 0:
                if self.fast_mode:
                    p_def = p
                else:
                    p_def = deepcopy(p)
                self.resolved_pulses.append(p_def)

            if len(chs_ese) != 0:
                if self.fast_mode and len(chs_def) + len(chs_split) == 0:
                    p_ese = p
                else:
                    p_ese = deepcopy(p)
                channel_mask = channels - chs_ese
                p_ese.pulse_obj.channel_mask |= channel_mask

                el_name = default_ese_element
                p_ese.pulse_obj.element_name = el_name

                # modify p_ese in case there are already non-ese pulses added
                # to have ese pulse start at the same time
                if len(chs_def) != 0:
                    p_ese.ref_pulse = p.pulse_obj.name
                    p_ese.ref_point = 0
                    p_ese.ref_point_new = 0
                    p_ese.basis_rotation = {}
                    p_ese.delay = 0
                    p_ese.pulse_obj.name += '_ese'
                    p_ese.is_ese_copy = True

                if p_ese.pulse_obj.codeword == "no_codeword":
                    self.resolved_pulses.append(p_ese)
                else:
                    log.warning('enforce_single_element cannot use codewords, '
                                f'ignoring {p.pulse_obj.name} on channels '
                                f'{", ".join(list(channels))}')

            if len(chs_split) != 0:
                p_split = deepcopy(p)
                channel_mask = channels - chs_split
                p_split.pulse_obj.channel_mask |= channel_mask

                el_name = f'default_split_{index}_{self.name}'
                index += 1
                p_split.pulse_obj.element_name = el_name

                if len(chs_def) != 0:
                    p_split.ref_pulse = p.pulse_obj.name
                    p_split.ref_point = 0
                    p_split.ref_point_new = 0
                    p_split.basis_rotation = {}
                    p_split.delay = 0
                    p_split.pulse_obj.name += '_split'
                    p_split.is_ese_copy = True
                elif len(chs_ese) != 0:
                    raise NotImplementedError(f'Having pulse {p.pulse_obj.name}'
                                              f' distributed over only ese and '
                                              f'split channles is not '
                                              f'implemented.')

                if p_split.pulse_obj.codeword == "no_codeword":
                    self.resolved_pulses.append(p_split)
                else:
                    log.warning('Split pulses cannot use codewords, '
                                f'ignoring {p.pulse_obj.name} on channels '
                                f'{", ".join(list(channels))}')

            if p_def is not None:
                p_def.pulse_obj.channel_mask = ch_mask_def

    def resolve_timing(self, resolve_block_align=True):
        """
        For each pulse in the resolved_pulses list, this method:
            * updates the _t0 of the pulse by using the timing description of
              the UnresolvedPulse
            * saves the resolved pulse in the elements ordered dictionary by
              ascending element start time and the pulses in each element by
              ascending _t0
            * orderes the resolved_pulses list by ascending pulse middle

        :param resolve_block_align: (bool) whether to resolve alignment of
            simultaneous blocks (default True)
        """

        self.elements = odict()
        if self.resolved_pulses == []:
            self.join_or_split_elements()

        visited_pulses = {}
        t_shift = 0
        i = 0

        pulses = self.gen_refpoint_dict()

        # First resolve pulses referencing init_start (directly or indirectly).
        # Pulses referencing segment_start (directly or indirectly) will get
        # resolved afterwards (see end of following while loop).
        # Note that the while loop below accepts either pulses or an int
        # (interpreted as absolute time) as values in the dict. We set the
        # init_start time to absolute time 0 for now, but will shift all
        # pulses later on (at the end of the following while loop) so that
        # finally 'segment_start' will become the zero point.
        ref_pulses_dict = {'init_start': 0}
        # the ..._dict_all will collect all reference pulses, while the
        # ..._dict only contains the recently added ones.
        ref_pulses_dict_all = deepcopy(ref_pulses_dict)
        # resolve pulses referenced to those in ref_pulses_dict
        while len(ref_pulses_dict) > 0:
            # the ..._dict_new will become the ..._dict of the next
            # iteration, i.e., it collects the newly added pulses in the
            # current iteration, and those will be considered as recently
            # added pulses in the next iteration.
            ref_pulses_dict_new = {}
            for name, pulse in ref_pulses_dict.items():
                for p in pulses.get(name, []):
                    if isinstance(p.ref_pulse, list):
                        if p.pulse_obj.name in visited_pulses:
                            continue
                        if not all([ref_pulse in ref_pulses_dict_all for
                                    ref_pulse in p.ref_pulse]):
                            continue

                        t0_list = []
                        delay_list = [p.delay] * len(p.ref_pulse) if not isinstance(p.delay, list) else p.delay
                        ref_point_list = [p.ref_point] * len(p.ref_pulse) if not isinstance(p.ref_point, list) \
                            else p.ref_point

                        for (ref_pulse, delay, ref_point) in zip(p.ref_pulse, delay_list, ref_point_list):
                            t0_list.append(ref_pulses_dict_all[ref_pulse].pulse_obj.algorithm_time() + delay -
                                           p.ref_point_new * p.cached_length +
                                           ref_point * ref_pulses_dict_all[ref_pulse].cached_length)

                        if p.ref_function == 'max':
                            t0 = max(t0_list)
                        elif p.ref_function == 'min':
                            t0 = min(t0_list)
                        elif p.ref_function == 'mean':
                            t0 = np.mean(t0_list)
                        else:
                            raise ValueError('Passed invalid value for ' +
                                'ref_function. Allowed values are: max, min, mean.' +
                                ' Default value: max')
                    else:
                        if isinstance(pulse, (float, int)):
                            # ref_pulses_dict provided an already resolved
                            # timing instead of a pulse
                            t_ref = pulse
                        else:  # ref_pulses_dict provided a pulse
                            t_ref = pulse.pulse_obj._t0 + \
                                    p.ref_point * pulse.cached_length
                        t0 = t_ref + p.delay - p.ref_point_new * p.cached_length

                    p.pulse_obj._t0 = t0

                    # add p.name to reference list if it is used as a key
                    # in pulses
                    if p.pulse_obj.name in pulses:
                        ref_pulses_dict_new.update({p.pulse_obj.name: p})

                    visited_pulses[p.pulse_obj.name] = (t0, i, p)
                    i += 1

            if not len(ref_pulses_dict_new):  # Nothing further to do
                if 'segment_start' not in ref_pulses_dict_all:
                    # We have only resolved init pulses up to now, but we
                    # still need to resolve the main part (referenced to
                    # segment_start).
                    # let the main part of the segment start after the end
                    # of the last init pulse (or at time 0 if there is no init)
                    t_init_end = max(
                        [p.pulse_obj.algorithm_time() + p.cached_length
                         for (_, _, p) in visited_pulses.values()] or [0])
                    if self._algo_start['name'] == 'segment_start':
                        # remember to shift all init pulses so that
                        # segment_start will be at time 0
                        t_shift = t_init_end
                    # resolve pulses referencing segment_start directly in
                    # the next iteration and those referencing segment_start
                    # indirectly in the following iterations
                    ref_pulses_dict_new = {'segment_start': t_init_end}
                elif len(visited_pulses) == 0:
                    # main part already resolved, but no pulses visited
                    raise ValueError(
                        'No pulse references to the segment start!')
            ref_pulses_dict = ref_pulses_dict_new
            ref_pulses_dict_all.update(ref_pulses_dict_new)

        if len(visited_pulses) != len(self.resolved_pulses):
            log.error(f"{len(self.resolved_pulses)} pulses to be resolved, "
                      f"but only {len(visited_pulses)} pulses visited. "
                      f"Pulses that have not been visited:")
            vp = [p for _, _, p in visited_pulses.values()]
            for p in self.resolved_pulses:
                if p not in vp:
                    log.error(p)
            raise Exception('Not all pulses have been resolved.')

        if self._algo_start['search'] is not None:
            if self._algo_start['name'] is None:
                log.warning(
                    f"{self.name}: The algorithm start search pattern "
                    f"{self._algo_start['search']} did not match any pulse. "
                    f"Using segment_start as start time.")
                t_shift = t_init_end
            else:
                t_shift = [p.pulse_obj.algorithm_time()
                           for _, _, p in visited_pulses.values()
                           if p.pulse_obj.name == self._algo_start['name']][0]
        if t_shift:
            for ind, (t0, i_p, p) in visited_pulses.items():
                p.pulse_obj.algorithm_time(
                    p.pulse_obj.algorithm_time() - t_shift)
                visited_pulses[ind] = (t0 - t_shift, i_p, p)

        if resolve_block_align:
            re_resolve = False
            for _, _, p in visited_pulses.values():
                if p.block_align is not None:
                    n = p.pulse_obj.name
                    end_pulse = ref_pulses_dict_all[n[:-len('start')] + 'end']
                    simultaneous_end_pulse = ref_pulses_dict_all[
                        n[:n[:-len('-|-start')].rfind('-|-') + 3] +
                        'simultaneous_end_pulse']
                    Delta_t = p.block_align * (
                            simultaneous_end_pulse.pulse_obj.algorithm_time() -
                            end_pulse.pulse_obj.algorithm_time())
                    if abs(Delta_t) > 1e-14:
                        p.delay += Delta_t
                        re_resolve = True
                    p.block_align = None
            if re_resolve:
                self.resolve_timing(resolve_block_align=False)
                return

        # adds the resolved pulses to the elements OrderedDictionary
        for (t0, i, p) in sorted(visited_pulses.values()):
            if p.pulse_obj.element_name not in self.elements:
                self.elements[p.pulse_obj.element_name] = [p.pulse_obj]
            elif p.pulse_obj.element_name in self.elements:
                self.elements[p.pulse_obj.element_name].append(p.pulse_obj)

        # sort resolved_pulses by ascending pulse middle. Used for Z_gate
        # resolution
        for i in visited_pulses:
            t0 = visited_pulses[i][0]
            p = visited_pulses[i][2]
            visited_pulses[i] = (t0 + p.cached_length / 2,
                                 visited_pulses[i][1], p)

        self.resolved_pulses = [
            p for (_, _, p) in sorted(visited_pulses.values())]

    def add_flux_crosstalk_cancellation_channels(self):
        pulsar_calibration_key = self.pulsar.flux_crosstalk_cancellation()
        flux_channels = self.pulsar.flux_channels()
        cancellation_mtx = self.pulsar.flux_crosstalk_cancellation_mtx()
        shift_mtx = self.pulsar.flux_crosstalk_cancellation_shift_mtx()
        for p in self.resolved_pulses:
            calibration_key = getattr(p.pulse_obj,
                                      'crosstalk_cancellation_key', None)
            if calibration_key is None:
                calibration_key = pulsar_calibration_key
            if calibration_key in (True, None):
                calibration_key = 'default'
            if not calibration_key:
                continue
            if any([ch in flux_channels[calibration_key] for ch in
                    p.pulse_obj.channels]):
                p.pulse_obj.crosstalk_cancellation_channels = \
                    flux_channels[calibration_key]
                p.pulse_obj.crosstalk_cancellation_mtx = \
                    cancellation_mtx[calibration_key]
                p.pulse_obj.crosstalk_cancellation_shift_mtx = shift_mtx
                if p.pulse_obj.crosstalk_cancellation_shift_mtx is not None:
                    p.pulse_obj.crosstalk_cancellation_shift_mtx = \
                        p.pulse_obj.crosstalk_cancellation_shift_mtx\
                            .get(calibration_key, None)

    def resolve_internal_modulation(self):
        """Processes internal-modulation-relevant information for this
        segment. For every channel (AWG module) that this segment distributes
        over, this method will:
        1. Check if internal modulation for this channel is turned on in PycQED.
        2. Check if all pulses types on this channel are compatible with
        internal modulation.
        3. Check if the configurations of all pulses on this channel are
        compatible with each other for internal modulation.
        4. If all checks above pass, updates pulse parameter and pass
        modulation parameters to element metadata."""

        # Update dictionary {channel: element_name} to attribute
        # self.elements_on_channel
        self.update_channel_elements()
        # Create metadata dictionary entries for all element in this segment
        self._initialize_element_metadata()

        for channel in self.elements_on_channel.keys():
            # Only look at I channel internal modulation configurations. Q
            # channel configurations will be the corresponding I channel
            # configurations.
            if not self.pulsar.is_i_channel(cname=channel):
                continue

            # Check if this channel supports internal modulation, and if
            # internal modulation is turned on for this channel. If not,
            # we will skip this channel.
            if not self.pulsar.check_channel_parameter(
                awg=self.pulsar.get_channel_awg(channel).name,
                channel=channel,
                parameter_suffix="_internal_modulation"
            ):
                continue

            # check if all pulses types on this channel are compatible with
            # internal modulation. If not, we will print a warning message
            # and disable internal modulation on this channel.
            if not self._internal_mod_check_pulse_type(channel=channel):
                logging.warning(
                    f"In segment {self.name}: not all pulses supports "
                    f"internal modulation on channel {channel}. This channel "
                    f"will not be internally modulated in the current "
                    f"sequence."
                )
                continue

            # Check if the configurations of all pulses are compatible with
            # each other when we turn on internal modulation. If not, we will
            # print a warning message and not configure internal modulation on
            # this channel.
            pulse_params_allow_internal_mod, check_values = \
                self._internal_mod_check_pulse_params(channel=channel)
            if not pulse_params_allow_internal_mod:
                logging.warning(
                    f"In segment {self.name}: internal modulation "
                    f"parameters are not compatible among the pulses. This "
                    f"channel will not be internally modulated in the current "
                    f"sequence."
                )
                continue

            # We have made sure that internal modulation is applicable to
            # this channel. We will change pulse settings and pass modulation
            # configuration to element metadata.
            self._internal_mod_update_params(
                channel=channel,
                check_values=check_values,
            )

    def update_channel_elements(self):
        """Updates attribute self.elements_on_channel to a dictionary {
        channel: set(element_name on this channel)}"""
        self.elements_on_channel = {}
        for elname in self.elements:
            channels = self.get_element_channels(elname)
            for channel in channels:
                if channel not in self.elements_on_channel.keys():
                    self.elements_on_channel[channel] = set()
                self.elements_on_channel[channel].add(elname)

    def _internal_mod_check_pulse_type(self, channel: str,):
        """Check if all pulse types on this channel are compatible with
        internal modulation.

        Args:
            channel (str): name of the channel to check.

        Returns:
            pulse_type_allow_internal_mod (bool): Boolean value indication
                whether all pulses on this channel falls within the category
                where internal modulation is supported.
        """
        # Check all pulses in all elements on this channel.
        for element_name in self.elements_on_channel[channel]:
            for pulse in self.elements[element_name]:
                # An element can include pulses that are played in different
                # AWG channels. Here we only process the pulses that are
                # played on the current channel.
                if channel not in pulse.channels:
                    continue
                if not pulse.SUPPORT_INTERNAL_MOD:
                    return False
        return True

    def _internal_mod_check_pulse_params(
            self,
            channel: str,
    ):
        """Check if pulse parameters on this channel allows internal
        modulation. This requires (1) I and Q channel of each pulse come
        from the same channel pair, and the index of I channel is smaller.
        (2) parameters in the class variable internal_mod_pulse_params_to_check
        are the same for all pulses.

        Args:
            channel: name of the channel to check.

        Returns:
            pulse_parameter_allow_internal_mod (tuple): A tuple of 2. The
                first value is a Boolean indicating whether all pulse
                parameters are compatible with internal modulation. If the
                first value is True, the second value will be a dictionary
                of check parameter values. If the first value is False,
                the second value will be an empty dictionary.
        """

        # Create a dict with entries to record check parameter values for
        # pulses on this channel.
        check_values = {}
        for param in self.internal_mod_pulse_params_to_check:
            check_values[param] = None

        # Check all pulses in all elements on this channel.
        for element_name in self.elements_on_channel[channel]:
            for pulse in self.elements[element_name]:
                # An element can include pulses that are played in different
                # AWG channels. Here we only process the pulses that are
                # played on the current channel.
                if channel not in pulse.channels:
                    continue

                # Records check parameter values of the first pulse on this
                # channel, and compare the following pulse parameters with
                # this value.
                for param in self.internal_mod_pulse_params_to_check:
                    if check_values[param] is None:
                        check_values[param] = getattr(pulse, param)
                    elif check_values[param] != getattr(pulse, param):
                        return False, {}

                # Check if I and Q channel of this pulse belong to the
                # same channel pair and if they are in the correct order (the Q
                # channel index being larger than the I channel index).
                if not self.pulsar.is_channel_pair(
                        cname1=pulse.I_channel,
                        cname2=pulse.Q_channel,
                        require_ordered=True,
                ):
                    return False, {}
        return True, check_values

    def _internal_mod_update_params(
            self,
            channel: str,
            check_values: dict,
    ):
        """Pass modulation-relevant pulse parameters to element metadata and
        resets pulse parameters.

        Args:
            channel: (str) name of the channel to check.
            check_values: (dict) valid values collected from pulse parameter
                check.
        """
        if not (hasattr(self.mod_config, channel) or len(check_values)):
            # No internal modulation configuration for this channel.
            return

        mod_frequency = self._internal_mod_find_maximum_frequency(
            channel=channel)
        for elname in self.elements_on_channel[channel]:
            channel_metadata = dict()
            channel_metadata["mod_frequency"] = mod_frequency

            # Find the phase of the first pulse on this channel. Pass this
            # channel initial phase to element metadata.
            channel_metadata["phase"] = self._internal_mod_update_init_phase(
                channel=channel,
                elname=elname,
            )

            # Write internal modulation settings collected from pulses to
            # element metadata.
            for param in self.internal_mod_pulse_params_to_check:
                if param in check_values.keys():
                    channel_metadata[param] = check_values[param]

            # Write internal modulation settings passed from segment
            # initialization parameters to element metadata.
            for param, value in self.mod_config.get(channel, {}).items():
                if param in channel_metadata.keys():
                    raise RuntimeError(
                        f"In segment {self.name}: modulation configuration "
                        f"'{param}' has repetitive definition from segment "
                        f"initialization parameters and from pulse "
                        f"parameters. This may be caused by enabling"
                        f"'{channel}_internal_modulation' "
                        f"while doing spectroscopy measurement. Please "
                        f"disable the parameter when doing spectroscopy "
                        f"measurements."
                    )
                else:
                    channel_metadata[param] = value

            # Resets pulse parameters that are already passed to element
            # metadata.
            for pulse in self.elements[elname]:
                if channel in pulse.channels:
                    pulse.mod_frequency = round(
                        pulse.mod_frequency - mod_frequency,
                        self.FREQUENCY_ROUNDING_DIGITS)
                    pulse.alpha = 1
                    pulse.phi_skew = 0

            self.element_metadata[elname]["mod_config"][channel] = \
                channel_metadata

    def _internal_mod_update_init_phase(
            self,
            channel: str,
            elname: str,
    ):
        """Find the phase of the first pulse on the given channel within the
        given element. Pass this initial phase to element metadata and
        subtract this value from all pulses in this element.

        Args:
            channel (str): channel name.
            elname (str): element name.

        Return:
            init_phase (float): initial phase of the first pulse. If there is no
                actual pulse on the channel within this element, returns 0.
        """
        init_phase = 0.0
        for pulse in self.elements[elname]:
            if channel in pulse.channels:
                init_phase = pulse.phase
                break

        # init_phase will be included in the command table entries when internal
        # modulation is on.
        for pulse in self.elements[elname]:
            if channel in pulse.channels:
                pulse.phase = round(pulse.phase - init_phase,
                                    self.PHASE_ROUNDING_DIGITS)

        return init_phase

    def _internal_mod_find_maximum_frequency(
            self,
            channel: str,
    ):
        """Find the maximum pulse modulation frequency on this channel. For
        typical qubit/qutrit drive pulses, this frequency is the
        ge-transition modulation frequency.

        Args:
            channel (str): channel name.

        Returns:
            max_freq (float): max modulation frequency.
        """
        # Use a dictionary as a counter for modulation frequencies
        frequency_bin = dict()
        for elname in self.elements_on_channel[channel]:
            for pulse in self.elements[elname]:
                # Round frequency to avoid duplicates due to floating
                # point rounding granularity.
                if channel in pulse.channels:
                    rounded_freq = round(pulse.mod_frequency,
                                         self.FREQUENCY_ROUNDING_DIGITS)
                    if rounded_freq in frequency_bin.keys():
                        frequency_bin[rounded_freq] += 1
                    else:
                        frequency_bin[rounded_freq] = 1
        return max(frequency_bin.keys())

    def _initialize_element_metadata(self):
        """Create metadata dictionary entries for all elements in this
        segment."""

        for elname in self.elements.keys():
            self.element_metadata[elname] = {"mod_config": {}}

    def add_charge_compensation(self):
        """
        Adds charge compensation pulse to channels with pulsar parameter
        charge_buildup_compensation.
        """
        t_end = -float('inf')
        pulse_area = {}
        compensation_chan = set()

        # Find channels where charge compensation should be applied
        for c in self.pulsar.channels:
            if self.pulsar.get('{}_type'.format(c)) != 'analog':
                continue
            if self.pulsar.get('{}_charge_buildup_compensation'.format(c)):
                compensation_chan.add(c)

        # Caches {c: self.pulsar.get_trigger_group(c)}
        # Note that groups is modified by self.tvals
        groups = {}
        # * generate the pulse_area dictionary containing for each channel
        #   that has to be compensated the sum of all pulse areas on that
        #   channel + the name of the last element
        # * and find the end time of the last pulse of the segment
        for element in self.element_start_end.keys():
            # finds the channels of AWGs with that element
            awg_channels = set()
            for group in self.element_start_end[element]:
                chan = set(self.pulsar.get_trigger_group_channels(group))
                awg_channels = awg_channels.union(chan)

            tvals = None
            for pulse in self.elements[element]:
                # Find the end of the last pulse of the segment
                t_end = max(t_end, pulse.algorithm_time() + pulse.length)

                for c in pulse.masked_channels():
                    if c not in compensation_chan:
                        continue
                    if c not in groups:
                        groups[c] = self.pulsar.get_trigger_group(c)
                    if c not in pulse_area:
                        pulse_area[c] = [0, None]
                    if pulse.is_net_zero:
                        pulse_area[c][1] = element
                        continue

                    element_start_time = self.get_element_start(
                        element, groups[c])
                    pulse_start = self.time2sample(
                        pulse.element_time(element_start_time), channel=c)
                    pulse_end = self.time2sample(
                        pulse.element_time(element_start_time) + pulse.length,
                        channel=c)

                    # Calculate the tvals dictionary for the element
                    if tvals is None:
                        tvals = self.tvals(compensation_chan & awg_channels,
                                           element, groups)
                    pulse_area[c][0] += pulse.pulse_area(
                        c, tvals[c][pulse_start:pulse_end])
                    # Overwrite this entry for all elements. The last
                    # element on that channel will be the one that
                    # is saved.
                    pulse_area[c][1] = element

        # Add all compensation pulses to the last element after the last pulse
        # of the segment and for each element with a compensation pulse save
        # the pulse with the greatest length to determine the new length
        # of the element
        i = 1
        comp_i = 1
        comp_dict = {}
        longest_pulse = {}
        for c in pulse_area:
            comp_delay = self.pulsar.get(
                '{}_compensation_pulse_delay'.format(c))
            amp = self.pulsar.get('{}_amp'.format(c))
            amp *= self.pulsar.get('{}_compensation_pulse_scale'.format(c))

            # If pulse lenght was smaller than min_length, the amplitude will
            # be reduced
            length = abs(pulse_area[c][0] / amp)
            awg = self.pulsar.get('{}_awg'.format(c))
            min_length = self.pulsar.get(
                '{}_compensation_pulse_min_length'.format(awg))
            if length < min_length:
                length = min_length
                amp = abs(pulse_area[c][0] / length)

            if pulse_area[c][0] > 0:
                amp = -amp

            last_element = pulse_area[c][1]
            # for RO elements create a seperate element for compensation pulses
            if last_element in self.acquisition_elements:
                RO_group = groups[c]
                if RO_group not in comp_dict:
                    # FIXME We create a segment here, but it will never get
                    #  triggered because trigger pulses are generated before
                    #  calling add_charge_compensation. This bug causes
                    #  hard-to-debug behavior, e.g., when using FP-assisted
                    #  RO without enabling enforce_single_element for the
                    #  flux AWG.
                    log.warning(
                        'Segment: Creating a separate element for charge '
                        'compensation. This might let your experiment fail. '
                        'Have you forgotten to enable enforce_single_element '
                        'for the flux AWG?')
                    last_element = 'compensation_el{}_{}'.format(
                        comp_i, self.name)
                    comp_dict[RO_group] = last_element
                    self.elements[last_element] = []
                    self.element_start_end[last_element] = {RO_group: [t_end, 0]}
                    self.elements_on_awg[RO_group].append(last_element)
                    comp_i += 1
                else:
                    last_element = comp_dict[RO_group]

            kw = {
                'amplitude': amp,
                'buffer_length_start': comp_delay,
                'buffer_length_end': comp_delay,
                'pulse_length': length,
                'gaussian_filter_sigma': self.pulsar.get(
                    '{}_compensation_pulse_gaussian_filter_sigma'.format(c))
            }
            pulse = pl.BufferedSquarePulse(
                last_element, c, name='compensation_pulse_{}'.format(i), **kw)
            self.extra_pulses.append(pulse)
            i += 1

            # Set the pulse to start after the last pulse of the sequence
            pulse.algorithm_time(t_end)

            # Save the length of the longer pulse in longest_pulse dictionary
            group = groups[c]
            total_length = 2 * comp_delay + length
            longest_pulse[(last_element,group)] = \
                    max(longest_pulse.get((last_element,group),0), total_length)

            self.add_pulse_to_element(last_element, pulse)

        # Here we update the length of the modified elements manually because
        # calling self.element_start_length for an automatic calculation might
        # overwrite modifications that were potentially done in
        # self.gen_trigger_el.
        for (el, group) in longest_pulse:
            length_comp = longest_pulse[(el, group)]
            el_start = self.get_element_start(el, group)
            el_buffer = self.pulsar.min_element_buffer() or 0.
            new_end = t_end + length_comp + el_buffer
            awg = self.pulsar.get_awg_from_trigger_group(group)
            new_samples = self.time2sample(new_end - el_start, awg=awg)
            # make sure that element length is multiple of
            # sample granularity
            gran = self.pulsar.get('{}_granularity'.format(awg))
            if new_samples % gran != 0:
                new_samples += gran - new_samples % gran
            self.element_start_end[el][group][1] = new_samples

    def gen_refpoint_dict(self):
        """
        Returns a dictionary of UnresolvedPulses with their reference_points as
        keys.
        """

        pulses = {}
        for pulse in self.resolved_pulses:
            ref_pulse_list = pulse.ref_pulse
            if not isinstance(ref_pulse_list, list):
                ref_pulse_list = [ref_pulse_list]
            for p in ref_pulse_list:
                if p not in pulses:
                    pulses[p] = [pulse]
                else:
                    pulses[p].append(pulse)

        return pulses

    def gen_elements_on_awg(self):
        """
        Updates the self.elements_on_AWG dictionary
        """

        if self.elements == odict():
            self.resolve_timing()

        self.elements_on_awg = {}
        groups = {}

        for element in self.elements:
            for pulse in self.elements[element]:
                for channel in pulse.masked_channels():
                    if channel not in groups:
                        groups[channel] = self.pulsar.get_trigger_group(channel)
                    group = groups[channel]
                    if group not in self.elements_on_awg:
                        self.elements_on_awg[group] = [element]
                    elif element not in self.elements_on_awg[group]:
                        self.elements_on_awg[group].append(element)

    def find_trigger_group_hierarchy(self):
        masters = {group for group in self.pulsar.trigger_groups
            if len(self.pulsar.get_trigger_channels(group)) == 0}

        # generate dictionary triggering_groups (keys are trigger
        # groups of triggering AWG and values are trigger groups
        # of triggered AWGs) and triggered_groups (keys are triggered
        # groups of AWGs and values are trigger groups of triggering AWGs)
        triggering_groups = {}
        triggered_groups = {}
        groups = self.pulsar.trigger_groups - masters
        for group in groups:
            for channel in self.pulsar.get_trigger_channels(group):
                trigger_group = self.pulsar.get_trigger_group(channel)
                if trigger_group not in triggering_groups:
                    triggering_groups[trigger_group] = []
                triggering_groups[trigger_group].append(group)
                if group not in triggered_groups:
                    triggered_groups[group] = []
                triggered_groups[group].append(trigger_group)

        # implement Kahn's algorithm to sort the trigger groups by hierarchy
        trigger_groups = masters
        group_hierarchy = []

        while len(trigger_groups) != 0:
            group = trigger_groups.pop()
            group_hierarchy.append(group)
            if group not in triggering_groups:
                continue
            for triggered_group in triggering_groups[group]:
                triggered_groups[triggered_group].remove(group)
                if len(triggered_groups[triggered_group]) == 0:
                    trigger_groups.add(triggered_group)

        group_hierarchy.reverse()
        return group_hierarchy

    def gen_trigger_el(self, allow_overlap=False):
        """
        For each resolved pulse with a nonempty list of trigger_channels:
            * instatiates a trigger pulse on each of the triggering channels,
              placed in a suitable element on the triggering AWG.
        For each element:
            For each AWG the element is played on, this method:
                * adds the element to the elements_on_AWG dictionary
                * instantiates a trigger pulse on the triggering channel of the
                  AWG, placed in a suitable element on the triggering AWG,
                  taking AWG delay into account.
                * adds the trigger pulse to the elements list

        For debugging, self.skip_trigger can be set to a list of trigger group
        names for which the triggering should be skipped (by using a 0-amplitude
        trigger pulse).

        Note the Pulsar parameters {AWG}_trigger_channels and
        trigger_pulse_parameters.

        :param allow_overlap: (bool, default: False) see _test_overlap
        """
        i = 1
        def add_trigger_pulses(trigger_pulses):
            if len(trigger_pulses) == 0:
                return

            nonlocal i
            # used for updating the length of the trigger elements after adding
            # the trigger pulses
            trigger_el_set = set()

            trig_pulse_params = self.pulsar.trigger_pulse_parameters()

            for ch, trigger_pulse_time, pars in trigger_pulses:
                trigger_group = self.pulsar.get_trigger_group(ch)
                # Find the element to play the trigger pulse in.
                # If there is no element on that AWG create a new element
                if self.elements_on_awg.get(trigger_group, None) is None:
                    trigger_element = f'trigger_element_{self.name}'
                # else find the element that is closest to the
                # trigger pulse
                else:
                    trigger_element = self.find_trigger_element(
                            trigger_group, trigger_pulse_time)

                # Get the default trigger pulse parameters
                kw = deepcopy(self.trigger_pars)
                # overwrite with parameters provided in pulsar for trigger
                # pulses on the current channel
                pars_keys = [ch]
                # FIXME The following check based on minimum time will fail
                #  if the same trigger channel is used both as a trigger
                #  channel in a pulse and as a trigger channel for an AWG.
                #  However, there should anyways not be any reasonable use
                #  case for this.
                if self.is_first_segment and trigger_pulse_time == min(
                        [tp[1] for tp in trigger_pulses if tp[0] == ch]):
                    # Possibly overwrite with special params for the first
                    # trigger pulse on this channel.
                    pars_keys += [f'{ch}_first']
                for k in pars_keys:
                    kw.update(trig_pulse_params.get(k, {}))
                # overwrite with parameters provided in trigger_pulses
                kw.update(pars)
                # Create the trigger pulse
                trig_pulse = pl.BufferedSquarePulse(
                    trigger_element,
                    channel=ch,
                    name='trigger_pulse_{}'.format(i),
                    **kw)
                self.extra_pulses.append(trig_pulse)
                i += 1
                trig_pulse.algorithm_time(trigger_pulse_time
                                          + kw.get('pulse_delay', 0)
                                          - 0.25/self.pulsar.clock(ch)
                                          - kw['buffer_length_start'])

                # Add trigger element and pulse to seg.elements
                if trig_pulse.element_name in self.elements:
                    self.add_pulse_to_element(trig_pulse.element_name,
                                              trig_pulse)
                else:
                    self.elements[trig_pulse.element_name] = [trig_pulse]

                # Add the trigger_element to elements_on_awg[trigger_group]
                if trigger_group not in self.elements_on_awg:
                    self.elements_on_awg[trigger_group] = [trigger_element]
                elif trigger_element not in self.elements_on_awg[trigger_group]:
                    self.elements_on_awg[trigger_group].append(trigger_element)

                trigger_el_set = trigger_el_set | {
                    (trigger_group, trigger_element)}

            # For all trigger elements update the start and length
            # after having added the trigger pulses
            for (awg, el) in trigger_el_set:
                self.element_start_length(el, awg)

        # Generate the dictionary elements_on_awg, that for each AWG contains
        # a list of the elements on that AWG
        self.gen_elements_on_awg()

        # First, add trigger pulses that are requested in pulse parameters
        # FIXME We need to test and possibly debug the case where multiple
        #  pulses requests triggers on the same channel at the same time.
        #  This situation will, e.g., arise when performing readout of
        #  multiple qubits on the same VC707 at the same time (which is not
        #  done currently).
        trigger_pulses = []
        for p in self.resolved_pulses:
            pobj = p.pulse_obj
            for ch in pobj.trigger_channels:
                trigger_pulses.append(
                    (ch, pobj.algorithm_time(), pobj.trigger_pars))
        add_trigger_pulses(trigger_pulses)

        # Find the AWG hierarchy. Needed to add the trigger pulses first to
        # the AWG that do not trigger any other AWGs, then the AWGs that
        # trigger these AWGs and so on.
        trigger_group_hierarchy = self.find_trigger_group_hierarchy()
        # Initialize variables for resolving the main trigger time
        masters, delays = {}, {}
        t_main_trig = np.inf

        for group in trigger_group_hierarchy:
            if group not in self.elements_on_awg:
                continue
            if len(self.pulsar.get_trigger_channels(group)) == 0:
                # master AWG directly triggered by main trigger
                # find and store the first element
                masters[group] = self.find_trigger_element(group, -np.inf)
                # determine required main trigger timer, taking into account
                # the delay settings of this master trigger group
                delays[group] = self.pulsar.get_trigger_delay(group)
                start_end = self.element_start_end[masters[group]][group]
                t_main_trig = min(start_end[0] + delays[group], t_main_trig)
                continue  # for master AWG no trigger_pulse has to be added

            trigger_pulses = []
            for element in self.elements_on_awg[group]:
                # Calculate the trigger pulse time
                [el_start, _] = self.element_start_length(element, group)

                trigger_pulse_time = el_start \
                                     + self.pulsar.get_trigger_delay(group)

                # Find the trigger channels that trigger the AWG
                for channel in self.pulsar.get_trigger_channels(group):
                    trigger_pulses.append(
                        (channel, trigger_pulse_time,
                         {'amplitude': 0}
                         if group in getattr(self, 'skip_trigger', []) else {}))

            add_trigger_pulses(trigger_pulses)

        # if a fixed main trigger time is set: check compatibility and
        # overwrite with fixed value
        t_main_trig_setting = self.pulsar.main_trigger_time()
        if t_main_trig_setting != 'auto':
            if t_main_trig < t_main_trig_setting:
                raise ValueError(
                    f'Fixed main trigger time {t_main_trig_setting} is too '
                    f'late for this segment, which starts at {t_main_trig}.')
            t_main_trig = t_main_trig_setting
        for group in masters:
            # update element start such that the waveforms start at the
            # main trigger time determined above, taking into account the
            # delay settings (subtract a positive/negative delay from the
            # start time = waveform gets more/less zeros at the start =
            # original waveform starts later/earlier)
            self.element_start_length(
                masters[group], group, t_start=t_main_trig - delays[group])

        # checks if elements on AWGs overlap
        self._test_overlap(allow_overlap=allow_overlap)
        # checks if there is only one element on the master AWG
        self._test_trigger_awg()

    def find_trigger_element(self, trigger_group, trigger_pulse_time):
        """
        For a trigger_group of an AWG that is used for generating triggers
        as well as normal pulses, this method returns the name of the
        element to which the trigger pulse is closest.
        """

        time_distance = []

        for element in self.elements_on_awg[trigger_group]:
            [el_start, samples] = self.element_start_length(
                element, trigger_group)
            el_end = el_start + self.sample2time(
                samples,
                awg=self.pulsar.get_awg_from_trigger_group(trigger_group))
            distance_start_end = [
                [
                    abs(trigger_pulse_time + self.trigger_pars['length'] / 2 -
                        el_start), element
                ],
                [
                    abs(trigger_pulse_time + self.trigger_pars['length'] / 2 -
                        el_end), element
                ]
            ]

            time_distance += distance_start_end

        trigger_element = min(time_distance)[1]

        return trigger_element

    def get_element_end(self, element, group):
        """
        This method returns the end of an element on an AWG in algorithm_time
        """

        samples = self.element_start_end[element][group][1]
        awg = self.pulsar.get_awg_from_trigger_group(group)
        length = self.sample2time(samples, awg=awg)
        return self.element_start_end[element][group][0] + length

    def get_element_start(self, element, group):
        """
        This method returns the start of an element on an AWG trigger group
        in algorithm_time
        """
        return self.element_start_end[element][group][0]

    def get_segment_start_end(self):
        """
        Returns the start and end of the segment in algorithm_time
        """
        for i in range(2):
            start_end_times = np.array(
                [[self.get_element_start(el, group),
                  self.get_element_end(el, group)]
                 for group, v in self.elements_on_awg.items() for el in v])
            if len(start_end_times) > 0:
                # the segment has been resolved before
                break
            # Resolve the segment and retry. We set store_segment_length_timer
            # to False to avoid that resolve_segment calls
            # get_segment_start_end, which might cause an infinite loop in
            # some pathological cases.
            self.resolve_segment(store_segment_length_timer=False)
        return np.min(start_end_times[:, 0]), np.max(start_end_times[:, 1])

    def _test_overlap(self, allow_overlap=False, tol=1e-12,
                      track_and_ignore=False):
        """
        Tests for all AWGs if any of their elements overlap.

        :param allow_overlap: (bool, default: False) If this is False,
            an execption is raised in case of overlapping elements.
            Otherwise, only a warning is shown (useful for plotting while
            debugging overlaps).
        :param track_and_ignore: flag that allows the code to continue
            in case an overlap is detected. The code keeps track of these
            elements by adding them to self.overlapping_elements
        """

        self.gen_elements_on_awg()
        overlapping_elements = []

        for group in self.elements_on_awg:
            el_list = []
            i = 0
            for el in self.elements_on_awg[group]:
                # add element and or group to element_start_end
                if el not in self.element_start_end or group not in \
                        self.element_start_end[el]:
                    self.element_start_length(el, group)
                el_list.append([self.element_start_end[el][group][0], i, el])
                i += 1

            el_list.sort()

            for i in range(len(el_list) - 1):
                prev_el = el_list[i][2]
                el_prev_start = self.get_element_start(prev_el, group)
                el_prev_end = self.get_element_end(prev_el, group)
                el_length = el_prev_end - el_prev_start

                # If element length is shorter than min length, 0s will be
                # appended by pulsar. Test for elements with at least
                # min_el_len if they overlap.
                min_el_len = self.pulsar.get('{}_min_length'.format(
                    self.pulsar.get_awg_from_trigger_group(group)))
                if el_length < min_el_len:
                    el_prev_end = el_prev_start + min_el_len

                el_new_start = el_list[i + 1][0]

                if (el_prev_end - el_new_start) > tol:
                    if track_and_ignore:
                        # add set of two overlapping elements to list
                        overlapping_elements.append({prev_el,
                                                          el_list[i + 1][2]})
                        # test whether any of the following elements also
                        # overlaps with previous element
                        for j in range(i+2, len(el_list)):
                            if el_prev_end - el_list[j][0] > tol:
                                overlapping_elements.append(
                                    {prev_el, el_list[j][2]})
                            else:
                                # once a successive element does not
                                # overlap, non of the following
                                # elements overlap either
                                break
                    else:
                        msg = f'{prev_el} (ends at {el_prev_end*1e6:.4f}us) ' \
                              f'and {el_list[i + 1][2]} (' \
                              f'starts at {el_new_start*1e6:.4f}us) overlap ' \
                              f'on {group}'
                        if allow_overlap:
                            log.warning(msg)
                        else:
                            raise ValueError(msg)

        if track_and_ignore:
            return overlapping_elements

    def resolve_overlap(self):
        """
        Routine to resolve overlapping elements. Will be exectued if
        self.resolve_overlapping_elements is True. This code
        first goes through the list of overlapping elements that are
        pairwise overlapping and clusters them into lists
        of overlapping elements, where two different lists of overlapping
        elements have no overlapping elements with
        one another. At the end the code combines all elements of each
        list into a new element.
        """
        self.gen_elements_on_awg()
        overlapping_elements = self._test_overlap(track_and_ignore=True)

        if len(overlapping_elements) == 0:
            return

        # add first two overlapping elements to list
        joint_overlapping_elements = [overlapping_elements[0]]

        new_cluster = True
        for i in range(len(overlapping_elements) - 1):
            # making use of overlapping elements being sorted
            # check whether the next set of elements from
            # overlapping_elements shares an element name with
            # the previous entry in joint_overlapping_elements
            if len(joint_overlapping_elements[-1] & \
                   overlapping_elements[i + 1]) != 0:
                joint_overlapping_elements[-1] = \
                    joint_overlapping_elements[-1] | \
                    overlapping_elements[i + 1]
                new_cluster = False

            # if the new element from overlapping_elements overlaps
            # with none of the previously added elements in
            # joint_overlapping_elements (i.e. if new_cluster=True)
            # add it as a new cluster.
            if new_cluster:
                joint_overlapping_elements.append(overlapping_elements[i + 1])
            new_cluster = True

        for i in range(len(joint_overlapping_elements)):
            self._combine_elements(joint_overlapping_elements[i],
                                   'overlapping_el_{}_{}'.format(i, self.name))


    def _combine_elements(self, elements, combined_el_name):
        """
        Routine to properly combine elements in the segment.
        :param elements: list or set of elements in the segment to be combined
        :param combined_el_name: name of the combined element
        :return:
        """

        new_pulse_list = []

        for el in elements:
            new_pulse_list += self.elements.pop(el)
            # remove it from element_start_end
            self.element_start_end.pop(el)

        # update pulse objects with new name of element
        for p in new_pulse_list:
            p.element_name = combined_el_name
        # add new element
        self.elements[combined_el_name] = new_pulse_list
        # update new elements_on_awg
        self.gen_elements_on_awg()

        # update element_start_end
        for group in self.pulsar.trigger_groups:
            self.element_start_length(combined_el_name, group)


    def _test_trigger_awg(self):
        """
        Checks if there is more than one element on the AWGs that are not
        triggered by another AWG.

        Note that we assume that gen_elements_on_awg has been called before.
        """
        for group in self.elements_on_awg:
            if len(self.pulsar.get_trigger_channels(group)) != 0:
                continue
            if len(self.elements_on_awg[group]) > 1:
                raise ValueError(
                    'There is more than one element on trigger '
                    'group {}'.format(group))

    def resolve_mirror(self):
        """
        Resolves amplitude mirroring for pulses that have a mirror_pattern
        property.

        Pulses are categorized by their op_code and by whether or not they
        are a copy created by enforce_single_element. The mirror_pattern
        decides which pulses within a category get mirrored. The mirroring
        is performed by multiplying all pulse parameters that contain
        'amplitude' in their name by -1 (and adding a mirror_correction if
        it is provided), see Pulse.mirror_amplitudes().

        mirror_pattern:
        - 'none'/'all': no/all pulses are mirrored
        - 'odd'/'even': the i-th occurrence of a pulse from the category is
          mirrored if i is odd (1, 3, ...) / even (2, 4, ...). Note that i is
          meant as a natural number (i.e., 1-indexed and not 0-indexed).
        - a list of bools (or of anything that can be interpreted as a bool).
          In this case, the j-th element of the list indicates whether the
          j-th occurrence of a pulse from the category is mirrored. If there
          are more occurrences than elements in the list, the list is
          repeated periodically.

        mirror_correction:
        None (no corrections) or a dict, where each key is a pulse
        parameter name and the corresponding value specifies an additive
        constant to be added after mirroring of this parameter. For parameters
        not found in the dict, no correction is applied.
        """
        op_counts = {}
        for p in self.resolved_pulses:
            pulse_category = (p.op_code, getattr(p, "is_ese_copy", False))
            if pulse_category not in op_counts:
                op_counts[pulse_category] = 0
            op_counts[pulse_category] += 1
            p_obj, pattern = p.pulse_obj.get_mirror_pulse_obj_and_pattern()
            # interpret string pattern ('none'/'all'/'odd'/'even')
            if pattern is None or pattern == 'none':
                continue  # do not mirror
            for pa1, pa2 in [('all', [1]), ('even', [0, 1]), ('odd', [1, 0])]:
                if pattern == pa1:
                    pattern = pa2
            # periodically extend pattern if needed
            pattern = deepcopy(pattern)
            while len(pattern) < op_counts[pulse_category]:
                pattern += pattern
            # check whether the pulse should be mirrored
            if not pattern[op_counts[pulse_category] - 1]:
                continue  # do not mirror
            # mirror all parameters that have 'amplitude' in their name
            # (and apply mirror correction if applicable)
            p_obj.mirror_amplitudes()

    def resolve_Z_gates(self):
        """
        The phase of a basis rotation is acquired by an basis pulse, if the
        middle of the basis rotation pulse happens before the middle of the
        basis pulse. Using that self.resolved_pulses was sorted by
        self.resolve_timing() the acquired phases can be calculated.
        """

        basis_phases = {}

        for pulse in self.resolved_pulses:
            # The following if statement allows pulse objects to specify a
            # basis_rotation different from the one in the instrument settings.
            # Needed, e.g., for arbitrary-phase CZ gates, since, when resolved,
            # the pulse objects updates some of its attributes based on the
            # conditional phase, which may include the basis rotation. In
            # that case, the correct value is needed here.
            if getattr(pulse.pulse_obj, 'basis_rotation', None) is not None:
                pulse.basis_rotation = pulse.pulse_obj.basis_rotation

            for basis, rotation in pulse.basis_rotation.items():
                basis_phases[basis] = basis_phases.get(basis, 0) + rotation

            if pulse.basis is not None:
                # total_phase = original_phase - basis_rotation
                # basis_rotation is defined as the (^ right-handed) rotation
                # angle of the quantum state with respect to subsequent gates
                # (such that e.g. a virtual Z45 gate means basis_rotation=45)
                # meaning that here, in the rotating frame of the state, we
                # subtract basis_rotation from the phase of subsequent pulses
                pulse.pulse_obj.phase = pulse.original_phase - \
                                        basis_phases.get(pulse.basis, 0)

            # Avoid creating repetitive waveforms due to small rounding errors
            if hasattr(pulse.pulse_obj, "phase"):
                pulse.pulse_obj.phase = round(
                    round(pulse.pulse_obj.phase,
                          self.PHASE_ROUNDING_DIGITS) % 360.0,
                    self.PHASE_ROUNDING_DIGITS)

    def add_pulse_to_element(self, element, pulse):
        """Adds pulse to self.elements, and updates cached start and end times

        Args:
            element: element to which the pulse should be added
            pulse: pulse object
        In addition to adding pulse to self.elements[element], this method
        caches the corresponding start and end times of the element, for the
        corresponding trigger group. Start/end times are cached for each
        combination (element, group) in self._element_start_end_raw.
        """
        self.elements[element].append(pulse)
        for el_group in self._element_start_end_raw:
            if el_group[0] == element:
                group_chs = self.pulsar.get_trigger_group_channels(el_group[1])
                t_start_raw, t_end = self._element_start_end_raw[el_group]
                for ch in pulse.masked_channels():
                    if ch in group_chs:
                        break
                else:
                    continue
                t_start_raw = min(pulse.algorithm_time(), t_start_raw)
                t_end = max(pulse.algorithm_time() + pulse.length, t_end)
                self._element_start_end_raw[el_group] = (t_start_raw, t_end)

    def element_start_length(self, element, trigger_group, t_start=np.inf):
        """
        Finds and saves the start and length of an element on an AWG
        trigger group in self.element_start_end.
        """
        if element not in self.element_start_end:
            self.element_start_end[element] = {}
        group_chs = self.pulsar.get_trigger_group_channels(trigger_group)
        # find element start, end and length
        t_end = -np.inf

        el_buffer = self.pulsar.min_element_buffer() or 0.
        el_group = (element, trigger_group)
        if el_group not in self._element_start_end_raw:
            t_start_raw = np.inf
            for pulse in self.elements[element]:
                for ch in pulse.masked_channels():
                    if ch in group_chs:
                        break
                else:
                    continue
                t_start_raw = min(pulse.algorithm_time() - el_buffer,
                                  t_start_raw)
                t_end = max(pulse.algorithm_time() + pulse.length + el_buffer,
                            t_end)
                self._element_start_end_raw[el_group] = (t_start_raw, t_end)
        else:
            t_start_raw, t_end = self._element_start_end_raw[el_group]
        t_start = min(t_start, t_start_raw)

        # if element is not on the awg provided, the function
        # shall return None. This is useful for
        # self._combine_elements which in some instances wants to
        # update start and length for an element on all AWGs
        # without taking care of whether the element is actually
        # on that AWG. One could think of splitting these two
        # aspects of the self.element_start_length.
        if t_start == np.inf or t_end == -np.inf:
            log.debug(f'Asked to find start of element {element} on AWG '
                      f'trigger group {trigger_group}, but element not '
                      f'on AWG.')
            return

        # Enforces the latest t_start of the element if the corresponding
        # pulsar parameter is specified, and does nothing otherwise.
        t_start = min(t_start,
                      self.pulsar.max_element_start_time() or np.inf)

        # make sure that element start is a multiple of element
        # start granularity
        # we allow rounding up of the start time by half a sample, otherwise
        # we round the start time down
        awg = self.pulsar.get_awg_from_trigger_group(trigger_group)
        start_gran = self.pulsar.get_element_start_granularity(trigger_group)
        sample_time = 1/self.pulsar.clock(awg=awg)
        if start_gran is not None:
            t_start = math.floor((t_start + 0.5*sample_time) / start_gran) \
                      * start_gran

        # make sure that the element length exceeds min length for the AWG,
        # and is a multiple of sample granularity
        gran = self.pulsar.get('{}_granularity'.format(awg))
        samples = self.time2sample(t_end - t_start, awg=awg)
        min_length_samples = self.time2sample(
            self.pulsar.get('{}_min_length'.format(awg)), awg=awg)
        samples = max(samples, min_length_samples)
        if samples % gran != 0:
            samples += gran - samples % gran

        self.element_start_end[element][trigger_group] = [t_start, samples]

        return [t_start, samples]

    @_with_pulsar_tmp_vals
    def waveforms(self, awgs=None, elements=None, channels=None,
                  codewords=None, trigger_groups=None):
        """
        After all the pulses have been added, the timing resolved and the
        trigger pulses added, the waveforms of the segment can be compiled.
        This method returns a dictionary:
        AWG_wfs =
          = {AWG_name:
                {(position_of_element, element_name):
                    {codeword:
                        {channel_id: channel_waveforms}
                    ...
                    }
                ...
                }
            ...
            }
        """

        # FIXME: this routine is only used for plotting waveforms of a segment.
        #        Consider removing this routine and adding functionality to plot
        #        single segments to the sequence.plot() routine.
        #        This routine might be less maintained than the sequence routine
        #        generate_waveforms_sequences().

        if awgs is None:
            awgs = set(self.pulsar.get_awg_from_trigger_group(group)
                       for group in self.elements_on_awg)
        if channels is None:
            channels = set(self.pulsar.channels)
        if elements is None:
            elements = set(self.elements)
        if trigger_groups is None:
            trigger_groups = set(self.elements_on_awg)

        awg_wfs = {}
        for group in trigger_groups:
            # only procede for AWGs with waveforms
            if group not in self.elements_on_awg:
                continue
            awg = self.pulsar.get_awg_from_trigger_group(group)
            if awg not in awgs:
                continue
            if awg not in awg_wfs:
                awg_wfs[awg] = {}
            channel_set = set(self.pulsar.get_trigger_group_channels(
                group)) & set(channels)
            if not channel_set:
                continue
            for i, element in enumerate(self.elements_on_awg[group]):
                if element not in elements:
                    continue
                if (i, element) not in awg_wfs[awg]:
                    awg_wfs[awg][(i, element)] = {}

                tvals = self.tvals(channel_set, element)
                wfs = {}
                element_start_time = self.get_element_start(element, group)
                for pulse in self.elements[element]:
                    # checks whether pulse is played on AWG
                    pulse_channels = pulse.masked_channels() & channel_set
                    if not pulse_channels:
                        continue
                    if codewords is not None and \
                            pulse.codeword not in codewords:
                        continue

                    # fills wfs with zeros for used channels
                    if pulse.codeword not in wfs:
                        wfs[pulse.codeword] = {}
                        for channel in pulse_channels:
                            wfs[pulse.codeword][channel] = np.zeros(
                                len(tvals[channel]))
                    else:
                        for channel in pulse_channels:
                            if channel not in wfs[pulse.codeword]:
                                wfs[pulse.codeword][channel] = np.zeros(
                                    len(tvals[channel]))

                    # calculate the pulse tvals
                    chan_tvals = {}
                    pulse_start = self.time2sample(
                        pulse.element_time(element_start_time), awg=awg)
                    pulse_end = self.time2sample(
                        pulse.element_time(element_start_time) + pulse.length,
                        awg=awg)
                    for channel in pulse_channels:
                        if self.fast_mode:
                            t_vals_ch = tvals[channel]
                        else:
                            t_vals_ch = tvals[channel].copy()
                        chan_tvals[channel] = t_vals_ch[pulse_start:pulse_end]

                    # calculate pulse waveforms
                    pulse_wfs = pulse.waveforms(chan_tvals)

                    # insert the waveforms at the correct position in wfs
                    # offset by the pulsar software channel delay
                    el_buffer = self.pulsar.min_element_buffer() or 0.
                    for channel in pulse_channels:
                        extra_delay = self.pulsar.get(channel + '_delay') or 0.
                        # extra 1e-12 to deal with numerical precision
                        if abs(extra_delay) > el_buffer + 1e-12:
                            raise Exception('Delay on channel {} exceeds the '
                                    'available pulse buffer!'.format(channel))
                        extra_delay_samples = self.time2sample(
                            extra_delay, awg=awg)
                        ps_mod = pulse_start + extra_delay_samples
                        pe_mod = pulse_end + extra_delay_samples
                        wfs[pulse.codeword][channel][ps_mod:pe_mod] += \
                            pulse_wfs[channel]

                # for codewords: add the pulses that do not have a codeword to
                # all codewords
                if 'no_codeword' in wfs:
                    for codeword in wfs:
                        if codeword != 'no_codeword':
                            for channel in wfs['no_codeword']:
                                if channel in wfs[codeword]:
                                    wfs[codeword][channel] += wfs[
                                        'no_codeword'][channel]
                                else:
                                    wfs[codeword][channel] = wfs[
                                        'no_codeword'][channel]


                # do predistortion
                for codeword in wfs:
                    for c in wfs[codeword]:
                        if not self.pulsar.get(
                                '{}_type'.format(c)) == 'analog':
                            continue
                        if not self.pulsar.get(
                                '{}_distortion'.format(c)) == 'precalculate':
                            continue

                        wf = wfs[codeword][c]

                        distortion_dict = self.distortion_dicts.get(c, None)
                        if distortion_dict is None:
                            distortion_dict = self.pulsar.get(
                                '{}_distortion_dict'.format(c))
                        else:
                            distortion_dict = \
                                flux_dist.process_filter_coeffs_dict(
                                    distortion_dict,
                                    default_dt=1 / self.pulsar.clock(
                                        channel=c))

                        fir_kernels = distortion_dict.get('FIR', None)
                        if fir_kernels is not None:
                            if hasattr(fir_kernels, '__iter__') and not \
                            hasattr(fir_kernels[0], '__iter__'): # 1 kernel
                                wf = flux_dist.filter_fir(fir_kernels, wf)
                            else:
                                for kernel in fir_kernels:
                                    wf = flux_dist.filter_fir(kernel, wf)
                        iir_filters = distortion_dict.get('IIR', None)
                        if iir_filters is not None:
                            wf = flux_dist.filter_iir(iir_filters[0],
                                                      iir_filters[1], wf)
                        wfs[codeword][c] = wf

                # truncation and normalization
                for codeword in wfs:
                    for c in wfs[codeword]:
                        # truncate all values that are out of bounds and
                        # normalize the waveforms
                        amp = self.pulsar.get('{}_amp'.format(c))
                        self._channel_amps[c] = amp
                        if self.pulsar.get('{}_type'.format(c)) == 'analog':
                            if np.max(wfs[codeword][c], initial=0) > amp:
                                logging.warning(
                                    'Clipping waveform {}: {} > {}'.format(
                                        c, np.max(wfs[codeword][c]), amp))
                            if np.min(wfs[codeword][c], initial=0) < -amp:
                                logging.warning(
                                    'Clipping waveform {}: {} < {}'.format(
                                        c, np.min(wfs[codeword][c]), -amp))
                            np.clip(
                                wfs[codeword][c],
                                -amp,
                                amp,
                                out=wfs[codeword][c])
                            # normalize wfs
                            wfs[codeword][c] = wfs[codeword][c] / amp
                        # marker channels have to be 1 or 0
                        elif self.pulsar.get('{}_type'.format(c)) == 'marker':
                            wfs[codeword][c] = (wfs[codeword][c] > 0)\
                                .astype(int)

                # save the waveforms in the dictionary
                for codeword in wfs:
                    if codeword not in awg_wfs[awg][(i, element)]:
                        awg_wfs[awg][(i, element)][codeword] = {}
                    for channel in wfs[codeword]:
                        awg_wfs[awg][(i, element)][codeword][self.pulsar.get(
                            '{}_id'.format(channel))] = (
                                wfs[codeword][channel])

        return awg_wfs

    def get_element_codewords(self, element, awg=None, trigger_group=None):
        """
        Return all codewords for pulses in an element. Use awg or
        group to filter for specific awg or group.

        Args:
            element (str): name of element to get codewords for
            awg (str): name of awg which to consider for finding codewords
            trigger_group (str): name of trigger group which to consider
                for finding codewords

        Returns:
            set of codewords
        """
        codewords = set()
        if awg is not None:
            awg_channels = set(self.pulsar.find_awg_channels(awg))
        if trigger_group is not None:
            group_channels = set(self.pulsar.get_trigger_group_channels(
                trigger_group))

        for pulse in self.elements[element]:
            channels = set(pulse.masked_channels())
            if awg is not None:
                channels = channels & awg_channels
            if trigger_group is not None:
                channels = channels & group_channels
            if len(channels) == 0:
                continue
            codewords.add(pulse.codeword)
        return codewords

    def get_element_channels(self, element, awg=None, trigger_group=None):
        """
        Return all channels an element is distributed over. Use awg or
        group to filter for specific awg or group.

        Args:
        element (str): name of element to get channels for
        awg (str): name of awg which to consider for finding channels
        trigger_group (str): name of trigger group which to consider
            for finding channels

        Returns:
            set of channels
        """
        channels = set([ch for pulse in self.elements[element] for ch in
                        pulse.masked_channels()])
        if awg is not None:
            channels &= set(self.pulsar.find_awg_channels(awg))
        if trigger_group is not None:
            channels &= set(self.pulsar.get_trigger_group_channels(
                trigger_group))
        return channels

    @_with_pulsar_tmp_vals
    def calculate_hash(self, elname, codeword, channel):
        if not self.pulsar.reuse_waveforms():
            # these hash entries avoid that the waveform is reused on another
            # channel or in another element/codeword
            hashlist = [self.name, elname, codeword, channel]
            if not self.pulsar.use_sequence_cache():
                return tuple(hashlist)
            # when sequence cache is used, we still need to add the other
            # hashables to allow pulsar to detect when a re-upload is required
        else:
            hashlist = []

        group = self.pulsar.get_trigger_group(channel)
        tstart, length = self.element_start_end[elname][group]
        hashlist.append(length)  # element length in samples
        if self.pulsar.get(f'{channel}_type') == 'analog' and \
                self.pulsar.get(f'{channel}_distortion') == 'precalculate':
            hashlist.append(repr(self.pulsar.get(
                f'{channel}_distortion_dict')))
        else:
            hashlist.append(self.pulsar.clock(channel=channel))  # clock rate
            for par in ['type', 'amp', 'internal_modulation']:
                chpar = f'{channel}_{par}'
                if chpar in self.pulsar.parameters:
                    hashlist.append(self.pulsar.get(chpar))
                else:
                    hashlist.append(False)
        hashlist.append(self.pulsar.get(f'{channel}_delay'))
        if self.pulsar.get(f'{channel}_type') == 'analog' and \
                self.pulsar.get(f'{channel}_charge_buildup_compensation'):
            for par in ['compensation_pulse_delay',
                        'compensation_pulse_gaussian_filter_sigma',
                        'compensation_pulse_scale']:
                hashlist.append(self.pulsar.get(f'{channel}_{par}'))

        for pulse in self.elements[elname]:
            if pulse.codeword in {'no_codeword', codeword}:
                hashlist += self.hashables(pulse, tstart, channel)
        return tuple(hashlist)

    @staticmethod
    def hashables(pulse, tstart, channel):
        """
        Wrapper for Pulse.hashables making sure to deal correctly with
        crosstalk cancellation channels.

        The hashables of a cancellation pulse has to include the hashables
        of all pulses that it cancels. This is needed to ensure that the
        cancellation pulse gets re-uploaded when any of the cancelled pulses
        changes. In addition it has to include the parameters of
        cancellation calibration, i.e., the relevant entries of the
        crosstalk cancellation matrix and of the shift matrix.

        :param pulse: a Pulse object
        :param tstart: (float) start time of the element
        :param channel: (str) channel name
        """
        if channel in pulse.crosstalk_cancellation_channels:
            hashables = []
            idx_c = pulse.crosstalk_cancellation_channels.index(channel)
            for c in pulse.channels:
                if c in pulse.crosstalk_cancellation_channels:
                    idx_c2 = pulse.crosstalk_cancellation_channels.index(c)
                    factor = pulse.crosstalk_cancellation_mtx[idx_c, idx_c2]
                    shift = pulse.crosstalk_cancellation_shift_mtx[
                        idx_c, idx_c2] \
                        if pulse.crosstalk_cancellation_shift_mtx is not \
                           None else 0
                    if factor != 0:
                        hashables += pulse.hashables(tstart, c)
                        hashables += [factor, shift]
            return hashables
        else:
            return pulse.hashables(tstart, channel)

    def tvals(self, channel_list, element, groups=None):
        """
        Returns a dictionary with channel names of the used channels in the
        element as keys and the tvals array for the channel as values.

        Note that groups is modified by this method. This allows to cache
        groups for each channel, to limit calls to pulsar.get_trigger_group
        """

        tvals = {}
        groups = groups or {}
        for channel in channel_list:
            if channel not in groups:
                groups[channel] = self.pulsar.get_trigger_group(channel)
            group = groups[channel]
            samples = self.element_start_end[element][group][1]
            tvals[channel] = np.arange(samples) / self.pulsar.clock(
                channel=channel) + self.get_element_start(element, group)

        return tvals

    def time2sample(self, t, **kw):
        """
        Converts time to a number of samples for a channel or AWG.
        """
        # FIXME: check whether this should be cached directly in segment
        return int(np.floor(t * self.pulsar.clock(**kw) + 0.5))

    def sample2time(self, samples, **kw):
        """
        Converts nubmer of samples to time for a channel or AWG.
        """
        return samples / self.pulsar.clock(**kw)

    def get_waveforms_export(self, instruments=None, channels=None,
                             trigger_groups=None, normalized_amplitudes=False):
        codeword_warning_issued = False
        self.resolve_segment()
        wfs = self.waveforms(awgs=instruments, channels=None,
                             trigger_groups=trigger_groups)
        t_start = min([t[0]
                       for t_ins in self.element_start_end.values()
                       for t in t_ins.values()])
        wfs_export = dict()
        sorted_keys = sorted(wfs.keys()) if instruments is None \
            else [i for i in instruments if i in wfs]
        for i, instr in enumerate(sorted_keys):
            wfs_export[instr] = dict()
            for elem_name, v in wfs[instr].items():
                for k, wf_per_ch in v.items():
                    if k != "no_codeword":
                        if not codeword_warning_issued:
                            log.warning(
                                'Codewords are currently not supported in '
                                'waveform export. Pulses with codewords will '
                                'be ignored.')
                            codeword_warning_issued = True
                        continue
                    sorted_chans = sorted(wf_per_ch.keys())
                    for n_wf, ch in enumerate(sorted_chans):
                        if channels is not None and \
                                ch not in channels.get(instr, []):
                            continue
                        ins_ch = f"{instr}_{ch}"
                        t_clk = 1 / self.pulsar.clock(channel=ins_ch)
                        wfs_export[instr].setdefault(ch, np.array(
                            [[t_start], [0]]))
                        w_exp = wfs_export[instr][ch]
                        wf = wf_per_ch[ch]
                        if not normalized_amplitudes:
                            wf = wf * self._channel_amps[f'{instr}_{ch}']
                        tvals = self.tvals([ins_ch], elem_name[1])[ins_ch]
                        w_add = np.arange(w_exp[0][-1], tvals[0], t_clk)
                        if len(w_add):
                            w_exp = np.append(
                                w_exp, np.array(
                                    [w_add[1:], np.zeros_like(w_add[1:])]),
                                axis=1)
                        wfs_export[instr][ch] = np.append(
                            w_exp, [tvals, wf], axis=1)
        return wfs_export

    def export_waveforms(self, filename, **kw):
        np.save(filename, self.get_waveforms_export(**kw))

    def plot(self, instruments=None, channels=None, legend=True,
             delays=None, savefig=False, prop_cycle=None, frameon=True,
             channel_map=None, plot_kwargs=None, axes=None, demodulate=False,
             show_and_close=True, col_ind=0, normalized_amplitudes=True,
             save_kwargs=None, figtitle_kwargs=None, sharex=True,
             trigger_groups=None):
        """
        Plots a segment. Can only be done if the segment can be resolved.
        :param instruments (list): instruments for which pulses have to be
            plotted. Defaults to all.
        :param channels (list):  channels to plot. defaults to all.
        :param delays (dict): keys are instruments, values are additional
            delays. If passed, the delay is substracted to the time values of
            this instrument, such that the pulses are plotted at timing when
            they physically occur. A key 'default' can be used to specify a
            delay for all instruments that are not explicitly given as keys.
        :param savefig: save the plot
        :param channel_map (dict): indicates which instrument channels
            correspond to which qubits. Keys = qb names, values = list of
            channels. eg. dict(qb2=['AWG8_ch3', "UHF_ch1"]). If provided,
            will plot each qubit on individual subplots.
        :param prop_cycle (dict):
        :param frameon (dict, bool):
        :param axes (array or axis): 2D array of matplotlib axes. if single
            axes, will be converted internally to array.
        :param demodulate (bool): plot only envelope of pulses by temporarily
            setting modulation and phase to 0. Need to recompile the sequence
        :param show_and_close: (bool) show and close the plot (default: True)
        :param col_ind: (int) when passed together with axes, this specifies
            in which column of subfigures the plots should be added
            (default: 0)
        :param normalized_amplitudes: (bool) whether amplitudes
            should be normalized to the voltage range of the channel
            (default: True)
        :param save_kwargs (dict): save kwargs passed on to fig.savefig if
        "savefig" is True.
        :param figtitle_kwargs (dict): figure.title kwargs passed on to fig.suptitle if
        not None
        :param sharex (bool): whether the xaxis is shared between the subfigures or not
        :param trigger_groups (list): trigger groups for which pulses have to be
            plotted. Defaults to all.
        :return: The figure and axes objects if show_and_close is False,
            otherwise no return value.
        """

        import matplotlib.pyplot as plt
        if delays is None:
            delays = dict()
        if plot_kwargs is None:
            plot_kwargs = dict()
            plot_kwargs['linewidth'] = 0.7
        try:
            # resolve segment and populate elements/waveforms
            self.resolve_segment(allow_overlap=True)
            if demodulate:
                for el in self.elements.values():
                    for pulse in el:
                        if hasattr(pulse, "mod_frequency"):
                            pulse.mod_frequency = 0
                        if hasattr(pulse, "phase"):
                            pulse.phase = 0
            wfs = self.waveforms(awgs=instruments, channels=None,
                                 trigger_groups=trigger_groups)
            n_instruments = len(wfs) if channel_map is None else \
                len(channel_map)
            if axes is not None:
                if np.ndim(axes) == 0:
                    axes = np.array([[axes]])
                fig = axes[0,0].get_figure()
                ax = axes
            else:
                fig, ax = plt.subplots(nrows=n_instruments, sharex=sharex,
                                       squeeze=False,
                                       figsize=(16, n_instruments * 3))
            if prop_cycle is not None:
                for a in ax[:,col_ind]:
                    a.set_prop_cycle(**prop_cycle)
            sorted_keys = sorted(wfs.keys()) if instruments is None \
                else [i for i in instruments if i in wfs]
            for i, instr in enumerate(sorted_keys):
                if instr not in delays and 'default' in delays:
                    delays[instr] = delays['default']
                # plotting
                for elem_name, v in wfs[instr].items():
                    for k, wf_per_ch in v.items():
                        if k == "no_codeword":
                            k = ""
                        sorted_chans = sorted(wf_per_ch.keys())
                        for n_wf, ch in enumerate(sorted_chans):
                            wf = wf_per_ch[ch]
                            if not normalized_amplitudes:
                                wf = wf * self._channel_amps[f'{instr}_{ch}']
                            if channels is None or \
                                    ch in channels.get(instr, []):
                                tvals = \
                                self.tvals([f"{instr}_{ch}"], elem_name[1])[
                                    f"{instr}_{ch}"] - delays.get(instr, 0)
                                if channel_map is None:
                                    # plot per device
                                    ax[i, col_ind].set_title(instr)
                                    artists = ax[i, col_ind].plot(
                                        tvals * 1e6, wf,
                                        label=f"{elem_name[1]}_{k}_{ch}",
                                        **plot_kwargs)
                                    for artist in artists:
                                        artist.pycqed_metadata = {'channel': ch,
                                                                  'element_name': elem_name[1],
                                                                  'codeword': k,
                                                                  'instrument': instr,
                                                                  }
                                else:
                                    # plot on each qubit subplot which includes
                                    # this channel in the channel map
                                    match = {i: qb_name
                                             for i, (qb_name, qb_chs) in
                                             enumerate(channel_map.items())
                                             if f"{instr}_{ch}" in qb_chs}
                                    for qbi, qb_name in match.items():
                                        ax[qbi, col_ind].set_title(qb_name)
                                        artists = ax[qbi, col_ind].plot(
                                            tvals * 1e6, wf,
                                            label=f"{elem_name[1]}"
                                                  f"_{k}_{instr}_{ch}",
                                            **plot_kwargs)
                                        for artist in artists:
                                            artist.pycqed_metadata = {'channel': ch,
                                                                      'element_name': elem_name[1],
                                                                      'codeword': k,
                                                                      'instrument': instr,
                                                                      }
                                        if demodulate: # filling
                                            ax[qbi, col_ind].fill_between(
                                                tvals * 1e6, wf,
                                                label=f"{elem_name[1]}_"
                                                      f"{k}_{instr}_{ch}",
                                                alpha=0.05,
                                                **plot_kwargs)

            # formatting
            for a in ax[:, col_ind]:
                if isinstance(frameon, bool):
                    frameon = {k: frameon for k in ['top', 'bottom',
                                                    "right", "left"]}
                a.spines["top"].set_visible(frameon.get("top", True))
                a.spines["right"].set_visible(frameon.get("right", True))
                a.spines["bottom"].set_visible(frameon.get("bottom", True))
                a.spines["left"].set_visible(frameon.get("left", True))
                if legend:
                    a.legend(loc=[1.02, 0], prop={'size': 8}, frameon=False)
                if normalized_amplitudes:
                    a.set_ylabel('Amplitude (norm.)')
                else:
                    a.set_ylabel('Voltage (V)')
            ax[-1, col_ind].set_xlabel('time ($\mu$s)')
            if figtitle_kwargs:
                fig.suptitle(f'{self.name}', **figtitle_kwargs)
            else:
                fig.suptitle(f'{self.name}', y=1.01)
            try:
                fig.align_ylabels()
            except AttributeError as e:
                # sometimes this fails e.g. if the axes on which the pulses
                # are plotted is an inset ax.
                log.warning('Could not align y labels in Figure.')
            plt.tight_layout()
            if savefig:
                if save_kwargs is None:
                    save_kwargs = dict(fname=f'{self.name}.png',
                                      bbox_inches="tight")
                fig.savefig(**save_kwargs)
            if show_and_close:
                plt.show()
                plt.close(fig)
                return
            else:
                return fig, ax
        except Exception as e:
            log.error(f"Could not plot: {self.name}")
            raise e

    def __repr__(self):
        string_repr = f"---- {self.name} ----\n"

        for i, p in enumerate(self.unresolved_pulses):
            string_repr += f"{i}: " + repr(p) + "\n"
        return string_repr

    def export_tikz(self, qb_names, tscale=1e-6, include_readout=True):
        def extract_qb(op_code, raise_error=True):
            pattern = r'qb(\d+)$'
            match = re.search(pattern, op_code)

            if match:
                return match.group(0)
            elif raise_error:
                raise ValueError(
                    "Invalid op_code format: should end with 'qb' followed by an integer.")
            else:
                return None
        last_z = [(-np.inf, 0)] * len(qb_names)

        output = ''
        z_output = ''
        start_output = '\\documentclass{standalone}\n\\usepackage{tikz}\n\\begin{document}\n\\scalebox{2}{'
        start_output += '\\begin{tikzpicture}[x=10cm,y=2cm]\n'
        start_output += '\\tikzstyle{CZdot} = [shape=circle, thick,draw,inner sep=0,minimum size=.5mm, fill=black]\n'
        start_output += '\\tikzstyle{gate} = [draw,fill=white,minimum width=1cm, rotate=90]\n'
        start_output += '\\tikzstyle{zgate} = [rotate=0]\n'
        tmin = np.inf
        tmax = -np.inf
        num_single_qb = 0
        num_two_qb = 0
        num_virtual = 0
        self.resolve_segment()
        for p in self.resolved_pulses:
            if p.op_code != '':
                l = p.pulse_obj.length
                t = p.pulse_obj._t0 + l / 2
                tmin = min(tmin, p.pulse_obj._t0)
                tmax = max(tmax, p.pulse_obj._t0 + p.pulse_obj.length)
                qb = qb_names.index(extract_qb(p.op_code))
                op_code = p.op_code[:-len(extract_qb(p.op_code))]
                qbt = 0
                if (qbt_name := extract_qb(op_code.strip(), raise_error=False)) is not None:
                    qbt = qb_names.index(qbt_name)
                    op_code = op_code.strip()[:-len(qbt_name)]

                if p.op_code.startswith('RO'):
                    if include_readout:
                        qb = qb_names.index(extract_qb(p.op_code))
                        output += f'\\draw[fill=white] ({t / tscale:.4f},-{qb} + 0.1) rectangle ++({l / tscale:.4f}, -0.2) node[midway] {{RO}};\n'
                    continue

                if p.op_code.startswith('PFM'):
                    qb = qb_names.index(extract_qb(p.op_code))
                    output += f'\\draw({t / tscale:.4f},-{qb}) node[ gate, minimum height={l / tscale * 10:.4f}mm] {{ \\tiny {op_code.replace("_", "")}}};\n'
                    continue

                if op_code[-1:] == 's':
                    op_code = op_code[:-1]
                if op_code[:2] == 'CZ' or op_code[:4] == 'upCZ':
                    num_two_qb += 1
                    pulse_name = op_code.rstrip('0123456789.')
                    if len(val := op_code[len(pulse_name):]):
                        # FIXME this - sign comes from the convention that
                        #  CZ = diag(1,1,1,e^-i*phi). We should at some point
                        #  verify that all code respects a single convention.
                        val = -float(val)
                        gate_formatted = f'{gate_type}{(factor * val):.1f}'.replace(
                            '.0', '')
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[CZdot] {{}} -- ({t / tscale:.4f},-{qbt}) node[gate, minimum height={l / tscale * 100:.4f}mm] {{\\tiny {gate_formatted}}};\n'
                    else:
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[CZdot] {{}} -- ({t / tscale:.4f},-{qbt}) node[CZdot] {{}};\n'
                elif op_code[0] == 'I':
                    continue
                else:
                    if op_code[0] == 'm':
                        factor = -1
                        op_code = op_code[1:]
                    else:
                        factor = 1
                    gate_type = 'R' + op_code[:1]
                    val = float(op_code[1:])
                    if val == 180:
                        gate_formatted = op_code[:1]
                    else:
                        gate_formatted = f'{gate_type}{(factor * val):.1f}'.replace(
                            '.0', '')
                    if l == 0:
                        if t - last_z[qb][0] > 1e-9:
                            z_height = 0 if (
                                        t - last_z[qb][0] > 100e-9 or last_z[qb][
                                    1] >= 3) else last_z[qb][1] + 1
                            z_output += f'\\draw[dashed,thick,shift={{(0,.03)}}] ({t / tscale:.4f},-{qb})--++(0,{0.3 + z_height * 0.1});\n'
                        else:
                            z_height = last_z[qb][1] + 1
                        z_output += f'\\draw({t / tscale:.4f},-{qb})  node[zgate,shift={{({(0, .35 + z_height * .1)})}}] {{\\tiny {gate_formatted}}};\n'
                        last_z[qb] = (t, z_height)
                        num_virtual += 1
                    else:
                        output += f'\\draw({t / tscale:.4f},-{qb})  node[gate, minimum height={l / tscale * 100:.4f}mm] {{\\tiny {gate_formatted}}};\n'
                        num_single_qb += 1
        qb_output = ''
        for qb, qb_name in enumerate(qb_names):
            qb_output += f'\draw ({tmin / tscale:.4f},-{qb}) node[left] {{{qb_name}}} -- ({tmax / tscale:.4f},-{qb});\n'
        output = start_output + qb_output + output + z_output
        axis_ycoord = -len(qb_names) + .4
        output += f'\\foreach\\x in {{{tmin / tscale},{tmin / tscale + .2},...,{tmax / tscale}}} \\pgfmathprintnumberto[fixed]{{\\x}}{{\\tmp}} \draw (\\x,{axis_ycoord})--++(0,-.1) node[below] {{\\tmp}} ;\n'
        output += f'\\draw[->] ({tmin / tscale},{axis_ycoord}) -- ({tmax / tscale},{axis_ycoord}) node[right] {{$t/\\mathrm{{\\mu s}}$}};\n'
        output += '\\end{tikzpicture}}\end{document}'
        output += f'\n% {num_single_qb} single-qubit gates, {num_two_qb} two-qubit gates, {num_virtual} virtual gates'
        return output

    def export_stim(self, qubit_coords=None,
                    transpiling_dict=None, resolve_segment=True, tol=1e-9):
        """
        Export the segment to stim format.

        Parameters:
        - qubit_coords (dict): dict of coordinites for each qubit,
         Useful to later display the stim circuit on a grid. Format like:
         {"qb1": (x, y), "qb2": (x2, y2), ...}

        - transpiling_dict (dict): Dictionary for transpiling operations.
            If None, use the default dictionary defined below
            (default_pycqed_to_stim_transpiling_dict)
        - resolve_segment (bool): Whether to resolve the segment.
        - tol (float): Tolerance for "same timing operation". only used to detect
         where to put TICKS in stim circuit, has no influence on the
         outcome of the circuit since there is no notion of "time" in stim

        Returns:
        - str: Stim-formatted string.
        """
        # FIXME: as soon as this dict is needed at multiple places, move it
        #  to a common place from where it can be imported.
        default_pycqed_to_stim_transpiling_dict = {
            'X180': ['X'],
            'X90': ['SQRT_X'],
            'mX90': ['SQRT_X_DAG'],
            'Y90': ['SQRT_Y'],
            'Y180': ['Y'],
            'mY90': ['SQRT_Y_DAG'],
            'CZ': ['CZ'],
            'Z180': ['Z'],
            'PFM_ef': [],
            'RO': ['M'],
        }

        def strip_qb_prefix(input_string):
            """
            Strip 'qb' prefix from the input string.
            """
            return re.sub(r'qb(\d+)', r'\1', input_string)

        def transpile_operation(op, transpiling_dict):
            output = ""

            if op.split(' ')[0] not in transpiling_dict:
                log.warning(f'Operation {op.split(" ")[0]} not in known '
                            f'operations: {transpiling_dict.keys()}')
            for stim_op in transpiling_dict.get(op.split(' ')[0], []):
                output += " ".join([stim_op] + strip_qb_prefix(op).split(' ')[1:]) + '\n'
            return output

        if resolve_segment:
            self.resolve_segment()
        if transpiling_dict is None:
            transpiling_dict = default_pycqed_to_stim_transpiling_dict
        # sort pulses by start time
        pulses = sorted(self.resolved_pulses, key=lambda p: p.pulse_obj._t0)
        ops = [(p.op_code, p.pulse_obj._t0) for p in pulses if
               hasattr(p, 'op_code') and p.op_code != '']

        tprev = np.min([op[1] for op in ops]) # earliest time
        circuit_str = f"# {self.name}\n"

        if qubit_coords is not None:
            for key, coords in qubit_coords.items():
                circuit_str += f"QUBIT_COORDS({', '.join(map(str, coords))}) {key[2:]}\n"

        for op, t in ops:
            if np.abs(t - tprev) > tol:
                circuit_str += 'TICK\n'
                tprev = t
            circuit_str += transpile_operation(op, transpiling_dict)

        return circuit_str

    def get_stim_circuit(self, qubit_coords=None,
                         transpiling_dict=None,
                         resolve_segment=True):
        """
        Get a stim Circuit from the current segment.

        See documentation of export_stim for more details.

        Returns:
        - Union[stim.Circuit, str]: Stim Circuit or string output of export_stim.
        """
        stim_circuit_str = self.export_stim(qubit_coords, transpiling_dict,
                                            resolve_segment=resolve_segment)

        try:
            import stim
        except ImportError:
            log.error("Stim module could not be found. Cannot return the circuit. "
                      "Please install the 'stim' module. Will return the string"
                      "of the stim circuit.")
            return stim_circuit_str

        return stim.Circuit(stim_circuit_str)


    def rename(self, new_name):
        """
        Renames a segment with the given new name. Hunts down element names in
        unresolved pulses and acquisition elements that might have made use of
        the old segment_name and renames them too.
        Note: this function relies on the convention that the element_name ends with
        "_segmentname".
        Args:
            new_name:

        Returns:

        """
        old_name = self.name

        # rename element names in unresolved_pulses and resolved_pulses making
        # use of the old name
        for p in self.unresolved_pulses + self.resolved_pulses:
            if hasattr(p.pulse_obj, "element_name") \
                    and p.pulse_obj.element_name.endswith(f"_{old_name}"):
                p.pulse_obj.element_name = \
                    p.pulse_obj.element_name[:-(len(old_name) + 1)] + '_' \
                    + new_name

        # rebuild acquisition elements that used the old segment name
        new_acq_elements = dict()
        for el, v in self.acquisition_elements.items():
            if el.endswith(f"_{old_name}"):
                new_acq_elements[el[:-(len(old_name) + 1)] + '_'
                                 + new_name] = v
            else:
                new_acq_elements[el] = v
                log.warning(f'Acquisition element name: {el} not ending'
                            f' with "_segmentname": {old_name}. Keeping '
                            f'current element name when renaming '
                            f'the segment.')
        self.acquisition_elements = new_acq_elements
        # enforce that start and end times get recalculated using the new
        # element names
        self.element_start_end = {}

        # rename segment name
        self.name = new_name

        # rename timer
        self.timer.name = new_name

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_seg = cls.__new__(cls)
        memo[id(self)] = new_seg
        for k, v in self.__dict__.items():
            if k == "pulsar": # the reference to pulsar cannot be deepcopied
                setattr(new_seg, k, v)
            else:
                setattr(new_seg, k, deepcopy(v, memo))
        return new_seg


class UnresolvedPulse:
    """
    pulse_pars: dictionary containing pulse parameters
    ref_pulse: 'segment_start', 'init_start', 'previous_pulse', pulse.name,
        or a list of multiple pulse.name.
        If the beginning of a block (virtual pulse whose name ends with
        "-|-start") is referenced to init_start, this block will be considered
        as an initialization block and will be placed before pulses that
        reference segment_start
    ref_point: 'start', 'end', 'middle', reference point of the reference pulse
    ref_point_new: 'start', 'end', 'middle', reference point of the new pulse
    ref_function: 'max', 'min', 'mean', specifies how timing is chosen if
        multiple pulse names are listed in ref_pulse (default: 'max')
    """

    def __init__(self, pulse_pars):
        self.ref_pulse = pulse_pars.get('ref_pulse', 'previous_pulse')
        alignments = {'start': 0, 'middle': 0.5, 'center': 0.5, 'end': 1}
        if pulse_pars.get('ref_point', 'end') == 'end':
            self.ref_point = 1
        elif pulse_pars.get('ref_point', 'end') == 'middle':
            self.ref_point = 0.5
        elif pulse_pars.get('ref_point', 'end') == 'start':
            self.ref_point = 0
        else:
            raise ValueError('Passed invalid value for ref_point. Allowed '
                'values are: start, end, middle. Default value: end')

        if pulse_pars.get('ref_point_new', 'start') == 'start':
            self.ref_point_new = 0
        elif pulse_pars.get('ref_point_new', 'start') == 'middle':
            self.ref_point_new = 0.5
        elif pulse_pars.get('ref_point_new', 'start') == 'end':
            self.ref_point_new = 1
        else:
            raise ValueError('Passed invalid value for ref_point_new. Allowed '
                'values are: start, end, middle. Default value: start')

        self.ref_function = pulse_pars.get('ref_function', 'max')
        self.block_align = pulse_pars.get('block_align', None)
        if self.block_align is not None:
            self.block_align = alignments.get(self.block_align,
                                              self.block_align)
        self.delay = pulse_pars.get('pulse_delay', 0)
        self.original_phase = pulse_pars.get('phase', 0)
        self.basis = pulse_pars.get('basis', None)
        self.operation_type = pulse_pars.get('operation_type', None)
        self.basis_rotation = pulse_pars.pop('basis_rotation', {})
        self.op_code = pulse_pars.get('op_code', '')

        pulse_func = bpl.get_pulse_class(pulse_pars['pulse_type'])
        self.pulse_obj = pulse_func(**pulse_pars)
        # allow a pulse to modify its op_code (e.g., for C-ARB gates)
        self.op_code = getattr(self.pulse_obj, 'op_code', self.op_code)

        if self.pulse_obj.codeword != 'no_codeword' and \
                self.basis_rotation != {}:
            raise Exception(
                'Codeword pulse {} does not support basis_rotation!'.format(
                    self.pulse_obj.name))
        # Segment: length may be a property depending on pulse settings
        # (allows flexibility in the pulse library). This caching makes sure
        # to call it only once during segment resolution (when it should not
        # change anymore), for speed reasons.
        self.cached_length = self.pulse_obj.length

    def __repr__(self):
        string_repr = self.pulse_obj.name
        if self.operation_type != None:
            string_repr += f"\n   operation_type: {self.operation_type}"
        string_repr += f"\n   ref_pulse: {self.ref_pulse}"
        if self.ref_point != 1:
            string_repr += f"\n   ref_point: {self.ref_point}"
        if self.delay != 0:
            string_repr += f"\n   delay: {self.delay}"
        if self.original_phase != 0:
            string_repr += f"\n   phase: {self.original_phase}"
        return string_repr
