# A Segment is the building block of Sequence Class. They are responsible
# for resolving pulse timing, Z gates, generating trigger pulses and adding
# charge compensation
#
# author: Michael Kerschbaum
# created: 4/2019

import numpy as np
import math
import logging
log = logging.getLogger(__name__)
from copy import deepcopy
import pycqed.measurement.waveform_control.pulse as bpl
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulsar as ps
import pycqed.measurement.waveform_control.fluxpulse_predistortion as flux_dist
from collections import OrderedDict as odict


class Segment:
    """
    Consists of a list of UnresolvedPulses, each of which contains information 
    about in which element the pulse is played and when it is played 
    (reference point + delay) as well as an instance of class Pulse.
    """

    def __init__(self, name, pulse_pars_list=[]):
        self.name = name
        self.pulsar = ps.Pulsar.get_instance()
        self.unresolved_pulses = []
        self.resolved_pulses = []
        self.previous_pulse = None
        self.elements = odict()
        self.element_start_end = {}
        self.elements_on_awg = {}
        self.trigger_pars = {
            'pulse_length': 50e-9,
            'amplitude': 0.5,
            'buffer_length_start': 25e-9
        }
        self.trigger_pars['length'] = self.trigger_pars['pulse_length'] + \
                                      self.trigger_pars['buffer_length_start']
        self._pulse_names = set()
        self.acquisition_elements = set()

        for pulse_pars in pulse_pars_list:
            self.add(pulse_pars)

    def add(self, pulse_pars):
        """
        Checks if all entries of the passed pulse_pars dictionary are valid
        and sets default values where necessary. After that an UnresolvedPulse
        is instantiated.
        """
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
        # and that RO pulses have their own elements if no element_name
        # was provided
        i = len(self.acquisition_elements) + 1

        if pars_copy.get('element_name', None) == None:
            if pars_copy.get('operation_type', None) == 'RO':
                pars_copy['element_name'] = \
                    'RO_element_{}_{}'.format(i, self.name)
            else:
                pars_copy['element_name'] = 'default_{}'.format(self.name)
        else:
            pars_copy['element_name'] += '_' + self.name


        # add element to set of acquisition elements
        if pars_copy.get('operation_type', None) == 'RO':
            if pars_copy['element_name'] not in self.acquisition_elements:
                self.acquisition_elements.add(pars_copy['element_name'])


        new_pulse = UnresolvedPulse(pars_copy)

        if new_pulse.ref_pulse == 'previous_pulse':
            if self.previous_pulse != None:
                new_pulse.ref_pulse = self.previous_pulse.pulse_obj.name
            # if the first pulse added to the segment has no ref_pulse
            # it is reference to segment_start by default
            elif self.previous_pulse == None and \
                 len(self.unresolved_pulses) == 0:
                new_pulse.ref_pulse = 'segment_start'
            else:
                raise ValueError('No previous pulse has been added!')

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

    def resolve_segment(self):
        """
        Top layer method of Segment class. After having addded all pulses,
            * pulse elements are updated to enforce single element per segment
                for the that AWGs configured this way.
            * the timing is resolved
            * the virtual Z gates are resolved
            * the trigger pulses are generated
            * the charge compensation pulses are added
        """
        self.enforce_single_element()
        self.resolve_timing()
        self.resolve_Z_gates()
        self.gen_trigger_el()
        self.add_charge_compensation()

    def enforce_single_element(self):
        self.resolved_pulses = deepcopy(self.unresolved_pulses)
        for p in self.resolved_pulses:
            for ch in p.pulse_obj.channels:
                ch_awg = self.pulsar.get(f'{ch}_awg')
                if self.pulsar.get(f'{ch_awg}_enforce_single_element'):
                    p.pulse_obj.element_name = f'default_{self.name}'
                    break

    def resolve_timing(self):
        """
        For each pulse in the resolved_pulses list, this method:
            * updates the _t0 of the pulse by using the timing description of
              the UnresolvedPulse
            * saves the resolved pulse in the elements ordered dictionary by 
              ascending element start time and the pulses in each element by 
              ascending _t0
            * orderes the resolved_pulses list by ascending pulse middle
        """

        self.elements = odict()
        if self.resolved_pulses == []:
            self.enforce_single_element()

        visited_pulses = []
        ref_pulses_dict = {}
        i = 0

        pulses = self.gen_refpoint_dict()

        # add pulses that refer to segment start
        for pulse in pulses['segment_start']:
            if pulse.pulse_obj.name in pulses:
                ref_pulses_dict.update({pulse.pulse_obj.name: pulse})
            t0 = pulse.delay - pulse.ref_point_new * pulse.pulse_obj.length
            pulse.pulse_obj.algorithm_time(t0)
            visited_pulses.append((t0, i, pulse))
            i += 1

        if len(visited_pulses) == 0:
            raise ValueError('No pulse references to the segment start!')

        ref_pulses_dict_all = deepcopy(ref_pulses_dict)
        # add remaining pulses
        while len(ref_pulses_dict) > 0:
            ref_pulses_dict_new = {}
            for name, pulse in ref_pulses_dict.items():
                for p in pulses[name]:
                    if isinstance(p.ref_pulse, list):
                        if p.pulse_obj.name in [vp[2].pulse_obj.name for vp
                                                in visited_pulses]:
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
                                           p.ref_point_new * p.pulse_obj.length +
                                           ref_point * ref_pulses_dict_all[ref_pulse].pulse_obj.length)

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
                        t0 = pulse.pulse_obj.algorithm_time() + p.delay - \
                            p.ref_point_new * p.pulse_obj.length + \
                            p.ref_point * pulse.pulse_obj.length

                    p.pulse_obj.algorithm_time(t0)

                    # add p.name to reference list if it is used as a key
                    # in pulses
                    if p.pulse_obj.name in pulses:
                        ref_pulses_dict_new.update({p.pulse_obj.name: p})

                    visited_pulses.append((t0, i, p))
                    i += 1

            ref_pulses_dict = ref_pulses_dict_new
            ref_pulses_dict_all.update(ref_pulses_dict_new)

        if len(visited_pulses) != len(self.resolved_pulses):
            log.error(f"{len(visited_pulses), len(self.resolved_pulses)}")
            for unpulse in visited_pulses:
                if unpulse not in self.resolved_pulses:
                    log.error(unpulse)
            raise Exception(f'Not all pulses have been resolved: '
                            f'{self.resolved_pulses}')

        # adds the resolved pulses to the elements OrderedDictionary
        for (t0, i, p) in sorted(visited_pulses):
            if p.pulse_obj.element_name not in self.elements:
                self.elements[p.pulse_obj.element_name] = [p.pulse_obj]
            elif p.pulse_obj.element_name in self.elements:
                self.elements[p.pulse_obj.element_name].append(p.pulse_obj)

        # sort resolved_pulses by ascending pulse middle. Used for Z_gate
        # resolution
        for i in range(len(visited_pulses)):
            t0 = visited_pulses[i][0]
            p = visited_pulses[i][2]
            visited_pulses[i] = (t0 + p.pulse_obj.length / 2,
                                 visited_pulses[i][1], p)

        ordered_unres_pulses = []
        for (t0, i, p) in sorted(visited_pulses):
            ordered_unres_pulses.append(p)

        self.resolved_pulses = ordered_unres_pulses

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

        # * generate the pulse_area dictionary containing for each channel
        #   that has to be compensated the sum of all pulse areas on that
        #   channel + the name of the last element
        # * and find the end time of the last pulse of the segment
        for element in self.element_start_end.keys():
            # finds the channels of AWGs with that element
            awg_channels = set()
            for awg in self.element_start_end[element]:
                chan = set(self.pulsar.find_awg_channels(awg))
                awg_channels = awg_channels.union(chan)

            # Calculate the tvals dictionary for the element
            tvals = self.tvals(compensation_chan & awg_channels, element)

            for pulse in self.elements[element]:
                # Find the end of the last pulse of the segment
                t_end = max(t_end, pulse.algorithm_time() + pulse.length)

                for c in pulse.channels:
                    if c not in compensation_chan:
                        continue
                    awg = self.pulsar.get('{}_awg'.format(c))
                    element_start_time = self.get_element_start(element, awg)
                    pulse_start = self.time2sample(
                        pulse.element_time(element_start_time), channel=c)
                    pulse_end = self.time2sample(
                        pulse.element_time(element_start_time) + pulse.length,
                        channel=c)

                    if c in pulse_area:
                        pulse_area[c][0] += pulse.pulse_area(
                            c, tvals[c][pulse_start:pulse_end])
                        # Overwrite this entry for all elements. The last
                        # element on that channel will be the one that
                        # is saved.
                        pulse_area[c][1] = element
                    else:
                        pulse_area[c] = [
                            pulse.pulse_area(
                                c, tvals[c][pulse_start:pulse_end]), element
                        ]

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
                RO_awg = self.pulsar.get('{}_awg'.format(c))
                if RO_awg not in comp_dict:
                    last_element = 'compensation_el{}_{}'.format(
                        comp_i, self.name)
                    comp_dict[RO_awg] = last_element
                    self.elements[last_element] = []
                    self.element_start_end[last_element] = {RO_awg: [t_end, 0]}
                    self.elements_on_awg[RO_awg].append(last_element)
                    comp_i += 1
                else:
                    last_element = comp_dict[RO_awg]

            kw = {
                'amplitude': amp,
                'buffer_length_start': comp_delay,
                'buffer_length_end': comp_delay,
                'pulse_length': length
            }
            pulse = pl.BufferedSquarePulse(
                last_element, c, name='compensation_pulse_{}'.format(i), **kw)
            i += 1

            # Set the pulse to start after the last pulse of the sequence
            pulse.algorithm_time(t_end)

            # Save the length of the longer pulse in longest_pulse dictionary
            total_length = 2 * comp_delay + length
            longest_pulse[(last_element,awg)] = \
                    max(longest_pulse.get((last_element,awg),0), total_length)

            self.elements[last_element].append(pulse)

        for (el, awg) in longest_pulse:
            length_comp = longest_pulse[(el, awg)]
            el_start = self.get_element_start(el, awg)
            new_end = t_end + length_comp
            new_samples = self.time2sample(new_end - el_start, awg=awg)
            self.element_start_end[el][awg][1] = new_samples

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

        for element in self.elements:
            for pulse in self.elements[element]:
                for channel in pulse.channels:
                    awg = self.pulsar.get(channel + '_awg')
                    if awg in self.elements_on_awg and \
                        element not in self.elements_on_awg[awg]:
                        self.elements_on_awg[awg].append(element)
                    elif awg not in self.elements_on_awg:
                        self.elements_on_awg[awg] = [element]

    def find_awg_hierarchy(self):
        masters = {awg for awg in self.pulsar.awgs
            if len(self.pulsar.get('{}_trigger_channels'.format(awg))) == 0}

        # generate dictionary triggering_awgs (keys are trigger AWGs and
        # values triggered AWGs) and tirggered_awgs (keys are triggered AWGs
        # and values triggering AWGs)
        triggering_awgs = {}
        triggered_awgs = {}
        awgs = set(self.pulsar.awgs) - masters
        for awg in awgs:
            for channel in self.pulsar.get('{}_trigger_channels'.format(awg)):
                trigger_awg = self.pulsar.get('{}_awg'.format(channel))
                if trigger_awg in triggering_awgs:
                    triggering_awgs[trigger_awg].append(awg)
                else:
                    triggering_awgs[trigger_awg] = [awg]
                if awg in triggered_awgs:
                    triggered_awgs[awg].append(trigger_awg)
                else:
                    triggered_awgs[awg] = [trigger_awg]

        # impletment Kahn's algorithm to sort the AWG by hierarchy
        trigger_awgs = masters
        awg_hierarchy = []

        while len(trigger_awgs) != 0:
            awg = trigger_awgs.pop()
            awg_hierarchy.append(awg)
            if awg not in triggering_awgs:
                continue
            for triggered_awg in triggering_awgs[awg]:
                triggered_awgs[triggered_awg].remove(awg)
                if len(triggered_awgs[triggered_awg]) == 0:
                    trigger_awgs.add(triggered_awg)

        awg_hierarchy.reverse()
        return awg_hierarchy

    def gen_trigger_el(self):
        """
        For each element:
            For each AWG the element is played on, this method:
                * adds the element to the elements_on_AWG dictionary
                * instatiates a trigger pulse on the triggering channel of the
                  AWG, placed in a suitable element on the triggering AWG,
                  taking AWG delay into account.
                * adds the trigger pulse to the elements list 
        """

        # Generate the dictionary elements_on_awg, that for each AWG contains
        # a list of the elements on that AWG
        self.gen_elements_on_awg()

        # Find the AWG hierarchy. Needed to add the trigger pulses first to
        # the AWG that do not trigger any other AWGs, then the AWGs that
        # trigger these AWGs and so on.
        awg_hierarchy = self.find_awg_hierarchy()

        i = 1
        for awg in awg_hierarchy:
            if awg not in self.elements_on_awg:
                continue

            # for master AWG no trigger_pulse has to be added
            if len(self.pulsar.get('{}_trigger_channels'.format(awg))) == 0:
                continue

            # used for updating the length of the trigger elements after adding
            # the trigger pulses
            trigger_el_set = set()

            for element in self.elements_on_awg[awg]:
                # Calculate the trigger pulse time
                [el_start, _] = self.element_start_length(element, awg)

                trigger_pulse_time = el_start - \
                                     - self.pulsar.get('{}_delay'.format(awg))\
                                     - self.trigger_pars['buffer_length_start']

                # Find the trigger_AWGs that trigger the AWG
                trigger_awgs = set()
                for channel in self.pulsar.get(
                        '{}_trigger_channels'.format(awg)):
                    trigger_awgs.add(self.pulsar.get('{}_awg'.format(channel)))

                # For each trigger_AWG, find the element to play the trigger
                # pulse in
                trigger_elements = {}
                for trigger_awg in trigger_awgs:
                    # if there is no element on that AWG create a new element
                    if self.elements_on_awg.get(trigger_awg, None) is None:
                        trigger_elements[
                            trigger_awg] = 'trigger_element_{}'.format(
                                self.name)
                    # else find the element that is closest to the
                    # trigger pulse
                    else:
                        trigger_elements[
                            trigger_awg] = self.find_trigger_element(
                                trigger_awg, trigger_pulse_time)

                # Add the trigger pulse to all triggering channels
                for channel in self.pulsar.get(
                        '{}_trigger_channels'.format(awg)):

                    trigger_awg = self.pulsar.get('{}_awg'.format(channel))
                    trig_pulse = pl.BufferedSquarePulse(
                        trigger_elements[trigger_awg],
                        channel=channel,
                        name='trigger_pulse_{}'.format(i),
                        **self.trigger_pars)
                    i += 1

                    trig_pulse.algorithm_time(trigger_pulse_time)

                    # Add trigger element and pulse to seg.elements
                    if trig_pulse.element_name in self.elements:
                        self.elements[trig_pulse.element_name].append(
                            trig_pulse)
                    else:
                        self.elements[trig_pulse.element_name] = [trig_pulse]

                    # Add the trigger_element to elements_on_awg[trigger_awg]
                    if trigger_awg not in self.elements_on_awg:
                        self.elements_on_awg[trigger_awg] = [
                            trigger_elements[trigger_awg]
                        ]
                    elif trigger_elements[
                            trigger_awg] not in self.elements_on_awg[
                                trigger_awg]:
                        self.elements_on_awg[trigger_awg].append(
                            trigger_elements[trigger_awg])

                    trigger_el_set = trigger_el_set | set(
                        trigger_elements.items())

            # For all trigger elements update the start and length
            # after having added the trigger pulses
            for (awg, el) in trigger_el_set:
                self.element_start_length(el, awg)

        # checks if elements on AWGs overlap
        self._test_overlap()
        # checks if there is only one element on the master AWG
        self._test_trigger_awg()

    def find_trigger_element(self, trigger_awg, trigger_pulse_time):
        """
        For a trigger_AWG that is used for generating triggers as well as 
        normal pulses, this method returns the name of the element to which the 
        trigger pulse is closest.
        """

        time_distance = []

        for element in self.elements_on_awg[trigger_awg]:
            [el_start, samples] = self.element_start_length(
                element, trigger_awg)
            el_end = el_start + self.sample2time(samples, awg=trigger_awg)
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

    def get_element_end(self, element, awg):
        """
        This method returns the end of an element on an AWG in algorithm_time 
        """

        samples = self.element_start_end[element][awg][1]
        length = self.sample2time(samples, awg=awg)
        return self.element_start_end[element][awg][0] + length

    def get_element_start(self, element, awg):
        """
        This method returns the start of an element on an AWG in algorithm_time 
        """
        return self.element_start_end[element][awg][0]

    def _test_overlap(self):
        """
        Tests for all AWGs if any of their elements overlap.
        """

        for awg in self.elements_on_awg:
            el_list = []
            i = 0
            for el in self.elements_on_awg[awg]:
                if el not in self.element_start_end:
                    self.element_start_length(el, awg)
                el_list.append([self.element_start_end[el][awg][0], i, el])
                i += 1

            el_list.sort()

            for i in range(len(el_list) - 1):
                prev_el = el_list[i][2]
                el_prev_start = self.get_element_start(prev_el, awg)
                el_prev_end = self.get_element_end(prev_el, awg)
                el_length = el_prev_end - el_prev_start

                # If element length is shorter than min length, 0s will be
                # appended by pulsar. Test for elements with at least
                # min_el_len if they overlap.
                min_el_len = self.pulsar.get('{}_min_length'.format(awg))
                if el_length < min_el_len:
                    el_prev_end = el_prev_start + min_el_len

                el_new_start = el_list[i + 1][0]

                if el_prev_end > el_new_start:
                    raise ValueError('{} and {} overlap on {}'.format(
                        prev_el, el_list[i + 1][2], awg))

    def _test_trigger_awg(self):
        """
        Checks if there is more than one element on the AWGs that are not 
        triggered by another AWG.
        """
        self.gen_elements_on_awg()

        for awg in self.elements_on_awg:
            if len(self.pulsar.get('{}_trigger_channels'.format(awg))) != 0:
                continue
            if len(self.elements_on_awg[awg]) > 1:
                raise ValueError(
                    'There is more than one element on {}'.format(awg))

    def resolve_Z_gates(self):
        """
        The phase of a basis rotation is acquired by an basis pulse, if the
        middle of the basis rotation pulse happens before the middle of the
        basis pulse. Using that self.resolved_pulses was sorted by
        self.resolve_timing() the acquired phases can be calculated.
        """

        basis_phases = {}

        for pulse in self.resolved_pulses:
            for basis, rotation in pulse.basis_rotation.items():
                basis_phases[basis] = basis_phases.get(basis, 0) + rotation

            if pulse.basis is not None:
                pulse.pulse_obj.phase = pulse.original_phase - \
                                        basis_phases.get(pulse.basis, 0)

    def element_start_length(self, element, awg):
        """
        Finds and saves the start and length of an element on AWG awg
        in self.element_start_end.
        """
        if element not in self.element_start_end:
            self.element_start_end[element] = {}

        # find element start, end and length
        t_start = float('inf')
        t_end = -float('inf')

        for pulse in self.elements[element]:
            t_start = min(pulse.algorithm_time(), t_start)
            t_end = max(pulse.algorithm_time() + pulse.length, t_end)

        # make sure that element start is a multiple of element
        # start granularity
        # we allow rounding up of the start time by half a sample, otherwise
        # we round the start time down
        start_gran = self.pulsar.get(
            '{}_element_start_granularity'.format(awg))
        sample_time = 1/self.pulsar.clock(awg=awg)
        if start_gran is not None:
            t_start = math.floor((t_start + 0.5*sample_time) / start_gran) \
                      * start_gran

        # make sure that element length is multiple of
        # sample granularity
        gran = self.pulsar.get('{}_granularity'.format(awg))
        samples = self.time2sample(t_end - t_start, awg=awg)
        if samples % gran != 0:
            samples += gran - samples % gran

        self.element_start_end[element][awg] = [t_start, samples]

        return [t_start, samples]

    def waveforms(self, awgs=None, elements=None, channels=None,
                        codewords=None):
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
        if awgs is None:
            awgs = set(self.elements_on_awg)
        if channels is None:
            channels = set(self.pulsar.channels)
        if elements is None:
            elements = set(self.elements)

        awg_wfs = {}
        for awg in awgs:
            # only procede for AWGs with waveforms
            if awg not in self.elements_on_awg:
                continue
            awg_wfs[awg] = {}
            channel_list = set(self.pulsar.find_awg_channels(awg)) & channels
            if channel_list == set():
                continue
            channel_list = list(channel_list)
            for i, element in enumerate(self.elements_on_awg[awg]):
                if element not in elements:
                    continue
                awg_wfs[awg][(i, element)] = {}
                tvals = self.tvals(channel_list, element)
                wfs = {}
                element_start_time = self.get_element_start(element, awg)
                for pulse in self.elements[element]:
                    # checks whether pulse is played on AWG
                    pulse_channels = set(pulse.channels) & set(channel_list)
                    if pulse_channels == set():
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
                        chan_tvals[channel] = tvals[channel].copy(
                        )[pulse_start:pulse_end]

                    # calculate pulse waveforms
                    pulse_wfs = pulse.waveforms(chan_tvals)

                    # insert the waveforms at the correct position in wfs
                    for channel in pulse_channels:
                        wfs[pulse.codeword][channel][
                            pulse_start:pulse_end] += pulse_wfs[channel]


                # for codewords: add the pulses that do not have a codeword to
                # all codewords
                if 'no_codeword' in wfs:
                    for codeword in wfs:
                        if codeword is not 'no_codeword':
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

                        distortion_dictionary = self.pulsar.get(
                            '{}_distortion_dict'.format(c))
                        fir_kernels = distortion_dictionary.get('FIR', None)
                        if fir_kernels is not None:
                            if hasattr(fir_kernels, '__iter__') and not \
                            hasattr(fir_kernels[0], '__iter__'): # 1 kernel
                                wf = flux_dist.filter_fir(fir_kernels, wf)
                            else:
                                for kernel in fir_kernels:
                                    wf = flux_dist.filter_fir(kernel, wf)
                        iir_filters = distortion_dictionary.get('IIR', None)
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
                        if self.pulsar.get('{}_type'.format(c)) == 'analog':
                            if np.max(wfs[codeword][c]) > amp:
                                logging.warning(
                                    'Clipping waveform {} > {}'.format(
                                        np.max(wfs[codeword][c]), amp))
                            if np.min(wfs[codeword][c]) < -amp:
                                logging.warning(
                                    'Clipping waveform {} < {}'.format(
                                        np.min(wfs[codeword][c]), -amp))
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
                                .astype(np.int)

                # save the waveforms in the dictionary
                for codeword in wfs:
                    awg_wfs[awg][(i, element)][codeword] = {}
                    for channel in wfs[codeword]:
                        awg_wfs[awg][(i, element)][codeword][self.pulsar.get(
                            '{}_id'.format(channel))] = (
                                wfs[codeword][channel])

        return awg_wfs

    def get_element_codewords(self, element, awg=None):
        codewords = set()
        if awg is not None:
            channels = set(self.pulsar.find_awg_channels(awg))
        for pulse in self.elements[element]:
            if awg is not None and len(set(pulse.channels) & channels) == 0:
                continue
            codewords.add(pulse.codeword)
        return codewords

    def get_element_channels(self, element, awg=None):
        channels = set()
        if awg is not None:
            awg_channels = set(self.pulsar.find_awg_channels(awg))
        for pulse in self.elements[element]:
            if awg is not None:
                channels |= set(pulse.channels) & awg_channels
            else:
                channels |= set(pulse.channels)
        return channels

    def calculate_hash(self, elname, codeword, channel):
        if not self.pulsar.reuse_waveforms():
            return (self.name, elname, codeword, channel)

        awg = self.pulsar.get(f'{channel}_awg')
        tstart, length = self.element_start_end[elname][awg]
        hashlist = []
        hashlist.append(length)  # element length in samples
        if self.pulsar.get(f'{channel}_type') == 'analog' and \
                self.pulsar.get(f'{channel}_distortion') == 'precalculate':
            # don't compare the kernels, just assume that all channels'
            # distortion kernels are different
            hashlist.append(channel)
        else:
            hashlist.append(self.pulsar.clock(channel=channel))  # clock rate
            for par in ['type', 'amp', 'internal_modulation']:
                try:
                    hashlist.append(self.pulsar.get(f'{channel}_{par}'))
                except KeyError:
                    hashlist.append(False)

        for pulse in self.elements[elname]:
            if pulse.codeword in {'no_codeword', codeword}:
                hashlist += pulse.hashables(tstart, channel)
        return tuple(hashlist)


    def tvals(self, channel_list, element):
        """
        Returns a dictionary with channel names of the used channels in the
        element as keys and the tvals array for the channel as values.
        """

        tvals = {}

        for channel in channel_list:
            samples = self.get_element_samples(element, channel)
            awg = self.pulsar.get('{}_awg'.format(channel))
            tvals[channel] = np.arange(samples) / self.pulsar.clock(
                channel=channel) + self.get_element_start(element, awg)

        return tvals

    def get_element_samples(self, element, instrument_ref):
        """
        Returns the number of samples the element occupies for the channel or
        AWG.
        """

        if instrument_ref in self.pulsar.channels:
            awg = self.pulsar.get('{}_awg'.format(instrument_ref))
        elif instrument_ref in self.pulsar.awgs:
            awg = instrument_ref
        else:
            raise Exception('instrument_ref has to be channel or AWG name!')

        return self.element_start_end[element][awg][1]

    def time2sample(self, t, **kw):
        """
        Converts time to a number of samples for a channel or AWG.
        """
        return int(t * self.pulsar.clock(**kw) + 0.5)

    def sample2time(self, samples, **kw):
        """
        Converts nubmer of samples to time for a channel or AWG.
        """
        return samples / self.pulsar.clock(**kw)

    def plot(self, instruments=None, channels=None, legend=True,
             delays=None, savefig=False, prop_cycle=None, frameon=True,
             channel_map=None, plot_kwargs=None, axes=None, demodulate=False):
        """
        Plots a segment. Can only be done if the segment can be resolved.
        :param instruments (list): instruments for which pulses have to be plotted.
            defaults to all.
        :param channels (list):  channels to plot. defaults to all.
        :param delays (dict): keys are instruments, values are additional delays.
            if passed, the delay is substracted to the time values of this
            instrument, such that the pulses are plotted at timing when they
            physically occur.
        :param savefig: save the plot
        :param channel_map (dict): indicates which instrument channels correspond to
            whichqubits. Keys = qb names, values = list of channels. eg.
            dict(qb2=['AWG8_ch3', "UHF_ch1"]). If provided, will plot each qubit
            on individual subplots.
        :param prop_cycle (dict):
        :param frameon (dict, bool):
        :param axes (array or axis): 2D array of matplotlib axes. if single axes,
            will be converted internally to array.
        :param demodulate (bool): plot only envelope of pulses by temporarily setting
            modulation and phase to 0. Need to recompile the sequence
        :return:
        """
        import matplotlib.pyplot as plt
        if delays is None:
            delays = dict()
        if plot_kwargs is None:
            plot_kwargs = dict()
            plot_kwargs['linewidth'] = 0.7
        try:
            # resolve segment and populate elements/waveforms
            self.resolve_segment()
            if demodulate:
                for el in self.elements.values():
                    for pulse in el:
                        if hasattr(pulse, "mod_frequency"):
                            pulse.mod_frequency = 0
                        if hasattr(pulse, "phase"):
                            pulse.phase = 0
            wfs = self.waveforms(awgs=instruments, channels=None)
            n_instruments = len(wfs) if channel_map is None else len(channel_map)
            if axes is not None:
                if np.ndim(axes) == 0:
                    axes = [[axes]]
                fig = axes[0,0].get_figure()
                ax = axes
            else:
                fig, ax = plt.subplots(nrows=n_instruments, sharex=True,
                                       squeeze=False,
                                       figsize=(16, n_instruments * 3))
            if prop_cycle is not None:
                for a in ax[:,0]:
                    a.set_prop_cycle(**prop_cycle)
            for i, instr in enumerate(wfs):
                # plotting
                for elem_name, v in wfs[instr].items():
                    for k, wf_per_ch in v.items():
                        for n_wf, (ch, wf) in enumerate(wf_per_ch.items()):
                            if channels is None or \
                                    ch in channels.get(instr, []):
                                tvals = \
                                self.tvals([f"{instr}_{ch}"], elem_name[1])[
                                    f"{instr}_{ch}"] - delays.get(instr, 0)
                                if channel_map is None:
                                    # plot per device
                                    ax[i, 0].plot(tvals * 1e6, wf,
                                                  label=f"{elem_name[1]}_{k}_{ch}",
                                                  **plot_kwargs)
                                else:
                                    # plot on each qubit subplot which includes
                                    # this channel in the channel map
                                    match = [i for i, (_, qb_chs) in
                                                     enumerate(channel_map.items())
                                                     if f"{instr}_{ch}" in qb_chs]
                                    for qbi in match:
                                        ax[qbi, 0].plot(tvals * 1e6, wf,
                                                      label=f"{elem_name[1]}_{k}_{ch}",
                                                      **plot_kwargs)
                                        if demodulate: # filling
                                            ax[qbi, 0].fill_between(tvals * 1e6, wf,
                                                            label=f"{elem_name[1]}_{k}_{ch}",
                                                            alpha=0.05,
                                                            **plot_kwargs)


            # formatting
            for a in ax[:,0]:
                if isinstance(frameon, bool):
                    frameon = {k: frameon for k in ['top', 'bottom',
                                                    "right", "left"]}
                a.spines["top"].set_visible(frameon.get("top", True))
                a.spines["right"].set_visible(frameon.get("right", True))
                a.spines["bottom"].set_visible(frameon.get("bottom", True))
                a.spines["left"].set_visible(frameon.get("left", True))
                if legend:
                    a.legend(loc=[1.02, 0], prop={'size': 8})
                a.set_ylabel('Voltage (V)')
            ax[-1, 0].set_xlabel('time ($\mu$s)')
            fig.suptitle(f'{self.name}')
            plt.tight_layout()
            if savefig:
                plt.savefig(f'{self.name}.png')
            # plt.show()
            return fig, ax
        except Exception as e:
            log.error(f"Could not plot: {self.name}")
            raise e
    def __repr__(self):
        string_repr = f"---- {self.name} ----\n"

        for i, p in enumerate(self.unresolved_pulses):
            string_repr += f"{i}: " + repr(p) + "\n"
        return string_repr

    def export_tikz(self, qb_names, tscale=1e-6):
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
            if p.op_code != '' and p.op_code[:2] != 'RO':
                l = p.pulse_obj.length
                t = p.pulse_obj._t0 + l / 2
                tmin = min(tmin, p.pulse_obj._t0)
                tmax = max(tmax, p.pulse_obj._t0 + p.pulse_obj.length)
                qb = qb_names.index(p.op_code[-3:])
                op_code = p.op_code[:-4]
                qbt = 0
                if op_code[-3:-1] == 'qb':
                    qbt = qb_names.index(op_code[-3:])
                    op_code = op_code[:-4]
                if op_code[-1:] == 's':
                    op_code = op_code[:-1]
                if op_code[:2] == 'CZ' or op_code[:4] == 'upCZ':
                    num_two_qb += 1
                    if len(op_code) > 4:
                        val = -float(op_code[4:])
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

    def rename(self, new_name):
        """
        Renames a segment with the given new name. Hunts down element names in
        unresolved pulses that might have made use of the old segment_name and renames
        them too.
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
        # rename segment name
        self.name = new_name

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
    ref_pulse: 'segment_start', 'previous_pulse', pulse.name, or a list of
        multiple pulse.name.
    ref_point: 'start', 'end', 'middle', reference point of the reference pulse
    ref_point_new: 'start', 'end', 'middle', reference point of the new pulse
    ref_function: 'max', 'min', 'mean', specifies how timing is chosen if
        multiple pulse names are listed in ref_pulse (default: 'max')
    """

    def __init__(self, pulse_pars):
        self.ref_pulse = pulse_pars.get('ref_pulse', 'previous_pulse')
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
        self.delay = pulse_pars.get('pulse_delay', 0)
        self.original_phase = pulse_pars.get('phase', 0)
        self.basis = pulse_pars.get('basis', None)
        self.operation_type = pulse_pars.get('operation_type', None)
        self.basis_rotation = pulse_pars.pop('basis_rotation', {})
        self.op_code = pulse_pars.get('op_code', '')

        pulse_func = None
        for module in bpl.pulse_libraries:
            try:
                pulse_func = getattr(module, pulse_pars['pulse_type'])
            except AttributeError:
                pass
        if pulse_func is None:
            raise KeyError('pulse_type {} not recognized'.format(
                pulse_pars['pulse_type']))

        self.pulse_obj = pulse_func(**pulse_pars)
        # allow a pulse to modify its op_code (e.g., for C-ARB gates)
        self.op_code = getattr(self.pulse_obj, 'op_code', self.op_code)

        if self.pulse_obj.codeword != 'no_codeword' and \
                self.basis_rotation != {}:
            raise Exception(
                'Codeword pulse {} does not support basis_rotation!'.format(
                    self.pulse_obj.name))

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
