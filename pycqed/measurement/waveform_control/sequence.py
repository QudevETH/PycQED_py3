# A Sequence contains segments which then contain the pulses. The Sequence
# provides the information for the AWGs, in which order to play the segments.
#
# author: Michael Kerschbaum
# created: 04/2019

import numpy as np
import pycqed.measurement.waveform_control.pulsar as ps
from collections import OrderedDict as odict
from copy import deepcopy, copy
import logging

from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)

class Sequence:
    """
    A Sequence consists of several segments, which can be played back on the 
    AWGs sequentially.
    """

    RENAMING_SEPARATOR = "+"
    AMPLITUDE_ROUNDING_DIGITS = 7
    """Specifies the rounding precision when processing waveform amplitudes 
    in harmonize_amplitude method. If this parameter has value n, then the 
    waveform amplitudes will be rounded to the n-th digit of V (volt)."""

    def __init__(self, name, segments=()):
        """
        Initializes a Sequence object
        Args:
            name: Name of the sequence
            segments (list, tuple): list of segments to add to the sequence
        """
        self.name = name
        self.timer = Timer(self.name)
        self.pulsar = ps.Pulsar.get_instance()
        self.segments = odict()
        self.awg_sequence = {}
        self.repeat_patterns = {}
        self.extend(segments)
        self.is_resolved = False
        self.awg_scaling_factors = dict()
        """A list of AWG names whose pulse amplitudes has processed with 
        method 'self.harmonize_amplitude'."""

    def add(self, segment):
        if segment.name in self.segments:
            raise NameError('Name {} already exisits in the sequence!'.format(
                segment.name))
        self.segments[segment.name] = segment
        if len(self.segments) == 1:
            self.segments[segment.name].is_first_segment = True
        self.timer.children.update({segment.name: segment.timer})

    def extend(self, segments):
        """
        Extends the sequence given a list of segments
        Args:
            segments (list): segments to add to the sequence
        """
        for seg in segments:
            self.add(seg)

    def upload(self, awgs_to_upload='all'):
        """Upload the sequence to the AWG using self.pulsar instrument.
        """
        if awgs_to_upload != 'all':
            log.warning('Sequence:upload: reducing upload overhead manually '
                        'with awgs_to_upload is deprecated. Set '
                        'pulsar.use_sequence_cache to True for automatic '
                        'reduction of upload overhead.')
        self.pulsar.program_awgs(self, awgs=awgs_to_upload)

    @Timer()
    def generate_waveforms_sequences(self, awgs=None,
                                     get_channel_hashes=False,
                                     resolve_segments=None,
                                     trigger_groups=None,
                                     awg_sequences=None):
        """
        Calculates and returns 
            * waveforms: a dictionary of waveforms used in the sequence,
                indexed by their hash value
            * sequences: For each awg, a list of elements, each element
                consisting of a waveform-hash for each codeword and each
                channel
        :param awgs: a list of AWG names. If None, waveforms will be
            generated for all AWGs used in the sequence.
        :param get_channel_hashes: (bool, default: False) do not create
            waveforms, and instead return a dict of channel-elements indexed
            by channel names, each channel-element consisting of a
            waveform-hash for each codeword on that channel.
        :param resolve_segments: (bool, optional) whether the segments in the
            sequence still need to be resolved. If not provided,
            self.is_resolved is checked to determine whether segments
            need to be resolved.
        :param awg_sequences: (dict, optional) Sequences, as previously
            returned by this method. Can be used for caching in case this
            method is called multiple times. If provided, it is assumed that
            this dict is complete (no error handling in case keys are missing).
        :return: a tuple of waveforms, sequences as described above if
            get_channel_hashes==False. Otherwise, a tuple channel_hashes,
            sequences.
            Note that the information contained in channel_hashes is the
            same as in sequences, but in a different structure, which has the
            channel name as highest-level key:
            sequences[awg][elname][cw][chid] == channel_hashes[ch][elname][cw]
        """
        waveforms = {}
        sequences = {}
        channel_hashes = {}
        if resolve_segments or (resolve_segments is None
                                and not self.is_resolved):
            for seg in self.segments.values():
                seg.resolve_segment()
                seg.gen_elements_on_awg()

        if trigger_groups is None:
            trigger_groups = set()
            for seg in self.segments.values():
                trigger_groups |= set(seg.elements_on_awg)

        if awgs is None:
            awgs = set()
            for group in trigger_groups:
                awgs.add(self.pulsar.get_awg_from_trigger_group(group))

        # Note that method 'self.generate_waveforms_sequences' will be
        # called by 'pulsar._program_awgs' multiple times, but we only
        # want to execute 'self.harmonize_amplitude' once. Otherwise,
        # we will have wrong pulse amplitudes and scaling parameters.
        for awg in awgs:
            if awg not in self.awg_scaling_factors.keys():
                self.awg_scaling_factors[awg] = \
                    self.harmonize_amplitude(awg)

        for segname, seg in self.segments.items():
            for group in trigger_groups:
                awg = self.pulsar.get_awg_from_trigger_group(group)
                if awg not in awgs:
                    continue
                scaling_factors = self.awg_scaling_factors[awg]
                sequences.setdefault(awg, odict())
                # Store name of the segment as key and None as value.
                # This is used when compiling docstrings in seqc.
                sequences[awg].setdefault(segname, None)

                # Take element metadata from the resolved segments.
                element_metadata = seg.element_metadata
                elnames = seg.elements_on_awg.get(group, [])
                # Determine when each element starts in the current group
                el_start_times = {
                    elname: seg.element_start_length(elname, group)[0]
                    for elname in elnames}
                # Loop through elements in the order of their start time
                for i in np.argsort(list(el_start_times.values())):
                    elname = elnames[i]
                    # uelname = element name unique within the AWG
                    # If elements are shared between trigger groups of an AWG,
                    # this ensures that the following logic correctly orders
                    # waveforms within each trigger group
                    uelname = group + '_' + elname
                    sequences[awg].setdefault(uelname, {'metadata': {}})
                    metadata = sequences[awg][uelname]['metadata']
                    for cw in seg.get_element_codewords(elname,
                                                        trigger_group=group):
                        sequences[awg][uelname].setdefault(cw, {})
                        for ch in seg.get_element_channels(elname,
                                                           trigger_group=group):
                            chid = self.pulsar.get(f'{ch}_id')
                            if awg_sequences:
                                h = awg_sequences[awg][uelname][cw][chid]
                            else:
                                h = seg.calculate_hash(elname, cw, ch,
                                                       trigger_group=group)
                            sequences[awg][uelname][cw][chid] = h
                            if get_channel_hashes:
                                if ch not in channel_hashes:
                                    channel_hashes[ch] = {}
                                if uelname not in channel_hashes[ch]:
                                    channel_hashes[ch][uelname] = {}
                                channel_hashes[ch][uelname][cw] = h
                            else:
                                if h not in waveforms:
                                    wf = seg.waveforms(awgs={awg},
                                        elements={elname}, channels={ch},
                                        codewords={cw})
                                    waveforms[h] = wf.popitem()[1].popitem()[1]\
                                                     .popitem()[1].popitem()[1]
                    if elname in seg.acquisition_elements:
                        metadata['acq'] = seg.acquisition_mode
                    else:
                        metadata['acq'] = False
                    metadata['allow_filter'] = seg.allow_filter
                    metadata.setdefault('trigger_groups', set())
                    metadata['trigger_groups'].add(group)
                    # Write modulation and sine configuration to element
                    if elname in element_metadata.keys() \
                            and 'mod_config' in element_metadata[elname].keys()\
                                and element_metadata[elname]['mod_config']:
                        metadata['mod_config'] = \
                            element_metadata[elname]['mod_config']
                    elif seg.mod_config:
                        metadata['mod_config'] = seg.mod_config
                    if seg.sine_config:
                        metadata['sine_config'] = seg.sine_config
                    # Pass command table scaling factors to element metadata
                    if elname in scaling_factors.keys():
                        metadata['scaling_factor'] = scaling_factors[elname]
                # Experimental feature to sweep values of nodes of ZI HDAWGs
                # in a hard sweep. See the comments above the sweep_params
                # property in Segment.
                if seg.sweep_params is not None and len(seg.sweep_params):
                    sequences[awg][group + '_' + elnames[0]]['metadata']['loop'] = len(
                        list(seg.sweep_params.values())[0])
                    sequences[awg][group + '_' + elnames[0]]['metadata']['sweep_params'] = \
                        {k[len(awg) + 1:]: v for k, v in
                         seg.sweep_params.items() if k.startswith(awg + '_')}
                    sequences[awg][group + '_' + elnames[-1]]['metadata']['end_loop'] = True
        self.is_resolved = True
        if get_channel_hashes:
            return channel_hashes, sequences
        else:
            return waveforms, sequences

    @staticmethod
    def harmonize_element_lengths(sequences, awgs=None):
        """
        Given a list of sequences, this function ensures for all AWGs and all
        elements that the element length is the same in all sequences. This is
        done by setting the length of each AWG element to the maximum length
        across all sequences. After this, overlap is checked and the sequences
        are marked as resolved.
        :param sequences: a list of sequences
        :param awgs: a list of AWG names. If None, lengths will be harmonized
            for all AWGs.
        """
        # Setting the property will prequery all AWG clock and amplitudes
        sequences[0].pulsar.awgs_prequeried = True
        seq_groups = []
        if awgs is None:
            awgs = sequences[0].pulsar.awgs
        # collect element lengths
        lengths = odict()
        for i, seq in enumerate(sequences):
            seq_groups.append(set())
            for seg in seq.segments.values():
                seg.resolve_segment()
                seg.gen_elements_on_awg()
            seq_groups[i] |= set(
                [group for group in seg.elements_on_awg
                 if seq.pulsar.get_awg_from_trigger_group(group) in awgs])
            for group in seq_groups[i]:
                if group not in lengths:
                    lengths[group] = odict()
                for segname, seg in seq.segments.items():
                    elnames = seg.elements_on_awg.get(group, [])
                    for elname in elnames:
                        if elname not in lengths[group]:
                            lengths[group][elname] = []
                        lengths[group][elname].append(
                            seg.element_start_end[elname][group][1])
        # set element lengths to the maximum of the collected values
        for i, seq in enumerate(sequences):
            for group in seq_groups[i]:
                for segname, seg in seq.segments.items():
                    elnames = seg.elements_on_awg.get(group, [])
                    for elname in elnames:
                        seg.element_start_end[elname][group][1] = max(
                            lengths[group][elname])
            # test for overlaps
            for segname, seg in seq.segments.items():
                seg._test_overlap()
            # mark sequence as resolved
            seq.is_resolved = True
        sequences[0].pulsar.awgs_prequeried = False

    def harmonize_amplitude(self, awg):
        """Rescale waveform amplitudes such that the largest pulse amplitude
        in an element is the same as the largest in that sequence. The 
        scaling factor is saved in the dictionary scaling_factors and 
        passed to element metadata, such that the original waveform can be 
        retrieved when generating command table entries. This allows reusing 
        waveforms to the largest extent based on wave hashes. Note that 
        the rescaling will be skipped on the target AWG modules where command 
        table is not activated.
        
        Args:
            awg: (str) AWG name to be processed.

        Returns:
            scaling_factors: (dict) factors to be passed to the command table
                entries in order to retrieve the original pulse amplitude.
        """
        scaling_factors = dict()

        # Command table wave sequencing is only implemented for
        # AWG interfaces that have awg_modules attribute.
        awg_interface = self.pulsar.awg_interfaces[awg]
        if not hasattr(awg_interface, "awg_modules"):
            return dict()

        awg_modules = awg_interface.awg_modules
        for awg_module in awg_modules:
            # harmonize_amplitude only affects AWGs whose command table wave
            # sequencing is enabled.
            if not self.pulsar.check_channel_parameter(
                    awg=awg_module.awg.name,
                    channel=awg_module.get_i_channel(),
                    parameter_suffix="_harmonize_amplitude"
            ):
                continue

            if not self.pulsar.check_channel_parameter(
                awg=awg_module.awg.name,
                channel=awg_module.get_i_channel(),
                parameter_suffix="_use_command_table"
            ):
                logging.warning(
                    f"On {awg_module.awg.name}_"
                    f"{awg_module.analog_channel_ids[0]}: PycQED will "
                    f"not harmonize amplitude for this AWG module "
                    f"because this feature only works when command table "
                    f"is enabled. Please set \"_use_command_table\" "
                    f"parameter to true for this AWG module if you want "
                    f"to allow harmonizing amplitude.")
                continue
            # Only check analog channels
            channel_ids = awg_module.analog_channel_ids
            # Collects all elements that are relevant to this AWG module.
            # Records the maximum amplitude in each element.
            elements_on_channel = dict()
            element_max_amp = dict()
            for segname, seg in self.segments.items():
                elements_on_channel[segname] = set()
                for chid in channel_ids:
                    channel = self.pulsar._id_channel(
                        chid, awg_interface.awg.name)
                    elements_on_channel[segname] |= \
                        seg.elements_on_channel.get(channel, set())

                for elname in elements_on_channel[segname]:
                    current_max_amp = 0
                    for pulse in seg.elements[elname]:
                        # boolean parameter indicating whether one pulse
                        # overlaps with the current AWG module
                        pulse_overlaps_with_channel = any([
                            self.pulsar.get(f'{channel}_id') in channel_ids
                            for channel in pulse.channels])
                        # boolean parameter indicating whether one pulse
                        # is played solely on the current AWG module
                        pulse_only_on_channel = all([
                            self.pulsar.get(f'{channel}_id') in channel_ids
                            for channel in pulse.channels])

                        if not pulse_overlaps_with_channel:
                            continue
                        elif not pulse_only_on_channel:
                            # The pulse only partially overlaps with the
                            # current AWG module. It is currently not
                            # supported to harmonize amplitude across
                            # multiple AWG modules.
                            raise RuntimeError(
                                f"On element {elname}: Harmonizing amplitude "
                                f"does not support pulse played across "
                                f"multiple AWG modules. Please disable "
                                f"harmonizing amplitude for all AWG modules "
                                f"relevant to this pulse."
                            )

                        if not pulse.SUPPORT_HARMONIZING_AMPLITUDE:
                            # Harmonizing amplitude is not allowed for this
                            # pulse type
                            raise RuntimeError(
                                f"On {awg_module.awg.name}: pulse {pulse.name} "
                                f"does not allow harmonizing amplitude. "
                                f"Please disable \"_harmonize_amplitude\" "
                                f"parameter for this AWG module."
                            )

                        if abs(getattr(pulse, "amplitude", 0)) > \
                                abs(current_max_amp):
                            current_max_amp = getattr(pulse, "amplitude", 0.0)
                    element_max_amp[elname] = current_max_amp

            if not len(element_max_amp) or \
                    max(element_max_amp.values()) == 0:
                # There is no element played on this awg module.
                continue
            # choose the pulse amplitude that has the largest absolute value
            max_val = max(element_max_amp.values())
            min_val = min(element_max_amp.values())
            sequence_max_amp = abs(max_val) \
                if abs(max_val) >= abs(min_val) else abs(min_val)
            for segname, seg in self.segments.items():
                for elname in elements_on_channel[segname]:
                    element_scaling_factor = \
                        element_max_amp[elname] / sequence_max_amp

                    # Saves element scaling factor to element metadata.
                    # This factors will be provided to command table
                    # entries to retrieve the correct absolute amplitude.
                    scaling_factors[elname] = dict()
                    for chid in channel_ids:
                        channel = self.pulsar._id_channel(
                            chid, awg_interface.awg.name)
                        scaling_factors[elname][channel] = \
                            element_scaling_factor

                    for pulse in seg.elements[elname]:
                        if len(pulse.channels) == 0 or \
                                not all([self.pulsar.get(f'{channel}_id')
                                         in channel_ids
                                         for channel in pulse.channels]):
                            continue
                        # Round amplitude to allow reusing waveform in
                        # the presence of finite numerical precision.
                        pulse.amplitude = round(
                            pulse.amplitude / element_scaling_factor,
                            self.AMPLITUDE_ROUNDING_DIGITS
                        ) if element_scaling_factor != 0 else \
                            round(sequence_max_amp,
                                  self.AMPLITUDE_ROUNDING_DIGITS)

        return scaling_factors

    def n_acq_elements(self, per_segment=False):
        """
        Gets the number of acquisition elements in the sequence.
        Args:
            per_segment (bool): Whether or not to return the number of
                acquisition elements per segment. Defaults to False.

        Returns:
            number of acquisition elements (list (if per_segment) or int)

        """
        n_readouts = [len(seg.acquisition_elements)
                      for seg in self.segments.values()]
        if not per_segment:
            n_readouts = np.sum(n_readouts)
        return n_readouts

    def n_segments(self):
        """
        Gets the number of segments in the sequence.
        """
        return len(self.segments)

    def repeat(self, pulse_name, operation_dict, pattern,
               pulse_channel_names=('I_channel', 'Q_channel')):
        """
        Creates a repetition dictionary keyed by awg channel for the pulse
        to be repeated.
        :param pulse_name: name of the pulse to repeat.
        :param operation_dict:
        :param pattern: repetition pattern (n_repetitions, nr_elements_per_loop or another loop-specification)
                        cf. Christian
        :param pulse_channel_names: names of the channels on which the pulse is
        applied.
        :return:
        """
        if operation_dict==None:
            pulse=pulse_name
        else:
            pulse = operation_dict[pulse_name]
        if not pulse.get('disable_repeat_pattern', False):
            repeat = dict()
            for ch in pulse_channel_names:
                if pulse[ch] is None:
                    continue
                repeat[pulse[ch]] = pattern
            self.repeat_patterns.update(repeat)
        return self.repeat_patterns

    def repeat_ro(self, pulse_name, operation_dict):
        """
        Wrapper for repeated readout
        :param pulse_name:
        :param operation_dict:
        :param sequence:
        :return:
        """
        return self.repeat(pulse_name, operation_dict,
                           (self.n_acq_elements(), 1))


    @staticmethod
    def merge(sequences, segment_limit=None, merge_repeat_patterns=True):
        """
        Merges a list of sequences. See documentation of Sequence.__add__()
        for more information on the merge of two sequences.
        Args:
            sequences (list): List of sequences to merge
            segment_limit (int): maximal number of segments in the merged sequence.
                if the total number of segments is higher, a list of sequences is
                returned. Default is None (all sequences are merged)
            merge_repeat_patterns (bool): Merges the readout pattern when
                 combining the sequences. If the readout pattern already exists, it adds
                 to the number of repetition of the pattern. Note that this behavior may
                 not work for all scenarios. In that case the patterns must be updated
                  manually after the merge and merge_repeat_patterns should be set to
                  False. Default: True.


        Returns: list of merged sequences

        Examples:
            >>> # No segment_limit
            >>> seq1 = Sequence('seq1')
            >>> seq1.extend(segments_of_seq1)  # 10 segments
            >>> seq2 = Sequence('seq2')
            >>> seq2.extend(segments_of_seq2) # 15 segments
            >>> seq_comb = Sequence.merge([seq1, seq2])
            >>> # returns a list with 1 sequence with 25 segments
            >>> # i.e. [seq1 + seq2]

            >>> # 20 segments limit
            >>> seq1 = Sequence('seq1')
            >>> seq1.extend(segments_of_seq1) # 10 segments
            >>> seq2 = Sequence('seq2')
            >>> seq2.extend(segments_of_seq2) # 15 segments
            >>> seq3 = Sequence('seq3')
            >>> seq3.extend(segments_of_seq3) # 5 segments
            >>> seq_comb = Sequence.merge([seq1, seq2, seq3])
            >>> # returns list of 2 sequences with 10 and 20 segments,
            >>> # i.e. [seq1, seq2 + seq3]


        """
        if len(sequences) == 0:
            raise ValueError("merge requires at least one sequence")
        elif len(sequences) == 1:
            # special case, return current sequence:
            return sequences
        sequences = [copy(s) for s in sequences]
        merged_seqs = [sequences[0]]
        if segment_limit is None:
            segment_limit = np.inf

        segment_counter = sequences[0].n_segments()
        seg_occurences = [{s: 1 for s in sequences[0].segments}]
        for seq in sequences[1:]:
            assert seq.n_segments() <= segment_limit, \
                f"Sequence {seq.name} has more segments ({seq.n_segments()})" \
                f" than the segment_limit ({segment_limit}). Cannot merge " \
                f"without cropping the sequence."
            # if over segment_limit, add another separate sequence
            # to merged sequences
            if merged_seqs[-1].n_segments() + seq.n_segments() > segment_limit:
                merged_seqs.append(seq)
                seg_occurences.append({s: 1 for s in seq.segments})
                segment_counter = seq.n_segments()
            # otherwise merge sequences
            else:
                for seg_name, segment in seq.segments.items():
                    segment.is_first_segment = False
                    try:
                        merged_seqs[-1].add(segment)
                    except NameError:  # in case segment name exists, create new name
                        seg_occurences[-1][seg_name] += 1
                        new_name =seg_name + \
                                  f"_copy_from_merge_" \
                                  f"{seg_occurences[-1][seg_name] - 1}"
                        segment.rename(new_name)
                        merged_seqs[-1].add(segment)

                segment_counter += seq.n_segments()

                # update name of merged seq
                merged_seqs[-1].rename(merged_seqs[-1].name + Sequence.RENAMING_SEPARATOR + seq.name)
                if merge_repeat_patterns:
                    for ch_name, pattern in seq.repeat_patterns.items():
                        # if channel is already present, update number of
                        # repetitions
                        if ch_name in merged_seqs[-1].repeat_patterns:
                            pattern_prev = \
                                merged_seqs[-1].repeat_patterns[ch_name]
                            if pattern_prev[1:] != pattern[1:]:
                                raise NotImplementedError(
                                    f"The repeat patterns for channel: "
                                    f"{ch_name} do not have the same "
                                    f"'outer loop' specification (see "
                                    f"docstring Sequence.repeat). Repeat "
                                    f"patterns cannot be merged automatically. "
                                    f"Set merge_repeat_patterns to False and "
                                    f"update the repeat patterns manually.")
                            pattern_updated = (pattern_prev[0] + pattern[0],
                                               *pattern_prev[1:])
                            merged_seqs[-1].repeat_patterns[ch_name] = \
                                pattern_updated
                        # add repeat pattern
                        else:
                            merged_seqs[-1].repeat_patterns.update(
                                {ch_name: pattern})
        # compress names
        for ms in merged_seqs:
            name_parts = ms.name.split(Sequence.RENAMING_SEPARATOR)
            if len(name_parts) > 2:
                ms.rename(f"compressed_{name_parts[0]}-{name_parts[-1]}")
        return merged_seqs

    @staticmethod
    def interleave_sequences(seq_list_list):
        """
        Interleave a list of Sequence instances.
        :param seq_list_list: list of lists of Sequence instances
        :return: list of interleaved Sequences
        """
        # make sure all sequence lists in seq_list_list have the same length
        if len(set([len(seq_list) for seq_list in seq_list_list])) != 1:
            raise ValueError('The sequence lists do not have the same length.')
        # make sure all sequence lists in seq_list_list have the same segments
        if len(set([seq_list[0].n_acq_elements() for
                        seq_list in seq_list_list])) != 1:
            raise ValueError('The sequence lists do not have the same number '
                             'of segments.')

        interleaved_seqs = len(seq_list_list) * len(seq_list_list[0]) * ['']
        for i in range(len(seq_list_list)):
            interleaved_seqs[i::len(seq_list_list)] = seq_list_list[i]

        # rename sequences and timers
        for i, seq in enumerate(interleaved_seqs):
            seq.rename(f"Interleaved_Sequence_{i}")

        mc_points = [np.arange(interleaved_seqs[0].n_acq_elements()),
                     np.arange(len(interleaved_seqs))]

        return interleaved_seqs, mc_points

    @staticmethod
    def compress_2D_sweep(sequences, segment_limit=None,
                          merge_repeat_patterns=True, mc_points=None):
        """
        Compresses a list of sequences to a lower number of sequences
        (if possible), each of which containing the same amount of segments
        (assumes fixed number of readout per segment) while respecting the
        segment_limit (memory limit). Note that all sequences MUST have the
        same number of segments. Wraps the Sequence.merge() by computing an
        effective segment limit that minimizes the total number of sequences
        (to reduce upload time overhead) while keeping the (new)
        number of segments per sequence constant (it currently is a limitation
        of 2D sweeps that  all sequences must have same number of readouts)
        Args:
            sequences (list): list of sequences to compress, which all have
                the same number of segments
            segment_limit (int): maximal number of segments that can be in
                a sequence
            merge_repeat_patterns (bool): see docstring of Sequence.merge.
            mc_points: mc_points array of the original hardware sweep.
                Useful in case it differs from n_acq_elements().

        Returns: list of sequences for the compressed 2D sweep,
            new hardsweep points indices,
            new soft sweeppoints indices, and the compression factor

        """
        assert len(np.unique([s.n_segments() for s in sequences])) == 1, \
            "To allow compression, all sequences must have the same number " \
            "of segments"
        n_soft_sp = len(sequences)
        n_seg = sequences[0].n_segments()
        seg_lim_eff, factor = Sequence.compute_compression_seg_lim(
            n_soft_sp, n_seg, segment_limit)
        compressed_2D_sweep = Sequence.merge(sequences, seg_lim_eff,
                                              merge_repeat_patterns)
        if mc_points is None:
            hard_sp_ind = np.arange(compressed_2D_sweep[0].n_acq_elements())
            soft_sp_ind = np.arange(len(compressed_2D_sweep))
        else:
            hard_sp_ind = np.arange(len(mc_points)*len(sequences) //
                                    len(compressed_2D_sweep))
            soft_sp_ind = np.arange(len(compressed_2D_sweep))

        return compressed_2D_sweep, hard_sp_ind, soft_sp_ind, factor

    @staticmethod
    def compute_compression_seg_lim(n_soft_sp, n_seg, segment_limit=None):
        """
        Computes the maximum compression possible for a list of sequences

        See compress_2D_sweep for details.
        Args:
            n_soft_sp: original number of sequences (soft sweep points)
            n_seg: number of segments in one Sequence
            segment_limit: maximum allowed number of segments per Sequence

        Returns:
            seg_lim_eff: number of segments in one compressed Sequence
            factor: compression factor (size of a compressed Sequence / size of
                an uncompressed Sequence, which is >= 1)
        """
        from pycqed.utilities.math import factors
        if segment_limit is None:
            segment_limit = np.inf

        # compute possible compression factors
        compression_fact = np.sort(factors(n_soft_sp))[::-1]

        for factor in compression_fact:
            if factor * n_seg > segment_limit:
                # too many segments in sequence, check for smaller factors
                continue
            elif factor == 1:
                # no compression possible
                log.warning(f'No compression possible: \n'
                      f'segments per sequence: \t\t{n_seg} \n'
                      f'limit of segments per sequence:\t{segment_limit}\n'
                      f'number of sequences: \t\t{n_soft_sp}\n'
                      f'To enable a compression, change the '
                      f'limit of segments to {compression_fact[-2] * n_seg} '
                      f'or the number of sequences  to x such that x has a '
                      f'factor f larger than 1 for which f * '
                      f'{n_seg} < {segment_limit}, e.g. x = '
                      f'{np.floor(segment_limit / n_seg)} (full compression)')
            break
        seg_lim_eff = factor * n_seg
        return seg_lim_eff, factor

    def rename(self, new_name):
        self.name = new_name
        self.timer.name = new_name

    def __repr__(self):
        string_repr = f"####### {self.name} #######\n"
        for seg_name, seg in self.segments.items():
            string_repr += str(seg) + "\n"
        return string_repr
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new_seq = cls.__new__(cls)
        memo[id(self)] = new_seq
        for k, v in self.__dict__.items():
            if k == "pulsar": # the reference to pulsar cannot be deepcopied
                setattr(new_seq, k, v)
            else:
                setattr(new_seq, k, deepcopy(v, memo))
        return new_seq

    def __getitem__(self, i):
        """
        Return a segment by its name or index, or a list of segments
        corresponding to a slice.
        :param i: (str, int, slice, list) Name, index, slice,
            or list of names/indices identifying the segment(s).
        :return: segment or list of segments
        """
        if isinstance(i, list):
            return [self[i] for i in i]
        elif isinstance(i, str):
            if i not in self.segments:
                raise KeyError(f'No segment with name "{i}" in the sequence '
                               f'{self.name}.')
            return self.segments[i]
        else:
            try:
                return list(self.segments.values())[i]
            except IndexError:
                raise IndexError(
                    f'Segment index {i} out of range. The sequence '
                    f'{self.name} has {len(self.segments)} segments.')

    def keys(self):
        """
        Returns the segments names (keys to access segments via __getitem__).
        :return: a set-like object providing the keys
        """
        return self.segments.keys()

    def plot(self, segments=None, show_and_close=True, **segment_plot_kwargs):
        """
        :param segments: list of segment names to plot
        :param show_and_close: (bool) show and close the plots (default: True)
        :param segment_plot_kwargs:
        :return: A list of tuples of figure and axes objects if show_and_close
            is False, otherwise no return value.
        """
        plots = []
        if segments is None:
            segments = self.segments.values()
        else:
            segments = [self.segments[s] for s in segments]
        for s in segments:
            plots.append(s.plot(show_and_close=show_and_close,
                                **segment_plot_kwargs))
        if show_and_close:
            return
        else:
            return plots
