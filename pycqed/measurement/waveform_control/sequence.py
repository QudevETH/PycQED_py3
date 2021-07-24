# A Sequence contains segments which then contain the pulses. The Sequence
# provides the information for the AWGs, in which order to play the segments.
#
# author: Michael Kerschbaum
# created: 04/2019

import numpy as np
import pycqed.measurement.waveform_control.pulsar as ps
from collections import OrderedDict as odict
from copy import deepcopy
import logging

from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)

class Sequence:
    """
    A Sequence consists of several segments, which can be played back on the 
    AWGs sequentially.
    """

    RENAMING_SEPARATOR = "+"

    def __init__(self, name, segments=()):
        """
        Initializes a Sequence object
        Args:
            name: Name of the sequence
            segments (list, tuple): list of segments to add to the sequence
        """
        self.name = name
        self.pulsar = ps.Pulsar.get_instance()
        self.segments = odict()
        self.awg_sequence = {}
        self.repeat_patterns = {}
        self.extend(segments)
        self.timer = Timer(self.name)

    def add(self, segment):
        if segment.name in self.segments:
            raise NameError('Name {} already exisits in the sequence!'.format(
                segment.name))
        self.segments[segment.name] = segment

    def extend(self, segments):
        """
        Extends the sequence given a list of segments
        Args:
            segments (list): segments to add to the sequence
        """
        for seg in segments:
            self.add(seg)

    @Timer()
    def generate_waveforms_sequences(self, awgs=None):
        """
        Calculates and returns 
            * a dictionary of waveforms used in the sequence, indexed
                by their hash value
            * For each awg, a list of elements, each element consisting of
                a waveform-hash for each codeword and each channel
        """
        waveforms = {}
        sequences = {}
        for seg in self.segments.values():
            seg.resolve_segment()
            seg.gen_elements_on_awg()

        if awgs is None:
            awgs = set()
            for seg in self.segments.values():
                awgs |= set(seg.elements_on_awg)

        for awg in awgs:
            sequences[awg] = odict()
            for segname, seg in self.segments.items():
                # Store the name of the segment
                sequences[awg][segname] = None
                for elname in seg.elements_on_awg.get(awg, []):
                    sequences[awg][elname] = {'metadata': {}}
                    for cw in seg.get_element_codewords(elname, awg=awg):
                        sequences[awg][elname][cw] = {}
                        for ch in seg.get_element_channels(elname, awg=awg):
                            h = seg.calculate_hash(elname, cw, ch)
                            chid = self.pulsar.get(f'{ch}_id')
                            sequences[awg][elname][cw][chid] = h
                            if h not in waveforms:
                                wf = seg.waveforms(awgs={awg}, 
                                    elements={elname}, channels={ch}, 
                                    codewords={cw})
                                waveforms[h] = wf.popitem()[1].popitem()[1]\
                                                 .popitem()[1].popitem()[1]
                    if elname in seg.acquisition_elements:
                        sequences[awg][elname]['metadata']['acq'] = True
                    else:
                        sequences[awg][elname]['metadata']['acq'] = False
        return waveforms, sequences
                
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
        repeat = dict()
        for ch in pulse_channel_names:
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
        sequences = [deepcopy(s) for s in sequences]
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
                merged_seqs[-1].name += "+" + seq.name
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
        from pycqed.utilities.math import factors
        n_soft_sp = len(sequences)
        n_seg = sequences[0].n_segments()
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