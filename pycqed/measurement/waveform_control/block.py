import logging
import numpy as np
from copy import deepcopy

log = logging.getLogger(__name__)

class Block:
    """
    A block is a building block for a Quantum Algorithm Experiment.

            :param block_start_position: position of the block start. Defaults to
            before first pulse in list. Position should be changed in case the
            first pulse played is not the first one in the list.
        :param block_end_position: position of the block end. Defaults to after
            last pulse. If given, block_end_position MUST take into account the
            block_start i.e. the block start is added to the pulses before inserting
            the block end at its defined location.
    """
    counter = 0
    INSIDE_BLOCKINFO_NAME = "BlockInfo"

    def __init__(self, block_name, pulse_list:list, pulse_modifs=None,
                 copy_pulses=True, **kw):
        self.name = block_name
        self.block_start = kw.get('block_start', {})
        self.block_end = kw.get('block_end', {})
        self.destroyed = False
        if pulse_modifs is not None and len(pulse_modifs):
            self.pulses = pulse_list  # will be copied in sweep_copy
            self.pulses = self.pulses_sweepcopy([pulse_modifs], [None])
        else:
            self.pulses = deepcopy(pulse_list) if copy_pulses else pulse_list

    def build(self, ref_point="end", ref_point_new="start",
              ref_pulse='previous_pulse', block_delay=0, name=None,
              sweep_dicts_list=None, sweep_index_list=None, destroy=False,
               **kwargs):
        """
        Adds the block shell recursively through the pulse list.
        Returns the flattened pulse list
        :param ref_point: to which point of the previous pulse should this block
        be referenced to. Defaults to "end".
        :param ref_point_new: which point of this block is reference to the
            ref_point. Currently only supports "start".
        :param ref_pulse: to which pulse in the list is this block referenced to.
            Defaults to "previous_pulse" in list.
        :param block_delay: delay before the start of the block
        :param name: a custom name used to prefix the pulses. If None,
            the block name plus a counter is used.
        :param sweep_dicts_list: To build a block that corresponds to a
            point of an N-dimensional sweep, this param is a list of N
            sweep_dicts following the usual pycqed conventions (TODO: where
            to find those?)
            or a SweepPoints object with N dimensions. Only used if also a
            sweep_index_list is provided. To have an effect, the block has
            to contain pulses with ParametricValues that refer to the
            parameters in the sweep_dicts.
            In addition, parameters of pulses can be modified by pulse modifier
            entries in the sweep dictionaries with the following format:

            'attr=X, key_1=val_1, ..., key_N=val_N'
            Searches all pulses p for which p[key_n] == val_n for all n,
            and sweeps the attribute attr of these pulses.
            If key_n is op_code, it suffices if the initial segment(s) of
            the op_code match (e.g., 'X180' will also match 'X180 qb1').
            If key_n is name or ref_pulse, it suffices if the original
            (ref)pulse name (before a potential change in a previous build())
            matches.

            'attr=X, key_1=val_1, ..., key_N=val_N, occurrence=i'
            Sweep the attribute only for the ith pulse matching the
            criteria, where i is an integer (zero-indexed)

        :param sweep_index_list: A list of N indices for an N-dimensional
            sweep. Determines for which sweep points from sweep_dicts_list
            the block should be build. Only used if also a sweep_dicts_list
            is provided.
        :param destroy: (bool) can be set to True for faster processing
            (avoid copying) if the block will not be needed anymore afterwards.
        :return:
        """
        if self.destroyed:
            raise ValueError('This block has already been destroyed.')
        if destroy:
            self.destroyed = True
        if ref_point_new != "start":
            raise NotImplementedError("For now can only refer blocks to 'start'")
        if name is None:
            name = self.name + (f"_{self.counter}" if self.counter > 0 else "")
            self.counter += 1

        block_start = {"name": f"start",
                       "pulse_type": "VirtualPulse",
                       "pulse_delay": block_delay,
                       "ref_pulse": ref_pulse,
                       "ref_point": ref_point}
        block_start.update(kwargs.get("block_start", self.block_start))
        block_end = {"name": f"end",
                     "pulse_type": "VirtualPulse"}
        block_end.update(kwargs.get("block_end", self.block_end))
        if sweep_dicts_list is not None and sweep_index_list is not None:
            pulses_built = self.pulses_sweepcopy(sweep_dicts_list, sweep_index_list)
        elif destroy:
            pulses_built = self.pulses
        else:
            pulses_built = deepcopy(self.pulses)

        # check if block_start/end  specified by user
        block_start_specified = False
        block_end_specified = False
        for p in pulses_built:
            if p.get("name", None) == "start":
                block_start = p #save reference
                block_start_specified = True
            elif p.get("name", None) == "end":
                block_end = p
                block_end_specified = True
        # add them if not specified
        if not block_start_specified:
            pulses_built = [block_start] + pulses_built
        if not block_end_specified:
            pulses_built = pulses_built + [block_end]

        for p in pulses_built:
            # if a dictionary wrapping a block is found, compile the inner block.
            if p.get("pulse_type", None) == self.INSIDE_BLOCKINFO_NAME:
                # p needs to have a block key
                assert 'block' in p, f"Inside block {p.get('name', 'Block')} " \
                    f"requires a key 'block' which refers to the uncompiled " \
                    f"block object."
                inside_block = p.pop('block')
                inside_block_pulses = inside_block.build(**p)
                # add all pulses of the inside block to the outer block
                pulses_built.extend(inside_block_pulses)

        # prepend block name to reference pulses and pulses names
        for p in pulses_built:
            # if the pulse has a name, prepend the blockname to it
            if p.get("name", None) is not None:
                p['name'] = name + "-|-" + p['name']

            ref_pulse = p.get("ref_pulse", "previous_pulse")
            p_is_block_start = self._is_block_start(p, block_start)

            # rename ref pulse within the block if not a special name
            escape_names = ("previous_pulse", "segment_start", "init_start")
            if isinstance(ref_pulse, list):
                p['ref_pulse'] = [name + "-|-" + rp for rp in p['ref_pulse']]
            else:
                if ref_pulse not in escape_names and not p_is_block_start:
                    p['ref_pulse'] = name + "-|-" + p['ref_pulse']

        return pulses_built

    def set_end_after_all_pulses(self, **block_end):
        if len(self.pulses):
            for i, p in enumerate(self.pulses):
                p['name'] = p.get('name', f"pulse_{i}")
            self.block_end.update({
                'ref_function': 'max',
                'ref_pulse': [p['name'] for p in self.pulses],
                'ref_point': 'end',
            })
        self.block_end.update(block_end)

    def _is_shell(self, pulse, block_start, block_end):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_start (dict): dictionary of the block start
            block_end (dict): dictionary of the block end

        Returns: whether pulse is a shell dictionary (bool)

        """
        return self._is_block_start(pulse, block_start) \
               or self._is_block_end(pulse, block_end)

    def _is_block_start(self, pulse, block_start):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_start (dict): dictionary of the block start

        Returns: whether pulse is a the block start dictionary (bool)

        """
        return pulse.get('name', None) == block_start['name']


    def _is_block_end(self, pulse, block_end):
        """
        Checks, based on the pulse name, whether a pulse belongs to the block shell.
        That is, if the pulse name is the same as the name of the block start or end.
        A simple equivalence p == block_start or p == p_end does not work as pulse
        could be a deepcopy of block_start, which would return False in the above
        expressions.
        Args:
            pulse (dict): pulse to check.
            block_end (dict): dictionary of the block end

        Returns: whether pulse is a the block end dictionary (bool)

        """
        return pulse.get('name', None) == block_end['name']


    def extend(self, additional_pulses):
        self.pulses.extend(additional_pulses)

    def __add__(self, other_block):
        return Block(f"{self.name}_{other_block.name}",
                     self.pulses + other_block.pulses)

    def __len__(self):
        return len(self.pulses)

    def __repr__(self):
        string_repr = f"---- {self.name} ----\n"
        for i, p in enumerate(self.pulses):
            string_repr += f"{i}: " + repr(p) + "\n"
        return string_repr

    def pulses_sweepcopy(self, sweep_dicts_list, index_list):
        """
        Returns a deepcopy of the pulse list where, based on the provided
        sweep_dicts_list and index_list, ParametricValue() objects
        in all pulses are resolved and further pulse modifiers are applied.

        :param sweep_dicts_list: see description of build()
        :param sweep_index_list: see description of build()

        :return:
        """
        if isinstance(index_list, int):
            index_list = [index_list]
        if isinstance(sweep_dicts_list, dict):
            sweep_dicts_list = [sweep_dicts_list]
        pulses = deepcopy(self.pulses)
        # resolve parametric values first
        for p in pulses:
            for attr, s in p.items():
                if getattr(s, '_is_parametric_value', False):
                    for sweep_dict, ind in zip(sweep_dicts_list, index_list):
                        if s.param in sweep_dict:
                            if 'op_code' in p:
                                p[attr], p['op_code'] = s.resolve(
                                    sweep_dict, ind, p['op_code'])
                            else:
                                # allows to add ParametricValues to block_start
                                # or block_end. See for ex the T1 class
                                p[attr] = s.resolve(sweep_dict, ind)

        # resolve pulse modifiers now (they could overwrite parametric values)
        for sweep_dict, ind in zip(sweep_dicts_list, index_list):
            for param, d in sweep_dict.items():
                if (pattern := parse_pulse_search_pattern(param)) is None:
                    continue
                attr = pattern.pop('attr', None)
                occ_counter = [0]
                for p in pulses:
                    if check_pulse_by_pattern(p, pattern, occ_counter):
                        if attr is None:
                            p.update(d)
                        else:
                            p.update({attr: ParametricValue(
                                param).resolve(sweep_dict, ind)})
                        if pattern.get('occurrence') is not None:
                            break
        return pulses

    def prefix_parametric_values(self, prefix, params=None):
        """
        Adds a prefix to the parameter name of ParametricValues in the pulses
        of the block.

        :param prefix: (str) prefix to be added
        :param params: (optional list of str) if given, prefix only these
            parameter names
        """
        for p in self.pulses:
            for k, s in p.items():
                if getattr(s, '_is_parametric_value', False):
                    if params is None or s.param in params:
                        s.param = prefix + s.param

    def parametric_values(self):
        return {(i, attr) : s for i, p in enumerate(self.pulses)
                for attr, s in p.items()
                if getattr(s, '_is_parametric_value', False)}


class ParametricValue:
    """
    A ParametricValue can be used as a placeholder for a pulse attribute that
    will be chosen based on a parameter provided later (e.g.,
    by Block.pulses_sweepcopy).

    :param param: a string specifying the name of the parameter.
    :param func: (optional) a function applied to the value of the sweep
        parameter to yield the value of the physical parameter, e.g. amplitude
    :param func_op_code: (optional) a function applied to the value of the
        sweep parameter to yield the value to store in the op_code (gate angle)
    :param op_split: (optional) cache a splitted version of the op_code of
        the pulse to allow for correct op_code resolution in cases of spaces
        in a mathematical expression in an op_code.

    """
    _is_parametric_value = True

    def __init__(self, param, func=None, func_op_code=None, op_split=None):
        self.param = param
        self.func = func
        self.op_split = op_split
        self.func_op_code = func_op_code

    def resolve(self, sweep_dict, ind=None, op_code=None):
        """
        Returns the resolved value of a pulse attribute for a chosen sweep
        point.

        :param sweep_dict: an entry of the sweep_dicts_list described in
            build(). Alternatively, a dict of the form {param_n: val_n},
            in which case ind is ignored.
        :param ind: The index of the desired sweep point in sweep_dict.
            None is only allowed in the case where ind is ignored (see
            above).
        :param op_code: (optional str) the op_code of the pulse to allow
            resolving a parametric expression in the op_code

        :return: the resolved numerical value. If op_code was provided,
            the resolved op_code is returned in addition.
        """
        d = sweep_dict[self.param]
        if not isinstance(d, list) and not isinstance(d, dict) and not \
                isinstance(d, tuple):
            v = d
        elif isinstance(sweep_dict[self.param], dict) and 'values' in \
                sweep_dict[self.param]:  # convention in old sweep_dicts
            v = d['values'][ind]
        else: # convention in SweepPoints class
            v = d[0][ind]
        v_processed = v if self.func is None else self.func(v)
        if op_code is not None:
            if f'[{self.param}]' in op_code:
                # if there is a cached splitted version, use that one instead
                # in order to allow for correct op_code resolution in cases
                # of in a mathematical expression in an op_code.
                # Example: op_code = "Y:2*[v] qb1" -> "Y:90 qb1"
                # TODO op_split might not be needed anymore after introducing
                #  func_op_code
                # FIXME: remove op_code caching as soon as a new op_code
                #  concept (e.g. tuples instead of space-separated strings)
                #  makes it obsolete
                op_split = [s for s in self.op_split] if self.op_split is not \
                            None else op_code.split(' ')
                param_start = op_split[0].find(':')
                v_code = v if not self.func_op_code else self.func_op_code(v)
                op_split[0] = f"{op_split[0][:param_start]}{v_code}"
                op_code = ' '.join(op_split)
            else:
                op_code = op_code.replace(f':{self.param} ', f"{v} ")
            return v_processed, op_code
        else:
            return v_processed

    def _copy_self(self):
        """
        Returns a copy of self, ensuring that self.func exists

        Note that this might be inefficient, since this makes use of a
        deepcopy, and creates a lambda function which is itself wrapped in a
        lambda function in the methods below, e.g. self.__add__.
        """
        new_parametric_value = deepcopy(self)
        if new_parametric_value.func is None:
            new_parametric_value.func = lambda x: x
        return new_parametric_value

    def __add__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: f(x) + other
        return pv

    __radd__ = __add__

    def __sub__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: f(x) - other
        return pv

    def __rsub__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: other - f(x)
        return pv

    def __neg__(self):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: -f(x)
        return pv

    def __mul__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: f(x) * other
        return pv

    __rmul__ = __mul__

    def __truediv__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: f(x) / other
        return pv

    def __rtruediv__(self, other):
        pv = self._copy_self()
        pv.func = lambda x, f=pv.func: other / f(x)
        return pv

    def __repr__(self):
        return super().__repr__() + f"({self.param})"


def parse_pulse_search_pattern(pattern):
    """Parse strings/ints that represent a pulse search pattern

    Args:
        pattern (str, int): the representation of the search pattern can be:
          - a string as described in the docstring of Block.build,
            param sweep_dicts_list
          - an int i, which will be interpreted as the string
            f'occurrence={i}'
          - the str 'all' for matching all pulses

    Returns:
        None if pattern is not a valid search pattern. Otherwise a dict,
        where each key-value pair corresponds to a criterion that needs to
        be fulfilled, as described in the docstring of Block.build,
        param sweep_dicts_list.
    """
    if isinstance(pattern, int):
        pattern = f'occurrence={pattern}'
    if pattern == 'all':
        return {}
    elif '=' not in pattern:
        return None
    else:
        return {l[0]: l[1] for l in [s.strip().split('=') for s
                                     in pattern.split(',')]}


def check_pulse_by_pattern(pulse, pattern, occurrence_counter):
    """Check whether a pulse matches a search pattern

    Args:
        pulse: the pulse to be checked
        pattern: a search pattern, provided either in the input format or in
            the return format of parse_pulse_search_pattern
        occurrence_counter: a single-entry list of int to count how many
            matches to the pattern have already been found in calls to this
            function. Note that occurence_counter is mutable and gets
            modified by this function. (The int is wrapped in a list
            in order to have it mutable.)

    Returns:
        a bool indicating whether the pulse matches the criteria in pattern
    """
    def check_candidate(k, v, p):
        attr = p.get(k, '')
        if k == 'op_code':
            return (attr + ' ').startswith(v + ' ')
        elif k in ['name', 'ref_pulse']:
            # make sure to also find pulses renamed by Block.build()
            return (attr == v or attr.endswith("-|-" + v))
        else:
            return (attr == v)

    if isinstance(pattern, (str, int)):
        pattern = parse_pulse_search_pattern(pattern)
    if pattern is None:
        return False

    occurrence = pattern.get('occurrence')
    pattern = {k: v for k, v in pattern.items() if k != 'occurrence'}
    if all([check_candidate(k, v, pulse) for k, v in pattern.items()]):
        occurrence_counter[0] += 1
        if occurrence is None or (
                int(occurrence) == occurrence_counter[0] - 1):
            return True
    return False

