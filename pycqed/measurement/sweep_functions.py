import logging
import time
import numpy as np
from pycqed.measurement import mc_parameter_wrapper
import qcodes


class Sweep_function(object):
    '''
    sweep_functions class for MeasurementControl(Instrument)
    '''

    def __init__(self, **kw):
        self.set_kw()

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass

    # note that set_paramter is only actively used in soft sweeps.
    # it is added here so that performing a "hard 2D" experiment
    # (see tests for MC) the missing set_parameter in the hard sweep does not
    # lead to unwanted errors
    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        pass

    def configure_upload(self, upload=True, upload_first=True,
                         start_pulsar=True):
        """Try to configure the sequence upload. Returns True if successful.

        This method on the highest level of dependencies allows for recursive
        search. Child classes that require special handling of this task need
        to implement their own configure_upload method.

        Args:
            upload (bool, optional): Determines whether the sweep function will
                upload in general. Defaults to True.
            upload_first (bool, optional): Determines if method prepare should
                upload the first sequence. Is governed by upload parameter.
                Defaults to True.
            start_pulsar (bool, optional): Whether to start pulsar in prepare.
                Defaults to True.

        Returns:
            bool: Whether the configuration was successful or not.
        """
        if hasattr(self, 'sweep_function'):
            return self.sweep_function.configure_upload(upload, upload_first,
                                                        start_pulsar)
        return False


class UploadingSweepFunctionMixin:
    def __init__(self, sequence=None, upload=True, upload_first=True,
                 start_pulsar=False, start_exclude_awgs=tuple(), **kw):
        """A mixin to extend any sweep function to be able to upload sequences.

        Args:
            sequence (:class:`~pycqed.measurement.waveform_control.sequence.Sequence`):
                Sequence of segments to sweep over.
            upload (bool, optional):
                Whether to upload the sequences before measurement.
                Defaults to True.
            start_pulsar (bool, optional):
                Whether (a sub set of) the used AWGs will be started directly
                after upload. This can be used, e.g., to start AWGs that have
                only one unique segment and do not need to be synchronized to
                other AWGs and therefore do not need to be stopped when
                switching to the next segment in the sweep. Defaults to False.
            start_exclude_awgs (collection[str], optional):
                A collection of AWG names that will not be started directly
                after upload in case start_pulsar is True. Defaults to empty
                tuple.
        """
        super().__init__(**kw)
        self.sequence = sequence
        self.upload = upload
        self.upload_first = upload_first
        self.start_pulsar = start_pulsar
        self.start_exclude_awgs = start_exclude_awgs

    def prepare(self, **kw):
        """Takes care of uploading the first sequence and starting the pulsar.

        Behaviour depends on self.upload_first and self.start_pulsar.

        Raises:
            ValueError: Raised if start_pulsar but sequence is None.
        """
        if self.upload_first:
            self.upload_sequence()
        if self.start_pulsar:
            if self.sequence is not None:
                self.sequence.pulsar.start(exclude=self.start_exclude_awgs)
            else:
                raise ValueError('Cannot start pulsar with sequence being None')

    def upload_sequence(self, force_upload=False):
        """Wrapper around meth:sequence.upload to ensure correct behaviour
        depending on self.upload

        Args:
            force_upload (bool, optional): Overwrites self.upload.
                Defaults to False.

        Raises:
            ValueError: Raised sequence is None.
        """
        if self.sequence is not None:
            raise ValueError('Cannot start pulsar with sequence being None')
        if self.upload or force_upload:
            self.sequence.upload()

    def configure_upload(self, upload=True, upload_first=True,
                        start_pulsar=True):
        """Overwrites parent method
        :meth:`~pycqed.measurement.sweep_function.Sweep_function.configure_upload`
        and sets the correspoding attributes.

        Args:
            upload (bool, optional): Defaults to True.
            upload_first (bool, optional): Defaults to True.
            start_pulsar (bool, optional): Defaults to True.

        Returns:
            bool: True, because UploadingSweepFunctionMixin can upload sequences.
        """
        self.upload = upload
        self.upload_first = upload_first
        self.start_pulsar = start_pulsar
        return True


class Soft_Sweep(Sweep_function):
    def __init__(self, **kw):
        self.set_kw()
        self.sweep_control = 'soft'


##############################################################################


class Elapsed_Time_Sweep(Soft_Sweep):
    """
    A sweep function to do a measurement periodically.
    Set the sweep points to the times at which you want to probe the
    detector function.
    """

    def __init__(self,
                 sweep_control='soft',
                 as_fast_as_possible: bool = False,
                 **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'Elapsed_Time_Sweep'
        self.parameter_name = 'Time'
        self.unit = 's'
        self.as_fast_as_possible = as_fast_as_possible
        self.time_first_set = None

    def set_parameter(self, val):
        if self.time_first_set is None:
            self.time_first_set = time.time()
            return 0
        elapsed_time = time.time() - self.time_first_set
        if self.as_fast_as_possible:
            return elapsed_time

        if elapsed_time > val:
            logging.warning(
                'Elapsed time {:.2f}s larger than desired {:2f}s'.format(
                    elapsed_time, val))
            return elapsed_time

        while (time.time() - self.time_first_set) < val:
            pass  # wait
        elapsed_time = time.time() - self.time_first_set
        return elapsed_time


class None_Sweep(Soft_Sweep):
    def __init__(self,
                 sweep_control='soft',
                 sweep_points=None,
                 name: str = 'None_Sweep',
                 parameter_name: str = 'pts',
                 unit: str = 'arb. unit',
                 **kw):
        super(None_Sweep, self).__init__()
        self.sweep_control = sweep_control
        self.name = name
        self.parameter_name = parameter_name
        self.unit = unit
        self.sweep_points = sweep_points

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be swept. Differs per sweep function
        '''
        pass


class None_Sweep_idx(None_Sweep):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.num_calls = 0

    def set_parameter(self, val):
        self.num_calls += 1


class Delayed_None_Sweep(Soft_Sweep):
    def __init__(self, sweep_control='soft', delay=0, mode='cycle_delay',
                 **kw):
        super().__init__()
        self.sweep_control = sweep_control
        self.name = 'None_Sweep'
        self.parameter_name = 'pts'
        self.unit = 'arb. unit'
        self.delay = delay
        self.time_last_set = 0
        self.mode = mode
        if delay > 60:
            logging.warning(
                'setting a delay of {:.g}s are you sure?'.format(delay))

    def set_parameter(self, val):
        '''
        Set the parameter(s) to be sweeped. Differs per sweep function
        '''
        if self.mode != 'cycle_delay':
            self.time_last_set = time.time()
        while (time.time() - self.time_last_set) < self.delay:
            pass  # wait
        if self.mode == 'cycle_delay':
            self.time_last_set = time.time()


###############################################################################
####################          Hardware Sweeps      ############################
###############################################################################


class Hard_Sweep(Sweep_function):
    def __init__(self, **kw):
        super().__init__()
        self.sweep_control = 'hard'
        self.parameter_name = 'None'
        self.name = 'Hard_Sweep'
        self.unit = 'a.u.'

    def start_acquistion(self):
        pass

    def set_parameter(self, value):
        logging.warning('set_parameter called for hardware sweep.')


class Segment_Sweep(Hard_Sweep):
    def __init__(self, **kw):
        super().__init__()
        self.parameter_name = 'Segment index'
        self.name = 'Segment_Sweep'
        self.unit = ''

class multi_sweep_function(Soft_Sweep):
    '''
    cascades several sweep functions into a single joint sweep functions.
    '''

    def __init__(self,
                 sweep_functions: list,
                 parameter_name=None,
                 name=None,
                 **kw):
        self.set_kw()
        self.sweep_control = 'soft'
        self.name = name or 'multi_sweep'
        self.parameter_name = parameter_name or 'multiple_parameters'
        self.sweep_functions = []
        self.add_sweep_functions(sweep_functions)

    def set_unit(self, unit=None):
        """Set self.unit either from self.sweep_functions or passed string.

        If the parameter unit is not specified and self.sweep_functions is empty
        the unit will be set to an ampty string ''.

        Args:
            unit (String, optional): Manually specified unit. Defaults to None.
        """
        if unit is not None:
            self.unit = unit
        elif len(self.sweep_functions) != 0:
            self.unit = self.sweep_functions[0].unit
        else:
            self.unit = ''

    def check_units(self):
        """Checks that all sweep functions inside self.sweep_functions have
        the same unit as specified in self.unit.

        Raises:
            ValueError: Thrown if self.sweep_functions contains a
                sweep_function with a unit different from self.unit
        """
        for i, sweep_function in enumerate(self.sweep_functions):
            if self.unit.lower() != sweep_function.unit.lower():
                raise ValueError(f'Unit {sweep_function.unit} of the'
                    f' sweepfunction {sweep_function.name} is not equal to the'
                    f' multi sweep unit {self.unit}. All sweep functions in a'
                    f' multi_sweep_function have to have the same unit.')

    def set_parameter(self, val):
        for sweep_function in self.sweep_functions:
            sweep_function.set_parameter(val)

    def add_sweep_functions(self, sweep_functions: list):
        """Adds the passed sweep_functions to the multi_sweep.

        Also takes care of changing the unit of self and calls self.check_units
        to make sure all sweep_fucntions are compatible.

        Args:
            sweep_functions (list): List of Sweep_function objects that will
                be added to the multi_sweep_function. The list might also
                contain qcodes.Parameter objects, these will be converted to
                Sweep_functions before they are added to the
                multi_sweep_function.
        """
        self.sweep_functions += [mc_parameter_wrapper.wrap_par_to_swf(s)
                                 if isinstance(s, qcodes.Parameter) else s
                                 for s in sweep_functions]
        self.set_unit()
        self.check_units()

    def add_sweep_function(self, sweep_function):
        """Adds a single sweep_function to the multi_sweep_function

        Implemented as simple wrapper around add_sweep_functions. See docstring
        of add_sweep_functions for details.
        """
        return self.add_sweep_functions([sweep_function])

    def insert_sweep_function(self, pos, sweep_function):
        """Inserts sweep_function at specified position to self.sweep_fucntions.

        Args:
            pos (int): Index of the position the sweep_function is inserted.
            sweep_function (Sweep_function or qcodes.Parameter): Sweep_function
                that is inserted into the self.sweep_functions list. If a
                qcodes.Parameter object is passed it will be converted to a
                Sweep_function beforehand.
        """
        if isinstance(sweep_function, qcodes.Parameter):
            sweep_function = mc_parameter_wrapper.wrap_par_to_swf(
                sweep_function
            )
        self.sweep_functions.insert(pos, sweep_function)
        self.set_unit()
        self.check_units()

    def prepare(self, **kw):
        for sweep_function in self.sweep_functions:
            sweep_function.prepare(**kw)

    def configure_upload(self, upload=True, upload_first=True,
                        start_pulsar=True):
        for sweep_function in self.sweep_functions:
            if sweep_function.configure_upload(upload, upload_first,
                                               start_pulsar):
                return True
        return False


class two_par_joint_sweep(Soft_Sweep):
    """
    Allows jointly sweeping two parameters while preserving their
    respective ratios.
    Allows par_A and par_B to be arrays of parameters
    """

    def __init__(self, par_A, par_B, preserve_ratio: bool = True, **kw):
        self.set_kw()
        self.unit = par_A.unit
        self.sweep_control = 'soft'
        self.par_A = par_A
        self.par_B = par_B
        self.name = par_A.name
        self.parameter_name = par_A.name
        if preserve_ratio:
            try:
                self.par_ratio = self.par_B.get() / self.par_A.get()
            except:
                self.par_ratio = (
                    self.par_B.get_latest() / self.par_A.get_latest())
        else:
            self.par_ratio = 1

    def set_parameter(self, val):
        self.par_A.set(val)
        self.par_B.set(val * self.par_ratio)


class Transformed_Sweep(Soft_Sweep):
    """
    A soft sweep function that calls another sweep function with a
    transformation applied.
    """

    def __init__(self,
                 sweep_function,
                 transformation,
                 name=None,
                 parameter_name=None,
                 unit=None):
        super().__init__()
        if isinstance(sweep_function, qcodes.Parameter):
            sweep_function = mc_parameter_wrapper.wrap_par_to_swf(
                sweep_function)
        if sweep_function.sweep_control != 'soft':
            raise ValueError(f'{self.__class__.__name__}: Only software '
                             f'sweeps supported')
        self.sweep_function = sweep_function
        self.transformation = transformation
        self.sweep_control = sweep_function.sweep_control
        self.name = self.sweep_function.name if name is None else name
        self.unit = self.sweep_function.unit if unit is None else unit
        self.parameter_name = self.default_param_name() \
            if parameter_name is None else parameter_name

    def default_param_name(self):
        return f'transformation of {self.sweep_function.parameter_name}'

    def prepare(self, *args, **kwargs):
        self.sweep_function.prepare(*args, **kwargs)

    def finish(self, *args, **kwargs):
        self.sweep_function.finish(*args, **kwargs)

    def set_parameter(self, val):
        self.sweep_function.set_parameter(self.transformation(val))


class Offset_Sweep(Transformed_Sweep):
    """
    A soft sweep function that calls another sweep function with an offset.
    """

    def __init__(self,
                 sweep_function,
                 offset,
                 name=None,
                 parameter_name=None,
                 unit=None):

        self.offset = offset
        super().__init__(sweep_function,
                 transformation=lambda x, o=offset : x + o,
                 name=name, parameter_name=parameter_name, unit=unit)

    def default_param_name(self):
        return self.sweep_function.parameter_name + \
               ' {:+} {}'.format(-self.offset, self.sweep_function.unit)


class Indexed_Sweep(Transformed_Sweep):
    """
    A soft sweep function that calls another sweep function with parameter
    values taken from a provided list of values.
    """

    def __init__(self, sweep_function, values, name=None, parameter_name=None,
                 unit=''):
        self.values = values
        super().__init__(sweep_function,
                 transformation=lambda i, v=self.values : v[i],
                 name=name, parameter_name=parameter_name, unit=unit)

    def default_param_name(self):
        return f'index of {self.sweep_function.parameter_name}'


class MajorMinorSweep(Soft_Sweep):
    """
    A soft sweep function that combines two sweep function such that the
    major sweep function takes only discrete values from a given set while
    the minor sweep function takes care of the difference between the
    discrete values and the desired sweep values.

    (further parameters as in multi_sweep_function)
    :param major_sweep_function: (obj) a soft sweep function or QCoDeS
        parameter to perform large steps of the sweep parameter
    :param minor_sweep_function: (obj) a soft sweep function or QCoDeS
        parameter to perform small steps of the sweep parameter
    :param major_values: (array, list) allowed values of the
        major_sweep_function
    """

    def __init__(self,
                 major_sweep_function,
                 minor_sweep_function,
                 major_values,
                 name=None,
                 parameter_name=None,
                 unit=None):
        super().__init__()

        self.major_sweep_function = \
            mc_parameter_wrapper.wrap_par_to_swf(major_sweep_function) \
                if isinstance(major_sweep_function, qcodes.Parameter) \
                else major_sweep_function
        self.minor_sweep_function = \
            mc_parameter_wrapper.wrap_par_to_swf(minor_sweep_function) \
                if isinstance(minor_sweep_function, qcodes.Parameter) \
                else minor_sweep_function
        if self.major_sweep_function.sweep_control != 'soft' or \
                self.minor_sweep_function.sweep_control != 'soft':
            raise ValueError('Offset_Sweep: Only software sweeps supported')
        self.sweep_control = 'soft'
        self.major_values = np.array(major_values)
        self.parameter_name = self.major_sweep_function.parameter_name \
            if parameter_name is None else parameter_name
        self.name = self.major_sweep_function.name if name is None else name
        self.unit = self.major_sweep_function.unit if unit is None else unit

    def prepare(self, *args, **kwargs):
        self.major_sweep_function.prepare(*args, **kwargs)
        self.minor_sweep_function.prepare(*args, **kwargs)

    def finish(self, *args, **kwargs):
        self.major_sweep_function.finish(*args, **kwargs)
        self.minor_sweep_function.finish(*args, **kwargs)

    def set_parameter(self, val):
        # find the closes allowed value of the major_sweep_function
        ind = np.argmin(np.abs(self.major_values - val))
        mval = self.major_values[ind]
        self.major_sweep_function.set_parameter(mval)
        # use the minor_sweep_function to bridge the difference to the
        # target value
        self.minor_sweep_function.set_parameter(val - mval)


class FilteredSweep(multi_sweep_function):
    """
    Records only a specified consecutive subset of segments of a
    SegmentHardSweep for each soft sweep point while performing the soft
    sweep defined in sweep_functions.

    (further parameters as in multi_sweep_function)
    :param sequence: The Sequence programmed to the AWGs.
    :param filter_lookup: (dict) A dictionary where each key is a soft sweep
        point and the corresponding value is a tuple of indices
        indicating the first and the last segment to be measured. (Segments
        with the property allow_filter set to False are always measured.)
    """
    def __init__(self,
                 sequence,
                 filter_lookup,
                 sweep_functions: list,
                 parameter_name=None,
                 name=None,
                 **kw):
        self.sequence = sequence
        self.allow_filter = [seg.allow_filter for seg in
                             sequence.segments.values()]
        self.filter_lookup = filter_lookup
        self.filtered_sweep = None
        super().__init__(sweep_functions, parameter_name, name, **kw)

    def set_parameter(self, val):
        # Set the soft sweep parameter in the sweep_functions. This is done
        # before updating filter_segments in pulsar. In case the following
        # set_parameter triggers a reprogramming, pulsar will skip the
        # programming of AWGs for which filter segments emulation is needed,
        # and those AWGs will be programmed during the following update of
        # filter_segments, so that only the needed segments of the new
        # sequence get programmed.
        super().set_parameter(val)
        # Determine the current segment filter and inform Pulsar.
        filter_segments = self.filter_lookup[val]
        self.sequence.pulsar.filter_segments(filter_segments)
        # The filtered_sweep property stores a mask indicating which
        # acquisition elements are recorded (will be accessed by MC to
        # handle the acquired data correctly).
        seg_mask = np.logical_not(self.allow_filter)
        seg_mask[filter_segments[0]:filter_segments[1] + 1] = True
        acqs = self.sequence.n_acq_elements(per_segment=True)
        self.filtered_sweep = [m for m, a in zip(seg_mask, acqs) for i in
                               range(a)]
