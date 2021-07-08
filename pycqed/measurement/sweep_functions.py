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
        self.sweep_functions = [mc_parameter_wrapper.wrap_par_to_swf(s)
                                if isinstance(s, qcodes.Parameter) else s
                                for s in sweep_functions]
        self.sweep_control = 'soft'
        self.name = name or 'multi_sweep'
        self.unit = sweep_functions[0].unit
        self.parameter_name = parameter_name or 'multiple_parameters'
        for i, sweep_function in enumerate(sweep_functions):
            if self.unit.lower() != sweep_function.unit.lower():
                raise ValueError('units of the sweepfunctions are not equal')

    def set_parameter(self, val):
        for sweep_function in self.sweep_functions:
            sweep_function.set_parameter(val)


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
    """A soft sweep function that combines two sweep function such that the
    major sweep function takes only discrete values from a given set while
    the minor sweep function takes care of the difference between the
    discrete values and the desired sweep values."""

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
        ind = np.argmin(np.abs(self.major_values - val))
        mval = self.major_values[ind]
        self.major_sweep_function.set_parameter(mval)
        self.minor_sweep_function.set_parameter(val - mval)


class FilteredSweep(multi_sweep_function):
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
        filter_segments = self.filter_lookup[val]
        self.sequence.pulsar.filter_segments(filter_segments)
        seg_mask = np.invert(self.allow_filter)
        seg_mask[filter_segments[0]:filter_segments[1] + 1] = True
        acqs = self.sequence.n_acq_elements(per_segment=True)
        self.filtered_sweep = [m for m, a in zip(seg_mask, acqs) for i in
                               range(a)]
        super().set_parameter(val)
