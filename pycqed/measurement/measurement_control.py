import types
import logging
log = logging.getLogger(__name__)
import time
from copy import deepcopy
import traceback

import numpy as np
import numbers
from scipy.optimize import fmin_powell
from pycqed.measurement import hdf5_data as h5d
from pycqed.utilities import general
from pycqed.utilities.general import dict_to_ordered_tuples
from pycqed.utilities.get_default_datadir import get_default_datadir

# used for axis labels
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod

# Used for auto qcodes parameter wrapping
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_det
from pycqed.analysis.tools.data_manipulation import get_generation_means

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from qcodes.plots.colors import color_cycle


try:
    import msvcrt  # used on windows to catch keyboard input
except:
    print('Could not import msvcrt (used for detecting keystrokes)')

try:
    from qcodes.plots.pyqtgraph import QtPlot
except Exception:
    print('pyqtgraph plotting not supported, '
          'try "from qcodes.plots.pyqtgraph import QtPlot" '
          'to see the full error')
    print('When instantiating an MC object,'
          ' be sure to set live_plot_enabled=False')


class MeasurementControl(Instrument):

    '''
    New version of Measurement Control that allows for adaptively determining
    data points.
    '''

    def __init__(self, name: str,
                 plotting_interval: float=3,
                 datadir: str=get_default_datadir(),
                 live_plot_enabled: bool=True, verbose: bool=True):
        super().__init__(name=name, server_name=None)

        self.add_parameter('datadir',
                           initial_value=datadir,
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        # Soft average is currently only available for "hard"
        # measurements. It does not work with adaptive measurements.
        self.add_parameter('soft_avg',
                           label='Number of soft averages',
                           parameter_class=ManualParameter,
                           vals=vals.Ints(1, int(1e8)),
                           initial_value=1)

        self.add_parameter('plotting_max_pts',
                           label='Maximum number of live plotting points',
                           parameter_class=ManualParameter,
                           vals=vals.Ints(1),
                           initial_value=4000)
        self.add_parameter('verbose',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(),
                           initial_value=verbose)
        self.add_parameter('live_plot_enabled',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(),
                           initial_value=live_plot_enabled)
        self.add_parameter('plotting_interval',
                           unit='s',
                           vals=vals.Numbers(min_value=0.001),
                           set_cmd=self._set_plotting_interval,
                           get_cmd=self._get_plotting_interval)
        self.add_parameter('persist_mode',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=True)
        self.add_parameter('skip_measurement',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)
        self.add_parameter('clean_interrupt',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=False)

        self.add_parameter(
            'cfg_clipping_mode', vals=vals.Bool(),
            docstring='Clipping mode, when True ignores ValueErrors  when '
            'setting parameters. This can be useful when running optimizations',
            parameter_class=ManualParameter,
            initial_value=False)

        self.add_parameter('instrument_monitor',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Strings())

        # pyqtgraph plotting process is reused for different measurements.
        if self.live_plot_enabled():
            self.main_QtPlot = QtPlot(
                window_title='Main plotmon of {}'.format(self.name),
                figsize=(600, 400))
            self.secondary_QtPlot = QtPlot(
                window_title='Secondary plotmon of {}'.format(self.name),
                figsize=(600, 400))

        self.plotting_interval(plotting_interval)

        self.soft_iteration = 0  # used as a counter for soft_avg
        self._persist_dat = None
        self._persist_xlabs = None
        self._persist_ylabs = None
        self._analysis_display = None

        self.exp_metadata = {}
        self._plotmon_axes_info = None
        self._persist_plotmon_axes_info = None

    ##############################################
    # Functions used to control the measurements #
    ##############################################

    def create_instrument_settings_file(self, label=None):
        '''
        Saves a snapshot of the current instrument settings without carrying
        out a measurement.

        :param label: (optional str) a label to be used in the filename
            (will be appended to the default label Instrument_settings)
        '''
        label = '' if label is None else '_' + label
        self.set_measurement_name('Instrument_settings' + label)
        with h5d.Data(name=self.get_measurement_name(),
                      datadir=self.datadir()) as self.data_object:
            self.save_instrument_settings(self.data_object)

    def update_sweep_points(self):
        sweep_points = self.get_sweep_points()
        if sweep_points is not None:
            self.set_sweep_points(np.tile(sweep_points,
                                          self.acq_data_len_scaling))

    def run(self, name: str=None, exp_metadata: dict=None,
            mode: str='1D', disable_snapshot_metadata: bool=False, **kw):
        '''
        Core of the Measurement control.

        Args:
            name (string):
                    Name of the measurement. This name is included in the
                    name of the data files.
            exp_metadata (dict):
                    Dictionary containing experimental metadata that is saved
                    to the data file at the location
                        file['Experimental Data']['Experimental Metadata']
            mode (str):
                    Measurement mode. Can '1D', '2D', or 'adaptive'.
            disable_snapshot_metadata (bool):
                    Disables metadata saving of the instrument snapshot.
                    This can be useful for performance reasons.
                    N.B. Do not use this unless you know what you are doing!
                    Except for special cases instrument settings should always
                    be saved in the datafile.
                    This is an argument instead of a parameter because this
                    should always be explicitly diabled in order to prevent
                    accidentally leaving it off.

        '''
        # Setting to zero at the start of every run, used in soft avg
        self.soft_iteration = 0
        self.set_measurement_name(name)
        self.print_measurement_start_msg()

        self.mode = mode
        # used in determining data writing indices (deprecated?)
        self.iteration = 0

        # used for determining data writing indices and soft averages
        self.total_nr_acquired_values = 0

        # used in get_percdone to scale the length of acquired data
        self.acq_data_len_scaling = self.detector_function.acq_data_len_scaling

        # update sweep_points based on self.acq_data_len_scaling
        self.update_sweep_points()

        # needs to be defined here because of the with statement below
        return_dict = {}
        self.last_sweep_pts = None  # used to prevent resetting same value

        if self.skip_measurement():
            return return_dict

        with h5d.Data(name=self.get_measurement_name(),
                      datadir=self.datadir()) as self.data_object:
            if exp_metadata is not None:
                self.exp_metadata = deepcopy(exp_metadata)
            else:
                # delete metadata from previous measurement
                self.exp_metadata = {}
            det_metadata = self.detector_function.generate_metadata()
            self.exp_metadata.update(det_metadata)
            self.save_exp_metadata(self.exp_metadata, self.data_object)
            try:
                self.check_keyboard_interrupt()
                self.get_measurement_begintime()
                if not disable_snapshot_metadata:
                    self.save_instrument_settings(self.data_object)
                self.create_experimentaldata_dataset()
                if mode is not 'adaptive':
                    try:
                        # required for 2D plotting and data storing.
                        # try except because some swf get the sweep points in the
                        # prepare statement. This needs a proper fix
                        self.xlen = len(self.get_sweep_points())
                    except:
                        self.xlen = 1
                if self.mode == '1D':
                    self.measure()
                elif self.mode == '2D':
                    self.measure_2D()
                elif self.mode == 'adaptive':
                    self.measure_soft_adaptive()
                else:
                    raise ValueError('Mode "{}" not recognized.'
                                     .format(self.mode))
            except KeyboardFinish as e:
                print(e)
            except KeyboardInterrupt as e:
                percentage_done = self.get_percdone()
                if percentage_done == 0 or not self.clean_interrupt():
                    raise e
                self.save_exp_metadata({'percentage_done': percentage_done},
                                       self.data_object)
                logging.warning('Caught a KeyboardInterrupt and there is '
                                'unsaved data. Trying clean exit to save '
                                'data.')
            result = self.dset[()]
            self.get_measurement_endtime()
            self.save_MC_metadata(self.data_object)  # timing labels etc
            return_dict = self.create_experiment_result_dict()

        self.finish(result)
        return return_dict

    def measure(self):
        if self.live_plot_enabled():
            self.initialize_plot_monitor()

        for sweep_function in self.sweep_functions:
            sweep_function.prepare()

        if (self.sweep_functions[0].sweep_control == 'soft' and
                self.detector_function.detector_control == 'soft'):
            self.detector_function.prepare()
            self.get_measurement_preparetime()
            self.measure_soft_static()

        elif self.detector_function.detector_control == 'hard':
            self.get_measurement_preparetime()
            sweep_points = self.get_sweep_points()

            while self.get_percdone() < 100:
                start_idx = self.get_datawriting_start_idx()
                if len(self.sweep_functions) == 1:
                    self.sweep_functions[0].set_parameter(
                        sweep_points[start_idx])
                    self.detector_function.prepare(
                        sweep_points=self.get_sweep_points())
                    self.measure_hard()
                else:  # If mode is 2D
                    for i, sweep_function in enumerate(self.sweep_functions):
                        swf_sweep_points = sweep_points[:, i]
                        val = swf_sweep_points[start_idx]
                        sweep_function.set_parameter(val)
                    self.detector_function.prepare(
                        sweep_points=sweep_points[
                            start_idx:start_idx+self.xlen, 0])
                    self.measure_hard()
        else:
            raise Exception('Sweep and Detector functions not '
                            + 'of the same type. \nAborting measurement')

        self.check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon(force_update=True)
        if self.mode == '2D' and self.detector_function.detector_control != \
                'hard':
            # If self.detector_function.detector_control == 'hard' in 2D mode,
            # the function update_plotmon_2D_hard is called inside
            # measure_hard, and we should not call update_plotmon_2D.
            self.update_plotmon_2D(force_update=True)
        elif self.mode == 'adaptive':
            self.update_plotmon_adaptive(force_update=True)
        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()

        return

    def measure_soft_static(self):
        for j in range(self.soft_avg()):
            self.soft_iteration = j
            for i, sweep_point in enumerate(self.sweep_points):
                self.measurement_function(sweep_point)

    def measure_soft_adaptive(self, method=None):
        '''
        Uses the adaptive function and keywords for that function as
        specified in self.af_pars()
        '''
        self.save_optimization_settings()
        self.adaptive_function = self.af_pars.pop('adaptive_function')
        if self.live_plot_enabled():
            self.initialize_plot_monitor()
            self.initialize_plot_monitor_adaptive()
        for sweep_function in self.sweep_functions:
            sweep_function.prepare()
        self.detector_function.prepare()
        self.get_measurement_preparetime()

        if self.adaptive_function == 'Powell':
            self.adaptive_function = fmin_powell
        if (isinstance(self.adaptive_function, types.FunctionType) or
                isinstance(self.adaptive_function, np.ufunc)):
            try:
                # exists so it is possible to extract the result
                # of an optimization post experiment
                self.adaptive_result = \
                    self.adaptive_function(self.optimization_function,
                                           **self.af_pars)
            except StopIteration:
                print('Reached f_termination: %s' % (self.f_termination))
        else:
            raise Exception('optimization function: "%s" not recognized'
                            % self.adaptive_function)
        self.save_optimization_results(self.adaptive_function,
                                       result=self.adaptive_result)

        for sweep_function in self.sweep_functions:
            sweep_function.finish()
        self.detector_function.finish()
        self.check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon(force_update=True)
        self.update_plotmon_adaptive(force_update=True)
        return

    def measure_hard(self):
        new_data = np.array(self.detector_function.get_values()).T

        ###########################
        # Shape determining block #
        ###########################

        datasetshape = self.dset.shape
        start_idx, stop_idx = self.get_datawriting_indices_update_ctr(new_data)
        new_datasetshape = (np.max([datasetshape[0], stop_idx]),
                            datasetshape[1])
        self.dset.resize(new_datasetshape)
        len_new_data = stop_idx-start_idx
        if len(np.shape(new_data)) == 1:
            old_vals = self.dset[start_idx:stop_idx,
                                 len(self.sweep_functions)]
            new_vals = ((new_data + old_vals*self.soft_iteration) /
                        (1+self.soft_iteration))

            self.dset[start_idx:stop_idx,
                      len(self.sweep_functions)] = new_vals
        else:
            old_vals = self.dset[start_idx:stop_idx,
                                 len(self.sweep_functions):]
            new_vals = ((new_data + old_vals * self.soft_iteration) /
                        (1 + self.soft_iteration))
            self.dset[start_idx:stop_idx,
                      len(self.sweep_functions):] = new_vals
        sweep_len = len(self.get_sweep_points().T) * self.acq_data_len_scaling


        ######################
        # DATA STORING BLOCK #
        ######################
        if sweep_len == len_new_data and self.mode == '1D':  # 1D sweep
            self.dset[:, 0] = self.get_sweep_points()
        else:
            try:
                if len(self.sweep_functions) != 1:
                    relevant_swp_points = self.get_sweep_points()[
                        start_idx:start_idx+len_new_data:]
                    self.dset[start_idx:, 0:len(self.sweep_functions)] = \
                        relevant_swp_points
                else:
                    self.dset[start_idx:, 0] = self.get_sweep_points()[
                        start_idx:start_idx+len_new_data:].T
            except Exception:
                # There are some cases where the sweep points are not
                # specified that you don't want to crash (e.g. on -off seq)
                logging.warning('You are in the exception case in '
                                'MC.measure_hard() DATA STORING BLOCK section. '
                                'Something might have gone wrong with your '
                                'measurement.')

        self.check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon()
        if self.mode == '2D':
            self.update_plotmon_2D_hard()
        self.iteration += 1
        self.print_progress(stop_idx)
        return new_data

    def measurement_function(self, x):
        '''
        Core measurement function used for soft sweeps
        '''
        if np.size(x) == 1:
            x = [x]
        if np.size(x) != len(self.sweep_functions):
            raise ValueError(
                'size of x "%s" not equal to # sweep functions' % x)
        for i, sweep_function in enumerate(self.sweep_functions[::-1]):
            # If statement below tests if the value is different from the
            # last value that was set, if it is the same the sweep function
            # will not be called. This is important when setting a parameter
            # is either expensive (e.g., loading a waveform) or has adverse
            # effects (e.g., phase scrambling when setting a MW frequency.

            # x[::-1] changes the order in which the parameters are set, so
            # it is first the outer sweep point and then the inner.This
            # is generally not important except for specifics: f.i. the phase
            # of an agilent generator is reset to 0 when the frequency is set.
            swp_pt = x[::-1][i]
            # The value that was actually set. Returned by the sweep
            # function if known.
            set_val = None
            if self.iteration == 0:
                # always set the first point
                set_val = sweep_function.set_parameter(swp_pt)
            else:
                # start_idx -1 refers to the last written value
                prev_swp_pt = self.last_sweep_pts[::-1][i]
                if swp_pt != prev_swp_pt:
                    # only set if not equal to previous point
                    try:
                        set_val = sweep_function.set_parameter(swp_pt)
                    except ValueError as e:
                        if self.cfg_clipping_mode():
                            log.warning(
                                'MC clipping mode caught exception:')
                            log.warning(e)
                        else:
                            raise e
                if isinstance(set_val, float):
                    # The Value in x is overwritten by the value that the
                    # sweep function returns. This allows saving the value
                    # that was actually set rather than the one that was
                    # intended. This does require custom support from
                    # a sweep function.
                    x[-i] = set_val
        
        # used for next iteration
        self.last_sweep_pts = x
        datasetshape = self.dset.shape
        # self.iteration = datasetshape[0] + 1

        vals = self.detector_function.acquire_data_point()
        start_idx, stop_idx = self.get_datawriting_indices_update_ctr(vals)
        # Resizing dataset and saving

        new_datasetshape = (np.max([datasetshape[0], stop_idx]),
                            datasetshape[1])
        self.dset.resize(new_datasetshape)
        new_data = np.append(x, vals)

        old_vals = self.dset[start_idx:stop_idx, :]
        new_vals = ((new_data + old_vals*self.soft_iteration) /
                    (1+self.soft_iteration))

        self.dset[start_idx:stop_idx, :] = new_vals
        # update plotmon
        self.check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon()
        if self.mode == '2D':
            self.update_plotmon_2D()
        elif self.mode == 'adaptive':
            self.update_plotmon_adaptive()
        self.iteration += 1
        if self.mode != 'adaptive':
            self.print_progress(stop_idx)
        return vals

    def optimization_function(self, x):
        '''
        A wrapper around the measurement function.
        It takes the following actions based on parameters specified
        in self.af_pars:
        - Rescales the function using the "x_scale" parameter, default is 1
        - Inverts the measured values if "minimize"==False
        - Compares measurement value with "f_termination" and raises an
        exception, that gets caught outside of the optimization loop, if
        the measured value is smaller than this f_termination.

        Measurement function with scaling to correct physical value
        '''
        if self.x_scale is not None:
            for i in range(len(x)):
                x[i] = float(x[i])/float(self.x_scale[i])

        vals = self.measurement_function(x)
        # This takes care of data that comes from a "single" segment of a
        # detector for a larger shape such as the UFHQC single int avg detector
        # that gives back data in the shape [[I_val_seg0, Q_val_seg0]]
        if len(np.shape(vals)) == 2:
            vals = np.array(vals)[:, 0]
        if self.minimize_optimization:
            if (self.f_termination is not None):
                if (vals < self.f_termination):
                    raise StopIteration()
        else:
            vals = self.measurement_function(x)
            # when maximizing interrupt when larger than condition before
            # inverting
            if (self.f_termination is not None):
                if (vals > self.f_termination):
                    raise StopIteration()
            vals = np.multiply(-1, vals)

        # to check if vals is an array with multiple values
        if hasattr(vals, '__iter__'):
            if len(vals) > 1:
                vals = vals[self.par_idx]

        return vals

    def finish(self, result):
        '''
        Deletes arrays to clean up memory and avoid memory related mistakes
        '''
        # this data can be plotted by enabling persist_mode
        self._persist_dat = result
        self._persist_xlabs = self.sweep_par_names
        self._persist_ylabs = self.detector_function.value_names
        self._persist_plotmon_axes_info = self._plotmon_axes_info

        for attr in ['TwoD_array',
                     'dset',
                     'sweep_points',
                     'sweep_points_2D',
                     'sweep_functions',
                     'xlen',
                     'ylen',
                     'iteration',
                     'soft_iteration']:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        if self._analysis_display is not None:
            self._analysis_display.update()

    ###################
    # 2D-measurements #
    ###################

    def run_2D(self, name=None, **kw):
        return self.run(name=name, mode='2D', **kw)

    def tile_sweep_pts_for_2D(self):
        self.xlen = len(self.get_sweep_points())
        self.ylen = len(self.sweep_points_2D)
        if np.size(self.get_sweep_points()[0]) == 1:
            # create inner loop pts
            self.sweep_pts_x = self.get_sweep_points()
            x_tiled = np.tile(self.sweep_pts_x, self.ylen)
            # create outer loop
            self.sweep_pts_y = self.sweep_points_2D
            y_rep = np.repeat(self.sweep_pts_y, self.xlen, axis=0)
            c = np.column_stack((x_tiled, y_rep))
            self.set_sweep_points(c)
            self.initialize_plot_monitor_2D()
        return

    def measure_2D(self, **kw):
        '''
        Sweeps over two parameters set by sweep_function and sweep_function_2D.
        The outer loop is set by sweep_function_2D, the inner loop by the
        sweep_function.

        Soft(ware) controlled sweep functions require soft detectors.
        Hard(ware) controlled sweep functions require hard detectors.
        '''
        self.tile_sweep_pts_for_2D()
        self.measure(**kw)
        return

    def set_sweep_function_2D(self, sweep_function):
        # If it is not a sweep function, assume it is a qc.parameter
        # and try to auto convert it it
        if not isinstance(sweep_function, swf.Sweep_function):
            sweep_function = wrap_par_to_swf(sweep_function)

        if len(self.sweep_functions) != 1:
            raise KeyError(
                'Specify sweepfunction 1D before specifying sweep_function 2D')
        else:
            self.sweep_functions.append(sweep_function)
            self.sweep_function_names.append(
                str(sweep_function.__class__.__name__))

    def set_sweep_points_2D(self, sweep_points_2D):
        self.sweep_functions[1].sweep_points = sweep_points_2D
        self.sweep_points_2D = sweep_points_2D

    ###########
    # Plotmon #
    ###########
    '''
    There are (will be) three kinds of plotmons, the regular plotmon,
    the 2D plotmon (which does a heatmap) and the adaptive plotmon.
    '''

    def _get_plotmon_axes_info(self):
        '''
        Returns a dict indexed by value_names, which contains information
        about plot labels, units, and translation of sweep_par values
        to values of sweep_points corresponding to the respective measure
        object.
        '''

        # extract sweep_points and related info from exp_metadata
        movnm = self.exp_metadata.get('meas_obj_value_names_map', None)
        mospm = self.exp_metadata.get('meas_obj_sweep_points_map', None)
        sp = self.exp_metadata.get('sweep_points', None)
        if sp is not None:
            try:
                sp = sp_mod.SweepPoints(sp)
            except Exception:
                sp = None
        # create a reverse lookup dictionary (value names measure object map)
        vnmom = None
        if movnm is not None:
            vnmom = {vn: mo for mo, vns in movnm.items() for vn in vns}

        # The following values are available even if there are no
        # sweep_points in exp_metadata.
        # labels and units for sweep points (x axes of 1D plots, x axes and y
        # axes of 2D plots)
        slabels = self.sweep_par_names
        sunits = self.sweep_par_units
        # labels and units for measured values (each one will be used for
        # the y axis of a 1D plot, and possible as the colorbar of a 2D plot)
        zlabels = self.detector_function.value_names
        zunits = self.detector_function.value_units

        # cf != 1 indicates a compressed 2D sweep
        cf = self.exp_metadata.get("compression_factor", 1)
        # Dict with infos for live plotting, indexed by value_names. This
        # allows to format the axes differently for each measured quantity.
        plotmon_axes_info = {}
        # This loop goes over the measured quantities, i.e., columns of
        # measured data.
        for j, vn in enumerate(self.detector_function.value_names):
            # The default if to use information that is always available.
            zlabel = zlabels[j]
            zunit = zunits[j]
            labels = [l for l in slabels]
            units = [u for u in sunits]
            sweep_vals = [[] for l in labels]

            # The plotmon_axes_info created here will be updated multiple
            # times whenever additional information has been extracted
            # inside the following try-except block. It is done like this in
            # order to keep the results of as many steps as possible in case
            # an execption occurs.
            plotmon_axes_info[vn] = dict(
                # labels and units for x axis/es of 1D plot(s)
                labels=labels,
                units=units,
                # label and unit for y axis of 1D plot / colorbar of 2D plot
                zlabel=zlabel,
                zunit=zunit,
                # labels and units for x and y axes of the 2D plot
                labels_2D=labels,
                units_2D=units,
                # Numerical values of the sweep points are used as x and y
                # coordinates of the points in the 2D plot
                sweep_vals=sweep_vals,
                # A loopup table to translate sweep_par values to values of
                # sweep_points corresponding to the respective measure
                # object is needed for x axis/es of the 1D plot(s).
                # If it stays empty, raw values will be used in the plots.
                lookup=[{}] * len(sweep_vals)
            )

            try:
                # Try to get the sweep_vals, which are used as x and y
                # coordinates of the points in the 2D plot, and which are
                # used to construct the lookup table below.
                if self.mode == '2D':
                    # For 2D measurements, these values have been created
                    # before in self.tile_sweep_pts_for_2D
                    sweep_vals[0] = self.sweep_pts_x
                    sweep_vals[1] = self.sweep_pts_y
                else:
                    sweep_vals[0] = self.get_sweep_points()
                    # Use an empty list if no sweep_vals are available
                    if sweep_vals[0] is None:
                        sweep_vals[0] = []
                    # In case of a 1D sweep over 2-dimensional vectors,
                    # we get a list of the vectors, but the sweep_vals
                    # should instead contain a list of x values in
                    # sweep_vals[0] and a list of y values in sweep_vals[1].
                    if np.asarray(sweep_vals[0]).ndim == 2:
                        sweep_vals = [sweep_vals[0][:, i] for i in range(
                            np.asarray(sweep_vals[0]).shape[1])]

                # Correct the sweep_vals if 2D sweep compression was used
                if cf != 1:
                    x, y = sweep_vals
                    n = int(len(y) * cf)  # number of y vals w/o compression
                    m = int(len(x) / cf)  # number of x vals w/o compression
                    new_x = x[:m]  # undo tiling of x vals
                    # Try to reconstruct reasonable y vals w/o compression.
                    # FIXME: If the compression is done by
                    #  Sequence.compress_2D_sweep, the points in the y
                    #  direction are anyways just indices, so that the
                    #  following code could be simplified. Will we ever use
                    #  compression in a different context for which the
                    #  following approach is needed and reasonable?
                    try:
                        # assumes constant spacing between swp for plotting
                        step = np.abs(y[-1] - y[-2])
                    except IndexError:
                        # This fallback is used to have a step value in the
                        # same order of magnitude as the value of the single
                        # sweep point
                        step = np.abs(y[0]) if y[0] != 0 else 1
                    new_y = list(y[0] + step * np.arange(0, n))
                    sweep_vals = [new_x, new_y]

                # We now try to extract better information about sweep_vals
                # by accessing the metadata, but we need to keep the sweep_vals
                # that are known to MC in order to construct the loopup
                # table later.
                new_sweep_vals = deepcopy(sweep_vals)

                # Try to find a measured object (mo) to which the measured
                # value belongs
                mo = vnmom.get(zlabel, None) if vnmom is not None else None
                # update the label of the measured value with the mo name
                if mo is not None:
                    zlabel = f'{mo}: {zlabel}'
                    plotmon_axes_info[vn].update(dict(zlabel=zlabel))
                # If sweep_points are available for this mo, we use those
                # to update the labels, units, sweep_vals. (Note that
                # sweep points can indeed be specific to a particular mo,
                # e.g., in parallel calibration measurements).
                if mo is not None and mospm is not None and sp is not None \
                        and mo in mospm:
                    # lookup dict to get the dimension (dim) to which sweep
                    # point (sp) belongs
                    dim_sp = {spn: sp.find_parameter(spn) for spn in
                              mospm[mo]}
                    for i in range(len(labels)):
                        # Get a list of all sp in dim i. If there are
                        # multiple sp for this mo in this dim, we use the
                        # first one.
                        spi = [spn for spn, dim in dim_sp.items() if dim == i]
                        # Only update labels, units, and sweep_vals if an sp
                        # was found
                        if len(spi):
                            labels[i] = sp.get_sweep_params_property(
                                'label', i, spi[0])
                            units[i] = sp.get_sweep_params_property(
                                'unit', i, spi[0])
                            # preliminary version of sweep_values,
                            # might still need to be modified below
                            tmp_sweep_vals = sp.get_sweep_params_property(
                                'values', i, spi[0])
                            # In the dim 0 (hard sweep direction), we might
                            # have to deal with calibration points (cp).
                            if i == 0:
                                # Add sweep points for cal states in the
                                # hard sweep direction.
                                # The following line is needed for eval.
                                CalibrationPoints = cp_mod.CalibrationPoints
                                try:
                                    # Try to get number of cal points from
                                    # metadata. This will allow us later on to
                                    # detect cases with multiple acquisition
                                    # elements.
                                    n_cp = len(eval(self.exp_metadata[
                                                        'cal_points']).states)
                                except Exception:
                                    # Guess number of cal states, which is
                                    # correct as long as we are not dealing
                                    # with multiple acquisition elements.
                                    n_cp = len(new_sweep_vals[i]) - len(
                                        tmp_sweep_vals)
                            else:
                                n_cp = 0  # number of cp is zero in other dims

                            # If number of cp is nonzero we need to extend
                            # the preliminary sweep_vals extracted above.
                            # Otherwise, we can use them as they are.
                            if n_cp > 0:
                                cp = cp_mod.CalibrationPoints
                                new_sweep_vals[i] = \
                                    cp.extend_sweep_points_by_n_cal_pts(
                                        n_cp, tmp_sweep_vals)
                            else:
                                new_sweep_vals[i] = tmp_sweep_vals

                # We can now already update labels and units with the values
                # extracted from sweep_points. For the sweep_vals, we still
                # have to handle some special cases below.
                plotmon_axes_info[vn].update(dict(labels=labels, units=units))

                # Here, we try to detect cases in which the sweep_points
                # cannot be used as sweep_vals. In this case, we fall back
                # to showing only the sweep indices (index of the sweep point).
                # In these cases, and in cases where sweep indices have
                # been provided in the first place, the label has to be
                # changed to 'sweep index' and the unit has to be removed.
                try:
                    for i in range(len(labels)):  # for each sweep dimension
                        if len(new_sweep_vals[i]) != len(sweep_vals[i]) or \
                                not isinstance(new_sweep_vals[i][0],
                                               numbers.Number):
                            # There seem to be multiple acquisition elements
                            # or the sweep points are not numeric.
                            # Fall back to sweep indices.
                            new_sweep_vals[i] = range(len(sweep_vals[i]))
                        # update label if sweep points look like sweep indices
                        if len(new_sweep_vals[i]):
                            try:
                                np.testing.assert_equal(
                                    list(new_sweep_vals[i]),
                                    list(range(len(new_sweep_vals[i]))))
                                labels[i] = 'sweep index'
                                units[i] = ''
                            except AssertionError:
                                pass
                except:
                    pass

                # Update labels and units (in case we show sweep indices).
                plotmon_axes_info[vn].update(dict(labels=labels, units=units))

                # create lookup table for 1D plots (main plotmon)
                try:
                    # the lookup table maps values in sweep_vals (sweep
                    # values used by MC) to values in new_sweep_vals (based
                    # on sweep_points and on the above corrections)
                    plotmon_axes_info[vn].update(dict(lookup=[
                        {t: n for t, n in zip(ts, ns)}
                        for ts, ns in zip(sweep_vals, new_sweep_vals)]))
                    # if 2D sweep compression was used
                    if cf != 1:
                        # In dim 0 (hard sweep), the sweep points known by
                        # MC are segment indices after compression. Multiple
                        # of these indices correspond to the same hard sweep
                        # point before compression. Therefore, the original
                        # sweep points (written into lookup above) have to
                        # be tiled.
                        plotmon_axes_info[vn]['lookup'][0] = {
                            k: v for k, v in zip(
                                self.sweep_pts_x,
                                np.tile(
                                    list(plotmon_axes_info[vn]['lookup'][
                                             0].values()),
                                    cf))}
                        # In dim 1 (soft sweep), update_plotmon will
                        # reconstruct the sweep indices of the uncompressed
                        # 2D sweep, so we can simply use the indices as keys
                        # in the lookup table. The vals in the lookup table
                        # are the original sweep points (written into lookup
                        # above).
                        plotmon_axes_info[vn]['lookup'][1] = {
                            i: v for i, v in enumerate(
                                plotmon_axes_info[vn]['lookup'][1].values())}
                except Exception:
                    # leave lookup table empty so that raw values are used
                    plotmon_axes_info[vn].update(
                        dict(lookup=[{}] * len(sweep_vals)))

                # create axes info for 2D plot in secondary plotmon
                if self.mode == '2D':
                    # copy values that were obtained above for the 1D plots
                    new_sweep_vals_2D = deepcopy(new_sweep_vals)
                    labels_2D = deepcopy(labels)
                    units_2D = deepcopy(units)

                    # If the new_sweep_vals are not equidistant (not
                    # supported by 2D plotmon), we have to fall back to
                    # displaying sweep indices in the 2D plot.
                    for i in range(len(labels)):  # for each sweep dim
                        # Check if the new_sweep_vals are not equidistant
                        diff = np.diff(new_sweep_vals[i])
                        if any([np.abs(d - diff[0]) / np.abs(diff[0]) > 1e-5
                                for d in diff]):
                            # fall back to sweep indices in 2D plot
                            new_sweep_vals_2D[i] = range(len(
                                new_sweep_vals[i]))
                            labels_2D[i] = 'sweep index'
                            units_2D[i] = ''

                    # Update the information for the 2D plot
                    plotmon_axes_info[vn].update(dict(
                        labels_2D=labels_2D,
                        units_2D=units_2D,
                        sweep_vals=new_sweep_vals_2D,
                    ))
            except Exception as e:
                # Unhandled errors in live plotting are not critical for the
                # measurement, so we log them as warnings.
                log.warning(traceback.format_exc())

        return plotmon_axes_info


    def initialize_plot_monitor(self):
        # new code
        try:
            if self.main_QtPlot.traces != []:
                self.main_QtPlot.clear()
            self.curves = []
            self._plotmon_axes_info = self._get_plotmon_axes_info()
            j = 0
            persist = self.persist_mode()
            if persist:
                # If any plot settings have changed, the plots should be
                # cleared by setting persist to False.
                try:
                    np.testing.assert_equal(self._persist_plotmon_axes_info,
                                            self._plotmon_axes_info)
                except AssertionError:
                    persist = False
            for yi, vn in enumerate(self.detector_function.value_names):
                axes_info = self._plotmon_axes_info[vn]
                # The measured-value axis is called z axis in the axis info
                # dict, but is the y axis of a 1D plot
                ylabel = axes_info['zlabel']
                yunit = axes_info['zunit']
                for xi, (xlabel, xunit) in enumerate(zip(axes_info['labels'],
                                                         axes_info['units'])):
                    if persist:  # plotting persist first so new data on top
                        yp = self._persist_dat[
                            :, yi+len(self.sweep_function_names)]
                        xp = self._persist_dat[:, xi]
                        # Update the sweep point values in the
                        # persist_data using the lookup table in the axis
                        # info dict
                        xp = [axes_info['lookup'][xi].get(xk, xk) for
                             xk in xp]
                        if len(xp) < self.plotting_max_pts():
                            self.main_QtPlot.add(x=xp, y=yp,
                                                 subplot=j+1,
                                                 color=0.75,  # a grayscale value
                                                 symbol='o', symbolSize=5)
                    self.main_QtPlot.add(x=[0], y=[0],
                                         xlabel=xlabel,
                                         xunit=xunit,
                                         ylabel=ylabel,
                                         yunit=yunit,
                                         subplot=j+1,
                                         color=color_cycle[j % len(color_cycle)],
                                         symbol='o', symbolSize=5)
                    self.curves.append(self.main_QtPlot.traces[-1])
                    j += 1
                self.main_QtPlot.win.nextRow()
        except Exception as e:
            log.warning(traceback.format_exc())

    def update_plotmon(self, force_update=False):
        # Note: plotting_max_pts takes precendence over force update
        if self.live_plot_enabled() and (self.dset.shape[0] <
                                         self.plotting_max_pts()):
            i = 0  # index of the plot
            try:
                time_since_last_mon_update = time.time() - self._mon_upd_time
            except:
                # creates the time variables if they did not exists yet
                self._mon_upd_time = time.time()
                time_since_last_mon_update = 1e9
            try:
                cf = self.exp_metadata.get("compression_factor", 1)
                if (time_since_last_mon_update > self.plotting_interval() or
                        force_update):

                    nr_sweep_funcs = len(self.sweep_function_names)
                    # for each column of measured data
                    for y_ind, vn in enumerate(
                            self.detector_function.value_names):
                        # get the axis info for this column
                        axes_info = self._plotmon_axes_info[vn]
                        # The first nr_sweep_funcs columns are sweep values
                        # (x axis in 1D plots), the following columns are
                        # data columns (y axis in 1D plots).
                        y = self.dset[:, nr_sweep_funcs + y_ind]
                        x_vals = [self.dset[:, x_ind] for x_ind in range(
                            nr_sweep_funcs)]
                        # If 2D sweep compression was used, calculate the soft
                        # sweep index of the uncompressed 2D sweep. We will
                        # either find the original soft sweep points in the
                        # lookup table by using this sweep index, or we will
                        # directly plot over the sweep index.
                        if cf != 1:
                            x_vals[1] = [int(x * cf
                                             + (i % len(self.sweep_pts_x)) /
                                             (len(self.sweep_pts_x) / cf))
                                         for i, x in enumerate(x_vals[1])]
                        # For all sweep dimensions
                        for x_ind, x in enumerate(x_vals):
                            # Update the sweep point values using the lookup
                            # table in the axis info dict. If a value is not
                            # found in the lookup table, the raw value is used.
                            x = [axes_info['lookup'][x_ind].get(xk, xk) for
                                 xk in x]

                            # Update the i-th plot with the y-values of the
                            # currently considered data column and the
                            # x-values of the currently considered sweep
                            # dimension.
                            self.curves[i]['config']['x'] = x
                            self.curves[i]['config']['y'] = y
                            i += 1
                    self._mon_upd_time = time.time()
                    self.main_QtPlot.update_plot()
            except Exception as e:
                log.warning(traceback.format_exc())

    def initialize_plot_monitor_2D(self):
        '''
        Preallocates a data array to be used for the update_plotmon_2D command.

        Made to work with at most 2 2D arrays (as this is how the labview code
        works). It should be easy to extend this function for more vals.
        '''
        if self.live_plot_enabled():
            try:
                self.time_last_2Dplot_update = time.time()
                self._plotmon_axes_info = self._get_plotmon_axes_info()
                sv = list(self._plotmon_axes_info.values())[0]['sweep_vals']
                self.TwoD_array = np.empty(
                    [len(sv[1]), len(sv[0]),
                     len(self.detector_function.value_names)])
                self.TwoD_array[:] = np.NAN
                self.secondary_QtPlot.clear()
                for j, vn in enumerate(self.detector_function.value_names):
                    axes_info = self._plotmon_axes_info[vn]
                    self.secondary_QtPlot.add(
                        x=axes_info['sweep_vals'][0],
                        y=axes_info['sweep_vals'][1],
                        z=self.TwoD_array[:, :, j],
                        xlabel=axes_info['labels_2D'][0],
                        xunit=axes_info['units_2D'][0],
                        ylabel=axes_info['labels_2D'][1],
                        yunit=axes_info['units_2D'][1],
                        zlabel=axes_info['zlabel'], zunit=axes_info['zunit'],
                        subplot=j+1, cmap='viridis'
                    )
            except Exception as e:
                log.warning(traceback.format_exc())

    def update_plotmon_2D(self, force_update=False):
        '''
        Adds latest measured value to the TwoD_array and sends it
        to the QC_QtPlot.
        '''
        if self.live_plot_enabled():
            try:
                i = int((self.iteration) % (self.xlen*self.ylen))
                x_ind = int(i % self.xlen)
                y_ind = int(i / self.xlen)
                for j in range(len(self.detector_function.value_names)):
                    z_ind = len(self.sweep_functions) + j
                    self.TwoD_array[y_ind, x_ind, j] = self.dset[i, z_ind]
                self.secondary_QtPlot.traces[j]['config'][
                    'z'] = self.TwoD_array[:, :, j]
                if (time.time() - self.time_last_2Dplot_update >
                        self.plotting_interval()
                        or self.iteration == len(self.sweep_points) or
                        force_update):
                    self.time_last_2Dplot_update = time.time()
                    self.secondary_QtPlot.update_plot()
            except Exception as e:
                log.warning(traceback.format_exc())

    def initialize_plot_monitor_adaptive(self):
        '''
        Uses the Qcodes plotting windows for plotting adaptive plot updates
        '''
        if self.adaptive_function.__module__ == 'cma.evolution_strategy':
            return self.initialize_plot_monitor_adaptive_cma()
        self.time_last_ad_plot_update = time.time()
        self.secondary_QtPlot.clear()

        zlabels = self.detector_function.value_names
        zunits = self.detector_function.value_units

        for j in range(len(self.detector_function.value_names)):
            self.secondary_QtPlot.add(x=[0],
                                      y=[0],
                                      xlabel='iteration',
                                      ylabel=zlabels[j],
                                      yunit=zunits[j],
                                      subplot=j+1,
                                      symbol='o', symbolSize=5)

    def update_plotmon_adaptive(self, force_update=False):
        if self.adaptive_function.__module__ == 'cma.evolution_strategy':
            return self.update_plotmon_adaptive_cma(force_update=force_update)

        if self.live_plot_enabled():
            try:
                if (time.time() - self.time_last_ad_plot_update >
                        self.plotting_interval() or force_update):
                    for j in range(len(self.detector_function.value_names)):
                        y_ind = len(self.sweep_functions) + j
                        y = self.dset[:, y_ind]
                        x = range(len(y))
                        self.secondary_QtPlot.traces[j]['config']['x'] = x
                        self.secondary_QtPlot.traces[j]['config']['y'] = y
                        self.time_last_ad_plot_update = time.time()
                        self.secondary_QtPlot.update_plot()
            except Exception as e:
                log.warning(traceback.format_exc())

    def initialize_plot_monitor_adaptive_cma(self):
        '''
        Uses the Qcodes plotting windows for plotting adaptive plot updates
        '''
        # new code
        if self.main_QtPlot.traces != []:
            self.main_QtPlot.clear()

        self.curves = []
        self.curves_best_ever = []
        self.curves_distr_mean = []

        xlabels = self.sweep_par_names
        xunits = self.sweep_par_units
        ylabels = self.detector_function.value_names
        yunits = self.detector_function.value_units

        j = 0
        if (self._persist_ylabs == ylabels and
                self._persist_xlabs == xlabels) and self.persist_mode():
            persist = True
        else:
            persist = False

        ##########################################
        # Main plotmon
        ##########################################
        for yi, ylab in enumerate(ylabels):
            for xi, xlab in enumerate(xlabels):
                if persist:  # plotting persist first so new data on top
                    yp = self._persist_dat[
                        :, yi+len(self.sweep_function_names)]
                    xp = self._persist_dat[:, xi]
                    if len(xp) < self.plotting_max_pts():
                        self.main_QtPlot.add(x=xp, y=yp,
                                             subplot=j+1,
                                             color=0.75,  # a grayscale value
                                             symbol='o',
                                             pen=None,  # makes it a scatter
                                             symbolSize=5)

                self.main_QtPlot.add(x=[0], y=[0],
                                     xlabel=xlab,
                                     xunit=xunits[xi],
                                     ylabel=ylab,
                                     yunit=yunits[yi],
                                     subplot=j+1,
                                     pen=None,
                                     color=color_cycle[0],
                                     symbol='o', symbolSize=5)
                self.curves.append(self.main_QtPlot.traces[-1])

                self.main_QtPlot.add(x=[0], y=[0],
                                     xlabel=xlab,
                                     xunit=xunits[xi],
                                     ylabel=ylab,
                                     yunit=yunits[yi],
                                     subplot=j+1,
                                     color=color_cycle[2],
                                     symbol='o', symbolSize=5)
                self.curves_distr_mean.append(self.main_QtPlot.traces[-1])

                self.main_QtPlot.add(x=[0], y=[0],
                                     xlabel=xlab,
                                     xunit=xunits[xi],
                                     ylabel=ylab,
                                     yunit=yunits[yi],
                                     subplot=j+1,
                                     pen=None,
                                     color=color_cycle[1],
                                     symbol='star',  symbolSize=10)
                self.curves_best_ever.append(self.main_QtPlot.traces[-1])

                j += 1
            self.main_QtPlot.win.nextRow()

        ##########################################
        # Secondary plotmon
        ##########################################

        self.secondary_QtPlot.clear()
        self.iter_traces = []
        self.iter_bever_traces = []
        self.iter_mean_traces = []
        for j in range(len(self.detector_function.value_names)):
            self.secondary_QtPlot.add(x=[0],
                                      y=[0],
                                      name='Measured values',
                                      xlabel='Iteration',
                                      x_unit='#',
                                      color=color_cycle[0],
                                      ylabel=ylabels[j],
                                      yunit=yunits[j],
                                      subplot=j+1,
                                      symbol='o', symbolSize=5)
            self.iter_traces.append(self.secondary_QtPlot.traces[-1])

            self.secondary_QtPlot.add(x=[0], y=[0],
                                      symbol='star', symbolSize=15,
                                      name='Best ever measured',
                                      color=color_cycle[1],
                                      xlabel='iteration',
                                      x_unit='#',
                                      ylabel=ylabels[j],
                                      yunit=yunits[j],
                                      subplot=j+1)
            self.iter_bever_traces.append(self.secondary_QtPlot.traces[-1])
            self.secondary_QtPlot.add(x=[0], y=[0],
                                      color=color_cycle[2],
                                      name='Generational mean',
                                      symbol='o', symbolSize=8,
                                      xlabel='iteration',
                                      x_unit='#',
                                      ylabel=ylabels[j],
                                      yunit=yunits[j],
                                      subplot=j+1)
            self.iter_mean_traces.append(self.secondary_QtPlot.traces[-1])

        # required for the first update call to work
        self.time_last_ad_plot_update = time.time()

    def update_plotmon_adaptive_cma(self, force_update=False):
        """
        Special adaptive plotmon for
        """

        if self.live_plot_enabled():
            try:
                if (time.time() - self.time_last_ad_plot_update >
                        self.plotting_interval() or force_update):
                    ##########################################
                    # Main plotmon
                    ##########################################
                    i = 0
                    nr_sweep_funcs = len(self.sweep_function_names)

                    # best_idx -1 as we count from 0 and best eval
                    # counts from 1.
                    best_index = int(self.opt_res_dset[-1, -1] - 1)

                    for j in range(len(self.detector_function.value_names)):
                        y_ind = nr_sweep_funcs + j

                        ##########################################
                        # Main plotmon
                        ##########################################
                        for x_ind in range(nr_sweep_funcs):

                            x = self.dset[:, x_ind]
                            y = self.dset[:, y_ind]

                            self.curves[i]['config']['x'] = x
                            self.curves[i]['config']['y'] = y

                            best_x = x[best_index]
                            best_y = y[best_index]
                            self.curves_best_ever[i]['config']['x'] = [best_x]
                            self.curves_best_ever[i]['config']['y'] = [best_y]
                            mean_x = self.opt_res_dset[:, 2+x_ind]
                            # std_x is needed to implement errorbars on X
                            # std_x = self.opt_res_dset[:, 2+nr_sweep_funcs+x_ind]
                            # to be replaced with an actual mean
                            mean_y = self.opt_res_dset[:, 2+2*nr_sweep_funcs]
                            mean_y = get_generation_means(
                                self.opt_res_dset[:, 1], y)
                            # TODO: turn into errorbars
                            self.curves_distr_mean[i]['config']['x'] = mean_x
                            self.curves_distr_mean[i]['config']['y'] = mean_y
                            i += 1
                        ##########################################
                        # Secondary plotmon
                        ##########################################
                        # Measured value vs function evaluation
                        y = self.dset[:, y_ind]
                        x = range(len(y))
                        self.iter_traces[j]['config']['x'] = x
                        self.iter_traces[j]['config']['y'] = y

                        # generational means
                        gen_idx = self.opt_res_dset[:, 1]
                        self.iter_mean_traces[j]['config']['x'] = gen_idx
                        self.iter_mean_traces[j]['config']['y'] = mean_y

                        # This plots the best ever measured value vs iteration
                        # number of evals column
                        best_evals_idx = (
                            self.opt_res_dset[:, -1] - 1).astype(int)
                        best_func_val = y[best_evals_idx]
                        self.iter_bever_traces[j]['config']['x'] = best_evals_idx
                        self.iter_bever_traces[j]['config']['y'] = best_func_val

                    self.main_QtPlot.update_plot()
                    self.secondary_QtPlot.update_plot()

                    self.time_last_ad_plot_update = time.time()

            except Exception as e:
                log.warning(traceback.format_exc())

    def update_plotmon_2D_hard(self):
        '''
        Adds latest datarow to the TwoD_array and send it
        to the QC_QtPlot.
        Note that the plotmon only supports evenly spaced lattices.
        '''
        try:
            if self.live_plot_enabled():
                i = int((self.iteration) % self.ylen)
                y_ind = i
                cf = self.exp_metadata.get('compression_factor', 1)
                for j in range(len(self.detector_function.value_names)):
                    z_ind = len(self.sweep_functions) + j
                    data_row = self.dset[
                        i*self.xlen:(i+1)*self.xlen, z_ind]
                    if cf != 1:
                        # reshape data according to compression factor
                        data_reshaped = data_row.reshape((cf, int(len(data_row)/cf)))
                        y_start = self.iteration*cf % self.TwoD_array.shape[0]
                        y_end = y_start + cf
                        self.TwoD_array[y_start:y_end, :, j] = data_reshaped
                    else:
                        self.TwoD_array[y_ind, :, j] = data_row
                    self.secondary_QtPlot.traces[j]['config']['z'] = \
                        self.TwoD_array[:, :, j]

                if (time.time() - self.time_last_2Dplot_update >
                        self.plotting_interval()
                        or self.iteration + 1 == len(
                            self.sweep_points) / self.xlen):
                    self.time_last_2Dplot_update = time.time()
                    self.secondary_QtPlot.update_plot()
        except Exception as e:
            log.warning(traceback.format_exc())

    def _set_plotting_interval(self, plotting_interval):
        if hasattr(self, 'main_QtPlot'):
            self.main_QtPlot.interval = plotting_interval
            self.secondary_QtPlot.interval = plotting_interval
        self._plotting_interval = plotting_interval

    def _get_plotting_interval(self):
        return self._plotting_interval

    def clear_persitent_plot(self):
        self._persist_dat = None
        self._persist_xlabs = None
        self._persist_ylabs = None
        self._persist_plotmon_axes_info = None

    def update_instrument_monitor(self):
        if self.instrument_monitor() is not None:
            inst_mon = self.find_instrument(self.instrument_monitor())
            inst_mon.update()

    ##################################
    # Small helper/utility functions #
    ##################################

    def get_data_object(self):
        '''
        Used for external functions to write to a datafile.
        This is used in time_domain_measurement as a hack and is not
        recommended.
        '''
        return self.data_object

    def get_column_names(self):
        self.column_names = []
        self.sweep_par_names = []
        self.sweep_par_units = []

        for sweep_function in self.sweep_functions:
            self.column_names.append(sweep_function.parameter_name+' (' +
                                     sweep_function.unit+')')

            self.sweep_par_names.append(sweep_function.parameter_name)
            self.sweep_par_units.append(sweep_function.unit)

        for i, val_name in enumerate(self.detector_function.value_names):
            self.column_names.append(
                val_name+' (' + self.detector_function.value_units[i] + ')')
        return self.column_names

    def create_experimentaldata_dataset(self):
        if 'Experimental Data' in self.data_object:
            data_group = self.data_object['Experimental Data']
        else:
            data_group = self.data_object.create_group('Experimental Data')
        self.dset = data_group.create_dataset(
            'Data', (0, len(self.sweep_functions) +
                     len(self.detector_function.value_names)),
            maxshape=(None, len(self.sweep_functions) +
                      len(self.detector_function.value_names)),
            dtype='float64')
        self.get_column_names()
        self.dset.attrs['column_names'] = h5d.encode_to_utf8(self.column_names)
        # Added to tell analysis how to extract the data
        data_group.attrs['datasaving_format'] = h5d.encode_to_utf8('Version 2')
        data_group.attrs['sweep_parameter_names'] = h5d.encode_to_utf8(
            self.sweep_par_names)
        data_group.attrs['sweep_parameter_units'] = h5d.encode_to_utf8(
            self.sweep_par_units)

        data_group.attrs['value_names'] = h5d.encode_to_utf8(
            self.detector_function.value_names)
        data_group.attrs['value_units'] = h5d.encode_to_utf8(
            self.detector_function.value_units)

    def create_experiment_result_dict(self):
        try:
            # only exists as an open dataset when running an
            # optimization
            opt_res_dset = self.opt_res_dset[()]
        except (ValueError, AttributeError) as e:
            opt_res_dset = None

        result_dict = {
            "dset": self.dset[()],
            "opt_res_dset": opt_res_dset,
            "sweep_parameter_names": self.sweep_par_names,
            "sweep_parameter_units": self.sweep_par_units,
            "value_names": self.detector_function.value_names,
            "value_units": self.detector_function.value_units
        }
        return result_dict

    def save_optimization_settings(self):
        '''
        Saves the parameters used for optimization
        '''
        opt_sets_grp = self.data_object.create_group('Optimization settings')
        param_list = dict_to_ordered_tuples(self.af_pars)
        for (param, val) in param_list:
            opt_sets_grp.attrs[param] = str(val)

    def save_cma_optimization_results(self, es):
        """
        This function is to be used as the callback when running cma.fmin.
        It get's handed an instance of an EvolutionaryStrategy (es).
        From here it extracts the results and stores these in the hdf5 file
        of the experiment.
        """
        # code extra verbose to understand what is going on
        generation = es.result.iterations
        evals = es.result.evaluations  # number of evals at start of each gen
        xfavorite = es.result.xfavorite  # center of distribution, best est
        stds = es.result.stds   # stds of distribution, stds of xfavorite
        fbest = es.result.fbest  # best ever measured
        xbest = es.result.xbest  # coordinates of best ever measured
        evals_best = es.result.evals_best  # index of best measurement

        if not self.minimize_optimization:
            fbest = -fbest

        results_array = np.concatenate([[generation, evals],
                                        xfavorite, stds,
                                        [fbest], xbest, [evals_best]])
        if (not 'optimization_result'
                in self.data_object['Experimental Data'].keys()):
            opt_res_grp = self.data_object['Experimental Data']
            self.opt_res_dset = opt_res_grp.create_dataset(
                'optimization_result', (0, len(results_array)),
                maxshape=(None, len(results_array)),
                dtype='float64')

            # FIXME: Jan 2018, add the names of the parameters to column names
            self.opt_res_dset.attrs['column_names'] = h5d.encode_to_utf8(
                'generation, ' + 'evaluations, ' +
                'xfavorite, ' * len(xfavorite) +
                'stds, '*len(stds) +
                'fbest, ' + 'xbest, '*len(xbest) +
                'best evaluation,')

        old_shape = self.opt_res_dset.shape
        new_shape = (old_shape[0]+1, old_shape[1])
        self.opt_res_dset.resize(new_shape)
        self.opt_res_dset[-1, :] = results_array

    def save_optimization_results(self, adaptive_function, result):
        """
        Saves the result of an adaptive measurement (optimization) to
        the hdf5 file.

        Contains some hardcoded data reshufling based on known adaptive
        functions.
        """
        opt_res_grp = self.data_object.create_group('Optimization_result')

        if adaptive_function.__module__ == 'cma.evolution_strategy':
            res_dict = {'xopt':  result[0],
                        'fopt':  result[1],
                        'evalsopt': result[2],
                        'evals': result[3],
                        'iterations': result[4],
                        'xmean': result[5],
                        'stds': result[6],
                        'stop': result[-3]}
            # entries below cannot be stored
            # 'cmaes': result[-2],
            # 'logger': result[-1]}
        elif adaptive_function.__module__ == 'pycqed.measurement.optimization':
            res_dict = {'xopt':  result[0],
                        'fopt':  result[1]}
        else:
            res_dict = {'opt':  result}
        h5d.write_dict_to_hdf5(res_dict, entry_point=opt_res_grp)

    def save_instrument_settings(self, data_object=None, *args):
        '''
        uses QCodes station snapshot to save the last known value of any
        parameter. Only saves the value and not the update time (which is
        known in the snapshot)
        '''


        import numpy
        import sys
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=sys.maxsize)

        if data_object is None:
            data_object = self.data_object
        if not hasattr(self, 'station'):
            log.warning('No station object specified, could not save',
                            ' instrument settings')
        else:
            # # This saves the snapshot of the entire setup
            # snap_grp = data_object.create_group('Snapshot')
            # snap = self.station.snapshot()
            # h5d.write_dict_to_hdf5(snap, entry_point=snap_grp)

            # Below is old style saving of snapshot, exists for the sake of
            # preserving deprecated functionality
            set_grp = data_object.create_group('Instrument settings')
            inslist = dict_to_ordered_tuples(self.station.components)
            for (iname, ins) in inslist:
                instrument_grp = set_grp.create_group(iname)
                par_snap = ins.snapshot()['parameters']
                parameter_list = dict_to_ordered_tuples(par_snap)
                for (p_name, p) in parameter_list:
                    try:
                        val = repr(p['value'])
                    except KeyError:
                        val = ''
                    instrument_grp.attrs[p_name] = val
        numpy.set_printoptions(**opt)

    def save_MC_metadata(self, data_object=None, *args):
        '''
        Saves metadata on the MC (such as timings)
        '''
        set_grp = data_object.create_group('MC settings')

        bt = set_grp.create_dataset('begintime', (9, 1))
        bt[:, 0] = np.array(time.localtime(self.begintime))
        pt = set_grp.create_dataset('preparetime', (9, 1))
        pt[:, 0] = np.array(time.localtime(self.preparetime))
        et = set_grp.create_dataset('endtime', (9, 1))
        et[:, 0] = np.array(time.localtime(self.endtime))

        set_grp.attrs['mode'] = self.mode
        set_grp.attrs['measurement_name'] = self.measurement_name
        set_grp.attrs['live_plot_enabled'] = self.live_plot_enabled()
        sha1_id, diff = self.get_git_info()
        set_grp.attrs['git_sha1_id'] = sha1_id
        set_grp.attrs['git_diff'] = diff

    @classmethod
    def save_exp_metadata(self, metadata: dict, data_object):
        '''
        Saves experiment metadata to the data file. The metadata is saved at
            file['Experimental Data']['Experimental Metadata']

        Args:
            metadata (dict):
                    Simple dictionary without nesting. An attribute will be
                    created for every key in this dictionary.
            data_object:
                    An open hdf5 data object.
        '''
        if 'Experimental Data' in data_object:
            data_group = data_object['Experimental Data']
        else:
            data_group = data_object.create_group('Experimental Data')

        if 'Experimental Metadata' in data_group:
            metadata_group = data_group['Experimental Metadata']
        else:
            metadata_group = data_group.create_group('Experimental Metadata')

        h5d.write_dict_to_hdf5(metadata, entry_point=metadata_group)

    def get_percdone(self):
        percdone = self.total_nr_acquired_values / (
            np.shape(self.get_sweep_points())[0] * self.soft_avg()) * 100
        return percdone

    def print_progress(self, stop_idx=None):
        if self.verbose():
            acquired_points = self.dset.shape[0]
            total_nr_pts = len(self.get_sweep_points())
            percdone = self.get_percdone()
            elapsed_time = time.time() - self.begintime
            # The trailing spaces are to overwrite some characters in case the
            # previous progress message was longer.
            progress_message = "\r {percdone}% completed \telapsed time: "\
                "{t_elapsed}s \ttime left: {t_left}s     ".format(
                    percdone=int(percdone),
                    t_elapsed=round(elapsed_time, 1),
                    t_left=round((100.-percdone)/(percdone) *
                                 elapsed_time, 1) if
                    percdone != 0 else '')

            if percdone != 100:
                end_char = ''
            else:
                end_char = '\n'
            print('\r', progress_message, end=end_char)

    def is_complete(self):
        """
        Returns True if enough data has been acquired.
        """
        acquired_points = self.dset.shape[0]
        total_nr_pts = np.shape(self.get_sweep_points())[0]
        if acquired_points < total_nr_pts:
            return False
        elif acquired_points >= total_nr_pts:
            if self.soft_avg() != 1 and self.soft_iteration == 0:
                return False
            else:
                return True

    def print_measurement_start_msg(self):
        if self.verbose():
            if len(self.sweep_functions) == 1:
                print('Starting measurement: %s' % self.get_measurement_name())
                print('Sweep function: %s' %
                      self.get_sweep_function_names()[0])
                print('Detector function: %s'
                      % self.get_detector_function_name())
            else:
                print('Starting measurement: %s' % self.get_measurement_name())
                for i, sweep_function in enumerate(self.sweep_functions):
                    print('Sweep function %d: %s' % (
                        i, self.sweep_function_names[i]))
                print('Detector function: %s'
                      % self.get_detector_function_name())

    def get_datetimestamp(self):
        return time.strftime('%Y%m%d_%H%M%S', time.localtime())

    def get_datawriting_start_idx(self):
        if self.mode == 'adaptive':
            max_sweep_points = np.inf
        else:
            max_sweep_points = np.shape(self.get_sweep_points())[0]

        start_idx = int(self.total_nr_acquired_values % max_sweep_points)
        self.soft_iteration = int(
            self.total_nr_acquired_values//max_sweep_points)

        return start_idx

    def get_datawriting_indices_update_ctr(self, new_data,
                                           update: bool=True):
        """
        Calculates the start and stop indices required for
        storing a hard measurement.

        N.B. this also updates the "total_nr_acquired_values" counter.
        """

        # This is the case if the detector returns a simple float or int
        if len(np.shape(new_data)) == 0:
            xlen = 1
        # This is the case for a 1D hard detector or an N-D soft detector
        elif len(np.shape(new_data)) == 1:
            # Soft detector (returns values 1 by 1)
            if len(self.detector_function.value_names) == np.shape(new_data)[0]:
                xlen = 1
            else:  # 1D Hard detector (returns values in chunks)
                xlen = len(new_data)
        else:
            if self.detector_function.detector_control == 'soft':
                # FIXME: this is an inconsistency that should not be there.
                xlen = np.shape(new_data)[1]
            else:
                # in case of an N-D Hard detector dataset
                xlen = np.shape(new_data)[0]

        start_idx = self.get_datawriting_start_idx()
        stop_idx = start_idx + xlen

        if update:
            # Sometimes one wants to know the start/stop idx without
            self.total_nr_acquired_values += xlen

        return start_idx, stop_idx

    def check_keyboard_interrupt(self):
        try:  # Try except statement is to make it work on non windows pc
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if b'q' in key:
                    # this causes a KeyBoardInterrupt
                    raise KeyboardInterrupt('Human "q" terminated experiment.')
                elif b'f' in key:
                    # this should not raise an exception
                    raise KeyboardFinish(
                        'Human "f" terminated experiment safely.')
        except Exception:
            pass

    ####################################
    # Non-parameter get/set functions  #
    ####################################

    def set_sweep_function(self, sweep_function):
        '''
        Used if only 1 sweep function is set.
        '''
        # If it is not a sweep function, assume it is a qc.parameter
        # and try to auto convert it it
        if not isinstance(sweep_function, swf.Sweep_function):
            sweep_function = wrap_par_to_swf(sweep_function)
        self.sweep_functions = [sweep_function]
        self.set_sweep_function_names(
            [str(sweep_function.name)])

    def get_sweep_function(self):
        return self.sweep_functions[0]

    def set_sweep_functions(self, sweep_functions):
        '''
        Used to set an arbitrary number of sweep functions.
        '''
        sweep_function_names = []
        for i, sweep_func in enumerate(sweep_functions):
            # If it is not a sweep function, assume it is a qc.parameter
            # and try to auto convert it it
            if not hasattr(sweep_func, 'sweep_control'):
                sweep_func = wrap_par_to_swf(sweep_func)
                sweep_functions[i] = sweep_func
            sweep_function_names.append(str(sweep_func.name))
        self.sweep_functions = sweep_functions
        self.set_sweep_function_names(sweep_function_names)

    def get_sweep_functions(self):
        return self.sweep_functions

    def set_sweep_function_names(self, swfname):
        self.sweep_function_names = swfname

    def get_sweep_function_names(self):
        return self.sweep_function_names

    def set_detector_function(self, detector_function,
                              wrapped_det_control='soft'):
        """
        Sets the detector function. If a parameter is passed instead it
        will attempt to wrap it to a detector function.
        """
        if not hasattr(detector_function, 'detector_control'):
            detector_function = wrap_par_to_det(detector_function,
                                                wrapped_det_control)
        self.detector_function = detector_function
        self.set_detector_function_name(detector_function.name)

    def get_detector_function(self):
        return self.detector_function

    def set_detector_function_name(self, dfname):
        self._dfname = dfname

    def get_detector_function_name(self):
        return self._dfname

    ################################
    # Parameter get/set functions  #
    ################################

    def get_git_info(self):
        self.git_info = general.get_git_info()
        return self.git_info

    def get_measurement_begintime(self):
        self.begintime = time.time()
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def get_measurement_endtime(self):
        self.endtime = time.time()
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def get_measurement_preparetime(self):
        self.preparetime = time.time()
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def set_sweep_points(self, sweep_points):
        self.sweep_points = np.array(sweep_points)
        # line below is because some sweep funcs have their own sweep points
        # attached
        # This is a mighty bad line! Should be adding sweep points to the
        # individual sweep funcs
        if len(np.shape(sweep_points)) == 1:
            self.sweep_functions[0].sweep_points = np.array(sweep_points)

    def get_sweep_points(self):
        if hasattr(self, 'sweep_points'):
            return self.sweep_points
        else:
            return getattr(self.sweep_functions[0], 'sweep_points', None)

    def set_adaptive_function_parameters(self, adaptive_function_parameters):
        """
        adaptive_function_parameters: Dictionary containing options for
            running adaptive mode.

        The following arguments are reserved keywords. All other entries in
        the dictionary get passed to the adaptive function in the measurement
        loop.

        Reserved keywords:
            "adaptive_function":    function
            "x_scale": (array)     rescales sweep parameters for
                adaptive function, defaults to None (no rescaling).
                Each sweep_function/parameter is rescaled by dividing by
                the respective component of x_scale.
            "minimize": True        Bool, inverts value to allow minimizing
                                    or maximizing
            "f_termination" None    terminates the loop if the measured value
                                    is smaller than this value
            "par_idx": 0            If a parameter returns multiple values,
                                    specifies which one to use.

        Common keywords (used in python nelder_mead implementation):
            "x0":                   list of initial values
            "initial_step"
            "no_improv_break"
            "maxiter"
        """
        self.af_pars = adaptive_function_parameters

        # x_scale is expected to be an array or list.
        self.x_scale = self.af_pars.pop('x_scale', None)
        self.par_idx = self.af_pars.pop('par_idx', 0)
        # Determines if the optimization will minimize or maximize
        self.minimize_optimization = self.af_pars.pop('minimize', True)
        self.f_termination = self.af_pars.pop('f_termination', None)

        # ensures the cma optimization results are saved during the experiment
        if (self.af_pars['adaptive_function'].__module__ ==
                'cma.evolution_strategy' and 'callback' not in self.af_pars):
            self.af_pars['callback'] = self.save_cma_optimization_results

    def get_adaptive_function_parameters(self):
        return self.af_pars

    def set_measurement_name(self, measurement_name):
        if measurement_name is None:
            self.measurement_name = 'Measurement'
        else:
            self.measurement_name = measurement_name

    def get_measurement_name(self):
        return self.measurement_name

    def set_optimization_method(self, optimization_method):
        self.optimization_method = optimization_method

    def get_optimization_method(self):
        return self.optimization_method

    ################################
    # Actual parameters            #
    ################################

    def get_idn(self):
        """
        Required as a standard interface for QCoDeS instruments.
        """
        return {'vendor': 'PycQED', 'model': 'MeasurementControl',
                'serial': '', 'firmware': '2.0'}

    def analysis_display(self, ad):
        self._analysis_display = ad


class KeyboardFinish(KeyboardInterrupt):
    """
    Indicates that the user safely aborts/finishes the experiment.
    Used to finish the experiment without raising an exception.
    """
    pass
