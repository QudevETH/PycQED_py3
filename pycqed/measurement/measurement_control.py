import types
import logging

from pycqed.utilities.timer import Timer

log = logging.getLogger(__name__)
import time
from copy import deepcopy
import traceback
import requests

import numpy as np
import numbers
from scipy.optimize import fmin_powell

import pycqed.version
from pycqed.utilities import general
from pycqed.utilities.io import hdf5 as h5d
from pycqed.utilities.general import dict_to_ordered_tuples
from pycqed.utilities.get_default_datadir import get_default_datadir

# used for saving instrument settings
from pycqed.instrument_drivers import instrument as pycqedins


# used for axis labels
from pycqed.measurement import sweep_points as sp_mod
from pycqed.measurement.calibration import calibration_points as cp_mod

# Used for auto qcodes parameter wrapping
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import awg_sweep_functions as awg_swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_swf
from pycqed.measurement.mc_parameter_wrapper import wrap_par_to_det
from pycqed.analysis.tools.data_manipulation import get_generation_means

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
try:
    from qcodes_loop.plots.colors import color_cycle
except ModuleNotFoundError:
    from qcodes.plots.colors import color_cycle

from pycqed.utilities.errors import NoProgressError

try:
    import msvcrt  # used on windows to catch keyboard input
except:
    print('Could not import msvcrt (used for detecting keystrokes)')

try:
    from qcodes_loop.plots.pyqtgraph import QtPlot
except ModuleNotFoundError:
    try:
        from qcodes.plots.pyqtgraph import QtPlot
    except Exception:
        log.warning(
            'pyqtgraph plotting not supported. When instantiating an '
            'MC object, be sure to set live_plot_enabled=False. '
            'The full traceback follows:'
        )
        log.warning(traceback.format_exc())

EXPERIMENTAL_DATA_GROUP_NAME = 'Experimental Data'


class MeasurementControl(Instrument):

    '''
    New version of Measurement Control that allows for adaptively determining
    data points.
    '''

    def __init__(self, name: str,
                 plotting_interval: float=3,
                 datadir: str=get_default_datadir(),
                 live_plot_enabled: bool=True, verbose: bool=True):
        super().__init__(name=name)

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
        self.add_parameter('cyclic_soft_avg',
                           label='Cyclic soft averaging',
                           docstring='If set to True, soft averaging is '
                                     'performed cyclically over the soft '
                                     'sweep points. Consecutive soft '
                                     'averaging currently only works if the '
                                     'sweep in dim 0 is a hard sweep. '
                                     'Otherwise it falls  back to measuring '
                                     'cyclically in dim 1.'
                                     'Default: True',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(),
                           initial_value=True)
        self.add_parameter('soft_repetitions',
                           label='Number of soft repetitions',
                           docstring='Repeat hard measurements multiple '
                                     'times in a software loop to collect '
                                     'more results. All obtained results are '
                                     'appended to the data table in the HDF '
                                     'file (i.e., no soft averaging is done '
                                     'on them). This is currently only '
                                     'implemented for "hard" measurements.',
                           parameter_class=ManualParameter,
                           vals=vals.Ints(1, int(1e8)),
                           initial_value=1)
        self.add_parameter('program_only_on_change',
                           label='Program AWGs only on change',
                           docstring='Programs AWGs only if the soft sweep '
                                     'parameters changed from the last '
                                     'iteration. This speeds up measurements '
                                     'e.g. when soft averaging with '
                                     'cyclic_soft_avg = False.',
                           parameter_class=ManualParameter,
                           vals=vals.Bool(),
                           initial_value=False)

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
        self.add_parameter(
            'live_plot_2D_update', parameter_class=ManualParameter,
            vals=vals.Enum('on', 'row', 'off'), initial_value='on',
            docstring='Whether the 2D plotmon should be updated whenever new '
                      'data is available ("on"), only once per row ("row"), '
                      'or not at all ("off").')
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
        self.add_parameter(
            'clean_interrupt', docstring=
            'Whether data that has already been received from acquisition '
            'instruments should be stored in case of a KeyboardInterrupt.',
            vals=vals.Bool(), parameter_class=ManualParameter,
            initial_value=True)
        self.add_parameter('compress_dataset',
                           vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           initial_value=True)
        self.add_parameter(
            'max_attempts', docstring=
            'Maximum number of attempts. Values larger than 1 will mean that '
            'MC automatically retries in case of crashes during measurements. '
            'Optional: A list of function handles can be assigned to the '
            'retry_cleanup_functions property of the MeasurementControl '
            'instrument, which will then be called as callback functions '
            'between attempts (without passing any parameters to them). '
            'This can, e.g., be used to clear errors in AWGs. It can be '
            'useful to have setup-specific cleanup functions (e.g., defined '
            'in a notebook or init script).',
            vals=vals.Ints(),
            parameter_class=ManualParameter,
            initial_value=1)
        self.add_parameter(
            'no_progress_interval', docstring=
            'Time (in seconds) after which a measurement that does not make '
            'progress is reported.',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=600)
        self.add_parameter(
            'no_progress_kill_interval', docstring=
            'Time (in seconds) after which a measurement that does not make '
            'progress is interrupted.',
            vals=vals.Numbers(),
            parameter_class=ManualParameter,
            initial_value=np.inf)

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

        self.add_parameter(
            'settings_file_format',
            docstring='File format of the file which contains the instrument'
                      'settings.',
            initial_value='hdf5',
            vals=vals.Enum('hdf5', 'pickle', 'msgpack'),
            parameter_class=ManualParameter)

        self.add_parameter(
            'settings_file_compression',
            vals=vals.Bool(),
            docstring='True if file should be compressed with blosc2. '
                      'Does not support hdf5 files.',
            initial_value=True,
            parameter_class=ManualParameter
        )

        self.add_parameter('last_timestamp',
                           initial_value=None,
                           docstring='Timestamp of MC in the format '
                                     '%Y%m%d_%H%M%S.',
                           parameter_class=ManualParameter)

        # pyqtgraph plotting process is reused for different measurements.
        if self.live_plot_enabled():
            self.open_plotmon_windows()

        self.plotting_interval(plotting_interval)

        self.soft_iteration = 0  # used as a counter for soft_avg
        self._persist_dat = None
        self._persist_xlabs = None
        self._persist_ylabs = None
        self._analysis_display = None
        self.timer = Timer()

        self.exp_metadata = {}
        self._plotmon_axes_info = None
        self._persist_plotmon_axes_info = None

        # the following properties are used by get_percdone
        self._last_percdone_value = 0  # last progress
        self._last_percdone_change_time = 0  # last time progress changed
        self._last_percdone_log_time = 0  # last time progress was logged

        self.parameter_checks = {}

        # We initiallize the adaptive_function parameters
        self.af_pars = {}
        self.data_processing_function = self._default_data_processing_function
        """Data processing function for adaptive measurements, see docstring
        of _default_data_processing_function."""

    ##############################################
    # Functions used to control the measurements #
    ##############################################

    def create_instrument_settings_file(self, label=None):
        '''
        Saves a snapshot of the current instrument settings without carrying
        out a measurement.
        File format is taken from parameter "settings_file_format".

        :param label: (optional str) a label to be used in the filename
            (will be appended to the default label Instrument_settings)
        '''
        label = '' if label is None else '_' + label
        self.set_measurement_name('Instrument_settings' + label)
        self.last_timestamp(self.get_datetimestamp())
        if self.settings_file_format() == 'hdf5':
            with h5d.Data(name=self.get_measurement_name(),
                          datadir=self.datadir(),
                          timestamp=self.last_timestamp()) as self.data_object:
                self.save_instrument_settings(self.data_object)
        else:
            self.save_instrument_settings()

    def update_sweep_points(self):
        sweep_points = self.get_sweep_points()
        if sweep_points is not None:
            self.set_sweep_points(np.tile(
                sweep_points,
                self.acq_data_len_scaling * self.soft_repetitions()))
    @Timer()
    def run(self, name: str=None, exp_metadata: dict=None,
            mode: str='1D', disable_snapshot_metadata: bool=False,
            previous_attempts=0, **kw):
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
            previous_attempts (int): indicates how many times the measurement
                    has already been tried. This is usually not passed by
                    the calling function, but only used by run() when it
                    calls itself recursively.
        '''

        def try_finish():
            try:
                self.detector_function.finish()
            except Exception:
                pass  # no need to raise an exception if cleanup fails

        self.timer = Timer("MeasurementControl")
        # reset properties that are used by get_percdone
        self._last_percdone_value = 0
        self._last_percdone_change_time = 0
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
        if previous_attempts == 0:
            # The following call modifies the sweep points and should thus
            # only be called in the first attempt (to avoid modifying them
            # multiple times).
            self.update_sweep_points()

        # needs to be defined here because of the with statement below
        return_dict = {}
        self.last_sweep_pts = None  # used to prevent resetting same value

        self.begintime = None
        self.preparetime = None
        self.endtime = None

        if self.skip_measurement():
            return return_dict

        self.last_timestamp(self.get_datetimestamp())
        with h5d.Data(name=self.get_measurement_name(),
                      datadir=self.datadir(),
                      timestamp=self.last_timestamp()) \
                as self.data_object:
            if exp_metadata is not None:
                self.exp_metadata = deepcopy(exp_metadata)
            else:
                # delete metadata from previous measurement
                self.exp_metadata = {}
            det_metadata = self.detector_function.generate_metadata()
            self.exp_metadata.update(det_metadata)
            self.save_exp_metadata(self.exp_metadata)
            exception = None
            try:
                self.check_keyboard_interrupt()
                self.get_measurement_begintime()
                if not disable_snapshot_metadata:
                    self.save_instrument_settings(self.data_object)
                self.perform_parameter_checks()
                self.create_experimentaldata_dataset()
                if mode != 'adaptive':
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
                try_finish()
                print(e)
            except KeyboardInterrupt as e:
                try_finish()
                percentage_done = self.get_percdone()
                if percentage_done == 0 or not self.clean_interrupt():
                    raise e
                self.save_exp_metadata({'percentage_done': percentage_done})
                log.warning('Caught a KeyboardInterrupt and there is '
                            'unsaved data. Trying clean exit to save data.')
            except Exception as e:
                try_finish()
                # We store the exception instead of raising it directly.
                # After creating the results dict and storing end time +
                # metadata, the exception will be either logged (if an
                # automatic retry is triggered) or raised.
                exception = e
                log.error(traceback.format_exc())
            result = self.dset[()]
            self.get_measurement_endtime()
            self.save_MC_metadata(self.data_object)  # timing labels etc
            # FIXME: Nathan 2020.12.03.
            #  save_timer could alternatively be placed in quantum Experiment,
            #  which could make more sense semantically and would allow to get "run.end"
            #  in the MC timer (i.e. the end of this function).
            #  Qexp  needs to save its own timer anyway, but by having it here for now
            #  we're sure that experiment that are not based on Qexp still have some timers
            #  saved.
            self.save_timers(self.data_object)
            return_dict = self.create_experiment_result_dict()
            if exception is not None:  # exception occurred in above try-block
                if previous_attempts + 1 < self.max_attempts():
                    # Maximum number of attempts not reached. Log to logger
                    # and to slack, and retry.
                    msg = f'MC: Error during measurement (attempt ' \
                          f'{previous_attempts + 1} of {self.max_attempts()}' \
                          f'). Retrying.'
                    self.log_to_slack(msg)
                    log.error(msg)
                    # Call the retry_cleanup_functions if there are any (see
                    # docstring of parameter max_attempts).
                    [fnc() for fnc in getattr(
                        self, 'retry_cleanup_functions', [])]
                    # sweep points should be extracted again from the sweep
                    # function to avoid tiling them multiple times (in every
                    # attempt) in tile_sweep_pts_for_2D
                    del self.sweep_points
                    # Call run recursively to start the next attempt
                    return_dict = self.run(
                        name=name, exp_metadata=exp_metadata, mode=mode,
                        disable_snapshot_metadata=disable_snapshot_metadata,
                        previous_attempts=previous_attempts + 1, **kw)
                else:
                    # maximum number of attempts reached
                    raise exception

        self.finish(result)
        return return_dict

    @Timer()
    def measure(self):
        if self._live_plot_enabled():
            self.initialize_plot_monitor()

        self.timer.checkpoint("MeasurementControl.measure.prepare.start")
        for sweep_function in self.sweep_functions:
            sweep_function.prepare()
        self.timer.checkpoint("MeasurementControl.measure.prepare.end")

        if (self.sweep_functions[0].sweep_control == 'soft' and
                self.detector_function.detector_control == 'soft'):
            self.detector_function.prepare()
            self.get_measurement_preparetime()
            self.measure_soft_static()

        elif self.detector_function.detector_control == 'hard':
            self.get_measurement_preparetime()
            sweep_points = self.get_sweep_points()
            last_val = {}

            while self.get_percdone() < 100:
                start_idx = self.get_datawriting_start_idx()
                if len(self.sweep_functions) == 1:
                    sp = sweep_points[
                         :len(sweep_points) // self.soft_repetitions()]
                    self.sweep_functions[0].set_parameter(sp[start_idx])
                    self.detector_function.prepare(sweep_points=sp)
                    self.measure_hard()
                else:  # If mode is 2D
                    for i, sweep_function in enumerate(self.sweep_functions):
                        swf_sweep_points = sweep_points[:, i]
                        val = swf_sweep_points[start_idx]
                        if self.program_only_on_change() and \
                                last_val.get(i) == val:
                            continue
                        last_val[i] = val
                        # prepare in 2D sweeps is done in set_parameters (though not always
                        # for first upload). Therefore, a common checkpoint is used
                        # in the timer to collect upload times in a single place
                        self.timer.checkpoint("MeasurementControl.measure.prepare.start")
                        sweep_function.set_parameter(val)
                        self.timer.checkpoint("MeasurementControl.measure.prepare.end")
                    sp = sweep_points[start_idx:start_idx+self.xlen, 0]
                    sp = sp[:len(sp) // self.soft_repetitions()]
                    # If the soft sweep function has the filtered_sweep
                    # attribute, the AWGs are programmed in a way that only
                    # the subset of acquisitions indicated by the filter is
                    # performed. In this case, the sweep points passed to
                    # the detector function need to be filtered, and the
                    # filter needs to be passed to measure_hard.
                    filtered_sweep = getattr(self.sweep_functions[1],
                                             'filtered_sweep', None)
                    if filtered_sweep is not None:
                        sp = sp[filtered_sweep]
                    self.detector_function.prepare(sweep_points=sp)
                    self.measure_hard(filtered_sweep)
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
                self.measurement_function(sweep_point, index=i)

    @Timer()
    def measure_soft_adaptive(self, method=None):
        '''
        Uses the adaptive function and keywords for that function as
        specified in self.af_pars()

        FIXME this method is used in a very convoluted way:
         - the user passes an adaptive_function, which is the optimiser and
           itself expects a function returning data as its argument
         - the user passes data_processing_function, which is an analysis
         - measure_soft_adaptive calls the adaptive_function, passing
           optimization_function to it
         - optimization_function calls measurement_function (which calls
           the detectors), and data_processing_function, and returns values
         - adaptive_function gets the values, takes a decision, and returns
           an optimal (set of) sweep point(s)
         One possible cleaner way could be:
         - measure_soft_adaptive calls the adaptive_function, passing
           measurement_function to it
         - data_processing_function is part of the adaptive_function and
           just called by it
         - adaptive_function finally takes the decisions, and returns
           optimal sweep point(s)
         In addition, one could replace the adaptive_function by a base
         class, including a data_processing_function analysis and an
         optimisation method (defaulting to doing a min/max optimisation as
         currently in optimization_function).
        '''
        self.save_optimization_settings()
        self.adaptive_function = self.af_pars.pop('adaptive_function')
        self.data_processing_function = self.af_pars.pop(
            'data_processing_function', self._default_data_processing_function)
        if self._live_plot_enabled():
            self.initialize_plot_monitor()
            self.initialize_plot_monitor_adaptive()
        self.timer.checkpoint(
            "MeasurementControl.measure_soft_adaptive.prepare.start")
        for sweep_function in self.sweep_functions:
            sweep_function.prepare()
        if self.detector_function.detector_control != 'hard':
            # A hard detector requires sweep points, these will only be
            # generated later in measurement_function, which then takes care of
            # calling self.detector_function.prepare(sp).
            self.detector_function.prepare()
        self.timer.checkpoint(
            "MeasurementControl.measure_soft_adaptive.prepare.end")
        self.get_measurement_preparetime()

        if self.adaptive_function == 'Powell':
            self.adaptive_function = fmin_powell
        self.timer.checkpoint(
            "MeasurementControl.measure_soft_adaptive.adaptive_function.start")
        if callable(self.adaptive_function):
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
        self.timer.checkpoint(
            "MeasurementControl.measure_soft_adaptive.adaptive_function.end")
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

    @Timer()
    def measure_hard(self, filtered_sweep=None):
        """
        :param filtered_sweep: (None or list of bools) indicates which of the
            acquisition elements will be played by the AWGs (True) and which
            ones will be skipped (False). Default: None, in which case all
            acquisition elements will be played.
        """
        n_acquired = 0
        for i_rep in range(self.soft_repetitions()):
            # Tell the detector_function to call print_progress for intermediate
            # progress reports during get_detector_function.values.
            self.detector_function.progress_callback = (
                lambda x, n=n_acquired: self.print_progress(x + n))
            this_new_data = np.array(self.detector_function.get_values()).T
            n_acquired += this_new_data.shape[0]
            new_data = this_new_data if i_rep == 0 else np.concatenate(
                [new_data, this_new_data])
            if i_rep < self.soft_repetitions() - 1:
                self.print_progress(n_acquired)
        self.detector_function.progress_callback = None  # clean up

        if filtered_sweep is not None:
            # Extend the data array by adding NaN for data points that have
            # not been measured.
            shape = list(new_data.shape)
            shape[0] = len(filtered_sweep)
            new_data_full = np.zeros(shape) * np.nan
            if len(shape) > 1:
                new_data_full[filtered_sweep, :] = new_data
            else:
                new_data_full[filtered_sweep] = new_data
            new_data = new_data_full

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
                                 self._get_nr_sweep_point_columns()]
            new_vals = ((new_data + old_vals*self.soft_iteration) /
                        (1+self.soft_iteration))

            self.dset[start_idx:stop_idx,
                      self._get_nr_sweep_point_columns()] = new_vals
        else:
            old_vals = self.dset[start_idx:stop_idx,
                                 self._get_nr_sweep_point_columns():]
            new_vals = ((new_data + old_vals * self.soft_iteration) /
                        (1 + self.soft_iteration))
            self.dset[start_idx:stop_idx,
                      self._get_nr_sweep_point_columns():] = new_vals
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
                        start_idx:stop_idx]
                    self.dset[start_idx:stop_idx, 0:len(self.sweep_functions)] = \
                        relevant_swp_points
                else:
                    self.dset[start_idx:stop_idx, 0] = self.get_sweep_points()[
                        start_idx:start_idx+len_new_data:].T
            except Exception:
                # There are some cases where the sweep points are not
                # specified that you don't want to crash (e.g. on -off seq)
                log.warning('You are in the exception case in '
                            'MC.measure_hard() DATA STORING BLOCK section. '
                            'Something might have gone wrong with your '
                            'measurement.')
                log.warning(traceback.format_exc())

        self.check_keyboard_interrupt()
        self.update_instrument_monitor()
        self.update_plotmon()
        if self.mode == '2D':
            self.update_plotmon_2D_hard()
        self.iteration += 1
        self.print_progress()
        return new_data

    @Timer()
    def measurement_function(self, x, index=None):
        '''
        Core measurement function used for soft sweeps

        FIXME: not tested for len(self.sweep_functions) > 2
        '''
        if np.size(x) == 1:
            x = [x]
        # The len()==1 condition is a consistency check because batch_mode
        # is currently only implemented for the case of a single sweep
        # function (e.g., break in the if statement inside the for loop)
        batch_mode = (len(self.sweep_functions) == 1 and
                      self.sweep_functions[0].supports_batch_mode)
        if np.size(x) != len(self.sweep_functions) and not batch_mode:
            raise ValueError(
                'size of x "%s" not equal to # sweep functions' % x)
        # The following will be set to True (in the second-last iteration,
        # sweep dimension 1) if the sweep point can be skipped (in the
        # last iteration, sweep dimension 0) in a filtered sweep.
        filter_out = False
        for i, sweep_function in enumerate(self.sweep_functions[::-1]):
            if batch_mode:
                # Here, x corresponds to a tuple of circuit parameters or a
                # list of tuples of circuit parameters, see
                # `BlockSoftHardSweep` for details.
                x = np.atleast_2d(x)
                self.timer.checkpoint(
                    "MeasurementControl.measure_soft_adaptive"
                    ".adaptive_function.swf.set_parameter.start")
                sweep_function.set_parameter(x)
                self.timer.checkpoint(
                    "MeasurementControl.measure_soft_adaptive"
                    ".adaptive_function.swf.set_parameter.end")
                # Detector functions assume to receive the sweep points
                # tiled, according to the number in acq_data_len_scaling.
                # Example SSRO: acq_data_len_scaling equals the number of
                # shots and prepare will get a sweep point for each shot of
                # each segment, see IntegratingAveragingPollDetector.prepare.
                self.detector_function.prepare(
                    np.tile(x, self.acq_data_len_scaling))
                break
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
                # ::-1 as in the for loop (outer dimensions first)
                prev_swp_pt = self.last_sweep_pts[::-1][i]
                if swp_pt != prev_swp_pt and not filter_out:
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
            if index is not None and i == len(self.sweep_functions) - 2:
                # We are performing a static measurement and are in the
                # second-to-last iteration of the loop, i.e., second sweep
                # dimesion (according to the original ordering of sweep
                # dimensions from innermost to outermost).
                fsw = getattr(sweep_function, 'filtered_sweep', None)
                if fsw is not None:
                    # calculate the index within sweep dimension 0
                    xindex = index % self.xlen
                    # set filter_out to True if filtered_sweep indicates
                    # that the point can be skipped
                    filter_out = (xindex < len(fsw) and not fsw[xindex])

        # used for next iteration
        if filter_out and self.iteration > 0:
            # do not update dimension 0 if the sweep point was not set above
            self.last_sweep_pts[1:] = x[1:]
        else:
            self.last_sweep_pts = x
        datasetshape = self.dset.shape
        # self.iteration = datasetshape[0] + 1

        if filter_out:
            vals = np.ones(len(self.detector_function.value_names)) * np.nan
        else:
            vals = self.detector_function.acquire_data_point()

        if batch_mode:
            # FIXME: add an explaining comment why the transpose is needed
            vals = vals.T
        start_idx, stop_idx = self.get_datawriting_indices_update_ctr(vals)
        # Resizing dataset and saving

        new_datasetshape = (np.max([datasetshape[0], stop_idx]),
                            datasetshape[1])
        self.dset.resize(new_datasetshape)
        if batch_mode:
            # Because x is allowed to be a list of tuples (batch sampling), we
            # need to reshape and reformat x and vals accordingly before we can
            # save them to the dset.
            x = np.atleast_2d(x) # to unify format of x
            vals = vals.reshape((-1, len(self.detector_function.value_names)))
            # the following np.concatenate ensures that the measured values are
            # concatenated with the correct parameters in x.
            new_data = np.concatenate(
                (np.array(list(x) * int(vals.shape[0] / x.shape[0])), vals),
                axis=-1
            )
        else:
            # FIXME: the batch_mode code above is supposed to also treat the
            #  case without batch mode correctly. However, until someone
            #  verifies this rigorously (both for measure_soft_adaptive and for
            #  measure_soft_static with 1D, 2D, 3D sweeps) and adds explaining
            #  comments, we rather play safe and explicitly keep the
            #  previous implementation as else branch.
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
            self.print_progress()
        return vals

    @staticmethod
    def _default_data_processing_function(vals, dset):
        """Default data processing function for adaptive measurements.

        This is used in optimization_function (mode = adaptive) if no
        custom function is provided via af_pars['data_processing_function'].

        The default processing takes the first column if the data has two
        columns. A potential use case of this is for data consisting of
        magnitude and phase. In case of a single data column, the default
        processing is an identity operation.

        Args:
            vals (array): Array with the output of measurement_function.
            dset (array): The data set self.dset will be passed here. Not
                used in the default processing, but included as an argument
                to allow custom data processing functions to access the dset.
        """
        if len(np.shape(vals)) == 2:
            vals = np.array(vals)[:, 0]
        return vals

    @Timer()
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
        self.timer.checkpoint(
            "MeasurementControl.data_processing_function.start")
        vals = self.data_processing_function(vals, self.dset)
        self.timer.checkpoint(
            "MeasurementControl.data_processing_function.end")
        if self.minimize_optimization:
            if (self.f_termination is not None):
                if (vals < self.f_termination):
                    raise StopIteration()
        else:
            # when maximizing, interrupt when larger than condition before
            # inverting
            if (self.f_termination is not None):
                if (vals > self.f_termination):
                    raise StopIteration()
            vals = np.multiply(-1, vals)

        # to check if vals is an array with multiple values
        if hasattr(vals, '__iter__'):
            if len(vals) > 1 and self.par_idx is not None:
                vals = vals[self.par_idx]

        return vals

    def finish(self, result):
        '''
        Deletes arrays to clean up memory and avoid memory related mistakes
        '''
        # this data can be plotted by enabling persist_mode
        n = len(self.sweep_par_names)
        if self._live_plot_enabled():
            self._persist_dat = np.concatenate([
                result[:, :n],
                self.detector_function.live_plot_transform(result[:, n:])
            ], axis=1)
            self._persist_xlabs = self.sweep_par_names
            self._persist_ylabs = self.detector_function.live_plot_labels
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
    def _live_plot_enabled(self):
        if hasattr(self, 'detector_function') and \
                not getattr(self.detector_function, 'live_plot_allowed', True):
            return False
        else:
            return self.live_plot_enabled()

    def open_plotmon_windows(self, name=None, close_previous_windows=True):
        """Opens the windows of the main and secondary plotting monitor.

        This method is called in the init if live_plot_enabled is True,
        and it can also be called by the user, e.g., if the windows have
        been closed by accident.

        Args:
            name (str): A name to be shown in the title bar of the windows.
                Defaults to None, in which case the name of the MC object is
                used.
            close_previous_windows (bool): Specifies whether the previously
                used plotmon windows should be closed before creating the
                new ones. Default: True.
        """
        if close_previous_windows:
            for plotmon in ['main_QtPlot', 'secondary_QtPlot']:
                try:
                    getattr(self, plotmon).win.close()
                except Exception:
                    # Either the window did not exist, or an error occured.
                    # This can be ignored: in the worst case, an unused
                    # window will stay open.
                    pass
        if name is None:
            name = self.name
        self.main_QtPlot = QtPlot(
            window_title=f'Main plotmon of {name}', figsize=(600, 400))
        self.secondary_QtPlot = QtPlot(
            window_title=f'Secondary plotmon of {name}', figsize=(600, 400))

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
        zlabels = self.detector_function.live_plot_labels
        zunits = self.detector_function.live_plot_units

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
                mo = vnmom.get(vn, None) if vnmom is not None else None
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
                        if len(new_sweep_vals[i]) == 1:  # single sweep point
                            continue  # the following check is not needed
                        # Check if the new_sweep_vals are not equidistant
                        # or if they have all the same value
                        diff = np.diff(new_sweep_vals[i])
                        # To check for equidistant points, the distances
                        # are normalized to the distance between the first
                        # two points. The normalization is needed to choose a
                        # meaningful threshold for distinguishing
                        # non-equdistant points from rounding errors.
                        # The order matters: we need to first check whether
                        # the first two points are the same, and then check
                        # for equidistant points. Otherwise, we might get a
                        # division by zero error during the normalization.
                        if np.abs(diff[0]) == 0 or any(
                                [np.abs(d - diff[0]) / np.abs(diff[0]) > 1e-5
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
        if self._live_plot_enabled() and (self.dset.shape[0] <
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
                    ydata = self.detector_function.live_plot_transform(
                        self.dset[:, nr_sweep_funcs:])
                    for y_ind, vn in enumerate(
                            self.detector_function.value_names):
                        # get the axis info for this column
                        axes_info = self._plotmon_axes_info[vn]
                        # The first nr_sweep_funcs columns are sweep values
                        # (x axis in 1D plots), the following columns are
                        # data columns (y axis in 1D plots).
                        y = ydata[:, y_ind]
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
        if self._live_plot_enabled() and self.live_plot_2D_update() != 'off':
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

    @Timer()
    def update_plotmon_2D(self, force_update=False):
        '''
        Adds latest measured value to the TwoD_array and sends it
        to the QC_QtPlot.
        '''
        if self._live_plot_enabled() and self.live_plot_2D_update() != 'off':
            try:
                i = int((self.iteration) % (self.xlen*self.ylen))
                x_ind = int(i % self.xlen)
                y_ind = int(i / self.xlen)
                self.TwoD_array[y_ind, x_ind, :] = \
                    self.detector_function.live_plot_transform(
                        self.dset[i, len(self.sweep_functions):])
                for j in range(len(self.detector_function.value_names)):
                    self.secondary_QtPlot.traces[j]['config'][
                        'z'] = self.TwoD_array[:, :, j]
                if (time.time() - self.time_last_2Dplot_update >
                        self.plotting_interval()
                        or self.iteration == len(self.sweep_points) or
                        force_update) and (
                        self.live_plot_2D_update() != 'row'
                        or (not (self.iteration + 1) % self.xlen)):
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

        if self._live_plot_enabled():
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

        if self._live_plot_enabled():
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

    @Timer()
    def update_plotmon_2D_hard(self):
        '''
        Adds latest datarow to the TwoD_array and send it
        to the QC_QtPlot.
        Note that the plotmon only supports evenly spaced lattices.
        '''
        try:
            if self._live_plot_enabled() and self.live_plot_2D_update() != 'off':
                if self.cyclic_soft_avg():
                    i = int(self.iteration % self.ylen)
                else:
                    i = int(self.iteration // self.soft_avg())

                cf = self.exp_metadata.get('compression_factor', 1)
                data = self.detector_function.live_plot_transform(
                    self.dset[i * self.xlen:(i + 1) * self.xlen,
                              len(self.sweep_functions):])
                for j in range(len(self.detector_function.value_names)):
                    data_row = data[:, j]
                    if cf != 1:
                        # reshape data according to compression factor
                        data_reshaped = data_row.reshape((cf, int(len(data_row)/cf)))
                        y_start = self.iteration*cf % self.TwoD_array.shape[0]
                        y_end = y_start + cf
                        self.TwoD_array[y_start:y_end, :, j] = data_reshaped
                    else:
                        self.TwoD_array[i, :, j] = data_row
                    self.secondary_QtPlot.traces[j]['config']['z'] = \
                        self.TwoD_array[:, :, j]
                is_last_it = self.iteration + 1 == \
                             (len(self.get_sweep_points()) * self.soft_avg()) \
                             // self.xlen
                if (time.time() - self.time_last_2Dplot_update >
                        self.plotting_interval() or is_last_it):
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

    def _get_experimentaldata_group(self):
        if EXPERIMENTAL_DATA_GROUP_NAME in self.data_object:
            return self.data_object[EXPERIMENTAL_DATA_GROUP_NAME]
        else:
            return self.data_object.create_group(EXPERIMENTAL_DATA_GROUP_NAME)

    def _get_create_dataset_kwargs(self):
        """
        Get the kwargs to pass to create_dataset in
        create_experimentaldata_dataset and save_extra_data.

        Returns:
            dict with the kwargs
        """
        kwargs = {}
        if self.compress_dataset():
            kwargs.update({'compression': "gzip", 'compression_opts': 9})
        return kwargs

    def _get_nr_sweep_point_columns(self):
        return np.sum([sweep_function.get_nr_parameters() \
            for sweep_function in self.sweep_functions])

    def create_experimentaldata_dataset(self):
        data_group = self._get_experimentaldata_group()
        self.dset = data_group.create_dataset(
            'Data', (0, self._get_nr_sweep_point_columns() +
                     len(self.detector_function.value_names)),
            maxshape=(None, self._get_nr_sweep_point_columns() +
                      len(self.detector_function.value_names)),
            dtype='float64', **self._get_create_dataset_kwargs())
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

    def save_extra_data(self, group_name, dataset_name, data,
                        column_names=None):
        """Save further data in addition to the Data table

        This can, e.g., be used to save raw data in cases where the data in
        the Data table is processed in software, or data that is provided by
        the acquisition instrument in addition to the main measurement data.
        Note that the method can only be used during a run of MC, i.e.,
        while self.data_object is open.

        Note that this method is provided as a callback function to
        self.detector_function.

        Args:
            group_name (str): Name of the subgroup inside experimental data
                group. The subgroup is automatically created if it does
                not exist.
            dataset_name (str): The name of the dataset in which the data
                should be stored. If the dataset exists already, the shape
                of the new data must be compatible with the dimensions of
                the existing dataset such that it can be appended in axis 0.
                Note that the name can contain slashes indicating a
                hierarchy of subgroups with only the part behind the last
                slash interpreted as dataset name.
            data (np.array): the data to be stored
            column_names (None or list of str): names of the columns of the
                data array. If this is not None and a new dataset is
                created, the list is stored as an attribute column_names of
                the new dataset.
        """
        data_group = self._get_experimentaldata_group()
        if group_name in data_group:
            group = data_group[group_name]
        else:
            group = data_group.create_group(group_name)
        if dataset_name not in group:
            dset = group.create_dataset(dataset_name, data=data,
                                        maxshape=[None] * len(data.shape),
                                        **self._get_create_dataset_kwargs())
            if column_names is not None:
                dset.attrs['column_names'] = h5d.encode_to_utf8(column_names)
        else:
            dset = group[dataset_name]
            # FIXME: we should check whether dimensions and column names
            #  are the same
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0]:] = data

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
                in self.data_object[EXPERIMENTAL_DATA_GROUP_NAME].keys()):
            opt_res_grp = self.data_object[EXPERIMENTAL_DATA_GROUP_NAME]
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

    def save_instrument_settings(self, data_object=None, mode='xb', *args):
        '''
        uses QCodes station snapshot to save the last known value of any
        parameter. Only saves the value and not the update time (which is
        known in the snapshot)
        File format in which the snapshot is saved is specified in parameter
        'settings_file_format'.
        mode: define which mode you want to open the file in.
            Default 'xb' creates the file and returns error if file exist
            'wb' to overwrite existing file
        '''

        if self.settings_file_format() == 'hdf5':
            def save_settings_in_hdf(data_object):
                if not hasattr(self, 'station'):
                    log.warning('No station object specified, could not save '
                                'instrument settings')
                else:
                    # # This saves the snapshot of the entire setup
                    # snap_grp = data_object.create_group('Snapshot')
                    # snap = self.station.snapshot()
                    # h5d.write_dict_to_hdf5(snap, entry_point=snap_grp)

                    # Below is old style saving of snapshot, exists for the sake of
                    # preserving deprecated functionality. Here only the values
                    # of the parameters are saved.
                    set_grp = data_object.create_group('Instrument settings')
                    inslist = dict_to_ordered_tuples(self.station.components)
                    for (iname, ins) in inslist:
                        instrument_grp = set_grp.create_group(iname)
                        inst_snapshot = ins.snapshot()
                        MeasurementControl.store_snapshot_parameters(
                            inst_snapshot,
                            entry_point=instrument_grp,
                            instrument=ins)
                numpy.set_printoptions(**opt)
            import numpy
            import sys
            opt = numpy.get_printoptions()
            numpy.set_printoptions(threshold=sys.maxsize)
            if data_object is None:
                data_object = self.data_object
            with numpy.printoptions(threshold=sys.maxsize):
                # checks if data object is closed and opens it if necessary in ,
                # a context manager, such that it is closed after save method.
                if not data_object.__bool__():
                    with h5d.Data(name=self.get_measurement_name(),
                      datadir=self.datadir(),
                      timestamp=self.last_timestamp(),
                                           auto_increase=False) as data_object:
                        MeasurementControl._save_station_in_hdf(data_object,
                                                                self.station)
                else:
                    # hdf file was already opened and does not need to be
                    # closed at the end, because save_instrument_settings was
                    # called inside a context manager and may be used
                    # after calling save_instrument_settings (e.g. MC.run())
                    MeasurementControl._save_station_in_hdf(data_object,
                                                            self.station)

        else:
            if self.settings_file_format() == 'msgpack':
                from pycqed.utilities.io.msgpack import MsgDumper as Dumper
            elif self.settings_file_format() == 'pickle':
                from pycqed.utilities.io.pickle import PickleDumper as Dumper
            else:
                raise NotImplementedError(
                    f"Format '{self.settings_file_format()}' not known.")

            snapshot = self.station.snapshot()

            # QCodes uses two containers for the snapshots of station.components
            # objects if called via station.snapshot: 'instruments' if the
            # object is an qcodes instrument else 'components'
            # E.g. remote instruments are not recognized as qcodes instruments and
            # must therefore be moved to staion['instruments'] by hand.
            # These kind of 'special' instruments all inherit from
            # pycqedins.FurtherInstrumentsDictMixIn.
            for k in list(snapshot['components'].keys()):
                if isinstance(self.station.components[k],
                              pycqedins.FurtherInstrumentsDictMixIn):
                    snapshot['instruments'][k] = snapshot['components'].pop(k)

            dumper = Dumper(name=self.get_measurement_name(),
                            datadir=self.datadir(),
                            data=snapshot,
                            compression=self.settings_file_compression(),
                            timestamp=self.last_timestamp())
            dumper.dump(mode=mode)

    @staticmethod
    def store_snapshot_parameters(inst_snapshot, entry_point,
                                  instrument):
        """
        Save the values of keys in the "parameters" entry of inst_snapshot.
        If inst_snapshot contains "submodules," a new subgroup inside the
        instrument group will be created for each key in "submodules" and the
        values of the "parameters" entry will be stored.
        :param inst_snapshot: (dict) snapshot of a QCoDeS instrument
        :param entry_point: (hdf5 group.file) location in the nested hdf5
            structure where to write to.
        :param instrument: (obj) instrument whose snapshot is saved (or
            submodule/channel in recursive calls)
        """

        # If a whitelist exists, we store only children (submodules, channels,
        # parameters) that are on the whitelist. Whitelisted submodules and
        # channels are by default stored including all their children,
        # except if the submodule/channel again has a whitelist specifying
        # that only a subset of its children should be stored.
        sp_wl = getattr(instrument, '_snapshot_whitelist', None)

        for k in ['submodules', 'channels']:
            if k not in inst_snapshot:
                continue
            # store the parameters from the items in submodules, which
            # are snapshots of QCoDeS instruments
            for key, submod_snapshot in inst_snapshot[k].items():
                if k == 'channels':
                    subins = [ch for ch in instrument if ch.name == key][0]
                else:  # submodules
                    subins = instrument.submodules[key]
                if getattr(subins, 'snapshot_exclude', False):
                    # qcodes does not implement snapshot_exclude for
                    # submodules and channels, so we implement it here.
                    continue
                if sp_wl is not None and key not in sp_wl:
                    # If a whitelist exists, only include submodules/channels
                    # that are in the snapshot_whitelist
                    continue
                submod_grp = entry_point.create_group(key)
                MeasurementControl.store_snapshot_parameters(
                    submod_snapshot, entry_point=submod_grp, instrument=subins)

        if 'parameters' in inst_snapshot:
            par_snap = inst_snapshot['parameters']
            parameter_list = dict_to_ordered_tuples(par_snap)
            for (p_name, p) in parameter_list:
                if sp_wl is not None and p_name not in sp_wl:
                    # If a whitelist exists, only include parameters that are
                    # in the snapshot_whitelist.
                    continue
                val = repr(p.get('value', ''))
                entry_point.attrs[p_name] = val
        entry_point.attrs["__class__"] = inst_snapshot['__class__']

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
        set_grp.attrs['live_plot_enabled'] = self._live_plot_enabled()
        sha1_id, diff = self.get_git_info()
        set_grp.attrs['git_sha1_id'] = sha1_id
        set_grp.attrs['git_diff'] = diff
        set_grp.attrs['pycqed_version'] = pycqed.version.__version__

    def save_exp_metadata(self, metadata: dict):
        '''
        Saves experiment metadata to the data file. The metadata is saved at
            file['Experimental Data']['Experimental Metadata']

        Args:
            metadata (dict):
                    Simple dictionary without nesting. An attribute will be
                    created for every key in this dictionary.
        '''
        data_group = self._get_experimentaldata_group()

        if 'Experimental Metadata' in data_group:
            metadata_group = data_group['Experimental Metadata']
        else:
            metadata_group = data_group.create_group('Experimental Metadata')

        h5d.write_dict_to_hdf5(metadata, entry_point=metadata_group)

    def save_timers(self, data_object, detector=True, MC=True):
        timer_group = data_object.get(Timer.HDF_GRP_NAME)
        if timer_group is None:
            timer_group = data_object.create_group(Timer.HDF_GRP_NAME)
        objects = (self.detector_function, self)
        for obj, cond in zip(objects, (detector, MC)):
            try:
                obj.timer.save(timer_group)
            except Exception:
                log.error(f"Could not save timer for object: {obj}.")
                traceback.print_exc()

    def add_parameter_check(self, parameter, check_function):
        """
        Configure a parameter check that shall be performed at the start of
        every measurement.
        :param parameter: (qcodes parameter) the parameter for which a
            check should be added
        :param check_function: (function) a function that takes the
            parameter value as input and returns True in case of a
            successful check (e.g., if the value of the parameter is within
            an expected range), and otherwise False or a string that
            explains the unsuccessful check.
        """
        self.parameter_checks[parameter] = check_function

    def remove_parameter_check(self, parameter):
        """
        Remove (a) parameter check(s).
        :param parameter: (qcodes parameter or list thereof) the parameter(s)
            for which the check(s) should be removed
        """
        if isinstance(parameter, list):
            [self.remove_parameter_check(p) for p in parameter]
        elif parameter in self.parameter_checks:
            self.parameter_checks.pop(parameter)
        else:
            log.warning(f'No check was configured for parameter {parameter}.')

    def perform_parameter_checks(self, update=False):
        """Perform all parameter checks added via add_parameter_check

        Args:
            update (bool or None): Whether the parameter should be updated,
            see docstring of _BaseParameter.snapshot_base in qcodes.
        """
        for p, check_function in self.parameter_checks.items():
            try:
                val = p.snapshot(update=update)['value']
                res = check_function(val)
                if res != True:  # False or a string (error message)
                    log.warning(
                        f'Parameter {p.full_name} has an uncommon value: '
                        f'{val}.' + (f" ({res})" if res is not False else ''))
            except Exception as e:
                log.warning(
                    f'Could not run parameter check for {p}: {e}')


    def get_percdone(self, current_acq=0):
        """
        Determine the current progress (in percent) based on how much data
        MC has already received and possibly based on the progress of the
        current acquisition.

        In addition, get_percdone monitors whether progress has been made
        since the last call to get_percdone, and it can log a warning to
        slack and/or raise an exception if no progress is made for a
        specified number of seconds (specified by the qcodes parameters
        no_progress_interval and no_progress_kill_interval, respectively).

        :param current_acq: number of acquired samples in the current
            acquisition. (Example: if 40 samples with averaging over 2**10
            will be acquired in the current acquisition, and the current
            averaging progress is 300, then current_acq should be set to
            40*300/2**10.)
        """
        if current_acq == np.nan:
            # np.nan indicates that progress reporting is in principle
            # supported, but failed. This will be treated like no progress.
            percdone = np.nan
        else:
            percdone = (self.total_nr_acquired_values + current_acq) / (
                np.shape(self.get_sweep_points())[0] * self.soft_avg()) * 100
        try:
            now = time.time()
            if percdone != np.nan and percdone != self._last_percdone_value:
                # progress was made
                self._last_percdone_value = percdone
                self._last_percdone_change_time = now
                log.debug(f'MC: percdone = {self._last_percdone_value} at '
                          f'{self._last_percdone_change_time}')
            elif self._last_percdone_change_time == 0:
                # first progress check: initialize _last_percdone_change_time
                self._last_percdone_change_time = now
                self._last_percdone_log_time = self._last_percdone_change_time
            else:  # no progress was made
                no_prog_inter = self.no_progress_interval()
                no_prog_inter2 = self.no_progress_kill_interval()
                no_prog_min = (now - self._last_percdone_change_time) / 60
                log.debug(f'MC: no_prog_min = {no_prog_min}, '
                          f'percdone = {percdone}')
                msg = f'The current measurement has not made any progress ' \
                      f'for {no_prog_min: .01f} minutes.'
                if now - self._last_percdone_change_time > no_prog_inter \
                        and now - self._last_percdone_log_time > no_prog_inter:
                    log.warning(msg)
                    self.log_to_slack(msg)
                    self._last_percdone_log_time = now
                if now - self._last_percdone_change_time > no_prog_inter2:
                    log.debug(f'MC: raising NoProgressError')
                    raise NoProgressError(msg)
        except NoProgressError:
            raise
        except Exception as e:
            log.debug(f'MC: error while checking progress: {repr(e)}')
        return percdone

    def print_progress(self, current_acq=0):
        """
        Prints the progress of the current measurement.

        :param current_acq: see docstring of get_percdone
        """
        # The following method is always called because it includes the
        # no-progress checks.
        percdone = self.get_percdone(current_acq=current_acq)
        if self.verbose() and percdone != np.nan:
            elapsed_time = time.time() - self.begintime
            t_left = round((100. - percdone) / (percdone) *
                           elapsed_time, 1) if percdone != 0 else '??'
            t_end = time.strftime('%H:%M:%S', time.localtime(time.time() +
                                  + t_left)) if percdone != 0 else '??'
            # The trailing spaces are to overwrite some characters in case the
            # previous progress message was longer. (Due to \r, the string
            # output will start at the beginning of the current line and
            # each character of the new string will overwrite a character
            # of the previous output in the current line.)
            progress_message = (
                "\r{timestamp}\t{percdone}% completed \telapsed time: "
                "{t_elapsed}s \ttime left: {t_left}s\t(until {t_end})     "
                "").format(
                    timestamp=time.strftime('%H:%M:%S', time.localtime()),
                    percdone=int(percdone),
                    t_elapsed=round(elapsed_time, 1),
                    t_left=t_left,
                    t_end=t_end,)

            if percdone != 100 or current_acq:
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

        if self.detector_function.detector_control == 'hard' and \
                len(self.sweep_functions) > 1 and \
                not self.cyclic_soft_avg():
            y_ind = ((self.total_nr_acquired_values // self.xlen)
                     // self.soft_avg())
            start_idx = y_ind * self.xlen
            self.soft_iteration = ((self.total_nr_acquired_values // self.xlen)
                                   % self.soft_avg())
        else:
            start_idx = int(
                self.total_nr_acquired_values % max_sweep_points)
            self.soft_iteration = int(
                self.total_nr_acquired_values // max_sweep_points)

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
        self.detector_function.extra_data_callback = self.save_extra_data

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
                                    specifies which one to use. If set to
                                    None, there will be no selection and all
                                    values are passed on.
            "data_processing_function":    function. Overwrites
                                           _default_data_processing_function

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

    def log_to_slack(self, message):
        """
        Send a message to Slack. If self.slack_webhook is not set,
        the message is only logged in the logger with loglevel INFO.
        If self.slack_channel is not set, the default channel of the webhook
        is used.

        :param message: The message that should be logged to slack.

        Note: The webhook and the channel are properties and not parameters,
        to avoid that they get stored in the instruments settings snapshot.
        """
        log.info(f'MC: {message}')
        if not hasattr(self, 'slack_webhook'):
            log.info(f'MC: Not logging to slack because slack_webhook is not '
                     f'defined.')
            return
        try:
            payload = {"text": message}
            if hasattr(self, 'slack_channel'):
                payload["channel"] = self.slack_channel
            res = requests.post(self.slack_webhook, json=payload)
            res_text = res.text
        except Exception as e:
            res_text = repr(e)
        if res_text != 'ok':
            log.warning(f'MC: Error while logging to slack: {res_text}')


class KeyboardFinish(KeyboardInterrupt):
    """
    Indicates that the user safely aborts/finishes the experiment.
    Used to finish the experiment without raising an exception.
    """
    pass
