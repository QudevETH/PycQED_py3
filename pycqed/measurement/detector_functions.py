"""
Module containing a collection of detector functions used by the
Measurement Control.
"""
import traceback
import numpy as np
from copy import deepcopy
import time
from string import ascii_uppercase
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.utilities.timer import Timer
from qcodes.instrument.parameter import _BaseParameter
from qcodes.instrument.base import Instrument
from pycqed.utilities.errors import NoProgressError
from pycqed.measurement.waveform_control import pulsar as ps
import logging
log = logging.getLogger(__name__)


class Detector_Function(object):

    '''
    Detector_Function class for MeasurementControl
    '''

    def __init__(self, **kw):
        self.name = self.__class__.__name__
        self.set_kw()
        self.value_names = ['val A', 'val B']
        self.value_units = ['arb. units', 'arb. units']
        # to be used by MC.get_percdone() and the IntegratingAveragingPollDetector
        self.acq_data_len_scaling = 1
        self.timer = Timer(self.name)
        # The following properties are not implemented in all detector
        # functions (i.e., might be ignored in some detector functions),
        # but are created here to have a common interface.
        self.progress_callback = kw.get('progress_callback', None)
        self.progress_callback_interval = kw.get(
            'progress_callback_interval', 5)  # in seconds
        # tells MC whether to show live plotting for the measurement
        self.live_plot_allowed = kw.get('live_plot_allowed', True)
        self.live_plot_transform_type = kw.get('live_plot_transform_type',
                                               None)
        self.extra_data_callback = None

    def set_kw(self, **kw):
        '''
        convert keywords to attributes
        '''
        for key in list(kw.keys()):
            exec('self.%s = %s' % (key, kw[key]))

    def get_values(self):
        pass

    def prepare(self, **kw):
        pass

    def finish(self, **kw):
        pass

    def generate_metadata(self):
        """
        Creates a dict det_metadata with all the attributes of itself.
        :return: {'Detector Metadata': det_metadata}
        """
        try:
            # Go through all the attributes of itself, pass them to
            # savable_attribute_value, and store them in det_metadata
            det_metadata = {k: self.savable_attribute_value(v, self.name)
                            for k, v in self.__dict__.items()}

            # Change the 'detectors' entry from a list of dicts to a dict with
            # keys uhfName_detectorName
            detectors_dict = {}
            for d in det_metadata.pop('detectors', []):
                # isinstance(d, dict) only if self was a multi-detector function
                if isinstance(d, dict):
                    # d will never contain the key "detectors" because the
                    # framework currently does not allow to pass an instance of
                    # UHFQC_multi_detector in the "detectors" attribute of
                    # UHFQC_Base since UHFQC_multi_detector does not have the
                    # attribute "UHFQC" (Steph, 23.10.2020)
                    if 'acq_devs' in d:
                        # d["acq_devs"] will always contain one item because of how
                        # savable_attribute_value was written.
                        detectors_dict.update(
                            {f'{d["acq_devs"][0]} {d["name"]}': d})
                    else:
                        detectors_dict.update({f'{d["name"]}': d})
                elif isinstance(d, str):
                    # In a single detector we only have 'detectors': [self.name]
                    # This line ensures that each single detector has an item
                    # 'detector': self.name in its saved metadata, whether or
                    # not it is contained in a MultiPollDetector.
                    # This should probably be cleaned up once we start using
                    # the detector metadata more in the analysis.
                    detectors_dict = [d]
                    break
            if len(detectors_dict):
                det_metadata['detectors'] = detectors_dict

            return {'Detector Metadata': det_metadata}
        except Exception:
            # Unhandled errors in metadata creation are not critical for the
            # measurement, so we log them as warnings.
            log.warning(traceback.format_exc())
            return {}

    @staticmethod
    def savable_attribute_value(attr_val, det_name):
        """
        Helper function for converting the attribute of a Detector_Function
        (or its children) to a format that will make the entry more meaningful
        when saved to an hdf file.
        In particular,  this function makes sure that if any of the det_func
        attributes are class instances (like det_func.AWG), they are passed to
        the metadata as class_instance.name instead of class_instance, in which
        case it would be saves as a string "<Pulsar: Pulsar>".

        This function also nicely resolves the detectors attribute of the
        detector functions, which would otherwise also be saved as
        ["<pycqed.measurement.detector_functions.UHFQC_classifier_detector
         at 0x22bf280a400>",
         "<pycqed.measurement.detector_functions.UHFQC_classifier_detector
         at 0x22bf280a208>"].
         It parses this list and replaces each instance with its __dict__
         attribute.

        :param attr_val: attribute value of a Detector_Function instance or
            an instance of its children
        :param det_name: name of a Detector_Function instance or an instance
            of its children
        :return: converted attribute value
        """
        if isinstance(attr_val, Detector_Function):
            if hasattr(attr_val, 'detectors') and \
                    det_name != attr_val.detectors[0].name:
                return {k: Detector_Function.savable_attribute_value(
                    v, attr_val.name)
                    for k, v in attr_val.__dict__.items()}
            else:
                return attr_val.name
        elif isinstance(attr_val, Instrument):
            try:
                return attr_val.name
            except AttributeError:
                return repr(attr_val)
        elif callable(attr_val):
            return repr(attr_val)
        elif isinstance(attr_val, (list, tuple)):
            return [Detector_Function.savable_attribute_value(av, det_name)
                    for av in attr_val]
        else:
            return attr_val

    def live_plot_transform(self, data):
        """Transform data for the liveplot

        Args:
            data: array of shape (i, ) or (n, i) with i being the
                number of data points per sweep point and n being an arbitrary
                integer.
        """
        if self.live_plot_transform_type == 'mag_phase':
            x = np.atleast_2d(data.T).T
            y = np.zeros_like(x)
            x = x[:, ::2] + 1j * x[:, 1::2]
            y[:, ::2], y[:, 1::2] = np.abs(x), np.angle(x)
            return y.reshape(data.shape)
        else:
            return data

    @property
    def live_plot_labels(self):
        if self.live_plot_transform_type == 'mag_phase':
            return [f'{x}_{l}'
                    for x in self.value_names[::2] for l in ['mag', '_phase']]
        else:
            return self.value_names

    @property
    def live_plot_units(self):
        return self.value_units


class Multi_Detector(Detector_Function):
    """
    Combines several detectors of the same type (hard/soft) into a single
    detector.
    """

    def __init__(self, detectors: list,
                 det_idx_suffix: bool=True, **kw):
        """
        detectors     (list): a list of detectors to combine.
        det_idx_suffix(bool): if True suffixes the value names with
                "_det{idx}" where idx refers to the relevant detector.
        """
        self.detectors = detectors
        self.name = 'Multi_detector'
        self.value_names = []
        self.value_units = []
        self.detector_values_length = []
        for i, detector in enumerate(detectors):
            for detector_value_name in detector.value_names:
                if det_idx_suffix:
                    detector_value_name += '_det{}'.format(i)
                self.value_names.append(detector_value_name)
            self.detector_values_length.append([len(detector.value_names)])
            for detector_value_unit in detector.value_units:
                self.value_units.append(detector_value_unit)

        self.detector_control = self.detectors[0].detector_control
        for d in self.detectors:
            if d.detector_control != self.detector_control:
                raise ValueError('All detectors should be of the same type')

    def prepare(self, **kw):
        for detector in self.detectors:
            detector.prepare(**kw)

    def get_values(self):
        values_list = []
        for detector in self.detectors:
            new_values = detector.get_values()
            values_list.append(new_values)
        values = np.concatenate(values_list)
        return values

    def acquire_data_point(self):
        # N.B. get_values and acquire_data point are virtually identical.
        # the only reason for their existence is a historical distinction
        # between hard and soft detectors that leads to some confusing data
        # shape related problems, hence the append vs concatenate
        values = []
        for detector in self.detectors:
            new_values = detector.acquire_data_point()
            values = np.append(values, new_values)
        return values

    def finish(self):
        for detector in self.detectors:
            detector.finish()

    def live_plot_transform(self, data):
        original_shape = data.shape
        data = np.atleast_2d(data)
        ind = 0
        for i, d in enumerate(self.detectors):
            data[:, ind:ind + self.detector_values_length[i]] \
                = d.live_plot_transform(data[:, ind:ind + self.detector_values_length[i]])
            ind += self.detector_values_length[i]
        return data.reshape(original_shape)


class IndexDetector(Detector_Function):
    """Detector function that indexes the result of another detector function.

    Args:
        detector:
            detector function that returns multiple values per sweep point
        index:
            Index if the element returned from the original detector function
            output. Can be an integer or a tuple. If an integer, then value is
            interpreted as a channel index of the original detector function.
            In case of a tuple, the first value corresponds to a channel index
            and the second value corresponds to an index in the original
            hardware sweep. Using a tuple converts a hardware sweep to a
            software sweep.
    """

    def __init__(self, detector, index):
        super().__init__()
        self.detector = detector
        self.index = index
        self.name = detector.name + '[{}]'.format(index)
        if isinstance(self.index, tuple):
            self.value_names = [detector.value_names[index[0]]]
            self.value_units = [detector.value_units[index[0]]]
            self.detector_control = 'soft'
        else:
            self.value_names = [detector.value_names[index]]
            self.value_units = [detector.value_units[index]]
            self.detector_control = detector.detector_control
            self.index = [index]

    def prepare(self, **kw):
        self.detector.prepare(**kw)

    def get_values(self):
        v = self.detector.get_values()
        # equivalent to v[self.index[0]][self.index[1]]...[self.index[-1]]
        for i in self.index:
            v = v[i]
        return v

    def acquire_data_point(self):
        v = self.detector.get_values()
        # equivalent to v[self.index[0]][self.index[1]]...[self.index[-1]]
        for i in self.index:
            v = v[i]
        return v

    def finish(self):
        self.detector.finish()


class SumDetector(Detector_Function):
    """A detector function that adds up the channel values of another detector

    Args:
        detector: Underlying detector with several channels to be added
        indices: Channel indices of the underlying detector that will be added
    """

    def __init__(self, detector, indices=None):
        super().__init__()
        self.detector = detector
        if indices is None:
            indices = np.arange(len(detector.value_names))
        self.indices = indices
        self.name = detector.name + '_sum'
        self.value_names = [detector.value_names[indices[0]]]
        self.value_units = [detector.value_units[indices[0]]]
        self.detector_control = detector.detector_control

    def prepare(self, **kw):
        self.detector.prepare(**kw)

    def get_values(self):
        return [np.array(self.detector.get_values())[self.indices]
                .sum(axis=0)]

    def acquire_data_point(self):
        return [np.array(self.detector.acquire_data_point())[self.indices]
                .sum(axis=0)]

    def finish(self):
        self.detector.finish()

###############################################################################
###############################################################################
####################             None Detector             ####################
###############################################################################
###############################################################################


class None_Detector(Detector_Function):

    def __init__(self, **kw):
        super(None_Detector, self).__init__()
        self.detector_control = 'soft'
        self.set_kw()
        self.name = 'None_Detector'
        self.value_names = ['None']
        self.value_units = ['None']

    def acquire_data_point(self, **kw):
        '''
        Returns something random for testing
        '''
        return np.random.random()


class Hard_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_control = 'hard'

    def prepare(self, sweep_points=None):
        pass

    def finish(self):
        pass


class Soft_Detector(Detector_Function):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.detector_control = 'soft'

    def acquire_data_point(self, **kw):
        return np.random.random()

    def prepare(self, sweep_points=None):
        pass


##########################################################################
##########################################################################
####################     Hardware Controlled Detectors     ###############
##########################################################################
##########################################################################


class Dummy_Detector_Hard(Hard_Detector):

    def __init__(self, delay=0, noise=0, **kw):
        super(Dummy_Detector_Hard, self).__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['distance', 'Power']
        self.value_units = ['m', 'W']
        self.delay = delay
        self.noise = noise
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points
        noise = self.noise * (np.random.rand(2, len(x)) - .5)
        data = np.array([np.sin(x / np.pi),
                         np.cos(x/np.pi)])
        data += noise
        time.sleep(self.delay)
        # Counter used in test suite to test how many times data was acquired.
        self.times_called += 1

        return data

class Dummy_Shots_Detector(Hard_Detector):

    def __init__(self, max_shots=10, **kw):
        super().__init__()
        self.set_kw()
        self.detector_control = 'hard'
        self.value_names = ['shots']
        self.value_units = ['m']
        self.max_shots = max_shots
        self.times_called = 0

    def prepare(self, sweep_points):
        self.sweep_points = sweep_points

    def get_values(self):
        x = self.sweep_points

        start_idx = self.times_called*self.max_shots % len(x)

        dat = x[start_idx:start_idx+self.max_shots]
        self.times_called += 1
        return dat


class Sweep_pts_detector(Detector_Function):

    """
    Returns the sweep points, used for testing purposes
    """

    def __init__(self, params, chunk_size=80):
        self.detector_control = 'hard'
        self.value_names = []
        self.value_units = []
        self.chunk_size = chunk_size
        self.i = 0
        for par in params:
            self.value_names += [par.name]
            self.value_units += [par.units]

    def prepare(self, sweep_points):
        self.i = 0
        self.sweep_points = sweep_points

    def get_values(self):
        return self.get()

    def acquire_data_point(self):
        return self.get()

    def get(self):
        print('passing chunk {}'.format(self.i))
        start_idx = self.i*self.chunk_size
        end_idx = start_idx + self.chunk_size
        self.i += 1
        time.sleep(.2)
        if len(np.shape(self.sweep_points)) == 2:
            return self.sweep_points[start_idx:end_idx, :].T
        else:
            return self.sweep_points[start_idx:end_idx]

##############################################################################
##############################################################################
####################     Software Controlled Detectors     ###################
##############################################################################
##############################################################################


class Dummy_Detector_Soft(Soft_Detector):

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        return np.array([np.sin(x/np.pi), np.cos(x/np.pi)])


class Dummy_Detector_Soft_diff_shape(Soft_Detector):
    # For testing purpose, returns data in a slightly different shape

    def __init__(self, delay=0, **kw):
        self.set_kw()
        self.delay = delay
        self.detector_control = 'soft'
        self.name = 'Dummy_Detector_Soft'
        self.value_names = ['I', 'Q']
        self.value_units = ['V', 'V']
        self.i = 0
        # self.x can be used to set x value externally
        self.x = None

    def acquire_data_point(self, **kw):
        if self.x is None:
            x = self.i/15.
        self.i += 1
        time.sleep(self.delay)
        # This is the format an N-D detector returns data in.
        return np.array([[np.sin(x/np.pi), np.cos(x/np.pi)]]).reshape(2, -1)


class Function_Detector(Soft_Detector):
    """
    Defines a detector function that wraps around an user-defined function.
    Inputs are:
        get_function (callable) : function used for acquiring values
        value_names (list) : names of the elements returned by the function
        value_units (list) : units of the elements returned by the function
        result_keys (list) : keys of the dictionary returned by the function
                             if not None
        msmt_kw   (dict)   : kwargs for the get_function, dict items can be
            values or parameters. If they are parameters the output of the
            get method will be used for each get_function evaluation.

        prepare_function (callable): function used as the prepare method
        prepare_kw (dict)   : kwargs for the prepare function
        always_prepare (bool) : if True calls prepare every time data is
            acquried

    The input function get_function must return a dictionary.
    The contents(keys) of this dictionary are going to be the measured
    values to be plotted and stored by PycQED
    """

    def __init__(self, get_function, value_names=None,
                 detector_control: str='soft',
                 value_units: list=None, msmt_kw: dict ={},
                 result_keys: list=None,
                 prepare_function=None, prepare_function_kw: dict={},
                 always_prepare: bool=False, **kw):
        super().__init__()
        self.get_function = get_function
        self.result_keys = result_keys
        self.value_names = value_names
        self.value_units = value_units
        self.msmt_kw = msmt_kw
        self.detector_control = detector_control
        if self.value_names is None:
            self.value_names = result_keys
        if self.value_units is None:
            self.value_units = ['a.u.'] * len(self.value_names)

        self.prepare_function = prepare_function
        self.prepare_function_kw = prepare_function_kw
        self.always_prepare = always_prepare

    def prepare(self, **kw):
        if self.prepare_function is not None:
            self.prepare_function(**self.prepare_function_kwargs)

    def acquire_data_point(self, **kw):
        measurement_kwargs = {}
        # If an entry has a get method that will be used to set the value.
        # This makes parameters work in this context.
        for key, item in self.msmt_kw.items():
            if isinstance(item, _BaseParameter):
                value = item.get()
            else:
                value = item
            measurement_kwargs[key] = value

        # Call the function
        result = self.get_function(**measurement_kwargs)
        if self.result_keys is None:
            return result
        else:
            results = [result[key] for key in self.result_keys]
            if len(results) == 1:
                return results[0]  # for a single entry we don't want a list
            return results

    def get_values(self):
        return self.acquire_data_point()



# --------------------------------------------
# Polling detector functions to be used with acquisition devices defined in
# pycqed.instrument_drivers.acquisition_devices
# --------------------------------------------

class PollDetector(Hard_Detector):
    """
    Base Class for all polling detectors. A polling detector is one that calls
    the poll method of an acquisition device.
    """

    def __init__(self, acq_dev=None, detectors=None,
                 prepare_and_finish_pulsar=False,
                 **kw):
        """
        Init of the PollDetector base class.

        Args
            acq_dev: instance of AcquisitionDevice. Must be provided when a
                single polling detector is passed to detectors.
            detectors (list): poling detectors from this module to be used for
                acquisition
            prepare_and_finish_pulsar (bool, optional): Whether to start and
                stop all other AWGs in pulsar in addition to the AWGs being part
                of the df itself. Defaults to False.

        Keyword args: passed to parent class

        Creates self.det_from_acq_dev as a dict with values the polling
        detectors in detectors, and keys the acquisition device names used by
        these detectors.
        """
        super().__init__(**kw)
        self.always_prepare = False
        self.prepare_and_finish_pulsar = prepare_and_finish_pulsar

        if detectors is None:
            # if no detector is provided then itself is the only detector
            self.detectors = [self]
            if acq_dev is None:
                raise ValueError("An acquisition device is required when "
                                 "using a single polling detector.")
            self.acq_dev = acq_dev
        else:
            # in multi acq_dev mode several detectors are passed.
            self.detectors = [p[1] for p in sorted(
                [(d.acq_dev.name, d) for d in detectors])]
        self.AWG = None

        self.acq_devs = [d.acq_dev for d in self.detectors]
        self.det_from_acq_dev = {k.name: v for k, v in zip(self.acq_devs,
                                                           self.detectors)}
        self.progress_scaling = None

    def prepare(self, sweep_points=None):
        if self.prepare_and_finish_pulsar:
            ps.Pulsar.get_instance().start(
                exclude=[awg.name for awg in self.get_awgs()])
        for acq_dev in self.acq_devs:
            acq_dev.timer = self.timer

    def get_awgs(self):
        return [self.AWG]

    @Timer()
    def poll_data(self):
        """
        Acquire raw data from the acquisition device by calling its poll
        function. Raises error if not all data was acquired as expected from
        the nr_sweep_points attribute of the detector functions.

        Returns:
            raw data: dict of the form {acq_dev.name: raw data array}
        """
        if self.AWG is not None:
            self.timer.checkpoint("PollDetector.poll_data.AWG_restart.start")
            self.AWG.stop()

        for acq_dev in self.acq_devs:
            # Allow the acqusition device to store additional data
            acq_dev.extra_data_callback = self.extra_data_callback
            # Final preparations for an acquisition.
            acq_dev.prepare_poll()

        if self.AWG is not None:
            self.AWG.start(stop_first=False)
            self.timer.checkpoint("PollDetector.poll_data.AWG_restart.end")

        # Initialize dicts to store data and status
        acq_paths = {acq_dev.name: acq_dev.acquisition_nodes()
                     for acq_dev in self.acq_devs}
        data = {k: {n: [] for n in range(len(v))}
                for k, v in acq_paths.items()}
        gotem = {k: [False] * len(v) for k, v in acq_paths.items()}

        # Initialize variables for intermediate progress reporting
        t_callback = time.time()
        print_progress = (self.progress_callback is not None
                          and self.progress_scaling is not None)

        # Acquire data
        accumulated_time = 0
        self.timer.checkpoint("PollDetector.poll_data.loop.start")
        # FIXME: why only check acq_devs[0].timeout()?
        while accumulated_time < self.acq_devs[0].timeout() and \
                not all(np.concatenate(list(gotem.values()))):
            dataset = {}
            for acq_dev in self.acq_devs:
                if not all(gotem[acq_dev.name]):
                    time.sleep(0.01)
                    dataset[acq_dev.name] = acq_dev.poll(0.01)
            for acq_dev_name in dataset.keys():
                n_sp = self.det_from_acq_dev[acq_dev_name].nr_sweep_points
                for n, p in enumerate(acq_paths[acq_dev_name]):
                    if p not in dataset[acq_dev_name]:
                        continue
                    data[acq_dev_name][n] = np.concatenate([
                        data[acq_dev_name][n], *dataset[acq_dev_name][p]])
                    n_data = len(data[acq_dev_name][n])
                    if n_data >= n_sp:
                        gotem[acq_dev_name][n] = True
                        if n_data > n_sp:
                            log.warning(f'Received more data than expected '
                                        f'from {acq_dev_name}. Ignoring '
                                        f'{n_data-n_sp} values.')
                            data[acq_dev_name][n] = data[acq_dev_name][n][:n_sp]

            accumulated_time += 0.01 * len(self.acq_devs)
            if print_progress:
                t_now = time.time()
                if t_now - t_callback > self.progress_callback_interval:
                    try:
                        n_acq = [acq_dev.acquisition_progress()
                                 for acq_dev in self.acq_devs]
                        if any([n > 0 for n in n_acq]):
                            # the following calculation works both if
                            # self.progress_scaling is a vector/list or a scalar
                            progress = np.mean(np.multiply(
                                n_acq, 1 / np.array(self.progress_scaling)))
                            self.progress_callback(progress)
                    except NoProgressError:
                        raise
                    except Exception as e:
                        # printing progress is optional
                        log.debug(f'poll_data: Could not print progress: {e}')
                    t_callback = t_now

        self.timer.checkpoint("PollDetector.poll_data.loop.end")

        if not all(np.concatenate(list(gotem.values()))):
            if self.AWG is not None:
                self.AWG.stop()
            for acq_dev in self.acq_devs:
                acq_dev.acquisition_finalize()
                for n, c in enumerate(acq_paths[acq_dev.name]):
                    if n in data[acq_dev.name]:
                        n_swp = len(data[acq_dev.name][n])
                        tot_swp = self.det_from_acq_dev[
                            acq_dev.name].nr_sweep_points
                        log.info(f"\t: Channel {n}: Got {n_swp} of {tot_swp} "
                                 f"samples")
            raise TimeoutError("Error: Didn't get all results!")

        data_raw = {acq_dev.name: np.array([data[acq_dev.name][key]
                    for key in sorted(data[acq_dev.name].keys())])
                    for acq_dev in self.acq_devs}

        return data_raw

    def get_values(self):
        """
        Wrapper around poll_data and process_data. Used for getting the data
        from a polling detector, which can process the raw data returned by
        poll data.

        Returns:
            processed data array of the same shape as the raw data
        """
        if self.always_prepare:
            self.prepare()
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.acq_dev.name])
        return data_processed

    def process_data(self, data_raw):
        """
        Process a raw data array as returned by poll_data().

        Args:
            data_raw (array): raw data array to be processed

        Returns:
             processed data array

        No processing done by default. Should be overwritten by child classes.
        """
        return data_raw

    def finish(self):
        """
        Takes care of setting instruments into a known state at the end of
        acquisition.
        """
        if self.prepare_and_finish_pulsar:
            ps.Pulsar.get_instance().stop()
        elif self.AWG is not None:
            self.AWG.stop()

        for d in self.detectors:
            d.acq_dev.acquisition_finalize()


class MultiPollDetector(PollDetector):
    """
    Combines several polling detectors into a single detector.
    """
    class MultiAWGWrapper:
        """
        Wrapper to flexibly define the master AWG to be restarted by the
        detector function.

        This allows for instance to only restart the acquisition device (and
        not the whole Pulsar) for measurements with only one segment.
        """
        def __init__(self, master_awg, awgs=()):
            self.master_awg = master_awg
            self.awgs = list(set(awgs))

        def start(self, **kw):
            for awg in self.awgs:
                awg.start(**kw)
            self.master_awg.start(**kw)

        def stop(self):
            for awg in self.awgs:
                awg.stop()
            self.master_awg.stop()

    def __init__(self, detectors, AWG=None, **kw):
        """
        Init of the MultiPollDetector base class.

        Args
            detectors (list): polling detectors from this module to be used for
                acquisition
            AWG (qcodes instrument): AWG that will be treated as a master AWG
                by wrapping it in a MultiAWGWrapper together with the AWGs of
                the individual detectors

        Keyword args: passed to parent class
        """
        super().__init__(detectors=detectors, **kw)
        self.value_names = []
        self.value_units = []
        self.live_plot_allowed = []  # to be used by MC

        if AWG is not None:  # treat as master AWG
            self.AWG = self.MultiAWGWrapper(AWG,
                                            [d.AWG for d in self.detectors])
        else:
            self.AWG = None
        for d in self.detectors:
            if d.prepare_and_finish_pulsar:
                self.prepare_and_finish_pulsar = True
                d.prepare_and_finish_pulsar = False
            self.value_names += [vn + ' ' + d.acq_dev.name for vn in
                                 d.value_names]
            self.value_units += d.value_units
            self.live_plot_allowed += [d.live_plot_allowed]
            if d.AWG is not None\
                    and not isinstance(self.AWG, self.MultiAWGWrapper):
                if self.AWG is None:
                    self.AWG = d.AWG
                elif self.AWG != d.AWG:
                    raise Exception('Not all AWG instances in '
                                    'MultiPollDetector are the same.')
                d.AWG = None

        # disable live plotting if any of the detectors requests it
        self.live_plot_allowed = all(self.live_plot_allowed)
        # to be used in MC.get_percdone()
        self.acq_data_len_scaling = self.detectors[0].acq_data_len_scaling
        self.detector_control = self.detectors[0].detector_control
        # Check that these are consistent over all detectors
        # (can be made more compact if this is to be done for more params)
        for det in self.detectors:
            if det.acq_data_len_scaling != self.acq_data_len_scaling:
                raise ValueError(f"Detectors {det} and {self.detectors} have" +
                                 " a different acq_data_len_scaling which is" +
                                 " currently not supported")
            if det.detector_control != self.detector_control:
                raise ValueError(f"Detectors {det} and {self.detectors} have" +
                                 " a different detector_control which is" +
                                 " currently not supported")

        # currently only has support for classifier detector data
        self.correlated = kw.get('correlated', False)
        self.averaged = kw.get('averaged', True)
        if 'classifier' in self.detectors[0].name:
            self.correlated = self.detectors[0].get_values_function_kwargs.get(
                'correlated', False)
            self.averaged = self.detectors[0].get_values_function_kwargs.get(
                'averaged', True)

        if self.correlated:
            self.value_names += ['correlation']
            self.value_units += ['']

        self.live_plot_transform_type = self.detectors[
            0].live_plot_transform_type

    @Timer()
    def prepare(self, sweep_points=None):
        """
        Calls the prepare method of each polling detector in self.detectors
        and defines self.progress_scaling to be used in poll_data to decide
        whether to display the acquisition progress.

        Args:
            sweep_points (numpy array): array of sweep points as passed by
                MeasurementControl
        """
        super().prepare()
        if self.detector_control == 'hard' and sweep_points is None:
            raise ValueError("Sweep points must be set for a hard detector")
        for d in self.detectors:
            d.prepare(sweep_points)
        self.progress_scaling = [
            getattr(d, 'progress_scaling', None) for d in self.detectors]
        if any([a is None for a in self.progress_scaling]):
            self.progress_scaling = None

    def get_awgs(self):
        if isinstance(self.AWG, self.MultiAWGWrapper):
            return [self.AWG.master_awg] + self.AWG.awgs
        return [self.AWG]

    def get_values(self):
        """
        Get raw acquisition data from poll_data and process it by calling
        process_data of each polling detectors in self.detectors.
        In addition, calls get_correlations_classif_det if self.correlated is
        True and the acquisition type was single shot.

        Returns:
            processed data array of the same shape as the raw data + 1 if
            self.correlated is True
        """
        data_raw = self.poll_data()

        data_processed = [self.det_from_acq_dev[acq_dev].process_data(d)
                          for acq_dev, d in data_raw.items()]
        data_processed = np.concatenate(data_processed)
        if self.correlated:
            if not self.detectors[0].get_values_function_kwargs.get(
                    'averaged', True):
                data_for_corr = data_processed
            else:
                data_for_corr = np.concatenate([d for d in data_raw.values()])
            corr_data = self.get_correlations_classif_det(data_for_corr)
            data_processed = np.concatenate([data_processed, corr_data], axis=0)

        return data_processed

    def acquire_data_point(self):
        return self.get_values()

    def get_correlations_classif_det(self, data):
        """
        Correlate the single shot data obtained with the ClassifyingPollDetector
        for n qubits. Correlates all the measured qubits. It is currently not
        implemented to correlate subsets of qubits.

        For example, for two qubits:
            - if both qubits are in g or f ---> correlator = 0
            - if both qubits are in e ---> correlator = 0
            - if one qubit is in g or f but the other in e ---> correlator = 1

        Args:
            data (numpy array): single shot data for n qubits which will be
            correlated; must have shape (nr ro channels, nr data points)

        Returns:
            correlated data array of shape (1, nr single shots)
        """
        classifier_params_list = []
        state_prob_mtx_list = []
        for d in self.detectors:
            classifier_params_list += d.classifier_params_list
            if self.detectors[0].get_values_function_kwargs.get(
                    'ro_corrected_stored_mtx', False):
                state_prob_mtx_list += d.state_prob_mtx_list
        if len(state_prob_mtx_list) == 0:
            state_prob_mtx_list = None
        d0 = self.detectors[0]
        nr_states = len(d0.state_labels)
        all_ch_pairs = [d.channel_str_mobj for d in self.detectors]
        all_ch_pairs = [e0 for e1 in all_ch_pairs for e0 in e1]

        data_processed = data
        if self.detectors[0].get_values_function_kwargs.get(
                'averaged', True):
            ro_corrected_seq_cal_mtx = d0.get_values_function_kwargs.get(
                'ro_corrected_seq_cal_mtx', False)
            if ro_corrected_seq_cal_mtx:
                raise NotImplementedError(
                    'Cannot apply data correction based on calibration '
                    'state_prob_mtx from measurement sequence when '
                    'correlated==True because correlations are calculated '
                    'per shot and then averaged over shots. It does not make '
                    'sense to correct with the correlated calibration points.')

            d0_get_values_function_kwargs = d0.get_values_function_kwargs
            d0_channel_str_mobj = d0.channel_str_mobj
            get_values_function_kwargs = deepcopy(d0_get_values_function_kwargs)
            get_values_function_kwargs.update({
                'averaged': False,
                'state_prob_mtx': state_prob_mtx_list,
                'classifier_params': classifier_params_list})
            d0.channel_str_mobj = all_ch_pairs
            d0.get_values_function_kwargs = get_values_function_kwargs
            data_processed = d0.process_data(data)
            d0.channel_str_mobj = d0_channel_str_mobj
            d0.get_values_function_kwargs = d0_get_values_function_kwargs

        # can only correlate corresponding probabilities on all channels;
        # it cannot correlate selected channels
        q = data_processed.shape[0] // nr_states  # nr of qubits
        # creates array with nr_sweep_points columns and
        # nr_qubits rows, where all entries are 0, 1, or 2. The entry in each
        # column is the state of the respective qubit, 0=g, 1=e, 2=f.
        qb_states_list = [np.argmax(
            data_processed[i * nr_states: i * nr_states + nr_states, :],
            axis=0) for i in range(q)]

        # correlate the shots of the two qubits as follows:
        # if both qubits are in g or f ---> correlator = 0
        # if both qubits are in e ---> correlator = 0
        # if one qubit is in g or f but the other in e ---> correlator = 1
        corr_data = np.sum(np.array(qb_states_list) % 2, axis=0) % 2
        if self.averaged:
            corr_data = np.reshape(corr_data,
                                   (d0.nr_shots, d0.nr_sweep_points/d0.nr_shots))
            corr_data = np.mean(corr_data, axis=0)
        corr_data = np.reshape(corr_data, (1, corr_data.size))

        return corr_data

    def finish(self):
        """
        Takes care of setting instruments into a known state at the end of
        acquisition by calling the finish method of each polling detector in
        self.detectors.
        TODO: shouldn't it call the super method?
        """
        if self.AWG is not None:
            self.AWG.stop()
        for d in self.detectors:
            d.finish()


class AveragingPollDetector(PollDetector):

    """
    Polling detector used for acquiring averaged timetraces.
    """

    def __init__(self, acq_dev, AWG=None, channels=((0, 0), (0, 1)),
                 nr_averages=1024, acquisition_length=2.275e-6, **kw):
        """
        Init of the AveragingPollDetector class.

        Args:
            acq_dev ()
            :param AWG: instance of AcquisitionDevice. Must be provided when a
                single polling detector is passed to detectors.
            channels (tuple or list): Channels on which the acquisition should
                be performed. Each channel is identified by a tuple of
                acquisition unit and quadrature index (0=I, 1=Q). See also
                the docstring of AcquisitionDevice.acquisition_initialize.
            nr_averages (int): number of acquisition averages as a power of 2.
            acquisition_length (float): acquisition duration in seconds

        Keyword args: passed to parent class

        """
        super().__init__(acq_dev, **kw)
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, ch in enumerate(self.channels):
            self.value_names[i] = f'{acq_dev.name}_{ch[0]}_ch{ch[1]}'
            self.value_units[i] = 'V'
        self.AWG = AWG
        self.acquisition_length = acquisition_length
        self.nr_averages = nr_averages
        self.progress_scaling = nr_averages

    @Timer()
    def prepare(self, sweep_points):
        """
        Prepares instruments for acquisition by calling
        self.acq_dev.acquisition_initialize.

        Args:
            sweep_points (numpy array): array of sweep points as passed by
                MeasurementControl
        """
        super().prepare()
        if self.AWG is not None:
            self.AWG.stop()
        self.nr_sweep_points = len(sweep_points)
        self.acq_dev.acquisition_initialize(
            channels=self.channels,
            n_results=1,
            acquisition_length=self.acquisition_length,
            averages=self.nr_averages,
            loop_cnt=int(self.nr_averages),
            mode='avg')


class IntegratingAveragingPollDetector(PollDetector):
    """
    Detector used for integrated average acquisition.
    """

    def __init__(self, acq_dev, AWG=None,
                 integration_length: float = 1e-6,
                 nr_averages: int = 1024,
                 channels: list = ((0, 0), (0, 1)),
                 data_type: str = 'raw',  # FIXME: more general default value?
                 polar: bool = False,
                 single_int_avg: bool = False,
                 chunk_size: int = None,
                 values_per_point: int = 1,
                 values_per_point_suffix: list = None,
                 always_prepare: bool = False,
                 prepare_function=None,
                 prepare_function_kwargs: dict = None,
                 **kw):
        """
        Init of the IntegratingAveragingPollDetector.

        Args:
            acq_dev (instrument) : data acquisition device
            AWG   (instrument) : device responsible for starting and stopping
                    the experiment, can also be a central controller
            integration_length (float): integration length in seconds
            nr_averages (int)         : nr of averages per data point
                IMPORTANT: this must be a power of 2
            data_type (str) :  options are
                - raw            -> returns raw data in V
                - raw_corr       -> correlations mode: multiplies results from
                                    pairs of integration channels
                - lin_trans      -> applies the linear transformation matrix and
                                    subtracts the offsets defined in the UHFQC.
                                    This is typically used for crosstalk
                                    suppression and normalization. Requires
                                    optimal weights.
                - digitized      -> returns fraction of shots based on the
                                    threshold defined in the UHFQC. Requires
                                    optimal weights.
                - digitized_corr -> correlations mode after threshold: XNOR of
                                    thresholded results from pairs of
                                    integration channels.
                                    NOTE: thresholds need to be set outside the
                                    detector object.
            polar (bool)     : if True returns data in polar coordinates
                useful for e.g., spectroscopy. Defaults to 'False'.
            single_int_avg (bool): if True makes this a soft detector

            Args relating to changing the amount of points being detected:

            chunk_size    (int)  : used in single shot readout experiments.
            values_per_point (int): number of values to measure per sweep point.
                    creates extra column/value_names in the dataset for each
                    channel.
            values_per_point_suffix (list): suffix to add to channel names for
                    each value. should be a list of strings with lenght equal to
                    values per point.
            always_prepare (bool) : when True the acquire/get_values method will
                first call the prepare statement. This is particularly important
                when it is both a single_int_avg detector and acquires multiple
                segments per point.
            prepare_function (callable): function to be called in prepare
            prepare_function_kwargs (dict): kwargs for prepare_function
        """
        super().__init__(acq_dev, **kw)
        self.name = '{}_integrated_average'.format(data_type)
        self.channels = deepcopy(channels)

        self.value_names = [f'{acq_dev.name}_{ch[0]}_{data_type} w{ch[1]}'
                            for ch in self.channels]
        value_properties = acq_dev.get_value_properties(
            data_type, integration_length)
        self.value_units = ([value_properties['value_unit']] *
                            len(self.channels))
        self.scaling_factor = value_properties['scaling_factor']
        self.value_names, self.value_units = self._add_value_name_suffix(
            value_names=self.value_names, value_units=self.value_units,
            values_per_point=values_per_point,
            values_per_point_suffix=values_per_point_suffix)

        self.single_int_avg = single_int_avg
        if self.single_int_avg:
            self.detector_control = 'soft'
        # useful in combination with single int_avg
        self.always_prepare = always_prepare
        self.values_per_point = values_per_point

        self.AWG = AWG
        self.nr_averages = nr_averages
        self.progress_scaling = nr_averages
        self.nr_shots = 1
        self.integration_length = integration_length
        self.data_type = data_type
        self.chunk_size = chunk_size

        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self.set_polar(polar)

    def _add_value_name_suffix(self, value_names: list, value_units: list,
                               values_per_point: int,
                               values_per_point_suffix: list):
        """
        For use with multiple values_per_point. Adds the strings provided in
        the values_per_point_suffix list.
        """
        if values_per_point == 1:
            return value_names, value_units
        else:
            new_value_names = []
            new_value_units = []
            if values_per_point_suffix is None:
                values_per_point_suffix = ascii_uppercase[:len(value_names)]

            for vn, vu in zip(value_names, value_units):
                for val_suffix in values_per_point_suffix:
                    new_value_names.append('{} {}'.format(vn, val_suffix))
                    new_value_units.append(vu)
            return new_value_names, new_value_units

    def set_polar(self, polar=True):
        """Sets df attribute and adopts value units and names if necessary

        Args:
            polar (bool, optional): Whether the data should be converted to
                polar or not. Also affects the units and value names.
                Defaults to True.
        """
        self.polar = polar
        if self.polar:
            if len(self.channels) != 2:
                raise ValueError('Length of "{}" is not 2'.format(
                                 self.channels))
            self.value_names[0] = 'Magn'
            self.value_names[1] = 'Phase'
            self.value_units[1] = 'deg'

    def process_data(self, data_raw, polar=None, reshape_data=True):
        """
        Process the raw data array as returned by poll_data().
         - multiplies by self.scaling_factor
         - calls acq_dev.correct_offset (see docstring there)
         - calls convert_to_polar if needed
         - reshapes data if needed

        Args:
            data_raw (array): raw data array to be processed
            polar (bool): whether to convert to polar data (True), see
                docstring of convert_to_polar. Defaults to self.polar if
                None.
            reshape_data (bool): whether to reshape the processed data based on
                len(self.value_names) // len(self.channels).

        Returns:
             processed data array of the same shape as data_raw
        """
        data = data_raw * self.scaling_factor
        data = self.acq_dev.correct_offset(self.channels, data)

        if polar is None:
            polar = self.polar
        if polar:
            data = self.convert_to_polar(data)

        if reshape_data:
            n_virtual_channels = len(self.value_names) // len(self.channels)
            data = np.reshape(
                data.T, (-1, n_virtual_channels, len(self.channels))).T
            data = data.reshape((len(self.value_names), -1))

        return data

    def convert_to_polar(self, data):
        """
        Convert 2-channel IQ data to signal magnitude and phase.

        Args:
            data (array): 2-channel IQ data array to be converted

        Returns:
            converted data as array of shape (2, nr points) where the first row
                is the signal magnitude and the second is the signal phase
        """
        if len(data) != 2:
            raise ValueError(
                'Expect 2 channels for rotation. Got {}'.format(len(data)))
        I = data[0]
        Q = data[1]
        S21 = I + 1j*Q
        data[0] = np.abs(S21)
        data[1] = np.angle(S21) * 180 / np.pi
        return data

    def acquire_data_point(self):
        """
        Calls self.get_values().
        """
        return self.get_values()

    @Timer()
    def prepare(self, sweep_points=None):
        """
        Prepares instruments for acquisition:
         - defines self.nr_sweep_points based on sweep_points,
            self.values_per_point, self.single_int_avg, self.chunk_size,
            self.acq_data_len_scaling, and self.nr_shots. Will be checked in
            poll_data of the parent class.
         - calls prepare_function if defined
         - calls self.acq_dev.acquisition_initialize

        Args:
            sweep_points (numpy array): array of sweep points as passed by
                MeasurementControl
        """
        super().prepare()
        if self.AWG is not None:
            self.AWG.stop()
        # Determine the number of sweep points and set them
        if sweep_points is None or self.single_int_avg:
            # this case will be used when it is a soft detector
            # Note: single_int_avg overrides chunk_size
            # single_int_avg = True is equivalent to chunk_size = 1
            self.nr_sweep_points = self.values_per_point
        else:
            self.nr_sweep_points = len(sweep_points) * self.values_per_point
            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.values_per_point

        # Optionally perform extra actions on prepare
        # This snippet is placed here so that it has a chance to modify the
        # nr_sweep_points
        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        # Determine the true number of sweep points in cases where MC has
        # tiled the sweep points for SSRO. This does not have an effect in
        # the current class (where acq_data_len_scaling is 1), but it
        # might be relevant for child classes.
        if self.acq_data_len_scaling != 1:
            assert self.nr_sweep_points % self.acq_data_len_scaling == 0
            self.nr_sweep_points = (self.nr_sweep_points //
                                    self.acq_data_len_scaling)
        self.nr_sweep_points *= self.nr_shots
        # Note that self.nr_shots is 1 in this class, but might be different
        # in child classes.
        self.acq_dev.acquisition_initialize(
            channels=self.channels,
            n_results=self.nr_sweep_points,
            acquisition_length=self.integration_length,
            averages=self.nr_averages,
            loop_cnt=int(self.nr_shots * self.nr_averages),
            mode='int_avg', data_type=self.data_type,
        )


class ScopePollDetector(PollDetector):
    """
    Detector for scope measurements.

    Attributes:
        data_type (str) :  options are
            - timedomain: Returns time traces (possibly averaged and/or
              single-shot)
            - fft: Returns the absolute value of the Fourier' transform
                       of the data.
            - fft_power: Squares the data before averaging and taking the
                             Fourier' transform.
    """

    def __init__(self,
                 acq_dev,
                 AWG,
                 channels,
                 nr_shots,
                 integration_length,
                 nr_averages,
                 data_type,
                 **kw):
        super().__init__(acq_dev=acq_dev, detectors=None, **kw)
        self.name = f'{data_type}_scope'
        self.channels = channels
        self.integration_length = integration_length
        self.nr_averages = nr_averages
        self.data_type = data_type
        self.AWG = AWG
        self.nr_sweep_points = None
        self.values_per_point = 1
        self.nr_shots = nr_shots
        if self.data_type == 'timedomain':
            # Normal number of shots (MC will expect that many timetraces)
            self.acq_data_len_scaling = self.nr_shots
        elif self.data_type == 'fft':
            raise NotImplementedError("Amplitude FFT mode not implemented!")
        elif self.data_type == 'fft_power':
            # Multiple shots aren't implemented for power spectrum measurements
            self.acq_data_len_scaling = 1

    def prepare(self, sweep_points=None):

        super().prepare()
        self.nr_sweep_points = len(sweep_points)
        if self.data_type == 'fft_power':
            # Number of points of the spectrum to be returned
            n_results = self.nr_sweep_points
        elif self.data_type == 'timedomain':
            # Meaning 1 timetrace. Could be extended e.g. if hardware allows
            # TV-mode avg of timetraces
            n_results = 1
        else:
            raise ValueError

        self.acq_dev.acquisition_initialize(
            channels=self.channels,
            n_results=n_results,
            acquisition_length=self.integration_length,
            averages=self.nr_averages,
            loop_cnt=self.nr_shots * self.nr_averages,
            mode='scope', data_type=self.data_type,
        )


class UHFQC_correlation_detector(IntegratingAveragingPollDetector):
    """
    Detector used for correlation mode with the UHFQC.
    The argument 'correlations' is a list of tuples specifying which channels
    are correlated, and on which channel the correlated signal is output.
    For instance, 'correlations=[(0, 1, 3)]' will put the correlation of
    channels 0 and 1 on channel 3.
    """

    def __init__(self, acq_dev, AWG=None, integration_length=1e-6,
                 nr_averages=1024,  polar=True,
                 channels: list = ((0, 0), (0, 1)),
                 correlations: list = (((0, 0), (0, 1))),
                 data_type: str = 'raw_corr',
                 used_channels=None, value_names=None, single_int_avg=False,
                 **kw):
        if not acq_dev.IDN()['model'].startswith('UHF'):
            raise NotImplementedError(
                f'UHFQC_correlation_detector is not implemented for '
                f'{acq_dev.name}, but only for ZI UHF devices.')
        if data_type not in ['raw_corr', 'digitized_corr']:
            raise ValueError('data_type can only be "raw_corr" or '
                             '"digitized_corr" for this detector.')

        super().__init__(
            acq_dev, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, polar=polar,
            channels=channels, single_int_avg=single_int_avg,
            data_type=data_type,
            **kw)

        self.name = '{}_UHFQC_correlation_detector'.format(data_type)
        self.correlations = correlations
        self.thresholding = self.data_type == 'digitized_corr'

        self.used_channels = used_channels
        if self.used_channels is None:
            self.used_channels = self.channels

        if value_names is not None:
            self.value_names = value_names
        else:
            for corr in correlations:
                self.value_names += ['corr ({},{})'.format(corr[0], corr[1])]
        value_unit = acq_dev.get_value_properties(
            self.data_type, self.integration_length)['value_unit']
        for _ in correlations:
            self.value_units += [f'({value_unit})^2']

        self.define_correlation_channels()

    def prepare(self, sweep_points=None):
        super().prepare(sweep_points=sweep_points)
        self.set_up_correlation_weights()

    def define_correlation_channels(self):
        self.correlation_channels = []
        used_channels = deepcopy(self.used_channels)
        for corr in self.correlations:
            # Start by assigning channels
            # We can assume that channels belong to acquisition unit 0 because
            # this detector is specific to the UHF. (The acquisition device
            # would raise an error otherwise.)
            if corr[0] not in used_channels or corr[1] not in used_channels:
                raise ValueError('Correlations should be in used channels')

            correlation_channel = None

            for ch in range(self.acq_dev.n_acq_int_channels):
                ch = (0, ch)
                # Find the first unused channel to set up as correlation
                if ch not in used_channels:
                    # selects the lowest available free channel
                    correlation_channel = ch
                    self.channels += [ch]
                    self.correlation_channels += [correlation_channel]

                    print('Using channel {} for correlation ({}, {}).'
                          .format(ch, corr[0], corr[1]))
                    # correlation mode is turned on in the
                    # set_up_correlation_weights method
                    break
                    # FIXME, can currently only use one correlation

            if correlation_channel is None:
                raise ValueError('No free channel available for correlation.')
            else:
                used_channels += [correlation_channel]

    def set_up_correlation_weights(self):
        # Configure correlation mode
        for ch in self.channels:
            if ch not in self.correlation_channels:
                # Disable correlation mode as this is used for normal
                # acquisition
                self.acq_dev.set(f'qas_0_correlations_{ch[1]}_enable', 0)

        for correlation_channel, corr in zip(self.correlation_channels,
                                             self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            copy_int_weights_real = \
                np.array(self.acq_dev.get(
                    f'qas_0_integration_weights_{corr[0][1]}_real')).astype(float)
            copy_int_weights_imag = \
                np.array(self.acq_dev.get(
                    f'qas_0_integration_weights_{corr[0][1]}_imag')).astype(float)

            copy_rot_matrix = self.acq_dev.get(f'qas_0_rotations_{corr[0][1]}')

            self.acq_dev.set(
                f'qas_0_integration_weights_{correlation_channel[1]}_real',
                copy_int_weights_real)
            self.acq_dev.set(
                f'qas_0_integration_weights_{correlation_channel[1]}_imag',
                copy_int_weights_imag)

            self.acq_dev.set(
                f'qas_0_rotations_{correlation_channel[1]}',
                copy_rot_matrix)

            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.acq_dev.set(
                f'qas_0_correlations_{correlation_channel[1]}_enable', 1)
            self.acq_dev.set(
                f'qas_0_correlations_{correlation_channel[1]}_source', corr[1][1])

            # If thresholding is enabled, set the threshold for the correlation
            # channel.
            if self.thresholding:
                thresh_level = \
                    self.acq_dev.get(f'qas_0_thresholds_{corr[0][1]}_level')
                self.acq_dev.set(
                    f'qas_0_thresholds_{correlation_channel[1]}_level',
                    thresh_level)

    def process_data(self, data_raw):
        if self.thresholding:
            data = data_raw
        else:
            data = []
            for n, ch in enumerate(self.used_channels):
                if ch in self.correlation_channels:
                    # 1.5 is a workaround due to a bug in the UHF
                    data.append(1.5 * np.array(data_raw[n]) *
                                self.scaling_factor**2)
                else:
                    data.append(np.array(data_raw[n]) * self.scaling_factor)
        return data


class IntegratingSingleShotPollDetector(IntegratingAveragingPollDetector):
    """
    Detector used for integrated single-shot acquisition.
    """

    def __init__(self, acq_dev, nr_shots: int = 4094, **kw):
        """
        Init of the IntegratingSingleShotPollDetector.
        See the IntegratingAveragingPollDetector for the full dostring of the
        accepted input parameters.

        Args:
            acq_dev (instrument): data acquisition device
            nr_shots (int)     : number of acquisition shots

        Keyword args:
            passed to the init of the parent class. In addition,
            the following keyword arguments are understood:
            - live_plot_allowed: (bool, default: False) whether to allow MC to
              use live plotting
        """
        super().__init__(acq_dev, nr_averages=1, **kw)

        self.name = '{}_integration_logging_det'.format(self.data_type)
        self.nr_shots = nr_shots
        self.acq_data_len_scaling = self.nr_shots  # to be used in MC
        # Disable MC live plotting by default for SSRO acquisition
        self.live_plot_allowed = kw.get('live_plot_allowed', False)


class ClassifyingPollDetector(IntegratingSingleShotPollDetector):
    """
    Hybrid detector function:
     - the acq_dev is configured to return single shots, but this function can
     then return either single shots, or averaged sweep points (can average
     over the shots for each segment)
     - This class always performs classification of each single shot data
     into either a 3 state probability tuple (shot_i_pg, shot_i_pe, shot_i_pf)
     if qutrit is True, or a 2 state probability tuple (shot_i_pg, shot_i_pe).
     Here shot_i_pg, shot_i_pe, (shot_i_pf) \in [0, 1] and
     shot_i_pg + shot_i_pe (+ shot_i_pf) = 1.

    See the IntegratingPollDetector for the full dostring of the accepted
    input parameters.

    The only additional keyword argument that is used by this class but not by
    its parent class is get_values_function_kwargs. This parameter is a dict
    where the user can specify how he wants this detector function to process
    the shots.
    get_values_function_kwargs can contain:
     - classifier_params_list (list or dict): THIS ENTRY MUST EXIST. This class
        always classifies into state probabilities and it does so based on this
        parameter. It is either the QuDev_transmon acq_classifier_params
        attribute (if only one qubit is measured), or list of these dictionaries
        (if multiple qubits are measured).
     - averaged (bool; default: True): decides whether to average over the
        shots of each segment. If True, this class returns averaged data with
        size == len(sweep_points). If False, this class returns single shot
        data with size == len(sweep_points)*nr_shots
     - thresholded (bool; default: False): decides whether to assign each shot
        to only one state. This is different from classification, and comes
        after the classification step.
        If True, for each classified shot (shot_i_pg, shot_i_pe, shot_i_pf),
        sets the entry corresponding to the max of the three probabilities to 1
        and the other two entries to 0.
        Ex: For (shot_i_pg, shot_i_pe, shot_i_pf) = (0.04, 0.9, 0.06),
        this options produces (0, 1, 0).
        FIXME: (Steph 14.05.2020) Should we rename this param to "assigned"?
     - state_prob_mtx_list (np.ndarray or list or None; default: None)
        If not None, the data can be corrected with the inverse of these arrays.
        Either the QuDev_transmon acq_state_prob_mtx attribute (if only
        one qubit is measured), or list of these arrays (if multiple
        qubits are measured).
     - ro_corrected_stored_mtx (bool: default: False): decides whether to
        correct the data with the inverse of the matrices in state_prob_mtx_list
     - ro_corrected_seq_cal_mtx (bool: default: False): decides whether to
        correct the data with the inverse of the calibration state_prob_mtx '
        extracted from the measurement sequence. THIS CAN ONLY BE DONE IF
        THE DATA IS AVERAGED FIRST (averaged == True)

    Keyword args:
        kw is passed to the init of the parent class. In addition,
        the following keyword arguments are understood:
            - live_plot_allowed: (bool) whether to allow MC to use live plotting
              By default, True if averaged acquisition and False if single shot.
    """

    def __init__(self, acq_dev, *args, **kw):
        self.qutrit = kw.pop('qutrit', True)
        super().__init__(acq_dev, *args, **kw)

        self.get_values_function_kwargs = kw.get('get_values_function_kwargs',
                                                 None)
        self.name = '{}_classifier_det'.format(self.data_type)

        self.state_labels = ['pg', 'pe', 'pf'] if self.qutrit else ['pg', 'pe']
        classifier_params = self.get_values_function_kwargs.get(
            'classifier_params', [])
        self.n_meas_objs = 1 if not len(classifier_params) else \
                len(classifier_params)
        k = len(self.channels) // self.n_meas_objs
        # this will give <acq unit>_<1st wint ch><2nd wint ch>, e.g., '0_01'
        self.channel_str_mobj = [(
            str(self.channels[k*j][0]),
            ''.join([str(ch[1]) for ch in self.channels[k*j:k*j+k]])
        ) for j in range(self.n_meas_objs)]

        self.classified = self.get_values_function_kwargs.get('classified',
                                                              True)
        if self.classified:
            self.value_names = ['']*(
                    len(self.state_labels) * len(self.channel_str_mobj))
            idx = 0
            for ch_pair in self.channel_str_mobj:
                for state in self.state_labels:
                    self.value_names[idx] = \
                        f'{acq_dev.name}_{ch_pair[0]}_{state} w{ch_pair[1]}'
                    idx += 1
            self.value_units = [self.value_units[0]] * len(self.value_names)

        if self.get_values_function_kwargs.get('averaged', True):
            self.acq_data_len_scaling = 1
            # The following value is only used for correct progress
            # calculation in poll_data.
            self.progress_scaling = self.nr_shots
            self.live_plot_allowed = kw.get('live_plot_allowed', True)
        else:
            # Disable MC live plotting by default for SSRO acquisition
            self.live_plot_allowed = kw.get('live_plot_allowed', False)

    def process_data(self, data_raw, **kw):
        """
        Process the raw data array as returned by poll_data().
        Calls the following methods (if enabled) in the order given here:
         - self.classify_shots if self.classified is True
         - self.threshold_shots if both thresholded and self.classified are True
         - self.average_shots if averaged is True
         - self.reshapes data if needed
         - readout correction:
            - only if averaged is True and thresholded is False
            - self.correct_readout_stored_mtx if ro_corrected_stored_mtx is True
            - self.correct_readout_stored_mtx if ro_corrected_seq_cal_mtx
                is True and averaged is True

        Args:
            data_raw (array): raw data array to be processed

        Keyword args:
            taken from self.get_values_function_kwargs (see docstring in init)

        Returns:
             processed data array of the same shape as data_raw
        """
        data_processed = super().process_data(data_raw, polar=True,
                                              reshape_data=False).T
        nr_states = len(self.state_labels)
        thresholded = self.get_values_function_kwargs.get('thresholded', True)
        averaged = self.get_values_function_kwargs.get('averaged', True)
        if self.classified:
            # Classify data into qutrit states
            self.classifier_params_list = self.get_values_function_kwargs.get(
                'classifier_params', None)
            if self.classifier_params_list is None:
                raise ValueError('Please specify the classifier '
                                 'parameters list.')
            if not isinstance(self.classifier_params_list, list):
                self.classifier_params_list = [self.classifier_params_list]
            data_processed = self.classify_shots(data_processed,
                                                 self.classifier_params_list,
                                                 nr_states)

        if thresholded:
            if not self.classified:
                raise NotImplementedError(
                    'Currently the threshold_shots only works if the data '
                    'was first classified.')
            data_processed = self.threshold_shots(data_processed, nr_states)

        if averaged:
            data_processed = self.average_shots(data_processed)

        # do readout correction
        ro_corrected_seq_cal_mtx = self.get_values_function_kwargs.get(
            'ro_corrected_seq_cal_mtx', False)
        ro_corrected_stored_mtx = self.get_values_function_kwargs.get(
            'ro_corrected_stored_mtx', False)
        if ro_corrected_seq_cal_mtx and ro_corrected_stored_mtx:
            raise ValueError('"ro_corrected_seq_cal_mtx" and '
                             '"ro_corrected_stored_mtx" cannot both be True.')
        ro_corrected = ro_corrected_seq_cal_mtx or ro_corrected_stored_mtx
        if (ro_corrected and thresholded) and not averaged:
            raise ValueError('It does not make sense to apply readout '
                             'correction if thresholded==True and '
                             'averaged==False.')
        if ro_corrected_stored_mtx:
            # correct data with matrices from state_prob_mtx_list
            data_processed = self.correct_readout_stored_mtx(data_processed,
                                                             nr_states)
        elif ro_corrected_seq_cal_mtx:
            # correct data with the calibration matrix extracted from
            # the data array
            if not averaged:
                raise NotImplementedError(
                    'Data correction based on calibration state_prob_mtx '
                    'from measurement sequence is currently only '
                    'implemented for averaged data (averaged==True).')
            data_processed = self.correct_readout_seq_cal_mtx(data_processed,
                                                              nr_states)

        return data_processed.T

    def classify_shots(self, data, classifier_params_list, nr_states):
        """
        Classify raw single shots into qubit or qutrit state probabilities.

        Args:
            data (numpy array): single shot data to be classified with shape
                (nr acquisition channels, nr shots)
            classifier_params_list (list or dict): either the QuDev_transmon
                acq_classifier_params attribute (if only one qubit is measured),
                or list of these dictionaries (if multiple qubits are measured).
                Passed to a_tools.predict_gm_proba_from_clf.
            nr_states (int): number of transmon states:
                2 for qubit, 3 for qutrit.

        Returns
            classified data as array of shape (nr shots, nr_states)
        """
        classified_data = np.zeros((self.nr_sweep_points,
                                    nr_states*len(self.channel_str_mobj)))
        k = len(self.channels) // self.n_meas_objs
        for i in range(len(self.channel_str_mobj)):
            # classify each shot into (pg, pe, pf)
            # clf_data will have shape
            # (self.nr_sweep_points, nr_states)
            # where len(nr_sweep_points) = len(mc_sweep_points) * nr_shots
            mobj_data = data[:, k*i: k*i+k]
            clf_data = a_tools.predict_gm_proba_from_clf(
                mobj_data, classifier_params_list[i])
            classified_data[:, nr_states * i: nr_states * i + nr_states] = \
                clf_data

        return classified_data

    def threshold_shots(self, data, nr_states):
        """
        Assign each classified shot to a given state. For example,
        (0.01, 0.98, 0.01) --> (0, 1, 0), i.e. qutrit in e.

        Args:
            data (numpy array): data classified into state probabilities
            nr_states (int): number of transmon states:
                2 for qubit, 3 for qutrit.

        Returns:
            assigned data of the same shape as data
        """
        thresholded_data = np.zeros_like(data)
        for i in range(len(self.channel_str_mobj)):
            # For each shot, set the largest probability entry to 1, and the
            # other two to 0. I.e. assign to one state.
            # clf_data must be 2 dimensional, rows are shots*sweep_points,
            # columns are nr_states
            thresh_data = np.isclose(np.repeat(
                [np.arange(nr_states)],
                data[:, nr_states*i: nr_states*(i+1)].shape[0], axis=0).T,
                     np.argmax(data[:, nr_states*i: nr_states*(i+1)], axis=1)).T
            thresholded_data[:, nr_states * i: nr_states * i + nr_states] = \
                thresh_data

        return thresholded_data

    def average_shots(self, data):
        """
        Average single shot data.

        Args:
            data (numpy array): single shot data

        Returns
            averaged data of shape
            (self.nr_sweep_points//self.nr_shots, data.shape[1])
        """
        # reshape into
        # (nr_shots, nr_sweep_points//self.nr_shots, nr_data_columns)
        # then average over nr_shots
        averaged_data = np.reshape(
            data, (self.nr_shots, self.nr_sweep_points//self.nr_shots,
                   data.shape[-1]))
        # average over shots
        averaged_data = np.mean(averaged_data, axis=0)

        return averaged_data

    def correct_readout_stored_mtx(self, data, nr_states):
        """
        Correct for readout errors by multiplying with the inverse of
            state_prob_mtx given in get_values_function_kwargs
            (see init docstring)

        Args:
            data (numpy array): averaged data with
                self.nr_sweep_points//self.nr_shots rows, and columns given
                by nr acquisition channels or nr of transmon states
            nr_states (int): number of transmon states:
                2 for qubit, 3 for qutrit.

        Returns
            readout corrected data array of the same shape as data
        """
        corrected_data = np.zeros_like(data)

        self.state_prob_mtx_list = self.get_values_function_kwargs.get(
            'state_prob_mtx', None)
        if self.state_prob_mtx_list is not None and \
                not isinstance(self.state_prob_mtx_list, list):
            self.state_prob_mtx_list = [self.state_prob_mtx_list]

        for i in range(len(self.channel_str_mobj)):
            if self.state_prob_mtx_list is not None and \
                    self.state_prob_mtx_list[i] is not None:
                corr_data = np.linalg.inv(self.state_prob_mtx_list[i]).T @ \
                            data[:, nr_states*i: nr_states*(i+1)].T
                log.info('Data corrected based on previously-measured '
                         'state_prob_mtx.')
            else:
                raise ValueError(f'state_prob_mtx for index {i} is None.')
            corrected_data[:, nr_states * i: nr_states * i + nr_states] = \
                corr_data.T

        return corrected_data

    def correct_readout_seq_cal_mtx(self, data, nr_states):
        """
        Correct for readout errors by multiplying with the inverse of the
        calibration matrix extracted from the data array, if calibration
        segments were measured.

        Args:
            data (numpy array): averaged data with
                self.nr_sweep_points//self.nr_shots rows, and columns given
                by nr acquisition channels or nr of transmon states
            nr_states (int): number of transmon states:
                2 for qubit, 3 for qutrit.

        Returns
            readout corrected data array of the same shape as data
        """
        corrected_data = np.zeros_like(data)
        for i in range(len(self.channel_str_mobj)):
            # get cal matrix
            calibration_matrix = data[:, nr_states*i: nr_states*(i+1)][
                                 -nr_states:, :]
            # normalize
            calibration_matrix = calibration_matrix.astype('float') / \
                                 calibration_matrix.sum(axis=1)[:, np.newaxis]
            # correct data
            # corr_data = np.linalg.inv(calibration_matrix).T @ \
            #             data[:, nr_states*i: nr_states*(i+1)].T
            corr_data = calibration_matrix.T @ \
                        data[:, nr_states*i: nr_states*(i+1)].T
            log.info('Data corrected based on calibration state_prob_mtx '
                     'from measurement sequence.')
            corrected_data[:, nr_states * i: nr_states * i + nr_states] = \
                corr_data.T

        return corrected_data


class UHFQC_scope_detector(Hard_Detector):
    """
    Detector used for acquiring averaged timetraces and their Fourier'
    transforms using the scope module of the UHF.

    Requires the DIG option on the UHF to work. Required for the segmented
    acquisition, that is needed to square the signal before averaging.

    Args:
        UHFQC: An UHFQC instance to use.
        AWG: A pulsar or an AWG to start at the beginning of the measurement.
            Typically used to generate pulses to look at and or provide
            triggers.
        channels: Tuple of UHFQA hardware channel indices.
            Indices can be either 0 or 1, as the UHF has 2 input channels.
        nr_averages: Number of times to average.
        nr_samples: Number of samples in a scope trace.
            Actual value will be rounded to the highest power of two that is at
            most the provided value.
        fft_mode: Data processing mode.
            Can be one of the following:
                'timedomain': Records timedomain traces.
                'fft': Returns the absolute value of the Fourier' transform
                       of the data.
                'fft_power': Squares the data before averaging and taking the
                             Fourier' transform.
    Keyword arguments:
        trigger: Boolean, whether to wait for a trigger to start acquisition.
            Defaults to True if AWG instance is given, False otherwise.
        trigger_channel: Integer, specifing the triggering channel.
            Typical values here are
                0: Signal Input 1,
                1: Signal Input 2,
                2: Trigger Input 1,
                3: Trigger Input 2.
            See the UHFQA manual for a full list. Defaults to Trigger Input 1.
        trigger_level: Trigger activation level. Defaults to 0.1 V.

    """
    def __init__(self, UHFQC, AWG=None, channels=(0, 1),
                 nr_averages=20, acquisition_length=2.275e-6,
                 fft_mode='timedomain', **kw):
        super().__init__(**kw)

        self.UHFQC = UHFQC
        self.scope = UHFQC.daq.scopeModule()
        self.AWG = AWG
        self.channels = channels
        nr_samples = self.UHFQC.convert_time_to_n_samples(acquisition_length)
        self.nr_samples = max(int(2**np.floor(np.log2(nr_samples))), 4096)
        self.nr_averages = nr_averages
        self.fft_mode = fft_mode
        self.value_names = [f'{UHFQC.name}_ch{ch}' for ch in channels]
        if fft_mode == 'fft_power':
            self.value_units = ['V^2' for _ in channels]
        else:
            self.value_units = ['V' for _ in channels]
        self.trigger_channel = kw.get('trigger_channel', 2)
        self.trigger_level = kw.get('trigger_level', 0.1)
        self.trigger = kw.get('trigger', self.AWG is not None)

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()

        self.scope.unsubscribe(f'/{self.UHFQC.devname}/scopes/0/wave')
        self.scope.finish()

    def get_values(self):
        self.UHFQC.scopes_0_single(1)
        self.UHFQC.scopes_0_enable(1)
        self.scope.subscribe(f'/{self.UHFQC.devname}/scopes/0/wave')
        self.scope.execute()
        if self.AWG is not None:
            self.AWG.start()
        result = self.scope.read()
        while int(self.scope.progress()) != 1:
            time.sleep(0.1)
            result = self.scope.read()
        return [x.reshape(self.nr_averages, -1).mean(0) for
                x in result[self.UHFQC.devname]['scopes']['0']['wave'][0][0]['wave']]

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        self.UHFQC.scopes_0_enable(0)
        self.UHFQC.scopes_0_length(self.nr_samples)
        self.UHFQC.scopes_0_channel((1 << len(self.channels)) - 1)
        self.UHFQC.scopes_0_segments_count(self.nr_averages)
        self.UHFQC.scopes_0_segments_enable(1)

        for i, ch in enumerate(self.channels):
            self.UHFQC.set(f'scopes_0_channels_{i}_inputselect', ch)

        self.UHFQC.scopes_0_single(1)
        if self.fft_mode == 'fft_power':
            self.scope.set('scopeModule/mode', 3)
            self.scope.set('scopeModule/fft/power', 1)
        elif self.fft_mode == 'fft':
            self.scope.set('scopeModule/mode', 3)
            self.scope.set('scopeModule/fft/power', 0)
        elif self.fft_mode == 'timedomain':
            self.scope.set('scopeModule/mode', 1)
            self.scope.set('scopeModule/fft/power', 0)
        else:
            raise ValueError("Invalid fft_mode. Allowed options are "
                             "'timedomain', 'fft' and 'fft_power'")
        self.scope.set('scopeModule/averager/weight', 1)

        self.UHFQC.scopes_0_trigenable(self.trigger)
        self.UHFQC.scopes_0_trigchannel(self.trigger_channel)
        self.UHFQC.scopes_0_triglevel(self.trigger_level)
        self.UHFQC.scopes_0_trigslope(0)

    def get_sweep_vals(self):
        if self.fft_mode == 'timedomain':
            return np.linspace(0, self.nr_samples/1.8e9, self.nr_samples, endpoint=False)
        elif self.fft_mode in ('fft', 'fft_power'):
            return np.linspace(0, 0.9e9, self.nr_samples//2, endpoint=False)