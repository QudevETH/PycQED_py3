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
        # to be used by MC.get_percdone()
        self.acq_data_len_scaling = 1
        self.timer = Timer(self.name)
        # The following properties are not implemented in all detector
        # functions (i.e., might be ignored in some detector functions),
        # but are created here to have a common interface.
        self.progress_callback = kw.get('progress_callback', None)
        self.progress_callback_interval = kw.get(
            'progress_callback_interval', 5)  # in seconds
        # tells MC whether to show live plotting for the measurement
        self.live_plot_enabled = kw.get('live_plot_enabled', True)

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
                    if 'UHFs' in d:
                        # d["UHFs"] will always contain one item because of how
                        # savable_attribute_value was written.
                        detectors_dict.update(
                            {f'{d["UHFs"][0]} {d["name"]}': d})
                    else:
                        detectors_dict.update({f'{d["name"]}': d})
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
        for i, detector in enumerate(detectors):
            for detector_value_name in detector.value_names:
                if det_idx_suffix:
                    detector_value_name += '_det{}'.format(i)
                self.value_names.append(detector_value_name)
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
        for i in self.index:
            v = v[i]
        return v

    def acquire_data_point(self):
        v = self.detector.get_values()
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
# Zurich Instruments UHFQC detector functions
# --------------------------------------------

class UHFQC_Base(Hard_Detector):
    """
    Base Class for all UHF detectors
    """
    def __init__(self, UHFQC=None, detectors=None, **kw):
        super().__init__(**kw)
        if detectors is None:
            # if no detector is provided then itself is the only detector
            self.detectors = [self]
            assert UHFQC is not None, "UHFQC required when using single detector"
            self.UHFQC = UHFQC
        else:
            # in multi uhf mode several detectors are passed.
            self.detectors = [p[1] for p in sorted(
                [(d.UHFQC.devname, d) for d in detectors], reverse=True)]
        self.AWG = None

        self.UHFs = [d.UHFQC for d in self.detectors]
        self.UHF_map = {UHF.name: i
                   for UHF, i in zip(self.UHFs, range(len(self.detectors)))}
        self.nr_averages = None
        self.ro_mode = 'rl'

    def _check_hardware_limitations(self):
        """
        This method should be used to check whether the measurement settings
        are supported by the hardware.
        Currently, it only checks whether the total number of acquisitions in
        case of a single-shot readout is supported by the UHF
        (1048576 is hardcoded).
        """

        for i, d in enumerate(self.detectors):
            if hasattr(d, 'nr_shots'):
                # Either integration_logging_det or classifier_detector

                # For the integration logging detector,
                # nr_sweep_points = nr_shots * nr_segments
                total_nr_shots_acq = d.nr_sweep_points
                if hasattr(d, 'classified'):
                    # For the classifier detector
                    # nr_sweep_points = nr_segments like for the integrated
                    # averaged detector, but the UHF is programmed to return
                    # single shots.
                    total_nr_shots_acq *= d.nr_shots
                if total_nr_shots_acq > 2**20:
                    raise ValueError(f'For detector function number {i}, '
                                     f'({d.name}) nr. segments * nr. shots = '
                                     f'{total_nr_shots_acq} > 1048576 '
                                     f'supported by the UHF. Please reduce the '
                                     f'compression_seg_lim, the number of 1D '
                                     f'sweep points, or the nr_shots.')

    @Timer()
    def poll_data(self):
        if self.AWG is not None:
            self.timer.checkpoint("UHFQC_Base.poll_data.AWG_restart.start")
            self.AWG.stop()

        for UHF in self.UHFs:
            UHF.set('qas_0_result_enable', 1)

        if self.AWG is not None:
            self.AWG.start(stop_first=False)
            self.timer.checkpoint("UHFQC_Base.poll_data.AWG_restart.end")

        acq_paths = {UHF.name: UHF._acquisition_nodes for UHF in self.UHFs}

        data = {UHF.name: {k: [] for k, dummy in enumerate(UHF._acquisition_nodes)}
                for UHF in self.UHFs}

        # Acquire data
        gotem = {UHF.name: [False] * len(UHF._acquisition_nodes) for UHF in
                 self.UHFs}
        accumulated_time = 0

        print_progress = (self.progress_callback is not None
                          and self.nr_averages is not None)
        if print_progress:
            t_callback = time.time()
            n_acq_last = [0] * len(self.UHFs)
            n_acq_add = [0] * len(self.UHFs)
        self.timer.checkpoint("UHFQC_Base.poll_data.loop.start")
        while accumulated_time < self.UHFs[0].timeout() and \
                not all(np.concatenate(list(gotem.values()))):
            dataset = {}
            for UHF in self.UHFs:
                if not all(gotem[UHF.name]):
                    time.sleep(0.01)
                    dataset[UHF.name] = UHF.poll(0.01)
            for UHFname in dataset.keys():
                for n, p in enumerate(acq_paths[UHFname]):
                    if p in dataset[UHFname]:
                        for v in dataset[UHFname][p]:
                            data[UHFname][n] = np.concatenate(
                                (data[UHFname][n], v['vector']))

                            if len(data[UHFname][n]) >= self.detectors[
                                self.UHF_map[UHFname]].nr_sweep_points:
                                gotem[UHFname][n] = True
            accumulated_time += 0.01 * len(self.UHFs)
            if print_progress:
                t_now = time.time()
                if t_now - t_callback > self.progress_callback_interval:
                    try:
                        if self.ro_mode == 'rl':
                            n_acq = [UHF.qas_0_result_acquired() for UHF in
                                     self.UHFs]
                        else:
                            n_acq = [UHF.qas_0_monitor_acquired() for UHF in
                                     self.UHFs]
                        # Two special cases are treated in the following lines.
                        # - The UHF reports 0 when it is done. In this case,
                        #   we keep the last known progress values n_acq_last.
                        #   This means that the progress indicator will stay
                        #   at that last known progress value during data
                        #   transfer, and MC will update the progress to the
                        #   correct value once it takes over control after
                        #   the end of the data transfer.
                        # - A workaround is needed because the UHF truncates
                        #   qas_0_result_acquired at 2**18 (and starts
                        #   counting from 0 again). A decrease in n_acq
                        #   compared to the last function call indicates
                        #   that this has happened and that we need to add
                        #   2**18 to the progess values.
                        n_acq_add = [
                            n_add + (2 ** 18 if n < n_last else 0) if n > 0
                            else n_add + n_last
                            for n, n_last, n_add in zip(n_acq, n_acq_last,
                                                        n_acq_add)]
                        n_acq_last = n_acq
                        n_acq = np.array(n_acq) + np.array(n_acq_add)
                        if any([n > 0 for n in n_acq]):
                            # the following calculation works both if
                            # self.nr_averages is a vector/list or a scalar
                            progress = np.mean(np.multiply(
                                n_acq, 1 / np.array(self.nr_averages)))
                            self.progress_callback(progress)
                    except NoProgressError:
                        raise
                    except Exception as e:
                        # printing progress is optional
                        log.debug(f'poll_data: Could not print progress: {e}')
                    t_callback = t_now

        self.timer.checkpoint("UHFQC_Base.poll_data.loop.end")

        if not all(np.concatenate(list(gotem.values()))):
            for UHF in self.UHFs:
                UHF.acquisition_finalize()
                for n, c in enumerate(UHF._acquisition_nodes):
                    if n in data[UHF.name]:
                        n_swp = len(data[UHF.name][n])
                        tot_swp = self.detectors[
                            self.UHF_map[UHF.name]].nr_sweep_points
                        log.info(f"\t: Channel {n}: Got {n_swp} of {tot_swp} "
                                 f"samples")
                raise TimeoutError("Error: Didn't get all results!")

        data_raw = {UHF.name: np.array([data[UHF.name][key]
                    for key in sorted(data[UHF.name].keys())]) for UHF in
                    self.UHFs}

        return data_raw

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()

        for d in self.detectors:
            d.UHFQC.acquisition_finalize()


class UHFQC_multi_detector(UHFQC_Base):
    """
    Combines several UHF detectors into a single detector
    """
    def __init__(self, detectors, **kw):
        super().__init__(detectors=detectors, **kw)
        self.AWG = None
        self.value_names = []
        self.value_units = []
        self.live_plot_enabled = []  # to be used by MC

        for d in self.detectors:
            self.value_names += [vn + ' ' + d.UHFQC.name for vn in d.value_names]
            self.value_units += d.value_units
            self.live_plot_enabled += [d.live_plot_enabled]
            if d.AWG is not None:
                if self.AWG is None:
                    self.AWG = d.AWG
                elif self.AWG != d.AWG:
                    raise Exception('Not all AWG instances in UHFQC_multi_detector'
                                    ' are the same')
                d.AWG = None

        # if any of the detectors is doing an SSRO acquisition, then disable
        # live plotting
        self.live_plot_enabled = all(self.live_plot_enabled)
        # to be used in MC.get_percdone()
        self.acq_data_len_scaling = \
            self.detectors[0].acq_data_len_scaling

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

    @Timer()
    def prepare(self, sweep_points):
        for d in self.detectors:
            d.prepare(sweep_points)
        self.nr_averages = [
            getattr(d, 'nr_averages', None) for d in self.detectors]
        if any([a is None for a in self.nr_averages]):
            self.nr_averages = None

    def get_values(self):
        raw_data = self.poll_data()

        processed_data = [self.detectors[self.UHF_map[UHF]].process_data(d)
                          for UHF, d in raw_data.items()]
        processed_data = np.concatenate(processed_data)
        if self.correlated:
            if not self.detectors[0].get_values_function_kwargs.get(
                    'averaged', True):
                data_for_corr = processed_data
            else:
                data_for_corr = np.concatenate([d for d in raw_data.values()])
            corr_data = self.get_correlations_classif_det(data_for_corr)
            processed_data = np.concatenate([processed_data, corr_data], axis=0)

        return processed_data

    def get_correlations_classif_det(self, data):
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

        processed_data = data
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
            processed_data = d0.process_data(data)
            d0.channel_str_mobj = d0_channel_str_mobj
            d0.get_values_function_kwargs = d0_get_values_function_kwargs

        # can only correlate corresponding probabilities on all channels;
        # it cannot correlate selected channels
        q = processed_data.shape[0] // nr_states  # nr of qubits
        # creates array with nr_sweep_points x nr_shots columns and
        # nr_qubits rows, where all entries are 0, 1, or 2. The entry in each
        # column is the state of the respective qubit, 0=g, 1=e, 2=f.
        qb_states_list = [np.argmax(
            processed_data[i * nr_states: i * nr_states + nr_states, :],
            axis=0) for i in range(q)]

        # correlate the shots of the two qubits as follows:
        # if both qubits are in g or f ---> correlator = 0
        # if both qubits are in e ---> correlator = 0
        # if one qubit is in g or f but the other in e ---> correlator = 1
        corr_data = np.sum(np.array(qb_states_list) % 2, axis=0) % 2
        if self.averaged:
            corr_data = np.reshape(corr_data, (d0.nr_shots, d0.nr_sweep_points))
            corr_data = np.mean(corr_data, axis=0)
        corr_data = np.reshape(corr_data, (1, corr_data.size))

        return corr_data

    def finish(self):
        if self.AWG is not None:
            self.AWG.stop()
        for d in self.detectors:
            d.finish()


class UHFQC_input_average_detector(UHFQC_Base):

    """
    Detector used for acquiring averaged timetraces withe the UHFQC
    """

    def __init__(self, UHFQC, AWG=None, channels=(0, 1),
                 nr_averages=1024, nr_samples=4096, **kw):
        super(UHFQC_input_average_detector, self).__init__(UHFQC, **kw)
        self.channels = channels
        self.value_names = ['']*len(self.channels)
        self.value_units = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = f'{UHFQC.name}_ch{channel}'
            self.value_units[i] = 'V'
            # UHFQC returned data is in Volts
            # January 2018 verified by Xavi
        self.AWG = AWG
        self.nr_samples = nr_samples - nr_samples % 4
        self.nr_averages = nr_averages

    def get_values(self):
        raw_data = self.poll_data()
        processed_data = self.process_data(raw_data[self.UHFQC.name])
        return processed_data

    def process_data(self, raw_data):
        # IMPORTANT: No re-scaling factor needed for input average mode as
        # the UHFQC returns volts
        # Verified January 2018 by Xavi
        return raw_data

    @Timer()
    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()
        self.nr_sweep_points = self.nr_samples
        self.ro_mode = 'iavg'
        self._check_hardware_limitations()
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels,
                                          samples=self.nr_samples,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages),
                                          mode=self.ro_mode)



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
                 nr_averages=20, nr_samples=4096, fft_mode='timedomain',
                 **kw):
        super().__init__(**kw)

        self.UHFQC = UHFQC
        self.scope = UHFQC.daq.scopeModule()
        self.AWG = AWG
        self.channels = channels
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


class UHFQC_integrated_average_detector(UHFQC_Base):

    """
    Detector used for integrated average results with the UHFQC

    Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                the experiment, can also be a central controller

        integration_length (float): integration length in seconds
        nr_averages (int)         : nr of averages per data point
            IMPORTANT: this must be a power of 2

        result_logging_mode (str) :  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.

        real_imag (bool)     : if False returns data in polar coordinates
                                useful for e.g., spectroscopy
                                #FIXME -> should be named "polar"
        single_int_avg (bool): if True makes this a soft detector

        Args relating to changing the amoung of points being detected:

        seg_per_point (int)  : number of segments per sweep point,
                does not do any renaming or reshaping.
                Here for deprecation reasons.
        chunk_size    (int)  : used in single shot readout experiments.
        values_per_point (int): number of values to measure per sweep point.
                creates extra column/value_names in the dataset for each channel.
        values_per_point_suffex (list): suffex to add to channel names for
                each value. should be a list of strings with lenght equal to
                values per point.
        always_prepare (bool) : when True the acquire/get_values method will
            first call the prepare statement. This is particularly important
            when it is both a single_int_avg detector and acquires multiple
            segments per point.
    """

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6, nr_averages: int=1024,
                 channels: list=(0, 1, 2, 3), result_logging_mode: str='raw',
                 real_imag: bool=True,
                 seg_per_point: int =1, single_int_avg: bool =False,
                 chunk_size: int=None,
                 values_per_point: int=1, values_per_point_suffex: list=None,
                 always_prepare: bool=False,
                 prepare_function=None, prepare_function_kwargs: dict=None,
                 **kw):

        super().__init__(UHFQC, **kw)

        self.name = '{}_UHFQC_integrated_average'.format(result_logging_mode)
        self.channels = deepcopy(channels)
        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = \
                '{}_{} w{}'.format(UHFQC.name, result_logging_mode, channel)
        if result_logging_mode == 'raw':
            # Units are only valid when using SSB or DSB demodulation.
            # value corrsponds to the peak voltage of a cosine with the
            # demodulation frequency.
            self.value_units = ['Vpeak']*len(self.channels)
            self.scaling_factor = 1/(1.8e9*integration_length)
        elif result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.channels)
            self.scaling_factor = 1
        elif result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.channels)
            self.scaling_factor = 1#/0.00146484375

        self.value_names, self.value_units = self._add_value_name_suffex(
            value_names=self.value_names, value_units=self.value_units,
            values_per_point=values_per_point,
            values_per_point_suffex=values_per_point_suffex)
        self.single_int_avg = single_int_avg
        if self.single_int_avg:
            self.detector_control = 'soft'
        # useful in combination with single int_avg
        self.always_prepare = always_prepare
        # Directly specifying seg_per_point is deprecated. values_per_point
        # replaces this functionality -MAR Dec 2017
        self.seg_per_point = max(seg_per_point, values_per_point)

        self.AWG = AWG
        self.nr_averages = nr_averages
        self.integration_length = integration_length
        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode
        self.chunk_size = chunk_size

        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        self._set_real_imag(real_imag)

    def _add_value_name_suffex(self, value_names: list, value_units: list,
                               values_per_point: int,
                               values_per_point_suffex: list):
        """
        For use with multiple values_per_point. Adds
        """
        if values_per_point == 1:
            return value_names, value_units
        else:
            new_value_names = []
            new_value_units = []
            if values_per_point_suffex is None:
                values_per_point_suffex = ascii_uppercase[:len(value_names)]

            for vn, vu in zip(value_names, value_units):
                for val_suffix in values_per_point_suffex:
                    new_value_names.append('{} {}'.format(vn, val_suffix))
                    new_value_units.append(vu)
            return new_value_names, new_value_units

    def _set_real_imag(self, real_imag=False):
        """
        Function so that real imag can be changed after initialization
        """

        self.real_imag = real_imag
        # Commented this out as it is already done in the init -MAR Dec 2017
        # if self.result_logging_mode == 'raw':
        #     self.value_units = ['V']*len(self.channels)
        # else:
        #     self.value_units = ['']*len(self.channels)

        if not self.real_imag:
            if len(self.channels) != 2:
                raise ValueError('Length of "{}" is not 2'.format(
                                 self.channels))
            self.value_names[0] = 'Magn'
            self.value_names[1] = 'Phase'
            self.value_units[1] = 'deg'

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])

        # if self.AWG is not None:
        #     self.AWG.stop()
        #
        #
        # # starting AWG
        # if self.AWG is not None:
        #     # self.AWG.start(exclude=['UHF1'])
        #     self.AWG.start()

        # time.sleep(0.1)
        #
        # data_raw = self.UHFQC.acquisition_poll(
        #     samples=self.nr_sweep_points, arm=False, acquisition_time=0.1)
        # the self.channels should be the same as data_raw.keys().
        # this is to be tested (MAR 26-9-2017)
        return data_processed

    def process_data(self, data_raw):
        data = data_raw * self.scaling_factor
        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel))
        if not self.real_imag:
            data = self.convert_to_polar(data)

        no_virtual_channels = len(self.value_names)//len(self.channels)

        data = np.reshape(data.T,
                          (-1, no_virtual_channels, len(self.channels))).T
        data = data.reshape((len(self.value_names), -1))

        return data

    def convert_to_polar(self, data):
        if len(data) != 2:
            raise ValueError('Expect 2 channels for rotation. Got {}'.format(
                             len(data)))
        I = data[0]
        Q = data[1]
        S21 = I + 1j*Q
        data[0] = np.abs(S21)
        data[1] = np.angle(S21)/(2*np.pi)*360
        return data

    def acquire_data_point(self):
        return self.get_values()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()

        # Determine the number of sweep points and set them
        if sweep_points is None or self.single_int_avg:
            # this case will be used when it is a soft detector
            # Note: single_int_avg overrides chunk_size
            # single_int_avg = True is equivalent to chunk_size = 1
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points)*self.seg_per_point

        # this sets the result to integration and rotation outcome
            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

            if (self.chunk_size is not None and
                    self.chunk_size < self.nr_sweep_points):
                # Chunk size is defined and smaller than total number of sweep
                # points -> only acquire one chunk
                self.nr_sweep_points = self.chunk_size * self.seg_per_point

        # Optionally perform extra actions on prepare
        # This snippet is placed here so that it has a chance to modify the
        # nr_sweep_points in a UHFQC detector
        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        self.ro_mode = 'rl'
        self._check_hardware_limitations()

        # Do not enable the rerun button; the AWG program uses userregs/0 to
        # define the number of iterations in the loop
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.qas_0_integration_length(int(self.integration_length*1.8e9))
        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_sweep_points,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages),
                                          mode=self.ro_mode)


class UHFQC_correlation_detector(UHFQC_integrated_average_detector):
    """
    Detector used for correlation mode with the UHFQC.
    The argument 'correlations' is a list of tuples specifying which channels
    are correlated, and on which channel the correlated signal is output.
    For instance, 'correlations=[(0, 1, 3)]' will put the correlation of
    channels 0 and 1 on channel 3.
    """

    def __init__(self, UHFQC, AWG=None, integration_length=1e-6,
                 nr_averages=1024,  real_imag=True,
                 channels: list = [0, 1], correlations: list=[(0, 1)],
                 result_logging_mode: str='raw',
                 used_channels=None, value_names=None,
                 seg_per_point=1, single_int_avg=False,
                 **kw):
        super().__init__(
            UHFQC, AWG=AWG, integration_length=integration_length,
            nr_averages=nr_averages, real_imag=real_imag,
            channels=channels,
            seg_per_point=seg_per_point, single_int_avg=single_int_avg,
            result_logging_mode=result_logging_mode,
            **kw)

        self.name = '{}_UHFQC_correlation_detector'.format(result_logging_mode)
        self.result_logging_mode = result_logging_mode
        self.correlations = correlations
        self.thresholding = self.result_logging_mode == 'digitized'
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        self.result_logging_mode_idx = res_logging_indices[
            self.result_logging_mode]

        self.used_channels = used_channels
        if self.used_channels is None:
            self.used_channels = self.channels

        if value_names is None:
            self.value_names = []
            for ch in channels:
                self.value_names += ['{}_{} w{}'.format(
                    UHFQC.name, self.result_logging_mode, ch)]
        else:
            self.value_names = value_names

        # Note that V^2 is in brackets to prevent confusion with unit prefixes
        if self.result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.value_names) + \
                               ['(V^2)']*len(self.correlations)
            self.scaling_factor = 1/(1.8e9*integration_length)
        elif self.result_logging_mode == 'lin_trans':
            self.value_units = ['a.u.']*len(self.value_names) + \
                               ['a.u.']*len(self.correlations)
            self.scaling_factor = 1
        elif self.result_logging_mode == 'digitized':
            self.value_units = ['frac']*len(self.value_names) + \
                               ['normalized']*len(self.correlations)
            self.scaling_factor = 1

        for corr in correlations:
            self.value_names += ['corr ({},{})'.format(corr[0], corr[1])]

        self.define_correlation_channels()

    def prepare(self, sweep_points=None):
        if self.AWG is not None:
            self.AWG.stop()
        if sweep_points is None or self.single_int_avg:
            self.nr_sweep_points = self.seg_per_point
        else:
            self.nr_sweep_points = len(sweep_points) * self.seg_per_point
        self._check_hardware_limitations()

        self.UHFQC.qas_0_integration_length(int(self.integration_length*(1.8e9)))

        self.set_up_correlation_weights()

        self.ro_mode = 'rl'
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels,
                                          samples=self.nr_sweep_points,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_averages*self.nr_sweep_points),
                                          mode=self.ro_mode)

    def define_correlation_channels(self):
        self.correlation_channels = []
        used_channels = deepcopy(self.used_channels)
        for corr in self.correlations:
            # Start by assigning channels
            if corr[0] not in used_channels or corr[1] not in used_channels:
                raise ValueError('Correlations should be in used channels')

            correlation_channel = -1

            # # 9 is the (current) max number of weights in the UHFQA
            for ch in range(9):
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

            if correlation_channel < 0:
                raise ValueError('No free channel available for correlation.')
            else:
                used_channels += [ch]

    def set_up_correlation_weights(self):
        if self.thresholding:
            # correlations mode after threshold
            # NOTE: thresholds need to be set outside the detctor object.
            self.UHFQC.qas_0_result_source(5)
        else:
            # correlations mode before threshold
            self.UHFQC.qas_0_result_source(4)
        # Configure correlation mode
        for ch in self.channels:
            if ch not in self.correlation_channels:
                # Disable correlation mode as this is used for normal
                # acquisition
                self.UHFQC.set('qas_0_correlations_{}_enable'.format(ch), 0)

        for correlation_channel, corr in zip(self.correlation_channels,
                                             self.correlations):
            # Duplicate source channel to the correlation channel and select
            # second channel as channel to correlate with.
            copy_int_weights_real = \
                np.array(self.UHFQC.get(
                    f'qas_0_integration_weights_{corr[0]}_real')).astype(float)
            copy_int_weights_imag = \
                np.array(self.UHFQC.get(
                    f'qas_0_integration_weights_{corr[0]}_imag')).astype(float)

            copy_rot_matrix = self.UHFQC.get('qas_0_rotations_{}'.format(corr[0]))

            self.UHFQC.set(
                'qas_0_integration_weights_{}_real'.format(correlation_channel),
                copy_int_weights_real)
            self.UHFQC.set(
                'qas_0_integration_weights_{}_imag'.format(correlation_channel),
                copy_int_weights_imag)

            self.UHFQC.set(
                'qas_0_rotations_{}'.format(correlation_channel),
                copy_rot_matrix)

            # Enable correlation mode one the correlation output channel and
            # set the source to the second source channel
            self.UHFQC.set(f'qas_0_correlations_{correlation_channel}_enable', 1)
            self.UHFQC.set(f'qas_0_correlations_{correlation_channel}_source',
                           corr[1])

            # If thresholding is enabled, set the threshold for the correlation
            # channel.
            if self.thresholding:
                thresh_level = \
                    self.UHFQC.get('qas_0_thresholds_{}_level'.format(corr[0]))
                self.UHFQC.set(
                    'qas_0_thresholds_{}_level'.format(correlation_channel),
                    thresh_level)

    def get_values(self):
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])
        return data_processed

    def process_data(self, data_raw):
        if self.thresholding:
            data = data_raw
        else:
            data = []
            for n, ch in enumerate(self.used_channels):
                if ch in self.correlation_channels:
                    data.append(1.5 * np.array(data_raw[n]) *
                                self.scaling_factor**2)
                else:
                    data.append(np.array(data_raw[n]) * self.scaling_factor)
        return data


class UHFQC_integration_logging_det(UHFQC_Base):

    """
    Detector used for integrated single-shot results with the UHFQC

    Args:
        UHFQC (instrument) : data acquisition device
        AWG   (instrument) : device responsible for starting and stopping
                             the experiment, can also be a central controller.
        integration_length (float): integration length in seconds
        nr_shots (int)     : nr of shots (max is 4095)
        channels (list)    : index (channel) of UHFQC weight functions to use

        result_logging_mode (str):  options are
            - raw        -> returns raw data in V
            - lin_trans  -> applies the linear transformation matrix and
                            subtracts the offsets defined in the UFHQC.
                            This is typically used for crosstalk suppression
                            and normalization. Requires optimal weights.
            - digitized  -> returns fraction of shots based on the threshold
                            defined in the UFHQC. Requires optimal weights.
        always_prepare (bool) : when True the acquire/get_values method will
            first call the prepare statement. This is particularly important
            when it is both a single_int_avg detector and acquires multiple
            segments per point.
    """

    def __init__(self, UHFQC, AWG=None,
                 integration_length: float=1e-6,
                 nr_shots: int=4094,
                 channels: list=(0, 1),
                 result_logging_mode: str='raw',
                 always_prepare: bool=False,
                 prepare_function=None,
                 prepare_function_kwargs: dict=None,
                 **kw):

        super().__init__(UHFQC, **kw)

        self.name = '{}_UHFQC_integration_logging_det'.format(
            result_logging_mode)
        self.channels = channels

        self.value_names = ['']*len(self.channels)
        for i, channel in enumerate(self.channels):
            self.value_names[i] = \
                '{}_{} w{}'.format(UHFQC.name, result_logging_mode, channel)
        if result_logging_mode == 'raw':
            self.value_units = ['V']*len(self.channels)
            self.scaling_factor = 1  # /(1.8e9*integration_length)
        else:
            self.value_units = ['']*len(self.channels)
            self.scaling_factor = 1

        self.AWG = AWG
        self.integration_length = integration_length
        self.nr_shots = nr_shots
        self.nr_averages = 1  # for single shot readout
        # to be used in MC
        self.acq_data_len_scaling = self.nr_shots

        # 0/1/2 crosstalk supressed /digitized/raw
        res_logging_indices = {'lin_trans': 0, 'digitized': 1, 'raw': 2}
        # mode 3 is statistics logging, this is implemented in a
        # different detector
        self.result_logging_mode_idx = res_logging_indices[result_logging_mode]
        self.result_logging_mode = result_logging_mode

        self.always_prepare = always_prepare
        self.prepare_function = prepare_function
        self.prepare_function_kwargs = prepare_function_kwargs
        # Disable MC live plotting by default for SSRO acquisition
        self.live_plot_enabled = kw.get('live_plot_enabled', False)

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()

        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        self.nr_sweep_points = len(sweep_points)
        self._check_hardware_limitations()

        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)
        self.ro_mode = 'rl'
        self.UHFQC.qas_0_integration_length(int(self.integration_length*1.8e9))
        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(channels=self.channels, 
                                          samples=self.nr_sweep_points,
                                          averages=self.nr_averages,
                                          loop_cnt=int(self.nr_shots),
                                          mode=self.ro_mode)

    def get_values(self):
        if self.always_prepare:
            self.prepare()
        data_raw = self.poll_data()
        data_processed = self.process_data(data_raw[self.UHFQC.name])
        return data_processed

    def process_data(self, data_raw):
        data = data_raw * self.scaling_factor

        # Corrects offsets after crosstalk suppression matrix in UFHQC
        if self.result_logging_mode == 'lin_trans':
            for i, channel in enumerate(self.channels):
                data[i] = data[i]-self.UHFQC.get(
                    'qas_0_trans_offset_weightfunction_{}'.format(channel))
        return data


class UHFQC_classifier_detector(UHFQC_integration_logging_det):
    """
    Hybrid detector function to be used for qutrit readout:
     - only works with 2 readout channels per qutrit!
     - the UHF is configured to return single shots, but this function can then
     return either single shots, or averaged sweep points (can average over the
     shots for each segment)
     - This class always performs classification of 2-channel single shots data
     (shot_i_ch0, shot_i_ch1) into a 3 state probability
     (shot_i_pg, shot_i_pe, shot_i_pf), for all recorded shots. Here
     shot_i_pg, shot_i_pe, shot_i_pf \in [0, 1] and
     shot_i_pg + shot_i_pe + shot_i_pf = 1.

    See the UHFQC_integration_logging_det for the full dostring of the accepted
    input parameters.

    The only additional keyword argument that is used by this class but not by
    its parent class is get_values_function_kwargs. This parameter is a dict
    where the user can specify how he wants this detector function to process
    the shots.
    get_values_function_kwargs can contain:
     - classifier_params_list (list or dict): THIS ENTRY MUST EXIST. This class
        always classifies qutrits and it does so based on this parameter.
        It is either the QuDev_transmon acq_classifier_params attribute (if only
        one qubit is measured), or list of these dictionaries (if multiple
        qubits are measured).
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
    """

    def __init__(self, UHFQC, *args, **kw):
        self.qutrit = kw.pop('qutrit', True)
        super().__init__(UHFQC, *args, **kw)

        self.get_values_function_kwargs = kw.get('get_values_function_kwargs',
                                                 None)
        self.name = '{}_UHFQC_classifier_det'.format(self.result_logging_mode)

        self.state_labels = ['pg', 'pe', 'pf'] if self.qutrit else ['pg', 'pe']
        classifier_params = self.get_values_function_kwargs.get(
            'classifier_params', [])
        self.n_meas_objs = 1 if not len(classifier_params) else \
                len(classifier_params)
        k = len(self.channels) // self.n_meas_objs
        self.channel_str_mobj = [str(ch) for ch in self.channels]
        self.channel_str_mobj = [''.join(self.channel_str_mobj[k*j: k*j+k])
                                 for j in range(self.n_meas_objs)]

        self.classified = self.get_values_function_kwargs.get('classified', True)
        if self.classified:
            self.value_names = ['']*(
                    len(self.state_labels) * len(self.channel_str_mobj))
            idx = 0
            for ch_pair in self.channel_str_mobj:
                for state in self.state_labels:
                    self.value_names[idx] = '{}_{} w{}'.format(UHFQC.name,
                                                               state, ch_pair)
                    idx += 1
            self.value_units = [self.value_units[0]] * len(self.value_names)

        if self.get_values_function_kwargs.get('averaged', True):
            self.acq_data_len_scaling = 1  # to be used in MC
            # The following value is only used for correct progress
            # calculation in poll_data. It will not influence the behavior of
            # the acquisition device since UHFQC_classifier_detector.prepare
            # passes averages=1 to qudev_acquisition_initialize.
            self.nr_averages = self.nr_shots
            self.live_plot_enabled = kw.get('live_plot_enabled', True)
        else:
            # Disable MC live plotting by default for SSRO acquisition
            self.live_plot_enabled = kw.get('live_plot_enabled', False)

    def prepare(self, sweep_points):
        if self.AWG is not None:
            self.AWG.stop()

        if self.prepare_function_kwargs is not None:
            if self.prepare_function is not None:
                self.prepare_function(**self.prepare_function_kwargs)
        else:
            if self.prepare_function is not None:
                self.prepare_function()

        assert len(sweep_points) % self.acq_data_len_scaling == 0
        self.nr_sweep_points = len(sweep_points) // self.acq_data_len_scaling
        self.ro_mode = 'rl'
        self._check_hardware_limitations()

        # The averaging-count is used to specify how many times the AWG program
        # should run
        self.UHFQC.awgs_0_single(1)
        self.UHFQC.qas_0_integration_length(int(self.integration_length*1.8e9))

        self.UHFQC.qas_0_result_source(self.result_logging_mode_idx)
        self.UHFQC.qudev_acquisition_initialize(
            channels=self.channels,
            samples=self.nr_shots * self.nr_sweep_points,
            averages=1,  # for single shot readout
            loop_cnt=int(self.nr_shots), mode=self.ro_mode)

    def process_data(self, data_raw):
        processed_data = super().process_data(data_raw).T
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
            processed_data = self.classify_shots(processed_data,
                                                 self.classifier_params_list,
                                                 nr_states)

        if thresholded:
            if not self.classified:
                raise NotImplementedError(
                    'Currently the threshold_shots only works if the data '
                    'was first classified.')
            processed_data = self.threshold_shots(processed_data, nr_states)

        if averaged:
            processed_data = self.average_shots(processed_data)

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
            processed_data = self.correct_readout_stored_mtx(processed_data,
                                                             nr_states)
        elif ro_corrected_seq_cal_mtx:
            # correct data with the calibration matrix extracted from
            # the data array
            if not averaged:
                raise NotImplementedError(
                    'Data correction based on calibration state_prob_mtx '
                    'from measurement sequence is currently only '
                    'implemented for averaged data (averaged==True).')
            processed_data = self.correct_readout_seq_cal_mtx(processed_data,
                                                              nr_states)

        return processed_data.T

    def classify_shots(self, data, classifier_params_list, nr_states):
        classified_data = np.zeros((self.nr_sweep_points*self.nr_shots,
                                    nr_states*len(self.channel_str_mobj)))
        k = len(self.channels) // self.n_meas_objs
        for i in range(len(self.channel_str_mobj)):
            # classify each shot into (pg, pe, pf)
            # clf_data will have shape
            # (nr_shots * self.nr_sweep_points, nr_states)
            mobj_data = data[:, k*i: k*i+k]
            clf_data = a_tools.predict_gm_proba_from_clf(
                mobj_data, classifier_params_list[i])
            classified_data[:, nr_states * i: nr_states * i + nr_states] = \
                clf_data

        return classified_data

    def threshold_shots(self, data, nr_states):
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
        # reshape into (nr_shots, nr_sweep_points, nr_data_columns) then
        # average over nr_shots
        averaged_data = np.reshape(
            data, (self.nr_shots, self.nr_sweep_points, data.shape[-1]))
        # average over shots
        averaged_data = np.mean(averaged_data, axis=0)

        return averaged_data

    def correct_readout_stored_mtx(self, data, nr_states):
        # correct data with matrices from state_prob_mtx_list
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
        # correct data with the calibration matrix extracted from
        # the data array
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