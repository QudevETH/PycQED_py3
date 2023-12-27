import os
import sys
import numpy as np
import h5py
import json
import time
import datetime
import pickle
from pycqed.utilities.io import hdf5 as h5d
from pycqed.analysis import analysis_toolbox as a_tools
import errno
import pycqed as pq
import glob
from os.path import dirname, exists
from os import makedirs
import logging
import subprocess
from functools import reduce  # forward compatibility for Python 3
import operator
import string
import functools
from zipfile import ZipFile


from copy import deepcopy
log = logging.getLogger(__name__)
try:
    import msvcrt  # used on windows to catch keyboard input
except:
    pass

digs = string.digits + string.ascii_letters


def get_git_info():
    """
    Returns the SHA1 ID (hash) of the current git HEAD plus a diff against the HEAD
    The hash is shortened to the first 10 digits.

    :return: hash string, diff string
    """

    diff = "Could not extract diff"
    githash = '00000'
    try:
        # Refers to the global qc_config
        PycQEDdir = pq.__path__[0]
        githash = subprocess.check_output(['git', 'rev-parse',
                                           '--short=10', 'HEAD'], cwd=PycQEDdir)
        diff = subprocess.run(['git', '-C', PycQEDdir, "diff"],
                              stdout=subprocess.PIPE).stdout.decode('utf-8')
    except Exception:
        pass
    return githash, diff


def str_to_bool(s):
    valid = {'true': True, 't': True, '1': True,
             'false': False, 'f': False, '0': False, }
    if s.lower() not in valid:
        raise KeyError('{} not a valid boolean string'.format(s))
    b = valid[s.lower()]
    return b


def bool_to_int_str(b):
    if b:
        return '1'
    else:
        return '0'


def int_to_bin(x, w, lsb_last=True):
    """
    Converts an integer to a binary string of a specified width
    x (int) : input integer to be converted
    w (int) : desired width
    lsb_last (bool): if False, reverts the string e.g., int(1) = 001 -> 100
    """
    bin_str = '{0:{fill}{width}b}'.format((int(x) + 2**w) % 2**w,
                                          fill='0', width=w)
    if lsb_last:
        return bin_str
    else:
        return bin_str[::-1]


def int2base(x: int, base: int, fixed_length: int=None):
    """
    Convert an integer to string representation in a certain base.
    Useful for e.g., iterating over combinations of prepared states.

    Args:
        x    (int)          : the value to convert
        base (int)          : the base to covnert to
        fixed_length (int)  : if specified prepends zeros
    """
    if x < 0:
        sign = -1
    elif x == 0:
        string_repr = digs[0]
        if fixed_length is None:
            return string_repr
        else:
            return string_repr.zfill(fixed_length)

    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()
    string_repr = ''.join(digits)
    if fixed_length is None:
        return string_repr
    else:
        return string_repr.zfill(fixed_length)


def mopen(filename, mode='w'):
    if not exists(dirname(filename)):
        try:
            makedirs(dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    file = open(filename, mode='w')
    return file


def dict_to_ordered_tuples(dic):
    '''Convert a dictionary to a list of tuples, sorted by key.'''
    if dic is None:
        return []
    keys = dic.keys()
    # keys.sort()
    ret = [(key, dic[key]) for key in keys]
    return ret


def to_hex_string(byteval):
    '''
    Returns a hex representation of bytes for printing purposes
    '''
    return "b'" + ''.join('\\x{:02x}'.format(x) for x in byteval) + "'"


def load_settings(instrument,
                  label: str='', folder: str=None,
                  timestamp: str=None, update=True, **kw):
    '''
    Loads settings from an hdf5 file onto the instrument handed to the
    function. By default uses the last hdf5 file in the datadirectory.
    By giving a label or timestamp another file can be chosen as the
    settings file.

    Args:
        instrument (instrument) : instrument onto which settings
            should be loaded. Can be an instrument name (str) if update is
            set to False.
        label (str)           : label used for finding the last datafile
        folder (str)        : exact filepath of the hdf5 file to load.
            if filepath is specified, this takes precedence over the file
            locating options (label, timestamp etc.).
        timestamp (str)       : timestamp of file in the datadir
        update (bool, default True): if set to False, the loaded settings
            will be returned instead of updating them in the instrument.

    Kwargs:
        params_to_set (list)    : list of strings referring to the parameters
            that should be set for the instrument
    '''
    from numpy import array  # DO not remove. Used in eval(array(...))
    from collections import OrderedDict  # DO NOT remove. Used in eval()
    from pycqed.utilities import settings_manager as setman
    if folder is None:
        folder_specified = False
    else:
        folder_specified = True

    if isinstance(instrument, str) and not update:
        instrument_name = instrument
    else:
        instrument_name = instrument.name
    verbose = kw.pop('verbose', True)
    older_than = kw.pop('older_than', None)
    success = False
    count = 0
    # Will try multiple times in case the last measurements failed and
    # created corrupt data files.
    while success is False and count < 10:
        if folder is None:
            folder = a_tools.get_folder(timestamp=timestamp, label=label,
                                        older_than=older_than)
        if verbose:
            print('Folder used: {}'.format(folder))

        try:
            station = setman.get_station_from_file(folder=folder,
                                                   param_path=[instrument_name])
            ins_loaded = station.components[instrument_name]

            if verbose:
                print('Loaded settings successfully from the file.')
            params_to_set = kw.pop('params_to_set', None)
            if params_to_set is not None:
                if len(params_to_set) == 0:
                    log.warning('The list of parameters to update is empty.')
                if verbose and update:
                    print('Setting parameters {} for {}.'.format(
                        params_to_set, instrument_name))
                params_to_set = [(param, val()) for (param, val) in
                                 ins_loaded.parameters.items() if param in
                                 params_to_set]
            else:
                if verbose and update:
                    print('Setting parameters for {}.'.format(instrument_name))
                params_to_set = [
                    (param, val()) for (param, val) in ins_loaded.parameters.items()
                    if param not in getattr(
                        instrument, '_params_to_not_load', {})]

            if not update:
                params_dict = {parameter : value for parameter, value in \
                        params_to_set}
                return params_dict

            for parameter, value in params_to_set:
                if parameter in instrument.parameters.keys() and \
                        hasattr(instrument.parameters[parameter], 'set'):
                    try:
                        instrument.set(parameter, value)
                    except Exception:
                        log.error('Could not set parameter '
                                  '"%s" to "%s" '
                                  'for instrument "%s"' % (
                                      parameter, value,
                                      instrument_name))
            success = True
        except Exception as e:
            logging.warning(e)
            success = False
            if timestamp is None and not folder_specified:
                print('Trying next folder.')
                older_than = os.path.split(folder)[0][-8:] \
                             + '_' + os.path.split(folder)[1][:6]
                folder = None
            else:
                break
        count += 1

    if not success:
        log.error('Could not open settings for instrument {}.'.format(
            instrument_name))
    print()
    return


def load_settings_onto_instrument_v2(instrument, load_from_instr: str=None,
                                     label: str='', folder: str=None,
                                     timestamp: str=None):
    '''
    Loads settings from an hdf5 file onto the instrument handed to the
    function. By default uses the last hdf5 file in the datadirectory.
    By giving a label or timestamp another file can be chosen as the
    settings file.

    Args:
        instrument (instrument) : instrument onto which settings should be
            loaded
        load_from_instr (str) : optional name of another instrument from
            which to load the settings.
        label (str)           : label used for finding the last datafile
        folder (str)        : exact filepath of the hdf5 file to load.
            if filepath is specified, this takes precedence over the file
            locating options (label, timestamp etc.).
        timestamp (str)       : timestamp of file in the datadir


    '''

    older_than = None
    # folder = None
    instrument_name = instrument.name
    success = False
    count = 0
    # Will try multiple times in case the last measurements failed and
    # created corrupt data files.
    while success is False and count < 10:
        try:
            if folder is None:
                folder = a_tools.get_folder(timestamp=timestamp, label=label,
                                            older_than=older_than)
                filepath = a_tools.measurement_filename(folder)

            f = h5py.File(filepath, 'r')
            snapshot = {}
            h5d.read_dict_from_hdf5(snapshot, h5_group=f['Snapshot'])

            if load_from_instr is None:
                ins_group = snapshot['instruments'][instrument_name]
            else:
                ins_group = snapshot['instruments'][load_from_instr]
            success = True
        except Exception as e:
            logging.warning(e)
            older_than = os.path.split(folder)[0][-8:] \
                + '_' + os.path.split(folder)[1][:6]
            folder = None
            success = False
        count += 1

    if not success:
        logging.warning('Could not open settings for instrument "%s"' % (
            instrument_name))
        try:
            f.close()
        except:
            pass
        return False

    for parname, par in ins_group['parameters'].items():
        try:
            if hasattr(instrument.parameters[parname], 'set'):
                instrument.set(parname, par['value'])
        except Exception as e:
            print('Could not set parameter: "{}" to "{}" '
                  'for instrument "{}"'.format(parname, par['value'],
                                               instrument_name))
            logging.warning(e)
    f.close()
    return True



def send_email(subject='PycQED needs your attention!',
               body='', email=None):
    # Import smtplib for the actual sending function
    import smtplib
    # Here are the email package modules we'll need
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if email is None:
        email = qt.config['e-mail']

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = subject
    family = 'serwan.asaad@gmail.com'
    msg['From'] = 'Lamaserati@tudelft.nl'
    msg['To'] = email
    msg.attach(MIMEText(body, 'plain'))

    # Send the email via our own SMTP server.
    s = smtplib.SMTP_SSL('smtp.gmail.com')
    s.login('DCLabemail@gmail.com', 'DiCarloLab')
    s.sendmail(email, family, msg.as_string())
    s.quit()


def list_available_serial_ports():
    '''
    Lists serial ports

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of available serial ports

    Frunction from :
    http://stackoverflow.com/questions/12090503/
        listing-available-com-ports-with-python
    '''
    import serial
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(256)]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')

    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')

    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def add_suffix_to_dict_keys(inputDict, suffix):
    return {str(key)+suffix: (value) for key, value in inputDict.items()}


def execfile(path, global_vars=None, local_vars=None):
    """
    Args:
        path (str)  : filepath of the file to be executed
        global_vars : use globals() to use globals from namespace
        local_vars  : use locals() to use locals from namespace

    execfile function that existed in python 2 but does not exists in python3.
    """
    with open(path, 'r') as f:
        code = compile(f.read(), path, 'exec')
        exec(code, global_vars, local_vars)


def span_num(center: float, span: float, num: int, endpoint: bool=True):
    """
    Creates a linear span of points around center
    Args:
        center (float) : center of the array
        span   (float) : span the total range of values to span
        num      (int) : the number of points in the span
        endpoint (bool): whether to include the endpoint

    """
    return np.linspace(center-span/2, center+span/2, num, endpoint=endpoint)


def span_step(center: float, span: float, step: float, endpoint: bool=True):
    """
    Creates a range of points spanned around a center
    Args:
        center (float) : center of the array
        span   (float) : span the total range of values to span
        step   (float) : the stepsize between points in the array
        endpoint (bool): whether to include the endpoint in the span

    """
    # True*step/100 in the arange ensures the right boundary is included
    return np.arange(center-span/2, center+span/2+endpoint*step/100, step)


def gen_sweep_pts(start: float=None, stop: float=None,
                  center: float=0, span: float=None,
                  num: int=None, step: float=None, endpoint=True):
    """
    Generates an array of sweep points based on different types of input
    arguments.
    Boundaries of the array can be specified using either start/stop or
    using center/span. The points can be specified using either num or step.

    Args:
        start  (float) : start of the array
        stop   (float) : end of the array
        center (float) : center of the array
                         N.B. 0 is chosen as a sensible default for the span.
                         it is argued that no such sensible default exists
                         for the other types of input.
        span   (float) : span the total range of values to span

        num      (int) : number of points in the array
        step   (float) : the stepsize between points in the array
        endpoint (bool): whether to include the endpoint

    """
    if (start is not None) and (stop is not None):
        if num is not None:
            return np.linspace(start, stop, num, endpoint=endpoint)
        elif step is not None:
            # numpy arange does not natively support endpoint
            return np.arange(start, stop + endpoint*step/100, step)
        else:
            raise ValueError('Either "num" or "step" must be specified')
    elif (center is not None) and (span is not None):
        if num is not None:
            return span_num(center, span, num, endpoint=endpoint)
        elif step is not None:
            return span_step(center, span, step, endpoint=endpoint)
        else:
            raise ValueError('Either "num" or "step" must be specified')
    else:
        raise ValueError('Either ("start" and "stop") or '
                         '("center" and "span") must be specified')


def getFromDict(dataDict: dict, mapList: list):
    """
    get a value from a nested dictionary by specifying a list of keys

    Args:
        dataDict: nested dictionary to get the value from
        mapList : list of strings specifying the key of the item to get
    Returns:
        value from dictionary

    example:
        example_dict = {'a': {'nest_a': 5, 'nest_b': 8}
                        'b': 4}
        getFromDict(example_dict, ['a', 'nest_a']) -> 5
    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict: dict, mapList: list, value):
    """
    set a value in a nested dictionary by specifying the location using a list
    of key.

    Args:
        dataDict: nested dictionary to set the value in
        mapList : list of strings specifying the key of the item to set
        value   : the value to set

    example:
        example_dict = {'a': {'nest_a': 5, 'nest_b': 8}
                        'b': 4}
        example_dict_after = getFromDict(example_dict, ['a', 'nest_a'], 6)
        example_dict = {'a': {'nest_a': 6, 'nest_b': 8}
                        'b': 4}
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def is_more_rencent(filename: str, comparison_filename: str):
    """
    Returns True if the contents of "filename" has changed more recently
    than the contents of "comparison_filename".
    """
    return os.path.getmtime(filename) > os.path.getmtime(comparison_filename)


def get_required_upload_information(pulses : list, station):
    """
    Returns a list of AWGs required for the list of input pulses
    """

    #Have to add all master AWG channels such that trigger channels are not empty
    master_AWG = station.pulsar.master_AWG()
    required_AWGs = []
    required_channels = []
    used_AWGs = station.pulsar.used_AWGs()


    for pulse in pulses:
        for key in pulse.keys():
            if not 'channel' in key:
                continue
            channel = pulse[key]
            if isinstance(channel, dict):
                # the the CZ pulse has aux_channels_dict parameter
                for ch in channel:
                    if not 'AWG' in ch:
                        continue
                    AWG = ch.split('_')[0]
                    if AWG == master_AWG:
                        for c in station.pulsar.channels:
                            if master_AWG in c and c not in required_channels:
                                required_channels.append(c)
                            if AWG in used_AWGs and AWG not in required_AWGs:
                                required_AWGs.append(AWG)
                            continue
                    if AWG in used_AWGs and AWG not in required_AWGs:
                        required_AWGs.append(AWG)
                    if not ch in required_channels:
                        required_channels.append(ch)
            else:
                if not 'AWG' in channel:
                    continue
                AWG = channel.split('_')[0]
                if AWG == master_AWG:
                    for c in station.pulsar.channels:
                        if master_AWG in c and c not in required_channels:
                            required_channels.append(c)
                        if AWG in used_AWGs and AWG not in required_AWGs:
                            required_AWGs.append(AWG)
                        continue
                if AWG in used_AWGs and AWG not in required_AWGs:
                    required_AWGs.append(AWG)
                if not channel in required_channels:
                    required_channels.append(channel)

    return required_channels, required_AWGs

def dictionify(obj, only=None, exclude=None):
    """
    Takes an arbitrary object and returns a dict with all variables/internal
    states of the object (i.e. not functions)
    Args:
        obj: object
        only (list): take only specific attributes
        exclude (list): exclude specific attributes

    Returns: dict form of the object

    """
    obj_dict = vars(obj)
    if only is not None:
        assert np.ndim(only) == 1, "'only' must be of type list or array " \
                                   "of attributes to include"
        for k in obj_dict:
            if k not in only:
                obj_dict.pop(k)
    if exclude is not None:
        assert np.ndim(exclude) == 1, "'exclude' must be a list or array of" \
                                      " attributes to exclude"
        for k in obj_dict:
            if k in exclude:
                obj_dict.pop(k)
    return obj_dict

class NumpyJsonEncoder(json.JSONEncoder):
    '''
    JSON encoder subclass that converts Numpy types to native python types
    for saving in JSON files.
    Also converts datetime objects to strings.
    '''
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, datetime.datetime):
            return str(o)
        else:
            return super().default(o)


class KeyboardFinish(KeyboardInterrupt):
    """
    Indicates that the user safely aborts/finishes the experiment.
    Used to finish the experiment without raising an exception.
    """

    pass


def check_keyboard_interrupt():
    try:  # Try except statement is to make it work on non windows pc
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if b"q" in key:
                # this causes a KeyBoardInterrupt
                raise KeyboardInterrupt('Human "q" terminated experiment.')
            elif b"f" in key:
                # this should not raise an exception
                raise KeyboardFinish('Human "f" terminated experiment safely.')
    except Exception:
        pass


class TemporaryValue:
    """
    This context manager allows to change a given QCodes parameter
    to a new value, and the original value is reverted upon exit of the context
    manager.

    Args:
        *param_value_pairs: 2-tuples of qcodes parameters and their temporary
                            values

    Example:
        # measure qubit spectroscopy at a different readout frequency without
        # setting the parameter value
        with TemporaryValue((qb1.ro_freq, 6e9)):
            qb1.measure_spectroscopy(...)
    """

    def __init__(self, *param_value_pairs):
        if len(param_value_pairs) > 0 and \
                not isinstance(param_value_pairs[0], (tuple, list)):
            param_value_pairs = (param_value_pairs,)
        self.param_value_pairs = param_value_pairs
        self.old_value_pairs = []

    def __enter__(self):
        log.debug('Entered TemporaryValueContext')
        try:
            self.old_value_pairs = \
                [(param, param()) for param, value in self.param_value_pairs]
            for param, value in self.param_value_pairs:
                param(value)
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, type, value, traceback):
        for param, value in self.old_value_pairs:
            param(value)
        log.debug('Exited TemporaryValueContext')


# Alias in order to have the class definition in camel case, but keep
# backwards compatibility
temporary_value = TemporaryValue


def configure_qubit_mux_drive(qubits, lo_freqs_dict):
    """Configure qubits for multiplexed drive.

    This helper function configures the given qubits for multiplexed drive
    as follows:
    - set the readout IF of the qubits such that the LO frequency of qubits
      sharing an LO is compatible.

    By passing a list with only a single qubit, the function can also be
    used to ensure that a given LO frequency is used for a qubit, even in a
    non-multiplexed drive setting.

    Args:
        qubits (list of qubit objects): The qubits for which the drive
            should be configured.
        lo_freqs_dict (dict): A dict where each key identifies a drive LO
            in one of the formats as returned by qb.get_ge_lo_identifier,
            and the corresponding value determines the LO frequency that
            this LO should use.
    """
    for qb in qubits:
        qb_ge_mwg = qb.get_ge_lo_identifier()
        if qb_ge_mwg not in lo_freqs_dict:
            log.info(f'{qb.name}: {qb_ge_mwg} not'
                     f'found in lo_freqs_dict.')
            continue
        qb.ge_mod_freq(qb.ge_freq()-lo_freqs_dict[qb_ge_mwg])


def configure_qubit_mux_readout(qubits, lo_freqs_dict, set_mod_freq=True):
    """Configure qubits for multiplexed readout.

    This helper function configures the given qubits for multiplexed readout
    as follows:
    - assign unique acquisition channels for qubits sharing an LO. This
      assumes that qubits sharing an LO also share an acquisition unit of an
      acquisition decvice.
    - set the readout IF of the qubits such that the LO frequency of qubits
      sharing an LO is compatible.

    By passing a list with only a single qubit, the function can also be
    used to ensure that a given LO frequency is used for a qubit, even in a
    non-multiplexed readout setting. Note that the acquisition I/Q channel
    indices of the qubit are set to 0 and 1 in this case.

    Args:
        qubits (list of qubit objects): The qubits for which the readout
            should be configured.
        lo_freqs_dict (dict): A dict where each key identifies a readout LO
            in one of the formats as returned by qb.get_ro_lo_identifier,
            and the corresponding value determines the LO frequency that
            this LO should use.
        set_mod_freq (bool): Specifies whether the qubits modulation frequency
            is adjusted according to the Lo freq. in lo_freqs_dict or not.
            Defaults to True.
    """
    idx = {}
    for qb in qubits:
        qb_ro_mwg = qb.get_ro_lo_identifier()
        idx[qb_ro_mwg] = idx.setdefault(qb_ro_mwg, -1) + 1
        qb.acq_I_channel(2 * idx[qb_ro_mwg])
        qb.acq_Q_channel(2 * idx[qb_ro_mwg] + 1)
        if qb_ro_mwg not in lo_freqs_dict:
            log.info(f'{qb.name}: {qb_ro_mwg} not found in lo_freqs_dict.')
            continue
        if set_mod_freq:
            qb.ro_mod_freq(qb.ro_freq() - lo_freqs_dict[qb_ro_mwg])


def configure_qubit_feedback_params(qubits, for_ef=None, set_thresholds=False):
    for qb in qubits:
        ge_ch = qb.ge_I_channel()
        acq_ch = qb.acq_I_channel()
        pulsar = qb.instr_pulsar.get_instr()
        AWG = qb.find_instrument(pulsar.get(f'{ge_ch}_awg'))
        if len(pulsar.get(f'{AWG.name}_trigger_channels')) > 0:
            AWG.dios_0_mode(2)
            vawg = (int(pulsar.get(f'{ge_ch}_id')[2:])-1)//2
            AWG.set(f'awgs_{vawg}_dio_mask_shift', 1+acq_ch)
            if (two_dio_bits := for_ef) is None:
                two_dio_bits = (len(qb.get_acq_int_channels()) == 2)
            # The case with two dio bits assumes channel I and Q are
            # consecutive both on the acquisition device and the AWG.
            AWG.set(f'awgs_{vawg}_dio_mask_value',
                    0b11 if two_dio_bits else 0b1)
        acq_dev = qb.instr_acq.get_instr()
        acq_dev.dios_0_mode(2)
        if set_thresholds:
            upload_classif_thresholds(qb)


def upload_classif_thresholds(qb, clf_params=None, add=None):
    """Sets classification thresholds for active reset

    Converts the thresholds from the classifier params of the qubit into
    actual thresholds as seen by the acquisition device (the former have the
    dimension of voltages, the latter integrated voltages over time),
    and uploads them to the acquisition device.

    Args:
        qb (QuDev_transmon): qubit object
        clf_params (dict): dictionary containing the thresholds that must
            be set on the corresponding UHF channel(s).
            If None, then defaults to qb.acq_classifier_params().
        add (dict): Optional offsets to add to the uploaded thresholds,
            in voltage units.

    Note: this could also be a qubit method.
    """
    if clf_params is None:
        clf_params = qb.acq_classifier_params()
    if add is None:
        add = {0: 0, 1: 0}
    instr = qb.instr_acq.get_instr()
    ths = {
        k: v + add[k]
        for k, v in clf_params['thresholds'].items()
    }
    # Rescale thresholds to match integrated values in the acq. intrument
    ths = {k: instr.acq_sampling_rate * qb.acq_length() * v
           for k, v in ths.items()}
    # Upload thresholds
    for key, th in ths.items():
        ch = {0: 'I', 1: 'Q'}[key]
        channel_id = qb.parameters[f'acq_{ch}_channel']()
        instr.parameters[f'qas_0_thresholds_{channel_id}_level'](ths[key])


def find_symmetry_index(data):
    data = data.copy()
    data -= data.mean()
    corr = []
    for iflip in np.arange(0, len(data)-0.5, 0.5):
        span = min(iflip, len(data)-1-iflip)
        data_filtered = data[int(iflip-span):int(iflip+span+1)]
        corr.append((data_filtered*data_filtered[::-1]).sum())
    return np.argmax(corr), corr


def get_pycqed_appdata_dir():
    """
    Returns the path to the pycqed application data dir.
    """
    if os.name == 'nt':
        path = os.path.expandvars(r'%LOCALAPPDATA%\pycqed')
    else:
        path = os.path.expanduser('~/.pycqed')
    os.makedirs(path, exist_ok=True)
    return path


def default_awg_dir():
    """
    Returns the path of an awg subfolder in the pycqed application data dir.
    """
    path = os.path.join(get_pycqed_appdata_dir(), 'awg')
    os.makedirs(path, exist_ok=True)
    return path


def raise_warning_image(destination_path, warning_image_path=None):
    """
    Copy the image specified by warning_image_path to the folder specified by
    destination_path.
    :param destination_path: folder where the warning image is to be copied
    :param image_path: full path, including image name and extension, to
        the warning image to be copied. If None, assumes WARNING.png exists in
        the module folder.
    :return:
    """
    if 'WARNING.png' not in os.listdir(destination_path):
        import shutil
        if warning_image_path is None:
            warning_image_path = os.path.abspath(sys.modules[__name__].__file__)
            warning_image_path = os.path.split(warning_image_path)[0]
            warning_image_path = os.path.abspath(os.path.join(
                warning_image_path, 'WARNING.png'))

        destination = os.path.abspath(os.path.join(destination_path,
                                                   'WARNING.png'))
        shutil.copy2(warning_image_path, destination)


def write_warning_message_to_text_file(destination_path, message, filename=None):
    """
    Write a warning message to a text file.
    :param destination_path: folder to the text file. If file does not yet
        exist, it will be created.
    :param message: string with the message to be written into the text file.
    :param filename: string with name of the warning message file. If None,
        uses 'warning_message'. If text file does not exist, it will be created.
        If text file already exists, the message will be appended.
    :return:
    """
    if filename is None:
        filename = 'warning_message'

    # Add extension if not contained in filename.
    if not len(os.path.splitext(filename)[-1]):
        filename += '.txt'

    # Adding this will ensure that if a future message is appended to the file,
    # the new message will be separated by an empty line from the current
    # message.
    message += '\n\n'

    # Prepend timestamp to the message
    message = f'{datetime.datetime.now():%Y%m%d_%H%M%S}\n{message}'

    # Write message to file
    file = open(os.path.join(destination_path, filename), 'a+')
    file.writelines(message)
    file.close()


def write_logfile(filename, content, directory):
    """

    Creates a file filename.log in directory.

    Args:
        filename (str): name of .log file
        content (str): content of the .log file
        directory (str): path where the .log file will be created

    Returns:
        full path to the .log file
    """
    logfile = os.path.join(directory, f"{filename}.log")
    f = open(logfile, 'a+')
    f.write(content)
    f.write("\n")
    f.close()
    return logfile


def zipfolder(zip_filename, folder, directory):
    """
    Creates a compressed file from folder.

    Args:
        zip_filename (str): name of the compressed file
        folder (str): path to folder that will be compressed
        directory (str): path where the compressed file will be created
    """
    with ZipFile(os.path.join(directory, f'{zip_filename}.zip'), 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                zipObj.write(filePath, os.path.relpath(filePath, folder))


def save_zibugreport(interactive=True, save_folder=None):
    """
    Saves a detailed bug report of ZI devices.

    For each ZI device in the running kernel, it saves the following:
        - firmware git revision
        - bitstream git revision
        - awg source string
        - compiler status string
        - errors on the device
        - device snapshot
    In addition, the following are saved:
        - current firmware and fpga versions running on each ZI device
        - current versions of installed zhinst modules. These are typically
            - zhinst-core
            - zhinst-deviceutils
            - zhinst-qcodes
            - zhinst-toolkit
            - zhinst-utils
        - last sequence in pulsar
        - csv files with the last programmed waves

    Args:
        interactive (bool): if True, opens a notepad file and prompts the user
            to paste the error message reported by pycqed.
        save_folder (str or None): location where the bug report will be saved.
            If None, the latest measurement timestamp is taken.

    Returns:
        exceptions (dict): exceptions raised while trying to dump the bug report
    """
    exceptions = {}

    # get the pulsar instance
    from pycqed.measurement.waveform_control import pulsar as ps
    pulsar = ps.Pulsar.get_instance()

    # create the save folder
    if save_folder is None:
        save_folder = a_tools.latest_data()
    brdir = os.path.join(
        save_folder,
        f"bugreport_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}")
    os.mkdir(brdir)

    # save the waveform files
    try:
        zipfolder('waves', pulsar.awg_interfaces[
            list(pulsar.awg_interfaces)[0]]._zi_wave_dir(), brdir)
    except Exception as e:
        exceptions['waves'] = e

    # get all connected ZI instruments
    instruments = get_all_connected_zi_instruments()

    # save versions of zhinst modules
    versions_zhinst, exceps = get_zhinst_modules_versions()
    exceptions.update(exceps)
    write_logfile('versions_zhinst', repr(versions_zhinst), brdir)

    # save firmware and fpgs versions
    versions_devs, exceps = get_zhinst_firmware_versions(instruments)
    exceptions.update(exceps)
    write_logfile('versions', repr(versions_devs), brdir)

    # save the firmware git revision
    for dev in instruments:
        try:
            fw_git_revision_node = \
                f"/{dev.devname}/raw/system/revisions/firmware"
            fw_git_revision_string = \
                dev.daq.get(fw_git_revision_node, flat=True)[
                    fw_git_revision_node][0]['vector']
            fw_git_revision_dict = json.loads(fw_git_revision_string)
            write_logfile(os.path.join(
                brdir, f'{dev.name}_{dev.devname}_firmware_revision'),
                repr(fw_git_revision_dict), brdir)
        except Exception as e:
            exceptions[f'{dev.name}_{dev.devname}_firmware_revision'] = e

    # save the bitstream git revision
    for dev in instruments:
        try:
            bs_git_revision_node = \
                f"/{dev.devname}/raw/system/revisions/bitstream"
            bs_git_revision_string = \
                dev.daq.get(bs_git_revision_node, flat=True)[
                    bs_git_revision_node][0]['vector']
            bs_git_revision_dict = json.loads(bs_git_revision_string)
            write_logfile(os.path.join(
                brdir, f'{dev.name}_{dev.devname}_bitstream_revision'),
                repr(bs_git_revision_dict), brdir)
        except Exception as e:
            exceptions[f'{dev.name}_{dev.devname}_bitstream_revision'] = e

    # save awg_source_strings
    for dev in instruments:
        try:
            write_logfile(os.path.join(
                brdir, f'{dev.name}_{dev.devname}_awg_source_strings'),
                repr(getattr(dev, '_awg_source_strings', {})), brdir)
        except Exception as e:
            exceptions[f'{dev.name}_{dev.devname}_awg_source_strings'] = e

    # save compiler status strings
    for dev in instruments:
        try:
            write_logfile(os.path.join(
                brdir, f'{dev.name}_{dev.devname}_compiler_statusstring'),
                getattr(dev, 'compiler_statusstring', ''), brdir)
        except Exception as e:
            exceptions[f'{dev.name}_{dev.devname}_compiler_statusstring'] = e

    # save snapshots
    for dev in instruments:
        try:
            write_logfile(
                os.path.join(brdir, f'{dev.name}_{dev.devname}_snapshot'),
                repr(dev.snapshot()), brdir)
        except Exception as e:
            exceptions[f'{dev.name}_{dev.devname}_snapshot'] = e

    # save errors reported by the devices
    for dev in instruments:
        try:
            write_logfile(
                os.path.join(brdir, f'{dev.name}_{dev.devname}_errors'),
                repr(json.loads(dev.getv('raw/error/json/errors'))), brdir)
        except Exception as e:
            try:
                # for QCodes-based devices
                err_dict = dev.daq.get(f'{dev.devname}/raw/error/json/errors',
                                       settingsonly=False)
                err_str = err_dict[dev.devname][
                    'raw']['error']['json']['errors'][0]['vector']
                write_logfile(
                    os.path.join(brdir, f'{dev.name}_{dev.devname}_errors'),
                    repr(json.loads(err_str)), brdir)
            except Exception as e:
                exceptions[f'{dev.name}_{dev.devname}_errors'] = e

    # save last sequence from pulsar
    try:
        seq = deepcopy(pulsar.last_sequence)
        if seq is not None:
            seq.pulsar = None
            for seg in seq.segments.values():
                seg.pulsar = None
        with open(os.path.join(brdir, "last_sequence.p"), "wb") as pf:
            pickle.dump(seq, pf)
    except Exception as e:
        exceptions['last_sequence'] = e

    # prompt user to save std output into a textfile
    if interactive:
        try:
            f = write_logfile('stdout', 'COPY AND PASTE STDOUT HERE', brdir)
            print(
                'Please look for the open notepad window and paste the output '
                'of the last running cell.')
            os.system(f'notepad {f}')
        except Exception as e:
            exceptions['stdout'] = e

    # print in kernel
    print(f'Bug report files saved to {brdir}')
    from pprint import pprint
    pprint(versions_zhinst)
    pprint(versions_devs)

    if len(exceptions):
        print('\n' +
              f'Not all elements of the bugreport could be saved. '
              f'Exceptions occurred during: {list(exceptions.keys())}')
    return exceptions


def get_all_connected_zi_instruments():
    """
    Gets all the Zurich Instruments devices that have been instantiated in
    the running kernel.

    Returns:
        list with instances of ZI device drivers
    """
    from qcodes.station import Station
    if Station.default is not None:
        all_inst = Station.default.components
    else:
        from pycqed.instrument_drivers.instrument import Instrument
        all_inst = Instrument._all_instruments
    return [inst for inst in all_inst.values()
            if inst.get_idn().get('vendor', '') in
            ['ZurichInstruments', 'Zurich Instruments']]

def get_zhinst_modules_versions():
    """
    Gets the current versions of the installed zhinst packages.

    Returns:
        versions (dict): module names as keys, version numbers as values
        exceptions (dict): module names as keys, errors as values if an error
            has occurred when trying to get the version number of a module
    """
    versions, exceptions = {}, {}
    import zhinst
    submodules = [sm for sm in dir(zhinst) if not sm.startswith('__')]
    for sm in submodules:
        try:
            versions[f'zhinst-{sm}'] = zhinst.__dict__[sm].__version__
        except Exception as e:
            exceptions[sm] = e
    return versions, exceptions


def get_zhinst_firmware_versions(zi_instruments=None):
    """
    Gets the firmware and fpga versions of ZI instruments.

    Args:
        zi_instruments (list or None): instances of ZI instrument drivers

    Returns:
        versions (dict): ZI instrument drivers as keys, dict with
            version numbers as values
        exceptions (dict): ZI instrument drivers as keys, errors as values if
            an error has occurred when trying to get a version number for a
            ZI device
    """
    if zi_instruments is None:
        zi_instruments = get_all_connected_zi_instruments()

    versions, exceptions = {}, {}
    for node in ['system/fwrevision', 'system/fpgarevision']:
        versions[node] = {}
        for dev in zi_instruments:
            try:
                versions[node][f'{dev.name} - {dev.devname}'] = dev.geti(node)
            except Exception:
                try:
                    # for QCodes-based devices
                    versions[node][f'{dev.name} - {dev.devname}'] = \
                        dev.daq.getInt(f'{dev.devname}/system/fwrevision')
                except Exception as e:
                    exceptions[f'{node} for {dev.devname}'] = e
    return versions, exceptions


class TempLogLevel:
    """
    With Handler to temporarily change the log level of a logger
    """
    LOG_LEVELS = dict(debug=logging.DEBUG, info=logging.INFO,
                      warning=logging.WARNING, error=logging.ERROR,
                      critical=logging.CRITICAL, fatal=logging.FATAL)

    def __init__(self, logger, log_level="info"):
        """
        Instantiate a TemporaryLogLevel.
        Args:
            logger (logging.Logger): logger of which the level should
                be changed temporarily.
            log_level (str): Desired temporary log level: "debug", "info",
                "warning", "error", "critical", "fatal". Strings can also be
                all caps, e.g. "INFO". Defaults to "info".
        """
        self.logger = logger
        self.log_level = logger.level
        self.temp_log_level = self.LOG_LEVELS.get(log_level.lower(), log_level)

    def __enter__(self):
        self.logger.setLevel(self.temp_log_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.log_level)


def assert_not_none(*param_names):
    """
    Decorator that ensures that all (keyword) arguments of the decorated function
     `f` provided in param_names are not None.
    Args:
        *param_names (str): One or several strings indicating the name of the
            (keyword) arguments which should not have a value of None
    Returns:
         decorated function.
    Examples:
        >>>  class Test:
        >>>    @assert_not_none('arg2', 'kwarg1', "other_kwarg")
        >>>    def test(self, arg1, arg2, kwarg1=0, kwarg2=None, **kwargs):
        >>>        pass
        >>> t = Test()
        >>> t.test('a', "b") # does not raise an error
        >>> t.test('a', None) # raises error because arg2 is None
        >>> t.test('a', 'b', None,) # raises error because kwarg1 is passed
        >>>                         # as positional argument with a value of None
        >>> t.test('a', 'b', "c", something=None) # does not raise an error
        >>> t.test('a', 'b', "c", other_kwarg=None) # raises an error because
        >>>                                         # other_kwarg is None
    Raises:
        ValueError if a (keyword) argument mentioned in param_names is None.
    """
    import inspect

    def check(f):
        @functools.wraps(f)
        def wrapped_func(*args, **kwds):
            signature_args_and_kwargs = inspect.getfullargspec(f).args
            default_kwarg_values = inspect.getfullargspec(f).defaults
            error_msg = ' {name} is None, but {name} should not be None when ' \
                        'passed to ' + f.__qualname__

            # check if a positional argument is None or a signature keyword
            # argument is None:
            # take argument values and default values of keyword arguments
            # which are not passed as positional arguments (since keyword
            # arguments can also be provided as positional arguments)
            x = len(signature_args_and_kwargs) - len(
                args)  # index of first needed default keyword arg value
            for (name, value) in zip(signature_args_and_kwargs,
                                     args + default_kwarg_values[x:]):
                # print(name, value)
                if name in param_names and value is None:
                    raise ValueError(error_msg.format(f=f, name=name))

            # check if a passed keyword argument is None
            for (name, value) in kwds.items():
                if name in param_names and value is None:
                    raise ValueError(error_msg.format(f=f, name=name))

            return f(*args, **kwds)

        wrapped_func.__name__ = f.__name__
        return wrapped_func

    return check
