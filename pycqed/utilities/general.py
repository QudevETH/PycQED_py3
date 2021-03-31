import os
import sys
import ast
import numpy as np
import h5py
import json
import datetime
from contextlib import contextmanager
from pycqed.measurement import hdf5_data as h5d
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
from collections import OrderedDict  # for eval in load_settings
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
            filepath = a_tools.measurement_filename(folder)
            f = h5py.File(filepath, 'r')
            sets_group = f['Instrument settings']
            ins_group = sets_group[instrument_name]

            if verbose:
                print('Loaded settings successfully from the HDF file.')

            params_to_set = kw.pop('params_to_set', [])
            if len(params_to_set)>0:
                if verbose and update:
                    print('Setting parameters {} for {}.'.format(
                        params_to_set, instrument_name))
                params_to_set = [(param, val) for (param, val) in
                                ins_group.attrs.items() if param in
                                 params_to_set]
            else:
                if verbose and update:
                    print('Setting parameters for {}.'.format(instrument_name))
                params_to_set = ins_group.attrs.items()

            if not update:
                params_dict = {parameter : value for parameter, value in \
                        params_to_set}
                f.close()
                return params_dict

            for parameter, value in params_to_set:
                if parameter in instrument.parameters.keys() and \
                        hasattr(instrument.parameters[parameter], 'set'):
                    if value == 'None':  # None is saved as string in hdf5
                        try:
                            instrument.set(parameter, None)
                        except Exception:
                            print('Could not set parameter "%s" to "%s" for '
                                  'instrument "%s"' % (
                                      parameter, value, instrument_name))
                    elif value == 'False':
                        try:
                            instrument.set(parameter, False)
                        except Exception:
                            print('Could not set parameter "%s" to "%s" for '
                                  'instrument "%s"' % (
                                      parameter, value, instrument_name))
                    elif value == 'True':
                        try:
                            instrument.set(parameter, True)
                        except Exception:
                            print('Could not set parameter "%s" to "%s" for '
                                  'instrument "%s"' % (
                                      parameter, value, instrument_name))
                    else:
                        try:
                            instrument.set(parameter, int(value))
                        except Exception:
                            try:
                                instrument.set(parameter, float(value))
                            except Exception:
                                try:
                                    instrument.set(parameter, eval(value))
                                except Exception:
                                    try:
                                        instrument.set(parameter,
                                                       value)
                                    except Exception:
                                        log.error('Could not set parameter '
                                              '"%s" to "%s" '
                                              'for instrument "%s"' % (
                                                  parameter, value,
                                                  instrument_name))

            success = True
            f.close()
        except Exception as e:
            logging.warning(e)
            success = False
            try:
                f.close()
            except:
                pass
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



def temporary_value(*param_value_pairs):
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
        with temporary_values((qb1.ro_freq, 6e9)):
            qb1.measure_spectroscopy(...)
    """

    class TemporaryValueContext:
        def __init__(self, *param_value_pairs):
            if len(param_value_pairs) > 0 and \
                    not isinstance(param_value_pairs[0], (tuple, list)):
                param_value_pairs = (param_value_pairs,)
            self.param_value_pairs = param_value_pairs
            self.old_value_pairs = []

        def __enter__(self):
            log.debug('Entered TemporaryValueContext')
            self.old_value_pairs = \
                [(param, param()) for param, value in self.param_value_pairs]
            for param, value in self.param_value_pairs:
                param(value)
    
        def __exit__(self, type, value, traceback):
            for param, value in self.old_value_pairs: 
                param(value)
            log.debug('Exited TemporaryValueContext')
    
    return TemporaryValueContext(*param_value_pairs)


def configure_qubit_mux_drive(qubits, lo_freqs_dict):
    mwgs_set = set()
    for qb in qubits:
        qb_ge_mwg = qb.instr_ge_lo()
        if qb_ge_mwg not in lo_freqs_dict:
            raise ValueError(
                f'{qb_ge_mwg} for {qb.name} not found in lo_freqs_dict.')
        else:
            qb.ge_mod_freq(qb.ge_freq()-lo_freqs_dict[qb_ge_mwg])
            if qb_ge_mwg not in mwgs_set:
                qb.instr_ge_lo.get_instr().frequency(lo_freqs_dict[qb_ge_mwg])
                mwgs_set.add(qb_ge_mwg)


def configure_qubit_mux_readout(qubits, lo_freqs_dict):
    mwgs_set = set()
    idx = {}
    for lo in lo_freqs_dict:
        idx[lo] = 0

    for i, qb in enumerate(qubits):
        qb_ro_mwg = qb.instr_ro_lo()
        if qb_ro_mwg not in lo_freqs_dict:
            raise ValueError(
                f'{qb_ro_mwg} for {qb.name} not found in lo_freqs_dict.')
        else:
            qb.ro_mod_freq(qb.ro_freq() - lo_freqs_dict[qb_ro_mwg])
            qb.acq_I_channel(2 * idx[qb_ro_mwg])
            qb.acq_Q_channel(2 * idx[qb_ro_mwg] + 1)
            idx[qb_ro_mwg] += 1
            if qb_ro_mwg not in mwgs_set:
                qb.instr_ro_lo.get_instr().frequency(lo_freqs_dict[qb_ro_mwg])
                mwgs_set.add(qb_ro_mwg)


def configure_qubit_feedback_params(qubits, for_ef=False):
    if for_ef:
        raise NotImplementedError('for_ef feedback_params')
    for qb in qubits:
        ge_ch = qb.ge_I_channel()
        pulsar = qb.instr_pulsar.get_instr()
        AWG = qb.find_instrument(pulsar.get(f'{ge_ch}_awg'))
        vawg = (int(pulsar.get(f'{ge_ch}_id')[2:])-1)//2
        acq_ch = qb.acq_I_channel()
        AWG.set(f'awgs_{vawg}_dio_mask_shift', 1+acq_ch)
        AWG.set(f'awgs_{vawg}_dio_mask_value', 1)
        UHF = qb.instr_uhf.get_instr()
        threshs = qb.acq_classifier_params()
        if threshs is not None:
            threshs = threshs.get('thresholds', None)
        if threshs is not None:
            UHF.set(f'qas_0_thresholds_{acq_ch}_level', threshs[0])


def find_symmetry_index(data):
    data = data.copy()
    data -= data.mean()
    corr = []
    for iflip in np.arange(0, len(data)-0.5, 0.5):
        span = min(iflip, len(data)-1-iflip)
        data_filtered = data[np.int(iflip-span):np.int(iflip+span+1)]
        corr.append((data_filtered*data_filtered[::-1]).sum())
    return np.argmax(corr), corr
