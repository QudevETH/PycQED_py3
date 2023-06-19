"""
Module for handling HDF5 data within qcodes.
Based on hdf5 datawrapper from qtlab originally by Reinier Heeres and
Wolfgang Pfaff.

Contains:
- a data class (HDF5Data) which is essentially a wrapper of a h5py data
  object, adapted for usage with qcodes
- name generators in the style of qtlab Data objects
- functions to create standard data sets
"""

import os
import time
import h5py
import numpy as np
import logging

# Do not remove, used inside eval()
from numpy import array
from collections import OrderedDict

from pycqed.utilities.io.base_io import Loader, file_extensions, \
    DateTimeGenerator
from pycqed.instrument_drivers.mock_qcodes_interface import Parameter, \
    Instrument, Station

log = logging.getLogger(__name__)

try:
    import qutip
    qutip_imported = True
except Exception:
    log.warning('qutip was not imported. qutip objects will be stored as '
                'strings.')
    qutip_imported = False


class Data(h5py.File):

    def __init__(self, name: str, datadir: str,
                 timestamp: str = None, auto_increase=True):
        """
        Creates an empty data set including the file, for which the currently
        set file name generator is used.

        kwargs:
            name (string) : base name of the file
            datadir (string) : A folder will be created within the datadir
                using the standard timestamp structure
            auto_increase (bool) : if true, filepath name is increased by 1s
                if file already exists
        """
        self._name = name

        if timestamp is None:
            self._localtime = time.localtime()
        else:
            self._localtime = time.strptime(timestamp, '%Y%m%d_%H%M%S')

        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        self.filepath = DateTimeGenerator().new_filename(
            self, folder=datadir, auto_increase=auto_increase)

        self.filepath = self.filepath.replace("%timemark", self._timemark)

        self.folder, self._filename = os.path.split(self.filepath)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        super(Data, self).__init__(self.filepath, 'a')
        self.flush()


def encode_to_utf8(s):
    """
    Required because h5py does not support python3 strings
    """
    # converts byte type to string because of h5py datasaving
    if isinstance(s, str):
        s = s.encode('utf-8')
    # If it is an array of value decodes individual entries
    elif isinstance(s, (np.ndarray, list, tuple)):
        s = [s.encode('utf-8') for s in s]
    return s


def write_dict_to_hdf5(data_dict: dict, entry_point, overwrite=False):
    """
    Args:
        data_dict (dict): dictionary to write to hdf5 file
        entry_point (hdf5 group.file) : location in the nested hdf5 structure
            where to write to.
    """
    for key, item in data_dict.items():
        if isinstance(key, tuple):
            key = str(key)

        # Basic types
        if isinstance(item, (str, float, int, bool, np.number,
                             np.float_, np.int_, np.bool_)):
            if not hasattr(key, "encode"):
                key = repr(key)
            try:
                entry_point.attrs[key] = item
            except Exception as e:
                log.error(e)
                log.error('Exception occurred while writing'
                      ' {}:{} of type {}'.format(key, item, type(item)))
        elif isinstance(item, np.ndarray) or (
                qutip_imported and isinstance(item, qutip.qobj.Qobj)):
            if qutip_imported and isinstance(item, qutip.qobj.Qobj):
                item = item.full()
            try:
                entry_point.create_dataset(key, data=item)
            except RuntimeError:
                if overwrite:
                    del entry_point[key]
                    entry_point.create_dataset(key, data=item)
                else:
                    raise
        elif item is None:
            # as h5py does not support saving None as attribute
            # I create special string, note that this can create
            # unexpected behaviour if someone saves a string with this name
            entry_point.attrs[key] = 'NoneType:__None__'

        elif isinstance(item, dict):
            try:
                entry_point.create_group(key)
            except RuntimeError:
                if overwrite:
                    del entry_point[key]
                    entry_point.create_group(key)
                else:
                    raise
            except AttributeError:
                # can happen if key is a tuple
                # try to transform key to string
                key = str(key)
                entry_point.create_group(key)

            write_dict_to_hdf5(data_dict=item,
                               entry_point=entry_point[key],
                               overwrite=overwrite)
        elif isinstance(item, (list, tuple)):
            if len(item) > 0:
                elt_type = type(item[0])
                # Lists of a single type, are stored as an hdf5 dset
                if (all(isinstance(x, elt_type) for x in item) and
                        not isinstance(item[0], dict) and
                        not isinstance(item, tuple) and
                        not isinstance(item[0], list)):
                    if isinstance(item[0], (int, float,
                                            np.int32, np.int64)):
                        try:
                            entry_point.create_dataset(key, 
                                                       data=np.array(item))
                        except RuntimeError:
                            if overwrite:
                                del entry_point[key]
                                entry_point.create_dataset(key, 
                                                           data=np.array(item))
                            else:
                                raise
                        entry_point[key].attrs['list_type'] = 'array'

                    # strings are saved as a special dtype hdf5 dataset
                    elif isinstance(item[0], str):
                        dt = h5py.special_dtype(vlen=str)
                        data = np.array(item)
                        data = data.reshape((-1, 1))
                        try:
                            ds = entry_point.create_dataset(
                                key, (len(data), 1), dtype=dt)
                        except RuntimeError:
                            if overwrite:
                                del entry_point[key]
                                entry_point.create_dataset(
                                    key, (len(data), 1), dtype=dt)
                            else:
                                raise
                        ds.attrs['list_type'] = 'str'
                        ds[:] = data
                    else:
                        log.debug(
                            'List of type "{}" for "{}":"{}" not '
                            'supported, storing as string'.format(
                                elt_type, key, item))
                        entry_point.attrs[key] = str(item)
                # Storing of generic lists/tuples
                else:
                    try:
                        entry_point.create_group(key)
                    except RuntimeError:
                        if overwrite:
                            del entry_point[key]
                            entry_point.create_group(key)
                        else:
                            raise
                    # N.B. item is of type list
                    list_dct = {'list_idx_{}'.format(idx): entry for
                                idx, entry in enumerate(item)}
                    group_attrs = entry_point[key].attrs
                    if isinstance(item, tuple):
                        group_attrs['list_type'] = 'generic_tuple'
                    else:
                        group_attrs['list_type'] = 'generic_list'
                    group_attrs['list_length'] = len(item)
                    write_dict_to_hdf5(
                        data_dict=list_dct,
                        entry_point=entry_point[key],
                        overwrite=overwrite)
            else:
                # as h5py does not support saving None as attribute
                entry_point.attrs[key] = 'NoneType:__emptylist__'

        else:
            log.debug(
                'Type "{}" for "{}" (key): "{}" (item) at location {} '
                'not supported, '
                'storing as string'.format(type(item), key, item,
                                           entry_point))
            entry_point.attrs[key] = str(item)


def read_from_hdf5(path_to_key_or_attribute, h5_group):
    """
    Wrapper to extract either attributes or keys from open hdf5 files. Using
    read_attribute_from_hdf5 and read_dict_from_hdf5.
    Args:
        path_to_key_or_attribute (str): path to the attribute or key
            separated by '.'
        h5_group (hdf5 file/group): hdf5 file or group from which to read.

    Returns: key (as a dict) or attribute or 'not found' if it could not find
        the specified path

    """
    param_value = read_attribute_from_hdf5(path_to_key_or_attribute, h5_group)
    if param_value == 'not found':
        group_name = '/'.join(path_to_key_or_attribute.split('.'))
        try:
            param_value = read_dict_from_hdf5({}, h5_group[group_name])
        except Exception:
            return 'not found'
    return param_value


def read_attribute_from_hdf5(path_to_attribute, h5_group):
    """
    Low level extractor for parameters
    Args:
        path_to_attribute (str): path to the attribute separated by '.'
        h5_group (hdf5 file/group): hdf5 file or group from which to read.

    Returns: attribute value or 'not found' if it can not find path_to_attribute

    """
    param_value = 'not found'

    try:
        if len(path_to_attribute.split('.')) == 1:
            param_value = decode_attribute_value(h5_group.attrs[path_to_attribute])
        else:
            group_name = '/'.join(path_to_attribute.split('.')[:-1])
            par_name = path_to_attribute.split('.')[-1]
            group = h5_group[group_name]
            attrs = list(group.attrs)
            if par_name in attrs:
                param_value = decode_attribute_value(
                    group.attrs[par_name])
    except Exception as e:
        param_value = 'not found'

    return param_value


def read_dict_from_hdf5(data_dict: dict, h5_group):
    """
    Reads a dictionary from an hdf5 file or group that was written using the
    corresponding "write_dict_to_hdf5" function defined above.

    Args:
        data_dict (dict):
                dictionary to which to add entries being read out.
                This argument exists because it allows nested calls of this
                function to add the data to an existing data_dict.
        h5_group  (hdf5 group):
                hdf5 file or group from which to read.
    """
    # if 'list_type' not in h5_group.attrs:
    for key, item in h5_group.items():
        if isinstance(item, h5py.Group):
            data_dict[key] = {}
            data_dict[key] = read_dict_from_hdf5(data_dict[key],
                                                 item)
        else:  # item either a group or a dataset
            if 'list_type' not in item.attrs:
                data_dict[key] = item[()]
            elif item.attrs['list_type'] == 'str':
                # lists of strings needs some special care, see also
                # the writing part in the writing function above.
                list_of_str = [x[0] for x in item[()]]
                data_dict[key] = [x.decode('utf-8') if isinstance(x, bytes)
                                  else x for x in list_of_str]

            else:
                data_dict[key] = list(item[()])
    for key, item in h5_group.attrs.items():
        if isinstance(item, str):
            # Extracts "None" as an exception as h5py does not support
            # storing None, nested if statement to avoid elementwise
            # comparison warning
            if item == 'NoneType:__None__':
                item = None
            elif item == 'NoneType:__emptylist__':
                item = []
        data_dict[key] = item

    if 'list_type' in h5_group.attrs:
        if (h5_group.attrs['list_type'] == 'generic_list' or
                h5_group.attrs['list_type'] == 'generic_tuple'):
            list_dict = data_dict
            data_list = []
            for i in range(list_dict['list_length']):
                data_list.append(list_dict['list_idx_{}'.format(i)])

            if h5_group.attrs['list_type'] == 'generic_tuple':
                return tuple(data_list)
            else:
                return data_list
        else:
            raise NotImplementedError('cannot read "list_type":"{}"'.format(
                h5_group.attrs['list_type']))
    return data_dict


def decode_attribute_value(param_value):
    """
       Converts byte type to the true type of a parameter loaded from a file.

       Args:
           param_value: the raw value of the parameter as retrieved from the HDF
               file

       Returns:
           the converted parameter value
       """
    if isinstance(param_value, bytes):
        param_value = param_value.decode('utf-8')
    # If it is an array of value decodes individual entries
    if isinstance(param_value, np.ndarray) or isinstance(param_value, list):
        param_value = [av.decode('utf-8') if isinstance(av, bytes)
                       else av for av in param_value]
    try:
        return eval(param_value)
    except Exception:
        return param_value


class HDF5Loader(Loader):

    def __init__(self, timestamp=None, filepath=None, extension=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         extension=extension)

        if self.extension is None:
            self.extension = file_extensions['hdf5']
        self.filepath = self.get_filepath(**kwargs)
        self.h5mode = kwargs.get('h5mode', 'r')

    @staticmethod
    def load_instrument(inst_name, inst_group):
        """
        Loads instrument object from settings given as a hdf object.
        Args:
            inst_name (str): Name of the instrument
            inst_group (hdf group): Values of the instrument as a group.

        Returns: Instrument object.

        """
        inst = Instrument(inst_name)
        # load parameters
        for param_name in list(inst_group.attrs):
            param_value = decode_attribute_value(
                inst_group.attrs[param_name])
            par = Parameter(param_name, param_value)
            inst.add_parameter(par)
        # load class name
        if '__class__' in inst_group:
            inst.add_classname(
                decode_attribute_value(inst_group.attrs['__class__']))
        # load submodules
        for submod_name, submod_group in list(inst_group.items()):
            submod_inst = HDF5Loader.load_instrument(submod_name, submod_group)
            inst.add_submodule(submod_name, submod_inst)
        return inst

    @staticmethod
    def load_component(comp_name, comp):
        pass

    def get_station(self, param_path=None):
        """
        Loads settings from a config file into a station.
        param_path (list): list of parameters which are loaded into a
            station. If None, all parameters are loaded into the station.
            Parameters must be of the form %inst_name%.%param_name%.

        """
        with h5py.File(self.filepath, 'r') as file:
            config_file = file['Instrument settings']
            station = Station(timestamp=self.timestamp)
            if param_path is None:
                for inst_name, inst_group in list(config_file.items()):
                    inst = self.load_instrument(inst_name, inst_group)
                    station.add_component(inst)
            else:
                for path_to_param in param_path:
                    param_value = read_attribute_from_hdf5(path_to_param,
                                                           config_file)
                    if param_value == 'not found':
                        continue
                    # if param path is only an instrument (%inst_name%)
                    if len(path_to_param.split('.')) == 1:
                        inst_path = path_to_param
                        inst = Instrument(inst_path[-1])
                    else:
                        param_name = path_to_param.split('.')[-1]
                        param = Parameter(param_name, param_value)

                        inst_path = path_to_param.split('.')[:-1]

                        inst = Instrument(inst_path[-1])
                        inst.add_parameter(param)

                    # if more than one instrument name exists, the instrument
                    # must be a submodule
                    for inst_name in inst_path[-1:0:-1]:
                        submod = inst
                        inst = Instrument(inst_name)
                        inst.add_submodule(submod.name, submod)
                    if inst.name in station.components.keys():
                        station.components[inst.name].update(inst)
                    else:
                        station.add_component(inst)

        return station

    def get_snapshot(self):
        return self.get_station().snapshot()
