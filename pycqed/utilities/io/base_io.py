#  base input-output class for reading and writing files
# should not be initiated alone and only serves as a skeleton for child classes
from __future__ import annotations

import os
import time
from pathlib import Path
import logging

from pycqed.instrument_drivers import mock_qcodes_interface as mqcodes

logger = logging.getLogger(__name__)

# file extensions used to dump and load files. Extensions are ordered beginning
# with the filetype which should be favoured when opening a file with the same
# filenames.
file_extensions = {
    'msgpack': '.msg', 'msgpack_comp': '.msgc', 'pickle': '.pickle',
    'pickle_comp': '.picklec', 'hdf5': '.hdf5'}


class Dumper:
    """
    Generic class to serialize dictionaries into a file.
    """

    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        """
        Creates a folder for the file.
        Args:
            name (str): additional label for the file
            data (dict): Dictionary which will be serialized
            datadir (str): root directory
            compression (bool): True if the file should be compressed with
                blosc2
            timestamp (str): timestamp which is used to create the folder and
                filename. It has to be in the format '%Y%m%d_%H%M%S'
        """
        self._name = name
        self.data = data
        self.compression = compression

        if timestamp == None:
            self._localtime = time.localtime()
        else:
            self._localtime = time.strptime(timestamp, '%Y%m%d_%H%M%S')
        self._timestamp = time.asctime(self._localtime)
        self._timemark = time.strftime('%H%M%S', self._localtime)
        self._datemark = time.strftime('%Y%m%d', self._localtime)

        # sets the file path
        self.filepath = DateTimeGenerator().new_filename(
            self, folder=datadir, auto_increase=False)

        self.filepath = self.filepath.replace("%timemark", self._timemark)

        self.folder, self._filename = os.path.split(self.filepath)
        # creates the folder if needed
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    @staticmethod
    def compress_file(file):
        import blosc2
        return blosc2.compress(file)


class Loader:
    """
    Generic class to load instruments and parameters from a file.
    """

    def __init__(self, timestamp: str = None, filepath: str = None, **kwargs):
        """
        Initialization of generic loader. Not intended to be used as standalone
        instance, but only via its children classes.
        Args:
            timestamp (str): timestamp of the file in the format. Supports all
                formats from a_tools.get_folder
            filepath (str): filepath of the file, overwrites timestamp and
                folder (in kwargs)
            **kwargs: folder (str): Optional, folder of the file if distinct from
                a_tools.datadir
                file_id (str): suffix of the file
                extension (str): custom extension of the file
                compression (bool): True if file is compressed with blosc2,
                    usually extracted from file extension
        """
        self.filepath = filepath
        self.timestamp = timestamp
        self.extension = kwargs.get('extension', None)

    def get_filepath(self, **kwargs):
        """
        If no explicit filepath is given, a_tools.get_folder tries to get the
        directory based on the timestamp and the directory defined by
        a_tools.datadir, a_tools.fetch_data_dir, respectively.

        Returns: filepath as a string

        """
        if self.filepath is not None:
            return self.filepath
        from pycqed.analysis import analysis_toolbox as a_tools
        if self.timestamp is not None:
            self.folder = a_tools.get_folder(self.timestamp,
                                         **kwargs)
        else:
            self.folder = kwargs.get('folder', None)
        self.filepath = a_tools.measurement_filename(self.folder,
                                                     ext=self.extension[1:],
                                                     **kwargs)
        return self.filepath

    @staticmethod
    def get_file_format(timestamp=None, folder=None, filepath=None,
                        file_id=None):
        """
        Returns the file format of a given timestamp.
        If several files with the same filename but different file extensions
        are found, the file with the extension first mentioned in
        file_extensions will be considered.
        Args:
            timestamp (str): timestamp of the file
            folder (str): folder of the file if different from a_tools.datadir
            filepath (str): Optional filepath, overwrites timestamp
            file_id (str): suffix of the file name

        Returns (str): file format as a string (see file_extensions)

        """
        if filepath is None:
            from pycqed.analysis import analysis_toolbox as a_tools
            if timestamp is None:
                # if only folder is given, folder needs to point directly to
                # location of settings files
                folder_dir = folder
            else:
                folder_dir = a_tools.get_folder(timestamp, folder=folder)
            dirname = os.path.split(folder_dir)[1]
            path = Path(folder_dir)

            if file_id is not None:
                dirname = dirname + file_id
            filepath = sorted(path.glob(dirname + ".*"))

            if len(filepath) > 1:
                for format, extension in file_extensions.items():
                    for path in filepath:
                        file_name, file_extension = os.path.splitext(path)
                        if extension == file_extension:
                            logger.warning(
                                f"More than one file found for timestamp "
                                f"'{timestamp}'. File in format '{format}' will"
                                f" be considered.")
                            return format
                raise KeyError(f"More than one file found for "
                               f"timestamp '{timestamp}' and none matches the "
                               f"standard file extensions '{file_extensions}'.")
            elif len(filepath) == 0:
                raise KeyError(f"No file found for timestamp '{timestamp}'")
            else:
                filepath = filepath[0]
        file_name, file_extension = os.path.splitext(filepath)

        for format, extension in file_extensions.items():
            if file_extension == extension:
                return format

        raise KeyError(f"File extension '{file_extension}' not in "
                       f"standard form '{file_extensions}'")

    def load_instrument(self, inst_name, inst_dict):
        """
        Loads instrument object from settings given as a dictionary.
        Args:
            inst_name (str): Instrument name (key of the snapshot/dictionary)
            inst_dict (dict): Instrument values given as a dictionary

        Returns: Instrument object

        """
        inst = mqcodes.Instrument(inst_name)
        # load parameters
        if 'parameters' in inst_dict:
            for param_name, param_value in inst_dict['parameters'].items():
                inst.add_parameter(self.load_parameter(param_name, param_value))
        # load class name
        if '__class__' in inst_dict:
            inst.add_classname(inst_dict['__class__'])
        # load submodules, treats submodules and channels as the same.
        for k in ['submodules', 'channels']:
            if k not in inst_dict:
                continue
            for submod_name, submod_dict in inst_dict[k].items():
                submod_inst = self.load_instrument(
                    submod_name, submod_dict)
                inst.add_submodule(submod_name, submod_inst)
        return inst

    def load_parameter(self, param_name, param_value):
        return mqcodes.Parameter(param_name, param_value)

    @staticmethod
    def load_component(comp_name, comp):
        """
        Hollow function. Can be used in the future to load components which are
        not instruments.
        Args:
            comp_name (str): Component name (key of the snapshot/dictionary)
            comp (dict): Component values given as a dictionary

        Returns:
        """
        pass

    @staticmethod
    def decompress_file(file):
        import blosc2
        return blosc2.decompress(file)

    def get_station(self, param_path=None):
        """
        Returns a station build from a snapshot. This snapshot is loaded from
        a file using self.get_snapshot() and the timestamp given by the
        initialization of the Loader object.
        Returns (Station): station build from the snapshot.
        """
        if param_path is not None:
            logger.warning("For files which are not HDF5, the entire file will"
                           " be loaded in to the station regardless of "
                           "param_path.")
        station = Station(timestamp=self.timestamp)
        snap = self.get_snapshot()
        for inst_name, inst_dict in snap['instruments'].items():
            inst = self.load_instrument(inst_name, inst_dict)
            station.add_component(inst)

        for comp_name, comp_dict in snap['components'].items():
            # So far, components are not considered.
            # Loader.load_component is a hollow function.
            logger.warning('Components are not considered. '
                           'Loader.load_component is a hollow function.')
            comp = self.load_component(comp_name, comp_dict)
            if comp is not None:
                station.add_component(comp)

        return station

    def get_snapshot(self):
        return dict()


class DateTimeGenerator:
    """
    Class to generate filenames / directories based on the date and time.
    """

    def __init__(self):
        pass

    def create_data_dir(self, datadir: str, name: str=None, ts=None,
                        datesubdir: bool=True, timesubdir: bool=True,
                        auto_increase: bool = True):
        """
        Create and return a new data directory.

        Input:
            datadir (string): base directory
            name (string): optional name of measurement
            ts (time.localtime()): timestamp which will be used
                if timesubdir=True
            datesubdir (bool): whether to create a subdirectory for the date
            timesubdir (bool): whether to create a subdirectory for the time
            auto_increase (bool): ensures that timestamp is unique and if not
                increases by 1s until it is.

        Output:
            The directory to place the new file in
        """

        path = datadir
        if ts is None:
            ts = time.localtime()
        if datesubdir:
            path = os.path.join(path, time.strftime('%Y%m%d', ts))
        if timesubdir:
            tsd = time.strftime('%H%M%S', ts)
            timestamp_verified = False
            counter = 0
            # Verify if timestamp is unique by seeing if the folder exists
            while not timestamp_verified and auto_increase:
                counter += 1
                try:
                    measdirs = [d for d in os.listdir(path)
                                if d[:6] == tsd]
                    if len(measdirs) == 0:
                        timestamp_verified = True
                    else:
                        # if timestamp not unique, add one second
                        # This is quite a hack
                        ts = time.localtime((time.mktime(ts)+1))
                        tsd = time.strftime('%H%M%S', ts)
                    if counter >= 3600:
                        raise Exception()
                except OSError as err:
                    if 'cannot find the path specified' in str(err):
                        timestamp_verified = True
                    elif 'No such file or directory' in str(err):
                        timestamp_verified = True
                    else:
                        raise err
            if name is not None:
                path = os.path.join(path, tsd+'_'+name)
            else:
                path = os.path.join(path, tsd)

        return path, tsd

    def new_filename(self, data_obj, folder, auto_increase: bool = True):
        """Return a new filename, based on name and timestamp."""
        path, tstr = self.create_data_dir(folder,
                                          name=data_obj._name,
                                          ts=data_obj._localtime,
                                          auto_increase=auto_increase)
        filename = '%s_%s.hdf5' % (tstr, data_obj._name)
        return os.path.join(path, filename)