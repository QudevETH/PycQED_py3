# from pycqed.analysis_v3 import helper_functions as hlp_mod
import logging
import numpy as np
import h5py

from pycqed.instrument_drivers.mock_qcodes_interface import Station, \
    ParameterNotFoundError
from pycqed.utilities.io.base_io import Loader
import pycqed.gui.dict_viewer as dict_viewer
from collections import OrderedDict

from pycqed.utilities.io.hdf5 import HDF5Loader
from pycqed.utilities.io.msgpack import MsgLoader
from pycqed.utilities.io.pickle import PickleLoader

logger = logging.getLogger(__name__)


class IndexableDict(OrderedDict):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as keyerror:
            if isinstance(item, int):
                try:
                    return super().__getitem__(list(self.keys())[item])
                except IndexError as e:
                    logger.error(f"Integer seems to be out of range of list"
                                 f" '{list(self.keys())}'")
                    raise e
            else:
                raise keyerror


class SettingsManager:
    """
    Class which contains different station. Stations can be added in two ways:
    1. Loaded from saved settings in a given file. Supported file types:
        hdf5, pickle (.obj), msgpack (.msg)
    2. Added from a preexisting station. Either settings_manager.Station or
        QCodes.Station (or any other snapshotable type)
    Snapshot of stations can be displayed via dict_viewer module.
    """

    def __init__(self, station=None, timestamp: str = None):
        """
        Initialization of SettingsManager instance. Can be called with a
        preexisting station (settings_manager.Station or QCodes.Station)
        Args:
            station (mock_qcodes_interface.Station or qcodes.Station):
                optional, adds given station to the settings manager
        """
        self.stations = IndexableDict()
        if station is not None:
            self.add_station(station, timestamp)

    def add_station(self, station, timestamp: str):
        """
        Adds a given station with a timestamp as its unique name.
        Args:
            station (mock_qcodes_interface.Station): Station object which will
                be added.
                Either QCodes station or a mock station class
            timestamp (str): Timestamp to call the station.
                Is converted into unified format.

        """
        if timestamp is not None:
            from pycqed.analysis import analysis_toolbox as a_tools
            timestamp = '_'.join(a_tools.verify_timestamp(timestamp))
        else:
            raise TypeError(f'Cannot add station, because the timestamp is '
                            f'not specified.')
        if hasattr(station, 'snapshot'):
            self.stations[timestamp] = station
            station.settings_manager = self
        else:
            raise TypeError(f'Cannot add station "{timestamp}", because the '
                            'station is not a QCode or Mock station class '
                            'or is not snapshotable.')

    def load_from_file(self, timestamp: str, folder=None, filepath=None,
                       file_id=None, param_path=None):
        """
        Loads station into the settings manager from saved files.
        Files contain dictionary of the snapshot (pickle, msgpack) or a hdf5
        representation of the snapshot.
        Args:
            timestamp (str): Supports all formats from a_tools.get_folder
            folder (str): Optional, folder of the file if distinct from
                a_tools.datadir
            file_id (str): suffix of the file
            filepath (str): filepath of the file, overwrites timestamp and
                folder
            param_path (list(str)): List of path to parameter which should be
                loaded to the station. Used to load HDF5 files faster. Set to
                None of entire file should be loaded.

        Returns: Station which was created by the saved data.
        """
        station = get_station_from_file(timestamp=timestamp, folder=folder,
                                        filepath=filepath, file_id=file_id,
                                        param_path=param_path)

        self.add_station(station, timestamp)
        return station

    def update_station(self, timestamp, param_path=None, **kwargs):
        """
        Only relevant for HDF-files. Other files will load the entire station
        to the settings manager due to small execution time.
        Updates the station specified by the timestamp with the parameters given
        in param_path. If no param_path are given, the entire station will be
        loaded from the timestamp
        Args:
            timestamp (str, int): timestamp of the station which should be
                updated. If timestamp not in settings manager, the station
                will be loaded in to the settings manager.
                If int it tries to retrieve the timestamp from the loaded
                stations.
            param_path (list): list of parameters which are loaded into a
                station. If None, all parameters are loaded into the station.
                Parameters must be of the form %inst_name%.%param_name%.
            **kwargs: See docstring of self.load_from_file.
        """
        if isinstance(timestamp, int):
            try:
                timestamp = list(self.stations.keys())[timestamp]
            except IndexError as e:
                logger.error(f"Integer seems to be out of range of list"
                             f" '{list(self.stations.keys())}'")
                raise e

        if timestamp not in self.stations.keys():
            self.load_from_file(timestamp, param_path=param_path, **kwargs)
        else:
            if param_path is None:
                # Remove the reference to allow python garbage collection to
                # collect the previous settings_manager if it is not needed
                # anymore.
                self.stations[timestamp].settings_manager = None
                self.stations.pop(timestamp)
                self.load_from_file(timestamp=timestamp)
            else:
                tmp_station = get_station_from_file(timestamp=timestamp,
                                                    param_path=param_path,
                                                    **kwargs)
                self.stations[timestamp].update(tmp_station)

    def get_parameter(self, path_to_param, timestamp,):
        '''
        Returns the parameter value of the parameter specified by
        'path_to_param' from the station of the timestamp. If the parameter or
        the timestamp does not exist in the SettingsManager object, it tries to
        update the station. See self.update_station.
        Args:
            path_to_param (str): path to the parameter of the form
                %instrument%.%parameter% or %instrument%.%submodule.%parameter%
            timestamp (int, str): timestamp or index of the station.

        Returns: Parameter (raw) value, see docstring Station.get().
        TODO: Test for qcodes stations.
        '''
        try:
            # FIXME: If an instrument is requested (e.g. .get('qb1')), but the
            #  instrument was just partially loaded (e.g. only 'qb1.T1' exists),
            #  it will return what was partially loaded and not what is in the
            #  file.
            return self.stations[timestamp].get(path_to_param)
        # If station is not loaded to the SM a KeyError will be raised.
        # If parameter does not exist in station a ParameterNotFoundError will
        # be raised.
        except (KeyError, ParameterNotFoundError) as e:
            self.update_station(timestamp, [path_to_param])
            # second try after station was updated
            # Exception will be raised here if update was not helpful
            return self.stations[timestamp].get(path_to_param)

    def get_instrument_objects(self, timestamp=-1):
        """
        Returns a list of Instrument objects which are assigned to the station
        with a given timestamp.
        Args:
            timestamp(str, int): timestamp or index of the station. Default -1
                refers to the last station added to self.stations.

        Returns: List of Instrument objects.

        """
        return list(self.stations[timestamp].components.values())

    def spawn_snapshot_viewer(self, timestamp, new_process=False):
        """
        Opens a gui window to display the snapshot of the given station.0
        Args:
            timestamp (str): address the station which is already loaded onto
                the settings manager.
            new_process (bool): True if new process should be started, which
                does not block the IPython kernel. False by default because
                it takes some time to start the new process.
        Returns:

        """
        if timestamp not in self.stations.keys():
            print(f"timestamp '{timestamp}' will be loaded onto the station.")
            self.load_from_file(timestamp)
        snapshot_viewer = dict_viewer.SnapshotViewer(
            snapshot=self.stations[timestamp].snapshot(),
            timestamp=timestamp)
        snapshot_viewer.spawn_viewer(new_process=new_process)

    def _compare_dict_instances(self, dict_list, name_list):
        """
        Helper function which compares n dictionaries and returns a combined
        dictionary with the intersection of all keys which values do not
        coincide
        Args:
            dict_list (list of dict): list of dictionaries which are compared
            name_list (list of strings): list of unique names of the
                dictionaries (e.g. timestamp). Names are used as Timestamps
                 objects for unique keys for the combined dictionaries.

        Returns (dict): combined dictionary of with values which do not coincide
            between the different dictionaries

        """
        all_keys = set()
        all_diff = {}
        # set of all keys
        for dic in dict_list:
            all_keys.update(set(dic.keys()))
        for key in all_keys:
            diff = {}
            key_not_in_dicts = False
            dicts_with_key = []

            for i, dic in enumerate(dict_list):
                if key not in dic.keys():
                    # flags that at least one of the dictionaries does not have
                    # the particular key
                    key_not_in_dicts = True
                else:
                    dicts_with_key.append(i)

            # if at least one of the dictionaries does not have the key, the
            # compared dictionaries is build
            if key_not_in_dicts:
                diff[key] = \
                    {Timestamp(name_list[i]): dict_list[i][key]
                     for i in dicts_with_key}
            else:
                # if all the dictionaries have the key, the values are compared
                for i, dic in enumerate(dict_list[1:]):
                    try:
                        np.testing.assert_equal(dict_list[0][key], dic[key])
                    except AssertionError:
                        # items do not coincide
                        if all(isinstance(dic[key], dict) for dic in dict_list):
                            diff[key] = self._compare_dict_instances(
                                [dic[key] for dic in dict_list], name_list)
                            break
                        else:
                            # this occurs when not all items are dictionaries
                            diff[key] = \
                                {Timestamp(name_list[i]): dic[key]
                                 for i, dic in enumerate(dict_list)}
                            break
            all_diff.update(diff)

        return all_diff

    def _compare_station_components(self, timestamps, instruments='all',
                                    reduced_compare=False):
        """
        Helper function to compare multiple stations
        Args:
            timestamps (list of str): timestamps of the station of the settings
                manager which are compared
            instruments (str, list of str): either 'all' (default) or a list of
                instrument names to compare only a subset of instruments
            reduced_compare (bool): if True it compares only the reduced
                snapshot of the stations (i.e. only values and no metadata)

        Returns:
            dict and str of the compared dictionary and the messages of the
            missing instruments/components

        """
        all_components = set()
        all_msg = []
        all_diff = {}

        # create a set of all components of all given stations
        for tsp in timestamps:
            all_components.update(set(self.stations[tsp].components.keys()))

        # check which components are in all stations
        for component in all_components:
            if instruments != 'all' and component not in instruments:
                continue

            # marker if all stations contain component
            component_not_in_all_stations = False
            comp_dict = {}
            for tsp in timestamps:
                if component not in self.stations[tsp].components.keys():
                    all_msg.append(
                        f'\nComponent/Instrument "{component}" missing in dict '
                        f'{tsp}.\n')
                    component_not_in_all_stations = True
                else:
                    comp_dict[Timestamp(tsp)] = 'exists'
            if component_not_in_all_stations:
                # if one of the stations does not have the component, the
                # comparison of this component cannot be done
                # (You have to compare like with like)
                all_diff[component] = comp_dict
            else:
                # snapshots of the components
                if reduced_compare:
                    component_snaps = \
                        [self.stations[tsp].components[component].snapshot(
                            reduced=reduced_compare)
                            for tsp in timestamps]
                else:
                    component_snaps = \
                        [self.stations[tsp].components[component].snapshot()
                         for tsp in timestamps]

                # comparison of the snapshots
                diff = self._compare_dict_instances(component_snaps, timestamps)
                if diff != {}:
                    all_diff[component] = diff

        return all_diff, all_msg

    def compare_stations(self, timestamps, instruments='all',
                         reduced_compare=False, output='viewer',
                         new_process=False):
        """
        Compare instrument settings from n different station in the settings
        manager.
        Args:
            timestamps (list of str): timestamps of the stations which are
                compared. If station is not part of the settings_manager object
                it will be loaded into the settings_manager.
            instruments (str, list of str): either 'all' (default) or a list of
                instrument names to compare only a subset of instruments
            reduced_compare (bool): if True it compares only the reduced
                snapshot of the stations (i.e. only values and no metadata)
            output (str): One of the following output formats:
                'str': return comparison report as str
                'dict': return comparison results as a dict
                'viewer' (default): opens a gui with the compared dictionary
            new_process (bool): True if new process should be started, which
                does not block the IPython kernel. False by default because
                it takes some time to start the new process.
        """
        if timestamps == 'all':
            ts_list = list(self.stations.keys())
        elif isinstance(timestamps, list):
            ts_list = timestamps
        else:
            raise NotImplementedError(f'Timestamp "{timestamps}" is not a '
                                      f'list of timestamps or "all"')

        # checks if stations with respective timestamps are already in the
        # settings manager. otherwise, it loads the station into the sm
        for tsp in set(ts_list).difference(self.stations.keys()):
            print(f"timestamp '{tsp}' will be loaded onto the station.")
            self.load_from_file(tsp)

        # for QCode station reduced comparison is not supported
        if reduced_compare:
            for ts in ts_list:
                if not isinstance(self.stations[ts], Station):
                    logger.warning(
                        f"QCode stations do not support reduced comparison. "
                        f"reduced_compare is set to False.")
                    reduced_compare = False

        diff, msg = self._compare_station_components(
            ts_list, instruments=instruments, reduced_compare=reduced_compare)

        if output == 'str':
            return msg
        elif output == 'dict':
            return diff
        elif output == 'viewer':
            snapshot_viewer = dict_viewer.SnapshotViewer(
                snapshot=diff,
                timestamp=ts_list)
            snapshot_viewer.spawn_viewer(new_process=new_process)


class Timestamp(str):
    """
    Generic timestamp class. Used to distinguish between regular strings and
    timestamp strings. Can be extended with custom timestamp functions (e.g.
    validator of timestamps, output in a particular format, ...)
    """
    pass


def get_loader_from_format(file_format, **kwargs):
    """
    Returns a loader instance of the respectice file format which is passed as
    an argument.
    Args:
        file_format (str): File format as a string, see base_io.file_extensions
        **kwargs: Arguments of the loader class. See respective loader init
        docstring.

    Returns: Loader instance.

    """
    if 'hdf5' in file_format:
        return HDF5Loader(**kwargs)
    elif 'pickle' in file_format:
        return PickleLoader(**kwargs)
    elif 'msgpack' in file_format:
        return MsgLoader(**kwargs)
    else:
        raise NotImplementedError(f"File format '{file_format}' "
                                  f"not supported!")


def get_loader_from_file(timestamp=None, folder=None, filepath=None,
                         file_id=None, **params):
    """
    Returns a loader instance which loads instrument settings from a file
        specified by a timestamp or a filepath. If filepath is not none, it
        overwrites the timestamp.
    Args:
        timestamp(str): In the format YYYYMMDD_HHMMSS
        folder(str): Folder which hosts the timestamp. a_tools.datadir will not
            be considered when folder is not none.
        filepath(str): The actual filepath of the file. Overwrites timestamp and
            folder.
        file_id(str): File id, can be used when several instrument settings have
            the same timestamp (see a_tools.measurement_filename())
        **kwargs: additional kwargs for the specific loader classes, e.g.
            custom file extensions.

    Returns: respective loader class instance (HDF5Loader, PickleLoader or
        MsgLoader)

    """
    file_format = Loader.get_file_format(timestamp=timestamp, folder=folder,
                                         filepath=filepath, file_id=file_id)
    if 'comp' in file_format:
        compression = True
    else:
        compression = False
    kwargs = dict(timestamp=timestamp, filepath=filepath,
                  compression=compression, folder=folder,
                  file_id=file_id)
    return get_loader_from_format(file_format, **kwargs)


def get_station_from_file(timestamp=None, folder=None, filepath=None,
                          file_id=None, param_path=None, **kwargs):
    """
    Returns a station object from an instrument settings file. The file can be
    specified by a timestamp, folder path or filepath. The list param_path can
    be specified to load only certain parameters from the settings file.

    Args:
        timestamp(str): In the format YYYYMMDD_HHMMSS
        folder(str): Folder which hosts the timestamp. a_tools.datadir will not
            be considered when folder is not none.
        filepath(str): The actual filepath of the file. Overwrites timestamp and
            folder.
        file_id(str): File id, can be used when several instrument settings have
            the same timestamp (see a_tools.measurement_filename())
        param_path (list(str)): List of path to parameters which are loaded.
            Reduces the loading time of hdf5-files tremendously.
            If not specified, the entire instrument settings will be loaded into
            the station (always like this for non-hdf5 file formats).
            e.g.: ['qb1.T1', 'AWG2.sigouts_3_range']
        **kwargs: additional kwargs for the specific loader classes, e.g.
            custom file extensions.

    Returns: mock_qcodes_interface.station instance with the instrument
        settings loaded into it.

    """
    return get_loader_from_file(timestamp=timestamp, folder=folder,
                                filepath=filepath, file_id=file_id) \
        .get_station(param_path=param_path)


def convert_settings_to_hdf(timestamp: str):
    """
    Creates/writes settings to a hdf5-file specified by a timestamp.
    Write the instrument settings into the preexisting hdf-file with the
    same timestamp from any settings file supported by the settings manager.
    If the hdf-file does not exist, it creates a hdf-file with the same
    filename as the settings file.
    This serves as a helper to ensure compatibility with user-notebooks which
    rely on instrument settings being stored in hdf5-files.

    Args:
        timestamp(str): Timestamp of the settings file.
    """
    from pycqed.analysis import analysis_toolbox as a_tools
    from pycqed.measurement.measurement_control import MeasurementControl
    from pycqed.utilities.io import base_io

    station = get_station_from_file(timestamp)
    fn = a_tools.measurement_filename(a_tools.get_folder(timestamp))
    # if hdf-file does not exist, the filename of the settings file is copied
    if fn is None:
        file_format = Loader.get_file_format(timestamp=timestamp)
        ext = base_io.file_extensions[file_format]
        # a_tools expects extension without a dot (e.g. 'hdf'),
        # the extension dict in base_io stores it with a dot (e.g. '.hdf')
        fn = a_tools.measurement_filename(a_tools.get_folder(timestamp),
                                          ext=ext[1:])
        fn = fn[:-len(ext)] + '.hdf'
    with h5py.File(fn, 'a') as hdf_file:
        MeasurementControl.save_station_in_hdf(hdf_file, station)
