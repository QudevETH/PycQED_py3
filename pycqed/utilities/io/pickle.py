from pycqed.utilities.io.base_io import Loader, file_extensions, Dumper


class PickleLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         **kwargs)

        self.compression = kwargs.get('compression', False)
        if self.extension is None:
            if self.compression:
                self.extension = file_extensions['pickle_comp']
            else:
                self.extension = file_extensions['pickle']
        self.filepath = self.get_filepath(**kwargs)

    def get_snapshot(self):
        """
        Opens the pickle file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import pickle

        with open(self.filepath, 'rb') as f:
            if self.compression:
                byte_data = Loader.decompress_file(f.read())
                snap = pickle.loads(byte_data)
            else:
                snap = pickle.load(f)
        return snap


class PickleDumper(Dumper):

    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        super().__init__(name, data, datadir=datadir, compression=compression,
                         timestamp=timestamp)
        if self.compression:
            self.filepath = self.filepath.replace(
                ".hdf5", file_extensions['pickle_comp'][0])
        else:
            self.filepath = self.filepath.replace(
                ".hdf5", file_extensions['pickle'][0])

    def dump(self, mode='xb'):
        """
        Dumps the data as a binary into a pickle file with optional compression
        mode: define which mode you want to open the file in.
            Default 'xb' creates the file and returns error if file exist
        """
        import pickle
        with open(self.filepath, mode=mode) as file:
            packed = pickle.dumps(self.data)
            if self.compression:
                packed = Dumper.compress_file(packed)
            file.write(packed)
