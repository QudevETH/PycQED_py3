from pycqed.utilities.io.base_io import Dumper, file_extensions, Loader
from pycqed.instrument_drivers import mock_qcodes_interface as mqcodes
import enum

class MsgDumper(Dumper):
    """
    Class to dump dictionaries into msg files.
    """
    def __init__(self, name: str, data: dict, datadir: str = None,
                 compression=False, timestamp: str = None):
        super().__init__(name, data, datadir=datadir, compression=compression,
                         timestamp=timestamp)
        if self.compression:
            self.filepath = self.filepath.replace(
                ".hdf5", file_extensions['msgpack_comp'])
        else:
            self.filepath = self.filepath.replace(".hdf5",
                                                  file_extensions['msgpack'])

    def dump(self, mode='xb'):
        """
        Dumps the data as a binary into a msg file with optional compression
        mode: define which mode you want to open the file in.
            Default 'xb' creates the file and returns error if file exist
        """
        import msgpack
        import msgpack_numpy as msg_np
        msg_np.patch()

        def _pack_default(obj):
            """
            This function is used to transform non-serializable objects into
            serializable objects.
            Function of if-statements which is passed to the Packer object (see
            msgpack_numpy.Packer). Returns an object which can be serialized by
            msgpack.
            Args:
                obj: Object which should be serialized and might not be
                serializable.
            Returns:
                Serializable object.
        """
            if isinstance(obj, enum.Enum):
                return obj.value
            return obj

        with open(self.filepath, mode=mode) as file:
            packed = msgpack.packb(self.data, default=_pack_default)
            if self.compression:
                packed = Dumper.compress_file(packed)
            file.write(packed)


class MsgLoader(Loader):

    def __init__(self, timestamp=None, filepath=None, **kwargs):
        super().__init__(timestamp=timestamp, filepath=filepath,
                         **kwargs)

        self.compression = kwargs.get('compression', False)
        if self.extension is None:
            if self.compression:
                self.extension = file_extensions['msgpack_comp']
            else:
                self.extension = file_extensions['msgpack']
        self.filepath = self.get_filepath(**kwargs)

    def get_snapshot(self):
        """
        Opens the msg file and returns the saved snapshot as a dictionary.
        Returns: snapshot as a dictionary.
        """
        import msgpack
        import msgpack_numpy
        # To save and load numpy arrays in msgpack.
        # Note that numpy arrays deserialized by msgpack-numpy are read-only
        # and must be copied if they are to be modified
        msgpack_numpy.patch()

        with open(self.filepath, 'rb') as f:
            if self.compression:
                byte_data = Loader.decompress_file(f.read())
            else:
                byte_data = f.read()
            # To avoid problems with lists while unpacking use_list has to be
            # set to false. Otherwise, msgpack arrays are converted to Python
            # lists. To allow for tuples as dict keys, strict_map_key has to
            # be set to false. For more information
            # https://stackoverflow.com/questions/66835419/msgpack-dictionary
            # -with-tuple-keys and
            # https://github.com/msgpack/msgpack-python#major-breaking
            # -changes-in-msgpack-10
            snap = msgpack.unpackb(
                byte_data, use_list=False, strict_map_key=False)
        return snap

    def load_parameter(self, param_name, param_value):
        """
        Creates a mock qcodes parameter object from the name and value given as
        input. If the value is a (nested) tuple, it will be converted to a
        (nested) list.
        (Same for the entry 'value' if param_value is a dictionary).
        This is needed, because msgpack deserializes all list and tuple objects
        as tuples (these objects are indistinguishable for msgpack) but some
        qcodes parameters need lists as input (e.g. device.qb_names).
        Args:
            param_name (str): Name of the parameter.
            param_value (dict, obj): Value of the parameter.

        Returns: mock_qcodes_interface.Parameter object

        """
        def convert_tuple_to_list(tp):
            # converts recursively a nested tuple into a nested list
            return list(convert_tuple_to_list(t) for t in tp) \
                if isinstance(tp, tuple) else tp
        if isinstance(param_value, tuple):
            param_value = convert_tuple_to_list(param_value)
        elif isinstance(param_value, dict):
            if 'value' in param_value.keys():
                if isinstance(param_value['value'], tuple):
                    param_value['value'] = \
                        convert_tuple_to_list(param_value['value'])
        return mqcodes.Parameter(param_name, param_value)
