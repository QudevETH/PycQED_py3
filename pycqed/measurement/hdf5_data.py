# Wrapper module

import logging
log = logging.getLogger(__name__)
import pycqed.utilities.io.hdf5 as hdf5
import pycqed.utilities.io.base_io as base_io

log.warning("Deprecated module, will be removed in a future MR;"
                " Please use utilities.io.hdf5 instead.")

DateTimeGenerator = base_io.DateTimeGenerator
Data = hdf5.Data
encode_to_utf8 = hdf5.encode_to_utf8
write_dict_to_hdf5 = hdf5.write_dict_to_hdf5
read_dict_from_hdf5 = hdf5.read_dict_from_hdf5
