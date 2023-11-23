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


def get_hdf_group_by_name(parent_group: h5py.Group, group_name: str) -> \
        h5py.Group:
    """Gets group named `group_name` and creates it if it doesn't exist.

    Args:
        parent_group: a group of file under which to look/create the group
            `group_name`.
        group_name: string value of the group one wants to get.

    Returns:
        h5py.Group: either existing or a new group named `group_name`.
    """
    try:
        group = parent_group.create_group(group_name)
    except ValueError:
        # If the group already exists.
        group = parent_group[group_name]
    return group
