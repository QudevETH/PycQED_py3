from collections import OrderedDict


class IndexableOrderedDict(OrderedDict):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __getitem__(self, i):
        """
        :param i: key, integer or slice
        :return: element or new SweepPoints instance
        """
        if isinstance(i, int):
            new_data = list(super(OrderedDict, self).values())[i]
        # if isinstance(i, slice):
        #     new_data = list(super(OrderedDict, self).values())[i]
        #     new_data = self.__class__(new_data)
        else:
            new_data = super().__getitem__(i)
        return new_data
