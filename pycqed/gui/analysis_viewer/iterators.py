"""Iterator classes used in `pycqed.gui.analysis_viewer.analysis_viewer` module.

Used to iterate through experiments/timestamps and figures inside them.
"""
from typing import Any, Optional, Self

from pycqed.analysis import analysis_toolbox
from pycqed.analysis_v2 import base_analysis


class BidirectionalIterator(object):
    """Generic iterator class traversable in both directions.

    Attributes:
        collection: collection of object to traverse.
        index: Integer, stores the position of current element.

    Raises:
        StopIteration: if iteration end is reached in either direction.
    """

    def __init__(self, collection: list):
        """Initializes the instance with collection provided.

        Args:
            collection: list of object to iterate over.
       """
        self.collection = collection
        self.index: int = -1

    def next(self) -> Any:
        """Gets next element and points `index` to it.

        Raises:
            StopIteration: if iteration end is reached in forward direction.

        Returns:
            An element of the collection the index/pointer points to.
        """
        try:
            index = self.index + 1
            result = self.collection[index]
            self.index = index
        except IndexError:
            raise StopIteration
        return result

    def prev(self) -> Any:
        """Gets previous element and points `index` to it.

        Raises:
            StopIteration: if iteration end is reached in backward direction.

        Returns:
            An element of the collection the index/pointer points to.
        """
        index = self.index - 1
        if index < 0:
            raise StopIteration
        self.index = index
        return self.collection[self.index]

    def last(self) -> Any:
        """Gets the last element and points `index` to it.

        Returns:
            The last element of the collection.
        """
        self.index = len(self.collection) - 1
        return self.collection[-1]

    def __iter__(self) -> Self:
        """Ensures compliance with python iterator interface.

        Returns:
            BidirectionalIterator: self.
        """
        return self

    def __next__(self) -> Any:
        """Ensures compliance with python iterator interface.

        Returns:
            An element of the collection the index/pointer points to.
        """
        return self.next()


class TimestampBidirectionalIterator(object):
    """Specific two level iterator class: daystamps, timestamps.

    Iterates over timestamps in current daystamp and daystamps.
    If `next()` is called on the last timestamp in a daystamp,
    the `daystamp_iterator` goes forward by one and the first timestamp
    of that daystamp is returned.
    If `prev()` is called for the first timestamp, the `daystamp_iterator`
    goes back by one and the last timestamp of that daystamp is returned.

    Attributes:
        daystamp_iterator: `BidirectionalIterator` iterator for
            `daystamp_collection` provided.
        timestamp_iterator: `BidirectionalIterator` for timestamps in the
            daystamp the `daystamp_iterator` points to.

    Raises:
        StopIteration: if iteration end is reached in either direction.
    """

    def __init__(self, daystamp_collection: list):
        """Initializes the instance with `daystamp_collection` provided.

        Creates `daystamp_iterator` from provided `daystamp_collection` and
        `timestamp_iterator` from the first element of `daystamp_iterator`.

        Args:
            daystamp_collection: collection of daystamps that are passed to
                `BidirectionalIterator` to create `daystamp_iterator`.
        """
        self.timestamp_iterator: BidirectionalIterator
        self.daystamp_iterator: BidirectionalIterator = (
            BidirectionalIterator(daystamp_collection))
        self.set_new_timestamp_iterator_by_daystamp(
            self.daystamp_iterator.next()
        )

    def set_new_timestamp_iterator_by_daystamp(self, daystamp: str):
        """Gets timestamps by `daystamp` and creates `timestamp_iterator`.

        Args:
            daystamp: string by which to construct `timestamp_iterator`.

        Raises:
            ValueError: if `daystamp` passed to
            `analysis_toolbox.verify_daystamp` is invalid.
        """
        analysis_toolbox.verify_daystamp(daystamp)
        self.timestamp_iterator = BidirectionalIterator(
            analysis_toolbox.get_timestamps_by_daystamp(daystamp)[1])

    def next(self) -> str:
        """Gets next element.

        If it currently points to the last timestamp, then it gets the next
        daystamp and returns the first timestamp of that day.

        Raises:
            StopIteration: if iteration end is reached in forward direction.

        Returns:
            str: A timestamp folder path. For example:
                'C:\\\\Users\\\\Username\\\\pycqed\\\\data\\\\20230511
                \\\\141202_resonator_scan_qb1'.
        """
        try:
            result = self.timestamp_iterator.next()
        except StopIteration:
            try:
                self.set_new_timestamp_iterator_by_daystamp(
                    self.daystamp_iterator.next())
                result = self.timestamp_iterator.next()
            except ValueError:
                raise StopIteration
        return result

    def prev(self) -> str:
        """Gets previous element.

        If it currently points to the first timestamp, then it gets the
        previous daystamp and returns the last timestamp of that day.

        Raises:
            StopIteration: if iteration end is reached in backward direction.

        Returns:
            str: A timestamp folder path. For example:
                'C:\\\\Users\\\\Username\\\\pycqed\\\\data\\\\20230511
                \\\\141202_resonator_scan_qb1'.
        """
        try:
            result = self.timestamp_iterator.prev()
        except StopIteration:
            try:
                self.set_new_timestamp_iterator_by_daystamp(
                    self.daystamp_iterator.prev())
                result = self.timestamp_iterator.last()
            except ValueError:
                raise StopIteration
        return result

    def __iter__(self) -> Self:
        """Ensures compliance with python iterator interface.

        Returns:
            TimestampBidirectionalIterator: self
        """
        return self

    def __next__(self) -> str:
        """Ensures compliance with python iterator interface.

        Returns:
            str: A timestamp folder path. For example:
                'C:\\\\Users\\\\Username\\\\pycqed\\\\data\\\\20230511
                \\\\141202_resonator_scan_qb1'.
        """
        return self.next()


class ExperimentIterator(object):
    """Specific iterator for `BaseDataAnalysis` objects created from timestamps.

    Wrapping class to use with `TimestampBidirectionalIterator`. It converts
    the timestamps into `BaseDataAnalysis` objects if a given timestamp has a
    job string saved in its HDF5 file. If not, it traverses the iterator (in
    the given direction) until it finds a timestamp with a job file. Then
    from the job string it recreates the experiment object and return it.

    Attributes:
        timestamp_bidirectional_iterator: `TimestampBidirectionalIterator`
            for iteration

    Raises:
        StopIteration: if iteration end is reached in either direction.
    """

    def __init__(self,
                 timestamp_bidirectional_iterator:
                 TimestampBidirectionalIterator):
        """Initializes with `timestamp_bidirectional_iterator` provided.

        Args:
            timestamp_bidirectional_iterator:
                `TimestampBidirectionalIterator` used for actual iteration,
                from whose elements `BaseDataAnalysis` objects are recovered.
        """
        self.timestamp_bidirectional_iterator: TimestampBidirectionalIterator \
            = timestamp_bidirectional_iterator

    def set_pointer_to_timestamp(self, timestamp_folder: str):
        """Points `timestamp_bidirectional_iterator` to `timestamp_folder`.

        Args:
            `timestamp_folder`: string which specifies full path to a folder
                that holds the HDF5 file of the experiment. For example
                'C:\\\\Users\\\\Username\\\\pycqed\\\\data\\\\20230511
                \\\\141202_resonator_scan_qb1'.
        """
        while True:
            current = self.timestamp_bidirectional_iterator.next()
            if timestamp_folder == current:
                break

    def next(self) -> base_analysis.BaseDataAnalysis:
        """Gets next element.

        Iterates over `self.timestamp_bidirectional_iterator` in a forward
        direction until it finds an experiment with a job string saved in
        HDF5 file from which an analysis/experiment object can be constructed
        and returns it.

        Raises:
            StopIteration: if iteration end is reached in forward direction.

        Returns:
            base_analysis.BaseDataAnalysis: `BaseDataAnalysis` object.
        """
        analysis_object = None
        while not isinstance(analysis_object, base_analysis.BaseDataAnalysis):
            experiment_folder_path = (
                self.timestamp_bidirectional_iterator.next())
            analysis_object = self._get_analysis_object(experiment_folder_path)
        return analysis_object

    def prev(self) -> base_analysis.BaseDataAnalysis:
        """Gets previous element.

        Iterates over `self.timestamp_bidirectional_iterator` in a backward
        direction until it finds an experiment with a job string saved in
        HDF5 file from which an analysis/experiment object can be constructed
        and returns it.

        Raises:
            StopIteration: if iteration end is reached in backward direction.

        Returns:
            base_analysis.BaseDataAnalysis: `BaseDataAnalysis` object.
        """
        analysis_object = None
        while not isinstance(analysis_object, base_analysis.BaseDataAnalysis):
            experiment_folder_path = (
                self.timestamp_bidirectional_iterator.prev())
            analysis_object = self._get_analysis_object(experiment_folder_path)
        return analysis_object

    @staticmethod
    def _get_analysis_object(experiment_folder_path: str) -> Optional[
        base_analysis.BaseDataAnalysis]:
        """Reconstructs `BaseDataAnalysis` from `experiment_folder_path`.

        Args:
            `experiment_folder_path`: string which specifies full path to a
                folder that holds the HDF5 file of the experiment.
        Returns:
            Optional[base_analysis.BaseDataAnalysis]: `BaseDataAnalysis`
                object reconstructed from job string in the file or None.
        """
        file_path = analysis_toolbox.measurement_filename(
            experiment_folder_path
        )
        try:
            return (base_analysis.BaseDataAnalysis
                    .get_analysis_object_from_hdf5_file_path(file_path))
        except Exception:
            return None

    def __iter__(self) -> Self:
        """Ensures compliance with python iterator interface.

        Returns:
            ExperimentIterator: self.
        """
        return self

    def __next__(self) -> base_analysis.BaseDataAnalysis:
        """Ensures compliance with python iterator interface.

        Returns:
            base_analysis.BaseDataAnalysis: `BaseDataAnalysis` object.
        """
        return self.next()
