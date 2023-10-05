import logging
from pycqed.measurement import sweep_functions as swf

log = logging.getLogger(__name__)

class File(swf.Hard_Sweep):

    def __init__(self, filename, AWG, title=None, NoElements=None, upload=True):
        self.upload = upload
        self.AWG = AWG
        if title:
            self.name = title
        else:
            self.name = filename
        self.filename = filename + '_FILE'
        self.upload = upload
        self.parameter_name = 'amplitude'
        self.unit = 'V'

    def prepare(self, **kw):
        if self.upload:
            self.AWG.set_setup_filename(self.filename)


class SegmentHardSweep(swf.UploadingSweepFunction, swf.Hard_Sweep):
    # The following allows adding the class as placeholder in a
    # multi_sweep_function
    unit = ''

    def __init__(self, sequence, upload=True, parameter_name='None', unit='',
                 start_pulsar=False, start_exclude_awgs=(), **kw):
        """A hardware sweep over segments in a sequence.

        Args:
            sequence (:class:`~pycqed.measurement.waveform_control.sequence.Sequence`):
                Sequence of segments to sweep over.
            upload (bool, optional):
                Whether to upload the sequences before measurement.
                Defaults to True.
            parameter_name (str, optional):
                Name for the sweep parameter. Defaults to 'None'.
            unit (str, optional):
                Unit for the sweep parameter. Defaults to ''.
            start_pulsar (bool, optional):
                Whether (a sub set of) the used AWGs will be started directly
                after upload. This can be used, e.g., to start AWGs that have
                only one unique segment and do not need to be synchronized to
                other AWGs and therefore do not need to be stopped when
                switching to the next segment in the sweep. Defaults to False.
            start_exclude_awgs (collection[str], optional):
                A collection of AWG names that will not be started directly
                after upload in case start_pulsar is True. Defaults to ().
        """
        super().__init__(sequence=sequence, upload=upload,
                         upload_first=True,
                         start_pulsar=start_pulsar,
                         start_exclude_awgs=start_exclude_awgs,
                         parameter_name=parameter_name, unit=unit, **kw)
        self.name = 'Segment hard sweep'

    def set_parameter(self, value):
        pass


class SegmentSoftSweep(swf.UploadingSweepFunction, swf.Soft_Sweep):
    # The following allows adding the class as placeholder in a
    # multi_sweep_function
    unit = ''

    def __init__(self, sequence_list, parameter_name='None', unit='',
                 upload_first=False, upload=True, **kw):
        super().__init__(sequence=sequence_list[0], upload=upload,
                         upload_first=upload_first,
                         parameter_name=parameter_name, unit=unit, **kw)
        self.name = 'Segment soft sweep'
        self.sequence_list = sequence_list
        self._is_first_sweep_point = True

    def set_parameter(self, val, **kw):
        self.sequence = self.sequence_list[int(val)]
        if self._is_first_sweep_point and val == 0:
            # upload has been done in self.prepare or by the hard sweep
            pass
        else:
            self.upload_sequence()
        self._is_first_sweep_point = False
