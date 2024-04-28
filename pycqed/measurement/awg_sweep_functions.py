import logging
from pycqed.measurement import sweep_functions as swf
from pycqed.measurement import sweep_points as sp_mod

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


class BlockSoftHardSweep(swf.UploadingSweepFunction, swf.Soft_Sweep):

    supports_batch_mode = True

    def __init__(self, circuit_builder, params, block=None,
                 block_func=None, parameter_name='None',
                 unit='', upload=True, **kw):
        """Sweep function used to efficiently iterate between different
        parameter sets of a parameterized quantum circuit (represented by a
        `Block`).

        This can be used e.g. for adaptive measurements where the sequences
            are not known in advance.
        Args:
            circuit_builder (CircuitBuilder): Instance of CircuitBuilder that
                is used to compile the block into sequences.
            params (list[str]): List of the ParametricValues names in `block`.
            block (Block, optional): Block that contains ParametricValues and
                is compiled into sequences using circuit_builder. Either block
                or block_func need to be specified. Defaults to None.
            block_func (Callable, optional): Function that is passed to
                sweep_n_dim. Either block or block_func need to be specified.
                Defaults to None.
            parameter_name (str, optional): _description_. Defaults to 'None'.
            unit (str, optional): Unit of the sweep parameter. Defaults to ''.
            upload (bool, optional): Whether to upload the sequences before
                measurement. Defaults to True.
            kw (optional): Keyword arguments, e.g., `sweep_kwargs` which will
                be passed to `sweep_n_dim` in `self.set_parameter`.
        """
        super().__init__(sequence=None, upload=upload,
                         upload_first=False,
                         parameter_name=parameter_name, unit=unit, **kw)
        self.name = 'Block soft sweep'
        self.block = block
        self.block_func = block_func
        self.circuit_builder = circuit_builder
        self.params = params
        self.sweep_points = None

    def set_parameter(self, vals, **kw):
        """Compiles the `Block` for the given values into a sequence and
        uploads it to hardware.

        Args:
            vals (array): Parameter values to be swept. Shape: [number of
            different sets of parameters (= number of points to be
            measured), number of parameters (= len(self.params))]

        `self.circuit_builder.sweep_n_dim` is used to convert vals into a hard
        sweep sequence which is subsequently uploaded to hardware.
        """
        self.sweep_points = sp_mod.SweepPoints([{
            p: (vals[:, i], '', p)
            for i, p in enumerate(self.params)}])
        seqs, _ = self.circuit_builder.sweep_n_dim(
            sweep_points=self.sweep_points, body_block=self.block,
            body_block_func=self.block_func,
            **(getattr(self, 'sweep_kwargs', {})))
        self.sequence = seqs[0]
        self.upload_sequence()

    def configure_upload(self, upload=True, upload_first=False,
                         start_pulsar=True):
        super().configure_upload(upload, upload_first, start_pulsar)
        return True

    def get_nr_parameters(self):
        return len(self.params)


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
