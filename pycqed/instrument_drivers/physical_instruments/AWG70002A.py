from qcodes.instrument_drivers.tektronix import AWG70002A
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from qcodes import validators as vals

import io
import zipfile as zf



class AWG70002A(AWG70002A.AWG70002A):
    """
    This is the PycQED wrapper driver for the Tektronix AWG70002A
    Arbitrary Waveform Generator.

    Inherited from the QCoDeS driver for Tektronix AWG70002A series AWG's.

    All the actual driver meat is in the QCoDeS superclass AWG70000A.
    """

    def __init__(self, name: str, address: str,
                 timeout: float=10, **kwargs: Any) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).
        """

        super().__init__(name, address, timeout=timeout, **kwargs)
        self.add_parameter('current_drive',
                           label='Current file system drive',
                           set_cmd='MMEMory:MSIS "{}"',
                           get_cmd='MMEMory:MSIS?',
                           initial_value = 'C:',
                           vals=vals.Strings())

    def start(self, **kwargs) -> None:
        """Start the AWG function.
        :param kwargs: currently ignored, added for compatibilty with other
            instruments that accept kwargs in start().
        """
        return self.play(wait_for_running = False)

    @staticmethod
    def makeSEQXFile(trig_waits: Sequence[int],
                     nreps: Sequence[int],
                     event_jumps: Sequence[int],
                     event_jump_to: Sequence[int],
                     go_to: Sequence[int],
                     wfms: Dict,
                     seqname: str,
                     sequence: Sequence[Sequence[str]]) -> bytes:
        """
        Make a full .seqx file (bundle)
        A .seqx file can presumably hold several sequences, but for now
        we support only packing a single sequence

        For a single sequence, a .seqx file is a bundle of two files and
        two folders:

        /Sequences
            sequence.sml

        /Waveforms
            wfm1.wfmx
            wfm2.wfmx
            ...

        setup.xml
        userNotes.txt

        Args:
            trig_waits: Wait for a trigger? If yes, you must specify the
                trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            nreps: No. of repetitions. 0 corresponds to infinite.
            event_jumps: Jump when event triggered? If yes, you must specify
                the trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            event_jump_to: Jump target in case of event. 1-indexed,
                0 means next. Must be specified for all elements.
            go_to: Which element to play next. 1-indexed, 0 means next.
            wfms: A dictionary of waveforms with the key being a waveform name
                and the value -- numpy array describing the waveform.
            seqname: The name of the sequence. This name will appear in the
                sequence list. Note that all spaces are converted to '_'
            sequence: The sequence to be played on the AWG. Contains waveform
                names for each channel of the AWG for each element of the seq.

        Returns:
            The binary .seqx file, ready to be sent to the instrument.
        """

        # input sanitising to avoid spaces in filenames
        seqname = seqname.replace(' ', '_')

        sml_file = AWG70002A._makeSMLFile(trig_waits, nreps,
                                          event_jumps, event_jump_to,
                                          go_to, sequence,
                                          seqname,
                                          len(sequence[0]))

        user_file = b''
        setup_file = AWG70002A._makeSetupFile(seqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode='a')
        zipfile.writestr(f'Sequences/{seqname}.sml', sml_file)

        for (name, wfm) in wfms.items():
            zipfile.writestr(f'Waveforms/{name}.wfmx',
                             AWG70002A.makeWFMXFile(wfm, 2.0))
        # We don't want to normalize the waveform as PycQED already
        # takes care of that. Therefore, we send the amplitude vpp to be 2.0,
        # meaning the qcodes driver normalization scale of 1.

        zipfile.writestr('setup.xml', setup_file)
        zipfile.writestr('userNotes.txt', user_file)
        zipfile.close()

        buffer.seek(0)
        seqx = buffer.getvalue()
        buffer.close()

        return seqx
