"""
MULTI CORE COMPILER

Version 1.1.0

History:
- 1.1.0: Some improvments. Removed the session parameter from the MCC, not needed anymore
- 1.0.0: Initial revisiion.
"""

import warnings
import numpy as np
import time
import contextlib
import os.path
import tempfile
import zhinst.toolkit.nodetree.helper as helper
from zhinst.toolkit.nodetree import Node, NodeTree


# Check min ziPython version at import

from zhinst.ziPython import __version__ as zi_python_version

MIN_ZIPYTHON = "22.02.29711"
ver_list = [int(i) for i in zi_python_version.split(".")]
min_ver_list = [int(i) for i in MIN_ZIPYTHON.split(".")]


for min_vi, vi in zip(min_ver_list, ver_list):
    if vi < min_vi:
        raise RuntimeError(
            "zhinst.ziPython version does not match the minimum required version "
            f"for MultiCoreCompiler: {zi_python_version} < {MIN_ZIPYTHON}. "
            "Use `pip install --upgrade zhinst` to get the latest version."
        )

    if vi > min_vi:
        break


class MultiCoreCompiler:
    """A class to compile and upload sequences using multiple threads

    Args:
        awgs (AWG): A list of the AWG Nodes that are target for parallel compilation
        use_tempdir (bool, optional): Use a temporary directory to store the generated sequences
    """

    def __init__(self, *awgs, use_tempdir=False):
        if use_tempdir:
            self._filedir = tempfile.TemporaryDirectory()

        self._awgs = {}
        self.filename = "awg_default"
        for awg in awgs:
            self.add_awg(awg)

    def __del__(self):
        # Stop the AWG modules
        for awg, awg_nodetree in self._awgs.items():
            try:
                awg_nodetree.connection.finish()
            except RuntimeError as e:
                # continue cleanup even if the awg could not be stopped.
                print(f"Failed to stop AWG {awg}: {e}")

        # cleanup the temp directory
        if hasattr(self, "_filedir"):
            self._filedir.cleanup()

    def _get_elf_filename(self, awg, absolute=True):
        """Get the ELF filename

        Args:
            awg (AWG): The AWG Node
            absolute (bool, optional): If True, return the full absolute path. Otherwise
            only the reduce name to be passed to the AWG module. Defaults to True.

        Returns:
            str: the ELF filename
        """
        elf_filename = self.filename + ".elf"
        if not absolute:
            return elf_filename
        # Get the toolkit AWG object if it's a qcodes AWG object
        awg_tk = awg if isinstance(awg, Node) else awg._tk_object
        # Get directory path from independent AWG module.
        base_dir = self._awgs[awg].directory()
        return os.path.normpath(
            os.path.join(
                base_dir,
                "awg/elf",
                f"{awg_tk._serial:s}_{awg_tk._index:d}_{elf_filename:s}",
            )
        )

    def add_awg(self, awg):
        """Add a AWG core to the multiple compilation

        Args:
            awg (AWG): The AWG Nodes that is target for parallel compilation
        """

        # Get the toolkit AWG object if it's a qcodes AWG object
        awg_tk = awg if isinstance(awg, Node) else awg._tk_object

        # Create the AWG module
        raw_awg = awg_tk.root.connection.awgModule()
        awg_nodetree = NodeTree(raw_awg)
        awg_nodetree.device(awg_tk._serial)
        awg_nodetree.index(awg_tk._index)

        # Disable automatic upload of ELF to the device
        awg_nodetree.compiler.upload(False)

        # Set the sequncer type, if needed (SHFQC only, but it's no harm to set it
        # also on SHFSG or SHFQA)
        if "sgchannels" in awg_tk.raw_tree:
            awg_nodetree.sequencertype("sg")
        elif "qachannels" in awg_tk.raw_tree:
            awg_nodetree.sequencertype("qa")

        # Use tempdir for ELF if present
        if hasattr(self, "_filedir"):
            awg_nodetree.directory(self._filedir.name)

        # start the AWG module thread
        raw_awg.execute()

        # add the module to the list
        self._awgs[awg] = awg_nodetree

    def load_sequencer_program(self, awg, seqc):
        """Compiles the current sequence program on the AWG Core.
        Not blocking, compilation is started, but not waited for its end.
        The compiled sequence is (temporary) stored on disk in the file
        specified by the property 'filename'

        Args:
            awg (AWG): The AWG Node
            seqc (str): The sequence to be compiled
        """
        if awg not in self._awgs.keys():
            raise ValueError("This AWG has not been initialized")
        awg_nodetree = self._awgs[awg]
        awg_nodetree.elf.file(self._get_elf_filename(awg, absolute=False))
        awg_nodetree.compiler.sourcestring(seqc)

    def wait_compile_and_upload(self, upload_timeout=5):
        """Check that all AWG nodes finished compilation and upload the ELF(s) to
        the device(s)

        Args:
            upload_timeout: Timeout for waiting for completion of compilation.

        Raises:
            RuntimeError: Raise an error if compilation failed
        """
        self.wait_compile()
        self.upload(upload_timeout)

    def wait_compile(self):
        """Check that all AWG nodes finished compilation

        Raises:
            RuntimeError: Raise an error if compilation failed
        """

        for awg_nodetree in self._awgs.values():
            while True:
                status = awg_nodetree.compiler.status()
                if status == 0:
                    # Compilation done
                    break
                elif status == 1:
                    # compilation failed
                    raise RuntimeError(
                        "Compilation falied for device",
                        awg_nodetree.device(),
                        "on core",
                        awg_nodetree.index(),
                        ".",
                        "Error message:",
                        awg_nodetree.compiler.statusstring(),
                    )
                elif status == 2:
                    # compilation succeded, with warnings
                    warnings.warn(
                        "Compilation succesful, with warnings for device",
                        awg_nodetree.device(),
                        "on core",
                        awg_nodetree.index(),
                        ".",
                        "Error message:",
                        awg_nodetree.compiler.statusstring(),
                        RuntimeWarning,
                    )
                    break
                else:
                    # compilation still ongoing
                    time.sleep(0.001)

    def upload(self, upload_timeout=5):
        """Upload all the compiled sequence to the respecitve devices.

        Args:
            upload_timeout (int, optional): Timeout of upload in seconds. Defaults to 5s.

        Raises:
            RuntimeError: Raise an error if the device doesn't report a successful upload
        """
        # Get a list of devices
        devices = set(
            [
                awg.root if isinstance(awg, Node) else awg._tk_object.root
                for awg in self._awgs.keys()
            ]
        )

        # Upload all the ELFs
        # for perfomance reasons, send them with a single transactional set.
        # However, due to https://github.com/zhinst/zhinst-toolkit/issues/134, we need to start a
        # set_transaction for each device
        with contextlib.ExitStack() as stack:
            # start the set_transaction for each device. set() is used to get the unique device from their awgs
            [
                stack.enter_context(helper.create_or_append_set_transaction(i))
                for i in devices
            ]
            for awg_obj in self._awgs.keys():
                # load the ELF from disk and send it to the device
                with open(self._get_elf_filename(awg_obj, absolute=True), "rb") as f:
                    vector = np.frombuffer(f.read(), dtype=np.uint32)
                    awg_obj.elf.data(vector)

        # check that upload is finished and the ELF is loaded into the device
        start = time.time()
        for awg_obj in self._awgs.keys():
            while not awg_obj.ready(deep=False):
                if time.time() - start > upload_timeout:
                    raise RuntimeError(
                        "Sequencer", str(awg_obj), "didn't report ready before timeout"
                    )
                time.sleep(0.001)
