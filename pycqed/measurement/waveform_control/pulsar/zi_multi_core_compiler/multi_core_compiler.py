import warnings
import numpy as np
import time
import contextlib
import os.path
import tempfile
import zhinst.toolkit
import zhinst.toolkit.nodetree.helper as helper


#Check min ziPython version at import

from zhinst.ziPython import __version__ as zi_python_version
MIN_ZIPYTHON = "22.02.29711"
ver_list = [int(i) for i in zi_python_version.split('.')]
min_ver_list = [int(i) for i in MIN_ZIPYTHON.split('.')]


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
        self._session = None
        for awg in awgs:
            self.add_awg(awg)

    def __del__(self):
        #Stop the AWG modules
        for awg_mod in self._awgs.values():
            awg_mod.raw_module.finish()

        #cleaup the temp directory
        if hasattr(self, "_filedir"):
            self._filedir.cleanup()

    def _get_fn(self, awg, absolute=True):
        """Get the ELF filename

        Args:
            awg (AWG): The AWG Node
            absolute (bool, optional): If True, return the full absolute path. Otherwise 
            only the reduce name to be passed to the AWG module. Defaults to True.

        Returns:
            str: the ELF filename
        """

        #Get the toolkit AWG object if it's a qcodes AWG object
        awg_tk = awg if isinstance(awg, (zhinst.toolkit.driver.nodes.awg.AWG, zhinst.toolkit.driver.nodes.generator.Generator)) else awg._tk_object

        awg_mod = self._awgs[awg]
        fn = "awg_default.elf"
        if absolute:
            base_dir = awg_mod.directory()
            device = awg_tk._serial
            index = awg_tk._index
            return os.path.normpath(
                os.path.join(base_dir, "awg/elf", f"{device:s}_{index:d}_{fn:s}")
            )
        else:
            return fn

    def add_awg(self, awg):
        """Add a AWG core to the multiple compilation

        Args:
            awg (AWG): The AWG Nodes that is target for parallel compilation
        """

        #Get the toolkit AWG object if it's a qcodes AWG object
        awg_tk = awg if isinstance(awg, (zhinst.toolkit.driver.nodes.awg.AWG, zhinst.toolkit.driver.nodes.generator.Generator)) else awg._tk_object

        # Store the session for later usage
        if self._session is None:
            self._session = awg_tk._session
        else:
            if self._session != awg_tk._session:
                raise RuntimeError('All devices must belong to the same session!')

        #Create the AWG module
        awg_mod = self._session.modules.create_awg_module()
        raw_awg = awg_mod.raw_module
        awg_mod.device(awg_tk._serial)
        awg_mod.index(awg_tk._index)

        #Disable automatic upload of ELF to the device
        awg_mod.compiler.upload(False)

        # Set the sequncer type, if needed (SHFQC only, but it's no harm to set it
        # also on SHFSG or SHFQA)
        is_sgchannel = 'sgchannels' in str(awg_tk)
        is_qachannel = 'qachannels' in str(awg_tk)

        if is_sgchannel:
            awg_mod.sequencertype('sg')
        elif is_qachannel:
            awg_mod.sequencertype('qa')
        
        # Use tempdir for ELF if present
        if hasattr(self, "_filedir"):
            awg_mod.directory(self._filedir.name)

        # start the AWG module thread
        raw_awg.execute()

        # add the module to the list
        self._awgs[awg] = awg_mod

    def load_sequencer_program(self, awg, seqc):
        """Compiles the current sequence program on the AWG Core.
        Not blocking, compilation is started, but not waited for its end.

        Args:
            awg (AWG): The AWG Node
            seqc (str): The sequence to be compiled
        """
        if awg not in self._awgs.keys():
            raise ValueError('This AWG has not been initialized')
        awg_mod = self._awgs[awg]
        awg_mod.elf.file(self._get_fn(awg, absolute=False))
        awg_mod.compiler.sourcestring(seqc)

    def wait_compile_and_upload(self, upload_timeout = 5):
        """Check that all AWG nodes finished compilation and upload the ELF to
        the device(s)

        Raises:
            RuntimeError: _description_
        """
        self.wait_compile()
        self.upload(upload_timeout)

    def wait_compile(self):
        """Check that all AWG nodes finished compilation

        Raises:
            RuntimeError: _description_
        """

        for awg_mod in self._awgs.values():
            while True:
                status = awg_mod.compiler.status()
                if status == 0:
                    # Compilation done
                    break
                elif status == 1:
                    # compilation failed
                    raise RuntimeError(
                        "Compilation falied for device",
                        awg_mod.device(),
                        "on core",
                        awg_mod.index(),
                        ".",
                        "Error message:",
                        awg_mod.compiler.statusstring(),
                    )
                elif status == 2:
                    # compilation succeded, with warnings
                    warnings.warn(
                        "Compilation succesful, with warnings for device",
                        awg_mod.device(),
                        "on core",
                        awg_mod.index(),
                        ".",
                        "Error message:",
                        awg_mod.compiler.statusstring(),
                        RuntimeWarning,
                    )
                    break
                else:
                    # compilation still ongoing
                    time.sleep(0.001)

    def upload(self, upload_timeout = 5):
        #Get all the toolkit AWG objects if it's a qcodes AWG object
        awgs_tk = [awg if isinstance(awg, (zhinst.toolkit.driver.nodes.awg.AWG, zhinst.toolkit.driver.nodes.generator.Generator)) else awg._tk_object for awg in self._awgs.keys()]
        #Get a list of devices
        devices = set([awg.root for awg in awgs_tk])

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
                fn = self._get_fn(awg_obj, absolute=True)
                with open(fn, "rb") as f:
                    data = f.read()
                    vector = np.frombuffer(data, dtype=np.uint32)
                    awg_obj.elf.data(vector)

        # check that upload is finished and the ELF is loaded into the device
        start = time.time()
        for awg_obj in self._awgs.keys():
            while not awg_obj.ready(deep=False):
                if time.time() - start > upload_timeout:
                    raise RuntimeError('Sequencer', str(awg_obj), 'didn\'t report ready before timeout')
                time.sleep(0.001)
