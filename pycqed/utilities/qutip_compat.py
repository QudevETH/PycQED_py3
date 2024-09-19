try:
    import qutip as qtp  # To be able to test the version below
    from qutip import *  # To access all qutip contents from this module
    from packaging import version

    if version.parse(qtp.__version__) < version.parse('5'):
        # Versions before 5 contain qip as a submodule
        pass
    else:
        import qutip_qip as qip  # Will be available as this_module.qip
        # Import specific submodules used in pycqed
        # these will be available as this_module.qip.submodule
        import qutip_qip.operations
        import qutip_qip.circuit
    is_imported = True
except ImportError as e:
    import logging
    log = logging.getLogger(__name__)
    is_imported = False
