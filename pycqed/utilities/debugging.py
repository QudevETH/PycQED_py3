import types
import inspect

def livepatch(obj, cls):
    """Overwrite methods of an object with a new version of its class.

    There might be edge cases in which this does not work properly.

    FIXME: does it work for classmethods?

    Args:
        obj: the object
        cls: the new version of the class (e.g., after reloading a module)

    Examples:
        from importlib import reload
        from pycqed.utilities.debugging import livepatch
        from pycqed.measurement.waveform_control import pulsar as pulsar_mod
        reload(pulsar_mod)
        livepatch(pulsar, pulsar_mod.Pulsar)
    """
    # get all attrs (methods/properties) of new class
    new_dir = dir(cls)
    # copy attrs that only exist in old class (e.g., class variables added
    # at run time)
    for k in dir(obj.__class__):
        if k not in new_dir:
            setattr(cls, k, getattr(obj.__class__, k))
    # change the obj to be an instance of the new class
    obj.__class__ = cls
    # copy the new methods
    for k in new_dir:  # for all attr names of the new class
        # get the attr by its name
        m = getattr(cls, k)
        # note that methods of an instance are functions when looking at the
        # class
        if not isinstance(m, types.FunctionType):
            continue  # Not a method (could be, e.g., a class constant)
        if not isinstance(inspect.getattr_static(cls, k), staticmethod):
            # if it is not a staticmethod: convert it to a method of obj
            m = types.MethodType(m, obj)
        # overwrite the method in the obj with the new version
        setattr(obj, k, m)
