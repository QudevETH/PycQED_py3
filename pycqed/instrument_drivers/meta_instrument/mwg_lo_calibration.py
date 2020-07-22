import scipy as sp

from qcodes import validators as vals
from qcodes.instrument.parameter import ManualParameter


def mwg_with_lo_calibration_template(mwg_class):
    """
    A class decorator that adds dynamic parameter update functionality to
    an existing microwave generator class. Useful for example to update the
    LO-leakage calibration as the frequency is swept.

    The derived class will have two new parameters: `lo_cal_data` and
    `lo_cal_interp_kind`. The structure of the `lo_cal_data` should be
    the following:
        {name: (param, freqs, cal_vals)}
    Here name is a user-defined label for the parameter, param is a qcodes
    parameter to be updated, and freqs and cal_cals are lists of frequency and
    parameter values that will be used for the parameter value interpolation.
    The parameter `lo_cal_interp_kind` can be used to switch between different
    interpolation functions.

    Args:
        mwg_class: A microwave generator driver class that will be updated to
            change extra parameters whenever the frequency is changed.
    Returns:
        A class that inherits from mwg_class with the added functionality.
    """

    class MWGWithLOCalibration(mwg_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.add_parameter('lo_cal_data',
                               vals=vals.Dict(),
                               parameter_class=ManualParameter,
                               initial_value=dict())
            self.add_parameter('lo_cal_interp_kind',
                               vals=vals.Enum(
                                   'linear', 'nearest', 'zero', 'slinear',
                                   'quadratic', 'cubic', 'previous', 'next'),
                               parameter_class=ManualParameter,
                               initial_value='linear')

            self.frequency.set_parser = \
                lambda val, lo_cal_data=self.lo_cal_data, \
                       lo_cal_interp_kind=self.lo_cal_interp_kind: \
                   self.lo_calib(val, lo_cal_data, lo_cal_interp_kind)

        @staticmethod
        def lo_calib(val, lo_cal_data, lo_cal_interp_kind):
            for par, freqs, cal_vals in lo_cal_data().values():
                par(float(sp.interpolate.interp1d(
                    freqs, cal_vals, kind=lo_cal_interp_kind(),
                    fill_value=(min(cal_vals), max(cal_vals)))(val)))
            return val

    return MWGWithLOCalibration
