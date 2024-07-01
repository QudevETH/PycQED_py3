import pytest
import numpy as np
import matplotlib.pyplot as plt
from pycqed.analysis.tools.plotting import SI_prefix_and_scale_factor
from pycqed.analysis.tools.plotting import set_xlabel, set_ylabel
from pycqed.analysis.tools.plotting import SI_val_to_msg_str


def test_non_SI():
    unit = 'arb.unit.'
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
    assert scale_factor == 1
    assert unit == post_unit

@pytest.mark.parametrize("val, expected_scale_factor, expected_post_unit", [
    (5, 1, ' V'),
    (5000, 1/1000, 'kV'),
    (0.05, 1000, 'mV')
])
def test_SI_scale_factors(val, expected_scale_factor, expected_post_unit):
    unit = 'V'
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=val, unit=unit)
    assert scale_factor == expected_scale_factor
    assert post_unit == expected_post_unit

def test_label_scaling():
    """
    This test creates a dummy plot and checks if the tick labels are
    rescaled correctly
    """
    f, ax = plt.subplots()
    x = np.linspace(-6, 6, 101)
    y = np.cos(x)
    ax.plot(x*1000, y/1e5)

    set_xlabel(ax, 'Distance', 'm')
    set_ylabel(ax, 'Amplitude', 'V')

    xlab = ax.get_xlabel()
    ylab = ax.get_ylabel()
    assert xlab == 'Distance (km)'
    assert ylab == 'Amplitude (μV)'

def test_SI_val_to_msg_str():
    val, unit = SI_val_to_msg_str(1030, 'm')
    assert val == str(1.03)
    assert unit == 'km'
