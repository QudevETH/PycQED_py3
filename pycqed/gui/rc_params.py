import matplotlib

def gui_rc_params():
    """Returns matplotlib rc parameter default values for the PycQED GUI"""
    gui_rc_params = matplotlib.rc_params()
    gui_rc_params.update({
        'axes.titlesize': 'medium',
        'figure.autolayout': True,
        'legend.fontsize': 'small',
        'legend.framealpha': 0.5,
        'lines.linewidth': 1.0,
    })
    return gui_rc_params
