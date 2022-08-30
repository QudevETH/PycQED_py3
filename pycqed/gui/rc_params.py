import matplotlib

def gui_rc_params():
    """Returns matplotlib rc parameter default values for the PycQED GUI"""
    GUI_RC_PARAMS = matplotlib.rc_params()
    GUI_RC_PARAMS.update({
        'axes.titlesize': 'medium',
        'figure.autolayout': True,
        'legend.fontsize': 'small',
        'legend.framealpha': 0.5,
        'lines.linewidth': 1.0,
    })
    return GUI_RC_PARAMS
