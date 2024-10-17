
Tips and Tricks for Analysis
================================

Hamiltonian Fitting
^^^^^^^^^^^^^^^^^^^

You can access the Transmon calculation methods (calculations based on Hamiltonian models stored in qb.fit_ge_freq_from_dc_offset) for a specific qubit even without running a virtual setup.

A quick example:

.. code-block:: python

    import pycqed.utilities.settings_manager as setman
    station = setman.get_station_from_file($timestamp$)
    print(station.qb1.calculate_frequency(bias=0.3))
    print(station.qb42.generate_scaled_volt_freq_conv())
