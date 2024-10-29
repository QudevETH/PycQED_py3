import os

import h5py
import numpy as np
import pycqed.utilities.general as gen
import pytest
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.instrument_drivers.physical_instruments.dummy_instruments import (
    DummyParHolder,
)
from pycqed.measurement import measurement_control
from pycqed.utilities.io import hdf5 as h5d
from qcodes import station

# FIXME: Disabled file in pytest


@pytest.fixture
def setup_station(request):
    def setup():
        station_obj = station.Station()
        datadir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
        )
        print(f"datadir: {datadir}")
        MC = measurement_control.MeasurementControl(
            "MC", live_plot_enabled=False, verbose=False
        )
        MC.station = station_obj
        MC.datadir(datadir)
        a_tools.datadir = datadir
        station_obj.add_component(MC)

        mock_parabola = DummyParHolder("mock_parabola")
        station_obj.add_component(mock_parabola)
        mock_parabola_2 = DummyParHolder("mock_parabola_2")
        station_obj.add_component(mock_parabola_2)

        return station_obj, MC, mock_parabola, mock_parabola_2, datadir

    def teardown(components):
        station_obj, MC, mock_parabola, mock_parabola_2, _ = components
        MC.close()
        mock_parabola.close()
        mock_parabola_2.close()

    components = setup()
    request.addfinalizer(lambda: teardown(components))
    return components


@pytest.mark.skip(reason="FIXME: Entire test file disabled - need to talk to Jakob")
def test_storing_and_loading_station_snapshot(setup_station):
    station_obj, _, mock_parabola_2, _, datadir = setup_station

    print(f"datadir: {datadir}")

    mock_parabola_2.x(1)
    mock_parabola_2.y(2.245)
    mock_parabola_2.array_like(np.linspace(0, 11, 23))

    snap = station_obj.snapshot(update=True)
    data_object = h5d.Data(name="test_object_snap", datadir=datadir)
    h5d.write_dict_to_hdf5(snap, data_object)
    data_object.close()

    filepath = data_object.filepath
    assert os.path.isfile(filepath), f"File {filepath} does not exist"

    new_dict = {}
    with h5py.File(filepath, "r") as opened_hdf5_file:
        h5d.read_dict_from_hdf5(new_dict, opened_hdf5_file)

        assert snap.keys() == new_dict.keys()
        assert snap["instruments"].keys() == new_dict["instruments"].keys()
        mock_parab_pars = snap["instruments"]["mock_parabola_2"]["parameters"]

        assert mock_parab_pars["x"]["value"] == 1
        assert mock_parab_pars["y"]["value"] == 2.245
        np.testing.assert_array_equal(
            mock_parab_pars["array_like"]["value"], np.linspace(0, 11, 23)
        )


@pytest.mark.skip(reason="FIXME: Entire test file disabled - need to talk to Jakob")
def test_writing_and_reading_dicts_to_hdf5(setup_station):
    _, _, _, _, datadir = setup_station

    test_dict = {
        "list_of_ints": list(np.arange(5)),
        "list_of_floats": list(np.arange(5.1)),
        "some_bool": True,
        "weird_dict": {"a": 5},
        "dataset1": np.linspace(0, 20, 31),
        "dataset2": np.array([[2, 3, 4, 5], [2, 3, 1, 2]]),
        "list_of_mixed_type": ["hello", 4, 4.2, {"a": 5}, [4, 3]],
        "tuple_of_mixed_type": tuple(["hello", 4, 4.2, {"a": 5}, [4, 3]]),
        "a list of strings": ["my ", "name ", "is ", "earl."],
        "some_np_bool": bool(True),
        "list_of_dicts": [{"a": 5}, {"b": 3}],
        "some_int": 3,
        "some_float": 3.5,
        "some_np_int": int(3),
        "some_np_float": float(3.5),
    }

    data_object = h5d.Data(name="test_object", datadir=datadir)
    h5d.write_dict_to_hdf5(test_dict, data_object)
    data_object.close()

    filepath = data_object.filepath
    assert os.path.exists(filepath), f"File does not exist: {filepath}"

    new_dict = {}
    with h5py.File(filepath, "r") as opened_hdf5_file:
        h5d.read_dict_from_hdf5(new_dict, opened_hdf5_file)

        assert test_dict.keys() == new_dict.keys()
        assert test_dict["list_of_ints"] == new_dict["list_of_ints"]
        assert test_dict["list_of_floats"] == new_dict["list_of_floats"]
        assert test_dict["weird_dict"] == new_dict["weird_dict"]
        assert test_dict["some_bool"] == new_dict["some_bool"]
        assert test_dict["list_of_dicts"] == new_dict["list_of_dicts"]
        assert test_dict["list_of_mixed_type"] == new_dict["list_of_mixed_type"]
        assert test_dict["list_of_mixed_type"][0] == new_dict["list_of_mixed_type"][0]
        assert test_dict["list_of_mixed_type"][2] == new_dict["list_of_mixed_type"][2]
        assert test_dict["tuple_of_mixed_type"] == new_dict["tuple_of_mixed_type"]
        assert isinstance(test_dict["tuple_of_mixed_type"], tuple) and isinstance(
            new_dict["tuple_of_mixed_type"], tuple
        )
        assert test_dict["tuple_of_mixed_type"][0] == new_dict["tuple_of_mixed_type"][0]
        assert test_dict["tuple_of_mixed_type"][2] == new_dict["tuple_of_mixed_type"][2]
        assert test_dict["some_np_bool"] == new_dict["some_np_bool"]
        assert test_dict["some_int"] == new_dict["some_int"]
        assert test_dict["some_np_float"] == new_dict["some_np_float"]
        assert test_dict["a list of strings"] == new_dict["a list of strings"]
        assert test_dict["a list of strings"][0] == new_dict["a list of strings"][0]


@pytest.mark.skip(reason="FIXME: Entire test file disabled - need to talk to Jakob")
def test_loading_settings_onto_instrument(setup_station):
    _, MC, mock_parabola, mock_parabola_2, _ = setup_station

    arr = np.linspace(12, 42, 11)
    mock_parabola.array_like(arr)
    mock_parabola.x(42.23)
    mock_parabola.y(2)
    mock_parabola.status(True)
    mock_parabola.dict_like({"a": {"b": [2, 3, 5]}})

    MC.set_sweep_function(mock_parabola.x)
    MC.set_sweep_points([0, 1])
    MC.set_detector_function(mock_parabola.skewed_parabola)
    MC.run("test_MC_snapshot_storing")
    mock_parabola.array_like(arr + 5)
    mock_parabola.x(13)

    np.testing.assert_array_equal(mock_parabola.array_like(), arr + 5)
    assert mock_parabola.x() == 13

    gen.load_settings_onto_instrument_v2(
        mock_parabola, label="test_MC_snapshot_storing"
    )

    np.testing.assert_array_equal(mock_parabola.array_like(), arr)
    assert mock_parabola.x() == 42.23

    gen.load_settings_onto_instrument_v2(
        mock_parabola_2,
        load_from_instr=mock_parabola.name,
        label="test_MC_snapshot_storing",
    )
    assert mock_parabola_2.y() == 2
    assert mock_parabola_2.status()
    assert mock_parabola_2.dict_like() == {"a": {"b": [2, 3, 5]}}
