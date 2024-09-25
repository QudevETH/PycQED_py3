import pytest
import matplotlib.pyplot as plt
import numpy as np
import pycqed.analysis_v2.plotting.aggregation as plta
import pycqed.analysis.analysis_toolbox as a_tools
import pycqed.measurement.quantum_experiment as qe_mod

TEST_DATA_DIR = r'Q:\USERS\nathan\data\xld'
a_tools.datadir = TEST_DATA_DIR


@pytest.fixture
def sample_data():
    return {
        (0, 0): np.random.rand(10),
        (0, 1): np.random.rand(10),
        (1, 0): np.random.rand(10),
        (1, 1): np.random.rand(10),
    }
@pytest.fixture
def sample_qubit_data():
    return {
        'qb1': np.random.rand(10),
        'qb2': np.random.rand(10),
        'qb3': np.random.rand(10),
        'qb4': np.random.rand(10),
    }

@pytest.fixture
def sample_pair_data():
    def decaying_exponential(size, tau):
        """Generate a decaying exponential dataset with a specific tau."""
        return np.exp(-np.linspace(0, tau, size))

    return {
        ('qb1', 'qb2'): decaying_exponential(100, tau=1),
        ('qb1', 'qb3'): decaying_exponential(100, tau=2),
        ('qb4', 'qb2'): decaying_exponential(100, tau=3),
        ('qb4', 'qb3'): decaying_exponential(100, tau=4),
        ('qb4', 'qb5'): decaying_exponential(100, tau=5),
        ('qb4', 'qb9'): decaying_exponential(100, tau=6),
        ('qb6', 'qb5'): decaying_exponential(100, tau=7),
        ('qb6', 'qb11'): decaying_exponential(10, tau=8),
        ('qb8', 'qb3'): decaying_exponential(100, tau=9),
        ('qb8', 'qb7'): decaying_exponential(100, tau=10),
        ('qb8', 'qb9'): decaying_exponential(100, tau=11),
        ('qb8', 'qb13'): decaying_exponential(100, tau=12),
        ('qb10', 'qb5'): decaying_exponential(100, tau=13),
        ('qb10', 'qb9'): decaying_exponential(100, tau=14),
        ('qb10', 'qb11'): decaying_exponential(100, tau=15),
        ('qb10', 'qb15'): decaying_exponential(100, tau=16),
        ('qb12', 'qb7'): decaying_exponential(100, tau=17),
        ('qb12', 'qb13'): decaying_exponential(100, tau=18),
        ('qb14', 'qb9'): decaying_exponential(100, tau=19),
        ('qb14', 'qb13'): decaying_exponential(100, tau=20),
        ('qb14', 'qb15'): decaying_exponential(100, tau=21),
        ('qb14', 'qb16'): decaying_exponential(100, tau=22),
        ('qb17', 'qb15'): decaying_exponential(100, tau=23),
        ('qb17', 'qb16'): decaying_exponential(100, tau=24),
    }

def test_plot_on_grid(sample_data):
    def plot_func(ax, data):
        ax.plot(data)
    fig, axes = plta.plot_on_grid(sample_data, plot_func)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 2)

@pytest.mark.parametrize('fig_info', [dict(qb1=dict(timestamp='20240606_000101',
                                                    fig_name=f'Rabi_qb1'),
                                           qb2=dict(timestamp='20240606_000101',
                                                    fig_name=f'Rabi_qb2')),
                                    {'qb1': {'timestamp': '20240813_005423'},
                                     'qb2': {'timestamp': '20240813_005844'},
                                     'qb3': {'timestamp': '20240813_010311'},
                                     'qb4': {'timestamp': '20240813_010745'},
                                     'qb5': {'timestamp': '20240813_011222'},
                                     'qb6': {'timestamp': '20240813_011700'},
                                     'qb7': {'timestamp': '20240813_012135'},
                                     'qb8': {'timestamp': '20240813_012608'},
                                     'qb9': {'timestamp': '20240813_013046'},
                                     'qb10': {'timestamp': '20240813_013518'},
                                     'qb11': {'timestamp': '20240813_013942'},
                                     'qb12': {'timestamp': '20240813_014413'},
                                     'qb13': {'timestamp': '20240813_014858'},
                                     'qb14': {'timestamp': '20240813_015344'},
                                     'qb15': {'timestamp': '20240813_015826'},
                                     'qb16': {'timestamp': '20240813_020301'},
                                     'qb17': {'timestamp': '20240813_020742'}}])
def test_plot_on_qubit_grid(fig_info):
    plta.plot_on_qubit_grid(fig_info, plta.fig_from_measurement_plot_func)


def test_plot_on_pair_grid(sample_pair_data):
    def plot_func(ax, data):
        ax.plot(data)

    fig, axes = plta.plot_on_pair_grid(sample_pair_data, plot_func)


def test_get_qubit_grid():
    qubits = ['qb1', 'qb2', 'qb3', 'qb4']
    fig, axes = plta.get_qubit_grid(qubits)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 2)

@pytest.mark.parametrize('timestamps', (['20240606_000101'], # rabi 9 qubits single file
                                        ))
def test_calibration_plot_aggregator_from_timestamps(timestamps):
    aggregator = plta.CalibrationPlotAggregator.from_timestamps(timestamps)
    fig, axes = aggregator.plot_on_qubit_grid()

@pytest.mark.parametrize('timestamps', (['20240606_000101'], # rabi 9 qubits single file
                                        ))
def test_calibration_plot_aggregator_from_quantum_experiments(timestamps):
    qes = [qe_mod.QuantumExperiment() for _ in range(len(timestamps))]
    for qe, t in zip(qes, timestamps):
        qe.timestamp = t
    aggregator = plta.CalibrationPlotAggregator.from_quantum_experiments(qes)
    fig, axes = aggregator.plot_on_qubit_grid()

    assert len(aggregator.fig_info) > 0
