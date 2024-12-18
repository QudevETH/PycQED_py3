import pytest

import pycqed.measurement.waveform_control.pulse_library as pl_mod
import pycqed.measurement.waveform_control.circuit_builder as cb_mod


@pytest.mark.skip(reason="FIXME: Move to integration tests")
@pytest.fixture(scope="module", autouse=True)
def virtual_setup():
    from pycqedscripts.init.xld.ATC264_M191_S17V3P4_A1 import initialize_setup
    init_dict = initialize_setup(
        virtual_setup=True,
        # datadir="pydata",
    )
    globals().update(**init_dict)

def test_get_pulses():
    cb = cb_mod.CircuitBuilder(
        dev=dev,
        qubits=qubits,
        cz_pulse_name='CZ_nztc',
        decompose_rotation_gates={'CZ_nztc': [['qb4', 'qb3']]},
    )

    op_dicts = []
    op_dicts += cb.get_pulses(f'X qb2')
    op_dicts += cb.get_pulses(f'X 1')
    op_dicts += cb.get_pulses(f'mX10 qb4')
    op_dicts += cb.get_pulses(f'X10 3')
    op_dicts += cb.get_pulses(f'X:[theta]+90 3')
    op_dicts += cb.get_pulses(f'mZ:[theta]+90 2 3')
    op_dicts += cb.get_pulses(f'CZ qb4 qb3')
    op_dicts += cb.get_pulses(f'CZ qb3 qb4')
    op_dicts += cb.get_pulses(f'CZ 3 2')
    op_dicts += cb.get_pulses(f'CZ 2 3')
    op_dicts += cb.get_pulses(f'CZ_up qb4 qb3')
    op_dicts += cb.get_pulses(f'CZ_up qb3 qb4')
    op_dicts += cb.get_pulses(f'CZ_up 3 2')
    op_dicts += cb.get_pulses(f'CZ_up 2 3')
    op_dicts += cb.get_pulses(f'CZ_nztc qb4 qb3')
    op_dicts += cb.get_pulses(f'CZ_nztc qb3 qb4')
    op_dicts += cb.get_pulses(f'CZ_nztc 3 2')
    op_dicts += cb.get_pulses(f'CZ_nztc 2 3')
    op_dicts += cb.get_pulses(f'CZ_nztc42.123 qb4 qb5')  # not decomposed
    op_dicts += cb.get_pulses(f'CZ_nztc42.123 qb4 qb3')  # decomposed
    ## Not implemented:
    # op_dicts += cb.get_pulses(f'mCZ:[theta]+90 2 3')
    # op_dicts += cb.get_pulses('CZ:2*[theta] qb3 qb4')
