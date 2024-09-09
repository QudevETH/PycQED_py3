import os
import tempfile

import numpy
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_base_instrument as zibi
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_PQSC as PQ
import pytest


@pytest.fixture(scope="class")
def pqsc():
    print("Connecting...")
    pqsc = PQ.ZI_PQSC(
        name="MOCK_PQSC", server="emulator", device="dev0000", interface="1GbE"
    )
    yield pqsc
    print("Disconnecting...")
    pqsc.close()


@pytest.mark.hardware
def test_instantiation(pqsc):
    assert pqsc.devname == "dev0000"
