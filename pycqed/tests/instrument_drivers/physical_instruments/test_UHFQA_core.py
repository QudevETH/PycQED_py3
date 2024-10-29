import contextlib
import io

import numpy as np
import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQA_core as UHF
import pytest


@pytest.mark.hardware
class Test_UHFQA_core:
    # FIXME: Mocking fails right now, why?
    @classmethod
    def setup_class(cls):
        cls.uhf = UHF.UHFQA_core(
            name="MOCK_UHF", server="emulator", device="dev2109", interface="1GbE"
        )

        cls.uhf.reset_waveforms_zeros()

    @classmethod
    def teardown_class(cls):
        cls.uhf.close()

    def test_instantiation(self):
        assert self.uhf.devname == "dev2109"

    def test_assure_ext_clock(self):
        self.uhf.assure_ext_clock()
        assert self.uhf.system_extclk() == 1

    def test_clock_freq(self):
        assert self.uhf.clock_freq() == 1.8e9

    def test_load_default_settings(self):
        self.uhf.load_default_settings()
        assert self.uhf.download_crosstalk_matrix().tolist() == np.eye(10).tolist()

    def test_print_overview(self, capsys):
        self.uhf.print_overview()
        captured = capsys.readouterr()
        assert "Crosstalk overview" in captured.out

    def test_print_correlation_overview(self, capsys):
        self.uhf.print_correlation_overview()
        captured = capsys.readouterr()
        assert "Correlations overview" in captured.out

    def test_print_deskew_overview(self, capsys):
        self.uhf.print_deskew_overview()
        captured = capsys.readouterr()
        assert "Deskew overview" in captured.out

    def test_print_crosstalk_overview(self, capsys):
        self.uhf.print_crosstalk_overview()
        captured = capsys.readouterr()
        assert "Crosstalk overview" in captured.out

    def test_print_integration_overview(self, capsys):
        self.uhf.print_integration_overview()
        captured = capsys.readouterr()
        assert "Integration overview" in captured.out

    def test_print_rotations_overview(self, capsys):
        self.uhf.print_rotations_overview()
        captured = capsys.readouterr()
        assert "Rotations overview" in captured.out

    def test_print_thresholds_overview(self, capsys):
        self.uhf.print_thresholds_overview()
        captured = capsys.readouterr()
        assert "Thresholds overview" in captured.out

    def test_print_user_regs_overview(self, capsys):
        self.uhf.print_user_regs_overview()
        captured = capsys.readouterr()
        assert "User registers overview" in captured.out


@pytest.mark.hardware
class TestUHFQA:
    def test_minimum_holdoff(self, uhf):
        # Test without averaging
        uhf.qas_0_integration_length(128)
        uhf.qas_0_result_averages(1)
        uhf.qas_0_delay(0)
        assert uhf.minimum_holdoff() == 800 / 1.8e9
        uhf.qas_0_delay(896)
        assert uhf.minimum_holdoff() == (896 + 16) / 1.8e9
        uhf.qas_0_integration_length(2048)
        assert uhf.minimum_holdoff() == (2048) / 1.8e9

        # Test with averaging
        uhf.qas_0_result_averages(16)
        uhf.qas_0_delay(0)
        uhf.qas_0_integration_length(128)
        assert uhf.minimum_holdoff() == 2560 / 1.8e9
        uhf.qas_0_delay(896)
        assert uhf.minimum_holdoff() == 2560 / 1.8e9
        uhf.qas_0_integration_length(4096)
        assert uhf.minimum_holdoff() == 4096 / 1.8e9

    def test_crosstalk_matrix(self, uhf):
        mat = np.random.random((10, 10))
        uhf.upload_crosstalk_matrix(mat)
        new_mat = uhf.download_crosstalk_matrix()
        assert np.allclose(mat, new_mat)

    def test_reset_crosstalk_matrix(self, uhf):
        mat = np.random.random((10, 10))
        uhf.upload_crosstalk_matrix(mat)
        uhf.reset_crosstalk_matrix()
        reset_mat = uhf.download_crosstalk_matrix()
        assert np.allclose(np.eye(10), reset_mat)

    def test_reset_acquisition_params(self, uhf):
        for i in range(16):
            uhf.set(f"awgs_0_userregs_{i}", i)

        uhf.reset_acquisition_params()
        values = [uhf.get(f"awgs_0_userregs_{i}") for i in range(16)]
        assert values == [0] * 16

    def test_correlation_settings(self, uhf):
        uhf.qas_0_correlations_5_enable(1)
        uhf.qas_0_correlations_5_source(3)

        assert uhf.qas_0_correlations_5_enable() == 1
        assert uhf.qas_0_correlations_5_source() == 3

    def test_thresholds_correlation_settings(self, uhf):
        uhf.qas_0_thresholds_5_correlation_enable(1)
        uhf.qas_0_thresholds_5_correlation_source(3)

        assert uhf.qas_0_thresholds_5_correlation_enable() == 1
        assert uhf.qas_0_thresholds_5_correlation_source() == 3

    def test_reset_correlation_settings(self, uhf):
        uhf.qas_0_correlations_5_enable(1)
        uhf.qas_0_correlations_5_source(3)
        uhf.qas_0_thresholds_5_correlation_enable(1)
        uhf.qas_0_thresholds_5_correlation_source(3)

        uhf.reset_correlation_params()

        assert uhf.qas_0_correlations_5_enable() == 0
        assert uhf.qas_0_correlations_5_source() == 0
        assert uhf.qas_0_thresholds_5_correlation_enable() == 0
        assert uhf.qas_0_thresholds_5_correlation_source() == 0

    def test_reset_rotation_params(self, uhf):
        uhf.qas_0_rotations_3(1 - 1j)
        assert uhf.qas_0_rotations_3() == (1 - 1j)
        uhf.reset_rotation_params()
        assert uhf.qas_0_rotations_3() == (1 + 1j)

    def test_start(self, uhf):
        uhf.start()
