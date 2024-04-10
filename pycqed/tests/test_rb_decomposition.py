import unittest
import pycqed as pq
import numpy as np
import qutip as qtp
import os

from pycqed.measurement.randomized_benchmarking import two_qubit_clifford_group as tqc
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb


class Test_rb_decomposition(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.standard_pulses = {
            "I": qtp.qeye(2),
            "Z0": qtp.qeye(2),
            "X180": qtp.sigmax(),
            "mX180": qtp.sigmax(),
            "Y180": qtp.sigmay(),
            "mY180": qtp.sigmay(),
            "X90": qtp.rotation(qtp.sigmax(), np.pi / 2),
            "mX90": qtp.rotation(qtp.sigmax(), -np.pi / 2),
            "Y90": qtp.rotation(qtp.sigmay(), np.pi / 2),
            "mY90": qtp.rotation(qtp.sigmay(), -np.pi / 2),
            "Z90": qtp.rotation(qtp.sigmaz(), np.pi / 2),
            "mZ90": qtp.rotation(qtp.sigmaz(), -np.pi / 2),
            "Z180": qtp.sigmaz(),
            "mZ180": qtp.sigmaz(),
            "CZ": qtp.gates.cphase(np.pi),
        }

    def test_file_generation(self):
        filedir = os.path.join(
            pq.__path__[0],
            "measurement",
            "randomized_benchmarking",
            "clifford_hash_tables",
        )
        if "single_qubit_hash_lut.txt" in os.listdir(filedir):
            os.remove(os.path.join(filedir, "single_qubit_hash_lut.txt"))
        if "two_qubit_hash_lut.txt" in os.listdir(filedir):
            os.remove(os.path.join(filedir, "two_qubit_hash_lut.txt"))
        tqc.CLut.create_lut_files()
        self.assertIn("single_qubit_hash_lut.txt", os.listdir(filedir))
        self.assertIn("two_qubit_hash_lut.txt", os.listdir(filedir))

    def test_recovery_single_qubit_rb(self):
        cliffords = [0, 1, 50, 100]
        nr_seeds = 100

        for cl in cliffords:
            for _ in range(nr_seeds):
                cl_seq = rb.randomized_benchmarking_sequence(
                    cl, desired_net_cl=0, interleaved_gate=None
                )
                for decomp in ["HZ", "XY"]:
                    pulse_keys = rb.decompose_clifford_seq(cl_seq, gate_decomp=decomp)

                    gproduct = qtp.tensor(qtp.identity(2))
                    for pk in pulse_keys:
                        gproduct = self.standard_pulses[pk] * gproduct

                    x = gproduct.full() / gproduct.full()[0][0]
                    self.assertTrue(
                        np.all(
                            (
                                np.allclose(np.real(x), np.eye(2)),
                                np.allclose(np.imag(x), np.zeros(2)),
                            )
                        )
                    )

    def test_recovery_Y180_irb(self):
        cliffords = [0, 1, 50, 100]
        nr_seeds = 100

        for cl in cliffords:
            for _ in range(nr_seeds):
                cl_seq = rb.randomized_benchmarking_sequence(
                    cl, desired_net_cl=0, interleaved_gate="Y180"
                )
                for decomp in ["HZ", "XY"]:
                    pulse_keys = rb.decompose_clifford_seq(cl_seq, gate_decomp=decomp)

                    gproduct = qtp.tensor(qtp.identity(2))
                    for pk in pulse_keys:
                        gproduct = self.standard_pulses[pk] * gproduct

                    x = gproduct.full() / gproduct.full()[0][0]
                    self.assertTrue(
                        np.all(
                            (
                                np.allclose(np.real(x), np.eye(2)),
                                np.allclose(np.imag(x), np.zeros(2)),
                            )
                        )
                    )

    def test_recovery_two_qubit_rb(self):
        cliffords = [0, 1, 50]
        nr_seeds = 50

        for cl in cliffords:
            for _ in range(nr_seeds):
                cl_seq = rb.randomized_benchmarking_sequence_new(
                    cl,
                    number_of_qubits=2,
                    max_clifford_idx=11520,
                    interleaving_cl=None,
                    desired_net_cl=0,
                )
                for decomp in ["HZ", "XY"]:
                    tqc.gate_decomposition = rb.get_clifford_decomposition(decomp)
                    pulse_tuples_list_all = []
                    for i, idx in enumerate(cl_seq):
                        pulse_tuples_list = tqc.TwoQubitClifford(idx).gate_decomposition
                        pulse_tuples_list_all += pulse_tuples_list

                    gproduct = qtp.tensor(qtp.identity(2), qtp.identity(2))
                    for i, cl_tup in enumerate(pulse_tuples_list_all):
                        if cl_tup[0] == "CZ":
                            gproduct = self.standard_pulses[cl_tup[0]] * gproduct
                        else:
                            eye_2qb = [qtp.identity(2), qtp.identity(2)]
                            eye_2qb[int(cl_tup[1][-1])] = self.standard_pulses[
                                cl_tup[0]
                            ]
                            gproduct = qtp.tensor(eye_2qb) * gproduct

                    x = gproduct.full() / gproduct.full()[0][0]
                    self.assertTrue(
                        np.all(
                            (
                                np.allclose(np.real(x), np.eye(4)),
                                np.allclose(np.imag(x), np.zeros(4)),
                            )
                        )
                    )

    def test_recovery_cz_irb(self):
        cliffords = [0, 1, 50]
        nr_seeds = 50

        for cl in cliffords:
            for _ in range(nr_seeds):
                cl_seq = rb.randomized_benchmarking_sequence_new(
                    cl,
                    number_of_qubits=2,
                    max_clifford_idx=11520,
                    interleaving_cl=4368,
                    desired_net_cl=0,
                )
                for decomp in ["HZ", "XY"]:
                    tqc.gate_decomposition = rb.get_clifford_decomposition(decomp)
                    pulse_tuples_list_all = []
                    for i, idx in enumerate(cl_seq):
                        pulse_tuples_list = tqc.TwoQubitClifford(idx).gate_decomposition
                        pulse_tuples_list_all += pulse_tuples_list

                    gproduct = qtp.tensor(qtp.identity(2), qtp.identity(2))
                    for i, cl_tup in enumerate(pulse_tuples_list_all):
                        if cl_tup[0] == "CZ":
                            gproduct = self.standard_pulses[cl_tup[0]] * gproduct
                        else:
                            eye_2qb = [qtp.identity(2), qtp.identity(2)]
                            eye_2qb[int(cl_tup[1][-1])] = self.standard_pulses[
                                cl_tup[0]
                            ]
                            gproduct = qtp.tensor(eye_2qb) * gproduct

                    x = gproduct.full() / gproduct.full()[0][0]
                    self.assertTrue(
                        np.all(
                            (
                                np.allclose(np.real(x), np.eye(4)),
                                np.allclose(np.imag(x), np.zeros(4)),
                            )
                        )
                    )
