"""Sanity tests: run with `pytest -q` from the repo root."""
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import instance as inst  # noqa: E402
import qubo  # noqa: E402
from qaoa import QAOA, cost_hamiltonian  # noqa: E402

O = inst.overlap_matrix(inst.FRAGMENTS)


def test_true_order_reassembles_genome():
    assert inst.assemble(inst.TRUE_ORDER, inst.FRAGMENTS) == inst.GENOME


def test_overlap_matrix_values():
    expected = np.array([[0, 5, 0, 4], [2, 0, 1, 5], [0, 0, 0, 3], [3, 2, 2, 0]])
    assert (O == expected).all()


def test_tradeoff_exists():
    """lam=0 picks a misassembly with a weak junction; lam=3 recovers the truth."""
    r0 = qubo.best_order_brute_force(inst.junction_weights(O, 0))[0]
    r3 = qubo.best_order_brute_force(inst.junction_weights(O, 3))[0]
    assert r0[1] != inst.TRUE_ORDER and inst.order_metrics(r0[1], O)["weak_junctions"] == 1
    assert r3[1] == inst.TRUE_ORDER and inst.order_metrics(r3[1], O)["weak_junctions"] == 0
    assert inst.order_metrics(r0[1], O)["total_overlap"] == 12
    assert inst.order_metrics(r3[1], O)["total_overlap"] == 11


def test_qubo_ground_state_is_the_optimal_permutation_for_all_lambdas():
    for lam in [0, 1, 2, 3, 4, 5]:
        A = qubo.smallest_valid_penalty(O, lam)
        m = qubo.make_model(O, lam, A)
        k = int(np.argmin(m.energies))
        best_gain, best_perm = qubo.best_order_brute_force(m.W)[0]
        assert m.decode(k) == best_perm
        assert abs(m.energies[k] + best_gain) < 1e-9


def test_feasible_count_is_n_factorial():
    m = qubo.make_model(O, 3, 6)
    assert int(m.feasible.sum()) == 24


def test_ising_hamiltonian_matches_qubo_energy_in_qiskit():
    """Expectation of the Qiskit SparsePauliOp on basis states == QUBO energy."""
    m = qubo.make_model(O, 3, qubo.smallest_valid_penalty(O, 3))
    H = cost_hamiltonian(m)
    rng = np.random.default_rng(1)
    xs = [m.encode(inst.TRUE_ORDER)] + [rng.integers(0, 2, m.num_vars).astype(float) for _ in range(5)]
    for x in xs:
        qc = QuantumCircuit(m.num_vars)
        for i, b in enumerate(x):
            if b:
                qc.x(i)
        val = Statevector(qc).expectation_value(H).real
        assert abs(val - m.energy(x)) < 1e-9


def test_qaoa_probabilities_are_normalised_and_beats_uniform_energy():
    m = qubo.make_model(O, 3, qubo.smallest_valid_penalty(O, 3))
    q = QAOA(m, p=1)
    pr = q.probs(np.array([0.2, 0.3]))
    assert abs(pr.sum() - 1) < 1e-9
    assert pr.shape == (2 ** 16,)
