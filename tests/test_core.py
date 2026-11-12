import math
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baselines as bl  # noqa: E402
from grover import (diffuser, grover_adaptive_search, grover_circuit, marked_states,  # noqa: E402
                    phase_oracle, simulate_p_marked, theory_p_marked)
from plans import PlanProblem, bits_to_index, default_problem, index_to_bits  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    return default_problem()


def test_feasible_counts(problem):
    # C1 alone: sum_{j<=4} C(12, j) = 1 + 12 + 66 + 220 + 495 = 794
    assert int(problem.feasible_mask(use_c2=False).sum()) == 794 == sum(math.comb(12, j) for j in range(5))
    m = problem.feasible_mask()
    assert 0 < m.sum() < 794
    # every C1&C2-feasible plan really satisfies both constraints
    for i in np.flatnonzero(m)[:50]:
        assert problem.feasible(index_to_bits(i, 12))


def test_cost_matches_manual_calculation(problem):
    ph = problem.phantom
    x = np.zeros(12, dtype=int)
    x[[2, 3, 8, 9]] = 1                                   # 60, 90, 240, 270 degrees
    d = sum(ph.D[a] for a in (2, 3, 8, 9))                # explicit beam-by-beam sum
    manual = (1.0 * np.mean((d[ph.tumour] - 1.0) ** 2)
              + 1.0 * np.mean(d[ph.oar] ** 2) + 0.05 * np.mean(d[ph.normal] ** 2))
    assert problem.cost(x) == pytest.approx(manual, rel=1e-12)
    assert problem.costs()[bits_to_index(x)] == pytest.approx(manual, rel=1e-12)
    # one beam gives 0.25 at the isocentre
    iso = np.flatnonzero((ph.x == 0) & (ph.y == 0))[0]
    assert np.allclose(ph.D[:, iso], 0.25)


def test_bit_order_is_qiskit_little_endian(problem):
    assert bits_to_index([1] + [0] * 11) == 1                 # qubit 0 <-> beam 0
    assert list(index_to_bits(0b100, 12)[:3]) == [0, 0, 1]
    qc = QuantumCircuit(12)
    qc.x(3)                                                   # beam 3 on
    assert int(np.argmax(np.abs(Statevector(qc).data))) == 8  # |..001000> = index 8


def test_unique_optimum_and_brute_force(problem):
    opt = problem.optimum()
    assert opt["unique"]
    assert bl.brute_force(problem)["index"] == opt["index"]
    assert bl.brute_force(problem, feasible_only=True)["index"] == opt["index"]
    assert bl.nnls_bound(problem)["bound"] <= opt["cost"] + 1e-12


def test_oracle_marks_exactly_the_right_states(problem):
    y = float(np.sort(problem.costs()[problem.feasible_mask()])[5])
    marked = marked_states(problem, y)
    assert marked.sum() == 5
    qc = QuantumCircuit(12)
    qc.h(range(12))
    before = Statevector(qc).data
    qc.append(phase_oracle(marked), range(12))
    after = Statevector(qc).data
    ratio = np.real(after / before)
    assert np.allclose(ratio[marked], -1) and np.allclose(ratio[~marked], 1)
    # cross-check against the definition: feasible AND F < y
    F, feas = problem.costs(), problem.feasible_mask()
    for i in np.flatnonzero(marked):
        assert feas[i] and F[i] < y


def test_one_grover_iteration_matches_theory(problem):
    y = float(np.sort(problem.costs()[problem.feasible_mask()])[4])
    marked = marked_states(problem, y)
    M, N = int(marked.sum()), 4096
    p = simulate_p_marked(12, marked, [0, 1, 2])
    assert p == pytest.approx(theory_p_marked(M, N, [0, 1, 2]), abs=1e-10)
    assert p[1] == pytest.approx(math.sin(3 * math.asin(math.sqrt(M / N))) ** 2, abs=1e-10)


def test_diffuser_is_inversion_about_mean():
    n = 4
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    out = Statevector(psi).evolve(diffuser(n)).data
    expected = -(2 * psi.mean() - psi)                   # -(2|s><s| - I) psi
    assert np.allclose(out, expected)


def test_gas_finds_brute_force_optimum_seeded(problem):
    res = grover_adaptive_search(problem, seed=1000)
    assert res["found_opt"]
    assert res["best"] == bl.brute_force(problem)["index"]
    assert res["oracle_calls"] <= 30 * 64
    again = grover_adaptive_search(problem, seed=1000)    # reproducible
    assert again["history"] == res["history"]


def test_grover_circuit_measures_plans(problem):
    qc = grover_circuit(12, marked_states(problem, 1.0), 2)
    assert qc.num_qubits == 12 and qc.count_ops()["measure"] == 12
