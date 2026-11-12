"""Grover Adaptive Search (GAS) for beam-angle selection, written from scratch in Qiskit.

Algorithm (credits in README): Durr & Hoyer (1996) minimum finding, driven by the
Boyer-Brassard-Hoyer-Tapp (1998) random-iteration schedule; constrained-cost oracle in
the spirit of Gilliam, Woerner & Gonciulea (2021). No qiskit_optimization code is used.

Qubit a <-> beam a; basis state |i> <-> plan with bit a of i = x_a (Qiskit little-endian).

O_y   : |x> -> -|x>  iff  x is feasible (C1 and C2)  AND  F(x) < y        (phase oracle)
W     : H^n X^n (multi-controlled Z) X^n H^n  = -(2|s><s| - I)              (diffuser)
G     : W O_y                                                            (one Grover iteration)
"""
from __future__ import annotations

import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

from plans import PlanProblem

LAMBDA = 6 / 5


def default_patience(N: int, lam: float = LAMBDA) -> int:
    """Consecutive non-improving rounds before GAS stops: enough rounds for the BBHT range m
    to grow from 1 to its sqrt(N) cap (ceil(log_lam sqrt N)) plus 2 rounds at the cap.
    N = 4096 -> 25. (The often-quoted 15 stops while m <= 15 and misses the optimum; see ablation.)"""
    return math.ceil(math.log(math.sqrt(N)) / math.log(lam)) + 2


# ---------------------------------------------------------------- circuit pieces
def marked_states(problem: PlanProblem, y: float) -> np.ndarray:
    """Boolean truth table of the oracle: feasible AND F(x) < y."""
    return problem.feasible_mask() & (problem.costs() < y)


def phase_oracle(marked: np.ndarray) -> DiagonalGate:
    """Diagonal +-1 phase oracle from a truth table (entry i <-> basis state |i>)."""
    diag = np.where(marked, -1.0, 1.0)
    gate = DiagonalGate(diag.tolist())
    gate.label = "O_y"
    return gate


def diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.mcp(math.pi, list(range(n - 1)), n - 1)  # multi-controlled Z
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(n: int, marked: np.ndarray, r: int, measure: bool = True,
                   oracle: DiagonalGate | None = None) -> QuantumCircuit:
    """H^n followed by r Grover iterations (oracle + diffuser), optionally measured."""
    oracle = oracle if oracle is not None else phase_oracle(marked)
    diff = diffuser(n)
    qc = QuantumCircuit(n, name=f"Grover_r{r}")
    qc.h(range(n))
    for _ in range(r):
        qc.append(oracle, range(n))
        qc.compose(diff, range(n), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def theory_p_marked(M: int, N: int, r) -> np.ndarray:
    """P(measure a marked state) after r iterations: sin^2((2r+1) theta), sin^2 theta = M/N."""
    theta = math.asin(math.sqrt(M / N))
    return np.sin((2 * np.asarray(r) + 1) * theta) ** 2


def simulate_p_marked(n: int, marked: np.ndarray, r_values, sim: AerSimulator | None = None) -> np.ndarray:
    """Exact P(marked) from the Aer statevector after r iterations (no sampling noise)."""
    sim = sim or AerSimulator(method="statevector")
    oracle = phase_oracle(marked)
    circs = []
    for r in r_values:
        qc = grover_circuit(n, marked, int(r), measure=False, oracle=oracle)
        qc.save_statevector()
        circs.append(qc)
    res = sim.run(circs).result()
    return np.array([float(np.sum(np.abs(np.asarray(res.get_statevector(i))) ** 2 * marked))
                     for i in range(len(circs))])


# ---------------------------------------------------------------- GAS
def grover_adaptive_search(problem: PlanProblem, seed: int, sim: AerSimulator | None = None,
                           budget: int | None = None, patience: int | None = None,
                           lam: float = LAMBDA) -> dict:
    """One seeded GAS run. Every measurement is a 1-shot AerSimulator execution.

    Counts: oracle_calls = total Grover iterations (each applies O_y once);
            evaluations  = classical F(x) evaluations (1 initial + 1 per measurement).
    """
    rng = np.random.default_rng(seed)
    sim = sim or AerSimulator(method="statevector")
    n, N = problem.n, problem.N
    F, feas = problem.costs(), problem.feasible_mask()
    opt = problem.optimum()["index"]
    budget = budget if budget is not None else int(30 * math.sqrt(N))
    m_max = math.sqrt(N)
    patience = default_patience(N, lam) if patience is None else patience

    best = int(rng.choice(np.flatnonzero(feas)))       # random feasible starting plan
    y = float(F[best])
    calls, evals = 0, 1                                # the start plan costs one evaluation
    m, fails, rounds = 1.0, 0, 0
    calls_to_opt = 0 if best == opt else None
    evals_to_opt = 1 if best == opt else None
    hist = [dict(round=0, r=0, calls=0, y=y, x=best, improved=True)]
    oracle, oracle_y = None, None

    while calls < budget and fails < patience:
        r = int(rng.integers(0, math.ceil(m)))         # uniform in [0, ceil(m) - 1]
        r = min(r, budget - calls)
        if oracle_y != y:                              # rebuild O_y only when y changes
            marked = feas & (F < y)
            oracle, oracle_y = phase_oracle(marked), y
        qc = grover_circuit(n, marked, r, oracle=oracle)
        counts = sim.run(qc, shots=1, seed_simulator=int(rng.integers(2**31 - 1))).result().get_counts()
        x = int(next(iter(counts)).replace(" ", ""), 2)
        calls += r
        evals += 1
        rounds += 1
        improved = bool(feas[x] and F[x] < y)
        if improved:
            y, best, m, fails = float(F[x]), x, 1.0, 0
        else:
            m, fails = min(lam * m, m_max), fails + 1
        if best == opt and calls_to_opt is None:
            calls_to_opt, evals_to_opt = calls, evals
        hist.append(dict(round=rounds, r=r, calls=calls, y=y, x=x, improved=improved))

    return dict(seed=seed, best=best, best_cost=y, found_opt=bool(best == opt), patience=patience,
                oracle_calls=calls, evaluations=evals, rounds=rounds,
                oracle_calls_to_opt=calls_to_opt, evaluations_to_opt=evals_to_opt,
                history=hist)


# ---------------------------------------------------------------- resources
def circuit_stats(problem: PlanProblem, y: float | None = None, seed: int = 0) -> dict:
    """Depth and gate counts of one Grover iteration, before and after transpiling to a
    hardware-style basis {cx, rz, sx, x}. Shows the true cost of a truth-table oracle."""
    n = problem.n
    y = problem.optimum()["second_cost"] if y is None else y
    marked = marked_states(problem, y)
    it = QuantumCircuit(n)
    it.append(phase_oracle(marked), range(n))
    it.compose(diffuser(n), range(n), inplace=True)
    t0 = time.perf_counter()
    tq = transpile(it, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed)
    t_tr = time.perf_counter() - t0
    ops = {k: int(v) for k, v in tq.count_ops().items()}
    orc = QuantumCircuit(n)
    orc.append(phase_oracle(marked), range(n))
    tq_o = transpile(orc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed)
    tq_d = transpile(diffuser(n), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                     seed_transpiler=seed)
    return dict(n_qubits=n, ancillas=0, logical_ops={k: int(v) for k, v in it.count_ops().items()},
                logical_depth=int(it.depth()),
                transpiled_basis=["cx", "rz", "sx", "x"], transpiled_ops=ops,
                transpiled_depth=int(tq.depth()), transpiled_size=int(tq.size()),
                cx_per_iteration=ops.get("cx", 0),
                oracle_cx=int(tq_o.count_ops().get("cx", 0)), oracle_depth=int(tq_o.depth()),
                diffuser_cx=int(tq_d.count_ops().get("cx", 0)), diffuser_depth=int(tq_d.depth()),
                transpile_seconds=round(t_tr, 3), marked_for_stats=int(marked.sum()))
