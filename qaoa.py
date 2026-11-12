"""QAOA for the fragment-ordering QUBO, implemented with Qiskit.

Circuit (p layers), on n^2 qubits:
    |+>^n  ->  [ exp(-i gamma_k H_C)  exp(-i beta_k sum_i X_i) ]_{k=1..p}  ->  measure
where H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j is the Ising form of the QUBO
(divided by a constant so gamma lives on an O(1) scale).

Simulation: qiskit-aer statevector. The classical optimiser (COBYLA) updates the
2p angles; the objective is either the expected energy or CVaR_alpha of the
measured energy distribution (Barkoutsos et al., Quantum 4, 256, 2020).
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from qubo import QUBOModel, qubo_to_ising


def cost_hamiltonian(model: QUBOModel) -> SparsePauliOp:
    """Ising Hamiltonian (with constant) as a Qiskit SparsePauliOp."""
    h, J, c0 = qubo_to_ising(model.Q, model.offset)
    terms = [("", [], c0)]
    terms += [("Z", [i], float(h[i])) for i in range(len(h)) if h[i] != 0.0]
    terms += [("ZZ", [i, j], float(v)) for (i, j), v in J.items()]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=model.num_vars).simplify()


def qaoa_circuit(model: QUBOModel, p: int):
    """Parametrised QAOA circuit. Returns (circuit, [gamma..., beta...] parameters)."""
    h, J, _ = qubo_to_ising(model.Q, model.offset)
    scale = max(np.abs(h).max(), max(abs(v) for v in J.values()))
    nq = model.num_vars
    gam, bet = ParameterVector("g", p), ParameterVector("b", p)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for k in range(p):
        for i in range(nq):
            if h[i] != 0.0:
                qc.rz(2 * gam[k] * float(h[i] / scale), i)
        for (i, j), v in J.items():
            qc.rzz(2 * gam[k] * float(v / scale), i, j)
        for i in range(nq):
            qc.rx(2 * bet[k], i)
    return qc, list(gam) + list(bet)


class QAOA:
    def __init__(self, model: QUBOModel, p: int):
        self.model, self.p = model, p
        self.circuit, self.params = qaoa_circuit(model, p)
        self.sim = AerSimulator(method="statevector")
        sv = self.circuit.copy()
        sv.save_statevector()
        self._tqc = transpile(sv, self.sim, optimization_level=0)
        self.energies = model.energies
        self.nfev = 0

    # ---- simulation ---------------------------------------------------
    def probs(self, theta) -> np.ndarray:
        bound = self._tqc.assign_parameters(dict(zip(self.params, theta)))
        self.nfev += 1
        sv = self.sim.run(bound).result().get_statevector()
        return np.asarray(sv.probabilities())

    def expectation(self, theta) -> float:
        return float(self.probs(theta) @ self.energies)

    def cvar(self, theta, alpha: float = 0.1) -> float:
        pr = self.probs(theta)
        order = np.argsort(self.energies)
        cum = np.cumsum(pr[order])
        w = pr[order].copy()
        over = np.searchsorted(cum, alpha)  # first index where cumulative mass >= alpha
        if over < len(w):
            w[over] -= cum[over] - alpha
            w[over + 1:] = 0.0
        return float(w @ self.energies[order] / alpha)

    # ---- training -----------------------------------------------------
    def fit(self, objective: str = "expectation", alpha: float = 0.1, restarts: int = 8,
            maxiter: int = 250, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        f = self.expectation if objective == "expectation" else (lambda th: self.cvar(th, alpha))
        p = self.p
        best = None
        self.nfev = 0
        t0 = time.perf_counter()
        for r in range(restarts):
            if r == 0:  # linear-ramp start (annealing-inspired; Sack & Serbyn, Quantum 5, 491, 2021)
                ramp = (np.arange(p) + 0.5) / p
                x0 = np.concatenate([0.8 * ramp, 0.8 * (1 - ramp)])
            else:
                x0 = np.concatenate([rng.uniform(0, 1.5, p), rng.uniform(0, 1.0, p)])
            res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
            if best is None or res.fun < best.fun:
                best = res
        out = self.evaluate(best.x)
        out.update({"objective": objective, "theta": best.x.tolist(), "p": p,
                    "restarts": restarts, "nfev": self.nfev,
                    "seconds": time.perf_counter() - t0})
        return out

    # ---- evaluation ---------------------------------------------------
    def evaluate(self, theta, shots: int = 4096, seed: int = 0) -> dict:
        m = self.model
        pr = self.probs(theta)
        E = self.energies
        k_opt = int(np.argmin(E))
        uniform_mean = float(E.mean())
        exp_E = float(pr @ E)
        rng = np.random.default_rng(seed)
        samples = rng.choice(len(pr), size=shots, p=pr / pr.sum())
        feas = m.feasible[samples]
        best_shot = None
        if feas.any():
            cand = samples[feas]
            best_shot = m.decode(int(cand[np.argmin(E[cand])]))
        feasible_idx = np.flatnonzero(m.feasible)
        top_feasible = m.decode(int(feasible_idx[np.argmax(pr[feasible_idx])]))
        return {
            "expected_energy": exp_E,
            "uniform_mean_energy": uniform_mean,
            "optimal_energy": float(E[k_opt]),
            "energy_improvement": (uniform_mean - exp_E) / (uniform_mean - float(E[k_opt])),
            "p_optimum": float(pr[k_opt]),
            "p_feasible": float(pr[m.feasible].sum()),
            "p_feasible_uniform": float(m.feasible.mean()),
            "optimum_order": m.decode(k_opt),
            "most_probable_feasible_order": top_feasible,
            "best_order_in_shots": best_shot,
            "shots": shots,
            "probs": pr,
        }

    def circuit_stats(self) -> dict:
        """Gate/depth statistics after transpiling to a hardware-like basis (all-to-all coupling)."""
        bound = self.circuit.assign_parameters(dict(zip(self.params, np.full(len(self.params), 0.37))))
        t = transpile(bound, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        ops = dict(t.count_ops())
        return {"qubits": t.num_qubits, "depth": t.depth(), "cx": int(ops.get("cx", 0)),
                "total_gates": int(sum(ops.values())), "ops": {k: int(v) for k, v in ops.items()}}
