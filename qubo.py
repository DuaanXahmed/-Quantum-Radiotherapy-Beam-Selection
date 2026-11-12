"""QUBO / Ising formulation of the fragment-ordering problem.

Binary variable x[i,t] = 1  <=>  fragment i is placed at position t
(flattened as var = i*N + t, so N fragments need N^2 qubits).

Energy to MINIMISE:
    E(x) = A * sum_i (1 - sum_t x[i,t])^2          # each fragment used once
         + A * sum_t (1 - sum_i x[i,t])^2          # each position filled once
         - sum_t sum_{i!=j} W[i,j] x[i,t] x[j,t+1] # junction gains (objectives A and B)

with W[i,j] = o[i,j] - lam*[o[i,j] < tau]  (see instance.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import permutations

import numpy as np

import instance as inst

MAX_BRUTE_VARS = 22  # enumerating all 2^n bitstrings is only done for small n


def var(i: int, t: int, n: int) -> int:
    return i * n + t


def _add_onehot(Q: np.ndarray, variables: list[int], A: float) -> None:
    # (1 - S)^2 = 1 - S + 2*sum_{a<b} x_a x_b   for binary x (the "1" goes to the offset)
    for v in variables:
        Q[v, v] += -A
    for a in range(len(variables)):
        for b in range(a + 1, len(variables)):
            u, w = sorted((variables[a], variables[b]))
            Q[u, w] += 2 * A


def build_qubo(W: np.ndarray, A: float) -> tuple[np.ndarray, float]:
    """Return (Q, offset) with E(x) = x^T Q x + offset, Q upper-triangular."""
    n = W.shape[0]
    Q = np.zeros((n * n, n * n))
    offset = 0.0
    for i in range(n):
        _add_onehot(Q, [var(i, t, n) for t in range(n)], A)
        offset += A
    for t in range(n):
        _add_onehot(Q, [var(i, t, n) for i in range(n)], A)
        offset += A
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    a, b = sorted((var(i, t, n), var(j, t + 1, n)))
                    Q[a, b] += -W[i, j]
    return Q, offset


def qubo_to_ising(Q: np.ndarray, offset: float):
    """x = (1 - Z)/2  ->  E = c0 + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""
    nv = Q.shape[0]
    h = np.zeros(nv)
    J: dict[tuple[int, int], float] = {}
    c0 = offset
    for i in range(nv):
        c0 += Q[i, i] / 2
        h[i] += -Q[i, i] / 2
        for j in range(i + 1, nv):
            q = Q[i, j]
            if q != 0.0:
                c0 += q / 4
                h[i] += -q / 4
                h[j] += -q / 4
                J[(i, j)] = q / 4
    return h, J, c0


@dataclass
class QUBOModel:
    n: int  # number of fragments
    A: float  # constraint penalty weight
    lam: float
    W: np.ndarray
    Q: np.ndarray
    offset: float

    @property
    def num_vars(self) -> int:
        return self.n * self.n

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    # ---- exhaustive tables (only for small instances) -------------------
    @cached_property
    def _bits(self) -> np.ndarray:
        if self.num_vars > MAX_BRUTE_VARS:
            raise ValueError("instance too large for exhaustive tables")
        k = np.arange(2 ** self.num_vars)[:, None]
        return ((k >> np.arange(self.num_vars)) & 1).astype(np.float64)

    @cached_property
    def energies(self) -> np.ndarray:
        """E(x) for every bitstring; index k has bit i = qubit i (Qiskit little-endian)."""
        B = self._bits
        return ((B @ self.Q) * B).sum(axis=1) + self.offset

    @cached_property
    def feasible(self) -> np.ndarray:
        B = self._bits.reshape(-1, self.n, self.n)  # [k, fragment, position]
        return (B.sum(axis=2) == 1).all(axis=1) & (B.sum(axis=1) == 1).all(axis=1)

    def decode(self, k: int) -> tuple[int, ...] | None:
        """Bitstring index -> fragment order, or None if infeasible."""
        if not self.feasible[k]:
            return None
        B = self._bits[k].reshape(self.n, self.n)
        return tuple(int(B[:, t].argmax()) for t in range(self.n))

    @cached_property
    def order_to_index(self) -> dict:
        return {self.decode(k): int(k) for k in np.flatnonzero(self.feasible)}

    def encode(self, order) -> np.ndarray:
        x = np.zeros(self.num_vars)
        for t, i in enumerate(order):
            x[var(i, t, self.n)] = 1
        return x


def make_model(O: np.ndarray, lam: float, A: float, tau: int = inst.TAU) -> QUBOModel:
    W = inst.junction_weights(O, lam, tau)
    Q, offset = build_qubo(W, A)
    return QUBOModel(n=O.shape[0], A=A, lam=lam, W=W, Q=Q, offset=offset)


def best_order_brute_force(W: np.ndarray):
    n = W.shape[0]
    scored = sorted(((inst.gain(p, W), p) for p in permutations(range(n))), reverse=True)
    return scored


def smallest_valid_penalty(O: np.ndarray, lam: float, tau: int = inst.TAU,
                           candidates=(2, 3, 4, 5, 6, 8, 10, 12, 16)) -> float:
    """Smallest A such that the QUBO ground state is the unique feasible optimum."""
    W = inst.junction_weights(O, lam, tau)
    scored = best_order_brute_force(W)
    best_gain, best_perm = scored[0]
    for A in candidates:
        m = make_model(O, lam, A, tau)
        k = int(np.argmin(m.energies))
        if m.decode(k) == best_perm and abs(m.energies[k] + best_gain) < 1e-9:
            return float(A)
    raise RuntimeError("no candidate penalty makes the QUBO ground state feasible+optimal")
