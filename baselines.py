"""Classical / exact baselines for the fragment-ordering problem.

* brute_force      : exact, enumerates all N! orderings
* greedy           : multi-start nearest-neighbour on the junction weights
* sa_permutation   : simulated annealing over orderings (swap moves)
* sa_qubo          : simulated annealing on the SAME QUBO the quantum circuit sees
* random_sampling  : uniform random bitstrings on the same QUBO
"""
from __future__ import annotations

import math
import time
from itertools import permutations

import numpy as np

import instance as inst
from qubo import QUBOModel


def brute_force(W: np.ndarray) -> dict:
    t0 = time.perf_counter()
    n = W.shape[0]
    scored = sorted(((inst.gain(p, W), p) for p in permutations(range(n))), reverse=True)
    best_gain, best = scored[0]
    return {"order": best, "gain": best_gain, "unique": scored[0][0] > scored[1][0],
            "all": scored, "seconds": time.perf_counter() - t0}


def greedy(W: np.ndarray) -> dict:
    t0 = time.perf_counter()
    n = W.shape[0]
    best, best_gain = None, -math.inf
    for s in range(n):
        order, left = [s], set(range(n)) - {s}
        while left:
            nxt = max(left, key=lambda j: W[order[-1], j])
            order.append(nxt)
            left.remove(nxt)
        g = inst.gain(order, W)
        if g > best_gain:
            best, best_gain = tuple(order), g
    return {"order": best, "gain": best_gain, "seconds": time.perf_counter() - t0}


def sa_permutation(W: np.ndarray, rng: np.random.Generator, steps: int = 2000,
                   T0: float = 3.0, T1: float = 0.05) -> dict:
    t0 = time.perf_counter()
    n = W.shape[0]
    cur = list(rng.permutation(n))
    cur_g = inst.gain(cur, W)
    best, best_g = tuple(cur), cur_g
    for s in range(steps):
        T = T0 * (T1 / T0) ** (s / max(steps - 1, 1))
        a, b = rng.choice(n, size=2, replace=False)
        cand = cur.copy()
        cand[a], cand[b] = cand[b], cand[a]
        g = inst.gain(cand, W)
        if g >= cur_g or rng.random() < math.exp((g - cur_g) / T):
            cur, cur_g = cand, g
            if cur_g > best_g:
                best, best_g = tuple(cur), cur_g
    return {"order": best, "gain": best_g, "seconds": time.perf_counter() - t0}


def sa_qubo(model: QUBOModel, rng: np.random.Generator, steps: int = 20000,
            T0: float | None = None, T1: float = 0.05) -> dict:
    """Single-bit-flip simulated annealing on E(x) = x^T Q x + offset.
    Returns the best FEASIBLE state seen (or None)."""
    t0 = time.perf_counter()
    nv, n = model.num_vars, model.n
    T0 = T0 if T0 is not None else model.A
    diag = np.diag(model.Q).copy()
    S = model.Q + model.Q.T
    np.fill_diagonal(S, 0.0)
    x = rng.integers(0, 2, size=nv).astype(float)
    E = model.energy(x)
    best_E, best_x = math.inf, None

    def is_feasible(v):
        M = v.reshape(n, n)
        return bool((M.sum(axis=1) == 1).all() and (M.sum(axis=0) == 1).all())

    for s in range(steps):
        T = T0 * (T1 / T0) ** (s / max(steps - 1, 1))
        k = int(rng.integers(nv))
        d = 1.0 - 2.0 * x[k]
        delta = d * (diag[k] + S[k] @ x)
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            x[k] += d
            E += delta
            if E < best_E - 1e-12 and is_feasible(x):
                best_E, best_x = E, x.copy()
    order = None
    if best_x is not None:
        M = best_x.reshape(n, n)
        order = tuple(int(M[:, t].argmax()) for t in range(n))
    return {"order": order, "energy": best_E, "seconds": time.perf_counter() - t0}


def random_sampling(model: QUBOModel, rng: np.random.Generator, shots: int = 4096) -> dict:
    t0 = time.perf_counter()
    ks = rng.integers(0, 2 ** model.num_vars, size=shots)
    feas = model.feasible[ks]
    best_E = math.inf
    best_k = None
    if feas.any():
        cand = ks[feas]
        j = int(np.argmin(model.energies[cand]))
        best_k, best_E = int(cand[j]), float(model.energies[cand[j]])
    return {"order": model.decode(best_k) if best_k is not None else None,
            "energy": best_E, "feasible_fraction": float(feas.mean()),
            "seconds": time.perf_counter() - t0}
