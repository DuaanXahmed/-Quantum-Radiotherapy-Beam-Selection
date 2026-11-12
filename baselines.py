"""Classical baselines: exact enumeration, random search, greedy, simulated annealing, NNLS bound."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import nnls

from plans import PRESCRIPTION, PlanProblem, index_to_bits


def brute_force(problem: PlanProblem, feasible_only: bool = False) -> dict:
    """Exact optimum by evaluating plans one at a time (honest per-plan loop, not vectorised)."""
    feas = problem.feasible_mask()
    candidates = np.flatnonzero(feas) if feasible_only else np.arange(problem.N)
    t0 = time.perf_counter()
    best, best_F = -1, math.inf
    for i in candidates:
        x = index_to_bits(int(i), problem.n)
        if not feasible_only and not problem.feasible(x):
            continue
        f = problem.cost(x)
        if f < best_F:
            best, best_F = int(i), f
    dt = time.perf_counter() - t0
    return dict(index=best, cost=best_F, evaluations=int(candidates.size), seconds=dt)


def random_search(problem: PlanProblem, trials: int = 1000, seed: int = 7) -> dict:
    """Random search without replacement: evaluations until the optimum is drawn."""
    rng = np.random.default_rng(seed)
    opt = problem.optimum()["index"]
    feas_idx = np.flatnonzero(problem.feasible_mask())
    N, M = problem.N, feas_idx.size
    all_states = np.array([int(np.flatnonzero(rng.permutation(N) == opt)[0]) + 1 for _ in range(trials)])
    feas_pos = int(np.flatnonzero(feas_idx == opt)[0])
    feas_only = np.array([int(np.flatnonzero(rng.permutation(M) == feas_pos)[0]) + 1 for _ in range(trials)])
    return dict(trials=trials,
                all_states=dict(mean=float(all_states.mean()), median=float(np.median(all_states)),
                                analytic=(N + 1) / 2, samples=all_states.tolist()),
                feasible_only=dict(mean=float(feas_only.mean()), median=float(np.median(feas_only)),
                                   analytic=(M + 1) / 2, samples=feas_only.tolist()))


def greedy(problem: PlanProblem) -> dict:
    """Forward selection: repeatedly add the beam that lowers F most while staying feasible."""
    x = np.zeros(problem.n, dtype=np.int8)
    f, evals, path = problem.cost(x), 1, []
    while x.sum() < problem.k:
        best_a, best_f = None, f
        for a in np.flatnonzero(x == 0):
            x[a] = 1
            evals += 1
            if problem.feasible(x):
                fa = problem.cost(x)
                if fa < best_f:
                    best_a, best_f = int(a), fa
            x[a] = 0
        if best_a is None:
            break
        x[best_a], f = 1, best_f
        path.append(float(problem.phantom.angles_deg[best_a]))
    idx = int((x.astype(int) << np.arange(problem.n)).sum())
    return dict(index=idx, cost=float(f), evaluations=evals, beams_added_in_order=path,
                found_opt=bool(idx == problem.optimum()["index"]))


def simulated_annealing(problem: PlanProblem, seed: int, steps: int = 300,
                        T0: float = 0.02, T1: float = 1e-4, p_swap: float = 0.5) -> dict:
    """Constraint-aware SA. Moves: flip one beam on/off, or (prob. p_swap) swap a used beam
    for an unused one, which keeps the beam count (plain flips get trapped at the k-beam limit).
    Infeasible proposals are rejected outright."""
    rng = np.random.default_rng(seed)
    F, feas = problem.costs(), problem.feasible_mask()
    opt = problem.optimum()["index"]
    cur = int(rng.choice(np.flatnonzero(feas)))
    best, evals, evals_to_opt = cur, 1, (1 if cur == opt else None)
    for s in range(steps):
        T = T0 * (T1 / T0) ** (s / max(1, steps - 1))
        on = [a for a in range(problem.n) if cur >> a & 1]
        off = [a for a in range(problem.n) if not cur >> a & 1]
        if on and off and rng.random() < p_swap:
            nxt = cur ^ (1 << int(rng.choice(on))) ^ (1 << int(rng.choice(off)))
        else:
            nxt = cur ^ (1 << int(rng.integers(problem.n)))
        evals += 1
        if not feas[nxt]:
            continue
        dF = F[nxt] - F[cur]
        if dF < 0 or rng.random() < math.exp(-dF / T):
            cur = nxt
            if F[cur] < F[best]:
                best = cur
                if best == opt and evals_to_opt is None:
                    evals_to_opt = evals
    return dict(seed=seed, index=int(best), cost=float(F[best]), found_opt=bool(best == opt),
                evaluations=evals, evaluations_to_opt=evals_to_opt)


def nnls_bound(problem: PlanProblem) -> dict:
    """Continuous relaxation: beam weights w >= 0 (any intensity), no beam limit, no C2.
    min_w F(w) is a weighted least-squares problem; every binary plan is a feasible w,
    so this value is a valid lower bound on F for every plan."""
    ph = problem.phantom
    blocks, rhs = [], []
    for mask, w, target in [(ph.tumour, problem.w_T, PRESCRIPTION), (ph.oar, problem.w_O, 0.0),
                            (ph.normal, problem.w_N, 0.0)]:
        s = math.sqrt(w / mask.sum())
        blocks.append(s * ph.D[:, mask].T)
        rhs.append(np.full(mask.sum(), s * target))
    A, b = np.vstack(blocks), np.concatenate(rhs)
    w, res = nnls(A, b)
    return dict(bound=float(res**2), weights=[float(v) for v in w],
                beams_used=[float(a) for a, v in zip(ph.angles_deg, w) if v > 1e-6])
