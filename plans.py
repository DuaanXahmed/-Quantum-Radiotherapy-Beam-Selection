"""Treatment-plan model: cost F(x), constraints, bit encoding and exact enumeration.

Encoding: plan x in {0,1}^n, x_a = 1 if beam a is used. The integer index of a plan is
sum_a x_a 2^a, i.e. bit a of the index is beam a. This is Qiskit's little-endian order,
so computational basis state |index> on n qubits *is* the plan (qubit a <-> beam a).

Objective (minimise):
    F(x) = w_T * mean_tumour (d - 1)^2 + w_O * mean_OAR d^2 + w_N * mean_normal d^2
Hard constraints:
    C1  sum_a x_a <= k                       (beam limit)
    C2  max_OAR d <= oar_max                  (DVH-style maximum-dose limit)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phantom import Phantom, build_phantom

PRESCRIPTION = 1.0
DEFAULTS = dict(w_T=1.0, w_O=1.0, w_N=0.05, k=4, oar_max=0.6)


def index_to_bits(idx, n: int) -> np.ndarray:
    idx = np.asarray(idx)
    return ((idx[..., None] >> np.arange(n)) & 1).astype(np.int8)


def bits_to_index(bits) -> int:
    bits = np.asarray(bits).astype(int)
    return int((bits << np.arange(bits.size)).sum())


def bitstring(idx: int, n: int) -> str:
    """Qiskit-style bitstring (qubit n-1 on the left)."""
    return format(idx, f"0{n}b")


@dataclass
class PlanProblem:
    phantom: Phantom
    w_T: float = DEFAULTS["w_T"]
    w_O: float = DEFAULTS["w_O"]
    w_N: float = DEFAULTS["w_N"]
    k: int = DEFAULTS["k"]
    oar_max: float = DEFAULTS["oar_max"]
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------- single plan ----------
    @property
    def n(self) -> int:
        return self.phantom.n_beams

    @property
    def N(self) -> int:
        return 2 ** self.n

    def dose(self, x) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.phantom.D

    def terms(self, x) -> dict:
        d, ph = self.dose(x), self.phantom
        return dict(tumour_err=float(np.mean((d[ph.tumour] - PRESCRIPTION) ** 2)),
                    oar_sq=float(np.mean(d[ph.oar] ** 2)),
                    normal_sq=float(np.mean(d[ph.normal] ** 2)),
                    oar_mean=float(np.mean(d[ph.oar])),
                    oar_maxdose=float(np.max(d[ph.oar])),
                    tumour_mean=float(np.mean(d[ph.tumour])),
                    tumour_min=float(np.min(d[ph.tumour])),
                    n_beams=int(np.sum(x)))

    def cost(self, x) -> float:
        t = self.terms(x)
        return self.w_T * t["tumour_err"] + self.w_O * t["oar_sq"] + self.w_N * t["normal_sq"]

    def feasible(self, x) -> bool:
        t = self.terms(x)
        return t["n_beams"] <= self.k and t["oar_maxdose"] <= self.oar_max + 1e-12

    # ---------- all 2^n plans at once ----------
    def table(self) -> dict:
        """Vectorised evaluation of every plan (exact enumeration)."""
        if "table" not in self._cache:
            ph = self.phantom
            bits = index_to_bits(np.arange(self.N), self.n)
            d = bits.astype(float) @ ph.D
            self._cache["table"] = dict(
                bits=bits,
                n_beams=bits.sum(1),
                tumour_err=((d[:, ph.tumour] - PRESCRIPTION) ** 2).mean(1),
                oar_sq=(d[:, ph.oar] ** 2).mean(1),
                normal_sq=(d[:, ph.normal] ** 2).mean(1),
                oar_mean=d[:, ph.oar].mean(1),
                oar_maxdose=d[:, ph.oar].max(1),
            )
        return self._cache["table"]

    def costs(self, w_O: float | None = None) -> np.ndarray:
        t = self.table()
        w_O = self.w_O if w_O is None else w_O
        return self.w_T * t["tumour_err"] + w_O * t["oar_sq"] + self.w_N * t["normal_sq"]

    def feasible_mask(self, use_c2: bool = True) -> np.ndarray:
        t = self.table()
        m = t["n_beams"] <= self.k
        if use_c2:
            m &= t["oar_maxdose"] <= self.oar_max + 1e-12
        return m

    def optimum(self, w_O: float | None = None) -> dict:
        F, feas = self.costs(w_O), self.feasible_mask()
        Ff = np.where(feas, F, np.inf)
        order = np.argsort(Ff, kind="stable")
        best, second = int(order[0]), int(order[1])
        gap = float(Ff[second] - Ff[best])
        return dict(index=best, cost=float(Ff[best]), second_index=second,
                    second_cost=float(Ff[second]), gap=gap,
                    unique=bool(gap > 1e-9 * max(1.0, abs(Ff[best]))))

    def beams(self, idx: int) -> list[float]:
        bits = index_to_bits(idx, self.n)
        return [float(a) for a, b in zip(self.phantom.angles_deg, bits) if b]


def default_problem(n_beams: int = 12, **kw) -> PlanProblem:
    return PlanProblem(build_phantom(n_beams), **kw)


def weight_sweep(problem: PlanProblem, w_O_values=(0.0, 0.5, 1.0, 2.0, 5.0)) -> list[dict]:
    """Exact optimum for each OAR weight -> points on the trade-off (Pareto) front."""
    t, out = problem.table(), []
    for w in w_O_values:
        o = problem.optimum(w)
        i = o["index"]
        out.append(dict(w_O=float(w), index=i, bitstring=bitstring(i, problem.n),
                        beams=problem.beams(i), cost=o["cost"], unique=o["unique"],
                        tumour_err=float(t["tumour_err"][i]), oar_mean=float(t["oar_mean"][i]),
                        oar_maxdose=float(t["oar_maxdose"][i]), n_beams=int(t["n_beams"][i])))
    return out


def pareto_front(tumour_err: np.ndarray, oar_mean: np.ndarray) -> np.ndarray:
    """Indices (into the given arrays) of non-dominated points, both objectives minimised."""
    order = np.lexsort((oar_mean, tumour_err))
    front, best_oar = [], np.inf
    for i in order:
        if oar_mean[i] < best_oar - 1e-12:
            front.append(int(i))
            best_oar = oar_mean[i]
    return np.array(front, dtype=int)
