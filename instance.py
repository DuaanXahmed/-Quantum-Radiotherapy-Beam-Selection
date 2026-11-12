"""Toy DNA fragment-ordering instance and junction scoring.

The problem: order N short DNA fragments (reads) into one sequence.
Each adjacent pair (i -> j) is a "junction" with overlap length o_ij
(longest proper suffix of fragment i that equals a prefix of fragment j).

Two competing objectives
  A. contiguity   : maximise the total overlap along the ordering
  B. reliability  : minimise the number of weak junctions (overlap < TAU),
                    because weak junctions are where repeats cause misassembly

They are combined with a penalty weight lam:  W_ij = o_ij - lam * [o_ij < TAU].
Sweeping lam traces the trade-off between A and B.
"""
from __future__ import annotations

import numpy as np

# Ground truth is used ONLY to evaluate results, never by the optimisers.
GENOME = "TACACCATACCAT"
FRAGMENTS = ["ATACCA", "TACCAT", "TACACC", "ACCATA"]  # shuffled length-6 reads
TRUE_ORDER = (2, 3, 0, 1)  # F2 -> F3 -> F0 -> F1 reconstructs GENOME
TAU = 3  # confidence threshold, in bases


def overlap(a: str, b: str) -> int:
    """Length of the longest proper suffix of `a` that equals a prefix of `b`."""
    for k in range(min(len(a), len(b)) - 1, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def overlap_matrix(frags: list[str]) -> np.ndarray:
    n = len(frags)
    O = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                O[i, j] = overlap(frags[i], frags[j])
    return O


def junction_weights(O: np.ndarray, lam: float, tau: int = TAU) -> np.ndarray:
    """W_ij = o_ij - lam * [o_ij < tau]  (diagonal unused, set to 0)."""
    W = O.astype(float) - lam * (O < tau)
    np.fill_diagonal(W, 0.0)
    return W


def order_metrics(order, O: np.ndarray, tau: int = TAU) -> dict:
    """Objective A (total overlap) and objective B (weak junction count)."""
    juncs = [int(O[i, j]) for i, j in zip(order, order[1:])]
    return {
        "order": [int(i) for i in order],
        "total_overlap": int(sum(juncs)),
        "weak_junctions": int(sum(o < tau for o in juncs)),
        "junction_overlaps": juncs,
    }


def gain(order, W: np.ndarray) -> float:
    """Scalarised objective (to be maximised): sum of junction weights."""
    return float(sum(W[i, j] for i, j in zip(order, order[1:])))


def assemble(order, frags: list[str]) -> str:
    """Merge fragments along `order` using their exact suffix/prefix overlaps."""
    seq = frags[order[0]]
    for i, j in zip(order, order[1:]):
        seq += frags[j][overlap(frags[i], frags[j]):]
    return seq


if __name__ == "__main__":
    O = overlap_matrix(FRAGMENTS)
    print("overlap matrix (row -> col):\n", O)
    print("true order assembles to:", assemble(TRUE_ORDER, FRAGMENTS), "| genome:", GENOME)
