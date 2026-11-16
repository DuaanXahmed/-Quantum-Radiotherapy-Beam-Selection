"""Table-free Quantum Arithmetic Oracle (Gilliam et al. 2021) & Subspace Preparation.

Computes F(x) - y < 0 directly in quantum superposition using a QFT-based Draper
phase adder on m ancilla qubits without evaluating all 2^n plans classically.
Also provides Dicke-subspace state preparation over states with Hamming weight <= k.
"""
from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

from plans import PRESCRIPTION, PlanProblem, index_to_bits


def extract_qubo_coefficients(problem: PlanProblem) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract polynomial coefficients of F(x) = c0 + sum_a L[a]*x_a + sum_{a<b} Q[a,b]*x_a*x_b
    directly from the n x V dose matrix D (O(n^2 * V) time, NEVER builds the 2^n table!).
    Uses the binary identity x_a^2 = x_a.
    """
    ph = problem.phantom
    n = problem.n
    D = ph.D

    # Weight per voxel v across the 3 regions
    w_v = np.zeros(ph.n_voxels, dtype=float)
    w_v[ph.tumour] = problem.w_T / ph.tumour.sum()
    w_v[ph.oar] = problem.w_O / ph.oar.sum()
    w_v[ph.normal] = problem.w_N / ph.normal.sum()

    # Target dose per voxel (1.0 in tumour, 0.0 elsewhere)
    target_v = np.zeros(ph.n_voxels, dtype=float)
    target_v[ph.tumour] = PRESCRIPTION

    # Constant term c0 = sum_v w_v * target_v^2 (= w_T * 1.0)
    c0 = float(np.sum(w_v * (target_v ** 2)))

    # Quadratic matrix M_ab = sum_v w_v * D[a,v] * D[b,v]
    M = (D * w_v) @ D.T

    # Linear vector from -2 * target_v * d_v + diagonal of M (since x_a^2 = x_a)
    linear_target = -2.0 * ((D * w_v) @ target_v)
    L = np.diag(M) + linear_target

    # Off-diagonal quadratic interaction Q_ab = 2 * M_ab for a < b
    Q = np.triu(2.0 * M, k=1)
    return c0, L, Q


def subspace_state_prep(n: int, k: int, blocked_beams: list[int] | None = None) -> tuple[QuantumCircuit, int]:
    """Prepare uniform superposition ONLY over valid-count plans (sum x_a <= k and no blocked beams).
    Shrinks search space from 2^n to sum_{j=0}^k C(n, j) (~5.1x smaller for n=12, k=4).
    """
    N = 2 ** n
    bits = index_to_bits(np.arange(N), n)
    valid = bits.sum(axis=1) <= k
    if blocked_beams:
        for b in blocked_beams:
            if 0 <= b < n:
                valid &= (bits[:, b] == 0)

    count = int(valid.sum())
    amps = np.where(valid, 1.0 / math.sqrt(count), 0.0)
    qc = QuantumCircuit(n, name=f"Dicke_le_{k}")
    qc.append(StatePreparation(amps), range(n))

    # Yahan transpile use karein taake gates optimize ho jayein:
    qc_opt = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
    return qc_opt, count


def subspace_diffuser(prep_qc: QuantumCircuit) -> QuantumCircuit:
    """Diffuser restricted to the feasible k-beam subspace: A * (I - 2|0><0|) * A^dagger."""
    n = prep_qc.num_qubits
    qc = QuantumCircuit(n, name="Subspace_Diffuser")
    qc.compose(prep_qc.inverse(), range(n), inplace=True)
    qc.x(range(n))
    qc.mcp(math.pi, list(range(n - 1)), n - 1)
    qc.x(range(n))
    qc.compose(prep_qc, range(n), inplace=True)
    return qc


def _qft_no_swap(m: int) -> QuantumCircuit:
    """Native QFT circuit matching little-endian phase order (q[0]=LSB, q[m-1]=MSB)."""
    qc = QuantumCircuit(m, name="QFT")
    for j in range(m - 1, -1, -1):
        qc.h(j)
        for k in range(j):
            qc.cp(math.pi / (2 ** (j - k)), k, j)
    return qc


def build_qft_arithmetic_oracle(problem: PlanProblem, y: float, m_ancilla: int = 7) -> QuantumCircuit:
    """Gate-level Gilliam et al. (2021) oracle using QFT phase addition.
    Encodes integer-scaled (F(x) - y) into m_ancilla qubits in two's complement;
    if F(x) - y < 0, the MSB (sign qubit) is |1>, triggering a Z phase kickback.
    """
    n = problem.n
    c0, L, Q = extract_qubo_coefficients(problem)

    # Scale coefficients to integers in [-2^(m-1), 2^(m-1) - 1]
    max_val = abs(c0 - y) + np.sum(np.abs(L)) + np.sum(np.abs(Q))
    scale = (2 ** (m_ancilla - 1) - 2) / max(max_val, 1e-6)

    c_int = int(round((c0 - y) * scale)) - 1  # -1 ensures strict inequality F(x) < y
    L_int = np.round(L * scale).astype(int)
    Q_int = np.round(Q * scale).astype(int)

    qc = QuantumCircuit(n + m_ancilla, name="Arithmetic_O_y")
    anc = list(range(n, n + m_ancilla))

    def add_fourier_phases(sign: int):
        for k_idx, q_anc in enumerate(anc):
            lam_base = sign * (2 * math.pi) / (2 ** (k_idx + 1))
            if c_int != 0:
                qc.p(c_int * lam_base, q_anc)
            for a in range(n):
                if L_int[a] != 0:
                    qc.cp(L_int[a] * lam_base, a, q_anc)
            for a in range(n):
                for b in range(a + 1, n):
                    if Q_int[a, b] != 0:
                        qc.mcp(Q_int[a, b] * lam_base, [a, b], q_anc)

    # 1. Put ancillas in Fourier basis
    qc.h(anc)
    # 2. Add scaled (F(x) - y) via controlled rotations
    add_fourier_phases(+1)
    qft_circ = _qft_no_swap(m_ancilla)
    # 3. Inverse QFT maps phases to two's complement binary integer
    qc.compose(qft_circ.inverse(), anc, inplace=True)
    # 4. Sign bit (MSB = anc[-1]) is |1> iff F(x) < y -> Phase Kickback!
    qc.z(anc[-1])
    # 5. Uncompute ancillas back to |0>
    qc.compose(qft_circ, anc, inplace=True)
    add_fourier_phases(-1)
    qc.h(anc)

    return qc