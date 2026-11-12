"""Live demo for the 3-minute video: shuffled reads -> QUBO -> Qiskit QAOA -> recovered genome.

    python demo.py             # uses the angles trained by run_experiments.py (about 10 s)
    python demo.py --train     # trains a small QAOA live instead (about 30 s more)
    python demo.py --pause     # waits for Enter between steps, so you can narrate

The ground-truth genome is printed and used only to score the answer at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

import baselines
import instance as inst
import qubo
from qaoa import QAOA, cost_hamiltonian

RESULTS = Path(__file__).parent / "results" / "results.json"
LAM, SHOTS, SEED = 3, 4096, 2026


def order_str(o):
    return " -> ".join(f"F{i}" for i in o)


def step(n, title, pause):
    if pause:
        input("\n[Enter] ")
    print(f"\n{'=' * 72}\nSTEP {n}. {title}\n{'=' * 72}")


def trained_angles(p: int, objective: str = "cvar"):
    if not RESULTS.exists():
        return None
    for r in json.loads(RESULTS.read_text())["qaoa"]:
        if r["lambda"] == LAM and r["p"] == p and r["objective"] == objective:
            return np.array(r["theta"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train QAOA live instead of loading saved angles")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    ap.add_argument("-p", type=int, default=3, help="QAOA layers (default 3)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ 1
    step(1, "The input: 4 shuffled DNA reads from the sequencer", args.pause)
    O = inst.overlap_matrix(inst.FRAGMENTS)
    for i, f in enumerate(inst.FRAGMENTS):
        print(f"  F{i}: {f}")
    print("\n  Overlap o_ij = longest suffix of Fi that equals a prefix of Fj:")
    print("        " + "  ".join(f"F{j}" for j in range(4)))
    for i in range(4):
        print(f"   F{i}  " + "  ".join(f"{O[i, j]:>2}" if i != j else " -" for j in range(4)))
    print(f"\n  A junction with overlap < {inst.TAU} is WEAK: repeats make short overlaps unreliable.")

    # ------------------------------------------------------------------ 2
    step(2, "Two competing goals: contiguity vs reliability (exact, all 24 orderings)", args.pause)
    for lam, label in ((0, "maximise overlap only"), (LAM, "also penalise weak junctions")):
        bf = baselines.brute_force(inst.junction_weights(O, lam))
        m = inst.order_metrics(bf["order"], O)
        seq = inst.assemble(bf["order"], inst.FRAGMENTS)
        print(f"  lambda = {lam} ({label}):")
        print(f"     best ordering {order_str(bf['order'])}   overlap {m['total_overlap']}, "
              f"weak junctions {m['weak_junctions']}")
        print(f"     assembles to  {seq}")
    print(f"  true genome     {inst.GENOME}   (used only for scoring)")
    print("\n  Maximum overlap alone gives the WRONG sequence. That is the trade-off we optimise.")

    # ------------------------------------------------------------------ 3
    step(3, "Map to a QUBO, then to a 16-qubit Ising Hamiltonian", args.pause)
    A = qubo.smallest_valid_penalty(O, LAM)
    model = qubo.make_model(O, LAM, A)
    H = cost_hamiltonian(model)
    print("  x[i,t] = 1  <=>  fragment i at position t   ->  4 x 4 = 16 binary variables = 16 qubits")
    print(f"  E(x) = A*(one-hot penalties) - sum W_ij x[i,t] x[j,t+1],   A = {A:g}, lambda = {LAM}")
    print(f"  Qiskit SparsePauliOp: {H.num_qubits} qubits, {len(H)} Pauli terms "
          f"({sum(1 for s in H.paulis.to_labels() if s.count('Z') == 2)} ZZ couplings)")
    print(f"  Valid orderings: 24 of 2^16 = {2 ** 16} bitstrings ({24 / 2 ** 16:.4%})")

    # ------------------------------------------------------------------ 4
    step(4, f"Build the QAOA circuit (p = {args.p}) and get its angles", args.pause)
    q = QAOA(model, args.p)
    stats = q.circuit_stats()
    print(f"  {stats['qubits']} qubits, depth {stats['depth']}, {stats['cx']} CX gates "
          f"(basis rz/sx/x/cx, all-to-all)")
    theta = None if args.train else trained_angles(args.p)
    if theta is None:
        print("  Training with COBYLA on the CVaR objective (alpha = 0.1) ...", flush=True)
        t0 = time.perf_counter()
        theta = np.array(q.fit("cvar", alpha=0.1, restarts=2, maxiter=150, seed=SEED)["theta"])
        print(f"  trained in {time.perf_counter() - t0:.1f} s")
    else:
        print("  Loaded the angles trained by run_experiments.py (CVaR objective).")
    print("  gamma =", np.round(theta[:args.p], 3), "  beta =", np.round(theta[args.p:], 3))

    # ------------------------------------------------------------------ 5
    step(5, f"Run the circuit: {SHOTS} measured shots on the Qiskit Aer sampler", args.pause)
    qc = q.circuit.assign_parameters(dict(zip(q.params, theta)))
    qc.measure_all()
    sampler = SamplerV2(seed=SEED)
    t0 = time.perf_counter()
    counts = sampler.run([transpile(qc, AerSimulator())], shots=SHOTS).result()[0].data.meas.get_counts()
    print(f"  sampled in {time.perf_counter() - t0:.2f} s; {len(counts)} distinct bitstrings")
    valid = Counter()
    for bits, c in counts.items():
        order = model.decode(int(bits, 2))  # Qiskit bitstrings are little-endian: qubit 0 is rightmost
        if order is not None:
            valid[order] += c
    n_valid = sum(valid.values())
    print(f"  valid orderings: {n_valid}/{SHOTS} shots = {n_valid / SHOTS:.1%}   "
          f"(random bitstrings: {24 / 2 ** 16:.3%}, so about {n_valid / SHOTS / (24 / 2 ** 16):.0f}x)")
    print("\n  Most frequent valid orderings:")
    for order, c in valid.most_common(5):
        m = inst.order_metrics(order, O)
        print(f"    {order_str(order)}   {c:>4} shots   overlap {m['total_overlap']:>2}, weak {m['weak_junctions']}")

    # ------------------------------------------------------------------ 6
    step(6, "Decode: keep the best valid ordering and check it against the exact answer", args.pause)
    best = max(valid, key=lambda o: inst.gain(o, model.W)) if valid else None
    exact = baselines.brute_force(model.W)
    if best is None:
        print("  No valid ordering sampled. Try more shots or more layers.")
        return
    seq = inst.assemble(best, inst.FRAGMENTS)
    m = inst.order_metrics(best, O)
    print(f"  QAOA answer     {order_str(best)}   overlap {m['total_overlap']}, weak junctions {m['weak_junctions']}")
    print(f"  exact optimum   {order_str(exact['order'])}   (brute force over all 24 orderings, "
          f"{exact['seconds'] * 1e6:.0f} us)")
    print(f"\n  assembled       {seq}")
    print(f"  true genome     {inst.GENOME}")
    ok = seq == inst.GENOME
    print(f"\n  {'GENOME RECOVERED' if ok else 'NOT the true genome'}"
          f"{' and matches the exact optimum' if tuple(best) == tuple(exact['order']) else ''}.")
    print("\n  Honest note: at 4 fragments brute force is instant. The demo shows a correct, tested\n"
          "  quantum pipeline, not a quantum speed-up.")


if __name__ == "__main__":
    main()
