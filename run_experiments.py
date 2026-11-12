"""Run every experiment reported in the write-up and save results/results.json.

    python run_experiments.py            # full run (~4-6 min on a laptop CPU)
    python run_experiments.py --quick    # smoke test (~30 s)

All randomness is seeded, so the numbers are reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import baselines
import instance as inst
import qubo
from qaoa import QAOA

OUT = Path(__file__).parent / "results"
SEED = 2026


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def pareto_sweep(O, lams):
    rows = []
    for lam in lams:
        W = inst.junction_weights(O, lam)
        bf = baselines.brute_force(W)
        m = inst.order_metrics(bf["order"], O)
        m.update({"lambda": lam, "scalar_gain": bf["gain"], "unique_optimum": bf["unique"],
                  "matches_ground_truth": bf["order"] == inst.TRUE_ORDER,
                  "assembled": inst.assemble(bf["order"], inst.FRAGMENTS),
                  "penalty_A": qubo.smallest_valid_penalty(O, lam)})
        rows.append(m)
    return rows


def all_orderings(O):
    import itertools
    return [inst.order_metrics(p, O) for p in itertools.permutations(range(O.shape[0]))]


def qaoa_runs(O, quick):
    ps = [1, 2] if quick else [1, 2, 3, 4]
    restarts, maxiter = (2, 100) if quick else (5, 200)
    runs, keep = [], {}
    for lam in (3, 0):
        A = qubo.smallest_valid_penalty(O, lam)
        model = qubo.make_model(O, lam, A)
        for p in (ps if lam == 3 else [max(ps)]):
            q = QAOA(model, p)
            stats = q.circuit_stats()
            for objective in ("expectation", "cvar"):
                r = q.fit(objective, alpha=0.1, restarts=restarts, maxiter=maxiter, seed=SEED)
                probs = r.pop("probs")
                feas_probs = {str(model.decode(int(k))): float(probs[k]) for k in np.flatnonzero(model.feasible)}
                r.update({"lambda": lam, "penalty_A": A, "circuit": stats,
                          "optimum_metrics": inst.order_metrics(r["optimum_order"], O),
                          "top_metrics": inst.order_metrics(r["most_probable_feasible_order"], O),
                          "top_is_ground_truth": r["most_probable_feasible_order"] == inst.TRUE_ORDER,
                          "shots_p_find_optimum": 1 - (1 - r["p_optimum"]) ** r["shots"],
                          "feasible_probs": feas_probs})
                runs.append(r)
                print(f"  lam={lam} p={p} {objective:11s} P(opt)={r['p_optimum']:.4f} "
                      f"P(feas)={r['p_feasible']:.4f} E={r['expected_energy']:.2f} "
                      f"({r['seconds']:.0f}s, nfev={r['nfev']})", flush=True)
    return runs


def baseline_runs(O, quick):
    out = {}
    for lam in (3, 0):
        W = inst.junction_weights(O, lam)
        A = qubo.smallest_valid_penalty(O, lam)
        model = qubo.make_model(O, lam, A)
        bf = baselines.brute_force(W)
        opt = bf["order"]
        gr = baselines.greedy(W)
        n_sa, n_sq, n_rand = (20, 10, 20) if quick else (100, 50, 200)
        rng = np.random.default_rng(SEED)
        sa = [baselines.sa_permutation(W, rng) for _ in range(n_sa)]
        sq = [baselines.sa_qubo(model, rng, steps=20000) for _ in range(n_sq)]
        rs = [baselines.random_sampling(model, rng, shots=4096) for _ in range(n_rand)]
        out[str(lam)] = {
            "optimum": opt, "optimum_gain": bf["gain"],
            "brute_force_seconds": bf["seconds"], "num_orderings": math.factorial(O.shape[0]),
            "greedy": {"order": gr["order"], "finds_optimum": gr["order"] == opt, "seconds": gr["seconds"]},
            "sa_permutation": {"runs": n_sa, "success_rate": float(np.mean([r["order"] == opt for r in sa])),
                               "mean_seconds": float(np.mean([r["seconds"] for r in sa]))},
            "sa_qubo": {"runs": n_sq, "success_rate": float(np.mean([r["order"] == opt for r in sq])),
                        "feasible_rate": float(np.mean([r["order"] is not None for r in sq])),
                        "mean_seconds": float(np.mean([r["seconds"] for r in sq]))},
            "random_sampling": {"repeats": n_rand, "shots": 4096,
                                "p_feasible_per_sample": float(np.mean([r["feasible_fraction"] for r in rs])),
                                "p_find_optimum_in_4096": float(np.mean([r["order"] == opt for r in rs])),
                                "analytic_p_find_optimum_in_4096": 1 - (1 - 1 / 2 ** 16) ** 4096},
        }
        print(f"  baselines lam={lam}: greedy ok={out[str(lam)]['greedy']['finds_optimum']} "
              f"SA-perm={out[str(lam)]['sa_permutation']['success_rate']:.2f} "
              f"SA-QUBO={out[str(lam)]['sa_qubo']['success_rate']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    O = inst.overlap_matrix(inst.FRAGMENTS)

    print("[1/3] lambda sweep (exact)")
    sweep = pareto_sweep(O, [0, 0.5, 1, 1.5, 2, 3, 4])  # lam=1 is an exact tie between the two orderings
    for r in sweep:
        print(f"  lam={r['lambda']}: order={r['order']} overlap={r['total_overlap']} "
              f"weak={r['weak_junctions']} truth={r['matches_ground_truth']}")
    print("[2/3] classical baselines")
    base = baseline_runs(O, args.quick)
    print("[3/3] QAOA (Qiskit Aer statevector)")
    runs = qaoa_runs(O, args.quick)

    results = {
        "instance": {"genome": inst.GENOME, "fragments": inst.FRAGMENTS, "true_order": inst.TRUE_ORDER,
                     "tau": inst.TAU, "overlap_matrix": O, "num_qubits": len(inst.FRAGMENTS) ** 2},
        "lambda_sweep": sweep,
        "all_orderings": all_orderings(O),
        "baselines": base,
        "qaoa": runs,
        "quick_mode": args.quick,
        "total_seconds": time.perf_counter() - t0,
    }
    (OUT / "results.json").write_text(json.dumps(jsonable(results), indent=2))
    print(f"saved results/results.json  ({results['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
