"""Run every experiment and write results/results.json.

    python run_experiments.py            # full run   (~3-4 min on a laptop)
    python run_experiments.py --quick    # smoke run  (< 60 s), same pipeline, fewer trials
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
from qiskit_aer import AerSimulator

import baselines as bl
from grover import (LAMBDA, circuit_stats, default_patience, grover_adaptive_search, simulate_p_marked,
                    theory_p_marked)
from plans import (DEFAULTS, PlanProblem, bitstring, default_problem, pareto_front,
                   weight_sweep)

RESULTS = Path(__file__).parent / "results"
W_O_SWEEP = (0.0, 0.5, 1.0, 2.0, 5.0)
GAS_SEED0, SA_SEED0, SCALE_SEED0, RS_SEED = 1000, 2000, 3000, 7


def summarise_gas(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["found_opt"]]
    to_opt = np.array([r["oracle_calls_to_opt"] for r in ok], dtype=float)
    ev_opt = np.array([r["evaluations_to_opt"] for r in ok], dtype=float)
    tot = np.array([r["oracle_calls"] for r in runs], dtype=float)
    ev = np.array([r["evaluations"] for r in runs], dtype=float)
    q = lambda a, f: float(f(a)) if a.size else None
    return dict(trials=len(runs), success_rate=len(ok) / len(runs),
                oracle_calls_to_opt=dict(mean=q(to_opt, np.mean), median=q(to_opt, np.median),
                                         p90=q(to_opt, lambda a: np.percentile(a, 90)),
                                         max=q(to_opt, np.max)),
                evaluations_to_opt=dict(mean=q(ev_opt, np.mean), median=q(ev_opt, np.median)),
                oracle_calls_total=dict(mean=float(tot.mean()), median=float(np.median(tot))),
                evaluations_total=dict(mean=float(ev.mean()), median=float(np.median(ev))))


def run_gas(problem: PlanProblem, seeds, sim, patience: int) -> list[dict]:
    return [grover_adaptive_search(problem, s, sim=sim, patience=patience) for s in seeds]


def main(quick: bool) -> dict:
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    sim = AerSimulator(method="statevector")
    n_gas, n_sa, n_abl = (12, 20, 8) if quick else (100, 100, 50)
    # trials per problem size for the scaling study (n = 12 reuses the main GAS runs)
    scale_trials = {8: 6, 10: 6, 12: 12} if quick else {8: 60, 10: 60, 12: 100, 14: 20}

    p = default_problem()
    n, N = p.n, p.N
    tab, F, feas = p.table(), p.costs(), p.feasible_mask()
    feas_c1 = p.feasible_mask(use_c2=False)
    opt = p.optimum()
    P = default_patience(N)
    print(f"[instance] n={n} beams, N={N} plans, C1-feasible={int(feas_c1.sum())}, "
          f"C1&C2-feasible={int(feas.sum())}")

    # ---------------- exact optimum, sanity checks, trade-off sweep
    sweep = weight_sweep(p, W_O_SWEEP)
    for s in sweep:
        g = bl.greedy(PlanProblem(p.phantom, w_O=s["w_O"]))
        s["greedy_beams"], s["greedy_found_opt"] = p.beams(g["index"]), g["found_opt"]
    sanity = dict(unique_optimum=opt["unique"], feasible_nonempty=bool(feas.any()),
                  optimum_changes_with_w_O=len({s["index"] for s in sweep}) > 1,
                  all_sweep_optima_unique=all(s["unique"] for s in sweep),
                  c1_count_is_794=int(feas_c1.sum()) == 794)
    print(f"[exact] optimum beams={p.beams(opt['index'])} F={opt['cost']:.6f}  sanity={sanity}")

    fi = np.flatnonzero(feas)
    front = fi[pareto_front(tab["tumour_err"][fi], tab["oar_mean"][fi])]

    # ---------------- classical baselines
    bf_all, bf_feas = bl.brute_force(p), bl.brute_force(p, feasible_only=True)
    rs = bl.random_search(p, trials=1000, seed=RS_SEED)
    gr = bl.greedy(p)
    sa_runs = [bl.simulated_annealing(p, SA_SEED0 + s, steps=300) for s in range(n_sa)]
    sa_ok = [r for r in sa_runs if r["found_opt"]]
    nn = bl.nnls_bound(p)
    print(f"[classical] brute force {bf_all['seconds']*1e3:.1f} ms, greedy found_opt={gr['found_opt']} "
          f"({gr['evaluations']} evals), SA success={len(sa_ok)}/{n_sa}, NNLS bound={nn['bound']:.5f}")

    # ---------------- Grover: theory vs statevector simulation
    sorted_feas_F = np.sort(F[feas])
    check = []
    for M in (1, 4, 16):
        y = float(sorted_feas_F[M])                 # exactly M feasible plans have F < y
        marked = feas & (F < y)
        r_vals = list(range(0, 61 if not quick else 31))
        sim_p = simulate_p_marked(n, marked, r_vals, sim)
        th_p = theory_p_marked(int(marked.sum()), N, r_vals)
        check.append(dict(M=int(marked.sum()), threshold=y, r=r_vals, theory=th_p.tolist(),
                          simulated=sim_p.tolist(), max_abs_error=float(np.max(np.abs(sim_p - th_p))),
                          r_opt=int(round(math.pi / (4 * math.asin(math.sqrt(M / N))) - 0.5))))
    print("[grover check] max |sim - theory| =", max(c["max_abs_error"] for c in check))
    stats = circuit_stats(p)
    print(f"[circuit] logical depth {stats['logical_depth']}, transpiled depth {stats['transpiled_depth']}, "
          f"CX/iteration {stats['cx_per_iteration']}")

    # ---------------- GAS main study
    t0 = time.perf_counter()
    gas_runs = run_gas(p, range(GAS_SEED0, GAS_SEED0 + n_gas), sim, P)
    gas_time = time.perf_counter() - t0
    gas = summarise_gas(gas_runs)
    gas.update(patience=P, budget=int(30 * math.sqrt(N)), lam=LAMBDA, seconds=gas_time,
               per_trial=[{k: r[k] for k in ("seed", "found_opt", "best", "best_cost", "oracle_calls",
                                             "evaluations", "rounds", "oracle_calls_to_opt",
                                             "evaluations_to_opt")} for r in gas_runs],
               traces=[dict(seed=r["seed"], calls=[h["calls"] for h in r["history"]],
                            y=[h["y"] for h in r["history"]]) for r in gas_runs[:6]],
               demo_history=gas_runs[0]["history"])
    print(f"[GAS] success {gas['success_rate']:.0%}, mean oracle calls to optimum "
          f"{gas['oracle_calls_to_opt']['mean']:.1f}, total {gas['oracle_calls_total']['mean']:.1f} "
          f"({gas_time:.1f}s)")

    # ---------------- termination-rule ablation
    ablation = []
    for pat in (15, 20, P):
        runs = gas_runs[:n_abl] if pat == P else run_gas(p, range(GAS_SEED0, GAS_SEED0 + n_abl), sim, pat)
        s = summarise_gas(runs)
        ablation.append(dict(patience=pat, trials=len(runs), success_rate=s["success_rate"],
                             mean_calls_to_opt=s["oracle_calls_to_opt"]["mean"],
                             mean_calls_total=s["oracle_calls_total"]["mean"]))
    print("[ablation]", [(a["patience"], round(a["success_rate"], 2)) for a in ablation])

    # ---------------- scaling with problem size
    scaling = []
    for nb, n_trials in scale_trials.items():
        q = p if nb == n else default_problem(nb)
        Nq, Mq = q.N, int(q.feasible_mask().sum())
        if nb == n:
            runs = gas_runs[:n_trials]
        else:
            runs = run_gas(q, range(SCALE_SEED0, SCALE_SEED0 + n_trials), sim, default_patience(Nq))
        s = summarise_gas(runs)
        scaling.append(dict(n=nb, N=Nq, feasible=Mq, unique=q.optimum()["unique"], trials=len(runs),
                            sqrt_N=math.sqrt(Nq), random_all_expected=(Nq + 1) / 2,
                            random_feasible_expected=(Mq + 1) / 2, brute_force=Nq,
                            gas_success=s["success_rate"], gas_calls_to_opt=s["oracle_calls_to_opt"]["mean"],
                            gas_calls_total=s["oracle_calls_total"]["mean"],
                            gas_evals_total=s["evaluations_total"]["mean"]))
        print(f"[scaling] n={nb} N={Nq} feasible={Mq} GAS calls->opt {s['oracle_calls_to_opt']['mean']}"
              f" success {s['success_rate']:.0%}")

    t_opt = p.terms(tab["bits"][opt["index"]])
    out = dict(
        config=dict(quick=quick, weights=dict(w_T=p.w_T, w_O=p.w_O, w_N=p.w_N), k=p.k,
                    oar_max=p.oar_max, w_O_sweep=list(W_O_SWEEP), seeds=dict(gas=GAS_SEED0, sa=SA_SEED0,
                    scaling=SCALE_SEED0, random_search=RS_SEED), gas_trials=n_gas, sa_trials=n_sa,
                    versions=dict(python=platform.python_version(), qiskit=qiskit.__version__,
                                  qiskit_aer=qiskit_aer.__version__, numpy=np.__version__)),
        instance=dict(n=n, N=N, angles_deg=p.phantom.angles_deg.tolist(), **p.phantom.meta,
                      feasible_c1=int(feas_c1.sum()), feasible=int(feas.sum())),
        sanity=sanity,
        optimum=dict(index=opt["index"], bitstring=bitstring(opt["index"], n), beams=p.beams(opt["index"]),
                     cost=opt["cost"], second_index=opt["second_index"], second_beams=p.beams(opt["second_index"]),
                     second_cost=opt["second_cost"], gap=opt["gap"], **t_opt),
        sweep=sweep,
        pareto=dict(feasible_index=fi.tolist(), tumour_err=tab["tumour_err"][fi].tolist(),
                    oar_mean=tab["oar_mean"][fi].tolist(), n_beams=tab["n_beams"][fi].tolist(),
                    front_index=front.tolist()),
        baselines=dict(
            brute_force_all=bf_all, brute_force_feasible=bf_feas,
            random_search=rs, greedy=gr, nnls=nn,
            simulated_annealing=dict(trials=n_sa, steps=300, success_rate=len(sa_ok) / n_sa,
                                     mean_evals_to_opt=float(np.mean([r["evaluations_to_opt"] for r in sa_ok]))
                                     if sa_ok else None, evaluations_per_run=301)),
        grover_check=check, circuit_stats=stats, gas=gas, ablation=ablation, scaling=scaling,
        runtime_seconds=time.perf_counter() - t_start)
    path = RESULTS / ("results_quick.json" if quick else "results.json")
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[done] wrote {path.relative_to(Path(__file__).parent)} in {out['runtime_seconds']:.1f}s")
    print_demo(out)
    return out


def print_demo(res: dict) -> None:
    """Human-readable walk-through of one seeded GAS run (used in the live demo)."""
    h, opt = res["gas"]["demo_history"], res["optimum"]
    n = res["instance"]["n"]
    ang = res["instance"]["angles_deg"]
    beams = lambda i: [int(a) for b, a in enumerate(ang) if i >> b & 1]
    print(f"\nGAS demo, seed {res['config']['seeds']['gas']}: rounds where the threshold y improved "
          f"(exact optimum {[int(b) for b in opt['beams']]}, F={opt['cost']:.5f})")
    print(" round  Grover its r  oracle calls so far  measured |x> (qubit 11..0)  beams              new y")
    for e in h:
        if e["improved"]:
            tag = "  <- random feasible start" if e["round"] == 0 else ("  <- EXACT OPTIMUM" if e["x"] == opt["index"] else "")
            print(f" {e['round']:>5} {e['r']:>13} {e['calls']:>20}  {bitstring(e['x'], n):>26}  "
                  f"{str(beams(e['x'])):<18} {e['y']:.5f}{tag}")
    last = h[-1]
    print(f" stopped after {last['round']} rounds / {last['calls']} oracle calls "
          f"({res['gas']['patience']} consecutive rounds without improvement)")
    g = res["gas"]
    print(f"\nSummary over {g['trials']} seeded GAS runs: success {g['success_rate']:.0%}, "
          f"oracle calls to optimum mean {g['oracle_calls_to_opt']['mean']:.1f} vs "
          f"random search {res['baselines']['random_search']['all_states']['analytic']} (all plans) / "
          f"{res['baselines']['random_search']['feasible_only']['analytic']} (feasible only)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (< 60 s)")
    main(ap.parse_args().quick)
