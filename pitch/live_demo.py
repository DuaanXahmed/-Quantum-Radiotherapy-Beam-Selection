"""Plain-English live demo for the pitch video. Uses the project modules; changes nothing.

    .venv/bin/python pitch/live_demo.py          # paced for narration (~30 s)
    .venv/bin/python pitch/live_demo.py --fast   # no pauses
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import baselines as bl  # noqa: E402
from grover import grover_adaptive_search  # noqa: E402
from plans import default_problem, index_to_bits  # noqa: E402

FAST = "--fast" in sys.argv
B, G, Y, C, DIM, X = "\033[1m", "\033[32m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"


def say(text="", pause=0.7):
    print(text, flush=True)
    if not FAST:
        time.sleep(pause)


def plan(p, i):
    return ", ".join(f"{int(a)}°" for a in p.beams(i)) or "no beams"


p = default_problem()
line = "=" * 74

say(f"\n{B}{line}\n  LIVE DEMO · Quantum search for radiotherapy beam angles\n{line}{X}", 1.2)

say(f"\n{B}THE PATIENT{X} {DIM}(synthetic model, no real patient data){X}")
say("  • 12 possible beam angles: 0°, 30°, 60°, ... 330°")
say("  • Rules: at most 4 beams, and the nearby organ may get at most 60% of the dose")
say(f"  • {p.N:,} possible plans, but only {int(p.feasible_mask().sum())} follow the rules", 1.5)

say(f"\n{B}STEP 1 · Classical answer key{X}: check every plan one by one (brute force)")
bf = bl.brute_force(p)
say(f"  Checked {p.N:,} plans → best plan: {B}{plan(p, bf['index'])}{X}   "
    f"score {bf['cost']:.3f}  {DIM}(lower score = better plan){X}", 1.8)

say(f"\n{B}STEP 2 · Quantum search{X} on a 12-qubit simulator (Qiskit)")
say("  Each qubit = one beam. The quantum circuit boosts every plan that")
say("  follows the rules AND beats the best plan found so far, then we measure.")
say(f"  {DIM}running...{X}", 0.3)
run = grover_adaptive_search(p, seed=1000)
say("")
say(f"  {B}{'Quantum queries':>15}   {'Plan suggested by the quantum search':<34} {'Score':>6}{X}")
say(f"  {'-' * 15}   {'-' * 34} {'-' * 6}", 0.4)
for h in run["history"]:
    if not h["improved"]:
        continue
    if h["round"] == 0:
        tag = f"{DIM}random starting plan{X}"
    elif h["x"] == bf["index"]:
        tag = f"{G}{B}★ BEST PLAN{X}"
    else:
        tag = f"{Y}↓ better{X}"
    say(f"  {h['calls']:>15}   {plan(p, h['x']):<34} {h['y']:>6.3f}  {tag}", 1.1)

say(f"\n{B}STEP 3 · Check{X}", 0.8)
same = run["best"] == bf["index"]
say(f"  Quantum answer:   {B}{plan(p, run['best'])}{X}")
say(f"  Classical answer: {B}{plan(p, bf['index'])}{X}   "
    + (f"{G}{B}✓ identical{X}" if same else f"{Y}✗ different{X}"), 1.2)
t = p.terms(index_to_bits(run["best"], p.n))
say(f"  Tumour: every part gets ≥ {100 * t['tumour_min']:.0f}% of the prescribed dose")
say(f"  Organ at risk: peak dose {100 * t['oar_maxdose']:.0f}% (limit 60%) · beams used: {t['n_beams']} (limit 4)", 1.2)

res = json.loads((ROOT / "results" / "results.json").read_text())["gas"]
avg = res["oracle_calls_to_opt"]["mean"]
say(f"\n{B}HOW FAST?{X}")
say(f"  This run reached the best plan after {run['oracle_calls_to_opt']} quantum queries.")
say(f"  Over {res['trials']} test runs: found the best plan {100 * res['success_rate']:.0f}% of the time, "
    f"{B}{avg:.0f}{X} queries on average.")
say(f"  Random guessing would need about {B}2,048{X} tries → {C}{B}{2048.5 / avg:.0f}× fewer{X} with quantum search.\n")
