# Quantum-assisted DNA fragment ordering with QAOA (Qiskit)

**Qiskit Fall Fest 2026 Hackathon: Healthcare track**

Ordering DNA fragments (reads) into a genome is a core step in time-critical genomic
diagnostics such as oncology panels. The two competing goals are:

- **Contiguity:** maximise the total overlap between neighbouring fragments.
- **Reliability:** minimise *weak junctions* (overlap shorter than `tau = 3` bases). Repeats make weak junctions
  the places where assemblers join the wrong fragments.

We formulate this as a QUBO on `N^2 = 16` qubits (`N = 4` fragments), solve it with a
Qiskit **QAOA** circuit on the Aer simulator, and compare against an exact brute-force solver
and three classical heuristics.

> **Honest summary.** The quantum part really runs and really works: QAOA puts about
> 250x (valid orderings) and about 1,100x (the optimal ordering) more probability mass than random
> sampling, and finds the exact optimum in 4,096 shots. But `N = 4` has only 24 candidate
> orderings, so a classical solver is exact in microseconds. **We do not claim a quantum
> advantage.** See "Limitations" below and the report.

## Repository layout

| File | Purpose |
|---|---|
| `PROBLEM_STATEMENT.md` | Problem statement: objective, constraints, users, instance, results, limitations |
| `demo.py` | Step-by-step live demo used in the video (reads -> QUBO -> QAOA shots -> recovered genome) |
| `instance.py` | Toy instance (4 fragments, ground-truth genome), overlap matrix, weak-junction scoring |
| `qubo.py` | One-hot QUBO builder, QUBO to Ising conversion, penalty-weight selection, exact energies |
| `qaoa.py` | Qiskit QAOA circuit (RZ/RZZ cost layer, RX mixer), COBYLA training (expected energy or CVaR), evaluation |
| `baselines.py` | Brute force (exact), greedy, simulated annealing on permutations, simulated annealing on the same QUBO, random sampling |
| `run_experiments.py` | Runs every experiment in the report and writes `results/results.json` |
| `plots.py` | Regenerates all figures in `results/` from `results.json` |
| `tests/test_formulation.py` | 7 unit tests (formulation correctness, Qiskit Hamiltonian = QUBO energy) |
| `results/` | `results.json`, `run_log.txt`, `fig1..fig4*.png` (fig4 is the architecture diagram) |

## Setup

Python 3.10 or newer (tested on Python 3.12 with Qiskit 2.x and Qiskit Aer).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
pytest -q tests                    # 7 tests, about 1 s
python demo.py                     # 10-second end-to-end demo (add --pause to step through it)
python run_experiments.py --quick  # smoke test, about 30 s
python run_experiments.py          # full run used in the report, about 5 min on a laptop CPU
python plots.py                    # rebuild the figures in results/
```

All randomness is seeded (`SEED = 2026`), so a full run reproduces the reported numbers up to
floating-point differences between machines.

## Problem formulation (short version)

- **Instance:** genome `TACACCATACCAT`, shuffled into 4 reads of length 6. The ground truth is used *only* for scoring.
- **Variables:** `x[i,t] = 1` if fragment `i` is placed at position `t` (qubit index `i*N + t`).
- **Constraints:** each fragment appears once, and each position holds one fragment (one-hot penalties with weight `A`).
- **Objective:** maximise `sum_t W[i,j]` over adjacent placements, with `W_ij = o_ij - lambda * [o_ij < tau]`.
  `lambda = 0` is pure max-overlap and returns a *misassembly* (overlap 12, one weak junction).
  `lambda > 1` recovers the true genome (overlap 11, no weak junction). `lambda = 1` is an exact tie.
- **Quantum algorithm:** QAOA with depth `p = 1..4`, started from a linear-ramp initialisation, trained with COBYLA
  on either the expected energy or CVaR (alpha = 0.1).

## Headline results (full run, ideal simulator)

| Setting (lambda = 3) | P(valid ordering) | P(optimum) | Uniform random |
|---|---|---|---|
| QAOA p = 1, expected energy | 2.2% | 0.21% | 0.037% / 0.0015% |
| QAOA p = 3, CVaR | 9.1% | 1.69% | 0.037% / 0.0015% |
| QAOA p = 4, CVaR | 7.6% | 1.47% | 0.037% / 0.0015% |

Full tables, plots and discussion: see the report (`DNA_QAOA_Report.docx`).

## Limitations (details in the report)

- **Tiny instance:** `N = 4` is brute-forceable (24 orderings). Greedy finds the optimum, and simulated annealing succeeds on 96 to 100% of runs. No advantage is claimed.
- **Simulator only, no noise.** On real hardware the depth-248 / 672-CX circuits (p = 4) would be dominated by noise.
- **Qubit scaling:** the one-hot encoding needs `N^2` qubits and the valid fraction `N!/2^(N^2)` shrinks fast. Real assemblers use de Bruijn / string graphs with millions of reads.
- **Hand-picked instance:** it was found by random search so that the repeat-induced misassembly appears. `lambda`, `tau` and the penalty `A` were tuned on this one instance.
- **Optimiser limits:** results are not monotone in `p` (p = 4 is slightly worse than p = 3), which points to local minima and a capped COBYLA budget rather than a physical effect.

## Credits and references

All code in this repository was written for this hackathon; no tutorial or third-party code was copied. We use the
Qiskit and Qiskit Aer libraries, NumPy, SciPy and Matplotlib. Algorithms are implemented from the original papers:

- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm* (2014), arXiv:1411.4028
- Barkoutsos et al., *Improving Variational Quantum Optimization using CVaR* (Quantum 4, 256, 2020)
- Sack and Serbyn, *Quantum annealing initialization of the QAOA* (Quantum 5, 491, 2021)
- Lucas, *Ising formulations of many NP problems* (Front. Phys. 2, 5, 2014)
