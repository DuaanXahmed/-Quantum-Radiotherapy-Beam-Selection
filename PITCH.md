# 3-minute pitch and demo script

Target: 2:45 spoken, which leaves a 15-second buffer under the 3:00 limit. Read at a normal pace.

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:20 | Title / README | "When a cancer or infection patient's DNA is sequenced, the machine gives back thousands of short shuffled pieces. They must be put back in order before doctors can look for disease-causing changes, and that has to happen fast. We framed this ordering as a quantum optimisation problem in the healthcare domain." |
| 0:20 to 0:50 | `results/fig1_tradeoff.png` | "There are two goals that compete. Contiguity: neighbours should overlap as much as possible. Reliability: avoid weak junctions, where pieces overlap by fewer than 3 bases, because repeated DNA makes short overlaps look convincing. On our instance, maximum overlap alone (orange) picks the wrong genome. Penalising weak junctions with a weight lambda (blue) gives the true genome. Lambda is the dial between the two goals." |
| 0:50 to 1:15 | `results/fig4_architecture.png` | "Each fragment-position pair is a binary variable, so 4 fragments give 16 qubits. One-hot penalties enforce that each fragment is used once and each position is filled once. We convert the QUBO to an Ising Hamiltonian and run QAOA in Qiskit on the Aer simulator. COBYLA tunes the angles using expected energy or CVaR. We then sample, throw away invalid orderings, and keep the best valid one." |
| 1:15 to 2:05 | Terminal: `python demo.py` | Narrate each step as it prints: "Here are the four reads and their overlap matrix... the exact solver shows the trade-off... here is the 16-qubit Hamiltonian... the QAOA circuit at three layers... 4,096 measured shots... about 9 percent of shots are valid orderings, roughly 250 times the random rate... and the best one assembles to exactly the hidden genome, matching the brute-force optimum." |
| 2:05 to 2:30 | `results/fig2_qaoa_vs_p.png` and the baselines table in the report | "Against baselines: the optimum appears in 1.7 percent of shots, about 1,100 times random. CVaR beats expected energy at every depth. Brute force, greedy and simulated annealing also find it, and we check every answer against the exact solver." |
| 2:30 to 2:50 | Limitations section of the report | "We are honest about the limits. No quantum advantage: 24 orderings is microseconds for brute force. Ideal simulator only, no noise. The encoding needs N squared qubits. The instance and lambda were tuned by hand, and deeper circuits were not always better, which points to the optimiser. What we deliver is a complete, tested path from a clinical trade-off to a working Qiskit circuit." |
| 2:50 | GitHub page | "Code, tests and report are on GitHub. Thank you." |

## Recording checklist

1. Open Terminal, `cd` into the repo, `source .venv/bin/activate`, and run `python demo.py` once so it is fast on camera.
2. Make the terminal font large (Cmd and + a few times).
3. Record with QuickTime (File, New Screen Recording) or press Cmd+Shift+5.
4. Upload to Google Drive. Set sharing to **Anyone with the link: Viewer**. Open the link in a private window to check it.

## Likely judge questions

- **Why QAOA and not Grover or VQE?** The problem is a QUBO, which maps directly to an Ising Hamiltonian and QAOA's RZ/RZZ gates. QAOA returns a distribution we can sample, which suits picking the best valid ordering.
- **Is the quantum part really doing work?** Yes. The final answer is picked from the circuit's measured shots. The circuit raises the probability of a valid ordering about 250-fold and of the optimum about 1,100-fold over random bitstrings.
- **How did you choose the penalty A?** As the smallest value from a list for which the exact QUBO ground state is the valid optimum, which we check by enumerating all 2^16 states. A unit test confirms this for lambda from 0 to 5.
- **Why CVaR?** It optimises the best 10% of shots instead of the average, and we only keep the best shot. It gave a higher probability of the optimum at every depth.
- **Why is p = 4 worse than p = 3?** A deeper circuit can represent everything a shallower one can, so this comes from the optimiser: a capped COBYLA budget and local minima.
- **How would it scale?** Badly with this encoding: N^2 qubits, and the valid fraction N!/2^(N^2) collapses. Real assemblers use de Bruijn or string graphs. Next steps would be constraint-preserving XY mixers and noise-aware runs.
- **What does lambda = 1 mean?** It is an exact tie between the two orderings (11 vs 12 minus 1), so any lambda above 1 selects the true genome.
