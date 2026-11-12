# Quantum-Assisted DNA Fragment Ordering for Time-Critical Genomic Diagnostics

**Domain:** Healthcare (clinical genomics) · **Algorithm:** QAOA in Qiskit · **Scale:** 4 fragments, 16 qubits

## The problem in one sentence
How do we put shuffled pieces of DNA back in the right order, quickly, so the sequence is both well-connected and trustworthy?

## Background
- To read a patient's DNA, a sequencer produces many short pieces called **fragments** (reads). They come out shuffled.
- Fragments must be re-ordered into one sequence before doctors can look for disease-causing changes.
- This is urgent in time-critical cases such as cancer and infection diagnostics.
- **Neighbouring fragments overlap.** The end of one matches the start of the next, and that overlap is the clue for ordering them.

## Why it is hard: two goals that compete
- **Contiguity:** maximise the total overlap between neighbours, which gives a compact, well-supported sequence.
- **Reliability:** avoid **weak junctions**, meaning neighbours that overlap by fewer than 3 bases.
- **The conflict:** repeated DNA makes short overlaps look convincing, so the ordering with the most overlap can still be wrong.
- **What goes wrong:** a wrong junction can hide or invent a genetic variant.

## Who it serves
- Laboratory bioinformaticians who assemble the data.
- Clinicians who depend on their reports.
- Patients whose treatment depends on an accurate sequence.

## What we decide
- The position of each fragment in the final sequence: `x[i,t] = 1` if fragment `i` sits at position `t`.
- **Constraints:** each fragment is used exactly once, and each position holds exactly one fragment.

## Objective
- Maximise `W = overlap − λ × (number of weak junctions)`, i.e. per junction `W_ij = o_ij − λ·[o_ij < 3]`.
- **λ is a dial.** λ = 0 means maximum overlap only, and a higher λ means "avoid weak junctions".
- Sweeping λ shows the trade-off between the two goals.

## Our test instance
- Hidden genome: `TACACCATACCAT` (13 bases), cut into **4 shuffled fragments of 6 bases**.
- That gives **24 possible orderings**.
- **Maximum overlap alone (λ = 0)** picks a wrong ordering: overlap 12, with one weak junction.
- **Penalising weak junctions (λ > 1)** picks the true genome: overlap 11, with no weak junction. (λ = 1 is an exact tie.)
- The true answer is used only to score results, never by the solvers.

## How the quantum computer is used
- The problem is written as a **QUBO**, a binary optimisation problem, with 16 variables.
- It is converted to a 16-qubit Ising Hamiltonian.
- We solve it with **QAOA** in **Qiskit** on the Aer simulator.
- A classical optimiser (COBYLA) tunes the circuit, using either expected energy or CVaR.
- We sample the circuit, discard invalid orderings, and keep the best valid one.

## How we judge success
- Compare against an **exact brute-force solver**, greedy, simulated annealing and random sampling.
- **Metrics:** probability of a valid ordering, probability of the optimal ordering, and whether the true genome is recovered.

## Results (λ = 3, ideal simulator)
- About **9.1%** of shots are valid orderings, roughly **250×** the random rate.
- The optimum appears in **1.69%** of shots, roughly **1,100×** the random rate.
- QAOA found the exact optimum within 4,096 shots essentially every time.

## Honest limitations
- **No quantum advantage claimed.** With 4 fragments, brute force solves it in microseconds.
- Simulator only, with no noise or hardware.
- The encoding needs N² qubits, so it scales poorly.
- The instance was hand-picked, and λ, the overlap threshold and the penalty weight were tuned on it.
- Results are not monotone in circuit depth, which points to the optimiser.
- The biology is simplified: no sequencing errors, reverse strands or coverage.
