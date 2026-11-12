# Problem statement

**Quantum Grover Adaptive Search for Radiotherapy Beam-Angle Selection under a Beam Limit**
Qiskit Fall Fest 2026, The Quantum Optimization Challenge. Domain: **Healthcare**

## Motivation
About half of all cancer patients receive radiotherapy. For each patient, a medical physicist or radiation oncologist
chooses the **gantry angles** that deliver the radiation beams. Each beam adds dose to the tumour, but it also irradiates
healthy tissue on its way in and out. Each extra beam also adds treatment time, machine time and QA work, so clinics
**limit the number of beams**. Many clinics still pick angles from templates or experience. Picking the best set is a
non-convex combinatorial problem: n candidate angles give 2ⁿ candidate plans.

**Users:** medical physicists and radiation oncologists. **Beneficiaries:** patients, who get better organ sparing and
shorter treatments.

## Decision variables
x_a ∈ {0, 1} for each of 12 candidate angles (0°, 30°, …, 330°), where x_a = 1 means beam a is used. That gives 2¹² = 4096 plans,
mapped one-to-one onto 12 qubits.

## Objective (competing goals)
Minimise

F(x) = w_T · mean_tumour (d_v − 1)² + w_O · mean_OAR d_v² + w_N · mean_normal d_v², where d_v(x) = Σ_a x_a D[a, v].

The three terms are tumour coverage (prescription 1.0), sparing of the organ at risk (OAR), and normal-tissue sparing.
Default weights are w_T = 1, w_O = 1, w_N = 0.05. Sweeping w_O ∈ {0, 0.5, 1, 2, 5} traces the trade-off.

## Constraints (limited resources and safety)
- **C1 beam limit:** Σ_a x_a ≤ 4. This leaves 794 of 4096 plans.
- **C2 OAR maximum dose:** max over OAR voxels of d_v ≤ 0.6. This is a dose-volume-style limit and is not quadratic.
  With C1 it leaves 204 feasible plans.

## Test instance
A synthetic 2-D phantom (21×21 voxels; tumour of radius 3 at the centre; OAR of radius 2 about 6 voxels below it), 7-voxel-wide
parallel beams with exponential depth attenuation, and a precomputed dose matrix D. It uses no patient data, and all seeds are fixed.

## Measurable outcomes
- Exact optimum (brute force) and whether the quantum method finds it (success rate over 100 seeds).
- Oracle calls (quantum) vs cost evaluations (classical) needed to reach the optimum.
- Tumour error, OAR mean and max dose, and beams used for every plan on the trade-off front.

## Approach
Grover Adaptive Search (Dürr–Høyer minimum finding with the BBHT schedule), written from scratch in Qiskit. A
truth-table phase oracle marks plans that are *feasible and better than the current best*, so both constraints are
enforced exactly, with no penalty terms and no slack qubits. Results are checked against brute force, random search,
greedy selection, simulated annealing and an NNLS lower bound.
