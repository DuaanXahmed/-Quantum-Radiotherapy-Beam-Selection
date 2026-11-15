# Pitch video script: Quantum Radiotherapy Beam Selection (≤ 3:00)

Slides: `pitch/Radiotherapy_GAS_Pitch.pptx` (7 slides; this script is also in each slide's speaker notes)

## Before you record
1. Open the deck in PowerPoint and start the slideshow (or use Presenter View to see the notes).
2. Open a Terminal, enlarge the font with **Cmd +**, and run:
   ```
   cd ~/Desktop/"Radio Therapy Quantum"
   clear
   ```
   (Keep the quotes: the folder name has spaces. Skip `source .venv/bin/activate`; it broke when the folder was renamed.)
3. Press **Cmd+Shift+5**, choose **Record Entire Screen**, set **Options → Microphone**, then click **Record**.

---

## Slide 1: Title (0:00–0:03)
> "Hi, this is our Qiskit Fall Fest project in healthcare: quantum search for radiotherapy beam selection."

## Slide 2: The problem (0:03–0:25)
> "About half of all cancer patients get radiotherapy. For every patient, a medical physicist has to decide which
> angles the radiation beams come from. Each beam hits the tumour, but it also passes through healthy tissue, and some
> angles go straight through a critical organ. Each extra beam costs treatment time, so clinics cap the number of
> beams. That's three competing goals and one limited resource."

## Slide 3: Formulation and trade-off (0:25–1:00)
> "We model a 2-D phantom: a tumour, an organ at risk just below it, and normal tissue. There are 12 candidate
> angles and at most 4 beams. One binary variable per beam gives 4,096 plans. The cost adds tumour dose error, organ
> dose and normal-tissue dose. There are two hard constraints: at most four beams, and the organ's maximum dose must
> stay below 0.6. That second one is not quadratic. Only 204 plans are feasible, and this plot shows every one. As we
> weight the organ more, the best plan changes: from beams through the organ, to beams around it, to a 3-beam plan."

## Slide 4: Quantum solution (1:00–1:35)
> "Each qubit is one beam, so each basis state of 12 qubits *is* a treatment plan. Our oracle flips the phase of
> every plan that is feasible and better than the best so far. Because it's a yes/no test, it enforces the beam limit
> and the max-dose rule exactly: no penalty tuning, no slack qubits. QAOA can't do that. Grover amplification makes
> those plans likely to be measured. We measure one shot, check it classically, lower the threshold and repeat. It's
> all written from scratch in Qiskit, and the simulation matches Grover's theory to 10⁻¹⁴."

## Slide 5: Live demo (1:35–2:10). Switch to the terminal
Type this and press Enter (it prints step by step, about 30 s, so talk along with it):
```
.venv/bin/python pitch/live_demo.py
```
**"THE PATIENT"** appears:
> "Here's the live demo. Our patient has 12 possible beam angles. We can use at most 4 beams, and the organ next to
> the tumour has a dose limit. That's 4,096 possible plans, and only 204 follow the rules."

**"STEP 1"** appears:
> "First, the classical answer key: checking every plan gives the best one, 60, 90, 240 and 270 degrees."

**"STEP 2"** rows appear one by one:
> "Now the quantum search, on a 12-qubit simulator. Each row is a plan the quantum search found that beats the
> previous one. The score keeps dropping... until it lands on the best plan."

**"STEP 3"** and **"HOW FAST?"** appear:
> "The quantum answer matches the classical answer key exactly. The tumour gets its full dose, and the organ stays at
> 52%, under its 60% limit. On average the quantum search needs 126 queries; random guessing needs about 2,048."

Switch back to the slideshow.

*(Optional, only if you have time: `.venv/bin/python -m pytest -q` shows the 9 correctness tests passing.)*

## Slide 6: Results (2:10–2:35)
> "Over 100 seeded runs, Grover Adaptive Search found the exact optimum every time, using 126 oracle calls on
> average. Unstructured classical search needs about 2,048 evaluations, 16 times more. From 8 to 14 beams that gap
> grows from 7 to 36 times, close to the square-root-of-N line."

## Slide 7: Limitations and message (2:35–3:00)
> "To be honest: classical methods that use the structure, like greedy search, annealing, or listing only the feasible
> plans, are as fast or faster at this size. Our advantage is in queries, not wall-clock time. Our oracle is a
> precomputed table, and a real one would need fault-tolerant hardware. Our message: quantum search is a natural
> engine for constrained treatment planning, because any clinical rule becomes part of the oracle. We built a
> verified, reproducible blueprint for that future. Thank you."

---

## After recording
Stop the recording from the menu bar; the file lands on your Desktop. Upload it to Google Drive, set sharing to
**Anyone with the link**, and submit that link.
