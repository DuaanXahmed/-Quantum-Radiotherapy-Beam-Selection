"""Build pitch/Radiotherapy_GAS_Pitch.pptx (7 slides, speaker notes = script). Numbers from results.json."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
r = json.loads((RES / "results.json").read_text())
g, rs, sc = r["gas"], r["baselines"]["random_search"], r["scaling"]
mean, M = g["oracle_calls_to_opt"]["mean"], r["instance"]["feasible"]
ratio = [s["random_all_expected"] / s["gas_calls_to_opt"] for s in sc]

NAVY, TEAL, ORANGE, INK, MUTED, WHITE, TINT = (RGBColor.from_string(c) for c in
                                               ("0B2E4A", "1C7293", "E8692F", "1D1D1D", "5B6770", "FFFFFF", "EEF4F8"))
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font="Calibri", align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = runs if isinstance(runs, list) else [runs]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        parts = ln if isinstance(ln, list) else [(ln, bold, color)]
        for t, b, c in parts:
            run = p.add_run()
            run.text = t
            run.font.size = Pt(size)
            run.font.bold = b
            run.font.color.rgb = c
            run.font.name = font
    return tb


def card(slide, x, y, w, h, fill=TINT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def dot(slide, x, y, label, color=TEAL):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.5), Inches(0.5))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    tf = s.text_frame
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = WHITE


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.35, 12.1, 0.8, t, size=34, bold=True, color=WHITE if dark else NAVY, font="Cambria")
    if sub:
        text(slide, 0.6, 1.1, 12.1, 0.5, sub, size=16, color=RGBColor.from_string("CADCFC") if dark else MUTED)


def pic(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(RES / name), Inches(x), Inches(y), **kw)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# 1 ---------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, 0.8, 1.6, 11.7, 2.2, "Quantum Grover Adaptive Search for Radiotherapy Beam-Angle Selection", size=44,
     bold=True, color=WHITE, font="Cambria")
text(s, 0.8, 3.9, 11.5, 0.6, "Choosing which radiation beams to use, under a beam limit and an organ-dose limit",
     size=22, color=RGBColor.from_string("CADCFC"))
text(s, 0.8, 5.6, 11.5, 0.5, "Qiskit Fall Fest 2026  ·  The Quantum Optimization Challenge  ·  Domain: Healthcare",
     size=16, color=WHITE)
text(s, 0.8, 6.1, 11.5, 0.5, "github.com/DuaanXahmed/Quantum-Radiotherapy-Beam-Selection", size=14,
     color=RGBColor.from_string("9FB8D0"))
notes(s, "[0:00, ~3 s] Show this slide while you start: 'Hi, this is our Qiskit Fall Fest project in healthcare: "
         "quantum search for radiotherapy beam selection.' Then move to slide 2.")

# 2 ---------------------------------------------------------------- problem
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "The problem: which beam angles?", "Radiotherapy planning, a real clinical decision for every patient")
items = [("1", "Who it serves", "Medical physicists & radiation oncologists; patients benefit"),
         ("2", "Three competing goals", "Full dose to the tumour · spare the organ at risk · spare normal tissue"),
         ("3", "Limited resource", "Number of beams: each one costs treatment & machine time")]
for i, (n, h, b) in enumerate(items):
    y = 1.85 + i * 1.6
    card(s, 0.6, y, 4.9, 1.35)
    dot(s, 0.8, y + 0.42, n, ORANGE if i == 1 else TEAL)
    text(s, 1.5, y + 0.15, 3.9, 0.45, h, size=18, bold=True, color=NAVY)
    text(s, 1.5, y + 0.58, 3.9, 0.75, b, size=14, color=INK)
pic(s, "fig1_phantom_dose.png", 5.8, 1.85, w=7.0)
notes(s, "[0:00-0:25] About half of all cancer patients get radiotherapy. For every patient, a medical physicist has "
         "to decide which angles the radiation beams come from. Each beam hits the tumour, but it also passes through "
         "healthy tissue, and some angles go straight through a critical organ. Each extra beam costs treatment time, "
         "so clinics cap the number of beams. Three competing goals, one limited resource.")

# 3 ---------------------------------------------------------------- formulation
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Formulation and the trade-off", f"12 candidate beams → 4,096 plans → {M} feasible")
blocks = [("Variables", "x_a ∈ {0,1}, one per beam (12 beams = 12 qubits)"),
          ("Objective (minimise)", "F = tumour dose error + w_O · organ dose + w_N · normal-tissue dose"),
          ("Constraint C1", "At most 4 beams (794 of 4,096 plans)"),
          ("Constraint C2", f"Organ maximum dose ≤ 0.6, not quadratic ({M} plans left)")]
for i, (h, b) in enumerate(blocks):
    y = 1.85 + i * 1.22
    card(s, 0.6, y, 5.0, 1.05)
    text(s, 0.8, y + 0.1, 4.6, 0.4, h, size=16, bold=True, color=ORANGE if "C2" in h else NAVY)
    text(s, 0.8, y + 0.48, 4.6, 0.55, b, size=13.5, color=INK)
pic(s, "fig2_pareto.png", 5.95, 1.75, h=5.3)
notes(s, "[0:25-1:00] We model a 2-D phantom: a tumour, an organ at risk just below it, and normal tissue. There are "
         "12 candidate angles and at most 4 beams. One binary variable per beam gives 4,096 plans. The cost adds tumour "
         "dose error, organ dose and normal-tissue dose. Two hard constraints: at most four beams, and the organ's "
         f"maximum dose must stay below 0.6. That second one is not quadratic. Only {M} plans are feasible, and this "
         "plot shows every one. As we weight the organ more, the best plan changes: from beams through the organ, to "
         "beams around it, to a 3-beam plan.")

# 4 ---------------------------------------------------------------- quantum solution
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Quantum solution: Grover Adaptive Search", "Written from scratch in Qiskit (no GroverOptimizer or tutorial code)")
pic(s, "fig6_architecture.png", 0.9, 1.65, w=11.5)
pts = [("Qubit = beam", "each basis state IS a treatment plan"),
       ("Oracle", "marks feasible plans better than the best so far, so constraints are exact"),
       ("Why not QAOA", "no penalties or slack qubits; provable √N query bound")]
for i, (h, b) in enumerate(pts):
    x = 0.6 + i * 4.1
    card(s, x, 6.25, 3.9, 0.95)
    text(s, x + 0.15, 6.3, 3.6, 0.9, [[(h + ": ", True, NAVY), (b, False, INK)]], size=13)
notes(s, "[1:00-1:35] Each qubit is one beam, so each basis state of 12 qubits is a treatment plan. Our oracle flips the "
         "phase of every plan that is feasible and better than the best so far. Because it's a yes/no test, it enforces "
         "the beam limit and the max-dose rule exactly: no penalty tuning, no slack qubits. QAOA can't do that. Grover "
         "amplification makes those plans likely to be measured. We measure one shot, check it classically, lower the "
         "threshold and repeat. It's all written from scratch in Qiskit, and the simulation matches Grover's theory to "
         "10 to the minus 14.")

# 5 ---------------------------------------------------------------- live demo
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
title(s, "Live demo", "Switch to the terminal now", dark=True)
for i, (cmd, what) in enumerate([(".venv/bin/python pitch/live_demo.py",
                                  "Step 1: classical answer key  ·  Step 2: quantum search finds better and better plans"),
                                 ("Result: quantum answer = classical answer ✓",
                                  "Best plan 60°, 90°, 240°, 270°: full tumour dose, organ at 52% (limit 60%)")]):
    y = 2.1 + i * 2.2
    box = card(s, 0.8, y, 11.7, 0.9, fill=RGBColor.from_string("111820"))
    text(s, 1.1, y + 0.12, 11.2, 0.7, "$ " + cmd, size=26, color=RGBColor.from_string("7EE2A8"), font="Courier New")
    text(s, 1.1, y + 1.0, 11.2, 0.8, what, size=17, color=WHITE)
notes(s, "[1:35-2:10] Switch to the terminal and run: .venv/bin/python pitch/live_demo.py. "
         "PATIENT: 'Our patient has 12 possible beam angles, at most 4 beams, and a dose limit on the nearby organ: 4,096 "
         "plans, only 204 follow the rules.' STEP 1: 'The classical answer key: the best plan is 60, 90, 240 and 270 "
         "degrees.' STEP 2: 'Now the quantum search on a 12-qubit simulator. Each row is a better plan; the score keeps "
         "dropping until it lands on the best plan.' STEP 3: 'The quantum answer matches exactly. Full tumour dose, organ "
         "at 52% under its 60% limit. 126 queries on average versus about 2,048 for random guessing.'")

# 6 ---------------------------------------------------------------- results
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Results vs classical baselines", f"{g['trials']} seeded runs, every number checked against brute force")
stats = [(f"{100 * g['success_rate']:.0f}%", "found the exact optimum"),
         (f"{mean:.0f}", "oracle calls on average"),
         (f"{rs['all_states']['analytic'] / mean:.0f}×", "fewer queries than unstructured search (2,048)"),
         (f"{ratio[0]:.0f}× → {ratio[-1]:.0f}×", "gap grows from 8 to 14 beams")]
for i, (big, lab) in enumerate(stats):
    y = 1.8 + i * 1.32
    text(s, 0.6, y, 3.6, 0.75, big, size=40, bold=True, color=ORANGE if i == 2 else NAVY, font="Cambria")
    text(s, 0.6, y + 0.72, 3.6, 0.5, lab, size=13, color=MUTED)
pic(s, "fig3_oracle_calls.png", 4.3, 2.0, w=8.7)
notes(s, f"[2:10-2:35] Over {g['trials']} seeded runs, Grover Adaptive Search found the exact optimum every time, using "
         f"{mean:.0f} oracle calls on average. Unstructured classical search needs about 2,048 evaluations: "
         f"{rs['all_states']['analytic'] / mean:.0f} times more. From 8 to 14 beams that gap grows from {ratio[0]:.0f} "
         f"to {ratio[-1]:.0f} times, close to the square-root-of-N line.")

# 7 ---------------------------------------------------------------- limits + message
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
title(s, "Honest limitations, and our message", dark=True)
lims = ["At 12 beams, structure-aware classical methods (greedy, annealing, feasible-only search) need fewer queries",
        "Advantage is in queries, not wall-clock time; the oracle is a precomputed truth table",
        "A real oracle needs reversible arithmetic + fault-tolerant hardware (Babbush et al. 2021)",
        "Synthetic 2-D phantom, ideal simulator: not clinical software"]
card(s, 0.6, 1.5, 6.2, 5.4, fill=RGBColor.from_string("163E5E"))
text(s, 0.9, 1.7, 5.7, 0.5, "Limitations", size=20, bold=True, color=RGBColor.from_string("F4A47C"))
text(s, 0.9, 2.3, 5.7, 4.5, ["• " + l for l in lims], size=15, color=WHITE, space=12)
text(s, 7.2, 1.7, 5.6, 0.5, "Final message", size=20, bold=True, color=RGBColor.from_string("7EE2A8"))
text(s, 7.2, 2.3, 5.6, 3.6, ["Any clinical rule can become the oracle.",
                             "Quantum search is a natural engine for constrained treatment planning.",
                             "We built a verified, reproducible blueprint for that future."],
     size=24, color=WHITE, font="Cambria", space=16)
text(s, 7.2, 6.2, 5.6, 0.5, "Thank you", size=20, bold=True, color=RGBColor.from_string("CADCFC"))
notes(s, "[2:35-3:00] To be honest: classical methods that use the structure, like greedy search, annealing, or listing "
         "only the feasible plans, are as fast or faster at this size. Our advantage is in queries, not wall-clock time. "
         "Our oracle is a precomputed table, and a real one would need fault-tolerant hardware. Our message: quantum "
         "search is a natural engine for constrained treatment planning, because any clinical rule becomes part of the "
         "oracle. We built a verified, reproducible blueprint for that future. Thank you.")

out = Path(__file__).parent / "Radiotherapy_GAS_Pitch.pptx"
prs.save(out)
print("wrote", out)
