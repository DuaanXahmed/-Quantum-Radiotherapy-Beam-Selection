"""Build report/Radiotherapy_GAS_Report.docx. Every number is read from results/results.json.

    python make_report.py [--video-url URL]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from phantom import build_phantom

ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = ROOT / "report" / "Radiotherapy_GAS_Report.docx"
REPO = "https://github.com/DuaanXahmed/Quantum-Radiotherapy-Beam-Selection"
TITLE = "Quantum Grover Adaptive Search for Radiotherapy Beam-Angle Selection under a Beam Limit"
BLUE = RGBColor(0x1F, 0x5F, 0xAD)
INK2 = RGBColor(0x52, 0x51, 0x4E)

doc = Document()
_fig = [0]
_tab = [0]


# ------------------------------------------------------------------ helpers
def setup():
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(5)
    st.paragraph_format.line_spacing = 1.12
    for name, size in (("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 11)):
        h = doc.styles[name]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.color.rgb = BLUE
        h.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        h.paragraph_format.space_after = Pt(4)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)
    # page numbers in the footer
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def rich(p, text, size=None, color=None):
    """Add text with **bold** and *italic* markup to paragraph p."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return p


def P(text, style=None, align=None, size=None, color=None, after=None):
    p = doc.add_paragraph(style=style)
    rich(p, text, size, color)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def H(text, level=1):
    return doc.add_heading(text, level)


def bullets(items, style="List Bullet"):
    for it in items:
        p = P(it, style=style, after=2)
        p.paragraph_format.left_indent = Inches(0.3)


def eq(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Cambria Math"
    r.italic = True
    r.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(4)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tcPr.append(sh)


def table(header, rows, widths=None, caption=None, font=9):
    if caption:
        _tab[0] += 1
        cp = P(f"**Table {_tab[0]}.** {caption}", size=9, color=INK2, after=2)
        cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        rich(c.paragraphs[0], f"**{h}**", font)
        shade(c, "DCE8F7")
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = ""
            rich(c.paragraphs[0], str(v), font)
            if i % 2 == 0:
                shade(c, "F6F6F3")
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)
    for row in t.rows:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def callout(text, fill="EAF2FC"):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.text = ""
    shade(c, fill)
    lines = text if isinstance(text, list) else [text]
    rich(c.paragraphs[0], lines[0], 10)
    for ln in lines[1:]:
        rich(c.add_paragraph(), ln, 10)
    for p in c.paragraphs:
        p.paragraph_format.space_after = Pt(3)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(name, caption, width=6.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(RES / name), width=Inches(width))
    _fig[0] += 1
    P(f"**Figure {_fig[0]}.** {caption}", size=9, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def f(x, d=4):
    return f"{x:.{d}f}"


def beams(b):
    return "{" + ", ".join(f"{int(a)}°" for a in b) + "}"


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ report
def build(res, video_url):
    inst, opt, cfg = res["instance"], res["optimum"], res["config"]
    bl, gas, st = res["baselines"], res["gas"], res["circuit_stats"]
    rs, sa, gr, nn = bl["random_search"], bl["simulated_annealing"], bl["greedy"], bl["nnls"]
    N, n, M = inst["N"], inst["n"], inst["feasible"]
    g_mean = gas["oracle_calls_to_opt"]["mean"]
    g_med = gas["oracle_calls_to_opt"]["median"]
    sc = res["scaling"]
    ratio = [s["random_all_expected"] / s["gas_calls_to_opt"] for s in sc]
    import numpy as np
    expo = float(np.polyfit(np.log([s["N"] for s in sc]), np.log([s["gas_calls_to_opt"] for s in sc]), 1)[0])
    sweep = res["sweep"]
    wins = [x["n"] for x in sc if x["gas_calls_to_opt"] < x["random_feasible_expected"]]
    loses = [x["n"] for x in sc if x["gas_calls_to_opt"] >= x["random_feasible_expected"]]
    big = sc[1:] if len(sc) >= 3 else sc
    expo_hi = float(np.polyfit(np.log([x["N"] for x in big]), np.log([x["gas_calls_to_opt"] for x in big]), 1)[0])
    nlist = lambda l: ", ".join(str(v) for v in l)
    distinct = []
    for s in sweep:
        if s["index"] not in [d["index"] for d in distinct]:
            distinct.append(s)
    greedy_ok = sum(s["greedy_found_opt"] for s in sweep)
    max_err = max(c["max_abs_error"] for c in res["grover_check"])
    abl = {a["patience"]: a for a in res["ablation"]}
    P_def = gas["patience"]

    # ---------------- title block
    p = P(f"**{TITLE}**", align=WD_ALIGN_PARAGRAPH.CENTER, size=19, color=BLUE, after=4)
    P("Qiskit Fall Fest 2026 · The Quantum Optimization Challenge · Domain: **Healthcare**",
      align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color=INK2, after=2)
    P(f"Code: {REPO}   ·   3-minute demo: {video_url}", align=WD_ALIGN_PARAGRAPH.CENTER, size=9,
      color=INK2, after=10)

    callout([
        "**At a glance.** A radiotherapy plan must choose which gantry angles to use. We select up to "
        f"k = {cfg['k']} of {n} candidate beams on a synthetic 2-D phantom to cover a tumour while sparing a nearby "
        f"organ at risk (OAR) and normal tissue, subject to a hard OAR maximum-dose limit. The {N} possible plans map "
        f"one-to-one onto the basis states of {n} qubits. A from-scratch **Grover Adaptive Search (GAS)** in Qiskit "
        "uses a phase oracle that marks *feasible plans better than the current best* and finds the minimum.",
        f"**Result.** The exact optimum is {beams(opt['beams'])} (F = {f(opt['cost'], 5)}, unique). GAS found it in "
        f"**{pct(gas['success_rate'])} of {gas['trials']} seeded runs** with a mean of **{g_mean:.0f} oracle calls** "
        f"(median {g_med:.0f}). Classical unstructured search needs (N+1)/2 = {rs['all_states']['analytic']:.0f} evaluations. "
        f"That is {(rs['all_states']['analytic'] / g_mean):.0f}× more, and the gap widens as the problem grows "
        f"({', '.join(f'{r:.0f}×' for r in ratio)} for n = {', '.join(str(s['n']) for s in sc)} beams; "
        f"GAS queries grow as about N^{expo_hi:.2f} for n = {big[0]['n']}–{big[-1]['n']}, vs N¹ for unstructured search).",
        "**Honest caveat.** This is a *query-count* advantage under an ideal oracle. At n = 12, classical methods that "
        f"use the structure need fewer queries: greedy {gr['evaluations']}, annealing {sa['mean_evals_to_opt']:.0f}, random search over "
        f"only the {M} feasible plans {rs['feasible_only']['analytic']:.1f}. Against that last baseline the picture across sizes "
        f"is mixed (GAS fewer at n = {nlist(wins)}, more at n = {nlist(loses)}). Our oracle is a precomputed truth table. "
        "Section 6 sets out what a real advantage would require.",
    ])

    # ---------------- 1 problem
    H("1. What is the problem")
    P("**Who it serves.** Medical physicists and radiation oncologists who plan external-beam radiotherapy. "
      "The patients treated with those plans benefit. About half of all cancer patients receive radiotherapy at some point "
      "in their treatment, so plan quality and machine time matter at scale.")
    P("**The decision.** A linear accelerator rotates around the patient on a gantry. The planner picks which "
      "gantry angles deliver radiation beams. Every beam adds dose to the tumour, but it also passes through healthy tissue on the way "
      "in and out. Three objectives compete:")
    bullets(["**(a) Tumour coverage**: deliver the prescribed dose (1.0) uniformly to the tumour;",
             "**(b) Organ-at-risk (OAR) sparing**: keep dose low in a nearby critical organ (e.g. spinal cord or rectum);",
             "**(c) Normal-tissue sparing**: limit the integral dose to everything else."])
    P("**The limited resource** is the number of beams. Each extra beam lengthens treatment and machine time and "
      "adds QA workload, so clinics cap it. Beam-angle selection is a non-convex combinatorial problem: with n candidate "
      "angles there are 2ⁿ candidate plans.")

    H("1.1 Formulation", 2)
    P("**Decision variables.** x_a ∈ {0, 1} for a = 0 … 11 (x_a = 1 means beam a at angle 30·a° is used), "
      f"giving N = 2¹² = {N} plans. The dose to voxel v is linear in x:")
    eq("d_v(x) = Σₐ xₐ · D[a, v]")
    P("**Objective** (minimise; the weights express clinical priorities):")
    eq("F(x) = w_T · mean_tumour (d_v − 1)²  +  w_O · mean_OAR d_v²  +  w_N · mean_normal d_v²")
    P(f"with defaults w_T = {cfg['weights']['w_T']:g}, w_O = {cfg['weights']['w_O']:g}, w_N = {cfg['weights']['w_N']:g}. "
      "**Hard constraints:**")
    bullets([f"**C1 beam limit:** Σₐ xₐ ≤ k = {cfg['k']}. This leaves {inst['feasible_c1']} of {N} plans "
             f"(= Σ_(j≤4) C(12, j), verified in code);",
             f"**C2 OAR maximum dose:** max_(v∈OAR) d_v(x) ≤ {cfg['oar_max']}. This is a dose-volume-style limit and it is "
             f"*not* quadratic in x. With C1 it leaves **M = {M} feasible plans**."])
    table(["Item", "Value"], [
        ["Phantom", f"{inst['grid']}×{inst['grid']} voxel grid; body = disk of radius {inst['body_radius']:g} "
                    f"({inst['n_voxels']} voxels)"],
        ["Tumour (target)", f"disk, radius {inst['tumour_radius']:g}, at the isocentre ({inst['n_tumour']} voxels)"],
        ["Organ at risk", f"disk, radius {inst['oar_radius']:g}, centred at (x, y) = ({inst['oar_centre'][0]:g}, "
                          f"{inst['oar_centre'][1]:g}) ({inst['n_oar']} voxels)"],
        ["Normal tissue", f"rest of the body ({inst['n_normal']} voxels)"],
        ["Candidate beams", f"{n} gantry angles 0°, 30°, …, 330°; parallel strip {inst['beam_width']:g} voxels wide"],
        ["Dose model", f"exp(−{inst['mu']:g} · depth from body surface) inside the strip; each beam scaled to "
                       f"{inst['dose_per_beam']:g} at the isocentre (≈ 4 beams reach the prescription 1.0)"],
        ["Search space", f"N = {N} plans; {inst['feasible_c1']} satisfy C1; {M} satisfy C1 and C2"],
        ["Exact optimum", f"{beams(opt['beams'])}, F = {f(opt['cost'], 5)}; runner-up {beams(opt['second_beams'])}, "
                          f"F = {f(opt['second_cost'], 5)} (unique, gap {opt['gap']:.1e})"],
    ], widths=[1.5, 5.2], caption="Test instance. It is synthetic, with no patient data. All seeds are fixed.")
    ph = build_phantom(n)
    cover = (ph.D[:, ph.oar] > 0).mean(1)            # fraction of OAR voxels inside each beam
    crossing = [int(a) for a, v in zip(ph.angles_deg, cover) if v >= 0.5]
    sparing = [int(a) for a, v in zip(ph.angles_deg, cover) if v <= 0.1]
    figure("fig1_phantom_dose.png",
           "Left: the phantom and the 12 candidate beam directions. Right: dose of the exact optimal plan. It avoids "
           f"the {len(crossing)} beams that pass through most of the OAR ({', '.join(f'{a}°' for a in crossing)}; each covers "
           f"≥ {100 * min(v for v in cover if v >= 0.5):.0f}% of OAR voxels) and uses the {len(sparing)} beams that touch at most "
           f"one OAR voxel ({', '.join(f'{a}°' for a in sparing)}). It covers the tumour "
           f"(min dose {opt['tumour_min']:.3f}) and keeps the OAR maximum at {opt['oar_maxdose']:.3f}, below the "
           f"{cfg['oar_max']} limit.")

    # ---------------- 2 why
    H("2. Why this problem")
    bullets([
        "**It is a real clinical bottleneck.** Many clinics still choose beam angles from templates (\"class "
        "solutions\") or planner experience. Better angles can give the same tumour dose with less damage to "
        "organs. Fewer beams mean shorter treatments.",
        "**It is the kind of problem the challenge asks for.** It has three competing objectives and a hard limit on a "
        "scarce resource (beams). Section 5 shows the trade-off quantitatively: as the OAR weight w_O rises, the "
        f"optimal plan changes ({len(distinct)} distinct optima across w_O ∈ "
        f"{{{', '.join(f'{s['w_O']:g}' for s in sweep)}}}).",
        "**It is hard in general.** The objective over beam *sets* is non-convex. Real planning uses 36–180 "
        "candidate angles (10° to 2° spacing) and 5–9 beams. Choosing 7 of 72 angles alone gives C(72, 7) ≈ "
        f"{math.comb(72, 7):.1e} plans, each needing its own fluence optimisation.",
        "**The constraints are not quadratic.** Maximum-dose limits like C2 are what clinicians actually write "
        "down. A QUBO/QAOA formulation must approximate them with penalties and slack qubits. Grover search can enforce "
        "them exactly, which makes this a good fit for the algorithm we chose.",
        "**Originality.** Quantum work in healthcare usually targets scheduling or equipment allocation. We are not aware "
        "of a Grover-based treatment-planning demonstration of this kind. We built a dose model, a constrained clinical "
        "objective and a Pareto study around it.",
    ])

    # ---------------- 3 classical
    H("3. Existing classical methods")
    P("Beam-angle optimisation (BAO) has been studied since the early 1990s (e.g. Bortfeld & Schlegel, 1993). "
      "The main families are:")
    table(["Approach", "How it works", "Guarantee / weakness"], [
        ["Class solutions / templates", "Equi-spaced or site-specific standard angles chosen by the planner",
         "Fast; no optimality, ignores the patient's anatomy"],
        ["Exhaustive enumeration", "Evaluate every plan", "Exact; cost grows as 2ⁿ (or C(n, k))"],
        ["Greedy / iterative beam addition", "Add the beam that helps most, repeat", "Fast; can get stuck (non-convex)"],
        ["Simulated annealing, genetic algorithms", "Stochastic local search over beam sets",
         "Good in practice; no certificate of optimality, needs tuning"],
        ["Mixed-integer programming (branch-and-bound)", "Binary beam variables + dose constraints, solved exactly",
         "Exact with proof of optimality; can become very slow at clinical scale"],
        ["Beam's-eye-view scoring", "Rank single beams by target-vs-OAR overlap", "Cheap; ignores how beams interact"],
    ], widths=[1.8, 2.6, 2.3], caption="Classical approaches to beam-angle selection.")
    P("**Baselines we implemented** on the same instance (all seeded, same cost function, results in Table 7):")
    bullets([
        f"**Brute force**: all {N} plans, and the {M} feasible plans only. Exact; {bl['brute_force_all']['seconds']*1e3:.0f} ms in Python.",
        f"**Random search without replacement**: 1000 seeded trials; analytic (N+1)/2 = "
        f"{rs['all_states']['analytic']:.1f} over all plans and (M+1)/2 = {rs['feasible_only']['analytic']:.1f} over feasible plans.",
        f"**Greedy forward selection**: {gr['evaluations']} evaluations; finds the optimum at the default weights but "
        f"only at {greedy_ok} of {len(sweep)} weight settings.",
        f"**Simulated annealing**: constraint-aware, with flip and swap moves, {sa['steps']} steps, {sa['trials']} seeds; success "
        f"{pct(sa['success_rate'])}.",
        f"**NNLS lower bound**: continuous beam intensities with no beam limit (scipy nnls). F ≥ {f(nn['bound'], 5)} "
        "for every plan, which certifies how good the optimum is.",
    ])
    callout("**Takeaway.** At 12 beams classical methods solve this instance easily. That is intended: the small size "
            "lets us check the quantum result against an exact answer. The difficulty is in scaling. Exact methods "
            "grow exponentially, and fast heuristics give no guarantee (greedy fails at w_O = 0).", fill="F3F2EE")

    # ---------------- 4 why quantum
    H("4. Why we need quantum computing")
    P("Finding a plan that is *feasible and better than the current best* is a search with a yes/no test. "
      "Without exploitable structure, a classical algorithm needs about N/(2M) evaluations to find one of M marked "
      "items among N. **Grover's algorithm needs about (π/4)·√(N/M) oracle calls.** That is a provable quadratic "
      "reduction in queries (Grover, 1996). Dürr and Høyer (1996) turned this into **minimum finding** with "
      "O(√N) queries: repeat Grover search with a threshold that falls each time. Boyer, Brassard, Høyer and Tapp (1998) "
      "showed how to do this when M is unknown, using a random iteration count. Gilliam, Woerner and Gonciulea (2021) "
      "applied the scheme to constrained binary optimisation and called it Grover Adaptive Search.")
    table(["", "Grover Adaptive Search (ours)", "QAOA / VQE (QUBO-based)"], [
        ["Cost function", "Any computable F(x)", "Must be quadratic (QUBO) or a low-order polynomial"],
        ["Constraint C1 (Σx ≤ k)", "Checked exactly inside the oracle", "Penalty term + slack qubits"],
        ["Constraint C2 (max dose ≤ 0.6)", "Checked exactly inside the oracle", "No exact quadratic form; needs approximation"],
        ["Penalty tuning", "None", "Penalty weights must be tuned; infeasible outputs are possible"],
        ["Guarantee", "Provable query bound O(√N); exact optimum with high probability",
         "Heuristic; approximation ratio depends on depth p and training"],
        ["Qubits here", f"{n} (one per beam), 0 ancillas", f"{n} + slack qubits for C1"],
    ], widths=[1.6, 2.6, 2.5], caption="Why GAS rather than QAOA for this problem.")
    P("So the quantum part is not a showcase: it is the search engine. The circuit does the work of "
      "finding an improving plan, and the classical side only checks the one plan that was measured. "
      "We do not claim a wall-clock speedup at this size; the claim is about *how the number of queries grows* (Section 5.6).")

    # ---------------- 5 quantum solution
    H("5. The quantum solution")
    H("5.1 Mapping the problem to qubits", 2)
    P(f"Qubit a represents beam a. The computational basis state |i⟩ on {n} qubits *is* the plan whose beam a is on "
      "exactly when bit a of i is 1. This is Qiskit's little-endian order, so qubit 0 is beam 0° and the leftmost character of a "
      f"measured bitstring is beam 330°. The optimum {beams(opt['beams'])} is the state |{opt['bitstring']}⟩ "
      f"(index {opt['index']}). Starting from H^⊗{n}|0⟩, the register holds all {N} plans in equal superposition.")
    H("5.2 Oracle, diffuser and Grover iteration", 2)
    eq("O_y |x⟩ = −|x⟩  if  x satisfies C1 and C2  and  F(x) < y,     O_y |x⟩ = |x⟩  otherwise")
    eq("W = H^⊗n X^⊗n (multi-controlled Z) X^⊗n H^⊗n = −(2|s⟩⟨s| − I),      G = W · O_y")
    P("The oracle is a `DiagonalGate` built from a ±1 truth table, so it carries both hard constraints and the "
      "threshold exactly. The diffuser uses an 11-controlled phase(π) gate (`mcp`). After r iterations the "
      "probability of measuring one of the M marked plans is sin²((2r+1)θ), where sin²θ = M/N (Figure 3).")
    H("5.3 Grover Adaptive Search loop", 2)
    for s in [
        "Initialise: draw a random feasible plan x₀ (seeded) and set y = F(x₀), best = x₀, m = 1, λ = 6/5.",
        "Pick r uniformly from {0, …, ⌈m⌉ − 1} (BBHT schedule; this handles the unknown M).",
        f"Build H^⊗{n} · (W O_y)^r, measure **1 shot** on AerSimulator (seeded), and decode the bitstring into a plan x.",
        "Evaluate F(x) and the constraints classically. If x is feasible and F(x) < y: set y ← F(x), best ← x, m ← 1, and "
        "rebuild O_y. Otherwise: m ← min(λm, √N).",
        f"Stop after {P_def} consecutive non-improving rounds or an oracle budget of 30√N = {gas['budget']} calls. "
        "Return best.",
    ]:
        P(s, style="List Number", after=2)
    P(f"**Termination rule (our adaptation).** With λ = 6/5 the iteration range m needs ⌈log_λ √N⌉ = "
      f"{P_def - 2} failures to reach its √N cap. We therefore stop after ⌈log_λ √N⌉ + 2 = {P_def} failures, a rule "
      "that scales with N. The commonly quoted '15 rounds' stops while r ≤ 15. That is too few to amplify the last "
      "1–2 improving plans out of 4096, and the ablation in Table 4 shows the effect clearly.")
    table(["Consecutive failures before stop", "Runs", "Success (exact optimum)", "Mean oracle calls to optimum",
           "Mean total oracle calls"],
          [[f"{a['patience']}{'  (ours)' if a['patience'] == P_def else ''}", a["trials"], pct(a["success_rate"]),
            f"{a['mean_calls_to_opt']:.1f}" if a["mean_calls_to_opt"] else "n/a", f"{a['mean_calls_total']:.1f}"]
           for a in res["ablation"]],
          widths=[1.7, 0.6, 1.3, 1.6, 1.4], caption="Termination-rule ablation (same seeds).")

    H("5.4 Architecture and circuit resources", 2)
    figure("fig6_architecture.png", "Hybrid architecture. Blue = quantum components (Qiskit circuits on AerSimulator); "
           "grey = classical components. The only quantum-to-classical link is a single measured bitstring per round.")
    table(["Quantity", "Value"], [
        ["Qubits / ancillas", f"{st['n_qubits']} / {st['ancillas']}"],
        ["One Grover iteration, logical", f"depth {st['logical_depth']}: 1 DiagonalGate (oracle), 1 multi-controlled phase, "
                                          f"{st['logical_ops'].get('h', 0)} H, {st['logical_ops'].get('x', 0)} X"],
        ["One iteration transpiled to {cx, rz, sx, x}", f"depth {st['transpiled_depth']:,}, {st['transpiled_size']:,} gates, "
                                                         f"{st['cx_per_iteration']:,} CX"],
        ["  of which oracle", f"{st['oracle_cx']:,} CX (depth {st['oracle_depth']:,}); ≈ 2ⁿ, as expected for an arbitrary diagonal"],
        ["  of which diffuser", f"{st['diffuser_cx']:,} CX (depth {st['diffuser_depth']:,})"],
    ], widths=[2.4, 4.3], caption="Circuit statistics (Qiskit transpile, optimization_level = 1, seed 0).")
    P("The transpiled numbers matter: an arbitrary truth-table oracle costs about 2ⁿ two-qubit gates. This is the "
      "main practical limitation (Section 6).")

    H("5.5 Validation", 2)
    figure("fig5_grover_theory.png", f"Probability of measuring a marked plan after r Grover iterations: Aer "
           f"statevector simulation (dots) vs the formula sin²((2r+1)θ) (lines) for M = 1, 4, 16 marked plans. "
           f"Maximum deviation {max_err:.0e}, i.e. exact to machine precision.", width=5.6)
    P("The pytest suite (9 tests) checks: the feasible count (794 under C1); F against a beam-by-beam manual "
      "calculation; the qubit-to-beam bit order; that the oracle flips exactly the feasible, below-threshold plans; that one and two "
      "Grover iterations match theory; the diffuser as an inversion about the mean; and that a seeded GAS run returns "
      "the brute-force optimum and is reproducible.")

    H("5.6 Results", 2)
    P("**Trade-off between objectives (exact enumeration).**")
    table(["w_O", "Optimal beams", "F", "Tumour error", "OAR mean dose", "OAR max dose", "Greedy finds it?"],
          [[f"{s['w_O']:g}", beams(s["beams"]), f(s["cost"], 5), f"{s['tumour_err']:.2e}", f(s["oar_mean"], 3),
            f(s["oar_maxdose"], 3), "yes" if s["greedy_found_opt"] else "**no**"] for s in sweep],
          widths=[0.45, 1.6, 0.8, 0.95, 1.0, 0.95, 0.95], caption="Exact optimum for each OAR weight (every optimum unique).")
    figure("fig2_pareto.png", "Every feasible plan (grey) in the tumour-error vs OAR-dose plane, the Pareto front, and "
           "the optima for each w_O. Raising w_O moves the optimum from an OAR-crossing 4-beam plan to an OAR-avoiding "
           "4-beam plan, and finally to a 3-beam plan that gives up tumour coverage to protect the organ. The w_O = 0 "
           "plan sits off the front in this 2-D view because it trades the third objective (normal-tissue dose).",
           width=5.8)
    P("**GAS vs classical baselines** (n = 12, N = 4096, default weights).")
    rows = [
        ["**GAS (quantum, ours)**", f"**{g_mean:.0f}** oracle calls (median {g_med:.0f})", pct(gas["success_rate"]),
         f"{gas['oracle_calls_total']['mean']:.0f} oracle calls, {gas['evaluations_total']['mean']:.0f} F-evaluations",
         "Provable O(√N) queries; cost-agnostic oracle"],
        ["Brute force (all plans)", f"{N}", "100%", f"{N} evaluations", "Exact"],
        ["Brute force (feasible plans only)", f"{M}", "100%", f"{M} evaluations", "Exact; needs the feasible set listed in advance"],
        ["Random search, all plans", f"{rs['all_states']['analytic']:.1f} (sim. {rs['all_states']['mean']:.0f})", "100%",
         "until found", "Unstructured classical reference"],
        ["Random search, feasible only", f"{rs['feasible_only']['analytic']:.1f} (sim. {rs['feasible_only']['mean']:.0f})",
         "100%", "until found", "Uses structure"],
        ["Greedy forward selection", f"{gr['evaluations']}", "yes" if gr["found_opt"] else "no", f"{gr['evaluations']}",
         f"Heuristic; fails at {len(sweep) - greedy_ok} of {len(sweep)} weights"],
        ["Simulated annealing", f"{sa['mean_evals_to_opt']:.0f}", pct(sa["success_rate"]), f"{sa['evaluations_per_run']}",
         "Heuristic; needs tuning"],
        ["NNLS continuous bound", "n/a", "n/a", "1 solve",
         f"F ≥ {f(nn['bound'], 5)} (optimum is {opt['cost'] / nn['bound']:.2f}× the bound)"],
    ]
    table(["Method", "Queries to reach optimum (mean)", "Success", "Total cost per run", "Notes"], rows,
          widths=[1.65, 1.5, 0.65, 1.35, 1.65], caption=f"GAS ({gas['trials']} seeds) vs classical baselines.", font=8.5)
    figure("fig3_oracle_calls.png", "Left: queries needed to reach the exact optimum as the number of candidate beams grows "
           "(mean over seeded runs). GAS follows the √N reference while unstructured search grows as N. Right: "
           "the distribution of GAS oracle calls over seeds at n = 12, compared with the classical references.")
    table(["n beams", "N", "Feasible M", "√N", "GAS calls to opt.", "GAS / √N", "Random (N+1)/2", "Random ÷ GAS",
           "Random feasible (M+1)/2", "GAS success"],
          [[s["n"], s["N"], s["feasible"], f"{s['sqrt_N']:.0f}", f"{s['gas_calls_to_opt']:.1f}",
            f"{s['gas_calls_to_opt'] / s['sqrt_N']:.2f}", f"{s['random_all_expected']:.1f}",
            f"{s['random_all_expected'] / s['gas_calls_to_opt']:.1f}×", f"{s['random_feasible_expected']:.1f}",
            f"{pct(s['gas_success'])} ({s['trials']})"] for s in sc],
          widths=[0.5, 0.55, 0.6, 0.45, 0.75, 0.6, 0.75, 0.7, 0.85, 0.75],
          caption="Scaling study: same phantom with n equi-spaced candidate angles, k = 4.", font=8)
    figure("fig4_gas_convergence.png", "GAS convergence: the threshold y (best cost so far) against cumulative oracle "
           "calls for six seeded runs. Each step down is a measured plan that beat the previous best. The flat tail "
           "is the stopping phase, where no better plan exists.", width=5.6)
    P(f"**Reading the results.** (1) GAS is correct: it returned the unique brute-force optimum in "
      f"{pct(gas['success_rate'])} of runs. (2) It needed {g_mean:.0f} oracle calls on average, "
      f"{g_mean / math.sqrt(N):.1f}·√N, against {rs['all_states']['analytic']:.0f} for unstructured classical search. The "
      f"ratio grows from {ratio[0]:.0f}× to {ratio[-1]:.0f}× as n grows from {sc[0]['n']} to {sc[-1]['n']}. A log-log fit "
      f"over n = {big[0]['n']}–{big[-1]['n']} gives GAS queries ∝ N^{expo_hi:.2f}, close to the ideal Dürr–Høyer N^0.5 and "
      f"far below the N¹ of unstructured search. Over all sizes the fit is N^{expo:.2f}, because at n = {sc[0]['n']} "
      f"{sc[0]['feasible']} of {sc[0]['N']} plans are feasible and zero-iteration rounds (plain sampling, no oracle calls) "
      "often hit the optimum, which makes the smallest instance unusually cheap. (3) GAS does **not** robustly beat "
      f"structure-aware classical methods. At n = 12, greedy needs {gr['evaluations']} evaluations, annealing "
      f"{sa['mean_evals_to_opt']:.0f} and random search over the {M} feasible plans {rs['feasible_only']['analytic']:.1f}, all fewer "
      f"than GAS's {g_mean:.0f}. Against feasible-only random search the result is mixed across sizes: GAS needs fewer "
      f"queries at n = {nlist(wins)} and more at n = {nlist(loses)}. "
      + "(4) Including the stopping phase, a run costs "
      f"{gas['oracle_calls_total']['mean']:.0f} oracle calls. Proving that nothing better exists is the expensive part, "
      "as in classical search.")

    # ---------------- 6 limitations
    H("6. Limitations and honest reflection")
    bullets([
        "**The advantage is in query count only.** There is no wall-clock speedup on a simulator: a GAS run takes about "
        f"{gas['seconds'] / gas['trials']:.1f} s, while brute force takes {bl['brute_force_all']['seconds'] * 1e3:.0f} ms. There is "
        "also no advantage over structured solvers (branch-and-bound / MIQP) or over methods that use the feasible-set "
        "structure (Table 7).",
        "**The oracle is a precomputed truth table**, which means every plan is already evaluated classically to build it. "
        "A real oracle would compute dose, cost and constraints with reversible arithmetic circuits on "
        f"error-corrected hardware. Our truth-table oracle already needs {st['oracle_cx']:,} CX gates per call. Babbush et al. "
        "(PRX Quantum 2021) show that such overheads can cancel a quadratic speedup at realistic sizes.",
        "**12 qubits can be brute-forced.** Clinical planning uses about 36–180 angles, continuous beam intensities "
        "(fluence maps), 3-D anatomy and Monte-Carlo dose. Our instance is a deliberately small, verifiable model "
        "of that decision.",
        "**The phantom and dose model are simplified**: 2-D synthetic geometry, no scatter, no penumbra, no tissue "
        "heterogeneity. The weights and the OAR limit were tuned on this one instance. The OAR centre was placed 0.3 "
        "voxels off the symmetry axis so that mirror-image plans do not tie (Section 8).",
        f"**The stopping rule is a design choice.** With {P_def} failures GAS succeeded in {pct(gas['success_rate'])} of runs; "
        f"with 15 it succeeded in {pct(abl[15]['success_rate'])}. GAS is probabilistic and gives no certificate of "
        "optimality, unlike brute force or branch-and-bound.",
        "**The simulator is ideal and noise-free.** This is a research prototype, not clinical software, and must not be "
        "used for treatment decisions.",
    ])
    P("**Next technical step.** GAS currently searches all 2ⁿ plans, although only plans with ≤ k beams can be "
      "feasible. Preparing a Dicke-state superposition over the ≤ k-beam plans instead of H^⊗n would shrink the "
      f"quantum search space from {N} to {inst['feasible_c1']} plans and the query count by about "
      f"√({N}/{inst['feasible_c1']}) ≈ {math.sqrt(N / inst['feasible_c1']):.1f}×. That closes most of the gap to "
      "structure-aware classical search while keeping the exact constraint handling.")

    # ---------------- 7 conclusion
    H("7. Conclusion: the final message")
    callout([
        "**What we built.** We turned a real radiotherapy decision (which beams, under a beam limit and a hard "
        "organ-dose limit) into a 12-qubit search problem. We then solved it with a Grover Adaptive Search written "
        "from scratch in Qiskit, where the quantum circuit does the search and every number is checked against an "
        "exact classical answer.",
        f"**What we found.** GAS returned the exact optimal plan in {pct(gas['success_rate'])} of {gas['trials']} seeded runs. "
        f"It needed about {g_mean:.0f} oracle calls, compared with {rs['all_states']['analytic']:.0f} for "
        "unstructured classical search, and that gap widens with every beam added. The oracle "
        "enforces non-quadratic clinical constraints exactly, without penalties or extra qubits. QUBO-based methods cannot do this.",
        "**The final message.** Quantum search is a natural engine for *constrained* treatment-plan selection, because any "
        "clinical rule a planner can compute becomes part of the oracle. Its benefit is a provable √N reduction in "
        "queries, not faster planning today. To matter clinically it needs fault-tolerant hardware, efficient arithmetic "
        "oracles and much larger beam sets. Until then, the honest role of this work is a verified, reproducible "
        "blueprint for that future, with clear evidence of where classical methods still win.",
    ])

    # ---------------- 8 reproducibility + credits
    H("8. Reproducibility and credits")
    v = cfg["versions"]
    P(f"**Run it.** `pip install -r requirements.txt`, then `pytest -q`, then `python run_experiments.py` "
      f"(full run {res['runtime_seconds']:.0f} s; `--quick` < 60 s), then `python plots.py` and `python make_report.py`. "
      f"Seeds: GAS {cfg['seeds']['gas']}+i, SA {cfg['seeds']['sa']}+i, scaling {cfg['seeds']['scaling']}+i, random search "
      f"{cfg['seeds']['random_search']}. Python {v['python']}, Qiskit {v['qiskit']}, Qiskit Aer {v['qiskit_aer']}, "
      f"NumPy {v['numpy']}. All numbers in this report are read from results/results.json.")
    P("**Design choices.** OAR centre (0.3, −5.7) rather than (0, −6): with a perfectly symmetric phantom every "
      "asymmetric plan has a mirror twin of equal cost, so the optima at w_O = 0 and w_O = 5 tied. The small offset makes "
      "every optimum unique. Simulated annealing uses swap moves in addition to bit flips, because plain flips get trapped "
      "at the beam limit and that would make the baseline unfairly weak.")
    P("**Originality statement.** All code was written by the team. No `qiskit_optimization` (GroverOptimizer) or "
      "tutorial code is used. Only core Qiskit circuit primitives (`DiagonalGate`, `mcp`, `transpile`) and Qiskit Aer "
      "appear.")
    P("**References**", after=2)
    for r in [
        "L. K. Grover, “A fast quantum mechanical algorithm for database search,” STOC 1996; arXiv:quant-ph/9605043.",
        "C. Dürr and P. Høyer, “A quantum algorithm for finding the minimum,” 1996; arXiv:quant-ph/9607014.",
        "M. Boyer, G. Brassard, P. Høyer and A. Tapp, “Tight bounds on quantum searching,” Fortschritte der Physik 46, 493 (1998).",
        "A. Gilliam, S. Woerner and C. Gonciulea, “Grover Adaptive Search for Constrained Polynomial Binary "
        "Optimization,” Quantum 5, 428 (2021).",
        "R. Babbush et al., “Focus beyond quadratic speedups for error-corrected quantum advantage,” PRX Quantum 2, 010103 (2021).",
        "T. Bortfeld and W. Schlegel, “Optimization of beam orientations in radiation therapy: some theoretical "
        "considerations,” Physics in Medicine and Biology 38, 291 (1993).",
    ]:
        P(r, style="List Number", after=1, size=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-url", default="link added at submission")
    ap.add_argument("--quick", action="store_true", help="build from results_quick.json (layout check only)")
    a = ap.parse_args()
    res = json.loads((RES / ("results_quick.json" if a.quick else "results.json")).read_text())
    setup()
    build(res, a.video_url)
    OUT.parent.mkdir(exist_ok=True)
    doc.core_properties.title = TITLE
    doc.save(OUT)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
