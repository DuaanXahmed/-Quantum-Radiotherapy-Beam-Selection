"""Figures for the report, read from results/results.json (python plots.py [--quick])."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from phantom import BODY_RADIUS, GRID, build_phantom  # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"
# reference palette (validated categorical order) + neutral inks
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID_C, SURF = "#0b0b0b", "#52514e", "#9a9890", "#e4e3dd", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID_C,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    fig.savefig(RES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", RES / name)


def to_xy(ph, c):
    return ph.x, ph.y


# ------------------------------------------------------------------ 1 phantom + dose
def fig_phantom(res):
    ph = build_phantom(res["instance"]["n"])
    opt = res["optimum"]
    bits = [(opt["index"] >> a) & 1 for a in range(ph.n_beams)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    c = (GRID - 1) / 2
    ext = (-c - .5, c + .5, -c - .5, c + .5)

    lab = np.zeros(ph.n_voxels)
    lab[ph.normal], lab[ph.tumour], lab[ph.oar] = 1, 2, 3
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([SURF, "#dcdad2", ORANGE, VIOLET])
    ax = axs[0]
    ax.imshow(np.nan_to_num(ph.to_grid(lab), nan=0), cmap=cmap, vmin=0, vmax=3, extent=ext,
              origin="upper", interpolation="nearest")
    R = BODY_RADIUS + 1.6
    for a, ang in enumerate(ph.angles_deg):
        t = np.deg2rad(ang)
        sx, sy = R * np.sin(t), R * np.cos(t)
        ax.annotate("", xy=(0.62 * sx, 0.62 * sy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
        ax.text(1.12 * sx, 1.12 * sy, f"{int(ang)}°", ha="center", va="center", fontsize=8, color=INK2)
    ax.set_title("Synthetic phantom and 12 candidate beams")
    ax.legend(handles=[Patch(color=ORANGE, label=f"Tumour ({ph.meta['n_tumour']} voxels)"),
                       Patch(color=VIOLET, label=f"Organ at risk ({ph.meta['n_oar']} voxels)"),
                       Patch(color="#dcdad2", label=f"Normal tissue ({ph.meta['n_normal']} voxels)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, handlelength=1)

    ax = axs[1]
    dose = np.asarray(bits, float) @ ph.D
    im = ax.imshow(ph.to_grid(dose), cmap="Blues", vmin=0, vmax=1.1, extent=ext, origin="upper",
                   interpolation="nearest")
    for mask, col in ((ph.tumour, ORANGE), (ph.oar, VIOLET)):
        img = ph.to_grid(mask.astype(float), fill=0)
        yy, xx = np.mgrid[c:-c - 1:-1, -c:c + 1]
        ax.contour(xx, yy, img, levels=[0.5], colors=[col], linewidths=2)
    for a, ang in enumerate(ph.angles_deg):
        if bits[a]:
            t = np.deg2rad(ang)
            sx, sy = R * np.sin(t), R * np.cos(t)
            ax.annotate("", xy=(0.55 * sx, 0.55 * sy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
            ax.text(1.12 * sx, 1.12 * sy, f"{int(ang)}°", ha="center", va="center", fontsize=9,
                    color=INK, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("dose (prescription = 1.0)")
    ax.set_title(f"Dose of the exact optimal plan {[int(b) for b in opt['beams']]}")
    for ax in axs:
        ax.set_xlim(-c - 3, c + 3)
        ax.set_ylim(-c - 3, c + 3)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.04, f"Tumour mean dose {opt['tumour_mean']:.3f}, min {opt['tumour_min']:.3f}; "
             f"OAR mean {opt['oar_mean']:.3f}, max {opt['oar_maxdose']:.3f} (limit {res['config']['oar_max']})",
             ha="center", fontsize=9, color=INK2)
    save(fig, "fig1_phantom_dose.png")


# ------------------------------------------------------------------ 2 Pareto trade-off
def fig_pareto(res):
    P = res["pareto"]
    te, om = np.array(P["tumour_err"]), np.array(P["oar_mean"])
    fi = np.array(P["feasible_index"])
    pos = {int(i): k for k, i in enumerate(fi)}
    front = [pos[i] for i in P["front_index"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(om, te, s=14, color=MUTED, alpha=0.55, lw=0, label=f"all {len(fi)} feasible plans")
    fo = sorted(front, key=lambda k: om[k])
    ax.plot(om[fo], te[fo], color=INK2, lw=1.2, ls="--", label="Pareto front (non-dominated)", zorder=2)
    groups = {}
    for s in res["sweep"]:
        groups.setdefault(s["index"], []).append(s)
    cols = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
    for col, (idx, ss) in zip(cols, groups.items()):
        s = ss[0]
        ws = ", ".join(f"{x['w_O']:g}" for x in ss)
        ax.scatter([s["oar_mean"]], [s["tumour_err"]], s=110, color=col, edgecolor=SURF, lw=2, zorder=4,
                   label=f"optimum for w_O = {ws}: beams {[int(b) for b in s['beams']]}")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("OAR mean dose  (lower = organ better spared)")
    ax.set_ylabel("tumour error  mean (d − 1)²  (lower = better coverage)")
    ax.set_title("Trade-off: tumour coverage vs organ-at-risk sparing (exact enumeration)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, "fig2_pareto.png")


# ------------------------------------------------------------------ 3 oracle-call comparison
def fig_queries(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    sc = res["scaling"]
    N = np.array([s["N"] for s in sc], float)
    ax = axs[0]
    ax.plot(N, [s["brute_force"] for s in sc], "-o", color=INK2, ms=6, label="brute force (all N plans)")
    ax.plot(N, [s["random_all_expected"] for s in sc], "-o", color=ORANGE, ms=6,
            label="random search, all plans  (N+1)/2")
    ax.plot(N, [s["random_feasible_expected"] for s in sc], "-o", color=AQUA, ms=6,
            label="random search, feasible only  (M+1)/2")
    ax.plot(N, [s["gas_calls_to_opt"] for s in sc], "-o", color=BLUE, ms=8, lw=2.5,
            label="GAS (quantum): oracle calls to optimum")
    ax.plot(N, np.sqrt(N), ":", color=MUTED, lw=1.5, label="√N reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N)
    ax.set_xticklabels([f"{int(n)}\n(n={s['n']})" for n, s in zip(N, sc)])
    ax.set_xlabel("search-space size N = 2ⁿ  (n candidate beams)")
    ax.set_ylabel("cost-function / oracle queries")
    ax.set_title("Queries to reach the exact optimum vs problem size")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1]
    g = res["gas"]
    calls = np.array([t["oracle_calls_to_opt"] for t in g["per_trial"] if t["found_opt"]], float)
    bins = np.logspace(math.log10(8), math.log10(5000), 36)
    ax.hist(np.maximum(calls, 8), bins=bins, color=BLUE, edgecolor=SURF, lw=0.6,
            label=f"GAS oracle calls, {len(calls)}/{g['trials']} seeded runs")
    rs = res["baselines"]["random_search"]
    tr = ax.get_xaxis_transform()
    for (v, col, lab, yy, ha) in ((g["oracle_calls_to_opt"]["mean"], BLUE, "GAS mean", 0.97, "left"),
                                  (rs["feasible_only"]["analytic"], AQUA, "random search,\nfeasible only", 0.80, "right"),
                                  (rs["all_states"]["analytic"], ORANGE, "random search,\nall plans", 0.62, "right"),
                                  (res["instance"]["N"], INK2, "brute\nforce", 0.80, "left")):
        ax.axvline(v, color=col, lw=2, ls="-" if lab == "GAS mean" else "--")
        ax.text(v * (1.08 if ha == "left" else 0.93), yy, f"{lab}\n{v:.0f}", transform=tr, fontsize=8,
                color=INK, va="top", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(8, 6000)
    ax.set_xlabel("oracle calls (GAS) / evaluations (classical) to reach the optimum")
    ax.set_ylabel("number of seeded GAS runs")
    ax.set_title(f"n = {res['instance']['n']}: GAS over {g['trials']} seeds ({len(calls)} reach the optimum)")
    save(fig, "fig3_oracle_calls.png")


# ------------------------------------------------------------------ 4 GAS convergence
def fig_convergence(res):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    cols = [BLUE, ORANGE, AQUA, YELLOW, VIOLET, "#e87ba4"]
    for col, tr in zip(cols, res["gas"]["traces"]):
        ax.step(tr["calls"], tr["y"], where="post", color=col, lw=2, label=f"seed {tr['seed']}")
        ax.plot(tr["calls"][-1], tr["y"][-1], "o", color=col, ms=6)
    ax.axhline(res["optimum"]["cost"], color=INK, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], res["optimum"]["cost"], "  exact optimum F*", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("cumulative oracle calls (Grover iterations)")
    ax.set_ylabel("threshold y = best F found")
    ax.set_title("Grover Adaptive Search convergence (6 seeded runs)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig4_gas_convergence.png")


# ------------------------------------------------------------------ 5 Grover theory vs simulation
def fig_grover_check(res):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for col, c in zip([BLUE, ORANGE, AQUA], res["grover_check"]):
        r = np.array(c["r"])
        rr = np.linspace(0, r.max(), 400)
        th = math.asin(math.sqrt(c["M"] / res["instance"]["N"]))
        ax.plot(rr, np.sin((2 * rr + 1) * th) ** 2, color=col, lw=1.6, alpha=0.8)
        ax.plot(r, c["simulated"], "o", color=col, ms=4.5, mec=SURF, mew=0.8,
                label=f"M = {c['M']} marked plans (max |sim − theory| = {c['max_abs_error']:.0e})")
    ax.plot([], [], color=INK2, label="theory sin²((2r+1)θ), sin²θ = M/N")
    ax.set_xlabel("number of Grover iterations r")
    ax.set_ylabel("P(measure a marked plan)")
    ax.set_title("Grover amplification: Aer statevector simulation vs theory (N = 4096)")
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "fig5_grover_theory.png")


# ------------------------------------------------------------------ 6 architecture
def fig_architecture(res):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    QF, QE, CF, CE = "#e3edfa", BLUE, "#efeee9", INK2

    def box(x, y, w, h, title, sub, q=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=QF if q else CF, ec=QE if q else CE, lw=2 if q else 1.3))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h - 0.68, sub, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)

    def arrow(p, q, text=None, col=INK2, rad=0.0, off=(0, 0.12)):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=col, lw=1.6,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center",
                    fontsize=8, color=col)

    st = res["circuit_stats"]
    box(0.1, 3.0, 2.2, 1.8, "Phantom & dose", "21×21 voxels\ntumour / OAR / normal\n12 beams → D[a,v]")
    box(2.75, 3.0, 2.3, 1.8, "Cost + constraints", "F(x) = tumour + OAR\n+ normal-tissue terms\nC1: Σx ≤ 4  C2: OAR ≤ 0.6")
    box(5.5, 3.0, 2.3, 1.8, "Phase oracle O_y", "flip sign iff feasible\nand F(x) < y\nDiagonalGate (±1 table)", q=True)
    box(8.25, 3.0, 3.6, 1.8, "Grover circuit (Aer)", f"H⊗¹² → (O_y · W)^r → measure\nr ~ U[0, ⌈m⌉−1] (BBHT)\n"
        f"12 qubits, 0 ancillas, {st['cx_per_iteration']} CX/iter", q=True)
    box(8.25, 0.35, 3.6, 1.75, "Measure 1 shot → plan x", "evaluate F(x) classically\nfeasible & F(x) < y ?")
    box(4.3, 0.35, 3.5, 1.75, "Threshold update", "yes: y ← F(x), best ← x, m ← 1\nno:  m ← min(6/5·m, √N)\nstop: 25 fails / 30√N calls")
    box(0.1, 0.35, 3.3, 1.75, "Best plan → evaluation", "vs brute force, random search,\ngreedy, simulated annealing,\nNNLS bound; Pareto sweep")
    arrow((2.3, 3.9), (2.75, 3.9))
    arrow((5.05, 3.9), (5.5, 3.9))
    arrow((7.8, 3.9), (8.25, 3.9))
    arrow((10.05, 3.0), (10.05, 2.1), "bitstring", off=(0.45, 0))
    arrow((8.25, 1.2), (7.8, 1.2))
    arrow((6.05, 2.1), (6.65, 3.0), "new y → rebuild O_y", col=BLUE, rad=-0.25, off=(-0.95, 0.05))
    arrow((4.3, 1.2), (3.4, 1.2), "stop", off=(0, 0.1))
    ax.legend(handles=[Patch(fc=QF, ec=QE, lw=2, label="quantum component (Qiskit circuit on AerSimulator)"),
                       Patch(fc=CF, ec=CE, label="classical component (NumPy / SciPy)")],
              loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=9)
    save(fig, "fig6_architecture.png")


def main(quick=False):
    res = json.loads((RES / ("results_quick.json" if quick else "results.json")).read_text())
    for f in (fig_phantom, fig_pareto, fig_queries, fig_convergence, fig_grover_check, fig_architecture):
        f(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="plot results_quick.json")
    main(ap.parse_args().quick)
