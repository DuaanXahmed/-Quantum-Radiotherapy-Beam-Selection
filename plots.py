"""Make the report figures from results/results.json.   Usage: python plots.py

Style: light surface, thin marks, solid hairline grid, categorical colours taken in
fixed order from a validated palette (blue, orange; neutral grey for context).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

RES = Path(__file__).parent / "results"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"
BLUE_TINT = "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False,
})


def load():
    return json.loads((RES / "results.json").read_text())


def order_str(o):
    return "→".join(f"F{i}" for i in o)


# --------------------------------------------------------------------------- Fig 1
def fig_tradeoff(R):
    allo = R["all_orderings"]
    sweep = {r["lambda"]: r for r in R["lambda_sweep"]}
    mis, truth = sweep[0]["order"], sweep[3]["order"]
    cnt = Counter((o["total_overlap"], o["weak_junctions"]) for o in allo)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.grid(axis="x", visible=False)
    for (x, y), c in cnt.items():
        ax.scatter(x, y, s=70 * c, color=GREY, edgecolor=SURFACE, linewidth=2, zorder=2)
        if c > 1:
            ax.text(x, y + 0.2 + 0.012 * c, f"{c}", ha="center", va="bottom", fontsize=8, color=INK2)
    for order, col, lab in ((mis, ORANGE, "max-overlap optimum (λ = 0)"), (truth, BLUE, "reliability-aware optimum (λ > 1)")):
        m = next(o for o in allo if o["order"] == list(order))
        ax.scatter(m["total_overlap"], m["weak_junctions"], s=190, color=col, edgecolor=SURFACE,
                   linewidth=2, zorder=4)
    m0 = next(o for o in allo if o["order"] == list(mis))
    m3 = next(o for o in allo if o["order"] == list(truth))
    ax.plot([m3["total_overlap"], m0["total_overlap"]], [m3["weak_junctions"], m0["weak_junctions"]],
            color=INK2, linewidth=1.4, zorder=3)
    ax.annotate(f"{order_str(mis)}\nmisassembly: more overlap,\nbut one weak junction",
                (m0["total_overlap"], m0["weak_junctions"]), xytext=(-40, 52), textcoords="offset points",
                ha="right", va="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkA=2, shrinkB=8))
    ax.annotate(f"{order_str(truth)}\nrecovers the true genome", (m3["total_overlap"], m3["weak_junctions"]),
                xytext=(-16, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Total overlap along the ordering (bases), higher = more contiguous")
    ax.set_ylabel("Weak junctions (overlap < 3), fewer = better")
    ax.set_title("Two competing objectives: the max-overlap ordering is not the correct one",
                 loc="left", fontsize=11.5, fontweight="bold", pad=12)
    xs = [k[0] for k in cnt]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 1.0)
    ax.set_ylim(-0.5, max(k[1] for k in cnt) + 0.7)
    ax.set_xticks(range(min(xs), max(xs) + 1))
    ax.set_yticks(range(0, max(k[1] for k in cnt) + 1))
    handles = [Line2D([], [], marker="o", ls="", color=GREY, markersize=9, markeredgecolor=SURFACE,
                      label="other orderings (number = how many share a point; no number = 1; all 24 shown)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, markersize=10, markeredgecolor=SURFACE,
                      label="exact optimum with λ = 0 (maximise overlap only)"),
               Line2D([], [], marker="o", ls="", color=BLUE, markersize=10, markeredgecolor=SURFACE,
                      label="exact optimum with λ > 1 (penalise weak junctions)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8.5,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(RES / "fig1_tradeoff.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig 2
def fig_qaoa_vs_p(R):
    runs = [r for r in R["qaoa"] if r["lambda"] == 3]
    ps = sorted({r["p"] for r in runs})
    uni_feas, uni_opt = 24 / 2 ** 16, 1 / 2 ** 16
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    spec = [("p_feasible", "Probability a sample is a valid ordering", uni_feas),
            ("p_optimum", "Probability a sample is the optimal ordering", uni_opt)]
    for ax, (key, title, uni) in zip(axes, spec):
        ax.grid(axis="x", visible=False)
        ax.set_yscale("log")
        ax.axhline(uni, color=MUTED, linewidth=1.4, zorder=1)
        ax.text(ps[0] - 0.05, uni * 1.25, "uniform random sampling", color=INK2, fontsize=8.5, va="bottom")
        for objective, col, lab in (("expectation", BLUE, "expected energy"), ("cvar", ORANGE, "CVaR (α = 0.1)")):
            pts = sorted(((r["p"], r[key]) for r in runs if r["objective"] == objective))
            ax.plot(*zip(*pts), color=col, linewidth=1.8, zorder=3)
            ax.scatter(*zip(*pts), s=48, color=col, edgecolor=SURFACE, linewidth=2, zorder=4)
            dy = 8 if objective == "cvar" else -8
            ax.annotate(lab, pts[-1], xytext=(8, dy), textcoords="offset points", color=INK, fontsize=8.5,
                        va="center")
        ax.set_xticks(ps)
        ax.set_xlim(ps[0] - 0.15, ps[-1] + 0.95)
        ax.set_xlabel("QAOA layers p")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Probability (log scale)")
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE, label="objective: expected energy"),
               Line2D([], [], color=ORANGE, marker="o", markersize=7, markeredgecolor=SURFACE, label="objective: CVaR (α = 0.1)"),
               Line2D([], [], color=MUTED, linewidth=1.4, label="uniform random bitstrings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("QAOA concentrates probability on valid and optimal orderings (16 qubits, ideal simulator)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(RES / "fig2_qaoa_vs_p.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig 3
def fig_distribution(R):
    runs = [r for r in R["qaoa"] if r["lambda"] == 3]
    best = max(runs, key=lambda r: r["p_optimum"])
    sweep = {r["lambda"]: r for r in R["lambda_sweep"]}
    truth, mis = tuple(sweep[3]["order"]), tuple(sweep[0]["order"])
    items = sorted(((tuple(json.loads(k.replace("(", "[").replace(")", "]"))), v)
                    for k, v in best["feasible_probs"].items()), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    ax.grid(axis="y", visible=False)
    cols = [BLUE if o == truth else ORANGE if o == mis else GREY for o, _ in items]
    ax.barh(range(len(items)), [v * 100 for _, v in items], color=cols, height=0.66, edgecolor=SURFACE,
            linewidth=0)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([order_str(o) for o, _ in items], fontsize=7.8)
    for tick, (o, _) in zip(ax.get_yticklabels(), items):
        if o in (truth, mis):
            tick.set_color(INK)
            tick.set_fontweight("bold")
    for i, (o, v) in enumerate(items):
        if o in (truth, mis):
            ax.text(v * 100 + 0.03, i, f"{v * 100:.2f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Probability of measuring this ordering (%)")
    obj = "CVaR" if best["objective"] == "cvar" else "expected-energy"
    ax.set_title(f"QAOA output over the 24 valid orderings (p = {best['p']}, {obj} objective)",
                 loc="left", fontsize=11.5, fontweight="bold", pad=34)
    ax.text(0, 1.045, f"{best['p_feasible'] * 100:.1f}% of all shots are valid orderings (uniform random: 0.037%).\n"
            f"The optimum is measured {best['p_optimum'] * 100:.2f}% of the time (uniform random: 0.0015%).",
            transform=ax.transAxes, fontsize=8.8, color=INK2, va="top", linespacing=1.5)
    ax.legend(handles=[Patch(color=BLUE, label="optimal for λ = 3 (the true genome)"),
                       Patch(color=ORANGE, label="optimal for λ = 0 (misassembly)"),
                       Patch(color=GREY, label="other valid orderings")],
              loc="lower right", fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(RES / "fig3_qaoa_distribution.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- Fig 4
def box(ax, x, y, w, h, text, quantum=False, title_h=0):
    fc, ec = (BLUE_TINT, BLUE) if quantum else ("#f0efec", AXIS)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    head, *rest = text.split("\n")
    ax.text(x + w / 2, y + h - 2.4, head, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 5.8, "\n".join(rest), ha="center", va="top", fontsize=8, color=INK2,
            linespacing=1.45)


def arrow(ax, p0, p1, both=False):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=INK2,
                                                      lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=11))


def fig_architecture(R):
    fig, ax = plt.subplots(figsize=(13.4, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.grid(False)
    ax.text(1, 59.5, "Architecture: quantum-assisted DNA fragment ordering (Qiskit QAOA)", fontsize=12.5,
            fontweight="bold", va="center")
    y1, h = 33, 19
    box(ax, 1, y1, 14.5, h, "1. Input\nDNA fragments (N = 4)\nsuffix/prefix overlap\nmatrix o_ij\n[instance.py]")
    box(ax, 18.5, y1, 16.5, h, "2. Problem model\ncontiguity vs reliability\nW_ij = o_ij − λ·[o_ij < τ]\nλ sweeps the trade-off\n[instance.py]")
    box(ax, 38, y1, 16.5, h, "3. QUBO → Ising\nx[i,t]: N² = 16 qubits\none-hot penalties A\nH_C as SparsePauliOp\n[qubo.py, qaoa.py]")
    box(ax, 57.5, y1, 20, h, "4. QAOA circuit (Qiskit)\nH on all qubits, then p ×\n[RZ·RZZ(γ) cost, RX(β) mixer]\nAer statevector / shots\n[qaoa.py]", quantum=True)
    box(ax, 80.5, y1, 18, h, "6. Decode & filter\nkeep valid permutations\nbitstring → ordering\n→ assembled sequence\n[qubo.py]")
    box(ax, 57.5, 8, 20, 17, "5. Classical optimiser\nCOBYLA updates (γ, β)\nobjective: ⟨H_C⟩ or CVaR\n[qaoa.py]")
    box(ax, 38, 8, 16.5, 17, "Classical baselines\nbrute force (exact)\ngreedy · SA · random\n[baselines.py]")
    box(ax, 80.5, 8, 18, 17, "7. Evaluation\nP(optimum), P(valid),\nenergy vs uniform,\nvs exact optimum\n[run_experiments.py]")
    ym = y1 + h / 2
    arrow(ax, (15.5, ym), (18.5, ym))
    arrow(ax, (35, ym), (38, ym))
    arrow(ax, (54.5, ym), (57.5, ym))
    arrow(ax, (77.5, ym), (80.5, ym))
    arrow(ax, (67.5, y1), (67.5, 25), both=True)
    arrow(ax, (89.5, y1), (89.5, 25))
    arrow(ax, (46.25, y1), (46.25, 25))
    ax.plot([46.25, 46.25], [8, 3.5], color=INK2, lw=1.3)
    ax.plot([46.25, 89.5], [3.5, 3.5], color=INK2, lw=1.3)
    arrow(ax, (89.5, 3.5), (89.5, 8))
    ax.add_patch(FancyBboxPatch((1, 8), 14.5, 6.5, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE_TINT, ec=BLUE, lw=1.4))
    ax.text(8.25, 11.25, "quantum (Qiskit)", ha="center", va="center", fontsize=8.5, color=INK)
    ax.add_patch(FancyBboxPatch((1, 16.5), 14.5, 6.5, boxstyle="round,pad=0,rounding_size=1.2", fc="#f0efec", ec=AXIS, lw=1.4))
    ax.text(8.25, 19.75, "classical", ha="center", va="center", fontsize=8.5, color=INK)
    fig.savefig(RES / "fig4_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    R = load()
    fig_tradeoff(R)
    fig_qaoa_vs_p(R)
    fig_distribution(R)
    fig_architecture(R)
    print("figures written to", RES)


if __name__ == "__main__":
    main()
