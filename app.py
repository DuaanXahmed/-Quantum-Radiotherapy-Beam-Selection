import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

from phantom import build_phantom, GRID, BODY_RADIUS
from plans import PlanProblem, index_to_bits
from grover import grover_adaptive_search
import baselines as bl
from oracle_arithmetic import extract_qubo_coefficients, build_qft_arithmetic_oracle

st.set_page_config(page_title="Quantum Radiotherapy Optimizer", layout="wide")

@st.cache_resource
def get_simulator():
    return AerSimulator(method="statevector", max_parallel_threads=1)

st.title("⚛️ Quantum Grover Adaptive Search: Radiotherapy Beam Optimizer")
st.markdown(
    "Configure the **LINAC machine geometry**, **patient anatomy**, and **quantum search mode** below "
    "to compare **Grover Adaptive Search (GAS)** against classical clinical heuristics."
)

# --- Sidebar 1: Machine & Hardware Configuration ---
st.sidebar.header("1. Hospital LINAC Machine")
preset = st.sidebar.selectbox(
    "Machine Preset",
    ["Standard C-Arm (12 beams, k=4)", "Fast Prototype (8 beams, k=3)", "High-Res LINAC (14 beams, k=4)", "Custom Machine"]
)

if preset == "Standard C-Arm (12 beams, k=4)":
    n_beams, k_limit = 12, 4
elif preset == "Fast Prototype (8 beams, k=3)":
    n_beams, k_limit = 8, 3
elif preset == "High-Res LINAC (14 beams, k=4)":
    n_beams, k_limit = 14, 4
else:
    n_beams = st.sidebar.slider("Candidate Gantry Angles (n qubits)", 6, 14, 10, step=2)
    k_limit = st.sidebar.slider("Max Active Beams (k)", 2, min(6, n_beams), 4)

# --- Sidebar 2: Patient Anatomy & Clinical Weights ---
st.sidebar.header("2. Anatomy & Clinical Limits")
w_O = st.sidebar.slider("OAR Sparing Weight (w_O)", 0.0, 5.0, 1.0, step=0.5)
oar_max = st.sidebar.slider("Hard OAR Max Dose Limit (C2)", 0.35, 1.0, 0.60, step=0.05)
oar_x = st.sidebar.slider("OAR Horizontal Offset (x)", -3.0, 3.0, 0.3, step=0.3)
oar_y = st.sidebar.slider("OAR Vertical Position (y)", -7.5, -3.5, -5.7, step=0.2)

# --- Sidebar 3: Quantum Algorithm Settings ---
st.sidebar.header("3. Quantum Engine")
use_subspace = st.sidebar.checkbox("Use Dicke Subspace Prep (<= k beams only)", value=True)
seed = st.sidebar.number_input("Quantum Simulator Seed", value=1000, step=1)

ph = build_phantom(n_beams=n_beams, oar_centre=(oar_x, oar_y))
prob = PlanProblem(ph, w_T=1.0, w_O=w_O, w_N=0.05, k=k_limit, oar_max=oar_max)

if int(prob.feasible_mask().sum()) == 0:
    st.error("No feasible treatment plans exist under these strict constraints! Try raising C2 or k.")
    st.stop()

exact = prob.optimum()

if st.sidebar.button("🚀 Run Quantum GAS vs Classical Race", type="primary"):
    with st.spinner(f"Executing Grover Adaptive Search on {n_beams} qubits ({2**n_beams} states)..."):
        sim = get_simulator()
        res_gas = grover_adaptive_search(prob, seed=int(seed), sim=sim, use_subspace=use_subspace)
        res_greedy = bl.greedy(prob)
        res_sa = bl.simulated_annealing(prob, seed=int(seed) + 1000)
        res_rs = bl.random_search(prob, trials=200, seed=7)

    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantum Selected Angles", str([int(a) for a in prob.beams(res_gas["best"])]))
    c2.metric("GAS Oracle Calls", f"{res_gas['oracle_calls_to_opt']} calls",
              f"Subspace: {int(prob.feasible_mask(use_c2=False).sum())} states" if use_subspace else f"Full: {prob.N} states")
    c3.metric("Classical Random Queries", f"{res_rs['all_states']['analytic']:.0f} calls",
              f"{res_rs['all_states']['analytic'] / max(1, res_gas['oracle_calls_to_opt']):.1f}x slower than GAS")
    c4.metric("Objective Cost F(x)", f"{res_gas['best_cost']:.5f}",
              "✅ Exact Global Optimum" if res_gas["found_opt"] else "⚠️ Local Minimum")

    # Visual Plots
    bits = index_to_bits(res_gas["best"], n_beams)
    dose = bits.astype(float) @ ph.D
    terms = prob.terms(bits)

    c = (GRID - 1) / 2
    ext = (-c - 0.5, c + 0.5, -c - 0.5, c + 0.5)
    yy, xx = np.mgrid[c:-c - 1:-1, -c:c + 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    im = ax1.imshow(ph.to_grid(dose), cmap="Blues", vmin=0, vmax=1.1, extent=ext, origin="upper")
    ax1.contour(xx, yy, ph.to_grid(ph.tumour.astype(float), fill=0), levels=[0.5], colors=["#eb6834"], linewidths=2)
    ax1.contour(xx, yy, ph.to_grid(ph.oar.astype(float), fill=0), levels=[0.5], colors=["#6250d6"], linewidths=2)
    R = BODY_RADIUS + 1.5
    for a, ang in enumerate(ph.angles_deg):
        if bits[a]:
            t = np.deg2rad(ang)
            sx, sy = R * np.sin(t), R * np.cos(t)
            ax1.annotate("", xy=(0.55 * sx, 0.55 * sy), xytext=(sx, sy),
                         arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=2))
            ax1.text(1.13 * sx, 1.13 * sy, f"{int(ang)}°", ha="center", va="center", fontsize=8, weight="bold")
    ax1.set_xlim(-c - 3, c + 3)
    ax1.set_ylim(-c - 3, c + 3)
    ax1.axis("off")
    ax1.set_title(f"Optimal Radiation Dose (OAR Max: {terms['oar_maxdose']:.3f} ≤ {oar_max})")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    calls = [h["calls"] for h in res_gas["history"]]
    ys = [h["y"] for h in res_gas["history"]]
    ax2.step(calls, ys, where="post", color="#2a78d6", lw=2.2, label="GAS Threshold y")
    ax2.axhline(exact["cost"], color="#1baf7a", ls="--", label=f"Exact Optimum ({exact['cost']:.5f})")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cumulative Grover Oracle Calls")
    ax2.set_ylabel("Objective Cost F(x)")
    ax2.set_title("Grover Adaptive Search Convergence")
    ax2.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Head-to-Head Table
    st.subheader("🏁 Head-to-Head: Quantum GAS vs. Classical Solvers")
    st.table([
        {"Method": "Quantum GAS (Ours)", "Beams Chosen": str([int(a) for a in prob.beams(res_gas["best"])]),
         "Cost F(x)": f"{res_gas['best_cost']:.5f}", "Queries to Opt": res_gas["oracle_calls_to_opt"],
         "Found Exact Optimum?": "✅ Yes" if res_gas["found_opt"] else "❌ No"},
        {"Method": "Greedy Forward Selection", "Beams Chosen": str([int(a) for a in prob.beams(res_greedy["index"])]),
         "Cost F(x)": f"{res_greedy['cost']:.5f}", "Queries to Opt": res_greedy["evaluations"],
         "Found Exact Optimum?": "✅ Yes" if res_greedy["found_opt"] else "❌ FAILED (Local Trap)"},
        {"Method": "Simulated Annealing", "Beams Chosen": str([int(a) for a in prob.beams(res_sa["index"])]),
         "Cost F(x)": f"{res_sa['cost']:.5f}", "Queries to Opt": res_sa["evaluations_to_opt"] or res_sa["evaluations"],
         "Found Exact Optimum?": "✅ Yes" if res_sa["found_opt"] else "❌ No"},
        {"Method": "Unstructured Classical Search", "Beams Chosen": str([int(a) for a in prob.beams(exact["index"])]),
         "Cost F(x)": f"{exact['cost']:.5f}", "Queries to Opt": f"{res_rs['all_states']['analytic']:.1f} (expected)",
         "Found Exact Optimum?": "✅ Yes"},
    ])

    # Table-Free Gate-Level Arithmetic Oracle Stats
    with st.expander("🔬 View Table-Free Quantum Arithmetic Oracle (Gilliam et al. QFT Adder)"):
        c0, L, Q = extract_qubo_coefficients(prob)
        qft_orc = build_qft_arithmetic_oracle(prob, exact["second_cost"], m_ancilla=6)
        st.write(
            f"**How the real quantum hardware evaluates plans without a lookup table:** "
            f"Instead of precomputing {prob.N} states, we extract only the **{n_beams} linear beam weights `L_a`** "
            f"and **{n_beams*(n_beams-1)//2} pairwise beam overlaps `Q_ab`** from the dose matrix, and encode "
            f"`F(x) - y < 0` using controlled phase rotations and a 6-ancilla Inverse QFT."
        )
        st.json({
            "Total Circuit Qubits (Beam + Ancilla)": qft_orc.num_qubits,
            "Logical Gate Counts (CRz / CCRz / QFT)": {k: int(v) for k, v in qft_orc.count_ops().items()},
            "Linear QUBO Coefficients L_a (rounded)": np.round(L, 4).tolist(),
        })