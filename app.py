import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

from phantom import build_phantom, GRID, BODY_RADIUS
from plans import PlanProblem, index_to_bits
from grover import grover_adaptive_search

st.set_page_config(page_title="Quantum Radiotherapy Optimizer", layout="wide")
st.title("⚛️ Quantum Grover Adaptive Search for Radiotherapy Beam Selection")
st.markdown("Adjust clinical priorities below and run **Grover Adaptive Search (GAS)** on Qiskit AerSimulator to find the optimal gantry angles.")

# Sidebar Controls
st.sidebar.header("Clinical Parameters")
w_O = st.sidebar.slider("Organ-at-Risk (OAR) Sparing Weight (w_O)", 0.0, 5.0, 1.0, step=0.5)
oar_max = st.sidebar.slider("Hard OAR Max Dose Limit (C2)", 0.3, 1.0, 0.6, step=0.05)
seed = st.sidebar.number_input("Quantum Simulator Seed", value=1000, step=1)

# Build Problem
ph = build_phantom(n_beams=12)
prob = PlanProblem(ph, w_T=1.0, w_O=w_O, w_N=0.05, k=4, oar_max=oar_max)
exact = prob.optimum()

if st.sidebar.button("🚀 Run Quantum GAS Search", type="primary"):
    with st.spinner("Running Grover iterations on 12 qubits (4,096 states)..."):
        sim = AerSimulator(method="statevector")
        res = grover_adaptive_search(prob, seed=int(seed), sim=sim)

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Gantry Angles", str([int(a) for a in prob.beams(res["best"])]))
    col2.metric("Quantum Oracle Calls", f"{res['oracle_calls_to_opt']} calls", f"vs 2048 classical random")
    col3.metric("Objective Cost F(x)", f"{res['best_cost']:.5f}", "Exact Global Optimum" if res["found_opt"] else "Near Optimum")

    # Plot Dose Distribution
    bits = index_to_bits(res["best"], 12)
    dose = bits.astype(float) @ ph.D
    terms = prob.terms(bits)

    c = (GRID - 1) / 2
    ext = (-c - 0.5, c + 0.5, -c - 0.5, c + 0.5)
    yy, xx = np.mgrid[c:-c - 1:-1, -c:c + 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    im = ax1.imshow(ph.to_grid(dose), cmap="Blues", vmin=0, vmax=1.1, extent=ext, origin="upper")
    ax1.contour(xx, yy, ph.to_grid(ph.tumour.astype(float), fill=0), levels=[0.5], colors=["#eb6834"], linewidths=2)
    ax1.contour(xx, yy, ph.to_grid(ph.oar.astype(float), fill=0), levels=[0.5], colors=["#6250d6"], linewidths=2)
    ax1.set_title(f"Radiation Dose Map (OAR Max: {terms['oar_maxdose']:.3f} / {oar_max})")
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    calls = [h["calls"] for h in res["history"]]
    ys = [h["y"] for h in res["history"]]
    ax2.step(calls, ys, where="post", color="#2a78d6", lw=2)
    ax2.axhline(exact["cost"], color="#1baf7a", ls="--", label="Brute-Force Exact Optimum")
    ax2.set_yscale("log")
    ax2.set_xlabel("Grover Oracle Calls")
    ax2.set_ylabel("Best Cost Threshold y")
    ax2.set_title("GAS Threshold Convergence")
    ax2.legend()
    st.pyplot(fig)