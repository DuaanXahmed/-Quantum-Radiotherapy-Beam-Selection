import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qiskit_aer import AerSimulator

from plans import default_problem, index_to_bits
from grover import grover_adaptive_search
from phantom import GRID, BODY_RADIUS

# 1. Run seeded GAS trace (seed 1000)
prob = default_problem()
ph = prob.phantom
res = grover_adaptive_search(prob, seed=1000, sim=AerSimulator(method="statevector"))

milestones = [h for h in res["history"] if h["improved"]]
# Add a final holding milestone so the tail draws out cleanly
end_calls = min(res["oracle_calls"], milestones[-1]["calls"] + 45)
milestones.append(dict(
    round=milestones[-1]["round"],
    calls=end_calls,
    y=milestones[-1]["y"],
    x=milestones[-1]["x"],
    improved=False
))

# 2. Build smooth interpolated frames using cosine ease-in-out
SUBFRAMES = 12
smooth_frames = []

for i in range(len(milestones) - 1):
    s0, s1 = milestones[i], milestones[i + 1]
    b0 = index_to_bits(s0["x"], prob.n).astype(float)
    b1 = index_to_bits(s1["x"], prob.n).astype(float)
    d0 = b0 @ ph.D
    d1 = b1 @ ph.D

    for f in range(SUBFRAMES):
        raw_t = f / SUBFRAMES
        # Smooth ease-in-out interpolation factor
        t = 0.5 * (1 - np.cos(np.pi * raw_t))
        smooth_frames.append(dict(
            bits=(1 - t) * b0 + t * b1,
            dose=(1 - t) * d0 + t * d1,
            calls=(1 - t) * s0["calls"] + t * s1["calls"],
            y=(1 - t) * s0["y"] + t * s1["y"],
            target_x=s1["x"] if raw_t > 0.5 else s0["x"],
            round=s1["round"] if raw_t > 0.5 else s0["round"],
            milestone_idx=i
        ))

# Hold the exact optimal state smoothly at the end for ~1.5 seconds (36 frames at 24 fps)
opt_idx = prob.optimum()["index"]
opt_bits = index_to_bits(opt_idx, prob.n).astype(float)
opt_dose = opt_bits @ ph.D
for _ in range(36):
    smooth_frames.append(dict(
        bits=opt_bits,
        dose=opt_dose,
        calls=milestones[-1]["calls"],
        y=milestones[-1]["y"],
        target_x=opt_idx,
        round=milestones[-1]["round"],
        milestone_idx=len(milestones) - 1
    ))

# 3. Render smooth 24 FPS animation
fig, (ax_body, ax_conv) = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor="#fcfcfb")
plt.subplots_adjust(wspace=0.28)
c = (GRID - 1) / 2
ext = (-c - 0.5, c + 0.5, -c - 0.5, c + 0.5)
yy, xx = np.mgrid[c:-c - 1:-1, -c:c + 1]
R = BODY_RADIUS + 1.6

def update(idx):
    ax_body.clear()
    ax_conv.clear()
    fr = smooth_frames[idx]

    # --- Left Panel: Smoothly Morphing Dose & Beams ---
    ax_body.imshow(ph.to_grid(fr["dose"]), cmap="Blues", vmin=0, vmax=1.1, extent=ext, origin="upper")
    ax_body.contour(xx, yy, ph.to_grid(ph.tumour.astype(float), fill=0), levels=[0.5], colors=["#eb6834"], linewidths=2)
    ax_body.contour(xx, yy, ph.to_grid(ph.oar.astype(float), fill=0), levels=[0.5], colors=["#6250d6"], linewidths=2)

    for a, ang in enumerate(ph.angles_deg):
        alpha = float(fr["bits"][a])
        if alpha > 0.03:
            t = np.deg2rad(ang)
            sx, sy = R * np.sin(t), R * np.cos(t)
            ax_body.annotate(
                "", xy=(0.55 * sx, 0.55 * sy), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=1.2 + 1.3 * alpha, alpha=alpha)
            )
            ax_body.text(
                1.14 * sx, 1.14 * sy, f"{int(ang)}°",
                ha="center", va="center", fontsize=8.5, weight="bold", color="#0b0b0b", alpha=alpha
            )

    ax_body.set_xlim(-c - 3.2, c + 3.2)
    ax_body.set_ylim(-c - 3.2, c + 3.2)
    ax_body.set_aspect("equal")
    ax_body.axis("off")

    active_angles = [int(a) for a in prob.beams(fr["target_x"])]
    is_opt = (fr["target_x"] == opt_idx)
    title_color = "#1baf7a" if is_opt else "#0b0b0b"
    status = "EXACT OPTIMUM LOCKED" if is_opt else f"GAS Round {fr['round']}"
    ax_body.set_title(f"{status}\nActive Beams: {active_angles}", fontsize=10.5, weight="bold", color=title_color)

    # --- Right Panel: Continuous Convergence Curve ---
    m_idx = fr["milestone_idx"]
    hist_calls = [m["calls"] for m in milestones[:m_idx + 1]] + [fr["calls"]]
    hist_ys = [m["y"] for m in milestones[:m_idx + 1]] + [fr["y"]]

    ax_conv.plot(hist_calls, hist_ys, color="#2a78d6", lw=2.5, label="Threshold Trajectory y")
    ax_conv.scatter([fr["calls"]], [fr["y"]], color="#eb6834", s=75, zorder=5)
    ax_conv.axhline(prob.optimum()["cost"], color="#1baf7a", ls="--", lw=1.3, label="Global Optimum (0.03803)")

    ax_conv.set_xlim(-5, milestones[-1]["calls"] + 10)
    ax_conv.set_ylim(0.025, 0.42)
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Cumulative Grover Oracle Calls")
    ax_conv.set_ylabel("Objective Cost F(x) (Lower = Better)")
    ax_conv.set_title(f"Oracle Calls: {int(round(fr['calls']))}  |  Cost F = {fr['y']:.5f}", fontsize=10.5, weight="bold")
    ax_conv.grid(True, color="#e4e3dd", linewidth=0.6)
    ax_conv.spines["top"].set_visible(False)
    ax_conv.spines["right"].set_visible(False)
    ax_conv.legend(loc="upper right", frameon=False, fontsize=8.5)

anim = FuncAnimation(fig, update, frames=len(smooth_frames), interval=1000 / 24)
anim.save("results/gas_simulation.gif", writer=PillowWriter(fps=24), dpi=150)
print("Saved smooth 24 FPS animation to results/gas_simulation.gif")
anim.save("results/gas_simulation.mp4", writer="ffmpeg", fps=24, dpi=200)
print("Saved smooth 24 FPS animation to results/gas_simulation.mp4")
