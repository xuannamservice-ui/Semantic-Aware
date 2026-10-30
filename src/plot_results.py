"""
Publication Plotting Script
Generates 300 DPI publication-quality figures matching IEEE Transactions formatting.
"""

import json, os, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE style configuration
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# -------------------------------------------------------------
# Figure 1: Correlated Fading & Burst Erasure Distribution
# -------------------------------------------------------------
print("[*] Plotting Figure 1: Correlated Fading Burst Erasure Statistics...")
with open(DATA_DIR / "exp1_temporal_correlated_1m_results.json") as f:
    d1 = json.load(f)

bursts = list(range(1, 11))
prob_s = [d1["burst_probabilities"]["single_anchor"][f"P_burst_ge_{l}"] for l in bursts]
prob_d = [d1["burst_probabilities"]["dual_anchor"][f"P_burst_ge_{l}"] for l in bursts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

# Subplot 1: Probability of Burst Length >= L
ax1.semilogy(bursts, prob_s, 'o-', color='#D9534F', label='Single Anchor (Mode +1)', linewidth=2.2)
ax1.semilogy(bursts, prob_d, 's-', color='#0275D8', label='Proposed Anchor Pair (-2, +1)', linewidth=2.2)
ax1.axhline(1e-4, color='black', linestyle=':', label='URLLC Reliability Threshold (10^-4)')
ax1.set_xlabel('Consecutive Lost Slots (Burst Length $L$)')
ax1.set_ylabel(r'Probability $\Pr(\mathrm{Burst\ Length} \geq L)$')
ax1.set_title('(a) Burst Erasure Length Tail Probability')
ax1.grid(True)
ax1.legend()
ax1.set_xticks(bursts)

# Subplot 2: Lyapunov Virtual Queue Backlog Trajectory
trace_s = d1["sampled_queue_trajectories"]["trace_single"][:400]
trace_d = d1["sampled_queue_trajectories"]["trace_dual"][:400]
slots = np.arange(len(trace_s)) * 100

ax2.plot(slots, trace_s, color='#D9534F', alpha=0.85, label='Single Anchor (Backlog diverges)')
ax2.plot(slots, trace_d, color='#5CB85C', linewidth=2.0, label='Proposed Replicated Anchors (Stable)')
ax2.set_xlabel('Slot Index $t$')
ax2.set_ylabel('Virtual Queue Backlog $Z(t)$')
ax2.set_title('(b) Lyapunov Virtual Queue Stability')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
fig_path1 = FIG_DIR / "fig_exp1_burst_erasures_and_stability.png"
plt.savefig(fig_path1, dpi=300)
plt.close()
print(f"[+] Saved {fig_path1}")

# -------------------------------------------------------------
# Figure 2: Hardware Constraints & Laser RIN Sensitivity
# -------------------------------------------------------------
print("[*] Plotting Figure 2: Hardware Constraints & Laser RIN Operating Boundary...")
with open(DATA_DIR / "exp2_hardware_rin_sensitivity.json") as f:
    d2 = json.load(f)

# Group by config and plot Outage vs Laser RIN at nominal 0 dB loss and 1.5 dB loss
grid_data = d2["grid_data"]

rin_unique = sorted(list(set(g["laser_rin_db_hz"] for g in grid_data)))

plt.figure(figsize=(7.5, 5.2), dpi=300)

configs = [
    ("M8_Nr8", "Square Sorter ($M=8, N_r=8, d=0.5$)", "#D9534F", "o"),
    ("M8_Nr11", "Oversampled ($M=8, N_r=11, d=2.0$)", "#F0AD4E", "^"),
    ("M6_Nr11", "Mode Throttled ($M=6, N_r=11, d=3.0$)", "#5CB85C", "s")
]

for sname, label, color, marker in configs:
    # 0 dB excess loss
    outages_0db = [g["system_outage"] for g in grid_data if g["short_name"] == sname and g["excess_loss_db"] == 0.0]
    outages_1db = [g["system_outage"] for g in grid_data if g["short_name"] == sname and g["excess_loss_db"] == 1.5]
    
    plt.semilogy(rin_unique, outages_0db, f'{marker}-', color=color, label=f"{label} (Nominal loss)")
    plt.semilogy(rin_unique, outages_1db, f'{marker}--', color=color, alpha=0.6, label=f"{label} (+1.5 dB loss)")

plt.axhline(1e-4, color='black', linestyle=':', linewidth=1.8, label='URLLC Outage Target ($10^{-4}$)')
plt.axvline(-155, color='gray', linestyle='-.', alpha=0.7, label='Nominal RIN (-155 dB/Hz)')

plt.xlabel('Laser Relative Intensity Noise (RIN) [dB/Hz]')
plt.ylabel('System Outage Probability $P_{out}$')
plt.title('System Outage vs. Laser RIN and Mode Sorter Excess Loss')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()

fig_path2 = FIG_DIR / "fig_exp2_hardware_rin_envelope.png"
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"[+] Saved {fig_path2}")

# -------------------------------------------------------------
# Figure 3: Dynamic Precision Adaptation Pareto Frontier
# -------------------------------------------------------------
print("[*] Plotting Figure 3: Dynamic Precision Adaptation Pareto Frontier...")
with open(DATA_DIR / "exp3_dynamic_precision_pareto.json") as f:
    d3 = json.load(f)

deadlines = [r["deadline_ms"] for r in d3["data"]]
q_vals = [r["mean_ternary_fraction_Q"] for r in d3["data"]]
psnr_vals = [r["mean_psnr_db"] for r in d3["data"]]
energy_savings = [r["energy_savings_pct"] for r in d3["data"]]

fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

color = '#0275D8'
ax1.set_xlabel('End-to-End Latency Deadline $D_{max}$ (ms)')
ax1.set_ylabel('Mean Reconstructed PSNR (dB)', color=color)
line1 = ax1.plot(deadlines, psnr_vals, 'o-', color=color, linewidth=2.4, label='PSNR (dB)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = '#D9534F'
ax2.set_ylabel('Energy Savings vs. FP16 (%)', color=color)
line2 = ax2.plot(deadlines, energy_savings, 's--', color=color, linewidth=2.4, label='Energy Savings (%)')
ax2.tick_params(axis='y', labelcolor=color)

# Add vertical marker for extreme URLLC boundary
ax1.axvline(0.78, color='purple', linestyle=':', linewidth=1.8, label='Extreme URLLC (0.78 ms)')

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines] + ['Extreme URLLC Budget']
ax1.legend(lines + [plt.Line2D([0], [0], color='purple', linestyle=':')], labels, loc='center right')

plt.title('Dual-Regime Trade-off: Fidelity and Energy vs. Deadline Bound')
plt.tight_layout()

fig_path3 = FIG_DIR / "fig_exp3_dynamic_precision_pareto.png"
plt.savefig(fig_path3, dpi=300)
plt.close()
print(f"[+] Saved {fig_path3}")

print("[+] All publication figures generated successfully!")
