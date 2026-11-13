"""Publication figures drawn only from simulation and measurement output.

Replaces the curves of create_publication_figures.py, which are closed-form formulas
anchored at table values and must not be published as results.

Inputs (one row per point; templates with the header only are in figure_data/):

  figure_data/mc_outage_vs_snr.csv
      config,M,N_r,sigma_R2,gamma0_dB,n_realizations,n_outage,seed
      config is a short label such as "square" or "Nr11_M6".
  figure_data/gpu_latency_vs_Q.csv
      Q,n_runs,T_inf_mean_ms,T_inf_p99_ms,energy_J_per_slot
      only the ternary fractions that were actually profiled.
  figure_data/zernike_truncation.csv
      J,M,N_r,sigma_R2,n_realizations,n_outage,seed

Outputs: fig2_outage_prob.png, fig4_latency_energy_vs_q.png, fig9_ablation_zernike.png
(the file names used in access.tex). A figure whose input file is missing or empty is
skipped, so no figure is ever produced without data.

Usage:  python plot_from_simulation_data.py [--data figure_data] [--out .]
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Categorical order of the validated reference palette (fixed order, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

GAMMA0_OP_DB = 60.1      # operating point, Table "sim_params"
TARGET = 1e-4            # delta_target
D_MAX, T_FIX = 0.78, 0.116
Z95 = 1.959964

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if any(v.strip() for v in r.values())]


def wilson(k, n):
    """95% Wilson interval of a binomial proportion."""
    p = k / n
    den = 1 + Z95**2 / n
    c = (p + Z95**2 / (2 * n)) / den
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / den
    return max(c - h, 0.0), c + h


def style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def local_slope(g_db, p, k, min_events=50, window_db=5.0):
    """Slope -d log10 P / d log10 gamma0 over the last window with >= min_events per point."""
    ok = k >= min_events
    if ok.sum() < 2:
        return None
    g, pp = g_db[ok], p[ok]
    sel = g >= g.max() - window_db
    if sel.sum() < 2:
        return None
    fit = np.polyfit(g[sel] / 10.0, np.log10(pp[sel]), 1)
    return -fit[0]


def fig_outage(rows, out):
    groups = {}
    for r in rows:
        key = (r["config"], int(r["M"]), int(r["N_r"]), float(r["sigma_R2"]))
        groups.setdefault(key, []).append(r)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    for i, (key, rs) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0][2] - kv[0][1], -kv[0][3]))):
        if i >= len(SERIES):
            raise SystemExit("more than six series: split the figure instead of cycling colors")
        cfg, M, Nr, s2 = key
        rs.sort(key=lambda r: float(r["gamma0_dB"]))
        g = np.array([float(r["gamma0_dB"]) for r in rs])
        n = np.array([float(r["n_realizations"]) for r in rs])
        k = np.array([float(r["n_outage"]) for r in rs])
        keep = k > 0                    # zero-event points are not plotted as values
        p = k[keep] / n[keep]
        lo_hi = np.array([wilson(kk, nn) for kk, nn in zip(k[keep], n[keep])])
        d = 0.5 * (Nr - M + 1)          # geometric term k/2; the caption states the alpha/beta cap
        slope = local_slope(g[keep], p, k[keep])
        label = f"$M={M}$, $N_r={Nr}$, $\\sigma_R^2={s2:g}$ ($k/2={d:g}$"
        label += f", fitted slope {slope:.2f})" if slope is not None else ")"
        ls = "-" if Nr == M else "--"   # secondary encoding: square vs oversampled sorter
        ax.semilogy(g[keep], p, ls, color=SERIES[i], marker="o", label=label)
        ax.fill_between(g[keep], lo_hi[:, 0], lo_hi[:, 1], color=SERIES[i], alpha=0.15, linewidth=0)
    ax.axvline(GAMMA0_OP_DB, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(GAMMA0_OP_DB + 0.4, 0.5, "operating point\n$\\gamma_0 = 60.1$ dB", color=MUTED, fontsize=7, va="top")
    ax.axhline(TARGET, color=INK, linestyle="--", linewidth=1.0)
    ax.text(64.5, TARGET * 1.3, "$\\delta_{target} = 10^{-4}$", color=INK, fontsize=7, ha="right")
    ax.set_xlim(15, 65)
    ax.set_xlabel("Reference SNR per mode $\\gamma_0$ (dB)")
    ax.set_ylabel("System outage probability $P_{out}$")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_outage_prob.png", bbox_inches="tight")
    plt.close(fig)


def fig_latency(rows, out):
    rows.sort(key=lambda r: float(r["Q"]))
    q = np.array([float(r["Q"]) for r in rows])
    t = np.array([float(r["T_inf_mean_ms"]) for r in rows])
    e = np.array([float(r["energy_J_per_slot"]) for r in rows])
    # Two panels, one y-scale each (no dual axis).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)
    a1.plot(q, t, "o", color=SERIES[0], label="measured mean")
    if {0.0, 1.0} <= set(q.tolist()):
        t0, t1 = t[q == 0.0][0], t[q == 1.0][0]
        qq = np.linspace(0, 1, 25)
        a1.plot(qq, t1 + (t0 - t1) * (1 - qq), "-", color=SERIES[0], linewidth=1.2, alpha=0.6,
                label="linear model (scheduler)")
    # Budget of the fully ternary configuration (M = 8, tau_tx = 0.019 ms); the budget of
    # Q < 1 is smaller because the payload grows (Section VI-F).
    a1.axhline(D_MAX - T_FIX - 0.019, color=INK, linestyle="--", linewidth=1.0)
    a1.text(0.02, D_MAX - T_FIX - 0.019 + 0.02, "budget at $Q=1$: 0.645 ms", fontsize=7, color=INK)
    a1.set_xlabel("Ternary fraction $Q$")
    a1.set_ylabel("Inference latency $T_{inf}$ (ms)")
    a1.legend(frameon=False, loc="upper right")
    a2.plot(q, e, "o", color=SERIES[0])
    a2.set_xlabel("Ternary fraction $Q$")
    a2.set_ylabel("Energy per slot (J)")
    for a in (a1, a2):
        style(a)
        a.set_xlim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(out / "fig4_latency_energy_vs_q.png")
    plt.close(fig)


def fig_zernike(rows, out):
    rows.sort(key=lambda r: int(r["J"]))
    cfgs = {(int(r["M"]), int(r["N_r"]), float(r["sigma_R2"])) for r in rows}
    if len(cfgs) != 1:
        raise SystemExit(f"zernike_truncation.csv mixes configurations {sorted(cfgs)}; plot one per figure")
    M, Nr, s2 = cfgs.pop()
    J = np.array([int(r["J"]) for r in rows])
    n = np.array([float(r["n_realizations"]) for r in rows])
    k = np.array([float(r["n_outage"]) for r in rows])
    p = k / n
    ci = np.array([wilson(kk, nn) for kk, nn in zip(k, n)])
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    ax.errorbar(J, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="o-", color=SERIES[0], capsize=2, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Zernike truncation order $J$")
    ax.set_ylabel("System outage $P_{out}$")
    ax.set_title(f"$M={M}$, $N_r={Nr}$, $\\sigma_R^2={s2:g}$", fontsize=8, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig9_ablation_zernike.png")
    plt.close(fig)
    # table requested by the reviewers: relative error with respect to the largest J
    ref = p[-1]
    print("J, P_out, 95% CI, deviation from J_max")
    for jj, pp, (lo, hi) in zip(J, p, ci):
        print(f"{jj:4d}  {pp:.4e}  [{lo:.4e}, {hi:.4e}]  {100 * (pp / ref - 1):+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="figure_data")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    jobs = [("mc_outage_vs_snr.csv", fig_outage), ("gpu_latency_vs_Q.csv", fig_latency),
            ("zernike_truncation.csv", fig_zernike)]
    for name, fn in jobs:
        rows = read_rows(data / name)
        if not rows:
            print(f"skip {name}: no data rows")
            continue
        fn(rows, out)
        print(f"done {name}")


if __name__ == "__main__":
    main()
