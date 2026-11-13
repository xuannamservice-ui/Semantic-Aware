"""Fig. outage vs reference SNR (sigma_R^2 = 2.5, 1e7 slots) from modeset_summary.json; points with >= 10 events."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
s = json.load(open(sys.argv[1]))["2.5"]["configs"]
grid = list(range(30, 101, 5))
style = {"T8R8": ("k", "-", "o"), "T8R9": ("k", "--", "s"), "T8R11": ("k", ":", "^"),
         "T6R8": ("0.45", "-", "D"), "T6R11": ("0.45", "--", "o"), "T4R8": ("0.45", ":", "x")}
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
for k, (c, ls, mk) in style.items():
    o, n = s[k]["outage_grid"], s[k]["count_grid"]
    pts = [(g, p) for g, p, m in zip(grid, o, n) if m >= 10]
    ax.semilogy([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3, lw=1, label=k,
                mfc="none" if mk in "osD^" else c)
ax.set_xlabel(r"Reference SNR $\gamma_0$ (dB)"); ax.set_ylabel("Outage probability")
ax.set_xlim(30, 100); ax.set_ylim(1e-6, 1); ax.grid(True, which="both", lw=0.3, alpha=0.5)
ax.legend(fontsize=6.5, ncol=2, loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[2]); print("saved", sys.argv[2])
