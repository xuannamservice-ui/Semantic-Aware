"""System diagram: sender (codec), OAM-FSO link, edge (cache, concealment, rendering) and scheduler."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.16, 2.75), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")


def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="0.25", lw=0.6))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=5.6, weight="bold")
    ax.text(x + w / 2, y + h - 4.4, "\n".join(lines), ha="center", va="top", fontsize=4.8, linespacing=1.25)


def arrow(x0, y0, x1, y1, text=None, dy=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6, lw=0.7, color="0.2", ls=ls))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=4.8, color="0.15")


# section labels
for x, w, t in ((0.5, 31, "Cloud sender"), (34.5, 31, "OAM-FSO backhaul (1.5 km, turbulence)"), (68.5, 31, "Edge renderer")):
    ax.add_patch(FancyBboxPatch((x, 9.2), w, 30, boxstyle="round,pad=0.2,rounding_size=1.2", fc="none", ec="0.6", lw=0.5, ls="--"))
    ax.text(x + w / 2, 38.4, t, ha="center", va="top", fontsize=6.5, style="italic", color="0.2")

box(1.5, 25.2, 14, 10.5, "Layered codec", ["anchor 3DGS, entropy model", "base: position + 18 values", "enhancement: 66 values"], "#e8f0fb")
box(17, 25.2, 13.6, 10.5, "Packetization", ["tokens of 64 anchors", "(Morton order)", "token → stream mapping"], "#e8f0fb")
box(1.5, 11.2, 29.1, 10.8, "Semantic protection", ["base layer + positions replicated on the", "$n_\\mathcal{A}$ most reliable modes (anchor channel)", "enhancement tokens on the $M$ data streams"], "#dbe9f8")
box(35.5, 25.2, 13.8, 10.5, "OAM multiplexing", ["$M$ LG modes", "$\\ell \\in \\{\\pm1, \\dots, \\pm4\\}$", "IM/DD, OOK"], "#fdf1e3")
box(50.7, 25.2, 13.8, 10.5, "Sorter + MMSE", ["$N_r$ ports", "linear equalization", "HD-FEC threshold"], "#fdf1e3")
box(35.5, 11.2, 29, 10.8, "Per-slot failure pattern $\\mathcal{F}(t)$", ["random, correlated subset of failed streams", "outage $P_{out}$, decay order $\\frac{1}{2}\\min\\{\\hat{k},\\alpha,\\beta\\}$", "(Proposition 1, $10^7$-slot traces)"], "#fbe3c8")
box(69.5, 25.2, 14, 10.5, "Edge cache", ["received enhancement", "layers kept", "new anchors only"], "#e5f4e5")
box(85, 25.2, 14, 10.5, "Concealment", ["cross-attention to", "16 neighbours, joint", "training, FP / 1.58-bit"], "#d3edd3")
box(69.5, 11.2, 29.5, 10.8, "Decode + render", ["anchors → neural Gaussians (MLPs) → gsplat rasterizer", "frame period 11.1 ms at 90 frames/s", "quality $\\Psi$ against the loss-free rendering"], "#e5f4e5")

arrow(15.5, 30.5, 17, 30.5)
arrow(23.8, 25.2, 23.8, 22.0)
arrow(30.6, 16.6, 35.5, 30.5, "streams", dy=0.6)
arrow(49.3, 30.5, 50.7, 30.5)
arrow(57.6, 25.2, 57.6, 22.0)
arrow(64.5, 30.5, 69.5, 30.5, "received tokens", dy=0.6)
arrow(64.5, 16.6, 85, 25.2, None)
ax.text(73.0, 20.9, "lost tokens", fontsize=4.8, color="0.15", ha="center", rotation=22)
arrow(83.5, 30.5, 85, 30.5)
arrow(92, 25.2, 92, 22.0)
arrow(80.0, 25.2, 80.0, 22.0)

# scheduler
ax.add_patch(FancyBboxPatch((20, 0.6), 60, 6.4, boxstyle="round,pad=0.25,rounding_size=0.8", fc="#f2e8f7", ec="0.25", lw=0.6))
ax.text(50, 6.2, "Cross-layer scheduler (Algorithm 1)", ha="center", va="top", fontsize=6.2, weight="bold")
ax.text(50, 3.6, "configuration $(M, N_r)$, replication $n_\\mathcal{A}$, power allocation, precision $q$; virtual queue of failed slots, "
        "target $\\bar{a} \\leq 10^{-4}$", ha="center", va="top", fontsize=5.1)
arrow(30, 7.2, 16, 11.0, style="-|>", ls="--")
arrow(50, 7.2, 50, 11.0, style="-|>", ls="--")
arrow(70, 7.2, 84, 11.0, style="-|>", ls="--")
plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
plt.savefig("fig_system.png", dpi=300)
print("ok")
