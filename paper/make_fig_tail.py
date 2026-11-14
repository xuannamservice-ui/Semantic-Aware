"""Tail-quality figure.
(a) Cold-start outage slots of the T8R8 trace (interleaved mapping), scenes with a HAC evaluation: share of outage slots
    with Psi below the threshold, for the layered codec (drop, IDW, concealment network) and HAC (drop, IDW), each also
    with the (8, 7) MDS code of the frame-level evaluation (r = round(eps M) = 1 for eps = 1/6), which
    recovers the slots with one failed stream.
(b) Frame-level evaluation, T8R8, tau_c = 10 ms, all scenes: share of test frames with Psi below the threshold.
usage: python make_fig_tail.py <results root> <out.png> [<facts.json>]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT, OUT = sys.argv[1:3]
FACTS = sys.argv[3] if len(sys.argv) > 3 else None
R = os.path.join(ROOT, "results")
TH = np.linspace(20, 45, 101)
POL = "interleaved"

hac = {}
for f in glob.glob(os.path.join(ROOT, "hac/trace_eval/*_0.004_T8R8.json")):
    s = os.path.basename(f).split("_")[0]
    hac[s] = [r for r in json.load(open(f))["rows"] if r["policy"] == POL]
scenes = sorted(hac)
ours = {}
for s in scenes:
    ours[s] = [r for r in json.load(open(f"{R}/{s}/eval_layered_v2a_r1_T8R8_2.5.json"))["rows"] if r["policy"] == POL]


def curve(rows, method, mds=False):
    x = np.array([r["psnr_vs_full"] for r in rows if r["method"] == method])
    k = np.array([r["K"] for r in rows if r["method"] == method])
    if mds:
        x = np.where(k <= 1, 99.0, x)          # slots with one failed stream are recovered by the code
    return np.array([(x < t).mean() for t in TH]), x


fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6), dpi=300)
series = []
for rows, lab, meth, mds, st in (
        ([r for s in scenes for r in hac[s]], "HAC, drop", "drop", False, dict(color="0.55", ls="--")),
        ([r for s in scenes for r in hac[s]], "HAC, IDW", "idw", False, dict(color="0.35", ls="--")),
        ([r for s in scenes for r in hac[s]], "HAC, IDW + MDS", "idw", True, dict(color="0.1", ls="--")),
        ([r for s in scenes for r in ours[s]], "Layered, drop", "drop", False, dict(color="tab:orange")),
        ([r for s in scenes for r in ours[s]], "Layered, IDW", "idw", False, dict(color="tab:green")),
        ([r for s in scenes for r in ours[s]], "Layered, concealment", "gen_fp", False, dict(color="tab:blue", lw=1.6)),
        ([r for s in scenes for r in ours[s]], "Layered, concealment + MDS", "gen_fp", True, dict(color="tab:purple", lw=1.2))):
    y, x = curve(rows, meth, mds)
    ax[0].plot(TH, y, label=lab, lw=st.pop("lw", 1.0), **st)
    series.append((lab, float((x < 30).mean()), float((x < 35).mean()), len(x)))
ax[0].set_xlabel(r"Threshold $\Psi_{th}$ (dB)", fontsize=7)
ax[0].set_ylabel(r"$\Pr(\Psi < \Psi_{th} \mid$ outage slot$)$", fontsize=7)
ax[0].set_title(f"(a) Cold-start outage slots, T8R8 ({', '.join(scenes)})", fontsize=7)
ax[0].legend(fontsize=5.5, loc="lower right", frameon=False)
ax[0].tick_params(labelsize=6); ax[0].set_ylim(-0.02, 1.02); ax[0].grid(alpha=0.3, lw=0.4)

D = {}
for f in glob.glob(f"{R}/*/frames_v2a_r1_T8R8_tau10_eps0.166667.json"):
    fr = json.load(open(f))
    for r in fr["rows"]:
        if r["chain"] < 10:
            D.setdefault(r["scheme"], []).append(min(r["psi"], 99.0))
names = {"none": "None", "conceal": "Conceal", "arq": "ARQ", "arq+conceal": "ARQ + conceal", "mds": "MDS", "mds+conceal": "MDS + conceal"}
sty = {"none": dict(color="tab:orange"), "conceal": dict(color="tab:blue", lw=1.6), "arq": dict(color="0.4", ls="--"),
       "arq+conceal": dict(color="tab:cyan", ls="-."), "mds": dict(color="0.1", ls=":"), "mds+conceal": dict(color="tab:purple", ls="-.")}
n = None
for k in ("none", "arq", "mds", "conceal", "arq+conceal", "mds+conceal"):
    x = np.array(D[k]); n = len(x)
    y = np.array([(x < t).sum() for t in TH]) / n
    st = dict(sty[k]); ax[1].plot(TH, np.maximum(y, 1e-5), label=names[k], lw=st.pop("lw", 1.0), **st)
ax[1].set_yscale("log"); ax[1].set_ylim(1e-4 * 2, 2e-2)
ax[1].set_xlabel(r"Threshold $\Psi_{th}$ (dB)", fontsize=7)
ax[1].set_ylabel("Share of failed test frames", fontsize=7)
ax[1].set_title(f"(b) Frame level with edge cache, T8R8, {n} test frames", fontsize=7)
ax[1].legend(fontsize=5.5, loc="lower right", frameon=False, ncol=2); ax[1].tick_params(labelsize=6); ax[1].grid(alpha=0.3, lw=0.4, which="both")
plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=300)
res = {"scenes_hac": scenes, "cold_start": [dict(zip(("label", "p30", "p35", "n"), s)) for s in series]}
if FACTS:
    json.dump(res, open(FACTS, "w"), indent=1)
print(json.dumps(res, indent=1))
