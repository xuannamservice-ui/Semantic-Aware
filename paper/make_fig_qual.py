"""Qualitative figure: drop / IDW / concealment network / loss-free renders of one test view per scene (qual_render.py)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
rows = [("garden", "garden_v2_T8R8_K3"), ("kitchen", "kitchen_v5_T8R8_K3"), ("truck", "truck_v2_T8R8_K3"), ("playroom", "playroom_v4_T8R8_K3")]
cols = [("drop", "Drop"), ("idw", "IDW"), ("gen", "Concealment network"), ("full", "Loss-free")]
fig, ax = plt.subplots(len(rows), len(cols), figsize=(7.16, 4.6), dpi=300)
for i, (sc, stem) in enumerate(rows):
    d = json.load(open(stem + ".json"))
    for j, (k, t) in enumerate(cols):
        a = ax[i][j]; a.imshow(Image.open(f"{stem}_{k}.png"), aspect="auto"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.3)
        lab = f"{d[k]['psnr']:.2f} dB" + (f", $\\Psi$ = {d[k]['psi']:.1f} dB" if d[k]["psi"] is not None else "")
        a.set_xlabel(lab, fontsize=6, labelpad=1.5)
        if i == 0:
            a.set_title(t, fontsize=7, pad=2)
        if j == 0:
            a.set_ylabel(sc, fontsize=7, labelpad=2)
plt.subplots_adjust(left=0.03, right=0.995, top=0.955, bottom=0.03, wspace=0.03, hspace=0.2)
plt.savefig("fig_qualitative.png", dpi=300)
print("ok")
