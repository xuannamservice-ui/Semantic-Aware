"""Number of anchors that become visible for the first time per frame along the capture path (no losses)."""
import json, sys
import numpy as np, torch
from eval_layered import load
from scene_data import Scene, scene_location
out = {}
for s in sys.argv[1:]:
    scene = Scene(*scene_location(s, "data/360_v2"))
    model, L, nets, c, ck = load("results", s, "_v2a_r1")
    seen = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda"); new, vis = [], []
    with torch.no_grad():
        for vi in np.argsort(scene.names):
            gt, K, V = scene.view(int(vi)); idx = model.visible(K, V, scene.W, scene.H)
            new.append(int((~seen[idx]).sum())); vis.append(int(idx.numel())); seen[idx] = True
    out[s] = {"frames": len(new), "first": new[0], "new_mean_after_first": float(np.mean(new[1:])), "new_p95": float(np.percentile(new[1:], 95)),
              "new_max": int(max(new[1:])), "visible_mean": float(np.mean(vis))}
    print(s, out[s], flush=True)
json.dump(out, open("results/new_anchors.json", "w"), indent=1)
