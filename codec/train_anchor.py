"""Train the anchor-based (streamable) representation of one scene, initialised from the reference 3DGS model.

Anchors are the occupied voxels of the trained 3DGS centres (voxel size chosen for a target anchor count).
Training: L1 + 0.2 D-SSIM + 0.01 scale-volume regulariser (Scaffold-GS settings), quantization-aware from
step q_start on (fixed quantization steps, straight-through estimator). After training the script reports
test-view quality of the quantized model, the empirical entropy of the transmitted symbols (bits/anchor),
and a per-anchor importance score (opacity-weighted screen coverage accumulated over the training views).
Output: <out>/<scene>/anchor.pt, metrics_anchor.json
usage: python train_anchor.py --scene bicycle --anchors 80000 --steps 30000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import metrics
from anchor_model import AnchorModel, voxel_anchors
from scene_data import Scene, scene_location
from train_3dgs import OUTDOOR

QSTEP = {"feat": 0.05, "offsets": 0.02, "scaling": 0.02}


def pick_voxel(points, opacity, target):
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        n = voxel_anchors(points, opacity, mid).shape[0]
        if n > target:
            lo = mid
        else:
            hi = mid
    return hi


def entropy_bits(sym):
    """Empirical entropy (bits) of integer symbols, per column, summed over columns; sym: (A, C) int64."""
    tot = 0.0
    for c in range(sym.shape[1]):
        _, cnt = torch.unique(sym[:, c], return_counts=True)
        p = cnt.double() / cnt.sum()
        tot += float(-(p * torch.log2(p)).sum())
    return tot


def rate_report(model):
    att = {n: getattr(model, n).detach() for n in model.ATTRS}
    rep, A = {}, model.xyz.shape[0]
    for n, v in att.items():
        sym = torch.round(v / QSTEP[n]).long().reshape(A, -1)
        rep[n] = entropy_bits(sym)                                    # bits per anchor (memoryless, per channel)
    keys = torch.round(model.xyz / model.voxel - 0.5).long()
    keys = keys - keys.min(0).values
    rep["position_fixed"] = float(sum(math.ceil(math.log2(int(k) + 1)) for k in keys.max(0).values))
    rep["attributes_total"] = rep["feat"] + rep["offsets"] + rep["scaling"]
    return rep


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    out = os.path.join(a.out, a.scene)
    gs = torch.load(os.path.join(out, "gs.pt"))
    pts, opa = gs["means"].cuda(), torch.sigmoid(gs["opacities"]).cuda()
    voxel = pick_voxel(pts, opa, a.anchors)
    xyz = voxel_anchors(pts, opa, voxel)
    model = AnchorModel(xyz, voxel, feat_dim=a.feat_dim, k=a.k).cuda()
    for n in model.log_qstep:
        model.log_qstep[n].data.fill_(math.log(QSTEP[n])); model.log_qstep[n].requires_grad_(False)
    groups = [{"params": [model.feat], "lr": 0.0075, "f": 0.0075}, {"params": [model.offsets], "lr": 0.01, "f": 1e-4},
              {"params": [model.scaling], "lr": 0.007, "f": 0.007},
              {"params": model.mlp_opacity.parameters(), "lr": 0.002, "f": 2e-5},
              {"params": model.mlp_cov.parameters(), "lr": 0.004, "f": 0.004},
              {"params": model.mlp_color.parameters(), "lr": 0.008, "f": 5e-5}]
    opt = torch.optim.Adam(groups, eps=1e-15)
    init = [g["lr"] for g in opt.param_groups]
    t0, log = time.time(), []
    order = np.random.permutation(scene.train_idx)
    for step in range(a.steps):
        r = step / a.steps
        for g, lr0 in zip(opt.param_groups, init):
            g["lr"] = math.exp((1 - r) * math.log(lr0) + r * math.log(g["f"]))
        i = order[step % len(order)]
        if step % len(order) == len(order) - 1:
            order = np.random.permutation(scene.train_idx)
        gt, K, V = scene.view(i)
        idx = model.visible(K, V, scene.W, scene.H)
        attrs = model.attributes(quantized=step >= a.q_start)
        img, info, scales = model.render(attrs, idx, K, V, scene.W, scene.H)
        l1 = F.l1_loss(img, gt)
        ss = metrics.ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()
        loss = 0.8 * l1 + 0.2 * (1 - ss) + 0.01 * scales.prod(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "visible": int(idx.numel()), "t": time.time() - t0}
            log.append(rec); print(rec, flush=True)
    model.eval()
    with torch.no_grad():
        attrs = model.attributes(quantized=True)
        test = []
        for i in scene.test_idx:
            gt, K, V = scene.view(i)
            img, _, _ = model.render(attrs, model.visible(K, V, scene.W, scene.H), K, V, scene.W, scene.H)
            test.append(metrics.all_metrics(img, gt))
        imp = torch.zeros(model.xyz.shape[0], device="cuda")
        for i in scene.train_idx:
            _, K, V = scene.view(i)
            idx = model.visible(K, V, scene.W, scene.H)
            cam = -V[:3, :3].T @ V[:3, 3]
            x = model.xyz[idx]; d = x - cam[None]; dist = d.norm(dim=1, keepdim=True)
            inp = torch.cat([attrs["feat"][idx], d / dist, dist], 1)
            o = model.mlp_opacity(inp).clamp_min(0)                       # (n, k)
            sz = torch.exp(attrs["scaling"][idx][:, 3:]).mean(1)          # world size of the anchor's Gaussians
            pix = (K[0, 0] * sz / dist.squeeze(1)).clamp(max=64.0)       # projected radius in pixels
            imp[idx] += o.sum(1) * pix ** 2
    rate = rate_report(model)
    res = {"scene": a.scene, "anchors": int(model.xyz.shape[0]), "voxel": voxel, "feat_dim": a.feat_dim, "k": a.k,
           "qstep": QSTEP, "steps": a.steps, "seed": a.seed, "train_time_s": time.time() - t0,
           "test": {k: float(np.mean([m[k] for m in test])) for k in test[0]},
           "test_per_view": test, "rate_bits_per_anchor": rate,
           "scene_bytes_attributes": rate["attributes_total"] * model.xyz.shape[0] / 8,
           "scene_bytes_positions_fixed": rate["position_fixed"] * model.xyz.shape[0] / 8, "log": log}
    torch.save({"state": model.state_dict(), "voxel": voxel, "feat_dim": a.feat_dim, "k": a.k, "importance": imp.cpu()},
               os.path.join(out, "anchor.pt"))
    with open(os.path.join(out, "metrics_anchor.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("anchors", res["anchors"], "test", res["test"], "bits/anchor", rate, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--anchors", type=int, default=80_000)
    ap.add_argument("--feat-dim", type=int, default=32)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--q-start", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
