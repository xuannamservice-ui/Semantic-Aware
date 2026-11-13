"""Trace-driven evaluation of concealment on the held-out test views of one scene.

Loss patterns are the per-slot failure patterns of the OAM link simulator (phy/traces_<s2>.npz, 1e6 slots per
configuration, bit i = i-th transmitted mode failed). For a configuration the transmitted modes are ranked by
their marginal outage (phy/modeset_<s2>.json) and stream m of the packetiser is carried on the m-th most reliable
mode. For every test view, n_pat outage patterns are drawn with their empirical frequencies; for every pattern,
mapping policy and concealment method the view is rendered and compared with the ground-truth image and with the
loss-free rendering. Methods: none (lost anchors dropped), nn (attributes of the nearest received neighbour),
idw (inverse-distance mean of the received neighbours), gen_fp and gen_ternary (concealment network).
Output: <out>/<scene>/eval_<cfg>_<s2>.json
usage: python eval_stream.py --scene bicycle --cfg T6R11 --s2 2.5 --n-pat 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import metrics
from conceal_net import ConcealNet, knn
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_3dgs import OUTDOOR
from train_conceal import POLICIES, Concealer, flat_attrs, load_anchor, unflat

METHODS = ("none", "nn", "idw", "gen_fp", "gen_ternary")


def baseline_fill(method, lost, avail, X, xyz, nbr):
    nb = nbr[lost]
    r = avail[nb]
    has = r.any(1)
    if method == "nn":
        first = torch.argmax(r.int(), 1)
        out = X[nb[torch.arange(nb.shape[0], device=nb.device), first]]
    else:
        d = (xyz[nb] - xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
        w = r.float() / d
        out = (w[..., None] * X[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
    return out, has


def main(a):
    rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, imp = load_anchor(a.out, a.scene)
    X = flat_attrs(model)
    ck = torch.load(os.path.join(a.out, a.scene, "conceal.pt"))
    cfg = ck["cfg"]
    nbr = knn(model.xyz, cfg["kn"])
    mean, std = ck["mean"].cuda(), ck["std"].cuda()
    nets = {}
    for name, key, tern in (("gen_fp", "fp", False), ("gen_ternary", "ternary", True)):
        if ck.get(key) is not None:
            n = ConcealNet(cfg["attr_dim"], d=cfg["d"], layers=cfg["layers"], ternary=tern).cuda()
            n.load_state_dict(ck[key]); n.eval()
            nets[name] = Concealer(model, n, mean, std, nbr, ck["radius"])
    pk = Packetizer(model.xyz, model.voxel, imp, G=cfg["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    c = ms["configs"][a.cfg]
    tx = c["tx"]
    per_mode = c["per_mode_count"][c["snr_labels"].index("op")]
    rel = mode_reliability_order(per_mode, tx)
    tr = np.load(os.path.join(a.phy, f"traces_{a.s2}.npz"))[a.cfg]
    pats, cnt = np.unique(tr[tr > 0], return_counts=True)
    p_out = float((tr > 0).mean())
    M = len(tx)
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for vi in scene.test_idx:
            gt, K, V = scene.view(vi)
            idx = model.visible(K, V, scene.W, scene.H)
            full, _, _ = model.render(unflat(model, X), idx, K, V, scene.W, scene.H)
            m_full = metrics.all_metrics(full, gt)
            draw = rng.choice(len(pats), a.n_pat, p=cnt / cnt.sum())
            for pi in draw:
                failed = patterns_to_streams(pats[pi], tx, rel)
                for pol in POLICIES:
                    lost_m = pk.lost_mask(idx, M, pol, failed, np.random.default_rng(int(vi) * 1000 + int(pi)))
                    lost = idx[lost_m]
                    avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                    avail[idx[~lost_m]] = True
                    for meth in METHODS:
                        if meth.startswith("gen") and meth not in nets:
                            continue
                        Xh = X.clone()
                        keep = idx[~lost_m]
                        if meth == "none":
                            ridx = keep
                        else:
                            if meth in ("nn", "idw"):
                                fill, has = baseline_fill(meth, lost, avail, X, model.xyz, nbr)
                            else:
                                fill = nets[meth].predict(lost, avail, X)
                                has = torch.ones(lost.numel(), dtype=torch.bool, device="cuda")
                            Xh[lost] = fill
                            ridx = torch.cat([keep, lost[has]])
                        img, _, _ = model.render(unflat(model, Xh), ridx, K, V, scene.W, scene.H)
                        m = metrics.all_metrics(img, gt)
                        rows.append({"view": int(vi), "pattern": int(pats[pi]), "K": len(failed),
                                     "failed_streams": [int(x) for x in failed], "policy": pol, "method": meth,
                                     "lost_frac": float(lost_m.float().mean()), **m,
                                     "psnr_vs_full": metrics.psnr(img.clamp(0, 1), full.clamp(0, 1)),
                                     "full_psnr": m_full["psnr"], "full_ssim": m_full["ssim"],
                                     "full_lpips": m_full["lpips"]})
            print(f"view {vi}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for pol in POLICIES:
        for meth in METHODS:
            r = [x for x in rows if x["policy"] == pol and x["method"] == meth]
            if r:
                summ[f"{pol}/{meth}"] = {k: float(np.mean([x[k] for x in r])) for k in
                                         ("psnr", "ssim", "lpips", "psnr_vs_full", "lost_frac")}
    out = {"scene": a.scene, "cfg": a.cfg, "s2": a.s2, "tx": tx, "reliability_order": rel, "p_out_trace": p_out,
           "n_outage_slots": int(cnt.sum()), "n_pat_per_view": a.n_pat, "seed": a.seed, "summary": summ, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"eval_{a.cfg}_{a.s2}.json"), "w") as f:
        json.dump(out, f)
    for k, v in summ.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cfg", default="T6R11")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--n-pat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
