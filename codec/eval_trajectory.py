"""Trajectory evaluation with an edge cache and temporally correlated losses.

The camera follows the capture order of all views of the scene (one slot per view). The edge keeps a cache of
received enhancement layers (unbounded, or with LRU eviction to a capacity of cache_frac x anchors). In slot t the anchors visible in view t that are not cached are packetised and sent
on the M streams (importance mapping); the anchors of failed streams stay uncached, are concealed for this slot
and are sent again in the next slot in which they are visible (no explicit retransmission request). The base layer
and positions are always available (replicated anchor channel).
Losses: the slot-outage indicator of a window of the temporally correlated simulation (chains joined end to end) (run_traces_corr.py,
Kolmogorov temporal correlation, tau_c) gives the outage slots; in an outage slot the failure pattern is drawn
from the empirical pattern distribution of the 1e6-slot independent trace of the same configuration, conditioned
on an outage. Quality is measured at the held-out test views of the path (PSNR/SSIM/LPIPS against ground truth and
PSNR against the loss-free rendering). Several chains (start offsets) are averaged.
Output: <out>/<scene>/traj<tag>_<cfg>_tau<tau>.json
usage: python eval_trajectory.py --scene garden --tag _a150 --cfg T8R8 --tau 10 --chains 20
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import metrics
from conceal_net import knn
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_3dgs import OUTDOOR
from train_layered import LayeredConcealer


def outage_chains(phy, cfg, tau, n_slots):
    """Outage indicators (chains x n_slots) from the packed correlated traces."""
    rows = []
    for f in sorted(glob.glob(os.path.join(phy, "traces_corr", f"traces_corr_2.5_{tau:g}ms_s*.npz"))):
        z = np.load(f)
        bits = np.unpackbits(z[cfg], axis=1)[:, :int(z["n_slots"])].astype(bool)
        rows.append(bits)
    x = np.concatenate(rows)
    return None, x.reshape(-1)                 # chains joined end to end (one boundary every chain length)


def main(a):
    rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = {k: LayeredConcealer(model, L, n, stats, nbr, ck["radius"]) for k, n in nets.items() if k in a.methods}
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx = cf["tx"]; M = len(tx)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
    tr = np.load(os.path.join(a.phy, f"traces_{a.s2}.npz"))[a.cfg]
    pats, cnt = np.unique(tr[tr > 0], return_counts=True)
    order = np.argsort(scene.names)
    T = len(order)
    _, allc = outage_chains(a.phy, a.cfg, a.tau, T)
    test_set = set(int(i) for i in scene.test_idx)
    methods = ["drop", "idw"] + [m for m in a.methods if m in conc]
    rows, t0 = [], time.time()
    with torch.no_grad():
        full = {}
        for vi in scene.test_idx:
            gt, K, V = scene.view(vi)
            img, _, _ = model.render(at, model.visible(K, V, scene.W, scene.H), K, V, scene.W, scene.H)
            full[int(vi)] = img
        for ch in range(a.chains):
            off = rng.integers(0, allc.size - T + 1)
            out_ind = allc[off:off + T]
            for meth in methods:
                cached = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                last_seen = torch.full((model.xyz.shape[0],), -1, dtype=torch.long, device="cuda")
                cap = int(a.cache_frac * model.xyz.shape[0]) if a.cache_frac < 1 else None
                prng = np.random.default_rng(a.seed * 1000 + ch)
                n_out = 0
                for t, vi in enumerate(order):
                    vi = int(vi)
                    gt, K, V = scene.view(vi)
                    idx = model.visible(K, V, scene.W, scene.H)
                    new = idx[~cached[idx]]
                    lost_new = torch.zeros(0, dtype=torch.long, device="cuda")
                    if out_ind[t] and new.numel():
                        n_out += 1
                        failed = patterns_to_streams(prng.choice(pats, p=cnt / cnt.sum()), tx, rel)
                        lm = pk.lost_mask(new, M, "importance", failed, prng)
                        lost_new = new[lm]
                        cached[new[~lm]] = True
                    else:
                        cached[new] = True
                    last_seen[idx] = t
                    if cap is not None and int(cached.sum()) > cap:      # evict least recently visible anchors
                        cand = torch.nonzero(cached & (last_seen < t)).squeeze(1)
                        n_ev = min(int(cached.sum()) - cap, cand.numel())
                        if n_ev > 0:
                            cached[cand[torch.argsort(last_seen[cand])[:n_ev]]] = False
                    if vi not in test_set:
                        continue
                    avail = cached.clone()
                    e = enh.clone(); ridx = idx
                    lost = idx[~cached[idx]]
                    if lost.numel():
                        if meth == "drop":
                            ridx = idx[cached[idx]]
                        elif meth == "idw":
                            nb = nbr[lost]; r = avail[nb].float()
                            w = r / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
                            mm = (w[..., None] * enh[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
                            e[lost] = torch.where((r.sum(1) > 0)[:, None], mm, stats[2].expand_as(mm))
                        else:
                            e[lost] = conc[meth].predict(lost, avail, base, enh)
                    img, _, _ = model.render(L.merge(base, e), ridx, K, V, scene.W, scene.H)
                    m = metrics.all_metrics(img, gt)
                    rows.append({"chain": ch, "method": meth, "view": vi, "slot": t, "outage_slot": bool(out_ind[t]),
                                 "n_lost": int(lost.numel()), "n_visible": int(idx.numel()), **m,
                                 "psnr_vs_full": metrics.psnr(img.clamp(0, 1), full[vi].clamp(0, 1))})
            print(f"chain {ch}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for meth in methods:
        r = [x for x in rows if x["method"] == meth]
        rl = [x for x in r if x["n_lost"] > 0]
        summ[meth] = {"all": {k: float(np.mean([x[k] for x in r])) for k in ("psnr", "ssim", "lpips", "psnr_vs_full")},
                      "views_with_loss": len(rl),
                      "with_loss": {k: float(np.mean([x[k] for x in rl])) for k in ("psnr", "ssim", "lpips", "psnr_vs_full")} if rl else None,
                      "min_psnr_vs_full": float(min(x["psnr_vs_full"] for x in r))}
    res = {"scene": a.scene, "tag": a.tag, "cfg": a.cfg, "tau_c_ms": a.tau, "cache_frac": a.cache_frac, "chains": a.chains, "slots": T,
           "summary": summ, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"traj{a.tag}_{a.cfg}_tau{a.tau:g}_cache{a.cache_frac:g}.json"), "w") as f:
        json.dump(res, f)
    for k, v in summ.items():
        print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="_a150")
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--chains", type=int, default=20)
    ap.add_argument("--methods", nargs="*", default=["gen_fp", "gen_tern", "lite_fp", "lite_tern"])
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-frac", type=float, default=1.0, help="edge cache capacity as a fraction of the anchors (LRU)")
    main(ap.parse_args())
