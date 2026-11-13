"""Trace-driven evaluation of the layered representation on the held-out test views of one scene.

Loss patterns: per-slot failure patterns of the OAM link simulator (phy/traces_<s2>.npz, bit i = i-th transmitted
mode failed); stream m of the packetiser travels on the m-th most reliable mode (marginal outage from
phy/modeset_<s2>.json). The base layer and the positions travel on the replicated anchor pair and are received.
For every test view, n_pat outage patterns are drawn with their empirical frequencies.
Methods for the enhancement layer of lost anchors:
  drop     : lost anchors not rendered (no concealment, base layer unused)
  base     : enhancement set to its mean (base layer only)
  idw      : inverse-distance mean of the received neighbours' enhancement layers (base layer kept)
  gen_fp   : concealment network, full precision
  gen_tern : concealment network, ternary (1.58-bit) weights
Output: <out>/<scene>/eval_layered<tag>_<cfg>_<s2>.json
usage: python eval_layered.py --scene garden --tag _a80 --cfg T6R11 --s2 2.5 --n-pat 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import metrics
from anchor_model import AnchorModel
from conceal_net import EdgeConcealNet, LayeredConcealNet, LiteConcealNet, knn
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_3dgs import OUTDOOR
from train_layered import Layers, LayeredConcealer

POLICIES = ("random", "contiguous", "interleaved", "importance")
METHODS = ("drop", "base", "idw", "gen_fp", "gen_tern")


def load(out, scene, tag):
    ck = torch.load(os.path.join(out, scene, f"layered{tag}.pt"))
    st = ck["state"]
    model = AnchorModel(st["xyz"], ck["voxel"], feat_dim=ck["feat_dim"], k=ck["k"]).cuda()
    model.load_state_dict(st); model.eval()
    L = Layers(model, ck["base_dim"])
    c = ck["net_cfg"]
    nets = {}
    Net = LiteConcealNet if ck.get("net_type") == "lite" else LayeredConcealNet
    for name, key, tern in (("gen_fp", "net_fp", False), ("gen_tern", "net_ternary", True)):
        n = Net(c["bdim"], c["edim"], d=c["d"], layers=c["layers"], ternary=tern).cuda()
        n.load_state_dict(ck[key]); n.eval(); nets[name] = n
    for arch, Cls in (("lite", LiteConcealNet), ("edge", EdgeConcealNet)):
        lp = os.path.join(out, scene, f"{arch}{tag}.pt")
        if os.path.exists(lp):                  # deployment networks trained on the frozen representation
            lk = torch.load(lp)
            for name, key, tern in ((f"{arch}_fp", "net_fp", False), (f"{arch}_tern", "net_ternary", True)):
                n = Cls(c["bdim"], c["edim"], d=lk["cfg"]["d"], layers=lk["cfg"]["layers"], ternary=tern).cuda()
                n.load_state_dict(lk[key]); n.eval(); nets[name] = n
    return model, L, nets, c, ck


def main(a):
    rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True)
        base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = {k: LayeredConcealer(model, L, n, stats, nbr, ck["radius"]) for k, n in nets.items()}
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx = cf["tx"]
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
    tr = np.load(os.path.join(a.phy, f"traces_{a.s2}.npz"))[a.cfg]
    pats, cnt = np.unique(tr[tr > 0], return_counts=True)
    M = len(tx); rows = []; t0 = time.time()
    methods = ("drop", "base", "idw") + tuple(nets)
    with torch.no_grad():
        for vi in scene.test_idx:
            gt, K, V = scene.view(vi)
            idx = model.visible(K, V, scene.W, scene.H)
            full, _, _ = model.render(at, idx, K, V, scene.W, scene.H)
            mf = metrics.all_metrics(full, gt)
            for pi in rng.choice(len(pats), a.n_pat, p=cnt / cnt.sum()):
                failed = patterns_to_streams(pats[pi], tx, rel)
                for pol in POLICIES:
                    lost_m = pk.lost_mask(idx, M, pol, failed, np.random.default_rng(int(vi) * 1000 + int(pi)))
                    lost = idx[lost_m]
                    avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                    avail[idx[~lost_m]] = True
                    for meth in methods:
                        e = enh.clone(); ridx = idx
                        if meth == "drop":
                            ridx = idx[~lost_m]
                        elif meth == "base":
                            e[lost] = stats[2]
                        elif meth == "idw":
                            nb = nbr[lost]; r = avail[nb].float()
                            w = r / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
                            m_ = (w[..., None] * enh[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
                            has = r.sum(1) > 0
                            e[lost] = torch.where(has[:, None], m_, stats[2].expand_as(m_))
                        else:
                            e[lost] = conc[meth].predict(lost, avail, base, enh)
                        img, _, _ = model.render(L.merge(base, e), ridx, K, V, scene.W, scene.H)
                        m = metrics.all_metrics(img, gt)
                        rows.append({"view": int(vi), "pattern": int(pats[pi]), "K": len(failed), "policy": pol,
                                     "method": meth, "lost_frac": float(lost_m.float().mean()), **m,
                                     "psnr_vs_full": metrics.psnr(img.clamp(0, 1), full.clamp(0, 1)),
                                     "full_psnr": mf["psnr"], "full_ssim": mf["ssim"], "full_lpips": mf["lpips"]})
            print(f"view {vi}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for pol in POLICIES:
        for meth in methods:
            r = [x for x in rows if x["policy"] == pol and x["method"] == meth]
            summ[f"{pol}/{meth}"] = {k: float(np.mean([x[k] for x in r])) for k in
                                     ("psnr", "ssim", "lpips", "psnr_vs_full", "lost_frac")}
    summ["full"] = {k: float(np.mean([x["full_" + k] for x in rows])) for k in ("psnr", "ssim", "lpips")}
    res = {"scene": a.scene, "tag": a.tag, "cfg": a.cfg, "s2": a.s2, "tx": tx, "reliability_order": rel,
           "p_out_trace": float((tr > 0).mean()), "n_outage_slots": int(cnt.sum()), "n_pat": a.n_pat,
           "seed": a.seed, "summary": summ, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"eval_layered{a.tag}{a.suffix}_{a.cfg}_{a.s2}.json"), "w") as f:
        json.dump(res, f)
    for k, v in summ.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cfg", default="T6R11")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--n-pat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="suffix of the output file")
    main(ap.parse_args())
