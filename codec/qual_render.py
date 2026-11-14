"""Renders for the qualitative figure: one test view under one failure pattern of a trace (same protocol as eval_layered.py).

For the given scene and test view, the most frequent failure pattern of the trace with K failed streams is applied
with interleaved mapping; the view is rendered without loss, with the lost anchors dropped, with IDW interpolation and
with the concealment network (full precision). Images are saved as PNG with their PSNR against the ground truth and
Psi against the loss-free rendering.
Output: <out>/qual/<scene>_v<view>_<cfg>_K<K>_{gt,full,drop,idw,gen}.png and <scene>_v<view>_<cfg>_K<K>.json
usage: python qual_render.py --scene garden --view 0 --cfg T8R8 --K 2
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch

import metrics
from conceal_net import knn
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_layered import LayeredConcealer


def main(a):
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = LayeredConcealer(model, L, nets["gen_fp"], stats, nbr, ck["radius"])
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx = cf["tx"]; M = len(tx)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
    tr = np.load(os.path.join(a.phy, f"traces_{a.s2}.npz"))[a.cfg]
    pats, cnt = np.unique(tr[tr > 0], return_counts=True)
    ks = np.array([bin(int(p)).count("1") for p in pats])
    sel = np.where(ks == a.K)[0]; pat = int(pats[sel[np.argmax(cnt[sel])]])
    failed = patterns_to_streams(pat, tx, rel)
    vi = int(scene.test_idx[a.view])
    od = os.path.join(a.out, "qual"); os.makedirs(od, exist_ok=True)
    stem = os.path.join(od, f"{a.scene}_v{a.view}_{a.cfg}_K{a.K}")
    res = {"scene": a.scene, "view_index": vi, "cfg": a.cfg, "pattern": pat, "failed_streams": failed, "policy": "interleaved"}
    with torch.no_grad():
        gt, K, V = scene.view(vi)
        idx = model.visible(K, V, scene.W, scene.H)
        full, _, _ = model.render(at, idx, K, V, scene.W, scene.H)
        lm = pk.lost_mask(idx, M, "interleaved", failed, np.random.default_rng(vi * 1000 + pat))
        lost = idx[lm]
        avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda"); avail[idx[~lm]] = True
        res["lost_frac"] = float(lm.float().mean())
        imgs = {"gt": gt, "full": full}
        imgs["drop"], _, _ = model.render(at, idx[~lm], K, V, scene.W, scene.H)
        nb = nbr[lost]; r = avail[nb].float()
        w = r / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
        m_ = (w[..., None] * enh[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        e = enh.clone(); e[lost] = torch.where((r.sum(1) > 0)[:, None], m_, stats[2].expand_as(m_))
        imgs["idw"], _, _ = model.render(L.merge(base, e), idx, K, V, scene.W, scene.H)
        e = enh.clone(); e[lost] = conc.predict(lost, avail, base, enh)
        imgs["gen"], _, _ = model.render(L.merge(base, e), idx, K, V, scene.W, scene.H)
        for k, img in imgs.items():
            img = img.clamp(0, 1)
            imageio.imwrite(f"{stem}_{k}.png", (img.cpu().numpy() * 255).round().astype(np.uint8))
            if k != "gt":
                res[k] = {"psnr": metrics.psnr(img, gt), "psi": metrics.psnr(img, full.clamp(0, 1)) if k != "full" else None}
    json.dump(res, open(stem + ".json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--view", type=int, default=0, help="index into the test views")
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    main(ap.parse_args())
