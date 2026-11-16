"""Train the deployment concealment network (LiteConcealNet) on a frozen layered representation.

The representation (anchor attributes and decoder MLPs) is loaded from layered<tag>.pt and kept fixed. Each step:
training view, simulated loss of K of M streams (as in train_layered.py), prediction of the lost enhancement layers,
rendering; loss = 0.8 L1 + 0.2 D-SSIM of the concealed rendering + 0.1 normalised attribute MSE. Full-precision
training, then ternary (BitLinear) QAT initialised from it. Output: <out>/<scene>/lite<tag>.pt
usage: python train_lite.py --scene garden --tag _a150 --steps 10000 --steps-ternary 5000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from conceal_net import EdgeConcealNet, LiteConcealNet, MLPConcealNet, knn
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer
from train_3dgs import OUTDOOR
from train_layered import LayeredConcealer, rloss, simulate_loss


def main(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, _, c, ck = load(a.out, a.scene, a.tag)
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    out, log, t0 = {}, [], time.time()
    Net = {"edge": EdgeConcealNet, "lite": LiteConcealNet, "mlp": MLPConcealNet}[a.arch]
    net = Net(c["bdim"], c["edim"], d=a.d, layers=a.layers).cuda()
    for phase, steps in (("fp", a.steps), ("ternary", a.steps_ternary)):
        if phase == "ternary":
            t = Net(c["bdim"], c["edim"], d=a.d, layers=a.layers, ternary=True).cuda()
            t.load_state_dict(net.state_dict()); out["fp"] = {k: v.cpu() for k, v in net.state_dict().items()}; net = t
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr if phase == "fp" else a.lr / 4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
        conc = LayeredConcealer(model, L, net, stats, nbr, ck["radius"])
        for step in range(steps):
            i = rng.choice(scene.train_idx)
            gt, K, V = scene.view(i)
            idx = model.visible(K, V, scene.W, scene.H)
            lost, avail = simulate_loss(pk, idx, rng, model.xyz.shape[0])
            if lost.numel() > a.max_lost:
                lost = lost[torch.randperm(lost.numel(), device="cuda")[:a.max_lost]]
            pred = conc.predict(lost, avail, base, enh)
            e = enh.clone(); e[lost] = pred
            img, _, _ = model.render(L.merge(base, e), idx, K, V, scene.W, scene.H)
            la = F.mse_loss((pred - stats[2]) / stats[3], (enh[lost] - stats[2]) / stats[3])
            loss = rloss(img, gt) + 0.1 * la
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            if step % 1000 == 0 or step == steps - 1:
                rec = {"phase": phase, "step": step, "loss": float(loss), "attr_mse": float(la), "t": time.time() - t0}
                log.append(rec); print(rec, flush=True)
    out["ternary"] = {k: v.cpu() for k, v in net.state_dict().items()}
    torch.save({"net_fp": out["fp"], "net_ternary": out["ternary"], "cfg": {"d": a.d, "layers": a.layers, "arch": a.arch},
                "args": vars(a), "log": log}, os.path.join(a.out, a.scene, f"{a.arch}{a.tag}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="_a150")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--steps-ternary", type=int, default=5_000)
    ap.add_argument("--arch", default="lite", choices=("lite", "edge", "mlp"))
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-lost", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
