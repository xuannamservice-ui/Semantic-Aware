"""Train the generative concealment network of one scene (full precision, then ternary QAT).

Each step draws a training view, packetises its visible anchors into tokens and M streams (M in {6, 8}, a random
mapping policy), fails K streams (K = 1, 2, 3 with probabilities 0.7, 0.2, 0.1), predicts the attributes of the
lost anchors from their received neighbours, and renders the view with the concealed attributes.
Loss: 0.8 L1 + 0.2 D-SSIM of the rendering against the ground-truth image + 0.1 MSE of the normalised attributes.
Neighbours count as received only if they are visible in the slot and not lost (no cache of earlier slots).
Output: <out>/<scene>/conceal.pt, log_conceal.json
usage: python train_conceal.py --scene bicycle --steps-fp 20000 --steps-ternary 10000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import metrics
from anchor_model import AnchorModel
from conceal_net import ConcealNet, knn
from scene_data import Scene, scene_location
from streaming import Packetizer
from train_3dgs import OUTDOOR

POLICIES = ("random", "contiguous", "interleaved", "importance")


def load_anchor(out, scene_name):
    ck = torch.load(os.path.join(out, scene_name, "anchor.pt"))
    st = ck["state"]
    model = AnchorModel(st["xyz"], ck["voxel"], feat_dim=ck["feat_dim"], k=ck["k"]).cuda()
    model.load_state_dict(st)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["importance"].cuda()


def flat_attrs(model):
    with torch.no_grad():
        a = model.attributes(quantized=True)
    A = model.xyz.shape[0]
    return torch.cat([a["feat"], a["offsets"].reshape(A, -1), a["scaling"]], 1)


def unflat(model, v):
    F_, k = model.feat_dim, model.k
    return {"feat": v[:, :F_], "offsets": v[:, F_:F_ + 3 * k].reshape(-1, k, 3), "scaling": v[:, F_ + 3 * k:]}


class Concealer:
    """Neighbourhood gathering + network call, shared by training, evaluation and the latency benchmark."""

    def __init__(self, model, net, mean, std, nbr, radius):
        self.model, self.net, self.mean, self.std, self.nbr, self.radius = model, net, mean, std, nbr, radius

    def predict(self, lost, available, attrs_flat, chunk=16384):
        """lost: (L,) anchor ids; available: (A,) bool (received in this slot); returns (L, attr_dim) attributes."""
        outs = []
        for c in torch.split(lost, chunk):
            nb = self.nbr[c]                                                     # (B, Kn)
            rel = (self.model.xyz[nb] - self.model.xyz[c][:, None]) / self.radius
            recv = available[nb]
            a = (attrs_flat[nb] - self.mean) / self.std
            outs.append(self.net(rel, a, recv) * self.std + self.mean)
        return torch.cat(outs) if outs else attrs_flat[:0]


def main(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, imp = load_anchor(a.out, a.scene)
    X = flat_attrs(model)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-4)
    nbr = knn(model.xyz, a.kn)
    radius = float((model.xyz[nbr[:, -1]] - model.xyz).norm(dim=1).median())
    pk = Packetizer(model.xyz, model.voxel, imp, G=a.G)
    net = ConcealNet(X.shape[1], d=a.d, layers=a.layers).cuda()
    log, t0 = [], time.time()
    for phase, steps in (("fp", a.steps_fp), ("ternary", a.steps_ternary)):
        if steps == 0:
            continue
        if phase == "ternary":
            tnet = ConcealNet(X.shape[1], d=a.d, layers=a.layers, ternary=True).cuda()
            tnet.load_state_dict(net.state_dict()); net_fp = net; net = tnet
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr if phase == "fp" else a.lr / 4, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        conc = Concealer(model, net, mean, std, nbr, radius)
        for step in range(steps):
            i = rng.choice(scene.train_idx)
            gt, K, V = scene.view(i)
            idx = model.visible(K, V, scene.W, scene.H)
            M = int(rng.choice([6, 8]))
            Kf = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
            failed = rng.choice(M, Kf, replace=False)
            lost_m = pk.lost_mask(idx, M, POLICIES[rng.integers(4)], failed, rng)
            lost = idx[lost_m]
            avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
            avail[idx[~lost_m]] = True
            sub = lost[torch.randperm(lost.numel(), device="cuda")[:a.max_lost]] if lost.numel() > a.max_lost else lost
            pred = conc.predict(sub, avail, X)
            Xh = X.clone()
            Xh[sub] = pred
            img, _, _ = model.render(unflat(model, Xh), idx, K, V, scene.W, scene.H)
            l1 = F.l1_loss(img, gt)
            ss = metrics.ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()
            la = F.mse_loss((pred - mean) / std, (X[sub] - mean) / std)
            loss = 0.8 * l1 + 0.2 * (1 - ss) + 0.1 * la
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 500 == 0 or step == steps - 1:
                rec = {"phase": phase, "step": step, "loss": float(loss), "attr_mse": float(la), "lost": int(lost.numel()),
                       "visible": int(idx.numel()), "t": time.time() - t0}
                log.append(rec); print(rec, flush=True)
    torch.save({"fp": (net_fp if a.steps_ternary else net).state_dict(),
                "ternary": net.state_dict() if a.steps_ternary else None,
                "cfg": {"d": a.d, "layers": a.layers, "kn": a.kn, "G": a.G, "attr_dim": X.shape[1]},
                "mean": mean.cpu(), "std": std.cpu(), "radius": radius},
               os.path.join(a.out, a.scene, "conceal.pt"))
    with open(os.path.join(a.out, a.scene, "log_conceal.json"), "w") as f:
        json.dump({"args": vars(a), "log": log}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps-fp", type=int, default=20_000)
    ap.add_argument("--steps-ternary", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kn", type=int, default=16)
    ap.add_argument("--G", type=int, default=64)
    ap.add_argument("--max-lost", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
