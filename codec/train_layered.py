"""Concealment-aware layered anchor representation, trained jointly with the generative concealment network.

Attributes of anchor a are split into a base layer (first Fb feature channels and the 6 scaling values), sent on
the replicated anchor channel together with the anchor position, and an enhancement layer (remaining feature
channels and the k x 3 offsets), sent on the data streams and exposed to loss. Training:
  steps [0, warm):        clean rendering loss only
  steps [warm, steps):    clean loss + loss of a rendering in which the enhancement layer of the anchors on K
                          failed streams (M in {6, 8}, K in {1, 2, 3}, random mapping policy) is replaced by the
                          prediction of the concealment network; + 0.1 MSE of the prediction (target detached)
  then ternary steps:     representation frozen, BitLinear (1.58-bit) copy of the network fine-tuned (QAT).
Rendering losses: 0.8 L1 + 0.2 D-SSIM; scale-volume regulariser 0.01; attributes quantization-aware from q_start.
Output: <out>/<scene>/layered.pt, metrics_layered.json
usage: python train_layered.py --scene garden --anchors 80000 --steps 30000 --steps-ternary 8000
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
from conceal_net import LayeredConcealNet, LiteConcealNet, MLPConcealNet, knn
from scene_data import Scene, scene_location
from streaming import Packetizer
from train_3dgs import OUTDOOR
from train_anchor import QSTEP, entropy_bits, pick_voxel

TRAIN_POLICIES = ("random", "contiguous", "interleaved")


class RateModel(torch.nn.Module):
    """Factorised entropy model: per attribute channel, a Laplace density with learned location and scale. The bits of
    a quantized value q (step D) are -log2(F(q + D/2) - F(q - D/2)); used as a rate term during training and as the
    model code length at the end (arithmetic coding with this model needs about that many bits)."""

    def __init__(self, dims):
        super().__init__()
        self.loc = torch.nn.ParameterDict({n: torch.nn.Parameter(torch.zeros(d)) for n, d in dims.items()})
        self.log_b = torch.nn.ParameterDict({n: torch.nn.Parameter(torch.zeros(d)) for n, d in dims.items()})

    @staticmethod
    def _cdf(x):
        return 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs()))

    def bits(self, name, q, step):
        """q: (A, C) quantized values; returns (A, C) bits."""
        b = torch.exp(self.log_b[name]).clamp_min(1e-3 * step)
        mu = self.loc[name]
        p = self._cdf((q + step / 2 - mu) / b) - self._cdf((q - step / 2 - mu) / b)
        return -torch.log2(p.clamp_min(1e-9))


class Layers:
    """Split / merge of the attribute dictionary into base and enhancement layers."""

    def __init__(self, model, fb):
        self.m, self.fb, self.k = model, fb, model.k

    def split(self, at):
        A = at["feat"].shape[0]
        base = torch.cat([at["feat"][:, :self.fb], at["scaling"]], 1)
        enh = torch.cat([at["feat"][:, self.fb:], at["offsets"].reshape(A, -1)], 1)
        return base, enh

    def merge(self, base, enh):
        fb, k = self.fb, self.k
        fe = self.m.feat_dim - fb
        return {"feat": torch.cat([base[:, :fb], enh[:, :fe]], 1), "scaling": base[:, fb:],
                "offsets": enh[:, fe:].reshape(-1, k, 3)}


class LayeredConcealer:
    def __init__(self, model, layers, net, stats, nbr, radius):
        self.model, self.L, self.net, self.stats, self.nbr, self.radius = model, layers, net, stats, nbr, radius

    def predict(self, lost, avail, base, enh, chunk=16384):
        bm, bs, em, es = self.stats
        outs = []
        for c in torch.split(lost, chunk):
            nb = self.nbr[c]
            rel = (self.model.xyz[nb] - self.model.xyz[c][:, None]) / self.radius
            p = self.net(rel, (base[nb] - bm) / bs, (enh[nb] - em) / es, avail[nb], (base[c] - bm) / bs)
            outs.append(p * es + em)
        return torch.cat(outs) if outs else enh[:0]


def rloss(img, gt):
    return 0.8 * F.l1_loss(img, gt) + 0.2 * (1 - metrics.ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean())


def simulate_loss(pk, idx, rng, xyz_n):
    M = int(rng.choice([6, 8])); K = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
    lost_m = pk.lost_mask(idx, M, TRAIN_POLICIES[rng.integers(len(TRAIN_POLICIES))], rng.choice(M, K, replace=False), rng)
    avail = torch.zeros(xyz_n, dtype=torch.bool, device=idx.device)
    avail[idx[~lost_m]] = True
    return idx[lost_m], avail


def main(a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    out = os.path.join(a.out, a.scene)
    gs = torch.load(os.path.join(out, "gs.pt"))
    pts, opa = gs["means"].cuda(), torch.sigmoid(gs["opacities"]).cuda()
    voxel = pick_voxel(pts, opa, a.anchors)
    model = AnchorModel(voxel_anchors(pts, opa, voxel), voxel, feat_dim=a.feat_dim, k=a.k).cuda()
    for n in model.log_qstep:
        model.log_qstep[n].data.fill_(math.log(QSTEP[n])); model.log_qstep[n].requires_grad_(False)
    L = Layers(model, a.base_dim)
    bdim, edim = a.base_dim + 6, a.feat_dim - a.base_dim + 3 * a.k
    nbr = knn(model.xyz, a.kn)
    radius = float((model.xyz[nbr[:, -1]] - model.xyz).norm(dim=1).median())
    pk = Packetizer(model.xyz, voxel, torch.ones(model.xyz.shape[0], device="cuda"), G=a.G)
    Net = {"lite": LiteConcealNet, "mlp": MLPConcealNet}.get(a.net, LayeredConcealNet)
    net = Net(bdim, edim, d=a.d, layers=a.layers).cuda()
    A0 = model.xyz.shape[0]
    rate = RateModel({"feat": a.feat_dim, "offsets": 3 * a.k, "scaling": 6}).cuda()
    ropt = torch.optim.Adam(rate.parameters(), lr=1e-2)
    groups = [{"params": [model.feat], "lr": 0.0075, "f": 0.0075}, {"params": [model.offsets], "lr": 0.01, "f": 1e-4},
              {"params": [model.scaling], "lr": 0.007, "f": 0.007},
              {"params": model.mlp_opacity.parameters(), "lr": 0.002, "f": 2e-5},
              {"params": model.mlp_cov.parameters(), "lr": 0.004, "f": 0.004},
              {"params": model.mlp_color.parameters(), "lr": 0.008, "f": 5e-5}]
    opt = torch.optim.Adam(groups, eps=1e-15)
    init = [g["lr"] for g in opt.param_groups]
    nopt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    nsched = torch.optim.lr_scheduler.CosineAnnealingLR(nopt, a.steps - a.warm)
    stats = None
    t0, log = time.time(), []

    def get_stats(at):
        with torch.no_grad():
            b, e = L.split(at)
            return (b.mean(0), b.std(0).clamp_min(0.02), e.mean(0), e.std(0).clamp_min(0.02))   # floor ~ quantization step

    for step in range(a.steps):
        r = step / a.steps
        for g, lr0 in zip(opt.param_groups, init):
            g["lr"] = math.exp((1 - r) * math.log(lr0) + r * math.log(g["f"]))
        i = rng.choice(scene.train_idx)
        gt, K, V = scene.view(i)
        idx = model.visible(K, V, scene.W, scene.H)
        at = model.attributes(quantized=step >= a.q_start)
        img, _, scales = model.render(at, idx, K, V, scene.W, scene.H)
        loss = rloss(img, gt) + 0.01 * scales.prod(1).mean()
        lc = la = lr_ = torch.zeros((), device="cuda")
        if step >= a.q_start:
            vals = {n: at[n][idx].reshape(idx.numel(), -1) for n in model.ATTRS}
            # entropy-model fit on detached values (updates only the rate model)
            loss = loss + sum(rate.bits(n, v.detach(), QSTEP[n]).sum(1) for n, v in vals.items()).mean() * 1e-3
            if a.rate_lambda > 0:            # rate penalty on the representation (rate model detached)
                lr_ = sum(rate.bits(n, v, QSTEP[n]).sum(1) for n, v in vals.items()).mean()
                loss = loss + a.rate_lambda * lr_
        if step >= a.warm:
            if stats is None or step % 200 == 0:
                stats = get_stats(at)
            base, enh = L.split(at)
            if not a.aware:                       # separate design: concealment gradients do not reach the representation
                base, enh = base.detach(), enh.detach()
            lost, avail = simulate_loss(pk, idx, rng, model.xyz.shape[0])
            if lost.numel() > a.max_lost:
                lost = lost[torch.randperm(lost.numel(), device="cuda")[:a.max_lost]]
            conc = LayeredConcealer(model, L, net, stats, nbr, radius)
            pred = conc.predict(lost, avail, base, enh)
            enh_h = enh.clone(); enh_h[lost] = pred
            if a.aware:
                imgc, _, _ = model.render(L.merge(base, enh_h), idx, K, V, scene.W, scene.H)
            else:                                 # decoder MLPs excluded from the concealment-path graph
                mlps = [q for m in (model.mlp_opacity, model.mlp_cov, model.mlp_color) for q in m.parameters()]
                for q in mlps:
                    q.requires_grad_(False)
                imgc, _, _ = model.render(L.merge(base, enh_h), idx, K, V, scene.W, scene.H)
                for q in mlps:
                    q.requires_grad_(True)
            lc = rloss(imgc, gt)
            la = F.mse_loss((pred - stats[2]) / stats[3], ((enh[lost] - stats[2]) / stats[3]).detach())
            loss = loss + a.lam * lc + 0.1 * la
        opt.zero_grad(set_to_none=True); nopt.zero_grad(set_to_none=True); ropt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step >= a.q_start:
            ropt.step()
        if step >= a.warm:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); nopt.step(); nsched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "conc_loss": float(lc), "attr_mse": float(la), "bits_per_anchor": float(lr_),
                   "visible": int(idx.numel()), "t": time.time() - t0}
            log.append(rec); print(rec, flush=True)
    # ternary (1.58-bit) concealment network, representation frozen
    for p in model.parameters():
        p.requires_grad_(False)
    at = model.attributes(quantized=True); stats = get_stats(at); base, enh = L.split(at)
    tnet = Net(bdim, edim, d=a.d, layers=a.layers, ternary=True).cuda()
    tnet.load_state_dict(net.state_dict())
    topt = torch.optim.AdamW(tnet.parameters(), lr=a.lr / 4)
    tsched = torch.optim.lr_scheduler.CosineAnnealingLR(topt, max(a.steps_ternary, 1))
    for step in range(a.steps_ternary):
        i = rng.choice(scene.train_idx)
        gt, K, V = scene.view(i)
        idx = model.visible(K, V, scene.W, scene.H)
        lost, avail = simulate_loss(pk, idx, rng, model.xyz.shape[0])
        if lost.numel() > a.max_lost:
            lost = lost[torch.randperm(lost.numel(), device="cuda")[:a.max_lost]]
        pred = LayeredConcealer(model, L, tnet, stats, nbr, radius).predict(lost, avail, base, enh)
        enh_h = enh.clone(); enh_h[lost] = pred
        imgc, _, _ = model.render(L.merge(base, enh_h), idx, K, V, scene.W, scene.H)
        la = F.mse_loss((pred - stats[2]) / stats[3], (enh[lost] - stats[2]) / stats[3])
        loss = rloss(imgc, gt) + 0.1 * la
        topt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(tnet.parameters(), 1.0); topt.step(); tsched.step()
        if step % 1000 == 0 or step == a.steps_ternary - 1:
            rec = {"ternary_step": step, "loss": float(loss), "attr_mse": float(la), "t": time.time() - t0}
            log.append(rec); print(rec, flush=True)
    # clean test quality, importance, rates
    model.eval(); net.eval(); tnet.eval()
    with torch.no_grad():
        test = []
        for i in scene.test_idx:
            gt, K, V = scene.view(i)
            img, _, _ = model.render(at, model.visible(K, V, scene.W, scene.H), K, V, scene.W, scene.H)
            test.append(metrics.all_metrics(img, gt))
        imp = torch.zeros(model.xyz.shape[0], device="cuda")
        for i in scene.train_idx:
            _, K, V = scene.view(i)
            idx = model.visible(K, V, scene.W, scene.H)
            cam = -V[:3, :3].T @ V[:3, 3]
            d = model.xyz[idx] - cam[None]; dist = d.norm(dim=1, keepdim=True)
            o = model.mlp_opacity(torch.cat([at["feat"][idx], d / dist, dist], 1)).clamp_min(0)
            pix = (K[0, 0] * torch.exp(at["scaling"][idx][:, 3:]).mean(1) / dist.squeeze(1)).clamp(max=64.0)
            imp[idx] += o.sum(1) * pix ** 2
    # prune anchors that contribute nothing to any training view (never visible with positive opacity)
    keep = imp > a.prune * imp.mean()
    if not bool(keep.all()):
        with torch.no_grad():
            model.xyz = model.xyz[keep]
            for n in model.ATTRS:
                setattr(model, n, torch.nn.Parameter(getattr(model, n)[keep]))
        imp = imp[keep]
        at = model.attributes(quantized=True)
    A = model.xyz.shape[0]
    with torch.no_grad():
        model_bits = {n: float(rate.bits(n, at[n].reshape(A, -1), QSTEP[n]).sum(1).mean()) for n in model.ATTRS}
    sym = {n: torch.round(getattr(model, n).detach() / QSTEP[n]).long().reshape(A, -1) for n in model.ATTRS}
    fb = a.base_dim
    bits = {"base_feat": entropy_bits(sym["feat"][:, :fb]), "scaling": entropy_bits(sym["scaling"]),
            "enh_feat": entropy_bits(sym["feat"][:, fb:]), "offsets": entropy_bits(sym["offsets"])}
    keys = torch.round(model.xyz / voxel - 0.5).long(); keys = keys - keys.min(0).values
    bits["position_fixed"] = float(sum(math.ceil(math.log2(int(k) + 1)) for k in keys.max(0).values))
    bits["base_layer"] = bits["base_feat"] + bits["scaling"]
    bits["enh_layer"] = bits["enh_feat"] + bits["offsets"]
    res = {"scene": a.scene, "anchors": A, "voxel": voxel, "args": vars(a), "train_time_s": time.time() - t0,
           "test_clean": {k: float(np.mean([m[k] for m in test])) for k in test[0]}, "test_per_view": test,
           "rate_bits_per_anchor": bits, "model_bits_per_anchor": model_bits, "anchors_before_prune": A0,
           "scene_bytes": {"positions": bits["position_fixed"] * A / 8, "base": bits["base_layer"] * A / 8,
                           "enhancement": bits["enh_layer"] * A / 8}, "log": log}
    torch.save({"state": model.state_dict(), "rate_model": rate.state_dict(), "net_type": a.net, "voxel": voxel, "feat_dim": a.feat_dim, "k": a.k, "base_dim": a.base_dim,
                "importance": imp.cpu(), "net_fp": net.state_dict(), "net_ternary": tnet.state_dict(),
                "net_cfg": {"bdim": bdim, "edim": edim, "d": a.d, "layers": a.layers, "kn": a.kn, "G": a.G},
                "radius": radius}, os.path.join(out, f"layered{a.tag}.pt"))
    with open(os.path.join(out, f"metrics_layered{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("clean test", res["test_clean"], "bits", bits, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--anchors", type=int, default=80_000)
    ap.add_argument("--feat-dim", type=int, default=32)
    ap.add_argument("--base-dim", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--warm", type=int, default=3_000)
    ap.add_argument("--q-start", type=int, default=10_000)
    ap.add_argument("--steps-ternary", type=int, default=8_000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rate-lambda", type=float, default=0.0, help="weight of the rate term (per bit per anchor)")
    ap.add_argument("--net", default="ref", choices=("ref", "lite", "mlp"), help="concealment network trained jointly")
    ap.add_argument("--prune", type=float, default=0.0, help="drop anchors with importance <= prune * mean importance")
    ap.add_argument("--aware", type=int, default=1, help="1: joint (concealment-aware) training; 0: separate design")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kn", type=int, default=16)
    ap.add_argument("--G", type=int, default=64)
    ap.add_argument("--max-lost", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
