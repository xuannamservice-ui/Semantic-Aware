"""HAC under the failure traces of the OAM link (same protocol as eval_layered.py; no change to HAC itself).

Loads the decoded HAC model saved after HAC's own encode->decode round trip and renders the held-out test views with
HAC's renderer. For every test view, n_pat outage patterns are drawn from the 1e6-slot trace of a configuration with
their empirical frequencies (same generator, seed and order as eval_layered.py, so the same patterns per view). The
visible anchors are packetised exactly as in streaming.py (tokens of G Morton-consecutive anchors, token -> stream by
the mapping policy, stream m on the m-th most reliable mode); an anchor whose token is on a failed stream loses its
coded attributes (features, offsets, scalings). As for the layered codec, anchor positions are assumed to arrive on
the protected channel, and HAC's per-Gaussian masks are kept (both favourable to HAC).
Methods: drop (lost anchors not rendered), idw (inverse-distance mean of the attributes of the 16 nearest received
visible anchors), prior (the mean that HAC's own hash-grid context model predicts for the attributes of the anchor
from its position; HAC's best estimate without the anchor's bitstream). Metrics: HAC's own psnr / ssim / lpips(vgg) on clamped float renders, and Psi = PSNR against the
loss-free HAC rendering of the same view.
Output: --out JSON with rows and a summary per policy/method.
usage (in the HAC folder, HAC environment):
  python hac_trace_eval.py -s data/mipnerf360/garden -m outputs/mipnerf360/garden/0.004 --eval \
      --phy ~/NamNX'sPaper/Q1_2nd/codec/phy --cfg T8R8 --out garden_0.004_T8R8.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import prefilter_voxel, render  # noqa: E402
from lpipsPyTorch import lpips  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402

POLICIES = ("interleaved", "contiguous")


def morton_codes(keys):
    def spread(v):
        v = v & 0x1FFFFF
        v = (v | v << 32) & 0x1F00000000FFFF
        v = (v | v << 16) & 0x1F0000FF0000FF
        v = (v | v << 8) & 0x100F00F00F00F00F
        v = (v | v << 4) & 0x10C30C30C30C30C3
        v = (v | v << 2) & 0x1249249249249249
        return v
    return spread(keys[:, 0]) | (spread(keys[:, 1]) << 1) | (spread(keys[:, 2]) << 2)


def lost_mask(morton, idx, M, policy, failed, G):
    order = torch.argsort(morton[idx])
    T = (idx.numel() + G - 1) // G
    tid = torch.arange(idx.numel(), device=idx.device) // G
    st = torch.arange(T, device=idx.device) % M if policy == "interleaved" else \
        (torch.arange(T, device=idx.device) * M) // T
    f = torch.zeros(M, dtype=torch.bool, device=idx.device)
    if failed:
        f[torch.as_tensor(failed, device=idx.device)] = True
    out = torch.empty(idx.numel(), dtype=torch.bool, device=idx.device)
    out[order] = f[st[tid]]
    return out


def knn_idw(xyz, lost, recv, attrs, k=16, chunk=1024):
    """Inverse-distance mean of attrs over the k nearest anchors of recv for every anchor of lost."""
    outs = []
    P = xyz[recv]
    for c in torch.split(lost, chunk):
        d = torch.cdist(xyz[c], P)
        dk, ik = d.topk(k, dim=1, largest=False)
        w = 1.0 / dk.clamp_min(1e-6)
        nb = recv[ik]
        outs.append([(w.view(*w.shape, *([1] * (a.dim() - 1))) * a[nb]).sum(1) /
                     w.sum(1).view(-1, *([1] * (a.dim() - 1))) for a in attrs])
    return [torch.cat([o[i] for o in outs]) for i in range(len(attrs))]


def context_prior(g, idx, chunk=20000):
    """Means of HAC's context model (hash grid + MLP at the anchor position) for features, scalings and offsets."""
    fs, ss, os_ = [], [], []
    for c in torch.split(idx, chunk):
        ctx = g.get_grid_mlp(g.calc_interp_feat(g._anchor[c]))
        mf, _, msc, _, mo, _, _, _, _ = torch.split(
            ctx, [g.feat_dim, g.feat_dim, 6, 6, 3 * g.n_offsets, 3 * g.n_offsets, 1, 1, 1], dim=-1)
        fs.append(mf); ss.append(msc); os_.append(mo.reshape(-1, g.n_offsets, 3))
    return torch.cat(fs), torch.cat(ss), torch.cat(os_)


def main():
    ap = argparse.ArgumentParser()
    lp, pp = ModelParams(ap), PipelineParams(ap)
    ap.add_argument("--phy", required=True)
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--n-pat", type=int, default=6)
    ap.add_argument("--G", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="drop,idw", help="comma list of drop, idw, prior")
    args = ap.parse_args()
    dataset, pipe = lp.extract(args), pp.extract(args)
    global METHODS
    METHODS = tuple(args.methods.split(","))
    rng = np.random.default_rng(args.seed)
    phy = os.path.expanduser(args.phy)
    ms = json.load(open(os.path.join(phy, f"modeset_{args.s2}.json")))
    cf = ms["configs"][args.cfg]; tx = cf["tx"]; M = len(tx)
    pm = cf["per_mode_count"][cf["snr_labels"].index("op")]
    rel = [tx[i] for i in np.argsort(np.asarray(pm), kind="stable")]
    tr = np.load(os.path.join(phy, f"traces_{args.s2}.npz"))[args.cfg]
    pats, cnt = np.unique(tr[tr > 0], return_counts=True)

    with torch.no_grad():
        g = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                          dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                          n_features_per_level=4, log2_hashmap_size=13, log2_hashmap_size_2D=15,
                          decoded_version=True)
        scene = Scene(dataset, g, load_iteration=-1, shuffle=False)
        g.eval()
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        cams = scene.getTestCameras()
        valid = g.get_mask_anchor.clone().bool().view(-1)
        xyz = g._anchor.detach()
        cell = max(float(dataset.voxel_size), float((xyz.max(0).values - xyz.min(0).values).max()) / (2 ** 21 - 1))
        keys = torch.floor((xyz - xyz.min(0).values) / cell).long()     # voxel coordinates, < 2^21 per axis
        morton = morton_codes(keys)
        feat0, off0, sc0 = g._anchor_feat.clone(), g._offset.clone(), g._scaling.clone()
        if "prior" in METHODS:                  # bounds of the hash-grid coordinates, as in probe_loss.py
            from utils.encodings import Q_anchor
            N = int(valid.sum()); A = g._anchor[:N]
            q = np.load(os.path.join(dataset.model_path, "bitstreams", "_quantized_v.npy")).astype(np.int64)
            Ad = A.double().cpu().numpy(); xmin = np.zeros(3); itv = np.zeros(3)
            for c in range(3):
                X = np.stack([q[:, c].astype(np.float64), np.ones(N)], 1)
                itv[c], xmin[c] = np.linalg.lstsq(X, Ad[:, c], rcond=None)[0]
            g.x_bound_min = torch.tensor(xmin, dtype=torch.float32, device="cuda")[None]
            g.x_bound_max = torch.tensor((itv - 1e-6) / Q_anchor + xmin, dtype=torch.float32, device="cuda")[None]
        rows, t0 = [], time.time()

        def shot(cam, vis):
            img = torch.clamp(render(cam, g, pipe, bg, visible_mask=vis)["render"], 0.0, 1.0)
            return img

        for vi, cam in enumerate(cams):
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            vis = prefilter_voxel(cam, g, pipe, bg) & valid
            full = shot(cam, vis)
            fp = psnr(full, gt).mean().item()
            idx = vis.nonzero().view(-1)
            for pi in rng.choice(len(pats), args.n_pat, p=cnt / cnt.sum()):
                failed = [rel.index(tx[i]) for i in range(M) if (int(pats[pi]) >> i) & 1]
                for pol in POLICIES:
                    lm = lost_mask(morton, idx, M, pol, failed, args.G)
                    lost, recv = idx[lm], idx[~lm]
                    for meth in METHODS:
                        v = vis.clone()
                        if meth == "drop":
                            v[lost] = False
                        elif meth == "prior" and lost.numel():
                            f_, s_, o_ = context_prior(g, lost)
                            g._anchor_feat[lost] = f_.to(feat0.dtype); g._scaling[lost] = s_.to(sc0.dtype); g._offset[lost] = o_.to(off0.dtype)
                        elif lost.numel():
                            f_, o_, s_ = knn_idw(xyz, lost, recv, (feat0, off0, sc0))
                            g._anchor_feat[lost] = f_; g._offset[lost] = o_; g._scaling[lost] = s_
                        img = shot(cam, v)
                        g._anchor_feat.copy_(feat0); g._offset.copy_(off0); g._scaling.copy_(sc0)
                        rows.append({"view": vi, "pattern": int(pats[pi]), "K": len(failed), "policy": pol,
                                     "method": meth, "lost_frac": float(lm.float().mean()),
                                     "psnr": psnr(img, gt).mean().item(), "ssim": ssim(img, gt).mean().item(),
                                     "lpips": lpips(img, gt, net_type="vgg").mean().item(),
                                     "psnr_vs_full": psnr(img, full).mean().item(), "full_psnr": fp})
            print(f"view {vi}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for pol in POLICIES:
        for meth in METHODS:
            r = [x for x in rows if x["policy"] == pol and x["method"] == meth]
            summ[f"{pol}/{meth}"] = {k: float(np.mean([x[k] for x in r]))
                                     for k in ("psnr", "ssim", "lpips", "psnr_vs_full", "lost_frac")}
    summ["full"] = {"psnr": float(np.mean([x["full_psnr"] for x in rows]))}
    json.dump({"model_path": dataset.model_path, "cfg": args.cfg, "s2": args.s2, "tx": tx, "reliability_order": rel,
               "n_pat": args.n_pat, "seed": args.seed, "G": args.G, "n_anchors_valid": int(valid.sum()),
               "summary": summ, "rows": rows}, open(args.out, "w"))
    for k, v in summ.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
