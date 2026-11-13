"""Reference 3D Gaussian Splatting model per scene (gsplat rasterizer, MCMC densification with a cap on the count).

Settings follow the gsplat MCMC example (SfM initialisation, SH degree 3, L1 + 0.2 D-SSIM, opacity and scale
regularisation 0.01). Output: <out>/<scene>/gs.pt (parameters) and metrics.json (test views: every 8th image).
usage: python train_3dgs.py --scene bicycle --cap 500000 --steps 30000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

import metrics
from scene_data import Scene, scene_location

OUTDOOR = ("bicycle", "garden", "stump", "flowers", "treehill")


def knn_mean_dist(x, k=3):
    x = torch.as_tensor(x, device="cuda")
    d = []
    for c in torch.split(x, 4096):
        dd = torch.cdist(c, x)
        d.append(dd.topk(k + 1, largest=False).values[:, 1:].mean(1))
    return torch.cat(d)


def init_params(scene, sh_degree):
    pts = torch.from_numpy(scene.points).cuda()
    rgb = torch.from_numpy(scene.points_rgb).cuda()
    dist = knn_mean_dist(pts).clamp_min(1e-7)
    N = pts.shape[0]
    shs = torch.zeros(N, (sh_degree + 1) ** 2, 3, device="cuda")
    shs[:, 0] = (rgb - 0.5) / 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.rand(N, 4, device="cuda")),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device="cuda"))),
        "sh0": torch.nn.Parameter(shs[:, :1]),
        "shN": torch.nn.Parameter(shs[:, 1:]),
    })
    return params


def render(params, K, V, W, H, sh_degree):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]), colors, V[None], K[None], W, H,
                                     sh_degree=sh_degree, packed=False)
    return img[0], info


def evaluate(scene, params, sh_degree):
    out = []
    with torch.no_grad():
        for i in scene.test_idx:
            gt, K, V = scene.view(i)
            img, _ = render(params, K, V, scene.W, scene.H, sh_degree)
            out.append(metrics.all_metrics(img, gt))
    return {k: float(np.mean([m[k] for m in out])) for k in out[0]}


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    out = os.path.join(a.out, a.scene); os.makedirs(out, exist_ok=True)
    params = init_params(scene, a.sh_degree)
    s = scene.scene_scale
    lrs = {"means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / a.steps))
    strat = MCMCStrategy(cap_max=a.cap, refine_stop_iter=int(a.steps * 25 / 30))
    strat.check_sanity(params, opt)
    state = strat.initialize_state()
    t0 = time.time(); log = []
    order = np.random.permutation(scene.train_idx)
    for step in range(a.steps):
        i = order[step % len(order)]
        if step % len(order) == len(order) - 1:
            order = np.random.permutation(scene.train_idx)
        gt, K, V = scene.view(i)
        shd = min(step // 1000, a.sh_degree)
        img, info = render(params, K, V, scene.W, scene.H, shd)
        l1 = F.l1_loss(img, gt)
        ss = metrics.ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).mean()
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        strat.step_post_backward(params, opt, state, step, info, lr=sched.get_last_lr()[0])
        if step % 1000 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "n": int(params["means"].shape[0]), "t": time.time() - t0}
            log.append(rec); print(rec, flush=True)
    res = {"scene": a.scene, "img_dir": scene_location(a.scene, a.data)[1], "H": scene.H, "W": scene.W, "n_train": len(scene.train_idx),
           "n_test": len(scene.test_idx), "n_gaussians": int(params["means"].shape[0]), "steps": a.steps,
           "cap": a.cap, "seed": a.seed, "train_time_s": time.time() - t0, "test": evaluate(scene, params, a.sh_degree),
           "log": log}
    torch.save({k: v.detach().cpu() for k, v in params.items()} | {"offset": torch.from_numpy(scene.offset),
               "scene_scale": scene.scene_scale}, os.path.join(out, "gs.pt"))
    with open(os.path.join(out, "metrics_3dgs.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("test", res["test"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cap", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
