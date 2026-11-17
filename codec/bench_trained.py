"""Decoding and rasterization of the trained representations on their test views, against the synthetic benchmark of
bench_latency.py measured back to back on the same GPU (CUDA events).

bench_latency.py times decoding (anchors -> neural Gaussians) and rendering (model.render, which decodes again and then
rasterizes) on a random model; its idle run is results/bench_l40s_idle_final.json. Here, for every scene: the synthetic
stages (same model, camera and sizes as bench_latency.py, A_vis = 110,000, 1297 x 840), then every test view of the trained
model, then the synthetic stages again. Stages: decode (neural_gaussians), raster (rasterization of decoded Gaussians) and
render (decode + raster in one call). When the GPU is shared with other jobs, the ratio idle/shared of the synthetic
stages transfers the trained times to the idle GPU: dec_cal = dec * syn_idle_decode / syn_decode, ren_cal = ren *
syn_idle_render / syn_render (render = decode + raster, as in model.render).
Output: results/bench_trained.json and results/<scene>/bench_trained.json (per view)
usage: python bench_trained.py --scenes garden,room --n-warm 5 --n-rep 30
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
from gsplat import rasterization

from anchor_model import AnchorModel
from conceal_net import EdgeConcealNet, LayeredConcealNet, LiteConcealNet
from eval_layered import load
from scene_data import Scene, scene_location


def timeit(fn, n_warm, n_rep):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    for i in range(n_rep):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    t = np.array([s.elapsed_time(e) for s, e in zip(st, en)])
    return {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95))}


def stages(model, at, idx, K, V, W, H, n_warm, n_rep):
    cam = -V[:3, :3].T @ V[:3, 3]
    g = model.neural_gaussians(at, idx, cam)
    out = {"n_gauss": int(g[0].shape[0]),
           "decode": timeit(lambda: model.neural_gaussians(at, idx, cam), n_warm, n_rep),
           "raster": timeit(lambda: rasterization(*g, V[None], K[None], W, H, packed=False), n_warm, n_rep),
           "render": timeit(lambda: model.render(at, idx, K, V, W, H), n_warm, n_rep)}
    return out


def gpu_util():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def synthetic(a):
    torch.manual_seed(0)
    dev = "cuda"
    xyz = torch.rand(250_000, 3, device=dev) * 10
    model = AnchorModel(xyz, 0.02, feat_dim=48, k=10).to(dev).eval()
    with torch.no_grad():
        model.feat.normal_(); model.offsets.normal_(0, 0.5); model.scaling.add_(torch.randn_like(model.scaling) * 0.1)
    # same draws as bench_latency.py before its visible set, so that the synthetic view is the same
    torch.randint(0, 250_000, (250_000, 16), device=dev); torch.randn(250_000, 18, device=dev); torch.randn(250_000, 66, device=dev)
    torch.rand(250_000, device=dev)
    for net in (LayeredConcealNet(18, 66), LayeredConcealNet(18, 66), LiteConcealNet(18, 66, d=96, layers=2),
                LiteConcealNet(18, 66, d=96, layers=2), EdgeConcealNet(18, 66, d=96, layers=2)):
        del net
    K = torch.tensor([[1100., 0, 1297 / 2], [0, 1100., 840 / 2], [0, 0, 1]], device=dev)
    V = torch.eye(4, device=dev); V[2, 3] = 12.0
    vis = torch.randperm(250_000, device=dev)[:110_000]
    at = {"feat": model.feat, "offsets": model.offsets, "scaling": model.scaling}
    return lambda: stages(model, at, vis, K, V, 1297, 840, a.n_warm_syn, a.n_rep_syn)


def main(a):
    idle = json.load(open(a.idle))["stages"]
    ref = {"decode": idle["decode"]["mean"], "render": idle["render"]["mean"]}
    syn = synthetic(a)
    allres = {"idle_reference": ref, "scenes": {}}
    with torch.no_grad():
        for sc in a.scenes.split(","):
            u0 = gpu_util(); s0 = syn()
            scene = Scene(*scene_location(sc, a.data))
            model, L, nets, c, ck = load(a.out, sc, a.tag)
            at = model.attributes(quantized=True)
            views = []
            for vi in scene.test_idx:
                _, Kc, V = scene.view(int(vi))
                idx = model.visible(Kc, V, scene.W, scene.H)
                r = stages(model, at, idx, Kc, V, scene.W, scene.H, a.n_warm, a.n_rep)
                views.append({"view": int(vi), "n_visible": int(idx.numel()), **r})
            s1 = syn(); u1 = gpu_util()
            f_dec = ref["decode"] / np.mean([s0["decode"]["mean"], s1["decode"]["mean"]])
            f_ren = ref["render"] / np.mean([s0["render"]["mean"], s1["render"]["mean"]])
            for v in views:
                v["dec_cal"] = v["decode"]["mean"] * f_dec; v["ren_cal"] = v["render"]["mean"] * f_ren
            res = {"scene": sc, "W": scene.W, "H": scene.H, "gpu_before": u0, "gpu_after": u1, "syn_before": s0, "syn_after": s1,
                   "f_dec": float(f_dec), "f_ren": float(f_ren), "views": views}
            with open(os.path.join(a.out, sc, "bench_trained.json"), "w") as fh:
                json.dump(res, fh, indent=1)
            rc = np.array([v["ren_cal"] for v in views]); nv = np.array([v["n_visible"] for v in views])
            allres["scenes"][sc] = {"W": scene.W, "H": scene.H, "f_dec": float(f_dec), "f_ren": float(f_ren), "n_views": len(views),
                                    "n_visible_mean": float(nv.mean()), "n_visible_max": int(nv.max()),
                                    "ren_cal_mean": float(rc.mean()), "ren_cal_p95": float(np.percentile(rc, 95)), "ren_cal_max": float(rc.max()),
                                    "dec_cal_mean": float(np.mean([v["dec_cal"] for v in views])),
                                    "n_gauss_mean": float(np.mean([v["n_gauss"] for v in views])),
                                    "syn_n_gauss": s0["n_gauss"]}
            print(sc, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in allres["scenes"][sc].items()}, u0, "|", u1, flush=True)
            del model, L, nets, scene
            torch.cuda.empty_cache()
    with open(os.path.join(a.out, "bench_trained.json"), "w") as fh:
        json.dump(allres, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="garden,bicycle,stump,room,counter,kitchen,bonsai,truck,train,playroom,drjohnson")
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--idle", default="results/bench_l40s_idle_final.json")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--n-rep", type=int, default=30)
    ap.add_argument("--n-warm-syn", type=int, default=20)
    ap.add_argument("--n-rep-syn", type=int, default=200)
    main(ap.parse_args())
