"""Edge-side latency of the concealment pipeline (CUDA events), independent of trained weight values.

Stages timed separately, then together:
  gather   : neighbourhood gathering for L lost anchors (16-NN table lookup, relative positions, normalisation)
  conceal  : LayeredConcealNet forward for L lost anchors (fp32, fp16; BitLinear weights are executed as dense
             fp16 matmuls here, so the ternary model has the cost of the fp16 model)
  decode   : anchor -> neural Gaussians (shared MLPs) for A_vis visible anchors
  render   : gsplat rasterization of the resulting Gaussians at W x H
For each stage: n_warm warm-up runs, then n_rep timed runs; mean, P50, P99, P99.9 in ms.
usage: python bench_latency.py --out results/bench_<gpu>.json --A-vis 55000 --lost 1000,5000,10000,20000,30000
"""
import argparse
import json
import time

import numpy as np
import torch

from anchor_model import AnchorModel
from conceal_net import EdgeConcealNet, LayeredConcealNet, LiteConcealNet


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
    return {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p99": float(np.percentile(t, 99)),
            "p999": float(np.percentile(t, 99.9)), "n": int(n_rep)}


def main(a):
    torch.manual_seed(0)
    dev = "cuda"
    A, Kn, bdim, edim = a.anchors, 16, a.bdim, a.edim
    xyz = torch.rand(A, 3, device=dev) * 10
    model = AnchorModel(xyz, 0.02, feat_dim=a.feat_dim, k=10).to(dev).eval()
    with torch.no_grad():
        model.feat.normal_(); model.offsets.normal_(0, 0.5); model.scaling.add_(torch.randn_like(model.scaling) * 0.1)
    nbr = torch.randint(0, A, (A, Kn), device=dev)
    base = torch.randn(A, bdim, device=dev); enh = torch.randn(A, edim, device=dev)
    avail = torch.rand(A, device=dev) > 0.2
    net = LayeredConcealNet(bdim, edim).to(dev).eval()
    net16 = LayeredConcealNet(bdim, edim).to(dev).half().eval()
    lite = LiteConcealNet(bdim, edim, d=a.lite_d, layers=a.lite_layers).to(dev).eval()
    lite16 = LiteConcealNet(bdim, edim, d=a.lite_d, layers=a.lite_layers).to(dev).half().eval()
    lite16c = torch.compile(lite16, mode="max-autotune-no-cudagraphs", dynamic=False) if a.compile else None
    edge16 = EdgeConcealNet(bdim, edim, d=a.lite_d, layers=a.lite_layers).to(dev).half().eval()
    with torch.no_grad():                        # encodings of all anchors, computed once on arrival and cached
        kv_table = edge16.encode(base.half(), enh.half(), avail)
    K = torch.tensor([[1100., 0, a.W / 2], [0, 1100., a.H / 2], [0, 0, 1]], device=dev)
    V = torch.eye(4, device=dev); V[2, 3] = 12.0
    vis = torch.randperm(A, device=dev)[:a.A_vis]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "anchors": A, "A_vis": a.A_vis, "bdim": bdim, "edim": edim, "feat_dim": a.feat_dim,
           "W": a.W, "H": a.H, "n_rep": a.n_rep, "stages": {}}
    with torch.no_grad():
        for L in [int(x) for x in a.lost.split(",")]:
            lost = vis[torch.randperm(vis.numel(), device=dev)[:min(L, vis.numel())]] if L <= vis.numel() else \
                torch.randint(0, A, (L,), device=dev)

            def gather():
                nb = nbr[lost]
                return (xyz[nb] - xyz[lost][:, None]), base[nb], enh[nb], avail[nb], base[lost]
            g = gather()
            g16 = tuple(x.half() if x.dtype == torch.float32 else x for x in g)
            res["stages"][f"gather_L{L}"] = timeit(gather, a.n_warm, a.n_rep)
            def chunked(m, args):            # same chunking as LayeredConcealer.predict (16384 anchors per call)
                return [m(*(x[i:i + a.chunk] for x in args)) for i in range(0, args[0].shape[0], a.chunk)]
            if not a.lite_only:
                res["stages"][f"conceal_fp32_L{L}"] = timeit(lambda: chunked(net, g), a.n_warm, a.n_rep)
            if not a.lite_only:
                res["stages"][f"conceal_fp16_L{L}"] = timeit(lambda: chunked(net16, g16), a.n_warm, a.n_rep)
            res["stages"][f"lite_fp32_L{L}"] = timeit(lambda: chunked(lite, g), a.n_warm, a.n_rep)
            res["stages"][f"lite_fp16_L{L}"] = timeit(lambda: chunked(lite16, g16), a.n_warm, a.n_rep)
            nb = nbr[lost]; rel16 = (xyz[nb] - xyz[lost][:, None]).half(); ob16 = base[lost].half()
            res["stages"][f"edge_fp16_L{L}"] = timeit(lambda: [edge16.forward_table(kv_table, nb[i:i + a.chunk], rel16[i:i + a.chunk], ob16[i:i + a.chunk]) for i in range(0, L, a.chunk)], a.n_warm, a.n_rep)
            if lite16c is not None:
                res["stages"][f"lite_fp16_compiled_L{L}"] = timeit(lambda: chunked(lite16c, g16), a.n_warm, a.n_rep)
            print(L, {k: round(v["mean"], 4) for k, v in res["stages"].items() if k.endswith(f"_L{L}")}, flush=True)
        newA = torch.randperm(A, device=dev)[:a.A_vis]
        res["stages"]["edge_encode_Avis"] = timeit(lambda: edge16.encode(base[newA].half(), enh[newA].half(), avail[newA]), a.n_warm, a.n_rep)
        at = {"feat": model.feat, "offsets": model.offsets, "scaling": model.scaling}
        cam = -V[:3, :3].T @ V[:3, 3]
        res["stages"]["decode"] = timeit(lambda: model.neural_gaussians(at, vis, cam), a.n_warm, a.n_rep)
        res["stages"]["render"] = timeit(lambda: model.render(at, vis, K, V, a.W, a.H), a.n_warm, a.n_rep)
        print({k: round(v["mean"], 4) for k, v in res["stages"].items() if k in ("decode", "render")}, flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=150_000)
    ap.add_argument("--A-vis", type=int, default=55_000)
    ap.add_argument("--lost", default="1000,5000,10000,20000,30000")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--bdim", type=int, default=18, help="base-layer size (12 feature channels + 6 scalings)")
    ap.add_argument("--edim", type=int, default=66, help="enhancement size (36 feature channels + 30 offsets)")
    ap.add_argument("--feat-dim", type=int, default=48)
    ap.add_argument("--lite-d", type=int, default=96)
    ap.add_argument("--lite-layers", type=int, default=2)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--lite-only", type=int, default=0)
    ap.add_argument("--W", type=int, default=1297)
    ap.add_argument("--H", type=int, default=840)
    ap.add_argument("--n-warm", type=int, default=50)
    ap.add_argument("--n-rep", type=int, default=2000)
    main(ap.parse_args())
