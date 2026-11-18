"""Edge-side latency of the concealment pipeline (CUDA events), independent of trained weight values.

Stages timed separately, then together:
  gather   : neighbourhood gathering for L lost anchors (16-NN table lookup, relative positions, normalisation)
  conceal  : LayeredConcealNet forward for L lost anchors (fp32, fp16; BitLinear weights are executed as dense
             fp16 matmuls here, so the ternary model has the cost of the fp16 model)
  lite     : LiteConcealNet (cross-attention concealment network), fp32, fp16 eager, fp16 torch.compile
  edge     : EdgeConcealNet from its table of cached encodings, fp16
  mlp      : MLPConcealNet (the network of train_layered.py --net mlp, tag _v2m_r1: width 256, 2 hidden layers, inputs
             own base layer, inverse-distance means of the neighbours' base and received enhancement layers, received
             fraction): mlp_fp16 = whole forward in FP16, eager and compiled; mlp_mixed = the deployed forward of
             bench_e2e_frames.py / eval_cold4.py (inverse-distance means in FP32 from FP32 inputs, MLP in FP16), eager and compiled
  decode   : anchor -> neural Gaussians (shared MLPs) for A_vis visible anchors
  render   : gsplat rasterization of the resulting Gaussians at W x H
For each stage: n_warm warm-up runs, then n_rep timed runs; mean, P50, P99, P99.9 in ms.
usage: python bench_latency.py --out results/bench_<gpu>.json --A-vis 55000 --lost 1000,5000,10000,20000,30000
       python bench_latency.py --out results/bench_latency_rev6.json --lost 5000,10000,20000,30000 --A-vis 110000 --anchors 250000 \
           --n-warm 50 --n-rep 2000 --compile 1 --mlp 1
       python bench_latency.py --out <rep>.json --lost 5000,10000,20000,30000 --A-vis 110000 --anchors 250000 --n-warm 50 \
           --n-rep 2000 --compile 1 --only-mlp 1 [--profile-mlp 1]   (MLP rows only, same harness)
"""
import argparse
import json
import time

import numpy as np
import torch

from anchor_model import AnchorModel
from conceal_net import EdgeConcealNet, LayeredConcealNet, LiteConcealNet, MLPConcealNet


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


def profile_call(fn, n=20):
    """Launch-bound check of one timed call: host enqueue time (perf_counter around the call, GPU idle before, no
    synchronization inside), GPU kernel time per call (torch.profiler, sum over kernels) and kernels per call."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    hs = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); hs.append(1e3 * (time.perf_counter() - t))
    torch.cuda.synchronize()
    out = {"host_enqueue_ms_p50": float(np.median(hs)), "host_enqueue_ms_max": float(np.max(hs))}
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        ka = [k for k in pr.key_averages() if getattr(k, "self_device_time_total", 0) > 0]
        out["gpu_kernel_ms_per_call"] = float(sum(k.self_device_time_total for k in ka)) / 1e3 / n
        out["kernels_per_call"] = float(sum(k.count for k in ka)) / n
        out["top_kernels"] = [(k.key[:90], round(k.self_device_time_total / 1e3 / n, 4), round(k.count / n, 2))
                              for k in sorted(ka, key=lambda k: -k.self_device_time_total)[:8]]
    except Exception as e:                       # profiler unavailable: keep the host-side numbers
        out["profiler_error"] = repr(e)
    return out


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
    mlp16 = MLPConcealNet(bdim, edim, d=a.mlp_d, layers=a.mlp_layers).to(dev).half().eval() if a.mlp else None
    mlp16c = torch.compile(mlp16, mode="max-autotune-no-cudagraphs", dynamic=False) if (a.mlp and a.compile) else None

    def mlp_mixed(rel, base_, enh_, recv, own_base):
        """Deployed forward (bench_e2e_frames.Edge.mlp_forward): inverse-distance means in FP32, MLP in FP16."""
        w = 1.0 / rel.norm(dim=-1).clamp_min(1e-6)
        wr = w * recv.to(w.dtype)
        has = (wr.sum(1, keepdim=True) > 0).to(w.dtype)
        enh_m = (wr[..., None] * enh_).sum(1) / wr.sum(1, keepdim=True).clamp_min(1e-12)
        base_m = (w[..., None] * base_).sum(1) / w.sum(1, keepdim=True)
        frac = recv.to(w.dtype).mean(1, keepdim=True)
        return mlp16.mlp(torch.cat([own_base, base_m, enh_m * has, frac, has], -1).half())
    mlp_mixed_c = torch.compile(mlp_mixed, mode="max-autotune-no-cudagraphs", dynamic=False) if (a.mlp and a.compile) else None
    if a.mlp:
        res_mlp = {"mlp_params": sum(p.numel() for p in mlp16.parameters()), "mlp_d": a.mlp_d, "mlp_layers": a.mlp_layers}
    with torch.no_grad():                        # encodings of all anchors, computed once on arrival and cached
        kv_table = edge16.encode(base.half(), enh.half(), avail)
    K = torch.tensor([[1100., 0, a.W / 2], [0, 1100., a.H / 2], [0, 0, 1]], device=dev)
    V = torch.eye(4, device=dev); V[2, 3] = 12.0
    vis = torch.randperm(A, device=dev)[:a.A_vis]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "anchors": A, "A_vis": a.A_vis, "bdim": bdim, "edim": edim, "feat_dim": a.feat_dim,
           "W": a.W, "H": a.H, "n_rep": a.n_rep, "n_warm": a.n_warm, "chunk": a.chunk, "stages": {}}
    if a.mlp:
        res.update(res_mlp)
    with torch.no_grad():
        for L in [int(x) for x in a.lost.split(",")]:
            lost = vis[torch.randperm(vis.numel(), device=dev)[:min(L, vis.numel())]] if L <= vis.numel() else \
                torch.randint(0, A, (L,), device=dev)

            def gather():
                nb = nbr[lost]
                return (xyz[nb] - xyz[lost][:, None]), base[nb], enh[nb], avail[nb], base[lost]
            g = gather()
            g16 = tuple(x.half() if x.dtype == torch.float32 else x for x in g)
            def chunked(m, args):            # same chunking as LayeredConcealer.predict (16384 anchors per call)
                return [m(*(x[i:i + a.chunk] for x in args)) for i in range(0, args[0].shape[0], a.chunk)]
            if a.only_mlp:                   # MLP rows only (repetitions of the MLP rows in the same harness)
                res["stages"][f"mlp_fp16_L{L}"] = timeit(lambda: chunked(mlp16, g16), a.n_warm, a.n_rep)
                res["stages"][f"mlp_mixed_L{L}"] = timeit(lambda: chunked(mlp_mixed, g), a.n_warm, a.n_rep)
                if mlp16c is not None:
                    res["stages"][f"mlp_fp16_compiled_L{L}"] = timeit(lambda: chunked(mlp16c, g16), a.n_warm, a.n_rep)
                    res["stages"][f"mlp_mixed_compiled_L{L}"] = timeit(lambda: chunked(mlp_mixed_c, g), a.n_warm, a.n_rep)
                if a.profile_mlp:
                    pf = res.setdefault("mlp_profile", {})
                    pf[f"mlp_fp16_L{L}"] = profile_call(lambda: chunked(mlp16, g16))
                    if mlp16c is not None:
                        pf[f"mlp_fp16_compiled_L{L}"] = profile_call(lambda: chunked(mlp16c, g16))
                print(L, {k: round(v["mean"], 4) for k, v in res["stages"].items() if k.endswith(f"_L{L}")}, flush=True)
                continue
            res["stages"][f"gather_L{L}"] = timeit(gather, a.n_warm, a.n_rep)
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
            if a.mlp:
                res["stages"][f"mlp_fp16_L{L}"] = timeit(lambda: chunked(mlp16, g16), a.n_warm, a.n_rep)
                res["stages"][f"mlp_mixed_L{L}"] = timeit(lambda: chunked(mlp_mixed, g), a.n_warm, a.n_rep)
                if mlp16c is not None:
                    res["stages"][f"mlp_fp16_compiled_L{L}"] = timeit(lambda: chunked(mlp16c, g16), a.n_warm, a.n_rep)
                    res["stages"][f"mlp_mixed_compiled_L{L}"] = timeit(lambda: chunked(mlp_mixed_c, g), a.n_warm, a.n_rep)
            print(L, {k: round(v["mean"], 4) for k, v in res["stages"].items() if k.endswith(f"_L{L}")}, flush=True)
        if a.only_mlp:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
            return
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
    ap.add_argument("--mlp", type=int, default=0, help="also time MLPConcealNet (FP16 and deployed mixed precision, eager and compiled)")
    ap.add_argument("--only-mlp", type=int, default=0, help="time only the MLP rows (implies --mlp 1)")
    ap.add_argument("--profile-mlp", type=int, default=0, help="with --only-mlp: host enqueue time and kernel time per call")
    ap.add_argument("--mlp-d", type=int, default=256)
    ap.add_argument("--mlp-layers", type=int, default=2)
    ap.add_argument("--W", type=int, default=1297)
    ap.add_argument("--H", type=int, default=840)
    ap.add_argument("--n-warm", type=int, default=50)
    ap.add_argument("--n-rep", type=int, default=2000)
    a_ = ap.parse_args()
    a_.mlp = a_.mlp or a_.only_mlp
    main(a_)
