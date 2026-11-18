"""Scene-switch cost of the per-scene concealment networks on the edge GPU: load time, size and resident memory.

For every scene, the trained networks are exported once (untimed) to standalone FP16 state-dict files, as the edge would
store or receive them: attn (cross-attention concealment network of the paper, layered_v2a_r1.pt), edge (deployment
variant EdgeConcealNet, edge_v2a_r1.pt), mlp (MLP of two layers of width 256 trained on the frozen representation,
mlp_v2a_r1.pt) and mlp_joint (the jointly trained MLP, layered_v2m_r1.pt; garden, room, playroom).
Timed switch, in a running edge process with an initialized CUDA context, --reps times per scene and network:
  read   : file -> bytes; cold page cache (posix_fadvise DONTNEED on the file, residency checked with fincore) or warm
  deser  : torch.load from the bytes (CPU)
  build  : module construction and load_state_dict (CPU)
  h2d    : module to the GPU in FP16 (synchronized)
  first  : first inference on 1,000 and on 20,000 lost anchors (synchronized), and the same call again (second)
  reuse  : alternative path, weights copied into a resident module of the same architecture (state_dict copy)
switch = read + deser + build + h2d + first (1,000). "link" = deser + build + h2d + first, i.e. the network received as
bytes over the link. A fresh process (CUDA context creation included) is timed once per network type in a subprocess.
Per-scene state built at a switch: the neighbour table (16 nearest anchors of every anchor, conceal_net.knn) and, for
EdgeConcealNet, the table of cached keys/values of all anchors; both are timed and their memory is reported.
Memory: parameters, FP32/FP16 bytes and file size per network; torch.cuda.memory_allocated with N = 1 and N = 11
resident networks (and with their per-scene tables).
Output: --out-json (default results/rev6_switch.json)
usage: python bench_switch.py --reps 5
"""
import argparse
import gc
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from conceal_net import EdgeConcealNet, LiteConcealNet, MLPConcealNet, knn

SCENES = "garden,bicycle,stump,room,counter,kitchen,bonsai,truck,train,playroom,drjohnson"
ARCH = {"attn": LiteConcealNet, "edge": EdgeConcealNet, "mlp": MLPConcealNet, "mlp_joint": MLPConcealNet}


def sync():
    torch.cuda.synchronize()


def export(a, sc):
    """Standalone FP16 state dicts of the networks of a scene; returns {arch: (path, cfg, n_params, fp32_bytes)}."""
    os.makedirs(a.netdir, exist_ok=True)
    out = {}
    srcs = {"attn": (f"layered_v2a_r1.pt", "net_fp"), "edge": ("edge_v2a_r1.pt", "net_fp"), "mlp": ("mlp_v2a_r1.pt", "net_fp"),
            "mlp_joint": ("layered_v2m_r1.pt", "net_fp")}
    for arch, (fn, key) in srcs.items():
        p = os.path.join(a.out, sc, fn)
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["net_cfg"] if "net_cfg" in ck else {"bdim": 18, "edim": 66, **{k: ck["cfg"][k] for k in ("d", "layers")}}
        cfg = {"bdim": cfg.get("bdim", 18), "edim": cfg.get("edim", 66), "d": cfg["d"], "layers": cfg["layers"]}
        sd = ck[key]
        n_par = int(sum(v.numel() for v in sd.values()))
        dst = os.path.join(a.netdir, f"{sc}_{arch}.pt")
        torch.save({"cfg": cfg, "state": {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}}, dst)
        fd = os.open(dst, os.O_RDONLY); os.fsync(fd); os.close(fd)
        out[arch] = (dst, cfg, n_par, int(sum(v.numel() * v.element_size() for v in sd.values())))
        del ck
    return out


def inputs(arch, cfg, n, A=250_000):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g).half()
    if arch == "edge":
        return (torch.randint(0, A, (n, 16), device="cuda"), r(n, 16, 3), r(n, cfg["bdim"]))
    return (r(n, 16, 3), r(n, 16, cfg["bdim"]), r(n, 16, cfg["edim"]), torch.rand(n, 16, device="cuda") > 0.3, r(n, cfg["bdim"]))


def infer(arch, net, x, kv):
    with torch.no_grad():
        return net.forward_table(kv, *x) if arch == "edge" else net(*x)


def resident(path):
    try:
        out = subprocess.run(["fincore", "-b", "-n", "-o", "RES", path], capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError):
        return None


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def switch_once(arch, path, cfg, cold, x1, x20, kv):
    if cold:
        drop_cache(path)
    res0 = resident(path)
    t = {}
    t0 = time.perf_counter(); b = open(path, "rb").read(); t["read"] = time.perf_counter() - t0
    t0 = time.perf_counter(); ck = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False); t["deser"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    net = ARCH[arch](cfg["bdim"], cfg["edim"], d=cfg["d"], layers=cfg["layers"]); net.load_state_dict({k: v.float() for k, v in ck["state"].items()}); net.eval()
    t["build"] = time.perf_counter() - t0
    t0 = time.perf_counter(); net = net.half().cuda(); sync(); t["h2d"] = time.perf_counter() - t0
    t0 = time.perf_counter(); infer(arch, net, x1, kv); sync(); t["first_1k"] = time.perf_counter() - t0
    t0 = time.perf_counter(); infer(arch, net, x1, kv); sync(); t["second_1k"] = time.perf_counter() - t0
    t0 = time.perf_counter(); infer(arch, net, x20, kv); sync(); t["first_20k"] = time.perf_counter() - t0
    t0 = time.perf_counter(); infer(arch, net, x20, kv); sync(); t["second_20k"] = time.perf_counter() - t0
    t = {k: 1e3 * v for k, v in t.items()}
    t["switch"] = t["read"] + t["deser"] + t["build"] + t["h2d"] + t["first_1k"]
    t["link"] = t["switch"] - t["read"]
    t["resident_bytes_before_read"] = res0
    return t, net, ck


def reuse_once(arch, net, ck, x1, kv):
    t0 = time.perf_counter()
    with torch.no_grad():
        for k, v in net.state_dict().items():
            v.copy_(ck["state"][k].to("cuda", non_blocking=True))
    sync(); t = time.perf_counter() - t0
    t0 = time.perf_counter(); infer(arch, net, x1, kv); sync()
    return {"copy": 1e3 * t, "first_1k": 1e3 * (time.perf_counter() - t0)}


FRESH = r"""
import io, json, sys, time
t00 = time.perf_counter()
import torch
sys.path.insert(0, ".")
from bench_switch import ARCH, inputs, infer, drop_cache
arch, path = sys.argv[1], sys.argv[2]
t_imp = time.perf_counter() - t00
t0 = time.perf_counter(); torch.cuda.init(); torch.zeros(1, device="cuda"); torch.cuda.synchronize(); t_ctx = time.perf_counter() - t0
drop_cache(path)
t0 = time.perf_counter(); ck = torch.load(path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
net = ARCH[arch](cfg["bdim"], cfg["edim"], d=cfg["d"], layers=cfg["layers"]); net.load_state_dict({k: v.float() for k, v in ck["state"].items()})
net = net.eval().half().cuda(); torch.cuda.synchronize(); t_load = time.perf_counter() - t0
x1 = inputs(arch, cfg, 1000); kv = [torch.randn(250000, 2 * cfg["d"], device="cuda").half() for _ in range(cfg["layers"])] if arch == "edge" else None
torch.cuda.synchronize()
t0 = time.perf_counter(); infer(arch, net, x1, kv); torch.cuda.synchronize(); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); infer(arch, net, x1, kv); torch.cuda.synchronize(); t_second = time.perf_counter() - t0
print(json.dumps({"import_s": t_imp, "cuda_context_s": t_ctx, "load_to_gpu_ms": 1e3 * t_load, "first_1k_ms": 1e3 * t_first, "second_1k_ms": 1e3 * t_second}))
"""


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    return {"mean": float(v.mean()), "p50": float(np.median(v)), "max": float(v.max()), "min": float(v.min()), "n": int(v.size)} if v.size else {}


def main(a):
    from bench_e2e_frames import quiet
    log = []
    scenes = a.scenes.split(",")
    nets = {sc: export(a, sc) for sc in scenes}
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "reps": a.reps, "sizes": {}, "switch": {}, "fresh_process": {},
           "scene_state": {}, "memory": {}}
    for sc in scenes:
        for arch, (p, cfg, n_par, b32) in nets[sc].items():
            res["sizes"][f"{sc}/{arch}"] = {"params": n_par, "fp32_bytes": b32, "fp16_bytes": b32 // 2, "file_bytes": os.path.getsize(p), "cfg": cfg}
    torch.zeros(1, device="cuda"); sync()
    res["probe_start"] = quiet(log, a, "switch")
    # warm the kernels of every architecture once (a running edge process)
    kvs = {}
    for arch in ARCH:
        sc = next((s for s in scenes if arch in nets[s]), None)
        if sc is None:
            continue
        p, cfg, _, _ = nets[sc][arch]
        kvs[arch] = [torch.randn(250_000, 2 * cfg["d"], device="cuda").half() for _ in range(cfg["layers"])] if arch == "edge" else None
        switch_once(arch, p, cfg, False, inputs(arch, cfg, 1000), inputs(arch, cfg, 20000), kvs[arch])
    raw = {}
    for sc in scenes:
        for arch, (p, cfg, _, _) in nets[sc].items():
            x1, x20 = inputs(arch, cfg, 1000), inputs(arch, cfg, 20000)
            for cold in (True, False):
                for r in range(a.reps):
                    t, net, ck = switch_once(arch, p, cfg, cold, x1, x20, kvs[arch])
                    t.update({f"reuse_{k}": v for k, v in reuse_once(arch, net, ck, x1, kvs[arch]).items()})
                    raw.setdefault(f"{arch}/{'cold' if cold else 'warm'}", []).append({"scene": sc, **t})
                    del net
        print(sc, "switch done", flush=True)
    for k, rows in raw.items():
        res["switch"][k] = {f: stats([r[f] for r in rows]) for f in rows[0] if f != "scene"}
    res["switch_rows"] = raw
    # per-scene state built at a switch: neighbour table and EdgeConcealNet key/value table
    for sc in scenes:
        ck = torch.load(os.path.join(a.out, sc, "layered_v2a_r1.pt"), map_location="cpu", weights_only=False)
        xyz = ck["state"]["xyz"].float().cuda(); A = xyz.shape[0]
        ts = []
        for _ in range(a.reps):
            sync(); t0 = time.perf_counter(); nb = knn(xyz, 16); sync(); ts.append(1e3 * (time.perf_counter() - t0))
        p, cfg, _, _ = nets[sc]["edge"]
        e = EdgeConcealNet(18, 66, d=cfg["d"], layers=cfg["layers"]).cuda().half().eval()
        base, enh = torch.randn(A, 18, device="cuda").half(), torch.randn(A, 66, device="cuda").half(); rc = torch.ones(A, dtype=torch.bool, device="cuda")
        tk = []
        with torch.no_grad():
            e.encode(base, enh, rc); sync()
            for _ in range(a.reps):
                t0 = time.perf_counter(); kv = e.encode(base, enh, rc); sync(); tk.append(1e3 * (time.perf_counter() - t0))
        res["scene_state"][sc] = {"anchors": int(A), "knn_ms": stats(ts), "nbr_table_bytes": int(nb.numel() * nb.element_size()),
                                  "nbr_table_bytes_int32": int(nb.numel() * 4), "edge_kv_encode_ms": stats(tk),
                                  "edge_kv_table_bytes_fp16": int(sum(t.numel() * t.element_size() for t in kv))}
        del xyz, nb, e, base, enh, kv, ck
        torch.cuda.empty_cache()
    del x1, x20, kvs
    # resident memory for N = 1 and N = 11 networks (FP16), with and without the per-scene tables
    for arch in ARCH:
        have = [sc for sc in scenes if arch in nets[sc]]
        if not have:
            continue
        for N in (1, len(scenes)):
            gc.collect(); torch.cuda.empty_cache(); sync(); m0 = torch.cuda.memory_allocated()
            keep = []
            for i in range(N):
                sc = have[i % len(have)]; p, cfg, _, _ = nets[sc][arch]
                ck = torch.load(p, map_location="cpu", weights_only=False)
                n = ARCH[arch](cfg["bdim"], cfg["edim"], d=cfg["d"], layers=cfg["layers"]); n.load_state_dict({k: v.float() for k, v in ck["state"].items()})
                keep.append(n.half().cuda())
            sync(); m1 = torch.cuda.memory_allocated()
            res["memory"][f"{arch}/N{N}"] = {"bytes_networks": int(m1 - m0), "distinct_scenes": min(N, len(have)),
                                            "bytes_networks_plus_tables": int(m1 - m0 + sum(res["scene_state"][have[i % len(have)]]["nbr_table_bytes_int32"] +
                                                                                           (res["scene_state"][have[i % len(have)]]["edge_kv_table_bytes_fp16"] if arch == "edge" else 0)
                                                                                           for i in range(N)))}
            res["memory"][f"{arch}/N{N}"]["param_bytes"] = int(sum(q.numel() * q.element_size() for m in keep for q in m.parameters()))
            del keep, n, ck
            gc.collect()
    # fresh process: CUDA context creation and first switch
    for arch in ARCH:
        sc = next((s for s in scenes if arch in nets[s]), None)
        if sc is None:
            continue
        out = subprocess.run([sys.executable, "-c", FRESH, arch, nets[sc][arch][0]], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        try:
            res["fresh_process"][arch] = {"scene": sc, **json.loads(out.stdout.strip().splitlines()[-1])}
        except (IndexError, ValueError):
            res["fresh_process"][arch] = {"error": out.stderr[-2000:]}
    res["probe_end"] = quiet(log, a, "switch_end"); res["wait_log"] = log
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res["switch"].items():
        print(k, {f: round(v[f]["mean"], 3) for f in ("read", "deser", "build", "h2d", "first_1k", "second_1k", "first_20k", "switch", "link", "reuse_copy")})
    print(json.dumps(res["memory"])); print(json.dumps(res["fresh_process"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=SCENES)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--netdir", default="results/rev6_switch_nets")
    ap.add_argument("--out-json", default="results/rev6_switch.json")
    ap.add_argument("--no-probe", action="store_true")
    main(ap.parse_args())
