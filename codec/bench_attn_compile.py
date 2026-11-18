"""Eager FP16 vs torch.compile FP16 of the cross-attention concealment network (LayeredConcealer.predict incl. gathering),
garden _v2a_r1, whole-scene neighbour table, n random lost anchors; median of 10 after warm-up (compiled: warm-up on every size).
Output: results/rev6_attn_compile_vs_eager.json
usage: python bench_attn_compile.py
"""
import json, time, copy, torch, numpy as np
from eval_layered import load
from conceal_net import knn
from train_layered import LayeredConcealer
from eval_cold_matched import FP16Net
model, L, nets, c, ck = load("results", "garden", "_v2a_r1")
at = model.attributes(quantized=True); base, enh = [x.detach() for x in L.split(at)]
stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
nbr = knn(model.xyz, 16)
eager = FP16Net(nets["gen_fp"], "lite")
comp = FP16Net(nets["gen_fp"], "lite"); comp.net = torch.compile(comp.net, dynamic=True)
ce, cc = LayeredConcealer(model, L, eager, stats, nbr, ck["radius"]), LayeredConcealer(model, L, comp, stats, nbr, ck["radius"])
avail = torch.rand(model.xyz.shape[0], device="cuda") > 0.2
res = {}
with torch.no_grad():
    for n in (1000, 2000, 5000, 10000, 20000, 30000, 50000, 67000):
        lost = torch.randperm(model.xyz.shape[0], device="cuda")[:n]
        out = {}
        for name, cn in (("eager", ce), ("compiled", cc)):
            for _ in range(3): cn.predict(lost, avail, base, enh)
            torch.cuda.synchronize(); ts = []
            for _ in range(10):
                t0 = time.perf_counter(); cn.predict(lost, avail, base, enh); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
            out[name] = float(np.median(ts))
        res[n] = out; print(n, out, flush=True)
json.dump(res, open("results/rev6_attn_compile_vs_eager.json", "w"), indent=1)
