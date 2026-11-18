"""Validation and timing of knn_gpu.py on the cold-start lost sets of eval_cold_matched.py (same windows).

Mode dump  : for every scene, the anchors of each test view (model.visible) and the lost anchors of every window of
             results/<scene>/cold_matched_v2a_r1.json (runs T8R8:2.5, T8R8:10, T6R11:2.5; the failed streams of the
             trace in slots 0 .. S_A - 1, interleaved token assignment, exactly as eval_cold_matched.run_cold), checked
             against the recorded n_lost; saved to results/rev6_raw/knn_fast/<scene>.pt.
Mode check : exactness of knn_gpu.knn_view_global for every window against a float64 reference (scipy cKDTree in
             float64, exact search, then the candidates re-ranked by the float64 d2 of knn_gpu, ties to the lower
             index), plus torch float64 brute force (knn_gpu.knn_brute64) on the largest window of each scene and on
             256 random queries of every window; whole-scene tables (knn_table) against cKDTree for all anchors.
Mode time  : (run under the idle-GPU protocol) wall time from positions on the GPU to indices ready (synchronized),
             median of --reps after a warm-up, for every window (lost anchors among the view anchors), for all anchors of
             each view (the table eval_cold3.py --view-nbr builds) and for the whole-scene table.
Mode aggregate: results/rev6_knn_fast.json.
usage: python knn_gpu_eval.py --mode dump --scenes garden,...
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import torch

import knn_gpu
from streaming import Packetizer, mode_reliability_order, patterns_to_streams

SCENES = "garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson".split()
RUNS = ("T8R8:2.5", "T8R8:10", "T6R11:2.5")
RAW = "results/rev6_raw/knn_fast"


def cameras(scene, data):
    """K, world-to-camera matrices, W, H and the test views of a scene, as scene_data.Scene (no image decoding)."""
    import scene_data
    from PIL import Image

    def fake_imread(path):
        w, h = Image.open(path).size
        return np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (h, w, 3))
    old = scene_data.imageio.imread
    scene_data.imageio.imread = fake_imread
    try:
        sc = scene_data.Scene(*scene_data.scene_location(scene, data))
    finally:
        scene_data.imageio.imread = old
    return sc


def visible(xyz, K, V, W, H, margin=1.1):
    xc = (V[:3, :3] @ xyz.T + V[:3, 3:]).T
    z = xc[:, 2]
    u = K[0, 0] * xc[:, 0] / z.clamp_min(1e-6) + K[0, 2]
    v = K[1, 1] * xc[:, 1] / z.clamp_min(1e-6) + K[1, 2]
    mw, mh = (margin - 1) * W, (margin - 1) * H
    m = (z > 0.05) & (u > -mw) & (u < W + mw) & (v > -mh) & (v < H + mh)
    return torch.nonzero(m).squeeze(1)


def dump(a, scene):
    ck = torch.load(os.path.join(a.out, scene, f"layered{a.tag}.pt"), map_location="cuda")
    xyz = ck["state"]["xyz"].cuda()
    c = ck["net_cfg"]
    pk = Packetizer(xyz, ck["voxel"], ck["importance"].cuda(), G=c["G"])
    sc = cameras(scene, a.data)
    cm = json.load(open(os.path.join(a.out, scene, f"cold_matched{a.tag}.json")))
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    views, windows, bad = {}, [], 0
    with torch.no_grad():
        for vi in sc.test_idx:
            vi = int(vi)
            Kc = torch.from_numpy(sc.K[vi]).cuda(); V = torch.from_numpy(sc.viewmats[vi]).cuda()
            views[vi] = visible(xyz, Kc, V, sc.W, sc.H)
        for spec in RUNS:
            if spec not in cm["runs"]:
                continue
            cfg, tau = spec.split(":"); tau = float(tau); na = 4 if cfg == "T8R8" else 2
            cf = ms["configs"][cfg]; tx = cf["tx"]; M = len(tx)
            rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
            xs = [np.load(f)[cfg] for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz")))]
            pats = np.concatenate(xs)
            bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
            flat = bits.reshape(-1, M)
            anchor = min(itertools.combinations(range(M), na), key=lambda s_: int(flat[:, list(s_)].all(1).sum()))
            data_tx = [i for i in range(M) if i not in anchor]; Md = len(data_tx)
            data_streams = torch.tensor(sorted(rel.index(int(tx[i])) for i in data_tx), device="cuda")
            seen = set()
            for r in cm["runs"][spec]["rows"]:
                key = (r["view"], r["chain"], r["slot"])
                if key in seen:
                    continue
                seen.add(key)
                vi, ch, s, SA = r["view"], r["chain"], r["slot"], r["S_A"]
                idx = views[vi]
                assert idx.numel() == r["n_visible"], (scene, vi, idx.numel(), r["n_visible"])
                s_, tid = pk.token_ids(idx); T = int(tid[-1]) + 1
                slot_of = ((torch.arange(T, device="cuda") * SA) // T)[tid]
                fs = [torch.tensor([m in set(patterns_to_streams(int(pats[ch, s + t]), tx, rel)) for m in range(M)], device="cuda")
                      for t in range(SA)]
                tok_stream = data_streams[pk.assign(s_, tid, Md, "interleaved", None)][tid]
                fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                for t in range(SA):
                    fail |= (slot_of == t) & fs[t][tok_stream]
                lost = s_[fail]
                bad += int(lost.numel() != r["n_lost"])
                windows.append({"run": spec, "view": vi, "chain": ch, "slot": s, "n_visible": int(idx.numel()),
                                "n_lost": int(lost.numel()), "lost": lost.int().cpu()})
    os.makedirs(RAW, exist_ok=True)
    torch.save({"scene": scene, "xyz": xyz.float().cpu(), "radius": float(cm["radius"]), "voxel": float(ck["voxel"]), "kn": int(c["kn"]),
                "views": {k: v.int().cpu() for k, v in views.items()}, "windows": windows}, os.path.join(RAW, f"{scene}.pt"))
    print(f"{scene}: {len(views)} views, {len(windows)} windows, {sum(w['n_lost'] > 0 for w in windows)} with losses, "
          f"n_lost mismatches {bad}", flush=True)


# ---------------------------------------------------------------- reference
def d2_64(q, r):
    """float64 d2 with the bits of knn_gpu: (dx*dx + dy*dy) + dz*dz, dx = float64(q.x) - float64(r.x)."""
    q = q.astype(np.float32).astype(np.float64); r = r.astype(np.float32).astype(np.float64)
    dx = q[..., 0] - r[..., 0]; dy = q[..., 1] - r[..., 1]; dz = q[..., 2] - r[..., 2]
    return (dx * dx + dy * dy) + dz * dz


def ref_knn(ref, qpos, self_idx, k, workers=-1):
    """Exact k-NN (float64 KD-tree, then exact re-rank by (d2, index)); ref, qpos float32 numpy; returns (idx, d2, ties)
    with ties = the k-th and (k+1)-th neighbours at exactly the same float64 distance."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref.astype(np.float64))
    m = k + 9
    while True:
        dd, ii = tree.query(qpos.astype(np.float64), k=min(m, ref.shape[0]), workers=workers)
        ii = np.where(ii >= ref.shape[0], -1, ii)
        d2 = np.where(ii >= 0, d2_64(qpos[:, None, :], ref[np.clip(ii, 0, None)]), np.inf)
        d2 = np.where(ii == self_idx[:, None], np.inf, d2) if self_idx is not None else d2
        o = np.lexsort((ii, d2), axis=-1)
        d2 = np.take_along_axis(d2, o, 1); ii = np.take_along_axis(ii, o, 1)
        # every point outside the candidate set is at least as far as the KD-tree's last distance
        last = dd[:, -1] ** 2
        safe = (last * (1 - 1e-9) > d2[:, k - 1]) | (ii.shape[1] >= ref.shape[0])
        if safe.all():
            ties = d2[:, k - 1] == d2[:, k] if d2.shape[1] > k else np.zeros(len(d2), bool)
            return ii[:, :k], d2[:, :k], ties
        m *= 2


def check(a, scene):
    D = torch.load(os.path.join(RAW, f"{scene}.pt"))
    xyz = D["xyz"].cuda(); k = D["kn"]; xyz_np = D["xyz"].numpy()
    res = {"scene": scene, "k": k, "windows": []}
    t0 = time.time()
    # whole-scene table
    ti, td, fc = knn_gpu.knn_table(xyz, k, return_fallback=True)
    ti2, _, fc2 = knn_gpu.knn_table(xyz, k, h=D["radius"], return_fallback=True)
    ri, rd, ties = ref_knn(xyz_np, xyz_np, np.arange(len(xyz_np)), k)
    ti = ti.cpu().numpy()
    res["table"] = {"n": int(len(xyz_np)), "rows_identical": float((ti == ri).all(1).mean()),
                    "rows_identical_h_radius": float((ti2.cpu().numpy() == ri).all(1).mean()), "fallback_h_radius": fc2.tolist(),
                    "sets_identical": float((np.sort(ti, 1) == np.sort(ri, 1)).all(1).mean()), "boundary_ties": float(ties.mean()),
                    "fallback": fc.tolist()}
    print(scene, "table", res["table"], f"{time.time() - t0:.0f}s", flush=True)
    by_view = {}
    for w in D["windows"]:
        if w["n_lost"]:
            by_view.setdefault(w["view"], []).append(w)
    rng = np.random.default_rng(0)
    worst = max((w for ws in by_view.values() for w in ws), key=lambda w: w["n_lost"])
    for vi, ws in by_view.items():
        idx = D["views"][vi].long()
        ref = xyz_np[idx.numpy()]
        inv = np.full(len(xyz_np), -1, np.int64); inv[idx.numpy()] = np.arange(len(idx))
        allq = np.unique(np.concatenate([w["lost"].numpy() for w in ws]))
        ri, rd, ties = ref_knn(ref, xyz_np[allq], inv[allq], k)
        rmap = {int(g): j for j, g in enumerate(allq)}
        for w in ws:
            lost = w["lost"].long()
            gi, gd, fc = knn_gpu.knn_view_global(xyz, idx.cuda(), lost.cuda(), k, return_fallback=True)
            gi = gi.cpu().numpy()
            rows = np.array([rmap[int(g)] for g in lost.numpy()])
            rg = idx.numpy()[ri[rows]]
            same_rows = (gi == rg).all(1); same_sets = (np.sort(gi, 1) == np.sort(rg, 1)).all(1)
            gi2, _, fc2 = knn_gpu.knn_view_global(xyz, idx.cuda(), lost.cuda(), k, h=D["radius"], return_fallback=True)
            rec = {"run": w["run"], "view": vi, "chain": w["chain"], "slot": w["slot"], "n_lost": w["n_lost"], "n_visible": w["n_visible"],
                   "rows_identical": float(same_rows.mean()), "sets_identical": float(same_sets.mean()),
                   "rows_identical_h_radius": float((gi2.cpu().numpy() == rg).all(1).mean()),
                   "boundary_ties": float(ties[rows].mean()), "fallback": [a_ + b_ for a_, b_ in zip(fc.tolist(), fc2.tolist())]}
            # torch float64 brute force on 256 random queries (all queries of the largest window)
            sel = np.arange(len(lost)) if w is worst else rng.choice(len(lost), min(256, len(lost)), replace=False)
            sel_t = torch.from_numpy(sel)
            bi, bd = knn_gpu.knn_brute64(xyz[lost[sel_t].cuda()], xyz[idx.cuda()], k, torch.from_numpy(inv[lost[sel_t].numpy()]).cuda())
            bg = idx.numpy()[bi.cpu().numpy()]
            rec["brute64_n"] = int(len(sel)); rec["brute64_rows_identical"] = float((bg == gi[sel]).all(1).mean())
            res["windows"].append(rec)
    W = res["windows"]
    print(scene, f"{len(W)} windows, rows identical {np.mean([w['rows_identical'] for w in W]):.6f} "
          f"(min {min(w['rows_identical'] for w in W):.6f}), brute64 min {min(w['brute64_rows_identical'] for w in W):.6f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(RAW, f"{scene}_check.json"), "w") as fh:
        json.dump(res, fh)


# ---------------------------------------------------------------- timing
def wall(fn, reps):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts)), ts


def timing(a, scene):
    D = torch.load(os.path.join(RAW, f"{scene}.pt"))
    xyz = D["xyz"].cuda(); k = D["kn"]; h = a.h_radius * D["radius"] if a.h_radius > 0 else None
    knn_gpu.ext()
    res = {"scene": scene, "k": k, "reps": a.reps, "h": "auto" if h is None else f"{a.h_radius} x radius", "windows": [], "views": []}
    t0 = time.time()
    tab, ts = wall(lambda: knn_gpu.knn_table(xyz, k, h=h), a.reps)
    res["table_ms"] = tab; res["table_ms_all"] = ts; res["n_anchors"] = int(xyz.shape[0])
    views = {vi: v.cuda().long() for vi, v in D["views"].items()}
    for vi, idx in views.items():
        ms, _ = wall(lambda: knn_gpu.knn_view_global(xyz, idx, idx, k, h=h), a.reps)
        res["views"].append({"view": vi, "n_visible": int(idx.numel()), "all_view_anchors_ms": ms})
    for w in D["windows"]:
        if not w["n_lost"]:
            continue
        idx = views[w["view"]]; lost = w["lost"].cuda().long()
        ms, ts = wall(lambda: knn_gpu.knn_view_global(xyz, idx, lost, k, h=h), a.reps)
        res["windows"].append({"run": w["run"], "view": w["view"], "chain": w["chain"], "slot": w["slot"], "n_lost": w["n_lost"],
                               "n_visible": w["n_visible"], "ms": ms, "ms_all": ts})
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(RAW, f"{scene}_time{a.suffix}.json"), "w") as fh:
        json.dump(res, fh)
    W = [w["ms"] for w in res["windows"]]
    print(f"{scene}: table {tab:.3f} ms; windows {len(W)} p50 {np.percentile(W, 50):.3f} p99 {np.percentile(W, 99):.3f} "
          f"max {max(W):.3f} ms; {res['elapsed_s']:.0f}s", flush=True)


def dist(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())} if x.size else None


def aggregate(a):
    out = {"script": "knn_gpu_eval.py", "kernel": "knn_gpu.py", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "semantics": "k = 16 neighbours of every lost anchor among the anchors of its view (self excluded), exact under float64 "
                        "d2 = (dx*dx + dy*dy) + dz*dz of the FP32 positions, ties to the lower index; timing = wall clock from "
                        "positions on the GPU to indices ready (synchronized), median of the repetitions, idle GPU",
           "exactness": {}}
    C = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, "*_check.json")))]
    if C:
        W = [w for c in C for w in c["windows"]]
        out["exactness"] = {
            "scenes": [c["scene"] for c in C], "windows": len(W), "queries": int(sum(w["n_lost"] for w in W)),
            "rows_identical_fraction": float(sum(w["rows_identical"] * w["n_lost"] for w in W) / max(1, sum(w["n_lost"] for w in W))),
            "rows_identical_fraction_h_radius": float(sum(w["rows_identical_h_radius"] * w["n_lost"] for w in W) / max(1, sum(w["n_lost"] for w in W))),
            "sets_identical_fraction": float(sum(w["sets_identical"] * w["n_lost"] for w in W) / max(1, sum(w["n_lost"] for w in W))),
            "windows_all_rows_identical": int(sum(w["rows_identical"] == 1.0 for w in W)),
            "boundary_tie_fraction": float(sum(w["boundary_ties"] * w["n_lost"] for w in W) / max(1, sum(w["n_lost"] for w in W))),
            "brute64_queries": int(sum(w["brute64_n"] for w in W)),
            "brute64_rows_identical_min": float(min(w["brute64_rows_identical"] for w in W)),
            "exact_pass_queries": int(sum(w["fallback"][0] for w in W)), "brute_force_queries": int(sum(w["fallback"][1] for w in W)),
            "table": {c["scene"]: c["table"] for c in C}}
    for suffix, name in (("", "timing_h_radius"), ("_auto", "timing_auto_h")):
        T = {}
        for f in sorted(glob.glob(os.path.join(RAW, f"*_time{suffix}.json"))):
            t = json.load(open(f)); T[t["scene"]] = t
        if not T:
            continue
        allw = [w for t in T.values() for w in t["windows"]]
        tim = {"h": next(iter(T.values()))["h"], "reps": next(iter(T.values()))["reps"], "scenes": sorted(T)}
        for spec in RUNS:
            ws = [w for w in allw if w["run"] == spec]
            wc = max(ws, key=lambda w: w["n_lost"])
            sc_ = next(s for s, t in T.items() if wc in t["windows"])
            tim[spec] = {"knn_ms": dist([w["ms"] for w in ws]), "n_lost": dist([w["n_lost"] for w in ws]),
                         "worst_case": {"scene": sc_, "view": wc["view"], "n_lost": wc["n_lost"], "n_visible": wc["n_visible"], "ms": wc["ms"],
                                        "ms_all_reps": wc["ms_all"]},
                         "slowest_window_ms": max(w["ms"] for w in ws)}
        tim["all_windows"] = {"knn_ms": dist([w["ms"] for w in allw]), "n_lost": dist([w["n_lost"] for w in allw])}
        tim["all_view_anchors_ms"] = dist([v["all_view_anchors_ms"] for t in T.values() for v in t["views"]])
        tim["whole_scene_table_ms"] = {s: {"n_anchors": t["n_anchors"], "ms": t["table_ms"]} for s, t in T.items()}
        out[name] = tim
    with open(os.path.join(a.out, "rev6_knn_fast.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dump", choices=("dump", "check", "time", "aggregate"))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--h-radius", type=float, default=0.0, help="cell size in units of the median 16-NN distance (0: automatic)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.mode == "aggregate":
        aggregate(a)
    else:
        for s in a.scenes.split(","):
            {"dump": dump, "check": check, "time": timing}[a.mode](a, s)
