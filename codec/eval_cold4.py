"""Cold start, version 4: the windows, loss patterns, strata, schemes and scoring of eval_cold3.py, with the jointly
trained MLP (tag _v2m_r1: representation and concealment network trained together) and every edge decision taken on
the real clock while the edge pipeline executes on the GPU.

Transport (as eval_cold3.py, same seeds, so the same windows): the anchor payload of a view in S_A slots on the n_A
anchor modes, the enhancement tokens interleaved over the M - n_A data streams in the same slots (token = 64 Morton-
consecutive anchors), the concealment network in slot S_A + 1; rep2 / MDS (rate 1/2) double the enhancement load over
S_R slots; ARQ resends the lost tokens (and a lost anchor payload or network) in slots S_A + 2 and S_A + 4. Stratum E =
some stream fails in the window (all schemes but MDS), stratum D = the MDS code fails or all anchor copies fail.
Deviation from eval_cold3.py: the edge requests a retransmission only if something is outstanding, and a retransmission
slot t is used only if (t + 1) + predicted decoding of the outstanding tokens + predicted rendering + margin fits the
deadline (a runtime decision; eval_cold3.py used the modelled t_dr); MDS / replication decoding costs nothing beyond
the rANS decoding of the tokens (systematic code).

Edge, executed and timed (one CUDA stream, idle warm GPU; wall clock from the start of slot 1, CUDA events per stage):
  reception: the rANS streams of the tokens received in slot u (real bitstreams of the view, encoded once on the CPU)
             are decoded on the GPU (range_code2 kernel) as soon as slot u ends (busy wait until (u + 1) ms on the
             wall clock), so decoding overlaps reception and only the real tail remains after the last slot;
  neighbours: k = 16 exact neighbours of the lost anchors among the anchors of the view (knn_gpu.knn_view_global,
             executed after the last data slot; with ARQ right after slot S_A for the anchors lost then);
  concealment, anytime: the lost anchors sorted by depth (nearest to the camera first) and concealed by the MLP in
             chunks of --chunk anchors (inverse-distance means in FP32, MLP in FP16), only after the network has
             arrived (slot S_A + 1); before every chunk the GPU is synchronized and the real elapsed time t is checked:
             the chunk runs only if t + T_chunk + T_IDW(anchors left after it) + T_render(view) + margin <= deadline;
             anchors not reached get IDW from the same neighbours if t + T_IDW + T_render + margin <= deadline,
             otherwise the view is rendered from what arrived (plus the anchors already concealed); if the network is
             lost, IDW only (same check); before the neighbour search, t + T_kNN + T_IDW + T_render + margin <=
             deadline decides between concealment and rendering what arrived (T_* = predictors below);
  render:    decoding of the rendered anchors into Gaussians and rasterization (model.render).
Predictors (per scene, calibrated on its TRAINING views, which are never test windows, with the pipeline of the test
windows: execute() with an unbounded deadline on S_A-slot receptions with 5, 12 and 25 % of the tokens lost, MLP and
IDW variants, after a warm-up): render = least-squares line in the number of visible anchors shifted by the 99th
percentile of its residuals (--q-render, "p99 of the render of that view class"); kNN (line in lost and visible
anchors), IDW and rANS decoding (lines in anchors) and the MLP chunk shifted by the --q-stage percentile (median by
default): stacking a p99 for every small stage double-counts their tails, and the margin covers what they leave.
Host jitter and GPU sharing: each view starts after a quietness probe (quiet_wait: no other GPU process, no stalled
operation, probe median within 1.3x of the idle baseline); after its windows the probe and nvidia-smi are checked
again and a disturbed view is executed again (--max-redo); a window that misses is re-executed after a probe.
A window fails if the anchor payload is lost, Psi < threshold against the loss-free rendering, or the executed total
exceeds the deadline (11.1 ms one frame, _r5; 22.2 ms two frames, _r6). Every window is executed once; a window that
misses the deadline is executed --rerun more times (after a quietness probe) and the run with the median total is kept.
Loss-free feasibility per view and schedule (S_A slots, S_R slots): median of 3 executions (3 more after a quietness
probe if the median misses the one-frame deadline); a scheme whose loss-free
schedule misses the deadline fails at every cold start of that view (always_fail), as in eval_cold3.py.
Output: <out>/<scene>/cold4<tag>_<cfg>_tau<tau>_r5.json and _r6.json (structure of the cold3 files plus executed
timing fields); --aggregate pools them into <out>/rev6_cold4_summary.json, with the comparison against the modelled
cold3<tag>_*_r5/_r6 rows of the same windows.
--exec-frames 1 executes and writes only the one-frame (_r5) evaluation; the ARQ transport of the two-frame deadline
is still drawn (not executed), so the random stream of every window, and with it every window, is that of the full run.
usage: python eval_cold4.py --scene garden --runs T8R8:2.5,T8R8:10,T6R11:2.5,T6R11:10 --margin <ms> --chunk 8192
       python eval_cold4.py --scene garden --margin 0 --chunk 8192 --exec-frames 1 --suffix _run2   (repeat of _r5)
       python eval_cold4.py --aggregate
"""
import argparse
import copy
import gc
import glob
import itertools
import json
import math
import os
import time

import numpy as np

try:
    import torch

    import eval_cold3 as ec
    import metrics
    from bench_e2e_frames import GROUP, TOK, Coder, gpu_apps, gpu_state, wait_idle
    from eval_layered import load
    from knn_gpu import knn_view_global
    from scene_data import Scene, scene_location
    from streaming import Packetizer, mode_reliability_order, patterns_to_streams
except ImportError:                                  # --aggregate needs only numpy and scipy
    torch = None
    TOK = 64


def _nograd(f):
    return torch.no_grad()(f) if torch is not None else f

SCHEMES = ("none", "conceal", "rep2", "rep2_conceal", "mds", "mds_conceal", "arq", "arq_conceal")
PSI_MAX = 99.0
CLASSES = ("complete", "full", "anytime", "idw", "drop", "anchor_lost", "always_fail")


def sync():
    torch.cuda.synchronize()


def wait_until(t0, ms):
    tgt = t0 + ms * 1e-3
    while time.perf_counter() < tgt:
        pass


def probe(seconds):
    """Synchronized 512^3 FP32 products for `seconds`: (median ms, max ms)."""
    x = torch.randn(512, 512, device="cuda"); ts = []; t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter(); x @ x; sync(); ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts)), float(np.max(ts))


def quiet_wait(log, tag, base_med=None, max_wait=600.0):
    """Waits until the host and the GPU are quiet: no other process on the GPU and 0.3 s of synchronized 512^3 FP32
    products in which no operation takes more than 3x the median + 0.5 ms (other jobs on this virtual machine stall the
    launching thread for tens to hundreds of ms at times) and the median is within 1.3x of the idle baseline (a GPU
    shared with another context or throttled is slower throughout). Returns the seconds waited."""
    t_w = time.time()
    while True:
        others = [x for x in gpu_apps() if int(x.split(",")[0]) != os.getpid()]
        med, mx = probe(0.3)
        if (not others and mx <= 3 * med + 0.5 and (base_med is None or med <= 1.3 * base_med)) or time.time() - t_w > max_wait:
            if time.time() - t_w > 1:
                log.append({"t": time.time(), "tag": tag, "waited_s": time.time() - t_w, "probe_med": med, "probe_max": mx})
            return time.time() - t_w
        time.sleep(2.0)


# ---------------------------------------------------------------- bitstreams of a view
class ViewStreams:
    """rANS bitstreams of both layers of the visible anchors (Morton order s_np, tokens of 64 anchors), encoded once;
    sub(lay, toks) gives the packet of a subset of tokens (the streams the edge received), without re-encoding."""

    def __init__(self, coder, s_np):
        self.s_np, self.n = s_np, len(s_np); self.T = (self.n + TOK - 1) // TOK
        self.coder = coder; self.full, self.np = {}, {}
        for lay in ("base", "enh"):
            pk = coder.encode(lay, s_np); self.full[lay] = pk
            self.np[lay] = (pk["data"].numpy(), pk["doff"].numpy(), coder.C[lay])

    def sub(self, lay, toks):
        toks = np.asarray(toks, np.int64)
        if toks.size == 0:
            return None
        if toks.size == self.T:
            return self.full[lay]
        data, doff, C = self.np[lay]; ng = C // GROUP
        streams = (toks[:, None] * ng + np.arange(ng)).ravel()
        st = doff[streams]; ln = doff[streams + 1] - st
        nd = np.concatenate([[0], np.cumsum(ln)]).astype(np.int64)
        gi = np.repeat(st - nd[:-1], ln) + np.arange(nd[-1])
        tn = np.minimum(TOK, self.n - toks * TOK)
        lens = np.repeat(tn * GROUP, ng)
        tok_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        pos = (toks[:, None] * TOK + np.arange(TOK)).ravel(); pos = pos[pos < self.n]
        ids = self.s_np[pos]
        pin = lambda x: torch.from_numpy(np.ascontiguousarray(x)).pin_memory()
        return {"lay": lay, "n": int(ids.size), "bits": 16 * int(nd[-1]), "data": pin(data[gi]), "doff": pin(nd), "tok_off": pin(tok_off),
                "sc0": pin(np.tile(np.arange(0, C, GROUP, dtype=np.int32), toks.size)), "snc": pin(np.full(toks.size * ng, GROUP, np.int32)),
                "ids": pin(ids.astype(np.int64))}


# ---------------------------------------------------------------- edge state of a scene
class Edge:
    def __init__(self, a, scene_name):
        model, L, nets, c, ck = load(a.out, scene_name, a.tag)
        assert ck.get("net_type") == "mlp", "eval_cold4 expects the jointly trained MLP (tag _v2m_r1)"
        self.model, self.L, self.c, self.ck = model, L, c, ck
        self.xyz = model.xyz.contiguous(); self.radius = float(ck["radius"]); self.kn = c["kn"]
        self.net = copy.deepcopy(nets["gen_fp"]).half().eval()
        with torch.no_grad():
            at = model.attributes(quantized=True); base, enh = L.split(at)
            self.at, self.base, self.enh = at, base.detach().contiguous(), enh.detach().contiguous()
            self.stats = (self.base.mean(0), self.base.std(0).clamp_min(0.02), self.enh.mean(0), self.enh.std(0).clamp_min(0.02))
        self.pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
        assert c["G"] == TOK
        self.coder = Coder(model, L, ck, a.threads)
        self.base_tab = torch.zeros_like(self.base); self.enh_tab = torch.zeros_like(self.enh)
        self.avail = torch.zeros(self.xyz.shape[0], dtype=torch.bool, device="cuda")
        self.A = self.xyz.shape[0]

    def mlp_chunk(self, ids, nb):
        bm, bs, em, es = self.stats
        rel = (self.xyz[nb] - self.xyz[ids][:, None]) / self.radius
        w = 1.0 / rel.norm(dim=-1).clamp_min(1e-6)
        recv = self.avail[nb]; wr = w * recv.to(w.dtype)
        has = (wr.sum(1, keepdim=True) > 0).to(w.dtype)
        enh_m = (wr[..., None] * ((self.enh_tab[nb] - em) / es)).sum(1) / wr.sum(1, keepdim=True).clamp_min(1e-12)
        base_m = (w[..., None] * ((self.base_tab[nb] - bm) / bs)).sum(1) / w.sum(1, keepdim=True)
        frac = recv.to(w.dtype).mean(1, keepdim=True)
        own = (self.base_tab[ids] - bm) / bs
        out = self.net.mlp(torch.cat([own, base_m, enh_m * has, frac, has], -1).half())
        self.enh_tab[ids] = out.float() * es + em

    def idw(self, ids, nb):
        em = self.stats[2]
        r = self.avail[nb].float()
        w = r / (self.xyz[nb] - self.xyz[ids][:, None]).norm(dim=-1).clamp_min(1e-6)
        m_ = (w[..., None] * self.enh_tab[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        self.enh_tab[ids] = torch.where((r.sum(1) > 0)[:, None], m_, em.expand_as(m_))

    def render(self, ridx, Kc, V, W, H):
        if ridx.numel() == 0:                  # nothing arrived (gsplat cannot rasterize an empty set): a black frame
            return torch.zeros(H, W, 3, device="cuda")
        return self.model.render(self.L.merge(self.base_tab[ridx], self.enh_tab[ridx]), ridx, Kc, V, W, H)[0]


# ---------------------------------------------------------------- predictors (calibrated on training views)
def envelope(x, y, cols=1):
    """Least-squares line (or plane) through (x, y), shifted up by its largest residual: an upper envelope."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float)[:, i] if np.ndim(x) > 1 else np.asarray(x, float) for i in range(cols)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    res = np.asarray(y, float) - X @ beta
    beta[0] += float(res.max())
    return [float(b) for b in beta]


class Pred:
    def __init__(self, d):
        self.d = d

    def ren(self, n):
        a, b = self.d["ren"]; return a + b * n

    def idw(self, n):
        if n <= 0:
            return 0.0
        a, b = self.d["idw"]; return a + b * n

    def knn(self, q, n_vis):
        a, b, c = self.d["knn"]; return a + b * q + c * n_vis

    def dec(self, n):
        if n <= 0:
            return 0.0
        a, b = self.d["dec"]; return a + b * n

    def chunk(self, c):
        return self.d["chunk_p99"]


class _Zero:
    """Predictors that never cut anything (calibration runs with an unbounded deadline)."""
    def ren(self, n): return 0.0
    def idw(self, n): return 0.0
    def knn(self, q, n): return 0.0
    def dec(self, n): return 0.0
    def chunk(self, c): return 0.0


def p99_line(x, y, q=99):
    """Line through (x, y) by least squares, shifted up by the q-th percentile of its residuals; x may be 2-D."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    out = [float(b) for b in beta]; out[0] += float(np.percentile(res, q))
    return out, [float(np.median(res)), float(np.percentile(res, 99)), float(res.max())]


@_nograd
def calibrate(a, E, scene, b_A):
    """Timing predictors of this scene, measured on its TRAINING views with the pipeline of the test windows (execute()
    with an unbounded deadline): every view is received over its S_A slots with a random share of its tokens lost
    (5, 12 or 25 %), then neighbour search, MLP chunks (network in slot S_A + 1) or IDW for all lost anchors (network
    lost), render. Predictor = least-squares line (render: in the visible anchors; kNN: in lost and visible anchors;
    IDW: in lost anchors; decoding: in anchors) shifted by the 99th percentile of its residuals; MLP chunk = p99."""
    W, H = scene.W, scene.H
    tr = [int(v) for v in scene.train_idx]
    cal = tr[::max(1, len(tr) // a.cal_views)][:a.cal_views]
    rng = np.random.default_rng(1)
    # warm-up of every stage (clocks, kernels, allocator), discarded
    for wi, vi in enumerate(cal[:a.warm_views] * 2):
        Kc = torch.from_numpy(scene.K[vi]).cuda(); V = torch.from_numpy(scene.viewmats[vi]).cuda()
        idx = E.model.visible(Kc, V, W, H); E.base_tab[idx] = E.base[idx]; E.enh_tab[idx] = E.enh[idx]
        s_, tid = E.pk.token_ids(idx); T = int(tid[-1]) + 1
        lost = torch.sort(s_[torch.from_numpy(rng.random(T) < 0.2).cuda()[tid]]).values
        E.avail[idx] = True; E.avail[lost] = False
        t_end = time.perf_counter() + 0.5
        while time.perf_counter() < t_end:
            E.render(idx, Kc, V, W, H); nb = knn_view_global(E.xyz, idx, lost, E.kn, h=E.radius); E.idw(lost, nb)
            for c in a.cal_chunks:
                E.mlp_chunk(lost[:c], nb[:c])
            sync()
        E.avail[idx] = False; E.base_tab[idx] = 0; E.enh_tab[idx] = 0
    sync()
    # chunk-size study (MLP chunk alone, synchronized before and after, as in the anytime loop): informative
    chunk_t = {}
    for vi in cal[:a.warm_views]:
        Kc = torch.from_numpy(scene.K[vi]).cuda(); V = torch.from_numpy(scene.viewmats[vi]).cuda()
        idx = E.model.visible(Kc, V, W, H); E.base_tab[idx] = E.base[idx]; E.enh_tab[idx] = E.enh[idx]
        s_, tid = E.pk.token_ids(idx); T = int(tid[-1]) + 1
        lost = torch.sort(s_[torch.from_numpy(rng.random(T) < 0.3).cuda()[tid]]).values; Q = int(lost.numel())
        E.avail[idx] = True; E.avail[lost] = False
        nb = knn_view_global(E.xyz, idx, lost, E.kn, h=E.radius)
        for c in a.cal_chunks:
            for r in range(20):
                if Q < c:
                    break
                o = int(rng.integers(0, Q - c + 1))
                sync(); t0 = time.perf_counter(); E.mlp_chunk(lost[o:o + c], nb[o:o + c]); sync()
                chunk_t.setdefault(c, []).append(1e3 * (time.perf_counter() - t0))
        E.avail[idx] = False; E.base_tab[idx] = 0; E.enh_tab[idx] = 0
    # pipeline calibration
    ren_pts, knn_pts, idw_pts, chunk_pts, dec_pts = [], [], [], [], []
    Z = _Zero()
    for j, vi in enumerate(cal):
        Kc = torch.from_numpy(scene.K[vi]).cuda(); V = torch.from_numpy(scene.viewmats[vi]).cuda()
        idx = E.model.visible(Kc, V, W, H); n = int(idx.numel())
        s_, tid = E.pk.token_ids(idx); VS = ViewStreams(E.coder, s_.cpu().numpy()); T = VS.T
        SA = max(1, math.ceil((n * b_A + ec.MLP_MBIT) / ec.R0))
        stok = (np.arange(T) * SA) // T
        X = dict(idx=idx, K=Kc, V=V, W=W, H=H, n=n)
        if j % 4 == 0:
            for frac in (0.1, 0.5, 1.0):
                toks = np.flatnonzero(rng.random(T) < frac) if frac < 1 else np.arange(T)
                if toks.size:
                    pb, pe = VS.sub("base", toks), VS.sub("enh", toks)
                    for r in range(3):
                        sync(); t0 = time.perf_counter(); E.coder.decode(pb, E.base_tab); E.coder.decode(pe, E.enh_tab); sync()
                        dec_pts.append((pb["n"], 1e3 * (time.perf_counter() - t0)))
        for frac in (0.05, 0.12, 0.25):
            lt = rng.random(T) < frac
            for net in (True, False):
                slots = []
                for u in range(SA):
                    bt = np.flatnonzero(stok == u); et = np.flatnonzero((stok == u) & ~lt)
                    slots.append((float(u + 1), [VS.sub("base", bt)], [VS.sub("enh", et)] if et.size else []))
                plan = {"slots": slots, "conceal": True, "net_ms": float(SA + 1) if net else None, "knn_after": None}
                o = execute(E, X, plan, 1e9, 0.0, Z, a.chunk); o.pop("img")
                ren_pts.append((n, o["ren"]))
                if o["knn_ms"] is not None:
                    knn_pts.append((o["n_lost_rx"], n, o["knn_ms"]))
                if o["idw_ms"] is not None:
                    idw_pts.append((o["n_lost_rx"], o["idw_ms"]))
                chunk_pts += [x for x in o["chunk_ms"] if x[0] == a.chunk]
    E.base_tab.zero_(); E.enh_tab.zero_(); E.avail.zero_()
    qs = a.q_stage
    ren, ren_r = p99_line([p[0] for p in ren_pts], [p[1] for p in ren_pts], a.q_render)
    knn_, knn_r = p99_line([[p[0], p[1]] for p in knn_pts], [p[2] for p in knn_pts], qs)
    idw_, idw_r = p99_line([p[0] for p in idw_pts], [p[1] for p in idw_pts], qs)
    dec_, dec_r = p99_line([p[0] for p in dec_pts], [p[1] for p in dec_pts], qs)
    ch = np.array([p[1] for p in chunk_pts]) if chunk_pts else np.array([np.percentile(chunk_t.get(a.chunk, [0.5]), 99)])
    return {"cal_views": cal, "q_render": a.q_render, "q_stage": qs, "ren": ren, "ren_resid_p50_p99_max": ren_r, "knn": knn_, "knn_resid": knn_r,
            "idw": idw_, "idw_resid": idw_r, "dec": dec_, "dec_resid": dec_r, "chunk": a.chunk, "chunk_p99": float(np.percentile(ch, qs)),
            "chunk_q99": float(np.percentile(ch, 99)),
            "chunk_pipeline": {"n": int(ch.size), "p50": float(np.median(ch)), "p99": float(np.percentile(ch, 99)), "max": float(ch.max())},
            "chunk_stats": {str(c): {"n": len(v), "p50": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(np.max(v)),
                                     "p99_per_1k_anchors": float(np.percentile(v, 99)) / c * 1000} for c, v in chunk_t.items()},
            "ren_points": ren_pts, "knn_points": knn_pts, "idw_points": idw_pts, "dec_points": dec_pts, "chunk_points": chunk_pts}


# ---------------------------------------------------------------- one execution of the edge
@_nograd
def execute(E, X, plan, TF, margin, P, chunk):
    """Runs the edge for one window on the real clock. plan: slots = [(arrival ms, [base packets], [enh packets])],
    conceal (bool), net_ms (arrival of the network, None if lost), knn_after (slot index after which the neighbour search
    starts, None = after the last slot). Returns a dict with the outcome and timings; the image in out['img']."""
    idx, Kc, V, W, H, n_vis = X["idx"], X["K"], X["V"], X["W"], X["H"], X["n"]
    E.base_tab[idx] = 0; E.enh_tab[idx] = 0; E.avail[idx] = False
    ev = {k: torch.cuda.Event(enable_timing=True) for k in ("t0", "dec", "knn", "conc", "ren")}
    nb_all = lost_k = None
    gc.disable()
    sync()
    t0 = time.perf_counter(); ev["t0"].record()
    slots = plan["slots"]; conc = plan["conceal"]
    for i, (arr, bp, ep) in enumerate(slots):
        wait_until(t0, arr)
        for p in bp:
            E.coder.decode(p, E.base_tab)
        for p in ep:
            ids = E.coder.decode(p, E.enh_tab); E.avail[ids] = True
        if conc and plan.get("knn_after") == i:
            lost_k = idx[~E.avail[idx]]
            if lost_k.numel():
                nb_all = knn_view_global(E.xyz, idx, lost_k, E.kn, h=E.radius)
    ev["dec"].record()
    lost = idx[~E.avail[idx]]; Q = int(lost.numel())
    out = {"n_lost_rx": Q, "n_net": 0, "variant": "complete", "pred_total": None, "t_data_ms": slots[-1][0] if slots else 0.0,
           "knn_ms": None, "idw_ms": None, "chunk_ms": []}
    ridx = idx
    if Q and not conc:
        ridx = idx[E.avail[idx]]; out["variant"] = "drop"
    elif Q:
        end = TF - margin - P.ren(n_vis)
        t = 1e3 * (time.perf_counter() - t0)
        if nb_all is None:
            if t + P.knn(Q, n_vis) + P.idw(Q) > end:
                ridx = idx[E.avail[idx]]; out["variant"] = "drop"; out["pred_total"] = t + P.ren(n_vis)
            else:
                nb_all = knn_view_global(E.xyz, idx, lost, E.kn, h=E.radius); lost_k = lost
                sync(); out["knn_ms"] = 1e3 * (time.perf_counter() - t0) - t
        ev["knn"].record()
        if out["variant"] != "drop":
            nb = nb_all if lost_k.numel() == Q else nb_all[torch.searchsorted(lost_k, lost)]
            depth = (V[2, :3] @ E.xyz[lost].T + V[2, 3])
            o = torch.argsort(depth); lo_, nbo = lost[o], nb[o]
            done = 0
            if plan["net_ms"] is not None:
                if 1e3 * (time.perf_counter() - t0) < plan["net_ms"]:
                    wait_until(t0, plan["net_ms"])
                t_c = None
                while done < Q:
                    c = min(chunk, Q - done)
                    sync(); t = 1e3 * (time.perf_counter() - t0)
                    if t_c is not None:
                        out["chunk_ms"].append((c_prev, t - t_c))
                    if t + P.chunk(c) + P.idw(Q - done - c) > end:
                        t_c = None; break
                    out["pred_total"] = t + P.chunk(c) + P.idw(Q - done - c) + P.ren(n_vis)
                    t_c, c_prev = t, c
                    E.mlp_chunk(lo_[done:done + c], nbo[done:done + c]); done += c
            rest = Q - done; out["n_net"] = done
            if rest == 0:
                out["variant"] = "full"
                if plan["net_ms"] is not None and t_c is not None:
                    sync(); out["chunk_ms"].append((c_prev, 1e3 * (time.perf_counter() - t0) - t_c))
            else:
                sync(); t = 1e3 * (time.perf_counter() - t0)
                if t + P.idw(rest) <= end:
                    E.idw(lo_[done:], nbo[done:]); out["variant"] = "anytime" if done else "idw"
                    out["pred_total"] = t + P.idw(rest) + P.ren(n_vis)
                    sync(); out["idw_ms"] = 1e3 * (time.perf_counter() - t0) - t
                else:
                    ridx = torch.cat([idx[E.avail[idx]], lo_[:done]]); out["variant"] = "drop"
                    out["pred_total"] = t + P.ren(n_vis)
    else:
        ev["knn"].record()
    ev["conc"].record()
    img = E.render(ridx, Kc, V, W, H)
    ev["ren"].record(); sync()
    total = 1e3 * (time.perf_counter() - t0)
    gc.enable()
    el = lambda a_, b_: float(ev[a_].elapsed_time(ev[b_]))
    out.update(total=total, gpu_total=el("t0", "ren"), t_dec_end=el("t0", "dec"), t_conc_end=el("t0", "conc"), ren=el("conc", "ren"),
               conc=(el("dec", "conc") if Q and conc else 0.0), n_rendered=int(ridx.numel()), img=img)
    return out


# ---------------------------------------------------------------- transport of a window (eval_cold3.py semantics)
def transport(sch, ctx, prng, rt_ok):
    """Loss outcome of scheme sch in a window: anchor-level fail mask (Morton order), token arrival slots of the
    enhancement and anchor payload, network arrival slot, anchor/network loss and the last slot used."""
    SA, SR, St, fs, aset, nfd, nfa = ctx["SA"], ctx["SR"], ctx["St"], ctx["fs"], ctx["aset"], ctx["nfd"], ctx["nfa"]
    tok_stream, tok_all, slot_of, slot_rep, tid = ctx["tok_stream"], ctx["tok_all"], ctx["slot_of"], ctx["slot_rep"], ctx["tid"]
    M, Md, pair_of, pair_all = ctx["M"], ctx["Md"], ctx["pair_of"], ctx["pair_all"]
    dev = tok_stream.device
    anchor_lost = any(aset[:SA])
    if sch.startswith("rep2"):
        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
        for t in range(SR):
            po_, ts_ = (pair_of, tok_stream) if t < SA else (pair_all, tok_all)
            pf = torch.tensor([(m in po_) and (m in St[t]) and (po_[m] in St[t]) for m in range(M)], device=dev)
            fail |= (slot_rep == t) & pf[ts_]
        arr = slot_rep.clone(); last = SR
    elif sch.startswith("mds"):
        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
        for t in range(SR):
            if (nfd[t] > Md // 2) if t < SA else (nfa[t] > M // 2):
                fail |= (slot_rep == t) & fs[t][tok_stream if t < SA else tok_all]
        arr = slot_rep.clone(); last = SR
    else:
        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
        for t in range(SA):
            fail |= (slot_of == t) & fs[t][tok_stream]
        arr = slot_of.clone(); last = SA
    conc = sch.endswith("conceal")
    net_lost = aset[SA] if conc else False
    net_slot = SA if (conc and not net_lost) else None
    base_arr = slot_of.clone()
    lost_base_slots = [t for t in range(SA) if aset[t]]
    used = []
    if sch.startswith("arq"):
        for t in (SA + 1, SA + 3):
            outstanding = anchor_lost or net_lost or bool(fail.any())
            if not outstanding:
                break
            n_out = int(fail.sum()) + (int(sum(int((slot_of == u).sum()) for u in lost_base_slots)) if anchor_lost else 0)
            if not rt_ok(t, n_out):
                continue
            used.append(t)
            if anchor_lost and not aset[t]:
                anchor_lost = False
                for u in lost_base_slots:
                    base_arr[slot_of == u] = t
            if net_lost and not aset[t]:
                net_lost = False; net_slot = t
            if fail.any():
                ok_ = [m for m in range(M) if m not in St[t - 1]] or list(range(M))
                ut, inv = torch.unique(tid[fail], return_inverse=True)
                ns = torch.as_tensor(np.asarray(ok_)[prng.integers(0, len(ok_), int(ut.numel()))], device=dev)[inv]
                before = fail.clone()
                fail[before] = fs[t][ns]
                arr[before & ~fail] = t
            last = t + 1
    arr[fail] = -1
    return dict(fail=fail, arr=arr, base_arr=base_arr, anchor_lost=bool(anchor_lost), net_lost=bool(net_lost), net_slot=net_slot,
                last=last, rt_used=[t + 1 for t in used])


def make_plan(VS, tr, conc, arq):
    """Packets per arrival slot (token level; anchors of a token share its fate)."""
    arr_t = tr["arr"][::TOK].cpu().numpy(); barr_t = tr["base_arr"][::TOK].cpu().numpy()
    slots = []
    for u in range(int(tr["last"])):
        bt = np.flatnonzero(barr_t == u); et = np.flatnonzero(arr_t == u)
        if bt.size == 0 and et.size == 0:
            continue
        slots.append((float(u + 1), [VS.sub("base", bt)] if bt.size else [], [VS.sub("enh", et)] if et.size else []))
    net_ms = float(tr["net_slot"] + 1) if (conc and tr["net_slot"] is not None) else None
    knn_after = None
    if conc and arq and tr["rt_used"]:
        SA_ms = tr["SA"]
        knn_after = max([i for i, s in enumerate(slots) if s[0] <= SA_ms], default=None)
    return {"slots": slots, "conceal": conc, "net_ms": net_ms, "knn_after": knn_after}


def free_plan(VS, slot_tok_A, nslots):
    slots = []
    for u in range(nslots):
        bt = np.flatnonzero(slot_tok_A[0] == u); et = np.flatnonzero(slot_tok_A[1] == u)
        if bt.size or et.size:
            slots.append((float(u + 1), [VS.sub("base", bt)] if bt.size else [], [VS.sub("enh", et)] if et.size else []))
    return {"slots": slots, "conceal": False, "net_ms": None, "knn_after": None}


# ---------------------------------------------------------------- one scene
def run_scene(a):
    torch.backends.cudnn.benchmark = False
    scene = Scene(*scene_location(a.scene, a.data))
    W, H = scene.W, scene.H
    E = Edge(a, a.scene)
    model, ck = E.model, E.ck
    mt = json.load(open(os.path.join(a.out, a.scene, f"metrics_layered{a.tag}.json")))
    rc_path = os.path.join(a.out, a.scene, f"rangecode{a.tag}.json")
    base_bits = (json.load(open(rc_path))["layers"]["base"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.012 * mt["rate_bits_per_anchor"]["base_layer"])
    b_A = mt["rate_bits_per_anchor"]["position_fixed"] + base_bits
    b_E = (json.load(open(rc_path))["layers"]["enhancement"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.013 * mt["rate_bits_per_anchor"]["enh_layer"])
    busy = []
    waited = wait_idle({os.getpid()}, busy) if not a.no_wait else 0.0
    # warm-up and calibration on the training views
    t_cal = time.time()
    P_d = calibrate(a, E, scene, b_A); P = Pred(P_d)
    base_med = min(probe(0.5)[0] for _ in range(3))           # idle baseline of the quietness probe
    print(f"calibration {time.time() - t_cal:.0f}s: ren {P_d['ren']}, idw {P_d['idw']}, knn {P_d['knn']}, dec {P_d['dec']}, chunk p99 {P_d['chunk_p99']:.3f}", flush=True)
    test = [int(v) for v in scene.test_idx][:a.max_views] if a.max_views else [int(v) for v in scene.test_idx]
    # loss-free rendering, metrics, visible sets and bitstreams of the test views (outside the timed region)
    VW = {}
    with torch.no_grad():
        for vi in test:
            gt, Kc, V = scene.view(vi)
            idx = model.visible(Kc, V, W, H)
            full = model.render(E.at, idx, Kc, V, W, H)[0]
            s_, tid = E.pk.token_ids(idx)
            VW[vi] = dict(gt=gt, K=Kc, V=V, idx=idx, full=full, fm=metrics.all_metrics(full, gt), s_=s_, tid=tid,
                          VS=ViewStreams(E.coder, s_.cpu().numpy()))
    for spec in a.runs.split(","):
        cfg, tau = spec.split(":"); tau = float(tau); na = 4 if cfg == "T8R8" else 2
        ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
        cf = ms["configs"][cfg]; tx = cf["tx"]; M = len(tx)
        rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
        xs = []
        for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz"))):
            z = np.load(f); assert [int(x) for x in z[f"{cfg}_tx"]] == [int(x) for x in tx]; xs.append(z[cfg])
        pats = np.concatenate(xs)
        bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
        flat = bits.reshape(-1, M)
        jf = flat.T.astype(np.int64) @ flat.astype(np.int64)
        anchor = min(itertools.combinations(range(M), na), key=lambda s: int(flat[:, list(s)].all(1).sum()))
        anchor_streams = sorted(rel.index(int(tx[i])) for i in anchor)
        data_tx = [i for i in range(M) if i not in anchor]; Md = len(data_tx)
        data_streams = torch.tensor(sorted(rel.index(int(tx[i])) for i in data_tx), device="cuda")
        pairs = ec.pairing(data_tx, [], 0, jf)[1]
        pair_of = {}
        for p_, q_ in pairs:
            sp, sq = rel.index(int(tx[p_])), rel.index(int(tx[q_])); pair_of[sp] = sq; pair_of[sq] = sp
        pair_all = {}
        for p_, q_ in ec.pairing(list(range(M)), [], 0, jf)[1]:
            sp, sq = rel.index(int(tx[p_])), rel.index(int(tx[q_])); pair_all[sp] = sq; pair_all[sq] = sp
        all_streams = torch.arange(M, device="cuda")
        anyf = bits.any(-1); Kd = bits[..., data_tx].sum(-1); Ka = bits.sum(-1); alost = bits[..., list(anchor)].all(-1)
        strata = {}

        def strata_for(SA, SR):
            if (SA, SR) not in strata:
                WE = max(SA + 4, SR); nW = anyf.shape[1] - WE - 1
                Em = np.zeros((anyf.shape[0], nW), bool); D = np.zeros_like(Em)
                for t in range(WE):
                    Em |= anyf[:, t:nW + t]
                for t in range(SR):
                    D |= (Kd[:, t:nW + t] > Md // 2) if t < SA else (Ka[:, t:nW + t] > M // 2)
                for t in range(SA):
                    D |= alost[:, t:nW + t]
                strata[(SA, SR)] = dict(E=np.flatnonzero(Em.ravel()), D=np.flatnonzero(D.ravel()), pE=float(Em.mean()), pD=float(D.mean()), nW=nW, WE=WE)
            return strata[(SA, SR)]

        rng = np.random.default_rng(a.seed)
        res_f = {fr: {"views": [], "rows": []} for fr in a.frames}
        reexec = {fr: [] for fr in a.frames}
        t_run = time.time()
        for vi in [int(v) for v in scene.test_idx]:
            n_vis_ = None
            gt_needed = vi in VW
            if not gt_needed:                    # keep the random stream of eval_cold3.py for views not evaluated (--max-views)
                Kc = torch.from_numpy(scene.K[vi]).cuda(); V = torch.from_numpy(scene.viewmats[vi]).cuda()
                n_vis_ = int(model.visible(Kc, V, W, H).numel())
            else:
                n_vis_ = int(VW[vi]["idx"].numel())
            n_vis = n_vis_
            SA = max(1, math.ceil((n_vis * b_A + ec.MLP_MBIT) / ec.R0)); L2 = 2 * n_vis * b_E
            SR = SA + max(0, math.ceil((L2 - Md * ec.R0 * SA) / (M * ec.R0)))
            st = strata_for(SA, SR)
            evs = [("E", int(i)) for i in rng.choice(st["E"], min(a.n_ev, st["E"].size), replace=False)] + \
                  [("D", int(i)) for i in rng.choice(st["D"], min(a.n_mds, st["D"].size), replace=False)]
            if not gt_needed:
                continue
            Vw = VW[vi]; idx, s_, tid, VS = Vw["idx"], Vw["s_"], Vw["tid"], Vw["VS"]
            n0 = {fr: (len(res_f[fr]["views"]), len(res_f[fr]["rows"]), len(reexec[fr])) for fr in a.frames}
            for attempt in range(a.max_redo + 1):        # a view disturbed by another GPU process or a slowed GPU is executed again
                for fr in a.frames:
                    del res_f[fr]["views"][n0[fr][0]:]; del res_f[fr]["rows"][n0[fr][1]:]; del reexec[fr][n0[fr][2]:]
                if not a.no_wait and [x for x in gpu_apps() if int(x.split(",")[0]) != os.getpid()]:
                    waited += wait_idle({os.getpid()}, busy)          # another process appeared: wait, then warm up again
                    for _ in range(5):
                        model.render(E.at, idx, Vw["K"], Vw["V"], W, H)
                    sync()
                if not a.no_wait:
                    quiet_wait(busy, f"view {cfg} {tau:g} {vi}", base_med)
                X = dict(idx=idx, K=Vw["K"], V=Vw["V"], W=W, H=H, n=n_vis)
                T = int(tid[-1]) + 1
                ar = torch.arange(T, device="cuda")
                slot_of = ((ar * SA) // T)[tid]
                capc = torch.tensor([Md * ec.R0 if t < SA else M * ec.R0 for t in range(SR)], device="cuda", dtype=torch.float64).cumsum(0)
                pos_ = (ar.double() + 0.5) / T * min(L2, float(capc[-1]))
                slot_rep = torch.searchsorted(capc, pos_).clamp_max(SR - 1)[tid]
                # loss-free execution of both schedules (median of 3)
                sA = slot_of[::TOK].cpu().numpy(); sR = slot_rep[::TOK].cpu().numpy()
                free = {}
                for nm, (bt, et, ns) in (("A", (sA, sA, SA)), ("R", (sA, sR, SR))):
                    pl = free_plan(VS, (bt, et), ns)
                    tt = [execute(E, X, pl, 1e9, 0.0, P, a.chunk) for _ in range(a.free_reps)]
                    tt_ = sorted(tt, key=lambda o: o["total"])[len(tt) // 2]
                    if tt_["total"] > 11.1 * min(a.frames) * a.deadline_scale:      # a view that looks undeliverable: timed again in quiet conditions
                        quiet_wait(busy, f"free {cfg} {tau:g} {vi}", base_med)
                        tt = [execute(E, X, pl, 1e9, 0.0, P, a.chunk) for _ in range(a.free_reps)]
                        tt_ = sorted(tt, key=lambda o: o["total"])[len(tt) // 2]
                    free[nm] = {k: tt_[k] for k in ("total", "t_dec_end", "ren", "gpu_total")}
                    free[nm]["totals"] = [o["total"] for o in tt]
                    if nm == "A":
                        psi_free = min(metrics.psnr(tt_["img"].clamp(0, 1), Vw["full"].clamp(0, 1)), PSI_MAX)
                for fr in a.frames:
                    TF = 11.1 * fr * a.deadline_scale
                    always = {k: free["R" if k.startswith(("rep2", "mds")) else "A"]["total"] > TF for k in SCHEMES}
                    res_f[fr]["views"].append({"view": vi, "n_visible": n_vis, "S_A": SA, "S_R": SR, "pE": st["pE"], "pD": st["pD"],
                                               "t_dr": free["A"]["total"] - SA, "free_A": free["A"], "free_R": free["R"], "psi_free": psi_free,
                                               "ren_pred": P.ren(n_vis), "always_fail": {k: bool(v) for k, v in always.items()}})
                for wn, (kind, w) in enumerate(evs):
                    ch, s = divmod(w, st["nW"])
                    prng = np.random.default_rng(vi * 7919 + ch * 251 + s)
                    WE = st["WE"]
                    St = [set(patterns_to_streams(int(pats[ch, s + t]), tx, rel)) for t in range(WE + 1)]
                    ctx = dict(SA=SA, SR=SR, St=St, fs=[torch.tensor([m in St[t] for m in range(M)], device="cuda") for t in range(WE + 1)],
                               aset=[all(x in St[t] for x in anchor_streams) for t in range(WE + 1)],
                               nfd=[sum(int(m) in St[t] for m in data_streams.tolist()) for t in range(WE + 1)], nfa=[len(St[t]) for t in range(WE + 1)],
                               tok_stream=data_streams[E.pk.assign(s_, tid, Md, "interleaved", prng)][tid],
                               tok_all=all_streams[E.pk.assign(s_, tid, M, "interleaved", prng)][tid],
                               slot_of=slot_of, slot_rep=slot_rep, tid=tid, M=M, Md=Md, pair_of=pair_of, pair_all=pair_all)
                    schs = [x for x in SCHEMES if not x.startswith("mds")] if kind == "E" else [x for x in SCHEMES if x.startswith("mds")]
                    for sch in schs:
                        trs = {}
                        for fr in a.frames:                  # ARQ decisions depend on the deadline; the random draws are replayed per deadline
                            TF = 11.1 * fr * a.deadline_scale
                            st_ = prng.bit_generator.state
                            rt_ok = (lambda TF_: (lambda t, n_out: (t + 1) + P.dec(n_out) + P.ren(n_vis) + a.margin <= TF_))(TF)
                            trs[fr] = transport(sch, ctx, prng, rt_ok)
                            if fr != a.frames[-1]:
                                prng.bit_generator.state = st_
                        for fr in a.exec_frames:
                            TF = 11.1 * fr * a.deadline_scale
                            tr = trs[fr]; tr["SA"] = SA
                            always = res_f[fr]["views"][-1]["always_fail"][sch]
                            n_lost = int(tr["fail"].sum())
                            rec = {"view": vi, "stratum": kind, "win": wn, "chain": int(ch), "slot": int(s), "scheme": sch, "S_A": SA, "n_visible": n_vis,
                                   "variant": "always_fail" if always else "complete", "n_net": 0, "n_lost": n_lost, "anchor_lost": tr["anchor_lost"],
                                   "net_lost": tr["net_lost"], "deadline_miss": False, "last_slot": int(tr["last"]), "rt_used": tr["rt_used"]}
                            if always:
                                rec.update(deadline_miss=True, psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); res_f[fr]["rows"].append(rec); continue
                            if tr["anchor_lost"]:
                                rec.update(variant="anchor_lost", psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); res_f[fr]["rows"].append(rec); continue
                            conc = sch.endswith("conceal")
                            plan = make_plan(VS, tr, conc, sch.startswith("arq"))
                            o = execute(E, X, plan, TF, a.margin, P, a.chunk)
                            runs = [o["total"]]
                            if o["total"] > TF and a.rerun > 0:
                                quiet_wait(busy, f"rerun {cfg} {tau:g} {vi} {wn} {sch}", base_med)
                                alt = [o] + [execute(E, X, plan, TF, a.margin, P, a.chunk) for _ in range(a.rerun)]
                                med = sorted(alt[1:], key=lambda x_: x_["total"])[len(alt[1:]) // 2]
                                runs = [x_["total"] for x_ in alt]
                                reexec[fr].append({"view": vi, "win": wn, "scheme": sch, "first": o["total"], "reruns": runs[1:], "kept": med["total"],
                                                   "changed": bool(med["total"] <= TF)})
                                o = med
                            img = o.pop("img")
                            psi = min(metrics.psnr(img.clamp(0, 1), Vw["full"].clamp(0, 1)), PSI_MAX)
                            m_ = Vw["fm"] if (o["n_lost_rx"] == 0 and o["n_rendered"] == n_vis) else metrics.all_metrics(img, Vw["gt"])
                            rec.update(t_wall=time.time(), variant=o["variant"], n_net=o["n_net"], total_ms=o["total"], gpu_total_ms=o["gpu_total"], t_dec_end=o["t_dec_end"],
                                       t_conc_end=o["t_conc_end"], conc_ms=o["conc"], ren_ms=o["ren"], pred_total=o["pred_total"], t_data_ms=o["t_data_ms"],
                                       n_rendered=o["n_rendered"], runs=runs, psi_rendered=psi, net_ms=plan["net_ms"])
                            if o["total"] > TF:
                                rec.update(deadline_miss=True, psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0)
                            else:
                                rec.update(**m_, psi=psi)
                            res_f[fr]["rows"].append(rec)
                if a.no_wait:
                    break
                others = [x for x in gpu_apps() if int(x.split(",")[0]) != os.getpid()]
                pm, px = probe(0.15)
                if not others and pm <= 1.3 * base_med and px <= 10.0:
                    break
                busy.append({"t": time.time(), "tag": f"redo view {cfg} {tau:g} {vi}", "attempt": attempt, "others": others, "probe_med": pm, "probe_max": px, "base_med": base_med})
                print(f"view {vi} disturbed (others {others}, probe {pm:.3f}/{px:.1f} ms vs {base_med:.3f}): executed again", flush=True)
            gc.collect()
            print(f"{cfg} tau{tau:g} view {vi}: S_A {SA} S_R {SR}, free A {free['A']['total']:.2f} R {free['R']['total']:.2f} ms, "
                  f"{len(res_f[a.frames[0]]['rows'])} rows, {time.time() - t_run:.0f}s", flush=True)
        for fr in a.exec_frames:
            rows, views = res_f[fr]["rows"], res_f[fr]["views"]
            summ = summarize(rows, views)
            res = {"scene": a.scene, "cfg": cfg, "tau_c_ms": tau, "n_anchor": na, "b_A": b_A, "b_E": b_E, "t_ren": None,
                   "anchor_modes": [int(tx[i]) for i in anchor], "data_modes": [int(tx[i]) for i in data_tx],
                   "timing": "executed", "n_ev": a.n_ev, "n_mds": a.n_mds, "has_edge_variant": False,
                   "strata": {f"{k[0]},{k[1]}": {kk: vv for kk, vv in v.items() if kk in ("pE", "pD", "nW", "WE")} for k, v in strata.items()},
                   "summary": summ, "views": views, "rows": rows, "schemes": list(SCHEMES), "view_nbr": True, "t_ent": "executed", "frames": fr,
                   "mlp_fallback": False, "net_timing": "executed", "tag": a.tag, "deadline_ms": 11.1 * fr * a.deadline_scale, "margin_ms": a.margin, "chunk": a.chunk,
                   "predictors": {k: v for k, v in P_d.items() if not k.endswith("_points")}, "predictor_points": {k: v for k, v in P_d.items() if k.endswith("_points")},
                   "reexec": reexec[fr], "probe_base_ms": base_med, "gpu_state": gpu_state(), "waited_idle_s": waited, "busy_log": busy, "seed": a.seed}
            sfx = {1: "_r5", 2: "_r6"}[fr]
            with open(os.path.join(a.out, a.scene, f"cold4{a.tag}_{cfg}_tau{tau:g}{sfx}{a.suffix}.json"), "w") as fh:
                json.dump(res, fh)
            print(cfg, tau, fr, json.dumps({k: round(v, 5) for k, v in summ.items() if "/" in k}), flush=True)


def pview(rows, views, sch, th):
    out = {}
    for v in views:
        if v["always_fail"][sch]:
            out[v["view"]] = 1.0; continue
        kind, p_ = ("D", v["pD"]) if sch.startswith("mds") else ("E", v["pE"])
        rs = [x for x in rows if x["view"] == v["view"] and x["scheme"] == sch and x["stratum"] == kind]
        out[v["view"]] = p_ * float(np.mean([x["psi"] < th for x in rs])) if rs else 0.0
    return out


def summarize(rows, views):
    summ = {}
    for th in (30, 35):
        pv = {sch: pview(rows, views, sch, th) for sch in SCHEMES}
        for sch in SCHEMES:
            summ[f"{sch}/{th}"] = float(np.mean(list(pv[sch].values()))) if views else 0.0
        summ[f"adaptive/{th}"] = float(np.mean([pv["conceal"][v["view"]] if v["always_fail"]["mds_conceal"] else pv["mds_conceal"][v["view"]] for v in views])) if views else 0.0
    return summ


# ---------------------------------------------------------------- pooled summary
def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))


def dist(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max())} if x.size else {}


def aggregate(a, write=True):
    out = {"semantics": __doc__.split("usage")[0], "configs": {}, "validation_vs_modelled": {}}
    ALLS = ("garden", "bicycle", "stump", "room", "counter", "kitchen", "bonsai", "truck", "train", "playroom", "drjohnson")
    for cfg, tau in (("T8R8", "2.5"), ("T8R8", "10"), ("T6R11", "2.5"), ("T6R11", "10")):
        for sfx in ("_r5", "_r6"):
            files = [os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}{sfx}{a.suffix}.json") for s in ALLS]
            js = [json.load(open(f)) for f in files if os.path.exists(f)]
            if not js:
                continue
            key = f"{cfg}|tau{tau}|{sfx}"
            g = {"scenes": [j["scene"] for j in js], "deadline_ms": js[0]["deadline_ms"], "margin_ms": js[0]["margin_ms"], "chunk": js[0]["chunk"]}
            und = [(j["scene"], v["view"]) for j in js for v in j["views"] if v["always_fail"]["none"]]
            g["undeliverable_views"] = {"n": len(und), "of": sum(len(j["views"]) for j in js), "views": [f"{s}:{v}" for s, v in und],
                                        "frac": len(und) / max(1, sum(len(j["views"]) for j in js))}
            # probabilities per cold start, as fill.py: undeliverable views left out, mean over the views of a scene, then over the scenes
            pr = {}
            for j in js:
                xv = {v["view"] for v in j["views"] if v["always_fail"]["none"]}
                vs = [v for v in j["views"] if v["view"] not in xv]; rs = [r for r in j["rows"] if r["view"] not in xv]
                sm = summarize(rs, vs)
                for k, v in sm.items():
                    pr.setdefault(k, []).append(v)
            g["p_fail_per_cold_start"] = {k: float(np.mean(v)) for k, v in pr.items()}
            g["p_fail_all_views"] = {k: float(np.mean([j["summary"][k] for j in js])) for k in js[0]["summary"]}
            R = [dict(r, scene=j["scene"]) for j in js for r in j["rows"]]
            per = {}
            for sch in SCHEMES:
                kind = "D" if sch.startswith("mds") else "E"
                rs = [r for r in R if r["scheme"] == sch and r["stratum"] == kind]
                cls = {c: sum(r["variant"] == c for r in rs) for c in CLASSES}
                ex = [r for r in rs if r.get("total_ms") is not None]
                lost = [r for r in ex if r["n_lost"] > 0 or r["variant"] != "complete"]
                per[sch] = {"windows": len(rs), "classes": cls,
                            "classes_below30": {c: sum(r["variant"] == c and r["psi"] < 30 for r in rs) for c in CLASSES},
                            "executed": len(ex), "deadline_miss_executed": sum(bool(r["deadline_miss"]) for r in ex),
                            "deadline_miss_frac_executed": sum(bool(r["deadline_miss"]) for r in ex) / max(1, len(ex)),
                            "cp95_upper_miss": cp_upper(sum(bool(r["deadline_miss"]) for r in ex), len(ex)) if ex else None,
                            "below30": sum(r["psi"] < 30 for r in rs), "below35": sum(r["psi"] < 35 for r in rs),
                            "total_ms": dist([r["total_ms"] for r in ex]), "slack_ms": dist([g["deadline_ms"] - r["total_ms"] for r in ex]),
                            "overrun_vs_prediction_ms": dist([r["total_ms"] - r["pred_total"] for r in ex if r.get("pred_total") is not None]),
                            "n_net_frac": dist([r["n_net"] / r["n_lost"] for r in lost if r["n_lost"] and r["scheme"].endswith("conceal")])}
            g["schemes"] = per
            ex_all = [r for r in R if r.get("total_ms") is not None]
            g["executed_windows"] = len(ex_all)
            g["deadline_miss_executed"] = sum(bool(r["deadline_miss"]) for r in ex_all)
            g["deadline_miss_frac"] = g["deadline_miss_executed"] / max(1, len(ex_all))
            re_ = [x for j in js for x in j["reexec"]]
            g["reexecuted"] = {"n": len(re_), "changed_to_hit": sum(x["changed"] for x in re_)}
            g["predictors"] = {j["scene"]: {k: j["predictors"][k] for k in ("ren", "idw", "knn", "dec", "chunk_p99")} for j in js}
            g["chunk_stats"] = {j["scene"]: j["predictors"]["chunk_stats"] for j in js}
            out["configs"][key] = g
            # comparison with the modelled eval_cold3.py rows of the same windows
            val = {}
            for j in js:
                f3 = os.path.join(a.out, j["scene"], f"cold3{a.tag}_{cfg}_tau{tau}{sfx}.json")
                if not os.path.exists(f3):
                    continue
                m = json.load(open(f3))
                cnt = {}
                for r in m["rows"]:
                    k_ = (r["view"], r["stratum"], r["scheme"]); cnt.setdefault(k_, []).append(r)
                rows4 = {}
                for r in j["rows"]:
                    rows4.setdefault((r["view"], r["stratum"], r["scheme"]), []).append(r)
                for k_, rs3 in cnt.items():
                    rs4 = sorted(rows4.get(k_, []), key=lambda x_: x_["win"])
                    for i_, r3 in enumerate(rs3):
                        if i_ >= len(rs4):
                            continue
                        r4 = rs4[i_]; sch = k_[2]
                        v_ = val.setdefault(sch, {"windows": 0, "fail30_model": 0, "fail30_exec": 0, "both": 0, "n_lost_equal": 0, "miss_model": 0, "miss_exec": 0,
                                                  "variants_model": {}, "variants_exec": {}, "psi_diff_nonfail": []})
                        v_["windows"] += 1; v_["n_lost_equal"] += int(r3.get("n_lost") == r4["n_lost"])
                        f3_, f4_ = r3["psi"] < 30, r4["psi"] < 30
                        v_["fail30_model"] += f3_; v_["fail30_exec"] += f4_; v_["both"] += f3_ and f4_
                        v_["miss_model"] += bool(r3.get("deadline_miss")); v_["miss_exec"] += bool(r4["deadline_miss"])
                        v_["variants_model"][r3["variant"]] = v_["variants_model"].get(r3["variant"], 0) + 1
                        v_["variants_exec"][r4["variant"]] = v_["variants_exec"].get(r4["variant"], 0) + 1
                        if not f3_ and not f4_ and r3["psi"] < 99 and r4["psi"] < 99:
                            v_["psi_diff_nonfail"].append(r4["psi"] - r3["psi"])
                m_s = {k: float(v) for k, v in m["summary"].items()}
                val.setdefault("_p_model", {})[j["scene"]] = m_s
                val.setdefault("_p_exec", {})[j["scene"]] = j["summary"]
            for sch, v_ in val.items():
                if not sch.startswith("_"):
                    v_["psi_diff_nonfail"] = dist(v_["psi_diff_nonfail"])
            if val:
                out["validation_vs_modelled"][key] = val
    # Table tab:cold equivalent: columns T8R8 / T6R11 x tau_c 2.5 / 10 ms x one / two frames (fill.py order), rows = schemes
    cols = [f"{c}|tau{t}|{f}" for c, t in (("T8R8", "2.5"), ("T8R8", "10"), ("T6R11", "2.5"), ("T6R11", "10")) for f in ("_r5", "_r6")]
    for th in (30, 35):
        tab = {}
        for sch in SCHEMES[:6] + ("adaptive",) + SCHEMES[6:]:
            tab[sch] = {c: out["configs"][c]["p_fail_per_cold_start"][f"{sch}/{th}"] if c in out["configs"] else None for c in cols}
        tab["undeliverable_views_frac"] = {c: out["configs"][c]["undeliverable_views"]["frac"] if c in out["configs"] else None for c in cols}
        out[f"table_cold_{th}dB"] = tab
    out["table_columns"] = cols
    if not write:
        return out
    with open(os.path.join(a.out, f"rev6_cold4_summary{a.suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, g in out["configs"].items():
        print(k, g["scenes"], "undeliverable", g["undeliverable_views"]["n"], "/", g["undeliverable_views"]["of"], "misses", g["deadline_miss_executed"], "/", g["executed_windows"])
        print("   ", {s: round(v, 4) for s, v in g["p_fail_per_cold_start"].items() if s.endswith("/30")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden")
    ap.add_argument("--tag", default="_v2m_r1")
    ap.add_argument("--runs", default="T8R8:2.5,T8R8:10,T6R11:2.5,T6R11:10")
    ap.add_argument("--frames", default="1,2")
    ap.add_argument("--exec-frames", default="", help="deadlines (frames) executed and written (default: --frames); the others are only drawn")
    ap.add_argument("--margin", type=float, default=0.3, help="safety margin of the anytime deadline checks (ms)")
    ap.add_argument("--chunk", type=int, default=8192, help="lost anchors per MLP chunk")
    ap.add_argument("--rerun", type=int, default=3, help="re-executions of a window that misses the deadline (median kept)")
    ap.add_argument("--free-reps", type=int, default=3)
    ap.add_argument("--max-redo", type=int, default=3, help="re-executions of a view disturbed by another GPU process or a slowed GPU")
    ap.add_argument("--cal-views", type=int, default=40)
    ap.add_argument("--cal-reps", type=int, default=10)
    ap.add_argument("--warm-views", type=int, default=3)
    ap.add_argument("--q-render", type=float, default=99, help="percentile of the render predictor (residuals of the calibration fit)")
    ap.add_argument("--q-stage", type=float, default=50, help="percentile of the kNN, IDW, decoding and MLP-chunk predictors")
    ap.add_argument("--cal-chunks", default="1024,2048,4096,8192,16384")
    ap.add_argument("--n-ev", type=int, default=3)
    ap.add_argument("--n-mds", type=int, default=2)
    ap.add_argument("--max-views", type=int, default=0, help="test only the first N test views (smoke test)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--deadline-scale", type=float, default=1.0, help="functional tests on a busy GPU only")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for an idle GPU (functional tests only)")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    a.frames = [int(x) for x in a.frames.split(",")]
    a.exec_frames = [int(x) for x in a.exec_frames.split(",")] if a.exec_frames else a.frames
    assert set(a.exec_frames) <= set(a.frames)
    a.cal_chunks = [int(x) for x in a.cal_chunks.split(",")]
    if a.aggregate:
        aggregate(a)
    else:
        run_scene(a)
