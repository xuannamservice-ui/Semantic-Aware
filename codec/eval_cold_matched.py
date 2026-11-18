"""Cold start timed on the GPU with exactly the semantics of eval_cold3.py, against its modelled stage times; and the
GPU cost of the view-restricted neighbour search that eval_cold3.py --view-nbr assumes.

Mode cold (per scene and configuration): the windows of eval_cold3.py (same seeds; stratum E, n_ev per view, the random
draws of stratum D are consumed so that the windows are identical), schemes none and conceal, first view due within one
frame (11.1 ms) and within two frames (22.2 ms), plus one loss-free window per view. For every window the decision of
eval_cold3.py is made with its modelled times (T_GATHER, T_NET or T_MLP_NET, T_EDGE_NET, T_MLP_NET, t_enc, t_dr from
bench_trained.json plus the entropy tail max(te/S_A, te - (S_A - 1)) with te = t_ent n/112,000): render what arrived,
the network, the deployment variant, the MLP fallback, anytime with IDW, or IDW when the network is lost. The chosen
variant is then executed and timed on the GPU (CUDA-synchronized wall clock, median of --reps after warm-up):
  tail   : real GPU rANS decoding (range_code2 kernel) of the tokens of the last data slot (te_last) and of all received
           tokens (te_full); tail = max(te_last, te_full - (S_A - 1) ms), i.e. the slot-by-slot decoding during reception
  conc   : the executed concealment (depth sort, LayeredConcealer gathering from the view-restricted neighbour table,
           FP16 networks; the deployment variant includes the encoding of the received anchors), only when anchors are lost
  ren    : decoding of the rendered anchors into neural Gaussians and rasterization (model.render)
  knn    : not in eval_cold3.py: the search of the k = 16 neighbours of the lost anchors among the anchors of the view,
           brute force (chunked cdist + top-k) and on a uniform grid (27 cells, exact; fallback to brute force where the
           k-th candidate is farther than the cell size), timed separately and reported as an additional term
measured total = start + tail + conc + ren with start = S_A, or S_A + 1 when the edge waits for the network (as the
budget of eval_cold3.py); modelled total = start + t_dr + modelled concealment time of the chosen variant. A view is
infeasible loss-free if S_A + t_dr > deadline (model) or S_A + tail + ren > deadline (measured).
Mode knn-session: the same neighbour search for the missing anchors of every session frame with losses (frame paths of
eval_frames.py, T8R8, tau_c 10 ms, scheme conceal) among the anchors whose positions the edge holds (all anchors visible
in the frames so far), against the precomputed whole-scene table used by eval_frames.py and bench_e2e_frames.py.
Output: <out>/<scene>/cold_matched<tag><suffix>.json, <out>/<scene>/knn_session.json; --aggregate pools them into
results/rev6_latency_cold_matched.json and results/rev6_knn.json.
usage: python eval_cold_matched.py --scene garden --runs T8R8:2.5,T8R8:10,T6R11:2.5,T6R11:10 --mlp-fallback
       python eval_cold_matched.py --scene garden --tag _v2m_r1 --net-timing mlp --runs T8R8:2.5 --suffix _mlpj
       python eval_cold_matched.py --scene garden --mode knn-session
       python eval_cold_matched.py --aggregate
"""
import argparse
import copy
import glob
import itertools
import json
import math
import os
import time

import numpy as np
import torch

import eval_cold3 as ec
from bench_e2e_frames import Coder
from conceal_net import knn
from eval_frames import pattern_stream
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_layered import LayeredConcealer

SLOT = 1.0


def sync():
    torch.cuda.synchronize()


def timed(fn, reps):
    """Median wall time (ms) of fn over reps runs after one untimed run; returns (ms, last result)."""
    out = fn(); sync()
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); out = fn(); sync(); ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts)), out


# ---------------------------------------------------------------- neighbour search
def knn_brute(q, db, k, self_pos, chunk=4096):
    """k nearest anchors of the queries among db (chunked cdist + top-k, as conceal_net.knn); self_pos: index of each
    query in db (excluded)."""
    out = []
    for i in range(0, q.shape[0], chunk):
        d = torch.cdist(q[i:i + chunk], db)
        d[torch.arange(d.shape[0], device=d.device), self_pos[i:i + chunk]] = float("inf")
        out.append(d.topk(k, largest=False).indices)
    return torch.cat(out) if out else torch.zeros(0, k, dtype=torch.long, device=q.device)


def _grid_pass(q, self_pos, db, skey, order, lo, D, h, k):
    Q = q.shape[0]
    cq = torch.floor((q - lo) / h).long()
    off = torch.tensor([[i, j, l] for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1)], device=q.device)
    nc = cq[:, None, :] + off[None]
    nk = ((nc[..., 0] * D[1] + nc[..., 1]) * D[2] + nc[..., 2]).reshape(-1)
    st = torch.searchsorted(skey, nk); en = torch.searchsorted(skey, nk, right=True)
    lens = en - st
    tot = int(lens.sum())
    qid = torch.arange(Q, device=q.device).repeat_interleave(27).repeat_interleave(lens)
    offs = torch.cumsum(lens, 0) - lens
    within = torch.arange(tot, device=q.device) - offs.repeat_interleave(lens)
    cand = order[st.repeat_interleave(lens) + within]
    d2 = ((q[qid] - db[cand]) ** 2).sum(1)
    d2 = torch.where(cand == self_pos[qid], torch.full_like(d2, float("inf")), d2)
    p1 = torch.argsort(d2)
    perm = p1[torch.sort(qid[p1], stable=True)[1]]
    cs, ds = cand[perm], d2[perm]
    cnt = torch.bincount(qid, minlength=Q)
    start = torch.cumsum(cnt, 0) - cnt
    pos = (start[:, None] + torch.arange(k, device=q.device)[None]).clamp_max(max(cs.numel() - 1, 0))
    if cs.numel() == 0:
        return torch.zeros(Q, k, dtype=torch.long, device=q.device), torch.zeros(Q, dtype=torch.bool, device=q.device)
    ok = (cnt >= k + 1) & (ds[pos[:, -1]] <= h * h)          # exact: anything outside the 27 cells is farther than h
    return cs[pos], ok


def knn_grid(q, db, k, h, self_pos):
    """Exact k-NN on uniform grids: candidates from the 27 cells around the query cell (cell size h); queries whose k-th
    candidate is farther than h are searched again with cells of 2h, 4h, ... Returns (idx, number of re-searched queries)."""
    Q = q.shape[0]
    res = torch.zeros(Q, k, dtype=torch.long, device=q.device)
    todo = torch.arange(Q, device=q.device); n_re = 0
    while todo.numel():
        lo = db.min(0).values - h
        cdb = torch.floor((db - lo) / h).long(); D = cdb.max(0).values + 3
        skey, order = torch.sort((cdb[:, 0] * D[1] + cdb[:, 1]) * D[2] + cdb[:, 2])
        r, ok = _grid_pass(q[todo], self_pos[todo], db, skey, order, lo, D, h, k)
        res[todo[ok]] = r[ok]
        todo = todo[~ok]; n_re += int(todo.numel()); h = 2 * h
    return res, n_re


def check_equal(a, b, q, db):
    """Fraction of queries whose neighbour sets agree (ties in distance count as agreement)."""
    if a.shape[0] == 0:
        return 1.0
    da = ((q[:, None] - db[a]) ** 2).sum(-1).sort(1).values
    dbb = ((q[:, None] - db[b]) ** 2).sum(-1).sort(1).values
    return float(torch.isclose(da, dbb, rtol=1e-5, atol=1e-9).all(1).float().mean())


# ---------------------------------------------------------------- cold start
class FP16Net:
    """FP16 copy of a concealment network with the interface of LayeredConcealer.predict (the MLP's weighted means in
    FP32, see bench_e2e_frames.Edge.mlp_forward)."""

    def __init__(self, net, kind):
        self.kind = kind
        self.net = copy.deepcopy(net).half().eval()

    def __call__(self, rel, base, enh, recv, own):
        if self.kind == "mlp":
            w = 1.0 / rel.norm(dim=-1).clamp_min(1e-6); wr = w * recv.to(w.dtype)
            has = (wr.sum(1, keepdim=True) > 0).to(w.dtype)
            enh_m = (wr[..., None] * enh).sum(1) / wr.sum(1, keepdim=True).clamp_min(1e-12)
            base_m = (w[..., None] * base).sum(1) / w.sum(1, keepdim=True)
            frac = recv.to(w.dtype).mean(1, keepdim=True)
            return self.net.mlp(torch.cat([own, base_m, enh_m * has, frac, has], -1).half()).float()
        return self.net(rel.half(), base.half(), enh.half(), recv, own.half()).float()


def edge_predict(enet, model, stats, nbr, radius, lost, avail, base, enh, view):
    """Deployment variant as the edge runs it: encodings of the received anchors of the view, then cached keys/values."""
    bm, bs, em, es = stats
    tab = enet.encode(((base[view] - bm) / bs).half(), ((enh[view] - em) / es).half(), avail[view])   # (n_view, 2d) per layer
    inv = torch.full((model.xyz.shape[0],), -1, dtype=torch.long, device=view.device); inv[view] = torch.arange(view.numel(), device=view.device)
    outs = []
    for c in torch.split(lost, 16384):
        nb = nbr[c]
        rel = ((model.xyz[nb] - model.xyz[c][:, None]) / radius).half()
        outs.append(enet.forward_table(tab, inv[nb], rel, ((base[c] - bm) / bs).half()).float() * es + em)
    return torch.cat(outs)


def model_conc(variant, n_lost, n_net, t_enc, T_NET, has_mlp):
    """Concealment time that eval_cold3.py charges for the chosen variant (ms)."""
    G, t_of = ec.T_GATHER, ec.t_of
    if variant == "full":
        return t_of(n_lost, G) + t_of(n_lost, T_NET)
    if variant == "edge":
        return t_enc + t_of(n_lost, G) + t_of(n_lost, ec.T_EDGE_NET)
    if variant in ("mlp_fb", "mlp_fb_edge"):
        tn = ec.T_EDGE_NET if variant.endswith("edge") else T_NET
        r = n_lost - n_net
        return (t_enc if variant.endswith("edge") else 0.0) + t_of(n_lost, G) + t_of(n_net, tn) + (t_of(r, G) + t_of(r, ec.T_MLP_NET) if r else 0.0)
    if variant in ("anytime", "anytime_edge"):
        tn = ec.T_EDGE_NET if variant.endswith("edge") else T_NET
        return (t_enc if variant.endswith("edge") else 0.0) + t_of(n_lost, G) + t_of(n_lost - n_net, G) + t_of(n_net, tn)
    return 0.0                                                   # drop, idw (not charged by eval_cold3.py)


def run_cold(a):
    if a.net_timing == "mlp":
        ec.T_NET = ec.T_MLP_NET
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        base, enh = base.detach().contiguous(), enh.detach().contiguous()
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    kn = c["kn"]
    nbr = knn(model.xyz, kn)
    main_kind = "mlp" if ck.get("net_type") == "mlp" else "lite"
    net16 = FP16Net(nets["gen_fp"], main_kind)
    mlp16 = FP16Net(nets["mlp_fp"], "mlp") if ("mlp_fp" in nets and a.mlp_fallback) else None
    enet = copy.deepcopy(nets["edge_fp"]).half().eval() if "edge_fp" in nets else None
    conc = LayeredConcealer(model, L, net16, stats, nbr, ck["radius"])
    conc_mlp = LayeredConcealer(model, L, mlp16, stats, nbr, ck["radius"]) if mlp16 else None
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    coder = Coder(model, L, ck, a.threads)
    mt = json.load(open(os.path.join(a.out, a.scene, f"metrics_layered{a.tag}.json")))
    rc_path = os.path.join(a.out, a.scene, f"rangecode{a.tag}.json")
    base_bits = (json.load(open(rc_path))["layers"]["base"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.012 * mt["rate_bits_per_anchor"]["base_layer"])
    b_A = mt["rate_bits_per_anchor"]["position_fixed"] + base_bits
    b_E = (json.load(open(rc_path))["layers"]["enhancement"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.013 * mt["rate_bits_per_anchor"]["enh_layer"])
    bt_path = os.path.join(a.out, a.scene, "bench_trained.json")
    t_view = {v["view"]: v["ren_cal"] for v in json.load(open(bt_path))["views"]} if os.path.exists(bt_path) else None
    t_ren = 2.159 * scene.W * scene.H / (1297 * 840)
    W, H = scene.W, scene.H
    res = {"scene": a.scene, "tag": a.tag, "net_timing": a.net_timing, "mlp_fallback": mlp16 is not None, "has_edge": enet is not None,
           "kn": kn, "radius": float(ck["radius"]), "reps": a.reps, "t_ent": a.t_ent, "b_A": b_A, "b_E": b_E, "runs": {}}

    def decode_ms(ids_np):
        """GPU decoding (copy, rANS, dequantization, table write) of the base and enhancement tokens of the given anchors."""
        if ids_np is None or len(ids_np) == 0:
            return 0.0
        pb = coder.encode("base", ids_np); pe = coder.encode("enh", ids_np)
        tb, te = torch.zeros_like(base), torch.zeros_like(enh)
        return timed(lambda: (coder.decode(pb, tb), coder.decode(pe, te)), a.reps)[0]

    def decode_split_ms(base_ids, enh_ids):
        if len(base_ids) == 0 and len(enh_ids) == 0:
            return 0.0
        pb = coder.encode("base", base_ids) if len(base_ids) else None
        pe = coder.encode("enh", enh_ids) if len(enh_ids) else None
        tb, te = torch.zeros_like(base), torch.zeros_like(enh)
        return timed(lambda: (pb and coder.decode(pb, tb), pe and coder.decode(pe, te)), a.reps)[0]

    for spec in a.runs.split(","):
        cfg, tau = spec.split(":"); tau = float(tau); na = 4 if cfg == "T8R8" else 2
        ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
        cf = ms["configs"][cfg]; tx = cf["tx"]; M = len(tx)
        rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
        xs = []
        for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz"))):
            z = np.load(f); xs.append(z[cfg])
        pats = np.concatenate(xs)
        bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
        flat = bits.reshape(-1, M)
        anchor = min(itertools.combinations(range(M), na), key=lambda s_: int(flat[:, list(s_)].all(1).sum()))
        anchor_streams = sorted(rel.index(int(tx[i])) for i in anchor)
        data_tx = [i for i in range(M) if i not in anchor]; Md = len(data_tx)
        data_streams = torch.tensor(sorted(rel.index(int(tx[i])) for i in data_tx), device="cuda")
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
                strata[(SA, SR)] = dict(E=np.flatnonzero(Em.ravel()), D=np.flatnonzero(D.ravel()), pE=float(Em.mean()), nW=nW, WE=WE)
            return strata[(SA, SR)]

        rng = np.random.default_rng(a.seed)
        views, rows, t0 = [], [], time.time()
        with torch.no_grad():
            for vi in scene.test_idx:
                vi = int(vi)
                Kc = torch.from_numpy(scene.K[vi]).cuda(); V = torch.from_numpy(scene.viewmats[vi]).cuda()
                idx = model.visible(Kc, V, W, H); n_vis = int(idx.numel())
                tv = nbr.clone(); tv[idx] = idx[knn(model.xyz[idx], kn)]
                conc.nbr = tv
                if conc_mlp is not None:
                    conc_mlp.nbr = tv
                SA = max(1, math.ceil((n_vis * b_A + ec.MLP_MBIT) / ec.R0)); L2 = 2 * n_vis * b_E
                SR = SA + max(0, math.ceil((L2 - Md * ec.R0 * SA) / (M * ec.R0)))
                te_mod = a.t_ent * n_vis / 112_000
                tail_mod = max(te_mod / SA, te_mod - (SA - 1))
                ren_mod = t_view[vi] if t_view is not None else t_ren * n_vis / 110_000
                t_dr = ren_mod + tail_mod
                t_enc = 0.66 * n_vis / 110_000
                st = strata_for(SA, SR)
                evE = [int(i) for i in rng.choice(st["E"], min(a.n_ev, st["E"].size), replace=False)]
                rng.choice(st["D"], min(a.n_mds, st["D"].size), replace=False)
                s_, tid = pk.token_ids(idx); T = int(tid[-1]) + 1
                slot_of = ((torch.arange(T, device="cuda") * SA) // T)[tid]
                s_np = s_.cpu().numpy(); last_np = (slot_of == SA - 1).cpu().numpy()
                # loss-free: decoding during reception, then render the view
                te_full = decode_ms(s_np); te_last = decode_ms(s_np[last_np])
                tail_meas = max(te_last, te_full - (SA - 1) * SLOT)
                ren_meas, _ = timed(lambda: model.render(at, idx, Kc, V, W, H), a.reps)
                knn_view_ms, _ = timed(lambda: knn(model.xyz[idx], kn), 1)
                v = {"view": vi, "n_visible": n_vis, "S_A": SA, "S_R": SR, "te_model": te_mod, "tail_model": tail_mod, "ren_model": ren_mod,
                     "t_dr_model": t_dr, "te_full": te_full, "te_last": te_last, "tail_meas": tail_meas, "ren_meas": ren_meas,
                     "knn_all_view_anchors_ms": knn_view_ms}
                for fr in (1, 2):
                    v[f"free_total_model_f{fr}"] = SA + t_dr; v[f"free_total_meas_f{fr}"] = SA + tail_meas + ren_meas
                    v[f"infeasible_model_f{fr}"] = SA + t_dr > 11.1 * fr
                    v[f"infeasible_meas_f{fr}"] = SA + tail_meas + ren_meas > 11.1 * fr
                views.append(v)
                for w in evE:
                    ch, s = divmod(w, st["nW"])
                    prng = np.random.default_rng(vi * 7919 + ch * 251 + s)
                    St = [set(patterns_to_streams(int(pats[ch, s + t]), tx, rel)) for t in range(st["WE"] + 1)]
                    fs = [torch.tensor([m in St[t] for m in range(M)], device="cuda") for t in range(st["WE"] + 1)]
                    aset = [all(x in St[t] for x in anchor_streams) for t in range(st["WE"] + 1)]
                    tok_stream = data_streams[pk.assign(s_, tid, Md, "interleaved", prng)][tid]
                    fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                    for t in range(SA):
                        fail |= (slot_of == t) & fs[t][tok_stream]
                    anchor_lost = any(aset[:SA]); net_lost = aset[SA]
                    lost = s_[fail]; n_lost = int(lost.numel())
                    avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda"); avail[s_[~fail]] = True
                    recv_np = s_np[~fail.cpu().numpy()]
                    # tail with the tokens that arrived: base layer of every anchor, enhancement of the received ones
                    te_full_w = decode_split_ms(s_np, recv_np)
                    te_last_w = decode_split_ms(s_np[last_np], s_np[last_np & ~fail.cpu().numpy()])
                    tail_w = max(te_last_w, te_full_w - (SA - 1) * SLOT)
                    ren_recv, _ = timed(lambda: model.render(at, s_[~fail], Kc, V, W, H), a.reps)
                    kb = kg = None; nfb = 0
                    if n_lost and spec in a.knn_runs.split(","):
                        db = model.xyz[idx]; qx = model.xyz[lost]
                        inv = torch.full((model.xyz.shape[0],), -1, dtype=torch.long, device="cuda"); inv[idx] = torch.arange(n_vis, device="cuda")
                        sp = inv[lost]
                        kb, rb = timed(lambda: knn_brute(qx, db, kn, sp), 1)
                        kg, (rg, nfb) = timed(lambda: knn_grid(qx, db, kn, a.grid_h * ck["radius"], sp), a.reps)
                        agree = check_equal(rb[:2000], rg[:2000], qx[:2000], db)
                    base_row = {"view": vi, "chain": ch, "slot": s, "S_A": SA, "n_visible": n_vis, "n_lost": n_lost, "anchor_lost": anchor_lost,
                                "net_lost": net_lost, "te_full": te_full_w, "te_last": te_last_w, "tail_meas": tail_w, "tail_model": tail_mod,
                                "ren_model": ren_mod, "ren_meas_received": ren_recv, "ren_meas_full": ren_meas,
                                "knn_brute_ms": kb, "knn_grid_ms": kg, "knn_grid_fallback": nfb, "knn_grid_agree": agree if kb is not None else None}
                    # scheme none: render what arrived at S_A
                    for fr in (1, 2):
                        TF = 11.1 * fr
                        r = dict(base_row, scheme="none", frames=fr)
                        if anchor_lost:
                            r.update(variant="anchor_lost", total_model=None, total_meas=None, miss_model=True, miss_meas=True); rows.append(r); continue
                        r.update(variant="drop" if n_lost else "complete", start=SA, conc_model=0.0, conc_meas=0.0,
                                 total_model=SA + t_dr, total_meas=SA + tail_w + (ren_recv if n_lost else ren_meas))
                        r["miss_model"] = r["total_model"] > TF; r["miss_meas"] = r["total_meas"] > TF; rows.append(r)
                    # scheme conceal, decision of eval_cold3.py with its modelled times
                    for fr in (1, 2):
                        TF = 11.1 * fr
                        r = dict(base_row, scheme="conceal", frames=fr)
                        if anchor_lost:
                            r.update(variant="anchor_lost", total_model=None, total_meas=None, miss_model=True, miss_meas=True); rows.append(r); continue
                        if SA + t_dr > TF:
                            r.update(variant="always_fail", total_model=SA + t_dr, total_meas=SA + tail_w + ren_meas, miss_model=True)
                            r["miss_meas"] = r["total_meas"] > TF; rows.append(r); continue
                        last, net_slot = SA, SA
                        variant, n_net, start = "complete", 0, SA
                        conc_fn = None
                        if n_lost:
                            budget = TF - max(last, net_slot + 1) - t_dr
                            start = max(last, net_slot + 1)
                            if budget < 2 * ec.t_of(n_lost, ec.T_GATHER):
                                variant = "drop"; start = SA
                            elif net_lost:
                                variant = "idw"
                            else:
                                T_NET = ec.T_NET
                                nf = n_lost if ec.t_of(n_lost, ec.T_GATHER) + ec.t_of(n_lost, T_NET) <= budget else ec.anytime_count(n_lost, budget, T_NET)
                                ne = 0
                                if enet is not None and nf < n_lost:
                                    ne = n_lost if t_enc + ec.t_of(n_lost, ec.T_GATHER) + ec.t_of(n_lost, ec.T_EDGE_NET) <= budget else ec.anytime_count(n_lost, budget - t_enc, ec.T_EDGE_NET)
                                use_edge = ne > nf; n_net = ne if use_edge else nf
                                mm_ = -1
                                if conc_mlp is not None and n_net < n_lost:
                                    mf_ = ec.split_count(n_lost, budget, T_NET, ec.T_MLP_NET)
                                    me_ = ec.split_count(n_lost, budget - t_enc, ec.T_EDGE_NET, ec.T_MLP_NET) if enet is not None else -1
                                    use_edge = me_ > mf_; mm_ = me_ if use_edge else mf_
                                if n_net >= n_lost:
                                    variant = "edge" if use_edge else "full"
                                elif mm_ >= 0:
                                    variant = "mlp_fb_edge" if use_edge else "mlp_fb"; n_net = mm_
                                else:
                                    variant = "anytime_edge" if use_edge else "anytime"
                        r.update(variant=variant, n_net=int(n_net), start=start)
                        r["conc_model"] = model_conc(variant, n_lost, n_net, t_enc, ec.T_NET, conc_mlp is not None)
                        r["total_model"] = start + t_dr + r["conc_model"]
                        # execute the chosen variant
                        if variant in ("complete",):
                            cm, rm = 0.0, ren_meas
                        elif variant == "drop":
                            cm, rm = 0.0, ren_recv
                        else:
                            def conceal_run():
                                e = enh.clone()
                                if variant == "idw":
                                    nb = tv[lost]; rr = avail[nb].float()
                                    wgt = rr / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
                                    m_ = (wgt[..., None] * enh[nb]).sum(1) / wgt.sum(1, keepdim=True).clamp_min(1e-12)
                                    e[lost] = torch.where((rr.sum(1) > 0)[:, None], m_, stats[2].expand_as(m_))
                                    return e
                                if variant in ("full", "edge"):
                                    sub, rest = lost, lost[:0]
                                else:
                                    depth = (V[:3, :3] @ model.xyz[lost].T + V[:3, 3:]).T[:, 2]
                                    o = torch.argsort(depth); sub, rest = lost[o[:n_net]], lost[o[n_net:]]
                                if sub.numel():
                                    e[sub] = edge_predict(enet, model, stats, tv, ck["radius"], sub, avail, base, enh, idx) if variant.endswith("edge") \
                                        else conc.predict(sub, avail, base, enh)
                                if rest.numel():
                                    if variant.startswith("mlp_fb"):
                                        e[rest] = conc_mlp.predict(rest, avail, base, enh)
                                    else:
                                        nb = tv[rest]; rr = avail[nb].float()
                                        wgt = rr / (model.xyz[nb] - model.xyz[rest][:, None]).norm(dim=-1).clamp_min(1e-6)
                                        m_ = (wgt[..., None] * enh[nb]).sum(1) / wgt.sum(1, keepdim=True).clamp_min(1e-12)
                                        e[rest] = torch.where((rr.sum(1) > 0)[:, None], m_, stats[2].expand_as(m_))
                                return e
                            cm, e_ = timed(conceal_run, a.reps)
                            rm, _ = timed(lambda: model.render(L.merge(base, e_), idx, Kc, V, W, H), a.reps)
                        r.update(conc_meas=cm, ren_meas=rm, total_meas=start + tail_w + cm + rm)
                        r["total_meas_with_knn_grid"] = r["total_meas"] + (kg if (kg and variant not in ("complete", "drop")) else 0.0)
                        r["miss_model"] = r["total_model"] > TF; r["miss_meas"] = r["total_meas"] > TF
                        r["miss_meas_with_knn"] = r["total_meas_with_knn_grid"] > TF
                        rows.append(r)
                print(f"{cfg} tau{tau:g} view {vi}: S_A {SA}, free model {SA + t_dr:.2f} meas {SA + tail_meas + ren_meas:.2f}, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        res["runs"][f"{cfg}:{tau:g}"] = {"views": views, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"cold_matched{a.tag}{a.suffix}.json"), "w") as fh:
        json.dump(res, fh)


# ---------------------------------------------------------------- session neighbour search
def run_knn_session(a):
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    kn = c["kn"]
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cfg, tau = "T8R8", 10.0
    cf = ms["configs"][cfg]; tx_ms = cf["tx"]; M = len(tx_ms)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx_ms)
    stream, tx = pattern_stream(a.phy, cfg, tau)
    best = min(itertools.combinations(range(M), 4),
               key=lambda sub: int(np.count_nonzero((stream & sum(1 << i for i in sub)) == sum(1 << i for i in sub))))
    anchor_modes = [rel.index(int(tx[i])) for i in best]
    order = [int(v) for v in np.argsort(scene.names)]; T = len(order); F = 11
    W, H = scene.W, scene.H
    vis = {vi: model.visible(torch.from_numpy(scene.K[vi]).cuda(), torch.from_numpy(scene.viewmats[vi]).cuda(), W, H) for vi in order}
    rng = np.random.default_rng(0); rows = []
    with torch.no_grad():
        nbr_g = knn(model.xyz, kn)
        for ch in range(a.chains):
            off = rng.integers(0, stream.size - T * F); pats = stream[off:off + T * F].reshape(T, F)
            cached = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda"); known = torch.zeros_like(cached)
            prng = np.random.default_rng(ch)
            for f, vi in enumerate(order):
                idx = vis[vi]; new = idx[~cached[idx]]; known[idx] = True
                if new.numel():
                    S0 = set(patterns_to_streams(pats[f, 0], tx, rel))
                    if all(s_ in S0 for s_ in anchor_modes):
                        lost = new
                    else:
                        lm = pk.lost_mask(new, M, "interleaved", sorted(S0), prng) if S0 else torch.zeros(new.numel(), dtype=torch.bool, device="cuda")
                        lost = new[lm]
                    delivered = new[~torch.isin(new, lost)] if lost.numel() else new
                    cached[delivered] = True
                miss = idx[~cached[idx]]
                if miss.numel() == 0:
                    continue
                dbi = torch.nonzero(known).squeeze(1); db = model.xyz[dbi]; qx = model.xyz[miss]
                inv = torch.full((model.xyz.shape[0],), -1, dtype=torch.long, device="cuda"); inv[dbi] = torch.arange(dbi.numel(), device="cuda")
                sp = inv[miss]
                kb, rb = timed(lambda: knn_brute(qx, db, kn, sp), a.reps)
                kg, (rg, nfb) = timed(lambda: knn_grid(qx, db, kn, a.grid_h * ck["radius"], sp), a.reps)
                kt, _ = timed(lambda: nbr_g[miss], a.reps)
                rows.append({"chain": ch, "frame": f, "first": f == 0, "n_miss": int(miss.numel()), "n_known": int(dbi.numel()), "knn_brute_ms": kb,
                             "knn_grid_ms": kg, "grid_fallback": nfb, "agree": check_equal(rb[:2000], rg[:2000], qx[:2000], db), "table_lookup_ms": kt})
            print(f"chain {ch}: {len(rows)} frames with missing anchors", flush=True)
    with open(os.path.join(a.out, a.scene, "knn_session.json"), "w") as fh:
        json.dump({"scene": a.scene, "tag": a.tag, "kn": kn, "grid_h_over_radius": a.grid_h, "rows": rows}, fh)


# ---------------------------------------------------------------- pooled summaries
def dist(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max())} if x.size else {}


def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))


def aggregate(a):
    out = {"T_frame_ms": 11.1, "semantics": "eval_cold3.py --view-nbr --t-ent 1.73 [--mlp-fallback]; decisions with the modelled times, execution timed",
           "runs": {}, "views": {}}
    for suffix, tag in (("", "_v2a_r1"), ("_mlpj", "_v2m_r1")):
        files = sorted(glob.glob(os.path.join(a.out, "*", f"cold_matched{tag}{suffix}.json")))
        for f in files:
            d = json.load(open(f))
            for run, rr in d["runs"].items():
                key = f"{tag}{suffix}|{run}"
                g = out["runs"].setdefault(key, {"scenes": [], "views": [], "rows": []})
                g["scenes"].append(d["scene"])
                g["views"] += [dict(v, scene=d["scene"]) for v in rr["views"]]
                g["rows"] += [dict(r, scene=d["scene"]) for r in rr["rows"]]
    summ = {}
    for key, g in out["runs"].items():
        V = g["views"]; R = g["rows"]
        s = {"scenes": g["scenes"], "n_views": len(V)}
        for fr in (1, 2):
            im = [v for v in V if v[f"infeasible_model_f{fr}"]]; ime = [v for v in V if v[f"infeasible_meas_f{fr}"]]
            s[f"loss_free_infeasible_f{fr}"] = {"model": len(im), "measured": len(ime),
                                                "model_views": sorted(f"{v['scene']}:{v['view']}" for v in im),
                                                "measured_views": sorted(f"{v['scene']}:{v['view']}" for v in ime)}
            dis = [v for v in V if v[f"infeasible_model_f{fr}"] != v[f"infeasible_meas_f{fr}"]]
            s[f"loss_free_disagreeing_f{fr}"] = [{"view": f"{v['scene']}:{v['view']}", "S_A": v["S_A"], "model_total": v[f"free_total_model_f{fr}"],
                                                  "meas_total": v[f"free_total_meas_f{fr}"], "d_tail": v["tail_meas"] - v["tail_model"],
                                                  "d_ren": v["ren_meas"] - v["ren_model"]} for v in dis]
        s["loss_free_stage_diff"] = {"tail_meas_minus_model": dist([v["tail_meas"] - v["tail_model"] for v in V]),
                                     "ren_meas_minus_model": dist([v["ren_meas"] - v["ren_model"] for v in V]),
                                     "total_meas_minus_model": dist([v["free_total_meas_f1"] - v["free_total_model_f1"] for v in V]),
                                     "te_full_meas": dist([v["te_full"] for v in V]), "te_model": dist([v["te_model"] for v in V])}
        for sch in ("none", "conceal"):
            for fr in (1, 2):
                rs = [r for r in R if r["scheme"] == sch and r["frames"] == fr]
                n = len(rs); km = sum(bool(r["miss_model"]) for r in rs); ke = sum(bool(r["miss_meas"]) for r in rs)
                kk = sum(bool(r.get("miss_meas_with_knn", r["miss_meas"])) for r in rs)
                var = {}
                for r in rs:
                    var.setdefault(r["variant"], [0, 0, 0]); var[r["variant"]][0] += 1
                    var[r["variant"]][1] += bool(r["miss_model"]); var[r["variant"]][2] += bool(r["miss_meas"])
                ex = [r for r in rs if r.get("total_meas") is not None and r.get("total_model") is not None and r["variant"] not in ("anchor_lost",)]
                s[f"{sch}/f{fr}"] = {"windows": n, "deadline_miss_model": km, "deadline_miss_measured": ke, "deadline_miss_measured_with_knn_grid": kk,
                                     "frac_model": km / n if n else None, "frac_measured": ke / n if n else None,
                                     "cp95_upper_measured": cp_upper(ke, n) if n else None,
                                     "variants[count,miss_model,miss_meas]": var,
                                     "meas_minus_model_total": dist([r["total_meas"] - r["total_model"] for r in ex]),
                                     "conc_meas_minus_model": dist([r["conc_meas"] - r["conc_model"] for r in ex if r.get("conc_meas") is not None]),
                                     "disagreeing_windows": [{k_: r[k_] for k_ in ("scene", "view", "chain", "slot", "variant", "n_lost", "n_net", "S_A", "total_model", "total_meas",
                                                                                   "conc_model", "conc_meas", "tail_model", "tail_meas", "ren_model", "ren_meas") if k_ in r}
                                                             for r in rs if bool(r["miss_model"]) != bool(r["miss_meas"])][:60]}
        kr = [r for r in R if r["scheme"] == "conceal" and r["frames"] == 1 and r["n_lost"] > 0 and r.get("knn_brute_ms") is not None]
        s["knn_view_restricted"] = {"brute_ms": dist([r["knn_brute_ms"] for r in kr]), "grid_ms": dist([r["knn_grid_ms"] for r in kr]),
                                    "n_lost": dist([r["n_lost"] for r in kr]), "grid_agree_min": min((r["knn_grid_agree"] for r in kr), default=None),
                                    "grid_fallback_queries": dist([r["knn_grid_fallback"] for r in kr]),
                                    "worst_case": max(({"n_lost": r["n_lost"], "n_visible": r["n_visible"], "brute_ms": r["knn_brute_ms"], "grid_ms": r["knn_grid_ms"],
                                                        "scene": r["scene"], "view": r["view"]} for r in kr), key=lambda x: x["n_lost"], default=None),
                                    "all_view_anchors_ms": dist([v["knn_all_view_anchors_ms"] for v in V])}
        summ[key] = s
    out["summary"] = summ
    rows_out = {k: {"views": g["views"], "rows": g["rows"]} for k, g in out["runs"].items()}
    del out["runs"]
    with open(os.path.join(a.out, "rev6_latency_cold_matched.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out, "rev6_latency_cold_matched_rows.json"), "w") as fh:
        json.dump(rows_out, fh)
    # neighbour search summary
    kn = {"semantics": "k = 16 neighbours of lost anchors; cold start: among the anchors of the view (eval_cold3.py --view-nbr); "
                       "session: among the anchors whose positions the edge holds", "cold": {k: v["knn_view_restricted"] for k, v in summ.items()}}
    ses = []
    for f in sorted(glob.glob(os.path.join(a.out, "*", "knn_session.json"))):
        d = json.load(open(f)); ses += [dict(r, scene=d["scene"]) for r in d["rows"]]
    if ses:
        for part, rs in (("all", ses), ("steady_state", [r for r in ses if not r["first"]]), ("first_frame", [r for r in ses if r["first"]])):
            kn.setdefault("session", {})[part] = {"frames": len(rs), "n_miss": dist([r["n_miss"] for r in rs]), "brute_ms": dist([r["knn_brute_ms"] for r in rs]),
                                                  "grid_ms": dist([r["knn_grid_ms"] for r in rs]), "table_lookup_ms": dist([r["table_lookup_ms"] for r in rs]),
                                                  "grid_agree_min": min(r["agree"] for r in rs) if rs else None}
    with open(os.path.join(a.out, "rev6_knn.json"), "w") as fh:
        json.dump(kn, fh, indent=1)
    for k, v in summ.items():
        print(k, json.dumps({x: v[x] for x in v if x.startswith(("loss_free_infeasible", "none/", "conceal/"))})[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cold", choices=("cold", "knn-session"))
    ap.add_argument("--scene", default="garden")
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--runs", default="T8R8:2.5,T8R8:10,T6R11:2.5,T6R11:10")
    ap.add_argument("--net-timing", default="attn", choices=("attn", "mlp"))
    ap.add_argument("--mlp-fallback", action="store_true")
    ap.add_argument("--t-ent", type=float, default=1.73)
    ap.add_argument("--n-ev", type=int, default=3)
    ap.add_argument("--n-mds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--knn-runs", default="T8R8:2.5,T8R8:10", help="runs in which the neighbour search is timed")
    ap.add_argument("--grid-h", type=float, default=1.5, help="grid cell size in units of the median 16-NN distance")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a)
    elif a.mode == "cold":
        run_cold(a)
    else:
        run_knn_session(a)
