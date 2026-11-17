"""Cold start, version 3: per-view schedule, rendered MDS code, anytime concealment with the faster of the two networks.

Schedule of a view v with n_v visible anchors (slot = 1 ms, frame period 11.1 ms, r0 = 9.35 Mbit per stream and slot),
following the balanced bursts of the deadline condition of the paper: the anchor payload of a view (positions, base layer
at its real coded size, decoding MLPs of 17 kB) is larger than the enhancement share of a stream, so the n_A anchor modes
carry only the anchor payload, over S_A = ceil((n_v b_A + 0.14 Mbit) / r0) slots (2 for an average garden view), and the
enhancement tokens (Morton order) are interleaved over the M' = M - n_A other streams and spread evenly over the same S_A
slots (their share, n_v b_E / M', is always below the anchor payload);
  - conceal: the concealment network follows in slot S_A + 1 on the anchor modes, for every concealing scheme;
  - rep2: every enhancement token on a pair of the M' streams (perfect matching with the fewest joint failures), and
    mds: an (M', M'/2) systematic MDS code across the M' streams, which recovers a slot with at most M'/2 failed streams
    and otherwise loses the tokens of that slot on failed streams; both double the enhancement load; they use
    the M' data streams in the S_A slots of the anchor payload and all M streams after it (the anchor modes are free
    then), with the tokens spread in proportion to the capacity of each slot, so S_R = S_A + ceil(max(0, 2 n_v b_E -
    M' r0 S_A) / (M r0)); in a slot that uses all M streams the pairs and the (M, M/2) code are taken over all of them;
  - arq: the edge reports the lost tokens one slot after it receives them; the sender resends them, with a lost anchor
    payload or network, in slots S_A + 2 and S_A + 4, each token on a random stream (of all M) that did not fail in the
    slot before, if decoding and rendering still fit after the slot.
Decoding and rasterization take T_dr(v), the time of the trained model for this view (bench_trained.py, scaled to the idle
GPU); without that file, 2.16 ms (n_v / 110,000) (W H) / (1297 x 840) from the synthetic benchmark. Entropy decoding is
assumed to keep pace with the reception. A scheme whose data cannot be complete and rendered within the frame period
fails for this view in every window; with losses, concealment must fit into the time that is left: the network conceals
the lost anchors nearest to the camera that fit and IDW the rest (one more gathering pass); the deployment variant (needs
0.66 ms n_v / 110,000 to encode the received anchors) is used instead of the full network where it conceals more anchors
in time. Feedback is assumed to arrive without loss.
Sampling: per view, windows of max(S_A + 4, S_R) + 1 slots; stratum E = some stream fails in slots 1 to max(S_A + 4, S_R)
(all other windows lose nothing), with probability p_E over the complete traces, and stratum D = the MDS code fails in
one of the first S_R slots or all anchor copies fail in one of the first S_A slots (subset of E), with probability p_D. n_ev windows of E are rendered for every scheme, n_mds windows of D
for the two MDS schemes; P(fail | v) = p_E E[fail | E] (MDS: p_D E[fail | D]), or 1 if the scheme cannot meet the deadline
for this view at all, and the failure probability of a configuration is the mean of P(fail | v) over the test views.
Output: <out>/<scene>/cold3<tag>_<cfg>_tau<tau>.json
usage: python eval_cold3.py --scene garden --cfg T8R8 --tau 2.5 --n-anchor 4 --n-ev 3 --n-mds 2
"""
import argparse
import glob
import itertools
import json
import math
import os
import time

import numpy as np
import torch

import metrics
from conceal_net import knn
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_layered import LayeredConcealer

LK = np.array([0, 5e3, 1e4, 2e4, 3e4])
T_GATHER = np.array([0, 0.08, 0.17, 0.36, 0.53])
T_NET = np.array([0, 0.85, 1.81, 3.92, 5.96])                  # compiled FP16 concealment network (Table 13)
T_EDGE_NET = np.array([0, 0.63, 1.07, 2.24, 3.39])             # deployment variant, cached encodings
T_MLP_NET = np.array([0, 0.32, 0.42, 1.04, 1.27])              # MLP of two layers of width 256, FP16, measured with three cold-start runs on the GPU
T_FRAME, R0, MLP_MBIT = 11.1, 9.35e6, 0.14e6
SCHEMES = ("none", "conceal", "rep2", "rep2_conceal", "mds", "mds_conceal", "arq", "arq_conceal")
PSI_MAX = 99.0


def t_of(L, T):
    return float(np.interp(L, LK, T) if L <= LK[-1] else T[-1] + (L - LK[-1]) * (T[-1] - T[-2]) / (LK[-1] - LK[-2]))


def anytime_count(n_lost, budget, tnet):
    """Largest number of lost anchors the network can conceal while IDW handles the rest (one more gathering pass)."""
    lo, hi = 0, n_lost
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if t_of(n_lost, T_GATHER) + t_of(n_lost - mid, T_GATHER) + t_of(mid, tnet) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def split_count(n_lost, budget, tnet, tfb):
    """Largest number of lost anchors that the network conceals (nearest to the camera) while the fallback network
    conceals the rest within the budget (one gathering pass for each); -1 if even the fallback alone does not fit."""
    best = -1
    for m in range(n_lost, -1, -max(1, n_lost // 64)):
        r = n_lost - m
        t = t_of(n_lost, T_GATHER) + t_of(m, tnet) + (t_of(r, T_GATHER) + t_of(r, tfb) if r else 0.0)
        if t <= budget:
            return m
    t = 2 * t_of(n_lost, T_GATHER) + t_of(n_lost, tfb)
    return 0 if t <= budget else best


def pairing(left, acc, cost, jf):
    if not left:
        return cost, acc
    a0 = left[0]
    return min(pairing([x for x in left if x not in (a0, b0)], acc + [(a0, b0)], cost + jf[a0, b0], jf) for b0 in left[1:])


def main(a):
    global RUN, T_FRAME
    T_FRAME = 11.1 * a.frames                    # frames allowed for the first view (1: shown in the frame of the request)
    RUN = tuple(a.schemes.split(",")) if a.schemes else SCHEMES
    rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = LayeredConcealer(model, L, nets["gen_fp"], stats, nbr, ck["radius"])
    conc_edge = LayeredConcealer(model, L, nets["edge_fp"], stats, nbr, ck["radius"]) if "edge_fp" in nets else None
    conc_mlp = LayeredConcealer(model, L, nets["mlp_fp"], stats, nbr, ck["radius"]) if ("mlp_fp" in nets and a.mlp_fallback) else None
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    mt = json.load(open(os.path.join(a.out, a.scene, f"metrics_layered{a.tag}.json")))
    rc_path = os.path.join(a.out, a.scene, f"rangecode{a.tag}.json")
    base_bits = (json.load(open(rc_path))["layers"]["base"]["real_bits_per_anchor"] if os.path.exists(rc_path)
                 else 1.012 * mt["rate_bits_per_anchor"]["base_layer"])
    b_A = mt["rate_bits_per_anchor"]["position_fixed"] + base_bits     # bits per anchor on every anchor mode
    b_E = (json.load(open(rc_path))["layers"]["enhancement"]["real_bits_per_anchor"] if os.path.exists(rc_path)
           else 1.013 * mt["rate_bits_per_anchor"]["enhancement_layer"])
    bt_path = os.path.join(a.out, a.scene, "bench_trained.json")
    t_view = {v["view"]: v["ren_cal"] for v in json.load(open(bt_path))["views"]} if os.path.exists(bt_path) else None
    t_ren = 2.159 * scene.W * scene.H / (1297 * 840)
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx = cf["tx"]; M = len(tx)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
    xs = []
    for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{a.tau:g}ms_pat*.npz"))):
        z = np.load(f); assert [int(x) for x in z[f"{a.cfg}_tx"]] == [int(x) for x in tx]; xs.append(z[a.cfg])
    pats = np.concatenate(xs)
    bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
    flat = bits.reshape(-1, M)
    jf = flat.T.astype(np.int64) @ flat.astype(np.int64)
    anchor = min(itertools.combinations(range(M), a.n_anchor), key=lambda s: int(flat[:, list(s)].all(1).sum()))
    anchor_streams = sorted(rel.index(int(tx[i])) for i in anchor)
    data_tx = [i for i in range(M) if i not in anchor]; Md = len(data_tx)
    data_streams = torch.tensor(sorted(rel.index(int(tx[i])) for i in data_tx), device="cuda")
    pairs = pairing(data_tx, [], 0, jf)[1]
    pair_of = {}
    for p_, q_ in pairs:
        sp, sq = rel.index(int(tx[p_])), rel.index(int(tx[q_])); pair_of[sp] = sq; pair_of[sq] = sp
    pair_all = {}
    for p_, q_ in pairing(list(range(M)), [], 0, jf)[1]:
        sp, sq = rel.index(int(tx[p_])), rel.index(int(tx[q_])); pair_all[sp] = sq; pair_all[sq] = sp
    all_streams = torch.arange(M, device="cuda")
    anyf = bits.any(-1); Kd = bits[..., data_tx].sum(-1); Ka = bits.sum(-1); alost = bits[..., list(anchor)].all(-1)
    strata = {}

    def strata_for(SA, SR):
        if (SA, SR) not in strata:
            WE = max(SA + 4, SR); nW = anyf.shape[1] - WE - 1
            E = np.zeros((anyf.shape[0], nW), bool); D = np.zeros_like(E)
            for t in range(WE):
                E |= anyf[:, t:nW + t]
            for t in range(SR):
                D |= (Kd[:, t:nW + t] > Md // 2) if t < SA else (Ka[:, t:nW + t] > M // 2)
            for t in range(SA):
                D |= alost[:, t:nW + t]
            strata[(SA, SR)] = dict(E=np.flatnonzero(E.ravel()), D=np.flatnonzero(D.ravel()), pE=float(E.mean()), pD=float(D.mean()),
                                    nW=nW, WE=WE)
        return strata[(SA, SR)]

    def failed(ch, s, t):
        return set(patterns_to_streams(int(pats[ch, s + t]), tx, rel))

    NB = {"t": nbr}

    def idw(lost, avail):
        nb = NB["t"][lost]; r = avail[nb].float()
        w = r / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
        m_ = (w[..., None] * enh[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        return torch.where((r.sum(1) > 0)[:, None], m_, stats[2].expand_as(m_))

    rows, views, t0 = [], [], time.time()
    dev = "cuda"
    with torch.no_grad():
        for vi in scene.test_idx:
            vi = int(vi)
            gt, Kc, V = scene.view(vi)
            idx = model.visible(Kc, V, scene.W, scene.H)
            full, _, _ = model.render(at, idx, Kc, V, scene.W, scene.H)
            n_vis = int(idx.numel())
            if a.view_nbr:                      # neighbours only among the anchors of this view (the others have no base layer yet)
                tv_ = nbr.clone(); tv_[idx] = idx[knn(model.xyz[idx], c["kn"])]
                NB["t"] = tv_; conc.nbr = tv_
                if conc_edge is not None:
                    conc_edge.nbr = tv_
                if conc_mlp is not None:
                    conc_mlp.nbr = tv_
            SA = max(1, math.ceil((n_vis * b_A + MLP_MBIT) / R0)); L2 = 2 * n_vis * b_E; SR = SA + max(0, math.ceil((L2 - Md * R0 * SA) / (M * R0)))
            t_dr = t_view[vi] if t_view is not None else t_ren * n_vis / 110_000
            t_enc = 0.66 * n_vis / 110_000
            if a.t_ent > 0:                     # entropy decoding on the GPU keeps pace with the reception (it decodes the tokens of a slot
                te_ = a.t_ent * n_vis / 112_000   # while the next one arrives); what remains after the last data slot adds to decoding
                t_dr += max(te_ / SA, te_ - (SA - 1))
            st = strata_for(SA, SR)
            # completion time of the data of each scheme (slots) when nothing is lost, and the time left for concealment
            done = {"none": SA, "conceal": SA, "rep2": SR, "rep2_conceal": SR, "mds": SR, "mds_conceal": SR,
                    "arq": SA, "arq_conceal": SA}
            net_slot = {"conceal": SA, "rep2_conceal": SA, "mds_conceal": SA, "arq_conceal": SA}             # 0-based
            rt_slots = [t for t in (SA + 1, SA + 3) if t + 1 + t_dr <= T_FRAME]                               # 0-based, must fit
            always_fail = {k: done[k] + t_dr > T_FRAME for k in SCHEMES}
            s_, tid = pk.token_ids(idx); T = int(tid[-1]) + 1
            ar = torch.arange(T, device=dev)
            slot_of = ((ar * SA) // T)[tid]
            capc = torch.tensor([Md * R0 if t < SA else M * R0 for t in range(SR)], device=dev, dtype=torch.float64).cumsum(0)
            pos_ = (ar.double() + 0.5) / T * min(L2, float(capc[-1]))
            slot_rep = torch.searchsorted(capc, pos_).clamp_max(SR - 1)[tid]
            views.append({"view": vi, "n_visible": n_vis, "S_A": SA, "S_R": SR, "t_dr": t_dr, "pE": st["pE"], "pD": st["pD"],
                          "always_fail": {k: bool(v) for k, v in always_fail.items()}, "rt_slots": [t + 1 for t in rt_slots]})
            evs = [("E", int(i)) for i in rng.choice(st["E"], min(a.n_ev, st["E"].size), replace=False)] + \
                  [("D", int(i)) for i in rng.choice(st["D"], min(a.n_mds, st["D"].size), replace=False)]
            for kind, w in evs:
                ch, s = divmod(w, st["nW"])
                prng = np.random.default_rng(vi * 7919 + ch * 251 + s)
                St = [failed(ch, s, t) for t in range(st["WE"] + 1)]
                fs = [torch.tensor([m in St[t] for m in range(M)], device=dev) for t in range(st["WE"] + 1)]
                aset = [all(x in St[t] for x in anchor_streams) for t in range(st["WE"] + 1)]
                nfd = [sum(int(m) in St[t] for m in data_streams.tolist()) for t in range(st["WE"] + 1)]
                tok_stream = data_streams[pk.assign(s_, tid, Md, "interleaved", prng)][tid]
                tok_all = all_streams[pk.assign(s_, tid, M, "interleaved", prng)][tid]
                nfa = [len(St[t]) for t in range(st["WE"] + 1)]
                for sch in (tuple(x for x in RUN if not x.startswith("mds")) if kind == "E" else tuple(x for x in RUN if x.startswith("mds"))):
                    rec = {"view": vi, "stratum": kind, "scheme": sch, "S_A": SA, "n_visible": n_vis, "variant": "full", "n_net": 0,
                           "anchor_lost": False, "net_lost": False, "deadline_miss": False}
                    if always_fail[sch]:
                        rec.update(deadline_miss=True, n_lost=0, psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); rows.append(rec); continue
                    anchor_lost = any(aset[:SA])
                    if sch.startswith("rep2"):
                        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                        for t in range(SR):
                            po_, ts_ = (pair_of, tok_stream) if t < SA else (pair_all, tok_all)
                            pf = torch.tensor([(m in po_) and (m in St[t]) and (po_[m] in St[t]) for m in range(M)], device=dev)
                            fail |= (slot_rep == t) & pf[ts_]
                    elif sch.startswith("mds"):
                        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                        for t in range(SR):
                            if (nfd[t] > Md // 2) if t < SA else (nfa[t] > M // 2):
                                fail |= (slot_rep == t) & fs[t][tok_stream if t < SA else tok_all]
                    else:
                        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                        for t in range(SA):
                            fail |= (slot_of == t) & fs[t][tok_stream]
                    net_lost = aset[net_slot[sch]] if sch in net_slot else False
                    last = done[sch]
                    if sch.startswith("arq"):
                        for t in rt_slots:
                            if anchor_lost and not aset[t]:
                                anchor_lost = False
                            if net_lost and not aset[t]:
                                net_lost = False
                            if fail.any():
                                ok_ = [m for m in range(M) if m not in St[t - 1]] or list(range(M))
                                ut, inv = torch.unique(tid[fail], return_inverse=True)
                                ns = torch.as_tensor(np.asarray(ok_)[prng.integers(0, len(ok_), int(ut.numel()))], device=dev)[inv]
                                fail[fail.clone()] = fs[t][ns]
                            last = t + 1
                    lost = s_[fail]; n_lost = int(lost.numel())
                    rec.update(anchor_lost=bool(anchor_lost), net_lost=bool(net_lost), n_lost=n_lost)
                    if anchor_lost:
                        rec.update(psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); rows.append(rec); continue
                    e = enh.clone(); ridx = idx
                    if n_lost and sch.endswith("conceal"):
                        budget = T_FRAME - max(last, net_slot[sch] + 1) - t_dr
                        avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device=dev); avail[s_[~fail]] = True
                        if budget < 2 * t_of(n_lost, T_GATHER):   # no time to conceal: render what arrived, as without concealment
                            ridx = s_[~fail]; rec["variant"] = "drop"
                        elif net_lost:
                            e[lost] = idw(lost, avail); rec["variant"] = "idw"
                        else:
                            nf = n_lost if t_of(n_lost, T_GATHER) + t_of(n_lost, T_NET) <= budget else anytime_count(n_lost, budget, T_NET)
                            ne = 0
                            if conc_edge is not None and nf < n_lost:
                                ne = n_lost if t_enc + t_of(n_lost, T_GATHER) + t_of(n_lost, T_EDGE_NET) <= budget else anytime_count(n_lost, budget - t_enc, T_EDGE_NET)
                            use_edge = ne > nf; n_net = ne if use_edge else nf; net_ = conc_edge if use_edge else conc
                            mm_ = -1
                            if conc_mlp is not None and n_net < n_lost:   # MLP instead of IDW for the anchors the network cannot reach
                                mf_ = split_count(n_lost, budget, T_NET, T_MLP_NET)
                                me_ = split_count(n_lost, budget - t_enc, T_EDGE_NET, T_MLP_NET) if conc_edge is not None else -1
                                use_edge = me_ > mf_; mm_ = me_ if use_edge else mf_; net_ = conc_edge if use_edge else conc
                            if n_net >= n_lost:
                                e[lost] = net_.predict(lost, avail, base, enh); rec["variant"] = "edge" if use_edge else "full"
                            elif mm_ >= 0:
                                depth = (V[:3, :3] @ model.xyz[lost].T + V[:3, 3:]).T[:, 2]
                                order = torch.argsort(depth)
                                sub, rest = lost[order[:mm_]], lost[order[mm_:]]
                                if rest.numel():
                                    e[rest] = conc_mlp.predict(rest, avail, base, enh)
                                if mm_:
                                    e[sub] = net_.predict(sub, avail, base, enh)
                                n_net = mm_; rec["variant"] = "mlp_fb_edge" if use_edge else "mlp_fb"
                            else:
                                depth = (V[:3, :3] @ model.xyz[lost].T + V[:3, 3:]).T[:, 2]
                                order = torch.argsort(depth)
                                sub, rest = lost[order[:n_net]], lost[order[n_net:]]
                                e[rest] = idw(rest, avail)
                                if n_net:
                                    e[sub] = net_.predict(sub, avail, base, enh)
                                rec["variant"] = "anytime_edge" if use_edge else "anytime"
                            rec["n_net"] = int(n_net)
                    elif n_lost:
                        ridx = s_[~fail]
                    if last + t_dr > T_FRAME:
                        rec.update(deadline_miss=True, psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); rows.append(rec); continue
                    img = model.render(L.merge(base, e), ridx, Kc, V, scene.W, scene.H)[0] if ridx.numel() else torch.zeros_like(full)
                    m = metrics.all_metrics(img, gt)
                    rec.update(**m, psi=min(metrics.psnr(img.clamp(0, 1), full.clamp(0, 1)), PSI_MAX)); rows.append(rec)
            print(f"view {vi}: S_A {SA}, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for sch in RUN:
        for th in (30, 35):
            pv = []
            for v in views:
                if v["always_fail"][sch]:
                    pv.append(1.0); continue
                kind, p_ = ("D", v["pD"]) if sch.startswith("mds") else ("E", v["pE"])
                rs = [x for x in rows if x["view"] == v["view"] and x["scheme"] == sch and x["stratum"] == kind]
                pv.append(p_ * float(np.mean([x["psi"] < th for x in rs])) if rs else 0.0)
            summ[f"{sch}/{th}"] = float(np.mean(pv))
    res = {"scene": a.scene, "cfg": a.cfg, "tau_c_ms": a.tau, "n_anchor": a.n_anchor, "b_A": b_A, "t_ren": t_ren,
           "anchor_modes": [int(tx[i]) for i in anchor], "data_modes": [int(tx[i]) for i in data_tx], "b_E": b_E,
           "timing": "bench_trained" if t_view is not None else "synthetic", "n_ev": a.n_ev, "n_mds": a.n_mds, "has_edge_variant": conc_edge is not None,
           "strata": {f"{k[0]},{k[1]}": {kk: vv for kk, vv in v.items() if kk in ("pE", "pD", "nW", "WE")} for k, v in strata.items()},
           "summary": summ, "views": views, "rows": rows}
    res["schemes"] = list(RUN); res["view_nbr"] = bool(a.view_nbr); res["t_ent"] = a.t_ent; res["frames"] = a.frames; res["mlp_fallback"] = conc_mlp is not None
    with open(os.path.join(a.out, a.scene, f"cold3{a.tag}_{a.cfg}_tau{a.tau:g}{a.suffix}.json"), "w") as fh:
        json.dump(res, fh)
    for k, v in summ.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--n-anchor", type=int, default=4)
    ap.add_argument("--n-ev", type=int, default=3)
    ap.add_argument("--n-mds", type=int, default=2)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schemes", default="", help="comma list of schemes to render (default: all)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--mlp-fallback", action="store_true", help="the separately trained MLP conceals the lost anchors the network cannot reach in time")
    ap.add_argument("--frames", type=int, default=1, help="frame periods allowed for the first view of a cold start")
    ap.add_argument("--t-ent", type=float, default=0.0, help="GPU entropy decoding time of 112,000 anchors (ms, both layers)")
    ap.add_argument("--view-nbr", action="store_true", help="concealment neighbours only among the anchors of the view")
    main(ap.parse_args())
