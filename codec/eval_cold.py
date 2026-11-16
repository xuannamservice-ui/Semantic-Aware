"""Cold start with real renders: concealment, in-frame retransmission (ARQ) and replication of the enhancement layer,
on six-slot windows of the temporally correlated Model B failure traces (phy_sim/results/correlated/patterns).

Schedule (slot = 1 ms, frame period 11.1 ms). A full view is sent in slots 1 and 2: the first half of the
Morton-ordered tokens in slot 1, the second half in slot 2, interleaved over the M streams; positions and base layer
travel on the n_A anchor modes with the fewest joint failures in the trace. The scene data that the edge needs before
it can conceal (decoding MLPs and concealment network, 0.52 MB) follow in slot 3 on the anchor modes; if all anchor
modes fail in slot 3 (with ARQ: also in slots 4 and 6) the edge conceals the view with inverse-distance weighting
instead. With ARQ the edge reports the lost tokens one slot after it receives them and the sender resends them, and a
lost anchor payload, in slots 4 and 6 over all streams. rep2 sends every token on a pair of streams (perfect matching
with the fewest joint failures in the trace) in slots 1 to 3 and the scene data in slot 4. If the anchor payload of a
half is lost, the view fails. Decoding and rendering take 3.86 ms (Table 13 of the paper), so gathering and concealment
must fit 11.1 - 3 - 3.86 = 4.24 ms (conceal), 11.1 - 4 - 3.86 = 3.24 ms (rep2 + conceal) or 11.1 - 6 - 3.86 = 1.24 ms
(ARQ + conceal): the concealment network handles L lost anchors in T_FULL(L) (Table 13, interpolated). With --anytime,
if the budget is too short for all lost anchors, the network conceals the lost anchors nearest to the camera that fit
and inverse-distance weighting the rest; the cost of the weighting is counted as one more gathering pass over those
anchors, and the view fails only if gathering alone exceeds the budget. Without --anytime the deployment variant with
cached encodings is used where it was trained, otherwise the view fails.
Strata. Over all six-slot windows of the traces: A = no stream fails in slots 1 to 3 (no scheme loses anything),
B = a failure in slots 1 to 3 but none in slots 4 and 6 (ARQ recovers everything), C = failures in slots 1 to 3 and in
slot 4 or 6. The stratum probabilities come from the complete traces (10^7 slots); for every test view, n_ev windows
of B and of C are drawn uniformly and rendered for every scheme, so that
    Pr(fail) = Pr(B) Pr(fail | B) + Pr(C) Pr(fail | C)
with the conditional probabilities measured on rendered views. Nothing is scaled or extrapolated.
Output: <out>/<scene>/cold<tag>_<cfg>_tau<tau>.json
usage: python eval_cold.py --scene garden --cfg T8R8 --tau 2.5 --n-ev 3 [--anytime --schemes conceal,rep2_conceal,arq_conceal --suffix _any]
"""
import argparse
import glob
import itertools
import json
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
T_FULL = np.array([0, 0.08 + 0.85, 0.17 + 1.81, 0.36 + 3.92, 0.53 + 5.96])       # gathering + concealment, ms (Table 13)
T_EDGE = np.array([0, 0.08 + 0.63, 0.17 + 1.07, 0.36 + 2.24, 0.53 + 3.39])
T_GATHER = np.array([0, 0.08, 0.17, 0.36, 0.53])
T_NET = T_FULL - T_GATHER
T_DEC_REN = 3.86
T_FRAME = 11.1
SCHEMES = ("none", "conceal", "rep2", "rep2_conceal", "arq", "arq_conceal")
PSI_MAX = 99.0


def t_of(L, T):
    return float(np.interp(L, LK, T) if L <= LK[-1] else T[-1] + (L - LK[-1]) * (T[-1] - T[-2]) / (LK[-1] - LK[-2]))


def anytime_count(n_lost, budget):
    """Largest number of lost anchors the network can conceal while inverse-distance weighting handles the rest."""
    lo, hi = 0, n_lost
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if t_of(n_lost, T_GATHER) + t_of(n_lost - mid, T_GATHER) + t_of(mid, T_NET) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def pairing(left, acc, cost, jf):
    """Perfect matching of the streams with the fewest joint failures (exhaustive, M <= 8)."""
    if not left:
        return cost, acc
    a0 = left[0]
    return min(pairing([x for x in left if x not in (a0, b0)], acc + [(a0, b0)], cost + jf[a0, b0], jf) for b0 in left[1:])


def main(a):
    rng = np.random.default_rng(a.seed)
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = LayeredConcealer(model, L, nets["gen_fp"], stats, nbr, ck["radius"])
    conc_edge = LayeredConcealer(model, L, nets["edge_fp"], stats, nbr, ck["radius"]) if ("edge_fp" in nets and not a.anytime) else None
    schemes = tuple(a.schemes.split(",")) if a.schemes else SCHEMES
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx = cf["tx"]; M = len(tx)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
    xs = []
    for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{a.tau:g}ms_pat*.npz"))):
        z = np.load(f); assert [int(x) for x in z[f"{a.cfg}_tx"]] == [int(x) for x in tx]; xs.append(z[a.cfg])
    pats = np.concatenate(xs)                                                        # (chains, slots), bit i = tx[i] failed
    bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
    flat = bits.reshape(-1, M)
    jf = flat.T.astype(np.int64) @ flat.astype(np.int64)
    anchor = min(itertools.combinations(range(M), a.n_anchor), key=lambda s: int(flat[:, list(s)].all(1).sum()))
    anchor_streams = sorted(rel.index(int(tx[i])) for i in anchor)
    pairs = pairing(list(range(M)), [], 0, jf)[1]
    pair_of = {}
    for p_, q_ in pairs:
        sp, sq = rel.index(int(tx[p_])), rel.index(int(tx[q_])); pair_of[sp] = sq; pair_of[sq] = sp
    # strata over the six-slot windows
    anyf = bits.any(-1); nW = anyf.shape[1] - 6
    l123 = anyf[:, :nW] | anyf[:, 1:nW + 1] | anyf[:, 2:nW + 2]
    f46 = anyf[:, 3:nW + 3] | anyf[:, 5:nW + 5]
    strata = {"B": l123 & ~f46, "C": l123 & f46}
    p_strata = {"A": float((~l123).mean()), **{k: float(v.mean()) for k, v in strata.items()}}
    cand = {k: np.flatnonzero(v.ravel()) for k, v in strata.items()}
    dev = "cuda"

    def failed(ch, s, t):
        return set(patterns_to_streams(int(pats[ch, s + t]), tx, rel))

    def idw(lost, avail):
        nb = nbr[lost]; r = avail[nb].float()
        w = r / (model.xyz[nb] - model.xyz[lost][:, None]).norm(dim=-1).clamp_min(1e-6)
        m_ = (w[..., None] * enh[nb]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
        return torch.where((r.sum(1) > 0)[:, None], m_, stats[2].expand_as(m_))

    rows, t0 = [], time.time()
    with torch.no_grad():
        for vi in scene.test_idx:
            vi = int(vi)
            gt, K, V = scene.view(vi)
            idx = model.visible(K, V, scene.W, scene.H)
            full, _, _ = model.render(at, idx, K, V, scene.W, scene.H)
            s_, tid = pk.token_ids(idx); T = int(tid[-1]) + 1
            ar = torch.arange(T, device=dev)
            slot_of = torch.where(ar < (T + 1) // 2, 0, 1)[tid]                 # slot of every anchor's token (two halves)
            third_of = ((ar * 3) // T)[tid]                                     # replication: three slots
            n_vis = int(idx.numel())
            events = [(k, divmod(int(cand[k][i]), nW)) for k in ("B", "C") if cand[k].size
                      for i in rng.choice(cand[k].size, min(a.n_ev, cand[k].size), replace=False)]
            for kind, (ch, s) in events:
                prng = np.random.default_rng(vi * 7919 + ch * 251 + s)
                St = [failed(ch, s, t) for t in range(6)]
                fs = [torch.tensor([m in St[t] for m in range(M)], device=dev) for t in range(6)]
                aset = [all(x in St[t] for x in anchor_streams) for t in range(6)]
                tok_stream = pk.assign(s_, tid, M, "interleaved", prng)[tid]
                for sch in schemes:
                    if sch.startswith("rep2"):
                        pf = [torch.tensor([(m in St[t]) and (pair_of[m] in St[t]) for m in range(M)], device=dev) for t in range(3)]
                        fail = torch.zeros_like(tok_stream, dtype=torch.bool)
                        for t in range(3):
                            fail |= (third_of == t) & pf[t][tok_stream]
                        anchor_lost = any(aset[:3]); net_lost = aset[3]; budget = T_FRAME - 4 - T_DEC_REN
                    else:
                        fail = ((slot_of == 0) & fs[0][tok_stream]) | ((slot_of == 1) & fs[1][tok_stream])
                        anchor_lost = aset[0] or aset[1]; net_lost = aset[2]; budget = T_FRAME - 3 - T_DEC_REN
                    if sch.startswith("arq"):
                        if kind == "B":                                         # no failure in slots 4 and 6: everything recovered
                            fail[:] = False; anchor_lost = False; net_lost = False
                        else:
                            for t in (3, 5):
                                if anchor_lost and not aset[t]:
                                    anchor_lost = False
                                if net_lost and not aset[t]:
                                    net_lost = False
                                if fail.any():
                                    ut, inv = torch.unique(tid[fail], return_inverse=True)
                                    ns = torch.as_tensor(prng.integers(0, M, int(ut.numel())), device=dev)[inv]
                                    fail[fail.clone()] = fs[t][ns]
                        budget = T_FRAME - 6 - T_DEC_REN
                    lost = s_[fail]; n_lost = int(lost.numel())
                    rec = {"view": vi, "stratum": kind, "scheme": sch, "n_visible": n_vis, "n_lost": n_lost,
                           "anchor_lost": bool(anchor_lost), "net_lost": bool(net_lost), "deadline_miss": False, "variant": "full"}
                    if anchor_lost:
                        rec.update(psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); rows.append(rec); continue
                    e = enh.clone(); ridx = idx
                    if n_lost and sch.endswith("conceal"):
                        avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device=dev); avail[s_[~fail]] = True
                        if net_lost:
                            e[lost] = idw(lost, avail); rec["variant"] = "idw"
                        elif t_of(n_lost, T_FULL) <= budget:
                            e[lost] = conc.predict(lost, avail, base, enh)
                        elif a.anytime and 2 * t_of(n_lost, T_GATHER) <= budget:
                            n_net = anytime_count(n_lost, budget)
                            depth = (V[:3, :3] @ model.xyz[lost].T + V[:3, 3:]).T[:, 2]
                            order = torch.argsort(depth)                        # nearest to the camera first
                            sub, rest = lost[order[:n_net]], lost[order[n_net:]]
                            e[rest] = idw(rest, avail)
                            if n_net:
                                e[sub] = conc.predict(sub, avail, base, enh)
                            rec.update(variant="anytime", n_net=n_net)
                        elif conc_edge is not None and t_of(n_lost, T_EDGE) <= budget:
                            e[lost] = conc_edge.predict(lost, avail, base, enh); rec["variant"] = "edge"
                        else:
                            rec.update(deadline_miss=True, psnr=0.0, psi=0.0, ssim=0.0, lpips=1.0); rows.append(rec); continue
                    elif n_lost:
                        ridx = s_[~fail]
                    if ridx.numel():
                        img, _, _ = model.render(L.merge(base, e), ridx, K, V, scene.W, scene.H)
                    else:
                        img = torch.zeros_like(full)
                    m = metrics.all_metrics(img, gt)
                    rec.update(**m, psi=min(metrics.psnr(img.clamp(0, 1), full.clamp(0, 1)), PSI_MAX)); rows.append(rec)
            print(f"view {vi}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for kind in ("B", "C"):
        for sch in SCHEMES:
            rs = [x for x in rows if x["scheme"] == sch and x["stratum"] == kind]
            if not rs:
                continue
            psi = np.array([x["psi"] for x in rs]); ok = [x for x in rs if not (x["anchor_lost"] or x["deadline_miss"])]
            summ[f"{kind}/{sch}"] = {"n": len(rs), "k30": int((psi < 30).sum()), "k35": int((psi < 35).sum()),
                                     "anchor_lost": int(sum(x["anchor_lost"] for x in rs)), "net_lost": int(sum(x["net_lost"] for x in rs)),
                                     "deadline_miss": int(sum(x["deadline_miss"] for x in rs)),
                                     "edge_variant": int(sum(x["variant"] == "edge" for x in rs)), "idw_fallback": int(sum(x["variant"] == "idw" for x in rs)),
                                     "anytime": int(sum(x["variant"] == "anytime" for x in rs)),
                                     "net_share_anytime": float(np.mean([x["n_net"] / x["n_lost"] for x in rs if x["variant"] == "anytime"])) if any(x["variant"] == "anytime" for x in rs) else None,
                                     "psi_mean": float(np.mean([x["psi"] for x in ok])) if ok else None,
                                     "lost_frac_mean": float(np.mean([x["n_lost"] / x["n_visible"] for x in rs]))}
    res = {"scene": a.scene, "cfg": a.cfg, "tau_c_ms": a.tau, "n_anchor": a.n_anchor, "anchor_modes": [int(tx[i]) for i in anchor],
           "pairs": [[int(tx[p]), int(tx[q])] for p, q in pairs], "p_strata": p_strata, "n_windows": int(nW * anyf.shape[0]),
           "n_ev": a.n_ev, "has_edge_variant": conc_edge is not None, "anytime": bool(a.anytime), "schemes": list(schemes),
           "summary": summ, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"cold{a.tag}{a.suffix}_{a.cfg}_tau{a.tau:g}.json"), "w") as fh:
        json.dump(res, fh)
    print("strata", p_strata, flush=True)
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
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anytime", action="store_true", help="network on the nearest lost anchors that fit, IDW on the rest")
    ap.add_argument("--schemes", default="", help="comma-separated subset of the schemes (same windows: the draws do not depend on it)")
    ap.add_argument("--suffix", default="", help="appended to the tag in the output file name")
    main(ap.parse_args())
