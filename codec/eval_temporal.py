"""Temporal quality along the frame paths of eval_frames.py: flicker, popping and recovery after outage bursts.

Same paths, losses and schemes as eval_frames.py (T8R8, base layer on n_A = 4 modes, interleaved mapping, Model-B traces
at sigma_R^2 = 2.5 and tau_c = 10 and 2.5 ms, chains of the same seeds): none, conceal, arq and arq+conceal, where ARQ
retransmits in slots 3 and 5 on the streams that did not fail in the previous transmission slot, within eps = 1/6 of the
frame payload (the ARQ of Table tab:frames). Concealment uses the cross-attention network of the paper (gen_fp). Every
frame of the path is evaluated (not only the held-out test views), so that consecutive frames are 1/90 s apart:
  f_t   rendering at the edge, g_t loss-free rendering, y_t captured image of the view
  Psi_t      = PSNR(f_t, g_t)                      (infinite when nothing is missing)
  dPSNR_t    = PSNR(f_t, y_t) - PSNR(g_t, y_t)     (quality loss against the loss-free rendering, dB)
  tLPIPS_t   = |LPIPS(f_t, f_{t-1}) - LPIPS(g_t, g_{t-1})|   (LPIPS-VGG; 0 when neither frame has missing data)
  tPSNR_t    = PSNR(f_t - f_{t-1}, g_t - g_{t-1})  (frame differences, peak 1; infinite when neither frame is affected)
Popping: an anchor that is concealed (or missing) in frame t-1 and decoded in frame t changes abruptly; tLPIPS and tPSNR
at the frame where the missing anchors arrive measure it.
Bursts: maximal runs of consecutive outage slots (some stream failed) of the chain window; a burst covers frames f0..f1
(the frames of its first and last slot). Recovery after a burst = number of frames after f1 that are still degraded
(dPSNR < -0.5 dB) before the first frame within 0.5 dB of the loss-free rendering; exact recovery = frames after f1
until no anchor is missing. Only bursts that leave a frame with missing data in f0..f1 + 1 are counted.
Burst injection (--inject): the frame paths contain few long bursts (a path window is 3% of the trace), so the longest
runs of outage slots within the 250-slot chains of the whole trace (--n-inject per coherence time, up to 41 slots at
tau_c = 10 ms) are inserted, with their per-slot failure patterns, at the first slot of a frame at --n-pos positions
along an otherwise loss-free path (cache filled by the loss-free history); every scheme is then followed until no
anchor is missing. Reported per burst and position: frames with missing data, recovery frames after the burst, minimum
Psi and dPSNR, maximum tLPIPS. Output <out>/<scene>/temporal_inject_<cfg>.json.
Output: <out>/<scene>/temporal_<cfg>.json (per-frame series) and with --aggregate the pooled summary.
usage: python eval_temporal.py --scene garden [--chains 10]
       python eval_temporal.py --scene garden --inject
       python eval_temporal.py --aggregate --out-json results/rev6_temporal_T8R8.json
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import metrics
from conceal_net import knn
from eval_frames import pattern_stream
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_layered import LayeredConcealer

INF = 99.0


def psnr(a, b):
    m = float(F.mse_loss(a, b))
    return INF if m == 0 else float(-10 * np.log10(m))


def bursts(pats_flat):
    """Maximal runs of non-zero patterns: list of (first slot, last slot)."""
    out, s = [], None
    for i, p in enumerate(pats_flat):
        if p and s is None:
            s = i
        elif not p and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(pats_flat) - 1))
    return out


def main(a):
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = LayeredConcealer(model, L, nets[a.net], stats, nbr, ck["radius"])
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx_ms = cf["tx"]; M = len(tx_ms)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx_ms)
    order = [int(v) for v in np.argsort(scene.names)]; T = len(order); Fs = a.slots_per_frame
    W, H = scene.W, scene.H
    cams = {vi: (torch.from_numpy(scene.K[vi]).cuda(), torch.from_numpy(scene.viewmats[vi]).cuda()) for vi in order}
    vis = {vi: model.visible(*cams[vi], W, H) for vi in order}
    t0 = time.time()

    def render_g(vi):
        return model.render(at, vis[vi], *cams[vi], W, H)[0].clamp(0, 1)

    def gt(vi):
        return torch.from_numpy(scene.images[vi]).cuda().float() / 255.0

    # loss-free reference along the path
    with torch.no_grad():
        g_psnr, g_lp = [], [None]
        prev = None
        for f, vi in enumerate(order):
            g = render_g(vi); g_psnr.append(psnr(g, gt(vi)))
            if prev is not None:
                g_lp.append(metrics.lpips_vgg(g, prev))
            prev = g
    print(f"reference: {time.time() - t0:.0f}s", flush=True)
    res = {"scene": a.scene, "tag": a.tag, "cfg": a.cfg, "n_anchor": a.n_anchor, "net": a.net, "chains": a.chains, "frames": T, "eps": a.eps,
           "arq_last_slot": a.arq_last_slot, "g_psnr_gt": g_psnr, "g_lpips_consecutive": g_lp, "runs": {}}
    for tau in a.taus:
        stream, tx = pattern_stream(a.phy, a.cfg, tau)
        assert [int(x) for x in tx] == [int(x) for x in tx_ms]
        best = min(itertools.combinations(range(M), a.n_anchor),
                   key=lambda sub: int(np.count_nonzero((stream & sum(1 << i for i in sub)) == sum(1 << i for i in sub))))
        anchor_modes = [rel.index(int(tx[i])) for i in best]
        failed = lambda p: set(patterns_to_streams(p, tx, rel))
        rng = np.random.default_rng(a.seed)
        for ch in range(a.chains):
            off = rng.integers(0, stream.size - T * Fs)
            pats = stream[off:off + T * Fs].reshape(T, Fs)
            bl = bursts(pats.reshape(-1).tolist())
            for proto in ("plain", "arq"):
                cached = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                prng = np.random.default_rng(a.seed * 1000 + ch)
                miss_sets = []
                with torch.no_grad():
                    for f, vi in enumerate(order):                     # cache simulation of eval_frames.py
                        idx = vis[vi]; new = idx[~cached[idx]]
                        if new.numel():
                            S0 = failed(pats[f, 0])
                            if all(s in S0 for s in anchor_modes):
                                lost = new
                            else:
                                lm = pk.lost_mask(new, M, a.policy, sorted(S0), prng) if S0 else torch.zeros(new.numel(), dtype=torch.bool, device="cuda")
                                lost = new[lm]
                                if proto == "arq" and lost.numel():
                                    budget = int(a.eps * new.numel()); prev_ = S0
                                    for s in range(2, min(Fs, a.arq_last_slot), 2):
                                        if lost.numel() == 0 or budget <= 0:
                                            break
                                        send = lost[:budget]; budget -= send.numel()
                                        Ss = failed(pats[f, s])
                                        ok_ = [m_ for m_ in range(M) if m_ not in prev_] or list(range(M))
                                        fl_ = sorted(ok_.index(m_) for m_ in Ss if m_ in ok_)
                                        lm2 = pk.lost_mask(send, len(ok_), a.policy, fl_, prng) if fl_ else torch.zeros(send.numel(), dtype=torch.bool, device="cuda")
                                        prev_ = Ss
                                        lost = torch.cat([send[lm2], lost[send.numel():]])
                            delivered = new[~torch.isin(new, lost)] if lost.numel() else new
                            cached[delivered] = True
                        miss = idx[~cached[idx]]
                        miss_sets.append((miss.clone(), cached.clone() if miss.numel() else None))
                    for sch in ((("none", "conceal") if proto == "plain" else ("arq", "arq+conceal"))):
                        nm, psi, dps, tlp, tps = [], [], [], [], []
                        prev_f, prev_g, prev_aff = None, None, False
                        for f, vi in enumerate(order):
                            miss, cch = miss_sets[f]
                            aff = miss.numel() > 0
                            if aff or prev_aff:
                                g = render_g(vi)
                                if aff:
                                    idx = vis[vi]
                                    if sch.endswith("conceal"):
                                        e = enh.clone(); e[miss] = conc.predict(miss, cch, base, enh)
                                        img = model.render(L.merge(base, e), idx, *cams[vi], W, H)[0].clamp(0, 1)
                                    else:
                                        ridx = idx[cch[idx]]
                                        img = model.render(at, ridx, *cams[vi], W, H)[0].clamp(0, 1) if ridx.numel() else torch.zeros_like(g)
                                else:
                                    img = g
                                if f > 0:
                                    pf = prev_f if prev_f is not None else render_g(order[f - 1])
                                    pg = prev_g if prev_g is not None else render_g(order[f - 1])
                                    tlp.append(abs(metrics.lpips_vgg(img, pf) - g_lp[f]))
                                    tps.append(psnr(img - pf, g - pg))
                                else:
                                    tlp.append(0.0); tps.append(INF)
                                psi.append(psnr(img, g) if aff else INF)
                                dps.append(psnr(img, gt(vi)) - g_psnr[f] if aff else 0.0)
                                prev_f, prev_g = (img, g) if aff else (None, None)
                                if not aff:
                                    prev_f = prev_g = None
                            else:
                                tlp.append(0.0); tps.append(INF); psi.append(INF); dps.append(0.0); prev_f = prev_g = None
                            nm.append(int(miss.numel())); prev_aff = aff
                        # recovery after each burst
                        rec = []
                        degr = np.array(dps) < -a.recov_db; missing = np.array(nm) > 0
                        for s0, s1 in bl:
                            f0, f1 = s0 // Fs, s1 // Fs
                            if not missing[f0:min(T, f1 + 2)].any():
                                continue
                            r_q = 0
                            while f1 + 1 + r_q < T and degr[f1 + 1 + r_q]:
                                r_q += 1
                            r_x = 0
                            while f1 + 1 + r_x < T and missing[f1 + 1 + r_x]:
                                r_x += 1
                            seg = slice(f0, min(T, f1 + 2 + r_q))
                            rec.append({"burst_slots": s1 - s0 + 1, "f0": f0, "f1": f1, "recovery_frames": r_q, "exact_recovery_frames": r_x,
                                        "min_psi": float(min(psi[seg])), "min_dpsnr": float(min(dps[seg])), "max_tlpips": float(max(tlp[seg])),
                                        "first_slot_hit": bool(any(pats[f, 0] for f in range(f0, f1 + 1)))})
                        res["runs"].setdefault(f"{tau:g}/{sch}", []).append({"chain": ch, "n_missing": nm, "psi": psi, "dpsnr": dps, "tlpips": tlp,
                                                                             "tpsnr": tps, "bursts": rec, "max_burst_slots": max((b[1] - b[0] + 1 for b in bl), default=0)})
                print(f"tau {tau:g} chain {ch} {proto}: {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, a.scene, f"temporal_{a.cfg}{a.suffix}.json"), "w") as fh:
        json.dump(res, fh)


def longest_bursts(phy, cfg, tau, n):
    """The n longest runs of outage slots within the chains of the trace: list of (length, per-slot patterns)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz"))):
        z = np.load(f)[cfg]
        nz = z != 0
        cand = np.flatnonzero(nz.sum(1) >= 2)
        for r in cand:
            for s0, s1 in bursts(nz[r].tolist()):
                runs.append((s1 - s0 + 1, z[r, s0:s1 + 1].copy()))
        runs.sort(key=lambda x: -x[0]); runs = runs[:n]
    return runs


def inject(a):
    scene = Scene(*scene_location(a.scene, a.data))
    model, L, nets, c, ck = load(a.out, a.scene, a.tag)
    with torch.no_grad():
        at = model.attributes(quantized=True); base, enh = L.split(at)
        stats = (base.mean(0), base.std(0).clamp_min(0.02), enh.mean(0), enh.std(0).clamp_min(0.02))
    nbr = knn(model.xyz, c["kn"])
    conc = LayeredConcealer(model, L, nets[a.net], stats, nbr, ck["radius"])
    pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][a.cfg]; tx_ms = cf["tx"]; M = len(tx_ms)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx_ms)
    order = [int(v) for v in np.argsort(scene.names)]; T = len(order); Fs = a.slots_per_frame
    W, H = scene.W, scene.H
    cams = {vi: (torch.from_numpy(scene.K[vi]).cuda(), torch.from_numpy(scene.viewmats[vi]).cuda()) for vi in order}
    vis = {vi: model.visible(*cams[vi], W, H) for vi in order}
    gt = lambda vi: torch.from_numpy(scene.images[vi]).cuda().float() / 255.0
    render_g = lambda vi: model.render(at, vis[vi], *cams[vi], W, H)[0].clamp(0, 1)
    positions = [int(T * (i + 0.5) / a.n_pos) for i in range(a.n_pos)]
    res = {"scene": a.scene, "cfg": a.cfg, "net": a.net, "positions": positions, "frames": T, "runs": []}
    t0 = time.time()
    for tau, n_inj in zip(a.taus, a.n_inject):
        stream, tx = pattern_stream(a.phy, a.cfg, tau)
        best = min(itertools.combinations(range(M), a.n_anchor),
                   key=lambda sub: int(np.count_nonzero((stream & sum(1 << i for i in sub)) == sum(1 << i for i in sub))))
        anchor_modes = [rel.index(int(tx[i])) for i in best]
        del stream
        failed = lambda p_: set(patterns_to_streams(int(p_), tx, rel))
        for bi, (blen, seg) in enumerate(longest_bursts(a.phy, a.cfg, tau, n_inj)):
            for f_inj in positions:
                n_fr = min(T - f_inj, (blen + Fs - 1) // Fs + a.max_after)
                pats = np.zeros((n_fr, Fs), np.int64); pats.reshape(-1)[:blen] = seg
                f1 = (blen - 1) // Fs                                # last frame (relative) touched by the burst
                with torch.no_grad():
                    for proto in ("plain", "arq"):
                        cached = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                        for f in range(f_inj):
                            cached[vis[order[f]]] = True
                        prng = np.random.default_rng(a.seed)
                        miss_sets = []
                        for j in range(n_fr):
                            idx = vis[order[f_inj + j]]; new = idx[~cached[idx]]
                            if new.numel():
                                S0 = failed(pats[j, 0])
                                if all(s_ in S0 for s_ in anchor_modes):
                                    lost = new
                                else:
                                    lm = pk.lost_mask(new, M, a.policy, sorted(S0), prng) if S0 else torch.zeros(new.numel(), dtype=torch.bool, device="cuda")
                                    lost = new[lm]
                                    if proto == "arq" and lost.numel():
                                        budget = int(a.eps * new.numel()); prev_ = S0
                                        for s_ in range(2, min(Fs, a.arq_last_slot), 2):
                                            if lost.numel() == 0 or budget <= 0:
                                                break
                                            send = lost[:budget]; budget -= send.numel()
                                            Ss = failed(pats[j, s_])
                                            ok_ = [m_ for m_ in range(M) if m_ not in prev_] or list(range(M))
                                            fl_ = sorted(ok_.index(m_) for m_ in Ss if m_ in ok_)
                                            lm2 = pk.lost_mask(send, len(ok_), a.policy, fl_, prng) if fl_ else torch.zeros(send.numel(), dtype=torch.bool, device="cuda")
                                            prev_ = Ss
                                            lost = torch.cat([send[lm2], lost[send.numel():]])
                                delivered = new[~torch.isin(new, lost)] if lost.numel() else new
                                cached[delivered] = True
                            miss = idx[~cached[idx]]
                            miss_sets.append((miss.clone(), cached.clone() if miss.numel() else None))
                            if j > f1 and miss.numel() == 0:
                                break
                        for sch in ((("none", "conceal") if proto == "plain" else ("arq", "arq+conceal"))):
                            nm, psi, dps, tlp = [], [], [], []
                            prev_f = render_g(order[f_inj - 1]) if f_inj > 0 else None
                            prev_g = prev_f
                            for j, (miss, cch) in enumerate(miss_sets + [(torch.zeros(0, dtype=torch.long, device="cuda"), None)]):
                                if f_inj + j >= T:
                                    break
                                vi = order[f_inj + j]; g = render_g(vi); aff = miss.numel() > 0
                                if aff:
                                    idx = vis[vi]
                                    if sch.endswith("conceal"):
                                        e = enh.clone(); e[miss] = conc.predict(miss, cch, base, enh)
                                        img = model.render(L.merge(base, e), idx, *cams[vi], W, H)[0].clamp(0, 1)
                                    else:
                                        ridx = idx[cch[idx]]
                                        img = model.render(at, ridx, *cams[vi], W, H)[0].clamp(0, 1) if ridx.numel() else torch.zeros_like(g)
                                else:
                                    img = g
                                y = gt(vi)
                                nm.append(int(miss.numel())); psi.append(psnr(img, g) if aff else INF)
                                dps.append(psnr(img, y) - psnr(g, y) if aff else 0.0)
                                tlp.append(abs(metrics.lpips_vgg(img, prev_f) - metrics.lpips_vgg(g, prev_g)) if prev_f is not None and (aff or j > 0) else 0.0)
                                prev_f, prev_g = img, g
                            degr = [d < -a.recov_db for d in dps]; missing = [n_ > 0 for n_ in nm]
                            r_q = 0
                            while f1 + 1 + r_q < len(degr) and degr[f1 + 1 + r_q]:
                                r_q += 1
                            r_x = 0
                            while f1 + 1 + r_x < len(missing) and missing[f1 + 1 + r_x]:
                                r_x += 1
                            res["runs"].append({"tau": tau, "burst": bi, "burst_slots": int(blen), "frame": f_inj, "scheme": sch,
                                                "frames_with_missing": int(sum(missing)), "frames_degraded": int(sum(degr)),
                                                "recovery_frames": r_q, "exact_recovery_frames": r_x, "min_psi": float(min(psi)),
                                                "min_dpsnr": float(min(dps)), "max_tlpips": float(max(tlp)), "n_missing": nm, "psi": psi,
                                                "dpsnr": dps, "tlpips": tlp})
            print(f"tau {tau:g} burst {bi} ({blen} slots): {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, a.scene, f"temporal_inject_{a.cfg}{a.suffix}.json"), "w") as fh:
        json.dump(res, fh)


def p(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def aggregate(a):
    files = sorted(glob.glob(os.path.join(a.out, "*", f"temporal_{a.cfg}{a.suffix}.json")))
    S = [json.load(open(f)) for f in files]
    out = {"scenes": [s["scene"] for s in S], "cfg": a.cfg, "recovery_threshold_db": a.recov_db, "summary": {}, "worst_burst_per_scene": {}}
    keys = sorted({k for s in S for k in s["runs"]})
    for k in keys:
        nm, psi, dps, tlp, tps, rec, aff_tlp, aff_tps = [], [], [], [], [], [], [], []
        for s in S:
            for r in s["runs"].get(k, []):
                nm += r["n_missing"]; psi += r["psi"]; dps += r["dpsnr"]; tlp += r["tlpips"]; tps += r["tpsnr"]; rec += r["bursts"]
                for i in range(len(r["n_missing"])):
                    if r["n_missing"][i] > 0 or (i > 0 and r["n_missing"][i - 1] > 0):
                        aff_tlp.append(r["tlpips"][i]); aff_tps.append(r["tpsnr"][i])
        nm, psi, dps, tlp = map(np.array, (nm, psi, dps, tlp))
        aff = nm > 0
        rq = np.array([b["recovery_frames"] for b in rec]); rx = np.array([b["exact_recovery_frames"] for b in rec])
        out["summary"][k] = {
            "frames": int(nm.size), "frames_with_missing": int(aff.sum()),
            "psi_affected": {"mean": float(np.mean(psi[aff])) if aff.any() else None, "p1": p(psi[aff], 1), "min": float(psi[aff].min()) if aff.any() else None},
            "dpsnr_affected": {"mean": float(np.mean(dps[aff])) if aff.any() else None, "p1": p(dps[aff], 1)},
            "frames_dpsnr_below_-0.5": int((dps < -a.recov_db).sum()),
            "tlpips_all": {"mean": float(tlp.mean()), "p99": p(tlp, 99), "p999": p(tlp, 99.9), "max": float(tlp.max())},
            "tlpips_affected": {"n": len(aff_tlp), "mean": float(np.mean(aff_tlp)) if aff_tlp else None, "p99": p(aff_tlp, 99), "max": max(aff_tlp) if aff_tlp else None},
            "tpsnr_affected": {"n": len(aff_tps), "mean": float(np.mean(aff_tps)) if aff_tps else None, "p1": p(aff_tps, 1), "min": min(aff_tps) if aff_tps else None},
            "bursts_with_impact": len(rec),
            "recovery_frames": {"mean": float(rq.mean()) if rq.size else None, "p99": p(rq, 99), "max": int(rq.max()) if rq.size else None},
            "exact_recovery_frames": {"mean": float(rx.mean()) if rx.size else None, "p99": p(rx, 99), "max": int(rx.max()) if rx.size else None},
            "burst_slots_with_impact": {"mean": float(np.mean([b["burst_slots"] for b in rec])) if rec else None, "max": max((b["burst_slots"] for b in rec), default=None)}}
    for s in S:
        for k in keys:
            rs = s["runs"].get(k, [])
            cand = [(b["burst_slots"], r["chain"], b) for r in rs for b in r["bursts"]]
            longest = max((r["max_burst_slots"] for r in rs), default=0)
            if cand:
                bs, ch, b = max(cand, key=lambda x: (x[0], -x[2]["min_dpsnr"]))
                out["worst_burst_per_scene"].setdefault(s["scene"], {})[k] = {"longest_burst_slots_any": longest, "worst_impact_burst_slots": bs, "chain": ch, **b}
            else:
                out["worst_burst_per_scene"].setdefault(s["scene"], {})[k] = {"longest_burst_slots_any": longest, "worst_impact_burst_slots": None}
    inj = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out, "*", f"temporal_inject_{a.cfg}{a.suffix}.json")))]
    if inj:
        out["inject"] = {"scenes": [s["scene"] for s in inj], "summary": {}, "longest_burst_per_scene": {}}
        rows = [dict(r, scene=s["scene"]) for s in inj for r in s["runs"]]
        for k in sorted({(r["tau"], r["scheme"]) for r in rows}):
            rs = [r for r in rows if (r["tau"], r["scheme"]) == k]
            f = lambda n: np.array([r[n] for r in rs], float)
            out["inject"]["summary"][f"{k[0]:g}/{k[1]}"] = {
                "cases": len(rs), "burst_slots": {"min": int(f("burst_slots").min()), "max": int(f("burst_slots").max())},
                "frames_with_missing": {"mean": float(f("frames_with_missing").mean()), "max": int(f("frames_with_missing").max())},
                "frames_degraded": {"mean": float(f("frames_degraded").mean()), "p99": p(f("frames_degraded"), 99), "max": int(f("frames_degraded").max())},
                "recovery_frames": {"mean": float(f("recovery_frames").mean()), "p99": p(f("recovery_frames"), 99), "max": int(f("recovery_frames").max())},
                "exact_recovery_frames": {"mean": float(f("exact_recovery_frames").mean()), "p99": p(f("exact_recovery_frames"), 99), "max": int(f("exact_recovery_frames").max())},
                "min_psi": {"mean": float(np.minimum(f("min_psi"), INF).mean()), "p1": p(f("min_psi"), 1), "min": float(f("min_psi").min())},
                "min_dpsnr": {"mean": float(f("min_dpsnr").mean()), "p1": p(f("min_dpsnr"), 1), "min": float(f("min_dpsnr").min())},
                "max_tlpips": {"mean": float(f("max_tlpips").mean()), "p99": p(f("max_tlpips"), 99), "max": float(f("max_tlpips").max())},
                "cases_psi_below_30": int((f("min_psi") < 30).sum()), "cases_psi_below_35": int((f("min_psi") < 35).sum())}
        for s in inj:
            for tau in sorted({r["tau"] for r in s["runs"]}):
                rs = [r for r in s["runs"] if r["tau"] == tau]; mx = max(r["burst_slots"] for r in rs)
                out["inject"]["longest_burst_per_scene"].setdefault(s["scene"], {})[f"{tau:g}"] = {
                    sch: {"burst_slots": mx, "worst_min_psi": min(r["min_psi"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx),
                          "worst_min_dpsnr": min(r["min_dpsnr"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx),
                          "max_recovery_frames": max(r["recovery_frames"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx),
                          "max_exact_recovery_frames": max(r["exact_recovery_frames"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx),
                          "max_frames_with_missing": max(r["frames_with_missing"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx),
                          "max_tlpips": max(r["max_tlpips"] for r in rs if r["scheme"] == sch and r["burst_slots"] == mx)}
                    for sch in ("none", "conceal", "arq", "arq+conceal")}
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.get("inject", {}).get("summary", {}).items():
        print("inject", k, json.dumps(v))
    for k, v in out["summary"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden")
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--taus", type=float, nargs="*", default=[10.0, 2.5])
    ap.add_argument("--eps", type=float, default=1 / 6)
    ap.add_argument("--n-anchor", type=int, default=4)
    ap.add_argument("--slots-per-frame", type=int, default=11)
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--net", default="gen_fp")
    ap.add_argument("--policy", default="interleaved")
    ap.add_argument("--arq-last-slot", type=int, default=5)
    ap.add_argument("--recov-db", type=float, default=0.5)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--inject", action="store_true", help="burst injection along the path (see the docstring)")
    ap.add_argument("--n-inject", type=int, nargs="*", default=[8, 4], help="longest bursts injected per coherence time (same order as --taus)")
    ap.add_argument("--n-pos", type=int, default=5)
    ap.add_argument("--max-after", type=int, default=40, help="frames followed after the burst at most")
    ap.add_argument("--out-json", default="results/rev6_temporal_T8R8.json")
    a = ap.parse_args()
    aggregate(a) if a.aggregate else inject(a) if a.inject else main(a)
