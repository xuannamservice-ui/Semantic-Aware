"""Frame-level evaluation with an edge cache: concealment against erasure coding and in-frame retransmission.

Path: all views of the scene in capture order, one frame per view, frame period = F slots (F = 11, 11.1 ms at
90 frames/s with 1 ms slots). The edge caches received enhancement layers; for frame f the anchors visible in view f
and not cached are packetised (importance mapping, stream m on the m-th most reliable mode) and sent in the first slot
of the frame (token-to-stream mapping: --policy, interleaved by default). Losses: per-slot failure patterns of the temporally correlated simulation (run_traces_corr.py with stored
patterns; chains joined end to end). The base layer and positions travel on n_A replicated modes; if all of them fail in
the first slot the new anchors of the frame are not delivered at all.
Schemes, all with the same redundancy eps (fraction of the frame payload spent on protection):
  none      : lost enhancement data stays missing at the deadline (anchors dropped)
  conceal   : missing enhancement layers are predicted by the concealment network at the deadline
  arq       : the edge reports the lost tokens after one slot; the sender retransmits them in slot 3, 5, ... of the frame
              up to --arq-last-slot (every second slot, round-trip of one slot) while the retransmitted volume stays within
              eps of the frame payload; retransmissions use the same mapping and see the failure pattern of their slot, or
              with --arq-remap are spread over the streams that did not fail in the previous transmission slot
  mds       : an (M, M - r) erasure code across the streams with r = round(eps M): the data of a slot is recovered if at
              most r streams fail, otherwise the tokens on the failed streams are lost (tokens are mapped over all M
              streams in this evaluation, a slight overestimate of the loss of a systematic code)
  arq+conceal, mds+conceal : the same, followed by concealment of what is still missing at the deadline
Anchors not delivered in a frame are sent again in the next frame (they are still not cached).
Quality at the held-out test views: PSNR/SSIM/LPIPS against ground truth and Psi = PSNR against the loss-free rendering.
Output: <out>/<scene>/frames<tag>_<cfg>_tau<tau>_eps<eps>.json
usage: python eval_frames.py --scene garden --tag _v2a_r1 --cfg T8R8 --tau 10 --eps 0.167 --chains 10 [--stride 4]
"""
import argparse
import glob
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


def pattern_stream(phy, cfg, tau):
    xs, tx = [], None
    for f in sorted(glob.glob(os.path.join(phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz"))):
        z = np.load(f)
        xs.append(z[cfg].reshape(-1)); tx = list(z[f"{cfg}_tx"])
    return np.concatenate(xs), tx


def main(a):
    rng = np.random.default_rng(a.seed)
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
    stream, tx = pattern_stream(a.phy, a.cfg, a.tau)
    assert [int(x) for x in tx] == [int(x) for x in tx_ms], "mode order of the pattern traces differs"
    # base layer on the n_A modes with the lowest joint failure frequency in the trace
    import itertools
    best = min(itertools.combinations(range(M), a.n_anchor),
               key=lambda sub: int(np.count_nonzero((stream & sum(1 << i for i in sub)) == sum(1 << i for i in sub))))
    anchor_modes = [rel.index(int(tx[i])) for i in best]       # as stream indices
    r = int(round(a.eps * M))
    order = np.argsort(scene.names)
    # faster camera: every stride-th view of the capture path (test views always kept, so they are evaluated)
    test_set = set(int(i) for i in scene.test_idx)
    order = np.array([v for j, v in enumerate(order) if j % a.stride == 0 or int(v) in test_set])
    T = len(order); F = a.slots_per_frame
    test = set(int(i) for i in scene.test_idx)
    schemes = a.schemes.split(",") if a.schemes else ["none", "conceal", "arq", "arq+conceal", "mds", "mds+conceal"]
    rows, t0 = [], time.time()

    def failed_streams(p):
        return set(patterns_to_streams(p, tx, rel))

    with torch.no_grad():
        full, full_m, vis_cache = {}, {}, {}
        for vi in scene.test_idx:
            gt, K, V = scene.view(vi)
            img, _, _ = model.render(at, model.visible(K, V, scene.W, scene.H), K, V, scene.W, scene.H)
            full[int(vi)] = img; full_m[int(vi)] = metrics.all_metrics(img, gt)   # a frame with nothing missing renders exactly this
        for ch in range(a.chains):
            off = rng.integers(0, stream.size - T * F)
            pats = stream[off:off + T * F].reshape(T, F)
            for sch in schemes:
                cached = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
                prng = np.random.default_rng(a.seed * 1000 + ch)
                for f, vi in enumerate(order):
                    vi = int(vi)
                    if vi not in vis_cache:                     # camera, visible anchors and (test views) image: the same for every chain and scheme
                        gt_, K_, V_ = scene.view(vi)
                        vis_cache[vi] = (gt_.cpu() if vi in test else None, K_, V_, model.visible(K_, V_, scene.W, scene.H).cpu())
                    gt, K, V, idx = vis_cache[vi]
                    idx = idx.cuda(); gt = gt.cuda() if gt is not None else None
                    new = idx[~cached[idx]]
                    if new.numel():
                        S0 = failed_streams(pats[f, 0])
                        if all(s in S0 for s in anchor_modes):          # base layer lost: nothing new delivered
                            lost = new
                        else:
                            if sch.startswith("mds") and len(S0) <= r:
                                S0 = set()
                            lm = pk.lost_mask(new, M, a.policy, sorted(S0), prng) if S0 else \
                                torch.zeros(new.numel(), dtype=torch.bool, device="cuda")
                            lost = new[lm]
                            if sch.startswith("arq") and lost.numel():
                                budget = int(a.eps * new.numel())
                                prev = S0
                                for s in range(2, min(F, a.arq_last_slot), 2):   # retransmission slots 3, 5, ... (index 2, 4, ...)
                                    if lost.numel() == 0 or budget <= 0:
                                        break
                                    send = lost[:budget]; budget -= send.numel()
                                    Ss = failed_streams(pats[f, s])
                                    ok_ = [m_ for m_ in range(M) if m_ not in prev] if a.arq_remap else list(range(M))
                                    ok_ = ok_ or list(range(M))
                                    fl_ = sorted(ok_.index(m_) for m_ in Ss if m_ in ok_)
                                    lm2 = pk.lost_mask(send, len(ok_), a.policy, fl_, prng) if fl_ else \
                                        torch.zeros(send.numel(), dtype=torch.bool, device="cuda")
                                    prev = Ss
                                    lost = torch.cat([send[lm2], lost[send.numel():]])
                        delivered = new[~torch.isin(new, lost)] if lost.numel() else new
                        cached[delivered] = True
                    if vi not in test:
                        continue
                    miss = idx[~cached[idx]]
                    if miss.numel() == 0:
                        rows.append({"chain": ch, "scheme": sch, "view": vi, "frame": f, "n_missing": 0, "n_visible": int(idx.numel()), **full_m[vi], "psi": float("inf")})
                        continue
                    e = enh.clone(); ridx = idx
                    if miss.numel():
                        if sch.endswith("conceal"):
                            e[miss] = conc.predict(miss, cached.clone(), base, enh)
                        else:
                            ridx = idx[cached[idx]]
                    img, _, _ = model.render(L.merge(base, e), ridx, K, V, scene.W, scene.H)
                    m = metrics.all_metrics(img, gt)
                    rows.append({"chain": ch, "scheme": sch, "view": vi, "frame": f, "n_missing": int(miss.numel()),
                                 "n_visible": int(idx.numel()), **m,
                                 "psi": metrics.psnr(img.clamp(0, 1), full[vi].clamp(0, 1))})
            print(f"chain {ch}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for sch in schemes:
        rs = [x for x in rows if x["scheme"] == sch]
        psi = np.array([min(x["psi"], 99.0) for x in rs])
        summ[sch] = {"frames": len(rs), "frames_with_missing": int(sum(x["n_missing"] > 0 for x in rs)),
                     "psnr": float(np.mean([x["psnr"] for x in rs])), "lpips": float(np.mean([x["lpips"] for x in rs])),
                     **{f"p_fail_psi{t}": float((psi < t).mean()) for t in a.psi_th},
                     "psi_min": float(psi.min()), "psi_p01": float(np.percentile(psi, 1))}
    res = {"scene": a.scene, "tag": a.tag, "cfg": a.cfg, "tau_c_ms": a.tau, "eps": a.eps, "r_mds": r, "arq_last_slot": a.arq_last_slot, "arq_remap": bool(a.arq_remap),
           "anchor_modes": [int(tx[i]) for i in best], "policy": a.policy,
           "slots_per_frame": F, "stride": a.stride, "n_anchor_copies": a.n_anchor, "net": a.net, "chains": a.chains, "frames": T,
           "summary": summ, "rows": rows}
    with open(os.path.join(a.out, a.scene, f"frames{a.tag}_{a.cfg}_tau{a.tau:g}_eps{a.eps:g}" + (f"_stride{a.stride}" if a.stride > 1 else "") + a.suffix + ".json"), "w") as fh:
        json.dump(res, fh)
    for k, v in summ.items():
        print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1 / 6)
    ap.add_argument("--n-anchor", type=int, default=2, help="number of modes carrying the base layer")
    ap.add_argument("--slots-per-frame", type=int, default=11)
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1, help="use every stride-th view of the path (faster camera)")
    ap.add_argument("--net", default="gen_fp")
    ap.add_argument("--policy", default="interleaved", choices=("random", "contiguous", "interleaved", "importance"))
    ap.add_argument("--psi-th", type=float, nargs="*", default=[30.0, 35.0])
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arq-last-slot", type=int, default=11, help="last retransmission slot (1-based); 5 keeps rendering within the frame")
    ap.add_argument("--arq-remap", action="store_true", help="retransmit on the streams that did not fail in the previous transmission slot")
    ap.add_argument("--schemes", default="", help="comma-separated subset of the schemes")
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    main(ap.parse_args())
