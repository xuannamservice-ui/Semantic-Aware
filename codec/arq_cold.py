"""Cold start with in-frame retransmission (ARQ) against concealment, on the temporally correlated traces of Model B.

A full view is sent at a cold start in slots 1 and 2 of a frame period (half of the tokens in each, interleaved over the
M streams). The edge reports the lost tokens one slot after it receives them, and the sender resends them in slots 4 and
6, spread over all M streams; slot 8 would end at 8 ms, and decoding and rendering (3.86 ms, Table 13) would miss the
frame period of 11.1 ms. With K_t failed streams in slot t, the lost fraction of the enhancement layer is
    l_1 = (K_1 + K_2) / (2M)               without retransmission,
    l_ARQ = l_1 (K_4 / M) (K_6 / M)        after the two retransmissions (a resent token is lost if its new stream fails).
Quality at a lost fraction l is obtained from the measured outage slots of the codec evaluation by scaling their mean
squared error with the lost fraction, Psi(l) = Psi_r + 10 log10(l_r / l) for every measured slot r (drop for the schemes
without concealment, the concealment network otherwise), and Pr(Psi < 30 dB | l) is the share of the scaled slots below
the threshold. Every slot pair (s, s+1) of a chain with s + 6 inside the chain is a cold-start event.
Replication (rep2): every token of the enhancement layer is sent on two streams, which with four anchor copies takes three
slots (slots 1 to 3, (2 x 48.8 + 4 x 18.1)/8 = 21.3 Mbit per stream on T8R8); the streams are paired by the perfect
matching with the fewest joint failures in the traces, and a token is lost if both streams of its pair fail in its slot,
    l_rep = (1/3) sum_{t=1..3} (pairs with both streams failed in slot t) / (M/2).
rep2 + ARQ resends the remaining tokens once more in slots 5 and 7 (slot 7 ends at 7 ms, 7 + 3.86 < 11.1 ms).
usage: python arq_cold.py <codec results dir> <phy_sim/results/correlated/patterns> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

COD, TR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TH = (30.0, 35.0)
TX8 = (-4, -3, -2, -1, 1, 2, 3, 4); TX6 = (-3, -2, -1, 1, 2, 3)
TXS = {"T8R8": TX8, "T8R11": TX8, "T6R8": TX6, "T6R11": TX6}   # transmitted modes (phy_sim/run_modeset.CFG)


def rows(method):
    r = []
    for f in glob.glob(f"{COD}/*/eval_layered_v2a_r1_T8R8_2.5.json"):
        e = json.load(open(f))
        r += [(x["lost_frac"], x["psnr_vs_full"]) for x in e["rows"] if x["policy"] == "interleaved" and x["method"] == method]
    a = np.array(r)
    return a[:, 0], a[:, 1]


REF = {"drop": rows("drop"), "gen": rows("gen_fp")}


def p_fail(l, method, th):
    """Pr(Psi < th) at lost fraction l (vector), from the scaled measured slots."""
    lr, pr = REF[method]
    out = np.zeros(l.shape)
    nz = l > 0
    if nz.any():
        # Psi_r + 10 log10(lr / l) < th  <=>  l > lr 10^((Psi_r - th)/10)
        lim = np.sort(lr * 10 ** ((pr - th) / 10))
        out[nz] = np.searchsorted(lim, l[nz], side="left") / lim.size
    return out


def best_pairing(F):
    """Perfect matching of the M streams into M/2 pairs with the fewest joint failures (F: slots x M, bool)."""
    M = F.shape[-1]; F = F.reshape(-1, M)
    J = np.zeros((M, M), np.int64)
    for i in range(0, F.shape[0], 1_000_000):
        f = F[i:i + 1_000_000].astype(np.int32); J += f.T @ f
    best = (None, None)

    def rec(left, acc, cost):
        nonlocal best
        if not left:
            if best[0] is None or cost < best[0]:
                best = (cost, list(acc))
            return
        a = left[0]
        for b in left[1:]:
            rec([x for x in left if x not in (a, b)], acc + [(a, b)], cost + J[a, b])

    rec(list(range(M)), [], 0)
    return best[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), c + h]


def main():
    res = {"model": __doc__.split("\n")[0], "retransmission_slots": [4, 6], "configs": {}}
    for tau in ("2.5", "10"):
        files = sorted(glob.glob(f"{TR}/traces_corr_2.5_{tau}ms_pat*.npz"))
        for cfg in ("T8R8", "T6R8", "T8R11", "T6R11"):
            K = []
            for f in files:
                d = np.load(f)
                pat = d[cfg]; M = len(TXS[cfg])
                bits = np.unpackbits(pat[..., None], axis=-1).sum(-1)      # failed streams per slot
                K.append(bits)
            K = np.concatenate(K); M = len(TXS[cfg])
            n_s = K.shape[1]
            B = np.concatenate([np.unpackbits(np.load(f)[cfg][..., None], axis=-1, bitorder="little")[..., :M].astype(bool) for f in files])
            pairs = best_pairing(B)
            PF = np.stack([B[:, :, a] & B[:, :, b] for a, b in pairs], -1).sum(-1)          # failed pairs per slot
            S = range(n_s - 6)
            cat = lambda A, o: np.concatenate([A[:, s + o] for s in S])
            k1, k2, k4, k6 = cat(K, 0), cat(K, 1), cat(K, 3), cat(K, 5)
            l1 = (k1 + k2) / (2 * M); la = l1 * (k4 / M) * (k6 / M)
            lr = (cat(PF, 0) + cat(PF, 1) + cat(PF, 2)) / (3 * M / 2)
            lra = lr * (cat(K, 4) / M) * (cat(K, 6) / M)
            N = l1.size; hit = l1 > 0
            row = {"tau_ms": float(tau), "M": M, "events": int(N), "cold_starts_with_loss": int(hit.sum()),
                   "p_loss": float(hit.mean()), "p_loss_after_arq": float((la > 0).mean()),
                   "mean_lost_frac_given_loss": float(l1[hit].mean()) if hit.any() else 0.0,
                   "pairs": [[TXS[cfg][a], TXS[cfg][b]] for a, b in pairs],
                   "p_loss_rep2": float((lr > 0).mean())}
            for th in TH:
                for name, l, m in (("none", l1, "drop"), ("conceal", l1, "gen"), ("arq", la, "drop"), ("arq_conceal", la, "gen"),
                                   ("rep2", lr, "drop"), ("rep2_conceal", lr, "gen"), ("rep2_arq", lra, "drop"), ("rep2_arq_conceal", lra, "gen")):
                    p = p_fail(l, m, th)
                    row[f"{name}_{th:g}"] = float(p.mean())
                    row[f"{name}_{th:g}_given_loss"] = float(p[hit].mean()) if hit.any() else 0.0
            res["configs"][f"{cfg}_{tau}ms"] = row
            print(cfg, tau, {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
