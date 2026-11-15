"""Outage of the wavelength-reuse link versus turbulence strength (input of the time-varying scheduler scenario).

For each Rytov variance, configurations T8R11 and T6R11 with separate carriers and with four carriers (carrier of mode
l: l mod 4; beat interference as in run_beat.py) on common random numbers. Records the outage and the distribution of
the number of failed streams and the joint failures of every mode pair
(anchor channel).
Output: results/reuse_curve.json
usage: python run_reuse_curve.py 2000000 6
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
CFGS = ("T8R11", "T6R11")
CHUNK = 2_500
_ST = None


def init():
    global _ST
    base = of.OAMChannel(); gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Ur = np.stack([U[m] for m in rm.RX_ALL]); Ut = np.stack([U[m] for m in rm.TX8])
    Q = (np.conj(Ur)[:, None, :] * Ut[None, :, :] * gr["dA"]).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    _ST = dict(base=base, Qr=np.ascontiguousarray(Q.real), Qi=np.ascontiguousarray(Q.imag))


def work(args):
    s2, i, n = args
    base = _ST["base"]
    rng = np.random.default_rng(int(s2 * 1000) * 100_000 + 5101 + i)
    z = rng.standard_normal((n, base.j_list.size))
    I = base.scint(rng, n, s2)
    Phi = base.coeffs(z, s2) @ base.Z
    c, s = np.cos(Phi), np.sin(Phi)
    h = ((c @ _ST["Qr"] - s @ _ST["Qi"]) + 1j * (c @ _ST["Qi"] + s @ _ST["Qr"])).reshape(n, len(rm.RX_ALL), len(rm.TX8))
    g_all = h * np.sqrt(I)[:, None, :]
    out = {}
    for k in CFGS:
        tx, rx = rm.CFG[k]
        g = g_all[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]
        M, Nr = len(tx), len(rx); snr = rm.snr0(M); G = np.abs(g) ** 2
        f_sep = of.sinr_mmse(G, snr) < rm.GTH
        car = [int(l) % 4 for l in tx]
        Kb = np.zeros((n, Nr, Nr))
        for a in range(M):
            for b in range(a + 1, M):
                if car[a] == car[b]:
                    p = g[:, :, a] * np.conj(g[:, :, b])
                    Kb += np.real(p[:, :, None] * np.conj(p[:, None, :]))
        L = np.linalg.cholesky(np.eye(Nr)[None] + 2.0 * snr * Kb)
        f_w4 = of.sinr_mmse(np.linalg.solve(L, G), snr) < rm.GTH
        out[k] = np.bincount(f_sep.sum(1), minlength=9)[:9]
        out[k + "W4"] = np.bincount(f_w4.sum(1), minlength=9)[:9]
        out[k + "_pair"] = f_sep.T.astype(np.int64) @ f_sep.astype(np.int64)      # joint failures of mode pairs
        out[k + "W4_pair"] = f_w4.T.astype(np.int64) @ f_w4.astype(np.int64)
    return s2, out


def main(N, workers):
    t0 = time.time()
    jobs = [(s2, i, min(CHUNK, N - i * CHUNK)) for s2 in S2 for i in range(math.ceil(N / CHUNK))]
    tot = {s2: {} for s2 in S2}
    with Pool(workers, initializer=init) as pool:
        for j, (s2, r) in enumerate(pool.imap_unordered(work, jobs)):
            for k, v in r.items():
                tot[s2][k] = tot[s2][k] + v if k in tot[s2] else v.copy()
            if (j + 1) % 200 == 0:
                print(f"{j + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res = {"N": N, "s2": list(S2), "carrier_of_mode": "l mod 4", "curves": {}}
    for s2 in S2:
        res["curves"][str(s2)] = {}
        for k, v in tot[s2].items():
            if k.endswith("_pair"):
                continue
            cnt = int(N - v[0]); pair = tot[s2][k + "_pair"]
            tx = rm.CFG[k[:5]][0]; M = len(tx)
            iu = [(a, b) for a in range(M) for b in range(a + 1, M)]
            best = min(iu, key=lambda ab: pair[ab])
            res["curves"][str(s2)][k] = {"outage": cnt / N, "count": cnt, "nfail": v.tolist(),
                                         "pair_joint_fail": (pair / N).tolist(),
                                         "best_anchor_pair": [int(tx[best[0]]), int(tx[best[1]])],
                                         "best_anchor_loss": float(pair[best] / N)}
        print(s2, {k: (d["outage"], d["best_anchor_loss"]) for k, d in res["curves"][str(s2)].items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "reuse_curve.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6)
