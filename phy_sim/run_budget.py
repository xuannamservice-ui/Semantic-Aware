"""Combined impairment budget of T6R11 at sigma_R^2 = 2.5: the receiver impairments of run_sens2.py applied together
with laser RIN and extra sorter loss (run_tolerance.py), on common random numbers.

Every case uses a channel estimate from 64 pilot symbols per entry (additive error of variance 1/(64 P)) and intrinsic
sorter crosstalk of -20 dB; the cases differ in the decision threshold (the 7% HD-FEC threshold of G.975.1, or
RS(255,239), bracketed by BER 1e-3 and 1.8e-4) and in RIN and extra loss. Records the outage, the distribution of the
number of failed streams and the joint failures of all mode pairs, from which codec/tol_service.py-style failed-slot
rates follow. Output: results/budget_2.5.json
usage: python run_budget.py 10000000 7
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
from run_sens2 import mmse_filter, sinr_of

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = 2.5; CFG = "T6R11"; NPIL = 64; XT_DB = -20.0
CASES = [("ideal", None, 0.0, None, False), ("hd", None, 0.0, 3.8e-3, True)]
for ber in (1e-3, 1.8e-4):
    for rin, dl in ((None, 0.0), (-165.0, 0.0), (-160.0, 0.0), (None, 1.0), (None, 2.0), (-160.0, 2.0)):
        CASES.append((f"rs{ber:g}_rin{rin}_dl{dl:g}", rin, dl, ber, True))
for rin, dl in ((-165.0, 0.0), (-160.0, 0.0), (None, 2.0), (-160.0, 2.0)):
    CASES.append((f"hd_rin{rin}_dl{dl:g}", rin, dl, 3.8e-3, True))
CHUNK = 2_500
_ST = None


def init():
    global _ST
    base = of.OAMChannel(); gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Q = (np.conj(np.stack([U[m] for m in rm.RX_ALL]))[:, None, :] * np.stack([U[m] for m in rm.TX8])[None, :, :] * gr["dA"]
         ).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    ph = np.random.default_rng(77).uniform(0, 2 * np.pi, size=(len(rm.RX_ALL), 2))     # same sorter phases as run_sens2
    _ST = dict(base=base, Qr=np.ascontiguousarray(Q.real), Qi=np.ascontiguousarray(Q.imag), ph=ph)


def work(args):
    i, n = args
    st = _ST; base = st["base"]
    rng = np.random.default_rng(8800_000 + i)
    z = rng.standard_normal((n, base.j_list.size)); I = base.scint(rng, n, S2)
    Phi = base.coeffs(z, S2) @ base.Z
    c, s = np.cos(Phi), np.sin(Phi)
    h = ((c @ st["Qr"] - s @ st["Qi"]) + 1j * (c @ st["Qi"] + s @ st["Qr"])).reshape(n, len(rm.RX_ALL), len(rm.TX8))
    tx, rx = rm.CFG[CFG]; ti = [rm.TX8.index(m) for m in tx]; ri = [rm.RX_ALL.index(m) for m in rx]
    hk = h[:, :, ti]; cx = math.sqrt(10 ** (XT_DB / 10))
    hx = np.empty((n, len(ri), len(ti)), complex)
    for j, p in enumerate(ri):
        acc = hk[:, p].copy()
        if p - 1 >= 0:
            acc += cx * np.exp(1j * st["ph"][p, 0]) * hk[:, p - 1]
        if p + 1 < len(rm.RX_ALL):
            acc += cx * np.exp(1j * st["ph"][p, 1]) * hk[:, p + 1]
        hx[:, j] = acc / math.sqrt(1 + 2 * cx * cx)
    G0 = np.abs(hk[:, ri]) ** 2 * I[:, None, ti]
    Gx = np.abs(hx) ** 2 * I[:, None, ti]
    e = rng.standard_normal(Gx.shape)
    out = {}
    cache = {}
    for name, rin, dl, ber, impaired in CASES:
        P = rm.snr0(len(tx), dl, rin)
        key = (impaired, P)
        if key not in cache:
            if impaired:
                g, _, _ = sinr_of(mmse_filter(Gx + e / math.sqrt(NPIL * P), P), Gx, P)
            else:
                g = of.sinr_mmse(G0, P)
            cache[key] = g
        g = cache[key]
        f = g < (rm.GTH if ber is None else of.gamma_th_from_ber(ber))
        out[name] = (np.bincount(f.sum(1), minlength=7)[:7], f.T.astype(np.int64) @ f.astype(np.int64))
    return out


def main(N, workers):
    t0 = time.time()
    jobs = [(i, min(CHUNK, N - i * CHUNK)) for i in range(math.ceil(N / CHUNK))]
    tot = {}
    with Pool(workers, initializer=init) as pool:
        for j, r in enumerate(pool.imap_unordered(work, jobs)):
            for k, (h, pj) in r.items():
                a = tot.setdefault(k, [np.zeros(7, np.int64), np.zeros((6, 6), np.int64)])
                a[0] += h; a[1] += pj
            if (j + 1) % 400 == 0:
                print(f"{j + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res = {"s2": S2, "config": CFG, "N": N, "pilot_symbols": NPIL, "xt_dB": XT_DB, "cases": {}}
    tx = rm.CFG[CFG][0]
    for name, rin, dl, ber, imp in CASES:
        h, pj = tot[name]
        iu = [(a, b) for a in range(6) for b in range(a + 1, 6)]; best = min(iu, key=lambda ab: pj[ab])
        res["cases"][name] = {"rin_dB_Hz": rin, "extra_loss_dB": dl, "ber_threshold": ber, "impaired_receiver": imp,
                              "outage": float(1 - h[0] / N), "nfail_hist": h.tolist(),
                              "best_anchor_pair": [tx[best[0]], tx[best[1]]], "best_anchor_loss": float(pj[best] / N)}
        print(name, f"{1 - h[0] / N:.3g}", flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "budget_2.5.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 7)
