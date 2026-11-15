"""Receiver-side sensitivity of the outage (second set; common random numbers within each turbulence level).

Variants, one change at a time against the main model (separate carriers, linear MMSE, Gaussian treatment of the
residual interference, hard-decision threshold BER 3.8e-3):
  xt_<x>dB   intrinsic sorter crosstalk: port l also collects the field projected on the adjacent orders l-1 and
             l+1 with power coupling x dB and fixed random phases, g'_l = (h_l + c e^{j t1} h_{l-1} + c e^{j t2} h_{l+1})
             / sqrt(1 + 2c^2) (fields of one transmitted mode add coherently; separate carriers keep the
             cross-mode beat terms out of band).
  csi_<d>    imperfect channel knowledge: the MMSE filter is computed from G_hat = G (1 + d e), e ~ N(0, 1) per entry,
             and applied to the true G; SINR of the mismatched filter.
  pilot_<n>  pilot-based channel estimate: each entry of G is estimated from n OOK pilot symbols per stream, additive
             error N(0, 1/(n P)) in units of the receiver noise (the filter is again applied to the true G).
  commonY    small-scale scintillation factor common to all modes (fully correlated irradiance).
  exactBER   residual interference after MMSE kept binary: BER_m = E_b Q((a_mm + sum_n a_mn b_n) sqrt(P) / |w_m|),
             outage if any stream has BER > 3.8e-3 (evaluated exactly for slots with min SINR below gth + 10 dB).
  ber_<x>    other decision thresholds: 1.8e-4 and 1e-3 bracket the slot-failure waterfall of RS(255,239) with 4,583
             codewords per stream and slot (1e-6 and 0.5 slot-failure points), 2e-2 a soft-decision FEC threshold.
Output: results/sens2_<s2>.json
usage: python run_sens2.py 2.5 2000000 8
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import itertools
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import erfc

import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
CFGS = ("T8R8", "T6R11")
XT_DB = (-25.0, -20.0, -15.0)
CSI = (0.01, 0.05, 0.10)
PILOTS = (4, 64)
BER_ALT = (1.8e-4, 1e-3, 2e-2)
SEED = {1.0: 4101, 2.5: 4102}
CHUNK = 2_500
_ST = None


def qf(x):
    return 0.5 * erfc(x / math.sqrt(2.0))


def init(s2):
    global _ST
    base = of.OAMChannel()
    gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Ur = np.stack([U[m] for m in rm.RX_ALL]); Ut = np.stack([U[m] for m in rm.TX8])
    Q = (np.conj(Ur)[:, None, :] * Ut[None, :, :] * gr["dA"]).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    ph = np.random.default_rng(77).uniform(0, 2 * np.pi, size=(len(rm.RX_ALL), 2))   # fixed sorter phases
    _ST = dict(base=base, Qr=np.ascontiguousarray(Q.real), Qi=np.ascontiguousarray(Q.imag), ph=ph, s2=s2)


def mmse_filter(Gh, P):
    """W = (Gh^T Gh + I/P)^{-1} Gh^T, shape (B, M, N_r)."""
    M = Gh.shape[-1]
    GT = np.swapaxes(Gh, -1, -2)
    return np.linalg.solve(GT @ Gh + np.eye(M) / P, GT)


def sinr_of(W, G, P):
    A = W @ G
    d = np.diagonal(A, axis1=-2, axis2=-1)
    intf = (A ** 2).sum(-1) - d ** 2
    nz = (W ** 2).sum(-1)
    return P * d ** 2 / (P * intf + nz), A, nz


def exact_fail(A, nz, P, ber):
    """Any stream with exact BER (binary interferers, zero threshold after removing the mean) above ber."""
    B, M, _ = A.shape
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=M - 1)))           # (2^{M-1}, M-1)
    out = np.zeros(B, bool)
    for m in range(M):
        others = [n for n in range(M) if n != m]
        amp = A[:, m, m][:, None] + A[:, m, others] @ signs.T                       # (B, 2^{M-1})
        be = qf(amp * math.sqrt(P) / np.sqrt(nz[:, m])[:, None]).mean(1)
        out |= be > ber
    return out


def work(args):
    s2, i, n = args
    st = _ST; base = st["base"]
    rng = np.random.default_rng(SEED[s2] * 1000 + i)
    z = rng.standard_normal((n, base.j_list.size))
    al, be = base.gg_params(s2)
    X = rng.gamma(al, 1 / al, size=n)
    Y = rng.gamma(be, 1 / be, size=(n, len(rm.TX8)))
    Yc = rng.gamma(be, 1 / be, size=n)
    e = rng.standard_normal((n, len(rm.RX_ALL), len(rm.TX8)))
    Phi = base.coeffs(z, s2) @ base.Z
    c, s = np.cos(Phi), np.sin(Phi)
    h = ((c @ st["Qr"] - s @ st["Qi"]) + 1j * (c @ st["Qi"] + s @ st["Qr"])).reshape(n, len(rm.RX_ALL), len(rm.TX8))
    I = X[:, None] * Y; Ic = (X * Yc)[:, None] * np.ones((1, len(rm.TX8)))
    gth = rm.GTH; gth_alt = {b: of.gamma_th_from_ber(b) for b in BER_ALT}
    res = {}
    for k in CFGS:
        tx, rx = rm.CFG[k]
        ti = [rm.TX8.index(m) for m in tx]; ri = [rm.RX_ALL.index(m) for m in rx]
        P = rm.snr0(len(tx))
        hk = h[:, :, ti]
        G = np.abs(hk[:, ri]) ** 2 * I[:, None, ti]
        r = {}
        W = mmse_filter(G, P); g0, A, nz = sinr_of(W, G, P)
        r["baseline"] = (g0 < gth).any(1)
        for b, t in gth_alt.items():
            r[f"ber_{b:g}"] = (g0 < t).any(1)
        cand = (g0.min(1) < gth * 10.0)
        ex = r["baseline"].copy()
        if cand.any():
            ex[cand] = exact_fail(A[cand], nz[cand], P, of.BER_FEC)
            ex[~cand] = False
        r["exactBER"] = ex
        for x in XT_DB:
            cx = math.sqrt(10 ** (x / 10))
            hx = np.empty((n, len(ri), len(ti)), complex)
            for j, p in enumerate(ri):
                acc = hk[:, p].copy()
                if p - 1 >= 0:
                    acc += cx * np.exp(1j * st["ph"][p, 0]) * hk[:, p - 1]
                if p + 1 < len(rm.RX_ALL):
                    acc += cx * np.exp(1j * st["ph"][p, 1]) * hk[:, p + 1]
                hx[:, j] = acc / math.sqrt(1 + 2 * cx * cx)
            Gx = np.abs(hx) ** 2 * I[:, None, ti]
            r[f"xt_{x:g}dB"] = (of.sinr_mmse(Gx, P) < gth).any(1)
        for d in CSI:
            Gh = np.clip(G * (1 + d * e[:, ri][:, :, ti]), 0, None)
            gm, _, _ = sinr_of(mmse_filter(Gh, P), G, P)
            r[f"csi_{d:g}"] = (gm < gth).any(1)
        for npil in PILOTS:
            Gh = G + e[:, ri][:, :, ti] / math.sqrt(npil * P)
            gm, _, _ = sinr_of(mmse_filter(Gh, P), G, P)
            r[f"pilot_{npil}"] = (gm < gth).any(1)
        Gc = np.abs(hk[:, ri]) ** 2 * Ic[:, None, ti]
        r["commonY"] = (of.sinr_mmse(Gc, P) < gth).any(1)
        res[k] = {v: (int(f.sum()), int((f & ~r["baseline"]).sum()), int((~f & r["baseline"]).sum())) for v, f in r.items()}
    return res


def main(s2, N, workers):
    t0 = time.time()
    jobs = [(s2, i, min(CHUNK, N - i * CHUNK)) for i in range(math.ceil(N / CHUNK))]
    tot = {}
    with Pool(workers, initializer=init, initargs=(s2,)) as pool:
        for j, r in enumerate(pool.imap_unordered(work, jobs)):
            for k, d in r.items():
                for v, t in d.items():
                    a = tot.setdefault(k, {}).setdefault(v, [0, 0, 0])
                    for q in range(3):
                        a[q] += t[q]
            print(f"s2={s2} chunk {j + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    out = {"s2": s2, "N": N, "seed": SEED[s2], "gth_dB": 10 * math.log10(rm.GTH), "configs": {}}
    for k, d in tot.items():
        b = d["baseline"][0]
        out["configs"][k] = {}
        for v, (cnt, up, down) in d.items():
            # paired difference against the baseline on common random numbers
            diff = (cnt - b) / N; var = (up + down) / N - diff ** 2
            out["configs"][k][v] = {"outage": cnt / N, "count": cnt, "diff": diff,
                                    "diff_ci95": 1.96 * math.sqrt(max(var, 0) / N), "wilson95": wilson(cnt, N)}
            print(k, v, f"{cnt / N:.4g} ({diff:+.3g})", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"sens2_{s2}.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), c + h]


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 8)
