"""Channel variation inside a slot and aging of the channel estimate (sigma_R^2 = 2.5).

Model A keeps the channel constant within a slot. Here the channel evolves inside a slot of T_f = 1 ms along NPT = 11
time points (0, 0.1, ..., 1.0 ms) by the Gauss-Markov update of Model B, applied to the normalized Zernike vector and to
the Gaussian copulas of X and Y_n, with step correlation rho_s = rho(T_f)^(1/10), rho(T_f) = exp(-(T_f/tau_c)^(5/3)), so
that the correlation over the slot equals that of Model B (inside the slot this decays faster than the Kolmogorov law,
which is conservative). A stream then loses only the part of its slot payload sent while its SINR is below the
threshold; the time-averaged number of failed streams K_eff = sum_m (fraction of the points at which stream m fails)
measures the loss of a slot (its mean equals the mean number of failed streams of Model A).
Receivers:
  true      MMSE filter from the true channel at every point
  pilot1us  filter estimated from an orthogonal 64-symbol pilot block every 1 us (0.64% overhead at 10 GBd), applied to
            data on average 0.5 us later (Kolmogorov correlation over 0.5 us), estimation noise variance 1/(64 P)
  stale     filter estimated once at the start of the slot (64 pilot symbols), applied at 0.5 ms and 1.0 ms
Output: results/aging_<tau>ms.json with, per configuration and receiver, the snapshot outage, the share of slots with
any failure, and the histogram of the number of failed (stream, point) pairs (K_eff = count / NPT).
usage: python run_aging.py <tau_ms> <N> <workers>
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
from scipy import special, stats

import oamfso as of
import run_modeset as rm
from run_sens2 import mmse_filter, sinr_of

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = 2.5
CFGS = ("T8R8", "T6R11")
NPT = 11
NPIL = 64
CHUNK = 2_000
_ST = None


def init(tau):
    global _ST
    base = of.OAMChannel(); gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Q = (np.conj(np.stack([U[m] for m in rm.RX_ALL]))[:, None, :] * np.stack([U[m] for m in rm.TX8])[None, :, :] * gr["dA"]
         ).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    rho_slot = math.exp(-(1.0 / tau) ** (5 / 3))
    _ST = dict(base=base, Qr=np.ascontiguousarray(Q.real), Qi=np.ascontiguousarray(Q.imag), tau=tau,
               rs=rho_slot ** (1 / (NPT - 1)), rp=math.exp(-(0.0005 / tau) ** (5 / 3)), ab=base.gg_params(S2))


def chan(z, ux, uy):
    st = _ST; base = st["base"]; al, be = st["ab"]
    X = stats.gamma.ppf(special.ndtr(ux), al, scale=1 / al)
    Y = stats.gamma.ppf(special.ndtr(uy), be, scale=1 / be)
    Phi = base.coeffs(z, S2) @ base.Z
    c, s = np.cos(Phi), np.sin(Phi)
    P = ((c @ st["Qr"] - s @ st["Qi"]) ** 2 + (c @ st["Qi"] + s @ st["Qr"]) ** 2).reshape(z.shape[0], len(rm.RX_ALL), len(rm.TX8))
    return P * (X[:, None] * Y)[:, None, :]


def step(rng, z, ux, uy, r):
    q = math.sqrt(1 - r * r)
    return (r * z + q * rng.standard_normal(z.shape), r * ux + q * rng.standard_normal(ux.shape),
            r * uy + q * rng.standard_normal(uy.shape))


def sub(G, k):
    tx, rx = rm.CFG[k]
    return G[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]


def work(args):
    i, n = args
    st = _ST; base = st["base"]
    rng = np.random.default_rng(int(st["tau"] * 1000) * 100_000 + 7300 + i)
    z = rng.standard_normal((n, base.j_list.size)); ux = rng.standard_normal(n); uy = rng.standard_normal((n, len(rm.TX8)))
    gth = rm.GTH
    cnt = {k: {r: np.zeros(n, np.int64) for r in ("true", "pilot1us")} for k in CFGS}
    snap = {k: {} for k in CFGS}; stale = {k: {} for k in CFGS}; W0 = {}
    for t in range(NPT):
        if t:
            z, ux, uy = step(rng, z, ux, uy, st["rs"])
        G = chan(z, ux, uy)
        Gp = chan(*step(rng, z, ux, uy, st["rp"]))
        for k in CFGS:
            Gk = sub(G, k); P = rm.snr0(len(rm.CFG[k][0]))
            ft = of.sinr_mmse(Gk, P) < gth
            Gpk = sub(Gp, k)
            fp = sinr_of(mmse_filter(Gpk + rng.standard_normal(Gpk.shape) / math.sqrt(NPIL * P), P), Gk, P)[0] < gth
            cnt[k]["true"] += ft.sum(1); cnt[k]["pilot1us"] += fp.sum(1)
            if t == 0:
                snap[k] = {"true": int(ft.any(1).sum()), "pilot1us": int(fp.any(1).sum())}
                W0[k] = mmse_filter(Gk + rng.standard_normal(Gk.shape) / math.sqrt(NPIL * P), P)
            if t in (5, 10):
                stale[k]["0.5ms" if t == 5 else "1.0ms"] = int((sinr_of(W0[k], Gk, P)[0] < gth).any(1).sum())
    out = {}
    for k in CFGS:
        M = len(rm.CFG[k][0])
        out[k] = {"snap_" + r: v for r, v in snap[k].items()}
        out[k].update({"stale_" + r: v for r, v in stale[k].items()})
        for r, c in cnt[k].items():
            out[k]["hist_" + r] = np.bincount(c, minlength=M * NPT + 1)[:M * NPT + 1]
    return out


def main(tau, N, workers):
    t0 = time.time()
    jobs = [(i, min(CHUNK, N - i * CHUNK)) for i in range(math.ceil(N / CHUNK))]
    tot = {}
    with Pool(workers, initializer=init, initargs=(tau,)) as pool:
        for j, r in enumerate(pool.imap_unordered(work, jobs)):
            for k, d in r.items():
                for v, x in d.items():
                    tot.setdefault(k, {})[v] = tot.get(k, {}).get(v, 0) + x
            if (j + 1) % 100 == 0:
                print(f"tau={tau} {j + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    res = {"s2": S2, "tau_c_ms": tau, "N": N, "points_per_slot": NPT, "pilot_symbols": NPIL,
           "rho_slot": math.exp(-(1.0 / tau) ** (5 / 3)), "rho_step": math.exp(-(1.0 / tau) ** (5 / 3)) ** (1 / (NPT - 1)), "configs": {}}
    for k, d in tot.items():
        row = {}
        for v, x in d.items():
            if v.startswith("hist_"):
                h = np.asarray(x); row[v] = h.tolist()
                r = v[5:]
                row["any_" + r] = float(1 - h[0] / N)
                row["mean_Keff_" + r] = float((np.arange(h.size) * h).sum() / NPT / N)
            else:
                row[v] = x / N
        res["configs"][k] = row
        print(k, {v: (round(x, 6) if isinstance(x, float) else None) for v, x in row.items() if not v.startswith("hist")}, flush=True)
    json.dump(res, open(os.path.join(HERE, "results", f"aging_{tau:g}ms.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 7)
