"""Empirical check of assumption (D3) of Proposition 1 and of the local small-ball exponent of (D2).

With G = X C diag(Y), lambda'_m = lambda_min(D_{-m} C_{-m}^T C_{-m} D_{-m}) (D_{-m} = diag(Y_n), n != m) and delta_m(C)
the distance of column c_m of C from the span of the other columns, (D3) asks for a0, eps1 > 0 with
    Pr(lambda'_m >= a0 ||c_m||^2 | delta_m(C) <= eps) >= 1/2   for all eps <= eps1,
i.e. the conditional median of R_m = lambda'_m / ||c_m||^2 given delta_m(C) <= eps must stay bounded away from zero as
eps -> 0. The script draws the channel of oamfso.py (sigma_R^2 = 2.5, J = 435, aperture-averaged Gamma-Gamma Y_n; X
cancels in R_m) for the configurations T8R8, T6R11, T6R8, T8R11, T4R8 on common random numbers and, per stream m,
reports at eps = the empirical quantiles 1e-1 ... 1e-5 of delta_m(C) the conditional median of R_m with a bootstrap
95% interval and the number of conditioning samples. Samples with delta_m below a per-worker pilot threshold (2%
quantile of the first --nfull draws) are stored; the first --nfull draws of every worker are stored completely (used
for the 1e-1 level). The local small-ball exponent is the least-squares slope of log Pr(delta_m <= eps) against log eps
over the lowest two decades resolved by the sample (from the 30th smallest delta upwards), with a bootstrap interval.
usage: python check_d3.py --N 40000000 --workers 8 --out results/d3_check.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

CFGS = {"T8R8": ((-4, -3, -2, -1, 1, 2, 3, 4), (-4, -3, -2, -1, 1, 2, 3, 4)),
        "T6R11": ((-3, -2, -1, 1, 2, 3), tuple(range(-5, 6))),
        "T6R8": ((-3, -2, -1, 1, 2, 3), (-4, -3, -2, -1, 1, 2, 3, 4)),
        "T8R11": ((-4, -3, -2, -1, 1, 2, 3, 4), tuple(range(-5, 6))),
        "T4R8": ((-2, -1, 1, 2), (-4, -3, -2, -1, 1, 2, 3, 4))}
QS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
MIN_COND = 20


def lam_prime(CtC, Y, m):
    """lambda_min of D_{-m} C_{-m}^T C_{-m} D_{-m} for a batch."""
    keep = [n for n in range(CtC.shape[-1]) if n != m]
    A = CtC[:, keep][:, :, keep] * Y[:, keep, None] * Y[:, None, keep]
    return np.linalg.eigvalsh(A)[:, 0]


def worker(args):
    seed, n, batch, nfull, s2 = args
    import oamfso as of
    import run_modeset as rm
    ch = of.OAMChannel(modes=rm.TX8, rx_modes=rm.RX_ALL)
    _, be = ch.gg_params(s2)
    rng = np.random.default_rng(seed)
    idx = {k: ([rm.RX_ALL.index(m) for m in rx], [rm.TX8.index(m) for m in tx]) for k, (tx, rx) in CFGS.items()}
    full = {k: {"d": [], "r": []} for k in CFGS}
    small = {k: {"d": [], "r": [], "m": []} for k in CFGS}
    thr = {}
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = rng.standard_normal((b, ch.j_list.size))
        C = ch.power_coupling(ch.coeffs(z, s2)).astype(np.float64)
        Yall = rng.gamma(be, 1 / be, size=(b, 8))
        for k, (ri, ti) in idx.items():
            Cs = C[:, ri][:, :, ti]
            Y = Yall[:, ti]
            CtC = np.einsum("brm,brn->bmn", Cs, Cs)
            reg = CtC + 1e-14 * np.trace(CtC, axis1=1, axis2=2)[:, None, None] * np.eye(CtC.shape[1])
            inv = np.linalg.inv(reg)
            d = np.sqrt(1.0 / np.clip(np.diagonal(inv, axis1=1, axis2=2), 1e-300, None))     # (b, M)
            cn = np.diagonal(CtC, axis1=1, axis2=2)                                              # ||c_m||^2
            M = d.shape[1]
            if done < nfull:
                R = np.stack([lam_prime(CtC, Y, m) for m in range(M)], 1) / cn
                full[k]["d"].append(d.astype(np.float32)); full[k]["r"].append(R.astype(np.float32))
                if done + b >= nfull:
                    thr[k] = np.quantile(np.concatenate(full[k]["d"]), 0.02, axis=0)
            else:
                for m in range(M):
                    s = np.flatnonzero(d[:, m] <= thr[k][m])
                    if s.size:
                        small[k]["d"].append(d[s, m]); small[k]["m"].append(np.full(s.size, m, np.int8))
                        small[k]["r"].append(lam_prime(CtC[s], Y[s], m) / cn[s, m])
        done += b
    out = {}
    for k in CFGS:
        fd = np.concatenate(full[k]["d"])
        out[k] = dict(full_d=fd, full_r=np.concatenate(full[k]["r"]),
                      small_d=np.concatenate(small[k]["d"]) if small[k]["d"] else np.zeros(0),
                      small_r=np.concatenate(small[k]["r"]) if small[k]["r"] else np.zeros(0),
                      small_m=np.concatenate(small[k]["m"]) if small[k]["m"] else np.zeros(0, np.int8),
                      thr=thr[k], n_full=int(fd.shape[0]), n_rest=int(n - fd.shape[0]))
    return out


def boot_median(x, rng, B=1000):
    i = rng.integers(0, x.size, (B, x.size))
    m = np.median(x[i], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def slope_fit(dsmall, N, rng, B=500):
    """LS slope of log F(eps) vs log eps over [eps_lo, 100 eps_lo], eps_lo = 30th smallest delta."""
    ds = np.sort(dsmall)
    if ds.size < 300:
        return None
    lo = ds[29]; hi = 100 * lo
    if hi > ds[-1]:
        hi = ds[-1]
    grid = np.logspace(np.log10(lo), np.log10(hi), 9)

    def fit(x):
        F = np.searchsorted(np.sort(x), grid, side="right") / N
        ok = F > 0
        return np.polyfit(np.log(grid[ok]), np.log(F[ok]), 1)[0]
    s0 = fit(ds)
    bs = []
    for _ in range(B):
        nb = rng.poisson(ds.size)
        bs.append(fit(ds[rng.integers(0, ds.size, nb)]))
    return dict(slope=float(s0), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                eps_window=[float(lo), float(hi)], decades=float(np.log10(hi / lo)), count_at_lo=30, count_at_hi=int(np.searchsorted(ds, hi, side="right")))


def analyse(parts, rng):
    res = {}
    for k, (tx, rx) in CFGS.items():
        M = len(tx); kdim = len(rx) - M + 1
        n_full = sum(p[k]["n_full"] for p in parts); n_rest = sum(p[k]["n_rest"] for p in parts)
        fd = np.concatenate([p[k]["full_d"] for p in parts]); fr = np.concatenate([p[k]["full_r"] for p in parts])
        streams = []
        for m in range(M):
            # tail sample: everything below the smallest worker threshold, from all draws (full part + rest)
            tmin = min(p[k]["thr"][m] for p in parts)
            sd = np.concatenate([np.concatenate([p[k]["small_d"][p[k]["small_m"] == m] for p in parts]), fd[:, m]])
            sr = np.concatenate([np.concatenate([p[k]["small_r"][p[k]["small_m"] == m] for p in parts]), fr[:, m]])
            keep = sd <= tmin
            sd, sr = sd[keep], sr[keep]
            Ntot = n_full + n_rest
            row = dict(mode=tx[m], N=Ntot, uncond_median_R=float(np.median(fr[:, m])), levels=[])
            for q in QS:
                if q * Ntot < MIN_COND:
                    continue
                if q >= 0.02:     # from the completely stored part
                    eps = float(np.quantile(fd[:, m], q)); x = fr[fd[:, m] <= eps, m]
                else:
                    o = np.argsort(sd); nq = int(round(q * Ntot))
                    if nq > sd.size:
                        continue
                    eps = float(sd[o[nq - 1]]); x = sr[o[:nq]]
                row["levels"].append(dict(q=q, eps=eps, n_cond=int(x.size), median_R=float(np.median(x)), median_ci95=boot_median(x, rng),
                                          q10_R=float(np.quantile(x, 0.1)), min_R=float(x.min())))
            if kdim >= 1:
                row["small_ball"] = slope_fit(sd, Ntot, rng)
            streams.append(row)
        res[k] = dict(tx=tx, rx=rx, k=kdim, N=n_full + n_rest, streams=streams)
        for s in streams:
            lv = " | ".join(f"q={l['q']:.0e} eps={l['eps']:.2e} n={l['n_cond']} med={l['median_R']:.3g} [{l['median_ci95'][0]:.3g},{l['median_ci95'][1]:.3g}]" for l in s["levels"])
            sb = s.get("small_ball"); sbs = f" slope={sb['slope']:.2f} [{sb['ci95'][0]:.2f},{sb['ci95'][1]:.2f}] over {sb['decades']:.1f} dec" if sb else ""
            print(f"{k} k={kdim} m={s['mode']:+d}: uncond {s['uncond_median_R']:.3g} | {lv}{sbs}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--nfull", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=717001)
    ap.add_argument("--s2", type=float, default=2.5)
    ap.add_argument("--out", default="results/d3_check.json")
    a = ap.parse_args()
    t0 = time.time()
    per = a.N // a.workers
    with Pool(a.workers) as p:
        parts = p.map(worker, [(a.seed + 97 * i, per, a.batch, a.nfull, a.s2) for i in range(a.workers)])
    res = analyse(parts, np.random.default_rng(3))
    json.dump(dict(s2=a.s2, J=435, N=per * a.workers, workers=a.workers, seed=a.seed, nfull_per_worker=a.nfull,
                   runtime_s=time.time() - t0, configs=res), open(a.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
