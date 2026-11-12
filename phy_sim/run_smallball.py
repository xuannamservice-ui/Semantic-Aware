"""Small-ball test of assumption (D2) on the primary simulator (oamfso.py, same channel as run_modeset.py).

For every configuration of run_modeset.CFG, delta_m(C) is the distance of column c_m of the power-coupling
matrix C (|h_rn|^2, no scintillation) to the span of the other columns, delta_m^2 = 1/[(C^T C)^{-1}]_mm.
The script records Pr(delta_m(C) <= eps) on a log grid of eps (pooled over streams and per stream) and fits
the local exponent k_hat on three eps windows. Confidence intervals: G independent worker groups, fit per
group, t-interval of the group fits. Result: results/smallball_<s2>.json
usage: python run_smallball.py --s2 2.5 --N 4000000 --workers 12
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
EPS = np.logspace(-5, 0, 51)
WINDOWS = {"1e-3..1e-2": (1e-3, 1e-2), "1e-2.5..1e-1.5": (10 ** -2.5, 10 ** -1.5), "1e-3..1e-1": (1e-3, 1e-1)}
MIN_COUNT = 30
_CH = None


def work(args):
    seed, s2, n, batch = args
    global _CH
    if _CH is None:
        _CH = of.OAMChannel(modes=rm.TX8, rx_modes=rm.RX_ALL)
    ch = _CH
    rng = np.random.default_rng(seed)
    cnt = {k: np.zeros((EPS.size, len(tx)), np.int64) for k, (tx, rx) in rm.CFG.items()}
    q = {k: [] for k in rm.CFG}
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = rng.standard_normal((b, ch.j_list.size))
        C = ch.power_coupling(ch.coeffs(z, s2)).astype(np.float64)          # (b, 13, 8)
        for k, (tx, rx) in rm.CFG.items():
            ri = [rm.RX_ALL.index(m) for m in rx]
            ti = [rm.TX8.index(m) for m in tx]
            Cs = C[:, ri][:, :, ti]
            inv = np.linalg.inv(np.einsum("brm,brn->bmn", Cs, Cs))
            d = np.sqrt(1.0 / np.clip(np.diagonal(inv, axis1=1, axis2=2), 1e-300, None))   # (b, M)
            cnt[k] += (d[None, :, :] <= EPS[:, None, None]).sum(1)
            if len(q[k]) < 50:
                q[k].append(np.quantile(d, [0.001, 0.01, 0.1, 0.5]))
        done += b
    return cnt, n, {k: np.mean(v, 0).tolist() for k, v in q.items()}


def fit(p, n_events, lo, hi):
    sel = (EPS >= lo) & (EPS <= hi) & (n_events >= MIN_COUNT) & (p > 0)
    if sel.sum() < 4:
        return None
    r = stats.linregress(np.log(EPS[sel]), np.log(p[sel]))
    return float(r.slope)


def main(s2, N, workers, groups, batch, seed):
    t0 = time.time()
    per = N // workers
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(workers)]
    with Pool(workers) as pool:
        res = pool.map(work, [(sd, s2, per, batch) for sd in seeds])
    n_tot = per * workers
    out = {"s2": s2, "N": n_tot, "workers": workers, "groups": groups, "seed": seed, "eps": EPS.tolist(),
           "windows": {k: list(v) for k, v in WINDOWS.items()}, "min_count": MIN_COUNT, "configs": {}}
    gidx = np.array_split(np.arange(workers), groups)
    for k, (tx, rx) in rm.CFG.items():
        M, Nr = len(tx), len(rx)
        c_all = sum(r[0][k] for r in res)                      # (E, M)
        ev = c_all.sum(1)
        p = ev / (M * n_tot)
        e = {"M": M, "Nr": Nr, "k_nom": Nr - M + 1, "p_eps": p.tolist(), "events": ev.tolist(),
             "p_eps_per_mode": (c_all / n_tot).T.tolist(),
             "delta_quantiles_q001_q01_q1_q5": np.mean([r[2][k] for r in res], 0).tolist(), "fits": {}}
        for wn, (lo, hi) in WINDOWS.items():
            kp = fit(p, ev, lo, hi)
            gk = []
            for g in gidx:
                cg = sum(res[i][0][k] for i in g)
                eg = cg.sum(1)
                kg = fit(eg / (M * per * len(g)), eg * groups, lo, hi)   # same bins as the pooled fit
                if kg is not None:
                    gk.append(kg)
            ci = None
            if len(gk) >= 3:
                ci = float(stats.t.ppf(0.975, len(gk) - 1) * np.std(gk, ddof=1) / np.sqrt(len(gk)))
            e["fits"][wn] = {"k_hat": kp, "ci95": ci, "group_fits": gk}
        out["configs"][k] = e
        f = e["fits"]
        print(f"{k:6s} k_nom={Nr - M + 1:2d} " + "  ".join(
            f"{wn}: {f[wn]['k_hat'] if f[wn]['k_hat'] is None else round(f[wn]['k_hat'], 3)}"
            f"+-{f[wn]['ci95'] if f[wn]['ci95'] is None else round(f[wn]['ci95'], 3)}" for wn in WINDOWS), flush=True)
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, f"smallball_{s2}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, default=2.5)
    ap.add_argument("--N", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7701)
    a = ap.parse_args()
    main(a.s2, a.N, a.workers, a.groups, a.batch, a.seed)
